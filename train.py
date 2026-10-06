"""LeWM、Value-JEPA 与 Fast-LeWAM 共用的 Hydra/Lightning 训练入口。"""

import os
from datetime import timedelta
from pathlib import Path
import subprocess
import hashlib
import json
import fcntl

import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as pl
import stable_pretraining as spt
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from source.common.checkpoint import (
    CHECKPOINTS_DIRNAME,
    SaveCkptCallback,
    load_initial_model_state,
    resolve_training_resume_checkpoint,
)
from source.common.data import get_column_normalizer, get_img_preprocessor, load_dataset
from source.common.logging import get_run_dir, tee_output_to_file
from source.common.sampling import DistributedChunkLocalSampler


def _validate_round4_gpu_visibility(cfg):
    """Require the Round 4 single-card training contract before CUDA starts."""
    expansion = cfg.get("round5_phase5_expansion", {})
    is_phase5_expansion = bool(expansion.get("enabled", False))
    phase6_1 = cfg.get("round5_phase6_1", {})
    is_phase6_1 = bool(phase6_1.get("enabled", False))
    table2 = cfg.get("cvpr_table2", {})
    is_cvpr_table2 = bool(table2.get("enabled", False))
    if (
        str(cfg.get("train_mode", "")) != "stage_abde"
        and not is_phase5_expansion
        and not is_phase6_1
        and not is_cvpr_table2
    ):
        return
    accelerator = str(cfg.get("trainer", {}).get("accelerator", "auto"))
    if accelerator not in {"gpu", "cuda"}:
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        raise RuntimeError(
            "This GPU training run requires CUDA_VISIBLE_DEVICES set to one of GPU0-3"
        )
    values = [item.strip() for item in visible.split(",") if item.strip()]
    allowed = range(8) if is_phase6_1 else range(4)
    if len(values) != 1 or not values[0].isdigit() or int(values[0]) not in allowed:
        raise RuntimeError(
            "This training run requires exactly one explicitly selected permitted physical GPU; "
            f"got {visible!r}"
        )


def _phase6_1_training_manifest(cfg, policy, run_dir):
    """Archive the new FM parameters and run identity before the first update."""
    phase = cfg.get("round5_phase6_1", {})
    if not bool(phase.get("enabled", False)):
        return None
    model = getattr(policy, "model", policy)
    if str(cfg.get("train_mode", "")) != "stage_ab_fm" or not bool(
        getattr(model, "latent_flow_matching", False)
    ):
        raise ValueError("Phase 6.1 requires train_mode=stage_ab_fm and an FM model")
    if cfg.get("init_weights", None) or cfg.get("resume_ckpt", None):
        raise ValueError("Phase 6.1 training must start from scratch")
    if int(cfg.get("initial_epoch", 0)) != 0:
        raise ValueError("Phase 6.1 training must start at epoch zero")
    effective_batch = int(phase.get("effective_batch_size", 128))
    configured_batch = int(cfg.loader.batch_size) * int(
        cfg.trainer.accumulate_grad_batches
    )
    if configured_batch != effective_batch:
        raise ValueError(
            "Phase 6.1 micro-batch × accumulation must equal its effective batch; "
            f"got {configured_batch} != {effective_batch}"
        )

    new_prefixes = (
        "latent_flow_input.",
        "latent_flow_time_mlp.",
        "latent_flow_velocity_head.",
    )
    flow_state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
        if name.startswith(new_prefixes)
    }
    if not flow_state:
        raise RuntimeError("Phase 6.1 model has no separately initialized FM parameters")
    flow_path = run_dir / "phase6_1_initial_latent_flow_parameters.pt"
    torch.save(flow_state, flow_path)
    flow_digest = hashlib.sha256()
    for name, value in sorted(flow_state.items()):
        flow_digest.update(name.encode("utf-8"))
        flow_digest.update(str(tuple(value.shape)).encode("ascii"))
        flow_digest.update(value.numpy().tobytes())
    config_path = run_dir / "config.yaml"
    manifest = {
        "schema_version": 1,
        "task": str(HydraConfig.get().runtime.choices["data"]),
        "seed": int(cfg.seed),
        "epochs": int(cfg.trainer.max_epochs),
        "effective_batch_size": effective_batch,
        "micro_batch_size": int(cfg.loader.batch_size),
        "gradient_accumulation": int(cfg.trainer.accumulate_grad_batches),
        "train_mode": str(cfg.train_mode),
        "checkpoint_epoch": int(cfg.trainer.max_epochs),
        "dataset_name": str(cfg.data.dataset.name),
        "dataset_sha256": phase.get("dataset_sha256", None),
        "latent_flow_steps": int(model.latent_flow_steps),
        "latent_flow_parameter_count": int(
            sum(value.numel() for value in flow_state.values())
        ),
        "initial_latent_flow_parameters": str(flow_path.resolve()),
        "initial_latent_flow_parameters_sha256": flow_digest.hexdigest(),
        "initial_latent_flow_parameters_file_sha256": hashlib.sha256(
            flow_path.read_bytes()
        ).hexdigest(),
        "resolved_training_config": str(config_path.resolve()),
        "resolved_training_config_sha256": (
            hashlib.sha256(config_path.read_bytes()).hexdigest()
            if config_path.is_file()
            else None
        ),
        "source_commit": _round4_git_commit(),
    }
    identity_path = run_dir / "phase6_1_training_identity.json"
    identity_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote Phase 6.1 training identity: {identity_path}")
    return identity_path


class _Phase6_1EpochMetricsCallback(pl.Callback):
    """Persist aggregated A/FM training losses and integrated endpoint error."""

    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self._written_epochs = set()

    def _write_epoch(self, trainer):
        epoch = int(trainer.current_epoch) + 1
        if epoch in self._written_epochs:
            return
        values = {}
        for name, value in trainer.callback_metrics.items():
            if not str(name).startswith(("fit/", "validate/")):
                continue
            try:
                values[str(name)] = float(value.detach().cpu())
            except (AttributeError, TypeError, ValueError):
                continue
        record = {
            "epoch": epoch,
            "global_step": int(trainer.global_step),
            "metrics": values,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
        self._written_epochs.add(epoch)

    def on_validation_epoch_end(self, trainer, pl_module):
        del pl_module
        if not trainer.sanity_checking:
            self._write_epoch(trainer)

    def on_train_end(self, trainer, pl_module):
        del pl_module
        if not trainer.sanity_checking:
            self._write_epoch(trainer)


def _round4_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _initialize_from_weights(policy, checkpoint_path):
    """Initialize a policy from a model-only snapshot.

    This is intentionally separate from Lightning's ``ckpt_path`` resume:
    optimizer and loop state are unavailable in epoch weight snapshots.
    """
    state_dict = load_initial_model_state(checkpoint_path)
    model = getattr(policy, "model", policy)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Initial weight snapshot did not load strictly: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def _data_pipeline_options(cfg):
    """读取可选高速数据管线配置；旧 LeWM 配置缺省时保持原 CPU 预处理行为。"""
    pipeline = cfg.get("data_pipeline", {})
    return (
        bool(pipeline.get("gpu_image_preprocessing", False)),
        pipeline.get("hdf5_chunk_size", None),
    )


def _make_dataloaders(cfg, train_set, val_set, rnd_gen, chunk_size):
    """创建训练/验证 DataLoader，并在启用时安装 DDP chunk-local sampler。"""
    train_sampler = None
    val_sampler = None
    if chunk_size is not None:
        train_sampler = DistributedChunkLocalSampler(
            train_set,
            chunk_size=int(chunk_size),
            shuffle=True,
            seed=cfg.seed,
        )
        val_sampler = DistributedChunkLocalSampler(
            val_set,
            chunk_size=int(chunk_size),
            shuffle=False,
            seed=cfg.seed,
        )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(
        val_set,
        **cfg.loader,
        sampler=val_sampler,
        shuffle=False,
        drop_last=False,
    )
    return train, val


def _phase17_episode_partition(cfg, dataset):
    """Select training and validation windows by raw episode ownership."""
    manifest_path = cfg.get("episode_split_manifest", None)
    if not manifest_path:
        return None
    manifest_path = Path(str(manifest_path)).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Phase1.7 episode split not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") is None:
        raise ValueError("Phase1.7 episode split must include its task")
    required = ("train_episode_ids", "dev_episode_ids", "confirmation_episode_ids")
    if any(not isinstance(manifest.get(name), list) for name in required):
        raise ValueError("Phase1.7 split must include train/dev/confirmation episode IDs")
    train_ids, dev_ids, confirmation_ids = (
        np.asarray(manifest[name]) for name in required
    )
    as_keys = lambda values: {json.dumps(value.item() if isinstance(value, np.generic) else value, sort_keys=True) for value in values}
    train_keys, dev_keys, confirmation_keys = map(as_keys, (train_ids, dev_ids, confirmation_ids))
    if not train_keys or not dev_keys or not confirmation_keys:
        raise ValueError("Phase1.7 train/dev/confirmation splits must be non-empty")
    if train_keys & dev_keys or train_keys & confirmation_keys or dev_keys & confirmation_keys:
        raise ValueError("Phase1.7 train/dev/confirmation episode IDs overlap")

    # HDF5Dataset keeps only train columns in column_names, but its reader
    # still exposes metadata columns through get_col_data().
    episode_column = None
    for candidate in ("episode_idx", "ep_idx"):
        try:
            dataset.get_col_data(candidate)
        except (KeyError, ValueError):
            continue
        episode_column = candidate
        break
    if episode_column is None:
        raise ValueError("Phase1.7 training dataset has no episode_idx/ep_idx column")
    raw_episode_ids = np.asarray(dataset.get_col_data(episode_column))
    if raw_episode_ids.ndim != 1:
        raise ValueError("Phase1.7 episode IDs must be one-dimensional")
    episode_ids_by_clip = raw_episode_ids[np.asarray(dataset.offsets, dtype=np.int64)]
    clip_episode_index = np.fromiter(
        (int(item[0]) for item in dataset.clip_indices),
        dtype=np.int64,
        count=len(dataset),
    )
    sample_episode_ids = episode_ids_by_clip[clip_episode_index]
    train_indices = np.flatnonzero(np.isin(sample_episode_ids, train_ids))
    dev_indices = np.flatnonzero(np.isin(sample_episode_ids, dev_ids))
    confirmation_indices = np.flatnonzero(np.isin(sample_episode_ids, confirmation_ids))
    if not len(train_indices) or not len(dev_indices) or not len(confirmation_indices):
        raise ValueError(
            "Phase1.7 split maps to an empty window partition: "
            f"train={len(train_indices)}, dev={len(dev_indices)}, "
            f"confirmation={len(confirmation_indices)}"
        )
    train_row_indices = np.flatnonzero(np.isin(raw_episode_ids, train_ids))
    if not len(train_row_indices):
        raise ValueError("Phase1.7 training episodes map to no raw dataset rows")
    from torch.utils.data import Subset

    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "train_set": Subset(dataset, train_indices.tolist()),
        "val_set": Subset(dataset, dev_indices.tolist()),
        "train_row_indices": train_row_indices,
        "train_window_count": int(len(train_indices)),
        "dev_window_count": int(len(dev_indices)),
        "confirmation_window_count": int(len(confirmation_indices)),
        "episode_column": episode_column,
    }


def _state_dict_sha256(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().reshape(-1)
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _round5_phase5_training_manifest(
    cfg, dataset, train_set, val_set, normalizer_stats, run_dir
):
    """Persist the shared-window sample assignment and this run's identity."""
    experiment = cfg.round5_phase5_expansion
    task = str(experiment.task)
    clip_indices = np.asarray(dataset.clip_indices, dtype=np.int32)
    if not hasattr(train_set, "indices") or not hasattr(val_set, "indices"):
        raise TypeError("Phase5 requires index-based train and validation splits")
    train_indices = np.asarray(train_set.indices, dtype=np.int32)
    val_indices = np.asarray(val_set.indices, dtype=np.int32)
    if clip_indices.ndim != 2 or clip_indices.shape[1] != 2:
        raise ValueError("Phase5 clip indices must contain episode and start rows")
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError("Phase5 training and validation windows must be non-empty")

    manifest_root = Path(str(experiment.manifest_root)).expanduser()
    if not manifest_root.is_absolute():
        manifest_root = Path(__file__).resolve().parent / manifest_root
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / f"{task}_windows.npz"
    lock_path = manifest_root / f"{task}_windows.lock"
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if manifest_path.exists():
            with np.load(manifest_path) as stored:
                matches = (
                    np.array_equal(stored["clip_indices"], clip_indices)
                    and np.array_equal(stored["train_indices"], train_indices)
                    and np.array_equal(stored["val_indices"], val_indices)
                    and str(stored["dataset_sha256"].item())
                    == str(experiment.dataset_sha256)
                    and int(stored["effective_window_steps"].item())
                    == int(experiment.effective_window_steps)
                    and int(stored["clip_storage_steps"].item())
                    == int(experiment.clip_storage_steps)
                )
            if not matches:
                raise ValueError(
                    f"Shared Phase5 sample manifest does not match this run: {manifest_path}"
                )
        else:
            temporary_path = manifest_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary_path,
                clip_indices=clip_indices,
                train_indices=train_indices,
                val_indices=val_indices,
                dataset_sha256=np.asarray(str(experiment.dataset_sha256)),
                effective_window_steps=np.asarray(
                    int(experiment.effective_window_steps), dtype=np.int32
                ),
                clip_storage_steps=np.asarray(
                    int(experiment.clip_storage_steps), dtype=np.int32
                ),
            )
            temporary_path.replace(manifest_path)
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)

    repo_root = Path(__file__).resolve().parent
    code_paths = (
        Path(__file__).resolve(),
        repo_root / "source/policy/fast_lewam.py",
        repo_root / "source/model/fast_lewam/jepa.py",
        repo_root / "source/model/fast_lewam/round4.py",
        repo_root / "config/train/round5_phase5_expansion.yaml",
        repo_root / "config/train/policy/round4_ab.yaml",
        repo_root / f"config/train/data/{task}.yaml",
        repo_root / "config/round5/phase5_expansion.json",
        repo_root / "scripts/round5_phase5_expansion_train.py",
    )
    code_hashes = {
        str(path.relative_to(repo_root)): _sha256_file(path) for path in code_paths
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    micro_batch = int(cfg.loader.batch_size)
    accumulation = int(cfg.trainer.accumulate_grad_batches)
    effective_batch = int(experiment.effective_batch_size)
    updates_per_epoch = len(train_indices) // effective_batch
    if micro_batch * accumulation != effective_batch:
        raise ValueError(
            "Phase5 micro-batch × accumulation must equal effective batch size"
        )
    if len(train_indices) != updates_per_epoch * effective_batch:
        raise ValueError("Phase5 train split must end on an effective-batch boundary")

    dataset_path = getattr(dataset, "h5_path", None)
    identity = {
        "schema_version": 1,
        "task": task,
        "structure": str(experiment.structure),
        "seed": int(cfg.seed),
        "action_block_steps": int(experiment.action_block_steps),
        "action_horizon": int(cfg.action_horizon),
        "stage_a_goal_index": int(cfg.policy.stage_a_goal_index),
        "effective_window_steps": int(experiment.effective_window_steps),
        "clip_storage_steps": int(experiment.clip_storage_steps),
        "dataset": {
            "path": str(dataset_path),
            "sha256": str(experiment.dataset_sha256),
            "frameskip": int(dataset.frameskip),
            "num_steps": int(dataset.num_steps),
            "clip_count": int(len(clip_indices)),
        },
        "split": {
            "method": "torch.utils.data.random_split",
            "seed": int(cfg.seed),
            "train_fraction": float(cfg.train_split),
            "train_windows": int(len(train_indices)),
            "validation_windows": int(len(val_indices)),
            "effective_batch_size": effective_batch,
            "micro_batch_size": micro_batch,
            "gradient_accumulation": accumulation,
            "updates_per_epoch": int(updates_per_epoch),
            "epochs": int(cfg.trainer.max_epochs),
            "max_steps": int(cfg.trainer.get("max_steps", -1)),
            "total_updates": int(
                min(
                    updates_per_epoch * int(cfg.trainer.max_epochs),
                    int(cfg.trainer.max_steps)
                    if int(cfg.trainer.get("max_steps", -1)) > 0
                    else updates_per_epoch * int(cfg.trainer.max_epochs),
                )
            ),
        },
        "normalizer_stats": normalizer_stats,
        "sample_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
            "clip_indices_sha256": hashlib.sha256(clip_indices.tobytes()).hexdigest(),
            "train_indices_sha256": hashlib.sha256(train_indices.tobytes()).hexdigest(),
            "validation_indices_sha256": hashlib.sha256(val_indices.tobytes()).hexdigest(),
        },
        "code_commit": commit,
        "code_sha256": code_hashes,
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
    }
    output_path = run_dir / "round5_phase5_training_identity.json"
    output_path.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    """解析训练配置，构造数据、policy、日志与 checkpoint，并启动 Lightning fit。"""
    _validate_round4_gpu_visibility(cfg)
    pl.seed_everything(int(cfg.seed), workers=True)
    if str(cfg.get("train_mode", "")) == "stage_abde":
        with open_dict(cfg):
            cfg.round4_metadata = {
                "data_version": "full_original_dataset_window_split_v1",
                "split_mode": "window_level_random_split",
                "split_seed": int(cfg.seed),
                "train_split": float(cfg.train_split),
                "normalizer_source": "full_original_dataset",
                "online_transition_source": "excluded",
                "code_commit": _round4_git_commit(),
            }
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    run_dir, run_id = get_run_dir(cfg, dataset_name)
    log_path = tee_output_to_file(run_dir, dataset_name)

    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = load_dataset(
        dataset_name,
        transform=None,
        cache_dir=cache_dir,
        **dataset_cfg,
    )
    phase5_expansion = cfg.get("round5_phase5_expansion", {})
    phase5_expansion_enabled = bool(phase5_expansion.get("enabled", False))
    if phase5_expansion_enabled:
        effective_window_steps = int(phase5_expansion.effective_window_steps)
        clip_storage_steps = int(phase5_expansion.clip_storage_steps)
        goal_index = int(cfg.policy.stage_a_goal_index)
        if goal_index * int(dataset.frameskip) != effective_window_steps:
            raise ValueError(
                "Phase5 Stage-A goal index and action-block length must reach "
                "the common effective window"
            )
        if dataset.span < effective_window_steps + 1:
            raise ValueError(
                "Phase5 dataset sample does not contain the final goal observation"
            )
        if dataset.span > clip_storage_steps:
            raise ValueError(
                "Phase5 dataset sample exceeds the common clip-storage span"
            )
        dataset.clip_indices = [
            (int(episode_idx), int(start))
            for episode_idx, length in enumerate(dataset.lengths)
            if int(length) >= clip_storage_steps
            for start in range(int(length) - clip_storage_steps + 1)
        ]
        if not dataset.clip_indices:
            raise ValueError("Phase5 common training window produced no clips")
        print(
            "Using common Phase5 training window: "
            f"{len(dataset.clip_indices)} clips, "
            f"{effective_window_steps} effective steps, "
            f"{clip_storage_steps} stored steps"
        )
    phase17_split = _phase17_episode_partition(cfg, dataset)
    table2_enabled = bool(cfg.get("cvpr_table2", {}).get("enabled", False))
    gpu_image_preprocessing, chunk_size = _data_pipeline_options(cfg)
    transforms = []
    if not gpu_image_preprocessing:
        transforms.append(
            get_img_preprocessor(
                source="pixels", target="pixels", img_size=cfg.img_size
            )
        )

    normalizer_stats = {}
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            if phase5_expansion_enabled:
                normalizer, stats = get_column_normalizer(
                    dataset,
                    col,
                    col,
                    return_stats=True,
                )
                normalizer_stats[col] = stats
            elif table2_enabled:
                normalizer, stats = get_column_normalizer(
                    dataset,
                    col,
                    col,
                    return_stats=True,
                )
                normalizer_stats[col] = stats
            elif phase17_split is None:
                normalizer = get_column_normalizer(dataset, col, col)
            else:
                normalizer, stats = get_column_normalizer(
                    dataset,
                    col,
                    col,
                    row_indices=phase17_split["train_row_indices"],
                    return_stats=True,
                )
                phase17_split.setdefault("normalizer_stats", {})[col] = stats
            transforms.append(normalizer)

        # 一个模型动作 token 包含 frameskip 个环境动作，因此维度需同步写回配置。
        action_block_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
        if "action_encoder" in cfg.policy.model:
            cfg.policy.model.action_encoder.input_dim = action_block_dim
        if "action_dim" in cfg.policy.model:
            cfg.policy.model.action_dim = action_block_dim

    dataset.transform = spt.data.transforms.Compose(*transforms)
    if gpu_image_preprocessing:
        print("Using channel-first uint8 pixels with device-side ImageNet normalization")
    if chunk_size is not None:
        print(f"Using distributed HDF5 chunk-local sampler (chunk_size={chunk_size})")

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    if phase17_split is None:
        train_set, val_set = spt.data.random_split(
            dataset,
            lengths=[cfg.train_split, 1 - cfg.train_split],
            generator=rnd_gen,
        )
    else:
        train_set = phase17_split["train_set"]
        val_set = phase17_split["val_set"]
    if phase5_expansion_enabled:
        if not hasattr(train_set, "indices") or not hasattr(val_set, "indices"):
            raise TypeError("Phase5 requires index-based train and validation splits")
        effective_batch_size = int(phase5_expansion.effective_batch_size)
        train_indices = list(train_set.indices)
        kept_train_count = len(train_indices) // effective_batch_size * effective_batch_size
        if kept_train_count == 0:
            raise ValueError("Phase5 training split is smaller than one effective batch")
        train_set = torch.utils.data.Subset(
            dataset,
            train_indices[:kept_train_count],
        )
        print(
            "Phase5 effective-batch boundary: "
            f"{kept_train_count} training windows at "
            f"effective batch {effective_batch_size}"
        )
    train, val = _make_dataloaders(cfg, train_set, val_set, rnd_gen, chunk_size)

    policy = hydra.utils.instantiate(cfg.policy)

    initial_epoch = int(cfg.get("initial_epoch", 0))
    if initial_epoch < 0:
        raise ValueError("initial_epoch must be non-negative")
    initial_weights = cfg.get("init_weights", None)
    if initial_weights:
        _initialize_from_weights(policy, initial_weights)
        print(
            "Initialized model weights from "
            f"{initial_weights}; optimizer state was not restored"
        )
        if str(cfg.get("train_mode", "")) == "stage_abde":
            with open_dict(cfg):
                cfg.round4_metadata["resume_from_weights"] = str(initial_weights)
                cfg.round4_metadata["optimizer_state_restored"] = False
                cfg.round4_metadata["initial_epoch"] = initial_epoch
    if initial_epoch:
        policy._round4_epoch_offset = initial_epoch

    with open_dict(cfg):
        cfg.subdir = run_id
        cfg.trainer.default_root_dir = str(run_dir)

    logger = None
    if cfg.wandb.enabled:
        wandb_config = OmegaConf.to_container(cfg.wandb.config, resolve=True)
        wandb_config["id"] = run_id
        if not wandb_config.get("name") or wandb_config.get("name") == cfg.output_model_name:
            wandb_config["name"] = run_id
        wandb_config["save_dir"] = str(run_dir)
        os.environ["WANDB_DIR"] = str(run_dir)
        logger = WandbLogger(**wandb_config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    run_dir.mkdir(parents=True, exist_ok=True)
    if phase5_expansion_enabled:
        identity_path = _round5_phase5_training_manifest(
            cfg,
            dataset,
            train_set,
            val_set,
            normalizer_stats,
            run_dir,
        )
        print(f"Wrote Phase5 training identity: {identity_path}")
    if phase17_split is not None:
        state_model = getattr(policy, "model", policy)
        phase17_metadata = {
            "schema_version": 1,
            "task": str(phase17_split["manifest"]["task"]),
            "seed": int(cfg.seed),
            "episode_split_manifest": str(phase17_split["manifest_path"]),
            "episode_split_sha256": hashlib.sha256(
                phase17_split["manifest_path"].read_bytes()
            ).hexdigest(),
            "episode_column": phase17_split["episode_column"],
            "train_window_count": phase17_split["train_window_count"],
            "dev_window_count": phase17_split["dev_window_count"],
            "confirmation_window_count": phase17_split["confirmation_window_count"],
            "train_episode_count": len(phase17_split["manifest"]["train_episode_ids"]),
            "dev_episode_count": len(phase17_split["manifest"]["dev_episode_ids"]),
            "confirmation_episode_count": len(
                phase17_split["manifest"]["confirmation_episode_ids"]
            ),
            "normalizers": phase17_split.get("normalizer_stats", {}),
            "initial_model_state_sha256": _state_dict_sha256(state_model),
        }
        (run_dir / "phase1_7_metadata.json").write_text(
            json.dumps(phase17_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if table2_enabled:
        from source.common.cvpr_table2 import write_training_identity

        if cfg.get("init_weights", None) or cfg.get("resume_ckpt", None):
            raise ValueError("CVPR Table 2 core runs must start from fresh common initialization")
        table2_code_paths = (
            Path(__file__),
            Path("source/common/cvpr_table2.py"),
            Path("source/common/logging.py"),
            Path("source/policy/fast_lewam.py"),
            Path("source/model/fast_lewam/jepa.py"),
            Path("source/model/fast_lewam/round4.py"),
            Path("config/train/round4_ab.yaml"),
            Path("config/train/policy/round4_ab.yaml"),
            Path(f"config/train/data/{HydraConfig.get().runtime.choices['data']}.yaml"),
        )
        identity_path = write_training_identity(
            cfg=cfg,
            dataset=dataset,
            train_set=train_set,
            val_set=val_set,
            train_loader=train,
            policy=policy,
            normalizer_stats=normalizer_stats,
            run_dir=run_dir,
            code_paths=table2_code_paths,
        )
        print(f"Wrote CVPR Table 2 training identity: {identity_path}")
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    with open(run_dir / "run_paths.txt", "w") as f:
        f.write(f"run_dir={run_dir}\n")
        f.write(f"log_path={log_path}\n")
    _phase6_1_training_manifest(cfg, policy, run_dir)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.policy,
        # Table 2 freezes and evaluates epoch 10 only. Keeping intermediate
        # full model snapshots would multiply disk use across its 60 runs.
        epoch_interval=10 if table2_enabled else 1,
        output_dir=run_dir / CHECKPOINTS_DIRNAME,
        epoch_offset=initial_epoch,
    )
    checkpoint_interval_steps = cfg.get("checkpoint_interval_steps", None)
    checkpoint_trigger = {}
    if checkpoint_interval_steps is None:
        checkpoint_trigger["train_time_interval"] = timedelta(
            minutes=float(cfg.get("checkpoint_interval_minutes", 30))
        )
    else:
        checkpoint_interval_steps = int(checkpoint_interval_steps)
        if checkpoint_interval_steps < 1:
            raise ValueError("checkpoint_interval_steps must be a positive integer")
        # Lightning's time-triggered ModelCheckpoint path does not trigger for
        # manual-optimization modules. Step-based saving works for both modes.
        checkpoint_trigger["every_n_train_steps"] = checkpoint_interval_steps

    latest_checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir / CHECKPOINTS_DIRNAME,
        save_last=True,
        save_top_k=0,
        enable_version_counter=False,
        **checkpoint_trigger,
    )

    callbacks = [object_dump_callback, latest_checkpoint_callback]
    if bool(cfg.get("round5_phase6_1", {}).get("enabled", False)):
        callbacks.append(
            _Phase6_1EpochMetricsCallback(
                run_dir / "phase6_1_epoch_metrics.jsonl"
            )
        )
    if table2_enabled:
        from source.common.cvpr_table2 import Table2TrainingAuditCallback

        callbacks.append(
            Table2TrainingAuditCallback(run_dir / "table2_training_progress.json")
        )
    validation_diagnostics_cfg = cfg.get("validation_diagnostics", {})
    if validation_diagnostics_cfg.get("enabled", False):
        callback_kwargs = OmegaConf.to_container(
            validation_diagnostics_cfg, resolve=True
        )
        callback_kwargs.pop("enabled")
        from source.common.fast_lewam_validation import (
            FastLeWAMValidationDiagnosticsCallback,
        )

        callbacks.append(FastLeWAMValidationDiagnosticsCallback(**callback_kwargs))
    epoch_eval_cfg = cfg.get("epoch_eval", {})
    if epoch_eval_cfg.get("enabled", False):
        callback_kwargs = OmegaConf.to_container(epoch_eval_cfg, resolve=True)
        callback_kwargs.pop("enabled")
        if callback_kwargs.get("config_name") is None:
            callback_kwargs["config_name"] = HydraConfig.get().runtime.choices["data"]
        from source.common.epoch_eval import FastLeWAMEpochEvalCallback

        callbacks.append(FastLeWAMEpochEvalCallback(**callback_kwargs))
    round4_diagnostics_cfg = cfg.get("round4_diagnostics", {})
    if round4_diagnostics_cfg.get("enabled", False):
        from source.common.round4_diagnostics import Round4DiagnosticsCallback

        callback_kwargs = OmegaConf.to_container(
            round4_diagnostics_cfg, resolve=True
        )
        callback_kwargs.pop("enabled", None)
        callback_kwargs.setdefault("epoch_offset", initial_epoch)
        callbacks.append(Round4DiagnosticsCallback(**callback_kwargs))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    resume_ckpt = resolve_training_resume_checkpoint(
        cfg.get("resume_ckpt", None),
        run_dir,
        cfg.output_model_name,
    )
    if resume_ckpt is not None:
        print(f"Resuming training from checkpoint: {resume_ckpt}")

    trainer.fit(
        policy,
        train_dataloaders=train,
        val_dataloaders=val,
        ckpt_path=resume_ckpt,
    )


if __name__ == "__main__":
    run()
