"""Small offline Reacher continuation for the CVPR Table 1 refinement study."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from source.common.checkpoint import torch_load_compat
from source.common.round3_phase1 import CohortManifest
from source.common.round5_phase1_6 import stable_hash
from source.common.round5_phase1_6_train import (
    _load_model,
    _make_training_dataset,
    _training_run_config,
)


HORIZON = 5
ARMS = ("b_specific", "shared_preserve")
FROZEN_ROUND4_PREFIXES = (
    "d_",
    "e_",
    "inverse_dynamics_head.",
    "encoder.",
    "projector.",
)
B_SPECIFIC_PARAMETERS = {
    "latent_head.weight",
    "latent_head.bias",
    "query_tokens",
    "joint_positions",
    "mode_embedding.weight",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_key(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temp_name)
        with open(temp_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _episode_id_set(cohort_paths: list[Path]) -> tuple[set[str], dict[str, str]]:
    excluded: set[str] = set()
    hashes = {}
    for path in cohort_paths:
        manifest = CohortManifest.load(path)
        hashes[str(path.resolve())] = _sha256(path)
        excluded.update(_json_key(entry.episode_id) for entry in manifest.entries)
    return excluded, hashes


def prepare_offline_cache(
    *,
    checkpoint: str | Path,
    cohort_paths: list[str | Path],
    output_dir: str | Path,
    device: str,
    split_seed: int = 3072,
    cache_seed: int = 3072,
    split_fraction: float = 0.9,
    trajectories: int = 512,
    windows_per_trajectory: int = 16,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Cache 8192 source-training windows after excluding all Table 1 episodes."""
    checkpoint = Path(checkpoint).resolve(strict=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "offline_latents.pt"
    receipt_path = output / "cache_receipt.json"
    excluded, cohort_hashes = _episode_id_set([Path(path) for path in cohort_paths])
    checkpoint_hash = _sha256(checkpoint)
    cfg, cfg_path = _training_run_config(checkpoint)
    requested = {
        "schema_version": 1,
        "task": "reacher",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "training_config_sha256": _sha256(cfg_path),
        "cohort_sha256": cohort_hashes,
        "excluded_episode_ids_sha256": stable_hash(sorted(excluded)),
        "split_seed": int(split_seed),
        "split_fraction": float(split_fraction),
        "cache_seed": int(cache_seed),
        "trajectories": int(trajectories),
        "windows_per_trajectory": int(windows_per_trajectory),
    }
    if cache_path.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if all(receipt.get(key) == value for key, value in requested.items()) and receipt.get(
            "cache_sha256"
        ) == _sha256(cache_path):
            return receipt

    import stable_pretraining as spt

    dataset = _make_training_dataset(cfg)
    generator = torch.Generator().manual_seed(int(split_seed))
    train_subset, _validation_subset = spt.data.random_split(
        dataset,
        [float(split_fraction), 1.0 - float(split_fraction)],
        generator=generator,
    )
    train_indices = set(int(index) for index in train_subset.indices)
    grouped: dict[str, list[tuple[int, int, Any]]] = {}
    for data_index, record in enumerate(dataset.clip_indices):
        if int(data_index) not in train_indices:
            continue
        episode = record[0].item() if isinstance(record[0], np.generic) else record[0]
        key = _json_key(episode)
        if key in excluded:
            continue
        grouped.setdefault(key, []).append((int(data_index), int(record[1]), episode))
    eligible = [
        (key, values)
        for key, values in sorted(grouped.items())
        if len(values) >= int(windows_per_trajectory)
    ]
    if len(eligible) < int(trajectories):
        raise RuntimeError(
            f"only {len(eligible)} train-only Reacher episodes remain; need {trajectories}"
        )
    rng = np.random.default_rng(int(cache_seed))
    episode_order = rng.permutation(len(eligible))[: int(trajectories)]
    selected: list[tuple[int, int, Any]] = []
    selected_episode_records = []
    for position in episode_order:
        key, values = eligible[int(position)]
        picks = rng.choice(len(values), size=int(windows_per_trajectory), replace=False)
        chosen = [values[int(index)] for index in picks]
        selected.extend(chosen)
        selected_episode_records.append(
            {"episode_id": chosen[0][2], "local_starts": sorted(row[1] for row in chosen)}
        )
    selected.sort(key=lambda row: (_json_key(row[2]), row[1], row[0]))

    model, _ = _load_model(checkpoint, torch.device(device))
    model.eval().requires_grad_(False)
    latent_batches = []
    action_batches = []
    for start in range(0, len(selected), int(batch_size)):
        indices = [row[0] for row in selected[start : start + int(batch_size)]]
        samples = [dataset[index] for index in indices]
        pixels = torch.stack([torch.as_tensor(sample["pixels"]) for sample in samples])
        actions = torch.stack([torch.as_tensor(sample["action"]).float() for sample in samples])
        if pixels.ndim != 5 or pixels.shape[1] != HORIZON + 1:
            raise ValueError(f"expected pixels [B,6,C,H,W], got {tuple(pixels.shape)}")
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (
            HORIZON + 1,
            int(model.action_dim),
        ):
            raise ValueError(f"expected actions [B,6,{model.action_dim}], got {tuple(actions.shape)}")
        with torch.inference_mode():
            latents = model.encode_pixels(pixels.to(device, non_blocking=True)).float().cpu()
        latent_batches.append(latents)
        action_batches.append(torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0).cpu())
    latent = torch.cat(latent_batches).contiguous()
    action = torch.cat(action_batches).contiguous()
    if len(latent) != int(trajectories) * int(windows_per_trajectory):
        raise RuntimeError("cached training window count does not match its frozen budget")
    if not torch.isfinite(latent).all() or not torch.isfinite(action).all():
        raise RuntimeError("offline latent/action cache contains non-finite values")
    distance_blocks = latent[:, 1:, None, :] - latent[:, None, 1:, :]
    distances = distance_blocks.square().mean(dim=-1).reshape(-1)
    positive = distances[distances > 0]
    if positive.numel() == 0:
        raise RuntimeError("offline goal-distance scale has no positive values")
    distance_scale = float(positive.median())
    payload = {
        **requested,
        "latent": latent,
        "action": action,
        "episode_id": [row[2] for row in selected],
        "local_start": [row[1] for row in selected],
        "selected_episodes": selected_episode_records,
        "distance_scale": distance_scale,
        "window_count": len(latent),
    }
    _atomic_torch_save(cache_path, payload)
    receipt = {
        **requested,
        "cache_path": str(cache_path.resolve()),
        "cache_sha256": _sha256(cache_path),
        "window_count": len(latent),
        "episode_count": len(selected_episode_records),
        "distance_scale": distance_scale,
        "train_split_indices_sha256": stable_hash(sorted(train_indices)),
        "selected_episodes": selected_episode_records,
    }
    _atomic_json(receipt_path, receipt)
    del model, dataset
    return receipt


def _prepare_trainable_parameters(model: torch.nn.Module, arm: str):
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    mode_rows = {"b_specific": (1,), "shared_preserve": (0, 1)}[arm]
    frozen: dict[str, torch.Tensor] = {}
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(False)
        if name.startswith(FROZEN_ROUND4_PREFIXES):
            frozen[name] = parameter.detach().cpu().clone()
            continue
        if arm == "b_specific":
            should_train = name in B_SPECIFIC_PARAMETERS
        else:
            should_train = True
        if should_train:
            parameter.requires_grad_(True)
            trainable.append((name, parameter))
        else:
            frozen[name] = parameter.detach().cpu().clone()
    for name, buffer in model.named_buffers():
        if name.startswith(("encoder.", "projector.")):
            frozen[f"buffer::{name}"] = buffer.detach().cpu().clone()
    if not trainable or not any(name == "mode_embedding.weight" for name, _ in trainable):
        raise RuntimeError("training arm did not expose its intended Stage-B mode parameters")
    return trainable, frozen, mode_rows


def _trajectory_losses(predicted, target_latents, goal_latents, distance_scale, ranking_margin):
    scale = max(float(distance_scale), 1e-6)
    latent_loss = F.mse_loss(predicted.float(), target_latents.float())
    predicted_cost = (predicted.float() - goal_latents[:, None].float()).square().mean(dim=-1)
    target_cost = (target_latents.float() - goal_latents[:, None].float()).square().mean(dim=-1)
    cost_loss = F.smooth_l1_loss(predicted_cost / scale, target_cost / scale)
    left, right = torch.triu_indices(
        predicted.shape[1], predicted.shape[1], offset=1, device=predicted.device
    )
    true_delta = (target_cost[:, right] - target_cost[:, left]) / scale
    predicted_delta = (predicted_cost[:, right] - predicted_cost[:, left]) / scale
    valid = true_delta.abs() >= float(ranking_margin)
    if bool(valid.any()):
        signs = true_delta[valid].sign()
        temporal_loss = F.softplus(-signs * predicted_delta[valid]).mean()
    else:
        temporal_loss = predicted_cost.new_zeros(())
    return latent_loss, cost_loss, temporal_loss


def train_checkpoint(
    *,
    checkpoint: str | Path,
    checkpoint_sha256: str,
    cache_path: str | Path,
    output_dir: str | Path,
    arm: str,
    seed: int,
    device: str,
    updates: int = 2000,
    checkpoint_updates: tuple[int, ...] = (500, 1000, 2000),
    batch_size: int = 128,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-3,
    gradient_clip: float = 1.0,
    cost_loss_weight: float = 0.1,
    temporal_loss_weight: float = 0.1,
    actor_preserve_loss_weight: float = 1.0,
    ranking_margin: float = 0.05,
) -> dict[str, Any]:
    """Run one paired 2000-update Reacher continuation and save loadable weights."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if int(updates) < 1 or int(batch_size) != 128:
        raise ValueError("updates must be positive and batch_size must remain 128")
    checkpoint = Path(checkpoint).resolve(strict=True)
    cache_path = Path(cache_path).resolve(strict=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_sha = _sha256(cache_path)
    train_config = checkpoint.parent.parent / "config.yaml"
    if not train_config.is_file():
        raise FileNotFoundError(f"training config is missing: {train_config}")
    if _sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("base checkpoint SHA256 differs from the frozen training identity")
    from omegaconf import OmegaConf

    source_config, _source_config_path = _training_run_config(checkpoint)
    OmegaConf.save(source_config, output / "config.yaml", resolve=False)
    cache = torch_load_compat(cache_path, map_location="cpu")
    if cache.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("offline cache belongs to a different base checkpoint")
    if cache.get("window_count") != 8192:
        raise ValueError("offline cache must contain exactly 8192 source training windows")
    cache_latent = cache["latent"].float().contiguous()
    cache_action = cache["action"].float().contiguous()
    if cache_latent.shape[1] != HORIZON + 1 or cache_action.shape[1] != HORIZON + 1:
        raise ValueError("training latent and action windows must each contain six frames")
    distance_scale = float(cache["distance_scale"])

    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    rng = np.random.default_rng(int(seed))
    sample_schedule = rng.integers(
        0, len(cache_latent), size=(int(updates), int(batch_size)), dtype=np.int64
    )
    goal_schedule = rng.integers(
        1, HORIZON + 1, size=(int(updates), int(batch_size)), dtype=np.int64
    )
    time_schedule = rng.uniform(
        0.1, 0.9, size=(int(updates), int(batch_size))
    ).astype(np.float32)
    action_noise_schedule = rng.standard_normal(
        (int(updates), int(batch_size), HORIZON, int(cache_action.shape[-1]))
    ).astype(np.float32)
    schedule_hashes = {
        "sample_indices": hashlib.sha256(sample_schedule.tobytes()).hexdigest(),
        "goal_indices": hashlib.sha256(goal_schedule.tobytes()).hexdigest(),
        "timesteps": hashlib.sha256(time_schedule.tobytes()).hexdigest(),
        "action_noise": hashlib.sha256(action_noise_schedule.tobytes()).hexdigest(),
    }

    model, _ = _load_model(checkpoint, torch.device(device))
    model.train()
    model.encoder.eval()
    model.projector.eval()
    teacher = None
    if arm == "shared_preserve":
        teacher, _ = _load_model(checkpoint, torch.device(device))
        teacher.eval().requires_grad_(False)
    trainable, frozen_state, mode_rows = _prepare_trainable_parameters(model, arm)
    mode_parameter = dict(trainable)["mode_embedding.weight"]
    mode_reference = mode_parameter.detach().clone()
    parameter_groups = []
    for name, parameter in trainable:
        parameter_groups.append(
            {
                "params": [parameter],
                "weight_decay": 0.0 if name == "mode_embedding.weight" else float(weight_decay),
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups, lr=float(learning_rate))
    device_obj = torch.device(device)
    logs = []
    saved_paths = []
    save_updates = set(int(step) for step in checkpoint_updates) | {int(updates)}
    for update in range(int(updates)):
        ids = torch.as_tensor(sample_schedule[update], dtype=torch.long)
        goal_ids = torch.as_tensor(goal_schedule[update], dtype=torch.long)
        latent = cache_latent[ids].to(device_obj, non_blocking=True)
        actions = cache_action[ids, :HORIZON].to(device_obj, non_blocking=True)
        start = latent[:, 0]
        targets = latent[:, 1 : HORIZON + 1]
        goal = latent[torch.arange(len(ids)), goal_ids].to(device_obj, non_blocking=True)
        timestep = torch.as_tensor(time_schedule[update], dtype=torch.float32, device=device_obj)
        noise = torch.as_tensor(
            action_noise_schedule[update], dtype=actions.dtype, device=device_obj
        )
        noisy_actions = (1.0 - timestep[:, None, None]) * noise + timestep[:, None, None] * actions
        target_velocity = actions - noise

        optimizer.zero_grad(set_to_none=True)
        predicted = model(
            start,
            actions,
            torch.ones(len(ids), dtype=start.dtype, device=device_obj),
            mode="stage_b",
        )["predicted_latents"]
        latent_loss, cost_loss, temporal_loss = _trajectory_losses(
            predicted,
            targets,
            goal,
            distance_scale,
            ranking_margin,
        )
        loss = latent_loss + float(cost_loss_weight) * cost_loss + float(
            temporal_loss_weight
        ) * temporal_loss
        actor_loss = loss.new_zeros(())
        if arm == "shared_preserve":
            with torch.no_grad():
                teacher_velocity = teacher(
                    start,
                    noisy_actions,
                    timestep,
                    mode="stage_a",
                    goal_latent=goal,
                )["action_velocity"]
            student_velocity = model(
                start,
                noisy_actions,
                timestep,
                mode="stage_a",
                goal_latent=goal,
            )["action_velocity"]
            actor_loss = F.mse_loss(student_velocity.float(), teacher_velocity.float())
            loss = loss + float(actor_preserve_loss_weight) * actor_loss
        loss.backward()
        if mode_parameter.grad is not None:
            row_mask = torch.ones(mode_parameter.shape[0], dtype=torch.bool, device=device_obj)
            row_mask[list(mode_rows)] = False
            mode_parameter.grad[row_mask] = 0
        torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], float(gradient_clip))
        optimizer.step()
        with torch.no_grad():
            row_mask = torch.ones(mode_parameter.shape[0], dtype=torch.bool, device=device_obj)
            row_mask[list(mode_rows)] = False
            mode_parameter[row_mask] = mode_reference[row_mask]

        completed = update + 1
        if completed in {1, 100, 500, 1000, int(updates)}:
            record = {
                "update": completed,
                "loss": float(loss.detach().cpu()),
                "latent_loss": float(latent_loss.detach().cpu()),
                "cost_loss": float(cost_loss.detach().cpu()),
                "temporal_loss": float(temporal_loss.detach().cpu()),
                "actor_preserve_loss": float(actor_loss.detach().cpu()),
            }
            logs.append(record)
            _atomic_json(output / "training_progress.json", {"logs": logs})
        if completed in save_updates:
            name = (
                f"r4_ab_seed3072_refine_{arm}_seed{int(seed)}_weights_epoch_{completed}.pt"
            )
            path = output / "checkpoints" / name
            _atomic_torch_save(path, model.state_dict())
            saved_paths.append(str(path.resolve()))

    for name, before in frozen_state.items():
        if name.startswith("buffer::"):
            current = dict(model.named_buffers())[name[len("buffer::") :]]
        else:
            current = dict(model.named_parameters())[name]
        if not torch.equal(current.detach().cpu(), before):
            raise RuntimeError(f"frozen parameter or buffer changed during training: {name}")
    for row in set(range(mode_parameter.shape[0])) - set(mode_rows):
        if not torch.equal(mode_parameter[row].detach().cpu(), mode_reference[row].detach().cpu()):
            raise RuntimeError("a frozen mode-embedding row changed during training")

    receipt = {
        "schema_version": 1,
        "task": "reacher",
        "arm": arm,
        "training_seed": int(seed),
        "base_checkpoint": str(checkpoint),
        "base_checkpoint_sha256": checkpoint_sha256,
        "cache_path": str(cache_path),
        "cache_sha256": cache_sha,
        "updates": int(updates),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "mode_embedding_train_rows": list(mode_rows),
        "schedule_sha256": schedule_hashes,
        "frozen_parameter_names": sorted(name for name in frozen_state if not name.startswith("buffer::")),
        "frozen_buffer_names": sorted(name[len("buffer::") :] for name in frozen_state if name.startswith("buffer::")),
        "checkpoints": saved_paths,
        "logs": logs,
    }
    _atomic_json(output / "training_receipt.json", receipt)
    del model, teacher, cache
    return receipt
