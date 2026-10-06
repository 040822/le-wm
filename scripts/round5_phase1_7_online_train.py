#!/usr/bin/env python3
"""Prepare and train frozen-representation online scorer arms for Round 5 Phase1.7."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.checkpoint import load_policy_or_model, torch_load_compat
from source.common.eval import compose_eval_config, get_dataset, img_transform
from source.common.round3_phase1 import CohortManifest
from source.common.round5_phase1_6 import stable_hash


HORIZON = 5
FRAMESKIP = 5
BATCH_SIZE = 128
UPDATE_COUNT = 2000
OFFLINE_TRAJECTORIES = 512
OFFLINE_WINDOWS_PER_TRAJECTORY = 16
ARMS = {"offline", "branches", "rank_0.1", "rank_1.0"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        torch.save(value, temporary_name)
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _register_config_resolvers() -> None:
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)


def _run_dir(checkpoint: Path) -> Path:
    return checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent


def _read_training_config(checkpoint: Path) -> tuple[Any, Path]:
    path = _run_dir(checkpoint) / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint training config is missing: {path}")
    _register_config_resolvers()
    return OmegaConf.load(path), path


def _load_policy_model(checkpoint: Path, device: torch.device):
    policy, resolved = load_policy_or_model(str(checkpoint))
    if resolved is not None and Path(resolved).resolve() != checkpoint.resolve():
        raise ValueError(f"checkpoint resolver changed requested path: {resolved}")
    model = getattr(policy, "model", policy)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"checkpoint has no trainable torch model: {checkpoint}")
    model.to(device)
    return policy, model


def _model_kind(model: torch.nn.Module) -> str:
    if callable(getattr(model, "rollout_latents", None)) and callable(
        getattr(model, "encode_pixels", None)
    ):
        return "joint"
    if callable(getattr(model, "predict", None)) and callable(
        getattr(model, "action_encoder", None)
    ):
        return "lewm"
    raise TypeError(f"unsupported Phase1.7 scorer architecture: {type(model).__name__}")


def _encode_pixels(
    model: torch.nn.Module,
    images: list[np.ndarray] | np.ndarray,
    transform: Any,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    image_values = list(images) if not isinstance(images, np.ndarray) else list(images)
    if not image_values:
        raise ValueError("cannot encode an empty image batch")
    chunks = []
    model.eval()
    for start in range(0, len(image_values), int(batch_size)):
        batch = torch.stack(
            [transform(np.asarray(image)) for image in image_values[start : start + int(batch_size)]]
        ).to(device, non_blocking=True)
        with torch.inference_mode():
            if callable(getattr(model, "encode_pixels", None)):
                encoded = model.encode_pixels(batch)
            else:
                encoded = model.encode({"pixels": batch[:, None]})["emb"][:, 0]
        encoded = encoded.float().detach().cpu().numpy()
        if encoded.ndim != 2 or not np.isfinite(encoded).all():
            raise ValueError(f"image encoder returned invalid latents: {encoded.shape}")
        chunks.append(encoded)
    return np.concatenate(chunks, axis=0)


def _state_id(entry: Any) -> str:
    return f"episode={entry.episode_id};start={int(entry.start_step)};row={int(entry.row_index)}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _load_branch_cache(
    *,
    task: str,
    scorer: str,
    checkpoint: Path,
    branch_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    collection_path = branch_dir / "collection_manifest.json"
    cohort_path = branch_dir / "online_train_cohort.json"
    split_path = branch_dir / "branch_data_split.json"
    pool_path = branch_dir / "candidate_pool.npz"
    identity_path = branch_dir / "candidate_pool_identity.json"
    for required in (collection_path, cohort_path, split_path, pool_path, identity_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    if collection.get("status") != "completed":
        raise ValueError(f"online branch collection is not complete: {collection_path}")
    if collection.get("task") != task or int(collection.get("state_count", -1)) != 256:
        raise ValueError("online branch collection task/state count does not match Phase1.7")
    if int(collection.get("candidate_count", -1)) != 32:
        raise ValueError("online branch pool must contain 32 candidates per state")
    cohort = CohortManifest.load(cohort_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = list(split.get("train_state_ids", ()))
    dev_ids = list(split.get("dev_state_ids", ()))
    if len(train_ids) != 205 or len(dev_ids) != 51 or set(train_ids) & set(dev_ids):
        raise ValueError("formal online split must be disjoint 205/51 state IDs")
    if len(cohort.entries) != 256:
        raise ValueError("online cohort does not contain 256 states")

    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    actor_metadata_path = Path(str(collection["training_metadata"])).expanduser()
    if not actor_metadata_path.is_absolute():
        actor_metadata_path = (ROOT / actor_metadata_path).resolve()
    scorer_metadata_path = _run_dir(checkpoint) / "phase1_7_metadata.json"
    if not actor_metadata_path.is_file() or not scorer_metadata_path.is_file():
        raise FileNotFoundError("actor/scorer Phase1.7 training metadata is missing")
    actor_metadata = json.loads(actor_metadata_path.read_text(encoding="utf-8"))
    scorer_metadata = json.loads(scorer_metadata_path.read_text(encoding="utf-8"))
    actor_action_stats = actor_metadata.get("normalizers", {}).get("action", {})
    scorer_action_stats = scorer_metadata.get("normalizers", {}).get("action", {})
    for key in ("mean", "std"):
        if not np.array_equal(
            np.asarray(actor_action_stats.get(key), dtype=np.float64),
            np.asarray(scorer_action_stats.get(key), dtype=np.float64),
        ):
            raise ValueError("Joint actor and scorer train-only action normalizers differ")
    action_normalizer_sha256 = stable_hash(
        {"mean": actor_action_stats["mean"], "std": actor_action_stats["std"]}
    )
    with np.load(pool_path, allow_pickle=False) as archive:
        candidate_pool = np.asarray(archive["candidate_pool"], dtype=np.float32)
        candidate_type = np.asarray(archive["candidate_type"]).astype("U20")
    if candidate_pool.shape != (256, 32, HORIZON, 10):
        raise ValueError(f"unexpected candidate action pool shape: {candidate_pool.shape}")
    if identity.get("candidate_pool_sha256") != collection.get("candidate_pool_sha256"):
        raise ValueError("candidate pool identity disagrees with collection manifest")
    if identity.get("cohort_sha256") != cohort.computed_sha256:
        raise ValueError("candidate pool and branch cohort hashes differ")
    if int(np.sum(candidate_type == "actor")) != 16 or int(
        np.sum(candidate_type == "local_perturbation")
    ) != 16:
        raise ValueError("online candidate pool must contain 16 actor and 16 local candidates")

    ids = [_state_id(entry) for entry in cohort.entries]
    if set(ids) != set(train_ids) | set(dev_ids):
        raise ValueError("state split IDs do not exactly match the saved online cohort")
    split_kind = np.asarray(["train" if value in set(train_ids) else "dev" for value in ids], dtype="U8")
    record_paths = [
        path
        for candidate_index in range(32)
        for path in (
            branch_dir / "branches" / f"candidate_{candidate_index:04d}.jsonl",
            branch_dir / "branches" / "future_pixels" / f"candidate_{candidate_index:04d}.npz",
        )
    ]
    cache_path = output_dir / "branch_latents.pt"
    cache_identity_path = output_dir / "branch_cache_identity.json"
    if cache_path.is_file() and cache_identity_path.is_file() and all(
        path.is_file() for path in record_paths
    ):
        cache_identity = json.loads(cache_identity_path.read_text(encoding="utf-8"))
        source_hashes = {str(path.resolve()): _sha256_file(path) for path in record_paths}
        cache_is_current = (
            cache_identity.get("task") == task
            and cache_identity.get("scorer") == scorer
            and cache_identity.get("base_checkpoint_sha256") == _sha256_file(checkpoint)
            and cache_identity.get("cohort_sha256") == cohort.computed_sha256
            and cache_identity.get("candidate_pool_sha256") == collection["candidate_pool_sha256"]
            and cache_identity.get("branch_record_sha256") == source_hashes
            and cache_identity.get("branch_cache_sha256") == _sha256_file(cache_path)
        )
        if cache_is_current:
            cached = torch_load_compat(cache_path, map_location="cpu")
            if (
                cached.get("task") == task
                and cached.get("scorer") == scorer
                and cached.get("model_kind") == scorer
                and cached.get("base_checkpoint_sha256") == _sha256_file(checkpoint)
                and cached.get("cohort_sha256") == cohort.computed_sha256
                and cached.get("candidate_pool_sha256") == collection["candidate_pool_sha256"]
                and list(cached.get("state_ids", ())) == ids
                and np.array_equal(cached.get("state_split"), split_kind)
            ):
                print(
                    json.dumps(
                        {
                            "status": "reused_branch_cache",
                            "task": task,
                            "scorer": scorer,
                            "cache": str(cache_path),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return cached

    policy, model = _load_policy_model(checkpoint, device)
    del policy
    model_kind = _model_kind(model)
    if model_kind != scorer:
        raise ValueError(f"checkpoint architecture is {model_kind}, requested scorer={scorer}")
    cfg, _ = _read_training_config(checkpoint)
    eval_cfg = compose_eval_config(
        task,
        overrides=["eval.num_eval=1", "world.num_envs=1", "output.save_video=false"],
    )
    dataset = get_dataset(eval_cfg, eval_cfg.eval.dataset_name)
    transform = img_transform(eval_cfg)
    start_rows = [int(entry.row_index) for entry in cohort.entries]
    goal_rows = [int(entry.goal_row_index) for entry in cohort.entries]
    if any(row < 0 for row in (*start_rows, *goal_rows)):
        raise ValueError("online cohort contains an invalid dataset row")
    start_images = np.asarray(dataset.get_row_data(start_rows)["pixels"])
    goal_images = np.asarray(dataset.get_row_data(goal_rows)["pixels"])
    if len(start_images) != 256 or len(goal_images) != 256:
        raise ValueError("online start/goal dataset image count changed")
    start_latent = _encode_pixels(model, start_images, transform, device)
    goal_latent = _encode_pixels(model, goal_images, transform, device)
    latent_dim = int(start_latent.shape[-1])
    if goal_latent.shape != start_latent.shape:
        raise ValueError("start and goal latent shapes disagree")

    count = 256 * 32
    actions = np.zeros((count, HORIZON, 10), dtype=np.float32)
    future_latents = np.zeros((count, HORIZON, latent_dim), dtype=np.float32)
    valid_mask = np.zeros((count, HORIZON), dtype=np.bool_)
    branch_success = np.zeros((count,), dtype=np.bool_)
    valid_length = np.zeros((count,), dtype=np.int16)
    expected_manifest_hash = str(collection["candidate_pool_sha256"])

    for candidate_index in range(32):
        records_path = branch_dir / "branches" / f"candidate_{candidate_index:04d}.jsonl"
        pixel_path = branch_dir / "branches" / "future_pixels" / f"candidate_{candidate_index:04d}.npz"
        if not records_path.is_file() or not pixel_path.is_file():
            raise FileNotFoundError(f"branch candidate {candidate_index} is incomplete")
        rows = _read_jsonl(records_path)
        if len(rows) != 256:
            raise ValueError(f"candidate {candidate_index} has {len(rows)} states instead of 256")
        image_by_slot_step: dict[tuple[int, int], np.ndarray] = {}
        with np.load(pixel_path, allow_pickle=False) as archive:
            slots = np.asarray(archive["slot"], dtype=np.int64)
            steps = np.asarray(archive["raw_env_step"], dtype=np.int64)
            pixels = np.asarray(archive["pixels"])
            if pixels.ndim != 4 or pixels.dtype != np.uint8:
                raise ValueError(f"candidate {candidate_index} pixels must be uint8 HWC")
            if len(slots) != len(steps) or len(steps) != len(pixels):
                raise ValueError("future pixel archive columns have different lengths")
            for slot, step, image in zip(slots, steps, pixels):
                key = (int(slot), int(step))
                if key in image_by_slot_step:
                    raise ValueError(f"duplicate future image {key} in {pixel_path}")
                image_by_slot_step[key] = image
        pixel_archive_sha256 = _sha256_file(pixel_path)

        future_images: list[np.ndarray] = []
        future_positions: list[tuple[int, int]] = []
        for slot, row in enumerate(rows):
            global_index = slot * 32 + candidate_index
            if (
                int(row.get("slot", -1)) != slot
                or int(row.get("candidate_index", -1)) != candidate_index
                or row.get("candidate_pool_sha256") != expected_manifest_hash
                or row.get("state_id") != ids[slot]
                or row.get("outcome_status") != "completed"
            ):
                raise ValueError(f"branch identity mismatch at candidate={candidate_index}, slot={slot}")
            if row.get("future_pixels_sha256") != pixel_archive_sha256:
                raise ValueError(f"future pixel checksum mismatch: {pixel_path}")
            length = int(row.get("valid_length", -1))
            if length < 1 or length > 25 or len(row.get("steps", ())) != length:
                raise ValueError(f"invalid branch length at candidate={candidate_index}, slot={slot}")
            valid_length[global_index] = length
            branch_success[global_index] = bool(row.get("success", False))
            step_actions = []
            for step in row["steps"]:
                action = step.get("action_normalized")
                if action is None:
                    raise ValueError("online trace omitted a train-normalized primitive action")
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                if action.shape != (2,) or not np.isfinite(action).all():
                    raise ValueError(f"invalid normalized primitive action at branch {global_index}")
                step_actions.append(action)
            usable_blocks = min(HORIZON, length // FRAMESKIP)
            if usable_blocks:
                packed = np.asarray(step_actions[: usable_blocks * FRAMESKIP], dtype=np.float32).reshape(
                    usable_blocks, FRAMESKIP * 2
                )
                proposal = np.asarray(row.get("action"), dtype=np.float32)
                if proposal.shape != (HORIZON, 10) or not np.isfinite(proposal).all():
                    raise ValueError(f"invalid planned action block at branch {global_index}")
                if not np.allclose(
                    proposal, candidate_pool[slot, candidate_index], rtol=0.0, atol=1e-6
                ):
                    raise ValueError(f"branch action differs from the frozen pool at {global_index}")
                if not np.allclose(packed, proposal[:usable_blocks], rtol=0.0, atol=1e-4):
                    raise ValueError(
                        f"recorded actions do not reproduce candidate pool at branch {global_index}"
                    )
                actions[global_index, :usable_blocks] = packed
                for horizon_index in range(usable_blocks):
                    raw_step = (horizon_index + 1) * FRAMESKIP
                    image = image_by_slot_step.get((slot, raw_step))
                    if image is None:
                        raise ValueError(
                            f"missing executed future milestone {raw_step} at branch {global_index}"
                        )
                    valid_mask[global_index, horizon_index] = True
                    future_positions.append((global_index, horizon_index))
                    future_images.append(image)
            endpoint = image_by_slot_step.get((slot, length))
            if endpoint is None:
                raise ValueError(f"branch endpoint image is missing at candidate={candidate_index}, slot={slot}")

        if future_images:
            encoded = _encode_pixels(model, future_images, transform, device)
            if encoded.shape != (len(future_positions), latent_dim):
                raise ValueError("future image latent shape changed")
            for (global_index, horizon_index), latent in zip(future_positions, encoded):
                future_latents[global_index, horizon_index] = latent

    if int(valid_mask.sum()) == 0 or not np.isfinite(future_latents[valid_mask]).all():
        raise ValueError("online branch dataset has no usable finite transitions")
    train_state_mask = np.isin(np.asarray(ids, dtype="U"), np.asarray(train_ids, dtype="U"))
    dev_state_mask = ~train_state_mask
    train_rows = np.repeat(train_state_mask, 32)
    dev_rows = np.repeat(dev_state_mask, 32)
    payload = {
        "schema_version": 1,
        "task": task,
        "scorer": scorer,
        "model_kind": model_kind,
        "base_checkpoint": str(checkpoint.resolve()),
        "base_checkpoint_sha256": _sha256_file(checkpoint),
        "cohort_sha256": cohort.computed_sha256,
        "candidate_pool_sha256": collection["candidate_pool_sha256"],
        "action_normalizer_sha256": action_normalizer_sha256,
        "state_ids": ids,
        "state_split": split_kind,
        "actions": torch.from_numpy(actions),
        "start_latent": torch.from_numpy(np.repeat(start_latent, 32, axis=0).astype(np.float32)),
        "goal_latent": torch.from_numpy(np.repeat(goal_latent, 32, axis=0).astype(np.float32)),
        "future_latent": torch.from_numpy(future_latents),
        "valid_mask": torch.from_numpy(valid_mask),
        "valid_length": torch.from_numpy(valid_length),
        "branch_success_audit_only": torch.from_numpy(branch_success),
        "candidate_type": candidate_type,
        "train_rows": torch.from_numpy(train_rows),
        "dev_rows": torch.from_numpy(dev_rows),
        "record_sources": [str(path.resolve()) for path in record_paths],
    }
    _atomic_torch_save(cache_path, payload)
    identity_payload = {
        "task": task,
        "scorer": scorer,
        "base_checkpoint_sha256": payload["base_checkpoint_sha256"],
        "cohort_sha256": cohort.computed_sha256,
        "candidate_pool_sha256": collection["candidate_pool_sha256"],
        "branch_record_sha256": {str(path): _sha256_file(path) for path in record_paths},
        "branch_cache_sha256": _sha256_file(cache_path),
        "states": len(ids),
        "train_states": int(train_state_mask.sum()),
        "dev_states": int(dev_state_mask.sum()),
        "branches": len(actions),
        "valid_transitions": int(valid_mask.sum()),
        "success_labels_used_for_training": False,
        "target_source": "future images encoded by this scorer checkpoint's frozen encoder/projector",
    }
    _atomic_json(output_dir / "branch_cache_identity.json", identity_payload)
    return payload


def _encode_training_pixels(model, pixel_sequences: torch.Tensor, transform: Any, device: torch.device) -> torch.Tensor:
    if pixel_sequences.ndim != 5:
        raise ValueError(f"expected offline pixels [B,T,C,H,W], got {tuple(pixel_sequences.shape)}")
    batch, steps = pixel_sequences.shape[:2]
    raw = pixel_sequences.reshape(batch * steps, *pixel_sequences.shape[-3:])
    if raw.dtype == torch.uint8:
        transformed = [transform(frame) for frame in raw]
    else:
        # Saved training samples are either raw uint8 or already CHW float
        # tensors.  For float samples, the stable training pipeline used the
        # same ImageNet transform; retain its values to avoid double scaling.
        transformed = [frame.float() for frame in raw]
    batch_pixels = torch.stack(transformed).to(device, non_blocking=True)
    with torch.inference_mode():
        if callable(getattr(model, "encode_pixels", None)):
            encoded = model.encode_pixels(batch_pixels)
        else:
            encoded = model.encode({"pixels": batch_pixels[:, None]})["emb"][:, 0]
    return encoded.float().reshape(batch, steps, -1).detach().cpu()


def _load_offline_cache(
    *,
    task: str,
    scorer: str,
    checkpoint: Path,
    branch_dir: Path,
    output_dir: Path,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    checkpoint_hash = _sha256_file(checkpoint)
    metadata_path = _run_dir(checkpoint) / "phase1_7_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Phase1.7 checkpoint metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    split_path = Path(metadata["episode_split_manifest"]).expanduser()
    if not split_path.is_absolute():
        split_path = (ROOT / split_path).resolve()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("task") != task:
        raise ValueError("offline continuation episode split task mismatch")
    train_ids = split.get("train_episode_ids")
    if not isinstance(train_ids, list) or not train_ids:
        raise ValueError("offline continuation requires the registered Phase1.7 train split")

    cfg, cfg_path = _read_training_config(checkpoint)
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    from source.common.data import load_dataset
    import stable_pretraining as spt

    dataset = load_dataset(
        dataset_name,
        cache_dir=os.environ.get("LOCAL_DATASET_DIR"),
        transform=None,
        **dataset_cfg,
    )
    # ``keys_to_load`` intentionally omits the HDF5 episode metadata column,
    # so it is absent from ``dataset.column_names`` even though it remains in
    # the file. ``clip_indices`` already carries the episode ID for every
    # eligible training window; filter those IDs directly against the locked
    # train split instead of making an unused all-episode metadata read.
    train_keys = {
        json.dumps(value.item() if isinstance(value, np.generic) else value, sort_keys=True)
        for value in train_ids
    }

    # Recreate the source run's train-only column transforms from its receipt.
    # Pixel transforms remain deferred to the same eval-time ImageNet path used
    # by the online scorer cache.
    from source.common.data import ZScoreNormalizer
    normalizers = metadata.get("normalizers", {})
    action_stats = normalizers.get("action")
    if not isinstance(action_stats, dict) or "mean" not in action_stats or "std" not in action_stats:
        raise ValueError("checkpoint metadata lacks the train-only action normalizer")
    mean = torch.as_tensor(action_stats["mean"], dtype=torch.float32).reshape(1, -1)
    std = torch.as_tensor(action_stats["std"], dtype=torch.float32).reshape(1, -1)
    std = torch.where(std > 0, std, torch.ones_like(std))
    dataset.transform = spt.data.transforms.Compose(
        spt.data.transforms.WrapTorchTransform(
            ZScoreNormalizer(mean, std), source="action", target="action"
        )
    )

    clips = getattr(dataset, "clip_indices", None)
    if clips is None:
        raise TypeError("offline dataset does not expose frame-skip training windows")
    train_key_set = set(train_keys)
    grouped: dict[str, list[int]] = {}
    episode_by_key: dict[str, Any] = {}
    for index, record in enumerate(clips):
        episode = record[0].item() if isinstance(record[0], np.generic) else record[0]
        key = json.dumps(episode, sort_keys=True)
        if key in train_key_set:
            grouped.setdefault(key, []).append(index)
            episode_by_key[key] = episode
    eligible = [(key, values) for key, values in grouped.items() if len(values) >= OFFLINE_WINDOWS_PER_TRAJECTORY]
    if len(eligible) < OFFLINE_TRAJECTORIES:
        raise ValueError(f"only {len(eligible)} train episodes have enough offline windows")
    rng = np.random.default_rng(int(seed))
    episode_order = rng.permutation(len(eligible))[:OFFLINE_TRAJECTORIES]
    selected_indices: list[int] = []
    selected_episodes = []
    for position in episode_order:
        key, values = eligible[int(position)]
        picked = rng.choice(len(values), size=OFFLINE_WINDOWS_PER_TRAJECTORY, replace=False)
        selected_indices.extend(values[int(i)] for i in picked)
        selected_episodes.append(episode_by_key[key])
    selected_indices = sorted(set(selected_indices))
    if len(selected_indices) != OFFLINE_TRAJECTORIES * OFFLINE_WINDOWS_PER_TRAJECTORY:
        raise RuntimeError("offline continuation windows repeated within the selected cohort")

    policy, model = _load_policy_model(checkpoint, device)
    del policy
    model_kind = _model_kind(model)
    if model_kind != scorer:
        raise ValueError(f"offline checkpoint architecture is {model_kind}, expected {scorer}")
    eval_cfg = compose_eval_config(
        task,
        overrides=["eval.num_eval=1", "world.num_envs=1", "output.save_video=false"],
    )
    transform = img_transform(eval_cfg)
    latent_batches = []
    action_batches = []
    batch_size = 64
    for start in range(0, len(selected_indices), batch_size):
        indices = selected_indices[start : start + batch_size]
        samples = [dataset[index] for index in indices]
        pixels = torch.stack([torch.as_tensor(sample["pixels"]) for sample in samples])
        actions = torch.stack([torch.as_tensor(sample["action"]).float() for sample in samples])
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        latent = _encode_training_pixels(model, pixels, transform, device)
        if actions.ndim != 3 or actions.shape[0] != len(indices) or actions.shape[-1] != 10:
            raise ValueError(f"offline action window is not [B,T,10]: {tuple(actions.shape)}")
        if latent.shape[:2] != actions.shape[:2]:
            raise ValueError("offline images and actions have different temporal axes")
        latent_batches.append(latent)
        action_batches.append(actions.float().cpu())
    latent = torch.cat(latent_batches).contiguous()
    actions = torch.cat(action_batches).contiguous()
    if scorer == "joint" and latent.shape[1] != HORIZON + 1:
        raise ValueError(f"Joint offline windows need 6 images, got {latent.shape[1]}")
    history_size = int(OmegaConf.select(cfg, "history_size", default=1))
    if scorer == "lewm" and latent.shape[1] != history_size + 1:
        raise ValueError(
            f"LeWM offline windows need history_size+1={history_size + 1} images, got {latent.shape[1]}"
        )

    offline_cache = {
        "schema_version": 1,
        "task": task,
        "scorer": scorer,
        "model_kind": model_kind,
        "base_checkpoint": str(checkpoint.resolve()),
        "base_checkpoint_sha256": checkpoint_hash,
        "source_config_sha256": _sha256_file(cfg_path),
        "episode_split_sha256": _sha256_file(split_path),
        "sampled_train_episode_ids": selected_episodes,
        "sampled_dataset_indices": selected_indices,
        "latent": latent,
        "action": actions,
        "history_size": history_size,
    }
    path = output_dir / "offline_latents.pt"
    _atomic_torch_save(path, offline_cache)
    _atomic_json(
        output_dir / "offline_cache_identity.json",
        {
            "task": task,
            "scorer": scorer,
            "base_checkpoint_sha256": checkpoint_hash,
            "episode_split_sha256": _sha256_file(split_path),
            "offline_cache_sha256": _sha256_file(path),
            "windows": len(latent),
            "latent_shape": list(latent.shape),
            "action_shape": list(actions.shape),
            "train_only_episode_sampling": True,
            "train_only_action_normalizer": True,
        },
    )
    return offline_cache


def prepare(args: argparse.Namespace) -> None:
    from scripts import round5_phase1_5_diagnostics as phase15

    if args.gpu is None:
        raise ValueError("cache preparation requires a physical GPU ID")
    if str(args.device).startswith("cuda"):
        device = phase15._configure_device(
            args.device,
            str(args.gpu),
            minimum_free_mib=int(args.min_free_mib),
            max_load_per_cpu=float(args.max_load_per_cpu),
            minimum_available_mib=int(args.min_available_mib),
            minimum_swap_free_mib=int(args.min_swap_free_mib),
        )
    else:
        device = torch.device(args.device)
    if args.scorer not in {"joint", "lewm"}:
        raise ValueError("scorer must be joint or lewm")
    checkpoint = args.scorer_checkpoint.expanduser().resolve()
    branch_dir = args.branch_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    branch_payload = _load_branch_cache(
        task=args.task,
        scorer=args.scorer,
        checkpoint=checkpoint,
        branch_dir=branch_dir,
        output_dir=output_dir,
        device=device,
    )
    del branch_payload
    offline_payload = _load_offline_cache(
        task=args.task,
        scorer=args.scorer,
        checkpoint=checkpoint,
        branch_dir=branch_dir,
        output_dir=output_dir,
        device=device,
        seed=int(args.seed),
    )
    del offline_payload
    print(
        json.dumps(
            {
                "status": "prepared",
                "task": args.task,
                "scorer": args.scorer,
                "output_dir": str(output_dir),
                "branch_cache": str(output_dir / "branch_latents.pt"),
                "offline_cache": str(output_dir / "offline_latents.pt"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _branch_predictions(model, model_kind: str, start: torch.Tensor, actions: torch.Tensor, history_size: int):
    if model_kind == "joint":
        timestep = torch.ones(len(start), device=start.device, dtype=start.dtype)
        return model(start, actions, timestep, mode="stage_b")["predicted_latents"]
    batch = len(start)
    history = start[:, None, :].expand(batch, history_size, start.shape[-1]).contiguous()
    past_actions = actions.new_zeros((batch, history_size - 1, actions.shape[-1]))
    full_actions = torch.cat((past_actions, actions), dim=1)
    predictions = []
    embeddings = history
    for step in range(HORIZON):
        action_window = full_actions[:, step : step + history_size]
        action_embeddings = model.action_encoder(action_window)
        predicted = model.predict(embeddings[:, -history_size:], action_embeddings)[:, -1:]
        predictions.append(predicted)
        embeddings = torch.cat((embeddings, predicted), dim=1)
    return torch.cat(predictions, dim=1)


def _offline_loss(model, model_kind: str, latent: torch.Tensor, action: torch.Tensor, history_size: int):
    if model_kind == "joint":
        prediction = _branch_predictions(model, model_kind, latent[:, 0], action[:, :HORIZON], 1)
        target = latent[:, 1 : HORIZON + 1]
    else:
        act_emb = model.action_encoder(action)
        prediction = model.predict(latent[:, :history_size], act_emb[:, :history_size])
        target = latent[:, 1:]
    if prediction.shape != target.shape:
        raise ValueError(f"offline predictor/target shapes differ: {prediction.shape} vs {target.shape}")
    return (prediction.float() - target.float()).square().mean()


def _freeze_representation(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    frozen = {}
    for name, parameter in model.named_parameters():
        should_freeze = name.startswith(("encoder.", "projector."))
        parameter.requires_grad_(not should_freeze)
        if should_freeze:
            frozen[name] = parameter.detach().cpu().clone()
    for name, buffer in model.named_buffers():
        if name.startswith(("encoder.", "projector.")):
            frozen[f"buffer::{name}"] = buffer.detach().cpu().clone()
    if not frozen:
        raise RuntimeError("model has no image representation to freeze")
    return frozen


def train(args: argparse.Namespace) -> None:
    if args.gpu is None or int(args.gpu) not in range(4):
        raise ValueError("online scorer training is restricted to GPU0-3")
    if args.arm not in ARMS:
        raise ValueError(f"arm must be one of {sorted(ARMS)}")
    if args.updates < 1 or args.batch_size != BATCH_SIZE:
        raise ValueError("Phase1.7 online training is locked to batch 128 and positive updates")
    from scripts import round5_phase1_5_diagnostics as phase15

    device = phase15._configure_device(
        args.device,
        str(args.gpu),
        minimum_free_mib=max(
            int(args.min_free_mib),
            int((args.minimum_free_after_reserve_gib + args.estimated_peak_gib) * 1024),
        ),
        max_load_per_cpu=float(args.max_load_per_cpu),
        minimum_available_mib=int(args.min_available_mib),
        minimum_swap_free_mib=int(args.min_swap_free_mib),
    )
    if torch.cuda.is_available() and torch.cuda.mem_get_info(device)[0] < int(args.minimum_free_after_reserve_gib * 1024**3):
        raise RuntimeError("GPU does not retain the required training VRAM safety margin")
    checkpoint = args.scorer_checkpoint.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    branch_path = cache_dir / "branch_latents.pt"
    offline_path = cache_dir / "offline_latents.pt"
    if not branch_path.is_file() or not offline_path.is_file():
        raise FileNotFoundError("prepare branch and offline caches before training")
    branch_cache = torch_load_compat(branch_path, map_location="cpu")
    offline_cache = torch_load_compat(offline_path, map_location="cpu")
    checkpoint_hash = _sha256_file(checkpoint)
    if branch_cache["base_checkpoint_sha256"] != checkpoint_hash or offline_cache["base_checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("prepared cache belongs to another base checkpoint")
    if branch_cache["task"] != args.task or branch_cache["scorer"] != args.scorer:
        raise ValueError("branch cache identity does not match requested task/scorer")

    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    policy, model = _load_policy_model(checkpoint, device)
    model_kind = _model_kind(model)
    if model_kind != args.scorer:
        raise ValueError(f"requested scorer {args.scorer}, checkpoint is {model_kind}")
    cfg, _ = _read_training_config(checkpoint)
    history_size = int(OmegaConf.select(cfg, "history_size", default=1))
    frozen_reference = _freeze_representation(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "latest.pt"
    start_update = 0
    records: list[dict[str, Any]] = []
    if args.resume and state_path.is_file():
        resume = torch_load_compat(state_path, map_location="cpu")
        expected = (checkpoint_hash, branch_cache["candidate_pool_sha256"], args.arm)
        actual = (resume.get("base_checkpoint_sha256"), resume.get("candidate_pool_sha256"), resume.get("arm"))
        if actual != expected:
            raise ValueError("online training resume identity mismatch")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_update = int(resume["update"])
        records = list(resume.get("log", []))
        torch.set_rng_state(resume["torch_rng_state"])
        if device.type == "cuda" and resume.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(resume["cuda_rng_state"], device=device)

    # Immutable, paired sample schedules are prepared once and reused by the
    # branch-prediction and ranking arms.
    schedule_path = cache_dir / "training_schedule.npz"
    schedule_identity_path = cache_dir / "training_schedule.json"
    schedule_seed = int(args.seed)
    if schedule_path.is_file() and schedule_identity_path.is_file():
        schedule_identity = json.loads(schedule_identity_path.read_text(encoding="utf-8"))
        if schedule_identity.get("candidate_pool_sha256") != branch_cache["candidate_pool_sha256"]:
            raise ValueError("online update schedule belongs to another candidate pool")
        with np.load(schedule_path, allow_pickle=False) as archive:
            schedule = {key: np.asarray(archive[key]) for key in archive.files}
    else:
        if args.arm != "offline":
            raise FileNotFoundError("create the shared online/offline update schedule first")
        # The offline arm can bootstrap the deterministic common schedule; it
        # uses the same update budget but samples only offline train windows.
        schedule = {}
        schedule_identity = {}

    if args.arm == "offline" and not schedule:
        rng = np.random.default_rng(schedule_seed)
        offline_indices = rng.integers(
            0, len(offline_cache["latent"]), size=(int(args.updates), BATCH_SIZE), dtype=np.int64
        )
        train_states = np.flatnonzero(np.asarray(branch_cache["state_split"]) == "train")
        branch_rows = np.zeros((int(args.updates), BATCH_SIZE), dtype=np.int64)
        pair_left_pos = np.zeros((int(args.updates), BATCH_SIZE // 2), dtype=np.int16)
        pair_right_pos = np.zeros_like(pair_left_pos)
        pair_horizon = np.zeros_like(pair_left_pos)
        pair_sign = np.zeros_like(pair_left_pos, dtype=np.float32)
        candidate_valid = branch_cache["valid_mask"].numpy().reshape(256, 32, HORIZON).any(axis=-1)
        # `branch_rows` and pair labels use a flattened state-major index
        # (state * 32 + candidate), so keep the latent arrays in that layout.
        future = branch_cache["future_latent"].numpy()
        goals = branch_cache["goal_latent"].numpy()
        for update in range(int(args.updates)):
            chosen_states = rng.choice(train_states, size=32, replace=False)
            for group, state in enumerate(chosen_states):
                candidates = np.flatnonzero(candidate_valid[int(state)])
                if len(candidates) < 1:
                    raise ValueError(f"online state {state} has no valid future milestone")
                picked = rng.choice(candidates, size=4, replace=len(candidates) < 4)
                local_positions = np.arange(group * 4, group * 4 + 4)
                branch_rows[update, local_positions] = int(state) * 32 + picked
                for pair_index, (left_in_group, right_in_group) in enumerate(((0, 1), (2, 3))):
                    pair_slot = group * 2 + pair_index
                    left_candidate = int(picked[left_in_group])
                    right_candidate = int(picked[right_in_group])
                    left_global = int(state) * 32 + left_candidate
                    right_global = int(state) * 32 + right_candidate
                    pair_left_pos[update, pair_slot] = local_positions[left_in_group]
                    pair_right_pos[update, pair_slot] = local_positions[right_in_group]
                    common = np.flatnonzero(
                        branch_cache["valid_mask"][left_global].numpy()
                        & branch_cache["valid_mask"][right_global].numpy()
                    )
                    if left_candidate == right_candidate or len(common) == 0:
                        pair_sign[update, pair_slot] = 0.0
                        continue
                    h = int(common[-1])
                    target_left = float(np.mean(np.square(future[left_global, h] - goals[left_global])))
                    target_right = float(np.mean(np.square(future[right_global, h] - goals[right_global])))
                    difference = target_right - target_left
                    pair_horizon[update, pair_slot] = h
                    pair_sign[update, pair_slot] = 0.0 if abs(difference) <= 1e-8 else float(np.sign(difference))
        schedule = {
            "offline_indices": offline_indices,
            "branch_rows": branch_rows,
            "pair_left_pos": pair_left_pos,
            "pair_right_pos": pair_right_pos,
            "pair_horizon": pair_horizon,
            "pair_sign": pair_sign,
        }
        temporary = schedule_path.with_name(f".{schedule_path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **schedule)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, schedule_path)
        schedule_identity = {
            "seed": schedule_seed,
            "updates": int(args.updates),
            "batch_size": BATCH_SIZE,
            "candidate_pool_sha256": branch_cache["candidate_pool_sha256"],
            "offline_cache_sha256": _sha256_file(offline_path),
            "schedule_sha256": _sha256_file(schedule_path),
            "branch_update_layout": "32 train states x 4 candidate branches; 2 same-state pairs/state/update",
        }
        _atomic_json(schedule_identity_path, schedule_identity)

    if args.arm == "offline":
        if "offline_indices" not in schedule:
            raise ValueError("shared schedule is missing offline sample rows")
    else:
        required = {"branch_rows", "pair_left_pos", "pair_right_pos", "pair_horizon", "pair_sign"}
        if not required.issubset(schedule):
            raise ValueError("shared schedule is missing online branch row/pair indices")

    branch_tensors = {
        key: value
        for key, value in branch_cache.items()
        if torch.is_tensor(value)
    }
    offline_latent = offline_cache["latent"]
    offline_action = offline_cache["action"]
    update_count = min(int(args.updates), len(schedule.get("offline_indices", schedule.get("branch_rows", []))))
    if update_count < int(args.updates):
        raise ValueError("shared update schedule is shorter than the requested training run")

    rank_weight = {"offline": 0.0, "branches": 0.0, "rank_0.1": 0.1, "rank_1.0": 1.0}[args.arm]
    for update in range(start_update, int(args.updates)):
        model.train()
        model.encoder.eval()
        model.projector.eval()
        optimizer.zero_grad(set_to_none=True)
        if args.arm == "offline":
            ids = schedule["offline_indices"][update]
            latent = offline_latent[ids].to(device, non_blocking=True)
            action = offline_action[ids].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction_loss = _offline_loss(model, model_kind, latent, action, history_size)
            rank_loss = prediction_loss.new_zeros(())
        else:
            rows = schedule["branch_rows"][update]
            row_index = torch.as_tensor(rows, dtype=torch.long)
            start = branch_tensors["start_latent"][row_index].to(device, non_blocking=True)
            actions = branch_tensors["actions"][row_index].to(device, non_blocking=True)
            targets = branch_tensors["future_latent"][row_index].to(device, non_blocking=True)
            mask = branch_tensors["valid_mask"][row_index].to(device, non_blocking=True)
            goals = branch_tensors["goal_latent"][row_index].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                predicted = _branch_predictions(model, model_kind, start, actions, history_size)
                per_transition = (predicted.float() - targets.float()).square().mean(dim=-1)
                prediction_loss = (per_transition * mask.float()).sum() / mask.sum().clamp_min(1)
                rank_loss = prediction_loss.new_zeros(())
                if rank_weight > 0.0:
                    left_pos = torch.as_tensor(schedule["pair_left_pos"][update], device=device, dtype=torch.long)
                    right_pos = torch.as_tensor(schedule["pair_right_pos"][update], device=device, dtype=torch.long)
                    horizons = torch.as_tensor(schedule["pair_horizon"][update], device=device, dtype=torch.long)
                    signs = torch.as_tensor(schedule["pair_sign"][update], device=device, dtype=torch.float32)
                    active = signs != 0.0
                    if bool(active.any()):
                        h = horizons[active]
                        left = left_pos[active]
                        right = right_pos[active]
                        batch_rows = torch.arange(len(h), device=device)
                        cost_left = (predicted[left, h].float() - goals[left].float()).square().mean(dim=-1)
                        cost_right = (predicted[right, h].float() - goals[right].float()).square().mean(dim=-1)
                        rank_loss = F.softplus(-signs[active] * (cost_right - cost_left)).mean()
            loss = prediction_loss + rank_weight * rank_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            completed = update + 1
            if completed % 50 == 0 or completed == int(args.updates):
                row = {
                    "update": completed,
                    "arm": args.arm,
                    "loss": float(loss.detach().float().cpu()),
                    "prediction_loss": float(prediction_loss.detach().float().cpu()),
                    "ranking_loss": float(rank_loss.detach().float().cpu()),
                }
                records.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
            if completed % 500 == 0 or completed == int(args.updates):
                state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "update": completed,
                    "arm": args.arm,
                    "task": args.task,
                    "scorer": args.scorer,
                    "base_checkpoint_sha256": checkpoint_hash,
                    "candidate_pool_sha256": branch_cache["candidate_pool_sha256"],
                    "branch_cache_sha256": _sha256_file(branch_path),
                    "offline_cache_sha256": _sha256_file(offline_path),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                    "log": list(records),
                }
                _atomic_torch_save(state_path, state)
                _atomic_torch_save(output_dir / f"step_{completed}.pt", state)
        # End branch arm; offline control applies the same optimizer update and
        # checkpoint cadence with only the supervised prediction objective.
        if args.arm == "offline":
            loss = prediction_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            completed = update + 1
            if completed % 50 == 0 or completed == int(args.updates):
                row = {
                    "update": completed,
                    "arm": args.arm,
                    "loss": float(loss.detach().float().cpu()),
                    "prediction_loss": float(loss.detach().float().cpu()),
                    "ranking_loss": 0.0,
                }
                records.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
            if completed % 500 == 0 or completed == int(args.updates):
                state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "update": completed,
                    "arm": args.arm,
                    "task": args.task,
                    "scorer": args.scorer,
                    "base_checkpoint_sha256": checkpoint_hash,
                    "candidate_pool_sha256": branch_cache["candidate_pool_sha256"],
                    "branch_cache_sha256": _sha256_file(branch_path),
                    "offline_cache_sha256": _sha256_file(offline_path),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                    "log": list(records),
                }
                _atomic_torch_save(state_path, state)
                _atomic_torch_save(output_dir / f"step_{completed}.pt", state)

    current = model.state_dict()
    changed_frozen = []
    for name, expected in frozen_reference.items():
        key = name.split("buffer::", 1)[-1] if name.startswith("buffer::") else name
        if not torch.equal(expected, current[key].detach().cpu()):
            changed_frozen.append(name)
    if changed_frozen:
        raise RuntimeError(f"image representation changed during scorer tuning: {changed_frozen}")

    policy.eval()
    policy.to("cpu")
    run_dir = output_dir.resolve()
    base_run_dir = _run_dir(checkpoint)
    config_source = base_run_dir / "config.yaml"
    metadata_source = base_run_dir / "phase1_7_metadata.json"
    if not config_source.is_file() or not metadata_source.is_file():
        raise FileNotFoundError("base scorer config and Phase1.7 metadata are required for evaluation")
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_source, run_dir / "config.yaml")
    metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
    metadata["online_finetune"] = {
        "arm": args.arm,
        "base_checkpoint": str(checkpoint),
        "base_checkpoint_sha256": checkpoint_hash,
        "branch_cache_sha256": _sha256_file(branch_path),
        "offline_cache_sha256": _sha256_file(offline_path),
        "candidate_pool_sha256": branch_cache["candidate_pool_sha256"],
        "updates": int(args.updates),
        "batch_size": BATCH_SIZE,
        "rank_loss_weight": rank_weight,
        "representation_frozen_bitwise": True,
        "success_labels_used_for_training": False,
        "physical_distances_used_for_training": False,
    }
    _atomic_json(run_dir / "phase1_7_metadata.json", metadata)
    final_checkpoint = run_dir / "checkpoints" / f"{args.arm}_policy.ckpt"
    _atomic_torch_save(final_checkpoint, policy)
    result = {
        "status": "completed",
        "task": args.task,
        "scorer": args.scorer,
        "arm": args.arm,
        "rank_loss_weight": rank_weight,
        "updates": int(args.updates),
        "batch_size": BATCH_SIZE,
        "learning_rate": float(args.learning_rate),
        "checkpoint": str(final_checkpoint),
        "checkpoint_sha256": _sha256_file(final_checkpoint),
        "base_checkpoint": str(checkpoint),
        "base_checkpoint_sha256": checkpoint_hash,
        "branch_cache_sha256": _sha256_file(branch_path),
        "offline_cache_sha256": _sha256_file(offline_path),
        "candidate_pool_sha256": branch_cache["candidate_pool_sha256"],
        "representation_frozen_bitwise": True,
        "success_labels_used_for_training": False,
        "log": records,
    }
    _atomic_json(output_dir / "training_result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="encode online branches and matching offline windows")
    train_parser = commands.add_parser("train", help="run one 2,000-update scorer arm")
    for child in (prepare_parser, train_parser):
        child.add_argument("--task", required=True, choices=("pusht", "reacher"))
        child.add_argument("--scorer", required=True, choices=("joint", "lewm"))
        child.add_argument("--scorer-checkpoint", required=True)
        child.add_argument("--gpu", required=True, type=int, choices=range(8))
        child.add_argument("--device", default="cuda")
        child.add_argument("--min-free-mib", type=int, default=2048)
        child.add_argument("--max-load-per-cpu", type=float, default=0.75)
        child.add_argument("--min-available-mib", type=int, default=4096)
        child.add_argument("--min-swap-free-mib", type=int, default=2048)
    prepare_parser.add_argument("--branch-dir", required=True)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--seed", type=int, default=16047)
    train_parser.add_argument("--cache-dir", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    train_parser.add_argument("--updates", type=int, default=UPDATE_COUNT)
    train_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    train_parser.add_argument("--learning-rate", type=float, default=1e-5)
    train_parser.add_argument("--weight-decay", type=float, default=1e-3)
    train_parser.add_argument("--minimum-free-after-reserve-gib", type=float, default=6.0)
    train_parser.add_argument("--estimated-peak-gib", type=float, default=6.0)
    train_parser.add_argument("--seed", type=int, default=16047)
    train_parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.scorer_checkpoint = _resolve_path(args.scorer_checkpoint)
    if args.command == "prepare":
        args.branch_dir = _resolve_path(args.branch_dir)
        args.output_dir = _resolve_path(args.output_dir)
        prepare(args)
    else:
        args.cache_dir = _resolve_path(args.cache_dir)
        args.output_dir = _resolve_path(args.output_dir)
        train(args)


if __name__ == "__main__":
    main()
