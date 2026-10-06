"""Frozen-latent short-training interventions for Round 5 Phase1.6."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from .checkpoint import load_initial_model_state, torch_load_compat
from .data import get_column_normalizer, get_img_preprocessor, load_dataset
from .round5_phase1_6 import atomic_json, rng_for, stable_hash


ARMS = ("continue", "detach", "recorded")
FROZEN_PREFIXES = (
    "encoder.",
    "projector.",
    "d_positions",
    "d_velocity_head.",
    "e_state_positions",
    "e_query_positions",
    "e_query_content",
    "e_adjacent_input.",
    "inverse_dynamics_head.",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_key(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _training_run_config(checkpoint: Path) -> tuple[OmegaConf, Path]:
    run_dir = checkpoint.parent.parent
    path = run_dir / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"source training config is missing: {path}")
    return OmegaConf.load(path), path


def _make_training_dataset(cfg: Any):
    import stable_pretraining as spt

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    dataset = load_dataset(
        dataset_name,
        cache_dir=os.environ.get("LOCAL_DATASET_DIR"),
        **dataset_cfg,
    )
    transforms = []
    pipeline = cfg.get("data_pipeline", {})
    if not bool(pipeline.get("gpu_image_preprocessing", False)):
        transforms.append(
            get_img_preprocessor(source="pixels", target="pixels", img_size=int(cfg.img_size))
        )
    for column in cfg.data.dataset.keys_to_load:
        if str(column).startswith("pixels"):
            continue
        transforms.append(get_column_normalizer(dataset, str(column), str(column)))
    dataset.transform = spt.data.transforms.Compose(*transforms)
    return dataset


def _eligible_windows(
    dataset: Any,
    *,
    excluded_episode_ids: Sequence[Any],
    seed: int,
    trajectories: int,
    windows_per_trajectory: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    clips = getattr(dataset, "clip_indices", None)
    if clips is None:
        raise TypeError("training dataset must expose clip_indices")
    excluded = {_episode_key(value) for value in excluded_episode_ids}
    grouped: dict[str, list[tuple[int, int, Any]]] = {}
    for data_index, record in enumerate(clips):
        if len(record) < 2:
            raise ValueError("clip_indices entries must be (episode_id, local_start)")
        episode_id = record[0].item() if isinstance(record[0], np.generic) else record[0]
        start = int(record[1])
        key = _episode_key(episode_id)
        if key in excluded:
            continue
        grouped.setdefault(key, []).append((int(data_index), start, episode_id))

    eligible = [
        (key, values)
        for key, values in grouped.items()
        if len(values) >= int(windows_per_trajectory)
    ]
    rng = rng_for("round5_phase1_6_train_windows", int(seed))
    order = rng.permutation(len(eligible))
    if len(order) < int(trajectories):
        raise ValueError(
            f"only {len(order)} eligible training trajectories remain after exclusions; "
            f"need {trajectories}"
        )
    selected: list[tuple[int, int, Any]] = []
    episode_records = []
    for position in order[: int(trajectories)]:
        key, values = eligible[int(position)]
        local_rng = rng_for("round5_phase1_6_train_windows_per_episode", int(seed), key)
        picked = local_rng.choice(len(values), size=int(windows_per_trajectory), replace=False)
        chosen = [values[int(index)] for index in picked]
        selected.extend(chosen)
        episode_records.append({
            "episode_id": values[0][2],
            "available_windows": len(values),
            "selected_starts": sorted(item[1] for item in chosen),
        })
    selected.sort(key=lambda item: (_episode_key(item[2]), item[1], item[0]))
    return [item[0] for item in selected], episode_records


def _load_model(checkpoint: Path, device: torch.device):
    cfg, _ = _training_run_config(checkpoint)
    model = hydra.utils.instantiate(cfg.policy.model)
    state = load_initial_model_state(checkpoint)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict checkpoint load failed: {incompatible}")
    return model.to(device), cfg


def _extract_cache(
    *,
    task: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    excluded_episode_ids: Sequence[Any],
    output_dir: Path,
    device: torch.device,
    seed: int,
    trajectories: int,
    windows_per_trajectory: int,
    cache_batch_size: int = 64,
) -> Path:
    cache_path = output_dir / "frozen_latent_cache.pt"
    identity_path = output_dir / "cache_identity.json"
    requested_identity = {
        "task": task,
        "checkpoint_sha256": checkpoint_sha256,
        "seed": int(seed),
        "trajectories": int(trajectories),
        "windows_per_trajectory": int(windows_per_trajectory),
        "excluded_episode_ids_sha256": stable_hash(sorted(_episode_key(value) for value in excluded_episode_ids)),
    }
    if cache_path.is_file() and identity_path.is_file():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if (
            all(identity.get(key) == value for key, value in requested_identity.items())
            and identity.get("cache_sha256") == _sha256(cache_path)
        ):
            return cache_path

    training_cfg, _ = _training_run_config(checkpoint)
    dataset = _make_training_dataset(training_cfg)
    indices, episode_records = _eligible_windows(
        dataset,
        excluded_episode_ids=excluded_episode_ids,
        seed=seed,
        trajectories=trajectories,
        windows_per_trajectory=windows_per_trajectory,
    )
    model, _ = _load_model(checkpoint, device)
    model.eval()
    model.requires_grad_(False)
    latents: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    replaced_action_values = 0
    episode_ids: list[Any] = []
    starts: list[int] = []
    dataset_indices: list[int] = []
    for offset in range(0, len(indices), int(cache_batch_size)):
        batch_indices = indices[offset : offset + int(cache_batch_size)]
        samples = [dataset[index] for index in batch_indices]
        pixels = torch.stack([torch.as_tensor(sample["pixels"]) for sample in samples])
        raw_action = torch.stack([torch.as_tensor(sample["action"]).float() for sample in samples])
        replaced_action_values += int((~torch.isfinite(raw_action)).sum().item())
        # The source training policy applies torch.nan_to_num(batch["action"], 0.0)
        # before forming its action targets. Dataset z-score normalization can
        # produce NaNs for constant action columns, so cache the same effective
        # training input instead of propagating those NaNs into the loss.
        action = torch.nan_to_num(raw_action, nan=0.0)
        if pixels.ndim != 5 or tuple(pixels.shape[1:3]) != (6, 3):
            raise ValueError(f"expected cached pixels [B,6,3,H,W], got {tuple(pixels.shape)}")
        if tuple(action.shape[1:]) != (6, int(model.action_dim)):
            raise ValueError(f"expected cached actions [B,6,{model.action_dim}], got {tuple(action.shape)}")
        # The cached latents feed the FP32 intervention runs and must match
        # the frozen inference path. BF16 autocast produced batch-shape
        # dependent quantization differences large enough to fail the online
        # cache check, so keep this one-time encode in the checkpoint's native
        # precision.
        with torch.inference_mode():
            encoded = model.encode_pixels(pixels.to(device, non_blocking=True)).float().cpu()
        latents.append(encoded)
        actions.append(action.cpu().float())
        for data_index in batch_indices:
            episode_id, local_start = dataset.clip_indices[int(data_index)][:2]
            episode_ids.append(episode_id.item() if isinstance(episode_id, np.generic) else episode_id)
            starts.append(int(local_start))
            dataset_indices.append(int(data_index))

    latent_tensor = torch.cat(latents, dim=0).contiguous()
    action_tensor = torch.cat(actions, dim=0).contiguous()
    if len(latent_tensor) != trajectories * windows_per_trajectory:
        raise RuntimeError("training latent cache does not match the frozen window budget")
    if not torch.isfinite(action_tensor).all():
        raise RuntimeError("training action cache still contains non-finite values after source-equivalent preprocessing")

    # Verify that caching preserved the model's eval-mode image path and the
    # source training action transform on a fixed sample.
    # Match the first encoding batch shape. The ViT backend can select a
    # different batched kernel when re-encoding only a small sample, creating
    # avoidable numerical differences even in FP32.
    verification_count = min(int(cache_batch_size), len(indices))
    verify_samples = [dataset[index] for index in indices[:verification_count]]
    verify_pixels = torch.stack([torch.as_tensor(sample["pixels"]) for sample in verify_samples])
    with torch.inference_mode():
        online = model.encode_pixels(verify_pixels.to(device)).float().cpu()
    torch.testing.assert_close(online, latent_tensor[:verification_count], rtol=1e-4, atol=1e-5)
    verify_actions = torch.stack([torch.as_tensor(sample["action"]).float() for sample in verify_samples])
    verify_actions = torch.nan_to_num(verify_actions, nan=0.0)
    torch.testing.assert_close(verify_actions, action_tensor[:verification_count], rtol=0.0, atol=0.0)

    payload = {
        "task": task,
        "latent": latent_tensor,
        "action": action_tensor,
        "episode_id": episode_ids,
        "local_start": torch.as_tensor(starts, dtype=torch.int64),
        "dataset_index": torch.as_tensor(dataset_indices, dtype=torch.int64),
        "checkpoint_sha256": checkpoint_sha256,
        "sampled_trajectories": episode_records,
        "latent_encode_verification": {"samples": verification_count, "rtol": 1e-4, "atol": 1e-5},
        "action_transform_verification": {"samples": verification_count, "rtol": 0.0, "atol": 0.0},
        "action_nan_to_num": {
            "matches_source_training_preprocess": True,
            "nan_value": 0.0,
            "nonfinite_values_replaced": replaced_action_values,
        },
    }
    _atomic_torch_save(cache_path, payload)
    identity = {
        **requested_identity,
        "task": task,
        "cache_sha256": _sha256(cache_path),
        "windows": len(indices),
        "excluded_episode_count": len({_episode_key(value) for value in excluded_episode_ids}),
        "latent_encode_verification": payload["latent_encode_verification"],
        "action_transform_verification": payload["action_transform_verification"],
    }
    atomic_json(identity_path, identity)
    return cache_path


def _make_schedule(path: Path, *, task: str, seed: int, num_windows: int, updates: int, batch_size: int, horizon: int, action_dim: int) -> Path:
    identity_path = path.with_suffix(".json")
    requested = {
        "task": task,
        "seed": int(seed),
        "num_windows": int(num_windows),
        "updates": int(updates),
        "batch_size": int(batch_size),
        "horizon": int(horizon),
        "action_dim": int(action_dim),
    }
    if path.is_file():
        if not identity_path.is_file():
            raise ValueError(f"schedule is missing its identity receipt: {identity_path}")
        recorded = json.loads(identity_path.read_text(encoding="utf-8"))
        if any(recorded.get(key) != value for key, value in requested.items()):
            raise ValueError("existing short-train schedule does not match requested design")
        if recorded.get("schedule_sha256") != _sha256(path):
            raise ValueError("existing short-train schedule hash does not match its receipt")
        return path
    rng = rng_for("round5_phase1_6_shared_training_schedule", task, int(seed))
    arrays = {
        "batch_indices": rng.integers(0, num_windows, size=(updates, batch_size), dtype=np.int64),
        "timestep": rng.random(size=(updates, batch_size), dtype=np.float32),
        "noise": rng.standard_normal(size=(updates, batch_size, horizon, action_dim), dtype=np.float32),
        "use_prediction": rng.random(size=(updates, batch_size)) < 0.5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    atomic_json(identity_path, {**requested, "schedule_sha256": _sha256(path)})
    return path


def _matches_frozen(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in FROZEN_PREFIXES)


def _freeze_model(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    frozen: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if _matches_frozen(name):
            parameter.requires_grad_(False)
            frozen[name] = parameter.detach().cpu().clone()
        else:
            parameter.requires_grad_(True)
    for name, buffer in model.named_buffers():
        if name.startswith(("encoder.", "projector.")):
            frozen[f"buffer::{name}"] = buffer.detach().cpu().clone()
    if not hasattr(model, "mode_embedding"):
        raise TypeError("short-train checkpoint must be Round4FastLeWAM with mode_embedding")
    return frozen, {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if name.startswith(("d_", "e_", "inverse_dynamics_head."))}


def _loss_step(model, latent, action, timestep, noise, use_prediction, arm: str):
    horizon = int(model.action_horizon)
    clean_actions = action[:, :horizon]
    z0 = latent[:, 0]
    goal = latent[:, horizon]
    targets = latent[:, 1 : horizon + 1]
    noise = noise.to(device=clean_actions.device, dtype=clean_actions.dtype)
    timestep = timestep.to(device=clean_actions.device, dtype=clean_actions.dtype)
    use_prediction = use_prediction.to(device=clean_actions.device, dtype=torch.bool)
    noisy = (1.0 - timestep[:, None, None]) * noise + timestep[:, None, None] * clean_actions
    target_velocity = clean_actions - noise
    a_output = model(z0, noisy, timestep, mode="stage_a", goal_latent=goal)
    velocity = a_output["action_velocity"]
    clean_estimate = noisy + (1.0 - timestep[:, None, None]) * velocity
    if arm == "continue":
        predicted = clean_estimate
        b_actions = torch.where(use_prediction[:, None, None], predicted, clean_actions)
    elif arm == "detach":
        predicted = clean_estimate.detach()
        b_actions = torch.where(use_prediction[:, None, None], predicted, clean_actions)
    elif arm == "recorded":
        b_actions = clean_actions
    else:
        raise ValueError(f"unknown training arm {arm!r}")
    source_t = torch.where(use_prediction, timestep, torch.ones_like(timestep))
    b_output = model(z0, b_actions, source_t, mode="stage_b")
    predicted_latents = b_output["predicted_latents"]
    per_sample = (predicted_latents.float() - targets.float()).square().mean(dim=(1, 2))
    noise_weight = (source_t / 0.2).clamp(max=1.0)
    latent_loss = (noise_weight * per_sample).mean()
    action_loss = F.mse_loss(velocity.float(), target_velocity.float())
    return action_loss + latent_loss, action_loss, latent_loss, clean_estimate


def _check_gradient_paths(model, latent, action, device: torch.device) -> dict[str, float]:
    model.train()
    model.encoder.eval()
    model.projector.eval()
    count = min(4, len(latent))
    z = latent[:count].to(device)
    a = action[:count].to(device)
    t = torch.full((count,), 0.35, dtype=torch.float32, device=device)
    noise = torch.linspace(-1.0, 1.0, count * model.action_horizon * model.action_dim, device=device).reshape(count, model.action_horizon, model.action_dim)
    mask = torch.ones(count, dtype=torch.bool, device=device)
    norms = {}
    for arm in ARMS:
        model.zero_grad(set_to_none=True)
        _, _, latent_loss, clean = _loss_step(model, z, a, t, noise, mask, arm)
        if clean.requires_grad:
            clean.retain_grad()
        latent_loss.backward()
        grad = 0.0 if clean.grad is None else float(clean.grad.detach().float().norm().cpu())
        norms[arm] = grad
    model.zero_grad(set_to_none=True)
    if norms["continue"] <= 0.0 or norms["detach"] != 0.0 or norms["recorded"] != 0.0:
        raise RuntimeError(f"action-gradient intervention contract failed: {norms}")
    return norms


def prepare_short_train_cache(
    *, task: str, checkpoint: str | Path, checkpoint_sha256: str,
    excluded_episode_ids: Sequence[Any], output_dir: str | Path,
    device: str, seed: int, trajectories: int = 512,
    windows_per_trajectory: int = 16,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    output = Path(output_dir)
    target = _extract_cache(
        task=task,
        checkpoint=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        excluded_episode_ids=excluded_episode_ids,
        output_dir=output,
        device=torch.device(device),
        seed=seed,
        trajectories=trajectories,
        windows_per_trajectory=windows_per_trajectory,
    )
    cache_payload = torch_load_compat(target, map_location="cpu")
    schedule = _make_schedule(
        output / "shared_schedule.npz",
        task=task,
        seed=seed,
        num_windows=len(cache_payload["latent"]),
        updates=1000,
        batch_size=128,
        horizon=int(cache_payload["action"].shape[1]) - 1,
        action_dim=int(cache_payload["action"].shape[2]),
    )
    report = {
        "task": task,
        "cache": str(target),
        "cache_sha256": _sha256(target),
        "schedule": str(schedule),
        "schedule_sha256": _sha256(schedule),
        "windows": len(cache_payload["latent"]),
        "latent_shape": list(cache_payload["latent"].shape),
        "action_shape": list(cache_payload["action"].shape),
        "checkpoint_sha256": checkpoint_sha256,
        "shared_across_arms": True,
    }
    atomic_json(output / "prepare_receipt.json", report)
    return report


def run_short_train(
    *, task: str, arm: str, checkpoint: str | Path, checkpoint_sha256: str,
    cache_dir: str | Path, output_dir: str | Path, device: str,
    updates: int = 1000, batch_size: int = 128, learning_rate: float = 1e-5,
    weight_decay: float = 1e-3, gradient_clip: float = 1.0,
    resume: bool = False,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    device_obj = torch.device(device)
    root = Path(cache_dir)
    cache_path = root / "frozen_latent_cache.pt"
    schedule_path = root / "shared_schedule.npz"
    if not cache_path.is_file() or not schedule_path.is_file():
        raise FileNotFoundError("run prepare-short-train-cache before training")
    cache_hash = _sha256(cache_path)
    schedule_hash = _sha256(schedule_path)
    cache = torch_load_compat(cache_path, map_location="cpu")
    with np.load(schedule_path) as schedule_file:
        schedule = {key: schedule_file[key] for key in schedule_file.files}
    if len(schedule["batch_indices"]) < int(updates):
        raise ValueError("shared training schedule has fewer updates than requested")

    model, _ = _load_model(Path(checkpoint), device_obj)
    model.train()
    model.encoder.eval()
    model.projector.eval()
    frozen_state, de_state = _freeze_model(model)
    mode_embedding_de_reference = model.mode_embedding.weight[2:].detach().cpu().clone()
    params = []
    mode_embedding = model.mode_embedding.weight
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = {"params": [parameter]}
        if name == "mode_embedding.weight":
            group["weight_decay"] = 0.0
        else:
            group["weight_decay"] = float(weight_decay)
        params.append(group)
    optimizer = torch.optim.AdamW(params, lr=float(learning_rate))
    start = 0
    logs: list[dict[str, Any]] = []
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / "step_1000.pt"
    resume_path = output / "latest.pt"
    if resume and resume_path.is_file():
        state = torch_load_compat(resume_path, map_location="cpu")
        expected = (cache_hash, schedule_hash, checkpoint_sha256, arm)
        actual = (state.get("cache_sha256"), state.get("schedule_sha256"), state.get("base_checkpoint_sha256"), state.get("arm"))
        if actual != expected:
            raise ValueError(f"resume identity mismatch: {actual} != {expected}")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start = int(state["update"])
        logs = list(state.get("log", []))
        torch.set_rng_state(state["torch_rng_state"])
        if device_obj.type == "cuda" and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(state["cuda_rng_state"], device=device_obj)
    else:
        state_dict = load_initial_model_state(checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.train()
        model.encoder.eval()
        model.projector.eval()

    if start == 0:
        initial_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "update": 0,
            "arm": arm,
            "task": task,
            "cache_sha256": cache_hash,
            "schedule_sha256": schedule_hash,
            "base_checkpoint_sha256": checkpoint_sha256,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device_obj) if device_obj.type == "cuda" else None,
        }
        _atomic_torch_save(output / "step_0.pt", initial_state)
        _atomic_torch_save(resume_path, initial_state)

    latent_all = cache["latent"]
    action_all = cache["action"]
    gradient_norms = _check_gradient_paths(model, latent_all, action_all, device_obj)
    if start == 0:
        logs.append({"update": 0, "arm": arm, "gradient_path_norms": gradient_norms})

    for update in range(start, int(updates)):
        ids = schedule["batch_indices"][update]
        latent = latent_all[ids].to(device_obj, non_blocking=True)
        action = action_all[ids].to(device_obj, non_blocking=True)
        timestep = torch.from_numpy(schedule["timestep"][update]).to(device_obj)
        noise = torch.from_numpy(schedule["noise"][update]).to(device_obj)
        mask = torch.from_numpy(schedule["use_prediction"][update]).to(device_obj)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        model.encoder.eval()
        model.projector.eval()
        with torch.autocast(
            device_type=device_obj.type,
            dtype=torch.bfloat16,
            enabled=(device_obj.type == "cuda"),
        ):
            loss, action_loss, latent_loss, _ = _loss_step(
                model, latent, action, timestep, noise, mask, arm
            )
        loss.backward()
        if mode_embedding.grad is not None:
            mode_embedding.grad[2:].zero_()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(gradient_clip),
        )
        optimizer.step()
        completed = update + 1
        if completed in {500, int(updates)}:
            record = {
                "update": completed,
                "arm": arm,
                "loss": float(loss.detach().float().cpu()),
                "action_loss": float(action_loss.detach().float().cpu()),
                "latent_loss": float(latent_loss.detach().float().cpu()),
            }
            logs.append(record)
            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "update": completed,
                "arm": arm,
                "task": task,
                "cache_sha256": cache_hash,
                "schedule_sha256": schedule_hash,
                "base_checkpoint_sha256": checkpoint_sha256,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device_obj) if device_obj.type == "cuda" else None,
                "gradient_path_norms": gradient_norms,
                "log": list(logs),
            }
            _atomic_torch_save(resume_path, state)
            _atomic_torch_save(output / f"step_{completed}.pt", state)
            if completed == int(updates):
                _atomic_torch_save(final_path, state)
        if completed % 50 == 0:
            print({"task": task, "arm": arm, "update": completed, "loss": float(loss.detach().float().cpu())}, flush=True)

    # Frozen feature and D/E state must remain bitwise identical.
    current_state = model.state_dict()
    mismatches = []
    for name, reference in frozen_state.items():
        current = current_state[name.split("buffer::", 1)[-1]] if name.startswith("buffer::") else current_state[name]
        if not torch.equal(reference, current.detach().cpu()):
            mismatches.append(name)
    for name, reference in de_state.items():
        if not torch.equal(reference, current_state[name].detach().cpu()):
            mismatches.append(name)
    if not torch.equal(mode_embedding_de_reference, current_state["mode_embedding.weight"][2:].detach().cpu()):
        mismatches.append("mode_embedding.weight[C:E]")
    if mismatches:
        raise RuntimeError(f"frozen or D/E weights changed: {mismatches}")
    final_hash = _sha256(final_path)
    weights_path = output / "weights_step_1000.pt"
    _atomic_torch_save(
        weights_path,
        {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
    )
    final_weights_hash = _sha256(weights_path)
    report = {
        "status": "completed",
        "task": task,
        "arm": arm,
        "updates": int(updates),
        "checkpoint": str(weights_path),
        "checkpoint_sha256": final_weights_hash,
        "resume_bundle": str(final_path),
        "resume_bundle_sha256": final_hash,
        "base_checkpoint_sha256": checkpoint_sha256,
        "cache_sha256": cache_hash,
        "schedule_sha256": schedule_hash,
        "gradient_path_norms": gradient_norms,
        "frozen_state_bitwise_equal": True,
        "log": logs,
    }
    atomic_json(output / "training_result.json", report)
    return report
