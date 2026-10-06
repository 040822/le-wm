#!/usr/bin/env python3
"""Reproducible CVPR Table 3 observed-latent physical probes.

The first frozen task is the eight-attribute OGBench Cube probe. This script
freezes an episode-disjoint sample, encodes the same RGB rows independently
with the archived Table 1 CoWM-B and LeWM checkpoints, fits the specified
readouts, and writes row-level predictions and recomputable metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/cvpr/table3/v1/probe/cube"
DATA = ROOT / "data/datasets/ogbench/cube_single.h5"
TABLE1 = ROOT / "outputs/cvpr/table1/v1"
RNG_SEED = 20261005
ATTRIBUTES = {
    "joint_position": ("proprio_joint_pos", 6),
    "joint_velocity": ("proprio_joint_vel", 6),
    "end_effector_position": ("proprio_effector_pos", 3),
    "end_effector_yaw": ("proprio_effector_yaw", 1),
    "gripper": ("proprio_gripper_opening", 1),
    "block_position": ("privileged_block_0_pos", 3),
    "block_quaternion": ("privileged_block_0_quat", 4),
    "block_yaw": ("privileged_block_0_yaw", 1),
}
MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[None, :, None, None]
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[None, :, None, None]


def sha256_file(path: Path, progress: bool = False) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    done = 0
    started = time.time()
    with path.open("rb") as stream:
        while True:
            block = stream.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            done += len(block)
            if progress and (done == size or done // (2**30) != (done - len(block)) // (2**30)):
                print(f"SHA256 {path.name}: {done / 2**30:.1f}/{size / 2**30:.1f} GiB", flush=True)
    if progress:
        print(f"SHA256 {path.name} finished in {time.time() - started:.1f}s", flush=True)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def load_excluded_episodes() -> tuple[set[int], dict[str, str]]:
    excluded: set[int] = set()
    manifests: dict[str, str] = {}
    for path in sorted((TABLE1 / "cohorts/cube").glob("seed_*.json")):
        payload = json.loads(path.read_text())
        manifests[str(path.relative_to(ROOT))] = sha256_file(path)
        excluded.update(int(entry["episode_id"]) for entry in payload["entries"])
    if len(manifests) != 6:
        raise RuntimeError(f"expected 6 frozen Table 1 Cube cohorts, found {len(manifests)}")
    return excluded, manifests


def freeze_sample(*, accept_code_update: bool = False, code_update_reason: str | None = None) -> dict[str, Any]:
    excluded, cohort_hashes = load_excluded_episodes()
    rng = np.random.default_rng(RNG_SEED)
    with h5py.File(DATA, "r") as h5:
        lengths = np.asarray(h5["ep_len"], dtype=np.int64)
        offsets = np.asarray(h5["ep_offset"], dtype=np.int64)
        episode_ids = np.arange(len(lengths), dtype=np.int64)
        eligible = np.asarray([ep for ep in episode_ids if int(ep) not in excluded], dtype=np.int64)
        if len(eligible) < 1000:
            raise RuntimeError(f"only {len(eligible)} episodes remain after exclusions")
        chosen = np.sort(rng.choice(eligible, size=1000, replace=False))
        shuffled = chosen.copy()
        rng.shuffle(shuffled)
        n_train, n_val = 800, 100
        split_by_episode = {
            **{int(ep): "train" for ep in shuffled[:n_train]},
            **{int(ep): "validation" for ep in shuffled[n_train:n_train + n_val]},
            **{int(ep): "test" for ep in shuffled[n_train + n_val:]},
        }
        rows: list[tuple[int, int, int, str]] = []
        for ep in chosen:
            length = int(lengths[ep])
            steps = np.sort(rng.choice(length, size=min(100, length), replace=False))
            for step in steps:
                rows.append((int(offsets[ep] + step), int(ep), int(step), split_by_episode[int(ep)]))
        if len(rows) != 100_000:
            print(f"Sample contains {len(rows)} frames because some trajectories are shorter than 100 frames.", flush=True)
        if len({row[1] for row in rows}) != 1000:
            raise RuntimeError("episode sampling did not retain all selected trajectories")

    manifest_path = OUT / "row_manifest.csv"
    if manifest_path.exists():
        with manifest_path.open(newline="") as stream:
            stored_rows = list(csv.DictReader(stream))
        expected_rows = [(str(row), str(ep), str(step), split) for row, ep, step, split in rows]
        actual_rows = [(item["row_index"], item["episode_id"], item["step_index"], item["split"]) for item in stored_rows]
        if actual_rows != expected_rows:
            raise RuntimeError("existing row manifest differs from the frozen RNG selection")
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temp = manifest_path.with_suffix(".csv.tmp")
        with temp.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("row_index", "episode_id", "step_index", "split"))
            writer.writerows(rows)
        temp.replace(manifest_path)

    # An existing freeze is immutable. Re-running must reproduce it exactly.
    generated_manifest_sha = sha256_file(manifest_path)
    config_path = OUT / "frozen_config.json"
    data_stat = DATA.stat()
    if config_path.exists():
        old = json.loads(config_path.read_text())
        old_data = old["data"]
        if old_data.get("file_size_bytes") != data_stat.st_size or old_data.get("file_mtime_ns") != data_stat.st_mtime_ns:
            raise RuntimeError("dataset file identity changed after the probe was frozen")
        if old_data.get("row_manifest_sha256") != generated_manifest_sha:
            raise RuntimeError("row manifest hash differs from the frozen configuration")
        current_code_sha = sha256_file(Path(__file__))
        previous_code_sha = old.get("code", {}).get("sha256")
        if previous_code_sha != current_code_sha:
            if not accept_code_update or not code_update_reason:
                raise RuntimeError("probe script changed after freezing; use --accept-code-update with a reason only for a recovery fix")
            acceptance_path = OUT / "acceptance.json"
            failed_acceptance = acceptance_path.is_file() and json.loads(acceptance_path.read_text()).get("status") == "fail"
            if ((OUT / "test_predictions.npz").exists() or acceptance_path.exists()) and not failed_acceptance:
                raise RuntimeError("cannot change implementation after successful final prediction artifacts exist")
            if failed_acceptance and not accept_code_update:
                raise RuntimeError("failed acceptance artifacts require an explicitly recorded recovery code update")
            update_path = OUT / "implementation_updates.json"
            updates = json.loads(update_path.read_text()) if update_path.exists() else {"updates": []}
            updates["updates"].append({"previous_sha256": previous_code_sha, "new_sha256": current_code_sha, "reason": code_update_reason, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "completed_features_reused": (OUT / "features.npz").exists()})
            write_json(update_path, updates)
            old["code"]["sha256"] = current_code_sha
            old["code"].setdefault("implementation_updates", []).append(code_update_reason)
            write_json(config_path, old)
        return old
    data_sha = sha256_file(DATA, progress=True)
    checkpoint_paths = {
        "cowm": next((TABLE1 / "assets/cowm/cube/checkpoints").glob("*.pt")),
        "lewm": TABLE1 / "assets/lewm/cube/weights.pt",
    }
    checkpoints = {
        key: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
        for key, path in checkpoint_paths.items()
    }
    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except Exception:
        torchvision_version = "not_imported"
    try:
        import stable_pretraining
        sp_version = getattr(stable_pretraining, "__version__", "unknown")
    except Exception:
        sp_version = "unknown"
    config = {
        "schema_version": 1,
        "experiment": "cvpr_table3_cube_observed_latent_probe_v1",
        "frozen_at": "2026-10-05",
        "rng_seed": RNG_SEED,
        "data": {
            "path": str(DATA.relative_to(ROOT)),
            "sha256": data_sha,
            "rows": 2_010_000,
            "file_size_bytes": data_stat.st_size,
            "file_mtime_ns": data_stat.st_mtime_ns,
            "sampled_trajectories": 1000,
            "sampled_frames_per_trajectory": 100,
            "actual_sampled_frames": len(rows),
            "episode_split_counts": {"train": 800, "validation": 100, "test": 100},
            "split_unit": "episode_id",
            "excluded_table1_episode_count": len(excluded),
            "excluded_table1_episode_ids": sorted(excluded),
            "table1_cohort_manifests": cohort_hashes,
            "row_manifest": str(manifest_path.relative_to(ROOT)),
            "row_manifest_sha256": generated_manifest_sha,
        },
        "models": {
            "cowm": {**checkpoints["cowm"], "latent": "Round4FastLeWAM.encode_pixels output after projector", "dimension": 192},
            "lewm": {**checkpoints["lewm"], "latent": "LeWM.encode pixels_emb after projector", "dimension": 192},
        },
        "targets": {
            "joint_position": {"field": "proprio_joint_pos", "dimensions": 6, "units": "radians", "definition": "six robot joint positions"},
            "joint_velocity": {"field": "proprio_joint_vel", "dimensions": 6, "units": "radians/second", "definition": "six robot joint velocities"},
            "end_effector_position": {"field": "proprio_effector_pos", "dimensions": 3, "units": "meters", "frame": "MuJoCo world-frame site_xpos, raw XYZ coordinates"},
            "end_effector_yaw": {"field": "proprio_effector_yaw", "dimensions": 1, "units": "radians", "range": "wrapped by the OGBench SO(3) yaw convention"},
            "gripper": {"field": "proprio_gripper_opening", "dimensions": 1, "units": "unitless normalized fraction", "definition": "clip(qpos[gripper_opening_joint] / 0.8, 0, 1); 0=closed, 1=open; contact and velocity are excluded"},
            "block_position": {"field": "privileged_block_0_pos", "dimensions": 3, "units": "meters", "frame": "MuJoCo world coordinates; raw object free-joint qpos position"},
            "block_quaternion": {"field": "privileged_block_0_quat", "dimensions": 4, "units": "unitless", "order": "w/x/y/z", "definition": "MuJoCo free-joint quaternion"},
            "block_yaw": {"field": "privileged_block_0_yaw", "dimensions": 1, "units": "radians", "range": "OGBench SO(3) compute_yaw_radians convention"},
        },
        "input": {
            "source": "same-row HDF5 RGB pixels, channels-last uint8 to NCHW",
            "preprocessing": "float RGB [0,1], ImageNet mean/std; image size already 224x224, matching Table 1 eval transform",
            "features_are_model_specific": True,
            "labels_actions_goal_or_history_passed_to_encoder": False,
            "latent_layer": "encoder CLS token after each model's own projector; frozen eval mode",
        },
        "readouts": {
            "linear": {"kind": "Ridge with intercept", "alpha_grid": [0, 0.01, 0.1, 1, 10, 100], "selection": "minimum validation mean standardized MSE"},
            "mlp": {"architecture": [192, 256, 256, "target_dim"], "activation": "ReLU", "optimizer": "Adam", "learning_rate": 0.001, "batch_size": 256, "max_epochs": 100, "patience": 10, "seeds": [0, 1, 2]},
            "normalization": "feature and target mean/std fitted on probe-train only; zero-variance target std replaced by 1 and flagged",
            "metrics": ["standardized coordinate MSE", "mean coordinate Pearson r", "physical coordinate MSE", "physical coordinate MAE"],
            "overall": "unweighted mean of the eight attribute-level values",
            "shuffle_control": "train-set labels permuted across rows; deterministic Ridge baseline",
            "bootstrap": {"unit": "test episode", "replicates": 10000, "seed": RNG_SEED, "paired_models": True},
        },
        "disclosures": [
            "Probe-test rows are independent of probe fitting but may have appeared in world-model pretraining.",
            "The original LeWM publication does not specify all target mappings, aggregation weights, normalization or readout settings; this is the common rerun protocol in the Table 3 plan.",
            "Quaternion raw-coordinate MSE is the primary matching coordinate metric; sign-invariant rotation error is supplementary.",
        ],
        "target_source_refs": [
            "https://github.com/seohongpark/ogbench/blob/master/ogbench/manipspace/envs/cube_env.py",
            "https://github.com/seohongpark/ogbench/blob/master/ogbench/manipspace/envs/manipspace_env.py",
        ],
        "runtime": {"python": sys.version.split()[0], "pytorch": torch.__version__, "numpy": np.__version__, "h5py": h5py.__version__, "torchvision": torchvision_version, "stable_pretraining": sp_version, "platform": platform.platform()},
        "code": {"path": "scripts/cvpr_table3_probe.py", "sha256": sha256_file(Path(__file__))},
    }
    write_json(config_path, config)
    return json.loads(config_path.read_text())


def read_manifest() -> dict[str, np.ndarray]:
    path = OUT / "row_manifest.csv"
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {
        "row_index": np.asarray(data["row_index"], dtype=np.int64),
        "episode_id": np.asarray(data["episode_id"], dtype=np.int64),
        "step_index": np.asarray(data["step_index"], dtype=np.int64),
        "split": np.asarray(data["split"], dtype=str),
    }


def load_models(device: torch.device):
    from source.common.checkpoint import load_policy_or_model
    from source.common.remap import load_pretrained_remapped

    cowm_path = next((TABLE1 / "assets/cowm/cube/checkpoints").glob("*.pt"))
    cowm_loaded, _ = load_policy_or_model(str(cowm_path), cache_dir=str(ROOT / "data"))
    cowm = getattr(cowm_loaded, "model", cowm_loaded)
    lewm = load_pretrained_remapped(str(TABLE1 / "assets/lewm/cube"), cache_dir=str(ROOT / "data"))
    for model in (cowm, lewm):
        model.to(device).eval().requires_grad_(False)
    if not hasattr(cowm, "encode_pixels") or not hasattr(lewm, "encode"):
        raise TypeError(f"unexpected model types: {type(cowm)!r}, {type(lewm)!r}")
    return cowm, lewm


@torch.inference_mode()
def encode_batch(pixels_hwc: np.ndarray, cowm, lewm, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    # The archive and Table 1 artifacts store RGB uint8 images.
    raw = torch.from_numpy(np.ascontiguousarray(pixels_hwc)).permute(0, 3, 1, 2).contiguous().to(device)
    z_cowm = cowm.encode_pixels(raw)
    x = raw.float().div_(255.0)
    mean = torch.as_tensor(MEAN, device=device)
    std = torch.as_tensor(STD, device=device)
    x = (x - mean) / std
    z_lewm = lewm.encode({"pixels": x[:, None]})["emb"][:, 0]
    if z_cowm.ndim != 2 or z_lewm.ndim != 2 or z_cowm.shape != z_lewm.shape or z_cowm.shape[-1] != 192:
        raise RuntimeError(f"latent shape/interface mismatch: CoWM={tuple(z_cowm.shape)}, LeWM={tuple(z_lewm.shape)}")
    return z_cowm.float().cpu().numpy(), z_lewm.float().cpu().numpy()


def extract_features(config: dict[str, Any], manifest: dict[str, np.ndarray], device: torch.device) -> None:
    feature_path = OUT / "features.npz"
    if feature_path.exists():
        cached = np.load(feature_path)
        if len(cached["row_index"]) == len(manifest["row_index"]) and np.array_equal(cached["row_index"], manifest["row_index"]):
            print(f"reusing feature cache {feature_path}", flush=True)
            return
        raise RuntimeError("feature cache does not match the frozen row manifest")

    cowm, lewm = load_models(device)
    rows = manifest["row_index"]
    row_to_index = {int(row): i for i, row in enumerate(rows)}
    episodes = manifest["episode_id"]
    steps = manifest["step_index"]
    z_cowm = np.empty((len(rows), 192), dtype=np.float32)
    z_lewm = np.empty_like(z_cowm)
    pending_pixels: list[np.ndarray] = []
    pending_indices: list[int] = []

    def flush() -> None:
        if not pending_pixels:
            return
        a, b = encode_batch(np.stack(pending_pixels), cowm, lewm, device)
        z_cowm[pending_indices] = a
        z_lewm[pending_indices] = b
        pending_pixels.clear()
        pending_indices.clear()

    unique_episodes = np.unique(episodes)
    with h5py.File(DATA, "r") as h5:
        offsets = np.asarray(h5["ep_offset"], dtype=np.int64)
        lengths = np.asarray(h5["ep_len"], dtype=np.int64)
        for num, ep in enumerate(unique_episodes, 1):
            selected = np.flatnonzero(episodes == ep)
            # Load a trajectory as a contiguous HDF5 span, then retain its 100
            # frozen row indices. This respects compressed chunk locality.
            lo = int(offsets[ep])
            pixels = h5["pixels"][lo:lo + int(lengths[ep])]
            for local_index in selected:
                row = int(rows[local_index])
                local_step = int(steps[local_index])
                if row != lo + local_step:
                    raise RuntimeError("row manifest no longer aligns to dataset episode offsets")
                pending_pixels.append(pixels[local_step])
                pending_indices.append(int(local_index))
                if len(pending_pixels) >= 256:
                    flush()
            if num % 25 == 0 or num == len(unique_episodes):
                flush()
                print(f"encoded {num}/{len(unique_episodes)} episodes ({100*num/len(unique_episodes):.1f}%)", flush=True)
    flush()
    np.savez_compressed(feature_path.with_suffix(".npz.tmp"), row_index=rows, cowm=z_cowm, lewm=z_lewm)
    # numpy appends .npz to a path without that suffix.
    temp_path = feature_path.with_name(feature_path.stem + ".npz.tmp.npz")
    if temp_path.exists():
        temp_path.replace(feature_path)
    else:
        Path(str(feature_path) + ".tmp.npz").replace(feature_path)
    print(f"features saved: {feature_path} ({feature_path.stat().st_size / 2**20:.1f} MiB)", flush=True)
    del cowm, lewm


def load_targets(manifest: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    target_path = OUT / "targets.npz"
    if target_path.exists():
        cached = np.load(target_path)
        if np.array_equal(cached["row_index"], manifest["row_index"]):
            return {name: cached[name] for name in ["row_index", *ATTRIBUTES.keys()]}
        raise RuntimeError("target cache does not match the frozen row manifest")
    result: dict[str, np.ndarray] = {"row_index": manifest["row_index"]}
    with h5py.File(DATA, "r") as h5:
        for attribute, (field, dimensions) in ATTRIBUTES.items():
            target = np.asarray(h5[field][manifest["row_index"]], dtype=np.float32).reshape(-1, dimensions)
            if not np.all(np.isfinite(target)):
                raise RuntimeError(f"non-finite values in target field {field}")
            result[attribute] = target
    temp = OUT / "targets.tmp.npz"
    np.savez_compressed(temp, **result)
    temp.replace(target_path)
    return result


def pearson_per_coordinate(y: np.ndarray, pred: np.ndarray, constant: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    result = np.full(y.shape[1], np.nan, dtype=np.float64)
    for col in range(y.shape[1]):
        x = y[:, col]
        z = pred[:, col]
        if constant[col] or len(x) < 2:
            continue
        x = x - x.mean()
        z = z - z.mean()
        x_norm = float(np.linalg.norm(x))
        z_norm = float(np.linalg.norm(z))
        if x_norm <= 1e-14 or z_norm <= 1e-14:
            continue
        result[col] = float(np.dot(x, z) / (x_norm * z_norm))
    return result


def physical_auxiliary(attribute: str, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    values: dict[str, float] = {}
    if attribute in {"end_effector_yaw", "block_yaw"}:
        delta = np.arctan2(np.sin(pred - y), np.cos(pred - y))
        values["wrapped_angle_mae"] = float(np.mean(np.abs(delta)))
        values["wrapped_angle_rmse"] = float(np.sqrt(np.mean(delta**2)))
    if attribute == "block_quaternion":
        q_true = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        q_pred = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
        dot = np.abs(np.sum(q_true * q_pred, axis=1)).clip(0, 1)
        error = 2 * np.arccos(dot)
        values["quaternion_rotation_mae_radians"] = float(np.mean(error))
        values["quaternion_rotation_rmse_radians"] = float(np.sqrt(np.mean(error**2)))
    return values


def score_prediction(attribute: str, y_true: np.ndarray, y_pred: np.ndarray, target_mean: np.ndarray, target_scale: np.ndarray, constant: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    target_mean = np.asarray(target_mean, dtype=np.float64)
    target_scale = np.asarray(target_scale, dtype=np.float64)
    constant = np.asarray(constant, dtype=bool)
    z_true = (y_true - target_mean) / target_scale
    z_pred = (y_pred - target_mean) / target_scale
    mse_coord = np.mean((z_true - z_pred) ** 2, axis=0)
    r_coord = pearson_per_coordinate(y_true, y_pred, constant)
    return {
        "standardized_mse": float(np.mean(mse_coord)),
        "standardized_mse_by_coordinate": mse_coord.tolist(),
        "pearson_r": float(np.nanmean(r_coord)) if np.isfinite(r_coord).any() else None,
        "pearson_r_by_coordinate": [None if not np.isfinite(v) else float(v) for v in r_coord],
        "physical_mse": float(np.mean((y_true - y_pred) ** 2)),
        "physical_mae": float(np.mean(np.abs(y_true - y_pred))),
        "physical_mse_by_coordinate": np.mean((y_true - y_pred) ** 2, axis=0).tolist(),
        "physical_mae_by_coordinate": np.mean(np.abs(y_true - y_pred), axis=0).tolist(),
        "auxiliary": physical_auxiliary(attribute, y_true, y_pred),
    }


def make_mlp(input_dim: int, output_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, output_dim))


def fit_mlp(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
    *, seed: int, device: torch.device,
) -> tuple[nn.Module, int, float]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = make_mlp(x_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    best_state = None
    best_val = math.inf
    best_epoch = 0
    patience, stale = 10, 0
    for epoch in range(1, 101):
        model.train()
        order = torch.randperm(len(xt), device=device)
        for start in range(0, len(xt), 256):
            indices = order[start:start + 256]
            prediction = model(xt[indices])
            loss = torch.mean((prediction - yt[indices]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = torch.mean((model(xv) - yv) ** 2).item()
        if val_loss < best_val - 1e-10:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("MLP did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, best_epoch, float(best_val)


def cluster_bootstrap(
    y: np.ndarray, prediction_a: np.ndarray, prediction_b: np.ndarray,
    episodes: np.ndarray, target_mean: np.ndarray, target_scale: np.ndarray,
    constant: np.ndarray, rng: np.random.Generator, replicates: int = 10_000,
) -> dict[str, Any]:
    unique = np.unique(episodes)
    error_sum_a, error_sum_b, counts = [], [], []
    corr_stats_a, corr_stats_b = [], []
    for ep in unique:
        mask = episodes == ep
        truth = np.asarray(y[mask], dtype=np.float64)
        pred_a = np.asarray(prediction_a[mask], dtype=np.float64)
        pred_b = np.asarray(prediction_b[mask], dtype=np.float64)
        error_sum_a.append(np.sum(np.mean(((truth - pred_a) / target_scale) ** 2, axis=1)))
        error_sum_b.append(np.sum(np.mean(((truth - pred_b) / target_scale) ** 2, axis=1)))
        counts.append(len(truth))

        def sums(pred: np.ndarray) -> np.ndarray:
            return np.stack((truth.sum(0), pred.sum(0), (truth * truth).sum(0), (pred * pred).sum(0), (truth * pred).sum(0)), axis=-1)

        corr_stats_a.append(sums(pred_a))
        corr_stats_b.append(sums(pred_b))
    indices = rng.integers(0, len(unique), size=(replicates, len(unique)))
    count_boot = np.asarray(counts)[indices].sum(1)
    delta_mse = (np.asarray(error_sum_a)[indices].sum(1) - np.asarray(error_sum_b)[indices].sum(1)) / count_boot

    def pooled_r(statistics: np.ndarray) -> np.ndarray:
        # Sufficient statistics allow pooled test-row Pearson r to be
        # recomputed for each episode-cluster bootstrap sample.
        summed = statistics[indices].sum(axis=1)
        n = count_boot[:, None]
        sx, sy, sx2, sy2, sxy = (summed[:, :, i] for i in range(5))
        cov = n * sxy - sx * sy
        vx = n * sx2 - sx * sx
        vy = n * sy2 - sy * sy
        denom = np.sqrt(np.maximum(vx, 0) * np.maximum(vy, 0))
        values = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 0)
        values[:, constant] = np.nan
        valid = np.isfinite(values)
        valid_count = valid.sum(axis=1)
        totals = np.where(valid, values, 0.0).sum(axis=1)
        return np.divide(totals, valid_count, out=np.full(len(totals), np.nan), where=valid_count > 0)

    delta_r = pooled_r(np.asarray(corr_stats_a)) - pooled_r(np.asarray(corr_stats_b))
    valid_r = np.isfinite(delta_r)
    r_ci = np.quantile(delta_r[valid_r], [0.025, 0.975]).tolist() if valid_r.any() else [None, None]
    return {
        "delta_mse_cowm_minus_lewm_ci95": np.quantile(delta_mse, [0.025, 0.975]).tolist(),
        "delta_r_cowm_minus_lewm_ci95": r_ci,
        "test_episode_count": int(len(unique)),
        "replicates": int(replicates),
    }


def fit_all(config: dict[str, Any], manifest: dict[str, np.ndarray], device: torch.device) -> None:
    feature_archive = np.load(OUT / "features.npz")
    if not np.array_equal(feature_archive["row_index"], manifest["row_index"]):
        raise RuntimeError("feature archive rows do not match frozen row manifest")
    features = {name: np.asarray(feature_archive[name], dtype=np.float32) for name in ("cowm", "lewm")}
    targets = load_targets(manifest)
    split = manifest["split"]
    train_idx, val_idx, test_idx = [np.flatnonzero(split == name) for name in ("train", "validation", "test")]
    if len(np.unique(manifest["episode_id"][train_idx])) != 800 or len(np.unique(manifest["episode_id"][val_idx])) != 100 or len(np.unique(manifest["episode_id"][test_idx])) != 100:
        raise RuntimeError("episode split count differs from frozen protocol")
    if set(manifest["episode_id"][train_idx]) & set(manifest["episode_id"][val_idx]) or set(manifest["episode_id"][train_idx]) & set(manifest["episode_id"][test_idx]) or set(manifest["episode_id"][val_idx]) & set(manifest["episode_id"][test_idx]):
        raise RuntimeError("episode leakage across probe splits")

    from sklearn.linear_model import Ridge

    predictions: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    readout_config: dict[str, Any] = {}
    rng = np.random.default_rng(RNG_SEED)
    normalizers: dict[str, Any] = {}
    checkpoints_dir = OUT / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    for attribute, (field, dimensions) in ATTRIBUTES.items():
        y = targets[attribute]
        y_mean = y[train_idx].mean(0)
        y_std = y[train_idx].std(0)
        constant = y_std == 0
        y_scale = y_std.copy()
        y_scale[constant] = 1.0
        y_norm = (y - y_mean) / y_scale
        normalizers[attribute] = {"field": field, "mean": y_mean.tolist(), "std": y_scale.tolist(), "constant_train_coordinate": constant.tolist(), "train_rows": int(len(train_idx))}
        per_model: dict[str, dict[str, np.ndarray]] = {}
        shuffled_idx = np.random.default_rng(RNG_SEED + list(ATTRIBUTES).index(attribute)).permutation(train_idx)
        for model_name, raw_features in features.items():
            x_mean = raw_features[train_idx].mean(0)
            x_std = raw_features[train_idx].std(0)
            x_std[x_std == 0] = 1.0
            x = (raw_features - x_mean) / x_std
            pred_by_readout: dict[str, np.ndarray] = {}

            # Deterministic training-mean predictor.
            mean_pred = np.broadcast_to(y_mean, (len(test_idx), dimensions)).copy()
            pred_by_readout["train_mean"] = mean_pred
            mean_metrics = score_prediction(attribute, y[test_idx], mean_pred, y_mean, y_scale, constant)
            metric_rows.append({"task": "cube", "model": model_name, "attribute": attribute, "field": field, "readout": "TrainMean", "sample_count": len(test_idx), "coordinate_count": dimensions, "validation": {}, **mean_metrics})

            # Ridge: same train/validation rows and alpha grid for both latent spaces.
            best = None
            for alpha in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0):
                regressor = Ridge(alpha=alpha, fit_intercept=True)
                regressor.fit(x[train_idx], y_norm[train_idx])
                val_prediction = np.asarray(regressor.predict(x[val_idx])).reshape(-1, dimensions)
                val_mse = float(np.mean((val_prediction - y_norm[val_idx]) ** 2))
                if best is None or val_mse < best[0]:
                    best = (val_mse, alpha, regressor)
            assert best is not None
            linear_model = best[2]
            linear_prediction = np.asarray(linear_model.predict(x[test_idx])).reshape(-1, dimensions) * y_scale + y_mean
            saved_linear_prediction = linear_prediction.astype(np.float32)
            pred_by_readout["linear"] = saved_linear_prediction
            linear_metrics = score_prediction(attribute, y[test_idx], saved_linear_prediction, y_mean, y_scale, constant)
            linear_row = {"task": "cube", "model": model_name, "attribute": attribute, "field": field, "readout": "Linear", "sample_count": len(test_idx), "coordinate_count": dimensions, "validation": {"selected_alpha": best[1], "standardized_mse": best[0]}, **linear_metrics}
            metric_rows.append(linear_row)
            np.savez_compressed(checkpoints_dir / f"{model_name}_{attribute}_linear.npz", coef=np.asarray(linear_model.coef_), intercept=np.asarray(linear_model.intercept_), feature_mean=x_mean, feature_std=x_std, target_mean=y_mean, target_scale=y_scale, alpha=best[1])

            # Shuffled target control for a linear readout.
            shuffled = Ridge(alpha=float(best[1]), fit_intercept=True)
            shuffled.fit(x[train_idx], y_norm[shuffled_idx])
            shuffle_pred = np.asarray(shuffled.predict(x[test_idx])).reshape(-1, dimensions) * y_scale + y_mean
            metric_rows.append({"task": "cube", "model": model_name, "attribute": attribute, "field": field, "readout": "LinearLabelShuffle", "sample_count": len(test_idx), "coordinate_count": dimensions, "validation": {"label_permutation_seed": RNG_SEED}, **score_prediction(attribute, y[test_idx], shuffle_pred, y_mean, y_scale, constant)})

            # MLP, three separately initialized readout seeds.
            mlp_seeds = []
            for seed in (0, 1, 2):
                mlp, best_epoch, val_mse = fit_mlp(x[train_idx], y_norm[train_idx], x[val_idx], y_norm[val_idx], seed=seed, device=device)
                with torch.no_grad():
                    xtest = torch.as_tensor(x[test_idx], dtype=torch.float32, device=device)
                    prediction_norm = mlp(xtest).cpu().numpy()
                prediction = prediction_norm * y_scale + y_mean
                readout = f"mlp_seed{seed}"
                saved_prediction = prediction.astype(np.float32)
                pred_by_readout[readout] = saved_prediction
                metrics = score_prediction(attribute, y[test_idx], saved_prediction, y_mean, y_scale, constant)
                metric_rows.append({"task": "cube", "model": model_name, "attribute": attribute, "field": field, "readout": "MLP", "readout_seed": seed, "sample_count": len(test_idx), "coordinate_count": dimensions, "validation": {"best_epoch": best_epoch, "best_standardized_mse": val_mse}, **metrics})
                checkpoint_path = checkpoints_dir / f"{model_name}_{attribute}_mlp_seed{seed}.pt"
                torch.save({"state_dict": mlp.cpu().state_dict(), "input_dim": 192, "output_dim": dimensions, "feature_mean": x_mean, "feature_std": x_std, "target_mean": y_mean, "target_scale": y_scale, "seed": seed, "best_epoch": best_epoch, "validation_mse": val_mse}, checkpoint_path)
                mlp_seeds.append({"seed": seed, "best_epoch": best_epoch, "validation_mse": val_mse, "checkpoint": str(checkpoint_path.relative_to(ROOT)), "checkpoint_sha256": sha256_file(checkpoint_path)})
                del mlp, xtest
            per_model[model_name] = pred_by_readout
            readout_config[f"{model_name}.{attribute}"] = {"linear_alpha": best[1], "linear_val_mse": best[0], "mlp_seeds": mlp_seeds}
            normalizers.setdefault("features", {})[model_name] = {"mean": x_mean.tolist(), "std": x_std.tolist(), "fit_split": "train"}

        # Bootstrap fixed-model paired differences by test episode for each readout.
        for readout in ("linear", "mlp_seed0", "mlp_seed1", "mlp_seed2"):
            a = per_model["cowm"][readout]
            b = per_model["lewm"][readout]
            interval = cluster_bootstrap(y[test_idx], a, b, manifest["episode_id"][test_idx], y_mean, y_scale, constant, rng)
            for row in metric_rows:
                if row["attribute"] == attribute and row["readout"] == ("Linear" if readout == "linear" else "MLP") and (readout == "linear" or row.get("readout_seed") == int(readout[-1])):
                    row["paired_bootstrap_cowm_minus_lewm"] = interval
        for readout, prediction in per_model["cowm"].items():
            if readout == "train_mean":
                predictions[f"{attribute}__train_mean"] = prediction
            else:
                predictions[f"{attribute}__cowm__{readout}"] = prediction
        for readout, prediction in per_model["lewm"].items():
            if readout == "train_mean":
                predictions[f"{attribute}__train_mean"] = prediction
            else:
                predictions[f"{attribute}__lewm__{readout}"] = prediction
        print(f"fitted {attribute} ({dimensions} coordinates)", flush=True)

    # Aggregate attributes equally, as frozen in the protocol.
    for model_name in ("cowm", "lewm"):
        for readout in ("TrainMean", "Linear", "MLP"):
            rows = [r for r in metric_rows if r["model"] == model_name and r["readout"] == readout]
            if readout == "MLP":
                by_seed = []
                for seed in (0, 1, 2):
                    seed_rows = [r for r in rows if r.get("readout_seed") == seed]
                    valid_r = [r["pearson_r"] for r in seed_rows if r["pearson_r"] is not None]
                    by_seed.append({"seed": seed, "standardized_mse": float(np.mean([r["standardized_mse"] for r in seed_rows])), "pearson_r": float(np.mean(valid_r)) if valid_r else None})
                valid_overall_r = [x["pearson_r"] for x in by_seed if x["pearson_r"] is not None]
                aggregate = {"model": model_name, "attribute": "overall", "readout": readout, "seed_results": by_seed, "standardized_mse_mean": float(np.mean([x["standardized_mse"] for x in by_seed])), "standardized_mse_std": float(np.std([x["standardized_mse"] for x in by_seed], ddof=1)), "pearson_r_mean": float(np.mean(valid_overall_r)) if valid_overall_r else None}
            else:
                pearsons = [r["pearson_r"] for r in rows if r["pearson_r"] is not None]
                aggregate = {"model": model_name, "attribute": "overall", "readout": readout, "standardized_mse": float(np.mean([r["standardized_mse"] for r in rows])), "pearson_r": float(np.mean(pearsons)) if pearsons else None}
            metric_rows.append(aggregate)

    prediction_payload = {"test_row_index": manifest["row_index"][test_idx], "test_episode_id": manifest["episode_id"][test_idx], "test_step_index": manifest["step_index"][test_idx]}
    for attribute in ATTRIBUTES:
        prediction_payload[f"{attribute}__truth"] = targets[attribute][test_idx]
    prediction_payload.update(predictions)
    temp_pred = OUT / "test_predictions.tmp.npz"
    np.savez_compressed(temp_pred, **prediction_payload)
    temp_pred.replace(OUT / "test_predictions.npz")
    write_json(OUT / "normalizers.json", normalizers)
    write_json(OUT / "readout_selection.json", readout_config)
    write_json(OUT / "metrics.json", {"schema_version": 1, "metrics": metric_rows, "overall_aggregation": "equal weight across eight attributes; MLP reports mean and sample std across three readout seeds", "paired_interval": "10,000 episode-cluster paired bootstrap replicates; CoWM minus LeWM"})
    with (OUT / "metrics.csv").open("w", newline="") as stream:
        columns = ["model", "attribute", "field", "readout", "readout_seed", "sample_count", "coordinate_count", "standardized_mse", "pearson_r", "physical_mse", "physical_mae"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in metric_rows:
            writer.writerow(row)


def validate_outputs(manifest: dict[str, np.ndarray]) -> None:
    features = np.load(OUT / "features.npz")
    targets = np.load(OUT / "targets.npz")
    predictions = np.load(OUT / "test_predictions.npz")
    metrics = json.loads((OUT / "metrics.json").read_text())["metrics"]
    errors = []
    if not np.array_equal(features["row_index"], manifest["row_index"]): errors.append("feature row identity mismatch")
    if not np.array_equal(targets["row_index"], manifest["row_index"]): errors.append("target row identity mismatch")
    if not (set(manifest["episode_id"][manifest["split"] == "train"]).isdisjoint(manifest["episode_id"][manifest["split"] == "validation"]) and set(manifest["episode_id"][manifest["split"] == "train"]).isdisjoint(manifest["episode_id"][manifest["split"] == "test"]) and set(manifest["episode_id"][manifest["split"] == "validation"]).isdisjoint(manifest["episode_id"][manifest["split"] == "test"])): errors.append("episode leakage")
    required = {f"{attribute}__{model}__{readout}" for attribute in ATTRIBUTES for model in ("cowm", "lewm") for readout in ("linear", "mlp_seed0", "mlp_seed1", "mlp_seed2")}
    missing = required - set(predictions.files)
    if missing: errors.append(f"missing predictions: {sorted(missing)}")
    if sum(row.get("attribute") in ATTRIBUTES and row.get("model") in {"cowm", "lewm"} and row.get("readout") == "Linear" for row in metrics) != 16: errors.append("expected 16 final linear fits")
    if sum(row.get("attribute") in ATTRIBUTES and row.get("model") in {"cowm", "lewm"} and row.get("readout") == "MLP" for row in metrics) != 48: errors.append("expected 48 MLP fits")
    # Recompute the main metrics from the saved row-level truth and prediction archive.
    recomputed = 0
    normalizers = json.loads((OUT / "normalizers.json").read_text())
    for row in metrics:
        attr, model, readout = row.get("attribute"), row.get("model"), row.get("readout")
        if attr not in ATTRIBUTES or model not in {"cowm", "lewm"} or readout not in {"Linear", "MLP"}:
            continue
        suffix = "linear" if readout == "Linear" else f"mlp_seed{row['readout_seed']}"
        y = predictions[f"{attr}__truth"]
        p = predictions[f"{attr}__{model}__{suffix}"]
        stats = normalizers[attr]
        actual = score_prediction(attr, y, p, np.asarray(stats["mean"]), np.asarray(stats["std"]), np.asarray(stats["constant_train_coordinate"]))
        if not np.isclose(actual["standardized_mse"], row["standardized_mse"], rtol=1e-6, atol=1e-7): errors.append(f"MSE mismatch: {attr}/{model}/{readout}")
        stored_r = row["pearson_r"]
        actual_r = actual["pearson_r"]
        if stored_r is not None and actual_r is not None and not np.isclose(actual_r, stored_r, rtol=1e-6, atol=1e-7): errors.append(f"Pearson mismatch: {attr}/{model}/{readout}")
        recomputed += 1
    checks = {"eight_attributes_present": len(ATTRIBUTES) == 8, "feature_cache_same_rows": bool(np.array_equal(features["row_index"], manifest["row_index"])), "targets_same_rows": bool(np.array_equal(targets["row_index"], manifest["row_index"])), "episode_split_disjoint": "episode leakage" not in errors, "same_test_rows_for_both_models": bool(np.array_equal(predictions["test_row_index"], manifest["row_index"][manifest["split"] == "test"])), "input_contract_rgb_only": True, "metrics_recomputed_from_saved_predictions": recomputed == 64, "recomputed_final_readouts": recomputed}
    write_json(OUT / "acceptance.json", {"checks": checks, "errors": errors, "status": "pass" if not errors else "fail", "manifest_sha256": sha256_file(OUT / "row_manifest.csv"), "features_sha256": sha256_file(OUT / "features.npz"), "predictions_sha256": sha256_file(OUT / "test_predictions.npz"), "validated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if errors:
        raise RuntimeError("output acceptance failed: " + "; ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "extract", "fit", "validate"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--accept-code-update", action="store_true", help="record and accept a recovery-only code fix before final predictions exist")
    parser.add_argument("--code-update-reason", default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use CPU or repair GPU access")
    device = torch.device(args.device)
    config = freeze_sample(accept_code_update=args.accept_code_update, code_update_reason=args.code_update_reason)
    manifest = read_manifest()
    if args.stage in ("all", "extract"):
        extract_features(config, manifest, device)
    if args.stage in ("all", "fit"):
        fit_all(config, manifest, device)
    if args.stage in ("all", "fit", "validate"):
        validate_outputs(manifest)
    print(f"Cube probe stage {args.stage} completed under {OUT}", flush=True)


if __name__ == "__main__":
    main()
