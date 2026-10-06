#!/usr/bin/env python3
"""Run the three non-Cube observed-latent probes in the frozen Table 3 plan.

The Cube implementation remains frozen in cvpr_table3_probe.py. This runner
uses its shared encoder, readout and metric functions for TwoRoom, PushT and
Reacher, while keeping separate task schemas and output roots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CORE_PATH = ROOT / "scripts/cvpr_table3_probe.py"
_spec = importlib.util.spec_from_file_location("cvpr_table3_probe_core", CORE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load shared probe implementation at {CORE_PATH}")
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)

TABLE1 = ROOT / "outputs/cvpr/table1/v1"
OUTPUT_ROOT = ROOT / "outputs/cvpr/table3/v1/probe"
SEED = 20261005
TASKS: dict[str, dict[str, Any]] = {
    "tworoom": {
        "dataset": "data/datasets/tworoom.h5",
        "episode_field": "ep_idx",
        "attributes": {"agent_position": {"field": "pos_agent", "dimensions": 2, "slice": None, "units": "environment coordinate units (pixels)"}},
    },
    "pusht": {
        "dataset": "data/datasets/pusht.h5",
        "episode_field": "episode_idx",
        "attributes": {
            "agent_location": {"field": "state", "dimensions": 2, "slice": [0, 2], "units": "pixels"},
            "block_location": {"field": "state", "dimensions": 2, "slice": [2, 4], "units": "pixels"},
            "block_yaw": {"field": "state", "dimensions": 1, "slice": [4, 5], "units": "radians"},
        },
    },
    "reacher": {
        "dataset": "data/datasets/dmcontrol/reacher.h5",
        "episode_field": "ep_idx",
        "attributes": {
            "joint_position": {"field": "qpos", "dimensions": 2, "slice": None, "units": "radians"},
            "joint_velocity": {"field": "qvel", "dimensions": 2, "slice": None, "units": "radians/second"},
        },
    },
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def table1_exclusions(task: str) -> tuple[set[int], dict[str, str]]:
    episodes: set[int] = set()
    sources: dict[str, str] = {}
    for path in sorted((TABLE1 / f"cohorts/{task}").glob("seed_*.json")):
        payload = json.loads(path.read_text())
        sources[str(path.relative_to(ROOT))] = core.sha256_file(path)
        episodes.update(int(item["episode_id"]) for item in payload["entries"])
    if len(sources) != 6:
        raise RuntimeError(f"{task}: expected six Table 1 cohort manifests, got {len(sources)}")
    return episodes, sources


def freeze_sample(
    task: str,
    out: Path,
    *,
    accept_code_update: bool = False,
    code_update_reason: str | None = None,
) -> dict[str, Any]:
    spec = TASKS[task]
    data_path = ROOT / spec["dataset"]
    excluded, cohort_hashes = table1_exclusions(task)
    rng = np.random.default_rng(SEED)
    with h5py.File(data_path, "r") as h5:
        episode_col = np.asarray(h5[spec["episode_field"]], dtype=np.int64)
        unique_ids = np.unique(episode_col)
        lengths = np.asarray(h5["ep_len"], dtype=np.int64)
        offsets = np.asarray(h5["ep_offset"], dtype=np.int64)
        if len(unique_ids) != len(lengths) or len(offsets) != len(lengths):
            raise RuntimeError(f"{task}: episode metadata columns are inconsistent")
        # Dataset episode arrays use contiguous spans. Verify identity before
        # sampling so the manifest never relies on a guessed episode ordering.
        for index, episode_id in enumerate(unique_ids):
            start, length = int(offsets[index]), int(lengths[index])
            if int(episode_col[start]) != int(episode_id) or not np.all(episode_col[start:start + length] == episode_id):
                raise RuntimeError(f"{task}: ep_offset/ep_len do not map to {spec['episode_field']} value {episode_id}")
        eligible = np.asarray([ep for ep in unique_ids if int(ep) not in excluded], dtype=np.int64)
        n_trajectories = min(1000, len(eligible))
        if n_trajectories < 10:
            raise RuntimeError(f"{task}: only {n_trajectories} eligible trajectories")
        chosen = np.sort(rng.choice(eligible, size=n_trajectories, replace=False))
        shuffled = chosen.copy()
        rng.shuffle(shuffled)
        n_train = int(0.8 * n_trajectories)
        n_val = int(0.1 * n_trajectories)
        splits = {**{int(ep): "train" for ep in shuffled[:n_train]}, **{int(ep): "validation" for ep in shuffled[n_train:n_train + n_val]}, **{int(ep): "test" for ep in shuffled[n_train + n_val:]}}
        episode_index = {int(ep): idx for idx, ep in enumerate(unique_ids)}
        rows: list[tuple[int, int, int, str]] = []
        for ep in chosen:
            idx = episode_index[int(ep)]
            length = int(lengths[idx])
            selected_steps = np.sort(rng.choice(length, size=min(100, length), replace=False))
            rows.extend((int(offsets[idx] + step), int(ep), int(step), splits[int(ep)]) for step in selected_steps)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "row_manifest.csv"
    if manifest_path.exists():
        with manifest_path.open(newline="") as stream:
            prior = list(csv.DictReader(stream))
        current = [(str(r), str(ep), str(step), split) for r, ep, step, split in rows]
        stored = [(x["row_index"], x["episode_id"], x["step_index"], x["split"]) for x in prior]
        if current != stored:
            raise RuntimeError(f"{task}: existing frozen row manifest differs from RNG selection")
    else:
        with manifest_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("row_index", "episode_id", "step_index", "split"))
            writer.writerows(rows)

    data_stat = data_path.stat()
    config_path = out / "frozen_config.json"
    manifest_sha = core.sha256_file(manifest_path)
    script_sha = core.sha256_file(Path(__file__))
    helper_sha = core.sha256_file(CORE_PATH)
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if old["data"].get("file_size_bytes") != data_stat.st_size or old["data"].get("file_mtime_ns") != data_stat.st_mtime_ns:
            raise RuntimeError(f"{task}: dataset identity changed after freezing")
        if old["data"].get("row_manifest_sha256") != manifest_sha:
            raise RuntimeError(f"{task}: row manifest hash changed")
        old_script_sha = old["code"].get("sha256")
        old_helper_sha = old["code"].get("shared_probe_core_sha256")
        if old_script_sha != script_sha or old_helper_sha != helper_sha:
            if not accept_code_update or not code_update_reason:
                raise RuntimeError(f"{task}: implementation changed after freezing; use --accept-code-update with a recovery reason")
            acceptance_path = out / "acceptance.json"
            acceptance_status = json.loads(acceptance_path.read_text()).get("status") if acceptance_path.is_file() else None
            has_final_predictions = (out / "test_predictions.npz").exists()
            if (has_final_predictions or acceptance_status is not None) and acceptance_status != "fail":
                raise RuntimeError(f"{task}: cannot change code after successful final prediction artifacts exist")
            update_path = out / "implementation_updates.json"
            updates = json.loads(update_path.read_text()) if update_path.exists() else {"updates": []}
            updates["updates"].append({
                "previous_script_sha256": old_script_sha,
                "new_script_sha256": script_sha,
                "previous_shared_probe_core_sha256": old_helper_sha,
                "new_shared_probe_core_sha256": helper_sha,
                "reason": code_update_reason,
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "completed_features_reused": (out / "features.npz").exists(),
            })
            write_json(update_path, updates)
            old["code"]["sha256"] = script_sha
            old["code"]["shared_probe_core_sha256"] = helper_sha
            old["code"].setdefault("implementation_updates", []).append(code_update_reason)
            write_json(config_path, old)
        return old

    print(f"Hashing {data_path} ({data_stat.st_size / 2**30:.1f} GiB)", flush=True)
    data_sha = core.sha256_file(data_path, progress=True)
    checkpoints = {}
    for name, folder, pattern in (
        ("cowm", TABLE1 / f"assets/cowm/{task}/checkpoints", "*.pt"),
        ("lewm", TABLE1 / f"assets/lewm/{task}", "weights.pt"),
    ):
        checkpoint = next(folder.glob(pattern))
        checkpoints[name] = {"path": str(checkpoint.relative_to(ROOT)), "sha256": core.sha256_file(checkpoint)}
    schema = {
        "task": task,
        "episode_field": spec["episode_field"],
        "attributes": spec["attributes"],
        "targets_are_current_physical_state": True,
        "input": "same-row 224x224 RGB pixels only; no actions, goals, proprioception, history or target values enter either encoder",
        "source_dataset": spec["dataset"],
        "table1_cohort_episode_exclusion_count": len(excluded),
        "timestamp_alignment": "all labels read from the exact HDF5 row used for the RGB frame",
        "sources": {
            "tworoom": "local dataset pos_agent field; raw environment coordinates",
            "pusht": "local PushT state row: [0:2] agent XY, [2:4] block XY, [4:5] block orientation; trailing velocity fields excluded",
            "reacher": "local dm_control qpos/qvel rows; two arm joint angles and velocities",
        }[task],
    }
    write_json(out / "target_schema.json", schema)
    config = {
        "schema_version": 1,
        "experiment": f"cvpr_table3_{task}_observed_latent_probe_v1",
        "task": task,
        "rng_seed": SEED,
        "data": {"path": spec["dataset"], "sha256": data_sha, "file_size_bytes": data_stat.st_size, "file_mtime_ns": data_stat.st_mtime_ns, "sampled_trajectories": n_trajectories, "sampled_frames_per_trajectory": 100, "actual_sampled_frames": len(rows), "episode_split_counts": {"train": n_train, "validation": n_val, "test": n_trajectories - n_train - n_val}, "split_unit": "episode_id", "excluded_table1_episode_ids": sorted(excluded), "table1_cohort_manifests": cohort_hashes, "row_manifest": str(manifest_path.relative_to(ROOT)), "row_manifest_sha256": manifest_sha},
        "models": {name: {**checkpoints[name], "latent_dimension": 192, "latent": "own encoder CLS output after own projector; eval mode, frozen"} for name in checkpoints},
        "readouts": {"linear": {"type": "Ridge with intercept", "alpha_grid": [0, 0.01, 0.1, 1, 10, 100], "selection": "validation standardized MSE"}, "mlp": {"architecture": [192, 256, 256, "target_dim"], "activation": "ReLU", "optimizer": "Adam", "learning_rate": 0.001, "batch_size": 256, "max_epochs": 100, "patience": 10, "seeds": [0, 1, 2]}, "normalization": "fit from probe-train only", "bootstrap": {"unit": "test episode", "replicates": 10000, "seed": SEED}},
        "probe_method_source_audit": {"official_repository": "https://github.com/lucas-maes/le-wm/tree/main", "finding": "public repository files checked do not expose the probing CLI/module; use the documented common rerun protocol frozen in the Table 3 plan", "paper_table_source": "https://le-wm.github.io/"},
        "input_preprocessing": "RGB uint8 to NCHW float, [0,1], ImageNet mean/std; independent preprocessing in each model path",
        "disclosure": "Probe-test episodes are independent of the readout fit; world-model pretraining may have seen the same offline data.",
        "code": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": script_sha, "shared_probe_core": str(CORE_PATH.relative_to(ROOT)), "shared_probe_core_sha256": helper_sha},
        "runtime": {"python": sys.version.split()[0], "pytorch": torch.__version__, "numpy": np.__version__, "h5py": h5py.__version__},
    }
    write_json(config_path, config)
    return config


def read_manifest(out: Path) -> dict[str, np.ndarray]:
    values = np.genfromtxt(out / "row_manifest.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {key: np.asarray(values[key], dtype=str if key == "split" else np.int64) for key in ("row_index", "episode_id", "step_index", "split")}


def load_models(task: str, device: torch.device):
    from source.common.checkpoint import load_policy_or_model
    from source.common.remap import load_pretrained_remapped

    checkpoint = next((TABLE1 / f"assets/cowm/{task}/checkpoints").glob("*.pt"))
    cowm_loaded, _ = load_policy_or_model(str(checkpoint), cache_dir=str(ROOT / "data"))
    cowm = getattr(cowm_loaded, "model", cowm_loaded)
    lewm = load_pretrained_remapped(str(TABLE1 / f"assets/lewm/{task}"), cache_dir=str(ROOT / "data"))
    for model in (cowm, lewm):
        model.to(device).eval().requires_grad_(False)
    return cowm, lewm


@torch.inference_mode()
def extract_features(task: str, out: Path, manifest: dict[str, np.ndarray], device: torch.device) -> None:
    target_path = out / "features.npz"
    if target_path.exists():
        old = np.load(target_path)
        if np.array_equal(old["row_index"], manifest["row_index"]):
            return
        raise RuntimeError(f"{task}: cached feature rows do not match frozen sample")
    cowm, lewm = load_models(task, device)
    rows, episodes, steps = manifest["row_index"], manifest["episode_id"], manifest["step_index"]
    z_cowm = np.empty((len(rows), 192), dtype=np.float32)
    z_lewm = np.empty_like(z_cowm)
    pending_pixels: list[np.ndarray] = []
    pending_indices: list[int] = []
    with h5py.File(ROOT / TASKS[task]["dataset"], "r") as h5:
        offsets = np.asarray(h5["ep_offset"], dtype=np.int64)
        lengths = np.asarray(h5["ep_len"], dtype=np.int64)
        episode_ids = np.unique(np.asarray(h5[TASKS[task]["episode_field"]], dtype=np.int64))
        ep_slot = {int(ep): idx for idx, ep in enumerate(episode_ids)}

        def flush() -> None:
            if not pending_pixels:
                return
            za, zb = core.encode_batch(np.stack(pending_pixels), cowm, lewm, device)
            z_cowm[pending_indices] = za
            z_lewm[pending_indices] = zb
            pending_pixels.clear()
            pending_indices.clear()

        for count, ep in enumerate(np.unique(episodes), 1):
            slot = ep_slot[int(ep)]
            lo, length = int(offsets[slot]), int(lengths[slot])
            pixels = h5["pixels"][lo:lo + length]
            for index in np.flatnonzero(episodes == ep):
                if rows[index] != lo + steps[index]:
                    raise RuntimeError(f"{task}: row-to-frame alignment failure at {rows[index]}")
                pending_pixels.append(pixels[int(steps[index])])
                pending_indices.append(int(index))
                if len(pending_pixels) >= 256:
                    flush()
            if count % 25 == 0 or count == len(np.unique(episodes)):
                flush()
                print(f"{task}: encoded {count}/{len(np.unique(episodes))} episodes", flush=True)
        flush()
    temp = out / "features.tmp.npz"
    np.savez_compressed(temp, row_index=rows, cowm=z_cowm, lewm=z_lewm)
    temp.replace(target_path)


def load_targets(task: str, out: Path, manifest: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    target_path = out / "targets.npz"
    attrs = TASKS[task]["attributes"]
    if target_path.exists():
        saved = np.load(target_path)
        if np.array_equal(saved["row_index"], manifest["row_index"]):
            return {key: saved[key] for key in ["row_index", *attrs]}
        raise RuntimeError(f"{task}: cached targets do not match manifest")
    result: dict[str, np.ndarray] = {"row_index": manifest["row_index"]}
    with h5py.File(ROOT / TASKS[task]["dataset"], "r") as h5:
        for name, spec in attrs.items():
            values = np.asarray(h5[spec["field"]][manifest["row_index"]], dtype=np.float32)
            if spec["slice"] is not None:
                values = values[:, spec["slice"][0]:spec["slice"][1]]
            values = values.reshape(len(manifest["row_index"]), spec["dimensions"])
            if not np.isfinite(values).all():
                raise RuntimeError(f"{task}: non-finite values in {spec['field']}")
            result[name] = values
    temp = out / "targets.tmp.npz"
    np.savez_compressed(temp, **result)
    temp.replace(target_path)
    return result


def metric(task: str, attribute: str, y: np.ndarray, p: np.ndarray, mean: np.ndarray, scale: np.ndarray, constant: np.ndarray) -> dict[str, Any]:
    result = core.score_prediction(attribute, y, p, mean, scale, constant)
    if task == "reacher" and attribute == "joint_position":
        delta = np.arctan2(np.sin(p - y), np.cos(p - y))
        result["auxiliary"]["wrapped_angle_mae"] = float(np.mean(np.abs(delta)))
        result["auxiliary"]["wrapped_angle_rmse"] = float(np.sqrt(np.mean(delta ** 2)))
    return result


def fit_all(task: str, out: Path, manifest: dict[str, np.ndarray], device: torch.device) -> None:
    attrs = TASKS[task]["attributes"]
    feature_archive = np.load(out / "features.npz")
    if not np.array_equal(feature_archive["row_index"], manifest["row_index"]):
        raise RuntimeError(f"{task}: feature rows mismatch")
    features = {name: np.asarray(feature_archive[name], dtype=np.float32) for name in ("cowm", "lewm")}
    targets = load_targets(task, out, manifest)
    split = manifest["split"]
    train, val, test = [np.flatnonzero(split == name) for name in ("train", "validation", "test")]
    eps = [set(manifest["episode_id"][idx].tolist()) for idx in (train, val, test)]
    if any(eps[i] & eps[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError(f"{task}: episode leakage across train/validation/test")
    from sklearn.linear_model import Ridge

    metrics: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {"test_row_index": manifest["row_index"][test], "test_episode_id": manifest["episode_id"][test], "test_step_index": manifest["step_index"][test]}
    normalizers: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    rng = np.random.default_rng(SEED)
    for attr, attr_spec in attrs.items():
        y = targets[attr]
        mean, std = y[train].mean(0), y[train].std(0)
        constant = std == 0
        scale = std.copy()
        scale[constant] = 1.0
        yn = (y - mean) / scale
        normalizers[attr] = {"mean": mean.tolist(), "std": scale.tolist(), "constant_train_coordinate": constant.tolist(), "field": attr_spec["field"]}
        predictions[f"{attr}__truth"] = y[test]
        predictions[f"{attr}__train_mean"] = np.broadcast_to(mean, (len(test), len(mean))).copy()
        for model_name, raw in features.items():
            xm, xs = raw[train].mean(0), raw[train].std(0)
            xs[xs == 0] = 1.0
            x = (raw - xm) / xs
            train_mean_metrics = metric(task, attr, y[test], predictions[f"{attr}__train_mean"], mean, scale, constant)
            metrics.append({"task": task, "model": model_name, "attribute": attr, "readout": "TrainMean", "sample_count": len(test), "coordinate_count": len(mean), **train_mean_metrics})
            best = None
            for alpha in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0):
                candidate = Ridge(alpha=alpha, fit_intercept=True).fit(x[train], yn[train])
                vp = np.asarray(candidate.predict(x[val])).reshape(-1, len(mean))
                mse = float(np.mean((vp - yn[val]) ** 2))
                if best is None or mse < best[0]:
                    best = (mse, alpha, candidate)
            assert best is not None
            linear = np.asarray(best[2].predict(x[test])).reshape(-1, len(mean)) * scale + mean
            saved_linear = linear.astype(np.float32)
            predictions[f"{attr}__{model_name}__linear"] = saved_linear
            metrics.append({"task": task, "model": model_name, "attribute": attr, "readout": "Linear", "sample_count": len(test), "coordinate_count": len(mean), "validation": {"selected_alpha": best[1], "standardized_mse": best[0]}, **metric(task, attr, y[test], saved_linear, mean, scale, constant)})
            checkpoint_dir = out / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            np.savez_compressed(checkpoint_dir / f"{model_name}_{attr}_linear.npz", coef=best[2].coef_, intercept=best[2].intercept_, feature_mean=xm, feature_std=xs, target_mean=mean, target_scale=scale, alpha=best[1])
            shuffled = np.random.default_rng(SEED + list(attrs).index(attr)).permutation(train)
            control = Ridge(alpha=float(best[1]), fit_intercept=True).fit(x[train], yn[shuffled])
            shuffled_prediction = np.asarray(control.predict(x[test])).reshape(-1, len(mean)) * scale + mean
            metrics.append({"task": task, "model": model_name, "attribute": attr, "readout": "LinearLabelShuffle", "sample_count": len(test), "coordinate_count": len(mean), **metric(task, attr, y[test], shuffled_prediction, mean, scale, constant)})
            mlp_records = []
            for seed in (0, 1, 2):
                mlp, epoch, val_mse = core.fit_mlp(x[train], yn[train], x[val], yn[val], seed=seed, device=device)
                with torch.no_grad():
                    predn = mlp(torch.as_tensor(x[test], dtype=torch.float32, device=device)).cpu().numpy()
                pred = predn * scale + mean
                saved_pred = pred.astype(np.float32)
                predictions[f"{attr}__{model_name}__mlp_seed{seed}"] = saved_pred
                metrics.append({"task": task, "model": model_name, "attribute": attr, "readout": "MLP", "readout_seed": seed, "sample_count": len(test), "coordinate_count": len(mean), "validation": {"best_epoch": epoch, "standardized_mse": val_mse}, **metric(task, attr, y[test], saved_pred, mean, scale, constant)})
                checkpoint_path = checkpoint_dir / f"{model_name}_{attr}_mlp_seed{seed}.pt"
                torch.save({"state_dict": mlp.cpu().state_dict(), "feature_mean": xm, "feature_std": xs, "target_mean": mean, "target_scale": scale, "seed": seed, "best_epoch": epoch, "validation_mse": val_mse}, checkpoint_path)
                mlp_records.append({"seed": seed, "best_epoch": epoch, "validation_mse": val_mse, "checkpoint": str(checkpoint_path.relative_to(ROOT)), "sha256": core.sha256_file(checkpoint_path)})
                del mlp
            selections[f"{model_name}.{attr}"] = {"linear_alpha": best[1], "mlp": mlp_records}
        for seed in (None, 0, 1, 2):
            selector = "Linear" if seed is None else "MLP"
            key = "linear" if seed is None else f"mlp_seed{seed}"
            interval = core.cluster_bootstrap(y[test], predictions[f"{attr}__cowm__{key}"], predictions[f"{attr}__lewm__{key}"], manifest["episode_id"][test], mean, scale, constant, rng)
            for row in metrics:
                if row.get("attribute") == attr and row.get("readout") == selector and (seed is None or row.get("readout_seed") == seed):
                    row["paired_bootstrap_cowm_minus_lewm"] = interval
        print(f"{task}: fitted {attr}", flush=True)

    for model_name in ("cowm", "lewm"):
        for readout in ("TrainMean", "Linear", "MLP"):
            rows = [row for row in metrics if row["model"] == model_name and row["readout"] == readout]
            if readout == "MLP":
                seeds = []
                for seed in (0, 1, 2):
                    selected = [row for row in rows if row.get("readout_seed") == seed]
                    r = [row["pearson_r"] for row in selected if row["pearson_r"] is not None]
                    seeds.append({"seed": seed, "standardized_mse": float(np.mean([row["standardized_mse"] for row in selected])), "pearson_r": float(np.mean(r)) if r else None})
                metrics.append({"model": model_name, "attribute": "overall", "readout": readout, "seed_results": seeds, "standardized_mse_mean": float(np.mean([x["standardized_mse"] for x in seeds])), "standardized_mse_std": float(np.std([x["standardized_mse"] for x in seeds], ddof=1)), "pearson_r_mean": float(np.mean([x["pearson_r"] for x in seeds if x["pearson_r"] is not None])) if any(x["pearson_r"] is not None for x in seeds) else None})
            else:
                rs = [row["pearson_r"] for row in rows if row["pearson_r"] is not None]
                metrics.append({"model": model_name, "attribute": "overall", "readout": readout, "standardized_mse": float(np.mean([row["standardized_mse"] for row in rows])), "pearson_r": float(np.mean(rs)) if rs else None})
    temp = out / "test_predictions.tmp.npz"
    np.savez_compressed(temp, **predictions)
    temp.replace(out / "test_predictions.npz")
    write_json(out / "normalizers.json", normalizers)
    write_json(out / "readout_selection.json", selections)
    write_json(out / "metrics.json", {"schema_version": 1, "metrics": metrics, "overall_aggregation": "equal attribute weight"})
    with (out / "metrics.csv").open("w", newline="") as stream:
        columns = ["task", "model", "attribute", "readout", "readout_seed", "sample_count", "coordinate_count", "standardized_mse", "pearson_r", "physical_mse", "physical_mae"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)


def validate(task: str, out: Path, manifest: dict[str, np.ndarray]) -> None:
    features, targets, pred = np.load(out / "features.npz"), np.load(out / "targets.npz"), np.load(out / "test_predictions.npz")
    metric_rows = json.loads((out / "metrics.json").read_text())["metrics"]
    normalizers = json.loads((out / "normalizers.json").read_text())
    errors = []
    if not np.array_equal(features["row_index"], manifest["row_index"]): errors.append("feature rows differ")
    if not np.array_equal(targets["row_index"], manifest["row_index"]): errors.append("target rows differ")
    if not np.array_equal(pred["test_row_index"], manifest["row_index"][manifest["split"] == "test"]): errors.append("test rows differ")
    if set(manifest["episode_id"][manifest["split"] == "train"]) & set(manifest["episode_id"][manifest["split"] != "train"]): errors.append("episode leakage")
    attr_names = set(TASKS[task]["attributes"])
    final = [row for row in metric_rows if row.get("attribute") in attr_names and row.get("model") in {"cowm", "lewm"}]
    if sum(row["readout"] == "Linear" for row in final) != 2 * len(attr_names): errors.append("linear fit count mismatch")
    if sum(row["readout"] == "MLP" for row in final) != 6 * len(attr_names): errors.append("MLP fit count mismatch")
    recomputed = 0
    for row in final:
        if row["readout"] not in {"Linear", "MLP"}:
            continue
        attr, model_name = row["attribute"], row["model"]
        suffix = "linear" if row["readout"] == "Linear" else f"mlp_seed{row['readout_seed']}"
        stats = normalizers[attr]
        values = metric(task, attr, pred[f"{attr}__truth"], pred[f"{attr}__{model_name}__{suffix}"], np.asarray(stats["mean"]), np.asarray(stats["std"]), np.asarray(stats["constant_train_coordinate"]))
        if not np.isclose(values["standardized_mse"], row["standardized_mse"], rtol=1e-6, atol=1e-7):
            errors.append(f"standardized MSE cannot be recomputed for {attr}/{model_name}/{suffix}")
        if row["pearson_r"] is not None and values["pearson_r"] is not None and not np.isclose(values["pearson_r"], row["pearson_r"], rtol=1e-6, atol=1e-7):
            errors.append(f"Pearson r cannot be recomputed for {attr}/{model_name}/{suffix}")
        recomputed += 1
    write_json(out / "acceptance.json", {"status": "pass" if not errors else "fail", "errors": errors, "checks": {"episode_disjoint": "episode leakage" not in errors, "same_rows_both_models": "test rows differ" not in errors, "attribute_count": len(attr_names), "linear_fits": sum(row["readout"] == "Linear" for row in final), "mlp_fits": sum(row["readout"] == "MLP" for row in final), "metrics_recomputed_from_row_level_predictions": recomputed == 8 * len(attr_names), "recomputed_final_readouts": recomputed}, "row_manifest_sha256": core.sha256_file(out / "row_manifest.csv"), "features_sha256": core.sha256_file(out / "features.npz"), "predictions_sha256": core.sha256_file(out / "test_predictions.npz")})
    if errors:
        raise RuntimeError(f"{task}: output acceptance failed: {errors}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--stage", choices=("all", "extract", "fit", "validate"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--accept-code-update", action="store_true", help="record a recovery code fix when no successful final predictions exist")
    parser.add_argument("--code-update-reason")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    out = OUTPUT_ROOT / args.task
    config = freeze_sample(args.task, out, accept_code_update=args.accept_code_update, code_update_reason=args.code_update_reason)
    manifest = read_manifest(out)
    device = torch.device(args.device)
    if args.stage in {"all", "extract"}:
        extract_features(args.task, out, manifest, device)
    if args.stage in {"all", "fit"}:
        fit_all(args.task, out, manifest, device)
    if args.stage in {"all", "fit", "validate"}:
        validate(args.task, out, manifest)
    print(f"{args.task} probe stage {args.stage} completed at {out}", flush=True)


if __name__ == "__main__":
    main()
