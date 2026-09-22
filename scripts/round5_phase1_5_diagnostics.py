#!/usr/bin/env python3
"""Collect the non-closed-loop Round 5 Phase1.5 diagnostics.

The main Phase1.5 entry point owns the 2,424-condition closed-loop scan.  This
entry point owns diagnostics whose data must not be inferred from that scan:
trajectory-disjoint physical probes and small deterministic protocol checks.
Probe artifacts are written below the independent Phase1.5 output root.

Examples::

    python scripts/round5_phase1_5_diagnostics.py validate
    CUDA_VISIBLE_DEVICES=0 python scripts/round5_phase1_5_diagnostics.py probe \
        --task cube --gpu 0 --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.checkpoint import load_policy_or_model
from source.common.eval import compose_eval_config, get_dataset, img_transform
from source.common.round3_phase1 import CohortManifest
from source.common.round5_phase1_5 import (
    PHASE15_PROTOCOL_VARIANT,
    PHASE15_TASKS,
    atomic_write_json,
    fit_ridge_probe,
    make_probe_split,
)


DEFAULT_CONFIG = ROOT / "config" / "round5" / "phase1_5.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "round5" / "phase1_5_seed3072_legacy"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"diagnostic config must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu(value: str | None) -> str | None:
    if value is None:
        return None
    if not str(value).isdigit() or int(value) not in range(8):
        raise argparse.ArgumentTypeError("--gpu must select one physical GPU0-7")
    return str(value)


def _configure_device(device: str, gpu: str | None) -> torch.device:
    if str(device).startswith("cuda"):
        if gpu is None:
            raise ValueError("CUDA diagnostics require --gpu")
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {gpu} is unavailable")
    return torch.device(device)


def _manifest(config: Mapping[str, Any], task: str) -> CohortManifest:
    path = _resolve(config["cohort"]["paths"][task])
    value = CohortManifest.load(path)
    if value.protocol_variant != PHASE15_PROTOCOL_VARIANT or value.cohort_kind != "dev" or len(value.entries) != 50:
        raise ValueError(f"{task} is not the required legacy_50/dev cohort: {path}")
    expected = config["cohort"].get("sha256", {}).get(task)
    if expected is not None and value.computed_sha256 != expected:
        raise ValueError(f"{task} cohort hash changed")
    return value


def _checkpoint(config: Mapping[str, Any], task: str) -> tuple[Path, str]:
    path = _resolve(config["training"]["checkpoints"][task])
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = _sha256_file(path)
    expected = config["training"].get("checkpoint_sha256", {}).get(task)
    if expected is not None and digest != expected:
        raise ValueError(f"{task} checkpoint hash changed: {digest} != {expected}")
    return path, digest


def _episode_column(dataset: Any) -> str:
    if "episode_idx" in dataset.column_names:
        return "episode_idx"
    if "ep_idx" in dataset.column_names:
        return "ep_idx"
    raise ValueError("dataset has no episode index column")


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _probe_target(task: str, values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode cyclic physical fields as sin/cos while preserving the schema."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"probe targets must be [rows, dimensions], got {values.shape}")
    if task == "reacher":
        return np.concatenate((np.sin(values), np.cos(values)), axis=1), {
            "source": "qpos",
            "cyclic_dimensions": list(range(values.shape[1])),
        }
    if task == "pusht":
        if values.shape[1] < 5:
            raise ValueError("PushT state must contain an angle in column 4")
        return np.concatenate((values[:, :4], np.sin(values[:, 4:5]), np.cos(values[:, 4:5])), axis=1), {
            "source": "state",
            "cyclic_dimensions": [4],
        }
    return values, {"source": "privileged_block_0_pos" if task == "cube" else "proprio", "cyclic_dimensions": []}


def _probe_rows(
    dataset: Any,
    *,
    task: str,
    selected_trajectories: Sequence[Any],
    frames_per_trajectory: int,
) -> tuple[list[int], list[Any]]:
    episodes = np.asarray(dataset.get_col_data(_episode_column(dataset)))
    steps = np.asarray(dataset.get_col_data("step_idx"), dtype=np.int64)
    selected = {_scalar(item) for item in selected_trajectories}
    rows: list[int] = []
    ids: list[Any] = []
    for episode in selected_trajectories:
        episode_value = _scalar(episode)
        positions = np.flatnonzero(episodes == episode_value)
        if len(positions) == 0:
            continue
        count = min(int(frames_per_trajectory), len(positions))
        picked = np.linspace(0, len(positions) - 1, count, dtype=np.int64)
        for position in positions[picked]:
            rows.append(int(position))
            ids.append(episode_value)
    if not rows:
        raise ValueError(f"no probe rows found for {task}; selected={sorted(selected)!r}")
    return rows, ids


def _target_column(task: str) -> str:
    return {
        "cube": "privileged_block_0_pos",
        "pusht": "state",
        "reacher": "qpos",
        "tworoom": "proprio",
    }[task]


def _jsonable_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in {
            "model",
            "x_scaler",
            "y_scaler",
            # The complete arrays are kept in features_targets.npz.  Keeping
            # them in result.json would make the human-readable manifest
            # needlessly huge and would duplicate the probe data.
            "validation_prediction",
            "validation_target",
        }:
            continue
        if isinstance(item, np.ndarray):
            result[key] = item.tolist()
        elif isinstance(item, Mapping):
            result[key] = _jsonable_metrics(item)
        elif isinstance(item, (np.generic,)):
            result[key] = item.item()
        else:
            result[key] = item
    return result


def _collect_features(
    dataset: Any,
    model: Any,
    transform: Any,
    *,
    rows: Sequence[int],
    target_column: str,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), int(batch_size)):
            batch_rows = list(rows[start : start + int(batch_size)])
            raw = dataset.get_row_data(batch_rows)
            pixels = np.asarray(raw["pixels"])
            image_batch = torch.stack([transform(pixel) for pixel in pixels]).to(device)
            latent = model.encode_pixels(image_batch).float().detach().cpu().numpy()
            features.append(np.asarray(latent, dtype=np.float64))
            targets.append(np.asarray(raw[target_column], dtype=np.float64))
    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


def _fit_mlp(
    features: np.ndarray,
    targets: np.ndarray,
    trajectory_ids: Sequence[Any],
    split: Mapping[str, Sequence[Any]],
    *,
    seeds: Sequence[int] = (0, 1, 2),
    epochs: int = 50,
    patience: int = 5,
) -> list[dict[str, Any]]:
    """Fit the plan's supplementary one-hidden-layer probe."""
    from torch.utils.data import DataLoader, TensorDataset

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    ids = np.asarray(trajectory_ids, dtype=object)
    train_ids, validation_ids = set(split["train"]), set(split["validation"])
    train_mask = np.asarray([item in train_ids for item in ids])
    validation_mask = np.asarray([item in validation_ids for item in ids])
    x_mean, x_std = x[train_mask].mean(0), x[train_mask].std(0)
    y_mean, y_std = y[train_mask].mean(0), y[train_mask].std(0)
    x_std[x_std < 1e-6] = 1.0
    y_std[y_std < 1e-6] = 1.0
    x_train = torch.from_numpy((x[train_mask] - x_mean) / x_std)
    y_train = torch.from_numpy((y[train_mask] - y_mean) / y_std)
    x_validation = torch.from_numpy((x[validation_mask] - x_mean) / x_std)
    y_validation = y[validation_mask]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        model = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, y.shape[1]),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(int(seed)))
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        for _ in range(int(epochs)):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                val = model(x_validation).numpy() * y_std + y_mean
            val_loss = float(np.mean(np.square(val - y_validation)))
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= int(patience):
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        with torch.inference_mode():
            prediction = model(x_validation).numpy() * y_std + y_mean
        error = prediction - y_validation
        rows.append({
            "seed": int(seed),
            "epochs_run": int(_ + 1),
            "validation_count": int(len(y_validation)),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "per_dimension_mae": np.mean(np.abs(error), axis=0).tolist(),
        })
    return rows


def probe(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    tasks = PHASE15_TASKS if args.task == "all" else (args.task,)
    device = _configure_device(args.device, args.gpu)
    output_root = _resolve(args.output_root) / "diagnostics" / "probe"
    output_root.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    for task in tasks:
        manifest = _manifest(config, task)
        checkpoint, checkpoint_sha256 = _checkpoint(config, task)
        cfg = compose_eval_config(task, overrides=["dataset.keys_to_cache=[action]"])
        dataset = get_dataset(cfg, cfg.eval.dataset_name)
        episode_column = _episode_column(dataset)
        all_episodes = list(dict.fromkeys(_scalar(item) for item in np.asarray(dataset.get_col_data(episode_column))))
        excluded = {_scalar(entry.episode_id) for entry in manifest.entries}
        candidates = [item for item in all_episodes if item not in excluded]
        split = make_probe_split(
            candidates,
            eval_trajectory_ids=excluded,
            seed=int(args.seed),
            max_trajectories=int(args.max_trajectories),
        )
        chosen = tuple(split["train"]) + tuple(split["validation"])
        rows, trajectory_ids = _probe_rows(
            dataset,
            task=task,
            selected_trajectories=chosen,
            frames_per_trajectory=int(args.frames_per_trajectory),
        )
        model, resolved = load_policy_or_model(str(checkpoint))
        if resolved is not None and Path(resolved).resolve() != checkpoint.resolve():
            raise ValueError(f"checkpoint resolver changed requested path: {checkpoint}")
        model = getattr(model, "model", model).to(device).eval()
        transform = img_transform(cfg)
        features, raw_targets = _collect_features(
            dataset,
            model,
            transform,
            rows=rows,
            target_column=_target_column(task),
            device=device,
            batch_size=int(args.batch_size),
        )
        targets, target_schema = _probe_target(task, raw_targets)
        ridge = fit_ridge_probe(features, targets, trajectory_ids, split=split, seed=int(args.seed))
        result = {
            "schema_version": "round5_phase1_5_probe_v1",
            "status": "ok",
            "task": task,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "cohort_id": manifest.cohort_id,
            "cohort_sha256": manifest.computed_sha256,
            "seed": int(args.seed),
            "excluded_evaluation_trajectories": sorted(excluded, key=str),
            "trajectory_split": {key: list(value) for key, value in split.items()},
            "rows": len(rows),
            "feature_dimension": int(features.shape[1]),
            "target_dimension": int(targets.shape[1]),
            "target_schema": target_schema,
            "ridge": _jsonable_metrics(ridge),
        }
        if args.mlp:
            result["mlp"] = _fit_mlp(features, targets, trajectory_ids, split, seeds=(0, 1, 2))
        task_root = output_root / task
        task_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(task_root / "features_targets.npz", features=features, targets=targets, trajectory_ids=np.asarray(trajectory_ids, dtype=object))
        atomic_write_json(task_root / "result.json", result)
        all_results.append(result)
        print(json.dumps({"task": task, "rows": len(rows), "result": str(task_root / "result.json")}, ensure_ascii=False, sort_keys=True), flush=True)
    atomic_write_json(output_root / "summary.json", {"schema_version": "round5_phase1_5_probe_summary_v1", "results": all_results})


def validate(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    manifests = {task: _manifest(config, task) for task in PHASE15_TASKS}
    for task, manifest in manifests.items():
        if len(manifest.entries) != 50 or manifest.protocol_variant != "legacy":
            raise AssertionError(f"invalid {task} manifest")
    print(json.dumps({
        "status": "ok",
        "tasks": list(PHASE15_TASKS),
        "trajectory_split_seed": 2026,
        "max_trajectories": 1000,
        "max_frames_per_trajectory": 100,
        "excluded_cohort_entries": 50,
    }, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "probe"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--task", choices=(*PHASE15_TASKS, "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=_gpu)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-trajectories", type=int, default=1000)
    parser.add_argument("--frames-per-trajectory", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mlp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _load_config(_resolve(args.config))
    if args.command == "validate":
        validate(args, config)
    else:
        probe(args, config)


if __name__ == "__main__":
    main()
