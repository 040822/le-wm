"""Four-way Fast-LeWAM attention pilot protocol and identity helpers.

This module deliberately keeps experiment orchestration separate from the
model implementation.  A pilot variant is part of the immutable run identity;
if the current model/training entrypoint cannot implement that variant, the
worker records a readiness failure instead of manufacturing metrics.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from omegaconf import OmegaConf

from source.common.eval import EvaluationCohort

ROOT = Path(__file__).resolve().parents[2]
PERMITTED_PHYSICAL_GPUS = frozenset({0, 1, 2, 3})
OUTPUT_ROOT = Path("outputs/experiments/fast_lewam_attention_pilots/0820")
ATTENTION_VARIANTS = ("block_causal", "terminal_full")
TASKS = ("reacher", "pusht")
PILOT_GPU_MAPPING = {
    "reacher_block_causal": (0, "reacher", "block_causal"),
    "pusht_block_causal": (1, "pusht", "block_causal"),
    "reacher_terminal_full": (2, "reacher", "terminal_full"),
    "pusht_terminal_full": (3, "pusht", "terminal_full"),
}
GROUNDING_SLOTS = {
    ("reacher", 3): (8, 9, 0, 1, 2, 3),
    ("pusht", 3): (1, 4, 11, 16, 0, 2),
    ("reacher", 10): (8, 9, 27, 28, 30, 31, 44, 47, 49, 0, 1, 2, 3, 4, 5, 6, 7, 10),
    ("pusht", 10): (1, 4, 11, 16, 47, 48, 0, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14),
}
COHORT_SIZES = {1: 10, 3: 20, 10: 50}
CANONICAL_TRAINING = {
    "seed": 3072,
    "action_horizon": 5,
    "stage_b_timestep_mode": "legacy",
    "token_encoding": "legacy",
    "embed_dim": 192,
    "latent_head_layers": 6,
    "latent_head_dim": 192,
    "mlp_dim": 768,
    "heads": 6,
    "loader.batch_size": 128,
    "trainer.precision": "bf16",
    "optimizer.type": "AdamW",
    "optimizer.lr": 5e-5,
    "optimizer.weight_decay": 1e-3,
    "max_epochs": 10,
}

class PilotIdentityError(RuntimeError):
    """An output or continuation does not belong to the requested pilot."""

class StaleArtifactError(PilotIdentityError):
    """An identity-bound artifact changed after its marker was written."""

@dataclass(frozen=True)
class PilotLevel:
    epoch: int
    num_eval: int
    grounded_slots: int

@dataclass(frozen=True)
class PromotionThresholds:
    success_gain_pp: float = 5.0
    spearman_gain: float = 0.05
    regret_reduction_fraction: float = 0.10

@dataclass(frozen=True)
class Pilot:
    name: str
    gpu: int
    task: str
    attention_variant: str
    output_dir: Path
    baseline_run_dir: Path
    baseline_config: Path
    baseline_checkpoints: Mapping[int, Path]
    baseline_result: Path
    grounding_slots: Mapping[int, tuple[int, ...]]

@dataclass(frozen=True)
class PilotManifest:
    path: Path
    repository_root: Path
    output_root: Path
    seed: int
    levels: Mapping[int, PilotLevel]
    promotion: PromotionThresholds
    pilots: Mapping[str, Pilot]
    datasets: Mapping[str, Path]
    identity_files: tuple[Path, ...]
    identity_directories: tuple[Path, ...]
    sha256: str
    raw: Mapping[str, Any]

@dataclass(frozen=True)
class LaunchSpec:
    pilot: str
    gpu: int
    command: tuple[str, ...]
    environment: Mapping[str, str]
    output_dir: Path
    through_epoch: int

@dataclass(frozen=True)
class MemorySnapshot:
    gpu: int
    total_mib: int
    used_mib: int
    free_mib: int
    post_test_free_mib: int | None
    used_fraction: float
    peak_used_mib: int | None = None

def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value

def _digest(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()

def _resolve(root: Path, value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()

def file_sha256(path: str | Path) -> str:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _is_dataset(path: Path) -> bool:
    return path.suffix.lower() in {".h5", ".hdf5", ".zst", ".tar", ".tar.zst"}

def path_identity(path: str | Path, *, dataset=False) -> dict[str, Any]:
    """Return cheap stat identity for datasets and content identity otherwise."""
    path = Path(path).resolve()
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    value: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "inode": int(stat.st_ino),
        "device": int(stat.st_dev),
        "is_dir": bool(path.is_dir()),
    }
    if path.is_file() and not dataset and not _is_dataset(path):
        value["sha256"] = file_sha256(path)
    return value

def directory_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_dir():
        return path_identity(path)
    entries = []
    runtime_parts = {
        ".git", "__pycache__", ".pytest_cache", ".mypy_cache",
        ".ruff_cache", "runtime", "runs", "logs",
    }
    for child in sorted(path.rglob("*")):
        if (
            not child.is_file()
            or any(part in runtime_parts for part in child.parts)
            or child.name.startswith(".tmp-")
            or child.suffix in {".pyc", ".log", ".lock"}
        ):
            continue
        entries.append({
            "relative": str(child.relative_to(path)),
            **path_identity(child, dataset=_is_dataset(child)),
        })
    return {"path": str(path), "exists": True, "files": entries}

def _raw_container(path: Path) -> dict[str, Any]:
    loaded = OmegaConf.load(path)
    value = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(value, dict):
        raise ValueError("attention pilot manifest must contain a mapping")
    return value

def _pilot_paths(root: Path, values: Mapping[str, Any]) -> tuple[Path, Path, dict[int, Path], Path]:
    run_dir = _resolve(root, values["run_dir"])
    config = _resolve(root, values.get("config", str(Path(values["run_dir"]) / "config.yaml")))
    checkpoints = {
        int(epoch): _resolve(root, path)
        for epoch, path in values.get("checkpoints", {}).items()
    }
    result = _resolve(root, values.get("result", str(Path(values["run_dir"]) / "eval/epoch_10/stage_b/result.json")))
    if run_dir is None or config is None or result is None or any(path is None for path in checkpoints.values()):
        raise ValueError("baseline run_dir/config/checkpoints/result are required")
    return run_dir, config, {int(k): path for k, path in checkpoints.items()}, result

def _validate_levels(raw: Mapping[str, Any]) -> dict[int, PilotLevel]:
    levels = {}
    for key, value in raw.items():
        epoch = int(key)
        level = PilotLevel(
            epoch=epoch,
            num_eval=int(value["num_eval"]),
            grounded_slots=int(value.get("grounded_slots", 0)),
        )
        if epoch not in COHORT_SIZES or level.num_eval != COHORT_SIZES[epoch]:
            raise ValueError(f"level {epoch} must evaluate the fixed {COHORT_SIZES.get(epoch)}-row cohort")
        expected_slots = 0 if epoch == 1 else (6 if epoch == 3 else 18)
        if level.grounded_slots != expected_slots:
            raise ValueError(f"level {epoch} has wrong grounding slot budget")
        levels[epoch] = level
    if set(levels) != set(COHORT_SIZES):
        raise ValueError("manifest must define epoch levels 1, 3, and 10")
    return levels

def validate_gpu_mapping(pilots: Mapping[str, Any] | Iterable[Pilot]) -> None:
    """Require the exact four-card experiment schedule."""
    if isinstance(pilots, Mapping):
        names = set(pilots)
        values = pilots
    else:
        values = {pilot.name: pilot for pilot in pilots}
        names = set(values)
    if names != set(PILOT_GPU_MAPPING):
        raise ValueError(f"pilot mapping must be exactly {sorted(PILOT_GPU_MAPPING)}")
    seen = set()
    for name, expected in PILOT_GPU_MAPPING.items():
        value = values[name]
        gpu = int(value["gpu"] if isinstance(value, Mapping) else value.gpu)
        task = str(value["task"] if isinstance(value, Mapping) else value.task)
        variant = str(
            value["attention_variant"] if isinstance(value, Mapping)
            else value.attention_variant
        )
        if (gpu, task, variant) != expected:
            raise ValueError(f"{name} must map to GPU{expected[0]} {expected[1]} {expected[2]}")
        if gpu in seen:
            raise ValueError(f"GPU{gpu} is assigned more than once")
        seen.add(gpu)
    if seen != PERMITTED_PHYSICAL_GPUS:
        raise ValueError("only and all GPU0-GPU3 must be assigned")

def _identity_payload(raw: Mapping[str, Any], root: Path, pilots: Mapping[str, Pilot], datasets: Mapping[str, Path], identity_files: Sequence[Path], identity_dirs: Sequence[Path]) -> dict[str, Any]:
    baseline = {}
    for name, pilot in pilots.items():
        baseline[name] = {
            "config": path_identity(pilot.baseline_config),
            "result": path_identity(pilot.baseline_result),
            "checkpoints": {str(epoch): path_identity(path) for epoch, path in sorted(pilot.baseline_checkpoints.items())},
        }
    return {
        "manifest": raw,
        "canonical_training": CANONICAL_TRAINING,
        "identity_files": [path_identity(path) for path in identity_files],
        "identity_directories": [directory_identity(path) for path in identity_dirs],
        "datasets": {name: path_identity(path, dataset=True) for name, path in datasets.items()},
        "baselines": baseline,
        "cohort_references": {name: path_identity(pilot.baseline_result) for name, pilot in pilots.items()},
    }

def load_pilot_manifest(path: str | Path, *, strict_inputs: bool = False) -> PilotManifest:
    """Load and validate the decision-complete attention pilot manifest."""
    path = Path(path).expanduser().resolve()
    raw = _raw_container(path)
    if int(raw.get("version", 0)) != 1:
        raise ValueError("pilot manifest version must be 1")
    root_value = raw.get("repository_root", ".")
    root = _resolve(ROOT, root_value) if str(root_value) != "." else ROOT
    if root is None:
        raise ValueError("repository_root is required")
    levels = _validate_levels(raw.get("levels", {}))
    promotion = PromotionThresholds(**{
        key: float(value) for key, value in raw.get("promotion", {}).items()
        if key in {"success_gain_pp", "spearman_gain", "regret_reduction_fraction"}
    })
    datasets = {}
    for name, value in raw.get("datasets", {}).items():
        dataset_path = _resolve(root, value.get("path") if isinstance(value, Mapping) else value)
        if dataset_path is None:
            raise ValueError(f"dataset {name} has no path")
        datasets[str(name)] = dataset_path
    identity_files = tuple(
        item for item in (_resolve(root, value) for value in raw.get("identity_files", ()))
        if item is not None
    )
    identity_dirs = tuple(
        item for item in (_resolve(root, value) for value in raw.get("identity_directories", ()))
        if item is not None
    )
    pilots = {}
    baselines = raw.get("baselines", {})
    for name, value in raw.get("pilots", {}).items():
        baseline_name = str(value["baseline"])
        if baseline_name not in baselines:
            raise ValueError(f"pilot {name} references unknown baseline {baseline_name}")
        run_dir, config, checkpoints, result = _pilot_paths(root, baselines[baseline_name])
        output_dir = _resolve(root, value.get("output_dir", str(Path(str(raw.get("output_root", OUTPUT_ROOT))) / name)))
        pilot = Pilot(
            name=str(name),
            gpu=int(value["gpu"]),
            task=str(value["task"]),
            attention_variant=str(value["attention_variant"]),
            output_dir=output_dir if output_dir is not None else root / OUTPUT_ROOT / str(name),
            baseline_run_dir=run_dir,
            baseline_config=config,
            baseline_checkpoints=checkpoints,
            baseline_result=result,
            grounding_slots={
                int(level): tuple(int(slot) for slot in slots)
                for level, slots in value.get("grounding_slots", {}).items()
            },
        )
        pilots[name] = pilot
    validate_gpu_mapping(pilots)
    output_root = _resolve(root, raw.get("output_root", str(OUTPUT_ROOT)))
    if output_root is None:
        raise ValueError("output_root is required")
    # All pilot output paths are normalized under the fixed experiment root.
    for pilot in pilots.values():
        if output_root not in pilot.output_dir.parents:
            raise ValueError(f"pilot output is outside fixed output root: {pilot.output_dir}")
        for epoch, expected_slots in ((3, GROUNDING_SLOTS[(pilot.task, 3)]), (10, GROUNDING_SLOTS[(pilot.task, 10)])):
            if tuple(pilot.grounding_slots.get(epoch, ())) != expected_slots:
                raise ValueError(f"{pilot.name} has incorrect fixed grounding slots for epoch {epoch}")
    identity = _identity_payload(raw, root, pilots, datasets, identity_files, identity_dirs)
    manifest_sha256 = _digest(identity)
    manifest = PilotManifest(
        path=path,
        repository_root=root,
        output_root=output_root,
        seed=int(raw.get("seed", 3072)),
        levels=levels,
        promotion=promotion,
        pilots=pilots,
        datasets=datasets,
        identity_files=identity_files,
        identity_directories=identity_dirs,
        sha256=manifest_sha256,
        raw=raw,
    )
    if strict_inputs:
        validate_manifest_inputs(manifest)
    return manifest

def validate_manifest_inputs(manifest: PilotManifest) -> None:
    missing = []
    for path in manifest.identity_files + manifest.identity_directories:
        if not path.exists():
            missing.append(str(path))
    for path in manifest.datasets.values():
        if not path.is_file():
            missing.append(str(path))
    for pilot in manifest.pilots.values():
        required = [pilot.baseline_config, pilot.baseline_result, *pilot.baseline_checkpoints.values()]
        missing.extend(str(path) for path in required if not path.is_file())
    if missing:
        raise FileNotFoundError(f"attention pilot inputs are missing: {sorted(set(missing))}")
    # Reference results must be complete strict e10 results, not a partial cache.
    for pilot in manifest.pilots.values():
        try:
            cohorts = reference_cohorts(manifest, pilot)
            if len(cohorts[10].row_indices) != 50:
                raise ValueError
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise PilotIdentityError(f"invalid baseline e10 result: {pilot.baseline_result}") from exc

def reference_cohorts(manifest: PilotManifest, pilot_or_task: Pilot | str) -> dict[int, EvaluationCohort]:
    pilot = pilot_or_task if isinstance(pilot_or_task, Pilot) else next(
        item for item in manifest.pilots.values() if item.task == str(pilot_or_task)
    )
    payload = json.loads(pilot.baseline_result.read_text(encoding="utf-8"))
    if payload.get("status", "ok") != "ok":
        raise PilotIdentityError(f"baseline result is not successful: {pilot.baseline_result}")
    parameters = payload.get("parameters", {})
    rows = parameters.get("start_rows")
    episodes = parameters.get("episode_ids")
    starts = parameters.get("start_steps")
    if rows is None or episodes is None or starts is None:
        records = payload.get("episodes", ())
        rows = parameters.get("start_rows")
        episodes = [record.get("dataset_episode") for record in records]
        starts = [record.get("start_step") for record in records]
    if not (len(rows) == len(episodes) == len(starts) == 50):
        raise ValueError("baseline e10 result must contain exactly 50 cohort rows")
    rows = np.asarray(rows, dtype=np.int64)
    episodes = np.asarray(episodes)
    starts = np.asarray(starts, dtype=np.int64)
    if len(np.unique(rows)) != 50:
        raise ValueError("baseline e10 cohort rows must be unique")
    return {
        epoch: EvaluationCohort(
            row_indices=rows[:count].copy(),
            episode_ids=episodes[:count].copy(),
            start_steps=starts[:count].copy(),
        )
        for epoch, count in COHORT_SIZES.items()
    }

def cohort_for_level(manifest: PilotManifest, pilot_or_task: Pilot | str, epoch: int) -> EvaluationCohort:
    epoch = int(epoch)
    if epoch not in COHORT_SIZES:
        raise ValueError(f"unsupported evaluation epoch {epoch}")
    cohorts = reference_cohorts(manifest, pilot_or_task)
    return cohorts[epoch]

def cohort_identity(cohort: EvaluationCohort) -> str:
    return _digest({
        "row_indices": np.asarray(cohort.row_indices, dtype=np.int64).tolist(),
        "episode_ids": _jsonable(np.asarray(cohort.episode_ids)),
        "start_steps": np.asarray(cohort.start_steps, dtype=np.int64).tolist(),
    })

def validate_cohort(cohort: EvaluationCohort, *, expected_length: int, reference: EvaluationCohort | None = None) -> EvaluationCohort:
    if not isinstance(cohort, EvaluationCohort):
        raise TypeError("cohort must be an EvaluationCohort")
    values = (np.asarray(cohort.row_indices), np.asarray(cohort.episode_ids), np.asarray(cohort.start_steps))
    if any(value.ndim != 1 or len(value) != int(expected_length) for value in values):
        raise ValueError("cohort arrays must be one-dimensional and have the expected length")
    if len(np.unique(values[0])) != len(values[0]):
        raise ValueError("cohort rows must be unique")
    if reference is not None:
        for left, right in zip(values, (reference.row_indices, reference.episode_ids, reference.start_steps)):
            if not np.array_equal(left, np.asarray(right)):
                raise ValueError("cohort differs from the strict e10 reference")
    return EvaluationCohort(
        row_indices=values[0].astype(np.int64, copy=True),
        episode_ids=np.array(values[1], copy=True),
        start_steps=values[2].astype(np.int64, copy=True),
    )

def validate_grounding_slots(task: str, epoch: int, slots: Sequence[int]) -> tuple[int, ...]:
    expected = GROUNDING_SLOTS.get((str(task), int(epoch)))
    if expected is None:
        raise ValueError(f"no fixed grounding slot set for {task} epoch {epoch}")
    actual = tuple(int(slot) for slot in slots)
    if actual != expected:
        raise ValueError(f"{task} epoch {epoch} grounding slots differ from protocol")
    if len(set(actual)) != len(actual):
        raise ValueError("grounding slots must be unique")
    return actual

def expected_environment(gpu: int) -> dict[str, str]:
    gpu = int(gpu)
    if gpu not in PERMITTED_PHYSICAL_GPUS:
        raise ValueError("physical GPU must be GPU0-GPU3")
    return {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "MUJOCO_EGL_DEVICE_ID": str(gpu),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "EGL_DEVICE_ID": str(gpu),
        "HYDRA_FULL_ERROR": "1",
    }

def build_launch_specs(manifest: PilotManifest, *, through_epoch: int = 10, pilots: Sequence[str] | None = None, python: str | None = None, worker_script: str | None = None) -> tuple[LaunchSpec, ...]:
    through_epoch = int(through_epoch)
    if through_epoch not in COHORT_SIZES:
        raise ValueError("through_epoch must be one of 1, 3, or 10")
    selected = set(pilots) if pilots is not None else set(manifest.pilots)
    unknown = selected - set(manifest.pilots)
    if unknown:
        raise ValueError(f"unknown pilot(s): {sorted(unknown)}")
    worker = Path(worker_script) if worker_script else manifest.repository_root / "scripts/run_fast_lewam_attention_pilot_worker.py"
    specs = []
    for pilot in sorted((manifest.pilots[name] for name in selected), key=lambda item: item.gpu):
        environment = expected_environment(pilot.gpu)
        command = (
            python or sys.executable,
            "-u",
            str(worker),
            "--manifest", str(manifest.path),
            "--pilot", pilot.name,
            "--through-epoch", str(through_epoch),
            "--physical-gpu", str(pilot.gpu),
            "--device", "cuda:0",
        )
        specs.append(LaunchSpec(
            pilot=pilot.name,
            gpu=pilot.gpu,
            command=command,
            environment=environment,
            output_dir=pilot.output_dir,
            through_epoch=through_epoch,
        ))
    return tuple(specs)

def artifact_identity(paths: Iterable[str | Path], *, dataset=False) -> dict[str, Any]:
    return {str(Path(path).resolve()): path_identity(path, dataset=dataset) for path in paths}

def validate_artifact_identity(artifacts: Mapping[str, Any]) -> bool:
    if not artifacts:
        return False
    for path_text, expected in artifacts.items():
        path = Path(path_text)
        if not path.is_file():
            return False
        current = path_identity(path, dataset=bool(isinstance(expected, Mapping) and "sha256" not in expected))
        if isinstance(expected, str):
            if current.get("sha256") != expected:
                return False
        elif current != expected:
            return False
    return True

def phase_identity(manifest: PilotManifest, pilot: Pilot, *, epoch: int, phase: str, parent_checkpoint: str | Path | None = None, cohort: EvaluationCohort | None = None) -> dict[str, Any]:
    parent = None
    if parent_checkpoint is not None:
        parent = path_identity(parent_checkpoint)
        if not parent.get("exists"):
            raise FileNotFoundError(parent_checkpoint)
    return {
        "schema_version": 1,
        "manifest_sha256": manifest.sha256,
        "pilot": pilot.name,
        "gpu": pilot.gpu,
        "task": pilot.task,
        "attention_variant": pilot.attention_variant,
        "epoch": int(epoch),
        "phase": str(phase),
        "cohort_sha256": cohort_identity(cohort) if cohort is not None else None,
        "parent_checkpoint": parent,
        "canonical_training": CANONICAL_TRAINING,
    }

def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)

def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_phase_marker(path: str | Path, identity: Mapping[str, Any], artifacts: Mapping[str, Any], *, status: str = "complete", **metadata: Any) -> dict[str, Any]:
    if status not in {"complete", "pending_grounding", "failed"}:
        raise ValueError("unsupported phase marker status")
    value = {
        "schema_version": 1,
        "status": status,
        "identity": _jsonable(identity),
        "artifacts": _jsonable(artifacts),
        **_jsonable(metadata),
    }
    atomic_write_json(path, value)
    return value

def read_phase_marker(path: str | Path, expected_identity: Mapping[str, Any], *, require_complete: bool = True) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise StaleArtifactError(f"invalid phase marker: {path}") from exc
    if value.get("identity") != _jsonable(expected_identity):
        raise StaleArtifactError(f"phase marker identity mismatch: {path}")
    if require_complete and value.get("status") != "complete":
        raise StaleArtifactError(f"phase marker is not complete: {path}")
    if value.get("artifacts") and not validate_artifact_identity(value["artifacts"]):
        raise StaleArtifactError(f"phase marker artifacts are stale: {path}")
    return value

def canonical_train_overrides(*, epoch: int, task: str, attention_variant: str, run_dir: str | Path, parent_checkpoint: str | Path | None = None) -> tuple[str, ...]:
    if attention_variant not in ATTENTION_VARIANTS:
        raise ValueError(f"unsupported attention variant {attention_variant}")
    values = [
        f"data={task}",
        "train_mode=stage_b",
        "seed=3072",
        "action_horizon=5",
        "embed_dim=192",
        "latent_head_layers=6",
        "latent_head_dim=192",
        "stage_b_timestep_mode=legacy",
        "token_encoding=legacy",
        f"stage_b_attention_mode={attention_variant}",
        "loader.batch_size=128",
        "trainer.precision=bf16",
        "optimizer.type=AdamW",
        "optimizer.lr=5e-5",
        "optimizer.weight_decay=1e-3",
        "policy.model.heads=6",
        "policy.model.mlp_dim=768",
        f"trainer.max_epochs={int(epoch)}",
        "trainer.devices=1",
        "trainer.accelerator=gpu",
        "epoch_eval.enabled=false",
        "wandb.enabled=false",
        f"info=attention_pilot_{attention_variant}",
        f"hydra.run.dir={Path(run_dir)}",
    ]
    if parent_checkpoint is not None:
        values.append(f"resume_ckpt={Path(parent_checkpoint)}")
    return tuple(values)

def _success_rate_percent(value: Any) -> float:
    number = float(value)
    if not 0.0 <= number <= 100.0:
        raise ValueError("success rate must be in [0, 100] percent")
    return number

def _cohort_key(result: Mapping[str, Any]) -> tuple[tuple[Any, Any], ...]:
    records = result.get("episodes")
    if records is None:
        parameters = result.get("parameters", {})
        episodes = parameters.get("episode_ids", ())
        starts = parameters.get("start_steps", ())
        records = [{"dataset_episode": e, "start_step": s} for e, s in zip(episodes, starts)]
    return tuple((row.get("dataset_episode"), row.get("start_step")) for row in records)

def paired_success_metrics(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    baseline_cohort = _cohort_key(baseline)
    candidate_cohort = _cohort_key(candidate)
    if not baseline_cohort or baseline_cohort != candidate_cohort:
        raise ValueError("baseline and candidate evaluation cohorts differ")
    base = _success_rate_percent(baseline["success_rate"])
    trial = _success_rate_percent(candidate["success_rate"])
    return {
        "baseline_success_percent": base,
        "candidate_success_percent": trial,
        "success_delta_pp": trial - base,
    }

def regret_relative_reduction(baseline_regret: float, candidate_regret: float) -> float:
    baseline_regret = float(baseline_regret)
    candidate_regret = float(candidate_regret)
    if baseline_regret <= 0.0:
        raise ValueError("baseline physical best regret must be positive")
    if candidate_regret < 0.0:
        raise ValueError("candidate physical best regret cannot be negative")
    return (baseline_regret - candidate_regret) / baseline_regret

def _grounding_value(metrics: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    return None

def compute_promotion_metrics(baseline: Mapping[str, Any], candidate: Mapping[str, Any], *, baseline_grounding: Mapping[str, Any] | None = None, candidate_grounding: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result = paired_success_metrics(baseline, candidate)
    result["grounding_available"] = baseline_grounding is not None and candidate_grounding is not None
    if result["grounding_available"]:
        base_s = _grounding_value(baseline_grounding, "predicted_physical_spearman", "spearman")
        trial_s = _grounding_value(candidate_grounding, "predicted_physical_spearman", "spearman")
        base_r = _grounding_value(baseline_grounding, "physical_best_regret", "regret")
        trial_r = _grounding_value(candidate_grounding, "physical_best_regret", "regret")
        if None in (base_s, trial_s, base_r, trial_r):
            raise ValueError("grounding metrics require spearman and physical best regret")
        result["spearman_delta"] = trial_s - base_s
        result["regret_relative_reduction"] = regret_relative_reduction(base_r, trial_r)
    else:
        result["spearman_delta"] = None
        result["regret_relative_reduction"] = None
    return result

def decide_epoch3_promotion(baseline: Mapping[str, Any], candidate: Mapping[str, Any], *, baseline_grounding: Mapping[str, Any] | None = None, candidate_grounding: Mapping[str, Any] | None = None, thresholds: PromotionThresholds = PromotionThresholds()) -> str:
    metrics = compute_promotion_metrics(
        baseline, candidate,
        baseline_grounding=baseline_grounding,
        candidate_grounding=candidate_grounding,
    )
    delta = metrics["success_delta_pp"]
    if delta < 0.0:
        return "stop"
    if delta >= float(thresholds.success_gain_pp):
        return "promote"
    if not metrics["grounding_available"]:
        return "pending_grounding"
    if (
        metrics["spearman_delta"] >= float(thresholds.spearman_gain) - 1e-12
        and metrics["regret_relative_reduction"] >= float(thresholds.regret_reduction_fraction) - 1e-12
    ):
        return "promote"
    return "stop"

def validate_memory_guard(gpu: int, *, total_mib: int, used_mib: int, free_mib: int, post_test_free_mib: int | None = None, peak_used_mib: int | None = None, max_used_fraction: float = 0.85, min_free_mib: int = 7168) -> MemorySnapshot:
    gpu = int(gpu)
    if gpu not in PERMITTED_PHYSICAL_GPUS:
        raise ValueError("memory guard received prohibited GPU")
    total_mib, used_mib, free_mib = map(int, (total_mib, used_mib, free_mib))
    if total_mib <= 0 or min(used_mib, free_mib) < 0:
        raise ValueError("GPU memory values must be non-negative and total_mib positive")
    used_fraction = used_mib / total_mib
    if used_fraction >= float(max_used_fraction):
        raise RuntimeError(f"GPU{gpu} memory use {used_fraction:.3f} is not below {max_used_fraction:.3f}")
    if peak_used_mib is not None:
        peak_used_mib = int(peak_used_mib)
        if peak_used_mib < 0:
            raise ValueError("peak_used_mib must be non-negative")
        peak_fraction = peak_used_mib / total_mib
        if peak_fraction >= float(max_used_fraction):
            raise RuntimeError(f"GPU{gpu} fast-dev peak memory use {peak_fraction:.3f} is not below {max_used_fraction:.3f}")
    if free_mib < int(min_free_mib):
        raise RuntimeError(f"GPU{gpu} has only {free_mib} MiB free; {min_free_mib} MiB required")
    if post_test_free_mib is not None and int(post_test_free_mib) < int(min_free_mib):
        raise RuntimeError(f"GPU{gpu} post-test free memory is unsafe: {post_test_free_mib} MiB")
    return MemorySnapshot(gpu, total_mib, used_mib, free_mib, None if post_test_free_mib is None else int(post_test_free_mib), used_fraction, peak_used_mib)

def memory_guard_ok(*args, **kwargs) -> bool:
    try:
        validate_memory_guard(*args, **kwargs)
    except (RuntimeError, ValueError):
        return False
    return True

def parse_pmon_processes(output: str, gpu: int) -> list[str]:
    gpu = int(gpu)
    if gpu not in PERMITTED_PHYSICAL_GPUS:
        raise ValueError("pmon parser received prohibited GPU")
    rows = []
    for line in str(output).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if fields[0] == "-":
            continue
        try:
            row_gpu = int(fields[0])
        except ValueError as exc:
            raise ValueError(f"invalid nvidia-smi pmon row: {line!r}") from exc
        if row_gpu != gpu:
            raise RuntimeError(f"nvidia-smi pmon returned GPU{row_gpu} while GPU{gpu} was requested")
        if len(fields) < 3:
            raise ValueError(f"invalid nvidia-smi pmon row: {line!r}")
        if fields[1] == "-":
            continue
        rows.append(f"pid={fields[1]} type={fields[2]} command={fields[-1]}")
    return rows

# Compatibility aliases make the protocol helpers convenient for small
# CPU-side launch/verification scripts without duplicating policy logic.
load_manifest = load_pilot_manifest
validate_pilot_mapping = validate_gpu_mapping
parse_pmon = parse_pmon_processes
promotion_decision = decide_epoch3_promotion
validate_exact_cohort = validate_cohort


def paired_success_delta_pp(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[float, float, float]:
    metrics = paired_success_metrics(baseline, candidate)
    return (
        metrics["baseline_success_percent"],
        metrics["candidate_success_percent"],
        metrics["success_delta_pp"],
    )


__all__ = [
    "ATTENTION_VARIANTS", "CANONICAL_TRAINING", "COHORT_SIZES",
    "GROUNDING_SLOTS", "LaunchSpec", "MemorySnapshot", "OUTPUT_ROOT",
    "PERMITTED_PHYSICAL_GPUS", "PILOT_GPU_MAPPING", "Pilot", "PilotLevel",
    "PilotIdentityError", "PilotManifest", "PromotionThresholds",
    "StaleArtifactError", "artifact_identity", "atomic_write_json",
    "build_launch_specs", "canonical_train_overrides", "cohort_for_level",
    "cohort_identity", "compute_promotion_metrics", "decide_epoch3_promotion",
    "directory_identity", "expected_environment", "file_sha256",
    "load_pilot_manifest", "memory_guard_ok", "paired_success_metrics",
    "parse_pmon_processes", "path_identity", "phase_identity",
    "read_json", "read_phase_marker", "reference_cohorts",
    "regret_relative_reduction", "validate_artifact_identity",
    "validate_cohort", "validate_gpu_mapping", "validate_grounding_slots",
    "validate_manifest_inputs", "validate_memory_guard", "write_phase_marker",
    "load_manifest", "validate_pilot_mapping", "parse_pmon",
    "promotion_decision", "validate_exact_cohort", "paired_success_delta_pp",
]
