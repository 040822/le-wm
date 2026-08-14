"""Validated interfaces for the Fast-LeWAM parallel exploration pilots."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf

PERMITTED_PHYSICAL_GPUS = frozenset({0, 1, 2, 3})


_DIRECTIONS = {
    "offline_alignment",
    "action_tokens_25",
    "streaming_online",
}
_TASKS = {"reacher", "pusht", "cube"}


@dataclass(frozen=True)
class PilotLevel:
    max_epochs: int
    num_eval: int
    grounded_slots: int


@dataclass(frozen=True)
class PromotionThresholds:
    maximum_success_regression_pp: float
    success_gain_pp: float
    spearman_gain: float
    regret_reduction_fraction: float


@dataclass(frozen=True)
class Pilot:
    name: str
    gpu: int
    task: str
    direction: str
    output_dir: Path
    baseline_run_config: Path
    baseline_checkpoint: Path
    actor_run_config: Path | None
    actor_checkpoint: Path | None
    trace: Path | None
    slots: tuple[int, ...]


@dataclass(frozen=True)
class PilotManifest:
    path: Path
    repository_root: Path
    output_root: Path
    seed: int
    levels: Mapping[int, PilotLevel]
    promotion: PromotionThresholds
    replay: Mapping[str, Any]
    pilots: Mapping[str, Pilot]
    sha256: str


@dataclass(frozen=True)
class LaunchSpec:
    """One isolated single-GPU pilot worker launch."""

    pilot: str
    gpu: int
    command: tuple[str, ...]
    environment: Mapping[str, str]
    output_dir: Path


def _digest_mapping(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=256)
def _file_sha256_cached(
    path_text: str, size: int, mtime_ns: int, ctime_ns: int, inode: int
) -> str:
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    path = Path(path).resolve()
    stat = path.stat()
    return _file_sha256_cached(
        str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino
    )


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_pilot_manifest(path: str | Path) -> PilotManifest:
    """Load and fully validate the decision-complete four-way pilot manifest."""
    path = Path(path).expanduser().resolve()
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if int(raw.get("version", 0)) != 1:
        raise ValueError("pilot manifest version must be 1")
    root = Path(raw["repository_root"]).expanduser().resolve()
    output_root = _resolve(root, raw["output_root"])
    if output_root is None:
        raise ValueError("output_root is required")

    levels = {}
    for label, values in raw["levels"].items():
        number = int(label)
        level = PilotLevel(
            max_epochs=int(values["max_epochs"]),
            num_eval=int(values["num_eval"]),
            grounded_slots=int(values["grounded_slots"]),
        )
        if min(level.max_epochs, level.num_eval, level.grounded_slots) < 1:
            raise ValueError("pilot level budgets must be positive")
        levels[number] = level
    if set(levels) != {1, 2}:
        raise ValueError("pilot manifest must define levels 1 and 2")
    if levels[2].max_epochs <= levels[1].max_epochs:
        raise ValueError("level 2 max_epochs must exceed level 1")

    promotion = PromotionThresholds(**{
        key: float(value) for key, value in raw["promotion"].items()
    })
    pilots = {}
    seen_gpus = set()
    for name, values in raw["pilots"].items():
        gpu = int(values["gpu"])
        if gpu not in PERMITTED_PHYSICAL_GPUS:
            raise ValueError(
                f"pilot {name} selects prohibited GPU{gpu}; only GPU0-GPU3 are allowed"
            )
        if gpu in seen_gpus:
            raise ValueError(f"multiple pilots select GPU{gpu}")
        seen_gpus.add(gpu)
        task = str(values["task"])
        direction = str(values["direction"])
        if task not in _TASKS:
            raise ValueError(f"unsupported pilot task {task!r}")
        if direction not in _DIRECTIONS:
            raise ValueError(f"unsupported pilot direction {direction!r}")
        slots = tuple(int(slot) for slot in values.get("slots", ()))
        if len(slots) < levels[2].grounded_slots:
            raise ValueError(
                f"pilot {name} requires at least {levels[2].grounded_slots} fixed replay slots"
            )
        if len(set(slots)) != len(slots):
            raise ValueError(f"pilot {name} contains duplicate replay slots")
        pilot = Pilot(
            name=str(name),
            gpu=gpu,
            task=task,
            direction=direction,
            output_dir=output_root / str(name),
            baseline_run_config=_resolve(root, values["baseline_run_config"]),
            baseline_checkpoint=_resolve(root, values["baseline_checkpoint"]),
            actor_run_config=_resolve(root, values.get("actor_run_config")),
            actor_checkpoint=_resolve(root, values.get("actor_checkpoint")),
            trace=_resolve(root, values.get("trace")),
            slots=slots,
        )
        required = (pilot.baseline_run_config, pilot.baseline_checkpoint)
        if any(item is None for item in required):
            raise ValueError(f"pilot {name} is missing its baseline identity")
        pilots[name] = pilot
    if seen_gpus != PERMITTED_PHYSICAL_GPUS:
        raise ValueError("the four-way schedule must assign GPU0, GPU1, GPU2, GPU3")

    identity_files = {
        str(value): _resolve(root, value) for value in raw.get("identity_files", ())
    }
    missing_identity = [
        str(value) for value in identity_files.values()
        if value is None or not value.is_file()
    ]
    if missing_identity:
        raise FileNotFoundError(f"pilot identity files are missing: {missing_identity}")
    dataset_files = {
        str(value): _resolve(root, value) for value in raw.get("dataset_files", ())
    }
    missing_datasets = [
        str(value) for value in dataset_files.values()
        if value is None or not value.is_file()
    ]
    if missing_datasets:
        raise FileNotFoundError(f"pilot datasets are missing: {missing_datasets}")
    dataset_identity = {}
    for name, value in dataset_files.items():
        stat = os.stat(value)
        dataset_identity[name] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "inode": stat.st_ino,
            "device": stat.st_dev,
        }
    identity_directories = {}
    for directory_value in raw.get("identity_directories", ()):
        directory = _resolve(root, directory_value)
        if directory is None or not directory.is_dir():
            raise FileNotFoundError(
                f"pilot identity directory is missing: {directory}"
            )
        for value in sorted(directory.rglob("*.yaml")):
            identity_directories[str(value.relative_to(root))] = file_sha256(value)

    pilot_input_identity = {}
    for pilot in pilots.values():
        for label, value in (
            ("baseline_run_config", pilot.baseline_run_config),
            ("baseline_checkpoint", pilot.baseline_checkpoint),
            ("actor_run_config", pilot.actor_run_config),
            ("actor_checkpoint", pilot.actor_checkpoint),
            ("trace", pilot.trace),
        ):
            if value is None:
                continue
            if not value.is_file():
                raise FileNotFoundError(f"pilot input is missing: {value}")
            pilot_input_identity[f"{pilot.name}:{label}"] = {
                "path": str(value),
                "sha256": file_sha256(value),
            }

    manifest_sha256 = _digest_mapping({
        "manifest": raw,
        "identity_files": {
            name: file_sha256(value) for name, value in identity_files.items()
        },
        "identity_directories": identity_directories,
        "pilot_inputs": pilot_input_identity,
        "dataset_files": dataset_identity,
    })

    return PilotManifest(
        path=path,
        repository_root=root,
        output_root=output_root,
        seed=int(raw["seed"]),
        levels=levels,
        promotion=promotion,
        replay=dict(raw["replay"]),
        pilots=pilots,
        sha256=manifest_sha256,
    )


def reshape_action_blocks(
    actions: torch.Tensor, *, source_block: int, target_block: int
) -> torch.Tensor:
    """Change Action Block granularity without changing physical action order."""
    if actions.ndim < 2:
        raise ValueError("actions must end in [horizon, action_dim]")
    source_block = int(source_block)
    target_block = int(target_block)
    if source_block < 1 or target_block < 1:
        raise ValueError("action block lengths must be positive")
    packed_dim = int(actions.shape[-1])
    if packed_dim % source_block:
        raise ValueError("action_dim is not divisible by source_block")
    base_dim = packed_dim // source_block
    physical_steps = int(actions.shape[-2]) * source_block
    if physical_steps % target_block:
        raise ValueError("physical action span is not divisible by target_block")
    flat = actions.reshape(*actions.shape[:-2], physical_steps, base_dim)
    return flat.reshape(
        *actions.shape[:-2], physical_steps // target_block, base_dim * target_block
    )


def build_launch_specs(
    manifest: PilotManifest, *, level: int
) -> tuple[LaunchSpec, ...]:
    """Build four isolated worker commands without launching GPU work."""
    if int(level) not in manifest.levels:
        raise ValueError(f"unknown pilot level {level}")
    specs = []
    for pilot in sorted(manifest.pilots.values(), key=lambda item: item.gpu):
        environment = {
            "CUDA_VISIBLE_DEVICES": str(pilot.gpu),
            "MUJOCO_EGL_DEVICE_ID": str(pilot.gpu),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "HYDRA_FULL_ERROR": "1",
        }
        command = (
            "python",
            "-u",
            "scripts/run_fast_lewam_pilot_worker.py",
            "--manifest",
            str(manifest.path),
            "--pilot",
            pilot.name,
            "--level",
            str(int(level)),
            "--direction",
            pilot.direction,
            "--device",
            "cuda:0",
        )
        specs.append(
            LaunchSpec(
                pilot=pilot.name,
                gpu=pilot.gpu,
                command=command,
                environment=environment,
                output_dir=pilot.output_dir / f"level_{int(level)}",
            )
        )
    return tuple(specs)


def validate_artifact_mapping(artifacts) -> bool:
    return bool(
        artifacts
        and all(
            Path(path).is_file() and file_sha256(Path(path)) == expected
            for path, expected in artifacts.items()
        )
    )


def valid_summary_payload(value, manifest, pilot, level, *, require_promote=False):
    return bool(
        value.get("schema_version") == 2
        and value.get("status") == "ok"
        and value.get("manifest_sha256") == manifest.sha256
        and value.get("pilot") == pilot.name
        and int(value.get("level", -1)) == int(level)
        and (not require_promote or value.get("decision") == "promote")
        and validate_artifact_mapping(value.get("artifact_identity", {}))
    )


def validate_action_training_budget(
    control_examples, variant_examples, control_span, variant_span
):
    values = tuple(map(int, (
        control_examples, variant_examples, control_span, variant_span
    )))
    if min(values) < 1:
        raise ValueError("action training budget values must be positive")
    if values[0] != values[1] or values[2] != values[3]:
        raise RuntimeError(
            "action-token control and variant training budgets differ: "
            f"examples={values[0]}/{values[1]}, "
            f"physical_span={values[2]}/{values[3]}"
        )
    return {
        "control_examples": values[0],
        "variant_examples": values[1],
        "control_physical_span": values[2],
        "variant_physical_span": values[3],
    }


def paired_success_delta_pp(baseline_result, pilot_result) -> tuple[float, float, float]:
    """Validate an exact paired cohort and return percentage-point quantities."""
    def cohort(result):
        return tuple(
            (row.get("dataset_episode"), row.get("start_step"))
            for row in result["episodes"]
        )

    if cohort(baseline_result) != cohort(pilot_result):
        raise ValueError("baseline and pilot evaluation cohorts differ")
    baseline = float(baseline_result["success_rate"])
    pilot = float(pilot_result["success_rate"])
    if not (0.0 <= baseline <= 100.0 and 0.0 <= pilot <= 100.0):
        raise ValueError("success rates must be percentages in [0, 100]")
    return baseline, pilot, pilot - baseline


def decide_level_one(
    metrics: Mapping[str, float], thresholds: PromotionThresholds
) -> str:
    """Return the manifest-defined Level-1 promotion decision."""
    success_delta = float(metrics["success_delta_pp"])
    if success_delta < -thresholds.maximum_success_regression_pp:
        return "stop"
    signal = (
        success_delta >= thresholds.success_gain_pp
        or float(metrics["spearman_delta"]) >= thresholds.spearman_gain
        or float(metrics["regret_relative_delta"])
        <= -thresholds.regret_reduction_fraction
    )
    return "promote" if signal else "stop"


__all__ = [
    "LaunchSpec",
    "Pilot",
    "PilotLevel",
    "PilotManifest",
    "PromotionThresholds",
    "build_launch_specs",
    "decide_level_one",
    "file_sha256",
    "load_pilot_manifest",
    "paired_success_delta_pp",
    "reshape_action_blocks",
    "valid_summary_payload",
    "validate_action_training_budget",
    "validate_artifact_mapping",
]
