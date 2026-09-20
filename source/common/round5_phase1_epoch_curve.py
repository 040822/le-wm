"""Round 5 Phase 1 intermediate-epoch curve experiment helpers.

This module deliberately does not change the frozen Round 5 Phase 1 runner.
It owns the smaller, no-guidance condition matrix used to compare R4-AB
checkpoints from epochs 1 through 10.  Epoch 10 is represented by references
to the already-validated Phase 4.5-4 artifacts; epochs 1--9 are evaluated into
the independent epoch-curve output root.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .round3_phase1 import CohortManifest, wilson_interval
from .round3_validation import validate_result_payload
from .round5_phase1 import baseline_result_path, p2_protocol


CURVE_SCHEMA_VERSION = "round5_phase1_epoch_curve_v1"
CURVE_TASKS = ("cube", "pusht", "reacher", "tworoom")
CURVE_EPOCHS = tuple(range(1, 11))
CURVE_NEW_EPOCHS = tuple(range(1, 10))
CURVE_PROTOCOL_VARIANT = "legacy"
CURVE_INTEGRATOR = "euler"
CURVE_NON_CEM_PROTOCOL = "not_applicable"
CURVE_DEFAULT_EXPECTED_COUNTS = {
    "cube": 4,
    "pusht": 7,
    "reacher": 19,
    "tworoom": 4,
}
_CHECKPOINT_EPOCH_RE = re.compile(r"_weights_epoch_(?P<epoch>\d+)\.pt$")

ConditionKey = tuple[str, str, str, int | None, str, str]


def _default_condition_matrix() -> dict[str, list[dict[str, Any]]]:
    """Return the pre-registered 34-condition matrix."""

    flow_steps = (1, 2, 5, 10, 16, 32)
    matrix: dict[str, list[dict[str, Any]]] = {}
    for task in CURVE_TASKS:
        if task in {"cube", "tworoom"}:
            steps = (1,)
        elif task == "pusht":
            steps = (1, 2)
        else:
            steps = flow_steps
        protocol = p2_protocol(task)
        specs: list[dict[str, Any]] = []
        for mode in ("P0", "P2", "P3"):
            for step in steps:
                specs.append(
                    {
                        "mode": mode,
                        "cem_protocol": (
                            CURVE_NON_CEM_PROTOCOL
                            if mode in {"P0", "P3"}
                            else protocol
                        ),
                        "action_flow_steps": step,
                        "action_flow_integrator": CURVE_INTEGRATOR,
                        "guidance": "none",
                    }
                )
        # Keep P1 in the middle of the mode order when the matrix is rendered,
        # while retaining a stable mode order for the worker job list below.
        specs.insert(
            len(steps),
            {
                "mode": "P1",
                "cem_protocol": protocol,
                "action_flow_steps": None,
                "action_flow_integrator": "not_applicable",
                "guidance": "none",
            },
        )
        matrix[task] = specs
    # The insertion above puts P1 between P0 and P2, but for Pusht/Reacher it
    # also keeps P0's complete step block together, which is useful in logs.
    return matrix


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_condition(task: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and validate one config condition."""

    mode = str(raw.get("mode"))
    protocol = str(raw.get("cem_protocol"))
    step = raw.get("action_flow_steps")
    step = None if step is None else int(step)
    integrator = str(raw.get("action_flow_integrator", "euler"))
    guidance = str(raw.get("guidance", "none"))
    if mode not in {"P0", "P1", "P2", "P3"}:
        raise ValueError(f"unsupported epoch-curve mode for {task}: {mode!r}")
    if guidance != "none":
        raise ValueError("epoch-curve conditions must not use guidance")
    if mode == "P1":
        if step is not None or integrator != "not_applicable":
            raise ValueError("P1 must be invariant with not_applicable integrator")
        if protocol != p2_protocol(task):
            raise ValueError(
                f"P1/{task} must use cem_protocol={p2_protocol(task)!r}"
            )
    else:
        if step is None or step < 1 or integrator != CURVE_INTEGRATOR:
            raise ValueError(f"{mode} must use a positive Euler action-flow step")
        expected = (
            CURVE_NON_CEM_PROTOCOL
            if mode in {"P0", "P3"}
            else p2_protocol(task)
        )
        if protocol != expected:
            raise ValueError(f"{mode}/{task} must use cem_protocol={expected!r}")
    return {
        "task": task,
        "mode": mode,
        "cem_protocol": protocol,
        "action_flow_steps": step,
        "action_flow_integrator": integrator,
        "guidance": guidance,
    }


def condition_specs(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return the configured matrix and enforce the 4/7/19/4 contract."""

    raw_matrix = (
        config.get("condition_matrix") if isinstance(config, Mapping) else None
    )
    matrix = raw_matrix if isinstance(raw_matrix, Mapping) else _default_condition_matrix()
    conditions: list[dict[str, Any]] = []
    for task in CURVE_TASKS:
        raw_specs = matrix.get(task)
        if not isinstance(raw_specs, Sequence) or isinstance(raw_specs, (str, bytes)):
            raise ValueError(f"condition_matrix.{task} must be a list")
        normalized = [normalize_condition(task, item) for item in raw_specs]
        keys = [condition_key_from_spec(item) for item in normalized]
        if len(set(keys)) != len(keys):
            raise ValueError(f"condition_matrix.{task} contains duplicate conditions")
        expected_count = CURVE_DEFAULT_EXPECTED_COUNTS[task]
        if len(normalized) != expected_count:
            raise ValueError(
                f"condition_matrix.{task} has {len(normalized)} conditions; "
                f"expected {expected_count}"
            )
        conditions.extend(normalized)
    return conditions


def condition_key_from_spec(condition: Mapping[str, Any]) -> ConditionKey:
    return (
        str(condition["task"]),
        str(condition["mode"]),
        str(condition["cem_protocol"]),
        condition["action_flow_steps"],
        str(condition["action_flow_integrator"]),
        str(condition.get("guidance", "none")),
    )


def condition_name(condition: Mapping[str, Any]) -> str:
    step = (
        "invariant"
        if condition["action_flow_steps"] is None
        else f"step_{int(condition['action_flow_steps'])}"
    )
    return "/".join(
        (
            str(condition["mode"]),
            str(condition["cem_protocol"]),
            step,
            str(condition["action_flow_integrator"]),
        )
    )


def job_id(task: str, epoch: int, condition: Mapping[str, Any]) -> str:
    return f"{task}__epoch_{int(epoch):02d}__{condition_name(condition).replace('/', '__')}"


def job_specs(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    specs = condition_specs(config)
    jobs: list[dict[str, Any]] = []
    for epoch in CURVE_EPOCHS:
        for condition in specs:
            item = dict(condition)
            item["epoch"] = epoch
            item["job_id"] = job_id(str(condition["task"]), epoch, condition)
            item["kind"] = "reuse" if epoch == 10 else "evaluate"
            jobs.append(item)
    return jobs


def result_dir(
    root: str | Path,
    task: str,
    epoch: int,
    condition: Mapping[str, Any],
) -> Path:
    return (
        Path(root)
        / "conditions"
        / "legacy"
        / str(task)
        / f"epoch_{int(epoch):02d}"
        / condition_name(condition)
    )


def result_path(root: str | Path, task: str, epoch: int, condition: Mapping[str, Any]) -> Path:
    return result_dir(root, task, epoch, condition) / "result.json"


def reference_path(root: str | Path, task: str, epoch: int, condition: Mapping[str, Any]) -> Path:
    return result_dir(root, task, epoch, condition) / "reference.json"


def checkpoint_paths(
    config: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, dict[int, Path]]:
    """Resolve every declared epoch checkpoint and validate its filename epoch."""

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("config.training is missing")
    templates = training.get("checkpoint_patterns")
    if not isinstance(templates, Mapping):
        raise ValueError("config.training.checkpoint_patterns is missing")
    epoch_range = training.get("epoch_range", [1, 10])
    if not isinstance(epoch_range, Sequence) or len(epoch_range) != 2:
        raise ValueError("training.epoch_range must contain [first, last]")
    first, last = int(epoch_range[0]), int(epoch_range[1])
    if (first, last) != (1, 10):
        raise ValueError("epoch curve is frozen to epochs 1 through 10")
    result: dict[str, dict[int, Path]] = {}
    for task in CURVE_TASKS:
        try:
            template = str(templates[task])
        except KeyError as exc:
            raise ValueError(f"missing checkpoint pattern for {task}") from exc
        task_paths: dict[int, Path] = {}
        for epoch in CURVE_EPOCHS:
            path = _resolve(
                root,
                template.format(
                    task=task,
                    epoch=epoch,
                    seed=int(training.get("seed", 3072)),
                ),
            )
            match = _CHECKPOINT_EPOCH_RE.search(path.name)
            if match is None or int(match.group("epoch")) != epoch:
                raise ValueError(
                    f"checkpoint path for {task} epoch {epoch} does not encode that epoch: {path}"
                )
            if not path.is_file():
                raise FileNotFoundError(path)
            task_paths[epoch] = path.resolve()
        result[task] = task_paths
    return result


def checkpoint_hashes(paths: Mapping[str, Mapping[int, Path]]) -> dict[str, dict[int, str]]:
    return {
        task: {int(epoch): sha256_file(path) for epoch, path in epochs.items()}
        for task, epochs in paths.items()
    }


def validate_declared_checkpoint_hashes(
    config: Mapping[str, Any], hashes: Mapping[str, Mapping[int, str]]
) -> None:
    """Validate configured immutable hashes, currently required for epoch 10."""

    declared = config.get("training", {}).get("checkpoint_sha256", {})
    if not isinstance(declared, Mapping):
        raise ValueError("training.checkpoint_sha256 must be an object")
    for task in CURVE_TASKS:
        expected = declared.get(task, {})
        if not isinstance(expected, Mapping):
            raise ValueError(f"checkpoint_sha256.{task} must be an object")
        expected_epoch10 = expected.get("10")
        if expected_epoch10 is None:
            expected_epoch10 = expected.get(10)
        if str(expected_epoch10) != str(hashes[task][10]):
            raise ValueError(
                f"epoch 10 checkpoint hash mismatch for {task}: "
                f"{hashes[task][10]} != {expected_epoch10}"
            )


def baseline_source_path(
    config: Mapping[str, Any], task: str, condition: Mapping[str, Any], *, root: Path
) -> Path:
    reuse = config.get("epoch10_reuse")
    if not isinstance(reuse, Mapping):
        raise ValueError("config.epoch10_reuse is missing")
    baseline_root = _resolve(root, str(reuse["baseline_results_root"]))
    # baseline_result_path includes the frozen phase45-4 legacy/dev layout.
    return baseline_result_path(baseline_root, condition)


def _episode_identity(payload: Mapping[str, Any]) -> list[tuple[Any, Any, Any]]:
    return [
        (item.get("episode_id"), item.get("start_step"), item.get("row_index"))
        for item in payload.get("episodes", [])
    ]


def _expected_episode_identity(manifest: CohortManifest) -> list[tuple[Any, Any, Any]]:
    return [(entry.episode_id, entry.start_step, entry.row_index) for entry in manifest.entries]


def validate_result_payload_for_curve(
    payload: Mapping[str, Any],
    path: str | Path,
    *,
    manifest: CohortManifest,
    checkpoint: Path,
    checkpoint_sha256: str,
    epoch: int,
    condition: Mapping[str, Any],
) -> None:
    """Validate one evaluated or reused payload against the curve identity."""

    path = Path(path)
    if payload.get("status") != "ok":
        raise ValueError(f"result status is not ok: {path}")
    if int(payload.get("epoch", -1)) != int(epoch):
        raise ValueError(f"epoch mismatch in {path}")
    if payload.get("protocol_variant") != CURVE_PROTOCOL_VARIANT:
        raise ValueError(f"protocol variant mismatch in {path}")
    if payload.get("cohort_kind") != "dev":
        raise ValueError(f"epoch curve requires the legacy dev cohort: {path}")
    if str(payload.get("checkpoint")) != str(checkpoint.resolve()):
        raise ValueError(f"checkpoint path mismatch in {path}")
    recorded_hash = payload.get("checkpoint_sha256")
    if recorded_hash is not None and str(recorded_hash) != str(checkpoint_sha256):
        raise ValueError(f"checkpoint hash mismatch in result: {path}")
    if payload.get("cohort_id") != manifest.cohort_id:
        raise ValueError(f"cohort id mismatch in {path}")
    if payload.get("cohort_sha256") != manifest.computed_sha256:
        raise ValueError(f"cohort hash mismatch in {path}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(manifest.entries):
        raise ValueError(f"result must contain {len(manifest.entries)} episodes: {path}")
    if _episode_identity(payload) != _expected_episode_identity(manifest):
        raise ValueError(f"episode identity mismatch in {path}")
    expected_key = condition_key_from_spec(condition)
    from .round5_phase1 import condition_key

    if condition_key(payload) != expected_key:
        raise ValueError(
            f"condition mismatch in {path}: {condition_key(payload)} != {expected_key}"
        )
    validate_result_payload(
        payload,
        manifest=manifest,
        expected_count=len(manifest.entries),
        trace_path=payload.get("trace_path"),
    )


def row_from_payload(
    payload: Mapping[str, Any],
    *,
    task: str,
    epoch: int,
    condition: Mapping[str, Any],
    source: str,
    path: Path,
    checkpoint_sha256: str,
    source_result: Path | None = None,
) -> dict[str, Any]:
    successes = [bool(item.get("success", False)) for item in payload["episodes"]]
    low, high = wilson_interval(successes)
    planning = payload.get("round4_planning", {})
    return {
        "task": task,
        "epoch": int(epoch),
        "mode": str(condition["mode"]),
        "cem_protocol": str(condition["cem_protocol"]),
        "action_flow_steps": condition["action_flow_steps"],
        "action_flow_integrator": str(condition["action_flow_integrator"]),
        "guidance": "none",
        "source": source,
        "status": payload.get("status"),
        "episodes": len(successes),
        "successes": int(sum(successes)),
        "success_rate": float(sum(successes) / len(successes)),
        "success_rate_percent": float(sum(successes) / len(successes) * 100.0),
        "wilson_95_percent_low": float(low * 100.0),
        "wilson_95_percent_high": float(high * 100.0),
        "checkpoint": str(payload.get("checkpoint")),
        "checkpoint_sha256": checkpoint_sha256,
        "cohort_id": payload.get("cohort_id"),
        "cohort_sha256": payload.get("cohort_sha256"),
        "planning_mean_seconds": planning.get("planning_mean_seconds"),
        "planning_median_seconds": planning.get("planning_median_seconds"),
        "planning_p95_seconds": planning.get("planning_p95_seconds"),
        "forward_count": planning.get("forward_count"),
        "peak_memory_bytes": planning.get("peak_memory_bytes"),
        "path": str(path),
        "source_result": None if source_result is None else str(source_result),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def load_epoch_curve_results(
    config: Mapping[str, Any],
    output_root: str | Path,
    *,
    manifests: Mapping[str, CohortManifest],
    checkpoints: Mapping[str, Mapping[int, Path]],
    hashes: Mapping[str, Mapping[int, str]],
    root: Path,
    require_complete: bool = True,
) -> tuple[dict[tuple[int, ConditionKey], dict[str, Any]], dict[str, Any]]:
    """Load and validate evaluated rows plus epoch-10 reference rows."""

    output_root = Path(output_root)
    indexed: dict[tuple[int, ConditionKey], dict[str, Any]] = {}
    errors: list[str] = []
    jobs = job_specs(config)

    for job in jobs:
        task = str(job["task"])
        epoch = int(job["epoch"])
        condition = {key: value for key, value in job.items() if key in {
            "task", "mode", "cem_protocol", "action_flow_steps",
            "action_flow_integrator", "guidance",
        }}
        key = (epoch, condition_key_from_spec(condition))
        try:
            if epoch == 10:
                reference = reference_path(output_root, task, epoch, condition)
                if not reference.is_file() and not require_complete:
                    continue
                reference_payload = _load_json(reference)
                if reference_payload.get("status") != "ok":
                    raise ValueError(f"reference status is not ok: {reference}")
                source_result = Path(str(reference_payload["source_result"])).resolve()
                expected_source_hash = str(reference_payload["source_result_sha256"])
                if not source_result.is_file():
                    raise FileNotFoundError(source_result)
                observed_source_hash = sha256_file(source_result)
                if observed_source_hash != expected_source_hash:
                    raise ValueError(f"epoch 10 source result changed: {source_result}")
                payload = _load_json(source_result)
                validate_result_payload_for_curve(
                    payload,
                    source_result,
                    manifest=manifests[task],
                    checkpoint=checkpoints[task][epoch],
                    checkpoint_sha256=hashes[task][epoch],
                    epoch=epoch,
                    condition=condition,
                )
                if str(reference_payload.get("checkpoint_sha256")) != str(hashes[task][epoch]):
                    raise ValueError(f"epoch 10 reference checkpoint hash mismatch: {reference}")
                row = row_from_payload(
                    payload,
                    task=task,
                    epoch=epoch,
                    condition=condition,
                    source="epoch10_reuse",
                    path=source_result,
                    checkpoint_sha256=hashes[task][epoch],
                    source_result=source_result,
                )
                indexed[key] = {"payload": payload, "path": source_result, "row": row}
            else:
                result = result_path(output_root, task, epoch, condition)
                if not result.is_file() and not require_complete:
                    continue
                payload = _load_json(result)
                validate_result_payload_for_curve(
                    payload,
                    result,
                    manifest=manifests[task],
                    checkpoint=checkpoints[task][epoch],
                    checkpoint_sha256=hashes[task][epoch],
                    epoch=epoch,
                    condition=condition,
                )
                row = row_from_payload(
                    payload,
                    task=task,
                    epoch=epoch,
                    condition=condition,
                    source="evaluated",
                    path=result,
                    checkpoint_sha256=hashes[task][epoch],
                )
                indexed[key] = {"payload": payload, "path": result, "row": row}
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{job['job_id']}: {exc}")

    expected_keys = {
        (int(job["epoch"]), condition_key_from_spec(job)) for job in jobs
    }
    missing = sorted(expected_keys - set(indexed))
    if require_complete and missing:
        errors.append(f"missing {len(missing)} epoch-curve jobs")
    if errors:
        raise ValueError("invalid Round 5 epoch-curve results:\n" + "\n".join(errors[:40]))

    rows = [item["row"] for item in indexed.values()]
    validation = {
        "schema_version": CURVE_SCHEMA_VERSION,
        "result_count": len(indexed),
        "expected_count": len(jobs),
        "missing_count": len(missing),
        "missing": [list(item) for item in missing],
        "new_result_count": sum(row["source"] == "evaluated" for row in rows),
        "reuse_result_count": sum(row["source"] == "epoch10_reuse" for row in rows),
        "all_status_ok": all(row["status"] == "ok" for row in rows),
        "episodes_per_result": sorted({int(row["episodes"]) for row in rows}),
        "checkpoint_hashes": {
            task: {str(epoch): value for epoch, value in epoch_hashes.items()}
            for task, epoch_hashes in hashes.items()
        },
        "cohort_metadata": {
            task: {
                "cohort_id": manifests[task].cohort_id,
                "cohort_sha256": manifests[task].computed_sha256,
                "episodes": len(manifests[task].entries),
                "seed": manifests[task].seed,
                "goal_offset_steps": manifests[task].goal_offset_steps,
                "protocol_variant": manifests[task].protocol_variant,
            }
            for task in CURVE_TASKS
        },
        "output_root": str(output_root.resolve()),
        "baseline_results_root": str(
            _resolve(root, str(config["epoch10_reuse"]["baseline_results_root"])).resolve()
        ),
    }
    return indexed, validation


def dump_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


__all__ = [
    "CURVE_DEFAULT_EXPECTED_COUNTS",
    "CURVE_EPOCHS",
    "CURVE_INTEGRATOR",
    "CURVE_NEW_EPOCHS",
    "CURVE_NON_CEM_PROTOCOL",
    "CURVE_PROTOCOL_VARIANT",
    "CURVE_SCHEMA_VERSION",
    "CURVE_TASKS",
    "ConditionKey",
    "baseline_source_path",
    "checkpoint_hashes",
    "checkpoint_paths",
    "condition_key_from_spec",
    "condition_name",
    "condition_specs",
    "dump_json_atomic",
    "job_id",
    "job_specs",
    "load_epoch_curve_results",
    "reference_path",
    "result_dir",
    "result_path",
    "row_from_payload",
    "sha256_file",
    "validate_declared_checkpoint_hashes",
    "validate_result_payload_for_curve",
]
