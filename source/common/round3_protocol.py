"""Pure protocol rules for the Round 3 Phase 1 evaluation.

The evaluator in :mod:`source.common.eval` is deliberately coupled to the
installed ``stable-worldmodel`` runtime.  This module is the opposite seam:
it contains the frozen task geometry, cohort-independent statistics, and
serializable protocol constants.  Keeping those rules here makes it possible
to audit and test them without constructing a MuJoCo world.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np


ROUND3_PROTOCOL = "round3_phase1_v1"
PREDICATE_VERSION = "round3_runtime_predicates_v1"
COHORT_SCHEMA_VERSION = 1
TRACE_SCHEMA_VERSION = 1
ARTIFACT_REGISTRY_SCHEMA_VERSION = 1

TASKS = ("cube", "pusht", "reacher", "tworoom")
METHODS = ("e0_lewm", "e3_fast", "e5_fast")
PROTOCOL_VARIANTS = (
    "legacy",
    "sampling_revised",
    "tolerance_revised",
    "round3_revised",
)
FAST_STAGES = ("stage_a", "stage_a_shuffled_goal", "stage_b")

# These values are part of the Phase 1 protocol.  ``num_eval`` is supplied by
# the dev/final cohort (50 or 200), while all other values are fixed.
ROUND3_EVAL_DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "goal_offset_steps": 25,
    "eval_budget": 50,
    "horizon": 5,
    "receding_horizon": 5,
    "action_block": 5,
    "num_samples": 300,
    "n_steps": 30,
    "topk": 30,
    "var_scale": 1.0,
}


@dataclass(frozen=True)
class TaskPredicate:
    """A task's runtime success predicate and its audit metadata."""

    task: str
    current_field: str
    goal_field: str
    unit: str
    formula: str
    thresholds: Mapping[str, float]
    comparison: str
    continuous_hold: bool = False
    field_aliases: tuple[str, ...] = ()
    goal_aliases: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "current_field": self.current_field,
            "goal_field": self.goal_field,
            "unit": self.unit,
            "formula": self.formula,
            "thresholds": {
                str(key): float(value) for key, value in self.thresholds.items()
            },
            "comparison": self.comparison,
            "continuous_hold": self.continuous_hold,
            "field_aliases": list(self.field_aliases),
            "goal_aliases": list(self.goal_aliases),
        }


TASK_PREDICATES: dict[str, TaskPredicate] = {
    "cube": TaskPredicate(
        task="cube",
        current_field="privileged_block_0_pos",
        goal_field="goal_privileged_block_0_pos",
        unit="m",
        formula="||privileged_block_0_pos - goal_privileged_block_0_pos||_2 <= 0.04",
        thresholds={"position_l2": 0.04},
        comparison="less_than_or_equal",
        field_aliases=("privileged/block_0_pos", "block_0_pos"),
        goal_aliases=("goal_privileged/block_0_pos", "goal_block_0_pos"),
    ),
    "reacher": TaskPredicate(
        task="reacher",
        current_field="qpos",
        goal_field="goal_qpos",
        unit="rad",
        formula="all(abs(qpos - goal_qpos) < 0.05)",
        thresholds={"per_joint_abs": 0.05},
        comparison="strict_less_than",
        field_aliases=(),
        goal_aliases=(),
    ),
    "pusht": TaskPredicate(
        task="pusht",
        current_field="state",
        goal_field="goal_state",
        unit="px/rad",
        formula=(
            "||state[:4] - goal_state[:4]||_2 < 20 and "
            "circular_abs(state[4] - goal_state[4]) < pi/9"
        ),
        thresholds={"position_l2": 20.0, "angle_abs": float(np.pi / 9)},
        comparison="strict_less_than_for_each_component",
        field_aliases=(),
        goal_aliases=(),
    ),
    "tworoom": TaskPredicate(
        task="tworoom",
        current_field="proprio",
        goal_field="goal_proprio",
        unit="px",
        formula="||proprio - goal_proprio||_2 < 16",
        thresholds={"position_l2": 16.0},
        comparison="strict_less_than",
        field_aliases=("state",),
        goal_aliases=("goal_state",),
    ),
}


def _predicate_for(task: str) -> TaskPredicate:
    key = str(task).lower().replace("push-t", "pusht").replace("two_room", "tworoom")
    try:
        return TASK_PREDICATES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported Round 3 task {task!r}") from exc


def _batch_arrays(current: Any, goal: Any) -> tuple[np.ndarray, np.ndarray, bool]:
    current_arr = np.asarray(current, dtype=np.float64)
    goal_arr = np.asarray(goal, dtype=np.float64)
    scalar_input = current_arr.ndim == 1 and goal_arr.ndim == 1
    if current_arr.ndim == 0 or goal_arr.ndim == 0:
        raise ValueError("current and goal states must be at least one-dimensional")
    if current_arr.shape[-1] != goal_arr.shape[-1]:
        raise ValueError(
            "current and goal states have different final dimensions: "
            f"{current_arr.shape[-1]} != {goal_arr.shape[-1]}"
        )
    try:
        current_arr, goal_arr = np.broadcast_arrays(current_arr, goal_arr)
    except ValueError as exc:
        raise ValueError(
            f"current and goal states are not broadcastable: "
            f"{current_arr.shape} and {goal_arr.shape}"
        ) from exc
    if not np.all(np.isfinite(current_arr)) or not np.all(np.isfinite(goal_arr)):
        raise ValueError("current and goal states must contain finite values")
    return current_arr, goal_arr, scalar_input


def _circular_abs(angle: np.ndarray) -> np.ndarray:
    """Return the shortest absolute angular difference in radians."""
    return np.abs((angle + np.pi) % (2.0 * np.pi) - np.pi)


def evaluate_success(task: str, current: Any, goal: Any) -> bool | np.ndarray:
    """Evaluate the frozen runtime success predicate.

    A one-dimensional state returns a Python ``bool``; a leading batch
    dimension returns a boolean array.  The equality boundaries intentionally
    mirror the installed task implementations: Cube uses ``<=`` while the
    other three predicates use strict ``<``.
    """
    predicate = _predicate_for(task)
    current_arr, goal_arr, scalar_input = _batch_arrays(current, goal)
    diff = current_arr - goal_arr

    if predicate.task == "cube":
        result = np.linalg.norm(diff, axis=-1) <= predicate.thresholds["position_l2"]
    elif predicate.task == "reacher":
        result = np.all(
            np.abs(diff) < predicate.thresholds["per_joint_abs"], axis=-1
        )
    elif predicate.task == "pusht":
        if diff.shape[-1] < 5:
            raise ValueError("Push-T state must contain at least position and angle fields")
        position_distance = np.linalg.norm(diff[..., :4], axis=-1)
        angle_distance = _circular_abs(diff[..., 4])
        result = (position_distance < predicate.thresholds["position_l2"]) & (
            angle_distance < predicate.thresholds["angle_abs"]
        )
    else:
        result = np.linalg.norm(diff, axis=-1) < predicate.thresholds["position_l2"]

    if scalar_input:
        return bool(np.asarray(result).item())
    return np.asarray(result, dtype=bool)


def physical_distance(task: str, current: Any, goal: Any) -> float | np.ndarray:
    """Return the scalar distance used for start/terminal diagnostics.

    The distance is deliberately descriptive rather than a replacement for
    success.  For compatibility with the existing grounded diagnostics it is
    the Euclidean norm over the supplied state.  Push-T success additionally
    exposes position and circular-angle components through
    :func:`physical_distance_components`.
    """
    _predicate_for(task)
    current_arr, goal_arr, scalar_input = _batch_arrays(current, goal)
    result = np.linalg.norm(current_arr - goal_arr, axis=-1)
    if scalar_input:
        return float(np.asarray(result).item())
    return np.asarray(result, dtype=np.float64)


def physical_distance_components(
    task: str, current: Any, goal: Any
) -> dict[str, float | np.ndarray]:
    """Return task-specific diagnostic components alongside ``distance``."""
    current_arr, goal_arr, scalar_input = _batch_arrays(current, goal)
    diff = current_arr - goal_arr
    distance = np.linalg.norm(diff, axis=-1)
    components: dict[str, np.ndarray] = {"distance": distance}
    if _predicate_for(task).task == "pusht":
        if diff.shape[-1] < 5:
            raise ValueError("Push-T state must contain at least position and angle fields")
        components["position_distance"] = np.linalg.norm(diff[..., :4], axis=-1)
        components["angle_distance"] = _circular_abs(diff[..., 4])
    if scalar_input:
        return {key: float(np.asarray(value).item()) for key, value in components.items()}
    return components


def resolve_task_field(
    columns: Iterable[str], task: str, *, goal: bool = False
) -> str | None:
    """Resolve a canonical predicate field against dataset/runtime aliases."""
    predicate = _predicate_for(task)
    candidates = (predicate.goal_field, *predicate.goal_aliases) if goal else (
        predicate.current_field,
        *predicate.field_aliases,
    )
    available = {str(column) for column in columns}
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _runtime_sources(task: str) -> list[dict[str, str]]:
    """Locate the installed functions that define the task predicate.

    This is kept best-effort so a report can explicitly become ``unaccepted``
    when a future stable-worldmodel release changes its module layout.
    """
    sources: list[dict[str, str]] = []
    try:
        if task == "cube":
            from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

            objects = ((CubeEnv, "_compute_successes"), (CubeEnv, "set_target_pos"), (CubeEnv, "post_step"))
        elif task == "reacher":
            from stable_worldmodel.envs.dmcontrol.custom_tasks.reacher import (
                ReacherQPosMatchTask,
            )
            from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper

            objects = (
                (ReacherQPosMatchTask, "get_termination"),
                (ReacherDMControlWrapper, "set_target_qpos"),
                (ReacherDMControlWrapper, "_is_terminated"),
            )
        elif task == "pusht":
            from stable_worldmodel.envs.pusht.env import PushT

            objects = ((PushT, "eval_state"), (PushT, "_set_goal_state"))
        elif task == "tworoom":
            from stable_worldmodel.envs.two_room.env import TwoRoomEnv

            objects = ((TwoRoomEnv, "step"), (TwoRoomEnv, "_set_goal_state"))
        else:
            raise ValueError(task)
        for owner, name in objects:
            function = getattr(owner, name)
            source = inspect.getsource(function)
            sources.append(
                {
                    "object": f"{owner.__module__}.{owner.__qualname__}.{name}",
                    "source_sha256": _sha256_bytes(source.encode("utf-8")),
                    "source": source,
                }
            )
    except (ImportError, OSError, TypeError, AttributeError) as exc:
        sources.append({"error_type": type(exc).__name__, "error": str(exc)})
    return sources


def audit_predicate(task: str, *, goal_refresh: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a serializable audit record for one task's predicate."""
    predicate = _predicate_for(task)
    sources = _runtime_sources(predicate.task)
    accepted = bool(sources) and not any("error" in source for source in sources)
    record: dict[str, Any] = {
        "schema_version": 1,
        "protocol": ROUND3_PROTOCOL,
        "predicate_version": PREDICATE_VERSION,
        "task": predicate.task,
        "status": "accepted" if accepted else "unaccepted",
        "predicate": predicate.as_dict(),
        "task_mode": "qpos_match" if predicate.task == "reacher" else None,
        "runtime_formula": predicate.formula,
        "runtime_sources": sources,
        "goal_update_order": (
            "reset state, apply task goal setter, refresh the goal snapshot, "
            "then compute success after each environment step"
        ),
        "goal_field_refresh": (
            "the configured goal field is replaced with the selected target "
            "and must be visible to the next success computation"
        ),
        "continuous_hold": False,
        "goal_refresh": dict(goal_refresh) if goal_refresh is not None else {
            "status": "not_run",
            "checks_expected": 10,
        },
    }
    try:
        record["runtime_package_version"] = metadata.version("stable-worldmodel")
    except metadata.PackageNotFoundError:
        record["runtime_package_version"] = None
        record["status"] = "unaccepted"
    refresh = record["goal_refresh"]
    if goal_refresh is not None and (
        refresh.get("status") != "accepted" or int(refresh.get("checks_run", 0)) != 10
    ):
        record["status"] = "unaccepted"
    return record


def audit_all_predicates(
    *, goal_refresh: Mapping[str, Mapping[str, Any]] | None = None
) -> dict[str, dict[str, Any]]:
    """Audit all Phase 1 tasks without constructing an evaluator."""
    goal_refresh = goal_refresh or {}
    return {
        task: audit_predicate(task, goal_refresh=goal_refresh.get(task))
        for task in TASKS
    }


def write_predicate_audit(path: str | Path, task: str, *, goal_refresh=None) -> dict[str, Any]:
    """Write one task audit atomically and return its payload."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = audit_predicate(task, goal_refresh=goal_refresh)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return payload


def write_all_predicate_audits(
    output_dir: str | Path, *, goal_refresh: Mapping[str, Mapping[str, Any]] | None = None
) -> dict[str, dict[str, Any]]:
    """Write the four ``predicate_audit/{task}.json`` artifacts."""
    output = Path(output_dir)
    payloads = audit_all_predicates(goal_refresh=goal_refresh)
    for task, payload in payloads.items():
        target = output / f"{task}.json"
        temporary = target.with_suffix(target.suffix + ".tmp")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    return payloads


__all__ = [
    "ARTIFACT_REGISTRY_SCHEMA_VERSION",
    "COHORT_SCHEMA_VERSION",
    "FAST_STAGES",
    "METHODS",
    "PREDICATE_VERSION",
    "PROTOCOL_VARIANTS",
    "ROUND3_EVAL_DEFAULTS",
    "ROUND3_PROTOCOL",
    "TASKS",
    "TASK_PREDICATES",
    "TRACE_SCHEMA_VERSION",
    "TaskPredicate",
    "audit_all_predicates",
    "audit_predicate",
    "evaluate_success",
    "physical_distance",
    "physical_distance_components",
    "resolve_task_field",
    "write_all_predicate_audits",
    "write_predicate_audit",
]
