"""Round 5 Phase 1.5 planning, artifact, and diagnosis primitives.

This module is deliberately independent from :mod:`round5_phase1`.  Phase 1.5
has a larger search matrix, variable CEM budgets, a fixed diagnostic candidate
pool, and several result types.  Keeping its identity and validation rules
here prevents a Phase 1 result from being silently treated as Phase 1.5
evidence.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .round3_phase1 import CohortManifest
from .round3_protocol import evaluate_success, physical_distance


PHASE15_SCHEMA_VERSION = "round5_phase1_5_v1"
PHASE15_RESULT_SCHEMA_VERSION = "round5_phase1_5_result_v1"
PHASE15_TASKS = ("cube", "pusht", "reacher", "tworoom")
PHASE15_FLOW_STEPS = (1, 2, 5, 10, 16, 32)
PHASE15_EVAL_SEED = 42
PHASE15_STABILITY_SEEDS = (43, 44)
PHASE15_MAX_CONDITIONS = 2_600
PHASE15_MAX_EPISODES = 130_000
PHASE15_MAX_CONCURRENT_SCANS = 4
PHASE15_BOOTSTRAP_SAMPLES = 10_000
PHASE15_COHORT_KIND = "dev"
PHASE15_PROTOCOL_VARIANT = "legacy"
PHASE15_GOAL_OFFSET = 25
PHASE15_EVAL_BUDGET = 50
PHASE15_HORIZON = 5
PHASE15_ACTION_BLOCK = 5
PHASE15_RECEDING_HORIZON = 5
PHASE15_EXECUTION_SEMANTICS = {
    "normalizer": "full_dataset_mean_std_float32",
    "action_clip": "task_protocol; reacher cem-clip only",
    "proposal_dtype": "float32",
    "verifier_dtype": "float32",
    "rng": "torch_generator_seeded_per_condition",
    "candidate_noise": "draw_full_candidate_set_before_chunking",
    "integrator": "euler",
    "success_reset": "disabled_for_diagnostic_branches",
}

Condition = dict[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize an identity object without platform-dependent whitespace."""
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def p2_protocol(task: str) -> str:
    """Phase1 legacy CEM clipping: only Reacher uses candidate clipping."""
    task = str(task).lower()
    if task not in PHASE15_TASKS:
        raise ValueError(f"unknown Phase1.5 task {task!r}")
    return "cem-clip" if task == "reacher" else "legacy"


def _base_condition(
    *,
    group: str,
    task: str,
    family: str,
    mode: str,
    flow_steps: int | None,
    candidate_count: int | None = None,
    po_iterations: int | None = None,
    guidance_step_size: float | None = None,
    max_rms_offset: float | None = None,
    guidance: str = "none",
    guidance_last_steps: int | None = None,
    cem_num_samples: int | None = None,
    cem_iterations: int | None = None,
    cem_elite_ratio: float | None = None,
    cem_var_scale: float | None = None,
    evaluation_seed: int = PHASE15_EVAL_SEED,
    candidate_semantics: str = "proposal_set",
) -> Condition:
    if task not in PHASE15_TASKS:
        raise ValueError(f"unknown Phase1.5 task {task!r}")
    protocol = "not_applicable" if mode in {"P0", "P3"} else p2_protocol(task)
    if mode in {"P0", "P3"}:
        bound = "none"
    else:
        bound = "candidate_clip" if protocol == "cem-clip" else "none"
    return {
        "schema_version": PHASE15_SCHEMA_VERSION,
        "group": str(group),
        "task": task,
        "family": str(family),
        "mode": str(mode),
        "cem_protocol": protocol,
        "action_bound_mode": bound,
        "flow_steps": None if flow_steps is None else int(flow_steps),
        "candidate_count": None if candidate_count is None else int(candidate_count),
        "po_iterations": None if po_iterations is None else int(po_iterations),
        "guidance": str(guidance),
        "guidance_step_size": (
            None if guidance_step_size is None else float(guidance_step_size)
        ),
        "max_rms_offset": None if max_rms_offset is None else float(max_rms_offset),
        "guidance_last_steps": (
            None if guidance_last_steps is None else int(guidance_last_steps)
        ),
        "cem_num_samples": (
            None if cem_num_samples is None else int(cem_num_samples)
        ),
        "cem_iterations": None if cem_iterations is None else int(cem_iterations),
        "cem_elite_ratio": (
            None if cem_elite_ratio is None else float(cem_elite_ratio)
        ),
        "cem_var_scale": None if cem_var_scale is None else float(cem_var_scale),
        "evaluation_seed": int(evaluation_seed),
        "candidate_semantics": str(candidate_semantics),
        "integrator": "euler",
    }


def _guidance_values() -> tuple[float, ...]:
    return (0.003, 0.01, 0.03)


def _po_grid() -> Iterator[tuple[int, float, float]]:
    for iterations in (1, 5, 10):
        for step_size in _guidance_values():
            for offset in (0.05, 0.2, 0.5):
                yield iterations, step_size, offset


def _condition_grid_primary() -> list[Condition]:
    """Build the 2,424 primary conditions in the plan's stable order."""
    conditions: list[Condition] = []

    # A1: N=1 is P0; all larger candidate sets are P3.
    for task in PHASE15_TASKS:
        for flow_steps in PHASE15_FLOW_STEPS:
            for candidate_count in (1, 4, 16, 64, 128, 256):
                mode = "P0" if candidate_count == 1 else "P3"
                conditions.append(
                    _base_condition(
                        group="A1",
                        task=task,
                        family="proposal_ranking",
                        mode=mode,
                        flow_steps=flow_steps,
                        candidate_count=candidate_count,
                        candidate_semantics=(
                            "single_action" if candidate_count == 1 else "proposal_set"
                        ),
                    )
                )

    # A2: optimize every P0 proposal, then use its original direct output.
    for task in PHASE15_TASKS:
        for flow_steps in (1, 2, 5, 16):
            for iterations, step_size, offset in _po_grid():
                conditions.append(
                    _base_condition(
                        group="A2",
                        task=task,
                        family="p0_post_opt",
                        mode="P0",
                        flow_steps=flow_steps,
                        candidate_count=1,
                        po_iterations=iterations,
                        guidance="post_opt",
                        guidance_step_size=step_size,
                        max_rms_offset=offset,
                    )
                )

    # A3: guided flow.  L is unique per S, so S=2 contributes {1,2}, etc.
    for task in PHASE15_TASKS:
        for flow_steps in (2, 5, 16):
            last_steps = tuple(dict.fromkeys((1, min(3, flow_steps), flow_steps)))
            for covered_steps in last_steps:
                for iterations, step_size, offset in _po_grid():
                    conditions.append(
                        _base_condition(
                            group="A3",
                            task=task,
                            family="p0_guided_flow",
                            mode="P0",
                            flow_steps=flow_steps,
                            candidate_count=1,
                            po_iterations=iterations,
                            guidance="guided_flow",
                            guidance_step_size=step_size,
                            max_rms_offset=offset,
                            guidance_last_steps=covered_steps,
                        )
                    )

    # A4: optimize all candidates, or select one and refine it.
    for task in PHASE15_TASKS:
        for family, guidance in (
            ("p3_post_opt", "post_opt"),
            ("p3_refine", "post_opt_refine"),
        ):
            for flow_steps in (1, 2):
                for candidate_count in (16, 64, 256):
                    for iterations in (1, 5, 10):
                        for step_size in _guidance_values():
                            conditions.append(
                                _base_condition(
                                    group="A4",
                                    task=task,
                                    family=family,
                                    mode="P3",
                                    flow_steps=flow_steps,
                                    candidate_count=candidate_count,
                                    po_iterations=iterations,
                                    guidance=guidance,
                                    guidance_step_size=step_size,
                                    max_rms_offset=0.2,
                                )
                            )

    # A5: P3 guided flow, with its fixed L=min(3,S).
    for task in PHASE15_TASKS:
        for flow_steps in (2, 5):
            for candidate_count in (16, 64):
                for iterations in (1, 5):
                    for step_size in _guidance_values():
                        conditions.append(
                            _base_condition(
                                group="A5",
                                task=task,
                                family="p3_guided_flow",
                                mode="P3",
                                flow_steps=flow_steps,
                                candidate_count=candidate_count,
                                po_iterations=iterations,
                                guidance="guided_flow",
                                guidance_step_size=step_size,
                                max_rms_offset=0.2,
                                guidance_last_steps=min(3, flow_steps),
                            )
                        )

    # A6: CEM budget sweep.  P1 has no flow S; P2 uses S as actor warm start.
    for task in PHASE15_TASKS:
        for mode in ("P1", "P2"):
            steps = (None,) if mode == "P1" else (1, 2, 5, 16)
            for flow_steps in steps:
                for cem_num_samples in (100, 300, 1000):
                    for cem_iterations in (1, 3, 10, 30):
                        conditions.append(
                            _base_condition(
                                group="A6",
                                task=task,
                                family="cem_budget",
                                mode=mode,
                                flow_steps=flow_steps,
                                candidate_count=cem_num_samples,
                                cem_num_samples=cem_num_samples,
                                cem_iterations=cem_iterations,
                                cem_elite_ratio=0.1,
                                cem_var_scale=1.0,
                                candidate_semantics="cem_random" if mode == "P1" else "cem_actor_warm_start",
                            )
                        )

    # A7: P2 guidance is warm-start guidance inside CEM. GF at S=1 is the
    # numerical alias of PO and is intentionally omitted.
    for task in PHASE15_TASKS:
        for guidance, steps in (("post_opt", (1, 2)), ("guided_flow", (2,))):
            for flow_steps in steps:
                for iterations in (1, 5):
                    for step_size in _guidance_values():
                        for cem_iterations in (3, 10, 30):
                            conditions.append(
                                _base_condition(
                                    group="A7",
                                    task=task,
                                    family="p2_guidance",
                                    mode="P2",
                                    flow_steps=flow_steps,
                                    candidate_count=300,
                                    po_iterations=iterations,
                                    guidance=guidance,
                                    guidance_step_size=step_size,
                                    max_rms_offset=0.2,
                                    guidance_last_steps=min(3, flow_steps),
                                    cem_num_samples=300,
                                    cem_iterations=cem_iterations,
                                    cem_elite_ratio=0.1,
                                    cem_var_scale=1.0,
                                    candidate_semantics="cem_actor_warm_start",
                                )
                            )
    return conditions


def primary_condition_specs() -> list[Condition]:
    specs = _condition_grid_primary()
    if len(specs) != 2_424:
        raise AssertionError(f"Phase1.5 primary grid generated {len(specs)} conditions")
    return specs


def grid_counts(specs: Sequence[Mapping[str, Any]] | None = None) -> dict[str, int]:
    specs = primary_condition_specs() if specs is None else list(specs)
    counts: dict[str, int] = {}
    for spec in specs:
        group = str(spec["group"])
        counts[group] = counts.get(group, 0) + 1
    expected = {"A1": 144, "A2": 432, "A3": 864, "A4": 432, "A5": 96, "A6": 240, "A7": 216}
    if specs == primary_condition_specs() and counts != expected:
        raise AssertionError(f"unexpected Phase1.5 group counts: {counts}")
    return counts


def condition_identity(
    spec: Mapping[str, Any],
    *,
    checkpoint: str,
    checkpoint_sha256: str,
    cohort: CohortManifest | Mapping[str, Any],
    execution_semantics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the full identity used for result reuse and task IDs."""
    manifest = (
        cohort
        if isinstance(cohort, CohortManifest)
        else CohortManifest.from_dict(cohort, verify_hash=True)
    )
    if manifest.protocol_variant != PHASE15_PROTOCOL_VARIANT:
        raise ValueError("Phase1.5 accepts legacy cohorts only")
    if manifest.cohort_kind != PHASE15_COHORT_KIND or len(manifest.entries) != 50:
        raise ValueError("Phase1.5 requires a legacy_50/dev cohort")
    return {
        "schema_version": PHASE15_SCHEMA_VERSION,
        "condition": _jsonable(dict(spec)),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": str(checkpoint_sha256),
        "cohort_id": manifest.cohort_id,
        "cohort_sha256": manifest.computed_sha256,
        "cohort_kind": manifest.cohort_kind,
        "protocol_variant": manifest.protocol_variant,
        "evaluation_seed": int(spec.get("evaluation_seed", PHASE15_EVAL_SEED)),
        "execution_semantics": _jsonable(
            dict(PHASE15_EXECUTION_SEMANTICS)
            if execution_semantics is None
            else dict(execution_semantics)
        ),
    }


def condition_id(identity: Mapping[str, Any]) -> str:
    return stable_sha256(identity)[:20]


def condition_slug(spec: Mapping[str, Any]) -> str:
    values = [str(spec.get("group")), str(spec.get("family")), str(spec.get("mode"))]
    if spec.get("flow_steps") is not None:
        values.append(f"s{spec['flow_steps']}")
    if spec.get("candidate_count") is not None:
        values.append(f"n{spec['candidate_count']}")
    if spec.get("cem_iterations") is not None:
        values.append(f"i{spec['cem_iterations']}")
    if spec.get("guidance") != "none":
        values.append(str(spec["guidance"]))
    return "_".join("".join(ch if ch.isalnum() else "-" for ch in value) for value in values)


def result_path(
    output_root: str | Path,
    spec: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> Path:
    return (
        Path(output_root)
        / "conditions"
        / str(spec["task"])
        / str(spec["group"])
        / f"{condition_slug(spec)}__{condition_id(identity)}"
        / f"seed_{int(spec.get('evaluation_seed', PHASE15_EVAL_SEED))}"
        / "result.json"
    )


def _atomic_replace_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    _atomic_replace_bytes(
        target,
        (json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    return target


@contextmanager
def condition_lock(result: str | Path) -> Iterator[Path]:
    """Acquire an exclusive sibling lock without overwriting an active worker.

    A killed evaluator can leave its lock file behind.  Reclaim only locks
    whose recorded owner PID is no longer alive; an active lock still fails
    immediately so two evaluators cannot write the same condition.
    """
    lock_path = Path(result).with_name(".condition.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    try:
        while fd is None:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError as exc:
                try:
                    content = lock_path.read_text(encoding="ascii").strip()
                    owner_pid = int(content.split("=", 1)[1].split()[0])
                except (OSError, ValueError, IndexError):
                    owner_pid = None
                if owner_pid is None or _pid_is_alive(owner_pid):
                    raise RuntimeError(f"condition is locked: {lock_path}") from exc
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
        os.write(fd, f"pid={os.getpid()}\n".encode("ascii"))
        yield lock_path
    except FileExistsError as exc:
        raise RuntimeError(f"condition is locked: {lock_path}") from exc
    finally:
        if fd is not None:
            os.close(fd)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def _pid_is_alive(pid: int) -> bool:
    if pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def phase15_scan_slot(
    output_root: str | Path,
    *,
    max_slots: int = PHASE15_MAX_CONCURRENT_SCANS,
) -> Iterator[int]:
    """Reserve one of the bounded Phase1.5 scan slots.

    A scan evaluates a whole group and can keep several CPU-heavy workers
    alive for hours.  The slot files make the four-route limit explicit across
    independently launched shell processes.  Dead owners are reclaimed on
    the next launch; active owners are never overwritten.
    """
    limit = int(max_slots)
    if limit < 1:
        raise ValueError("max_slots must be positive")
    if limit > PHASE15_MAX_CONCURRENT_SCANS:
        raise ValueError(
            f"max_slots cannot exceed the Phase1.5 safety limit of "
            f"{PHASE15_MAX_CONCURRENT_SCANS}"
        )
    directory = Path(output_root) / "locks" / "scan_slots"
    directory.mkdir(parents=True, exist_ok=True)
    acquired: tuple[Path, str] | None = None
    for index in range(limit):
        path = directory / f"slot_{index}.lock"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                content = path.read_text(encoding="ascii").strip()
                owner_pid = int(content.split(" ", 1)[0].split("=", 1)[1])
            except (OSError, ValueError, IndexError):
                owner_pid = -1
            if not _pid_is_alive(owner_pid):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                try:
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                except FileExistsError:
                    continue
            else:
                continue
        token = f"pid={os.getpid()} token={stable_sha256([os.getpid(), time.time_ns()])}\n"
        os.write(fd, token.encode("ascii"))
        os.close(fd)
        acquired = (path, token)
        break
    if acquired is None:
        raise RuntimeError(
            f"Phase1.5 scan concurrency limit reached: {limit}; "
            f"active slots are under {directory}"
        )
    path, token = acquired
    try:
        yield int(path.stem.removeprefix("slot_"))
    finally:
        try:
            if path.read_text(encoding="ascii") == token:
                path.unlink()
        except FileNotFoundError:
            pass


def publish_result(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Publish result and a manifest marker atomically after validation."""
    if payload.get("status") != "ok":
        raise ValueError("only status=ok payloads can be published as results")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 50:
        raise ValueError("Phase1.5 result must contain exactly 50 episodes")
    target = Path(path)
    with condition_lock(target):
        atomic_write_json(target, payload)
        atomic_write_json(
            target.with_name("complete.json"),
            {
                "schema_version": PHASE15_RESULT_SCHEMA_VERSION,
                "status": "completed",
                "result": str(target),
                "result_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            },
        )
    return target


def mark_infrastructure_failure(path: str | Path, error: BaseException) -> Path:
    """Record an infrastructure failure separately from an episode failure."""
    target = Path(path)
    payload = {
        "schema_version": PHASE15_RESULT_SCHEMA_VERSION,
        "status": "failed_infrastructure",
        "error_type": type(error).__name__,
        "error": str(error),
        "result": str(target),
    }
    atomic_write_json(target.with_name("infrastructure_failure.json"), payload)
    return target.with_name("infrastructure_failure.json")


def _episode_identity(record: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return record.get("episode_id", record.get("dataset_episode")), record.get("start_step"), record.get("row_index")


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON result must be an object: {path}")
    return value


def validate_phase15_result(
    payload: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    manifest: CohortManifest,
    require_episodes: bool = True,
) -> dict[str, Any]:
    """Validate identity, cohort, and complete-episode invariants."""
    if payload.get("status") != "ok":
        raise ValueError("result status is not ok")
    observed = payload.get("phase15_identity")
    if not isinstance(observed, Mapping):
        raise ValueError("phase15_identity is missing")
    if canonical_json(observed) != canonical_json(expected_identity):
        raise ValueError("phase15_identity does not match the planned condition")
    if payload.get("protocol_variant") != PHASE15_PROTOCOL_VARIANT:
        raise ValueError("result protocol_variant is not legacy")
    if payload.get("cohort_kind") != PHASE15_COHORT_KIND:
        raise ValueError("result cohort_kind is not dev")
    if payload.get("cohort_id") != manifest.cohort_id:
        raise ValueError("result cohort_id does not match the manifest")
    if payload.get("cohort_sha256") != manifest.computed_sha256:
        raise ValueError("result cohort_sha256 does not match the manifest")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("result episodes must be a list")
    if require_episodes and len(episodes) != 50:
        raise ValueError("result must contain exactly 50 episodes")
    if episodes:
        expected = [_episode_identity(entry.as_dict()) for entry in manifest.entries]
        observed_ids = [_episode_identity(item) for item in episodes]
        if observed_ids != expected:
            raise ValueError("result episode identity/order differs from the cohort")
        if len(set(observed_ids)) != len(observed_ids):
            raise ValueError("result contains duplicate episode identities")
        if any(type(item.get("success")) is not bool for item in episodes):
            raise ValueError("episode success values must be booleans")
    return {
        "episodes": len(episodes),
        "successes": int(sum(bool(item.get("success", False)) for item in episodes)),
        "success_rate": payload.get("success_rate"),
        "complete": len(episodes) == 50,
    }


def _identity_matches_legacy_payload(
    payload: Mapping[str, Any], spec: Mapping[str, Any], *, checkpoint: str, cohort: CohortManifest
) -> bool:
    """Conservative fallback for old artifacts that predate phase15_identity."""
    if payload.get("status") != "ok" or len(payload.get("episodes", ())) != 50:
        return False
    if payload.get("checkpoint") != str(Path(checkpoint).resolve()):
        return False
    if payload.get("cohort_sha256") != cohort.computed_sha256 or payload.get("cohort_kind") != "dev":
        return False
    if payload.get("protocol_variant") != "legacy":
        return False
    planning = payload.get("round4_planning", {})
    parameters = payload.get("parameters", {})
    if payload.get("task") != spec["task"] or payload.get("round4_mode") != spec["mode"]:
        return False
    if int(parameters.get("seed", -1)) != int(spec.get("evaluation_seed", PHASE15_EVAL_SEED)):
        return False
    if planning.get("action_flow_steps") != spec.get("flow_steps"):
        return False
    if planning.get("guidance_mode", "none") != spec.get("guidance", "none"):
        return False
    if planning.get("cem_protocol") != spec.get("cem_protocol"):
        return False
    # A legacy artifact cannot prove the new execution semantics.  Reuse only
    # when it explicitly recorded the fields that affect candidate identity.
    proposal = planning.get("proposal_chunk_size")
    if spec.get("guidance") != "none" and proposal is None:
        return False
    if spec.get("candidate_count") not in (None, 1) and planning.get("candidate_count") != spec.get("candidate_count"):
        return False
    if spec.get("po_iterations") is not None and planning.get("guidance_inner_steps") != spec.get("po_iterations"):
        return False
    if spec.get("guidance_step_size") is not None and planning.get("guidance_step_size") != spec.get("guidance_step_size"):
        return False
    if spec.get("max_rms_offset") is not None and planning.get("guidance_max_rms_offset") != spec.get("max_rms_offset"):
        return False
    if spec.get("guidance_last_steps") is not None and planning.get("guidance_last_steps") != spec.get("guidance_last_steps"):
        return False
    if spec.get("cem_num_samples") is not None:
        cem = parameters.get("cem") or {}
        if int(cem.get("num_samples", -1)) != int(spec["cem_num_samples"]):
            return False
    if spec.get("cem_iterations") is not None:
        cem = parameters.get("cem") or {}
        if int(cem.get("n_steps", -1)) != int(spec["cem_iterations"]):
            return False
    return True


def _legacy_fingerprint(payload: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """Extract the fields needed to locate an old Phase1/Phase4 artifact."""
    planning = payload.get("round4_planning")
    if not isinstance(planning, Mapping):
        return None
    parameters = payload.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {}
    cem = parameters.get("cem")
    if not isinstance(cem, Mapping):
        cem = {}
    mode = payload.get("round4_mode", payload.get("stage"))
    if mode not in {"P0", "P1", "P2", "P3"}:
        return None
    guidance = planning.get("guidance_mode", "none")
    flow = planning.get("action_flow_steps")
    candidate = planning.get("candidate_count")
    if candidate is None:
        candidate = 1 if mode == "P0" else 300 if mode in {"P1", "P2"} else 64
    return (
        payload.get("task"),
        mode,
        planning.get("cem_protocol"),
        None if flow is None else int(flow),
        guidance,
        parameters.get("seed"),
        int(candidate),
        cem.get("num_samples"),
        cem.get("n_steps"),
        planning.get("action_bound_mode"),
        planning.get("guidance_step_size") if guidance != "none" else None,
        planning.get("guidance_inner_steps") if guidance != "none" else None,
        planning.get("guidance_last_steps") if guidance != "none" else None,
        planning.get("guidance_max_rms_offset") if guidance != "none" else None,
    )


def _spec_fingerprint(spec: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        spec.get("task"),
        spec.get("mode"),
        spec.get("cem_protocol"),
        spec.get("flow_steps"),
        spec.get("guidance", "none"),
        spec.get("evaluation_seed", PHASE15_EVAL_SEED),
        spec.get("candidate_count") or (1 if spec.get("mode") == "P0" else 300 if spec.get("mode") in {"P1", "P2"} else 64),
        spec.get("cem_num_samples"),
        spec.get("cem_iterations"),
        spec.get("action_bound_mode"),
        spec.get("guidance_step_size") if spec.get("guidance", "none") != "none" else None,
        spec.get("po_iterations") if spec.get("guidance", "none") != "none" else None,
        spec.get("guidance_last_steps") if spec.get("guidance", "none") != "none" else None,
        spec.get("max_rms_offset") if spec.get("guidance", "none") != "none" else None,
    )


def index_phase15_results(
    output_root: str | Path,
    specs: Sequence[Mapping[str, Any]],
    *,
    identities: Mapping[str, Mapping[str, Any]],
    manifests: Mapping[str, CohortManifest],
    checkpoints: Mapping[str, str],
    history_roots: Sequence[str | Path] = (),
) -> dict[str, dict[str, Any]]:
    """Index current and historical artifacts with explicit reuse scope."""
    output_root = Path(output_root)
    # Result paths already carry task/mode/protocol/step/guidance.  Bucket
    # paths as they are discovered and only parse artifacts that could match a
    # planned spec. Stream the walk because history roots can contain very
    # large result trees, and retaining every Path object needlessly raises
    # peak memory during indexing.
    history_by_hint: dict[tuple[Any, ...], list[Path]] = {}
    planned_hints = {
        (
            spec.get("task"),
            spec.get("mode"),
            spec.get("cem_protocol"),
            spec.get("flow_steps"),
            spec.get("guidance", "none"),
        )
        for spec in specs
    }

    def path_hint(path: Path) -> tuple[Any, ...] | None:
        parts = path.parts
        try:
            marker = parts.index("conditions")
            values = list(parts[marker + 2 :])  # skip conditions/legacy
            task, mode, protocol = values[:3]
            if values[3].startswith("step_"):
                guidance = "none"
                step_value = values[3]
            else:
                guidance = values[3]
                step_value = values[4]
            flow = None if step_value == "invariant" else int(step_value.removeprefix("step_"))
            return task, mode, protocol, flow, guidance
        except (ValueError, IndexError, AttributeError):
            return None
    for root in history_roots:
        for path in Path(root).rglob("result.json"):
            hint = path_hint(path)
            if hint is not None and hint in planned_hints:
                history_by_hint.setdefault(hint, []).append(path)
    indexed: dict[str, dict[str, Any]] = {}
    for spec in specs:
        key = condition_id(identities[stable_sha256(dict(spec))])
        current = result_path(output_root, spec, identities[stable_sha256(dict(spec))])
        hint = (
            spec.get("task"),
            spec.get("mode"),
            spec.get("cem_protocol"),
            spec.get("flow_steps"),
            spec.get("guidance", "none"),
        )
        history_candidates = history_by_hint.get(hint, ())
        candidates: list[tuple[Path, Mapping[str, Any] | None]] = [(current, None)]
        candidates.extend((path, None) for path in history_candidates)
        selected: dict[str, Any] | None = None
        failures: list[str] = []
        current_identity_rejected = False
        for path, preloaded in candidates:
            if not path.is_file():
                continue
            try:
                payload = _read_json(path) if preloaded is None else preloaded
                identity = identities[stable_sha256(dict(spec))]
                if "phase15_identity" in payload:
                    validation = validate_phase15_result(
                        payload,
                        expected_identity=identity,
                        manifest=manifests[str(spec["task"])],
                        require_episodes=False,
                    )
                elif _identity_matches_legacy_payload(
                    payload,
                    spec,
                    checkpoint=checkpoints[str(spec["task"])],
                    cohort=manifests[str(spec["task"])],
                ):
                    validation = {
                        "episodes": len(payload.get("episodes", [])),
                        "successes": int(round(float(payload.get("success_rate", 0.0)) * 50)),
                        "success_rate": payload.get("success_rate"),
                        "complete": len(payload.get("episodes", [])) == 50,
                    }
                else:
                    raise ValueError("artifact identity or execution semantics mismatch")
                selected = {
                    "condition_id": key,
                    "spec": dict(spec),
                    "status": "completed" if validation["complete"] else "reused_success_only",
                    "reuse_scope": "full" if validation["complete"] else "success_only",
                    "source": "current" if path == current else "history",
                    "reuse_reason": (
                        "current_artifact_already_available"
                        if path == current
                        else "compatible_historical_artifact_reused_full"
                        if validation["complete"]
                        else "compatible_historical_artifact_reused_success_only"
                    ),
                    "historical_path_candidate_count": len(history_candidates),
                    "path": str(path),
                    "payload": payload,
                    "validation": validation,
                }
                if path == current:
                    break
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                # A history root contains many valid results for other
                # conditions.  Identity mismatches are expected during the
                # scan and mean "no reusable artifact", rather than failure.
                if str(exc) == "artifact identity or execution semantics mismatch":
                    if path == current:
                        current_identity_rejected = True
                    continue
                if path == current or "phase15_identity" in str(exc):
                    failures.append(f"{path}: {exc}")
        if selected is None:
            selected = {
                "condition_id": key,
                "spec": dict(spec),
                "status": "failed" if failures else "pending",
                "reuse_scope": None,
                "source": None,
                "reuse_reason": (
                    "artifact_validation_failed"
                    if failures
                    else "current_artifact_identity_mismatch"
                    if current_identity_rejected
                    else "historical_candidates_did_not_match_identity_and_execution_semantics"
                    if history_candidates
                    else "no_historical_candidate_for_condition"
                ),
                "historical_path_candidate_count": len(history_candidates),
                "path": str(current),
                "errors": failures,
            }
        indexed[key] = selected
    return indexed


def sampling_stability_specs(
    primary: Sequence[Mapping[str, Any]] | None = None,
) -> list[Condition]:
    """Return the fixed 96-condition seed43/44 stability extension.

    The optional ``primary`` argument is used by callers that want to verify
    that every extension is a member of the primary grid after removing its
    evaluation seed.
    """
    del primary
    result: list[Condition] = []
    for task in PHASE15_TASKS:
        fixed: list[Condition] = []
        def add(**kwargs: Any) -> None:
            fixed.append(_base_condition(task=task, evaluation_seed=42, **kwargs))
        for flow_steps in (1, 2):
            add(group="S1", family="proposal_ranking", mode="P0", flow_steps=flow_steps, candidate_count=1)
        for flow_steps in (1, 2):
            add(group="S1", family="proposal_ranking", mode="P3", flow_steps=flow_steps, candidate_count=64)
        for flow_steps in (1, 2):
            add(group="S1", family="p0_post_opt", mode="P0", flow_steps=flow_steps, candidate_count=1, po_iterations=5, guidance="post_opt", guidance_step_size=0.01, max_rms_offset=0.2)
        add(group="S1", family="p0_guided_flow", mode="P0", flow_steps=2, candidate_count=1, po_iterations=5, guidance="guided_flow", guidance_step_size=0.01, max_rms_offset=0.2, guidance_last_steps=2)
        add(group="S1", family="p3_post_opt", mode="P3", flow_steps=2, candidate_count=64, po_iterations=5, guidance="post_opt", guidance_step_size=0.01, max_rms_offset=0.2)
        add(group="S1", family="p3_refine", mode="P3", flow_steps=2, candidate_count=64, po_iterations=5, guidance="post_opt_refine", guidance_step_size=0.01, max_rms_offset=0.2)
        add(group="S1", family="cem_budget", mode="P1", flow_steps=None, candidate_count=300, cem_num_samples=300, cem_iterations=30, cem_elite_ratio=0.1, cem_var_scale=1.0, candidate_semantics="cem_random")
        for flow_steps in (1, 2):
            add(group="S1", family="cem_budget", mode="P2", flow_steps=flow_steps, candidate_count=300, cem_num_samples=300, cem_iterations=30, cem_elite_ratio=0.1, cem_var_scale=1.0, candidate_semantics="cem_actor_warm_start")
        for seed in PHASE15_STABILITY_SEEDS:
            for spec in fixed:
                copy = dict(spec)
                copy["evaluation_seed"] = seed
                result.append(copy)
    if len(result) != 96:
        raise AssertionError(f"fixed Phase1.5 stability grid generated {len(result)} conditions")
    return result


def adaptive_stability_specs(
    rows: Sequence[Mapping[str, Any]],
    *,
    fixed: Sequence[Mapping[str, Any]] | None = None,
    seeds: Sequence[int] = PHASE15_STABILITY_SEEDS,
) -> list[Condition]:
    """Choose the top two completed primary rows in each planned category.

    This is intentionally a post-scan operation: the plan chooses these
    configurations by success rate, then breaks ties by planning work.  Rows
    without a complete success rate are ignored, and fixed stability rows are
    removed before the seed43/44 extension is returned.
    """
    fixed_specs = list(fixed or sampling_stability_specs())
    used = {stable_sha256(dict(spec)) for spec in fixed_specs}
    categories = (
        ("ordinary_p3", lambda spec: spec.get("mode") == "P3" and spec.get("guidance", "none") == "none"),
        ("single_po", lambda spec: spec.get("mode") == "P0" and spec.get("guidance") == "post_opt"),
        ("single_gf", lambda spec: spec.get("mode") == "P0" and spec.get("guidance") == "guided_flow"),
        ("guided_p3", lambda spec: spec.get("mode") == "P3" and spec.get("guidance", "none") != "none"),
        ("cem", lambda spec: spec.get("mode") in {"P1", "P2"}),
    )
    result: list[Condition] = []
    for task in PHASE15_TASKS:
        task_rows = [row for row in rows if row.get("task") == task and row.get("status") in {"completed", "reused_success_only"}]
        for category, predicate in categories:
            candidates = []
            for row in task_rows:
                spec = row.get("spec")
                success_rate = row.get("success_rate")
                if not isinstance(spec, Mapping) or spec.get("task") != task or not predicate(spec) or success_rate is None:
                    continue
                planning = row.get("planning_forward_count") or row.get("forward_count") or 0
                backward = row.get("guidance_backward_count") or 0
                candidates.append((
                    -float(success_rate),
                    int(planning) + int(backward),
                    int(planning),
                    str(row.get("condition_id", "")),
                    dict(spec),
                ))
            for _, _, _, _, base in sorted(candidates)[:2]:
                for seed in seeds:
                    copy = dict(base)
                    copy["evaluation_seed"] = int(seed)
                    key = stable_sha256(copy)
                    if key in used:
                        continue
                    used.add(key)
                    copy["stability_category"] = category
                    result.append(copy)
    if len(result) > 80:
        raise AssertionError("adaptive Phase1.5 stability extension exceeds 80 conditions")
    return result


def safe_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Pearson correlation with the plan's undefined-constant convention."""
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("correlation inputs must be equally shaped vectors")
    if len(x) < 2:
        return None
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("correlation inputs must be finite")
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def normalized_physical_distance(task: str, current: Any, goal: Any) -> float | np.ndarray:
    """Compute the task-specific distance normalized by its success threshold."""
    current_array = np.asarray(current, dtype=np.float64)
    goal_array = np.asarray(goal, dtype=np.float64)
    if current_array.shape != goal_array.shape or current_array.ndim < 1:
        raise ValueError("current and goal must have matching non-empty arrays")
    diff = current_array - goal_array
    if task == "cube":
        result = np.linalg.norm(diff, axis=-1) / 0.04
    elif task == "reacher":
        result = np.max(np.abs(diff), axis=-1) / 0.05
    elif task == "pusht":
        if diff.shape[-1] < 5:
            raise ValueError("PushT state must contain position and angle fields")
        position = np.linalg.norm(diff[..., :4], axis=-1) / 20.0
        angle = np.abs((diff[..., 4] + np.pi) % (2 * np.pi) - np.pi) / (np.pi / 9)
        result = np.maximum(position, angle)
    elif task == "tworoom":
        result = np.linalg.norm(diff, axis=-1) / 16.0
    else:
        raise ValueError(f"unknown physical-distance task {task!r}")
    return float(result) if np.ndim(result) == 0 else result


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(value, dtype=np.float64)))))


def _unit_direction(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    value = rng.normal(size=shape)
    norm = _rms(value)
    return value / norm if norm > 0 else np.ones(shape, dtype=np.float64)


def make_control_actions(
    anchors: np.ndarray,
    *,
    seed: int = 2026,
    action_block: int = PHASE15_ACTION_BLOCK,
    include_anchors: bool = True,
    physical_zero: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Create the deterministic control actions from the fixed-pool protocol."""
    values = np.asarray(anchors, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError("anchors must have shape [anchor, time, action]")
    if action_block < 1:
        raise ValueError("action_block must be positive")
    rng = np.random.default_rng(int(seed))
    actions: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    def add(
        value: np.ndarray,
        kind: str,
        anchor: int | None,
        **details: Any,
    ) -> None:
        actions.append(np.asarray(value, dtype=np.float64))
        metadata.append({"kind": kind, "anchor": anchor, **details})
    if include_anchors:
        for index, anchor in enumerate(values):
            add(anchor, "anchor", index)
    for anchor_index, anchor in enumerate(values):
        for direction_index in range(8):
            direction = _unit_direction(rng, anchor.shape)
            for scale in (0.03, 0.1, 0.3):
                add(
                    anchor + direction * scale,
                    "rms_perturbation",
                    anchor_index,
                    direction=direction_index,
                    scale=float(scale),
                    sign=1,
                )
                add(
                    anchor - direction * scale,
                    "rms_perturbation",
                    anchor_index,
                    direction=direction_index,
                    scale=float(scale),
                    sign=-1,
                )
        blocks = max(1, values.shape[1] // int(action_block))
        block_view = anchor[: blocks * action_block].reshape(blocks, action_block, -1)
        permutations = (
            np.arange(blocks - 1, -1, -1),
            np.roll(np.arange(blocks), 1),
            rng.permutation(blocks),
        )
        for permutation_index, permutation in enumerate(permutations):
            transformed = anchor.copy()
            transformed[: blocks * action_block] = block_view[permutation].reshape(
                blocks * action_block, -1
            )
            add(
                transformed,
                "block_transform",
                anchor_index,
                permutation=permutation_index,
            )
    for index in range(64):
        add(
            _unit_direction(rng, values[0].shape),
            "standard_gaussian",
            None,
            sample=index,
        )
    physical_zero_value = np.zeros_like(values[0]) if physical_zero is None else np.asarray(physical_zero, dtype=np.float64)
    if physical_zero_value.shape != values[0].shape or not np.all(np.isfinite(physical_zero_value)):
        raise ValueError("physical_zero must be finite and match one anchor shape")
    add(physical_zero_value, "physical_zero", None)
    add(np.zeros_like(values[0]), "normalized_zero", None)
    return np.stack(actions, axis=0), metadata


def deduplicate_actions(actions: np.ndarray) -> tuple[np.ndarray, list[int]]:
    values = np.asarray(actions)
    if values.ndim < 2:
        raise ValueError("actions must contain an action dimension")
    seen: dict[bytes, int] = {}
    keep: list[int] = []
    duplicates: list[int] = []
    for index, action in enumerate(values):
        key = np.ascontiguousarray(action).tobytes()
        if key in seen:
            duplicates.append(index)
        else:
            seen[key] = index
            keep.append(index)
    return values[keep], duplicates


def control_action_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize physical outcomes for the fixed-pool control actions."""
    if not records:
        raise ValueError("control records cannot be empty")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        kind = str(record.get("control_kind", record.get("kind", "unknown")))
        if record.get("true_distance") is None:
            raise ValueError("control records must contain true_distance")
        grouped.setdefault(kind, []).append(record)
    by_kind: dict[str, Any] = {}
    for kind, rows in sorted(grouped.items()):
        distances = np.asarray([float(row["true_distance"]) for row in rows], dtype=np.float64)
        successes = np.asarray([bool(row.get("success", False)) for row in rows], dtype=np.float64)
        if not np.all(np.isfinite(distances)):
            raise ValueError("control true distances must be finite")
        by_kind[kind] = {
            "rows": int(len(rows)),
            "states": int(len({row.get("state_id", row.get("slot")) for row in rows})),
            "control_indices": int(len({row.get("control_index") for row in rows})),
            "success_rate": float(np.mean(successes)),
            "mean_true_distance": float(np.mean(distances)),
            "median_true_distance": float(np.median(distances)),
        }
    return {
        "rows": int(len(records)),
        "kinds": by_kind,
        "physical_zero_is_separate": "physical_zero" in by_kind,
        "normalized_zero_is_separate": "normalized_zero" in by_kind,
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def candidate_selection_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    task: str,
    random_draws: int = 128,
    seed: int = 42,
) -> dict[str, Any]:
    """Measure oracle, B-selected, and random candidate outcomes by state."""
    grouped: dict[Any, list[Mapping[str, Any]]] = {}
    for record in records:
        state_id = record.get("state_id", record.get("episode_id"))
        grouped.setdefault(state_id, []).append(record)
    if not grouped:
        raise ValueError("candidate records cannot be empty")
    rng = np.random.default_rng(int(seed))
    state_rows: list[dict[str, Any]] = []
    for state_id, candidates in grouped.items():
        distances = np.asarray(
            [
                float(record["true_distance"])
                if "true_distance" in record
                else float(
                    normalized_physical_distance(
                        task, record["physical_state"], record["goal_state"]
                    )
                )
                for record in candidates
            ],
            dtype=np.float64,
        )
        costs = np.asarray([float(record["predicted_cost"]) for record in candidates], dtype=np.float64)
        if not np.all(np.isfinite(distances)) or not np.all(np.isfinite(costs)):
            raise ValueError("candidate distances and predicted costs must be finite")
        latent_costs: np.ndarray | None = None
        candidate_latent_values = []
        for record in candidates:
            value = (
                record.get("milestones", {})
                .get("25", {})
                .get("future_latent_cost")
            )
            if value is None:
                candidate_latent_values = []
                break
            candidate_latent_values.append(float(value))
        if candidate_latent_values:
            latent_costs = np.asarray(candidate_latent_values, dtype=np.float64)
            if not np.all(np.isfinite(latent_costs)):
                raise ValueError("candidate latent costs must be finite")
        selected_index = int(np.argmin(costs))
        oracle_index = int(np.argmin(distances))
        random_values = np.asarray(
            [distances[rng.integers(0, len(distances), size=len(distances))].mean() for _ in range(int(random_draws))],
            dtype=np.float64,
        )
        successes = [
            bool(record["success"])
            if "success" in record
            else bool(evaluate_success(task, record["physical_state"], record["goal_state"]))
            for record in candidates
        ]
        selected_success = bool(successes[selected_index])
        random_success_rate = float(
            np.mean(
                [
                    successes[rng.integers(0, len(successes))]
                    for _ in range(int(random_draws))
                ]
            )
        )
        state_rows.append(
            {
                "state_id": state_id,
                "candidate_count": len(candidates),
                "oracle_distance": float(distances[oracle_index]),
                "selected_distance": float(distances[selected_index]),
                "selection_regret": float(distances[selected_index] - distances[oracle_index]),
                "oracle_success": bool(any(successes)),
                "selected_success": selected_success,
                "random_success_rate": random_success_rate,
                "selected_minus_random_success_rate": float(selected_success)
                - random_success_rate,
                "random_distance": float(np.mean(random_values)),
                "predicted_true_distance_correlation": safe_correlation(costs, distances),
                "selected_index": selected_index,
                "oracle_latent_cost": None if latent_costs is None else float(np.min(latent_costs)),
                "selected_latent_cost": None if latent_costs is None else float(latent_costs[selected_index]),
            }
        )
    def mean(field: str) -> float | None:
        values = [row[field] for row in state_rows if row.get(field) is not None]
        return None if not values else float(np.mean(values))
    correlations = [
        row["predicted_true_distance_correlation"]
        for row in state_rows
        if row["predicted_true_distance_correlation"] is not None
    ]
    return {
        "states": len(state_rows),
        "candidate_rows": len(records),
        "state_rows": state_rows,
        "oracle_success_rate": float(np.mean([row["oracle_success"] for row in state_rows])),
        "selected_success_rate": float(np.mean([row["selected_success"] for row in state_rows])),
        "random_success_rate": mean("random_success_rate"),
        "selected_minus_random_success_rate": mean(
            "selected_minus_random_success_rate"
        ),
        "oracle_best_physical_distance": mean("oracle_distance"),
        "selected_physical_distance": mean("selected_distance"),
        "selection_regret": mean("selection_regret"),
        "oracle_best_latent_cost": mean("oracle_latent_cost"),
        "selected_latent_cost": mean("selected_latent_cost"),
        "predicted_true_distance_correlation": (
            None if not correlations else float(np.mean(correlations))
        ),
    }


def _validate_candidate_pool(
    records: Sequence[Mapping[str, Any]],
    *,
    flow_steps: Sequence[int],
    candidates_per_flow_step: int,
    state_ids: Sequence[Any] | None = None,
) -> tuple[list[dict[str, Any]], tuple[Any, ...]]:
    """Validate the fixed candidate-pool contract before calculating metrics."""
    expected_flow_steps = tuple(int(value) for value in flow_steps)
    if not expected_flow_steps or len(set(expected_flow_steps)) != len(expected_flow_steps):
        raise ValueError("flow_steps must be a non-empty sequence of unique values")
    candidate_count = int(candidates_per_flow_step)
    if candidate_count < 1:
        raise ValueError("candidates_per_flow_step must be positive")
    normalized: list[dict[str, Any]] = []
    groups: dict[tuple[Any, int], list[int]] = {}
    observed_states: list[Any] = []
    observed_state_keys: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("candidate-pool records must be mappings")
        if "state_id" not in record or "flow_steps" not in record or "candidate_index" not in record:
            raise ValueError("candidate-pool records need state_id, flow_steps, and candidate_index")
        state_id = record["state_id"]
        state_key = canonical_json(state_id)
        flow = int(record["flow_steps"])
        candidate_index = int(record["candidate_index"])
        if flow not in expected_flow_steps:
            raise ValueError(f"unexpected candidate-pool flow_steps={flow}")
        if candidate_index < 0 or candidate_index >= candidate_count:
            raise ValueError(f"candidate_index {candidate_index} is outside the fixed pool")
        if state_key not in observed_state_keys:
            observed_state_keys.add(state_key)
            observed_states.append(state_id)
        if "predicted_cost" not in record:
            raise ValueError("candidate-pool records need predicted_cost")
        try:
            predicted_cost = float(record["predicted_cost"])
        except (TypeError, ValueError) as exc:
            raise ValueError("predicted_cost must be finite") from exc
        if not np.isfinite(predicted_cost):
            raise ValueError("predicted_cost must be finite")
        if "action" not in record:
            raise ValueError("candidate-pool records need the proposed action sequence")
        action = np.asarray(record["action"], dtype=np.float64)
        if action.ndim < 1 or action.size == 0 or not np.all(np.isfinite(action)):
            raise ValueError("candidate-pool action sequences must be finite and non-empty")
        normalized_record = dict(record)
        normalized_record["flow_steps"] = flow
        normalized_record["candidate_index"] = candidate_index
        normalized_record["predicted_cost"] = predicted_cost
        groups.setdefault((state_key, flow), []).append(candidate_index)
        normalized.append(normalized_record)
    expected_states = tuple(observed_states if state_ids is None else state_ids)
    expected_state_keys = {canonical_json(value) for value in expected_states}
    if not expected_state_keys:
        raise ValueError("candidate-pool records must contain at least one state")
    if observed_state_keys != expected_state_keys:
        raise ValueError("candidate-pool state_ids do not match the requested states")
    expected_indices = set(range(candidate_count))
    for state_key in expected_state_keys:
        for flow in expected_flow_steps:
            indices = groups.get((state_key, flow), [])
            if set(indices) != expected_indices or len(indices) != candidate_count:
                raise ValueError(
                    "candidate pool is incomplete or duplicated for "
                    f"state={state_key}, flow_steps={flow}"
                )
    return normalized, expected_states


def candidate_pool_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    task: str,
    flow_steps: Sequence[int] = PHASE15_FLOW_STEPS,
    candidates_per_flow_step: int = 256,
    state_ids: Sequence[Any] | None = None,
    random_draws: int = 128,
    bootstrap_samples: int = PHASE15_BOOTSTRAP_SAMPLES,
    seed: int = 2026,
) -> dict[str, Any]:
    """Summarize the fixed pool by flow step with state-clustered intervals.

    Every flow step is evaluated against the same state set and exactly the
    requested number of candidate action sequences.  Candidate rows are never
    treated as iid observations: all confidence intervals resample states.
    """
    normalized, expected_states = _validate_candidate_pool(
        records,
        flow_steps=flow_steps,
        candidates_per_flow_step=candidates_per_flow_step,
        state_ids=state_ids,
    )
    by_flow: dict[str, dict[str, Any]] = {}
    pooled_selection_gain_rows: list[dict[str, Any]] = []
    pooled_selection_regret_rows: list[dict[str, Any]] = []
    for offset, flow in enumerate(tuple(int(value) for value in flow_steps)):
        subset = [record for record in normalized if int(record["flow_steps"]) == flow]
        metrics = candidate_selection_metrics(
            subset,
            task=task,
            random_draws=random_draws,
            seed=int(seed) + offset,
        )
        state_rows = metrics["state_rows"]
        pooled_selection_gain_rows.extend(
            {
                "state_id": row["state_id"],
                "selected_minus_random_success_rate": row[
                    "selected_minus_random_success_rate"
                ],
            }
            for row in state_rows
        )
        pooled_selection_regret_rows.extend(
            {
                "state_id": row["state_id"],
                "selection_regret": row["selection_regret"],
            }
            for row in state_rows
        )
        bootstrap_fields = [
            "oracle_success",
            "selected_success",
            "random_success_rate",
            "selected_minus_random_success_rate",
            "selection_regret",
            "oracle_distance",
            "selected_distance",
        ]
        if all(row.get("predicted_true_distance_correlation") is not None for row in state_rows):
            bootstrap_fields.append("predicted_true_distance_correlation")
        if all(row.get("oracle_latent_cost") is not None for row in state_rows):
            bootstrap_fields.extend(("oracle_latent_cost", "selected_latent_cost"))
        metrics["bootstrap"] = {
            field: cluster_bootstrap(
                state_rows,
                field,
                samples=bootstrap_samples,
                seed=int(seed) + offset * 101 + index,
            )
            for index, field in enumerate(bootstrap_fields)
        }
        by_flow[str(flow)] = metrics
    pooled_selection_gain_bootstrap = cluster_bootstrap(
        pooled_selection_gain_rows,
        "selected_minus_random_success_rate",
        samples=bootstrap_samples,
        seed=int(seed) + 10_000,
    )
    pooled_selection_regret_bootstrap = cluster_bootstrap(
        pooled_selection_regret_rows,
        "selection_regret",
        samples=bootstrap_samples,
        seed=int(seed) + 10_001,
    )
    return {
        "schema_version": "round5_phase1_5_candidate_pool_metrics_v1",
        "task": str(task),
        "states": len(expected_states),
        "candidate_rows": len(normalized),
        "flow_steps": [int(value) for value in flow_steps],
        "candidates_per_flow_step": int(candidates_per_flow_step),
        "complete": True,
        "selection_regret": pooled_selection_regret_bootstrap["estimate"],
        "selected_minus_random_success_rate": pooled_selection_gain_bootstrap[
            "estimate"
        ],
        "bootstrap": {
            "selection_regret": pooled_selection_regret_bootstrap,
            "selected_minus_random_success_rate": pooled_selection_gain_bootstrap
        },
        "by_flow_steps": by_flow,
    }


def guidance_effect_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize predicted and real effects, including model exploitation."""
    if not records:
        raise ValueError("guidance records cannot be empty")
    input_records = len(records)
    records = [record for record in records if record.get("paired_outcome_valid", True)]
    if not records:
        return {
            "records": 0,
            "input_records": input_records,
            "excluded_noncomparable_records": input_records,
            "predicted_improvement_mean": None,
            "true_latent_improvement_mean": None,
            "true_physical_improvement_mean": None,
            "true_improvement_fraction": None,
            "true_degradation_fraction": None,
            "model_exploitation_fraction": None,
            "predicted_true_effect_correlation": None,
            "action_rms_displacement_mean": None,
            "action_saturation_fraction": None,
        }
    def value(record: Mapping[str, Any], name: str, guided_name: str) -> float:
        observed = record.get(name, record.get(guided_name))
        if observed is None:
            raise ValueError(f"guidance record is missing {name!r} and {guided_name!r}")
        return float(observed)

    predicted = np.asarray(
        [
            value(record, "predicted_cost_before", "guided_predicted_cost_before")
            - value(record, "predicted_cost_after", "guided_predicted_cost_after")
            for record in records
        ]
    )
    true = np.asarray(
        [
            value(record, "true_cost_before", "guided_true_cost_before")
            - value(record, "true_cost_after", "guided_true_cost_after")
            for record in records
        ]
    )
    latent = np.asarray(
        [
            true[index]
            if record.get("true_latent_improvement") is None
            else float(record["true_latent_improvement"])
            for index, record in enumerate(records)
        ],
        dtype=np.float64,
    )
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        state_id = record.get("state_id", f"record:{index}")
        grouped.setdefault(canonical_json(state_id), []).append(index)
    state_predicted: list[float] = []
    state_true: list[float] = []
    state_latent: list[float] = []
    state_true_positive: list[float] = []
    state_true_negative: list[float] = []
    state_exploitation: list[float] = []
    state_displacements: list[float] = []
    state_saturation: list[float] = []
    for indices in grouped.values():
        state_predicted.append(float(np.mean(predicted[indices])))
        state_true.append(float(np.mean(true[indices])))
        state_latent.append(float(np.mean(latent[indices])))
        state_true_positive.append(float(np.mean(true[indices] > 0.0)))
        state_true_negative.append(float(np.mean(true[indices] < 0.0)))
        state_exploitation.append(
            float(np.mean((predicted[indices] > 0.0) & (true[indices] < 0.0)))
        )
        displacements = np.asarray(
            [
                float(
                    records[index].get(
                        "action_rms_displacement",
                        records[index].get("guided_action_rms_displacement", math.nan),
                    )
                )
                for index in indices
            ],
            dtype=np.float64,
        )
        finite = displacements[np.isfinite(displacements)]
        if len(finite):
            state_displacements.append(float(finite.mean()))
        state_saturation.append(
            float(np.mean([float(records[index].get("action_saturation_fraction", 0.0)) for index in indices]))
        )
    state_predicted_array = np.asarray(state_predicted, dtype=np.float64)
    state_true_array = np.asarray(state_true, dtype=np.float64)
    return {
        "records": len(records),
        "input_records": input_records,
        "excluded_noncomparable_records": input_records - len(records),
        "states": len(grouped),
        "predicted_improvement_mean": float(state_predicted_array.mean()),
        "true_latent_improvement_mean": float(np.mean(state_latent)),
        "true_physical_improvement_mean": float(state_true_array.mean()),
        "true_improvement_fraction": float(np.mean(state_true_positive)),
        "true_degradation_fraction": float(np.mean(state_true_negative)),
        "model_exploitation_fraction": float(np.mean(state_exploitation)),
        "predicted_true_effect_correlation": (
            None
            if len(state_predicted_array) < 2
            else safe_correlation(state_predicted_array, state_true_array)
        ),
        "action_rms_displacement_mean": None
        if not state_displacements
        else float(np.mean(state_displacements)),
        "action_saturation_fraction": float(np.mean(state_saturation)),
    }


def paired_guidance_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = PHASE15_BOOTSTRAP_SAMPLES,
    seed: int = 2026,
    rms_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Compare a guidance update with its same-displacement random control."""
    if not records:
        raise ValueError("paired guidance records cannot be empty")
    input_records = len(records)
    records = [record for record in records if record.get("paired_outcome_valid", True)]
    if not records:
        return {
            "schema_version": "round5_phase1_5_paired_guidance_metrics_v1",
            "records": 0,
            "input_records": input_records,
            "excluded_noncomparable_records": input_records,
            "states": 0,
            "bootstrap": {},
            "state_rows": [],
        }
    required = (
        "state_id",
        "guided_predicted_cost_before",
        "guided_predicted_cost_after",
        "guided_true_cost_before",
        "guided_true_cost_after",
        "random_true_cost_before",
        "random_true_cost_after",
        "guided_action_rms_displacement",
        "random_action_rms_displacement",
    )
    rows: list[dict[str, Any]] = []
    for record in records:
        missing = [field for field in required if field not in record]
        if missing:
            raise ValueError(f"paired guidance record is missing {missing}")
        values = {field: float(record[field]) for field in required if field != "state_id"}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("paired guidance values must be finite")
        if not np.isclose(
            values["guided_action_rms_displacement"],
            values["random_action_rms_displacement"],
            rtol=0.0,
            atol=float(rms_tolerance),
        ):
            raise ValueError("guided and random controls must have the same RMS displacement")
        predicted_improvement = (
            values["guided_predicted_cost_before"]
            - values["guided_predicted_cost_after"]
        )
        guided_improvement = (
            values["guided_true_cost_before"] - values["guided_true_cost_after"]
        )
        random_improvement = (
            values["random_true_cost_before"] - values["random_true_cost_after"]
        )
        rows.append(
            {
                "state_id": record["state_id"],
                "predicted_improvement": predicted_improvement,
                "guided_true_improvement": guided_improvement,
                "random_true_improvement": random_improvement,
                "paired_advantage": guided_improvement - random_improvement,
                "model_exploitation": bool(predicted_improvement > 0.0 and guided_improvement < 0.0),
                "guided_action_rms_displacement": values["guided_action_rms_displacement"],
                "random_action_rms_displacement": values["random_action_rms_displacement"],
            }
        )
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped_rows.setdefault(canonical_json(row["state_id"]), []).append(row)
    state_rows: list[dict[str, Any]] = []
    averaged_fields = (
        "predicted_improvement",
        "guided_true_improvement",
        "random_true_improvement",
        "paired_advantage",
        "model_exploitation",
        "guided_action_rms_displacement",
        "random_action_rms_displacement",
    )
    for state_group in grouped_rows.values():
        state_rows.append(
            {
                "state_id": state_group[0]["state_id"],
                **{
                    field: float(
                        np.mean([float(row[field]) for row in state_group])
                    )
                    for field in averaged_fields
                },
            }
        )
    result = {
        "schema_version": "round5_phase1_5_paired_guidance_metrics_v1",
        "records": len(rows),
        "input_records": input_records,
        "excluded_noncomparable_records": input_records - len(rows),
        "states": len(state_rows),
        "predicted_improvement_mean": float(np.mean([row["predicted_improvement"] for row in state_rows])),
        "guided_true_improvement_mean": float(np.mean([row["guided_true_improvement"] for row in state_rows])),
        "random_true_improvement_mean": float(np.mean([row["random_true_improvement"] for row in state_rows])),
        "paired_advantage_mean": float(np.mean([row["paired_advantage"] for row in state_rows])),
        "guided_true_improvement_fraction": float(np.mean([row["guided_true_improvement"] > 0.0 for row in state_rows])),
        "random_true_improvement_fraction": float(np.mean([row["random_true_improvement"] > 0.0 for row in state_rows])),
        "model_exploitation_fraction": float(np.mean([row["model_exploitation"] for row in state_rows])),
        "bootstrap": {
            field: cluster_bootstrap(
                state_rows,
                field,
                samples=bootstrap_samples,
                seed=int(seed) + index,
            )
            for index, field in enumerate(
                ("guided_true_improvement", "random_true_improvement", "paired_advantage")
            )
        },
        "state_rows": state_rows,
    }
    latent_records = [
        record
        for record in records
        if all(
            field in record and record[field] is not None
            for field in (
                "guided_true_latent_cost_before",
                "guided_true_latent_cost_after",
                "random_true_latent_cost_before",
                "random_true_latent_cost_after",
            )
        )
    ]
    if latent_records:
        latent_by_state: dict[str, list[tuple[float, float]]] = {}
        latent_state_ids: dict[str, Any] = {}
        for record in latent_records:
            key = canonical_json(record["state_id"])
            latent_state_ids[key] = record["state_id"]
            latent_by_state.setdefault(key, []).append(
                (
                    float(record["guided_true_latent_cost_before"])
                    - float(record["guided_true_latent_cost_after"]),
                    float(record["random_true_latent_cost_before"])
                    - float(record["random_true_latent_cost_after"]),
                )
            )
        latent_state_rows = [
            {
                "state_id": latent_state_ids[key],
                "guided_latent_improvement": float(
                    np.mean([item[0] for item in rows_for_state])
                ),
                "random_latent_improvement": float(
                    np.mean([item[1] for item in rows_for_state])
                ),
                "paired_latent_advantage": float(
                    np.mean([item[0] - item[1] for item in rows_for_state])
                ),
            }
            for key, rows_for_state in latent_by_state.items()
        ]
        result.update(
            {
                "true_latent_records": len(latent_records),
                "guided_true_latent_improvement_mean": float(
                    np.mean([row["guided_latent_improvement"] for row in latent_state_rows])
                ),
                "random_true_latent_improvement_mean": float(
                    np.mean([row["random_latent_improvement"] for row in latent_state_rows])
                ),
                "paired_true_latent_advantage_mean": float(
                    np.mean([row["paired_latent_advantage"] for row in latent_state_rows])
                ),
                "paired_true_latent_advantage_bootstrap": cluster_bootstrap(
                    latent_state_rows,
                    "paired_latent_advantage",
                    samples=bootstrap_samples,
                    seed=int(seed) + 17,
                ),
            }
        )
    return result


def matched_guidance_mode_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = PHASE15_BOOTSTRAP_SAMPLES,
    seed: int = 2026,
    start_cost_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Compare GF and PO physical improvement on identical states/configs.

    Configuration cells are matched on flow steps, candidate index, inner
    steps, step size, and maximum RMS offset.  For each shared cell, only
    state IDs present in both modes with the same starting physical cost are
    paired.  Configuration cells are averaged within each state before the
    state-cluster bootstrap, so configurations and states are not treated as
    independent observations.
    """
    input_records = len(records)
    config_fields = (
        "flow_steps",
        "candidate_index",
        "guidance_inner_steps",
        "guidance_step_size",
        "max_rms_offset",
    )
    by_mode: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
        "post_opt": {},
        "guided_flow": {},
    }
    configs: dict[str, dict[str, Any]] = {}
    state_ids: dict[str, Any] = {}
    valid_records = 0
    excluded_invalid_records = 0
    for record in records:
        mode = record.get("guidance")
        if mode not in by_mode:
            excluded_invalid_records += 1
            continue
        if not record.get("paired_outcome_valid", True):
            excluded_invalid_records += 1
            continue
        required = (
            *config_fields,
            "state_id",
            "guided_true_cost_before",
            "guided_true_cost_after",
        )
        missing = [field for field in required if field not in record]
        if missing:
            raise ValueError(f"matched guidance record is missing {missing}")
        raw_config = {field: record[field] for field in config_fields}
        try:
            config = {
                "flow_steps": int(raw_config["flow_steps"]),
                "candidate_index": int(raw_config["candidate_index"]),
                "guidance_inner_steps": int(raw_config["guidance_inner_steps"]),
                "guidance_step_size": float(raw_config["guidance_step_size"]),
                "max_rms_offset": float(raw_config["max_rms_offset"]),
                "guidance": str(mode),
            }
            before = float(record["guided_true_cost_before"])
            after = float(record["guided_true_cost_after"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                "matched guidance configuration and physical costs must be numeric"
            ) from error
        numeric_values = (
            before,
            after,
            config["guidance_step_size"],
            config["max_rms_offset"],
        )
        if not all(np.isfinite(value) for value in numeric_values):
            excluded_invalid_records += 1
            continue
        config_key = canonical_json({field: config[field] for field in config_fields})
        state_key = canonical_json(record["state_id"])
        configs[config_key] = {field: config[field] for field in config_fields}
        state_ids[state_key] = record["state_id"]
        by_mode[str(mode)].setdefault(config_key, {}).setdefault(state_key, []).append(
            {"before": before, "improvement": before - after}
        )
        valid_records += 1

    state_comparisons: dict[str, list[dict[str, Any]]] = {}
    matched_configurations: list[dict[str, Any]] = []
    mismatched_start_pairs = 0
    matched_record_pairs = 0
    shared_config_keys = sorted(
        set(by_mode["post_opt"]) & set(by_mode["guided_flow"])
    )
    for config_index, config_key in enumerate(shared_config_keys):
        po_states = by_mode["post_opt"][config_key]
        gf_states = by_mode["guided_flow"][config_key]
        matched_states = 0
        config_state_rows: list[dict[str, Any]] = []
        for state_key in sorted(set(po_states) & set(gf_states)):
            po_rows = po_states[state_key]
            gf_rows = gf_states[state_key]
            po_before = float(np.mean([row["before"] for row in po_rows]))
            gf_before = float(np.mean([row["before"] for row in gf_rows]))
            if not np.isclose(
                po_before,
                gf_before,
                rtol=0.0,
                atol=float(start_cost_tolerance),
            ):
                mismatched_start_pairs += 1
                continue
            po_improvement = float(np.mean([row["improvement"] for row in po_rows]))
            gf_improvement = float(np.mean([row["improvement"] for row in gf_rows]))
            comparison_row = {
                "state_id": state_ids[state_key],
                "gf_minus_po_physical_improvement": gf_improvement - po_improvement,
                "guided_flow_physical_improvement": gf_improvement,
                "post_opt_physical_improvement": po_improvement,
            }
            state_comparisons.setdefault(state_key, []).append(comparison_row)
            config_state_rows.append(comparison_row)
            matched_states += 1
            matched_record_pairs += min(len(po_rows), len(gf_rows))
        if matched_states:
            matched_configurations.append(
                {
                    **configs[config_key],
                    "status": "ok" if matched_states >= 2 else "insufficient_matched_data",
                    "matched_states": matched_states,
                    "gf_minus_po_physical_improvement_mean": float(
                        np.mean([
                            row["gf_minus_po_physical_improvement"]
                            for row in config_state_rows
                        ])
                    ),
                    "bootstrap": (
                        {
                            "gf_minus_po_physical_improvement": cluster_bootstrap(
                                config_state_rows,
                                "gf_minus_po_physical_improvement",
                                samples=bootstrap_samples,
                                seed=int(seed) + config_index,
                            )
                        }
                        if matched_states >= 2
                        else {}
                    ),
                }
            )

    state_rows = [
        {
            "state_id": state_ids[state_key],
            "matched_configuration_count": len(values),
            **{
                field: float(np.mean([row[field] for row in values]))
                for field in (
                    "gf_minus_po_physical_improvement",
                    "guided_flow_physical_improvement",
                    "post_opt_physical_improvement",
                )
            },
        }
        for state_key, values in sorted(state_comparisons.items())
    ]
    result: dict[str, Any] = {
        "schema_version": "round5_phase1_5_matched_guidance_comparison_v1",
        "status": "ok" if len(state_rows) >= 2 else "insufficient_matched_data",
        "input_records": input_records,
        "valid_records": valid_records,
        "excluded_invalid_records": excluded_invalid_records,
        "mismatched_start_pairs": mismatched_start_pairs,
        "matched_configuration_count": len(matched_configurations),
        "matched_state_configuration_pairs": sum(
            len(rows) for rows in state_comparisons.values()
        ),
        "matched_record_pairs": matched_record_pairs,
        "matched_states": len(state_rows),
        "matched_configurations": matched_configurations,
        "state_rows": state_rows,
    }
    if state_rows:
        result.update(
            {
                "gf_minus_po_physical_improvement_mean": float(
                    np.mean([row["gf_minus_po_physical_improvement"] for row in state_rows])
                ),
                "guided_flow_physical_improvement_mean": float(
                    np.mean([row["guided_flow_physical_improvement"] for row in state_rows])
                ),
                "post_opt_physical_improvement_mean": float(
                    np.mean([row["post_opt_physical_improvement"] for row in state_rows])
                ),
            }
        )
    result["bootstrap"] = (
        {
            "gf_minus_po_physical_improvement": cluster_bootstrap(
                state_rows,
                "gf_minus_po_physical_improvement",
                samples=bootstrap_samples,
                seed=int(seed),
            )
        }
        if len(state_rows) >= 2
        else {}
    )
    return result


def matched_guidance_mode_metrics_by_milestone(
    records: Sequence[Mapping[str, Any]],
    *,
    milestone_steps: Sequence[int] = (5, 10, 15, 20, 25),
    bootstrap_samples: int = PHASE15_BOOTSTRAP_SAMPLES,
    seed: int = 2026,
) -> dict[str, dict[str, Any]]:
    """Compare matched PO/GF outcomes only at the same physical step."""
    result: dict[str, dict[str, Any]] = {}
    for index, milestone_step in enumerate(milestone_steps):
        step = int(milestone_step)
        step_records = [
            record
            for record in records
            if record.get("physical_comparison_step") is not None
            and int(record["physical_comparison_step"]) == step
        ]
        if not step_records:
            continue
        result[str(step)] = matched_guidance_mode_metrics(
            step_records,
            bootstrap_samples=bootstrap_samples,
            seed=int(seed) + index,
        )
    return result


def cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    value: str | Callable[[Sequence[Mapping[str, Any]]], float],
    *,
    cluster_key: str = "state_id",
    samples: int = PHASE15_BOOTSTRAP_SAMPLES,
    seed: int = 2026,
) -> dict[str, Any]:
    """Bootstrap state clusters so candidate rows never act as iid samples."""
    if not rows:
        raise ValueError("bootstrap rows cannot be empty")
    grouped: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.get(cluster_key), []).append(row)
    keys = list(grouped)
    if int(samples) < 1:
        raise ValueError("bootstrap samples must be positive")
    if callable(value):
        cluster_values = np.asarray(
            [float(value(grouped[key])) for key in keys], dtype=np.float64
        )
    else:
        cluster_values = np.asarray(
            [
                np.mean([float(row[value]) for row in grouped[key]])
                for key in keys
            ],
            dtype=np.float64,
        )
    estimate = float(cluster_values.mean())
    rng = np.random.default_rng(int(seed))
    chosen = rng.integers(0, len(keys), size=(int(samples), len(keys)))
    draws = cluster_values[chosen].mean(axis=1)
    return {
        "estimate": estimate,
        "cluster_count": len(keys),
        "row_count": len(rows),
        "samples": int(samples),
        "seed": int(seed),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def make_probe_split(
    trajectory_ids: Sequence[Any],
    *,
    eval_trajectory_ids: Iterable[Any] = (),
    seed: int = 2026,
    max_trajectories: int = 1_000,
    train_fraction: float = 0.8,
) -> dict[str, tuple[Any, ...]]:
    """Create a deterministic trajectory-level probe split with no leakage."""
    unique = list(dict.fromkeys(trajectory_ids))
    eval_keys = set(eval_trajectory_ids)
    if eval_keys.intersection(unique):
        raise ValueError("probe trajectories overlap evaluation trajectories")
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    rng = np.random.default_rng(int(seed))
    if len(unique) > int(max_trajectories):
        unique = [unique[int(index)] for index in rng.choice(len(unique), int(max_trajectories), replace=False)]
    permutation = rng.permutation(len(unique))
    shuffled = [unique[int(index)] for index in permutation]
    split = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * float(train_fraction)))) if len(shuffled) > 1 else len(shuffled))
    return {"train": tuple(shuffled[:split]), "validation": tuple(shuffled[split:]), "excluded_eval": tuple(eval_keys)}


def _probe_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "count": int(len(target)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "per_dimension_mae": np.mean(np.abs(error), axis=0).tolist(),
    }


def fit_ridge_probe(
    features: np.ndarray,
    targets: np.ndarray,
    trajectory_ids: Sequence[Any],
    *,
    split: Mapping[str, Sequence[Any]],
    alphas: Sequence[float] = tuple(10.0 ** power for power in range(-4, 3)),
    seed: int = 2026,
) -> dict[str, Any]:
    """Fit a standardized-target Ridge probe with trajectory validation."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    # The encoder already returns float32 latents.  Keeping the probe design
    # matrix in that dtype avoids doubling its memory footprint on large
    # trajectory-disjoint samples.
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    ids = np.asarray(trajectory_ids, dtype=object)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(ids) != len(x):
        raise ValueError("probe features, targets, and trajectory_ids are mis-shaped")
    train_ids, validation_ids = set(split["train"]), set(split["validation"])
    if train_ids.intersection(validation_ids):
        raise ValueError("probe train and validation trajectories overlap")
    train_mask = np.asarray([item in train_ids for item in ids])
    validation_mask = np.asarray([item in validation_ids for item in ids])
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("probe split must select train and validation rows")
    x_scaler = StandardScaler().fit(x[train_mask])
    y_scaler = StandardScaler().fit(y[train_mask])
    x_train = x_scaler.transform(x[train_mask])
    x_validation = x_scaler.transform(x[validation_mask])
    y_train = y_scaler.transform(y[train_mask])
    best = None
    for alpha in alphas:
        model = Ridge(alpha=float(alpha), random_state=int(seed))
        model.fit(x_train, y_train)
        prediction = y_scaler.inverse_transform(model.predict(x_validation))
        score = float(np.sqrt(np.mean(np.square(prediction - y[validation_mask]))))
        if best is None or score < best[0]:
            best = (score, float(alpha), model)
    assert best is not None
    model = best[2]
    validation_prediction = y_scaler.inverse_transform(model.predict(x_validation))
    train_prediction = y_scaler.inverse_transform(model.predict(x_train))
    shuffled_targets = y[train_mask].copy()
    np.random.default_rng(int(seed)).shuffle(shuffled_targets, axis=0)
    shuffled_model = Ridge(alpha=best[1], random_state=int(seed)).fit(x_train, y_scaler.transform(shuffled_targets))
    shuffled_prediction = y_scaler.inverse_transform(shuffled_model.predict(x_scaler.transform(x[validation_mask])))
    mean_prediction = np.broadcast_to(np.mean(y[train_mask], axis=0), y[validation_mask].shape)
    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "alpha": best[1],
        "train_metrics": _probe_metrics(y[train_mask], train_prediction),
        "validation_metrics": _probe_metrics(y[validation_mask], validation_prediction),
        "mean_baseline_metrics": _probe_metrics(y[validation_mask], mean_prediction),
        "shuffled_label_metrics": _probe_metrics(y[validation_mask], shuffled_prediction),
        "validation_prediction": validation_prediction,
        "validation_target": y[validation_mask],
        "validation_ids": ids[validation_mask].tolist(),
    }


def circular_angle_error(prediction: Sequence[float], target: Sequence[float]) -> np.ndarray:
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return np.abs((difference + np.pi) % (2 * np.pi) - np.pi)


def synchronous_timing(
    fn: Callable[[], Any],
    *,
    warmup: int,
    runs: int,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Measure a complete inference callable after explicit warm-up."""
    if warmup < 0 or runs < 1:
        raise ValueError("warmup must be non-negative and runs must be positive")
    synchronize = synchronize or (lambda: None)
    for _ in range(int(warmup)):
        fn()
    synchronize()
    durations: list[float] = []
    for _ in range(int(runs)):
        started = time.perf_counter()
        fn()
        synchronize()
        durations.append(time.perf_counter() - started)
    values = np.asarray(durations, dtype=np.float64)
    return {
        "warmup": int(warmup),
        "runs": int(runs),
        "samples_seconds": values.tolist(),
        "p50_seconds": float(np.quantile(values, 0.5)),
        "p95_seconds": float(np.quantile(values, 0.95)),
        "mean_seconds": float(np.mean(values)),
        "batch50_throughput_per_second": float(50.0 / np.mean(values)),
    }


def summarize_timing_samples(
    samples: Sequence[float],
    *,
    warmup: int,
    runs: int,
    source: str = "synchronized_samples",
) -> dict[str, Any] | None:
    """Summarize a fixed timing window from already synchronized samples.

    This helper is used for scan artifacts whose timing events were collected
    during a full evaluation.  Callers should label that retrospective source
    explicitly; it does not turn mixed evaluation wall time into planning time.
    """
    if warmup < 0 or runs < 1:
        raise ValueError("warmup must be non-negative and runs must be positive")
    values = np.asarray(samples, dtype=np.float64)
    values = values[np.isfinite(values)]
    end = int(warmup) + int(runs)
    if len(values) < end:
        return None
    window = values[int(warmup) : end]
    return {
        "warmup": int(warmup),
        "runs": int(runs),
        "samples_seconds": window.tolist(),
        "p50_seconds": float(np.quantile(window, 0.50)),
        "p95_seconds": float(np.quantile(window, 0.95)),
        "mean_seconds": float(np.mean(window)),
        "batch50_throughput_per_second": float(50.0 / np.mean(window)),
        "source": str(source),
    }


def _environment_objects(env: Any) -> list[Any]:
    """Return known environment/simulator objects without following arbitrary fields."""
    queue = [env]
    result: list[Any] = []
    visited: set[int] = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        result.append(current)
        for name in ("unwrapped", "env", "dmc_env", "_env", "physics", "sim"):
            try:
                nested = getattr(current, name, None)
            except Exception:
                continue
            if nested is not None and nested is not current:
                queue.append(nested)
    return result


def _mujoco_model_data(env: Any) -> tuple[Any, Any] | None:
    """Find a MuJoCo model/data pair through Gymnasium or dm_control wrappers."""
    for current in _environment_objects(env):
        for model_name, data_name in (("_model", "_data"), ("model", "data")):
            try:
                model = getattr(current, model_name, None)
                data = getattr(current, data_name, None)
            except Exception:
                continue
            if model is None or data is None:
                continue
            model_ptr = getattr(model, "ptr", model)
            data_ptr = getattr(data, "ptr", data)
            if model_ptr is not None and data_ptr is not None:
                return model_ptr, data_ptr
    return None


def _pymunk_space(env: Any) -> tuple[Any, Any, list[Any]] | None:
    for current in _environment_objects(env):
        try:
            space = getattr(current, "space", None)
            bodies = list(space.bodies) if space is not None else []
        except Exception:
            continue
        if space is not None and bodies and callable(
            getattr(space, "reindex_shapes_for_body", None)
        ):
            return current, space, bodies
    return None


def _variation_values(space: Any) -> Any:
    children = getattr(space, "spaces", None)
    if isinstance(children, Mapping):
        return {key: _variation_values(child) for key, child in children.items()}
    if not hasattr(space, "value"):
        return None
    value = getattr(space, "value")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return deepcopy(value)


def _pymunk_reset_options(owner: Any, variations: Any) -> dict[str, Any] | None:
    """Capture the PushT reset inputs needed to rebuild its Pymunk space."""
    reset = getattr(owner, "reset", None)
    if not callable(reset) or not isinstance(variations, Mapping):
        return None
    agent = variations.get("agent")
    block = variations.get("block")
    goal_state = getattr(owner, "goal_state", None)
    if (
        not isinstance(agent, Mapping)
        or not isinstance(block, Mapping)
        or goal_state is None
    ):
        return None
    try:
        # PushT.reset steps the new space while placing the goal and start state.
        # Replaying its pre-step state preserves the same Pymunk contact setup.
        initial_state = np.concatenate(
            (
                np.asarray(agent["start_position"], dtype=np.float64).reshape(-1),
                np.asarray(block["start_position"], dtype=np.float64).reshape(-1),
                np.asarray(block["angle"], dtype=np.float64).reshape(1),
                np.asarray(agent["velocity"], dtype=np.float64).reshape(-1),
            )
        )
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "variation_values": deepcopy(variations),
        "goal_state": deepcopy(goal_state),
        "state": initial_state,
    }


def _navigation_position_state(env: Any) -> tuple[Any, Any, Any] | None:
    for current in _environment_objects(env):
        if not (
            hasattr(current, "agent_position")
            and hasattr(current, "target_position")
            and callable(getattr(current, "_set_state", None))
            and callable(getattr(current, "_set_goal_state", None))
        ):
            continue
        def as_numpy(value: Any) -> np.ndarray:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value).copy()

        variation_space = getattr(current, "variation_space", None)
        variations = None if variation_space is None else _variation_values(variation_space)
        return (
            current,
            as_numpy(current.agent_position),
            {
                "target_position": as_numpy(current.target_position),
                "variations": variations,
            },
        )
    return None


def _dm_control_task(env: Any) -> Any | None:
    for current in _environment_objects(env):
        try:
            task = getattr(current, "task", None)
        except Exception:
            continue
        if task is not None:
            return task
    return None


def _dm_control_task_random(env: Any) -> Any | None:
    task = _dm_control_task(env)
    if task is None:
        return None
    try:
        random_state = getattr(task, "random", None)
        if random_state is None:
            random_state = getattr(task, "_random", None)
    except Exception:
        return None
    if random_state is not None:
        if callable(getattr(random_state, "get_state", None)) and callable(
            getattr(random_state, "set_state", None)
        ):
            return random_state
    return None


def capture_rng_state(obj: Any) -> Any:
    numpy_states: list[dict[str, Any]] = []
    seen_generators: set[int] = set()
    for current in _environment_objects(obj):
        for name in ("np_random", "rng"):
            try:
                rng = getattr(current, name, None)
                bit_generator = getattr(rng, "bit_generator", None)
            except Exception:
                continue
            if bit_generator is not None and id(bit_generator) not in seen_generators:
                seen_generators.add(id(bit_generator))
                numpy_states.append(
                    {"attribute": name, "state": deepcopy(bit_generator.state)}
                )
    task_random = _dm_control_task_random(obj)
    task_state = (
        None if task_random is None else deepcopy(task_random.get_state())
    )
    if not numpy_states and task_state is None:
        return None
    return {
        "kind": "round5_environment_rng_v1",
        "numpy_generators": numpy_states,
        "dm_control_task_random": task_state,
    }


def restore_rng_state(obj: Any, state: Any) -> None:
    if state is None:
        return
    # Accept snapshots from the earlier bit-generator-only implementation.
    if not isinstance(state, Mapping) or state.get("kind") != "round5_environment_rng_v1":
        for current in _environment_objects(obj):
            try:
                rng = getattr(current, "np_random", None)
                bit_generator = getattr(rng, "bit_generator", None)
            except Exception:
                continue
            if bit_generator is not None:
                bit_generator.state = deepcopy(state)
                return
        raise ValueError("object does not expose np_random.bit_generator.state")

    numpy_states = state.get("numpy_generators")
    if numpy_states is None and state.get("numpy_generator") is not None:
        numpy_states = [{"attribute": "np_random", "state": state["numpy_generator"]}]
    if numpy_states:
        current_objects = _environment_objects(obj)
        seen_generators: set[int] = set()
        available: list[Any] = []
        for current in current_objects:
            for name in ("np_random", "rng"):
                try:
                    rng = getattr(current, name, None)
                    bit_generator = getattr(rng, "bit_generator", None)
                except Exception:
                    continue
                if bit_generator is not None and id(bit_generator) not in seen_generators:
                    seen_generators.add(id(bit_generator))
                    available.append(bit_generator)
        if len(available) < len(numpy_states):
            raise ValueError("environment lost one or more NumPy generators")
        for bit_generator, saved in zip(available, numpy_states):
            bit_generator.state = deepcopy(saved["state"])

    task_state = state.get("dm_control_task_random")
    if task_state is not None:
        task_random = _dm_control_task_random(obj)
        if task_random is None:
            raise ValueError("environment lost its dm_control task RNG")
        task_random.set_state(deepcopy(task_state))


def capture_environment_state(env: Any) -> dict[str, Any]:
    """Capture a restorable simulator state for diagnostic branch replay."""
    target = getattr(env, "unwrapped", env)
    getter = getattr(target, "get_state", None)
    setter = getattr(target, "set_state", None)
    if callable(getter) and callable(setter):
        return {"kind": "get_set_state", "state": deepcopy(getter())}
    sim = getattr(target, "sim", None)
    sim_getter = getattr(sim, "get_state", None)
    sim_setter = getattr(sim, "set_state", None)
    if callable(sim_getter) and callable(sim_setter):
        return {"kind": "sim_state", "state": deepcopy(sim_getter())}

    model_data = _mujoco_model_data(target)
    if model_data is not None:
        try:
            import mujoco

            model, data = model_data
            signature = mujoco.mjtState.mjSTATE_INTEGRATION
            size = int(mujoco.mj_stateSize(model, signature))
            state = np.empty(size, dtype=np.float64)
            mujoco.mj_getState(model, data, state, signature)
            task = _dm_control_task(target)
            task_state = (
                {"target_qpos": deepcopy(task.target_qpos)}
                if task is not None and hasattr(task, "target_qpos")
                else None
            )
            return {
                "kind": "mujoco_integration_state",
                "state_signature": int(signature),
                "state": state,
                "dm_control_task": task_state,
            }
        except (ImportError, AttributeError, TypeError, ValueError):
            pass

    pymunk = _pymunk_space(target)
    if pymunk is not None:
        owner, space, bodies = pymunk
        body_fields = (
            "position",
            "velocity",
            "force",
            "angle",
            "angular_velocity",
            "torque",
            "center_of_gravity",
            "is_sleeping",
        )
        body_states = []
        for body in bodies:
            body_state: dict[str, Any] = {}
            for name in body_fields:
                if not hasattr(body, name):
                    continue
                value = getattr(body, name)
                if name in {"position", "velocity", "force", "center_of_gravity"}:
                    value = np.asarray(tuple(value), dtype=np.float64)
                elif name != "is_sleeping":
                    value = float(value)
                else:
                    value = bool(value)
                body_state[name] = value
            body_states.append(body_state)
        space_state = {}
        for name in ("gravity", "damping", "iterations"):
            if not hasattr(space, name):
                continue
            value = getattr(space, name)
            space_state[name] = (
                np.asarray(tuple(value), dtype=np.float64)
                if name == "gravity"
                else deepcopy(value)
            )
        environment_fields = {}
        for name in ("goal_state", "goal_pose", "_goal"):
            if hasattr(owner, name):
                value = getattr(owner, name)
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                environment_fields[name] = deepcopy(value)
        variation_space = getattr(owner, "variation_space", None)
        variations = (
            None if variation_space is None else _variation_values(variation_space)
        )
        snapshot = {
            "kind": "pymunk_body_state",
            "bodies": body_states,
            "space": space_state,
            "environment_fields": environment_fields,
            "variations": variations,
        }
        reset_options = _pymunk_reset_options(owner, variations)
        if reset_options is not None:
            snapshot["pymunk_reset_options"] = reset_options
        return snapshot

    navigation = _navigation_position_state(target)
    if navigation is not None:
        _, agent_position, remaining = navigation
        return {
            "kind": "navigation_position_state",
            "agent_position": agent_position,
            **remaining,
        }

    data = getattr(target, "_data", getattr(target, "data", None))
    model = getattr(target, "_model", getattr(target, "model", None))
    if data is not None and hasattr(data, "qpos") and hasattr(data, "qvel"):
        snapshot: dict[str, Any] = {
            "kind": "mujoco_data",
            "qpos": np.asarray(data.qpos).copy(),
            "qvel": np.asarray(data.qvel).copy(),
        }
        for name in ("act", "ctrl", "time"):
            if hasattr(data, name):
                snapshot[name] = np.asarray(getattr(data, name)).copy()
        if model is not None:
            snapshot["model"] = model
        return snapshot
    raise ValueError("environment does not expose a supported restorable simulator state")


def restore_environment_state(env: Any, snapshot: Mapping[str, Any]) -> None:
    """Restore a snapshot produced by :func:`capture_environment_state`."""
    target = getattr(env, "unwrapped", env)
    kind = snapshot.get("kind")
    if kind == "get_set_state":
        setter = getattr(target, "set_state", None)
        if not callable(setter):
            raise ValueError("environment lost its set_state method")
        setter(deepcopy(snapshot["state"]))
        return
    if kind == "sim_state":
        sim = getattr(target, "sim", None)
        setter = getattr(sim, "set_state", None)
        if not callable(setter):
            raise ValueError("environment lost its simulator set_state method")
        setter(deepcopy(snapshot["state"]))
        forward = getattr(sim, "forward", None)
        if callable(forward):
            forward()
        return
    if kind == "mujoco_integration_state":
        model_data = _mujoco_model_data(target)
        if model_data is None:
            raise ValueError("environment lost its MuJoCo model/data objects")
        try:
            import mujoco

            model, data = model_data
            signature = mujoco.mjtState(int(snapshot["state_signature"]))
            state = np.asarray(snapshot["state"], dtype=np.float64)
            mujoco.mj_setState(model, data, state, signature)
            mujoco.mj_forward(model, data)
            task_state = snapshot.get("dm_control_task")
            if task_state is not None:
                task = _dm_control_task(target)
                if task is None or not hasattr(task, "target_qpos"):
                    raise ValueError("environment lost its dm_control task state")
                task.target_qpos = deepcopy(task_state["target_qpos"])
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            raise ValueError("could not restore the MuJoCo integration state") from exc
        return
    if kind == "pymunk_body_state":
        pymunk = _pymunk_space(target)
        if pymunk is None:
            raise ValueError("environment lost its Pymunk space")
        owner, space, bodies = pymunk
        body_states = snapshot.get("bodies")
        if not isinstance(body_states, list) or len(body_states) != len(bodies):
            raise ValueError("environment Pymunk body count changed")
        reset_options = snapshot.get("pymunk_reset_options")
        reset = getattr(owner, "reset", None)
        if isinstance(reset_options, Mapping) and callable(reset):
            # Rebuild geometry and solver state through the environment's own
            # reset path; assigning body coordinates alone loses both.
            reset(options=deepcopy(reset_options))
            pymunk = _pymunk_space(target)
            if pymunk is None:
                raise ValueError("environment lost its Pymunk space after reset")
            owner, space, bodies = pymunk
            if len(body_states) != len(bodies):
                raise ValueError("environment Pymunk body count changed after reset")
        variation_space = getattr(owner, "variation_space", None)
        expected_variations = snapshot.get("variations")
        if expected_variations is not None and variation_space is not None:
            if canonical_json(_variation_values(variation_space)) != canonical_json(
                expected_variations
            ):
                raise ValueError("environment Pymunk variations could not be restored")
        for body, body_state in zip(bodies, body_states):
            activate = getattr(body, "activate", None)
            if callable(activate):
                activate()
            for name in (
                "position",
                "velocity",
                "force",
                "angle",
                "angular_velocity",
                "torque",
                "center_of_gravity",
            ):
                if name in body_state and hasattr(body, name):
                    value = body_state[name]
                    if name in {"position", "velocity", "force", "center_of_gravity"}:
                        value = tuple(np.asarray(value, dtype=np.float64).tolist())
                    setattr(body, name, value)
            if body_state.get("is_sleeping"):
                sleep = getattr(body, "sleep", None)
                if callable(sleep):
                    sleep()
            space.reindex_shapes_for_body(body)
        for name, value in snapshot.get("space", {}).items():
            if hasattr(space, name):
                setattr(
                    space,
                    name,
                    tuple(np.asarray(value, dtype=np.float64).tolist())
                    if name == "gravity"
                    else deepcopy(value),
                )
        for name, value in snapshot.get("environment_fields", {}).items():
            if hasattr(owner, name):
                setattr(owner, name, deepcopy(value))
        return
    if kind == "navigation_position_state":
        navigation = _navigation_position_state(target)
        if navigation is None:
            raise ValueError("environment lost its navigation position state")
        owner = navigation[0]
        variation_space = getattr(owner, "variation_space", None)
        variations = snapshot.get("variations")
        if variations is not None and variation_space is not None:
            setter = getattr(variation_space, "set_value", None)
            if not callable(setter):
                raise ValueError("environment lost its variation-space setter")
            setter(deepcopy(variations))
        cache_params = getattr(owner, "_cache_params", None)
        if callable(cache_params):
            cache_params()
        owner._set_state(np.asarray(snapshot["agent_position"]).copy())
        owner._set_goal_state(np.asarray(snapshot["target_position"]).copy())
        return
    if kind == "mujoco_data":
        data = getattr(target, "_data", getattr(target, "data", None))
        if data is None:
            raise ValueError("environment lost its MuJoCo data object")
        for name in ("qpos", "qvel", "act", "ctrl", "time"):
            if name in snapshot and hasattr(data, name):
                value = np.asarray(snapshot[name])
                getattr(data, name)[...] = value
        model = snapshot.get("model", getattr(target, "_model", getattr(target, "model", None)))
        try:
            import mujoco

            mujoco.mj_forward(model, data)
        except (ImportError, TypeError):
            forward = getattr(data, "forward", None)
            if callable(forward):
                forward()
        return
    raise ValueError(f"unknown environment state snapshot kind: {kind!r}")


def verify_state_restore(
    make_env: Callable[[], Any],
    *,
    state: Any,
    action: Any,
    step: Callable[[Any, Any], Any] | None = None,
) -> dict[str, Any]:
    """Check deterministic replay while restoring simulator and RNG state."""
    step = step or (lambda env, value: env.step(value))
    observations: list[Any] = []
    for _ in range(2):
        env = make_env()
        try:
            if hasattr(env, "set_state"):
                env.set_state(deepcopy(state))
            rng_state = capture_rng_state(env)
            first = step(env, deepcopy(action))
            restore_rng_state(env, rng_state)
            if hasattr(env, "set_state"):
                env.set_state(deepcopy(state))
            second = step(env, deepcopy(action))
            observations.append((first, second))
        finally:
            if hasattr(env, "close"):
                env.close()
    equal = canonical_json(observations[0][0]) == canonical_json(observations[0][1])
    return {"deterministic": bool(equal), "comparisons": 1}


__all__ = [
    "Condition",
    "PHASE15_ACTION_BLOCK",
    "PHASE15_BOOTSTRAP_SAMPLES",
    "PHASE15_EVAL_BUDGET",
    "PHASE15_EVAL_SEED",
    "PHASE15_EXECUTION_SEMANTICS",
    "PHASE15_FLOW_STEPS",
    "PHASE15_MAX_CONDITIONS",
    "PHASE15_MAX_CONCURRENT_SCANS",
    "PHASE15_MAX_EPISODES",
    "PHASE15_PROTOCOL_VARIANT",
    "PHASE15_RECEDING_HORIZON",
    "PHASE15_RESULT_SCHEMA_VERSION",
    "PHASE15_SCHEMA_VERSION",
    "PHASE15_STABILITY_SEEDS",
    "PHASE15_TASKS",
    "atomic_write_json",
    "adaptive_stability_specs",
    "capture_environment_state",
    "candidate_pool_metrics",
    "candidate_selection_metrics",
    "canonical_json",
    "circular_angle_error",
    "cluster_bootstrap",
    "condition_id",
    "condition_identity",
    "condition_lock",
    "condition_slug",
    "control_action_metrics",
    "deduplicate_actions",
    "grid_counts",
    "guidance_effect_metrics",
    "matched_guidance_mode_metrics",
    "fit_ridge_probe",
    "index_phase15_results",
    "make_control_actions",
    "make_probe_split",
    "mark_infrastructure_failure",
    "normalized_physical_distance",
    "p2_protocol",
    "phase15_scan_slot",
    "paired_guidance_metrics",
    "phase15_condition_specs",
    "primary_condition_specs",
    "publish_result",
    "result_path",
    "restore_environment_state",
    "safe_correlation",
    "sampling_stability_specs",
    "stable_sha256",
    "synchronous_timing",
    "summarize_timing_samples",
    "validate_phase15_result",
    "verify_state_restore",
]


def phase15_condition_specs() -> list[Condition]:
    """Public alias used by scripts and tests."""
    return primary_condition_specs()
