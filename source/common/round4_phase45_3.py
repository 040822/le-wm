"""Analysis and validation helpers for Round 4 Phase 4.5-3.

The runner stores every condition below one fresh output root.  This module
keeps the analysis independent of process order and refuses to pair results
unless task, cohort, episode identity, and planning metadata all agree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .round3_phase1 import CohortManifest, paired_comparison, wilson_interval
from .round3_validation import validate_result_payload


PHASE45_SCHEMA_VERSION = "round4_phase45_3_v1"
PHASE45_TASKS = ("cube", "pusht", "reacher", "tworoom")
PHASE45_MODES = ("P0", "P1", "P2", "P3")
PHASE45_CEM_PROTOCOLS = ("legacy", "cem-clip", "cem-scale")
PHASE45_NON_CEM_PROTOCOL = "not_applicable"
PHASE45_FLOW_STEPS = (1, 2, 5, 10, 16, 32)
PHASE45_INTEGRATORS = ("euler", "heun")


def _condition_step(planning: Mapping[str, Any], mode: str) -> int | None:
    value = planning.get("action_flow_steps")
    if mode == "P1":
        if value is not None:
            raise ValueError("P1 must have invariant action_flow_steps=null")
        return None
    if value is None or int(value) < 1:
        raise ValueError(f"{mode} is missing a positive action_flow_steps value")
    return int(value)


def condition_key(payload: Mapping[str, Any]) -> tuple[str, str, str, int | None, str]:
    """Return the canonical condition identity encoded in a result."""
    task = str(payload.get("task"))
    mode = str(payload.get("round4_mode", payload.get("stage")))
    planning = payload.get("round4_planning")
    if not isinstance(planning, Mapping):
        raise ValueError("result is missing round4_planning metadata")
    protocol = str(planning.get("cem_protocol"))
    integrator = str(planning.get("action_flow_integrator"))
    step = _condition_step(planning, mode)
    if task not in PHASE45_TASKS or mode not in PHASE45_MODES:
        raise ValueError(f"unsupported Phase 4.5-3 task/mode: {task}/{mode}")
    if protocol not in (*PHASE45_CEM_PROTOCOLS, PHASE45_NON_CEM_PROTOCOL):
        raise ValueError(f"unsupported cem_protocol: {protocol!r}")
    if mode == "P1":
        if integrator != "not_applicable":
            raise ValueError("P1 must have action_flow_integrator=not_applicable")
    elif integrator not in PHASE45_INTEGRATORS:
        raise ValueError(f"{mode} has unsupported integrator {integrator!r}")
    if mode in {"P0", "P3"} and protocol != PHASE45_NON_CEM_PROTOCOL:
        raise ValueError(
            f"{mode} must use cem_protocol={PHASE45_NON_CEM_PROTOCOL!r}"
        )
    if mode in {"P0", "P3"} and planning.get("action_bound_mode") != "none":
        raise ValueError(f"{mode} must keep action_bound_mode=none")
    if mode in {"P1", "P2"}:
        expected_bound_mode = {
            "legacy": "none",
            "cem-clip": "candidate_clip",
            "cem-scale": "candidate_scale",
        }[protocol]
        if planning.get("action_bound_mode") != expected_bound_mode:
            raise ValueError(
                f"{mode}/{protocol} has action_bound_mode="
                f"{planning.get('action_bound_mode')!r}; expected {expected_bound_mode!r}"
            )
    if mode == "P1" and protocol == "legacy":
        return task, mode, protocol, None, integrator
    if mode == "P1":
        return task, mode, protocol, None, integrator
    if step not in PHASE45_FLOW_STEPS:
        raise ValueError(f"unexpected action flow step {step} for {mode}")
    return task, mode, protocol, step, integrator


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read result JSON {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"result JSON must contain an object: {path}")
    return value


def _cohort_identity(payload: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (
        payload.get("cohort_id"),
        payload.get("cohort_sha256"),
        payload.get("cohort_kind"),
    )


def _validate_result(
    payload: Mapping[str, Any],
    path: Path,
    *,
    manifest: CohortManifest | None,
    expected_checkpoint: str | None,
) -> tuple[str, str, str, int | None, str]:
    if payload.get("status") != "ok":
        raise ValueError(f"result status is not ok: {path}")
    if payload.get("cohort_kind") != "dev":
        raise ValueError(f"Phase 4.5-3 accepts dev results only: {path}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 50:
        raise ValueError(f"result must contain exactly 50 episodes: {path}")
    if expected_checkpoint is not None and str(payload.get("checkpoint")) != str(
        expected_checkpoint
    ):
        raise ValueError(
            f"checkpoint mismatch in {path}: {payload.get('checkpoint')!r} != "
            f"{expected_checkpoint!r}"
        )
    planning = payload.get("round4_planning")
    if not isinstance(planning, Mapping):
        raise ValueError(f"round4_planning metadata is missing: {path}")
    for field in (
        "cem_protocol",
        "action_bound_mode",
        "action_bounds",
        "planning_mean_seconds",
        "planning_median_seconds",
        "planning_p95_seconds",
        "forward_count",
        "peak_memory_bytes",
    ):
        if field not in planning:
            raise ValueError(f"round4_planning is missing {field}: {path}")
    bounds = planning.get("action_bounds")
    if not isinstance(bounds, Mapping) or any(
        bounds.get(field) is None
        for field in (
            "physical_low",
            "physical_high",
            "normalizer_mean",
            "normalizer_scale",
            "normalized_low",
            "normalized_high",
        )
    ):
        raise ValueError(f"action bounds/normalizer metadata is incomplete: {path}")
    if planning.get("cem_protocol") in {"cem-clip", "cem-scale"}:
        projections = planning.get("action_bound_projection")
        if not isinstance(projections, list) or not projections:
            raise ValueError(f"CEM projection metadata is missing: {path}")
        if any(
            float(item.get("projected_candidate_violation_fraction", 1.0))
            != 0.0
            for item in projections
            if isinstance(item, Mapping)
        ):
            raise ValueError(f"projected CEM candidates violate bounds: {path}")
    if manifest is not None:
        validate_result_payload(
            payload,
            manifest=manifest,
            expected_count=50,
            trace_path=payload.get("trace_path"),
        )
    key = condition_key(payload)
    return key


def _mean(values: Sequence[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return float(np.mean(numeric)) if numeric else None


def _projection_values(
    planning: Mapping[str, Any], field: str
) -> list[float]:
    events = planning.get("action_bound_projection", [])
    if not isinstance(events, list):
        return []
    values = []
    for event in events:
        if isinstance(event, Mapping) and event.get(field) is not None:
            values.append(float(event[field]))
    return values


def _projection_nested_values(
    planning: Mapping[str, Any], section: str, field: str
) -> list[float]:
    events = planning.get("action_bound_projection", [])
    if not isinstance(events, list):
        return []
    values = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        item = event.get(section)
        if isinstance(item, Mapping) and item.get(field) is not None:
            values.append(float(item[field]))
    return values


def _projection_side_values(
    planning: Mapping[str, Any], side: str, field: str
) -> list[float]:
    events = planning.get("action_bound_projection", [])
    if not isinstance(events, list):
        return []
    values = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        warm_start = event.get("warm_start")
        if not isinstance(warm_start, Mapping):
            continue
        item = warm_start.get(side)
        if isinstance(item, Mapping) and item.get(field) is not None:
            values.append(float(item[field]))
    return values


def _condition_row(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    task, mode, protocol, step, integrator = condition_key(payload)
    episodes = payload["episodes"]
    successes = [bool(item.get("success", False)) for item in episodes]
    low, high = wilson_interval(successes)
    planning = payload["round4_planning"]
    executed = planning.get("executed_action_bounds", {})
    normalized = executed.get("normalized", {}) if isinstance(executed, Mapping) else {}
    physical = executed.get("physical", {}) if isinstance(executed, Mapping) else {}
    return {
        "task": task,
        "mode": mode,
        "cem_protocol": protocol,
        "action_bound_mode": planning.get("action_bound_mode"),
        "action_flow_steps": step,
        "action_flow_integrator": integrator,
        "cohort_id": payload.get("cohort_id"),
        "cohort_sha256": payload.get("cohort_sha256"),
        "checkpoint": payload.get("checkpoint"),
        "status": payload.get("status"),
        "episodes": len(episodes),
        "successes": int(sum(successes)),
        "success_rate": float(np.mean(successes)),
        "success_rate_percent": float(np.mean(successes) * 100.0),
        "wilson_95_percent_low": float(low * 100.0),
        "wilson_95_percent_high": float(high * 100.0),
        "planning_mean_seconds": planning.get("planning_mean_seconds"),
        "planning_median_seconds": planning.get("planning_median_seconds"),
        "planning_p95_seconds": planning.get("planning_p95_seconds"),
        "forward_count": planning.get("forward_count"),
        "peak_memory_bytes": planning.get("peak_memory_bytes"),
        "normalized_true_bound_violation_fraction": normalized.get(
            "true_normalized_bound_violation_fraction"
        ),
        "normalized_action_vector_violation_fraction": normalized.get(
            "true_normalized_action_vector_violation_fraction"
        ),
        "normalized_trajectory_violation_fraction": normalized.get(
            "true_normalized_trajectory_violation_fraction"
        ),
        "physical_action_violation_fraction": physical.get(
            "physical_action_violation_fraction"
        ),
        "physical_action_max_violation": physical.get(
            "physical_action_max_violation_amplitude"
        ),
        "legacy_unit_threshold_fraction": normalized.get(
            "legacy_unit_threshold_fraction"
        ),
        "raw_candidate_violation_fraction": _mean(
            _projection_values(planning, "raw_candidate_violation_fraction")
        ),
        "projected_candidate_violation_fraction": _mean(
            _projection_values(planning, "projected_candidate_violation_fraction")
        ),
        "candidate_changed_fraction": _mean(
            _projection_values(planning, "candidate_changed_fraction")
        ),
        "warm_start_before_violation_fraction": _mean(
            _projection_side_values(
                planning,
                "before",
                "true_normalized_bound_violation_fraction",
            )
        ),
        "warm_start_after_violation_fraction": _mean(
            _projection_side_values(
                planning,
                "after",
                "true_normalized_bound_violation_fraction",
            )
        ),
        "scale_factor_median": _mean(
            [
                item.get("median")
                for item in (
                    event.get("scale_factor_distribution")
                    for event in planning.get("action_bound_projection", [])
                    if isinstance(event, Mapping)
                )
                if isinstance(item, Mapping)
            ]
        ),
        "path": str(path),
        "episodes_payload": episodes,
    }


def load_phase45_results(
    root: str | Path,
    *,
    manifests: Mapping[str, CohortManifest] | None = None,
    checkpoints: Mapping[str, str] | None = None,
    require_complete: bool = True,
) -> tuple[dict[tuple[str, str, str, int | None, str], dict[str, Any]], dict[str, Any]]:
    """Load, validate, and index all new dev condition results."""
    root = Path(root)
    results: dict[tuple[str, str, str, int | None, str], dict[str, Any]] = {}
    paths = sorted((root / "conditions").glob("**/result.json"))
    errors: list[str] = []
    for path in paths:
        try:
            payload = _load_json(path)
            key = _validate_result(
                payload,
                path,
                manifest=(manifests or {}).get(str(payload.get("task"))),
                expected_checkpoint=(checkpoints or {}).get(str(payload.get("task"))),
            )
            if key in results:
                raise ValueError(f"duplicate condition key {key}")
            results[key] = {
                "payload": payload,
                "path": path,
                "row": _condition_row(payload, path),
            }
        except (OSError, TypeError, ValueError, KeyError) as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        raise ValueError("invalid Phase 4.5-3 results:\n" + "\n".join(errors))

    expected: set[tuple[str, str, str, int | None, str]] = set()
    for task in PHASE45_TASKS:
        for mode in ("P0", "P3"):
            for step in PHASE45_FLOW_STEPS:
                for integrator in PHASE45_INTEGRATORS:
                    expected.add(
                        (task, mode, PHASE45_NON_CEM_PROTOCOL, step, integrator)
                    )
        for protocol in PHASE45_CEM_PROTOCOLS:
            expected.add((task, "P1", protocol, None, "not_applicable"))
            for step in PHASE45_FLOW_STEPS:
                for integrator in PHASE45_INTEGRATORS:
                    expected.add((task, "P2", protocol, step, integrator))
    missing = sorted(expected - set(results))
    if require_complete and missing:
        raise ValueError(f"missing {len(missing)} Phase 4.5-3 conditions: {missing[:8]}")
    validation = {
        "result_count": len(results),
        "expected_count": len(expected),
        "missing_count": len(missing),
        "missing": [list(item) for item in missing],
        "all_status_ok": all(item["payload"].get("status") == "ok" for item in results.values()),
        "dev_only": all(item["payload"].get("cohort_kind") == "dev" for item in results.values()),
    }
    return results, validation


def _comparison(
    indexed: Mapping[tuple[str, str, str, int | None, str], Mapping[str, Any]],
    *,
    label: str,
    baseline: tuple[str, str, str, int | None, str],
    treatment: tuple[str, str, str, int | None, str],
    category: str,
) -> dict[str, Any]:
    left = indexed[baseline]["payload"]
    right = indexed[treatment]["payload"]
    if _cohort_identity(left) != _cohort_identity(right):
        raise ValueError(f"cohort mismatch in comparison {label}")
    paired = paired_comparison(left["episodes"], right["episodes"])
    return {
        "label": label,
        "category": category,
        "baseline": list(baseline),
        "treatment": list(treatment),
        "baseline_success_rate": float(left["success_rate"]),
        "treatment_success_rate": float(right["success_rate"]),
        "baseline_success_rate_percent": float(left["success_rate"] * 100.0),
        "treatment_success_rate_percent": float(right["success_rate"] * 100.0),
        "delta_pp": float((right["success_rate"] - left["success_rate"]) * 100.0),
        "paired": paired,
        "net_success_delta": int(paired["improved"] - paired["regressed"]),
    }


def analyze_phase45_results(
    indexed: Mapping[tuple[str, str, str, int | None, str], Mapping[str, Any]],
    *,
    lewm_rows: Sequence[Mapping[str, Any]] = (),
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce per-condition rows and every pre-registered paired comparison."""
    comparisons: list[dict[str, Any]] = []

    def add(label, baseline, treatment, category):
        comparisons.append(
            _comparison(
                indexed,
                label=label,
                baseline=baseline,
                treatment=treatment,
                category=category,
            )
        )

    for task in PHASE45_TASKS:
        for protocol in PHASE45_CEM_PROTOCOLS:
            for integrator in PHASE45_INTEGRATORS:
                baseline = (task, "P2", protocol, 16, integrator)
                for step in PHASE45_FLOW_STEPS:
                    if step == 16:
                        continue
                    add(
                        f"{task}/P2/{protocol}/{integrator}/step_{step}_vs_step_16",
                        baseline,
                        (task, "P2", protocol, step, integrator),
                        "P2_step_vs_step16",
                    )

        for mode in ("P0", "P2", "P3"):
            protocols = (
                (PHASE45_NON_CEM_PROTOCOL,)
                if mode != "P2"
                else PHASE45_CEM_PROTOCOLS
            )
            for protocol in protocols:
                for step in PHASE45_FLOW_STEPS:
                    add(
                        f"{task}/{mode}/{protocol}/heun_vs_euler/step_{step}",
                        (task, mode, protocol, step, "euler"),
                        (task, mode, protocol, step, "heun"),
                        "euler_vs_heun",
                    )

        for left, right, label in (
            ("cem-clip", "legacy", "P1_cem_clip_vs_legacy"),
            ("cem-scale", "cem-clip", "P1_cem_scale_vs_clip"),
            ("cem-scale", "legacy", "P1_cem_scale_vs_legacy"),
        ):
            add(
                f"{task}/{label}",
                (task, "P1", right, None, "not_applicable"),
                (task, "P1", left, None, "not_applicable"),
                "P1_protocol",
            )

        for step in PHASE45_FLOW_STEPS:
            for integrator in PHASE45_INTEGRATORS:
                for left, right, label in (
                    ("cem-clip", "legacy", "P2_cem_clip_vs_legacy"),
                    ("cem-scale", "cem-clip", "P2_cem_scale_vs_clip"),
                    ("cem-scale", "legacy", "P2_cem_scale_vs_legacy"),
                ):
                    add(
                        f"{task}/{label}/{integrator}/step_{step}",
                        (task, "P2", right, step, integrator),
                        (task, "P2", left, step, integrator),
                        "P2_protocol",
                    )

    return {
        "schema_version": PHASE45_SCHEMA_VERSION,
        "validation": dict(validation or {}),
        "rows": [indexed[key]["row"] for key in sorted(indexed)],
        "comparisons": comparisons,
        "lewm": [dict(row) for row in lewm_rows],
    }


__all__ = [
    "PHASE45_CEM_PROTOCOLS",
    "PHASE45_FLOW_STEPS",
    "PHASE45_INTEGRATORS",
    "PHASE45_NON_CEM_PROTOCOL",
    "PHASE45_MODES",
    "PHASE45_SCHEMA_VERSION",
    "PHASE45_TASKS",
    "analyze_phase45_results",
    "condition_key",
    "load_phase45_results",
]
