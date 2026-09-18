"""Validation and analysis helpers for Round 4 Phase 4.5-4.

Phase 4.5-4 deliberately combines two different 50-episode cohorts.  This
module keeps the cohort identity in every condition key and only permits
episode-level pairing within one cohort.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .round3_phase1 import CohortManifest, paired_comparison, wilson_interval
from .round3_validation import validate_cohort_manifest, validate_result_payload


PHASE454_SCHEMA_VERSION = "round4_phase45_4_v1"
PHASE454_TASKS = ("cube", "pusht", "reacher", "tworoom")
PHASE454_FLOW_STEPS = (1, 2, 5, 10, 16, 32)
PHASE454_STEP_REFERENCE = 16
PHASE454_INTEGRATOR = "euler"
PHASE454_NON_CEM_PROTOCOL = "not_applicable"
PHASE454_CEM_PROTOCOLS = ("legacy", "cem-clip", "cem-scale")
PHASE454_PROTOCOL_VARIANTS = ("legacy", "round3_revised")

ConditionKey = tuple[str, str, str, int | None, str, str]


def _step_condition(task: str, mode: str, protocol: str, step: int) -> dict[str, Any]:
    return {
        "task": task,
        "mode": mode,
        "cem_protocol": protocol,
        "action_flow_steps": step,
        "action_flow_integrator": PHASE454_INTEGRATOR,
    }


def _invariant_condition(task: str, protocol: str) -> dict[str, Any]:
    return {
        "task": task,
        "mode": "P1",
        "cem_protocol": protocol,
        "action_flow_steps": None,
        "action_flow_integrator": "not_applicable",
    }


def phase454_condition_specs() -> list[dict[str, Any]]:
    """Return the pre-registered 96-condition matrix in stable order."""
    conditions: list[dict[str, Any]] = []
    for task in PHASE454_TASKS:
        conditions.extend(
            _step_condition(task, "P0", PHASE454_NON_CEM_PROTOCOL, step)
            for step in PHASE454_FLOW_STEPS
        )
        conditions.append(_invariant_condition(task, "legacy"))
        conditions.extend(
            _step_condition(task, "P2", "legacy", step)
            for step in PHASE454_FLOW_STEPS
        )
        conditions.extend(
            _step_condition(task, "P3", PHASE454_NON_CEM_PROTOCOL, step)
            for step in PHASE454_FLOW_STEPS
        )
        if task == "reacher":
            for protocol in ("cem-clip", "cem-scale"):
                conditions.append(_invariant_condition(task, protocol))
                conditions.extend(
                    _step_condition(task, "P2", protocol, step)
                    for step in PHASE454_FLOW_STEPS
                )
        if task == "tworoom":
            conditions.extend(
                _step_condition(task, "P2", "cem-clip", step)
                for step in PHASE454_FLOW_STEPS
            )
    return conditions


def condition_name(condition: Mapping[str, Any]) -> str:
    """Return the stable result-directory suffix for one matrix condition."""
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
            "dev",
        )
    )


def _condition_key_from_spec(
    condition: Mapping[str, Any], protocol_variant: str
) -> ConditionKey:
    return (
        str(condition["task"]),
        str(condition["mode"]),
        str(condition["cem_protocol"]),
        condition["action_flow_steps"],
        str(condition["action_flow_integrator"]),
        str(protocol_variant),
    )


def _condition_step(planning: Mapping[str, Any], mode: str) -> int | None:
    value = planning.get("action_flow_steps")
    if mode == "P1":
        if value is not None:
            raise ValueError("P1 must have invariant action_flow_steps=null")
        return None
    if value is None or int(value) not in PHASE454_FLOW_STEPS:
        raise ValueError(f"{mode} must use action_flow_steps 1 or 2")
    return int(value)


def condition_key(payload: Mapping[str, Any]) -> ConditionKey:
    """Return condition identity, including the result's cohort protocol."""
    task = str(payload.get("task"))
    mode = str(payload.get("round4_mode", payload.get("stage")))
    protocol_variant = str(payload.get("protocol_variant"))
    planning = payload.get("round4_planning")
    if not isinstance(planning, Mapping):
        raise ValueError("result is missing round4_planning metadata")
    protocol = str(planning.get("cem_protocol"))
    integrator = str(planning.get("action_flow_integrator"))
    if task not in PHASE454_TASKS or mode not in {"P0", "P1", "P2", "P3"}:
        raise ValueError(f"unsupported Phase 4.5-4 task/mode: {task}/{mode}")
    if protocol_variant not in PHASE454_PROTOCOL_VARIANTS:
        raise ValueError(f"unsupported protocol variant: {protocol_variant!r}")
    if mode == "P1":
        step = _condition_step(planning, mode)
        if integrator != "not_applicable":
            raise ValueError("P1 must have action_flow_integrator=not_applicable")
        if protocol not in PHASE454_CEM_PROTOCOLS:
            raise ValueError(f"P1 has unsupported cem_protocol {protocol!r}")
    else:
        step = _condition_step(planning, mode)
        if integrator != PHASE454_INTEGRATOR:
            raise ValueError(f"{mode} must use Euler integration")
        if mode in {"P0", "P3"}:
            if protocol != PHASE454_NON_CEM_PROTOCOL:
                raise ValueError(f"{mode} must use cem_protocol=not_applicable")
            if planning.get("action_bound_mode") != "none":
                raise ValueError(f"{mode} must keep action_bound_mode=none")
        elif protocol not in PHASE454_CEM_PROTOCOLS:
            raise ValueError(f"{mode} has unsupported cem_protocol {protocol!r}")
    if mode in {"P1", "P2"}:
        expected_bound_mode = {
            "legacy": "none",
            "cem-clip": "candidate_clip",
            "cem-scale": "candidate_scale",
        }[protocol]
        if planning.get("action_bound_mode") != expected_bound_mode:
            raise ValueError(
                f"{mode}/{protocol} has action_bound_mode="
                f"{planning.get('action_bound_mode')!r}; "
                f"expected {expected_bound_mode!r}"
            )
    return task, mode, protocol, step, integrator, protocol_variant


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
    manifest: CohortManifest,
    expected_checkpoint: str,
    expected_protocol_variant: str,
) -> ConditionKey:
    if payload.get("status") != "ok":
        raise ValueError(f"result status is not ok: {path}")
    if payload.get("protocol_variant") != expected_protocol_variant:
        raise ValueError(
            f"protocol_variant mismatch in {path}: "
            f"{payload.get('protocol_variant')!r} != {expected_protocol_variant!r}"
        )
    if payload.get("cohort_kind") != "dev":
        raise ValueError(f"Phase 4.5-4 accepts 50-episode dev results only: {path}")
    if len(payload.get("episodes", ())) != 50:
        raise ValueError(f"result must contain exactly 50 episodes: {path}")
    if str(payload.get("checkpoint")) != str(expected_checkpoint):
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
            float(item.get("projected_candidate_violation_fraction", 1.0)) != 0.0
            for item in projections
            if isinstance(item, Mapping)
        ):
            raise ValueError(f"projected CEM candidates violate bounds: {path}")
    validate_result_payload(
        payload,
        manifest=manifest,
        expected_count=50,
        trace_path=payload.get("trace_path"),
    )
    return condition_key(payload)


def _condition_row(
    payload: Mapping[str, Any], path: Path, *, cohort_label: str
) -> dict[str, Any]:
    task, mode, protocol, step, integrator, protocol_variant = condition_key(payload)
    successes = [bool(item.get("success", False)) for item in payload["episodes"]]
    low, high = wilson_interval(successes)
    planning = payload["round4_planning"]
    return {
        "cohort": cohort_label,
        "protocol_variant": protocol_variant,
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
        "episodes": len(successes),
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
        "path": str(path),
    }


def _manifest_paths(config: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    cohort = config["cohort"]
    section = cohort[label]
    if not isinstance(section, Mapping) or not isinstance(section.get("paths"), Mapping):
        raise ValueError(f"config.cohort.{label}.paths must be a mapping")
    return section["paths"]


def load_phase454_results(
    root: str | Path,
    *,
    dev_root: str | Path,
    manifests: Mapping[str, Mapping[str, CohortManifest]],
    checkpoints: Mapping[str, str],
    require_complete: bool = True,
) -> tuple[dict[ConditionKey, dict[str, Any]], dict[str, Any]]:
    """Load legacy artifacts and directly reference the Phase 4.5-3 dev artifacts."""
    root = Path(root)
    dev_root = Path(dev_root)
    indexed: dict[ConditionKey, dict[str, Any]] = {}
    errors: list[str] = []
    specs = phase454_condition_specs()

    for task in PHASE454_TASKS:
        for variant, base_root in (("legacy", root), ("round3_revised", dev_root)):
            manifest = manifests[task][variant]
            expected_root = base_root / "conditions"
            if variant == "legacy":
                expected_root = expected_root / "legacy"
            for condition in (item for item in specs if item["task"] == task):
                path = expected_root / task / condition_name(condition) / "result.json"
                try:
                    payload = _load_json(path)
                    expected = _condition_key_from_spec(condition, variant)
                    key = _validate_result(
                        payload,
                        path,
                        manifest=manifest,
                        expected_checkpoint=checkpoints[task],
                        expected_protocol_variant=variant,
                    )
                    if key != expected:
                        raise ValueError(f"condition {key} != expected {expected}")
                    if key in indexed:
                        raise ValueError(f"duplicate condition key {key}")
                    indexed[key] = {
                        "payload": payload,
                        "path": path,
                        "row": _condition_row(
                            payload,
                            path,
                            cohort_label="legacy" if variant == "legacy" else "dev",
                        ),
                    }
                except (OSError, TypeError, ValueError, KeyError) as exc:
                    errors.append(f"{path}: {exc}")

    expected = {
        _condition_key_from_spec(condition, variant)
        for condition in specs
        for variant in PHASE454_PROTOCOL_VARIANTS
    }
    missing = sorted(expected - set(indexed))
    if require_complete and missing:
        errors.append(f"missing {len(missing)} Phase 4.5-4 conditions: {missing[:8]}")
    if errors:
        raise ValueError("invalid Phase 4.5-4 results:\n" + "\n".join(errors))

    manifest_metadata = {
        variant: {
            task: {
                "path": None,
                "cohort_id": manifests[task][variant].cohort_id,
                "cohort_sha256": manifests[task][variant].computed_sha256,
                "protocol_variant": manifests[task][variant].protocol_variant,
                "cohort_kind": manifests[task][variant].cohort_kind,
                "episodes": len(manifests[task][variant].entries),
                "seed": manifests[task][variant].seed,
                "goal_offset_steps": manifests[task][variant].goal_offset_steps,
            }
            for task in PHASE454_TASKS
        }
        for variant in PHASE454_PROTOCOL_VARIANTS
    }
    validation = {
        "result_count": len(indexed),
        "expected_count": len(expected),
        "missing_count": len(missing),
        "missing": [list(item) for item in missing],
        "legacy_result_count": sum(
            key[-1] == "legacy" for key in indexed
        ),
        "dev_result_count": sum(
            key[-1] == "round3_revised" for key in indexed
        ),
        "legacy_expected_count": len(specs),
        "dev_expected_count": len(specs),
        "all_status_ok": all(
            item["payload"].get("status") == "ok" for item in indexed.values()
        ),
        "no_heun": all(
            item["row"]["action_flow_integrator"]
            in {PHASE454_INTEGRATOR, "not_applicable"}
            for item in indexed.values()
        ),
        "flow_steps_within_grid": all(
            item["row"]["action_flow_steps"] in {None, *PHASE454_FLOW_STEPS}
            for item in indexed.values()
        ),
        "episode_identity_validated": True,
        "manifest_metadata": manifest_metadata,
        "legacy_root": str(root),
        "dev_root": str(dev_root),
    }
    return indexed, validation


def _episode_identity(payload: Mapping[str, Any]) -> set[tuple[Any, Any, Any]]:
    return {
        (
            item.get("episode_id"),
            item.get("start_step"),
            item.get("row_index"),
        )
        for item in payload["episodes"]
    }


def _comparison(
    indexed: Mapping[ConditionKey, Mapping[str, Any]],
    *,
    label: str,
    baseline: ConditionKey,
    treatment: ConditionKey,
    category: str,
) -> dict[str, Any]:
    left = indexed[baseline]
    right = indexed[treatment]
    left_payload = left["payload"]
    right_payload = right["payload"]
    cross_cohort = baseline[-1] != treatment[-1]
    baseline_rate = float(left_payload["success_rate"])
    treatment_rate = float(right_payload["success_rate"])
    result = {
        "label": label,
        "category": category,
        "baseline": list(baseline),
        "treatment": list(treatment),
        "baseline_cohort": left["row"]["cohort"],
        "treatment_cohort": right["row"]["cohort"],
        "baseline_cohort_id": left_payload.get("cohort_id"),
        "treatment_cohort_id": right_payload.get("cohort_id"),
        "baseline_success_rate": baseline_rate,
        "treatment_success_rate": treatment_rate,
        "baseline_success_rate_percent": baseline_rate * 100.0,
        "treatment_success_rate_percent": treatment_rate * 100.0,
        "baseline_wilson_95_percent": [
            left["row"]["wilson_95_percent_low"],
            left["row"]["wilson_95_percent_high"],
        ],
        "treatment_wilson_95_percent": [
            right["row"]["wilson_95_percent_low"],
            right["row"]["wilson_95_percent_high"],
        ],
        "delta_pp": (treatment_rate - baseline_rate) * 100.0,
        "absolute_delta_pp": abs(treatment_rate - baseline_rate) * 100.0,
        "comparison_type": "unpaired_descriptive" if cross_cohort else "paired",
        "paired": None,
        "net_success_delta": None,
    }
    if not cross_cohort:
        if _cohort_identity(left_payload) != _cohort_identity(right_payload):
            raise ValueError(f"cohort metadata mismatch in comparison {label}")
        if _episode_identity(left_payload) != _episode_identity(right_payload):
            raise ValueError(f"episode identity mismatch in comparison {label}")
        paired = paired_comparison(
            left_payload["episodes"], right_payload["episodes"]
        )
        result["paired"] = paired
        result["net_success_delta"] = int(
            paired["improved"] - paired["regressed"]
        )
    return result


def analyze_phase454_results(
    indexed: Mapping[ConditionKey, Mapping[str, Any]],
    *,
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce full cohort rows and registered within/cohort comparisons."""
    specs = phase454_condition_specs()
    comparisons: list[dict[str, Any]] = []

    def key(condition: Mapping[str, Any], variant: str) -> ConditionKey:
        return _condition_key_from_spec(condition, variant)

    def add(
        label: str,
        baseline: ConditionKey,
        treatment: ConditionKey,
        category: str,
    ) -> None:
        comparisons.append(
            _comparison(
                indexed,
                label=label,
                baseline=baseline,
                treatment=treatment,
                category=category,
            )
        )

    for task in PHASE454_TASKS:
        task_specs = [item for item in specs if item["task"] == task]
        for condition in task_specs:
            add(
                f"{task}/{condition_name(condition)}/legacy_vs_dev",
                key(condition, "legacy"),
                key(condition, "round3_revised"),
                "legacy_vs_dev",
            )

        step_modes = [
            ("P0", PHASE454_NON_CEM_PROTOCOL),
            ("P2", "legacy"),
            ("P3", PHASE454_NON_CEM_PROTOCOL),
        ]
        if task == "reacher":
            step_modes += [("P2", "cem-clip"), ("P2", "cem-scale")]
        elif task == "tworoom":
            step_modes += [("P2", "cem-clip")]

        for variant in PHASE454_PROTOCOL_VARIANTS:
            label = "legacy" if variant == "legacy" else "dev"
            for mode, protocol in step_modes:
                for step in PHASE454_FLOW_STEPS:
                    if step == PHASE454_STEP_REFERENCE:
                        continue
                    add(
                        f"{label}/{task}/{mode}/{protocol}/step_{step}_vs_step_{PHASE454_STEP_REFERENCE}",
                        (
                            task,
                            mode,
                            protocol,
                            PHASE454_STEP_REFERENCE,
                            PHASE454_INTEGRATOR,
                            variant,
                        ),
                        (task, mode, protocol, step, PHASE454_INTEGRATOR, variant),
                        "step_vs_step16",
                    )

            if task == "reacher":
                for treatment, category in (
                    ("cem-clip", "cem_clip_vs_legacy"),
                    ("cem-scale", "cem_scale_vs_legacy"),
                ):
                    add(
                        f"{label}/{task}/P1/{treatment}_vs_legacy",
                        (task, "P1", "legacy", None, "not_applicable", variant),
                        (task, "P1", treatment, None, "not_applicable", variant),
                        category,
                    )
                add(
                    f"{label}/{task}/P1/cem_scale_vs_cem_clip",
                    (task, "P1", "cem-clip", None, "not_applicable", variant),
                    (task, "P1", "cem-scale", None, "not_applicable", variant),
                    "cem_scale_vs_cem_clip",
                )
            if task in {"reacher", "tworoom"}:
                treatments = (
                    (("cem-clip", "cem_clip_vs_legacy"),)
                    if task == "tworoom"
                    else (
                        ("cem-clip", "cem_clip_vs_legacy"),
                        ("cem-scale", "cem_scale_vs_legacy"),
                        ("cem-scale", "cem_scale_vs_cem_clip"),
                    )
                )
                for step in PHASE454_FLOW_STEPS:
                    for treatment, category in treatments:
                        baseline_protocol = (
                            "cem-clip" if category == "cem_scale_vs_cem_clip" else "legacy"
                        )
                        add(
                            f"{label}/{task}/P2/{category}/step_{step}",
                            (
                                task,
                                "P2",
                                baseline_protocol,
                                step,
                                PHASE454_INTEGRATOR,
                                variant,
                            ),
                            (task, "P2", treatment, step, PHASE454_INTEGRATOR, variant),
                            category,
                        )

    cohort_summary = {}
    for variant, label in (("legacy", "legacy"), ("round3_revised", "dev")):
        variant_rows = [
            item["row"] for item in indexed.values()
            if item["row"]["cohort"] == label
        ]
        cohort_summary[label] = {
            "protocol_variant": variant,
            "conditions": len(variant_rows),
            "tasks": {
                task: sum(item["task"] == task for item in variant_rows)
                for task in PHASE454_TASKS
            },
            "episodes_per_condition": sorted(
                {item["episodes"] for item in variant_rows}
            ),
        }

    directional_conclusions = {}
    for category in (
        "step_vs_step16",
        "cem_clip_vs_legacy",
        "cem_scale_vs_legacy",
        "cem_scale_vs_cem_clip",
        "legacy_vs_dev",
    ):
        selected = [item for item in comparisons if item["category"] == category]
        if category == "legacy_vs_dev":
            directional_conclusions[category] = {
                "comparisons": len(selected),
                "positive": sum(item["delta_pp"] > 0 for item in selected),
                "negative": sum(item["delta_pp"] < 0 for item in selected),
                "zero": sum(item["delta_pp"] == 0 for item in selected),
            }
        else:
            paired_items = [item for item in selected if item["paired"] is not None]
            directional_conclusions[category] = {
                "comparisons": len(paired_items),
                "positive": sum(item["delta_pp"] > 0 for item in paired_items),
                "negative": sum(item["delta_pp"] < 0 for item in paired_items),
                "zero": sum(item["delta_pp"] == 0 for item in paired_items),
            }

    return {
        "schema_version": PHASE454_SCHEMA_VERSION,
        "validation": dict(validation or {}),
        "cohort_summary": cohort_summary,
        "directional_conclusions": directional_conclusions,
        "rows": [indexed[item]["row"] for item in sorted(indexed)],
        "comparisons": comparisons,
    }


__all__ = [
    "ConditionKey",
    "PHASE454_CEM_PROTOCOLS",
    "PHASE454_FLOW_STEPS",
    "PHASE454_INTEGRATOR",
    "PHASE454_NON_CEM_PROTOCOL",
    "PHASE454_PROTOCOL_VARIANTS",
    "PHASE454_SCHEMA_VERSION",
    "PHASE454_TASKS",
    "analyze_phase454_results",
    "condition_key",
    "condition_name",
    "load_phase454_results",
    "phase454_condition_specs",
]
