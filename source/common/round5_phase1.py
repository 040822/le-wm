"""Round 5 Phase 1 condition matrix, validation, and analysis helpers.

Phase 1 reuses the canonical Phase 4.5-4 legacy artifacts as the no-guidance
baselines and adds guided-flow / post-opt conditions on the same R4-AB
checkpoint and 50-episode legacy cohort.  Every condition key carries the
guidance mode so guided and unguided rows can never be confused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .round3_phase1 import CohortManifest, paired_comparison, wilson_interval
from .round3_validation import validate_result_payload


ROUND5_SCHEMA_VERSION = "round5_phase1_v1"
ROUND5_TASKS = ("cube", "pusht", "reacher", "tworoom")
ROUND5_FLOW_STEPS = (1, 2, 5, 10, 16, 32)
ROUND5_STEP_REFERENCE = 16
ROUND5_INTEGRATOR = "euler"
ROUND5_NON_CEM_PROTOCOL = "not_applicable"
ROUND5_PROTOCOL_VARIANT = "legacy"
ROUND5_GUIDANCE_MODES = ("guided_flow", "post_opt")
ROUND5_P3_GUIDANCE_MODES = ("guided_flow", "post_opt", "post_opt_refine")
ROUND5_ALL_GUIDANCE = ("none",) + ROUND5_P3_GUIDANCE_MODES

ConditionKey = tuple[str, str, str, int | None, str, str]


def p2_protocol(task: str) -> str:
    """Return the frozen P2 CEM protocol: Reacher uses candidate-clip."""
    return "cem-clip" if str(task) == "reacher" else "legacy"


def _spec(
    task: str,
    mode: str,
    protocol: str,
    step: int | None,
    guidance: str,
    integrator: str,
) -> dict[str, Any]:
    return {
        "task": task,
        "mode": mode,
        "cem_protocol": protocol,
        "action_flow_steps": step,
        "action_flow_integrator": integrator,
        "guidance": guidance,
    }


def guidance_condition_specs() -> list[dict[str, Any]]:
    """Return the 168 new guidance conditions in stable order."""
    conditions: list[dict[str, Any]] = []
    for task in ROUND5_TASKS:
        for step in ROUND5_FLOW_STEPS:
            for guidance in ROUND5_GUIDANCE_MODES:
                conditions.append(
                    _spec(task, "P0", ROUND5_NON_CEM_PROTOCOL, step, guidance, ROUND5_INTEGRATOR)
                )
                conditions.append(
                    _spec(task, "P2", p2_protocol(task), step, guidance, ROUND5_INTEGRATOR)
                )
            for guidance in ROUND5_P3_GUIDANCE_MODES:
                conditions.append(
                    _spec(task, "P3", ROUND5_NON_CEM_PROTOCOL, step, guidance, ROUND5_INTEGRATOR)
                )
    return conditions


def baseline_condition_specs() -> list[dict[str, Any]]:
    """Return the no-guidance conditions reused from Phase 4.5-4."""
    conditions: list[dict[str, Any]] = []
    for task in ROUND5_TASKS:
        for step in ROUND5_FLOW_STEPS:
            conditions.append(
                _spec(task, "P0", ROUND5_NON_CEM_PROTOCOL, step, "none", ROUND5_INTEGRATOR)
            )
        conditions.append(
            _spec(task, "P1", p2_protocol(task), None, "none", "not_applicable")
        )
        for step in ROUND5_FLOW_STEPS:
            conditions.append(
                _spec(task, "P2", p2_protocol(task), step, "none", ROUND5_INTEGRATOR)
            )
        for step in ROUND5_FLOW_STEPS:
            conditions.append(
                _spec(task, "P3", ROUND5_NON_CEM_PROTOCOL, step, "none", ROUND5_INTEGRATOR)
            )
    return conditions


def condition_name(condition: Mapping[str, Any]) -> str:
    """Return the stable result-directory suffix for one condition."""
    step = (
        "invariant"
        if condition["action_flow_steps"] is None
        else f"step_{int(condition['action_flow_steps'])}"
    )
    return "/".join(
        (
            str(condition["mode"]),
            str(condition["cem_protocol"]),
            str(condition["guidance"]),
            step,
            str(condition["action_flow_integrator"]),
            "dev",
        )
    )


def baseline_result_path(root: str | Path, condition: Mapping[str, Any]) -> Path:
    """Return the Phase 4.5-4 artifact path for a no-guidance condition."""
    root = Path(root)
    if condition["mode"] == "P1":
        suffix = "P1/{protocol}/invariant/not_applicable/dev".format(
            protocol=condition["cem_protocol"]
        )
    else:
        suffix = "{mode}/{protocol}/step_{step}/{integrator}/dev".format(
            mode=condition["mode"],
            protocol=condition["cem_protocol"],
            step=int(condition["action_flow_steps"]),
            integrator=condition["action_flow_integrator"],
        )
    return root / "conditions" / "legacy" / str(condition["task"]) / suffix / "result.json"


def condition_key_from_spec(
    condition: Mapping[str, Any], protocol_variant: str = ROUND5_PROTOCOL_VARIANT
) -> ConditionKey:
    return (
        str(condition["task"]),
        str(condition["mode"]),
        str(condition["cem_protocol"]),
        condition["action_flow_steps"],
        str(condition["action_flow_integrator"]),
        str(condition["guidance"]),
    )


def condition_key(payload: Mapping[str, Any]) -> ConditionKey:
    """Return condition identity, including the guidance mode."""
    task = str(payload.get("task"))
    mode = str(payload.get("round4_mode", payload.get("stage")))
    planning = payload.get("round4_planning")
    if not isinstance(planning, Mapping):
        raise ValueError("result is missing round4_planning metadata")
    protocol = str(planning.get("cem_protocol"))
    guidance = str(planning.get("guidance_mode", "none"))
    integrator = str(planning.get("action_flow_integrator"))
    if task not in ROUND5_TASKS or mode not in {"P0", "P1", "P2", "P3"}:
        raise ValueError(f"unsupported Round 5 task/mode: {task}/{mode}")
    if guidance not in ROUND5_ALL_GUIDANCE:
        raise ValueError(f"unsupported guidance mode: {guidance!r}")
    if mode == "P1":
        if planning.get("action_flow_steps") is not None:
            raise ValueError("P1 must have invariant action_flow_steps=null")
        if integrator != "not_applicable":
            raise ValueError("P1 must have action_flow_integrator=not_applicable")
        if guidance != "none":
            raise ValueError("P1 guidance is not defined")
        if protocol not in {"legacy", "cem-clip"}:
            raise ValueError(f"P1 has unsupported cem_protocol {protocol!r}")
        step = None
    else:
        step = planning.get("action_flow_steps")
        if step is None or int(step) not in ROUND5_FLOW_STEPS:
            raise ValueError(f"{mode} must use a registered action_flow_steps")
        step = int(step)
        if integrator != ROUND5_INTEGRATOR:
            raise ValueError(f"{mode} must use Euler integration")
        if guidance == "post_opt_refine" and mode != "P3":
            raise ValueError("post_opt_refine is only defined for P3")
        if mode in {"P0", "P3"}:
            if protocol != ROUND5_NON_CEM_PROTOCOL:
                raise ValueError(f"{mode} must use cem_protocol=not_applicable")
            if planning.get("action_bound_mode") != "none":
                raise ValueError(f"{mode} must keep action_bound_mode=none")
        elif protocol not in {"legacy", "cem-clip"}:
            raise ValueError(f"{mode} has unsupported cem_protocol {protocol!r}")
    if mode == "P2" and protocol != p2_protocol(task):
        raise ValueError(
            f"P2/{task} must use cem_protocol={p2_protocol(task)!r}, got {protocol!r}"
        )
    if mode == "P1" and protocol != p2_protocol(task):
        raise ValueError(
            f"P1/{task} must use cem_protocol={p2_protocol(task)!r}, got {protocol!r}"
        )
    if guidance != "none" and mode in {"P0", "P2", "P3"}:
        expected_bound = {
            "P0": "none",
            "P2": "none" if protocol == "legacy" else "candidate_clip",
            "P3": "none",
        }[mode]
        if planning.get("action_bound_mode") != expected_bound:
            raise ValueError(
                f"{mode}/{protocol}/{guidance} has action_bound_mode="
                f"{planning.get('action_bound_mode')!r}; expected {expected_bound!r}"
            )
    return task, mode, protocol, step, integrator, guidance


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
) -> ConditionKey:
    if payload.get("status") != "ok":
        raise ValueError(f"result status is not ok: {path}")
    if payload.get("protocol_variant") != ROUND5_PROTOCOL_VARIANT:
        raise ValueError(
            f"protocol_variant mismatch in {path}: "
            f"{payload.get('protocol_variant')!r} != {ROUND5_PROTOCOL_VARIANT!r}"
        )
    if payload.get("cohort_kind") != "dev":
        raise ValueError(f"Round 5 Phase 1 accepts 50-episode dev results only: {path}")
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
        "planning_mean_seconds",
        "planning_median_seconds",
        "planning_p95_seconds",
        "forward_count",
        "peak_memory_bytes",
    ):
        if field not in planning:
            raise ValueError(f"round4_planning is missing {field}: {path}")
    validate_result_payload(
        payload,
        manifest=manifest,
        expected_count=50,
        trace_path=payload.get("trace_path"),
    )
    return condition_key(payload)


def _condition_row(
    payload: Mapping[str, Any], path: Path, *, source: str
) -> dict[str, Any]:
    task, mode, protocol, step, integrator, guidance = condition_key(payload)
    successes = [bool(item.get("success", False)) for item in payload["episodes"]]
    low, high = wilson_interval(successes)
    planning = payload["round4_planning"]
    return {
        "source": source,
        "protocol_variant": ROUND5_PROTOCOL_VARIANT,
        "task": task,
        "mode": mode,
        "cem_protocol": protocol,
        "action_bound_mode": planning.get("action_bound_mode"),
        "guidance": guidance,
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
        "guidance_backward_count": planning.get("guidance_backward_count"),
        "peak_memory_bytes": planning.get("peak_memory_bytes"),
        "path": str(path),
    }


def load_round5_results(
    root: str | Path,
    *,
    baseline_root: str | Path,
    manifests: Mapping[str, CohortManifest],
    checkpoints: Mapping[str, str],
    require_complete: bool = True,
) -> tuple[dict[ConditionKey, dict[str, Any]], dict[str, Any]]:
    """Load new guidance artifacts plus the reused Phase 4.5-4 baselines."""
    root = Path(root)
    baseline_root = Path(baseline_root)
    indexed: dict[ConditionKey, dict[str, Any]] = {}
    errors: list[str] = []

    def ingest(
        spec: Mapping[str, Any],
        path: Path,
        source: str,
        expected: ConditionKey,
    ) -> None:
        if not path.is_file():
            return
        try:
            payload = _load_json(path)
            key = _validate_result(
                payload,
                path,
                manifest=manifests[str(spec["task"])],
                expected_checkpoint=checkpoints[str(spec["task"])],
            )
            if key != expected:
                raise ValueError(f"condition {key} != expected {expected}")
            if key in indexed:
                raise ValueError(f"duplicate condition key {key}")
            indexed[key] = {
                "payload": payload,
                "path": path,
                "row": _condition_row(payload, path, source=source),
            }
        except (OSError, TypeError, ValueError, KeyError) as exc:
            errors.append(f"{path}: {exc}")

    new_specs = guidance_condition_specs()
    baseline_specs = baseline_condition_specs()
    for spec in baseline_specs:
        ingest(
            spec,
            baseline_result_path(baseline_root, spec),
            "baseline_reuse",
            condition_key_from_spec(spec),
        )
    for spec in new_specs:
        path = root / "conditions" / "legacy" / str(spec["task"]) / condition_name(spec) / "result.json"
        ingest(spec, path, "new", condition_key_from_spec(spec))

    expected = {
        condition_key_from_spec(spec)
        for spec in (*baseline_specs, *new_specs)
    }
    missing = sorted(expected - set(indexed))
    if require_complete and missing:
        errors.append(f"missing {len(missing)} Round 5 conditions: {missing[:8]}")
    if errors:
        raise ValueError("invalid Round 5 Phase 1 results:\n" + "\n".join(errors))

    baseline_count = sum(
        item["row"]["source"] == "baseline_reuse" for item in indexed.values()
    )
    new_count = sum(item["row"]["source"] == "new" for item in indexed.values())
    manifest_metadata = {
        task: {
            "path": None,
            "cohort_id": manifests[task].cohort_id,
            "cohort_sha256": manifests[task].computed_sha256,
            "protocol_variant": manifests[task].protocol_variant,
            "cohort_kind": manifests[task].cohort_kind,
            "episodes": len(manifests[task].entries),
            "seed": manifests[task].seed,
            "goal_offset_steps": manifests[task].goal_offset_steps,
        }
        for task in ROUND5_TASKS
    }
    validation = {
        "result_count": len(indexed),
        "expected_count": len(expected),
        "missing_count": len(missing),
        "missing": [list(item) for item in missing],
        "baseline_result_count": baseline_count,
        "baseline_expected_count": len(baseline_specs),
        "new_result_count": new_count,
        "new_expected_count": len(new_specs),
        "all_status_ok": all(
            item["payload"].get("status") == "ok" for item in indexed.values()
        ),
        "episode_identity_validated": True,
        "manifest_metadata": manifest_metadata,
        "baseline_root": str(baseline_root),
        "new_root": str(root),
    }
    return indexed, validation


def _episode_identity(payload: Mapping[str, Any]) -> set[tuple[Any, Any, Any]]:
    return {
        (item.get("episode_id"), item.get("start_step"), item.get("row_index"))
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
    baseline_rate = float(left_payload["success_rate"])
    treatment_rate = float(right_payload["success_rate"])
    if _cohort_identity(left_payload) != _cohort_identity(right_payload):
        raise ValueError(f"cohort metadata mismatch in comparison {label}")
    if _episode_identity(left_payload) != _episode_identity(right_payload):
        raise ValueError(f"episode identity mismatch in comparison {label}")
    paired = paired_comparison(left_payload["episodes"], right_payload["episodes"])
    return {
        "label": label,
        "category": category,
        "baseline": list(baseline),
        "treatment": list(treatment),
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
        "comparison_type": "paired",
        "paired": paired,
        "net_success_delta": int(paired["improved"] - paired["regressed"]),
    }


def analyze_round5_results(
    indexed: Mapping[ConditionKey, Mapping[str, Any]],
    *,
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce guidance-vs-none, P3-semantics and step comparisons."""
    comparisons: list[dict[str, Any]] = []

    def key(spec: Mapping[str, Any]) -> ConditionKey:
        return condition_key_from_spec(spec)

    def has(k: ConditionKey) -> bool:
        return k in indexed

    def add(label: str, baseline: ConditionKey, treatment: ConditionKey, category: str) -> None:
        if not has(baseline) or not has(treatment):
            return
        comparisons.append(
            _comparison(
                indexed,
                label=label,
                baseline=baseline,
                treatment=treatment,
                category=category,
            )
        )

    for task in ROUND5_TASKS:
        p1_key = condition_key_from_spec(
            _spec(task, "P1", p2_protocol(task), None, "none", "not_applicable")
        )
        for step in ROUND5_FLOW_STEPS:
            for mode in ("P0", "P2", "P3"):
                protocol = (
                    ROUND5_NON_CEM_PROTOCOL
                    if mode in {"P0", "P3"}
                    else p2_protocol(task)
                )
                add(
                    f"{task}/{mode}/step_{step}_vs_P1",
                    p1_key,
                    key(_spec(task, mode, protocol, step, "none", ROUND5_INTEGRATOR)),
                    "p1_vs_learned",
                )
        for step in ROUND5_FLOW_STEPS:
            # Guidance vs the matching no-guidance baseline.
            for mode in ("P0", "P2", "P3"):
                protocol = ROUND5_NON_CEM_PROTOCOL if mode in {"P0", "P3"} else p2_protocol(task)
                base_spec = _spec(task, mode, protocol, step, "none", ROUND5_INTEGRATOR)
                modes = ROUND5_GUIDANCE_MODES if mode != "P3" else ROUND5_P3_GUIDANCE_MODES
                for guidance in modes:
                    treatment_spec = _spec(task, mode, protocol, step, guidance, ROUND5_INTEGRATOR)
                    add(
                        f"{task}/{mode}/{guidance}/step_{step}_vs_none",
                        key(base_spec),
                        key(treatment_spec),
                        "guidance_vs_none",
                    )
            # P3 guidance semantics.
            for left, right in (
                ("post_opt_refine", "post_opt"),
                ("post_opt_refine", "guided_flow"),
                ("post_opt", "guided_flow"),
            ):
                add(
                    f"{task}/P3/{left}_vs_{right}/step_{step}",
                    key(_spec(task, "P3", ROUND5_NON_CEM_PROTOCOL, step, right, ROUND5_INTEGRATOR)),
                    key(_spec(task, "P3", ROUND5_NON_CEM_PROTOCOL, step, left, ROUND5_INTEGRATOR)),
                    "p3_guidance_semantics",
                )
            # Step effects within every guidance condition.
            for mode in ("P0", "P2", "P3"):
                protocol = ROUND5_NON_CEM_PROTOCOL if mode in {"P0", "P3"} else p2_protocol(task)
                modes = ("none",) + (
                    ROUND5_GUIDANCE_MODES if mode != "P3" else ROUND5_P3_GUIDANCE_MODES
                )
                for guidance in modes:
                    if step == ROUND5_STEP_REFERENCE:
                        continue
                    add(
                        f"{task}/{mode}/{protocol}/{guidance}/step_{step}_vs_step_{ROUND5_STEP_REFERENCE}",
                        key(_spec(task, mode, protocol, ROUND5_STEP_REFERENCE, guidance, ROUND5_INTEGRATOR)),
                        key(_spec(task, mode, protocol, step, guidance, ROUND5_INTEGRATOR)),
                        "step_vs_step16",
                    )

    rows = [indexed[item]["row"] for item in sorted(indexed)]
    new_rows = [row for row in rows if row["source"] == "new"]
    directional = {}
    for category in (
        "p1_vs_learned",
        "guidance_vs_none",
        "p3_guidance_semantics",
        "step_vs_step16",
    ):
        selected = [item for item in comparisons if item["category"] == category]
        directional[category] = {
            "comparisons": len(selected),
            "positive": sum(item["delta_pp"] > 0 for item in selected),
            "negative": sum(item["delta_pp"] < 0 for item in selected),
            "zero": sum(item["delta_pp"] == 0 for item in selected),
        }

    return {
        "schema_version": ROUND5_SCHEMA_VERSION,
        "validation": dict(validation or {}),
        "directional_conclusions": directional,
        "rows": rows,
        "new_rows": new_rows,
        "comparisons": comparisons,
    }


__all__ = [
    "ConditionKey",
    "ROUND5_ALL_GUIDANCE",
    "ROUND5_FLOW_STEPS",
    "ROUND5_GUIDANCE_MODES",
    "ROUND5_INTEGRATOR",
    "ROUND5_NON_CEM_PROTOCOL",
    "ROUND5_P3_GUIDANCE_MODES",
    "ROUND5_PROTOCOL_VARIANT",
    "ROUND5_SCHEMA_VERSION",
    "ROUND5_STEP_REFERENCE",
    "ROUND5_TASKS",
    "analyze_round5_results",
    "baseline_condition_specs",
    "baseline_result_path",
    "condition_key",
    "condition_key_from_spec",
    "condition_name",
    "guidance_condition_specs",
    "load_round5_results",
    "p2_protocol",
]
