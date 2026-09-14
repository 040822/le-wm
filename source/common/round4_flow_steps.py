"""Analysis helpers for the Round 4 action-flow step ablation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from source.common.round3_phase1 import paired_comparison
from source.common.round4_protocol import ROUND4_TASKS


DEFAULT_FLOW_STEPS = (1, 2, 5, 10, 16)
DEFAULT_ACTION_MODES = ("P0", "P2", "P3")
DEFAULT_INVARIANT_MODES = ("P1",)
DEFAULT_COHORT_KINDS = ("dev", "final")


def _result_path(root: Path, task: str, mode: str, cohort_kind: str) -> Path:
    return root / task / mode / cohort_kind / "result.json"


def _load_result(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing result artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result artifact must be an object: {path}")
    if payload.get("status") != "ok":
        raise ValueError(f"result artifact is not ok: {path}")
    return payload


def _identity(payload: Mapping[str, Any]) -> tuple[Any, Any]:
    return payload.get("cohort_id"), payload.get("cohort_sha256")


def _planning_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    planning = payload.get("round4_planning")
    if not isinstance(planning, Mapping):
        return {}
    keys = (
        "planning_median_seconds",
        "planning_p95_seconds",
        "forward_count",
        "peak_memory_bytes",
    )
    return {key: planning.get(key) for key in keys}


def _validate_result_identity(
    payload: Mapping[str, Any],
    *,
    task: str,
    mode: str,
    cohort_kind: str,
    path: Path,
) -> None:
    observed = (
        payload.get("task"),
        payload.get("round4_mode"),
        payload.get("cohort_kind"),
    )
    expected = (task, mode, cohort_kind)
    if observed != expected:
        raise ValueError(
            f"result identity mismatch for {expected}: {observed} in {path}"
        )


def _validate_action_step_metadata(
    payload: Mapping[str, Any], *, expected: int, path: Path
) -> None:
    planning = payload.get("round4_planning")
    if not isinstance(planning, Mapping):
        raise ValueError(f"result is missing round4_planning metadata: {path}")
    observed = planning.get("action_flow_steps")
    if observed is None or int(observed) != int(expected):
        raise ValueError(
            f"action_flow_steps metadata mismatch: expected {expected}, "
            f"got {observed!r} in {path}"
        )


def _validate_checkpoint(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], path: Path
) -> None:
    baseline_checkpoint = baseline.get("checkpoint")
    candidate_checkpoint = candidate.get("checkpoint")
    if (
        baseline_checkpoint is not None
        and candidate_checkpoint is not None
        and baseline_checkpoint != candidate_checkpoint
    ):
        raise ValueError(f"checkpoint mismatch in flow-step comparison: {path}")


def _row(
    *,
    task: str,
    mode: str,
    cohort_kind: str,
    flow_steps: int | None,
    payload: Mapping[str, Any],
    path: Path,
    canonical: Mapping[str, Any] | None = None,
    canonical_path: Path | None = None,
    invariant: bool = False,
) -> dict[str, Any]:
    rate = float(payload["success_rate"]) * 100.0
    result: dict[str, Any] = {
        "task": task,
        "mode": mode,
        "cohort_kind": cohort_kind,
        "flow_steps": flow_steps,
        "invariant": invariant,
        "success_rate": float(payload["success_rate"]),
        "success_rate_percent": rate,
        "planning": _planning_payload(payload),
        "path": str(path),
    }
    if canonical is None or canonical_path is None:
        result.update(
            {
                "canonical_success_rate": None,
                "canonical_success_rate_percent": None,
                "delta_pp_vs_canonical": None,
                "paired_vs_canonical": None,
                "canonical_path": None,
            }
        )
        return result

    canonical_rate = float(canonical["success_rate"]) * 100.0
    try:
        paired = paired_comparison(
            canonical.get("episodes", []), payload.get("episodes", [])
        )
    except ValueError as exc:
        raise ValueError(
            f"episode alignment mismatch for {task}/{mode}/{cohort_kind}/"
            f"steps_{flow_steps}: {exc}"
        ) from exc
    result.update(
        {
            "canonical_success_rate": float(canonical["success_rate"]),
            "canonical_success_rate_percent": canonical_rate,
            "delta_pp_vs_canonical": rate - canonical_rate,
            "paired_vs_canonical": paired,
            "canonical_path": str(canonical_path),
        }
    )
    return result


def analyze_flow_step_results(
    step_roots: Mapping[int, str | Path],
    invariant_root: str | Path,
    *,
    tasks: Sequence[str] = ROUND4_TASKS,
    action_modes: Sequence[str] = DEFAULT_ACTION_MODES,
    invariant_modes: Sequence[str] = DEFAULT_INVARIANT_MODES,
    cohort_kinds: Sequence[str] = DEFAULT_COHORT_KINDS,
    canonical_step: int = 16,
) -> dict[str, Any]:
    """Analyze step-dependent action planners against the canonical step.

    Every action-mode result must carry the requested ``action_flow_steps``
    metadata.  Results are paired by episode id and start step, while P1 is
    loaded once from ``invariant_root`` because it does not use the actor.
    """
    normalized_roots = {int(step): Path(root) for step, root in step_roots.items()}
    if not normalized_roots:
        raise ValueError("at least one flow-step result root is required")
    if int(canonical_step) not in normalized_roots:
        raise ValueError("canonical_step must be present in step_roots")
    if any(step < 1 for step in normalized_roots):
        raise ValueError("flow steps must be positive")

    rows: list[dict[str, Any]] = []
    canonical_step = int(canonical_step)
    canonical_root = normalized_roots[canonical_step]
    invariant_root = Path(invariant_root)

    for task in tasks:
        if task not in ROUND4_TASKS:
            raise ValueError(f"unknown Round 4 task: {task}")
        for mode in action_modes:
            for cohort_kind in cohort_kinds:
                canonical_path = _result_path(
                    canonical_root, task, mode, cohort_kind
                )
                canonical = _load_result(canonical_path)
                _validate_result_identity(
                    canonical,
                    task=task,
                    mode=mode,
                    cohort_kind=cohort_kind,
                    path=canonical_path,
                )
                _validate_action_step_metadata(
                    canonical, expected=canonical_step, path=canonical_path
                )
                for step, root in sorted(normalized_roots.items()):
                    path = _result_path(root, task, mode, cohort_kind)
                    payload = _load_result(path)
                    _validate_result_identity(
                        payload,
                        task=task,
                        mode=mode,
                        cohort_kind=cohort_kind,
                        path=path,
                    )
                    _validate_action_step_metadata(payload, expected=step, path=path)
                    if _identity(payload) != _identity(canonical):
                        raise ValueError(
                            f"cohort identity mismatch for "
                            f"{task}/{mode}/{cohort_kind}/steps_{step}"
                        )
                    _validate_checkpoint(canonical, payload, path)
                    rows.append(
                        _row(
                            task=task,
                            mode=mode,
                            cohort_kind=cohort_kind,
                            flow_steps=step,
                            payload=payload,
                            path=path,
                            canonical=canonical,
                            canonical_path=canonical_path,
                        )
                    )

        for mode in invariant_modes:
            for cohort_kind in cohort_kinds:
                path = _result_path(invariant_root, task, mode, cohort_kind)
                payload = _load_result(path)
                _validate_result_identity(
                    payload,
                    task=task,
                    mode=mode,
                    cohort_kind=cohort_kind,
                    path=path,
                )
                rows.append(
                    _row(
                        task=task,
                        mode=mode,
                        cohort_kind=cohort_kind,
                        flow_steps=None,
                        payload=payload,
                        path=path,
                        invariant=True,
                    )
                )

    summaries: dict[str, dict[str, Any]] = {}
    for mode in action_modes:
        for cohort_kind in cohort_kinds:
            for step in sorted(normalized_roots):
                selected = [
                    row
                    for row in rows
                    if row["mode"] == mode
                    and row["cohort_kind"] == cohort_kind
                    and row["flow_steps"] == step
                ]
                if not selected:
                    continue
                key = f"{mode}/{cohort_kind}/steps_{step}"
                summaries[key] = {
                    "task_count": len(selected),
                    "mean_success_rate_percent": sum(
                        row["success_rate_percent"] for row in selected
                    )
                    / len(selected),
                    "mean_delta_pp_vs_canonical": sum(
                        row["delta_pp_vs_canonical"] for row in selected
                    )
                    / len(selected),
                }
    for mode in invariant_modes:
        for cohort_kind in cohort_kinds:
            selected = [
                row
                for row in rows
                if row["mode"] == mode and row["cohort_kind"] == cohort_kind
            ]
            if selected:
                key = f"{mode}/{cohort_kind}/invariant"
                summaries[key] = {
                    "task_count": len(selected),
                    "mean_success_rate_percent": sum(
                        row["success_rate_percent"] for row in selected
                    )
                    / len(selected),
                }

    return {
        "schema_version": "round4_flow_step_ablation_v1",
        "experiment": "R4-AB-FM-step-ablation",
        "canonical_step": canonical_step,
        "flow_steps": sorted(normalized_roots),
        "tasks": list(tasks),
        "action_modes": list(action_modes),
        "invariant_modes": list(invariant_modes),
        "cohort_kinds": list(cohort_kinds),
        "rows": rows,
        "summaries": summaries,
    }


__all__ = [
    "DEFAULT_ACTION_MODES",
    "DEFAULT_COHORT_KINDS",
    "DEFAULT_FLOW_STEPS",
    "DEFAULT_INVARIANT_MODES",
    "analyze_flow_step_results",
]
