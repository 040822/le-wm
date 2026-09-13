"""Cross-root comparisons for matched Round 4 training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from source.common.round3_phase1 import paired_comparison
from source.common.round4_protocol import ROUND4_TASKS


DEFAULT_COMPARISON_MODES = ("P0", "P1", "P2", "P3")
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


def _source_root(
    default_root: Path,
    mode: str,
    overrides: Mapping[str, str | Path] | None,
) -> Path:
    if overrides and mode in overrides:
        return Path(overrides[mode])
    return default_root


def _cohort_identity(payload: Mapping[str, Any]) -> tuple[Any, Any]:
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


def compare_result_roots(
    baseline_root: str | Path,
    treatment_root: str | Path,
    *,
    tasks: Sequence[str] = ROUND4_TASKS,
    modes: Sequence[str] = DEFAULT_COMPARISON_MODES,
    cohort_kinds: Sequence[str] = DEFAULT_COHORT_KINDS,
    baseline_mode_roots: Mapping[str, str | Path] | None = None,
    treatment_mode_roots: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Compare matched result artifacts from two training conditions.

    The interface accepts optional per-mode roots because the canonical R4
    P3 artifacts use a candidate-batch-64 sidecar while P0--P2 use the base
    result root.  Every pair must share cohort identity and episode starts.
    """
    baseline_root = Path(baseline_root)
    treatment_root = Path(treatment_root)
    rows: list[dict[str, Any]] = []

    for task in tasks:
        if task not in ROUND4_TASKS:
            raise ValueError(f"unknown Round 4 task: {task}")
        for mode in modes:
            for cohort_kind in cohort_kinds:
                baseline_path = _result_path(
                    _source_root(baseline_root, mode, baseline_mode_roots),
                    task,
                    mode,
                    cohort_kind,
                )
                treatment_path = _result_path(
                    _source_root(treatment_root, mode, treatment_mode_roots),
                    task,
                    mode,
                    cohort_kind,
                )
                baseline = _load_result(baseline_path)
                treatment = _load_result(treatment_path)

                expected = (task, mode, cohort_kind)
                for label, payload, path in (
                    ("baseline", baseline, baseline_path),
                    ("treatment", treatment, treatment_path),
                ):
                    observed = (
                        payload.get("task"),
                        payload.get("round4_mode"),
                        payload.get("cohort_kind"),
                    )
                    if observed != expected:
                        raise ValueError(
                            f"{label} result identity mismatch for {expected}: "
                            f"{observed} in {path}"
                        )

                if _cohort_identity(baseline) != _cohort_identity(treatment):
                    raise ValueError(
                        f"cohort identity mismatch for {task}/{mode}/{cohort_kind}"
                    )

                try:
                    paired = paired_comparison(
                        baseline.get("episodes", []), treatment.get("episodes", [])
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"episode alignment mismatch for {task}/{mode}/{cohort_kind}: {exc}"
                    ) from exc

                baseline_rate = float(baseline["success_rate"])
                treatment_rate = float(treatment["success_rate"])
                rows.append(
                    {
                        "task": task,
                        "mode": mode,
                        "cohort_kind": cohort_kind,
                        "baseline_success_rate": baseline_rate,
                        "treatment_success_rate": treatment_rate,
                        "baseline_success_rate_percent": baseline_rate * 100.0,
                        "treatment_success_rate_percent": treatment_rate * 100.0,
                        "delta_pp": (treatment_rate - baseline_rate) * 100.0,
                        "baseline_episodes": len(baseline.get("episodes", [])),
                        "treatment_episodes": len(treatment.get("episodes", [])),
                        "paired": paired,
                        "baseline_planning": _planning_payload(baseline),
                        "treatment_planning": _planning_payload(treatment),
                        "baseline_path": str(baseline_path),
                        "treatment_path": str(treatment_path),
                    }
                )

    summaries: dict[str, dict[str, float | int]] = {}
    for mode in modes:
        for cohort_kind in cohort_kinds:
            selected = [
                row
                for row in rows
                if row["mode"] == mode and row["cohort_kind"] == cohort_kind
            ]
            if not selected:
                continue
            count = len(selected)
            summaries[f"{mode}/{cohort_kind}"] = {
                "task_count": count,
                "baseline_mean_success_rate_percent": sum(
                    row["baseline_success_rate_percent"] for row in selected
                )
                / count,
                "treatment_mean_success_rate_percent": sum(
                    row["treatment_success_rate_percent"] for row in selected
                )
                / count,
                "mean_delta_pp": sum(row["delta_pp"] for row in selected) / count,
            }

    return {
        "schema_version": "round4_ab_compare_v1",
        "baseline_label": "R4-AB",
        "treatment_label": "R4-ABDE",
        "baseline_root": str(baseline_root),
        "treatment_root": str(treatment_root),
        "tasks": list(tasks),
        "modes": list(modes),
        "cohort_kinds": list(cohort_kinds),
        "rows": rows,
        "summaries": summaries,
    }


__all__ = ["DEFAULT_COMPARISON_MODES", "DEFAULT_COHORT_KINDS", "compare_result_roots"]
