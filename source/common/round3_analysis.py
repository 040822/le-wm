"""Post-rollout relabelling and protocol-sensitivity helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .round3_phase1 import (
    action_is_legal,
    paired_comparison,
    protocol_sensitivity,
    summarize_episodes,
)
from .round3_protocol import evaluate_success, physical_distance


def relabel_trace(
    records: Sequence[Mapping[str, Any]],
    *,
    task: str,
    action_block: int = 5,
) -> list[dict[str, Any]]:
    """Recompute labels from one trace without changing its actions or states.

    This is the common path for ``legacy -> tolerance_revised``.  It refuses
    to invent a label when a step is missing either audited state field; the
    result then carries ``relabel_status=insufficient_fields``.
    """
    relabelled = []
    for original in records:
        record = deepcopy(dict(original))
        steps = record.get("steps", [])
        success_steps = []
        distances = []
        missing = 0
        for step in steps:
            current = step.get("current")
            goal = step.get("goal")
            if current is None or goal is None:
                missing += 1
                step["relabelled_success"] = None
                continue
            try:
                current_array = np.asarray(current, dtype=np.float64)
                goal_array = np.asarray(goal, dtype=np.float64)
                step_success = bool(evaluate_success(task, current_array, goal_array))
                step_distance = float(physical_distance(task, current_array, goal_array))
            except (TypeError, ValueError):
                missing += 1
                step["relabelled_success"] = None
                continue
            step["relabelled_success"] = step_success
            step["relabelled_distance"] = step_distance
            distances.append(step_distance)
            if step_success:
                success_steps.append(int(step.get("raw_env_step", len(success_steps) + 1)))
        windows = []
        for start in range(max(0, len(steps) - int(action_block) + 1)):
            window = steps[start : start + int(action_block)]
            if len(window) != int(action_block):
                continue
            neutral = all(bool(item.get("neutral_action", False)) for item in window)
            legal = all(item.get("action_legal") is True for item in window)
            held = all(item.get("relabelled_success") is True for item in window)
            if legal:
                windows.append({"start_step": window[0].get("raw_env_step"), "success": bool(neutral and held)})
        record["success"] = bool(success_steps)
        record["first_success_step"] = min(success_steps) if success_steps else None
        record["terminal_distance"] = distances[-1] if distances else None
        record["hold_windows"] = windows
        record["hold_success"] = bool(any(item["success"] for item in windows))
        record["relabel_status"] = "insufficient_fields" if missing else "ok"
        record["relabel_missing_field_count"] = int(missing)
        relabelled.append(record)
    return relabelled


def analyze_protocol_files(
    paths: Mapping[str, str], *, task: str, action_block: int = 5
) -> dict[str, Any]:
    """Load four result JSON files and return protocol sensitivity metrics."""
    import json
    from pathlib import Path

    values = {
        name: json.loads(Path(path).read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    legacy = values["legacy"].get("episodes", [])
    sampling = values["sampling_revised"].get("episodes", [])
    tolerance = values["tolerance_revised"].get("episodes", [])
    round3 = values["round3_revised"].get("episodes", [])
    return {
        "task": task,
        "summaries": {name: summarize_episodes(value) for name, value in values.items()},
        "sensitivity": protocol_sensitivity(legacy, sampling, tolerance, round3),
        "relabel_status": {
            "legacy_to_tolerance_revised": [
                item.get("relabel_status") for item in relabel_trace(legacy, task=task, action_block=action_block)
            ]
        },
    }


__all__ = [
    "analyze_protocol_files",
    "paired_comparison",
    "protocol_sensitivity",
    "relabel_trace",
]
