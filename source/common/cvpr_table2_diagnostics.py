"""Protocol helpers for CVPR Table 2 fixed-pool and PO2 diagnostics."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from .round3_protocol import TASK_PREDICATES


def source_state_id(task: str, entry: Mapping[str, Any]) -> str:
    """Return a stable identifier for one source episode/start-state pair."""
    payload = {
        "task": str(task),
        "episode_id": entry["episode_id"],
        "start_step": int(entry["start_step"]),
        "row_index": int(entry["row_index"]),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def physical_cost(task: str, current: Sequence[float], goal: Sequence[float]) -> float:
    """Return the frozen task-interpretable scalar physical cost."""
    current_array = np.asarray(current, dtype=np.float64).reshape(-1)
    goal_array = np.asarray(goal, dtype=np.float64).reshape(-1)
    if current_array.shape != goal_array.shape or not np.isfinite(current_array).all() or not np.isfinite(goal_array).all():
        raise ValueError("physical states must be finite vectors with matching shapes")
    difference = current_array - goal_array
    if task == "pusht":
        if len(difference) < 5:
            raise ValueError("PushT states must contain xyxy and angle components")
        thresholds = TASK_PREDICATES[task].thresholds
        position = float(np.linalg.norm(difference[:4])) / float(thresholds["position_l2"])
        angle = abs(float((difference[4] + np.pi) % (2.0 * np.pi) - np.pi)) / float(thresholds["angle_abs"])
        return max(position, angle)
    if task not in TASK_PREDICATES:
        raise ValueError(f"unsupported Table 2 task: {task!r}")
    return float(np.linalg.norm(difference))


def _milestone(steps: Sequence[Mapping[str, Any]], target: int) -> Mapping[str, Any] | None:
    eligible = [item for item in steps if int(item.get("raw_env_step", -1)) <= int(target)]
    return eligible[-1] if eligible else None


def candidate_truth_record(
    *,
    task: str,
    evaluation_seed: int,
    entry: Mapping[str, Any],
    slot: int,
    candidate_index: int,
    episode: Mapping[str, Any],
) -> dict[str, Any]:
    """Reduce a native simulator branch trace to auditable 5/25-step outcomes."""
    steps = list(episode.get("steps", ()))
    if not steps:
        raise RuntimeError(f"branch produced no simulator steps for slot {slot}")
    result: dict[str, Any] = {
        "task": task,
        "evaluation_seed": int(evaluation_seed),
        "state_id": source_state_id(task, entry),
        "slot": int(slot),
        "episode_id": entry["episode_id"],
        "row_index": int(entry["row_index"]),
        "start_step": int(entry["start_step"]),
        "candidate_index": int(candidate_index),
        "effective_valid_length": len(steps),
        "environment_success_at_budget": bool(episode.get("success", False)),
    }
    for target in (5, 25):
        step = _milestone(steps, target)
        success = any(
            item.get("env_success") is True
            for item in steps
            if int(item.get("raw_env_step", -1)) <= target
        )
        reached = step is not None and int(step.get("raw_env_step", -1)) >= target
        result[f"success_by_{target}"] = bool(success)
        result[f"valid_at_{target}"] = bool(reached)
        if reached:
            current, goal = step.get("current"), step.get("goal")
            if current is None or goal is None:
                raise RuntimeError(f"branch lacks physical state at milestone {target}")
            result[f"physical_cost_at_{target}"] = physical_cost(task, current, goal)
            result[f"current_at_{target}"] = current
            result[f"goal_at_{target}"] = goal
            result[f"raw_env_step_at_{target}"] = int(step["raw_env_step"])
        else:
            result[f"physical_cost_at_{target}"] = None
            result[f"current_at_{target}"] = None
            result[f"goal_at_{target}"] = None
            result[f"raw_env_step_at_{target}"] = None
    return result


def fixed_pool_rank_metrics(
    costs: np.ndarray,
    truth_rows: Sequence[Mapping[str, Any]],
    *,
    state_count: int,
    candidate_count: int,
) -> dict[str, Any]:
    """Compute per-source-state selector outcomes and descriptive pool metrics."""
    costs = np.asarray(costs, dtype=np.float64)
    if costs.shape != (state_count, candidate_count) or not np.isfinite(costs).all():
        raise ValueError("candidate costs must be finite with shape [states, candidates]")
    if len(truth_rows) != state_count * candidate_count:
        raise ValueError("branch truth rows do not cover every candidate and state")
    truth_by_slot: list[list[Mapping[str, Any] | None]] = [
        [None] * candidate_count for _ in range(state_count)
    ]
    for row in truth_rows:
        slot, candidate = int(row["slot"]), int(row["candidate_index"])
        if not (0 <= slot < state_count and 0 <= candidate < candidate_count):
            raise ValueError("candidate truth row contains an out-of-range index")
        if truth_by_slot[slot][candidate] is not None:
            raise ValueError("duplicate candidate truth row")
        truth_by_slot[slot][candidate] = row
    if any(item is None for group in truth_by_slot for item in group):
        raise ValueError("candidate truth matrix contains missing rows")

    state_rows: list[dict[str, Any]] = []
    for slot, group in enumerate(truth_by_slot):
        truth = [item for item in group if item is not None]
        success = np.asarray([bool(item["success_by_25"]) for item in truth])
        valid = np.asarray([bool(item["valid_at_25"]) for item in truth])
        physical = np.asarray(
            [
                np.nan if item["physical_cost_at_25"] is None else float(item["physical_cost_at_25"])
                for item in truth
            ],
            dtype=np.float64,
        )
        selected = int(np.argmin(costs[slot]))
        valid_indices = np.flatnonzero(valid & np.isfinite(physical))
        selected_valid = int(valid_indices[np.argmin(costs[slot, valid_indices])]) if len(valid_indices) else None
        if len(valid_indices) == candidate_count:
            oracle_cost = float(np.min(physical))
            selected_cost = float(physical[selected])
            regret = selected_cost - oracle_cost
        else:
            oracle_cost = selected_cost = regret = None
        common_regret = None
        if selected_valid is not None:
            common_regret = float(physical[selected_valid] - np.min(physical[valid_indices]))
        if len(valid_indices) > 1 and np.ptp(costs[slot, valid_indices]) > 0 and np.ptp(physical[valid_indices]) > 0:
            rho = float(spearmanr(costs[slot, valid_indices], physical[valid_indices]).statistic)
            if not np.isfinite(rho):
                rho = None
        else:
            rho = None
        state_rows.append(
            {
                "slot": slot,
                "state_id": truth[0]["state_id"],
                "episode_id": truth[0]["episode_id"],
                "row_index": truth[0]["row_index"],
                "start_step": truth[0]["start_step"],
                "selected_candidate_index": selected,
                "selected_success_by_25": bool(success[selected]),
                "pool_success_fraction_25": float(success.mean()),
                "pool_has_success_by_25": bool(success.any()),
                "valid_candidate_count_25": int(len(valid_indices)),
                "complete_pool_valid_at_25": bool(len(valid_indices) == candidate_count),
                "selected_physical_cost_at_25": (
                    float(physical[selected]) if np.isfinite(physical[selected]) else None
                ),
                "complete_pool_oracle_cost_at_25": oracle_cost,
                "complete_pool_regret_at_25": regret,
                "valid_subpool_selected_index": selected_valid,
                "valid_subpool_regret_at_25": common_regret,
                "valid_subpool_spearman": rho,
            }
        )

    def mean(key: str) -> float | None:
        values = [row[key] for row in state_rows if row[key] is not None]
        return float(np.mean(values)) if values else None

    return {
        "state_count": int(state_count),
        "candidate_count": int(candidate_count),
        "complete_pool_state_count": sum(row["complete_pool_valid_at_25"] for row in state_rows),
        "mean_selected_success_by_25": mean("selected_success_by_25"),
        "mean_pool_success_fraction_25": mean("pool_success_fraction_25"),
        "mean_pool_has_success_by_25": mean("pool_has_success_by_25"),
        "mean_valid_candidate_count_25": mean("valid_candidate_count_25"),
        "mean_complete_pool_regret_at_25": mean("complete_pool_regret_at_25"),
        "mean_valid_subpool_regret_at_25": mean("valid_subpool_regret_at_25"),
        "mean_valid_subpool_spearman": mean("valid_subpool_spearman"),
        "by_state": state_rows,
    }

