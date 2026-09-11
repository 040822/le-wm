"""Frozen settings and decision rules for the Round 4 exploration."""

from __future__ import annotations

from typing import Any, Mapping


ROUND4_PROTOCOL = "round3_revised"
ROUND4_TRAINING_SEED = 3072
ROUND4_EXPANSION_SEEDS = (3073, 3074)
ROUND4_TASKS = ("cube", "pusht", "reacher", "tworoom")
ROUND4_MODES = ("P0", "P0-shuf", "P1", "P2", "P3", "P4", "P4-first")

ROUND4_DEFAULTS: dict[str, Any] = {
    "dev_episodes": 50,
    "final_episodes": 200,
    "seed": 42,
    "goal_offset_steps": 25,
    "eval_budget": 50,
    "horizon": 5,
    "receding_horizon": 5,
    "action_block": 5,
    "cem": {
        "num_samples": 300,
        "topk": 30,
        "n_steps": 30,
        "var_scale": 1.0,
    },
    "best_of_n": {
        "num_candidates": 64,
        "flow_steps": 16,
        "solver_batch_size": 1,
    },
}


def validate_round4_mode(mode: str) -> str:
    value = str(mode)
    if value not in ROUND4_MODES:
        raise ValueError(f"unknown Round 4 mode {mode!r}; expected {ROUND4_MODES}")
    return value


def mode_spec(mode: str) -> dict[str, Any]:
    """Return the non-negotiable proposal/verifier contract for one mode."""
    validate_round4_mode(mode)
    if mode == "P0":
        return {
            "proposal_source": "action",
            "candidate_count": 1,
            "flow_steps": 16,
            "idm_type": "none",
            "verifier": "none",
            "selection_rule": "direct",
        }
    if mode == "P0-shuf":
        spec = mode_spec("P0")
        spec["goal_mode"] = "shuffled"
        return spec
    if mode == "P1":
        return {
            "proposal_source": "cem_random",
            "candidate_count": 300,
            "flow_steps": None,
            "idm_type": "none",
            "verifier": "stage_b",
            "selection_rule": "cem_argmin",
            "cem_iterations": 30,
            "cem_topk": 30,
        }
    if mode == "P2":
        spec = mode_spec("P1")
        spec.update({"proposal_source": "cem_actor_warm_start", "actor_flow_steps": 10})
        return spec
    if mode == "P3":
        return {
            "proposal_source": "action",
            "candidate_count": 64,
            "flow_steps": 16,
            "idm_type": "none",
            "verifier": "stage_b",
            "selection_rule": "argmin_verifier",
        }
    if mode == "P4-first":
        return {
            "proposal_source": "latent",
            "candidate_count": 64,
            "flow_steps": 16,
            "idm_type": "direct_shared_dit",
            "verifier": "none",
            "selection_rule": "first",
        }
    return {
        "proposal_source": "latent",
        "candidate_count": 64,
        "flow_steps": 16,
        "idm_type": "direct_shared_dit",
        "verifier": "stage_b",
        "selection_rule": "argmin_verifier",
    }


def should_expand_seed(
    dev_rows: Mapping[str, Mapping[str, Any]],
    *,
    candidate: str = "P4",
    baseline: str = "P3",
    planning_baseline: str = "P2",
) -> tuple[bool, dict[str, Any]]:
    """Apply the pre-registered epoch-10 dev expansion gate."""
    tasks = tuple(ROUND4_TASKS)

    def rate(task: str, mode: str) -> float:
        value = dev_rows[task][mode]
        if isinstance(value, Mapping):
            value = value.get("success_rate")
        if value is None:
            raise ValueError(f"missing success_rate for {task}/{mode}")
        return float(value)

    p4 = {task: rate(task, candidate) for task in tasks}
    p3 = {task: rate(task, baseline) for task in tasks}
    p2 = {task: rate(task, planning_baseline) for task in tasks}
    performance_delta_p3 = {task: 100.0 * (p4[task] - p3[task]) for task in tasks}
    performance_delta_p2 = {task: 100.0 * (p4[task] - p2[task]) for task in tasks}
    performance_gate_p3 = (
        sum(delta >= 5.0 for delta in performance_delta_p3.values()) >= 2
        and all(delta >= -5.0 for delta in performance_delta_p3.values())
    )
    performance_gate_p2 = (
        sum(delta >= 5.0 for delta in performance_delta_p2.values()) >= 2
        and all(delta >= -5.0 for delta in performance_delta_p2.values())
    )
    performance_gate = performance_gate_p3 or performance_gate_p2
    medians = {}
    for task in tasks:
        p4_row = dev_rows[task][candidate]
        p2_row = dev_rows[task][planning_baseline]
        p4_median = p4_row.get("planning_median_seconds") if isinstance(p4_row, Mapping) else None
        p2_median = p2_row.get("planning_median_seconds") if isinstance(p2_row, Mapping) else None
        medians[task] = {"P4": p4_median, "P2": p2_median}
    efficiency_values = [
        medians[task]["P4"] is not None
        and medians[task]["P2"] is not None
        and float(medians[task]["P4"]) <= 0.5 * float(medians[task]["P2"])
        for task in tasks
    ]
    efficiency_gate = (
        all(delta >= -2.0 for delta in performance_delta_p2.values())
        and sum(efficiency_values) >= 2
    )
    details = {
        "performance_gate": performance_gate,
        "performance_gate_vs_P3": performance_gate_p3,
        "performance_gate_vs_P2": performance_gate_p2,
        "efficiency_gate": efficiency_gate,
        "performance_delta_pp_vs_P3": performance_delta_p3,
        "performance_delta_pp_vs_P2": performance_delta_p2,
        "planning_medians": medians,
    }
    return bool(performance_gate or efficiency_gate), details


__all__ = [
    "ROUND4_DEFAULTS",
    "ROUND4_EXPANSION_SEEDS",
    "ROUND4_MODES",
    "ROUND4_PROTOCOL",
    "ROUND4_TASKS",
    "ROUND4_TRAINING_SEED",
    "mode_spec",
    "should_expand_seed",
    "validate_round4_mode",
]
