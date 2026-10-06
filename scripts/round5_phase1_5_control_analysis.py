#!/usr/bin/env python3
"""State-paired analysis for the fixed Phase1.5 control action pool."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.common.round5_phase1_5 import normalized_physical_distance


BOOTSTRAP_SAMPLES = 10_000
EXPECTED_STATES = 50
EXPECTED_CONTROLS = 222


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _outcome(row: Mapping[str, Any]) -> tuple[float, float]:
    distance = float(row["true_distance"])
    success = float(bool(row["success"]))
    if not math.isfinite(distance):
        raise ValueError("control true_distance must be finite")
    return distance, success


def _state_mean(rows: Iterable[Mapping[str, Any]]) -> tuple[float, float]:
    outcomes = [_outcome(row) for row in rows]
    if not outcomes:
        raise ValueError("cannot summarize an empty per-state action group")
    values = np.asarray(outcomes, dtype=np.float64)
    return float(values[:, 0].mean()), float(values[:, 1].mean())


def _bootstrap(
    values: Iterable[float], *, seed: int, expected_states: int | None = EXPECTED_STATES
) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if (
        array.ndim != 1
        or len(array) == 0
        or (expected_states is not None and len(array) != expected_states)
        or not np.all(np.isfinite(array))
    ):
        raise ValueError("expected a non-empty finite vector of state-level values")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    draws = array[indices].mean(axis=1)
    return {
        "estimate": float(array.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "state_count": int(len(array)),
        "cohort_coverage": float(len(array) / EXPECTED_STATES),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": int(seed),
    }


def _summarize_state_values(
    per_state: Mapping[str, tuple[float, float]], *, seed: int
) -> dict[str, Any]:
    state_ids = sorted(per_state)
    return {
        "states": len(state_ids),
        "absolute": {
            "recorded_endpoint_distance": _bootstrap(
                (per_state[state][0] for state in state_ids), seed=seed
            ),
            "success_rate": _bootstrap(
                (per_state[state][1] for state in state_ids), seed=seed + 1
            ),
        },
    }


def _summarize_paired(
    per_state: Mapping[str, tuple[float, float]],
    baseline: Mapping[str, tuple[float, float]],
    *,
    seed: int,
    baseline_label: str,
    variants_per_state: int,
) -> dict[str, Any]:
    if set(per_state) != set(baseline) or len(per_state) != EXPECTED_STATES:
        raise ValueError("paired control and baseline state sets must match all 50 states")
    result = _summarize_state_values(per_state, seed=seed)
    state_ids = sorted(per_state)
    result["paired_to"] = baseline_label
    result["variants_per_state"] = int(variants_per_state)
    result["paired_effect"] = {
        "recorded_endpoint_distance_delta": _bootstrap(
            (per_state[state][0] - baseline[state][0] for state in state_ids),
            seed=seed + 2,
        ),
        "success_rate_delta": _bootstrap(
            (per_state[state][1] - baseline[state][1] for state in state_ids),
            seed=seed + 3,
        ),
    }
    return result


def _exact_milestone_distance(
    task: str, row: Mapping[str, Any], target: int
) -> float | None:
    milestone = row.get("milestones", {}).get(str(target))
    if not isinstance(milestone, Mapping) or int(milestone.get("raw_env_step", -1)) != target:
        return None
    current = milestone.get("current")
    goal = milestone.get("goal")
    if current is None or goal is None:
        return None
    value = float(normalized_physical_distance(task, current, goal))
    return value if math.isfinite(value) else None


def _latest_common_milestone(
    task: str,
    treatment: Mapping[str, Any],
    baseline_rows: Iterable[Mapping[str, Any]],
) -> tuple[int, float, float] | None:
    baselines = list(baseline_rows)
    if not baselines:
        raise ValueError("a common-milestone comparison needs a baseline")
    for target in (25, 20, 15, 10, 5):
        treatment_distance = _exact_milestone_distance(task, treatment, target)
        baseline_distances = [
            _exact_milestone_distance(task, row, target) for row in baselines
        ]
        if treatment_distance is not None and all(
            distance is not None for distance in baseline_distances
        ):
            return (
                target,
                treatment_distance,
                float(np.mean(np.asarray(baseline_distances, dtype=np.float64))),
            )
    return None


def _summarize_common_milestone_distance(
    task: str,
    controls_by_state: Mapping[str, list[dict[str, Any]]],
    baselines_by_state: Mapping[str, list[dict[str, Any]]],
    *,
    seed: int,
) -> dict[str, Any]:
    state_values: dict[str, tuple[float, float, float]] = {}
    pairs_per_state: dict[str, int] = {}
    step_counts: dict[int, int] = defaultdict(int)
    pair_count = 0
    for state_id, control_rows in controls_by_state.items():
        effects: list[tuple[int, float, float]] = []
        for control in control_rows:
            matched = _latest_common_milestone(
                task, control, baselines_by_state[state_id]
            )
            if matched is not None:
                effects.append(matched)
                step_counts[matched[0]] += 1
        pairs_per_state[state_id] = len(effects)
        if effects:
            pair_count += len(effects)
            state_values[state_id] = (
                float(np.mean([item[1] for item in effects])),
                float(np.mean([item[2] for item in effects])),
                float(np.mean([item[1] - item[2] for item in effects])),
            )
    if not state_values:
        return {
            "paired_action_state_count": 0,
            "total_action_state_count": sum(len(rows) for rows in controls_by_state.values()),
            "state_count_with_pairs": 0,
            "state_coverage": 0.0,
            "matched_raw_step_counts": {str(step): step_counts.get(step, 0) for step in (5, 10, 15, 20, 25)},
            "paired_effect": None,
        }
    state_ids = sorted(state_values)
    return {
        "definition": "For each control action and state, use the latest exact shared raw step from 5/10/15/20/25; average valid action pairs within state, then bootstrap states.",
        "paired_action_state_count": pair_count,
        "total_action_state_count": sum(len(rows) for rows in controls_by_state.values()),
        "state_count_with_pairs": len(state_ids),
        "state_coverage": float(len(state_ids) / EXPECTED_STATES),
        "valid_pairs_per_state": {
            "mean_across_all_states": float(
                np.mean([pairs_per_state.get(state_id, 0) for state_id in controls_by_state])
            ),
            "min": int(min(pairs_per_state.values())),
            "max": int(max(pairs_per_state.values())),
        },
        "matched_raw_step_counts": {
            str(step): int(step_counts.get(step, 0)) for step in (5, 10, 15, 20, 25)
        },
        "absolute_control_distance": _bootstrap(
            (state_values[state_id][0] for state_id in state_ids),
            seed=seed,
            expected_states=None,
        ),
        "absolute_baseline_distance": _bootstrap(
            (state_values[state_id][1] for state_id in state_ids),
            seed=seed + 1,
            expected_states=None,
        ),
        "paired_effect": _bootstrap(
            (state_values[state_id][2] for state_id in state_ids),
            seed=seed + 2,
            expected_states=None,
        ),
    }


def _stable_seed(base_seed: int, label: str) -> int:
    return int(base_seed) + zlib.crc32(label.encode("utf-8")) % 1_000_000


def analyze_task(output_root: Path, task: str, base_seed: int) -> dict[str, Any]:
    task_root = output_root / "diagnostics" / "candidate_pool" / task
    manifest_path = task_root / "manifest.json"
    records_path = task_root / "controls.jsonl"
    if not manifest_path.exists() or not records_path.exists():
        raise FileNotFoundError(f"missing completed control artifacts for {task}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed"
        or manifest.get("control_status") != "completed"
        or int(manifest.get("control_actions", -1)) != EXPECTED_CONTROLS
    ):
        raise ValueError(f"{task} manifest does not describe a completed 222-action pool")

    rows = _read_jsonl(records_path)
    if len(rows) != EXPECTED_CONTROLS * EXPECTED_STATES:
        raise ValueError(
            f"{task} has {len(rows)} control rows; expected "
            f"{EXPECTED_CONTROLS * EXPECTED_STATES}"
        )

    states: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    metadata_by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        state_id = str(row["state_id"])
        index = int(row["control_index"])
        if row.get("outcome_status") != "completed":
            raise ValueError(f"{task} control {index} state {state_id} is not completed")
        metadata = dict(row["control_metadata"])
        if row.get("control_kind") != metadata.get("kind"):
            raise ValueError(f"{task} control {index} has inconsistent kind metadata")
        previous_metadata = metadata_by_index.setdefault(index, metadata)
        if metadata != previous_metadata:
            raise ValueError(f"{task} metadata differs across states for control {index}")
        _outcome(row)
        if index in states[state_id]:
            raise ValueError(f"{task} duplicates control {index} for state {state_id}")
        states[state_id][index] = row

    state_ids = sorted(states)
    expected_indices = set(range(EXPECTED_CONTROLS))
    if len(state_ids) != EXPECTED_STATES:
        raise ValueError(f"{task} has {len(state_ids)} states; expected {EXPECTED_STATES}")
    for state_id in state_ids:
        if set(states[state_id]) != expected_indices:
            raise ValueError(f"{task} state {state_id} does not contain all 222 controls")

    anchors: dict[int, dict[str, tuple[float, float]]] = {
        anchor: {} for anchor in range(3)
    }
    anchor_rows: dict[int, dict[str, dict[str, Any]]] = {
        anchor: {} for anchor in range(3)
    }
    rms: dict[tuple[int, float, int], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    blocks: dict[tuple[int, int], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    gaussians: dict[str, list[dict[str, Any]]] = defaultdict(list)
    physical_zero: dict[str, list[dict[str, Any]]] = defaultdict(list)
    normalized_zero: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for state_id in state_ids:
        for row in states[state_id].values():
            metadata = row["control_metadata"]
            kind = metadata["kind"]
            if kind == "anchor":
                anchor = int(metadata["anchor"])
                anchors[anchor][state_id] = _outcome(row)
                anchor_rows[anchor][state_id] = row
            elif kind == "rms_perturbation":
                key = (
                    int(metadata["anchor"]),
                    float(metadata["scale"]),
                    int(metadata["sign"]),
                )
                rms[key][state_id].append(row)
            elif kind == "block_transform":
                key = (int(metadata["anchor"]), int(metadata["permutation"]))
                blocks[key][state_id].append(row)
            elif kind == "standard_gaussian":
                gaussians[state_id].append(row)
            elif kind == "physical_zero":
                physical_zero[state_id].append(row)
            elif kind == "normalized_zero":
                normalized_zero[state_id].append(row)
            else:
                raise ValueError(f"{task} has unexpected control kind {kind!r}")

    for anchor in range(3):
        if len(anchors[anchor]) != EXPECTED_STATES:
            raise ValueError(f"{task} anchor {anchor} is missing states")
    for key, groups in rms.items():
        if len(groups) != EXPECTED_STATES or any(
            len(groups[state_id]) != 8
            or {int(row["control_metadata"]["direction"]) for row in groups[state_id]}
            != set(range(8))
            for state_id in state_ids
        ):
            raise ValueError(f"{task} RMS group {key} is not 8-direction complete")
    if len(rms) != 18:
        raise ValueError(f"{task} has {len(rms)} RMS groups; expected 18")
    for key, groups in blocks.items():
        if len(groups) != EXPECTED_STATES or any(len(groups[state_id]) != 1 for state_id in state_ids):
            raise ValueError(f"{task} block-transform group {key} is incomplete")
    if len(blocks) != 9:
        raise ValueError(f"{task} has {len(blocks)} block-transform groups; expected 9")
    for label, groups, expected_count in (
        ("standard_gaussian", gaussians, 64),
        ("physical_zero", physical_zero, 1),
        ("normalized_zero", normalized_zero, 1),
    ):
        if set(groups) != set(state_ids) or any(
            len(groups[state_id]) != expected_count for state_id in state_ids
        ):
            raise ValueError(f"{task} {label} coverage is incomplete")

    results: dict[str, Any] = {
        "task": task,
        "source": str(records_path),
        "manifest": str(manifest_path),
        "protocol": {
            "control_actions": EXPECTED_CONTROLS,
            "states": EXPECTED_STATES,
            "rows": len(rows),
            "recorded_endpoint_distance": "normalized distance at the latest recorded raw step at or before 25; an earlier termination uses its last valid step",
            "common_milestone_distance": "secondary comparison at the latest exact shared raw step among 5/10/15/20/25; only pairs with a shared milestone are included",
            "baseline_anchors": ["data", "S=1 first proposal", "S=2 first proposal"],
            "gaussian_action_definition": "64 fixed N(0,1) samples, each normalized to RMS=1; not elementwise iid Gaussian at execution",
            "bootstrap": "10,000 state-cluster resamples; average variants within state before resampling",
            "success_metric": "success event within the 50-step evaluation budget",
            "effect_sign": "negative distance deltas and positive success_rate_delta favor the control",
        },
        "limitations": [
            "Intervals describe the fixed 50-state legacy cohort; they do not establish generalization to new states.",
            "Each state/action has one recorded rollout, so intervals do not estimate rollout-to-rollout variability.",
            "RMS directions and the 64 Gaussian actions are fixed by the protocol; group comparisons are exploratory and have no multiplicity correction.",
            "Recorded endpoint-distance comparisons may use different raw steps after early termination; consult common_milestone_distance coverage before interpreting physical-distance deltas.",
        ],
        "anchors": {},
        "rms_perturbations": {},
        "block_transforms": {},
        "block_transform_means": {},
        "rms_normalized_gaussian": {},
        "physical_zero": {},
        "normalized_zero": {},
    }
    for anchor in range(3):
        label = f"anchor_{anchor}"
        results["anchors"][label] = _summarize_state_values(
            anchors[anchor], seed=_stable_seed(base_seed, f"{task}:{label}")
        )

    for key in sorted(rms):
        anchor, scale, sign = key
        label = f"anchor_{anchor}/scale_{scale:g}/sign_{'plus' if sign > 0 else 'minus'}"
        values = {
            state_id: _state_mean(rms[key][state_id]) for state_id in state_ids
        }
        results["rms_perturbations"][label] = _summarize_paired(
            values,
            anchors[anchor],
            seed=_stable_seed(base_seed, f"{task}:rms:{label}"),
            baseline_label=f"anchor_{anchor}",
            variants_per_state=8,
        )
        results["rms_perturbations"][label]["common_milestone_distance"] = (
            _summarize_common_milestone_distance(
                task,
                rms[key],
                {state_id: [anchor_rows[anchor][state_id]] for state_id in state_ids},
                seed=_stable_seed(base_seed, f"{task}:rms_common:{label}"),
            )
        )

    for key in sorted(blocks):
        anchor, permutation = key
        label = f"anchor_{anchor}/permutation_{permutation}"
        values = {
            state_id: _state_mean(blocks[key][state_id]) for state_id in state_ids
        }
        results["block_transforms"][label] = _summarize_paired(
            values,
            anchors[anchor],
            seed=_stable_seed(base_seed, f"{task}:block:{label}"),
            baseline_label=f"anchor_{anchor}",
            variants_per_state=1,
        )
        results["block_transforms"][label]["common_milestone_distance"] = (
            _summarize_common_milestone_distance(
                task,
                blocks[key],
                {state_id: [anchor_rows[anchor][state_id]] for state_id in state_ids},
                seed=_stable_seed(base_seed, f"{task}:block_common:{label}"),
            )
        )
    for anchor in range(3):
        label = f"anchor_{anchor}"
        values = {}
        for state_id in state_ids:
            rows_for_anchor = [
                blocks[(anchor, permutation)][state_id][0] for permutation in range(3)
            ]
            values[state_id] = _state_mean(rows_for_anchor)
        results["block_transform_means"][label] = _summarize_paired(
            values,
            anchors[anchor],
            seed=_stable_seed(base_seed, f"{task}:block_mean:{label}"),
            baseline_label=f"anchor_{anchor}",
            variants_per_state=3,
        )
        combined_rows = {
            state_id: [
                blocks[(anchor, permutation)][state_id][0] for permutation in range(3)
            ]
            for state_id in state_ids
        }
        results["block_transform_means"][label]["common_milestone_distance"] = (
            _summarize_common_milestone_distance(
                task,
                combined_rows,
                {state_id: [anchor_rows[anchor][state_id]] for state_id in state_ids},
                seed=_stable_seed(base_seed, f"{task}:block_mean_common:{label}"),
            )
        )

    composite_anchor = {
        state_id: (
            float(np.mean([anchors[anchor][state_id][0] for anchor in range(3)])),
            float(np.mean([anchors[anchor][state_id][1] for anchor in range(3)])),
        )
        for state_id in state_ids
    }
    for key, groups, destination, count in (
        ("standard_gaussian", gaussians, "rms_normalized_gaussian", 64),
        ("physical_zero", physical_zero, "physical_zero", 1),
        ("normalized_zero", normalized_zero, "normalized_zero", 1),
    ):
        values = {state_id: _state_mean(groups[state_id]) for state_id in state_ids}
        results[destination] = _summarize_paired(
            values,
            composite_anchor,
            seed=_stable_seed(base_seed, f"{task}:{key}"),
            baseline_label="same-state mean of the three anchors",
            variants_per_state=count,
        )
        results[destination]["common_milestone_distance"] = (
            _summarize_common_milestone_distance(
                task,
                groups,
                {
                    state_id: [anchor_rows[anchor][state_id] for anchor in range(3)]
                    for state_id in state_ids
                },
                seed=_stable_seed(base_seed, f"{task}:{key}:common"),
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", choices=("cube", "pusht", "reacher", "tworoom"), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    result = analyze_task(args.output_root, args.task, args.seed)
    output = (
        args.output_root
        / "diagnostics"
        / "candidate_pool"
        / args.task
        / "control_paired_effects.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"analysis": str(output), "task": args.task}, ensure_ascii=False))


if __name__ == "__main__":
    main()
