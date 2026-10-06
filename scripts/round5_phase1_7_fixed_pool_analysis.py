#!/usr/bin/env python3
"""Recompute paired top-k success coverage from a Phase1.7 fixed-pool artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paired_bootstrap(
    difference: np.ndarray, *, scorer: str, seed: int, draws: int
) -> dict:
    difference = np.asarray(difference, dtype=np.float64)
    if difference.ndim != 1 or not len(difference) or draws < 1:
        raise ValueError("paired bootstrap needs states and at least one draw")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(draws, len(difference)))
    means = difference[indices].mean(axis=1)
    return {
        "n_states": int(len(difference)),
        f"mean_{scorer}_minus_joint": float(difference.mean()),
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
    }


def analyze(artifact_dir: Path, *, seed: int, draws: int) -> Path:
    artifact_dir = artifact_dir.resolve()
    summary_path = artifact_dir / "summary.json"
    pool_path = artifact_dir / "fixed_pool.npz"
    branch_dir = artifact_dir / "branches"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scorer = str(summary.get("scorer", "lewm"))
    with np.load(pool_path) as pool:
        joint_costs = np.asarray(pool["joint_costs"], dtype=np.float64)
        scorer_cost_key = f"{scorer}_costs"
        scorer_costs = np.asarray(pool[scorer_cost_key], dtype=np.float64)

    if joint_costs.shape != scorer_costs.shape or joint_costs.ndim != 2:
        raise ValueError(
            "Joint and verifier cost matrices must have one shared [state, candidate] shape"
        )
    state_count, candidate_count = joint_costs.shape
    if state_count < 1 or candidate_count < 1:
        raise ValueError("fixed_pool.npz must contain at least one state and candidate")
    if state_count != int(summary["state_count"]):
        raise ValueError("fixed_pool.npz state count differs from summary.json")
    if candidate_count != int(summary["candidate_count"]):
        raise ValueError("fixed_pool.npz candidate count differs from summary.json")

    successes = np.zeros((state_count, candidate_count), dtype=bool)
    seen = np.zeros((state_count, candidate_count), dtype=bool)
    outcome_files = sorted(branch_dir.glob("candidate_*.jsonl"))
    if len(outcome_files) != candidate_count:
        raise ValueError(
            f"expected {candidate_count} candidate files, found {len(outcome_files)}"
        )
    outcome_record_count = 0
    for path in outcome_files:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                slot = int(row["slot"])
                candidate = int(row["candidate_index"])
                if not (0 <= slot < state_count and 0 <= candidate < candidate_count):
                    raise ValueError(
                        f"outcome index is outside the captured pool: {slot}, {candidate}"
                    )
                if seen[slot, candidate]:
                    raise ValueError(
                        f"duplicate branch outcome for state={slot}, candidate={candidate}"
                    )
                if "success_by_25" not in row:
                    raise ValueError(f"branch outcome lacks success_by_25: {path}")
                if not isinstance(row["success_by_25"], bool):
                    raise ValueError(f"success_by_25 must be a boolean: {path}")
                if not np.isclose(
                    float(row["predicted_cost_joint"]),
                    joint_costs[slot, candidate],
                    rtol=1e-6,
                    atol=1e-8,
                ):
                    raise ValueError("Joint branch score differs from fixed_pool.npz")
                if not np.isclose(
                    float(row[f"predicted_cost_{scorer}"]),
                    scorer_costs[slot, candidate],
                    rtol=1e-6,
                    atol=1e-8,
                ):
                    raise ValueError("verifier branch score differs from fixed_pool.npz")
                seen[slot, candidate] = True
                successes[slot, candidate] = bool(row["success_by_25"])
                outcome_record_count += 1
    if not seen.all():
        missing = np.argwhere(~seen)
        raise ValueError(
            "branch outcomes are incomplete; first missing indices: "
            f"{missing[:5].tolist()}"
        )
    if outcome_record_count != int(summary["outcome_record_count"]):
        raise ValueError("branch record count differs from summary.json")

    per_state_success_count = successes.sum(axis=1)
    topk_results = []
    for k in (1, 3, 5, 10, 20):
        width = min(k, candidate_count)
        ranked_success: dict[str, np.ndarray] = {}
        for name, costs in (("joint", joint_costs), (scorer, scorer_costs)):
            order = np.argsort(costs, axis=1, kind="stable")[:, :width]
            ranked_success[name] = np.take_along_axis(successes, order, axis=1).any(axis=1)

        joint = ranked_success["joint"]
        scored = ranked_success[scorer]
        scorer_only = int(np.sum(~joint & scored))
        joint_only = int(np.sum(joint & ~scored))
        discordant = scorer_only + joint_only
        random_expectation = np.asarray(
            [
                1.0
                - math.comb(candidate_count - int(count), width)
                / math.comb(candidate_count, width)
                if count < candidate_count
                else 1.0
                for count in per_state_success_count
            ],
            dtype=np.float64,
        )
        topk_results.append(
            {
                "k": int(width),
                "joint_coverage": float(joint.mean()),
                f"{scorer}_coverage": float(scored.mean()),
                "random_expected_coverage": float(random_expectation.mean()),
                f"paired_bootstrap_{scorer}_minus_joint": _paired_bootstrap(
                    scored.astype(np.float64) - joint.astype(np.float64),
                    scorer=scorer,
                    seed=seed + width,
                    draws=draws,
                ),
                "discordant_states": {
                    "joint_only_success": joint_only,
                    f"{scorer}_only_success": scorer_only,
                },
                "exact_mcnemar_two_sided_p": float(
                    binomtest(min(joint_only, scorer_only), discordant, 0.5).pvalue
                    if discordant
                    else 1.0
                ),
                "p_value_adjusted_for_k_scan": False,
            }
        )

    top1_observations = (
        ("joint", topk_results[0]["joint_coverage"]),
        (scorer, topk_results[0][f"{scorer}_coverage"]),
    )
    for name, observed in top1_observations:
        expected = float(summary["metrics"][name]["mean"]["top1_success_25"])
        if not np.isclose(observed, expected, atol=1e-12):
            raise ValueError(f"top-1 coverage for {name} differs from summary metrics")

    joint_rows = summary["metrics"]["joint"]["by_state"]
    scorer_rows = summary["metrics"][scorer]["by_state"]
    paired_metric_differences = {}
    for offset, key in enumerate(
        (
            "top1_success_5",
            "top1_success_25",
            "top1_distance_25",
            "physical_regret_25",
            "pairwise_sign_accuracy",
            "spearman_cost_distance",
            "oracle_success_25",
            "oracle_best_top5",
        )
    ):
        joint_values = np.asarray(
            [np.nan if row[key] is None else row[key] for row in joint_rows],
            dtype=np.float64,
        )
        scorer_values = np.asarray(
            [np.nan if row[key] is None else row[key] for row in scorer_rows],
            dtype=np.float64,
        )
        if len(joint_values) != state_count or len(scorer_values) != state_count:
            raise ValueError(f"state-level metric count differs from the fixed pool: {key}")
        valid = np.isfinite(joint_values) & np.isfinite(scorer_values)
        paired_metric_differences[key] = _paired_bootstrap(
            scorer_values[valid] - joint_values[valid],
            scorer=scorer,
            seed=seed + 1000 + offset,
            draws=draws,
        )

    result = {
        "schema_version": 1,
        "task": summary["task"],
        "scorer": scorer,
        "cohort_id": summary["cohort_id"],
        "cohort_sha256": summary["cohort_sha256"],
        "state_count": state_count,
        "candidate_count": candidate_count,
        "success_event": "success_by_25",
        "random_baseline": (
            "exact hypergeometric expectation from each state's candidate successes"
        ),
        "bootstrap_unit": "starting state",
        "bootstrap_seed": int(seed),
        "bootstrap_draws": int(draws),
        "topk_p_values_adjusted_for_scan": False,
        "source_artifacts": {
            "summary_sha256": _sha256(summary_path),
            "fixed_pool_sha256": _sha256(pool_path),
            "outcome_record_count": outcome_record_count,
        },
        f"paired_metric_differences_{scorer}_minus_joint": paired_metric_differences,
        "top1_success_25_exact_mcnemar": summary[
            f"paired_comparison_{scorer}_minus_joint"
        ][
            "top1_success_25_exact_mcnemar"
        ],
        "topk": topk_results,
    }
    output_path = artifact_dir / "success_coverage_analysis.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(output_path),
                "state_count": state_count,
                "candidate_count": candidate_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=16031)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    analyze(args.artifact_dir, seed=args.bootstrap_seed, draws=args.bootstrap_draws)


if __name__ == "__main__":
    main()
