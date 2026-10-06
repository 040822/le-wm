#!/usr/bin/env python3
"""Estimate task-by-scorer interactions from paired Phase1.7 fixed-pool summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paired_effect(summary: dict, metric: str, scorer: str) -> np.ndarray:
    joint_rows = summary["metrics"]["joint"]["by_state"]
    scorer_rows = summary["metrics"][scorer]["by_state"]
    if len(joint_rows) != len(scorer_rows):
        raise ValueError(f"Joint and scorer state counts differ for {summary['task']}")
    joint = np.asarray(
        [np.nan if row[metric] is None else row[metric] for row in joint_rows],
        dtype=np.float64,
    )
    scored = np.asarray(
        [np.nan if row[metric] is None else row[metric] for row in scorer_rows],
        dtype=np.float64,
    )
    valid = np.isfinite(joint) & np.isfinite(scored)
    effect = scored[valid] - joint[valid]
    if not len(effect):
        raise ValueError(f"no paired states for {summary['task']} metric {metric}")
    return effect


def analyze(
    pusht_summary_path: Path,
    reacher_summary_path: Path,
    output_path: Path,
    *,
    seed: int,
    draws: int,
) -> Path:
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    pusht_summary_path = pusht_summary_path.resolve()
    reacher_summary_path = reacher_summary_path.resolve()
    pusht = json.loads(pusht_summary_path.read_text(encoding="utf-8"))
    reacher = json.loads(reacher_summary_path.read_text(encoding="utf-8"))
    if pusht["task"] != "pusht" or reacher["task"] != "reacher":
        raise ValueError("provide PushT and Reacher fixed-pool summaries in the correct order")
    if pusht["candidate_count"] != reacher["candidate_count"]:
        raise ValueError("task comparisons use different candidate pool sizes")
    pusht_scorer = str(pusht.get("scorer", "lewm"))
    reacher_scorer = str(reacher.get("scorer", "lewm"))
    if pusht_scorer != reacher_scorer:
        raise ValueError("PushT and Reacher summaries use different verifier arms")
    scorer = pusht_scorer

    metric_results = {}
    for offset, metric in enumerate(
        ("pairwise_sign_accuracy", "spearman_cost_distance", "top1_success_25")
    ):
        pusht_effect = _paired_effect(pusht, metric, scorer)
        reacher_effect = _paired_effect(reacher, metric, scorer)
        rng = np.random.default_rng(seed + offset)
        pusht_indices = rng.integers(
            0, len(pusht_effect), size=(draws, len(pusht_effect))
        )
        reacher_indices = rng.integers(
            0, len(reacher_effect), size=(draws, len(reacher_effect))
        )
        interaction_draws = (
            reacher_effect[reacher_indices].mean(axis=1)
            - pusht_effect[pusht_indices].mean(axis=1)
        )
        metric_results[metric] = {
            f"pusht_{scorer}_minus_joint": float(pusht_effect.mean()),
            "pusht_paired_state_count": int(len(pusht_effect)),
            f"reacher_{scorer}_minus_joint": float(reacher_effect.mean()),
            "reacher_paired_state_count": int(len(reacher_effect)),
            "interaction_reacher_minus_pusht": float(
                reacher_effect.mean() - pusht_effect.mean()
            ),
            "interaction_ci95": [
                float(value)
                for value in np.quantile(interaction_draws, [0.025, 0.975])
            ],
        }

    result = {
        "schema_version": 1,
        "interaction_definition": (
            f"mean({scorer} - Joint in Reacher) - mean({scorer} - Joint in PushT)"
        ),
        "scorer": scorer,
        "bootstrap_unit": "starting state, resampled separately within each task",
        "bootstrap_seed": int(seed),
        "bootstrap_draws": int(draws),
        "intervals_adjusted_for_metric_scan": False,
        "pusht_cohort_id": pusht["cohort_id"],
        "reacher_cohort_id": reacher["cohort_id"],
        "source_summaries": {
            "pusht_sha256": _sha256(pusht_summary_path),
            "reacher_sha256": _sha256(reacher_summary_path),
        },
        "metrics": metric_results,
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps({"status": "ok", "output": str(output_path)}, sort_keys=True))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pusht-summary", required=True, type=Path)
    parser.add_argument("--reacher-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=16033)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    analyze(
        args.pusht_summary,
        args.reacher_summary,
        args.output,
        seed=args.bootstrap_seed,
        draws=args.bootstrap_draws,
    )


if __name__ == "__main__":
    main()
