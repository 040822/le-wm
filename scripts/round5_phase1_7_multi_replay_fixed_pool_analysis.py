#!/usr/bin/env python3
"""Compare saved fixed-pool rankers against each independently saved replay.

This is an epoch-2 dev repeatability diagnostic. It does not combine rollout
outcomes across replays or treat them as terminal/confirmatory evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.round5_phase1_7_fixed_pool import _rank_metrics
from scripts.round5_phase1_7_rescore import _load_outcomes, _metric_comparison


TASKS = ("pusht", "reacher")
SEEDS = (3072, 4096)
ARRAYS_TO_MATCH = ("candidates", "physical_actions", "candidate_noise", "joint_costs")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pool(summary: dict) -> dict[str, np.ndarray]:
    path = Path(summary["fixed_pool_artifact"]).resolve()
    with np.load(path) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _index_outcomes(rows: list[dict]) -> dict[tuple[int, int], dict]:
    result = {}
    for row in rows:
        key = (int(row["slot"]), int(row["candidate_index"]))
        if key in result:
            raise ValueError(f"duplicate physical outcome key: {key}")
        result[key] = row
    return result


def _replay_difference(left: list[dict], right: list[dict]) -> dict:
    left_by_key = _index_outcomes(left)
    right_by_key = _index_outcomes(right)
    if left_by_key.keys() != right_by_key.keys():
        raise ValueError("the two replay outcome sets do not contain matching branches")

    success_changed = 0
    valid_changed = 0
    distance_changed = 0
    distance_abs_diffs = []
    for key in left_by_key:
        a, b = left_by_key[key], right_by_key[key]
        success_changed += int(bool(a["success_by_25"]) != bool(b["success_by_25"]))
        valid_changed += int(bool(a["valid_at_25"]) != bool(b["valid_at_25"]))
        da, db = a.get("distance_at_25"), b.get("distance_at_25")
        if da is None or db is None:
            distance_changed += int(da != db)
        else:
            delta = abs(float(da) - float(db))
            distance_changed += int(delta > 1e-9)
            distance_abs_diffs.append(delta)
    return {
        "matched_branch_count": len(left_by_key),
        "success_by_25_changed_count": success_changed,
        "valid_at_25_changed_count": valid_changed,
        "distance_at_25_changed_count": distance_changed,
        "mean_abs_distance_at_25_difference_when_both_numeric": (
            float(np.mean(distance_abs_diffs)) if distance_abs_diffs else None
        ),
    }


def _analyze_pair(task: str, seed: int) -> dict:
    cohort_id = f"{task}_phase1_7_dev_16027_v1"
    base = ROOT / "outputs/round5/phase1_7"
    fixed_root = base / f"epoch2_retry_ablation_dev_fixed_pool_s{seed}" / task / cohort_id
    b_summary_path = fixed_root / "b_only/summary.json"
    rc_summary_path = fixed_root / "recorded_control/summary.json"
    pair_root = base / "epoch2_retry_ablation_fixed_pool_pair" / f"{task}_s{seed}"
    pair_json_path = pair_root / "paired_analysis.json"
    scores_path = pair_root / "paired_analysis.scores.npz"

    b_summary = json.loads(b_summary_path.read_text(encoding="utf-8"))
    rc_summary = json.loads(rc_summary_path.read_text(encoding="utf-8"))
    pair_metadata = json.loads(pair_json_path.read_text(encoding="utf-8"))
    if b_summary.get("scorer") != "b_only" or rc_summary.get("scorer") != "recorded_control":
        raise ValueError(f"unexpected scorer summaries for {task}/{seed}")
    if pair_metadata.get("training_epoch") != 2 or pair_metadata.get("state_count") != 100:
        raise ValueError(f"unexpected analysis cohort for {task}/{seed}")
    for key in ("task", "cohort_sha256"):
        if b_summary.get(key) != rc_summary.get(key):
            raise ValueError(f"B-only and Recorded-control {key} differ for {task}/{seed}")
    if b_summary.get("cohort_sha256") != pair_metadata.get("cohort_sha256"):
        raise ValueError(f"paired analysis cohort differs for {task}/{seed}")
    if b_summary.get("actor_checkpoint_sha256") != rc_summary.get("actor_checkpoint_sha256"):
        raise ValueError(f"actor checkpoints differ for {task}/{seed}")

    b_pool = _load_pool(b_summary)
    rc_pool = _load_pool(rc_summary)
    with np.load(scores_path) as archive:
        scores = {key: archive[key].copy() for key in archive.files}
    for key in ARRAYS_TO_MATCH:
        if not np.array_equal(b_pool[key], rc_pool[key]):
            raise ValueError(f"B-only and Recorded-control {key} differ for {task}/{seed}")
        if not np.array_equal(b_pool[key], scores[key]):
            raise ValueError(f"saved scorer array {key} differs from reference pool for {task}/{seed}")
    if not np.array_equal(b_pool["b_only_costs"], scores["b_only_costs"]):
        raise ValueError(f"saved B-only costs differ from reference pool for {task}/{seed}")
    if not np.array_equal(scores["candidates"], rc_pool["candidates"]):
        raise ValueError(f"Recorded-control action candidates differ for {task}/{seed}")

    b_outcomes = _load_outcomes(b_summary_path, b_pool["candidates"])
    rc_outcomes = _load_outcomes(rc_summary_path, rc_pool["candidates"])
    rankers = {
        "joint_b": scores["joint_costs"],
        "b_only": scores["b_only_costs"],
        "lewm": scores["lewm_costs"],
    }
    replay_results = {}
    for replay_name, outcomes in (("b_only_replay", b_outcomes), ("recorded_control_replay", rc_outcomes)):
        model_metrics = {
            name: _rank_metrics(costs, outcomes, seed=16027)
            for name, costs in rankers.items()
        }
        replay_results[replay_name] = {
            "outcome_record_count": len(outcomes),
            "mean_metrics": {name: value["mean"] for name, value in model_metrics.items()},
            "paired_comparisons": {
                "joint_b_minus_lewm": _metric_comparison(
                    model_metrics["lewm"], model_metrics["joint_b"]
                ),
                "b_only_minus_lewm": _metric_comparison(
                    model_metrics["lewm"], model_metrics["b_only"]
                ),
            },
        }

    return {
        "task": task,
        "training_seed": seed,
        "cohort_id": cohort_id,
        "state_count": int(b_summary["state_count"]),
        "candidate_count": int(b_summary["candidate_count"]),
        "candidate_pool_exact_across_replays": True,
        "environment_replayed_for_lewm": False,
        "input_hashes": {
            "b_only_summary_sha256": _sha256(b_summary_path),
            "recorded_control_summary_sha256": _sha256(rc_summary_path),
            "paired_scores_sha256": _sha256(scores_path),
        },
        "replay_outcome_differences": _replay_difference(b_outcomes, rc_outcomes),
        "results_by_replay": replay_results,
    }


def main() -> None:
    output_path = (
        ROOT
        / "outputs/round5/phase1_7/epoch2_retry_ablation_multi_replay/fixed_pool_multi_replay_analysis.json"
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis: {output_path}")
    results = [_analyze_pair(task, seed) for task in TASKS for seed in SEEDS]
    payload = {
        "schema_version": 1,
        "classification": "epoch-2 dev; same saved scores and exact action pool evaluated against two separate nonidentical replays; not terminal or confirmation",
        "bootstrap_unit": "starting state within each replay",
        "replays_pooled": False,
        "intervals_adjusted_for_metric_scan": False,
        "comparisons": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(output_path)
    print(json.dumps({"status": "ok", "output": str(output_path), "comparisons": len(results)}))


if __name__ == "__main__":
    main()
