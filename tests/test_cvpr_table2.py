import math
import json
import threading

import numpy as np
import pytest

from source.common.cvpr_table2_diagnostics import (
    candidate_truth_record,
    fixed_pool_rank_metrics,
    physical_cost,
)


def test_pusht_physical_cost_wraps_angles_and_scales_by_success_thresholds():
    cost = physical_cost(
        "pusht",
        [0.0, 0.0, 0.0, 0.0, math.radians(179)],
        [0.0, 0.0, 0.0, 0.0, math.radians(-179)],
    )

    assert cost == pytest.approx(0.1, rel=1e-8)


def test_candidate_truth_keeps_early_native_success_separate_from_valid_25_step_state():
    entry = {"episode_id": 7, "row_index": 70, "start_step": 10}
    episode = {
        "success": True,
        "steps": [
            {
                "raw_env_step": 10,
                "current": [0.0, 0.0],
                "goal": [1.0, 0.0],
                "env_success": True,
            }
        ],
    }

    row = candidate_truth_record(
        task="tworoom",
        evaluation_seed=42,
        entry=entry,
        slot=0,
        candidate_index=3,
        episode=episode,
    )

    assert row["success_by_25"] is True
    assert row["valid_at_25"] is False
    assert row["physical_cost_at_25"] is None


def test_fixed_pool_ranking_reports_complete_pool_regret_and_valid_subpool():
    truth = []
    candidate_success = [False, True, False]
    candidate_cost = [3.0, 2.0, 1.0]
    for slot in range(2):
        for candidate in range(3):
            valid = not (slot == 1 and candidate == 2)
            truth.append(
                {
                    "slot": slot,
                    "candidate_index": candidate,
                    "state_id": f"state-{slot}",
                    "episode_id": slot,
                    "row_index": slot * 10,
                    "start_step": 2,
                    "success_by_25": candidate_success[candidate],
                    "valid_at_25": valid,
                    "physical_cost_at_25": candidate_cost[candidate] if valid else None,
                }
            )

    result = fixed_pool_rank_metrics(
        np.tile(np.asarray([0.0, 1.0, 2.0]), (2, 1)),
        truth,
        state_count=2,
        candidate_count=3,
    )

    assert result["complete_pool_state_count"] == 1
    assert result["mean_selected_success_by_25"] == 0.0
    assert result["mean_pool_has_success_by_25"] == 1.0
    assert result["mean_complete_pool_regret_at_25"] == 2.0
    assert result["by_state"][1]["valid_candidate_count_25"] == 2


def test_concurrent_training_status_persistence_does_not_deadlock(tmp_path, monkeypatch):
    from scripts import cvpr_table2

    monkeypatch.setattr(cvpr_table2, "OUTPUT_ROOT", tmp_path)
    wave_results = []
    barrier = threading.Barrier(3)
    results = [
        {"run_id": f"task/seed_3072/arm_{index}", "status": "complete"}
        for index in range(3)
    ]

    def record(result):
        barrier.wait()
        cvpr_table2._record_training_result(result, wave_results, 3072)

    threads = [threading.Thread(target=record, args=(result,), daemon=True) for result in results]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    saved = json.loads(
        (tmp_path / "summary/training_status.json").read_text(encoding="utf-8")
    )
    assert {item["run_id"] for item in saved["runs"]} == {
        item["run_id"] for item in results
    }
