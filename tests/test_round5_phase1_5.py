import tempfile
import unittest
from pathlib import Path

import numpy as np

from source.common.round5_phase1_5 import (
    PHASE15_BOOTSTRAP_SAMPLES,
    adaptive_stability_specs,
    candidate_selection_metrics,
    cluster_bootstrap,
    condition_id,
    condition_identity,
    deduplicate_actions,
    grid_counts,
    guidance_effect_metrics,
    make_control_actions,
    make_probe_split,
    normalized_physical_distance,
    phase15_scan_slot,
    primary_condition_specs,
    safe_correlation,
    sampling_stability_specs,
    synchronous_timing,
)


class Round5Phase15GridTests(unittest.TestCase):
    def test_primary_grid_matches_plan(self):
        specs = primary_condition_specs()
        self.assertEqual(len(specs), 2424)
        self.assertEqual(
            grid_counts(specs),
            {"A1": 144, "A2": 432, "A3": 864, "A4": 432, "A5": 96, "A6": 240, "A7": 216},
        )
        self.assertEqual(len(sampling_stability_specs()), 96)

    def test_a1_and_a7_remove_the_declared_aliases(self):
        specs = primary_condition_specs()
        a1 = [item for item in specs if item["group"] == "A1"]
        self.assertEqual(sum(item["mode"] == "P0" for item in a1), 24)
        self.assertFalse(
            any(
                item["group"] == "A7"
                and item["guidance"] == "guided_flow"
                and item["flow_steps"] == 1
                for item in specs
            )
        )


class Round5Phase15DiagnosticsTests(unittest.TestCase):
    def test_constant_correlation_is_undefined(self):
        self.assertIsNone(safe_correlation([1, 1, 1], [1, 2, 3]))

    def test_controls_are_deterministic_and_deduplicatable(self):
        anchors = np.zeros((3, 25, 2), dtype=np.float64)
        first, first_meta = make_control_actions(anchors, seed=7)
        second, second_meta = make_control_actions(anchors, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_meta, second_meta)
        unique, duplicates = deduplicate_actions(first)
        self.assertEqual(len(unique) + len(duplicates), len(first))
        self.assertGreater(len(duplicates), 0)

    def test_task_normalized_distance_uses_plan_thresholds(self):
        self.assertAlmostEqual(normalized_physical_distance("cube", [0.04, 0], [0, 0]), 1.0)
        self.assertAlmostEqual(normalized_physical_distance("reacher", [0.05, 0], [0, 0]), 1.0)
        self.assertAlmostEqual(normalized_physical_distance("tworoom", [16, 0], [0, 0]), 1.0)
        self.assertAlmostEqual(
            normalized_physical_distance("pusht", [20, 0, 0, 0, 0], [0, 0, 0, 0, 0]),
            1.0,
        )

    def test_selection_reports_oracle_and_selection_regret(self):
        records = [
            {"state_id": "a", "predicted_cost": 0.2, "true_distance": 2.0, "physical_state": [2, 0], "goal_state": [0, 0]},
            {"state_id": "a", "predicted_cost": 0.1, "true_distance": 1.0, "physical_state": [1, 0], "goal_state": [0, 0]},
            {"state_id": "b", "predicted_cost": 0.1, "true_distance": 3.0, "physical_state": [3, 0], "goal_state": [0, 0]},
            {"state_id": "b", "predicted_cost": 0.2, "true_distance": 1.0, "physical_state": [1, 0], "goal_state": [0, 0]},
        ]
        result = candidate_selection_metrics(records, task="tworoom", random_draws=8)
        self.assertEqual(result["states"], 2)
        self.assertAlmostEqual(result["selection_regret"], 1.0)
        self.assertEqual(result["predicted_true_distance_correlation"], 0.0)

    def test_guidance_metrics_marks_model_exploitation(self):
        result = guidance_effect_metrics(
            [
                {
                    "predicted_cost_before": 2,
                    "predicted_cost_after": 1,
                    "true_cost_before": 1,
                    "true_cost_after": 2,
                    "action_saturation_fraction": 0.25,
                }
            ]
        )
        self.assertEqual(result["model_exploitation_fraction"], 1.0)
        self.assertEqual(result["true_degradation_fraction"], 1.0)

    def test_bootstrap_samples_states(self):
        rows = [{"state_id": index, "value": float(index)} for index in range(4)]
        result = cluster_bootstrap(rows, "value", samples=32, seed=3)
        self.assertEqual(result["cluster_count"], 4)
        self.assertEqual(result["samples"], 32)

    def test_probe_split_is_trajectory_disjoint(self):
        split = make_probe_split(range(10), eval_trajectory_ids=(99,), seed=2026)
        self.assertTrue(set(split["train"]).isdisjoint(split["validation"]))
        self.assertTrue(set(split["train"]).isdisjoint(split["excluded_eval"]))
        self.assertTrue(set(split["validation"]).isdisjoint(split["excluded_eval"]))

    def test_synchronous_timing_runs_requested_calls(self):
        calls = []
        result = synchronous_timing(lambda: calls.append(1), warmup=2, runs=3)
        self.assertEqual(len(calls), 5)
        self.assertEqual(result["runs"], 3)
        self.assertEqual(len(result["samples_seconds"]), 3)

    def test_scan_slots_bound_concurrent_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            with phase15_scan_slot(directory, max_slots=1) as slot:
                self.assertEqual(slot, 0)
                with self.assertRaises(RuntimeError):
                    with phase15_scan_slot(directory, max_slots=1):
                        pass
            self.assertFalse(
                (Path(directory) / "locks" / "scan_slots" / "slot_0.lock").exists()
            )

    def test_adaptive_stability_picks_two_per_category(self):
        rows = []
        categories = [("P3", "none"), ("P0", "post_opt"), ("P0", "guided_flow"), ("P3", "post_opt"), ("P1", "none")]
        all_specs = primary_condition_specs()
        for task in ("cube", "pusht", "reacher", "tworoom"):
            for index, (mode, guidance) in enumerate(categories):
                spec = next(
                    item for item in all_specs
                    if item["task"] == task and item["mode"] == mode and item["guidance"] == guidance
                )
                rows.append({
                    "task": task,
                    "status": "completed",
                    "spec": spec,
                    "condition_id": f"{task}-{index}",
                    "success_rate": 1.0 - index * 0.1,
                    "forward_count": index,
                })
        result = adaptive_stability_specs(rows)
        self.assertEqual(len(result), 40)
        self.assertEqual({item["evaluation_seed"] for item in result}, {43, 44})


if __name__ == "__main__":
    unittest.main()
