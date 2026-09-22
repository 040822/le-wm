import tempfile
import unittest
from pathlib import Path

import numpy as np
import gymnasium as gym
from sklearn.preprocessing import StandardScaler

from scripts.round5_phase1_5_diagnostics import FixedCandidatePolicy

from source.common.round5_phase1_5 import (
    PHASE15_BOOTSTRAP_SAMPLES,
    adaptive_stability_specs,
    capture_environment_state,
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
    candidate_pool_metrics,
    paired_guidance_metrics,
    phase15_scan_slot,
    primary_condition_specs,
    safe_correlation,
    sampling_stability_specs,
    synchronous_timing,
    restore_environment_state,
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
    def test_fixed_candidate_policy_replays_then_holds_zero(self):
        class FakeEnv:
            num_envs = 2
            action_space = gym.spaces.Box(-1.0, 1.0, shape=(2, 2))

        processor = StandardScaler().fit(
            np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        )
        policy = FixedCandidatePolicy(
            np.asarray(
                [
                    [[-0.5, 0.5], [0.25, -0.25]],
                    [[0.1, 0.2], [0.3, 0.4]],
                ]
            ),
            process={"action": processor},
            action_block=1,
        )
        policy.set_env(FakeEnv())
        info = {"terminated": np.array([False, False]), "truncated": np.array([False, False])}
        first = policy.get_action(info)
        second = policy.get_action(info)
        third = policy.get_action(info)
        self.assertEqual(first.shape, (2, 2))
        np.testing.assert_allclose(first, processor.inverse_transform([[-0.5, 0.5], [0.1, 0.2]]))
        np.testing.assert_allclose(second, processor.inverse_transform([[0.25, -0.25], [0.3, 0.4]]))
        np.testing.assert_array_equal(third, np.zeros((2, 2), dtype=np.float32))

    def test_environment_state_snapshot_round_trips(self):
        class FakeEnv:
            def __init__(self):
                self.value = np.array([1.0, 2.0])

            def get_state(self):
                return self.value.copy()

            def set_state(self, state):
                self.value = np.asarray(state, dtype=np.float64).copy()

        env = FakeEnv()
        snapshot = capture_environment_state(env)
        env.value[:] = 9.0
        restore_environment_state(env, snapshot)
        np.testing.assert_array_equal(env.value, [1.0, 2.0])

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

    def test_controls_keep_physical_and_normalized_zero_distinct(self):
        anchors = np.zeros((1, 5, 2), dtype=np.float64)
        physical_zero = np.full((5, 2), 0.25, dtype=np.float64)
        actions, metadata = make_control_actions(
            anchors,
            seed=7,
            physical_zero=physical_zero,
        )
        physical_indices = [index for index, item in enumerate(metadata) if item["kind"] == "physical_zero"]
        normalized_indices = [index for index, item in enumerate(metadata) if item["kind"] == "normalized_zero"]
        self.assertEqual(len(physical_indices), 1)
        self.assertEqual(len(normalized_indices), 1)
        np.testing.assert_array_equal(actions[physical_indices[0]], physical_zero)
        np.testing.assert_array_equal(actions[normalized_indices[0]], np.zeros_like(physical_zero))

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

    def test_candidate_pool_requires_complete_state_by_flow_grid(self):
        records = []
        for flow_steps in (1, 2):
            for state_id in ("a", "b"):
                for candidate_index in range(3):
                    records.append(
                        {
                            "state_id": state_id,
                            "flow_steps": flow_steps,
                            "candidate_index": candidate_index,
                            "predicted_cost": float(candidate_index),
                            "true_distance": float(3 - candidate_index),
                            "success": candidate_index == 2,
                            "action": [[float(candidate_index)]],
                        }
                    )
        result = candidate_pool_metrics(
            records,
            task="tworoom",
            flow_steps=(1, 2),
            candidates_per_flow_step=3,
            bootstrap_samples=8,
        )
        self.assertEqual(result["states"], 2)
        self.assertEqual(result["candidate_rows"], 12)
        self.assertEqual(set(result["by_flow_steps"]), {"1", "2"})
        self.assertEqual(result["by_flow_steps"]["1"]["candidate_rows"], 6)

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

    def test_paired_guidance_compares_same_displacement_random_control(self):
        result = paired_guidance_metrics(
            [
                {
                    "state_id": "a",
                    "guided_predicted_cost_before": 2.0,
                    "guided_predicted_cost_after": 1.0,
                    "guided_true_cost_before": 2.0,
                    "guided_true_cost_after": 1.0,
                    "random_true_cost_before": 2.0,
                    "random_true_cost_after": 1.5,
                    "guided_action_rms_displacement": 0.2,
                    "random_action_rms_displacement": 0.2,
                }
            ],
            bootstrap_samples=8,
        )
        self.assertAlmostEqual(result["paired_advantage_mean"], 0.5)
        self.assertEqual(result["guided_true_improvement_fraction"], 1.0)

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
