import unittest
import json
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from stable_worldmodel.solver import CEMSolver

from source.diagnostics.stage_b_episode_failures import (
    CEMTraceCollector,
    TracedCEMSolver,
    TracedPolicy,
    build_prefix_replay_actions,
    build_shared_initial_panel,
    classify_failure,
    load_diagnostic_manifest,
    select_diagnostic_slots,
    select_grounding_panel,
    should_expand_grounding,
    write_trace_artifacts,
)


class QuadraticCost(torch.nn.Module):
    def get_cost(self, info, actions):
        target = info["target"]
        return (actions - target).square().mean(dim=(-1, -2))


class FakeEnv:
    num_envs = 3
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(3, 1))


def make_solver(seed=19):
    solver = CEMSolver(
        QuadraticCost(),
        batch_size=2,
        num_samples=8,
        n_steps=2,
        topk=3,
        seed=seed,
    )
    config = type(
        "Plan",
        (),
        {"horizon": 2, "action_block": 1},
    )()
    solver.configure(
        action_space=FakeEnv.action_space,
        n_envs=FakeEnv.num_envs,
        config=config,
    )
    return solver


class FakePlanningPolicy:
    def __init__(self, solver):
        self.solver = solver
        self.process = {}

    def set_env(self, env):
        self.env = env

    def get_action(self, info):
        alive = np.flatnonzero(~np.asarray(info["terminated"], dtype=bool))
        sliced = {
            key: value[alive]
            for key, value in info.items()
        }
        planned = self.solver(sliced)["actions"][:, 0].numpy()
        actions = np.full((self.env.num_envs, 1), np.nan, dtype=np.float32)
        actions[alive] = planned
        return actions


class StageBEpisodeTraceTests(unittest.TestCase):
    def test_trace_is_rng_and_action_equivalent_to_untraced_cem(self):
        original = make_solver()
        traced_inner = make_solver()
        collector = CEMTraceCollector(detailed_steps={0, 1})
        traced = TracedCEMSolver(traced_inner, collector)
        info = {"target": torch.tensor([[[0.2]], [[-0.4]], [[0.7]]])}
        traced_info = {
            **info,
            "_diagnostic_slot_id": torch.tensor([0, 1, 2]),
        }

        expected = original(info)
        actual = traced(traced_info)

        torch.testing.assert_close(actual["actions"], expected["actions"], rtol=0, atol=0)
        self.assertEqual(actual["costs"], expected["costs"])
        self.assertTrue(torch.equal(original.torch_gen.get_state(), traced_inner.torch_gen.get_state()))
        self.assertEqual(len(collector.summaries), 3 * 2)
        self.assertEqual(len(collector.details), 3 * 2)

    def test_slot_and_replan_context_survives_partial_termination(self):
        collector = CEMTraceCollector(detailed_steps={0})
        policy = TracedPolicy(
            FakePlanningPolicy(TracedCEMSolver(make_solver(), collector)),
            collector,
        )
        policy.set_env(FakeEnv())
        base = {
            "target": np.array([[[0.2]], [[-0.4]], [[0.7]]], dtype=np.float32),
        }

        policy.get_action({**base, "terminated": np.array([False, False, False])})
        policy.get_action({**base, "terminated": np.array([False, True, False])})

        contexts = {(row["slot"], row["replan"]) for row in collector.summaries}
        self.assertEqual(
            contexts,
            {(0, 0), (1, 0), (2, 0), (0, 1), (2, 1)},
        )
        self.assertEqual(len(collector.executed_raw[0]), 2)
        self.assertEqual(len(collector.executed_raw[1]), 1)
        self.assertEqual(len(collector.executed_raw[2]), 2)

    def test_column_shaped_termination_mask_records_actions_per_environment(self):
        collector = CEMTraceCollector(detailed_steps={0})
        policy = TracedPolicy(
            FakePlanningPolicy(TracedCEMSolver(make_solver(), collector)),
            collector,
        )
        policy.set_env(FakeEnv())

        policy.get_action(
            {
                "target": np.array(
                    [[[0.2]], [[-0.4]], [[0.7]]], dtype=np.float32
                ),
                "terminated": np.array([[False], [True], [False]]),
            }
        )

        self.assertEqual(sorted(collector.executed_raw), [0, 2])

    def test_grounding_panel_is_seeded_deduplicated_and_adaptive(self):
        candidates = torch.arange(12, dtype=torch.float32).reshape(6, 2, 1)
        kwargs = {
            "candidates": candidates,
            "costs": torch.tensor([3.0, 0.0, 4.0, 1.0, 5.0, 2.0]),
            "topk_indices": torch.tensor([1, 3]),
            "previous_mean": candidates[0],
            "best_ever": candidates[1],
            "anchors": {"expert": candidates[1], "actor": candidates[5]},
            "non_elite_count": 2,
            "seed": 42,
        }
        panel_a = select_grounding_panel(**kwargs)
        panel_b = select_grounding_panel(**kwargs)

        torch.testing.assert_close(panel_a.actions, panel_b.actions, rtol=0, atol=0)
        self.assertEqual(panel_a.sources, panel_b.sources)
        self.assertEqual(
            len(panel_a.actions),
            len({x.numpy().tobytes() for x in panel_a.actions}),
        )
        self.assertEqual(
            should_expand_grounding(
                panel_successes=[False] * len(panel_a.actions),
                final_selection_success=False,
                error_decomposition_clear=True,
            ),
            "no_success_in_panel",
        )
        self.assertIsNone(
            should_expand_grounding(
                panel_successes=[True, False],
                final_selection_success=False,
                error_decomposition_clear=True,
            )
        )

    def test_shared_initial_panel_contains_fixed_candidate_families(self):
        expert = torch.zeros(2, 3)
        actor = torch.ones(2, 3)

        first = build_shared_initial_panel(
            expert,
            actor,
            random_candidates=3,
            actor_neighbors=2,
            actor_noise_std=0.1,
            generator=torch.Generator().manual_seed(42),
        )
        second = build_shared_initial_panel(
            expert,
            actor,
            random_candidates=3,
            actor_neighbors=2,
            actor_noise_std=0.1,
            generator=torch.Generator().manual_seed(42),
        )

        torch.testing.assert_close(first.actions, second.actions, rtol=0, atol=0)
        self.assertEqual(first.sources, second.sources)
        self.assertEqual(
            first.sources[:6],
            (
                "expert",
                "zero",
                "time_reverse",
                "time_roll",
                "actor",
                "random_0",
            ),
        )
        self.assertEqual(len(first.actions), 10)

    def test_slot_selection_uses_failures_controls_and_e0_rescues(self):
        selected = select_diagnostic_slots(
            {
                "e0": [False, True, True, True, False, True],
                "e1_384": [True, False, True, True, True, True],
                "e3_384": [True, True, False, True, True, True],
            }
        )

        self.assertEqual(selected["failure_union"], [1, 2])
        self.assertEqual(selected["controls"], [3, 5])
        self.assertEqual(selected["e0_rescues"], [0, 4])
        self.assertEqual(selected["slots"], [0, 1, 2, 3, 4, 5])

    def test_prefix_replay_restores_the_same_second_replan_state(self):
        scaler = StandardScaler().fit(
            np.array([[-2.0], [0.0], [2.0]], dtype=np.float32)
        )
        candidates = torch.tensor(
            [[[0.0], [1.0]], [[-1.0], [0.5]]],
            dtype=torch.float32,
        )
        prefix = np.array([[0.5], [-0.25]], dtype=np.float32)

        replay = build_prefix_replay_actions(
            candidates,
            scaler=scaler,
            action_block=1,
            prefix_raw=prefix,
        )

        np.testing.assert_array_equal(replay[:, :2], np.repeat(prefix[None], 2, axis=0))
        initial_state = 3.0
        restored = initial_state + replay[:, :2, 0].sum(axis=1)
        direct = initial_state + prefix[:, 0].sum()
        np.testing.assert_allclose(restored, [direct, direct], rtol=0, atol=0)

    def test_synthetic_evidence_triggers_all_mechanism_labels(self):
        cases = [
            (
                {
                    "all_candidates_grounded": True,
                    "candidate_successes": [False, False],
                    "final_success": False,
                },
                "coverage_failure",
            ),
            (
                {
                    "all_candidates_grounded": True,
                    "candidate_successes": [False, True],
                    "final_success": False,
                },
                "ranking_failure",
            ),
            (
                {
                    "final_success": False,
                    "predicted_true_latent_spearman": -0.8,
                    "true_latent_physical_spearman": 0.9,
                },
                "dynamics_error",
            ),
            (
                {
                    "final_success": False,
                    "predicted_true_latent_spearman": 0.9,
                    "true_latent_physical_spearman": -0.8,
                },
                "latent_metric_error",
            ),
            (
                {
                    "final_success": False,
                    "predicted_elite_cost_0_5_29": [3.0, 2.0, 1.0],
                    "physical_cost_0_5_29": [1.0, 2.0, 3.0],
                    "earlier_best_physical_cost": 0.5,
                    "final_mean_physical_cost": 3.0,
                },
                "cem_exploitation",
            ),
            (
                {
                    "final_success": False,
                    "physical_distance_start_after_first_final": [4.0, 1.0, 3.0],
                },
                "replan_regression",
            ),
            (
                {
                    "final_success": False,
                    "failed_action_out_of_bounds_fraction": 0.4,
                    "reference_action_out_of_bounds_fraction": 0.0,
                    "failed_action_norm": 8.0,
                    "reference_action_norm": 2.0,
                },
                "action_ood",
            ),
        ]
        for evidence, expected in cases:
            with self.subTest(expected):
                self.assertIn(expected, classify_failure(evidence))

    def test_ranking_requires_failed_final_selection_and_ood_uses_or(self):
        self.assertNotIn(
            "ranking_failure",
            classify_failure(
                {
                    "final_success": False,
                    "final_selection_success": True,
                    "candidate_successes": [True, False],
                }
            ),
        )
        labels = classify_failure(
            {
                "final_success": False,
                "failed_action_out_of_bounds_fraction": 0.2,
                "reference_action_out_of_bounds_fraction": 0.0,
                "failed_action_norm": 2.0,
                "reference_action_norm": 2.0,
            }
        )
        self.assertIn("action_ood", labels)

    def test_manifest_fixes_the_exact_eight_run_matrix(self):
        manifest = load_diagnostic_manifest(
            "config/diagnostics/stage_b_episode_failures.yaml"
        )

        self.assertEqual(len(manifest.runs), 8)
        self.assertEqual(
            set(manifest.runs),
            {
                "pusht_e0",
                "pusht_e1_192",
                "pusht_e1_384",
                "pusht_e3_192",
                "pusht_e3_384",
                "reacher_e0",
                "reacher_e1_384",
                "reacher_e3_384",
            },
        )
        for run in manifest.runs.values():
            self.assertTrue(run.checkpoint.is_file(), run.checkpoint)
            self.assertTrue(run.reference_result.is_file(), run.reference_result)
            self.assertEqual(run.epoch, 10)

    def test_smoke_artifacts_have_complete_finite_schema(self):
        collector = CEMTraceCollector(detailed_steps={0, 1})
        collector.summaries = [
            {
                "slot": 0,
                "replan": 0,
                "iteration": step,
                "candidate_count": 8,
                "cost_min": 0.1,
                "cost_mean": 0.2,
                "elite_cost_mean": 0.15,
                "mean_norm": 0.3,
                "variance_mean": 0.4,
                "mean_shift": 0.5,
                "elite_spread": 0.6,
                "candidate_min": -1.0,
                "candidate_max": 1.0,
                "candidate_abs_mean": 0.2,
                "candidate_out_of_bounds_fraction": 0.0,
            }
            for step in range(2)
        ]
        collector.details = [
            {
                "slot": 0,
                "replan": 0,
                "iteration": step,
                "candidates": torch.zeros(8, 2, 1),
                "costs": torch.zeros(8),
                "topk_indices": torch.tensor([0, 1]),
                "elites": torch.zeros(2, 2, 1),
                "previous_mean": torch.zeros(2, 1),
                "previous_variance": torch.ones(2, 1),
                "updated_mean": torch.zeros(2, 1),
                "updated_variance": torch.ones(2, 1),
            }
            for step in range(2)
        ]
        collector.plans = [
            {
                "slot": 0,
                "replan": 0,
                "normalized_plan": torch.zeros(2, 1),
                "executed_prefix_before": np.empty((0, 1), dtype=np.float32),
            }
        ]
        collector.executed_raw[0].append(np.array([0.0], dtype=np.float32))
        collector.executed_normalized[0].append(
            np.array([0.0], dtype=np.float32)
        )

        with tempfile.TemporaryDirectory() as root:
            summary = write_trace_artifacts(
                Path(root),
                run_label="smoke",
                collector=collector,
                actual_successes=[False],
                expected_successes=[False],
                metadata={"num_samples": 8, "n_steps": 2, "topk": 2},
            )
            payload = json.loads(
                (Path(root) / "reproduction.json").read_text()
            )
            slot = json.loads(
                (Path(root) / "slots" / "slot_00.json").read_text()
            )

        self.assertEqual(summary["status"], "reproducible")
        self.assertEqual(payload["trace"]["summary_rows"], 2)
        self.assertEqual(payload["trace"]["detail_rows"], 2)
        self.assertEqual(slot["replans"][0]["iteration_count"], 2)
        self.assertEqual(slot["executed_raw_action_count"], 1)


if __name__ == "__main__":
    unittest.main()
