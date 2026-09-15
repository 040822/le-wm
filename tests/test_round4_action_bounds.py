import unittest

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler

from source.common.round4_action_bounds import (
    NormalizedActionBounds,
    compute_normalized_action_bounds,
    normalized_action_stats,
    project_normalized_actions,
)
from source.policy.fast_lewam_eval import ProjectedCEMSolver
from scripts.round4_reacher_action_bounds import _projection_stats


class Round4ActionBoundsTests(unittest.TestCase):
    def test_normalized_bounds_round_trip_through_existing_scaler(self):
        raw_actions = np.array(
            [
                [-0.8, -0.2],
                [-0.1, 0.4],
                [0.5, 0.9],
            ],
            dtype=np.float64,
        )
        scaler = StandardScaler().fit(raw_actions)
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        bounds = compute_normalized_action_bounds(
            action_space,
            scaler,
            action_block=2,
        )

        np.testing.assert_allclose(
            scaler.inverse_transform(bounds.normalized_low[:2][None, :])[0],
            action_space.low,
        )
        np.testing.assert_allclose(
            scaler.inverse_transform(bounds.normalized_high[:2][None, :])[0],
            action_space.high,
        )
        expanded_low, expanded_high = bounds.for_action_dim(4)
        self.assertEqual(expanded_low.shape, (4,))
        self.assertEqual(expanded_high.shape, (4,))
        np.testing.assert_allclose(
            expanded_low,
            np.tile(bounds.normalized_low, 2),
        )

    def test_clip_projects_each_action_coordinate_without_mutating_input(self):
        bounds = NormalizedActionBounds(
            physical_low=np.array([-1.0, -1.0]),
            physical_high=np.array([1.0, 1.0]),
            normalized_low=np.array([-2.0, -1.0]),
            normalized_high=np.array([2.0, 1.0]),
        )
        actions = np.array([[[3.0, -3.0], [1.5, 0.5]]])

        projected = project_normalized_actions(actions, bounds, mode="clip")

        np.testing.assert_allclose(
            projected,
            np.array([[[2.0, -1.0], [1.5, 0.5]]]),
        )
        np.testing.assert_allclose(actions, np.array([[[3.0, -3.0], [1.5, 0.5]]]))

    def test_global_scale_uses_one_factor_for_each_path(self):
        bounds = NormalizedActionBounds(
            physical_low=np.array([-1.0, -1.0]),
            physical_high=np.array([1.0, 1.0]),
            normalized_low=np.array([-2.0, -1.0]),
            normalized_high=np.array([2.0, 1.0]),
        )
        actions = np.array([[[4.0, -2.0], [2.0, 0.0]]])

        projected = project_normalized_actions(actions, bounds, mode="global_scale")

        np.testing.assert_allclose(
            projected,
            np.array([[[2.0, -1.0], [1.0, 0.0]]]),
        )

    def test_global_scale_handles_batched_single_actions(self):
        bounds = NormalizedActionBounds(
            physical_low=np.array([-1.0, -1.0]),
            physical_high=np.array([1.0, 1.0]),
            normalized_low=np.array([-2.0, -1.0]),
            normalized_high=np.array([2.0, 1.0]),
        )
        actions = np.array([[4.0, -2.0], [1.0, 0.5]])

        projected = project_normalized_actions(actions, bounds, mode="global_scale")

        self.assertEqual(projected.shape, actions.shape)
        np.testing.assert_allclose(projected, np.array([[2.0, -1.0], [1.0, 0.5]]))

    def test_torch_projection_preserves_device_dtype_and_shape(self):
        bounds = NormalizedActionBounds(
            physical_low=np.array([-1.0, -1.0]),
            physical_high=np.array([1.0, 1.0]),
            normalized_low=np.array([-2.0, -1.0]),
            normalized_high=np.array([2.0, 1.0]),
        )
        import torch

        actions = torch.tensor([[[3.0, -3.0], [1.5, 0.5]]], dtype=torch.float32)
        projected = project_normalized_actions(actions, bounds, mode="clip")

        self.assertIsInstance(projected, torch.Tensor)
        self.assertEqual(projected.dtype, actions.dtype)
        self.assertEqual(projected.shape, actions.shape)
        torch.testing.assert_close(
            projected,
            torch.tensor([[[2.0, -1.0], [1.5, 0.5]]]),
        )

    def test_stats_distinguish_legacy_unit_proxy_from_true_bounds(self):
        bounds = NormalizedActionBounds(
            physical_low=np.array([-1.0, -1.0]),
            physical_high=np.array([1.0, 1.0]),
            normalized_low=np.array([-2.0, -2.0]),
            normalized_high=np.array([2.0, 2.0]),
        )
        stats = normalized_action_stats(
            np.array([[[1.5, 0.0], [2.5, -2.5]]]),
            bounds,
        )

        self.assertAlmostEqual(stats["legacy_unit_threshold_fraction"], 0.75)
        self.assertAlmostEqual(
            stats["true_normalized_bound_violation_fraction"],
            0.5,
        )

    def test_projected_cem_scores_and_returns_only_bounded_candidates(self):
        class CostModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.max_seen = 0.0

            def get_cost(self, info, candidates):
                del info
                self.max_seen = max(self.max_seen, float(candidates.abs().max()))
                return candidates.square().mean(dim=(-1, -2))

        model = CostModel()
        base_solver = __import__(
            "stable_worldmodel.solver",
            fromlist=["CEMSolver"],
        ).CEMSolver(
            model=model,
            batch_size=1,
            num_samples=4,
            var_scale=1.0,
            n_steps=2,
            topk=2,
            device="cpu",
            seed=7,
        )
        config = OmegaConf.create(
            {"horizon": 2, "receding_horizon": 2, "action_block": 1}
        )
        base_solver.configure(
            action_space=gym.spaces.Box(-1.0, 1.0, shape=(1, 2)),
            n_envs=1,
            config=config,
        )
        solver = ProjectedCEMSolver(base_solver)
        bounds = NormalizedActionBounds(
            physical_low=np.array([-1.0, -1.0]),
            physical_high=np.array([1.0, 1.0]),
            normalized_low=np.array([-0.25, -0.5]),
            normalized_high=np.array([0.25, 0.5]),
        )
        solver.set_action_bounds(bounds)

        result = solver.solve({"state": torch.zeros(1, 1)})

        self.assertLessEqual(model.max_seen, 0.5 + 1e-6)
        self.assertLessEqual(float(result["actions"].abs().max()), 0.5 + 1e-6)
        projection = result["action_bound_projection"]
        self.assertEqual(projection["mode"], "clip")
        self.assertEqual(projection["projected_candidate_violation_fraction"], 0.0)
        self.assertGreater(projection["raw_candidate_violation_fraction"], 0.0)

    def test_report_stats_separate_warm_start_and_cem_candidate_metrics(self):
        payload = {
            "round4_planning": {
                "action_bound_projection": [
                    {
                        "warm_start": {
                            "before": {
                                "legacy_unit_threshold_fraction": 0.4,
                                "true_normalized_bound_violation_fraction": 0.2,
                            },
                            "after": {
                                "legacy_unit_threshold_fraction": 0.3,
                                "true_normalized_bound_violation_fraction": 0.0,
                            },
                            "changed_fraction": 0.1,
                            "mean_abs_delta": 0.02,
                            "max_abs_delta": 0.3,
                        },
                        "raw_candidate_violation_fraction": 0.4,
                        "projected_candidate_violation_fraction": 0.0,
                        "candidate_changed_fraction": 0.4,
                    }
                ]
            }
        }

        stats = _projection_stats(payload)

        self.assertAlmostEqual(stats["true_normalized_bound_before_fraction"], 0.2)
        self.assertAlmostEqual(stats["true_normalized_bound_after_fraction"], 0.0)
        self.assertAlmostEqual(stats["raw_candidate_violation_fraction"], 0.4)
        self.assertAlmostEqual(stats["projected_candidate_violation_fraction"], 0.0)
        self.assertAlmostEqual(stats["candidate_changed_fraction"], 0.4)


if __name__ == "__main__":
    unittest.main()
