import unittest

import numpy as np
import torch

from source.common.round4_reacher_diagnosis import (
    action_statistics,
    run_cem_from_latents,
    sample_action_flow,
    terminal_metrics,
)


class _ToyFlow:
    action_horizon = 1
    action_dim = 1

    def sample_actions(self, z0, *, noise, num_steps, goal_latent):
        actions = noise.clone()
        for step in range(num_steps):
            timestep = z0.new_full((z0.shape[0],), step / num_steps)
            actions = actions + (actions + timestep[:, None, None]) / num_steps
        return actions

    def __call__(self, z0, actions, timestep, *, mode, goal_latent):
        self.last_mode = mode
        return {"action_velocity": actions + timestep[:, None, None]}


class _ToyCost:
    def get_cost_from_latents(self, z0, goal_latent, candidates):
        return (candidates - goal_latent[:, None, None, :1]).square().mean(dim=(-1, -2))


class Round4ReacherDiagnosisTests(unittest.TestCase):
    def test_terminal_metrics_use_strict_per_joint_threshold(self):
        metrics = terminal_metrics(
            np.array([0.05, 0.0]),
            np.zeros(2),
            actions=np.array([[0.2, -0.3], [1.2, 0.0]]),
        )
        self.assertFalse(metrics["success"])
        self.assertAlmostEqual(metrics["max_joint_error"], 0.05)
        self.assertAlmostEqual(metrics["terminal_margin"], 0.0)
        self.assertEqual(metrics["action_out_of_range_fraction"], 0.25)

    def test_heun_uses_the_same_initial_noise_and_expected_one_step_update(self):
        model = _ToyFlow()
        z0 = torch.zeros(1, 1)
        goal = torch.zeros(1, 1)
        noise = torch.zeros(1, 1, 1)

        euler = sample_action_flow(
            model, z0, goal, noise, steps=1, integrator="euler"
        )
        heun = sample_action_flow(
            model, z0, goal, noise, steps=1, integrator="heun"
        )

        torch.testing.assert_close(euler, torch.zeros_like(euler))
        torch.testing.assert_close(heun, torch.full_like(heun, 0.5))
        self.assertEqual(model.last_mode, "stage_a")

    def test_cem_replay_is_deterministic_and_respects_batch_one_semantics(self):
        model = _ToyCost()
        z0 = torch.zeros(2, 1)
        goal = torch.tensor([[1.0], [2.0]])
        draws = torch.zeros(2, 2, 3, 1, 1)
        draws[:, :, 1] = torch.tensor([[[0.5]], [[-0.5]]])
        init = torch.zeros(2, 1, 1)

        first = run_cem_from_latents(
            model,
            z0,
            goal,
            init_actions=init,
            random_draws=draws,
            topk=2,
            solver_batch_size=1,
        )
        second = run_cem_from_latents(
            model,
            z0,
            goal,
            init_actions=init,
            random_draws=draws,
            topk=2,
            solver_batch_size=1,
        )
        torch.testing.assert_close(first["actions"], second["actions"])
        self.assertEqual(first["solver_batch_size"], 1)
        self.assertEqual(len(first["initial_cost"]), 2)

    def test_action_statistics_expose_normalized_out_of_range_fraction(self):
        stats = action_statistics(
            torch.tensor([[[0.0, 1.5], [-2.0, 0.5]]], dtype=torch.float32)
        )
        self.assertAlmostEqual(stats["action_out_of_range_fraction"], 0.5)
        self.assertAlmostEqual(stats["action_max_abs"], 2.0)


if __name__ == "__main__":
    unittest.main()
