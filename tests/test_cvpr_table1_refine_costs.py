import unittest

import torch

from source.model.fast_lewam.jepa import FastLeWAM
from source.common.cvpr_table1_refine_train import (
    _prepare_trainable_parameters,
    _trajectory_losses,
)
from tests.test_round4_model import make_round4_model


class CoWMGoalDistanceReductionTests(unittest.TestCase):
    def setUp(self):
        self.predicted = torch.tensor([[[3.0], [1.0], [4.0]]], requires_grad=True)
        self.goal = torch.zeros(1, 1)

    def test_endpoint_preserves_terminal_distance(self):
        cost = FastLeWAM.reduce_goal_distance_costs(
            self.predicted, self.goal
        )
        self.assertTrue(torch.equal(cost, torch.tensor([16.0])))

    def test_minimum_uses_predicted_future_blocks_only(self):
        cost = FastLeWAM.reduce_goal_distance_costs(
            self.predicted, self.goal, score_reduction="minimum"
        )
        self.assertTrue(torch.equal(cost, torch.tensor([1.0])))

    def test_mixed_uses_terminal_weight_and_has_finite_gradient(self):
        cost = FastLeWAM.reduce_goal_distance_costs(
            self.predicted,
            self.goal,
            score_reduction="mixed",
            terminal_weight=0.25,
        )
        self.assertTrue(torch.allclose(cost, torch.tensor([4.75])))
        cost.sum().backward()
        self.assertTrue(torch.isfinite(self.predicted.grad).all())

    def test_invalid_mode_and_weight_are_rejected(self):
        with self.assertRaises(ValueError):
            FastLeWAM.reduce_goal_distance_costs(
                self.predicted, self.goal, score_reduction="softmin"
            )
        with self.assertRaises(ValueError):
            FastLeWAM.reduce_goal_distance_costs(
                self.predicted,
                self.goal,
                score_reduction="mixed",
                terminal_weight=1.1,
            )

    def test_temporal_loss_is_zero_when_every_target_distance_ties(self):
        predicted = torch.zeros(2, 5, 8, requires_grad=True)
        target = torch.zeros(2, 5, 8)
        goals = torch.zeros(2, 8)
        latent, cost, temporal = _trajectory_losses(predicted, target, goals, 1.0, 0.05)
        self.assertEqual(float(temporal), 0.0)
        self.assertTrue(torch.isfinite(latent + cost + temporal))

    def test_b_specific_arm_only_unfreezes_stage_b_parameters(self):
        model = make_round4_model()
        trainable, frozen, rows = _prepare_trainable_parameters(model, "b_specific")
        self.assertEqual(set(rows), {1})
        self.assertEqual(
            {name for name, _ in trainable},
            {
                "latent_head.weight",
                "latent_head.bias",
                "query_tokens",
                "joint_positions",
                "mode_embedding.weight",
            },
        )
        self.assertTrue(any(name.startswith("predictor.") for name in frozen))


if __name__ == "__main__":
    unittest.main()
