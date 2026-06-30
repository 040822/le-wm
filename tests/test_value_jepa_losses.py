import unittest

import torch

from source.model.value_jepa.losses import (
    expectile_loss,
    squared_latent_distance,
    value_td_loss,
)


class ValueJEPALossTests(unittest.TestCase):
    def test_expectile_loss_weights_positive_and_negative_td_errors(self):
        td_error = torch.tensor([-2.0, 3.0])

        loss = expectile_loss(td_error, tau=0.8)

        expected = torch.tensor((0.2 * 4.0 + 0.8 * 9.0) / 2.0)
        self.assertTrue(torch.allclose(loss, expected))

    def test_value_td_loss_stops_gradient_through_following_target(self):
        current = torch.tensor([[0.0, 0.0]], requires_grad=True)
        following = torch.tensor([[1.0, 0.0]], requires_grad=True)
        goal = torch.tensor([[2.0, 0.0]], requires_grad=True)
        reward = torch.tensor([-1.0])

        result = value_td_loss(
            current=current,
            following=following,
            goal=goal,
            reward=reward,
            gamma=0.9,
            tau=0.8,
        )
        result.loss.backward()

        self.assertIsNotNone(current.grad)
        self.assertIsNone(following.grad)
        self.assertIsNotNone(goal.grad)
        self.assertTrue(result.loss.isfinite())

    def test_squared_latent_distance_preserves_batch_shape(self):
        x = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        y = torch.tensor([[[1.0, 1.0], [2.0, 1.0]]])

        distance = squared_latent_distance(x, y)

        self.assertEqual(distance.shape, (1, 2))
        self.assertTrue(torch.allclose(distance, torch.tensor([[1.0, 0.5]])))


if __name__ == "__main__":
    unittest.main()
