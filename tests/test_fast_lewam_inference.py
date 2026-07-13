import unittest

import torch

from tests.test_fast_lewam_model import make_model


class FastLeWAMInferenceTests(unittest.TestCase):
    def test_stage_a_and_stage_c_sample_complete_chunks(self):
        model = make_model().eval()
        z0 = torch.randn(2, 8)
        noise = torch.randn(2, 3, 4)

        stage_a = model.sample_actions(z0, noise=noise, num_steps=3)
        stage_c = model.sample_joint(z0, noise=noise, num_steps=3)

        self.assertEqual(stage_a.shape, (2, 3, 4))
        self.assertEqual(stage_c["actions"].shape, (2, 3, 4))
        self.assertEqual(stage_c["predicted_latents"].shape, (2, 3, 8))

    def test_stage_b_cost_is_parallel_and_differentiable_for_solver(self):
        model = make_model().eval()
        batch, samples = 2, 4
        info = {
            "pixels": torch.randn(batch, samples, 2, 3, 8, 8),
            "goal": torch.randn(batch, samples, 2, 3, 8, 8),
        }
        candidates = torch.randn(batch, samples, 3, 4, requires_grad=True)

        cost = model.get_cost(info, candidates)
        cost.sum().backward()

        self.assertEqual(cost.shape, (batch, samples))
        self.assertIsNotNone(candidates.grad)
        self.assertGreater(candidates.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
