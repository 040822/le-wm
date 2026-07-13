import unittest
from types import SimpleNamespace

import torch
from torch import nn

from source.model.fast_lewam.jepa import FastLeWAM
from source.model.fast_lewam.modules import causal_attention_mask


class TinyEncoder(nn.Module):
    """Small encoder adapter used to exercise the public FastLeWAM interface."""

    def __init__(self, latent_dim):
        super().__init__()
        self.proj = nn.Linear(3, latent_dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        pooled = pixels.mean(dim=(-2, -1))
        cls = self.proj(pooled)
        return SimpleNamespace(last_hidden_state=cls[:, None])


def make_model(action_horizon=3):
    torch.manual_seed(7)
    return FastLeWAM(
        encoder=TinyEncoder(latent_dim=8),
        projector=nn.Identity(),
        latent_dim=8,
        action_dim=4,
        action_horizon=action_horizon,
        latent_head_dim=12,
        latent_head_layers=2,
        heads=3,
        mlp_dim=24,
        dropout=0.0,
    )


class FastLeWAMModelTests(unittest.TestCase):
    def test_modes_return_action_and_latent_chunks_with_expected_shapes(self):
        model = make_model()
        z0 = torch.randn(2, 8)
        actions = torch.randn(2, 3, 4)
        timestep = torch.tensor([0.2, 0.8])

        stage_a = model(z0, actions, timestep, mode="stage_a")
        stage_b = model(z0, actions, torch.ones(2), mode="stage_b")
        stage_ab = model(z0, actions, timestep, mode="stage_ab")
        stage_c = model(z0, actions, timestep, mode="stage_c")

        self.assertEqual(stage_a["action_velocity"].shape, (2, 3, 4))
        self.assertEqual(stage_b["predicted_latents"].shape, (2, 3, 8))
        self.assertEqual(stage_ab["clean_action"].shape, (2, 3, 4))
        self.assertEqual(stage_ab["predicted_latents"].shape, (2, 3, 8))
        self.assertEqual(stage_c["action_velocity"].shape, (2, 3, 4))
        self.assertEqual(stage_c["predicted_latents"].shape, (2, 3, 8))

    def test_causal_mask_and_predictions_hide_future_actions(self):
        mask = causal_attention_mask(action_horizon=3)
        self.assertEqual(mask.shape, (7, 7))
        self.assertTrue(torch.equal(mask, torch.ones(7, 7, dtype=torch.bool).tril()))

        model = make_model().eval()
        z0 = torch.randn(1, 8)
        clean_actions = torch.randn(1, 3, 4)
        changed = clean_actions.clone()
        changed[:, 2] += 100.0

        before = model(z0, clean_actions, torch.ones(1), mode="stage_b")
        after = model(z0, changed, torch.ones(1), mode="stage_b")

        self.assertTrue(
            torch.equal(
                before["predicted_latents"][:, :2],
                after["predicted_latents"][:, :2],
            )
        )

    def test_latent_loss_reaches_stage_a_action_head_without_detach(self):
        model = make_model()
        z0 = torch.randn(2, 8)
        noisy_actions = torch.randn(2, 3, 4)
        timestep = torch.tensor([0.3, 0.7])

        output = model(
            z0,
            noisy_actions,
            timestep,
            mode="stage_ab",
            detach_clean_action=False,
        )
        output["predicted_latents"].square().mean().backward()

        grad = model.action_head.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0.0)

    def test_stage_b_call_does_not_change_stage_a_output(self):
        model = make_model().eval()
        z0 = torch.randn(2, 8)
        noisy_actions = torch.randn(2, 3, 4)
        timestep = torch.tensor([0.25, 0.75])

        action_only = model(z0, noisy_actions, timestep, mode="stage_a")
        with_latents = model(z0, noisy_actions, timestep, mode="stage_ab")

        self.assertTrue(
            torch.equal(
                action_only["action_velocity"],
                with_latents["action_velocity"],
            )
        )


if __name__ == "__main__":
    unittest.main()
