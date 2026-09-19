import unittest
from types import SimpleNamespace

import torch
from torch import nn

from source.model.fast_lewam.round4 import (
    MODE_IDS,
    Round4FastLeWAM,
    d_attention_mask,
    e_attention_mask,
)


class TinyEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.proj = nn.Linear(3, latent_dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        pooled = pixels.mean(dim=(-2, -1))
        return SimpleNamespace(last_hidden_state=self.proj(pooled)[:, None])


def make_round4_model(**kwargs):
    torch.manual_seed(13)
    config = dict(
        encoder=TinyEncoder(8),
        projector=nn.Identity(),
        latent_dim=8,
        action_dim=4,
        action_horizon=5,
        latent_head_dim=16,
        latent_head_layers=2,
        heads=4,
        mlp_dim=32,
        stage_a_goal_injection="token",
        activation_checkpointing=False,
    )
    config.update(kwargs)
    return Round4FastLeWAM(**config)


class Round4ModelTests(unittest.TestCase):
    def test_mode_ids_are_stable_and_masks_match_token_contract(self):
        self.assertEqual(MODE_IDS, {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4})

        d_mask = d_attention_mask(5)
        self.assertEqual(d_mask.shape, (6, 6))
        self.assertTrue(torch.equal(d_mask[:2, :2], torch.ones(2, 2, dtype=torch.bool)))
        self.assertFalse(d_mask[:2, 2:].any())
        self.assertTrue(d_mask[2:, :].all())

        e_mask = e_attention_mask(5)
        self.assertEqual(e_mask.shape, (11, 11))
        self.assertTrue(e_mask[:6, :6].all())
        self.assertFalse(e_mask[:6, 6:].any())
        self.assertTrue(e_mask[6:, :].all())

    def test_d_keeps_endpoints_and_noise_changes_interior(self):
        model = make_round4_model().eval()
        z0 = torch.randn(2, 8)
        zg = torch.randn(2, 8)

        first = model.sample_latent_paths(
            z0, zg, num_samples=3, num_steps=2,
            generator=torch.Generator().manual_seed(1),
        )
        second = model.sample_latent_paths(
            z0, zg, num_samples=3, num_steps=2,
            generator=torch.Generator().manual_seed(2),
        )

        self.assertEqual(first.shape, (2, 3, 6, 8))
        self.assertTrue(torch.equal(first[:, :, 0], z0[:, None].expand_as(first[:, :, 0])))
        self.assertTrue(torch.equal(first[:, :, -1], zg[:, None].expand_as(first[:, :, -1])))
        self.assertFalse(torch.equal(first[:, :, 1:-1], second[:, :, 1:-1]))

    def test_e_maps_h_plus_one_states_to_h_action_blocks_without_timestep(self):
        model = make_round4_model().eval()
        path = torch.randn(2, 3, 6, 8)
        actions = model.decode_latent_paths(path)
        self.assertEqual(actions.shape, (2, 3, 5, 4))

        direct = model.predict_inverse_dynamics(path[:, 0])
        self.assertEqual(direct.shape, (2, 5, 4))

    def test_abde_branches_share_dit_and_preserve_gradient_boundaries(self):
        model = make_round4_model()
        pixels = torch.randn(2, 6, 3, 8, 8)
        actions = torch.randn(2, 5, 4)
        embeddings = model.encode_pixels(pixels)
        z0 = embeddings[:, 0]
        zg = embeddings[:, -1]
        interior = embeddings[:, 1:-1]
        noise = torch.randn_like(interior)
        t = torch.full((2,), 0.5)
        noisy = (1.0 - t[:, None, None]) * noise + t[:, None, None] * interior.detach()

        loss = model(
            z0, actions, t, mode="stage_a", goal_latent=zg
        )["action_velocity"].square().mean()
        loss = loss + model.predict_latent_velocity(z0, zg, noisy, t).square().mean()
        loss = loss + model.predict_inverse_dynamics(embeddings).square().mean()
        loss.backward()

        self.assertEqual(model.mode_embedding.num_embeddings, 5)
        self.assertIsNotNone(model.predictor.layers[0].attention.to_qkv.weight.grad)
        self.assertGreater(model.predictor.layers[0].attention.to_qkv.weight.grad.abs().sum().item(), 0.0)
        self.assertTrue(interior.requires_grad)

        interior_leaf = interior.detach().requires_grad_(True)
        target = (interior_leaf.detach() - noise)
        d_loss = model.predict_latent_velocity(z0.detach(), zg.detach(), interior_leaf, t).square().mean()
        d_loss = d_loss + target.square().mean() * 0.0
        d_loss.backward()
        self.assertIsNotNone(interior_leaf.grad)

    def test_activation_checkpointing_preserves_forward_and_gradient(self):
        plain = make_round4_model(activation_checkpointing=False)
        checkpointed = make_round4_model(activation_checkpointing=True)
        checkpointed.load_state_dict(plain.state_dict(), strict=True)
        plain.train()
        checkpointed.train()
        z0 = torch.randn(2, 8)
        zg = torch.randn(2, 8)
        actions = torch.randn(2, 5, 4)
        t = torch.full((2,), 0.4)

        plain_out = plain(z0, actions, t, mode="stage_a", goal_latent=zg)["action_velocity"]
        checkpoint_out = checkpointed(z0, actions, t, mode="stage_a", goal_latent=zg)["action_velocity"]
        torch.testing.assert_close(plain_out, checkpoint_out)
        plain_out.square().mean().backward()
        checkpoint_out.square().mean().backward()
        torch.testing.assert_close(
            plain.predictor.layers[0].attention.to_qkv.weight.grad,
            checkpointed.predictor.layers[0].attention.to_qkv.weight.grad,
        )

    def test_round4_checkpoint_round_trip_is_strict_and_old_model_schema_stays_unchanged(self):
        model = make_round4_model()
        restored = make_round4_model()
        incompatible = restored.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

        from tests.test_fast_lewam_model import make_model

        legacy = make_model()
        legacy_restored = make_model()
        incompatible = legacy_restored.load_state_dict(legacy.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(legacy.state_dict(), strict=True)

    def test_post_optimize_actions_reduces_latent_cost_within_trust_region(self):
        model = make_round4_model().eval()
        torch.manual_seed(3)
        z0 = torch.randn(2, 8)
        zg = torch.randn(2, 8)
        actions = torch.randn(2, 5, 4)
        before = model.get_cost_from_latents(z0, zg, actions[:, None])[:, 0]
        refined = model.post_optimize_actions(
            z0, zg, actions, step_size=0.05, inner_steps=3, max_rms_offset=0.2
        )
        after = model.get_cost_from_latents(z0, zg, refined[:, None])[:, 0]
        self.assertEqual(refined.shape, actions.shape)
        self.assertTrue(torch.isfinite(refined).all())
        self.assertLessEqual(float((refined - actions).pow(2).mean()), 0.2 + 1e-6)
        self.assertLessEqual(float(after.mean()), float(before.mean()) + 1e-6)
        self.assertEqual(model.last_guidance_stats["mode"], "post_opt_refine")

    def test_post_optimize_actions_rejects_bad_shapes(self):
        model = make_round4_model().eval()
        with self.assertRaises(ValueError):
            model.post_optimize_actions(
                torch.randn(2, 8), torch.randn(2, 8), torch.randn(5, 4)
            )


if __name__ == "__main__":
    unittest.main()
