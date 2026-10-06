import unittest
from unittest.mock import patch

import torch

from source.common.round5_phase6_3 import deferred_refinement_checks, encoder_bf16, fixed_condition_cache
from source.common.round5_phase6_3 import preprocess_optimized
from tests.test_round4_model import make_round4_model


class Phase63Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(63)
        self.model = make_round4_model().eval().requires_grad_(False)
        self.z0 = torch.randn(2, self.model.latent_dim)
        self.goal = torch.randn_like(self.z0)
        self.actions = torch.randn(2, self.model.action_horizon, self.model.action_dim)

    def refine(self):
        return self.model.post_optimize_actions(self.z0, self.goal, self.actions,
            inner_steps=5, collect_guidance_diagnostics=False)

    def test_e1_preserves_actions_and_every_gradient_without_parameter_gradients(self):
        original_grad = torch.autograd.grad
        recorded = []

        def spy(*args, **kwargs):
            result = original_grad(*args, **kwargs)
            recorded.append(result[0].clone())
            return result

        with patch("torch.autograd.grad", spy):
            baseline = self.refine()
            gradients = recorded[:]
            recorded.clear()
            with deferred_refinement_checks(self.model):
                optimized = self.refine()
        self.assertEqual(len(recorded), 5)
        self.assertTrue(torch.equal(baseline, optimized))
        for left, right in zip(gradients, recorded):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertNotIn("post_optimize_actions", self.model.__dict__)
        self.assertNotIn("_normalize_guidance_gradient", self.model.__dict__)

    def test_e1_rejects_nan_and_inf_latents_and_gradients_and_recovers(self):
        original_grad = torch.autograd.grad
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with deferred_refinement_checks(self.model):
                original = self.z0.clone()
                self.z0[0, 0] = invalid
                with self.assertRaises(ValueError):
                    self.refine()
                self.z0 = original

                def corrupt(*args, **kwargs):
                    result = original_grad(*args, **kwargs)
                    bad = result[0].clone()
                    bad[0, 0, 0] = invalid
                    return (bad,)

                with patch("torch.autograd.grad", corrupt):
                    with self.assertRaises(FloatingPointError):
                        self.refine()
                self.assertTrue(torch.isfinite(self.refine()).all())

    def test_e4_encoder_returns_fp32_and_restores_method(self):
        pixels = torch.rand(2, 3, 8, 8)
        with encoder_bf16(self.model):
            self.assertEqual(self.model.encode_pixels(pixels).dtype, torch.float32)
        self.assertNotIn("encode_pixels", self.model.__dict__)

    def test_e2_preserves_candidate_rng_costs_and_refinement_gradients(self):
        def run():
            generator = torch.Generator().manual_seed(6302)
            candidates = self.model.sample_actions(self.z0, goal_latent=self.goal,
                num_steps=2, generator=generator)
            costs = self.model.get_cost_from_latents(self.z0, self.goal, candidates[:, None])
            recorded = []
            original_grad = torch.autograd.grad

            def spy(*args, **kwargs):
                result = original_grad(*args, **kwargs)
                recorded.append(result[0].clone())
                return result

            with patch("torch.autograd.grad", spy):
                refined = self.refine()
            return candidates, costs, refined, recorded, generator.get_state()

        baseline = run()
        with fixed_condition_cache(self.model) as cache:
            optimized = run()
            self.assertGreater(cache.hits, 0)
        for left, right in zip(baseline, optimized):
            if isinstance(left, list):
                for a, b in zip(left, right):
                    self.assertTrue(torch.equal(a, b))
            else:
                self.assertTrue(torch.equal(left, right))
        self.assertNotIn("forward", self.model.z_condition.__dict__)

    def test_e2_invalidates_mutations_replans_and_batch_views(self):
        z = torch.randn(64, self.model.latent_dim)
        original = self.model.z_condition.forward
        with fixed_condition_cache(self.model) as cache:
            full = self.model.z_condition(z)
            self.assertTrue(torch.equal(full, original(z)))
            self.assertIs(self.model.z_condition(z), full)
            for index in (0, 1, 17, 63):
                self.assertTrue(torch.equal(self.model.z_condition(z[index:index+1]),
                                            original(z[index:index+1])))
            z[0].add_(1)
            self.assertTrue(torch.equal(self.model.z_condition(z), original(z)))
            self.model.encode_pixels(torch.rand(2, 3, 8, 8))
            self.assertEqual(cache.values, {})
            differentiable = z.detach().requires_grad_(True)
            self.model.z_condition(differentiable).sum().backward()
            self.assertIsNotNone(differentiable.grad)

    def test_e2_requires_frozen_parameters(self):
        self.model.z_condition.weight.requires_grad_(True)
        with self.assertRaises(ValueError):
            with fixed_condition_cache(self.model):
                pass

    def test_e3_native_pipeline_parity_and_input_ownership_for_both_adapters(self):
        import numpy as np
        from torchvision.transforms import v2
        from sklearn.preprocessing import StandardScaler
        from source.policy.round4 import make_round4_policy
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.Resize(size=(16, 16))])
        for mode in ("P0", "P3"):
            policy = make_round4_policy(self.model, mode=mode,
                plan_config={"horizon": 5, "receding_horizon": 5, "action_block": 1},
                process={"proprio": StandardScaler().fit(np.arange(12).reshape(4, 3))},
                transform={"pixels": transform, "goal": transform}, device="cpu")
            for batch in (1, 50):
                for dtype in (np.uint8, np.float32):
                    rng = np.random.default_rng(6303)
                    images = rng.integers(0, 256, (batch, 2, 13, 17, 3), dtype=np.uint8).astype(dtype)
                    info = {"pixels": images.copy(), "goal": images.copy(),
                            "proprio": rng.normal(size=(batch, 2, 3)), "terminated": np.zeros(batch, bool)}
                    original = {k: v.copy() for k, v in info.items()}
                    baseline = policy._prepare_info(info)
                    with preprocess_optimized(policy):
                        optimized = policy._prepare_info(info)
                    for key in info:
                        self.assertTrue(np.array_equal(info[key], original[key]))
                        self.assertTrue(torch.equal(baseline[key], optimized[key]), (mode, batch, dtype, key))
                    optimized["pixels"].zero_()
                    self.assertTrue(np.array_equal(info["pixels"], original["pixels"]))


if __name__ == "__main__":
    unittest.main()
