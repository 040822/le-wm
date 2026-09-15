import unittest

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

from source.policy.fast_lewam_eval import (
    FastLeWAMChunkPolicy,
    StageBModelView,
    make_fast_lewam_policy,
)
from tests.test_fast_lewam_model import make_model


class FakeVectorEnv:
    num_envs = 2
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2, 2))


class CountingImageTransform:
    def __init__(self):
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        return image.as_subclass(torch.Tensor).float()


class FastLeWAMEvalTests(unittest.TestCase):
    def test_non_token_guidance_requests_goal_observations(self):
        transform = CountingImageTransform()
        policy = FastLeWAMChunkPolicy(
            make_model().eval(),
            mode="stage_a",
            action_block=2,
            receding_horizon_blocks=1,
            inference_steps=1,
            guidance_mode="post_opt",
            guidance_inner_steps=1,
            transform={"pixels": transform, "goal": transform},
        )
        policy.set_env(FakeVectorEnv())
        pixels = np.random.randn(2, 1, 8, 8, 3).astype(np.float32)
        goal = np.random.randn(2, 1, 8, 8, 3).astype(np.float32)

        action = policy.get_action({"pixels": pixels, "goal": goal})

        self.assertEqual(action.shape, (2, 2))
        self.assertTrue(np.isfinite(action).all())

    def test_stage_a_buffered_action_skips_observation_preprocessing(self):
        transform = CountingImageTransform()
        policy = FastLeWAMChunkPolicy(
            make_model().eval(),
            mode="stage_a",
            action_block=2,
            receding_horizon_blocks=1,
            inference_steps=1,
            transform={"pixels": transform},
        )
        policy.set_env(FakeVectorEnv())
        info = {"pixels": np.random.randn(2, 1, 8, 8, 3).astype(np.float32)}

        first = policy.get_action(info)
        second = policy.get_action(info)

        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.isfinite(second).all())
        self.assertEqual(transform.calls, 2)

    def test_partial_replan_preprocesses_only_flushed_environments(self):
        transform = CountingImageTransform()
        policy = FastLeWAMChunkPolicy(
            make_model().eval(),
            mode="stage_a",
            action_block=2,
            receding_horizon_blocks=1,
            inference_steps=1,
            transform={"pixels": transform},
        )
        policy.set_env(FakeVectorEnv())
        pixels = np.random.randn(2, 1, 8, 8, 3).astype(np.float32)
        policy.get_action({"pixels": pixels})

        action = policy.get_action(
            {"pixels": pixels, "_needs_flush": np.array([True, False])}
        )

        self.assertTrue(np.isfinite(action).all())
        self.assertEqual(transform.calls, 3)

    def test_stage_c_buffered_action_skips_observation_preprocessing(self):
        transform = CountingImageTransform()
        policy = FastLeWAMChunkPolicy(
            make_model().eval(),
            mode="stage_c",
            action_block=2,
            receding_horizon_blocks=1,
            inference_steps=1,
            transform={"pixels": transform},
        )
        policy.set_env(FakeVectorEnv())
        info = {"pixels": np.random.randn(2, 1, 8, 8, 3).astype(np.float32)}

        first = policy.get_action(info)
        second = policy.get_action(info)

        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.isfinite(second).all())
        self.assertEqual(transform.calls, 2)

    def test_terminated_environment_does_not_replan_or_consume_buffer(self):
        transform = CountingImageTransform()
        policy = FastLeWAMChunkPolicy(
            make_model().eval(),
            mode="stage_a",
            action_block=2,
            receding_horizon_blocks=1,
            inference_steps=1,
            transform={"pixels": transform},
        )
        policy.set_env(FakeVectorEnv())
        pixels = np.random.randn(2, 1, 8, 8, 3).astype(np.float32)
        policy.get_action({"pixels": pixels})

        action = policy.get_action(
            {"pixels": pixels, "terminated": np.array([True, False])}
        )

        self.assertTrue(np.isnan(action[0]).all())
        self.assertTrue(np.isfinite(action[1]).all())
        self.assertEqual(transform.calls, 2)
        self.assertEqual(len(policy._action_buffer[0]), 1)

    def test_partial_replan_uses_cyclically_shifted_goal(self):
        model = make_model(stage_a_goal_injection="token").eval()
        encoded = []
        model.encoder.register_forward_pre_hook(
            lambda _module, inputs: encoded.append(inputs[0].detach().clone())
        )
        policy = FastLeWAMChunkPolicy(
            model,
            mode="stage_a",
            action_block=2,
            receding_horizon_blocks=1,
            inference_steps=1,
            goal_mode="cyclic_shift",
        )
        policy.set_env(FakeVectorEnv())
        pixels = np.zeros((2, 1, 3, 8, 8), dtype=np.float32)
        goal = np.stack(
            [np.ones((1, 3, 8, 8)), np.full((1, 3, 8, 8), 2.0)]
        ).astype(np.float32)
        policy.get_action({"pixels": pixels, "goal": goal})
        encoded.clear()

        policy.get_action(
            {
                "pixels": pixels,
                "goal": goal,
                "_needs_flush": np.array([True, False]),
            }
        )

        self.assertEqual(encoded[-1].shape[0], 2)
        self.assertAlmostEqual(encoded[-1][1].mean().item(), 2.0)

    def test_stage_a_policy_buffers_one_complete_action_block(self):
        model = make_model().eval()
        policy = FastLeWAMChunkPolicy(
            model,
            mode="stage_a",
            action_block=2,
            receding_horizon_blocks=1,
            inference_steps=2,
        )
        policy.set_env(FakeVectorEnv())
        info = {"pixels": np.random.randn(2, 1, 3, 8, 8).astype(np.float32)}

        first = policy.get_action(info)
        second = policy.get_action(info)

        self.assertEqual(first.shape, (2, 2))
        self.assertEqual(second.shape, (2, 2))
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.isfinite(second).all())

    def test_stage_a_factory_forwards_sampling_seed(self):
        model = make_model().eval()
        policy = make_fast_lewam_policy(
            model,
            solver_cfg=None,
            plan_config={
                "horizon": 3,
                "receding_horizon": 1,
                "action_block": 2,
            },
            process={},
            transform={},
            device="cpu",
            mode="stage_a",
            seed=123,
        )

        self.assertEqual(policy.seed, 123)

    def test_direct_stages_use_shared_plan_receding_horizon(self):
        model = make_model().eval()

        for stage in ("stage_a", "stage_c"):
            policy = make_fast_lewam_policy(
                model,
                solver_cfg=None,
                plan_config={
                    "horizon": 3,
                    "receding_horizon": 3,
                    "action_block": 2,
                },
                process={},
                transform={},
                device="cpu",
                mode=stage,
            )
            self.assertEqual(policy.receding_horizon_blocks, 3)

    def test_stage_b_policy_exposes_cost_only_model_to_existing_solver(self):
        model = make_model().eval()
        solver_cfg = OmegaConf.create(
            {
                "_target_": "stable_worldmodel.solver.CEMSolver",
                "model": "???",
                "batch_size": 1,
                "num_samples": 2,
                "n_steps": 1,
                "topk": 1,
                "device": "cpu",
            }
        )
        policy = make_fast_lewam_policy(
            model,
            solver_cfg=solver_cfg,
            plan_config={
                "horizon": 3,
                "receding_horizon": 1,
                "action_block": 2,
            },
            process={},
            transform={},
            device="cpu",
            mode="stage_b",
        )

        self.assertIsInstance(policy.solver.model, StageBModelView)
        self.assertFalse(hasattr(policy.solver.model, "get_action"))

        class WrongActionVectorEnv(FakeVectorEnv):
            single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))
            action_space = gym.spaces.Box(-1.0, 1.0, shape=(2, 3))

        with self.assertRaisesRegex(ValueError, "model action_dim=4"):
            policy.set_env(WrongActionVectorEnv())


    def test_stage_b_cached_cost_matches_candidate_expanded_cost(self):
        model = make_model().eval()
        view = StageBModelView(model)
        batch, samples = 2, 4
        base_pixels = torch.randn(batch, 2, 3, 8, 8)
        base_goal = torch.randn(batch, 2, 3, 8, 8)
        shared_info = {
            "pixels": base_pixels[:, None].expand(-1, samples, -1, -1, -1, -1),
            "goal": base_goal[:, None].expand(-1, samples, -1, -1, -1, -1),
        }
        legacy_info = {
            key: value.clone() for key, value in shared_info.items()
        }
        candidates = torch.randn(batch, samples, 3, 4)

        expected = model.get_cost(legacy_info, candidates)
        actual = view.get_cost(shared_info, candidates)

        torch.testing.assert_close(actual, expected)

    def test_stage_b_view_preserves_candidate_specific_observations(self):
        model = make_model().eval()
        view = StageBModelView(model)
        batch, samples = 2, 4
        info = {
            "pixels": torch.randn(batch, samples, 2, 3, 8, 8),
            "goal": torch.randn(batch, samples, 2, 3, 8, 8),
        }
        reference = {key: value.clone() for key, value in info.items()}
        candidates = torch.randn(batch, samples, 3, 4)

        expected = model.get_cost(reference, candidates)
        actual = view.get_cost(info, candidates)

        torch.testing.assert_close(actual, expected)


    def test_stage_b_cached_cost_preserves_mixed_image_dtypes(self):
        model = make_model().eval()
        view = StageBModelView(model)
        batch, samples = 2, 4
        info = {
            "pixels": torch.randint(
                0, 256, (batch, 1, 2, 3, 8, 8), dtype=torch.uint8
            ).expand(-1, samples, -1, -1, -1, -1),
            "goal": torch.randn(batch, 1, 2, 3, 8, 8).expand(
                -1, samples, -1, -1, -1, -1
            ),
        }
        candidates = torch.randn(batch, samples, 3, 4)

        actual = view.get_cost(info, candidates)
        expected = model.get_cost(
            {"pixels": info["pixels"], "goal": info["goal"]}, candidates
        )

        torch.testing.assert_close(actual, expected)


    def test_stage_b_cache_is_invalidated_when_observations_are_replaced(self):
        model = make_model().eval()
        view = StageBModelView(model)
        batch, samples = 2, 4
        info = {
            "pixels": torch.randn(batch, 1, 2, 3, 8, 8).expand(
                -1, samples, -1, -1, -1, -1
            ),
            "goal": torch.randn(batch, 1, 2, 3, 8, 8).expand(
                -1, samples, -1, -1, -1, -1
            ),
        }
        candidates = torch.randn(batch, samples, 3, 4)
        view.get_cost(info, candidates)
        replacement_pixels = torch.randn(batch, 1, 2, 3, 8, 8).expand(
            -1, samples, -1, -1, -1, -1
        )
        info["pixels"] = replacement_pixels

        actual = view.get_cost(info, candidates)
        expected = model.get_cost(
            {"pixels": replacement_pixels, "goal": info["goal"]}, candidates
        )

        torch.testing.assert_close(actual, expected)


    def test_token_stage_a_cyclic_shift_uses_a_goal_derangement(self):
        model = make_model(stage_a_goal_injection="token").eval()
        policy = FastLeWAMChunkPolicy(
            model, mode="stage_a", action_block=2, inference_steps=1,
            goal_mode="cyclic_shift",
        )
        policy.set_env(FakeVectorEnv())
        self.assertFalse(np.any(policy._goal_indices == np.arange(2)))
        info = {
            "pixels": np.zeros((2, 1, 3, 8, 8), dtype=np.float32),
            "goal": np.ones((2, 1, 3, 8, 8), dtype=np.float32),
        }
        self.assertTrue(np.isfinite(policy.get_action(info)).all())
if __name__ == "__main__":
    unittest.main()
