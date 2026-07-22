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


class FastLeWAMEvalTests(unittest.TestCase):
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
