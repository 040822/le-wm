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


if __name__ == "__main__":
    unittest.main()
