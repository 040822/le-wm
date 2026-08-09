import unittest

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

from source.policy.fast_lewam_eval import FastLeWAMChunkPolicy, make_fast_lewam_policy
from tests.test_fast_lewam_model import make_model


class FakeVectorEnv:
    num_envs = 2
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2, 2))


def solver_config():
    return OmegaConf.create(
        {
            "_target_": "stable_worldmodel.solver.CEMSolver",
            "model": "???",
            "batch_size": 1,
            "num_samples": 2,
            "n_steps": 1,
            "topk": 1,
            "device": "cpu",
            "seed": 9,
        }
    )


class FastLeWAMSolverSmokeTests(unittest.TestCase):
    def test_existing_cem_world_policy_executes_stage_b_cost(self):
        policy = make_fast_lewam_policy(
            make_model().eval(),
            solver_cfg=solver_config(),
            plan_config={
                "horizon": 3,
                "receding_horizon": 1,
                "action_block": 2,
                "warm_start": False,
            },
            process={},
            transform={},
            device="cpu",
            mode="stage_b",
        )
        policy.set_env(FakeVectorEnv())
        info = {
            "pixels": np.random.randn(2, 1, 3, 8, 8).astype(np.float32),
            "goal": np.random.randn(2, 1, 3, 8, 8).astype(np.float32),
        }

        action = policy.get_action(info)

        self.assertEqual(action.shape, (2, 2))
        self.assertTrue(np.isfinite(action).all())

    def test_actor_warm_start_reproduces_the_first_cem_plan(self):
        info = {
            "pixels": np.random.randn(2, 1, 3, 8, 8).astype(np.float32),
            "goal": np.random.randn(2, 1, 3, 8, 8).astype(np.float32),
        }
        policies = [
            make_fast_lewam_policy(
                make_model(stage_a_goal_injection="token").eval(),
                solver_cfg=solver_config(),
                plan_config={
                    "horizon": 3,
                    "receding_horizon": 1,
                    "action_block": 2,
                    "warm_start": False,
                },
                process={},
                transform={},
                device="cpu",
                mode="stage_b",
                actor_warm_start=True,
                inference_steps=2,
                seed=31,
            )
            for _ in range(2)
        ]
        for policy in policies:
            policy.set_env(FakeVectorEnv())

        first = policies[0].get_action(info)
        second = policies[1].get_action(info)

        self.assertEqual(first.shape, (2, 2))
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.array_equal(first, second))

    def test_actor_warm_start_seed_changes_the_initial_actor_sequence(self):
        info = {
            "pixels": torch.randn(2, 1, 3, 8, 8),
            "goal": torch.randn(2, 1, 3, 8, 8),
        }
        policies = [
            make_fast_lewam_policy(
                make_model(stage_a_goal_injection="token").eval(),
                solver_cfg=solver_config(),
                plan_config={
                    "horizon": 3,
                    "receding_horizon": 1,
                    "action_block": 2,
                    "warm_start": False,
                },
                process={},
                transform={},
                device="cpu",
                mode="stage_b",
                actor_warm_start=True,
                inference_steps=2,
                seed=seed,
            )
            for seed in (31, 32)
        ]

        first = policies[0].solver.model.get_action(info, horizon=3)
        second = policies[1].solver.model.get_action(info, horizon=3)

        self.assertFalse(torch.equal(first, second))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_actor_warm_start_moves_observations_to_the_model_device(self):
        solver_cfg = solver_config()
        solver_cfg.device = "cuda"
        policy = make_fast_lewam_policy(
            make_model(stage_a_goal_injection="token").eval(),
            solver_cfg=solver_cfg,
            plan_config={
                "horizon": 3,
                "receding_horizon": 1,
                "action_block": 2,
                "warm_start": False,
            },
            process={},
            transform={},
            device="cuda",
            mode="stage_b",
            actor_warm_start=True,
            inference_steps=2,
            seed=31,
        )
        policy.set_env(FakeVectorEnv())
        info = {
            "pixels": np.random.randn(2, 1, 3, 8, 8).astype(np.float32),
            "goal": np.random.randn(2, 1, 3, 8, 8).astype(np.float32),
        }

        action = policy.get_action(info)

        self.assertEqual(action.shape, (2, 2))
        self.assertTrue(np.isfinite(action).all())

    def test_direct_policy_seed_reproduces_first_action_chunk(self):
        info = {"pixels": np.random.randn(2, 1, 3, 8, 8).astype(np.float32)}
        policies = [
            FastLeWAMChunkPolicy(
                make_model().eval(),
                mode="stage_a",
                action_block=2,
                receding_horizon_blocks=1,
                inference_steps=2,
                seed=31,
            )
            for _ in range(2)
        ]
        for policy in policies:
            policy.set_env(FakeVectorEnv())

        first = policies[0].get_action(info)
        second = policies[1].get_action(info)

        self.assertTrue(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
