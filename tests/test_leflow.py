import unittest

import torch
from gymnasium.spaces import Box
from omegaconf import OmegaConf
from torch import nn

from source.model.leflow.latent_planner import (
    InverseDynamics,
    LatentPathFlow,
    LatentPlannerRuntime,
    LearnedLatentPathSolver,
)
from source.common.eval import DatasetEvaluationSession
from stable_worldmodel.policy import PlanConfig


class TinyLeWM(nn.Module):
    def __init__(self, latent_dim=4, action_dim=2):
        super().__init__()
        self.action_projection = nn.Linear(action_dim, latent_dim)

    def encode(self, info):
        pixels = info["pixels"].float()
        values = pixels.mean(dim=tuple(range(2, pixels.ndim)))
        info["emb"] = values[..., None].expand(*values.shape, 4)
        return info

    def action_encoder(self, actions):
        return self.action_projection(actions)

    def predict(self, embeddings, action_embeddings):
        return embeddings + action_embeddings


def make_runtime():
    flow = LatentPathFlow(
        latent_dim=4,
        hidden_dim=16,
        depth=1,
        max_horizon=8,
        time_dim=8,
    )
    inverse = InverseDynamics(
        latent_dim=4,
        action_dim=2,
        hidden_dim=16,
        depth=1,
    )
    return LatentPlannerRuntime(
        lewm=TinyLeWM(),
        flow=flow,
        inverse_dynamics=inverse,
        action_block=1,
    ).eval()


class LeFlowTests(unittest.TestCase):
    def test_sampled_path_has_exact_start_and_goal_endpoints(self):
        runtime = make_runtime()
        z_start = torch.randn(2, 4)
        z_goal = torch.randn(2, 4)

        paths = runtime.sample_paths(
            z_start,
            z_goal,
            horizon=5,
            num_samples=3,
            flow_steps=2,
            generator=torch.Generator().manual_seed(7),
        )

        self.assertEqual(paths.shape, (2, 3, 6, 4))
        self.assertTrue(
            torch.equal(paths[:, :, 0], z_start[:, None].expand_as(paths[:, :, 0]))
        )
        self.assertTrue(
            torch.equal(paths[:, :, -1], z_goal[:, None].expand_as(paths[:, :, -1]))
        )

    def test_decode_actions_maps_each_path_transition_to_an_action(self):
        runtime = make_runtime()
        paths = torch.randn(2, 3, 6, 4)

        actions = runtime.decode_actions(paths)

        self.assertEqual(actions.shape, (2, 3, 5, 2))

    def test_solver_returns_selected_action_plan_through_worldmodel_interface(self):
        runtime = make_runtime()
        solver = LearnedLatentPathSolver(
            model=runtime,
            num_samples=3,
            flow_steps=2,
            device="cpu",
            seed=11,
        )
        solver.configure(
            action_space=Box(low=-1.0, high=1.0, shape=(1, 2), dtype=float),
            n_envs=1,
            config=PlanConfig(
                horizon=5,
                receding_horizon=5,
                action_block=1,
            ),
        )

        result = solver.solve(
            {
                "pixels": torch.zeros(1, 3, 8, 8),
                "goal": torch.ones(1, 3, 8, 8),
            }
        )

        self.assertEqual(result["actions"].shape, (1, 5, 2))
        self.assertEqual(result["goal_costs"].shape, (1, 3))
        self.assertEqual(len(result["costs"]), 1)

    def test_existing_evaluation_session_adapts_runtime_to_latent_flow_solver(self):
        session = DatasetEvaluationSession.__new__(DatasetEvaluationSession)
        session.cfg = OmegaConf.create(
            {
                "seed": 42,
                "solver": {"_target_": "stable_worldmodel.solver.CEMSolver"},
                "plan_config": {
                    "horizon": 5,
                    "receding_horizon": 5,
                    "action_block": 1,
                },
            }
        )
        session.process = {}
        session.transform = {}

        policy = session._build_policy(make_runtime(), identity=None, device="cpu")

        self.assertEqual(type(policy.solver).__name__, "LearnedLatentPathSolver")
        self.assertEqual(policy.solver.num_samples, 64)
        self.assertEqual(policy.solver.flow_steps, 16)


if __name__ == "__main__":
    unittest.main()
