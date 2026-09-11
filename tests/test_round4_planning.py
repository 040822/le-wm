import unittest

import gymnasium as gym
import numpy as np
import torch

from source.policy.round4 import (
    Round4BestOfNPolicy,
    score_candidates_in_chunks,
)
from tests.test_round4_model import make_round4_model


class Round4PlanningTests(unittest.TestCase):
    def test_candidate_split_does_not_change_verifier_argmin(self):
        model = make_round4_model().eval()
        policy = Round4BestOfNPolicy(
            model,
            proposal_source="latent",
            num_candidates=6,
            flow_steps=2,
            action_block=1,
            receding_horizon_blocks=5,
            solver_batch_size=1,
        )
        z_start = torch.randn(2, 8)
        z_goal = torch.randn(2, 8)
        candidates = torch.randn(2, 6, 5, 4)
        whole = policy.score_candidates(z_start, z_goal, candidates)
        split = score_candidates_in_chunks(
            lambda start, goal, actions: model.get_cost_from_latents(start, goal, actions),
            z_start,
            z_goal,
            candidates,
            solver_batch_size=2,
            candidate_batch_size=2,
        )
        torch.testing.assert_close(whole, split)
        self.assertTrue(torch.equal(whole.argmin(dim=1), split.argmin(dim=1)))

    def test_latent_proposal_is_decoded_then_verified_only_with_actions(self):
        model = make_round4_model().eval()
        policy = Round4BestOfNPolicy(
            model,
            proposal_source="latent",
            num_candidates=4,
            flow_steps=2,
            action_block=1,
            receding_horizon_blocks=5,
        )
        z_start = torch.randn(1, 8)
        z_goal = torch.randn(1, 8)
        paths = model.sample_latent_paths(
            z_start,
            z_goal,
            num_samples=4,
            num_steps=2,
            generator=torch.Generator().manual_seed(4),
        )
        actions = model.decode_latent_paths(paths)
        costs = policy.score_candidates(z_start, z_goal, actions)
        self.assertEqual(actions.shape, (1, 4, 5, 4))
        self.assertEqual(costs.shape, (1, 4))
        self.assertTrue(torch.isfinite(costs).all())

    def test_first_candidate_mode_never_needs_a_verifier(self):
        model = make_round4_model().eval()
        policy = Round4BestOfNPolicy(
            model,
            proposal_source="latent",
            num_candidates=4,
            flow_steps=2,
            action_block=1,
            receding_horizon_blocks=5,
            verifier="none",
            selection_rule="first",
        )
        self.assertEqual(policy.metadata()["selection_rule"], "first")
        self.assertEqual(policy.metadata()["verifier"], "none")

    def test_policy_executes_one_receding_action_block_and_records_plan_metadata(self):
        class FakeEnv:
            num_envs = 1
            single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,))
            action_space = gym.spaces.Box(-1.0, 1.0, shape=(1, 4))

        policy = Round4BestOfNPolicy(
            make_round4_model().eval(),
            proposal_source="latent",
            num_candidates=4,
            flow_steps=2,
            action_block=1,
            receding_horizon_blocks=1,
        )
        policy.set_env(FakeEnv())
        action = policy.get_action(
            {
                "pixels": torch.randn(1, 1, 3, 8, 8),
                "goal": torch.randn(1, 1, 3, 8, 8),
            }
        )
        self.assertEqual(action.shape, (1, 4))
        self.assertTrue(np.isfinite(action).all())
        self.assertEqual(policy.planning_events[-1]["proposal_source"], "latent")
        self.assertEqual(policy.planning_events[-1]["candidate_count"], 4)
        self.assertGreater(policy.planning_events[-1]["forward_count"], 0)


if __name__ == "__main__":
    unittest.main()
