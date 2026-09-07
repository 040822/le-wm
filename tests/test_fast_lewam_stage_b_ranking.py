import unittest

import numpy as np
import torch

from scripts.diagnose_fast_lewam_stage_b_ranking import (
    build_candidate_set,
    compute_physical_terminal_cost,
    summarize_candidate_pools,
    summarize_ranking,
)


class FastLeWAMStageBRankingTests(unittest.TestCase):
    def test_candidate_set_keeps_anchor_candidates_and_family_counts(self):
        expert = torch.zeros(2, 3)
        actor = torch.ones(2, 3)

        candidates, families = build_candidate_set(
            expert,
            actor,
            random_candidates=3,
            actor_neighbors=2,
            expert_neighbors=2,
            expert_noise_std=0.1,
            actor_noise_std=1.0,
            generator=torch.Generator().manual_seed(7),
        )

        self.assertEqual(candidates.shape, (12, 2, 3))
        self.assertEqual(families[:5], [
            "expert",
            "actor",
            "zero",
            "time_reverse",
            "time_roll",
        ])
        self.assertTrue(torch.equal(candidates[0], expert))
        self.assertTrue(torch.equal(candidates[1], actor))
        self.assertEqual(families.count("random"), 3)
        self.assertEqual(families.count("actor_neighbor"), 2)
        self.assertEqual(families.count("expert_neighbor"), 2)

    def test_ranking_summary_exposes_inverted_order_and_regret(self):
        summary = summarize_ranking(
            predicted_cost=np.array([0.0, 1.0, 2.0]),
            true_cost=np.array([2.0, 1.0, 0.0]),
            successes=np.array([False, True, True]),
            families=["actor", "random", "expert"],
            topk=1,
        )

        self.assertAlmostEqual(summary["spearman"], -1.0)
        self.assertEqual(summary["topk_recall"], 0.0)
        self.assertEqual(summary["predicted_top1_success"], 0.0)
        self.assertEqual(summary["oracle_success"], 1.0)
        self.assertAlmostEqual(summary["best_candidate_regret"], 2.0)
        self.assertEqual(summary["predicted_top1_family"], "actor")

    def test_candidate_pools_compare_zero_actor_and_expert_coverage(self):
        pools = summarize_candidate_pools(
            predicted_cost=np.array([2.0, 0.0, 3.0, 1.0, 0.5, 0.2]),
            true_cost=np.array([2.0, 1.0, 3.0, 0.0, 0.5, 0.1]),
            successes=np.array([False, True, False, True, True, True]),
            families=[
                "zero",
                "random",
                "actor",
                "actor_neighbor",
                "expert",
                "expert_neighbor",
            ],
            topk=1,
        )

        self.assertEqual(pools["zero_centered"]["oracle_success"], 1.0)
        self.assertEqual(pools["actor_centered"]["predicted_top1_success"], 1.0)
        self.assertEqual(pools["expert_centered"]["candidate_success_rate"], 1.0)

    def test_physical_terminal_cost_matches_task_success_geometry(self):
        pusht = compute_physical_terminal_cost(
            "pusht",
            {
                "state": np.array([[[1.0, 2.0, 3.0]]]),
                "goal_state": np.array([[[1.0, 4.0, 6.0]]]),
            },
        )
        reacher = compute_physical_terminal_cost(
            "reacher",
            {
                "qpos": np.array([[[1.0, 2.0]]]),
                "goal_qpos": np.array([[[4.0, 6.0]]]),
            },
        )

        self.assertTrue(np.allclose(pusht, [np.sqrt(13.0)]))
        self.assertTrue(np.allclose(reacher, [5.0]))


if __name__ == "__main__":
    unittest.main()
