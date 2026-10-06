import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from source.common.eval import DatasetEvaluationSession
from source.common.round3_eval import _attach_cem_archive_callback
from source.model.leflow.latent_planner import LatentPlannerRuntime
from source.model.fast_lewam.jepa import FastLeWAM


class LeFlowEvalSeedResolutionTest(unittest.TestCase):
    def test_eval_policy_seed_does_not_resolve_standalone_solver_seed(self):
        session = object.__new__(DatasetEvaluationSession)
        session.cfg = OmegaConf.create(
            {
                "solver": {"_target_": "some.other.Solver"},
                "eval": {"policy_seed": 20042},
                "seed": 10042,
                "plan_config": {},
            }
        )
        session.process = object()
        session.transform = {}
        runtime = object.__new__(LatentPlannerRuntime)
        expected_policy = object()

        with patch(
            "source.policy.leflow.make_leflow_policy",
            return_value=expected_policy,
        ):
            actual_policy = session._build_policy(
                runtime, identity=None, device="cpu"
            )

        self.assertIs(actual_policy, expected_policy)
        self.assertEqual(session.cfg.solver.seed, 20042)
        self.assertIsNone(session.cfg.solver.checkpoint)

    def test_leflow_solver_config_skips_cem_archive_callback(self):
        policy = object()
        cfg = OmegaConf.create(
            {"solver": {"_target_": "LearnedLatentPathSolver", "flow_steps": 16}}
        )

        with patch("source.common.cvpr_table1.attach_cem_archive_callback") as attach:
            result = _attach_cem_archive_callback(policy, cfg)

        self.assertIsNone(result)
        attach.assert_not_called()

    def test_cem_solver_config_keeps_candidate_archive_callback(self):
        policy = object()
        cfg = OmegaConf.create({"solver": {"n_steps": 30, "topk": 30}})
        expected_callback = object()

        with patch(
            "source.common.cvpr_table1.attach_cem_archive_callback",
            return_value=expected_callback,
        ) as attach:
            result = _attach_cem_archive_callback(policy, cfg)

        self.assertIs(result, expected_callback)
        attach.assert_called_once_with(policy, iterations=30, topk=30)

    def test_latent_action_generation_forwards_diagnostic_flag(self):
        class StubModel:
            action_horizon = 2
            stage_a_goal_injection = "none"

            def sample_actions(self, z0, **kwargs):
                self.sample_kwargs = kwargs
                return torch.zeros((z0.shape[0], self.action_horizon, 1))

        model = StubModel()
        z0 = torch.zeros((1, 4))

        FastLeWAM.get_action_from_latents(
            model,
            z0,
            None,
            horizon=1,
            collect_guidance_diagnostics=False,
        )

        self.assertFalse(model.sample_kwargs["collect_guidance_diagnostics"])


if __name__ == "__main__":
    unittest.main()
