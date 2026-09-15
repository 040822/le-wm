import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from tests.test_fast_lewam_model import make_model
from source.experiments.round3_guidance_diagnostics import (
    GuidanceDiagnosticConfig,
    e2_method_specs,
    finite_difference_guidance_check,
    make_signed_guidance_candidates,
    make_guidance_manifest,
)


class Round3GuidanceTests(unittest.TestCase):
    def test_guidance_manifest_keeps_all_report_only_e2_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            payload = make_guidance_manifest(
                task="reacher",
                checkpoint=Path("outputs/fast_lewam/reacher/0717_/checkpoints/fast_lewam_weights_epoch_10.pt"),
                output=output,
            )
            self.assertEqual(set(payload["e2"]), set(e2_method_specs(GuidanceDiagnosticConfig())))
            self.assertEqual(payload["decision_policy"], "report-only; user decides whether guidance enters fixed replay")
            self.assertEqual(json.loads(output.read_text())["config"]["flow_steps"], 10)

    def test_signed_probe_panel_has_equal_rms_and_expected_labels(self):
        actor = torch.zeros(3, 4)
        gradient = torch.ones_like(actor)
        panel, labels = make_signed_guidance_candidates(actor, gradient, rms=0.03, seed=2)
        self.assertEqual(labels[1], "negative_latent_gradient")
        torch.testing.assert_close(
            (panel - actor).square().mean(dim=(1, 2)).sqrt(),
            torch.tensor([0.0, 0.03, 0.03, 0.03]),
            rtol=0.0,
            atol=1e-6,
        )

    def test_finite_difference_probe_confirms_cost_gradient_sign(self):
        model = make_model().eval()
        torch.manual_seed(21)
        result = finite_difference_guidance_check(
            model,
            torch.randn(1, 8),
            torch.randn(1, 8),
            torch.randn(1, 3, 4),
            epsilon=1e-3,
        )
        self.assertTrue(result["sign_match"])
        self.assertTrue(result["descent_probe_lower_cost"])
    def test_none_guidance_reproduces_the_existing_euler_path(self):
        model = make_model(stage_a_goal_injection="token").eval()
        z0 = torch.randn(2, 8)
        goal = torch.randn(2, 8)
        noise = torch.randn(2, 3, 4)

        expected = model.sample_actions(
            z0,
            noise=noise,
            goal_latent=goal,
            num_steps=4,
        )
        actual = model.sample_actions(
            z0,
            noise=noise,
            goal_latent=goal,
            num_steps=4,
            guidance_mode="none",
        )

        torch.testing.assert_close(actual, expected)
        self.assertEqual(model.last_guidance_stats["backward_count"], 0)

    def test_guided_flow_is_finite_and_does_not_leave_parameter_gradients(self):
        model = make_model(stage_a_goal_injection="token").eval()
        z0 = torch.randn(2, 8)
        goal = torch.randn(2, 8)
        noise = torch.randn(2, 3, 4)
        before = copy.deepcopy(model.state_dict())

        actions = model.sample_actions(
            z0,
            noise=noise,
            goal_latent=goal,
            num_steps=4,
            guidance_mode="guided_flow",
            guidance_step_size=0.01,
            guidance_last_steps=2,
            guidance_inner_steps=2,
            guidance_max_rms_offset=0.20,
        )

        self.assertEqual(actions.shape, (2, 3, 4))
        self.assertTrue(torch.isfinite(actions).all())
        self.assertGreater(model.last_guidance_stats["backward_count"], 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for name, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[name])

    def test_non_token_guidance_encodes_goal_in_raw_get_action(self):
        model = make_model().eval()
        pixels = torch.randn(2, 1, 3, 8, 8)
        goal = torch.randn(2, 1, 3, 8, 8)
        action = model.get_action(
            {"pixels": pixels, "goal": goal},
            horizon=3,
            num_steps=1,
            guidance_mode="post_opt",
            guidance_inner_steps=1,
        )
        self.assertEqual(tuple(action.shape), (2, 3, 4))
        self.assertTrue(torch.isfinite(action).all())


if __name__ == "__main__":
    unittest.main()
