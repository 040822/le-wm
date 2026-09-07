import unittest
from unittest.mock import patch

import torch

from source.model.fast_lewam.modules import SIGReg
from source.policy.fast_lewam import FastLeWAMPolicy, fast_lewam_forward
from tests.test_fast_lewam_model import make_model


class FakeTrainingModule:
    def __init__(
        self,
        stage_a_goal_injection="none",
        action_horizon=3,
        token_encoding="legacy",
    ):
        self.model = make_model(
            action_horizon=action_horizon,
            stage_a_goal_injection=stage_a_goal_injection,
            token_encoding=token_encoding,
        )
        self.sigreg = SIGReg(knots=5, num_proj=8)
        self.current_epoch = 5
        self.logged = None

    def log_dict(self, metrics, **kwargs):
        self.logged = metrics


class RecordingStageBTimestepModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.stage_b_timesteps = []

    def encode_pixels(self, pixels):
        return self.model.encode_pixels(pixels)

    def forward(self, z0, actions, timestep, **kwargs):
        if kwargs.get("mode") == "stage_b":
            self.stage_b_timesteps.append(timestep.detach().clone())
        return self.model(z0, actions, timestep, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class ZeroSIGReg:
    def __call__(self, embeddings):
        return embeddings.new_zeros(())


class TerminalOnlyTrainingModel(torch.nn.Module):
    action_dim = 2
    action_horizon = 3
    latent_dim = 2
    stage_b_attention_mode = "terminal_full"
    stage_b_dynamics = "parallel_prefix"

    def encode_pixels(self, pixels):
        values = pixels[:, :, 0, 0, 0]
        return torch.stack((values, values + 10.0), dim=-1)

    def predict_training_latents(self, states, actions, task_condition=None):
        return states.new_zeros((states.shape[0], 1, states.shape[-1]))


class TerminalOnlyTrainingModule:
    def __init__(self):
        self.model = TerminalOnlyTrainingModel()
        self.sigreg = ZeroSIGReg()
        self.current_epoch = 0
        self.logged = None

    def log_dict(self, metrics, **kwargs):
        self.logged = metrics


class FastLeWAMPolicyTests(unittest.TestCase):
    def test_terminal_stage_b_loss_uses_exact_z_h_without_prefix_broadcast(self):
        module = TerminalOnlyTrainingModule()
        # The fourth frame is z_H; use a separate tensor so all prefix values
        # are visibly different from the endpoint target.
        pixels = torch.tensor(
            [
                [[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]],
                [[[[10.0]]], [[[20.0]]], [[[30.0]]], [[[40.0]]]],
            ]
        )
        batch = {
            "pixels": pixels,
            "action": torch.zeros(2, 3, 2),
        }

        output = fast_lewam_forward(
            module,
            batch=batch,
            stage="fit",
            action_horizon=3,
            train_mode="stage_b",
            lambda_latent=1.0,
            lambda_sigreg=0.0,
            detach_clean_action=False,
            latent_loss_noise_threshold=0.2,
            latent_action_mix_epochs=0,
        )

        expected_terminal = output["emb"][:, -1:].square().mean()
        broadcasted_prefix = output["emb"][:, 1:].square().mean()
        self.assertEqual(output["predicted_latents"].shape, (2, 1, 2))
        self.assertTrue(
            torch.equal(output["latent_terminal_loss"], expected_terminal)
        )
        self.assertTrue(
            torch.equal(
                output["weighted_latent_terminal_loss"], expected_terminal
            )
        )
        self.assertTrue(
            torch.equal(output["latent_prefix_loss"], output["latent_terminal_loss"])
        )
        self.assertNotEqual(output["latent_terminal_loss"].item(), broadcasted_prefix.item())
        self.assertIn("fit/latent_terminal_loss", module.logged)
        self.assertIn("fit/weighted_latent_terminal_loss", module.logged)

    def test_non_strict_attention_requires_parallel_stage_b_training(self):
        for attention_mode in ("block_causal", "terminal_full"):
            for train_mode, dynamics in (
                ("stage_ab", "parallel_prefix"),
                ("stage_c", "parallel_prefix"),
                ("stage_b", "serial_one_step"),
            ):
                with self.subTest(
                    attention_mode=attention_mode,
                    train_mode=train_mode,
                    dynamics=dynamics,
                ):
                    model = make_model(stage_b_dynamics=dynamics)
                    model.stage_b_attention_mode = attention_mode
                    with self.assertRaisesRegex(ValueError, "non-strict"):
                        FastLeWAMPolicy(
                            model=model,
                            sigreg=SIGReg(knots=5, num_proj=8),
                            action_horizon=3,
                            optimizer=None,
                            train_mode=train_mode,
                        )

    def test_serial_stage_b_training_uses_true_one_step_transitions(self):
        torch.manual_seed(23)
        module = FakeTrainingModule()
        module.model.stage_b_dynamics = "serial_one_step"
        batch = {
            "pixels": torch.randn(2, 4, 3, 8, 8),
            "action": torch.randn(2, 4, 4),
        }

        output = fast_lewam_forward(
            module,
            batch=batch,
            stage="fit",
            action_horizon=3,
            train_mode="stage_b",
            lambda_latent=1.0,
            lambda_sigreg=0.09,
            detach_clean_action=False,
            latent_loss_noise_threshold=0.2,
            latent_action_mix_epochs=10,
        )
        expected = torch.stack(
            [
                module.model.predict_one_step(
                    output["emb"][:, index], batch["action"][:, index]
                )
                for index in range(3)
            ],
            dim=1,
        )

        self.assertTrue(torch.equal(output["predicted_latents"], expected))

    def test_clean_action_timestep_mode_keeps_source_t_only_for_loss_weighting(self):
        torch.manual_seed(11)
        module = FakeTrainingModule(stage_a_goal_injection="token")
        module.model = RecordingStageBTimestepModel(module.model)
        batch = {
            "pixels": torch.randn(2, 4, 3, 8, 8),
            "action": torch.randn(2, 4, 4),
        }

        with patch(
            "source.policy.fast_lewam.torch.rand",
            side_effect=[
                torch.tensor([0.1, 0.8]),
                torch.tensor([0.0, 1.0]),
            ],
        ):
            output = fast_lewam_forward(
                module,
                batch=batch,
                stage="fit",
                action_horizon=3,
                train_mode="stage_ab",
                lambda_latent=0.4,
                lambda_sigreg=0.2,
                detach_clean_action=True,
                latent_loss_noise_threshold=1.0,
                latent_action_mix_epochs=10,
                stage_b_timestep_mode="clean_action",
            )

        self.assertTrue(
            torch.equal(module.model.stage_b_timesteps[-1], torch.ones(2))
        )
        self.assertAlmostEqual(output["noise_weight"].item(), 0.55, places=6)

    def test_legacy_timestep_mode_preserves_source_t_conditioning(self):
        torch.manual_seed(11)
        module = FakeTrainingModule(stage_a_goal_injection="token")
        module.model = RecordingStageBTimestepModel(module.model)
        batch = {
            "pixels": torch.randn(2, 4, 3, 8, 8),
            "action": torch.randn(2, 4, 4),
        }

        with patch(
            "source.policy.fast_lewam.torch.rand",
            side_effect=[
                torch.tensor([0.1, 0.8]),
                torch.tensor([0.0, 1.0]),
            ],
        ):
            output = fast_lewam_forward(
                module,
                batch=batch,
                stage="fit",
                action_horizon=3,
                train_mode="stage_ab",
                lambda_latent=0.4,
                lambda_sigreg=0.2,
                detach_clean_action=True,
                latent_loss_noise_threshold=1.0,
                latent_action_mix_epochs=10,
                stage_b_timestep_mode="legacy",
            )

        self.assertTrue(
            torch.equal(
                module.model.stage_b_timesteps[-1],
                torch.tensor([0.1, 1.0]),
            )
        )
        self.assertAlmostEqual(output["noise_weight"].item(), 0.55, places=6)

    def test_all_switch_combinations_run_h5_forward_and_backward(self):
        for stage_b_timestep_mode in ("legacy", "clean_action"):
            for token_encoding in ("legacy", "physical_time_type"):
                with self.subTest(
                    stage_b_timestep_mode=stage_b_timestep_mode,
                    token_encoding=token_encoding,
                ):
                    torch.manual_seed(17)
                    module = FakeTrainingModule(
                        stage_a_goal_injection="token",
                        action_horizon=5,
                        token_encoding=token_encoding,
                    )
                    batch = {
                        "pixels": torch.randn(2, 6, 3, 8, 8),
                        "action": torch.randn(2, 6, 4),
                    }

                    output = fast_lewam_forward(
                        module,
                        batch=batch,
                        stage="fit",
                        action_horizon=5,
                        train_mode="stage_ab",
                        lambda_latent=1.0,
                        lambda_sigreg=0.09,
                        detach_clean_action=True,
                        latent_loss_noise_threshold=0.2,
                        latent_action_mix_epochs=10,
                        stage_b_timestep_mode=stage_b_timestep_mode,
                    )
                    output["loss"].backward()

                    self.assertEqual(output["predicted_latents"].shape, (2, 5, 8))
                    self.assertTrue(output["loss"].isfinite())
                    self.assertIsNotNone(module.model.time_mlp[0].weight.grad)

    def test_stage_b_training_uses_only_latent_and_sigreg_losses(self):
        torch.manual_seed(7)
        module = FakeTrainingModule()
        batch = {
            "pixels": torch.randn(2, 4, 3, 8, 8),
            "action": torch.randn(2, 4, 4),
        }
        output = fast_lewam_forward(
            module,
            batch=batch,
            stage="fit",
            action_horizon=3,
            train_mode="stage_b",
            lambda_latent=0.4,
            lambda_sigreg=0.2,
            detach_clean_action=True,
            latent_loss_noise_threshold=0.25,
            latent_action_mix_epochs=0,
        )
        expected = 0.4 * output["latent_prefix_loss"] + 0.2 * output["sigreg_loss"]
        expected_latents = module.model(
            output["emb"][:, 0],
            batch["action"][:, :3],
            torch.ones(2),
            mode="stage_b",
        )["predicted_latents"]
        self.assertTrue(torch.allclose(output["loss"], expected))
        self.assertTrue(torch.allclose(output["predicted_latents"], expected_latents))
        self.assertEqual(output["predicted_latents"].shape, (2, 3, 8))
        self.assertEqual(output["noise_weight"].item(), 1.0)
        self.assertNotIn("action_loss", output)
        self.assertNotIn("action_velocity", output)
        self.assertNotIn("clean_action", output)
        self.assertNotIn("predicted_action_probability", output)
        self.assertNotIn("fit/action_loss", module.logged)

    def test_stage_ab_training_combines_flow_latent_and_sigreg_losses(self):
        torch.manual_seed(11)
        module = FakeTrainingModule(stage_a_goal_injection="token")
        batch = {
            "pixels": torch.randn(2, 4, 3, 8, 8),
            "action": torch.randn(2, 4, 4),
        }

        output = fast_lewam_forward(
            module,
            batch=batch,
            stage="fit",
            action_horizon=3,
            train_mode="stage_ab",
            lambda_latent=0.4,
            lambda_sigreg=0.2,
            detach_clean_action=False,
            latent_loss_noise_threshold=0.25,
            latent_action_mix_epochs=10,
        )

        expected = (
            output["action_loss"]
            + 0.4 * output["weighted_latent_prefix_loss"]
            + 0.2 * output["sigreg_loss"]
        )
        self.assertTrue(torch.allclose(output["loss"], expected))
        self.assertEqual(output["predicted_latents"].shape, (2, 3, 8))
        self.assertEqual(output["clean_action"].shape, (2, 3, 4))
        self.assertEqual(output["predicted_action_probability"].item(), 0.5)
        self.assertTrue(output["loss"].isfinite())
        self.assertIn("fit/weighted_latent_prefix_loss", module.logged)

    def test_clean_actions_receive_full_latent_loss_weight(self):
        torch.manual_seed(11)
        module = FakeTrainingModule()
        module.current_epoch = 0
        batch = {
            "pixels": torch.randn(2, 4, 3, 8, 8),
            "action": torch.randn(2, 4, 4),
        }

        with patch(
            "source.policy.fast_lewam.torch.rand",
            return_value=torch.tensor([0.05, 0.10]),
        ):
            output = fast_lewam_forward(
                module,
                batch=batch,
                stage="fit",
                action_horizon=3,
                train_mode="stage_ab",
                lambda_latent=0.4,
                lambda_sigreg=0.2,
                detach_clean_action=False,
                latent_loss_noise_threshold=0.25,
                latent_action_mix_epochs=10,
            )

        self.assertEqual(output["predicted_action_probability"].item(), 0.0)
        self.assertEqual(output["noise_weight"].item(), 1.0)
        self.assertTrue(
            torch.allclose(
                output["weighted_latent_prefix_loss"],
                output["latent_prefix_loss"],
            )
        )

if __name__ == "__main__":
    unittest.main()
