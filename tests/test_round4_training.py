import unittest

import torch

from source.model.fast_lewam.modules import SIGReg
from source.common.round4_diagnostics import compute_idm_diagnostics
from source.policy.fast_lewam import FastLeWAMPolicy, fast_lewam_forward
from tests.test_round4_model import make_round4_model


class FakeTrainingModule:
    def __init__(self):
        self.model = make_round4_model()
        self.sigreg = SIGReg(knots=5, num_proj=8)
        self.current_epoch = 10
        self.logged = {}

    def log_dict(self, metrics, **kwargs):
        self.logged.update(metrics)


class Round4TrainingTests(unittest.TestCase):
    def test_abde_forward_uses_one_encoder_call_and_returns_all_losses(self):
        module = FakeTrainingModule()
        calls = []
        module.model.encoder.register_forward_hook(lambda *_: calls.append(True))
        batch = {
            "pixels": torch.randn(2, 6, 3, 8, 8),
            "action": torch.randn(2, 5, 4),
        }

        output = fast_lewam_forward(
            module,
            batch=batch,
            stage="fit",
            action_horizon=5,
            train_mode="stage_abde",
            lambda_latent=1.0,
            lambda_sigreg=0.09,
            lambda_d=1.0,
            lambda_e=1.0,
            detach_clean_action=False,
            latent_loss_noise_threshold=0.2,
            latent_action_mix_epochs=10,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(output["predicted_latents"].shape, (2, 5, 8))
        self.assertEqual(output["action_velocity"].shape, (2, 5, 4))
        self.assertEqual(output["d_loss"].ndim, 0)
        self.assertEqual(output["e_loss"].ndim, 0)
        self.assertIn("fit/d_loss", module.logged)
        self.assertIn("fit/e_action_mse", module.logged)
        output["loss"].backward()
        self.assertTrue(output["loss"].isfinite())
        self.assertIsNotNone(module.model.d_velocity_head.weight.grad)
        self.assertIsNotNone(module.model.inverse_dynamics_head.weight.grad)

    def test_policy_accepts_stage_abde_only_with_round4_model(self):
        policy = FastLeWAMPolicy(
            model=make_round4_model(),
            sigreg=SIGReg(knots=5, num_proj=8),
            action_horizon=5,
            optimizer=None,
            train_mode="stage_abde",
            lambda_d=1.0,
            lambda_e=1.0,
        )
        self.assertEqual(policy.train_mode, "stage_abde")
        self.assertEqual(policy.lambda_d, 1.0)
        self.assertEqual(policy.lambda_e, 1.0)

    def test_legacy_model_cannot_be_used_for_stage_abde(self):
        from tests.test_fast_lewam_model import make_model

        with self.assertRaisesRegex(TypeError, "Round4FastLeWAM"):
            FastLeWAMPolicy(
                model=make_model(),
                sigreg=SIGReg(knots=5, num_proj=8),
                action_horizon=3,
                optimizer=None,
                train_mode="stage_abde",
            )

    def test_offline_idm_diagnostics_keep_real_and_generated_path_errors_separate(self):
        model = make_round4_model().eval()
        real_paths = torch.randn(2, 6, 8)
        real_actions = torch.randn(2, 5, 4)
        generated = real_paths.clone()
        diagnostics = compute_idm_diagnostics(
            model,
            real_paths,
            real_actions,
            z_start=real_paths[:, 0],
            z_goal=real_paths[:, -1],
            generated_paths=generated,
        )
        self.assertIn("real_path_idm_action_mse", diagnostics)
        self.assertIn("decoded_action_b_verifier_mse", diagnostics)
        self.assertIn("generated_path_b_validation_mse", diagnostics)
        self.assertEqual(diagnostics["generated_endpoint_mse"], 0.0)


if __name__ == "__main__":
    unittest.main()
