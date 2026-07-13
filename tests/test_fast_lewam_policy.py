import unittest

import torch

from source.model.fast_lewam.modules import SIGReg
from source.policy.fast_lewam import fast_lewam_forward
from tests.test_fast_lewam_model import make_model


class FakeTrainingModule:
    def __init__(self):
        self.model = make_model()
        self.sigreg = SIGReg(knots=5, num_proj=8)
        self.current_epoch = 5
        self.logged = None

    def log_dict(self, metrics, **kwargs):
        self.logged = metrics


class FastLeWAMPolicyTests(unittest.TestCase):
    def test_stage_ab_training_combines_flow_latent_and_sigreg_losses(self):
        torch.manual_seed(11)
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


if __name__ == "__main__":
    unittest.main()
