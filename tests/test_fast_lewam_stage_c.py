import unittest

import torch

from tests.test_fast_lewam_policy import FakeTrainingModule
from source.policy.fast_lewam import fast_lewam_forward


class FastLeWAMStageCTrainingTests(unittest.TestCase):
    def test_stage_c_joint_training_has_finite_action_and_latent_losses(self):
        torch.manual_seed(17)
        module = FakeTrainingModule()
        output = fast_lewam_forward(
            module,
            batch={
                "pixels": torch.randn(2, 4, 3, 8, 8),
                "action": torch.randn(2, 4, 4),
            },
            stage="fit",
            action_horizon=3,
            train_mode="stage_c",
            lambda_latent=1.0,
            lambda_sigreg=0.1,
            detach_clean_action=False,
            latent_loss_noise_threshold=0.2,
            latent_action_mix_epochs=10,
        )

        output["loss"].backward()
        self.assertTrue(output["action_loss"].isfinite())
        self.assertTrue(output["latent_prefix_loss"].isfinite())
        self.assertIsNotNone(module.model.action_head.weight.grad)


if __name__ == "__main__":
    unittest.main()
