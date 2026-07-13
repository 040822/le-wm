import unittest
from pathlib import Path

import stable_pretraining  # noqa: F401 - registers OmegaConf's eval resolver
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from source.common.checkpoint import load_policy_or_model
from source.common.data import load_dataset
from source.model.fast_lewam.jepa import FastLeWAM
from source.model.fast_lewam.modules import SIGReg
from source.policy.fast_lewam import fast_lewam_forward
from tests.test_fast_lewam_model import TinyEncoder


ROOT = Path(__file__).resolve().parents[1]
PUSHT_PATH = ROOT / "data" / "datasets" / "pusht.h5"
LEGACY_CHECKPOINT = ROOT / "data" / "checkpoints" / "quetinll" / "lewm-pusht" / "weights.pt"


class FakeTrainingModule:
    def __init__(self, model, current_epoch=0):
        self.model = model
        self.sigreg = SIGReg(knots=5, num_proj=8)
        self.current_epoch = current_epoch

    def log_dict(self, metrics, **kwargs):
        pass


def make_pusht_model():
    return FastLeWAM(
        encoder=TinyEncoder(latent_dim=8),
        projector=torch.nn.Identity(),
        latent_dim=8,
        action_dim=10,
        action_horizon=5,
        latent_head_dim=12,
        latent_head_layers=1,
        heads=3,
        mlp_dim=24,
        dropout=0.0,
        inference_steps=2,
    )


class FastLeWAMIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(PUSHT_PATH.exists(), "local PushT dataset is unavailable")
    def test_real_pusht_window_runs_training_forward_and_backward(self):
        dataset = load_dataset(
            "pusht.h5",
            cache_dir=str(ROOT / "data"),
            num_steps=6,
            frameskip=5,
            keys_to_load=["pixels", "action"],
        )
        sample = dataset[0]
        module = FakeTrainingModule(make_pusht_model())
        output = fast_lewam_forward(
            module,
            batch={key: value.unsqueeze(0) for key, value in sample.items()},
            stage="fit",
            action_horizon=5,
            train_mode="stage_ab",
            lambda_latent=1.0,
            lambda_sigreg=0.09,
            detach_clean_action=False,
            latent_loss_noise_threshold=0.2,
            latent_action_mix_epochs=20,
        )
        output["loss"].backward()

        self.assertEqual(tuple(sample["pixels"].shape), (6, 3, 224, 224))
        self.assertEqual(tuple(sample["action"].shape), (6, 10))
        self.assertTrue(output["loss"].isfinite())

    def test_hydra_instantiates_fast_and_legacy_policy_configs(self):
        with initialize_config_dir(
            config_dir=str(ROOT / "config" / "train"), version_base=None
        ):
            fast = compose(
                config_name="fast_lewam",
                overrides=["data=pusht", "policy.model.action_dim=10"],
            )
            legacy = compose(
                config_name="lewm",
                overrides=["data=pusht", "policy.model.action_encoder.input_dim=10"],
            )
            fast_policy = instantiate(fast.policy)
            legacy_policy = instantiate(legacy.policy)

        self.assertEqual(fast.data.dataset.num_steps, 6)
        self.assertEqual(fast_policy.model.action_horizon, 5)
        self.assertEqual(legacy_policy.model.action_encoder.patch_embed.in_channels, 10)

    @unittest.skipUnless(
        LEGACY_CHECKPOINT.exists(), "local legacy LeWM checkpoint is unavailable"
    )
    def test_legacy_lewm_checkpoint_still_loads(self):
        model, path = load_policy_or_model("quetinll/lewm-pusht/weights.pt")
        self.assertEqual(type(model).__name__, "JEPA")
        self.assertEqual(path, None)
        self.assertGreater(sum(parameter.numel() for parameter in model.parameters()), 0)

    def test_epoch_zero_latent_loss_updates_shared_stage_a_blocks(self):
        torch.manual_seed(23)
        module = FakeTrainingModule(make_pusht_model(), current_epoch=0)
        output = fast_lewam_forward(
            module,
            batch={
                "pixels": torch.randn(2, 6, 3, 8, 8),
                "action": torch.randn(2, 6, 10),
            },
            stage="fit",
            action_horizon=5,
            train_mode="stage_ab",
            lambda_latent=1.0,
            lambda_sigreg=0.0,
            detach_clean_action=False,
            latent_loss_noise_threshold=0.2,
            latent_action_mix_epochs=20,
        )
        output["weighted_latent_prefix_loss"].backward()
        grad = module.model.predictor.layers[0].attention.to_qkv.weight.grad

        self.assertEqual(output["predicted_action_probability"].item(), 0.0)
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
