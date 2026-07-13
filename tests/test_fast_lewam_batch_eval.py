import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from eval_fast_lewam import (
    discover_weight_checkpoints,
    load_model_from_weights,
    resolve_eval_dataset_name,
)


class FastLeWAMBatchEvalTests(unittest.TestCase):
    def test_discovers_numeric_epochs_and_supports_filtering(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint_dir = Path(root)
            for epoch in (10, 2, 1):
                (checkpoint_dir / f"fast_lewam_weights_epoch_{epoch}.pt").touch()
            (checkpoint_dir / "unrelated.pt").touch()

            discovered = discover_weight_checkpoints(checkpoint_dir)
            selected = discover_weight_checkpoints(checkpoint_dir, epochs=[10, 1])

        self.assertEqual([item.epoch for item in discovered], [1, 2, 10])
        self.assertEqual([item.epoch for item in selected], [1, 10])

    def test_missing_requested_epoch_is_reported(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint_dir = Path(root)
            (checkpoint_dir / "fast_lewam_weights_epoch_1.pt").touch()

            with self.assertRaisesRegex(FileNotFoundError, "epoch.*2"):
                discover_weight_checkpoints(checkpoint_dir, epochs=[2])

    def test_rebuilds_model_from_run_config_and_loads_weights_strictly(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            checkpoint = root / "weights.pt"
            expected = torch.nn.Linear(3, 2)
            with torch.no_grad():
                expected.weight.fill_(1.25)
                expected.bias.fill_(-0.5)
            torch.save(expected.state_dict(), checkpoint)
            cfg = OmegaConf.create(
                {
                    "policy": {
                        "model": {
                            "_target_": "torch.nn.Linear",
                            "in_features": 3,
                            "out_features": 2,
                        }
                    }
                }
            )

            loaded = load_model_from_weights(cfg, checkpoint, device="cpu")

        self.assertFalse(loaded.training)
        self.assertTrue(torch.equal(loaded.weight, expected.weight))
        self.assertTrue(torch.equal(loaded.bias, expected.bias))

    def test_eval_dataset_defaults_to_training_run_and_allows_override(self):
        cfg = OmegaConf.create(
            {"data": {"dataset": {"name": "ogbench/cube_single.h5"}}}
        )

        inferred = resolve_eval_dataset_name(cfg)
        overridden = resolve_eval_dataset_name(cfg, "ogbench/cube_single_expert")

        self.assertEqual(inferred, "ogbench/cube_single")
        self.assertEqual(overridden, "ogbench/cube_single_expert")


if __name__ == "__main__":
    unittest.main()
