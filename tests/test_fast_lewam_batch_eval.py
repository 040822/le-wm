import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from omegaconf import OmegaConf

from eval_fast_lewam import (
    FAST_LEWAM_STAGES,
    discover_weight_checkpoints,
    load_model_from_weights,
    parse_cli,
    resolve_eval_config_name,
    run_evaluation,
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

    def test_eval_task_uses_saved_hydra_choice_and_allows_explicit_override(self):
        cfg = OmegaConf.create({"data": {"dataset": {"name": "pusht.h5"}}})
        with tempfile.TemporaryDirectory() as root:
            hydra_dir = Path(root) / ".hydra"
            hydra_dir.mkdir()
            OmegaConf.save(
                OmegaConf.create(
                    {"hydra": {"runtime": {"choices": {"data": "reacher"}}}}
                ),
                hydra_dir / "hydra.yaml",
            )
            inferred = resolve_eval_config_name(Path(root), cfg)
            overridden = resolve_eval_config_name(Path(root), cfg, "cube")

        self.assertEqual(inferred, "reacher")
        self.assertEqual(overridden, "cube")

    def test_cli_keeps_batch_arguments_and_forwards_eval_dotlist(self):
        args, overrides = parse_cli(
            [
                "checkpoints",
                "--epochs",
                "5",
                "10",
                "seed=7",
                "solver.n_steps=12",
            ]
        )

        self.assertEqual(args.epochs, [5, 10])
        self.assertEqual(tuple(args.stages), FAST_LEWAM_STAGES)
        self.assertEqual(overrides, ("seed=7", "solver.n_steps=12"))

    def test_cli_allows_explicit_actor_warm_start_without_changing_defaults(self):
        args, overrides = parse_cli(
            [
                "checkpoints",
                "--stages",
                "stage_b_actor_warm_start",
            ]
        )

        self.assertEqual(args.stages, ["stage_b_actor_warm_start"])
        self.assertNotIn("stage_b_actor_warm_start", FAST_LEWAM_STAGES)
        self.assertEqual(overrides, ())

    def test_setup_failure_writes_machine_and_human_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint_dir = Path(root) / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "fast_lewam_weights_epoch_1.pt").touch()
            args = Namespace(
                checkpoint_dir=str(checkpoint_dir),
                epochs=None,
                run_config=None,
                config_name=None,
                stages=FAST_LEWAM_STAGES,
            )

            with self.assertRaisesRegex(FileNotFoundError, "run config"):
                run_evaluation(args)

            failure_dir = Path(root) / "eval" / "setup"
            self.assertTrue((failure_dir / "failure.json").is_file())
            self.assertTrue((failure_dir / "failure.txt").is_file())

if __name__ == "__main__":
    unittest.main()
