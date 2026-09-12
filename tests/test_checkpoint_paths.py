import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from source.common.checkpoint import (
    CHECKPOINTS_DIRNAME,
    LAST_CHECKPOINT_FILENAME,
    _leflow_checkpoint_candidates,
    get_policy_eval_paths,
    get_policy_results_path,
    load_policy_or_model,
    policy_checkpoint_candidates,
    resolve_training_resume_checkpoint,
    training_resume_checkpoint_candidates,
)


class CheckpointPathTests(unittest.TestCase):
    def test_loads_relative_epoch_weights_from_saved_training_config(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
            run_dir = Path(tmpdir)
            checkpoint_dir = run_dir / CHECKPOINTS_DIRNAME
            checkpoint_dir.mkdir()
            checkpoint = checkpoint_dir / "lewm_weights_epoch_10.pt"
            expected = torch.nn.Linear(3, 2)
            with torch.no_grad():
                expected.weight.fill_(1.25)
                expected.bias.fill_(-0.5)
            torch.save(expected.state_dict(), checkpoint)
            OmegaConf.save(
                OmegaConf.create(
                    {
                        "policy": {
                            "model": {
                                "_target_": "torch.nn.Linear",
                                "in_features": 3,
                                "out_features": 2,
                            }
                        }
                    }
                ),
                run_dir / "config.yaml",
            )

            relative_checkpoint = checkpoint.relative_to(Path.cwd())
            loaded, resolved_checkpoint = load_policy_or_model(relative_checkpoint)

        self.assertTrue(torch.equal(loaded.weight, expected.weight))
        self.assertTrue(torch.equal(loaded.bias, expected.bias))
        self.assertEqual(resolved_checkpoint, checkpoint.resolve())

    def test_loads_cache_relative_epoch_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            run_dir = cache_dir / "run"
            checkpoint_dir = run_dir / CHECKPOINTS_DIRNAME
            checkpoint_dir.mkdir(parents=True)
            checkpoint = checkpoint_dir / "lewm_weights_epoch_2.pt"
            expected = torch.nn.Linear(2, 1)
            torch.save(expected.state_dict(), checkpoint)
            OmegaConf.save(
                OmegaConf.create(
                    {
                        "policy": {
                            "model": {
                                "_target_": "torch.nn.Linear",
                                "in_features": 2,
                                "out_features": 1,
                            }
                        }
                    }
                ),
                run_dir / "config.yaml",
            )

            loaded, resolved_checkpoint = load_policy_or_model(
                "run/checkpoints/lewm_weights_epoch_2.pt",
                cache_dir=cache_dir,
            )

        self.assertTrue(torch.equal(loaded.weight, expected.weight))
        self.assertEqual(resolved_checkpoint, checkpoint)

    def test_leflow_candidates_prefer_direct_relative_path(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
            checkpoint = Path(tmpdir) / "latent_planner.pt"
            checkpoint.write_bytes(b"fixture")
            candidates = _leflow_checkpoint_candidates(
                checkpoint.relative_to(Path.cwd()), cache_dir=Path("/cache")
            )

        self.assertEqual(candidates[0], checkpoint.relative_to(Path.cwd()))

    def test_policy_prefix_checks_old_and_nested_checkpoint_layouts(self):
        run_dir = Path("/tmp/lewm_run")

        candidates = policy_checkpoint_candidates(run_dir / "value_jepa")

        self.assertEqual(candidates[0], run_dir / "value_jepa_policy.ckpt")
        self.assertEqual(candidates[1], run_dir / "checkpoints" / "value_jepa_policy.ckpt")
        self.assertIn(run_dir / "value_jepa_object.ckpt", candidates)
        self.assertIn(run_dir / "checkpoints" / "value_jepa_object.ckpt", candidates)

    def test_policy_ckpt_path_can_resolve_nested_checkpoint_candidate(self):
        run_dir = Path("/tmp/lewm_run")

        candidates = policy_checkpoint_candidates(run_dir / "value_jepa_policy.ckpt")

        self.assertEqual(candidates[0], run_dir / "value_jepa_policy.ckpt")
        self.assertEqual(candidates[1], run_dir / "checkpoints" / "value_jepa_policy.ckpt")
        self.assertIn(run_dir / "value_jepa_object.ckpt", candidates)

    def test_object_ckpt_path_can_resolve_nested_checkpoint_candidate(self):
        run_dir = Path("/tmp/lewm_run")

        candidates = policy_checkpoint_candidates(run_dir / "value_jepa_object.ckpt")

        self.assertEqual(candidates[0], run_dir / "value_jepa_object.ckpt")
        self.assertEqual(candidates[1], run_dir / "checkpoints" / "value_jepa_object.ckpt")
        self.assertIn(run_dir / "value_jepa_policy.ckpt", candidates)

    def test_training_resume_candidates_prefer_last_checkpoint(self):
        run_dir = Path("/tmp/lewm_run")

        candidates = training_resume_checkpoint_candidates(run_dir, "value_jepa")

        self.assertEqual(
            candidates[0],
            run_dir / CHECKPOINTS_DIRNAME / LAST_CHECKPOINT_FILENAME,
        )
        self.assertEqual(
            candidates[1],
            run_dir / CHECKPOINTS_DIRNAME / "value_jepa_weights.ckpt",
        )
        self.assertEqual(candidates[2], run_dir / "value_jepa_weights.ckpt")

    def test_training_resume_resolves_existing_last_checkpoint_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            checkpoint_dir = run_dir / CHECKPOINTS_DIRNAME
            checkpoint_dir.mkdir()
            latest = checkpoint_dir / LAST_CHECKPOINT_FILENAME
            legacy = checkpoint_dir / "value_jepa_weights.ckpt"
            legacy.write_text("legacy")
            latest.write_text("latest")

            self.assertEqual(
                resolve_training_resume_checkpoint(None, run_dir, "value_jepa"),
                str(latest),
            )

    def test_training_resume_respects_explicit_checkpoint(self):
        self.assertEqual(
            resolve_training_resume_checkpoint("/tmp/manual.ckpt", "/tmp/run", "lewm"),
            "/tmp/manual.ckpt",
        )

    def test_eval_paths_live_next_to_run_when_checkpoint_is_nested(self):
        run_dir = Path("/tmp/lewm_run")
        ckpt_path = run_dir / "checkpoints" / "value_jepa_policy.ckpt"

        self.assertEqual(get_policy_results_path("ignored", ckpt_path=ckpt_path), run_dir)
        self.assertEqual(
            get_policy_results_path(run_dir / "checkpoints" / "value_jepa"),
            run_dir,
        )
        self.assertEqual(
            get_policy_eval_paths("ignored", ckpt_path=ckpt_path),
            (run_dir / "eval", run_dir / "eval" / "videos"),
        )


if __name__ == "__main__":
    unittest.main()
