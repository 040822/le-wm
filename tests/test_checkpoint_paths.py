import tempfile
import unittest
from pathlib import Path

from source.common.checkpoint import (
    CHECKPOINTS_DIRNAME,
    LAST_CHECKPOINT_FILENAME,
    get_policy_eval_paths,
    get_policy_results_path,
    policy_checkpoint_candidates,
    resolve_training_resume_checkpoint,
    training_resume_checkpoint_candidates,
)


class CheckpointPathTests(unittest.TestCase):
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
