import json
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from scripts.round4 import build_parser
from source.common.round4_artifacts import (
    sha256_file,
    validate_artifact_manifest,
    validate_leflow_manifest,
)
from source.common.round4_eval import _TimedSolver, validate_gpu_visibility
from source.common.round4_protocol import mode_spec, should_expand_seed


class Round4ProtocolTests(unittest.TestCase):
    def test_cem_solver_timing_records_planning_and_forward_counts(self):
        class FakeSolver:
            batch_size = 1
            n_steps = 30
            device = "cpu"

            def solve(self, info_dict, init_action=None):
                return {"actions": []}

        timed = _TimedSolver(FakeSolver())
        self.assertEqual(timed.solve({"pixels": [0, 1]})["actions"], [])
        self.assertEqual(len(timed.events), 1)
        self.assertEqual(timed.events[0]["forward_count"], 60)
        self.assertIsNone(timed.events[0]["peak_memory_bytes"])
        self.assertGreaterEqual(timed.events[0]["planning_seconds"], 0.0)

    def test_cem_timing_preserves_action_bound_projection_diagnostics(self):
        projection = {"mode": "clip", "projected_candidate_violation_fraction": 0.0}

        class FakeModel:
            last_action_bound_projection = projection

        class FakeSolver:
            batch_size = 1
            n_steps = 1
            device = "cpu"
            model = FakeModel()

            def solve(self, info_dict, init_action=None):
                del info_dict, init_action
                return {"actions": []}

        timed = _TimedSolver(FakeSolver())
        timed.solve({"pixels": [0]})
        self.assertEqual(timed.events[0]["action_bound_projection"], projection)

    def test_cuda_evaluation_binds_one_permitted_gpu_to_mujoco_egl(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}, clear=True):
            validate_gpu_visibility("cuda")
            self.assertEqual(os.environ["MUJOCO_EGL_DEVICE_ID"], "2")
            self.assertEqual(os.environ["MUJOCO_GL"], "egl")
            self.assertEqual(os.environ["PYOPENGL_PLATFORM"], "egl")
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}, clear=True):
            with self.assertRaisesRegex(EnvironmentError, "exactly one"):
                validate_gpu_visibility("cuda")

    def test_cli_exposes_independent_modes_and_requires_explicit_gpu_for_training(self):
        parsed = build_parser().parse_args(["evaluate", "cube", "P4", "--cohort", "c", "--checkpoint", "p"])
        self.assertEqual(parsed.mode, "P4")
        self.assertIsNone(parsed.action_flow_steps)
        parsed_steps = build_parser().parse_args(
            [
                "evaluate",
                "cube",
                "P3",
                "--cohort",
                "c",
                "--checkpoint",
                "p",
                "--action-flow-steps",
                "2",
            ]
        )
        self.assertEqual(parsed_steps.action_flow_steps, 2)
        train = build_parser().parse_args(["train", "cube", "--gpu", "2", "--dry-run"])
        self.assertEqual(train.gpu, "2")
        ab_train = build_parser().parse_args(
            ["train", "cube", "--gpu", "2", "--config-name", "round4_ab", "--dry-run"]
        )
        self.assertEqual(ab_train.config_name, "round4_ab")
        continuation = build_parser().parse_args(
            [
                "train",
                "cube",
                "--gpu",
                "0",
                "--init-weights",
                "/tmp/epoch_9.pt",
                "--initial-epoch",
                "9",
                "--run-dir",
                "/tmp/r4-cube",
                "--disable-wandb",
            ]
        )
        self.assertEqual(continuation.initial_epoch, 9)
        self.assertTrue(continuation.disable_wandb)

    def test_expansion_gate_requires_two_five_point_tasks_and_no_large_regression(self):
        rows = {
            task: {
                "P2": {"success_rate": 0.5, "planning_median_seconds": 10.0},
                "P3": {"success_rate": 0.5},
                "P4": {"success_rate": 0.55, "planning_median_seconds": 9.0},
            }
            for task in ("cube", "pusht", "reacher", "tworoom")
        }
        rows["cube"]["P4"]["success_rate"] = 0.60
        expand, details = should_expand_seed(rows)
        self.assertTrue(expand)
        self.assertTrue(details["performance_gate"])

    def test_artifact_manifest_checks_sha256_and_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            planner = root / "planner.pt"
            lewm = root / "lewm.pt"
            config = root / "config.yaml"
            planner.write_bytes(b"planner")
            lewm.write_bytes(b"lewm")
            config.write_text("horizon: 5\n", encoding="utf-8")
            payload = {
                "root": str(root),
                "tasks": {
                    "cube": {
                        "planner_checkpoint": planner.name,
                        "planner_checkpoint_sha256": sha256_file(planner),
                        "lewm_checkpoint": lewm.name,
                        "lewm_checkpoint_sha256": sha256_file(lewm),
                        "config": config.name,
                        "config_sha256": sha256_file(config),
                    }
                }
            }
            result = validate_artifact_manifest(payload, require_files=True)
            self.assertEqual(result["status"], "accepted")
            payload["tasks"]["cube"]["config_sha256"] = "bad"
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                validate_artifact_manifest(payload)

    def test_checked_in_leflow_manifest_accepts_all_four_local_payloads(self):
        result = validate_leflow_manifest(Path("config/round4/leflow_artifacts.json"))
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(set(result["tasks"]), {"cube", "pusht", "reacher", "tworoom"})


if __name__ == "__main__":
    unittest.main()
