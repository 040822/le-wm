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
from source.common.eval import compose_eval_config
from source.common.round4_eval import (
    _TimedSolver,
    validate_gpu_visibility,
    validate_round4_config,
)
from source.common.round4_protocol import (
    ROUND4_DEFAULTS,
    mode_spec,
    resolve_cem_protocol,
    should_expand_seed,
)


class Round4ProtocolTests(unittest.TestCase):
    def test_native_action_structure_override_keeps_other_protocol_fields_frozen(self):
        cfg = compose_eval_config(
            "pusht",
            (
                "output.save_video=false",
                "plan_config.horizon=25",
                "plan_config.receding_horizon=25",
                "plan_config.action_block=1",
            ),
        )
        with self.assertRaisesRegex(ValueError, "plan_config.horizon"):
            validate_round4_config(cfg, "P3")
        validate_round4_config(
            cfg, "P3", allow_action_structure_override=True
        )
        cfg.plan_config.receding_horizon = 26
        with self.assertRaisesRegex(ValueError, "receding_horizon"):
            validate_round4_config(
                cfg, "P3", allow_action_structure_override=True
            )

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

    def test_cem_timing_includes_actor_warm_start_and_guidance_counts(self):
        class WarmStartModel:
            last_warm_start_stats = {"stale": True}

        class FakeSolver:
            batch_size = 2
            n_steps = 3
            device = "cpu"
            model = WarmStartModel()

            def solve(self, info_dict, init_action=None):
                del init_action
                self.model.last_warm_start_stats = {
                    "stage_a_forward_count": 4,
                    "stage_b_forward_count": 2,
                    "backward_count": 2,
                }
                return {"actions": [len(info_dict["pixels"])]}

        solver = FakeSolver()
        timed = _TimedSolver(solver)
        timed.solve({"pixels": [0, 1, 2]})
        event = timed.events[0]
        self.assertEqual(event["stage_a_forward_count"], 4)
        self.assertEqual(event["stage_b_forward_count"], 8)
        self.assertEqual(event["guidance_backward_count"], 2)
        self.assertEqual(event["forward_count"], 12)
        self.assertTrue(event["forward_count_is_exact"])

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

    def test_best_of_n_solver_batch_size_is_frozen_at_the_dev_world_size(self):
        self.assertEqual(ROUND4_DEFAULTS["best_of_n"]["solver_batch_size"], 50)
        parsed = build_parser().parse_args(
            ["evaluate", "cube", "P3", "--cohort", "c", "--checkpoint", "p"]
        )
        self.assertIsNone(parsed.solver_batch_size)

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
        parsed_protocol = build_parser().parse_args(
            [
                "evaluate",
                "cube",
                "P2",
                "--cohort",
                "c",
                "--checkpoint",
                "p",
                "--cem-protocol",
                "cem-scale",
            ]
        )
        self.assertEqual(parsed_protocol.cem_protocol, "cem-scale")
        self.assertEqual(
            resolve_cem_protocol(
                parsed_protocol.mode,
                parsed_protocol.cem_protocol,
                parsed_protocol.action_bound_mode,
            ),
            ("cem-scale", "candidate_scale"),
        )
        self.assertEqual(
            resolve_cem_protocol("P0", "not_applicable", "none"),
            ("not_applicable", "none"),
        )
        with self.assertRaises(ValueError):
            resolve_cem_protocol("P1", "cem-clip", "warm_start_clip")
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
