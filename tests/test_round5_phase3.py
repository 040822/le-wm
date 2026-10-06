import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from scripts.round5_phase3 import (
    _attach_result_metadata,
    _dataset_keys,
    _read_result,
    _render_report,
    _successes,
    _validate_training_outputs,
    _validate_video_file,
    _validate_video_result,
    fast_conditions,
    p2_protocol,
)
from source.common.eval import select_eval_cohort
from source.common.phase3_compat import (
    build_phase3_legacy_manifest,
    patch_phase3_environments,
    set_phase3_scene_goal,
)


class _TinyDataset:
    column_names = ["ep_idx", "step_idx"]

    def __init__(self):
        self._episodes = np.repeat(np.arange(4), 12)
        self._steps = np.tile(np.arange(12), 4)

    def get_col_data(self, name):
        if name == "ep_idx":
            return self._episodes
        if name == "step_idx":
            return self._steps
        raise KeyError(name)

    def get_row_data(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        return {"ep_idx": self._episodes[rows], "step_idx": self._steps[rows]}


class Round5Phase3Tests(unittest.TestCase):
    def test_report_discloses_resume_metadata_for_each_lewm_task(self):
        rows = []
        for task in ("scene", "finger", "humanoid"):
            rows.extend(
                [
                    {
                        "task": task,
                        "method": "fast_lewam",
                        "condition": "P3/not_applicable/none/step_1/euler",
                        "successes": 1,
                        "success_rate_percent": 2.0,
                        "wilson_95_percent_low": 0.1,
                        "wilson_95_percent_high": 10.0,
                        "training_config_path": f"/training/fastlewam/{task}/config.yaml",
                        "training_config_sha256": "a" * 64,
                        "training_seed": 3072,
                        "evaluation_seed": 42,
                    },
                    {
                        "task": task,
                        "method": "lewm",
                        "condition": "standard",
                        "successes": 2,
                        "success_rate_percent": 4.0,
                        "wilson_95_percent_low": 0.5,
                        "wilson_95_percent_high": 13.0,
                        "training_config_path": f"/training/lewm/{task}/config.yaml",
                        "training_config_sha256": "b" * 64,
                        "training_seed": 3072,
                        "evaluation_seed": 42,
                    },
                    {
                        "task": task,
                        "method": "leflow",
                        "condition": "standard",
                        "successes": 3,
                        "success_rate_percent": 6.0,
                        "wilson_95_percent_low": 1.0,
                        "wilson_95_percent_high": 16.0,
                        "training_config_path": f"/training/leflow/{task}/config.json",
                        "training_config_sha256": "c" * 64,
                        "training_seed": 3072,
                        "evaluation_seed": 42,
                    },
                ]
            )
        analysis = {
            "rows": rows,
            "primary_comparisons": [
                {
                    "task": task,
                    "baseline": method,
                    "delta_pp": 0.0,
                    "improved": 0,
                    "regressed": 0,
                    "mcnemar_exact_two_sided_p": 1.0,
                }
                for task in ("scene", "finger", "humanoid")
                for method in ("lewm", "leflow")
            ],
            "flow_step_sensitivity": [],
            "guidance_deltas": [],
            "result_count": 189,
            "expected_result_count": 189,
            "training_outputs": [
                {
                    "task": "scene",
                    "method": "fastlewam",
                    "epoch": 10,
                    "checkpoint": "/checkpoints/r4_ab_weights_epoch_10.pt",
                }
            ],
            "videos": [
                {
                    "task": "scene",
                    "method": "fastlewam",
                    "condition": "p3_step_1",
                    "path": "/videos/scene_p3.mp4",
                    "size_bytes": 12345,
                    "width": 736,
                    "height": 288,
                    "source_cohort_sha256": "d" * 64,
                }
            ],
            "scene_target_task_groups": [],
        }

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            records = {
                "scene": {"source_run": "scene-resume", "epoch": 10},
                "finger": {
                    "source_run": "finger-resume",
                    "initial_epoch": 9,
                    "optimizer_state_restored": False,
                },
                "humanoid": {
                    "source_run": "humanoid-resume",
                    "initial_epoch": 6,
                    "completed_epoch": 10,
                    "seed": 3072,
                    "optimizer_state_restored": False,
                    "resume_method": "model_weights_with_epoch_offset",
                },
            }
            for task, record in records.items():
                path = output_root / "training" / "lewm" / task / "resume_metadata.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record), encoding="utf-8")
            logs = output_root / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            (logs / "phase3_environment_smoke_cpu.log").write_text(
                '{"task":"scene","status":"ok"}\n', encoding="utf-8"
            )
            (logs / "phase3_unit_tests.log").write_text(
                "Ran 8 tests\nOK\n", encoding="utf-8"
            )

            report = _render_report(analysis, output_root / "report.md", output_root)
            (logs / "pre_evaluation_gate.json").write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "unit_tests": {"status": "ok", "count": 13, "log": "logs/phase3_unit_tests.log"},
                        "environment_smoke": {"status": "ok", "log": "logs/phase3_environment_smoke_cpu.log"},
                        "end_to_end_smoke": {
                            "status": "ok",
                            "method": "fastlewam",
                            "condition": "p3_step_1",
                            "episodes_per_task": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            gated_report = _render_report(analysis, output_root / "report.md", output_root)

        for source in ("scene-resume", "finger-resume", "humanoid-resume"):
            self.assertIn(source, report)
        self.assertIn("optimizer state restored=false", report)
        self.assertIn("批量实验启动后补做", report)
        self.assertIn("phase3_environment_smoke_cpu.log", report)
        self.assertIn("phase3_unit_tests.log", report)
        self.assertIn("## 训练配置与种子", report)
        self.assertIn("训练 seed 3072、评测 seed 42", report)
        self.assertIn("## Epoch 10 checkpoint 验收", report)
        self.assertIn("scene / fastlewam", report)
        self.assertIn("共检查 1 段视频", report)
        self.assertIn("736×288", report)
        self.assertIn("本报告对应的完整复评矩阵", gated_report)
        self.assertIn("单测：ok（13 项）", gated_report)
        self.assertIn("Scene、Finger、Humanoid 各 1 episode", gated_report)

    def test_result_metadata_records_and_validates_training_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            training_config = (
                output_root / "training" / "fastlewam" / "scene" / "config.yaml"
            )
            training_config.parent.mkdir(parents=True)
            training_config.write_text("seed: 3072\ntrain_split: 0.9\n", encoding="utf-8")
            episodes = [{"success": index < 7} for index in range(50)]
            payload = {
                "status": "ok",
                "parameters": {"seed": 42},
                "episodes": episodes,
                "checkpoint": str(output_root / "fake_checkpoint.pt"),
                "epoch": 10,
                "cohort_sha256": "frozen-cohort-hash",
            }
            (output_root / "fake_checkpoint.pt").write_bytes(b"checkpoint")

            _attach_result_metadata(
                payload,
                method="fastlewam",
                task="scene",
                output_root=output_root,
            )
            result_dir = output_root / "result"
            result_dir.mkdir()
            (result_dir / "result.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            row = _read_result(
                result_dir,
                method="fast_lewam",
                task="scene",
                condition="P3/step1",
            )

        self.assertEqual(payload["training_seed"], 3072)
        self.assertEqual(payload["evaluation_seed"], 42)
        self.assertEqual(payload["success_count"], 7)
        self.assertEqual(payload["episode_success_vector"], [index < 7 for index in range(50)])
        self.assertEqual(row["training_seed"], 3072)
        self.assertEqual(row["evaluation_seed"], 42)
        self.assertEqual(row["successes"], 7)
        self.assertNotIn("payload", row)

    def test_training_output_validation_requires_all_epoch10_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            expected_paths = {
                "fastlewam": "checkpoints/r4_ab_weights_epoch_10.pt",
                "lewm": "checkpoints/lewm_weights_epoch_10.pt",
                "leflow": "latent_planner_epoch_10.pt",
            }
            for task in ("scene", "finger", "humanoid"):
                for method, checkpoint_name in expected_paths.items():
                    training_root = output_root / "training" / method / task
                    training_root.mkdir(parents=True)
                    if method == "leflow":
                        (training_root / "config.json").write_text(
                            json.dumps({"seed": 3072}), encoding="utf-8"
                        )
                        (training_root / "training_complete.json").write_text(
                            json.dumps({"status": "ok", "epochs": 10}), encoding="utf-8"
                        )
                        (training_root / "latent_planner.pt").write_bytes(b"final")
                    else:
                        (training_root / "config.yaml").write_text(
                            "seed: 3072\n", encoding="utf-8"
                        )
                    checkpoint = training_root / checkpoint_name
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint.write_bytes(b"checkpoint")

            outputs = _validate_training_outputs(output_root)
            self.assertEqual(len(outputs), 9)
            self.assertTrue(all(item["epoch"] == 10 for item in outputs))

            completion = (
                output_root
                / "training"
                / "leflow"
                / "humanoid"
                / "training_complete.json"
            )
            completion.write_text(
                json.dumps({"status": "ok", "epochs": 9}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "did not complete 10 epochs"):
                _validate_training_outputs(output_root)

    def test_success_vector_rejects_missing_or_non_boolean_values(self):
        valid_episodes = [{"success": False} for _ in range(50)]
        self.assertEqual(_successes({"episodes": valid_episodes}), [False] * 50)

        missing = [*valid_episodes[:-1], {}]
        with self.assertRaisesRegex(ValueError, "boolean runtime success"):
            _successes({"episodes": missing})

        nan_value = [*valid_episodes[:-1], {"success": float("nan")}]
        with self.assertRaisesRegex(ValueError, "boolean runtime success"):
            _successes({"episodes": nan_value})

    def test_video_result_must_match_first_frozen_cohort_entry(self):
        manifest = SimpleNamespace(
            cohort_id="scene_legacy_50_phase3_v1",
            computed_sha256="frozen-hash",
            entries=[
                SimpleNamespace(
                    episode_id=7,
                    row_index=100,
                    start_step=12,
                    goal_row_index=37,
                )
            ],
        )
        payload = {
            "status": "ok",
            "phase3_video": {
                "source_cohort_id": manifest.cohort_id,
                "source_cohort_sha256": manifest.computed_sha256,
                "source_entry_index": 0,
            },
            "episodes": [
                {
                    "dataset_episode": 7,
                    "row_index": 100,
                    "start_step": 12,
                    "goal_row_index": 37,
                }
            ],
        }
        self.assertEqual(
            _validate_video_result(
                payload, manifest=manifest, result_path=Path("video.json")
            ),
            "frozen-hash",
        )

        payload["phase3_video"]["source_entry_index"] = 1
        with self.assertRaisesRegex(ValueError, "first frozen cohort entry"):
            _validate_video_result(
                payload, manifest=manifest, result_path=Path("video.json")
            )

    def test_video_file_validation_requires_a_video_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "env_0.mp4"
            video_path.write_bytes(b"video")
            valid_probe = SimpleNamespace(
                returncode=0,
                stdout='{"streams":[{"codec_type":"video","width":320,"height":240}]}',
                stderr="",
            )
            with patch("scripts.round5_phase3.subprocess.run", return_value=valid_probe):
                self.assertEqual(
                    _validate_video_file(video_path), {"width": 320, "height": 240}
                )

            audio_only_probe = SimpleNamespace(
                returncode=0,
                stdout='{"streams":[{"codec_type":"audio"}]}',
                stderr="",
            )
            with patch(
                "scripts.round5_phase3.subprocess.run", return_value=audio_only_probe
            ):
                with self.assertRaisesRegex(ValueError, "no decodable video stream"):
                    _validate_video_file(video_path)

    def test_scene_component_success_thresholds_and_button_equality(self):
        from stable_worldmodel.envs.ogbench.scene_env import SceneEnv

        def compute_successes(*, cube_delta=0.0, button_states=(1, 0), drawer=0.0, window=0.0):
            joints = {
                "object_joint_0": SimpleNamespace(qpos=np.asarray([cube_delta, 0.0, 0.0])),
                "drawer_slide": SimpleNamespace(qpos=np.asarray([drawer])),
                "window_slide": SimpleNamespace(qpos=np.asarray([window])),
            }
            env = SimpleNamespace(
                _num_cubes=1,
                _cube_target_mocap_ids=[0],
                _num_buttons=2,
                _cur_button_states=np.asarray(button_states),
                _target_button_states=np.asarray([1, 0]),
                _target_drawer_pos=0.0,
                _target_window_pos=0.0,
                _data=SimpleNamespace(
                    joint=lambda name: joints[name],
                    mocap_pos=np.zeros((1, 3)),
                ),
            )
            return SceneEnv._compute_successes(env)

        self.assertEqual(
            compute_successes(cube_delta=0.04, button_states=(1, 0), drawer=0.04, window=-0.04),
            ([True], [True, True], True, True),
        )
        self.assertEqual(compute_successes(cube_delta=0.040001)[0], [False])
        self.assertEqual(compute_successes(button_states=(1, 1))[1], [True, False])
        self.assertFalse(compute_successes(drawer=0.040001)[2])
        self.assertFalse(compute_successes(window=-0.040001)[3])

    def test_scene_task_success_requires_every_component(self):
        from stable_worldmodel.envs.ogbench.scene_env import SceneEnv

        all_success = ([True], [True, True], True, True)
        component_failures = (
            ([False], [True, True], True, True),
            ([True], [False, True], True, True),
            ([True], [True, False], True, True),
            ([True], [True, True], False, True),
            ([True], [True, True], True, False),
        )

        def make_env(successes):
            joints = {
                "buttonbox_joint_0": SimpleNamespace(qpos=np.asarray([0.0])),
                "buttonbox_joint_1": SimpleNamespace(qpos=np.asarray([0.0])),
            }
            return SimpleNamespace(
                _mode="eval",
                _num_buttons=2,
                _prev_ob_info={
                    "privileged/button_0_pos": np.asarray([0.0]),
                    "privileged/button_1_pos": np.asarray([0.0]),
                },
                _data=SimpleNamespace(joint=lambda name: joints[name]),
                _cur_button_states=np.asarray([0, 0]),
                _num_button_states=3,
                _apply_button_states=lambda: None,
                _compute_successes=lambda: successes,
                _visualize_info=False,
                _num_cubes=1,
                _cube_target_geom_ids_list=[[]],
                _success=None,
            )

        env = make_env(all_success)
        SceneEnv.post_step(env)
        self.assertTrue(env._success)
        for successes in component_failures:
            env = make_env(successes)
            SceneEnv.post_step(env)
            self.assertFalse(env._success, successes)

    def test_finger_and_humanoid_success_predicate_boundaries(self):
        patch_phase3_environments()
        from stable_worldmodel.envs.dmcontrol.finger import FingerDMControlWrapper
        from stable_worldmodel.envs.dmcontrol.humanoid import HumanoidDMControlWrapper

        for reward, expected in ((0.999999, False), (1.0, True)):
            env = SimpleNamespace(
                env=SimpleNamespace(
                    task=SimpleNamespace(get_reward=lambda _physics, value=reward: value),
                    physics=object(),
                )
            )
            self.assertEqual(FingerDMControlWrapper._is_terminated(env, step=1), expected)

        def humanoid_terminated(head_height, torso_upright, velocity):
            physics = SimpleNamespace(
                head_height=lambda: head_height,
                torso_upright=lambda: torso_upright,
                center_of_mass_velocity=lambda: np.asarray(velocity),
            )
            env = SimpleNamespace(env=SimpleNamespace(physics=physics))
            return HumanoidDMControlWrapper._is_terminated(env, step=1)

        self.assertTrue(humanoid_terminated(1.4, 0.9, [1.0, 0.0, 100.0]))
        self.assertFalse(humanoid_terminated(1.399999, 0.9, [1.0, 0.0, 0.0]))
        self.assertFalse(humanoid_terminated(1.4, 0.899999, [1.0, 0.0, 0.0]))
        self.assertFalse(humanoid_terminated(1.4, 0.9, [0.999999, 0.0, 0.0]))

    def test_scene_goal_setter_applies_all_future_physical_targets(self):
        class SceneTargetRecorder:
            def __init__(self):
                self.calls = []

            def set_cube_target_pos(self, cube_id, position, quaternion):
                self.calls.append(("cube", cube_id, position.copy(), quaternion.copy()))

            def set_target_button_state(self, button_id, state):
                self.calls.append(("button", button_id, state))

            def set_target_drawer_pos(self, position):
                self.calls.append(("drawer", position))

            def set_target_window_pos(self, position):
                self.calls.append(("window", position))

        env = SceneTargetRecorder()
        set_phase3_scene_goal(
            env,
            target_block_pos=np.asarray([0.1, 0.2, 0.3]),
            target_block_quat=np.asarray([1.0, 0.0, 0.0, 0.0]),
            target_button_0_state=np.asarray(1),
            target_button_1_state=np.asarray(2),
            target_drawer_pos=np.asarray([0.04]),
            target_window_pos=np.asarray([0.12]),
        )

        self.assertEqual(env.calls[0][0:2], ("cube", 0))
        np.testing.assert_array_equal(env.calls[0][2], [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(env.calls[0][3], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(env.calls[1:], [
            ("button", 0, 1),
            ("button", 1, 2),
            ("drawer", 0.04),
            ("window", 0.12),
        ])

    def test_scene_targets_are_loaded_from_the_future_goal_row(self):
        physical_goal_fields = {
            "privileged_block_0_pos",
            "privileged_block_0_quat",
            "privileged_button_0_state",
            "privileged_button_1_state",
            "privileged_drawer_pos",
            "privileged_window_pos",
        }
        self.assertTrue(physical_goal_fields.issubset(set(_dataset_keys("scene"))))

        config_path = Path(__file__).resolve().parents[1] / "config" / "eval" / "scene.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        goal_callable = next(
            item for item in config["eval"]["callables"]
            if item["method"] == "set_phase3_scene_goal"
        )
        actual_mapping = {
            name: value["value"]
            for name, value in goal_callable["args"].items()
        }
        self.assertEqual(
            actual_mapping,
            {
                "target_block_pos": "goal_privileged_block_0_pos",
                "target_block_quat": "goal_privileged_block_0_quat",
                "target_button_0_state": "goal_privileged_button_0_state",
                "target_button_1_state": "goal_privileged_button_1_state",
                "target_drawer_pos": "goal_privileged_drawer_pos",
                "target_window_pos": "goal_privileged_window_pos",
            },
        )

    def test_condition_matrix_has_frozen_61_conditions_per_task(self):
        for task in ("scene", "finger", "humanoid"):
            conditions = fast_conditions(task)
            self.assertEqual(len(conditions), 61)
            self.assertEqual(sum(item["guidance"] == "none" for item in conditions), 19)
            self.assertEqual(sum(item["guidance"] != "none" for item in conditions), 42)
        self.assertEqual(p2_protocol("scene"), "legacy")
        self.assertEqual(p2_protocol("finger"), "cem-clip")
        self.assertEqual(p2_protocol("humanoid"), "cem-clip")

    def test_phase3_manifest_matches_shared_legacy_selector(self):
        dataset = _TinyDataset()
        manifest = build_phase3_legacy_manifest(
            dataset,
            task="scene",
            seed=42,
            goal_offset_steps=2,
            num_eval=3,
        )
        selected = select_eval_cohort(
            dataset,
            goal_offset_steps=2,
            num_eval=3,
            seed=42,
        )
        self.assertTrue(
            np.array_equal(
                selected.row_indices,
                np.asarray([entry.row_index for entry in manifest.entries]),
            )
        )
        self.assertEqual(manifest.protocol_variant, "legacy")
        self.assertEqual(manifest.diagnostics["environment_success_source"], "runtime_termination")


if __name__ == "__main__":
    unittest.main()
