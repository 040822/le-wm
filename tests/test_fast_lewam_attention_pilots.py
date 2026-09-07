import json
import os
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from source.common.eval import (
    DatasetEvaluationSession,
    EvaluationCohort,
    EvaluationIdentity,
)
from source.experiments.fast_lewam_attention_pilots import (
    GROUNDING_SLOTS,
    PILOT_GPU_MAPPING,
    Pilot,
    StaleArtifactError,
    artifact_identity,
    atomic_write_json,
    build_launch_specs,
    canonical_train_overrides,
    cohort_for_level,
    decide_epoch3_promotion,
    file_sha256,
    load_pilot_manifest,
    memory_guard_ok,
    paired_success_metrics,
    parse_pmon_processes,
    read_phase_marker,
    validate_artifact_identity,
    validate_cohort,
    validate_gpu_mapping,
    validate_grounding_slots,
    validate_memory_guard,
    write_phase_marker,
)
from scripts.run_fast_lewam_attention_pilots import (
    _parse_pmon_processes,
    _pid_matches,
    _summary,
)
from scripts.run_fast_lewam_attention_pilot_worker import (
    _build_train_command,
    _fast_dev_train_overrides,
    _evaluation_cache_identity,
    _evaluation_identity_path,
    _grounding_cache_identity,
    _stage_epochs,
    _validate_evaluation_cache,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config/experiments/fast_lewam_attention_pilots.yaml"


class _FakeDataset:
    column_names = ("episode_idx", "step_idx")

    def __init__(self):
        self.columns = {
            "episode_idx": np.array([0, 0, 0, 1, 1, 1]),
            "step_idx": np.array([0, 1, 2, 0, 1, 2]),
        }

    def get_col_data(self, name):
        return self.columns[name]

    def get_row_data(self, rows):
        return {name: values[rows] for name, values in self.columns.items()}


def _result(success, episodes):
    return {
        "status": "ok",
        "success_rate": success,
        "episodes": [
            {"dataset_episode": episode, "start_step": step}
            for episode, step in episodes
        ],
    }


class FastLeWAMAttentionPilotTests(unittest.TestCase):
    def test_fast_lewam_train_config_declares_resume_checkpoint(self):
        cfg = OmegaConf.load(ROOT / "config/train/fast_lewam.yaml")
        self.assertIn("resume_ckpt", cfg)
        self.assertIsNone(cfg.resume_ckpt)

    def test_pid_matches_current_process_cmdline(self):
        command = [
            value.decode(errors="replace")
            for value in Path(f"/proc/{os.getpid()}/cmdline").read_bytes().split(b"\0")
            if value
        ]
        self.assertTrue(_pid_matches(os.getpid(), command))
        self.assertFalse(_pid_matches(os.getpid(), command + ["not-the-command"]))


    def test_manifest_has_exact_four_way_mapping_and_nested_cohorts(self):
        manifest = load_pilot_manifest(MANIFEST)
        self.assertEqual(
            {
                name: (pilot.gpu, pilot.task, pilot.attention_variant)
                for name, pilot in manifest.pilots.items()
            },
            PILOT_GPU_MAPPING,
        )
        self.assertEqual(manifest.output_root, ROOT / "outputs/experiments/fast_lewam_attention_pilots/0820")
        for pilot in manifest.pilots.values():
            cohorts = [cohort_for_level(manifest, pilot, epoch) for epoch in (1, 3, 10)]
            self.assertEqual([len(value.row_indices) for value in cohorts], [10, 20, 50])
            np.testing.assert_array_equal(cohorts[0].row_indices, cohorts[2].row_indices[:10])
            np.testing.assert_array_equal(cohorts[1].row_indices, cohorts[2].row_indices[:20])

    def test_manifest_rejects_duplicate_or_prohibited_gpu(self):
        manifest = load_pilot_manifest(MANIFEST)
        values = {
            name: {
                "gpu": pilot.gpu,
                "task": pilot.task,
                "attention_variant": pilot.attention_variant,
            }
            for name, pilot in manifest.pilots.items()
        }
        values["pusht_block_causal"]["gpu"] = 4
        with self.assertRaises(ValueError):
            validate_gpu_mapping(values)
        values["pusht_block_causal"]["gpu"] = 0
        with self.assertRaises(ValueError):
            validate_gpu_mapping(values)

    def test_launch_specs_bind_physical_gpu_and_logical_cuda_zero(self):
        manifest = load_pilot_manifest(MANIFEST)
        specs = build_launch_specs(manifest, through_epoch=3)
        self.assertEqual([spec.gpu for spec in specs], [0, 1, 2, 3])
        for spec in specs:
            self.assertEqual(spec.environment["CUDA_VISIBLE_DEVICES"], str(spec.gpu))
            self.assertEqual(spec.environment["MUJOCO_EGL_DEVICE_ID"], str(spec.gpu))
            self.assertIn("--device", spec.command)
            self.assertIn("cuda:0", spec.command)
            self.assertIn("--physical-gpu", spec.command)

    def test_explicit_cohort_is_validated_against_dataset_rows(self):
        cfg = OmegaConf.create({
            "cache_dir": None,
            "seed": 42,
            "world": {"num_envs": 2},
            "dataset": {"keys_to_cache": []},
            "plan_config": {"horizon": 1, "receding_horizon": 1, "action_block": 1},
            "solver": {"device": "cpu"},
            "eval": {
                "num_eval": 2,
                "goal_offset_steps": 0,
                "eval_budget": 2,
                "img_size": 8,
                "dataset_name": "fake",
                "callables": [],
            },
        })
        cohort = EvaluationCohort(
            row_indices=np.array([0, 4]),
            episode_ids=np.array([0, 1]),
            start_steps=np.array([0, 1]),
        )
        session = DatasetEvaluationSession(
            cfg, task="fake", dataset=_FakeDataset(), cohort=cohort
        )
        np.testing.assert_array_equal(session.cohort.row_indices, [0, 4])
        bad = EvaluationCohort(
            row_indices=np.array([0, 4]),
            episode_ids=np.array([0, 0]),
            start_steps=np.array([0, 1]),
        )
        with self.assertRaisesRegex(ValueError, "episode_ids"):
            DatasetEvaluationSession(
                cfg, task="fake", dataset=_FakeDataset(), cohort=bad
            )

    def test_validate_cohort_rejects_changed_reference_rows(self):
        manifest = load_pilot_manifest(MANIFEST)
        expected = cohort_for_level(manifest, "reacher", 3)
        changed = EvaluationCohort(
            row_indices=expected.row_indices,
            episode_ids=expected.episode_ids,
            start_steps=expected.start_steps + 1,
        )
        with self.assertRaises(ValueError):
            validate_cohort(changed, expected_length=20, reference=expected)

    def test_promotion_uses_percentage_points_and_exact_paired_units(self):
        episodes = [(1, 2), (3, 4)]
        self.assertEqual(
            decide_epoch3_promotion(_result(80.0, episodes), _result(85.0, episodes)),
            "promote",
        )
        self.assertEqual(
            decide_epoch3_promotion(_result(80.0, episodes), _result(79.0, episodes)),
            "stop",
        )
        with self.assertRaisesRegex(ValueError, "cohorts differ"):
            decide_epoch3_promotion(
                _result(80.0, episodes),
                _result(85.0, [(1, 3), (3, 4)]),
            )

    def test_promotion_grounding_branch_requires_both_signals(self):
        episodes = [(1, 2), (3, 4)]
        baseline = _result(80.0, episodes)
        candidate = _result(82.0, episodes)
        base_ground = {
            "predicted_physical_spearman": 0.20,
            "physical_best_regret": 10.0,
        }
        candidate_ground = {
            "predicted_physical_spearman": 0.25,
            "physical_best_regret": 9.0,
        }
        self.assertEqual(
            decide_epoch3_promotion(
                baseline, candidate,
                baseline_grounding=base_ground,
                candidate_grounding=candidate_ground,
            ),
            "promote",
        )
        candidate_ground["physical_best_regret"] = 9.1
        self.assertEqual(
            decide_epoch3_promotion(
                baseline, candidate,
                baseline_grounding=base_ground,
                candidate_grounding=candidate_ground,
            ),
            "stop",
        )
        self.assertEqual(
            decide_epoch3_promotion(baseline, candidate),
            "pending_grounding",
        )

    def test_fixed_slot_sets_are_exact(self):
        self.assertEqual(
            validate_grounding_slots("reacher", 3, [8, 9, 0, 1, 2, 3]),
            GROUNDING_SLOTS[("reacher", 3)],
        )
        with self.assertRaises(ValueError):
            validate_grounding_slots("reacher", 3, [0, 1, 2, 3, 8, 10])
        with self.assertRaises(ValueError):
            validate_grounding_slots("pusht", 10, [1, 4, 11])

    def test_pmon_parser_and_memory_peak_guard(self):
        output = "# gpu pid type sm mem enc dec jpg ofa command\n  2 1234 C 1 2 - - - - python\n"
        self.assertEqual(_parse_pmon_processes(output, 2), ["pid=1234 type=C command=python"])
        idle = "  2 - - 0 0 - - - - -\n"
        self.assertEqual(parse_pmon_processes(idle, 2), [])
        with self.assertRaisesRegex(RuntimeError, "GPU1"):
            parse_pmon_processes("  1 - - 0 0 - - - - -\n", 2)
        with self.assertRaisesRegex(RuntimeError, "GPU1"):
            parse_pmon_processes(output, 1)
        with self.assertRaises(RuntimeError):
            validate_memory_guard(0, total_mib=10000, used_mib=100, free_mib=9000, peak_used_mib=8500)
        with self.assertRaises(RuntimeError):
            validate_memory_guard(0, total_mib=10000, used_mib=100, free_mib=9000, post_test_free_mib=7167)
        self.assertFalse(memory_guard_ok(0, total_mib=10000, used_mib=8500, free_mib=9000))
        self.assertTrue(memory_guard_ok(0, total_mib=10000, used_mib=100, free_mib=9000, peak_used_mib=8000))

    def test_percentage_semantics_keep_one_percent_as_one_pp(self):
        episodes = [(1, 2), (3, 4)]
        metrics = paired_success_metrics(
            _result(0.0, episodes), _result(1.0, episodes)
        )
        self.assertEqual(metrics["baseline_success_percent"], 0.0)
        self.assertEqual(metrics["candidate_success_percent"], 1.0)
        self.assertEqual(metrics["success_delta_pp"], 1.0)

    def test_canonical_overrides_disable_strategy_and_validation_side_effects(self):
        overrides = canonical_train_overrides(
            epoch=3,
            task="reacher",
            attention_variant="block_causal",
            run_dir="/tmp/attention-pilot",
        )
        self.assertIn("wandb.enabled=false", overrides)
        self.assertNotIn("trainer.strategy=auto", overrides)
        self.assertNotIn("validation_diagnostics.enabled=false", overrides)

    def test_fast_dev_overrides_use_one_step_constant_scheduler(self):
        overrides = _fast_dev_train_overrides()
        self.assertEqual(
            overrides,
            (
                "+trainer.fast_dev_run=true",
                "trainer.max_epochs=1",
                "+policy.scheduler.type=ConstantLR",
                "+policy.scheduler.factor=1.0",
                "+policy.scheduler.total_iters=1",
            ),
        )


    def test_eval_sidecar_rejects_checkpoint_and_config_byte_changes(self):
        manifest = load_pilot_manifest(MANIFEST)
        pilot = manifest.pilots["reacher_block_causal"]
        expected = cohort_for_level(manifest, pilot, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "candidate.ckpt"
            run_config = root / "config.yaml"
            output = root / "eval"
            result = output / "result.json"
            checkpoint.write_bytes(b"checkpoint-v1")
            run_config.write_bytes(b"config-v1")
            result.parent.mkdir()
            result.write_text(
                json.dumps(_result(
                    50.0,
                    list(zip(expected.episode_ids.tolist(), expected.start_steps.tolist())),
                )),
                encoding="utf-8",
            )
            identity = _evaluation_cache_identity(
                manifest,
                pilot,
                epoch=1,
                checkpoint=checkpoint,
                run_config=run_config,
                cohort=expected,
            )
            sidecar = _evaluation_identity_path(output)
            atomic_write_json(
                sidecar,
                {"schema_version": 1, "identity": identity,
                 "result_sha256": file_sha256(result)},
            )
            _validate_evaluation_cache(result, sidecar, identity)
            checkpoint.write_bytes(b"checkpoint-v2")
            changed_identity = _evaluation_cache_identity(
                manifest,
                pilot,
                epoch=1,
                checkpoint=checkpoint,
                run_config=run_config,
                cohort=expected,
            )
            with self.assertRaises(StaleArtifactError):
                _validate_evaluation_cache(result, sidecar, changed_identity)
            atomic_write_json(
                sidecar,
                {"schema_version": 1, "identity": changed_identity,
                 "result_sha256": file_sha256(result)},
            )
            run_config.write_bytes(b"config-v2")
            changed_config_identity = _evaluation_cache_identity(
                manifest,
                pilot,
                epoch=1,
                checkpoint=checkpoint,
                run_config=run_config,
                cohort=expected,
            )
            with self.assertRaises(StaleArtifactError):
                _validate_evaluation_cache(result, sidecar, changed_config_identity)

    def test_grounding_identity_changes_for_candidate_input_bytes(self):
        manifest = load_pilot_manifest(MANIFEST)
        source_pilot = manifest.pilots["reacher_block_causal"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_config = root / "baseline.yaml"
            baseline_checkpoint = root / "baseline.ckpt"
            candidate_config = root / "candidate.yaml"
            candidate_checkpoint = root / "candidate.ckpt"
            candidate_result = root / "candidate.json"
            for path, value in (
                (baseline_config, b"baseline-config"),
                (baseline_checkpoint, b"baseline-checkpoint"),
                (candidate_config, b"candidate-config-v1"),
                (candidate_checkpoint, b"candidate-checkpoint-v1"),
            ):
                path.write_bytes(value)
            candidate_result.write_bytes(source_pilot.baseline_result.read_bytes())
            pilot = Pilot(
                name=source_pilot.name,
                gpu=source_pilot.gpu,
                task=source_pilot.task,
                attention_variant=source_pilot.attention_variant,
                output_dir=root / "output",
                baseline_run_dir=root,
                baseline_config=baseline_config,
                baseline_checkpoints={3: baseline_checkpoint, 10: baseline_checkpoint},
                baseline_result=source_pilot.baseline_result,
                grounding_slots=source_pilot.grounding_slots,
            )
            first = _grounding_cache_identity(
                manifest,
                pilot,
                epoch=3,
                baseline_result=source_pilot.baseline_result,
                candidate_result=candidate_result,
                candidate_config=candidate_config,
                candidate_checkpoint=candidate_checkpoint,
            )
            self.assertEqual(first["cohort_sha256"], _grounding_cache_identity(
                manifest,
                pilot,
                epoch=3,
                baseline_result=source_pilot.baseline_result,
                candidate_result=candidate_result,
                candidate_config=candidate_config,
                candidate_checkpoint=candidate_checkpoint,
            )["cohort_sha256"])
            candidate_checkpoint.write_bytes(b"candidate-checkpoint-v2")
            second = _grounding_cache_identity(
                manifest,
                pilot,
                epoch=3,
                baseline_result=source_pilot.baseline_result,
                candidate_result=candidate_result,
                candidate_config=candidate_config,
                candidate_checkpoint=candidate_checkpoint,
            )
            self.assertNotEqual(
                first["input_artifacts"]["candidate_checkpoint"],
                second["input_artifacts"]["candidate_checkpoint"],
            )
            candidate_config.write_bytes(b"candidate-config-v2")
            third = _grounding_cache_identity(
                manifest,
                pilot,
                epoch=3,
                baseline_result=source_pilot.baseline_result,
                candidate_result=candidate_result,
                candidate_config=candidate_config,
                candidate_checkpoint=candidate_checkpoint,
            )
            self.assertNotEqual(
                second["input_artifacts"]["candidate_config"],
                third["input_artifacts"]["candidate_config"],
            )
            candidate_result.write_bytes(candidate_result.read_bytes() + b" ")
            fourth = _grounding_cache_identity(
                manifest,
                pilot,
                epoch=3,
                baseline_result=source_pilot.baseline_result,
                candidate_result=candidate_result,
                candidate_config=candidate_config,
                candidate_checkpoint=candidate_checkpoint,
            )
            self.assertNotEqual(
                third["input_artifacts"]["candidate_result"],
                fourth["input_artifacts"]["candidate_result"],
            )

    def test_identity_bound_artifacts_reject_stale_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"v1")
            identity = artifact_identity([path])
            self.assertTrue(validate_artifact_identity(identity))
            path.write_bytes(b"v2")
            self.assertFalse(validate_artifact_identity(identity))

    def test_phase_marker_rejects_stale_parent_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"ok")
            marker = Path(directory) / "phase.json"
            identity = {"manifest_sha256": "m1", "parent": {"sha256": "p1"}}
            write_phase_marker(marker, identity, artifact_identity([path]))
            self.assertTrue(read_phase_marker(marker, identity))
            with self.assertRaises(StaleArtifactError):
                read_phase_marker(marker, {**identity, "parent": {"sha256": "p2"}})

    def test_resume_stage_command_is_full_checkpoint_bound(self):
        manifest = load_pilot_manifest(MANIFEST)
        pilot = manifest.pilots["reacher_block_causal"]
        command = _build_train_command(
            manifest,
            pilot,
            epoch=3,
            run_dir="/tmp/attention-pilot-stage",
            parent_checkpoint="/tmp/epoch_1/checkpoints/last.ckpt",
        )
        self.assertIn("stage_b_attention_mode=block_causal", command)
        self.assertIn("resume_ckpt=/tmp/epoch_1/checkpoints/last.ckpt", command)
        self.assertEqual(_stage_epochs(10), (1, 3, 10))

    def test_summarize_writes_atomic_terminal_aware_top_level_status(self):
        manifest = load_pilot_manifest(MANIFEST)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilots = {}
            for source_pilot in manifest.pilots.values():
                output_dir = root / source_pilot.name
                pilots[source_pilot.name] = replace(
                    source_pilot, output_dir=output_dir
                )
                summary_dir = output_dir / "summaries"
                summary_dir.mkdir(parents=True)
                (summary_dir / "epoch_1.json").write_text(
                    json.dumps({"status": "ok"}), encoding="utf-8"
                )
            local = replace(
                manifest,
                output_root=root / "summary-root",
                pilots=pilots,
            )
            one = _summary(local, through_epoch=1)
            self.assertEqual(one["status"], "ok")
            self.assertEqual(
                json.loads((local.output_root / "summary.json").read_text())["status"],
                "ok",
            )
            for pilot in pilots.values():
                (pilot.output_dir / "summaries" / "epoch_3.json").write_text(
                    json.dumps({"status": "ok", "decision": "stop"}),
                    encoding="utf-8",
                )
            ten = _summary(local, through_epoch=10)
            self.assertEqual(ten["status"], "ok")
            for pilot in pilots.values():
                (pilot.output_dir / "summaries" / "epoch_1.json").unlink()
                (pilot.output_dir / "summaries" / "epoch_3.json").unlink()
            running_pilot = next(iter(pilots.values()))
            command_bytes = Path(f"/proc/{os.getpid()}/cmdline").read_bytes()
            command = [
                value.decode(errors="replace")
                for value in command_bytes.split(b"\0")
                if value
            ]
            (running_pilot.output_dir / "process.json").write_text(
                json.dumps({
                    "pid": os.getpid(),
                    "command": command,
                    "manifest_sha256": local.sha256,
                }),
                encoding="utf-8",
            )
            running = _summary(local, through_epoch=10)
            self.assertEqual(running["status"], "running")

    def test_identity_directories_ignore_python_runtime_cache(self):
        from source.experiments.fast_lewam_attention_pilots import directory_identity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "x.pyc").write_bytes(b"one")
            before = directory_identity(root)
            (root / "__pycache__" / "x.pyc").write_bytes(b"two")
            self.assertEqual(before, directory_identity(root))


if __name__ == "__main__":
    unittest.main()
