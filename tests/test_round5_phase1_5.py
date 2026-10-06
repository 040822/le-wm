import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import gymnasium as gym
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
from scripts import round5_phase1_5_diagnostics as phase15_diagnostics

from scripts.round5_phase1_5 import (
    _coarse_success_latency_frontier,
    _coarse_timing_frontier_sensitivity,
    _coarse_timing_sensitivity_report_lines,
    _complete_coarse_timing,
    _complete_fine_timing,
    _diagnostic_decisions,
    _fine_timing_retry_status,
    _guided_flow_selection_gate,
    _history_reuse_audit,
    _host_preflight,
    _matched_guidance_timing_comparisons,
    _resolve_scan_sources,
    _scan_unbounded,
    _select_method_candidates,
    _shared_method_key,
    _timing_config_key,
    _timing_interference_status,
    merge_fine_timing_staging_roots,
    merge_timing_staging_roots,
    _render_report,
    build_parser,
)
from scripts.round5_phase1_5_diagnostics import (
    FixedCandidatePolicy,
    _branch_file_matches,
    _collect_features,
    _common_guidance_outcome,
    _control_pool_actions,
    _current_control_artifacts_match,
    _extract_step_pixel_batch,
    _pack_primitive_actions_for_replay,
    _probe_candidate_predictions,
    _same_rms_random_actions,
    _terminal_env_slots,
)
from source.common.eval import compose_eval_config, img_transform
from source.common.round4_eval import _FirstActionTimingTap, _TimingCaptureComplete

from source.common.round5_phase1_5 import (
    PHASE15_BOOTSTRAP_SAMPLES,
    adaptive_stability_specs,
    capture_environment_state,
    candidate_selection_metrics,
    cluster_bootstrap,
    condition_id,
    condition_identity,
    deduplicate_actions,
    grid_counts,
    guidance_effect_metrics,
    matched_guidance_mode_metrics,
    matched_guidance_mode_metrics_by_milestone,
    make_control_actions,
    make_probe_split,
    normalized_physical_distance,
    candidate_pool_metrics,
    condition_lock,
    control_action_metrics,
    capture_rng_state,
    fit_ridge_probe,
    paired_guidance_metrics,
    phase15_scan_slot,
    primary_condition_specs,
    safe_correlation,
    sampling_stability_specs,
    stable_sha256,
    synchronous_timing,
    summarize_timing_samples,
    restore_environment_state,
    restore_rng_state,
)


class Round5Phase15GridTests(unittest.TestCase):
    def test_primary_grid_matches_plan(self):
        specs = primary_condition_specs()
        self.assertEqual(len(specs), 2424)
        self.assertEqual(
            grid_counts(specs),
            {"A1": 144, "A2": 432, "A3": 864, "A4": 432, "A5": 96, "A6": 240, "A7": 216},
        )
        self.assertEqual(len(sampling_stability_specs()), 96)

    def test_history_reuse_audit_records_source_and_nonreuse_reason(self):
        audit = _history_reuse_audit(
            {
                "items": {
                    "a": {
                        "source": "current",
                        "reuse_reason": "current_artifact_already_available",
                        "historical_path_candidate_count": 2,
                    },
                    "b": {
                        "source": None,
                        "reuse_reason": "historical_candidates_did_not_match_identity_and_execution_semantics",
                        "historical_path_candidate_count": 3,
                    },
                }
            },
            {
                "compatible_history_count": 1,
                "compatible_full_reuse_count": 1,
                "compatible_success_only_count": 0,
                "compatible_by_task": {"cube": 1},
                "compatible_by_historical_round": {"round4": 1},
                "current_index_source_counts_for_matches": {"current": 1},
                "current_artifact_exists_counts": {"True": 1},
            },
        )

        self.assertEqual(audit["source_counts"], {"current": 1, "none": 1})
        self.assertEqual(
            audit["reuse_reason_counts"],
            {
                "current_artifact_already_available": 1,
                "historical_candidates_did_not_match_identity_and_execution_semantics": 1,
            },
        )
        self.assertEqual(audit["historical_path_candidate_pairs"], 5)
        self.assertTrue(audit["historical_path_candidate_pairs_are_hints_only"])
        self.assertEqual(
            audit["verified_historical_candidates"],
            {
                "conditions": 1,
                "full_reuse": 1,
                "success_only": 0,
                "by_task": {"cube": 1},
                "by_round": {"round4": 1},
                "current_index_sources": {"current": 1},
                "current_artifacts_present": {"True": 1},
            },
        )

    def test_scan_resolves_current_and_history_before_scheduling_new_runs(self):
        specs = [
            {"task": "cube", "group": "A1"},
            {"task": "reacher", "group": "A1"},
            {"task": "tworoom", "group": "A1"},
        ]
        identities = {
            stable_sha256(spec): {"identity": spec["task"]}
            for spec in specs
        }
        history_ids = {
            spec["task"]: condition_id(identities[stable_sha256(spec)])
            for spec in specs
        }
        with tempfile.TemporaryDirectory() as directory:
            current_path = Path(directory) / "cube-result.json"
            current_path.write_text("{}", encoding="utf-8")

            def target_for_spec(_output_root, spec, _identity):
                if spec["task"] == "cube":
                    return current_path
                return Path(directory) / f"{spec['task']}-result.json"

            history_index = {
                history_ids["reacher"]: {
                    "condition_id": history_ids["reacher"],
                    "source": "history",
                    "reuse_scope": "full",
                    "reuse_reason": "compatible_historical_artifact_reused_full",
                    "validation": {"episodes": 50},
                    "path": "/history/reacher/result.json",
                },
                history_ids["tworoom"]: {
                    "condition_id": history_ids["tworoom"],
                    "source": None,
                    "status": "pending",
                },
            }
            with patch(
                "scripts.round5_phase1_5.result_path",
                side_effect=target_for_spec,
            ), patch(
                "scripts.round5_phase1_5.validate_phase15_result",
                return_value={"episodes": 50},
            ), patch(
                "scripts.round5_phase1_5.index_phase15_results",
                return_value=history_index,
            ) as index_history:
                pending, reused = _resolve_scan_sources(
                    specs,
                    output_root=Path(directory),
                    identities=identities,
                    manifests={"cube": object()},
                    checkpoints={},
                    history_roots=["/history"],
                )

        self.assertEqual([spec["task"] for spec in pending], ["tworoom"])
        self.assertEqual(
            [(item["source"], item["condition_id"]) for item in reused],
            [("current", history_ids["cube"]), ("history", history_ids["reacher"])],
        )
        self.assertEqual(reused[1]["episodes"], 50)
        index_history.assert_called_once()

    def test_scan_does_not_touch_gpu_when_every_condition_is_reusable(self):
        args = SimpleNamespace(
            include_stability=False,
            include_adaptive_stability=False,
            output_root="unused-output-root",
        )
        with patch(
            "scripts.round5_phase1_5._selected_specs",
            return_value=[{"task": "cube"}],
        ), patch(
            "scripts.round5_phase1_5._load_manifests",
            return_value={},
        ), patch(
            "scripts.round5_phase1_5._checkpoint_paths",
            return_value=({}, {}),
        ), patch(
            "scripts.round5_phase1_5._identity_map",
            return_value={},
        ), patch(
            "scripts.round5_phase1_5._resolve_scan_sources",
            return_value=([], [{"source": "history", "condition_id": "history-hit"}]),
        ), patch(
            "scripts.round5_phase1_5._configure_device",
        ) as configure_device, patch("sys.stdout"):
            _scan_unbounded(args, {})

        configure_device.assert_not_called()

    def test_method_selection_reports_unconverged_without_clean_timing(self):
        rows = [
            {
                "candidate_id": "slow-and-interfered",
                "fine_timing_eligible": False,
                "mean_success_rate": 0.98,
                "mean_batch1_p50_seconds": 0.1,
                "mean_model_calls_per_decision": 1.0,
                "task_success_rate": {
                    "cube": 0.98,
                    "pusht": 0.98,
                    "reacher": 0.98,
                    "tworoom": 0.98,
                },
            }
        ]

        main, backup = _select_method_candidates(rows)

        self.assertIsNone(main)
        self.assertIsNone(backup)
        self.assertFalse(rows[0]["method_pareto"])

    def test_a1_and_a7_remove_the_declared_aliases(self):
        specs = primary_condition_specs()
        a1 = [item for item in specs if item["group"] == "A1"]
        self.assertEqual(sum(item["mode"] == "P0" for item in a1), 24)
        self.assertFalse(
            any(
                item["group"] == "A7"
                and item["guidance"] == "guided_flow"
                and item["flow_steps"] == 1
                for item in specs
            )
        )

    def test_scan_defaults_to_one_route(self):
        args = build_parser().parse_args(["scan"])
        self.assertEqual(args.max_concurrent_scans, 1)

    def test_fine_timing_selection_uses_full_plan_flags(self):
        args = build_parser().parse_args(
            ["select-fine-timing", "--include-stability", "--include-adaptive-stability"]
        )
        self.assertTrue(args.include_stability)
        self.assertTrue(args.include_adaptive_stability)

    def test_coarse_frontier_is_per_task_and_deduplicates_identical_points(self):
        rows = [
            {"task": "cube", "condition_id": "slower", "success_rate": 0.8, "batch1_p50_seconds": 0.2, "forward_count": 10},
            {"task": "cube", "condition_id": "same_but_cheaper", "success_rate": 0.8, "batch1_p50_seconds": 0.2, "forward_count": 4},
            {"task": "cube", "condition_id": "best_success", "success_rate": 0.9, "batch1_p50_seconds": 0.3, "forward_count": 20},
            {"task": "cube", "condition_id": "dominated", "success_rate": 0.7, "batch1_p50_seconds": 0.4, "forward_count": 2},
            {"task": "reacher", "condition_id": "other_task", "success_rate": 0.7, "batch1_p50_seconds": 0.4, "forward_count": 2},
        ]
        result = _coarse_success_latency_frontier(rows)
        self.assertEqual(
            [(row["task"], row["condition_id"]) for row in result],
            [("cube", "same_but_cheaper"), ("cube", "best_success"), ("reacher", "other_task")],
        )

    def test_coarse_frontier_sensitivity_keeps_observed_and_boundary_unflagged_frontiers(self):
        no_external = "no_external_compute_observed_at_boundaries"
        rows = [
            {
                "task": "cube",
                "condition_id": "interfered_observed_frontier",
                "success_rate": 0.9,
                "batch1_p50_seconds": 0.1,
                "batch50_p50_seconds": 1.0,
                "batch1_interference_status": "interference_flagged",
                "coarse_timing_interference": {
                    "batch1": "interference_flagged",
                    "batch50": "interference_flagged",
                },
            },
            {
                "task": "cube",
                "condition_id": "clean_low_success",
                "success_rate": 0.8,
                "batch1_p50_seconds": 0.2,
                "batch50_p50_seconds": 2.0,
                "batch1_interference_status": no_external,
                "coarse_timing_interference": {
                    "batch1": no_external,
                    "batch50": no_external,
                },
            },
            {
                "task": "cube",
                "condition_id": "clean_high_success",
                "success_rate": 0.9,
                "batch1_p50_seconds": 0.3,
                "batch50_p50_seconds": 3.0,
                "batch1_interference_status": no_external,
                "coarse_timing_interference": {
                    "batch1": no_external,
                    "batch50": no_external,
                },
            },
        ]

        result = _coarse_timing_frontier_sensitivity(rows)

        self.assertEqual(result["observed_frontier_condition_ids"], ["interfered_observed_frontier"])
        self.assertEqual(
            set(result["batch1_boundary_unflagged_frontier_condition_ids"]),
            {"clean_low_success", "clean_high_success"},
        )
        self.assertEqual(
            result["interference_status_counts"]["batch1"],
            {no_external: 2, "interference_flagged": 1},
        )
        selected = {
            row["condition_id"]: row
            for row in result["selected_frontier_conditions"]
        }
        self.assertEqual(
            set(selected),
            {"interfered_observed_frontier", "clean_low_success", "clean_high_success"},
        )
        self.assertTrue(selected["interfered_observed_frontier"]["isolated_retest_recommended"])
        self.assertFalse(selected["clean_high_success"]["isolated_retest_recommended"])

    def test_timing_interference_missing_snapshot_is_not_treated_as_clear(self):
        self.assertEqual(_timing_interference_status(None), "missing_snapshot")
        self.assertEqual(_timing_interference_status({}), "missing_snapshot")
        self.assertEqual(
            _timing_interference_status(
                {"gpu_interference": {"status": "interference_flagged"}}
            ),
            "interference_flagged",
        )

    def test_interfered_fine_window_retries_both_batches_and_clean_window_does_not(self):
        no_external = "no_external_compute_observed_at_boundaries"
        self.assertIsNone(_fine_timing_retry_status(None))
        self.assertEqual(
            _fine_timing_retry_status({"batch1": {"gpu_interference": {"status": no_external}}}),
            {"batch1": no_external, "batch50": "missing_snapshot"},
        )
        self.assertIsNone(
            _fine_timing_retry_status(
                {
                    "batch1": {"gpu_interference": {"status": no_external}},
                    "batch50": {"gpu_interference": {"status": no_external}},
                }
            )
        )

    def test_timing_staging_merge_is_idempotent_and_preserves_repeats(self):
        clear = "no_external_compute_observed_at_boundaries"

        def timing_record(batch50_p50, batch50_status):
            return {
                "condition_id": "a" * 20,
                "condition_identity": {"task": "cube", "identity": "same"},
                "spec": {"task": "cube", "group": "A1"},
                "coarse_timing": {
                    "warmup": 5,
                    "runs": 10,
                    "batch1": {
                        "runs": 10,
                        "p50_seconds": 0.1,
                        "gpu_interference": {"status": clear},
                    },
                    "batch50": {
                        "runs": 10,
                        "p50_seconds": batch50_p50,
                        "gpu_interference": {"status": batch50_status},
                    },
                },
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            staging.mkdir()
            source = staging / f"{'a' * 20}.json"
            source.write_text(json.dumps(timing_record(1.0, clear)), encoding="utf-8")

            copied = merge_timing_staging_roots(
                [staging], output_root=root / "output"
            )
            self.assertEqual(copied["totals"]["copied"], 1)

            source.write_text(
                json.dumps(timing_record(9.0, "interference_flagged")),
                encoding="utf-8",
            )
            repeated = merge_timing_staging_roots(
                [staging], output_root=root / "output"
            )
            self.assertEqual(repeated["totals"]["repeated"], 1)
            target = root / "output" / "analysis" / "timing_conditions" / source.name
            merged = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(merged["coarse_timing"]["batch50"]["p50_seconds"], 1.0)
            self.assertEqual(
                merged["coarse_timing"]["batch50"]["gpu_interference"]["status"],
                "interference_flagged",
            )
            self.assertEqual(len(merged["coarse_timing_repeats"]), 1)
            self.assertEqual(
                merged["coarse_timing_repeats"][0]["coarse_timing"]["batch50"]["p50_seconds"],
                9.0,
            )

            duplicate = merge_timing_staging_roots(
                [staging], output_root=root / "output"
            )
            self.assertEqual(duplicate["totals"]["identical"], 1)
            merged_again = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(len(merged_again["coarse_timing_repeats"]), 1)

    def test_timing_staging_merge_accepts_output_root_parent(self):
        condition_id = "c" * 20
        record = {
            "condition_id": condition_id,
            "condition_identity": {"task": "cube", "identity": "same"},
            "coarse_timing": {
                "warmup": 5,
                "runs": 10,
                "batch1": {"runs": 10, "p50_seconds": 0.1},
                "batch50": {"runs": 10, "p50_seconds": 1.0},
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging_output = root / "staging_run"
            timing_conditions = staging_output / "analysis" / "timing_conditions"
            timing_conditions.mkdir(parents=True)
            (timing_conditions / f"{condition_id}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )

            merged = merge_timing_staging_roots(
                [staging_output], output_root=root / "output"
            )

            self.assertEqual(merged["totals"]["copied"], 1)
            self.assertIn(str(timing_conditions.resolve()), merged["staging_roots"])

    def test_timing_staging_merge_rejects_condition_identity_conflicts(self):
        record = {
            "condition_id": "b" * 20,
            "condition_identity": {"task": "cube", "identity": "source"},
            "coarse_timing": {
                "warmup": 5,
                "runs": 10,
                "batch1": {"runs": 10},
                "batch50": {"runs": 10},
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            staging.mkdir()
            target_root = root / "output" / "analysis" / "timing_conditions"
            target_root.mkdir(parents=True)
            target = target_root / f"{'b' * 20}.json"
            target.write_text(
                json.dumps({**record, "condition_identity": {"task": "cube", "identity": "other"}}),
                encoding="utf-8",
            )
            (staging / target.name).write_text(json.dumps(record), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "condition identity conflict"):
                merge_timing_staging_roots([staging], output_root=root / "output")

    def test_coarse_timing_completeness_enforces_planned_window_sizes(self):
        record = {
            "coarse_timing": {
                "warmup": 5,
                "runs": 10,
                "batch1": {"runs": 10},
                "batch50": {"runs": 10},
            }
        }
        self.assertTrue(_complete_coarse_timing(record))
        self.assertFalse(
            _complete_coarse_timing(
                {"coarse_timing": {**record["coarse_timing"], "warmup": 4}}
            )
        )
        self.assertFalse(
            _complete_coarse_timing(
                {"coarse_timing": {**record["coarse_timing"], "runs": 9}}
            )
        )
        self.assertFalse(
            _complete_coarse_timing(
                {
                    "coarse_timing": {
                        **record["coarse_timing"],
                        "batch50": {"runs": 9},
                    }
                }
            )
        )

    def test_fine_timing_completeness_enforces_planned_window_sizes(self):
        record = {
            "fine_timing": {
                "warmup": 20,
                "runs": 100,
                "batch1": {"runs": 100, "samples_seconds": [0.1] * 100},
                "batch50": {"runs": 100, "samples_seconds": [1.0] * 100},
            }
        }
        self.assertTrue(_complete_fine_timing(record))
        self.assertFalse(
            _complete_fine_timing(
                {"fine_timing": {**record["fine_timing"], "warmup": 19}}
            )
        )
        self.assertFalse(
            _complete_fine_timing(
                {
                    "fine_timing": {
                        **record["fine_timing"],
                        "batch50": {"runs": 99, "samples_seconds": [1.0] * 99},
                    }
                }
            )
        )

    def test_fine_timing_staging_merge_preserves_coarse_and_is_idempotent(self):
        condition_id_value = "d" * 20
        coarse = {
            "warmup": 5,
            "runs": 10,
            "batch1": {"runs": 10, "p50_seconds": 0.1},
            "batch50": {"runs": 10, "p50_seconds": 1.0},
        }
        fine = {
            "warmup": 20,
            "runs": 100,
            "batch1": {"runs": 100, "samples_seconds": [0.2] * 100},
            "batch50": {"runs": 100, "samples_seconds": [2.0] * 100},
        }
        record = {
            "condition_id": condition_id_value,
            "condition_identity": {"task": "cube", "identity": "same"},
            "spec": {"task": "cube", "group": "A1"},
            "coarse_timing": coarse,
            "fine_timing": fine,
            "fine_timing_retries": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging_run"
            stage_timing = staging / "analysis" / "timing_conditions"
            stage_timing.mkdir(parents=True)
            output_timing = root / "output" / "analysis" / "timing_conditions"
            output_timing.mkdir(parents=True)
            target = output_timing / f"{condition_id_value}.json"
            target.write_text(
                json.dumps({**record, "fine_timing": None}), encoding="utf-8"
            )
            (stage_timing / target.name).write_text(
                json.dumps(record), encoding="utf-8"
            )

            merged = merge_fine_timing_staging_roots(
                [staging], output_root=root / "output"
            )
            self.assertEqual(merged["totals"]["copied"], 1)
            merged_record = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(merged_record["coarse_timing"], coarse)
            self.assertEqual(merged_record["fine_timing"], fine)

            repeated = merge_fine_timing_staging_roots(
                [staging], output_root=root / "output"
            )
            self.assertEqual(repeated["totals"]["identical"], 1)

    def test_fine_timing_staging_merge_rejects_changed_coarse_timing(self):
        condition_id_value = "e" * 20
        coarse = {
            "warmup": 5,
            "runs": 10,
            "batch1": {"runs": 10},
            "batch50": {"runs": 10},
        }
        fine = {
            "warmup": 20,
            "runs": 100,
            "batch1": {"runs": 100},
            "batch50": {"runs": 100},
        }
        record = {
            "condition_id": condition_id_value,
            "condition_identity": {"task": "cube", "identity": "same"},
            "coarse_timing": coarse,
            "fine_timing": fine,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            staging.mkdir()
            target_root = root / "output" / "analysis" / "timing_conditions"
            target_root.mkdir(parents=True)
            target = target_root / f"{condition_id_value}.json"
            changed_coarse = {
                **coarse,
                "batch50": {"runs": 10, "p50_seconds": 2.0},
            }
            target.write_text(
                json.dumps({**record, "coarse_timing": changed_coarse}),
                encoding="utf-8",
            )
            (staging / target.name).write_text(json.dumps(record), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "coarse timing conflict"):
                merge_fine_timing_staging_roots(
                    [staging], output_root=root / "output"
                )

    def test_fixed_stability_anchors_map_to_primary_and_shared_method_configs(self):
        primary = primary_condition_specs()
        primary_timing_keys = {_timing_config_key(spec) for spec in primary}
        fixed = sampling_stability_specs()
        self.assertTrue(all(_timing_config_key(spec) in primary_timing_keys for spec in fixed))
        self.assertEqual(len({_shared_method_key(spec) for spec in fixed}), 12)

    def test_host_preflight_checks_load_and_memory_headroom(self):
        with patch("scripts.round5_phase1_5.os.cpu_count", return_value=8), patch(
            "scripts.round5_phase1_5.os.getloadavg", return_value=(2.0, 1.5, 1.0)
        ), patch(
            "scripts.round5_phase1_5._meminfo_mib",
            return_value={"MemTotal": 1024, "MemAvailable": 512},
        ):
            snapshot = _host_preflight(max_load_per_cpu=0.5, minimum_available_mib=256)
        self.assertEqual(snapshot["cpu_count"], 8)
        self.assertEqual(snapshot["memory_available_mib"], 512)
        self.assertAlmostEqual(snapshot["load_per_cpu"], 0.25)

        with patch("scripts.round5_phase1_5.os.cpu_count", return_value=2), patch(
            "scripts.round5_phase1_5.os.getloadavg", return_value=(2.0, 1.5, 1.0)
        ), patch(
            "scripts.round5_phase1_5._meminfo_mib",
            return_value={"MemTotal": 1024, "MemAvailable": 512},
        ), self.assertRaisesRegex(RuntimeError, "CPU load"):
            _host_preflight(max_load_per_cpu=0.5, minimum_available_mib=256)

        with patch("scripts.round5_phase1_5.os.cpu_count", return_value=8), patch(
            "scripts.round5_phase1_5.os.getloadavg", return_value=(0.5, 0.5, 0.5)
        ), patch(
            "scripts.round5_phase1_5._meminfo_mib",
            return_value={"MemTotal": 1024, "MemAvailable": 128},
        ), self.assertRaisesRegex(RuntimeError, "memory headroom"):
            _host_preflight(max_load_per_cpu=0.5, minimum_available_mib=256)

        with patch("scripts.round5_phase1_5.os.cpu_count", return_value=8), patch(
            "scripts.round5_phase1_5.os.getloadavg", return_value=(0.5, 0.5, 0.5)
        ), patch(
            "scripts.round5_phase1_5._meminfo_mib",
            return_value={"MemTotal": 1024, "MemAvailable": 512, "SwapFree": 128},
        ), self.assertRaisesRegex(RuntimeError, "swap headroom"):
            _host_preflight(
                max_load_per_cpu=0.5,
                minimum_available_mib=256,
                minimum_swap_free_mib=256,
            )


class Round5Phase15DiagnosticsTests(unittest.TestCase):
    def test_guidance_milestone_summary_separates_steps_and_unpaired_rows(self):
        summary = phase15_diagnostics.guidance_milestone_step_summary(
            [
                {
                    "guidance": "post_opt",
                    "paired_outcome_valid": True,
                    "physical_comparison_step": 20,
                },
                {
                    "guidance": "post_opt",
                    "paired_outcome_valid": True,
                    "physical_comparison_step": 25,
                },
                {"guidance": "post_opt", "paired_outcome_valid": False},
                {
                    "guidance": "guided_flow",
                    "paired_outcome_valid": True,
                    "physical_comparison_step": 10,
                },
                {"guidance": "guided_flow", "paired_outcome_valid": True},
            ]
        )

        self.assertEqual(
            summary["post_opt"],
            {
                "source_records": 3,
                "paired_records": 2,
                "unpaired_records": 1,
                "paired_records_missing_step": 0,
                "step_counts": {"5": 0, "10": 0, "15": 0, "20": 1, "25": 1},
                "unexpected_step_counts": {},
            },
        )
        self.assertEqual(summary["guided_flow"]["step_counts"]["10"], 1)
        self.assertEqual(summary["guided_flow"]["paired_records_missing_step"], 1)

    def assert_tree_allclose(self, left, right):
        if isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_allclose(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for left_item, right_item in zip(left, right):
                self.assert_tree_allclose(left_item, right_item)
        elif hasattr(left, "detach"):
            np.testing.assert_allclose(
                left.detach().cpu().numpy(),
                right.detach().cpu().numpy(),
                rtol=0.0,
                atol=1e-6,
            )
        elif isinstance(left, (np.ndarray, np.number, int, float, bool)):
            np.testing.assert_allclose(
                np.asarray(left), np.asarray(right), rtol=0.0, atol=1e-6
            )
        else:
            self.assertEqual(left, right)

    def test_fixed_candidate_policy_replays_then_holds_zero(self):
        class FakeEnv:
            num_envs = 2
            action_space = gym.spaces.Box(-1.0, 1.0, shape=(2, 2))

        processor = StandardScaler().fit(
            np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        )
        policy = FixedCandidatePolicy(
            np.asarray(
                [
                    [[-0.5, 0.5], [0.25, -0.25]],
                    [[0.1, 0.2], [0.3, 0.4]],
                ]
            ),
            process={"action": processor},
            action_block=1,
        )
        policy.set_env(FakeEnv())
        info = {"terminated": np.array([False, False]), "truncated": np.array([False, False])}
        first = policy.get_action(info)
        second = policy.get_action(info)
        third = policy.get_action(info)
        self.assertEqual(first.shape, (2, 2))
        np.testing.assert_allclose(first, processor.inverse_transform([[-0.5, 0.5], [0.1, 0.2]]))
        np.testing.assert_allclose(second, processor.inverse_transform([[0.25, -0.25], [0.3, 0.4]]))
        np.testing.assert_array_equal(third, np.zeros((2, 2), dtype=np.float32))

    def test_vector_step_pixel_batch_is_read_from_infos(self):
        pixels = np.zeros((2, 1, 8, 8, 3), dtype=np.uint8)
        result = (
            None,
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=bool),
            np.zeros(2, dtype=bool),
            {"pixels": pixels},
        )
        self.assertIs(_extract_step_pixel_batch(result, 2), pixels)
        terminal = (result[0], result[1], np.array([False, True]), result[3], result[4])
        self.assertEqual(_terminal_env_slots(terminal, 2), [1])

    def test_environment_state_snapshot_round_trips(self):
        class FakeEnv:
            def __init__(self):
                self.value = np.array([1.0, 2.0])

            def get_state(self):
                return self.value.copy()

            def set_state(self, state):
                self.value = np.asarray(state, dtype=np.float64).copy()

        env = FakeEnv()
        snapshot = capture_environment_state(env)
        env.value[:] = 9.0
        restore_environment_state(env, snapshot)
        np.testing.assert_array_equal(env.value, [1.0, 2.0])

    def test_dm_control_environment_state_and_rng_replay(self):
        env = gym.make("swm/ReacherDMControl-v0", task="qpos_match")
        try:
            env.reset(seed=42)
            target = env.unwrapped
            snapshot = capture_environment_state(env)
            rng_snapshot = capture_rng_state(env)
            task_random = target.dmc_env._env.task.random
            expected_random = task_random.rand()
            restore_rng_state(env, rng_snapshot)
            self.assertEqual(task_random.rand(), expected_random)

            action = np.asarray(env.action_space.low) + 0.67 * (
                np.asarray(env.action_space.high)
                - np.asarray(env.action_space.low)
            )
            action = action.astype(env.action_space.dtype)
            first = env.step(action)
            restore_environment_state(env, snapshot)
            restore_rng_state(env, rng_snapshot)
            second = env.step(action)
            np.testing.assert_allclose(first[0], second[0], rtol=0.0, atol=1e-7)
            self.assertAlmostEqual(first[1], second[1], places=7)
            self.assertEqual(first[2:4], second[2:4])
            for key in ("qpos", "qvel"):
                np.testing.assert_allclose(
                    first[4][key], second[4][key], rtol=0.0, atol=1e-7
                )
        finally:
            env.close()

    def test_pymunk_and_navigation_states_restore_across_environments(self):
        for task, expected_kind in (
            ("pusht", "pymunk_body_state"),
            ("tworoom", "navigation_position_state"),
        ):
            with self.subTest(task=task):
                cfg = compose_eval_config(
                    task,
                    overrides=["eval.num_eval=1", "output.save_video=false"],
                )
                world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
                world_cfg["max_episode_steps"] = 100
                first_world = swm.World(**world_cfg, add_pixels=False)
                second_world = swm.World(**world_cfg, add_pixels=False)
                try:
                    first_env = first_world.envs.envs[0]
                    second_env = second_world.envs.envs[0]
                    first_env.reset(seed=42)
                    second_env.reset(seed=42)
                    state = capture_environment_state(first_env)
                    self.assertEqual(state["kind"], expected_kind)
                    rng_state = capture_rng_state(first_env)
                    action = np.asarray(first_env.action_space.low) + 0.67 * (
                        np.asarray(first_env.action_space.high)
                        - np.asarray(first_env.action_space.low)
                    )
                    action = action.astype(first_env.action_space.dtype)
                    first = first_env.step(action)
                    restore_environment_state(second_env, state)
                    restore_rng_state(second_env, rng_state)
                    second = second_env.step(action)
                    self.assert_tree_allclose(first[0], second[0])
                    self.assert_tree_allclose(first[1:4], second[1:4])
                    self.assert_tree_allclose(first[4], second[4])
                finally:
                    first_world.close()
                    second_world.close()

    def test_pusht_state_restore_survives_variation_resampling(self):
        cfg = compose_eval_config(
            "pusht", overrides=["eval.num_eval=1", "output.save_video=false"]
        )
        world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
        world_cfg["max_episode_steps"] = 100
        world = swm.World(**world_cfg, add_pixels=False)
        try:
            env = world.envs.envs[0]
            for _ in range(3):
                env.reset(seed=42)
                snapshot = capture_environment_state(env)
                rng_snapshot = capture_rng_state(env)
                action = np.asarray(env.action_space.low) + 0.67 * (
                    np.asarray(env.action_space.high)
                    - np.asarray(env.action_space.low)
                )
                action = action.astype(env.action_space.dtype)
                expected = env.step(action)

                env.reset()
                reset_snapshot = capture_environment_state(env)
                self.assertFalse(
                    np.array_equal(
                        snapshot["variations"]["agent"]["start_position"],
                        reset_snapshot["variations"]["agent"]["start_position"],
                    )
                )
                restore_environment_state(env, snapshot)
                restore_rng_state(env, rng_snapshot)
                actual = env.step(action)

                self.assert_tree_allclose(expected[0], actual[0])
                self.assert_tree_allclose(expected[1:4], actual[1:4])
                expected_info = {
                    key: value for key, value in expected[4].items() if key != "id"
                }
                actual_info = {
                    key: value for key, value in actual[4].items() if key != "id"
                }
                self.assert_tree_allclose(expected_info, actual_info)
        finally:
            world.close()

    def test_probe_feature_collection_batches_transform_and_restores_sampler_order(self):
        class RowOnlyArray:
            def __init__(self, values):
                self.values = np.asarray(values)

            def __getitem__(self, rows):
                if isinstance(rows, slice):
                    raise AssertionError("probe feature extraction must not read a contiguous frame span")
                return self.values[rows]

        class FakeDataset:
            column_names = ["episode_idx"]

            def __init__(self):
                pixels = [
                    np.full((224, 224, 3), value, dtype=np.uint8)
                    for value in (0, 32, 64, 96, 128)
                ]
                self.h5_file = {
                    "pixels": RowOnlyArray(np.stack(pixels)),
                    "target": RowOnlyArray(np.arange(5, dtype=np.float64)[:, None]),
                }

            def get_col_data(self, key):
                if key != "episode_idx":
                    raise AssertionError(f"unexpected column: {key}")
                return np.array([0, 1, 1, 2, 2])

        class FakeModel(torch.nn.Module):
            def encode_pixels(self, batch):
                if batch.ndim != 4 or batch.shape[1] != 3:
                    raise AssertionError(f"expected batched BCHW images, got {tuple(batch.shape)}")
                return batch.mean(dim=(2, 3))

        dataset = FakeDataset()
        transform = img_transform(compose_eval_config("cube"))
        rows = [4, 1, 3, 0]
        features, targets = _collect_features(
            dataset,
            FakeModel(),
            transform,
            rows=rows,
            target_column="target",
            device=torch.device("cpu"),
            batch_size=2,
        )

        expected = torch.stack([transform(dataset.h5_file["pixels"][row]) for row in rows])
        expected = expected.mean(dim=(2, 3)).numpy()
        np.testing.assert_allclose(features, expected, rtol=0, atol=1e-6)
        self.assertEqual(features.dtype, np.float32)
        np.testing.assert_array_equal(targets[:, 0], [4.0, 1.0, 3.0, 0.0])

    def test_candidate_probe_distinguishes_real_and_predicted_future_latents(self):
        class IdentityScaler:
            def transform(self, value):
                return np.asarray(value)

            def inverse_transform(self, value):
                return np.asarray(value)

        class FirstThreeFeatures:
            def predict(self, value):
                return np.asarray(value)[:, :3]

        metrics = _probe_candidate_predictions(
            [
                {
                    "predicted_future_latent": [1.0, 0.0, 0.0],
                    "milestones": {
                        "5": {
                            "future_latent": [0.0, 0.0, 0.0],
                            "current": [0.0, 0.0, 0.0],
                        }
                    },
                }
            ],
            task="cube",
            ridge={
                "model": FirstThreeFeatures(),
                "x_scaler": IdentityScaler(),
                "y_scaler": IdentityScaler(),
            },
        )
        self.assertEqual(metrics["real_future_latent"]["mae"], 0.0)
        self.assertAlmostEqual(metrics["predicted_future_latent"]["mae"], 1.0 / 3.0)

    def test_ridge_probe_keeps_large_feature_matrices_in_float32(self):
        rng = np.random.default_rng(9)
        features = rng.normal(size=(64, 12)).astype(np.float64)
        targets = np.stack(
            (features[:, 0] - 0.5 * features[:, 1], 2.0 * features[:, 2]),
            axis=1,
        )
        trajectory_ids = np.repeat(np.arange(8), 8)
        result = fit_ridge_probe(
            features,
            targets,
            trajectory_ids,
            split={"train": tuple(range(6)), "validation": (6, 7)},
            alphas=(1e-4, 1e-3, 1e-2),
        )
        self.assertEqual(result["model"].coef_.dtype, np.float32)
        self.assertEqual(result["validation_metrics"]["count"], 16)

    def test_action_normalization_round_trip_through_fixed_candidate_policy(self):
        class FakeEnv:
            num_envs = 2
            action_space = gym.spaces.Box(-10.0, 10.0, shape=(2, 2))

        physical = np.array([[0.2, -0.4], [1.5, 2.0]], dtype=np.float64)
        processor = StandardScaler().fit(
            np.array([[-2.0, -1.0], [0.0, 0.0], [2.0, 3.0]])
        )
        normalized = processor.transform(physical)
        policy = FixedCandidatePolicy(
            normalized.reshape(2, 1, 2),
            process={"action": processor},
            action_block=1,
        )
        policy.set_env(FakeEnv())

        actual = policy.get_action(
            {"terminated": np.array([False, False]), "truncated": np.array([False, False])}
        )

        np.testing.assert_allclose(actual, physical, rtol=0, atol=1e-7)

    def test_guided_flow_one_step_matches_post_optimization(self):
        from source.model.fast_lewam.jepa import FastLeWAM

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = FastLeWAM(
                encoder=torch.nn.Identity(),
                projector=torch.nn.Identity(),
                latent_dim=8,
                action_dim=2,
                action_horizon=5,
                latent_head_dim=8,
                latent_head_layers=1,
                heads=2,
                mlp_dim=16,
                inference_steps=1,
                stage_a_goal_injection="token",
            ).eval()
            z0 = torch.randn(2, 8)
            goal = torch.randn(2, 8)
            noise = torch.randn(2, 5, 2)

            common = {
                "noise": noise,
                "num_steps": 1,
                "goal_latent": goal,
                "guidance_step_size": 0.01,
                "guidance_last_steps": 1,
                "guidance_inner_steps": 1,
                "guidance_max_rms_offset": 0.2,
            }
            post_opt = model.sample_actions(z0, guidance_mode="post_opt", **common)
            guided_flow = model.sample_actions(z0, guidance_mode="guided_flow", **common)

        torch.testing.assert_close(guided_flow, post_opt, rtol=1e-6, atol=1e-7)

    def test_constant_correlation_is_undefined(self):
        self.assertIsNone(safe_correlation([1, 1, 1], [1, 2, 3]))

    def test_controls_are_deterministic_and_deduplicatable(self):
        anchors = np.zeros((3, 25, 2), dtype=np.float64)
        first, first_meta = make_control_actions(anchors, seed=7)
        second, second_meta = make_control_actions(anchors, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_meta, second_meta)
        unique, duplicates = deduplicate_actions(first)
        self.assertEqual(len(unique) + len(duplicates), len(first))
        self.assertGreater(len(duplicates), 0)

    def test_control_pool_reblocks_candidate_anchors_as_primitive_sequences(self):
        class IdentityProcessor:
            def transform(self, value):
                return np.asarray(value, dtype=np.float64)

            def inverse_transform(self, value):
                return np.asarray(value, dtype=np.float64)

        class FakeDataset:
            column_names = ("episode_idx",)

            def __init__(self, actions):
                self.actions = np.asarray(actions, dtype=np.float64)

            def get_col_data(self, column):
                if column != "episode_idx":
                    raise AssertionError(f"unexpected column: {column}")
                return np.full(len(self.actions), 7)

            def get_row_data(self, indices):
                return {"action": self.actions[np.asarray(indices, dtype=np.int64)]}

        data_actions = np.arange(50, dtype=np.float64).reshape(25, 2)
        manifest = SimpleNamespace(
            entries=(
                SimpleNamespace(row_index=0, episode_id=7, start_step=0),
            )
        )
        proposals = [
            {
                "flow_steps": flow_steps,
                "candidate_index": 0,
                "slot": 0,
                "action": (data_actions + flow_steps).reshape(5, 10).tolist(),
            }
            for flow_steps in (1, 2)
        ]

        controls, metadata = _control_pool_actions(
            dataset=FakeDataset(data_actions),
            manifest=manifest,
            process={"action": IdentityProcessor()},
            proposals=proposals,
            seed=7,
            action_block=5,
        )

        self.assertEqual(controls.shape, (222, 1, 25, 2))
        self.assertEqual([item["kind"] for item in metadata[:3]], ["anchor"] * 3)
        np.testing.assert_array_equal(
            controls[:3, 0],
            np.stack((data_actions, data_actions + 1, data_actions + 2)),
        )
        packed = _pack_primitive_actions_for_replay(
            controls[:3, 0], action_block=5, horizon=5
        )
        self.assertEqual(packed.shape, (3, 5, 10))
        policy = FixedCandidatePolicy(
            packed,
            process={"action": IdentityProcessor()},
            action_block=5,
        )
        np.testing.assert_array_equal(policy._physical_actions, controls[:3, 0])

    def test_control_pool_has_the_planned_222_actions_and_provenance(self):
        anchors = np.zeros((3, 25, 2), dtype=np.float64)
        actions, metadata = make_control_actions(anchors, seed=7)
        self.assertEqual(actions.shape, (222, 25, 2))
        self.assertEqual(len(metadata), 222)
        self.assertEqual(sum(item["kind"] == "anchor" for item in metadata), 3)
        self.assertEqual(sum(item["kind"] == "rms_perturbation" for item in metadata), 144)
        self.assertEqual(sum(item["kind"] == "block_transform" for item in metadata), 9)
        self.assertEqual(sum(item["kind"] == "standard_gaussian" for item in metadata), 64)
        self.assertEqual(sum(item["kind"] == "physical_zero" for item in metadata), 1)
        self.assertEqual(sum(item["kind"] == "normalized_zero" for item in metadata), 1)
        self.assertTrue(all("direction" in item for item in metadata if item["kind"] == "rms_perturbation"))
        self.assertTrue(all("permutation" in item for item in metadata if item["kind"] == "block_transform"))
        transforms_by_anchor = {
            anchor: sorted(item["permutation"] for item in metadata if item["kind"] == "block_transform" and item["anchor"] == anchor)
            for anchor in range(3)
        }
        self.assertEqual(transforms_by_anchor, {0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2]})

    def test_controls_only_reuses_only_a_complete_matching_222_action_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            task_root = Path(temporary) / "diagnostics" / "candidate_pool" / "cube"
            task_root.mkdir(parents=True)
            records_path = task_root / "records.jsonl"
            controls_path = task_root / "controls.jsonl"
            records_path.write_text("{}\n{}\n", encoding="utf-8")
            controls_path.write_text("{}\n" * 222, encoding="utf-8")
            checkpoint = Path(temporary) / "checkpoint.pt"
            cohort = SimpleNamespace(
                cohort_id="legacy-cohort",
                computed_sha256="cohort-sha256",
            )
            payload = {
                "schema_version": "round5_phase1_5_candidate_pool_manifest_v1",
                "status": "completed",
                "task": "cube",
                "flow_steps": [1, 2],
                "candidates_per_flow_step": 1,
                "executed_candidates_per_flow_step": 1,
                "states": 1,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": "checkpoint-sha256",
                "cohort_id": cohort.cohort_id,
                "cohort_sha256": cohort.computed_sha256,
                "control_actions": 222,
                "control_protocol_version": "phase1_5_fixed_pool_controls_v1",
                "control_status": "completed",
                "records": str(records_path.resolve()),
                "controls_records": str(controls_path.resolve()),
            }
            manifest_path = task_root / "manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            identity = {
                "task": "cube",
                "flow_steps": (1, 2),
                "proposal_count": 1,
                "expected_states": 1,
                "checkpoint": checkpoint,
                "checkpoint_sha256": "checkpoint-sha256",
                "cohort": cohort,
            }

            self.assertTrue(_current_control_artifacts_match(task_root, **identity))

            payload["control_actions"] = 240
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(_current_control_artifacts_match(task_root, **identity))

            payload["control_actions"] = 222
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            controls_path.write_text("{}\n" * 221, encoding="utf-8")
            self.assertFalse(_current_control_artifacts_match(task_root, **identity))

    def test_control_metrics_keep_zero_action_kinds_separate(self):
        result = control_action_metrics(
            [
                {"control_kind": "physical_zero", "state_id": "a", "true_distance": 1.0, "success": False},
                {"control_kind": "normalized_zero", "state_id": "a", "true_distance": 2.0, "success": True},
            ]
        )
        self.assertTrue(result["physical_zero_is_separate"])
        self.assertTrue(result["normalized_zero_is_separate"])
        self.assertEqual(set(result["kinds"]), {"physical_zero", "normalized_zero"})

    def test_controls_keep_physical_and_normalized_zero_distinct(self):
        anchors = np.zeros((1, 5, 2), dtype=np.float64)
        physical_zero = np.full((5, 2), 0.25, dtype=np.float64)
        actions, metadata = make_control_actions(
            anchors,
            seed=7,
            physical_zero=physical_zero,
        )
        physical_indices = [index for index, item in enumerate(metadata) if item["kind"] == "physical_zero"]
        normalized_indices = [index for index, item in enumerate(metadata) if item["kind"] == "normalized_zero"]
        self.assertEqual(len(physical_indices), 1)
        self.assertEqual(len(normalized_indices), 1)
        np.testing.assert_array_equal(actions[physical_indices[0]], physical_zero)
        np.testing.assert_array_equal(actions[normalized_indices[0]], np.zeros_like(physical_zero))

    def test_task_normalized_distance_uses_plan_thresholds(self):
        self.assertAlmostEqual(normalized_physical_distance("cube", [0.04, 0], [0, 0]), 1.0)
        self.assertAlmostEqual(normalized_physical_distance("reacher", [0.05, 0], [0, 0]), 1.0)
        self.assertAlmostEqual(normalized_physical_distance("tworoom", [16, 0], [0, 0]), 1.0)
        self.assertAlmostEqual(
            normalized_physical_distance("pusht", [20, 0, 0, 0, 0], [0, 0, 0, 0, 0]),
            1.0,
        )

    def test_selection_reports_oracle_and_selection_regret(self):
        records = [
            {"state_id": "a", "predicted_cost": 0.2, "true_distance": 2.0, "physical_state": [2, 0], "goal_state": [0, 0], "milestones": {"25": {"future_latent_cost": 2.0}}},
            {"state_id": "a", "predicted_cost": 0.1, "true_distance": 1.0, "physical_state": [1, 0], "goal_state": [0, 0], "milestones": {"25": {"future_latent_cost": 1.0}}},
            {"state_id": "b", "predicted_cost": 0.1, "true_distance": 3.0, "physical_state": [3, 0], "goal_state": [0, 0], "milestones": {"25": {"future_latent_cost": 3.0}}},
            {"state_id": "b", "predicted_cost": 0.2, "true_distance": 1.0, "physical_state": [1, 0], "goal_state": [0, 0], "milestones": {"25": {"future_latent_cost": 1.0}}},
        ]
        result = candidate_selection_metrics(records, task="tworoom", random_draws=8)
        self.assertEqual(result["states"], 2)
        self.assertAlmostEqual(result["selection_regret"], 1.0)
        self.assertEqual(result["predicted_true_distance_correlation"], 0.0)
        self.assertAlmostEqual(result["selected_latent_cost"], 2.0)

    def test_candidate_pool_bootstraps_paired_selection_gain_by_state(self):
        records = []
        for flow_steps in (1, 2):
            for state_id, selected_success in (("a", True), ("b", False)):
                records.extend(
                    [
                        {
                            "state_id": state_id,
                            "flow_steps": flow_steps,
                            "candidate_index": 0,
                            "predicted_cost": 0.0,
                            "true_distance": 0.0 if selected_success else 2.0,
                            "success": selected_success,
                            "action": [[0.0]],
                        },
                        {
                            "state_id": state_id,
                            "flow_steps": flow_steps,
                            "candidate_index": 1,
                            "predicted_cost": 1.0,
                            "true_distance": 2.0 if selected_success else 0.0,
                            "success": not selected_success,
                            "action": [[1.0]],
                        },
                    ]
                )
        result = candidate_pool_metrics(
            records,
            task="tworoom",
            flow_steps=(1, 2),
            candidates_per_flow_step=2,
            random_draws=64,
            bootstrap_samples=32,
        )
        paired = result["bootstrap"]["selected_minus_random_success_rate"]
        regret = result["bootstrap"]["selection_regret"]
        self.assertEqual(paired["cluster_count"], 2)
        self.assertEqual(paired["row_count"], 4)
        self.assertEqual(regret["cluster_count"], 2)
        self.assertEqual(regret["row_count"], 4)
        self.assertAlmostEqual(
            result["selected_minus_random_success_rate"], paired["estimate"]
        )
        self.assertAlmostEqual(result["selection_regret"], regret["estimate"])
        for metrics in result["by_flow_steps"].values():
            flow_bootstrap = metrics["bootstrap"]["selected_minus_random_success_rate"]
            self.assertEqual(flow_bootstrap["cluster_count"], 2)
            self.assertEqual(flow_bootstrap["row_count"], 2)

    def test_p3_decision_requires_paired_state_cluster_interval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            diagnostics = root / "diagnostics"
            diagnostics.mkdir()
            for task in ("cube", "pusht", "reacher", "tworoom"):
                (diagnostics / f"summary_{task}.json").write_text(
                    json.dumps(
                        {
                            "candidate_pool": {
                                "selection_regret": 0.25,
                                "selected_minus_random_success_rate": 0.1,
                                "bootstrap": {
                                    "selected_minus_random_success_rate": {
                                        "ci95": [0.02, 0.18]
                                    }
                                },
                            }
                        }
                    ),
                    encoding="utf-8",
                )
            decisions = _diagnostic_decisions(root, [])
        self.assertIn("4/4 个任务的状态聚类 95% CI 下界大于 0", decisions[0])
        self.assertIn("oracle−B-selected 归一化距离 regret 为 0.250", decisions[0])
        self.assertIn("支持跨任务保留 P3", decisions[0])

    def test_incomplete_diagnostic_coverage_is_not_reported_as_four_task_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            diagnostics = root / "diagnostics"
            analysis = root / "analysis"
            diagnostics.mkdir()
            analysis.mkdir()
            (analysis / "method_selection.json").write_text(
                json.dumps(
                    {
                        "main_candidate_id": "p3-main",
                        "candidates": [
                            {
                                "candidate_id": "p3-main",
                                "config": {
                                    "family": "proposal_ranking",
                                    "mode": "P3",
                                    "flow_steps": 2,
                                    "candidate_count": 64,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            summaries = {
                "pusht": {"gain": 0.05, "ci95": [-0.01, 0.11], "regret": 0.5},
                "reacher": {"gain": 0.02, "ci95": [0.01, 0.03], "regret": 0.25},
                "tworoom": {"gain": 0.03, "ci95": [0.01, 0.05]},
            }
            for task, metrics in summaries.items():
                (diagnostics / f"summary_{task}.json").write_text(
                    json.dumps(
                        {
                            "candidate_pool": {
                                "selection_regret": metrics.get("regret"),
                                "selected_minus_random_success_rate": metrics["gain"],
                                "bootstrap": {
                                    "selected_minus_random_success_rate": {
                                        "ci95": metrics["ci95"]
                                    }
                                },
                            },
                            "guidance_by_variant": [
                                {
                                    "condition": {"guidance": mode},
                                    "paired_guidance": {"paired_advantage_mean": 0.1},
                                }
                                for mode in ("post_opt", "guided_flow")
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
            decisions = _diagnostic_decisions(root, [])

        self.assertIn("已完成 3/4 个任务的平均", decisions[0])
        self.assertIn("2/3 个已完成任务", decisions[0])
        self.assertIn("已有 2/4 个任务的平均 oracle−B-selected", decisions[0])
        self.assertIn("增益区间 3/4，regret 2/4，待补 cube, tworoom", decisions[0])
        self.assertIn("统一三 seed 成功率/精测延迟规则仍将 proposal_ranking/P3, S=2, N=64 选为工程主配置", decisions[0])
        self.assertIn("不等同于候选池诊断证明 P3 排序具有稳定的跨任务优势", decisions[0])
        self.assertNotIn("支持跨任务保留 P3", decisions[0])
        self.assertIn("下一轮 R4-AB 消融待诊断完成后确定", decisions[3])

    def test_coarse_timing_report_uses_completed_clean_fine_retests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis = root / "analysis"
            timing_root = analysis / "timing_conditions"
            timing_root.mkdir(parents=True)
            condition_id_value = "f" * 20
            (analysis / "fine_timing_selection.json").write_text(
                json.dumps(
                    {
                        "selected_conditions": [
                            {
                                "condition_id": condition_id_value,
                                "task": "cube",
                                "frontier_membership": ["observed"],
                                "isolated_retest_recommended": True,
                            }
                        ],
                        "fixed_anchor_count": 48,
                        "adaptive_anchor_count": 40,
                        "interference_suspect_frontier_condition_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            clear = "no_external_compute_observed_at_boundaries"
            (timing_root / f"{condition_id_value}.json").write_text(
                json.dumps(
                    {
                        "condition_id": condition_id_value,
                        "coarse_timing": {
                            "warmup": 5,
                            "runs": 10,
                            "batch1": {"runs": 10},
                            "batch50": {"runs": 10},
                        },
                        "fine_timing": {
                            "warmup": 20,
                            "runs": 100,
                            "batch1": {
                                "runs": 100,
                                "gpu_interference": {"status": clear},
                            },
                            "batch50": {
                                "runs": 100,
                                "gpu_interference": {"status": clear},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            lines = _coarse_timing_sensitivity_report_lines(root)

        self.assertIn("已完成隔离精测 1/1 个", lines[-1])
        self.assertIn("边界均未观察到外部计算 1/1 个", lines[-1])
        self.assertNotIn("精测时应重测", "\n".join(lines))

    def test_gf_decision_requires_stable_paired_benefit_or_all_task_timing_advantage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            diagnostics = root / "diagnostics"
            analysis = root / "analysis"
            diagnostics.mkdir()
            analysis.mkdir()
            for task in ("cube", "pusht", "reacher", "tworoom"):
                (diagnostics / f"summary_{task}.json").write_text(
                    json.dumps(
                        {
                            "matched_guidance_comparison": {
                                "gf_minus_po_physical_improvement_mean": 0.0,
                                "bootstrap": {
                                    "gf_minus_po_physical_improvement": {
                                        "ci95": [-0.1, 0.1]
                                    }
                                },
                            }
                        }
                    ),
                    encoding="utf-8",
                )
            method_path = analysis / "method_selection.json"
            method_path.write_text(
                json.dumps(
                    {
                        "guidance_mode_efficiency_comparisons": [
                            {"guided_flow_faster_all_tasks": False, "guided_flow_faster_task_count": 2}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            decisions = _diagnostic_decisions(root, [])
            self.assertIn("按计划从主方法中移除 GF", decisions[2])

            method_path.write_text(
                json.dumps(
                    {
                        "guidance_mode_efficiency_comparisons": [
                            {"guided_flow_faster_all_tasks": True, "guided_flow_faster_task_count": 4}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            decisions = _diagnostic_decisions(root, [])
            self.assertIn("支持作为效率方案保留 GF", decisions[2])

            for task in ("cube", "pusht", "reacher", "tworoom"):
                (diagnostics / f"summary_{task}.json").write_text(
                    json.dumps(
                        {
                            "matched_guidance_comparison": {
                                "gf_minus_po_physical_improvement_mean": 0.1,
                                "bootstrap": {
                                    "gf_minus_po_physical_improvement": {
                                        "ci95": [0.01, 0.2]
                                    }
                                },
                            }
                        }
                    ),
                    encoding="utf-8",
                )
            decisions = _diagnostic_decisions(root, [])
            self.assertIn("支持稳定真实收益，保留 GF", decisions[2])

    def test_guided_flow_gate_excludes_only_with_complete_negative_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            diagnostics = root / "diagnostics"
            diagnostics.mkdir()
            for task in ("cube", "pusht", "reacher", "tworoom"):
                (diagnostics / f"summary_{task}.json").write_text(
                    json.dumps(
                        {
                            "matched_guidance_comparison": {
                                "gf_minus_po_physical_improvement_mean": 0.0,
                                "bootstrap": {
                                    "gf_minus_po_physical_improvement": {
                                        "ci95": [-0.1, 0.1]
                                    }
                                },
                            }
                        }
                    ),
                    encoding="utf-8",
                )
            measured_timing = [
                {
                    "task_count": 4,
                    "guided_flow_faster_all_tasks": False,
                }
            ]
            gate = _guided_flow_selection_gate(root, measured_timing)
            self.assertEqual(gate["status"], "exclude_guided_flow_from_main_method")
            self.assertTrue(gate["exclude_guided_flow_from_main_method"])

            gate = _guided_flow_selection_gate(
                root,
                [{"task_count": 4, "guided_flow_faster_all_tasks": True}],
            )
            self.assertEqual(gate["status"], "retain_guided_flow_measured_efficiency_advantage")
            self.assertFalse(gate["exclude_guided_flow_from_main_method"])

            (diagnostics / "summary_cube.json").unlink()
            gate = _guided_flow_selection_gate(root, measured_timing)
            self.assertEqual(gate["status"], "guided_flow_evidence_incomplete")
            self.assertFalse(gate["exclude_guided_flow_from_main_method"])

    def test_candidate_pool_requires_complete_state_by_flow_grid(self):
        records = []
        for flow_steps in (1, 2):
            for state_id in ("a", "b"):
                for candidate_index in range(3):
                    records.append(
                        {
                            "state_id": state_id,
                            "flow_steps": flow_steps,
                            "candidate_index": candidate_index,
                            "predicted_cost": float(candidate_index),
                            "true_distance": float(3 - candidate_index),
                            "success": candidate_index == 2,
                            "action": [[float(candidate_index)]],
                        }
                    )
        result = candidate_pool_metrics(
            records,
            task="tworoom",
            flow_steps=(1, 2),
            candidates_per_flow_step=3,
            bootstrap_samples=8,
        )
        self.assertEqual(result["states"], 2)
        self.assertEqual(result["candidate_rows"], 12)
        self.assertEqual(set(result["by_flow_steps"]), {"1", "2"})
        self.assertEqual(result["by_flow_steps"]["1"]["candidate_rows"], 6)

    def test_guidance_metrics_marks_model_exploitation(self):
        result = guidance_effect_metrics(
            [
                {
                    "predicted_cost_before": 2,
                    "predicted_cost_after": 1,
                    "true_cost_before": 1,
                    "true_cost_after": 2,
                    "action_saturation_fraction": 0.25,
                }
            ]
        )
        self.assertEqual(result["model_exploitation_fraction"], 1.0)
        self.assertEqual(result["true_degradation_fraction"], 1.0)

    def test_guidance_metrics_accepts_persisted_paired_schema(self):
        result = guidance_effect_metrics(
            [
                {
                    "guided_predicted_cost_before": 2.0,
                    "guided_predicted_cost_after": 1.0,
                    "guided_true_cost_before": 1.0,
                    "guided_true_cost_after": 2.0,
                    "guided_action_rms_displacement": 0.25,
                }
            ]
        )
        self.assertEqual(result["model_exploitation_fraction"], 1.0)
        self.assertEqual(result["action_rms_displacement_mean"], 0.25)

    def test_matched_guidance_modes_pair_exact_configs_and_bootstrap_states(self):
        records = []

        def add(mode, state, flow, candidate, steps, eta, improvement, *, before=10.0):
            records.append(
                {
                    "guidance": mode,
                    "flow_steps": flow,
                    "candidate_index": candidate,
                    "guidance_inner_steps": steps,
                    "guidance_step_size": eta,
                    "max_rms_offset": 0.2,
                    "state_id": state,
                    "guided_true_cost_before": before,
                    "guided_true_cost_after": before - improvement,
                    "paired_outcome_valid": True,
                }
            )

        for mode, improvement in (("post_opt", 1.0), ("guided_flow", 2.0)):
            add(mode, "a", 2, 0, 1, 0.01, improvement)
        for mode, improvement in (("post_opt", 2.0), ("guided_flow", 3.0)):
            add(mode, "b", 2, 0, 1, 0.01, improvement)
        for mode, improvement in (("post_opt", 0.5), ("guided_flow", 2.5)):
            add(mode, "a", 5, 1, 5, 0.01, improvement)
        add("post_opt", "c", 2, 0, 1, 0.01, 1.0, before=10.0)
        add("guided_flow", "c", 2, 0, 1, 0.01, 2.0, before=11.0)
        add("post_opt", "a", 2, 0, 1, 0.02, 99.0)

        result = matched_guidance_mode_metrics(records, bootstrap_samples=64, seed=7)

        self.assertEqual(result["matched_configuration_count"], 2)
        self.assertEqual(result["matched_states"], 2)
        self.assertEqual(result["matched_state_configuration_pairs"], 3)
        self.assertEqual(result["mismatched_start_pairs"], 1)
        self.assertAlmostEqual(result["gf_minus_po_physical_improvement_mean"], 1.25)
        self.assertEqual(
            result["bootstrap"]["gf_minus_po_physical_improvement"]["cluster_count"],
            2,
        )
        config_results = result["matched_configurations"]
        self.assertAlmostEqual(
            config_results[0]["gf_minus_po_physical_improvement_mean"],
            1.0,
        )
        self.assertEqual(
            config_results[0]["bootstrap"]["gf_minus_po_physical_improvement"]["cluster_count"],
            2,
        )

    def test_matched_guidance_comparison_stratifies_physical_milestones(self):
        records = []

        for step, po_improvements, gf_improvements in (
            (5, (0.5, 0.75), (1.5, 1.75)),
            (10, (1.0, 1.0), (-1.0, -1.0)),
        ):
            for state_index, state in enumerate(("a", "b")):
                before = 10.0 + state_index
                for mode, improvement in (
                    ("post_opt", po_improvements[state_index]),
                    ("guided_flow", gf_improvements[state_index]),
                ):
                    records.append(
                        {
                            "guidance": mode,
                            "flow_steps": 2,
                            "candidate_index": 0,
                            "guidance_inner_steps": 1,
                            "guidance_step_size": 0.01,
                            "max_rms_offset": 0.2,
                            "state_id": state,
                            "physical_comparison_step": step,
                            "guided_true_cost_before": before,
                            "guided_true_cost_after": before - improvement,
                            "paired_outcome_valid": True,
                        }
                    )

        result = matched_guidance_mode_metrics_by_milestone(
            records,
            milestone_steps=(5, 10),
            bootstrap_samples=64,
            seed=7,
        )

        self.assertEqual(set(result), {"5", "10"})
        self.assertAlmostEqual(
            result["5"]["gf_minus_po_physical_improvement_mean"],
            1.0,
        )
        self.assertAlmostEqual(
            result["10"]["gf_minus_po_physical_improvement_mean"],
            -2.0,
        )
        self.assertEqual(result["5"]["matched_states"], 2)
        self.assertEqual(result["10"]["matched_states"], 2)

    def test_matched_guidance_timing_requires_identical_config_and_clean_four_task_windows(self):
        tasks = ("cube", "pusht", "reacher", "tworoom")

        def candidate(mode, latency, *, step_size=0.01, eligible=True):
            return {
                "candidate_id": mode,
                "fine_timing_eligible": eligible,
                "config": {
                    "family": "p0_post_opt" if mode == "post_opt" else "p0_guided_flow",
                    "mode": "P0",
                    "flow_steps": 2,
                    "candidate_count": 1,
                    "po_iterations": 5,
                    "guidance": mode,
                    "guidance_step_size": step_size,
                    "max_rms_offset": 0.2,
                    "guidance_last_steps": 2 if mode == "guided_flow" else None,
                },
                "timing_by_task": {
                    task: {"batch1_p50_seconds": latency}
                    for task in tasks
                },
            }

        comparisons = _matched_guidance_timing_comparisons(
            [candidate("post_opt", 1.0), candidate("guided_flow", 0.9)]
        )
        self.assertEqual(len(comparisons), 1)
        self.assertTrue(comparisons[0]["guided_flow_faster_all_tasks"])
        self.assertEqual(comparisons[0]["guided_flow_faster_task_count"], 4)
        self.assertAlmostEqual(
            comparisons[0]["guided_flow_minus_post_opt_mean_batch1_p50_seconds"],
            -0.1,
        )
        self.assertEqual(
            _matched_guidance_timing_comparisons(
                [candidate("post_opt", 1.0), candidate("guided_flow", 0.9, step_size=0.02)]
            ),
            [],
        )
        self.assertEqual(
            _matched_guidance_timing_comparisons(
                [candidate("post_opt", 1.0, eligible=False), candidate("guided_flow", 0.9)]
            ),
            [],
        )

    def test_guidance_metrics_excludes_short_unpaired_branches(self):
        valid = {
            "state_id": "a",
            "guided_predicted_cost_before": 2.0,
            "guided_predicted_cost_after": 1.0,
            "guided_true_cost_before": 2.0,
            "guided_true_cost_after": 1.0,
            "random_true_cost_before": 2.0,
            "random_true_cost_after": 1.5,
            "guided_action_rms_displacement": 0.2,
            "random_action_rms_displacement": 0.2,
            "paired_outcome_valid": True,
        }
        invalid = {**valid, "state_id": "b", "paired_outcome_valid": False}
        result = paired_guidance_metrics([valid, invalid], bootstrap_samples=8)
        self.assertEqual(result["records"], 1)
        self.assertEqual(result["input_records"], 2)
        self.assertEqual(result["excluded_noncomparable_records"], 1)

    def test_same_rms_random_actions_match_guidance_displacement(self):
        baseline = np.zeros((3, 5, 4), dtype=np.float32)
        guided = np.stack(
            [
                np.full((5, 4), 0.0, dtype=np.float32),
                np.full((5, 4), 0.1, dtype=np.float32),
                np.arange(20, dtype=np.float32).reshape(5, 4) / 100,
            ]
        )
        random_actions, random_rms = _same_rms_random_actions(
            baseline, guided, seed=2026
        )
        guided_rms = np.sqrt(np.mean(np.square(guided - baseline), axis=(1, 2)))
        np.testing.assert_allclose(random_rms, guided_rms, atol=1e-7, rtol=0)
        again, _ = _same_rms_random_actions(baseline, guided, seed=2026)
        np.testing.assert_array_equal(random_actions, again)

    def test_guidance_outcome_uses_latest_shared_real_milestone(self):
        def branch(distance, latent):
            return {
                "milestones": {
                    "25": {
                        "raw_env_step": 20,
                        "current": [distance, 0.0],
                        "goal": [0.0, 0.0],
                        "future_latent_cost": latent,
                    },
                    "20": {
                        "raw_env_step": 20,
                        "current": [distance, 0.0],
                        "goal": [0.0, 0.0],
                        "future_latent_cost": latent,
                    },
                }
            }

        result = _common_guidance_outcome(
            "cube", branch(0.04, 1.0), branch(0.02, 0.5), branch(0.03, 0.7)
        )
        self.assertEqual(result["step"], 20)
        self.assertAlmostEqual(result["physical_costs"][0], 1.0)
        self.assertEqual(result["latent_costs"], [1.0, 0.5, 0.7])

    def test_fixed_pool_replay_accepts_small_stored_action_rounding_drift(self):
        baseline = np.asarray([[[0.1, -0.3]]], dtype=np.float32)
        replayed = baseline.copy()
        replayed[0, 0, 0] += np.float32(9.8348e-6)
        self.assertTrue(
            phase15_diagnostics._fixed_pool_actions_match_replay(replayed, baseline)
        )

        replayed[0, 0, 0] += np.float32(5e-5)
        self.assertFalse(
            phase15_diagnostics._fixed_pool_actions_match_replay(replayed, baseline)
        )

    def test_guidance_branch_outcomes_attach_trace_fields_before_comparison(self):
        baseline = {
            "task": "cube",
            "state_id": "episode=1;start=2;row=3",
            "slot": 0,
            "episode_id": 1,
            "start_step": 2,
            "row_index": 3,
            "flow_steps": 2,
            "candidate_index": 0,
            "predicted_cost": 1.0,
            "action": [[0.0]],
            "success": False,
            "true_distance": 0.04,
            "valid_length": 25,
            "milestones": {
                "25": {
                    "raw_env_step": 25,
                    "current": [0.04, 0.0],
                    "goal": [0.0, 0.0],
                    "future_latent_cost": 1.0,
                }
            },
        }

        def raw_episode(distance, latent_cost, success):
            return {
                "slot": 0,
                "success": success,
                "steps": [
                    {
                        "raw_env_step": 25,
                        "current": [distance, 0.0],
                        "goal": [0.0, 0.0],
                        "distance": distance,
                        "termination_reason": "success" if success else "budget",
                    }
                ],
                "future_latents": {"25": [0.1, 0.2]},
                "future_latent_costs": {"25": latent_cost},
            }

        actions = np.full((1, 5, 1), 0.25, dtype=np.float32)
        guided = phase15_diagnostics._attach_guidance_branch_outcomes(
            [baseline], [raw_episode(0.02, 0.5, True)], actions, task="cube"
        )
        random = phase15_diagnostics._attach_guidance_branch_outcomes(
            [baseline], [raw_episode(0.06, 1.5, False)], -actions, task="cube"
        )

        self.assertEqual(guided[0]["valid_length"], 1)
        self.assertEqual(guided[0]["action"], actions[0].tolist())
        self.assertNotIn("predicted_cost", guided[0])
        paired = _common_guidance_outcome("cube", baseline, guided[0], random[0])
        self.assertEqual(paired["step"], 25)
        self.assertEqual(paired["latent_costs"], [1.0, 0.5, 1.5])

    def test_candidate_branch_resume_checks_identity_and_action(self):
        expected = {
            "task": "cube",
            "state_id": "episode=1;start=2;row=3",
            "slot": 0,
            "episode_id": 1,
            "start_step": 2,
            "row_index": 3,
            "flow_steps": 2,
            "candidate_index": 0,
            "predicted_cost": 0.5,
            "action": [[0.1, -0.2]],
        }
        actual = {**expected, "outcome_status": "completed", "true_distance": 1.0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "branch.jsonl"
            path.write_text(json.dumps(actual) + "\n", encoding="utf-8")
            self.assertTrue(_branch_file_matches(path, [expected]))
            self.assertFalse(
                _branch_file_matches(path, [{**expected, "action": [[0.2, -0.2]]}])
            )

    def test_paired_guidance_compares_same_displacement_random_control(self):
        result = paired_guidance_metrics(
            [
                {
                    "state_id": "a",
                    "guided_predicted_cost_before": 2.0,
                    "guided_predicted_cost_after": 1.0,
                    "guided_true_cost_before": 2.0,
                    "guided_true_cost_after": 1.0,
                    "random_true_cost_before": 2.0,
                    "random_true_cost_after": 1.5,
                    "guided_action_rms_displacement": 0.2,
                    "random_action_rms_displacement": 0.2,
                }
            ],
            bootstrap_samples=8,
        )
        self.assertAlmostEqual(result["paired_advantage_mean"], 0.5)
        self.assertEqual(result["guided_true_improvement_fraction"], 1.0)

    def test_guidance_summary_gives_each_state_equal_weight(self):
        def record(state, before, after):
            return {
                "state_id": state,
                "guided_predicted_cost_before": 2.0,
                "guided_predicted_cost_after": 1.0,
                "guided_true_cost_before": before,
                "guided_true_cost_after": after,
                "random_true_cost_before": before,
                "random_true_cost_after": before,
                "guided_action_rms_displacement": 0.2,
                "random_action_rms_displacement": 0.2,
            }

        result = paired_guidance_metrics(
            [record("a", 2.0, 1.0), record("a", 2.0, 1.0), record("b", 2.0, 2.0)],
            bootstrap_samples=8,
        )
        self.assertEqual(result["records"], 3)
        self.assertEqual(result["states"], 2)
        self.assertAlmostEqual(result["guided_true_improvement_mean"], 0.5)

    def test_bootstrap_samples_states(self):
        rows = [{"state_id": index, "value": float(index)} for index in range(4)]
        result = cluster_bootstrap(rows, "value", samples=32, seed=3)
        self.assertEqual(result["cluster_count"], 4)
        self.assertEqual(result["samples"], 32)

    def test_cluster_bootstrap_averages_rows_within_each_state_first(self):
        rows = [
            {"state_id": "a", "value": 1.0},
            {"state_id": "a", "value": 1.0},
            {"state_id": "a", "value": 1.0},
            {"state_id": "b", "value": 5.0},
        ]
        result = cluster_bootstrap(rows, "value", samples=32, seed=3)
        self.assertEqual(result["estimate"], 3.0)

    def test_probe_split_is_trajectory_disjoint(self):
        split = make_probe_split(range(10), eval_trajectory_ids=(99,), seed=2026)
        self.assertTrue(set(split["train"]).isdisjoint(split["validation"]))
        self.assertTrue(set(split["train"]).isdisjoint(split["excluded_eval"]))
        self.assertTrue(set(split["validation"]).isdisjoint(split["excluded_eval"]))

    def test_synchronous_timing_runs_requested_calls(self):
        calls = []
        result = synchronous_timing(lambda: calls.append(1), warmup=2, runs=3)
        self.assertEqual(len(calls), 5)
        self.assertEqual(result["runs"], 3)
        self.assertEqual(len(result["samples_seconds"]), 3)

    def test_timing_windows_keep_warmup_and_measurement_counts(self):
        result = summarize_timing_samples(range(20), warmup=5, runs=10)
        self.assertEqual(result["warmup"], 5)
        self.assertEqual(result["runs"], 10)
        self.assertEqual(result["samples_seconds"], list(range(5, 15)))
        self.assertIsNone(summarize_timing_samples(range(10), warmup=5, runs=10))

    def test_timing_tap_captures_first_real_input_then_stops(self):
        class Policy:
            def __init__(self):
                self.calls = []

            def set_env(self, env):
                self.env = env

            def get_action(self, info):
                self.calls.append(info)
                return "action"

        policy = Policy()
        captured = []
        tap = _FirstActionTimingTap(
            policy,
            lambda underlying, info: captured.append((underlying, info)) or {"runs": 2},
        )
        info = {"pixels": np.zeros((1, 2, 2, 3))}
        with self.assertRaises(_TimingCaptureComplete) as raised:
            tap.get_action(info)
        self.assertEqual(raised.exception.result, {"runs": 2})
        self.assertEqual(captured, [(policy, info)])
        self.assertEqual(policy.calls, [])
        self.assertEqual(tap.get_action(info), "action")
        self.assertEqual(policy.calls, [info])

    def test_scan_slots_bound_concurrent_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            with phase15_scan_slot(directory, max_slots=1) as slot:
                self.assertEqual(slot, 0)
                with self.assertRaises(RuntimeError):
                    with phase15_scan_slot(directory, max_slots=1):
                        pass
            self.assertFalse(
                (Path(directory) / "locks" / "scan_slots" / "slot_0.lock").exists()
            )

    def test_condition_lock_reclaims_dead_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "condition" / "result.json"
            lock = result.with_name(".condition.lock")
            lock.parent.mkdir(parents=True)
            lock.write_text("pid=999999999\n", encoding="ascii")
            with condition_lock(result):
                self.assertEqual(lock.read_text(encoding="ascii"), f"pid={os.getpid()}\n")
            self.assertFalse(lock.exists())

    def test_adaptive_stability_picks_two_per_category(self):
        rows = []
        categories = [("P3", "none"), ("P0", "post_opt"), ("P0", "guided_flow"), ("P3", "post_opt"), ("P1", "none")]
        all_specs = primary_condition_specs()
        for task in ("cube", "pusht", "reacher", "tworoom"):
            for index, (mode, guidance) in enumerate(categories):
                spec = next(
                    item for item in all_specs
                    if item["task"] == task and item["mode"] == mode and item["guidance"] == guidance
                )
                rows.append({
                    "task": task,
                    "status": "completed",
                    "spec": spec,
                    "condition_id": f"{task}-{index}",
                    "success_rate": 1.0 - index * 0.1,
                    "forward_count": index,
                })
        result = adaptive_stability_specs(rows)
        self.assertEqual(len(result), 40)
        self.assertEqual({item["evaluation_seed"] for item in result}, {43, 44})


class Round5Phase15ReportTests(unittest.TestCase):
    def test_report_renders_paired_control_effects_and_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "outputs" / "round5" / "phase1_5"
            control_root = output_root / "diagnostics" / "candidate_pool" / "pusht"
            control_root.mkdir(parents=True)

            def effect(estimate, ci95):
                return {"estimate": estimate, "ci95": ci95, "state_count": 50}

            payload = {"protocol": {"states": 50}}
            for name, success_effect, pair_count, state_count in (
                ("rms_normalized_gaussian", 0.125, 300, 48),
                ("physical_zero", -0.1, 142, 45),
                ("normalized_zero", 0.0, 150, 50),
            ):
                payload[name] = {
                    "paired_effect": {
                        "success_rate_delta": effect(
                            success_effect,
                            [success_effect - 0.1, success_effect + 0.1],
                        ),
                    },
                    "common_milestone_distance": {
                        "paired_effect": effect(0.75, [0.25, 1.25]),
                        "paired_action_state_count": pair_count,
                        "total_action_state_count": 3200 if name == "rms_normalized_gaussian" else 150,
                        "state_count_with_pairs": state_count,
                    },
                }
            (control_root / "control_paired_effects.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            cube_root = output_root / "diagnostics" / "candidate_pool" / "cube"
            cube_root.mkdir(parents=True)
            (cube_root / "control_compatibility_audit.json").write_text(
                json.dumps({
                    "source": {"actions": 240, "rows": 12000, "sha256": "source-sha"},
                    "result": {"actions": 222, "rows": 11100, "sha256": "current-sha"},
                    "compatibility_review": {
                        "action_comparison": "all retained actions matched exactly",
                    },
                    "preserved_outcomes": {"outcomes_reexecuted": False},
                    "per_action_files": {"preserved_legacy_file_count": 240},
                }),
                encoding="utf-8",
            )

            report = _render_report(
                config_path=Path("config/round5/phase1_5.json"),
                output_root=output_root,
                rows=[],
                index_payload={"counts": {"completed": 0}, "items": {}},
            )

        self.assertIn("### 同状态配对控制效应", report)
        self.assertIn(
            "| PushT | normalized Gaussian | +0.125 [+0.025, +0.225] | "
            "+0.750 [+0.250, +1.250] | "
            "50/50 success states; 300/3,200 pairs; 48/50 distance states |",
            report,
        )
        self.assertIn("| PushT | physical zero | -0.100", report)
        self.assertIn("| PushT | normalized zero | +0.000", report)
        self.assertIn(
            "Cube 控制池由 240 个源动作、12,000 条记录迁移为 222 个动作、11,100 条记录",
            report,
        )
        self.assertIn("是否重跑环境结果：否；保留旧版逐动作文件 240 份。", report)
        self.assertIn("源聚合 `source-sha`；当前聚合 `current-sha`", report)

    def test_probe_table_reads_nested_validation_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "outputs" / "round5" / "phase1_5"
            probe_root = output_root / "diagnostics" / "probe"
            probe_root.mkdir(parents=True)
            (probe_root / "summary.json").write_text(
                json.dumps({
                    "results": [
                        {
                            "task": "reacher",
                            "status": "ok",
                            "ridge": {
                                "validation_metrics": {"count": 123, "mae": 0.25},
                            },
                            "random_encoder_ridge": {
                                "validation_metrics": {"count": 123, "mae": 0.75},
                            },
                            "mlp": [
                                {"mae": 0.2},
                                {"mae": 0.3},
                                {"mae": 0.4},
                            ],
                        },
                        {
                            "task": "cube",
                            "status": "ok",
                            "ridge": {"validation_count": 40, "mae": 0.6},
                            "random_encoder_ridge": {"mae": 0.8},
                        },
                        {
                            "task": "pusht",
                            "status": "ok",
                            "ridge": {
                                "validation_metrics": {"count": None, "mae": None},
                                "validation_count": 77,
                                "mae": 0.125,
                            },
                            "random_encoder_ridge": {
                                "validation_metrics": {"count": None, "mae": None},
                                "mae": 0.875,
                            },
                        },
                    ],
                }),
                encoding="utf-8",
            )

            report = _render_report(
                config_path=Path("config/round5/phase1_5.json"),
                output_root=output_root,
                rows=[],
                index_payload={"counts": {"completed": 0}, "items": {}},
            )

        self.assertIn(
            "| reacher | 123 | 0.2500 | 0.7500 | 0.3000 | — | — |",
            report,
        )
        self.assertIn("| cube | 40 | 0.6000 | 0.8000 | — | — | — |", report)
        self.assertIn("| pusht | 77 | 0.1250 | 0.8750 | — | — | — |", report)

    def test_report_includes_coarse_frontier_and_interference_sensitivity(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "outputs" / "round5" / "phase1_5"
            analysis_root = output_root / "analysis"
            analysis_root.mkdir(parents=True)
            (analysis_root / "fine_timing_selection.json").write_text(
                json.dumps(
                    {
                        "fixed_anchor_count": 1,
                        "adaptive_anchor_count": 2,
                        "interference_suspect_frontier_condition_count": 1,
                        "coarse_frontier_sensitivity": {
                            "interference_status_counts": {
                                "batch1": {
                                    "interference_flagged": 1,
                                    "no_external_compute_observed_at_boundaries": 3,
                                },
                                "batch50": {
                                    "interference_flagged": 2,
                                    "no_external_compute_observed_at_boundaries": 2,
                                },
                            }
                        },
                        "selected_conditions": [
                            {
                                "task": "cube",
                                "frontier_membership": ["observed", "batch1_boundary_unflagged"],
                                "isolated_retest_recommended": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = _render_report(
                config_path=Path("config/round5/phase1_5.json"),
                output_root=output_root,
                rows=[],
                index_payload={"counts": {"completed": 0}, "items": {}},
            )

        self.assertIn("粗测敏感性按 seed42 主网格逐任务比较", report)
        self.assertIn("| batch1 | interference_flagged | 1 |", report)
        self.assertIn("| cube | 1 | 1 | 1 | 1 |", report)
        self.assertIn("1 个固定锚点和 2 个自适应锚点", report)


if __name__ == "__main__":
    unittest.main()
