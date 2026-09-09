import json
import runpy
import tempfile
import unittest
from pathlib import Path


class Stage1RemainingTests(unittest.TestCase):
    @staticmethod
    def _episodes(success=True):
        return [
            {
                "slot": 0,
                "episode_id": 9,
                "dataset_episode": 9,
                "start_step": 1,
                "row_index": 2,
                "goal_row_index": 7,
                "goal_step": 26,
                "success": success,
                "initial_success": False,
                "start_distance": 1.0,
                "terminal_distance": 0.0 if success else 0.03,
                "first_success_step": 1 if success else None,
                "hold_success": False,
                "rollout_failed": False,
                "missing_field_count": 0,
                "invalid_action_count": 0,
                "early_terminated": False,
                "steps": [
                    {
                        "raw_env_step": 1,
                        "current": [0.0, 0.0, 0.0],
                        "goal": [0.0, 0.0, 0.0] if success else [0.03, 0.0, 0.0],
                        "action_legal": True,
                        "neutral_action": True,
                        "terminated": False,
                        "truncated": False,
                        "termination_reason": "running",
                    }
                ],
            }
        ]

    @staticmethod
    def _manifest(variant):
        from source.common.round3_phase1 import CohortEntry, CohortManifest

        entry = CohortEntry(
            row_index=2,
            episode_id=9,
            start_step=1,
            goal_row_index=7,
            goal_step=26,
            start_distance=1.0,
            initially_successful=False,
        )
        return CohortManifest(
            task="cube",
            cohort_id=f"cube_dev_{variant}",
            cohort_kind="dev",
            protocol_variant=variant,
            seed=42,
            goal_offset_steps=25,
            entries=(entry,),
            episode_split={"dev": (9,), "final": (), "online": ()},
        )

    @classmethod
    def _write_source_variant(cls, output, variant, *, success=True):
        from source.common.round3_phase1 import (
            canonical_result_path,
            canonical_trace_path,
            enrich_result_payload,
            write_round3_result,
        )

        manifest = cls._manifest(variant)
        cohort_root = output / "cohorts" / "cube"
        manifest_path = cohort_root / f"dev_{variant}.json"
        manifest.save(manifest_path)
        result_path = canonical_result_path(
            output,
            task="cube",
            method="e0_lewm",
            protocol_variant=variant,
            stage="stage_b",
            cohort_kind="dev",
            cohort_sha256=manifest.computed_sha256,
        )
        trace_path = canonical_trace_path(
            output,
            task="cube",
            method="e0_lewm",
            protocol_variant=variant,
            stage="stage_b",
            cohort_kind="dev",
            cohort_sha256=manifest.computed_sha256,
        )
        episodes = cls._episodes(success=success)
        payload = enrich_result_payload(
            {"status": "ok", "task": "cube"},
            manifest=manifest,
            protocol_variant=variant,
            trace_records=episodes,
        )
        write_round3_result(
            payload,
            result_path.parent,
            trace_records=episodes,
            trace_output_dir=trace_path.parent,
        )
        return manifest, result_path, trace_path

    @staticmethod
    def _run_cli(script_name, output):
        root = Path(__file__).resolve().parents[1]
        namespace = runpy.run_path(
            str(root / "scripts" / script_name),
            run_name=f"stage1_remaining_{Path(script_name).stem}",
        )
        namespace["main"](
            [
                "cube",
                "e0_lewm",
                "stage_b",
                "--output",
                str(output),
            ]
        )

    @staticmethod
    def _artifact_bytes(result_path, trace_path):
        return {
            result_path: result_path.read_bytes(),
            result_path.parent / "metrics.txt": (
                result_path.parent / "metrics.txt"
            ).read_bytes(),
            trace_path: trace_path.read_bytes(),
        }

    def test_round3_derive_only_derives_tolerance_from_legacy(self):
        from source.common.round3_phase1 import (
            canonical_result_path,
            canonical_trace_path,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "phase1"
            legacy = self._write_source_variant(output, "legacy", success=False)
            self._write_source_variant(output, "sampling_revised", success=True)
            round3 = self._write_source_variant(output, "round3_revised", success=True)
            frozen = {
                **self._artifact_bytes(legacy[1], legacy[2]),
                **self._artifact_bytes(round3[1], round3[2]),
            }
            before_result_files = {
                path for path in (output / "results").rglob("*") if path.is_file()
            }
            before_trace_files = {
                path for path in (output / "traces").rglob("*") if path.is_file()
            }

            tolerance_manifest = self._manifest("tolerance_revised")
            tolerance_manifest.save(
                output / "cohorts" / "cube" / "dev_tolerance_revised.json"
            )
            tolerance_result = canonical_result_path(
                output,
                task="cube",
                method="e0_lewm",
                protocol_variant="tolerance_revised",
                stage="stage_b",
                cohort_kind="dev",
                cohort_sha256=tolerance_manifest.computed_sha256,
            )
            tolerance_trace = canonical_trace_path(
                output,
                task="cube",
                method="e0_lewm",
                protocol_variant="tolerance_revised",
                stage="stage_b",
                cohort_kind="dev",
                cohort_sha256=tolerance_manifest.computed_sha256,
            )

            self._run_cli("round3_derive.py", output)

            self.assertTrue(tolerance_result.is_file())
            self.assertTrue(tolerance_trace.is_file())
            self.assertEqual(
                {
                    path
                    for path in (output / "results").rglob("*")
                    if path.is_file()
                }
                - before_result_files,
                {tolerance_result, tolerance_result.parent / "metrics.txt"},
            )
            self.assertEqual(
                {
                    path
                    for path in (output / "traces").rglob("*")
                    if path.is_file()
                }
                - before_trace_files,
                {tolerance_trace},
            )
            self.assertEqual(
                json.loads(tolerance_result.read_text(encoding="utf-8"))["derived_from"][
                    "mapping"
                ],
                {
                    "source_protocol_variant": "legacy",
                    "target_protocol_variant": "tolerance_revised",
                },
            )
            self.assertEqual(
                frozen,
                {
                    **self._artifact_bytes(legacy[1], legacy[2]),
                    **self._artifact_bytes(round3[1], round3[2]),
                },
            )

    def test_round3_derive_all_creates_sampling_and_tolerance_without_rewriting_round3(self):
        from source.common.round3_phase1 import (
            canonical_result_path,
            canonical_trace_path,
            load_result,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "phase1"
            legacy = self._write_source_variant(output, "legacy", success=False)
            round3 = self._write_source_variant(output, "round3_revised", success=True)
            sampling_manifest = self._manifest("sampling_revised")
            sampling_manifest.save(
                output / "cohorts" / "cube" / "dev_sampling_revised.json"
            )
            tolerance_manifest = self._manifest("tolerance_revised")
            tolerance_manifest.save(
                output / "cohorts" / "cube" / "dev_tolerance_revised.json"
            )
            frozen = {
                **self._artifact_bytes(legacy[1], legacy[2]),
                **self._artifact_bytes(round3[1], round3[2]),
            }
            before_result_files = {
                path for path in (output / "results").rglob("*") if path.is_file()
            }
            before_trace_files = {
                path for path in (output / "traces").rglob("*") if path.is_file()
            }

            self._run_cli("round3_derive_all.py", output)

            sampling_result = canonical_result_path(
                output,
                task="cube",
                method="e0_lewm",
                protocol_variant="sampling_revised",
                stage="stage_b",
                cohort_kind="dev",
                cohort_sha256=sampling_manifest.computed_sha256,
            )
            sampling_trace = canonical_trace_path(
                output,
                task="cube",
                method="e0_lewm",
                protocol_variant="sampling_revised",
                stage="stage_b",
                cohort_kind="dev",
                cohort_sha256=sampling_manifest.computed_sha256,
            )
            tolerance_result = canonical_result_path(
                output,
                task="cube",
                method="e0_lewm",
                protocol_variant="tolerance_revised",
                stage="stage_b",
                cohort_kind="dev",
                cohort_sha256=tolerance_manifest.computed_sha256,
            )
            tolerance_trace = canonical_trace_path(
                output,
                task="cube",
                method="e0_lewm",
                protocol_variant="tolerance_revised",
                stage="stage_b",
                cohort_kind="dev",
                cohort_sha256=tolerance_manifest.computed_sha256,
            )
            self.assertTrue(sampling_result.is_file())
            self.assertTrue(sampling_trace.is_file())
            self.assertTrue(tolerance_result.is_file())
            self.assertTrue(tolerance_trace.is_file())
            self.assertEqual(
                {
                    path
                    for path in (output / "results").rglob("*")
                    if path.is_file()
                }
                - before_result_files,
                {
                    sampling_result,
                    sampling_result.parent / "metrics.txt",
                    tolerance_result,
                    tolerance_result.parent / "metrics.txt",
                },
            )
            self.assertEqual(
                {
                    path
                    for path in (output / "traces").rglob("*")
                    if path.is_file()
                }
                - before_trace_files,
                {sampling_trace, tolerance_trace},
            )
            self.assertEqual(
                load_result(sampling_result)["derived_from"]["mapping"],
                {
                    "source_protocol_variant": "round3_revised",
                    "target_protocol_variant": "sampling_revised",
                },
            )
            self.assertEqual(
                load_result(tolerance_result)["derived_from"]["mapping"],
                {
                    "source_protocol_variant": "legacy",
                    "target_protocol_variant": "tolerance_revised",
                },
            )
            self.assertEqual(
                frozen,
                {
                    **self._artifact_bytes(legacy[1], legacy[2]),
                    **self._artifact_bytes(round3[1], round3[2]),
                },
            )

    def test_dev_and_final_artifacts_are_canonically_isolated(self):
        from source.common.round3_phase1 import canonical_result_path

        with tempfile.TemporaryDirectory() as tmp:
            common = dict(
                root=Path(tmp) / "results",
                task="cube",
                method="random",
                protocol_variant="round3_revised",
                stage="stage_b",
                cohort_sha256="a" * 64,
            )
            dev = canonical_result_path(cohort_kind="dev", **common)
            final = canonical_result_path(cohort_kind="final", **common)
            self.assertNotEqual(dev, final)
            self.assertEqual(dev.parent.name, "a" * 64)
            self.assertEqual(dev.parent.parent.name, "dev")
            self.assertEqual(final.parent.name, "a" * 64)
            self.assertEqual(final.parent.parent.name, "final")

    def test_preflight_rejects_overwrite_without_partial_publish(self):
        from source.common.round3_derivation import preflight_derived_publish

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            target = root / "target"
            source.write_text("source", encoding="utf-8")
            target.mkdir()
            existing = {
                target / "result.json": b"existing-result",
                target / "metrics.txt": b"existing-metrics",
                target / "episodes.jsonl": b"existing-trace",
            }
            for path, data in existing.items():
                path.write_bytes(data)
            with self.assertRaises(FileExistsError):
                preflight_derived_publish(
                    source_result_path=source,
                    target_result_dir=target,
                    target_trace_dir=target,
                )
            self.assertEqual(
                {path: path.read_bytes() for path in existing}, existing
            )

    def test_bad_trace_hash_and_episode_order_are_rejected(self):
        from source.common.round3_derivation import validate_derivation_source

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "episodes.jsonl"
            episodes = [
                {
                    "slot": 0,
                    "episode_id": "ep-0",
                    "dataset_episode": 0,
                    "start_step": 0,
                    "row_index": 0,
                    "goal_row_index": 0,
                    "goal_step": 0,
                    "steps": [
                        {
                            "raw_env_step": 0,
                            "terminated": False,
                            "truncated": False,
                            "termination_reason": "running",
                        }
                    ],
                },
                {
                    "slot": 1,
                    "episode_id": "ep-1",
                    "dataset_episode": 1,
                    "start_step": 0,
                    "row_index": 1,
                    "goal_row_index": 1,
                    "goal_step": 0,
                    "steps": [
                        {
                            "raw_env_step": 0,
                            "terminated": False,
                            "truncated": False,
                            "termination_reason": "running",
                        }
                    ],
                },
            ]
            trace.write_text(
                "".join(json.dumps(ep, sort_keys=True) + "\n" for ep in episodes),
                encoding="utf-8",
            )
            manifest = {
                "task": "cube",
                "protocol_variant": "legacy",
                "cohort_kind": "dev",
                "entries": [
                    {key: episode[key] for key in (
                        "slot",
                        "episode_id",
                        "dataset_episode",
                        "start_step",
                        "row_index",
                        "goal_row_index",
                        "goal_step",
                    )}
                    for episode in episodes
                ],
            }
            payload = {
                "task": "cube",
                "protocol_variant": "legacy",
                "episodes": episodes,
                "trace_content_sha256": "0" * 64,
            }
            with self.assertRaises(ValueError):
                validate_derivation_source(
                    payload,
                    manifest=manifest,
                    trace_path=trace,
                    task="cube",
                    source_result_path=root / "source.json",
                )

            payload["trace_content_sha256"] = None
            payload["episodes"] = list(reversed(episodes))
            with self.assertRaises(ValueError):
                validate_derivation_source(
                    payload,
                    manifest=manifest,
                    trace_path=trace,
                    task="cube",
                    source_result_path=root / "source.json",
                )

    def test_missing_weight_is_not_not_run(self):
        from source.common.round3_phase1 import classify_result_status

        missing_weight = classify_result_status(
            registry_available=False,
            result_exists=False,
            result_valid=None,
        )
        not_run = classify_result_status(
            registry_available=True,
            result_exists=False,
            result_valid=None,
        )
        self.assertEqual(missing_weight["status"], "missing_weight")
        self.assertEqual(missing_weight["decision"], "unavailable")
        self.assertEqual(not_run["status"], "not_run")
        self.assertNotEqual(missing_weight["status"], not_run["status"])


if __name__ == "__main__":
    unittest.main()
