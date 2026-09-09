import json
import tempfile
import unittest
from pathlib import Path


class Stage1RemainingTests(unittest.TestCase):
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
