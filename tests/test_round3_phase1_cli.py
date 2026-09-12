import json
import tempfile
import unittest
from pathlib import Path

from scripts.round3_phase1 import build_parser
from source.common.round3_phase1 import build_artifact_registry, run_synthetic_goal_refresh_checks
from source.common.round3_sampling import build_revised_cohorts as direct_build_revised_cohorts


class Round3CliTests(unittest.TestCase):
    def test_parser_exposes_audit_registry_cohort_and_evaluate(self):
        parser = build_parser()
        cases = {
            "audit": ["audit"],
            "registry": ["registry"],
            "cohort": ["cohort", "cube"],
            "legacy-cohort": ["legacy-cohort", "cube"],
            "analyze": ["analyze"],
            "evaluate": ["evaluate", "cube", "e0_lewm", "stage_b", "--cohort", "c", "--checkpoint", "p"],
        }
        for arguments in cases.values():
            self.assertIsNotNone(parser.parse_args(arguments))

        parsed = parser.parse_args(
            [
                "evaluate",
                "cube",
                "leflow",
                "stage_b",
                "--cohort",
                "c",
                "--checkpoint",
                "p",
            ]
        )
        self.assertEqual(parsed.method, "leflow")

    def test_synthetic_refresh_is_deterministic_and_complete(self):
        first = run_synthetic_goal_refresh_checks("reacher", count=10, seed=42)
        second = run_synthetic_goal_refresh_checks("reacher", count=10, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first["checks_run"], 10)
        self.assertEqual(first["status"], "accepted")

    def test_registry_marks_missing_entry_without_substitution(self):
        self.assertTrue(callable(direct_build_revised_cohorts))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "spec.json"
            spec.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "task": "tworoom",
                                "method": "e3_fast",
                                "stages": ["stage_b"],
                                "checkpoint": None,
                                "train_config": None,
                                "historical_result": None,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            registry = build_artifact_registry(spec, repo_root=root)
            entry = registry["entries"][0]
            self.assertEqual(entry["status"], "missing")
            self.assertFalse(entry["participates"])
            self.assertIsNone(entry["checkpoint"])


if __name__ == "__main__":
    unittest.main()
