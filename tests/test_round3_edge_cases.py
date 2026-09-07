import tempfile
import unittest
from pathlib import Path

import numpy as np

from source.common.round3_phase1 import (
    CohortEntry,
    CohortManifest,
    Round3TraceCollector,
    protocol_sensitivity,
    write_phase1_matrix,
)


def manifest(task="reacher"):
    return CohortManifest(
        task=task,
        cohort_id="edge_dev",
        cohort_kind="dev",
        protocol_variant="round3_revised",
        seed=42,
        goal_offset_steps=25,
        entries=(
            CohortEntry(
                row_index=0,
                episode_id=1,
                start_step=0,
                goal_row_index=25,
                goal_step=25,
                start_distance=1.0,
                initially_successful=False,
            ),
        ),
        episode_split={"dev": (1,)},
    )


class Round3EdgeCaseTests(unittest.TestCase):
    def test_protocol_sensitivity_does_not_pair_different_cohorts(self):
        legacy = [{"episode_id": 1, "start_step": 0, "success": False}]
        revised = [{"episode_id": 2, "start_step": 0, "success": True}]
        result = protocol_sensitivity(legacy, revised, legacy, revised)
        comparison = result["legacy_to_sampling_revised"]
        self.assertEqual(comparison["comparison_type"], "unpaired_descriptive")
        self.assertEqual(comparison["delta_pp"], 100.0)
        self.assertIsNone(comparison["paired"])

    def test_nan_environment_success_is_not_counted_as_true(self):
        collector = Round3TraceCollector("reacher", manifest())
        collector.record_step(
            np.zeros((1, 2)),
            {
                "qpos": np.zeros((1, 1, 2)),
                "goal_qpos": np.zeros((1, 1, 2)),
                "success": np.array([[np.nan]]),
                "terminated": np.array([[False]]),
                "truncated": np.array([[False]]),
            },
            raw_env_step=1,
        )
        record = collector.finalize([True], eval_budget=50)[0]
        self.assertIsNone(record["steps"][0]["env_success"])
        self.assertTrue(record["success"])

    def test_matrix_includes_missing_fixed_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = write_phase1_matrix(Path(directory) / "results", Path(directory) / "matrix.csv")
        self.assertEqual(len(rows), 28 * 4)
        self.assertTrue(all(row["status"] == "missing" for row in rows))


if __name__ == "__main__":
    unittest.main()
