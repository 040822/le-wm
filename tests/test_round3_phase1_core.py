import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from source.common.round3_phase1 import (
    CohortManifest,
    build_revised_cohorts,
    paired_comparison,
    run_goal_refresh_checks,
    summarize_episodes,
    wilson_interval,
)


class FakeDataset:
    def __init__(self, columns):
        self._columns = {key: np.asarray(value) for key, value in columns.items()}
        self.column_names = tuple(self._columns)

    def get_col_data(self, name):
        return self._columns[name]


def make_dataset(episode_count=300, episode_length=30):
    episodes = []
    steps = []
    states = []
    for episode in range(episode_count):
        for step in range(episode_length):
            angle = episode * 0.037 + step * (0.11 + episode * 0.0007)
            episodes.append(episode)
            steps.append(step)
            states.append([np.cos(angle), np.sin(angle), 0.0])
    return FakeDataset(
        {
            "episode_idx": episodes,
            "step_idx": steps,
            "privileged_block_0_pos": states,
        }
    )


class Round3CohortTests(unittest.TestCase):
    def test_revised_manifests_are_reproducible_hashed_and_disjoint(self):
        dataset = make_dataset()
        first = build_revised_cohorts(
            dataset,
            task="cube",
            seed=42,
            dev_count=50,
            final_count=100,
            online_count=20,
        )
        second = build_revised_cohorts(
            dataset,
            task="cube",
            seed=42,
            dev_count=50,
            final_count=100,
            online_count=20,
        )
        self.assertEqual(first["dev"].as_dict(), second["dev"].as_dict())
        self.assertEqual(first["final"].as_dict(), second["final"].as_dict())
        self.assertEqual(first["dev"].cohort_sha256, first["dev"].computed_sha256)
        self.assertEqual(len(first["dev"].entries), 50)
        self.assertEqual(len(first["final"].entries), 100)
        dev = {entry.episode_id for entry in first["dev"].entries}
        final = {entry.episode_id for entry in first["final"].entries}
        self.assertFalse(dev & final)
        self.assertEqual(first["dev"].selected_counts, (10, 10, 10, 10, 10))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dev.json"
            first["dev"].save(path)
            loaded = CohortManifest.load(path)
            self.assertEqual(loaded.computed_sha256, first["dev"].computed_sha256)

    def test_manifest_tampering_is_rejected(self):
        manifest = build_revised_cohorts(
            make_dataset(episode_count=300),
            task="cube",
            dev_count=50,
            final_count=100,
            online_count=20,
        )["dev"]
        value = manifest.as_dict()
        value["entries"][0]["start_step"] += 1
        with self.assertRaises(ValueError):
            CohortManifest.from_dict(value)


class Round3AnalysisTests(unittest.TestCase):
    def test_wilson_and_paired_categories(self):
        low, high = wilson_interval([True, False, True, False])
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        baseline = [
            {"episode_id": 1, "start_step": 0, "success": False},
            {"episode_id": 2, "start_step": 0, "success": True},
            {"episode_id": 3, "start_step": 0, "success": True},
            {"episode_id": 4, "start_step": 0, "success": False},
        ]
        candidate = [
            {"episode_id": 1, "start_step": 0, "success": True},
            {"episode_id": 2, "start_step": 0, "success": False},
            {"episode_id": 3, "start_step": 0, "success": True},
            {"episode_id": 4, "start_step": 0, "success": False},
        ]
        comparison = paired_comparison(baseline, candidate)
        self.assertEqual(
            {comparison[key] for key in ("improved", "regressed", "stable_success", "stable_failure")},
            {1},
        )
        self.assertEqual(comparison["delta_pp"], 0.0)
        summary = summarize_episodes(
            [
                {"success": True, "initial_success": False, "start_distance": 2, "terminal_distance": 0.1, "first_success_step": 4, "hold_success": True},
                {"success": False, "initial_success": False, "start_distance": 4, "terminal_distance": 1.0, "first_success_step": None, "hold_success": False},
            ]
        )
        self.assertEqual(summary["successes"], 1)
        self.assertEqual(summary["distance_quantiles"]["median"], 3.0)

    def test_goal_refresh_adapter_records_ten_checks(self):
        state = {"goal": np.zeros(2), "current": np.ones(2)}
        cases = tuple(
            {"old_goal": [0.0, 0.0], "new_goal": [1.0 + index, 1.0 + index], "current": [1.0 + index, 1.0 + index]}
            for index in range(10)
        )

        def set_goal(value):
            state["goal"] = np.asarray(value)

        result = run_goal_refresh_checks(
            "tworoom",
            cases,
            set_goal=set_goal,
            read_goal=lambda: state["goal"],
            read_success=lambda: np.linalg.norm(state["current"] - state["goal"]) < 16,
        )
        self.assertEqual(result["checks_run"], 10)
        self.assertEqual(result["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
