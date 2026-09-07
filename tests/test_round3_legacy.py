import unittest

import numpy as np

from source.common.eval import select_eval_cohort
from source.common.round3_phase1 import build_legacy_manifest


class Dataset:
    column_names = ("episode_idx", "step_idx", "privileged_block_0_pos")

    def __init__(self):
        self.values = {
            "episode_idx": np.repeat(np.arange(3), 30),
            "step_idx": np.tile(np.arange(30), 3),
            "privileged_block_0_pos": np.zeros((90, 3)),
        }

    def get_col_data(self, name):
        return self.values[name]
    def get_row_data(self, rows):
        rows = np.asarray(rows)
        return {key: value[rows] for key, value in self.values.items()}


class LegacyCompatibilityTests(unittest.TestCase):
    def test_rows_match_existing_selector_and_record_last_row_quirk(self):
        dataset = Dataset()
        expected = select_eval_cohort(
            dataset,
            goal_offset_steps=25,
            num_eval=5,
            seed=42,
        )
        manifest = build_legacy_manifest(
            dataset,
            task="cube",
            goal_offset_steps=25,
            num_eval=5,
            seed=42,
        )
        self.assertEqual(manifest.protocol_variant, "legacy")
        self.assertEqual(
            [entry.row_index for entry in manifest.entries],
            expected.row_indices.tolist(),
        )
        self.assertEqual(
            [entry.start_step for entry in manifest.entries],
            expected.start_steps.tolist(),
        )
        self.assertIsNotNone(manifest.diagnostics["global_last_row_excluded"])


if __name__ == "__main__":
    unittest.main()
