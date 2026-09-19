import unittest

import numpy as np

from scripts.round5_phase3 import fast_conditions, p2_protocol
from source.common.eval import select_eval_cohort
from source.common.phase3_compat import build_phase3_legacy_manifest


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
