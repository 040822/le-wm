import unittest
from types import SimpleNamespace

import numpy as np

from source.common.round3_phase1 import CohortEntry, CohortManifest, Round3TraceCollector
from source.common.round3_eval import validate_gpu_visibility
from source.common.eval import compose_eval_config
from source.common.round3_eval import validate_round3_config


class TraceTests(unittest.TestCase):
    def test_trace_records_predicate_distance_and_neutral_hold_window(self):
        entry = CohortEntry(
            row_index=10,
            episode_id=7,
            start_step=0,
            goal_row_index=35,
            goal_step=25,
            start_distance=1.0,
            initially_successful=False,
            start_state=(1.0, 0.0, 0.0),
            goal_state=(0.0, 0.0, 0.0),
        )
        manifest = CohortManifest(
            task="cube",
            cohort_id="cube_dev_round3_v1",
            cohort_kind="dev",
            protocol_variant="round3_revised",
            seed=42,
            goal_offset_steps=25,
            entries=(entry,),
            episode_split={"dev": (7,), "final": (), "online": ()},
        )
        action_space = SimpleNamespace(
            low=np.asarray([-1.0, -1.0]),
            high=np.asarray([1.0, 1.0]),
        )
        collector = Round3TraceCollector(
            "cube",
            manifest,
            action_block=5,
            action_space=action_space,
            neutral_action=[0.25, 0.0],
        )
        for step in range(5):
            collector.record_step(
                np.asarray([[0.25, 0.0]]),
                {
                    "privileged_block_0_pos": np.asarray([[0.0, 0.0, 0.0]]),
                    "goal_privileged_block_0_pos": np.asarray([[0.0, 0.0, 0.0]]),
                    "success": np.asarray([True]),
                    "terminated": np.asarray([False]),
                    "truncated": np.asarray([False]),
                },
                raw_env_step=step + 1,
            )
        record = collector.finalize([True], eval_budget=50)[0]
        self.assertEqual(record["first_success_step"], 1)
        self.assertEqual(record["terminal_distance"], 0.0)
        self.assertTrue(record["hold_success"])
        self.assertEqual(record["missing_field_count"], 0)
        self.assertEqual(record["invalid_action_count"], 0)
        self.assertEqual(record["steps"][0]["neutral_action_source"], "declared")

        validate_round3_config(compose_eval_config("cube"))
        validate_gpu_visibility("cpu")

if __name__ == "__main__":
    unittest.main()
