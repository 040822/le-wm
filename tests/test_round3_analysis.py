import unittest

from source.common.round3_analysis import relabel_trace


class RelabelTests(unittest.TestCase):
    def test_relabel_uses_same_trace_and_keeps_missing_fields_explicit(self):
        records = [
            {
                "episode_id": 1,
                "steps": [
                    {
                        "raw_env_step": 1,
                        "current": [0.0, 0.0],
                        "goal": [0.0, 0.0],
                        "neutral_action": True,
                        "action_legal": True,
                    },
                    {
                        "raw_env_step": 2,
                        "current": None,
                        "goal": None,
                        "neutral_action": True,
                        "action_legal": True,
                    },
                ],
            }
        ]
        result = relabel_trace(records, task="tworoom", action_block=1)
        self.assertTrue(result[0]["success"])
        self.assertEqual(result[0]["relabel_status"], "insufficient_fields")
        self.assertEqual(records[0]["steps"][0].get("relabelled_success"), None)


if __name__ == "__main__":
    unittest.main()
