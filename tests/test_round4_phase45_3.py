import unittest

from scripts.round4_phase45_3 import _conditions, _condition_name, _select_conditions
from source.common.round4_phase45_3 import condition_key


class Round4Phase453Tests(unittest.TestCase):
    def test_matrix_contains_only_dev_condition_axes(self):
        conditions = _conditions()

        self.assertEqual(len(conditions), 63)
        self.assertEqual(
            sum(item["mode"] == "P1" for item in conditions),
            3,
        )
        self.assertEqual(
            sum(item["mode"] == "P2" for item in conditions),
            36,
        )
        self.assertTrue(all(item["cem_protocol"] != "legacy" for item in conditions if item["mode"] in {"P0", "P3"}))
        self.assertTrue(all(_condition_name(item).endswith("/dev") for item in conditions))

    def test_condition_key_requires_explicit_protocol_metadata(self):
        payload = {
            "task": "reacher",
            "round4_mode": "P2",
            "round4_planning": {
                "cem_protocol": "cem-scale",
                "action_bound_mode": "candidate_scale",
                "action_flow_steps": 16,
                "action_flow_integrator": "heun",
            },
        }

        self.assertEqual(
            condition_key(payload),
            ("reacher", "P2", "cem-scale", 16, "heun"),
        )
        payload["round4_planning"]["action_bound_mode"] = "candidate_clip"
        with self.assertRaises(ValueError):
            condition_key(payload)

    def test_condition_index_selection_supports_late_shards(self):
        conditions = _conditions()

        selected = _select_conditions(conditions, [50, 62])

        self.assertEqual(selected, [conditions[50], conditions[62]])
        with self.assertRaises(ValueError):
            _select_conditions(conditions, [50, 50])
        with self.assertRaises(ValueError):
            _select_conditions(conditions, [63])


if __name__ == "__main__":
    unittest.main()
