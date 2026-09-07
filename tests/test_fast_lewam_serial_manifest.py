import unittest

from scripts.prepare_fast_lewam_serial_control_manifest import _slot_categories


def _result(successes):
    return {
        "parameters": {
            "episode_ids": list(range(len(successes))),
            "start_steps": list(range(len(successes))),
            "start_rows": list(range(len(successes))),
        },
        "episodes": [{"success": value} for value in successes],
    }


class FastLeWAMSerialManifestTests(unittest.TestCase):
    def test_slot_categories_cover_the_fixed_parallel_serial_cohort(self):
        categories = _slot_categories(
            _result([True, False, True, False]),
            _result([False, True, True, False]),
        )

        self.assertEqual(categories["regression"], [0])
        self.assertEqual(categories["improvement_control"], [1])
        self.assertEqual(categories["stable_success"], [2])
        self.assertEqual(categories["stable_failure"], [3])

    def test_slot_categories_reject_different_cohorts(self):
        parallel = _result([True])
        serial = _result([True])
        serial["parameters"]["start_rows"] = [99]

        with self.assertRaisesRegex(ValueError, "start_rows"):
            _slot_categories(parallel, serial)


if __name__ == "__main__":
    unittest.main()
