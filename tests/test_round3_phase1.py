import unittest

import numpy as np

from source.common.round3_protocol import (
    PREDICATE_VERSION,
    TASK_PREDICATES,
    evaluate_success,
    physical_distance,
)


class Round3PredicateTests(unittest.TestCase):
    def test_static_predicates_match_runtime_fields_units_and_boundaries(self):
        self.assertEqual(PREDICATE_VERSION, "round3_runtime_predicates_v1")
        self.assertEqual(TASK_PREDICATES["cube"].current_field, "privileged_block_0_pos")
        self.assertEqual(TASK_PREDICATES["cube"].goal_field, "goal_privileged_block_0_pos")
        self.assertEqual(TASK_PREDICATES["cube"].unit, "m")
        self.assertEqual(TASK_PREDICATES["reacher"].unit, "rad")
        self.assertEqual(TASK_PREDICATES["pusht"].unit, "px/rad")
        self.assertEqual(TASK_PREDICATES["tworoom"].unit, "px")

        self.assertTrue(evaluate_success("cube", [0.04, 0.0, 0.0], [0.0, 0.0, 0.0]))
        self.assertFalse(evaluate_success("cube", [0.040001, 0.0, 0.0], [0.0, 0.0, 0.0]))

        self.assertTrue(evaluate_success("reacher", [0.049, -0.049], [0.0, 0.0]))
        self.assertFalse(evaluate_success("reacher", [0.05, 0.0], [0.0, 0.0]))

        self.assertTrue(evaluate_success("pusht", [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]))
        self.assertFalse(evaluate_success("pusht", [20, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]))
        self.assertFalse(
            evaluate_success(
                "pusht",
                [0, 0, 0, 0, np.pi / 9, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            )
        )

        self.assertTrue(evaluate_success("tworoom", [15.999, 0], [0, 0]))
        self.assertFalse(evaluate_success("tworoom", [16.0, 0], [0, 0]))

    def test_physical_distance_is_vectorized_and_keeps_task_geometry(self):
        np.testing.assert_allclose(
            physical_distance("cube", [[0, 0, 0], [3, 4, 0]], [0, 0, 0]),
            [0.0, 5.0],
        )
        np.testing.assert_allclose(
            physical_distance("reacher", [[1, 2], [4, 6]], [0, 0]),
            [np.sqrt(5), np.sqrt(52)],
        )
        np.testing.assert_allclose(
            physical_distance("pusht", [[1, 2, 3, 4, 0, 5, 6]], [0, 0, 0, 0, 0, 0, 0]),
            [np.sqrt(91)],
        )


if __name__ == "__main__":
    unittest.main()
