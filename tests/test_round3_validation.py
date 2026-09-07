import unittest

from source.common.round3_validation import validate_result_payload


class ResultValidationTests(unittest.TestCase):
    def test_incomplete_result_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_result_payload(
                {
                    "protocol": "round3_phase1_v1",
                    "status": "ok",
                    "predicate_version": "round3_runtime_predicates_v1",
                    "cohort_sha256": "0" * 64,
                    "episodes": [{"success": True}],
                },
                expected_count=2,
            )

    def test_complete_result_is_accepted(self):
        validate_result_payload(
            {
                "protocol": "round3_phase1_v1",
                "status": "ok",
                "predicate_version": "round3_runtime_predicates_v1",
                "cohort_sha256": "0" * 64,
                "episodes": [{"success": True}],
                "summary": {"success_vector": [True]},
            },
            expected_count=1,
        )


if __name__ == "__main__":
    unittest.main()
