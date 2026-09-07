import unittest

from source.common.round3_protocol import audit_all_predicates


class RuntimeAuditTests(unittest.TestCase):
    def test_installed_runtime_sources_are_available_for_all_tasks(self):
        audits = audit_all_predicates()
        self.assertEqual(set(audits), {"cube", "pusht", "reacher", "tworoom"})
        for task, audit in audits.items():
            self.assertEqual(audit["status"], "accepted", task)
            self.assertEqual(audit["goal_refresh"]["checks_expected"], 10)


if __name__ == "__main__":
    unittest.main()
