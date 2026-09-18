from unittest import mock
import unittest

from scripts.round3_resource_preflight import _parse_processes, preflight


class Round3ResourcePreflightTests(unittest.TestCase):
    def test_only_distinct_allowed_gpus_are_accepted(self):
        with self.assertRaises(ValueError):
            preflight([4])
        with self.assertRaises(ValueError):
            preflight([0, 0])

    def test_pmon_parser_ignores_idle_rows(self):
        self.assertEqual(_parse_processes("# gpu pid type sm mem enc dec command\n0 - - - - - - -\n", 0), [])

    def test_shared_mode_keeps_memory_guard_and_records_existing_processes(self):
        pmon = mock.Mock(stdout="# gpu pid type sm mem enc dec command\n1 123 C 20 1 - - - - python\n")
        memory = mock.Mock(stdout="39000\n")
        with mock.patch("scripts.round3_resource_preflight._run", side_effect=[pmon, memory]):
            result = preflight([1], minimum_free_mib=24000, allow_existing_compute=True)
        self.assertTrue(result["shared_compute_allowed"])
        self.assertEqual(result["free_memory_mib"]["1"], 39000)
        self.assertEqual(result["compute_processes"]["1"], ["1 123 C 20 1 - - - - python"])


if __name__ == "__main__":
    unittest.main()
