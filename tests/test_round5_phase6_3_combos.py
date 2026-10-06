from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import tempfile
import unittest

from scripts.round5_phase6_3 import METHODS
from scripts.round5_phase6_3_combos import (
    closed_loop_e4_fingerprint,
    enter_components,
    validate_selection,
)


def alias(components, target):
    return {"status": "alias", "components": components, "alias_of": target}


def selected(components):
    return {"status": "selected", "components": components}


def empty_selection():
    return {
        "schema_version": 1,
        "versions": {
            "C-FP32": {method: alias([], "E0") for method in METHODS},
            "C-BF16": {method: alias([], "C-FP32") for method in METHODS},
        },
    }


class Phase63CombinationTests(unittest.TestCase):
    def test_accepts_noop_aliases_and_selected_combinations(self):
        spec = empty_selection()
        spec["versions"]["C-FP32"]["P0"] = alias(["E2"], "E2")
        spec["versions"]["C-BF16"]["P0"] = alias(["E2"], "C-FP32")
        method = "P3-PO-refine-L"
        spec["versions"]["C-FP32"][method] = selected(["E1", "E3"])
        spec["versions"]["C-BF16"][method] = selected(["E1", "E3", "E4"])

        normalized = validate_selection(spec)

        self.assertEqual(normalized["versions"]["C-BF16"][method]["components"],
                         ["E1", "E3", "E4"])

    def test_rejects_inapplicable_e1_and_bf16_without_e4(self):
        spec = empty_selection()
        spec["versions"]["C-FP32"]["P0"] = selected(["E1", "E2"])
        with self.assertRaisesRegex(ValueError, "does not apply"):
            validate_selection(spec)

        spec = empty_selection()
        spec["versions"]["C-BF16"]["P0"] = selected(["E2"])
        with self.assertRaisesRegex(ValueError, "must include E4"):
            validate_selection(spec)

    def test_component_scopes_enter_and_exit_in_compatible_order(self):
        events = []

        def scope(name):
            @contextmanager
            def context():
                events.append("enter:" + name)
                try:
                    yield
                finally:
                    events.append("exit:" + name)
            return context

        with ExitStack() as stack:
            enter_components(stack, ["E1", "E2", "E3", "E4"],
                             preprocess=scope("E3"), encoder=scope("E4"),
                             cache=scope("E2"), deferred_checks=scope("E1"))
            self.assertEqual(events, ["enter:E3", "enter:E4", "enter:E2", "enter:E1"])

        self.assertEqual(events, ["enter:E3", "enter:E4", "enter:E2", "enter:E1",
                                  "exit:E1", "exit:E2", "exit:E4", "exit:E3"])

    def test_e4_evidence_fingerprint_ignores_later_non_e4_rows(self):
        report = {
            "runner_sha256": "first",
            "task_equal_average": {"E4": {"P0": 0.01}, "E0": {"P0": 0.0}},
            "conditions": [
                {"version": "E4", "method": "P0", "task": "tworoom", "status": "complete"},
                {"version": "E0", "method": "P0", "task": "tworoom", "status": "complete"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.json"
            path.write_text(json.dumps(report))
            original = closed_loop_e4_fingerprint(path)
            report["runner_sha256"] = "second"
            report["task_equal_average"]["E0"]["P0"] = 1.0
            report["conditions"][1]["status"] = "incomplete"
            path.write_text(json.dumps(report))
            self.assertEqual(closed_loop_e4_fingerprint(path), original)
            report["conditions"][0]["status"] = "incomplete"
            path.write_text(json.dumps(report))
            self.assertNotEqual(closed_loop_e4_fingerprint(path), original)


if __name__ == "__main__":
    unittest.main()
