import json
import tempfile
import unittest
from pathlib import Path

from source.common.round4_flow_steps import analyze_flow_step_results


def _write_result(
    root: Path,
    task: str,
    mode: str,
    cohort: str,
    success: list[bool],
    *,
    action_flow_steps: int | None = None,
):
    target = root / task / mode / cohort
    target.mkdir(parents=True)
    episodes = [
        {
            "episode_id": index,
            "start_step": 10,
            "success": value,
        }
        for index, value in enumerate(success)
    ]
    payload = {
        "status": "ok",
        "task": task,
        "round4_mode": mode,
        "cohort_kind": cohort,
        "cohort_id": f"{task}_{cohort}",
        "cohort_sha256": f"{task}-{cohort}-sha",
        "checkpoint": "r4-ab-epoch10.pt",
        "success_rate": sum(success) / len(success),
        "episodes": episodes,
        "round4_planning": {
            "action_flow_steps": action_flow_steps,
            "planning_median_seconds": 1.25,
            "planning_p95_seconds": 2.0,
            "forward_count": 64,
            "peak_memory_bytes": 123,
        },
    }
    (target / "result.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


class Round4FlowStepTests(unittest.TestCase):
    def test_step_analysis_reports_paired_delta_and_invariant_p1(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            step_one = root / "steps_1"
            step_sixteen = root / "steps_16"
            invariant = root / "invariant_p1"
            _write_result(
                step_one,
                "cube",
                "P3",
                "dev",
                [False, True, False, False],
                action_flow_steps=1,
            )
            _write_result(
                step_sixteen,
                "cube",
                "P3",
                "dev",
                [True, True, False, False],
                action_flow_steps=16,
            )
            _write_result(
                invariant,
                "cube",
                "P1",
                "dev",
                [True, True, True, False],
            )

            result = analyze_flow_step_results(
                {1: step_one, 16: step_sixteen},
                invariant,
                tasks=("cube",),
                action_modes=("P3",),
                cohort_kinds=("dev",),
            )

            step_row = next(
                row
                for row in result["rows"]
                if row["mode"] == "P3" and row["flow_steps"] == 1
            )
            self.assertEqual(step_row["success_rate_percent"], 25.0)
            self.assertEqual(step_row["canonical_success_rate_percent"], 50.0)
            self.assertEqual(step_row["delta_pp_vs_canonical"], -25.0)
            self.assertEqual(step_row["paired_vs_canonical"]["improved"], 0)
            self.assertEqual(step_row["paired_vs_canonical"]["regressed"], 1)

            p1_row = next(row for row in result["rows"] if row["mode"] == "P1")
            self.assertTrue(p1_row["invariant"])
            self.assertIsNone(p1_row["flow_steps"])
            self.assertEqual(p1_row["success_rate_percent"], 75.0)

    def test_step_analysis_rejects_wrong_action_flow_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            step_one = root / "steps_1"
            step_sixteen = root / "steps_16"
            invariant = root / "invariant_p1"
            _write_result(
                step_one,
                "cube",
                "P0",
                "dev",
                [True],
                action_flow_steps=2,
            )
            _write_result(
                step_sixteen,
                "cube",
                "P0",
                "dev",
                [True],
                action_flow_steps=16,
            )
            _write_result(invariant, "cube", "P1", "dev", [True])

            with self.assertRaisesRegex(ValueError, "action_flow_steps"):
                analyze_flow_step_results(
                    {1: step_one, 16: step_sixteen},
                    invariant,
                    tasks=("cube",),
                    action_modes=("P0",),
                    cohort_kinds=("dev",),
                )


if __name__ == "__main__":
    unittest.main()
