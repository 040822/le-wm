import json
import tempfile
import unittest
from pathlib import Path

import yaml

from source.common.round4_compare import compare_result_roots


def _write_result(root: Path, task: str, mode: str, cohort: str, success: list[bool]) -> None:
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
        "cohort_sha256": "cohort-sha",
        "success_rate": sum(success) / len(success),
        "episodes": episodes,
        "round4_planning": {
            "planning_median_seconds": 1.25,
            "planning_p95_seconds": 2.0,
            "forward_count": 64,
            "peak_memory_bytes": 123,
        },
    }
    (target / "result.json").write_text(json.dumps(payload), encoding="utf-8")


class Round4ABCompareTests(unittest.TestCase):
    def test_round4_ab_config_keeps_shared_round4_model_and_disables_de_losses(self):
        config = yaml.safe_load(
            Path("config/train/round4_ab.yaml").read_text(encoding="utf-8")
        )
        policy = yaml.safe_load(
            Path("config/train/policy/round4_ab.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["train_mode"], "stage_ab")
        self.assertEqual(config["loss"]["d"]["lambda"], 0.0)
        self.assertEqual(config["loss"]["e"]["lambda"], 0.0)
        self.assertEqual(
            policy["model"]["_target_"], "source.model.fast_lewam.round4.Round4FastLeWAM"
        )

    def test_compare_result_roots_aligns_cohorts_and_reports_paired_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ab = root / "ab"
            abde = root / "abde"
            _write_result(ab, "cube", "P3", "dev", [False, True, False, False])
            _write_result(abde, "cube", "P3", "dev", [True, True, False, False])

            result = compare_result_roots(
                ab,
                abde,
                tasks=("cube",),
                modes=("P3",),
                cohort_kinds=("dev",),
            )

            row = result["rows"][0]
            self.assertEqual(row["delta_pp"], 25.0)
            self.assertEqual(row["paired"]["improved"], 1)
            self.assertEqual(row["paired"]["regressed"], 0)
            self.assertEqual(row["baseline_planning"]["forward_count"], 64)

    def test_compare_rejects_different_cohort_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ab = root / "ab"
            abde = root / "abde"
            _write_result(ab, "cube", "P3", "dev", [False, True])
            _write_result(abde, "cube", "P3", "dev", [True, True])
            payload_path = abde / "cube" / "P3" / "dev" / "result.json"
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            payload["cohort_sha256"] = "different-cohort"
            payload_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "cohort identity"):
                compare_result_roots(
                    ab,
                    abde,
                    tasks=("cube",),
                    modes=("P3",),
                    cohort_kinds=("dev",),
                )


if __name__ == "__main__":
    unittest.main()
