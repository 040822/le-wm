import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_fast_lewam_experiment import summarize_experiment


class FastLeWAMExperimentSummaryTests(unittest.TestCase):
    def test_verifies_artifacts_and_summarizes_results_and_resources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            checkpoints = run_dir / "checkpoints"
            checkpoints.mkdir()
            (checkpoints / "fast_lewam_weights_epoch_10.pt").write_bytes(b"weights")
            (checkpoints / "last.ckpt").write_bytes(b"last")

            for stage, success_rate in (("stage_a", 92.0), ("stage_b", 84.0)):
                result_dir = run_dir / "eval" / "epoch_10" / stage
                result_dir.mkdir(parents=True)
                (result_dir / "result.json").write_text(
                    json.dumps(
                        {
                            "status": "ok",
                            "success_rate": success_rate,
                            "evaluation_seconds": 12.5,
                            "parameters": {"num_eval": 50},
                        }
                    ),
                    encoding="utf-8",
                )

            with (run_dir / "gpu_memory_samples.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["timestamp", "phase", "memory_used_mib"],
                )
                writer.writeheader()
                writer.writerow(
                    {"timestamp": 1, "phase": "train", "memory_used_mib": 100}
                )
                writer.writerow(
                    {"timestamp": 2, "phase": "eval", "memory_used_mib": 120}
                )

            metrics_dir = run_dir / "lightning_logs" / "version_0"
            metrics_dir.mkdir(parents=True)
            with (metrics_dir / "metrics.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "validate/stage_b_clean_terminal_mse",
                        "validate/stage_b_predicted_terminal_mse",
                        "validate/stage_b_expert_preference_accuracy",
                        "validate/stage_b_expert_top1_rate",
                        "validate/stage_b_expert_negative_margin",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "validate/stage_b_clean_terminal_mse": 0.25,
                        "validate/stage_b_predicted_terminal_mse": 0.5,
                        "validate/stage_b_expert_preference_accuracy": 0.75,
                        "validate/stage_b_expert_top1_rate": 0.5,
                        "validate/stage_b_expert_negative_margin": 0.1,
                    }
                )

            summary = summarize_experiment(
                run_dir,
                elapsed_seconds=42,
                stages=("stage_a", "stage_b"),
            )

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["elapsed_seconds"], 42)
            self.assertEqual(summary["peak_device_memory_used_mib"], 120)
            self.assertEqual(summary["stages"]["stage_b"]["success_rate"], 84.0)
            self.assertEqual(
                summary["validation"]["validate/stage_b_clean_terminal_mse"],
                0.25,
            )
            self.assertTrue((run_dir / "acceptance_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
