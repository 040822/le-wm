import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_fast_lewam_serial_one_step.sh"


class FastLeWAMSerialRunnerTests(unittest.TestCase):
    def test_runner_rejects_prohibited_gpu_before_launch(self):
        result = subprocess.run(
            ["bash", str(RUNNER), "reacher", "4"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("only GPU0-GPU3", result.stderr)

    def test_runner_rejects_unknown_task_before_launch(self):
        result = subprocess.run(
            ["bash", str(RUNNER), "cube", "1"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("expected reacher or pusht", result.stderr)

    def test_runner_enforces_fixed_task_gpu_mapping(self):
        result = subprocess.run(
            ["bash", str(RUNNER), "reacher", "2"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Reacher on GPU1", result.stderr)


if __name__ == "__main__":
    unittest.main()
