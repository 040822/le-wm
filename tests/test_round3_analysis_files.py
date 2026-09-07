import json
import tempfile
import unittest
from pathlib import Path

from source.common.round3_analysis import analyze_protocol_files


class ProtocolFileAnalysisTests(unittest.TestCase):
    def test_summaries_are_computed_from_episode_lists(self):
        episode = {
            "episode_id": 1,
            "start_step": 0,
            "success": True,
            "initial_success": False,
            "hold_success": False,
            "start_distance": 1.0,
            "terminal_distance": 0.5,
            "first_success_step": 3,
            "steps": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for variant in ("legacy", "sampling_revised", "tolerance_revised", "round3_revised"):
                path = root / f"{variant}.json"
                path.write_text(json.dumps({"episodes": [episode]}), encoding="utf-8")
                paths[variant] = str(path)
            result = analyze_protocol_files(paths, task="cube")
            self.assertEqual(result["summaries"]["legacy"]["num_episodes"], 1)
            self.assertEqual(result["summaries"]["legacy"]["successes"], 1)
            self.assertEqual(result["summaries"]["round3_revised"]["num_episodes"], 1)


if __name__ == "__main__":
    unittest.main()
