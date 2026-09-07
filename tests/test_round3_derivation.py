import tempfile
import unittest
from pathlib import Path

from source.common.round3_derivation import derive_protocol_result_file
from source.common.round3_phase1 import CohortEntry, CohortManifest


class Round3DerivationTests(unittest.TestCase):
    def test_relabelled_result_keeps_trace_and_uses_target_manifest(self):
        entry = CohortEntry(
            row_index=2,
            episode_id=9,
            start_step=1,
            goal_row_index=7,
            goal_step=6,
            start_distance=1.0,
            initially_successful=False,
        )
        manifest = CohortManifest(
            task="cube",
            cohort_id="cube_dev_tolerance_revised",
            cohort_kind="dev",
            protocol_variant="tolerance_revised",
            seed=42,
            goal_offset_steps=25,
            entries=(entry,),
            episode_split={"dev": (9,), "final": (), "online": ()},
        )
        source = {
            "protocol": "round3_phase1_v1",
            "protocol_variant": "legacy",
            "status": "ok",
            "cohort_sha256": "legacy-hash",
            "episodes": [
                {
                    "episode_id": 9,
                    "start_step": 1,
                    "row_index": 2,
                    "success": False,
                    "steps": [
                        {
                            "raw_env_step": 1,
                            "current": [0.0, 0.0, 0.0],
                            "goal": [0.03, 0.0, 0.0],
                            "action_legal": True,
                            "neutral_action": True,
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.json"
            manifest_path = root / "manifest.json"
            target = root / "result"
            source_path.write_text(__import__("json").dumps(source), encoding="utf-8")
            manifest.save(manifest_path)
            result = derive_protocol_result_file(
                source_path,
                target,
                manifest_path=manifest_path,
                task="cube",
                protocol_variant="tolerance_revised",
                trace_output_dir=root / "trace",
            )
            self.assertEqual(result["protocol_variant"], "tolerance_revised")
            self.assertEqual(result["cohort_sha256"], manifest.computed_sha256)
            self.assertEqual(result["summary"]["successes"], 1)
            self.assertTrue((root / "trace" / "episodes.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
