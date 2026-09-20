import json
from pathlib import Path
import tempfile
import unittest

from source.common.round3_phase1 import CohortManifest
from source.common.round5_phase1_epoch_curve import (
    CURVE_DEFAULT_EXPECTED_COUNTS,
    CURVE_EPOCHS,
    CURVE_NEW_EPOCHS,
    baseline_source_path,
    checkpoint_hashes,
    checkpoint_paths,
    condition_key_from_spec,
    condition_specs,
    dump_json_atomic,
    job_specs,
    load_epoch_curve_results,
    reference_path,
    validate_declared_checkpoint_hashes,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/round5/phase1_epoch_curve.json"


class Round5Phase1EpochCurveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_matrix_and_job_counts_match_registered_protocol(self):
        specs = condition_specs(self.config)
        self.assertEqual(len(specs), 34)
        self.assertEqual(
            {task: sum(item["task"] == task for item in specs) for task in CURVE_DEFAULT_EXPECTED_COUNTS},
            CURVE_DEFAULT_EXPECTED_COUNTS,
        )
        jobs = job_specs(self.config)
        self.assertEqual(len(jobs), 340)
        self.assertEqual(sum(item["epoch"] in CURVE_NEW_EPOCHS for item in jobs), 306)
        self.assertEqual(sum(item["epoch"] == 10 for item in jobs), 34)
        self.assertEqual(len({item["job_id"] for item in jobs}), 340)

    def test_condition_matrix_has_only_registered_modes_and_protocols(self):
        specs = condition_specs(self.config)
        for spec in specs:
            self.assertEqual(spec["guidance"], "none")
            if spec["mode"] == "P1":
                self.assertIsNone(spec["action_flow_steps"])
                self.assertEqual(spec["action_flow_integrator"], "not_applicable")
            if spec["task"] == "reacher" and spec["mode"] in {"P1", "P2"}:
                self.assertEqual(spec["cem_protocol"], "cem-clip")
            if spec["task"] != "reacher" and spec["mode"] in {"P1", "P2"}:
                self.assertEqual(spec["cem_protocol"], "legacy")

    def test_epoch_checkpoint_discovery_and_declared_epoch10_hashes(self):
        paths = checkpoint_paths(self.config, root=ROOT)
        self.assertEqual(set(paths), {"cube", "pusht", "reacher", "tworoom"})
        for task in paths:
            self.assertEqual(tuple(sorted(paths[task])), CURVE_EPOCHS)
            for epoch, path in paths[task].items():
                self.assertTrue(path.is_file(), (task, epoch, path))
                self.assertIn(f"epoch_{epoch}.pt", path.name)
        hashes = checkpoint_hashes(paths)
        validate_declared_checkpoint_hashes(self.config, hashes)

    def test_epoch10_references_are_validated_and_aggregate_as_34_rows(self):
        manifests = {
            task: CohortManifest.load(
                ROOT / self.config["cohort"]["legacy"]["paths"][task]
            )
            for task in ("cube", "pusht", "reacher", "tworoom")
        }
        checkpoints = checkpoint_paths(self.config, root=ROOT)
        hashes = checkpoint_hashes(checkpoints)
        with tempfile.TemporaryDirectory(prefix="round5-epoch-curve-") as directory:
            output_root = Path(directory)
            for spec in condition_specs(self.config):
                source = baseline_source_path(
                    self.config,
                    spec["task"],
                    spec,
                    root=ROOT,
                ).resolve()
                target = reference_path(output_root, spec["task"], 10, spec)
                dump_json_atomic(
                    target,
                    {
                        "schema_version": "round5_phase1_epoch_curve_v1",
                        "status": "ok",
                        "kind": "epoch10_reference",
                        "task": spec["task"],
                        "epoch": 10,
                        "job_id": "test",
                        "condition": spec,
                        "source": "round4_phase45_4_legacy",
                        "source_result": str(source),
                        "source_result_sha256": __import__(
                            "source.common.round5_phase1_epoch_curve",
                            fromlist=["sha256_file"],
                        ).sha256_file(source),
                        "checkpoint": str(checkpoints[spec["task"]][10]),
                        "checkpoint_sha256": hashes[spec["task"]][10],
                    },
                )
            indexed, validation = load_epoch_curve_results(
                self.config,
                output_root,
                manifests=manifests,
                checkpoints=checkpoints,
                hashes=hashes,
                root=ROOT,
                require_complete=False,
            )
            self.assertEqual(len(indexed), 34)
            self.assertEqual(validation["reuse_result_count"], 34)
            self.assertEqual(validation["new_result_count"], 0)
            self.assertEqual(validation["expected_count"], 340)
            self.assertEqual(validation["missing_count"], 306)
            self.assertTrue(validation["all_status_ok"])

    def test_checkpoint_or_epoch_mismatch_is_rejected(self):
        jobs = job_specs(self.config)
        self.assertNotEqual(
            condition_key_from_spec(jobs[0]),
            condition_key_from_spec({**jobs[0], "mode": "P3"}),
        )
        with self.assertRaises(ValueError):
            checkpoint_paths(
                {
                    **self.config,
                    "training": {
                        **self.config["training"],
                        "checkpoint_patterns": {
                            **self.config["training"]["checkpoint_patterns"],
                            "cube": self.config["training"]["checkpoint_patterns"]["cube"].replace(
                                "{epoch}", "10"
                            ),
                        },
                    },
                },
                root=ROOT,
            )


if __name__ == "__main__":
    unittest.main()
