import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from source.common.eval import EvaluationIdentity, compose_eval_config
from source.common.round3_eval import run_round3_evaluation
from source.common.round3_phase1 import (
    CohortEntry,
    CohortManifest,
    Round3TraceCollector,
    summarize_episodes,
)
from source.common.round3_protocol import ROUND3_PROTOCOL
from source.common.round3_validation import validate_cohort_manifest, validate_result_payload


def _manifest():
    return CohortManifest(
        task="cube",
        cohort_id="cube_dev_round3_v1",
        cohort_kind="dev",
        protocol_variant="round3_revised",
        seed=42,
        goal_offset_steps=25,
        entries=(
            CohortEntry(10, 7, 0, 35, 25, 1.0, False),
            CohortEntry(20, 8, 2, 47, 27, 2.0, False),
        ),
        episode_split={"dev": (7, 8), "final": (), "online": ()},
    )


def _legacy_manifest():
    return CohortManifest(
        task="cube",
        cohort_id="cube_custom_legacy_v1",
        cohort_kind="custom",
        protocol_variant="legacy",
        seed=42,
        goal_offset_steps=25,
        entries=(
            CohortEntry(10, 9, 0, None, None, None, None),
            CohortEntry(20, 9, 1, None, None, None, None),
        ),
        episode_split={"selected": (9, 9)},
    )


def _episodes():
    return [
        {"slot": 0, "episode_id": 7, "start_step": 0, "row_index": 10,
         "goal_row_index": 35, "goal_step": 25, "success": True,
         "initial_success": False, "start_distance": 1.0,
         "terminal_distance": 0.0, "first_success_step": 4,
         "hold_success": True, "rollout_failed": False,
         "missing_field_count": 0, "invalid_action_count": 0,
         "early_terminated": False},
        {"slot": 1, "episode_id": 8, "start_step": 2, "row_index": 20,
         "goal_row_index": 47, "goal_step": 27, "success": False,
         "initial_success": False, "start_distance": 2.0,
         "terminal_distance": 1.0, "first_success_step": None,
         "hold_success": False, "rollout_failed": False,
         "missing_field_count": 0, "invalid_action_count": 0,
         "early_terminated": False},
    ]


def _legacy_episodes():
    return [
        {
            "slot": 0,
            "episode_id": 9,
            "dataset_episode": 9,
            "start_step": 0,
            "row_index": 10,
            "success": True,
            "initial_success": False,
            "start_distance": None,
            "terminal_distance": None,
            "first_success_step": None,
            "hold_success": False,
            "rollout_failed": False,
            "missing_field_count": 0,
            "invalid_action_count": 0,
            "early_terminated": False,
        },
        {
            "slot": 1,
            "episode_id": 9,
            "dataset_episode": 9,
            "start_step": 1,
            "row_index": 20,
            "success": False,
            "initial_success": False,
            "start_distance": None,
            "terminal_distance": None,
            "first_success_step": None,
            "hold_success": False,
            "rollout_failed": False,
            "missing_field_count": 0,
            "invalid_action_count": 0,
            "early_terminated": False,
        },
    ]


def _payload(manifest=None, episodes=None):
    manifest = manifest or _manifest()
    episodes = _episodes() if episodes is None else episodes
    return {
        "schema_version": 2,
        "protocol": ROUND3_PROTOCOL,
        "protocol_variant": manifest.protocol_variant,
        "status": "ok",
        "task": manifest.task,
        "cohort_id": manifest.cohort_id,
        "cohort_kind": manifest.cohort_kind,
        "cohort_sha256": manifest.computed_sha256,
        "predicate_version": "round3_runtime_predicates_v1",
        "cohort": manifest.as_dict(),
        "episodes": episodes,
        "summary": summarize_episodes(episodes),
    }


class ResultValidationTests(unittest.TestCase):
    def test_incomplete_result_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_result_payload(
                {
                    "protocol": ROUND3_PROTOCOL,
                    "status": "ok",
                    "predicate_version": "round3_runtime_predicates_v1",
                    "cohort_sha256": "0" * 64,
                    "episodes": [{"success": True}],
                },
                expected_count=2,
            )

    def test_complete_result_is_accepted(self):
        manifest = _manifest()
        validate_cohort_manifest(manifest, expected_count=2)
        validate_result_payload(_payload(manifest), manifest=manifest, expected_count=2)

    def test_embedded_cohort_hash_mismatch_is_rejected(self):
        manifest = _manifest()
        payload = _payload(manifest)
        payload["cohort"]["cohort_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_result_payload(payload, manifest=manifest)

    def test_missing_episode_is_rejected(self):
        manifest = _manifest()
        payload = _payload(manifest)
        payload["episodes"][1]["episode_id"] = 999
        with self.assertRaises(ValueError):
            validate_result_payload(payload, manifest=manifest)

    def test_episode_order_is_rejected(self):
        manifest = _manifest()
        payload = _payload(manifest)
        payload["episodes"] = list(reversed(payload["episodes"]))
        with self.assertRaises(ValueError):
            validate_result_payload(payload, manifest=manifest)

    def test_duplicate_episode_is_rejected(self):
        manifest = _manifest()
        payload = _payload(manifest)
        payload["episodes"][1] = copy.deepcopy(payload["episodes"][0])
        with self.assertRaises(ValueError):
            validate_result_payload(payload, manifest=manifest)

    def test_forged_summary_is_rejected(self):
        manifest = _manifest()
        payload = _payload(manifest)
        payload["summary"]["successes"] += 1
        with self.assertRaises(ValueError):
            validate_result_payload(payload, manifest=manifest)

    def test_trace_missing_episode_is_rejected(self):
        manifest = _manifest()
        payload = _payload(manifest)
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "episodes.jsonl"
            trace_path.write_text(json.dumps(payload["episodes"][0]) + "\n", encoding="utf-8")
            payload["trace_path"] = str(trace_path)
            with self.assertRaises(ValueError):
                validate_result_payload(payload, manifest=manifest)

    def test_trace_different_content_is_rejected(self):
        manifest = _manifest()
        payload = _payload(manifest)
        altered = copy.deepcopy(payload["episodes"])
        altered[0]["success"] = False
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "episodes.jsonl"
            trace_path.write_text(
                "\n".join(json.dumps(item) for item in altered) + "\n",
                encoding="utf-8",
            )
            payload["trace_path"] = str(trace_path)
            with self.assertRaises(ValueError):
                validate_result_payload(payload, manifest=manifest)


class RegressionContractTests(unittest.TestCase):
    def test_legacy_repeated_raw_episode_is_valid(self):
        for kind in ("custom", "dev"):
            manifest = replace(_legacy_manifest(), cohort_kind=kind, cohort_sha256=None)
            payload = _payload(manifest, _legacy_episodes())
            validate_cohort_manifest(manifest, expected_count=2)
            validate_result_payload(payload, manifest=manifest, expected_count=2)

    def test_duplicate_complete_identity_is_rejected(self):
        episodes = _episodes()
        episodes[1] = copy.deepcopy(episodes[0])
        payload = _payload(episodes=episodes)
        payload.pop("cohort")
        payload["cohort_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_result_payload(payload, expected_count=2)

    def test_collector_shape_without_top_level_goal_fields_is_valid(self):
        entry = CohortEntry(
            row_index=10,
            episode_id=7,
            start_step=0,
            goal_row_index=35,
            goal_step=25,
            start_distance=1.0,
            initially_successful=False,
            start_state=(1.0, 0.0, 0.0),
            goal_state=(0.0, 0.0, 0.0),
        )
        manifest = CohortManifest(
            task="cube",
            cohort_id="cube_dev_collector_v1",
            cohort_kind="dev",
            protocol_variant="round3_revised",
            seed=42,
            goal_offset_steps=25,
            entries=(entry,),
            episode_split={"dev": (7,), "final": (), "online": ()},
        )
        collector = Round3TraceCollector("cube", manifest, action_block=1)
        collector.record_step(
            np.asarray([[0.0]]),
            {
                "privileged_block_0_pos": np.asarray([[0.0, 0.0, 0.0]]),
                "goal_privileged_block_0_pos": np.asarray([[0.0, 0.0, 0.0]]),
                "success": np.asarray([True]),
                "terminated": np.asarray([False]),
                "truncated": np.asarray([False]),
            },
            raw_env_step=1,
        )
        record = collector.finalize([True], eval_budget=50)[0]
        self.assertNotIn("goal_step", record)
        self.assertNotIn("goal_state", record)
        validate_result_payload(
            _payload(manifest, [record]), manifest=manifest, expected_count=1
        )

    def test_eval_publishes_enriched_payload_contract(self):
        cfg = compose_eval_config(
            "cube", ("eval.num_eval=2", "output.save_video=false")
        )
        manifest = _manifest()

        class FakeWorld:
            envs = None

            def set_policy(self, policy):
                self.policy = policy

            def close(self):
                pass

        class FakeSession:
            def __init__(self, cfg, *, task, dataset, cohort):
                self.dataset = dataset
                self.cohort = cohort
                self.world_factory = lambda **kwargs: FakeWorld()

            def _build_policy(self, policy_or_model, identity, device):
                return object()

        metrics = {
            "episode_successes": np.asarray([True, False]),
            "success_rate": 50.0,
        }
        identity = EvaluationIdentity("test", "random")
        with patch("source.common.eval.DatasetEvaluationSession", FakeSession):
            with patch(
                "source.common.eval.evaluate_from_dataset_compat",
                return_value=metrics,
            ):
                with tempfile.TemporaryDirectory() as directory:
                    result = run_round3_evaluation(
                        cfg,
                        task="cube",
                        policy_or_model=object(),
                        identity=identity,
                        manifest=manifest,
                        output_dir=Path(directory) / "result",
                        device="cpu",
                        trace=False,
                    )
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["cohort_kind"], "dev")
        self.assertEqual(result["summary"]["num_episodes"], 2)
        self.assertEqual(result["summary"]["successes"], 1)

    def test_eval_rejects_manifest_before_session_construction(self):
        cfg = compose_eval_config(
            "cube", ("eval.num_eval=2", "output.save_video=false")
        )
        invalid = replace(_manifest(), task="reacher", cohort_sha256=None)
        with patch("source.common.eval.DatasetEvaluationSession") as session_cls:
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    run_round3_evaluation(
                        cfg,
                        task="cube",
                        policy_or_model=object(),
                        identity=EvaluationIdentity("test", "random"),
                        manifest=invalid,
                        output_dir=Path(directory) / "result",
                        device="cpu",
                        trace=False,
                    )
        session_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
