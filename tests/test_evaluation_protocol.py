import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
from omegaconf import OmegaConf
from stable_worldmodel.protocols import Actionable

from source.common.eval import (
    EpisodeEvaluation,
    EvaluationIdentity,
    EvaluationResult,
    compose_eval_config,
    DatasetEvaluationSession,
    select_eval_cohort,
    write_evaluation_artifacts,
)
from tests.test_fast_lewam_model import make_model


class FakeDataset:
    column_names = ["episode_idx", "step_idx"]

    def __init__(self):
        self.columns = {
            "episode_idx": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
            "step_idx": np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]),
        }

    def get_col_data(self, name):
        return self.columns[name]

    def get_row_data(self, indices):
        return {name: values[indices] for name, values in self.columns.items()}


class EvaluationProtocolTests(unittest.TestCase):
    def test_shared_config_rejects_plan_longer_than_eval_budget(self):
        with self.assertRaisesRegex(ValueError, "eval_budget"):
            compose_eval_config(
                "cube",
                overrides=("plan_config.horizon=11",),
            )

    def test_shared_config_uses_task_yaml_and_hydra_style_overrides(self):
        cfg = compose_eval_config(
            "cube",
            overrides=("eval.num_eval=3", "seed=7", "output.save_video=false"),
        )

        self.assertEqual(cfg.world.env_name, "swm/OGBCube-v0")
        self.assertEqual(cfg.world.num_envs, 3)
        self.assertEqual(cfg.solver.seed, 7)
        self.assertFalse(cfg.output.save_video)

    def test_cohort_reproduces_upstream_lewm_global_last_row_exclusion(self):
        dataset = FakeDataset()
        valid_indices = np.array([0, 1, 2, 5, 6, 7, 8])
        expected_positions = np.random.default_rng(42).choice(
            len(valid_indices) - 1,
            size=3,
            replace=False,
        )
        expected_rows = np.sort(valid_indices[expected_positions])

        cohort = select_eval_cohort(
            dataset,
            goal_offset_steps=2,
            num_eval=3,
            seed=42,
        )

        np.testing.assert_array_equal(cohort.row_indices, expected_rows)
        np.testing.assert_array_equal(
            cohort.episode_ids,
            dataset.columns["episode_idx"][expected_rows],
        )
        np.testing.assert_array_equal(
            cohort.start_steps,
            dataset.columns["step_idx"][expected_rows],
        )
        self.assertNotIn(8, cohort.row_indices)

    def test_result_artifacts_share_one_structured_source(self):
        result = EvaluationResult(
            task="cube",
            local_dataset="ogbench/cube_single",
            benchmark_dataset="ogbench/cube_single_expert",
            entrypoint="eval_fast_lewam",
            policy_kind="fast_lewam",
            checkpoint="weights_epoch_10.pt",
            epoch=10,
            stage="stage_a",
            parameters={"seed": 42, "num_eval": 2},
            evaluation_seconds=12.5,
            success_rate=50.0,
            episodes=(
                EpisodeEvaluation(
                    slot=0,
                    dataset_episode=3,
                    start_step=7,
                    success=True,
                    video="videos/env_0.mp4",
                ),
                EpisodeEvaluation(
                    slot=1,
                    dataset_episode=4,
                    start_step=9,
                    success=False,
                    video="videos/env_1.mp4",
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as root:
            write_evaluation_artifacts(result, root)
            payload = json.loads((Path(root) / "result.json").read_text())
            text = (Path(root) / "metrics.txt").read_text()

        self.assertEqual(payload["protocol"], "lewm_upstream_v1")
        self.assertEqual(payload["success_rate"], 50.0)
        self.assertEqual(payload["episodes"][0]["dataset_episode"], 3)
        self.assertIn("evaluation_seconds: 12.500000", text)
        self.assertIn("slot=1 episode=4 start=9 success=false", text)

    def test_session_runs_dataset_evaluation_and_writes_episode_provenance(self):
        cfg = OmegaConf.create(
            {
                "cache_dir": None,
                "seed": 42,
                "world": {"env_name": "fake", "num_envs": 2},
                "dataset": {"keys_to_cache": []},
                "plan_config": {
                    "horizon": 1,
                    "receding_horizon": 1,
                    "action_block": 1,
                },
                "solver": {"device": "cpu", "seed": 42},
                "eval": {
                    "num_eval": 2,
                    "goal_offset_steps": 2,
                    "eval_budget": 4,
                    "img_size": 8,
                    "dataset_name": "local_cube",
                    "benchmark_dataset_name": "official_cube",
                    "callables": [],
                },
                "output": {"save_video": False},
            }
        )
        calls = []
        closed = []

        class FakeWorld:
            def __init__(self, **kwargs):
                pass

            def set_policy(self, policy):
                self.policy = policy

            def evaluate(self, **kwargs):
                calls.append(kwargs)
                return {
                    "success_rate": 50.0,
                    "episode_successes": np.array([True, False]),
                    "seeds": None,
                }

            def close(self):
                closed.append(True)

        session = DatasetEvaluationSession(
            cfg,
            task="cube",
            dataset=FakeDataset(),
            world_factory=FakeWorld,
        )
        policy = swm.policy.RandomPolicy(seed=42)
        identity = EvaluationIdentity(
            entrypoint="eval",
            policy_kind="random",
        )

        with tempfile.TemporaryDirectory() as root:
            result = session.evaluate(policy, identity=identity, output_dir=root)
            payload = json.loads((Path(root) / "result.json").read_text())

        self.assertEqual(calls[0]["start_steps"], result.parameters["start_steps"])
        self.assertEqual(closed, [True])
        self.assertEqual([episode.success for episode in result.episodes], [True, False])
        self.assertEqual(payload["local_dataset"], "local_cube")
        self.assertEqual(payload["benchmark_dataset"], "official_cube")

    def test_failed_rerun_preserves_previous_success_snapshot(self):
        cfg = OmegaConf.create(
            {
                "cache_dir": None,
                "seed": 42,
                "world": {"env_name": "fake", "num_envs": 2},
                "dataset": {"keys_to_cache": []},
                "plan_config": {
                    "horizon": 1,
                    "receding_horizon": 1,
                    "action_block": 1,
                },
                "solver": {"device": "cpu", "seed": 42},
                "eval": {
                    "num_eval": 2,
                    "goal_offset_steps": 2,
                    "eval_budget": 4,
                    "img_size": 8,
                    "dataset_name": "local_cube",
                    "benchmark_dataset_name": "official_cube",
                    "callables": [],
                },
                "output": {"save_video": False},
            }
        )

        class FailingWorld:
            def __init__(self, **kwargs):
                self.env = type("Env", (), {"close": lambda self: None})()

            def set_policy(self, policy):
                self.policy = policy

            def evaluate(self, **kwargs):
                raise RuntimeError("rollout failed")

        session = DatasetEvaluationSession(
            cfg,
            task="cube",
            dataset=FakeDataset(),
            world_factory=FailingWorld,
        )
        identity = EvaluationIdentity(entrypoint="eval", policy_kind="random")

        with tempfile.TemporaryDirectory() as root:
            result_path = Path(root) / "result.json"
            result_path.write_text('{"status": "old-success"}\n')
            with self.assertRaisesRegex(RuntimeError, "rollout failed"):
                session.evaluate(
                    swm.policy.RandomPolicy(seed=42),
                    identity=identity,
                    output_dir=root,
                )
            payload = json.loads(result_path.read_text())

        self.assertEqual(payload["status"], "old-success")

    def test_actor_warm_start_stage_records_distinct_protocol_metadata(self):
        cfg = OmegaConf.create(
            {
                "cache_dir": None,
                "seed": 42,
                "world": {"env_name": "fake", "num_envs": 2},
                "dataset": {"keys_to_cache": []},
                "plan_config": {
                    "horizon": 1,
                    "receding_horizon": 1,
                    "action_block": 1,
                },
                "solver": {
                    "_target_": "stable_worldmodel.solver.CEMSolver",
                    "model": "???",
                    "batch_size": 1,
                    "num_samples": 2,
                    "n_steps": 1,
                    "topk": 1,
                    "device": "cpu",
                    "seed": 42,
                },
                "fast_lewam": {"inference_steps": 2},
                "eval": {
                    "num_eval": 2,
                    "goal_offset_steps": 2,
                    "eval_budget": 4,
                    "img_size": 8,
                    "dataset_name": "local_cube",
                    "benchmark_dataset_name": "official_cube",
                    "callables": [],
                },
                "output": {"save_video": False},
            }
        )
        policies = []

        class FakeWorld:
            def __init__(self, **kwargs):
                pass

            def set_policy(self, policy):
                policies.append(policy)

            def evaluate(self, **kwargs):
                return {
                    "success_rate": 50.0,
                    "episode_successes": np.array([True, False]),
                    "seeds": None,
                }

            def close(self):
                pass

        session = DatasetEvaluationSession(
            cfg,
            task="cube",
            dataset=FakeDataset(),
            world_factory=FakeWorld,
        )
        identity = EvaluationIdentity(
            entrypoint="eval_fast_lewam",
            policy_kind="fast_lewam",
            checkpoint="weights_epoch_10.pt",
            epoch=10,
            stage="stage_b_actor_warm_start",
        )

        with tempfile.TemporaryDirectory() as root:
            result = session.evaluate(
                make_model(
                    action_horizon=1,
                    stage_a_goal_injection="token",
                ),
                identity=identity,
                output_dir=root,
                device="cpu",
            )

        self.assertIsInstance(policies[0].solver.model, Actionable)
        self.assertTrue(result.parameters["actor_warm_start"])
        self.assertEqual(result.parameters["actor_seed"], 42)
        self.assertEqual(result.parameters["inference_steps"], 2)


if __name__ == "__main__":
    unittest.main()
