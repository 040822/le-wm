import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import source.common.round3_runtime_audit as runtime_audit
from source.common.eval import EvaluationIdentity, compose_eval_config
from source.common.round3_eval import run_round3_evaluation
from source.common.round3_phase1 import CohortEntry, CohortManifest
from source.common.round3_protocol import audit_all_predicates
from source.common.round3_runtime_audit import run_independent_cpu_neutral_hold


class RuntimeAuditTests(unittest.TestCase):
    def test_installed_runtime_sources_are_available_for_all_tasks(self):
        audits = audit_all_predicates()
        self.assertEqual(set(audits), {"cube", "pusht", "reacher", "tworoom"})
        for task, audit in audits.items():
            self.assertEqual(audit["status"], "accepted", task)
            self.assertEqual(audit["goal_refresh"]["checks_expected"], 10)

    def test_evaluation_requires_one_pure_cpu_hold_diagnostic_before_publish(self):
        manifest = CohortManifest(
            task="cube",
            cohort_id="cube_dev_runtime_hold_v1",
            cohort_kind="dev",
            protocol_variant="round3_revised",
            seed=42,
            goal_offset_steps=25,
            entries=(CohortEntry(10, 7, 0, 35, 25, 1.0, False),),
            episode_split={"dev": (7,), "final": (), "online": ()},
        )
        cfg = compose_eval_config("cube", ("eval.num_eval=1", "output.save_video=false"))
        identity = EvaluationIdentity("audit-test", "random")
        metrics = {"episode_successes": np.asarray([True]), "success_rate": 100.0}
        calls = []

        def pure_runner(*args, **kwargs):
            calls.append((args, kwargs))
            forbidden = {"policy", "policy_or_model", "model", "model_or_policy", "cem"}
            self.assertFalse(forbidden.intersection(kwargs))
            return {
                "status": "accepted",
                "task": "cube",
                "policy_identity": {
                    "entrypoint": identity.entrypoint,
                    "policy_kind": identity.policy_kind,
                    "checkpoint": identity.checkpoint,
                    "epoch": identity.epoch,
                    "stage": identity.stage,
                },
                "protocol": "round3_phase1_v1",
                "protocol_variant": manifest.protocol_variant,
                "stage": identity.stage,
                "cohort_id": manifest.cohort_id,
                "cohort_sha256": manifest.computed_sha256,
                "seed": manifest.seed,
                "independent": True,
                "independent_cpu_runtime": True,
                "cpu_only": True,
                "model_used": False,
                "cem_used": False,
                "neutral_action": [0.0] * 5,
                "action_legal": True,
                "action_source": "env_default",
                "action_reason": "deterministic neutral action from environment metadata",
                "action_bounds": {"low": [-1.0] * 5, "high": [1.0] * 5},
                "action_legality": {
                    "legal": True,
                    "source": "env_default",
                    "bounds_source": "action_space_bounds",
                    "bounds": {"low": [-1.0] * 5, "high": [1.0] * 5},
                    "reason": "deterministic neutral action from environment metadata",
                },
                "raw_step_count": 5,
                "env_steps": 5,
                "raw_steps": [
                    {
                        "raw_step": step,
                        "action": [0.0] * 5,
                        "success": True,
                        "predicate_success": True,
                        "env_success": True,
                        "terminated": False,
                        "truncated": False,
                        "termination_reason": "running",
                        "state": [0.0, 0.0, 0.0],
                        "goal": [0.0, 0.0, 0.0],
                        "distance_components": {"distance": 0.0},
                        "distance_unit": "m",
                        "distance_aggregation": "named task components; success uses frozen predicate",
                        "distance_source": "independent_cpu_runtime_state_goal",
                    }
                    for step in range(1, 6)
                ],
            }

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

        with patch("source.common.eval.DatasetEvaluationSession", FakeSession), patch(
            "source.common.eval.evaluate_from_dataset_compat", return_value=metrics
        ), patch("source.common.round3_eval.run_independent_cpu_neutral_hold", side_effect=pure_runner), patch(
            "source.common.round3_eval.write_round3_result", side_effect=lambda payload, *a, **k: dict(payload)
        ) as writer:
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
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], ("cube",))
        self.assertEqual(calls[0][1]["seed"], manifest.seed)
        self.assertEqual(calls[0][1]["protocol"], "round3_phase1_v1")
        self.assertEqual(calls[0][1]["protocol_variant"], manifest.protocol_variant)
        self.assertEqual(calls[0][1]["stage"], identity.stage)
        self.assertEqual(calls[0][1]["cohort_id"], manifest.cohort_id)
        self.assertEqual(calls[0][1]["cohort_sha256"], manifest.computed_sha256)
        self.assertEqual(
            calls[0][1]["policy_identity"],
            {
                "entrypoint": identity.entrypoint,
                "policy_kind": identity.policy_kind,
                "checkpoint": identity.checkpoint,
                "epoch": identity.epoch,
                "stage": identity.stage,
            },
        )
        diagnostic = result["neutral_hold_diagnostic"]
        self.assertTrue(diagnostic["independent"])
        self.assertTrue(diagnostic["independent_cpu_runtime"])
        self.assertFalse(diagnostic["model_used"])
        self.assertFalse(diagnostic["cem_used"])
        self.assertEqual([row["raw_step"] for row in diagnostic["raw_steps"]], [1, 2, 3, 4, 5])
        writer.assert_called_once()

        for bad in ({}, {"status": "unaccepted", "independent": True}):
            with patch("source.common.eval.DatasetEvaluationSession", FakeSession), patch(
                "source.common.eval.evaluate_from_dataset_compat", return_value=metrics
            ), patch("source.common.round3_eval.run_independent_cpu_neutral_hold", return_value=bad), patch(
                "source.common.round3_eval.write_round3_result"
            ) as rejected_writer:
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaises(ValueError):
                        run_round3_evaluation(
                            cfg,
                            task="cube",
                            policy_or_model=object(),
                            identity=identity,
                            manifest=manifest,
                            output_dir=Path(directory) / "rejected",
                            device="cpu",
                            trace=False,
                        )
            rejected_writer.assert_not_called()

    @staticmethod
    def _adapter_double(task, step_factory):
        action_size = {"cube": 5, "pusht": 2, "reacher": 2, "tworoom": 2}[task]
        state_size = {"cube": 3, "pusht": 7, "reacher": 2, "tworoom": 2}[task]
        calls = []
        action_space = SimpleNamespace(
            low=np.full(action_size, -1.0),
            high=np.full(action_size, 1.0),
            shape=(action_size,),
        )
        current_goal = np.zeros(state_size, dtype=np.float64)
        reset_info = {
            "neutral_action": [0.0] * action_size,
            "neutral_action_source": f"{task}_adapter",
            "neutral_action_reason": f"{task} task-specific neutral action",
            "action_bounds": {
                "low": [-1.0] * action_size,
                "high": [1.0] * action_size,
            },
        }

        class RawEnv:
            def __init__(self, space):
                self.action_space = space
                self.raw_calls = 0

            def step(self, action):
                self.raw_calls += 1
                raise AssertionError("raw env.step bypassed the task adapter")

            def close(self):
                pass

        env = RawEnv(action_space)

        def set_goal(value):
            current_goal[:] = np.asarray(value, dtype=np.float64)

        def adapter_step(action):
            calls.append(np.asarray(action, dtype=np.float64).copy())
            return step_factory(len(calls), action)

        adapter = {
            "env": env,
            "backend": f"test_{task}_adapter",
            "reset": lambda seed: (None, dict(reset_info)),
            "state": lambda observation, info: np.array(current_goal, copy=True),
            "goal": lambda observation, info: np.array(current_goal, copy=True),
            "set_goal": set_goal,
            "step": adapter_step,
            "action_space": action_space,
            "continuous": task == "reacher",
            "relative": None,
        }
        return adapter, env, calls

    @staticmethod
    def _hold_identity():
        return {
            "entrypoint": "adapter-test",
            "policy_kind": "independent_cpu_runtime",
            "checkpoint": None,
            "epoch": None,
            "stage": None,
        }

    def test_neutral_hold_uses_task_adapter_and_allows_success_termination(self):
        def step_factory(index, action):
            return (
                None,
                0.0,
                index == 1,
                False,
                {
                    "success": True,
                    "termination_reason": "success" if index == 1 else "running",
                },
            )

        adapter, env, calls = self._adapter_double("cube", step_factory)
        action_helper = runtime_audit._task_neutral_action
        parser = runtime_audit._parse_step_result
        with patch.object(
            runtime_audit, "_make_neutral_runtime_adapter", return_value=adapter
        ), patch.object(
            runtime_audit, "_task_neutral_action", wraps=action_helper
        ) as action_selector, patch.object(
            runtime_audit, "_parse_step_result", wraps=parser
        ) as step_parser:
            diagnostic = run_independent_cpu_neutral_hold(
                "cube",
                seed=42,
                protocol="round3_phase1_v1",
                protocol_variant="round3_revised",
                stage=None,
                cohort_id="cube-adapter-test",
                cohort_sha256="a" * 64,
                policy_identity=self._hold_identity(),
            )
        self.assertEqual(diagnostic["status"], "accepted")
        self.assertEqual(env.raw_calls, 0)
        self.assertEqual(len(calls), 5)
        self.assertEqual(action_selector.call_count, 1)
        self.assertEqual(step_parser.call_count, 5)
        self.assertEqual(diagnostic["action_source"], "cube_adapter")
        self.assertEqual(
            [row["raw_step"] for row in diagnostic["raw_steps"]], [1, 2, 3, 4, 5]
        )
        self.assertTrue(diagnostic["raw_steps"][0]["terminated"])
        self.assertEqual(diagnostic["raw_steps"][0]["termination_reason"], "success")

    def test_neutral_hold_truncation_fails_closed_after_adapter_step(self):
        def step_factory(index, action):
            truncated = index == 2
            return (
                None,
                0.0,
                False,
                truncated,
                {
                    "success": True,
                    "termination_reason": "time_limit" if truncated else "running",
                },
            )

        adapter, env, calls = self._adapter_double("cube", step_factory)
        with patch.object(
            runtime_audit, "_make_neutral_runtime_adapter", return_value=adapter
        ):
            diagnostic = run_independent_cpu_neutral_hold(
                "cube",
                cohort_id="cube-truncation-test",
                cohort_sha256="b" * 64,
                policy_identity=self._hold_identity(),
            )
        self.assertEqual(diagnostic["status"], "unaccepted")
        self.assertEqual(len(calls), 2)
        self.assertEqual(env.raw_calls, 0)

    def test_reacher_ignores_nan_and_non_bool_success_sentinels(self):
        sentinels = iter(
            (
                np.asarray([np.nan]),
                np.asarray([0.5]),
                np.asarray([True]),
                np.asarray([np.nan]),
                np.asarray([1.0]),
            )
        )

        def step_factory(index, action):
            return (
                None,
                0.0,
                False,
                False,
                {"success": next(sentinels), "termination_reason": "running"},
            )

        adapter, env, calls = self._adapter_double("reacher", step_factory)
        with patch.object(
            runtime_audit, "_make_neutral_runtime_adapter", return_value=adapter
        ):
            diagnostic = run_independent_cpu_neutral_hold(
                "reacher",
                cohort_id="reacher-sentinel-test",
                cohort_sha256="c" * 64,
                policy_identity=self._hold_identity(),
            )
        self.assertEqual(diagnostic["status"], "accepted")
        self.assertEqual(len(calls), 5)
        self.assertEqual(env.raw_calls, 0)
        self.assertTrue(all(row["env_success"] is None for row in diagnostic["raw_steps"]))
        self.assertTrue(
            all(row["termination_reason"] == "running" for row in diagnostic["raw_steps"])
        )

    def test_neutral_hold_unknown_termination_reason_fails_closed(self):
        def step_factory(index, action):
            return (
                None,
                0.0,
                False,
                False,
                {"success": True, "termination_reason": "unknown"},
            )

        adapter, env, calls = self._adapter_double("pusht", step_factory)
        with patch.object(
            runtime_audit, "_make_neutral_runtime_adapter", return_value=adapter
        ):
            diagnostic = run_independent_cpu_neutral_hold(
                "pusht",
                cohort_id="pusht-termination-test",
                cohort_sha256="d" * 64,
                policy_identity=self._hold_identity(),
            )
        self.assertEqual(diagnostic["status"], "unaccepted")
        self.assertEqual(len(calls), 1)
        self.assertEqual(env.raw_calls, 0)

if __name__ == "__main__":
    unittest.main()
