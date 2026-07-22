import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from source.common.epoch_eval import (
    FastLeWAMEpochEvalCallback,
    FastLeWAMEpochEvaluator,
)


class FakeStrategy:
    def __init__(self):
        self.barriers = []

    def barrier(self, name=None):
        self.barriers.append(name)


class FakeTrainer:
    def __init__(self, epoch=0, global_zero=True, root=None):
        self.current_epoch = epoch
        self.is_global_zero = global_zero
        self.strategy = FakeStrategy()
        self.default_root_dir = root or "/tmp/fast-lewam-epoch-eval-test"
        self.global_step = 10
        self.logger = None


class FakeModule:
    def __init__(self):
        self.model = nn.Linear(2, 2)


class FastLeWAMEpochEvalCallbackTests(unittest.TestCase):
    def test_due_epoch_evaluates_both_modes_only_on_global_zero(self):
        calls = []
        callback = FastLeWAMEpochEvalCallback(
            config_name="cube",
            stages=("stage_a", "stage_b"),
            every_n_epochs=1,
            evaluator=lambda **kwargs: calls.append(kwargs["stage"]) or {},
        )
        trainer = FakeTrainer(epoch=0, global_zero=True)

        callback.on_train_epoch_end(trainer, FakeModule())

        self.assertEqual(calls, ["stage_a", "stage_b"])
        self.assertEqual(len(trainer.strategy.barriers), 2)

        calls.clear()
        worker = FakeTrainer(epoch=0, global_zero=False)
        callback.on_train_epoch_end(worker, FakeModule())
        self.assertEqual(calls, [])
        self.assertEqual(len(worker.strategy.barriers), 2)

    def test_callback_restores_training_and_gradient_state(self):
        module = FakeModule()
        module.model.train()
        original = [parameter.requires_grad for parameter in module.model.parameters()]
        rng_state = torch.get_rng_state().clone()

        def evaluator(**kwargs):
            kwargs["model"].eval().requires_grad_(False)
            torch.rand(4)
            return {}

        callback = FastLeWAMEpochEvalCallback(evaluator=evaluator)
        callback.on_train_epoch_end(FakeTrainer(), module)

        self.assertTrue(module.model.training)
        self.assertEqual(
            [parameter.requires_grad for parameter in module.model.parameters()],
            original,
        )
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_state))

    def test_skipped_epoch_does_not_enter_barrier_or_evaluate(self):
        calls = []
        callback = FastLeWAMEpochEvalCallback(
            config_name="cube",
            every_n_epochs=2,
            evaluator=lambda **kwargs: calls.append(kwargs) or {},
        )
        trainer = FakeTrainer(epoch=0)

        callback.on_train_epoch_end(trainer, FakeModule())
        self.assertEqual(calls, [])
        self.assertEqual(trainer.strategy.barriers, [])

    def test_eval_seed_is_resolved_into_cem_solver(self):
        evaluator = FastLeWAMEpochEvaluator(
            config_name="cube",
            eval_overrides=("eval.num_eval=1", "seed=123"),
        )
        self.assertEqual(evaluator.cfg.seed, 123)
        self.assertEqual(evaluator.cfg.solver.seed, 123)
        self.assertEqual(evaluator.cfg.eval.num_eval, 1)

    def test_stage_a_failure_does_not_skip_stage_b(self):
        calls = []

        def evaluator(**kwargs):
            stage = kwargs["stage"]
            calls.append(stage)
            if stage == "stage_a":
                raise RuntimeError("stage a failed")
            return {}

        callback = FastLeWAMEpochEvalCallback(
            stages=("stage_a", "stage_b"), evaluator=evaluator
        )
        with tempfile.TemporaryDirectory() as root:
            trainer = FakeTrainer(root=root)
            callback.on_train_epoch_end(trainer, FakeModule())
            failure = (
                Path(root)
                / "eval"
                / "epoch_1"
                / "stage_a"
                / "failure.txt"
            )
            self.assertTrue(failure.exists())
        self.assertEqual(calls, ["stage_a", "stage_b"])
        self.assertEqual(len(trainer.strategy.barriers), 2)

    def test_default_stages_include_stage_c_reference(self):
        calls = []
        callback = FastLeWAMEpochEvalCallback(
            evaluator=lambda **kwargs: calls.append(kwargs["stage"]) or {},
        )
        callback.on_train_epoch_end(FakeTrainer(), FakeModule())
        self.assertEqual(
            calls,
            ["stage_a", "stage_a_shuffled_goal", "stage_b", "stage_c"],
        )

    def test_stage_b_only_policy_evaluates_only_stage_b(self):
        calls = []
        callback = FastLeWAMEpochEvalCallback(
            evaluator=lambda **kwargs: calls.append(kwargs["stage"]) or {},
        )
        module = FakeModule()
        module.train_mode = "stage_b"

        callback.on_train_epoch_end(FakeTrainer(), module)

        self.assertEqual(calls, ["stage_b"])

    def test_metric_logging_skips_none_metadata(self):
        logged = []
        logger_type = type(
            "Logger", (), {"log_metrics": lambda self, metrics, step: logged.append(metrics)}
        )
        trainer = FakeTrainer()
        trainer.logger = logger_type()
        metrics = {
            "success_rate": 80.0,
            "episode_successes": np.array(
                [True, True, False, True, False, True, True, True, True, True]
            ),
            "seeds": None,
        }

        FastLeWAMEpochEvalCallback._log_metrics(trainer, "stage_a", metrics)

        self.assertEqual(logged, [{"epoch_eval/stage_a/success_rate": 80.0}])

if __name__ == "__main__":
    unittest.main()
