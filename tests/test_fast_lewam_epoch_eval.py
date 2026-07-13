import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from source.common.epoch_eval import (
    FastLeWAMEpochEvalCallback,
    _load_eval_config,
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
            modes=("stage_a", "stage_b"),
            every_n_epochs=1,
            evaluator=lambda **kwargs: calls.append(kwargs["mode"]) or {},
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
        cfg = _load_eval_config("cube", 1, {}, seed=123)

        self.assertEqual(cfg.seed, 123)
        self.assertEqual(cfg.solver.seed, 123)

    def test_stage_a_failure_does_not_skip_stage_b(self):
        calls = []

        def evaluator(**kwargs):
            mode = kwargs["mode"]
            calls.append(mode)
            if mode == "stage_a":
                raise RuntimeError("stage a failed")
            return {}

        callback = FastLeWAMEpochEvalCallback(evaluator=evaluator)
        with tempfile.TemporaryDirectory() as root:
            trainer = FakeTrainer(root=root)
            callback.on_train_epoch_end(trainer, FakeModule())
            failure = (
                Path(root)
                / "epoch_eval"
                / "epoch_1"
                / "stage_a"
                / "failure.txt"
            )
            self.assertTrue(failure.exists())
        self.assertEqual(calls, ["stage_a", "stage_b"])
        self.assertEqual(len(trainer.strategy.barriers), 2)

if __name__ == "__main__":
    unittest.main()
