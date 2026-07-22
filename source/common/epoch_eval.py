"""Fast-LeWAM training-time adapter for the shared evaluation protocol."""

import os
import random
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

from source.common.eval import (
    DatasetEvaluationSession,
    EvaluationIdentity,
    EvaluationResult,
    compose_eval_config,
    write_evaluation_failure,
)


FAST_LEWAM_STAGES = (
    "stage_a",
    "stage_a_shuffled_goal",
    "stage_b",
    "stage_c",
)


class FastLeWAMEpochEvaluator:
    """Evaluate in-memory Fast-LeWAM models through one cached dataset session."""

    def __init__(self, config_name="cube", eval_overrides=()):
        self.config_name = config_name
        self.cfg = compose_eval_config(config_name, eval_overrides)
        self._session = None

    def _get_session(self):
        if self._session is None:
            self._session = DatasetEvaluationSession(
                self.cfg,
                task=self.config_name,
            )
        return self._session

    def __call__(self, *, model, stage, epoch, output_dir, checkpoint=None):
        leaf = Path(output_dir) / f"epoch_{epoch}" / stage
        identity = EvaluationIdentity(
            entrypoint="fast_lewam_epoch_eval",
            policy_kind="fast_lewam",
            checkpoint=str(checkpoint) if checkpoint is not None else None,
            epoch=int(epoch),
            stage=stage,
        )
        return self._get_session().evaluate(
            model,
            identity=identity,
            output_dir=leaf,
        )


class FastLeWAMEpochEvalCallback(Callback):
    """Run shared Fast-LeWAM evaluation on rank zero at selected epochs."""

    def __init__(
        self,
        config_name="cube",
        stages=FAST_LEWAM_STAGES,
        every_n_epochs=1,
        eval_overrides=(),
        evaluator=None,
    ):
        super().__init__()
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be positive")
        if not stages or any(stage not in FAST_LEWAM_STAGES for stage in stages):
            raise ValueError("stages contain an unsupported Fast-LeWAM eval stage")
        self.stages = tuple(stages)
        self.every_n_epochs = every_n_epochs
        self.evaluator = evaluator or FastLeWAMEpochEvaluator(
            config_name=config_name,
            eval_overrides=eval_overrides,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs:
            return
        stages = self.stages
        if getattr(pl_module, "train_mode", None) == "stage_b":
            stages = tuple(stage for stage in stages if stage == "stage_b")
        if not stages:
            return
        trainer.strategy.barrier("fast-lewam-epoch-eval-start")
        if trainer.is_global_zero:
            model = pl_module.model
            output_dir = Path(trainer.default_root_dir) / "eval"
            was_training = model.training
            requires_grad = [parameter.requires_grad for parameter in model.parameters()]
            python_rng_state = random.getstate()
            numpy_rng_state = np.random.get_state()
            torch_rng_state = torch.get_rng_state()
            model_device = next(model.parameters()).device
            cuda_rng_state = (
                torch.cuda.get_rng_state(model_device)
                if model_device.type == "cuda"
                else None
            )
            try:
                for stage in stages:
                    try:
                        result = self.evaluator(
                            model=model,
                            stage=stage,
                            epoch=epoch,
                            output_dir=output_dir,
                        )
                        self._log_metrics(trainer, stage, result)
                    except Exception as exc:
                        failure_dir = output_dir / f"epoch_{epoch}" / stage
                        error = traceback.format_exc()
                        write_evaluation_failure(failure_dir, exc, error)
                        print(f"Fast-LeWAM {stage} epoch eval failed; continuing.")
                        traceback.print_exc()
                        self._log_metrics(trainer, stage, {"failed": 1.0})
            finally:
                random.setstate(python_rng_state)
                np.random.set_state(numpy_rng_state)
                torch.set_rng_state(torch_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state(cuda_rng_state, model_device)
                model.train(was_training)
                for parameter, flag in zip(model.parameters(), requires_grad):
                    parameter.requires_grad_(flag)
        trainer.strategy.barrier("fast-lewam-epoch-eval-end")

    @staticmethod
    def _log_metrics(trainer, stage, result):
        if trainer.logger is None:
            return
        if isinstance(result, EvaluationResult):
            metrics = {
                "success_rate": result.success_rate,
                "evaluation_seconds": result.evaluation_seconds,
            }
        elif isinstance(result, dict):
            metrics = result
        else:
            return
        scalar = {}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                if value.numel() != 1 or torch.is_complex(value):
                    continue
                value = value.detach().cpu().item()
            else:
                array = np.asarray(value)
                if array.size != 1 or array.dtype.kind not in "buif":
                    continue
                value = array.item()
            scalar[f"epoch_eval/{stage}/{key}"] = float(value)
        if scalar:
            trainer.logger.log_metrics(scalar, step=trainer.global_step)


__all__ = [
    "FAST_LEWAM_STAGES",
    "FastLeWAMEpochEvalCallback",
    "FastLeWAMEpochEvaluator",
]
