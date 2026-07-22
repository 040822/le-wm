"""Evaluate LeWM or a random policy with the canonical LeWM protocol."""

import os

os.environ["MUJOCO_GL"] = "egl"

import traceback
from pathlib import Path

import hydra
import stable_worldmodel as swm
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from source.common.checkpoint import get_policy_eval_paths, load_policy_or_model
from source.common.eval import (
    DatasetEvaluationSession,
    EvaluationIdentity,
    write_evaluation_failure,
)
from source.model.fast_lewam.jepa import FastLeWAM


def validate_generic_eval_policy(policy_or_model):
    """Keep the original eval entrypoint focused on non-Fast policies."""
    model = getattr(policy_or_model, "model", policy_or_model)
    if isinstance(model, FastLeWAM):
        raise ValueError(
            "eval.py does not evaluate Fast-LeWAM; use eval_fast_lewam.py "
            "and select one or more stages"
        )
    return policy_or_model


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    policy_name = cfg.get("policy", "random")
    checkpoint = None
    eval_path = Path(HydraConfig.get().runtime.output_dir) / "eval"
    try:
        if policy_name == "random":
            policy_or_model = swm.policy.RandomPolicy(seed=int(cfg.seed))
            policy_kind = "random"
        else:
            eval_path, _ = get_policy_eval_paths(policy_name)
            policy_or_model, checkpoint = load_policy_or_model(policy_name)
            validate_generic_eval_policy(policy_or_model)
            eval_path, _ = get_policy_eval_paths(
                policy_name, ckpt_path=checkpoint
            )
            policy_kind = "lewm"

        task = str(cfg.eval.task_name)
        session = DatasetEvaluationSession(cfg, task=task)
        identity = EvaluationIdentity(
            entrypoint="eval",
            policy_kind=policy_kind,
            checkpoint=str(checkpoint) if checkpoint is not None else None,
        )
        result = session.evaluate(
            policy_or_model,
            identity=identity,
            output_dir=eval_path,
        )
    except Exception as exc:
        write_evaluation_failure(eval_path, exc, traceback.format_exc())
        raise

    print(
        f"success_rate={result.success_rate:.2f}, "
        f"evaluation_seconds={result.evaluation_seconds:.3f}"
    )


if __name__ == "__main__":
    run()
