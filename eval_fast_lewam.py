"""Batch-evaluate Fast-LeWAM epoch weights with the canonical LeWM protocol.

Examples:
  CUDA_VISIBLE_DEVICES=1 python eval_fast_lewam.py outputs/fast_lewam/cube/0715_goal/checkpoints --epochs 10

"""

import argparse
import re
import traceback
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from source.common.checkpoint import torch_load_compat
from source.common.epoch_eval import FAST_LEWAM_STAGES
from source.common.eval import (
    DatasetEvaluationSession,
    EvaluationIdentity,
    compose_eval_config,
    write_evaluation_failure,
)


_WEIGHT_PATTERN = re.compile(r".+_weights_epoch_(\d+)\.pt$")
_DATASET_TASKS = {
    "pusht": "pusht",
    "pusht_expert_train": "pusht",
    "ogbench/cube_single": "cube",
    "ogbench/cube_single_expert": "cube",
    "dmcontrol/reacher": "reacher",
    "dmc/reacher_random": "reacher",
    "tworoom": "tworoom",
}

FAST_LEWAM_OFFLINE_STAGES = (
    *FAST_LEWAM_STAGES,
    "stage_b_actor_warm_start",
)


@dataclass(frozen=True)
class WeightCheckpoint:
    epoch: int
    path: Path


def discover_weight_checkpoints(checkpoint_dir, epochs=None):
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"checkpoint directory does not exist: {checkpoint_dir}")
    found = {}
    for path in checkpoint_dir.glob("*_weights_epoch_*.pt"):
        match = _WEIGHT_PATTERN.fullmatch(path.name)
        if match is not None:
            found[int(match.group(1))] = path
    if not found:
        raise FileNotFoundError(
            f"no '*_weights_epoch_N.pt' files found in {checkpoint_dir}"
        )
    requested = sorted(set(int(epoch) for epoch in epochs)) if epochs else sorted(found)
    missing = [epoch for epoch in requested if epoch not in found]
    if missing:
        raise FileNotFoundError(
            f"requested epoch(s) {missing} not found in {checkpoint_dir}; "
            f"available epochs: {sorted(found)}"
        )
    return [WeightCheckpoint(epoch=epoch, path=found[epoch]) for epoch in requested]


def load_model_from_weights(run_cfg, checkpoint_path, device="cuda"):
    model = hydra.utils.instantiate(run_cfg.policy.model)
    state_dict = torch_load_compat(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"expected a state_dict mapping in {checkpoint_path}, "
            f"got {type(state_dict).__name__}"
        )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval().requires_grad_(False)


def resolve_eval_config_name(run_dir, run_cfg, explicit=None):
    """Resolve task from an explicit name, saved Hydra choice, or dataset alias."""
    if explicit:
        return str(explicit)
    hydra_path = Path(run_dir) / ".hydra" / "hydra.yaml"
    if hydra_path.is_file():
        hydra_cfg = OmegaConf.load(hydra_path)
        choice = OmegaConf.select(hydra_cfg, "hydra.runtime.choices.data")
        if choice:
            return str(choice)
    saved = OmegaConf.select(run_cfg, "epoch_eval.config_name")
    if saved:
        return str(saved)
    dataset_name = str(run_cfg.data.dataset.name)
    if dataset_name.endswith(".h5"):
        dataset_name = dataset_name[:-3]
    if dataset_name in _DATASET_TASKS:
        return _DATASET_TASKS[dataset_name]
    raise ValueError(
        f"cannot infer eval task from training dataset {dataset_name!r}; "
        "pass --config-name"
    )


def run_evaluation(args, eval_overrides=()):
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    run_dir = checkpoint_dir.parent
    output_dir = run_dir / "eval"
    try:
        checkpoints = discover_weight_checkpoints(
            checkpoint_dir, args.epochs
        )
        config_path = (
            Path(args.run_config).expanduser().resolve()
            if args.run_config
            else run_dir / "config.yaml"
        )
        if not config_path.is_file():
            raise FileNotFoundError(
                f"training run config not found: {config_path}"
            )
        run_cfg = OmegaConf.load(config_path)
        config_name = resolve_eval_config_name(
            run_dir, run_cfg, args.config_name
        )
        cfg = compose_eval_config(config_name, eval_overrides)
        session = DatasetEvaluationSession(cfg, task=config_name)
    except Exception as exc:
        write_evaluation_failure(
            output_dir / "setup", exc, traceback.format_exc()
        )
        raise
    device = str(cfg.solver.get("device", "cuda"))
    failures = []
    model = None

    for checkpoint in checkpoints:
        if model is not None:
            del model
            model = None
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        print(f"\n=== epoch {checkpoint.epoch}: {checkpoint.path.name} ===")
        try:
            model = load_model_from_weights(run_cfg, checkpoint.path, device)
        except Exception as exc:
            model = None
            error = traceback.format_exc()
            failure_dir = output_dir / f"epoch_{checkpoint.epoch}" / "load"
            write_evaluation_failure(failure_dir, exc, error)
            failures.append((checkpoint.epoch, "load", str(exc)))
            print(error)
            continue

        for stage in args.stages:
            leaf = output_dir / f"epoch_{checkpoint.epoch}" / stage
            identity = EvaluationIdentity(
                entrypoint="eval_fast_lewam",
                policy_kind="fast_lewam",
                checkpoint=str(checkpoint.path),
                epoch=checkpoint.epoch,
                stage=stage,
            )
            try:
                result = session.evaluate(
                    model,
                    identity=identity,
                    output_dir=leaf,
                    device=device,
                )
                print(
                    f"{stage}: success_rate={result.success_rate:.2f}, "
                    f"evaluation_seconds={result.evaluation_seconds:.3f}"
                )
            except Exception as exc:
                error = traceback.format_exc()
                write_evaluation_failure(leaf, exc, error)
                failures.append((checkpoint.epoch, stage, str(exc)))
                print(error)

    if model is not None:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if failures:
        details = "; ".join(
            f"epoch {epoch} {stage}: {error}"
            for epoch, stage, error in failures
        )
        raise RuntimeError(f"one or more evaluations failed: {details}")
    return output_dir


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", help="directory containing epoch weights")
    parser.add_argument("--epochs", nargs="+", type=int, help="selected epochs")
    parser.add_argument("--config-name", help="eval task; inferred from run by default")
    parser.add_argument("--run-config", help="training config; defaults to <run>/config.yaml")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=FAST_LEWAM_OFFLINE_STAGES,
        default=FAST_LEWAM_STAGES,
    )
    return parser


def parse_cli(argv=None):
    raw = list(argv) if argv is not None else None
    if raw is None:
        import sys

        raw = sys.argv[1:]
    overrides = tuple(
        token for token in raw if "=" in token and not token.startswith("--")
    )
    batch_args = [token for token in raw if token not in overrides and token != "--"]
    args = build_parser().parse_args(batch_args)
    return args, overrides


def main():
    args, overrides = parse_cli()
    output_dir = run_evaluation(args, overrides)
    print(f"\nEvaluation results: {output_dir}")


if __name__ == "__main__":
    main()
