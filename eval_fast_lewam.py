"""
批量加载 Fast-LeWAM epoch 权重并评测 Stage A 与 Stage B。

  CUDA_VISIBLE_DEVICES=1 python3 eval_fast_lewam.py outputs/fast_lewam/cube/20260713_194403_043616150/checkpoints --epochs 4

"""

import argparse
import json
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from source.common.checkpoint import torch_load_compat


_WEIGHT_PATTERN = re.compile(r".+_weights_epoch_(\d+)\.pt$")


@dataclass(frozen=True)
class WeightCheckpoint:
    """一个带有训练 epoch 编号的纯模型 state_dict checkpoint。"""

    epoch: int
    path: Path


def discover_weight_checkpoints(checkpoint_dir, epochs=None):
    """发现并按数字 epoch 排序权重；指定 epoch 缺失时给出明确错误。"""
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
    """从训练 run 配置重建模型，并严格载入指定 epoch 的纯 state_dict。"""
    model = hydra.utils.instantiate(run_cfg.policy.model)
    state_dict = torch_load_compat(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"expected a state_dict mapping in {checkpoint_path}, "
            f"got {type(state_dict).__name__}"
        )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval().requires_grad_(False)


def resolve_eval_dataset_name(run_cfg, override=None):
    """优先使用 CLI 覆盖，否则从训练 run 配置推导可直接使用的数据集名。"""
    if override:
        return override
    try:
        name = str(run_cfg.data.dataset.name)
    except Exception as exc:
        raise ValueError("run config has no data.dataset.name; use --dataset-name") from exc
    return name[:-3] if name.endswith(".h5") else name


def _jsonable(value):
    """把 tensor、NumPy 标量和数组转换为 JSON 可序列化对象。"""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_summary(path, summary):
    """每完成一个 mode 就刷新汇总文件，中断时仍保留已有结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def run_evaluation(args):
    """依次加载选定权重并执行 Stage A/B 环境评测与视频录制。"""
    checkpoints = discover_weight_checkpoints(args.checkpoint_dir, args.epochs)
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    run_dir = checkpoint_dir.parent
    config_path = (
        Path(args.run_config).expanduser().resolve()
        if args.run_config
        else run_dir / "config.yaml"
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"training run config not found: {config_path}")

    run_cfg = OmegaConf.load(config_path)
    dataset_name = resolve_eval_dataset_name(run_cfg, args.dataset_name)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else run_dir / "eval_fast_lewam"
    )
    solver_overrides = {
        key: value
        for key, value in {
            "num_samples": args.num_samples,
            "n_steps": args.cem_steps,
            "topk": args.topk,
        }.items()
        if value is not None
    }

    # 延迟导入，避免仅查看 --help 时初始化 MuJoCo/World。
    from source.common.epoch_eval import FastLeWAMEpochEvaluator

    evaluator = FastLeWAMEpochEvaluator(
        config_name=args.config_name,
        num_eval=args.num_eval,
        solver_overrides=solver_overrides,
        seed=args.seed,
        save_video=not args.no_video,
        dataset_name=dataset_name,
    )
    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "run_config": str(config_path),
        "output_dir": str(output_dir),
        "config_name": args.config_name,
        "dataset_name": evaluator.cfg.eval.dataset_name,
        "num_eval": args.num_eval,
        "modes": list(args.modes),
        "checkpoints": {},
    }
    summary_path = output_dir / "summary.json"
    failures = []
    model = None

    for checkpoint in checkpoints:
        if model is not None:
            del model
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
        epoch_result = {"path": str(checkpoint.path), "modes": {}}
        summary["checkpoints"][str(checkpoint.epoch)] = epoch_result
        print(f"\n=== epoch {checkpoint.epoch}: {checkpoint.path.name} ===")
        try:
            model = load_model_from_weights(run_cfg, checkpoint.path, args.device)
        except Exception as exc:
            error = traceback.format_exc()
            failure_dir = output_dir / f"epoch_{checkpoint.epoch}"
            failure_dir.mkdir(parents=True, exist_ok=True)
            with (failure_dir / "load_failure.txt").open("w", encoding="utf-8") as file:
                file.write(error)
            epoch_result["load"] = {
                "status": "failed",
                "error": str(exc),
            }
            failures.append((checkpoint.epoch, "load", str(exc)))
            model = None
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
            _write_summary(summary_path, summary)
            print(error)
            continue

        for mode in args.modes:
            start = time.perf_counter()
            try:
                metrics = evaluator(
                    model=model,
                    mode=mode,
                    epoch=checkpoint.epoch,
                    output_dir=output_dir,
                )
                elapsed = time.perf_counter() - start
                epoch_result["modes"][mode] = {
                    "status": "ok",
                    "elapsed_seconds": elapsed,
                    "metrics": _jsonable(metrics),
                }
                print(f"{mode}: {metrics} ({elapsed:.1f}s)")
            except Exception as exc:
                elapsed = time.perf_counter() - start
                failure_dir = output_dir / f"epoch_{checkpoint.epoch}" / mode
                failure_dir.mkdir(parents=True, exist_ok=True)
                error = traceback.format_exc()
                with (failure_dir / "failure.txt").open("w", encoding="utf-8") as file:
                    file.write(error)
                epoch_result["modes"][mode] = {
                    "status": "failed",
                    "elapsed_seconds": elapsed,
                    "error": str(exc),
                }
                failures.append((checkpoint.epoch, mode, str(exc)))
                print(error)
            _write_summary(summary_path, summary)

    if model is not None:
        del model
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    if failures:
        details = "; ".join(
            f"epoch {epoch} {mode}: {error}" for epoch, mode, error in failures
        )
        raise RuntimeError(f"one or more evaluations failed: {details}")
    return summary_path


def build_parser():
    """定义批量 Fast-LeWAM 评测的命令行接口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", help="包含 *_weights_epoch_N.pt 的目录")
    parser.add_argument("--epochs", nargs="+", type=int, help="只评测指定 epoch")
    parser.add_argument("--config-name", default="cube", help="config/eval 下的任务名")
    parser.add_argument(
        "--dataset-name",
        help="覆盖 eval 数据集；默认从训练 run config 推导",
    )
    parser.add_argument("--run-config", help="训练 config.yaml；默认取 checkpoints 上一级")
    parser.add_argument("--output-dir", help="默认 <run>/eval_fast_lewam")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("stage_a", "stage_b"),
        default=("stage_a", "stage_b"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, help="覆盖 Stage B CEM num_samples", default=300)
    parser.add_argument("--cem-steps", type=int, help="覆盖 Stage B CEM n_steps", default=30)
    parser.add_argument("--topk", type=int, help="覆盖 Stage B CEM topk", default=30)
    parser.add_argument("--no-video", action="store_true", help="关闭视频生成")
    return parser


def main():
    """解析 CLI，完成评测，并打印汇总文件位置。"""
    summary_path = run_evaluation(build_parser().parse_args())
    print(f"\nEvaluation summary: {summary_path}")


if __name__ == "__main__":
    main()
