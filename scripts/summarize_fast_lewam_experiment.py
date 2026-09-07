"""Verify and summarize one isolated Fast-LeWAM experiment run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_VALIDATION_METRICS = (
    "validate/stage_b_clean_terminal_mse",
    "validate/stage_b_predicted_terminal_mse",
    "validate/stage_b_expert_preference_accuracy",
    "validate/stage_b_expert_top1_rate",
    "validate/stage_b_expert_negative_margin",
)


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required experiment artifact is missing: {path}")
    return path


def _stage_summary(run_dir: Path, stage: str) -> dict:
    result_path = _require_file(
        run_dir / "eval" / "epoch_10" / stage / "result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "ok":
        raise RuntimeError(
            f"evaluation stage {stage!r} did not finish successfully: "
            f"status={result.get('status')!r}"
        )
    return {
        "success_rate": float(result["success_rate"]),
        "evaluation_seconds": float(result["evaluation_seconds"]),
        "num_eval": int(result.get("parameters", {}).get("num_eval", 0)),
        "result_path": str(result_path),
    }


def _peak_device_memory(run_dir: Path) -> int | None:
    samples_path = run_dir / "gpu_memory_samples.csv"
    if not samples_path.is_file():
        return None
    peak = None
    with samples_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = row.get("memory_used_mib", "").strip()
            if value:
                memory_used = int(value)
                peak = memory_used if peak is None else max(peak, memory_used)
    return peak


def _validation_metrics(run_dir: Path) -> dict[str, float]:
    values = {}
    for metrics_path in sorted(run_dir.glob("lightning_logs/*/metrics.csv")):
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for name, raw_value in row.items():
                    if not name.startswith("validate/") or not raw_value:
                        continue
                    values[name] = float(raw_value)
    missing = [
        name for name in REQUIRED_VALIDATION_METRICS if name not in values
    ]
    if missing:
        raise RuntimeError(
            f"required validation diagnostics are missing: {missing}"
        )
    return values


def summarize_experiment(
    run_dir,
    *,
    elapsed_seconds: int,
    stages: tuple[str, ...],
) -> dict:
    run_dir = Path(run_dir).expanduser().resolve()
    checkpoints = run_dir / "checkpoints"
    weights_path = _require_file(
        checkpoints / "fast_lewam_weights_epoch_10.pt"
    )
    last_path = _require_file(checkpoints / "last.ckpt")
    stage_results = {stage: _stage_summary(run_dir, stage) for stage in stages}
    summary = {
        "status": "ok",
        "run_dir": str(run_dir),
        "elapsed_seconds": int(elapsed_seconds),
        "peak_device_memory_used_mib": _peak_device_memory(run_dir),
        "checkpoints": {
            "epoch_10_weights": str(weights_path),
            "last": str(last_path),
        },
        "stages": stage_results,
        "validation": _validation_metrics(run_dir),
    }
    output_path = run_dir / "acceptance_summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--elapsed-seconds", required=True, type=int)
    parser.add_argument("--stages", nargs="+", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = summarize_experiment(
        args.run_dir,
        elapsed_seconds=args.elapsed_seconds,
        stages=tuple(args.stages),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
