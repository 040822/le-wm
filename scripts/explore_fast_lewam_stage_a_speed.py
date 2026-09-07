"""Evaluate production Stage-A inference steps with trained checkpoints."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from omegaconf import OmegaConf

from eval_fast_lewam import load_model_from_weights
from source.common.eval import (
    DatasetEvaluationSession,
    EvaluationIdentity,
    compose_eval_config,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--num-eval", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    weight = args.run_dir / "checkpoints/fast_lewam_weights_epoch_10.pt"
    cfg = compose_eval_config(
        args.task,
        (f"eval.num_eval={args.num_eval}", "output.save_video=false"),
    )
    session = DatasetEvaluationSession(cfg, task=args.task)
    model = load_model_from_weights(
        OmegaConf.load(args.run_dir / "config.yaml"), weight, device="cuda"
    )
    identity = EvaluationIdentity(
        entrypoint="stage_a_speed_validation",
        policy_kind="fast_lewam",
        checkpoint=str(weight),
        epoch=10,
        stage="stage_a",
    )
    results = []

    def evaluate(label, steps):
        session.cfg.fast_lewam = {"inference_steps": steps}
        result = session.evaluate(
            model, identity=identity, output_dir=args.output / label, device="cuda"
        )
        row = {
            "label": label,
            "steps": steps,
            "success_rate": result.success_rate,
            "evaluation_seconds": result.evaluation_seconds,
            "episode_successes": [bool(item.success) for item in result.episodes],
        }
        results.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "episode_successes"}), flush=True)

    for steps in (10, 5, 2, 1):
        evaluate(f"production_fp32_steps_{steps}", steps)

    baseline = np.asarray(results[0]["episode_successes"], dtype=bool)
    for row in results:
        candidate = np.asarray(row["episode_successes"], dtype=bool)
        row.update(
            speedup_vs_production_10=results[0]["evaluation_seconds"]
            / row["evaluation_seconds"],
            regressed=int((baseline & ~candidate).sum()),
            improved=int((~baseline & candidate).sum()),
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
