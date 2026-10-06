#!/usr/bin/env python3
"""Evaluate a converted official Sub-JEPA model with the shared LeWM protocol."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("tworoom", "pusht", "reacher", "cube"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.device == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1", "2", "3"}:
            parser.error("Select one physical GPU0-3 explicitly with CUDA_VISIBLE_DEVICES")
        from source.common.gpu_environment import configure_mujoco_egl_device

        configure_mujoco_egl_device()

    from omegaconf import OmegaConf
    from scripts.prepare_subjepa_official import sha256
    from source.common.checkpoint import load_policy_or_model
    from source.common.eval import (
        DatasetEvaluationSession, EvaluationIdentity, compose_eval_config, write_evaluation_failure,
    )
    from source.model.subjepa.official import OfficialSubJEPA

    checkpoint = (args.checkpoint or ROOT / "data/checkpoints/subjepa_official" / args.task / "subjepa.pt").resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir or ROOT / "outputs/subjepa_official" / args.task / f"seed_{args.seed}" / stamp
    if output_dir.exists():
        raise FileExistsError(f"Evaluation output already exists; choose a new attempt: {output_dir}")
    try:
        model, loaded_path = load_policy_or_model(checkpoint, cache_dir=args.cache_dir)
        if not isinstance(model, OfficialSubJEPA) or model.provenance["task"] != args.task:
            raise ValueError("Checkpoint method/task does not match the requested Sub-JEPA evaluation")
        checkpoint_hash = sha256(loaded_path)
        manifest_path = loaded_path.parent / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if checkpoint_hash != manifest["converted_sha256"]:
                raise ValueError("Converted checkpoint SHA256 differs from its preparation manifest")
        cfg = compose_eval_config(args.task, [
            f"seed={args.seed}", f"cache_dir={args.cache_dir.resolve()}",
            f"solver.device={args.device}", *args.override,
        ])
        if int(cfg.plan_config.action_block) != model.provenance["frameskip"]:
            raise ValueError("Evaluation action_block differs from the checkpoint's training frameskip")
        session = DatasetEvaluationSession(cfg, task=args.task)
        result = session.evaluate(
            model, identity=EvaluationIdentity(
                entrypoint="eval_subjepa_official", policy_kind="subjepa_official",
                checkpoint=str(loaded_path),
            ), output_dir=output_dir, device=args.device,
        )
        OmegaConf.save(cfg, output_dir / "resolved_config.yaml")
        (output_dir / "checkpoint_identity.json").write_text(json.dumps({
            "checkpoint": str(loaded_path), "sha256": checkpoint_hash,
            "provenance": model.provenance,
        }, indent=2) + "\n")
        print(json.dumps({
            "task": args.task, "seed": args.seed, "num_episodes": len(result.episodes),
            "success_rate": result.success_rate, "output_dir": str(output_dir),
        }), flush=True)
    except Exception as exc:
        write_evaluation_failure(output_dir, exc, traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
