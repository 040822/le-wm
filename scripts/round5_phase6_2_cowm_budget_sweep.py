#!/usr/bin/env python3
"""Evaluate the frozen CoWM-Selection Reacher baseline at extended budgets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "config/round5/phase6_2_baseline_budget_sweep.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gpu_snapshot(gpu: int, *, enforce_margin: bool) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    query = subprocess.run(
        [
            "nvidia-smi", "-i", str(gpu),
            "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()
    fields = next(csv.reader([query], skipinitialspace=True))
    if len(fields) != 8 or int(fields[0]) != gpu:
        raise RuntimeError(f"unexpected nvidia-smi response: {query!r}")
    value = {
        "physical_id": int(fields[0]),
        "memory_total_mib": int(fields[1]),
        "memory_used_mib": int(fields[2]),
        "memory_free_mib": int(fields[3]),
        "utilization_percent": int(fields[4]),
        "uuid": fields[5],
        "name": fields[6],
        "driver_version": fields[7],
    }
    minimum = 12 * 1024
    if enforce_margin and value["memory_free_mib"] < minimum:
        raise RuntimeError(
            f"GPU {gpu} has {value['memory_free_mib'] / 1024:.2f} GiB free; "
            "12 GiB minimum safety margin is required"
        )
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[60, 75, 90, 110])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    allowed_budgets = {int(item) for item in protocol["budgets"]}
    if not args.budgets or len(set(args.budgets)) != len(args.budgets):
        raise ValueError("budgets must be a nonempty list without duplicates")
    if not set(args.budgets) <= allowed_budgets:
        raise ValueError(f"budgets must be selected from {sorted(allowed_budgets)}")

    from source.common.gpu_environment import configure_mujoco_egl_device

    selected_gpu = configure_mujoco_egl_device()
    if selected_gpu != args.gpu or selected_gpu not in range(4):
        raise RuntimeError(
            f"--gpu={args.gpu} must equal CUDA_VISIBLE_DEVICES={selected_gpu} in GPU0–3"
        )
    _gpu_snapshot(selected_gpu, enforce_margin=True)

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from source.common.checkpoint import load_policy_or_model
    from source.common.eval import EvaluationIdentity, compose_eval_config
    from source.common.round3_phase1 import CohortManifest
    from source.common.round4_eval import run_round4_evaluation

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")

    method = config["methods"]["cowm"]
    checkpoint = ROOT / method["checkpoint"]
    actual_checkpoint_sha = _sha256(checkpoint)
    if actual_checkpoint_sha != method["checkpoint_sha256"]:
        raise RuntimeError("CoWM checkpoint SHA256 differs from the frozen config")
    manifest = CohortManifest.load(ROOT / config["cohort"]["path"])
    if manifest.computed_sha256 != config["cohort"]["sha256"]:
        raise RuntimeError("Reacher cohort SHA256 differs from the frozen config")
    if len(manifest.entries) != int(config["cohort"]["episodes"]):
        raise RuntimeError("Reacher cohort episode count differs from the frozen config")

    model, resolved_checkpoint = load_policy_or_model(str(checkpoint), cache_dir=str(ROOT / "data"))
    model = getattr(model, "model", model)
    settings = method["settings"]
    output_root = ROOT / config["output_root"]
    runner_sha = _sha256(Path(__file__))
    config_sha = _sha256(args.config)

    for budget in args.budgets:
        target = output_root / f"budget_{budget}" / "cowm" / "25_25"
        result_path = target / "result.json"
        if result_path.is_file():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            metadata = existing.get("phase6_2_baseline_sweep", {})
            if metadata.get("budget") == budget and metadata.get("checkpoint_sha256") == actual_checkpoint_sha:
                print(json.dumps({"status": "resumed", "method": "cowm", "budget": budget, "result": str(result_path)}, flush=True))
                continue
            raise RuntimeError(f"refusing to overwrite incompatible result: {result_path}")
        if target.exists() and any(target.iterdir()):
            raise RuntimeError(f"refusing to overwrite partial output: {target}")

        gpu_before = _gpu_snapshot(selected_gpu, enforce_margin=True)
        cohort_seed = int(config["cohort"]["seed"])
        environment_seed = cohort_seed + int(config["seed_policy"]["environment_seed_offset"])
        policy_seed = cohort_seed + int(config["seed_policy"]["policy_seed_offset"])
        torch.manual_seed(policy_seed)
        torch.cuda.manual_seed_all(policy_seed)
        np.random.seed(policy_seed % (2**32 - 1))
        overrides = [
            "output.save_video=false",
            f"eval.eval_budget={budget}",
            f"eval.num_eval={config['cohort']['episodes']}",
            f"eval.goal_offset_steps={config['cohort']['goal_offset_steps']}",
            "plan_config.horizon=5",
            "plan_config.receding_horizon=5",
            "plan_config.action_block=5",
            f"eval.policy_seed={policy_seed}",
        ]
        cfg = compose_eval_config("reacher", overrides=overrides)
        cfg.seed = environment_seed
        cfg.solver.seed = policy_seed
        identity = EvaluationIdentity(
            entrypoint="round5_phase6_2_cowm_budget_sweep",
            policy_kind="fast_lewam",
            checkpoint=str((resolved_checkpoint or checkpoint).resolve()),
            epoch=10,
            stage="stage_b",
        )
        started = time.perf_counter()
        result = run_round4_evaluation(
            cfg,
            task="reacher",
            policy_or_model=model,
            mode=str(settings["mode"]),
            identity=identity,
            manifest=manifest,
            output_dir=target,
            trace_output_dir=target / "trace",
            device="cuda:0",
            trace=True,
            candidate_count=int(settings["candidate_count"]),
            flow_steps=int(settings["action_flow_steps"]),
            action_flow_steps=int(settings["action_flow_steps"]),
            solver_batch_size=int(settings["solver_batch_size"]),
            candidate_batch_size=int(settings["candidate_batch_size"]),
            action_flow_integrator=str(settings["action_flow_integrator"]),
            action_bound_mode=str(settings["action_bound_mode"]),
            cem_protocol="not_applicable",
            guidance_mode=str(settings["guidance_mode"]),
            guidance_step_size=0.01,
            guidance_last_steps=2,
            guidance_inner_steps=1,
            guidance_max_rms_offset=0.2,
            proposal_chunk_size=int(settings["proposal_chunk_size"]),
            allowed_protocol_variants=("legacy",),
            execute_steps=int(protocol["execute_steps"]),
            score_horizon_blocks=int(protocol["plan_horizon_blocks"]),
            policy_seed=policy_seed,
            allow_solver_config_override=True,
            allow_evaluation_seed_override=True,
            allow_eval_budget_override=True,
            allow_cohort_seed_mismatch=True,
        )
        if result.get("status") != "ok":
            raise RuntimeError(f"CoWM evaluation returned {result.get('status')!r}")
        gpu_after = _gpu_snapshot(selected_gpu, enforce_margin=False)
        result_path = target / "result.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["phase6_2_baseline_sweep"] = {
            "method": "cowm",
            "label": method["label"],
            "budget": int(budget),
            "execute_steps": int(protocol["execute_steps"]),
            "score_steps": int(protocol["score_steps"]),
            "cohort_seed": cohort_seed,
            "environment_seed": environment_seed,
            "policy_seed": policy_seed,
            "checkpoint": str((resolved_checkpoint or checkpoint).resolve()),
            "checkpoint_sha256": actual_checkpoint_sha,
            "cohort_sha256": manifest.computed_sha256,
            "config_sha256": config_sha,
            "runner_sha256": runner_sha,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "wall_seconds": time.perf_counter() - started,
            "settings": settings,
            "baseline_result": method["baseline_result"],
        }
        _write_json(result_path, payload)
        print(json.dumps({
            "status": "ok",
            "method": "cowm",
            "budget": int(budget),
            "successes": sum(bool(row["success"]) for row in payload["episodes"]),
            "episodes": len(payload["episodes"]),
            "gpu": selected_gpu,
            "result": str(result_path),
        }), flush=True)


if __name__ == "__main__":
    main()
