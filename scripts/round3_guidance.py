"""Run Round 3 E1/E2 guidance diagnostics before fixed-replay collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASKS = ("reacher", "pusht")
DEFAULT_CHECKPOINTS = {
    "reacher": ROOT / "outputs/fast_lewam/reacher/0717_/checkpoints/fast_lewam_weights_epoch_10.pt",
    "pusht": ROOT / "outputs/fast_lewam/pusht/0803_e5_cross_head_grad/checkpoints/fast_lewam_weights_epoch_10.pt",
}


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def run_e2(*, task: str, checkpoint: Path, output_dir: Path, device: str) -> dict:
    from source.common.round3_eval import validate_gpu_visibility
    from source.common.gpu_environment import configure_mujoco_egl_device

    validate_gpu_visibility(device)
    if str(device).startswith("cuda"):
        configure_mujoco_egl_device()
    from omegaconf import OmegaConf

    from source.common.checkpoint import load_policy_or_model
    from source.common.eval import EvaluationIdentity, compose_eval_config, get_dataset
    from source.common.round3_eval import run_round3_evaluation
    from source.common.round3_phase1 import CohortManifest
    from source.experiments.round3_guidance_diagnostics import (
        GuidanceDiagnosticConfig,
        e2_method_specs,
    )

    cohort = CohortManifest.load(
        ROOT / "outputs/round3/phase1/cohorts" / task / "dev_round3_revised.json"
    )
    if cohort.task != task or cohort.cohort_kind != "dev" or len(cohort.entries) != 50:
        raise ValueError("E2 requires the frozen 50-episode round3_revised dev cohort")
    config = GuidanceDiagnosticConfig()
    specs = e2_method_specs(config)
    cfg = compose_eval_config(task, (f"solver.device={device}", "output.save_video=false"))
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    results = {}
    for method, spec in specs.items():
        overrides = {
            "eval": {"num_eval": 50},
            "world": {"num_envs": 50},
            "output": {"save_video": False},
        }
        if spec.get("solver_samples") is not None:
            overrides["solver"] = {
                "num_samples": int(spec["solver_samples"]),
                "topk": int(spec.get("solver_topk", config.cem_topk)),
            }
        method_cfg = OmegaConf.merge(cfg, overrides)
        model, resolved = load_policy_or_model(str(checkpoint))
        identity = EvaluationIdentity(
            entrypoint="round3_guidance",
            policy_kind="e5_fast",
            checkpoint=str(resolved or checkpoint),
            epoch=10,
            stage=str(spec["stage"]),
            actor_warm_start=bool(spec.get("actor_warm_start", False)),
            guidance_mode=str(spec.get("guidance_mode", "none")),
            guidance_step_size=config.guidance_step_size,
            guidance_last_steps=config.guidance_last_steps,
            guidance_inner_steps=config.guidance_inner_steps,
            guidance_max_rms_offset=config.guidance_max_rms_offset,
        )
        result = run_round3_evaluation(
            method_cfg,
            task=task,
            policy_or_model=model,
            identity=identity,
            manifest=cohort,
            dataset=dataset,
            output_dir=output_dir / method,
            trace_output_dir=None,
            device=device,
            trace=False,
            allow_solver_budget_overrides=True,
        )
        results[method] = {
            "success_rate": float(result["success_rate"]),
            "episodes": len(result["episodes"]),
            "cohort_sha256": cohort.computed_sha256,
            "result_path": str(output_dir / method / "result.json"),
            "spec": spec,
        }
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    payload = {
        "schema_version": 1,
        "phase": "round3_e2_guidance",
        "task": task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
        "cohort": {
            "path": str(ROOT / "outputs/round3/phase1/cohorts" / task / "dev_round3_revised.json"),
            "kind": "dev",
            "episodes": 50,
            "canonical_sha256": cohort.computed_sha256,
        },
        "config": __import__("dataclasses").asdict(config),
        "methods": results,
        "decision": "report_only_user_selects_collector",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "e2_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def run_e1(*, task: str, checkpoint: Path, pool_path: Path, output_dir: Path, device: str) -> dict:
    """Collect the paired 64-state signed-gradient E1 panels."""
    from source.experiments.round3_guidance_diagnostics import (
        GuidanceDiagnosticConfig,
        summarize_e1_rows,
    )
    from source.experiments.round3_grounded_collection import collect_grounded_replay

    config = GuidanceDiagnosticConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for rms in config.e1_rms:
        replay_path = output_dir / f"rms_{rms:g}.pt"
        report = collect_grounded_replay(
            task=task,
            checkpoint=checkpoint,
            pool_path=pool_path,
            output=replay_path,
            device=device,
            groups=config.e1_states,
            pool_offset=0,
            seed=config.seed,
            guidance_mode="none",
            panel_mode="signed_gradient",
            perturbation_rms=float(rms),
            collector_version="round3_e1_signed_gradient_v1",
            resume=True,
            allow_empty_replay=True,
        )
        reports.append(report)
    rows = []
    for report in reports:
        for group in report["groups"]:
            costs = group["physical_terminal_costs"]
            latent = group["candidate_latent_costs"]
            rms = float(group["requested_perturbation_rms"])
            for index, method in enumerate(group["candidate_sources"]):
                rows.append(
                    {
                        "group_id": group["group_id"],
                        "rms": rms,
                        "method": method,
                        "physical_cost_delta": float(costs[index] - costs[0]),
                        "latent_cost_delta": float(latent[index] - latent[0]),
                        "success": bool(group["successes"][index]),
                    }
                )
    payload = {
        "schema_version": 1,
        "phase": "round3_e1_guidance",
        "task": task,
        "checkpoint": str(checkpoint),
        "config": __import__("dataclasses").asdict(config),
        "collection_reports": reports,
        "excluded_episode_ids": sorted(
            {
                entry["episode_id"]
                for report in reports
                for entry in json.loads(
                    Path(report["group_manifest"]).read_text(encoding="utf-8")
                )["entries"]
            },
            key=repr,
        ),
        "summary": summarize_e1_rows(rows),
        "rows": rows,
        "decision": "report_only_user_selects_collector",
    }
    (output_dir / "e1_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--task", choices=TASKS, required=True)
    manifest.add_argument("--checkpoint", default=None)
    manifest.add_argument("--output", required=True)

    e2 = sub.add_parser("e2")
    e2.add_argument("--task", choices=TASKS, required=True)
    e2.add_argument("--checkpoint", default=None)
    e2.add_argument("--output-dir", required=True)
    e2.add_argument("--device", default="cuda:0")

    e1 = sub.add_parser("e1")
    e1.add_argument("--task", choices=TASKS, required=True)
    e1.add_argument("--checkpoint", default=None)
    e1.add_argument("--pool", required=True)
    e1.add_argument("--output-dir", required=True)
    e1.add_argument("--device", default="cuda:0")

    gradient = sub.add_parser("gradient-check")
    gradient.add_argument("--task", choices=TASKS, required=True)
    gradient.add_argument("--checkpoint", default=None)
    gradient.add_argument("--device", default="cpu")
    gradient.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    checkpoint = _path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINTS[args.task]
    if args.command == "manifest":
        from source.experiments.round3_guidance_diagnostics import make_guidance_manifest

        payload = make_guidance_manifest(
            task=args.task,
            checkpoint=checkpoint,
            output=_path(args.output),
        )
    elif args.command == "e2":
        payload = run_e2(
            task=args.task,
            checkpoint=checkpoint,
            output_dir=_path(args.output_dir),
            device=args.device,
        )
    elif args.command == "e1":
        from source.common.round3_eval import validate_gpu_visibility

        validate_gpu_visibility(args.device)
        payload = run_e1(
            task=args.task,
            checkpoint=checkpoint,
            pool_path=_path(args.pool),
            output_dir=_path(args.output_dir),
            device=args.device,
        )
    else:
        from source.common.checkpoint import load_policy_or_model
        from source.experiments.round3_guidance_diagnostics import finite_difference_guidance_check

        from source.common.round3_eval import validate_gpu_visibility

        validate_gpu_visibility(args.device)
        model, _ = load_policy_or_model(str(checkpoint))
        model = getattr(model, "model", model).to(args.device).eval()
        torch.manual_seed(42)
        z0 = torch.randn(1, int(model.latent_dim), device=args.device)
        goal = torch.randn(1, int(model.latent_dim), device=args.device)
        actions = torch.randn(
            1,
            int(model.action_horizon),
            int(model.action_dim),
            device=args.device,
        )
        payload = finite_difference_guidance_check(model, z0, goal, actions)
        path = _path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "task": args.task, "output": str(args.output if hasattr(args, "output") else args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
