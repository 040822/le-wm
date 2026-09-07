"""Command line driver for the auditable Round 3 Phase 1 workflow.

The commands that inspect assets, predicates, cohorts, and prior results are
CPU-only.  The optional ``evaluate`` command is intentionally explicit about
the cohort and checkpoint; it never discovers or substitutes an intermediate
weight.  GPU callers must set ``CUDA_VISIBLE_DEVICES`` to IDs in 0--3.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

from source.common.round3_phase1 import (
    CohortManifest,
    build_artifact_registry,
    build_legacy_manifest,
    build_revised_cohorts,
    enrich_result_payload,
    load_result,
    run_synthetic_goal_refresh_checks,
    write_artifact_registry,
    write_phase1_matrix,
    write_phase1_report,
    write_round3_result,
)
from source.common.round3_runtime_audit import run_runtime_goal_refresh_audit
from source.common.round3_variants import build_development_protocol_manifests
from source.common.round3_protocol import (
    FAST_STAGES,
    PREDICATE_VERSION,
    ROUND3_EVAL_DEFAULTS,
    ROUND3_PROTOCOL,
    TASKS,
    audit_all_predicates,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "round3" / "phase1"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_gpu_visibility(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        raise RuntimeError(
            "GPU evaluation requires CUDA_VISIBLE_DEVICES explicitly set to a subset of 0,1,2,3"
        )
    values = [item.strip() for item in visible.split(",") if item.strip()]
    if not values or any(not item.isdigit() or int(item) not in range(4) for item in values):
        raise RuntimeError(
            f"prohibited GPU visibility {visible!r}; only physical GPUs 0,1,2,3 are allowed"
        )


def command_audit(args: argparse.Namespace) -> None:
    output = Path(args.output)
    refresh = {}
    if args.runtime_refresh:
        refresh = {
            task: run_runtime_goal_refresh_audit(task, count=10, seed=42)
            for task in TASKS
        }
    elif args.synthetic_refresh:
        refresh = {
            task: run_synthetic_goal_refresh_checks(task, count=10, seed=42)
            for task in TASKS
        }
    payloads = audit_all_predicates(goal_refresh=refresh)
    for task, payload in payloads.items():
        _write_json(output / "predicate_audit" / f"{task}.json", payload)
    print(json.dumps({task: value["status"] for task, value in payloads.items()}, sort_keys=True))


def command_registry(args: argparse.Namespace) -> None:
    payload = write_artifact_registry(
        args.spec,
        Path(args.output) / "artifact_registry.json",
        repo_root=ROOT,
    )
    counts = {}
    for entry in payload["entries"]:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    print(json.dumps(counts, sort_keys=True))


def _load_dataset(task: str, dataset_name: str | None, cache_dir: str | None, num_eval: int):
    from source.common.eval import compose_eval_config, get_dataset
    from omegaconf import OmegaConf

    overrides = [f"eval.num_eval={int(num_eval)}"]
    if dataset_name:
        overrides.append(f"eval.dataset_name={dataset_name}")
    if cache_dir:
        overrides.append(f"cache_dir={cache_dir}")
    cfg = compose_eval_config(task, overrides=overrides)
    cfg = OmegaConf.merge(cfg, {"output": {"save_video": False}})
    return cfg, get_dataset(cfg, cfg.eval.dataset_name)


def command_cohort(args: argparse.Namespace) -> None:
    cfg, dataset = _load_dataset(args.task, args.dataset_name, args.cache_dir, args.dev)
    manifests = build_revised_cohorts(
        dataset,
        task=args.task,
        seed=args.seed,
        goal_offset_steps=args.goal_offset_steps,
        dev_count=args.dev,
        final_count=args.final,
        online_count=args.online,
    )
    output = Path(args.output) / "cohorts" / args.task
    for name, manifest in manifests.items():
        manifest.save(output / f"{name}.json")
    print(json.dumps({name: manifest.cohort_sha256 for name, manifest in manifests.items()}, sort_keys=True))

def command_dev_protocols(args: argparse.Namespace) -> None:
    _, dataset = _load_dataset(args.task, args.dataset_name, args.cache_dir, 50)
    manifests = build_development_protocol_manifests(
        dataset,
        task=args.task,
        seed=args.seed,
        goal_offset_steps=args.goal_offset_steps,
    )
    output = Path(args.output) / "cohorts" / args.task
    for variant, manifest in manifests.items():
        manifest.save(output / f"dev_{variant}.json")
    print(json.dumps({variant: manifest.computed_sha256 for variant, manifest in manifests.items()}, sort_keys=True))



def command_legacy_cohort(args: argparse.Namespace) -> None:
    _, dataset = _load_dataset(args.task, args.dataset_name, args.cache_dir, args.num_eval)
    manifest = build_legacy_manifest(
        dataset,
        task=args.task,
        seed=args.seed,
        goal_offset_steps=args.goal_offset_steps,
        num_eval=args.num_eval,
    )
    target = Path(args.output) / "cohorts" / args.task / f"legacy_{args.num_eval}.json"
    manifest.save(target)
    print(manifest.computed_sha256)


def command_analyze(args: argparse.Namespace) -> None:
    output = Path(args.output)
    rows = write_phase1_matrix(output / "results", output / "phase1_matrix.csv")
    registry_path = output / "artifact_registry.json"
    registry = load_result(registry_path) if registry_path.is_file() else None
    write_phase1_report(rows, output / "phase1_report.md", registry=registry)
    print(f"matrix_rows={len(rows)}")


def command_evaluate(args: argparse.Namespace) -> None:
    _validate_gpu_visibility(args.device)
    if args.task not in TASKS:
        raise ValueError(f"unsupported task {args.task!r}")
    if args.method not in {"e0_lewm", "e3_fast", "e5_fast"}:
        raise ValueError(f"unsupported method {args.method!r}")
    if args.method == "e0_lewm" and args.stage != "stage_b":
        raise ValueError("E0 LeWM is registered only for Stage B")
    if args.method != "e0_lewm" and args.stage not in FAST_STAGES:
        raise ValueError(f"Fast-LeWAM stage must be one of {FAST_STAGES}")
    manifest = CohortManifest.load(args.cohort)
    if manifest.task != args.task:
        raise ValueError("cohort task and evaluation task differ")
    if manifest.protocol_variant != args.protocol_variant:
        raise ValueError("cohort protocol variant and requested protocol differ")
    from omegaconf import OmegaConf
    from source.common.checkpoint import load_policy_or_model
    from source.common.eval import EvaluationIdentity
    from source.common.round3_eval import run_round3_evaluation
    from source.common.eval import compose_eval_config

    overrides = [
        f"eval.num_eval={len(manifest.entries)}",
        f"eval.dataset_name={args.dataset_name}" if args.dataset_name else "",
        f"solver.device={args.device}",
        "output.save_video=false",
    ]
    overrides = [item for item in overrides if item]
    cfg = compose_eval_config(args.task, overrides=overrides)
    cfg = OmegaConf.merge(cfg, {"world": {"num_envs": len(manifest.entries)}})
    policy_or_model, checkpoint = load_policy_or_model(args.checkpoint)
    identity = EvaluationIdentity(
        entrypoint="round3_phase1",
        policy_kind="lewm" if args.method == "e0_lewm" else "fast_lewam",
        checkpoint=str(checkpoint or args.checkpoint),
        epoch=args.epoch,
        stage=None if args.method == "e0_lewm" else args.stage,
    )
    output_dir = Path(args.output) / "results" / args.task / args.method / args.protocol_variant / args.stage
    result = run_round3_evaluation(
        cfg,
        task=args.task,
        policy_or_model=policy_or_model,
        identity=identity,
        manifest=manifest,
        output_dir=output_dir,
        device=args.device,
        trace=True,
    )
    print("success_rate={:.4f} cohort_sha256={}".format(result["success_rate"], manifest.computed_sha256))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--output", default=str(DEFAULT_OUTPUT))
    audit.add_argument("--synthetic-refresh", action="store_true", help="run the adapter contract with deterministic in-memory states")
    audit.add_argument("--runtime-refresh", action="store_true", help="bind the ten checks to installed CPU environments")
    audit.set_defaults(function=command_audit)
    registry = subparsers.add_parser("registry")
    registry.add_argument("--spec", default=str(ROOT / "config" / "round3" / "phase1_artifacts.json"))
    protocols = subparsers.add_parser("dev-protocols")
    protocols.add_argument("task", choices=TASKS)
    protocols.add_argument("--dataset-name")
    protocols.add_argument("--cache-dir")
    protocols.add_argument("--output", default=str(DEFAULT_OUTPUT))
    protocols.add_argument("--seed", type=int, default=int(ROUND3_EVAL_DEFAULTS["seed"]))
    protocols.add_argument("--goal-offset-steps", type=int, default=int(ROUND3_EVAL_DEFAULTS["goal_offset_steps"]))
    protocols.set_defaults(function=command_dev_protocols)

    registry.add_argument("--output", default=str(DEFAULT_OUTPUT))
    registry.set_defaults(function=command_registry)

    cohort = subparsers.add_parser("cohort")
    cohort.add_argument("task", choices=TASKS)
    cohort.add_argument("--dataset-name")
    cohort.add_argument("--cache-dir")
    cohort.add_argument("--output", default=str(DEFAULT_OUTPUT))
    cohort.add_argument("--seed", type=int, default=int(ROUND3_EVAL_DEFAULTS["seed"]))
    cohort.add_argument("--goal-offset-steps", type=int, default=int(ROUND3_EVAL_DEFAULTS["goal_offset_steps"]))
    cohort.add_argument("--dev", type=int, default=50)
    cohort.add_argument("--final", type=int, default=200)
    cohort.add_argument("--online", type=int)
    cohort.set_defaults(function=command_cohort)

    legacy = subparsers.add_parser("legacy-cohort")
    legacy.add_argument("task", choices=TASKS)
    legacy.add_argument("--dataset-name")
    legacy.add_argument("--cache-dir")
    legacy.add_argument("--num-eval", type=int, default=50)
    legacy.add_argument("--output", default=str(DEFAULT_OUTPUT))
    legacy.add_argument("--seed", type=int, default=int(ROUND3_EVAL_DEFAULTS["seed"]))
    legacy.add_argument("--goal-offset-steps", type=int, default=int(ROUND3_EVAL_DEFAULTS["goal_offset_steps"]))
    legacy.set_defaults(function=command_legacy_cohort)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--output", default=str(DEFAULT_OUTPUT))
    analyze.set_defaults(function=command_analyze)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("task", choices=TASKS)
    evaluate.add_argument("method", choices=("e0_lewm", "e3_fast", "e5_fast"))
    evaluate.add_argument("stage", choices=("stage_a", "stage_a_shuffled_goal", "stage_b"))
    evaluate.add_argument("--protocol-variant", choices=("legacy", "sampling_revised", "tolerance_revised", "round3_revised"), default="round3_revised")
    evaluate.add_argument("--cohort", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--epoch", type=int, default=10)
    evaluate.add_argument("--dataset-name")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--output", default=str(DEFAULT_OUTPUT))
    evaluate.set_defaults(function=command_evaluate)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.function(args)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
