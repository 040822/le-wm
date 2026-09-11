#!/usr/bin/env python3
"""Round 4 standalone training, evaluation, and artifact tooling.

The evaluator reuses the frozen Round 3 cohort and trace implementation but
has its own P0--P4 mode registry.  P3/P4 therefore never need to masquerade as
a CEM configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

# Executing ``python scripts/round4.py`` puts ``scripts/`` (not the repository
# root) on sys.path.  Keep the standalone entrypoint usable from any cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.checkpoint import load_policy_or_model
from source.common.eval import EvaluationIdentity, compose_eval_config
from source.common.round3_phase1 import CohortManifest, paired_comparison
from source.common.round3_eval import run_round3_evaluation
from source.common.round4_artifacts import validate_artifact_manifest, validate_leflow_manifest
from source.common.round4_eval import run_round4_evaluation, validate_gpu_visibility
from source.common.round4_protocol import (
    ROUND4_MODES,
    ROUND4_TASKS,
    mode_spec,
    should_expand_seed,
    validate_round4_mode,
)


DEFAULT_OUTPUT = ROOT / "outputs" / "round4"


def _validate_gpu_argument(gpu: str | None) -> str:
    if gpu is None or not str(gpu).strip():
        raise ValueError("GPU commands require --gpu with one of physical GPUs 0,1,2,3")
    values = [item.strip() for item in str(gpu).split(",") if item.strip()]
    if (
        len(values) != 1
        or not values[0].isdigit()
        or int(values[0]) not in range(4)
    ):
        raise ValueError(
            f"GPU visibility must select exactly one physical GPU0-3; got {gpu!r}"
        )
    return values[0]


def command_validate(args: argparse.Namespace) -> None:
    for mode in ROUND4_MODES:
        print(json.dumps({"mode": mode, **mode_spec(mode)}, sort_keys=True))
    if args.artifacts:
        print(json.dumps(validate_artifact_manifest(args.artifacts), sort_keys=True))


def command_validate_leflow(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            validate_leflow_manifest(args.manifest),
            indent=2,
            sort_keys=True,
        )
    )


def command_train(args: argparse.Namespace) -> None:
    task = str(args.task)
    gpu = _validate_gpu_argument(args.gpu)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    command = [
        sys.executable,
        str(ROOT / "train.py"),
        "--config-name",
        "round4_abde",
        f"data={task}",
        f"seed={int(args.seed)}",
        "trainer.devices=1",
        "trainer.strategy=auto",
    ]
    if args.max_epochs is not None:
        command.append(f"trainer.max_epochs={int(args.max_epochs)}")
    if args.output_model_name:
        command.append(f"output_model_name={args.output_model_name}")
    print("CUDA_VISIBLE_DEVICES=" + gpu)
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def _evaluation_config(task: str, manifest: CohortManifest, dataset_name: str | None, device: str):
    overrides = [
        f"eval.num_eval={len(manifest.entries)}",
        "output.save_video=false",
        f"solver.device={device}",
    ]
    if dataset_name:
        overrides.append(f"eval.dataset_name={dataset_name}")
    cfg = compose_eval_config(task, overrides=overrides)
    cfg.solver.device = device
    return cfg


def command_evaluate(args: argparse.Namespace) -> None:
    mode = validate_round4_mode(args.mode)
    if str(args.device).startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = _validate_gpu_argument(args.gpu)
    validate_gpu_visibility(args.device)
    manifest = CohortManifest.load(args.cohort)
    cfg = _evaluation_config(args.task, manifest, args.dataset_name, args.device)
    policy_or_model, checkpoint = load_policy_or_model(args.checkpoint)
    identity = EvaluationIdentity(
        entrypoint="round4",
        policy_kind="round4_shared_dit",
        checkpoint=str(checkpoint or args.checkpoint),
        epoch=int(args.epoch),
        stage=mode,
    )
    target = Path(args.output) / args.task / mode / manifest.cohort_kind
    result = run_round4_evaluation(
        cfg,
        task=args.task,
        policy_or_model=policy_or_model,
        mode=mode,
        identity=identity,
        manifest=manifest,
        output_dir=target,
        trace_output_dir=target / "trace",
        device=args.device,
        trace=True,
        candidate_count=int(args.candidate_count),
        flow_steps=int(args.flow_steps),
        solver_batch_size=int(args.solver_batch_size),
        candidate_batch_size=args.candidate_batch_size,
    )
    print(json.dumps({"mode": mode, "result": str(target / "result.json"), "success_rate": result["success_rate"]}, sort_keys=True))


def command_evaluate_leflow(args: argparse.Namespace) -> None:
    """Re-run one frozen LeFlow reference under the same cohort protocol."""
    if str(args.device).startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = _validate_gpu_argument(args.gpu)
    validate_gpu_visibility(args.device)
    manifest = CohortManifest.load(args.cohort)
    cfg = _evaluation_config(args.task, manifest, args.dataset_name, args.device)
    policy_or_model, checkpoint = load_policy_or_model(args.checkpoint)
    from source.model.leflow.latent_planner import LatentPlannerRuntime

    if not isinstance(policy_or_model, LatentPlannerRuntime):
        raise TypeError("evaluate-leflow requires an integrated LeFlow latent_planner.pt payload")
    identity = EvaluationIdentity(
        entrypoint="round4",
        policy_kind="leflow",
        checkpoint=str(checkpoint or args.checkpoint),
        epoch=None,
        stage="stage_b",
    )
    target = Path(args.output) / args.task / "leflow" / manifest.cohort_kind
    result = run_round3_evaluation(
        cfg,
        task=args.task,
        policy_or_model=policy_or_model,
        identity=identity,
        manifest=manifest,
        output_dir=target,
        trace_output_dir=target / "trace",
        device=args.device,
        trace=True,
    )
    print(json.dumps({"method": "leflow", "result": str(target / "result.json"), "success_rate": result["success_rate"]}, sort_keys=True))


def _load_results(root: Path):
    rows = []
    for path in sorted(root.glob("**/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        mode = payload.get("round4_mode", payload.get("stage"))
        rows.append(
            {
                "path": str(path),
                "task": payload.get("task"),
                "mode": mode,
                "cohort_kind": payload.get("cohort_kind"),
                "success_rate": payload.get("success_rate"),
                "evaluation_seconds": payload.get("evaluation_seconds"),
                "planning_median_seconds": payload.get("round4_planning", {}).get("planning_median_seconds"),
                "episodes": payload.get("episodes", []),
            }
        )
    return rows


def command_analyze(args: argparse.Namespace) -> None:
    rows = _load_results(Path(args.results_root))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "task",
                "mode",
                "cohort_kind",
                "success_rate",
                "evaluation_seconds",
                "planning_median_seconds",
                "path",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    grouped = {}
    for row in rows:
        grouped[(row.get("task"), row.get("cohort_kind"), row.get("mode"))] = row
    pairs = {}
    for task in ROUND4_TASKS:
        for cohort_kind in ("dev", "final"):
            for left, right in (("P4", "P3"), ("P4", "P2"), ("P4", "P4-first")):
                candidate = grouped.get((task, cohort_kind, left))
                baseline = grouped.get((task, cohort_kind, right))
                if candidate is None or baseline is None:
                    continue
                try:
                    pairs[f"{task}/{cohort_kind}/{left}_vs_{right}"] = paired_comparison(
                        baseline["episodes"], candidate["episodes"]
                    )
                except ValueError as exc:
                    pairs[f"{task}/{cohort_kind}/{left}_vs_{right}"] = {
                        "status": "unavailable",
                        "reason": str(exc),
                    }
    analysis_path = target.with_suffix(".json")
    analysis_path.write_text(
        json.dumps({"rows": rows, "paired": pairs}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "output": str(target), "paired": str(analysis_path)}, sort_keys=True))


def command_expansion_decision(args: argparse.Namespace) -> None:
    payload = json.loads(Path(args.dev).read_text(encoding="utf-8"))
    should_expand, details = should_expand_seed(payload)
    result = {"expand": should_expand, **details}
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--artifacts")
    validate.set_defaults(function=command_validate)

    validate_leflow = subparsers.add_parser("validate-leflow")
    validate_leflow.add_argument("--manifest", required=True)
    validate_leflow.set_defaults(function=command_validate_leflow)

    train = subparsers.add_parser("train")
    train.add_argument("task", choices=ROUND4_TASKS)
    train.add_argument("--gpu", required=True, help="physical GPU subset, e.g. 2")
    train.add_argument("--seed", type=int, default=3072)
    train.add_argument("--max-epochs", type=int)
    train.add_argument("--output-model-name")
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(function=command_train)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("task", choices=ROUND4_TASKS)
    evaluate.add_argument("mode", choices=ROUND4_MODES)
    evaluate.add_argument("--cohort", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--epoch", type=int, default=10)
    evaluate.add_argument("--dataset-name")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--gpu", help="required for CUDA evaluation; one physical GPU0-3")
    evaluate.add_argument("--output", default=str(DEFAULT_OUTPUT))
    evaluate.add_argument("--candidate-count", type=int, default=64)
    evaluate.add_argument("--flow-steps", type=int, default=16)
    evaluate.add_argument("--solver-batch-size", type=int, default=1)
    evaluate.add_argument("--candidate-batch-size", type=int)
    evaluate.set_defaults(function=command_evaluate)

    evaluate_leflow = subparsers.add_parser("evaluate-leflow")
    evaluate_leflow.add_argument("task", choices=ROUND4_TASKS)
    evaluate_leflow.add_argument("--cohort", required=True)
    evaluate_leflow.add_argument("--checkpoint", required=True)
    evaluate_leflow.add_argument("--dataset-name")
    evaluate_leflow.add_argument("--device", default="cpu")
    evaluate_leflow.add_argument("--gpu", help="required for CUDA evaluation; one physical GPU0-3")
    evaluate_leflow.add_argument("--output", default=str(DEFAULT_OUTPUT))
    evaluate_leflow.set_defaults(function=command_evaluate_leflow)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--results-root", default=str(DEFAULT_OUTPUT))
    analyze.add_argument("--output", default=str(DEFAULT_OUTPUT / "round4_matrix.csv"))
    analyze.set_defaults(function=command_analyze)

    decision = subparsers.add_parser("expansion-decision")
    decision.add_argument("--dev", required=True)
    decision.add_argument("--output", default=str(DEFAULT_OUTPUT / "expansion_decision.json"))
    decision.set_defaults(function=command_expansion_decision)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
