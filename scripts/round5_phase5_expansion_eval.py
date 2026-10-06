#!/usr/bin/env python3
"""Evaluate the five native action structures on frozen PushT/Reacher dev cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.checkpoint import load_policy_or_model
from source.common.eval import EvaluationIdentity, compose_eval_config
from source.common.round3_phase1 import CohortManifest
from source.common.round4_eval import run_round4_evaluation, validate_gpu_visibility


DEFAULT_CONFIG = ROOT / "config" / "round5" / "phase5_expansion_eval.json"
BASE_ACTION_DIMS = {"pusht": 2, "reacher": 2}
CODE_FILES = (
    "config/round5/phase5_expansion_eval.json",
    "scripts/round5_phase5_expansion_eval.py",
    "source/common/round4_eval.py",
    "source/common/round4_protocol.py",
    "source/common/eval.py",
    "source/common/round3_eval.py",
    "source/policy/round4.py",
    "source/policy/fast_lewam.py",
    "source/policy/fast_lewam_eval.py",
    "source/model/fast_lewam/jepa.py",
    "source/model/fast_lewam/round4.py",
)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"evaluation config must be a JSON object: {path}")
    return value


def _condition_specs(config: Mapping[str, Any], task: str, structure: Mapping[str, Any]):
    name = str(structure["name"])
    block = int(structure["action_block_steps"])
    horizon = int(structure["action_horizon"])
    max_steps = block * horizon
    executes = [int(value) for value in structure["execute_steps"]]
    scores = [int(value) for value in structure["score_steps"]]
    if any(value < 1 or value > max_steps for value in executes + scores):
        raise ValueError(f"invalid native evaluation grid for {name}")

    specs = []
    for execute_steps in executes:
        for score_steps in scores:
            if score_steps % block:
                raise ValueError(
                    f"score step {score_steps} is not representable by {name} blocks"
                )
            score_blocks = score_steps // block
            if score_blocks > horizon:
                raise ValueError(f"score step {score_steps} exceeds {name} prediction span")
            specs.append(
                _make_spec(
                    task,
                    name,
                    "P3",
                    execute_steps,
                    score_steps,
                    score_blocks,
                    config,
                )
            )
        for method in ("P0", "Random-64"):
            specs.append(
                _make_spec(task, name, method, execute_steps, None, None, config)
            )
    return specs


def _make_spec(task, structure, method, execute_steps, score_steps, score_blocks, config):
    evaluation = config["evaluation"]
    method_key = {"P3": "p3", "P0": "p0", "Random-64": "random64"}[method]
    suffix = (
        f"{method_key}_exec{execute_steps}_score{score_steps}"
        if score_steps is not None
        else f"{method_key}_exec{execute_steps}"
    )
    mode = "P3" if method == "Random-64" else method
    selection_rule = "random" if method == "Random-64" else None
    condition = {
        "task": str(task),
        "structure": str(structure),
        "method": str(method),
        "mode": mode,
        "selection_rule": selection_rule or ("argmin_verifier" if mode == "P3" else "direct"),
        "execute_steps": int(execute_steps),
        "score_steps_env": None if score_steps is None else int(score_steps),
        "score_horizon_blocks": None if score_blocks is None else int(score_blocks),
        "candidate_count": (
            int(evaluation["p3_candidate_count"]) if mode == "P3" else 1
        ),
        "action_flow_steps": int(evaluation["action_flow_steps"]),
        "action_flow_integrator": str(evaluation["action_flow_integrator"]),
        "action_bound_mode": str(evaluation["action_bound_mode"]),
        "evaluation_seed": int(evaluation["seed"]),
        "policy_seed": int(evaluation["seed"]),
    }
    return {
        "id": suffix,
        "method": method,
        "mode": mode,
        "selection_rule": selection_rule,
        "execute_steps": int(execute_steps),
        "score_steps_env": None if score_steps is None else int(score_steps),
        "score_horizon_blocks": None if score_blocks is None else int(score_blocks),
        "condition": condition,
        "condition_sha256": _canonical_sha256(condition),
    }


def _all_specs(config: Mapping[str, Any]):
    for task in config["tasks"]:
        for structure in config["structures"]:
            for spec in _condition_specs(config, str(task), structure):
                yield str(task), str(structure["name"]), spec


def _load_inputs(config: Mapping[str, Any]):
    state_path = _resolve(config["training"]["state"])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    hashes_path = state_path.parent / "dataset_hashes.json"
    dataset_hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    outputs: dict[tuple[str, str], dict[str, Any]] = {}

    for task in config["tasks"]:
        task = str(task)
        cohort_config = config["cohorts"][task]
        manifest_path = _resolve(cohort_config["path"])
        manifest = CohortManifest.load(manifest_path)
        if manifest.task != task or manifest.cohort_kind != "dev":
            raise ValueError(f"wrong task or cohort kind in {manifest_path}")
        if manifest.protocol_variant != cohort_config["protocol_variant"]:
            raise ValueError(f"cohort protocol mismatch for {task}")
        if manifest.computed_sha256 != cohort_config["sha256"]:
            raise ValueError(f"cohort hash mismatch for {task}")
        evaluation = config["evaluation"]
        if len(manifest.entries) != int(evaluation["num_eval"]):
            raise ValueError(f"wrong dev cohort size for {task}")
        if int(manifest.goal_offset_steps) != int(evaluation["goal_offset_steps"]):
            raise ValueError(f"wrong goal offset for {task}")
        if any(entry.initially_successful for entry in manifest.entries):
            raise ValueError(f"dev cohort for {task} contains an initially successful start")

        expected_dataset_hash = str(state["dataset_hashes"][task])
        dataset_entry = dataset_hashes.get(task, {})
        observed_dataset_hash = (
            dataset_entry.get("sha256") if isinstance(dataset_entry, Mapping) else dataset_entry
        )
        if observed_dataset_hash != expected_dataset_hash:
            raise ValueError(f"scheduler/dataset hash disagreement for {task}")
        for structure in config["structures"]:
            name = str(structure["name"])
            job_id = f"{task}_{name}"
            job = state["jobs"][job_id]
            if job.get("status") != "complete":
                raise ValueError(f"training job is not complete: {job_id}")
            identity_path = _resolve(
                config["training"]["identity_pattern"].format(
                    task=task, structure=name
                )
            )
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            checkpoint = _resolve(
                config["training"]["checkpoint_pattern"].format(
                    task=task, structure=name
                )
            )
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                raise FileNotFoundError(f"missing training checkpoint: {checkpoint}")
            checkpoint_sha = _sha256_file(checkpoint)
            if checkpoint_sha != job.get("checkpoint_sha256"):
                raise ValueError(f"checkpoint hash differs from scheduler state: {job_id}")
            if Path(str(job.get("checkpoint"))).resolve() != checkpoint.resolve():
                raise ValueError(f"checkpoint path differs from scheduler state: {job_id}")
            expected = {
                "task": task,
                "structure": name,
                "seed": int(config["seed"]),
                "action_block_steps": int(structure["action_block_steps"]),
                "action_horizon": int(structure["action_horizon"]),
                "stage_a_goal_index": 25 // int(structure["action_block_steps"]),
                "dataset.sha256": expected_dataset_hash,
            }
            actual = {
                "task": identity.get("task"),
                "structure": identity.get("structure"),
                "seed": identity.get("seed"),
                "action_block_steps": identity.get("action_block_steps"),
                "action_horizon": identity.get("action_horizon"),
                "stage_a_goal_index": identity.get("stage_a_goal_index"),
                "dataset.sha256": identity.get("dataset", {}).get("sha256"),
            }
            if actual != expected:
                raise ValueError(f"training identity mismatch for {job_id}: {actual}")
            outputs[(task, name)] = {
                "manifest": manifest,
                "manifest_path": manifest_path,
                "identity": identity,
                "identity_path": identity_path,
                "identity_sha256": _sha256_file(identity_path),
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_sha,
                "dataset_sha256": expected_dataset_hash,
            }
    return state_path, state, outputs


def _code_identity(config_path: Path):
    return {name: _sha256_file(ROOT / name) for name in CODE_FILES} | {
        "config_sha256": _sha256_file(config_path)
    }


def _result_path(config: Mapping[str, Any], task: str, structure: str, spec):
    return _resolve(config["output_root"]) / task / structure / spec["id"] / "result.json"


def _validate_result(payload, *, manifest, expected_metadata, result_path):
    if payload.get("status") != "ok":
        raise ValueError(f"evaluation result status is not ok: {result_path}")
    metadata = payload.get("round5_phase5_expansion_eval")
    if metadata != expected_metadata:
        raise ValueError(f"evaluation identity mismatch: {result_path}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(manifest.entries):
        raise ValueError(f"evaluation episode count mismatch: {result_path}")
    expected_ids = [entry.episode_id for entry in manifest.entries]
    observed_ids = [episode.get("episode_id") for episode in episodes]
    if observed_ids != expected_ids:
        raise ValueError(f"evaluation episode order or identity mismatch: {result_path}")
    successes = [bool(episode["success"]) for episode in episodes]
    observed_rate = float(np.mean(successes))
    if not math.isclose(float(payload.get("success_rate", -1)), observed_rate, abs_tol=1e-12):
        raise ValueError(f"evaluation success rate mismatch: {result_path}")


def validate(args: argparse.Namespace):
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    state_path, state, inputs = _load_inputs(config)
    specs = list(_all_specs(config))
    summary = {
        "status": "ok",
        "config": str(config_path.resolve()),
        "training_state": str(state_path.resolve()),
        "training_jobs_complete": sum(
            1 for job in state["jobs"].values() if job.get("status") == "complete"
        ),
        "cohorts": {
            task: {
                "path": str(_resolve(config["cohorts"][task]["path"]).resolve()),
                "sha256": inputs[(task, str(config["structures"][0]["name"]))][
                    "manifest"
                ].computed_sha256,
                "episodes": len(
                    inputs[(task, str(config["structures"][0]["name"]))]["manifest"].entries
                ),
            }
            for task in config["tasks"]
        },
        "conditions": len(specs),
        "conditions_by_task": {
            task: sum(1 for item_task, _, _ in specs if item_task == task)
            for task in config["tasks"]
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def dry_run(args: argparse.Namespace):
    config = _load_config(_resolve(args.config))
    rows = []
    for task, structure, spec in _all_specs(config):
        rows.append(
            {
                "task": task,
                "structure": structure,
                "condition": spec["condition"],
                "id": spec["id"],
                "result": str(_result_path(config, task, structure, spec)),
            }
        )
    print(json.dumps({"count": len(rows), "conditions": rows}, ensure_ascii=False, indent=2))


def run_worker(args: argparse.Namespace):
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    if args.task not in config["tasks"]:
        raise ValueError(f"task is outside the configured scope: {args.task}")
    structures = {str(value["name"]): value for value in config["structures"]}
    if args.structure not in structures:
        raise ValueError(f"unknown structure: {args.structure}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != str(args.gpu):
        raise EnvironmentError(
            f"CUDA_VISIBLE_DEVICES must be exactly the selected physical GPU {args.gpu}; got {visible!r}"
        )

    _, _, inputs = _load_inputs(config)
    structure_config = structures[args.structure]
    context = inputs[(args.task, args.structure)]
    manifest = context["manifest"]
    evaluation = config["evaluation"]
    all_specs = _condition_specs(config, args.task, structure_config)
    if args.condition:
        selected = [spec for spec in all_specs if spec["id"] == args.condition]
        if not selected:
            raise ValueError(f"condition is not in this matrix: {args.condition}")
    else:
        selected = all_specs

    import torch

    validate_gpu_visibility("cuda")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise EnvironmentError("worker must see exactly one CUDA device")
    free_bytes, _ = torch.cuda.mem_get_info(0)
    free_mib = int(free_bytes // (1024 * 1024))
    minimum_free_mib = int(evaluation["minimum_free_mib"])
    if free_mib < minimum_free_mib:
        raise MemoryError(
            f"GPU {args.gpu} has {free_mib} MiB free, below the {minimum_free_mib} MiB safety margin"
        )

    model, resolved_checkpoint = load_policy_or_model(str(context["checkpoint"]))
    if resolved_checkpoint is not None and Path(resolved_checkpoint).resolve() != context["checkpoint"].resolve():
        raise ValueError("checkpoint loader resolved a different weights file")
    expected_block = int(structure_config["action_block_steps"])
    expected_horizon = int(structure_config["action_horizon"])
    expected_action_dim = BASE_ACTION_DIMS[args.task] * expected_block
    if int(getattr(model, "action_horizon", -1)) != expected_horizon:
        raise ValueError("loaded checkpoint action_horizon differs from its identity")
    if int(getattr(model, "action_dim", -1)) != expected_action_dim:
        raise ValueError("loaded checkpoint action_dim differs from its native block size")

    code_identity = _code_identity(config_path)
    output_root = _resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "status": "worker_started",
                "task": args.task,
                "structure": args.structure,
                "gpu": int(args.gpu),
                "free_mib": free_mib,
                "pending_conditions": len(selected),
                "checkpoint_sha256": context["checkpoint_sha256"],
                "cohort_sha256": manifest.computed_sha256,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for spec in selected:
        result_path = _result_path(config, args.task, args.structure, spec)
        expected_metadata = {
            "schema_version": 1,
            "stage": str(config["stage"]),
            "task": args.task,
            "structure": args.structure,
            "training_identity_path": str(context["identity_path"].resolve()),
            "training_identity_sha256": context["identity_sha256"],
            "dataset_sha256": context["dataset_sha256"],
            "checkpoint": str(context["checkpoint"].resolve()),
            "checkpoint_sha256": context["checkpoint_sha256"],
            "cohort_path": str(context["manifest_path"].resolve()),
            "cohort_id": manifest.cohort_id,
            "cohort_sha256": manifest.computed_sha256,
            "condition": spec["condition"],
            "condition_sha256": spec["condition_sha256"],
            "code_identity": code_identity,
            "gpu": int(args.gpu),
            "cuda_visible_devices": visible,
        }
        if result_path.is_file():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            _validate_result(
                payload,
                manifest=manifest,
                expected_metadata=expected_metadata,
                result_path=result_path,
            )
            print(json.dumps({"status": "resumed", "result": str(result_path)}), flush=True)
            continue
        if result_path.parent.exists() and any(result_path.parent.iterdir()):
            raise FileExistsError(
                f"condition has partial artifacts without a valid result: {result_path.parent}"
            )

        overrides = (
            "output.save_video=false",
            f"seed={int(evaluation['seed'])}",
            f"plan_config.horizon={expected_horizon}",
            f"plan_config.receding_horizon={expected_horizon}",
            f"plan_config.action_block={expected_block}",
        )
        cfg = compose_eval_config(args.task, overrides)
        mode = str(spec["mode"])
        identity = EvaluationIdentity(
            entrypoint="round5_phase5_expansion_eval",
            policy_kind="round4_shared_dit",
            checkpoint=str(context["checkpoint"].resolve()),
            epoch=10,
            stage=mode,
        )
        payload = run_round4_evaluation(
            cfg,
            task=args.task,
            policy_or_model=model,
            mode=mode,
            identity=identity,
            manifest=manifest,
            output_dir=result_path.parent,
            trace_output_dir=result_path.parent / "trace",
            device="cuda",
            trace=bool(evaluation["trace"]),
            candidate_count=int(evaluation["p3_candidate_count"]),
            flow_steps=int(evaluation["action_flow_steps"]),
            action_flow_steps=int(evaluation["action_flow_steps"]),
            solver_batch_size=int(evaluation["solver_batch_size"]),
            candidate_batch_size=int(evaluation["candidate_batch_size"]),
            action_flow_integrator=str(evaluation["action_flow_integrator"]),
            action_bound_mode=str(evaluation["action_bound_mode"]),
            allowed_protocol_variants=(manifest.protocol_variant,),
            policy_seed=int(evaluation["seed"]),
            selection_rule=spec["selection_rule"],
            execute_steps=int(spec["execute_steps"]),
            score_horizon_blocks=spec["score_horizon_blocks"],
            allow_action_structure_override=True,
            video_slots=0,
        )
        if payload.get("status") != "ok":
            raise RuntimeError(f"evaluation did not produce a result: {payload.get('status')}")
        payload["round5_phase5_expansion_eval"] = expected_metadata
        _atomic_json(result_path, payload)
        _validate_result(
            payload,
            manifest=manifest,
            expected_metadata=expected_metadata,
            result_path=result_path,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "task": args.task,
                    "structure": args.structure,
                    "condition": spec["id"],
                    "success_rate": payload["success_rate"],
                    "seconds": payload["evaluation_seconds"],
                    "result": str(result_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        torch.cuda.empty_cache()


def _wilson(successes: int, total: int):
    if total <= 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def analyze(args: argparse.Namespace):
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    _, _, inputs = _load_inputs(config)
    rows = []
    for task, structure, spec in _all_specs(config):
        result_path = _result_path(config, task, structure, spec)
        if not result_path.is_file():
            raise FileNotFoundError(f"missing evaluation result: {result_path}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        context = inputs[(task, structure)]
        expected_metadata = {
            "schema_version": 1,
            "stage": str(config["stage"]),
            "task": task,
            "structure": structure,
            "training_identity_path": str(context["identity_path"].resolve()),
            "training_identity_sha256": context["identity_sha256"],
            "dataset_sha256": context["dataset_sha256"],
            "checkpoint": str(context["checkpoint"].resolve()),
            "checkpoint_sha256": context["checkpoint_sha256"],
            "cohort_path": str(context["manifest_path"].resolve()),
            "cohort_id": context["manifest"].cohort_id,
            "cohort_sha256": context["manifest"].computed_sha256,
            "condition": spec["condition"],
            "condition_sha256": spec["condition_sha256"],
            "code_identity": _code_identity(config_path),
            "gpu": payload.get("round5_phase5_expansion_eval", {}).get("gpu"),
            "cuda_visible_devices": payload.get("round5_phase5_expansion_eval", {}).get(
                "cuda_visible_devices"
            ),
        }
        stored = payload.get("round5_phase5_expansion_eval", {})
        expected_metadata["gpu"] = stored.get("gpu")
        expected_metadata["cuda_visible_devices"] = stored.get("cuda_visible_devices")
        _validate_result(
            payload,
            manifest=context["manifest"],
            expected_metadata=expected_metadata,
            result_path=result_path,
        )
        episodes = payload["episodes"]
        successes = int(sum(bool(episode["success"]) for episode in episodes))
        rows.append(
            {
                "task": task,
                "structure": structure,
                "method": spec["method"],
                "execute_steps": spec["execute_steps"],
                "score_steps_env": spec["score_steps_env"],
                "successes": successes,
                "episodes": len(episodes),
                "success_rate": successes / len(episodes),
                "wilson_95_ci": _wilson(successes, len(episodes)),
                "evaluation_seconds": payload["evaluation_seconds"],
                "checkpoint_sha256": context["checkpoint_sha256"],
                "cohort_sha256": context["manifest"].computed_sha256,
                "result": str(result_path.resolve()),
            }
        )

    output_root = _resolve(config["output_root"])
    analysis_root = output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_root / "conditions.csv"
    keys = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        import csv

        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "ok",
        "config": str(config_path.resolve()),
        "stage": config["stage"],
        "tasks": config["tasks"],
        "cohort_sizes": {task: int(config["evaluation"]["num_eval"]) for task in config["tasks"]},
        "conditions": len(rows),
        "rows": rows,
    }
    _atomic_json(analysis_root / "analysis.json", summary)
    lines = [
        "# Round 5 Phase 5 原生动作结构开发集评测",
        "",
        "本结果覆盖 PushT 的 round3_revised 50 起点和 Reacher 的 legacy 50 起点。两项任务使用各自冻结的开发集，成功率按任务分别报告，不合并作四任务宏平均，也不据此完成全计划的结构选择。",
        "",
        "所有结构使用 P3（S=2，N=64）、P0 和 Random-64；统一物理动作裁剪。新起点复核、近目标训练对照及后续推理策略选择不在本轮运行范围内。",
        "",
        "| task | structure | method | execute | score | success | Wilson 95% CI | eval seconds |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ci = row["wilson_95_ci"]
        score = "—" if row["score_steps_env"] is None else str(row["score_steps_env"])
        lines.append(
            f"| {row['task']} | {row['structure']} | {row['method']} | "
            f"{row['execute_steps']} | {score} | {row['successes']}/{row['episodes']} "
            f"({100 * row['success_rate']:.1f}%) | [{100 * ci[0]:.1f}%, {100 * ci[1]:.1f}%] | "
            f"{row['evaluation_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 产物",
            "",
            f"- 条件汇总：`{csv_path}`",
            f"- 机器可读分析：`{analysis_root / 'analysis.json'}`",
            "- 每个评测单元的 cohort、checkpoint、配置和代码身份见其 `result.json`。",
            "",
        ]
    )
    (analysis_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", "conditions": len(rows), "report": str(analysis_root / "report.md")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("validate", "verify training checkpoints, identities, and dev cohorts"),
        ("dry-run", "print the full native action structure evaluation matrix"),
        ("analyze", "validate all results and write the summary report"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("--config", default=str(DEFAULT_CONFIG))
    worker = subparsers.add_parser("run-worker", help=argparse.SUPPRESS)
    worker.add_argument("--config", default=str(DEFAULT_CONFIG))
    worker.add_argument("--task", choices=("pusht", "reacher"), required=True)
    worker.add_argument("--structure", required=True)
    worker.add_argument("--gpu", type=int, choices=range(8), required=True)
    worker.add_argument("--condition", help="run one exact condition from the matrix")
    worker.set_defaults()
    args = parser.parse_args()
    if args.command == "validate":
        validate(args)
    elif args.command == "dry-run":
        dry_run(args)
    elif args.command == "run-worker":
        run_worker(args)
    elif args.command == "analyze":
        analyze(args)


if __name__ == "__main__":
    main()
