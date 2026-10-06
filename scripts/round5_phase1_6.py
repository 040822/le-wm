#!/usr/bin/env python3
"""Prepare, calibrate, run, and summarize Round 5 Phase1.6.

GPU evaluation commands must be launched with an explicit single physical GPU,
for example ``CUDA_VISIBLE_DEVICES=3 python scripts/round5_phase1_6.py ... --gpu 3``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.checkpoint import load_policy_or_model
from source.common.eval import EvaluationIdentity, compose_eval_config, get_dataset
from source.common.round3_phase1 import CohortManifest
from source.common.round4_eval import run_round4_evaluation
from source.common.round5_phase1_6 import (
    atomic_json,
    build_confirmation_cohort,
    canonical_json,
    paired_state_metrics,
    stable_hash,
)


DEFAULT_CONFIG = ROOT / "config/round5/phase1_6.json"
METHODS = {
    "p0_s2": {
        "mode": "P0", "flow_steps": 2, "candidate_count": 1,
        "guidance": "none", "action_bound_mode": "clip",
        "cem_protocol": "not_applicable",
    },
    "random64_s2": {
        "mode": "P3", "flow_steps": 2, "candidate_count": 64,
        "selection_rule": "random", "guidance": "none",
        "action_bound_mode": "clip", "cem_protocol": "not_applicable",
    },
    "p3_s2": {
        "mode": "P3", "flow_steps": 2, "candidate_count": 64,
        "guidance": "none", "action_bound_mode": "clip",
        "cem_protocol": "not_applicable",
    },
    "p0_po2_s2": {
        "mode": "P0", "flow_steps": 2, "candidate_count": 1,
        "guidance": "post_opt", "inner_steps": 2, "step_size": 0.01,
        "max_rms_offset": 0.2, "action_bound_mode": "clip",
        "cem_protocol": "not_applicable",
    },
    "p0_gf2_s2": {
        "mode": "P0", "flow_steps": 2, "candidate_count": 1,
        "guidance": "guided_flow", "inner_steps": 1,
        "guidance_last_steps": 2, "step_size": 0.01,
        "max_rms_offset": 0.2, "action_bound_mode": "clip",
        "cem_protocol": "not_applicable",
    },
    "p3_po5_s2": {
        "mode": "P3", "flow_steps": 2, "candidate_count": 64,
        "guidance": "post_opt", "inner_steps": 5, "step_size": 0.01,
        "max_rms_offset": 0.2, "action_bound_mode": "clip",
        "cem_protocol": "not_applicable",
    },
    "p1_cem_300x30": {
        "mode": "P1", "flow_steps": None, "candidate_count": 300,
        "guidance": "none", "cem_samples": 300, "cem_iterations": 30,
        "cem_topk": 30, "action_bound_mode": "candidate_clip",
        "cem_protocol": "cem-clip",
    },
    "p2_cem_300x30": {
        "mode": "P2", "flow_steps": 1, "candidate_count": 300,
        "guidance": "none", "cem_samples": 300, "cem_iterations": 30,
        "cem_topk": 30, "action_bound_mode": "candidate_clip",
        "cem_protocol": "cem-clip",
    },
    "p2_small_64x5": {
        "mode": "P2", "flow_steps": 1, "candidate_count": 64,
        "guidance": "none", "cem_samples": 64, "cem_iterations": 5,
        "cem_topk": 7, "action_bound_mode": "candidate_clip",
        "cem_protocol": "cem-clip",
    },
}


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _load_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("invalid Round5 Phase1.6 config")
    return value


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint(config: Mapping[str, Any], task: str) -> tuple[Path, str]:
    path = _resolve(config["training"]["checkpoints"][task])
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = _sha256_file(path)
    expected = config["training"].get("checkpoint_sha256", {}).get(task)
    if expected is not None and digest != expected:
        raise ValueError(f"checkpoint hash mismatch for {task}: {digest} != {expected}")
    return path, digest


def _task_cfg(task: str, config: Mapping[str, Any], *, seed: int, count: int, device: str = "cpu", method: Mapping[str, Any] | None = None):
    method = dict(method or {})
    samples = int(method.get("cem_samples", 300))
    iterations = int(method.get("cem_iterations", 30))
    topk = int(method.get("cem_topk", 30))
    return compose_eval_config(
        task,
        overrides=[
            f"seed={int(seed)}",
            f"eval.num_eval={int(count)}",
            f"eval.goal_offset_steps={int(config['evaluation']['goal_offset_steps'])}",
            f"eval.eval_budget={int(config['evaluation']['eval_budget'])}",
            f"plan_config.horizon={int(config['evaluation']['horizon'])}",
            f"plan_config.receding_horizon={int(config['evaluation']['receding_horizon'])}",
            f"plan_config.action_block={int(config['evaluation']['action_block'])}",
            "output.save_video=false",
            f"solver.device={device}",
            f"solver.num_samples={samples}",
            f"solver.topk={topk}",
            f"solver.n_steps={iterations}",
            "solver.var_scale=1.0",
        ],
    )


def _historical_episode_ids(task: str, config: Mapping[str, Any]) -> list[Any]:
    paths: list[Path] = []
    source_config_path = _resolve(config["source_config"])
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    phase15 = source_config.get("cohort", {}).get("paths", {}).get(task)
    if phase15:
        paths.append(_resolve(phase15))

    artifact_path = ROOT / "config/round4/cohort_artifacts.json"
    if artifact_path.is_file():
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        for record in artifact.get("tasks", {}).get(task, {}).values():
            if isinstance(record, Mapping) and record.get("path"):
                paths.append((artifact_path.parent / str(record["path"])).resolve())

    seen: set[str] = set()
    ids: list[Any] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"registered historical cohort is missing: {path}")
        manifest = CohortManifest.load(path)
        if manifest.task != task:
            raise ValueError(f"historical cohort task mismatch: {path}")
        for entry in manifest.entries:
            key = json.dumps(entry.episode_id, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                seen.add(key)
                ids.append(entry.episode_id)
    if not ids:
        raise ValueError(f"no registered historical episode IDs found for {task}")
    return ids


def validate(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    errors = []
    checkpoints = {}
    for task in config["evaluation"]["tasks"]:
        try:
            path, digest = _checkpoint(config, task)
            checkpoints[task] = {"path": str(path), "sha256": digest}
        except Exception as exc:
            errors.append(f"{task} checkpoint: {exc}")
    if config["protocol"] != "round3_revised":
        errors.append("protocol must be round3_revised")
    if config["evaluation"]["action_bound_mode"] != "clip":
        errors.append("Phase1.6 non-CEM actions must use physical-range clip")
    if len(config["evaluation"]["seeds"]) < 2:
        errors.append("confirmation evaluation requires at least two inference seeds")
    if errors:
        raise RuntimeError("\n".join(errors))
    print(json.dumps({"status": "ok", "config": str(_resolve(args.config)), "checkpoints": checkpoints, "methods": list(METHODS)}, ensure_ascii=False, sort_keys=True))


def prepare(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    tasks = config["evaluation"]["tasks"] if args.task == "all" else [args.task]
    cohort_root = _resolve(config["paths"]["output_root"]) / "cohorts"
    for task in tasks:
        cfg = _task_cfg(task, config, seed=int(config["cohorts"]["new_seed"]), count=int(config["cohorts"]["new_count_per_task"]))
        dataset = get_dataset(cfg, cfg.eval.dataset_name)
        excluded = _historical_episode_ids(task, config)
        manifest = build_confirmation_cohort(
            dataset,
            task=task,
            count=int(config["cohorts"]["new_count_per_task"]),
            seed=int(config["cohorts"]["new_seed"]),
            goal_offset_steps=int(config["cohorts"]["goal_offset_steps"]),
            excluded_episode_ids=excluded,
        )
        target = cohort_root / task / "confirmation.json"
        manifest.save(target)
        atomic_json(target.with_name("receipt.json"), {
            "task": task,
            "manifest": str(target),
            "cohort_id": manifest.cohort_id,
            "cohort_sha256": manifest.computed_sha256,
            "entries": len(manifest.entries),
            "excluded_historical_episodes": len(excluded),
            "eligible_nonhistorical_episodes": manifest.diagnostics.get("eligible_nonhistorical_episode_count"),
            "sampling_rule": dict(manifest.sampling_rule),
        })
        print(json.dumps({"task": task, "cohort": str(target), "entries": len(manifest.entries), "sha256": manifest.computed_sha256, "excluded_historical": len(excluded)}, ensure_ascii=False, sort_keys=True), flush=True)


def _host_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    budget = config["budget"]
    cpus = max(1, os.cpu_count() or 1)
    load1, load5, load15 = os.getloadavg()
    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        name, sep, raw = line.partition(":")
        fields = raw.strip().split()
        if sep and fields and fields[0].isdigit():
            meminfo[name] = int(fields[0]) // 1024
    available = meminfo.get("MemAvailable", 0)
    min_available = int(float(budget["min_host_available_gib"]) * 1024)
    ratio = float(load1) / cpus
    snapshot = {
        "cpus": cpus, "load1": load1, "load5": load5, "load15": load15,
        "load_per_cpu": ratio, "max_load_per_cpu": float(budget["max_load_per_cpu"]),
        "available_mib": available, "required_available_mib": min_available,
    }
    if ratio > float(budget["max_load_per_cpu"]):
        raise RuntimeError(f"CPU load guard blocks worker startup: {snapshot}")
    if available < min_available:
        raise RuntimeError(f"host memory guard blocks worker startup: {snapshot}")
    return snapshot


def _gpu_preflight(
    gpu: str,
    config: Mapping[str, Any],
    *,
    minimum_free_mib: int = 3500,
) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(gpu):
        raise RuntimeError(f"launch must set CUDA_VISIBLE_DEVICES={gpu}; got {visible!r}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    query = subprocess.run(
        ["nvidia-smi", "--id", str(gpu), "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        cwd=ROOT, env=env, text=True, capture_output=True, check=True,
    )
    values = [item.strip() for item in query.stdout.strip().split(",")]
    if len(values) != 6:
        raise RuntimeError(f"unexpected nvidia-smi output: {query.stdout!r}")
    snapshot = {
        "gpu": int(values[0]), "name": values[1], "total_mib": int(values[2]),
        "used_mib": int(values[3]), "free_mib": int(values[4]),
        "utilization_percent": int(values[5]), "required_free_mib": int(minimum_free_mib),
    }
    if snapshot["free_mib"] < int(minimum_free_mib):
        raise RuntimeError(f"GPU{gpu} has insufficient free VRAM: {snapshot}")
    if int(gpu) >= 4:
        if not _resolve("outputs/round5/phase1_6_seed3072/analysis/eval_memory_budget.json").is_file():
            raise RuntimeError("GPU4-7 require completed Phase1.6 low-VRAM calibration first")
        memory_record = json.loads(_resolve("outputs/round5/phase1_6_seed3072/analysis/eval_memory_budget.json").read_text(encoding="utf-8"))
        measured_gib = float(memory_record.get("maximum_peak_allocated_gib", float("inf")))
        cap = float(config["budget"]["gpu4_7_max_task_memory_gib"])
        if measured_gib > cap:
            raise RuntimeError(f"GPU4-7 low-VRAM guard: calibrated task peak {measured_gib:.2f} GiB > {cap:.2f} GiB")
        snapshot["calibrated_task_peak_gib"] = measured_gib
    return snapshot


def _load_manifest(config: Mapping[str, Any], task: str) -> CohortManifest:
    path = _resolve(config["paths"]["output_root"]) / "cohorts" / task / "confirmation.json"
    if not path.is_file():
        raise FileNotFoundError(f"run `prepare --task {task}` first: {path}")
    manifest = CohortManifest.load(path)
    if len(manifest.entries) != int(config["cohorts"]["new_count_per_task"]):
        raise ValueError(f"wrong cohort size in {path}")
    if manifest.computed_sha256 != manifest.cohort_sha256:
        raise ValueError(f"cohort hash mismatch: {path}")
    return manifest


def _code_identity() -> dict[str, Any]:
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "git_revision": revision,
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "helper_sha256": _sha256_file(ROOT / "source/common/round5_phase1_6.py"),
        "round4_eval_sha256": _sha256_file(ROOT / "source/common/round4_eval.py"),
        "round4_policy_sha256": _sha256_file(ROOT / "source/policy/round4.py"),
    }


def _run_condition(
    *, config: Mapping[str, Any], task: str, method_id: str, seed: int,
    manifest: CohortManifest, checkpoint: Path, checkpoint_sha256: str,
    model: Any, dataset: Any, device: str, output_dir: Path,
    config_path: Path, calibration: bool = False,
) -> dict[str, Any]:
    from source.common.round5_phase1_5 import condition_lock

    method = dict(METHODS[method_id])
    identity = {
        "experiment": "round5_phase1_6",
        "task": task,
        "method": method_id,
        "method_spec": method,
        "evaluation_seed": int(seed),
        "cohort_sha256": manifest.computed_sha256,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": _sha256_file(config_path),
        "code": _code_identity(),
    }
    result_path = output_dir / "result.json"
    complete_path = output_dir / "complete.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if complete_path.is_file() and result_path.is_file():
        old = json.loads(result_path.read_text(encoding="utf-8"))
        if old.get("phase1_6_identity") == identity and old.get("status") == "ok":
            return {"task": task, "method": method_id, "seed": seed, "status": "reused", "path": str(result_path), "success_rate": old.get("success_rate")}
        previous_identity = old.get("phase1_6_identity", {})
        previous_code = dict(previous_identity.get("code", {}))
        current_code = dict(identity.get("code", {}))
        previous_code.pop("script_sha256", None)
        current_code.pop("script_sha256", None)
        comparable_previous = {key: value for key, value in previous_identity.items() if key != "code"}
        comparable_current = {key: value for key, value in identity.items() if key != "code"}
        if (
            old.get("status") == "ok"
            and comparable_previous == comparable_current
            and previous_code == current_code
        ):
            return {
                "task": task, "method": method_id, "seed": seed,
                "status": "reused_prior_runner_revision",
                "path": str(result_path), "success_rate": old.get("success_rate"),
                "executed_script_sha256": previous_identity.get("code", {}).get("script_sha256"),
                "current_script_sha256": identity.get("code", {}).get("script_sha256"),
            }

    cfg = _task_cfg(task, config, seed=seed, count=len(manifest.entries), device=device, method=method)
    mode = str(method["mode"])
    spec = dict(method)
    cfg_hash = stable_hash({"config": config, "method": spec, "task": task, "seed": seed})
    eval_identity = EvaluationIdentity(
        entrypoint="round5_phase1_6_calibration" if calibration else "round5_phase1_6",
                policy_kind="round4_shared_dit",
                checkpoint=str(checkpoint.resolve()),
        epoch=(int(config["training"]["epoch"]) if checkpoint.resolve() == _resolve(config["training"]["checkpoints"][task]).resolve() else None),
        stage=mode,
    )
    peak_bytes = 0
    import torch
    local_device = torch.device(device)
    if local_device.type == "cuda":
        torch.cuda.synchronize(local_device)
        torch.cuda.reset_peak_memory_stats(local_device)
    with condition_lock(result_path):
        try:
            result = run_round4_evaluation(
                cfg,
                task=task,
                policy_or_model=model,
                mode=mode,
                identity=eval_identity,
                manifest=manifest,
                output_dir=output_dir,
                trace_output_dir=output_dir / "trace",
                dataset=dataset,
                device=device,
                trace=True,
                candidate_count=int(method["candidate_count"]),
                flow_steps=int(method.get("flow_steps") or 16),
                action_flow_steps=method.get("flow_steps"),
                solver_batch_size=int(config["evaluation"]["solver_batch_size"]),
                candidate_batch_size=int(config["evaluation"]["candidate_batch_size"]),
                action_flow_integrator=str(config["evaluation"]["action_flow_integrator"]),
                action_bound_mode=str(method["action_bound_mode"]),
                cem_protocol=str(method["cem_protocol"]),
                guidance_mode=str(method.get("guidance", "none")),
                guidance_step_size=float(method.get("step_size", 0.01)),
                guidance_last_steps=int(method.get("guidance_last_steps", 5)),
                guidance_inner_steps=int(method.get("inner_steps", 5)),
                guidance_max_rms_offset=float(method.get("max_rms_offset", 0.2)),
                proposal_chunk_size=int(config["evaluation"]["proposal_chunk_size"]),
                allowed_protocol_variants=(str(config["protocol"]),),
                allow_variable_candidate_count=True,
                allow_solver_config_override=True,
                allow_evaluation_seed_override=True,
                allow_cohort_seed_mismatch=True,
                selection_rule=method.get("selection_rule"),
            )
            if local_device.type == "cuda":
                torch.cuda.synchronize(local_device)
                peak_bytes = int(torch.cuda.max_memory_allocated(local_device))
            result = dict(result)
            result["phase1_6_identity"] = identity
            result["phase1_6_condition"] = {
                "method": method_id,
                "method_spec": method,
                "seed": int(seed),
                "config_hash": cfg_hash,
                "checkpoint_sha256": checkpoint_sha256,
                "cohort_sha256": manifest.computed_sha256,
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "peak_memory_bytes": peak_bytes or None,
            }
            atomic_json(result_path, result)
            atomic_json(complete_path, {
                "schema_version": "round5_phase1_6_result_v1",
                "status": "completed",
                "result": str(result_path),
                "result_sha256": _sha256_file(result_path),
                "identity_sha256": stable_hash(identity),
            })
        except Exception as exc:
            atomic_json(output_dir / "failure.json", {
                "status": "failed", "error_type": type(exc).__name__,
                "error": str(exc), "traceback": traceback.format_exc(),
                "identity": identity,
            })
            raise
    return {"task": task, "method": method_id, "seed": seed, "status": "completed", "path": str(result_path), "success_rate": result.get("success_rate"), "peak_memory_bytes": peak_bytes or None}


def _load_runtime(task: str, config: Mapping[str, Any]):
    checkpoint, digest = _checkpoint(config, task)
    model, resolved = load_policy_or_model(str(checkpoint))
    if resolved is not None and Path(resolved).resolve() != checkpoint.resolve():
        raise ValueError(f"checkpoint resolver changed requested path: {checkpoint}")
    return checkpoint, digest, model


def _persist_memory_budget(config: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
    root = _resolve(config["paths"]["output_root"])
    target = root / "analysis/eval_memory_budget.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(target.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        existing = []
        if target.is_file():
            current = json.loads(target.read_text(encoding="utf-8"))
            existing = list(current.get("calibration_results", ()))

        # Several task/seed workers can finish together. Merge under the lock so
        # an atomic replace cannot discard another worker's memory receipt.
        merged: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in [*existing, *(dict(item) for item in records)]:
            key = (
                row.get("path"), row.get("task"), row.get("method"),
                row.get("seed"), row.get("status"),
            )
            merged[key] = row
        merged_records = list(merged.values())
        peaks = [
            int(row["peak_memory_bytes"])
            for row in merged_records
            if row.get("peak_memory_bytes")
        ]
        maximum = max(peaks) if peaks else None
        atomic_json(target, {
            "schema_version": "round5_phase1_6_memory_budget_v1",
            "calibration_results": merged_records,
            "maximum_peak_allocated_bytes": maximum,
            "maximum_peak_allocated_gib": (maximum / (1024 ** 3)) if maximum else None,
            "gpu4_7_cap_gib": float(config["budget"]["gpu4_7_max_task_memory_gib"]),
            "eligible_for_gpu4_7": bool(maximum) and maximum / (1024 ** 3) <= float(config["budget"]["gpu4_7_max_task_memory_gib"]),
        })


def calibrate(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    task = args.task
    if task == "all":
        task = "reacher"
    host = _host_preflight(config)
    gpu = str(args.gpu)
    gpu_info = _gpu_preflight(gpu, config, minimum_free_mib=int(args.min_free_mib))
    print(json.dumps({"host_preflight": host, "gpu_preflight": gpu_info}, sort_keys=True), flush=True)
    manifest = _load_manifest(config, task)
    selected_entries = tuple(manifest.entries[:5])
    selected_ids = tuple(entry.episode_id for entry in selected_entries)
    selected_keys = {
        json.dumps(value, ensure_ascii=False, sort_keys=True) for value in selected_ids
    }
    calibration_split = {
        key: tuple(
            value
            for value in values
            if json.dumps(value, ensure_ascii=False, sort_keys=True) not in selected_keys
        )
        for key, values in manifest.episode_split.items()
    }
    calibration_split["custom"] = selected_ids
    small = replace(
        manifest,
        cohort_id=f"{manifest.cohort_id}_calibration5",
        cohort_kind="custom",
        entries=selected_entries,
        episode_split=calibration_split,
        cohort_sha256=None,
    )
    checkpoint, checkpoint_hash, model = _load_runtime(task, config)
    base_cfg = _task_cfg(task, config, seed=int(args.seed), count=5, device="cuda")
    dataset = get_dataset(base_cfg, base_cfg.eval.dataset_name)
    methods = args.method or ["p0_s2", "p3_s2", "p1_cem_300x30"]
    result_root = _resolve(config["paths"]["output_root"]) / "calibration" / task
    records = []
    for method in methods:
        record = _run_condition(
            config=config, task=task, method_id=method, seed=int(args.seed),
            manifest=small, checkpoint=checkpoint, checkpoint_sha256=checkpoint_hash,
            model=model, dataset=dataset, device="cuda",
            output_dir=result_root / method / f"seed_{args.seed}",
            config_path=_resolve(args.config), calibration=True,
        )
        records.append(record)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
    _persist_memory_budget(config, records)
    atomic_json(result_root / "calibration_summary.json", {
        "task": task, "episodes_per_method": 5, "gpu": gpu,
        "methods": records,
    })


def run(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    task = args.task
    if task == "all":
        raise ValueError("run requires one --task so a worker holds one task model and dataset")
    host = _host_preflight(config)
    gpu_info = _gpu_preflight(str(args.gpu), config, minimum_free_mib=int(args.min_free_mib))
    print(json.dumps({"host_preflight": host, "gpu_preflight": gpu_info}, sort_keys=True), flush=True)
    manifest = _load_manifest(config, task)
    checkpoint, checkpoint_hash, model = _load_runtime(task, config)
    if args.checkpoint:
        if not args.result_tag:
            raise ValueError("--checkpoint requires a distinct --result-tag to preserve frozen-model results")
        from source.common.checkpoint import load_initial_model_state

        checkpoint = _resolve(args.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_hash = _sha256_file(checkpoint)
        model.load_state_dict(load_initial_model_state(checkpoint), strict=True)
    base_cfg = _task_cfg(task, config, seed=int(args.seed), count=len(manifest.entries), device="cuda")
    dataset = get_dataset(base_cfg, base_cfg.eval.dataset_name)
    methods = args.method or list(METHODS)
    result_root = _resolve(args.output_root or config["paths"]["output_root"])
    records = []
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {list(METHODS)}")
        seed_leaf = f"seed_{int(args.seed)}"
        if args.result_tag:
            seed_leaf += f"_{args.result_tag}"
        target = result_root / "conditions" / task / method / seed_leaf
        record = _run_condition(
            config=config, task=task, method_id=method, seed=int(args.seed),
            manifest=manifest, checkpoint=checkpoint, checkpoint_sha256=checkpoint_hash,
            model=model, dataset=dataset, device="cuda", output_dir=target,
            config_path=_resolve(args.config),
        )
        records.append(record)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
    memory_path = result_root / "analysis" / "eval_memory_budget.json"
    existing_memory = []
    if memory_path.is_file():
        existing_memory = json.loads(memory_path.read_text(encoding="utf-8")).get("calibration_results", [])
    _persist_memory_budget(config, [*existing_memory, *records])
    atomic_json(result_root / "analysis" / f"worker_{task}_seed_{args.seed}_gpu_{args.gpu}.json", {
        "gpu": int(args.gpu), "task": task, "seed": int(args.seed), "conditions": records,
    })


def _training_exclusions(task: str, config: Mapping[str, Any]) -> list[Any]:
    values = list(_historical_episode_ids(task, config))
    manifest = _load_manifest(config, task)
    values.extend(entry.episode_id for entry in manifest.entries)
    return values


def prepare_train_cache(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    if args.task not in config["short_train"]["tasks"]:
        raise ValueError(f"short-train cache is only planned for {config['short_train']['tasks']}")
    if args.gpu is None or int(args.gpu) not in range(4):
        raise ValueError("short-train cache encoding requires --gpu 0-3")
    host = _host_preflight(config)
    gpu = _gpu_preflight(str(args.gpu), config, minimum_free_mib=int(args.min_free_mib))
    print(json.dumps({"host_preflight": host, "gpu_preflight": gpu}, sort_keys=True), flush=True)
    checkpoint, checkpoint_hash = _checkpoint(config, args.task)
    root = _resolve(config["paths"]["output_root"]) / "short_train" / args.task / "cache"
    from source.common.round5_phase1_6_train import prepare_short_train_cache

    record = prepare_short_train_cache(
        task=args.task,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        excluded_episode_ids=_training_exclusions(args.task, config),
        output_dir=root,
        device="cuda",
        seed=int(config["short_train"]["seed"]),
        trajectories=int(config["short_train"]["max_trajectories"]),
        windows_per_trajectory=int(config["short_train"]["max_windows_per_trajectory"]),
    )
    print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)


def short_train(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    if args.task not in config["short_train"]["tasks"]:
        raise ValueError(f"short training is only planned for {config['short_train']['tasks']}")
    if args.arm not in config["short_train"]["arms"]:
        raise ValueError(f"unknown training arm {args.arm!r}")
    if args.gpu is None or int(args.gpu) not in range(4):
        raise ValueError("short training is restricted to GPU0-3")
    host = _host_preflight(config)
    gpu = _gpu_preflight(str(args.gpu), config, minimum_free_mib=int(args.min_free_mib))
    if gpu["free_mib"] < int(config["budget"]["training_memory_margin_gib"] * 1024):
        raise RuntimeError(f"training requires at least {config['budget']['training_memory_margin_gib']} GiB free VRAM: {gpu}")
    print(json.dumps({"host_preflight": host, "gpu_preflight": gpu}, sort_keys=True), flush=True)
    checkpoint, checkpoint_hash = _checkpoint(config, args.task)
    root = _resolve(config["paths"]["output_root"]) / "short_train" / args.task
    from source.common.round5_phase1_6_train import run_short_train

    record = run_short_train(
        task=args.task,
        arm=args.arm,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        cache_dir=root / "cache",
        output_dir=root / args.arm,
        device="cuda",
        updates=int(args.updates or config["short_train"]["updates"]),
        batch_size=int(config["short_train"]["batch_size"]),
        learning_rate=float(config["short_train"]["learning_rate"]),
        weight_decay=float(config["short_train"]["weight_decay"]),
        gradient_clip=float(config["short_train"]["gradient_clip"]),
        resume=bool(args.resume),
    )
    print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)


def analyze(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    output_root = _resolve(args.output_root or config["paths"]["output_root"])
    rows = []
    by_task_method: dict[str, dict[str, dict[str, list[float]]]] = {}
    for task in config["evaluation"]["tasks"]:
        by_task_method[task] = {}
        for method in METHODS:
            state_values: dict[str, list[float]] = {}
            for seed in config["evaluation"]["seeds"]:
                path = output_root / "conditions" / task / method / f"seed_{seed}" / "result.json"
                if not path.is_file():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") != "ok":
                    continue
                for episode in payload.get("episodes", []):
                    key = f"{episode.get('episode_id')}:{episode.get('start_step')}"
                    state_values.setdefault(key, []).append(float(bool(episode.get("success"))))
                rows.append({
                    "task": task, "method": method, "seed": int(seed),
                    "episodes": len(payload.get("episodes", [])),
                    "success_rate": payload.get("success_rate"),
                    "evaluation_seconds": payload.get("evaluation_seconds"),
                    "planning_median_seconds": payload.get("round4_planning", {}).get("planning_median_seconds"),
                    "peak_memory_bytes": payload.get("phase1_6_condition", {}).get("peak_memory_bytes"),
                })
            by_task_method[task][method] = state_values
    effects = {}
    comparisons = (("p3_s2", "random64_s2"), ("p0_po2_s2", "p0_s2"), ("p0_gf2_s2", "p0_po2_s2"), ("p3_po5_s2", "p3_s2"))
    for task, method_rows in by_task_method.items():
        effects[task] = {}
        for left, right in comparisons:
            effects[task][f"{left}_minus_{right}"] = paired_state_metrics(
                method_rows.get(left, {}), method_rows.get(right, {}),
                samples=int(config["mechanism"]["bootstrap_samples"]),
                seed=int(config["cohorts"]["new_seed"]),
            )
    payload = {
        "schema_version": "round5_phase1_6_analysis_v1",
        "config": str(_resolve(args.config)),
        "output_root": str(output_root),
        "completed_condition_rows": rows,
        "paired_success_effects": effects,
        "missing_expected_conditions": [
            {"task": task, "method": method, "seed": seed}
            for task in config["evaluation"]["tasks"]
            for method in METHODS
            for seed in config["evaluation"]["seeds"]
            if not (output_root / "conditions" / task / method / f"seed_{seed}" / "result.json").is_file()
        ],
    }
    target = output_root / "analysis/summary.json"
    atomic_json(target, payload)
    print(json.dumps({"analysis": str(target), "completed": len(rows), "missing": len(payload["missing_expected_conditions"])}, ensure_ascii=False, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "prepare", "prepare-train-cache", "train", "calibrate", "run", "analyze"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root")
    parser.add_argument("--task", choices=("cube", "pusht", "reacher", "tworoom", "all"), default="all")
    parser.add_argument("--method", action="append", choices=tuple(METHODS), help="repeat to run a subset of conditions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, choices=range(8))
    parser.add_argument("--min-free-mib", type=int, default=3500)
    parser.add_argument("--checkpoint", help="optional short-train checkpoint for paired post-training evaluation")
    parser.add_argument("--result-tag", help="required output tag when evaluating a non-frozen checkpoint")
    parser.add_argument("--arm", choices=("continue", "detach", "recorded"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--updates", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    if args.command == "validate":
        validate(args, config)
    elif args.command == "prepare":
        prepare(args, config)
    elif args.command == "prepare-train-cache":
        if args.gpu is None:
            raise ValueError("prepare-train-cache requires --gpu 0-3")
        prepare_train_cache(args, config)
    elif args.command == "train":
        if args.gpu is None or args.arm is None:
            raise ValueError("train requires --gpu 0-3 and --arm")
        short_train(args, config)
    elif args.command == "calibrate":
        if args.gpu is None:
            raise ValueError("calibrate requires --gpu")
        calibrate(args, config)
    elif args.command == "run":
        if args.gpu is None:
            raise ValueError("run requires --gpu")
        run(args, config)
    elif args.command == "analyze":
        analyze(args, config)


if __name__ == "__main__":
    main()
