#!/usr/bin/env python3
"""Run, resume, aggregate, and plot the Round 5 Phase 1 epoch curve."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Reduce allocator fragmentation when a worker shares a GPU with another
# evaluation process.  This does not change the frozen evaluation protocol.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from source.common.checkpoint import load_policy_or_model
from source.common.eval import EvaluationIdentity, compose_eval_config
from source.common.round3_phase1 import CohortManifest
from source.common.round4_eval import run_round4_evaluation, validate_gpu_visibility
from source.common.round5_phase1_epoch_curve import (
    CURVE_EPOCHS,
    CURVE_NEW_EPOCHS,
    CURVE_SCHEMA_VERSION,
    CURVE_TASKS,
    baseline_source_path,
    checkpoint_hashes,
    checkpoint_paths,
    condition_key_from_spec,
    condition_name,
    condition_specs,
    dump_json_atomic,
    job_specs,
    load_epoch_curve_results,
    reference_path,
    result_dir,
    result_path,
    row_from_payload,
    sha256_file,
    validate_declared_checkpoint_hashes,
    validate_result_payload_for_curve,
)


DEFAULT_CONFIG = ROOT / "config" / "round5" / "phase1_epoch_curve.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "round5" / "phase1_epoch_curve_seed3072_legacy"
DEFAULT_REPORT = ROOT / "docs" / "report" / "round5" / "round5_phase1_epoch_curve_report.md"
DEFAULT_MIN_FREE_MIB = 3500
DEFAULT_CLAIM_TTL_SECONDS = 6 * 60 * 60


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"epoch-curve config must be an object: {path}")
    return value


def _gpu(value: str | None) -> str | None:
    if value is None:
        return None
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(values) != 1 or not values[0].isdigit() or int(values[0]) not in range(8):
        raise argparse.ArgumentTypeError("--gpu must select exactly one physical GPU0-7")
    return values[0]


def _gpu_preflight(gpu: str, minimum_free_mib: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [
        "nvidia-smi",
        "--id",
        str(gpu),
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect GPU{gpu} before epoch-curve worker startup") from exc
    line = next((item.strip() for item in completed.stdout.splitlines() if item.strip()), "")
    fields = [item.strip() for item in line.split(",")]
    if len(fields) < 6:
        raise RuntimeError(f"nvidia-smi returned an invalid GPU{gpu} snapshot: {line!r}")
    snapshot = {
        "gpu": int(gpu),
        "reported_index": fields[0],
        "name": fields[1],
        "memory_total_mib": int(fields[2]),
        "memory_used_mib": int(fields[3]),
        "memory_free_mib": int(fields[4]),
        "utilization_percent": int(fields[5]),
        "minimum_free_mib": int(minimum_free_mib),
    }
    if snapshot["memory_free_mib"] < int(minimum_free_mib):
        raise RuntimeError(
            f"GPU{gpu} has only {snapshot['memory_free_mib']} MiB free; "
            f"need at least {minimum_free_mib} MiB"
        )
    print(json.dumps({"gpu_preflight": snapshot}, ensure_ascii=False, sort_keys=True), flush=True)
    return snapshot


def _configure_device(device: str, gpu: str | None, minimum_free_mib: int) -> None:
    if str(device).startswith("cuda"):
        if gpu is None:
            raise ValueError("CUDA execution requires --gpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        validate_gpu_visibility(device)
        _gpu_preflight(gpu, minimum_free_mib)


def _load_manifests(config: Mapping[str, Any]) -> dict[str, CohortManifest]:
    paths = config["cohort"]["legacy"]["paths"]
    manifests = {task: CohortManifest.load(_resolve(paths[task])) for task in CURVE_TASKS}
    for task, manifest in manifests.items():
        if (
            manifest.cohort_kind != "dev"
            or manifest.protocol_variant != "legacy"
            or len(manifest.entries) != 50
        ):
            raise ValueError(f"epoch curve requires canonical legacy_50 dev cohort: {task}")
    return manifests


def _compose(task: str, manifest: CohortManifest, device: str):
    cfg = compose_eval_config(
        task,
        overrides=[
            f"eval.num_eval={len(manifest.entries)}",
            "output.save_video=false",
            f"solver.device={device}",
            "solver.num_samples=300",
            "solver.topk=30",
            "solver.n_steps=30",
            "solver.var_scale=1.0",
        ],
    )
    cfg.solver.device = device
    return cfg


def _condition_for_job(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: job[key]
        for key in (
            "task",
            "mode",
            "cem_protocol",
            "action_flow_steps",
            "action_flow_integrator",
            "guidance",
        )
    }


def _claim_path(output_root: Path, job_id_value: str) -> Path:
    return output_root / "claims" / f"{job_id_value}.claim"


def _claim(output_root: Path, job_id_value: str, ttl_seconds: int) -> bool:
    path = _claim_path(output_root, job_id_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job_id_value,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at": time.time(),
    }
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            stale = time.time() - path.stat().st_mtime > int(ttl_seconds)
        except FileNotFoundError:
            return _claim(output_root, job_id_value, ttl_seconds)
        if not stale:
            return False
        try:
            path.unlink()
        except FileNotFoundError:
            return _claim(output_root, job_id_value, ttl_seconds)
        return _claim(output_root, job_id_value, ttl_seconds)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return True


def _release(output_root: Path, job_id_value: str) -> None:
    try:
        _claim_path(output_root, job_id_value).unlink()
    except FileNotFoundError:
        pass


def _existing_payload(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"result must be a JSON object: {path}")
    return value


def _ensure_target_is_safe(target: Path) -> None:
    if not target.exists():
        return
    leftovers = [item for item in target.iterdir() if item.name != "failure.json"]
    if leftovers and not (target / "result.json").is_file():
        raise FileExistsError(f"condition directory is partially populated: {target}")
    if (target / "failure.json").is_file() and not leftovers:
        (target / "failure.json").unlink()


def _write_failure(target: Path, job: Mapping[str, Any], error: BaseException) -> None:
    dump_json_atomic(
        target / "failure.json",
        {
            "schema_version": CURVE_SCHEMA_VERSION,
            "status": "failed",
            "job_id": job["job_id"],
            "task": job["task"],
            "epoch": job["epoch"],
            "condition": _condition_for_job(job),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        },
    )


def _write_result_metadata(path: Path, *, job: Mapping[str, Any], checkpoint_hash: str) -> dict[str, Any]:
    payload = _existing_payload(path)
    if payload is None:
        raise FileNotFoundError(path)
    payload["checkpoint_sha256"] = checkpoint_hash
    payload["epoch_curve_job_id"] = job["job_id"]
    payload["epoch_curve_source"] = "evaluated"
    dump_json_atomic(path, payload)
    return payload


def _reuse_epoch10(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    root: Path,
    job: Mapping[str, Any],
    manifest: CohortManifest,
    checkpoint: Path,
    checkpoint_hash: str,
) -> None:
    condition = _condition_for_job(job)
    target = reference_path(output_root, str(job["task"]), 10, condition)
    target.parent.mkdir(parents=True, exist_ok=True)
    source = baseline_source_path(config, str(job["task"]), condition, root=root).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash = sha256_file(source)
    existing = _existing_payload(target)
    if existing is not None:
        if (
            existing.get("status") != "ok"
            or str(existing.get("source_result")) != str(source)
            or str(existing.get("source_result_sha256")) != source_hash
        ):
            raise ValueError(f"stale epoch 10 reference: {target}")
        source_payload = _existing_payload(source)
        if source_payload is None:
            raise ValueError(f"missing epoch 10 baseline result: {source}")
        validate_result_payload_for_curve(
            source_payload,
            source,
            manifest=manifest,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_hash,
            epoch=10,
            condition=condition,
        )
        return
    _ensure_target_is_safe(target.parent)
    source_payload = _existing_payload(source)
    if source_payload is None:
        raise ValueError(f"missing epoch 10 baseline result: {source}")
    validate_result_payload_for_curve(
        source_payload,
        source,
        manifest=manifest,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        epoch=10,
        condition=condition,
    )
    dump_json_atomic(
        target,
        {
            "schema_version": CURVE_SCHEMA_VERSION,
            "status": "ok",
            "kind": "epoch10_reference",
            "task": job["task"],
            "epoch": 10,
            "job_id": job["job_id"],
            "condition": condition,
            "source": "round4_phase45_4_legacy",
            "source_result": str(source),
            "source_result_sha256": source_hash,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "cohort_id": manifest.cohort_id,
            "cohort_sha256": manifest.computed_sha256,
        },
    )


def _run_job(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    root: Path,
    job: Mapping[str, Any],
    manifest: CohortManifest,
    checkpoint: Path,
    checkpoint_hash: str,
    device: str,
    model: Any,
    trace: bool,
    smoke: bool = False,
) -> dict[str, Any] | None:
    task = str(job["task"])
    epoch = int(job["epoch"])
    condition = _condition_for_job(job)
    if epoch == 10:
        _reuse_epoch10(
            config=config,
            output_root=output_root,
            root=root,
            job=job,
            manifest=manifest,
            checkpoint=checkpoint,
            checkpoint_hash=checkpoint_hash,
        )
        return None

    target = (
        output_root / "smoke" / task / f"epoch_{epoch:02d}" / condition_name(condition)
        if smoke
        else result_dir(output_root, task, epoch, condition)
    )
    target.mkdir(parents=True, exist_ok=True)
    result = target / "result.json"
    existing = _existing_payload(result)
    if existing is not None:
        validate_result_payload_for_curve(
            existing,
            result,
            manifest=manifest,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_hash,
            epoch=epoch,
            condition=condition,
        )
        return existing
    _ensure_target_is_safe(target)
    cfg = _compose(task, manifest, device)
    mode = str(condition["mode"])
    identity = EvaluationIdentity(
        entrypoint="round5_phase1_epoch_curve",
        policy_kind="round4_shared_dit",
        checkpoint=str(checkpoint),
        epoch=epoch,
        stage=mode,
    )
    payload = run_round4_evaluation(
        cfg,
        task=task,
        policy_or_model=model,
        mode=mode,
        identity=identity,
        manifest=manifest,
        output_dir=target,
        trace_output_dir=target / "trace",
        device=device,
        trace=trace,
        candidate_count=1 if mode == "P0" else 64,
        flow_steps=16,
        action_flow_steps=condition["action_flow_steps"],
        solver_batch_size=1,
        candidate_batch_size=64,
        actor_warm_start_scale=1.0,
        action_flow_integrator=str(condition["action_flow_integrator"]),
        cem_protocol=str(condition["cem_protocol"]),
        allowed_protocol_variants=("legacy",),
    )
    observed = _write_result_metadata(result, job=job, checkpoint_hash=checkpoint_hash)
    validate_result_payload_for_curve(
        observed,
        result,
        manifest=manifest,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        epoch=epoch,
        condition=condition,
    )
    print(json.dumps({"status": "ok", "job_id": job["job_id"], "result": str(result)}, sort_keys=True), flush=True)
    return payload


def _select_jobs(
    config: Mapping[str, Any],
    *,
    task: str,
    epoch: int | None,
    indices: Sequence[int] | None,
    shard_index: int,
    num_shards: int,
) -> list[dict[str, Any]]:
    if task not in CURVE_TASKS:
        raise ValueError(f"unknown task {task!r}")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    conditions = condition_specs(config)
    if indices is not None:
        normalized = [int(index) for index in indices]
        if len(set(normalized)) != len(normalized):
            raise ValueError("--condition-index values must be unique")
        if any(index < 0 or index >= len(conditions) for index in normalized):
            raise ValueError("--condition-index is outside the 34-condition matrix")
        selected_keys = {condition_key_from_spec(conditions[index]) for index in normalized}
    else:
        selected_keys = None
    jobs = []
    for index, job in enumerate(job_specs(config)):
        if str(job["task"]) != task:
            continue
        if epoch is not None and int(job["epoch"]) != int(epoch):
            continue
        if selected_keys is not None and condition_key_from_spec(job) not in selected_keys:
            continue
        # Sharding is over the per-task job stream, not over the global matrix.
        task_index = sum(
            1
            for prior in job_specs(config)[:index]
            if str(prior["task"]) == task
        )
        if task_index % num_shards != shard_index:
            continue
        jobs.append(job)
    return jobs


def run_worker(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    task = str(args.task)
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _configure_device(args.device, args.gpu, int(args.min_free_mib))
    manifests = _load_manifests(config)
    checkpoints = checkpoint_paths(config, root=ROOT)
    hashes = checkpoint_hashes(checkpoints)
    validate_declared_checkpoint_hashes(config, hashes)
    jobs = _select_jobs(
        config,
        task=task,
        epoch=args.epoch,
        indices=args.condition_index,
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
    )
    if args.smoke:
        jobs = [job for job in jobs if int(job["epoch"]) == 1][:1]
    print(json.dumps({"task": task, "jobs": len(jobs), "output_root": str(output_root)}, sort_keys=True), flush=True)
    model = None
    loaded_epoch: int | None = None
    failures = 0
    smoke_manifest = None
    if args.smoke:
        original = manifests[task]
        owner = "dev" if "dev" in original.episode_split else "selected"
        split = dict(original.episode_split)
        split[owner] = (original.entries[0].episode_id,)
        smoke_manifest = replace(
            original,
            cohort_id=f"{original.cohort_id}_smoke_1",
            entries=(original.entries[0],),
            episode_split=split,
            diagnostics={**dict(original.diagnostics), "smoke": True},
            cohort_sha256=None,
        )
    for job in jobs:
        job_id_value = str(job["job_id"])
        if not _claim(output_root, job_id_value, int(args.claim_ttl_seconds)):
            continue
        try:
            epoch = int(job["epoch"])
            if epoch != 10 and (model is None or loaded_epoch != epoch):
                checkpoint = checkpoints[task][epoch]
                model, resolved = load_policy_or_model(str(checkpoint))
                if resolved is not None and Path(resolved).resolve() != checkpoint.resolve():
                    raise ValueError(f"checkpoint resolver changed requested path: {checkpoint}")
                loaded_epoch = epoch
            checkpoint = checkpoints[task][epoch]
            _run_job(
                config=config,
                output_root=output_root,
                root=ROOT,
                job=job,
                manifest=smoke_manifest if smoke_manifest is not None else manifests[task],
                checkpoint=checkpoint,
                checkpoint_hash=hashes[task][epoch],
                device=args.device,
                model=model,
                trace=bool(args.trace),
                smoke=bool(args.smoke),
            )
        except Exception as exc:  # keep independent jobs resumable
            failures += 1
            target = (
                output_root / "smoke" / task / f"epoch_{int(job['epoch']):02d}" / condition_name(_condition_for_job(job))
                if args.smoke
                else result_dir(output_root, task, int(job["epoch"]), _condition_for_job(job))
            )
            target.mkdir(parents=True, exist_ok=True)
            _write_failure(target, job, exc)
            print(json.dumps({"status": "failed", "job_id": job_id_value, "error": str(exc)}, sort_keys=True), flush=True)
            if str(args.device).startswith("cuda"):
                try:
                    import gc
                    import torch

                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        finally:
            _release(output_root, job_id_value)
    if failures:
        raise RuntimeError(f"{failures} epoch-curve jobs failed; inspect failure.json and resume")


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}" if isinstance(value, (float, int, np.floating)) else str(value)


def _write_plots(rows: Sequence[Mapping[str, Any]], output_root: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_root = output_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for task in CURVE_TASKS:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        by_mode = {mode: axes.flat[index] for index, mode in enumerate(("P0", "P1", "P2", "P3"))}
        task_rows = [row for row in rows if row["task"] == task]
        for mode, axis in by_mode.items():
            mode_rows = [row for row in task_rows if row["mode"] == mode]
            labels = sorted({
                "invariant" if row["action_flow_steps"] is None else f"step {row['action_flow_steps']}"
                for row in mode_rows
            }, key=lambda value: (value != "invariant", value))
            for label in labels:
                selected = [
                    row for row in mode_rows
                    if ("invariant" if row["action_flow_steps"] is None else f"step {row['action_flow_steps']}") == label
                ]
                selected.sort(key=lambda row: int(row["epoch"]))
                axis.plot(
                    [row["epoch"] for row in selected],
                    [row["success_rate_percent"] for row in selected],
                    marker="o",
                    linewidth=1.5,
                    label=label,
                )
            axis.set_title(mode)
            axis.set_xticks(list(CURVE_EPOCHS))
            axis.grid(alpha=0.25)
            if mode == "P0":
                axis.set_ylabel("success rate (%)")
            if mode in {"P2", "P3"}:
                axis.set_xlabel("training epoch")
            if mode_rows:
                axis.legend(fontsize=8)
        fig.suptitle(f"Round 5 Phase 1 epoch curve — {task}")
        fig.tight_layout()
        for suffix in ("png", "svg", "pdf"):
            path = plot_root / f"{task}_success_rate_by_epoch.{suffix}"
            fig.savefig(path, dpi=160 if suffix == "png" else None)
            paths.append(path)
        plt.close(fig)
    return paths


def _render_report(
    *,
    config_path: Path,
    output_root: Path,
    report_path: Path,
    validation: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Round 5 Phase 1：中间 epoch 推理曲线",
        "",
        "本实验固定 R4-AB seed 3072、legacy_50 cohort 和 Phase 1 推理协议，比较 epoch 1–10 的成功率曲线。epoch 1–9 为新评测；epoch 10 只引用已经完成并校验的 Phase 4.5-4 legacy baseline，不重跑、不复制旧结果。",
        "",
        f"- 配置：`{config_path}`",
        f"- 输出根目录：`{output_root}`",
        f"- 条件点：`{validation.get('result_count')}/{validation.get('expected_count')}`；新增评测 `{validation.get('new_result_count')}`，epoch10 复用 `{validation.get('reuse_result_count')}`。",
        f"- status=ok：`{validation.get('all_status_ok')}`；episodes/point：`{validation.get('episodes_per_result')}`。",
        "",
        "## 评测矩阵",
        "",
        "| task | conditions/epoch | 条件定义 |",
        "|---|---:|---|",
    ]
    for task in CURVE_TASKS:
        task_rows = [row for row in rows if row["task"] == task and int(row["epoch"]) == 1]
        definition = ", ".join(
            f"{row['mode']}/{('inv' if row['action_flow_steps'] is None else row['action_flow_steps'])}/{row['cem_protocol']}"
            for row in task_rows
        )
        lines.append(f"| {task} | {len(task_rows)} | {definition} |")
    lines.extend(
        [
            "",
            "## 验收身份",
            "",
            "所有点均要求 checkpoint 路径、checkpoint SHA256、cohort id/SHA256、epoch、50 个 episode 的 `(episode_id, start_step, row_index)` 序列和条件 key 一致。",
            "",
            "| task | cohort | cohort SHA256 | epoch checkpoint SHA256 |",
            "|---|---|---|---|",
        ]
    )
    for task, metadata in validation.get("cohort_metadata", {}).items():
        hashes = validation.get("checkpoint_hashes", {}).get(task, {})
        lines.append(
            f"| {task} | `{metadata.get('cohort_id')}` | `{metadata.get('cohort_sha256')}` | epoch10 `{hashes.get('10')}` |"
        )
    lines.extend(["", "## 成功率数据", "", "结果已写入 `epoch_curve.csv` 和 `epoch_curve.json`。下表列出每个曲线点：", "", "| task | epoch | mode | protocol | step | success | Wilson 95% | source |", "|---|---:|---|---|---:|---|---|---|"])
    for row in rows:
        step = "inv" if row["action_flow_steps"] is None else row["action_flow_steps"]
        lines.append(
            f"| {row['task']} | {row['epoch']} | {row['mode']} | {row['cem_protocol']} | {step} | {row['successes']}/{row['episodes']} ({row['success_rate_percent']:.1f}%) | [{row['wilson_95_percent_low']:.1f}, {row['wilson_95_percent_high']:.1f}]% | {row['source']} |"
        )
    lines.extend(
        [
            "",
            "## 曲线图",
            "",
            "每个 task 一张 P0/P1/P2/P3 四分面图；PNG、SVG、PDF 三种格式均写入 `plots/`。",
            "",
        ]
    )
    for task in CURVE_TASKS:
        lines.append(f"- `{task}_success_rate_by_epoch.png` / `.svg` / `.pdf`")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "这是单训练 seed、单 checkpoint 系列、单 legacy cohort 的描述性曲线；epoch10 是历史 artifact 复用点。曲线只覆盖无 guidance 的 P0–P3 条件，不改变冻结的 Phase 1 guidance 结论或报告。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    output_root = _resolve(args.output_root)
    manifests = _load_manifests(config)
    checkpoints = checkpoint_paths(config, root=ROOT)
    hashes = checkpoint_hashes(checkpoints)
    validate_declared_checkpoint_hashes(config, hashes)
    indexed, validation = load_epoch_curve_results(
        config,
        output_root,
        manifests=manifests,
        checkpoints=checkpoints,
        hashes=hashes,
        root=ROOT,
        require_complete=True,
    )
    rows = sorted(
        (item["row"] for item in indexed.values()),
        key=lambda row: (
            CURVE_TASKS.index(str(row["task"])),
            int(row["epoch"]),
            str(row["mode"]),
            -1 if row["action_flow_steps"] is None else int(row["action_flow_steps"]),
        ),
    )
    payload = {
        "schema_version": CURVE_SCHEMA_VERSION,
        "experiment": config.get("experiment"),
        "config": str(_resolve(args.config).resolve()),
        "output_root": str(output_root.resolve()),
        "validation": validation,
        "rows": rows,
    }
    dump_json_atomic(output_root / "epoch_curve.json", payload)
    fields = list(rows[0].keys()) if rows else []
    import csv

    csv_path = output_root / "epoch_curve.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _write_plots(rows, output_root)
    report_path = _resolve(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(
            config_path=_resolve(args.config).resolve(),
            output_root=output_root.resolve(),
            report_path=report_path.resolve(),
            validation=validation,
            rows=rows,
        ),
        encoding="utf-8",
    )
    summary = {
        "epoch_curve_json": str(output_root / "epoch_curve.json"),
        "epoch_curve_csv": str(csv_path),
        "report": str(report_path),
        "conditions": len(rows),
        "plots": str(output_root / "plots"),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return payload


def dry_run(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    jobs = job_specs(config)
    task = str(args.task)
    if task != "all":
        jobs = [job for job in jobs if job["task"] == task]
    if args.epoch is not None:
        jobs = [job for job in jobs if int(job["epoch"]) == int(args.epoch)]
    conditions = condition_specs(config)
    selected = conditions
    if args.condition_index is not None:
        selected = [conditions[int(index)] for index in args.condition_index]
        selected_keys = {condition_key_from_spec(item) for item in selected}
        jobs = [job for job in jobs if condition_key_from_spec(job) in selected_keys]
    print(
        json.dumps(
            {
                "tasks": list(CURVE_TASKS if task == "all" else (task,)),
                "conditions_per_epoch": {task_name: sum(item["task"] == task_name for item in conditions) for task_name in CURVE_TASKS},
                "new_jobs": sum(int(job["epoch"]) in CURVE_NEW_EPOCHS for job in jobs),
                "epoch10_reuse_points": sum(int(job["epoch"]) == 10 for job in jobs),
                "total_points": len(jobs),
                "checkpoint_epochs": list(CURVE_EPOCHS),
                "output_root": str(_resolve(args.output_root)),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "analyze", "dry-run"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT))
    parser.add_argument("--task", choices=(*CURVE_TASKS, "all"), default="all")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=_gpu)
    parser.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB)
    parser.add_argument("--claim-ttl-seconds", type=int, default=DEFAULT_CLAIM_TTL_SECONDS)
    parser.add_argument("--condition-index", action="append", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    if args.command == "dry-run":
        dry_run(args, config)
    elif args.command == "analyze":
        analyze(args, config)
    else:
        if args.task == "all":
            raise ValueError("worker requires --task")
        run_worker(args, config)


if __name__ == "__main__":
    main()
