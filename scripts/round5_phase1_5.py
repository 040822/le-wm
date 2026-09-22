#!/usr/bin/env python3
"""Run, index, and analyze the independent Round 5 Phase1.5 protocol.

Examples::

    python scripts/round5_phase1_5.py validate
    python scripts/round5_phase1_5.py dry-run --group A6
    CUDA_VISIBLE_DEVICES=0 python scripts/round5_phase1_5.py scan --task cube --gpu 0
    python scripts/round5_phase1_5.py index --allow-incomplete
    python scripts/round5_phase1_5.py analyze --allow-incomplete

The scan command is intentionally resumable.  It writes only under the
Phase1.5 output root and reuses compatible historical results through the
index command; the frozen Phase1 output is never modified.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.checkpoint import load_policy_or_model
from source.common.eval import EvaluationIdentity, compose_eval_config
from source.common.round3_phase1 import CohortManifest
from source.common.round4_eval import run_round4_evaluation, validate_gpu_visibility
from source.common.round5_phase1_5 import (
    PHASE15_BOOTSTRAP_SAMPLES,
    PHASE15_EXECUTION_SEMANTICS,
    PHASE15_EVAL_SEED,
    PHASE15_FLOW_STEPS,
    PHASE15_PROTOCOL_VARIANT,
    PHASE15_STABILITY_SEEDS,
    PHASE15_TASKS,
    adaptive_stability_specs,
    atomic_write_json,
    canonical_json,
    cluster_bootstrap,
    condition_id,
    condition_identity,
    condition_lock,
    grid_counts,
    index_phase15_results,
    make_control_actions,
    primary_condition_specs,
    phase15_scan_slot,
    result_path,
    sampling_stability_specs,
    stable_sha256,
    validate_phase15_result,
)


DEFAULT_CONFIG = ROOT / "config" / "round5" / "phase1_5.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "round5" / "phase1_5_seed3072_legacy"
DEFAULT_REPORT = ROOT / "docs" / "report" / "round5" / "round5_phase1_5_report.md"
DEFAULT_MIN_FREE_MIB = 3500
DEFAULT_MAX_LOAD_PER_CPU = 0.75
DEFAULT_MIN_AVAILABLE_MIB = 8192


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Phase1.5 config must be an object: {path}")
    return value


def _load_manifests(config: Mapping[str, Any]) -> dict[str, CohortManifest]:
    result: dict[str, CohortManifest] = {}
    for task in PHASE15_TASKS:
        path = _resolve(config["cohort"]["paths"][task])
        manifest = CohortManifest.load(path)
        if manifest.protocol_variant != PHASE15_PROTOCOL_VARIANT or manifest.cohort_kind != "dev" or len(manifest.entries) != 50:
            raise ValueError(f"{task} is not a legacy_50/dev Phase1.5 cohort: {path}")
        expected = config["cohort"].get("sha256", {}).get(task)
        if expected is not None and manifest.computed_sha256 != expected:
            raise ValueError(f"{task} cohort SHA256 changed: {manifest.computed_sha256} != {expected}")
        result[task] = manifest
    return result


def _checkpoint_paths(config: Mapping[str, Any]) -> tuple[dict[str, Path], dict[str, str]]:
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for task in PHASE15_TASKS:
        path = _resolve(config["training"]["checkpoints"][task])
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = _sha256_file(path)
        expected = config["training"].get("checkpoint_sha256", {}).get(task)
        if expected is not None and observed != expected:
            raise ValueError(f"{task} checkpoint SHA256 changed: {observed} != {expected}")
        paths[task] = path
        hashes[task] = observed
    return paths, hashes


def _gpu(value: str | None) -> str | None:
    if value is None:
        return None
    if not str(value).isdigit() or int(value) not in range(8):
        raise argparse.ArgumentTypeError("--gpu must select one physical GPU0-7")
    return str(value)


def _gpu_preflight(gpu: str, minimum_free_mib: int) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
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
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect GPU{gpu} before Phase1.5 startup") from exc
    line = next((item.strip() for item in completed.stdout.splitlines() if item.strip()), "")
    fields = [item.strip() for item in line.split(",")]
    if len(fields) < 6:
        raise RuntimeError(f"invalid nvidia-smi response for GPU{gpu}: {line!r}")
    snapshot = {
        "gpu": int(gpu),
        "name": fields[1],
        "memory_total_mib": int(fields[2]),
        "memory_used_mib": int(fields[3]),
        "memory_free_mib": int(fields[4]),
        "utilization_percent": int(fields[5]),
        "minimum_free_mib": int(minimum_free_mib),
    }
    if snapshot["memory_free_mib"] < int(minimum_free_mib):
        raise RuntimeError(
            f"GPU{gpu} has {snapshot['memory_free_mib']} MiB free; "
            f"Phase1.5 requires {minimum_free_mib} MiB"
        )
    print(json.dumps({"gpu_preflight": snapshot}, sort_keys=True))
    return snapshot


def _meminfo_mib() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise RuntimeError("cannot inspect host memory before Phase1.5 startup") from exc
    for line in lines:
        name, separator, raw_value = line.partition(":")
        if not separator:
            continue
        fields = raw_value.strip().split()
        if not fields or not fields[0].isdigit():
            continue
        # Linux reports /proc/meminfo sizes in KiB.  Keep the conversion
        # explicit so the startup record is comparable with nvidia-smi MiB.
        values[name] = int(fields[0]) // 1024
    return values


def _host_preflight(
    *,
    max_load_per_cpu: float,
    minimum_available_mib: int,
) -> dict[str, Any]:
    if max_load_per_cpu <= 0:
        raise ValueError("--max-load-per-cpu must be positive")
    if minimum_available_mib < 0:
        raise ValueError("--min-available-mib cannot be negative")
    cpu_count = max(1, int(os.cpu_count() or 1))
    try:
        load_1, load_5, load_15 = os.getloadavg()
    except OSError as exc:
        raise RuntimeError("cannot inspect host CPU load before Phase1.5 startup") from exc
    memory = _meminfo_mib()
    available_mib = memory.get("MemAvailable")
    if available_mib is None:
        raise RuntimeError("/proc/meminfo does not provide MemAvailable")
    snapshot = {
        "cpu_count": cpu_count,
        "load_1": float(load_1),
        "load_5": float(load_5),
        "load_15": float(load_15),
        "load_per_cpu": float(load_1) / cpu_count,
        "max_load_per_cpu": float(max_load_per_cpu),
        "memory_total_mib": memory.get("MemTotal"),
        "memory_available_mib": int(available_mib),
        "minimum_available_mib": int(minimum_available_mib),
    }
    if snapshot["load_per_cpu"] > float(max_load_per_cpu):
        raise RuntimeError(
            "host CPU load is above the Phase1.5 startup limit: "
            f"load1={load_1:.2f}, cpus={cpu_count}, "
            f"ratio={snapshot['load_per_cpu']:.3f} > {max_load_per_cpu:.3f}"
        )
    if int(available_mib) < int(minimum_available_mib):
        raise RuntimeError(
            "host memory headroom is below the Phase1.5 startup limit: "
            f"available={available_mib} MiB < {minimum_available_mib} MiB"
        )
    print(json.dumps({"host_preflight": snapshot}, sort_keys=True))
    return snapshot


def _configure_device(
    device: str,
    gpu: str | None,
    minimum_free_mib: int,
    max_load_per_cpu: float,
    minimum_available_mib: int,
) -> None:
    if str(device).startswith("cuda"):
        if gpu is None:
            raise ValueError("CUDA Phase1.5 runs require --gpu")
        _host_preflight(
            max_load_per_cpu=max_load_per_cpu,
            minimum_available_mib=minimum_available_mib,
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        validate_gpu_visibility(device)
        _gpu_preflight(gpu, minimum_free_mib)


def _selected_specs(
    args: argparse.Namespace,
    *,
    include_stability: bool = False,
) -> list[dict[str, Any]]:
    specs = list(primary_condition_specs())
    if include_stability:
        specs.extend(sampling_stability_specs())
    if args.group is not None:
        specs = [item for item in specs if item["group"] == args.group]
    if args.task != "all":
        specs = [item for item in specs if item["task"] == args.task]
    if args.condition_index is not None:
        specs = [specs[index] for index in args.condition_index]
    if args.limit is not None:
        specs = specs[: int(args.limit)]
    return specs


def _load_adaptive_specs(output_root: Path) -> list[dict[str, Any]]:
    path = output_root / "analysis" / "adaptive_stability_specs.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"adaptive stability selection is missing: {path}; run select-stability first"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    specs = value.get("specs") if isinstance(value, Mapping) else None
    if not isinstance(specs, list) or not all(isinstance(item, Mapping) for item in specs):
        raise ValueError(f"invalid adaptive stability selection: {path}")
    return [dict(item) for item in specs]


def _identity_map(
    specs: Sequence[Mapping[str, Any]],
    manifests: Mapping[str, CohortManifest],
    checkpoints: Mapping[str, Path],
    checkpoint_hashes: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {}
    for spec in specs:
        key = stable_sha256(dict(spec))
        identities[key] = condition_identity(
            spec,
            checkpoint=str(checkpoints[str(spec["task"])]),
            checkpoint_sha256=checkpoint_hashes[str(spec["task"])],
            cohort=manifests[str(spec["task"])],
        )
    return identities


def _identity_for(spec: Mapping[str, Any], identities: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    return identities[stable_sha256(dict(spec))]


def _compose(task: str, spec: Mapping[str, Any], device: str):
    cem_samples = int(spec.get("cem_num_samples") or 300)
    cem_iterations = int(spec.get("cem_iterations") or 30)
    topk = max(1, int(round(cem_samples * float(spec.get("cem_elite_ratio") or 0.1))))
    return compose_eval_config(
        task,
        overrides=[
            "eval.num_eval=50",
            "eval.goal_offset_steps=25",
            "eval.eval_budget=50",
            "plan_config.horizon=5",
            "plan_config.receding_horizon=5",
            "plan_config.action_block=5",
            "output.save_video=false",
            f"solver.device={device}",
            f"solver.num_samples={cem_samples}",
            f"solver.topk={topk}",
            f"solver.n_steps={cem_iterations}",
            f"solver.var_scale={float(spec.get('cem_var_scale') or 1.0)}",
        ],
    )


def _run_spec(
    spec: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    output_root: Path,
    manifest: CohortManifest,
    checkpoint: Path,
    checkpoint_sha256: str,
    device: str,
    model: Any,
) -> dict[str, Any]:
    identity = condition_identity(
        spec,
        checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_sha256,
        cohort=manifest,
    )
    target = result_path(output_root, spec, identity)
    if target.is_file():
        payload = json.loads(target.read_text(encoding="utf-8"))
        validate_phase15_result(payload, expected_identity=identity, manifest=manifest)
        return {"status": "reused", "path": str(target), "condition_id": condition_id(identity)}
    guidance = config["guidance"]
    cfg = _compose(str(spec["task"]), spec, device)
    mode = str(spec["mode"])
    action_flow_steps = spec.get("flow_steps") if mode != "P1" else None
    candidate_count = int(spec.get("candidate_count") or (1 if mode == "P0" else 64))
    if mode == "P1":
        candidate_count = int(spec.get("cem_num_samples") or 300)
    elif mode == "P2":
        candidate_count = int(spec.get("cem_num_samples") or 300)
    identity_metadata = EvaluationIdentity(
        entrypoint="round5_phase1_5",
        policy_kind="round4_shared_dit",
        checkpoint=str(checkpoint.resolve()),
        epoch=int(config["training"]["epoch"]),
        stage=mode,
    )
    guidance_mode = str(spec.get("guidance", "none"))
    inner_steps = int(spec.get("po_iterations") or guidance.get("default_inner_steps", 5))
    last_steps = int(spec.get("guidance_last_steps") or guidance.get("default_last_steps", 5))
    step_size = float(spec.get("guidance_step_size") or guidance.get("default_step_size", 0.01))
    max_rms_offset = float(spec.get("max_rms_offset") or guidance.get("default_max_rms_offset", 0.2))
    try:
        with condition_lock(target):
            payload = run_round4_evaluation(
                cfg,
                task=str(spec["task"]),
                policy_or_model=model,
                mode=mode,
                identity=identity_metadata,
                manifest=manifest,
                output_dir=target.parent,
                trace_output_dir=target.parent / "trace",
                device=device,
                trace=True,
                candidate_count=candidate_count,
                flow_steps=int(spec.get("flow_steps") or 16),
                action_flow_steps=action_flow_steps,
                solver_batch_size=int(config["evaluation"]["solver_batch_size"]),
                candidate_batch_size=int(config["evaluation"]["candidate_batch_size"]),
                action_flow_integrator="euler",
                action_bound_mode=str(spec["action_bound_mode"]),
                cem_protocol=str(spec["cem_protocol"]),
                guidance_mode=guidance_mode,
                guidance_step_size=step_size,
                guidance_last_steps=last_steps,
                guidance_inner_steps=inner_steps,
                guidance_max_rms_offset=max_rms_offset,
                proposal_chunk_size=int(config["evaluation"]["proposal_chunk_size"]),
                allowed_protocol_variants=("legacy",),
                allow_variable_candidate_count=True,
                allow_solver_config_override=True,
            )
            payload = dict(payload)
            payload["phase15_identity"] = identity
            payload["phase15_condition"] = dict(spec)
            payload["phase15_execution_semantics"] = dict(PHASE15_EXECUTION_SEMANTICS)
            payload["phase15_result_schema_version"] = "round5_phase1_5_result_v1"
            validate_phase15_result(payload, expected_identity=identity, manifest=manifest)
            atomic_write_json(target, payload)
            atomic_write_json(
                target.with_name("complete.json"),
                {
                    "schema_version": "round5_phase1_5_result_v1",
                    "status": "completed",
                    "result": str(target),
                    "result_sha256": _sha256_file(target),
                },
            )
    except Exception:
        # The caller records infrastructure failures.  An episode failure in a
        # valid evaluator result remains a normal status=ok result.
        raise
    return {"status": "completed", "path": str(target), "condition_id": condition_id(identity)}


def _scan_unbounded(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    specs = _selected_specs(args, include_stability=args.include_stability)
    if args.include_adaptive_stability:
        specs.extend(_load_adaptive_specs(_resolve(args.output_root)))
    if not specs:
        raise ValueError("no Phase1.5 conditions selected")
    manifests = _load_manifests(config)
    checkpoints, checkpoint_hashes = _checkpoint_paths(config)
    _configure_device(
        args.device,
        args.gpu,
        args.min_free_mib,
        args.max_load_per_cpu,
        args.min_available_mib,
    )
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for spec in specs:
        grouped.setdefault(str(spec["task"]), []).append(spec)
    for task, task_specs in grouped.items():
        model, resolved = load_policy_or_model(str(checkpoints[task]))
        if resolved is not None and Path(resolved).resolve() != checkpoints[task].resolve():
            raise ValueError(f"checkpoint resolver changed requested path: {checkpoints[task]}")
        for spec in task_specs:
            try:
                print(json.dumps(_run_spec(
                    spec,
                    config=config,
                    output_root=output_root,
                    manifest=manifests[task],
                    checkpoint=checkpoints[task],
                    checkpoint_sha256=checkpoint_hashes[task],
                    device=args.device,
                    model=model,
                ), ensure_ascii=False, sort_keys=True))
            except Exception as exc:
                from source.common.round5_phase1_5 import mark_infrastructure_failure
                identity = condition_identity(
                    spec,
                    checkpoint=str(checkpoints[task]),
                    checkpoint_sha256=checkpoint_hashes[task],
                    cohort=manifests[task],
                )
                mark_infrastructure_failure(result_path(output_root, spec, identity), exc)
                raise


def scan(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output_root = _resolve(args.output_root)
    with phase15_scan_slot(output_root, max_slots=int(args.max_concurrent_scans)):
        _scan_unbounded(args, config)


def index(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    specs = _selected_specs(args, include_stability=args.include_stability)
    if args.include_adaptive_stability:
        specs.extend(_load_adaptive_specs(_resolve(args.output_root)))
    manifests = _load_manifests(config)
    checkpoints, checkpoint_hashes = _checkpoint_paths(config)
    identities = _identity_map(specs, manifests, checkpoints, checkpoint_hashes)
    indexed = index_phase15_results(
        _resolve(args.output_root),
        specs,
        identities=identities,
        manifests=manifests,
        checkpoints={task: str(path.resolve()) for task, path in checkpoints.items()},
        history_roots=[_resolve(path) for path in config.get("historical_roots", ())],
    )
    counts: dict[str, int] = {}
    for item in indexed.values():
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    compact_items: dict[str, dict[str, Any]] = {}
    for key, item in indexed.items():
        compact = {name: value for name, value in item.items() if name != "payload"}
        artifact = item.get("payload", {})
        if isinstance(artifact, Mapping):
            compact["payload_summary"] = {
                "task": artifact.get("task"),
                "round4_mode": artifact.get("round4_mode", artifact.get("stage")),
                "success_rate": artifact.get("success_rate"),
                "round4_planning": artifact.get("round4_planning", {}),
                "phase15_condition": artifact.get("phase15_condition"),
            }
        compact_items[key] = compact
    payload = {
        "schema_version": "round5_phase1_5_index_v1",
        "output_root": str(_resolve(args.output_root)),
        "conditions": len(indexed),
        "counts": counts,
        "items": compact_items,
    }
    target = _resolve(args.output_root) / "analysis" / "index.json"
    atomic_write_json(target, payload)
    print(json.dumps({"index": str(target), "conditions": len(indexed), "counts": counts}, ensure_ascii=False, sort_keys=True))
    if not args.allow_incomplete and counts.get("completed", 0) + counts.get("reused_success_only", 0) != len(indexed):
        raise RuntimeError("Phase1.5 index is incomplete; use --allow-incomplete to inspect it")
    return payload


def _rows_from_index(index_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in index_payload["items"].values():
        row: dict[str, Any] = {
            "condition_id": item["condition_id"],
            "status": item["status"],
            "source": item.get("source"),
            "path": item.get("path"),
        }
        payload = item.get("payload", item.get("payload_summary", {}))
        spec = item.get("spec", payload.get("phase15_condition", {}))
        row.update({
            "spec": dict(spec) if isinstance(spec, Mapping) else {},
            "task": spec.get("task", payload.get("task")),
            "group": spec.get("group"),
            "family": spec.get("family"),
            "mode": spec.get("mode", payload.get("round4_mode")),
            "flow_steps": spec.get("flow_steps"),
            "candidate_count": spec.get("candidate_count"),
            "guidance": spec.get("guidance", "none"),
            "success_rate": payload.get("success_rate", item.get("validation", {}).get("success_rate")),
            "successes": item.get("validation", {}).get("successes"),
            "episodes": item.get("validation", {}).get("episodes"),
        })
        planning = payload.get("round4_planning", {})
        row.update({
            "planning_median_seconds": planning.get("planning_median_seconds"),
            "planning_p95_seconds": planning.get("planning_p95_seconds"),
            "forward_count": planning.get("forward_count"),
            "planning_forward_count": planning.get("forward_count"),
            "guidance_backward_count": planning.get("guidance_backward_count"),
            "peak_memory_bytes": planning.get("peak_memory_bytes"),
        })
        rows.append(row)
    return rows


def _render_report(
    *,
    config_path: Path,
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    index_payload: Mapping[str, Any],
) -> str:
    counts = index_payload["counts"]
    group_rows: list[str] = []
    planned_counts = {"A1": 144, "A2": 432, "A3": 864, "A4": 432, "A5": 96, "A6": 240, "A7": 216}
    for group in ("A1", "A2", "A3", "A4", "A5", "A6", "A7"):
        values = [row for row in rows if row.get("group") == group]
        group_rows.append(
            f"| {group} | {planned_counts[group]} | {sum(row.get('status') in {'completed', 'reused_success_only'} for row in values)} | "
            f"{planned_counts[group] - sum(row.get('status') in {'completed', 'reused_success_only'} for row in values)} |"
        )
    complete = [row for row in rows if row.get("success_rate") is not None]
    complete.sort(key=lambda row: float(row.get("success_rate") or -1.0), reverse=True)
    top_lines = [
        "| task | group | family | success | p50 planning (s) | source |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in complete[:20]:
        top_lines.append(
            f"| {row.get('task', '—')} | {row.get('group', '—')} | {row.get('family', '—')} | "
            f"{float(row['success_rate']) * 100:.1f}% | {row.get('planning_median_seconds', '—')} | {row.get('source', '—')} |"
        )
    if not complete:
        top_lines.append("| — | — | — | — | — | — |")
    if counts.get("pending", 0) or counts.get("failed", 0):
        decisions = (
            "本报告仍处于扫描阶段，条件尚未齐全；P3、PO/GF、CEM 的最终取舍暂记为尚未收敛。"
        )
    else:
        decisions = (
            "最终四项判断必须结合同状态候选池、真实物理后果和隔离计时结果填写；"
            "若这些诊断产物没有跨任务稳定占优，按协议报告尚未收敛。"
        )
    return "\n".join(
        [
            "# Round5 Phase1.5：冻结模型的决策能力诊断与推理方案收敛",
            "",
            "本报告由独立 Phase1.5 入口生成。结果只使用 R4-AB seed3072 epoch10 和 legacy_50，"
            "不修改 Phase1 冻结结果。闭环成功率是探索性证据，重复这 50 个起点不能替代最终泛化评测。",
            "",
            f"- 配置：`{config_path}`",
            f"- 输出根目录：`{output_root}`",
            f"- 条件状态：`{json.dumps(counts, ensure_ascii=False, sort_keys=True)}`",
            "",
            "## 主扫描验收",
            "",
            "| group | planned | indexed | pending |",
            "|---|---:|---:|---:|",
            *group_rows,
            "",
            "主扫描网格为 2,424 条条件、121,200 个 episode；固定 seed43/44 稳定性扩展最多增加 176 条条件。",
            "历史结果只有在 checkpoint、legacy cohort、normalizer、动作裁剪、精度、随机数和候选生成语义都一致时才复用；缺轨迹历史结果只进入 success-only 统计。",
            "",
            "## 当前最高成功率条件（描述性）",
            "",
            *top_lines,
            "",
            "## 诊断与决策",
            "",
            "- 候选池：报告 oracle、B-selected、selection regret，并按状态聚类 bootstrap；常量候选池相关系数保持 undefined。",
            "- 梯度修正：同时报告预测 latent cost、真实 latent cost 和物理距离改善，以及 model exploitation。",
            "- Probe：按轨迹拆分 train/validation，排除评测轨迹；真实未来图像与 B 预测 future latent 分开报告。",
            "- 计时：包含编码、提案、评分/梯度、CEM 更新和动作输出；环境时间单独报告。",
            f"- 计时明细：`{output_root / 'analysis' / 'timing.json'}`；规划延迟来自同步的 per-replan 样本，环境 wall clock 单列。",
            "",
            f"{decisions}",
            "",
            "## 产物",
            "",
            f"- 索引：`{output_root / 'analysis' / 'index.json'}`",
            f"- 条件 CSV：`{output_root / 'analysis' / 'conditions.csv'}`",
            f"- 计时：`{output_root / 'analysis' / 'timing.json'}`",
            f"- 本报告：`{output_root.parents[2] / 'docs' / 'report' / 'round5' / 'round5_phase1_5_report.md'}`",
            "",
        ]
    )


def analyze(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    index_path = _resolve(args.output_root) / "analysis" / "index.json"
    if not index_path.is_file() or args.refresh_index:
        index_payload = index(args, config)
    else:
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    rows = _rows_from_index(index_payload)
    output_root = _resolve(args.output_root)
    analysis = {
        "schema_version": "round5_phase1_5_analysis_v1",
        "experiment": "Round 5 Phase 1.5",
        "config": str(_resolve(args.config)),
        "output_root": str(output_root),
        "primary_conditions": 2424,
        "stability_conditions": len(sampling_stability_specs()),
        "rows": rows,
        "counts": index_payload["counts"],
        "bootstrap": {"samples": PHASE15_BOOTSTRAP_SAMPLES, "cluster": "state"},
        "decision_status": "pending" if index_payload["counts"].get("pending", 0) else "ready_for_diagnostic_review",
    }
    analysis_path = output_root / "analysis" / "analysis.json"
    atomic_write_json(analysis_path, analysis)
    if rows:
        fields = sorted({key for row in rows for key in row})
        csv_path = output_root / "analysis" / "conditions.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    report_path = _resolve(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(
            config_path=_resolve(args.config),
            output_root=output_root,
            rows=rows,
            index_payload=index_payload,
        ),
        encoding="utf-8",
    )
    result = {"analysis": str(analysis_path), "report": str(report_path), "conditions": len(rows), "counts": index_payload["counts"]}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def timing(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract fair planning timing from completed full-path evaluations.

    ``round4_eval`` records synchronized per-replan timings around encoding,
    proposal, verifier/guidance, CEM updates, and action selection.  The
    environment wall clock is retained separately as ``evaluation_seconds``;
    it is never used as the planning latency claim.
    """
    del config
    index_path = _resolve(args.output_root) / "analysis" / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"timing requires an index: {index_path}")
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for item in index_payload.get("items", {}).values():
        if item.get("status") not in {"completed", "reused_success_only"}:
            continue
        path = item.get("path")
        if not path or not Path(path).is_file():
            continue
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        planning = payload.get("round4_planning", {})
        samples = np.asarray(
            planning.get("planning_samples_seconds", ()), dtype=np.float64
        )
        samples = samples[np.isfinite(samples)]
        if len(samples) == 0 and planning.get("planning_median_seconds") is not None:
            samples = np.asarray([float(planning["planning_median_seconds"])])
        if len(samples) == 0:
            continue
        spec = item.get("spec", payload.get("phase15_condition", {}))
        records.append(
            {
                "condition_id": item.get("condition_id"),
                "task": spec.get("task", payload.get("task")),
                "group": spec.get("group"),
                "family": spec.get("family"),
                "mode": spec.get("mode", payload.get("round4_mode")),
                "guidance": spec.get("guidance", "none"),
                "planning_p50_seconds": float(np.quantile(samples, 0.50)),
                "planning_p95_seconds": float(np.quantile(samples, 0.95)),
                "batch50_throughput_per_second": float(50.0 / np.mean(samples)),
                "planning_samples": int(len(samples)),
                "forward_count": planning.get("forward_count"),
                "guidance_backward_count": planning.get("guidance_backward_count"),
                "peak_memory_bytes": planning.get("peak_memory_bytes"),
                "evaluation_seconds_including_environment": payload.get(
                    "evaluation_seconds"
                ),
                "environment_separate_from_planning": True,
                "source": item.get("source"),
            }
        )
    output = _resolve(args.output_root) / "analysis" / "timing.json"
    atomic_write_json(
        output,
        {
            "schema_version": "round5_phase1_5_timing_v1",
            "method": "synchronized per-replan samples recorded by round4_eval",
            "environment_wall_clock_is_separate": True,
            "warmup_runs": "not applicable to retrospective scan events",
            "records": records,
            "conditions": len(records),
        },
    )
    result = {"timing": str(output), "conditions": len(records)}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def select_stability(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    """Select the post-scan adaptive stability extension from primary rows."""
    index_path = _resolve(args.output_root) / "analysis" / "index.json"
    if not index_path.is_file() or args.refresh_index:
        index_payload = index(args, config)
    else:
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    rows = _rows_from_index(index_payload)
    primary_rows = [
        row for row in rows
        if isinstance(row.get("spec"), Mapping)
        and int(row["spec"].get("evaluation_seed", PHASE15_EVAL_SEED)) == PHASE15_EVAL_SEED
    ]
    specs = adaptive_stability_specs(primary_rows)
    target = _resolve(args.output_root) / "analysis" / "adaptive_stability_specs.json"
    atomic_write_json(
        target,
        {
            "schema_version": "round5_phase1_5_adaptive_stability_v1",
            "source_index": str(index_path),
            "primary_rows": len(primary_rows),
            "specs": specs,
        },
    )
    print(json.dumps({"adaptive_stability": str(target), "conditions": len(specs)}, ensure_ascii=False, sort_keys=True))


def validate(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    specs = primary_condition_specs()
    stability = sampling_stability_specs()
    anchors = np.zeros((3, 25, 2), dtype=np.float64)
    controls, metadata = make_control_actions(anchors, seed=2026)
    payload = {
        "protocol": PHASE15_PROTOCOL_VARIANT,
        "primary_conditions": len(specs),
        "group_counts": grid_counts(specs),
        "fixed_stability_conditions": len(stability),
        "maximum_conditions": len(specs) + 176,
        "maximum_episodes": (len(specs) + 176) * 50,
        "control_actions": len(controls),
        "control_metadata": len(metadata),
        "flow_steps": list(PHASE15_FLOW_STEPS),
        "execution_semantics": dict(PHASE15_EXECUTION_SEMANTICS),
        "status": "ok",
    }
    if payload["maximum_conditions"] > 2600 or payload["maximum_episodes"] > 130000:
        raise AssertionError("Phase1.5 stability budget exceeds the plan")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def dry_run(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    specs = _selected_specs(args, include_stability=args.include_stability)
    print(json.dumps({
        "status": "ok",
        "config": str(_resolve(args.config)),
        "conditions_selected": len(specs),
        "primary_conditions": 2424,
        "fixed_stability_conditions": 96,
        "tasks": sorted({str(spec["task"]) for spec in specs}),
        "groups": sorted({str(spec["group"]) for spec in specs}),
        "gpu_preference": [0, 1, 7],
        "output_root": str(_resolve(args.output_root)),
    }, ensure_ascii=False, sort_keys=True))


def calibrate(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    """Run the plan's five-start smoke calibration for selected tasks."""
    tasks = PHASE15_TASKS if args.task == "all" else (args.task,)
    manifests = _load_manifests(config)
    checkpoints, checkpoint_hashes = _checkpoint_paths(config)
    _configure_device(
        args.device,
        args.gpu,
        args.min_free_mib,
        args.max_load_per_cpu,
        args.min_available_mib,
    )
    calibration_specs = [
        {
            "name": "P0_s1",
            "mode": "P0",
            "flow_steps": 1,
            "candidate_count": 1,
            "cem_protocol": "not_applicable",
            "action_bound_mode": "none",
            "guidance": "none",
        },
        {
            "name": "P3_s1_n64",
            "mode": "P3",
            "flow_steps": 1,
            "candidate_count": 64,
            "cem_protocol": "not_applicable",
            "action_bound_mode": "none",
            "guidance": "none",
        },
        {
            "name": "P0_po_s1",
            "mode": "P0",
            "flow_steps": 1,
            "candidate_count": 1,
            "cem_protocol": "not_applicable",
            "action_bound_mode": "none",
            "guidance": "post_opt",
            "po_iterations": 5,
            "guidance_step_size": 0.01,
            "max_rms_offset": 0.2,
        },
        {
            "name": "P0_gf_s2",
            "mode": "P0",
            "flow_steps": 2,
            "candidate_count": 1,
            "cem_protocol": "not_applicable",
            "action_bound_mode": "none",
            "guidance": "guided_flow",
            "po_iterations": 5,
            "guidance_step_size": 0.01,
            "max_rms_offset": 0.2,
            "guidance_last_steps": 2,
        },
        {
            "name": "P1_c100_i1",
            "mode": "P1",
            "flow_steps": None,
            "candidate_count": 100,
            "cem_protocol": "cem-clip",
            "action_bound_mode": "candidate_clip",
            "guidance": "none",
            "cem_num_samples": 100,
            "cem_iterations": 1,
            "cem_elite_ratio": 0.1,
            "cem_var_scale": 1.0,
        },
        {
            "name": "P2_c300_i3",
            "mode": "P2",
            "flow_steps": 1,
            "candidate_count": 300,
            "cem_protocol": "cem-clip",
            "action_bound_mode": "candidate_clip",
            "guidance": "none",
            "cem_num_samples": 300,
            "cem_iterations": 3,
            "cem_elite_ratio": 0.1,
            "cem_var_scale": 1.0,
        },
    ]
    if args.limit is not None:
        calibration_specs = calibration_specs[: int(args.limit)]
    target_root = _resolve(args.output_root) / "calibration"
    target_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for task in tasks:
        source_manifest = manifests[task]
        temporary = replace(
            source_manifest,
            cohort_id=f"{source_manifest.cohort_id}_calibration5",
            cohort_kind="custom",
            entries=tuple(source_manifest.entries[:5]),
            episode_split={
                **source_manifest.episode_split,
                "selected": tuple(entry.episode_id for entry in source_manifest.entries[:5]),
            },
        )
        temporary = replace(temporary, cohort_sha256=temporary.computed_sha256)
        model, resolved = load_policy_or_model(str(checkpoints[task]))
        if resolved is not None and Path(resolved).resolve() != checkpoints[task].resolve():
            raise ValueError(f"checkpoint resolver changed requested path: {checkpoints[task]}")
        for spec in calibration_specs:
            if spec["mode"] in {"P1", "P2"} and task != "reacher":
                spec = {**spec, "cem_protocol": "legacy", "action_bound_mode": "none"}
            name = f"{task}/{spec['name']}"
            cfg = _compose(task, spec, args.device)
            cfg.eval.num_eval = 5
            cfg.world.num_envs = 5
            output_dir = target_root / task / str(spec["name"])
            try:
                result = run_round4_evaluation(
                    cfg,
                    task=task,
                    policy_or_model=model,
                    mode=str(spec["mode"]),
                    identity=EvaluationIdentity(
                        entrypoint="round5_phase1_5_calibration",
                        policy_kind="round4_shared_dit",
                        checkpoint=str(checkpoints[task].resolve()),
                        epoch=int(config["training"]["epoch"]),
                        stage=str(spec["mode"]),
                    ),
                    manifest=temporary,
                    output_dir=output_dir,
                    trace=True,
                    device=args.device,
                    candidate_count=int(spec["candidate_count"]),
                    flow_steps=int(spec["flow_steps"] or 16),
                    action_flow_steps=spec["flow_steps"],
                    solver_batch_size=1,
                    candidate_batch_size=64,
                    action_bound_mode=str(spec["action_bound_mode"]),
                    cem_protocol=str(spec["cem_protocol"]),
                    guidance_mode=str(spec["guidance"]),
                    guidance_step_size=float(spec.get("guidance_step_size", 0.01)),
                    guidance_last_steps=int(spec.get("guidance_last_steps", 5)),
                    guidance_inner_steps=int(spec.get("po_iterations", 5)),
                    guidance_max_rms_offset=float(spec.get("max_rms_offset", 0.2)),
                    proposal_chunk_size=512,
                    allowed_protocol_variants=("legacy",),
                    allow_variable_candidate_count=True,
                    allow_solver_config_override=True,
                )
                records.append({"task": task, "condition": spec["name"], "status": "ok", "success_rate": result["success_rate"]})
            except Exception as exc:
                records.append({"task": task, "condition": spec["name"], "status": "failed", "error": str(exc)})
                raise
    atomic_write_json(target_root / "calibration.json", {"tasks": list(tasks), "episodes_per_condition": 5, "results": records})
    print(json.dumps({"calibration": str(target_root / 'calibration.json'), "conditions": len(records)}, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "dry-run", "calibrate", "scan", "index", "analyze", "timing", "select-stability"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT))
    parser.add_argument("--task", choices=(*PHASE15_TASKS, "all"), default="all")
    parser.add_argument("--group")
    parser.add_argument("--condition-index", action="append", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-stability", action="store_true")
    parser.add_argument("--include-adaptive-stability", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--refresh-index", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=_gpu)
    parser.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB)
    parser.add_argument("--min-available-mib", type=int, default=DEFAULT_MIN_AVAILABLE_MIB)
    parser.add_argument("--max-load-per-cpu", type=float, default=DEFAULT_MAX_LOAD_PER_CPU)
    parser.add_argument("--max-concurrent-scans", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _load_config(_resolve(args.config))
    if args.command == "validate":
        validate(args, config)
    elif args.command == "dry-run":
        dry_run(args, config)
    elif args.command == "calibrate":
        calibrate(args, config)
    elif args.command == "scan":
        scan(args, config)
    elif args.command == "index":
        index(args, config)
    elif args.command == "select-stability":
        select_stability(args, config)
    elif args.command == "timing":
        timing(args, config)
    else:
        analyze(args, config)


if __name__ == "__main__":
    main()
