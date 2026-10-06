#!/usr/bin/env python3
"""Run, index, and analyze the independent Round 5 Phase1.5 protocol.

Examples::

    python scripts/round5_phase1_5.py validate
    python scripts/round5_phase1_5.py dry-run --group A6
    CUDA_VISIBLE_DEVICES=0 python scripts/round5_phase1_5.py scan --task cube --gpu 0
    python scripts/round5_phase1_5.py index --allow-incomplete
    python scripts/round5_phase1_5.py analyze --allow-incomplete

The scan command is intentionally resumable. It validates current artifacts,
reuses compatible historical results before launching evaluations, and writes
new results only under the Phase1.5 output root; frozen Phase1 output is never
modified.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
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
    synchronous_timing,
    validate_phase15_result,
)


DEFAULT_CONFIG = ROOT / "config" / "round5" / "phase1_5.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "round5" / "phase1_5_seed3072_legacy"
DEFAULT_REPORT = ROOT / "docs" / "report" / "round5" / "round5_phase1_5_report.md"
DEFAULT_MIN_FREE_MIB = 3500
DEFAULT_MAX_LOAD_PER_CPU = 0.75
DEFAULT_MIN_AVAILABLE_MIB = 8192
DEFAULT_MIN_SWAP_FREE_MIB = 0


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
    minimum_swap_free_mib: int = DEFAULT_MIN_SWAP_FREE_MIB,
) -> dict[str, Any]:
    if max_load_per_cpu <= 0:
        raise ValueError("--max-load-per-cpu must be positive")
    if minimum_available_mib < 0:
        raise ValueError("--min-available-mib cannot be negative")
    if minimum_swap_free_mib < 0:
        raise ValueError("--min-swap-free-mib cannot be negative")
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
        "swap_free_mib": memory.get("SwapFree"),
        "minimum_swap_free_mib": int(minimum_swap_free_mib),
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
    swap_free_mib = snapshot.get("swap_free_mib")
    if swap_free_mib is not None and int(swap_free_mib) < int(minimum_swap_free_mib):
        raise RuntimeError(
            "host swap headroom is below the Phase1.5 startup limit: "
            f"free={swap_free_mib} MiB < {minimum_swap_free_mib} MiB"
        )
    print(json.dumps({"host_preflight": snapshot}, sort_keys=True))
    return snapshot


def _configure_device(
    device: str,
    gpu: str | None,
    minimum_free_mib: int,
    max_load_per_cpu: float,
    minimum_available_mib: int,
    minimum_swap_free_mib: int = DEFAULT_MIN_SWAP_FREE_MIB,
) -> None:
    if str(device).startswith("cuda"):
        if gpu is None:
            raise ValueError("CUDA Phase1.5 runs require --gpu")
        _host_preflight(
            max_load_per_cpu=max_load_per_cpu,
            minimum_available_mib=minimum_available_mib,
            minimum_swap_free_mib=minimum_swap_free_mib,
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
    identities = _identity_map(specs, manifests, checkpoints, checkpoint_hashes)
    output_root = _resolve(args.output_root)
    pending_specs, reused = _resolve_scan_sources(
        specs,
        output_root=output_root,
        identities=identities,
        manifests=manifests,
        checkpoints={task: str(path.resolve()) for task, path in checkpoints.items()},
        history_roots=[_resolve(path) for path in config.get("historical_roots", ())],
    )
    for item in reused:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True), flush=True)
    if not pending_specs:
        return
    _configure_device(
        args.device,
        args.gpu,
        args.min_free_mib,
        args.max_load_per_cpu,
        args.min_available_mib,
        args.min_swap_free_mib,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for spec in pending_specs:
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


def _resolve_scan_sources(
    specs: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
    identities: Mapping[str, Mapping[str, Any]],
    manifests: Mapping[str, CohortManifest],
    checkpoints: Mapping[str, str],
    history_roots: Sequence[str | Path],
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    """Validate current results and resolve reusable history before evaluation."""
    pending: list[Mapping[str, Any]] = []
    reused: list[dict[str, Any]] = []
    missing_current: list[Mapping[str, Any]] = []
    for spec in specs:
        identity = identities[stable_sha256(dict(spec))]
        target = result_path(output_root, spec, identity)
        if not target.is_file():
            missing_current.append(spec)
            continue
        payload = json.loads(target.read_text(encoding="utf-8"))
        validation = validate_phase15_result(
            payload,
            expected_identity=identity,
            manifest=manifests[str(spec["task"])],
        )
        reused.append({
            "status": "reused",
            "source": "current",
            "reuse_scope": "full",
            "reuse_reason": "current_artifact_already_available",
            "condition_id": condition_id(identity),
            "episodes": validation["episodes"],
            "path": str(target),
        })

    if not missing_current:
        return pending, reused

    # An empty temporary root ensures only historical artifacts can satisfy
    # missing current conditions. Historic results stay read-only; a condition
    # is skipped only after full identity and execution-semantics validation.
    with tempfile.TemporaryDirectory(prefix="round5-phase1-5-history-") as temporary_root:
        history_index = index_phase15_results(
            temporary_root,
            missing_current,
            identities=identities,
            manifests=manifests,
            checkpoints=checkpoints,
            history_roots=history_roots,
        )
    for spec in missing_current:
        identity = identities[stable_sha256(dict(spec))]
        item = history_index[condition_id(identity)]
        if item.get("source") == "history":
            reused.append({
                "status": "reused",
                "source": "history",
                "reuse_scope": item.get("reuse_scope"),
                "reuse_reason": item.get("reuse_reason"),
                "condition_id": item["condition_id"],
                "episodes": item.get("validation", {}).get("episodes"),
                "path": item["path"],
            })
        else:
            pending.append(spec)
    return pending, reused


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
            "reuse_reason": item.get("reuse_reason"),
            "historical_path_candidate_count": item.get("historical_path_candidate_count"),
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


def _history_reuse_audit(
    index_payload: Mapping[str, Any],
    verified_history: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize per-condition historical reuse decisions for reports."""
    source_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    path_candidate_pairs = 0
    items = index_payload.get("items", {})
    if isinstance(items, Mapping):
        for item in items.values():
            if not isinstance(item, Mapping):
                continue
            source = str(item.get("source") or "none")
            reason = str(item.get("reuse_reason") or "not_recorded")
            source_counts[source] = source_counts.get(source, 0) + 1
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            path_candidate_pairs += int(item.get("historical_path_candidate_count") or 0)
    audit = {
        "source_counts": source_counts,
        "reuse_reason_counts": reason_counts,
        "historical_path_candidate_pairs": path_candidate_pairs,
        "historical_path_candidate_pairs_are_hints_only": True,
    }
    if verified_history is not None:
        audit["verified_historical_candidates"] = {
            "conditions": int(verified_history.get("compatible_history_count", 0)),
            "full_reuse": int(verified_history.get("compatible_full_reuse_count", 0)),
            "success_only": int(verified_history.get("compatible_success_only_count", 0)),
            "by_task": dict(verified_history.get("compatible_by_task", {})),
            "by_round": dict(verified_history.get("compatible_by_historical_round", {})),
            "current_index_sources": dict(
                verified_history.get("current_index_source_counts_for_matches", {})
            ),
            "current_artifacts_present": dict(
                verified_history.get("current_artifact_exists_counts", {})
            ),
        }
    return audit


def _build_verified_history_audit(
    output_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact historical matches independently of current artifacts."""
    primary_specs = primary_condition_specs()
    fixed_stability_specs = sampling_stability_specs()
    adaptive_specs = _load_adaptive_specs(output_root)
    specs = [*primary_specs, *fixed_stability_specs, *adaptive_specs]
    manifests = _load_manifests(config)
    checkpoints, checkpoint_hashes = _checkpoint_paths(config)
    identities = _identity_map(specs, manifests, checkpoints, checkpoint_hashes)
    history_roots = [_resolve(path) for path in config.get("historical_roots", ())]
    with tempfile.TemporaryDirectory(prefix="round5-phase1-5-history-audit-") as temporary_root:
        history_index = index_phase15_results(
            temporary_root,
            specs,
            identities=identities,
            manifests=manifests,
            checkpoints={task: str(path.resolve()) for task, path in checkpoints.items()},
            history_roots=history_roots,
        )

    current_index_path = output_root / "analysis" / "index.json"
    current_index = (
        json.loads(current_index_path.read_text(encoding="utf-8"))
        if current_index_path.is_file()
        else {}
    )
    current_items = current_index.get("items", {})
    matches: list[dict[str, Any]] = []
    for item in history_index.values():
        if item.get("source") != "history":
            continue
        spec = item["spec"]
        identity = identities[stable_sha256(dict(spec))]
        current_path = result_path(output_root, spec, identity)
        current_item = current_items.get(item["condition_id"], {})
        matches.append({
            "condition_id": item["condition_id"],
            "task": spec.get("task"),
            "group": spec.get("group"),
            "mode": spec.get("mode"),
            "family": spec.get("family"),
            "flow_steps": spec.get("flow_steps"),
            "candidate_count": spec.get("candidate_count"),
            "guidance": spec.get("guidance"),
            "reuse_scope": item.get("reuse_scope"),
            "historical_path": item["path"],
            "current_index_source": current_item.get("source"),
            "current_index_reason": current_item.get("reuse_reason"),
            "current_artifact_path": str(current_path),
            "current_artifact_exists": current_path.is_file(),
        })

    def tally(values: Sequence[Any]) -> dict[str, int]:
        result: dict[str, int] = {}
        for value in values:
            key = str(value if value is not None else "none")
            result[key] = result.get(key, 0) + 1
        return dict(sorted(result.items()))

    round_counts: dict[str, int] = {}
    for item in matches:
        path = Path(item["historical_path"])
        source_round = "other"
        for round_name, root in (("round4", history_roots[0:1]), ("round5", history_roots[1:2])):
            if any(path.is_relative_to(candidate) for candidate in root):
                source_round = round_name
                break
        round_counts[source_round] = round_counts.get(source_round, 0) + 1

    payload = {
        "schema_version": "round5_phase1_5_history_reuse_audit_v1",
        "scope": "primary_fixed_and_adaptive_stability_history_only_identity_and_execution_semantics_validation",
        "historical_roots": [str(root) for root in history_roots],
        "current_index_sha256": _sha256_file(current_index_path) if current_index_path.is_file() else None,
        "conditions_audited": len(history_index),
        "primary_conditions": len(primary_specs),
        "fixed_stability_conditions": len(fixed_stability_specs),
        "adaptive_stability_conditions": len(adaptive_specs),
        "history_source_counts": tally([item.get("source") for item in history_index.values()]),
        "history_status_counts": tally([item.get("status") for item in history_index.values()]),
        "compatible_history_count": len(matches),
        "compatible_full_reuse_count": sum(item.get("reuse_scope") == "full" for item in matches),
        "compatible_success_only_count": sum(item.get("reuse_scope") == "success_only" for item in matches),
        "compatible_by_task": tally([item.get("task") for item in matches]),
        "compatible_by_group": tally([item.get("group") for item in matches]),
        "compatible_by_historical_round": dict(sorted(round_counts.items())),
        "current_index_source_counts_for_matches": tally(
            [item.get("current_index_source") for item in matches]
        ),
        "current_artifact_exists_counts": tally(
            [item.get("current_artifact_exists") for item in matches]
        ),
        "provenance_note": (
            "Historical compatibility is independently verified. Existing current artifacts "
            "remain labeled current and are not relabeled as reused."
        ),
        "matches": matches,
    }
    target = output_root / "analysis" / "history_reuse_audit.json"
    atomic_write_json(target, payload)
    return payload


def _diagnostic_decisions(
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Render conservative four-question decisions from persisted evidence."""
    summaries: dict[str, Mapping[str, Any]] = {}
    for task in PHASE15_TASKS:
        summary_path = output_root / "diagnostics" / f"summary_{task}.json"
        if summary_path.is_file():
            value = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(value, Mapping):
                summaries[task] = value

    p3_status = "尚无固定候选池证据"
    p3_supported = None
    task_gains: dict[str, float] = {}
    task_gain_intervals: dict[str, tuple[float, float]] = {}
    task_regrets: dict[str, float] = {}
    for task, summary in summaries.items():
        candidate = summary.get("candidate_pool")
        if not isinstance(candidate, Mapping):
            continue
        gain = candidate.get("selected_minus_random_success_rate")
        bootstrap = candidate.get("bootstrap", {})
        gain_interval = (
            bootstrap.get("selected_minus_random_success_rate", {}).get("ci95")
            if isinstance(bootstrap, Mapping)
            and isinstance(bootstrap.get("selected_minus_random_success_rate"), Mapping)
            else None
        )
        if (
            gain is not None
            and isinstance(gain_interval, Sequence)
            and len(gain_interval) == 2
        ):
            task_gains[task] = float(gain)
            task_gain_intervals[task] = (
                float(gain_interval[0]),
                float(gain_interval[1]),
            )
        regret = candidate.get("selection_regret")
        if regret is not None:
            task_regrets[task] = float(regret)
    if task_gains:
        positive_tasks = sum(
            task_gains[task] > 0.0 and task_gain_intervals[task][0] > 0.0
            for task in task_gains
        )
        mean_gain = float(np.mean(list(task_gains.values())))
        task_count = len(PHASE15_TASKS)
        gain_coverage = len(task_gains)
        regret_coverage = len(task_regrets)
        evidence_complete = gain_coverage == task_count and regret_coverage == task_count
        p3_supported = positive_tasks == task_count if evidence_complete else None
        gain_scope = (
            "跨任务平均"
            if gain_coverage == task_count
            else f"已完成 {gain_coverage}/{task_count} 个任务的平均"
        )
        positive_scope = (
            f"{positive_tasks}/{task_count} 个任务"
            if gain_coverage == task_count
            else f"{positive_tasks}/{gain_coverage} 个已完成任务"
        )
        if regret_coverage == task_count:
            regret_status = (
                f"跨任务平均 oracle−B-selected 归一化距离 regret 为 "
                f"{float(np.mean(list(task_regrets.values()))):.3f}"
            )
        elif task_regrets:
            regret_status = (
                f"已有 {regret_coverage}/{task_count} 个任务的平均 oracle−B-selected "
                f"归一化距离 regret 为 {float(np.mean(list(task_regrets.values()))):.3f}"
            )
        else:
            regret_status = "oracle−B-selected 距离 regret 尚未汇总"
        p3_status = (
            f"六个 S 等权的 B-selected−随机成功率差在{gain_scope}上为 {mean_gain:+.3f}，"
            f"{positive_scope}的状态聚类 95% CI 下界大于 0；"
            + regret_status
        )
        if evidence_complete:
            p3_status += (
                "；支持跨任务保留 P3"
                if p3_supported
                else "；尚不支持跨任务稳定保留 P3"
            )
        else:
            missing_tasks = [
                task for task in PHASE15_TASKS
                if task not in task_gains or task not in task_regrets
            ]
            p3_status += (
                f"；候选池诊断未完整覆盖四任务（增益区间 {gain_coverage}/{task_count}，"
                f"regret {regret_coverage}/{task_count}，待补 {', '.join(missing_tasks)}），"
                "暂不作跨任务判断"
            )
    else:
        p3_status = "候选池配对成功率差的状态聚类区间尚未生成"

    guidance_by_task: dict[str, dict[str, list[float]]] = {}
    matched_guidance_by_task: dict[str, tuple[float, float, float]] = {}
    for task, summary in summaries.items():
        comparison = summary.get("matched_guidance_comparison")
        comparison_bootstrap = (
            comparison.get("bootstrap", {}).get("gf_minus_po_physical_improvement")
            if isinstance(comparison, Mapping)
            and isinstance(comparison.get("bootstrap"), Mapping)
            else None
        )
        comparison_ci = (
            comparison_bootstrap.get("ci95")
            if isinstance(comparison_bootstrap, Mapping)
            else None
        )
        comparison_mean = (
            comparison.get("gf_minus_po_physical_improvement_mean")
            if isinstance(comparison, Mapping)
            else None
        )
        if (
            comparison_mean is not None
            and isinstance(comparison_ci, Sequence)
            and len(comparison_ci) == 2
        ):
            matched_guidance_by_task[task] = (
                float(comparison_mean),
                float(comparison_ci[0]),
                float(comparison_ci[1]),
            )
        for variant in summary.get("guidance_by_variant", ()):
            if not isinstance(variant, Mapping):
                continue
            condition = variant.get("condition", {})
            paired = variant.get("paired_guidance", {})
            mode = condition.get("guidance") if isinstance(condition, Mapping) else None
            advantage = (
                paired.get("paired_advantage_mean")
                if isinstance(paired, Mapping)
                else None
            )
            if mode in {"post_opt", "guided_flow"} and advantage is not None:
                guidance_by_task.setdefault(task, {}).setdefault(str(mode), []).append(
                    float(advantage)
                )
    guidance_task_means = {
        task: {mode: float(np.mean(values)) for mode, values in modes.items() if values}
        for task, modes in guidance_by_task.items()
    }
    mode_means: dict[str, float] = {}
    mode_positive_tasks: dict[str, int] = {}
    for mode in ("post_opt", "guided_flow"):
        values = [
            task_values[mode]
            for task_values in guidance_task_means.values()
            if mode in task_values
        ]
        if values:
            mode_means[mode] = float(np.mean(values))
            mode_positive_tasks[mode] = sum(value > 0.0 for value in values)
    guidance_supported = None
    if mode_means:
        complete_modes = [
            mode for mode in ("post_opt", "guided_flow")
            if sum(mode in values for values in guidance_task_means.values())
            == len(PHASE15_TASKS)
        ]
        if any(
            mode_positive_tasks.get(mode, 0) == len(PHASE15_TASKS)
            for mode in complete_modes
        ):
            guidance_supported = True
        elif len(complete_modes) == 2:
            guidance_supported = False
    method_path = output_root / "analysis" / "method_selection.json"
    method_selection: Mapping[str, Any] = {}
    if method_path.is_file():
        value = json.loads(method_path.read_text(encoding="utf-8"))
        if isinstance(value, Mapping):
            method_selection = value
    matched_timing = [
        item
        for item in method_selection.get("guidance_mode_efficiency_comparisons", ())
        if isinstance(item, Mapping)
    ]
    efficient_guidance_configs = sum(
        item.get("guided_flow_faster_all_tasks") is True
        for item in matched_timing
    )
    if len(matched_guidance_by_task) == len(PHASE15_TASKS):
        stable_real_gain_tasks = sum(
            matched_guidance_by_task[task][1] > 0.0
            for task in PHASE15_TASKS
        )
        mean_gf_minus_po = float(
            np.mean([matched_guidance_by_task[task][0] for task in PHASE15_TASKS])
        )
        real_gain_status = (
            f"GF−PO 状态配对真实物理改善差为 {mean_gf_minus_po:+.4f}，"
            f"{stable_real_gain_tasks}/{len(PHASE15_TASKS)} 个任务的状态聚类 95% CI 下界大于 0"
        )
        real_gain_supported = stable_real_gain_tasks == len(PHASE15_TASKS)
    else:
        real_gain_status = (
            f"GF−PO 状态配对真实物理改善差仅有 "
            f"{len(matched_guidance_by_task)}/{len(PHASE15_TASKS)} 个任务的完整区间"
        )
        real_gain_supported = False
    if not method_selection:
        efficiency_status = "完整跨任务匹配计时尚未完成"
        efficiency_measured = False
    elif matched_timing:
        efficiency_status = (
            f"{efficient_guidance_configs}/{len(matched_timing)} 个完整匹配配置在四任务均测得 GF 更低的单环境 p50 延迟"
        )
        efficiency_measured = True
    else:
        efficiency_status = "没有完整的 PO/GF 四任务匹配精测延迟对"
        efficiency_measured = False
    if real_gain_supported:
        guidance_status = real_gain_status + "；支持稳定真实收益，保留 GF"
    elif efficient_guidance_configs:
        guidance_status = real_gain_status + "；" + efficiency_status + "，支持作为效率方案保留 GF"
    elif efficiency_measured and len(matched_guidance_by_task) == len(PHASE15_TASKS):
        guidance_status = (
            real_gain_status + "；" + efficiency_status
            + "；未见稳定收益或全任务效率优势，按计划从主方法中移除 GF"
        )
    else:
        guidance_status = real_gain_status + "；" + efficiency_status + "，证据不足以决定是否单独保留 GF"
    candidates = [
        row for row in method_selection.get("candidates", ())
        if isinstance(row, Mapping)
    ]
    by_candidate_id = {str(row.get("candidate_id")): row for row in candidates}
    main = by_candidate_id.get(str(method_selection.get("main_candidate_id")))
    main_config = main.get("config", {}) if isinstance(main, Mapping) else {}
    if p3_supported is not True and main_config.get("mode") == "P3":
        family = str(main_config.get("family") or "P3")
        description = f"{family}/P3"
        if main_config.get("flow_steps") is not None:
            description += f", S={main_config['flow_steps']}"
        if main_config.get("candidate_count") is not None:
            description += f", N={main_config['candidate_count']}"
        p3_status += (
            f"；不过统一三 seed 成功率/精测延迟规则仍将 {description} 选为工程主配置；"
            "该配置选择不等同于候选池诊断证明 P3 排序具有稳定的跨任务优势"
        )
    cem_candidates = [
        row for row in candidates
        if row.get("config", {}).get("family") == "cem_budget"
    ]
    best_cem = max(cem_candidates, key=lambda row: float(row["mean_success_rate"])) if cem_candidates else None
    main_family = main.get("config", {}).get("family") if isinstance(main, Mapping) else None
    is_single_proposal_correction = main_family in {"p0_post_opt", "p0_guided_flow"}
    selected_guidance_mode = (
        "post_opt" if main_family == "p0_post_opt"
        else "guided_flow" if main_family == "p0_guided_flow"
        else None
    )
    selected_guidance_gain = mode_means.get(selected_guidance_mode) if selected_guidance_mode else None
    if not method_selection:
        replacement_status = "跨任务三 seed 与精测延迟选择尚未完成"
    elif main is None or best_cem is None:
        replacement_status = "跨任务统一规则未选出可收敛主方案，暂不判定替代 CEM"
    elif not is_single_proposal_correction:
        replacement_status = "当前跨任务主方案不是经过配对实测的单候选 PO/GF，尚无替代 CEM 的直接证据"
    else:
        task_floor = all(
            float(main["task_success_rate"][task])
            >= float(best_cem["task_success_rate"][task]) - 0.04
            for task in PHASE15_TASKS
        )
        faster = (
            float(main["mean_batch1_p50_seconds"])
            < float(best_cem["mean_batch1_p50_seconds"])
        )
        paired_gain = selected_guidance_gain is not None and selected_guidance_gain > 0.0
        if task_floor and faster and paired_gain:
            replacement_status = (
                "跨任务主方案满足每任务 4 个百分点成功率容差、单环境延迟更低，"
                "且真实配对修正收益为正；支持在本 cohort 中以该 PO/GF 配置替代 CEM"
            )
        else:
            reasons = []
            if not task_floor:
                reasons.append("至少一项任务低于 CEM 4 个百分点容差")
            if not faster:
                reasons.append("单环境延迟未更低")
            if not paired_gain:
                reasons.append("真实配对修正收益未为正")
            replacement_status = "PO/GF 暂不能替代 CEM：" + "；".join(reasons)

    if p3_supported is None or guidance_supported is None:
        training_status = "下一轮 R4-AB 消融待诊断完成后确定"
    elif p3_supported and guidance_supported:
        training_status = "优先验证 proposal-verification 与 B→A 梯度匹配训练消融"
    elif p3_supported:
        training_status = "优先验证 proposal-verification 与预测动作输入匹配训练消融"
    else:
        training_status = "优先修正 B 的动作后果匹配训练，再扩展推理模块"
    return [
        f"1. P3：{p3_status}。",
        f"2. PO/GF 替代 CEM：{replacement_status}。",
        f"3. GF：{guidance_status}。",
        f"4. R4-AB 下一轮：{training_status}。",
    ]


def _coarse_timing_sensitivity_report_lines(output_root: Path) -> list[str]:
    selection_path = output_root / "analysis" / "fine_timing_selection.json"
    if not selection_path.is_file():
        return ["粗测前沿敏感性选择尚未生成；待完整粗测结束后再比较两种前沿。"]

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    sensitivity = selection.get("coarse_frontier_sensitivity", {})
    if not isinstance(sensitivity, Mapping):
        sensitivity = {}
    selected_rows = [
        row
        for row in selection.get("selected_conditions", ())
        if isinstance(row, Mapping) and row.get("frontier_membership")
    ]
    status_counts = sensitivity.get("interference_status_counts", {})
    lines = [
        (
            "粗测敏感性按 seed42 主网格逐任务比较观测到的 batch1 前沿，"
            "以及仅保留 batch1 边界采样未观察到外部计算的前沿。"
            "边界采样不能证明整个测量窗口隔离；最终选型只使用精测两个窗口均未观察到外部计算的配置。"
        ),
        "",
        "粗测窗口边界干扰状态计数：",
        "",
        "| window | status | count |",
        "|---|---|---:|",
    ]
    for batch_name in ("batch1", "batch50"):
        counts = status_counts.get(batch_name, {})
        if not isinstance(counts, Mapping) or not counts:
            lines.append(f"| {batch_name} | — | — |")
            continue
        for status, count in sorted(counts.items(), key=lambda item: str(item[0])):
            lines.append(f"| {batch_name} | {status} | {int(count)} |")
    lines.extend(
        [
            "",
            "两种 batch1 前沿及隔离复测候选：",
            "",
            "| task | observed frontier | batch1 boundary-unflagged frontier | union frontier | coarse interference-suspect retests |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for task in PHASE15_TASKS:
        task_rows = [row for row in selected_rows if row.get("task") == task]
        observed = sum(
            "observed" in row.get("frontier_membership", ()) for row in task_rows
        )
        boundary_unflagged = sum(
            "batch1_boundary_unflagged" in row.get("frontier_membership", ())
            for row in task_rows
        )
        suspect = sum(bool(row.get("isolated_retest_recommended")) for row in task_rows)
        union = sum(bool(row.get("frontier_membership")) for row in task_rows)
        lines.append(
            f"| {task} | {observed} | {boundary_unflagged} | {union} | {suspect} |"
        )
    suspect_rows = [
        row for row in selected_rows if row.get("isolated_retest_recommended")
    ]
    completed_retests = 0
    clean_retests = 0
    timing_root = output_root / "analysis" / "timing_conditions"
    for row in suspect_rows:
        condition_id_value = str(row.get("condition_id", ""))
        timing_path = timing_root / f"{condition_id_value}.json"
        if not condition_id_value or not timing_path.is_file():
            continue
        record = json.loads(timing_path.read_text(encoding="utf-8"))
        if not _complete_fine_timing(record):
            continue
        completed_retests += 1
        fine = record["fine_timing"]
        statuses = [
            _timing_interference_status(fine.get(batch_name))
            for batch_name in ("batch1", "batch50")
        ]
        if all(status == _NO_EXTERNAL_COMPUTE_AT_BOUNDARIES for status in statuses):
            clean_retests += 1
    lines.extend(
        [
            "",
            (
                f"全任务共保留 {len(selected_rows)} 个前沿配置，并另保留 "
                f"{selection.get('fixed_anchor_count', 0)} 个固定锚点和 "
                f"{selection.get('adaptive_anchor_count', 0)} 个自适应锚点；"
                f"前沿中的受干扰/缺少快照候选有 "
                f"{selection.get('interference_suspect_frontier_condition_count', 0)} 个；"
                f"其中已完成隔离精测 {completed_retests}/{len(suspect_rows)} 个，"
                f"batch1 与 batch50 边界均未观察到外部计算 {clean_retests}/{len(suspect_rows)} 个。"
            ),
        ]
    )
    return lines


def _format_report_effect(effect: Any) -> str:
    if not isinstance(effect, Mapping) or effect.get("estimate") is None:
        return "—"
    estimate = float(effect["estimate"])
    text = f"{estimate:+.3f}"
    interval = effect.get("ci95")
    if (
        isinstance(interval, Sequence)
        and not isinstance(interval, (str, bytes))
        and len(interval) == 2
    ):
        text += f" [{float(interval[0]):+.3f}, {float(interval[1]):+.3f}]"
    return text


def _format_report_count(value: Any) -> str:
    return "—" if value is None else f"{int(value):,}"


def _control_paired_effect_report_lines(output_root: Path) -> list[str]:
    lines = [
        "| task | control action | success Δ [state-bootstrap 95% CI] | common-step distance Δ [state-bootstrap 95% CI] | coverage (success states; common-step action-state pairs; distance states) |",
        "|---|---|---:|---:|---:|",
    ]
    labels = {
        "rms_normalized_gaussian": "normalized Gaussian",
        "physical_zero": "physical zero",
        "normalized_zero": "normalized zero",
    }
    task_labels = {
        "pusht": "PushT",
        "reacher": "Reacher",
        "tworoom": "TwoRoom",
        "cube": "Cube",
    }
    root = output_root / "diagnostics" / "candidate_pool"
    for task, task_label in task_labels.items():
        path = root / task / "control_paired_effects.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = payload.get("protocol", {})
        cohort_states = protocol.get("states", "—")
        for key, label in labels.items():
            control = payload.get(key, {})
            success_effect = control.get("paired_effect", {}).get(
                "success_rate_delta"
            )
            common = control.get("common_milestone_distance", {})
            distance_effect = common.get("paired_effect")
            paired_count = common.get("paired_action_state_count")
            total_count = common.get("total_action_state_count")
            paired_states = common.get("state_count_with_pairs")
            success_states = (
                success_effect.get("state_count")
                if isinstance(success_effect, Mapping)
                else None
            )
            success_coverage = (
                f"{int(success_states)}/{int(cohort_states)} success states"
                if success_states is not None and cohort_states != "—"
                else "— success states"
            )
            if paired_count is None or total_count is None:
                pair_coverage = "— pairs"
            else:
                pair_coverage = (
                    f"{int(paired_count):,}/{int(total_count):,} pairs"
                )
            if paired_states is not None and cohort_states != "—":
                pair_coverage += (
                    f"; {int(paired_states)}/{int(cohort_states)} distance states"
                )
            lines.append(
                f"| {task_label} | {label} | {_format_report_effect(success_effect)} | "
                f"{_format_report_effect(distance_effect)} | "
                f"{success_coverage}; {pair_coverage} |"
            )
    if len(lines) == 2:
        lines.append("| — | — | — | — | — |")
    return lines


def _control_migration_report_lines(output_root: Path) -> list[str]:
    audit_path = (
        output_root
        / "diagnostics"
        / "candidate_pool"
        / "cube"
        / "control_compatibility_audit.json"
    )
    if not audit_path.is_file():
        return ["Cube 控制池迁移审计尚未生成。"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    source = audit.get("source", {})
    result = audit.get("result", {})
    preserved_files = audit.get("per_action_files", {})
    outcomes = audit.get("preserved_outcomes", {})
    review = audit.get("compatibility_review", {})
    outcomes_reexecuted = outcomes.get("outcomes_reexecuted")
    if outcomes_reexecuted is True:
        outcomes_text = "是"
    elif outcomes_reexecuted is False:
        outcomes_text = "否"
    else:
        outcomes_text = "审计未记录"
    return [
        (
            f"Cube 控制池由 {_format_report_count(source.get('actions'))} 个源动作、"
            f"{_format_report_count(source.get('rows'))} 条记录迁移为 "
            f"{_format_report_count(result.get('actions'))} 个动作、"
            f"{_format_report_count(result.get('rows'))} 条记录。逐动作核对："
            f"{review.get('action_comparison', '审计未提供')}。"
        ),
        (
            f"是否重跑环境结果：{outcomes_text}；"
            f"保留旧版逐动作文件 "
            f"{_format_report_count(preserved_files.get('preserved_legacy_file_count'))} 份。"
        ),
        (
            f"SHA-256：源聚合 `{source.get('sha256', '—')}`；"
            f"当前聚合 `{result.get('sha256', '—')}`。"
        ),
    ]


def _render_report(
    *,
    config_path: Path,
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    index_payload: Mapping[str, Any],
    verified_history_audit: Mapping[str, Any] | None = None,
) -> str:
    counts = index_payload["counts"]
    history_audit = _history_reuse_audit(index_payload, verified_history_audit)
    coarse_timing_sensitivity_lines = _coarse_timing_sensitivity_report_lines(
        output_root
    )
    control_paired_lines = _control_paired_effect_report_lines(output_root)
    control_migration_lines = _control_migration_report_lines(output_root)
    group_rows: list[str] = []
    planned_counts = {"A1": 144, "A2": 432, "A3": 864, "A4": 432, "A5": 96, "A6": 240, "A7": 216}
    for group in ("A1", "A2", "A3", "A4", "A5", "A6", "A7"):
        values = [
            row for row in rows
            if row.get("group") == group
            and int(row.get("spec", {}).get("evaluation_seed", PHASE15_EVAL_SEED)) == PHASE15_EVAL_SEED
        ]
        completed_count = sum(row.get("status") == "completed" for row in values)
        group_rows.append(
            f"| {group} | {planned_counts[group]} | {completed_count} | "
            f"{planned_counts[group] - completed_count} |"
        )
    complete = [
        row for row in rows
        if row.get("success_rate") is not None
        and int(row.get("spec", {}).get("evaluation_seed", PHASE15_EVAL_SEED)) == PHASE15_EVAL_SEED
        and str(row.get("group", "")).startswith("A")
    ]
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

    fixed_count = sum(
        row.get("group") == "S1"
        and int(row.get("spec", {}).get("evaluation_seed", PHASE15_EVAL_SEED)) != PHASE15_EVAL_SEED
        for row in rows
    )
    adaptive_count = sum(
        row.get("spec", {}).get("stability_category") is not None for row in rows
    )

    probe_lines = [
        "| task | validation rows | Ridge MAE | random encoder Ridge MAE | MLP MAE mean (3 seeds) | real future-latent MAE | B predicted-latent MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    probe_summary_path = output_root / "diagnostics" / "probe" / "summary.json"
    if probe_summary_path.is_file():
        probe_summary = json.loads(probe_summary_path.read_text(encoding="utf-8"))
        probe_results = [
            item for item in probe_summary.get("results", ())
            if isinstance(item, Mapping) and item.get("status") == "ok"
        ]
        for item in sorted(probe_results, key=lambda value: str(value.get("task"))):
            ridge = item.get("ridge", {})
            random_ridge = item.get("random_encoder_ridge", {})
            ridge_validation = ridge.get("validation_metrics", {})
            if not isinstance(ridge_validation, Mapping):
                ridge_validation = {}
            random_validation = random_ridge.get("validation_metrics", {})
            if not isinstance(random_validation, Mapping):
                random_validation = {}
            mlp_rows = item.get("mlp", ())
            mlp_mae = (
                float(np.mean([float(value["mae"]) for value in mlp_rows]))
                if mlp_rows else None
            )
            ridge_mae_value = ridge_validation.get("mae")
            if ridge_mae_value is None:
                ridge_mae_value = ridge.get("mae")
            ridge_mae = float(ridge_mae_value) if ridge_mae_value is not None else None
            random_mae_value = random_validation.get("mae")
            if random_mae_value is None:
                random_mae_value = random_ridge.get("mae")
            random_mae = (
                float(random_mae_value) if random_mae_value is not None else None
            )
            validation_count = ridge_validation.get("count")
            if validation_count is None:
                validation_count = ridge_validation.get("validation_count")
            if validation_count is None:
                validation_count = ridge.get("validation_count", "—")
            candidate_probe = item.get("candidate_pool_probe", {})
            real_future = candidate_probe.get("real_future_latent", {})
            predicted_future = candidate_probe.get("predicted_future_latent", {})
            real_future_mae = (
                float(real_future["mae"])
                if isinstance(real_future, Mapping) and real_future.get("mae") is not None
                else None
            )
            predicted_future_mae = (
                float(predicted_future["mae"])
                if isinstance(predicted_future, Mapping) and predicted_future.get("mae") is not None
                else None
            )
            probe_lines.append(
                f"| {item.get('task')} | {validation_count} | "
                f"{'—' if ridge_mae is None else format(ridge_mae, '.4f')} | "
                f"{'—' if random_mae is None else format(random_mae, '.4f')} | "
                f"{'—' if mlp_mae is None else format(mlp_mae, '.4f')} | "
                f"{'—' if real_future_mae is None else format(real_future_mae, '.4f')} | "
                f"{'—' if predicted_future_mae is None else format(predicted_future_mae, '.4f')} |"
            )
    if len(probe_lines) == 2:
        probe_lines.append("| — | — | — | — | — | — | — |")

    candidate_lines = [
        "| task | S | oracle success | B-selected success [95% CI] | random success [95% CI] | B−random success [paired 95% CI] | selection regret [95% CI] | cost-distance correlation [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    control_lines = [
        "| task | control kind | states | success rate | mean normalized physical distance |",
        "|---|---|---:|---:|---:|",
    ]
    guidance_lines = [
        "| task | guidance | variants | physical paired advantage, mean [mean config CI] | predicted improvement | true latent improvement | true physical improvement | model exploitation |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    milestone_lines = [
        "| task | guidance | paired rows | unpaired rows | step 5 | step 10 | step 15 | step 20 | step 25 | paired rows missing step |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    guidance_comparison_lines = [
        "| task | matched configurations | matched states | pooled-milestone GF−PO physical improvement (exploratory) [state-clustered 95% CI] |",
        "|---|---:|---:|---:|",
    ]
    guidance_milestone_comparison_lines = [
        "| task | common physical step | matched configurations | matched states | GF−PO true physical improvement [state-clustered 95% CI] |",
        "|---|---:|---:|---:|---:|",
    ]
    guidance_config_comparison_lines = [
        "| task | S | candidate | K | η | R | states | pooled-milestone GF−PO physical improvement (exploratory) [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in PHASE15_TASKS:
        summary_path = output_root / "diagnostics" / f"summary_{task}.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        milestone_summary = summary.get("guidance_milestone_steps", {})
        for mode in ("post_opt", "guided_flow"):
            milestone = milestone_summary.get(mode, {})
            step_counts = milestone.get("step_counts", {})
            milestone_lines.append(
                f"| {task} | {mode} | {milestone.get('paired_records', '—')} | "
                f"{milestone.get('unpaired_records', '—')} | "
                + " | ".join(str(step_counts.get(str(step), "—")) for step in (5, 10, 15, 20, 25))
                + f" | {milestone.get('paired_records_missing_step', '—')} |"
            )
        matched_comparison = summary.get("matched_guidance_comparison", {})
        if not isinstance(matched_comparison, Mapping):
            matched_comparison = {}
        matched_bootstrap = matched_comparison.get("bootstrap", {})
        matched_metric = (
            matched_bootstrap.get("gf_minus_po_physical_improvement", {})
            if isinstance(matched_bootstrap, Mapping)
            else {}
        )
        matched_interval = matched_metric.get("ci95") if isinstance(matched_metric, Mapping) else None
        matched_mean = matched_comparison.get("gf_minus_po_physical_improvement_mean")
        if matched_mean is None:
            comparison_text = "—"
        else:
            comparison_text = format(float(matched_mean), ".4f")
            if isinstance(matched_interval, Sequence) and len(matched_interval) == 2:
                comparison_text += (
                    " [" + format(float(matched_interval[0]), ".4f")
                    + ", " + format(float(matched_interval[1]), ".4f") + "]"
                )
        guidance_comparison_lines.append(
            f"| {task} | {matched_comparison.get('matched_configuration_count', 0)} | "
            f"{matched_comparison.get('matched_states', 0)} | {comparison_text} |"
        )
        milestone_comparisons = summary.get("matched_guidance_by_milestone", {})
        if isinstance(milestone_comparisons, Mapping):
            for step, step_comparison in sorted(
                milestone_comparisons.items(), key=lambda item: int(item[0])
            ):
                if not isinstance(step_comparison, Mapping):
                    continue
                step_bootstrap = step_comparison.get("bootstrap", {})
                step_metric = (
                    step_bootstrap.get("gf_minus_po_physical_improvement", {})
                    if isinstance(step_bootstrap, Mapping)
                    else {}
                )
                step_interval = (
                    step_metric.get("ci95") if isinstance(step_metric, Mapping) else None
                )
                step_mean = step_comparison.get("gf_minus_po_physical_improvement_mean")
                step_text = "—" if step_mean is None else format(float(step_mean), ".4f")
                if isinstance(step_interval, Sequence) and len(step_interval) == 2:
                    step_text += (
                        " [" + format(float(step_interval[0]), ".4f")
                        + ", " + format(float(step_interval[1]), ".4f") + "]"
                    )
                guidance_milestone_comparison_lines.append(
                    f"| {task} | {step} | "
                    f"{step_comparison.get('matched_configuration_count', 0)} | "
                    f"{step_comparison.get('matched_states', 0)} | {step_text} |"
                )
        for config_comparison in matched_comparison.get("matched_configurations", ()):
            if not isinstance(config_comparison, Mapping):
                continue
            config_bootstrap = config_comparison.get("bootstrap", {})
            config_metric = (
                config_bootstrap.get("gf_minus_po_physical_improvement", {})
                if isinstance(config_bootstrap, Mapping)
                else {}
            )
            config_interval = (
                config_metric.get("ci95") if isinstance(config_metric, Mapping) else None
            )
            config_mean = config_comparison.get("gf_minus_po_physical_improvement_mean")
            config_text = "—" if config_mean is None else format(float(config_mean), ".4f")
            if isinstance(config_interval, Sequence) and len(config_interval) == 2:
                config_text += (
                    " [" + format(float(config_interval[0]), ".4f")
                    + ", " + format(float(config_interval[1]), ".4f") + "]"
                )
            guidance_config_comparison_lines.append(
                f"| {task} | {config_comparison.get('flow_steps', '—')} | "
                f"{config_comparison.get('candidate_index', '—')} | "
                f"{config_comparison.get('guidance_inner_steps', '—')} | "
                f"{config_comparison.get('guidance_step_size', '—')} | "
                f"{config_comparison.get('max_rms_offset', '—')} | "
                f"{config_comparison.get('matched_states', '—')} | {config_text} |"
            )
        pool = summary.get("candidate_pool", {})
        for flow, metrics in sorted(
            pool.get("by_flow_steps", {}).items(), key=lambda item: int(item[0])
        ):
            bootstrap = metrics.get("bootstrap", {})
            selected_ci = bootstrap.get("selected_success", {}).get("ci95")
            random_ci = bootstrap.get("random_success_rate", {}).get("ci95")
            paired_gain_ci = bootstrap.get("selected_minus_random_success_rate", {}).get("ci95")
            regret_ci = bootstrap.get("selection_regret", {}).get("ci95")
            correlation_ci = bootstrap.get("predicted_true_distance_correlation", {}).get("ci95")
            selected_text = format(float(metrics["selected_success_rate"]), ".3f")
            random_text = format(float(metrics["random_success_rate"]), ".3f")
            paired_gain = metrics.get("selected_minus_random_success_rate")
            paired_gain_text = "—" if paired_gain is None else format(float(paired_gain), ".3f")
            regret_text = format(float(metrics["selection_regret"]), ".3f")
            correlation = metrics.get("predicted_true_distance_correlation")
            correlation_text = "undefined" if correlation is None else format(float(correlation), ".3f")
            if isinstance(selected_ci, Sequence) and len(selected_ci) == 2:
                selected_text += (
                    " [" + format(float(selected_ci[0]), ".3f")
                    + ", " + format(float(selected_ci[1]), ".3f") + "]"
                )
            if isinstance(regret_ci, Sequence) and len(regret_ci) == 2:
                regret_text += (
                    " [" + format(float(regret_ci[0]), ".3f")
                    + ", " + format(float(regret_ci[1]), ".3f") + "]"
                )
            if isinstance(random_ci, Sequence) and len(random_ci) == 2:
                random_text += (
                    " [" + format(float(random_ci[0]), ".3f")
                    + ", " + format(float(random_ci[1]), ".3f") + "]"
                )
            if isinstance(paired_gain_ci, Sequence) and len(paired_gain_ci) == 2:
                paired_gain_text += (
                    " [" + format(float(paired_gain_ci[0]), ".3f")
                    + ", " + format(float(paired_gain_ci[1]), ".3f") + "]"
                )
            if isinstance(correlation_ci, Sequence) and len(correlation_ci) == 2:
                correlation_text += (
                    " [" + format(float(correlation_ci[0]), ".3f")
                    + ", " + format(float(correlation_ci[1]), ".3f") + "]"
                )
            candidate_lines.append(
                f"| {task} | {flow} | {float(metrics['oracle_success_rate']):.3f} | "
                f"{selected_text} | "
                f"{random_text} | {paired_gain_text} | {regret_text} | {correlation_text} |"
            )
        controls = summary.get("controls", {})
        for kind, metrics in sorted(controls.get("kinds", {}).items()):
            control_lines.append(
                f"| {task} | {kind} | {metrics.get('states', '—')} | "
                f"{float(metrics['success_rate']):.3f} | "
                f"{float(metrics['mean_true_distance']):.3f} |"
            )
        by_mode: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
        for variant in summary.get("guidance_by_variant", ()):
            condition = variant.get("condition", {})
            paired = variant.get("paired_guidance", {})
            effect = variant.get("guidance", {})
            mode = condition.get("guidance") if isinstance(condition, Mapping) else None
            if mode in {"post_opt", "guided_flow"} and isinstance(paired, Mapping):
                by_mode.setdefault(str(mode), []).append(
                    (paired, effect if isinstance(effect, Mapping) else {})
                )
        for mode, variant_metrics in sorted(by_mode.items()):
            advantages = [
                float(paired["paired_advantage_mean"])
                for paired, _ in variant_metrics
                if paired.get("paired_advantage_mean") is not None
            ]
            ci_lowers = [
                float(paired["bootstrap"]["paired_advantage"]["ci95"][0])
                for paired, _ in variant_metrics
                if isinstance(paired.get("bootstrap", {}).get("paired_advantage"), Mapping)
            ]
            ci_uppers = [
                float(paired["bootstrap"]["paired_advantage"]["ci95"][1])
                for paired, _ in variant_metrics
                if isinstance(paired.get("bootstrap", {}).get("paired_advantage"), Mapping)
            ]
            predicted_improvements = [
                float(effect["predicted_improvement_mean"])
                for _, effect in variant_metrics
                if effect.get("predicted_improvement_mean") is not None
            ]
            latent_improvements = [
                float(effect["true_latent_improvement_mean"])
                for _, effect in variant_metrics
                if effect.get("true_latent_improvement_mean") is not None
            ]
            physical_improvements = [
                float(effect["true_physical_improvement_mean"])
                for _, effect in variant_metrics
                if effect.get("true_physical_improvement_mean") is not None
            ]
            exploitations = [
                float(effect["model_exploitation_fraction"])
                for _, effect in variant_metrics
                if effect.get("model_exploitation_fraction") is not None
            ]
            advantage_text = "—" if not advantages else format(float(np.mean(advantages)), ".4f")
            if ci_lowers and ci_uppers:
                advantage_text += (
                    " [" + format(float(np.mean(ci_lowers)), ".4f")
                    + ", " + format(float(np.mean(ci_uppers)), ".4f") + "]"
                )
            predicted_text = "—" if not predicted_improvements else format(float(np.mean(predicted_improvements)), ".4f")
            latent_text = "—" if not latent_improvements else format(float(np.mean(latent_improvements)), ".4f")
            physical_text = "—" if not physical_improvements else format(float(np.mean(physical_improvements)), ".4f")
            exploitation_text = "—" if not exploitations else format(float(np.mean(exploitations)), ".3f")
            guidance_lines.append(
                f"| {task} | {mode} | {len(variant_metrics)} | {advantage_text} | "
                f"{predicted_text} | {latent_text} | {physical_text} | {exploitation_text} |"
            )
    if len(candidate_lines) == 2:
        candidate_lines.append("| — | — | — | — | — | — | — | — |")
    if len(control_lines) == 2:
        control_lines.append("| — | — | — | — | — |")
    if len(guidance_lines) == 2:
        guidance_lines.append("| — | — | — | — | — | — | — | — |")
    if len(milestone_lines) == 2:
        milestone_lines.append("| — | — | — | — | — | — | — | — | — | — |")
    if len(guidance_comparison_lines) == 2:
        guidance_comparison_lines.append("| — | — | — | — |")
    if len(guidance_milestone_comparison_lines) == 2:
        guidance_milestone_comparison_lines.append("| — | — | — | — | — |")
    if len(guidance_config_comparison_lines) == 2:
        guidance_config_comparison_lines.append("| — | — | — | — | — | — | — | — |")

    guidance_timing_lines = [
        "| matched config | GF−PO mean batch1 p50 (s) | tasks with GF faster | all four faster |",
        "|---|---:|---:|---|",
    ]
    guidance_method_path = output_root / "analysis" / "method_selection.json"
    if guidance_method_path.is_file():
        guidance_method = json.loads(guidance_method_path.read_text(encoding="utf-8"))
        for comparison in guidance_method.get("guidance_mode_efficiency_comparisons", ()):
            if not isinstance(comparison, Mapping):
                continue
            matched_config = comparison.get("matched_config", {})
            if not isinstance(matched_config, Mapping):
                matched_config = {}
            config_text = (
                f"{matched_config.get('mode', '—')}, S={matched_config.get('flow_steps', '—')}, "
                f"N={matched_config.get('candidate_count', '—')}, K={matched_config.get('po_iterations', '—')}"
            )
            guidance_timing_lines.append(
                f"| {config_text} | "
                f"{float(comparison['guided_flow_minus_post_opt_mean_batch1_p50_seconds']):+.4f} | "
                f"{comparison.get('guided_flow_faster_task_count', 0)}/4 | "
                f"{'是' if comparison.get('guided_flow_faster_all_tasks') else '否'} |"
            )
    if len(guidance_timing_lines) == 2:
        guidance_timing_lines.append("| — | — | — | — |")

    method_lines = [
        "| role | family / mode | aggregate success | batch1 p50 / p95 (s) | batch50 throughput (obs/s) |",
        "|---|---|---:|---:|---:|",
    ]
    timing_detail_lines = [
        "| role | method | success | batch1 p50 / p95 (s) | batch50 throughput (obs/s) | A / B / B-backward calls per decision | max batch1 allocated memory (GiB) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    pareto_lines = [
        "| method | aggregate success | batch1 p50 (s) |",
        "|---|---:|---:|",
    ]
    relative_lines: list[str] = []
    method_path = output_root / "analysis" / "method_selection.json"
    if method_path.is_file():
        method_selection = json.loads(method_path.read_text(encoding="utf-8"))
        method_candidates = [
            item for item in method_selection.get("candidates", ())
            if isinstance(item, Mapping)
        ]
        by_id = {str(item.get("candidate_id")): item for item in method_candidates}
        descriptions: dict[str, str] = {}

        def method_description(item: Mapping[str, Any]) -> str:
            method_config = item.get("config", {})
            return (
                f"{method_config.get('family', '—')}/{method_config.get('mode', '—')}, "
                f"S={method_config.get('flow_steps', '—')}, "
                f"N={method_config.get('candidate_count', '—')}, "
                f"K={method_config.get('po_iterations', '—')}"
            )

        cem_references: dict[str, Mapping[str, Any]] = {}
        for cem_mode in ("P1", "P2"):
            choices = [
                item for item in method_candidates
                if item.get("config", {}).get("mode") == cem_mode
            ]
            if choices:
                cem_references[cem_mode] = max(
                    choices,
                    key=lambda item: (
                        float(item.get("mean_success_rate", 0.0)),
                        -float(item.get("mean_batch1_p50_seconds", float("inf"))),
                        str(item.get("candidate_id", "")),
                    ),
                )
        for role, key in (
            ("主推方案", "main_candidate_id"),
            ("Pareto 备选", "backup_candidate_id"),
        ):
            item = by_id.get(str(method_selection.get(key)))
            if item is None:
                method_lines.append(f"| {role} | 尚未收敛 | — | — | — |")
                continue
            description = method_description(item)
            descriptions[str(item.get("candidate_id"))] = description
            method_lines.append(
                f"| {role} | {description} | {float(item['mean_success_rate']):.3f} | "
                f"{float(item['mean_batch1_p50_seconds']):.4f} / "
                f"{float(item['mean_batch1_p95_seconds']):.4f} | "
                f"{float(item['mean_batch50_throughput_per_second']):.1f} |"
            )
            peak_bytes = item.get("max_batch1_peak_memory_bytes")
            peak_gib = "—" if peak_bytes is None else format(float(peak_bytes) / (1024 ** 3), ".2f")
            call_counts = " / ".join(
                format(float(item.get(field, 0.0)), ".1f")
                for field in (
                    "mean_stage_a_forward_calls_per_decision",
                    "mean_stage_b_forward_calls_per_decision",
                    "mean_guidance_backward_calls_per_decision",
                )
            )
            timing_detail_lines.append(
                f"| {role} | {description} | {float(item['mean_success_rate']):.3f} | "
                f"{float(item['mean_batch1_p50_seconds']):.4f} / "
                f"{float(item['mean_batch1_p95_seconds']):.4f} | "
                f"{float(item['mean_batch50_throughput_per_second']):.1f} | "
                f"{call_counts} | {peak_gib} |"
            )
            for cem_mode, reference in cem_references.items():
                reference_latency = float(reference["mean_batch1_p50_seconds"])
                selected_latency = float(item["mean_batch1_p50_seconds"])
                speedup = reference_latency / selected_latency if selected_latency > 0 else float("nan")
                success_gap = 100.0 * (
                    float(item["mean_success_rate"])
                    - float(reference["mean_success_rate"])
                )
                relative_lines.append(
                    f"{role} 相对成功率最高的 {cem_mode} CEM 配置（{method_description(reference)}）："
                    f"成功率差 {success_gap:+.1f} 个百分点，batch1 p50 延迟加速 {speedup:.2f}×。"
                )
        for cem_mode, reference in cem_references.items():
            description = method_description(reference)
            descriptions[str(reference.get("candidate_id"))] = description
            peak_bytes = reference.get("max_batch1_peak_memory_bytes")
            peak_gib = "—" if peak_bytes is None else format(float(peak_bytes) / (1024 ** 3), ".2f")
            call_counts = " / ".join(
                format(float(reference.get(field, 0.0)), ".1f")
                for field in (
                    "mean_stage_a_forward_calls_per_decision",
                    "mean_stage_b_forward_calls_per_decision",
                    "mean_guidance_backward_calls_per_decision",
                )
            )
            timing_detail_lines.append(
                f"| {cem_mode} CEM 成功率参照 | {description} | "
                f"{float(reference['mean_success_rate']):.3f} | "
                f"{float(reference['mean_batch1_p50_seconds']):.4f} / "
                f"{float(reference['mean_batch1_p95_seconds']):.4f} | "
                f"{float(reference['mean_batch50_throughput_per_second']):.1f} | "
                f"{call_counts} | {peak_gib} |"
            )
        pareto_candidates = sorted(
            (item for item in method_candidates if item.get("method_pareto")),
            key=lambda item: (
                -float(item.get("mean_success_rate", 0.0)),
                float(item.get("mean_batch1_p50_seconds", float("inf"))),
            ),
        )
        for item in pareto_candidates[:20]:
            description = descriptions.get(
                str(item.get("candidate_id")), method_description(item)
            )
            pareto_lines.append(
                f"| {description} | {float(item['mean_success_rate']):.3f} | "
                f"{float(item['mean_batch1_p50_seconds']):.4f} |"
            )
        if len(pareto_lines) == 2:
            pareto_lines.append("| — | — | — |")
    else:
        method_lines.append("| 主推方案 / Pareto 备选 | 精测完成后选择 | — | — | — |")
        timing_detail_lines.append("| — | 细测完成后生成 | — | — | — | — | — |")
        pareto_lines.append("| 精测完成后生成 | — | — |")

    if counts.get("pending", 0) or counts.get("failed", 0):
        decisions = (
            "本报告仍处于扫描阶段，条件尚未齐全；P3、PO/GF、CEM 的最终取舍暂记为尚未收敛。"
        )
    else:
        decisions = "\n".join(_diagnostic_decisions(output_root, rows))
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
            f"- 历史复用审计：{json.dumps(history_audit, ensure_ascii=False, sort_keys=True)}",
            "",
            "## 主扫描验收",
            "",
            "| group | planned | indexed | pending |",
            "|---|---:|---:|---:|",
            *group_rows,
            "",
            f"主网格 {len(primary_condition_specs())} 条条件；固定稳定性 {fixed_count} 条、自适应稳定性 {adaptive_count} 条，"
            f"合计 {len(rows)} 条。每个完成条件覆盖原 50 个 cohort episode。",
            "历史结果只有在 checkpoint、legacy cohort、normalizer、动作裁剪、精度、随机数和候选生成语义都一致时才复用；缺轨迹历史结果只进入 success-only 统计。",
            (
                "历史只读审计另外逐条件核验 checkpoint、cohort 与执行语义。"
                f"其中 {verified_history_audit.get('compatible_full_reuse_count', 0)} 条可完整复用，"
                f"{verified_history_audit.get('compatible_success_only_count', 0)} 条仅可复用成功率；"
                "已存在的 current 结果仍标记为 current，不计入历史复用。"
                if verified_history_audit is not None
                else "历史只读审计尚未生成。"
            ),
            "",
            "## Seed 42 主扫描最高成功率条件（描述性）",
            "",
            *top_lines,
            "",
            "## 跨任务方法选择",
            "",
            "候选来自固定稳定性网格与在四任务均可配对的自适应配置。主方案按三个推理 seed、四任务平均成功率选择；距最高值不超过 2 个百分点时优先选精测延迟更低者，并要求任何任务不低于该任务最佳配置 4 个百分点以上。另保留一个成功率—延迟 Pareto 备选。",
            "",
            *method_lines,
            "",
            "### 精测延迟、吞吐与模型调用",
            "",
            "调用数按一次 replanning 计；列顺序为 Stage A 前向、Stage B 前向、Stage B 引导反向。峰值为 batch1 的 PyTorch 已分配显存最大值。CEM 的 B 前向按每轮、每个 solver batch 的评分调用计；P2 warm start 计入 A/B/反向调用。",
            "",
            *timing_detail_lines,
            "",
            *relative_lines,
            "",
            "### 跨任务成功率—延迟 Pareto 候选",
            "",
            *pareto_lines,
            "",
            "## 诊断与决策",
            "",
            "### 物理 probe",
            "",
            "按轨迹拆分 train/validation，并排除评测轨迹。下表误差使用真实未来图像编码；B 预测 future latent 的读出误差另存于 probe JSON。",
            "",
            *probe_lines,
            "",
            "### 固定候选池",
            "",
            "成功率和 selection regret 先按状态计算，置信区间按状态聚类 bootstrap。",
            "",
            *candidate_lines,
            "各 S 单独汇报，以免跨 S 汇总掩盖提案分布差异。oracle 反映 A 候选池的覆盖，B-selected 与随机选择的配对差更直接反映 B 的排序增益；两者需结合解读。",
            "",
            "### 固定控制动作",
            "",
            "物理零动作和归一化零动作作为不同控制分别统计；方向扰动按固定随机种子生成。",
            "",
            *control_lines,
            "",
            "### 同状态配对控制效应",
            "",
            "控制效应为控制动作减去同一状态三个 anchor 的平均结果；该合成基线不是实际执行策略。成功率正值表示控制成功更多，距离正值表示控制距离更差。成功率按 `legacy_50` 全部状态配对；距离使用双方共同有效的 raw milestone（5/10/15/20/25）。95% 区间以状态为单位进行 10,000 次 bootstrap。每个动作/状态只有一次记录，因此区间不包含 rollout 随机性；固定方向与控制组比较属于探索性分析，未做多重比较校正。",
            "",
            *control_paired_lines,
            "",
            "### Cube 控制池迁移审计",
            "",
            *control_migration_lines,
            "",
            "### 配对梯度修正",
            "",
            "Guidance 成功率沿用评估器记录的 `episode_successes`，表示 `eval_budget=50` 内出现成功事件；它不等同于统一第 25 个 primitive step 的终态成功率。物理距离改变量单独按每个起点最后一个共同有效 milestone（5/10/15/20/25）配对。两类指标分别解读，并同时报告共同 milestone 的步数分布。",
            "",
            "每项修正与相同 RMS 位移的随机方向对照，按共同有效 milestone 配对；没有共同 milestone 的分支不进入配对均值。",
            "",
            *guidance_lines,
            "",
            "### 共同物理 milestone 分布",
            "",
            "下表计数单位是 Guidance 变体×起点记录；同一起点在多个配置中会重复出现，不能当作独立样本数。配对距离取每条记录最后一个共同有效 milestone。",
            "",
            *milestone_lines,
            "",
            "下面的 pooled-milestone GF−PO 表按相同配置和起点匹配，但每条记录使用其最后一个共同有效步数；由于步数分布可能不同，该汇总和逐配置表仅作探索性描述。正数表示 GF 改善更多，区间按起点聚类。",
            "",
            *guidance_comparison_lines,
            "",
            *guidance_config_comparison_lines,
            "",
            "主要物理比较按完全相同的 `physical_comparison_step` 分层，再匹配 flow steps、候选索引、K、η、R 和起点真实物理代价。该步数是 baseline、guided、random 三条分支最后共同有效的 milestone，受终止与结果可用性影响；分层由结果后的可观测长度决定，各步可能对应不同起点/配置子集。因此这些差值是描述性诊断，不能解释为同一批起点随时间的变化，也不是全50起点的因果效应。配置在起点内先平均，bootstrap 以起点为单位；跨步数/配置检视未做多重比较校正。正数表示 GF 改善更多。",
            "",
            *guidance_milestone_comparison_lines,
            "",
            "计时只纳入四任务精测均无干扰的匹配配置，延迟差为负表示 GF 更快。",
            "",
            *guidance_timing_lines,
            "",
            "### 公平计时",
            "",
            f"完整策略推理在固定真实观测上同步计时，包含编码、提案、评分/梯度、CEM 更新和动作输出；环境运行时间单列。明细：`{output_root / 'analysis' / 'timing.json'}`。",
            "单环境报告 batch1 p50/p95；并行报告 batch50 延迟和吞吐。细测测量窗口前后记录 GPU 外部计算进程，存在干扰的窗口应排除主要加速结论。",
            "",
            *coarse_timing_sensitivity_lines,
            "",
            f"{decisions}",
            "",
            "## 产物",
            "",
            f"- 索引：`{output_root / 'analysis' / 'index.json'}`",
            f"- 历史只读审计：`{output_root / 'analysis' / 'history_reuse_audit.json'}`",
            f"- 条件 CSV：`{output_root / 'analysis' / 'conditions.csv'}`",
            f"- 计时：`{output_root / 'analysis' / 'timing.json'}`",
            f"- 细测选择：`{output_root / 'analysis' / 'fine_timing_selection.json'}`",
            f"- 跨任务方法选择：`{output_root / 'analysis' / 'method_selection.json'}`",
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
    verified_history_audit = _build_verified_history_audit(output_root, config)
    history_audit = _history_reuse_audit(index_payload, verified_history_audit)
    adaptive_count = len(_load_adaptive_specs(output_root))
    method_selection_path = output_root / "analysis" / "method_selection.json"
    method_selection = (
        json.loads(method_selection_path.read_text(encoding="utf-8"))
        if method_selection_path.is_file()
        else None
    )
    diagnostic_summaries = {
        task: str(output_root / "diagnostics" / f"summary_{task}.json")
        for task in PHASE15_TASKS
        if (output_root / "diagnostics" / f"summary_{task}.json").is_file()
    }
    analysis = {
        "schema_version": "round5_phase1_5_analysis_v1",
        "experiment": "Round 5 Phase 1.5",
        "config": str(_resolve(args.config)),
        "output_root": str(output_root),
        "primary_conditions": 2424,
        "fixed_stability_conditions": len(sampling_stability_specs()),
        "adaptive_stability_conditions": adaptive_count,
        "stability_conditions": len(sampling_stability_specs()) + adaptive_count,
        "rows": rows,
        "counts": index_payload["counts"],
        "history_reuse_audit": history_audit,
        "verified_history_reuse_audit": str(
            output_root / "analysis" / "history_reuse_audit.json"
        ),
        "bootstrap": {"samples": PHASE15_BOOTSTRAP_SAMPLES, "cluster": "state"},
        "diagnostic_summaries": diagnostic_summaries,
        "method_selection": method_selection,
        "decision_status": (
            "pending"
            if index_payload["counts"].get("pending", 0)
            or index_payload["counts"].get("failed", 0)
            else "ready_for_diagnostic_review"
        ),
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
            verified_history_audit=verified_history_audit,
        ),
        encoding="utf-8",
    )
    result = {"analysis": str(analysis_path), "report": str(report_path), "conditions": len(rows), "counts": index_payload["counts"]}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def timing(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    """Measure complete policy inference on fixed real cohort observations."""
    output_root = _resolve(args.output_root)
    if args.retry_interfered_fine and not args.fine:
        raise ValueError("--retry-interfered-fine requires --fine")
    specs = _selected_specs(args, include_stability=args.include_stability)
    if args.include_adaptive_stability:
        specs.extend(_load_adaptive_specs(output_root))
    if not specs:
        raise ValueError("no Phase1.5 conditions selected for timing")
    _configure_device(
        args.device,
        args.gpu,
        args.min_free_mib,
        args.max_load_per_cpu,
        args.min_available_mib,
        args.min_swap_free_mib,
    )
    manifests = _load_manifests(config)
    checkpoints, checkpoint_hashes = _checkpoint_paths(config)
    fine_selection_ids: set[str] | None = None
    fine_selection_path = output_root / "analysis" / "fine_timing_selection.json"
    if args.fine:
        if not fine_selection_path.is_file():
            raise FileNotFoundError(
                "fine timing selection is missing; run select-fine-timing after all coarse windows complete: "
                f"{fine_selection_path}"
            )
        selection = json.loads(fine_selection_path.read_text(encoding="utf-8"))
        fine_selection_ids = {
            str(item["condition_id"])
            for item in selection.get("selected_conditions", ())
            if isinstance(item, Mapping) and item.get("condition_id") is not None
        }
        if not fine_selection_ids:
            raise ValueError(f"fine timing selection contains no conditions: {fine_selection_path}")
    timing_root = output_root / "analysis" / "timing_conditions"
    timing_root.mkdir(parents=True, exist_ok=True)
    window_name = "fine_timing" if args.fine else "coarse_timing"
    warmup, runs = (20, 100) if args.fine else (5, 10)
    records: list[dict[str, Any]] = []
    loaded_models: dict[str, Any] = {}

    for spec in specs:
        task = str(spec["task"])
        checkpoint = checkpoints[task]
        manifest = manifests[task]
        identity = condition_identity(
            spec,
            checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_hashes[task],
            cohort=manifest,
        )
        current_id = condition_id(identity)
        if fine_selection_ids is not None and current_id not in fine_selection_ids:
            continue
        target = timing_root / f"{current_id}.json"
        with condition_lock(target):
            record = (
                json.loads(target.read_text(encoding="utf-8"))
                if target.is_file()
                else {
                    "schema_version": "round5_phase1_5_timing_condition_v2",
                    "condition_id": current_id,
                    "condition_identity": identity,
                    "spec": dict(spec),
                    "task": task,
                    "coarse_timing": None,
                    "fine_timing": None,
                }
            )
            existing_window = record.get(window_name)
            if not isinstance(existing_window, Mapping):
                existing_window = {}
            if existing_window and (
                int(existing_window.get("warmup", -1)) != warmup
                or int(existing_window.get("runs", -1)) != runs
            ):
                raise ValueError(
                    f"stored timing window has different run counts for {current_id}"
                )
            if args.fine and args.retry_interfered_fine and existing_window:
                statuses = _fine_timing_retry_status(existing_window)
                if statuses is not None:
                    def summarize_attempt(batch: Mapping[str, Any] | None) -> dict[str, Any] | None:
                        if not isinstance(batch, Mapping):
                            return None
                        return {
                            "p50_seconds": batch.get("p50_seconds"),
                            "p95_seconds": batch.get("p95_seconds"),
                            "gpu_interference": batch.get("gpu_interference"),
                        }

                    record.setdefault("fine_timing_retries", []).append(
                        {
                            "reason": "external_compute_observed_or_missing_boundary_snapshot",
                            "previous_batch_statuses": statuses,
                            "previous_batches": {
                                batch_name: summarize_attempt(existing_window.get(batch_name))
                                for batch_name in ("batch1", "batch50")
                            },
                        }
                    )
                    record["fine_timing"] = None
                    atomic_write_json(target, record)
                    existing_window = {}
            if all(name in existing_window for name in ("batch1", "batch50")):
                records.append(record)
                continue
            if args.fine and not all(
                name in (record.get("coarse_timing") or {})
                for name in ("batch1", "batch50")
            ):
                raise ValueError(
                    f"fine timing requires a completed coarse window for {current_id}"
                )
            if task not in loaded_models:
                model, resolved = load_policy_or_model(str(checkpoint))
                if resolved is not None and Path(resolved).resolve() != checkpoint.resolve():
                    raise ValueError(f"checkpoint resolver changed requested path: {checkpoint}")
                loaded_models[task] = model
            model = loaded_models[task]

            mode = str(spec["mode"])
            guidance = config["guidance"]
            candidate_count = int(
                spec.get("candidate_count") or (1 if mode == "P0" else 64)
            )
            if mode in {"P1", "P2"}:
                candidate_count = int(spec.get("cem_num_samples") or 300)
            inner_steps = int(
                spec.get("po_iterations") or guidance.get("default_inner_steps", 5)
            )
            last_steps = int(
                spec.get("guidance_last_steps") or guidance.get("default_last_steps", 5)
            )
            step_size = float(
                spec.get("guidance_step_size") or guidance.get("default_step_size", 0.01)
            )
            max_rms_offset = float(
                spec.get("max_rms_offset")
                or guidance.get("default_max_rms_offset", 0.2)
            )
            window_results: dict[str, Any] = dict(existing_window)
            for batch_name, batch_size in (("batch1", 1), ("batch50", 50)):
                if batch_name in window_results:
                    continue
                cfg = _compose(task, spec, args.device)
                cfg.eval.num_eval = batch_size
                cfg.world.num_envs = batch_size
                batch_manifest = manifest
                if batch_size == 1:
                    batch_manifest = replace(
                        manifest,
                        cohort_id=f"{manifest.cohort_id}_timing1",
                        cohort_kind="custom",
                        entries=(manifest.entries[0],),
                        episode_split={
                            **manifest.episode_split,
                            "selected": (manifest.entries[0].episode_id,),
                        },
                    )
                    batch_manifest = replace(
                        batch_manifest,
                        cohort_sha256=batch_manifest.computed_sha256,
                    )

                def capture_timing(policy, *call_args, _batch_size=batch_size, **call_kwargs):
                    info = call_kwargs.get("info_dict")
                    if info is None and call_args:
                        info = call_args[0]
                    if not isinstance(info, Mapping):
                        raise TypeError("timing capture expected a real policy info mapping")
                    environment_batch_size = int(
                        getattr(getattr(policy, "env", None), "num_envs", _batch_size)
                    )

                    def infer_fixed_input():
                        replay_info = dict(info)
                        replay_info["_needs_flush"] = np.ones(
                            environment_batch_size, dtype=bool
                        )
                        return policy.get_action(replay_info)

                    if str(args.device).startswith("cuda"):
                        import torch

                        synchronize = lambda: torch.cuda.synchronize(args.device)
                    else:
                        synchronize = None
                    measured = synchronous_timing(
                        infer_fixed_input,
                        warmup=warmup,
                        runs=runs,
                        synchronize=synchronize,
                    )
                    events = list(getattr(policy, "planning_events", ()))
                    timed_events = events[-runs:]
                    if len(timed_events) != runs:
                        raise RuntimeError(
                            "policy did not record one planning event per timed inference"
                        )
                    if mode == "P0":
                        a_forward_count = sum(
                            int(
                                event.get("guidance_stats", {}).get(
                                    "stage_a_forward_count", event.get("forward_count", 0)
                                )
                            )
                            for event in timed_events
                        )
                        b_forward_count = sum(
                            int(
                                event.get("guidance_stats", {}).get(
                                    "stage_b_forward_count", 0
                                )
                            )
                            for event in timed_events
                        )
                    elif mode in {"P1", "P2"}:
                        a_forward_count = sum(
                            int(event.get("stage_a_forward_count", 0))
                            for event in timed_events
                        )
                        b_forward_count = sum(
                            int(
                                event.get(
                                    "stage_b_forward_count",
                                    event.get("forward_count", 0),
                                )
                            )
                            for event in timed_events
                        )
                    else:
                        a_forward_count = sum(
                            int(
                                event.get(
                                    "stage_a_forward_count",
                                    event.get("proposal_forward_count", 0),
                                )
                            )
                            for event in timed_events
                        )
                        b_forward_count = sum(
                            int(event.get("stage_b_forward_count", 0))
                            + int(event.get("verifier_forward_count", 0))
                            for event in timed_events
                        )
                    backward_count = sum(
                        int(
                            event.get(
                                "guidance_backward_count",
                                event.get("guidance_stats", {}).get("backward_count", 0),
                            )
                        )
                        for event in timed_events
                    )
                    peak_values = [
                        int(event["peak_memory_bytes"])
                        for event in timed_events
                        if event.get("peak_memory_bytes") is not None
                    ]
                    return {
                        **measured,
                        "environment_batch_size": environment_batch_size,
                        "throughput_per_second": float(
                            environment_batch_size / measured["mean_seconds"]
                        ),
                        "batch50_throughput_per_second": (
                            float(50.0 / measured["mean_seconds"])
                            if environment_batch_size == 50
                            else None
                        ),
                        "a_forward_count": int(a_forward_count),
                        "b_forward_count": int(b_forward_count),
                        "forward_count": int(a_forward_count + b_forward_count),
                        "guidance_backward_count": int(backward_count),
                        "peak_memory_bytes": max(peak_values) if peak_values else None,
                        "timed_planning_events": len(timed_events),
                        "planning_event_count_source": (
                            "policy_model_counters_and_exact_cem_iteration_count"
                            if mode in {"P1", "P2"}
                            else "policy_planning_events"
                        ),
                        "forward_count_is_exact": mode in {"P0", "P1", "P2", "P3"},
                        "forward_count_scope": (
                            "CEM B iteration calls plus P2 actor A/B and guidance counters"
                            if mode in {"P1", "P2"}
                            else "model-native proposal and verifier call counters"
                        ),
                    }

                timing_start = _gpu_compute_snapshot(args.gpu) if args.gpu else None
                result = run_round4_evaluation(
                    cfg,
                    task=task,
                    policy_or_model=model,
                    mode=mode,
                    identity=EvaluationIdentity(
                        entrypoint="round5_phase1_5_timing",
                        policy_kind="round4_shared_dit",
                        checkpoint=str(checkpoint.resolve()),
                        epoch=int(config["training"]["epoch"]),
                        stage=mode,
                    ),
                    manifest=batch_manifest,
                    output_dir=timing_root / "scratch" / current_id / batch_name,
                    device=args.device,
                    trace=False,
                    candidate_count=candidate_count,
                    flow_steps=int(spec.get("flow_steps") or 16),
                    action_flow_steps=spec.get("flow_steps") if mode != "P1" else None,
                    solver_batch_size=int(config["evaluation"]["solver_batch_size"]),
                    candidate_batch_size=int(config["evaluation"]["candidate_batch_size"]),
                    action_flow_integrator=str(config["evaluation"]["integrator"]),
                    action_bound_mode=str(spec["action_bound_mode"]),
                    cem_protocol=str(spec["cem_protocol"]),
                    guidance_mode=str(spec.get("guidance", "none")),
                    guidance_step_size=step_size,
                    guidance_last_steps=last_steps,
                    guidance_inner_steps=inner_steps,
                    guidance_max_rms_offset=max_rms_offset,
                    proposal_chunk_size=int(config["evaluation"]["proposal_chunk_size"]),
                    allowed_protocol_variants=("legacy",),
                    allow_variable_candidate_count=True,
                    allow_solver_config_override=True,
                    timing_capture_callback=capture_timing,
                )
                timing_end = _gpu_compute_snapshot(args.gpu) if args.gpu else None
                if result.get("status") != "timing_capture":
                    raise RuntimeError(
                        f"{task}/{current_id}/{batch_name} completed without a timing capture"
                    )
                measured = dict(result["timing_capture"])
                measured["gpu_interference"] = _timing_interference(
                    timing_start, timing_end, own_pid=os.getpid()
                )
                window_results[batch_name] = measured
                record[window_name] = {
                    "warmup": warmup,
                    "runs": runs,
                    "attempt": 1 + len(record.get("fine_timing_retries", ()))
                    if args.fine
                    else 1,
                    "source": "complete_policy_inference_on_fixed_real_cohort_observation",
                    "environment_timing_included": False,
                    **window_results,
                }
                atomic_write_json(target, record)
            records.append(record)
            print(
                json.dumps(
                    {
                        "condition_id": current_id,
                        "task": task,
                        "window": window_name,
                        "batch1_p50_seconds": window_results["batch1"]["p50_seconds"],
                        "batch50_p50_seconds": window_results["batch50"]["p50_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    all_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(timing_root.glob("*.json"))
    ]
    output = output_root / "analysis" / "timing.json"
    atomic_write_json(
        output,
        {
            "schema_version": "round5_phase1_5_timing_v2",
            "method": "synchronized complete-policy calls on fixed observations from the legacy_50 cohort",
            "environment_wall_clock_is_separate": True,
            "coarse_window": {"warmup": 5, "runs": 10},
            "fine_window": {"warmup": 20, "runs": 100},
            "fine_selection": str(fine_selection_path) if args.fine else None,
            "records": all_records,
            "conditions": len(all_records),
        },
    )
    result = {"timing": str(output), "conditions": len(records), "window": window_name}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def _gpu_compute_snapshot(gpu: str | None) -> dict[str, Any] | None:
    if gpu is None:
        return None
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [
        "nvidia-smi",
        "--id",
        str(gpu),
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    processes = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            processes.append({"pid": int(fields[0]), "used_gpu_memory_mib": int(fields[1])})
        except ValueError:
            continue
    return {"gpu": int(gpu), "compute_processes": processes}


def _timing_interference(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
    *,
    own_pid: int,
) -> dict[str, Any]:
    observed = set()
    for snapshot in (before, after):
        if snapshot is None:
            continue
        observed.update(
            int(item["pid"])
            for item in snapshot.get("compute_processes", ())
            if int(item.get("pid", -1)) != int(own_pid)
        )
    return {
        "external_compute_pids_at_window_boundaries": sorted(observed),
        "external_compute_process_observed": bool(observed),
        "status": "interference_flagged" if observed else "no_external_compute_observed_at_boundaries",
    }


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


def _coarse_success_latency_frontier(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return per-task non-dominated rows, maximizing success and minimizing latency."""
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task"]), []).append(row)

    frontier: list[dict[str, Any]] = []
    for task in sorted(by_task):
        candidates = by_task[task]
        # Identical observed points add no frontier information. Keep the one
        # requiring the fewest model calls, then use its stable ID as a tiebreak.
        by_point: dict[tuple[float, float], Mapping[str, Any]] = {}
        for row in candidates:
            point = (float(row["success_rate"]), float(row["batch1_p50_seconds"]))
            previous = by_point.get(point)
            key = (
                int(row.get("forward_count") or 0) + int(row.get("backward_count") or 0),
                str(row["condition_id"]),
            )
            if previous is None or key < (
                int(previous.get("forward_count") or 0) + int(previous.get("backward_count") or 0),
                str(previous["condition_id"]),
            ):
                by_point[point] = row

        points = list(by_point.values())
        for row in points:
            success = float(row["success_rate"])
            latency = float(row["batch1_p50_seconds"])
            dominated = any(
                float(other["success_rate"]) >= success
                and float(other["batch1_p50_seconds"]) <= latency
                and (
                    float(other["success_rate"]) > success
                    or float(other["batch1_p50_seconds"]) < latency
                )
                for other in points
            )
            if not dominated:
                frontier.append(dict(row))
    return sorted(
        frontier,
        key=lambda row: (
            str(row["task"]),
            float(row["batch1_p50_seconds"]),
            -float(row["success_rate"]),
            str(row["condition_id"]),
        ),
    )


_NO_EXTERNAL_COMPUTE_AT_BOUNDARIES = "no_external_compute_observed_at_boundaries"


def _timing_interference_status(batch: Mapping[str, Any] | None) -> str:
    if not isinstance(batch, Mapping):
        return "missing_snapshot"
    interference = batch.get("gpu_interference")
    if not isinstance(interference, Mapping):
        return "missing_snapshot"
    status = interference.get("status")
    return str(status) if status else "missing_snapshot"


def _fine_timing_retry_status(
    fine_window: Mapping[str, Any] | None,
) -> dict[str, str] | None:
    if not isinstance(fine_window, Mapping) or not fine_window:
        return None
    statuses = {
        batch_name: _timing_interference_status(fine_window.get(batch_name))
        for batch_name in ("batch1", "batch50")
    }
    if any(
        status != _NO_EXTERNAL_COMPUTE_AT_BOUNDARIES
        for status in statuses.values()
    ):
        return statuses
    return None


def _complete_coarse_timing(record: Mapping[str, Any]) -> bool:
    coarse = record.get("coarse_timing")
    if not isinstance(coarse, Mapping):
        return False
    try:
        if int(coarse.get("warmup", -1)) != 5 or int(coarse.get("runs", -1)) != 10:
            return False
        return all(
            isinstance(coarse.get(batch_name), Mapping)
            and int(coarse[batch_name].get("runs", -1)) == 10
            for batch_name in ("batch1", "batch50")
        )
    except (TypeError, ValueError):
        return False


def _complete_fine_timing(record: Mapping[str, Any]) -> bool:
    fine = record.get("fine_timing")
    if not isinstance(fine, Mapping):
        return False
    try:
        if int(fine.get("warmup", -1)) != 20 or int(fine.get("runs", -1)) != 100:
            return False
        for batch_name in ("batch1", "batch50"):
            batch = fine.get(batch_name)
            if not isinstance(batch, Mapping) or int(batch.get("runs", -1)) != 100:
                return False
            samples = batch.get("samples_seconds")
            if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)):
                if len(samples) != 100:
                    return False
        return True
    except (TypeError, ValueError):
        return False


def merge_timing_staging_roots(
    staging_roots: Sequence[str | Path],
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    """Persist completed coarse timing records from staging roots.

    A repeated condition is preserved as an additional attempt rather than
    replacing the first result. Interference status is combined
    conservatively, while the original timing samples remain the primary
    coarse window used for screening.
    """
    if not staging_roots:
        raise ValueError("merge-timing-staging requires at least one --staging-root")
    destination_root = Path(output_root) / "analysis" / "timing_conditions"
    destination_root.mkdir(parents=True, exist_ok=True)
    per_root: dict[str, dict[str, int]] = {}

    def combine_interference(
        primary: Mapping[str, Any],
        repeated: Mapping[str, Any],
    ) -> dict[str, Any]:
        primary_interference = primary.get("gpu_interference")
        repeated_interference = repeated.get("gpu_interference")
        primary_interference = (
            primary_interference if isinstance(primary_interference, Mapping) else {}
        )
        repeated_interference = (
            repeated_interference if isinstance(repeated_interference, Mapping) else {}
        )
        statuses = {
            _timing_interference_status(primary),
            _timing_interference_status(repeated),
        }
        pids: set[int] = set()
        for interference in (primary_interference, repeated_interference):
            for raw_pid in interference.get(
                "external_compute_pids_at_window_boundaries", ()
            ):
                try:
                    pids.add(int(raw_pid))
                except (TypeError, ValueError):
                    continue
        if "interference_flagged" in statuses:
            status = "interference_flagged"
        elif statuses == {_NO_EXTERNAL_COMPUTE_AT_BOUNDARIES}:
            status = _NO_EXTERNAL_COMPUTE_AT_BOUNDARIES
        else:
            status = "missing_snapshot"
        return {
            "external_compute_pids_at_window_boundaries": sorted(pids),
            "external_compute_process_observed": (
                bool(pids)
                or bool(primary_interference.get("external_compute_process_observed"))
                or bool(repeated_interference.get("external_compute_process_observed"))
                or status == "interference_flagged"
            ),
            "status": status,
        }

    for raw_root in staging_roots:
        source_root = Path(raw_root).expanduser().resolve()
        root_counts = {
            "copied": 0,
            "repeated": 0,
            "identical": 0,
            "incomplete": 0,
            "invalid": 0,
        }
        if not source_root.is_dir():
            raise FileNotFoundError(f"timing staging root does not exist: {source_root}")
        if source_root.name != "timing_conditions":
            nested_roots = (
                source_root / "analysis" / "timing_conditions",
                source_root / "timing_conditions",
            )
            for nested_root in nested_roots:
                if nested_root.is_dir() and any(nested_root.glob("*.json")):
                    source_root = nested_root.resolve()
                    break
        per_root[str(source_root)] = root_counts
        if source_root.resolve() == destination_root.resolve():
            raise ValueError("staging root cannot be the destination timing root")
        for source_path in sorted(source_root.glob("*.json")):
            try:
                source = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                root_counts["invalid"] += 1
                continue
            if not isinstance(source, Mapping) or not _complete_coarse_timing(source):
                root_counts["incomplete"] += 1
                continue
            current_id = str(source.get("condition_id") or source_path.stem)
            if current_id != source_path.stem:
                raise ValueError(
                    f"staging condition id does not match file name: {source_path}"
                )
            identity = source.get("condition_identity")
            if not isinstance(identity, Mapping):
                raise ValueError(f"staging record lacks condition identity: {source_path}")

            target = destination_root / f"{current_id}.json"
            with condition_lock(target):
                if not target.exists():
                    atomic_write_json(target, source)
                    root_counts["copied"] += 1
                    continue
                destination = json.loads(target.read_text(encoding="utf-8"))
                if destination.get("condition_identity") != identity:
                    raise ValueError(
                        f"condition identity conflict for {current_id}: {source_path}"
                    )
                if not _complete_coarse_timing(destination):
                    destination["coarse_timing"] = source["coarse_timing"]
                    atomic_write_json(target, destination)
                    root_counts["copied"] += 1
                    continue
                if destination.get("coarse_timing") == source.get("coarse_timing"):
                    root_counts["identical"] += 1
                    continue

                repeated_coarse = source["coarse_timing"]
                digest = hashlib.sha256(
                    json.dumps(
                        repeated_coarse,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                repeats = destination.setdefault("coarse_timing_repeats", [])
                if not isinstance(repeats, list):
                    raise ValueError(
                        f"invalid coarse timing repeat history for {current_id}"
                    )
                if not any(
                    isinstance(item, Mapping)
                    and item.get("coarse_timing_sha256") == digest
                    for item in repeats
                ):
                    repeats.append(
                        {
                            "source_root": str(source_root),
                            "source_file": str(source_path),
                            "coarse_timing_sha256": digest,
                            "coarse_timing": repeated_coarse,
                        }
                    )
                else:
                    root_counts["identical"] += 1
                    continue
                for batch_name in ("batch1", "batch50"):
                    primary = destination["coarse_timing"][batch_name]
                    repeated = repeated_coarse[batch_name]
                    primary["gpu_interference"] = combine_interference(
                        primary, repeated
                    )
                atomic_write_json(target, destination)
                root_counts["repeated"] += 1

    return {
        "output_timing_root": str(destination_root),
        "staging_roots": per_root,
        "totals": {
            key: sum(counts[key] for counts in per_root.values())
            for key in ("copied", "repeated", "identical", "incomplete", "invalid")
        },
    }


def merge_fine_timing_staging_roots(
    staging_roots: Sequence[str | Path],
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    """Merge completed fine windows without replacing official coarse timing."""
    if not staging_roots:
        raise ValueError(
            "merge-fine-timing-staging requires at least one --staging-root"
        )
    destination_root = Path(output_root) / "analysis" / "timing_conditions"
    destination_root.mkdir(parents=True, exist_ok=True)
    per_root: dict[str, dict[str, int]] = {}

    def digest(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    for raw_root in staging_roots:
        source_root = Path(raw_root).expanduser().resolve()
        root_counts = {"copied": 0, "identical": 0, "incomplete": 0, "invalid": 0}
        if not source_root.is_dir():
            raise FileNotFoundError(f"timing staging root does not exist: {source_root}")
        if source_root.name != "timing_conditions":
            nested_roots = (
                source_root / "analysis" / "timing_conditions",
                source_root / "timing_conditions",
            )
            for nested_root in nested_roots:
                if nested_root.is_dir() and any(nested_root.glob("*.json")):
                    source_root = nested_root.resolve()
                    break
        per_root[str(source_root)] = root_counts
        if source_root.resolve() == destination_root.resolve():
            raise ValueError("staging root cannot be the destination timing root")

        for source_path in sorted(source_root.glob("*.json")):
            try:
                source = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                root_counts["invalid"] += 1
                continue
            if (
                not isinstance(source, Mapping)
                or not _complete_coarse_timing(source)
                or not _complete_fine_timing(source)
            ):
                root_counts["incomplete"] += 1
                continue
            current_id = str(source.get("condition_id") or source_path.stem)
            if current_id != source_path.stem:
                raise ValueError(
                    f"staging condition id does not match file name: {source_path}"
                )
            identity = source.get("condition_identity")
            if not isinstance(identity, Mapping):
                raise ValueError(f"staging record lacks condition identity: {source_path}")

            target = destination_root / f"{current_id}.json"
            if not target.is_file():
                raise FileNotFoundError(
                    f"official coarse timing record is missing for fine result {current_id}"
                )
            with condition_lock(target):
                destination = json.loads(target.read_text(encoding="utf-8"))
                if destination.get("condition_identity") != identity:
                    raise ValueError(
                        f"condition identity conflict for {current_id}: {source_path}"
                    )
                if not _complete_coarse_timing(destination):
                    raise ValueError(
                        f"official coarse timing is incomplete for fine result {current_id}"
                    )
                if destination.get("coarse_timing") != source.get("coarse_timing"):
                    raise ValueError(
                        f"coarse timing conflict for fine result {current_id}: {source_path}"
                    )

                existing_fine = destination.get("fine_timing")
                if _complete_fine_timing(destination) and existing_fine != source["fine_timing"]:
                    raise ValueError(
                        f"complete fine timing conflict for {current_id}: {source_path}"
                    )
                retries = destination.get("fine_timing_retries", [])
                source_retries = source.get("fine_timing_retries", [])
                if not isinstance(retries, list) or not isinstance(source_retries, list):
                    raise ValueError(f"invalid fine timing retry history for {current_id}")
                retry_digests = {digest(item) for item in retries}
                merged_retries = list(retries)
                for item in source_retries:
                    item_digest = digest(item)
                    if item_digest not in retry_digests:
                        merged_retries.append(item)
                        retry_digests.add(item_digest)

                changed = not _complete_fine_timing(destination)
                if changed:
                    destination["fine_timing"] = source["fine_timing"]
                if merged_retries != retries:
                    destination["fine_timing_retries"] = merged_retries
                    changed = True
                if changed:
                    atomic_write_json(target, destination)
                    root_counts["copied"] += 1
                else:
                    root_counts["identical"] += 1

    return {
        "output_timing_root": str(destination_root),
        "staging_roots": per_root,
        "totals": {
            key: sum(counts[key] for counts in per_root.values())
            for key in ("copied", "identical", "incomplete", "invalid")
        },
    }


def _coarse_timing_frontier_sensitivity(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare the observed frontier with a batch1 boundary-unflagged frontier.

    Boundary samples are only evidence that no external compute process was
    observed at the start/end of a window; they do not prove the whole window
    was isolated. Keep the observed frontier too, so candidates it exposes can
    be remeasured under isolated fine timing.
    """
    observed = _coarse_success_latency_frontier(rows)
    boundary_unflagged_rows = [
        row
        for row in rows
        if row.get("batch1_interference_status")
        == _NO_EXTERNAL_COMPUTE_AT_BOUNDARIES
    ]
    boundary_unflagged = _coarse_success_latency_frontier(boundary_unflagged_rows)

    selected: dict[str, dict[str, Any]] = {}
    for membership, frontier in (
        ("observed", observed),
        ("batch1_boundary_unflagged", boundary_unflagged),
    ):
        for row in frontier:
            condition = dict(row)
            condition_id_value = str(condition["condition_id"])
            current = selected.get(condition_id_value)
            if current is None:
                condition["frontier_membership"] = []
                selected[condition_id_value] = condition
                current = condition
            current["frontier_membership"].append(membership)
    selected_rows = []
    for condition in selected.values():
        statuses = condition.get("coarse_timing_interference", {})
        condition["isolated_retest_recommended"] = (
            bool(condition["frontier_membership"])
            and (
                not isinstance(statuses, Mapping)
                or any(
                    statuses.get(batch_name)
                    != _NO_EXTERNAL_COMPUTE_AT_BOUNDARIES
                    for batch_name in ("batch1", "batch50")
                )
            )
        )
        selected_rows.append(condition)

    selected_rows.sort(
        key=lambda row: (
            str(row["task"]),
            float(row["batch1_p50_seconds"]),
            -float(row["success_rate"]),
            str(row["condition_id"]),
        )
    )
    interference_status_counts: dict[str, dict[str, int]] = {
        "batch1": {},
        "batch50": {},
    }
    for row in rows:
        statuses = row.get("coarse_timing_interference", {})
        for batch_name in ("batch1", "batch50"):
            status = (
                str(statuses.get(batch_name) or "missing_snapshot")
                if isinstance(statuses, Mapping)
                else "missing_snapshot"
            )
            batch_counts = interference_status_counts[batch_name]
            batch_counts[status] = batch_counts.get(status, 0) + 1
    return {
        "policy": (
            "select the union of the observed batch1 frontier and the batch1 "
            "frontier restricted to windows with no external compute observed "
            "at boundaries; retain fixed anchors separately; boundary status "
            "does not prove full-window isolation"
        ),
        "candidate_count": len(rows),
        "batch1_boundary_unflagged_candidate_count": len(boundary_unflagged_rows),
        "interference_status_counts": {
            batch_name: dict(sorted(counts.items()))
            for batch_name, counts in interference_status_counts.items()
        },
        "observed_frontier_condition_ids": [
            str(row["condition_id"]) for row in observed
        ],
        "batch1_boundary_unflagged_frontier_condition_ids": [
            str(row["condition_id"]) for row in boundary_unflagged
        ],
        "selected_frontier_conditions": selected_rows,
    }


def _timing_config_key(spec: Mapping[str, Any]) -> str:
    """Key a configuration while ignoring scan labels and evaluation seed."""
    config = dict(spec)
    config.pop("group", None)
    config.pop("evaluation_seed", None)
    config.pop("stability_category", None)
    # S1 used the generic proposal-set label for P0, while A1 labels its sole
    # action explicitly. Both execute the same P0 policy path.
    if config.get("mode") == "P0" and config.get("candidate_count") == 1:
        config["candidate_semantics"] = "single_action"
    return stable_sha256(config)


def _shared_method_key(spec: Mapping[str, Any]) -> str:
    """Key a cross-task method configuration, excluding task-specific protocol labels."""
    config = dict(spec)
    for key in ("group", "task", "evaluation_seed", "stability_category"):
        config.pop(key, None)
    # CEM clipping is fixed by the task protocol, not selected as a method knob.
    config.pop("cem_protocol", None)
    config.pop("action_bound_mode", None)
    if config.get("mode") == "P0" and config.get("candidate_count") == 1:
        config["candidate_semantics"] = "single_action"
    return stable_sha256(config)


def select_fine_timing(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    """Select fixed anchors and seed-42 coarse success-latency frontier rows."""
    output_root = _resolve(args.output_root)
    index_path = output_root / "analysis" / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Phase1.5 index is missing: {index_path}")
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    indexed_items = list(index_payload.get("items", {}).values())

    adaptive_specs = _load_adaptive_specs(output_root)
    fixed_specs = sampling_stability_specs()
    all_specs = [*primary_condition_specs(), *fixed_specs, *adaptive_specs]
    if len(all_specs) != 2600:
        raise RuntimeError(
            f"fine timing selection expects the complete 2,600-condition plan, got {len(all_specs)}"
        )
    if args.include_stability is False or args.include_adaptive_stability is False:
        raise ValueError(
            "select-fine-timing requires --include-stability and --include-adaptive-stability "
            "so it can verify the full 2,600-condition index"
        )
    expected_keys = {stable_sha256(dict(spec)) for spec in all_specs}
    indexed_by_spec: dict[str, Mapping[str, Any]] = {}
    for item in indexed_items:
        spec = item.get("spec")
        if isinstance(spec, Mapping):
            indexed_by_spec[stable_sha256(dict(spec))] = item
    missing_specs = expected_keys - indexed_by_spec.keys()
    if missing_specs:
        raise RuntimeError(
            f"Phase1.5 index is missing {len(missing_specs)} planned conditions; "
            "run index with both stability flags after the scans finish"
        )
    incomplete_specs = [
        item.get("condition_id")
        for key, item in indexed_by_spec.items()
        if key in expected_keys
        and (item.get("status") != "completed" or not item.get("validation", {}).get("complete"))
    ]
    if incomplete_specs:
        raise RuntimeError(
            f"fine timing selection requires complete 50-episode results for every planned condition; "
            f"{len(incomplete_specs)} are incomplete (first IDs: {incomplete_specs[:10]})"
        )

    manifests = _load_manifests(config)
    checkpoints, checkpoint_hashes = _checkpoint_paths(config)
    expected_ids: dict[str, str] = {}
    for spec in all_specs:
        task = str(spec["task"])
        identity = condition_identity(
            spec,
            checkpoint=str(checkpoints[task]),
            checkpoint_sha256=checkpoint_hashes[task],
            cohort=manifests[task],
        )
        expected_ids[stable_sha256(dict(spec))] = condition_id(identity)

    timing_root = output_root / "analysis" / "timing_conditions"
    timing_by_id: dict[str, Mapping[str, Any]] = {}
    for path in timing_root.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        timing_by_id[str(record.get("condition_id", path.stem))] = record

    missing_coarse = []
    for current_id in expected_ids.values():
        timing_record = timing_by_id.get(current_id)
        if not isinstance(timing_record, Mapping) or not _complete_coarse_timing(
            timing_record
        ):
            missing_coarse.append(current_id)
    if missing_coarse:
        missing = sorted(set(missing_coarse))
        raise RuntimeError(
            f"fine timing selection requires complete coarse windows for all {len(all_specs)} conditions; "
            f"{len(missing)} are missing (first IDs: {missing[:10]})"
        )

    primary_by_hash = {
        stable_sha256(dict(spec)): spec
        for spec in primary_condition_specs()
    }
    primary_by_timing_config = {
        _timing_config_key(spec): stable_sha256(dict(spec))
        for spec in primary_condition_specs()
    }
    anchor_hashes: dict[str, set[str]] = {}
    for source, specs in (("fixed_stability_anchor", fixed_specs), ("adaptive_stability_anchor", adaptive_specs)):
        for spec in specs:
            config_key = _timing_config_key(spec)
            key = primary_by_timing_config.get(config_key)
            if key is None:
                raise RuntimeError(f"fine-timing anchor is not represented in the seed-42 primary grid: {spec}")
            anchor_hashes.setdefault(key, set()).add(source)

    selected: dict[str, dict[str, Any]] = {}
    for key, reasons in anchor_hashes.items():
        condition = indexed_by_spec[key]
        current_id = expected_ids[key]
        timing_record = timing_by_id.get(current_id)
        coarse = timing_record.get("coarse_timing") if isinstance(timing_record, Mapping) else None
        batch1 = coarse.get("batch1") if isinstance(coarse, Mapping) else None
        batch50 = coarse.get("batch50") if isinstance(coarse, Mapping) else None
        assert isinstance(batch1, Mapping) and isinstance(batch50, Mapping)
        spec = dict(condition["spec"])
        selected[current_id] = {
            "condition_id": current_id,
            "spec": spec,
            "task": str(spec["task"]),
            "success_rate": float(condition["validation"]["success_rate"]),
            "batch1_p50_seconds": float(batch1["p50_seconds"]),
            "batch50_p50_seconds": float(batch50["p50_seconds"]),
            "reasons": sorted(reasons),
            "coarse_timing_interference": {
                "batch1": _timing_interference_status(batch1),
                "batch50": _timing_interference_status(batch50),
            },
            "frontier_membership": [],
            "isolated_retest_recommended": False,
        }

    pareto_candidates: list[dict[str, Any]] = []
    for key, spec in primary_by_hash.items():
        item = indexed_by_spec[key]
        current_id = expected_ids[key]
        validation = item.get("validation", {})
        timing_record = timing_by_id.get(current_id)
        coarse = timing_record.get("coarse_timing") if isinstance(timing_record, Mapping) else None
        batch1 = coarse.get("batch1") if isinstance(coarse, Mapping) else None
        batch50 = coarse.get("batch50") if isinstance(coarse, Mapping) else None
        assert isinstance(batch1, Mapping) and isinstance(batch50, Mapping)
        pareto_candidates.append({
            "condition_id": current_id,
            "spec": dict(spec),
            "task": str(spec["task"]),
            "success_rate": float(validation["success_rate"]),
            "batch1_p50_seconds": float(batch1["p50_seconds"]),
            "batch50_p50_seconds": float(batch50["p50_seconds"]),
            "forward_count": int(batch1.get("forward_count") or 0),
            "backward_count": int(batch1.get("guidance_backward_count") or 0),
            "coarse_timing_interference": {
                "batch1": _timing_interference_status(batch1),
                "batch50": _timing_interference_status(batch50),
            },
            "batch1_interference_status": _timing_interference_status(batch1),
        })
    frontier_sensitivity = _coarse_timing_frontier_sensitivity(pareto_candidates)
    for row in frontier_sensitivity["selected_frontier_conditions"]:
        current_id = str(row["condition_id"])
        frontier_reasons = []
        if "observed" in row["frontier_membership"]:
            frontier_reasons.append("observed_coarse_pareto_frontier")
        if "batch1_boundary_unflagged" in row["frontier_membership"]:
            frontier_reasons.append(
                "batch1_boundary_unflagged_coarse_pareto_frontier"
            )
        if row["isolated_retest_recommended"]:
            frontier_reasons.append("coarse_frontier_interference_suspect_retest")
        if current_id in selected:
            selected[current_id]["reasons"].extend(frontier_reasons)
            selected[current_id]["reasons"].sort()
            selected[current_id]["frontier_membership"] = row[
                "frontier_membership"
            ]
            selected[current_id]["isolated_retest_recommended"] = row[
                "isolated_retest_recommended"
            ]
        else:
            selected[current_id] = {
                **row,
                "reasons": frontier_reasons,
            }

    primary_condition_ids = {
        str(row["condition_id"]) for row in pareto_candidates
    }
    for condition_id_value, row in selected.items():
        row["reasons"] = sorted(set(row["reasons"]))
        if condition_id_value not in primary_condition_ids:
            row["coarse_timing_interference"] = {
                "batch1": _timing_interference_status(
                    ((timing_by_id.get(condition_id_value) or {}).get("coarse_timing") or {}).get("batch1")
                ),
                "batch50": _timing_interference_status(
                    ((timing_by_id.get(condition_id_value) or {}).get("coarse_timing") or {}).get("batch50")
                ),
            }

    result = {
        "schema_version": "round5_phase1_5_fine_timing_selection_v1",
        "source_index": str(index_path),
        "coarse_timing_root": str(timing_root),
        "criteria": {
            "fixed_anchors": "seed-42 equivalents of all fixed seed-43/44 stability configurations",
            "adaptive_anchors": "seed-42 equivalents of all adaptive stability configurations",
            "pareto": "union of the observed per-task frontier and the per-task frontier restricted to batch1 windows with no external compute observed at boundaries",
            "exact_point_tie_break": "fewest model calls, then condition_id",
            "interference_evidence_limit": "boundary samples do not prove full-window isolation; flagged observed-frontier conditions should be repeated in fine timing and fine-window boundary statuses gate method selection",
        },
        "coarse_frontier_sensitivity": {
            key: value
            for key, value in frontier_sensitivity.items()
            if key != "selected_frontier_conditions"
        },
        "planned_condition_count": len(all_specs),
        "fixed_anchor_count": len({_timing_config_key(spec) for spec in fixed_specs}),
        "adaptive_anchor_count": len({_timing_config_key(spec) for spec in adaptive_specs}),
        "pareto_candidate_count": len(pareto_candidates),
        "interference_suspect_frontier_condition_count": sum(
            bool(row["isolated_retest_recommended"]) for row in selected.values()
        ),
        "selected_conditions": sorted(selected.values(), key=lambda row: (str(row["task"]), str(row["condition_id"]))),
    }
    target = output_root / "analysis" / "fine_timing_selection.json"
    atomic_write_json(target, result)
    output = {"fine_timing_selection": str(target), "selected_conditions": len(selected), "pareto_conditions": len(_coarse_success_latency_frontier(pareto_candidates))}
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return output


def _select_method_candidates(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Choose a main and Pareto backup using only uncontaminated fine timings."""
    if not rows:
        return None, None
    best_success = max(float(row["mean_success_rate"]) for row in rows)
    best_by_task = {
        task: max(float(row["task_success_rate"][task]) for row in rows)
        for task in PHASE15_TASKS
    }
    for row in rows:
        row["within_two_points_of_best"] = (
            float(row["mean_success_rate"]) >= best_success - 0.02
        )
        row["task_floor_satisfied"] = all(
            float(row["task_success_rate"][task]) >= best_by_task[task] - 0.04
            for task in PHASE15_TASKS
        )
        row["method_pareto"] = False

    timing_eligible = [row for row in rows if row.get("fine_timing_eligible")]
    for row in timing_eligible:
        row["method_pareto"] = not any(
            float(other["mean_success_rate"]) >= float(row["mean_success_rate"])
            and float(other["mean_batch1_p50_seconds"])
            <= float(row["mean_batch1_p50_seconds"])
            and (
                float(other["mean_success_rate"]) > float(row["mean_success_rate"])
                or float(other["mean_batch1_p50_seconds"])
                < float(row["mean_batch1_p50_seconds"])
            )
            for other in timing_eligible
            if other["candidate_id"] != row["candidate_id"]
        )

    eligible = [
        row for row in timing_eligible
        if row["within_two_points_of_best"] and row["task_floor_satisfied"]
    ]
    main = min(
        eligible,
        key=lambda row: (
            float(row["mean_batch1_p50_seconds"]),
            float(row["mean_model_calls_per_decision"]),
            -float(row["mean_success_rate"]),
            str(row["candidate_id"]),
        ),
    ) if eligible else None
    alternatives = [
        row for row in timing_eligible
        if row["method_pareto"]
        and (main is None or row["candidate_id"] != main["candidate_id"])
    ]
    backup = min(
        alternatives,
        key=lambda row: (
            -float(row["mean_success_rate"]),
            float(row["mean_batch1_p50_seconds"]),
            float(row["mean_model_calls_per_decision"]),
        ),
    ) if alternatives else None
    return main, backup


def _matched_guidance_timing_comparisons(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Report fair PO/GF timing pairs with identical non-guidance settings."""
    by_config: dict[str, dict[str, Mapping[str, Any]]] = {}
    config_values: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        config = candidate.get("config")
        if not isinstance(config, Mapping):
            continue
        mode = config.get("guidance")
        if mode not in {"post_opt", "guided_flow"}:
            continue
        comparable_config = {
            key: value
            for key, value in config.items()
            if key not in {"family", "guidance", "guidance_last_steps"}
        }
        config_key = canonical_json(comparable_config)
        config_values[config_key] = comparable_config
        by_config.setdefault(config_key, {})[str(mode)] = candidate

    comparisons: list[dict[str, Any]] = []
    for config_key in sorted(by_config):
        modes = by_config[config_key]
        post_opt = modes.get("post_opt")
        guided_flow = modes.get("guided_flow")
        if post_opt is None or guided_flow is None:
            continue
        if not post_opt.get("fine_timing_eligible") or not guided_flow.get("fine_timing_eligible"):
            continue
        po_timing = post_opt.get("timing_by_task")
        gf_timing = guided_flow.get("timing_by_task")
        if not isinstance(po_timing, Mapping) or not isinstance(gf_timing, Mapping):
            continue
        if any(
            task not in po_timing
            or task not in gf_timing
            or po_timing[task].get("batch1_p50_seconds") is None
            or gf_timing[task].get("batch1_p50_seconds") is None
            for task in PHASE15_TASKS
        ):
            continue
        by_task = {
            task: {
                "post_opt_batch1_p50_seconds": float(po_timing[task]["batch1_p50_seconds"]),
                "guided_flow_batch1_p50_seconds": float(gf_timing[task]["batch1_p50_seconds"]),
                "guided_flow_minus_post_opt_seconds": (
                    float(gf_timing[task]["batch1_p50_seconds"])
                    - float(po_timing[task]["batch1_p50_seconds"])
                ),
            }
            for task in PHASE15_TASKS
        }
        task_differences = [
            values["guided_flow_minus_post_opt_seconds"]
            for values in by_task.values()
        ]
        comparisons.append(
            {
                "matched_config": config_values[config_key],
                "post_opt_candidate_id": post_opt.get("candidate_id"),
                "guided_flow_candidate_id": guided_flow.get("candidate_id"),
                "post_opt_mean_batch1_p50_seconds": float(
                    np.mean([item["post_opt_batch1_p50_seconds"] for item in by_task.values()])
                ),
                "guided_flow_mean_batch1_p50_seconds": float(
                    np.mean([item["guided_flow_batch1_p50_seconds"] for item in by_task.values()])
                ),
                "guided_flow_minus_post_opt_mean_batch1_p50_seconds": float(
                    np.mean(task_differences)
                ),
                "guided_flow_faster_task_count": sum(value < 0.0 for value in task_differences),
                "guided_flow_faster_all_tasks": all(value < 0.0 for value in task_differences),
                "task_count": len(PHASE15_TASKS),
                "by_task": by_task,
            }
        )
    return comparisons


def _guided_flow_selection_gate(
    output_root: Path,
    timing_comparisons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the plan's GF promotion gate only when both evidence sources exist."""
    task_intervals: dict[str, tuple[float, float]] = {}
    for task in PHASE15_TASKS:
        summary_path = output_root / "diagnostics" / f"summary_{task}.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        comparison = summary.get("matched_guidance_comparison")
        bootstrap = comparison.get("bootstrap") if isinstance(comparison, Mapping) else None
        metric = (
            bootstrap.get("gf_minus_po_physical_improvement")
            if isinstance(bootstrap, Mapping)
            else None
        )
        interval = metric.get("ci95") if isinstance(metric, Mapping) else None
        estimate = comparison.get("gf_minus_po_physical_improvement_mean") if isinstance(comparison, Mapping) else None
        if estimate is not None and isinstance(interval, Sequence) and len(interval) == 2:
            task_intervals[task] = (float(interval[0]), float(interval[1]))

    physical_evidence_complete = len(task_intervals) == len(PHASE15_TASKS)
    stable_real_benefit = physical_evidence_complete and all(
        task_intervals[task][0] > 0.0 for task in PHASE15_TASKS
    )
    timing_comparisons = [
        item
        for item in timing_comparisons
        if item.get("task_count") == len(PHASE15_TASKS)
    ]
    efficiency_advantage = any(
        item.get("guided_flow_faster_all_tasks") is True
        for item in timing_comparisons
    )
    timing_evidence_complete = bool(timing_comparisons)
    gate_applied = (
        physical_evidence_complete
        and timing_evidence_complete
        and not stable_real_benefit
        and not efficiency_advantage
    )
    if stable_real_benefit:
        status = "retain_guided_flow_stable_real_benefit"
    elif efficiency_advantage:
        status = "retain_guided_flow_measured_efficiency_advantage"
    elif gate_applied:
        status = "exclude_guided_flow_from_main_method"
    else:
        status = "guided_flow_evidence_incomplete"
    return {
        "status": status,
        "physical_evidence_complete": physical_evidence_complete,
        "physical_tasks_with_intervals": len(task_intervals),
        "stable_real_benefit_all_tasks": stable_real_benefit,
        "matched_timing_comparison_count": len(timing_comparisons),
        "timing_evidence_complete": timing_evidence_complete,
        "measured_efficiency_advantage_all_tasks": efficiency_advantage,
        "exclude_guided_flow_from_main_method": gate_applied,
        "physical_intervals_by_task": {
            task: {"ci95": list(interval)} for task, interval in task_intervals.items()
        },
    }


def select_methods(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the plan's shared four-task, three-seed convergence rule."""
    output_root = _resolve(args.output_root)
    index_path = output_root / "analysis" / "index.json"
    selection_path = output_root / "analysis" / "fine_timing_selection.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Phase1.5 index is missing: {index_path}")
    if not selection_path.is_file():
        raise FileNotFoundError(f"fine timing selection is missing: {selection_path}")
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    indexed_by_spec = {
        stable_sha256(dict(item["spec"])): item
        for item in index_payload.get("items", {}).values()
        if isinstance(item.get("spec"), Mapping)
    }
    primary_specs = primary_condition_specs()
    fixed_specs = sampling_stability_specs()
    adaptive_specs = _load_adaptive_specs(output_root)
    expected_specs = [*primary_specs, *fixed_specs, *adaptive_specs]
    if len(expected_specs) != 2600 or any(
        stable_sha256(dict(spec)) not in indexed_by_spec for spec in expected_specs
    ):
        raise RuntimeError("method selection requires the complete 2,600-condition index")

    primary_by_timing_config = {
        _timing_config_key(spec): indexed_by_spec[stable_sha256(dict(spec))]
        for spec in primary_specs
    }
    timing_root = output_root / "analysis" / "timing_conditions"
    timing_by_id: dict[str, Mapping[str, Any]] = {}
    for path in timing_root.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        timing_by_id[str(record.get("condition_id", path.stem))] = record
    fine_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    fine_ids = {
        str(row["condition_id"])
        for row in fine_selection.get("selected_conditions", ())
        if isinstance(row, Mapping) and row.get("condition_id") is not None
    }

    candidates: dict[str, dict[str, Any]] = {}
    seeds = (PHASE15_EVAL_SEED, *PHASE15_STABILITY_SEEDS)
    for stability_spec in [*fixed_specs, *adaptive_specs]:
        task = str(stability_spec["task"])
        seed = int(stability_spec["evaluation_seed"])
        shared_key = _shared_method_key(stability_spec)
        candidate = candidates.setdefault(
            shared_key,
            {
                "candidate_id": shared_key[:16],
                "config": {
                    key: value
                    for key, value in stability_spec.items()
                    if key not in {
                        "task", "group", "evaluation_seed", "stability_category",
                        "cem_protocol", "action_bound_mode",
                    }
                },
                "success_by_task_seed": {},
                "condition_ids_by_task": {},
                "timing_by_task": {},
                "timing_interference_by_task": {},
                "calls_by_task": {},
            },
        )
        if seed == PHASE15_EVAL_SEED:
            raise AssertionError("stability extensions must contain only seeds 43 and 44")

        stability_item = indexed_by_spec.get(stable_sha256(dict(stability_spec)))
        if stability_item is None:
            raise RuntimeError(
                f"stability condition is missing from the index: {stability_spec}"
            )
        if stability_item.get("status") != "completed" or not stability_item.get("validation", {}).get("complete"):
            raise RuntimeError(
                f"method selection needs complete seed-{seed} result: {stability_item.get('condition_id')}"
            )
        task_seed_key = f"{task}:{seed}"
        candidate["success_by_task_seed"].setdefault(
            task_seed_key, float(stability_item["validation"]["success_rate"])
        )
        candidate["condition_ids_by_task"].setdefault(task, {}).setdefault(
            str(seed), str(stability_item["condition_id"])
        )

        base_item = primary_by_timing_config.get(_timing_config_key(stability_spec))
        if base_item is None:
            raise RuntimeError(
                f"stability configuration has no seed-42 primary match: {stability_spec}"
            )
        if base_item.get("status") != "completed" or not base_item.get("validation", {}).get("complete"):
            raise RuntimeError(
                f"method selection needs complete seed-42 result: {base_item.get('condition_id')}"
            )
        base_condition_id = str(base_item["condition_id"])
        candidate["success_by_task_seed"].setdefault(
            f"{task}:{PHASE15_EVAL_SEED}",
            float(base_item["validation"]["success_rate"]),
        )
        candidate["condition_ids_by_task"].setdefault(task, {}).setdefault(
            str(PHASE15_EVAL_SEED), base_condition_id
        )

        if base_condition_id not in fine_ids:
            raise RuntimeError(f"fixed anchor is missing a fine timing window: {base_condition_id}")
        timing_record = timing_by_id.get(base_condition_id)
        fine = timing_record.get("fine_timing") if isinstance(timing_record, Mapping) else None
        batch1 = fine.get("batch1") if isinstance(fine, Mapping) else None
        batch50 = fine.get("batch50") if isinstance(fine, Mapping) else None
        if not isinstance(batch1, Mapping) or not isinstance(batch50, Mapping):
            raise RuntimeError(f"fixed anchor has no complete fine timing window: {base_condition_id}")
        runs = int(batch1.get("runs") or 100)
        interference_statuses = []
        for batch in (batch1, batch50):
            interference = batch.get("gpu_interference", {})
            interference_statuses.append(
                interference.get("status")
                if isinstance(interference, Mapping)
                else None
            )
        candidate["timing_interference_by_task"][task] = interference_statuses
        candidate["timing_by_task"][task] = {
            "batch1_p50_seconds": float(batch1["p50_seconds"]),
            "batch1_p95_seconds": float(batch1["p95_seconds"]),
            "batch50_p50_seconds": float(batch50["p50_seconds"]),
            "batch50_throughput_per_second": float(batch50["throughput_per_second"]),
            "stage_a_forward_calls_per_decision": float(
                batch1.get("a_forward_count", 0)
            ) / max(runs, 1),
            "stage_b_forward_calls_per_decision": float(
                batch1.get("b_forward_count", 0)
            ) / max(runs, 1),
            "guidance_backward_calls_per_decision": float(
                batch1.get("guidance_backward_count", 0)
            ) / max(runs, 1),
            "peak_memory_bytes": batch1.get("peak_memory_bytes"),
        }
        candidate["calls_by_task"][task] = float(
            (
                int(batch1.get("forward_count") or 0)
                + int(batch1.get("guidance_backward_count") or 0)
            )
            / max(runs, 1)
        )

    rows: list[dict[str, Any]] = []
    timing_ineligible_candidates: list[dict[str, Any]] = []
    expected_task_seeds = {
        f"{task}:{seed}" for task in PHASE15_TASKS for seed in seeds
    }
    for candidate in candidates.values():
        if (
            set(candidate["success_by_task_seed"]) != expected_task_seeds
            or set(candidate["timing_by_task"]) != set(PHASE15_TASKS)
        ):
            continue
        bad_windows = {
            task: statuses
            for task, statuses in candidate["timing_interference_by_task"].items()
            if len(statuses) != 2
            or any(status != "no_external_compute_observed_at_boundaries" for status in statuses)
        }
        if bad_windows:
            timing_ineligible_candidates.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "reason": "fine timing window observed external GPU work or lacked an interference snapshot",
                    "task_windows": bad_windows,
                }
            )
        candidate["fine_timing_eligible"] = not bool(bad_windows)
        candidate["task_success_rate"] = {
            task: float(
                np.mean([
                    candidate["success_by_task_seed"][f"{task}:{seed}"]
                    for seed in seeds
                ])
            )
            for task in PHASE15_TASKS
        }
        candidate["mean_success_rate"] = float(
            np.mean(list(candidate["success_by_task_seed"].values()))
        )
        candidate["mean_batch1_p50_seconds"] = float(
            np.mean([
                candidate["timing_by_task"][task]["batch1_p50_seconds"]
                for task in PHASE15_TASKS
            ])
        )
        candidate["mean_batch50_p50_seconds"] = float(
            np.mean([
                candidate["timing_by_task"][task]["batch50_p50_seconds"]
                for task in PHASE15_TASKS
            ])
        )
        candidate["mean_batch1_p95_seconds"] = float(
            np.mean([
                candidate["timing_by_task"][task]["batch1_p95_seconds"]
                for task in PHASE15_TASKS
            ])
        )
        candidate["mean_batch50_throughput_per_second"] = float(
            np.mean([
                candidate["timing_by_task"][task]["batch50_throughput_per_second"]
                for task in PHASE15_TASKS
            ])
        )
        candidate["mean_model_calls_per_decision"] = float(
            np.mean(list(candidate["calls_by_task"].values()))
        )
        candidate["mean_stage_a_forward_calls_per_decision"] = float(
            np.mean([
                candidate["timing_by_task"][task]["stage_a_forward_calls_per_decision"]
                for task in PHASE15_TASKS
            ])
        )
        candidate["mean_stage_b_forward_calls_per_decision"] = float(
            np.mean([
                candidate["timing_by_task"][task]["stage_b_forward_calls_per_decision"]
                for task in PHASE15_TASKS
            ])
        )
        candidate["mean_guidance_backward_calls_per_decision"] = float(
            np.mean([
                candidate["timing_by_task"][task]["guidance_backward_calls_per_decision"]
                for task in PHASE15_TASKS
            ])
        )
        peak_memory_values = [
            int(candidate["timing_by_task"][task]["peak_memory_bytes"])
            for task in PHASE15_TASKS
            if candidate["timing_by_task"][task]["peak_memory_bytes"] is not None
        ]
        candidate["max_batch1_peak_memory_bytes"] = (
            max(peak_memory_values) if peak_memory_values else None
        )
        rows.append(candidate)
    if not rows:
        raise RuntimeError(
            "no fixed stability method has complete four-task, three-seed evidence"
        )

    guidance_timing_comparisons = _matched_guidance_timing_comparisons(rows)
    guidance_gate = _guided_flow_selection_gate(
        output_root,
        guidance_timing_comparisons,
    )
    selection_rows: list[dict[str, Any]] = []
    for row in rows:
        guided_flow = row.get("config", {}).get("guidance") == "guided_flow"
        row["method_selection_eligible"] = not (
            guided_flow and guidance_gate["exclude_guided_flow_from_main_method"]
        )
        if row["method_selection_eligible"]:
            selection_rows.append(row)
    main, backup = _select_method_candidates(selection_rows)
    result = {
        "schema_version": "round5_phase1_5_method_selection_v1",
        "source_index": str(index_path),
        "source_fine_timing_selection": str(selection_path),
        "rules": {
            "aggregate": "mean success rate over four tasks and evaluation seeds 42, 43, 44",
            "near_tie": "within 2 percentage points of the highest aggregate success rate",
            "per_task_floor": "no more than 4 percentage points below the best candidate on any task",
            "latency_tiebreak": "lowest four-task mean single-environment fine batch1 p50; then fewest model calls per decision",
            "backup": "highest-success remaining candidate on the aggregate success-latency Pareto frontier",
            "guidance_efficiency": "matched PO/GF configurations with uncontaminated fine batch1 p50 timings on all four tasks",
            "guidance_gate": "exclude guided_flow from main/backup selection only when all-task state-bootstrap comparisons and matched four-task timings are complete and show neither a stable real benefit nor a measured all-task latency advantage",
        },
        "candidate_count": len(rows),
        "method_selection_eligible_candidate_count": len(selection_rows),
        "timing_eligible_candidate_count": sum(
            bool(row.get("fine_timing_eligible")) for row in rows
        ),
        "timing_ineligible_candidates": timing_ineligible_candidates,
        "guidance_gate": guidance_gate,
        "guidance_mode_efficiency_comparisons": guidance_timing_comparisons,
        "main_candidate_id": None if main is None else main["candidate_id"],
        "backup_candidate_id": None if backup is None else backup["candidate_id"],
        "converged": main is not None,
        "candidates": sorted(
            rows,
            key=lambda row: (
                -float(row["mean_success_rate"]),
                float(row["mean_batch1_p50_seconds"]),
                str(row["candidate_id"]),
            ),
        ),
    }
    target = output_root / "analysis" / "method_selection.json"
    atomic_write_json(target, result)
    output = {
        "method_selection": str(target),
        "candidate_count": len(rows),
        "converged": main is not None,
        "main_candidate_id": result["main_candidate_id"],
        "backup_candidate_id": result["backup_candidate_id"],
    }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return output


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
    adaptive_specs = (
        _load_adaptive_specs(_resolve(args.output_root))
        if args.include_adaptive_stability
        else []
    )
    specs.extend(adaptive_specs)
    print(json.dumps({
        "status": "ok",
        "config": str(_resolve(args.config)),
        "conditions_selected": len(specs),
        "primary_conditions": 2424,
        "fixed_stability_conditions": 96,
        "adaptive_stability_conditions": len(adaptive_specs),
        "tasks": sorted({str(spec["task"]) for spec in specs}),
        "groups": sorted({str(spec["group"]) for spec in specs}),
        "gpu_preference": [0, 1, 2, 3],
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
        args.min_swap_free_mib,
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
        {
            "name": "P1_c300_i30",
            "mode": "P1",
            "flow_steps": None,
            "candidate_count": 300,
            "cem_protocol": "cem-clip",
            "action_bound_mode": "candidate_clip",
            "guidance": "none",
            "cem_num_samples": 300,
            "cem_iterations": 30,
            "cem_elite_ratio": 0.1,
            "cem_var_scale": 1.0,
        },
        {
            "name": "P2_c300_i30",
            "mode": "P2",
            "flow_steps": 1,
            "candidate_count": 300,
            "cem_protocol": "cem-clip",
            "action_bound_mode": "candidate_clip",
            "guidance": "none",
            "cem_num_samples": 300,
            "cem_iterations": 30,
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
            result_file = output_dir / "result.json"
            if result_file.is_file():
                cached = json.loads(result_file.read_text(encoding="utf-8"))
                records.append(
                    {
                        "task": task,
                        "condition": spec["name"],
                        "status": "reused",
                        "success_rate": cached.get("success_rate"),
                        "evaluation_seconds": cached.get("evaluation_seconds"),
                    }
                )
                continue
            usage_before = resource.getrusage(resource.RUSAGE_SELF)
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
                usage_after = resource.getrusage(resource.RUSAGE_SELF)
                planning = result.get("round4_planning", {})
                records.append(
                    {
                        "task": task,
                        "condition": spec["name"],
                        "status": "ok",
                        "success_rate": result["success_rate"],
                        "evaluation_seconds": result.get("evaluation_seconds"),
                        "planning_samples": len(planning.get("planning_samples_seconds", ())),
                        "planning_peak_memory_bytes": planning.get("peak_memory_bytes"),
                        "process_cpu_seconds": float(
                            (usage_after.ru_utime + usage_after.ru_stime)
                            - (usage_before.ru_utime + usage_before.ru_stime)
                        ),
                        "process_max_rss_kib": int(usage_after.ru_maxrss),
                    }
                )
            except Exception as exc:
                records.append({"task": task, "condition": spec["name"], "status": "failed", "error": str(exc)})
                raise
    atomic_write_json(target_root / "calibration.json", {"tasks": list(tasks), "episodes_per_condition": 5, "results": records})
    print(json.dumps({"calibration": str(target_root / 'calibration.json'), "conditions": len(records)}, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "validate",
            "dry-run",
            "calibrate",
            "scan",
            "index",
            "analyze",
            "timing",
            "select-stability",
            "select-fine-timing",
            "select-methods",
            "merge-timing-staging",
            "merge-fine-timing-staging",
        ),
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT))
    parser.add_argument("--task", choices=(*PHASE15_TASKS, "all"), default="all")
    parser.add_argument("--group")
    parser.add_argument("--staging-root", action="append", default=[])
    parser.add_argument("--condition-index", action="append", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-stability", action="store_true")
    parser.add_argument("--include-adaptive-stability", action="store_true")
    parser.add_argument(
        "--fine",
        action="store_true",
        help="run the 20 warm-up + 100 synchronized-call timing window",
    )
    parser.add_argument(
        "--retry-interfered-fine",
        action="store_true",
        help=(
            "with --fine, repeat both batches for an existing fine window if either "
            "window has external compute observed at boundaries or lacks a snapshot"
        ),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--refresh-index", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=_gpu)
    parser.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB)
    parser.add_argument("--min-available-mib", type=int, default=DEFAULT_MIN_AVAILABLE_MIB)
    parser.add_argument("--min-swap-free-mib", type=int, default=DEFAULT_MIN_SWAP_FREE_MIB)
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
    elif args.command == "select-fine-timing":
        select_fine_timing(args, config)
    elif args.command == "select-methods":
        select_methods(args, config)
    elif args.command == "merge-timing-staging":
        print(
            json.dumps(
                merge_timing_staging_roots(
                    args.staging_root,
                    output_root=_resolve(args.output_root),
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    elif args.command == "merge-fine-timing-staging":
        print(
            json.dumps(
                merge_fine_timing_staging_roots(
                    args.staging_root,
                    output_root=_resolve(args.output_root),
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    elif args.command == "timing":
        timing(args, config)
    else:
        analyze(args, config)


if __name__ == "__main__":
    main()
