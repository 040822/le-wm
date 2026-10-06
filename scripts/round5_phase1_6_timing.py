#!/usr/bin/env python3
"""Run isolated Round5 Phase1.6 inference timing on fixed real observations.

Example: ``CUDA_VISIBLE_DEVICES=3 python scripts/round5_phase1_6_timing.py --gpu 3``
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

CPU_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
for _name, _value in CPU_THREAD_ENV.items():
    os.environ[_name] = _value

TORCH_INTRAOP_THREADS = 1
TORCH_INTEROP_THREADS = 1

import numpy as np
import torch

torch.set_num_threads(TORCH_INTRAOP_THREADS)
torch.set_num_interop_threads(TORCH_INTEROP_THREADS)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.checkpoint import load_policy_or_model
from source.common.eval import EvaluationIdentity, get_dataset
from source.common.round4_eval import run_round4_evaluation
from source.common.round5_phase1_5 import synchronous_timing
from source.common.round5_phase1_6 import atomic_json, stable_hash, stable_seed
from scripts.round5_phase1_6 import (
    METHODS,
    _checkpoint,
    _gpu_preflight,
    _host_preflight,
    _load_config,
    _load_manifest,
    _resolve,
    _task_cfg,
)


def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gpu_snapshot(gpu: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    proc = subprocess.run(
        [
            "nvidia-smi", "--id", str(gpu), "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT, env=env, check=True, capture_output=True, text=True,
    )
    rows = []
    for line in proc.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 2:
            try:
                rows.append({"pid": int(fields[0]), "used_gpu_memory_mib": int(fields[1])})
            except ValueError:
                continue
    return {"gpu": int(gpu), "compute_processes": rows}


def _host_load_snapshot() -> dict[str, Any]:
    load1, load5, load15 = os.getloadavg()
    cpus = max(1, int(os.cpu_count() or 1))
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_count": cpus,
        "load1": float(load1),
        "load5": float(load5),
        "load15": float(load15),
        "load1_per_cpu": float(load1 / cpus),
    }


def _interference(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        int(row["pid"])
        for snapshot in (before, after)
        for row in snapshot.get("compute_processes", ())
        if int(row.get("pid", -1)) != os.getpid()
    }
    return {
        "external_compute_pids_at_boundaries": sorted(observed),
        "external_compute_process_observed": bool(observed),
        "status": "interference_flagged" if observed else "no_external_compute_observed_at_boundaries",
    }


def _subset_manifest(manifest, indices: Sequence[int], label: str):
    entries = tuple(manifest.entries[int(index)] for index in indices)
    selected = {json.dumps(item.episode_id, ensure_ascii=False, sort_keys=True) for item in entries}
    split = {}
    for key, values in manifest.episode_split.items():
        split[key] = tuple(
            value for value in values
            if json.dumps(value, ensure_ascii=False, sort_keys=True) not in selected
        )
    split["custom"] = tuple(item.episode_id for item in entries)
    result = replace(
        manifest,
        cohort_id=f"{manifest.cohort_id}_timing_{label}",
        cohort_kind="custom",
        entries=entries,
        episode_split=split,
        cohort_sha256=None,
    )
    return replace(result, cohort_sha256=result.computed_sha256)


def _observation_groups(count: int, batch_size: int, task: str, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(stable_seed("phase1_6_timing_observations", task, int(seed), batch_size))
    if batch_size == 1:
        order = rng.permutation(count)
        return [[int(item)] for item in order[:5]]
    if batch_size != 50 or count < batch_size:
        raise ValueError(f"unsupported timing batch={batch_size}, cohort={count}")
    return [
        [int(item) for item in rng.choice(count, size=batch_size, replace=False)]
        for _ in range(5)
    ]


def _event_counts(policy: Any, mode: str, events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    a_forwards = b_forwards = backwards = 0
    peak_memory = []
    for event in events:
        guidance = event.get("guidance_stats", {})
        if mode == "P0":
            a_forwards += int(guidance.get("stage_a_forward_count", event.get("forward_count", 0)))
            b_forwards += int(guidance.get("stage_b_forward_count", 0))
        elif mode in {"P1", "P2"}:
            a_forwards += int(event.get("stage_a_forward_count", 0))
            b_forwards += int(event.get("stage_b_forward_count", event.get("forward_count", 0)))
        else:
            a_forwards += int(event.get("stage_a_forward_count", event.get("proposal_forward_count", 0)))
            b_forwards += int(event.get("stage_b_forward_count", 0)) + int(event.get("verifier_forward_count", 0))
        backwards += int(event.get("guidance_backward_count", guidance.get("backward_count", 0)))
        if event.get("peak_memory_bytes") is not None:
            peak_memory.append(int(event["peak_memory_bytes"]))
    return {
        "a_forward_count": a_forwards,
        "b_forward_count": b_forwards,
        "forward_count": a_forwards + b_forwards,
        "guidance_backward_count": backwards,
        "peak_memory_bytes": max(peak_memory) if peak_memory else None,
        "timed_planning_events": len(events),
        "forward_count_is_exact": mode in {"P0", "P1", "P2", "P3"},
    }


def _timing_group(
    *, task: str, method_id: str, method: Mapping[str, Any], seed: int,
    indices: Sequence[int], group_index: int, batch_size: int, config: Mapping[str, Any],
    manifest, model: Any, dataset: Any, checkpoint: Path, gpu: int,
) -> dict[str, Any]:
    import torch

    small = _subset_manifest(manifest, indices, f"b{batch_size}_g{group_index}")
    cfg = _task_cfg(task, config, seed=seed, count=batch_size, device="cuda", method=method)
    mode = str(method["mode"])
    target = _resolve(config["paths"]["output_root"]) / "timing" / "scratch" / task / method_id / f"batch_{batch_size}" / f"group_{group_index}"
    events = []

    def capture(policy, *call_args, **call_kwargs):
        info = call_kwargs.get("info_dict")
        if info is None and call_args:
            info = call_args[0]
        if not isinstance(info, Mapping):
            raise TypeError("timing capture expected a real policy info mapping")

        def infer_fixed_input():
            replay = dict(info)
            replay["_needs_flush"] = np.ones(batch_size, dtype=bool)
            return policy.get_action(replay)

        start_event_count = len(getattr(policy, "planning_events", ()))
        measured = synchronous_timing(
            infer_fixed_input,
            warmup=2,
            runs=10,
            synchronize=lambda: _sync("cuda"),
        )
        policy_events = list(getattr(policy, "planning_events", ()))
        timed = policy_events[-10:]
        if len(timed) != 10:
            raise RuntimeError("expected one planning event for each timed call")
        events.extend(timed)
        return {
            **measured,
            **_event_counts(policy, mode, timed),
            "environment_batch_size": batch_size,
            "source_observations": len(indices),
            "planning_events_before_timing": start_event_count,
        }

    host_before = _host_load_snapshot()
    before = _gpu_snapshot(gpu)
    result = run_round4_evaluation(
        cfg,
        task=task,
        policy_or_model=model,
        mode=mode,
        identity=EvaluationIdentity(
            entrypoint="round5_phase1_6_isolated_timing",
            policy_kind="round4_shared_dit",
            checkpoint=str(checkpoint.resolve()),
            epoch=int(config["training"]["epoch"]),
            stage=mode,
        ),
        manifest=small,
        output_dir=target,
        dataset=dataset,
        device="cuda",
        trace=False,
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
        timing_capture_callback=capture,
    )
    after = _gpu_snapshot(gpu)
    host_after = _host_load_snapshot()
    if result.get("status") != "timing_capture":
        raise RuntimeError(f"timing callback did not capture: {task}/{method_id}/batch{batch_size}")
    measured = dict(result["timing_capture"])
    measured["observation_group"] = int(group_index)
    measured["observation_indices"] = [int(value) for value in indices]
    measured["gpu_interference"] = _interference(before, after)
    measured["gpu_before"] = before
    measured["gpu_after"] = after
    measured["host_load_before"] = host_before
    measured["host_load_after"] = host_after
    return measured


def _aggregate(
    groups: Sequence[Mapping[str, Any]], batch_size: int,
    max_load_per_cpu: float,
) -> dict[str, Any]:
    samples = [
        float(value)
        for group in groups
        for value in group.get("samples_seconds", ())
        if np.isfinite(float(value))
    ]
    if not samples:
        raise ValueError("timing window has no samples")
    values = np.asarray(samples, dtype=np.float64)
    external = any(bool(g.get("gpu_interference", {}).get("external_compute_process_observed")) for g in groups)
    group_host_load = []
    for group in groups:
        before = dict(group["host_load_before"])
        after = dict(group["host_load_after"])
        observed_pressure = max(float(before["load1_per_cpu"]), float(after["load1_per_cpu"])) > max_load_per_cpu
        group_host_load.append({
            "before": before,
            "after": after,
            "load_threshold_per_cpu": float(max_load_per_cpu),
            "host_pressure_observed": observed_pressure,
        })
    host_pressure = any(group["host_pressure_observed"] for group in group_host_load)
    return {
        "warmup_total": int(sum(int(g.get("warmup", 0)) for g in groups)),
        "runs_total": len(samples),
        "observation_batches": len(groups),
        "batch_size": int(batch_size),
        "samples_seconds": samples,
        "p50_seconds": float(np.quantile(values, 0.50)),
        "p95_seconds": float(np.quantile(values, 0.95)),
        "mean_seconds": float(values.mean()),
        "throughput_per_second": float(batch_size / values.mean()),
        "batch50_throughput_per_second": float(50.0 / values.mean()) if batch_size == 50 else None,
        "peak_memory_bytes": max((int(g["peak_memory_bytes"]) for g in groups if g.get("peak_memory_bytes") is not None), default=None),
        "a_forward_count_per_inference": float(sum(int(g.get("a_forward_count", 0)) for g in groups) / len(samples)),
        "b_forward_count_per_inference": float(sum(int(g.get("b_forward_count", 0)) for g in groups) / len(samples)),
        "guidance_backward_count_per_inference": float(sum(int(g.get("guidance_backward_count", 0)) for g in groups) / len(samples)),
        "interference_observed": external,
        "group_interference": [g.get("gpu_interference") for g in groups],
        "host_pressure_observed": host_pressure,
        "group_host_load": group_host_load,
        "source": "complete_policy_inference_on_five_fixed_real_observation_batches",
        "environment_step_time_included": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config)
    host = _host_preflight(config)
    gpu = _gpu_preflight(str(args.gpu), config, minimum_free_mib=int(args.min_free_mib))
    if int(gpu["free_mib"]) < int(config["budget"]["evaluation_memory_margin_gib"] * 1024):
        raise RuntimeError(f"timing GPU lacks the configured free-memory margin: {gpu}")
    print(json.dumps({"host_preflight": host, "gpu_preflight": gpu}, sort_keys=True), flush=True)
    tasks = list(config["evaluation"]["tasks"]) if args.task == "all" else [str(args.task)]
    methods = list(args.method or METHODS)
    root = _resolve(config["paths"]["output_root"])
    summary_path = _resolve(args.summary_path) if args.summary_path is not None else root / "timing" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    config_hash = _sha256_file(_resolve(args.config))
    max_load_per_cpu = float(config["budget"]["max_load_per_cpu"])
    if summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("config_sha256") != config_hash:
            raise ValueError("existing timing summary belongs to a different config")
        conditions_by_key = {
            (str(item.get("task")), str(item.get("method")), int(item.get("seed", -1))): item
            for item in previous.get("conditions", ())
        }
    else:
        conditions_by_key = {}
    timing_code = {
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "round4_eval_sha256": _sha256_file(ROOT / "source/common/round4_eval.py"),
        "round4_policy_sha256": _sha256_file(ROOT / "source/policy/round4.py"),
    }

    def write_summary() -> None:
        expected = [
            {"task": task_name, "method": method_name, "seed": int(args.seed)}
            for task_name in tasks for method_name in METHODS
        ]
        atomic_json(summary_path, {
            "schema_version": "round5_phase1_6_timing_v1",
            "config_sha256": config_hash,
            "gpu": int(args.gpu),
            "host_preflight": host,
            "gpu_preflight": gpu,
            "cpu_thread_policy": {
                "environment": dict(CPU_THREAD_ENV),
                "torch_intraop_threads": TORCH_INTRAOP_THREADS,
                "torch_interop_threads": TORCH_INTEROP_THREADS,
            },
            "conditions": list(conditions_by_key.values()),
            "pending": [
                item for item in expected
                if (item["task"], item["method"], item["seed"]) not in conditions_by_key
            ],
        })

    for task in tasks:
        checkpoint, checkpoint_hash = _checkpoint(config, task)
        model, resolved = load_policy_or_model(str(checkpoint))
        if resolved is not None and Path(resolved).resolve() != checkpoint.resolve():
            raise ValueError(f"checkpoint resolver changed requested path: {resolved}")
        model = model.to("cuda").eval()
        manifest = _load_manifest(config, task)
        base_cfg = _task_cfg(task, config, seed=int(args.seed), count=len(manifest.entries), device="cuda")
        dataset = get_dataset(base_cfg, base_cfg.eval.dataset_name)
        for method_id in methods:
            current_host = _host_load_snapshot()
            if current_host["load1_per_cpu"] > max_load_per_cpu:
                write_summary()
                print(json.dumps({
                    "status": "paused_before_dispatch_for_host_load",
                    "task": task,
                    "next_method": method_id,
                    "host_load": current_host,
                    "max_load_per_cpu": max_load_per_cpu,
                    "completed_conditions": len(conditions_by_key),
                }, sort_keys=True), flush=True)
                del model
                return {
                    "conditions": len(conditions_by_key),
                    "timing_summary": str(summary_path),
                    "paused_for_host_load": True,
                }
            method = METHODS[method_id]
            key = (task, method_id, int(args.seed))
            previous_condition = conditions_by_key.get(key)
            if previous_condition is not None:
                same_identity = (
                    previous_condition.get("checkpoint_sha256") == checkpoint_hash
                    and previous_condition.get("cohort_sha256") == manifest.computed_sha256
                    and previous_condition.get("timing_code") == timing_code
                )
                if not same_identity:
                    raise ValueError(f"existing timing condition identity changed: {key}")
                if bool(previous_condition.get("interference_eligible_for_speedup_claim")):
                    print(json.dumps({"task": task, "method": method_id, "status": "reused_timing"}, sort_keys=True), flush=True)
                    continue

            superseded_timing_attempts = []
            if previous_condition is None:
                attempt = 1
                timing_attempts = []
            else:
                superseded_timing_attempts = list(previous_condition.get("superseded_timing_attempts", ()))
                timing_attempts = list(previous_condition.get("timing_attempts", ()))
                if int(previous_condition.get("attempt", 1)) >= 3:
                    superseded_timing_attempts.extend(timing_attempts)
                    timing_attempts = []
                    attempt = 1
                else:
                    attempt = int(previous_condition.get("attempt", 1)) + 1
            attempt_payload = {}
            while True:
                groups_by_batch = {}
                for batch_size in (1, 50):
                    groups = []
                    for group_index, indices in enumerate(_observation_groups(len(manifest.entries), batch_size, task, int(args.seed))):
                        groups.append(_timing_group(
                            task=task, method_id=method_id, method=method, seed=int(args.seed),
                            indices=indices, group_index=group_index, batch_size=batch_size,
                            config=config, manifest=manifest, model=model, dataset=dataset,
                            checkpoint=checkpoint, gpu=int(args.gpu),
                        ))
                        print(json.dumps({"task": task, "method": method_id, "batch_size": batch_size, "group": group_index, "p50": float(np.quantile(groups[-1]["samples_seconds"], 0.5)), "gpu_interference": groups[-1]["gpu_interference"]["status"], "host_load1_per_cpu_before": groups[-1]["host_load_before"]["load1_per_cpu"], "host_load1_per_cpu_after": groups[-1]["host_load_after"]["load1_per_cpu"], "attempt": attempt}, sort_keys=True), flush=True)
                    groups_by_batch[f"batch{batch_size}"] = _aggregate(
                        groups, batch_size, max_load_per_cpu,
                    )
                interference_seen = any(value.get("interference_observed") for value in groups_by_batch.values())
                host_pressure_seen = any(value.get("host_pressure_observed") for value in groups_by_batch.values())
                attempt_payload = {
                    "attempt": attempt,
                    **groups_by_batch,
                    "interference_observed": interference_seen,
                    "host_pressure_observed": host_pressure_seen,
                }
                timing_attempts.append(attempt_payload)
                if not interference_seen and not host_pressure_seen or attempt >= 3:
                    break
                attempt += 1

            conditions_by_key[key] = {
                "task": task,
                "method": method_id,
                "method_spec": method,
                "checkpoint_sha256": checkpoint_hash,
                "cohort_sha256": manifest.computed_sha256,
                "seed": int(args.seed),
                "timing_code": timing_code,
                "attempt": attempt,
                "timing_attempts": timing_attempts,
                "superseded_timing_attempts": superseded_timing_attempts,
                "interference_observed": bool(attempt_payload.get("interference_observed")),
                "host_pressure_observed": bool(attempt_payload.get("host_pressure_observed")),
                "interference_eligible_for_speedup_claim": (
                    not bool(attempt_payload.get("interference_observed"))
                    and not bool(attempt_payload.get("host_pressure_observed"))
                ),
                "batch1": attempt_payload["batch1"],
                "batch50": attempt_payload["batch50"],
            }
            write_summary()
        del model
    return {"conditions": len(conditions_by_key), "timing_summary": str(summary_path)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/round5/phase1_6.json")
    parser.add_argument("--task", choices=("cube", "pusht", "reacher", "tworoom", "all"), default="all")
    parser.add_argument("--method", action="append", choices=tuple(METHODS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, choices=range(4), required=True, help="timing runs are isolated on GPU0-3")
    parser.add_argument("--min-free-mib", type=int, default=3500)
    parser.add_argument("--summary-path", type=Path, help="write an independent task-shard summary")
    args = parser.parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
