#!/usr/bin/env python3
"""Fixed-input timing and simulator-grounded ranking diagnostics for Phase 6.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.checkpoint import load_policy_or_model
from source.common.eval import (
    DatasetEvaluationSession,
    EvaluationIdentity,
    compose_eval_config,
    get_dataset,
)
from source.common.round3_phase1 import CohortManifest
from source.common.round4_eval import _DiagnosticCaptureComplete, run_round4_evaluation, validate_gpu_visibility
from source.common.data import get_column_normalizer, load_dataset
from source.policy.fast_lewam import fast_lewam_forward
from scripts import round5_phase1_5_diagnostics as p15
from scripts import round5_phase1_7_fixed_pool as fixed_pool
from scripts import round5_phase6_1 as phase


TIMING_WARMUP = 10
TIMING_RUNS = 50
TIMING_MODES = ("P0", "P1", "P2", "P3")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(phase._jsonable(payload), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _gpu_snapshot(gpu: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    import subprocess

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            str(gpu),
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [item.strip() for item in completed.stdout.splitlines()[0].split(",")]
    if len(fields) != 6:
        raise RuntimeError(f"invalid GPU{gpu} snapshot: {completed.stdout!r}")
    return {
        "gpu": int(fields[0]),
        "name": fields[1],
        "memory_total_mib": int(fields[2]),
        "memory_used_mib": int(fields[3]),
        "memory_free_mib": int(fields[4]),
        "utilization_percent": int(fields[5]),
    }


def _timing_path(root: Path, task: str, arm: str, spec: Mapping[str, Any], batch: int) -> Path:
    return (
        root
        / "analysis"
        / "timing_conditions"
        / f"{task}_{arm}_exec25_{spec['mode']}_{spec['guidance']}_batch{int(batch)}.json"
    )


def _timing_manifest(manifest: CohortManifest, batch: int) -> CohortManifest:
    if int(batch) == len(manifest.entries):
        return manifest
    if int(batch) != 1:
        raise ValueError("Phase 6.1 timing supports only batch sizes 1 and 50")
    result = phase.replace(
        manifest,
        cohort_id=f"{manifest.cohort_id}_phase6_1_timing1",
        cohort_kind="custom",
        entries=(manifest.entries[0],),
        episode_split={
            **manifest.episode_split,
            "selected": (manifest.entries[0].episode_id,),
        },
    )
    return phase.replace(result, cohort_sha256=result.computed_sha256)


def _timing_specs() -> list[dict[str, Any]]:
    return [
        spec
        for spec in phase.condition_specs()
        if int(spec["execute_steps"]) == 25
        and int(spec["score_steps_env"]) == 25
    ]


def _benchmark_one(
    *,
    task: str,
    arm: str,
    spec: Mapping[str, Any],
    batch: int,
    config: Mapping[str, Any],
    manifest: CohortManifest,
    checkpoint: Path,
    checkpoint_sha256: str,
    model: Any,
    gpu: int,
) -> dict[str, Any]:
    output_root = phase._resolve(config["output_root"])
    target = _timing_path(output_root, task, arm, spec, batch)
    identity = {
        "task": task,
        "arm": arm,
        "condition": phase._condition_identity(spec),
        "batch_size": int(batch),
        "warmup": TIMING_WARMUP,
        "runs": TIMING_RUNS,
        "checkpoint_sha256": checkpoint_sha256,
        "cohort_sha256": manifest.computed_sha256,
    }
    if target.is_file():
        saved = json.loads(target.read_text(encoding="utf-8"))
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError(f"existing Phase 6.1 timing record identity mismatch: {target}")
        if len(saved.get("samples_seconds", ())) != TIMING_RUNS:
            raise ValueError(f"existing timing record has the wrong sample count: {target}")
        return saved

    device = "cuda:0"
    cfg = phase._eval_config(task, config, device)
    cfg.eval.num_eval = int(batch)
    cfg.world.num_envs = int(batch)
    timing_manifest = _timing_manifest(manifest, batch)
    guidance = str(spec["guidance"])
    guidance_values = phase.GUIDANCE_CONFIG[guidance]
    noise_schedule = phase._LatentNoiseSchedule(
        task=task,
        seed=int(config["evaluation"]["seed"]),
        manifest=timing_manifest,
    )
    evaluation_identity = EvaluationIdentity(
        entrypoint="round5_phase6_1_fixed_input_timing",
        policy_kind="round4_shared_dit",
        checkpoint=str(checkpoint.resolve()),
        epoch=10,
        stage=str(spec["mode"]),
    )

    def capture(policy, *call_args, **call_kwargs):
        info = call_kwargs.get("info_dict")
        if info is None and call_args:
            info = call_args[0]
        if not isinstance(info, Mapping):
            raise TypeError("timing capture expected the actual policy observation mapping")
        environment_batch = int(
            getattr(getattr(policy, "env", None), "num_envs", int(batch))
        )

        def infer_same_observation():
            replay = dict(info)
            replay["_needs_flush"] = np.ones(environment_batch, dtype=bool)
            return policy.get_action(replay)

        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(TIMING_WARMUP):
            infer_same_observation()
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(TIMING_RUNS):
            started = time.perf_counter()
            infer_same_observation()
            torch.cuda.synchronize(device)
            samples.append(time.perf_counter() - started)
        events = list(getattr(policy, "planning_events", ()))
        timed = events[-TIMING_RUNS:]
        if len(timed) != TIMING_RUNS:
            raise RuntimeError(f"expected {TIMING_RUNS} planning events, observed {len(timed)}")
        mode = str(spec["mode"])
        if mode == "P0":
            stage_a = sum(
                int(event.get("guidance_stats", {}).get("stage_a_forward_count", event.get("forward_count", 0)))
                for event in timed
            )
            stage_b = sum(
                int(event.get("guidance_stats", {}).get("stage_b_forward_count", 0))
                for event in timed
            )
        elif mode in {"P1", "P2"}:
            stage_a = sum(int(event.get("stage_a_forward_count", 0)) for event in timed)
            stage_b = sum(
                int(event.get("stage_b_forward_count", event.get("forward_count", 0)))
                for event in timed
            )
        else:
            stage_a = sum(
                int(event.get("stage_a_forward_count", event.get("proposal_forward_count", 0)))
                for event in timed
            )
            stage_b = sum(
                int(event.get("stage_b_forward_count", 0))
                + int(event.get("verifier_forward_count", 0))
                for event in timed
            )
        backward = sum(
            int(event.get("guidance_backward_count", event.get("guidance_stats", {}).get("backward_count", 0)))
            for event in timed
        )
        backward_invocations = sum(
            int(event.get("guidance_backward_invocation_count", 0))
            for event in timed
        )
        peaks = [
            int(event["peak_memory_bytes"])
            for event in timed
            if event.get("peak_memory_bytes") is not None
        ]
        values = np.asarray(samples, dtype=np.float64)
        return {
            "environment_batch_size": environment_batch,
            "samples_seconds": values.tolist(),
            "p50_seconds": float(np.quantile(values, 0.50)),
            "p95_seconds": float(np.quantile(values, 0.95)),
            "mean_seconds": float(values.mean()),
            "cumulative_planning_time_seconds": float(values.sum()),
            "throughput_per_second": float(environment_batch / values.mean()),
            "stage_a_forward_count_total": int(stage_a),
            "stage_b_forward_count_total": int(stage_b),
            "guidance_backward_count_total": int(backward),
            "guidance_backward_invocation_count_total": int(
                backward_invocations
            ),
            "stage_a_forwards_per_call": float(stage_a / TIMING_RUNS),
            "stage_b_forwards_per_call": float(stage_b / TIMING_RUNS),
            "guidance_backwards_per_call": float(backward / TIMING_RUNS),
            "guidance_backward_invocations_per_call": float(
                backward_invocations / TIMING_RUNS
            ),
            "timed_planning_events": len(timed),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_event_memory_bytes": max(peaks) if peaks else None,
        }

    gpu_before = _gpu_snapshot(gpu)
    cpu_load_before = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    result = run_round4_evaluation(
        cfg,
        task=task,
        policy_or_model=model,
        mode=str(spec["mode"]),
        identity=evaluation_identity,
        manifest=timing_manifest,
        output_dir=output_root / "timing" / "scratch" / task / arm / f"{spec['mode']}_{guidance}" / f"batch_{batch}",
        device=device,
        trace=False,
        candidate_count=64,
        flow_steps=2,
        action_flow_steps=spec["action_flow_steps"],
        solver_batch_size=1,
        candidate_batch_size=64,
        actor_warm_start_scale=1.0,
        action_flow_integrator="euler",
        cem_protocol=str(spec["cem_protocol"]),
        action_bound_mode="none",
        guidance_mode=guidance,
        guidance_step_size=0.01,
        guidance_last_steps=int(guidance_values["last_steps"]),
        guidance_inner_steps=int(guidance_values["inner_steps"]),
        guidance_max_rms_offset=0.2,
        proposal_chunk_size=512,
        allowed_protocol_variants=("legacy",),
        allow_solver_config_override=True,
        allow_eval_budget_override=False,
        policy_seed=int(config["evaluation"]["seed"]),
        latent_noise_schedule=noise_schedule,
        execute_steps=25,
        score_horizon_blocks=5,
        timing_capture_callback=capture,
    )
    if result.get("status") != "timing_capture":
        raise RuntimeError(f"timing evaluator returned {result.get('status')!r}")
    captured = dict(result["timing_capture"])
    gpu_after = _gpu_snapshot(gpu)
    cpu_load_after = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    record = {
        **identity,
        "schema_version": "round5_phase6_1_timing_v1",
        "condition_sha256": phase._canonical_sha256(identity["condition"]),
        "timing_observation_cohort_id": timing_manifest.cohort_id,
        "checkpoint": str(checkpoint.resolve()),
        "source": "real legacy_50 observation; policy inference only; environment stepping excluded",
        "precision": "fp32",
        "process_id": os.getpid(),
        "logical_cpu_count": os.cpu_count(),
        "system_load_average_before": cpu_load_before,
        "system_load_average_after": cpu_load_after,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "code_identity": {
            **phase._code_identity(),
            "diagnostics_script_sha256": _sha256_file(Path(__file__).resolve()),
            "fixed_pool_capture_script_sha256": _sha256_file(
                Path(fixed_pool.__file__).resolve()
            ),
        },
        **captured,
    }
    _atomic_json(target, record)
    print(
        json.dumps(
            {
                "status": "timed",
                "task": task,
                "arm": arm,
                "mode": spec["mode"],
                "guidance": guidance,
                "batch": batch,
                "p50_seconds": record["p50_seconds"],
                "result": str(target),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return record


def benchmark(args: argparse.Namespace) -> None:
    config = phase._load_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    preflight = phase._gpu_preflight(args.gpu, args.min_free_mib)
    validate_gpu_visibility("cuda:0")
    if preflight["memory_free_mib"] < args.min_free_mib:
        raise RuntimeError("timing GPU did not meet the required VRAM margin")
    manifest, _ = phase._cohort(args.task, config)
    checkpoint, checkpoint_sha256 = phase._checkpoint(args.task, args.arm, config)
    model, resolved = load_policy_or_model(str(checkpoint))
    if resolved is None or Path(resolved).resolve() != checkpoint.resolve():
        raise ValueError("timing checkpoint resolver changed the selected weight file")
    if args.arm == "fm" and not bool(getattr(model, "latent_flow_matching", False)):
        raise ValueError("timing FM arm checkpoint has no latent flow matching")
    model.to("cuda:0").eval()
    specs = _timing_specs()
    indexes = list(range(len(specs))) if args.indices is None else [
        int(value) for value in args.indices.split(",") if value.strip()
    ]
    if len(set(indexes)) != len(indexes) or any(index < 0 or index >= len(specs) for index in indexes):
        raise ValueError("--indices contains duplicate or out-of-range timing indexes")
    for index in indexes:
        for batch in (1, 50):
            _benchmark_one(
                task=args.task,
                arm=args.arm,
                spec=specs[index],
                batch=batch,
                config=config,
                manifest=manifest,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                model=model,
                gpu=args.gpu,
            )


def _pool_identity_path(root: Path, task: str) -> Path:
    return root / "diagnostics" / "fixed_pool_ranking" / task


def diagnose_ranking(args: argparse.Namespace) -> None:
    config = phase._load_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    phase._gpu_preflight(args.gpu, args.min_free_mib)
    validate_gpu_visibility("cuda:0")
    task = args.task
    manifest, _ = phase._cohort(task, config)
    actor_path, actor_sha = phase._checkpoint(task, "regression", config)
    fm_path, fm_sha = phase._checkpoint(task, "fm", config)
    root = _pool_identity_path(phase._resolve(config["output_root"]), task)
    summary_path = root / "summary.json"
    identity = {
        "task": task,
        "cohort_sha256": manifest.computed_sha256,
        "actor_checkpoint_sha256": actor_sha,
        "regression_scorer_sha256": actor_sha,
        "fm_scorer_sha256": fm_sha,
        "candidate_count": 64,
        "action_bound_mode": "none",
        "action_flow_steps": 2,
        "candidate_noise": "fixed policy seed 42; same original actor and replan",
        "latent_noise": "FM uses sha256(task,seed,episode_id,replan); Regression-B has no latent sampling",
        "simulator_ground_truth_episodes": len(manifest.entries) * 64,
        "code_identity": {
            **phase._code_identity(),
            "diagnostics_script_sha256": _sha256_file(Path(__file__).resolve()),
            "fixed_pool_capture_script_sha256": _sha256_file(
                Path(fixed_pool.__file__).resolve()
            ),
            "simulator_diagnostics_script_sha256": _sha256_file(
                Path(p15.__file__).resolve()
            ),
        },
    }
    if summary_path.is_file():
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError(f"existing ranking diagnostic identity mismatch: {summary_path}")
        print(json.dumps({"status": "ranking_diagnostic_resumed", "summary": str(summary_path)}))
        return

    device = "cuda:0"
    cfg = compose_eval_config(
        task,
        overrides=[
            "eval.num_eval=50",
            "eval.goal_offset_steps=25",
            "eval.eval_budget=50",
            f"seed={int(config['evaluation']['seed'])}",
            "output.save_video=false",
            f"solver.device={device}",
        ],
    )
    # The ranking truth is the physical state after exactly one generated 25-step plan.
    cfg.eval.eval_budget = 25
    cfg.world.num_envs = len(manifest.entries)
    cfg.solver.device = device
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    actor_policy, resolved_actor = load_policy_or_model(str(actor_path))
    fm_policy, resolved_fm = load_policy_or_model(str(fm_path))
    if Path(resolved_actor).resolve() != actor_path.resolve() or Path(resolved_fm).resolve() != fm_path.resolve():
        raise ValueError("ranking diagnostic checkpoint resolver changed an input path")
    actor_model = getattr(actor_policy, "model", actor_policy).to(device).eval()
    fm_model = getattr(fm_policy, "model", fm_policy).to(device).eval()
    latent_schedule = phase._LatentNoiseSchedule(
        task=task,
        seed=int(config["evaluation"]["seed"]),
        manifest=manifest,
    )

    regression = fixed_pool._capture_pool(
        task=task,
        name="regression",
        actor=actor_policy,
        verifier=None,
        actor_path=actor_path,
        manifest=manifest,
        cfg=cfg,
        dataset=dataset,
        output_dir=root / "regression_capture",
        device=device,
        candidate_count=64,
        flow_steps=2,
        training_epoch=10,
        action_bound_mode="none",
        latent_noise_schedule=latent_schedule,
        allowed_protocol_variants=("legacy",),
        allow_eval_budget_override=True,
    )
    fm_scored = fixed_pool._capture_pool(
        task=task,
        name="fm",
        actor=actor_policy,
        verifier=fm_model,
        actor_path=actor_path,
        manifest=manifest,
        cfg=cfg,
        dataset=dataset,
        output_dir=root / "fm_score_capture",
        device=device,
        candidate_count=64,
        flow_steps=2,
        training_epoch=10,
        action_bound_mode="none",
        latent_noise_schedule=latent_schedule,
        allowed_protocol_variants=("legacy",),
        allow_eval_budget_override=True,
    )
    if not np.array_equal(regression["candidates"], fm_scored["candidates"]):
        raise RuntimeError("Regression and FM scores did not receive the same actor candidate pool")
    if not np.array_equal(regression["noise"], fm_scored["noise"]):
        raise RuntimeError("the original actor candidate noise changed between scorer captures")
    pool_sha = _sha256_array(regression["candidates"])
    if regression["noise_sha256"] != fm_scored["noise_sha256"]:
        raise RuntimeError("the candidate noise provenance changed between scorers")
    fm_latent_noise_sha = fm_scored["event_metadata"].get("latent_noise_sha256")
    if not fm_latent_noise_sha:
        raise RuntimeError("the FM fixed-pool scorer did not expose its latent noise path")

    session = DatasetEvaluationSession(
        cfg, task=task, dataset=dataset, cohort=manifest.to_evaluation_cohort()
    )
    action_processor = session.process.get("action")
    if action_processor is None:
        raise RuntimeError("fixed-pool true-value replay requires the action processor")
    physical = fixed_pool._physical_actions(
        regression["candidates"],
        action_processor,
        int(cfg.plan_config.action_block),
    )
    branch_state: dict[str, Any] = {}
    outcome_root = root / "true_outcomes"
    outcome_root.mkdir(parents=True, exist_ok=True)
    all_outcomes: list[dict[str, Any]] = []
    for candidate_index in range(64):
        expected = []
        for slot, entry in enumerate(manifest.entries):
            expected.append(
                {
                    "task": task,
                    "state_id": p15._candidate_state_id(entry),
                    "slot": slot,
                    "episode_id": entry.episode_id,
                    "start_step": int(entry.start_step),
                    "row_index": int(entry.row_index),
                    "flow_steps": 2,
                    "candidate_index": candidate_index,
                    "action": regression["candidates"][slot, candidate_index].tolist(),
                    "physical_action": physical[slot, candidate_index].tolist(),
                    "candidate_pool_sha256": pool_sha,
                    "outcome_status": "pending",
                }
            )
        branch_path = outcome_root / f"candidate_{candidate_index:04d}.jsonl"
        if p15._branch_file_matches(branch_path, expected):
            all_outcomes.extend(p15._read_records(branch_path))
            continue
        episodes = p15._run_fixed_candidate(
            cfg=cfg,
            task=task,
            manifest=manifest,
            normalized_actions=regression["candidates"][:, candidate_index],
            process=session.process,
            model=actor_model,
            transform=session.transform["pixels"],
            device=device,
            output_dir=outcome_root / f"candidate_{candidate_index:04d}",
            dataset=dataset,
            branch_state=branch_state,
            capture_future_latents=False,
            evaluation_session=session,
        )
        records = fixed_pool._attach_milestones(expected, episodes)
        for record in records:
            record["outcome_status"] = "completed"
        p15._write_jsonl(branch_path, records)
        all_outcomes.extend(records)
        if (candidate_index + 1) % 8 == 0:
            print(f"simulator-grounded candidates {candidate_index + 1}/64", flush=True)

    metrics = {
        "regression": fixed_pool._rank_metrics(regression["costs"], all_outcomes, seed=61061),
        "fm": fixed_pool._rank_metrics(fm_scored["costs"], all_outcomes, seed=61061),
    }
    paired = {}
    for key in ("top1_success_5", "top1_success_25", "top1_distance_25", "physical_regret_25", "spearman_cost_distance", "pairwise_sign_accuracy"):
        left = np.asarray([row[key] for row in metrics["regression"]["by_state"]], dtype=np.float64)
        right = np.asarray([row[key] for row in metrics["fm"]["by_state"]], dtype=np.float64)
        paired[key] = fixed_pool._paired_bootstrap(left, right, seed=61062)
    regression_top1 = np.asarray(
        [row["top1_success_25"] for row in metrics["regression"]["by_state"]],
        dtype=bool,
    )
    fm_top1 = np.asarray(
        [row["top1_success_25"] for row in metrics["fm"]["by_state"]], dtype=bool
    )
    gains = int(np.sum(~regression_top1 & fm_top1))
    losses = int(np.sum(regression_top1 & ~fm_top1))
    from scipy.stats import binomtest

    paired["top1_success_25_exact_mcnemar"] = {
        "fm_only_success": gains,
        "regression_only_success": losses,
        "two_sided_p": float(
            binomtest(min(gains, losses), gains + losses, 0.5).pvalue
            if gains + losses
            else 1.0
        ),
    }
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outcome_groups = [[] for _ in manifest.entries]
    for outcome in all_outcomes:
        outcome_groups[int(outcome["slot"])].append(outcome)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    for axis, scorer in zip(axes, ("regression", "fm")):
        scorer_costs = regression["costs"] if scorer == "regression" else fm_scored["costs"]
        predicted_percentiles = []
        true_percentiles = []
        for slot, outcomes in enumerate(outcome_groups):
            ordered = sorted(outcomes, key=lambda row: int(row["candidate_index"]))
            distances = np.asarray(
                [np.nan if row.get("distance_at_25") is None else row["distance_at_25"] for row in ordered],
                dtype=np.float64,
            )
            valid = np.isfinite(distances) & np.asarray(
                [bool(row["valid_at_25"]) for row in ordered], dtype=bool
            )
            if valid.sum() < 2:
                continue
            cost_order = np.argsort(scorer_costs[slot], kind="stable")
            cost_rank = np.empty(len(cost_order), dtype=np.float64)
            cost_rank[cost_order] = np.linspace(0.0, 1.0, len(cost_order))
            distance_order = np.argsort(distances[valid], kind="stable")
            distance_rank = np.empty(int(valid.sum()), dtype=np.float64)
            distance_rank[distance_order] = np.linspace(0.0, 1.0, int(valid.sum()))
            predicted_percentiles.extend(cost_rank[valid].tolist())
            true_percentiles.extend(distance_rank.tolist())
        axis.scatter(predicted_percentiles, true_percentiles, s=6, alpha=0.14, rasterized=True)
        axis.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
        axis.set_title("Regression-B" if scorer == "regression" else "FM-B")
        axis.set_xlabel("predicted cost percentile (lower is preferred)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("simulator distance percentile (lower is better)")
    figure.suptitle(f"{task}: same actor candidate pool, 50 initial states × 64 actions")
    figure.tight_layout()
    plot_path = root / "cost_vs_true_distance.png"
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    pool_path = root / "fixed_pool.npz"
    temporary = pool_path.with_name(f".{pool_path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            candidates=regression["candidates"],
            physical_actions=physical,
            regression_costs=regression["costs"],
            fm_costs=fm_scored["costs"],
            candidate_noise=regression["noise"],
            regression_z_start=regression["z_start"],
            regression_z_goal=regression["z_goal"],
            fm_z_start=fm_scored["verifier_z_start"],
            fm_z_goal=fm_scored["verifier_z_goal"],
        )
    os.replace(temporary, pool_path)
    summary = {
        **identity,
        "schema_version": 1,
        "actor_checkpoint": str(actor_path.resolve()),
        "fm_checkpoint": str(fm_path.resolve()),
        "candidate_pool_sha256": pool_sha,
        "ranking_plot": str(plot_path.resolve()),
        "ranking_plot_sha256": _sha256_file(plot_path),
        "candidate_noise_sha256": regression["noise_sha256"],
        "latent_noise_sha256_fm": fm_latent_noise_sha,
        "fixed_pool_artifact": str(pool_path.resolve()),
        "fixed_pool_file_sha256": _sha256_file(pool_path),
        "same_candidate_pool_for_both_scorers": True,
        "candidate_sampling_stability": {
            "same_seed_and_original_actor_regenerated_identical_pool": True,
            "candidate_noise_sha256_matches_across_scorers": True,
            "fm_latent_noise_hash_recorded_and_fixed_per_replan": True,
            "candidate_pool_sha256": pool_sha,
        },
        "scorer_encoder_context": "each checkpoint encoded the same recorded initial observation and goal",
        "true_value_work": {
            "source": "simulator replay from the legacy_50 initial states",
            "candidate_episodes": len(all_outcomes),
            "candidate_pool_size_per_state": 64,
            "executed_environment_steps_per_candidate": 25,
            "total_candidate_environment_steps": len(all_outcomes) * 25,
            "separately_reported_from_closed_loop_matrix": True,
        },
        "metrics": metrics,
        "paired_fm_minus_regression": paired,
    }
    _atomic_json(summary_path, summary)
    print(json.dumps({"status": "ranking_diagnostic_complete", "summary": str(summary_path), "metrics": {name: value["mean"] for name, value in metrics.items()}}, ensure_ascii=False, sort_keys=True), flush=True)


def training_validation(args: argparse.Namespace) -> None:
    """Compute epoch-10 validation velocity and integrated error from the frozen split."""
    config = phase._load_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    preflight = phase._gpu_preflight(args.gpu, args.min_free_mib)
    validate_gpu_visibility("cuda:0")
    if preflight["memory_free_mib"] < args.min_free_mib:
        raise RuntimeError("posthoc validation GPU did not meet the required VRAM margin")
    run_dir = phase._resolve(config["training"]["output_root"]) / f"fm_b_seed3072_{args.task}"
    output_path = run_dir / "phase6_1_endpoint_validation.json"
    checkpoint, checkpoint_sha256 = phase._checkpoint(args.task, "fm", config)
    identity_path = run_dir / "phase6_1_training_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    config_path = Path(identity["resolved_training_config"])
    cfg = OmegaConf.load(config_path)
    # This offline validator does not use WandB metadata; resolving that subtree
    # would require Hydra's runtime-only `now` and `info` resolvers. Resolve the
    # dataset window length directly instead of relying on the training CLI's
    # `eval` resolver.
    cfg.pop("wandb", None)
    cfg.data.dataset.num_steps = int(cfg.num_preds) + int(cfg.history_size)
    OmegaConf.resolve(cfg)
    dataset_config = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_config.pop("name")
    dataset = load_dataset(
        dataset_name,
        cache_dir=os.environ.get("LOCAL_DATASET_DIR"),
        **dataset_config,
    )
    transforms = []
    for column in cfg.data.dataset.keys_to_load:
        if str(column).startswith("pixels"):
            continue
        transforms.append(get_column_normalizer(dataset, str(column), str(column)))
    import stable_pretraining as spt

    dataset.transform = spt.data.transforms.Compose(*transforms)
    generator = torch.Generator().manual_seed(int(cfg.seed))
    _, validation_set = spt.data.random_split(
        dataset,
        [float(cfg.train_split), 1.0 - float(cfg.train_split)],
        generator=generator,
    )
    loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=int(cfg.loader.batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    model, resolved = load_policy_or_model(str(checkpoint))
    if resolved is None or Path(resolved).resolve() != checkpoint.resolve():
        raise ValueError("posthoc validation checkpoint resolver changed the checkpoint path")
    if not bool(getattr(model, "latent_flow_matching", False)):
        raise ValueError("posthoc validation requires a latent-flow checkpoint")
    model = getattr(model, "model", model).to("cuda:0").eval()

    class ValidationPolicy:
        def __init__(self, module):
            self.model = module
            self.sigreg = lambda embeddings: embeddings.new_zeros(())
            self.rng_seed = int(cfg.seed)
            self._fast_lewam_rng_streams = {}

        @staticmethod
        def log_dict(*_args, **_kwargs):
            return None

    policy = ValidationPolicy(model)
    sums = {"velocity": 0.0, "weighted_velocity": 0.0, "endpoint_mse": 0.0}
    sample_count = 0
    for batch in loader:
        batch = {
            key: value.to("cuda:0", non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        batch_size = int(batch["pixels"].shape[0])
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            result = fast_lewam_forward(
                policy,
                batch,
                "validate",
                int(cfg.action_horizon),
                "stage_ab_fm",
                1.0,
                0.09,
                bool(cfg.detach_clean_action),
                float(cfg.latent_loss_noise_threshold),
                int(cfg.latent_action_mix_epochs),
                str(cfg.stage_b_timestep_mode),
                str(cfg.stage_b_action_source),
                stage_a_goal_index=cfg.policy.get("stage_a_goal_index", None),
            )
        sums["velocity"] += float(result["latent_flow_velocity_loss"].float().cpu()) * batch_size
        sums["weighted_velocity"] += float(
            result["weighted_latent_flow_velocity_loss"].float().cpu()
        ) * batch_size
        sums["endpoint_mse"] += float(result["latent_flow_endpoint_mse"].float().cpu()) * batch_size
        sample_count += batch_size
        if sample_count % 8192 == 0 or sample_count == len(validation_set):
            print(f"posthoc validation {sample_count}/{len(validation_set)}", flush=True)
    if sample_count != len(validation_set):
        raise RuntimeError(
            f"posthoc validation processed {sample_count} examples; expected {len(validation_set)}"
        )
    report = {
        "schema_version": 1,
        "task": args.task,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "training_identity": str(identity_path.resolve()),
        "training_identity_sha256": _sha256_file(identity_path),
        "resolved_config": str(config_path.resolve()),
        "resolved_config_sha256": _sha256_file(config_path),
        "dataset_sha256": identity.get("dataset_sha256"),
        "validation_split": "stable_pretraining.data.random_split, seed 3072, 90/10 (same function used by train.py)",
        "validation_sample_count": sample_count,
        "batch_size": int(cfg.loader.batch_size),
        "autocast_precision": "bf16-mixed",
        "latent_flow_steps": int(model.latent_flow_steps),
        "latent_flow_velocity_loss": sums["velocity"] / sample_count,
        "weighted_latent_flow_velocity_loss": sums["weighted_velocity"] / sample_count,
        "latent_flow_endpoint_mse": sums["endpoint_mse"] / sample_count,
        "metric_definition": "same validation forward as fast_lewam_forward; Euler integrated from the velocity-flow noise with K=2",
    }
    _atomic_json(output_path, report)
    print(
        json.dumps(
            {
                "status": "training_validation_complete",
                "task": args.task,
                "metrics": {
                    key: report[key]
                    for key in (
                        "latent_flow_velocity_loss",
                        "weighted_latent_flow_velocity_loss",
                        "latent_flow_endpoint_mse",
                    )
                },
                "result": str(output_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    timing_parser = subparsers.add_parser("benchmark", help="run/resume one task and arm fixed-input timing matrix")
    timing_parser.add_argument("--task", choices=phase.TASKS, required=True)
    timing_parser.add_argument("--arm", choices=phase.ARMS, required=True)
    timing_parser.add_argument("--gpu", type=int, required=True)
    timing_parser.add_argument("--min-free-mib", type=int, default=phase.MIN_FREE_MIB)
    timing_parser.add_argument("--indices", help="comma-separated indexes in the 11 full-horizon timing conditions")
    timing_parser.set_defaults(function=benchmark)
    ranking_parser = subparsers.add_parser("fixed-pool-ranking", help="compare both scorers on one original-actor pool")
    ranking_parser.add_argument("--task", choices=phase.TASKS, required=True)
    ranking_parser.add_argument("--gpu", type=int, required=True)
    ranking_parser.add_argument("--min-free-mib", type=int, default=phase.MIN_FREE_MIB)
    ranking_parser.set_defaults(function=diagnose_ranking)
    validation_parser = subparsers.add_parser("training-validation", help="compute epoch-10 held-out FM validation metrics")
    validation_parser.add_argument("--task", choices=phase.TASKS, required=True)
    validation_parser.add_argument("--gpu", type=int, required=True)
    validation_parser.add_argument("--min-free-mib", type=int, default=phase.MIN_FREE_MIB)
    validation_parser.set_defaults(function=training_validation)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
