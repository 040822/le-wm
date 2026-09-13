"""Standalone Round 4 evaluator built on the frozen Round 3 cohort runtime."""

from __future__ import annotations

import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from source.policy.round4 import make_round4_policy

from .round3_phase1 import (
    CohortManifest,
    Round3TraceCollector,
    _PolicyTap,
    enrich_result_payload,
    summarize_episodes,
    write_round3_result,
)
from .round3_validation import (
    RESULT_SCHEMA_VERSION,
    trace_content_sha256,
    trace_summary_sha256,
    validate_cohort_manifest,
    validate_result_payload,
)
from .round4_protocol import (
    ROUND4_DEFAULTS,
    mode_spec,
    validate_round4_mode,
)
from .gpu_environment import configure_mujoco_egl_device


def _sync_solver_device(solver: Any) -> None:
    """Synchronize a CUDA CEM solver before wall-clock measurements."""
    device = getattr(solver, "device", None)
    if device is None or not str(device).startswith("cuda"):
        return
    import torch

    torch.cuda.synchronize(device)


class _TimedSolver:
    """Delegate to a frozen solver while recording one event per CEM solve."""

    def __init__(self, solver: Any):
        self._solver = solver
        self.events: list[dict[str, Any]] = []

    def __getattr__(self, name: str):
        return getattr(self._solver, name)

    def _run(self, call, *args, **kwargs):
        info_dict = kwargs.get("info_dict")
        if info_dict is None and args:
            info_dict = args[0]
        try:
            first_value = next(iter(info_dict.values()))
            environment_batch_size = len(first_value)
        except (AttributeError, StopIteration, TypeError):
            environment_batch_size = 0
        batch_size = max(1, int(getattr(self._solver, "batch_size", 1)))
        n_steps = max(1, int(getattr(self._solver, "n_steps", 1)))
        device = getattr(self._solver, "device", None)
        peak_memory_bytes = None
        if device is not None and str(device).startswith("cuda"):
            import torch

            torch.cuda.reset_peak_memory_stats(device)
        _sync_solver_device(self._solver)
        started = time.perf_counter()
        result = call(*args, **kwargs)
        _sync_solver_device(self._solver)
        elapsed = time.perf_counter() - started
        if device is not None and str(device).startswith("cuda"):
            peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
        self.events.append(
            {
                "encode_seconds": 0.0,
                "proposal_seconds": float(elapsed),
                "verify_seconds": 0.0,
                "flow_seconds": 0.0,
                "idm_seconds": 0.0,
                "planning_seconds": float(elapsed),
                "environment_batch_size": int(environment_batch_size),
                "peak_memory_bytes": peak_memory_bytes,
                "forward_count": int(
                    n_steps
                    * ((environment_batch_size + batch_size - 1) // batch_size)
                ),
            }
        )
        return result

    def __call__(self, *args, **kwargs):
        return self._run(self._solver, *args, **kwargs)

    def solve(self, *args, **kwargs):
        return self._run(self._solver.solve, *args, **kwargs)


def _time_cem_policy(policy: Any, mode: str) -> None:
    """Install timing only for Round 4's P1/P2 CEM policies."""
    if mode not in {"P1", "P2"}:
        return
    solver = getattr(policy, "solver", None)
    if solver is None or not hasattr(solver, "solve"):
        return
    timed_solver = _TimedSolver(solver)
    policy.solver = timed_solver
    policy.planning_events = timed_solver.events


def validate_gpu_visibility(device: str) -> None:
    """Require one permitted physical GPU and bind MuJoCo/EGL to it."""
    if not str(device).startswith("cuda"):
        return
    # This shared guard validates the physical allow-list, rejects multiple
    # visible devices, and sets MUJOCO_GL/PYOPENGL_PLATFORM/EGL explicitly
    # before the evaluator constructs any rendering environment.
    configure_mujoco_egl_device()


def validate_round4_config(cfg: Any, mode: str) -> None:
    """Validate frozen shared settings without forcing CEM on P3/P4."""
    mode = validate_round4_mode(mode)
    expected = {
        "seed": ROUND4_DEFAULTS["seed"],
        "eval.goal_offset_steps": ROUND4_DEFAULTS["goal_offset_steps"],
        "eval.eval_budget": ROUND4_DEFAULTS["eval_budget"],
        "plan_config.horizon": ROUND4_DEFAULTS["horizon"],
        "plan_config.receding_horizon": ROUND4_DEFAULTS["receding_horizon"],
        "plan_config.action_block": ROUND4_DEFAULTS["action_block"],
    }
    for path, expected_value in expected.items():
        current = OmegaConf.select(cfg, path)
        if current != expected_value:
            raise ValueError(
                f"Round 4 config drift at {path}: expected {expected_value!r}, "
                f"got {current!r}"
            )
    if mode in {"P1", "P2"}:
        for path, expected_value in (
            ("solver.num_samples", ROUND4_DEFAULTS["cem"]["num_samples"]),
            ("solver.n_steps", ROUND4_DEFAULTS["cem"]["n_steps"]),
            ("solver.topk", ROUND4_DEFAULTS["cem"]["topk"]),
            ("solver.var_scale", ROUND4_DEFAULTS["cem"]["var_scale"]),
        ):
            current = OmegaConf.select(cfg, path)
            if current != expected_value:
                raise ValueError(
                    f"Round 4 CEM config drift at {path}: expected {expected_value!r}, "
                    f"got {current!r}"
                )


def _identity_value(identity: Any, name: str, default=None):
    if isinstance(identity, dict):
        return identity.get(name, default)
    return getattr(identity, name, default)


def _planning_summary(policy: Any) -> dict[str, Any]:
    events = list(getattr(policy, "planning_events", ()))
    if not events:
        return {"replans": 0, "planning_median_seconds": None, "planning_p95_seconds": None}
    totals = [
        float(item.get("encode_seconds", 0.0))
        + float(item.get("proposal_seconds", 0.0))
        + float(item.get("verify_seconds", 0.0))
        for item in events
    ]
    result = {
        "replans": len(events),
        "planning_median_seconds": float(median(totals)),
        "planning_p95_seconds": float(np.quantile(totals, 0.95)),
        "encode_median_seconds": float(median(float(item.get("encode_seconds", 0.0)) for item in events)),
        "proposal_median_seconds": float(median(float(item.get("proposal_seconds", 0.0)) for item in events)),
        "flow_median_seconds": float(median(float(item.get("flow_seconds", 0.0)) for item in events)),
        "idm_median_seconds": float(median(float(item.get("idm_seconds", 0.0)) for item in events)),
        "verifier_median_seconds": float(median(float(item.get("verify_seconds", 0.0)) for item in events)),
        "forward_events": len(events),
        "forward_count": int(sum(int(item.get("forward_count", 0)) for item in events)),
    }
    peak_values = [
        int(item["peak_memory_bytes"])
        for item in events
        if item.get("peak_memory_bytes") is not None
    ]
    result["peak_memory_bytes"] = max(peak_values) if peak_values else None
    return result


def run_round4_evaluation(
    cfg: Any,
    *,
    task: str,
    policy_or_model: Any,
    mode: str,
    identity: Any,
    manifest: CohortManifest,
    output_dir: str | Path,
    dataset: Any | None = None,
    trace_output_dir: str | Path | None = None,
    device: str | None = None,
    trace: bool = True,
    candidate_count: int = 64,
    flow_steps: int = 16,
    solver_batch_size: int = 1,
    candidate_batch_size: int | None = None,
) -> dict[str, Any]:
    """Run a Round 4 mode and publish cohort-bound result and trace artifacts."""
    mode = validate_round4_mode(mode)
    device = str(device or cfg.get("solver", {}).get("device", "cuda"))
    validate_gpu_visibility(device)
    validate_round4_config(cfg, mode)
    if mode in {"P0-shuf", "P4-first"} and manifest.cohort_kind == "final":
        raise ValueError(f"{mode} is a development-only Round 4 diagnostic")
    if mode in {"P3", "P4", "P4-first"}:
        if int(candidate_count) != int(ROUND4_DEFAULTS["best_of_n"]["num_candidates"]):
            raise ValueError("Round 4 best-of-N candidate count is frozen at 64")
        if int(flow_steps) != int(ROUND4_DEFAULTS["best_of_n"]["flow_steps"]):
            raise ValueError("Round 4 best-of-N flow steps are frozen at 16")
    validate_cohort_manifest(
        manifest, task=task, expected_count=int(cfg.eval.num_eval)
    )
    if manifest.protocol_variant != "round3_revised":
        raise ValueError(
            "Round 4 requires the frozen round3_revised cohort variant; "
            f"got {manifest.protocol_variant!r}"
        )
    if int(cfg.eval.goal_offset_steps) != int(manifest.goal_offset_steps):
        raise ValueError("evaluation goal offset differs from the frozen cohort")
    if int(cfg.seed) != int(manifest.seed):
        raise ValueError("evaluation seed differs from the frozen cohort")

    from .eval import DatasetEvaluationSession, evaluate_from_dataset_compat

    session = DatasetEvaluationSession(
        cfg,
        task=task,
        dataset=dataset,
        cohort=manifest.to_evaluation_cohort(),
    )
    policy = make_round4_policy(
        policy_or_model,
        mode=mode,
        solver_cfg=cfg.solver,
        plan_config=cfg.plan_config,
        process=session.process,
        transform=session.transform,
        device=device,
        seed=int(cfg.seed),
        candidate_count=candidate_count,
        flow_steps=flow_steps,
        solver_batch_size=solver_batch_size,
        candidate_batch_size=candidate_batch_size,
    )
    _time_cem_policy(policy, mode)
    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    world_cfg["max_episode_steps"] = 2 * int(cfg.eval.eval_budget)
    save_video = bool(cfg.get("output", {}).get("save_video", False))
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    video_dir = target / "videos"
    if save_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    collector = None
    original_step = None
    world = None
    envs = None
    try:
        world = session.world_factory(**world_cfg, image_shape=(224, 224))
        envs = getattr(world, "envs", None)
        if trace:
            collector = Round3TraceCollector(
                task,
                manifest,
                action_block=int(cfg.plan_config.action_block),
                action_space=getattr(envs, "single_action_space", None),
            )
        world.set_policy(_PolicyTap(policy))
        if collector is not None and envs is not None and hasattr(envs, "step"):
            original_step = envs.step
            step_counter = {"value": 0}

            def traced_step(actions, *args, **kwargs):
                result = original_step(actions, *args, **kwargs)
                step_counter["value"] += 1
                infos = result[-1] if isinstance(result, tuple) and result else result
                collector.record_step(
                    actions, infos, raw_env_step=step_counter["value"]
                )
                return result

            envs.step = traced_step
        started = time.perf_counter()
        metrics = evaluate_from_dataset_compat(
            world=world,
            dataset=session.dataset,
            eval_start_idx=session.cohort.start_steps,
            eval_episodes=session.cohort.episode_ids,
            cfg=cfg,
            video_path=video_dir,
            save_video=save_video,
        )
        elapsed = time.perf_counter() - started
    finally:
        if envs is not None and original_step is not None:
            envs.step = original_step
        if world is not None and hasattr(world, "close"):
            world.close()
        elif envs is not None and hasattr(envs, "close"):
            envs.close()

    successes = np.asarray(metrics["episode_successes"], dtype=bool).reshape(-1)
    if len(successes) != len(manifest.entries):
        raise ValueError("environment returned a success vector with the wrong length")
    records = (
        collector.finalize(successes.tolist(), eval_budget=int(cfg.eval.eval_budget))
        if collector is not None
        else [
            {
                "slot": index,
                "episode_id": entry.episode_id,
                "start_step": entry.start_step,
                "row_index": entry.row_index,
                "success": bool(successes[index]),
            }
            for index, entry in enumerate(manifest.entries)
        ]
    )
    for index, (record, entry) in enumerate(zip(records, manifest.entries)):
        record.setdefault("slot", index)
        record.setdefault("episode_id", entry.episode_id)
        record.setdefault("dataset_episode", entry.episode_id)
        record.setdefault("start_step", entry.start_step)
        record.setdefault("row_index", entry.row_index)
        record.setdefault("goal_row_index", entry.goal_row_index)

    spec = mode_spec(mode)
    planning = _planning_summary(policy)
    policy_meta = dict(spec)
    policy_meta.update(getattr(policy, "metadata", lambda: {})())
    policy_meta.update(planning)
    for record in records:
        record.setdefault("planning", dict(policy_meta))
    base = {
        "task": task,
        "local_dataset": str(cfg.eval.dataset_name),
        "benchmark_dataset": str(cfg.eval.get("benchmark_dataset_name", cfg.eval.dataset_name)),
        "entrypoint": _identity_value(identity, "entrypoint", "round4"),
        "policy_kind": _identity_value(identity, "policy_kind", "round4"),
        "checkpoint": _identity_value(identity, "checkpoint"),
        "epoch": _identity_value(identity, "epoch"),
        "stage": mode,
        "parameters": {
            "seed": int(cfg.seed),
            "num_eval": len(manifest.entries),
            "goal_offset_steps": int(cfg.eval.goal_offset_steps),
            "eval_budget": int(cfg.eval.eval_budget),
            "horizon": int(cfg.plan_config.horizon),
            "receding_horizon": int(cfg.plan_config.receding_horizon),
            "action_block": int(cfg.plan_config.action_block),
            "cohort_id": manifest.cohort_id,
            "cohort_sha256": manifest.computed_sha256,
            "proposal": policy_meta,
            "cem": (
                OmegaConf.to_container(cfg.solver, resolve=True)
                if mode in {"P1", "P2"}
                else None
            ),
            "solver_batch_size": int(solver_batch_size),
        },
        "evaluation_seconds": float(elapsed),
        "success_rate": float(successes.mean()),
        "runtime_success_rate_percent": float(metrics["success_rate"]),
        "episodes": records,
    }
    payload = enrich_result_payload(
        base,
        manifest=manifest,
        protocol_variant=manifest.protocol_variant,
        trace_records=records if collector is not None else None,
    )
    payload["round4_mode"] = mode
    payload["round4_planning"] = policy_meta
    payload.setdefault("status", "ok")
    payload["schema_version"] = RESULT_SCHEMA_VERSION
    payload["cohort_kind"] = manifest.cohort_kind
    if collector is not None:
        payload["trace_content_sha256"] = trace_content_sha256(records)
        payload["trace_sha256"] = payload["trace_content_sha256"]
        payload["trace_summary_sha256"] = trace_summary_sha256(payload["summary"])
    else:
        payload["summary"] = summarize_episodes(records)
        payload["success_rate"] = payload["summary"]["success_rate"]
    validate_result_payload(
        payload,
        manifest=manifest,
        expected_count=len(manifest.entries),
        trace_records=records if collector is not None else None,
    )
    return write_round3_result(
        payload,
        target,
        trace_records=records if collector is not None else None,
        trace_output_dir=trace_output_dir,
    )


__all__ = [
    "run_round4_evaluation",
    "validate_gpu_visibility",
    "validate_round4_config",
]
