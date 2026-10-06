"""Trace-enabled execution seam for Round 3 Phase 1.

This wrapper adapts the installed ``stable_worldmodel`` evaluator at runtime
instead of editing the external package.  The legacy evaluator still owns
goal setup and rollout semantics; the wrapper observes actions and the
environment-pool return value to produce the Phase 1 trace.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np
from omegaconf import OmegaConf

from .round3_phase1 import (
    CohortManifest,
    Round3TraceCollector,
    _PolicyTap,
    enrich_result_payload,
    summarize_episodes,
    write_round3_result,
)
from .round3_protocol import ROUND3_EVAL_DEFAULTS, ROUND3_PROTOCOL
from .round3_validation import (
    RESULT_SCHEMA_VERSION,
    trace_content_sha256,
    trace_summary_sha256,
    validate_cohort_manifest,
    validate_result_payload,
)
from .round3_runtime_audit import (
    run_independent_cpu_neutral_hold,
    validate_neutral_hold_diagnostic,
)
from .round4_action_bounds import (
    compute_normalized_action_bounds,
    summarize_executed_action_bounds,
)
from .gpu_environment import PERMITTED_PHYSICAL_GPUS


def validate_gpu_visibility(device: str) -> None:
    """Reject accidental use of GPUs outside the repository's allow-list."""
    if not str(device).startswith("cuda"):
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        raise RuntimeError("set CUDA_VISIBLE_DEVICES explicitly before GPU evaluation")
    ids = [value.strip() for value in visible.split(",") if value.strip()]
    if not ids or any(
        not value.isdigit() or int(value) not in PERMITTED_PHYSICAL_GPUS
        for value in ids
    ):
        allowed = ",".join(str(value) for value in sorted(PERMITTED_PHYSICAL_GPUS))
        raise RuntimeError(f"only physical GPUs {allowed} may be visible; got {visible!r}")


def validate_round3_config(
    cfg: Any,
    *,
    allow_solver_budget_overrides: bool = False,
    allow_evaluation_seed_override: bool = False,
) -> None:
    """Reject evaluation drift except for explicit guidance solver budgets."""
    expected = {
        "eval.goal_offset_steps": ROUND3_EVAL_DEFAULTS["goal_offset_steps"],
        "eval.eval_budget": ROUND3_EVAL_DEFAULTS["eval_budget"],
        "plan_config.horizon": ROUND3_EVAL_DEFAULTS["horizon"],
        "plan_config.receding_horizon": ROUND3_EVAL_DEFAULTS["receding_horizon"],
        "plan_config.action_block": ROUND3_EVAL_DEFAULTS["action_block"],
        "solver.num_samples": ROUND3_EVAL_DEFAULTS["num_samples"],
        "solver.n_steps": ROUND3_EVAL_DEFAULTS["n_steps"],
        "solver.topk": ROUND3_EVAL_DEFAULTS["topk"],
        "solver.var_scale": ROUND3_EVAL_DEFAULTS["var_scale"],
    }
    if not allow_evaluation_seed_override:
        expected["seed"] = ROUND3_EVAL_DEFAULTS["seed"]
    allowed = (
        {"solver.num_samples", "solver.topk"}
        if allow_solver_budget_overrides
        else set()
    )
    for path, expected_value in expected.items():
        if path in allowed:
            continue
        current = OmegaConf.select(cfg, path)
        if current != expected_value:
            raise ValueError(
                f"Phase 1 config drift at {path}: expected {expected_value!r}, got {current!r}"
            )


def _result_infos(step_result: Any) -> Any:
    """Find the info object in common EnvPool step return shapes."""
    if isinstance(step_result, tuple):
        if not step_result:
            return None
        return step_result[-1]
    return step_result


@contextmanager
def _video_slot_limit(max_slots: int | None):
    """Limit dataset-driven example videos without changing evaluated slots."""
    if max_slots is None:
        yield
        return
    from stable_worldmodel.world import world as world_module

    original = world_module.save_panel_videos

    def save_limited(video_dir, panels, fps=15):
        limited = {}
        for name, values in panels.items():
            if isinstance(values, dict):
                limited[name] = {
                    key: value for key, value in values.items() if int(key) < int(max_slots)
                }
            else:
                limited[name] = values[: int(max_slots)]
        return original(video_dir, limited, fps=fps)

    world_module.save_panel_videos = save_limited
    try:
        yield
    finally:
        world_module.save_panel_videos = original


def _parameters(cfg: Any, manifest: CohortManifest, *, save_video: bool) -> dict[str, Any]:
    return {
        "seed": int(cfg.seed),
        "environment_seed": int(cfg.seed),
        "cohort_sampling_seed": int(manifest.seed),
        "policy_seed": int(
            OmegaConf.select(
                cfg,
                "eval.policy_seed",
                default=OmegaConf.select(cfg, "solver.seed", default=cfg.seed),
            )
        ),
        "num_eval": len(manifest.entries),
        "goal_offset_steps": int(cfg.eval.goal_offset_steps),
        "eval_budget": int(cfg.eval.eval_budget),
        "horizon": int(cfg.plan_config.horizon),
        "receding_horizon": int(cfg.plan_config.receding_horizon),
        "action_block": int(cfg.plan_config.action_block),
        "start_rows": [entry.row_index for entry in manifest.entries],
        "episode_ids": [entry.episode_id for entry in manifest.entries],
        "start_steps": [entry.start_step for entry in manifest.entries],
        "cohort_id": manifest.cohort_id,
        "cohort_sha256": manifest.computed_sha256,
        "plan_config": OmegaConf.to_container(cfg.plan_config, resolve=True),
        "solver": OmegaConf.to_container(cfg.solver, resolve=True),
        "save_video": bool(save_video),
    }


def _attach_cem_archive_callback(policy: Any, cfg: Any):
    """Attach candidate capture only when the policy config has a CEM budget."""
    iterations = OmegaConf.select(cfg, "solver.n_steps")
    topk = OmegaConf.select(cfg, "solver.topk")
    if iterations is None or topk is None:
        return None

    from .cvpr_table1 import attach_cem_archive_callback

    return attach_cem_archive_callback(
        policy,
        iterations=int(iterations),
        topk=int(topk),
    )


def _bind_episode_identity(
    records: list[dict[str, Any]], manifest: CohortManifest
) -> list[dict[str, Any]]:
    """Fill identity fields omitted by the runtime collector without masking errors."""
    bound: list[dict[str, Any]] = []
    for index, (record, entry) in enumerate(zip(records, manifest.entries)):
        item = dict(record)
        item.setdefault("slot", index)
        item.setdefault("episode_id", entry.episode_id)
        item.setdefault("dataset_episode", entry.episode_id)
        item.setdefault("start_step", entry.start_step)
        item.setdefault("row_index", entry.row_index)
        item.setdefault("goal_row_index", entry.goal_row_index)
        if entry.goal_step is not None:
            item.setdefault("goal_step", entry.goal_step)
        if entry.goal_state is not None:
            item.setdefault("goal_state", list(entry.goal_state))
        bound.append(item)
    if len(bound) != len(manifest.entries):
        raise ValueError("episode records and manifest entries have different lengths")
    return bound


class _PlanningEventTap(_PolicyTap):
    """Measure a complete policy call and record which live slots replanned."""

    def __init__(self, policy: Any, *, device: str):
        super().__init__(policy)
        self.device = str(device)
        self.events: list[dict[str, Any]] = []

    def _active_slots(self, info_dict) -> list[int]:
        count = int(getattr(getattr(self.policy, "env", None), "num_envs", 1))
        terminated = info_dict.get("terminated") if isinstance(info_dict, dict) else None
        dead = (
            np.asarray(terminated, dtype=bool)
            if terminated is not None
            else np.zeros(count, dtype=bool)
        )
        flush_value = info_dict.get("_needs_flush") if isinstance(info_dict, dict) else None
        flush = (
            np.asarray(flush_value, dtype=bool)
            if flush_value is not None
            else np.zeros(count, dtype=bool)
        )
        buffers = getattr(self.policy, "_action_buffer", None)
        if buffers is None:
            return [index for index in range(min(count, len(dead))) if not dead[index]]
        return [
            index
            for index in range(min(count, len(dead), len(buffers)))
            if not dead[index] and (flush[index] or not buffers[index])
        ]

    def _algorithm_events(self, before: dict[str, int]) -> list[dict[str, Any]]:
        sources = []
        direct = getattr(self.policy, "planning_events", None)
        if isinstance(direct, list):
            sources.append(("policy", direct))
        cem_capture = getattr(self.policy, "cvpr_cem_capture", None)
        cem_events = getattr(cem_capture, "events", None)
        if isinstance(cem_events, list):
            sources.append(("policy.cvpr_cem_capture", cem_events))
        solver = getattr(self.policy, "solver", None)
        for name in ("events", "planning_events"):
            value = getattr(solver, name, None)
            if isinstance(value, list):
                sources.append((f"solver.{name}", value))
        underlying = getattr(solver, "_solver", None)
        value = getattr(underlying, "planning_events", None)
        if isinstance(value, list):
            sources.append(("solver._solver.planning_events", value))
        result = []
        seen = set()
        for source, values in sources:
            if id(values) in seen:
                continue
            seen.add(id(values))
            start = before.get(source, 0)
            result.extend(
                {"source": source, "source_index": index, **dict(event)}
                for index, event in enumerate(values[start:], start=start)
            )
        return result

    def _call(self, fn, *args, **kwargs):
        info_dict = args[0] if args else kwargs.get("info_dict", {})
        active = self._active_slots(info_dict)
        if not active:
            action = fn(*args, **kwargs)
            self.last_action = action
            return action
        solver = getattr(self.policy, "solver", None)
        before = {}
        for source, value in (
            ("policy", getattr(self.policy, "planning_events", None)),
            (
                "policy.cvpr_cem_capture",
                getattr(getattr(self.policy, "cvpr_cem_capture", None), "events", None),
            ),
            ("solver.events", getattr(solver, "events", None)),
            ("solver.planning_events", getattr(solver, "planning_events", None)),
            (
                "solver._solver.planning_events",
                getattr(getattr(solver, "_solver", None), "planning_events", None),
            ),
        ):
            if isinstance(value, list):
                before[source] = len(value)
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        action = fn(*args, **kwargs)
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        self.events.append(
            {
                "event_index": len(self.events),
                "replan_indices": [int(index) for index in active],
                "environment_batch_size": len(active),
                "wall_seconds": float(elapsed),
                "algorithm_events": self._algorithm_events(before),
            }
        )
        self.last_action = action
        return action

    def get_action(self, *args, **kwargs):
        return self._call(self.policy.get_action, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        call = self.policy if callable(self.policy) else self.policy.get_action
        return self._call(call, *args, **kwargs)


def _bind_planning_events(records, events):
    counts = [0 for _ in records]
    indices = [[] for _ in records]
    amortized = [0.0 for _ in records]
    for event in events:
        slots = [int(index) for index in event["replan_indices"]]
        active = max(1, len(slots))
        for slot in slots:
            if 0 <= slot < len(records):
                counts[slot] += 1
                indices[slot].append(int(event["event_index"]))
                amortized[slot] += float(event["wall_seconds"]) / active
    for index, record in enumerate(records):
        record["episode_replan_count"] = counts[index]
        record["episode_replan_event_indices"] = indices[index]
        record["episode_amortized_planning_seconds"] = amortized[index]
    return {
        "planning_events": events,
        "batch_replan_count": len(events),
        "batch_planning_wall_seconds": float(
            sum(float(event["wall_seconds"]) for event in events)
        ),
    }


def run_round3_evaluation(
    cfg: Any,
    *,
    task: str,
    policy_or_model: Any,
    identity: Any,
    manifest: CohortManifest,
    output_dir: str | Path,
    dataset: Any | None = None,
    trace_output_dir: str | Path | None = None,
    device: str | None = None,
    trace: bool = True,
    allow_solver_budget_overrides: bool = False,
    planning_timing: bool = False,
    allow_evaluation_seed_override: bool = False,
    allow_cohort_seed_mismatch: bool = False,
    timing_capture_callback: Callable[..., dict[str, Any]] | None = None,
    video_slots: int | None = None,
) -> dict[str, Any]:
    """Run one registered weight and publish a Phase 1 result plus trace."""
    device = str(device or cfg.solver.get("device", "cuda"))
    validate_gpu_visibility(device)
    validate_round3_config(
        cfg,
        allow_solver_budget_overrides=allow_solver_budget_overrides,
        allow_evaluation_seed_override=allow_evaluation_seed_override,
    )
    expected_count = int(cfg.eval.num_eval)
    validate_cohort_manifest(manifest, task=task, expected_count=expected_count)
    if int(cfg.eval.goal_offset_steps) != int(manifest.goal_offset_steps):
        raise ValueError("evaluation goal offset differs from the frozen cohort")
    if not allow_cohort_seed_mismatch and int(cfg.seed) != int(manifest.seed):
        raise ValueError("evaluation seed differs from the frozen cohort")

    policy_identity = {
        "entrypoint": identity.entrypoint,
        "policy_kind": identity.policy_kind,
        "checkpoint": identity.checkpoint,
        "epoch": identity.epoch,
        "stage": identity.stage,
    }
    neutral_hold_diagnostic = run_independent_cpu_neutral_hold(
        task,
        seed=int(manifest.seed),
        protocol=ROUND3_PROTOCOL,
        protocol_variant=manifest.protocol_variant,
        stage=identity.stage,
        cohort_id=manifest.cohort_id,
        cohort_sha256=manifest.computed_sha256,
        policy_identity=policy_identity,
    )
    validate_neutral_hold_diagnostic(
        neutral_hold_diagnostic,
        task=task,
        expected_policy_identity=policy_identity,
        expected_protocol=ROUND3_PROTOCOL,
        expected_protocol_variant=manifest.protocol_variant,
        expected_stage=identity.stage,
        expected_cohort_id=manifest.cohort_id,
        expected_cohort_sha256=manifest.computed_sha256,
        expected_seed=int(manifest.seed),
    )

    from .eval import DatasetEvaluationSession, evaluate_from_dataset_compat

    session = DatasetEvaluationSession(
        cfg,
        task=task,
        dataset=dataset,
        cohort=manifest.to_evaluation_cohort(),
    )
    policy = session._build_policy(policy_or_model, identity, device)
    if timing_capture_callback is None:
        _attach_cem_archive_callback(policy, cfg)
    if planning_timing and hasattr(policy, "solver"):
        # Reuse the Round 4 timing seam so LeWM's reference measurement uses
        # the same synchronized CEM boundary as the Fast-LeWAM conditions.
        from .round4_eval import _TimedSolver

        timed_solver = _TimedSolver(policy.solver)
        policy.solver = timed_solver
        policy.planning_events = timed_solver.events
    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    world_cfg["max_episode_steps"] = 2 * int(cfg.eval.eval_budget)
    save_video = bool(cfg.get("output", {}).get("save_video", False))
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    video_dir = target / "videos"
    if save_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    world = session.world_factory(**world_cfg, image_shape=(224, 224))
    collector = None
    original_step = None
    envs = getattr(world, "envs", None)
    # A few legacy/test session adapters intentionally expose only the
    # dataset, cohort and world factory.  Action-bound diagnostics are
    # optional for those adapters, so treat a missing process registry as an
    # empty one rather than changing the evaluator's compatibility contract.
    process = getattr(session, "process", {})
    action_processor = (
        process.get("action") if hasattr(process, "get") else None
    )
    action_bounds = None
    if action_processor is not None and envs is not None:
        action_bounds = compute_normalized_action_bounds(
            envs.single_action_space,
            action_processor,
            action_block=int(cfg.plan_config.action_block),
        )
    if trace:
        action_space = getattr(envs, "single_action_space", None)
        collector = Round3TraceCollector(
            task,
            manifest,
            action_block=int(cfg.plan_config.action_block),
            action_space=action_space,
            action_processor=action_processor,
            normalized_action_bounds=action_bounds,
        )
    try:
        if timing_capture_callback is not None:
            from .round4_eval import _FirstActionTimingTap

            world.set_policy(_FirstActionTimingTap(policy, timing_capture_callback))
        else:
            policy_tap = _PlanningEventTap(policy, device=device)
            world.set_policy(policy_tap)
        if collector is not None and envs is not None and hasattr(envs, "step"):
            original_step = envs.step
            step_counter = {"value": 0}

            def traced_step(actions, *args, **kwargs):
                result = original_step(actions, *args, **kwargs)
                step_counter["value"] += 1
                collector.record_step(
                    actions,
                    _result_infos(result),
                    raw_env_step=step_counter["value"],
                )
                return result

            envs.step = traced_step
        started = time.perf_counter()
        try:
            with _video_slot_limit(video_slots if save_video else None):
                metrics = evaluate_from_dataset_compat(
                    world=world,
                    dataset=session.dataset,
                    eval_start_idx=session.cohort.start_steps,
                    eval_episodes=session.cohort.episode_ids,
                    cfg=cfg,
                    video_path=video_dir,
                    save_video=save_video,
                )
        except Exception as exc:
            from .round4_eval import _TimingCaptureComplete

            if not isinstance(exc, _TimingCaptureComplete):
                raise
            return {
                "status": "timing_capture",
                "timing_capture": exc.result,
                "policy_metadata": dict(getattr(policy, "metadata", lambda: {})()),
            }
        elapsed = time.perf_counter() - started
    finally:
        if envs is not None and original_step is not None:
            envs.step = original_step
        if hasattr(world, "close"):
            world.close()
        elif envs is not None and hasattr(envs, "close"):
            envs.close()

    successes = np.asarray(metrics["episode_successes"], dtype=bool).reshape(-1)
    if len(successes) != len(manifest.entries):
        raise ValueError("environment returned a success vector with the wrong length")
    raw_trace_records = (
        collector.finalize(successes.tolist(), eval_budget=int(cfg.eval.eval_budget))
        if collector is not None
        else None
    )
    episode_records = raw_trace_records
    if episode_records is None:
        episode_records = [
            {
                "slot": index,
                "episode_id": entry.episode_id,
                "start_step": entry.start_step,
                "row_index": entry.row_index,
                "success": bool(successes[index]),
            }
            for index, entry in enumerate(manifest.entries)
        ]
    episode_records = _bind_episode_identity(episode_records, manifest)
    batch_planning = (
        _bind_planning_events(episode_records, policy_tap.events)
        if trace and timing_capture_callback is None
        else {"planning_events": [], "batch_replan_count": 0, "batch_planning_wall_seconds": 0.0}
    )
    trace_records = episode_records if raw_trace_records is not None else None
    base = {
        "task": task,
        "local_dataset": str(cfg.eval.dataset_name),
        "benchmark_dataset": str(cfg.eval.get("benchmark_dataset_name", cfg.eval.dataset_name)),
        "entrypoint": identity.entrypoint,
        "policy_kind": identity.policy_kind,
        "checkpoint": identity.checkpoint,
        "epoch": identity.epoch,
        "stage": identity.stage,
        "parameters": _parameters(cfg, manifest, save_video=save_video),
        "evaluation_seconds": float(elapsed),
        "success_rate": float(successes.mean()),
        "runtime_success_rate_percent": float(metrics["success_rate"]),
        "episodes": episode_records,
        **batch_planning,
    }
    payload = enrich_result_payload(
        base,
        manifest=manifest,
        protocol_variant=manifest.protocol_variant,
        trace_records=trace_records,
    )
    payload["neutral_hold_diagnostic"] = neutral_hold_diagnostic
    cem_capture = getattr(policy, "cvpr_cem_capture", None)
    if cem_capture is not None:
        payload["cem_candidate_archive"] = cem_capture.events
    if planning_timing:
        from .round4_eval import _planning_summary

        payload["round4_planning"] = {
            "cem_protocol": "legacy",
            "action_bound_mode": "none",
            "action_flow_steps": None,
            "action_flow_integrator": "not_applicable",
            **_planning_summary(policy),
        }
        if action_bounds is not None:
            payload["round4_planning"]["action_bounds"] = action_bounds.metadata(
                action_dim=None
            )
            payload["round4_planning"]["executed_action_bounds"] = (
                summarize_executed_action_bounds(
                    payload["episodes"],
                    action_bounds,
                    action_processor,
                )
            )
    payload.setdefault("status", "ok")
    payload["schema_version"] = RESULT_SCHEMA_VERSION
    payload["cohort_kind"] = manifest.cohort_kind
    if trace_records is None:
        payload["summary"] = summarize_episodes(payload["episodes"])
        payload["success_rate"] = payload["summary"]["success_rate"]
    else:
        payload["trace_content_sha256"] = trace_content_sha256(trace_records)
        payload["trace_sha256"] = payload["trace_content_sha256"]
        payload["trace_summary_sha256"] = trace_summary_sha256(payload["summary"])
    validate_result_payload(
        payload,
        manifest=manifest,
        expected_count=expected_count,
        trace_records=trace_records,
    )
    return write_round3_result(
        payload,
        target,
        trace_records=trace_records,
        trace_output_dir=trace_output_dir,
    )


__all__ = ["run_round3_evaluation", "validate_gpu_visibility", "validate_round3_config"]
