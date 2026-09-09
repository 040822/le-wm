"""Trace-enabled execution seam for Round 3 Phase 1.

This wrapper adapts the installed ``stable_worldmodel`` evaluator at runtime
instead of editing the external package.  The legacy evaluator still owns
goal setup and rollout semantics; the wrapper observes actions and the
environment-pool return value to produce the Phase 1 trace.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

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


def validate_gpu_visibility(device: str) -> None:
    """Reject accidental use of GPUs outside the repository's allow-list."""
    if not str(device).startswith("cuda"):
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        raise RuntimeError("set CUDA_VISIBLE_DEVICES explicitly before GPU evaluation")
    ids = [value.strip() for value in visible.split(",") if value.strip()]
    if not ids or any(not value.isdigit() or int(value) not in range(4) for value in ids):
        raise RuntimeError(f"only physical GPUs 0,1,2,3 may be visible; got {visible!r}")


def validate_round3_config(cfg: Any) -> None:
    """Reject any evaluation config that drifts from the frozen Phase 1 values."""
    expected = {
        "seed": ROUND3_EVAL_DEFAULTS["seed"],
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
    for path, expected_value in expected.items():
        current = OmegaConf.select(cfg, path)
        if current != expected_value:
            raise ValueError(f"Phase 1 config drift at {path}: expected {expected_value!r}, got {current!r}")


def _result_infos(step_result: Any) -> Any:
    """Find the info object in common EnvPool step return shapes."""
    if isinstance(step_result, tuple):
        if not step_result:
            return None
        return step_result[-1]
    return step_result


def _parameters(cfg: Any, manifest: CohortManifest, *, save_video: bool) -> dict[str, Any]:
    return {
        "seed": int(cfg.seed),
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
) -> dict[str, Any]:
    """Run one registered weight and publish a Phase 1 result plus trace."""
    device = str(device or cfg.solver.get("device", "cuda"))
    validate_gpu_visibility(device)
    validate_round3_config(cfg)
    expected_count = int(cfg.eval.num_eval)
    validate_cohort_manifest(manifest, task=task, expected_count=expected_count)
    if int(cfg.eval.goal_offset_steps) != int(manifest.goal_offset_steps):
        raise ValueError("evaluation goal offset differs from the frozen cohort")
    if int(cfg.seed) != int(manifest.seed):
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
    if trace:
        action_space = getattr(envs, "single_action_space", None)
        collector = Round3TraceCollector(
            task,
            manifest,
            action_block=int(cfg.plan_config.action_block),
            action_space=action_space,
        )
    try:
        world.set_policy(_PolicyTap(policy))
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
    }
    payload = enrich_result_payload(
        base,
        manifest=manifest,
        protocol_variant=manifest.protocol_variant,
        trace_records=trace_records,
    )
    payload["neutral_hold_diagnostic"] = neutral_hold_diagnostic
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
