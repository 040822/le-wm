"""Reusable Round 3 online collection and cohort-pool helpers.

The command-line entry points in :mod:`scripts` call these functions, while
the experiment modules use them directly.  Keeping the implementation here
prevents the reusable source layer from depending on a CLI module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from source.common.eval import compose_eval_config, get_dataset
from source.common.round3_phase1 import _jsonable, sha256_file
from source.common.round3_sampling import build_revised_cohorts


ROOT = Path(__file__).resolve().parents[2]


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def derive_online_pool(
    *, output: Path, online_count: int | None = None, task: str = "cube"
) -> dict[str, Any]:
    """Derive the episode-disjoint online pool from the frozen cohorts."""
    cfg = compose_eval_config(task, ("output.save_video=false",))
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    manifests = build_revised_cohorts(
        dataset,
        task=task,
        seed=int(cfg.seed),
        goal_offset_steps=int(cfg.eval.goal_offset_steps),
        dev_count=50,
        final_count=200,
        online_count=online_count,
    )
    dev = manifests["dev"]
    final = manifests["final"]
    online_ids = tuple(dev.episode_split["online"])
    if online_count is not None:
        online_ids = online_ids[: int(online_count)]
    payload = {
        "schema_version": 1,
        "phase": "round3_phase2",
        "task": task,
        "protocol": "round3_revised",
        "seed": int(cfg.seed),
        "goal_offset_steps": int(cfg.eval.goal_offset_steps),
        "dataset_name": str(cfg.eval.dataset_name),
        "source_cohorts": {
            "dev": {
                "path": f"outputs/round3/phase1/cohorts/{task}/dev_round3_revised.json",
                "sha256": sha256_file(
                    ROOT
                    / f"outputs/round3/phase1/cohorts/{task}/dev_round3_revised.json"
                ),
            },
            "final": {
                "path": f"outputs/round3/phase1/cohorts/{task}/final.json",
                "sha256": sha256_file(
                    ROOT / f"outputs/round3/phase1/cohorts/{task}/final.json"
                ),
            },
        },
        "excluded_episode_ids": [
            _jsonable(entry.episode_id) for entry in (*dev.entries, *final.entries)
        ],
        "online_episode_ids": [_jsonable(value) for value in online_ids],
    }
    payload["content_sha256"] = _canonical_hash(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    return payload


def _online_start_pairs(
    dataset: Any,
    episode_ids: list[Any],
    *,
    goal_offset: int,
    seed: int,
    limit: int,
) -> list[tuple[Any, int]]:
    """Select deterministic valid episode/start-step pairs."""
    episode_column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    all_episodes = np.asarray(dataset.get_col_data(episode_column))
    all_steps = np.asarray(dataset.get_col_data("step_idx"), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    pairs = []
    for episode in episode_ids:
        rows = np.flatnonzero(all_episodes == episode)
        if len(rows) == 0:
            continue
        max_step = int(all_steps[rows].max())
        valid = rows[all_steps[rows] + int(goal_offset) < max_step + 1]
        if len(valid) == 0:
            continue
        row = int(valid[int(rng.integers(0, len(valid)))])
        pairs.append((episode, int(all_steps[row])))
        if len(pairs) >= int(limit):
            break
    if not pairs:
        raise ValueError("online pool has no valid episode/start pairs")
    return pairs


def collect_online(
    *,
    pool_path: Path,
    checkpoint: Path,
    output: Path,
    device: str,
    episodes: int,
    chunk_size: int,
    target_environment_steps: int | None = None,
    pool_offset: int = 0,
    solver_samples: int | None = None,
    solver_steps: int | None = None,
    solver_topk: int | None = None,
    model_override: torch.nn.Module | None = None,
    resolved_checkpoint_override: Path | None = None,
    allow_empty_replay: bool = False,
    actor_warm_start: bool = False,
    guidance_mode: str = "none",
    guidance_step_size: float = 0.01,
    guidance_last_steps: int = 5,
    guidance_inner_steps: int = 5,
    guidance_max_rms_offset: float = 0.20,
    collector_version: str | None = None,
    task: str = "cube",
) -> dict[str, Any]:
    """Collect executed transitions with the frozen task CEM policy."""
    from source.common.checkpoint import load_policy_or_model
    from source.common.eval import (
        compose_eval_config,
        fit_eval_processors,
        get_dataset,
        img_transform,
    )
    from source.common.round3_eval import validate_gpu_visibility
    from source.experiments.round3_phase2 import concatenate_replays, save_transition_replay
    from source.experiments.round3_phase2_collection import ExecutedTransitionRecorder
    from source.model.fast_lewam.jepa import FastLeWAM
    from source.policy.dispatch import make_world_policy

    validate_gpu_visibility(device)
    if str(device).startswith("cuda"):
        from source.common.gpu_environment import configure_mujoco_egl_device

        configure_mujoco_egl_device()
    from omegaconf import OmegaConf
    import stable_worldmodel as swm

    if int(pool_offset) < 0:
        raise ValueError("pool_offset must be non-negative")
    if target_environment_steps is not None and int(target_environment_steps) < 1:
        raise ValueError("target_environment_steps must be positive")
    if target_environment_steps is not None and int(chunk_size) != 1:
        raise ValueError(
            "exact target_environment_steps requires chunk_size=1 so the final "
            "episode budget can be clipped without overshooting"
        )
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    cfg = compose_eval_config(task, (f"solver.device={device}", "output.save_video=false"))
    if solver_samples is not None:
        cfg.solver.num_samples = int(solver_samples)
    if solver_steps is not None:
        cfg.solver.n_steps = int(solver_steps)
    if solver_topk is not None:
        cfg.solver.topk = int(solver_topk)
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    candidate_episode_ids = pool["online_episode_ids"][int(pool_offset) :]
    pair_limit = (
        len(candidate_episode_ids)
        if target_environment_steps is not None
        else int(episodes)
    )
    pairs = _online_start_pairs(
        dataset,
        candidate_episode_ids,
        goal_offset=int(cfg.eval.goal_offset_steps),
        seed=int(cfg.seed),
        limit=pair_limit,
    )
    if model_override is None:
        model, resolved_checkpoint = load_policy_or_model(str(checkpoint))
    else:
        model = model_override
        resolved_checkpoint = Path(resolved_checkpoint_override or checkpoint).resolve()
    model_for_type_check = getattr(model, "model", model)
    is_fast_lewam = isinstance(model_for_type_check, FastLeWAM)
    process = fit_eval_processors(dataset, cfg.dataset.keys_to_cache)
    image_transform = img_transform(cfg)

    def normalize_action(actions: torch.Tensor) -> torch.Tensor:
        normalized = process["action"].transform(actions.numpy())
        return torch.as_tensor(normalized).float()

    replays = []
    total_environment_steps = 0
    episode_results = []
    used_pairs = 0
    target_steps = (
        int(target_environment_steps)
        if target_environment_steps is not None
        else int(episodes) * int(cfg.eval.eval_budget)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(pairs), int(chunk_size)):
        chunk = pairs[offset : offset + int(chunk_size)]
        used_pairs += len(chunk)
        remaining_steps = max(1, int(target_steps) - int(total_environment_steps))
        episode_budget = (
            min(int(cfg.eval.eval_budget), remaining_steps)
            if target_environment_steps is not None
            else int(cfg.eval.eval_budget)
        )
        cfg.world.num_envs = len(chunk)
        policy_kwargs = {
            "solver_cfg": cfg.solver,
            "plan_config": cfg.plan_config,
            "process": process,
            "transform": {"pixels": image_transform, "goal": image_transform},
            "device": device,
        }
        if is_fast_lewam:
            policy_kwargs["mode"] = "stage_b"
            policy_kwargs.update(
                {
                    "actor_warm_start": bool(actor_warm_start),
                    "guidance_mode": str(guidance_mode),
                    "guidance_step_size": float(guidance_step_size),
                    "guidance_last_steps": int(guidance_last_steps),
                    "guidance_inner_steps": int(guidance_inner_steps),
                    "guidance_max_rms_offset": float(guidance_max_rms_offset),
                }
            )
        policy = make_world_policy(model, **policy_kwargs)
        recorder = ExecutedTransitionRecorder(
            policy,
            episode_ids=[item[0] for item in chunk],
            start_steps=[item[1] for item in chunk],
            history_size=(int(model_for_type_check.action_horizon) if is_fast_lewam else 3),
            action_block=int(cfg.plan_config.action_block),
            model_version=f"{resolved_checkpoint}:{sha256_file(resolved_checkpoint)}",
            normalize_action=normalize_action,
            observation_transform=image_transform,
            data_kind="continuous",
            collector_version=collector_version,
        )
        world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
        world_cfg["num_envs"] = len(chunk)
        world_cfg["max_episode_steps"] = 2 * int(episode_budget)
        world = swm.World(**world_cfg, image_shape=(224, 224))
        world.set_policy(recorder)
        try:
            metrics = world.evaluate(
                dataset=dataset,
                episodes_idx=[item[0] for item in chunk],
                start_steps=[item[1] for item in chunk],
                goal_offset=int(cfg.eval.goal_offset_steps),
                eval_budget=int(episode_budget),
                callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
                reset_mode="wait",
            )
            total_environment_steps += recorder.environment_steps
            episode_results.extend(bool(value) for value in metrics["episode_successes"])
            try:
                from source.diagnostics.stage_b_epoch_pair_ranking import (
                    compute_pair_physical_terminal_cost,
                )

                try:
                    start_costs = compute_pair_physical_terminal_cost(
                        task, recorder.initial_physical_info
                    )
                except (KeyError, TypeError, ValueError):
                    start_costs = None
                try:
                    terminal_costs = compute_pair_physical_terminal_cost(task, world.infos)
                except (KeyError, TypeError, ValueError):
                    terminal_costs = None
                outcome_payload = {
                    (episode_id, env_index): {
                        "success": bool(success),
                        "physical_start_cost": (
                            None
                            if start_costs is None
                            else float(start_costs[env_index])
                        ),
                        "physical_terminal_cost": (
                            None
                            if terminal_costs is None
                            else float(terminal_costs[env_index])
                        ),
                    }
                    for env_index, (episode_id, success) in enumerate(
                        zip(
                            (item[0] for item in chunk),
                            metrics["episode_successes"],
                        )
                    )
                }
                replays.append(recorder.replay(source="online", outcomes=outcome_payload))
            except RuntimeError:
                # A very short terminated episode may not contain one full
                # history window; keep its outcome but do not fabricate data.
                pass
        finally:
            world.close()
        if total_environment_steps >= target_steps:
            break

    if not replays and not allow_empty_replay:
        raise RuntimeError("collection produced no complete transition windows")
    if target_environment_steps is not None and total_environment_steps != target_steps:
        raise RuntimeError(
            f"online collection ended at {total_environment_steps} environment steps; "
            f"target was {target_steps}"
        )
    replay = concatenate_replays(replays) if replays else None
    if replay is not None:
        save_transition_replay(output, replay)
    report = {
        "schema_version": 1,
        "phase": "round3_phase2",
        "task": task,
        "arm": "online_adapt_collection",
        "checkpoint": str(resolved_checkpoint),
        "checkpoint_sha256": sha256_file(resolved_checkpoint),
        "episodes_requested": int(episodes),
        "target_environment_steps": int(target_steps),
        "environment_steps": int(total_environment_steps),
        "pool_offset": int(pool_offset),
        "next_pool_offset": int(pool_offset) + int(used_pairs),
        "episode_successes": episode_results,
        "replay_path": None if replay is None else str(output),
        "replay_sha256": None if replay is None else replay.content_sha256(),
        "replay_count": 0 if replay is None else replay.count,
        "status": "ok" if replay is not None else "ok_no_complete_transition_window",
        "complete_transition_window": replay is not None,
        "collector": {
            "actor_warm_start": bool(actor_warm_start),
            "guidance_mode": str(guidance_mode),
            "guidance_step_size": float(guidance_step_size),
            "guidance_last_steps": int(guidance_last_steps),
            "guidance_inner_steps": int(guidance_inner_steps),
            "guidance_max_rms_offset": float(guidance_max_rms_offset),
            "collector_version": collector_version,
        },
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


__all__ = ["_online_start_pairs", "collect_online", "derive_online_pool"]
