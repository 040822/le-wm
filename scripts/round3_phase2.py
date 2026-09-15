"""Round 3 Phase 2 preparation and CPU checks.

The first command derives the episode-disjoint online pool from the frozen
Round 3 cohort rule. It performs no environment rollout and does not require a
CUDA device.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.eval import compose_eval_config, get_dataset
from source.common.round3_phase1 import _jsonable, sha256_file
from source.common.round3_sampling import build_revised_cohorts

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
                "sha256": sha256_file(ROOT / f"outputs/round3/phase1/cohorts/{task}/dev_round3_revised.json"),
            },
            "final": {
                "path": f"outputs/round3/phase1/cohorts/{task}/final.json",
                "sha256": sha256_file(ROOT / f"outputs/round3/phase1/cohorts/{task}/final.json"),
            },
        },
        "excluded_episode_ids": [
            _jsonable(entry.episode_id)
            for entry in (*dev.entries, *final.entries)
        ],
        "online_episode_ids": [_jsonable(value) for value in online_ids],
    }
    payload["content_sha256"] = _canonical_hash(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return payload


def _online_start_pairs(dataset: Any, episode_ids: list[Any], *, goal_offset: int, seed: int, limit: int):
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
    from source.common.eval import compose_eval_config, fit_eval_processors, get_dataset, img_transform
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
            history_size=(
                int(model_for_type_check.action_horizon)
                if is_fast_lewam
                else 3
            ),
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
                    terminal_costs = compute_pair_physical_terminal_cost(
                        task, world.infos
                    )
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
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _state_digest(model: torch.nn.Module, prefixes: tuple[str, ...] = ()) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if prefixes and not any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_phase2_checkpoint(
    model: torch.nn.Module,
    adapter: Any,
    path: Path,
) -> dict[str, Any]:
    """Save an inference-loadable model plus a resumable optimizer sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(model, temporary)
    temporary.replace(path)
    state_path = path.with_suffix(path.suffix + ".state.pt")
    state_tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "model_sha256": sha256_file(path),
            "optimizer": adapter.optimizer.state_dict(),
            "optimizer_steps": int(adapter.optimizer_steps),
            "sampler_seed": int(adapter.sampler_seed),
        },
        state_tmp,
    )
    state_tmp.replace(state_path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "state_path": str(state_path),
        "state_sha256": sha256_file(state_path),
    }


def _restore_phase2_adapter(adapter: Any, state_path: Path, model_path: Path) -> None:
    payload = torch.load(state_path, map_location=adapter.device, weights_only=False)
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported Phase 2 optimizer sidecar: {state_path}")
    if payload.get("model_sha256") != sha256_file(model_path):
        raise ValueError(f"Phase 2 optimizer sidecar does not match model: {state_path}")
    adapter.optimizer.load_state_dict(payload["optimizer"])
    adapter.optimizer_steps = int(payload["optimizer_steps"])
    adapter.sampler_seed = int(payload["sampler_seed"])
    adapter.prepare_for_update()


def _evaluate_phase2_dev(
    *,
    model: torch.nn.Module,
    checkpoint_identity: Path,
    output_dir: Path,
    device: str,
    task: str = "cube",
) -> dict[str, Any]:
    """Evaluate one in-memory E0 model on the frozen task dev cohort."""
    from source.common.eval import EvaluationIdentity
    from source.common.round3_eval import run_round3_evaluation
    from source.common.round3_phase1 import CohortManifest

    cfg = compose_eval_config(
        task,
        (f"solver.device={device}", "output.save_video=false"),
    )
    manifest = CohortManifest.load(
        ROOT / f"outputs/round3/phase1/cohorts/{task}/dev_round3_revised.json"
    )
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    identity = EvaluationIdentity(
        entrypoint="round3_phase2",
        policy_kind="lewm",
        checkpoint=str(checkpoint_identity),
        epoch=None,
        stage=None,
    )
    result = run_round3_evaluation(
        cfg,
        task=task,
        policy_or_model=model,
        identity=identity,
        manifest=manifest,
        dataset=dataset,
        output_dir=output_dir,
        trace_output_dir=output_dir / "trace",
        device=device,
        trace=True,
    )
    # Keep the result compact in the phase runner state file; the canonical
    # result and trace remain in output_dir.
    return {
        "output_dir": str(output_dir),
        "result_path": str(output_dir / "result.json"),
        "success_rate": float(result["success_rate"]),
        "cohort_sha256": manifest.computed_sha256,
        "status": result.get("status", "ok"),
    }


def _load_offline_batch(
    *,
    dataset: Any,
    process: Any,
    image_transform: Any,
    episode_ids: tuple[Any, ...],
    count: int,
    seed: int,
    model_version: str,
    action_block: int,
    history_size: int = 3,
):
    from source.experiments.round3_phase2 import (
        build_offline_window_manifest,
        load_offline_replay,
    )

    manifest = build_offline_window_manifest(
        dataset,
        episode_ids=episode_ids,
        count=int(count),
        seed=int(seed),
        frameskip=int(action_block),
        num_steps=int(history_size) + 1,
    )
    replay = load_offline_replay(
        dataset,
        manifest,
        process=process,
        observation_transform=image_transform,
        model_version=model_version,
        history_size=int(history_size),
        frameskip=int(action_block),
        num_steps=int(history_size) + 1,
    )
    return replay, manifest


def run_cpu_dry_run(
    *,
    checkpoint: Path,
    online_replay_path: Path,
    output_dir: Path,
    offline_samples: int,
    task: str = "cube",
) -> dict[str, Any]:
    """Exercise all three Phase 2 arms without claiming a GPU evaluation."""
    from source.common.eval import fit_eval_processors, img_transform
    from source.common.checkpoint import load_policy_or_model
    from source.experiments.round3_phase2 import (
        StageBOnlineAdapter,
        build_offline_window_manifest,
        concatenate_replays,
        load_offline_replay,
        load_transition_replay,
    )

    cfg = compose_eval_config(task, ("output.save_video=false",))
    np.random.seed(3072)
    torch.manual_seed(3072)
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    process = fit_eval_processors(dataset, cfg.dataset.keys_to_cache)
    image_transform = img_transform(cfg)
    model_version = f"{checkpoint.resolve()}:{sha256_file(checkpoint)}"
    online_pool_path = ROOT / f"outputs/round3/phase2/{task}_online_pool.json"
    online_pool = json.loads(online_pool_path.read_text(encoding="utf-8"))
    offline_episode_ids = tuple(online_pool["online_episode_ids"])
    manifest = build_offline_window_manifest(
        dataset,
        episode_ids=offline_episode_ids,
        count=int(offline_samples),
        seed=int(cfg.seed),
        frameskip=int(cfg.plan_config.action_block),
        num_steps=4,
    )
    offline_replay = load_offline_replay(
        dataset,
        manifest,
        process=process,
        observation_transform=image_transform,
        model_version=model_version,
        history_size=3,
        frameskip=int(cfg.plan_config.action_block),
        num_steps=4,
    )
    online_replay = load_transition_replay(online_replay_path)
    mixed_replay = concatenate_replays((offline_replay, online_replay))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "offline_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": task,
                "source": "original_e0_training_data",
                "episode_pool": "all_raw_episodes_excluding_dev_and_final",
                "episode_pool_manifest": str(online_pool_path),
                "episode_pool_sha256": online_pool["content_sha256"],
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "entries": list(manifest),
                "content_sha256": _canonical_hash(list(manifest)),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    arms: dict[str, Any] = {}
    frozen_model, _ = load_policy_or_model(str(checkpoint))
    frozen_digest = _state_digest(frozen_model, ("encoder", "projector"))
    arms["freeze"] = {
        "update": False,
        "optimizer_steps": 0,
        "encoder_projector_sha256": frozen_digest,
        "status": "ok",
    }

    offline_model, _ = load_policy_or_model(str(checkpoint))
    offline_before = _state_digest(offline_model, ("encoder", "projector"))
    offline_adapter = StageBOnlineAdapter(
        offline_model,
        learning_rate=1e-5,
        weight_decay=1e-3,
        gradient_clip_norm=1.0,
        seed=3072,
        device="cpu",
    )
    offline_report = offline_adapter.update(
        offline_replay,
        batch_size=min(64, offline_replay.count),
        updates=1,
        mixed=False,
        source="offline",
    )
    arms["offline_continue"] = {
        "update": True,
        "replay_sha256": offline_replay.content_sha256(),
        "update_report": asdict(offline_report),
        "encoder_projector_sha256_before": offline_before,
        "encoder_projector_sha256_after": _state_digest(
            offline_model, ("encoder", "projector")
        ),
        "status": "ok",
    }

    online_model, _ = load_policy_or_model(str(checkpoint))
    online_before = _state_digest(online_model, ("encoder", "projector"))
    online_adapter = StageBOnlineAdapter(
        online_model,
        learning_rate=1e-5,
        weight_decay=1e-3,
        gradient_clip_norm=1.0,
        seed=3072,
        device="cpu",
    )
    online_report = online_adapter.update(
        mixed_replay,
        batch_size=min(64, mixed_replay.count),
        updates=1,
        mixed=True,
    )
    arms["online_adapt"] = {
        "update": True,
        "offline_replay_sha256": offline_replay.content_sha256(),
        "online_replay_sha256": online_replay.content_sha256(),
        "mixed_replay_count": mixed_replay.count,
        "update_report": asdict(online_report),
        "encoder_projector_sha256_before": online_before,
        "encoder_projector_sha256_after": _state_digest(
            online_model, ("encoder", "projector")
        ),
        "status": "ok",
    }

    report = {
        "schema_version": 1,
        "phase": "round3_phase2",
        "task": task,
        "protocol": "round3_revised",
        "kind": "cpu_dry_run",
        "status": "ok",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "online_replay": {
            "path": str(online_replay_path),
            "sha256": online_replay.content_sha256(),
            "count": online_replay.count,
        },
        "offline_replay": {
            "sha256": offline_replay.content_sha256(),
            "count": offline_replay.count,
            "episode_pool": "all_raw_episodes_excluding_dev_and_final",
        },
        "arms": arms,
        "checkpoint_naming_pattern": "{task}_{arm}_envsteps_{environment_steps:06d}.ckpt",
        "evaluation_nodes": [0, 5000, 10000, 20000],
        "evaluation_status": "gpu_evaluation_pending_resource_window",
    }
    report_path = output_dir / "phase2_cpu_dry_run_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def run_phase2(
    *,
    checkpoint: Path,
    pool_path: Path,
    output_root: Path,
    device: str,
    max_environment_steps: int = 20000,
    update_interval: int = 100,
    batch_size: int = 64,
    chunk_size: int = 1,
    skip_eval: bool = False,
    resume: bool = False,
    task: str = "cube",
) -> dict[str, Any]:
    """Run the complete one-seed task Phase 2 protocol.

    This is intentionally a single process: the online model and optimizer
    stay resident between chunks, while each executed-transition shard is
    persisted before the next update.  No GPU is selected implicitly; callers
    must export an allow-listed ``CUDA_VISIBLE_DEVICES`` and pass the logical
    device explicitly (normally ``cuda:0`` after selecting one physical GPU).
    """
    from source.common.checkpoint import load_policy_or_model
    from source.common.eval import fit_eval_processors, img_transform
    from source.common.round3_eval import validate_gpu_visibility
    from source.experiments.round3_phase2 import (
        ReplayShardPool,
        StageBOnlineAdapter,
        concatenate_replays,
        phase2_update_steps,
    )

    validate_gpu_visibility(device)
    if not str(device).startswith("cuda"):
        raise ValueError("run_phase2 requires an explicitly selected CUDA device")
    if int(max_environment_steps) < 1 or int(update_interval) < 1:
        raise ValueError("environment steps and update interval must be positive")
    if int(max_environment_steps) % int(update_interval):
        raise ValueError("max_environment_steps must be divisible by update_interval")
    if int(batch_size) < 2 or int(batch_size) % 2:
        raise ValueError("batch_size must be an even positive number")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    state_path = output_root / "run_state.json"
    if not resume and output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty Phase 2 output: {output_root}")
    if resume and not state_path.is_file():
        raise FileNotFoundError(f"Phase 2 resume state is missing: {state_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    replay_root = output_root / "replay_shards"
    checkpoint_root = output_root / "checkpoints"
    eval_root = output_root / "evals"
    cfg = compose_eval_config(task, (f"solver.device={device}", "output.save_video=false"))
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    process = fit_eval_processors(dataset, cfg.dataset.keys_to_cache)
    image_transform = img_transform(cfg)
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    offline_episode_ids = tuple(pool["online_episode_ids"])

    base_model, resolved_base = load_policy_or_model(str(checkpoint))
    prior_state = json.loads(state_path.read_text(encoding="utf-8")) if resume else None
    prior_updates = (
        [record["update"] for record in prior_state.get("records", []) if "update" in record]
        if prior_state is not None
        else []
    )
    if resume and prior_state.get("status") == "ok":
        raise ValueError("Phase 2 run is already complete; use a new output root")
    if resume and (
        int(prior_state.get("update_interval", update_interval)) != int(update_interval)
        or int(prior_state.get("max_environment_steps", max_environment_steps))
        != int(max_environment_steps)
    ):
        raise ValueError("resume schedule differs from the persisted Phase 2 run")
    if prior_updates:
        last_update = prior_updates[-1]
        offline_model, _ = load_policy_or_model(last_update["offline_checkpoint"]["path"])
        online_model, _ = load_policy_or_model(last_update["online_checkpoint"]["path"])
    else:
        offline_model, _ = load_policy_or_model(str(checkpoint))
        online_model, _ = load_policy_or_model(str(checkpoint))
    offline_adapter = StageBOnlineAdapter(
        offline_model,
        learning_rate=1e-5,
        weight_decay=1e-3,
        gradient_clip_norm=1.0,
        seed=3072,
        device=device,
    )
    online_adapter = StageBOnlineAdapter(
        online_model,
        learning_rate=1e-5,
        weight_decay=1e-3,
        gradient_clip_norm=1.0,
        seed=3072,
        device=device,
    )
    replay_pool = ReplayShardPool(seed=3072)
    eval_steps = {0, 5000, 10000, 20000}
    if int(max_environment_steps) != 20000:
        eval_steps = {step for step in eval_steps if step <= int(max_environment_steps)}
        eval_steps.add(int(max_environment_steps))
    records: list[dict[str, Any]] = (
        [] if prior_state is None else list(prior_state.get("records", []))
    )
    pool_offset = 0 if prior_state is None else int(prior_state.get("pool_offset", 0))
    current_steps = (
        0 if prior_state is None else int(prior_state.get("current_environment_steps", 0))
    )
    online_checkpoint = Path(resolved_base).resolve()
    offline_checkpoint = Path(resolved_base).resolve()
    if prior_updates:
        last_update = prior_updates[-1]
        offline_checkpoint = Path(last_update["offline_checkpoint"]["path"])
        online_checkpoint = Path(last_update["online_checkpoint"]["path"])
        _restore_phase2_adapter(
            offline_adapter,
            Path(last_update["offline_checkpoint"]["state_path"]),
            offline_checkpoint,
        )
        _restore_phase2_adapter(
            online_adapter,
            Path(last_update["online_checkpoint"]["state_path"]),
            online_checkpoint,
        )
        for shard in prior_state.get("replay_shards", []):
            replay_pool.add(
                shard["path"],
                count=int(shard["count"]),
                content_sha256=str(shard["content_sha256"]),
            )

    def evaluate_boundary(step: int) -> None:
        nonlocal online_checkpoint, offline_checkpoint
        if skip_eval:
            records.append({"environment_steps": int(step), "evaluation": "skipped"})
            return
        boundary = {"environment_steps": int(step), "arms": {}}
        for arm, model, adapter, checkpoint_identity in (
            ("freeze", base_model, None, Path(resolved_base)),
            ("offline_continue", offline_model, offline_adapter, offline_checkpoint),
            ("online_adapt", online_model, online_adapter, online_checkpoint),
        ):
            result = _evaluate_phase2_dev(
                model=model,
                checkpoint_identity=checkpoint_identity,
                output_dir=eval_root / arm / f"envsteps_{int(step):06d}",
                device=device,
                task=task,
            )
            boundary["arms"][arm] = result
            if adapter is not None:
                adapter.prepare_for_update()
        records.append(boundary)

    if prior_state is None:
        evaluate_boundary(0)
    for update_index, target_steps in enumerate(
        (
            step
            for step in phase2_update_steps(int(max_environment_steps), int(update_interval))
            if step > current_steps
        ),
        start=int(current_steps) // int(update_interval) + 1,
    ):
        shard_path = replay_root / f"online_envsteps_{int(target_steps):06d}.pt"
        collect_report = collect_online(
            pool_path=pool_path,
            checkpoint=online_checkpoint,
            output=shard_path,
            device=device,
            episodes=max(1, int(update_interval) // int(cfg.eval.eval_budget)),
            chunk_size=int(chunk_size),
            target_environment_steps=int(update_interval),
            pool_offset=int(pool_offset),
            model_override=online_model,
            resolved_checkpoint_override=online_checkpoint,
            task=task,
        )
        replay_pool.add(
            shard_path,
            count=int(collect_report["replay_count"]),
            content_sha256=str(collect_report["replay_sha256"]),
        )
        pool_offset = int(collect_report["next_pool_offset"])
        collected_steps = int(collect_report["environment_steps"])
        if collected_steps != int(update_interval):
            raise RuntimeError(
                "Phase 2 chunk did not produce the exact frozen update interval: "
                f"{collected_steps} != {int(update_interval)}"
            )
        current_steps += collected_steps
        if current_steps != int(target_steps):
            raise RuntimeError(
                f"Phase 2 step provenance drifted: accumulated {current_steps}, "
                f"scheduled {int(target_steps)}"
            )

        offline_replay, offline_manifest = _load_offline_batch(
            dataset=dataset,
            process=process,
            image_transform=image_transform,
            episode_ids=offline_episode_ids,
            count=int(batch_size),
            seed=3072 + update_index,
            model_version=f"{Path(resolved_base).resolve()}:{sha256_file(resolved_base)}",
            action_block=int(cfg.plan_config.action_block),
        )
        offline_update = offline_adapter.update(
            offline_replay,
            batch_size=int(batch_size),
            updates=1,
            mixed=False,
            source="offline",
        )
        if replay_pool.count > 0:
            online_replay = replay_pool.sample(int(batch_size) // 2)
            mixed_replay = concatenate_replays((offline_replay, online_replay))
            online_update = online_adapter.update(
                mixed_replay,
                batch_size=int(batch_size),
                updates=1,
                mixed=True,
            )
            online_update_skipped_reason = None
        else:
            online_replay = None
            online_update = None
            online_update_skipped_reason = "no_complete_online_transition_window"

        parent_offline_checkpoint = offline_checkpoint
        parent_online_checkpoint = online_checkpoint
        offline_checkpoint = checkpoint_root / (
            f"{task}_offline_continue_envsteps_{int(current_steps):06d}.ckpt"
        )
        online_checkpoint = checkpoint_root / (
            f"{task}_online_adapt_envsteps_{int(current_steps):06d}.ckpt"
        )
        offline_checkpoint_record = _save_phase2_checkpoint(
            offline_model, offline_adapter, offline_checkpoint
        )
        online_checkpoint_record = _save_phase2_checkpoint(
            online_model, online_adapter, online_checkpoint
        )
        update_record = {
            "environment_steps": int(current_steps),
            "update_index": int(update_index),
            "collect": collect_report,
            "offline_manifest_sha256": _canonical_hash(list(offline_manifest)),
            "offline_replay_sha256": offline_replay.content_sha256(),
            "online_sample_sha256": (
                None if online_replay is None else online_replay.content_sha256()
            ),
            "online_update_skipped_reason": online_update_skipped_reason,
            "replay_pool_count": int(replay_pool.count),
            "parent_offline_checkpoint": {
                "path": str(parent_offline_checkpoint),
                "sha256": sha256_file(parent_offline_checkpoint),
            },
            "parent_online_checkpoint": {
                "path": str(parent_online_checkpoint),
                "sha256": sha256_file(parent_online_checkpoint),
            },
            "offline_update": asdict(offline_update),
            "online_update": None if online_update is None else asdict(online_update),
            "offline_checkpoint": offline_checkpoint_record,
            "online_checkpoint": online_checkpoint_record,
        }
        records.append({"update": update_record})
        if int(current_steps) in eval_steps:
            evaluate_boundary(int(current_steps))
        _write_json_atomic(
            output_root / "run_state.json",
            {
                "schema_version": 1,
                "phase": "round3_phase2",
                "task": task,
                "status": "running",
                "checkpoint": str(Path(resolved_base).resolve()),
                "checkpoint_sha256": sha256_file(resolved_base),
                "device": str(device),
                "max_environment_steps": int(max_environment_steps),
                "update_interval": int(update_interval),
                "current_environment_steps": int(current_steps),
                "pool_offset": int(pool_offset),
                "replay_shards": [asdict(shard) for shard in replay_pool.shards],
                "records": records,
            },
        )

    final = {
        "schema_version": 1,
        "phase": "round3_phase2",
        "task": task,
        "status": "ok",
        "checkpoint": str(Path(resolved_base).resolve()),
        "checkpoint_sha256": sha256_file(resolved_base),
        "device": str(device),
        "max_environment_steps": int(max_environment_steps),
        "update_interval": int(update_interval),
        "current_environment_steps": int(current_steps),
        "pool_offset": int(pool_offset),
        "replay_shards": [asdict(shard) for shard in replay_pool.shards],
        "records": records,
        "evaluation_status": "skipped" if skip_eval else "ok",
    }
    _write_json_atomic(output_root / "run_state.json", final)
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    derive = subparsers.add_parser("derive-online-pool")
    derive.add_argument("--task", default="cube")
    derive.add_argument(
        "--output",
        default="outputs/round3/phase2/cube_online_pool.json",
    )
    derive.add_argument("--online-count", type=int, default=None)
    derive.set_defaults(function=lambda args: derive_online_pool(
        output=ROOT / args.output,
        online_count=args.online_count,
        task=args.task,
    ))
    collect = subparsers.add_parser("collect-online")
    collect.add_argument("--task", default="cube")
    collect.add_argument("--pool", default="outputs/round3/phase2/cube_online_pool.json")
    collect.add_argument(
        "--checkpoint",
        default="outputs/lewm/cube/0803_e0_lewm_baseline_bs128/checkpoints/lewm_weights_epoch_10.pt",
    )
    collect.add_argument("--output", default="outputs/round3/phase2/cube_online_replay.pt")
    collect.add_argument("--device", default="cuda")
    collect.add_argument("--episodes", type=int, default=20)
    collect.add_argument("--chunk-size", type=int, default=1)
    collect.add_argument("--target-environment-steps", type=int, default=None)
    collect.add_argument("--pool-offset", type=int, default=0)
    collect.add_argument("--solver-samples", type=int, default=None)
    collect.add_argument("--solver-steps", type=int, default=None)
    collect.add_argument("--solver-topk", type=int, default=None)
    collect.add_argument("--allow-empty-replay", action="store_true")
    collect.set_defaults(function=lambda args: collect_online(
        pool_path=ROOT / args.pool,
        checkpoint=ROOT / args.checkpoint,
        output=ROOT / args.output,
        device=args.device,
        episodes=args.episodes,
        chunk_size=args.chunk_size,
        target_environment_steps=args.target_environment_steps,
        pool_offset=args.pool_offset,
        solver_samples=args.solver_samples,
        solver_steps=args.solver_steps,
        solver_topk=args.solver_topk,
        allow_empty_replay=args.allow_empty_replay,
        task=args.task,
    ))
    dry_run = subparsers.add_parser("cpu-dry-run")
    dry_run.add_argument("--task", default="cube")
    dry_run.add_argument(
        "--checkpoint",
        default="outputs/lewm/cube/0803_e0_lewm_baseline_bs128/checkpoints/lewm_weights_epoch_10.pt",
    )
    dry_run.add_argument(
        "--online-replay",
        default="outputs/round3/phase2/smoke_cpu/cube_online_replay.pt",
    )
    dry_run.add_argument(
        "--output-dir",
        default="outputs/round3/phase2/cpu_dry_run/cube",
    )
    dry_run.add_argument("--offline-samples", type=int, default=4)
    dry_run.set_defaults(function=lambda args: run_cpu_dry_run(
        checkpoint=ROOT / args.checkpoint,
        online_replay_path=ROOT / args.online_replay,
        output_dir=ROOT / args.output_dir,
        offline_samples=args.offline_samples,
        task=args.task,
    ))
    runner = subparsers.add_parser("run-phase2")
    runner.add_argument("--task", default="cube")
    runner.add_argument(
        "--checkpoint",
        default="outputs/lewm/cube/0803_e0_lewm_baseline_bs128/checkpoints/lewm_weights_epoch_10.pt",
    )
    runner.add_argument(
        "--pool",
        default="outputs/round3/phase2/cube_online_pool.json",
    )
    runner.add_argument(
        "--output-root",
        default="outputs/round3/phase2/cube_e0_three_arms",
    )
    runner.add_argument("--device", default="cuda:0")
    runner.add_argument("--max-environment-steps", type=int, default=20000)
    runner.add_argument("--update-interval", type=int, default=100)
    runner.add_argument("--batch-size", type=int, default=64)
    runner.add_argument("--chunk-size", type=int, default=1)
    runner.add_argument("--skip-eval", action="store_true")
    runner.add_argument("--resume", action="store_true")
    runner.set_defaults(function=lambda args: run_phase2(
        checkpoint=ROOT / args.checkpoint,
        pool_path=ROOT / args.pool,
        output_root=ROOT / args.output_root,
        device=args.device,
        max_environment_steps=args.max_environment_steps,
        update_interval=args.update_interval,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        skip_eval=args.skip_eval,
        resume=args.resume,
        task=args.task,
    ))
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    payload = args.function(args)
    if args.command == "derive-online-pool":
        summary = {
            "output": str(ROOT / args.output),
            "online_episode_count": len(payload["online_episode_ids"]),
            "content_sha256": payload["content_sha256"],
        }
    elif args.command == "run-phase2":
        summary = {
            "output_root": str(ROOT / args.output_root),
            "status": payload["status"],
            "max_environment_steps": payload["max_environment_steps"],
            "evaluation_status": payload["evaluation_status"],
        }
    else:
        summary = payload
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
