"""Round 3 legacy E5 online post-training validation.

This runner mirrors the Cube E0 Phase 2 protocol while respecting the native
Fast-LeWAM geometry: five action blocks are predicted together, so each replay
row contains six observations and five executed action blocks.  The visual
encoder/projector stay frozen; the Fast Stage-B latent dynamics path is
updated from a fixed-size source-balanced replay minibatch.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.round3_phase2 import (
    _canonical_hash,
    _load_offline_batch,
    _restore_phase2_adapter,
    _save_phase2_checkpoint,
    _state_digest,
    _write_json_atomic,
    collect_online,
)
from source.common.checkpoint import load_policy_or_model
from source.common.eval import (
    EvaluationIdentity,
    compose_eval_config,
    fit_eval_processors,
    get_dataset,
    img_transform,
)
from source.common.round3_eval import run_round3_evaluation, validate_gpu_visibility
from source.common.round3_phase1 import CohortManifest, sha256_file
from source.common.round3_sampling import build_revised_cohorts
from source.experiments.round3_phase2 import (
    FastStageBOnlineAdapter,
    ReplayShardPool,
    concatenate_replays,
)
from source.model.fast_lewam.jepa import FastLeWAM


def _evaluate_fast_dev(
    *,
    model: torch.nn.Module,
    checkpoint_identity: Path,
    output_dir: Path,
    device: str,
    task: str = "cube",
) -> dict[str, Any]:
    """Evaluate one Fast E5 model on the frozen task dev cohort."""
    cfg = compose_eval_config(
        task,
        (f"solver.device={device}", "output.save_video=false"),
    )
    manifest = CohortManifest.load(
        ROOT / f"outputs/round3/phase1/cohorts/{task}/dev_round3_revised.json"
    )
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    identity = EvaluationIdentity(
        entrypoint="round3_phase2_fast",
        policy_kind="e5_fast",
        checkpoint=str(checkpoint_identity),
        epoch=10,
        stage="stage_b",
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
    return {
        "output_dir": str(output_dir),
        "result_path": str(output_dir / "result.json"),
        "success_rate": float(result["success_rate"]),
        "cohort_id": manifest.cohort_id,
        "cohort_sha256": manifest.computed_sha256,
        "status": result.get("status", "ok"),
    }


def _fast_config_metadata(checkpoint: Path, resolved: Path) -> dict[str, Any]:
    model, _ = load_policy_or_model(str(checkpoint))
    if not isinstance(model, FastLeWAM):
        raise TypeError(f"legacy E5 checkpoint did not load as FastLeWAM: {resolved}")
    return {
        "model_type": type(model).__name__,
        "token_encoding": str(model.token_encoding),
        "stage_b_dynamics": str(model.stage_b_dynamics),
        "stage_b_attention_mode": str(model.stage_b_attention_mode),
        "latent_dim": int(model.latent_dim),
        "action_dim": int(model.action_dim),
        "action_horizon": int(model.action_horizon),
        "checkpoint": str(resolved),
        "checkpoint_sha256": sha256_file(resolved),
    }


def run_fast_phase2(
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
    """Run the one-seed legacy E5 task online validation."""
    from source.common.round3_eval import validate_gpu_visibility
    from source.experiments.round3_phase2 import phase2_update_steps

    validate_gpu_visibility(device)
    if not str(device).startswith("cuda"):
        raise ValueError("run_fast_phase2 requires an explicitly selected CUDA device")
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
    if not isinstance(base_model, FastLeWAM):
        raise TypeError(f"expected FastLeWAM legacy E5 checkpoint, got {type(base_model).__name__}")
    prior_state = json.loads(state_path.read_text(encoding="utf-8")) if resume else None
    prior_updates = (
        [record["update"] for record in prior_state.get("records", []) if "update" in record]
        if prior_state is not None
        else []
    )
    if resume and prior_state.get("status") == "ok":
        raise ValueError("Fast Phase 2 run is already complete; use a new output root")
    if resume and (
        int(prior_state.get("update_interval", update_interval)) != int(update_interval)
        or int(prior_state.get("max_environment_steps", max_environment_steps))
        != int(max_environment_steps)
    ):
        raise ValueError("resume schedule differs from the persisted Fast Phase 2 run")

    if prior_updates:
        last_update = prior_updates[-1]
        offline_model, _ = load_policy_or_model(last_update["offline_checkpoint"]["path"])
        online_model, _ = load_policy_or_model(last_update["online_checkpoint"]["path"])
    else:
        offline_model, _ = load_policy_or_model(str(checkpoint))
        online_model, _ = load_policy_or_model(str(checkpoint))
    for name, model in (("offline", offline_model), ("online", online_model)):
        if not isinstance(model, FastLeWAM):
            raise TypeError(f"{name} continuation checkpoint did not load as FastLeWAM")

    offline_adapter = FastStageBOnlineAdapter(
        offline_model,
        learning_rate=1e-5,
        weight_decay=1e-3,
        gradient_clip_norm=1.0,
        seed=3072,
        device=device,
    )
    online_adapter = FastStageBOnlineAdapter(
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
    records: list[dict[str, Any]] = [] if prior_state is None else list(prior_state.get("records", []))
    pool_offset = 0 if prior_state is None else int(prior_state.get("pool_offset", 0))
    current_steps = 0 if prior_state is None else int(prior_state.get("current_environment_steps", 0))
    offline_checkpoint = Path(resolved_base).resolve()
    online_checkpoint = Path(resolved_base).resolve()
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

    model_metadata = _fast_config_metadata(checkpoint, Path(resolved_base).resolve())

    def evaluate_boundary(step: int) -> None:
        nonlocal offline_checkpoint, online_checkpoint
        if skip_eval:
            records.append({"environment_steps": int(step), "evaluation": "skipped"})
            return
        boundary = {"environment_steps": int(step), "arms": {}}
        for arm, model, adapter, identity in (
            ("freeze", base_model, None, Path(resolved_base)),
            ("offline_continue", offline_model, offline_adapter, offline_checkpoint),
            ("online_adapt", online_model, online_adapter, online_checkpoint),
        ):
            result = _evaluate_fast_dev(
                model=model,
                checkpoint_identity=identity,
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
            allow_empty_replay=True,
            task=task,
        )
        if int(collect_report["replay_count"]) > 0:
            replay_pool.add(
                shard_path,
                count=int(collect_report["replay_count"]),
                content_sha256=str(collect_report["replay_sha256"]),
            )
        pool_offset = int(collect_report["next_pool_offset"])
        collected_steps = int(collect_report["environment_steps"])
        if collected_steps != int(update_interval):
            raise RuntimeError(f"Fast Phase 2 chunk produced {collected_steps} steps, expected {update_interval}")
        current_steps += collected_steps
        if current_steps != int(target_steps):
            raise RuntimeError(f"Fast Phase 2 step drift: accumulated {current_steps}, scheduled {target_steps}")

        offline_replay, offline_manifest = _load_offline_batch(
            dataset=dataset,
            process=process,
            image_transform=image_transform,
            episode_ids=offline_episode_ids,
            count=int(batch_size),
            seed=3072 + update_index,
            model_version=f"{Path(resolved_base).resolve()}:{sha256_file(resolved_base)}",
            action_block=int(cfg.plan_config.action_block),
            history_size=int(online_model.action_horizon),
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
        offline_checkpoint = checkpoint_root / f"{task}_e5_legacy_offline_continue_envsteps_{int(current_steps):06d}.ckpt"
        online_checkpoint = checkpoint_root / f"{task}_e5_legacy_online_adapt_envsteps_{int(current_steps):06d}.ckpt"
        offline_checkpoint_record = _save_phase2_checkpoint(offline_model, offline_adapter, offline_checkpoint)
        online_checkpoint_record = _save_phase2_checkpoint(online_model, online_adapter, online_checkpoint)
        update_record = {
            "environment_steps": int(current_steps),
            "update_index": int(update_index),
            "history_size": int(online_model.action_horizon),
            "action_block": int(cfg.plan_config.action_block),
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
                "phase": "round3_phase2_fast_e5_legacy",
                "task": task,
                "method": "e5_fast",
                "status": "running",
                **model_metadata,
                "device": str(device),
                "max_environment_steps": int(max_environment_steps),
                "update_interval": int(update_interval),
                "batch_size": int(batch_size),
                "replay_history_size": int(online_model.action_horizon),
                "current_environment_steps": int(current_steps),
                "pool_offset": int(pool_offset),
                "replay_shards": [asdict(shard) for shard in replay_pool.shards],
                "records": records,
            },
        )

    final = {
        "schema_version": 1,
        "phase": "round3_phase2_fast_e5_legacy",
        "task": task,
        "method": "e5_fast",
        "status": "ok",
        **model_metadata,
        "device": str(device),
        "max_environment_steps": int(max_environment_steps),
        "update_interval": int(update_interval),
        "batch_size": int(batch_size),
        "replay_history_size": int(online_model.action_horizon),
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
    runner = parser.add_subparsers(dest="command", required=True).add_parser("run-phase2-fast")
    runner.add_argument("--task", default="cube")
    runner.add_argument(
        "--checkpoint",
        default="outputs/fast_lewam/cube/0803_e5_cross_head_grad/checkpoints/fast_lewam_weights_epoch_10.pt",
    )
    runner.add_argument("--pool", default="outputs/round3/phase2/cube_online_pool.json")
    runner.add_argument(
        "--output-root",
        default="outputs/round3/phase2/cube_e5_legacy_three_arms_100step",
    )
    runner.add_argument("--device", default="cuda:0")
    runner.add_argument("--max-environment-steps", type=int, default=20000)
    runner.add_argument("--update-interval", type=int, default=100)
    runner.add_argument("--batch-size", type=int, default=64)
    runner.add_argument("--chunk-size", type=int, default=1)
    runner.add_argument("--skip-eval", action="store_true")
    runner.add_argument("--resume", action="store_true")
    runner.set_defaults(function=lambda args: run_fast_phase2(
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
    print(json.dumps({
        "output_root": str(ROOT / args.output_root),
        "status": payload["status"],
        "max_environment_steps": payload["max_environment_steps"],
        "evaluation_status": payload["evaluation_status"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
