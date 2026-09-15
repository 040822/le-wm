"""Prepare and run the Round 3 online-supervision experiments.

The command deliberately separates replay preparation, fixed-replay training,
and final-cohort evaluation.  This makes it possible to inspect the immutable
20k-environment-step dataset before spending GPU time on T1--T6.

GPU commands must be launched with an explicit allow-listed visibility, for
example ``CUDA_VISIBLE_DEVICES=0 python scripts/round3_online_supervision.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUPPORTED_TASKS = ("reacher", "pusht")
DEFAULT_CHECKPOINTS = {
    "reacher": ROOT
    / "outputs/fast_lewam/reacher/0717_/checkpoints/fast_lewam_weights_epoch_10.pt",
    "pusht": ROOT
    / "outputs/fast_lewam/pusht/0803_e5_cross_head_grad/checkpoints/fast_lewam_weights_epoch_10.pt",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _path(value: str | Path) -> Path:
    raw = Path(value)
    return raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()


def _report_path(replay_path: Path) -> Path:
    report_path = replay_path.with_suffix(replay_path.suffix + ".json")
    if report_path.is_file():
        return report_path
    # collect_online historically published ``continuous.json`` while the
    # other replay builders use ``<replay>.pt.json``. Keep resume compatible
    # with that already completed artifact and use the canonical path for new
    # reports.
    legacy_path = replay_path.with_suffix(".json")
    return legacy_path if legacy_path.is_file() else report_path


def _read_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        raise ValueError(f"collection report is not complete: {path}")
    return payload


def _validate_grounded_budget(report: dict[str, Any], requested_steps: int) -> int:
    """Validate grounded group coverage while preserving early terminations."""
    expected_groups = int(requested_steps) // 100
    groups = report.get("groups", ())
    if len(groups) != expected_groups:
        raise RuntimeError(
            f"grounded collection produced {len(groups)} groups; "
            f"expected {expected_groups}"
        )
    actual_steps = int(report.get("environment_steps", -1))
    if actual_steps < 0 or actual_steps > int(requested_steps):
        raise RuntimeError(
            f"grounded collection used {actual_steps} environment steps; "
            f"nominal budget was {requested_steps}"
        )
    return actual_steps


def build_offline_replay(
    *,
    task: str,
    checkpoint: Path,
    pool_path: Path,
    output: Path,
    count: int,
    seed: int,
) -> dict[str, Any]:
    """Materialize a bounded offline anchor pool with the true goal frame."""
    from source.common.eval import compose_eval_config, fit_eval_processors, get_dataset, img_transform
    from source.experiments.round3_phase2 import (
        build_offline_window_manifest,
        load_offline_replay,
        save_transition_replay,
    )

    cfg = compose_eval_config(str(task), ("output.save_video=false",))
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    process = fit_eval_processors(dataset, cfg.dataset.keys_to_cache)
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    episode_ids = tuple(pool["online_episode_ids"])
    manifest = build_offline_window_manifest(
        dataset,
        episode_ids=episode_ids,
        count=int(count),
        seed=int(seed),
        frameskip=int(cfg.plan_config.action_block),
        num_steps=int(cfg.plan_config.horizon) + 1,
    )
    replay = load_offline_replay(
        dataset,
        manifest,
        process=process,
        observation_transform=img_transform(cfg),
        model_version=f"{checkpoint}:{_file_sha256(checkpoint)}",
        history_size=int(cfg.plan_config.horizon),
        frameskip=int(cfg.plan_config.action_block),
        num_steps=int(cfg.plan_config.horizon) + 1,
        goal_offset_steps=int(cfg.eval.goal_offset_steps),
    )
    save_transition_replay(output, replay)
    report = {
        "schema_version": 1,
        "phase": "round3_offline_anchor_pool",
        "task": str(task),
        "status": "ok",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "pool": str(pool_path),
        "pool_sha256": _file_sha256(pool_path),
        "count": replay.count,
        "replay_path": str(output),
        "replay_sha256": replay.content_sha256(),
        "window_manifest": list(manifest),
        "window_manifest_sha256": hashlib.sha256(
            json.dumps(list(manifest), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "goal_offset_steps": int(cfg.eval.goal_offset_steps),
    }
    _write_json_atomic(_report_path(output), report)
    return report


def prepare_fixed_replay(
    *,
    task: str,
    checkpoint: Path,
    pool_path: Path,
    output_root: Path,
    device: str,
    seed: int = 3072,
    offline_windows: int = 512,
    continuous_steps: int = 16_000,
    grounded_steps: int = 4_000,
    guidance_mode: str = "none",
    guidance_step_size: float = 0.01,
    guidance_last_steps: int = 5,
    guidance_inner_steps: int = 5,
    guidance_max_rms_offset: float = 0.20,
    actor_warm_start: bool = False,
    guidance_report: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Build the one immutable offline+continuous+grounded training replay."""
    if str(task) not in SUPPORTED_TASKS:
        raise ValueError(f"Round3 online supervision only supports {SUPPORTED_TASKS}")
    if int(continuous_steps) != 16_000 or int(grounded_steps) != 4_000:
        raise ValueError("fixed replay budget is exactly 16,000 continuous + 4,000 grounded steps")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output_root.exists() and not resume and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty fixed replay root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    effective_actor_warm_start = bool(actor_warm_start or str(guidance_mode) != "none")
    source_pool_path = pool_path
    if not pool_path.is_file():
        from source.experiments.round3_online_collection import derive_online_pool

        derive_online_pool(output=pool_path, task=str(task))

    excluded_episode_ids: list[Any] = []
    if guidance_report is None:
        candidate_report = (
            output_root.parent / "e1" / "e1_report.json"
        )
        guidance_report = candidate_report if candidate_report.is_file() else None
    if guidance_report is not None:
        guidance_payload = json.loads(guidance_report.read_text(encoding="utf-8"))
        excluded_episode_ids = list(guidance_payload.get("excluded_episode_ids", ()))
        if len(excluded_episode_ids) != 64:
            raise ValueError(
                "guidance report must declare exactly 64 E1 exclusion episode IDs"
            )
        pool_payload = json.loads(pool_path.read_text(encoding="utf-8"))
        excluded_keys = {repr(value) for value in excluded_episode_ids}
        pool_payload["online_episode_ids"] = [
            value
            for value in pool_payload["online_episode_ids"]
            if repr(value) not in excluded_keys
        ]
        pool_payload["excluded_episode_ids"] = excluded_episode_ids
        pool_payload["source_pool_sha256"] = _file_sha256(pool_path)
        pool_payload["content_sha256"] = hashlib.sha256(
            json.dumps(
                _jsonable(pool_payload),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        collection_pool_path = output_root / "replay" / "online_pool_excluding_e1.json"
        _write_json_atomic(collection_pool_path, pool_payload)
    else:
        collection_pool_path = pool_path

    replay_root = output_root / "replay"
    offline_path = replay_root / "offline.pt"
    continuous_path = replay_root / "continuous.pt"
    grounded_path = replay_root / "grounded.pt"
    if resume and offline_path.is_file():
        offline_report = _read_report(_report_path(offline_path))
    else:
        offline_report = build_offline_replay(
            task=str(task),
            checkpoint=checkpoint,
            pool_path=collection_pool_path,
            output=offline_path,
            count=int(offline_windows),
            seed=int(seed),
        )

    if resume and continuous_path.is_file():
        continuous_report = _read_report(_report_path(continuous_path))
    else:
        from source.experiments.round3_online_collection import collect_online

        continuous_report = collect_online(
            pool_path=collection_pool_path,
            checkpoint=checkpoint,
            output=continuous_path,
            device=str(device),
            episodes=1,
            chunk_size=1,
            target_environment_steps=int(continuous_steps),
            pool_offset=0,
            solver_samples=64,
            solver_steps=30,
            solver_topk=30,
            actor_warm_start=effective_actor_warm_start,
            guidance_mode=str(guidance_mode),
            guidance_step_size=float(guidance_step_size),
            guidance_last_steps=int(guidance_last_steps),
            guidance_inner_steps=int(guidance_inner_steps),
            guidance_max_rms_offset=float(guidance_max_rms_offset),
            collector_version="round3_continuous_v1",
            task=str(task),
        )
    if int(continuous_report.get("environment_steps", -1)) != int(continuous_steps):
        raise RuntimeError("continuous collection did not meet the exact fixed budget")
    if not continuous_report.get("replay_path"):
        raise RuntimeError("continuous collection returned no complete replay")

    grounded_offset = int(continuous_report.get("next_pool_offset", 0))
    if resume and grounded_path.is_file():
        grounded_report = _read_report(_report_path(grounded_path))
    else:
        from source.experiments.round3_grounded_collection import collect_grounded_replay

        grounded_report = collect_grounded_replay(
            task=str(task),
            checkpoint=checkpoint,
            pool_path=collection_pool_path,
            output=grounded_path,
            device=str(device),
            groups=int(grounded_steps // 100),
            pool_offset=grounded_offset,
            seed=int(seed),
            guidance_mode=str(guidance_mode),
            guidance_step_size=float(guidance_step_size),
            guidance_last_steps=int(guidance_last_steps),
            guidance_inner_steps=int(guidance_inner_steps),
            guidance_max_rms_offset=float(guidance_max_rms_offset),
            collector_version="round3_grounded_v1",
            resume=bool(resume),
            # Some sampled starts terminate before one complete history
            # window. They still consume the fixed 100 environment steps for
            # the group, but should not abort the remaining grounded panel.
            allow_empty_replay=True,
        )
    grounded_actual_steps = _validate_grounded_budget(
        grounded_report,
        int(grounded_steps),
    )

    from source.experiments.round3_phase2 import concatenate_replays, load_transition_replay, save_transition_replay

    offline = load_transition_replay(offline_path)
    continuous = load_transition_replay(continuous_path)
    grounded = load_transition_replay(grounded_path)
    if continuous.successes is None or continuous.physical_start_costs is None or continuous.physical_terminal_costs is None:
        raise RuntimeError("continuous replay is missing physical outcome metadata")
    if grounded.successes is None or grounded.physical_start_costs is None or grounded.physical_terminal_costs is None:
        raise RuntimeError("grounded replay is missing physical outcome metadata")
    replay = concatenate_replays((offline, continuous, grounded))
    fixed_path = output_root / "fixed_replay.pt"
    save_transition_replay(fixed_path, replay)
    manifest = {
        "schema_version": 1,
        "phase": "round3_fixed_replay_supervision",
        "task": str(task),
        "status": "ok",
        "seed": int(seed),
        "checkpoint": {"path": str(checkpoint), "sha256": _file_sha256(checkpoint)},
        "pool": {"path": str(collection_pool_path), "sha256": _file_sha256(collection_pool_path)},
        "source_pool": {
            "path": str(source_pool_path),
            "sha256": _file_sha256(source_pool_path),
            "excluded_episode_count": len(excluded_episode_ids),
            "guidance_report": None if guidance_report is None else str(guidance_report),
        },
        "guidance": {
            "actor_warm_start": effective_actor_warm_start,
            "mode": str(guidance_mode),
            "step_size": float(guidance_step_size),
            "last_steps": int(guidance_last_steps),
            "inner_steps": int(guidance_inner_steps),
            "max_rms_offset": float(guidance_max_rms_offset),
        },
        "budget": {
            "continuous_environment_steps": int(continuous_steps),
            "grounded_environment_steps": int(grounded_steps),
            "grounded_environment_steps_actual": int(grounded_actual_steps),
            "total_environment_steps": int(continuous_steps + grounded_steps),
            "total_environment_steps_actual": int(
                continuous_report["environment_steps"] + grounded_actual_steps
            ),
            "grounded_groups": int(grounded_steps // 100),
            "candidates_per_group": 4,
            "action_blocks_per_candidate": 5,
            "action_block_environment_steps": 5,
        },
        "replays": {
            "offline": {"path": str(offline_path), "sha256": offline.content_sha256(), "count": offline.count},
            "continuous": {"path": str(continuous_path), "sha256": continuous.content_sha256(), "count": continuous.count},
            "grounded": {"path": str(grounded_path), "sha256": grounded.content_sha256(), "count": grounded.count},
            "fixed": {"path": str(fixed_path), "sha256": replay.content_sha256(), "count": replay.count},
        },
        "collection_reports": {
            "continuous": continuous_report,
            "grounded": grounded_report,
        },
    }
    _write_json_atomic(output_root / "manifest.json", manifest)
    return manifest


def train_fixed_replay(
    *,
    task: str,
    checkpoint: Path,
    replay_path: Path,
    output_root: Path,
    device: str,
    seed: int = 3072,
    max_updates: int = 200,
    curve_interval: int = 10,
    arms: tuple[str, ...] = (),
    resume: bool = False,
    evaluate: bool = True,
) -> dict[str, Any]:
    """Run T0--T6 from the same checkpoint and fixed replay."""
    from source.experiments.round3_online_runner import TRAINING_ARMS, run_fixed_replay_training

    selected = TRAINING_ARMS if not arms else tuple(arms)
    evaluator = None
    if evaluate:
        from source.experiments.round3_final_eval import evaluate_final_checkpoint

        def evaluator(task_name: str, arm: str, checkpoint_path: Path, update_step: int):
            result = evaluate_final_checkpoint(
                checkpoint=checkpoint_path,
                task=task_name,
                output_dir=output_root / arm / "final_eval" / f"update_{int(update_step):06d}",
                device=str(device),
                resume=True,
            )
            return {
                "success_rate": float(result["success_rate"]),
                "episodes": len(result["episodes"]),
                "cohort_sha256": str(result["cohort_sha256"]),
            }

    return run_fixed_replay_training(
        checkpoint=checkpoint,
        replay_path=replay_path,
        output_root=output_root,
        task=str(task),
        device=str(device),
        seed=int(seed),
        max_updates=int(max_updates),
        curve_interval=int(curve_interval),
        arms=tuple(selected),
        evaluator=evaluator,
        resume=bool(resume),
    )


def train_closed_loop(
    *,
    task: str,
    checkpoint: Path,
    offline_replay_path: Path,
    pool_path: Path,
    output_root: Path,
    device: str,
    seed: int = 3072,
    max_updates: int = 200,
    curve_interval: int = 10,
    arms: tuple[str, ...] = (),
    guidance_mode: str = "none",
    guidance_step_size: float = 0.01,
    guidance_last_steps: int = 5,
    guidance_inner_steps: int = 5,
    guidance_max_rms_offset: float = 0.20,
    actor_warm_start: bool = False,
    fixed_training_root: Path | None = None,
    resume: bool = False,
    evaluate: bool = True,
) -> dict[str, Any]:
    """Run the 100-env-step/update 80/20 closed-loop protocol."""
    from source.common.checkpoint import load_policy_or_model
    from source.experiments.round3_closed_loop import run_closed_loop_training
    from source.experiments.round3_online_runner import TRAINING_ARMS, _load_default_model
    from source.experiments.round3_phase2 import load_transition_replay

    selected = TRAINING_ARMS if not arms else tuple(arms)
    loss_weights = {}
    requires_frozen_calibration = any(
        arm in {
            "t3_online_ranking",
            "t4_online_offline_a",
            "t5_online_hindsight_a",
            "t6_online_distill_a",
        }
        for arm in selected
    )
    if requires_frozen_calibration and fixed_training_root is None:
        raise ValueError(
            "closed-loop T3--T6 must receive --fixed-training-root so Stage-A/B "
            "calibration weights are reused unchanged"
        )
    if fixed_training_root is not None:
        for arm in selected:
            state_path = fixed_training_root / arm / "run_state.json"
            if not state_path.is_file():
                raise FileNotFoundError(
                    f"fixed-replay state is required to reuse calibration: {state_path}"
                )
            state = json.loads(state_path.read_text(encoding="utf-8"))
            values = state.get("loss_weights")
            if values is not None and arm != "t0_frozen":
                from source.experiments.round3_online_supervision import LossWeights

                loss_weights[arm] = LossWeights(**values)
            elif arm in {
                "t3_online_ranking",
                "t4_online_offline_a",
                "t5_online_hindsight_a",
                "t6_online_distill_a",
            }:
                raise ValueError(
                    f"fixed-replay state has no frozen loss weights for {arm}: {state_path}"
                )
        shared_offline_a = [
            float(loss_weights[arm].offline_a)
            for arm in (
                "t4_online_offline_a",
                "t5_online_hindsight_a",
                "t6_online_distill_a",
            )
            if arm in loss_weights
        ]
        if shared_offline_a and any(
            abs(value - shared_offline_a[0]) > 1e-12 for value in shared_offline_a[1:]
        ):
            raise ValueError("closed-loop T4/T5/T6 offline-A calibration weights differ")

    def collector(arm, step, kind, model, shard_path):
        # Each continuous event consumes two 50-step source episodes; each
        # grounded event consumes one four-candidate group episode.  Deriving
        # the offset from the update step also makes resume deterministic.
        source_offset = sum(
            2 if (index % 5) else 1
            for index in range(1, int(step))
        )
        if kind == "continuous":
            from source.experiments.round3_online_collection import collect_online

            report = collect_online(
                pool_path=pool_path,
                checkpoint=checkpoint,
                output=shard_path,
                device=str(device),
                episodes=1,
                chunk_size=1,
                target_environment_steps=100,
                pool_offset=source_offset,
                solver_samples=64,
                solver_steps=30,
                solver_topk=30,
                model_override=model,
                resolved_checkpoint_override=checkpoint,
                actor_warm_start=bool(actor_warm_start or str(guidance_mode) != "none"),
                guidance_mode=str(guidance_mode),
                guidance_step_size=float(guidance_step_size),
                guidance_last_steps=int(guidance_last_steps),
                guidance_inner_steps=int(guidance_inner_steps),
                guidance_max_rms_offset=float(guidance_max_rms_offset),
                collector_version=f"round3_closed_loop_continuous_v1_step_{int(step):06d}",
                task=str(task),
            )
        else:
            from source.experiments.round3_grounded_collection import collect_grounded_replay

            report = collect_grounded_replay(
                task=str(task),
                checkpoint=checkpoint,
                pool_path=pool_path,
                output=shard_path,
                device=str(device),
                groups=1,
                pool_offset=source_offset,
                seed=int(seed) + int(step),
                guidance_mode=str(guidance_mode),
                guidance_step_size=float(guidance_step_size),
                guidance_last_steps=int(guidance_last_steps),
                guidance_inner_steps=int(guidance_inner_steps),
                guidance_max_rms_offset=float(guidance_max_rms_offset),
                collector_version=f"round3_closed_loop_grounded_v1_step_{int(step):06d}",
                model_override=model,
                resolved_checkpoint_override=checkpoint,
                resume=bool(resume),
            )
        report = dict(report)
        report["source_pool_offset"] = int(source_offset)
        report["next_pool_offset"] = int(source_offset) + (2 if kind == "continuous" else 1)
        if report.get("replay_count", report.get("replay", {}).get("count", 0)) < 1:
            return None, report
        replay = load_transition_replay(shard_path)
        return replay, report

    evaluator = None
    if evaluate:
        from source.experiments.round3_final_eval import evaluate_final_checkpoint

        def evaluator(task_name, arm, checkpoint_path, update_step):
            result = evaluate_final_checkpoint(
                checkpoint=checkpoint_path,
                task=task_name,
                output_dir=output_root / arm / "final_eval" / f"update_{int(update_step):06d}",
                device=str(device),
                resume=True,
            )
            return {
                "success_rate": float(result["success_rate"]),
                "episodes": len(result["episodes"]),
                "cohort_sha256": str(result["cohort_sha256"]),
            }

    return run_closed_loop_training(
        checkpoint=checkpoint,
        offline_replay_path=offline_replay_path,
        output_root=output_root,
        task=str(task),
        device=str(device),
        seed=int(seed),
        max_updates=int(max_updates),
        curve_interval=int(curve_interval),
        arms=tuple(selected),
        model_loader=_load_default_model,
        collector=collector,
        evaluator=evaluator,
        loss_weights_by_arm=loss_weights,
        resume=bool(resume),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(command):
        command.add_argument("--task", choices=SUPPORTED_TASKS, required=True)
        command.add_argument("--checkpoint", default=None)
        command.add_argument("--pool", default=None)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--seed", type=int, default=3072)
        command.add_argument("--guidance-mode", choices=("none", "post_opt", "guided_flow"), default="none")
        command.add_argument("--guidance-step-size", type=float, default=0.01)
        command.add_argument("--guidance-last-steps", type=int, default=5)
        command.add_argument("--guidance-inner-steps", type=int, default=5)
        command.add_argument("--guidance-max-rms-offset", type=float, default=0.20)
        command.add_argument("--actor-warm-start", action="store_true")

    prepare = subparsers.add_parser("prepare-fixed-replay")
    common(prepare)
    prepare.add_argument("--output-root", default=None)
    prepare.add_argument("--offline-windows", type=int, default=512)
    prepare.add_argument("--guidance-report", default=None)
    prepare.add_argument("--resume", action="store_true")

    train = subparsers.add_parser("train-fixed-replay")
    common(train)
    train.add_argument("--replay", required=True)
    train.add_argument("--output-root", required=True)
    train.add_argument("--max-updates", type=int, default=200)
    train.add_argument("--curve-interval", type=int, default=10)
    train.add_argument("--arms", nargs="*", default=[])
    train.add_argument("--resume", action="store_true")
    train.add_argument("--no-eval", dest="evaluate", action="store_false")
    train.set_defaults(evaluate=True)

    closed = subparsers.add_parser("train-closed-loop")
    common(closed)
    closed.add_argument("--offline-replay", required=True)
    closed.add_argument("--output-root", required=True)
    closed.add_argument("--fixed-training-root", default=None)
    closed.add_argument("--max-updates", type=int, default=200)
    closed.add_argument("--curve-interval", type=int, default=10)
    closed.add_argument("--arms", nargs="*", default=[])
    closed.add_argument("--resume", action="store_true")
    closed.add_argument("--no-eval", dest="evaluate", action="store_false")
    closed.set_defaults(evaluate=True)

    dry = subparsers.add_parser("dry-run")
    dry.add_argument("--task", choices=SUPPORTED_TASKS, required=True)
    dry.add_argument("--device", default="cuda:0")
    dry.add_argument("--max-updates", type=int, default=200)
    dry.add_argument("--curve-interval", type=int, default=10)
    dry.add_argument("--arms", nargs="*", default=[])
    dry.set_defaults(function=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "dry-run":
        from source.common.round3_eval import validate_gpu_visibility

        if str(args.device).startswith("cuda"):
            validate_gpu_visibility(args.device)
        from source.experiments.round3_online_runner import TRAINING_ARMS
        from source.experiments.round3_update_curve import optimizer_update_steps

        print(
            json.dumps(
                {
                    "task": args.task,
                    "device": args.device,
                    "max_optimizer_updates": int(args.max_updates),
                    "curve_steps": list(optimizer_update_steps(args.max_updates, args.curve_interval)),
                    "arms": list(args.arms or TRAINING_ARMS),
                    "gpu_visibility_rule": "CUDA_VISIBLE_DEVICES must contain only 0,1,2,3",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if str(args.device).startswith("cuda"):
        from source.common.gpu_environment import configure_mujoco_egl_device
        from source.common.round3_eval import validate_gpu_visibility

        validate_gpu_visibility(args.device)
        configure_mujoco_egl_device()

    checkpoint = _path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINTS[args.task]
    pool = (
        _path(args.pool)
        if args.pool
        else ROOT / "outputs/round3/phase2" / f"{args.task}_online_pool.json"
    )
    if args.command == "prepare-fixed-replay":
        output_root = (
            _path(args.output_root)
            if args.output_root
            else ROOT / "outputs/round3/online_supervision" / args.task / "fixed_replay_v1"
        )
        result = prepare_fixed_replay(
            task=args.task,
            checkpoint=checkpoint,
            pool_path=pool,
            output_root=output_root,
            device=args.device,
            seed=args.seed,
            offline_windows=args.offline_windows,
            guidance_mode=args.guidance_mode,
            guidance_step_size=args.guidance_step_size,
            guidance_last_steps=args.guidance_last_steps,
            guidance_inner_steps=args.guidance_inner_steps,
            guidance_max_rms_offset=args.guidance_max_rms_offset,
            actor_warm_start=args.actor_warm_start,
            guidance_report=(
                None if args.guidance_report is None else _path(args.guidance_report)
            ),
            resume=args.resume,
        )
    elif args.command == "train-fixed-replay":
        result = train_fixed_replay(
            task=args.task,
            checkpoint=checkpoint,
            replay_path=_path(args.replay),
            output_root=_path(args.output_root),
            device=args.device,
            seed=args.seed,
            max_updates=args.max_updates,
            curve_interval=args.curve_interval,
            arms=tuple(args.arms),
            resume=args.resume,
            evaluate=args.evaluate,
        )
    else:
        result = train_closed_loop(
            task=args.task,
            checkpoint=checkpoint,
            offline_replay_path=_path(args.offline_replay),
            pool_path=pool,
            output_root=_path(args.output_root),
            device=args.device,
            seed=args.seed,
            max_updates=args.max_updates,
            curve_interval=args.curve_interval,
            arms=tuple(args.arms),
            guidance_mode=args.guidance_mode,
            guidance_step_size=args.guidance_step_size,
            guidance_last_steps=args.guidance_last_steps,
            guidance_inner_steps=args.guidance_inner_steps,
            guidance_max_rms_offset=args.guidance_max_rms_offset,
            actor_warm_start=args.actor_warm_start,
            fixed_training_root=(
                None if args.fixed_training_root is None else _path(args.fixed_training_root)
            ),
            resume=args.resume,
            evaluate=args.evaluate,
        )
    print(json.dumps({"status": result.get("status"), "task": args.task}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
