"""Closed-loop Round 3 online-training orchestration.

This module owns the 100-environment-step/update schedule and the resumable
replay-shard seam.  Runtime collection and final evaluation are injected so
the schedule can be tested on CPU without importing EGL or MuJoCo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from source.experiments.round3_online_runner import (
    SUPPORTED_TASKS,
    _file_sha256,
    _jsonable,
    _save_adapter_state,
    _save_model_checkpoint,
    _write_json_atomic,
)
from source.experiments.round3_online_supervision import (
    FastOnlineSupervisionAdapter,
    LossWeights,
    SupervisionBatchSampler,
    TRAINING_ARMS,
)
from source.experiments.round3_phase2 import (
    TransitionReplay,
    concatenate_replays,
    load_transition_replay,
    save_transition_replay,
)
from source.experiments.round3_update_curve import (
    make_curve_row,
    optimizer_update_steps,
    write_update_curve,
)


def closed_loop_collection_kind(optimizer_update_step: int) -> str:
    """Return the 4:1 continuous/grounded event for one update."""
    step = int(optimizer_update_step)
    if step < 1:
        raise ValueError("optimizer_update_step must be positive")
    return "grounded" if step % 5 == 0 else "continuous"


def closed_loop_environment_steps(
    optimizer_update_step: int,
    *,
    environment_steps_per_update: int = 100,
) -> int:
    if int(environment_steps_per_update) < 1:
        raise ValueError("environment_steps_per_update must be positive")
    if int(optimizer_update_step) < 0:
        raise ValueError("optimizer_update_step must be non-negative")
    return int(optimizer_update_step) * int(environment_steps_per_update)


def validate_closed_loop_schedule(
    max_updates: int = 200,
    *,
    environment_steps_per_update: int = 100,
) -> tuple[dict[str, Any], ...]:
    """Materialize and audit the fixed 80/20 schedule."""
    maximum = int(max_updates)
    if maximum < 1:
        raise ValueError("max_updates must be positive")
    rows = []
    for step in range(1, maximum + 1):
        kind = closed_loop_collection_kind(step)
        rows.append(
            {
                "optimizer_update_step": step,
                "environment_steps": closed_loop_environment_steps(
                    step,
                    environment_steps_per_update=environment_steps_per_update,
                ),
                "collection_kind": kind,
                "collection_environment_steps": int(environment_steps_per_update),
                "continuous_environment_steps": (
                    int(environment_steps_per_update) if kind == "continuous" else 0
                ),
                "grounded_environment_steps": (
                    int(environment_steps_per_update) if kind == "grounded" else 0
                ),
            }
        )
    continuous = sum(row["continuous_environment_steps"] for row in rows)
    grounded = sum(row["grounded_environment_steps"] for row in rows)
    if continuous * 20 != grounded * 80 and maximum % 5 == 0:
        raise AssertionError("closed-loop schedule is not 80/20")
    return tuple(rows)


Collector = Callable[
    [str, int, str, Any, Path],
    tuple[TransitionReplay | None, Mapping[str, Any]],
]
Evaluator = Callable[[str, str, Path, int], Mapping[str, Any]]
ModelLoader = Callable[[Path, str], tuple[Any, Path]]


def _pending_curve_row(step: int, checkpoint: Path, checkpoint_hash: str, *, environment_steps: int) -> dict[str, Any]:
    return {
        "optimizer_update_step": int(step),
        "environment_steps": int(environment_steps),
        "success_rate": None,
        "success_rate_percent": None,
        "episodes": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "cohort_sha256": None,
        "status": "evaluation_pending",
        "skipped_reason": "no evaluator supplied",
    }


def run_closed_loop_training(
    *,
    checkpoint: str | Path,
    offline_replay_path: str | Path,
    output_root: str | Path,
    task: str,
    device: str = "cpu",
    seed: int = 3072,
    max_updates: int = 200,
    curve_interval: int = 10,
    environment_steps_per_update: int = 100,
    arms: tuple[str, ...] = TRAINING_ARMS,
    model_loader: ModelLoader,
    collector: Collector | None = None,
    evaluator: Evaluator | None = None,
    loss_weights_by_arm: Mapping[str, LossWeights] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run closed-loop arms from one checkpoint and a shared offline pool."""
    if str(task) not in SUPPORTED_TASKS:
        raise ValueError(f"closed-loop Round3 training only supports {SUPPORTED_TASKS}")
    if str(device).startswith("cuda"):
        from source.common.round3_eval import validate_gpu_visibility

        validate_gpu_visibility(str(device))
    maximum = int(max_updates)
    interval = int(curve_interval)
    if maximum < 1 or interval < 1 or maximum % interval:
        raise ValueError("max_updates must be positive and divisible by curve_interval")
    schedule = validate_closed_loop_schedule(
        maximum,
        environment_steps_per_update=int(environment_steps_per_update),
    )
    arms = tuple(str(arm) for arm in arms)
    unknown = [arm for arm in arms if arm not in TRAINING_ARMS]
    if unknown or not arms:
        raise ValueError(f"unknown or empty closed-loop arms: {unknown}")
    checkpoint = Path(checkpoint).resolve()
    offline_replay_path = Path(offline_replay_path).resolve()
    output_root = Path(output_root).resolve()
    if not checkpoint.is_file() or not offline_replay_path.is_file():
        raise FileNotFoundError("checkpoint and offline replay must exist")
    if output_root.exists() and not resume and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty closed-loop root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    offline = load_transition_replay(offline_replay_path)
    root_state = {
        "schema_version": 1,
        "phase": "round3_closed_loop_online_supervision",
        "task": str(task),
        "checkpoint": {"path": str(checkpoint), "sha256": _file_sha256(checkpoint)},
        "offline_replay": {
            "path": str(offline_replay_path),
            "sha256": offline.content_sha256(),
            "count": offline.count,
        },
        "device": str(device),
        "seed": int(seed),
        "max_optimizer_updates": maximum,
        "curve_interval": interval,
        "environment_steps_per_update": int(environment_steps_per_update),
        "schedule_summary": {
            "continuous_environment_steps": sum(
                row["continuous_environment_steps"] for row in schedule
            ),
            "grounded_environment_steps": sum(
                row["grounded_environment_steps"] for row in schedule
            ),
            "ratio": "80/20 over complete five-update blocks",
        },
        "arms": list(arms),
        "status": "running",
    }
    manifest_path = output_root / "manifest.json"
    if resume and manifest_path.is_file():
        persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "task",
            "checkpoint",
            "offline_replay",
            "device",
            "seed",
            "max_optimizer_updates",
            "curve_interval",
            "environment_steps_per_update",
            "arms",
        ):
            if persisted.get(key) != root_state.get(key):
                raise ValueError(f"closed-loop manifest drift at {key}")
    _write_json_atomic(manifest_path, root_state)

    results: dict[str, Any] = {}
    for arm in arms:
        arm_dir = output_root / arm
        state_path = arm_dir / "run_state.json"
        prior = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if resume and state_path.is_file()
            else None
        )
        if prior is not None and prior.get("status") == "ok":
            results[arm] = prior
            continue
        if arm == "t0_frozen":
            # T0 is the same frozen model at every horizontal location.  Its
            # curve is intentionally held at environment step zero.
            model, _ = model_loader(checkpoint, device)
            initial = arm_dir / "checkpoints" / f"{arm}_update_000000.ckpt"
            record = _save_model_checkpoint(model, initial)
            rows = []
            for step in optimizer_update_steps(maximum, interval):
                if evaluator is None:
                    rows.append(
                        _pending_curve_row(
                            step,
                            initial,
                            record["sha256"],
                            environment_steps=0,
                        )
                    )
                else:
                    evaluation = dict(evaluator(str(task), arm, initial, int(step)))
                    rows.append(
                        make_curve_row(
                            optimizer_update_step=int(step),
                            environment_steps=0,
                            success_rate=float(evaluation["success_rate"]),
                            episodes=int(evaluation.get("episodes", 200)),
                            checkpoint=str(initial),
                            checkpoint_sha256=record["sha256"],
                            cohort_sha256=str(evaluation["cohort_sha256"]),
                        )
                    )
            curve = _publish_curve(
                arm_dir,
                task=str(task),
                arm=arm,
                rows=rows,
                maximum=maximum,
                interval=interval,
                device=device,
                evaluator=evaluator,
            )
            state = {
                **root_state,
                "arm": arm,
                "status": "ok" if evaluator is not None else "training_ok_evaluation_pending",
                "optimizer_update_step": maximum,
                "environment_steps": 0,
                "checkpoint": record,
                "replay_cursor": {
                    "next_optimizer_update_step": maximum + 1,
                    "next_collection_event": maximum + 1,
                },
                "curve": curve,
                "curve_rows": rows,
                "updates": [],
            }
            _write_json_atomic(state_path, _jsonable(state))
            results[arm] = state
            continue

        checkpoint_for_arm = (
            Path(prior["checkpoint"]["path"]) if prior is not None else checkpoint
        )
        model, _ = model_loader(checkpoint_for_arm, device)
        adapter = FastOnlineSupervisionAdapter(
            model,
            arm=arm,
            seed=int(seed),
            device=device,
            loss_weights=(
                None
                if loss_weights_by_arm is None
                else loss_weights_by_arm.get(arm)
            ),
        )
        dynamic_replays: list[TransitionReplay] = []
        shard_records: list[dict[str, Any]] = []
        update_start = 0
        curve_rows: list[dict[str, Any]] = []
        update_history: list[dict[str, Any]] = []
        if prior is not None:
            adapter_state_path = Path(prior["adapter_state_path"])
            adapter.load_state_dict(
                __import__("torch").load(
                    adapter_state_path,
                    map_location=device,
                    weights_only=False,
                )
            )
            update_start = int(adapter.optimizer_steps)
            curve_rows = [dict(row) for row in prior.get("curve_rows", [])]
            update_history = [dict(row) for row in prior.get("updates", [])]
            for shard in prior.get("replay_shards", []):
                path = Path(shard["path"])
                replay = load_transition_replay(path)
                if replay.count != int(shard["count"]) or replay.content_sha256() != shard["sha256"]:
                    raise ValueError(f"closed-loop shard metadata mismatch: {path}")
                dynamic_replays.append(replay)
                shard_records.append(dict(shard))

        if update_start == 0 and evaluator is not None:
            initial = arm_dir / "checkpoints" / f"{arm}_update_000000.ckpt"
            initial_record = _save_model_checkpoint(model, initial)
            evaluation = dict(evaluator(str(task), arm, initial, 0))
            curve_rows.append(
                make_curve_row(
                    optimizer_update_step=0,
                    environment_steps=0,
                    success_rate=float(evaluation["success_rate"]),
                    episodes=int(evaluation.get("episodes", 200)),
                    checkpoint=str(initial),
                    checkpoint_sha256=initial_record["sha256"],
                    cohort_sha256=str(evaluation["cohort_sha256"]),
                )
            )
        elif update_start == 0:
            initial = arm_dir / "checkpoints" / f"{arm}_update_000000.ckpt"
            initial_record = _save_model_checkpoint(model, initial)
            curve_rows.append(
                _pending_curve_row(0, initial, initial_record["sha256"], environment_steps=0)
            )

        for step in range(update_start + 1, maximum + 1):
            kind = closed_loop_collection_kind(step)
            event_dir = arm_dir / "replay_shards"
            event_dir.mkdir(parents=True, exist_ok=True)
            shard_path = event_dir / f"{arm}_update_{step:06d}_{kind}.pt"
            if arm == "t1_offline_b_mse" or collector is None:
                collected = None
                collect_report = {
                    "status": "skipped",
                    "kind": kind,
                    "environment_steps": int(environment_steps_per_update),
                    "skipped_reason": (
                        "offline_only_arm" if arm == "t1_offline_b_mse" else "no collector supplied"
                    ),
                }
            else:
                collected, collect_report = collector(
                    arm,
                    int(step),
                    kind,
                    model,
                    shard_path,
                )
            if collected is not None:
                if collected.count < 1:
                    raise ValueError("collector returned an empty replay")
                if shard_path.is_file():
                    persisted = load_transition_replay(shard_path)
                    if persisted.content_sha256() != collected.content_sha256():
                        raise ValueError(f"collector shard already exists with different contents: {shard_path}")
                else:
                    save_transition_replay(shard_path, collected)
                dynamic_replays.append(collected)
                shard_records.append(
                    {
                        "path": str(shard_path),
                        "count": collected.count,
                        "sha256": collected.content_sha256(),
                        "kind": kind,
                        "optimizer_update_step": int(step),
                    }
                )
            combined = concatenate_replays((offline, *dynamic_replays))
            batch = SupervisionBatchSampler(
                combined,
                seed=int(seed) + int(step) - 1,
            ).sample(
                arm,
                batch_size=64,
                auxiliary_batch_size=32,
                require_grounded=any(
                    value == "grounded" for value in combined.resolved_data_kinds
                ),
            )
            report = adapter.update(combined, batch=batch)
            payload = report.to_dict()
            payload.update(
                {
                    "optimizer_step": int(step),
                    "environment_steps": closed_loop_environment_steps(
                        step,
                        environment_steps_per_update=environment_steps_per_update,
                    ),
                    "collection_kind": kind,
                    "collection": _jsonable(collect_report),
                }
            )
            update_history.append(payload)
            checkpoint_path = arm_dir / "checkpoints" / f"{arm}_update_{step:06d}.ckpt"
            checkpoint_record = _save_model_checkpoint(adapter.model, checkpoint_path)
            adapter_state_path = arm_dir / "adapter_state.pt"
            adapter_state_sha256 = _save_adapter_state(adapter, adapter_state_path)
            if step % interval == 0 or step == maximum:
                if evaluator is None:
                    curve_rows.append(
                        _pending_curve_row(
                            step,
                            checkpoint_path,
                            checkpoint_record["sha256"],
                            environment_steps=closed_loop_environment_steps(
                                step,
                                environment_steps_per_update=environment_steps_per_update,
                            ),
                        )
                    )
                else:
                    evaluation = dict(evaluator(str(task), arm, checkpoint_path, int(step)))
                    curve_rows.append(
                        make_curve_row(
                            optimizer_update_step=int(step),
                            environment_steps=closed_loop_environment_steps(
                                step,
                                environment_steps_per_update=environment_steps_per_update,
                            ),
                            success_rate=float(evaluation["success_rate"]),
                            episodes=int(evaluation.get("episodes", 200)),
                            checkpoint=str(checkpoint_path),
                            checkpoint_sha256=checkpoint_record["sha256"],
                            cohort_sha256=str(evaluation["cohort_sha256"]),
                        )
                    )
            state = {
                **root_state,
                "arm": arm,
                "status": "running",
                "optimizer_update_step": int(step),
                "environment_steps": closed_loop_environment_steps(
                    step,
                    environment_steps_per_update=environment_steps_per_update,
                ),
                "checkpoint": checkpoint_record,
                "adapter_state_path": str(adapter_state_path),
                "adapter_state_sha256": adapter_state_sha256,
                "calibration": adapter.calibration,
                "loss_weights": _jsonable(adapter.loss_weights),
                "replay_cursor": {
                    "next_optimizer_update_step": int(step) + 1,
                    "next_collection_event": int(step) + 1,
                    "next_source_pool_offset": _jsonable(
                        collect_report.get("next_pool_offset")
                    ),
                },
                "replay_shards": shard_records,
                "last_update": payload,
                "updates": update_history,
                "curve_rows": curve_rows,
            }
            _write_json_atomic(state_path, _jsonable(state))

        if evaluator is not None:
            refreshed_rows: list[dict[str, Any]] = []
            for row in curve_rows:
                if row.get("status") == "ok" and row.get("cohort_sha256"):
                    refreshed_rows.append(dict(row))
                    continue
                checkpoint_path = Path(row["checkpoint"])
                evaluation = dict(
                    evaluator(
                        str(task),
                        arm,
                        checkpoint_path,
                        int(row["optimizer_update_step"]),
                    )
                )
                refreshed_rows.append(
                    make_curve_row(
                        optimizer_update_step=int(row["optimizer_update_step"]),
                        environment_steps=int(row["environment_steps"]),
                        success_rate=float(evaluation["success_rate"]),
                        episodes=int(evaluation.get("episodes", 200)),
                        checkpoint=str(checkpoint_path),
                        checkpoint_sha256=str(
                            row.get("checkpoint_sha256") or _file_sha256(checkpoint_path)
                        ),
                        cohort_sha256=str(evaluation["cohort_sha256"]),
                    )
                )
            curve_rows = refreshed_rows
        curve = _publish_curve(
            arm_dir,
            task=str(task),
            arm=arm,
            rows=curve_rows,
            maximum=maximum,
            interval=interval,
            device=device,
            evaluator=evaluator,
        )
        final_state = {
            **root_state,
            "arm": arm,
            "status": "ok" if evaluator is not None else "training_ok_evaluation_pending",
            "optimizer_update_step": maximum,
            "environment_steps": closed_loop_environment_steps(
                maximum,
                environment_steps_per_update=environment_steps_per_update,
            ),
            "checkpoint": {
                "path": str(arm_dir / "checkpoints" / f"{arm}_update_{maximum:06d}.ckpt"),
                "sha256": _file_sha256(
                    arm_dir / "checkpoints" / f"{arm}_update_{maximum:06d}.ckpt"
                ),
            },
            "adapter_state_path": str(arm_dir / "adapter_state.pt"),
            "adapter_state_sha256": _file_sha256(arm_dir / "adapter_state.pt"),
            "calibration": adapter.calibration,
            "loss_weights": _jsonable(adapter.loss_weights),
            "replay_cursor": {
                "next_optimizer_update_step": maximum + 1,
                "next_collection_event": maximum + 1,
                "next_source_pool_offset": (
                    None
                    if not update_history
                    else update_history[-1].get("collection", {}).get("next_pool_offset")
                ),
            },
            "replay_shards": shard_records,
            "curve": curve,
            "curve_rows": curve_rows,
            "updates": update_history,
        }
        _write_json_atomic(state_path, _jsonable(final_state))
        results[arm] = final_state

    result = {**root_state, "status": "ok" if evaluator is not None else "training_ok_evaluation_pending", "arm_results": results}
    _write_json_atomic(output_root / "result.json", _jsonable(result))
    return result


def _publish_curve(
    arm_dir: Path,
    *,
    task: str,
    arm: str,
    rows: list[dict[str, Any]],
    maximum: int,
    interval: int,
    device: str,
    evaluator: Evaluator | None,
) -> dict[str, Any]:
    if evaluator is None:
        return write_update_curve(
            arm_dir / "curve",
            task=str(task),
            experiment="round3_closed_loop_online_supervision",
            arm=arm,
            rows=rows,
            cohort={
                "kind": "final",
                "episodes": 200,
                "canonical_sha256": None,
            },
            visible_physical_gpus=(str(device).removeprefix("cuda:"),)
            if str(device).startswith("cuda")
            else (),
            max_updates=maximum,
            interval=interval,
            require_complete=False,
        )
    cohorts = {row.get("cohort_sha256") for row in rows}
    if len(cohorts) != 1 or None in cohorts:
        raise ValueError(f"curve for {arm} uses multiple or missing cohort hashes: {cohorts}")
    return write_update_curve(
        arm_dir / "curve",
        task=str(task),
        experiment="round3_closed_loop_online_supervision",
        arm=arm,
        rows=rows,
        cohort={"kind": "final", "episodes": 200, "canonical_sha256": next(iter(cohorts))},
        visible_physical_gpus=(str(device).removeprefix("cuda:"),)
        if str(device).startswith("cuda")
        else (),
        max_updates=maximum,
        interval=interval,
        require_complete=True,
    )


__all__ = [
    "closed_loop_collection_kind",
    "closed_loop_environment_steps",
    "run_closed_loop_training",
    "validate_closed_loop_schedule",
]
