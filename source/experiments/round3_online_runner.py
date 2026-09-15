"""Resumable fixed-replay training runner for Round 3 T1--T6 arms."""

from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from source.experiments.round3_online_supervision import (
    FastOnlineSupervisionAdapter,
    SupervisionBatch,
    TRAINING_ARMS,
    SupervisionBatchSampler,
)
from source.experiments.round3_phase2 import load_transition_replay
from source.experiments.round3_update_curve import (
    make_curve_row,
    optimizer_update_steps,
    write_update_curve,
)


SUPPORTED_TASKS = ("reacher", "pusht")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_model_checkpoint(model: torch.nn.Module, path: Path) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(model, temporary)
    temporary.replace(path)
    return {"path": str(path), "sha256": _file_sha256(path)}


def _save_adapter_state(adapter: FastOnlineSupervisionAdapter, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(adapter.state_dict(), temporary)
    temporary.replace(path)
    return _file_sha256(path)


def _load_default_model(checkpoint: Path, device: str | torch.device):
    from source.common.checkpoint import load_policy_or_model

    model, resolved = load_policy_or_model(str(checkpoint))
    return model.to(device), Path(resolved or checkpoint)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _batch_from_dict(payload: Mapping[str, Any]) -> SupervisionBatch:
    """Rehydrate a persisted batch manifest without invoking a new RNG draw."""
    return SupervisionBatch(
        b_indices=tuple(int(value) for value in payload["b_indices"]),
        offline_anchor_indices=tuple(
            int(value) for value in payload.get("offline_anchor_indices", ())
        ),
        offline_replacement_indices=tuple(
            int(value) for value in payload.get("offline_replacement_indices", ())
        ),
        offline_a_indices=tuple(
            int(value) for value in payload.get("offline_a_indices", ())
        ),
        ranking_pairs=tuple(
            (int(pair[0]), int(pair[1]))
            for pair in payload.get("ranking_pairs", ())
        ),
        hindsight_indices=tuple(
            int(value) for value in payload.get("hindsight_indices", ())
        ),
        distill_indices=tuple(
            int(value) for value in payload.get("distill_indices", ())
        ),
        counts=payload.get("counts"),
    )


def _batch_manifest(
    *,
    replay,
    arm: str,
    seed: int,
    maximum: int,
    batch_size: int,
    auxiliary_batch_size: int,
    require_grounded: bool,
    path: Path,
    resume: bool,
) -> dict[str, Any]:
    """Create or verify the immutable per-update sampling manifest."""
    replay_hash = replay.content_sha256()
    if resume and path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "arm": str(arm),
            "seed": int(seed),
            "max_optimizer_updates": int(maximum),
            "batch_size": int(batch_size),
            "auxiliary_batch_size": int(auxiliary_batch_size),
            "require_grounded": bool(require_grounded),
            "replay_sha256": replay_hash,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(
                    f"batch manifest drift for {arm} at {key}: "
                    f"expected {value!r}, got {payload.get(key)!r}"
                )
        if len(payload.get("updates", ())) != int(maximum):
            raise ValueError(f"batch manifest for {arm} has incomplete update coverage")
        declared_hash = payload.get("content_sha256")
        unsigned = dict(payload)
        unsigned.pop("content_sha256", None)
        actual_hash = hashlib.sha256(
            json.dumps(
                _jsonable(unsigned),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if declared_hash != actual_hash:
            raise ValueError(f"batch manifest hash mismatch: {path}")
        return payload

    sampler = SupervisionBatchSampler(replay, seed=int(seed))
    updates = []
    for update_step in range(1, int(maximum) + 1):
        batch = sampler.sample(
            arm,
            batch_size=int(batch_size),
            auxiliary_batch_size=int(auxiliary_batch_size),
            require_grounded=bool(require_grounded),
        )
        updates.append(
            {
                "optimizer_update_step": int(update_step),
                "batch": batch.to_dict(),
            }
        )
    payload = {
        "schema_version": 1,
        "arm": str(arm),
        "seed": int(seed),
        "max_optimizer_updates": int(maximum),
        "batch_size": int(batch_size),
        "auxiliary_batch_size": int(auxiliary_batch_size),
        "require_grounded": bool(require_grounded),
        "replay_sha256": replay_hash,
        "updates": updates,
    }
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _write_json_atomic(path, payload)
    return payload


def run_fixed_replay_training(
    *,
    checkpoint: str | Path,
    replay_path: str | Path,
    output_root: str | Path,
    task: str,
    device: str | torch.device = "cpu",
    seed: int = 3072,
    max_updates: int = 200,
    curve_interval: int = 10,
    environment_steps_per_update: int = 100,
    batch_size: int = 64,
    auxiliary_batch_size: int = 32,
    require_grounded: bool = True,
    arms: tuple[str, ...] = TRAINING_ARMS,
    model_loader: Callable[[Path, str | torch.device], tuple[torch.nn.Module, Path]] | None = None,
    evaluator: Callable[[str, str, Path, int], Mapping[str, Any]] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Train all requested arms on one immutable replay and publish progress.

    ``evaluator`` is intentionally injected: the CPU path can verify every
    training artifact without importing MuJoCo/EGL, while the GPU entrypoint
    can provide the canonical final-cohort evaluator.
    """
    maximum = int(max_updates)
    interval = int(curve_interval)
    if maximum < 1 or interval < 1 or maximum % interval:
        raise ValueError("max_updates must be positive and divisible by curve_interval")
    if int(environment_steps_per_update) < 1:
        raise ValueError("environment_steps_per_update must be positive")
    if int(batch_size) != 64 or int(auxiliary_batch_size) != 32:
        raise ValueError(
            "Round3 fixed-replay supervision uses batch_size=64 and "
            "auxiliary_batch_size=32"
        )
    if str(task) not in SUPPORTED_TASKS:
        raise ValueError(f"Round3 fixed-replay training only supports {SUPPORTED_TASKS}")
    if str(device).startswith("cuda"):
        from source.common.round3_eval import validate_gpu_visibility

        validate_gpu_visibility(str(device))
    arms = tuple(str(arm) for arm in arms)
    unknown = [arm for arm in arms if arm not in TRAINING_ARMS]
    if unknown:
        raise ValueError(f"unknown Round3 training arms: {unknown}")
    if not arms:
        raise ValueError("at least one training arm is required")

    checkpoint = Path(checkpoint).resolve()
    replay_path = Path(replay_path).resolve()
    output_root = Path(output_root).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not replay_path.is_file():
        raise FileNotFoundError(replay_path)
    if output_root.exists() and not resume and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    replay = load_transition_replay(replay_path)
    loader = model_loader or _load_default_model
    expected_curve_steps = optimizer_update_steps(maximum, interval)
    root_state = {
        "schema_version": 1,
        "phase": "round3_fixed_replay_supervision",
        "task": str(task),
        "checkpoint": {"path": str(checkpoint), "sha256": _file_sha256(checkpoint)},
        "replay": {"path": str(replay_path), "sha256": replay.content_sha256(), "count": replay.count},
        "device": str(device),
        "seed": int(seed),
        "max_optimizer_updates": maximum,
        "curve_interval": interval,
        "environment_steps_per_update": int(environment_steps_per_update),
        "batch_size": int(batch_size),
        "auxiliary_batch_size": int(auxiliary_batch_size),
        "require_grounded": bool(require_grounded),
        "arms": list(arms),
        "status": "running",
    }
    root_manifest_path = output_root / "manifest.json"
    if resume and root_manifest_path.is_file():
        persisted_root = json.loads(root_manifest_path.read_text(encoding="utf-8"))
        for key in (
            "task",
            "checkpoint",
            "replay",
            "device",
            "seed",
            "max_optimizer_updates",
            "curve_interval",
            "environment_steps_per_update",
            "batch_size",
            "auxiliary_batch_size",
            "require_grounded",
            "arms",
        ):
            if persisted_root.get(key) != root_state.get(key):
                raise ValueError(f"fixed-replay manifest drift at {key}")
    _write_json_atomic(root_manifest_path, root_state)
    arm_results: dict[str, Any] = {}

    for arm in arms:
        arm_dir = output_root / arm
        state_path = arm_dir / "run_state.json"
        batches_path = arm_dir / "batch_manifest.json"
        batches_payload = _batch_manifest(
            replay=replay,
            arm=arm,
            seed=int(seed),
            maximum=maximum,
            batch_size=int(batch_size),
            auxiliary_batch_size=int(auxiliary_batch_size),
            require_grounded=bool(require_grounded),
            path=batches_path,
            resume=bool(resume),
        )
        batches_by_step = {
            int(item["optimizer_update_step"]): _batch_from_dict(item["batch"])
            for item in batches_payload["updates"]
        }
        prior = json.loads(state_path.read_text(encoding="utf-8")) if resume and state_path.is_file() else None
        if prior is not None and prior.get("status") == "ok":
            arm_results[arm] = prior
            continue
        if arm == "t0_frozen":
            model, resolved = loader(checkpoint, device)
            checkpoints_dir = arm_dir / "checkpoints"
            initial_checkpoint = checkpoints_dir / f"{arm}_update_000000.ckpt"
            if initial_checkpoint.is_file():
                checkpoint_record = {
                    "path": str(initial_checkpoint),
                    "sha256": _file_sha256(initial_checkpoint),
                }
            else:
                checkpoint_record = _save_model_checkpoint(model, initial_checkpoint)
            frozen_rows: list[dict[str, Any]] = []
            for update_step in expected_curve_steps:
                if evaluator is not None:
                    evaluation = dict(
                        evaluator(str(task), arm, initial_checkpoint, int(update_step))
                    )
                    frozen_rows.append(
                        make_curve_row(
                            optimizer_update_step=int(update_step),
                            environment_steps=0,
                            success_rate=float(evaluation["success_rate"]),
                            episodes=int(evaluation.get("episodes", 200)),
                            checkpoint=str(initial_checkpoint),
                            checkpoint_sha256=checkpoint_record["sha256"],
                            cohort_sha256=str(evaluation["cohort_sha256"]),
                        )
                    )
                else:
                    frozen_rows.append(
                        {
                            "optimizer_update_step": int(update_step),
                            "environment_steps": 0,
                            "success_rate": None,
                            "success_rate_percent": None,
                            "episodes": 0,
                            "checkpoint": str(initial_checkpoint),
                            "checkpoint_sha256": checkpoint_record["sha256"],
                            "cohort_sha256": None,
                            "status": "evaluation_pending",
                            "skipped_reason": "no evaluator supplied",
                        }
                    )
            if evaluator is not None:
                cohorts = {row["cohort_sha256"] for row in frozen_rows}
                if len(cohorts) != 1:
                    raise ValueError(f"curve for {arm} uses multiple cohort hashes: {cohorts}")
                curve_metadata = write_update_curve(
                    arm_dir / "curve",
                    task=str(task),
                    experiment="round3_fixed_replay_supervision",
                    arm=arm,
                    rows=frozen_rows,
                    cohort={
                        "kind": "final",
                        "episodes": 200,
                        "canonical_sha256": next(iter(cohorts)),
                    },
                    visible_physical_gpus=(str(device).removeprefix("cuda:"),)
                    if str(device).startswith("cuda")
                    else (),
                    max_updates=maximum,
                    interval=interval,
                    require_complete=True,
                )
            else:
                curve_metadata = write_update_curve(
                    arm_dir / "curve",
                    task=str(task),
                    experiment="round3_fixed_replay_supervision",
                    arm=arm,
                    rows=frozen_rows,
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
            frozen_state = {
                **root_state,
                "arm": arm,
                "status": "ok" if evaluator is not None else "training_ok_evaluation_pending",
                "optimizer_update_step": maximum,
                "environment_steps": 0,
                "checkpoint": checkpoint_record,
                "replay_cursor": {
                    "next_optimizer_update_step": maximum + 1,
                    "batch_manifest_update_step": maximum,
                },
                "curve": curve_metadata,
                "curve_rows": frozen_rows,
                "updates": [],
            }
            _write_json_atomic(state_path, _jsonable(frozen_state))
            arm_results[arm] = frozen_state
            del model
            continue
        update_start = int(prior.get("optimizer_update_step", 0)) if prior else 0
        if update_start < 0 or update_start > maximum:
            raise ValueError(f"invalid resume optimizer step for {arm}: {update_start}")
        checkpoint_for_arm = (
            Path(prior["checkpoint"]["path"]) if prior is not None else checkpoint
        )
        if not checkpoint_for_arm.is_file():
            raise FileNotFoundError(checkpoint_for_arm)
        adapter_model, resolved = loader(checkpoint_for_arm, device)
        adapter = FastOnlineSupervisionAdapter(
            adapter_model,
            arm=arm,
            seed=int(seed),
            device=device,
        )
        adapter_state_path: Path | None = None
        if prior is not None:
            adapter_state_path = Path(prior["adapter_state_path"])
            if not adapter_state_path.is_file():
                raise FileNotFoundError(adapter_state_path)
            adapter.load_state_dict(
                torch.load(adapter_state_path, map_location=device, weights_only=False)
            )
            if int(adapter.optimizer_steps) != int(update_start):
                raise ValueError(
                    f"resume optimizer step mismatch for {arm}: "
                    f"state={adapter.optimizer_steps}, run_state={update_start}"
                )

        checkpoints_dir = arm_dir / "checkpoints"
        curve_rows: list[dict[str, Any]] = []
        update_history: list[dict[str, Any]] = []
        if update_start == 0:
            initial_checkpoint = checkpoints_dir / f"{arm}_update_000000.ckpt"
            checkpoint_record = _save_model_checkpoint(adapter.model, initial_checkpoint)
            if evaluator is not None:
                evaluation = dict(evaluator(str(task), arm, initial_checkpoint, 0))
                curve_rows.append(
                    make_curve_row(
                        optimizer_update_step=0,
                        environment_steps=0,
                        success_rate=float(evaluation["success_rate"]),
                        episodes=int(evaluation.get("episodes", 200)),
                        checkpoint=str(initial_checkpoint),
                        checkpoint_sha256=checkpoint_record["sha256"],
                        cohort_sha256=str(evaluation["cohort_sha256"]),
                    )
                )
        elif prior is not None:
            curve_rows = [dict(row) for row in prior.get("curve_rows", [])]
            update_history = [dict(item) for item in prior.get("updates", [])]

        if arm != "t1_offline_b_mse" and adapter.calibration is None:
            adapter.calibrate(replay, batches_by_step[1])

        for update_step in range(update_start + 1, maximum + 1):
            batch = batches_by_step[update_step]
            report = adapter.update(replay, batch=batch)
            if update_step != adapter.optimizer_steps:
                raise RuntimeError(
                    f"adapter update drifted for {arm}: {adapter.optimizer_steps} != {update_step}"
                )
            update_payload = report.to_dict()
            update_payload["optimizer_step"] = int(update_step)
            update_payload["environment_steps"] = int(update_step) * int(
                environment_steps_per_update
            )
            update_history.append(update_payload)
            if update_step % interval == 0 or update_step == maximum:
                checkpoint_path = checkpoints_dir / f"{arm}_update_{update_step:06d}.ckpt"
                checkpoint_record = _save_model_checkpoint(adapter.model, checkpoint_path)
                adapter_state_path = arm_dir / "adapter_state.pt"
                adapter_state_sha256 = _save_adapter_state(adapter, adapter_state_path)
                if evaluator is not None:
                    evaluation = dict(evaluator(str(task), arm, checkpoint_path, update_step))
                    curve_rows.append(
                        make_curve_row(
                            optimizer_update_step=update_step,
                            environment_steps=update_step * int(environment_steps_per_update),
                            success_rate=float(evaluation["success_rate"]),
                            episodes=int(evaluation.get("episodes", 200)),
                            checkpoint=str(checkpoint_path),
                            checkpoint_sha256=checkpoint_record["sha256"],
                            cohort_sha256=str(evaluation["cohort_sha256"]),
                        )
                    )
                else:
                    curve_rows.append(
                        {
                            "optimizer_update_step": int(update_step),
                            "environment_steps": int(update_step) * int(environment_steps_per_update),
                            "success_rate": None,
                            "success_rate_percent": None,
                            "episodes": 0,
                            "checkpoint": str(checkpoint_path),
                            "checkpoint_sha256": checkpoint_record["sha256"],
                            "cohort_sha256": None,
                            "status": "evaluation_pending",
                            "skipped_reason": "no evaluator supplied",
                        }
                    )
                state = {
                    **root_state,
                    "arm": arm,
                    "status": "running",
                    "optimizer_update_step": int(update_step),
                    "environment_steps": int(update_step) * int(environment_steps_per_update),
                    "checkpoint": checkpoint_record,
                    "adapter_state_path": str(adapter_state_path),
                    "adapter_state_sha256": adapter_state_sha256,
                    "last_update": update_payload,
                    "updates": update_history,
                    "batch_manifest": {
                        "path": str(batches_path),
                        "sha256": _file_sha256(batches_path),
                    },
                    "replay_cursor": {
                        "next_optimizer_update_step": int(update_step) + 1,
                        "batch_manifest_update_step": int(update_step),
                    },
                    "calibration": adapter.calibration,
                    "loss_weights": _jsonable(adapter.loss_weights),
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
            cohorts = {row["cohort_sha256"] for row in curve_rows}
            if len(cohorts) != 1:
                raise ValueError(f"curve for {arm} uses multiple cohort hashes: {cohorts}")
            curve_metadata = write_update_curve(
                arm_dir / "curve",
                task=str(task),
                experiment="round3_fixed_replay_supervision",
                arm=arm,
                rows=curve_rows,
                cohort={
                    "kind": "final",
                    "episodes": 200,
                    "canonical_sha256": next(iter(cohorts)),
                },
                visible_physical_gpus=(str(device).removeprefix("cuda:"),)
                if str(device).startswith("cuda")
                else (),
                max_updates=maximum,
                interval=interval,
                require_complete=True,
            )
        else:
            curve_metadata = write_update_curve(
                arm_dir / "curve",
                task=str(task),
                experiment="round3_fixed_replay_supervision",
                arm=arm,
                rows=curve_rows,
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
        final_checkpoint = checkpoints_dir / f"{arm}_update_{maximum:06d}.ckpt"
        final_state = {
            **root_state,
            "arm": arm,
            "status": "ok" if evaluator is not None else "training_ok_evaluation_pending",
            "optimizer_update_step": maximum,
            "environment_steps": maximum * int(environment_steps_per_update),
            "checkpoint": {
                "path": str(final_checkpoint),
                "sha256": _file_sha256(final_checkpoint),
            },
            "adapter_state_path": str(arm_dir / "adapter_state.pt"),
            "adapter_state_sha256": _file_sha256(arm_dir / "adapter_state.pt"),
            "batch_manifest": {
                "path": str(batches_path),
                "sha256": _file_sha256(batches_path),
            },
            "replay_cursor": {
                "next_optimizer_update_step": maximum + 1,
                "batch_manifest_update_step": maximum,
            },
            "calibration": adapter.calibration,
            "loss_weights": _jsonable(adapter.loss_weights),
            "updates": update_history,
            "curve": curve_metadata,
            "curve_rows": curve_rows,
        }
        _write_json_atomic(state_path, _jsonable(final_state))
        arm_results[arm] = final_state

    result = {
        **root_state,
        "status": "ok" if evaluator is not None else "training_ok_evaluation_pending",
        "arm_results": arm_results,
    }
    _write_json_atomic(output_root / "result.json", _jsonable(result))
    return result


__all__ = ["run_fixed_replay_training"]
