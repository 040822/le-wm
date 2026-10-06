"""Frozen CVPR Table 2 training matrix and run-level provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

import numpy as np
from lightning.pytorch.callbacks import Callback


TASKS = ("tworoom", "pusht", "reacher", "cube")
SEEDS = (3072, 4096, 5120)
ARMS: dict[str, dict[str, Any]] = {
    "a_only": {
        "label": "A-only",
        "train_mode": "stage_a",
        "stage_b_action_source": "joint",
        "detach_clean_action": False,
        "lambda_latent": 0.0,
        "lambda_sigreg": 0.09,
    },
    "b_only_clean": {
        "label": "B-only-clean",
        "train_mode": "stage_b",
        "stage_b_action_source": "joint",
        "detach_clean_action": False,
        "lambda_latent": 1.0,
        "lambda_sigreg": 0.09,
    },
    "shared_recorded_control": {
        "label": "Shared-Recorded-control",
        "train_mode": "stage_ab",
        "stage_b_action_source": "recorded_control",
        "detach_clean_action": False,
        "lambda_latent": 1.0,
        "lambda_sigreg": 0.09,
    },
    "shared_pred_detach": {
        "label": "Shared-Pred-Detach",
        "train_mode": "stage_ab",
        "stage_b_action_source": "joint",
        "detach_clean_action": True,
        "lambda_latent": 1.0,
        "lambda_sigreg": 0.09,
    },
    "shared_pred_full": {
        "label": "Shared-Pred-Full",
        "train_mode": "stage_ab",
        "stage_b_action_source": "joint",
        "detach_clean_action": False,
        "lambda_latent": 1.0,
        "lambda_sigreg": 0.09,
    },
}

RNG_STREAMS = {
    "stage_a_timestep": 0x13579BDF,
    "stage_a_noise": 0x2468ACE0,
    "stage_b_action_mix": 0x51A7B00B,
}

_SHARED_PREFIXES = (
    "encoder.",
    "projector.",
    "predictor.",
    "latent_input.",
    "action_input.",
    "z_condition.",
    "time_mlp.",
    "mode_embedding.",
)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def write_json_atomic(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def sha256_state_dict(module, *, prefixes: tuple[str, ...] | None = None) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if prefixes is not None and not name.startswith(prefixes):
            continue
        value = tensor.detach().cpu().contiguous().reshape(-1)
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def arm_config(arm: str) -> dict[str, Any]:
    try:
        return dict(ARMS[arm])
    except KeyError as exc:
        raise ValueError(f"unknown Table 2 arm {arm!r}; expected one of {tuple(ARMS)}") from exc


def run_id(task: str, arm: str, seed: int) -> str:
    if task not in TASKS:
        raise ValueError(f"unknown Table 2 task {task!r}")
    arm_config(arm)
    if int(seed) not in SEEDS:
        raise ValueError(f"unknown Table 2 training seed {seed!r}")
    return f"{task}/seed_{int(seed)}/{arm}"


def matrix_rows(
    *, tasks: tuple[str, ...] = TASKS, seeds: tuple[int, ...] = SEEDS
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        if int(seed) not in SEEDS:
            raise ValueError(f"seed {seed} is not part of the frozen Table 2 matrix")
        for task in tasks:
            if task not in TASKS:
                raise ValueError(f"task {task} is not part of the frozen Table 2 matrix")
            for arm, definition in ARMS.items():
                rows.append(
                    {
                        "task": task,
                        "seed": int(seed),
                        "arm": arm,
                        "label": definition["label"],
                        "run_id": run_id(task, arm, int(seed)),
                    }
                )
    return rows


def _subset_indices(dataset_split) -> np.ndarray:
    indices = getattr(dataset_split, "indices", None)
    if indices is None:
        raise TypeError("Table 2 requires index-based train/validation splits")
    result = np.asarray(indices, dtype=np.int64)
    if result.ndim != 1 or not len(result):
        raise ValueError("Table 2 split indices must be non-empty one-dimensional arrays")
    return result


def write_training_identity(
    *,
    cfg,
    dataset,
    train_set,
    val_set,
    train_loader,
    policy,
    normalizer_stats: Mapping[str, Any],
    run_dir: str | Path,
    code_paths: tuple[str | Path, ...],
) -> Path:
    """Persist the common initialization, data split, order, and random streams."""
    import torch

    table2 = cfg.cvpr_table2
    task = str(table2.task)
    arm = str(table2.arm)
    seed = int(cfg.seed)
    definition = arm_config(arm)
    if task not in TASKS or seed not in SEEDS:
        raise ValueError("Table 2 training identity is outside the frozen matrix")
    if str(cfg.train_mode) != definition["train_mode"]:
        raise ValueError("Table 2 arm train_mode differs from its frozen definition")
    if str(cfg.stage_b_action_source) != definition["stage_b_action_source"]:
        raise ValueError("Table 2 arm B action source differs from its frozen definition")
    if bool(cfg.detach_clean_action) != bool(definition["detach_clean_action"]):
        raise ValueError("Table 2 arm detach flag differs from its frozen definition")

    train_indices = _subset_indices(train_set)
    val_indices = _subset_indices(val_set)
    if np.intersect1d(train_indices, val_indices).size:
        raise ValueError("Table 2 train and validation sample indices overlap")
    sampler = getattr(train_loader, "sampler", None)
    if not callable(getattr(sampler, "_ordered_indices", None)):
        raise RuntimeError("Table 2 requires a sampler with auditable epoch order")
    order_hashes = {}
    original_epoch = int(sampler.epoch)
    try:
        for epoch in range(int(cfg.trainer.max_epochs)):
            sampler.epoch = epoch
            order = np.asarray(sampler._ordered_indices(), dtype=np.int64)
            if len(order) != len(train_indices):
                raise RuntimeError("Table 2 epoch order does not cover the training split")
            order_hashes[str(epoch + 1)] = hashlib.sha256(order.tobytes()).hexdigest()
    finally:
        sampler.epoch = original_epoch

    model = getattr(policy, "model", policy)
    dataset_path = Path(str(getattr(dataset, "h5_path", ""))).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Table 2 dataset path is unavailable: {dataset_path}")
    code_hashes = {}
    for value in code_paths:
        path = Path(value).resolve()
        code_hashes[str(path)] = sha256_file(path)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    encoder_parameter_ids = {
        id(parameter) for parameter in getattr(model, "encoder", ()).parameters()
    } if hasattr(model, "encoder") else set()
    encoder_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if id(parameter) in encoder_parameter_ids
    )
    active_encoder_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if id(parameter) in encoder_parameter_ids and parameter.requires_grad
    )
    expected_dataset_hashes = table2.get("dataset_sha256", {})
    frozen_dataset_sha256 = expected_dataset_hashes.get(task)
    if frozen_dataset_sha256 is None:
        frozen_dataset_sha256 = sha256_file(dataset_path)
    identity = {
        "schema_version": 1,
        "experiment": "cvpr_table2_v1",
        "task": task,
        "arm": arm,
        "arm_definition": definition,
        "training_seed": seed,
        "run_id": run_id(task, arm, seed),
        "initial_model_sha256": sha256_state_dict(model),
        "initial_shared_representation_sha256": sha256_state_dict(
            model, prefixes=_SHARED_PREFIXES
        ),
        "initial_module_sha256": {
            name: sha256_state_dict(getattr(model, name))
            for name in ("encoder", "projector", "predictor")
            if hasattr(model, name)
        },
        "model_parameters": {
            "total": int(total_parameters),
            "trainable": int(trainable_parameters),
            "total_including_encoder": int(total_parameters),
            "total_excluding_encoder": int(total_parameters - encoder_parameters),
            "trainable_including_encoder": int(trainable_parameters),
            "trainable_excluding_encoder": int(
                trainable_parameters - active_encoder_parameters
            ),
            "encoder_total": int(encoder_parameters),
            "encoder_trainable": int(active_encoder_parameters),
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": str(frozen_dataset_sha256),
            "frameskip": int(dataset.frameskip),
            "num_steps": int(dataset.num_steps),
            "clip_count": int(len(dataset.clip_indices)),
        },
        "split": {
            "method": "torch.utils.data.random_split",
            "seed": seed,
            "train_split": float(cfg.train_split),
            "train_windows": int(len(train_indices)),
            "validation_windows": int(len(val_indices)),
            "train_indices_sha256": hashlib.sha256(train_indices.tobytes()).hexdigest(),
            "validation_indices_sha256": hashlib.sha256(val_indices.tobytes()).hexdigest(),
            "sample_order_sha256_by_epoch": order_hashes,
            "sampler": type(sampler).__name__,
            "sampler_seed": int(getattr(sampler, "seed", seed)),
            "sampler_chunk_size": int(getattr(sampler, "chunk_size", 0)) or None,
        },
        "normalizer": {
            "stats": dict(normalizer_stats),
            "sha256": sha256_value(dict(normalizer_stats)),
            "source": "full_original_dataset",
        },
        "optimization": {
            "epochs": int(cfg.trainer.max_epochs),
            "batch_size": int(cfg.loader.batch_size),
            "updates_per_epoch": int(len(train_loader)),
            "expected_total_updates": int(len(train_loader) * cfg.trainer.max_epochs),
            "optimizer": {
                "type": str(cfg.optimizer.type),
                "learning_rate": float(cfg.optimizer.lr),
                "weight_decay": float(cfg.optimizer.weight_decay),
            },
            "precision": str(cfg.trainer.precision),
        },
        "rng": {
            "root_seed": seed,
            "streams": {
                name: int((seed + offset) % (2**63 - 1))
                for name, offset in RNG_STREAMS.items()
            },
            "data_sampler": "separate deterministic epoch-local generator",
        },
        "code_sha256": code_hashes,
    }
    path = Path(run_dir) / "table2_training_identity.json"
    write_json_atomic(path, identity)
    return path


class Table2TrainingAuditCallback(Callback):
    """Record observed epoch samples, optimizer updates, runtime, and memory."""

    def __init__(self, output_path: str | Path):
        super().__init__()
        self.output_path = Path(output_path)
        self.epochs: list[dict[str, Any]] = []
        self._fit_started = None
        self._epoch_started = None
        self._epoch_examples = 0
        self._epoch_batches = 0
        self._optimizer_hook_steps = 0
        self._epoch_start_global_step = 0

    def on_fit_start(self, trainer, pl_module):
        self._fit_started = time.perf_counter()

    def on_train_epoch_start(self, trainer, pl_module):
        import torch

        self._epoch_started = time.perf_counter()
        self._epoch_examples = 0
        self._epoch_batches = 0
        self._optimizer_hook_steps = 0
        self._epoch_start_global_step = int(trainer.global_step)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        actions = batch.get("action") if isinstance(batch, Mapping) else None
        if actions is not None and hasattr(actions, "shape"):
            self._epoch_examples += int(actions.shape[0])
        self._epoch_batches += 1

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self._optimizer_hook_steps += 1

    def on_train_epoch_end(self, trainer, pl_module):
        import torch

        elapsed = time.perf_counter() - float(self._epoch_started or time.perf_counter())
        global_step_delta = int(trainer.global_step) - self._epoch_start_global_step
        peak_allocated = None
        peak_reserved = None
        if torch.cuda.is_available():
            peak_allocated = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
        record = {
            "epoch": int(trainer.current_epoch) + 1,
            "training_samples": int(self._epoch_examples),
            "batches": int(self._epoch_batches),
            "optimizer_steps_global_step_delta": int(global_step_delta),
            "optimizer_steps_before_step_hook": int(self._optimizer_hook_steps),
            "wall_seconds": float(elapsed),
            "gpu_hours_wallclock": float(elapsed / 3600.0)
            if torch.cuda.is_available()
            else 0.0,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
        }
        self.epochs.append(record)
        self._write_progress(trainer, complete=False)

    def on_fit_end(self, trainer, pl_module):
        self._write_progress(trainer, complete=True)

    def _write_progress(self, trainer, *, complete: bool):
        elapsed = (
            None
            if self._fit_started is None
            else float(time.perf_counter() - self._fit_started)
        )
        total_gpu_hours = (
            sum(item["gpu_hours_wallclock"] for item in self.epochs)
            if self.epochs
            else 0.0
        )
        write_json_atomic(
            self.output_path,
            {
                "schema_version": 1,
                "status": "complete" if complete else "running",
                "global_step": int(trainer.global_step),
                "fit_wall_seconds": elapsed,
                "gpu_hours_wallclock": float(total_gpu_hours),
                "epochs": list(self.epochs),
            },
        )


__all__ = [
    "ARMS",
    "RNG_STREAMS",
    "SEEDS",
    "TASKS",
    "Table2TrainingAuditCallback",
    "arm_config",
    "matrix_rows",
    "run_id",
    "sha256_file",
    "sha256_state_dict",
    "sha256_value",
    "write_training_identity",
    "write_json_atomic",
]
