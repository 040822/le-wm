#!/usr/bin/env python3
"""Prepare, execute, time, and summarize the frozen CVPR Table 1 matrix."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import copy
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.cvpr_table1 import (
    BOUND_MODES,
    SEEDS,
    TASKS,
    all_methods,
    attach_cem_archive_callback,
    evaluation_cells,
    identity_hash,
    install_timing_cem,
    matrix_counts,
    prioritized_cells,
    timing_conditions,
)


CONFIG_PATH = ROOT / "config/cvpr/table1.json"
TRAINING_MANIFEST_PATH = ROOT / "config/round5/phase1_6.json"
LEFLOW_MANIFEST_PATH = ROOT / "config/round4/leflow_artifacts.json"
SUBJEPA_MANIFEST_PATH = ROOT / "config/baselines/subjepa_official.json"
REPORT_NAME = "cvpr_table1_report.md"
SOURCE_SNAPSHOT_PATHS = (
    "config/cvpr/table1.json",
    "config/eval/tworoom.yaml",
    "config/eval/pusht.yaml",
    "config/eval/reacher.yaml",
    "config/eval/cube.yaml",
    "config/eval/launcher/local.yaml",
    "config/eval/solver/cem.yaml",
    "config/eval/solver/latent_flow.yaml",
    "source/common/cvpr_table1.py",
    "source/common/checkpoint.py",
    "source/common/eval.py",
    "source/common/remap.py",
    "source/common/round3_eval.py",
    "source/common/round4_eval.py",
    "source/common/round3_phase1.py",
    "source/common/round4_action_bounds.py",
    "source/policy/dispatch.py",
    "source/model/fast_lewam/jepa.py",
    "source/model/leflow/latent_planner.py",
    "source/model/subjepa/official.py",
    "source/policy/fast_lewam_eval.py",
    "source/policy/leflow.py",
    "source/policy/round4.py",
    "scripts/cvpr_table1.py",
)


def _jsonable(value: Any):
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    return value


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _read_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path, *, chunk_size=8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _output_root(args=None) -> Path:
    configured = _repo_path(_load_config()["outputs"]["root"])
    if args is not None and getattr(args, "root", None):
        configured = Path(args.root).expanduser().resolve()
    return configured


def _free_disk_gib(path: Path = ROOT) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def _copy_independent(source: Path, target: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink():
            raise RuntimeError(f"archive target must not be a symlink: {target}")
        if _sha256(source) != _sha256(target):
            raise FileExistsError(f"archive target exists with different content: {target}")
    else:
        shutil.copy2(source, target)
    if target.is_symlink() or os.path.samefile(source, target):
        raise RuntimeError(f"archive is not an independent file copy: {target}")
    source_hash = _sha256(source)
    target_hash = _sha256(target)
    if source_hash != target_hash:
        raise IOError(f"SHA256 mismatch after copying {source} to {target}")
    return {
        "source": str(source),
        "archive": str(target),
        "bytes": target.stat().st_size,
        "source_sha256": source_hash,
        "archive_sha256": target_hash,
    }


def _relative_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (manifest_path.parent / path).resolve()


def _asset_source_map() -> dict[str, Any]:
    training = _read_json(TRAINING_MANIFEST_PATH)
    leflow = _read_json(LEFLOW_MANIFEST_PATH)
    subjepa = _read_json(SUBJEPA_MANIFEST_PATH)
    mapped = {}
    for task in TASKS:
        cowm_checkpoint = _repo_path(training["training"]["checkpoints"][task]).resolve()
        lewm_folder = ROOT / "data/checkpoints" / "quentinll" / {
            "tworoom": "lewm-tworooms",
            "pusht": "lewm-pusht",
            "reacher": "lewm-reacher",
            "cube": "lewm-cube",
        }[task]
        leflow_item = leflow["tasks"][task]
        leflow_checkpoint = _relative_manifest_path(
            LEFLOW_MANIFEST_PATH, leflow_item["planner_checkpoint"]
        )
        leflow_config = _relative_manifest_path(
            LEFLOW_MANIFEST_PATH, leflow_item["config"]
        )
        subjepa_folder = ROOT / subjepa["output_root"] / task
        mapped[task] = {
            "cowm_checkpoint": cowm_checkpoint,
            "cowm_run_config": cowm_checkpoint.parent.parent / "config.yaml",
            "lewm_folder": lewm_folder,
            "leflow_checkpoint": leflow_checkpoint,
            "leflow_config": leflow_config,
            "subjepa_folder": subjepa_folder,
        }
    return mapped


def _copy_directory_files(source: Path, target: Path, names: tuple[str, ...]):
    records = []
    for name in names:
        item = source / name
        if item.is_file():
            records.append(_copy_independent(item, target / name))
    return records


def _prepare_assets(root: Path) -> dict[str, Any]:
    assets_dir = root / "assets"
    manifest_path = root / "provenance/assets_manifest.json"
    existing = _read_json(manifest_path)
    if existing is not None:
        for record in existing["files"]:
            archived = Path(record["archive"])
            if not archived.is_file() or _sha256(archived) != record["archive_sha256"]:
                raise RuntimeError(f"previously frozen asset has changed or is missing: {archived}")
        _verify_asset_manifest(existing, root)
        return existing

    records = []
    by_task = {}
    for task, source in _asset_source_map().items():
        task_records = {}
        cowm_dest = assets_dir / "cowm" / task
        task_records["cowm_checkpoint"] = _copy_independent(
            source["cowm_checkpoint"],
            cowm_dest / "checkpoints" / source["cowm_checkpoint"].name,
        )
        task_records["cowm_config"] = _copy_independent(
            source["cowm_run_config"], cowm_dest / "config.yaml"
        )
        lewm_dest = assets_dir / "lewm" / task
        task_records["lewm"] = _copy_directory_files(
            source["lewm_folder"], lewm_dest, ("weights.pt", "config.json", "README.md")
        )
        leflow_dest = assets_dir / "leflow" / task
        task_records["leflow_checkpoint"] = _copy_independent(
            source["leflow_checkpoint"], leflow_dest / "latent_planner.pt"
        )
        task_records["leflow_config"] = _copy_independent(
            source["leflow_config"], leflow_dest / "latent_planner_config.yaml"
        )
        subjepa_source = source["subjepa_folder"]
        subjepa_dest = assets_dir / "subjepa" / task
        task_records["subjepa"] = _copy_directory_files(
            subjepa_source,
            subjepa_dest,
            ("subjepa.pt", "manifest.json", "model_config.json"),
        )
        task_records["subjepa_raw"] = _copy_directory_files(
            subjepa_source / "raw",
            subjepa_dest / "raw",
            ("config.yml", f"{task}_subjepa_object.ckpt"),
        )
        records.extend([task_records["cowm_checkpoint"], task_records["cowm_config"]])
        records.extend(task_records["lewm"])
        records.extend([task_records["leflow_checkpoint"], task_records["leflow_config"]])
        records.extend(task_records["subjepa"])
        records.extend(task_records["subjepa_raw"])
        by_task[task] = task_records

    raw_source = ROOT / "data/checkpoints/subjepa_official/raw_source"
    raw_source_records = _copy_directory_files(
        raw_source,
        assets_dir / "subjepa/raw_source",
        ("jepa.py", "module.py", "LICENSE"),
    )
    records.extend(raw_source_records)
    result = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": records,
        "tasks": by_task,
        "subjepa_source_files": raw_source_records,
    }
    _write_json(manifest_path, result)
    _verify_asset_manifest(result, root)
    return result


def _verify_asset_manifest(asset_manifest: Mapping[str, Any], root: Path) -> None:
    """Check copied weights against the declared upstream and conversion hashes."""
    training = _read_json(TRAINING_MANIFEST_PATH)
    leflow = _read_json(LEFLOW_MANIFEST_PATH)
    subjepa = _read_json(SUBJEPA_MANIFEST_PATH)
    records = {str(record["archive"]): record for record in asset_manifest["files"]}
    expected: dict[Path, str] = {}
    for task in TASKS:
        cowm = _repo_path(training["training"]["checkpoints"][task]).resolve()
        expected[root / "assets/cowm" / task / "checkpoints" / cowm.name] = training["training"]["checkpoint_sha256"][task]
        flow = leflow["tasks"][task]
        expected[root / "assets/leflow" / task / "latent_planner.pt"] = flow["planner_checkpoint_sha256"]
        expected[root / "assets/leflow" / task / "latent_planner_config.yaml"] = flow["config_sha256"]
        expected[root / "assets/lewm" / task / "weights.pt"] = flow["lewm_checkpoint_sha256"]
        official_task = subjepa["tasks"][task]
        subjepa_manifest = _read_json(ROOT / subjepa["output_root"] / task / "manifest.json")
        expected[root / "assets/subjepa" / task / "raw" / f"{task}_subjepa_object.ckpt"] = official_task["weights_sha256"]
        expected[root / "assets/subjepa" / task / "subjepa.pt"] = subjepa_manifest["converted_sha256"]
    for name, digest in subjepa["source_files"].items():
        expected[root / "assets/subjepa/raw_source" / name] = digest
    for target, digest in expected.items():
        record = records.get(str(target.resolve()))
        if record is None:
            raise RuntimeError(f"frozen archive manifest is missing an expected weight: {target}")
        if record["source_sha256"] != digest or record["archive_sha256"] != digest:
            raise RuntimeError(
                f"declared model hash mismatch for {target}: expected {digest}, "
                f"source={record['source_sha256']} archive={record['archive_sha256']}"
            )


def _dataset_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    return {
        task: _repo_path(path).resolve()
        for task, path in config["resources"]["input_datasets"].items()
    }


def _hash_input_datasets(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    target = root / "provenance/input_datasets.json"
    previous = _read_json(target)
    paths = _dataset_paths(config)
    if previous is not None:
        for task, record in previous["tasks"].items():
            path = paths[task]
            if not path.is_file() or path.stat().st_size != record["bytes"]:
                raise RuntimeError(f"frozen input dataset changed: {path}")
            if _sha256(path) != record["sha256"]:
                raise RuntimeError(f"frozen input dataset SHA256 changed: {path}")
        return previous
    rows = {}
    for task, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required CVPR dataset is missing: {path}")
        rows[task] = {
            "configured_path": config["resources"]["input_datasets"][task],
            "resolved_path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "copied_to_archive": False,
        }
    result = {"schema_version": 1, "tasks": rows}
    _write_json(target, result)
    return result


def _snapshot_provenance(root: Path, config: Mapping[str, Any]) -> None:
    provenance = root / "provenance"
    source_hashes = _current_source_hashes()
    revision_id = _code_revision_id(source_hashes)
    snapshot = provenance / "code_snapshots" / revision_id
    for relative, digest in source_hashes.items():
        copied = _copy_independent(ROOT / relative, snapshot / relative)
        if copied["source_sha256"] != digest:
            raise RuntimeError(f"source changed while creating code snapshot: {relative}")
    try:
        status = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain=v1", "--branch"],
            check=True,
            capture_output=True,
        ).stdout
        diff = subprocess.run(
            ["git", "-C", str(ROOT), "diff", "--binary", "HEAD"],
            check=True,
            capture_output=True,
        ).stdout
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("could not freeze the active worktree provenance") from exc
    revision_dir = provenance / "source_revisions" / revision_id
    revision_dir.mkdir(parents=True, exist_ok=True)
    (revision_dir / "git_status.txt").write_bytes(status)
    (revision_dir / "uncommitted_tracked_diff.patch").write_bytes(diff)
    packages = (
        "numpy",
        "torch",
        "stable-worldmodel",
        "stable-pretraining",
        "transformers",
        "hydra-core",
        "gymnasium",
        "h5py",
        "opencv-python",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    environment_path = revision_dir / "execution_environment.json"
    _write_json(
        environment_path,
        {
            "git_commit": commit,
            "python": sys.version,
            "platform": platform.platform(),
            "platform_sys": sys.platform,
            "cpu_model": platform.processor(),
            "cpu_count": os.cpu_count(),
            "packages": versions,
            "cpu_threads": config["timing"]["cpu_threads"],
            "configuration_sha256": hashlib.sha256(_json_bytes(config)).hexdigest(),
            "entrypoint": str(ROOT / "scripts/cvpr_table1.py"),
        },
    )
    _write_json(
        provenance / "active_source_revision.json",
        {
            "revision_id": revision_id,
            "code_snapshot_dir": str(snapshot.resolve()),
            "source_hashes": source_hashes,
            "git_status_path": str((revision_dir / "git_status.txt").resolve()),
            "uncommitted_tracked_diff_path": str(
                (revision_dir / "uncommitted_tracked_diff.patch").resolve()
            ),
            "execution_environment_path": str(environment_path.resolve()),
            "execution_environment_sha256": _sha256(environment_path),
        },
    )


def _current_source_hashes() -> dict[str, str]:
    source_hashes = {}
    for relative in SOURCE_SNAPSHOT_PATHS:
        source = ROOT / relative
        if source.is_file():
            source_hashes[relative] = _sha256(source)
    return source_hashes


def _code_revision_id(source_hashes: Mapping[str, str]) -> str:
    return hashlib.sha256(_json_bytes(source_hashes)).hexdigest()[:20]


def _build_manifest_set(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    from source.common.eval import compose_eval_config, get_dataset
    from source.common.round3_phase1 import build_legacy_manifest

    paths = _dataset_paths(config)
    overlap = {}
    cohort_hashes = {}
    for task in TASKS:
        eval_cfg = compose_eval_config(
            task,
            [
                f"eval.dataset_name={paths[task]}",
                f"cache_dir={ROOT / 'data'}",
                "eval.num_eval=50",
            ],
        )
        dataset = get_dataset(eval_cfg, eval_cfg.eval.dataset_name)
        previous_keys = {}
        task_hashes = {}
        for seed in SEEDS:
            manifest = build_legacy_manifest(
                dataset,
                task=task,
                seed=seed,
                goal_offset_steps=25,
                num_eval=50,
            )
            manifest_path = root / "cohorts" / task / f"seed_{seed}.json"
            if manifest_path.exists():
                from source.common.round3_phase1 import CohortManifest

                saved = CohortManifest.load(manifest_path)
                if saved.computed_sha256 != manifest.computed_sha256:
                    raise RuntimeError(
                        f"cohort changed after freeze: {manifest_path}"
                    )
            else:
                manifest.save(manifest_path)
            keys = [
                json.dumps([entry.episode_id, entry.start_step], sort_keys=True)
                for entry in manifest.entries
            ]
            current = set(keys)
            previous_keys[seed] = current
            task_hashes[str(seed)] = manifest.computed_sha256
            task_hashes.setdefault("diagnostics", {})[str(seed)] = dict(
                manifest.diagnostics
            )
        overlap[task] = {
            f"{left}_vs_{right}": len(previous_keys[left] & previous_keys[right])
            for index, left in enumerate(SEEDS)
            for right in SEEDS[index + 1 :]
        }
        cohort_hashes[task] = task_hashes
    result = {
        "protocol_variant": "legacy",
        "sampling_rule": "source.common.eval.select_eval_cohort exact compatibility arithmetic",
        "task_seed_cohort_sha256": cohort_hashes,
        "within_seed_duplicate_start_count": 0,
        "cross_seed_start_overlap_counts": overlap,
        "initial_successes_retained": True,
        "same_trajectory_different_start_allowed": True,
    }
    _write_json(root / "cohorts/cohort_diagnostics.json", result)
    return result


def command_matrix(args) -> None:
    cells = evaluation_cells()
    result = {
        **matrix_counts(),
        "methods": all_methods(),
        "sample_seeds": list(SEEDS),
        "tasks": list(TASKS),
        "timing_conditions": len(timing_conditions()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_prepare(args) -> None:
    root = _output_root(args)
    root.mkdir(parents=True, exist_ok=True)
    config = _load_config()
    previous_freeze = _read_json(root / "freeze.json", {})
    previous_revision = _read_json(root / "provenance/active_source_revision.json")
    if (
        previous_freeze.get("formal_evaluations_started")
        or previous_freeze.get("timing_started")
    ) and previous_revision:
        if _code_revision_id(_current_source_hashes()) != previous_revision.get("revision_id"):
            raise RuntimeError(
                "source code changed after formal work started; preserving the frozen experiment record"
            )
    frozen_path = root / "frozen_config.json"
    if frozen_path.exists():
        existing = _read_json(frozen_path)
        if existing != config:
            raise RuntimeError(f"refusing to change frozen protocol at {frozen_path}")
    else:
        _write_json(frozen_path, config)
    _copy_independent(CONFIG_PATH, root / "provenance/config/table1.json")
    for manifest in (TRAINING_MANIFEST_PATH, LEFLOW_MANIFEST_PATH, SUBJEPA_MANIFEST_PATH):
        _copy_independent(
            manifest,
            root / "provenance/config" / manifest.name,
        )
    assets = _prepare_assets(root)
    input_hashes = _hash_input_datasets(root, config)
    cohorts = _build_manifest_set(root, config)
    _snapshot_provenance(root, config)
    cells = evaluation_cells()
    status_path = root / "status.json"
    if not status_path.exists():
        _write_json(
            status_path,
            {
                "schema_version": 1,
                "cells": {cell["cell_id"]: {"status": "missing"} for cell in cells},
            },
        )
    _write_json(
        root / "freeze.json",
        {
            "status": "frozen",
            "configuration_sha256": hashlib.sha256(_json_bytes(config)).hexdigest(),
            "dataset_hashes_frozen": True,
            "cohorts_frozen": True,
            "assets_frozen": True,
            "matrix": matrix_counts(),
            "timing_conditions": len(timing_conditions()),
            "asset_file_count": len(assets["files"]),
            "dataset_sha256": {
                task: record["sha256"] for task, record in input_hashes["tasks"].items()
            },
            "cohort_manifest_count": len(TASKS) * len(SEEDS),
            "cross_seed_overlap_recorded": bool(cohorts["cross_seed_start_overlap_counts"]),
            "source_revision_id": _read_json(
                root / "provenance/active_source_revision.json"
            )["revision_id"],
            "formal_evaluations_started": bool(previous_freeze.get("formal_evaluations_started", False)),
            "formal_evaluations_started_utc": previous_freeze.get("formal_evaluations_started_utc"),
            "timing_started": bool(previous_freeze.get("timing_started", False)),
            "timing_started_utc": previous_freeze.get("timing_started_utc"),
        },
    )
    print(json.dumps({"root": str(root), "assets": len(assets["files"]), **matrix_counts()}))


def command_snapshot_refresh(args) -> None:
    """Capture a new immutable code revision without rescanning frozen data."""
    root = _output_root(args)
    required = (
        root / "provenance/input_datasets.json",
        root / "cohorts/cohort_diagnostics.json",
        root / "provenance/assets_manifest.json",
        root / "freeze.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot refresh source snapshot before preparation is complete: {missing}")
    for task in TASKS:
        for seed in SEEDS:
            cohort = root / "cohorts" / task / f"seed_{seed}.json"
            if not cohort.is_file():
                raise RuntimeError(f"cannot refresh source snapshot; cohort is missing: {cohort}")
    config = _load_config()
    _snapshot_provenance(root, config)
    revision = _read_json(root / "provenance/active_source_revision.json")
    freeze = _read_json(root / "freeze.json")
    freeze["source_revision_id"] = revision["revision_id"]
    freeze["source_snapshot_refreshed_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    _write_json(root / "freeze.json", freeze)
    print(json.dumps({"root": str(root), "source_revision_id": revision["revision_id"]}, indent=2))


def _initialize_status(root: Path) -> Path:
    path = root / "status.json"
    if path.exists():
        return path
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        path,
        {
            "schema_version": 1,
            "cells": {
                cell["cell_id"]: {"status": "missing"}
                for cell in evaluation_cells()
            },
        },
    )
    return path


@contextmanager
def _exclusive_lock(path: Path, *, blocking: bool = True):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(stream.fileno(), flags)
        except BlockingIOError as exc:
            raise RuntimeError(f"lock is already held: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _mutate_status(root: Path, cell_id: str, patch: Mapping[str, Any]) -> None:
    path = _initialize_status(root)
    with _exclusive_lock(root / ".locks/status.lock"):
        state = _read_json(path)
        cells = state.setdefault("cells", {})
        current = dict(cells.get(cell_id, {"status": "missing"}))
        current.update(dict(patch))
        current["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cells[cell_id] = current
        _write_json(path, state)


def _mark_freeze_activity(root: Path, *, key: str) -> None:
    freeze_path = root / "freeze.json"
    with _exclusive_lock(root / ".locks/freeze.lock"):
        freeze = _read_json(freeze_path, {})
        freeze[key] = True
        freeze.setdefault(f"{key}_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        _write_json(freeze_path, freeze)


def _asset_record_index(asset_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(record["archive"]): record for record in asset_manifest.get("files", [])}


def _cell_identity(root: Path, cell: Mapping[str, Any]) -> dict[str, Any]:
    assets = _read_json(root / "provenance/assets_manifest.json")
    if not assets:
        raise RuntimeError("assets are not archived; run the archive-assets command first")
    record_index = _asset_record_index(assets)
    task = str(cell["task"])
    family = str(cell["family"])
    if family == "cowm":
        required = [
            root / "assets/cowm" / task / "config.yaml",
            *sorted((root / "assets/cowm" / task / "checkpoints").glob("*.pt")),
        ]
    elif family in {"lewm", "leflow"}:
        required = [
            root / "assets/lewm" / task / "weights.pt",
            root / "assets/lewm" / task / "config.json",
        ]
        if family == "leflow":
            required.extend(
                [
                    root / "assets/leflow" / task / "latent_planner.pt",
                    root / "assets/leflow" / task / "latent_planner_config.yaml",
                ]
            )
    elif family == "subjepa":
        required = [root / "assets/subjepa" / task / "subjepa.pt"]
    else:
        raise ValueError(f"unknown model family: {family}")
    asset_hashes = []
    for path in required:
        archive_path = str(path.resolve())
        record = record_index.get(archive_path)
        if record is None or not path.is_file():
            raise RuntimeError(f"required archived asset is missing from manifest: {path}")
        if _sha256(path) != record["archive_sha256"]:
            raise RuntimeError(f"archived asset hash no longer matches manifest: {path}")
        asset_hashes.append(
            {"path": archive_path, "sha256": record["archive_sha256"]}
        )
    config = _load_config()
    cohort_path = root / "cohorts" / task / f"seed_{int(cell['evaluation_seed'])}.json"
    if not cohort_path.is_file():
        raise RuntimeError(f"cohort manifest is missing: {cohort_path}")
    cohort_hash = _sha256(cohort_path)
    source_revision = _read_json(root / "provenance/active_source_revision.json")
    if not source_revision:
        raise RuntimeError("source code provenance is not frozen; run prepare first")
    code_snapshot = Path(source_revision["code_snapshot_dir"])
    source_hashes = source_revision["source_hashes"]
    for relative, expected_hash in source_hashes.items():
        current = ROOT / relative
        snapshot = code_snapshot / relative
        if not current.is_file() or _sha256(current) != expected_hash:
            raise RuntimeError(
                f"active source differs from frozen snapshot for {relative}; rerun prepare before dispatch"
            )
        if not snapshot.is_file() or _sha256(snapshot) != expected_hash:
            raise RuntimeError(f"frozen code snapshot is missing or changed: {relative}")
    environment_path = Path(source_revision["execution_environment_path"])
    if _sha256(environment_path) != source_revision["execution_environment_sha256"]:
        raise RuntimeError("frozen execution-environment record changed")
    return {
        "cell": dict(cell),
        "configuration_sha256": hashlib.sha256(_json_bytes(config)).hexdigest(),
        "asset_sha256": asset_hashes,
        "cohort_file_sha256": cohort_hash,
        "source_revision_id": source_revision["revision_id"],
        "code_snapshot_sha256": source_hashes,
        "execution_environment_sha256": source_revision["execution_environment_sha256"],
    }


def _result_is_complete(attempt_dir: Path, identity_sha256: str) -> bool:
    result_path = attempt_dir / "result.json"
    trace_path = attempt_dir / "episodes.jsonl"
    if not result_path.is_file() or not trace_path.is_file():
        return False
    try:
        result = _read_json(result_path)
    except (OSError, json.JSONDecodeError):
        return False
    table_identity = result.get("cvpr_table1", {})
    episodes = result.get("episodes", [])
    return (
        result.get("status") == "ok"
        and len(episodes) == 50
        and table_identity.get("identity_sha256") == identity_sha256
        and table_identity.get("trace_sha256") == _sha256(trace_path)
        and math.isfinite(float(result.get("success_rate", float("nan"))))
    )


def _next_attempt_dir(root: Path, cell_id: str) -> tuple[int, Path]:
    parent = root / "runs" / cell_id
    parent.mkdir(parents=True, exist_ok=True)
    existing = []
    for candidate in parent.glob("attempt_*"):
        try:
            existing.append(int(candidate.name.removeprefix("attempt_")))
        except ValueError:
            continue
    attempt = max(existing, default=0) + 1
    return attempt, parent / f"attempt_{attempt:03d}"


def _check_dispatch_resources(gpu: int, config: Mapping[str, Any]) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.strip() != str(gpu):
        raise RuntimeError(
            f"worker for physical GPU {gpu} requires CUDA_VISIBLE_DEVICES={gpu!s}, got {visible!r}"
        )
    permitted = set(int(value) for value in config["resources"]["permitted_gpus"])
    if gpu not in permitted or gpu not in {0, 1, 2, 3}:
        raise RuntimeError(f"physical GPU {gpu} is not permitted by the frozen plan")
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is required for the GPU preflight check")
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.free,memory.total,utilization.gpu,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    fields = [field.strip() for field in query.split(",")]
    if len(fields) != 6:
        raise RuntimeError(f"unexpected nvidia-smi preflight response: {query!r}")
    free_mib, total_mib, utilization = (int(fields[index]) for index in range(3))
    free_gib = free_mib / 1024
    required_gib = float(config["resources"]["evaluation_free_vram_margin_gib"])
    if free_gib < required_gib:
        raise RuntimeError(
            f"GPU {gpu} has {free_gib:.2f} GiB free; frozen margin is {required_gib:.2f} GiB"
        )
    disk_free_gib = _free_disk_gib(ROOT)
    minimum_disk_gib = float(config["resources"]["stop_dispatch_free_disk_gib"])
    if disk_free_gib < minimum_disk_gib:
        raise RuntimeError(
            f"only {disk_free_gib:.2f} GiB disk remains; stop-dispatch floor is {minimum_disk_gib:.2f} GiB"
        )
    return {
        "gpu": gpu,
        "gpu_uuid": fields[3],
        "gpu_name": fields[4],
        "driver_version": fields[5],
        "free_vram_gib": free_gib,
        "total_vram_gib": total_mib / 1024,
        "gpu_utilization_percent": utilization,
        "free_disk_gib": disk_free_gib,
    }


def _load_models(root: Path, task: str, phase: int, device: str) -> dict[str, Any]:
    from source.common.checkpoint import load_policy_or_model

    models: dict[str, Any] = {}
    assets = root / "assets"
    if phase == 1:
        cowm_path = next((assets / "cowm" / task / "checkpoints").glob("*.pt"))
        cowm, _resolved = load_policy_or_model(str(cowm_path), cache_dir=str(ROOT / "data"))
        models["cowm"] = getattr(cowm, "model", cowm)

        from source.common.remap import load_pretrained_remapped

        lewm_path = assets / "lewm" / task
        lewm = load_pretrained_remapped(str(lewm_path), cache_dir=str(ROOT / "data"))
        models["lewm"] = lewm.eval().requires_grad_(False)

        from source.model.leflow.latent_planner import LatentPlannerRuntime

        models["leflow"] = LatentPlannerRuntime.from_checkpoint(
            assets / "leflow" / task / "latent_planner.pt",
            device="cpu",
            lewm_model=lewm,
        )
    elif phase == 2:
        subjepa_path = assets / "subjepa" / task / "subjepa.pt"
        subjepa, _resolved = load_policy_or_model(
            str(subjepa_path), cache_dir=str(ROOT / "data")
        )
        models["subjepa"] = getattr(subjepa, "model", subjepa)
    else:
        raise ValueError(f"unsupported evaluation phase: {phase}")
    del device
    return models


def _cell_config(cell: Mapping[str, Any]):
    from source.common.eval import compose_eval_config

    config = _load_config()
    task = str(cell["task"])
    dataset_path = _dataset_paths(config)[task]
    cfg = compose_eval_config(task)
    cfg.eval.dataset_name = str(dataset_path)
    cfg.eval.benchmark_dataset_name = str(config["resources"]["input_datasets"][task])
    cfg.eval.num_eval = int(config["protocol"]["episodes_per_seed"])
    cfg.eval.goal_offset_steps = int(config["protocol"]["goal_offset_steps"])
    cfg.eval.eval_budget = int(config["protocol"]["eval_budget"])
    cfg.eval.policy_seed = int(cell["policy_seed"])
    cfg.seed = int(cell["environment_seed"])
    cfg.world.num_envs = int(config["protocol"]["world_batch_size"])
    cfg.world.max_episode_steps = 2 * int(config["protocol"]["eval_budget"])
    cfg.plan_config.horizon = int(config["protocol"]["horizon"])
    cfg.plan_config.receding_horizon = int(config["protocol"]["receding_horizon"])
    cfg.plan_config.action_block = int(config["protocol"]["action_block"])
    cfg.solver.num_samples = int(config["protocol"]["cem"]["num_samples"])
    cfg.solver.n_steps = int(config["protocol"]["cem"]["iterations"])
    cfg.solver.topk = int(config["protocol"]["cem"]["topk"])
    cfg.solver.var_scale = float(config["protocol"]["cem"]["var_scale"])
    cfg.solver.batch_size = int(config["protocol"]["cem"]["batch_size"])
    cfg.solver.seed = int(cell["policy_seed"])
    cfg.solver.device = "cuda:0"
    cfg.output.save_video = int(cell["evaluation_seed"]) == 42
    return cfg


def _run_one_cell(
    *, root: Path, cell: Mapping[str, Any], attempt_dir: Path,
    identity: Mapping[str, Any], identity_sha256: str, gpu: int,
    models: Mapping[str, Any], dataset: Any,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from source.common.eval import EvaluationIdentity
    from source.common.round3_phase1 import CohortManifest
    from source.common.round3_eval import run_round3_evaluation
    from source.common.round4_eval import run_round4_evaluation

    task = str(cell["task"])
    cfg = _cell_config(cell)
    cohort_path = root / "cohorts" / task / f"seed_{int(cell['evaluation_seed'])}.json"
    manifest = CohortManifest.load(cohort_path)
    _write_json(attempt_dir / "cell_identity.json", {
        "identity": dict(identity), "identity_sha256": identity_sha256,
        "attempt": attempt_dir.name,
    })
    OmegaConf.save(cfg, attempt_dir / "resolved_config.yaml", resolve=True)
    _write_json(
        attempt_dir / "command.json",
        {
            "argv": list(sys.argv),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "working_directory": str(ROOT),
            "preflight": _check_dispatch_resources(gpu, _load_config()),
        },
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    torch.manual_seed(int(cell["policy_seed"]))
    np.random.seed(int(cell["policy_seed"]) % (2**32 - 1))

    family = str(cell["family"])
    model = models[family]
    identity_record = EvaluationIdentity(
        entrypoint="cvpr_table1",
        policy_kind={
            "cowm": "fast_lewam",
            "lewm": "lewm",
            "leflow": "leflow",
            "subjepa": "subjepa_official",
        }[family],
        checkpoint=next(
            (item["path"] for item in identity["asset_sha256"] if item["path"].endswith((".pt", "/weights.pt"))),
            None,
        ),
        epoch=10 if family == "cowm" else None,
        stage="stage_b" if family == "cowm" else None,
        guidance_mode=str(cell.get("guidance_mode", "none")),
        guidance_step_size=float(cell.get("guidance_step_size", _load_config()["protocol"]["cowm"]["guidance_step_size"])),
        guidance_last_steps=int(cell.get("guidance_last_steps") or 2),
        guidance_inner_steps=int(cell.get("guidance_inner_steps") or 1),
        guidance_max_rms_offset=float(cell.get("guidance_max_rms_offset", _load_config()["protocol"]["cowm"]["guidance_max_rms_offset"])),
    )

    batch_started = time.perf_counter()
    if family == "cowm":
        result = run_round4_evaluation(
            cfg, task=task, policy_or_model=model, mode=str(cell["mode"]),
            identity=identity_record, manifest=manifest,
            output_dir=attempt_dir, trace_output_dir=attempt_dir,
            device="cuda:0", trace=True, candidate_count=int(_load_config()["protocol"]["cowm"]["candidate_count"]),
            flow_steps=int(_load_config()["protocol"]["cowm"]["action_flow_steps"]),
            action_flow_steps=cell.get("action_flow_steps"),
            solver_batch_size=int(_load_config()["protocol"]["cowm"]["solver_batch_size"]),
            candidate_batch_size=int(_load_config()["protocol"]["cowm"]["candidate_batch_size"]),
            action_flow_integrator=str(_load_config()["protocol"]["cowm"]["integrator"]),
            action_bound_mode=str(cell["action_bound_mode"]),
            cem_protocol=str(cell["cem_protocol"]),
            guidance_mode=str(cell["guidance_mode"]),
            guidance_step_size=float(cell.get("guidance_step_size", _load_config()["protocol"]["cowm"]["guidance_step_size"])),
            guidance_last_steps=int(cell.get("guidance_last_steps") or 2),
            guidance_inner_steps=int(cell.get("guidance_inner_steps") or 1),
            guidance_max_rms_offset=float(cell.get("guidance_max_rms_offset", _load_config()["protocol"]["cowm"]["guidance_max_rms_offset"])),
            proposal_chunk_size=int(_load_config()["protocol"]["cowm"]["proposal_chunk_size"]),
            allowed_protocol_variants=("legacy",),
            allow_solver_config_override=True,
            allow_evaluation_seed_override=True,
            allow_cohort_seed_mismatch=True,
            policy_seed=int(cell["policy_seed"]),
            execute_steps=cell.get("execute_steps"),
            score_horizon_blocks=cell.get("score_horizon_blocks"),
            score_reduction=str(cell.get("score_reduction", "endpoint")),
            terminal_weight=float(cell.get("terminal_weight", 1.0)),
            video_slots=2,
            dataset=dataset,
        )
    else:
        result = run_round3_evaluation(
            cfg, task=task, policy_or_model=model, identity=identity_record,
            manifest=manifest, output_dir=attempt_dir,
            trace_output_dir=attempt_dir, device="cuda:0", trace=True,
            allow_solver_budget_overrides=False,
            allow_evaluation_seed_override=True,
            allow_cohort_seed_mismatch=True,
            video_slots=2, dataset=dataset,
        )
    batch_wall_seconds = time.perf_counter() - batch_started
    result_path = attempt_dir / "result.json"
    if not result_path.is_file():
        raise RuntimeError(f"evaluation did not publish result.json: {result_path}")
    published = _read_json(result_path)
    trace_path = attempt_dir / "episodes.jsonl"
    if published.get("status") != "ok" or len(published.get("episodes", [])) != 50:
        raise RuntimeError("evaluation result is not a complete 50-episode success record")
    if not trace_path.is_file():
        raise RuntimeError("evaluation did not publish the expected episode trace")
    published["cvpr_table1"] = {
        "identity_sha256": identity_sha256,
        "cell_id": cell["cell_id"],
        "method_id": cell["method_id"],
        "stage": int(cell["stage"]),
        "attempt": attempt_dir.name,
        "trace_sha256": _sha256(trace_path),
        "preflight": _check_dispatch_resources(gpu, _load_config()),
        "result_return_status": result.get("status", "ok"),
    }
    published["batch_evaluation_wall_seconds"] = float(batch_wall_seconds)
    _write_json(result_path, published)
    return published


def command_archive_assets(args) -> None:
    root = _output_root(args)
    config = _load_config()
    free = _free_disk_gib(ROOT)
    floor = float(config["resources"]["stop_dispatch_free_disk_gib"])
    if free < floor:
        raise RuntimeError(f"only {free:.2f} GiB disk remains; archive stopped below {floor:.2f} GiB")
    frozen_path = root / "frozen_config.json"
    root.mkdir(parents=True, exist_ok=True)
    if frozen_path.exists() and _read_json(frozen_path) != config:
        raise RuntimeError(f"refusing to change frozen protocol at {frozen_path}")
    if not frozen_path.exists():
        _write_json(frozen_path, config)
    _copy_independent(CONFIG_PATH, root / "provenance/config/table1.json")
    for manifest in (TRAINING_MANIFEST_PATH, LEFLOW_MANIFEST_PATH, SUBJEPA_MANIFEST_PATH):
        _copy_independent(manifest, root / "provenance/config" / manifest.name)
    assets = _prepare_assets(root)
    _initialize_status(root)
    print(json.dumps({"root": str(root), "archive_file_count": len(assets["files"]), "free_disk_gib": _free_disk_gib(ROOT)}, indent=2))


def _run_worker(args) -> None:
    root = _output_root(args)
    config = _load_config()
    if not (root / "provenance/assets_manifest.json").is_file():
        raise RuntimeError("formal work requires archived assets; run archive-assets first")
    if not (root / "cohorts/cohort_diagnostics.json").is_file():
        raise RuntimeError("formal work requires frozen cohorts; run prepare first")
    gpu = int(args.gpu)
    task_cells = prioritized_cells(task=str(args.task), phase=int(args.phase))
    if args.cell_id:
        task_cells = [cell for cell in task_cells if cell["cell_id"] == args.cell_id]
        if not task_cells:
            raise ValueError(f"cell does not belong to task/phase selection: {args.cell_id}")
    status_path = _initialize_status(root)
    with _exclusive_lock(root / f".locks/gpu_{gpu}.lock", blocking=False):
        preflight = _check_dispatch_resources(gpu, config)
        _mark_freeze_activity(root, key="formal_evaluations_started")
        print(f"GPU {gpu} worker preflight: {json.dumps(preflight)}", flush=True)
        family_set = {cell["family"] for cell in task_cells}
        model_bank = _load_models(root, str(args.task), int(args.phase), f"cuda:0")
        if set(model_bank) != family_set:
            # Keep the phase's models loaded once, while requiring exact assets
            # for only the requested worker queue.
            missing = family_set - set(model_bank)
            if missing:
                raise RuntimeError(f"model bank is missing method families: {sorted(missing)}")
        cfg_for_dataset = _cell_config(task_cells[0])
        from source.common.eval import get_dataset

        dataset = get_dataset(cfg_for_dataset, cfg_for_dataset.eval.dataset_name)
        for cell in task_cells:
            identity = _cell_identity(root, cell)
            identity_sha256 = identity_hash(identity)
            cell_lock = root / ".locks" / f"cell_{cell['cell_id']}.lock"
            with _exclusive_lock(cell_lock, blocking=False):
                parent = root / "runs" / cell["cell_id"]
                matching = []
                for attempt_dir in sorted(parent.glob("attempt_*")) if parent.exists() else []:
                    if _result_is_complete(attempt_dir, identity_sha256):
                        matching.append(attempt_dir)
                if matching:
                    _mutate_status(root, cell["cell_id"], {
                        "status": "complete", "identity_sha256": identity_sha256,
                        "attempt": matching[-1].name, "skipped_verified": True,
                    })
                    continue
                attempt, attempt_dir = _next_attempt_dir(root, cell["cell_id"])
                _mutate_status(root, cell["cell_id"], {
                    "status": "running", "identity_sha256": identity_sha256,
                    "attempt": attempt_dir.name, "gpu": gpu,
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
                try:
                    attempt_dir.mkdir(parents=True, exist_ok=False)
                    with (attempt_dir / "worker.log").open("w", encoding="utf-8") as log:
                        with redirect_stdout(log), redirect_stderr(log):
                            result = _run_one_cell(
                                root=root, cell=cell, attempt_dir=attempt_dir,
                                identity=identity, identity_sha256=identity_sha256,
                                gpu=gpu, models=model_bank, dataset=dataset,
                            )
                    _mutate_status(root, cell["cell_id"], {
                        "status": "complete", "identity_sha256": identity_sha256,
                        "attempt": attempt_dir.name,
                        "success_rate": result["success_rate"],
                        "result_sha256": _sha256(attempt_dir / "result.json"),
                        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    })
                    print(f"completed {cell['cell_id']}: {result['success_rate']:.4f}", flush=True)
                except BaseException as exc:
                    trace = traceback.format_exc()
                    attempt_dir.mkdir(parents=True, exist_ok=True)
                    _write_json(attempt_dir / "failure.json", {
                        "status": "failed", "error_type": type(exc).__name__,
                        "error": str(exc), "traceback": trace,
                    })
                    _mutate_status(root, cell["cell_id"], {
                        "status": "failed", "identity_sha256": identity_sha256,
                        "attempt": attempt_dir.name, "error_type": type(exc).__name__,
                        "error": str(exc),
                        "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    })
                    print(f"failed {cell['cell_id']}: {exc}", flush=True)
                    if any(
                        marker in str(exc)
                        for marker in (
                            "GiB free; frozen margin",
                            "disk remains; stop-dispatch floor",
                            "nvidia-smi is required",
                            "unexpected nvidia-smi preflight response",
                        )
                    ):
                        print("resource preflight failed; stopping dispatch", flush=True)
                        break
                    if args.stop_on_error:
                        raise


def command_worker(args) -> None:
    _run_worker(args)


def command_status(args) -> None:
    root = _output_root(args)
    status = _read_json(root / "status.json", {"cells": {}})
    counts: dict[str, int] = {}
    for record in status.get("cells", {}).values():
        key = str(record.get("status", "missing"))
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"root": str(root), "counts": counts, "matrix": matrix_counts()}, indent=2))


def _cell_result_path(root: Path, cell: Mapping[str, Any], status: Mapping[str, Any]):
    entry = status.get("cells", {}).get(cell["cell_id"], {})
    if entry.get("status") != "complete" or not entry.get("attempt"):
        return None
    attempt = root / "runs" / str(cell["cell_id"]) / str(entry["attempt"])
    identity = str(entry.get("identity_sha256", ""))
    if not _result_is_complete(attempt, identity):
        return None
    return attempt / "result.json"


def _load_success_result(root: Path, cell: Mapping[str, Any], status: Mapping[str, Any]):
    result_path = _cell_result_path(root, cell, status)
    if result_path is None:
        return None
    result = _read_json(result_path)
    episode_records = result.get("episodes", [])
    return {
        "success_rate": float(result["success_rate"]),
        "result_path": str(result_path),
        "episode_count": len(result["episodes"]),
        "identity_sha256": result["cvpr_table1"]["identity_sha256"],
        "batch_evaluation_wall_seconds": float(
            result.get("batch_evaluation_wall_seconds", result.get("evaluation_seconds", 0.0))
        ),
        "policy_environment_evaluation_seconds": float(result.get("evaluation_seconds", 0.0)),
        "batch_planning_wall_seconds": float(result.get("batch_planning_wall_seconds", 0.0)),
        "batch_replan_count": int(result.get("batch_replan_count", 0)),
        "episode_replan_count": sum(int(record.get("episode_replan_count", 0)) for record in episode_records),
        "episode_steps_executed": sum(int(record.get("steps_executed", 0)) for record in episode_records),
        "episode_amortized_planning_seconds": sum(float(record.get("episode_amortized_planning_seconds", 0.0)) for record in episode_records),
    }


def _primary_rows() -> list[tuple[str, str, str]]:
    """Return (report label, method_id, source rule) for the frozen six-row table."""
    return [
        ("LeWM", "lewm__main", "direct"),
        ("LeFlow", "leflow__main", "direct"),
        ("CoWM-Selection", "cowm_p3__main", "direct"),
        ("CoWM-Refinement (P0-PO-L)", "cowm_p0_po_l__main", "direct"),
        ("CoWM-Selection†", "cowm_selection_dagger__main", "reacher_dagger_else_p3"),
        ("Sub-JEPA", "subjepa__main", "direct"),
    ]


def _cell_lookup() -> dict[tuple[str, str, int], dict[str, Any]]:
    return {
        (str(cell["method_id"]), str(cell["task"]), int(cell["evaluation_seed"])): cell
        for cell in evaluation_cells()
    }


def command_summarize(args) -> None:
    root = _output_root(args)
    status = _read_json(root / "status.json", {"cells": {}})
    cells = evaluation_cells()
    lookup = _cell_lookup()
    all_rows = []
    value_index: dict[tuple[str, str, int], dict[str, Any] | None] = {}
    for cell in cells:
        value = _load_success_result(root, cell, status)
        key = (str(cell["method_id"]), str(cell["task"]), int(cell["evaluation_seed"]))
        value_index[key] = value
        all_rows.append(
            {
                "cell_id": cell["cell_id"],
                "method_id": cell["method_id"],
                "family": cell["family"],
                "task": cell["task"],
                "evaluation_seed": cell["evaluation_seed"],
                "stage": cell["stage"],
                "status": "complete" if value is not None else status.get("cells", {}).get(cell["cell_id"], {}).get("status", "missing"),
                "success_rate": "NA" if value is None else f"{value['success_rate']:.8f}",
                "episode_count": 0 if value is None else value["episode_count"],
                "batch_evaluation_wall_seconds": "NA" if value is None else f"{value['batch_evaluation_wall_seconds']:.6f}",
                "policy_environment_evaluation_seconds": "NA" if value is None else f"{value['policy_environment_evaluation_seconds']:.6f}",
                "batch_planning_wall_seconds": "NA" if value is None else f"{value['batch_planning_wall_seconds']:.6f}",
                "batch_replan_count": "NA" if value is None else value["batch_replan_count"],
                "episode_replan_count": "NA" if value is None else value["episode_replan_count"],
                "episode_steps_executed": "NA" if value is None else value["episode_steps_executed"],
                "episode_amortized_planning_seconds": "NA" if value is None else f"{value['episode_amortized_planning_seconds']:.6f}",
                "result_path": "" if value is None else value["result_path"],
            }
        )

    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(summary_dir / "all_independent_cells.csv", all_rows)
    _write_csv(
        summary_dir / "all_cell_runtime.csv",
        [
            {key: row[key] for key in (
                "cell_id", "method_id", "task", "evaluation_seed", "status",
                "batch_evaluation_wall_seconds", "batch_planning_wall_seconds",
                "batch_replan_count", "episode_replan_count", "episode_steps_executed",
                "episode_amortized_planning_seconds", "result_path",
            )}
            for row in all_rows
        ],
    )

    primary = []
    report_rows = []
    for label, method_id, source_rule in _primary_rows():
        for task in TASKS:
            task_seed_values = []
            for seed in SEEDS:
                source_method = method_id
                if source_rule == "reacher_dagger_else_p3" and task != "reacher":
                    source_method = "cowm_p3__main"
                cell = lookup.get((source_method, task, seed))
                value = None if cell is None else _load_success_result(root, cell, status)
                task_seed_values.append(value)
                primary.append(
                    {
                        "method": label,
                        "method_id": method_id,
                        "task": task,
                        "seed": seed,
                        "source_cell_id": "NA" if cell is None else cell["cell_id"],
                        "success_rate": "NA" if value is None else f"{value['success_rate']:.8f}",
                        "episode_count": 0 if value is None else value["episode_count"],
                        "result_path": "" if value is None else value["result_path"],
                    }
                )
            observed = [row["success_rate"] for row in task_seed_values if row is not None]
            if len(observed) == len(SEEDS):
                import numpy as np

                mean = float(np.mean(observed))
                sample_std = float(np.std(observed, ddof=1))
                task_summary = f"{100 * mean:.2f} ± {100 * sample_std:.2f}"
            else:
                task_summary = "NA"
            report_rows.append((label, task, task_summary))
        per_seed_means = []
        for seed in SEEDS:
            task_values = []
            for task in TASKS:
                source_method = method_id
                if source_rule == "reacher_dagger_else_p3" and task != "reacher":
                    source_method = "cowm_p3__main"
                value = value_index.get((source_method, task, seed))
                if value is not None:
                    task_values.append(value["success_rate"])
            if len(task_values) == len(TASKS):
                per_seed_means.append(sum(task_values) / len(TASKS))
        if len(per_seed_means) == len(SEEDS):
            import numpy as np

            average_summary = f"{100 * np.mean(per_seed_means):.2f} ± {100 * np.std(per_seed_means, ddof=1):.2f}"
        else:
            average_summary = "NA"
        report_rows.append((label, "四任务平均", average_summary))
    _write_csv(summary_dir / "table1_six_rows_per_seed.csv", primary)

    timing_summary = _read_json(root / "timing_summary.json", {})
    timing_by_condition = timing_summary.get("conditions", {})
    timing_rows = []
    for condition_id, condition in sorted(timing_by_condition.items()):
        identity = condition.get("condition", {})
        timing_rows.append(
            {
                "condition_id": condition_id,
                "status": condition.get("status", "missing"),
                "family": identity.get("family", ""),
                "method_id": identity.get("method_id", ""),
                "task": identity.get("task", ""),
                "execution_seed": identity.get("evaluation_seed", 42),
                "execute_steps": identity.get("execute_steps", 25),
                "mean_ms": "NA" if condition.get("mean_ms") is None else f"{float(condition['mean_ms']):.6f}",
                "nominal_ms_per_planned_step": "NA" if condition.get("mean_ms") is None else f"{float(condition['mean_ms']) / max(1, int(identity.get('execute_steps', 25))):.6f}",
                "p50_ms": "NA" if condition.get("p50_ms") is None else f"{float(condition['p50_ms']):.6f}",
                "p95_ms": "NA" if condition.get("p95_ms") is None else f"{float(condition['p95_ms']):.6f}",
                "measurement_count": condition.get("measurement_count", 0),
                "unique_states": condition.get("unique_states", 0),
                "loaded_model_baseline_vram_mib": "NA" if condition.get("loaded_model_baseline_vram_bytes") is None else f"{condition['loaded_model_baseline_vram_bytes'] / (1024**2):.3f}",
                "planning_peak_vram_mib": "NA" if condition.get("planning_peak_vram_bytes") is None else f"{condition['planning_peak_vram_bytes'] / (1024**2):.3f}",
                "attempt": condition.get("attempt", ""),
                "raw_result": str(root / "timing/runs" / condition_id / str(condition.get("attempt", "")) / "result.json") if condition.get("attempt") else "",
            }
        )
    _write_csv(summary_dir / "timing_conditions.csv", timing_rows)
    for label, method_id, _source_rule in _primary_rows():
        method_map = {
            "LeWM": "lewm__main",
            "LeFlow": "leflow__main",
            "CoWM-Selection": "cowm_p3__main",
            "CoWM-Refinement (P0-PO-L)": "cowm_p0_po_l__main",
            "CoWM-Selection†": "cowm_selection_dagger__main",
            "Sub-JEPA": "subjepa__main",
        }
        method = method_map[label]
        condition_means = []
        for task in TASKS:
            condition_id = f"{method}__{task}__seed_42"
            condition = timing_by_condition.get(condition_id)
            if condition is not None and condition.get("status") == "complete":
                condition_means.append(float(condition["mean_ms"]))
        if label == "CoWM-Selection†":
            # Three task measurements are reused from CoWM-Selection.
            reacher = timing_by_condition.get("cowm_selection_dagger__main__reacher__seed_42")
            condition_means = [
                float(timing_by_condition[f"cowm_p3__main__{task}__seed_42"]["mean_ms"])
                for task in ("tworoom", "pusht", "cube")
                if timing_by_condition.get(f"cowm_p3__main__{task}__seed_42", {}).get("status") == "complete"
            ]
            if reacher is not None and reacher.get("status") == "complete":
                condition_means.append(float(reacher["mean_ms"]))
        mean_latency = sum(condition_means) / len(TASKS) if len(condition_means) == len(TASKS) else None
        for task in TASKS:
            condition_method = (
                "cowm_p3__main"
                if label == "CoWM-Selection†" and task != "reacher"
                else method
            )
            condition_id = f"{condition_method}__{task}__seed_42"
            condition = timing_by_condition.get(condition_id)
            latency = "NA" if condition is None or condition.get("status") != "complete" else f"{float(condition['mean_ms']):.4f}"
            primary.append(
                {
                    "method": label,
                    "method_id": method,
                    "task": f"latency_{task}_ms",
                    "seed": "42",
                    "source_cell_id": condition_id,
                    "success_rate": latency,
                    "episode_count": 0,
                    "result_path": "",
                }
            )
        primary.append(
            {
                "method": label,
                "method_id": method,
                "task": "latency_equal_task_mean_ms",
                "seed": "42",
                "source_cell_id": "timing_summary.json",
                "success_rate": "NA" if mean_latency is None else f"{mean_latency:.4f}",
                "episode_count": 0,
                "result_path": "",
            }
        )
    _write_csv(summary_dir / "table1_with_timing.csv", primary)

    total_complete = sum(1 for cell in cells if _load_success_result(root, cell, status) is not None)
    report_lines = [
        "# CVPR Table 1 formal evaluation report",
        "",
        "Protocol: `outputs/cvpr/table1/v1/frozen_config.json`. Results are loaded only when their result and episode trace match the frozen cell identity; missing or failed results are shown as `NA`.",
        "",
        f"Evaluation matrix progress: {total_complete}/{len(cells)} independent cells; target episode count {len(cells) * 50}.",
        "",
        "Success rate is reported as mean ± sample standard deviation (`ddof=1`) across the six evaluation seeds. The final average is the mean of each seed's four-task average. The error bars describe evaluation-seed variation, not training-seed variation.",
        "",
        "| Method | TwoRoom | PushT | Reacher | OGBench-Cube | Four-task average |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    by_method = {}
    for label, task, value in report_rows:
        by_method.setdefault(label, {})[task] = value
    for label, _method_id, _rule in _primary_rows():
        values = by_method.get(label, {})
        report_lines.append(
            "| " + " | ".join(
                [label] + [values.get(task, "NA") for task in (*TASKS, "四任务平均")]
            ) + " |"
        )
    report_lines.extend(
        [
            "",
            "For CoWM-Selection†, the Reacher cell uses the frozen 10/10 protocol; TwoRoom, PushT, and Cube reuse the corresponding CoWM-Selection cells. Crop-mode conditions are reported separately in `summary/all_independent_cells.csv`.",
            "",
            "Timing uses the separately frozen batch=1 initial full-replanning protocol: runtime-warmed initial replanning with native initial history/padding and no existing target-latent cache. The table currently includes only completed timing conditions. `summary/timing_conditions.csv` reports per-task mean, P50/P95, unique states, nominal latency per planned execution step, and loaded-baseline/planning-peak VRAM; each referenced timing result preserves all 250 raw samples and parity evidence.",
            "",
            "| Method | Mean initial replanning latency (ms) |",
            "|---|---:|",
        ]
    )
    for label, method_id, _rule in _primary_rows():
        method = {"LeWM": "lewm__main", "LeFlow": "leflow__main", "CoWM-Selection": "cowm_p3__main", "CoWM-Refinement (P0-PO-L)": "cowm_p0_po_l__main", "CoWM-Selection†": "cowm_selection_dagger__main", "Sub-JEPA": "subjepa__main"}[label]
        values = []
        for task in TASKS:
            key = f"{method}__{task}__seed_42"
            value = timing_by_condition.get(key)
            if value is not None and value.get("status") == "complete":
                values.append(float(value["mean_ms"]))
        if label == "CoWM-Selection†":
            values = [
                float(timing_by_condition[key]["mean_ms"])
                for key in (
                    "cowm_p3__main__tworoom__seed_42",
                    "cowm_p3__main__pusht__seed_42",
                    "cowm_selection_dagger__main__reacher__seed_42",
                    "cowm_p3__main__cube__seed_42",
                )
                if timing_by_condition.get(key, {}).get("status") == "complete"
            ]
        latency = "NA" if len(values) != len(TASKS) else f"{sum(values) / len(TASKS):.4f}"
        report_lines.append(f"| {label} | {latency} |")
    report_lines.extend(
        [
            "",
            "See `summary/table1_six_rows_per_seed.csv` and `summary/all_independent_cells.csv` to reconstruct the reported aggregates from per-cell outputs.",
            "Batch-runtime accounting is in `summary/all_cell_runtime.csv`: each batch planning call is counted once, while episode replan counts, executed steps, and amortized planning time are derived from its recorded participants.",
        ]
    )
    report_root = _repo_path(_load_config()["outputs"]["report_root"])
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / REPORT_NAME).write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    _write_json(
        summary_dir / "summary_manifest.json",
        {
            "status": "complete" if total_complete == len(cells) else "partial",
            "independent_cells_complete": total_complete,
            "independent_cells_target": len(cells),
            "report": str(report_root / REPORT_NAME),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    print(json.dumps({"status": "complete" if total_complete == len(cells) else "partial", "completed_cells": total_complete, "target_cells": len(cells), "report": str(report_root / REPORT_NAME)}, indent=2))


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class _SingleSlotTimingEnv:
    def __init__(self, single_action_space):
        import numpy as np
        from gymnasium.spaces import Box

        self.num_envs = 1
        self.single_action_space = single_action_space
        self.action_space = Box(
            low=np.expand_dims(np.asarray(single_action_space.low), 0),
            high=np.expand_dims(np.asarray(single_action_space.high), 0),
            dtype=single_action_space.dtype,
        )


def _cpu_copy(value: Any):
    import numpy as np
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, Mapping):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    return copy.deepcopy(value)


def _info_slot(info: Mapping[str, Any], index: int, total: int):
    import numpy as np
    import torch

    selected = {}
    for key, value in info.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == total:
            selected[key] = value[index : index + 1].detach().cpu().clone()
        elif isinstance(value, np.ndarray) and value.ndim and value.shape[0] == total:
            selected[key] = np.array(value[index : index + 1], copy=True)
        elif isinstance(value, list) and len(value) == total:
            selected[key] = [copy.deepcopy(value[index])]
        else:
            selected[key] = _cpu_copy(value)
    selected["terminated"] = np.zeros(1, dtype=bool)
    selected["_needs_flush"] = np.ones(1, dtype=bool)
    return selected


def _walk_policy_objects(policy: Any):
    seen = set()
    pending = [policy]
    while pending:
        item = pending.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        yield item
        for name in ("solver", "_solver", "model", "fast_model", "verifier_model"):
            nested = getattr(item, name, None)
            if nested is not None and nested is not item:
                pending.append(nested)


def _reset_timing_policy(policy: Any, env: Any, seed: int) -> None:
    import numpy as np
    import torch

    policy.set_env(env)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    for item in _walk_policy_objects(policy):
        set_seed = getattr(item, "set_seed", None)
        if callable(set_seed):
            set_seed(int(seed))
        for name in ("_generators", "_selection_generators"):
            generators = getattr(item, name, None)
            if isinstance(generators, dict):
                generators.clear()
        generator = getattr(item, "torch_gen", None)
        if generator is not None and hasattr(generator, "manual_seed"):
            generator.manual_seed(int(seed))
        for name in ("_goal_latent_cache", "last_timing_selection"):
            if hasattr(item, name):
                value = getattr(item, name)
                if name == "_goal_latent_cache" and isinstance(value, list):
                    setattr(item, name, [None] * env.num_envs)
                else:
                    setattr(item, name, None)


def _set_timing_mode(policy: Any):
    for item in _walk_policy_objects(policy):
        if hasattr(item, "timing_mode"):
            item.timing_mode = True
        if hasattr(item, "capture_timing_selection"):
            item.capture_timing_selection = True
    return install_timing_cem(policy)


def _cpu_action(value: Any):
    import numpy as np
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, Mapping):
        return {str(key): _cpu_action(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_cpu_action(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _policy_selection(policy: Any, *, timed_solver=None, timing_mode: bool):
    import numpy as np
    import torch

    selected = None
    if timing_mode and timed_solver is not None:
        batches = getattr(timed_solver, "last_selected_indices", None)
        if batches:
            tensors = [value.detach().cpu().reshape(-1) for value in batches if torch.is_tensor(value)]
            selected = torch.cat(tensors).numpy() if tensors else None
    if selected is None and not timing_mode:
        capture = getattr(policy, "cvpr_cem_capture", None)
        if capture is not None and capture.events:
            candidate = capture.events[-1].get("selected_candidate_indices")
            if candidate is not None:
                selected = np.asarray(candidate)
    if selected is None:
        for item in _walk_policy_objects(policy):
            candidate = getattr(item, "last_timing_selection", None)
            if candidate is not None:
                if torch.is_tensor(candidate):
                    selected = candidate.detach().cpu().numpy()
                else:
                    selected = np.asarray(candidate)
                break
    if selected is None and not timing_mode:
        for item in _walk_policy_objects(policy):
            events = getattr(item, "planning_events", None)
            if events:
                candidate = events[-1].get("selected_indices")
                if candidate is not None:
                    selected = np.asarray(candidate)
                    break
    return None if selected is None else np.asarray(selected).reshape(-1).tolist()


def _sync_cuda():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize(device=0)


def _timed_policy_call(policy: Any, info: Mapping[str, Any], seed: int, env: Any):
    import time
    import torch

    _reset_timing_policy(policy, env, seed)
    _sync_cuda()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=0)
    started = time.perf_counter()
    action = policy.get_action(copy.deepcopy(info))
    action_cpu = _cpu_action(action)
    _sync_cuda()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device=0))
        if torch.cuda.is_available()
        else 0
    )
    return action_cpu, elapsed_ms, peak_bytes


def _compare_action_outputs(normal: Any, timed: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    import numpy as np

    def arrays(value):
        if isinstance(value, Mapping):
            return {key: arrays(item) for key, item in value.items()}
        if isinstance(value, list):
            return [arrays(item) for item in value]
        try:
            return np.asarray(value)
        except Exception:
            return value

    left, right = arrays(normal), arrays(timed)

    def compare(a, b, path="action"):
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                return {"path": path, "equal": False, "reason": "mapping keys differ"}
            for key in a:
                result = compare(a[key], b[key], f"{path}.{key}")
                if not result["equal"]:
                    return result
            return {"path": path, "equal": True}
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return {"path": path, "equal": False, "reason": "list lengths differ"}
            for index, (left_value, right_value) in enumerate(zip(a, b)):
                result = compare(left_value, right_value, f"{path}[{index}]")
                if not result["equal"]:
                    return result
            return {"path": path, "equal": True}
        try:
            left_array = np.asarray(a)
            right_array = np.asarray(b)
            equal = left_array.shape == right_array.shape and np.allclose(
                left_array, right_array, atol=atol, rtol=rtol, equal_nan=False
            )
            maximum = (
                float(np.max(np.abs(left_array - right_array)))
                if left_array.size and left_array.shape == right_array.shape
                else None
            )
            return {
                "path": path,
                "equal": bool(equal),
                "max_absolute_error": maximum,
                "normal_shape": list(left_array.shape),
                "timing_shape": list(right_array.shape),
            }
        except (TypeError, ValueError):
            equal = a == b
            return {"path": path, "equal": bool(equal)}

    return compare(left, right)


def _check_timing_idle(gpu: int, config: Mapping[str, Any]) -> dict[str, Any]:
    if gpu != int(config["timing"]["gpu"]):
        raise RuntimeError(f"independent timing is frozen to GPU {config['timing']['gpu']}")
    preflight = _check_dispatch_resources(gpu, config)
    process_query = subprocess.run(
        [
            "nvidia-smi", "-i", str(gpu),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    other_processes = []
    for line in process_query.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3 and fields[0].isdigit() and int(fields[0]) != os.getpid():
            other_processes.append(fields)
    if other_processes:
        raise RuntimeError(f"timing requires idle GPU {gpu}; other compute processes: {other_processes}")
    if not process_query and int(preflight["gpu_utilization_percent"]) > 5:
        raise RuntimeError(
            f"timing requires idle GPU {gpu}; utilization is "
            f"{preflight['gpu_utilization_percent']}% with no visible compute process"
        )
    cpu_count = max(1, os.cpu_count() or 1)
    load_1, load_5, load_15 = os.getloadavg()
    normalized_load = float(load_1) / cpu_count
    if normalized_load > 0.75:
        raise RuntimeError(
            f"timing requires a quiet CPU; 1-minute normalized load is {normalized_load:.3f}"
        )
    return {
        **preflight,
        "other_compute_processes": other_processes,
        "cpu_count": cpu_count,
        "cpu_load_1m": load_1,
        "cpu_load_5m": load_5,
        "cpu_load_15m": load_15,
        "cpu_load_1m_per_cpu": normalized_load,
    }


def _timing_cell_identity(root: Path, cell: Mapping[str, Any]) -> dict[str, Any]:
    base = _cell_identity(root, cell)
    base["timing_protocol"] = _load_config()["timing"]
    base["precision"] = _load_config()["protocol"]["precision"]
    return base


def _condition_attempt(root: Path, cell_id: str) -> tuple[int, Path]:
    parent = root / "timing/runs" / cell_id
    parent.mkdir(parents=True, exist_ok=True)
    prior = []
    for path in parent.glob("attempt_*"):
        try:
            prior.append(int(path.name.removeprefix("attempt_")))
        except ValueError:
            pass
    number = max(prior, default=0) + 1
    return number, parent / f"attempt_{number:03d}"


def _capture_and_measure_condition(
    *, root: Path, cell: Mapping[str, Any], identity_sha256: str,
    config: Mapping[str, Any], model: Any, dataset: Any,
    manifest: Any, attempt_dir: Path,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from source.common.cvpr_table1 import attach_cem_archive_callback
    from source.common.round3_eval import run_round3_evaluation
    from source.common.round4_eval import run_round4_evaluation

    cfg = _cell_config(cell)
    cfg.output.save_video = False
    family = str(cell["family"])
    identity = type("TimingIdentity", (), {})()
    identity.entrypoint = "cvpr_table1_timing"
    identity.policy_kind = {
        "cowm": "fast_lewam", "lewm": "lewm",
        "leflow": "leflow", "subjepa": "subjepa_official",
    }[family]
    identity.checkpoint = None
    identity.epoch = 10 if family == "cowm" else None
    identity.stage = "stage_b" if family == "cowm" else None
    identity.actor_warm_start = False
    identity.guidance_mode = str(cell.get("guidance_mode", "none"))
    identity.guidance_step_size = float(cell.get("guidance_step_size", config["protocol"]["cowm"]["guidance_step_size"]))
    identity.guidance_last_steps = int(cell.get("guidance_last_steps") or 2)
    identity.guidance_inner_steps = int(cell.get("guidance_inner_steps") or 1)
    identity.guidance_max_rms_offset = float(cell.get("guidance_max_rms_offset", config["protocol"]["cowm"]["guidance_max_rms_offset"]))

    measurements: dict[str, Any] = {}

    def callback(policy, *args, **kwargs):
        info = args[0] if args else kwargs.get("info_dict")
        if not isinstance(info, Mapping):
            raise TypeError(f"timing policy input must be a mapping, got {type(info).__name__}")
        single_action_space = policy.env.single_action_space
        proxy = _SingleSlotTimingEnv(single_action_space)
        slots = int(policy.env.num_envs)
        if slots != len(manifest.entries):
            raise RuntimeError(
                f"timing input batch has {slots} slots, expected {len(manifest.entries)}"
            )
        input_info = _cpu_copy(info)

        cem_capture = None
        if family in {"lewm", "subjepa"} or (family == "cowm" and cell["mode"] in {"P1", "P2"}):
            cem_capture = attach_cem_archive_callback(
                policy,
                iterations=int(config["protocol"]["cem"]["iterations"]),
                topk=int(config["protocol"]["cem"]["topk"]),
            )
        for item in _walk_policy_objects(policy):
            if hasattr(item, "capture_timing_selection"):
                item.capture_timing_selection = True

        base_seed = int(config["timing"]["rng_seed_base"])
        parity_info = _info_slot(input_info, 0, slots)
        parity_seed = base_seed
        if cem_capture is not None:
            cem_capture.events.clear()
        normal_action, _normal_ms, _normal_peak = _timed_policy_call(
            policy, parity_info, parity_seed, proxy
        )
        normal_selection = _policy_selection(policy, timing_mode=False)

        timed_solver = _set_timing_mode(policy)
        timed_action, _timed_ms, _timed_peak = _timed_policy_call(
            policy, parity_info, parity_seed, proxy
        )
        timed_selection = _policy_selection(
            policy, timed_solver=timed_solver, timing_mode=True
        )
        tolerance = {
            "atol": float(config["timing"]["normal_timing_parity_atol"]),
            "rtol": float(config["timing"]["normal_timing_parity_rtol"]),
        }
        action_parity = _compare_action_outputs(
            normal_action, timed_action, atol=tolerance["atol"], rtol=tolerance["rtol"]
        )
        selection_available = normal_selection is not None or timed_selection is not None
        selection_parity = (
            {
                "available": True,
                "equal": normal_selection == timed_selection,
                "normal": normal_selection,
                "timing": timed_selection,
            }
            if selection_available
            else {"available": False, "equal": True, "reason": "single-candidate method"}
        )
        if not action_parity["equal"] or not selection_parity["equal"]:
            raise RuntimeError(
                "normal/timing parity failed before measurement: "
                + json.dumps({"action": action_parity, "selection": selection_parity})
            )

        baseline_bytes = int(torch.cuda.memory_allocated(device=0))
        state_order = list(range(slots))
        warmups = []
        for state_index in state_order[: int(config["timing"]["warmup_states"])]:
            seed = base_seed + 50000 + state_index
            info_slot = _info_slot(input_info, state_index, slots)
            _timed_action, elapsed, peak = _timed_policy_call(policy, info_slot, seed, proxy)
            warmups.append({"state_index": state_index, "rng_seed": seed, "wall_ms": elapsed, "peak_bytes": peak})

        raw_samples = []
        repeats = int(config["timing"]["repeats_per_state"])
        for state_index in state_order:
            for repeat_index in range(repeats):
                seed = base_seed + state_index * repeats + repeat_index
                info_slot = _info_slot(input_info, state_index, slots)
                _timed_action, elapsed, peak = _timed_policy_call(policy, info_slot, seed, proxy)
                raw_samples.append(
                    {
                        "state_index": state_index,
                        "episode_id": manifest.entries[state_index].episode_id,
                        "start_step": manifest.entries[state_index].start_step,
                        "repeat_index": repeat_index,
                        "rng_seed": seed,
                        "wall_ms": elapsed,
                        "peak_vram_bytes": peak,
                    }
                )
                if len(raw_samples) % 25 == 0:
                    _write_json(attempt_dir / "progress.json", {
                        "completed_measurements": len(raw_samples),
                        "target_measurements": slots * repeats,
                        "last_state_index": state_index,
                        "last_repeat_index": repeat_index,
                    })
                    with (attempt_dir / "raw_samples.jsonl").open("a", encoding="utf-8") as stream:
                        for sample in raw_samples[-25:]:
                            stream.write(json.dumps(_jsonable(sample), ensure_ascii=False) + "\n")

        samples_ms = np.asarray([sample["wall_ms"] for sample in raw_samples], dtype=np.float64)
        peak_vram = max(sample["peak_vram_bytes"] for sample in raw_samples)
        post_cpu_load = float(os.getloadavg()[0]) / max(1, os.cpu_count() or 1)
        pre_gpu = _check_timing_idle(int(config["timing"]["gpu"]), config)
        measurements.update(
            {
                "status": "complete",
                "identity_sha256": identity_sha256,
                "condition_id": cell["cell_id"],
                "condition": dict(cell),
                "mean_ms": float(samples_ms.mean()),
                "p50_ms": float(np.percentile(samples_ms, 50)),
                "p95_ms": float(np.percentile(samples_ms, 95)),
                "measurement_count": int(len(samples_ms)),
                "unique_states": int(len(set(sample["state_index"] for sample in raw_samples))),
                "repeats_per_state": repeats,
                "loaded_model_baseline_vram_bytes": baseline_bytes,
                "planning_peak_vram_bytes": int(peak_vram),
                "peak_includes_preprocess_and_refinement": True,
                "warmup_count": len(warmups),
                "warmup_samples": warmups,
                "raw_samples": raw_samples,
                "state_order": state_order,
                "rng_seed_base": base_seed,
                "normal_timing_parity": {
                    "tolerance": tolerance,
                    "action": action_parity,
                    "selection": selection_parity,
                },
                "gpu_after": pre_gpu,
                "cpu_load_1m_per_cpu_after": post_cpu_load,
                "timing_boundary": "policy call through CPU action output, synchronized CUDA, perf_counter",
            }
        )
        return measurements

    attempt_dir.mkdir(parents=True, exist_ok=False)
    _write_json(attempt_dir / "cell_identity.json", {"identity": dict(cell), "identity_sha256": identity_sha256})
    _write_json(attempt_dir / "rng_schedule.json", {
        "state_order": list(range(int(config["timing"]["unique_states"]))),
        "warmup_rng_seeds": [int(config["timing"]["rng_seed_base"]) + 50000 + index for index in range(int(config["timing"]["warmup_states"]))],
        "measurement_rng_seeds": [int(config["timing"]["rng_seed_base"]) + index for index in range(int(config["timing"]["unique_states"]) * int(config["timing"]["repeats_per_state"]))],
        "execution_order": "state_index ascending, repeat_index ascending",
    })
    if family == "cowm":
        returned = run_round4_evaluation(
            cfg, task=str(cell["task"]), policy_or_model=model,
            mode=str(cell["mode"]), identity=identity, manifest=manifest,
            output_dir=attempt_dir / "capture_runtime", trace=False,
            device="cuda:0", timing_capture_callback=callback,
            candidate_count=int(config["protocol"]["cowm"]["candidate_count"]),
            flow_steps=int(config["protocol"]["cowm"]["action_flow_steps"]),
            action_flow_steps=cell.get("action_flow_steps"),
            solver_batch_size=int(config["protocol"]["cowm"]["solver_batch_size"]),
            candidate_batch_size=int(config["protocol"]["cowm"]["candidate_batch_size"]),
            action_flow_integrator=str(config["protocol"]["cowm"]["integrator"]),
            action_bound_mode=str(cell["action_bound_mode"]),
            cem_protocol=str(cell["cem_protocol"]),
            guidance_mode=str(cell["guidance_mode"]),
            guidance_step_size=float(cell.get("guidance_step_size", config["protocol"]["cowm"]["guidance_step_size"])),
            guidance_last_steps=int(cell.get("guidance_last_steps") or 2),
            guidance_inner_steps=int(cell.get("guidance_inner_steps") or 1),
            guidance_max_rms_offset=float(cell.get("guidance_max_rms_offset", config["protocol"]["cowm"]["guidance_max_rms_offset"])),
            proposal_chunk_size=int(config["protocol"]["cowm"]["proposal_chunk_size"]),
            allowed_protocol_variants=("legacy",),
            allow_solver_config_override=True,
            allow_evaluation_seed_override=True,
            allow_cohort_seed_mismatch=True,
            policy_seed=int(cell["policy_seed"]),
            execute_steps=cell.get("execute_steps"),
            score_horizon_blocks=cell.get("score_horizon_blocks"),
            score_reduction=str(cell.get("score_reduction", "endpoint")),
            terminal_weight=float(cell.get("terminal_weight", 1.0)),
            dataset=dataset,
        )
    else:
        returned = run_round3_evaluation(
            cfg, task=str(cell["task"]), policy_or_model=model,
            identity=identity, manifest=manifest,
            output_dir=attempt_dir / "capture_runtime", trace=False,
            device="cuda:0", timing_capture_callback=callback,
            allow_evaluation_seed_override=True,
            allow_cohort_seed_mismatch=True,
            dataset=dataset,
        )
    if returned.get("status") != "timing_capture":
        raise RuntimeError(f"timing capture did not stop at the first policy input: {returned.get('status')}")
    measured = returned["timing_capture"]
    if measured.get("status") != "complete":
        raise RuntimeError(f"timing condition did not complete: {measured.get('status')}")
    _write_json(attempt_dir / "result.json", measured)
    return measured


def _timing_summary(root: Path, conditions: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = {}
    for cell in conditions:
        cell_id = str(cell["cell_id"])
        parent = root / "timing/runs" / cell_id
        complete = None
        for attempt in sorted(parent.glob("attempt_*")) if parent.exists() else []:
            result_path = attempt / "result.json"
            if result_path.is_file():
                value = _read_json(result_path)
                if value.get("status") == "complete":
                    complete = value
        rows[cell_id] = complete or {"status": "missing", "condition_id": cell_id}
    return {
        "schema_version": 1,
        "timing_conditions_target": len(conditions),
        "conditions_complete": sum(row.get("status") == "complete" for row in rows.values()),
        "conditions": rows,
    }


def command_timing(args) -> None:
    root = _output_root(args)
    config = _load_config()
    gpu = int(args.gpu)
    conditions = timing_conditions()
    if args.cell_id:
        conditions = [cell for cell in conditions if cell["cell_id"] == args.cell_id]
        if not conditions:
            raise ValueError(f"unknown timing condition: {args.cell_id}")
    if not (root / "cohorts/cohort_diagnostics.json").is_file():
        raise RuntimeError("formal timing requires frozen cohorts; run prepare first")
    with _exclusive_lock(root / f".locks/gpu_{gpu}.lock", blocking=False):
        preflight = _check_timing_idle(gpu, config)
        _mark_freeze_activity(root, key="timing_started")
        order_path = root / "timing_execution_order.json"
        if not order_path.exists():
            _write_json(order_path, {
                "condition_order": [cell["cell_id"] for cell in timing_conditions()],
                "state_order": list(range(int(config["timing"]["unique_states"]))),
                "rng_seed_base": int(config["timing"]["rng_seed_base"]),
                "repeats_per_state": int(config["timing"]["repeats_per_state"]),
                "gpu": gpu,
                "initial_preflight": preflight,
            })
        print(f"timing preflight: {json.dumps(preflight)}", flush=True)
        loaded: dict[tuple[str, int], dict[str, Any]] = {}
        datasets: dict[str, Any] = {}
        for cell in conditions:
            task = str(cell["task"])
            phase = int(cell["stage"])
            key = (task, phase)
            if key not in loaded:
                loaded[key] = _load_models(root, task, phase, "cuda:0")
            if task not in datasets:
                cfg = _cell_config(cell)
                from source.common.eval import get_dataset

                datasets[task] = get_dataset(cfg, cfg.eval.dataset_name)
            identity = _timing_cell_identity(root, cell)
            identity_sha256 = identity_hash(identity)
            parent = root / "timing/runs" / cell["cell_id"]
            complete = False
            for attempt in sorted(parent.glob("attempt_*")) if parent.exists() else []:
                result_path = attempt / "result.json"
                if result_path.is_file():
                    value = _read_json(result_path)
                    if value.get("status") == "complete" and value.get("identity_sha256") == identity_sha256:
                        complete = True
            if complete:
                continue
            attempts = 0
            while attempts < 3:
                attempts += 1
                before = _check_timing_idle(gpu, config)
                number, attempt_dir = _condition_attempt(root, str(cell["cell_id"]))
                try:
                    manifest_path = root / "cohorts" / task / "seed_42.json"
                    from source.common.round3_phase1 import CohortManifest

                    manifest = CohortManifest.load(manifest_path)
                    value = _capture_and_measure_condition(
                        root=root, cell=cell, identity_sha256=identity_sha256,
                        config=config, model=loaded[key][str(cell["family"])],
                        dataset=datasets[task], manifest=manifest,
                        attempt_dir=attempt_dir,
                    )
                    value["attempt"] = attempt_dir.name
                    value["gpu_before"] = before
                    after = _check_timing_idle(gpu, config)
                    value["gpu_after"] = after
                    if after["cpu_load_1m_per_cpu"] > 0.75:
                        value["status"] = "interfered"
                        value["interference_reason"] = "CPU load exceeded 0.75 per core after condition"
                    else:
                        value["status"] = "complete"
                    _write_json(attempt_dir / "result.json", value)
                    if value["status"] == "complete":
                        break
                except BaseException as exc:
                    attempt_dir.mkdir(parents=True, exist_ok=True)
                    _write_json(attempt_dir / "failure.json", {
                        "status": "failed", "error_type": type(exc).__name__,
                        "error": str(exc), "traceback": traceback.format_exc(),
                    })
                    # An idle-GPU check failure is an external scheduling condition,
                    # so keep the attempt and stop instead of starting another one.
                    raise
            _write_json(root / "timing_summary.json", _timing_summary(root, timing_conditions()))
            if attempts >= 3 and value.get("status") != "complete":
                raise RuntimeError(f"timing condition remained interfered after {attempts} full attempts: {cell['cell_id']}")
            print(f"timed {cell['cell_id']}: {value.get('mean_ms', 'NA')} ms", flush=True)
        summary = _timing_summary(root, timing_conditions())
        summary["gpu"] = gpu
        summary["precision"] = config["protocol"]["precision"]
        summary["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_json(root / "timing_summary.json", summary)
        print(json.dumps({"conditions_complete": summary["conditions_complete"], "conditions_target": summary["timing_conditions_target"]}, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser("matrix", help="print the frozen matrix")
    matrix.add_argument("--root")
    matrix.set_defaults(func=command_matrix)
    for name, function in (("prepare", command_prepare), ("snapshot-refresh", command_snapshot_refresh), ("archive-assets", command_archive_assets), ("status", command_status), ("summarize", command_summarize)):
        command = subparsers.add_parser(name)
        command.add_argument("--root")
        command.set_defaults(func=function)
    worker = subparsers.add_parser("worker", help="evaluate cells for one task and phase")
    worker.add_argument("--root")
    worker.add_argument("--gpu", type=int, required=True)
    worker.add_argument("--task", choices=TASKS, required=True)
    worker.add_argument("--phase", choices=(1, 2), type=int, required=True)
    worker.add_argument("--cell-id")
    worker.add_argument("--stop-on-error", action="store_true")
    worker.set_defaults(func=command_worker)
    timing = subparsers.add_parser("timing", help="run batch=1 planning latency conditions")
    timing.add_argument("--root")
    timing.add_argument("--gpu", type=int, required=True)
    timing.add_argument("--cell-id")
    timing.set_defaults(func=command_timing)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
