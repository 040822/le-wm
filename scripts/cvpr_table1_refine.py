#!/usr/bin/env python3
"""Prepare and execute the Reacher-focused CVPR Table 1 CoWM refinement study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import numpy as np
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "config/cvpr/table1_refine.json"
TRAIN_MANIFEST = ROOT / "config/round5/phase1_6.json"
BASE_SCRIPT = ROOT / "scripts/cvpr_table1.py"
REPORT_PATH = ROOT / "docs/report/cvpr/table1/cvpr_table1_refine_report.md"

_spec = importlib.util.spec_from_file_location("_cvpr_table1_base", BASE_SCRIPT)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load Table 1 runner: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)
base.CONFIG_PATH = CONFIG_PATH
base.REPORT_NAME = REPORT_PATH.name
base.SOURCE_SNAPSHOT_PATHS = tuple(
    dict.fromkeys(
        (*base.SOURCE_SNAPSHOT_PATHS,
         "config/cvpr/table1_refine.json",
         "scripts/cvpr_table1_refine.py",
         "source/common/cvpr_table1_refine_train.py")
    )
)

from source.common.cvpr_table1 import SEEDS, TASKS, _cell
from source.common.round3_phase1 import CohortManifest
from source.common.round5_phase1_6 import stable_hash
from source.common.cvpr_table1_refine_train import prepare_offline_cache, train_checkpoint


def _config() -> dict[str, Any]:
    return base._load_config()


def _root(args=None) -> Path:
    return base._output_root(args)


def _atomic_json(path: Path, value: Any) -> None:
    base._write_json(path, value)


def _read_json(path: Path, default=None):
    return base._read_json(path, default)


def _mark_freeze_activity(root: Path, key: str) -> None:
    freeze_path = root / "freeze.json"
    freeze = _read_json(freeze_path)
    if not freeze:
        raise FileNotFoundError("experiment freeze is missing; run prepare first")
    if not freeze.get(key):
        freeze[key] = True
        freeze[f"{key}_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _atomic_json(freeze_path, freeze)


def _sha(path: Path) -> str:
    return base._sha256(path)


def _refine_config() -> dict[str, Any]:
    return _config()["refinement"]


def _score_specs() -> list[dict[str, Any]]:
    return _refine_config()["score_reductions"]


def _base_checkpoint(root: Path, task: str = "reacher") -> Path:
    paths = sorted((root / "assets/cowm" / task / "checkpoints").glob("*.pt"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one archived CoWM checkpoint for {task}, found {paths}")
    return paths[0]


def _custom_cells(method_id: str, label: str, seed: int, *, task="reacher", **fields):
    method = {
        "method_id": method_id,
        "id": label,
        "label": label,
        "family": "cowm",
        "stage": 1,
        "mode": "P3",
        "guidance_mode": "none",
        "action_flow_steps": 2,
        "action_bound_mode": "none",
        "cem_protocol": "not_applicable",
        "guidance_step_size": 0.01,
        "guidance_last_steps": 2,
        "guidance_inner_steps": 1,
        "guidance_max_rms_offset": 0.2,
        "score_horizon_blocks": None,
        "score_reduction": "endpoint",
        "terminal_weight": 1.0,
        **fields,
    }
    return _cell(method, task, seed)


def _source_identity(root: Path, cell: Mapping[str, Any], checkpoint: Path, *, training_receipt=None):
    identity = base._cell_identity(root, cell)
    record = {"path": str(checkpoint.resolve()), "sha256": _sha(checkpoint)}
    if cell["family"] == "cowm":
        identity["asset_sha256"] = [
            item
            for item in identity["asset_sha256"]
            if not str(item["path"]).endswith(".pt")
        ] + [record]
        identity["checkpoint_sha256"] = record["sha256"]
    else:
        identity["runtime_checkpoint"] = record
    if training_receipt is not None:
        identity["training_receipt_sha256"] = _sha(training_receipt)
    return identity


def _load_model(checkpoint: Path, device: str):
    from source.common.checkpoint import load_policy_or_model

    policy, resolved = load_policy_or_model(str(checkpoint), cache_dir=str(ROOT / "data"))
    if resolved is None or Path(resolved).resolve() != checkpoint.resolve():
        raise RuntimeError(f"checkpoint resolver changed requested checkpoint: {resolved}")
    model = getattr(policy, "model", policy)
    return model.to(device).eval()


def _next_attempt(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in parent.glob("attempt_*"):
        try:
            numbers.append(int(path.name.removeprefix("attempt_")))
        except ValueError:
            continue
    return parent / f"attempt_{max(numbers, default=0) + 1:03d}"


def _run_eval_cell(
    *, root: Path, gpu: int, cell: Mapping[str, Any], checkpoint: Path,
    model: Any, training_receipt: Path | None = None, dataset=None,
) -> dict[str, Any]:
    _mark_freeze_activity(root, "formal_evaluations_started")
    identity = _source_identity(
        root, cell, checkpoint, training_receipt=training_receipt
    )
    identity_sha = base.identity_hash(identity)
    parent = root / "refinement/evaluations" / str(cell["cell_id"])
    for prior in sorted(parent.glob("attempt_*")) if parent.exists() else []:
        result_path = prior / "result.json"
        trace_path = prior / "episodes.jsonl"
        result = _read_json(result_path)
        if (
            result
            and result.get("cvpr_table1", {}).get("identity_sha256") == identity_sha
            and result.get("status") == "ok"
            and trace_path.is_file()
            and result.get("cvpr_table1", {}).get("trace_sha256") == _sha(trace_path)
        ):
            return _publish_refine_fields(
                result, cell, identity, identity_sha, result_path
            )
    attempt = _next_attempt(parent)
    attempt.mkdir(parents=True)
    _atomic_json(
        attempt / "cell_identity.json",
        {"cell": dict(cell), "identity": identity, "identity_sha256": identity_sha},
    )
    _atomic_json(
        attempt / "status.json",
        {"status": "running", "gpu": int(gpu), "started_utc": time.time()},
    )
    if dataset is None:
        from source.common.eval import get_dataset

        cfg = base._cell_config(cell)
        dataset = get_dataset(cfg, cfg.eval.dataset_name)
    try:
        result = base._run_one_cell(
            root=root,
            cell=cell,
            attempt_dir=attempt,
            identity=identity,
            identity_sha256=identity_sha,
            gpu=int(gpu),
            models={"cowm": model},
            dataset=dataset,
        )
        result = _publish_refine_fields(
            result, cell, identity, identity_sha, attempt / "result.json"
        )
        _atomic_json(
            attempt / "status.json",
            {
                "status": "complete",
                "identity_sha256": identity_sha,
                "successes": result["successes"],
                "episodes": result["num_episodes"],
                "completed_utc": time.time(),
            },
        )
        return result
    except BaseException as exc:
        _atomic_json(
            attempt / "failure.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def _publish_refine_fields(result, cell, identity, identity_sha, result_path):
    episodes = result.get("episodes", [])
    if result.get("status") != "ok" or len(episodes) != 50:
        raise RuntimeError("refinement result must contain exactly 50 completed episodes")
    successes = sum(bool(episode.get("success")) for episode in episodes)
    if not np.isclose(float(result.get("success_rate", -1)), successes / len(episodes)):
        raise RuntimeError("top-level success rate disagrees with the 50 episode records")
    fields = {
        "cell": dict(cell),
        "evaluation_seed": int(cell["evaluation_seed"]),
        "environment_seed": int(cell["environment_seed"]),
        "policy_seed": int(cell["policy_seed"]),
        "successes": int(successes),
        "num_episodes": int(len(episodes)),
        "refinement_identity_sha256": identity_sha,
        "refinement_identity": dict(identity),
    }
    changed = any(result.get(key) != value for key, value in fields.items())
    result.update(fields)
    if changed:
        _atomic_json(result_path, result)
    return result


def _run_reacher_score_cells(
    *, root: Path, gpu: int, checkpoint: Path, checkpoint_id: str,
    score_specs: list[dict[str, Any]], training_receipt: Path | None = None,
) -> None:
    from source.common.eval import get_dataset

    config = _config()
    first_cell = _custom_cells(
        f"cvpr_refine_{checkpoint_id}_{score_specs[0]['id']}",
        f"Refine {checkpoint_id} {score_specs[0]['id']}",
        SEEDS[0],
        score_reduction=score_specs[0]["mode"],
        terminal_weight=score_specs[0]["terminal_weight"],
    )
    dataset_cfg = base._cell_config(first_cell)
    dataset = get_dataset(dataset_cfg, dataset_cfg.eval.dataset_name)
    model = _load_model(checkpoint, "cuda:0")
    try:
        for score in score_specs:
            for seed in SEEDS:
                cell = _custom_cells(
                    f"cvpr_refine_{checkpoint_id}_{score['id']}",
                    f"CoWM P3 {checkpoint_id} {score['id']}",
                    seed,
                    score_reduction=score["mode"],
                    terminal_weight=float(score["terminal_weight"]),
                    checkpoint_label=checkpoint_id,
                    score_id=score["id"],
                )
                _run_eval_cell(
                    root=root,
                    gpu=gpu,
                    cell=cell,
                    checkpoint=checkpoint,
                    model=model,
                    training_receipt=training_receipt,
                    dataset=dataset,
                )
                print(f"complete {cell['cell_id']}", flush=True)
    finally:
        del model, dataset
        import torch

        torch.cuda.empty_cache()


def _cohort_paths(root: Path) -> list[Path]:
    return [root / "cohorts/reacher" / f"seed_{seed}.json" for seed in SEEDS]


def _training_source(root: Path) -> tuple[Path, str]:
    manifest = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
    record = manifest["training"]
    source_checkpoint = base._repo_path(record["checkpoints"]["reacher"]).resolve(strict=True)
    expected = record["checkpoint_sha256"]["reacher"]
    if _sha(source_checkpoint) != expected:
        raise RuntimeError("the source Reacher checkpoint SHA256 differs from Phase 1.6 manifest")
    checkpoint = _base_checkpoint(root)
    if _sha(checkpoint) != expected:
        raise RuntimeError("the archived Reacher checkpoint SHA256 differs from Phase 1.6 manifest")
    return checkpoint, expected


def command_prepare(args) -> None:
    root = _root(args)
    base.command_archive_assets(SimpleNamespace(root=str(root)))
    config = _config()
    previous_freeze = _read_json(root / "freeze.json", {})
    previous_revision = _read_json(root / "provenance/active_source_revision.json")
    current_revision_id = base._code_revision_id(base._current_source_hashes())
    if (
        previous_freeze.get("formal_evaluations_started")
        or previous_freeze.get("timing_started")
    ) and previous_revision and current_revision_id != previous_revision.get("revision_id"):
        raise RuntimeError(
            "source code changed after formal work began; preserve the frozen experiment snapshot"
        )
    base._copy_independent(CONFIG_PATH, root / "provenance/config/table1_refine.json")
    reference_root = ROOT / "outputs/cvpr/table1/v1"
    reference_config = _read_json(reference_root / "frozen_config.json")
    if not reference_config:
        raise FileNotFoundError("the frozen Table 1 v1 configuration is missing")
    compatibility_keys = ("tasks", "seeds", "seed_policy")
    for key in compatibility_keys:
        if config[key] != reference_config[key]:
            raise RuntimeError(f"refinement config changed frozen Table 1 {key}")
    for key in (
        "variant",
        "goal_offset_steps",
        "episodes_per_seed",
        "eval_budget",
        "horizon",
        "receding_horizon",
        "action_block",
        "precision",
        "autocast",
        "tf32",
        "compile",
    ):
        if config["protocol"].get(key) != reference_config["protocol"].get(key):
            raise RuntimeError(f"refinement config changed frozen Table 1 protocol.{key}")

    input_manifest_source = reference_root / "provenance/input_datasets.json"
    input_manifest = _read_json(input_manifest_source)
    if not input_manifest:
        raise FileNotFoundError("frozen Table 1 input dataset hashes are missing")
    current_paths = base._dataset_paths(config)
    for task, record in input_manifest["tasks"].items():
        path = current_paths[task]
        if (
            str(path) != record["resolved_path"]
            or not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or config["resources"]["input_datasets"][task] != record["configured_path"]
        ):
            raise RuntimeError(f"Table 1 frozen dataset identity changed for {task}: {path}")
    base._copy_independent(input_manifest_source, root / "provenance/input_datasets.json")

    diagnostics_source = reference_root / "cohorts/cohort_diagnostics.json"
    diagnostics = _read_json(diagnostics_source)
    if not diagnostics or diagnostics.get("task_seed_cohort_sha256") is None:
        raise RuntimeError("frozen Table 1 cohort diagnostics are incomplete")
    base._copy_independent(diagnostics_source, root / "cohorts/cohort_diagnostics.json")
    from source.common.round3_phase1 import CohortManifest

    for task in TASKS:
        for seed in SEEDS:
            source = reference_root / "cohorts" / task / f"seed_{seed}.json"
            destination = root / "cohorts" / task / f"seed_{seed}.json"
            base._copy_independent(source, destination)
            manifest = CohortManifest.load(destination)
            expected = diagnostics["task_seed_cohort_sha256"][task][str(seed)]
            if manifest.computed_sha256 != expected:
                raise RuntimeError(f"copied Table 1 cohort hash mismatch: {destination}")

    base._snapshot_provenance(root, config)
    source_revision = _read_json(root / "provenance/active_source_revision.json")
    assets = _read_json(root / "provenance/assets_manifest.json")
    has_evaluation_results = any(
        (root / "refinement/evaluations").glob("*/attempt_*/result.json")
    )
    has_timing_results = any(
        (root / "refinement/timing").glob("*/result.json")
    )
    freeze = {
        "status": "frozen",
        "configuration_sha256": hashlib.sha256(base._json_bytes(config)).hexdigest(),
        "dataset_hashes_frozen": True,
        "cohorts_frozen": True,
        "assets_frozen": True,
        "matrix": {
            "p3_reproduction_cells": 24,
            "new_base_score_cells": 18,
            "trained_score_cells": 96,
            "refinement_cells": 72,
            "planned_evaluation_cells": 210,
            "planned_episodes": 10500,
            "training_runs": 4,
        },
        "timing_conditions": 40,
        "asset_file_count": len(assets["files"]),
        "dataset_sha256": {
            task: record["sha256"] for task, record in input_manifest["tasks"].items()
        },
        "cohort_manifest_count": len(TASKS) * len(SEEDS),
        "cross_seed_overlap_recorded": bool(
            diagnostics.get("cross_seed_start_overlap_counts")
        ),
        "source_revision_id": source_revision["revision_id"],
        "formal_evaluations_started": bool(
            previous_freeze.get("formal_evaluations_started") or has_evaluation_results
        ),
        "formal_evaluations_started_utc": previous_freeze.get(
            "formal_evaluations_started_utc"
        ),
        "timing_started": bool(previous_freeze.get("timing_started") or has_timing_results),
        "timing_started_utc": previous_freeze.get("timing_started_utc"),
    }
    _atomic_json(root / "freeze.json", freeze)
    print(json.dumps({"root": str(root), "cohorts": 24, "assets": freeze["asset_file_count"], "source_revision_id": source_revision["revision_id"]}, indent=2))


def command_reproduce(args) -> None:
    root = _root(args)
    _preflight(args.gpu)
    from source.common.eval import get_dataset

    for task in TASKS:
        checkpoint = _base_checkpoint(root, task)
        model = _load_model(checkpoint, "cuda:0")
        method = {
            "method_id": "cowm_p3__main",
            "id": "CoWM-Selection",
            "label": "CoWM-Selection",
            "family": "cowm",
            "stage": 1,
            "mode": "P3",
            "guidance_mode": "none",
            "action_flow_steps": 2,
            "action_bound_mode": "none",
            "cem_protocol": "not_applicable",
            "guidance_inner_steps": 1,
            "guidance_last_steps": 2,
        }
        dataset = None
        try:
            for seed in SEEDS:
                cell = _cell(method, task, seed)
                if dataset is None:
                    cfg = base._cell_config(cell)
                    dataset = get_dataset(cfg, cfg.eval.dataset_name)
                _run_eval_cell(
                    root=root,
                    gpu=args.gpu,
                    cell=cell,
                    checkpoint=checkpoint,
                    model=model,
                    dataset=dataset,
                )
                print(f"complete {cell['cell_id']}", flush=True)
        finally:
            del model, dataset
            import torch

            torch.cuda.empty_cache()
    _verify_p3_replication(root)


def _first_complete_result(parent: Path) -> dict[str, Any]:
    for result_path in sorted(parent.glob("attempt_*/result.json"), reverse=True):
        result = _read_json(result_path)
        trace_path = result_path.parent / "episodes.jsonl"
        if (
            result
            and result.get("status") == "ok"
            and len(result.get("episodes", [])) == 50
            and trace_path.is_file()
            and result.get("cvpr_table1", {}).get("trace_sha256") == _sha(trace_path)
        ):
            return result
    raise RuntimeError(f"no complete trace-verified evaluation exists under {parent}")


def _p3_selection_events(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    events = []
    for batch in result.get("planning_events", []):
        for event in batch.get("algorithm_events", []):
            events.append(
                {
                    "candidate_noise_sha256": event.get("candidate_noise_sha256"),
                    "selected_indices": event.get("selected_indices"),
                    "replan_indices": event.get("replan_indices"),
                    "candidate_count": event.get("candidate_count"),
                    "costs": event.get("costs"),
                }
            )
    return events


def _verify_p3_replication(root: Path) -> None:
    verification = {"status": "complete", "cells": {}}
    official_root = ROOT / "outputs/cvpr/table1/v1/runs"
    for task in TASKS:
        for seed in SEEDS:
            cell_id = f"cowm_p3__main__{task}__seed_{seed}"
            current = _first_complete_result(root / "refinement/evaluations" / cell_id)
            reference = _first_complete_result(official_root / cell_id)
            current_success = [bool(item["success"]) for item in current["episodes"]]
            reference_success = [bool(item["success"]) for item in reference["episodes"]]
            current_events = _p3_selection_events(current)
            reference_events = _p3_selection_events(reference)
            same_selection_signature = len(current_events) == len(reference_events)
            initial_costs_equal = False
            later_cost_drift = []
            if same_selection_signature:
                initial_costs_equal = bool(current_events)
                for event_index, (now, before) in enumerate(zip(current_events, reference_events)):
                    for key in ("candidate_noise_sha256", "selected_indices", "replan_indices", "candidate_count"):
                        if now[key] != before[key]:
                            same_selection_signature = False
                            break
                    if not same_selection_signature:
                        break
                    current_costs = now.get("costs")
                    reference_costs = before.get("costs")
                    costs_equal = current_costs is None and reference_costs is None
                    if current_costs is not None and reference_costs is not None:
                        costs_equal = bool(
                            np.allclose(
                                np.asarray(current_costs, dtype=np.float64),
                                np.asarray(reference_costs, dtype=np.float64),
                                atol=1e-7,
                                rtol=1e-5,
                            )
                        )
                    if event_index == 0:
                        initial_costs_equal = costs_equal
                    elif not costs_equal:
                        max_abs = None
                        if current_costs is not None and reference_costs is not None:
                            left = np.asarray(current_costs, dtype=np.float64)
                            right = np.asarray(reference_costs, dtype=np.float64)
                            if left.shape == right.shape:
                                max_abs = float(np.max(np.abs(left - right)))
                        later_cost_drift.append(
                            {"event_index": event_index, "max_abs_cost_delta": max_abs}
                        )
            same_cohort = current.get("cohort_sha256") == reference.get("cohort_sha256")
            same_success = current_success == reference_success
            verification["cells"][cell_id] = {
                "cohort_equal": same_cohort,
                "candidate_noise_and_selected_indices_equal": same_selection_signature,
                "initial_64_candidate_costs_equal": initial_costs_equal,
                "later_replan_cost_drift": later_cost_drift,
                "success_vector_equal": same_success,
                "successes": sum(current_success),
            }
            if not (
                same_cohort
                and same_selection_signature
                and initial_costs_equal
                and same_success
            ):
                verification["status"] = "failed"
                _atomic_json(root / "refinement/reproduction_verification.json", verification)
                raise RuntimeError(
                    f"Table 1 P3 reproduction mismatch in {cell_id}: "
                    f"cohort={same_cohort}, candidate/selection={same_selection_signature}, "
                    f"initial_costs={initial_costs_equal}, success_vector={same_success}; "
                    "stop before refinement experiments"
                )
    _atomic_json(root / "refinement/reproduction_verification.json", verification)
    print("P3 reproduction verified against the frozen Table 1 run.", flush=True)


def command_cache(args) -> None:
    root = _root(args)
    _preflight(args.gpu)
    checkpoint, _ = _training_source(root)
    settings = _refine_config()["training"]
    receipt = prepare_offline_cache(
        checkpoint=checkpoint,
        cohort_paths=_cohort_paths(root),
        output_dir=root / "training/reacher/offline_cache",
        device="cuda:0",
        split_seed=int(settings["train_split_seed"]),
        cache_seed=int(settings["cache_seed"]),
        split_fraction=float(settings["train_split_fraction"]),
        trajectories=int(settings["trajectories"]),
        windows_per_trajectory=int(settings["windows_per_trajectory"]),
    )
    print(json.dumps({key: receipt[key] for key in ("cache_path", "cache_sha256", "window_count", "distance_scale")}, indent=2))


def command_train(args) -> None:
    root = _root(args)
    _preflight(args.gpu)
    _mark_freeze_activity(root, "formal_evaluations_started")
    settings = _refine_config()["training"]
    checkpoint, checkpoint_sha = _training_source(root)
    cache_path = root / "training/reacher/offline_cache/offline_latents.pt"
    if not cache_path.is_file():
        raise FileNotFoundError("run prepare-cache before training")
    out = root / "training/reacher" / args.arm / f"seed_{int(args.seed)}"
    receipt = train_checkpoint(
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        cache_path=cache_path,
        output_dir=out,
        arm=args.arm,
        seed=int(args.seed),
        device="cuda:0",
        updates=int(settings["updates"]),
        checkpoint_updates=tuple(int(value) for value in settings["checkpoint_updates"]),
        batch_size=int(settings["batch_size"]),
        learning_rate=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        gradient_clip=float(settings["gradient_clip"]),
        cost_loss_weight=float(settings["cost_loss_weight"]),
        temporal_loss_weight=float(settings["temporal_loss_weight"]),
        actor_preserve_loss_weight=float(settings["actor_preserve_loss_weight"]),
        ranking_margin=float(settings["ranking_margin"]),
    )
    print(json.dumps({"arm": args.arm, "seed": args.seed, "checkpoints": receipt["checkpoints"]}, indent=2))


def _training_checkpoint(root: Path, arm: str, seed: int) -> tuple[Path, Path]:
    run = root / "training/reacher" / arm / f"seed_{int(seed)}"
    receipt_path = run / "training_receipt.json"
    receipt = _read_json(receipt_path)
    if not receipt or int(receipt.get("updates", 0)) != int(_refine_config()["training"]["updates"]):
        raise RuntimeError(f"training receipt is missing or incomplete: {receipt_path}")
    target_suffix = f"_weights_epoch_{int(receipt['updates'])}.pt"
    paths = [Path(value) for value in receipt["checkpoints"] if value.endswith(target_suffix)]
    if len(paths) != 1 or not paths[0].is_file():
        raise RuntimeError(f"final training checkpoint is missing from {receipt_path}")
    return paths[0], receipt_path


def command_score(args) -> None:
    root = _root(args)
    _preflight(args.gpu)
    specs = _score_specs()
    if args.checkpoint_id == "base":
        checkpoint = _base_checkpoint(root)
        _run_reacher_score_cells(
            root=root,
            gpu=args.gpu,
            checkpoint=checkpoint,
            checkpoint_id="base",
            score_specs=[item for item in specs if item["id"] != "endpoint"],
        )
        return
    if not args.arm or args.seed is None:
        raise ValueError("trained score runs require --arm and --seed")
    checkpoint, receipt = _training_checkpoint(root, args.arm, args.seed)
    checkpoint_id = f"{args.arm}_seed{args.seed}"
    _run_reacher_score_cells(
        root=root,
        gpu=args.gpu,
        checkpoint=checkpoint,
        checkpoint_id=checkpoint_id,
        score_specs=specs,
        training_receipt=receipt,
    )


def _all_result_attempts(root: Path):
    latest_by_cell = {}
    for path in sorted((root / "refinement/evaluations").glob("*/attempt_*/result.json")):
        result = _read_json(path)
        if result and result.get("status") == "ok":
            meta = result.get("cvpr_table1", {})
            trace = path.parent / "episodes.jsonl"
            if (
                trace.is_file()
                and meta.get("trace_sha256") == _sha(trace)
                and len(result.get("episodes", [])) == 50
            ):
                cell = result.get("cell")
                if cell is None:
                    cell = _read_json(path.parent / "cell_identity.json", {}).get("cell")
                if cell is None:
                    continue
                result = _publish_refine_fields(
                    result,
                    cell,
                    result.get("refinement_identity", {}),
                    result.get("refinement_identity_sha256", meta.get("identity_sha256", "")),
                    path,
                )
                cell_id = str(cell["cell_id"])
                attempt_number = int(path.parent.name.removeprefix("attempt_"))
                previous = latest_by_cell.get(cell_id)
                if previous is None or attempt_number > previous[0]:
                    latest_by_cell[cell_id] = (attempt_number, result)
    yield from (item[1] for _cell_id, item in sorted(latest_by_cell.items()))


def _aggregate(result_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed = {int(record["evaluation_seed"]): record for record in result_records}
    if set(by_seed) != set(SEEDS) or len(by_seed) != len(SEEDS):
        raise RuntimeError(f"expected one result for each Table 1 seed, got {sorted(by_seed)}")
    rates = [float(by_seed[seed]["success_rate"]) for seed in SEEDS]
    return {
        "successes": sum(int(by_seed[seed]["successes"]) for seed in SEEDS),
        "episodes": sum(int(by_seed[seed]["num_episodes"]) for seed in SEEDS),
        "per_seed": rates,
        "mean": float(np.mean(rates)),
        "std": float(np.std(rates, ddof=1)),
    }


def _selection_candidates(root: Path) -> list[dict[str, Any]]:
    records = list(_all_result_attempts(root))
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in records:
        cell = result.get("cell", {})
        if cell.get("task") != "reacher" or cell.get("guidance_mode", "none") != "none":
            continue
        label = cell.get("checkpoint_label", "base")
        score_id = cell.get("score_id", "endpoint")
        groups.setdefault(f"{label}:{score_id}", []).append(result)
    candidates = []
    for key, values in groups.items():
        try:
            summary = _aggregate(values)
        except RuntimeError:
            continue
        checkpoint_label, score_id = key.split(":", 1)
        candidates.append({"key": key, "checkpoint_label": checkpoint_label, "score_id": score_id, **summary})
    return candidates


def _time_record_path(root: Path, cell_id: str) -> Path:
    return root / "refinement/timing" / cell_id / "result.json"


def _time_cell(*, root: Path, gpu: int, cell: Mapping[str, Any], checkpoint: Path, model: Any, dataset=None):
    _mark_freeze_activity(root, "timing_started")
    config = _config()
    training_receipt = None
    checkpoint_label = cell.get("checkpoint_label")
    if checkpoint_label and checkpoint_label != "base":
        arm, seed_text = str(checkpoint_label).rsplit("_seed", 1)
        training_receipt = _training_checkpoint(root, arm, int(seed_text))[1]
    identity = _source_identity(
        root, cell, checkpoint, training_receipt=training_receipt
    )
    identity_sha = base.identity_hash({"cell": dict(cell), **identity, "timing": config["timing"]})
    result_path = _time_record_path(root, str(cell["cell_id"]))
    prior = _read_json(result_path)
    if prior and prior.get("identity_sha256") == identity_sha and prior.get("status") == "complete":
        return prior
    if dataset is None:
        from source.common.eval import get_dataset

        cfg = base._cell_config(cell)
        dataset = get_dataset(cfg, cfg.eval.dataset_name)
    cohort = CohortManifest.load(
        root / "cohorts" / str(cell["task"]) / f"seed_{int(cell['evaluation_seed'])}.json"
    )
    attempt = _next_attempt(result_path.parent)
    _preflight(gpu, timing=True)
    result = base._capture_and_measure_condition(
        root=root,
        cell=cell,
        identity_sha256=identity_sha,
        config=config,
        model=model,
        dataset=dataset,
        manifest=cohort,
        attempt_dir=attempt,
    )
    if result.get("status") != "complete":
        raise RuntimeError(f"timing condition did not complete: {cell['cell_id']}")
    _atomic_json(result_path, {**result, "identity_sha256": identity_sha, "checkpoint": str(checkpoint)})
    return result


def _preflight(gpu: int, timing=False) -> dict[str, Any]:
    config = _config()
    check = base._check_timing_idle if timing else base._check_dispatch_resources
    if timing:
        result = check(int(gpu), config)
    else:
        result = check(int(gpu), config)
    print(f"GPU {gpu} preflight: {json.dumps(result)}", flush=True)
    return result


def _selection_cells(root: Path) -> list[tuple[dict[str, Any], Path]]:
    cells = []
    base_path = _base_checkpoint(root)
    base_checkpoint_id = "base"
    for score in _score_specs():
        cell = _custom_cells(
            f"cvpr_refine_{base_checkpoint_id}_{score['id']}",
            f"CoWM P3 {base_checkpoint_id} {score['id']}",
            42,
            score_reduction=score["mode"],
            terminal_weight=float(score["terminal_weight"]),
            checkpoint_label=base_checkpoint_id,
            score_id=score["id"],
        )
        cells.append((cell, base_path))
    for arm in _refine_config()["training"]["arms"]:
        for seed in _refine_config()["training"]["seeds"]:
            checkpoint, _receipt = _training_checkpoint(root, arm, seed)
            label = f"{arm}_seed{seed}"
            for score in _score_specs():
                cell = _custom_cells(
                    f"cvpr_refine_{label}_{score['id']}",
                    f"CoWM P3 {label} {score['id']}",
                    42,
                    score_reduction=score["mode"],
                    terminal_weight=float(score["terminal_weight"]),
                    checkpoint_label=label,
                    score_id=score["id"],
                )
                cells.append((cell, checkpoint))
    return cells


def _original_timing_cells() -> list[dict[str, Any]]:
    from source.common.cvpr_table1 import all_methods

    methods = {str(method["method_id"]): method for method in all_methods()}
    cells = []
    for task in TASKS:
        for method_id in ("cowm_p3__main", "leflow__main"):
            cell = _cell(methods[method_id], task, 42)
            cell["cell_id"] = f"timing__{method_id}__{task}__seed_42"
            cells.append(cell)
    return cells


def command_time_selection(args) -> None:
    root = _root(args)
    _preflight(args.gpu, timing=True)
    from source.common.eval import get_dataset

    datasets = {}
    timed_cells = [cell for cell, _checkpoint in _selection_cells(root)] + _original_timing_cells()
    for cell in timed_cells:
        cell = {
            **cell,
            "evaluation_seed": 42,
            "environment_seed": 10042,
            "policy_seed": 20042,
        }
        if not cell["cell_id"].startswith("timing__"):
            cell["cell_id"] = f"timing__{cell['method_id']}__reacher__seed_42"
        if cell["method_id"] == "cowm_p3__main":
            checkpoint = _base_checkpoint(root, str(cell["task"]))
            model = _load_model(checkpoint, "cuda:0")
        elif cell["family"] == "leflow":
            checkpoint = _leflow_checkpoint(root, str(cell["task"]))
            model = base._load_models(root, str(cell["task"]), 1, "cuda:0")["leflow"]
        else:
            label = str(cell["checkpoint_label"])
            if label == "base":
                checkpoint = _base_checkpoint(root)
            else:
                arm, seed_text = label.rsplit("_seed", 1)
                checkpoint = _training_checkpoint(root, arm, int(seed_text))[0]
            model = _load_model(checkpoint, "cuda:0")
        try:
            task = str(cell["task"])
            if task not in datasets:
                cfg = base._cell_config(cell)
                datasets[task] = get_dataset(cfg, cfg.eval.dataset_name)
            _time_cell(
                root=root,
                gpu=args.gpu,
                cell=cell,
                checkpoint=checkpoint,
                model=model,
                dataset=datasets[task],
            )
            print(f"timed {cell['cell_id']}", flush=True)
        finally:
            del model
            import torch

            torch.cuda.empty_cache()


def _leflow_checkpoint(root: Path, task: str) -> Path:
    path = root / "assets/leflow" / task / "latent_planner.pt"
    if not path.is_file():
        raise FileNotFoundError(f"archived LeFlow planner is missing: {path}")
    return path


def command_lock_selection(args) -> None:
    root = _root(args)
    candidates = _selection_candidates(root)
    if not candidates:
        raise RuntimeError("no complete six-seed Selection candidates were found")
    timings = {}
    for cell, checkpoint in _selection_cells(root):
        label = cell["checkpoint_label"]
        score_id = cell["score_id"]
        timing_id = f"timing__{cell['method_id']}__reacher__seed_42"
        timed = _read_json(_time_record_path(root, timing_id))
        if timed and timed.get("status") == "complete":
            timings[f"{label}:{score_id}"] = float(timed["mean_ms"])
    base_means = {}
    leflow_means = {}
    for task in TASKS:
        for method_id, destination in (("cowm_p3__main", base_means), ("leflow__main", leflow_means)):
            cell_id = f"{method_id}__{task}__seed_42"
            timed = _read_json(_time_record_path(root, cell_id))
            if timed and timed.get("status") == "complete":
                destination[task] = float(timed["mean_ms"])
    leflow_mean = float(np.mean(list(leflow_means.values()))) if len(leflow_means) == 4 else None
    p3_other_sum = sum(base_means.get(task, float("inf")) for task in TASKS if task != "reacher")
    reacher_limit = None if leflow_mean is None else 4 * leflow_mean - p3_other_sum
    eligible = []
    for candidate in candidates:
        candidate["reacher_mean_ms"] = timings.get(candidate["key"])
        candidate["latency_eligible"] = (
            leflow_mean is not None
            and candidate["reacher_mean_ms"] is not None
            and candidate["reacher_mean_ms"] < reacher_limit
        )
        candidate["macro_mean"] = (
            (candidate["mean"] + sum(
                float(_summary_original_p3(root, task)["mean"])
                for task in TASKS if task != "reacher"
            )) / 4.0
        )
        if candidate["latency_eligible"]:
            eligible.append(candidate)
    if not eligible:
        raise RuntimeError("no Selection candidate has complete timing below the LeFlow macro latency")
    eligible.sort(
        key=lambda item: (
            -int(item["successes"]),
            float(item["std"]),
            float(item["reacher_mean_ms"]),
            0 if item["checkpoint_label"] == "base" else 1,
            item["key"],
        )
    )
    selected = eligible[0]
    checkpoint = (
        _base_checkpoint(root)
        if selected["checkpoint_label"] == "base"
        else _training_checkpoint(
            root,
            selected["checkpoint_label"].rsplit("_seed", 1)[0],
            int(selected["checkpoint_label"].rsplit("_seed", 1)[1]),
        )[0]
    )
    score = next(item for item in _score_specs() if item["id"] == selected["score_id"])
    lock = {
        "schema_version": 1,
        "selection": selected,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha(checkpoint),
        "score": score,
        "latency_gate": {
            "leflow_macro_mean_ms": leflow_mean,
            "reacher_max_mean_ms": reacher_limit,
            "p3_other_task_mean_ms": base_means,
        },
        "all_candidates": candidates,
    }
    _atomic_json(root / "refinement/selection_lock.json", lock)
    print(json.dumps({"selected": selected, "checkpoint": str(checkpoint), "score": score}, indent=2))


def _summary_original_p3(root: Path, task: str) -> dict[str, Any]:
    records = []
    for seed in SEEDS:
        cell_id = f"cowm_p3__main__{task}__seed_{seed}"
        matches = list((root / "refinement/evaluations" / cell_id).glob("attempt_*/result.json"))
        if not matches:
            raise RuntimeError(f"original P3 replication is missing: {cell_id}")
        result = _read_json(sorted(matches)[-1])
        records.append(result)
    return _aggregate(records)


def command_refine(args) -> None:
    root = _root(args)
    _preflight(args.gpu)
    lock = _read_json(root / "refinement/selection_lock.json")
    if not lock:
        raise FileNotFoundError("run lock-selection before evaluating refinement")
    checkpoint = Path(lock["checkpoint"]).resolve(strict=True)
    if _sha(checkpoint) != lock["checkpoint_sha256"]:
        raise RuntimeError("locked Selection checkpoint hash has changed")
    score = lock["score"]
    specs = _refine_config()["refinement"]
    from source.common.eval import get_dataset

    first = specs[0]
    seed_cell = _custom_cells(
        "refine_timing_seed",
        "refine timing",
        42,
        score_reduction=score["mode"],
        terminal_weight=float(score["terminal_weight"]),
        guidance_mode=first["guidance"],
        guidance_inner_steps=int(first["inner_steps"]),
        guidance_last_steps=int(first.get("last_steps", 2)),
        guidance_step_size=float(first["step_size"]),
    )
    cfg = base._cell_config(seed_cell)
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    model = _load_model(checkpoint, "cuda:0")
    try:
        for spec in specs:
            for seed in SEEDS:
                cell = _custom_cells(
                    f"cvpr_refine_final_{spec['id']}",
                    f"CoWM P3 + {spec['id']}",
                    seed,
                    score_reduction=score["mode"],
                    terminal_weight=float(score["terminal_weight"]),
                    guidance_mode=spec["guidance"],
                    guidance_inner_steps=int(spec["inner_steps"]),
                    guidance_last_steps=int(spec.get("last_steps", 2)),
                    guidance_step_size=float(spec["step_size"]),
                    checkpoint_label=lock["selection"]["checkpoint_label"],
                    score_id=score["id"],
                    refinement_id=spec["id"],
                )
                _run_eval_cell(
                    root=root,
                    gpu=args.gpu,
                    cell=cell,
                    checkpoint=checkpoint,
                    model=model,
                    training_receipt=(
                        None
                        if lock["selection"]["checkpoint_label"] == "base"
                        else _training_checkpoint(
                            root,
                            lock["selection"]["checkpoint_label"].rsplit("_seed", 1)[0],
                            int(lock["selection"]["checkpoint_label"].rsplit("_seed", 1)[1]),
                        )[1]
                    ),
                    dataset=dataset,
                )
                print(f"complete {cell['cell_id']}", flush=True)
    finally:
        del model, dataset
        import torch

        torch.cuda.empty_cache()


def command_time_refinement(args) -> None:
    root = _root(args)
    _preflight(args.gpu, timing=True)
    lock = _read_json(root / "refinement/selection_lock.json")
    if not lock:
        raise FileNotFoundError("run lock-selection before timing refinement")
    checkpoint = Path(lock["checkpoint"]).resolve(strict=True)
    if _sha(checkpoint) != lock["checkpoint_sha256"]:
        raise RuntimeError("locked Selection checkpoint hash has changed")
    score = lock["score"]
    specs = _refine_config()["refinement"]
    from source.common.eval import get_dataset

    first = specs[0]
    seed_cell = _custom_cells(
        "refine_timing_seed",
        "refine timing",
        42,
        score_reduction=score["mode"],
        terminal_weight=float(score["terminal_weight"]),
        guidance_mode=first["guidance"],
        guidance_inner_steps=int(first["inner_steps"]),
        guidance_last_steps=int(first.get("last_steps", 2)),
        guidance_step_size=float(first["step_size"]),
    )
    cfg = base._cell_config(seed_cell)
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    model = _load_model(checkpoint, "cuda:0")
    try:
        for spec in specs:
            cell = _custom_cells(
                f"cvpr_refine_final_{spec['id']}",
                f"CoWM P3 + {spec['id']}",
                42,
                score_reduction=score["mode"],
                terminal_weight=float(score["terminal_weight"]),
                guidance_mode=spec["guidance"],
                guidance_inner_steps=int(spec["inner_steps"]),
                guidance_last_steps=int(spec.get("last_steps", 2)),
                guidance_step_size=float(spec["step_size"]),
                checkpoint_label=lock["selection"]["checkpoint_label"],
                score_id=score["id"],
                refinement_id=spec["id"],
            )
            cell.update(
                evaluation_seed=42,
                environment_seed=10042,
                policy_seed=20042,
                cell_id=f"timing__cvpr_refine_final_{spec['id']}__reacher__seed_42",
            )
            _time_cell(
                root=root,
                gpu=args.gpu,
                cell=cell,
                checkpoint=checkpoint,
                model=model,
                dataset=dataset,
            )
            print(f"timed {cell['cell_id']}", flush=True)
    finally:
        del model, dataset
        import torch

        torch.cuda.empty_cache()


def _base_alias_summary(root: Path, method: str, task: str):
    if method == "p3":
        return _summary_original_p3(root, task)
    raise ValueError(method)


def command_report(args) -> None:
    root = _root(args)
    lock = _read_json(root / "refinement/selection_lock.json")
    if not lock:
        raise FileNotFoundError("Selection lock is missing")
    selection = lock["selection"]
    label = selection["checkpoint_label"]
    score_id = selection["score_id"]
    all_results = list(_all_result_attempts(root))
    result_map = {}
    refinement_scan = []
    for task in TASKS:
        if task != "reacher":
            result_map[(task, "selection")] = _base_alias_summary(root, "p3", task)
            result_map[(task, "refinement")] = result_map[(task, "selection")]
            continue
        selection_cells = [
            result for result in all_results
            if result.get("task") == "reacher"
            and result.get("cell", {}).get("checkpoint_label") == label
            and result.get("cell", {}).get("score_id") == score_id
            and result.get("cell", {}).get("guidance_mode", "none") == "none"
        ]
        result_map[(task, "selection")] = _aggregate(selection_cells)
        refine_cells = [
            result for result in all_results
            if result.get("task") == "reacher"
            and result.get("cell", {}).get("refinement_id")
        ]
        eligible_refinements = []
        latency_limit = lock["latency_gate"].get("reacher_max_mean_ms")
        for spec in _refine_config()["refinement"]:
            rows = [result for result in refine_cells if result["cell"].get("refinement_id") == spec["id"]]
            try:
                summary = _aggregate(rows)
            except RuntimeError:
                refinement_scan.append({"id": spec["id"], "complete": False})
                continue
            timing_id = f"timing__cvpr_refine_final_{spec['id']}__reacher__seed_42"
            timing = _read_json(_time_record_path(root, timing_id))
            if not timing or timing.get("status") != "complete" or latency_limit is None:
                refinement_scan.append(
                    {
                        "id": spec["id"],
                        "complete": True,
                        "timing_complete": False,
                        "summary": summary,
                    }
                )
                continue
            summary["reacher_mean_ms"] = float(timing["mean_ms"])
            summary["latency_eligible"] = summary["reacher_mean_ms"] < float(latency_limit)
            summary["refinement_id"] = spec["id"]
            summary["update_count"] = (
                int(spec["inner_steps"]) * 2
                if spec["guidance"] == "guided_flow"
                else int(spec["inner_steps"])
            )
            summary["complete"] = True
            summary["timing_complete"] = True
            summary["latency_limit_ms"] = float(latency_limit)
            refinement_scan.append({"id": spec["id"], **summary})
            if (
                summary["latency_eligible"]
                and summary["successes"] > result_map[(task, "selection")]["successes"]
            ):
                eligible_refinements.append(summary)
        eligible_refinements.sort(
            key=lambda item: (
                -int(item["successes"]),
                int(item["update_count"]),
                float(item["std"]),
                float(item["reacher_mean_ms"]),
                str(item["refinement_id"]),
            )
        )
        best_refinement = eligible_refinements[0] if eligible_refinements else None
        result_map[(task, "refinement")] = best_refinement or result_map[(task, "selection")]
    rows = []
    for method, title in (("selection", "CoWM-Selection"), ("refinement", "CoWM-Selection + Refinement")):
        means = [result_map[(task, method)]["mean"] for task in TASKS]
        macro_per_seed = [
            sum(result_map[(task, method)]["per_seed"][index] for task in TASKS) / 4.0
            for index in range(len(SEEDS))
        ]
        rows.append((title, means, float(np.mean(macro_per_seed)), float(np.std(macro_per_seed, ddof=1))))
    chosen_refinement = result_map[("reacher", "refinement")]
    if chosen_refinement.get("refinement_id"):
        refinement_note = (
            f"Refinement最终选择 `{chosen_refinement['refinement_id']}`，"
            f"成功数 {chosen_refinement['successes']}/300，"
            f"均延迟 {chosen_refinement.get('reacher_mean_ms')} ms；"
            f"Reacher动态延迟上限 {lock['latency_gate'].get('reacher_max_mean_ms')} ms。"
        )
    else:
        refinement_note = "没有 refinement 配置同时超过 Selection 成功数并通过预设延迟门槛；第二行回退为 Selection。"
    lines = [
        "# CVPR Table 1 CoWM refinement 实验报告",
        "",
        f"方案锁定权重：`{lock['checkpoint']}`（SHA256 `{lock['checkpoint_sha256']}`）。",
        f"Reacher评分：`{score_id}`；终点权重 `{lock['score']['terminal_weight']}`。",
        f"Selection六seed成功数：{selection['successes']}/300；Reacher实测均延迟 `{selection.get('reacher_mean_ms')} ms`。",
        "评测使用Table 1六个seed、每seed50 episodes、目标偏移25、执行预算50、25/25反馈。",
        "Reacher以外三个任务复用本轮重跑的原P3单元。seed标准差来自评测seed；六seed已用于benchmark方案选择。",
        refinement_note,
        "",
        "| 方法 | TwoRoom | PushT | Reacher | OGBench-Cube | 四任务平均 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for title, means, macro, std in rows:
        shown = [f"{100*value:.2f}%" for value in means]
        lines.append(f"| {title} | " + " | ".join(shown + [f"{100*macro:.2f} ± {100*std:.2f}%"]) + " |")
    p3 = _summary_original_p3(root, "reacher")
    lines.extend([
        "",
        "## Reacher逐seed成功率",
        "",
        "| seed | 原P3 | Selection | Selection + Refinement |",
        "|---:|---:|---:|---:|",
    ])
    for index, seed in enumerate(SEEDS):
        p = p3["per_seed"][index]
        a = result_map[("reacher", "selection")]["per_seed"][index]
        b = result_map[("reacher", "refinement")]["per_seed"][index]
        lines.append(f"| {seed} | {100*p:.1f}% | {100*a:.1f}% | {100*b:.1f}% |")
    lines.extend(
        [
            "",
            "## Reacher refinement 扫描",
            "",
            "| 配置 | 完成 | 成功数/300 | 六seed成功率 | 相对Selection | 均延迟(ms) | 延迟合格 |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for candidate in refinement_scan:
        summary = candidate.get("summary", candidate)
        if not candidate.get("complete") or not candidate.get("timing_complete"):
            lines.append(f"| {candidate['id']} | 否 | — | — | — | — | 未完成 |")
            continue
        eligible = bool(candidate.get("latency_eligible"))
        selected_mark = "（最终选择）" if chosen_refinement.get("refinement_id") == candidate["id"] else ""
        lines.append(
            f"| {candidate['id']}{selected_mark} | 是 | {summary['successes']} | "
            f"{100*summary['mean']:.2f} ± {100*summary['std']:.2f}% | "
            f"{summary['successes'] - selection['successes']:+d} | "
            f"{summary['reacher_mean_ms']:.3f} | {'是' if eligible else '否'} |"
        )
    lines.extend(
        [
            "",
            "## Selection候选与延迟门槛",
            "",
            "```json",
            json.dumps(lock, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(str(REPORT_PATH))


def command_status(args) -> None:
    root = _root(args)
    results = list(_all_result_attempts(root))
    training = list((root / "training/reacher").glob("*/seed_*/training_receipt.json"))
    timing = list((root / "refinement/timing").glob("*/result.json"))
    print(json.dumps({"root": str(root), "complete_eval_cells": len(results), "training_receipts": len(training), "timing_conditions": len(timing), "selection_lock": (root / "refinement/selection_lock.json").is_file()}, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (
        ("prepare", command_prepare),
        ("reproduce", command_reproduce),
        ("verify-reproduction", lambda args: _verify_p3_replication(_root(args))),
        ("prepare-cache", command_cache),
        ("train", command_train),
        ("score", command_score),
        ("time-selection", command_time_selection),
        ("lock-selection", command_lock_selection),
        ("refine", command_refine),
        ("time-refinement", command_time_refinement),
        ("report", command_report),
        ("status", command_status),
    ):
        sub = commands.add_parser(name)
        sub.set_defaults(func=function)
        if name not in {"prepare", "verify-reproduction", "status", "report", "lock-selection"}:
            sub.add_argument("--gpu", type=int, required=True)
        if name == "train":
            sub.add_argument("--arm", choices=("b_specific", "shared_preserve"), required=True)
            sub.add_argument("--seed", type=int, choices=(3072, 4096), required=True)
        if name == "score":
            sub.add_argument("--checkpoint-id", choices=("base", "trained"), default="base")
            sub.add_argument("--arm", choices=("b_specific", "shared_preserve"))
            sub.add_argument("--seed", type=int, choices=(3072, 4096))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
