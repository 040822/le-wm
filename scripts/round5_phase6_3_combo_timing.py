#!/usr/bin/env python3
"""Formally time selected Phase 6.3 combinations with adjacent E0 brackets."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cvpr_table1 as table
from scripts import round5_phase6_3 as phase
from scripts import round5_phase6_3_timing as timing
from scripts.round5_phase6_3_combos import (
    COMBO_VERSIONS,
    entry_for,
    closed_loop_e4_fingerprint,
    enter_components,
    load_selection,
)
from source.common.cvpr_table1 import TASKS, timing_conditions
from source.common.round5_phase6_3 import (
    deferred_refinement_checks,
    encoder_bf16,
    fixed_condition_cache,
)

RUNS = phase.OUT / "timing/combinations/runs"
PROGRESS = phase.OUT / "timing/combinations/formal_progress.json"


def _verify_inputs(frozen: dict, selection_sha256: str) -> list[dict]:
    if phase.source_hashes() != frozen["source_hashes"]:
        raise RuntimeError("source differs from the frozen Phase 6.3 snapshot")
    if table._sha256(phase.MAIN / "frozen_config.json") != frozen["main_table_frozen_config_sha256"]:
        raise RuntimeError("main Table 1 frozen configuration changed")
    selected_path = phase.OUT / "summary/selected_combinations.json"
    if table._sha256(selected_path) != selection_sha256:
        raise RuntimeError("combination selection changed after it was frozen")
    refs = table._read_json(phase.OUT / "provenance/assets_and_cohorts.json")
    for ref in refs:
        if table._sha256(Path(ref["path"])) != ref["sha256"]:
            raise RuntimeError(f"frozen asset or cohort changed: {ref['path']}")
    return refs


def _selected_cells(selection: dict) -> dict[str, list[tuple[dict, list[str]]]]:
    by_version = {version: [] for version in COMBO_VERSIONS}
    for cell in timing_conditions():
        if cell["family"] != "cowm" or cell["id"] not in phase.METHODS:
            continue
        if not cell["method_id"].endswith("__main"):
            continue
        for version in COMBO_VERSIONS:
            choice = entry_for(selection, version, cell["id"])
            if choice["status"] == "selected":
                by_version[version].append((cell, list(choice["components"])))
    return by_version


def _identity(cell, *, owner_version, block_version, block, components,
              selection_sha256, driver_sha, frozen, refs):
    timed_cell = {**cell, "_timing_version": block_version, "_timing_block": block}
    evidence_version = block_version
    if block_version in COMBO_VERSIONS:
        evidence_version = "E4" if "E4" in components else "E0"
    identity = timing._identity(
        phase.OUT, timed_cell, evidence_version, block, driver_sha, frozen, refs
    )
    identity.update({
        "version": block_version,
        "combo_owner_version": owner_version,
        "combo_components": list(components),
        "active_components": list(components) if block_version in COMBO_VERSIONS else [],
        "combination_selection_sha256": selection_sha256,
        "version_switches": {
            **identity["version_switches"],
            "combination_components": list(components),
        },
        "effective_encoder_precision": (
            "bfloat16_encoder" if block_version in COMBO_VERSIONS and "E4" in components
            else "float32"
        ),
    })
    return timed_cell, identity


def _attempt_dir(version: str, block: str, cell_id: str, identity_sha256: str) -> Path | None:
    parent = RUNS / version / block / cell_id
    for path in sorted(parent.glob("attempt_*/result.json"), reverse=True) if parent.exists() else []:
        result = table._read_json(path, {})
        if (result.get("status") == "complete" and result.get("formal_latency") is True
                and result.get("identity_sha256") == identity_sha256):
            return path.parent
    return None


def _run_condition(*, args, config, version, owner_version, cell, components,
                   selection_sha256, driver_sha, frozen, refs, loaded, datasets,
                   manifest_cache, progress):
    from source.common.round3_phase1 import CohortManifest
    from source.common.eval import get_dataset
    from source.common.checkpoint import load_policy_or_model
    import torch

    table._check_timing_idle(args.gpu, config)
    block_index = {"E0-before": "00_E0", "combo": f"01_{version}",
                   "E0-after": "02_E0"}[progress["current_segment"]]
    block_version = "E0" if progress["current_segment"].startswith("E0") else version
    timed_cell, identity = _identity(
        cell,
        owner_version=owner_version,
        block_version=block_version,
        block=block_index,
        components=components,
        selection_sha256=selection_sha256,
        driver_sha=driver_sha,
        frozen=frozen,
        refs=refs,
    )
    identity_sha = table.identity_hash(identity)
    cell_id = str(cell["cell_id"])
    existing = _attempt_dir(owner_version, block_index, cell_id, identity_sha)
    if existing is not None:
        progress["completed_conditions"] += 1
        progress["rows"].append({"version": owner_version, "cell_id": cell_id,
                                 "block": block_index, "status": "complete_reused",
                                 "result": str((existing / "result.json").relative_to(ROOT))})
        table._write_json(PROGRESS, progress)
        print(f"skip complete {owner_version}/{block_index}/{cell_id}", flush=True)
        return

    parent = RUNS / owner_version / block_index / cell_id
    prior_numbers = []
    for path in parent.glob("attempt_*") if parent.exists() else []:
        try:
            prior_numbers.append(int(path.name.removeprefix("attempt_")))
        except ValueError:
            pass
    attempt = parent / f"attempt_{max(prior_numbers, default=0) + 1:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        command = {
            "argv": list(sys.argv),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "driver_sha256": driver_sha,
            "preflight_before_model_load": phase.preflight(args.gpu, config),
        }
        task = str(cell["task"])
        if task not in loaded:
            checkpoint = phase.MAIN / "assets/cowm" / task / "checkpoints/r4_ab_seed3072_weights_epoch_10.pt"
            package, _ = load_policy_or_model(str(checkpoint), cache_dir=str(ROOT / "data"))
            loaded[task] = getattr(package, "model", package).eval().requires_grad_(False)
            del package
        if task not in datasets:
            cfg = table._cell_config(cell)
            datasets[task] = get_dataset(cfg, cfg.eval.dataset_name)
        if task not in manifest_cache:
            manifest_cache[task] = CohortManifest.load(
                phase.MAIN / "cohorts" / task / "seed_42.json"
            )

        version_config = copy.deepcopy(config)
        if block_version != "E0" and "E4" in components:
            version_config["protocol"]["precision"] = "bfloat16_encoder"
        original_config_loader = table._load_config
        table._load_config = lambda: version_config
        with ExitStack() as stack:
            if block_version != "E0":
                enter_components(
                    stack,
                    components,
                    preprocess=phase.preprocess_factory,
                    encoder=lambda: encoder_bf16(loaded[task]),
                    cache=lambda: fixed_condition_cache(loaded[task]),
                    deferred_checks=lambda: deferred_refinement_checks(loaded[task]),
                )
            measurement_preflight = table._check_timing_idle(args.gpu, version_config)
            result = table._capture_and_measure_condition(
                root=phase.OUT,
                cell=timed_cell,
                identity_sha256=identity_sha,
                config=version_config,
                model=loaded[task],
                dataset=datasets[task],
                manifest=manifest_cache[task],
                attempt_dir=attempt,
            )
        table._load_config = original_config_loader
        result.update({
            "version": block_version,
            "combo_owner_version": owner_version,
            "combo_components": list(components),
            "active_components": list(components) if block_version in COMBO_VERSIONS else [],
            "effective_encoder_precision": identity["effective_encoder_precision"],
            "execution_block": block_index,
            "formal_latency": True,
            "gpu_before": command["preflight_before_model_load"],
            "gpu_before_measurement": measurement_preflight,
            "gpu_after": result.get("gpu_after"),
            "source_revision": frozen["source_revision"],
            "timing_driver_sha256": driver_sha,
            "combination_selection_sha256": selection_sha256,
            "identity": identity,
            "identity_sha256": identity_sha,
        })
        if not result.get("gpu_after") or result["gpu_after"].get("gpu_uuid") != command["preflight_before_model_load"].get("gpu_uuid"):
            raise RuntimeError("GPU identity changed during the formal condition")
        table._write_json(attempt / "cell_identity.json", {
            "identity": identity, "identity_sha256": identity_sha,
            "attempt": attempt.name, "command": command,
        })
        table._write_json(attempt / "result.json", result)
        progress["completed_conditions"] += 1
        progress["rows"].append({"version": owner_version, "cell_id": cell_id,
                                 "block": block_index, "status": "complete",
                                 "result": str((attempt / "result.json").relative_to(ROOT))})
        table._write_json(PROGRESS, progress)
        print(f"formal {owner_version}/{block_index}/{cell_id}: {result['mean_ms']:.4f} ms", flush=True)
    except BaseException as exc:
        table._write_json(attempt / "failure.json", {
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "identity": identity,
            "identity_sha256": identity_sha,
        })
        raise
    finally:
        table._load_config = original_config_loader if "original_config_loader" in locals() else table._load_config


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise RuntimeError(f"launch with CUDA_VISIBLE_DEVICES={args.gpu}")
    frozen = table._read_json(phase.OUT / "frozen_config.json")
    selection, selection_sha256 = load_selection()
    refs = _verify_inputs(frozen, selection_sha256)
    analysis_path = phase.OUT / "summary/formal_timing_analysis.json"
    closed_path = phase.OUT / "summary/closed_loop_paired.json"
    if table._sha256(analysis_path) != selection["timing_analysis_sha256"]:
        raise RuntimeError("formal timing analysis changed after combination selection")
    if closed_loop_e4_fingerprint(closed_path) != selection["E4_closed_loop_evidence_sha256"]:
        raise RuntimeError("E4 closed-loop analysis changed after combination selection")
    by_version = _selected_cells(selection)
    selected_count = sum(len(cells) for cells in by_version.values())
    progress = {
        "status": "running" if selected_count else "complete_alias_only",
        "gpu": args.gpu,
        "selected_combo_conditions": selected_count,
        "conditions_total": 3 * selected_count,
        "completed_conditions": 0,
        "selection_sha256": selection_sha256,
        "source_revision": frozen["source_revision"],
        "rows": [],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    table._write_json(PROGRESS, progress)
    if selected_count == 0:
        return

    config = copy.deepcopy(frozen["main_table"])
    config["timing"]["gpu"] = args.gpu
    config["resources"]["permitted_gpus"] = list(range(8))
    config["outputs"]["root"] = str(phase.OUT)
    table._load_config = lambda: config
    table._check_dispatch_resources = phase.preflight
    table._check_timing_idle(args.gpu, config)

    driver_path = Path(__file__).resolve()
    driver_sha = table._sha256(driver_path)
    archive = phase.OUT / "provenance/combo_timing_driver" / driver_sha / driver_path.name
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(driver_path, archive)
    if table._sha256(archive) != driver_sha:
        raise RuntimeError("combo timing driver snapshot verification failed")

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    loaded, datasets, manifest_cache = {}, {}, {}
    try:
        for version in COMBO_VERSIONS:
            for cell, components in by_version[version]:
                for segment in ("E0-before", "combo", "E0-after"):
                    progress["current_segment"] = segment
                    progress["current_version"] = version
                    progress["current_cell"] = cell["cell_id"]
                    table._write_json(PROGRESS, progress)
                    _run_condition(
                        args=args, config=config, version=version,
                        owner_version=version, cell=cell, components=components,
                        selection_sha256=selection_sha256, driver_sha=driver_sha,
                        frozen=frozen, refs=refs, loaded=loaded, datasets=datasets,
                        manifest_cache=manifest_cache, progress=progress,
                    )
                if cell["task"] in loaded:
                    del loaded[cell["task"]]
                    torch.cuda.empty_cache()
        progress.update({"status": "complete", "current_segment": None,
                         "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        table._write_json(PROGRESS, progress)
    except BaseException:
        progress["status"] = "failed"
        progress["failed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        table._write_json(PROGRESS, progress)
        raise
    finally:
        del loaded, datasets
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    lock_path = phase.OUT / "timing/combinations/driver.lock"
    with table._exclusive_lock(lock_path, blocking=False):
        run(args)


if __name__ == "__main__":
    main()
