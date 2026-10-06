#!/usr/bin/env python3
"""Run the frozen, interleaved Phase 6.3 batch=1 formal timing matrix.

The primary CVPR archive is read-only. Each bracket and method condition is
stored under outputs/round5/phase6_3/timing/ with the full asset, cohort,
source, software, device, and driver identity.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import hashlib
import json
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
from source.common.cvpr_table1 import TASKS, timing_conditions
from source.common.round5_phase6_3 import deferred_refinement_checks, encoder_bf16, fixed_condition_cache


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _asset_refs(task):
    refs = table._read_json(phase.OUT / "provenance/assets_and_cohorts.json")
    selected = [r for r in refs if r["task"] == task]
    required = {r["kind"] for r in selected}
    if not {"cowm_checkpoint", "cowm_config", "cohort"}.issubset(required):
        raise RuntimeError(f"incomplete frozen assets for {task}")
    for ref in selected:
        if table._sha256(Path(ref["path"])) != ref["sha256"]:
            raise RuntimeError(f"frozen asset or cohort changed: {ref['path']}")
    return selected


def _verify_snapshot(frozen):
    hashes = phase.source_hashes()
    if hashes != frozen["source_hashes"]:
        changed = sorted(set(hashes) | set(frozen["source_hashes"]))
        changed = [p for p in changed if hashes.get(p) != frozen["source_hashes"].get(p)]
        raise RuntimeError(f"Phase 6.3 source snapshot changed: {changed[:20]}")
    revision = phase.OUT / "provenance/code_snapshots" / frozen["source_revision"]
    for relative, digest in hashes.items():
        if table._sha256(revision / relative) != digest:
            raise RuntimeError(f"missing or changed Phase 6.3 source snapshot: {relative}")
    return hashes


def _latest_audit(frozen):
    path = phase.OUT / "summary/initial_state_validation.json"
    audit = table._read_json(path, {})
    if audit.get("source_revision") != frozen["source_revision"]:
        raise RuntimeError("initial-state evidence has a different source identity")
    return audit


def _version_gate(version, conditions, audit):
    if version == "E0":
        return
    for cell in conditions:
        if version == "E1" and cell["guidance_mode"] != "post_opt_refine":
            continue
        identity = cell["cell_id"]
        row = next((r for r in audit["conditions"] if r["version"] == version
                    and r["condition_id"] == identity and r["status"] == "complete"), None)
        if row is None:
            raise RuntimeError(f"formal timing gated: {version} initial-state validation incomplete for {identity}")
        if version in {"E1", "E2", "E3"} and row.get("initial_parity_passed") is not True:
            raise RuntimeError(f"formal timing gated: {version} parity failed for {identity}")
        if version == "E4":
            result = table._read_json(ROOT / row["result"])
            rawpath = Path(row["result"]).parent / "raw_parity.pt"
            import torch
            raw = torch.load(rawpath, map_location="cpu", weights_only=True)
            for state in raw:
                for side in ("baseline", "variant"):
                    if not all(torch.isfinite(t).all().item() for values in state[side].values() for t in values):
                        raise RuntimeError(f"formal timing gated: non-finite E4 output in {identity}")


def _identity(root, cell, version, block, driver_sha, frozen, refs):
    task = cell["task"]
    relevant = [r for r in refs if r["task"] == task]
    config_path = phase.MAIN / "provenance/config/table1.json"
    environment_path = phase.OUT / "provenance/environment.json"
    cohort = next(r for r in relevant if r["kind"] == "cohort" and r["seed"] == 42)
    return {
        "cell": dict(cell), "version": version, "execution_block": block,
        "source_revision": frozen["source_revision"],
        "phase6_3_frozen_config_sha256": table._sha256(phase.OUT / "frozen_config.json"),
        "main_table_frozen_config_sha256": table._sha256(phase.MAIN / "frozen_config.json"),
        "main_table_config_sha256": table._sha256(config_path),
        "timing_protocol": copy.deepcopy(frozen["main_table"]["timing"]),
        "protocol": copy.deepcopy(frozen["main_table"]["protocol"]),
        "assets_sha256": [{"path": r["path"], "sha256": r["sha256"]}
                           for r in relevant if r["kind"] != "cohort"],
        "cohort_file": cohort["path"], "cohort_file_sha256": cohort["sha256"],
        "cohort_semantic_sha256": cohort["cohort_sha256"],
        "environment_sha256": table._sha256(environment_path),
        "code_snapshot_sha256": frozen["source_hashes"],
        "timing_driver_sha256": driver_sha,
        "version_switches": copy.deepcopy(frozen["versions"][version]),
        "effective_encoder_precision": "bfloat16_encoder" if version == "E4" else "float32",
        "device": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "physical_gpu": int(os.environ["CUDA_VISIBLE_DEVICES"])},
    }


def _source_identity_patch(frozen, refs, driver_sha):
    """Route the shared CVPR identity validator to Phase 6.3's frozen records."""
    original_identity = table._cell_identity
    original_load = table._load_models
    original_config = table._load_config
    config = copy.deepcopy(frozen["main_table"])
    table._load_config = lambda: config

    def cell_identity(root, cell):
        values = _identity(root, cell, cell.get("_timing_version", "E0"),
                            cell.get("_timing_block", "E0"), driver_sha, frozen, refs)
        return {
            "cell": dict(cell),
            "configuration_sha256": hashlib.sha256(table._json_bytes(config)).hexdigest(),
            "asset_sha256": values["assets_sha256"],
            "cohort_file_sha256": values["cohort_file_sha256"],
            "source_revision_id": frozen["source_revision"],
            "code_snapshot_sha256": frozen["source_hashes"],
            "execution_environment_sha256": values["environment_sha256"],
            "timing_driver_sha256": driver_sha,
            "timing_version": cell.get("_timing_version", "E0"),
            "execution_block": cell.get("_timing_block", "E0"),
        }

    def load_models(root, task, stage, device):
        return original_load(phase.MAIN, task, stage, device)

    table._cell_identity = cell_identity
    table._load_models = load_models
    return original_identity, original_load, original_config


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise RuntimeError(f"launch with CUDA_VISIBLE_DEVICES={args.gpu}")
    frozen = table._read_json(phase.OUT / "frozen_config.json")
    hashes = _verify_snapshot(frozen)
    if table._sha256(phase.MAIN / "frozen_config.json") != frozen["main_table_frozen_config_sha256"]:
        raise RuntimeError("main Table 1 frozen configuration changed")
    audit = _latest_audit(frozen)
    config = copy.deepcopy(frozen["main_table"])
    config["timing"]["gpu"] = args.gpu
    config["resources"]["permitted_gpus"] = list(range(8))
    config["outputs"]["root"] = str(phase.OUT)
    table._load_config = lambda: config
    table._check_dispatch_resources = phase.preflight
    table._check_timing_idle(args.gpu, config)
    conditions = [c for c in timing_conditions()
                  if c["family"] == "cowm" and c["id"] in phase.METHODS
                  and c["method_id"].endswith("__main")]
    refs = table._read_json(phase.OUT / "provenance/assets_and_cohorts.json")
    for task in TASKS:
        _asset_refs(task)
    block_order = list(frozen["formal_version_order"])
    if not block_order:
        raise RuntimeError("the frozen formal timing schedule has no execution blocks")
    for version in block_order:
        _version_gate(version, conditions, audit)
    driver_path = Path(__file__).resolve()
    driver_sha = table._sha256(driver_path)
    archive = phase.OUT / "provenance/timing_driver" / driver_sha / driver_path.name
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(driver_path, archive)
    if table._sha256(archive) != driver_sha:
        raise RuntimeError("timing driver snapshot verification failed")
    old_id, old_load, old_config = _source_identity_patch(frozen, refs, driver_sha)
    import torch
    import numpy as np
    from source.common.eval import get_dataset
    from source.common.round3_phase1 import CohortManifest
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    loaded, datasets = {}, {}
    rows = {}
    status_path = phase.OUT / "timing/formal_progress.json"
    initial = phase.preflight(args.gpu, config)
    prior_results = (phase.OUT / "timing/runs").glob("*/**/attempt_*/result.json")
    for path in prior_results:
        prior = table._read_json(path, {})
        if prior.get("formal_latency") is not True or prior.get("status") != "complete":
            continue
        if prior.get("timing_driver_sha256") != driver_sha:
            raise RuntimeError(f"completed formal data used another timing driver: {path}")
        prior_gpu = prior.get("gpu_before", {})
        if prior_gpu.get("gpu_uuid") != initial.get("gpu_uuid"):
            raise RuntimeError(f"completed formal data used another GPU: {path}")
    total_conditions = sum(8 if version == "E1" else len(conditions) for version in block_order)
    not_applicable = {"E1": [c["cell_id"] for c in conditions
                               if c["guidance_mode"] != "post_opt_refine"]}
    table._write_json(status_path, {"status": "running", "gpu": args.gpu,
        "initial_preflight": initial, "blocks": block_order, "conditions_total": total_conditions,
        "not_applicable": not_applicable,
        "timing_driver_sha256": driver_sha, "source_revision": frozen["source_revision"]})
    try:
        for block_index, version in enumerate(block_order):
            cells = [c for c in conditions if version != "E1" or c["guidance_mode"] == "post_opt_refine"]
            for cell_index, original_cell in enumerate(cells):
                table._check_timing_idle(args.gpu, config)
                cell = {**original_cell, "_timing_version": version, "_timing_block": f"{block_index:02d}_{version}"}
                task = cell["task"]
                identity = _identity(phase.OUT, cell, version, cell["_timing_block"],
                                     driver_sha, frozen, refs)
                identity_sha = table.identity_hash(identity)
                cell_id = str(cell["cell_id"])
                parent = phase.OUT / "timing/runs" / version / cell_id
                prior = []
                for path in parent.glob("attempt_*") if parent.exists() else []:
                    try:
                        prior.append(int(path.name.removeprefix("attempt_")))
                    except ValueError:
                        pass
                if any((value := table._read_json(path / "result.json", {})).get("status") == "complete"
                       and value.get("identity_sha256") == identity_sha for path in parent.glob("attempt_*") if parent.exists()):
                    rows.setdefault(version, {})[cell_id] = "complete"
                    continue
                attempt = parent / f"attempt_{max(prior, default=0)+1:03d}"
                command = {"argv": sys.argv, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                           "driver_sha256": driver_sha, "gpu_preflight": phase.preflight(args.gpu, config)}
                try:
                    key = (task, int(cell["stage"]))
                    if key not in loaded:
                        loaded[key] = table._load_models(phase.MAIN, task, key[1], "cuda:0")
                    if task not in datasets:
                        cfg = phase.table._cell_config(cell)
                        from source.common.eval import get_dataset
                        datasets[task] = get_dataset(cfg, cfg.eval.dataset_name)
                    manifest = CohortManifest.load(phase.MAIN / "cohorts" / task / "seed_42.json")
                    stage_config = copy.deepcopy(config)
                    version_config = copy.deepcopy(stage_config)
                    variant = {"E0": None, "E1": deferred_refinement_checks,
                               "E2": fixed_condition_cache, "E3": None,
                               "E4": encoder_bf16}[version]
                    original_config_loader = table._load_config
                    if version == "E4":
                        version_config["protocol"]["precision"] = "bfloat16_encoder"
                    table._load_config = lambda: version_config
                    stack = ExitStack()
                    if version == "E3":
                        stack.enter_context(phase.preprocess_factory())
                    elif variant is not None:
                        stack.enter_context(variant(loaded[key]["cowm"]))
                    with stack:
                        measurement_preflight = table._check_timing_idle(args.gpu, version_config)
                        result = table._capture_and_measure_condition(root=phase.OUT, cell=cell,
                            identity_sha256=identity_sha, config=version_config,
                            model=loaded[key]["cowm"], dataset=datasets[task], manifest=manifest,
                            attempt_dir=attempt)
                    result["gpu_before_measurement"] = measurement_preflight
                    table._load_config = original_config_loader
                    table._write_json(attempt / "cell_identity.json", {"identity": identity,
                        "identity_sha256": identity_sha, "attempt": attempt.name, "command": command})
                    table._write_json(attempt / "rng_schedule.json", {
                        "state_order": list(range(50)), "warmup_rng_seeds": [910000+50000+i for i in range(10)],
                        "measurement_rng_seeds": [910000+i for i in range(250)],
                        "execution_order": "state_index ascending, repeat_index ascending"})
                    result.update({"attempt": attempt.name, "identity": identity,
                        "version": version, "execution_block": cell["_timing_block"],
                        "formal_latency": True, "gpu_before": command["gpu_preflight"],
                        "measurement_count": len(result["raw_samples"]),
                        "repeats_per_state": 5, "states": result["unique_states"],
                        "source_revision": frozen["source_revision"],
                        "timing_driver_sha256": driver_sha})
                    after = table._check_timing_idle(args.gpu, config)
                    result["gpu_after"] = after
                    if after["cpu_load_1m_per_cpu"] > 0.75:
                        result["status"] = "interfered"
                        result["interference_reason"] = "CPU load exceeded frozen threshold"
                    else:
                        result["status"] = "complete"
                    result["identity_sha256"] = identity_sha
                    table._write_json(attempt / "result.json", result)
                    rows.setdefault(version, {})[cell_id] = result["status"]
                    table._write_json(status_path, {"status": "running", "gpu": args.gpu,
                        "blocks": block_order, "rows": rows, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "conditions_total": total_conditions, "not_applicable": not_applicable,
                        "timing_driver_sha256": driver_sha})
                    if result["status"] != "complete":
                        raise RuntimeError(f"formal timing interference for {cell_id}; raw attempt preserved")
                    print(f"formal {version} {cell_id}: {result['mean_ms']:.4f} ms", flush=True)
                except BaseException as exc:
                    attempt.mkdir(parents=True, exist_ok=True)
                    table._write_json(attempt / "failure.json", {"status": "failed", "error_type": type(exc).__name__,
                        "error": str(exc), "traceback": traceback.format_exc()})
                    raise
        table._write_json(phase.OUT / "timing/formal_progress.json", {"status": "complete", "gpu": args.gpu,
            "blocks": block_order, "rows": rows, "timing_driver_sha256": driver_sha,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    finally:
        table._cell_identity, table._load_models, table._load_config = old_id, old_load, old_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
