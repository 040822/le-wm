#!/usr/bin/env python3
"""Analyze formal latency results for Phase 6.3 selected combinations."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cvpr_table1 as table
from scripts.round5_phase6_3 import METHODS, OUT
from scripts.round5_phase6_3_combos import COMBO_VERSIONS, entry_for, load_selection
from scripts.round5_phase6_3_timing_analysis import _paired_summary, _stratified_task_interval
from source.common.cvpr_table1 import TASKS, timing_conditions

RUNS = OUT / "timing/combinations/runs"
OUTPUT = OUT / "summary/combo_formal_timing_analysis.json"


def _load_result(owner_version, block, cell, components, selection_sha256):
    parent = RUNS / owner_version / block / cell["cell_id"]
    for result_path in sorted(parent.glob("attempt_*/result.json"), reverse=True) if parent.exists() else []:
        value = table._read_json(result_path, {})
        identity = value.get("identity", {})
        expected_active = components if block == f"01_{owner_version}" else []
        expected_precision = "bfloat16_encoder" if "E4" in expected_active else "float32"
        if (value.get("status") != "complete" or value.get("formal_latency") is not True
                or value.get("measurement_count") != 250 or len(value.get("raw_samples", [])) != 250
                or value.get("unique_states") != 50 or value.get("repeats_per_state") != 5
                or value.get("identity_sha256") != table.identity_hash(identity)
                or value.get("combo_owner_version") != owner_version
                or value.get("combo_components") != components
                or value.get("active_components") != expected_active
                or value.get("effective_encoder_precision") != expected_precision
                or value.get("combination_selection_sha256") != selection_sha256
                or identity.get("combo_components") != components
                or identity.get("active_components") != expected_active
                or identity.get("effective_encoder_precision") != expected_precision
                or value.get("execution_block") != block
                or value.get("normal_timing_parity", {}).get("action", {}).get("equal") is not True
                or value.get("normal_timing_parity", {}).get("selection", {}).get("equal") is not True):
            continue
        gpu = value.get("gpu_before", {})
        after = value.get("gpu_after", {})
        before_measurement = value.get("gpu_before_measurement", {})
        if (not gpu.get("gpu_uuid") or gpu.get("gpu_uuid") != after.get("gpu_uuid")
                or gpu.get("gpu_uuid") != before_measurement.get("gpu_uuid")
                or not value.get("timing_driver_sha256")):
            continue
        return value, result_path
    return None, None


def summarize():
    import numpy as np

    selection, selection_sha256 = load_selection()
    selected_cells = [c for c in timing_conditions() if c["family"] == "cowm"
                      and c["id"] in METHODS and c["method_id"].endswith("__main")]
    conditions, means = [], {}
    hardware, drivers = set(), set()
    for version in COMBO_VERSIONS:
        means[version] = {}
        for method in METHODS:
            choice = entry_for(selection, version, method)
            for cell in (c for c in selected_cells if c["id"] == method):
                task = cell["task"]
                if choice["status"] == "alias":
                    conditions.append({"version": version, "method": method, "task": task,
                                       "status": "alias", "alias_of": choice["alias_of"],
                                       "components": choice["components"]})
                    continue
                components = list(choice["components"])
                combo_block = f"01_{version}"
                blocks = ["00_E0", combo_block, "02_E0"]
                results = []
                paths = []
                for block in blocks:
                    value, path = _load_result(version, block, cell, components, selection_sha256)
                    if value is None:
                        results = []
                        break
                    results.append(value)
                    paths.append(path)
                if len(results) != 3:
                    conditions.append({"version": version, "method": method, "task": task,
                                       "status": "incomplete", "components": components,
                                       "completed_blocks": [str(p.relative_to(ROOT)) for p in paths]})
                    continue
                devices = {(r["gpu_before"]["gpu_uuid"], r["gpu_before"].get("gpu_name"),
                            r["gpu_before"].get("driver_version")) for r in results}
                block_drivers = {r["timing_driver_sha256"] for r in results}
                if (len(devices) != 1 or None in next(iter(devices))
                        or len(block_drivers) != 1):
                    raise RuntimeError(f"combination bracket identity mismatch: {version}/{method}/{task}")
                hardware.add(next(iter(devices)))
                drivers.add(next(iter(block_drivers)))
                summary, matrix = _paired_summary(results[1], [results[0], results[2]],
                    seed=632000 + COMBO_VERSIONS.index(version) * 1000
                         + METHODS.index(method) * 10 + TASKS.index(task))
                conditions.append({"version": version, "method": method, "task": task,
                    "status": "complete", "components": components, **summary,
                    "results": [str(path.relative_to(ROOT)) for path in paths]})
                means.setdefault(version, {}).setdefault(method, {})[task] = {
                    "summary": summary, "matrix": matrix, "paths": paths,
                }
            task_data = means[version].get(method, {})
            if len(task_data) == len(TASKS):
                base_means = [task_data[t]["summary"]["baseline"]["mean_ms"] for t in TASKS]
                variant_means = [task_data[t]["summary"]["variant"]["mean_ms"] for t in TASKS]
                matrices = [task_data[t]["matrix"] for t in TASKS]
                deltas = [task_data[t]["summary"]["paired_delta_ms"] for t in TASKS]
                means[version][method] = {
                    "four_task_equal_mean_delta_ms": float(np.mean(deltas)),
                    "four_task_equal_mean_delta_95_ci_ms": _stratified_task_interval(
                        matrices, seed=633000 + COMBO_VERSIONS.index(version) * 100 + METHODS.index(method)),
                    "baseline_four_task_equal_mean_ms": float(np.mean(base_means)),
                    "variant_four_task_equal_mean_ms": float(np.mean(variant_means)),
                    "relative_speedup_fraction": float(1 - np.mean(variant_means) / np.mean(base_means)),
                    "task_conditions": len(TASKS),
                }
            elif task_data:
                means[version][method] = {"status": "incomplete", "complete_tasks": sorted(task_data)}

    report = {
        "schema_version": 1,
        "scope": "formal batch=1 combination latency with same-GPU adjacent E0 brackets",
        "analysis_sha256": table._sha256(Path(__file__)),
        "combination_selection_sha256": selection_sha256,
        "timing_driver_sha256": next(iter(drivers)) if len(drivers) == 1 else None,
        "hardware_identities": [list(item) for item in sorted(hardware)],
        "conditions_complete": sum(r["status"] == "complete" for r in conditions),
        "conditions_target": len(conditions),
        "four_task_equal_means": means,
        "conditions": conditions,
    }
    table._write_json(OUTPUT, report)
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)),
                      "complete": report["conditions_complete"],
                      "target": report["conditions_target"],
                      "four_task_equal_means": means}, ensure_ascii=False))


if __name__ == "__main__":
    summarize()
