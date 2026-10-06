#!/usr/bin/env python3
"""Audit and analyze the frozen Phase 6.3 formal timing brackets."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cvpr_table1 as table
from scripts.round5_phase6_3 import OUT, METHODS, cells

BRACKETS = {
    "E1": ("00_E0", "02_E0"),
    "E2": ("02_E0", "04_E0"),
    "E3": ("04_E0", "06_E0"),
    "E4": ("06_E0",),
}


def _load(version, cell_id, block):
    parent = OUT / "timing/runs" / version / cell_id
    candidates = sorted(parent.glob("attempt_*/result.json")) if parent.exists() else []
    for path in reversed(candidates):
        value = table._read_json(path)
        if (value.get("status") != "complete" or value.get("formal_latency") is not True
                or value.get("execution_block") != block):
            continue
        samples = value.get("raw_samples", [])
        if (value.get("measurement_count") != 250 or len(samples) != 250
                or value.get("unique_states") != 50 or value.get("repeats_per_state") != 5):
            continue
        if not value.get("timing_driver_sha256") or not value.get("gpu_before", {}).get("gpu_uuid"):
            continue
        if value.get("identity_sha256") != table.identity_hash(value.get("identity", {})):
            continue
        parity = value.get("normal_timing_parity", {})
        action = parity.get("action", {})
        selection = parity.get("selection", {})
        if action.get("equal") is not True or selection.get("equal") is not True:
            continue
        return value, path
    return None, None


def _percentiles(values):
    import numpy as np
    array = np.asarray(values, dtype=np.float64)
    return {"mean_ms": float(array.mean()), "p50_ms": float(np.percentile(array, 50)),
            "p95_ms": float(np.percentile(array, 95))}


def _paired_summary(variant, baselines, *, seed):
    import numpy as np
    base_maps = []
    for baseline in baselines:
        base_maps.append({(int(s["state_index"]), int(s["repeat_index"])): float(s["wall_ms"])
                          for s in baseline["raw_samples"]})
    variant_map = {(int(s["state_index"]), int(s["repeat_index"])): float(s["wall_ms"])
                   for s in variant["raw_samples"]}
    keys = sorted(variant_map)
    if len(keys) != 250 or any(set(base) != set(keys) for base in base_maps):
        raise RuntimeError("formal bracket samples do not have identical state/repeat keys")
    base = np.asarray([np.mean([b[key] for b in base_maps]) for key in keys], dtype=np.float64)
    changed = np.asarray([variant_map[key] for key in keys], dtype=np.float64)
    delta = changed - base
    matrix = delta.reshape(50, 5)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 50, size=(20_000, 50))
    boot = matrix[indices].mean(axis=(1, 2))
    baseline_stats = _percentiles(base)
    variant_stats = _percentiles(changed)
    summary = {
        "baseline_brackets": len(base_maps), "baseline": baseline_stats,
        "variant": variant_stats, "paired_delta_ms": float(delta.mean()),
        "paired_delta_95_ci_ms": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "relative_speedup_fraction": float(1 - changed.mean() / base.mean()),
        "samples": 250, "unique_states": 50, "repeats_per_state": 5,
    }
    return summary, matrix


def _stratified_task_interval(matrices, *, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    distribution = np.zeros(20_000, dtype=np.float64)
    for matrix in matrices:
        indices = rng.integers(0, 50, size=(20_000, 50))
        distribution += matrix[indices].mean(axis=(1, 2)) / len(matrices)
    return [float(x) for x in np.quantile(distribution, [0.025, 0.975])]


def summarize():
    from source.common.cvpr_table1 import TASKS
    versions = ("E1", "E2", "E3", "E4")
    conditions = []
    equal_task = {}
    hardware_identities = set()
    timing_driver_hashes = set()
    for version in versions:
        equal_task[version] = {}
        for method in METHODS:
            if version == "E1" and method in {"P0", "P3"}:
                for task in TASKS:
                    conditions.append({"version": version, "method": method, "task": task,
                                       "status": "not_applicable"})
                continue
            deltas, baseline_means, variant_means = [], [], []
            baseline_p50s, variant_p50s, baseline_p95s, variant_p95s = [], [], [], []
            task_matrices = []
            for task in TASKS:
                cell = next(c for c in cells() if c["id"] == method and c["task"] == task)
                variant, variant_path = _load(version, cell["cell_id"], f"{'01' if version == 'E1' else {'E2':'03','E3':'05','E4':'07'}[version]}_{version}")
                if variant is None:
                    row = {"version": version, "method": method, "task": task,
                           "status": "incomplete", "completed_brackets": []}
                    conditions.append(row)
                    continue
                before, before_path = _load("E0", cell["cell_id"], BRACKETS[version][0])
                after = after_path = None
                if len(BRACKETS[version]) == 2:
                    after, after_path = _load("E0", cell["cell_id"], BRACKETS[version][1])
                if before is None or (len(BRACKETS[version]) == 2 and after is None):
                    row = {"version": version, "method": method, "task": task,
                           "status": "incomplete", "completed_brackets": [str(variant_path)]}
                    conditions.append(row)
                    continue
                baselines = [before] + ([after] if after is not None else [])
                gpu_identities = {
                    (value.get("gpu_before", {}).get("gpu_uuid"),
                     value.get("gpu_before", {}).get("gpu_name"),
                     value.get("gpu_before", {}).get("driver_version"))
                    for value in [variant, *baselines]
                }
                if len(gpu_identities) != 1 or None in next(iter(gpu_identities)):
                    raise RuntimeError(f"formal bracket hardware mismatch for {version}/{method}/{task}: {gpu_identities}")
                for value in [variant, *baselines]:
                    gpu = value["gpu_before"]
                    hardware_identities.add((gpu["gpu_uuid"], gpu["gpu_name"], gpu["driver_version"]))
                    timing_driver_hashes.add(value["timing_driver_sha256"])
                summary, matrix = _paired_summary(variant, baselines,
                    seed=630630 + versions.index(version) * 1000 + METHODS.index(method) * 10 + TASKS.index(task))
                row = {"version": version, "method": method, "task": task,
                       "status": "complete", **summary,
                       "results": [str(variant_path.relative_to(ROOT)), str(before_path.relative_to(ROOT))]
                                  + ([str(after_path.relative_to(ROOT))] if after_path else [])}
                conditions.append(row)
                deltas.append(summary["paired_delta_ms"])
                baseline_means.append(summary["baseline"]["mean_ms"])
                variant_means.append(summary["variant"]["mean_ms"])
                baseline_p50s.append(summary["baseline"]["p50_ms"])
                variant_p50s.append(summary["variant"]["p50_ms"])
                baseline_p95s.append(summary["baseline"]["p95_ms"])
                variant_p95s.append(summary["variant"]["p95_ms"])
                task_matrices.append(matrix)
            if len(deltas) == len(TASKS):
                baseline_equal_mean = float(sum(baseline_means) / len(TASKS))
                variant_equal_mean = float(sum(variant_means) / len(TASKS))
                equal_task[version][method] = {
                    "four_task_equal_mean_delta_ms": float(sum(deltas) / len(deltas)),
                    "four_task_equal_mean_delta_95_ci_ms": _stratified_task_interval(
                        task_matrices, seed=631000 + versions.index(version) * 100 + METHODS.index(method)),
                    "baseline_four_task_equal_mean_ms": baseline_equal_mean,
                    "variant_four_task_equal_mean_ms": variant_equal_mean,
                    "relative_speedup_fraction": float(1 - variant_equal_mean / baseline_equal_mean),
                    "baseline_four_task_equal_p50_ms": float(sum(baseline_p50s) / len(TASKS)),
                    "variant_four_task_equal_p50_ms": float(sum(variant_p50s) / len(TASKS)),
                    "baseline_four_task_equal_p95_ms": float(sum(baseline_p95s) / len(TASKS)),
                    "variant_four_task_equal_p95_ms": float(sum(variant_p95s) / len(TASKS)),
                    "task_conditions": len(deltas),
                }

    if len(hardware_identities) > 1 or len(timing_driver_hashes) > 1:
        raise RuntimeError(f"formal timing run mixed hardware or driver identities: "
                           f"hardware={hardware_identities}, drivers={timing_driver_hashes}")
    report = {
        "schema_version": 1,
        "scope": "formally gated batch=1 latency; paired same-state/repeat samples",
        "source_revision": table._read_json(OUT / "frozen_config.json")["source_revision"],
        "bracket_order": BRACKETS,
        "paired_interval": "95% percentile cluster bootstrap over 50 states, retaining five repeats per sampled state; 20,000 replicates",
        "hardware_identities": [list(value) for value in sorted(hardware_identities)],
        "timing_driver_hashes": sorted(timing_driver_hashes),
        "conditions_complete": sum(row["status"] == "complete" for row in conditions),
        "conditions_target": len(conditions),
        "four_task_equal_means": equal_task,
        "conditions": conditions,
    }
    path = OUT / "summary/formal_timing_analysis.json"
    table._write_json(path, report)
    print(json.dumps({"output": str(path.relative_to(ROOT)),
                      "complete": report["conditions_complete"], "target": report["conditions_target"],
                      "four_task_equal_means": equal_task}, ensure_ascii=False))


if __name__ == "__main__":
    summarize()
