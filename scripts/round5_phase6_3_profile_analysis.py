#!/usr/bin/env python3
"""Summarize labeled E0 profiler scopes without treating them as latency."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cvpr_table1 as table
from scripts.round5_phase6_3 import OUT, cells


def summarize():
    import numpy as np
    scope_names = (
        "input_transform", "input_transform_and_h2d", "encoder_projector",
        "A_generation", "B_scoring", "B_cost", "refinement",
        "refinement_forward", "refinement_backward", "refinement_normalize",
        "refinement_rms_constraint", "cpu_return",
    )
    by_cell = []
    for cell in cells():
        parent = OUT / "profile/E0" / cell["cell_id"]
        attempts = sorted(parent.glob("attempt_*"))
        selected = next((p for p in reversed(attempts)
                         if (p / "result.json").is_file() and (p / "operators.json").is_file()), None)
        if selected is None:
            raise RuntimeError(f"missing labeled E0 profile: {cell['cell_id']}")
        result = table._read_json(selected / "result.json")
        operators = table._read_json(selected / "operators.json")
        if result.get("status") != "complete" or result.get("profiled_calls") != 5:
            raise RuntimeError(f"invalid profiler result: {selected}")
        grouped = {name: {"self_cpu_ms_over_5_calls": 0.0,
                          "self_device_ms_over_5_calls": 0.0} for name in scope_names}
        for row in operators:
            key = str(row.get("key", ""))
            if key.startswith("phase6_3/"):
                name = key.removeprefix("phase6_3/")
                if name in grouped:
                    grouped[name]["self_cpu_ms_over_5_calls"] += float(row.get("self_cpu_time_us", 0.0)) / 1000
                    grouped[name]["self_device_ms_over_5_calls"] += float(row.get("self_device_time_us", 0.0)) / 1000
        by_cell.append({"cell_id": cell["cell_id"], "method": cell["id"], "task": cell["task"],
                        "attempt": selected.name, "profiled_calls": 5, "stages": grouped})

    groups = {}
    for method in dict.fromkeys(row["method"] for row in by_cell):
        groups[method] = {}
        selected = [row for row in by_cell if row["method"] == method]
        for name in scope_names:
            groups[method][name] = {}
            for metric in ("self_cpu_ms_over_5_calls", "self_device_ms_over_5_calls"):
                values = np.asarray([row["stages"][name][metric] / 5 for row in selected], dtype=np.float64)
                groups[method][name][metric.replace("_over_5_calls", "_per_call")] = {
                    "task_count": int(values.size), "mean_ms": float(values.mean()),
                    "p50_ms": float(np.percentile(values, 50)),
                    "p95_ms": float(np.percentile(values, 95)),
                }
    report = {
        "scope": "E0 component profiler diagnostic only; not end-to-end latency",
        "profiled_calls_per_condition": 5,
        "aggregation": "self times from labeled record_function scopes; scopes are nested and not additive",
        "conditions_complete": len(by_cell), "conditions_target": 16,
        "by_method_and_stage_per_call": groups,
        "by_cell_five_call_sums": by_cell,
    }
    output = OUT / "summary/profile_stage_summary.json"
    table._write_json(output, report)
    compact = {
        method: {
            stage: {
                "cpu_mean_ms": round(values["self_cpu_ms_per_call"]["mean_ms"], 4),
                "device_mean_ms": round(values["self_device_ms_per_call"]["mean_ms"], 4),
            }
            for stage, values in stages.items()
            if stage in {"input_transform", "input_transform_and_h2d", "encoder_projector",
                         "A_generation", "B_cost", "refinement", "cpu_return"}
        }
        for method, stages in groups.items()
    }
    print(json.dumps({"output": str(output.relative_to(ROOT)), "conditions": len(by_cell),
                      "mean_per_call_ms": compact}, ensure_ascii=False))


if __name__ == "__main__":
    summarize()
