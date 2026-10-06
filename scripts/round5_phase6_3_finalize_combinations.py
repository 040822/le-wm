#!/usr/bin/env python3
"""Apply the preregistered latency gate to measured Phase 6.3 combinations."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cvpr_table1 as table
from scripts.round5_phase6_3 import METHODS, OUT
from scripts.round5_phase6_3_combos import COMBO_VERSIONS, load_selection
from source.common.cvpr_table1 import TASKS

TIMING_PATH = OUT / "summary/combo_formal_timing_analysis.json"
OUTPUT_PATH = OUT / "summary/final_combinations.json"
HELPERS_PATH = ROOT / "scripts/round5_phase6_3_combos.py"
SCRIPT_PATH = Path(__file__).resolve()
ANALYSIS_SCRIPT_PATH = ROOT / "scripts/round5_phase6_3_combo_timing_analysis.py"


def finalize() -> dict:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"final combination decision already frozen: {OUTPUT_PATH}")
    selection, selection_sha256 = load_selection()
    timing = table._read_json(TIMING_PATH)
    if timing.get("combination_selection_sha256") != selection_sha256:
        raise RuntimeError("combination timing used a different candidate selection")
    if timing.get("analysis_sha256") != table._sha256(ANALYSIS_SCRIPT_PATH):
        raise RuntimeError("combination timing analysis code changed after producing its report")
    if table._sha256(OUT / "summary/formal_timing_analysis.json") != selection.get("timing_analysis_sha256"):
        raise RuntimeError("single-variable timing evidence changed after combination selection")
    results = {}
    for version in COMBO_VERSIONS:
        results[version] = {}
        for method in METHODS:
            candidate = selection["versions"][version][method]
            if candidate["status"] == "alias":
                results[version][method] = {
                    "status": "alias", "alias_of": candidate["alias_of"],
                    "components": candidate["components"],
                    "reason": "implementation is identical to the referenced version; no duplicate timing run",
                }
                continue
            conditions = [row for row in timing.get("conditions", [])
                          if row.get("version") == version and row.get("method") == method]
            if len(conditions) != len(TASKS) or any(row.get("status") != "complete" for row in conditions):
                raise RuntimeError(f"combination timing is incomplete: {version}/{method}")
            summary = timing.get("four_task_equal_means", {}).get(version, {}).get(method)
            if summary is None or summary.get("task_conditions") != len(TASKS):
                raise RuntimeError(f"missing four-task combination summary: {version}/{method}")
            delta = float(summary["four_task_equal_mean_delta_ms"])
            interval = [float(x) for x in summary["four_task_equal_mean_delta_95_ci_ms"]]
            effective = delta < 0 and interval[1] < 0
            results[version][method] = {
                "status": "effective" if effective else "not_effective",
                "components": candidate["components"],
                "four_task_equal_mean_delta_ms": delta,
                "paired_delta_95_ci_ms": interval,
                "relative_speedup_fraction": float(summary["relative_speedup_fraction"]),
                "results": [path for row in conditions for path in row.get("results", [])],
                "rule": "effective only when the four-task paired mean latency delta is negative and the 95% CI upper bound is below zero",
            }
    result = {
        "schema_version": 1,
        "combination_selection_sha256": selection_sha256,
        "timing_analysis": str(TIMING_PATH.relative_to(ROOT)),
        "timing_analysis_sha256": hashlib.sha256(TIMING_PATH.read_bytes()).hexdigest(),
        "finalizer_code_sha256": hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
        "combination_helpers_sha256": hashlib.sha256(HELPERS_PATH.read_bytes()).hexdigest(),
        "versions": results,
    }
    table._write_json(OUTPUT_PATH, result)
    print({"output": str(OUTPUT_PATH.relative_to(ROOT)), "versions": results})
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    finalize()


if __name__ == "__main__":
    main()
