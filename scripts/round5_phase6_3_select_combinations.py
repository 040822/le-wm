#!/usr/bin/env python3
"""Freeze Phase 6.3 combination membership from completed formal evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cvpr_table1 as table
from scripts.round5_phase6_3 import METHODS, OUT
from scripts.round5_phase6_3_combos import closed_loop_e4_fingerprint, validate_selection
from source.common.cvpr_table1 import TASKS

TIMING_PATH = OUT / "summary/formal_timing_analysis.json"
CLOSED_LOOP_PATH = OUT / "summary/closed_loop_paired.json"
OUTPUT_PATH = OUT / "summary/selected_combinations.json"
SCRIPT_PATH = Path(__file__).resolve()
HELPERS_PATH = ROOT / "scripts/round5_phase6_3_combos.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_complete_evidence(timing: dict) -> None:
    rows = timing.get("conditions", [])
    for version in ("E1", "E2", "E3", "E4"):
        for method in METHODS:
            if version == "E1" and method in {"P0", "P3"}:
                continue
            matching = [r for r in rows if r.get("version") == version
                        and r.get("method") == method]
            if len(matching) != len(TASKS) or any(r.get("status") != "complete" for r in matching):
                raise RuntimeError(f"formal timing evidence incomplete: {version}/{method}")
            if {r.get("task") for r in matching} != set(TASKS):
                raise RuntimeError(f"formal timing task coverage incomplete: {version}/{method}")
            summary = timing.get("four_task_equal_means", {}).get(version, {}).get(method)
            if summary is None or summary.get("four_task_equal_mean_delta_95_ci_ms") is None:
                raise RuntimeError(f"missing paired latency summary: {version}/{method}")


def _effective(timing: dict, version: str, method: str) -> tuple[bool, dict]:
    summary = timing["four_task_equal_means"].get(version, {}).get(method)
    if summary is None:
        return False, {"eligible": False, "reason": "not applicable or incomplete"}
    delta = float(summary["four_task_equal_mean_delta_ms"])
    interval = [float(v) for v in summary["four_task_equal_mean_delta_95_ci_ms"]]
    effective = delta < 0.0 and interval[1] < 0.0
    return effective, {
        "eligible": effective,
        "four_task_equal_mean_delta_ms": delta,
        "paired_delta_95_ci_ms": interval,
        "relative_speedup_fraction": float(summary["relative_speedup_fraction"]),
        "rule": "include only if the four-task mean delta is negative and the paired 95% CI upper bound is below zero",
    }


def _require_e4_closed_loop(method: str, closed_loop: dict) -> None:
    rows = [r for r in closed_loop.get("conditions", [])
            if r.get("version") == "E4" and r.get("method") == method]
    if len(rows) != len(TASKS) or any(r.get("status") != "complete" for r in rows):
        raise RuntimeError(f"E4 closed-loop evidence incomplete: {method}")
    if {r.get("task") for r in rows} != set(TASKS):
        raise RuntimeError(f"E4 closed-loop task coverage incomplete: {method}")


def _alias_or_selected(version: str, components: list[str], *, alias: str | None = None) -> dict:
    if alias is None:
        if not components:
            alias = "E0"
        elif len(components) == 1:
            alias = components[0]
    if alias is not None:
        return {"status": "alias", "components": components, "alias_of": alias}
    return {"status": "selected", "components": components}


def select() -> dict:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"selection already frozen: {OUTPUT_PATH}")
    timing = table._read_json(TIMING_PATH)
    _require_complete_evidence(timing)
    closed_loop = table._read_json(CLOSED_LOOP_PATH)
    selected = {"C-FP32": {}, "C-BF16": {}}
    decisions = {}
    for method in METHODS:
        fp32_components = []
        decisions[method] = {}
        for component in ("E1", "E2", "E3"):
            if component == "E1" and method in {"P0", "P3"}:
                decisions[method][component] = {"eligible": False, "reason": "not applicable"}
                continue
            effective, evidence = _effective(timing, component, method)
            decisions[method][component] = evidence
            if effective:
                fp32_components.append(component)
        selected["C-FP32"][method] = _alias_or_selected("C-FP32", fp32_components)

        e4_effective, e4_evidence = _effective(timing, "E4", method)
        if e4_effective:
            _require_e4_closed_loop(method, closed_loop)
        decisions[method]["E4"] = {
            **e4_evidence,
            "closed_loop_complete": e4_effective,
            "closed_loop_evidence": "summary/closed_loop_paired.json",
            "precision_variant": True,
        }
        bf16_components = fp32_components + (["E4"] if e4_effective else [])
        if not e4_effective:
            selected["C-BF16"][method] = _alias_or_selected(
                "C-BF16", bf16_components, alias="C-FP32")
        else:
            selected["C-BF16"][method] = _alias_or_selected("C-BF16", bf16_components)

    artifact = {
        "schema_version": 1,
        "source_revision": table._read_json(OUT / "frozen_config.json")["source_revision"],
        "timing_analysis": str(TIMING_PATH.relative_to(ROOT)),
        "timing_analysis_sha256": _sha(TIMING_PATH),
        "E4_closed_loop_analysis": str(CLOSED_LOOP_PATH.relative_to(ROOT)),
        "E4_closed_loop_evidence_sha256": closed_loop_e4_fingerprint(CLOSED_LOOP_PATH),
        "selection_script_sha256": _sha(SCRIPT_PATH),
        "combination_helpers_sha256": _sha(HELPERS_PATH),
        "selection_rule": "A single-variable optimization enters a method's FP32 combination only when its four-task equal-weight paired mean latency delta is negative and the paired 95% CI upper bound is below zero. E1 is inapplicable to P0/P3. E4 additionally requires complete E4 closed-loop evidence; its success-rate interval is reported without a non-inferiority claim.",
        "decisions": decisions,
        "versions": selected,
    }
    artifact = validate_selection(artifact)
    table._write_json(OUTPUT_PATH, artifact)
    print(json.dumps({"output": str(OUTPUT_PATH.relative_to(ROOT)), "versions": selected}, ensure_ascii=False))
    return artifact


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    select()


if __name__ == "__main__":
    main()
