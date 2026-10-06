#!/usr/bin/env python3
"""Compare the complete Phase 6.3 E0 episodes with frozen Table 1 outcomes."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cvpr_table1 as table
from scripts.round5_phase6_3 import MAIN, OUT, METHODS
from scripts.round5_phase6_3_closed_loop_analysis import load_cell
from source.common.cvpr_table1 import evaluation_cells


def _main_result(cell):
    parent = MAIN / "runs" / cell["cell_id"]
    for path in sorted(parent.glob("attempt_*/result.json"), reverse=True):
        result = table._read_json(path)
        if result.get("status") != "ok" or len(result.get("episodes", [])) != 50:
            continue
        if result.get("cvpr_table1", {}).get("cell_id") != cell["cell_id"]:
            continue
        trace = path.parent / "episodes.jsonl"
        if (not trace.is_file() or
                result.get("cvpr_table1", {}).get("trace_sha256") != table._sha256(trace)):
            continue
        return result, path
    return None, None


def _episode_outcome(episode):
    """Keep the forensic mismatch record small and focused on its outcome."""
    return {
        "success": bool(episode.get("success")),
        "steps_executed": int(episode.get("steps_executed", len(episode.get("steps", [])))),
        "first_success_step": episode.get("first_success_step"),
        "terminal_distance": episode.get("terminal_distance"),
        "episode_replan_count": episode.get("episode_replan_count"),
        "early_terminated": episode.get("early_terminated"),
    }


def _max_abs_action_difference(left, right):
    steps = min(len(left.get("steps", [])), len(right.get("steps", [])))
    differences = []
    for index in range(steps):
        a = left["steps"][index].get("action", [])
        b = right["steps"][index].get("action", [])
        if len(a) != len(b):
            return None
        differences.extend(abs(float(x) - float(y)) for x, y in zip(a, b))
    return max(differences, default=0.0)


def _initial_current_difference(left, right):
    try:
        a = left["steps"][0]["current"]
        b = right["steps"][0]["current"]
        if len(a) != len(b):
            return None
        return max(abs(float(x) - float(y)) for x, y in zip(a, b))
    except (IndexError, KeyError, TypeError):
        return None


def compare():
    phase_source_hashes = table._read_json(OUT / "frozen_config.json")["source_hashes"]
    conditions = [c for c in evaluation_cells() if c["family"] == "cowm"
                  and c["id"] in METHODS and c["method_id"].endswith("__main")]
    rows = []
    for cell in conditions:
        baseline, baseline_path = _main_result(cell)
        current, current_path = load_cell("E0", cell["id"], cell["task"], cell["evaluation_seed"])
        row = {"condition_id": cell["cell_id"], "method": cell["id"], "task": cell["task"],
               "seed": cell["evaluation_seed"], "status": "missing"}
        if baseline is None or current is None:
            row["missing_main_table"] = baseline is None
            row["missing_E0"] = current is None
            rows.append(row)
            continue
        left = {(int(e["dataset_episode"]), int(e["start_step"])): bool(e["success"])
                for e in baseline["episodes"]}
        right = {(int(e["dataset_episode"]), int(e["start_step"])): bool(e["success"])
                 for e in current["episodes"]}
        left_episodes = {(int(e["dataset_episode"]), int(e["start_step"])): e
                         for e in baseline["episodes"]}
        right_episodes = {(int(e["dataset_episode"]), int(e["start_step"])): e
                          for e in current["episodes"]}
        cohort_equal = baseline.get("cohort_sha256") == current.get("cohort_sha256")
        mismatch_keys = left.keys() != right.keys()
        differing = (sum(left[key] != right[key] for key in left.keys() & right.keys())
                     if not mismatch_keys else None)
        main_identity = table._read_json(baseline_path.parent / "cell_identity.json")["identity"]
        main_source_hashes = main_identity.get("code_snapshot_sha256", {})
        shared_source_paths = sorted(set(main_source_hashes) & set(phase_source_hashes))
        source_differences = [path for path in shared_source_paths
                              if main_source_hashes[path] != phase_source_hashes[path]]
        mismatch_details = []
        for key in sorted(left.keys() & right.keys()):
            if left[key] == right[key]:
                continue
            a, b = left_episodes[key], right_episodes[key]
            mismatch_details.append({
                "dataset_episode": key[0], "start_step": key[1],
                "main": _episode_outcome(a), "E0": _episode_outcome(b),
                "common_prefix_max_abs_action_difference": _max_abs_action_difference(a, b),
                "first_recorded_current_max_abs_difference": _initial_current_difference(a, b),
            })
        row.update({"status": "complete" if cohort_equal and not mismatch_keys and differing == 0 else "mismatch",
                    "episodes": len(left), "cohort_sha256_equal": cohort_equal,
                    "episode_keys_equal": not mismatch_keys, "success_mismatches": differing,
                    "mismatched_episodes": mismatch_details,
                    "main_result": str(baseline_path.relative_to(ROOT)),
                    "E0_result": str(current_path.relative_to(ROOT)),
                    "main_trace_sha256": baseline["cvpr_table1"]["trace_sha256"],
                    "E0_trace_sha256": current["cvpr_table1"]["trace_sha256"],
                    "main_gpu": baseline.get("cvpr_table1", {}).get("preflight", {}).get("gpu"),
                    "E0_gpu": current.get("cvpr_table1", {}).get("preflight", {}).get("gpu"),
                    "git_commit_equal": baseline.get("code_commit") == current.get("code_commit"),
                    "main_source_snapshot_files": len(main_source_hashes),
                    "E0_source_snapshot_files": len(phase_source_hashes),
                    "overlapping_source_files": len(shared_source_paths),
                    "matching_overlapping_source_files": len(shared_source_paths) - len(source_differences),
                    "differing_overlapping_source_files": source_differences})
        rows.append(row)

    report = {
        "scope": "outcome comparison of E0 vs frozen Table 1 on matched task/method/seed cohorts; code snapshots are compared separately",
        "source_snapshot_comparison": {
            "E0_source_snapshot_files": len(phase_source_hashes),
            "main_source_snapshot_files": len(main_source_hashes),
            "overlapping_files": len(shared_source_paths),
            "matching_files": len(shared_source_paths) - len(source_differences),
            "differing_files": source_differences,
        },
        "complete_conditions": sum(row["status"] == "complete" for row in rows),
        "mismatched_conditions": sum(row["status"] == "mismatch" for row in rows),
        "missing_conditions": sum(row["status"] == "missing" for row in rows),
        "target_conditions": 96,
        "conditions": rows,
    }
    path = OUT / "summary/baseline_closed_loop_parity.json"
    table._write_json(path, report)
    print(json.dumps({"output": str(path.relative_to(ROOT)),
                      "complete": report["complete_conditions"],
                      "mismatched": report["mismatched_conditions"],
                      "missing": report["missing_conditions"]}))


if __name__ == "__main__":
    compare()
