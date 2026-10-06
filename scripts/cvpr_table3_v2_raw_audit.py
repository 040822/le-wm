#!/usr/bin/env python3
"""Main-agent raw integrity audit for CVPR Table 3 v2 replay cells."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "outputs/cvpr/table3/v2"
TASKS = ("tworoom", "pusht", "reacher", "cube")
PILOT_SEED = 42
ATTEMPT = "attempt_001"


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path, compressed: bool = False) -> list[dict[str, Any]]:
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def audit_global_lineage() -> dict[str, str]:
    frozen_path = V2 / "frozen_config.json"
    refs_path = V2 / "source_refs.json"
    frozen = read_json(frozen_path)
    refs = read_json(refs_path)
    frozen_sha = file_sha(frozen_path)
    refs_sha = file_sha(refs_path)
    require(frozen.get("source_refs_sha256") == refs_sha, "source_refs hash differs from parent freeze")
    plan_path = ROOT / frozen["plan_path"]
    require(file_sha(plan_path) == frozen.get("plan_sha256"), "plan hash differs from parent freeze")
    for relative, expected in frozen["implementation_code_sha256"].items():
        require(file_sha(ROOT / relative) == expected, f"implementation hash mismatch: {relative}")

    padding_path = V2 / "analysis/input_padding_audit.json"
    padding = read_json(padding_path)
    require(padding.get("all_pass") is True, "input padding audit did not pass")
    require(padding.get("cells_checked") == 24, "input padding audit did not cover all 24 source pools")
    require(padding.get("source_refs_sha256") == refs_sha, "input padding audit uses different source refs")
    return {
        "frozen_config_sha256": frozen_sha,
        "source_refs_sha256": refs_sha,
        "plan_sha256": frozen["plan_sha256"],
        "input_padding_audit_sha256": file_sha(padding_path),
    }


def audit_cell(task: str, seed: int, lineage: dict[str, str]) -> dict[str, Any]:
    base = V2 / "3a" / task / f"seed_{seed}" / ATTEMPT
    acceptance_path = base / "acceptance.json"
    summary_path = base / "summary.json"
    require(acceptance_path.is_file() and summary_path.is_file(), f"missing acceptance/summary: {task}/seed_{seed}")
    acceptance = read_json(acceptance_path)
    summary = read_json(summary_path)
    cell_frozen_path = base / "frozen_config.json"
    cell_frozen = read_json(cell_frozen_path)
    require(acceptance.get("status") == "pass", f"acceptance status is not pass: {task}/seed_{seed}")
    require(acceptance.get("errors") == [], f"acceptance has errors: {task}/seed_{seed}")
    require(acceptance.get("summary_sha256") == file_sha(summary_path), f"summary hash mismatch: {task}/seed_{seed}")
    require(summary.get("task") == task and summary.get("evaluation_seed") == seed, f"summary identity mismatch: {task}/seed_{seed}")
    require(summary.get("frozen_config_sha256") == file_sha(cell_frozen_path), f"cell frozen config hash mismatch: {task}/seed_{seed}")
    require(cell_frozen.get("global_frozen_config_sha256") == lineage["frozen_config_sha256"], f"parent frozen config mismatch: {task}/seed_{seed}")
    require(cell_frozen.get("global_source_refs_sha256") == lineage["source_refs_sha256"], f"global source refs mismatch: {task}/seed_{seed}")
    replay = summary.get("replay", {})
    require(replay.get("candidate_count") == 64, f"candidate count mismatch: {task}/seed_{seed}")
    require(replay.get("branch_record_count") == replay.get("trace_record_count") == 3200, f"replay row count mismatch: {task}/seed_{seed}")

    state_entries = read_json(base / "state_index.json")["entries"]
    require(len(state_entries) == 50, f"state index count mismatch: {task}/seed_{seed}")
    branch_root = base / "branches"
    trace_root = base / "traces"
    pool_path = base / "fixed_pool.npz"
    branch_hashes = summary["branch_file_sha256"]
    trace_hashes = summary["trace_file_sha256"]
    require(len(branch_hashes) == len(trace_hashes) == 64, f"hash inventory size mismatch: {task}/seed_{seed}")

    endpoint_types: Counter[str] = Counter()
    total_rows = 0
    total_steps = 0
    with np.load(pool_path) as pool:
        candidates = pool["candidates"]
        physical_actions = pool["physical_actions"]
        cowm_costs = pool["cowm_b_costs"]
        lewm_costs = pool["lewm_costs"]
        require(candidates.shape[:2] == physical_actions.shape[:2] == (50, 64), f"fixed pool shape mismatch: {task}/seed_{seed}")

        for candidate_index in range(64):
            stem = f"candidate_{candidate_index:04d}"
            branch_path = branch_root / f"{stem}.jsonl"
            trace_path = trace_root / f"{stem}.jsonl.gz"
            require(branch_path.is_file() and trace_path.is_file(), f"missing branch/trace: {task}/seed_{seed}/{stem}")
            require(branch_hashes.get(branch_path.name) == file_sha(branch_path), f"branch file hash mismatch: {task}/seed_{seed}/{stem}")
            require(trace_hashes.get(trace_path.name) == file_sha(trace_path), f"trace file hash mismatch: {task}/seed_{seed}/{stem}")
            branch_rows = read_jsonl(branch_path)
            trace_rows = read_jsonl(trace_path, compressed=True)
            require(len(branch_rows) == len(trace_rows) == 50, f"per-candidate row count mismatch: {task}/seed_{seed}/{stem}")
            require([r["slot"] for r in branch_rows] == [r["slot"] for r in trace_rows] == list(range(50)), f"slot order mismatch: {task}/seed_{seed}/{stem}")

            for slot, (row, trace_row) in enumerate(zip(branch_rows, trace_rows)):
                state = state_entries[slot]
                require(row["candidate_index"] == trace_row["candidate_index"] == candidate_index, f"candidate index mismatch: {task}/{stem}/{slot}")
                require(row["task"] == trace_row["task"] == task, f"task mismatch: {task}/{stem}/{slot}")
                require(row["slot"] == trace_row["slot"] == state["slot"] == slot, f"slot mismatch: {task}/{stem}/{slot}")
                for key in ("state_id", "episode_id"):
                    require(row[key] == trace_row[key] == state[key], f"state identity mismatch: {task}/{stem}/{slot}/{key}")
                for key in ("row_index", "start_step"):
                    require(row[key] == state[key], f"source row mismatch: {task}/{stem}/{slot}/{key}")
                require(np.array_equal(np.asarray(row["action"], dtype=np.float32), candidates[slot, candidate_index]), f"normalized action mismatch: {task}/{stem}/{slot}")
                require(np.array_equal(np.asarray(row["physical_action"], dtype=np.float32), physical_actions[slot, candidate_index]), f"physical action mismatch: {task}/{stem}/{slot}")
                require(row["predicted_cost_cowm_b"] == float(cowm_costs[slot, candidate_index]), f"CoWM-B score mismatch: {task}/{stem}/{slot}")
                require(row["predicted_cost_lewm"] == float(lewm_costs[slot, candidate_index]), f"LeWM score mismatch: {task}/{stem}/{slot}")
                require(row["trajectory_trace_file"] == trace_path.name, f"trace filename mismatch: {task}/{stem}/{slot}")

                steps = trace_row["steps"]
                require(row["trajectory_trace_sha256"] == canonical_sha(steps), f"trajectory hash mismatch: {task}/{stem}/{slot}")
                require(1 <= len(steps) <= 25, f"invalid trajectory length: {task}/{stem}/{slot}")
                require(row["trajectory_step_count"] == row["effective_valid_length"] == len(steps), f"trajectory length mismatch: {task}/{stem}/{slot}")
                last = steps[-1]
                pairs = (
                    ("endpoint_current", "current"),
                    ("endpoint_goal", "goal"),
                    ("endpoint_raw_env_step", "raw_env_step"),
                    ("endpoint_terminated", "terminated"),
                    ("endpoint_truncated", "truncated"),
                    ("endpoint_env_success", "env_success"),
                    ("endpoint_predicate_success", "predicate_success"),
                    ("endpoint_termination_reason", "termination_reason"),
                    ("endpoint_termination_reason_source", "termination_reason_source"),
                )
                for row_key, step_key in pairs:
                    require(row[row_key] == last[step_key], f"endpoint mismatch: {task}/{stem}/{slot}/{row_key}")
                raw_steps = [step["raw_env_step"] for step in steps]
                require(all(right > left for left, right in zip(raw_steps, raw_steps[1:])), f"raw step order mismatch: {task}/{stem}/{slot}")
                endpoint_types[row["endpoint_type"]] += 1
                total_rows += 1
                total_steps += len(steps)

    return {
        "task": task,
        "seed": seed,
        "acceptance_sha256": file_sha(acceptance_path),
        "summary_sha256": file_sha(summary_path),
        "branch_candidate_files": len(branch_hashes),
        "trace_candidate_files": len(trace_hashes),
        "branch_trace_rows_crosschecked": total_rows,
        "primitive_trace_steps_crosschecked": total_steps,
        "endpoint_types": dict(sorted(endpoint_types.items())),
        "status": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-only", action="store_true", help="audit the four seed-42 pilot cells")
    args = parser.parse_args()
    lineage = audit_global_lineage()
    frozen = read_json(V2 / "frozen_config.json")
    seeds = [PILOT_SEED] if args.pilot_only else frozen["evaluation_seeds"]
    cells = [audit_cell(task, seed, lineage) for task in TASKS for seed in seeds]
    output = {
        "schema_version": 1,
        "execution_role": "main_agent",
        "actual_model": "GPT-6 (exact deployment identifier unavailable)",
        "reasoning_effort": "not_reported",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_scope": "four seed-42 pilot cells" if args.pilot_only else "all 24 planned task-seed cells",
        "script_sha256": file_sha(Path(__file__)),
        "lineage": lineage,
        "cells_checked": len(cells),
        "candidate_rows_crosschecked": sum(c["branch_trace_rows_crosschecked"] for c in cells),
        "primitive_trace_steps_crosschecked": sum(c["primitive_trace_steps_crosschecked"] for c in cells),
        "cells": cells,
        "status": "pass",
    }
    destination = V2 / "analysis" / ("raw_audit_pilot.json" if args.pilot_only else "raw_audit_full.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps({"status": output["status"], "cells_checked": len(cells), "candidate_rows_crosschecked": output["candidate_rows_crosschecked"], "output": str(destination.relative_to(ROOT))}, sort_keys=True))


if __name__ == "__main__":
    main()
