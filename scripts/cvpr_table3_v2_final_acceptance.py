#!/usr/bin/env python3
"""Run the main-agent final acceptance and bind reports to audited inputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "outputs/cvpr/table3/v2"
REPORTS = ROOT / "docs/report/cvpr/table3"
DEST = V2 / "analysis/main_agent_final_acceptance.json"
TASKS = ("cube", "pusht", "reacher", "tworoom")
SEEDS = (42, 100, 2026, 3407, 1234, 4444)
REPORT_NAMES = (
    "cvpr_table3_report.md",
    "cvpr_table3a_ranking_report.md",
    "cvpr_table3b_refinement_report.md",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def render_reports() -> None:
    generator = ROOT / "scripts/cvpr_table3_v2_report.py"
    spec = importlib.util.spec_from_file_location("cvpr_table3_v2_report", generator)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load report generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def audit_raw_branches(full: dict) -> tuple[dict, int, int]:
    cell_map = {(cell["task"], cell["seed"]): cell for cell in full["cells"]}
    aggregate = {
        task: {"rows": 0, "endpoint_types": Counter(), "termination_reasons": Counter(), "lengths": Counter(), "terminated": 0, "truncated": 0}
        for task in TASKS
    }
    branch_files = 0
    for task in TASKS:
        for seed in SEEDS:
            local = aggregate[task]
            rows = 0
            cell_dir = V2 / "3a" / task / f"seed_{seed}" / "attempt_001"
            source_summary = read(cell_dir / "summary.json")
            for index in range(64):
                path = V2 / "3a" / task / f"seed_{seed}" / "attempt_001" / "branches" / f"candidate_{index:04d}.jsonl"
                if not path.is_file():
                    raise ValueError(f"missing branch file: {rel(path)}")
                branch_files += 1
                expected_branch_hash = source_summary.get("branch_file_sha256", {}).get(path.name)
                if expected_branch_hash is None or sha(path) != expected_branch_hash:
                    raise ValueError(f"raw branch SHA256 mismatch: {rel(path)}")
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        rows += 1
                        local["rows"] += 1
                        terminated = bool(row["endpoint_terminated"])
                        truncated = bool(row["endpoint_truncated"])
                        if terminated and truncated:
                            raise ValueError(f"simultaneous termination/truncation in {rel(path)}")
                        local["terminated"] += int(terminated)
                        local["truncated"] += int(truncated)
                        kind = "terminated" if terminated or truncated else "budget_25"
                        local["endpoint_types"][kind] += 1
                        local["termination_reasons"][row["endpoint_termination_reason"]] += 1
                        local["lengths"][int(row["effective_valid_length"])] += 1
            if rows != 50 * 64:
                raise ValueError(f"{task}/seed_{seed} has {rows} rows, expected 3200")
    # Compare per-cell raw branch aggregates to the independently rebuilt summaries.
    for task in TASKS:
        for seed in SEEDS:
            cell = cell_map[(task, seed)]
            cell_counts = {"endpoint_types": Counter(), "termination_reasons": Counter(), "lengths": Counter()}
            for index in range(64):
                path = V2 / "3a" / task / f"seed_{seed}" / "attempt_001" / "branches" / f"candidate_{index:04d}.jsonl"
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        kind = "terminated" if row["endpoint_terminated"] or row["endpoint_truncated"] else "budget_25"
                        cell_counts["endpoint_types"][kind] += 1
                        cell_counts["termination_reasons"][row["endpoint_termination_reason"]] += 1
                        cell_counts["lengths"][int(row["effective_valid_length"])] += 1
            expected = {
                "endpoint_types": cell["endpoint_types"],
                "termination_reasons": cell["termination_reasons"],
                "lengths": {int(k): v for k, v in cell["lengths"].items()},
            }
            for name in expected:
                if dict(cell_counts[name]) != expected[name]:
                    raise ValueError(f"raw branch aggregate mismatch: {task}/seed_{seed}/{name}")
    total_steps = sum(cell["primitive_trace_steps_crosschecked"] for cell in read(V2 / "analysis/raw_audit_full.json")["cells"])
    return aggregate, branch_files, total_steps


def check_report_links(allow_missing_acceptance_link: bool = False) -> None:
    for name in REPORT_NAMES:
        path = REPORTS / name
        if not path.is_file():
            raise ValueError(f"missing report {rel(path)}")
        content = path.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", content):
            if target.startswith(("https://", "http://", "mailto:")):
                continue
            local = target.split("#", 1)[0]
            if not local:
                continue
            resolved = (path.parent / local).resolve()
            if allow_missing_acceptance_link and resolved == DEST and not resolved.exists():
                continue
            if not resolved.is_file():
                raise ValueError(f"broken report link in {rel(path)}: {target}")


def build(allow_missing_acceptance_link: bool = False) -> dict:
    frozen_path = V2 / "frozen_config.json"
    refs_path = V2 / "source_refs.json"
    freeze_audit_path = V2 / "freeze_audit.json"
    plan_path = ROOT / "docs/plan/cvpr_table3_plan_v2.md"
    addendum_path = ROOT / "docs/plan/cvpr_table3_plan_v2_execution_addendum_20261006.md"
    full_path = V2 / "analysis/full.json"
    raw_path = V2 / "analysis/raw_audit_full.json"
    manifest_path = V2 / "launch_manifest.json"
    full, raw, manifest = read(full_path), read(raw_path), read(manifest_path)
    frozen, refs, freeze_audit = read(frozen_path), read(refs_path), read(freeze_audit_path)

    if sha(plan_path) != frozen["plan_sha256"]:
        raise ValueError("frozen plan SHA256 mismatch")
    if sha(frozen_path) != freeze_audit["frozen_config_sha256"]:
        raise ValueError("frozen config SHA256 mismatch")
    if sha(refs_path) != freeze_audit["source_refs_sha256"]:
        raise ValueError("source refs SHA256 mismatch")
    if full.get("processed_cells") != 24 or full.get("expected_full_cells") != 24 or full.get("missing_selected_cells"):
        raise ValueError("full analysis does not contain all 24 cells")
    if raw.get("status") != "pass" or raw.get("cells_checked") != 24 or len(raw.get("cells", [])) != 24:
        raise ValueError("full raw audit did not pass all 24 cells")
    if raw.get("candidate_rows_crosschecked") != 76800:
        raise ValueError("unexpected candidate row audit total")
    if raw.get("script_sha256") != sha(ROOT / "scripts/cvpr_table3_v2_raw_audit.py"):
        raise ValueError("raw audit output does not match the current audit script")
    if full.get("analysis_code_sha256") != sha(ROOT / "scripts/cvpr_table3_v2_analysis.py"):
        raise ValueError("full summary does not match the current analysis script")
    if full.get("analysis_lock_sha256") != sha(V2 / "analysis/analysis_lock.json"):
        raise ValueError("full summary does not match the analysis lock")
    if len(full.get("primary_tests", [])) != 16:
        raise ValueError("planned primary test family is incomplete")
    estimable = [test for test in full["primary_tests"] if test.get("p_two_sided") is not None]
    if len(estimable) != 15:
        raise ValueError("expected 15 estimable primary tests")
    if sum(test.get("p_holm") is not None for test in estimable) != 15:
        raise ValueError("Holm adjustment missing for an estimable test")
    nonestimable = [test for test in full["primary_tests"] if test.get("p_two_sided") is None]
    if len(nonestimable) != 1 or "cube" != nonestimable[0].get("task"):
        raise ValueError("unexpected nonestimable primary contrast")
    if manifest.get("execution_role") != "main_agent" or manifest.get("subagents_used_for_execution") is not False:
        raise ValueError("launch manifest does not establish main-agent-only execution")

    for cell in raw["cells"]:
        task, seed = cell["task"], cell["seed"]
        if task not in TASKS or seed not in SEEDS:
            raise ValueError(f"unexpected raw audit cell: {task}/seed_{seed}")
        cell_path = V2 / "3a" / task / f"seed_{seed}" / "attempt_001"
        if sha(cell_path / "acceptance.json") != cell["acceptance_sha256"]:
            raise ValueError(f"raw acceptance file changed: {task}/seed_{seed}")
        if sha(cell_path / "summary.json") != cell["summary_sha256"]:
            raise ValueError(f"raw summary file changed: {task}/seed_{seed}")
        acceptance = read(cell_path / "acceptance.json")
        if acceptance.get("status") != "pass" or acceptance.get("errors"):
            raise ValueError(f"raw cell acceptance failed: {task}/seed_{seed}")
        if acceptance.get("summary_sha256") != cell["summary_sha256"]:
            raise ValueError(f"acceptance-to-summary hash mismatch: {task}/seed_{seed}")

    aggregate, branch_files, raw_steps = audit_raw_branches(full)
    if branch_files != 24 * 64:
        raise ValueError("unexpected raw branch file count")
    if sum(cell.get("status") == "pass" for cell in raw["cells"]) != 24:
        raise ValueError("raw audit contains a failed cell")

    # The 3b source cross-check is recorded for every replayed task-seed cell.
    if len(full["cells"]) != 24 or any("source_3b_baseline_crosscheck" not in c for c in full["cells"]):
        raise ValueError("3b candidate-0/source improvement cross-check missing")
    max_delta_error = max(c["source_3b_baseline_crosscheck"]["max_absolute_delta_difference"] for c in full["cells"])
    if max_delta_error > 1e-6:
        raise ValueError("3b source improvement mismatch exceeds frozen tolerance")

    if not (REPORTS / "cvpr_table3c_probe_report.md").is_file() or not (REPORTS / "cvpr_table3c_result_check.json").is_file():
        raise ValueError("retained 3c report or check is missing")
    for old in (
        "cvpr_table3_report.md",
        "cvpr_table3a_ranking_report.md",
        "cvpr_table3b_refinement_report.md",
    ):
        if not (REPORTS / "archived" / old).is_file():
            raise ValueError(f"archived v1 report missing: {old}")

    check_report_links(allow_missing_acceptance_link)
    files = [
        plan_path,
        addendum_path,
        frozen_path,
        refs_path,
        freeze_audit_path,
        V2 / "analysis/analysis_lock.json",
        V2 / "analysis/raw_audit_full.json",
        V2 / "analysis/full.json",
        ROOT / "scripts/cvpr_table3_v2_raw_audit.py",
        ROOT / "scripts/cvpr_table3_v2_analysis.py",
        ROOT / "scripts/cvpr_table3_v2_report.py",
        ROOT / "scripts/cvpr_table3_v2_final_acceptance.py",
        manifest_path,
        *(REPORTS / name for name in REPORT_NAMES),
        REPORTS / "cvpr_table3c_probe_report.md",
        REPORTS / "cvpr_table3c_result_check.json",
    ]
    by_task = {}
    for task in TASKS:
        types = aggregate[task]["endpoint_types"]
        reasons = aggregate[task]["termination_reasons"]
        lengths = aggregate[task]["lengths"]
        by_task[task] = {
            "raw_branch_rows": aggregate[task]["rows"],
            "terminated": aggregate[task]["terminated"],
            "truncated": aggregate[task]["truncated"],
            "budget_25": types.get("budget_25", 0),
            "termination_reasons": dict(reasons),
            "length_lt_25": sum(v for k, v in lengths.items() if k < 25),
            "length_eq_25": lengths.get(25, 0),
        }
    return {
        "schema_version": 1,
        "status": "pass",
        "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": {
            "role": "main_agent",
            "actual_model": full.get("actual_model"),
            "reasoning_effort": full.get("reasoning_effort"),
            "subagents_used": False,
            "execution_steering": rel(addendum_path),
            "execution_steering_sha256": sha(addendum_path),
            "scope_note": "用户最新指示要求 main-agent-only；此要求通过执行补充覆盖 v2 正文原先的 subagent 分工。未声称独立 subagent 复核。",
        },
        "commands": [
            "python scripts/cvpr_table3_v2_raw_audit.py",
            "python scripts/cvpr_table3_v2_analysis.py",
            "python scripts/cvpr_table3_v2_report.py",
            "python scripts/cvpr_table3_v2_final_acceptance.py",
        ],
        "inputs_sha256": {rel(path): sha(path) for path in files},
        "coverage": {
            "tasks": list(TASKS),
            "evaluation_seeds": list(SEEDS),
            "task_seed_cells": raw["cells_checked"],
            "candidate_rows_crosschecked": raw["candidate_rows_crosschecked"],
            "raw_branch_files_reopened": branch_files,
            "primitive_trace_steps_crosschecked": raw_steps,
            "source_state_appearances_per_task": 300,
            "endpoint_coverage_by_task": by_task,
        },
        "statistics": {
            "bootstrap_replicates": full["statistics"]["replicates"],
            "rng_seed": full["statistics"]["rng_seed"],
            "planned_primary_tests": 16,
            "estimable_primary_tests": 15,
            "holm_adjusted_estimable_tests": 15,
            "nonestimable_primary_test": nonestimable[0]["comparison"],
            "nonestimable_reason": nonestimable[0].get("nonestimable_reason"),
            "3b_source_candidate0_max_abs_delta_error": max_delta_error,
        },
        "reports": {name: sha(REPORTS / name) for name in REPORT_NAMES},
        "checks": {
            "frozen_plan_and_source_hash_chain": "pass",
            "24_cell_raw_replay_audit": "pass",
            "complete_candidate_pool_and_trace_recomputation": "pass",
            "endpoint_flags_lengths_reasons_crosschecked_against_raw_branch_rows": "pass",
            "3b_candidate0_and_paired_improvement_source_crosscheck": "pass",
            "statistical_family_and_holm_reconstruction": "pass",
            "reports_rendered_from_full_machine_summary": "pass",
            "report_links_and_v1_archive_and_retained_3c": "pass",
            "main_agent_only_execution_and_review": "pass",
        },
        "interpretation_and_limits": [
            "3a 的变长真实终点及其检验族是在观察 v1 后形成的 benchmark 扩展，不是独立确认性结果。",
            "单 checkpoint 的评估 seed 不是训练重复。",
            "3b 的真实第 25 步覆盖受提前终止影响；Cube 无有效配对，TwoRoom 的 CoWM-PO 只有 4 个有效状态。",
            "运行时仅暴露 GPT-6 家族标识，精确部署名和 reasoning effort 未报告；验收不推测未提供的信息。",
            "analysis/full.json 保留 is_final_acceptance=false 作为机器重算产物的状态标记；本独立文件记录 main-agent 最终验收并通过哈希绑定。",
        ],
    }


def verify_existing(expected: dict) -> None:
    if not DEST.is_file():
        raise ValueError("final acceptance record is missing")
    stored = read(DEST)
    for key, value in expected.items():
        if key == "accepted_at_utc":
            continue
        if stored.get(key) != value:
            raise ValueError(f"final acceptance record differs from current evidence: {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify the existing acceptance record without rewriting it")
    args = parser.parse_args()
    if not args.check:
        render_reports()
    if args.check:
        payload = build()
        verify_existing(payload)
        print(json.dumps({"status": "pass", "acceptance": rel(DEST), "reports_verified": len(REPORT_NAMES)}))
        return
    payload = build(allow_missing_acceptance_link=True)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_existing(build())
    print(json.dumps({"status": "pass", "acceptance": rel(DEST), "reports_verified": len(REPORT_NAMES)}))


if __name__ == "__main__":
    main()
