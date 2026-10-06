#!/usr/bin/env python3
"""Validate and freeze the inputs/protocol for the Table 3 v2 replay."""

from __future__ import annotations

import hashlib
import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("tworoom", "pusht", "reacher", "cube")
SEEDS = (42, 100, 2026, 3407, 1234, 4444)
V1 = ROOT / "outputs/cvpr/table3/v1"
V2 = ROOT / "outputs/cvpr/table3/v2"
GUIDED = ("cowm_po", "lewm_po", "cowm_gf")
VARIANTS = {f"{method}{suffix}" for method in GUIDED for suffix in ("", *(f"_random_{i}" for i in range(5)))}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def audit_dataset(task: str) -> dict[str, Any]:
    """Hash the physical dataset once and verify it against the accepted v1 lock."""
    reference_config = read(V1 / "3a" / task / "seed_42" / "frozen_config.json")
    expected = reference_config["dataset"]
    path = ROOT / expected["path"]
    print(f"dataset_sha256_start task={task} size={int(expected['file_size_bytes'])}", flush=True)
    before = path.stat()
    actual_sha = sha(path)
    after = path.stat()
    signature_before = (before.st_size, before.st_mtime_ns)
    signature_after = (after.st_size, after.st_mtime_ns)
    if signature_before != signature_after:
        raise ValueError(f"{task}: dataset changed during the SHA256 audit")
    if before.st_size != int(expected["file_size_bytes"]):
        raise ValueError(f"{task}: dataset size differs from accepted v1 metadata")
    if actual_sha != expected["sha256"]:
        raise ValueError(f"{task}: actual dataset SHA256 differs from accepted v1 metadata")
    print(f"dataset_sha256_done task={task} sha256={actual_sha}", flush=True)
    return {
        "path": expected["path"],
        "sha256": actual_sha,
        "file_size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "audit_method": "full_file_sha256_at_v2_freeze",
        "v1_reference_config_sha256": sha(V1 / "3a" / task / "seed_42" / "frozen_config.json"),
    }


def reuse_dataset_audit(task: str, previous_refs: dict[str, Any]) -> dict[str, Any]:
    """Carry forward a just-completed full hash when the file signature is unchanged."""
    try:
        audit = previous_refs["cells"][f"{task}/seed_42"]["3a"]["dataset"]
    except KeyError as exc:
        raise ValueError(f"previous source refs lack the {task} dataset audit") from exc
    expected = read(V1 / "3a" / task / "seed_42" / "frozen_config.json")["dataset"]
    expected_config_hash = sha(V1 / "3a" / task / "seed_42" / "frozen_config.json")
    path = ROOT / expected["path"]
    stat = path.stat()
    if (
        audit.get("path") != expected["path"]
        or audit.get("sha256") != expected["sha256"]
        or audit.get("v1_reference_config_sha256") != expected_config_hash
        or int(audit.get("file_size_bytes", -1)) != int(stat.st_size)
        or int(audit.get("file_size_bytes", -1)) != int(expected["file_size_bytes"])
        or int(audit.get("mtime_ns", -1)) != int(stat.st_mtime_ns)
    ):
        raise ValueError(f"{task}: cannot reuse dataset SHA audit because identity/signature changed")
    carried = dict(audit)
    carried["audit_method"] = "full_file_sha256_reused_from_prior_v2_freeze_with_unchanged_size_and_mtime"
    return carried


def verify_3a(task: str, seed: int, dataset_audit: dict[str, Any]) -> dict[str, Any]:
    cell = V1 / "3a" / task / f"seed_{seed}"
    acceptance, summary, frozen = (read(cell / name) for name in ("acceptance.json", "summary.json", "frozen_config.json"))
    dataset = frozen.get("dataset", {})
    if dataset.get("path") != dataset_audit["path"] or dataset.get("sha256") != dataset_audit["sha256"]:
        raise ValueError(f"{task}/{seed}: v1 dataset identity differs from the full v2 SHA256 audit")
    if int(dataset.get("file_size_bytes", -1)) != dataset_audit["file_size_bytes"]:
        raise ValueError(f"{task}/{seed}: v1 dataset size differs from the full v2 SHA256 audit")
    if acceptance.get("status") != "pass" or acceptance.get("errors") != []:
        raise ValueError(f"{task}/{seed}: 3a source is not accepted")
    if acceptance.get("summary_sha256") != sha(cell / "summary.json"):
        raise ValueError(f"{task}/{seed}: 3a summary hash mismatch")
    if acceptance.get("fixed_pool_sha256") != sha(cell / "fixed_pool.npz") or summary.get("fixed_pool_sha256") != sha(cell / "fixed_pool.npz"):
        raise ValueError(f"{task}/{seed}: 3a pool hash mismatch")
    hashes = summary.get("branch_file_sha256", {})
    expected_names = {f"candidate_{i:04d}.jsonl" for i in range(64)}
    if set(hashes) != expected_names:
        raise ValueError(f"{task}/{seed}: incomplete 3a branch hash set")
    candidate_rows: dict[int, list[dict[str, Any]]] = {}
    for name, expected_hash in hashes.items():
        path = cell / "branches" / name
        if sha(path) != expected_hash:
            raise ValueError(f"{task}/{seed}: branch hash mismatch: {name}")
        rows = jsonl(path)
        if len(rows) != 50 or [int(r["slot"]) for r in rows] != list(range(50)):
            raise ValueError(f"{task}/{seed}: bad branch row/slot coverage: {name}")
        candidate = int(rows[0]["candidate_index"])
        for row in rows:
            if int(row["candidate_index"]) != candidate:
                raise ValueError(f"{task}/{seed}: mixed candidate indices in {name}")
            for key in ("predicted_cost_cowm_b", "predicted_cost_lewm", "terminal_selection_cost"):
                if not finite(row.get(key)):
                    raise ValueError(f"{task}/{seed}: nonfinite/missing {key} in {name}")
        candidate_rows[candidate] = rows
    if set(candidate_rows) != set(range(64)):
        raise ValueError(f"{task}/{seed}: candidate indices are incomplete")
    pool_path = cell / "fixed_pool.npz"
    import numpy as np
    with np.load(pool_path) as data:
        if data["candidates"].shape[:2] != (50, 64):
            raise ValueError(f"{task}/{seed}: candidate array shape mismatch")
        for matrix_name, row_key in (("cowm_b_costs", "predicted_cost_cowm_b"), ("lewm_costs", "predicted_cost_lewm")):
            matrix = data[matrix_name]
            for candidate, rows in candidate_rows.items():
                values = [float(r[row_key]) for r in rows]
                if not np.array_equal(matrix[:, candidate], np.asarray(values, dtype=np.float64)):
                    raise ValueError(f"{task}/{seed}: pool scores disagree with branch {row_key}")
    return {
        "acceptance_sha256": sha(cell / "acceptance.json"),
        "summary_sha256": sha(cell / "summary.json"),
        "frozen_config_sha256": sha(cell / "frozen_config.json"),
        "fixed_pool_sha256": sha(pool_path),
        "cohort_sha256": summary["cohort_sha256"],
        "actor_checkpoint_sha256": summary["actor_checkpoint_sha256"],
        "lewm_checkpoint_sha256": summary["verifier_checkpoint_sha256"],
        "action_normalizer_sha256": summary["action_normalizer_sha256"],
        "dataset": dataset_audit,
        "candidate_noise_sha256": summary["candidate_noise_sha256"],
        "branch_file_sha256": hashes,
        "row_count": 3200,
    }


def verify_3b(task: str, seed: int, source_3a: dict[str, Any]) -> dict[str, Any]:
    cell = V1 / "3b" / task / f"seed_{seed}"
    acceptance, summary, frozen = (read(cell / name) for name in ("acceptance.json", "summary.json", "frozen_config.json"))
    if acceptance.get("status") != "pass" or acceptance.get("errors") != []:
        raise ValueError(f"{task}/{seed}: 3b source is not accepted")
    if acceptance.get("summary_sha256") != sha(cell / "summary.json") or acceptance.get("frozen_config_sha256") != sha(cell / "frozen_config.json"):
        raise ValueError(f"{task}/{seed}: 3b acceptance hash mismatch")
    if acceptance.get("action_archive_sha256") != sha(cell / "actions.npz") or summary.get("action_archive_sha256") != sha(cell / "actions.npz"):
        raise ValueError(f"{task}/{seed}: 3b action archive hash mismatch")
    if frozen.get("source_fixed_pool_sha256") != source_3a["fixed_pool_sha256"]:
        raise ValueError(f"{task}/{seed}: 3b points to a different 3a candidate pool")
    if frozen.get("protocol", {}).get("improvement_tolerance") != 1e-6:
        raise ValueError(f"{task}/{seed}: v1 3b improvement tolerance changed")
    hashes = summary.get("branch_file_sha256", {})
    if set(hashes) != VARIANTS or int(summary.get("branch_replay_count", -1)) != 18:
        raise ValueError(f"{task}/{seed}: incomplete 3b variant set")
    count = 0
    for name, expected_hash in hashes.items():
        path = cell / "branches" / f"{name}.jsonl"
        if sha(path) != expected_hash:
            raise ValueError(f"{task}/{seed}: 3b branch hash mismatch: {name}")
        rows = jsonl(path)
        if len(rows) != 50 or [int(r["slot"]) for r in rows] != list(range(50)):
            raise ValueError(f"{task}/{seed}: bad 3b branch row/slot coverage: {name}")
        for row in rows:
            if row.get("valid_at_25") is True:
                state = row.get("milestone_state_at_25")
                if not isinstance(state, dict) or not finite(row.get("physical_cost_at_25")):
                    raise ValueError(f"{task}/{seed}: valid 3b step25 row lacks source state/cost: {name}")
            count += 1
    return {
        "acceptance_sha256": sha(cell / "acceptance.json"),
        "summary_sha256": sha(cell / "summary.json"),
        "frozen_config_sha256": sha(cell / "frozen_config.json"),
        "actions_sha256": sha(cell / "actions.npz"),
        "branch_file_sha256": hashes,
        "branch_record_count": count,
        "improvement_tolerance": 1e-6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-dataset-audit-from",
        type=Path,
        help="carry forward a prior full SHA256 audit only when current size and mtime are unchanged",
    )
    args = parser.parse_args()
    V2.mkdir(parents=True, exist_ok=True)
    for name in ("source_refs.json", "frozen_config.json", "freeze_audit.json"):
        if (V2 / name).exists():
            raise FileExistsError(f"v2 frozen artifact already exists; refusing overwrite: {V2 / name}")
    if (V2 / "3a").exists() and any((V2 / "3a").iterdir()):
        raise FileExistsError("v2 3a run artifacts already exist")
    previous_refs = (
        read(args.reuse_dataset_audit_from)
        if args.reuse_dataset_audit_from is not None
        else None
    )
    refs: dict[str, Any] = {"schema_version": 1, "source_protocol": "accepted CVPR Table 3 v1", "cells": {}}
    for task in TASKS:
        dataset_audit = (
            audit_dataset(task)
            if previous_refs is None
            else reuse_dataset_audit(task, previous_refs)
        )
        for seed in SEEDS:
            key = f"{task}/seed_{seed}"
            source_3a = verify_3a(task, seed, dataset_audit)
            source_3b = verify_3b(task, seed, source_3a)
            refs["cells"][key] = {"3a": source_3a, "3b": source_3b}
    refs_path = V2 / "source_refs.json"
    refs_path.write_text(json.dumps(refs, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protocol = {
        "schema_version": 1,
        "experiment": "cvpr_table3_v2",
        "date": "2026-10-05",
        "plan_path": "docs/plan/cvpr_table3_plan_v2.md",
        "plan_sha256": sha(ROOT / "docs/plan/cvpr_table3_plan_v2.md"),
        "source_refs_sha256": sha(refs_path),
        "implementation_code_sha256": {
            path: sha(ROOT / path)
            for path in (
                "scripts/cvpr_table3_v2_freeze.py",
                "scripts/cvpr_table3_v2_replay.py",
                "scripts/cvpr_table3_fixed_pool.py",
                "scripts/cvpr_table3_refinement_report.py",
                "scripts/cvpr_table3_rank_report.py",
                "scripts/cvpr_table3_refinement.py",
                "scripts/round5_phase1_5_diagnostics.py",
                "scripts/round5_phase1_5.py",
                "source/common/checkpoint.py",
                "source/common/eval.py",
                "source/common/round3_phase1.py",
                "source/common/round3_eval.py",
                "source/common/round3_protocol.py",
                "source/common/round4_eval.py",
                "source/common/round5_phase1_5.py",
            )
        },
        "source_policy": "Reuse the exact accepted v1 candidate actions, score matrices, refinement actions, and eligible step25 outcomes; replay all 3a candidates to persist auditable endpoint vectors and traces.",
        "tasks": list(TASKS),
        "evaluation_seeds": list(SEEDS),
        "states_per_task_seed": 50,
        "candidates_per_state": 64,
        "3a": {
            "main_endpoint": "last observed physical state at native termination/truncation or budget step 25",
            "step25_sensitivity": "requires actual step25 state for all 64 candidates",
            "endpoint_cost": "v1 physical selection_cost recomputed from persisted endpoint current/goal vectors",
            "true_oracle_set": "exact float64 equality to the minimum recorded endpoint cost; no epsilon",
            "prediction_tie_break": "lowest frozen candidate_index",
            "spearman": "average rank; constant or nonfinite sequence is NA",
            "hit_at_5": "predicted top five intersects exact true-oracle set",
            "random_hit_at_5": "1-C(64-m,5)/C(64,5) for m exact minimizers",
            "regret": "selected endpoint cost minus minimum endpoint cost; native task units, no cross-task raw macro average",
            "minimum_complete_pool_size": 64,
        },
        "task_physical_definitions": {
            "cube": {
                "current_field": "privileged_block_0_pos",
                "goal_field": "goal_privileged_block_0_pos",
                "field_aliases": ["privileged/block_0_pos", "block_0_pos"],
                "goal_aliases": ["goal_privileged/block_0_pos", "goal_block_0_pos"],
                "state_unit": "m",
                "selection_cost": "L2(current - goal)",
                "selection_cost_unit": "m",
                "success_predicate": "||privileged_block_0_pos - goal_privileged_block_0_pos||_2 <= 0.04",
                "success_thresholds": {"position_l2": 0.04},
                "success_comparison": "less_than_or_equal",
                "angle_wrapping": False,
            },
            "reacher": {
                "current_field": "qpos",
                "goal_field": "goal_qpos",
                "field_aliases": [],
                "goal_aliases": [],
                "state_unit": "rad",
                "selection_cost": "L2(current - goal)",
                "selection_cost_unit": "rad",
                "success_predicate": "all(abs(qpos - goal_qpos) < 0.05)",
                "success_thresholds": {"per_joint_abs": 0.05},
                "success_comparison": "strict_less_than",
                "angle_wrapping": False,
            },
            "pusht": {
                "current_field": "state",
                "goal_field": "goal_state",
                "field_aliases": [],
                "goal_aliases": [],
                "state_unit": "px/rad",
                "selection_cost": "max(L2((current-goal)[:4])/20 px, wrapped_abs(current[4]-goal[4])/(pi/9 rad))",
                "selection_cost_unit": "dimensionless normalized physical error",
                "success_predicate": "||state[:4] - goal_state[:4]||_2 < 20 and circular_abs(state[4] - goal_state[4]) < pi/9",
                "success_thresholds": {"position_l2": 20.0, "angle_abs": math.pi / 9.0},
                "success_comparison": "strict_less_than_for_each_component",
                "angle_wrapping": "abs((delta + pi) mod (2*pi) - pi)",
            },
            "tworoom": {
                "current_field": "proprio",
                "goal_field": "goal_proprio",
                "field_aliases": ["state"],
                "goal_aliases": ["goal_state"],
                "state_unit": "px",
                "selection_cost": "L2(current - goal)",
                "selection_cost_unit": "environment_position_units",
                "success_predicate": "||proprio - goal_proprio||_2 < 16",
                "success_thresholds": {"position_l2": 16.0},
                "success_comparison": "strict_less_than",
                "angle_wrapping": False,
            },
        },
        "3b": {
            "main_endpoint": "actual step25 only, matching the accepted v1 protocol",
            "improvement_tolerance_by_task": {task: 1e-6 for task in TASKS},
            "protocol_source": "all accepted v1 frozen configs use improvement_tolerance=1e-6",
            "random_direction_aggregation": "first average eligible matched directions within state; do not count directions as independent states",
        },
        "statistics": {
            "source_episode_cluster_bootstrap_replicates": 10000,
            "rng_seed": 20261005,
            "same_source_episode_across_seeds": "jointly resampled",
            "estimable_evaluation_seeds": "equal weight; missing seed excluded and reported",
            "primary_holm_family_size": 16,
            "primary_metrics": ["3a/spearman", "3a/hit_at_5", "3a/selection_regret", "3b/cowm_po_minus_matched_random_improvement"],
            "test": "two-sided paired episode-cluster bootstrap null test; Holm over estimable comparisons",
        },
        "model_retraining": False,
        "candidate_regeneration": False,
        "plan_acceptance_required_model": "gpt-6.1-sol",
        "plan_acceptance_required_reasoning_effort": "medium",
        "report_generation_required_model": "gpt-6.1-sol",
        "report_generation_required_reasoning_effort": "medium",
    }
    frozen_path = V2 / "frozen_config.json"
    frozen_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    acceptance = {
        "status": "ready_for_reviewer",
        "validated_3a_cells": 24,
        "validated_3b_cells": 24,
        "validated_3a_candidate_rows": 76800,
        "validated_3b_variant_rows": 21600,
        "source_refs_sha256": sha(refs_path),
        "frozen_config_sha256": sha(frozen_path),
    }
    (V2 / "freeze_audit.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, sort_keys=True))


if __name__ == "__main__":
    main()
