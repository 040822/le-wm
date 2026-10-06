#!/usr/bin/env python3
"""Verify and aggregate accepted CVPR Table 3a fixed-pool ranking cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POOL_ROOT = ROOT / "outputs/cvpr/table3/v1/3a"
DEFAULT_OUTPUT = POOL_ROOT / "aggregate"
TASKS = ("tworoom", "pusht", "reacher", "cube")
SEEDS = (42, 100, 2026, 3407, 1234, 4444)
SELECTORS = ("cowm_b", "lewm")
METRICS = {
    "top1_success_25": "Selected success within 25 steps",
    "success_oracle_pool_25": "Pool success coverage",
    "uniform_random_expected_success_25": "Uniform-pool expected success",
    "uniform_random_expected_cost_at_25": "Uniform-pool expected physical cost (complete pool)",
    "complete_64_valid_state_fraction": "States with all 64 valid step-25 outcomes",
    "complete_64_selected_physical_regret_at_25": "Complete-pool selected physical regret",
    "complete_64_spearman_cost_vs_physical_at_25": "Complete-pool cost/physical Spearman",
    "common_valid_subpool_candidate_count": "Mean valid candidate count at step 25",
    "common_valid_subpool_selected_success_25": "Common-valid-subpool selected success",
    "common_valid_subpool_selected_physical_regret_at_25": "Common-valid-subpool selected physical regret",
    "common_valid_subpool_spearman_cost_vs_physical_at_25": "Common-valid-subpool cost/physical Spearman",
    "uniform_random_expected_cost_in_valid_subpool_at_25": "Uniform-pool expected physical cost (common valid subpool)",
    "valid_fraction_5": "Valid candidate fraction at step 5",
    "valid_fraction_25": "Valid candidate fraction at step 25",
}
PAIRED_METRICS = (
    "top1_success_25",
    "complete_64_selected_physical_regret_at_25",
    "complete_64_spearman_cost_vs_physical_at_25",
    "common_valid_subpool_selected_physical_regret_at_25",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and np.isfinite(value)


def _summary(values: list[Any]) -> dict[str, Any]:
    finite = [float(value) for value in values if _finite_number(value)]
    return {
        "mean": float(np.mean(finite)) if finite else None,
        "sample_std": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
        "n": len(finite),
    }


def _verify_cell(task: str, seed: int) -> dict[str, Any]:
    cell = POOL_ROOT / task / f"seed_{seed}"
    acceptance_path = cell / "acceptance.json"
    summary_path = cell / "summary.json"
    if not acceptance_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"{task}/seed_{seed}: accepted summary artifacts are missing")
    acceptance = _read_json(acceptance_path)
    summary = _read_json(summary_path)
    if acceptance.get("status") != "pass" or acceptance.get("errors"):
        raise RuntimeError(f"{task}/seed_{seed}: source 3a acceptance is not pass")
    if acceptance.get("summary_sha256") != _sha256(summary_path):
        raise RuntimeError(f"{task}/seed_{seed}: summary hash differs from acceptance record")
    if summary.get("task") != task or int(summary.get("evaluation_seed", -1)) != seed:
        raise RuntimeError(f"{task}/seed_{seed}: summary identity mismatch")
    if int(summary.get("state_count", -1)) != 50 or int(summary.get("candidate_count", -1)) != 64:
        raise RuntimeError(f"{task}/seed_{seed}: expected a 50-state, 64-candidate cell")
    if (
        int(summary.get("flow_steps", -1)) != 2
        or summary.get("integrator") != "euler"
        or summary.get("action_bound_mode") != "none"
        or int(summary.get("replay_budget", -1)) != 25
    ):
        raise RuntimeError(f"{task}/seed_{seed}: frozen 3a replay protocol differs from plan")
    interop = summary.get("action_interop_audit", {})
    if interop.get("status") != "pass":
        raise RuntimeError(f"{task}/seed_{seed}: action-coordinate audit did not pass")
    required_checks = {
        "candidate_pool_identical_for_cowm_and_lewm": True,
        "candidate_noise_identical_for_cowm_and_lewm": True,
        "every_state_has_candidate_indices_0_through_63": True,
        "restore_repeatability": True,
        "action_coordinate_roundtrip": True,
        "branch_files_sha256_recorded": True,
        "replay_budget": 25,
        "no_candidate_projection": True,
    }
    checks = acceptance.get("checks", {})
    if any(checks.get(name) != expected for name, expected in required_checks.items()):
        raise RuntimeError(f"{task}/seed_{seed}: required source acceptance checks are incomplete")
    initial_state_strategy = summary.get(
        "branch_initial_state_strategy", "captured_environment_and_rng_snapshot"
    )
    shared_snapshot_replay = checks.get(
        "all_64_candidates_replayed_from_shared_snapshot"
    ) is True
    seeded_start_replay = checks.get(
        "all_64_candidates_replayed_from_seeded_start_state"
    ) is True
    if initial_state_strategy == "captured_environment_and_rng_snapshot":
        if not shared_snapshot_replay:
            raise RuntimeError(f"{task}/seed_{seed}: shared snapshot replay acceptance is missing")
    elif initial_state_strategy == "seeded_dataset_reset_with_fresh_branch_snapshot":
        if not seeded_start_replay:
            raise RuntimeError(f"{task}/seed_{seed}: seeded-start replay acceptance is missing")
        if checks.get("same_start_state_replayability") is not True or checks.get("dataset_reset_seed_applied") is not True:
            raise RuntimeError(f"{task}/seed_{seed}: deterministic PushT reset acceptance is incomplete")
    else:
        raise RuntimeError(f"{task}/seed_{seed}: unrecognized branch initial-state strategy")
    expected_world_lifecycle = (
        "fresh_world_with_seeded_dataset_reset"
        if initial_state_strategy == "seeded_dataset_reset_with_fresh_branch_snapshot"
        else "persistent_world_with_snapshot_restore"
    )
    reported_world_lifecycle = summary.get("branch_world_lifecycle")
    if reported_world_lifecycle is not None and reported_world_lifecycle != expected_world_lifecycle:
        raise RuntimeError(f"{task}/seed_{seed}: branch world lifecycle differs from its reset strategy")
    if "world_lifecycle_matches_protocol" in checks and checks.get("world_lifecycle_matches_protocol") is not True:
        raise RuntimeError(f"{task}/seed_{seed}: world lifecycle acceptance check failed")
    has_legacy_cohort_check = checks.get("table1_50_episode_cohort") is True
    has_state_appearance_check = checks.get("table1_50_state_appearances") is True
    has_legacy_episode_identity_check = checks.get("unique_source_episode_ids") is True
    has_state_identity_check = checks.get("unique_source_state_ids") is True
    if not (has_legacy_cohort_check or has_state_appearance_check):
        raise RuntimeError(f"{task}/seed_{seed}: source cohort appearance count was not accepted")
    if not (has_legacy_episode_identity_check or has_state_identity_check):
        raise RuntimeError(f"{task}/seed_{seed}: source identity acceptance check is missing")
    if has_state_identity_check and (
        checks.get("duplicate_source_episode_appearances_preserved") is not True
        or checks.get("episode_cluster_bootstrap") is not True
    ):
        raise RuntimeError(f"{task}/seed_{seed}: repeated source episodes were not explicitly preserved and clustered")
    if acceptance.get("fixed_pool_sha256") != summary.get("fixed_pool_sha256"):
        raise RuntimeError(f"{task}/seed_{seed}: accepted pool hash differs from summary")
    pool_path = cell / "fixed_pool.npz"
    if not pool_path.is_file() or _sha256(pool_path) != summary.get("fixed_pool_sha256"):
        raise RuntimeError(f"{task}/seed_{seed}: fixed-pool archive hash mismatch")
    restore = summary.get("restore_check", {})
    if restore.get("status") != "pass":
        raise RuntimeError(f"{task}/seed_{seed}: restore-and-repeat audit did not pass")
    if restore.get("replay_strategy", initial_state_strategy) != initial_state_strategy:
        raise RuntimeError(f"{task}/seed_{seed}: replay audit strategy differs from frozen protocol")
    if initial_state_strategy == "seeded_dataset_reset_with_fresh_branch_snapshot":
        if (
            restore.get("same_start_state_replayability") is not True
            or restore.get("dataset_reset_seed_applied") is not True
            or restore.get("captured_snapshot_reused") is not False
            or int(restore.get("checked_states", -1)) != 50
        ):
            raise RuntimeError(f"{task}/seed_{seed}: seeded PushT replay audit is incomplete")

    branch_hashes = summary.get("branch_file_sha256", {})
    expected_names = {f"candidate_{index:04d}.jsonl" for index in range(64)}
    if set(branch_hashes) != expected_names:
        raise RuntimeError(f"{task}/seed_{seed}: summary does not list all 64 candidate branches")
    expected_identity: dict[int, tuple[Any, Any]] = {}
    for candidate_index in range(64):
        name = f"candidate_{candidate_index:04d}.jsonl"
        branch_path = cell / "branches" / name
        if not branch_path.is_file() or _sha256(branch_path) != branch_hashes[name]:
            raise RuntimeError(f"{task}/seed_{seed}: missing or changed branch {name}")
        rows = [
            json.loads(line)
            for line in branch_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(rows) != 50 or any(int(row.get("candidate_index", -1)) != candidate_index for row in rows):
            raise RuntimeError(f"{task}/seed_{seed}: {name} must contain 50 rows for candidate {candidate_index}")
        if any(row.get("outcome_status") != "completed" for row in rows):
            raise RuntimeError(f"{task}/seed_{seed}: {name} contains an incomplete outcome")
        slots = [int(row["slot"]) for row in rows]
        if sorted(slots) != list(range(50)):
            raise RuntimeError(f"{task}/seed_{seed}: {name} has duplicate or missing state slots")
        for row in rows:
            slot = int(row["slot"])
            identity = (row.get("state_id"), row.get("episode_id"))
            if candidate_index == 0:
                expected_identity[slot] = identity
            elif identity != expected_identity[slot]:
                raise RuntimeError(f"{task}/seed_{seed}: state identity changed at slot {slot}, candidate {candidate_index}")

    unique_state_ids = {identity[0] for identity in expected_identity.values()}
    unique_episode_ids = {
        json.dumps(identity[1], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for identity in expected_identity.values()
    }
    if len(expected_identity) != 50 or len(unique_state_ids) != 50:
        raise RuntimeError(f"{task}/seed_{seed}: cohort slots must map to 50 unique source states")
    identity_counts = {
        "appearance_count": len(expected_identity),
        "unique_source_state_count": len(unique_state_ids),
        "unique_source_episode_count": len(unique_episode_ids),
        "duplicate_source_episode_appearances": len(expected_identity) - len(unique_episode_ids),
    }
    reported_identity_counts = summary.get("cohort_identity")
    if reported_identity_counts is not None:
        for name, expected in identity_counts.items():
            if int(reported_identity_counts.get(name, -1)) != expected:
                raise RuntimeError(f"{task}/seed_{seed}: cohort identity count {name} differs from branch rows")

    metrics = summary.get("metrics", {})
    for selector in SELECTORS:
        selector_metrics = metrics.get(selector, {})
        if int(selector_metrics.get("state_count", -1)) != 50 or int(selector_metrics.get("candidate_count", -1)) != 64:
            raise RuntimeError(f"{task}/seed_{seed}: missing complete ranking metrics for {selector}")
        if len(selector_metrics.get("by_state", [])) != 50:
            raise RuntimeError(f"{task}/seed_{seed}: {selector} per-state results are incomplete")
        means = selector_metrics.get("mean", {})
        if any(metric not in means for metric in METRICS):
            raise RuntimeError(f"{task}/seed_{seed}: {selector} is missing a required ranking metric")
    paired = summary.get("paired_lewm_minus_cowm_b", {})
    if any(metric not in paired for metric in PAIRED_METRICS):
        raise RuntimeError(f"{task}/seed_{seed}: paired selector comparisons are incomplete")
    if any(int(paired[metric].get("replicates", 0)) != 10_000 for metric in PAIRED_METRICS):
        raise RuntimeError(f"{task}/seed_{seed}: paired source-episode bootstrap is not 10,000 replicates")
    return {
        "summary_sha256": _sha256(summary_path),
        "fixed_pool_sha256": summary["fixed_pool_sha256"],
        "branch_count": len(branch_hashes),
        "candidate_outcome_rows": 50 * 64,
        "cohort_identity": identity_counts,
        "branch_initial_state_strategy": initial_state_strategy,
        "branch_world_lifecycle": reported_world_lifecycle or expected_world_lifecycle,
        "summary": summary,
    }


def _aggregate_cells(cells: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    by_task: dict[str, Any] = {}
    by_seed_macro: dict[str, Any] = {}
    for task in TASKS:
        by_task[task] = {}
        for selector in SELECTORS:
            by_task[task][selector] = {}
            for metric in METRICS:
                values = [
                    cells[task][seed]["summary"]["metrics"][selector]["mean"].get(metric)
                    for seed in SEEDS
                ]
                by_task[task][selector][metric] = {
                    **_summary(values),
                    "by_evaluation_seed": {str(seed): value for seed, value in zip(SEEDS, values)},
                }
    for selector in SELECTORS:
        by_seed_macro[selector] = {}
        for metric in METRICS:
            seed_values: dict[str, Any] = {}
            for seed in SEEDS:
                task_values = [
                    cells[task][seed]["summary"]["metrics"][selector]["mean"].get(metric)
                    for task in TASKS
                ]
                finite = [float(value) for value in task_values if _finite_number(value)]
                seed_values[str(seed)] = {
                    "mean": float(np.mean(finite)) if finite else None,
                    "task_count": len(finite),
                    "tasks": TASKS,
                }
            across = _summary([row["mean"] for row in seed_values.values()])
            task_counts = [int(row["task_count"]) for row in seed_values.values()]
            by_seed_macro[selector][metric] = {
                **across,
                "by_evaluation_seed": seed_values,
                "task_count_by_seed": {seed: row["task_count"] for seed, row in seed_values.items()},
                "task_count_range": [min(task_counts), max(task_counts)],
                "aggregation": "mean tasks within each evaluation seed, then mean/sample std across seeds",
            }
    paired: dict[str, Any] = {}
    for task in TASKS:
        paired[task] = {}
        for metric in PAIRED_METRICS:
            values = [
                cells[task][seed]["summary"]["paired_lewm_minus_cowm_b"][metric].get(
                    "mean_difference_left_minus_right"
                )
                for seed in SEEDS
            ]
            paired[task][metric] = {
                **_summary(values),
                "difference_sign": "LeWM minus CoWM-B; positive means a higher metric value for LeWM",
                "by_evaluation_seed": {str(seed): value for seed, value in zip(SEEDS, values)},
            }
    paired_macro: dict[str, Any] = {}
    for metric in PAIRED_METRICS:
        seed_values: dict[str, Any] = {}
        for seed in SEEDS:
            task_values = [
                cells[task][seed]["summary"]["paired_lewm_minus_cowm_b"][metric].get(
                    "mean_difference_left_minus_right"
                )
                for task in TASKS
            ]
            finite = [float(value) for value in task_values if _finite_number(value)]
            seed_values[str(seed)] = {
                "mean": float(np.mean(finite)) if finite else None,
                "task_count": len(finite),
            }
        counts = [row["task_count"] for row in seed_values.values()]
        paired_macro[metric] = {
            **_summary([row["mean"] for row in seed_values.values()]),
            "by_evaluation_seed": seed_values,
            "task_count_range": [min(counts), max(counts)],
            "difference_sign": "LeWM minus CoWM-B; positive means a higher metric value for LeWM",
        }
    return {
        "by_task_across_evaluation_seeds": by_task,
        "macro_across_tasks_by_evaluation_seed": by_seed_macro,
        "paired_lewm_minus_cowm_b_across_evaluation_seeds": paired,
        "paired_lewm_minus_cowm_b_macro_across_tasks_by_evaluation_seed": paired_macro,
    }


def _format(summary: dict[str, Any]) -> str:
    mean, sd, count = summary.get("mean"), summary.get("sample_std"), summary.get("n", 0)
    if mean is None:
        return f"NA (n={count})"
    if sd is None:
        return f"{mean:.4f} (n={count})"
    return f"{mean:.4f} ± {sd:.4f} (n={count})"


def _markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        "# CVPR Table 3a fixed-pool ranking summary",
        "",
        "Each task entry reports the mean and sample standard deviation across the six Table 1 evaluation seeds. The macro section first averages task means within each seed, then summarizes those six seed-level means. Undefined continuous metrics remain NA when their planned valid denominator is empty.",
        "",
    ]
    task_values = aggregate["by_task_across_evaluation_seeds"]
    for task in TASKS:
        lines.extend(
            [
                f"## {task.title()}",
                "",
                "| Selector | Metric | Mean ± sample SD across seeds |",
                "|---|---|---:|",
            ]
        )
        for selector in SELECTORS:
            for metric, label in METRICS.items():
                value = task_values[task][selector][metric]
                lines.append(f"| {selector} | {label} | {_format(value)} |")
        lines.append("")
    lines.extend(
        [
            "## Macro average across tasks",
            "",
            "| Selector | Metric | Mean ± sample SD across evaluation seeds | Available tasks per seed |",
            "|---|---|---:|---:|",
        ]
    )
    macro = aggregate["macro_across_tasks_by_evaluation_seed"]
    for selector in SELECTORS:
        for metric, label in METRICS.items():
            value = macro[selector][metric]
            task_min, task_max = value["task_count_range"]
            lines.append(
                f"| {selector} | {label} | {_format(value)} | {task_min}–{task_max}/{len(TASKS)} |"
            )
    lines.extend(
        [
            "",
            "All paired differences are LeWM minus CoWM-B, so positive values mean a higher metric value for LeWM. For success metrics, higher favors LeWM; for regret, lower favors LeWM. These are descriptive differences across evaluation seeds; per-cell source-episode-cluster bootstrap intervals remain in the source summaries.",
            "",
            "## Paired selector differences",
            "",
            "| Task | Metric | LeWM − CoWM-B mean ± sample SD |",
            "|---|---|---:|",
        ]
    )
    paired = aggregate["paired_lewm_minus_cowm_b_across_evaluation_seeds"]
    for task in TASKS:
        for metric in PAIRED_METRICS:
            lines.append(f"| {task} | {metric} | {_format(paired[task][metric])} |")
    paired_macro = aggregate["paired_lewm_minus_cowm_b_macro_across_tasks_by_evaluation_seed"]
    lines.extend(
        [
            "",
            "### Paired macro difference across tasks",
            "",
            "| Metric | LeWM − CoWM-B mean ± sample SD across seeds | Available tasks per seed |",
            "|---|---:|---:|",
        ]
    )
    for metric in PAIRED_METRICS:
        value = paired_macro[metric]
        task_min, task_max = value["task_count_range"]
        lines.append(
            f"| {metric} | {_format(value)} | {task_min}–{task_max}/{len(TASKS)} |"
        )
    return "\n".join(lines) + "\n"


def rebuild(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    cells: dict[str, dict[int, dict[str, Any]]] = {}
    for task in TASKS:
        cells[task] = {}
        for seed in SEEDS:
            cells[task][seed] = _verify_cell(task, seed)
    aggregate = _aggregate_cells(cells)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "experiment": "cvpr_table3_fixed_pool_rank_aggregate_v1",
        "task_order": TASKS,
        "evaluation_seeds": SEEDS,
        "selector_order": SELECTORS,
        "metric_labels": METRICS,
        "source_cells": {
            task: {
                str(seed): {
                    "summary_sha256": cells[task][seed]["summary_sha256"],
                    "fixed_pool_sha256": cells[task][seed]["fixed_pool_sha256"],
                    "branch_count": cells[task][seed]["branch_count"],
                    "candidate_outcome_rows": cells[task][seed]["candidate_outcome_rows"],
                    "cohort_identity": cells[task][seed]["cohort_identity"],
                }
                for seed in SEEDS
            }
            for task in TASKS
        },
        **aggregate,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path = output_dir / "table3a_ranking.md"
    markdown_path.write_text(_markdown(aggregate), encoding="utf-8")
    acceptance = {
        "status": "pass",
        "errors": [],
        "checks": {
            "accepted_task_seed_cells": len(TASKS) * len(SEEDS),
            "all_64_branch_hashes_reverified_per_cell": True,
            "all_50_state_rows_per_candidate_reverified": True,
            "same_state_identity_across_candidate_branches": True,
            "task_and_macro_seed_aggregates_written": True,
            "undefined_denominators_preserved": True,
        },
        "source_cell_count": len(TASKS) * len(SEEDS),
        "summary_sha256": _sha256(summary_path),
        "markdown_sha256": _sha256(markdown_path),
    }
    (output_dir / "acceptance.json").write_text(
        json.dumps(acceptance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"summary": str(summary_path), "markdown": str(markdown_path), "acceptance": acceptance}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = rebuild(args.output_dir)
    except (FileNotFoundError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        raise SystemExit(f"Table 3a aggregation refused: {exc}") from exc
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
