#!/usr/bin/env python3
"""Strictly verify and aggregate CVPR Table 3b matched-action refinements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASKS = ("tworoom", "pusht", "reacher", "cube")
SEEDS = (42, 100, 2026, 3407, 1234, 4444)
REFINEMENT_ROOT = ROOT / "outputs/cvpr/table3/v1/3b"
POOL_ROOT = ROOT / "outputs/cvpr/table3/v1/3a"
DEFAULT_OUTPUT = REFINEMENT_ROOT / "aggregate"
BOOTSTRAP_SEED = 20261005
BOOTSTRAP_REPLICATES = 10_000
GUIDED = ("cowm_po", "lewm_po", "cowm_gf")
VARIANTS = set(GUIDED) | {
    f"{method}_random_{index}" for method in GUIDED for index in range(5)
}
REQUIRED_ACCEPTANCE_CHECKS = (
    "source_3a_pool_passed",
    "candidate_0_reproduced_from_frozen_noise",
    "coWM_candidate0_cost_recomputed",
    "leWM_candidate0_cost_recomputed",
    "candidate0_branch_matches_3a",
    "all_three_corrections_replayed",
    "all_15_random_controls_replayed",
    "each_control_has_50_predeclared_direction_records",
    "unmatchable_controls_preserved",
    "action_archive_sha256_recorded",
    "extra_candidate_projection_disabled",
    "branch_trace_hashes_recorded",
    "paired_bootstrap_10000",
)
PRIMARY_METHODS = ("cowm_po", "lewm_po", "cowm_gf")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def _compare_baseline(reference: list[dict[str, Any]], replayed: list[dict[str, Any]]) -> dict[str, Any]:
    left = {int(row["slot"]): row for row in reference}
    right = {int(row["slot"]): row for row in replayed}
    if left.keys() != right.keys() or len(left) != 50:
        raise RuntimeError("candidate-0 replay does not contain the exact 50-state cohort")
    max_abs = 0.0
    comparisons = 0
    for slot in range(50):
        a, b = left[slot], right[slot]
        for key in (
            "native_success_after_budget",
            "success_by_5",
            "success_by_25",
            "effective_valid_length",
        ):
            if a.get(key) != b.get(key):
                raise RuntimeError(f"candidate-0 repeat differs for {key}, slot={slot}")
        if len(a.get("steps", [])) != len(b.get("steps", [])):
            raise RuntimeError(f"candidate-0 repeat length differs at slot {slot}")
        for step_a, step_b in zip(a.get("steps", []), b.get("steps", [])):
            for key in ("terminated", "truncated", "env_success", "predicate_success"):
                if step_a.get(key) != step_b.get(key):
                    raise RuntimeError(f"candidate-0 repeat differs for {key}, slot={slot}")
            for key in ("current", "goal", "action"):
                va, vb = step_a.get(key), step_b.get(key)
                if va is None or vb is None:
                    if va != vb:
                        raise RuntimeError(f"candidate-0 repeat differs for {key}, slot={slot}")
                    continue
                xa, xb = np.asarray(va, dtype=np.float64), np.asarray(vb, dtype=np.float64)
                if xa.shape != xb.shape:
                    raise RuntimeError(f"candidate-0 repeat {key} shape differs")
                error = float(np.max(np.abs(xa - xb))) if xa.size else 0.0
                max_abs = max(max_abs, error)
                comparisons += 1
                if not np.allclose(xa, xb, atol=1e-6, rtol=1e-6):
                    raise RuntimeError(f"candidate-0 repeat diverged for {key}, slot={slot}")
    return {
        "status": "pass",
        "checked_states": 50,
        "state_array_comparisons": comparisons,
        "max_absolute_difference": max_abs,
    }


def _stable_seed(*parts: Any) -> int:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _verify_cell(task: str, seed: int, input_root: Path) -> dict[str, Any]:
    cell = input_root / task / f"seed_{seed}"
    acceptance_path = cell / "acceptance.json"
    summary_path = cell / "summary.json"
    frozen_path = cell / "frozen_config.json"
    action_path = cell / "actions.npz"
    baseline_path = cell / "baseline_replay_check.jsonl"
    for path in (acceptance_path, summary_path, frozen_path, action_path, baseline_path):
        if not path.is_file():
            raise FileNotFoundError(f"{task}/seed_{seed}: missing {path.name}")

    acceptance = _read_json(acceptance_path)
    summary = _read_json(summary_path)
    frozen = _read_json(frozen_path)
    if acceptance.get("status") != "pass" or acceptance.get("errors") != []:
        raise RuntimeError(f"{task}/seed_{seed}: refinement acceptance is not pass")
    if acceptance.get("summary_sha256") != _sha256(summary_path):
        raise RuntimeError(f"{task}/seed_{seed}: summary hash differs from acceptance")
    if acceptance.get("frozen_config_sha256") != _sha256(frozen_path):
        raise RuntimeError(f"{task}/seed_{seed}: frozen config hash differs from acceptance")
    if summary.get("task") != task or int(summary.get("evaluation_seed", -1)) != seed:
        raise RuntimeError(f"{task}/seed_{seed}: summary identity mismatch")
    checks = acceptance.get("checks", {})
    failed = [name for name in REQUIRED_ACCEPTANCE_CHECKS if checks.get(name) is not True]
    if failed:
        raise RuntimeError(f"{task}/seed_{seed}: failed acceptance checks {failed}")
    if int(summary.get("branch_replay_count", -1)) != 18:
        raise RuntimeError(f"{task}/seed_{seed}: expected 18 replayed variants")
    if set(summary.get("branch_file_sha256", {})) != VARIANTS:
        raise RuntimeError(f"{task}/seed_{seed}: variant hash set differs from the frozen protocol")
    if _sha256(action_path) != summary.get("action_archive_sha256"):
        raise RuntimeError(f"{task}/seed_{seed}: action archive hash mismatch")
    if acceptance.get("action_archive_sha256") != summary.get("action_archive_sha256"):
        raise RuntimeError(f"{task}/seed_{seed}: acceptance/action archive hashes disagree")

    source_cell = POOL_ROOT / task / f"seed_{seed}"
    source_acceptance_path = source_cell / "acceptance.json"
    source_summary_path = source_cell / "summary.json"
    source_branch = source_cell / "branches" / "candidate_0000.jsonl"
    if not all(path.is_file() for path in (source_acceptance_path, source_summary_path, source_branch)):
        raise FileNotFoundError(f"{task}/seed_{seed}: accepted 3a source artifacts are missing")
    source_acceptance = _read_json(source_acceptance_path)
    source_summary = _read_json(source_summary_path)
    if source_acceptance.get("status") != "pass":
        raise RuntimeError(f"{task}/seed_{seed}: source 3a cell is not accepted")
    if frozen.get("source_3a_acceptance_sha256") != _sha256(source_acceptance_path):
        raise RuntimeError(f"{task}/seed_{seed}: refinement points to a different 3a acceptance")
    if frozen.get("source_fixed_pool_sha256") != source_summary.get("fixed_pool_sha256"):
        raise RuntimeError(f"{task}/seed_{seed}: refinement points to a different fixed pool")
    source_branch_hash = source_summary.get("branch_file_sha256", {}).get("candidate_0000.jsonl")
    if source_branch_hash != _sha256(source_branch):
        raise RuntimeError(f"{task}/seed_{seed}: source candidate-0 trace hash mismatch")

    baseline = _read_jsonl(baseline_path)
    reference = _read_jsonl(source_branch)
    restore = _compare_baseline(reference, baseline)
    reported_restore = summary.get("restore_check", {})
    if reported_restore.get("status") != "pass":
        raise RuntimeError(f"{task}/seed_{seed}: candidate-0 restore check did not pass")
    if restore.get("status") != "pass" or restore.get("checked_states") != 50:
        raise RuntimeError(f"{task}/seed_{seed}: independent candidate-0 replay check failed")
    regeneration_protocol = frozen.get("protocol", {}).get("candidate_regeneration")
    if regeneration_protocol is not None:
        regeneration = summary.get("candidate_regeneration_check", {})
        if (
            regeneration.get("status") != "pass"
            or regeneration.get("candidate_batch_shape") != [50, 64]
            or int(regeneration.get("verified_candidate_count", -1)) != 50 * 64
            or float(regeneration.get("max_absolute_difference", float("inf")))
            > float(regeneration_protocol.get("atol", 0.0))
        ):
            raise RuntimeError(
                f"{task}/seed_{seed}: full 50x64 candidate-noise replay check failed"
            )

    branch_rows: dict[str, list[dict[str, Any]]] = {}
    for variant in sorted(VARIANTS):
        path = cell / "branches" / f"{variant}.jsonl"
        if not path.is_file() or _sha256(path) != summary["branch_file_sha256"][variant]:
            raise RuntimeError(f"{task}/seed_{seed}: {variant} trace is missing or has a bad hash")
        rows = _read_jsonl(path)
        if len(rows) != 50 or [int(row["slot"]) for row in rows] != list(range(50)):
            raise RuntimeError(f"{task}/seed_{seed}: {variant} does not have ordered slots 0..49")
        if any(row.get("variant") != variant for row in rows):
            raise RuntimeError(f"{task}/seed_{seed}: {variant} trace contains another variant")
        if any(row.get("outcome_status") != "completed" for row in rows):
            raise RuntimeError(f"{task}/seed_{seed}: {variant} has incomplete outcome rows")
        if variant.endswith(tuple(f"_random_{index}" for index in range(5))):
            for row in rows:
                match = row.get("random_match")
                if not isinstance(match, dict) or match.get("status") not in {"matched", "unmatchable"}:
                    raise RuntimeError(f"{task}/seed_{seed}: {variant} lost its direction-match record")
        branch_rows[variant] = rows

    metrics = summary.get("metrics", {})
    if set(GUIDED) - set(metrics):
        raise RuntimeError(f"{task}/seed_{seed}: guided method metrics are incomplete")
    for method in GUIDED:
        comparison = metrics.get(f"{method}_versus_equal_rms_random")
        if not isinstance(comparison, dict) or int(comparison.get("control_direction_count", -1)) != 5:
            raise RuntimeError(f"{task}/seed_{seed}: {method} random-control metrics are missing")

    return {
        "root": cell,
        "summary": summary,
        "frozen_config": frozen,
        "baseline": baseline,
        "branches": branch_rows,
        "source_summary": source_summary,
        "independent_restore_check": restore,
    }


def _sample_summary(values: list[Any]) -> dict[str, Any]:
    clean = [float(value) for value in values if _finite(value)]
    return {
        "mean": float(np.mean(clean)) if clean else None,
        "sample_std": float(np.std(clean, ddof=1)) if len(clean) > 1 else None,
        "n_evaluation_seeds": len(clean),
    }


def _clustered_seed_bootstrap(
    by_seed: dict[int, list[tuple[Any, float]]],
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    episodes: dict[str, dict[int, list[float]]] = {}
    for evaluation_seed, observations in by_seed.items():
        for episode_id, value in observations:
            if _finite(value):
                key = json.dumps(episode_id, ensure_ascii=False, sort_keys=True)
                episodes.setdefault(key, {}).setdefault(evaluation_seed, []).append(float(value))
    if not episodes:
        return {
            "mean": None,
            "ci95": None,
            "bootstrap_p_two_sided": None,
            "evaluation_seed_count": 0,
            "source_episode_cluster_count": 0,
            "state_appearance_count": 0,
            "replicates": replicates,
        }

    episode_keys = sorted(episodes)
    evaluation_seeds = sorted(
        {
            evaluation_seed
            for by_eval_seed in episodes.values()
            for evaluation_seed in by_eval_seed
        }
    )
    per_seed_means = [
        float(
            np.mean(
                [
                    value
                    for episode in episodes.values()
                    for value in episode.get(evaluation_seed, [])
                ]
            )
        )
        for evaluation_seed in evaluation_seeds
    ]
    point = float(np.mean(per_seed_means))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(episode_keys), size=(replicates, len(episode_keys)))
    multiplicity = np.zeros((replicates, len(episode_keys)), dtype=np.float64)
    np.add.at(
        multiplicity,
        (np.repeat(np.arange(replicates), len(episode_keys)), picks.reshape(-1)),
        1.0,
    )
    draws_by_seed = []
    for evaluation_seed in evaluation_seeds:
        cluster_sums = np.asarray(
            [
                sum(episodes[key].get(evaluation_seed, []))
                for key in episode_keys
            ],
            dtype=np.float64,
        )
        cluster_counts = np.asarray(
            [
                len(episodes[key].get(evaluation_seed, []))
                for key in episode_keys
            ],
            dtype=np.float64,
        )
        sampled_sum = multiplicity @ cluster_sums
        sampled_count = multiplicity @ cluster_counts
        draws_by_seed.append(
            np.divide(
                sampled_sum,
                sampled_count,
                out=np.full(replicates, np.nan, dtype=np.float64),
                where=sampled_count > 0,
            )
        )
    seed_draws = np.stack(draws_by_seed, axis=0)
    draws = np.nanmean(seed_draws, axis=0)
    state_count = sum(
        len(values)
        for by_eval_seed in episodes.values()
        for values in by_eval_seed.values()
    )
    centered = draws - point
    tail = abs(point)
    lower_tail = (float(np.count_nonzero(centered <= -tail)) + 1.0) / (replicates + 1.0)
    upper_tail = (float(np.count_nonzero(centered >= tail)) + 1.0) / (replicates + 1.0)
    p_value = min(
        1.0,
        2.0 * min(lower_tail, upper_tail),
    )
    return {
        "mean": point,
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "bootstrap_p_two_sided": p_value,
        "evaluation_seed_count": len(evaluation_seeds),
        "source_episode_cluster_count": len(episode_keys),
        "state_appearance_count": state_count,
        "replicates": replicates,
        "aggregation": "globally resample source episode clusters within task; preserve cross-seed episode repeats and equally average estimable seed means",
    }


def _selector_differences(
    cells: dict[str, dict[int, dict[str, Any]]], task: str, metric: str
) -> dict[int, list[tuple[Any, float]]]:
    result: dict[int, list[tuple[Any, float]]] = {}
    for seed in SEEDS:
        summary = cells[task][seed]["source_summary"]
        left = summary["metrics"]["lewm"]["by_state"]
        right = summary["metrics"]["cowm_b"]["by_state"]
        observations = []
        for a, b in zip(left, right):
            if a.get("episode_id") != b.get("episode_id"):
                raise RuntimeError(f"{task}/seed_{seed}: selector rows are not paired")
            x, y = a.get(metric), b.get(metric)
            if _finite(x) and _finite(y):
                observations.append((a["episode_id"], float(x) - float(y)))
        result[seed] = observations
    return result


def _po_random_differences(
    cells: dict[str, dict[int, dict[str, Any]]], task: str
) -> dict[int, list[tuple[Any, float]]]:
    result: dict[int, list[tuple[Any, float]]] = {}
    for seed in SEEDS:
        cell = cells[task][seed]
        baseline = cell["baseline"]
        guide = cell["branches"]["cowm_po"]
        controls = [
            cell["branches"][f"cowm_po_random_{index}"] for index in range(5)
        ]
        observations = []
        for slot, (base_row, guide_row) in enumerate(zip(baseline, guide)):
            base_cost = base_row.get("physical_cost_at_25")
            guide_cost = guide_row.get("physical_cost_at_25")
            if (
                base_row.get("valid_at_25") is not True
                or guide_row.get("valid_at_25") is not True
                or not _finite(base_cost)
                or not _finite(guide_cost)
            ):
                continue
            random_improvements = []
            for rows in controls:
                row = rows[slot]
                match = row.get("random_match", {})
                cost = row.get("physical_cost_at_25")
                if (
                    match.get("status") == "matched"
                    and row.get("valid_at_25") is True
                    and _finite(cost)
                ):
                    random_improvements.append(float(base_cost) - float(cost))
            if random_improvements:
                guide_improvement = float(base_cost) - float(guide_cost)
                observations.append(
                    (
                        base_row["episode_id"],
                        guide_improvement - float(np.mean(random_improvements)),
                    )
                )
        result[seed] = observations
    return result


def _holm_adjust(hypotheses: list[dict[str, Any]]) -> None:
    eligible = [row for row in hypotheses if _finite(row.get("bootstrap_p_two_sided"))]
    eligible.sort(key=lambda row: float(row["bootstrap_p_two_sided"]))
    count = len(eligible)
    running = 0.0
    for index, row in enumerate(eligible):
        adjusted = min(1.0, (count - index) * float(row["bootstrap_p_two_sided"]))
        running = max(running, adjusted)
        row["holm_adjusted_p"] = running
    for row in hypotheses:
        row.setdefault("holm_adjusted_p", None)
        row["estimable_holm_family_size"] = count
        row["planned_holm_family_size"] = 12


def _aggregate(cells: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    descriptive: dict[str, Any] = {}
    for task in TASKS:
        descriptive[task] = {}
        for method in GUIDED:
            cell_values: dict[str, list[Any]] = {
                "true_cost_improvement": [],
                "improved_state_fraction": [],
                "misleading_prediction_fraction": [],
                "success_rate_change": [],
                "valid_25_state_count": [],
            }
            for seed in SEEDS:
                metrics = cells[task][seed]["summary"]["metrics"][method]
                improvement = metrics.get("true_cost_improvement_original_minus_updated", {})
                success = metrics.get("success_rate_change", {})
                cell_values["true_cost_improvement"].append(improvement.get("mean"))
                cell_values["improved_state_fraction"].append(metrics.get("improved_state_fraction"))
                cell_values["misleading_prediction_fraction"].append(
                    metrics.get("misleading_predicted_improvement_worsened_true_cost_fraction")
                )
                cell_values["success_rate_change"].append(success.get("mean") if isinstance(success, dict) else None)
                cell_values["valid_25_state_count"].append(metrics.get("valid_25_state_count"))
            descriptive[task][method] = {
                key: _sample_summary(values)
                for key, values in cell_values.items()
                if key != "valid_25_state_count"
            }
            counts = [int(value) for value in cell_values["valid_25_state_count"] if _finite(value)]
            descriptive[task][method]["valid_25_state_count_range"] = (
                [min(counts), max(counts)] if counts else None
            )
            descriptive[task][method]["by_evaluation_seed"] = {
                str(seed): cells[task][seed]["summary"]["metrics"][method]
                for seed in SEEDS
            }

        for method in GUIDED:
            comparison_values = []
            for seed in SEEDS:
                comparison = cells[task][seed]["summary"]["metrics"][
                    f"{method}_versus_equal_rms_random"
                ]["mean_improvement_difference_method_minus_random"]
                comparison_values.append(comparison.get("mean"))
            descriptive[task][f"{method}_versus_equal_rms_random"] = _sample_summary(
                comparison_values
            )

    hypotheses: list[dict[str, Any]] = []
    for task in TASKS:
        for key, label, metric in (
            (
                "lewm_minus_cowm_b_selected_success",
                "LeWM − CoWM-B selected success by 25",
                "top1_success_25",
            ),
            (
                "lewm_minus_cowm_b_complete64_regret",
                "LeWM − CoWM-B complete-pool physical regret",
                "complete_64_selected_physical_regret_at_25",
            ),
        ):
            values = _selector_differences(cells, task, metric)
            estimate = _clustered_seed_bootstrap(
                values, seed=BOOTSTRAP_SEED + _stable_seed(task, key) % 100_000
            )
            hypotheses.append(
                {
                    "id": f"{task}:{key}",
                    "task": task,
                    "comparison": label,
                    "difference_direction": "LeWM minus CoWM-B",
                    **estimate,
                }
            )
        values = _po_random_differences(cells, task)
        estimate = _clustered_seed_bootstrap(
            values, seed=BOOTSTRAP_SEED + _stable_seed(task, "cowm_po_random") % 100_000
        )
        hypotheses.append(
            {
                "id": f"{task}:cowm_po_vs_equal_rms_random",
                "task": task,
                "comparison": "CoWM-PO − equal-RMS random true improvement",
                "difference_direction": "CoWM-PO minus mean of matched random controls",
                **estimate,
            }
        )
    _holm_adjust(hypotheses)
    return {
        "by_task_and_method": descriptive,
        "primary_hypotheses": hypotheses,
        "bootstrap": {
            "unit": "source episode, shared across evaluation seeds within task",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed_base": BOOTSTRAP_SEED,
            "seed_aggregation": "equal mean across estimable evaluation seeds; repeated source episodes remain in one cluster",
            "holm_planned_family_size": 12,
        },
    }


def _fmt(summary: dict[str, Any]) -> str:
    mean = summary.get("mean")
    sd = summary.get("sample_std")
    n = summary.get("n_evaluation_seeds", 0)
    if mean is None:
        return f"NA (n={n})"
    if sd is None:
        return f"{mean:.4f} (n={n})"
    return f"{mean:.4f} ± {sd:.4f} (n={n})"


def _markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        "# CVPR Table 3b matched-action refinement summary",
        "",
        "Each task/method entry gives the mean and sample standard deviation of six Table 1 evaluation-seed estimates. Cost improvement is physical cost before minus after, so positive favors the correction. Denominators include only paired states with valid step-25 outcomes; equal-RMS controls additionally require a matched direction. The primary family uses episode-cluster bootstrap within evaluation seed and Holm adjustment across estimable tests.",
        "",
    ]
    for task in TASKS:
        lines.extend(
            [
                f"## {task.title()}",
                "",
                "| Method | True cost improvement | Improved-state fraction | Misleading-prediction fraction | Valid step-25 states per seed |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method in GUIDED:
            row = aggregate["by_task_and_method"][task][method]
            count_range = row.get("valid_25_state_count_range")
            count_text = "NA" if count_range is None else f"{count_range[0]}–{count_range[1]}"
            lines.append(
                f"| {method} | {_fmt(row['true_cost_improvement'])} | "
                f"{_fmt(row['improved_state_fraction'])} | "
                f"{_fmt(row['misleading_prediction_fraction'])} | {count_text} |"
            )
        lines.extend(
            [
                "",
                "| Correction | Improvement difference vs equal-RMS random |",
                "|---|---:|",
            ]
        )
        for method in GUIDED:
            comparison_key = f"{method}_versus_equal_rms_random"
            lines.append(
                f"| {method} | {_fmt(aggregate['by_task_and_method'][task][comparison_key])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Primary paired comparisons",
            "",
            "| Task | Comparison | Mean difference | Episode-cluster 95% CI | Bootstrap p | Holm p | Seeds / clusters / state appearances |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate["primary_hypotheses"]:
        ci = row.get("ci95")
        ci_text = "NA" if ci is None else f"[{ci[0]:.4g}, {ci[1]:.4g}]"
        p = row.get("bootstrap_p_two_sided")
        p_holm = row.get("holm_adjusted_p")
        p_text = "NA" if p is None else f"{p:.4g}"
        p_holm_text = "NA" if p_holm is None else f"{p_holm:.4g}"
        mean_text = "NA" if row["mean"] is None else f"{row['mean']:.4g}"
        n_text = (
            f"{row['evaluation_seed_count']} / {row['source_episode_cluster_count']} / "
            f"{row['state_appearance_count']}"
        )
        lines.append(
            f"| {row['task']} | {row['comparison']} | "
            f"{mean_text} | "
            f"{ci_text} | {p_text} | {p_holm_text} | {n_text} |"
        )
    estimable = sum(
        row.get("bootstrap_p_two_sided") is not None
        for row in aggregate["primary_hypotheses"]
    )
    lines.extend(
        [
            "",
            f"The family contains 12 planned hypotheses; {estimable} were estimable with nonempty planned denominators. Undefined physical-regret or refinement comparisons remain NA and are not imputed.",
            "",
        ]
    )
    return "\n".join(lines)


def rebuild(
    input_root: Path = REFINEMENT_ROOT, output_dir: Path = DEFAULT_OUTPUT
) -> dict[str, Any]:
    cells: dict[str, dict[int, dict[str, Any]]] = {}
    for task in TASKS:
        cells[task] = {}
        for seed in SEEDS:
            cells[task][seed] = _verify_cell(task, seed, input_root)
    aggregate = _aggregate(cells)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "table3b_refinement.md"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "cvpr_table3_refinement_aggregate_v1",
                "accepted_cell_count": len(TASKS) * len(SEEDS),
                "input_cells": {
                    task: {
                        str(seed): {
                            "summary_sha256": _sha256(
                                cells[task][seed]["root"] / "summary.json"
                            ),
                            "acceptance_sha256": _sha256(
                                cells[task][seed]["root"] / "acceptance.json"
                            ),
                        }
                        for seed in SEEDS
                    }
                    for task in TASKS
                },
                **aggregate,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(aggregate), encoding="utf-8")
    acceptance = {
        "status": "pass",
        "errors": [],
        "checks": {
            "all_24_cells_verified": True,
            "all_18_variants_per_cell_hash_checked": True,
            "candidate0_replay_independently_compared": True,
            "primary_family_holm_adjusted": True,
            "undefined_metrics_preserved_as_na": True,
        },
        "summary_sha256": _sha256(summary_path),
        "markdown_sha256": _sha256(markdown_path),
    }
    (output_dir / "acceptance.json").write_text(
        json.dumps(acceptance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "accepted_cell_count": len(TASKS) * len(SEEDS),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
        "acceptance": acceptance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=REFINEMENT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = rebuild(args.input_root.resolve(), args.output_dir.resolve())
    except Exception as exc:
        raise SystemExit(f"Table 3b aggregation refused: {exc}") from exc
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
