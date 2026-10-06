#!/usr/bin/env python3
"""Verify and summarize the CVPR Table 3 closed-loop bridge."""

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

from source.common.cvpr_table3_bridge import make_state_indexed_schedules


TASKS = ("tworoom", "pusht", "reacher", "cube")
SEEDS = (42, 100, 2026, 3407, 1234, 4444)
VARIANTS = ("random64", "lewm_rerank")
INPUT_ROOT = ROOT / "outputs/cvpr/table3/v1/bridge"
OUTPUT_ROOT = INPUT_ROOT / "aggregate"
BOOTSTRAP_SEED = 20261005
BOOTSTRAP_REPLICATES = 10_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_cell(task: str, seed: int) -> dict[str, Any]:
    pair_root = INPUT_ROOT / task / f"seed_{seed}"
    summary_path = pair_root / "pair_summary.json"
    frozen_path = pair_root / "frozen_config.json"
    if not summary_path.is_file() or not frozen_path.is_file():
        raise FileNotFoundError(f"missing bridge pair artifacts: {pair_root}")
    pair = _json(summary_path)
    frozen = _json(frozen_path)
    if pair.get("status") != "pass":
        raise RuntimeError(f"{task}/seed_{seed}: bridge pair was not accepted")
    if pair.get("task") != task or int(pair.get("evaluation_seed", -1)) != seed:
        raise RuntimeError(f"{task}/seed_{seed}: pair identity differs")
    if pair.get("cohort_sha256") != frozen.get("cohort_sha256"):
        raise RuntimeError(f"{task}/seed_{seed}: cohort hash differs from frozen config")
    if pair.get("candidate_noise_mismatch_count") != 0:
        raise RuntimeError(f"{task}/seed_{seed}: candidate noise schedules do not match")
    if pair.get("first_replan_pool_mismatch_count") != 0:
        raise RuntimeError(f"{task}/seed_{seed}: first replan pools do not match")

    outcomes = {}
    traces = {}
    for variant in VARIANTS:
        outcome = pair.get("variants", {}).get(variant)
        if not isinstance(outcome, dict) or int(outcome.get("episode_count", -1)) != 50:
            raise RuntimeError(f"{task}/seed_{seed}: {variant} is missing 50 episodes")
        attempt = ROOT / outcome["attempt"]
        result_path = attempt / "result.json"
        trace_path = attempt / "episodes.jsonl"
        audit_path = attempt / "schedule_audit.jsonl"
        acceptance_path = attempt / "acceptance.json"
        for path in (result_path, trace_path, audit_path, acceptance_path):
            if not path.is_file():
                raise FileNotFoundError(f"{task}/seed_{seed}/{variant}: missing {path}")
        acceptance = _json(acceptance_path)
        if acceptance.get("status") != "pass" or acceptance.get("errors") != []:
            raise RuntimeError(f"{task}/seed_{seed}/{variant}: acceptance did not pass")
        if acceptance.get("result_sha256") != _sha256(result_path):
            raise RuntimeError(f"{task}/seed_{seed}/{variant}: result hash mismatch")
        if outcome.get("result_sha256") != _sha256(result_path):
            raise RuntimeError(f"{task}/seed_{seed}/{variant}: pair/result hashes disagree")
        if acceptance.get("trace_sha256") != _sha256(trace_path):
            raise RuntimeError(f"{task}/seed_{seed}/{variant}: trace hash mismatch")
        if outcome.get("trace_sha256") != _sha256(trace_path):
            raise RuntimeError(f"{task}/seed_{seed}/{variant}: pair/trace hashes disagree")
        if acceptance.get("schedule_audit_sha256") != _sha256(audit_path):
            raise RuntimeError(f"{task}/seed_{seed}/{variant}: schedule audit hash mismatch")
        result = _json(result_path)
        trace = _jsonl(trace_path)
        audit = _jsonl(audit_path)
        if result.get("status") != "ok" or len(result.get("episodes", [])) != 50:
            raise RuntimeError(f"{task}/seed_{seed}/{variant}: result is incomplete")
        if len(trace) != 50 or [int(row["slot"]) for row in trace] != list(range(50)):
            raise RuntimeError(f"{task}/seed_{seed}/{variant}: trace slots are incomplete")
        if any(not isinstance(row.get("success"), bool) for row in trace):
            raise RuntimeError(f"{task}/seed_{seed}/{variant}: success labels are missing")
        result_episodes = {
            int(row["slot"]): row for row in result.get("episodes", [])
        }
        if set(result_episodes) != set(range(50)):
            raise RuntimeError(f"{task}/seed_{seed}/{variant}: result episode slots are incomplete")
        for row in trace:
            result_row = result_episodes[int(row["slot"])]
            for key in ("episode_id", "row_index", "goal_row_index", "success"):
                if row.get(key) != result_row.get(key):
                    raise RuntimeError(
                        f"{task}/seed_{seed}/{variant}: result/trace {key} differs at slot {row['slot']}"
                    )
        audit_map = {
            (int(row["cohort_slot"]), int(row["replan_index"])): row
            for row in audit
        }
        if len(audit_map) != len(audit):
            raise RuntimeError(f"{task}/seed_{seed}/{variant}: duplicate schedule audit keys")
        outcomes[variant] = {
            **outcome,
            "success_rate_recomputed": float(np.mean([row["success"] for row in trace])),
        }
        if not math.isclose(
            outcomes[variant]["success_rate_recomputed"],
            float(result["success_rate"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"{task}/seed_{seed}/{variant}: reported success rate differs")
        traces[variant] = trace
        outcomes[variant]["schedule_audit"] = audit_map

    noise_schedule, selection_schedule, _metadata = make_state_indexed_schedules(
        task=task,
        evaluation_seed=seed,
        policy_seed=seed + 20_000,
    )
    random_audit = outcomes["random64"]["schedule_audit"]
    lewm_audit = outcomes["lewm_rerank"]["schedule_audit"]
    shared = sorted(set(random_audit) & set(lewm_audit))
    if not shared:
        raise RuntimeError(f"{task}/seed_{seed}: no shared replan schedule keys")
    for key in shared:
        if random_audit[key]["candidate_noise_sha256"] != lewm_audit[key]["candidate_noise_sha256"]:
            raise RuntimeError(f"{task}/seed_{seed}: paired noise differs at {key}")
        expected_index = int(
            selection_schedule((key,), num_candidates=64, device="cpu")[0]
        )
        if int(random_audit[key]["selected_index"]) != expected_index:
            raise RuntimeError(f"{task}/seed_{seed}: Random-64 selection differs at {key}")
    first = [key for key in shared if key[1] == 0]
    if {key[0] for key in first} != set(range(50)):
        raise RuntimeError(f"{task}/seed_{seed}: first replan did not cover all cohort states")
    if any(
        random_audit[key]["candidate_pool_sha256"]
        != lewm_audit[key]["candidate_pool_sha256"]
        for key in first
    ):
        raise RuntimeError(f"{task}/seed_{seed}: initial candidate pools differ")

    trace_by_variant = {
        variant: {int(row["slot"]): row for row in rows}
        for variant, rows in traces.items()
    }
    differences = []
    for slot in range(50):
        left, right = trace_by_variant["random64"][slot], trace_by_variant["lewm_rerank"][slot]
        if (left.get("episode_id"), left.get("row_index"), left.get("goal_row_index")) != (
            right.get("episode_id"), right.get("row_index"), right.get("goal_row_index")
        ):
            raise RuntimeError(f"{task}/seed_{seed}: paired episode identities differ at slot {slot}")
        differences.append(
            {
                "task": task,
                "evaluation_seed": seed,
                "slot": slot,
                "episode_id": left["episode_id"],
                "random64_success": bool(left["success"]),
                "lewm_rerank_success": bool(right["success"]),
                "lewm_minus_random": int(bool(right["success"])) - int(bool(left["success"])),
            }
        )
    return {
        "task": task,
        "evaluation_seed": seed,
        "pair_root": str(pair_root.relative_to(ROOT)),
        "source_refs": pair.get("source_refs", {}),
        "pair_summary_sha256": _sha256(summary_path),
        "frozen_config_sha256": _sha256(frozen_path),
        "shared_schedule_key_count": len(shared),
        "first_replan_pool_count": len(first),
        "variants": {
            variant: {
                key: value
                for key, value in outcome.items()
                if key != "schedule_audit"
            }
            for variant, outcome in outcomes.items()
        },
        "paired_episode_differences": differences,
    }


def _cluster_bootstrap(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    clusters: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        key = (str(row["task"]), int(row["episode_id"]))
        clusters.setdefault(key, []).append(float(row["lewm_minus_random"]))
    keys = sorted(clusters)
    if not keys:
        return {"mean": None, "ci95": None, "cluster_count": 0, "replicates": BOOTSTRAP_REPLICATES}
    mean = float(np.mean([value for key in keys for value in clusters[key]]))
    rng = np.random.default_rng(int(seed))
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(keys), size=len(keys))
        sample = [value for draw in sampled for value in clusters[keys[int(draw)]]]
        draws[index] = float(np.mean(sample))
    return {
        "mean": mean,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "cluster_count": len(keys),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": int(seed),
        "unit": "source episode, grouped within task",
    }


def _macro_cluster_bootstrap(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    clusters: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        task = str(row["task"])
        episode_id = int(row["episode_id"])
        clusters.setdefault(task, {}).setdefault(episode_id, []).append(
            float(row["lewm_minus_random"])
        )
    if set(clusters) != set(TASKS):
        raise RuntimeError("macro bridge bootstrap requires all four tasks")
    keys = {task: sorted(values) for task, values in clusters.items()}
    task_means = {
        task: float(np.mean([value for episode in episode_ids for value in clusters[task][episode]]))
        for task, episode_ids in keys.items()
    }
    rng = np.random.default_rng(int(seed))
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        task_draws = []
        for task in TASKS:
            episode_ids = keys[task]
            sampled = rng.integers(0, len(episode_ids), size=len(episode_ids))
            sample = [
                value
                for index in sampled
                for value in clusters[task][episode_ids[int(index)]]
            ]
            task_draws.append(float(np.mean(sample)))
        draws[replicate] = float(np.mean(task_draws))
    return {
        "mean": float(np.mean(list(task_means.values()))),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "cluster_count_by_task": {task: len(keys[task]) for task in TASKS},
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": int(seed),
        "unit": "source episode, resampled within task; task macro-average",
    }


def aggregate() -> dict[str, Any]:
    cells = {
        (task, seed): _read_cell(task, seed)
        for task in TASKS
        for seed in SEEDS
    }
    task_metrics = {}
    per_seed = []
    all_differences = []
    for task_index, task in enumerate(TASKS):
        rows = [
            row
            for seed in SEEDS
            for row in cells[(task, seed)]["paired_episode_differences"]
        ]
        all_differences.extend(rows)
        random_by_seed = [
            cells[(task, seed)]["variants"]["random64"]["success_rate_recomputed"]
            for seed in SEEDS
        ]
        lewm_by_seed = [
            cells[(task, seed)]["variants"]["lewm_rerank"]["success_rate_recomputed"]
            for seed in SEEDS
        ]
        task_metrics[task] = {
            "random64_success_rate_by_seed": {
                str(seed): cells[(task, seed)]["variants"]["random64"]["success_rate_recomputed"]
                for seed in SEEDS
            },
            "lewm_rerank_success_rate_by_seed": {
                str(seed): cells[(task, seed)]["variants"]["lewm_rerank"]["success_rate_recomputed"]
                for seed in SEEDS
            },
            "random64_mean_success_rate": float(np.mean(random_by_seed)),
            "random64_sample_std_across_evaluation_seeds": float(np.std(random_by_seed, ddof=1)),
            "lewm_rerank_mean_success_rate": float(np.mean(lewm_by_seed)),
            "lewm_rerank_sample_std_across_evaluation_seeds": float(np.std(lewm_by_seed, ddof=1)),
            "lewm_minus_random_cluster_bootstrap": _cluster_bootstrap(
                rows, BOOTSTRAP_SEED + task_index
            ),
            "table1_cowm_p3_reference_success_rate_by_seed": {
                str(seed): cells[(task, seed)]["source_refs"]["table1_cowm_p3"]["success_rate"]
                for seed in SEEDS
            },
        }
    for seed in SEEDS:
        random_rates = [
            cells[(task, seed)]["variants"]["random64"]["success_rate_recomputed"]
            for task in TASKS
        ]
        lewm_rates = [
            cells[(task, seed)]["variants"]["lewm_rerank"]["success_rate_recomputed"]
            for task in TASKS
        ]
        per_seed.append(
            {
                "evaluation_seed": seed,
                "random64_macro_success_rate": float(np.mean(random_rates)),
                "lewm_rerank_macro_success_rate": float(np.mean(lewm_rates)),
                "lewm_minus_random_macro_success_rate": float(np.mean(lewm_rates) - np.mean(random_rates)),
            }
        )
    macro_differences = [row["lewm_minus_random_macro_success_rate"] for row in per_seed]
    sample_std = float(np.std(macro_differences, ddof=1)) if len(macro_differences) > 1 else None
    return {
        "schema_version": 1,
        "experiment": "cvpr_table3_closed_loop_bridge_v1",
        "status": "complete",
        "task_count": len(TASKS),
        "evaluation_seed_count": len(SEEDS),
        "variant_count": len(VARIANTS),
        "completed_pair_count": len(cells),
        "episode_count_per_variant": len(cells) * 50,
        "total_episode_count": len(cells) * 50 * len(VARIANTS),
        "schedule_audit": {
            "paired_noise_mismatch_count": 0,
            "first_replan_pool_mismatch_count": 0,
            "first_replan_state_count": sum(cell["first_replan_pool_count"] for cell in cells.values()),
            "candidate_pools_claimed_equal_after_divergence": False,
        },
        "task_metrics": task_metrics,
        "macro_success_rate_by_seed": per_seed,
        "macro_success_rate": {
            "random64_mean": float(np.mean([row["random64_macro_success_rate"] for row in per_seed])),
            "random64_sample_std_across_evaluation_seeds": float(np.std([row["random64_macro_success_rate"] for row in per_seed], ddof=1)),
            "lewm_rerank_mean": float(np.mean([row["lewm_rerank_macro_success_rate"] for row in per_seed])),
            "lewm_rerank_sample_std_across_evaluation_seeds": float(np.std([row["lewm_rerank_macro_success_rate"] for row in per_seed], ddof=1)),
            "lewm_minus_random_mean": float(np.mean(macro_differences)),
            "lewm_minus_random_sample_std_across_evaluation_seeds": sample_std,
            "lewm_minus_random_cluster_bootstrap": _macro_cluster_bootstrap(
                all_differences, BOOTSTRAP_SEED + 100
            ),
        },
        "source_cells": {
            f"{task}/seed_{seed}": {
                "pair_summary_sha256": cells[(task, seed)]["pair_summary_sha256"],
                "frozen_config_sha256": cells[(task, seed)]["frozen_config_sha256"],
            }
            for task in TASKS
            for seed in SEEDS
        },
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# CVPR Table 3 Closed-Loop Bridge",
        "",
        "The bridge evaluates Random-64 and same-actor LeWM reranking on the frozen Table 1 50-state cohorts. Both methods use the same CoWM action generator, 64 candidates, Euler S=2, and a 25-step execution budget.",
        "",
        f"Acceptance: {summary['completed_pair_count']}/24 paired task-seed cells, {summary['episode_count_per_variant']} episodes per method ({summary['total_episode_count']} total).",
        "",
        f"For each cohort slot and replan count, a frozen random schedule supplies the same initial action noise to both methods; Random-64 also uses a state-indexed uniform candidate index. The two first-replan candidate pools matched for all {summary['schedule_audit']['first_replan_state_count']} states. Later trajectories can diverge, so only the noise schedule is paired after the first executed plan; later candidate pools are not claimed identical.",
        "",
        "## Success rate by task",
        "",
        "| Task | Random-64 SR (mean ± SD across 6 seeds) | LeWM rerank SR (mean ± SD across 6 seeds) | LeWM − Random, source-episode bootstrap 95% CI | Table 1 CoWM-P3 reference mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        row = summary["task_metrics"][task]
        paired = row["lewm_minus_random_cluster_bootstrap"]
        p3 = list(row["table1_cowm_p3_reference_success_rate_by_seed"].values())
        p3_mean = float(np.mean(p3))
        lines.append(
            f"| {task} | {row['random64_mean_success_rate']:.3f} ± {row['random64_sample_std_across_evaluation_seeds']:.3f} | {row['lewm_rerank_mean_success_rate']:.3f} ± {row['lewm_rerank_sample_std_across_evaluation_seeds']:.3f} | {paired['mean']:+.3f} [{paired['ci95'][0]:+.3f}, {paired['ci95'][1]:+.3f}] | {p3_mean:.3f} |"
        )
    macro = summary["macro_success_rate"]
    paired = macro["lewm_minus_random_cluster_bootstrap"]
    lines.extend(
        [
            "",
            "## Macro summary",
            "",
            f"Across the four tasks, seed-macro SR is {macro['random64_mean']:.3f} ± {macro['random64_sample_std_across_evaluation_seeds']:.3f} for Random-64 and {macro['lewm_rerank_mean']:.3f} ± {macro['lewm_rerank_sample_std_across_evaluation_seeds']:.3f} for LeWM reranking (sample SD across evaluation seeds). The paired macro difference is {macro['lewm_minus_random_mean']:+.3f}; its source-episode cluster-bootstrap 95% CI is [{paired['ci95'][0]:+.3f}, {paired['ci95'][1]:+.3f}].",
            "",
            "Table 1 CoWM-P3 values are included as a contextual reference from their accepted source cells. They were not rerun under this bridge's state-indexed random schedule, so they are not treated as a matched closed-loop arm.",
            "",
            "Detailed per-seed results, source hashes, replan schedule checks, and bootstrap settings are in `outputs/cvpr/table3/v1/bridge/aggregate/summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()
    summary = aggregate()
    output_root = Path(args.output_root).resolve()
    _write_json(output_root / "summary.json", summary)
    (output_root / "cvpr_table3_bridge_report.md").write_text(
        _render_report(summary), encoding="utf-8"
    )
    print(f"accepted {summary['completed_pair_count']}/24 bridge pairs; wrote {output_root}")


if __name__ == "__main__":
    main()
