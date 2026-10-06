#!/usr/bin/env python3
"""Cohort-paired Phase 6.3 success and workload analysis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cvpr_table1 as table
from scripts.round5_phase6_3 import OUT, METHODS
from scripts.round5_phase6_3_combos import (
    COMBO_VERSIONS,
    entry_for,
    load_final_selection,
    load_selection,
)
from source.common.cvpr_table1 import TASKS, SEEDS, evaluation_cells


def paired_bootstrap(seed_rows, *, replicates=20_000, seed=63063):
    """Seed-stratified episode-pair bootstrap for a four-task equal mean."""
    import numpy as np
    rng = np.random.default_rng(seed)
    by_task = []
    selected_tasks = [task for task in TASKS if task in seed_rows]
    for task in selected_tasks:
        per_seed = [np.asarray(seed_rows[task][s], dtype=np.float64) for s in SEEDS]
        draws = np.zeros((replicates, len(SEEDS)), dtype=np.float64)
        for j, values in enumerate(per_seed):
            choices = rng.integers(0, len(values), size=(replicates, len(values)))
            draws[:, j] = values[choices].mean(axis=1)
        by_task.append(draws.mean(axis=1))
    distribution = np.stack(by_task, axis=1).mean(axis=1)
    return [float(x) for x in np.quantile(distribution, [0.025, 0.975])]


def load_cell(version, method, task, seed, *, components=None, selection_sha256=None,
              final_selection_sha256=None):
    cell = next(c for c in evaluation_cells() if c["family"] == "cowm" and c["id"] == method
                and c["task"] == task and c["evaluation_seed"] == seed
                and c["method_id"].endswith("__main"))
    parent = OUT / "closed_loop" / version / cell["cell_id"]
    candidates = sorted(parent.glob("attempt_*/result.json"))
    for result_path in reversed(candidates):
        value = table._read_json(result_path)
        phase = value.get("phase6_3", {})
        combo_matches = (version not in COMBO_VERSIONS or
                          (phase.get("components") == components and
                          phase.get("combination_selection_sha256") == selection_sha256 and
                          phase.get("final_combination_gate_sha256") == final_selection_sha256))
        if (value.get("status") == "ok" and len(value.get("episodes", [])) == 50
                and phase.get("identity", {}).get("cell") == cell and combo_matches):
            trace = result_path.parent / "episodes.jsonl"
            if trace.is_file() and value.get("cvpr_table1", {}).get("trace_sha256") == table._sha256(trace):
                return value, result_path
    return None, None


def summarize():
    import numpy as np
    runner_path = ROOT / "scripts/round5_phase6_3_closed_loop.py"
    analysis_path = Path(__file__)
    available = ["E0", "E1", "E2", "E3", "E4", "C-FP32", "C-BF16"]
    selection = selection_sha256 = final_selection = final_selection_sha256 = None
    if (OUT / "summary/selected_combinations.json").is_file():
        selection, selection_sha256 = load_selection()
    if (OUT / "summary/final_combinations.json").is_file():
        final_selection, final_selection_sha256 = load_final_selection()
    methods = [c["id"] for c in evaluation_cells() if c["family"] == "cowm"
               and c["id"] in METHODS and c["method_id"].endswith("__main")]
    methods = list(dict.fromkeys(methods))
    if set(methods) != set(METHODS):
        raise RuntimeError(f"closed-loop registry mismatch: {methods}")
    rows, paired = [], {}
    for version in available:
        for method in METHODS:
            for task in TASKS:
                if version == "E1" and method in {"P0", "P3"}:
                    rows.append({"version": version, "method": method, "task": task,
                        "status": "not_applicable", "alias_of": {"version": "E0", "method": method,
                        "task": task}, "reason": "refinement synchronization direction does not apply"})
                    continue
                components = None
                if version in COMBO_VERSIONS:
                    if selection is None:
                        rows.append({"version": version, "method": method, "task": task,
                                     "status": "pending_selection"})
                        continue
                    choice = entry_for(selection, version, method)
                    if choice["status"] == "alias":
                        rows.append({"version": version, "method": method, "task": task,
                                     "status": "alias", "alias_of": choice["alias_of"],
                                     "components": choice["components"]})
                        continue
                    if final_selection is None:
                        rows.append({"version": version, "method": method, "task": task,
                                     "status": "pending_combo_timing",
                                     "components": choice["components"]})
                        continue
                    final_choice = final_selection["versions"][version][method]
                    if final_choice["status"] == "alias":
                        rows.append({"version": version, "method": method, "task": task,
                                     "status": "alias", "alias_of": final_choice["alias_of"],
                                     "components": choice["components"]})
                        continue
                    if final_choice["status"] == "not_effective":
                        rows.append({"version": version, "method": method, "task": task,
                                     "status": "not_effective", "components": choice["components"],
                                     "latency_gate": final_choice})
                        continue
                    components = list(choice["components"])
                baseline_seed, variant_seed, transitions = {}, {}, {"success_to_failure": 0,
                    "failure_to_success": 0, "unchanged_success": 0, "unchanged_failure": 0}
                per_seed = []
                evidence = []
                for seed in SEEDS:
                    base, bp = load_cell("E0", method, task, seed)
                    variant, vp = (load_cell(version, method, task, seed,
                        components=components, selection_sha256=selection_sha256,
                        final_selection_sha256=final_selection_sha256)
                        if version != "E0" else (base, bp))
                    if base is None or variant is None:
                        per_seed.append({"seed": seed, "status": "missing"})
                        continue
                    left = {(int(e["dataset_episode"]), int(e["start_step"])): e for e in base["episodes"]}
                    right = {(int(e["dataset_episode"]), int(e["start_step"])): e for e in variant["episodes"]}
                    if left.keys() != right.keys() or len(left) != 50:
                        raise RuntimeError(f"paired cohort mismatch: {version}/{method}/{task}/{seed}")
                    pair_delta = []
                    for key in sorted(left):
                        x, y = bool(left[key]["success"]), bool(right[key]["success"])
                        pair_delta.append(float(y)-float(x))
                        if x and not y:
                            transitions["success_to_failure"] += 1
                        elif y and not x:
                            transitions["failure_to_success"] += 1
                        elif x:
                            transitions["unchanged_success"] += 1
                        else:
                            transitions["unchanged_failure"] += 1
                    baseline_seed[seed] = [float(e["success"]) for e in left.values()]
                    variant_seed[seed] = [float(e["success"]) for e in right.values()]
                    per_seed.append({"seed": seed, "status": "complete", "episodes": 50,
                        "baseline_success_rate": float(np.mean(baseline_seed[seed])),
                        "variant_success_rate": float(np.mean(variant_seed[seed])),
                        "paired_success_difference": float(np.mean(pair_delta)),
                        "episode_count_total_steps": int(sum(e.get("steps_executed", len(e.get("steps", []))) for e in right.values())),
                        "episode_replans_total": int(sum(e.get("episode_replan_count", 0) for e in right.values())),
                        "episode_planning_exposure_total_seconds": float(sum(e.get("episode_planning_wall_seconds_exposed", 0) for e in right.values())),
                        "batch_planning_wall_seconds": float(variant.get("batch_planning_wall_seconds", 0)),
                        "batch_replan_count": int(variant.get("batch_replan_count", 0))})
                    evidence.extend([str(bp.relative_to(ROOT)), str(vp.relative_to(ROOT))])
                complete = len(variant_seed) == len(SEEDS)
                if complete:
                    deltas = {s: [y-x for x, y in zip(baseline_seed[s], variant_seed[s])]
                              for s in SEEDS}
                    raw_delta = [d for s in SEEDS for d in deltas[s]]
                    row = {"version": version, "method": method, "task": task,
                        "status": "complete", "episode_count": 1200,
                        "baseline_seed_mean": float(np.mean([np.mean(baseline_seed[s]) for s in SEEDS])),
                        "baseline_seed_sample_sd": float(np.std([np.mean(baseline_seed[s]) for s in SEEDS], ddof=1)),
                        "variant_seed_mean": float(np.mean([np.mean(variant_seed[s]) for s in SEEDS])),
                        "variant_seed_sample_sd": float(np.std([np.mean(variant_seed[s]) for s in SEEDS], ddof=1)),
                        "paired_difference": float(np.mean(raw_delta)), "paired_bootstrap_95_ci": paired_bootstrap({task: deltas}),
                        "paired_transitions": transitions, "per_seed": per_seed,
                        "episode_computation": per_seed, "results": list(dict.fromkeys(evidence))}
                else:
                    row = {"version": version, "method": method, "task": task, "status": "incomplete",
                           "completed_seeds": len(variant_seed), "target_episodes": 1200,
                           "per_seed": per_seed, "paired_transitions_so_far": transitions,
                           "results": list(dict.fromkeys(evidence))}
                rows.append(row)
                if complete:
                    paired.setdefault(version, {}).setdefault(method, {})[task] = {
                        s: [y-x for x, y in zip(baseline_seed[s], variant_seed[s])] for s in SEEDS}
    task_means = {}
    for version, methods_data in paired.items():
        task_means[version] = {}
        for method, tasks_data in methods_data.items():
            if len(tasks_data) == len(TASKS):
                delta_by_seed = {task: {s: float(np.mean(tasks_data[task][s])) for s in SEEDS}
                                 for task in TASKS}
                seed_means = [float(np.mean([delta_by_seed[t][s] for t in TASKS])) for s in SEEDS]
                task_means[version][method] = {
                    "four_task_equal_mean_difference": float(np.mean(seed_means)),
                    "six_seed_sample_sd": float(np.std(seed_means, ddof=1)),
                    "paired_bootstrap_95_ci": paired_bootstrap(tasks_data)}
    report = {"schema_version": 1, "scope": "cohort-matched closed-loop outcomes; no latency claim",
              "runner_sha256": table._sha256(runner_path), "analysis_sha256": table._sha256(analysis_path),
              "combination_selection_sha256": selection_sha256,
              "final_combination_gate_sha256": final_selection_sha256,
              "task_equal_average": task_means, "conditions": rows,
              "complete_conditions": sum(x["status"] == "complete" for x in rows),
              "target_conditions": len(rows),
              "paired_interval": "95% percentile bootstrap stratified by task and seed, resampling matched episode pairs within each seed; 20,000 replicates",
              "notes": ["E0 self-comparison is included as a manifest and outcome consistency check.",
                        "Incomplete cells are not zero-filled.",
                        "Per-episode planning seconds are amortized exposure from 50-way closed-loop calls; they are not single-environment latency."]}
    table._write_json(OUT / "summary/closed_loop_paired.json", report)
    print(json.dumps({"complete_conditions": report["complete_conditions"], "target_conditions": len(rows),
                      "versions": {v: sum(x["version"] == v and x["status"] == "complete" for x in rows)
                                   for v in available}}))


if __name__ == "__main__":
    summarize()
