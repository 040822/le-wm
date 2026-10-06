"""Paper-facing analysis helpers for Round 5 Phase1.6 artifacts.

This module is deliberately separate from the live evaluation runner so that
analysis improvements do not change the code identity of in-flight runs.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .round5_phase1_6 import atomic_json, stable_hash, stable_seed


PRIMARY_COMPARISONS = (
    ("p3_s2", "random64_s2", "P3 - Random-64"),
    ("p0_po2_s2", "p0_s2", "PO - P0"),
    ("p0_gf2_s2", "p0_po2_s2", "GF - PO"),
)
EXPLORATORY_COMPARISONS = (("p3_po5_s2", "p3_s2", "P3+PO - P3"),)
CEM_REFERENCE_COMPARISONS = (
    ("p3_s2", "p2_cem_300x30", "P3 - P2 CEM (300x30)"),
    ("p3_po5_s2", "p2_cem_300x30", "P3+PO - P2 CEM (300x30)"),
    ("p3_s2", "p2_small_64x5", "P3 - P2-small CEM (64x5)"),
    ("p3_po5_s2", "p2_small_64x5", "P3+PO - P2-small CEM (64x5)"),
)
TRAINING_COMPARISONS = (
    ("continue", "frozen", "Continue - Frozen"),
    ("detach", "continue", "Detach - Continue"),
    ("recorded", "detach", "Recorded - Detach"),
)


def _json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _result_path(root: Path, task: str, method: str, seed: int, tag: str | None) -> Path:
    leaf = f"seed_{int(seed)}" + (f"_{tag}" if tag else "")
    return root / "conditions" / task / method / leaf / "result.json"


def _episode_values(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for episode in payload.get("episodes", ()):
        key = f"{episode.get('episode_id')}:{episode.get('start_step')}"
        steps = {
            int(row["raw_env_step"]): row
            for row in episode.get("steps", ())
            if isinstance(row, Mapping) and row.get("raw_env_step") is not None
        }
        first_success = episode.get("first_success_step")
        outcomes: dict[str, Any] = {
            "success": float(bool(episode.get("success"))),
            "terminal_distance": _finite_or_none(episode.get("terminal_distance")),
            "first_success_step": first_success,
            "steps_executed": int(episode.get("steps_executed", len(steps))),
        }
        for fixed_step in (5, 25):
            outcomes[f"success_by_{fixed_step}"] = (
                1.0 if first_success is not None and int(first_success) <= fixed_step
                else 0.0 if outcomes["steps_executed"] >= fixed_step
                else None
            )
            sample = steps.get(fixed_step)
            outcomes[f"distance_at_{fixed_step}"] = (
                _finite_or_none(sample.get("distance")) if sample else None
            )
        values[key] = outcomes
    return values


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _load_condition(
    root: Path, task: str, method: str, seeds: Sequence[int], tag: str | None = None
) -> dict[str, Any]:
    per_seed: dict[int, dict[str, dict[str, Any]]] = {}
    result_rows = []
    for seed in seeds:
        path = _result_path(root, task, method, seed, tag)
        payload = _json(path)
        if payload is None or payload.get("status") != "ok":
            continue
        per_seed[int(seed)] = _episode_values(payload)
        phase = payload.get("phase1_6_condition", {})
        result_rows.append({
            "seed": int(seed),
            "result_path": str(path),
            "result_sha256": stable_hash(payload),
            "episodes": len(payload.get("episodes", ())),
            "success_rate": _finite_or_none(payload.get("success_rate")),
            "evaluation_seconds": _finite_or_none(payload.get("evaluation_seconds")),
            "planning_p50_seconds": _finite_or_none(
                payload.get("round4_planning", {}).get("planning_median_seconds")
            ),
            "peak_memory_bytes": phase.get("peak_memory_bytes"),
            "checkpoint_sha256": phase.get("checkpoint_sha256"),
            "cohort_sha256": phase.get("cohort_sha256"),
        })
    return {"per_seed": per_seed, "rows": result_rows}


def _cluster_differences(
    left: Mapping[int, Mapping[str, Mapping[str, Any]]],
    right: Mapping[int, Mapping[str, Mapping[str, Any]]],
    metric: str,
) -> dict[str, list[float]]:
    common_seeds = sorted(set(left) & set(right))
    state_ids = set.intersection(*(
        set(left[seed]) & set(right[seed]) for seed in common_seeds
    )) if common_seeds else set()
    differences: dict[str, list[float]] = {}
    for state in sorted(state_ids):
        rows = []
        for seed in common_seeds:
            a = _finite_or_none(left[seed][state].get(metric))
            b = _finite_or_none(right[seed][state].get(metric))
            if a is not None and b is not None:
                rows.append(a - b)
        if rows:
            differences[state] = rows
    return differences


def _cluster_summary(
    values: Mapping[str, Sequence[float]], *, samples: int, seed: int
) -> dict[str, Any]:
    state_means = np.asarray([
        np.mean(np.asarray(rows, dtype=np.float64))
        for _, rows in sorted(values.items())
        if len(rows)
    ], dtype=np.float64)
    if len(state_means) == 0:
        return {"estimate": None, "ci95": [None, None], "one_sided_95_lower": None, "p_signflip": None, "states": 0, "rows": 0}
    rng = np.random.default_rng(stable_seed("phase1_6_cluster_bootstrap", seed))
    indices = rng.integers(0, len(state_means), size=(samples, len(state_means)))
    draws = state_means[indices].mean(axis=1)
    estimate = float(state_means.mean())
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(samples, len(state_means)))
    null_draws = (signs * state_means[None, :]).mean(axis=1)
    pvalue = float((1 + np.count_nonzero(np.abs(null_draws) >= abs(estimate))) / (samples + 1))
    return {
        "estimate": estimate,
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "one_sided_95_lower": float(np.quantile(draws, 0.05)),
        "p_signflip": pvalue,
        "states": int(len(state_means)),
        "rows": int(sum(len(rows) for rows in values.values())),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
    }


def _holm(rows: Sequence[dict[str, Any]], p_key: str = "p_signflip") -> None:
    valid = [(index, row.get(p_key)) for index, row in enumerate(rows) if row.get(p_key) is not None]
    ordered = sorted(valid, key=lambda item: float(item[1]))
    count = len(ordered)
    running = 0.0
    for rank, (index, pvalue) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * float(pvalue))
        running = max(running, adjusted)
        rows[index]["p_holm"] = running


def _metric_summary(
    conditions: Mapping[tuple[str, str], Mapping[str, Any]],
    *, tasks: Sequence[str], comparisons: Sequence[tuple[str, str, str]],
    metric: str, samples: int, seed: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {"metric": metric, "by_task": {}, "macro_task_mean": {}}
    family_rows: list[dict[str, Any]] = []
    macro_differences: dict[str, dict[str, list[float]]] = {}
    for task_index, task in enumerate(tasks):
        results["by_task"][task] = {}
        for left, right, label in comparisons:
            a = conditions.get((task, left), {}).get("per_seed", {})
            b = conditions.get((task, right), {}).get("per_seed", {})
            diff = _cluster_differences(a, b, metric)
            detail = _cluster_summary(
                diff, samples=samples, seed=seed + task_index * 97 + len(family_rows)
            )
            detail.update({"comparison": label, "left": left, "right": right})
            results["by_task"][task][label] = detail
            macro_differences.setdefault(label, {})[task] = diff
            family_rows.append(detail)
    _holm(family_rows)
    # Task-stratified bootstrap preserves equal task weighting for the macro mean.
    for _, _, label in comparisons:
        task_values = macro_differences.get(label, {})
        point_estimates = []
        draws_by_task = []
        state_count = 0
        for task_index, task in enumerate(tasks):
            state_means = np.asarray([
                np.mean(np.asarray(rows, dtype=np.float64))
                for rows in task_values.get(task, {}).values() if len(rows)
            ], dtype=np.float64)
            if len(state_means) == 0:
                continue
            point_estimates.append(float(state_means.mean()))
            state_count += len(state_means)
            rng = np.random.default_rng(stable_seed("phase1_6_macro_bootstrap", seed, label, task))
            indices = rng.integers(0, len(state_means), size=(samples, len(state_means)))
            draws_by_task.append(state_means[indices].mean(axis=1))
        if point_estimates:
            draws = np.stack(draws_by_task, axis=0).mean(axis=0)
            results["macro_task_mean"][label] = {
                "estimate": float(np.mean(point_estimates)),
                "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
                "one_sided_95_lower": float(np.quantile(draws, 0.05)),
                "tasks": len(point_estimates),
                "states": state_count,
                "bootstrap_samples": samples,
                "task_weights": "equal",
            }
        else:
            results["macro_task_mean"][label] = {"estimate": None, "ci95": [None, None], "one_sided_95_lower": None, "tasks": 0, "states": 0}
    return results


def _timing(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = _json(root / "timing" / "summary.json") or {}
    out = {}
    for row in payload.get("conditions", ()):
        task, method = row.get("task"), row.get("method")
        if task and method:
            out[(str(task), str(method))] = dict(row)
    return out


def _tex_escape(text: str) -> str:
    mapping = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    return "".join(mapping.get(char, char) for char in text)


def _write_closed_loop_table(
    target_dir: Path, tasks: Sequence[str], methods: Sequence[str],
    conditions: Mapping[tuple[str, str], Mapping[str, Any]],
    timings: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for method in methods:
        row: dict[str, Any] = {"method": method}
        for task in tasks:
            condition = conditions.get((task, method), {})
            episodes = [
                episode
                for seed_map in condition.get("per_seed", {}).values()
                for episode in seed_map.values()
            ]
            row[f"{task}_success_percent"] = (
                100.0 * float(np.mean([e["success"] for e in episodes])) if episodes else None
            )
            timing = timings.get((task, method), {})
            row[f"{task}_batch1_p50_ms"] = _ms(timing.get("batch1", {}).get("p50_seconds"))
            row[f"{task}_batch1_p95_ms"] = _ms(timing.get("batch1", {}).get("p95_seconds"))
            row[f"{task}_batch50_throughput"] = _finite_or_none(
                timing.get("batch50", {}).get("throughput_per_second")
            )
        task_rates = [row.get(f"{task}_success_percent") for task in tasks]
        task_rates = [value for value in task_rates if value is not None]
        row["macro_success_percent"] = float(np.mean(task_rates)) if task_rates else None
        rows.append(row)

    csv_path = target_dir / "closed_loop_table.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["method"])
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(target_dir / "closed_loop_table.json", {"rows": rows})

    tex_rows = []
    for row in rows:
        cells = [_tex_escape(row["method"])]
        for task in tasks:
            sr = row.get(f"{task}_success_percent")
            latency = row.get(f"{task}_batch1_p50_ms")
            cells.extend(("--" if sr is None else f"{sr:.1f}", "--" if latency is None else f"{latency:.2f}"))
        macro = row.get("macro_success_percent")
        cells.append("--" if macro is None else f"{macro:.1f}")
        tex_rows.append(" & ".join(cells) + r" \\")
    header = ["Method"]
    for task in tasks:
        title = {"cube": "Cube", "pusht": "PushT", "reacher": "Reacher", "tworoom": "TwoRoom"}.get(task, task)
        header.extend((f"{title} SR (\\%)", f"{title} p50 (ms)"))
    header.append("Macro SR (\\%)")
    columns = "l" + "rr" * len(tasks) + "r"
    tex = (
        f"\\begin{{tabular}}{{{columns}}}\n\\toprule\n"
        + " & ".join(header) + r" \\" + "\n\\midrule\n"
        + "\n".join(tex_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (target_dir / "closed_loop_table.tex").write_text(tex, encoding="utf-8")
    return {
        "csv": str(csv_path),
        "json": str(target_dir / "closed_loop_table.json"),
        "latex": str(target_dir / "closed_loop_table.tex"),
        "latency_available": any(
            row.get(f"{task}_batch1_p50_ms") is not None for row in rows for task in tasks
        ),
        "rows": len(rows),
    }


def _coverage_summary(
    conditions: Mapping[tuple[str, str], Mapping[str, Any]], tasks: Sequence[str], methods: Sequence[str]
) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        for method in methods:
            condition = conditions.get((task, method), {})
            metrics = (
                "success", "success_by_5", "success_by_25", "distance_at_5",
                "distance_at_25", "terminal_distance",
            )
            counts = {name: 0 for name in metrics}
            total = 0
            per_seed = {}
            for seed, episodes in condition.get("per_seed", {}).items():
                per_seed_total = len(episodes)
                total += per_seed_total
                seed_counts = {name: 0 for name in metrics}
                for episode in episodes.values():
                    for name in metrics:
                        if episode.get(name) is not None:
                            counts[name] += 1
                            seed_counts[name] += 1
                per_seed[str(seed)] = {
                    name: {
                        "valid": seed_counts[name],
                        "total": per_seed_total,
                        "fraction": seed_counts[name] / per_seed_total if per_seed_total else None,
                    }
                    for name in metrics
                }
            rows.append({
                "task": task, "method": method, "total_episodes": total,
                "metrics": {
                    name: {"valid": count, "total": total, "fraction": count / total if total else None}
                    for name, count in counts.items()
                },
                "by_seed": per_seed,
            })
    return rows


def _training_analysis(
    root: Path, tasks: Sequence[str], seeds: Sequence[int], bootstrap_samples: int,
    tags: Mapping[str, str],
) -> dict[str, Any]:
    tasks = tuple(tasks)
    methods = ("p0_s2", "p3_s2")
    arms = ("frozen", "continue", "detach", "recorded")
    conditions: dict[tuple[str, str, str], dict[str, Any]] = {}
    for task in tasks:
        for arm in arms:
            tag = None if arm == "frozen" else tags.get(arm, arm)
            for method in methods:
                conditions[(task, arm, method)] = _load_condition(root, task, method, seeds, tag)

    comparisons = TRAINING_COMPARISONS
    summaries: dict[str, Any] = {}
    family_rows = []
    for task_index, task in enumerate(tasks):
        summaries[task] = {}
        for method_index, method in enumerate(methods):
            summaries[task][method] = {}
            for left, right, label in comparisons:
                a = conditions[(task, left, method)].get("per_seed", {})
                b = conditions[(task, right, method)].get("per_seed", {})
                diff = _cluster_differences(a, b, "success")
                detail = _cluster_summary(
                    diff, samples=bootstrap_samples,
                    seed=16026 + task_index * 431 + method_index * 67 + len(family_rows),
                )
                detail.update({"comparison": label, "left": left, "right": right, "task": task, "method": method})
                summaries[task][method][label] = detail
                if label != "Continue - Frozen":
                    family_rows.append(detail)
    _holm(family_rows)

    table_rows = []
    for task in tasks:
        for arm in arms:
            tag = None if arm == "frozen" else tags.get(arm, arm)
            for method in methods:
                payload = conditions[(task, arm, method)]
                episodes = [
                    episode for seed_map in payload.get("per_seed", {}).values()
                    for episode in seed_map.values()
                ]
                row = {
                    "task": task, "training_arm": arm, "inference_method": method,
                    "result_tag": tag,
                    "episodes": len(episodes),
                    "success_percent": 100.0 * float(np.mean([e["success"] for e in episodes])) if episodes else None,
                    "seeds_present": sorted(payload.get("per_seed", {})),
                }
                table_rows.append(row)
    return {
        "arms": list(arms),
        "result_tags": dict(tags),
        "primary_success_effects": summaries,
        "continue_minus_frozen_exploratory": {
            task: {
                method: rows.get("Continue - Frozen")
                for method, rows in per_method.items()
            }
            for task, per_method in summaries.items()
        },
        "holm_family": [
            {key: row.get(key) for key in ("task", "method", "comparison", "p_signflip", "p_holm", "estimate", "ci95", "states")}
            for row in family_rows
        ],
        "table_rows": table_rows,
        "conditions": {
            f"{task}/{arm}/{method}": {
                "seeds_present": sorted(payload.get("per_seed", {})),
                "episodes_by_seed": {str(seed): len(value) for seed, value in payload.get("per_seed", {}).items()},
            }
            for (task, arm, method), payload in conditions.items()
        },
    }


def _write_training_table(target_dir: Path, analysis: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(analysis.get("table_rows", ()))
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / "training_intervention_table.csv"
    fields = ["task", "training_arm", "inference_method", "result_tag", "episodes", "success_percent", "seeds_present"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "seeds_present": ",".join(str(x) for x in row["seeds_present"])})
    atomic_json(target_dir / "training_intervention_table.json", {"rows": rows})
    tasks = list(dict.fromkeys(row["task"] for row in rows))
    methods = list(dict.fromkeys(row["inference_method"] for row in rows))
    arms = list(dict.fromkeys(row["training_arm"] for row in rows))
    tex_lines = []
    for task in tasks:
        for method in methods:
            line = [_tex_escape(task), _tex_escape(method)]
            for arm in arms:
                row = next((r for r in rows if r["task"] == task and r["inference_method"] == method and r["training_arm"] == arm), None)
                rate = row.get("success_percent") if row else None
                line.append("--" if rate is None else f"{rate:.1f}")
            tex_lines.append(" & ".join(line) + r" \\")
    tex = (
        "\\begin{tabular}{ll" + "r" * len(arms) + "}\n\\toprule\n"
        + "Task & Inference" + " & " + " & ".join(_tex_escape(arm.title()) for arm in arms) + r" \\" + "\n\\midrule\n"
        + "\n".join(tex_lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (target_dir / "training_intervention_table.tex").write_text(tex, encoding="utf-8")
    return {"csv": str(csv_path), "json": str(target_dir / "training_intervention_table.json"), "latex": str(target_dir / "training_intervention_table.tex"), "rows": len(rows)}


def _ms(value: Any) -> float | None:
    finite = _finite_or_none(value)
    return finite * 1000.0 if finite is not None else None


def analyze_phase1_6(
    *, output_root: str | Path, tasks: Sequence[str], methods: Sequence[str],
    seeds: Sequence[int], bootstrap_samples: int = 10_000,
    training_tags: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    conditions = {
        (task, method): _load_condition(root, task, method, seeds)
        for task in tasks for method in methods
    }
    timing = _timing(root)
    complete = [
        {"task": task, "method": method,
         "seeds_present": sorted(conditions[(task, method)]["per_seed"]),
         "episodes_by_seed": {str(seed): len(episodes) for seed, episodes in conditions[(task, method)]["per_seed"].items()}}
        for task in tasks for method in methods
    ]
    summaries = {}
    for metric in ("success", "success_by_5", "success_by_25", "distance_at_5", "distance_at_25", "terminal_distance"):
        summaries[metric] = _metric_summary(
            conditions, tasks=tasks, comparisons=PRIMARY_COMPARISONS,
            metric=metric, samples=bootstrap_samples, seed=16026,
        )
    exploratory = {
        "by_task": {},
    }
    for task_index, task in enumerate(tasks):
        a = conditions.get((task, EXPLORATORY_COMPARISONS[0][0]), {}).get("per_seed", {})
        b = conditions.get((task, EXPLORATORY_COMPARISONS[0][1]), {}).get("per_seed", {})
        diff = _cluster_differences(a, b, "success")
        exploratory["by_task"][task] = _cluster_summary(
            diff, samples=bootstrap_samples, seed=16407 + task_index
        )
    cem_reference = _metric_summary(
        conditions, tasks=tasks, comparisons=CEM_REFERENCE_COMPARISONS,
        metric="success", samples=bootstrap_samples, seed=17826,
    )
    coverage = _coverage_summary(conditions, tasks, methods)
    training = _training_analysis(
        root, tasks=(task for task in tasks if task in ("pusht", "reacher")),
        seeds=seeds, bootstrap_samples=bootstrap_samples,
        tags=dict(training_tags or {}),
    )
    out = {
        "schema_version": "round5_phase1_6_paper_analysis_v1",
        "output_root": str(root.resolve()),
        "tasks": list(tasks), "methods": list(methods), "seeds": [int(x) for x in seeds],
        "conditions": complete,
        "condition_result_rows": [
            {"task": task, "method": method, **row}
            for (task, method), condition in conditions.items()
            for row in condition.get("rows", ())
        ],
        "metrics": summaries,
        "exploratory_p3_po_effects": exploratory,
        "cem_reference_success_effects": cem_reference,
        "fixed_step_coverage": coverage,
        "training": training,
        "timing_conditions": [
            {"task": task, "method": method, **dict(value)}
            for (task, method), value in sorted(timing.items())
        ],
        "notes": [
            "Fixed-time success is null when the rollout terminated before the requested step without prior success.",
            "Distances at steps 5 and 25 include only traces with that exact raw environment step; coverage is reported per condition.",
            "Macro effects use an equal-weight average over task-specific effects.",
            "CEM-reference effects are paired on the same confirmation starts; non-inferiority requires the prespecified confidence-bound margins and latency reduction, not a nonsignificant difference.",
            "Latency is read only from isolated synchronous timing artifacts; closed-loop wall time is not treated as inference latency.",
        ],
    }
    table = _write_closed_loop_table(root / "analysis/paper_tables", tasks, methods, conditions, timing)
    training_table = _write_training_table(root / "analysis/paper_tables", training)
    out["closed_loop_table"] = table
    out["training_table"] = training_table
    target = root / "analysis/paper_summary.json"
    atomic_json(target, out)
    return out


__all__ = ["analyze_phase1_6", "PRIMARY_COMPARISONS", "TRAINING_COMPARISONS"]
