#!/usr/bin/env python3
"""Export Phase1.6 analysis artifacts into the manuscript table directory."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.round5_phase1_6 import atomic_json


OUTPUT = ROOT / "outputs/round5/phase1_6_seed3072"
TABLES = ROOT / "paper/tables/phase1_6"
FIGURES = ROOT / "paper/fig"


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_tex(path: Path, spec: str, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    body = [" & ".join(row) + r" \\" for row in rows]
    table = (
        f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n"
        + " & ".join(headers) + r" \\" + "\n\\midrule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    path.write_text(table, encoding="utf-8")


def _p_text(value: Any) -> str:
    if value is None:
        return "--"
    value = float(value)
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _candidate_tables() -> dict[str, str]:
    source = OUTPUT / "mechanism/candidate_consequences.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    consequence_rows = []
    for task, task_payload in payload["tasks"].items():
        value = task_payload["candidate_consequence"]
        cosine = value["predicted_vs_true_latent_delta_cosine"]
        physical = value["predicted_cost_vs_true_physical_distance_sign_accuracy"]
        latent_sign = value["predicted_cost_vs_true_latent_cost_sign_accuracy"]
        shuffle = value["within_state_shuffle_negative_control"]
        consequence_rows.append({
            "task": task,
            "states": value["analyzed_states"],
            "candidate_pairs": value["candidate_pairs_preselected"],
            "latent_identifiability_coverage": value["latent_identifiability_coverage"],
            "predicted_true_latent_delta_cosine": cosine["estimate"],
            "latent_delta_ci95_low": cosine["ci95"][0],
            "latent_delta_ci95_high": cosine["ci95"][1],
            "cost_physical_sign_accuracy": physical["estimate"],
            "physical_sign_ci95_low": physical["ci95"][0],
            "physical_sign_ci95_high": physical["ci95"][1],
            "cost_latent_sign_accuracy": latent_sign["estimate"],
            "shuffle_latent_p_one_sided": shuffle["latent_delta_cosine_empirical_p_one_sided"],
            "shuffle_physical_sign_p_one_sided": shuffle["cost_sign_empirical_p_one_sided"],
            "excluded_clip_incompatible_states": value["excluded_clip_incompatible_states"],
        })
    _write_csv(TABLES / "action_consequence_summary.csv", consequence_rows)
    atomic_json(TABLES / "action_consequence_summary.json", {"rows": consequence_rows})
    consequence_tex = []
    for row in consequence_rows:
        consequence_tex.append([
            row["task"], str(row["states"]), str(row["candidate_pairs"]), f"{row['predicted_true_latent_delta_cosine']:.3f}",
            f"[{row['latent_delta_ci95_low']:.3f}, {row['latent_delta_ci95_high']:.3f}]",
            f"{100 * row['cost_physical_sign_accuracy']:.1f}\\%",
            f"[{100 * row['physical_sign_ci95_low']:.1f}, {100 * row['physical_sign_ci95_high']:.1f}]\\%",
            f"{row['shuffle_latent_p_one_sided']:.3f}",
            f"{row['shuffle_physical_sign_p_one_sided']:.3f}",
        ])
    _write_tex(
        TABLES / "action_consequence_summary.tex", "lrrrrrrrr",
        ["Task", "States", "Pairs", "$\\cos(\\Delta z,\\widehat{\\Delta z})$", "95\\% CI", "Cost sign acc.", "95\\% CI", "$p_{\\rm latent}$", "$p_{\\rm cost}$"],
        consequence_tex,
    )

    curve_path = OUTPUT / "mechanism/candidate_selection_n_curve.csv"
    curve_rows = _csv_rows(curve_path)
    selected_rows = []
    for task in ("pusht", "reacher"):
        for n in ("1", "4", "16", "64"):
            selection = {
                row["selector"]: row for row in curve_rows
                if row["task"] == task and row["n"] == n and row["metric"] == "success_by_25"
            }
            delta = selection["b_minus_random"]
            selected_rows.append({
                "task": task, "n": int(n),
                "b_success": float(selection["b"]["estimate"]),
                "b_ci95_low": float(selection["b"]["ci95_low"]),
                "b_ci95_high": float(selection["b"]["ci95_high"]),
                "random_success": float(selection["random_uniform"]["estimate"]),
                "latent_oracle_success": float(selection["latent_oracle"]["estimate"]),
                "physical_oracle_success": float(selection["physical_oracle"]["estimate"]),
                "b_minus_random": float(delta["estimate"]),
                "b_minus_random_ci95_low": float(delta["ci95_low"]),
                "b_minus_random_ci95_high": float(delta["ci95_high"]),
                "states": int(delta["states"]),
            })
    _write_csv(TABLES / "candidate_selection_summary.csv", selected_rows)
    atomic_json(TABLES / "candidate_selection_summary.json", {"rows": selected_rows})
    selection_tex = []
    for row in selected_rows:
        selection_tex.append([
            row["task"], str(row["n"]), f"{100*row['b_success']:.1f}\\%",
            f"{100*row['random_success']:.1f}\\%", f"{100*row['latent_oracle_success']:.1f}\\%",
            f"{100*row['physical_oracle_success']:.1f}\\%",
            f"{100*row['b_minus_random']:.1f} pp",
            f"[{100*row['b_minus_random_ci95_low']:.1f}, {100*row['b_minus_random_ci95_high']:.1f}]",
        ])
    _write_tex(
        TABLES / "candidate_selection_summary.tex", "llrrrrrr",
        ["Task", "$N$", "B", "Random", "Latent oracle", "Physical oracle", "B $-$ rand. (pp)", "95\\% CI"],
        selection_tex,
    )

    physical_rows = []
    for task in ("pusht", "reacher"):
        b_random_distance = next(row for row in curve_rows
            if row["task"] == task and row["n"] == "64"
            and row["metric"] == "distance_at_25_improvement"
            and row["selector"] == "random_minus_b")
        physical_regret = next(row for row in curve_rows
            if row["task"] == task and row["n"] == "64"
            and row["metric"] == "physical_distance_regret" and row["selector"] == "b")
        physical_rows.append({
            "task": task, "states": int(b_random_distance["states"]),
            "random_minus_b_distance_improvement": float(b_random_distance["estimate"]),
            "improvement_ci95_low": float(b_random_distance["ci95_low"]),
            "improvement_ci95_high": float(b_random_distance["ci95_high"]),
            "b_physical_oracle_regret": float(physical_regret["estimate"]),
            "regret_ci95_low": float(physical_regret["ci95_low"]),
            "regret_ci95_high": float(physical_regret["ci95_high"]),
        })
    _write_csv(TABLES / "candidate_selection_physical.csv", physical_rows)
    atomic_json(TABLES / "candidate_selection_physical.json", {"rows": physical_rows})
    _write_tex(
        TABLES / "candidate_selection_physical.tex", "lrrrrr",
        ["Task", "States", "Rand. $-$ B distance", "95\\% CI", "B physical regret", "95\\% CI"],
        [[
            row["task"], str(row["states"]), f"{row['random_minus_b_distance_improvement']:.3f}",
            f"[{row['improvement_ci95_low']:.3f}, {row['improvement_ci95_high']:.3f}]",
            f"{row['b_physical_oracle_regret']:.3f}",
            f"[{row['regret_ci95_low']:.3f}, {row['regret_ci95_high']:.3f}]",
        ] for row in physical_rows],
    )

    shutil.copy2(curve_path, TABLES / "candidate_selection_n_curve.csv")
    shutil.copy2(OUTPUT / "mechanism/candidate_consequences.csv", TABLES / "action_consequence.csv")
    return {
        "action_consequence_csv": str(TABLES / "action_consequence_summary.csv"),
        "candidate_selection_csv": str(TABLES / "candidate_selection_summary.csv"),
        "candidate_selection_physical_csv": str(TABLES / "candidate_selection_physical.csv"),
        "n_curve_csv": str(TABLES / "candidate_selection_n_curve.csv"),
    }


def _selection_figure() -> str:
    rows = _csv_rows(OUTPUT / "mechanism/candidate_selection_n_curve.csv")
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), sharey=True)
    selectors = (
        ("b", "B selected", "#0072B2"),
        ("random_uniform", "Random", "#D55E00"),
        ("latent_oracle", "Future-latent oracle", "#009E73"),
        ("physical_oracle", "Physical oracle", "#CC79A7"),
    )
    Ns = (1, 4, 16, 64)
    for axis, task in zip(axes, ("pusht", "reacher")):
        task_rows = [row for row in rows if row["task"] == task and row["metric"] == "success_by_25"]
        for selector, label, color in selectors:
            selected = [next(row for row in task_rows if row["selector"] == selector and int(row["n"]) == n) for n in Ns]
            values = np.asarray([float(row["estimate"]) for row in selected])
            low = np.asarray([float(row["ci95_low"]) for row in selected])
            high = np.asarray([float(row["ci95_high"]) for row in selected])
            axis.errorbar(Ns, values, yerr=np.stack((values-low, high-values)), marker="o", linewidth=1.6, capsize=3, label=label, color=color)
        axis.set_xscale("log", base=2)
        axis.set_xticks(Ns, labels=[str(n) for n in Ns])
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(task.title())
        axis.set_xlabel("Nested candidate count $N$")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Success by step 25")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    figure.suptitle("Legacy 50-state candidate diagnostic (exploratory)", y=1.02)
    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    target = FIGURES / "phase1_6_candidate_selection_curve.pdf"
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return str(target)


METHOD_LABELS = {
    "p0_s2": "P0",
    "random64_s2": "Random-64",
    "p3_s2": "P3",
    "p0_po2_s2": "P0+PO",
    "p0_gf2_s2": "P0+GF",
    "p3_po5_s2": "P3+PO",
    "p1_cem_300x30": "P1 CEM",
    "p2_cem_300x30": "P2 CEM",
    "p2_small_64x5": "P2-small",
}


def _manuscript_tables() -> dict[str, Any]:
    summary_path = OUTPUT / "analysis/paper_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tasks = tuple(summary["tasks"])
    methods = tuple(summary["methods"])
    condition_rows = summary.get("condition_result_rows", ())
    success_rows: list[dict[str, Any]] = []
    success_lookup: dict[tuple[str, str], float] = {}
    for method in methods:
        row: dict[str, Any] = {"method": method, "label": METHOD_LABELS.get(method, method)}
        rates = []
        for task in tasks:
            values = [item for item in condition_rows if item["task"] == task and item["method"] == method]
            denominator = sum(int(item.get("episodes", 0)) for item in values)
            successes = sum(
                float(item.get("success_rate") or 0.0) * int(item.get("episodes", 0))
                for item in values
            )
            rate = successes / denominator if denominator else None
            success_lookup[(task, method)] = rate if rate is not None else float("nan")
            row[f"{task}_success_percent"] = 100.0 * rate if rate is not None else None
            if rate is not None:
                rates.append(rate)
        row["macro_success_percent"] = 100.0 * float(np.mean(rates)) if rates else None
        success_rows.append(row)
    _write_csv(TABLES / "closed_loop_success.csv", success_rows)
    atomic_json(TABLES / "closed_loop_success.json", {"rows": success_rows})
    _write_tex(
        TABLES / "closed_loop_success.tex", "lrrrrr",
        ["Method", "Cube", "Push-T", "Reacher", "TwoRoom", "Macro"],
        [[
            row["label"],
            *["--" if row.get(f"{task}_success_percent") is None else f"{row[f'{task}_success_percent']:.1f}\\%" for task in tasks],
            "--" if row.get("macro_success_percent") is None else f"{row['macro_success_percent']:.1f}\\%",
        ] for row in success_rows],
    )

    contrast_labels = {
        "P3 - Random-64": "P3 $-$ Random-64",
        "PO - P0": "P0+PO $-$ P0",
        "GF - PO": "P0+GF $-$ P0+PO",
    }
    effect_rows = []
    success_metric = summary["metrics"]["success"]
    for task in tasks:
        for label, detail in success_metric["by_task"][task].items():
            effect_rows.append({
                "task": task,
                "comparison": label,
                "difference_pp": 100.0 * detail["estimate"],
                "ci95_low_pp": 100.0 * detail["ci95"][0],
                "ci95_high_pp": 100.0 * detail["ci95"][1],
                "p_signflip": detail.get("p_signflip"),
                "p_holm": detail.get("p_holm"),
                "states": detail["states"],
                "rows": detail["rows"],
            })
    for label, detail in success_metric["macro_task_mean"].items():
        effect_rows.append({
            "task": "equal-task macro", "comparison": label,
            "difference_pp": 100.0 * detail["estimate"],
            "ci95_low_pp": 100.0 * detail["ci95"][0],
            "ci95_high_pp": 100.0 * detail["ci95"][1],
            "p_signflip": None, "p_holm": None,
            "states": detail["states"], "rows": None,
        })
    _write_csv(TABLES / "primary_inference_effects.csv", effect_rows)
    atomic_json(TABLES / "primary_inference_effects.json", {"rows": effect_rows})
    _write_tex(
        TABLES / "primary_inference_effects.tex", "llrrrrr",
        ["Task", "Contrast", "Effect (pp)", "95\\% CI", "$p$", "$p_{\\rm Holm}$", "States"],
        [[
            row["task"], contrast_labels[row["comparison"]], f"{row['difference_pp']:.1f}",
            f"[{row['ci95_low_pp']:.1f}, {row['ci95_high_pp']:.1f}]",
            _p_text(row["p_signflip"]), _p_text(row["p_holm"]),
            str(row["states"]),
        ] for row in effect_rows],
    )

    cem_labels = {
        "P3 - P2 CEM (300x30)": "P3 $-$ P2 CEM",
        "P3+PO - P2 CEM (300x30)": "P3+PO $-$ P2 CEM",
        "P3 - P2-small CEM (64x5)": "P3 $-$ P2-small",
        "P3+PO - P2-small CEM (64x5)": "P3+PO $-$ P2-small",
    }
    cem_analysis = summary["cem_reference_success_effects"]
    cem_rows = []
    for task in tasks:
        for label, detail in cem_analysis["by_task"][task].items():
            cem_rows.append({
                "task": task, "comparison": label,
                "difference_pp": 100.0 * detail["estimate"],
                "ci95_low_pp": 100.0 * detail["ci95"][0],
                "ci95_high_pp": 100.0 * detail["ci95"][1],
                "p_signflip": detail["p_signflip"], "p_holm": detail["p_holm"],
                "states": detail["states"],
            })
    for label, detail in cem_analysis["macro_task_mean"].items():
        cem_rows.append({
            "task": "equal-task macro", "comparison": label,
            "difference_pp": 100.0 * detail["estimate"],
            "ci95_low_pp": 100.0 * detail["ci95"][0],
            "ci95_high_pp": 100.0 * detail["ci95"][1],
            "p_signflip": None, "p_holm": None, "states": detail["states"],
        })
    _write_csv(TABLES / "cem_reference_effects.csv", cem_rows)
    atomic_json(TABLES / "cem_reference_effects.json", {"rows": cem_rows})
    _write_tex(
        TABLES / "cem_reference_effects.tex", "llrrrrr",
        ["Task", "Contrast", "Effect (pp)", "95\\% CI", "$p$", "$p_{\\rm Holm}$", "States"],
        [[
            row["task"], cem_labels[row["comparison"]], f"{row['difference_pp']:.1f}",
            f"[{row['ci95_low_pp']:.1f}, {row['ci95_high_pp']:.1f}]",
            _p_text(row["p_signflip"]), _p_text(row["p_holm"]), str(row["states"]),
        ] for row in cem_rows],
    )

    train_rows = []
    for task, task_values in summary["training"]["primary_success_effects"].items():
        for method, method_values in task_values.items():
            for contrast, detail in method_values.items():
                train_rows.append({
                    "task": task, "method": METHOD_LABELS.get(method, method),
                    "comparison": contrast,
                    "difference_pp": 100.0 * detail["estimate"],
                    "ci95_low_pp": 100.0 * detail["ci95"][0],
                    "ci95_high_pp": 100.0 * detail["ci95"][1],
                    "p_signflip": detail.get("p_signflip"),
                    "p_holm": detail.get("p_holm"),
                    "states": detail["states"],
                })
    _write_csv(TABLES / "training_intervention_effects.csv", train_rows)
    atomic_json(TABLES / "training_intervention_effects.json", {"rows": train_rows})
    _write_tex(
        TABLES / "training_intervention_effects.tex", "lllrrrrr",
        ["Task", "Inference", "Contrast", "Effect (pp)", "95\\% CI", "$p$", "$p_{\\rm Holm}$", "States"],
        [[
            row["task"], row["method"], row["comparison"], f"{row['difference_pp']:.1f}",
            f"[{row['ci95_low_pp']:.1f}, {row['ci95_high_pp']:.1f}]",
            _p_text(row["p_signflip"]), _p_text(row["p_holm"]), str(row["states"]),
        ] for row in train_rows],
    )

    timing_path = OUTPUT / "timing/summary.json"
    timing_rows: list[dict[str, Any]] = []
    if timing_path.is_file():
        timing_summary = json.loads(timing_path.read_text(encoding="utf-8"))
        for condition in timing_summary.get("conditions", ()):
            task, method = condition["task"], condition["method"]
            timing_rows.append({
                "task": task, "method": method, "label": METHOD_LABELS.get(method, method),
                "seed": condition.get("seed"),
                "batch1_p50_ms": 1000.0 * condition.get("batch1", {}).get("p50_seconds", float("nan")),
                "batch1_p95_ms": 1000.0 * condition.get("batch1", {}).get("p95_seconds", float("nan")),
                "batch50_throughput_per_second": condition.get("batch50", {}).get("throughput_per_second"),
                "batch50_p50_ms": 1000.0 * condition.get("batch50", {}).get("p50_seconds", float("nan")),
                "peak_memory_bytes": condition.get("batch1", {}).get("peak_memory_bytes"),
                "interference_observed": condition.get("interference_observed"),
                "host_pressure_observed": condition.get("host_pressure_observed"),
                "eligible_for_speedup_claim": condition.get("interference_eligible_for_speedup_claim"),
            })
    timing_complete = len(timing_rows) == len(tasks) * len(methods)
    if timing_rows and timing_complete:
        _write_csv(TABLES / "inference_timing.csv", timing_rows)
        atomic_json(TABLES / "inference_timing.json", {"rows": timing_rows})
        timing_lookup = {(row["task"], row["method"]): row for row in timing_rows}
        _write_tex(
            TABLES / "inference_timing.tex", "lrrrr",
            ["Method", "Cube", "Push-T", "Reacher", "TwoRoom"],
            [[
                METHOD_LABELS.get(method, method),
                *["--" if (row := timing_lookup.get((task, method))) is None else f"{row['batch1_p50_ms']:.1f}/{row['batch1_p95_ms']:.1f}" + ("$^\\dagger$" if not row["eligible_for_speedup_claim"] else "") for task in tasks],
            ] for method in methods],
        )
        _write_tex(
            TABLES / "batch50_throughput.tex", "lrrrr",
            ["Method", "Cube", "Push-T", "Reacher", "TwoRoom"],
            [[
                METHOD_LABELS.get(method, method),
                *["--" if (row := timing_lookup.get((task, method))) is None else f"{row['batch50_throughput_per_second']:.1f}" + ("$^\\dagger$" if not row["eligible_for_speedup_claim"] else "") for task in tasks],
            ] for method in methods],
        )
        throughput_rows = [
            {"method": row["method"], "label": row["label"], "task": row["task"], "batch_size": 50,
             "throughput_per_second": row["batch50_throughput_per_second"],
             "p50_ms": row["batch50_p50_ms"], "interference_observed": row["interference_observed"],
             "host_pressure_observed": row["host_pressure_observed"],
             "eligible_for_speedup_claim": row["eligible_for_speedup_claim"]}
            for row in timing_rows
        ]
        _write_csv(TABLES / "batch50_throughput.csv", throughput_rows)
        atomic_json(TABLES / "batch50_throughput.json", {"rows": throughput_rows})
        _frontier_figure(success_lookup, timing_rows)

        timing_lookup = {(row["task"], row["method"]): row for row in timing_rows}
        substitution: dict[str, Any] = {}
        for policy, comparison in (
            ("p3_s2", "P3 - P2 CEM (300x30)"),
            ("p3_po5_s2", "P3+PO - P2 CEM (300x30)"),
        ):
            task_rows = {}
            for task in tasks:
                effect = cem_analysis["by_task"][task][comparison]
                candidate_time = timing_lookup.get((task, policy))
                cem_time = timing_lookup.get((task, "p2_cem_300x30"))
                reduction = None
                timing_clean = False
                if candidate_time and cem_time and candidate_time["eligible_for_speedup_claim"] and cem_time["eligible_for_speedup_claim"]:
                    reduction = 1.0 - candidate_time["batch1_p50_ms"] / cem_time["batch1_p50_ms"]
                    timing_clean = True
                task_rows[task] = {
                    "success_difference_pp": 100.0 * effect["estimate"],
                    "success_ci95_pp": [100.0 * x for x in effect["ci95"]],
                    "success_one_sided_95_lower_pp": 100.0 * effect["one_sided_95_lower"],
                    "task_noninferiority_margin_pp": -4.0,
                    "task_noninferiority_pass": 100.0 * effect["one_sided_95_lower"] > -4.0,
                    "p_holm": effect["p_holm"],
                    "latency_reduction_fraction": reduction,
                    "timing_uninterrupted": timing_clean,
                    "latency_reduction_pass": reduction is not None and reduction >= 0.20,
                }
            macro = cem_analysis["macro_task_mean"][comparison]
            macro_pass = 100.0 * macro["one_sided_95_lower"] > -2.0
            all_latency = all(row["latency_reduction_pass"] and row["timing_uninterrupted"] for row in task_rows.values())
            substitution[policy] = {
                "cem_reference": "P2 CEM 300x30",
                "task_results": task_rows,
                "macro_success_difference_pp": 100.0 * macro["estimate"],
                "macro_success_ci95_pp": [100.0 * x for x in macro["ci95"]],
                "macro_success_one_sided_95_lower_pp": 100.0 * macro["one_sided_95_lower"],
                "macro_noninferiority_margin_pp": -2.0,
                "macro_noninferiority_pass": 100.0 * macro["one_sided_95_lower"] > -2.0,
                "all_tasks_latency_reduction_at_least_20_percent": all_latency,
                "replace_large_cem_performance_latency_criteria_pass": macro_pass and all(
                    row["task_noninferiority_pass"] for row in task_rows.values()
                ) and all_latency,
            }
        atomic_json(TABLES / "cem_substitution_assessment.json", substitution)

    return {
        "closed_loop_success_rows": len(success_rows),
        "primary_effect_rows": len(effect_rows),
        "cem_reference_effect_rows": len(cem_rows),
        "training_effect_rows": len(train_rows),
        "timing_rows": len(timing_rows),
    }


def _frontier_figure(
    success: Mapping[tuple[str, str], float], timing_rows: Sequence[Mapping[str, Any]]
) -> str | None:
    if not timing_rows:
        return None
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {method: color_cycle[index % len(color_cycle)] for index, method in enumerate(METHOD_LABELS)}
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for axis, task in zip(axes.flat, ("cube", "pusht", "reacher", "tworoom")):
        for row in timing_rows:
            if row["task"] != task:
                continue
            method = row["method"]
            x = float(row["batch1_p50_ms"])
            y = 100.0 * float(success[(task, method)])
            axis.scatter(x, y, color=colors.get(method), marker="x" if not row["eligible_for_speedup_claim"] else "o", s=34)
            axis.annotate(METHOD_LABELS.get(method, method), (x, y), fontsize=6, xytext=(3, 2), textcoords="offset points")
        axis.set_title(task.title())
        axis.set_xlabel("Batch-1 p50 planning latency (ms)")
        axis.set_xlim(left=0)
        axis.set_ylim(0, 102)
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Closed-loop success (%)")
    axes[1, 0].set_ylabel("Closed-loop success (%)")
    figure.suptitle("Round5 Phase1.6 success--latency trade-off\n× marks a timing window with GPU or host pressure")
    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    target = FIGURES / "phase1_6_success_latency_frontier.pdf"
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return str(target)


def export_paper_tables() -> dict[str, Any]:
    TABLES.mkdir(parents=True, exist_ok=True)
    from scripts.round5_phase1_6 import METHODS
    from scripts.round5_phase1_6_guidance_analysis import analyze as analyze_guidance
    from source.common.round5_phase1_6_analysis import analyze_phase1_6

    analyze_phase1_6(
        output_root=OUTPUT,
        tasks=("cube", "pusht", "reacher", "tworoom"),
        methods=tuple(METHODS),
        seeds=(42, 43),
        bootstrap_samples=10_000,
        training_tags={"continue": "continue", "detach": "detach", "recorded": "recorded"},
    )
    analyze_guidance(OUTPUT, samples=10_000)
    sources = {
        "closed_loop_table": OUTPUT / "analysis/paper_tables/closed_loop_table",
        "training_intervention_table": OUTPUT / "analysis/paper_tables/training_intervention_table",
        "guidance_table": OUTPUT / "analysis/paper_tables/guidance_table",
    }
    copied: dict[str, list[str]] = {}
    for stem, source in sources.items():
        copied[stem] = []
        for suffix in ("csv", "json", "tex"):
            item = source.with_suffix("." + suffix)
            if item.is_file():
                target = TABLES / item.name
                shutil.copy2(item, target)
                copied[stem].append(str(target))
    mechanism = _candidate_tables()
    manuscript = _manuscript_tables()
    return {
        "tables": copied,
        "mechanism_tables": mechanism,
        "manuscript_tables": manuscript,
        "candidate_selection_figure": _selection_figure(),
    }


if __name__ == "__main__":
    print(json.dumps(export_paper_tables(), ensure_ascii=False, sort_keys=True))
