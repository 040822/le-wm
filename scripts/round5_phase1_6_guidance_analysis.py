#!/usr/bin/env python3
"""Paper-facing paired analysis for the Phase1.6 PO/GF direction experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from source.common.round5_phase1_6 import atomic_json, stable_seed


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/round5/phase1_6_seed3072"
TASKS = ("pusht", "reacher")
METHODS = ("post_opt", "guided_flow")
VARIANTS = {
    "post_opt": ("post_opt_s2_c0.jsonl", "post_opt_s2_c1.jsonl"),
    "guided_flow": ("guided_flow_s2_c0.jsonl", "guided_flow_s2_c1.jsonl"),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"no rows in {path}")
    return rows


def _state_metric(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            grouped.setdefault(str(row["state_id"]), []).append(value)
    return {state: float(np.mean(values)) for state, values in grouped.items()}


def _bootstrap(
    values: Mapping[str, float], *, samples: int, seed: int
) -> dict[str, Any]:
    state_values = np.asarray([value for _, value in sorted(values.items())], dtype=np.float64)
    if len(state_values) == 0:
        return {"estimate": None, "ci95": [None, None], "p_signflip": None, "states": 0}
    rng = np.random.default_rng(stable_seed("phase1_6_guidance_cluster", seed))
    indices = rng.integers(0, len(state_values), size=(samples, len(state_values)))
    draws = state_values[indices].mean(axis=1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(samples, len(state_values)))
    null = (signs * state_values[None, :]).mean(axis=1)
    estimate = float(state_values.mean())
    return {
        "estimate": estimate,
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "p_signflip": float((1 + np.count_nonzero(np.abs(null) >= abs(estimate))) / (samples + 1)),
        "states": int(len(state_values)),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
    }


def _paired_state_values(
    rows: Sequence[Mapping[str, Any]], field: str, *, predicate: str | None = None
) -> dict[str, float]:
    selected = [row for row in rows if row.get("paired_25_valid")]
    if predicate is not None:
        selected = [row for row in selected if row.get(predicate)]
    return _state_metric(selected, field)


def _holm(rows: Sequence[dict[str, Any]]) -> None:
    valid = [(i, row.get("p_signflip")) for i, row in enumerate(rows) if row.get("p_signflip") is not None]
    ordered = sorted(valid, key=lambda item: float(item[1]))
    running = 0.0
    for rank, (index, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * float(pvalue)))
        rows[index]["p_holm"] = running


def _merge_candidates(rows_by_candidate: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    by_state: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    for rows in rows_by_candidate:
        for row in rows:
            identity = (str(row["state_id"]), int(row["candidate_index"]))
            if identity in seen:
                raise RuntimeError(f"duplicate guidance state/candidate pair: {identity}")
            seen.add(identity)
            by_state.setdefault(identity[0], []).append(dict(row))
    for state, rows in by_state.items():
        candidates = sorted(int(row["candidate_index"]) for row in rows)
        if candidates != [0, 1]:
            raise RuntimeError(f"state {state} does not have exactly candidates 0 and 1: {candidates}")
    return [row for state in sorted(by_state) for row in by_state[state]]


def analyze(output_root: Path, *, samples: int = 10_000) -> dict[str, Any]:
    guidance_root = output_root / "mechanism/guidance"
    report: dict[str, Any] = {
        "schema_version": "round5_phase1_6_guidance_paper_analysis_v1",
        "sample_unit": "legacy cohort state; candidate0/candidate1 averaged within state",
        "bootstrap_samples": samples,
        "tasks": {},
    }
    primary_tests: list[dict[str, Any]] = []

    for task_index, task in enumerate(TASKS):
        task_root = guidance_root / task
        variants: dict[str, list[dict[str, Any]]] = {}
        coverage: dict[str, Any] = {}
        for method_index, method in enumerate(METHODS):
            files = VARIANTS[method]
            candidate_rows = [_read_jsonl(task_root / "variants" / name) for name in files]
            rows = _merge_candidates(candidate_rows)
            variants[method] = rows
            valid = [row for row in rows if row.get("paired_25_valid")]
            match = [row for row in rows if row.get("random_rms_match_valid")]
            coverage[method] = {
                "states": len({str(row["state_id"]) for row in rows}),
                "candidate_branches": len(rows),
                "paired_25_branches": len(valid),
                "paired_25_states": len({str(row["state_id"]) for row in valid}),
                "rms_match_branches": len(match),
                "rms_match_fraction": len(match) / len(rows) if rows else None,
                "mean_guidance_rms_displacement": float(np.mean([
                    row["guided_action_rms_displacement_after_clip"] for row in rows
                ])),
                "mean_random_rms_displacement": float(np.mean([
                    row["random_action_rms_displacement_after_clip"] for row in rows
                ])),
                "max_absolute_rms_match_error": float(np.max([
                    abs(float(row["random_rms_match_error"])) for row in rows
                ])),
                "mean_a_forward_calls_per_guided_branch": float(np.mean([
                    row.get("guidance_stats", {}).get("stage_a_forward_count", 0) for row in rows
                ])),
                "mean_b_forward_calls_per_guided_branch": float(np.mean([
                    row.get("guidance_stats", {}).get("stage_b_forward_count", 0) for row in rows
                ])),
                "mean_b_backward_calls_per_guided_branch": float(np.mean([
                    row.get("guidance_stats", {}).get("backward_count", 0) for row in rows
                ])),
            }

        method_summaries: dict[str, Any] = {}
        for method_index, method in enumerate(METHODS):
            rows = variants[method]
            paired = [row for row in rows if row.get("paired_25_valid")]
            metrics: dict[str, Any] = {}
            metric_fields = {
                "guided_physical_improvement_vs_baseline": "guided_physical_improvement_vs_baseline",
                "random_physical_improvement_vs_baseline": "random_physical_improvement_vs_baseline",
                "guided_minus_random_physical_improvement": "guided_minus_random_physical_improvement",
                "guided_latent_improvement_vs_baseline": "guided_latent_improvement_vs_baseline",
                "random_latent_improvement_vs_baseline": "random_latent_improvement_vs_baseline",
            }
            for label, field in metric_fields.items():
                state_values = _paired_state_values(rows, field)
                metrics[label] = _bootstrap(
                    state_values, samples=samples,
                    seed=16026 + task_index * 1000 + method_index * 100 + len(metrics),
                )
            success_by_state: dict[str, list[float]] = {}
            for row in paired:
                success_by_state.setdefault(str(row["state_id"]), []).append(
                    float(bool(row["success_by_25"]["guided"])) - float(bool(row["success_by_25"]["random"]))
                )
            metrics["guided_minus_random_success_by_25"] = _bootstrap(
                {state: float(np.mean(values)) for state, values in success_by_state.items()},
                samples=samples, seed=16026 + task_index * 1000 + method_index * 100 + 31,
            )
            contradiction = [
                row for row in paired
                if float(row["guided_latent_improvement_vs_baseline"]) > 0
                and float(row["guided_physical_improvement_vs_baseline"]) < 0
            ]
            metrics["prediction_improves_but_physical_distance_worsens_fraction"] = (
                len(contradiction) / len(paired) if paired else None
            )
            metrics["guided_success_by_25_percent"] = (
                100.0 * float(np.mean([
                    bool(row["success_by_25"]["guided"]) for row in paired
                ])) if paired else None
            )
            metrics["random_success_by_25_percent"] = (
                100.0 * float(np.mean([
                    bool(row["success_by_25"]["random"]) for row in paired
                ])) if paired else None
            )
            primary = dict(metrics["guided_minus_random_physical_improvement"])
            primary.update({"task": task, "method": method, "comparison": f"{method} - matched random"})
            primary_tests.append(primary)
            method_summaries[method] = metrics

        gf_po_state: dict[str, float] = {}
        po_rows = variants["post_opt"]
        gf_rows = variants["guided_flow"]
        po_values = _paired_state_values(po_rows, "guided_physical_improvement_vs_baseline")
        gf_values = _paired_state_values(gf_rows, "guided_physical_improvement_vs_baseline")
        common = sorted(set(po_values) & set(gf_values))
        gf_po_state = {state: gf_values[state] - po_values[state] for state in common}
        gf_po = _bootstrap(gf_po_state, samples=samples, seed=16026 + task_index * 1000 + 777)
        gf_po.update({"comparison": "GF - PO", "task": task})
        primary_tests.append(dict(gf_po))
        report["tasks"][task] = {
            "coverage": coverage,
            "methods": method_summaries,
            "gf_minus_po_guided_physical_improvement": gf_po,
            "gf_minus_po_state_values": gf_po_state,
            "state_values": {
                method: {
                    "guided_physical_improvement_vs_baseline": _paired_state_values(
                        variants[method], "guided_physical_improvement_vs_baseline"
                    ),
                    "random_physical_improvement_vs_baseline": _paired_state_values(
                        variants[method], "random_physical_improvement_vs_baseline"
                    ),
                    "guided_minus_random_physical_improvement": _paired_state_values(
                        variants[method], "guided_minus_random_physical_improvement"
                    ),
                }
                for method in METHODS
            },
            "prediction_cost_movement": {
                method: {
                    "guided_cost_lower_fraction": float(np.mean([
                        float(row["guided_predicted_cost"]) < float(row["baseline_predicted_cost"])
                        for row in variants[method]
                    ])),
                    "guided_cost_lower_and_physical_worse_fraction": method_summaries[method][
                        "prediction_improves_but_physical_distance_worsens_fraction"
                    ],
                }
                for method in METHODS
            },
        }

        reverse_path = task_root / "candidate0_block_reverse.jsonl"
        reverse_rows = _read_jsonl(reverse_path)
        reverse_diffs = {
            str(row["state_id"]): float(row["baseline_distance_25"]) - float(row["reversed_distance_25"])
            for row in reverse_rows if row.get("paired_25_valid")
        }
        reverse_success = {
            str(row["state_id"]): float(bool(row["reversed_success_by_25"])) - float(bool(row["baseline_success_by_25"]))
            for row in reverse_rows if row.get("paired_25_valid")
        }
        report["tasks"][task]["reverse_action_block_control"] = {
            "branches": len(reverse_rows),
            "physical_improvement_vs_baseline": _bootstrap(
                reverse_diffs, samples=samples, seed=16026 + task_index * 1000 + 888
            ),
            "success_difference_by_25": _bootstrap(
                reverse_success, samples=samples, seed=16026 + task_index * 1000 + 889
            ),
        }

    _holm(primary_tests)
    report["primary_test_family"] = [
        {key: row.get(key) for key in (
            "task", "method", "comparison", "estimate", "ci95", "p_signflip", "p_holm", "states"
        )}
        for row in primary_tests
    ]

    out_dir = output_root / "analysis/paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "analysis/guidance_paper_summary.json"
    atomic_json(json_path, report)
    csv_path = out_dir / "guidance_table.csv"
    table_rows = []
    for task in TASKS:
        for method in METHODS:
            summary = report["tasks"][task]["methods"][method]
            physical = summary["guided_physical_improvement_vs_baseline"]
            random = summary["random_physical_improvement_vs_baseline"]
            gap = summary["guided_minus_random_physical_improvement"]
            success = summary["guided_minus_random_success_by_25"]
            coverage_row = report["tasks"][task]["coverage"][method]
            table_rows.append({
                "task": task, "method": method,
                "states": physical["states"],
                "paired_branches": coverage_row["paired_25_branches"],
                "guided_minus_baseline_distance_improvement": physical["estimate"],
                "guided_vs_baseline_ci95_low": physical["ci95"][0],
                "guided_vs_baseline_ci95_high": physical["ci95"][1],
                "random_minus_baseline_distance_improvement": random["estimate"],
                "guided_minus_random_distance_improvement": gap["estimate"],
                "guided_minus_random_ci95_low": gap["ci95"][0],
                "guided_minus_random_ci95_high": gap["ci95"][1],
                "guided_minus_random_p_holm": next(
                    row.get("p_holm") for row in primary_tests
                    if row.get("task") == task and row.get("method") == method
                ),
                "guided_minus_random_success_pp": 100.0 * success["estimate"],
                "rms_match_fraction": coverage_row["rms_match_fraction"],
                "a_forward_calls_per_guided_branch": coverage_row["mean_a_forward_calls_per_guided_branch"],
                "b_forward_calls_per_guided_branch": coverage_row["mean_b_forward_calls_per_guided_branch"],
                "b_backward_calls_per_guided_branch": coverage_row["mean_b_backward_calls_per_guided_branch"],
                "prediction_better_physical_worse_fraction": summary[
                    "prediction_improves_but_physical_distance_worsens_fraction"
                ],
            })
    fields = list(table_rows[0]) if table_rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table_rows)
    atomic_json(out_dir / "guidance_table.json", {"rows": table_rows})
    tex_rows = []
    for row in table_rows:
        tex_rows.append(
            f"{row['task']} & {row['method']} & {row['guided_minus_baseline_distance_improvement']:.3f}"
            f" & [{row['guided_vs_baseline_ci95_low']:.3f}, {row['guided_vs_baseline_ci95_high']:.3f}]"
            f" & {row['guided_minus_random_distance_improvement']:.3f}"
            f" & [{row['guided_minus_random_ci95_low']:.3f}, {row['guided_minus_random_ci95_high']:.3f}]"
            f" & {row['guided_minus_random_success_pp']:.1f}"
            f" & {row['a_forward_calls_per_guided_branch']:.1f}/{row['b_forward_calls_per_guided_branch']:.1f}/{row['b_backward_calls_per_guided_branch']:.1f}"
            f" & {100.0 * row['rms_match_fraction']:.1f}\\% \\\\"
        )
    tex = (
        "\\begin{tabular}{llrrrrrrr}\n\\toprule\n"
        "Task & Method & $\\Delta d$ vs base & 95\\% CI & $\\Delta d$ vs rand. & 95\\% CI & $\\Delta$ success (pp) & A/B fwd, B bwd & RMS match \\\\\n"
        "\\midrule\n" + "\n".join(tex_rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex_path = out_dir / "guidance_table.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return {
        "summary": str(json_path), "csv": str(csv_path), "latex": str(tex_path),
        "primary_tests": len(primary_tests),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    result = analyze(args.output_root, samples=args.bootstrap_samples)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
