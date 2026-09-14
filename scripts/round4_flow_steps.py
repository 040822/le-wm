#!/usr/bin/env python3
"""Analyze the Round 4 R4-AB action-flow step ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.round4_flow_steps import (  # noqa: E402
    DEFAULT_ACTION_MODES,
    DEFAULT_COHORT_KINDS,
    DEFAULT_FLOW_STEPS,
    DEFAULT_INVARIANT_MODES,
    analyze_flow_step_results,
)
from source.common.round4_protocol import ROUND4_TASKS  # noqa: E402


def _write_csv(payload: dict, path: Path) -> None:
    fields = (
        "task",
        "mode",
        "cohort_kind",
        "flow_steps",
        "invariant",
        "success_rate_percent",
        "canonical_success_rate_percent",
        "delta_pp_vs_canonical",
        "paired_n",
        "paired_improved",
        "paired_regressed",
        "paired_mcnemar_exact_two_sided_p",
        "planning_median_seconds",
        "planning_p95_seconds",
        "forward_count",
        "peak_memory_bytes",
        "path",
        "canonical_path",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in payload["rows"]:
            paired = row.get("paired_vs_canonical") or {}
            planning = row.get("planning") or {}
            writer.writerow(
                {
                    "task": row["task"],
                    "mode": row["mode"],
                    "cohort_kind": row["cohort_kind"],
                    "flow_steps": row["flow_steps"],
                    "invariant": row["invariant"],
                    "success_rate_percent": row["success_rate_percent"],
                    "canonical_success_rate_percent": row[
                        "canonical_success_rate_percent"
                    ],
                    "delta_pp_vs_canonical": row["delta_pp_vs_canonical"],
                    "paired_n": paired.get("n"),
                    "paired_improved": paired.get("improved"),
                    "paired_regressed": paired.get("regressed"),
                    "paired_mcnemar_exact_two_sided_p": paired.get(
                        "mcnemar_exact_two_sided_p"
                    ),
                    "planning_median_seconds": planning.get(
                        "planning_median_seconds"
                    ),
                    "planning_p95_seconds": planning.get("planning_p95_seconds"),
                    "forward_count": planning.get("forward_count"),
                    "peak_memory_bytes": planning.get("peak_memory_bytes"),
                    "path": row["path"],
                    "canonical_path": row["canonical_path"],
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default=str(ROOT / "outputs/round4/ab_flow_steps_seed3072"),
        help="root containing steps_<N> result directories",
    )
    parser.add_argument(
        "--invariant-root",
        default=str(ROOT / "outputs/round4/ab_flow_steps_seed3072/invariant_p1"),
        help="root containing the invariant P1 results",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="directory receiving analysis.json and analysis.csv",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=list(DEFAULT_FLOW_STEPS),
    )
    parser.add_argument("--canonical-step", type=int, default=16)
    parser.add_argument("--tasks", nargs="+", default=list(ROUND4_TASKS))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    results_root = Path(args.results_root)
    step_roots = {step: results_root / f"steps_{step}" for step in args.steps}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    payload = analyze_flow_step_results(
        step_roots,
        args.invariant_root,
        tasks=tuple(args.tasks),
        action_modes=DEFAULT_ACTION_MODES,
        invariant_modes=DEFAULT_INVARIANT_MODES,
        cohort_kinds=DEFAULT_COHORT_KINDS,
        canonical_step=int(args.canonical_step),
    )
    json_path = output / "analysis.json"
    csv_path = output / "analysis.csv"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(payload, csv_path)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "rows": len(payload["rows"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
