#!/usr/bin/env python3
"""Compare R4-AB and R4-ABDE result roots on matched cohorts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.round4_compare import (  # noqa: E402
    DEFAULT_COMPARISON_MODES,
    compare_result_roots,
)


def _write_csv(payload: dict, path: Path) -> None:
    fields = (
        "task",
        "mode",
        "cohort_kind",
        "baseline_success_rate_percent",
        "treatment_success_rate_percent",
        "delta_pp",
        "paired_n",
        "paired_improved",
        "paired_regressed",
        "paired_mcnemar_exact_two_sided_p",
        "baseline_planning_median_seconds",
        "treatment_planning_median_seconds",
        "baseline_planning_p95_seconds",
        "treatment_planning_p95_seconds",
        "baseline_forward_count",
        "treatment_forward_count",
        "baseline_peak_memory_bytes",
        "treatment_peak_memory_bytes",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in payload["rows"]:
            paired = row["paired"]
            baseline = row["baseline_planning"]
            treatment = row["treatment_planning"]
            writer.writerow(
                {
                    "task": row["task"],
                    "mode": row["mode"],
                    "cohort_kind": row["cohort_kind"],
                    "baseline_success_rate_percent": row["baseline_success_rate_percent"],
                    "treatment_success_rate_percent": row["treatment_success_rate_percent"],
                    "delta_pp": row["delta_pp"],
                    "paired_n": paired.get("n"),
                    "paired_improved": paired.get("improved"),
                    "paired_regressed": paired.get("regressed"),
                    "paired_mcnemar_exact_two_sided_p": paired.get(
                        "mcnemar_exact_two_sided_p"
                    ),
                    "baseline_planning_median_seconds": baseline.get(
                        "planning_median_seconds"
                    ),
                    "treatment_planning_median_seconds": treatment.get(
                        "planning_median_seconds"
                    ),
                    "baseline_planning_p95_seconds": baseline.get(
                        "planning_p95_seconds"
                    ),
                    "treatment_planning_p95_seconds": treatment.get(
                        "planning_p95_seconds"
                    ),
                    "baseline_forward_count": baseline.get("forward_count"),
                    "treatment_forward_count": treatment.get("forward_count"),
                    "baseline_peak_memory_bytes": baseline.get("peak_memory_bytes"),
                    "treatment_peak_memory_bytes": treatment.get("peak_memory_bytes"),
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ab-root", required=True)
    parser.add_argument("--abde-root", required=True)
    parser.add_argument(
        "--ab-p3-root",
        help="optional canonical P3 root when P0--P2 live in another AB root",
    )
    parser.add_argument(
        "--abde-p3-root",
        help="optional canonical P3 root when P0--P2 live in another ABDE root",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="directory receiving analysis.json and analysis.csv",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(DEFAULT_COMPARISON_MODES),
        choices=("P0", "P1", "P2", "P3"),
    )
    parser.add_argument("--cohorts", nargs="+", default=["dev", "final"])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    payload = compare_result_roots(
        args.ab_root,
        args.abde_root,
        modes=args.modes,
        cohort_kinds=args.cohorts,
        baseline_mode_roots={"P3": args.ab_p3_root} if args.ab_p3_root else None,
        treatment_mode_roots={"P3": args.abde_p3_root} if args.abde_p3_root else None,
    )
    json_path = output / "analysis.json"
    csv_path = output / "analysis.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(payload, csv_path)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "rows": len(payload["rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
