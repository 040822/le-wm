"""Derive tolerance/joint Phase 1 results from immutable development traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from source.common.round3_derivation import derive_protocol_result_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "round3" / "phase1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("cube", "pusht", "reacher", "tworoom"))
    parser.add_argument("method", choices=("e0_lewm", "e3_fast", "e5_fast"))
    parser.add_argument(
        "stage", choices=("stage_a", "stage_a_shuffled_goal", "stage_b")
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--action-block", type=int, default=5)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    result_root = output / "results" / args.task / args.method
    cohort_root = output / "cohorts" / args.task
    derived = {}
    sources = {
        "tolerance_revised": result_root / "legacy" / args.stage / "result.json",
        "round3_revised": result_root / "sampling_revised" / args.stage / "result.json",
    }
    for variant, source in sources.items():
        target_dir = result_root / variant / args.stage
        derived[variant] = derive_protocol_result_file(
            source,
            target_dir,
            manifest_path=cohort_root / f"dev_{variant}.json",
            task=args.task,
            protocol_variant=variant,
            action_block=args.action_block,
            trace_output_dir=output / "traces" / args.task / args.method / variant / args.stage,
        )
    print(
        json.dumps(
            {variant: value.get("cohort_sha256") for variant, value in derived.items()},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
