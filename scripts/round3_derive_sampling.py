"""Derive sampling-revised metadata from an immutable round3 trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from source.common.round3_sampling_identity import derive_sampling_identity_result_file


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
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    result_root = output / "results" / args.task / args.method
    source = result_root / "round3_revised" / args.stage / "result.json"
    target = result_root / "sampling_revised" / args.stage
    payload = derive_sampling_identity_result_file(
        source,
        target,
        manifest_path=output / "cohorts" / args.task / "dev_sampling_revised.json",
        task=args.task,
        trace_output_dir=output / "traces" / args.task / args.method / "sampling_revised" / args.stage,
    )
    print(json.dumps({"cohort_sha256": payload.get("cohort_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
