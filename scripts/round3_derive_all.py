"""Complete the four-protocol development result set from immutable traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from source.common.round3_derivation import derive_protocol_result_file
from source.common.round3_phase1 import (
    CohortManifest,
    canonical_result_path,
    canonical_trace_path,
)
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
    parser.add_argument("--action-block", type=int, default=5)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    cohort_root = output / "cohorts" / args.task
    round3_manifest = CohortManifest.load(cohort_root / "dev_round3_revised.json")
    sampling_manifest = CohortManifest.load(cohort_root / "dev_sampling_revised.json")
    round3 = canonical_result_path(
        output,
        task=args.task,
        method=args.method,
        protocol_variant="round3_revised",
        stage=args.stage,
        cohort_kind=round3_manifest.cohort_kind,
        cohort_sha256=round3_manifest.computed_sha256,
    )
    sampling = canonical_result_path(
        output,
        task=args.task,
        method=args.method,
        protocol_variant="sampling_revised",
        stage=args.stage,
        cohort_kind=sampling_manifest.cohort_kind,
        cohort_sha256=sampling_manifest.computed_sha256,
    )
    if not sampling.is_file():
        sampling_trace = canonical_trace_path(
            output,
            task=args.task,
            method=args.method,
            protocol_variant="sampling_revised",
            stage=args.stage,
            cohort_kind=sampling_manifest.cohort_kind,
            cohort_sha256=sampling_manifest.computed_sha256,
        )
        derive_sampling_identity_result_file(
            round3,
            sampling,
            manifest_path=cohort_root / "dev_sampling_revised.json",
            task=args.task,
            trace_output_dir=sampling_trace.parent,
        )
    derived = {}
    mappings = {
        "tolerance_revised": "legacy",
        "round3_revised": "sampling_revised",
    }
    for variant, source_variant in mappings.items():
        source_manifest = CohortManifest.load(
            cohort_root / f"dev_{source_variant}.json"
        )
        target_manifest = CohortManifest.load(cohort_root / f"dev_{variant}.json")
        source = canonical_result_path(
            output,
            task=args.task,
            method=args.method,
            protocol_variant=source_variant,
            stage=args.stage,
            cohort_kind=source_manifest.cohort_kind,
            cohort_sha256=source_manifest.computed_sha256,
        )
        target = canonical_result_path(
            output,
            task=args.task,
            method=args.method,
            protocol_variant=variant,
            stage=args.stage,
            cohort_kind=target_manifest.cohort_kind,
            cohort_sha256=target_manifest.computed_sha256,
        )
        target_trace = canonical_trace_path(
            output,
            task=args.task,
            method=args.method,
            protocol_variant=variant,
            stage=args.stage,
            cohort_kind=target_manifest.cohort_kind,
            cohort_sha256=target_manifest.computed_sha256,
        )
        derived[variant] = derive_protocol_result_file(
            source,
            target,
            manifest_path=cohort_root / f"dev_{variant}.json",
            task=args.task,
            protocol_variant=variant,
            action_block=args.action_block,
            trace_output_dir=target_trace.parent,
        )
    print(
        json.dumps(
            {
                "sampling_revised": json.loads(sampling.read_text(encoding="utf-8")).get("cohort_sha256"),
                **{variant: value.get("cohort_sha256") for variant, value in derived.items()},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
