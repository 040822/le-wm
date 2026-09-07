"""Write per-entry protocol sensitivity analyses from Phase 1 result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from source.common.round3_analysis import analyze_protocol_files
from source.common.round3_phase1 import PHASE1_MATRIX
from source.common.round3_protocol import PROTOCOL_VARIANTS, TASKS
from source.common.round3_validation import validate_result_payload


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "round3" / "phase1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--action-block", type=int, default=5)
    return parser


def _result_paths(root: Path, task: str, method: str, stage: str) -> dict[str, Path]:
    return {
        variant: root / "results" / task / method / variant / stage / "result.json"
        for variant in PROTOCOL_VARIANTS
    }


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    analysis_root = output / "analysis"
    counts = {"ok": 0, "missing": 0, "invalid": 0}
    for task in TASKS:
        for method, stages in PHASE1_MATRIX[task].items():
            for stage in stages:
                paths = _result_paths(output, task, method, stage)
                status = "ok"
                payloads = {}
                for variant, path in paths.items():
                    if not path.is_file():
                        status = "missing"
                        break
                    try:
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        validate_result_payload(payload)
                        payloads[variant] = payload
                    except (OSError, json.JSONDecodeError, ValueError):
                        status = "invalid"
                        break
                target = analysis_root / task / method / f"{stage}.json"
                if status != "ok":
                    counts[status] += 1
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(
                        json.dumps(
                            {
                                "protocol": "round3_phase1_v1",
                                "task": task,
                                "method": method,
                                "stage": stage,
                                "status": status,
                                "missing_or_invalid": [
                                    variant for variant, path in paths.items() if not path.is_file()
                                ],
                            },
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    continue
                analysis = analyze_protocol_files(
                    {variant: str(path) for variant, path in paths.items()},
                    task=task,
                    action_block=args.action_block,
                )
                analysis.update({"method": method, "stage": stage, "status": "ok"})
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                counts["ok"] += 1
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
