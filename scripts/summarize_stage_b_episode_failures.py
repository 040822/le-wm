"""Summarize exact-replay and grounded Stage-B failure diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from source.diagnostics.stage_b_episode_failures import (
    load_diagnostic_manifest,
    select_diagnostic_slots,
)


def _reference_vector(run):
    payload = json.loads(run.reference_result.read_text(encoding="utf-8"))
    return [bool(episode["success"]) for episode in payload["episodes"]]


def _next_sections(labels):
    sections = set()
    if labels & {
        "coverage_failure",
        "ranking_failure",
        "cem_exploitation",
        "replan_regression",
        "action_ood",
    }:
        sections.add("§3.3")
    if "dynamics_error" in labels:
        sections.add("§3.4")
    if "latent_metric_error" in labels:
        sections.add("§5.1")
    return sorted(sections)


def summarize(*, manifest_path, output_dir):
    manifest = load_diagnostic_manifest(manifest_path)
    output_dir = Path(output_dir).expanduser().resolve()
    rows = []
    task_selections = {}
    all_labels = set()
    for task in ("pusht", "reacher"):
        vectors = {
            run.family: _reference_vector(run)
            for run in manifest.runs.values()
            if run.task == task
            and run.family in {"e0", "e1_384", "e3_384"}
        }
        task_selections[task] = select_diagnostic_slots(vectors)

    for label, run in manifest.runs.items():
        reproduction_path = output_dir / label / "reproduction.json"
        if not reproduction_path.is_file():
            rows.append(
                {
                    "run_label": label,
                    "task": run.task,
                    "family": run.family,
                    "status": "not_run",
                    "mismatch_slots": [],
                    "labels": [],
                }
            )
            continue
        reproduction = json.loads(
            reproduction_path.read_text(encoding="utf-8")
        )
        grounded_path = output_dir / label / "grounded" / "result.json"
        labels = set()
        if grounded_path.is_file():
            grounded = json.loads(grounded_path.read_text(encoding="utf-8"))
            for slot in grounded["slots"]:
                labels.update(slot["labels"])
        all_labels.update(labels)
        rows.append(
            {
                "run_label": label,
                "task": run.task,
                "family": run.family,
                "status": reproduction["status"],
                "mismatch_slots": reproduction["mismatch_slots"],
                "trace": reproduction["trace"],
                "labels": sorted(labels),
            }
        )

    completed = [row for row in rows if row["status"] != "not_run"]
    reproducible = [
        row for row in completed if row["status"] == "reproducible"
    ]
    summary = {
        "status": (
            "accepted"
            if len(completed) == len(rows)
            and len(reproducible) == len(rows)
            else "incomplete_or_non_reproducible"
        ),
        "runs_expected": len(rows),
        "runs_completed": len(completed),
        "runs_reproducible": len(reproducible),
        "slot_selection": task_selections,
        "recommended_next_sections": _next_sections(all_labels),
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    comparison = [
        "# Stage-B episode failure comparison",
        "",
        "| task | run | reproduction | mismatches | grounded labels |",
        "|---|---|---|---:|---|",
    ]
    for row in rows:
        comparison.append(
            f"| {row['task']} | {row['family']} | {row['status']} | "
            f"{len(row['mismatch_slots'])} | "
            f"{', '.join(row['labels']) or '—'} |"
        )
    (output_dir / "comparison.md").write_text(
        "\n".join(comparison) + "\n",
        encoding="utf-8",
    )

    report = [
        "# §2.3 当前失败 Episode 无重训诊断",
        "",
        f"- 完成 trace run：{len(completed)}/{len(rows)}",
        f"- 精确复现：{len(reproducible)}/{len(rows)}",
        f"- 验收状态：{summary['status']}",
        "",
        "## 复现门",
        "",
    ]
    for row in rows:
        mismatch = (
            f"；mismatch slots={row['mismatch_slots']}"
            if row["mismatch_slots"]
            else ""
        )
        report.append(
            f"- {row['run_label']}: {row['status']}{mismatch}"
        )
    report.extend(
        [
            "",
            "## 后续",
            "",
            (
                "- 当前复现门未全量通过；non-reproducible run 不进行机制归因，"
                "第二轮实验报告暂不同步为已验收结论。"
            ),
            (
                "- 已有 grounded 标签建议进入："
                + (
                    "、".join(summary["recommended_next_sections"])
                    if summary["recommended_next_sections"]
                    else "尚无（等待 grounding 或复现）"
                )
            ),
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    result = summarize(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )
    print(
        f"status={result['status']} "
        f"completed={result['runs_completed']}/{result['runs_expected']} "
        f"reproducible={result['runs_reproducible']}"
    )


if __name__ == "__main__":
    main()
