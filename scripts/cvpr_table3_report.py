#!/usr/bin/env python3
"""Rebuild CVPR Table 3 probe summaries from accepted row-level artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROBE_ROOT = ROOT / "outputs/cvpr/table3/v1/probe"
TASKS = ("cube", "tworoom", "pusht", "reacher")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _summary(values: list[float | None]) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return {
        "mean": float(np.mean(finite)) if finite else None,
        "sample_std": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
        "n": len(finite),
    }


def _metric_row(rows: list[dict[str, Any]], model: str, attribute: str, readout: str) -> dict[str, Any]:
    matches = [
        row for row in rows
        if row.get("model") == model and row.get("attribute") == attribute and row.get("readout") == readout
    ]
    if not matches:
        raise ValueError(f"missing accepted probe metric {model}/{attribute}/{readout}")
    if readout == "Linear":
        if len(matches) != 1:
            raise ValueError(f"expected one deterministic Linear fit, found {len(matches)}")
        row = matches[0]
        return {
            "standardized_mse": {"mean": row["standardized_mse"], "sample_std": None, "n": 1},
            "pearson_r": {"mean": row.get("pearson_r"), "sample_std": None, "n": int(row.get("pearson_r") is not None)},
            "physical_mse": row.get("physical_mse"),
            "physical_mae": row.get("physical_mae"),
            "sample_count": int(row["sample_count"]),
            "coordinate_count": int(row["coordinate_count"]),
        }
    if readout == "MLP":
        if len(matches) != 3 or {int(row["readout_seed"]) for row in matches} != {0, 1, 2}:
            raise ValueError(f"expected MLP readout seeds 0,1,2; found {len(matches)} rows")
        return {
            "standardized_mse": _summary([row["standardized_mse"] for row in matches]),
            "pearson_r": _summary([row.get("pearson_r") for row in matches]),
            "physical_mse": _summary([row.get("physical_mse") for row in matches]),
            "physical_mae": _summary([row.get("physical_mae") for row in matches]),
            "sample_count": int(matches[0]["sample_count"]),
            "coordinate_count": int(matches[0]["coordinate_count"]),
            "readout_seeds": [0, 1, 2],
        }
    raise ValueError(f"unsupported final readout {readout}")


def rebuild_probe_summary(task: str) -> dict[str, Any]:
    out = PROBE_ROOT / task
    acceptance = _read_json(out / "acceptance.json")
    if acceptance.get("status") != "pass" or acceptance.get("errors"):
        raise RuntimeError(f"{task}: probe output has not passed acceptance")
    config = _read_json(out / "frozen_config.json")
    schema = _read_json(out / "target_schema.json")
    metrics = _read_json(out / "metrics.json")["metrics"]
    with (out / "row_manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    splits: dict[str, list[dict[str, str]]] = {}
    for row in manifest:
        splits.setdefault(row["split"], []).append(row)
    for split in ("train", "validation", "test"):
        episodes = {row["episode_id"] for row in splits.get(split, [])}
        other = {
            row["episode_id"]
            for name, values in splits.items()
            if name != split
            for row in values
        }
        if episodes & other:
            raise RuntimeError(f"{task}: episode leakage in persisted row manifest")
    attributes = schema.get("attributes", {})
    summary = {
        "task": task,
        "acceptance": "pass",
        "data": {
            "sampled_frames": int(config["data"]["actual_sampled_frames"]),
            "episode_split_counts": config["data"]["episode_split_counts"],
            "row_split_counts": {name: len(values) for name, values in splits.items()},
            "test_episode_count": len({row["episode_id"] for row in splits.get("test", [])}),
            "dataset_path": config["data"]["path"],
            "dataset_sha256": config["data"]["sha256"],
            "row_manifest_sha256": acceptance.get("row_manifest_sha256", acceptance.get("manifest_sha256")),
        },
        "code": config["code"],
        "attributes": {},
        "overall": {},
        "source_disclosure": schema.get("source_audit", config.get("probe_method_source_audit")),
    }
    for attribute, target in attributes.items():
        summary["attributes"][attribute] = {
            "target": target,
            "cowm": {
                readout.lower(): _metric_row(metrics, "cowm", attribute, readout)
                for readout in ("Linear", "MLP")
            },
            "lewm": {
                readout.lower(): _metric_row(metrics, "lewm", attribute, readout)
                for readout in ("Linear", "MLP")
            },
        }
    for model in ("cowm", "lewm"):
        summary["overall"][model] = {}
        for readout in ("Linear", "MLP"):
            matches = [
                row for row in metrics
                if row.get("model") == model and row.get("attribute") == "overall" and row.get("readout") == readout
            ]
            if len(matches) != 1:
                raise ValueError(f"{task}: expected one Overall {model}/{readout} row")
            row = matches[0]
            if readout == "Linear":
                summary["overall"][model][readout.lower()] = {
                    "standardized_mse": {"mean": row["standardized_mse"], "sample_std": None, "n": 1},
                    "pearson_r": {"mean": row.get("pearson_r"), "sample_std": None, "n": int(row.get("pearson_r") is not None)},
                }
            else:
                seed_rows = row["seed_results"]
                summary["overall"][model][readout.lower()] = {
                    "standardized_mse": _summary([item["standardized_mse"] for item in seed_rows]),
                    "pearson_r": _summary([item.get("pearson_r") for item in seed_rows]),
                    "readout_seeds": [0, 1, 2],
                }
    return summary


def _format(value: dict[str, Any], key: str) -> str:
    item = value.get(key, {})
    mean, sd = item.get("mean"), item.get("sample_std")
    if mean is None:
        return "NA"
    if sd is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {sd:.4f}"


def _field_label(target: dict[str, Any]) -> str:
    field = str(target.get("field", ""))
    target_slice = target.get("slice")
    if target_slice is not None:
        field += f"[{int(target_slice[0])}:{int(target_slice[1])}]"
    return field


def _tex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def _tex_metric_cells(readout: dict[str, Any]) -> list[str]:
    def tex_value(key: str) -> str:
        item = readout.get(key, {})
        mean, sd = item.get("mean"), item.get("sample_std")
        if mean is None:
            return "NA"
        if sd is None:
            return f"${mean:.4f}$"
        return f"${mean:.4f} \\pm {sd:.4f}$"

    return [
        tex_value("standardized_mse"),
        tex_value("pearson_r"),
    ]


def _probe_tex(summaries: dict[str, dict[str, Any]]) -> str:
    labels = {
        "block_position": "Block position",
        "block_quaternion": "Block quaternion ($w/x/y/z$)",
        "block_yaw": "Block yaw",
        "end_effector_position": "End-effector position",
        "end_effector_yaw": "End-effector yaw",
        "gripper": "Gripper opening",
        "joint_position": "Joint position",
        "joint_velocity": "Joint velocity",
        "agent_position": "Agent position",
        "agent_location": "Agent location",
        "block_location": "Block location",
        "block_yaw": "Block yaw",
    }
    lines = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Quantity & Target / units & \multicolumn{4}{c}{LeWM} & \multicolumn{4}{c}{CoWM} \\",
        r" & & \multicolumn{2}{c}{Linear} & \multicolumn{2}{c}{MLP} & \multicolumn{2}{c}{Linear} & \multicolumn{2}{c}{MLP} \\",
        r" & & MSE$\downarrow$ & $r\uparrow$ & MSE$\downarrow$ & $r\uparrow$ & MSE$\downarrow$ & $r\uparrow$ & MSE$\downarrow$ & $r\uparrow$ \\",
        r"\midrule",
    ]
    for task in TASKS:
        summary = summaries[task]
        task_label = {"cube": "Cube (main)", "tworoom": "TwoRoom (auxiliary)", "pusht": "PushT (auxiliary)", "reacher": "Reacher (auxiliary)"}[task]
        lines.append(rf"\multicolumn{{10}}{{l}}{{\textbf{{{task_label}}}}} \\")
        for name, row in summary["attributes"].items():
            target = row["target"]
            field = _field_label(target)
            units = str(target.get("units", "unspecified"))
            display_name = labels.get(name, name.replace("_", " ").title())
            cells = (
                _tex_metric_cells(row["lewm"]["linear"])
                + _tex_metric_cells(row["lewm"]["mlp"])
                + _tex_metric_cells(row["cowm"]["linear"])
                + _tex_metric_cells(row["cowm"]["mlp"])
            )
            lines.append(
                rf"{display_name} & \texttt{{\detokenize{{{field}}}}} ({_tex_escape(units)}) & "
                + " & ".join(cells)
                + r" \\"
            )
        overall_cells: list[str] = []
        for model in ("lewm", "cowm"):
            for readout in ("linear", "mlp"):
                overall_cells.extend(_tex_metric_cells(summary["overall"][model][readout]))
        lines.extend(
            [
                rf"\textbf{{Overall}} & Equal weight over {len(summary['attributes'])} attributes & "
                + " & ".join(overall_cells)
                + r" \\",
                r"\addlinespace",
            ]
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return "\n".join(lines) + "\n"


def _probe_table(summaries: dict[str, dict[str, Any]], task: str) -> str:
    value = summaries[task]
    lines = [
        f"### {task.title()} observed physical readouts",
        "",
        "| Attribute | Target / units | LeWM Linear MSE / r | CoWM Linear MSE / r | LeWM MLP MSE / r | CoWM MLP MSE / r |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, row in value["attributes"].items():
        target = row["target"]
        units = target.get("units", "unspecified")
        field = _field_label(target)
        leo, cow = row["lewm"], row["cowm"]
        lines.append(
            f"| {name.replace('_', ' ').title()} | `{field}` ({units}) | "
            f"{_format(leo['linear'], 'standardized_mse')} / {_format(leo['linear'], 'pearson_r')} | "
            f"{_format(cow['linear'], 'standardized_mse')} / {_format(cow['linear'], 'pearson_r')} | "
            f"{_format(leo['mlp'], 'standardized_mse')} / {_format(leo['mlp'], 'pearson_r')} | "
            f"{_format(cow['mlp'], 'standardized_mse')} / {_format(cow['mlp'], 'pearson_r')} |"
        )
    lines.append(
        f"| **Overall (equal attribute weight)** | {len(value['attributes'])} attributes | "
        f"{_format(value['overall']['lewm']['linear'], 'standardized_mse')} / {_format(value['overall']['lewm']['linear'], 'pearson_r')} | "
        f"{_format(value['overall']['cowm']['linear'], 'standardized_mse')} / {_format(value['overall']['cowm']['linear'], 'pearson_r')} | "
        f"{_format(value['overall']['lewm']['mlp'], 'standardized_mse')} / {_format(value['overall']['lewm']['mlp'], 'pearson_r')} | "
        f"{_format(value['overall']['cowm']['mlp'], 'standardized_mse')} / {_format(value['overall']['cowm']['mlp'], 'pearson_r')} |"
    )
    lines.extend(
        [
            "",
            f"Readout test set: {value['data']['test_episode_count']} episodes, {value['data']['row_split_counts'].get('test', 0)} frames; sampled total {value['data']['sampled_frames']} frames.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(PROBE_ROOT / "aggregate_metrics.json"))
    parser.add_argument("--markdown", default=str(PROBE_ROOT / "aggregate_tables.md"))
    parser.add_argument("--tex", default=str(ROOT / "paper/tables/final_probe.tex"))
    args = parser.parse_args()
    summaries = {task: rebuild_probe_summary(task) for task in TASKS}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = "\n".join(_probe_table(summaries, task) for task in TASKS)
    Path(args.markdown).write_text(markdown + "\n", encoding="utf-8")
    Path(args.tex).write_text(_probe_tex(summaries), encoding="utf-8")
    print(json.dumps({"status": "ok", "tasks": TASKS, "summary": str(output), "tables": str(args.markdown), "tex": str(args.tex)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
