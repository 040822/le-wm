#!/usr/bin/env python3
"""Run and analyze the Round 4 Phase 4.5-3 dev-only matrix.

One process owns one task and one explicitly selected physical GPU.  The
script is intentionally resumable: a complete condition is reused, while a
partially written condition is treated as an error rather than overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.checkpoint import load_policy_or_model
from source.common.eval import EvaluationIdentity, compose_eval_config
from source.common.round3_eval import run_round3_evaluation
from source.common.round3_phase1 import CohortManifest, wilson_interval
from source.common.round4_eval import run_round4_evaluation, validate_gpu_visibility
from source.common.round4_phase45_3 import (
    PHASE45_CEM_PROTOCOLS,
    PHASE45_FLOW_STEPS,
    PHASE45_INTEGRATORS,
    PHASE45_NON_CEM_PROTOCOL,
    PHASE45_TASKS,
    analyze_phase45_results,
    condition_key,
    load_phase45_results,
)


DEFAULT_CONFIG = ROOT / "config" / "round4" / "phase45_3.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "round4" / "phase45_3_seed3072_dev"
DEFAULT_REPORT = ROOT / "docs" / "report" / "round4" / "round4_phase45-3_flow_step_report.md"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _gpu(value: str | None) -> str | None:
    if value is None:
        return None
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(values) != 1 or not values[0].isdigit() or int(values[0]) not in range(8):
        raise argparse.ArgumentTypeError(
            "--gpu must select exactly one physical GPU0-7"
        )
    return values[0]


def _configure_device(device: str, gpu: str | None) -> None:
    if str(device).startswith("cuda"):
        if gpu is None:
            raise ValueError("CUDA execution requires --gpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        validate_gpu_visibility(device)


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Phase 4.5-3 config must be an object: {path}")
    return value


def _conditions() -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    for mode in ("P0", "P3"):
        for step in PHASE45_FLOW_STEPS:
            for integrator in PHASE45_INTEGRATORS:
                conditions.append(
                    {
                        "mode": mode,
                        "cem_protocol": PHASE45_NON_CEM_PROTOCOL,
                        "action_flow_steps": int(step),
                        "action_flow_integrator": integrator,
                    }
                )
    for protocol in PHASE45_CEM_PROTOCOLS:
        conditions.append(
            {
                "mode": "P1",
                "cem_protocol": protocol,
                "action_flow_steps": None,
                "action_flow_integrator": "not_applicable",
            }
        )
        for step in PHASE45_FLOW_STEPS:
            for integrator in PHASE45_INTEGRATORS:
                conditions.append(
                    {
                        "mode": "P2",
                        "cem_protocol": protocol,
                        "action_flow_steps": int(step),
                        "action_flow_integrator": integrator,
                    }
                )
    return conditions


def _select_conditions(
    conditions: Sequence[Mapping[str, Any]],
    indices: Sequence[int] | None,
) -> list[Mapping[str, Any]]:
    if indices is None:
        return list(conditions)
    normalized = [int(index) for index in indices]
    if len(set(normalized)) != len(normalized):
        raise ValueError("--condition-index values must be unique")
    invalid = [index for index in normalized if index < 0 or index >= len(conditions)]
    if invalid:
        raise ValueError(
            f"condition indices out of range: {invalid}; valid range is 0..{len(conditions) - 1}"
        )
    return [conditions[index] for index in normalized]


def _condition_name(condition: Mapping[str, Any]) -> str:
    step = (
        "invariant"
        if condition["action_flow_steps"] is None
        else f"step_{int(condition['action_flow_steps'])}"
    )
    integrator = str(condition["action_flow_integrator"])
    return "/".join(
        (
            str(condition["mode"]),
            str(condition["cem_protocol"]),
            step,
            integrator,
            "dev",
        )
    )


def _condition_dir(root: Path, task: str, condition: Mapping[str, Any]) -> Path:
    return root / "conditions" / task / _condition_name(condition)


def _compose(task: str, manifest: CohortManifest, device: str):
    cfg = compose_eval_config(
        task,
        overrides=[
            f"eval.num_eval={len(manifest.entries)}",
            "output.save_video=false",
            f"solver.device={device}",
            "solver.num_samples=300",
            "solver.topk=30",
            "solver.n_steps=30",
            "solver.var_scale=1.0",
        ],
    )
    cfg.solver.device = device
    return cfg


def _expected_condition_key(task: str, condition: Mapping[str, Any]):
    return (
        task,
        str(condition["mode"]),
        str(condition["cem_protocol"]),
        condition["action_flow_steps"],
        str(condition["action_flow_integrator"]),
    )


def _reuse_or_raise(target: Path, expected: tuple[Any, ...]) -> dict[str, Any] | None:
    result_path = target / "result.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        observed = condition_key(payload)
        if observed != expected:
            raise ValueError(
                f"existing result has condition {observed}, expected {expected}: "
                f"{result_path}"
            )
        if payload.get("status") != "ok" or len(payload.get("episodes", [])) != 50:
            raise ValueError(f"existing result is incomplete: {result_path}")
        return payload
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(
            f"condition directory is partially populated; refusing overwrite: {target}"
        )
    return None


def _run_fast_condition(
    *,
    task: str,
    condition: Mapping[str, Any],
    config: Mapping[str, Any],
    output_root: Path,
    device: str,
    model: Any,
    manifest: CohortManifest,
) -> dict[str, Any]:
    target = _condition_dir(output_root, task, condition)
    expected = _expected_condition_key(task, condition)
    reused = _reuse_or_raise(target, expected)
    if reused is not None:
        print(json.dumps({"status": "reused", "result": str(target / "result.json")}))
        return reused
    cfg = _compose(task, manifest, device)
    identity = EvaluationIdentity(
        entrypoint="round4_phase45_3",
        policy_kind="round4_shared_dit",
        checkpoint=str(
            _resolve(config["training"]["checkpoints"][task]).resolve()
        ),
        epoch=int(config["training"]["epoch"]),
        stage=str(condition["mode"]),
    )
    payload = run_round4_evaluation(
        cfg,
        task=task,
        policy_or_model=model,
        mode=str(condition["mode"]),
        identity=identity,
        manifest=manifest,
        output_dir=target,
        trace_output_dir=target / "trace",
        device=device,
        trace=True,
        candidate_count=64,
        flow_steps=16,
        action_flow_steps=condition["action_flow_steps"],
        solver_batch_size=1,
        candidate_batch_size=64,
        actor_warm_start_scale=1.0,
        action_flow_integrator=(
            "euler"
            if condition["action_flow_integrator"] == "not_applicable"
            else condition["action_flow_integrator"]
        ),
        cem_protocol=str(condition["cem_protocol"]),
    )
    observed = condition_key(payload)
    if observed != expected:
        raise ValueError(f"new result condition {observed} != expected {expected}")
    print(json.dumps({"status": "ok", "result": str(target / "result.json")}))
    return payload


def _run_lewm(
    *,
    task: str,
    config: Mapping[str, Any],
    output_root: Path,
    device: str,
    manifest: CohortManifest,
) -> dict[str, Any]:
    target = output_root / "lewm" / task / "dev"
    result_path = target / "result.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok" or len(payload.get("episodes", [])) != 50:
            raise ValueError(f"existing LeWM result is incomplete: {result_path}")
        return payload
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"LeWM output directory is partially populated: {target}")
    checkpoint = _resolve(config["lewm"]["checkpoints"][task]).resolve()
    policy_or_model, resolved = load_policy_or_model(str(checkpoint))
    cfg = _compose(task, manifest, device)
    identity = EvaluationIdentity(
        entrypoint="round4_phase45_3",
        policy_kind="lewm_reference",
        checkpoint=str(resolved or checkpoint),
        epoch=int(config["lewm"]["epoch"]),
    )
    payload = run_round3_evaluation(
        cfg,
        task=task,
        policy_or_model=policy_or_model,
        identity=identity,
        manifest=manifest,
        output_dir=target,
        trace_output_dir=target / "trace",
        device=device,
        trace=True,
        planning_timing=True,
    )
    return payload


def run_task(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    task = str(args.task)
    if task not in PHASE45_TASKS:
        raise ValueError(f"unknown task {task!r}")
    manifest = CohortManifest.load(_resolve(config["cohort"]["paths"][task]))
    if manifest.cohort_kind != "dev" or len(manifest.entries) != 50:
        raise ValueError(f"Phase 4.5-3 requires a 50-episode dev cohort: {task}")
    checkpoint = _resolve(config["training"]["checkpoints"][task])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model, resolved = load_policy_or_model(str(checkpoint))
    if resolved is not None and Path(resolved).resolve() != checkpoint.resolve():
        raise ValueError(f"checkpoint resolver changed the requested path: {checkpoint}")
    _configure_device(args.device, args.gpu)
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    conditions = _select_conditions(_conditions(), args.condition_index)
    for condition in conditions:
        _run_fast_condition(
            task=task,
            condition=condition,
            config=config,
            output_root=output_root,
            device=args.device,
            model=model,
            manifest=manifest,
        )
    if not args.skip_lewm:
        # Release the Fast-LeWAM module before loading the independent LeWM
        # reference in the same task process, especially on GPU0 where other
        # workloads may already be resident.
        del model
        if str(args.device).startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
        _run_lewm(
            task=task,
            config=config,
            output_root=output_root,
            device=args.device,
            manifest=manifest,
        )


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _result_payload(indexed, key):
    item = indexed.get(key)
    return None if item is None else item["payload"]


def _table_rows(indexed, task: str, mode: str, protocol: str, integrator: str):
    rows = []
    for step in PHASE45_FLOW_STEPS:
        key = (task, mode, protocol, step, integrator)
        item = indexed.get(key)
        if item is not None:
            rows.append(item["row"])
    return rows


def _render_report(
    analysis: Mapping[str, Any],
    *,
    config_path: Path,
    output_root: Path,
    report_path: Path,
    include_heun: bool = True,
) -> str:
    all_rows = analysis["rows"]
    rows = [
        row
        for row in all_rows
        if include_heun or row["action_flow_integrator"] != "heun"
    ]
    indexed = {
        (
            row["task"],
            row["mode"],
            row["cem_protocol"],
            row["action_flow_steps"],
            row["action_flow_integrator"],
        ): {"row": row, "payload": None}
        for row in rows
    }
    title = "# Round 4 Phase 4.5-3：CEM-clip、candidate-scale、flow step 与 Heun 验证"
    if not include_heun:
        title = "# Round 4 Phase 4.5-3：CEM-clip、candidate-scale、flow step（without Heun）"
    condition_summary = (
        f"- 条件数：{len(rows)}；预期：{analysis['validation'].get('expected_count', '—')}；缺失：{analysis['validation'].get('missing_count', '—')}"
        if include_heun
        else f"- 原始分析条件数：{len(all_rows)}；本报告展示 Euler 条件：{len(rows)}；Heun 条件已排除；原始分析预期：{analysis['validation'].get('expected_count', '—')}"
    )
    lines = [
        title,
        "",
        "## 实验范围",
        "",
        "本报告基于 R4-AB、训练 seed 3072、epoch 10 权重，在 `round3_revised` 四任务 dev cohort（每任务 50 episodes）上完成。未重新训练，未运行 final 集；P0/P3 的 CEM metadata 使用 `not_applicable`。所有 paired 比较保持 episode identity、start index 和 cohort hash 一致。",
        "",
        f"- 配置：`{config_path}`",
        f"- 输出根目录：`{output_root}`",
        f"- 完整分析：`{output_root / 'analysis' / 'analysis.json'}`",
        condition_summary,
        "",
        "## 验收",
        "",
        f"`status=ok`：{analysis['validation'].get('all_status_ok')}；仅 dev：{analysis['validation'].get('dev_only')}。clip/scale 的 projected candidate violation 应为 0；该条件已由结果分析和 runner 的投影断言共同检查。"
        + (" 本版本只展示 Euler，不对 Heun 做结论；含 Heun 的原报告保持不变。" if not include_heun else ""),
        "",
        "## 成功率与 timing 摘要",
        "",
    ]

    def append_condition_table(title: str, selected: Sequence[Mapping[str, Any]]) -> None:
        lines.extend(
            [
                title,
                "",
                "| mode/protocol | integrator | step | success | Wilson 95% | planning mean/median/P95 (s) | normalized true violation | physical violation |",
                "|---|---|---:|---:|---|---:|---:|---:|",
            ]
        )
        ordered = sorted(
            selected,
            key=lambda row: (
                row["mode"],
                row["cem_protocol"],
                row["action_flow_integrator"],
                -1 if row["action_flow_steps"] is None else row["action_flow_steps"],
            ),
        )
        for row in ordered:
            step = "inv" if row["action_flow_steps"] is None else row["action_flow_steps"]
            lines.append(
                "| {mode}/{cem_protocol} | {integrator} | {step} | {successes}/50 ({rate:.1f}%) | [{lo:.1f}, {hi:.1f}]% | {mean}/{median}/{p95} | {norm} | {physical} |".format(
                    mode=row["mode"],
                    cem_protocol=row["cem_protocol"],
                    integrator=row["action_flow_integrator"],
                    step=step,
                    successes=row["successes"],
                    rate=row["success_rate_percent"],
                    lo=row["wilson_95_percent_low"],
                    hi=row["wilson_95_percent_high"],
                    mean=_fmt(row["planning_mean_seconds"]),
                    median=_fmt(row["planning_median_seconds"]),
                    p95=_fmt(row["planning_p95_seconds"]),
                    norm=_fmt(row["normalized_true_bound_violation_fraction"]),
                    physical=_fmt(row["physical_action_violation_fraction"]),
                )
            )
        lines.append("")

    def condition_cell(row: Mapping[str, Any] | None) -> str:
        if row is None:
            return "—"
        return (
            f"{row['successes']}/50 ({row['success_rate_percent']:.1f}%)<br>"
            f"CI [{row['wilson_95_percent_low']:.1f}, {row['wilson_95_percent_high']:.1f}]%<br>"
            f"t={_fmt(row['planning_mean_seconds'])}/{_fmt(row['planning_median_seconds'])}/{_fmt(row['planning_p95_seconds'])} s<br>"
            f"v={_fmt(row['normalized_true_bound_violation_fraction'])}/{_fmt(row['physical_action_violation_fraction'])}"
        )

    for task in PHASE45_TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        task_index = {
            (
                row["mode"],
                row["cem_protocol"],
                row["action_flow_steps"],
                row["action_flow_integrator"],
            ): row
            for row in task_rows
        }
        lines.extend(
            [
                f"### {task}",
                "",
                (
                    "以下先保留原来的完整摘要表，再给出三个互补的横向子表。子表单元格依次包含 success、Wilson 95% CI、planning mean/median/P95（秒）和 normalized/physical violation；`—` 表示该组合不适用。完整原始字段仍保存在 `conditions.csv` 与 `analysis.json`。"
                    if include_heun
                    else "以下保留原完整摘要表的 Euler 视图，并给出不含 Heun 的 `step` 与 `mode/protocol` 横向子表；Heun 子表在本版本中省略。子表单元格依次包含 success、Wilson 95% CI、planning mean/median/P95（秒）和 normalized/physical violation；含 Heun 的原报告保持不变。"
                ),
                "",
            ]
        )
        append_condition_table(
            "#### 完整摘要表（原表）",
            [
                row
                for row in task_rows
                if row["mode"] == "P1"
                or row["action_flow_steps"] == 16
                or (
                    row["mode"] in {"P0", "P3"}
                    and row["action_flow_integrator"] == "euler"
                )
            ],
        )

        lines.extend(
            [
                "#### `step` 子表（固定 mode/protocol + integrator，横向比较全部 step）",
                "",
                "| mode/protocol | integrator | step=1 | step=2 | step=5 | step=10 | step=16 | step=32 |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for mode, protocols in (
            ("P0", (PHASE45_NON_CEM_PROTOCOL,)),
            ("P2", PHASE45_CEM_PROTOCOLS),
            ("P3", (PHASE45_NON_CEM_PROTOCOL,)),
        ):
            for protocol in protocols:
                integrators = PHASE45_INTEGRATORS if include_heun else ("euler",)
                for integrator in integrators:
                    lines.append(
                        "| {mode}/{protocol} | {integrator} | {s1} | {s2} | {s5} | {s10} | {s16} | {s32} |".format(
                            mode=mode,
                            protocol=protocol,
                            integrator=integrator,
                            s1=condition_cell(task_index.get((mode, protocol, 1, integrator))),
                            s2=condition_cell(task_index.get((mode, protocol, 2, integrator))),
                            s5=condition_cell(task_index.get((mode, protocol, 5, integrator))),
                            s10=condition_cell(task_index.get((mode, protocol, 10, integrator))),
                            s16=condition_cell(task_index.get((mode, protocol, 16, integrator))),
                            s32=condition_cell(task_index.get((mode, protocol, 32, integrator))),
                        )
                    )
        lines.append("")

        if include_heun:
            lines.extend(
                [
                    "#### `heun` 子表（固定 mode/protocol + step，横向比较 Euler/Heun）",
                    "",
                    "| mode/protocol | step | euler | heun |",
                    "|---|---:|---|---|",
                ]
            )
            for mode, protocols in (
                ("P0", (PHASE45_NON_CEM_PROTOCOL,)),
                ("P2", PHASE45_CEM_PROTOCOLS),
                ("P3", (PHASE45_NON_CEM_PROTOCOL,)),
            ):
                for protocol in protocols:
                    for step in PHASE45_FLOW_STEPS:
                        lines.append(
                            "| {mode}/{protocol} | {step} | {euler} | {heun} |".format(
                                mode=mode,
                                protocol=protocol,
                                step=step,
                                euler=condition_cell(task_index.get((mode, protocol, step, "euler"))),
                                heun=condition_cell(task_index.get((mode, protocol, step, "heun"))),
                            )
                        )
            lines.append("")

        mode_protocol_columns = (
            ("P0", PHASE45_NON_CEM_PROTOCOL),
            ("P1", "legacy"),
            ("P1", "cem-clip"),
            ("P1", "cem-scale"),
            ("P2", "legacy"),
            ("P2", "cem-clip"),
            ("P2", "cem-scale"),
            ("P3", PHASE45_NON_CEM_PROTOCOL),
        )
        column_labels = [f"{mode}/{protocol}" for mode, protocol in mode_protocol_columns]
        lines.extend(
            [
                "#### `mode/protocol` 子表（固定 step + integrator，横向比较全部 mode/protocol）",
                "",
                "| step/integrator | " + " | ".join(column_labels) + " |",
                "|---|" + "---|" * len(mode_protocol_columns),
            ]
        )
        mode_protocol_rows = [("inv", "not_applicable")]
        mode_protocol_rows.extend(
            (str(step), integrator)
            for step in PHASE45_FLOW_STEPS
            for integrator in (PHASE45_INTEGRATORS if include_heun else ("euler",))
        )
        for step_label, integrator in mode_protocol_rows:
            if step_label == "inv":
                lookup = [
                    task_index.get((mode, protocol, None, "not_applicable"))
                    for mode, protocol in mode_protocol_columns
                ]
            else:
                step = int(step_label)
                lookup = [
                    task_index.get((mode, protocol, step, integrator))
                    for mode, protocol in mode_protocol_columns
                ]
            lines.append(
                "| {step}/{integrator} | ".format(
                    step=step_label,
                    integrator=integrator,
                )
                + " | ".join(condition_cell(row) for row in lookup)
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## 关键 paired 比较",
            "",
            "所有数字均为 treatment 相对 baseline；`improved/regressed/net` 是 episode-level paired 结果，p 值为双侧 exact McNemar。没有自动通过阈值，也没有根据 dev 结果选择最佳 step。",
            "",
            "| category | comparison | Δ success (pp) | improved | regressed | net | p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    comparisons = [
        item
        for item in analysis["comparisons"]
        if include_heun
        or (
            item["category"] != "euler_vs_heun"
            and "/heun/" not in str(item.get("label", ""))
        )
    ]
    key_comparisons = [
        item
        for item in comparisons
        if item["category"] in {"P1_protocol", "P2_protocol"}
        and ("step_16" in item["label"] or item["category"] == "P1_protocol")
    ]
    for item in key_comparisons:
        paired = item["paired"]
        lines.append(
            "| {category} | `{label}` | {delta:.1f} | {improved} | {regressed} | {net} | {p} |".format(
                category=item["category"],
                label=item["label"],
                delta=item["delta_pp"],
                improved=paired["improved"],
                regressed=paired["regressed"],
                net=item["net_success_delta"],
                p=_fmt(paired["mcnemar_exact_two_sided_p"], 6),
            )
        )
    lines.append("")

    comparison_categories = (
        ("P2_step_vs_step16", "euler_vs_heun")
        if include_heun
        else ("P2_step_vs_step16",)
    )
    for category in comparison_categories:
        selected = [item for item in comparisons if item["category"] == category]
        deltas = [float(item["delta_pp"]) for item in selected]
        if deltas:
            lines.append(
                f"- `{category}` 共 {len(deltas)} 个 paired comparison；逐任务/逐条件结果保存在 analysis JSON，Δ success 范围为 {_fmt(min(deltas), 1)} 到 {_fmt(max(deltas), 1)} pp。"
            )
    lines.extend(
        [
            "",
            "## 边界与 projection 统计",
            "",
            "真实 normalized 越界按逐坐标 `low - 1e-6` / `high + 1e-6` 判断；`abs(action_norm)>1` 只作为 legacy proxy。每个结果同时保存 physical bounds、normalizer mean/scale、完整 normalized bounds、raw/projected candidate、warm-start、scale factor 分布以及最终执行动作统计。",
            "",
            "| task | protocol | step/integrator | raw candidate violation | projected candidate violation | candidate changed | warm-start before/after | scale median |",
            "|---|---|---|---:|---:|---:|---|---:|",
        ]
    )
    for task in PHASE45_TASKS:
        for protocol in ("cem-clip", "cem-scale"):
            row = next(
                (
                    item
                    for item in rows
                    if item["task"] == task
                    and item["mode"] == "P2"
                    and item["cem_protocol"] == protocol
                    and item["action_flow_steps"] == 16
                    and item["action_flow_integrator"] == "euler"
                ),
                None,
            )
            if row is not None:
                lines.append(
                    "| {task} | {protocol} | P2 step16 Euler | {raw} | {projected} | {changed} | {before}/{after} | {scale} |".format(
                        task=task,
                        protocol=protocol,
                        raw=_fmt(row["raw_candidate_violation_fraction"]),
                        projected=_fmt(row["projected_candidate_violation_fraction"]),
                        changed=_fmt(row["candidate_changed_fraction"]),
                        before=_fmt(row["warm_start_before_violation_fraction"]),
                        after=_fmt(row["warm_start_after_violation_fraction"]),
                        scale=_fmt(row["scale_factor_median"]),
                    )
                )
    lines.extend(
        [
            "",
            "## LeWM latency reference",
            "",
            "LeWM 的 epoch10 checkpoint 在同一 dev cohort 和统一 300/30/30 CEM budget 下重新计时。这里报告的是 planning latency，不把历史 `evaluation_seconds` 直接当作 planning latency。",
            "",
            "| task | success | Wilson 95% | planning mean/median/P95 (s) | forward count | peak memory | result |",
            "|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in analysis.get("lewm", []):
        lines.append(
            "| {task} | {successes}/{episodes} ({rate:.1f}%) | [{lo:.1f}, {hi:.1f}]% | {mean}/{median}/{p95} | {forward} | {memory} | `{path}` |".format(
                task=row.get("task"),
                successes=row.get("successes", "—"),
                episodes=row.get("episodes", "—"),
                rate=row.get("success_rate_percent", float("nan")),
                lo=row.get("wilson_95_percent_low", float("nan")),
                hi=row.get("wilson_95_percent_high", float("nan")),
                mean=_fmt(row.get("planning_mean_seconds")),
                median=_fmt(row.get("planning_median_seconds")),
                p95=_fmt(row.get("planning_p95_seconds")),
                forward=row.get("forward_count", "—"),
                memory=row.get("peak_memory_bytes", "—"),
                path=row.get("path", "—"),
            )
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            (
                "本轮只支持在固定 R4-AB checkpoint、单训练 seed、固定 dev cohort 上做逐任务描述性结论。flow step、Euler/Heun、candidate-clip 和 candidate-scale 的结果不外推到 final，也不以跨任务平均成功率替代逐任务结论；若 paired discordance 很少，结论标记为证据不足。"
                if include_heun
                else "本报告只支持在固定 R4-AB checkpoint、单训练 seed、固定 dev cohort 上对 Euler 条件做逐任务描述性结论；不包含 Heun 结果，也不将 dev 结果外推到 final，不以跨任务平均成功率替代逐任务结论。含 Heun 的原报告保持不变。"
            ),
            "",
            f"- 报告生成路径：`{report_path}`",
            f"- 代码 commit（生成时）：`{_git_commit() or 'unknown'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    output_root = _resolve(args.output_root)
    manifests = {
        task: CohortManifest.load(_resolve(config["cohort"]["paths"][task]))
        for task in PHASE45_TASKS
    }
    checkpoints = {
        task: str(_resolve(config["training"]["checkpoints"][task]).resolve())
        for task in PHASE45_TASKS
    }
    indexed, validation = load_phase45_results(
        output_root,
        manifests=manifests,
        checkpoints=checkpoints,
        require_complete=True,
    )
    lewm_rows = []
    for task in PHASE45_TASKS:
        path = output_root / "lewm" / task / "dev" / "result.json"
        if not path.is_file():
            if not args.skip_lewm:
                raise FileNotFoundError(path)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok" or len(payload.get("episodes", [])) != 50:
            raise ValueError(f"invalid LeWM result: {path}")
        successes = [bool(item.get("success", False)) for item in payload["episodes"]]
        low, high = wilson_interval(successes)
        planning = payload.get("round4_planning", {})
        lewm_rows.append(
            {
                "task": task,
                "method": "LeWM",
                "episodes": len(successes),
                "successes": int(sum(successes)),
                "success_rate": float(np.mean(successes)),
                "success_rate_percent": float(np.mean(successes) * 100.0),
                "wilson_95_percent_low": float(low * 100.0),
                "wilson_95_percent_high": float(high * 100.0),
                "planning_mean_seconds": planning.get("planning_mean_seconds"),
                "planning_median_seconds": planning.get("planning_median_seconds"),
                "planning_p95_seconds": planning.get("planning_p95_seconds"),
                "forward_count": planning.get("forward_count"),
                "peak_memory_bytes": planning.get("peak_memory_bytes"),
                "path": str(path),
            }
        )
    analysis = analyze_phase45_results(
        indexed,
        lewm_rows=lewm_rows,
        validation=validation,
    )
    analysis["experiment"] = "Round 4 Phase 4.5-3"
    analysis["config"] = str(_resolve(args.config).resolve())
    analysis["config_sha256"] = _sha256_file(_resolve(args.config))
    analysis["output_root"] = str(output_root.resolve())
    analysis["code_commit"] = _git_commit()
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "analysis.json").write_text(
        json.dumps(_jsonable(analysis), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    fields = list(analysis["rows"][0].keys())
    with (analysis_dir / "conditions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in analysis["rows"]:
            writer.writerow({key: _jsonable(row.get(key)) for key in fields})
    comparison_fields = (
        "category",
        "label",
        "delta_pp",
        "net_success_delta",
        "baseline_success_rate_percent",
        "treatment_success_rate_percent",
        "paired",
    )
    with (analysis_dir / "comparisons.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_fields)
        writer.writeheader()
        for row in analysis["comparisons"]:
            writer.writerow({key: _jsonable(row.get(key)) for key in comparison_fields})
    report_path = _resolve(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(
            analysis,
            config_path=_resolve(args.config).resolve(),
            output_root=output_root.resolve(),
            report_path=report_path.resolve(),
            include_heun=not args.without_heun,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "analysis": str(analysis_dir / "analysis.json"),
                "conditions": len(analysis["rows"]),
                "comparisons": len(analysis["comparisons"]),
                "report": str(report_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT))
    parser.add_argument("--task", choices=(*PHASE45_TASKS, "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=_gpu, help="one physical GPU0-7")
    parser.add_argument(
        "--condition-index",
        action="append",
        type=int,
        help="run only the specified condition index; repeat for multiple indices",
    )
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument(
        "--without-heun",
        action="store_true",
        help="render a report that excludes Heun conditions and comparisons",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-lewm", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    if args.dry_run:
        conditions = _select_conditions(_conditions(), args.condition_index)
        tasks = PHASE45_TASKS if args.task == "all" else (args.task,)
        print(
            json.dumps(
                {
                    "tasks": list(tasks),
                    "conditions_per_task": len(conditions),
                    "fast_conditions_total": len(conditions) * len(tasks),
                    "lewm": not args.skip_lewm,
                    "output_root": str(_resolve(args.output_root)),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    if not args.analyze_only:
        tasks = PHASE45_TASKS if args.task == "all" else (args.task,)
        for task in tasks:
            task_args = argparse.Namespace(**vars(args))
            task_args.task = task
            run_task(task_args, config)
    if args.analyze_only or args.task == "all":
        analyze(args, config)


if __name__ == "__main__":
    main()
