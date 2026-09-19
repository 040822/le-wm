#!/usr/bin/env python3
"""Run and analyze Round 5 Phase 1 guidance conditions on the legacy cohort.

The no-guidance P0--P3 rows are the canonical Phase 4.5-4 legacy artifacts and
are referenced read-only.  This entrypoint only writes the new guided-flow and
post-opt conditions into its own output root.
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
from source.common.round3_phase1 import CohortManifest, wilson_interval
from source.common.round4_eval import (
    run_round4_evaluation,
    validate_gpu_visibility,
)
from source.common.round5_phase1 import (
    ROUND5_FLOW_STEPS,
    ROUND5_P3_GUIDANCE_MODES,
    ROUND5_STEP_REFERENCE,
    ROUND5_TASKS,
    analyze_round5_results,
    baseline_condition_specs,
    condition_key,
    condition_key_from_spec,
    condition_name,
    guidance_condition_specs,
    load_round5_results,
    p2_protocol,
)

DEFAULT_CONFIG = ROOT / "config" / "round5" / "phase1.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "round5" / "phase1_seed3072_legacy"
DEFAULT_REPORT = ROOT / "docs" / "report" / "round5" / "round5_phase1_report.md"
DEFAULT_MIN_FREE_MIB = 3500


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
        raise argparse.ArgumentTypeError("--gpu must select exactly one physical GPU0-7")
    return values[0]


def _gpu_preflight(gpu: str, *, minimum_free_mib: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [
        "nvidia-smi",
        "--id",
        str(gpu),
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect GPU{gpu} before Round 5 worker startup") from exc
    line = next((item.strip() for item in completed.stdout.splitlines() if item.strip()), "")
    fields = [item.strip() for item in line.split(",")]
    if len(fields) < 6:
        raise RuntimeError(f"nvidia-smi returned an invalid GPU{gpu} snapshot: {line!r}")
    snapshot = {
        "gpu": int(gpu),
        "reported_index": fields[0],
        "name": fields[1],
        "memory_total_mib": int(fields[2]),
        "memory_used_mib": int(fields[3]),
        "memory_free_mib": int(fields[4]),
        "utilization_percent": int(fields[5]),
        "minimum_free_mib": int(minimum_free_mib),
    }
    if snapshot["memory_free_mib"] < int(minimum_free_mib):
        raise RuntimeError(
            f"GPU{gpu} has only {snapshot['memory_free_mib']} MiB free; "
            f"need at least {minimum_free_mib} MiB"
        )
    print(json.dumps({"gpu_preflight": snapshot}, ensure_ascii=False, sort_keys=True))
    return snapshot


def _configure_device(device: str, gpu: str | None, *, minimum_free_mib: int) -> None:
    if str(device).startswith("cuda"):
        if gpu is None:
            raise ValueError("CUDA execution requires --gpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        validate_gpu_visibility(device)
        _gpu_preflight(gpu, minimum_free_mib=minimum_free_mib)


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Round 5 Phase 1 config must be an object: {path}")
    return value


def _select_conditions(
    conditions: Sequence[Mapping[str, Any]],
    indices: Sequence[int] | None,
    *,
    task: str,
    shard_index: int,
    num_shards: int,
) -> list[Mapping[str, Any]]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    if indices is not None:
        normalized = [int(index) for index in indices]
        if len(set(normalized)) != len(normalized):
            raise ValueError("--condition-index values must be unique")
        invalid = [index for index in normalized if index < 0 or index >= len(conditions)]
        if invalid:
            raise ValueError(
                f"condition indices out of range: {invalid}; "
                f"valid range is 0..{len(conditions) - 1}"
            )
        selected = [conditions[index] for index in normalized]
    else:
        selected = list(conditions)
    selected = [item for item in selected if str(item["task"]) == task]
    if num_shards > 1 and indices is None:
        selected = selected[shard_index::num_shards]
    return selected


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


def _condition_dir(root: Path, task: str, condition: Mapping[str, Any]) -> Path:
    return root / "conditions" / "legacy" / task / condition_name(condition)


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
    expected = condition_key_from_spec(condition)
    reused = _reuse_or_raise(target, expected)
    if reused is not None:
        print(json.dumps({"status": "reused", "result": str(target / "result.json")}))
        return reused
    guidance_cfg = config["guidance"]
    eval_cfg = config["evaluation"]
    cfg = _compose(task, manifest, device)
    identity = EvaluationIdentity(
        entrypoint="round5_phase1",
        policy_kind="round4_shared_dit",
        checkpoint=str(_resolve(config["training"]["checkpoints"][task]).resolve()),
        epoch=int(config["training"]["epoch"]),
        stage=str(condition["mode"]),
    )
    proposal_chunk = guidance_cfg.get("p3_proposal_chunk_size")
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
        candidate_count=int(eval_cfg["best_of_n_candidates"]),
        flow_steps=16,
        action_flow_steps=condition["action_flow_steps"],
        solver_batch_size=int(eval_cfg["solver_batch_size"]),
        candidate_batch_size=int(eval_cfg["candidate_batch_size"]),
        actor_warm_start_scale=1.0,
        action_flow_integrator=str(condition["action_flow_integrator"]),
        cem_protocol=str(condition["cem_protocol"]),
        guidance_mode=str(condition["guidance"]),
        guidance_step_size=float(guidance_cfg["step_size"]),
        guidance_last_steps=int(guidance_cfg["last_steps"]),
        guidance_inner_steps=int(guidance_cfg["inner_steps"]),
        guidance_max_rms_offset=float(guidance_cfg["max_rms_offset"]),
        proposal_chunk_size=(
            None if proposal_chunk is None else int(proposal_chunk)
        ),
        allowed_protocol_variants=("legacy",),
    )
    observed = condition_key(payload)
    if observed != expected:
        raise ValueError(f"new result condition {observed} != expected {expected}")
    recorded = str(payload["round4_planning"].get("guidance_mode", "none"))
    if recorded != str(condition["guidance"]):
        raise ValueError(
            f"result guidance_mode {recorded!r} != condition {condition['guidance']!r}"
        )
    print(json.dumps({"status": "ok", "result": str(target / "result.json")}))
    return payload


def run_task(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    task = str(args.task)
    if task not in ROUND5_TASKS:
        raise ValueError(f"unknown task {task!r}")
    legacy_paths = config["cohort"]["legacy"]["paths"]
    manifest = CohortManifest.load(_resolve(legacy_paths[task]))
    if (
        manifest.cohort_kind != "dev"
        or manifest.protocol_variant != "legacy"
        or len(manifest.entries) != 50
    ):
        raise ValueError(f"Round 5 Phase 1 requires canonical legacy_50 dev cohort: {task}")
    expected_sha = config["cohort"]["legacy"].get("sha256", {}).get(task)
    if expected_sha is not None and manifest.computed_sha256 != expected_sha:
        raise ValueError(
            f"legacy cohort hash changed for {task}: "
            f"{manifest.computed_sha256} != {expected_sha}"
        )
    checkpoint = _resolve(config["training"]["checkpoints"][task])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected_checkpoint_sha = config["training"].get("checkpoint_sha256", {}).get(task)
    if expected_checkpoint_sha is not None:
        observed = _sha256_file(checkpoint)
        if observed != expected_checkpoint_sha:
            raise ValueError(
                f"checkpoint hash changed for {task}: {observed} != {expected_checkpoint_sha}"
            )
    _configure_device(args.device, args.gpu, minimum_free_mib=int(args.min_free_mib))
    model, resolved = load_policy_or_model(str(checkpoint))
    if resolved is not None and Path(resolved).resolve() != checkpoint.resolve():
        raise ValueError(f"checkpoint resolver changed the requested path: {checkpoint}")
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selected = _select_conditions(
        guidance_condition_specs(),
        args.condition_index,
        task=task,
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
    )
    print(json.dumps({"task": task, "conditions": len(selected)}, sort_keys=True))
    for condition in selected:
        _run_fast_condition(
            task=task,
            condition=condition,
            config=config,
            output_root=output_root,
            device=args.device,
            model=model,
            manifest=manifest,
        )


def _load_manifests(config: Mapping[str, Any]) -> dict[str, CohortManifest]:
    legacy = config["cohort"]["legacy"]["paths"]
    return {task: CohortManifest.load(_resolve(legacy[task])) for task in ROUND5_TASKS}


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _cell(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return "—"
    return (
        f"{row['successes']}/50 ({row['success_rate_percent']:.1f}%)<br>"
        f"CI [{row['wilson_95_percent_low']:.1f}, {row['wilson_95_percent_high']:.1f}]%"
    )


def _render_report(
    analysis: Mapping[str, Any],
    *,
    config_path: Path,
    output_root: Path,
    report_path: Path,
) -> str:
    rows = list(analysis["rows"])
    comparisons = list(analysis["comparisons"])
    by_key = {
        (
            row["task"],
            row["mode"],
            row["cem_protocol"],
            row["action_flow_steps"],
            row["guidance"],
        ): row
        for row in rows
    }
    def _rate(task, mode, protocol, step, guidance):
        row = by_key.get((task, mode, protocol, step, guidance))
        return None if row is None else float(row["success_rate_percent"])

    def _two_step_cell(task, mode, protocol, guidance):
        first = _rate(task, mode, protocol, 1, guidance)
        second = _rate(task, mode, protocol, 2, guidance)
        if first is None and second is None:
            return "—"
        return f"{_fmt(first, 1)} / {_fmt(second, 1)}"

    summary_methods = (
        ("P0", "P0", "not_applicable", "none"),
        ("P1 (inv)", "P1", None, "none"),
        ("P2", "P2", None, "none"),
        ("P3", "P3", "not_applicable", "none"),
        ("P0+GF", "P0", "not_applicable", "guided_flow"),
        ("P0+PO", "P0", "not_applicable", "post_opt"),
        ("P2+GF", "P2", None, "guided_flow"),
        ("P2+PO", "P2", None, "post_opt"),
        ("P3+GF", "P3", "not_applicable", "guided_flow"),
        ("P3+PO", "P3", "not_applicable", "post_opt"),
        ("P3+refine", "P3", "not_applicable", "post_opt_refine"),
    )
    summary_lines = [
        "## 总表（step1 / step2 成功率 %）",
        "",
        "每格依次为 **step1 / step2** 的成功率（百分数）；`P1 (inv)` 不使用 action flow，"
        "对 step 不变，只列一个值。`P2` 在 reacher 使用 `cem-clip`，其余任务 `legacy`。",
        "GF = guided-flow，PO = post-opt，refine = 先 B 选优再 post-opt 精修。",
        "",
        "| 任务 | "
        + " | ".join(label for label, *_ in summary_methods)
        + " |",
        "|---|" + "|".join(["---"] * len(summary_methods)) + "|",
    ]
    for task in ROUND5_TASKS:
        p1_rate = _rate(task, "P1", p2_protocol(task), None, "none")
        p1_cell = "—" if p1_rate is None else f"{_fmt(p1_rate, 1)} (inv)"
        cells = []
        for label, mode, protocol, guidance in summary_methods:
            if mode == "P1":
                cells.append(p1_cell)
                continue
            resolved = p2_protocol(task) if protocol is None else protocol
            cells.append(_two_step_cell(task, mode, resolved, guidance))
        summary_lines.append(f"| {task} | " + " | ".join(cells) + " |")

    # Additional step=1-only summary table.  P1 is step-invariant; here it is
    # reported as a plain value without the "(inv)" annotation.
    summary_lines.extend(
        [
            "",
            "### step=1 总表（成功率 %）",
            "",
            "| 任务 | "
            + " | ".join(label.replace(" (inv)", "") for label, *_ in summary_methods)
            + " |",
            "|---|" + "|".join(["---"] * len(summary_methods)) + "|",
        ]
    )
    for task in ROUND5_TASKS:
        cells = []
        for label, mode, protocol, guidance in summary_methods:
            if mode == "P1":
                cells.append(_fmt(_rate(task, "P1", p2_protocol(task), None, "none"), 1))
                continue
            resolved = p2_protocol(task) if protocol is None else protocol
            cells.append(_fmt(_rate(task, mode, resolved, 1, guidance), 1))
        summary_lines.append(f"| {task} | " + " | ".join(cells) + " |")
    summary_lines.append("")

    lines = [
        "# Round 5 Phase 1：R4-AB guidance 与 P3 结合实验",
        "",
        *summary_lines,
        "## 实验范围",
        "",
        "本轮在 R4-AB seed 3072 epoch 10 的四个任务上，使用 canonical `legacy_50` "
        "旧协议 cohort（LeWM 上游口径，50 episodes）。P0/P1/P2/P3 的无 guidance "
        "结果直接引用 Phase 4.5-4；本轮新增 guided-flow 与 post-opt 条件。"
        "flow step 网格为 1/2/5/10/16/32，主要落点为 step 1/2。",
        "",
        f"- 配置：`{config_path}`",
        f"- 输出根目录：`{output_root}`",
        f"- baseline 来源：`{analysis['validation'].get('baseline_root', '—')}`（只读引用）",
        f"- 分析 JSON：`{output_root / 'analysis' / 'analysis.json'}`",
        "",
        "## 验收",
        "",
        f"- 条件总数：{analysis['validation'].get('result_count')} / "
        f"{analysis['validation'].get('expected_count')}；"
        f"baseline={analysis['validation'].get('baseline_result_count')} / "
        f"{analysis['validation'].get('baseline_expected_count')}，"
        f"new={analysis['validation'].get('new_result_count')} / "
        f"{analysis['validation'].get('new_expected_count')}。",
        f"- `status=ok`：{analysis['validation'].get('all_status_ok')}；每条件 50 episodes。",
        "",
        "## 条件定义",
        "",
        "guidance 条件：`late_steps=5, inner_steps=5, step_size=0.01, "
        "max_rms_offset=0.2`；`flow_steps` 等于该条件的 action flow step。",
        "P3 语义：`guided_flow`/`post_opt` 对每条候选生效；`post_opt_refine` "
        "先生成普通候选并由 B 选优，再对选中动作做 post-opt。",
        "P2 中 reacher 使用 cem-clip，其余任务 legacy；P0/P3 不使用 clip。",
        "",
    ]

    for task in ROUND5_TASKS:
        p1_protocol = p2_protocol(task)
        lines.extend(
            [
                f"## {task}",
                "",
                "### 基线（无 guidance，复用 Phase 4.5-4）",
                "",
                "| 模式 | protocol | step | success |",
                "|---|---|---:|---:|",
            ]
        )
        for mode in ("P0", "P1", "P2", "P3"):
            protocol = (
                "not_applicable"
                if mode in {"P0", "P3"}
                else p2_protocol(task)
            )
            if mode == "P1":
                row = by_key.get((task, "P1", protocol, None, "none"))
                p1_cell = "—" if row is None else _cell(row)
                lines.append(f"| P1 | {protocol} | inv | {p1_cell} |")
            else:
                for step in ROUND5_FLOW_STEPS:
                    row = by_key.get((task, mode, protocol, step, "none"))
                    lines.append(
                        f"| {mode} | {protocol} | {step} | {_cell(row)} |"
                    )
        lines.append("")
        lines.extend(
            [
                "### 新增 guidance 条件成功率",
                "",
                "| 模式 | protocol | guidance | "
                + " | ".join(f"step{s}" for s in ROUND5_FLOW_STEPS)
                + " |",
                "|---|---|---|" + "|".join(["---"] * len(ROUND5_FLOW_STEPS)) + "|",
            ]
        )
        for mode in ("P0", "P2", "P3"):
            protocol = "not_applicable" if mode in {"P0", "P3"} else p2_protocol(task)
            guidelines = ROUND5_P3_GUIDANCE_MODES if mode == "P3" else ("guided_flow", "post_opt")
            for guidance in guidelines:
                cells = [
                    _cell(by_key.get((task, mode, protocol, step, guidance)))
                    for step in ROUND5_FLOW_STEPS
                ]
                lines.append(
                    f"| {mode} | {protocol} | {guidance} | " + " | ".join(cells) + " |"
                )
        lines.append("")

        guidance_rows = sorted(
            (
                item
                for item in comparisons
                if item["category"] == "guidance_vs_none"
                and str(item["baseline"][0]) == task
            ),
            key=lambda item: (item["baseline"][1], item["treatment"][5], item["baseline"][3]),
        )
        lines.extend(
            [
                "### guidance 相对无 guidance（paired）",
                "",
                "| mode | guidance | step | none | guidance | Δ pp | improved | regressed | McNemar p |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in guidance_rows:
            paired = item["paired"]
            lines.append(
                f"| {item['baseline'][1]} | {item['treatment'][5]} | "
                f"{item['baseline'][3]} | {item['baseline_success_rate_percent']:.1f}% | "
                f"{item['treatment_success_rate_percent']:.1f}% | {item['delta_pp']:.1f} | "
                f"{paired['improved']} | {paired['regressed']} | "
                f"{_fmt(paired['mcnemar_exact_two_sided_p'], 6)} |"
            )
        lines.append("")

        p3_rows = sorted(
            (
                item
                for item in comparisons
                if item["category"] == "p3_guidance_semantics"
                and str(item["baseline"][0]) == task
            ),
            key=lambda item: item["baseline"][3],
        )
        lines.extend(
            [
                "### P3 guidance 语义对照（paired）",
                "",
                "| step | baseline | treatment | Δ pp | improved | regressed | McNemar p |",
                "|---:|---|---|---:|---:|---:|---:|",
            ]
        )
        for item in p3_rows:
            paired = item["paired"]
            lines.append(
                f"| {item['baseline'][3]} | {item['baseline'][5]} | {item['treatment'][5]} | "
                f"{item['delta_pp']:.1f} | {paired['improved']} | {paired['regressed']} | "
                f"{_fmt(paired['mcnemar_exact_two_sided_p'], 6)} |"
            )
        lines.append("")

        p1_rows = sorted(
            (
                item
                for item in comparisons
                if item["category"] == "p1_vs_learned"
                and str(item["baseline"][0]) == task
            ),
            key=lambda item: (item["treatment"][1], item["treatment"][3]),
        )
        lines.extend(
            [
                "### 学习 planner 相对 P1（随机初始化 CEM，paired）",
                "",
                "| mode | step | P1 | planner | Δ pp | improved | regressed | McNemar p |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in p1_rows:
            paired = item["paired"]
            lines.append(
                f"| {item['treatment'][1]} | {item['treatment'][3]} | "
                f"{item['baseline_success_rate_percent']:.1f}% | "
                f"{item['treatment_success_rate_percent']:.1f}% | {item['delta_pp']:.1f} | "
                f"{paired['improved']} | {paired['regressed']} | "
                f"{_fmt(paired['mcnemar_exact_two_sided_p'], 6)} |"
            )
        lines.append("")

    step_rows = sorted(
        (item for item in comparisons if item["category"] == "step_vs_step16"),
        key=lambda item: (item["baseline"][0], item["baseline"][1], item["baseline"][2], item["baseline"][5], item["treatment"][3]),
    )
    lines.extend(
        [
            "## step 相对 step16（paired，含 guidance）",
            "",
            "| task | mode | protocol | guidance | step | Δ pp | improved | regressed |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in step_rows:
        paired = item["paired"]
        lines.append(
            f"| {item['baseline'][0]} | {item['baseline'][1]} | {item['baseline'][2]} | "
            f"{item['baseline'][5]} | {item['treatment'][3]} | {item['delta_pp']:.1f} | "
            f"{paired['improved']} | {paired['regressed']} |"
        )
    lines.append("")

    lines.extend(["## 方向性结论", ""])
    for category, text in (
        ("p1_vs_learned", "学习 planner 相对 P1"),
        ("guidance_vs_none", "guidance 相对无 guidance"),
        ("p3_guidance_semantics", "P3 guidance 语义互比"),
        ("step_vs_step16", "step 相对 step16"),
    ):
        item = analysis["directional_conclusions"].get(category, {})
        lines.append(
            f"- {text}：{item.get('positive', 0)} 个正向、{item.get('negative', 0)} 个负向、"
            f"{item.get('zero', 0)} 个持平（比较数 {item.get('comparisons', 0)}）。"
        )
    lines.extend(
        [
            "- 所有比较均在同一 legacy cohort 的相同 episode identity 上配对。",
            "- 单 checkpoint、单训练 seed、50 episodes 的描述性结果，不外推到 final 或其他 seed。",
            "",
            "## 产物索引",
            "",
            f"- 分析 JSON：`{output_root / 'analysis' / 'analysis.json'}`",
            f"- 条件 CSV：`{output_root / 'analysis' / 'conditions.csv'}`",
            f"- 比较 CSV：`{output_root / 'analysis' / 'comparisons.csv'}`",
            f"- 本报告：`{report_path}`",
            f"- 代码 commit：`{_git_commit() or 'unknown'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    output_root = _resolve(args.output_root)
    baseline_root = _resolve(config["cohort"]["baseline_results_root"])
    manifests = _load_manifests(config)
    checkpoints = {
        task: str(_resolve(config["training"]["checkpoints"][task]).resolve())
        for task in ROUND5_TASKS
    }
    indexed, validation = load_round5_results(
        output_root,
        baseline_root=baseline_root,
        manifests=manifests,
        checkpoints=checkpoints,
        require_complete=True,
    )
    for task in ROUND5_TASKS:
        validation["manifest_metadata"][task]["path"] = str(
            _resolve(config["cohort"]["legacy"]["paths"][task]).resolve()
        )
    analysis = analyze_round5_results(indexed, validation=validation)
    analysis["experiment"] = "Round 5 Phase 1"
    analysis["config"] = str(_resolve(args.config).resolve())
    analysis["config_sha256"] = _sha256_file(_resolve(args.config))
    analysis["output_root"] = str(output_root.resolve())
    analysis["code_commit"] = _git_commit()
    analysis["config_data"] = config
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "analysis.json").write_text(
        json.dumps(_jsonable(analysis), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
        "baseline_success_rate_percent",
        "treatment_success_rate_percent",
        "net_success_delta",
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
    parser.add_argument("--task", choices=(*ROUND5_TASKS, "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=_gpu, help="one physical GPU0-7")
    parser.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB)
    parser.add_argument(
        "--condition-index",
        action="append",
        type=int,
        help="run only the specified global guidance condition index",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    if args.dry_run:
        conditions = guidance_condition_specs()
        tasks = ROUND5_TASKS if args.task == "all" else (args.task,)
        selected = [
            item
            for item in conditions
            if str(item["task"]) in tasks
        ]
        if args.condition_index is not None:
            selected = [
                conditions[index]
                for index in args.condition_index
                if str(conditions[index]["task"]) in tasks
            ]
        print(
            json.dumps(
                {
                    "tasks": list(tasks),
                    "new_conditions_total": len(conditions),
                    "conditions_selected": len(selected),
                    "baseline_reuse": True,
                    "output_root": str(_resolve(args.output_root)),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    if not args.analyze_only:
        tasks = ROUND5_TASKS if args.task == "all" else (args.task,)
        for task in tasks:
            task_args = argparse.Namespace(**vars(args))
            task_args.task = task
            run_task(task_args, config)
    if args.analyze_only or args.task == "all":
        analyze(args, config)


if __name__ == "__main__":
    main()
