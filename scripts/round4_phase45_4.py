#!/usr/bin/env python3
"""Run and analyze the Round 4 Phase 4.5-4 legacy/dev comparison.

The legacy cohort is evaluated into a fresh output root.  The dev cohort is
read directly from the completed Phase 4.5-3 output root and is never copied or
rerun by this entrypoint.
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
from source.common.round4_phase45_4 import (
    PHASE454_FLOW_STEPS,
    PHASE454_INTEGRATOR,
    PHASE454_NON_CEM_PROTOCOL,
    PHASE454_PROTOCOL_VARIANTS,
    PHASE454_TASKS,
    analyze_phase454_results,
    condition_key,
    condition_name,
    load_phase454_results,
    phase454_condition_specs,
)


DEFAULT_CONFIG = ROOT / "config" / "round4" / "phase45_4.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "round4" / "phase45_4_seed3072_legacy_dev"
DEFAULT_REPORT = ROOT / "docs" / "report" / "round4" / "round4_phase45-4_legacy_protocol_comparison_report.md"
DEFAULT_MIN_FREE_MIB = 4096


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


def _gpu_preflight(gpu: str, *, minimum_free_mib: int = DEFAULT_MIN_FREE_MIB) -> dict[str, Any]:
    """Check one physical GPU immediately before loading a worker model."""
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
        raise RuntimeError(
            f"cannot inspect GPU{gpu} before Phase 4.5-4 worker startup"
        ) from exc
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
        raise ValueError(f"Phase 4.5-4 config must be an object: {path}")
    return value


def _conditions() -> list[dict[str, Any]]:
    return phase454_condition_specs()


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
            f"condition indices out of range: {invalid}; "
            f"valid range is 0..{len(conditions) - 1}"
        )
    return [conditions[index] for index in normalized]


def _condition_name(condition: Mapping[str, Any]) -> str:
    return condition_name(condition)


def _condition_dir(root: Path, task: str, condition: Mapping[str, Any]) -> Path:
    return root / "conditions" / "legacy" / task / _condition_name(condition)


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


def _expected_condition_key(
    task: str, condition: Mapping[str, Any], protocol_variant: str
):
    return (
        task,
        str(condition["mode"]),
        str(condition["cem_protocol"]),
        condition["action_flow_steps"],
        str(condition["action_flow_integrator"]),
        protocol_variant,
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
    expected = _expected_condition_key(task, condition, "legacy")
    reused = _reuse_or_raise(target, expected)
    if reused is not None:
        print(json.dumps({"status": "reused", "result": str(target / "result.json")}))
        return reused
    cfg = _compose(task, manifest, device)
    identity = EvaluationIdentity(
        entrypoint="round4_phase45_4",
        policy_kind="round4_shared_dit",
        checkpoint=str(_resolve(config["training"]["checkpoints"][task]).resolve()),
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
        action_flow_integrator=str(condition["action_flow_integrator"]),
        cem_protocol=str(condition["cem_protocol"]),
        allowed_protocol_variants=("legacy", "round3_revised"),
    )
    observed = condition_key(payload)
    if observed != expected:
        raise ValueError(f"new result condition {observed} != expected {expected}")
    print(json.dumps({"status": "ok", "result": str(target / "result.json")}))
    return payload


def run_task(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    task = str(args.task)
    if task not in PHASE454_TASKS:
        raise ValueError(f"unknown task {task!r}")
    legacy_paths = config["cohort"]["legacy"]["paths"]
    manifest = CohortManifest.load(_resolve(legacy_paths[task]))
    if (
        manifest.cohort_kind != "dev"
        or manifest.protocol_variant != "legacy"
        or len(manifest.entries) != 50
    ):
        raise ValueError(f"Phase 4.5-4 requires canonical legacy_50 dev cohort: {task}")
    checkpoint = _resolve(config["training"]["checkpoints"][task])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    _configure_device(
        args.device,
        args.gpu,
        minimum_free_mib=int(args.min_free_mib),
    )
    model, resolved = load_policy_or_model(str(checkpoint))
    if resolved is not None and Path(resolved).resolve() != checkpoint.resolve():
        raise ValueError(f"checkpoint resolver changed the requested path: {checkpoint}")
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for condition in _select_conditions(_conditions(), args.condition_index):
        if condition["task"] != task:
            continue
        _run_fast_condition(
            task=task,
            condition=condition,
            config=config,
            output_root=output_root,
            device=args.device,
            model=model,
            manifest=manifest,
        )


def _cohort_section(config: Mapping[str, Any], variant: str) -> Mapping[str, Any]:
    """Return the config cohort section for a protocol variant.

    The config names the two sections ``legacy`` and ``dev`` while the protocol
    variant carried by manifests/results is ``legacy``/``round3_revised``.
    """
    for section in config["cohort"].values():
        if isinstance(section, Mapping) and section.get("protocol_variant") == variant:
            return section
    raise ValueError(f"config has no cohort section for protocol variant {variant!r}")


def _load_manifests(config: Mapping[str, Any]) -> dict[str, dict[str, CohortManifest]]:
    manifests = {}
    for task in PHASE454_TASKS:
        manifests[task] = {
            variant: CohortManifest.load(
                _resolve(_cohort_section(config, variant)["paths"][task])
            )
            for variant in PHASE454_PROTOCOL_VARIANTS
        }
    return manifests


def _condition_cell(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return "—"
    step = "inv" if row["action_flow_steps"] is None else str(row["action_flow_steps"])
    return (
        f"{step}: {row['successes']}/50 ({row['success_rate_percent']:.1f}%)<br>"
        f"CI [{row['wilson_95_percent_low']:.1f}, {row['wilson_95_percent_high']:.1f}]%"
    )


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


_AVERAGE_MODE_PROTOCOLS: tuple[tuple[str, str], ...] = (
    ("P0", PHASE454_NON_CEM_PROTOCOL),
    ("P1", "legacy"),
    ("P1", "cem-clip"),
    ("P1", "cem-scale"),
    ("P2", "legacy"),
    ("P2", "cem-clip"),
    ("P2", "cem-scale"),
    ("P3", PHASE454_NON_CEM_PROTOCOL),
)


def _step_label(step: Any) -> str:
    return "inv" if step is None else str(step)


def _mean_cell(group: Sequence[Mapping[str, Any]]) -> str:
    """Macro-average one condition across the tasks that registered it."""
    if not group:
        return "—"
    mean = sum(float(item["success_rate_percent"]) for item in group) / len(group)
    return f"{mean:.1f}%"


def _render_report(
    analysis: Mapping[str, Any],
    *,
    config_path: Path,
    output_root: Path,
    report_path: Path,
) -> str:
    rows = list(analysis["rows"])
    comparisons = list(analysis["comparisons"])
    lines = [
        "# Round 4 Phase 4.5-4：旧协议与 dev 对照实验",
        "",
        "## 实验范围",
        "",
        "本轮使用 R4-AB seed3072 的 epoch10 checkpoint，在四个 task 上比较两个独立的 50-episode cohort：canonical `legacy_50.json` 与已完成 Phase 4.5-3 的 `round3_revised` dev 结果。"
        + "每个 cohort 共 "
        + str(analysis["cohort_summary"]["legacy"]["conditions"])
        + " 个条件："
        + "、".join(
            f"{task} {count} 个"
            for task, count in analysis["cohort_summary"]["legacy"]["tasks"].items()
        )
        + "。只运行 Euler；flow step 网格为 1/2/5/10/16/32（step16 为参考点）。Reacher 额外包含 P1/P2 的 cem-clip 与 cem-scale，TwoRoom 额外包含 P2/cem-clip。不运行 Heun、final 或 LeWM。",
        "",
        f"- 配置：`{config_path}`",
        f"- 输出根目录：`{output_root}`",
        f"- dev 结果来源：`{analysis['validation'].get('dev_root', '—')}`（直接引用，未重跑）",
        f"- 分析生成路径：`{output_root / 'analysis' / 'analysis.json'}`",
        "",
        "## 验收与 cohort 完整性",
        "",
        f"- 条件结果：{analysis['validation'].get('result_count')} / {analysis['validation'].get('expected_count')}；legacy={analysis['validation'].get('legacy_result_count')} / {analysis['validation'].get('legacy_expected_count')}，dev={analysis['validation'].get('dev_result_count')} / {analysis['validation'].get('dev_expected_count')}。",
        f"- `status=ok`：{analysis['validation'].get('all_status_ok')}；无 Heun：{analysis['validation'].get('no_heun')}；flow step 在预注册网格内：{analysis['validation'].get('flow_steps_within_grid')}。",
        f"- 每个结果 50 episodes，episode identity 已按 manifest 校验：{analysis['validation'].get('episode_identity_validated')}。",
        "- legacy 与 dev 使用不同 cohort identity；两者之间只做成功率、Wilson 区间和绝对差异的 unpaired descriptive comparison，明确禁止 paired McNemar 检验。",
        "",
        "## Manifest、checkpoint 与 cohort hash",
        "",
        "| cohort | task | protocol_variant | cohort ID | cohort SHA256 | entries | seed | goal offset | manifest/result source |",
        "|---|---|---|---|---|---:|---:|---:|---|",
    ]
    metadata = analysis["validation"].get("manifest_metadata", {})
    for variant, label in (("legacy", "legacy"), ("round3_revised", "dev")):
        for task in PHASE454_TASKS:
            item = metadata.get(variant, {}).get(task, {})
            source = item.get("path", "—")
            lines.append(
                "| {cohort} | {task} | `{variant}` | `{cohort_id}` | `{sha}` | {episodes} | {seed} | {offset} | {source} |".format(
                    cohort=label,
                    task=task,
                    variant=variant,
                    cohort_id=item.get("cohort_id", "—"),
                    sha=item.get("cohort_sha256", "—"),
                    episodes=item.get("episodes", "—"),
                    seed=item.get("seed", "—"),
                    offset=item.get("goal_offset_steps", "—"),
                    source=source,
                )
            )
    lines.extend(
        [
            "",
            "checkpoint paths are embedded in every condition row and validated against the config:",
            "",
            "| task | checkpoint | SHA256 (config-time reference) |",
            "|---|---|---|",
        ]
    )
    for task in PHASE454_TASKS:
        checkpoint = _resolve(analysis["config_data"]["training"]["checkpoints"][task])
        lines.append(
            f"| {task} | `{checkpoint}` | `{_sha256_file(checkpoint) if checkpoint.is_file() else 'unavailable'}` |"
        )

    lines.extend(["", "## 完整条件表", ""])
    for task in PHASE454_TASKS:
        lines.extend(
            [
                f"### {task}",
                "",
                "| cohort | mode/protocol | integrator | step | success | Wilson 95% | planning mean/median/P95 (s) | result |",
                "|---|---|---|---:|---:|---|---:|---|",
            ]
        )
        task_rows = sorted(
            [row for row in rows if row["task"] == task],
            key=lambda row: (
                row["cohort"],
                row["mode"],
                row["cem_protocol"],
                -1 if row["action_flow_steps"] is None else row["action_flow_steps"],
            ),
        )
        for row in task_rows:
            step = "inv" if row["action_flow_steps"] is None else row["action_flow_steps"]
            lines.append(
                "| {cohort} | {mode}/{protocol} | {integrator} | {step} | {successes}/50 ({rate:.1f}%) | [{low:.1f}, {high:.1f}]% | {mean}/{median}/{p95} | `{path}` |".format(
                    cohort=row["cohort"],
                    mode=row["mode"],
                    protocol=row["cem_protocol"],
                    integrator=row["action_flow_integrator"],
                    step=step,
                    successes=row["successes"],
                    rate=row["success_rate_percent"],
                    low=row["wilson_95_percent_low"],
                    high=row["wilson_95_percent_high"],
                    mean=_fmt(row["planning_mean_seconds"]),
                    median=_fmt(row["planning_median_seconds"]),
                    p95=_fmt(row["planning_p95_seconds"]),
                    path=row["path"],
                )
            )

        step_columns = list(PHASE454_FLOW_STEPS)
        lines.extend(
            [
                "",
                "#### step 子表",
                "",
                "| cohort | mode/protocol | "
                + " | ".join(f"step{s}" for s in step_columns)
                + " |",
                "|---|---|" + "|".join(["---"] * len(step_columns)) + "|",
            ]
        )
        for cohort in ("legacy", "dev"):
            task_index = {
                (row["cohort"], row["mode"], row["cem_protocol"], row["action_flow_steps"]): row
                for row in task_rows
            }
            for mode, protocol in (
                ("P0", PHASE454_NON_CEM_PROTOCOL),
                ("P2", "legacy"),
                ("P2", "cem-clip"),
                ("P2", "cem-scale"),
                ("P3", PHASE454_NON_CEM_PROTOCOL),
            ):
                if not any(
                    (cohort, mode, protocol, step) in task_index
                    for step in step_columns
                ):
                    continue
                cells = [
                    _condition_cell(task_index.get((cohort, mode, protocol, step)))
                    for step in step_columns
                ]
                lines.append(
                    f"| {cohort} | {mode}/{protocol} | " + " | ".join(cells) + " |"
                )
        lines.extend(
            [
                "",
                "#### mode/protocol 子表",
                "",
                "| cohort | step | P0/not_applicable | P1/legacy | P1/cem-clip | P1/cem-scale | P2/legacy | P2/cem-clip | P2/cem-scale | P3/not_applicable |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for cohort in ("legacy", "dev"):
            task_index = {
                (row["cohort"], row["mode"], row["cem_protocol"], row["action_flow_steps"]): row
                for row in task_rows
            }
            for step in (None, *PHASE454_FLOW_STEPS):
                label = "inv" if step is None else str(step)
                cells = [
                    _condition_cell(task_index.get((cohort, mode, protocol, step)))
                    for mode, protocol in (
                        ("P0", PHASE454_NON_CEM_PROTOCOL),
                        ("P1", "legacy"),
                        ("P1", "cem-clip"),
                        ("P1", "cem-scale"),
                        ("P2", "legacy"),
                        ("P2", "cem-clip"),
                        ("P2", "cem-scale"),
                        ("P3", PHASE454_NON_CEM_PROTOCOL),
                    )
                ]
                lines.append(f"| {cohort} | {label} | " + " | ".join(cells) + " |")
        lines.append("")

    by_condition: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            row["cohort"],
            row["mode"],
            row["cem_protocol"],
            row["action_flow_steps"],
        )
        by_condition.setdefault(key, []).append(row)
    mode_rank = {combo: index for index, combo in enumerate(_AVERAGE_MODE_PROTOCOLS)}
    step_rank = {None: -1, **{step: index for index, step in enumerate(PHASE454_FLOW_STEPS)}}
    condition_keys = sorted(
        by_condition,
        key=lambda key: (
            key[0],
            mode_rank[(key[1], key[2])],
            step_rank[key[3]],
        ),
    )

    lines.extend(
        [
            "",
            "## 四任务平均",
            "",
            "本节把同一 condition 在注册了它的任务上做宏平均（每个任务等权），并给出合并成功数与 pooled Wilson 区间。",
            "cem-clip 仅覆盖 Reacher/TwoRoom，cem-scale 仅覆盖 Reacher，因此不同行的参与任务数可能不同，`tasks` 列给出参与任务数供核对。",
            "注意：四个任务的难度、成功谓词和动作维度不同，跨任务平均只作纵向汇总，绝对值不能与单任务或论文数字直接等同。",
            "",
            "### 四任务平均总表",
            "",
            "| cohort | mode/protocol | step | tasks | mean success | pooled success | pooled Wilson 95% |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for key in condition_keys:
        cohort, mode, protocol, step = key
        group = by_condition[key]
        tasks = len(group)
        successes = sum(int(item["successes"]) for item in group)
        episodes = sum(int(item["episodes"]) for item in group)
        mean_rate = sum(float(item["success_rate_percent"]) for item in group) / tasks
        low, high = wilson_interval(successes, episodes)
        lines.append(
            f"| {cohort} | {mode}/{protocol} | {_step_label(step)} | {tasks} | "
            f"{mean_rate:.1f}% | {successes}/{episodes} | "
            f"[{low * 100.0:.1f}, {high * 100.0:.1f}]% |"
        )

    lines.extend(
        [
            "",
            "### 四任务平均 step 子表",
            "",
            "| cohort | mode/protocol | tasks | "
            + " | ".join(f"step{step}" for step in PHASE454_FLOW_STEPS)
            + " |",
            "|---|---|---:|" + "|".join(["---"] * len(PHASE454_FLOW_STEPS)) + "|",
        ]
    )
    for cohort in ("legacy", "dev"):
        for mode, protocol in _AVERAGE_MODE_PROTOCOLS:
            any_present = False
            task_counts: set[int] = set()
            cells = []
            for step in PHASE454_FLOW_STEPS:
                group = by_condition.get((cohort, mode, protocol, step), [])
                if group:
                    any_present = True
                    task_counts.add(len(group))
                cells.append(_mean_cell(group))
            if not any_present:
                continue
            tasks_label = "/".join(str(count) for count in sorted(task_counts))
            lines.append(
                f"| {cohort} | {mode}/{protocol} | {tasks_label} | "
                + " | ".join(cells)
                + " |"
            )

    lines.extend(
        [
            "",
            "### 四任务平均 mode/protocol 子表",
            "",
            "| cohort | step | "
            + " | ".join(f"{mode}/{protocol}" for mode, protocol in _AVERAGE_MODE_PROTOCOLS)
            + " |",
            "|---|---|" + "|".join(["---"] * len(_AVERAGE_MODE_PROTOCOLS)) + "|",
        ]
    )
    for cohort in ("legacy", "dev"):
        for step in (None, *PHASE454_FLOW_STEPS):
            cells = [
                _mean_cell(by_condition.get((cohort, mode, protocol, step), []))
                for mode, protocol in _AVERAGE_MODE_PROTOCOLS
            ]
            lines.append(
                f"| {cohort} | {_step_label(step)} | " + " | ".join(cells) + " |"
            )

    within = sorted(
        (item for item in comparisons if item["comparison_type"] == "paired"),
        key=lambda item: (item["category"], item["label"]),
    )
    cross = [item for item in comparisons if item["category"] == "legacy_vs_dev"]
    lines.extend(
        [
            "",
            "## 同一 cohort 内 paired 比较",
            "",
            "这些比较只在同一 cohort 的相同 episode identity 上进行，因此可以报告 exact McNemar；每张表按 category 排序，step_vs_step16 与 protocol 对照可分别筛读。",
            "",
            "| category | comparison | Δ success (pp) | improved | regressed | net | exact McNemar p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in within:
        paired = item["paired"]
        lines.append(
            f"| {item['category']} | `{item['label']}` | {item['delta_pp']:.1f} | {paired['improved']} | {paired['regressed']} | {item['net_success_delta']} | {_fmt(paired['mcnemar_exact_two_sided_p'], 6)} |"
        )
    lines.extend(
        [
            "",
            "## legacy 与 dev 的 unpaired 对照",
            "",
            "两套 manifest 的 cohort ID/hash 不同；以下表格同时列出两边的成功率和 Wilson 区间，并给出 dev 相对 legacy 的 signed/absolute difference。禁止把这些行解释为 paired McNemar。",
            "",
            "| task | condition | legacy success / Wilson | dev success / Wilson | dev−legacy (pp) | absolute difference (pp) |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for item in cross:
        baseline_ci = item["baseline_wilson_95_percent"]
        treatment_ci = item["treatment_wilson_95_percent"]
        task = item["baseline"][0]
        condition = "/".join(str(value) for value in item["baseline"][1:5])
        lines.append(
            f"| {task} | `{condition}` | {item['baseline_success_rate_percent']:.1f}% [{baseline_ci[0]:.1f}, {baseline_ci[1]:.1f}]% | {item['treatment_success_rate_percent']:.1f}% [{treatment_ci[0]:.1f}, {treatment_ci[1]:.1f}]% | {item['delta_pp']:.1f} | {item['absolute_delta_pp']:.1f} |"
        )

    lines.extend(["", "## 方向性结论", ""])
    direction = analysis["directional_conclusions"]
    for category, text in (
        ("step_vs_step16", "step 相对 step16"),
        ("cem_clip_vs_legacy", "cem-clip 相对 legacy"),
        ("cem_scale_vs_legacy", "cem-scale 相对 legacy"),
        ("cem_scale_vs_cem_clip", "cem-scale 相对 cem-clip"),
        ("legacy_vs_dev", "dev 相对 legacy cohort"),
    ):
        item = direction.get(category, {})
        lines.append(
            f"- {text}：{item.get('positive', 0)} 个正向、{item.get('negative', 0)} 个负向、{item.get('zero', 0)} 个持平（比较数 {item.get('comparisons', 0)}）。"
        )
    lines.extend(
        [
            "- 判断原则：只有当同一 task/mode/protocol 的 legacy 与 dev signed difference 改变方向时，才称为 cohort 改变了该方向性结论；本报告不把不同 cohort 的 episode-level 成败强行配对。",
            "- 这些是固定 checkpoint、单 seed、每 cohort 50 episodes 的描述性结果，不外推到 final 或其他训练 seed。",
            "",
            "## 产物索引",
            "",
            f"- 分析 JSON：`{output_root / 'analysis' / 'analysis.json'}`",
            f"- 条件 CSV：`{output_root / 'analysis' / 'conditions.csv'}`",
            f"- 比较 CSV：`{output_root / 'analysis' / 'comparisons.csv'}`",
            f"- 本报告：`{report_path}`",
            f"- 代码 commit（生成时）：`{_git_commit() or 'unknown'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    output_root = _resolve(args.output_root)
    manifests = _load_manifests(config)
    checkpoints = {
        task: str(_resolve(config["training"]["checkpoints"][task]).resolve())
        for task in PHASE454_TASKS
    }
    dev_root = _resolve(_cohort_section(config, "round3_revised")["results_root"])
    indexed, validation = load_phase454_results(
        output_root,
        dev_root=dev_root,
        manifests=manifests,
        checkpoints=checkpoints,
        require_complete=True,
    )
    for variant in PHASE454_PROTOCOL_VARIANTS:
        for task in PHASE454_TASKS:
            validation["manifest_metadata"][variant][task]["path"] = str(
                _resolve(_cohort_section(config, variant)["paths"][task]).resolve()
            )
    analysis = analyze_phase454_results(indexed, validation=validation)
    analysis["experiment"] = "Round 4 Phase 4.5-4"
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
        "comparison_type",
        "delta_pp",
        "absolute_delta_pp",
        "baseline_success_rate_percent",
        "treatment_success_rate_percent",
        "baseline_wilson_95_percent",
        "treatment_wilson_95_percent",
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
    parser.add_argument("--task", choices=(*PHASE454_TASKS, "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=_gpu, help="one physical GPU0-7")
    parser.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB)
    parser.add_argument(
        "--condition-index",
        action="append",
        type=int,
        help="run only the specified global condition index; repeat for multiple indices",
    )
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    if args.dry_run:
        conditions = _select_conditions(_conditions(), args.condition_index)
        tasks = PHASE454_TASKS if args.task == "all" else (args.task,)
        selected = [item for item in conditions if item["task"] in tasks]
        print(
            json.dumps(
                {
                    "tasks": list(tasks),
                    "conditions_total": len(_conditions()),
                    "conditions_selected": len(selected),
                    "conditions_per_cohort": len(_conditions()),
                    "cohorts": ["legacy", "dev"],
                    "dev_rerun": False,
                    "output_root": str(_resolve(args.output_root)),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    if not args.analyze_only:
        tasks = PHASE454_TASKS if args.task == "all" else (args.task,)
        for task in tasks:
            task_args = argparse.Namespace(**vars(args))
            task_args.task = task
            run_task(task_args, config)
    if args.analyze_only or args.task == "all":
        analyze(args, config)


if __name__ == "__main__":
    main()
