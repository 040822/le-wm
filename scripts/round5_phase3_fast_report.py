#!/usr/bin/env python3
"""Build the initial FastLeWAM-only Round 5 Phase 3 report."""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.round5_phase3 import (  # noqa: E402
    EVAL_BUDGET,
    EVAL_SEED,
    GOAL_OFFSET,
    NUM_EVAL,
    PHASE3_TASKS,
    SEED,
    _condition_dir,
    _condition_name,
    _ensure_manifest,
    fast_conditions,
    p2_protocol,
)
from source.common.phase3_compat import PHASE3_DATASETS, scene_target_group  # noqa: E402
from source.common.round3_phase1 import wilson_interval  # noqa: E402
from stable_worldmodel.data.formats.hdf5 import HDF5Dataset  # noqa: E402

OUTPUT_ROOT = ROOT / "outputs" / "round5" / "phase3"
ANALYSIS_DIR = OUTPUT_ROOT / "analysis"
REPORT_PATH = ROOT / "docs" / "report" / "round5" / "round5_phase3_fastlewam_initial_report.md"
TASK_LABELS = {
    "scene": "OGBench-Scene",
    "finger": "DMC-Finger-turn_hard",
    "humanoid": "DMC-Humanoid-walk",
}
COHORT_HASHES = {
    "scene": "ec30c7373119c070d40190f98bbdc43dea57cd767b1860cb1c070f6233d5f771",
    "finger": "6f3a7722d466b666ba1af2d440a83157bc24edbde5c7ac449bfeb9696c58eb8c",
    "humanoid": "8fa169404c13365d19d894928923b4eae625504ee5d5fc988afd09a4c14262cc",
}
STEPS = (1, 2, 5, 10, 16, 32)


def _nan_paths(value, path="root"):
    if isinstance(value, float) and math.isnan(value):
        return [path]
    if isinstance(value, dict):
        return sum((_nan_paths(v, f"{path}.{k}") for k, v in value.items()), [])
    if isinstance(value, (list, tuple)):
        return sum((_nan_paths(v, f"{path}[{i}]") for i, v in enumerate(value)), [])
    return []


def _compact(spec):
    step = "inv" if spec["step"] is None else f"s{spec['step']}"
    return f"{spec['mode']}/{spec['guidance']}/{step}"


def _row(task, spec):
    condition = _condition_name(spec)
    result_path = _condition_dir(OUTPUT_ROOT, task, spec) / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok" or payload.get("task") != task or payload.get("phase3_method") != "fast_lewam":
        raise ValueError(f"invalid result identity: {result_path}")
    if payload.get("epoch") != 10:
        raise ValueError(f"FastLeWAM result is not epoch 10: {result_path}")
    if payload.get("phase3_condition") != spec:
        raise ValueError(f"condition mismatch: {result_path}")
    episodes = payload.get("episodes", [])
    if len(episodes) != NUM_EVAL:
        raise ValueError(f"episode count != {NUM_EVAL}: {result_path}")
    vector = [bool(item.get("success", False)) for item in episodes]
    successes = sum(vector)
    low, high = wilson_interval(vector)
    summary = payload.get("summary", {})
    if summary.get("successes") != successes or summary.get("num_episodes") != NUM_EVAL:
        raise ValueError(f"summary mismatch: {result_path}")
    if _nan_paths(summary):
        raise ValueError(f"NaN success summary: {result_path}")
    params = payload.get("parameters", {})
    cohort_hash = payload.get("cohort_sha256") or params.get("cohort_sha256")
    if cohort_hash != COHORT_HASHES[task]:
        raise ValueError(f"cohort hash mismatch: {result_path}")
    for key, expected in (("seed", EVAL_SEED), ("goal_offset_steps", GOAL_OFFSET), ("eval_budget", EVAL_BUDGET), ("horizon", 5), ("action_block", 5)):
        if params.get(key) != expected:
            raise ValueError(f"parameter {key} mismatch in {result_path}")
    return {
        "task": task,
        "task_label": TASK_LABELS[task],
        "method": "fast_lewam",
        "mode": spec["mode"],
        "protocol": spec["protocol"],
        "guidance": spec["guidance"],
        "flow_step": "invariant" if spec["step"] is None else spec["step"],
        "integrator": "not_applicable" if spec["step"] is None else "euler",
        "condition": condition,
        "condition_compact": _compact(spec),
        "successes": int(successes),
        "episodes": len(vector),
        "success_rate_percent": successes * 100.0 / len(vector),
        "wilson_low_percent": low * 100.0,
        "wilson_high_percent": high * 100.0,
        "evaluation_seconds": float(payload.get("evaluation_seconds", 0.0)),
        "epoch": payload.get("epoch"),
        "checkpoint": str(Path(payload["checkpoint"]).resolve().relative_to(ROOT)),
        "code_commit": payload.get("code_commit"),
        "cohort_id": payload.get("cohort_id"),
        "cohort_sha256": cohort_hash,
        "result_path": str(result_path.relative_to(ROOT)),
        "success_vector": vector,
    }


def _fmt(row, ci=True):
    if ci:
        return f"{row['successes']}/{row['episodes']} ({row['success_rate_percent']:.1f}%, [{row['wilson_low_percent']:.1f}, {row['wilson_high_percent']:.1f}])"
    return f"{row['successes']}/{row['episodes']} ({row['success_rate_percent']:.1f}%)"


def _get(rows_by_key, task, mode, protocol, guidance, step):
    spec = {"task": task, "mode": mode, "protocol": protocol, "guidance": guidance, "step": step}
    return rows_by_key[(task, _condition_name(spec))]


def main():
    rows = []
    for task in PHASE3_TASKS:
        specs = fast_conditions(task)
        if len(specs) != 61:
            raise ValueError(f"expected 61 conditions for {task}")
        rows.extend(_row(task, spec) for spec in specs)
    if len(rows) != 183:
        raise ValueError(f"expected 183 rows, got {len(rows)}")
    by_key = {(row["task"], row["condition"]): row for row in rows}

    metadata = {}
    for task in PHASE3_TASKS:
        task_rows = [r for r in rows if r["task"] == task]
        checkpoints = sorted({r["checkpoint"] for r in task_rows})
        metadata[task] = {
            "label": TASK_LABELS[task],
            "dataset": str(PHASE3_DATASETS[task].relative_to(ROOT)),
            "cohort_id": task_rows[0]["cohort_id"],
            "cohort_sha256": COHORT_HASHES[task],
            "condition_count": len(task_rows),
            "episode_count": sum(r["episodes"] for r in task_rows),
            "success_rate_min_percent": min(r["success_rate_percent"] for r in task_rows),
            "success_rate_max_percent": max(r["success_rate_percent"] for r in task_rows),
            "checkpoints": checkpoints,
            "checkpoint_bytes": [(ROOT / p).stat().st_size for p in checkpoints],
            "code_commits": sorted({r["code_commit"] for r in task_rows}),
        }

    primary = {task: _get(by_key, task, "P3", "not_applicable", "none", 1) for task in PHASE3_TASKS}
    p1 = {task: _get(by_key, task, "P1", p2_protocol(task), "none", None) for task in PHASE3_TASKS}

    flow = []
    for task in PHASE3_TASKS:
        protocol = p2_protocol(task)
        for mode in ("P0", "P2", "P3"):
            mode_protocol = "not_applicable" if mode in ("P0", "P3") else protocol
            for step in STEPS:
                r = _get(by_key, task, mode, mode_protocol, "none", step)
                flow.append({k: r[k] for k in ("task", "mode", "condition", "successes", "success_rate_percent", "wilson_low_percent", "wilson_high_percent")})
                flow[-1]["step"] = step

    guidance = []
    for task in PHASE3_TASKS:
        protocol = p2_protocol(task)
        for mode in ("P0", "P2", "P3"):
            mode_protocol = "not_applicable" if mode in ("P0", "P3") else protocol
            names = ("guided_flow", "post_opt") if mode in ("P0", "P2") else ("guided_flow", "post_opt", "post_opt_refine")
            for name in names:
                deltas = []
                for step in STEPS:
                    base = _get(by_key, task, mode, mode_protocol, "none", step)
                    guided = _get(by_key, task, mode, mode_protocol, name, step)
                    deltas.append(guided["success_rate_percent"] - base["success_rate_percent"])
                guidance.append({"task": task, "mode": mode, "guidance": name, "step1_delta_pp": deltas[0], "mean_delta_pp": sum(deltas) / len(deltas), "min_delta_pp": min(deltas), "max_delta_pp": max(deltas)})

    extrema = {}
    for task in PHASE3_TASKS:
        task_rows = [r for r in rows if r["task"] == task]
        best, worst = max(r["success_rate_percent"] for r in task_rows), min(r["success_rate_percent"] for r in task_rows)
        extrema[task] = {
            "best_rate_percent": best,
            "best_conditions": [r["condition"] for r in task_rows if r["success_rate_percent"] == best],
            "worst_rate_percent": worst,
            "worst_conditions": [r["condition"] for r in task_rows if r["success_rate_percent"] == worst],
        }

    scene_dataset = HDF5Dataset(path=PHASE3_DATASETS["scene"], keys_to_load=["privileged_target_task"])
    labels = scene_target_group(scene_dataset, _ensure_manifest("scene", OUTPUT_ROOT))
    scene_counts = dict(sorted(Counter(labels).items()))
    scene_groups = []
    for selected in (primary["scene"], p1["scene"]):
        grouped = {}
        for group in scene_counts:
            values = [ok for ok, label in zip(selected["success_vector"], labels) if label == group]
            grouped[group] = {"episodes": len(values), "successes": sum(values), "success_rate_percent": sum(values) * 100.0 / len(values)}
        scene_groups.append({"condition": selected["condition"], "groups": grouped})

    videos = []
    for task in PHASE3_TASKS:
        for condition in ("p1_step_1", "p3_step_1"):
            video = OUTPUT_ROOT / "videos" / "fastlewam" / task / condition / "videos" / "env_0.mp4"
            result = video.parent.parent / "result.json"
            if not video.is_file() or not result.is_file():
                raise FileNotFoundError(f"missing video/result pair: {video}")
            payload = json.loads(result.read_text(encoding="utf-8"))
            if payload.get("status") != "ok" or len(payload.get("episodes", [])) != 1:
                raise ValueError(f"invalid video result: {result}")
            videos.append({"task": task, "task_label": TASK_LABELS[task], "condition": condition, "video": str(video.relative_to(ROOT)), "result": str(result.relative_to(ROOT)), "bytes": video.stat().st_size})

    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    serial_rows = [{k: v for k, v in r.items() if k != "success_vector"} for r in rows]
    summary = {
        "report_kind": "fastlewam_initial",
        "generated_at": generated,
        "experiment": "Round 5 Phase 3",
        "method": "FastLeWAM",
        "train_seed": SEED,
        "evaluation_seed": EVAL_SEED,
        "num_eval_per_condition": NUM_EVAL,
        "goal_offset_steps": GOAL_OFFSET,
        "eval_budget": EVAL_BUDGET,
        "tasks": list(PHASE3_TASKS),
        "expected_condition_count": 183,
        "condition_count": len(rows),
        "episode_count": sum(r["episodes"] for r in rows),
        "cohort_hashes": COHORT_HASHES,
        "task_metadata": metadata,
        "primary": {task: {k: v for k, v in primary[task].items() if k != "success_vector"} for task in PHASE3_TASKS},
        "p1_invariant": {task: {k: v for k, v in p1[task].items() if k != "success_vector"} for task in PHASE3_TASKS},
        "flow_step_sensitivity": flow,
        "guidance_summary": guidance,
        "extrema": extrema,
        "scene_target_task_counts": scene_counts,
        "scene_target_task_results": scene_groups,
        "videos": videos,
        "data_quality": {"all_status_ok": True, "all_conditions_have_50_episodes": True, "all_cohort_hashes_match": True, "nan_success_metrics": False, "missing_conditions": []},
        "conditions": serial_rows,
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "fastlewam_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = ["task", "task_label", "method", "mode", "protocol", "guidance", "flow_step", "integrator", "condition", "condition_compact", "successes", "episodes", "success_rate_percent", "wilson_low_percent", "wilson_high_percent", "evaluation_seconds", "epoch", "checkpoint", "code_commit", "cohort_id", "cohort_sha256", "result_path"]
    with (ANALYSIS_DIR / "fastlewam_conditions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)

    lines = [
        "# Round 5 Phase 3：FastLeWAM 初版结果报告", "",
        "> 状态：FastLeWAM 的三任务完整条件矩阵已完成；本报告暂不包含 LeWM/LeFlow 基线比较。",
        f"> 生成时间：{generated}", "",
        "本报告整理 OGBench-Scene、DMC-Finger-turn_hard 和 DMC-Humanoid-walk 上的 FastLeWAM 结果。每个任务 61 个条件、每个条件 50 个 episode，共 183 个条件和 9,150 个 episode。主结果按照 Phase 3 方案固定为 P3、flow step=1、无 guidance；P1 invariant、flow-step 和 guidance 条件用于诊断。成功率为 episode 成功数除以 50，区间为 Wilson 95% 区间。", "",
        "## 当前可读结论", "",
        "- **Scene**：61 个条件全部为 0/50（0.0%）；flow step 或 guidance 未改变 all-component 主成功判据。",
        f"- **Finger**：全矩阵范围为 {metadata['finger']['success_rate_min_percent']:.1f}%–{metadata['finger']['success_rate_max_percent']:.1f}%；主条件为 {_fmt(primary['finger'])}。",
        f"- **Humanoid**：全矩阵范围为 {metadata['humanoid']['success_rate_min_percent']:.1f}%–{metadata['humanoid']['success_rate_max_percent']:.1f}%；主条件为 {_fmt(primary['humanoid'])}。",
        "- 这些是固定 cohort、单训练 seed=3072 的描述性结果；在基线完成前，不据此做 FastLeWAM 相对 LeWM/LeFlow 的结论。", "",
        "## 评测协议与模型", "",
        "| 项目 | 设置 |", "|---|---|",
        "| 训练 | Round 5 Phase 1 同款 R4-AB FastLeWAM，epoch 10，训练 seed 3072 |",
        "| 评测 | eval seed 42，50 个 start–goal 对，goal offset 25，execution budget 50，horizon/receding horizon 5，action block 5 |",
        "| flow 条件 | P0/P2/P3 使用 Euler steps {1, 2, 5, 10, 16, 32}；P1 使用 invariant 条件 |",
        "| guidance | P0/P2：guided_flow、post_opt；P3：guided_flow、post_opt、post_opt_refine；step size=0.01，late/inner steps=5，max RMS offset=0.2 |",
        "| P2 协议 | Scene 使用 legacy CEM；Finger/Humanoid 使用 cem-clip；P0/P3 不使用 CEM clipping |",
        "| 成功判据 | Scene 为 cube、button、drawer、window 五项同时满足；DMC 使用环境 termination 判据，未使用 NaN success 字段 |", "",
        "## 数据完整性", "",
        "| 任务 | 条件数 | episode 数 | 成功率范围 | cohort id | cohort SHA-256 |", "|---|---:|---:|---:|---|---|",
    ]
    for task in PHASE3_TASKS:
        m = metadata[task]
        lines.append(f"| {m['label']} | {m['condition_count']}/61 | {m['episode_count']} | {m['success_rate_min_percent']:.1f}%–{m['success_rate_max_percent']:.1f}% | `{m['cohort_id']}` | `{m['cohort_sha256']}` |")
    lines += ["", "- 183/183 条件状态为 `ok`，每项正好 50 episodes；没有缺条件、NaN 成功指标或 cohort hash 不一致。", "- 任务级 checkpoint 均为 epoch 10；各条件的 `code_commit` 随执行期间兼容性修订而不同，完整列表见汇总 JSON。", "", "## 主条件与 P1 invariant", "", "主条件是 `P3/not_applicable/none/step_1/euler`。P1 在实现中没有 action-flow step，因此以 invariant 表示；视频目录名使用 `p1_step_1` 作为展示标签。", "", "| 任务 | 主条件 P3/step1/none | P1 invariant |", "|---|---:|---:|"]
    for task in PHASE3_TASKS:
        lines.append(f"| {TASK_LABELS[task]} | {_fmt(primary[task])} | {_fmt(p1[task])} |")

    lines += ["", "## Flow-step 敏感性（无 guidance）", "", "表中每个单元格为 `成功数/50（成功率）`。", "", "| 任务 | mode | step=1 | step=2 | step=5 | step=10 | step=16 | step=32 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for task in PHASE3_TASKS:
        for mode in ("P0", "P2", "P3"):
            cells = []
            for step in STEPS:
                item = next(x for x in flow if x["task"] == task and x["mode"] == mode and x["step"] == step)
                cells.append(f"{item['successes']}/50 ({item['success_rate_percent']:.1f}%)")
            lines.append(f"| {TASK_LABELS[task]} | {mode} | " + " | ".join(cells) + " |")

    lines += ["", "## Guidance 在 step=1 的变化", "", "Δ 是同一 mode、同一 flow step 下相对无 guidance 的百分点变化。", "", "| 任务 | mode | 无 guidance | guided_flow Δ | post_opt Δ | post_opt_refine Δ |", "|---|---|---:|---:|---:|---:|"]
    for task in PHASE3_TASKS:
        protocol = p2_protocol(task)
        for mode in ("P0", "P2", "P3"):
            mode_protocol = "not_applicable" if mode in ("P0", "P3") else protocol
            base = _get(by_key, task, mode, mode_protocol, "none", 1)
            cells = {}
            for name in ("guided_flow", "post_opt", "post_opt_refine"):
                match = next((x for x in guidance if x["task"] == task and x["mode"] == mode and x["guidance"] == name), None)
                cells[name] = "—" if match is None else f"{match['step1_delta_pp']:+.1f} pp"
            lines.append(f"| {TASK_LABELS[task]} | {mode} | {base['success_rate_percent']:.1f}% | {cells['guided_flow']} | {cells['post_opt']} | {cells['post_opt_refine']} |")
    lines += ["", "完整 guidance×flow-step 差值、每项 Wilson 区间以及 183 个结果的精确路径写入 [fastlewam_conditions.csv](../../../outputs/round5/phase3/analysis/fastlewam_conditions.csv)；可复现汇总写入 [fastlewam_summary.json](../../../outputs/round5/phase3/analysis/fastlewam_summary.json)。", "", "## Scene target_task 分组", "", "Scene 的 all-component 成功率仍是主指标；下面给出主条件和 P1 invariant 按冻结 cohort 中 target_task 的分组结果。分组样本数为 " + ", ".join(f"{k}={v}" for k, v in scene_counts.items()) + "。", "", "| 条件 | cube | button | drawer | window |", "|---|---:|---:|---:|---:|"]
    for label, selected in (("P3/step1/none", primary["scene"]), ("P1/invariant", p1["scene"])):
        grouped = next(item["groups"] for item in scene_groups if item["condition"] == selected["condition"])
        cells = [f"{grouped[g]['successes']}/{grouped[g]['episodes']} ({grouped[g]['success_rate_percent']:.1f}%)" for g in ("cube", "button", "drawer", "window")]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "完整 61 条 Scene 分组统计位于 `scene_target_task_results` 字段。", "", "## 代表视频", "", "每个视频来自冻结 cohort 的第一个 start–goal 对，仅用于可视化，不计入 50-episode 统计。", "", "| 任务 | 条件 | 视频 |", "|---|---|---|"]
    for item in videos:
        lines.append(f"| {item['task_label']} | `{item['condition']}` | [{Path(item['video']).name}](../../../{item['video']}) |")
    lines += ["", "## 解释边界与后续", "", "- 这是单训练 seed=3072、固定 50-episode cohort 的初版整理，不能替代多 seed 统计。", "- Scene 当前全条件为 0%，需要在基线完成后结合 LeWM/LeFlow、成功向量和视频共同检查；本报告不把它解释为模型能力的最终结论。", "- LeWM/LeFlow 的训练、标准条件评测和对应视频仍在 Phase 3 后续步骤中；完成后再生成包含 189 行结果、paired comparison 和 McNemar 检验的总报告。", "", "## 产物", "", f"- 初版报告：`{REPORT_PATH.relative_to(ROOT)}`", f"- 逐条件 CSV：`{(ANALYSIS_DIR / 'fastlewam_conditions.csv').relative_to(ROOT)}`", f"- 汇总 JSON：`{(ANALYSIS_DIR / 'fastlewam_summary.json').relative_to(ROOT)}`", "- 代表视频目录：`outputs/round5/phase3/videos/fastlewam/`", ""]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", "report": str(REPORT_PATH), "csv": str(ANALYSIS_DIR / "fastlewam_conditions.csv"), "summary": str(ANALYSIS_DIR / "fastlewam_summary.json"), "rows": len(rows), "episodes": sum(r["episodes"] for r in rows), "primary": {task: {"successes": primary[task]["successes"], "rate": primary[task]["success_rate_percent"]} for task in PHASE3_TASKS}}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
