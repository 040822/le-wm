#!/usr/bin/env python3
"""Compare the 125-step LeWM/LeFlow Reacher runs with their 50-step baselines."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/round5/phase5_baseline_horizons"
EXTENDED = BASE / "budget_125"
REPORT = ROOT / "docs/report/round5/round5_phase6_2_lewm_leflow_report.md"


def _wilson(successes: int, n: int = 50) -> list[float]:
    z = 1.959963984540054
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [100 * (center - radius), 100 * (center + radius)]


def _paired(left: dict, right: dict, seed: int) -> dict:
    key = lambda episode: (episode["dataset_episode"], episode["start_step"])
    left_by_key = {key(row): bool(row["success"]) for row in left["episodes"]}
    right_by_key = {key(row): bool(row["success"]) for row in right["episodes"]}
    if left_by_key.keys() != right_by_key.keys() or len(left_by_key) != 50:
        raise ValueError("budget results do not share the same 50 evaluation starts")

    differences = np.asarray(
        [int(right_by_key[item]) - int(left_by_key[item]) for item in sorted(left_by_key)],
        dtype=np.int8,
    )
    improved = int(np.count_nonzero(differences > 0))
    regressed = int(np.count_nonzero(differences < 0))
    discordant = improved + regressed
    exact_p = (
        1.0
        if discordant == 0
        else min(
            1.0,
            2
            * sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1))
            / 2**discordant,
        )
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(differences), size=(10_000, len(differences)))
    bootstrap = np.quantile(differences[sampled].mean(axis=1) * 100, [0.025, 0.975])
    left_successes = sum(left_by_key.values())
    right_successes = sum(right_by_key.values())
    return {
        "budget50_successes": left_successes,
        "budget50_success_rate": left_successes / 50,
        "budget50_wilson95_pp": _wilson(left_successes),
        "budget125_successes": right_successes,
        "budget125_success_rate": right_successes / 50,
        "budget125_wilson95_pp": _wilson(right_successes),
        "delta_pp": float(differences.mean() * 100),
        "paired_bootstrap95_pp": [float(value) for value in bootstrap],
        "improved": improved,
        "regressed": regressed,
        "unchanged": int(np.count_nonzero(differences == 0)),
        "mcnemar_exact_p": exact_p,
    }


def _planning_summary(result: dict) -> dict:
    events = result["horizon_ablation"]["planning_events"]
    return {
        "planning_batch_events": len(events),
        "planning_seconds": sum(float(event["planning_seconds"]) for event in events),
        "evaluation_seconds": float(result["evaluation_seconds"]),
    }


def main() -> None:
    rows = []
    for method in ("lewm", "leflow"):
        left_path = BASE / method / "25_25/result.json"
        right_path = EXTENDED / method / "25_25/result.json"
        left, right = json.loads(left_path.read_text()), json.loads(right_path.read_text())
        left_meta, right_meta = left["horizon_ablation"], right["horizon_ablation"]
        for result, meta, expected_budget in ((left, left_meta, 50), (right, right_meta, 125)):
            if len(result["episodes"]) != 50 or int(result["parameters"]["eval_budget"]) != expected_budget:
                raise ValueError(f"invalid episode count or budget for {method} budget {expected_budget}")
            if meta["method"] != method or int(meta["execute_steps"]) != 25 or int(meta["score_steps"]) != 25:
                raise ValueError(f"unexpected condition in {method} budget {expected_budget}")
        for field in ("checkpoint_sha256", "cohort_sha256"):
            if left_meta[field] != right_meta[field]:
                raise ValueError(f"{method} budget runs differ in {field}")
        if left["parameters"]["seed"] != right["parameters"]["seed"]:
            raise ValueError(f"{method} budget runs differ in seed")

        row = {
            "method": method,
            "checkpoint": right["checkpoint"],
            "checkpoint_sha256": right_meta["checkpoint_sha256"],
            "cohort_sha256": right_meta["cohort_sha256"],
            **_paired(left, right, seed=62_025 + (0 if method == "lewm" else 1)),
            "budget50_cost": _planning_summary(left),
            "budget125_cost": _planning_summary(right),
            "budget125_gpu": right_meta["gpu"],
            "budget125_runner_sha256": right_meta["runner_sha256"],
            "budget125_score_audit": right_meta["score_audit"],
        }
        rows.append(row)

    # Holm adjustment across the two predeclared within-method comparisons.
    ordered = sorted(range(len(rows)), key=lambda index: rows[index]["mcnemar_exact_p"])
    previous = 0.0
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (len(rows) - rank) * rows[index]["mcnemar_exact_p"])
        previous = max(previous, adjusted)
        rows[index]["holm_p_two"] = previous

    analysis_dir = EXTENDED / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = analysis_dir / "paired_budget_comparison.json"
    analysis_path.write_text(json.dumps({"status": "complete", "conditions": rows}, indent=2) + "\n")

    lines = [
        "# Reacher：LEWM / LEFlow 预算 50→125 步扩展",
        "",
        "## 协议",
        "",
        "在原有 50 个 Reacher legacy 起点、推理 seed=42、目标偏移25步、25/25执行与评分条件下，将单回合环境步数上限从50提高到125。固定各方法原 checkpoint 和规划设置，不训练、不微调；两预算按相同 episode ID 与起始步配对。125步结果位于 `outputs/round5/phase5_baseline_horizons/budget_125/`。",
        "",
        "LEWM 使用本地训练 checkpoint 与原生 CEM；LEFlow 使用发布 planner 及其冻结的 LEWM 参考权重。两方法的横向差异不作单变量因果解释。",
        "",
        "## 配对结果",
        "",
        "| 方法 | 50步成功 | 125步成功 | 配对差 pp [bootstrap 95% CI] | 改善/退化/不变 | McNemar p | Holm p | 规划时间 50→125 (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lo, hi = row["paired_bootstrap95_pp"]
        lines.append(
            f"| {row['method']} | {row['budget50_successes']}/50 ({row['budget50_success_rate']:.0%}) "
            f"| {row['budget125_successes']}/50 ({row['budget125_success_rate']:.0%}) "
            f"| {row['delta_pp']:+.1f} [{lo:+.1f}, {hi:+.1f}] "
            f"| {row['improved']}/{row['regressed']}/{row['unchanged']} "
            f"| {row['mcnemar_exact_p']:.4g} | {row['holm_p_two']:.4g} "
            f"| {row['budget50_cost']['planning_seconds']:.2f}→{row['budget125_cost']['planning_seconds']:.2f} |"
        )
    lines += [
        "",
        "成功率区间为逐预算 Wilson 95% 区间；配对差的区间通过按 episode 重采样 bootstrap 10,000 次计算。检验为双侧 exact McNemar，Holm 校正覆盖两项方法内比较。单 checkpoint、单 seed、50 个起点仍属探索性结果。",
        "",
        "预算50结果沿用 `outputs/round5/phase5_baseline_horizons/` 中已完成的同条件基线；预算125结果保留逐 episode 成功记录、规划事件、checkpoint/cohort SHA、评分审计和 GPU 信息。两预算 runner 的源文件 hash 不同：本次 runner 改动增加预算参数、预算身份记录和对应输出路径，planner 与模型代码未改。",
        "",
        f"逐项身份及成本数据：`{analysis_path.relative_to(ROOT)}`。",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "analysis": str(analysis_path), "report": str(REPORT), "methods": [row["method"] for row in rows]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
