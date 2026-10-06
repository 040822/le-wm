#!/usr/bin/env python3
"""Summarize the 60/75/90/110-step Reacher baseline budget sweep."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/round5/phase6_2_baseline_budget_sweep.json"
REPORT = ROOT / "docs/report/round5/round5_phase6_2_baseline_budget_sweep.md"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_key(row: dict) -> tuple[int, int]:
    episode = row.get("dataset_episode", row.get("episode_id"))
    return int(episode), int(row["start_step"])


def _wilson(successes: int, n: int) -> list[float]:
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [100 * (center - radius), 100 * (center + radius)]


def _paired(baseline: dict, extended: dict, seed: int) -> dict:
    left = {_episode_key(row): bool(row["success"]) for row in baseline["episodes"]}
    right = {_episode_key(row): bool(row["success"]) for row in extended["episodes"]}
    if len(left) != 50 or left.keys() != right.keys():
        raise ValueError("baseline and extended budget must share the 50 episode starts")
    differences = np.asarray(
        [int(right[key]) - int(left[key]) for key in sorted(left)], dtype=np.int8
    )
    improved = int(np.count_nonzero(differences > 0))
    regressed = int(np.count_nonzero(differences < 0))
    discordant = improved + regressed
    p_value = (
        1.0
        if discordant == 0
        else min(
            1.0,
            2 * sum(math.comb(discordant, i) for i in range(min(improved, regressed) + 1))
            / (2**discordant),
        )
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), (10_000, len(differences)))
    interval = np.quantile(differences[indices].mean(axis=1) * 100, [0.025, 0.975])
    baseline_successes = sum(left.values())
    extended_successes = sum(right.values())
    return {
        "baseline_successes": baseline_successes,
        "baseline_success_rate": baseline_successes / 50,
        "baseline_wilson95_pp": _wilson(baseline_successes, 50),
        "successes": extended_successes,
        "success_rate": extended_successes / 50,
        "wilson95_pp": _wilson(extended_successes, 50),
        "delta_pp": float(differences.mean() * 100),
        "paired_bootstrap95_pp": [float(value) for value in interval],
        "improved": improved,
        "regressed": regressed,
        "unchanged": int(np.count_nonzero(differences == 0)),
        "mcnemar_exact_p": p_value,
    }


def _cost(result: dict, method: str) -> dict:
    if method == "cowm":
        events = result.get("planning_events", [])
        seconds = sum(float(event.get("wall_seconds", 0.0)) for event in events)
        calls = int(result.get("batch_replan_count", len(events)))
    else:
        meta = result["horizon_ablation"]
        events = meta["planning_events"]
        seconds = sum(float(event["planning_seconds"]) for event in events)
        calls = len(events)
    return {
        "planning_events": len(events),
        "planning_calls": calls,
        "planning_seconds": seconds,
        "evaluation_seconds": float(result["evaluation_seconds"]),
    }


def _metadata(result: dict, method: str) -> dict:
    if method == "cowm":
        return result["phase6_2_baseline_sweep"]
    return result["horizon_ablation"]


def _check_condition(result: dict, method: str, budget: int, config: dict) -> dict:
    metadata = _metadata(result, method)
    params = result["parameters"]
    if result.get("status") != "ok" or len(result.get("episodes", [])) != 50:
        raise ValueError(f"incomplete {method} result at budget {budget}")
    if int(params.get("eval_budget", -1)) != budget:
        raise ValueError(f"{method} result has incorrect eval_budget at {budget}")
    if int(metadata.get("execute_steps", -1)) != 25 or int(metadata.get("score_steps", -1)) != 25:
        raise ValueError(f"{method} is not evaluated at 25/25 for budget {budget}")
    expected_method = config["methods"][method]
    if metadata.get("checkpoint_sha256") != expected_method["checkpoint_sha256"]:
        raise ValueError(f"{method} checkpoint SHA mismatch at budget {budget}")
    if metadata.get("cohort_sha256") != config["cohort"]["sha256"]:
        raise ValueError(f"{method} cohort SHA mismatch at budget {budget}")
    if method == "cowm":
        if result.get("round4_mode") != "P3":
            raise ValueError("CoWM sweep must use the frozen P3 Selection baseline")
        if int(params.get("horizon", -1)) != 5 or int(params.get("receding_horizon", -1)) != 5:
            raise ValueError("CoWM proposal/execution horizon differs from 25/25")
        if int(params.get("score_horizon_blocks", -1)) != 5:
            raise ValueError("CoWM scoring horizon differs from 25 steps")
        if metadata.get("budget") != budget:
            raise ValueError("CoWM sweep metadata budget mismatch")
        seed_offsets = config["seed_policy"]
        cohort_seed = int(config["cohort"]["seed"])
        if metadata.get("cohort_seed") != cohort_seed:
            raise ValueError("CoWM cohort seed mismatch")
        if metadata.get("environment_seed") != cohort_seed + int(seed_offsets["environment_seed_offset"]):
            raise ValueError("CoWM environment seed does not match Table 1 offset")
        if metadata.get("policy_seed") != cohort_seed + int(seed_offsets["policy_seed_offset"]):
            raise ValueError("CoWM policy seed does not match Table 1 offset")
        if int(params.get("environment_seed", -1)) != metadata["environment_seed"]:
            raise ValueError("CoWM evaluator environment seed differs from sweep metadata")
        if int(params.get("policy_seed", -1)) != metadata["policy_seed"]:
            raise ValueError("CoWM evaluator policy seed differs from sweep metadata")
    else:
        if int(metadata.get("eval_budget", -1)) != budget:
            raise ValueError(f"{method} sweep metadata budget mismatch")
        plan = params.get("plan_config", {})
        if plan != {"horizon": 5, "receding_horizon": 5, "action_block": 5}:
            raise ValueError(f"{method} plan configuration differs from 25/25")
    return metadata


def main() -> None:
    config = _read(CONFIG)
    out = ROOT / config["output_root"]
    budgets = [int(value) for value in config["protocol"]["budgets"]]
    rows = []
    baselines = {}
    baseline_metadata = {}
    for method in ("cowm", "lewm", "leflow"):
        method_config = config["methods"][method]
        baseline_path = ROOT / method_config["baseline_result"]
        baseline = _read(baseline_path)
        if baseline.get("status") != "ok" or len(baseline.get("episodes", [])) != 50:
            raise ValueError(f"invalid 50-step baseline for {method}: {baseline_path}")
        if int(baseline.get("parameters", {}).get("eval_budget", -1)) != 50:
            raise ValueError(f"baseline budget is not 50 for {method}")
        baseline_episode_keys = {_episode_key(row) for row in baseline["episodes"]}
        if len(baseline_episode_keys) != 50:
            raise ValueError(f"duplicate episodes in {method} baseline")
        if method == "cowm":
            if baseline.get("round4_mode") != "P3":
                raise ValueError("CoWM baseline is not P3 Selection")
            expected_environment_seed = int(config["cohort"]["seed"]) + int(config["seed_policy"]["environment_seed_offset"])
            expected_policy_seed = int(config["cohort"]["seed"]) + int(config["seed_policy"]["policy_seed_offset"])
            if int(baseline["parameters"].get("environment_seed", -1)) != expected_environment_seed:
                raise ValueError("CoWM baseline environment seed differs from the frozen Table 1 offset")
            if int(baseline["parameters"].get("policy_seed", -1)) != expected_policy_seed:
                raise ValueError("CoWM baseline policy seed differs from the frozen Table 1 offset")
            checkpoint = ROOT / method_config["checkpoint"]
            if _sha256(checkpoint) != method_config["checkpoint_sha256"]:
                raise ValueError("CoWM baseline checkpoint SHA mismatch")
            baseline_cohort_sha = baseline.get("cohort_sha256")
        else:
            base_meta = baseline["horizon_ablation"]
            if base_meta["checkpoint_sha256"] != method_config["checkpoint_sha256"]:
                raise ValueError(f"{method} baseline checkpoint SHA mismatch")
            baseline_cohort_sha = base_meta["cohort_sha256"]
        if baseline_cohort_sha != config["cohort"]["sha256"]:
            raise ValueError(f"{method} baseline cohort SHA mismatch")
        baselines[method] = baseline
        baseline_metadata[method] = {
            "path": str(baseline_path.relative_to(ROOT)),
            "checkpoint_sha256": method_config["checkpoint_sha256"],
            "cohort_sha256": baseline_cohort_sha,
            "successes": sum(bool(item["success"]) for item in baseline["episodes"]),
        }

        for budget in budgets:
            result_path = out / f"budget_{budget}" / method / "25_25" / "result.json"
            result = _read(result_path)
            metadata = _check_condition(result, method, budget, config)
            comparison = _paired(
                baseline,
                result,
                seed=60_000 + 100 * budget + {"cowm": 1, "lewm": 2, "leflow": 3}[method],
            )
            rows.append({
                "method": method,
                "label": method_config["label"],
                "budget": budget,
                "result_path": str(result_path.relative_to(ROOT)),
                "checkpoint_sha256": method_config["checkpoint_sha256"],
                "cohort_sha256": config["cohort"]["sha256"],
                **comparison,
                "cost": _cost(result, method),
                "gpu": metadata.get("gpu", metadata.get("gpu_before", {}).get("physical_id")),
                "gpu_before": metadata.get("gpu_before"),
                "gpu_after": metadata.get("gpu_after"),
                "runner_sha256": metadata.get("runner_sha256"),
                "config_sha256": metadata.get("config_sha256"),
            })

    # One exploratory Holm family: all 12 method-by-budget comparisons vs 50.
    order = sorted(range(len(rows)), key=lambda index: rows[index]["mcnemar_exact_p"])
    previous = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(rows) - rank) * rows[index]["mcnemar_exact_p"])
        previous = max(previous, adjusted)
        rows[index]["holm_p_12"] = previous

    analysis_dir = out / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = analysis_dir / "budget_curve.json"
    csv_path = analysis_dir / "budget_curve.csv"
    analysis_path.write_text(json.dumps({
        "status": "complete",
        "config": str(CONFIG.relative_to(ROOT)),
        "baselines": baseline_metadata,
        "conditions": rows,
    }, indent=2, ensure_ascii=False) + "\n")
    columns = [
        "method", "label", "budget", "baseline_successes", "successes",
        "success_rate", "wilson95_pp", "delta_pp", "paired_bootstrap95_pp",
        "improved", "regressed", "unchanged", "mcnemar_exact_p", "holm_p_12",
        "planning_calls", "planning_seconds", "evaluation_seconds", "gpu",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            value = {**row, **row["cost"]}
            writer.writerow({column: value.get(column) for column in columns})

    lines = [
        "# Reacher：CoWM、LeWM、LeFlow 的执行预算探索",
        "",
        "## 协议",
        "",
        "三种方法均固定 25/25 执行与评分，每个预算评测相同的 50 个 Reacher legacy 起点（cohort seed=42），目标偏移25步。LEWM/LEFlow 使用 evaluation/policy seed=42；CoWM 沿用 Table 1 的环境 seed=10042、policy seed=20042。模型和规划设置保持不变，只改变每回合环境步数上限：60、75、90、110。非25整数倍预算只执行剩余步数，不向上取整。",
        "",
        "CoWM 使用 Table 1 的标准 CoWM-Selection（P3，64候选，Euler action-flow step=2）；LeWM 使用原生 CEM；LeFlow 使用原生 latent-flow sampler。三种方法分别与同 checkpoint/cohort 的预算50基线比较。各模型横向差异不作单变量因果解释。",
        "",
        "## 成功率与预算50配对差",
        "",
        "| 方法 | 上限 | 成功数/50 | 成功率 Wilson 95% CI | 相对50步差 pp [配对 bootstrap 95% CI] | 改善/退化/不变 | McNemar p | Holm p（12项） | 规划事件数 | 规划秒 | 评测秒 | GPU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lo, hi = row["wilson95_pp"]
        delta_lo, delta_hi = row["paired_bootstrap95_pp"]
        cost = row["cost"]
        lines.append(
            f"| {row['label']} | {row['budget']} | {row['successes']}/50 "
            f"| {row['success_rate']:.1%} [{lo:.1f}%, {hi:.1f}%] "
            f"| {row['delta_pp']:+.1f} [{delta_lo:+.1f}, {delta_hi:+.1f}] "
            f"| {row['improved']}/{row['regressed']}/{row['unchanged']} "
            f"| {row['mcnemar_exact_p']:.4g} | {row['holm_p_12']:.4g} "
            f"| {cost['planning_calls']} | {cost['planning_seconds']:.2f} "
            f"| {cost['evaluation_seconds']:.2f} | {row['gpu']} |"
        )
    lines += [
        "",
        "差值定义为该预算成功指示减去预算50基线成功指示；区间按 episode 配对 bootstrap 10,000 次计算。McNemar 为双侧精确检验；Holm 校正覆盖3个方法×4个预算的12项探索性比较。相同起点用于配对，但本扫参未审计不同预算下逐步动作/状态前缀是否完全一致。",
        "",
        "基线来源：CoWM 为 Table 1 Reacher seed=42、P3 main 的既有预算50结果；LeWM、LeFlow 为 Round 5 baseline horizons 的 25/25、预算50结果。三份基线和本次扩展均使用 cohort SHA256 `ff4f26ad3fd809e59e1b748c7c93c33a8e51b48e13d2505e89e5909d6121bf45`。",
        "",
        "规划事件数是 evaluator 记录的批次级事件数，不等于跨 episode 求和的每 episode 规划调用总数。两轮初始 CoWM 结果因 seed 未正确传入 evaluator 而失效，分别归档于 `outputs/round5/phase6_2_baseline_budgets/invalidated/cowm_environment42_policy42/` 和 `outputs/round5/phase6_2_baseline_budgets/invalidated/cowm_environment10042_policy10042/`；本报告只纳入 evaluator 参数确认使用 environment/policy seed=10042/20042 的正式重跑。",
        "",
        f"冻结配置：`{CONFIG.relative_to(ROOT)}`。JSON/CSV 明细：`{analysis_path.relative_to(ROOT)}`、`{csv_path.relative_to(ROOT)}`。",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "conditions": len(rows), "analysis": str(analysis_path), "report": str(REPORT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
