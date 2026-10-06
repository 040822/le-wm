#!/usr/bin/env python3
"""Pair the CoWM 25/25 budget sweep against each Phase 6.2 mode baseline."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import round5_phase6_2 as phase6
from source.common.round3_phase1 import wilson_interval


CONFIG_PATH = ROOT / "config/round5/phase6_2_cowm_mode_budget_sweep.json"
BASELINE_ROOT = ROOT / "outputs/round5/phase6_2"
BOOTSTRAP_SEED_BASE = 6_260_000


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mcnemar_exact(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _holm(rows: list[dict]) -> None:
    ordered = sorted(range(len(rows)), key=lambda index: rows[index]["mcnemar_exact_p"])
    running = 0.0
    count = len(ordered)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * rows[index]["mcnemar_exact_p"])
        running = max(running, adjusted)
        rows[index]["holm_p_40"] = running


def _label(mode: str, guidance: str) -> str:
    return f"{mode}-{ {'none': 'none', 'guided_flow': 'GF', 'post_opt': 'PO'}[guidance] }"


def _report(config: dict, baselines: list[dict], rows: list[dict], prefix_rows: list[dict]) -> str:
    prefix_ok = sum(row["prefix_match"] for row in prefix_rows)
    lines = [
        "# CoWM Reacher 25/25：10 种推理方式的执行预算探索",
        "",
        "## 协议与配对",
        "",
        f"固定 checkpoint `{config['tasks']['reacher']['checkpoint']}`，SHA256 `{config['tasks']['reacher']['checkpoint_sha256']}`；固定 50 个 Reacher 起点，cohort SHA256 `{config['tasks']['reacher']['cohort_sha256']}`。全部条件固定 execute/score=25/25、环境和策略 seed=42/42，只改变总执行预算 60、75、90、110。每个扩展条件与同一推理方式的 Phase 6.2 预算50结果配对。",
        "",
        "预算50基线成功数：",
        "",
        " | ".join(["推理方式", "成功数/50"]),
        " | ".join(["---", "---:"]),
    ]
    for row in baselines:
        lines.append(f"| {row['label']} | {row['baseline_successes']}/50 |")
    lines.extend(
        [
            "",
            f"50步预算与扩展预算的逐 episode action/state 前缀审计：{prefix_ok}/{len(prefix_rows)} 个方式×预算配对完全一致（检查至第50步及成功 episode 的首次成功步）。",
            "非整除预算只执行剩余环境步数，不向上取整。各模式保持原有 P0/P1/P2/P3、none/GF/PO 配置和候选/CEM 设置。",
            "",
            "## 预算结果与预算50配对差",
            "",
            "| 推理方式 | 上限 | 成功数/50 | 成功率 Wilson 95% CI | 配对差 pp [bootstrap 95% CI] | 改善/退化/不变 | McNemar p | Holm p（40项） | 平均规划调用 | 评测秒 | GPU |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        low, high = row["wilson_95_ci"]
        delta_low, delta_high = row["paired_bootstrap_ci95_pp"]
        lines.append(
            f"| {row['label']} | {row['budget']} | {row['successes']}/50 | "
            f"{row['success_rate_percent']:.1f}% [{low:.1f}%, {high:.1f}%] | "
            f"{row['paired_delta_pp']:+.1f} [{delta_low:+.1f}, {delta_high:+.1f}] | "
            f"{row['improved']}/{row['regressed']}/{row['unchanged']} | "
            f"{row['mcnemar_exact_p']:.4g} | {row['holm_p_40']:.4g} | "
            f"{row['mean_planning_calls']:.2f} | {row['evaluation_seconds']:.1f} | {row['gpu']} |"
        )
    lines.extend(
        [
            "",
            "配对差为扩展预算成功指示减去预算50成功指示；区间按相同起点 bootstrap 10,000 次计算。McNemar 使用双侧精确检验，Holm 校正覆盖10种推理方式×4个预算的40项探索性比较。",
            "",
            "## 解释边界",
            "",
            "本实验是单 checkpoint、单推理 seed、50 个固定起点的预算探索。模式间成功率不构成单因素因果比较；每行只与同一推理方式的预算50基线作配对。",
            "此前单独的 P3-none 预算表使用 Table 1 的 environment/policy seed=10042/20042；本表统一使用 Phase 6.2 seed=42/42 及其对应预算50基线，两个 seed 协议的数字不应直接拼接或互换。",
            "",
            f"冻结配置：`config/round5/phase6_2_cowm_mode_budget_sweep.json`。逐条件结果、bootstrap 明细与前缀审计见 `{config['output_root']}/analysis/`。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    config = phase6._load_config(CONFIG_PATH)
    manifest, _, checkpoint, _ = phase6._load_inputs(config)
    baseline_config = phase6._load_config(ROOT / "config/round5/phase6_2.json")
    specs = [
        spec for spec in phase6._condition_specs(config)
        if int(spec["execute_steps"]) == 25 and int(spec["score_steps_env"]) == 25
    ]
    if len(specs) != 10:
        raise ValueError(f"expected all 10 frozen 25/25 CoWM modes, found {len(specs)}")

    output_root = phase6._resolve(config["output_root"])
    budgets = [int(value) for value in config["evaluation"]["budgets"]]
    baseline_by_key: dict[tuple[str, str], dict] = {}
    rows: list[dict] = []
    prefix_rows: list[dict] = []

    for spec in specs:
        mode, guidance = str(spec["mode"]), str(spec["guidance"])
        key = (mode, guidance)
        baseline_path = phase6._condition_path(BASELINE_ROOT, 50, spec)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        phase6._validate_phase6_result(
            baseline, config=baseline_config, manifest=manifest, checkpoint=checkpoint,
            budget=50, spec=spec, require_metadata=True,
        )
        base_parameters = baseline["parameters"]
        if (
            int(base_parameters.get("seed", -1)) != 42
            or int(base_parameters.get("eval_budget", -1)) != 50
            or int(base_parameters.get("execute_steps", -1)) != 25
            or int(base_parameters.get("score_horizon_blocks", -1)) != 5
        ):
            raise ValueError(f"baseline protocol mismatch: {baseline_path}")
        baseline_successes = sum(bool(episode["success"]) for episode in baseline["episodes"])
        baseline_by_key[key] = {
            "payload": baseline,
            "path": str(baseline_path.relative_to(ROOT)),
            "successes": baseline_successes,
        }

        for budget in budgets:
            result_path = phase6._condition_path(output_root, budget, spec)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            phase6._validate_phase6_result(
                result, config=config, manifest=manifest, checkpoint=checkpoint,
                budget=budget, spec=spec, require_metadata=True,
            )
            params = result["parameters"]
            metadata = result["round5_phase6_2"]
            if (
                int(params.get("seed", -1)) != 42
                or int(params.get("eval_budget", -1)) != budget
                or int(params.get("execute_steps", -1)) != 25
                or int(params.get("score_horizon_blocks", -1)) != 5
                or metadata["seed_protocol"]["environment_seed"] != 42
                or metadata["seed_protocol"]["policy_seed"] != 42
            ):
                raise ValueError(f"extension protocol mismatch: {result_path}")
            episodes = result["episodes"]
            if len(episodes) != 50:
                raise ValueError(f"expected 50 episodes: {result_path}")
            if any(int(episode.get("steps_executed", 0)) > budget for episode in episodes):
                raise ValueError(f"episode exceeded total budget: {result_path}")
            base_episodes = baseline["episodes"]
            if [e.get("episode_id") for e in episodes] != [e.get("episode_id") for e in base_episodes]:
                raise ValueError(f"episode order differs from paired baseline: {result_path}")
            mismatches = phase6._trace_prefix_mismatches(baseline, result, prefix_steps=50)

            successes = sum(bool(episode["success"]) for episode in episodes)
            improved = sum(not bool(a["success"]) and bool(b["success"]) for a, b in zip(base_episodes, episodes))
            regressed = sum(bool(a["success"]) and not bool(b["success"]) for a, b in zip(base_episodes, episodes))
            bootstrap_seed = BOOTSTRAP_SEED_BASE + (zlib.crc32(f"{mode}:{guidance}:{budget}".encode()) % 1_000_000)
            paired = phase6._paired_cluster_bootstrap(base_episodes, episodes, seed=bootstrap_seed)
            ci = wilson_interval(successes, len(episodes))
            planning_calls = [int(e.get("episode_replan_count", 0)) for e in episodes]
            prefix_row = {
                "mode": mode,
                "guidance": guidance,
                "budget": budget,
                "prefix_match": not mismatches,
                "mismatch_count": len(mismatches),
                "mismatches": mismatches[:20],
            }
            prefix_rows.append(prefix_row)
            rows.append(
                {
                    "mode": mode,
                    "guidance": guidance,
                    "label": _label(mode, guidance),
                    "budget": budget,
                    "baseline_successes": baseline_successes,
                    "successes": successes,
                    "success_rate": successes / len(episodes),
                    "success_rate_percent": 100.0 * successes / len(episodes),
                    "wilson_95_ci": [100.0 * ci[0], 100.0 * ci[1]],
                    "paired_delta_pp": 100.0 * paired["estimate"],
                    "paired_bootstrap_ci95_pp": [100.0 * value for value in paired["ci95"]],
                    "bootstrap_samples": paired["bootstrap_samples"],
                    "bootstrap_seed": bootstrap_seed,
                    "improved": improved,
                    "regressed": regressed,
                    "unchanged": len(episodes) - improved - regressed,
                    "mcnemar_exact_p": _mcnemar_exact(improved, regressed),
                    "holm_p_40": None,
                    "mean_planning_calls": sum(planning_calls) / len(planning_calls),
                    "evaluation_seconds": float(result["evaluation_seconds"]),
                    "gpu": int(metadata["gpu"]["physical_id"]),
                    "baseline_result": baseline_by_key[key]["path"],
                    "result": str(result_path.relative_to(ROOT)),
                    "prefix_match": not mismatches,
                    "prefix_mismatch_count": len(mismatches),
                }
            )

    if len(rows) != 40:
        raise ValueError(f"expected 40 mode-budget evaluations, found {len(rows)}")
    _holm(rows)
    baselines = [
        {
            "mode": mode,
            "guidance": guidance,
            "label": _label(mode, guidance),
            "baseline_successes": baseline_by_key[(mode, guidance)]["successes"],
            "baseline_result": baseline_by_key[(mode, guidance)]["path"],
        }
        for mode, guidance in [(str(s["mode"]), str(s["guidance"])) for s in specs]
    ]
    analysis_dir = output_root / "analysis"
    analysis = {
        "status": "complete",
        "mode_count": len(specs),
        "budgets": budgets,
        "conditions": len(rows),
        "episodes_per_condition": 50,
        "total_extension_episode_runs": len(rows) * 50,
        "paired_baseline_seed_protocol": {"environment_seed": 42, "policy_seed": 42},
        "prefix_matches": sum(bool(row["prefix_match"]) for row in prefix_rows),
        "prefix_comparisons": len(prefix_rows),
        "bootstrap_samples": phase6.BOOTSTRAP_SAMPLES,
        "holm_comparisons": len(rows),
        "results": rows,
        "baselines": baselines,
        "prefix_audit": prefix_rows,
    }
    _write_json(analysis_dir / "cowm_mode_budget_sweep.json", analysis)
    _write_csv(analysis_dir / "cowm_mode_budget_sweep.csv", rows)
    _write_json(analysis_dir / "cowm_mode_budget_prefix_audit.json", {"comparisons": prefix_rows})
    report = _report(config, baselines, rows, prefix_rows)
    report_path = phase6._resolve(config["report_output"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "conditions": len(rows),
        "prefix_matches": analysis["prefix_matches"],
        "report": str(report_path),
        "analysis": str(analysis_dir / "cowm_mode_budget_sweep.json"),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
