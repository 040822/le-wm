#!/usr/bin/env python3
"""Render the CVPR Table 3 v2 reports from the audited machine summary."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "outputs/cvpr/table3/v2"
SUMMARY_PATH = V2 / "analysis/full.json"
AUDIT_PATH = V2 / "analysis/raw_audit_full.json"
REPORT_DIR = ROOT / "docs/report/cvpr/table3"

TASKS = ("cube", "pusht", "reacher", "tworoom")
TASK_LABEL = {
    "cube": "OGBench-Cube",
    "pusht": "PushT",
    "reacher": "Reacher",
    "tworoom": "TwoRoom",
}
UNITS = {
    "cube": "block position L2（m）",
    "pusht": "max(position L2 / 20 px, wrapped angle / (π/9))",
    "reacher": "joint-position L2（rad）",
    "tworoom": "position L2（environment units）",
}
METHOD_LABEL = {
    "cowm_po": "CoWM-PO",
    "lewm_po": "LeWM-PO",
    "cowm_gf": "CoWM-GF",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def number(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if abs(value) < 5e-13:
        return "0"
    return f"{value:.{digits}g}"


def rate(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "NA"
    return f"{100 * value:.{digits}f}%"


def percentage_points(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "NA"
    return f"{100 * value:.{digits}f} pp"


def interval(value: dict, as_rate: bool = False) -> str:
    if value.get("mean") is None:
        return "NA"
    mean, sd, ci = value["mean"], value.get("sample_std"), value.get("ci95")
    if as_rate:
        center = f"{rate(mean)} ± {percentage_points(sd)}"
        bounds = "NA" if ci is None else f"[{rate(ci[0])}, {rate(ci[1])}]"
    else:
        center = f"{number(mean)} ± {number(sd)}"
        bounds = "NA" if ci is None else f"[{number(ci[0])}, {number(ci[1])}]"
    return f"{center} {bounds}"


def short_ci(value: dict, as_rate: bool = False) -> str:
    if value.get("mean") is None:
        return "NA"
    ci = value.get("ci95")
    if as_rate:
        return f"{rate(value['mean'])} [{rate(ci[0])}, {rate(ci[1])}]" if ci else rate(value["mean"])
    return f"{number(value['mean'])} [{number(ci[0])}, {number(ci[1])}]" if ci else number(value["mean"])


def statistic_block(data: dict) -> str:
    return (
        f"{interval(data)}；有效状态 {data.get('valid_states', 0)}，"
        f"可估计 seed {data.get('estimable_seeds', 0)}/6"
    )


def main_3a(summary: dict, audit: dict) -> str:
    lines = [
        "# CVPR Table 3a v2：同一候选池上的排序准确性",
        "",
        "本文比较固定 CoWM actor 生成的同一组 64 条物理动作候选，由 CoWM-B 与 LeWM 分别评分。主指标评价整个候选池的排序及 top-5 识别，不以任务成功率代替 rerank 准确性。主终点是每条分支原生终止时的真实末状态，或未终止时的 25 步预算终点；提前终止的轨迹不补到 25 步。",
        "",
        "## 主结果",
        "",
        "各单元先逐状态计算，再对可估计的六个评估 seed 等权平均。表中为 seed 均值 ± seed 间样本标准差（ddof=1）[来源 episode 簇配对 bootstrap 95% CI]。`Hit@5` 的随机参照按每个状态真实并列最优候选数精确计算，因此不固定为 5/64。",
        "",
        "| 任务 | 指标 | CoWM-B | LeWM | 随机 Hit@5 参照 |",
        "|---|---|---:|---:|---:|",
    ]
    metrics = (
        ("spearman", "Spearman ρ ↑", False, False),
        ("hit_at_5", "Oracle Hit@5 ↑", True, True),
        ("selection_regret", "Selection regret ↓", False, False),
    )
    for task in TASKS:
        panel = summary["summaries"][task]["3a"]["endpoint"]
        for key, label, as_rate, show_random in metrics:
            a, b = panel["cowm_b"][key], panel["lewm"][key]
            valid = a.get("valid_states", 0)
            pool = panel["cowm_b"]["complete_pool_states"]
            n = f"{valid}/300（完整池 {pool}/300）"
            random_ref = interval(panel["cowm_b"]["random_hit_at_5"], as_rate=True) if show_random else "—"
            if key == "spearman":
                label += f"；ρ 定义数 {n}"
            elif key == "selection_regret":
                label += f"（{UNITS[task]}）"
            lines.append(
                f"| {TASK_LABEL[task]} | {label} | {interval(a, as_rate)} | {interval(b, as_rate)} | {random_ref} |"
            )
    lines.extend(
        [
            "",
            "`ρ` 行中的分母是排序相关可定义的状态数；其他两项使用全部完整 64 候选池。Cube 有 74/300 个状态的 64 条真实代价完全相同，故这些状态的 Spearman ρ 未定义（记 NA、不填零）；Hit@5 和 regret 仍保留这 300 个状态。四任务均无预测代价常数状态。regret 的物理量纲因任务而异，不跨任务直接宏平均。",
            "",
            "## CoWM-B − LeWM 配对差",
            "",
            "正的 ρ/Hit@5 差及负的 regret 差有利于 CoWM-B。p 值来自预设双侧来源 episode 簇配对 bootstrap；Holm 校正覆盖预设 16 项中可估计的 15 项。",
            "",
            "| 任务 | 指标 | CoWM-B − LeWM（均值 ± SD [95% CI]） | 有效状态 | raw p | Holm p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for test in summary["primary_tests"]:
        if not test["comparison"].startswith("3a/"):
            continue
        metric = test["comparison"].rsplit("/", 1)[1]
        label = {"spearman": "Spearman ρ", "hit_at_5": "Oracle Hit@5", "selection_regret": f"Selection regret（{UNITS[test['task']]}）"}[metric]
        estimate = interval(test, as_rate=metric == "hit_at_5")
        lines.append(
            f"| {TASK_LABEL[test['task']]} | {label} | {estimate} | {test['valid_states']}/300 | "
            f"{number(test['p_two_sided'])} | {number(test['p_holm'])} |"
        )
    lines.extend(
        [
            "",
            "Reacher 上 CoWM-B 三项点估计均较好，其中 selection regret 差通过 Holm 校正（Δ = −0.00875，95% CI [−0.0141, −0.00346]，Holm p = 0.0210）；Spearman 的 Holm p = 0.0663，未达到 0.05。PushT 的 Spearman 点估计偏向 CoWM-B，但 Hit@5 与 regret 点估计偏向 LeWM，均无显著的 Holm 校正差异。Cube 三项点估计均偏向 LeWM。TwoRoom 的 CoWM-B 点估计略好，但差异区间均跨零。整体不支持“CoWM-B 在所有任务上都更会 rerank”的概括。",
            "",
            "## 辅助指标与第 25 步敏感性分析",
            "",
            "| 任务 | Hit@1 CoWM-B | Hit@1 LeWM | 均匀选取期望 regret CoWM-B | 均匀选取期望 regret LeWM |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for task in TASKS:
        panel = summary["summaries"][task]["3a"]["endpoint"]
        lines.append(
            f"| {TASK_LABEL[task]} | {interval(panel['cowm_b']['hit_at_1'], True)} | {interval(panel['lewm']['hit_at_1'], True)} | "
            f"{interval(panel['cowm_b']['uniform_expected_regret'])}（{UNITS[task]}） | "
            f"{interval(panel['lewm']['uniform_expected_regret'])}（{UNITS[task]}） |"
        )
    lines.extend(
        [
            "",
            "第 25 步敏感性分析要求同一状态的全部 64 个分支都实际到达第 25 步。有效完整池很少，以下结果只作敏感性记录，不作为主结论。单元格依次为 ρ / Hit@5 / regret；每项为均值 [95% CI]，regret 单位同上。",
            "",
            "| 任务 | 完整池数/300 | CoWM-B：ρ / Hit@5 / regret | LeWM：ρ / Hit@5 / regret |",
            "|---|---:|---:|---:|",
        ]
    )
    for task in TASKS:
        panel = summary["summaries"][task]["3a"]["step25"]
        n = panel["cowm_b"]["complete_pool_states"]
        cells = []
        for scorer in ("cowm_b", "lewm"):
            a = panel[scorer]
            cells.append(
                " / ".join(
                    (
                        short_ci(a["spearman"]),
                        short_ci(a["hit_at_5"], True),
                        f"{short_ci(a['selection_regret'])} ({UNITS[task]})",
                    )
                )
            )
        lines.append(f"| {TASK_LABEL[task]} | {n}/300 | {cells[0]} | {cells[1]} |")
    lines.extend(
        [
            "",
            "第 25 步完整池覆盖为 TwoRoom 0/300、PushT 7/300、Reacher 27/300、Cube 0/300。PushT 与 Reacher 的估计基于极少完整池，不能据此替代变长实际终点的主面板。",
            "",
            "## 实际终点与轨迹长度",
            "",
            "下表聚合每任务 6 × 50 × 64 = 19,200 条已审计分支。检查到的原生终止均为成功终止；未终止分支在 25 步预算处结束。所有分支 `truncated` 标记均为 0。",
            "",
            "| 任务 | 原生 terminated（success） | truncated | 25 步预算终点（running） | 实际长度 <25 | 实际长度 =25 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    by_task = {task: {"endpoints": {}, "reasons": {}, "lengths": {}} for task in TASKS}
    for cell in summary["cells"]:
        t = cell["task"]
        for k, v in cell["endpoint_types"].items():
            by_task[t]["endpoints"][k] = by_task[t]["endpoints"].get(k, 0) + v
        for k, v in cell["termination_reasons"].items():
            by_task[t]["reasons"][k] = by_task[t]["reasons"].get(k, 0) + v
        for k, v in cell["lengths"].items():
            by_task[t]["lengths"][int(k)] = by_task[t]["lengths"].get(int(k), 0) + v
    for task in TASKS:
        x = by_task[task]
        assert x["reasons"].get("success", 0) == x["endpoints"].get("terminated", 0)
        assert x["reasons"].get("running", 0) == x["endpoints"].get("budget_25", 0)
        lt25 = sum(v for length, v in x["lengths"].items() if length < 25)
        eq25 = x["lengths"].get(25, 0)
        lines.append(
            f"| {TASK_LABEL[task]} | {x['endpoints'].get('terminated', 0)} | 0 | "
            f"{x['endpoints'].get('budget_25', 0)} | {lt25} | {eq25} |"
        )
    lines.extend(
        [
            "",
            "主终点是实际观测到的终态代价，因而分支执行长度并不完全相同；它与评分器的固定 25 步预测目标不完全等价。这是 v2 主面板的解释边界。",
            "",
            "## 协议与复现",
            "",
            "Spearman 并列使用平均秩；任一秩向量为常数时记 NA。真实 oracle 是 float64 代价严格等于该状态最小值的全部候选，不加 epsilon；预测代价并列由冻结候选索引打破。Hit@5 随机参照按真实 oracle 集合大小计算。物理代价定义见表中任务单位。",
            "",
            "置信区间使用 10,000 次来源 episode 簇配对 bootstrap（RNG=20261005），跨评估 seed 联合重采样同一来源 episode，并对可估计 seed 等权汇总。seed 是评估抽样，不是训练重复。本 v2 的变长终点及检验族是在观察 v1 后形成的 benchmark 扩展，不应称作独立确认性结果。",
            "",
            f"- 冻结计划：[cvpr_table3_plan_v2.md](../../../plan/cvpr_table3_plan_v2.md)",
            f"- 全量逐状态汇总：[full.json](../../../../outputs/cvpr/table3/v2/analysis/full.json)",
            f"- 全量原始审计：[raw_audit_full.json](../../../../outputs/cvpr/table3/v2/analysis/raw_audit_full.json)",
            f"- 冻结配置：[frozen_config.json](../../../../outputs/cvpr/table3/v2/frozen_config.json)",
            f"- 来源清单：[source_refs.json](../../../../outputs/cvpr/table3/v2/source_refs.json)",
            f"- Table 3 总报告：[cvpr_table3_report.md](cvpr_table3_report.md)",
            "",
        ]
    )
    return "\n".join(lines)


def report_3b(summary: dict) -> str:
    lines = [
        "# CVPR Table 3b v2：真实第 25 步动作修正效果",
        "",
        "本表保留平均真实改善和改善状态比例作为正文指标。每个状态以冻结候选池 index=0 的原动作作基线；`Δc = c(original) − c(updated)`，正值表示代价降低。改善状态定义为 `Δc > 1e-6`。主结果只纳入原动作与更新动作都实际到达第 25 步的配对状态；提前终止不会补值，NA 不是零改善。",
        "",
        "每项为按 seed 等权的均值 ± seed 间样本标准差（ddof=1）[来源 episode 簇配对 bootstrap 95% CI]。百分比中的标准差以百分点（pp）表示。方法与其随机对照各自使用有效状态集，因此两列不可直接相减；方法对匹配随机扰动的正式配对差单列报告。",
        "",
    ]
    for task in TASKS:
        table = summary["summaries"][task]["3b"]
        lines.extend(
            [
                f"## {TASK_LABEL[task]}",
                "",
                f"代价单位：{UNITS[task]}。",
                "",
                "| 方法 | 平均真实改善 Δc | 改善状态比例 | 方法有效状态；可估 seed | 幅度匹配随机 Δc | 随机改善比例 | 随机有效状态；可估 seed |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in ("cowm_po", "lewm_po", "cowm_gf"):
            x = table[method]
            imp, improved = x["improvement"], x["improved"]
            rand, rand_imp = x["random_mean_improvement"], x["random_improved_fraction"]
            method_n = f"{imp['valid_states']}/300；{imp['estimable_seeds']}/6"
            random_n = f"{rand['valid_states']}/300；{rand['estimable_seeds']}/6"
            lines.append(
                f"| {METHOD_LABEL[method]} | {interval(imp)} | {interval(improved, True)} | {method_n} | "
                f"{interval(rand)} | {interval(rand_imp, True)} | {random_n} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 有效样本与缺失原因",
            "",
            "以下三类对每种方法构成 300 个状态的互斥分解：原动作在第 25 步前终止/缺失；原动作可用但更新分支提前终止/缺失；有效配对。Cube 的更新分支也全部提前结束，但由于原动作的 300 个状态均没有真实第 25 步，按预设优先级归入“原动作不可用”。",
            "",
            "| 任务 | 方法 | 原动作不可用 | 更新分支不可用 | 有效配对 | 等 RMS 随机有效状态 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for task in TASKS:
        table = summary["summaries"][task]["3b"]
        for method in ("cowm_po", "lewm_po", "cowm_gf"):
            x = table[method]
            reasons = x["missing_reasons"]
            baseline = reasons.get("baseline_early_or_missing25", 0)
            update = reasons.get("updated_early_or_missing25", 0)
            valid = x["improvement"]["valid_states"]
            random_n = x["random_mean_improvement"]["valid_states"]
            lines.append(
                f"| {TASK_LABEL[task]} | {METHOD_LABEL[method]} | {baseline} | {update} | {valid} | {random_n} |"
            )
    lines.extend(
        [
            "",
            "随机对照为每种方法分别匹配的五个冻结随机方向；先在状态内平均可用方向，再以状态/来源 episode 做统计，方向本身不当作独立状态。样本列中的随机有效状态数可与方法有效状态数不同。",
            "",
            "## CoWM-PO 与幅度匹配随机扰动：预设配对检验",
            "",
            "该主检验在 CoWM-PO 与其幅度匹配随机扰动都有效的共同状态上比较 Δc。计划族共 16 项，其中 15 项可估计；表中 Holm p 对可估计项校正。",
            "",
            "| 任务 | CoWM-PO − 匹配随机 Δc（均值 ± SD [95% CI]） | 配对状态 | 可估 seed | raw p | Holm p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for test in summary["primary_tests"]:
        if not test["comparison"].startswith("3b/"):
            continue
        if test.get("mean") is None:
            estimate = "NA（无有效配对）"
        else:
            estimate = interval(test)
        lines.append(
            f"| {TASK_LABEL[test['task']]} | {estimate} | {test['valid_states']}/300 | {test['estimable_seeds']}/6 | "
            f"{number(test['p_two_sided'])} | {number(test['p_holm'])} |"
        )
    lines.extend(
        [
            "",
            "只有 Reacher 的 CoWM-PO 相对匹配随机扰动在 Holm 校正后仍为正（Δ = +0.01536，95% CI [+0.00911, +0.02136]，Holm p = 0.00150）。PushT 的区间跨零；TwoRoom 只有 4 个配对状态、来自 2 个 seed；Cube 没有可估计配对。",
            "",
            "## CoWM-PO 与 LeWM-PO 共同有效状态（探索性）",
            "",
            "只有共同有效状态可以支持两方法差异比较。此分析未纳入预设主检验族，不能作确认性结论。",
            "",
            "| 任务 | CoWM-PO − LeWM-PO 的 Δc 差 [95% CI] | 共同状态 | 可估 seed | raw p（未校正） |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for task in TASKS:
        x = summary["summaries"][task]["3b"]["cowm_minus_lewm_common_improvement_exploratory"]
        lines.append(
            f"| {TASK_LABEL[task]} | {short_ci(x)} | {x['valid_states']}/300 | {x['estimable_seeds']}/6 | "
            f"{number(x.get('p_two_sided'))} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "CoWM-PO 在 Reacher 的平均改善与改善状态比例均为正，且其相对幅度匹配随机扰动的预设配对差通过 Holm 校正；这个结果不自动推出它优于 LeWM-PO。PushT 的 CoWM-PO 点估计为正，但不确定区间跨零；TwoRoom 的 CoWM-PO 估计基于极少配对。Cube 没有第 25 步有效配对，因此正文必须保留 NA。总体结果不支持跨任务的普遍改善结论。",
            "",
            "评估 seed 是评估抽样，不是训练重复。bootstrap 按来源 episode 聚类，联合跨 seed 重采样并对可估 seed 等权汇总；区间为 10,000 次 percentile bootstrap（RNG=20261005）。",
            "",
            f"- 冻结计划：[cvpr_table3_plan_v2.md](../../../plan/cvpr_table3_plan_v2.md)",
            f"- 全量逐状态汇总：[full.json](../../../../outputs/cvpr/table3/v2/analysis/full.json)",
            f"- 全量原始审计：[raw_audit_full.json](../../../../outputs/cvpr/table3/v2/analysis/raw_audit_full.json)",
            f"- Table 3 总报告：[cvpr_table3_report.md](cvpr_table3_report.md)",
            "",
        ]
    )
    return "\n".join(lines)


def total_report(summary: dict, audit: dict) -> str:
    tests = summary["primary_tests"]
    significant = [x for x in tests if x.get("p_holm") is not None and x["p_holm"] < 0.05]
    sig_labels = []
    for test in significant:
        if test["comparison"].startswith("3a/"):
            sig_labels.append(f"{TASK_LABEL[test['task']]} 3a selection regret")
        else:
            sig_labels.append(f"{TASK_LABEL[test['task']]} 3b CoWM-PO 对匹配随机扰动的平均改善")
    sig_text = "；".join(sig_labels) or "无"
    return "\n".join(
        [
            "# CVPR Table 3 v2：排序准确性与真实动作修正",
            "",
            "Table 3a 衡量同一固定候选池上的 rerank 准确性；Table 3b 衡量第 25 步真实代价的配对动作修正。全量六个评估 seed、四项任务的 24 个 task–seed 单元已完成并通过逐候选原始审计。此报告对应冻结的 v2 方案；Table 3c 沿用已完成的 v1 探针，不在本轮重算。",
            "",
            "## 主要发现",
            "",
            "- **3a 没有跨任务的统一胜者。** Reacher 的 CoWM-B 选择 regret 较低，配对差经 Holm 校正后仍显著；其 Spearman 优势未通过校正。PushT 指标方向不一致，Cube 点估计偏向 LeWM，TwoRoom 差异小且区间跨零。详情见 [Table 3a 报告](cvpr_table3a_ranking_report.md)。",
            "- **3b 的证据集中在 Reacher。** CoWM-PO 相对其幅度匹配随机扰动的配对平均改善为 +0.01536（95% CI [+0.00911, +0.02136]，Holm p=0.00150）。PushT 区间跨零，TwoRoom 仅 4 个 CoWM-PO 有效状态，Cube 无实际第 25 步有效配对。详情见 [Table 3b 报告](cvpr_table3b_refinement_report.md)。",
            "- **Table 3a 主分母是 300 个完整 64 候选池/任务。** Cube 的 74 个真实常数状态不定义 Spearman，因此该任务 ρ 的分母为 226；其他指标仍使用全部 300 个状态。",
            "- **Table 3b 保留平均真实改善与改善状态比例。** 只使用原动作和更新动作都实际到达第 25 步的状态；NA 不填零。以状态数和可估计 seed 数明确呈现提前终止造成的覆盖差异。",
            "",
            "## 设计与统计",
            "",
            "六个评估 seed（42、100、2026、3407、1234、4444）各取 50 个源状态出现；每任务共 300 个状态。3a 对同一 CoWM actor 的 64 个物理候选分别以 CoWM-B 和 LeWM 评分，使用 Spearman ρ、Oracle Hit@5 和 selection regret。3b 从候选 index=0 的原动作开始，报告 `Δc = c(original) − c(updated)` 及 `P(Δc > 1e-6)`，并对每种方法分别提供 RMS 幅度匹配的随机方向参照。",
            "",
            "3a 主终点取原生终止末状态或 25 步预算末状态，记录实际长度；提前终止分支不伪造第 25 步。已审计分支总数为 76,800，轨迹 primitive steps 共 1,351,242。原生 terminated 终点全部记录为 success；没有 truncated 终点。3b 另按计划使用真实第 25 步配对后果，因此与 3a 的变长终点分开解释。",
            "",
            "报告的均值为对可估计 seed 均值等权计算，显示 seed 间样本标准差。95% CI 和双侧 p 值采用 10,000 次来源 episode 簇配对 bootstrap（RNG=20261005），同一来源 episode 跨评估 seed 联合重采样。16 项预设主检验中 15 项可估，Holm 校正应用于可估计子集；唯一不可估计项为 Cube 的 3b CoWM-PO 对匹配随机配对差（无有效配对）。通过 Holm 校正的主检验：" + sig_text + "。",
            "",
            "3a 变长终点与其检验族是在观察 v1 后制定的 benchmark 扩展，不构成独立确认性结果。单个 checkpoint 的评估 seed 不是训练重复。本轮不重训模型。",
            "",
            "## 证据文件与归档",
            "",
            f"- [冻结实验计划](../../../plan/cvpr_table3_plan_v2.md)",
            f"- [执行责任补充：main-agent-only](../../../plan/cvpr_table3_plan_v2_execution_addendum_20261006.md)",
            f"- [全量逐状态统计与检验](../../../../outputs/cvpr/table3/v2/analysis/full.json)",
            f"- [24 单元原始审计结果](../../../../outputs/cvpr/table3/v2/analysis/raw_audit_full.json)",
            f"- [候选来源与哈希清单](../../../../outputs/cvpr/table3/v2/source_refs.json)",
            f"- [冻结配置](../../../../outputs/cvpr/table3/v2/frozen_config.json)",
            f"- [旧版 v1 总报告、3a、3b 归档](archived/README.md)",
            f"- [沿用 v1 的 Table 3c 报告](cvpr_table3c_probe_report.md)",
            f"- [全量原始验收记录](../../../../outputs/cvpr/table3/v2/analysis/main_agent_final_acceptance.json)",
            "",
        ]
    )


def main() -> None:
    summary, audit = read(SUMMARY_PATH), read(AUDIT_PATH)
    if summary.get("processed_cells") != 24 or summary.get("missing_selected_cells"):
        raise SystemExit("full analysis is incomplete")
    if audit.get("status") != "pass" or audit.get("cells_checked") != 24:
        raise SystemExit("full raw audit is incomplete")
    if audit.get("candidate_rows_crosschecked") != 76800:
        raise SystemExit("unexpected raw candidate row count")
    if len(summary.get("primary_tests", [])) != 16:
        raise SystemExit("unexpected primary test family size")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    reports = {
        "cvpr_table3a_ranking_report.md": main_3a(summary, audit),
        "cvpr_table3b_refinement_report.md": report_3b(summary),
        "cvpr_table3_report.md": total_report(summary, audit),
    }
    for name, body in reports.items():
        path = REPORT_DIR / name
        path.write_text(body, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} ({len(body)} chars)")


if __name__ == "__main__":
    main()
