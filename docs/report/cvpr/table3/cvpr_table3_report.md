# CVPR Table 3 v2：排序准确性与真实动作修正

Table 3a 衡量同一固定候选池上的 rerank 准确性；Table 3b 衡量第 25 步真实代价的配对动作修正。全量六个评估 seed、四项任务的 24 个 task–seed 单元已完成并通过逐候选原始审计。此报告对应冻结的 v2 方案；Table 3c 沿用已完成的 v1 探针，不在本轮重算。

## 主要发现

- **3a 没有跨任务的统一胜者。** Reacher 的 CoWM-B 选择 regret 较低，配对差经 Holm 校正后仍显著；其 Spearman 优势未通过校正。PushT 指标方向不一致，Cube 点估计偏向 LeWM，TwoRoom 差异小且区间跨零。详情见 [Table 3a 报告](cvpr_table3a_ranking_report.md)。
- **3b 的证据集中在 Reacher。** CoWM-PO 相对其幅度匹配随机扰动的配对平均改善为 +0.01536（95% CI [+0.00911, +0.02136]，Holm p=0.00150）。PushT 区间跨零，TwoRoom 仅 4 个 CoWM-PO 有效状态，Cube 无实际第 25 步有效配对。详情见 [Table 3b 报告](cvpr_table3b_refinement_report.md)。
- **Table 3a 主分母是 300 个完整 64 候选池/任务。** Cube 的 74 个真实常数状态不定义 Spearman，因此该任务 ρ 的分母为 226；其他指标仍使用全部 300 个状态。
- **Table 3b 保留平均真实改善与改善状态比例。** 只使用原动作和更新动作都实际到达第 25 步的状态；NA 不填零。以状态数和可估计 seed 数明确呈现提前终止造成的覆盖差异。

## 设计与统计

六个评估 seed（42、100、2026、3407、1234、4444）各取 50 个源状态出现；每任务共 300 个状态。3a 对同一 CoWM actor 的 64 个物理候选分别以 CoWM-B 和 LeWM 评分，使用 Spearman ρ、Oracle Hit@5 和 selection regret。3b 从候选 index=0 的原动作开始，报告 `Δc = c(original) − c(updated)` 及 `P(Δc > 1e-6)`，并对每种方法分别提供 RMS 幅度匹配的随机方向参照。

3a 主终点取原生终止末状态或 25 步预算末状态，记录实际长度；提前终止分支不伪造第 25 步。已审计分支总数为 76,800，轨迹 primitive steps 共 1,351,242。原生 terminated 终点全部记录为 success；没有 truncated 终点。3b 另按计划使用真实第 25 步配对后果，因此与 3a 的变长终点分开解释。

报告的均值为对可估计 seed 均值等权计算，显示 seed 间样本标准差。95% CI 和双侧 p 值采用 10,000 次来源 episode 簇配对 bootstrap（RNG=20261005），同一来源 episode 跨评估 seed 联合重采样。16 项预设主检验中 15 项可估，Holm 校正应用于可估计子集；唯一不可估计项为 Cube 的 3b CoWM-PO 对匹配随机配对差（无有效配对）。通过 Holm 校正的主检验：Reacher 3a selection regret；Reacher 3b CoWM-PO 对匹配随机扰动的平均改善。

3a 变长终点与其检验族是在观察 v1 后制定的 benchmark 扩展，不构成独立确认性结果。单个 checkpoint 的评估 seed 不是训练重复。本轮不重训模型。

## 证据文件与归档

- [冻结实验计划](../../../plan/cvpr_table3_plan_v2.md)
- [执行责任补充：main-agent-only](../../../plan/cvpr_table3_plan_v2_execution_addendum_20261006.md)
- [全量逐状态统计与检验](../../../../outputs/cvpr/table3/v2/analysis/full.json)
- [24 单元原始审计结果](../../../../outputs/cvpr/table3/v2/analysis/raw_audit_full.json)
- [候选来源与哈希清单](../../../../outputs/cvpr/table3/v2/source_refs.json)
- [冻结配置](../../../../outputs/cvpr/table3/v2/frozen_config.json)
- [旧版 v1 总报告、3a、3b 归档](archived/README.md)
- [沿用 v1 的 Table 3c 报告](cvpr_table3c_probe_report.md)
- [全量原始验收记录](../../../../outputs/cvpr/table3/v2/analysis/main_agent_final_acceptance.json)
