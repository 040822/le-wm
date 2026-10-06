# Table 3 论文展示版：Reacher 与 PushT

本报告根据 Table 3 v2 整理，仅展示 Reacher 与 PushT，面向论文正文排版。任务选择依据是作者提出的 TwoRoom/Cube 性能已接近瓶颈；正式论文中，这一饱和判断应由主任务性能表支撑。Table 3 本身提供的直接证据是：这两个任务大量提前成功终止，导致第 25 步配对覆盖不足。这里是现有结果的展示子集，未重新开展实验或修改检验族。

## Table 3a：固定候选池的排序与选择

每任务 300 个状态，每状态 64 条相同的 CoWM actor 候选。表内为六个评估 seed 的均值 ± 样本标准差；百分比的标准差单位为百分点。粗体表示任务内较优点估计，† 表示 CoWM-B 相对 LeWM 的该指标差通过原检验族 Holm 校正（p < 0.05）。

| 任务 | 评分器 | Spearman ρ ↑ | Oracle Hit@5 (%) ↑ | Selection regret ↓ |
|---|---|---:|---:|---:|
| Reacher | LeWM | 0.204 ± 0.042 | 9.0 ± 3.7 | 0.0671 ± 0.0048 |
| Reacher | CoWM-B | **0.255 ± 0.037** | **11.7 ± 4.8** | **0.0583 ± 0.0087** † |
| PushT | LeWM | 0.056 ± 0.022 | **12.3 ± 5.0** | **0.2587 ± 0.0273** |
| PushT | CoWM-B | **0.069 ± 0.027** | 11.7 ± 2.3 | 0.2800 ± 0.0484 |

**表注。** ρ 衡量预测代价与真实代价的秩相关；Hit@5 衡量预测前五条中是否包含真实最优候选，其随机参照在两个任务上均为 7.8%；regret 为预测 top-1 的真实代价减去候选池最小真实代价。Reacher 代价为关节位置 L2（rad），PushT 为 max(position L2 / 20 px, wrapped angle / (π/9))，其中位置项覆盖智能体及物块 XY。不同任务的 regret 不直接比较。真实后果取原生终止末状态或第 25 步预算末状态；候选实际执行时长可能不同，与评分器固定 25 步预测目标不完全等价。

**解读。** Reacher 的 CoWM-B 三项点估计都较优，selection regret 相对 LeWM 约降低 13.0%，且配对差通过 Holm 校正。PushT 的指标方向不一致：CoWM-B 整体秩相关较高，LeWM 头部命中率和选择 regret 较好，三项差异均未通过校正。粗体本身不代表统计显著。

## Table 3b：真实动作修正效果

Δc = c(original) − c(updated)，越大越好；改善状态比例为 P(Δc > 10⁻⁶)。仅纳入原动作和更新动作都实际到达第 25 步的状态。所有展示行均有六个可估计评估 seed；N 是有效状态数，而非训练重复数。由于方法间有效集合不同，本表不对方法均值加“最优”粗体。

| 任务 | 修正方法 | 平均真实改善 Δc ↑ | 改善状态比例 (%) ↑ | 有效 N / 300 |
|---|---|---:|---:|---:|
| Reacher | LeWM-PO | +0.0103 ± 0.0053 | 59.9 ± 6.2 | 166 |
| Reacher | CoWM-PO | +0.0154 ± 0.0042 | 67.7 ± 5.9 | 164 |
| Reacher | CoWM-GF | +0.0127 ± 0.0026 | 72.6 ± 6.7 | 164 |
| PushT | LeWM-PO | −0.0225 ± 0.0397 | 43.1 ± 19.1 | 46 |
| PushT | CoWM-PO | +0.0363 ± 0.0916 | 68.6 ± 29.9 | 39 |
| PushT | CoWM-GF | +0.0229 ± 0.0460 | 63.7 ± 28.6 | 41 |

**表注。** 代价单位同 3a。PO 使用 K=2、步长 0.01、动作 RMS 上限 0.2；GF 使用 S=2，每个 Euler 步引导一次，总 K=2。提前终止不补成第 25 步状态。改善比例不是任务成功率；各方法均值不能直接相减为方法优势。Reacher 原动作可到第 25 步的状态为 183/300，PushT 为 55/300，进一步要求更新分支可用后得到表内 N。PushT 有效覆盖较低，平均改善的区间均跨零。

### 配对证据：CoWM-PO 是否优于同幅度随机扰动？

这张小表建议紧接 3b，或放入表注/补充材料。它在共同有效状态上比较，直接呈现正文主张的统计依据。

| 任务 | CoWM-PO − 匹配随机的 Δc ↑ | 95% CI | 配对 N / 300 | Holm p |
|---|---:|---:|---:|---:|
| Reacher | **+0.01536** | **[+0.00911, +0.02136]** | 164 | **0.0015** |
| PushT | +0.04340 | [−0.00788, +0.09499] | 39 | 1.0000 |

随机参照对每种方法分别匹配动作 RMS 幅度，每状态五个冻结随机方向；先在状态内汇总，不将方向当作独立状态。上述配对差是在共同状态上重算，不能用各自主表均值相减替代。

## 补充材料：随机参照与方法间配对比较

| 任务 | 匹配对象 | 随机平均 Δc | 随机改善比例 (%) | 随机有效 N / 300 |
|---|---|---:|---:|---:|
| Reacher | LeWM-PO | −0.00085 ± 0.00077 | 48.2 ± 4.5 | 183 |
| Reacher | CoWM-PO | −0.00008 ± 0.00095 | 51.7 ± 4.3 | 183 |
| Reacher | CoWM-GF | −0.00050 ± 0.00053 | 47.1 ± 2.5 | 182 |
| PushT | LeWM-PO | −0.00673 ± 0.00815 | 42.8 ± 3.9 | 54 |
| PushT | CoWM-PO | −0.01550 ± 0.01231 | 40.9 ± 7.0 | 54 |
| PushT | CoWM-GF | −0.00571 ± 0.00906 | 45.9 ± 6.9 | 54 |

| 比较 | 任务 | 配对差 | 95% CI | 共同 N | p |
|---|---|---:|---:|---:|---:|
| 3a：CoWM-B − LeWM，ρ | Reacher | +0.05101 | [+0.01701, +0.08601] | 300 | 0.0663（Holm） |
| 3a：CoWM-B − LeWM，Hit@5 | Reacher | +2.7 pp | [−1.7, +7.0] pp | 300 | 1.0000（Holm） |
| 3a：CoWM-B − LeWM，regret | Reacher | −0.00875 | [−0.01410, −0.00346] | 300 | 0.0210（Holm） |
| 3a：CoWM-B − LeWM，ρ | PushT | +0.01345 | [−0.01790, +0.04402] | 300 | 1.0000（Holm） |
| 3a：CoWM-B − LeWM，Hit@5 | PushT | −0.7 pp | [−5.4, +4.1] pp | 300 | 1.0000（Holm） |
| 3a：CoWM-B − LeWM，regret | PushT | +0.02125 | [−0.01913, +0.07534] | 300 | 1.0000（Holm） |
| 3b：CoWM-PO − LeWM-PO，Δc（探索性） | Reacher | +0.00526 | [−0.00094, +0.01152] | 157 | 0.0980（未校正） |
| 3b：CoWM-PO − LeWM-PO，Δc（探索性） | PushT | +0.02645 | [−0.02653, +0.08613] | 38 | 0.3626（未校正） |

## 可用于论文的英文正文

> Table 3 evaluates candidate ranking and action refinement on Reacher and PushT. On Reacher, CoWM-B reduces selection regret from 0.0671 to 0.0583 rad (a 13.0% reduction; Holm-adjusted p = 0.021). CoWM-PO also improves the true execution cost relative to amplitude-matched random perturbations by 0.01536 rad (95% CI [0.00911, 0.02136], Holm-adjusted p = 0.0015). On PushT, ranking metrics show mixed trends, while the positive mean improvement of CoWM-PO has a confidence interval spanning zero. The refinement results describe states with observed outcomes at step 25; differences between CoWM-PO and LeWM-PO on jointly valid states remain inconclusive.

可用的任务选择说明（需引用支持饱和判断的主性能表）：

> We focus this analysis on Reacher and PushT, as task performance on TwoRoom and Cube is already near saturation (Table X).

## 统计口径与来源

均值按六个评估 seed 等权汇总；± 为 seed 间样本标准差，非标准误。95% CI 来自 10,000 次来源 episode 簇配对 bootstrap。Holm 校正沿用原四任务、16 项计划检验中的 15 项可估计比较，未因正文只展示两个任务而缩小检验族。评估 seed 不代表独立训练重复。v2 变长终点与检验族是在观察 v1 后形成的 benchmark 扩展，不应称为独立确认性实验。

- [原始 3a 报告](cvpr_table3a_ranking_report.md)
- [原始 3b 报告](cvpr_table3b_refinement_report.md)
- [原始总报告](cvpr_table3_report.md)
- [冻结方案](../../../plan/cvpr_table3_plan_v2.md)
- [论文用 LaTeX 表格](cvpr_table3_paper_tables.tex)（需要 booktabs；包含 3a、3b 和随机配对证据表）
