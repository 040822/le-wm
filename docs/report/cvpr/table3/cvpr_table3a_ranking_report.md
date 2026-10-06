# CVPR Table 3a v2：同一候选池上的排序准确性

本文比较固定 CoWM actor 生成的同一组 64 条物理动作候选，由 CoWM-B 与 LeWM 分别评分。主指标评价整个候选池的排序及 top-5 识别，不以任务成功率代替 rerank 准确性。主终点是每条分支原生终止时的真实末状态，或未终止时的 25 步预算终点；提前终止的轨迹不补到 25 步。

## 主结果

各单元先逐状态计算，再对可估计的六个评估 seed 等权平均。表中为 seed 均值 ± seed 间样本标准差（ddof=1）[来源 episode 簇配对 bootstrap 95% CI]。`Hit@5` 的随机参照按每个状态真实并列最优候选数精确计算，因此不固定为 5/64。

| 任务 | 指标 | CoWM-B | LeWM | 随机 Hit@5 参照 |
|---|---|---:|---:|---:|
| OGBench-Cube | Spearman ρ ↑；ρ 定义数 226/300（完整池 300/300） | 0.01221 ± 0.03459 [-0.01052, 0.03495] | 0.03262 ± 0.03673 [0.007279, 0.05751] | — |
| OGBench-Cube | Oracle Hit@5 ↑ | 32.7% ± 7.4 pp [27.5%, 38.0%] | 36.3% ± 8.1 pp [31.0%, 41.9%] | 33.0% ± 8.0 pp [28.6%, 37.7%] |
| OGBench-Cube | Selection regret ↓（block position L2（m）） | 0.004886 ± 0.0008256 [0.004162, 0.005704] | 0.00433 ± 0.0007467 [0.003772, 0.004909] | — |
| PushT | Spearman ρ ↑；ρ 定义数 300/300（完整池 300/300） | 0.0694 ± 0.02749 [0.04185, 0.09695] | 0.05595 ± 0.02164 [0.02776, 0.08452] | — |
| PushT | Oracle Hit@5 ↑ | 11.7% ± 2.3 pp [8.1%, 15.4%] | 12.3% ± 5.0 pp [8.8%, 16.1%] | 7.8% ± 0.0 pp [7.8%, 7.8%] |
| PushT | Selection regret ↓（max(position L2 / 20 px, wrapped angle / (π/9))） | 0.28 ± 0.04836 [0.241, 0.3325] | 0.2587 ± 0.02725 [0.2384, 0.28] | — |
| Reacher | Spearman ρ ↑；ρ 定义数 300/300（完整池 300/300） | 0.2551 ± 0.03719 [0.2163, 0.2939] | 0.2041 ± 0.04199 [0.1644, 0.2447] | — |
| Reacher | Oracle Hit@5 ↑ | 11.7% ± 4.8 pp [8.2%, 15.3%] | 9.0% ± 3.7 pp [6.0%, 12.4%] | 7.8% ± 0.0 pp [7.8%, 7.8%] |
| Reacher | Selection regret ↓（joint-position L2（rad）） | 0.05834 ± 0.008705 [0.05151, 0.06568] | 0.06709 ± 0.004831 [0.06049, 0.07421] | — |
| TwoRoom | Spearman ρ ↑；ρ 定义数 300/300（完整池 300/300） | 0.06485 ± 0.05361 [0.04541, 0.08497] | 0.05923 ± 0.04596 [0.03869, 0.08071] | — |
| TwoRoom | Oracle Hit@5 ↑ | 9.7% ± 3.9 pp [6.4%, 13.1%] | 9.3% ± 4.3 pp [6.2%, 12.7%] | 7.8% ± 0.0 pp [7.8%, 7.8%] |
| TwoRoom | Selection regret ↓（position L2（environment units）） | 3.277 ± 0.2188 [3.08, 3.497] | 3.322 ± 0.3206 [3.037, 3.664] | — |

`ρ` 行中的分母是排序相关可定义的状态数；其他两项使用全部完整 64 候选池。Cube 有 74/300 个状态的 64 条真实代价完全相同，故这些状态的 Spearman ρ 未定义（记 NA、不填零）；Hit@5 和 regret 仍保留这 300 个状态。四任务均无预测代价常数状态。regret 的物理量纲因任务而异，不跨任务直接宏平均。

## CoWM-B − LeWM 配对差

正的 ρ/Hit@5 差及负的 regret 差有利于 CoWM-B。p 值来自预设双侧来源 episode 簇配对 bootstrap；Holm 校正覆盖预设 16 项中可估计的 15 项。

| 任务 | 指标 | CoWM-B − LeWM（均值 ± SD [95% CI]） | 有效状态 | raw p | Holm p |
|---|---|---:|---:|---:|---:|
| TwoRoom | Spearman ρ | 0.005615 ± 0.01109 [-0.008262, 0.01966] | 300/300 | 0.4337 | 1 |
| TwoRoom | Oracle Hit@5 | 0.3% ± 4.6 pp [-3.2%, 3.9%] | 300/300 | 0.8494 | 1 |
| TwoRoom | Selection regret（position L2（environment units）） | -0.04506 ± 0.4616 [-0.3936, 0.2767] | 300/300 | 0.7863 | 1 |
| PushT | Spearman ρ | 0.01345 ± 0.0167 [-0.0179, 0.04402] | 300/300 | 0.3927 | 1 |
| PushT | Oracle Hit@5 | -0.7% ± 6.3 pp [-5.4%, 4.1%] | 300/300 | 0.7817 | 1 |
| PushT | Selection regret（max(position L2 / 20 px, wrapped angle / (π/9))） | 0.02125 ± 0.0572 [-0.01913, 0.07534] | 300/300 | 0.3847 | 1 |
| Reacher | Spearman ρ | 0.05101 ± 0.04743 [0.01701, 0.08601] | 300/300 | 0.005099 | 0.06629 |
| Reacher | Oracle Hit@5 | 2.7% ± 3.7 pp [-1.7%, 7.0%] | 300/300 | 0.2207 | 1 |
| Reacher | Selection regret（joint-position L2（rad）） | -0.008747 ± 0.01088 [-0.0141, -0.003455] | 300/300 | 0.0015 | 0.021 |
| OGBench-Cube | Spearman ρ | -0.02041 ± 0.02168 [-0.04843, 0.007993] | 226/300 | 0.1533 | 1 |
| OGBench-Cube | Oracle Hit@5 | -3.7% ± 4.3 pp [-7.8%, 0.2%] | 300/300 | 0.07399 | 0.8879 |
| OGBench-Cube | Selection regret（block position L2（m）） | 0.0005558 ± 0.0004892 [-9.94e-05, 0.001305] | 300/300 | 0.1176 | 1 |

Reacher 上 CoWM-B 三项点估计均较好，其中 selection regret 差通过 Holm 校正（Δ = −0.00875，95% CI [−0.0141, −0.00346]，Holm p = 0.0210）；Spearman 的 Holm p = 0.0663，未达到 0.05。PushT 的 Spearman 点估计偏向 CoWM-B，但 Hit@5 与 regret 点估计偏向 LeWM，均无显著的 Holm 校正差异。Cube 三项点估计均偏向 LeWM。TwoRoom 的 CoWM-B 点估计略好，但差异区间均跨零。整体不支持“CoWM-B 在所有任务上都更会 rerank”的概括。

## 辅助指标与第 25 步敏感性分析

| 任务 | Hit@1 CoWM-B | Hit@1 LeWM | 均匀选取期望 regret CoWM-B | 均匀选取期望 regret LeWM |
|---|---:|---:|---:|---:|
| OGBench-Cube | 27.0% ± 9.2 pp [22.1%, 32.1%] | 28.0% ± 8.9 pp [23.0%, 33.2%] | 0.004812 ± 0.0009001 [0.004311, 0.005316]（block position L2（m）） | 0.004812 ± 0.0009001 [0.004311, 0.005316]（block position L2（m）） |
| PushT | 3.0% ± 1.1 pp [1.2%, 5.1%] | 2.3% ± 2.3 pp [0.8%, 4.1%] | 0.2978 ± 0.03335 [0.2679, 0.334]（max(position L2 / 20 px, wrapped angle / (π/9))） | 0.2978 ± 0.03335 [0.2679, 0.334]（max(position L2 / 20 px, wrapped angle / (π/9))） |
| Reacher | 5.0% ± 3.0 pp [2.6%, 7.6%] | 1.7% ± 2.0 pp [0.3%, 3.3%] | 0.07666 ± 0.004347 [0.07149, 0.08219]（joint-position L2（rad）） | 0.07666 ± 0.004347 [0.07149, 0.08219]（joint-position L2（rad）） |
| TwoRoom | 1.0% ± 2.4 pp [0.0%, 2.3%] | 2.3% ± 2.3 pp [0.8%, 4.2%] | 4.094 ± 0.8149 [3.766, 4.469]（position L2（environment units）） | 4.094 ± 0.8149 [3.766, 4.469]（position L2（environment units）） |

第 25 步敏感性分析要求同一状态的全部 64 个分支都实际到达第 25 步。有效完整池很少，以下结果只作敏感性记录，不作为主结论。单元格依次为 ρ / Hit@5 / regret；每项为均值 [95% CI]，regret 单位同上。

| 任务 | 完整池数/300 | CoWM-B：ρ / Hit@5 / regret | LeWM：ρ / Hit@5 / regret |
|---|---:|---:|---:|
| OGBench-Cube | 0/300 | NA / NA / NA (block position L2（m）) | NA / NA / NA (block position L2（m）) |
| PushT | 7/300 | 0.3031 [0.01824, 0.4819] / 30.0% [0.0%, 58.3%] / 0.4817 [0.1743, 0.8398] (max(position L2 / 20 px, wrapped angle / (π/9))) | 0.546 [0.4443, 0.6445] / 50.0% [11.1%, 75.0%] / 0.175 [0.1328, 0.2776] (max(position L2 / 20 px, wrapped angle / (π/9))) |
| Reacher | 27/300 | 0.6388 [0.5719, 0.7087] / 27.9% [14.6%, 46.1%] / 0.02348 [0.01493, 0.03147] (joint-position L2（rad）) | 0.2771 [0.1042, 0.4433] / 20.8% [8.3%, 37.5%] / 0.04395 [0.03087, 0.05795] (joint-position L2（rad）) |
| TwoRoom | 0/300 | NA / NA / NA (position L2（environment units）) | NA / NA / NA (position L2（environment units）) |

第 25 步完整池覆盖为 TwoRoom 0/300、PushT 7/300、Reacher 27/300、Cube 0/300。PushT 与 Reacher 的估计基于极少完整池，不能据此替代变长实际终点的主面板。

## 实际终点与轨迹长度

下表聚合每任务 6 × 50 × 64 = 19,200 条已审计分支。检查到的原生终止均为成功终止；未终止分支在 25 步预算处结束。所有分支 `truncated` 标记均为 0。

| 任务 | 原生 terminated（success） | truncated | 25 步预算终点（running） | 实际长度 <25 | 实际长度 =25 |
|---|---:|---:|---:|---:|---:|
| OGBench-Cube | 19177 | 0 | 23 | 19168 | 32 |
| PushT | 18211 | 0 | 989 | 15954 | 3246 |
| Reacher | 8201 | 0 | 10999 | 7096 | 12104 |
| TwoRoom | 18642 | 0 | 558 | 18611 | 589 |

主终点是实际观测到的终态代价，因而分支执行长度并不完全相同；它与评分器的固定 25 步预测目标不完全等价。这是 v2 主面板的解释边界。

## 协议与复现

Spearman 并列使用平均秩；任一秩向量为常数时记 NA。真实 oracle 是 float64 代价严格等于该状态最小值的全部候选，不加 epsilon；预测代价并列由冻结候选索引打破。Hit@5 随机参照按真实 oracle 集合大小计算。物理代价定义见表中任务单位。

置信区间使用 10,000 次来源 episode 簇配对 bootstrap（RNG=20261005），跨评估 seed 联合重采样同一来源 episode，并对可估计 seed 等权汇总。seed 是评估抽样，不是训练重复。本 v2 的变长终点及检验族是在观察 v1 后形成的 benchmark 扩展，不应称作独立确认性结果。

- 冻结计划：[cvpr_table3_plan_v2.md](../../../plan/cvpr_table3_plan_v2.md)
- 全量逐状态汇总：[full.json](../../../../outputs/cvpr/table3/v2/analysis/full.json)
- 全量原始审计：[raw_audit_full.json](../../../../outputs/cvpr/table3/v2/analysis/raw_audit_full.json)
- 冻结配置：[frozen_config.json](../../../../outputs/cvpr/table3/v2/frozen_config.json)
- 来源清单：[source_refs.json](../../../../outputs/cvpr/table3/v2/source_refs.json)
- Table 3 总报告：[cvpr_table3_report.md](cvpr_table3_report.md)
