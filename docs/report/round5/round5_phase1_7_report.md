# Round5 Phase1.7 实验报告（进行中）

更新日期：2026-10-01  
执行方案：[round5_phase1_7_plan.md](../../plan/round5_phase1_7_plan.md)  
配置：[phase1_7.json](../../../config/round5/phase1_7.json)

## 当前结论

目前不能证明 B 的动作排序稳定优于独立 LeWM。最终 epoch-10 checkpoint 的 100-start 固定池结果覆盖 PushT/Reacher × seed 3072/4096：Pairwise 与 Spearman 差的逐状态区间全部跨 0，两个评分器的绝对 pairwise accuracy 接近 0.5、Spearman 接近 0。Top-1 25 步成功差也都不确定。seed-3072 的连续指标点估计偏向 Joint-B，seed-4096 的 Reacher 则接近持平或略偏 LeWM；方向和大小没有形成一致优势。

更早的 seed-3072 epoch-2 锁定固定池曾显示 Joint-B 在 PushT、Reacher 的连续排序指标高于 LeWM，单项 state-bootstrap 区间排除 0。这是特定 checkpoint、actor 候选分布和复用 confirmation 起点上的局部结果，连续指标未作多重校正；seed-4096 与 epoch-10 final checkpoint 没有复现同样的差距。在线 Joint-B rank_1.0 的结果属于单独训练的评分器，应作为另一种 scorer 报告，不能当作原始 B 已普遍胜过 LeWM 的证据。

最终 epoch-10 Joint-B 与独立 LeWM 的 P3 配对评估使用每任务 200 个固定 confirmation 起点。PushT 两个 seed 的点估计分别偏向 Joint-B +4.0、+4.5 pp；Reacher 则 seed-3072 偏向 LeWM +4 pp、seed-4096 偏向 Joint-B +4 pp。四项中 seed-4096 PushT 的未校正 p=0.0225；完整的 Joint-B 对 LeWM/B-only 检验族尚待 B-only 终点产物齐备后校正，不能据此作显著性结论。该 cohort 也用于旧 checkpoint 和在线 scorer，不能视作独立新复验。

最终 checkpoint 的 100-start 固定池取自上述 P3 cohort 的前 100 个起点。PushT 与 Reacher 的完整 200-start 固定池现已在两个 seed 上完成：连续排序绝对值仍接近机会水平，评分器差异区间均跨 0；Top-1 与 regret 点估计方向混合，未显示一致优势；两个 seed 的任务×评分器交互区间也都包含 0。所有固定池都复用 P3 confirmation cohort，扩大到 200 个起点提高同 cohort 指标精度，但不构成独立 replication。

旧 checkpoint 的锁定 P3 曾显示 PushT 上 LeWM 比 Joint-B 高 4 pp（两任务 Holm p=0.043），Reacher 差异不明确。与 epoch-10 的结果并列看，排序和闭环优势会随训练终点、任务与 cohort 改变。现有证据不支持论文声称“B 一般比单独 LeWM 排序更好”或“B 稳定提升闭环成功率”；更准确的写法是：部分早期锁定候选池显示 B 的连续排序优势，但最终 checkpoint 的跨 seed、跨任务复核没有建立稳定差异。区间较宽，未检出差异也不等于两者等价。

主矩阵中的 Recorded-control 与 B-only 训练仍在运行。最终 B-only P3 齐备后，将执行同 actor 的配对核验和预定多重比较校正，再据此决定论文主张是否需要进一步收窄。

## 现有 checkpoint 对照

固定协议为 P3、N=64、Euler、2 步动作流、clip 边界、相同 A、相同 confirmation 起点和推理 seed 16028。确认集在查看结果前锁定，PushT 与 Reacher 各 200 个 episode；统计按 episode 配对，精确 McNemar 检验后对两个任务作 Holm 校正。锁定记录见 `outputs/round5/phase1_7/analysis_lock.json`。

| 任务 | Joint-B | 独立 LeWM | 随机选候选 | LeWM−Joint-B | 配对 discordant（LeWM 赢/输） | 精确双侧 p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| PushT | 183/200 (91.5%) | 191/200 (95.5%) | 183/200 (91.5%) | +4.0 pp | 9/1 | 0.0215 | 0.0430 |
| Reacher | 185/200 (92.5%) | 186/200 (93.0%) | 163/200 (81.5%) | +0.5 pp | 13/12 | 1.0000 | 1.0000 |

PushT 上，LeWM 的成功率高于 Joint-B，校正后仍达到本轮预设显著性标准。Reacher 的 0.5 pp 差异没有统计证据，按预注册解释为不确定，不能称为等价。两个核心任务上，独立 LeWM 与 Joint-B 都高于随机候选基线。

相同闭环的 planning p50 分别为：

| 任务 | Joint-B | 独立 LeWM | LeWM 延迟倍数 | 观测峰值显存 |
|---|---:|---:|---:|---:|
| PushT | 0.592 s | 1.786 s | 3.02× | 1.48 / 2.15 GiB |
| Reacher | 0.774 s | 2.455 s | 3.17× | 1.48 / 2.15 GiB |

因此，当前 LeWM 的闭环表现不弱，但规划延迟约为 Joint-B 的三倍。以上时间来自本轮机器上的完整 planning 计时；包含各自 verifier 编码及评分，不能只用单次网络 forward 时间替代。

开发集 100 episode 的结果仅用于前期判断：PushT 为 Joint 97%、LeWM 98%、Random 98%；Reacher 为 Joint 91%、LeWM 88%、Random 82%。开发集没有用于调 checkpoint 或修改 confirmation 协议。

饱和/扩展任务的当前覆盖结果如下。它们用于检查跨任务退化，不纳入 PushT/Reacher 的主要检验族。

| 任务 | 起点数 | Joint-B | 独立 LeWM | 随机 | 备注 |
|---|---:|---:|---:|---:|---|
| Cube | 100 | 100% | 100% | 100% | 饱和，区分力不足 |
| TwoRoom | 100 | 99% | 100% | 99% | LeWM 与 Joint 只差 1 个 episode；此处未作主要假设检验 |
| Humanoid | 50 | 0% | 0% | 12% (6/50) | 新 cohort 上 6 个成功都出现在 Random；每个学习评分器对 Random 的配对 discordant 为 0/6，双侧 p=0.0313（探索性） |
| Scene | 50 | 58% (29/50) | 58% (29/50) | 62% (31/50) | Joint/Random discordant 5/3，p=0.727；LeWM/Random 3/1，p=0.625。Wilson 95% 区间约为 44.2–70.6% |

Humanoid 的 Phase1.7 新 cohort 上 Joint 与 LeWM 都为 0/50，而 Random 为 6/50，六个成功全是 Random-only；每组 6 个 discordant 的双侧精确 p=0.0313。Scene 三组配对差异均不显著。扩展任务未纳入主要检验族，样本各 50，Humanoid 的 nominal p 值只作诊断信号，不作跨任务结论。Humanoid 旧 Phase3 legacy cohort 上，原报告 LeWM 为 10/50，使用 Phase1.7 入口并保留 `legacy` 标签复核后为 12/50（24%）。这说明环境兼容和 LeWM 推理路径能在历史 cohort 上恢复相近成功率；新 cohort 的落差更可能与起点/目标分布有关，但两个 cohort 协议不同，不能直接作因果比较。Scene/Humanoid 的扩展评测目前只是旧 checkpoint 覆盖，不用于核心排序主张。

## 固定候选池：100 状态 PushT 主诊断

在同一 A、同一噪声下生成每个状态 64 个候选，逐元素确认 Joint-B 与 LeWM 看到的动作池完全相同，再在共同 simulator snapshot 上回放全部 6,400 条 25 步物理分支。数据来自 PushT dev 100 状态 cohort（hash `d35e3bcb…6a177f05`）；当前比较仍使用已有 checkpoint。

| 100 状态指标 | Joint-B | 独立 LeWM |
|---|---:|---:|
| Pairwise sign accuracy（cost 对物理距离） | 0.5256 | 0.5294 |
| Spearman（cost 对物理距离） | 0.0706 | 0.0833 |
| Top-1 25 步物理距离 | 101.14 | 101.49 |
| Top-1 25 步成功 | 15/100 (15%) | 13/100 (13%) |
| 相同池均匀随机期望 25 步成功 | 12.1% | 12.1% |
| Pool oracle 25 步成功覆盖 | 95% | 95% |
| Oracle-best top-5 覆盖 | 11.58% | 11.58% |
| 25 步有效覆盖率 | 89.53% | 89.53% |

LeWM−Joint 的 top-1 成功率差为 −2 pp（bootstrap 95% CI：−6 至 +2 pp），精确 McNemar 的 discordant 为 Joint-only 3、LeWM-only 1，双侧 p=0.625。有效 top-1 距离/Regret 的配对差均为 +0.62，95% bootstrap CI 为 −1.70 至 +2.84（n=85）。Pairwise accuracy 差为 +0.0039（95% CI：−0.019 至 +0.028，n=94），Spearman 差为 +0.0127（95% CI：−0.052 至 +0.078，n=94）；两区间均包含 0。Pairwise accuracy 本身只比随机排序的 0.5 高约 0.03，两评分器的 oracle-best top-5 命中也相同。

另按每个评分器自己的排序统计“前 k 个中至少有一个成功动作”的状态比例。Joint/LeWM 分别为：top-1 15%/13%，top-3 20%/13%，top-5 22%/16%，top-10 31%/27%，top-20 45%/42%；按每状态成功候选数量计算的随机排序期望依次为 12.1%、16.1%、19.1%、26.0%、39.2%。top-3 的配对差为 −7 pp（LeWM−Joint，未校正 bootstrap 95% CI：−13 至 −2 pp，McNemar p=0.039），但这是开发集上增加的探索性 k 指标，且同时看了 5 个 k，不能当作确认性证据。明细见 `outputs/round5/phase1_7/fixed_pool/pusht/pusht_phase1_7_dev_16027_v1/success_ranking_analysis.json`；统一分析脚本复算的覆盖统计见同目录 `success_coverage_analysis.json`。整体上目前没有证明 B 优于 LeWM；锁定起点上的 top-3 差异不显著，但该终点与 P3 闭环共用起点，不能当作另一批独立复验。100 状态 dev 样本也不能证明两者等价。配对连续排名 bootstrap 见 `outputs/round5/phase1_7/fixed_pool/pusht/pusht_phase1_7_dev_16027_v1/paired_rank_bootstrap.json`。

此前 5 状态 smoke 的方向混合（LeWM 相关性较高、Joint top-1 距离稍低），现在由 100 状态结果取代；Reacher 固定池也已完成，跨任务对照见下节。

## 固定候选池：100 状态 Reacher 主诊断

Reacher 同样在相同 A/相同噪声下评分并物理回放 6,400 个候选分支，dev cohort hash 为 `45e7d7dd…daf4165a`。与 PushT 相比，这里的连续物理距离排序较可辨认，且更偏向独立 LeWM：

| 100 状态指标 | Joint-B | 独立 LeWM |
|---|---:|---:|
| Pairwise sign accuracy | 0.6123 | 0.6552 |
| Spearman（cost 对物理距离） | 0.2984 | 0.3908 |
| Top-1 25 步成功 | 52/100 (52%) | 50/100 (50%) |
| Pool oracle 25 步成功覆盖 | 94% | 94% |
| Oracle-best top-5 覆盖 | 9% | 12% |
| 25 步有效候选覆盖率 | 61.22% | 61.22% |

LeWM−Joint 的 pairwise accuracy 差为 +0.0429（配对 state bootstrap 95% CI：+0.0139 至 +0.0719），Spearman 差为 +0.0924（95% CI：+0.0175 至 +0.1686）；均为未作跨指标/任务校正的开发集区间。top-1 成功差为 −2 pp（95% CI：−11 至 +7 pp，McNemar p=0.832），没有闭环首段成功优势。top-1 距离配对差为 +0.0008（LeWM 略差，n=48，95% CI：−0.0048 至 +0.0066）。

按 top-k 成功分支覆盖，Joint/LeWM 分别为：top-1 52%/50%，top-3 64%/73%，top-5 73%/76%，top-10 82%/81%，top-20 88%/89%。top-3 数值偏向 LeWM +9 pp（95% CI：0 至 +18 pp，McNemar p=0.078），仍不构成确认性差异；候选池按随机抽取时的期望依次为 47.5%、70.4%、77.9%、84.8%、89.3%。逐状态结果见 `outputs/round5/phase1_7/fixed_pool/reacher/reacher_phase1_7_dev_16027_v1/success_ranking_analysis.json`，统一脚本复算的 top-k 覆盖数据在同目录 `success_coverage_analysis.json`，排名 bootstrap 见同目录的 `paired_rank_bootstrap.json`。

跨任务看，PushT 的连续排序接近随机且 top-3 成功覆盖数值偏向 Joint；Reacher 的连续排序偏向 LeWM，top-3 成功覆盖也偏向 LeWM。top-1 真实成功在两任务都没有显著差异。现有固定池证据因此反对“联合 B 有稳定、跨任务的排序优势”，同时也显示排序结论取决于任务和评价目标，需用独立锁定 cohort 与匹配训练复验。

## 固定候选池：PushT 锁定 confirmation 次要终点

本评估复用 P3 闭环确认中锁定的 PushT 200 个起点，使用相同 A、每状态 64 个候选及共同物理回放。该 cohort 独立于 dev 起点，但它与 P3 共用确认起点，因此是同一确认样本上的另一个终点，不构成新的独立重复实验。

| 指标 | Joint-B | 独立 LeWM | LeWM−Joint |
|---|---:|---:|---:|
| Top-1 25 步成功 | 19.0% | 13.0% | −6.0 pp |
| 均匀随机挑一个候选的精确期望 | 18.4% | 18.4% | — |
| Pairwise sign accuracy | 0.5208 | 0.5224 | +0.0016 |
| Spearman（cost 对物理距离） | 0.0579 | 0.0631 | +0.0052 |
| Top-1 物理距离 regret（各自均值） | 29.25 | 25.57 | −3.68 |

Top-1 成功的不一致起点为 Joint-only 23、LeWM-only 11，精确 McNemar 双侧 p=0.0576（未校正）。物理 regret 的完整配对样本 bootstrap 差为 −3.25（LeWM−Joint，95% CI：−8.10 至 +1.34，n=160）；上表模型均值差采用各自有效记录计算。两种评分器的连续排序相关性几乎相同并接近随机排序；regret 的方向略偏向 LeWM，Top-1 实际成功的方向略偏向 Joint，统计证据均不足以支持稳定排序优势。

补充报告按分数前 k 个候选中是否至少有一个成功的覆盖率。除 top-1 外，k 扫描属于探索性分析，表中 p 值均未作多重比较校正。

| k | Joint-B 覆盖 | 独立 LeWM 覆盖 | LeWM−Joint | 随机顺序期望 | 配对 McNemar p |
|---:|---:|---:|---:|---:|---:|
| 1 | 19.0% | 13.0% | −6.0 pp | 18.4% | 0.058 |
| 3 | 33.5% | 31.0% | −2.5 pp | 32.2% | 0.487 |
| 5 | 39.0% | 38.5% | −0.5 pp | 38.7% | 1.000 |
| 10 | 47.5% | 48.0% | +0.5 pp | 48.7% | 1.000 |
| 20 | 62.5% | 58.0% | −4.5 pp | 63.9% | 0.289 |

除 top-1 外，两套排序的前 k 覆盖非常接近，且与随机顺序差距有限。top-1 的名义差异没有通过精确配对检验；开发集 top-3 的 Joint 数值信号也没有在这批锁定起点上复现为明显差距。固定池选择效果因此不解释 P3 上 LeWM 较高的闭环成功率，后者可能来自后续观测处的连续重规划，而不是单次初始候选排序。完整逐分支数据、共同候选与摘要位于 `outputs/round5/phase1_7/fixed_pool_confirmation_16028_locked_v1/pusht/pusht_phase1_7_confirmation_16028_v1/`；统一复算的 top-k 数据保存在同目录 `success_coverage_analysis.json`。

## 固定候选池：Reacher 锁定 confirmation 次要终点

该评估复用 Reacher P3 闭环确认的 200 个起点，在同一 A 和相同候选噪声下评分 64 个候选，并共同回放全部 12,800 条物理分支。它独立于 dev cohort，但与 P3 使用同一批确认起点，不能算作 P3 的独立重复。逐分支结果、共同候选池与摘要位于 `outputs/round5/phase1_7/fixed_pool_confirmation_16028_locked_v1/reacher/reacher_phase1_7_confirmation_16028_v1/`；统一复算的 top-k 数据保存在同目录 `success_coverage_analysis.json`。

| 指标 | Joint-B | 独立 LeWM | LeWM−Joint-B |
|---|---:|---:|---:|
| Pairwise sign accuracy（cost 对物理距离） | 0.5939 | 0.6278 | +0.0340（95% CI：+0.0143 至 +0.0543） |
| Spearman（cost 对物理距离） | 0.2476 | 0.3319 | +0.0843（95% CI：+0.0322 至 +0.1353） |
| Top-1 25 步成功 | 53.0% | 49.0% | −4.0 pp（95% CI：−11 至 +3 pp） |
| Pool oracle 25 步成功覆盖 | 89.5% | 89.5% | 同一动作池 |
| Oracle-best top-5 覆盖 | 16.5% | 15.5% | −1.0 pp（95% CI：−6.5 至 +4.5 pp） |
| Top-1 物理 regret（各自有效状态） | 0.03614（117/200） | 0.03471（125/200） | 共同有效状态上的 LeWM−Joint：+0.00118（n=86，95% CI：−0.00279 至 +0.00497） |

Top-1 成功的不一致起点为 Joint-only 30、LeWM-only 22，精确 McNemar 双侧 p=0.332。连续排序相关性在未校正的 state bootstrap 区间中偏向 LeWM；物理 regret 的完整配对区间包含 0。指标对实际成功分支的覆盖则数值上偏向 Joint：

| k | Joint-B 覆盖 | 独立 LeWM 覆盖 | LeWM−Joint | 随机顺序期望 | 配对 McNemar p |
|---:|---:|---:|---:|---:|---:|
| 1 | 53.0% | 49.0% | −4.0 pp | 46.1% | 0.332 |
| 3 | 73.5% | 68.0% | −5.5 pp | 68.3% | 0.099 |
| 5 | 79.5% | 74.5% | −5.0 pp | 75.2% | 0.087 |
| 10 | 83.0% | 82.5% | −0.5 pp | 81.5% | 1.000 |
| 20 | 87.5% | 86.5% | −1.0 pp | 85.5% | 0.727 |

k>1 的 p 值未作多重比较校正，均属探索性结果。前 1/3/5 个候选的成功覆盖没有显著配对差异；两种排序在 top-10/20 几乎一致，且和随机顺序期望接近。Reacher 因此也没有呈现一致的单一胜者：LeWM 更贴近连续物理距离，Joint 的小 k 成功覆盖数值较高，但该方向未通过配对检验，并且与 P3 闭环方向不同。

直接检查任务×评分器交互时，先在每个任务内计算配对的 LeWM−Joint 效应，再按起点分别重采样 PushT 与 Reacher，报告 Reacher 减 PushT 的差。Pairwise 效应交互在 dev 为 +0.039（95% CI：+0.002 至 +0.076），在锁定固定池 cohort 为 +0.032（95% CI：+0.010 至 +0.055）；Spearman 交互分别为 +0.080（95% CI：−0.022 至 +0.180）和 +0.079（95% CI：+0.019 至 +0.139）。Top-1 实际成功率交互分别为 0 pp（95% CI：−10 至 +10 pp）和 +2 pp（95% CI：−7 至 +11 pp），均包含 0。区间未对指标扫描作校正；这提示连续排序优势可能随任务变化，但尚未显示相应的 top-1 成功交互。confirmation 固定池与 P3 复用起点，不能当作独立闭环复验。逐项结果见 `outputs/round5/phase1_7/fixed_pool/task_scoring_interaction_dev_16027.json` 和 `outputs/round5/phase1_7/fixed_pool_confirmation_16028_locked_v1/task_scoring_interaction_16028.json`；计算入口为 `scripts/round5_phase1_7_fixed_pool_interaction.py`。

## P2 CEM：5 状态校准

补充实现了独立 LeWM 参与 actor-warm-started P2 CEM 的路径：Joint A 生成 warm start，LeWM 根据同一真实观测/动作历史为 CEM 候选打分。5 状态 PushT smoke 中，Joint 与 LeWM 都为 5/5 成功；CEM 参数锁定为 300 samples、30 iterations、top-30、`cem-clip`。planning p50 分别为 1.337 s 和 10.869 s（LeWM 约 8.1×），观测峰值显存分别为 0.086 和 0.140 GiB。样本过小，只确认评估管线和历史输入可运行，不作性能结论。首次 100 状态 LeWM 评估因终止 episode 的历史动作块不完整而中断；修正非活跃行的历史补齐后，5 状态同 cohort smoke 再次通过（5/5，评估 11.30 s，峰值显存 0.140 GiB）。

同一 100 状态 dev cohort 的完整结果如下。成功结果按 episode 配对；两个任务的名义 McNemar p 值再以 Holm 法校正。完整评估耗时是同一机器上记录的 wall time，包含规划与环境交互。

| 任务 | Joint | 独立 LeWM | LeWM−Joint | discordant（Joint-only / LeWM-only） | 精确 p | Holm p | 完整评估时间（Joint / LeWM） | 峰值显存（Joint / LeWM） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PushT | 95/100 | 98/100 | +3 pp | 2 / 5 | 0.453 | 0.453 | 34.2 / 342.8 s (10.0×) | 391 / 633 MiB |
| Reacher | 95/100 | 100/100 | +5 pp | 0 / 5 | 0.0625 | 0.125 | 60.2 / 435.4 s (7.2×) | 392 / 633 MiB |

两任务的固定参数 dev 评估中，独立 LeWM 没有显著落后于 Joint；Reacher 的五个不一致 episode 全由 LeWM 成功，但校正后仍不显著。结果不能当作确认性主张。端到端成本明确更高：LeWM 运行时间约为 Joint 的 7.2–10.0 倍。P2 CEM 是闭环策略效果比较，不等同于逐候选固定池的 B 排序检验。`cem-clip` 是协议配置；底层结果记录为 `candidate_clip`。

扩展任务的同协议 CEM 结果如下，Cube/TwoRoom 各为 100 个状态，Scene/Humanoid 各为 50 个状态；这些检验是探索性覆盖，不并入 PushT/Reacher 主要检验族。

| 任务 | Joint | 独立 LeWM | discordant（Joint-only / LeWM-only） | 名义精确 p | 完整评估时间（Joint / LeWM） |
|---|---:|---:|---:|---:|---:|
| Cube | 89/100 | 87/100 | 7 / 5 | 0.774 | 107.6 / 320.2 s (2.98×) |
| TwoRoom | 100/100 | 100/100 | 0 / 0 | 1.000 | 38.2 / 296.9 s (7.79×) |
| Scene | 28/50 | 27/50 | 3 / 2 | 1.000 | 154.8 / 217.3 s (1.40×) |
| Humanoid | 6/50 | 6/50 | 0 / 0 | 1.000 | 144.4 / 225.9 s (1.56×) |

四项扩展结果都没有显示独立 LeWM 明显落后。Cube 上 Joint 数值高 2 pp；Scene 上高 2 pp；TwoRoom 饱和、Humanoid 相同。各任务名义配对检验均不显著；LeWM 完整评估仍慢 1.4–7.8 倍。六项任务的 P2 CEM 配对现已全部完成，但扩展任务样本较小，不能据此声称等价。

将非饱和任务的 CEM 不一致 episode 合并后，LeWM-only 与 Joint-only 分别为 17/12，精确配对 p=0.458。直接检验任务间差异时，对 Cube、PushT、Reacher、Scene 的 29 个不一致 episode 作条件似然比检验，任务×比较交互的 exact p=0.093；TwoRoom 与 Humanoid 因无不一致 episode 不提供交互信息。当前没有足够证据证明任务间效果异质，但任务方向确实混合；该分析属于探索性闭环效果分析，不是固定池排序交互检验。

## 新训练状态

本轮当前启动的八个训练任务统一使用 episode 互斥划分、仅训练 split 拟合的 action normalizer、batch 128、AdamW、学习率 5e-5。前三项原启动进程在 2026-09-30 05:24 因 W&B/Hydra `BrokenPipeError` 同时退出；只留下 epoch 2 权重，没有可供 Lightning 精确恢复的 `last.ckpt`。因此从 epoch 0 重启到独立的 `retry1` 目录，种子、划分、归一化统计和初始化权重哈希均与原任务一致，避免覆盖旧产物：

| 任务 | Seed | 臂 | 输出目录 | GPU |
|---|---:|---|---|---:|
| PushT | 3072 | Joint | `outputs/round5_phase1_7_pusht_s3072_joint_retry1` | 1 |
| PushT | 3072 | LeWM | `outputs/round5_phase1_7_pusht_s3072_lewm_retry3` | 0 |
| Reacher | 3072 | Joint | `outputs/round5_phase1_7_reacher_s3072_joint_retry1` | 3 |
| Reacher | 3072 | LeWM | `outputs/round5_phase1_7_reacher_s3072_lewm_retry2` | 2 |
| PushT | 4096 | Joint | `outputs/round5_phase1_7_pusht_s4096_joint_retry1` | 0 |
| PushT | 4096 | LeWM | `outputs/round5_phase1_7_pusht_s4096_lewm_retry1` | 1 |
| Reacher | 4096 | Joint | `outputs/round5_phase1_7_reacher_s4096_joint` | 3 |
| Reacher | 4096 | LeWM | `outputs/round5_phase1_7_reacher_s4096_lewm_retry1` | 2 |

最近一次训练日志快照（14:54 CST）：PushT/3072 Joint epoch 4（7,950/13,796），LeWM retry3 epoch 6（10,750/13,796）；Reacher/3072 Joint epoch 5（700/13,034），LeWM retry2 epoch 6（1,450/13,034）。PushT/4096 Joint epoch 4（2,250/13,796）、LeWM retry1 epoch 6（11,150/13,796）；Reacher/4096 Joint epoch 4（7,150/13,034），LeWM retry1 epoch 6（2,400/13,034）。八个训练进程仍在推进。14:54 CST GPU0–3 利用率均为 100%，空闲 14.2/13.2/13.0/8.0 GiB；GPU3 不满足训练预留要求，其余 GPU 仍被训练进程占用。GPU4/5 分别空闲约 22.4/22.5 GiB，GPU6/7 空闲。主机 load average 为 54.50/54.47/54.99（128 核），可用内存 151 GiB，swap 仅余 440 KiB；尚不启动新缓存或梯度任务。

PushT/4096 Joint 首次启动因重复添加已有 `data_pipeline` Hydra 键而在模型启动前退出，没有占用显存或产生训练权重；修正为普通 key override 后改在 `...joint_retry1` 目录启动。PushT/3072 LeWM retry1 在 8,650 步时停止，retry2 使用 `max_steps` 但会在 epoch 中途结束。为保持计划要求的完整 10 epoch 终点，四项早期 LeWM `max_steps` 会话随后均优雅停止并保留日志；当前 retry3/retry2/retry1/retry1 目录改用每 epoch `limit_train_batches`，分别与对应 Joint 每轮 batch 数配平。四个有效训练对的 split、normalizer、epoch 数和总更新数均匹配；所有早期停止目录均不纳入结果。

尝试将高显存训练派发到 GPU4–7 时被自动审批拦截：仓库 `AGENTS.md` 限定这些卡只用于低显存评测，即使用户已允许在显存足够时使用这些卡也不能绕过该约束。因此 GPU4–7 未用于训练；本轮 Reacher 固定池评估使用 GPU7 并已完成，训练继续使用 GPU0–3。

## epoch 2 匹配更新数固定池中期诊断

PushT 两个 seed 的 100 起点固定池回放均已完成。每个 seed 内，Joint-B 与同 seed 的独立 LeWM 对同一个 Joint actor 产生的 64 个候选评分，逐分支共 6,400 条；摘要均验证 `same_candidate_pool=true`。下表的 pairwise/Spearman 区间是按起点配对 bootstrap，物理 regret 区间来自逐起点配对 bootstrap；它们只描述当前 dev cohort，未作跨指标校正。两个 seed 使用同一 dev 起点集合，因此不合并当作 200 个独立起点。

| PushT seed | Pairwise Joint / LeWM | LeWM−Joint（95% CI） | Spearman Joint / LeWM | LeWM−Joint（95% CI） | Top-1 成功 Joint / LeWM | regret Joint / LeWM；LeWM−Joint（95% CI） | 配对 McNemar p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3072 | 0.5566 / 0.4674 | −0.0893（−0.1167, −0.0626） | 0.1602 / −0.0945 | −0.2547（−0.3322, −0.1789） | 0% / 1% | 47.28 / 59.78；+12.20（+4.81, +19.67） | 1.000 |
| 4096 | 0.5478 / 0.4860 | −0.0618（−0.0928, −0.0303） | 0.1319 / −0.0400 | −0.1719（−0.2601, −0.0820） | 4% / 2% | 44.08 / 50.51；+5.51（−2.61, +13.79） | 0.625 |

连续排序和距离 regret 的 epoch 2 点估计在 PushT 两个 seed 与 Reacher 两个 seed 都偏向 Joint-B；Reacher/4096 的 pairwise 和 Spearman 区间跨 0。Top-1 成功方向混合，且四个配对检验均不显著。这是当前匹配训练轨迹的初步排序信号，不能回答闭环优势是否稳定，也不能归因于联合训练机制：模型架构及初始化不同、起点来自同一个 dev cohort，并且 checkpoint 仍在训练早期。PushT 摘要位于 `outputs/round5/phase1_7/epoch2_matched_dev_100/pusht_s3072/pusht/pusht_phase1_7_dev_16027_v1/summary.json` 与 `outputs/round5/phase1_7/epoch2_matched_dev_100/pusht_s4096/pusht/pusht_phase1_7_dev_16027_v1/summary.json`；Reacher 摘要见下节。

## epoch 2 Reacher 固定池中期诊断

Reacher 两个训练 seed 的 Joint-B 与独立 LeWM 使用同一个对应 seed 的 Joint actor、同一 100 起点 dev cohort 和完全相同的 64 候选池。每组共回放 6,400 个分支。两个训练 seed 重用相同的 dev 起点，因此不能合并为 200 个独立起点。区间按起点配对 bootstrap，未作跨任务或跨指标校正。

| Reacher seed | Pairwise Joint / LeWM | LeWM−Joint（95% CI） | Spearman Joint / LeWM | LeWM−Joint（95% CI） | Top-1 成功 Joint / LeWM | regret Joint / LeWM；LeWM−Joint（95% CI） | 配对 McNemar p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3072 | 0.5765 / 0.4916 | −0.0849（−0.1068, −0.0620） | 0.2180 / −0.0228 | −0.2407（−0.3021, −0.1768） | 5% / 5% | 0.7085 / 0.8779；+0.189（+0.066, +0.311） | 1.000 |
| 4096 | 0.5302 / 0.5020 | −0.0282（−0.0655, +0.0106） | 0.0855 / 0.0055 | −0.0800（−0.1837, +0.0277） | 3% / 8% | 0.5751 / 0.6877；+0.1122（+0.0130, +0.2121） | 0.180 |

Pairwise、Spearman 和 regret 的点估计均偏向 Joint-B；seed 4096 的前两项区间跨 0。Top-1 成功率在 seed 3072 相同、seed 4096 数值偏向 LeWM，均没有显著配对差异。与 PushT 一样，这只是中期 dev 信号，不能代表训练终点或 confirmation。摘要位于 `outputs/round5/phase1_7/epoch2_matched_dev_100/reacher_s3072/reacher/reacher_phase1_7_dev_16027_v1/summary.json` 与 `outputs/round5/phase1_7/epoch2_matched_dev_100/reacher_s4096/reacher/reacher_phase1_7_dev_16027_v1/summary.json`。

## epoch 2 P3 闭环中期 dev

为检查固定池排序信号是否延伸到闭环，使用同一 Joint actor checkpoint、同一个 100 起点 dev cohort 和 epoch 2 Joint-B/LeWM，对两个 PushT seed 分别完成 P3。每个 seed 的两臂 episode ID 一一配对；不同训练 seed 的结果单独报告，因为它们重复使用同一批 dev 起点。成功率差的区间按 episode bootstrap 10,000 次计算；精确 McNemar p 值是未校正的探索性结果。

| PushT seed | Joint-B | 独立 LeWM | LeWM−Joint（95% CI） | discordant（LeWM-only / Joint-only） | 精确 p | 完整评估时间（Joint / LeWM） |
|---:|---:|---:|---:|---:|---:|---:|
| 3072 | 15/100 | 12/100 | −3 pp（−11, +5） | 7 / 10 | 0.629 | 23.6 / 42.1 s |
| 4096 | 16/100 | 10/100 | −6 pp（−13, +1） | 4 / 10 | 0.180 | 45.0 / 44.6 s |

两 seed 的闭环方向均偏向 Joint，但区间覆盖 0，不能声称 epoch 2 有显著闭环优势；它与固定池连续排序的一致方向是待 epoch 10 复验的信号。当前结果只用于训练轨迹诊断。3072 产物位于 `outputs/round5/phase1_7/epoch2_closed_loop_dev_100/pusht/pusht_phase1_7_dev_16027_v1/`，4096 位于 `outputs/round5/phase1_7/epoch2_closed_loop_dev_100_seed4096/pusht/pusht_phase1_7_dev_16027_v1/`；两组 evaluation context 均记录 `training_epoch=2`。

Reacher 两个 seed 的 P3 中期 dev 也已完成，均使用对应 Joint actor 和同一 100 episode cohort。配对 bootstrap 区间基于 10,000 次 episode 重采样，McNemar p 未校正。

| Reacher seed | Joint-B | 独立 LeWM | LeWM−Joint（95% CI） | discordant（LeWM-only / Joint-only） | 精确 p | 完整评估时间（Joint / LeWM） |
|---:|---:|---:|---:|---:|---:|---:|
| 3072 | 7/100 | 7/100 | 0 pp（−6, +6） | 5 / 5 | 1.000 | 41.2 / 41.5 s |
| 4096 | 4/100 | 10/100 | +6 pp（−2.5, +13） | 9 / 3 | 0.146 | 46.3 / 44.7 s |

Reacher/3072 持平，Reacher/4096 数值上偏向 LeWM；两组区间都跨 0。结合 PushT 的两组点估计偏向 Joint，闭环效果在任务/seed 间方向不一致。四个 seed 的评测重复使用同一 dev episode 集，不能跨 seed 累加样本量。分析文件为 `outputs/round5/phase1_7/epoch2_closed_loop_analysis/reacher_seed3072.json` 与 `outputs/round5/phase1_7/epoch2_closed_loop_analysis/reacher_seed4096.json`；Reacher 的 `evaluation_context` 与配对分析均记录 epoch 2。

## 在线分支采集：10 状态管线校准

按方案新增 `scripts/round5_phase1_7_online_branches.py`，从模型的 train episode split 中每 episode 抽一个状态，生成 16 个 Joint-A 候选和 16 个裁剪局部扰动，并在共同 cohort 起点上逐候选回放。固定 A 为 seed 3072、epoch 2 的 Joint actor；此轮只检查采集管线，使用 PushT/Reacher 各 10 个训练 episode。候选结果揭晓前锁定相同的状态级 8/2 train/dev 划分。每条分支最长 25 个 primitive step，保存实际执行动作、终止信息及 5/10/15/20/25 步和终止时的原始 224×224 图像；终止后不补造目标。训练目标仍需在 B 与 LeWM 各自冻结表征下由这些真实图像分别计算。

| 任务 | 训练状态 | 分支数 | A 候选 / 局部扰动 | 未来/终点图像记录 | 动作越界 | 提前成功终止的分支 |
|---|---:|---:|---:|---:|---:|---:|
| PushT | 10 | 320 | 16 / 16 每状态 | 1,600 / 320 个终点齐全 | 0 | 0 |
| Reacher | 10 | 320 | 16 / 16 每状态 | 1,564 / 320 个终点齐全 | 0 | 15 |

逐分支核对了 25 步上限、共有状态/起点身份、终点图像和动作合法性。Reacher 的 36 个缺失中间里程碑是提前终止后的未执行步；对应终止图像均被保留。分支成功数是同一 10 状态上重复的候选结果，不作为独立 episode 统计或模型性能结论。固定池原始产物位于 `outputs/round5/phase1_7/online_branch_smoke_10_seed3072/`；PushT 与 Reacher 的清单分别为 `pusht/actor_seed3072_epoch2_1a22d01dc81e/collection_manifest.json` 和 `reacher/actor_seed3072_epoch2_d04afe532f38/collection_manifest.json`。该校准批不进入正式在线训练。

## 正式在线分支采集与审计

两项正式采集均已完成。候选结果揭晓前已冻结每任务 256 个状态及 205/51 的训练/开发划分；每状态的 32 个候选包含 16 个 Joint-A 候选和 16 个局部扰动。逐候选审核了动作池身份、轨迹回放、归一化动作、有效长度、执行里程碑图像和终点图像：全部 8,192 条分支与冻结候选池一致，动作有限且无越界，所有有效里程碑和终点图像齐全，未给提前终止的分支补造后续观测。

| 任务 | 状态（train/dev） | 分支 | 提前终止分支 | 已执行未来里程碑图像 | 图像归档总数 | 审计中记录的成功分支* |
|---|---:|---:|---:|---:|---:|---:|
| PushT | 256（205/51） | 8,192 | 43 | 40,888 | 40,926 | 55 |
| Reacher | 256（205/51） | 8,192 | 434 | 39,474 | 39,854 | 443 |

*成功数只是同一状态下 32 条重复候选分支的采集字段，不是独立 episode 样本或模型性能结果。PushT/Reacher 的动作池哈希分别为 `17085e5e…2291f413` 和 `6caaeff8…1bb5323f`。数据位于 `outputs/round5/phase1_7/online_branches_epoch2_v1/`。

## 当前评测状态

- Reacher 固定候选池 dev 100 状态、6,400 个物理分支已完成；epoch 2 匹配训练的 PushT/Reacher × seed 3072/4096 固定池与 P3 诊断也全部完成，见前述中期结果。Reacher 与 PushT 的锁定 confirmation 固定池均完成 64/64 候选分支，并已报告 top-1、top-k 和连续排序指标。两任务锁定固定池均复用 P3 的同一批 200 个起点，独立于 dev cohort，但不能视为另一批独立于 P3 的确认起点。
- 09:15 CST 在名为 `independent_confirmation` 的输出目录重跑了旧 Joint/独立 LeWM 的 P3 评估。后续核对发现其 cohort SHA 与 `analysis_lock.json` 完全相同，200 个 episode ID 也与 confirmation 固定池 artifact 完全相同；PushT/Reacher 的 success-vector SHA 均与锁定结果相同。因此这是同一锁定 cohort 的复跑，只验证评估可复现，不能作为独立 P3 确认或增加样本量。该锁定 cohort 仍与当前 Phase1.7 新训练的 train split 互斥，后续匹配训练可继续使用它作确认集。沙箱首次启动因 CUDA 不可见而失败，随后按仓库规则提权并用相同 `CUDA_VISIBLE_DEVICES` 成功重跑。复算 artifact 为 `outputs/round5/phase1_7/independent_confirmation/analysis/closed_loop_pair_analysis.json`，复算入口为 `scripts/round5_phase1_7_closed_loop_analysis.py`。
- 为后续匹配训练消融，Phase1.7 闭环评估、固定池产物和统计入口已扩展为可加载 Fast-LeWAM 的独立 Stage-B checkpoint，支持 `recorded_control`、`b_only`、`recorded_clean` 分别与同一个 Joint-A 比较，并按评分器名称保存代价与配对统计。已用当前 PushT/Reacher 中间 checkpoint 在各自 dev cohort 的前 10 个起点完成低显存校准：每项 64 个候选、固定同池分支回放，GPU 峰值约 1.4 GiB。第一次校准发现 LeWM 适配器把 `LeWMPolicy` wrapper 当成 JEPA 模型调用；`scripts/round5_phase1_7.py` 已改为传入其 `.model`，两项重跑均以 `status=ok` 完成。校准产物位于 `outputs/round5/phase1_7/dev_checkpoint_smoke_10_fix1/`。PushT/Reacher × seed 3072/4096 的 epoch 2 匹配更新数 100 起点 dev 固定池评测已全部完成，结果分别见 PushT、Reacher 两节；四组 P3 dev 闭环也已完成，结果见上节。所有中间比较都不替代 epoch 10 终点，也不使用 confirmation cohort。
- 正式分支两任务的 256 状态采集、训练/开发划分和逐分支完整性审计均已完成，细节见上节。提前终止分支只保留真实已执行图像；成功字段仅用于审计，不用于排序训练，也不视作独立性能样本。
- 在线分支缓存管线已修正 episode 元数据读取；PushT Joint/LeWM、Reacher Joint 的缓存已完成。Reacher LeWM 因原 checkpoint 路径在训练继续后变化，另用不可变 epoch-2 snapshot 建立隔离缓存根 `outputs/round5/phase1_7/online_training/reacher/lewm_epoch2/`，不混用旧路径产物。缓存 identity 记录了 checkpoint、候选池、分支源文件、episode split 与 train-only normalizer。
- `scripts/round5_phase1_7_online_train.py` 的离线续训、分支未来状态预测及 λ=0.1/1.0 成对排序四臂均已完成每臂 2,000 次更新。PushT/Joint、PushT/LeWM、Reacher/Joint 结果位于各自 `online_training/{task}/{scorer}/arms/`；Reacher/LeWM 的匹配 epoch-2 组位于 `online_training/reacher/lewm_epoch2/arms/`。该组只用于后续锁定在线对照，不与纯离线主结果混合。图像 encoder/projector 固定，训练目标未读取成功标签或物理距离标签；梯度训练仍只在 GPU0–3 执行。
- PushT P2 CEM Joint/LeWM 100 状态结果已完成：95/100 与 98/100，配对 Holm p=0.453。
- Reacher P2 CEM Joint/LeWM 100 状态结果已完成：95/100 与 100/100，配对 Holm p=0.125。
- Cube、TwoRoom、Scene、Humanoid 的 P2 CEM 扩展配对已全部完成，结果见上表；六任务 CEM 总表已覆盖全部计划任务。
- 固定池 confirmation 输出单独写入 `outputs/round5/phase1_7/fixed_pool_confirmation_16028_locked_v1/`，不会覆盖已完成的 dev 产物或锁定的闭环结果。
- 四组固定池产物的 `success_coverage_analysis.json` 均由 `scripts/round5_phase1_7_fixed_pool_analysis.py` 生成；分析按起点 bootstrap 10,000 次，并记录精确超几何随机覆盖期望及未校正的 top-k McNemar p 值。
- PushT/Reacher 固定池的直接任务×评分器交互由 `scripts/round5_phase1_7_fixed_pool_interaction.py` 重算，分别保存 dev 与 confirmation 分析；confirmation 仍复用 P3 起点。
- Scene/Humanoid Random 覆盖评测已完成；Cube/TwoRoom 的三种 verifier 100 episode 覆盖也已完成。
- 2026-10-01 03:06 CST 更新：最终 epoch-10 Joint-B/LeWM 已在冻结的 200-start confirmation cohort 上启动 P3 配对评估，覆盖 PushT/Reacher × seed 3072/4096；每个训练 seed 使用独立输出根。八项均已完成。seed-4096 Reacher LeWM 初次因 GPU7 上四项并行导致显存 OOM，随后以单进程重跑完成；配对统计见本节的 epoch-10 final 表。该 cohort 此前已用于旧 checkpoint 的确认比较，不作为新的独立 episode cohort。用户手动扩容后主机 swap 总量约 79 GiB、当前约 74 GiB 可用；MemAvailable 约 357 GiB。主矩阵的 Recorded-control/B-only 八条训练仍在 GPU0–3 运行，03:05 快照均已进入 epoch 1；seed-3072 PushT 固定池已完成，Reacher 固定池为 40/64 分支。
- 2026-10-01 03:24 CST 更新：seed-3072 Reacher epoch-10 dev 固定池已完成 64/64 并生成配对分析和任务交互。该任务的 pairwise 差偏向 Joint-B、Spearman CI 跨 0、Top-1 持平；与 PushT 一起看不出显著任务×评分器交互。八个 Recorded-control/B-only 训练臂继续推进：PushT/3072 Recorded-control 已进 epoch 2，其余七条仍在 epoch 1；GPU0–3 各保留约 8.2 GiB。GPU4–7 空闲，主机 MemAvailable 约 369 GiB、swap 可用约 74 GiB、load 为 31/33/48（128 核）。
- 2026-10-01 03:27 CST 更新：在 GPU4–7 各启动一项 epoch-10 固定池二级诊断，覆盖 PushT/Reacher × seed 3072/4096，每项取冻结 confirmation cohort 的前 100 个起点、64 个共享候选；这些起点已在 P3/旧模型分析中使用，结果只作共池次级分析，不当作新独立 confirmation。四项均已通过启动检查。
- 2026-10-01 03:33 CST 更新：用户手动将 swap 扩至约 80 GB；系统识别为 79 GiB 总量，已用 5.8 GiB、可用 74 GiB，`MemAvailable` 约 347 GiB，当前没有主机内存压力。GPU4–7 的四项固定池诊断仍在运行，PushT 两项各报告 8/64 分支；Reacher 两项仍活跃。GPU0–3 的 Recorded-control/B-only 训练均有步数增长：PushT/3072 为 2,400/13,796 与 12,200/13,796，Reacher/3072 为 12,850/13,034 与 13,000/13,034；PushT/4096 为 10,900/13,796 与 7,750/13,796，Reacher/4096 为 7,600/13,034 与 5,100/13,034。GPU0–3 各约余 8.2 GiB，尚不增加训练任务；主机 load 为 93/90/71（128 核）。
- 2026-10-01 03:41 CST 复查：四项 GPU4–7 固定池评估均正常运行，PushT/3072 与 PushT/4096 各完成 24/64 分支，Reacher/3072 与 Reacher/4096 各完成 8/64；尚无最终 summary。八项训练均有进展：PushT/3072 Recorded-control 3,200/13,796、B-only 13,200/13,796；Reacher/3072 两臂均已进入 epoch 2，分别 550/13,034、950/13,034；PushT/4096 两臂为 11,700/13,796、8,650/13,796；Reacher/4096 两臂为 8,400/13,034、6,100/13,034。GPU0–3 各约余 8.2 GiB；GPU4/6 余约 46.1 GiB、GPU5/7 余约 39.8 GiB。主机 `MemAvailable` 约 344 GiB，总 swap 79 GiB、剩余 74 GiB，load 为 73/96/86（128 核）；未见 OOM 或主机内存压力。
- 2026-10-01 03:50 CST 复查：GPU4–7 固定池评估继续推进，PushT/3072 与 PushT/4096 各完成 40/64 分支，Reacher/3072 与 Reacher/4096 各完成 16/64，尚未生成最终 summary。训练进度为 PushT/3072 Recorded-control 4,200/13,796、B-only 13,750/13,796；Reacher/3072 两臂进入 epoch 2，分别 1,550/13,034、2,150/13,034；PushT/4096 两臂为 12,700/13,796、9,850/13,796；Reacher/4096 两臂为 9,350/13,034、7,300/13,034。GPU0–3 各余约 8.2 GiB；GPU4/6 余 46.1 GiB、GPU5 余 39.8 GiB、GPU7 余 41.6 GiB。主机 `MemAvailable` 约 344 GiB，总 swap 79 GiB、剩余 74 GiB，load 为 79/99/94（128 核）；未见 OOM 或 RAM 压力。新增 `scripts/round5_phase1_7_primary_family_analysis.py`，用于终点 P3 的同 actor 配对审计及每个任务×seed 内 Joint-B 对 LeWM/B-only 的 Holm 校正；待完整终点产物齐备后运行。
- 2026-10-01 03:56 CST 复查：文件计数显示 PushT/3072、PushT/4096 固定池各 53/64 分支，Reacher/3072、Reacher/4096 各 22/64；四个 `summary.json` 均尚未生成，四项任务句柄仍活跃。八条训练继续增长：PushT/3072 Recorded-control epoch 2 为 4,800/13,796、B-only 为 1,350/13,796；Reacher/3072 epoch 2 两臂为 2,200/13,034、2,900/13,034；PushT/4096 epoch 1 两臂为 13,300/13,796、10,550/13,796；Reacher/4096 epoch 1 两臂为 9,950/13,034、8,100/13,034。重新核对四组 Joint/Recorded-control/B-only metadata，初始模型 SHA、episode split SHA、normalizer 与训练窗口数均逐组三臂一致。GPU0–3 各余约 8.2 GiB；GPU4–7 分别余约 46.1、39.8、46.1、39.8 GiB。主机 `MemAvailable` 约 343 GiB，总 swap 79 GiB、可用 74 GiB，load 为 103/105/99（128 核）；无 OOM，当前不新增会争用 CPU/GPU 的任务。
- 2026-10-01 04:02 CST 复查：PushT/3072 与 PushT/4096 固定池各 63/64 分支；Reacher/3072 为 27/64、Reacher/4096 为 26/64，四项尚未生成 `summary.json`。训练继续增长：PushT/3072 Recorded-control epoch 2 为 5,350/13,796、B-only 为 2,100/13,796；Reacher/3072 epoch 2 两臂为 2,800/13,034、3,600/13,034；PushT/4096 epoch 1 两臂为 13,300/13,796、11,150/13,796；Reacher/4096 epoch 1 两臂为 10,500/13,034、8,850/13,034。GPU0–3 各余约 8.2 GiB；GPU4–7 分别余约 46.1、39.8、46.1、42.4 GiB。主机 `MemAvailable` 约 344 GiB，总 swap 79 GiB、可用 74 GiB；load 为 152/131/112（128 核），一分钟与五分钟值暂高于核心数；没有 OOM 或 RAM 压力，暂不叠加评测/训练。
- 2026-10-01 04:12 CST 更新：两项 PushT 100-start confirmation 固定池均完成 64/64 分支并生成 summary/逐状态分析。seed-3072 的 B−LeWM pairwise 差 +0.027（95% state-bootstrap CI [−0.011,+0.067]）、Spearman 差 +0.067（CI [−0.034,+0.168]），Top-1 为 B 14%/LeWM 19%；seed-4096 对应差值 +0.004（CI [−0.031,+0.041]）、+0.003（CI [−0.089,+0.100]），Top-1 为 16%/17%。连续指标 CI 均跨 0，top-1 的精确配对检验分别 p=0.125、p=1.000；confirmation 前 100 起点与 P3 cohort 复用。为降低排序指标的不确定度，已在空闲 GPU4、GPU6 各启动一项完整 200-start PushT 固定池扩展，单卡预检约有 48.5 GiB free，启动后分别约 45.2/48.1 GiB free；输出根 `epoch10_s3072_main_confirmation_fixed_pool_200` 与 `epoch10_s4096_main_confirmation_fixed_pool_200` 独立，四项 Reacher/PushT 扩展都明确归为同一 confirmation cohort 的交叉分析，不作独立复验。Reacher 100-start 两项当时各 33/64 分支仍在 GPU5/GPU7 运行。主机 `MemAvailable` 约 342 GiB、swap 可用 74 GiB，load 为 56/58/83（128 核），启动后资源正常。八条训练在上次 04:08 复查时均持续推进，GPU0–3 各约余 8.2 GiB。
- 2026-10-01 04:16 CST 复查：PushT 200-start 固定池两项各完成 5/64 分支；Reacher 100-start 两项分别为 42/64、41/64，均尚未生成 summary。八条训练都在推进：PushT/3072 Recorded-control epoch 2 为 6,950/13,796、B-only 为 4,050/13,796；Reacher/3072 epoch 2 两臂为 4,450/13,034、5,550/13,034；PushT/4096 epoch 2/1 两臂为 1,700/13,796、13,200/13,796；Reacher/4096 两臂为 12,150/13,034、10,800/13,034（均 epoch 1）。GPU0–3 各余约 8.2 GiB；GPU4/6 各余 44.3 GiB、GPU5 余 46.1 GiB、GPU7 余 39.8 GiB。主机 `MemAvailable` 约 341 GiB，总 swap 79 GiB、可用 74 GiB，load 为 82/82/87（128 核）；未见 OOM 或主机内存压力。
- 2026-10-01 04:21 CST 复查：PushT 200-start 扩展两项各 12/64；Reacher 100-start 两项为 47/64、46/64，均仍运行且没有 summary。训练进度：PushT/3072 Recorded-control epoch 2 为 7,500/13,796、B-only 为 4,650/13,796；Reacher/3072 epoch 2 为 5,000/13,034、6,150/13,034；PushT/4096 Recorded-control epoch 2 为 2,250/13,796、B-only epoch 1 为 13,750/13,796；Reacher/4096 Recorded-control epoch 1 为 12,700/13,034、B-only 为 11,450/13,034。GPU0–3 各余约 8.2 GiB，GPU4/6 各余 44.3 GiB、GPU5 余 46.1 GiB、GPU7 余 39.8 GiB。主机 `MemAvailable` 约 335 GiB、swap 可用 74 GiB，load 为 62/70/80（128 核）；全部任务正常推进。
- 2026-10-01 04:26 CST 复查：用户确认手动扩容 swap 至 80 GB；系统显示总量 79 GiB、已用 5.7 GiB、可用 74 GiB，`MemAvailable` 337 GiB。GPU4–7 的固定池任务仍活跃；Reacher 两项为 48/64 分支，尚无 Reacher 或 200-start PushT summary。训练继续增长：PushT/3072 Recorded-control epoch 2 为 8,000/13,796、B-only 为 5,250/13,796；Reacher/3072 两臂为 5,500/13,034、6,750/13,034；PushT/4096 Recorded-control epoch 2 为 2,750/13,796、B-only epoch 2 为 550/13,796；Reacher/4096 Recorded-control epoch 2 为 50/13,034、B-only epoch 1 为 12,050/13,034。GPU0–3 各余约 8.2 GiB，GPU4–7 余 44.3/44.2/44.3/46.1 GiB。沙箱内 GPU 查询无法连接驱动；按仓库要求使用显式 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` 提权只读复核成功。没有检测到主机内存压力。
- 2026-10-01 04:34 CST 复查：两项 Reacher 100-start 与两项 PushT 200-start 固定池进程仍活跃，尚无新 summary；Reacher 最近一次可见进度为 48/64 分支。八项训练继续推进，最新日志分别为 PushT/3072 Recorded-control 8,850/13,796、B-only 6,300/13,796；Reacher/3072 为 6,400/13,034、7,800/13,034；PushT/4096 Recorded-control 3,650/13,796、B-only epoch 2 为 1,600/13,796；Reacher/4096 Recorded-control epoch 2 为 1,000/13,034，B-only 已完成 epoch 1 并开始下一轮。GPU0–3 各余 8.2 GiB；GPU4–7 各余约 44.3/45.3/44.3/42.9 GiB。主机 `MemAvailable` 342 GiB、总 swap 79 GiB（已用 5.7、可用 74 GiB），load average 94.71/75.73/74.00（128 核）。当前不叠加新的 CPU 密集评估；待 Reacher 100-start 完成后再复用 GPU5/7 做 200-start 扩展。
- 2026-10-01 04:42 CST 更新：两项 Reacher 100-start 固定池均完成，逐状态 bootstrap 与按 seed 的 PushT/Reacher 交互分析已生成；结果见本节新增表格，未发现稳定跨 seed 排序优势。GPU5/7 已释放；两项 PushT 200-start 扩展仍在 GPU4/6 运行。八条训练持续推进：PushT/3072 Recorded-control 与 B-only 分别 9,700/13,796、7,350/13,796；Reacher/3072 为 7,250/13,034、8,800/13,034；PushT/4096 为 4,450/13,796、2,600/13,796；Reacher/4096 为 1,850/13,034、1,050/13,034（对应 Reacher/4096 B-only 已进入 epoch 2）。GPU0–3 各余约 8.2 GiB，GPU4–7 余 44.3/48.5/44.3/48.5 GiB。主机 `MemAvailable` 364 GiB、swap 总量 79 GiB（已用 5.7、可用 74 GiB），load 为 56.77/56.62/66.08（128 核）；未见 OOM 或内存压力。
- 2026-10-01 04:47 CST 更新：Reacher 200-start 固定池扩展已在 GPU5（seed 3072）和 GPU7（seed 4096）启动，输出根分别为 `epoch10_s3072_main_confirmation_fixed_pool_200`、`epoch10_s4096_main_confirmation_fixed_pool_200`；与前 100-start 共用同一冻结 confirmation cohort，属于扩展精度检查。显式限制 `CUDA_VISIBLE_DEVICES=5` / `7`；启动预检时两卡各有 48.5 GiB free，模型、HDF5 action 数据和 MuJoCo 均完成加载。一次启动复核显示 GPU5/7 分别使用约 8.1/7.6 GiB、尚余约 39.2/39.7 GiB，任务仍在启动初期且无异常。
- 2026-10-01 04:56 CST 复查：两项 Reacher 200-start 评估仍运行、未生成 summary；GPU5/7 分别占用约 16.8/4.2 GiB，剩余约 31.0/44.3 GiB。PushT 200-start seed-3072 已输出 40/64 分支，seed-4096 仍运行、尚无 summary；GPU4/6 各余约 44.3 GiB。八条训练均持续推进且已进入 epoch 2：PushT/3072 Recorded-control、B-only 为 11,250/13,796、9,200/13,796；Reacher/3072 为 8,850/13,034、10,650/13,034；PushT/4096 为 6,050/13,796、4,400/13,796；Reacher/4096 为 3,400/13,034、2,900/13,034。GPU0–3 各余约 8.2 GiB；主机 `MemAvailable` 324 GiB、总 swap 79 GiB（已用 3.9、可用 76 GiB），load 74.60/82.42/75.65（128 核）。没有 OOM 或主机内存压力。
- 2026-10-01 05:03 CST 复查：两个 PushT 200-start 固定池均已生成 63/64 个分支文件，尚无 summary；Reacher 200-start seed-3072/4096 分别为 4/64、5/64。八条训练仍在推进：PushT/3072 Recorded-control、B-only 为 11,900/13,796、10,150/13,796；Reacher/3072 为 9,600/13,034、11,550/13,034；PushT/4096 为 6,700/13,796、5,300/13,796；Reacher/4096 为 4,100/13,034、3,850/13,034。GPU0–3 各余约 8.2 GiB；GPU4/6 各余 44.3 GiB，GPU5/7 各余约 31.0 GiB。主机 `MemAvailable` 327 GiB、swap 79 GiB 总量中可用 76 GiB；load 为 137.59/121.04/96.38（128 核），短时一分钟值高于核心数。任务仍推进且未见 OOM；暂不增加 CPU 密集工作。
- 2026-10-01 05:10 CST 复查：PushT 200-start 固定池两项均已完成 64/64 并生成 summary 和逐状态 bootstrap。Pairwise/Spearman 差的区间均跨 0；Top-1 点估计分别偏 LeWM +1.5/+2.0 pp，配对精确检验 p=0.549/0.481；详细表格与交互待 Reacher 200-start 完成后补齐。Reacher 200-start 当前为 7/64、8/64，仍在 GPU5/7 运行。训练进度：PushT/3072 Recorded-control、B-only 为 12,650/13,796、11,000/13,796；Reacher/3072 为 10,350/13,034、12,450/13,034；PushT/4096 为 7,500/13,796、6,200/13,796；Reacher/4096 为 4,850/13,034、4,750/13,034。GPU4/6 已释放，各余 48.5 GiB；GPU5/7 各余 31.7 GiB；GPU0–3 各余约 8.2 GiB。主机 `MemAvailable` 328 GiB、swap 可用 76 GiB，load 为 36.78/75.44/87.37（128 核），无内存压力。
- 2026-10-01 05:18 CST 复查：PushT 200-start 分析均已落盘；Reacher 200-start 分支文件数为 12/64、13/64，summary 尚未生成。训练继续推进：PushT/3072 Recorded-control、B-only 为 13,450/13,796、12,050/13,796；Reacher/3072 Recorded-control 为 11,250/13,034，B-only 已进入 epoch 3、250/13,034；PushT/4096 为 8,300/13,796、7,200/13,796；Reacher/4096 为 5,650/13,034、5,750/13,034。GPU0–3 各余约 8.2 GiB；GPU4/6 各余 48.5 GiB，GPU5/7 余 35.4/44.3 GiB。主机 `MemAvailable` 330 GiB、总 swap 79 GiB（可用 76 GiB），load 40.09/48.94/69.52（128 核）；没有 OOM 或主机内存压力。
- 2026-10-01 05:24 CST 复查：Reacher 200-start 两个 seed 均推进至 16/64 分支，尚未生成 summary。八条训练继续增长：PushT/3072 Recorded-control epoch 3 为 250/13,796、B-only epoch 2 为 12,850/13,796；Reacher/3072 Recorded-control epoch 2 为 11,950/13,034、B-only epoch 3 为 1,050/13,034；PushT/4096 Recorded-control/B-only 为 8,950/13,796、7,950/13,796；Reacher/4096 为 6,350/13,034、6,500/13,034。GPU0–3 各余约 8.2 GiB，GPU4/6 各余 48.5 GiB、GPU5/7 余 35.4/36.7 GiB。主机 `MemAvailable` 326 GiB、swap 可用 76 GiB，load 42.90/43.91/60.41（128 核）；未见异常。
- 2026-10-01 05:30 CST 复查：Reacher 200-start 的 seed-3072/4096 均为 19/64 分支，仍无 summary。训练继续推进：PushT/3072 Recorded-control epoch 3 为 900/13,796、B-only epoch 2 为 13,600/13,796；Reacher/3072 Recorded-control epoch 2 为 12,550/13,034、B-only epoch 3 为 1,800/13,034；PushT/4096 Recorded-control/B-only 为 9,600/13,796、8,700/13,796；Reacher/4096 为 6,950/13,034、7,250/13,034。GPU0–3 各余约 8.2 GiB，GPU4/6 各余 48.5 GiB，GPU5/7 余约 31.2/31.0 GiB。主机 `MemAvailable` 329 GiB、swap 可用 76 GiB，load 39.46/40.98/53.83（128 核）；状态正常。
- 2026-10-01 05:36 CST 复查：Reacher 200-start 的两个 seed 均为 22/64 分支，尚未生成 summary。训练继续推进：PushT/3072 Recorded-control epoch 3 为 1,550/13,796、B-only epoch 3 为 500/13,796；Reacher/3072 Recorded-control epoch 3 为 100/13,034、B-only epoch 3 为 2,600/13,034；PushT/4096 Recorded-control/B-only 为 10,200/13,796、9,450/13,796；Reacher/4096 为 7,650/13,034、8,000/13,034。GPU0–3 各余约 8.2 GiB，GPU4/6 各余 48.5 GiB，GPU5/7 各余约 31.0 GiB。主机 `MemAvailable` 328 GiB、swap 可用 76 GiB，load 69.88/49.54/53.01（128 核）；无 OOM 或内存压力。
- 2026-10-01 05:42 CST 复查：Reacher 200-start 的 seed-3072/4096 均为 26/64 分支，summary 尚未生成。八条训练最新进度：PushT/3072 Recorded-control epoch 3 为 2,150/13,796、B-only 为 1,300/13,796；Reacher/3072 为 750/13,034、3,350/13,034；PushT/4096 为 10,850/13,796、10,200/13,796；Reacher/4096 为 8,300/13,034、8,750/13,034。GPU0–3 各余约 8.2 GiB，GPU4/6 各余 48.5 GiB，GPU5/7 余 43.1/44.3 GiB。主机 `MemAvailable` 323 GiB、swap 可用 76 GiB，load 43.10/45.26/49.92（128 核）；任务正常。
- 2026-10-01 05:48 CST 复查：Reacher 200-start 两项均为 29/64 分支，summary 尚未生成。训练进度：PushT/3072 Recorded-control epoch 3 为 2,800/13,796、B-only 为 2,050/13,796；Reacher/3072 为 1,400/13,034、4,100/13,034；PushT/4096 为 11,500/13,796、10,950/13,796；Reacher/4096 为 8,950/13,034、9,500/13,034。GPU0–3 各余约 8.2 GiB，GPU4/6 各余 48.5 GiB，GPU5/7 余 43.1/44.3 GiB。主机 `MemAvailable` 326 GiB、swap 可用 76 GiB，load 40.92/41.62/46.64（128 核）；无异常。
- 2026-10-01 05:53 CST 复查：Reacher 200-start 的 seed-3072/4096 均为 32/64 分支，summary 尚未生成。训练最新进度：PushT/3072 Recorded-control epoch 3 为 3,400/13,796、B-only 为 2,800/13,796；Reacher/3072 为 2,000/13,034、4,850/13,034；PushT/4096 为 12,100/13,796、11,700/13,796；Reacher/4096 为 9,600/13,034、10,250/13,034。GPU0–3 各余约 8.2 GiB，GPU4/6 各余 48.5 GiB，GPU5/7 余 43.1/31.0 GiB。主机 `MemAvailable` 327 GiB、swap 可用 76 GiB，load 40.11/40.95/44.80（128 核）；任务正常。
- 2026-10-01 05:59 CST 复查：Reacher 200-start seed-3072/4096 分别完成 36/64、35/64 分支，无 summary。训练继续推进：PushT/3072 Recorded-control epoch 3 为 4,050/13,796、B-only 为 3,550/13,796；Reacher/3072 为 2,700/13,034、5,600/13,034；PushT/4096 为 12,750/13,796、12,500/13,796；Reacher/4096 为 10,300/13,034、11,000/13,034。GPU0–3 各余约 8.2 GiB；GPU4/6 各余 48.5 GiB，GPU5/7 各余 31.0 GiB。主机 `MemAvailable` 326 GiB、swap 可用 76 GiB，load 38.66/40.10/43.25（128 核）；任务正常。
- 2026-10-01 06:05 CST 复查：Reacher 200-start seed-3072/4096 分别完成 40/64、38/64 分支，summary 未生成。训练进度：PushT/3072 Recorded-control epoch 3 为 4,700/13,796、B-only 为 4,300/13,796；Reacher/3072 为 3,300/13,034、6,350/13,034；PushT/4096 Recorded-control epoch 2 为 13,350/13,796、B-only 为 13,200/13,796；Reacher/4096 为 10,950/13,034、11,750/13,034。GPU0–3 各余约 8.2 GiB，GPU4/6 各余 48.5 GiB、GPU5/7 余 44.3/31.0 GiB。主机 `MemAvailable` 323 GiB、swap 可用 76 GiB，load 43.52/42.98/43.69（128 核）；无异常。
- 2026-10-01 06:11 CST 复查：Reacher 200-start seed-3072/4096 分别为 43/64、41/64 分支，尚无 summary。训练最新进度：PushT/3072 Recorded-control epoch 3 为 5,300/13,796、B-only 为 5,050/13,796；Reacher/3072 为 3,950/13,034、7,100/13,034；PushT/4096 Recorded-control epoch 3 为 200/13,796、B-only 为 150/13,796；Reacher/4096 为 11,600/13,034、12,450/13,034。GPU0–3 各余约 8.2 GiB，GPU4/6 各余 48.5 GiB，GPU5/7 余 44.3/31.0 GiB。主机 `MemAvailable` 323 GiB、swap 可用 76 GiB，load 39.27/40.44/42.24（128 核）；任务正常。
- 2026-10-01 06:17 CST 复查：Reacher 200-start seed-3072/4096 分别为 46/64、44/64 分支，尚无 summary。八条训练最新进度：PushT/3072 Recorded-control epoch 3 为 5,900/13,796、B-only 为 5,800/13,796；Reacher/3072 为 4,600/13,034、7,850/13,034；PushT/4096 Recorded-control epoch 3 为 800/13,796、B-only 为 850/13,796；Reacher/4096 Recorded-control epoch 2 为 12,250/13,034、B-only epoch 3 为 50/13,034。GPU0–3 各余约 8.2 GiB，GPU4/6 各余 48.5 GiB，GPU5/7 各余约 31.0 GiB。主机 `MemAvailable` 324 GiB、swap 可用 76 GiB，load 41.91/40.55/41.64（128 核）；状态正常。
- 2026-10-01 06:26 CST 复查：确认手动新增的 `/swapfile-phase1-7` 40 GiB 已启用；总 swap 约 79 GiB（8+32+40），已用约 3.5 GiB、可用约 76 GiB，`MemAvailable` 约 322 GiB。八条训练继续推进：PushT/3072 Recorded-control、B-only epoch 3 分别为 6,900/13,796、6,950/13,796；Reacher/3072 为 5,600/13,034、8,950/13,034；PushT/4096 为 1,750/13,796、2,050/13,796；Reacher/4096 两臂均进入 epoch 3，分别为 150/13,034、1,250/13,034。GPU0–3 各余约 8.2 GiB；GPU4/6 空闲，GPU5/7 仍在做 Reacher 200-start 固定池，尚未生成 summary（seed-4096 最近计数 48/64；seed-3072 最近明确计数 48/64）。任务仍推进，未见 OOM 或主机内存压力。
- 2026-10-01 06:33 CST 复查：Reacher 200-start seed-3072 已到 56/64，seed-4096 进程仍运行但本次轮询没有新的分支计数；两个 summary 均未生成。八条训练最新进度：PushT/3072 Recorded-control、B-only epoch 3 为 7,650/13,796、7,900/13,796；Reacher/3072 为 6,400/13,034、9,950/13,034；PushT/4096 为 2,550/13,796、3,000/13,796；Reacher/4096 为 950/13,034、2,200/13,034。GPU0–3 各余约 8.2 GiB，GPU4/6 空闲；GPU5/7 分别占用 11.7/10.0 GiB，评测仍有充足余量。主机 `MemAvailable` 321 GiB、总 swap 79 GiB（已用3.5、可用76 GiB），load 38.28/40.73/41.18（128核）；无异常。
- 2026-10-01 06:39 CST 复查：Reacher 200-start seed-3072/4096 日志均到 56/64 分支，summary 尚未生成，两个评估进程仍运行。训练继续推进：PushT/3072 Recorded-control、B-only epoch 3 为 8,300/13,796、8,650/13,796；Reacher/3072 为 7,050/13,034、10,650/13,034；PushT/4096 为 3,150/13,796、3,700/13,796；Reacher/4096 为 1,600/13,034、2,900/13,034。GPU0–3 各余约 8.2 GiB；GPU4/6 空闲；GPU5/7 分别余 31.7/43.8 GiB。主机 `MemAvailable` 325 GiB、swap 可用76 GiB，load 38.20/40.67/41.28（128核）；无异常。
- 2026-10-01 06:45 CST 复查：Reacher 200-start 两 seed 最新日志仍为 56/64，summary 未生成，进程保持运行；GPU5/7 当前各占用约 4.2 GiB，剩余约 44.3 GiB。八条训练均继续推进：PushT/3072 Recorded-control、B-only epoch 3 为 8,900/13,796、9,400/13,796；Reacher/3072 为 7,650/13,034、11,400/13,034；PushT/4096 为 3,800/13,796、4,450/13,796；Reacher/4096 为 2,250/13,034、3,650/13,034。GPU0–3 各余约 8.2 GiB；GPU4/6 空闲。主机 `MemAvailable` 322 GiB、swap 可用76 GiB，load 38.13/40.15/40.94（128核）；训练与评估进程仍运行，无 OOM。
- 2026-10-01 07:00 CST 复查：Reacher 两 seed 的 200-start 固定池均已完成；seed-4096 的 10,000 次 bootstrap 分析和 PushT/Reacher 任务交互文件也已生成。Reacher 的 200-start 排序差在两 seed 均接近 0，Top-1 成功分别为 11.5%/13.5% 和 14%/14%，没有稳定的 B 优势；跨任务交互区间均包含 0。八条训练继续推进：PushT/3072 Recorded-control、B-only epoch 3 为 10,550/13,796、11,300/13,796；Reacher/3072 Recorded-control epoch 3 为 9,350/13,034、B-only 已进入 epoch 4 为 200/13,034；PushT/4096 为 5,400/13,796、6,350/13,796；Reacher/4096 为 3,900/13,034、5,550/13,034。GPU0–3 各余约8.2 GiB、利用率接近100%；GPU4–7 当前空闲。主机 `MemAvailable` 369 GiB、swap 总量79 GiB（已用3.2、可用76 GiB），load 31.53/32.52/35.79（128核）；无内存压力。
- 2026-10-01 07:07 CST 复查：八条训练仍在推进：PushT/3072 Recorded-control、B-only epoch 3 为 11,400/13,796、12,300/13,796；Reacher/3072 Recorded-control epoch 3 为 10,200/13,034、B-only epoch 4 为 1,200/13,034；PushT/4096 为 6,300/13,796、7,350/13,796；Reacher/4096 为 4,800/13,034、6,550/13,034。GPU0–3 各余约8.2 GiB，利用率94–100%；GPU4–7 空闲。主机 `MemAvailable` 367 GiB、swap 总量79 GiB（已用3.2、可用76 GiB），load 31.27/31.23/33.83（128核）；无异常。
- 2026-10-01 07:13 CST 复查：八条训练继续前进：PushT/3072 Recorded-control、B-only epoch 3 为 12,050/13,796、13,050/13,796；Reacher/3072 Recorded-control epoch 3 为 10,850/13,034、B-only epoch 4 为 1,950/13,034；PushT/4096 为 6,950/13,796、8,100/13,796；Reacher/4096 为 5,450/13,034、7,300/13,034。GPU0–3 各余约8.2 GiB且利用率100%；GPU4–7 空闲。主机 `MemAvailable` 369 GiB、swap总量79 GiB（已用3.2、可用76 GiB），load 30.28/30.76/32.80（128核）；无异常。
- 2026-10-01 07:19 CST 复查：八条训练均有新进度：PushT/3072 Recorded-control、B-only epoch 3 为 12,650/13,796、13,750/13,796；Reacher/3072 Recorded-control epoch 3 为 11,500/13,034、B-only epoch 4 为 2,700/13,034；PushT/4096 为 7,550/13,796、8,800/13,796；Reacher/4096 为 6,050/13,034、8,050/13,034。GPU0–3 各余约8.2 GiB、利用率100%；GPU4–7 空闲。主机 `MemAvailable` 367 GiB、swap总量79 GiB（已用3.2、可用76 GiB），load 30.30/30.45/32.02（128核）；无异常。
- 2026-10-01 07:25 CST 复查：PushT/3072 B-only 已进入 epoch 4、650/13,796；其 Recorded-control 在 epoch 3 为 13,300/13,796。Reacher/3072 Recorded-control 为 epoch 3、12,150/13,034，B-only epoch 4、3,400/13,034。PushT/4096 Recorded-control、B-only epoch 3 为 8,150/13,796、9,550/13,796；Reacher/4096 为 6,700/13,034、8,750/13,034。八条训练均继续推进。GPU0–3 各余约8.2 GiB、利用率100%，GPU4–7 空闲；主机 `MemAvailable` 368 GiB、swap总量79 GiB（已用3.2、可用76 GiB），load 30.17/30.56/31.63（128核）；无异常。
- 2026-10-01 07:30 CST 复查：PushT/3072 Recorded-control 刚完成 epoch 3，B-only 已进入 epoch 4、100/13,796；Reacher/3072 Recorded-control epoch 3 为 12,750/13,034、B-only epoch 4 为 4,100/13,034。PushT/4096 Recorded-control、B-only epoch 3 为 8,800/13,796、10,250/13,796；Reacher/4096 为 7,300/13,034、9,500/13,034。八条训练均继续推进，GPU0–3 各余约8.2 GiB、利用率100%，GPU4–7 空闲。主机 `MemAvailable` 368 GiB、swap总量79 GiB（已用3.2、可用76 GiB），load 31.10/30.65/31.29（128核）；无异常。
- 2026-10-01 07:36 CST 复查：PushT/3072 Recorded-control、B-only epoch 4 分别为 700/13,796、2,100/13,796；Reacher/3072 两臂 epoch 4 分别为 250/13,034、4,850/13,034。PushT/4096 Recorded-control、B-only epoch 3 为 9,400/13,796、10,950/13,796；Reacher/4096 为 7,950/13,034、10,200/13,034。八条训练都在推进；GPU0–3 各余约8.2 GiB、利用率100%，GPU4–7 空闲。主机 `MemAvailable` 369 GiB、swap总量79 GiB（已用3.2、可用76 GiB），load 30.62/30.94/31.23（128核）；无异常。
- 2026-10-01 07:41 CST 复查：PushT/3072 Recorded-control、B-only epoch 4 为 1,300/13,796、2,800/13,796；Reacher/3072 为 900/13,034、5,600/13,034。PushT/4096 Recorded-control、B-only epoch 3 为 10,000/13,796、11,700/13,796；Reacher/4096 为 8,550/13,034、10,900/13,034。所有训练任务继续推进。GPU0–3 各余约8.2 GiB、利用率96–100%，GPU4–7 空闲；主机 `MemAvailable` 368 GiB、swap总量79 GiB（已用3.2、可用76 GiB），load 29.94/30.42/30.92（128核）；无异常。
- 2026-10-01 06:50 CST 复查：Reacher seed-3072 的 200-start 固定池已完成，10,000 次 state-bootstrap 分析及与 PushT 的 seed-3072 任务交互均已生成。Joint-B/LeWM pairwise 为 0.508/0.499（B−LeWM +0.008，CI [−0.006, +0.023]），Spearman 为 0.022/−0.001（差 +0.023，CI [−0.010, +0.056]），Top-1 为 11.5%/13.5%（LeWM−B +2 pp，CI [−4, +8]，精确 p=0.618）；交互区间也都包含0。seed-4096 评估进程仍运行，summary 尚未生成，最新已知进度 56/64。八条训练最新进度：PushT/3072 Recorded-control、B-only epoch 3 为 9,550/13,796、10,150/13,796；Reacher/3072 为 8,300/13,034、12,150/13,034；PushT/4096 为 4,400/13,796、5,200/13,796；Reacher/4096 为 2,900/13,034、4,400/13,034。GPU0–3 各余约 8.2 GiB，GPU5/7 分别空闲/剩余约40.1 GiB，GPU4/6 空闲。主机 `MemAvailable` 348 GiB、swap 总量79 GiB（已用3.3、可用76 GiB），load 32.33/35.70/38.79（128核）；无内存压力。















- 2026-10-01 00:28 CST 更新：在线 rank_1.0 的锁定 P3 六臂评估完成；Reacher confirmation 固定池 64/64 候选分支、共池重评分及 LeWM-vs-Joint 任务交互均完成，结果见本报告末尾。Reacher seed-4096 Recorded-control 新臂已在 GPU3 启动；其初始化权重 SHA、episode split SHA、action normalizer 和训练窗口数与同 seed Joint 臂一致。启动后 GPU3 仍保留约 8.2 GiB 空闲显存；主机总 swap 约 79 GiB、可用约 72 GiB。
- 2026-10-01 00:43 CST 更新：PushT seed-4096 B-only 已在 GPU2 启动；初始化权重 SHA、episode split SHA、normalizer 和窗口数均与对应 Joint 臂匹配。对当前已生成的 PushT/Reacher × seed 3072/4096 Joint、Recorded-control、B-only metadata 逐组复核，可比训练臂的初始权重、split、normalizer 和窗口数全部匹配；Reacher seed-4096 B-only 仍排队。启动采样时 GPU2 尚有约 27.2 GiB 空闲，模型仍在初始化；需在下一次规定间隔检查确认训练已进入首轮并复核峰值余量。
- 2026-10-01 00:49 CST 复核：新增 PushT/4096 B-only 已进入 epoch 0、750/13,796 step，GPU2 峰值后保留 8,205 MiB。其余新增臂进度为 PushT/3072 Recorded-control 12,150/13,796、B-only 5,150/13,796；Reacher/3072 Recorded-control 和 B-only 均为 7,650、5,150/13,034；PushT/4096 Recorded-control 6,750/13,796；Reacher/4096 Recorded-control 2,450/13,034。Reacher/4096 Joint 在 00:41 的日志为 epoch 9、9,500/13,034。GPU0–3 当前各保留约 8.0 GiB，主机 MemAvailable 约 373 GiB、swap 可用约 74 GiB、load 约 30/128 核；当前没有可再安全容纳一条同规格训练的卡。
- 2026-10-01 00:55 CST 快照：PushT/3072 Recorded-control 12,800/13,796、B-only 5,900/13,796；Reacher/3072 Recorded-control 与 B-only 分别 8,350/13,034、5,900/13,034；PushT/4096 Recorded-control 与 B-only 分别 7,400/13,796、1,500/13,796；Reacher/4096 Recorded-control 3,150/13,034，Joint 在 epoch 9 达 11,100/13,034。GPU0–3 均约 8.0 GiB free，主机 MemAvailable 约 368 GiB、swap 可用约 74 GiB、load 约 30/128 核；等 Reacher/4096 Joint 释放显存后补启动同 seed B-only。
- 2026-10-01 01:01 CST 快照：PushT/3072 Recorded-control 13,450/13,796、B-only 6,650/13,796；Reacher/3072 Recorded-control 与 B-only 为 9,000/13,034、6,650/13,034；PushT/4096 Recorded-control 与 B-only 为 8,050/13,796、2,250/13,796；Reacher/4096 Recorded-control 3,850/13,034，Joint epoch 9 达 11,800/13,034。GPU0–3 仍各约 8.0 GiB free，主机 MemAvailable 约 371 GiB、swap 可用约 74 GiB、load 约 30/128 核。
- 2026-10-01 01:08 CST 快照：PushT/3072 Recorded-control 已进入 epoch 1、350/13,796，B-only 7,450/13,796；Reacher/3072 Recorded-control 与 B-only 为 9,750、7,450/13,034；PushT/4096 Recorded-control 与 B-only 为 8,750、3,100/13,796；Reacher/4096 Recorded-control 4,600/13,034，Joint epoch 9 达 12,550/13,034。GPU0–3 各约 8.0 GiB free，主机 MemAvailable 约 369 GiB、swap 可用约 74 GiB、load 约 31/128 核。

## 统计与解释限制

- 新旧 cohort 均只保证与已登记历史评测 episode 互斥；对已有 checkpoint，不声称这些起点是训练未见数据。
- Scene/Humanoid 的 HDF5 schema 不提供 Round3 物理状态距离列；Phase1.7 按 episode 和 step_idx 抽取偏移 25 的图像目标、每 episode 至多一个起点，并记录互斥 train/dev/confirmation 清单。它们不具备核心任务的初始失败过滤和物理距离 regret，单独报告环境成功率。
- 旧 checkpoint 结果只回答“这两个已有系统在本协议中谁更有效”，不能隔离训练臂因果，也不能代表跨 seed 平均效果。当前新训练的 Joint 与独立 LeWM 也使用不同模型配置并分别随机初始化；即使 episode split、归一化和更新数匹配，Joint-B 对 LeWM 仍应解释为系统级排序比较，不能单独归因于联合训练。
- 同架构的 Joint、Recorded-control、B-only、Recorded-clean 消融须逐 seed 核验 `phase1_7_metadata.json` 中的 `initial_model_state_sha256`、episode split SHA 和 action normalizer 完全匹配；不匹配时不作训练机制因果结论，并用共享冻结初始权重重跑受影响臂。
- 显著性检验针对同 episode 的闭环成功结果；候选、环境内推理 seed 不作为独立样本。
- PushT/Reacher 固定池 confirmation、在线 rank_1.0 的锁定 P3 确认和共池重评分均已完成。在线确认在 PushT 上支持闭环成功率提升；Reacher 区间仍跨 0。新锁定固定池的连续排序两任务均偏向 B，但 epoch-10 匹配训练与 Recorded-control/B-only 消融仍未完成，不能据 epoch-2 结果作最终训练结论。

## 下一步

1. 完成当前 seed 3072/4096 的 Recorded-control、B-only 训练到预设 epoch 10；元数据已通过初始权重、episode split、normalizer 与窗口数匹配核验。
2. epoch-10 Joint-B/LeWM confirmation P3 配对分析已完成并按 seed 分开报告；该 cohort 曾用于旧 checkpoint/在线臂评估，结果不作为新独立 cohort 复制。待 B-only 最终评估完成后，再按预设完整比较族校正 p 值。
3. 主矩阵训练结束后，对 Recorded-control/B-only 与同 Joint actor 作预设的 dev/confirmation 比较，按方案执行 Holm 校正；Recorded-clean 和额外 seed 只在 GPU0–3 显存与主机负载满足安全预检时安排。
4. 在线分支改进的锁定确认已完成。继续将它与纯离线主线隔离，不把成功标签或物理距离用于训练，也不把与 P3 共用起点的固定池结果计作独立复验。
5. 汇总 final per-seed 结果后再决定论文表述：分别回答连续排序、闭环成功、训练机制和推理速度，不将未显著差异写成等价。

## 训练状态快照（15:04 CST）

八个主训练臂均继续推进：PushT/3072 Joint 为 epoch 4、9,000/13,796 step，LeWM 为 epoch 6、12,500/13,796；Reacher/3072 Joint 为 epoch 5、1,750/13,034，LeWM 为 epoch 6、2,900/13,034；PushT/4096 Joint 为 epoch 4、3,200/13,796，LeWM 为 epoch 6、12,900/13,796；Reacher/4096 Joint 为 epoch 4、8,200/13,034，LeWM 为 epoch 6、3,950/13,034。当前训练日志均有本轮更新。

资源快照：GPU0–3 利用率均为 100%，剩余显存依次约 14.2、13.2、13.0、8.0 GiB；GPU4–5 也处于 100% 利用率，且由其他进程占用约 25 GiB 显存；GPU6–7 空闲。主机 load average 为 54.73/54.98/54.75（128 核），可用内存约 157 GiB，但 39 GiB swap 中仅剩 127 MiB。由于 swap 未达到 8 GiB 缓存重试门槛，且普通进程预检的 2 GiB 下限也未满足，本轮不启动 Reacher LeWM 缓存或新训练。下次按至少 5 分钟间隔合并复查训练日志、GPU 与主机资源。

## 训练状态快照（15:11 CST）

八个主训练臂均有进展：PushT/3072 Joint epoch 4、9,750/13,796，LeWM epoch 6、13,750/13,796；Reacher/3072 Joint epoch 5、2,500/13,034，LeWM epoch 6、3,900/13,034；PushT/4096 Joint epoch 4、3,950/13,796，LeWM epoch 7、300/13,796；Reacher/4096 Joint epoch 4、8,950/13,034，LeWM epoch 6、5,050/13,034。

GPU0–3 利用率分别为 62%、100%、100%、89%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 由其他进程占用约 25 GiB 显存，利用率为 90% 和 10%；GPU6–7 空闲。主机 load average 为 55.35/55.54/55.06（128 核），可用内存约 157 GiB，swap 仅余 435 MiB。主机内存和 GPU 可用量未同时达到新训练预检要求，swap 仍低于缓存预检线，保持等待。

## 训练状态快照（15:17 CST）

八个主训练臂继续推进：PushT/3072 Joint epoch 4、10,450/13,796，LeWM epoch 7、1,100/13,796；Reacher/3072 Joint epoch 5、3,250/13,034，LeWM epoch 6、4,850/13,034；PushT/4096 Joint epoch 4、4,650/13,796，LeWM epoch 7、1,500/13,796；Reacher/4096 Joint epoch 4、9,650/13,034，LeWM epoch 6、6,150/13,034。

GPU0–3 利用率均为 100%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 暂无计算利用率但仍被其他进程占用约 25 GiB 显存；GPU6–7 空闲。主机 load average 为 46.56/52.87/54.28（128 核），可用内存约 159 GiB，swap 回升至 1.3 GiB，仍低于普通预检 2 GiB 和缓存重试 8 GiB 的门槛。无需也不应抢占外部 GPU 进程；继续等待训练和 swap 余量恢复。

## 训练状态快照（15:24 CST）

八个训练臂都有进展：PushT/3072 Joint epoch 4、11,150/13,796，LeWM epoch 7、2,300/13,796；Reacher/3072 Joint epoch 5、4,000/13,034，LeWM epoch 6、5,800/13,034；PushT/4096 Joint epoch 4、5,350/13,796，LeWM epoch 7、2,700/13,796；Reacher/4096 Joint epoch 4、10,350/13,034，LeWM epoch 6、7,250/13,034。

GPU0–3 利用率为 94%、100%、100%、95%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 当前利用率为 0%，但仍由其他进程占用约 25 GiB 显存；GPU6–7 空闲。主机 load average 为 56.93/53.69/53.97（128 核），可用内存约 155 GiB，swap 降至 148 KiB。缓存和训练预检都不满足，继续等待，不增加作业。

## 训练状态快照（15:30 CST）

八个训练臂继续推进：PushT/3072 Joint epoch 4、11,800/13,796，LeWM epoch 7、3,450/13,796；Reacher/3072 Joint epoch 5、4,700/13,034，LeWM epoch 6、6,600/13,034；PushT/4096 Joint epoch 4、6,000/13,796，LeWM epoch 7、3,750/13,796；Reacher/4096 Joint epoch 4、11,000/13,034，LeWM epoch 6、8,250/13,034。

GPU0–3 利用率均为 100%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 仍占用约 25 GiB 显存，当前利用率为 100% 和 76%；GPU6–7 空闲。主机 load average 为 57.64/56.45/55.07（128 核），可用内存约 160 GiB，swap 仅余 280 KiB。等待 Reacher LeWM 的剩余特征缓存与后续在线实验，不在预检失败时启动。

## 训练状态快照（15:36 CST）

八个训练臂继续更新：PushT/3072 Joint epoch 4、12,450/13,796，LeWM epoch 7、4,550/13,796；Reacher/3072 Joint epoch 5、5,350/13,034，LeWM epoch 6、7,500/13,034；PushT/4096 Joint epoch 4、6,600/13,796，LeWM epoch 7、4,850/13,796；Reacher/4096 Joint epoch 4、11,650/13,034，LeWM epoch 6、9,250/13,034。

GPU0–3 利用率为 100%、100%、100%、90%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 的其他进程占用约 25 GiB 显存，利用率为 51% 和 40%；GPU6–7 空闲。主机 load average 为 55.51/56.65/55.65（128 核），可用内存约 158 GiB，swap 仅余 344 KiB。训练进展正常，但缓存和新训练仍不满足主机资源预检。

## 训练状态快照（15:42 CST）

八个主训练臂继续推进：PushT/3072 Joint epoch 4、13,100/13,796，LeWM epoch 7、5,700/13,796；Reacher/3072 Joint epoch 5、6,050/13,034，LeWM epoch 6、8,400/13,034；PushT/4096 Joint epoch 4、7,250/13,796，LeWM epoch 7、6,000/13,796；Reacher/4096 Joint epoch 4、12,300/13,034，LeWM epoch 6、10,300/13,034。

GPU0–3 利用率为 100%、100%、100%、94%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 仍占用约 25 GiB 显存且利用率为 100%；GPU6–7 空闲。主机 load average 为 54.24/55.32/55.38（128 核），可用内存约 159 GiB，swap 仅余 316 KiB。虽有部分训练臂接近当前 step 总数，但 GPU 尚未释放、主机预检失败，继续等待。

## 训练状态快照（15:49 CST）

PushT/3072 Joint 日志记录 epoch 4 已完成（该 epoch 用时 7,660.6 秒），但整项 10-epoch 训练仍在继续。其余进度为：PushT/3072 LeWM epoch 7、6,950/13,796；Reacher/3072 Joint epoch 5、6,850/13,034，LeWM epoch 6、9,400/13,034；PushT/4096 Joint epoch 4、8,000/13,796，LeWM epoch 7、7,300/13,796；Reacher/4096 Joint epoch 4、13,000/13,034，LeWM epoch 6、11,400/13,034。

GPU0–3 利用率为 100%、100%、100%、78%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 仍有其他进程占用约 25 GiB 显存（当前利用率 13% 和 0%）；GPU6–7 空闲。主机 load average 为 63.79/56.89/55.82（128 核），可用内存约 147 GiB，但 swap 仅余 76 KiB。部分臂完成当前 epoch 不等于终点，当前仍不满足缓存/新训练预检。

## 训练状态快照（15:55 CST）

PushT/3072 Joint 已进入 epoch 5、700/13,796；Reacher/4096 Joint 已进入 epoch 5、600/13,034。其余进度为：PushT/3072 LeWM epoch 7、8,100/13,796；Reacher/3072 Joint epoch 5、7,600/13,034，LeWM epoch 6、10,350/13,034；PushT/4096 Joint epoch 4、8,700/13,796，LeWM epoch 7、8,450/13,796；Reacher/4096 LeWM epoch 6、12,450/13,034。

GPU0–3 利用率均为 100%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 当前利用率为 0%，仍分别被其他进程占用约 25 GiB 显存；GPU6–7 空闲。主机 load average 为 51.50/55.19/55.65（128 核），可用内存约 158 GiB，swap 回升至 15 MiB，仍远低于普通预检的 2 GiB 与缓存重试的 8 GiB。训练持续推进，新增任务继续等待预检条件。

## 训练状态快照（16:01 CST）

八个主训练臂仍在运行：PushT/3072 Joint epoch 5、1,300/13,796，LeWM epoch 7、9,200/13,796；Reacher/3072 Joint epoch 5、8,250/13,034，LeWM epoch 6、11,200/13,034；PushT/4096 Joint epoch 4、9,300/13,796，LeWM epoch 7、9,550/13,796；Reacher/4096 Joint epoch 5、1,200/13,034，LeWM epoch 7、250/13,034。

GPU0–3 利用率均为 100%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 当前利用率为 0%，但仍各被其他进程占用约 25 GiB 显存；GPU6–7 空闲。主机 load average 为 54.63/54.76/55.32（128 核），可用内存约 150 GiB，swap 仅余 84 KiB。训练持续推进，缓存与训练启动条件均未满足。

## 训练状态快照（16:08 CST）

八个主训练臂均有新步数：PushT/3072 Joint epoch 5、2,050/13,796，LeWM epoch 7、10,450/13,796；Reacher/3072 Joint epoch 5、9,000/13,034，LeWM epoch 6、12,250/13,034；PushT/4096 Joint epoch 4、10,050/13,796，LeWM epoch 7、10,800/13,796；Reacher/4096 Joint epoch 5、1,950/13,034，LeWM epoch 7、1,350/13,034。

GPU0–3 利用率均为 100%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 当前利用率为 0%，但仍各被其他进程占用约 25 GiB 显存；GPU6–7 空闲。主机 load average 为 58.45/55.03/55.11（128 核），可用内存约 155 GiB，swap 仅余 284 KiB。已获准在 GPU4–7 上进行后续低显存阶段；当前仍因 swap 预检未通过而暂缓启动。

## 训练状态快照（16:13 CST）

八个主训练臂均继续更新：PushT/3072 Joint epoch 5、2,650/13,796，LeWM epoch 7、11,450/13,796；Reacher/3072 Joint epoch 5、9,650/13,034，LeWM epoch 6、13,000/13,034；PushT/4096 Joint epoch 4、10,650/13,796，LeWM epoch 7、11,850/13,796；Reacher/4096 Joint epoch 5、2,600/13,034，LeWM epoch 7、2,300/13,034。

GPU0–3 利用率为 100%、100%、95%、89%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 利用率为 0%，仍各被其他进程占用约 25 GiB 显存；GPU6–7 空闲。主机 load average 为 59.01/56.00/55.37（128 核），可用内存约 156 GiB，swap 仅余 88 KiB。GPU4–7 使用授权已记录，当前仍等待 swap 达到缓存预检线。

## 训练状态快照（16:19 CST）

八个主训练臂继续推进：PushT/3072 Joint epoch 5、3,300/13,796，LeWM epoch 7、12,550/13,796；Reacher/3072 Joint epoch 5、10,300/13,034，LeWM epoch 7、800/13,034；PushT/4096 Joint epoch 4、11,250/13,796，LeWM epoch 7、12,950/13,796；Reacher/4096 Joint epoch 5、3,200/13,034，LeWM epoch 7、3,250/13,034。

GPU0–3 利用率均为 100%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 利用率为 0%，但仍各被其他进程占用约 25 GiB 显存；GPU6–7 空闲。主机 load average 为 55.73/54.50/54.79（128 核），可用内存约 152 GiB，swap 仅余 240 KiB。已有低显存任务获准使用 GPU4–7，但主机预检仍阻止其启动。

## 训练状态快照（16:25 CST）

八个主训练臂均继续推进：PushT/3072 Joint epoch 5、3,950/13,796，LeWM epoch 7、13,600/13,796；Reacher/3072 Joint epoch 5、10,950/13,034，LeWM epoch 7、1,700/13,034；PushT/4096 Joint epoch 4、11,900/13,796，LeWM epoch 8、150/13,796；Reacher/4096 Joint epoch 5、3,850/13,034，LeWM epoch 7、4,200/13,034。

GPU0–3 利用率均为 100%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 当前利用率为 0%，但仍各被其他进程占用约 25 GiB 显存；GPU6–7 空闲。主机 load average 为 56.80/55.25/54.94（128 核），可用内存约 150 GiB，swap 仅余 224 KiB。训练进展正常，剩余缓存和后续在线训练继续等待预检。

## 训练状态快照（16:30 CST）

八个主训练臂继续推进：PushT/3072 Joint epoch 5、4,450/13,796，LeWM epoch 8、600/13,796；Reacher/3072 Joint epoch 5、11,500/13,034，LeWM epoch 7、2,400/13,034；PushT/4096 Joint epoch 4、12,450/13,796，LeWM epoch 8、1,000/13,796；Reacher/4096 Joint epoch 5、4,350/13,034，LeWM epoch 7、4,950/13,034。

GPU0–3 利用率均为 100%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 利用率为 67% 和 47%，各有其他进程占用约 25 GiB 显存；GPU6–7 空闲。主机 load average 为 52.77/54.14/54.61（128 核），可用内存约 155 GiB，swap 回升到 728 KiB，仍低于最低预检线。继续等待，不追加任务。

## 训练状态快照（16:36 CST）

八个训练臂均在运行：PushT/3072 Joint epoch 5、5,050/13,796，LeWM epoch 8、1,650/13,796；Reacher/3072 Joint epoch 5、12,150/13,034，LeWM epoch 7、3,300/13,034；PushT/4096 Joint epoch 4、13,050/13,796，LeWM epoch 8、2,100/13,796；Reacher/4096 Joint epoch 5、5,000/13,034，LeWM epoch 7、5,900/13,034。

GPU0–3 利用率均为 100%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 当前利用率为 50% 和 6%，各仍有其他进程占用约 25 GiB 显存；GPU6–7 空闲。主机 load average 为 53.52/55.13/55.00（128 核），可用内存约 157 GiB，swap 升至 3.5 GiB。该值高于基础 2 GiB 预检线但低于在线缓存重试的 8 GiB 门槛，鉴于先前剧烈波动，等待稳定后再启动。

16:37 CST 的只读稳定性复核发现，free swap 在约一分钟内从 3.5 GiB 降至 304 KiB，而 MemAvailable 仍约 156 GiB。这确认了瞬时 swap 读数不适合作为启动条件；继续使用 8 GiB 缓存重试线并要求稳定复核。

## 训练状态快照（16:40 CST）

八个训练臂继续推进：PushT/3072 Joint epoch 5、5,550/13,796，LeWM epoch 8、2,450/13,796；Reacher/3072 Joint epoch 5、12,650/13,034，LeWM epoch 7、3,950/13,034；PushT/4096 Joint epoch 4、13,500/13,796，LeWM epoch 8、2,900/13,796；Reacher/4096 Joint epoch 5、5,500/13,034，LeWM epoch 7、6,600/13,034。

GPU0–3 利用率为 100%、100%、95%、100%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 当前利用率为 0%，仍各有其他进程占用约 25 GiB 显存；GPU6–7 空闲。主机 load average 为 53.39/53.62/54.35（128 核），可用内存约 152 GiB，swap 为 788 KiB。前次 3.5 GiB 峰值未保持，继续按稳定 8 GiB 门槛等待。

## 训练状态快照（16:46 CST）

八个主训练臂继续更新：PushT/3072 Joint epoch 5、6,150/13,796，LeWM epoch 8、3,550/13,796；Reacher/3072 Joint epoch 6、150/13,034，LeWM epoch 7、4,850/13,034；PushT/4096 Joint epoch 5、300/13,796，LeWM epoch 8、3,950/13,796；Reacher/4096 Joint epoch 5、6,150/13,034，LeWM epoch 7、7,550/13,034。

GPU0–3 利用率均为 100%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 当前利用率为 0%，仍各被其他进程占用约 25 GiB 显存；GPU6–7 空闲。主机 load average 为 56.90/55.62/54.89（128 核），可用内存约 151 GiB，swap 仅余 224 KiB。低显存缓存仍等待稳定达到 8 GiB 重试门槛。

## 训练状态快照（16:51 CST）

八个主训练臂均继续推进：PushT/3072 Joint epoch 5、6,800/13,796，LeWM epoch 8、4,600/13,796；Reacher/3072 Joint epoch 6、800/13,034，LeWM epoch 7、5,700/13,034；PushT/4096 Joint epoch 5、900/13,796，LeWM epoch 8、5,050/13,796；Reacher/4096 Joint epoch 5、6,800/13,034，LeWM epoch 7、8,500/13,034。

GPU0–3 利用率均为 100%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 当前利用率为 0%，仍各被其他进程占用约 25 GiB 显存；GPU6–7 空闲。主机 load average 为 48.45/51.82/53.52（128 核），可用内存约 152 GiB，swap 仅余 260 KiB。训练正常推进，缓存仍不满足 8 GiB 重试线。

## 训练状态快照（16:57 CST）

八个主训练臂继续更新：PushT/3072 Joint epoch 5、7,400/13,796，LeWM epoch 8、5,650/13,796；Reacher/3072 Joint epoch 6、1,450/13,034，LeWM epoch 7、6,600/13,034；PushT/4096 Joint epoch 5、1,550/13,796，LeWM epoch 8、6,100/13,796；Reacher/4096 Joint epoch 5、7,450/13,034，LeWM epoch 7、9,400/13,034。

GPU0–3 利用率均为 100%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 仍被其他进程占用约 25 GiB 显存，利用率为 100% 和 96%；GPU6–7 空闲。主机 load average 为 55.93/51.37/52.40（128 核），可用内存约 155 GiB，swap 仅余 712 KiB。低显存阶段继续等预检，训练推进正常。

## 训练状态快照（17:03 CST）

八个主训练臂持续更新：PushT/3072 Joint epoch 5、8,100/13,796，LeWM epoch 8、6,800/13,796；Reacher/3072 Joint epoch 6、2,150/13,034，LeWM epoch 7、7,550/13,034；PushT/4096 Joint epoch 5、2,200/13,796，LeWM epoch 8、7,250/13,796；Reacher/4096 Joint epoch 5、8,100/13,034，LeWM epoch 7、10,400/13,034。

GPU0–3 利用率均为 100%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 的其他任务占用约 25 GiB 显存，利用率为 100% 和 73%；GPU6–7 空闲。主机 load average 为 55.31/53.57/53.11（128 核），可用内存约 154 GiB，swap 回升到 1.1 GiB，仍低于最低预检线。继续等待安全启动窗口。

## 训练与swap快照（17:09 CST）

八个训练臂继续推进：PushT/3072 Joint epoch 5、8,750/13,796，LeWM epoch 8、7,950/13,796；Reacher/3072 Joint epoch 6、2,850/13,034，LeWM epoch 7、8,450/13,034；PushT/4096 Joint epoch 5、2,850/13,796，LeWM epoch 8、8,350/13,796；Reacher/4096 Joint epoch 5、8,800/13,034，LeWM epoch 7、11,350/13,034。

当时 free swap 瞬时升至约 6.0 GiB；约一分钟后的稳定性样本已降至 176 KiB，MemAvailable 约 154 GiB，因此没有启动缓存。GPU0–3 仍满载，GPU4–5 有外部工作负载，GPU6–7 空闲。Reacher/LeWM prepare 命令与 checkpoint 已定位；达标后使用 `CUDA_VISIBLE_DEVICES=6`、`--gpu 6` 启动，缓存预检继续要求 8 GiB 稳定 swap。

## 训练状态快照（17:17 CST）

八个主训练臂继续推进：PushT/3072 Joint epoch 5、9,550/13,796，LeWM epoch 8、9,350/13,796；Reacher/3072 Joint epoch 6、3,750/13,034，LeWM epoch 7、9,550/13,034；PushT/4096 Joint epoch 5、3,700/13,796，LeWM epoch 8、9,750/13,796；Reacher/4096 Joint epoch 5、9,650/13,034，LeWM epoch 7、12,600/13,034。

GPU0–3 利用率为 100%、100%、100%、93%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 利用率为 0%，仍各被其他进程占用约 25 GiB 显存；GPU6–7 空闲。主机 load average 为 59.51/57.97/55.33（128 核），可用内存约 153 GiB，swap 仅余 764 KiB。训练推进正常，Reacher LeWM 缓存继续等待稳定资源窗口。

## 训练状态快照（17:23 CST）

八个主训练臂均继续更新：PushT/3072 Joint epoch 5、10,200/13,796，LeWM epoch 8、10,450/13,796；Reacher/3072 Joint epoch 6、4,400/13,034，LeWM epoch 7、10,450/13,034；PushT/4096 Joint epoch 5、4,350/13,796，LeWM epoch 8、10,900/13,796；Reacher/4096 Joint epoch 5、10,300/13,034，LeWM epoch 8、450/13,034。

GPU0–3 利用率为 100%、100%、57%、100%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 利用率为 0%，仍各被其他进程占用约 25 GiB 显存；GPU6–7 空闲。主机 load average 为 54.59/57.15/55.90（128 核），可用内存约 155 GiB，swap 仅余 116 KiB。训练任务均在推进，缓存重试未达资源条件。

## 训练状态快照（17:27 CST）

八个主训练臂继续推进：PushT/3072 Joint epoch 5、10,650/13,796，LeWM epoch 8、11,150/13,796；Reacher/3072 Joint epoch 6、4,850/13,034，LeWM epoch 7、11,050/13,034；PushT/4096 Joint epoch 5、4,750/13,796，LeWM epoch 8、11,600/13,796；Reacher/4096 Joint epoch 5、10,700/13,034，LeWM epoch 8、1,050/13,034。

GPU0–3 利用率均为 100%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 当前利用率为 0%，仍各有其他进程占用约 25 GiB 显存；GPU6–7 空闲。主机 load average 为 59.38/57.20/56.12（128 核），可用内存约 149 GiB，swap 仅余 136 KiB。缓存命令已就绪，仍等待稳定达到 8 GiB 门槛。

## 训练状态快照（17:33 CST）

八个主训练臂均继续推进：PushT/3072 Joint epoch 5、11,300/13,796，LeWM epoch 8、12,250/13,796；Reacher/3072 Joint epoch 6、5,500/13,034，LeWM epoch 7、11,900/13,034；PushT/4096 Joint epoch 5、5,400/13,796，LeWM epoch 8、12,650/13,796；Reacher/4096 Joint epoch 5、11,350/13,034，LeWM epoch 8、2,000/13,034。

GPU0–3 利用率为 100%、87%、100%、100%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 当前利用率为 0%，仍各被其他进程占用约 25 GiB 显存；GPU6–7 空闲。主机 load average 为 61.07/57.59/56.41（128 核），可用内存约 155 GiB，swap 为 1.4 GiB，尚未达到基础预检线。继续等待。

## 训练状态快照（17:39 CST）

八个主训练臂继续推进：PushT/3072 Joint epoch 5、11,900/13,796，LeWM epoch 8、13,350/13,796；Reacher/3072 Joint epoch 6、6,200/13,034，LeWM epoch 7、12,750/13,034；PushT/4096 Joint epoch 5、6,050/13,796，LeWM epoch 8、13,750/13,796；Reacher/4096 Joint epoch 5、12,000/13,034，LeWM epoch 8、3,000/13,034。

GPU0–3 利用率为 100%、4%、100%、100%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 当前利用率为 0%，仍各被其他进程占用约 25 GiB 显存；GPU6–7 空闲。主机 load average 为 53.68/56.29/56.30（128 核），可用内存约 156 GiB，swap 升至 4.1 GiB但未达到8 GiB重试线。鉴于波动性，继续等待稳定余量。

17:40 CST 的只读稳定性复核显示，free swap 在约一分钟内从 4.1 GiB 降至 12 KiB，MemAvailable 仍约 153 GiB。低显存缓存未启动；需等待主机swap稳定恢复，而不是仅因 GPU6/7 空闲而开始。

## 训练状态快照（17:45 CST）

八个主训练臂均有新步数：PushT/3072 Joint epoch 5、12,600/13,796，LeWM epoch 9、550/13,796；Reacher/3072 Joint epoch 6、6,900/13,034，LeWM epoch 8、450/13,034；PushT/4096 Joint epoch 5、6,700/13,796，LeWM epoch 9、1,000/13,796；Reacher/4096 Joint epoch 5、12,650/13,034，LeWM epoch 8、4,000/13,034。

GPU0–3 利用率为 100%、100%、100%、97%，可用显存约 14.2、13.2、13.0、8.0 GiB；GPU4–5 当前利用率为 0%，仍各有其他进程占用约 25 GiB 显存；GPU6–7 空闲。主机 load average 为 54.70/57.86/57.23（128 核），可用内存约 146 GiB，swap 仅余 204 KiB。训练继续推进，缓存和新增梯度训练均等待预检通过。

## 训练状态快照（17:51 CST）

八个主训练臂继续运行：PushT/3072 Joint epoch 5、13,200/13,796，LeWM epoch 9、1,600/13,796；Reacher/3072 Joint epoch 6、7,600/13,034，LeWM epoch 8、1,300/13,034；PushT/4096 Joint epoch 5、7,350/13,796，LeWM epoch 9、2,050/13,796；Reacher/4096 Joint epoch 6、100/13,034，LeWM epoch 8、4,900/13,034。GPU0–3 满载；GPU4–5 的外部任务活跃；GPU6–7 空闲。主机可用内存约157 GiB，swap约3.1 GiB，仍低于8 GiB缓存线。

## Swap扩容可行性与审批状态

只读检查发现原有 `/swap.img`（8 GiB）和 `/swapfile`（32 GiB）几乎全部使用；根分区 `/dev/sda2` 为 ext4，约116 GiB可用，但实验账号不是root。自动审批拒绝了代理创建并启用临时16 GiB swapfile的特权请求，理由是共享主机的全局资源与I/O影响；代理没有执行系统级修改。用户选择自行扩容，并报告将总swap提高到约80 GiB。18:06后的只读检查确认新增 `/swapfile-phase1-7`（40 GiB）已启用，且未显示代理修改过 `/etc/fstab`。

## 训练与主机状态快照（18:01 CST）

八个训练臂继续推进：PushT/3072 Joint epoch 6、500/13,796，LeWM epoch 9、3,550/13,796；Reacher/3072 Joint epoch 6、8,800/13,034，LeWM epoch 8、2,900/13,034；PushT/4096 Joint epoch 5、8,500/13,796，LeWM epoch 9、4,050/13,796；Reacher/4096 Joint epoch 6、1,300/13,034，LeWM epoch 8、6,650/13,034。

GPU0–3 利用率均为100%，剩余显存约14.2、13.2、13.0、8.0 GiB；GPU4–5 利用率为0%，但仍各有约25 GiB显存被占用；GPU6和7利用率为0%，分别有约2.0 GiB和0.6 GiB显存占用。主机load average为60.77/58.80/58.12（128核），可用内存约144 GiB，swap约3.6 GiB。`swapon --show`仅列出原有8 GiB与32 GiB swapfile，尚未看到新增文件；缓存继续等待稳定超过8 GiB。


## Reacher LeWM缓存恢复（18:10 CST）

用户手动启用40 GiB `/swapfile-phase1-7` 后，系统总swap约79 GiB；free swap从约38 GiB在一分钟后保持约37 GiB。低显存缓存选择空闲的GPU7（GPU6已有约2 GiB占用）。首次在沙箱内运行因 `nvidia-smi` 无法访问驱动而在预检阶段退出，没有开始计算。按仓库要求，在相同 `CUDA_VISIBLE_DEVICES=7` 限制下提权重跑：主机预检通过（free swap 37,150 MiB、MemAvailable 162,809 MiB、load/core 0.451），GPU7预检通过（48,502 MiB free、0%利用率）。随后已从 Reacher HDF5 加载 action 数据，特征缓存任务正在执行；下次进度检查至少等待5分钟。

## Reacher LeWM缓存完成（18:18 CST）

`prepare` 命令以退出码0完成，生成分支和离线特征缓存：`outputs/round5/phase1_7/online_training/reacher/lewm/cache/branch_latents.pt`（44 MiB）与 `offline_latents.pt`（26 MiB），对应的两个 identity JSON 也已写入。GPU7已释放；该阶段没有触碰GPU0–3上的训练。此时主机可用内存约160 GiB、总swap约79 GiB/free约33 GiB；GPU0–3仍各有训练进程，故在线梯度微调继续等待安全空档。

18:18训练日志：PushT/3072 Joint epoch 6、2,300/13,796，LeWM epoch 9、6,600/13,796；Reacher/3072 Joint epoch 6、10,700/13,034，LeWM epoch 8、5,400/13,034；PushT/4096 Joint epoch 5、10,350/13,796，LeWM epoch 9、7,150/13,796；Reacher/4096 Joint epoch 6、3,150/13,034，LeWM epoch 8、9,300/13,034。GPU0–3利用率均为100%，GPU4–5仍由外部进程占用显存，GPU6–7空闲。

## Swap扩容后训练状态（18:27 CST）

用户手动新增的40 GiB `/swapfile-phase1-7` 已启用；18:27只读核验显示总swap约79 GiB、已用47 GiB、可用32 GiB，主机 `MemAvailable` 约159 GiB。代理没有修改swap配置。

18:26训练日志显示八个训练臂均继续更新：PushT/3072 Joint epoch 6、3,200/13,796，LeWM epoch 9、8,150/13,796；Reacher/3072 Joint epoch 6、11,600/13,034，LeWM epoch 8、6,700/13,034；PushT/4096 Joint epoch 5、11,200/13,796，LeWM epoch 9、8,650/13,796；Reacher/4096 Joint epoch 6、4,050/13,034，LeWM epoch 8、10,650/13,034。八个训练进程仍运行。

18:24左右的GPU只读状态为：GPU0–3利用率均100%，空闲显存约14.55/13.54/13.31/8.21 GiB；GPU4–5有外部进程占用约25 GiB显存，GPU6–7空闲。18:27主机load average为49.26/53.67/59.27（128核）。在线梯度训练仍须等待GPU0–3的安全空档。

## 在线微调：离线对照臂完成（18:54 CST）

在线训练首次启动在共享 schedule 生成阶段暴露索引布局错误：缓存潜变量展平顺序为 `state * 32 + candidate`，代码却 reshape 后仍用展平索引。该次尝试在任何梯度更新前退出。已修复 `scripts/round5_phase1_7_online_train.py`，让未来潜变量与目标潜变量保持与分支行索引一致的展平布局；按相同显卡可见性提权重跑后，schedule 生成成功，未触及训练标签。

PushT/Reacher × Joint/LeWM 四个离线续训臂均完成2,000次更新，batch 128、lr 1e-5，使用各自相同的分支与离线缓存并生成后续臂共用 schedule。结果位于 `outputs/round5/phase1_7/online_training/{pusht,reacher}/{joint,lewm}/arms/offline/`，每臂均生成 `training_result.json` 和 `checkpoints/offline_policy.ckpt`。末步训练损失分别为 PushT/Joint 0.3476、PushT/LeWM 0.0260、Reacher/Joint 0.1285、Reacher/LeWM 1.0160；这些是不同评分器各自的训练目标诊断值，不能当作跨模型排序优劣比较。

## 在线微调：分支预测臂（19:02 CST）

四个 `branches` 臂已按固定 schedule 启动。19:02快照显示 PushT/Joint 与 Reacher/Joint 完成2,000次更新；PushT/LeWM 到1,600次，Reacher/LeWM 到1,800次。各自训练日志和 `training_result.json` 位于对应 `arms/branches/` 目录。GPU0–2可用显存约27.2/26.2/13.0 GiB，GPU3约8.2 GiB；当前四卡利用率均接近或达到100%。

19:02主机load average为123.01/130.03/93.82（128核），短时每核负载已高于训练预检上限0.75；`MemAvailable`约182 GiB。总swap约79 GiB、已用约58 GiB、可用约21 GiB。分支臂完成后，等待CPU负载回到预检阈值内，再继续λ=0.1与λ=1.0排序臂，避免在高负载时叠加训练。

## 在线排序臂与 Reacher/LeWM checkpoint 锁定（19:22 CST）

四个分支预测臂全部完成2,000次更新。PushT/Joint、PushT/LeWM、Reacher/Joint的λ=0.1臂也各完成2,000次更新；这些训练损失只记录优化过程，尚不能据此判断排序性能。

Reacher/LeWM原缓存和已完成的offline/branches臂引用可变路径 `.../reacher_s3072_lewm_retry2/checkpoints/..._policy.ckpt`，缓存基线哈希为`10ad8455…`。主训练于19:09写入epoch 9终态后，该路径内容变化；λ=0.1训练的哈希校验因此拒绝启动，没有进行梯度更新。为保持epoch 2分支数据下的四臂同源，保留旧产物并另建隔离根目录 `outputs/round5/phase1_7/online_training/reacher/lewm_epoch2/`，用不可变快照 `epoch2_checkpoint_snapshots/reacher_s3072/lewm/checkpoints/..._weights_epoch_2.pt`（哈希`9109a462…`）重新生成缓存。19:22缓存准备完成；接下来在该目录下重跑offline、branches及两个λ排序臂。旧Reacher/LeWM三臂不并入这组匹配比较。

19:22主机load average为35.13/55.02/68.66（128核），`MemAvailable`约397 GiB，总swap约79 GiB、可用约61 GiB。GPU0–3可用显存约28.3/27.3/13.3/8.2 GiB；其余三项λ=0.1训练已完成，等待启动下一批排序臂。

## 在线 epoch-2 评分器的锁定确认（2026-09-30 至 2026-10-01，已完成）

查看 confirmation 结果前，已在 [`locked_confirmation_selection.json`](../../../outputs/round5/phase1_7/locked_confirmation_selection.json) 锁定 PushT 和 Reacher 的 Joint-B `rank_1.0` 臂。两项都使用 seed-3072、epoch-2 actor；主比较为独立 LeWM epoch-2。PushT/Reacher 各评估 200 个锁定起点，协议为 P3。这里评估的是用真实分支数据微调后的 B，不能替代前文对原始 B 的判断。

| 任务 | 锁定在线 Joint-B rank_1.0 | 独立 LeWM | 差值 | 配对不一致起点（B-only / LeWM-only） | bootstrap 95% CI | 精确 McNemar p | 两任务 Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| PushT | 37/200 (18.5%) | 20/200 (10.0%) | +8.5 pp | 26 / 9 | [+3.0, +14.0] pp | 0.00599 | 0.01198 |
| Reacher | 30/200 (15.0%) | 22/200 (11.0%) | +4.0 pp | 26 / 18 | [−2.5, +10.5] pp | 0.29122 | 0.29122 |

PushT 的在线 B 在这组锁定起点上高于独立 LeWM，配对检验经两任务 Holm 校正后仍显著。Reacher 的点估计也偏向在线 B，但区间跨 0。当前可支持 PushT 上这个特定在线臂的改进；跨任务的优势尚未建立。

六个预先登记的 P3 臂现已完成：

| 任务 | Joint-B rank_1.0 | 原 Joint-B | 独立 LeWM | Joint offline | LeWM rank_1.0 | LeWM offline |
|---|---:|---:|---:|---:|---:|---:|
| PushT | 18.5% (37/200) | 14.0% (28/200) | 10.0% (20/200) | 12.0% (24/200) | 10.5% (21/200) | 12.0% (24/200) |
| Reacher | 15.0% (30/200) | 11.0% (22/200) | 11.0% (22/200) | 8.5% (17/200) | 9.0% (18/200) | 11.5% (23/200) |

Joint-B rank_1.0 对原 Joint-B 的差值为 PushT +4.5 pp（CI [−1.5, +10.5]，p=0.188）、Reacher +4.0 pp（CI [−2.5, +10.5]，p=0.291）。对等更新 Joint offline 的差值为两任务各 +6.5 pp（CI [+1.0, +12.0]，未校正 p=0.035）。对 LeWM rank_1.0 的差值为 PushT +8.0 pp（CI [+2.5, +13.5]，未校正 p=0.007）、Reacher +6.0 pp（CI [+0.5, +11.5]，未校正 p=0.058）。这些次要比较没有多重校正；主要检验仍是上表中的独立 LeWM 对照。

若只看未微调的 epoch-2 原 Joint-B 与独立 LeWM，P3 top-1 成功差为 PushT +4.0 pp（CI [−1.5, +9.5]，McNemar p=0.200）和 Reacher 0 pp（CI [−5.5, +5.5]，p=1.000），均无闭环成功差异证据。结合两个任务固定池的连续排序指标都偏向 Joint-B，这批数据呈现“连续后果排序偏向 Joint-B、闭环 top-1 尚未分出稳定差异”的分离。它回答了原始 B 是否比单独 LeWM 更会排序的一个局部结果，但只覆盖一个 epoch-2 actor、两个任务和一个 confirmation cohort；Reacher 固定池交叉验证已完成，结果见本节下方。

PushT 的 confirmation 固定池使用上述 P3 cohort 的前 100 个起点，每个起点 64 个相同物理动作候选；它与 P3 共用起点，属于另一个终点。Joint-B、LeWM 和在线臂均在同一候选池及同一真实分支结果上评分。

| PushT 固定池评分器 | Top-1 25 步成功 | Pairwise sign accuracy | Spearman | 物理 regret（越低越好） |
|---|---:|---:|---:|---:|
| 原 Joint-B | 3/100 (3%) | 0.558 | 0.165 | 35.84 |
| 在线 Joint-B rank_1.0 | 3/100 (3%) | 0.551 | 0.144 | 42.00 |
| 独立 LeWM | 3/100 (3%) | 0.459 | −0.119 | 58.99 |

原 Joint-B 与 LeWM 的 Top-1 成功相同（McNemar p=1.0，差值 0 pp，bootstrap 95% CI [−4, +4] pp）。配对 state bootstrap 显示 Joint-B−LeWM 的 pairwise sign accuracy 差为 +0.099（95% CI [+0.075, +0.124]），Spearman 差为 +0.284（95% CI [+0.212, +0.353]）；有效 regret 状态上的 LeWM−Joint-B 为 +23.11（n=95，95% CI [+16.60, +29.80]）。在线 Joint-B rank_1.0 与 LeWM 的 Top-1 也相同（McNemar p=1.0，差值 0 pp，95% CI [−4, +4]）；Joint-B−LeWM 的 pairwise sign accuracy 差为 +0.092（95% CI [+0.066, +0.119]），Spearman 差为 +0.263（95% CI [+0.187, +0.340]），regret 差为 −17.56（n=95，95% CI [−24.89, −10.23]）。因此本批 PushT 固定池显示连续排序和 regret 指向 Joint-B，Top-1 成功没有分出高下。

### Reacher 固定池与跨任务结果

Reacher confirmation 固定池已完成：100 个锁定起点、每个起点 64 个相同候选动作，共 6,400 条物理分支结果。原 Joint-B、独立 LeWM 和在线评分臂均在相同候选池及真实物理后果上评分。

| Reacher 固定池评分器 | Top-1 25 步成功 | Pairwise sign accuracy | Spearman | 物理 regret（越低越好） |
|---|---:|---:|---:|---:|
| 原 Joint-B | 5/100 (5%) | 0.553 | 0.151 | 0.734 |
| 在线 Joint-B rank_1.0 | 11/100 (11%) | 0.659 | 0.428 | 0.528 |
| 独立 LeWM | 10/100 (10%) | 0.506 | 0.014 | 0.758 |

对原 Joint-B 与独立 LeWM，Joint-B 的 pairwise sign accuracy 高 +0.0468（配对 state bootstrap 95% CI [+0.0246, +0.0687]），Spearman 高 +0.1363（CI [+0.0738, +0.1980]）。但 Top-1 成功率低 5 pp（LeWM−Joint-B 的 CI [−2, +12] pp，精确 McNemar p=0.267）；有效 regret 状态上，LeWM−Joint-B 为 +0.0238（n=86，CI [−0.1027, +0.1532]），区间跨 0。因此原始 B 在这批 Reacher 候选池中的连续排序更符合物理后果，但没有显示更高的 Top-1 成功率或更低的 regret。

锁定在线 Joint-B rank_1.0 对独立 LeWM 的 Reacher 固定池差值为：pairwise sign accuracy +0.1528（CI [+0.1269, +0.1770]）、Spearman +0.4134（CI [+0.3415, +0.4787]）、regret −0.2357（n=82，CI [−0.3357, −0.1373]）。Top-1 成功率只高 1 pp（CI [−7, +8] pp，精确 McNemar p=1.0）。这与 P3 上 +4 pp、Holm 校正 p=0.291 的结果一致：连续排序指标偏向在线 B，闭环成功优势在 Reacher 尚未确定。

固定池的任务×评分器交互分析显示，LeWM 相对原 Joint-B 的 pairwise 差在 Reacher 比 PushT 高 +0.0521（95% CI [+0.0189, +0.0850]），Spearman 差高 +0.1473（CI [+0.0561, +0.2427]）；Top-1 成功交互为 +0.05（CI [−0.03, +0.13]）。这表示 LeWM 相对 B 的连续排序差距在 Reacher 较小，但两个任务中估计都仍偏向 Joint-B。交互区间未对三个指标的扫描作多重校正。

**阶段性判断（旧 checkpoint、该锁定 cohort；不代表最终 epoch-10 结论）：**这批固定池结果曾回答“B 的动作排序是否比单独 LeWM 好”：原始 Joint-B 在 PushT 与 Reacher 上的 pairwise 和 Spearman 排序都更好；锁定在线 rank_1.0 臂在两任务的连续排序指标也优于独立 LeWM。优势尚未稳定转化为跨任务的 Top-1/闭环成功提升：原始 Joint-B 的 P3 差值在 PushT 为 +4 pp（CI [−1.5, +9.5]，p=0.200）、Reacher 为 0 pp（CI [−5.5, +5.5]，p=1.000）；在线 rank_1.0 对 LeWM 的 Reacher P3 差值为 +4 pp（Holm p=0.291）。因此，这批旧 checkpoint 固定池支持“B 的连续排序更符合该候选池的物理后果”，但后续 epoch-10 固定池没有复现稳定差异；这些结果均不支持“B 在控制成功率上普遍优于 LeWM”。固定池使用与 P3 相同 confirmation cohort 的前 100 个起点，是不同终点的交叉检查，不是独立重复；各固定池连续指标的区间为逐项区间，未作多重校正。

Reacher 固定池及共池重评分数据见 [`online_confirmation_fixed_pool_v1`](../../../outputs/round5/phase1_7/online_confirmation_fixed_pool_v1/)，原始 LeWM 与 Joint-B 的跨任务交互见 [`task_scorer_interaction_lewm_base.json`](../../../outputs/round5/phase1_7/online_confirmation_fixed_pool_v1/task_scorer_interaction_lewm_base.json)，在线评分臂重评分见 [`reacher_common_rescore_v2.json`](../../../outputs/round5/phase1_7/online_confirmation_fixed_pool_v1/reacher_common_rescore_v2.json)。P3 配对分析结果见 [`analysis.json`](../../../outputs/round5/phase1_7/locked_confirmation_closed_loop_v1/analysis.json)，分析入口为 [`round5_phase1_7_locked_confirmation_analysis.py`](../../../scripts/round5_phase1_7_locked_confirmation_analysis.py)。

### epoch 10 dev P3 与固定池交叉检查

使用两个 seed 的最终 epoch-10 Joint actor、同一 dev cohort 和每任务 100 个配对起点，比较 Joint-B 与独立 LeWM 评分。该 cohort 已用于此前的开发诊断，以下结果只作训练终点的探索性检查，不作 confirmation，也不算独立重复。

| Seed | 任务 | Joint-B | 独立 LeWM | LeWM−Joint-B | 配对 95% bootstrap CI | 精确 McNemar p |
|---:|---|---:|---:|---:|---:|---:|
| 3072 | PushT | 89/100 (89%) | 88/100 (88%) | −1 pp | [−7, +6] pp | 1.000 |
| 3072 | Reacher | 20/100 (20%) | 20/100 (20%) | 0 pp | [−10, +10] pp | 1.000 |
| 4096 | PushT | 94/100 (94%) | 95/100 (95%) | +1 pp | [−3, +5] pp | 1.000 |
| 4096 | Reacher | 25/100 (25%) | 16/100 (16%) | −9 pp | [−19, +1] pp | 0.122 |

两 seed 的 PushT 闭环差异都只有 1 pp 且方向相反；Reacher 在 seed-3072 持平、seed-4096 点估计偏向 Joint-B 9 pp，但区间仍包含 0。四项都是未校正的开发集检验。评估使用同一 Joint actor 生成候选，改变评分器，因此是同 A 下的 B 排序闭环比较；结果尚未显示稳定的跨 seed 闭环优势。分析 artifact 见 [`epoch10_s3072_main_dev_p3/analysis`](../../../outputs/round5/phase1_7/epoch10_s3072_main_dev_p3/analysis/) 和 [`epoch10_main_dev_p3/analysis`](../../../outputs/round5/phase1_7/epoch10_main_dev_p3/analysis/)。

### epoch 10 final checkpoint：冻结 confirmation cohort 上的 P3 配对比较

Joint-B 与独立 LeWM 使用各自 seed 的最终 epoch-10 权重和同一个 Joint actor，在 PushT/Reacher 各 200 个冻结起点上完成 P3。两模型在每一组内使用相同 actor、episode ID 和推理 seed。配对 bootstrap 95% CI 与精确 McNemar p 值如下；p 值尚未校正方案中的 Joint-B 对 LeWM、Joint-B 对 B-only 完整检验族。

| 训练 seed | 任务 | Joint-B | 独立 LeWM | LeWM−Joint-B | 配对 95% bootstrap CI | Discordant（LeWM-only / Joint-only） | 精确 p（未校正） |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3072 | PushT | 181/200 (90.5%) | 173/200 (86.5%) | −4.0 pp | [−8.0, 0.0] pp | 5 / 13 | 0.0963 |
| 3072 | Reacher | 42/200 (21%) | 50/200 (25%) | +4.0 pp | [−4.0, +11.5] pp | 36 / 28 | 0.3817 |
| 4096 | PushT | 180/200 (90%) | 171/200 (85.5%) | −4.5 pp | [−8.0, −1.0] pp | 2 / 11 | 0.0225 |
| 4096 | Reacher | 52/200 (26%) | 44/200 (22%) | −4.0 pp | [−11.5, +3.5] pp | 25 / 33 | 0.3581 |

PushT 两个 seed 的点估计都偏向 Joint-B；Reacher 的方向随训练 seed 翻转。除 seed-4096 PushT 外，逐项 CI 均跨 0；该项的 p=0.0225 仍是未校正结果，按预注册的完整比较族完成 B-only 比较后再作显著性判断。更重要的是，这 4 组复用同一 frozen confirmation cohort；旧 checkpoint 和在线 scorer 也曾在这些 episode 上评估，因此不能把它们当作新的独立 cohort 复制，也不把两个训练 seed 的 800 个 episode 直接合并。整体不支持“Joint-B 在两个任务、两个 seed 上都稳定提高闭环成功率”。分析文件见 [`epoch10_s3072_main_confirmation/analysis`](../../../outputs/round5/phase1_7/epoch10_s3072_main_confirmation/analysis/) 与 [`epoch10_s4096_main_confirmation/analysis`](../../../outputs/round5/phase1_7/epoch10_s4096_main_confirmation/analysis/)。

seed-4096 PushT/Reacher epoch-10 固定池物理回放均已完成：每任务 100 个状态、每状态 64 个共同候选，各 6,400 条分支。

| 任务 | Pairwise sign accuracy（Joint-B / LeWM） | Spearman（Joint-B / LeWM） | Top-1 25 步成功（Joint-B / LeWM） | Regret 均值（Joint-B / LeWM） |
|---|---:|---:|---:|---:|
| PushT | 0.505 / 0.499 | 0.012 / −0.006 | 16% / 13% | 25.76 / 23.49 |
| Reacher | 0.508 / 0.503 | 0.023 / 0.010 | 10% / 9% | 0.422 / 0.520 |

PushT 的 Joint-B−LeWM pairwise 差为 +0.006（state-bootstrap 95% CI [−0.028, +0.039]），Spearman 差 +0.018（CI [−0.072, +0.109]）；Top-1 差 +3 pp（CI [−3, +9] pp，精确 McNemar p=0.508）。LeWM−Joint-B 的 regret 差为 −1.45（n=85，CI [−5.72, +3.09]），点估计偏向 LeWM。Reacher 的 Joint-B−LeWM pairwise 差 +0.005（CI [−0.009, +0.020]），Spearman 差 +0.013（CI [−0.029, +0.056]）；Top-1 差 +1 pp（CI [−6, +8] pp，p=1.000）。Reacher 的 LeWM−Joint-B regret 差 +0.098（n=84，CI [+0.025, +0.168]），点估计偏向 Joint-B；该单项区间未经对多指标扫描的校正。两任务的相关性排序都接近机会水平，任务×评分器交互区间也包含 0。整体没有形成稳定一致的 epoch-10 排序胜者。

该 dev cohort 与 epoch-2 P3/固定池诊断复用起点，连续指标 CI 未作多重校正；结果是一个 seed 的开发集检查，不是 confirmation。原始结果和逐状态 bootstrap 见 [`epoch10_main_dev_fixed_pool`](../../../outputs/round5/phase1_7/epoch10_main_dev_fixed_pool/)，任务交互见 [`task_scoring_interaction_epoch10.json`](../../../outputs/round5/phase1_7/epoch10_main_dev_fixed_pool/task_scoring_interaction_epoch10.json)。

seed-3072 PushT 的 epoch-10 固定池也已完成 100 个状态、每状态 64 个共同候选。Joint-B/LeWM 的 pairwise sign accuracy 为 0.516/0.519，Spearman 为 0.042/0.049；B−LeWM 差分别为 −0.003（95% state-bootstrap CI [−0.031, +0.024]）与 −0.006（CI [−0.085, +0.073]），都接近机会水平且区间跨 0。top-1 25 步成功为 14/100 对 8/100，B−LeWM +6 pp（配对 bootstrap 95% CI [+2, +11] pp，精确 McNemar p=0.031）；候选池 oracle 成功覆盖为 91%，同池随机单选期望约 9.9%。这提示 B 在该 PushT dev cohort 的 top-1 选择有一个局部信号，但该检验未对同时查看的 top-k/连续指标作多重校正，也没有连续排序优势相伴。物理 regret 的 LeWM−B 配对差为 −0.10（n=88，95% CI [−4.98, +4.33]），没有明显差异。Reacher 的 seed-3072 固定池已完成，结果见下一段。逐状态分析见 [`epoch10_s3072_main_dev_fixed_pool/pusht`](../../../outputs/round5/phase1_7/epoch10_s3072_main_dev_fixed_pool/pusht/pusht_phase1_7_dev_16027_v1/)。

同一 seed 的 Reacher epoch-10 固定池也已完成。Joint-B/LeWM 的 pairwise sign accuracy 为 0.499/0.474，B−LeWM 差 +0.026（95% state-bootstrap CI [+0.004, +0.048]）；两者绝对排序都接近机会水平。Spearman 为 −0.004/−0.048，B−LeWM 差 +0.045（CI [−0.003, +0.094]），区间跨 0。Top-1 25 步成功均为 11/100（配对差 0 pp，95% CI [−7, +7] pp，精确 McNemar p=1.000）；LeWM−B 的物理 regret 配对差 +0.048（n=83，CI [−0.037, +0.135]）。按任务比较 LeWM−B 的差值，Reacher−PushT 的 pairwise 交互为 −0.029（95% CI [−0.066, +0.007]），Spearman 交互为 −0.051（CI [−0.146, +0.040]），Top-1 交互为 +0.06（CI [−0.03, +0.15]）；都没有排除 0。指标和交互区间未校正多指标扫描；这批结果提供了一个局部 pairwise 信号，不建立稳定的任务间排序优势。逐状态分析与交互文件见 [`epoch10_s3072_main_dev_fixed_pool/reacher`](../../../outputs/round5/phase1_7/epoch10_s3072_main_dev_fixed_pool/reacher/reacher_phase1_7_dev_16027_v1/) 和 [`task_scoring_interaction_epoch10.json`](../../../outputs/round5/phase1_7/epoch10_s3072_main_dev_fixed_pool/task_scoring_interaction_epoch10.json)。

### epoch 10 confirmation 子集固定池交叉检查

最终 checkpoint 在同一冻结 confirmation cohort 的前 100 个起点上生成 64 个共同候选并完成物理回放。PushT 两个 seed 的连续指标差都没有排除 0，且 seed-3072 的 top-1 闭环覆盖数值反而偏向 LeWM；这与完整 200-start P3 的 PushT 点估计偏向 Joint-B 不同，说明终点和起点子集会影响观测方向。以下结果复用 P3 episode，固定池 CI 和 top-k 检验均未对扫描作多重校正。

| Seed | Pairwise sign accuracy（Joint-B / LeWM） | B−LeWM 差及 95% CI | Spearman（Joint-B / LeWM） | B−LeWM 差及 95% CI | Top-1 25 步成功（Joint-B / LeWM） | LeWM−Joint-B regret 差及 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 3072 | 0.517 / 0.491 | +0.027 [−0.011, +0.067] | 0.041 / −0.026 | +0.067 [−0.034, +0.168] | 14% / 19% | −1.02 [−4.26, +2.10] |
| 4096 | 0.512 / 0.509 | +0.004 [−0.031, +0.041] | 0.026 / 0.023 | +0.003 [−0.089, +0.100] | 16% / 17% | −0.23 [−3.64, +3.50] |

Pairwise 与 Spearman 的绝对值均接近机会排序。Top-1 的 LeWM−Joint-B 差在 seed-3072 为 +5 pp（state-bootstrap 95% CI [0, +10] pp，精确 McNemar p=0.125），seed-4096 为 +1 pp（CI [−3, +5] pp，p=1.000）；均没有显著差异。物理 regret 的 LeWM−Joint-B 差分别为 −1.02（n=82，95% CI [−4.26, +2.10]）与 −0.23（n=81，CI [−3.64, +3.50]）。因此在这次 final-checkpoint 共池分析里，B 没有稳定的连续排序或 top-1 优势；seed-3072 的连续点估计偏向 B，但不确定区间跨 0，top-1 点估计方向则偏向 LeWM。分析文件见 [`seed-3072 PushT`](../../../outputs/round5/phase1_7/epoch10_s3072_main_confirmation_fixed_pool/pusht/pusht_phase1_7_confirmation_16028_v1_fixed_pool_100/success_coverage_analysis.json) 与 [`seed-4096 PushT`](../../../outputs/round5/phase1_7/epoch10_s4096_main_confirmation_fixed_pool/pusht/pusht_phase1_7_confirmation_16028_v1_fixed_pool_100/success_coverage_analysis.json)。Reacher 固定池现已完成。每项覆盖 100 个状态、每状态 64 个共同候选：

| Seed | Pairwise sign accuracy（Joint-B / LeWM） | Joint-B−LeWM 差及 95% CI | Spearman（Joint-B / LeWM） | Joint-B−LeWM 差及 95% CI | Top-1 25 步成功（Joint-B / LeWM） | LeWM−Joint-B regret 差及 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 3072 | 0.508 / 0.499 | +0.009 [−0.010, +0.029] | 0.024 / −0.003 | +0.028 [−0.018, +0.075] | 11% / 17% | −0.052（n=78）[−0.134, +0.029] |
| 4096 | 0.503 / 0.505 | −0.002 [−0.015, +0.012] | 0.009 / 0.015 | −0.006 [−0.046, +0.033] | 14% / 15% | −0.034（n=80）[−0.125, +0.054] |

Pairwise 与 Spearman 的绝对值都接近机会水平。seed-3072 的 Joint-B−LeWM 连续指标点估计略偏向 B，seed-4096 则接近持平；各区间均跨 0。Top-1 的 LeWM−Joint-B 差分别为 +6 pp（95% CI [−2, +14] pp，精确 McNemar p=0.238）与 +1 pp（CI [−6, +9] pp，p=1.000）；regret 区间也跨 0。

同 seed 的 PushT/Reacher 任务×评分器交互（定义为 Reacher 的 LeWM−Joint 差减去 PushT 的 LeWM−Joint 差）如下；交互区间未校正指标扫描：

| Seed | Pairwise sign accuracy 交互及 95% CI | Spearman 交互及 95% CI | Top-1 成功率交互及 95% CI |
|---:|---:|---:|---:|
| 3072 | +0.017 [−0.025, +0.061] | +0.039 [−0.068, +0.150] | +0.01 [−0.08, +0.11] |
| 4096 | +0.005 [−0.032, +0.046] | +0.009 [−0.094, +0.112] | 0.00 [−0.09, +0.09] |

因此，这批 final-checkpoint 100-start 共池结果没有显示稳定的 B 相对独立 LeWM 排序优势，也没有可靠的任务交互；Top-1 方向略偏 LeWM 但不确定。它复用了 P3 confirmation cohort 的同一批起点，是交叉终点检查，不是独立复验；连续指标区间及交互均未对扫描作多重校正。逐状态结果见 [`seed-3072 Reacher`](../../../outputs/round5/phase1_7/epoch10_s3072_main_confirmation_fixed_pool/reacher/reacher_phase1_7_confirmation_16028_v1_fixed_pool_100/success_coverage_analysis.json) 与 [`seed-4096 Reacher`](../../../outputs/round5/phase1_7/epoch10_s4096_main_confirmation_fixed_pool/reacher/reacher_phase1_7_confirmation_16028_v1_fixed_pool_100/success_coverage_analysis.json)，任务交互见两个 seed 的 [`3072`](../../../outputs/round5/phase1_7/epoch10_s3072_main_confirmation_fixed_pool/task_scoring_interaction_confirmation_100.json) / [`4096`](../../../outputs/round5/phase1_7/epoch10_s4096_main_confirmation_fixed_pool/task_scoring_interaction_confirmation_100.json) 文件。


### 全 200-start PushT 固定池扩展

最终 epoch-10 Joint actor 在完整的 200-start confirmation cohort 上生成 64 个共同候选并完成物理回放。与前述 100-start 结果相同，这些起点也用于 P3；这里是同一 cohort 的次要终点扩展，不是独立 confirmation。

| Seed | Pairwise sign accuracy（Joint-B / LeWM） | LeWM−Joint-B 差及 95% CI | Spearman（Joint-B / LeWM） | LeWM−Joint-B 差及 95% CI | Top-1 25 步成功（Joint-B / LeWM） | LeWM−Joint-B regret 差及 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 3072 | 0.514 / 0.515 | +0.002 [−0.022, +0.026] | 0.040 / 0.040 | −0.000 [−0.069, +0.067] | 14.5% / 16.0% | −2.38（n=168）[−4.98, +0.38] |
| 4096 | 0.500 / 0.497 | −0.003 [−0.027, +0.020] | 0.004 / −0.007 | −0.010 [−0.074, +0.053] | 15.5% / 17.5% | +0.57（n=163）[−2.77, +3.90] |

两个 seed 的绝对 pairwise accuracy 都约为 0.5，Spearman 也接近 0。连续指标区间均跨 0，seed-3072 的 pairwise 点估计略偏 LeWM、seed-4096 略偏 Joint-B；Spearman 差很小。Top-1 点估计分别偏向 LeWM +1.5 pp（95% CI [−1.5, +5.0] pp，精确 McNemar p=0.549）和 +2.0 pp（CI [−2, +6] pp，p=0.481），regret 点估计方向相反且区间跨 0。完整 200-start 结果没有显示 B 稳定优于单独 LeWM；它也说明前 100 起点上的点估计会随子集变化。逐状态统计见 [`seed-3072 PushT 200`](../../../outputs/round5/phase1_7/epoch10_s3072_main_confirmation_fixed_pool_200/pusht/pusht_phase1_7_confirmation_16028_v1/success_coverage_analysis.json) 与 [`seed-4096 PushT 200`](../../../outputs/round5/phase1_7/epoch10_s4096_main_confirmation_fixed_pool_200/pusht/pusht_phase1_7_confirmation_16028_v1/success_coverage_analysis.json)。

### 全 200-start Reacher 固定池扩展

Reacher 两个 seed 的完整 200-start 固定池均已完成。下表中的连续差按 Joint-B−LeWM 表示，成功率和 regret 的配对差按 LeWM−Joint-B 表示。

| Seed | Pairwise sign accuracy（Joint-B / LeWM） | Joint-B−LeWM 差及 95% CI | Spearman（Joint-B / LeWM） | Joint-B−LeWM 差及 95% CI | Top-1 25 步成功（Joint-B / LeWM） | LeWM−Joint-B regret 差及 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 3072 | 0.508 / 0.499 | +0.008 [−0.006, +0.023] | 0.022 / −0.001 | +0.023 [−0.010, +0.056] | 11.5% / 13.5% | +0.021（n=159）[−0.033, +0.075] |
| 4096 | 0.502 / 0.499 | +0.003 [−0.008, +0.014] | 0.005 / −0.002 | +0.007 [−0.026, +0.038] | 14.0% / 14.0% | −0.007（n=159）[−0.066, +0.053] |

两 seed 的绝对连续排序都接近机会水平。Top-1 的 LeWM−Joint-B 差分别为 +2 pp（95% CI [−4, +8] pp，精确 McNemar p=0.618）和 0 pp（CI [−5.5, +5.5] pp，p=1.000）；regret 与连续排序区间均跨 0。因此这批完整 Reacher 固定池没有建立 B 稳定优于独立 LeWM 的证据。逐状态分析见 [`seed-3072`](../../../outputs/round5/phase1_7/epoch10_s3072_main_confirmation_fixed_pool_200/reacher/reacher_phase1_7_confirmation_16028_v1/success_coverage_analysis.json) 与 [`seed-4096`](../../../outputs/round5/phase1_7/epoch10_s4096_main_confirmation_fixed_pool_200/reacher/reacher_phase1_7_confirmation_16028_v1/success_coverage_analysis.json)。

把同 seed 的完整 PushT/Reacher 200-start 结果配对后，任务×评分器交互（定义为 Reacher 的 LeWM−Joint 差减去 PushT 的 LeWM−Joint 差）如下：

| Seed | Pairwise 交互及 95% CI | Spearman 交互及 95% CI | Top-1 交互及 95% CI |
|---:|---:|---:|---:|
| 3072 | −0.010 [−0.038, +0.019] | −0.023 [−0.097, +0.053] | +0.005 [−0.065, +0.070] |
| 4096 | +0.000 [−0.026, +0.027] | +0.004 [−0.066, +0.074] | −0.020 [−0.090, +0.050] |

所有交互区间均含 0。完整 200-start 固定池复用 P3 confirmation cohort；扩大状态数是同 cohort 的精度扩展，不是独立复验。交互文件见 [`seed-3072`](../../../outputs/round5/phase1_7/epoch10_s3072_main_confirmation_fixed_pool_200/task_scoring_interaction_confirmation_200.json) 与 [`seed-4096`](../../../outputs/round5/phase1_7/epoch10_s4096_main_confirmation_fixed_pool_200/task_scoring_interaction_confirmation_200.json)。

同一 seed-4096、同一 dev cohort 的 epoch-2 固定池曾显示更大的 Joint-B 连续排序差：pairwise 为 0.548/0.486（Joint-B−LeWM +0.062，95% CI [+0.030, +0.093]），Spearman 为 0.132/−0.040（差 +0.172，CI [+0.085, +0.263]）；Top-1 为 4%/2%（配对 p=0.625），regret 差的区间也跨 0。epoch-2 与 epoch-10 固定池由不同 epoch 的 Joint actor 生成，候选动作分布随之改变，因此这说明相对排序信号没有跨 checkpoint 复现，不能单独归因于 B 随训练变差。epoch-2 artifact 见 [`epoch2_matched_dev_100/pusht_s4096`](../../../outputs/round5/phase1_7/epoch2_matched_dev_100/pusht_s4096/pusht/pusht_phase1_7_dev_16027_v1/)；epoch-10 artifact 见 [`epoch10_main_dev_fixed_pool/pusht`](../../../outputs/round5/phase1_7/epoch10_main_dev_fixed_pool/pusht/pusht_phase1_7_dev_16027_v1/)。

## Recorded-control / B-only 重训状态（2026-10-01 08:10 CST）

在 2026-10-01 07:45 左右，PushT/Reacher × seed 3072/4096 的八个 Recorded-control 与 B-only 训练日志同时停止更新，并出现 `BrokenPipeError`。四个进程退出，另外四个在异常输出阶段挂住；截至 07:58，相关 metrics/checkpoint 已超过 13 分钟没有变化且 GPU 利用率为 0%。旧日志能确认管道在异常格式化/写入时报错，但无法恢复原始根异常。保留原目录及其部分 checkpoint；这些文件不含可精确恢复的优化器状态，因此中断产物不并入终点比较。

从 08:06 起，八臂均从原 seed/初始权重在独立 `retry3` 目录完整重跑，保持相同 episode split、训练 normalizer、10 epochs、batch 128、AdamW、lr 5e-5 和 weight decay 1e-3。标准输出和错误输出直接写入独立文件，并保留前台监控会话，避免再次依赖已关闭的 PTY 管道。

| GPU | 任务 / Seed | Recorded-control 目录 | B-only 目录 |
|---:|---|---|---|
| 0 | PushT / 3072 | `outputs/round5_phase1_7_pusht_s3072_recorded_control_retry3` | `outputs/round5_phase1_7_pusht_s3072_b_only_retry3` |
| 1 | Reacher / 4096 | `outputs/round5_phase1_7_reacher_s4096_recorded_control_retry3` | `outputs/round5_phase1_7_reacher_s4096_b_only_retry3` |
| 2 | PushT / 4096 | `outputs/round5_phase1_7_pusht_s4096_recorded_control_retry3` | `outputs/round5_phase1_7_pusht_s4096_b_only_retry3` |
| 3 | Reacher / 3072 | `outputs/round5_phase1_7_reacher_s3072_recorded_control_retry3` | `outputs/round5_phase1_7_reacher_s3072_b_only_retry3` |

08:09 启动核验时八个根进程均存活，最早启动的六臂已越过初始化并进入 epoch 0；GPU0–3 均为 100% 利用率，每卡用显存约 40.3/48 GiB，剩余约 8.0 GiB。主机 `MemAvailable` 约 384 GiB，79 GiB swap 中约 0.4 GiB 已用；GPU4–7 未用于训练。持久日志位于 `outputs/round5/phase1_7/restart_logs/*_retry3.log`。训练完成后仍须核验 `phase1_7_metadata.json` 中的初始权重哈希、split、normalizer 和窗口数，再纳入匹配消融比较。

08:12 对 retry3 的八份 `phase1_7_metadata.json` 完成启动前匹配审计：每份都与相应旧实验目录的初始权重哈希、episode split SHA、train window count 和完整 normalizer 值一致；每个 task/seed 内 Recorded-control 与 B-only 的这些字段也逐项一致。3072 初始权重 SHA 前缀为 `c42c3ab40cef`，4096 为 `4bc644abe790`；PushT split SHA 前缀为 `e6a791ae7146`（1,765,923 个 train windows），Reacher 为 `8b2bb172c91f`（1,668,400 个 train windows）。metadata 匹配门槛已通过；采样顺序仍按相同 seed/配置配对，训练结束时再核对实际优化步数与终态。

### 08:14 CST 训练进度

八个 retry3 任务均存活并继续更新，均处于 epoch 0：PushT/3072 的 Recorded-control 为 800/13,796 步、B-only 为 950/13,796；PushT/4096 分别为 800/13,796、950/13,796；Reacher/3072 分别为 800/13,034、950/13,034；Reacher/4096 分别为 600/13,034、700/13,034。GPU0–3 利用率均为 100%，每卡使用约 40.3 GiB、空余约 8.0 GiB。主机 load average 为 30.28/25.14/17.46（128 核），`MemAvailable` 约 382 GiB；79 GiB swap 约用了 0.4 GiB。

### 08:21 CST 训练进度

八个 retry3 进程均存活且持续更新：PushT/3072 Recorded-control 与 B-only 分别为 1,500/13,796、1,750/13,796 步；PushT/4096 分别为 1,500/13,796、1,750/13,796；Reacher/3072 分别为 1,500/13,034、1,800/13,034；Reacher/4096 分别为 1,350/13,034、1,550/13,034。均仍在 epoch 0。GPU0–3 保持 100% 利用率和约 8.0 GiB 空余显存。主机 load average 为 30.13/28.84/21.86（128 核），`MemAvailable` 约 383 GiB；79 GiB swap 约用了 0.4 GiB。按当前 1.8–2.1 steps/s 估算，完成 10 epochs 仍需较长训练时间。

### 08:29 CST 训练进度

八个 retry3 进程均存活并持续更新，仍在 epoch 0：PushT/3072 Recorded-control 与 B-only 分别为 2,350/13,796、2,750/13,796 步；PushT/4096 分别为 2,350/13,796、2,750/13,796；Reacher/3072 分别为 2,400/13,034、2,750/13,034；Reacher/4096 分别为 2,200/13,034、2,550/13,034。GPU0–3 利用率均为 100%，每卡使用约 40.3 GiB、空余约 8.0 GiB。主机 load average 为 29.93/29.73/25.07（128 核），`MemAvailable` 约 382 GiB；79 GiB swap 约用了 0.4 GiB。

### 08:35 CST 训练进度

八个 retry3 进程均存活并更新，仍在 epoch 0：PushT/3072 Recorded-control 与 B-only 分别为 3,050/13,796、3,500/13,796 步；PushT/4096 分别为 3,050/13,796、3,550/13,796；Reacher/3072 分别为 3,050/13,034、3,550/13,034；Reacher/4096 分别为 2,850/13,034、3,300/13,034。GPU0–3 均为 100% 利用率、每卡使用约 40.3 GiB 显存。主机 load average 为 29.64/29.80/26.64（128 核），`MemAvailable` 约 381 GiB；79 GiB swap 约用了 0.4 GiB。

### 08:42 CST 训练进度

八个 retry3 进程均存活并持续更新，仍在 epoch 0：PushT/3072 Recorded-control 与 B-only 分别为 3,800/13,796、4,400/13,796 步；PushT/4096 分别为 3,800/13,796、4,400/13,796；Reacher/3072 分别为 3,800/13,034、4,400/13,034；Reacher/4096 分别为 3,600/13,034、4,200/13,034。GPU0、2、3 利用率约 100%，GPU1 为 97%，每卡空余约 8.0 GiB。主机 load average 为 29.77/29.80/27.80（128 核），`MemAvailable` 约 379 GiB；79 GiB swap 约用了 0.4 GiB。

### 08:47 CST 训练进度

八个 retry3 进程均存活并持续更新，仍在 epoch 0：PushT/3072 Recorded-control 与 B-only 分别为 4,400/13,796、5,150/13,796 步；PushT/4096 分别为 4,450/13,796、5,150/13,796；Reacher/3072 分别为 4,450/13,034、5,150/13,034；Reacher/4096 分别为 4,250/13,034、4,950/13,034。GPU0–3 均为 100% 利用率，每卡余约 8.0 GiB。主机 load average 为 30.13/29.92/28.49（128 核），`MemAvailable` 约 376 GiB；79 GiB swap 约用了 0.4 GiB。

### 08:53 CST 训练进度

八个 retry3 进程均存活并持续更新，仍在 epoch 0：PushT/3072 Recorded-control 与 B-only 分别为 5,000/13,796、5,850/13,796 步；PushT/4096 分别为 5,050/13,796、5,850/13,796；Reacher/3072 分别为 5,100/13,034、5,850/13,034；Reacher/4096 分别为 4,850/13,034、5,650/13,034。GPU0–3 均为 100% 利用率，每卡余约 8.0 GiB。主机 load average 为 29.91/29.86/28.89（128 核），`MemAvailable` 约 375 GiB；79 GiB swap 约用了 0.4 GiB。

### 09:00 CST 训练进度

八个 retry3 进程均存活并持续更新，仍在 epoch 0：PushT/3072 Recorded-control 与 B-only 分别为 5,850/13,796、6,800/13,796 步；PushT/4096 分别为 5,900/13,796、6,800/13,796；Reacher/3072 分别为 5,900/13,034、6,850/13,034；Reacher/4096 分别为 5,700/13,034、6,600/13,034。GPU0–3 均为 100% 利用率，每卡空余约 8.0 GiB。主机 load average 为 30.89/30.32/29.44（128 核），`MemAvailable` 约 372 GiB；79 GiB swap 约用了 0.4 GiB。

### 09:07 CST 训练进度与 Recorded-clean 排队

retry3 八臂仍在 epoch 0 并持续更新：PushT/3072 Recorded-control 与 B-only 为 6,550/13,796、7,600/13,796 步；PushT/4096 为 6,600/13,796、7,600/13,796；Reacher/3072 为 6,600/13,034、7,650/13,034；Reacher/4096 为 6,400/13,034、7,400/13,034。GPU0–3 各使用约 40.3 GiB、利用率 100%；主机 `MemAvailable` 约 373 GiB，swap 约用了 0.4 GiB。

方案的 `recorded_clean` 不是 B-only：它使用 `train_mode=stage_ab`、保留 A/B 联合训练，但让 B 始终使用 recorded action 并采用 clean t=1/单位权重。本轮尚无该臂训练目录。已启动排队脚本 `outputs/round5/phase1_7/restart_logs/queue_recorded_clean_after_retry3.sh`（监控会话 9087）；它等待当前八臂终点权重、核验其 metadata 与 Joint 基线匹配，并在 GPU0–3 每卡空余至少 28,000 MiB 后启动 PushT/Reacher × seed 3072/4096 四个 Recorded-clean 十 epoch 训练。未满足任一门槛时不会启动。

09:08 再核验八个 retry3 Recorded-control/B-only 的初始 metadata 均与同 task/seed 的 Joint 基线完全一致（初始权重 SHA、episode split SHA、normalizer、train window count）；排队器中的终点前置门槛与该审计结果一致。

### 09:15 CST 主机资源与队列检查

用户已手动将 swap 扩至约 80 GiB；系统识别到 79 GiB，当前使用约 410 MiB。主机 `MemAvailable` 约 374 GiB，因此 swap 目前只是额外余量，没有缓解正在发生的内存压力。权限提升后的 GPU 检查确认八个 retry3 训练进程仍在运行、日志持续更新；GPU0–3 各由本实验两个进程共享，合计约占用 40.3 GiB 显存/卡，利用率 100%。

由于受限 shell 的 PID 视图不包含 GPU 上的训练进程，我已将 Recorded-clean 排队器的前置条件改为“八个 retry3 epoch-10 权重全部生成”，避免依赖 PID 可见性。其余 metadata 与 GPU 空闲显存门槛保留；已重启的排队器只会在八个最终权重齐全且 GPU0–3 每卡空闲至少 28 GiB 后启动四个 Recorded-clean 训练。

### 09:27 CST retry3 进度与终点评估排队

八个 Recorded-control/B-only retry3 进程仍正常推进，均在 epoch 0：PushT/3072 为 8,800/13,796 与 10,200/13,796 步；PushT/4096 为 8,850/13,796 与 10,250/13,796；Reacher/3072 为 8,900/13,034 与 10,250/13,034；Reacher/4096 为 8,650/13,034 与 10,050/13,034。GPU0–3 每卡约用 40.3 GiB、空余 8.0 GiB、利用率 95–100%；GPU4–7 检查时各有约 48.5 GiB 空闲。主机 `MemAvailable` 约 374 GiB、swap 已用约 410 MiB，load average 约 29.8（128 核）。

已新增终点评估排队脚本 `outputs/round5/phase1_7/restart_logs/queue_ablation_evaluations_after_retry3.sh`：等待八个 retry3 epoch-10 权重并复核匹配 metadata 后，运行 B-only 的 200-start P3 confirmation、预定 Joint-B/LeWM/B-only 主检验族校正、Recorded-control 的配对 P3，以及 B-only/Recorded-control 在 100-start dev 固定池上的共同动作排序评估。每轮最多四项并行；显存预检要求每张所选卡至少空余 6 GiB。北京时间 10:00–23:00 只有 GPU0–3 四卡各至少空余 34 GiB 时才在其上评估（给 28 GiB 训练启动门槛留出余量），否则将低显存评估放到 GPU4–7；夜间使用 GPU4–7。评估队列监控会话 27323 已启动，首轮权重门槛为 0/8，尚未实际启动评估进程。

存储核查：训练数据卷当前约余 60.6 GiB，根卷约余 74.9 GiB；本轮单模型 epoch 权重约 42–72 MB，现阶段空间可覆盖已排队训练和评估输出，后续继续监控磁盘余量。

### 09:39 CST 训练与评估队列复查

权限提升后的 NVML 复查再次确认八个 retry3 训练进程存活且 GPU0–3 利用率 96–100%。日志均更新到 09:39：PushT/3072 Recorded-control/B-only 为 10,100/13,796、11,700/13,796 步；PushT/4096 为 10,150/13,796、11,700/13,796；Reacher/3072 为 10,200/13,034、11,750/13,034；Reacher/4096 为 9,950/13,034、11,500/13,034，均仍在 epoch 0。八个终点权重为 0/8；Recorded-clean 队列与消融评估队列均处于等待状态，未额外占用 GPU。GPU4–7 各空闲约 48.5 GiB；主机 `MemAvailable` 约 371 GiB、swap 使用约 410 MiB，load average 约 29.8（128 核）。数据卷仍余约 60.6 GiB。

随后新增并启动 `outputs/round5/phase1_7/restart_logs/queue_recorded_clean_evaluations.sh`（监控会话 25582），等待四个 Recorded-clean epoch-10 权重齐备并复核 metadata，再运行同 Joint-A 的 200-start P3 与 100-start dev 固定池评估。初始门槛为 0/4；该队列尚未占用 GPU。

### 09:54 CST epoch-2 消融开发诊断

为利用现有且严格匹配的中期 checkpoint，已先在 `docs/plan/round5_phase1_7_plan.md` 增补一条开发期诊断规则：仅当完整 epoch-2 权重存在且初始权重 SHA、split、normalizer、窗口数均与 Joint 匹配时，才可评估早期 Recorded-control/B-only；不进入 epoch-10 主终点或 confirmation。上述八个旧中断目录的字段均匹配，epoch-2 actor checkpoint 也齐全。

随后启动 `outputs/round5/phase1_7/restart_logs/run_epoch2_ablation_dev_diagnostics.sh`，监控会话 11715。首波在 GPU4–7 并行运行 PushT/Reacher × seed 3072/4096 的 B-only P3 dev100；30 秒启动核验中四项均存活，每项约占 0.5–1.4 GiB 显存，PushT/3072 B-only 已完成，其余三项继续运行。后续波次将依次运行 Recorded-control P3 和两臂的 dev100 固定池排序。GPU0–3 的主训练也有进展：PushT/3072 与 /4096 B-only 接近 epoch 0 末尾，Reacher 两个 B-only 已进入 epoch 1；八个 epoch-10 权重仍为 0/8。主机 `MemAvailable` 约 362 GiB，79 GiB swap 仅用约 410 MiB；数据卷约余 60.6 GiB。

### 10:50 CST epoch-2 同 actor P3 复核与排序审计

为消除 Reacher/seed3072 的 actor provenance 差异，并与 B-only P3 使用完全相同的 Joint actor，PushT/Reacher × seed3072/4096 的 Joint、LeWM P3 各重跑 100 个 dev 起点。首轮临时输出路径没有区分训练 seed，两个 PushT seed 落入同一评估叶目录，触发拒绝覆盖；该轮共享目录结果全部排除。第二轮按训练 seed 隔离输出根目录，八项均为 `status=ok`，并逐项核对 actor 路径与 task/seed。结果和 Holm 分析见 [`matched primary-family analysis`](../../../outputs/round5/phase1_7/epoch2_retry_ablation_matched_primary_family_v2_analysis.json)；锁定输出位于 `epoch2_retry_ablation_matched_p3_baseline_s3072/` 和 `epoch2_retry_ablation_matched_p3_baseline_s4096/`。

下表的差值为比较臂减 Joint，CI 为按起点配对 bootstrap 95% 区间。每个 task/seed 有 LeWM、B-only 两项 P3 比较；右列 Holm p 在该 task/seed 的两项比较内调整。所有结果是 epoch-2 开发诊断，复用 dev cohort，不是 confirmation。

| Task / seed | Joint 成功 | LeWM 成功；差值 pp [95% CI]；Holm p | B-only 成功；差值 pp [95% CI]；Holm p |
|---|---:|---|---|
| PushT / 3072 | 15/100 | 12/100；−3 [−11, +5]；0.848 | 11/100；−4 [−11, +3]；0.848 |
| PushT / 4096 | 16/100 | 10/100；−6 [−13, +1]；0.359 | 18/100；+2 [−5, +9]；0.774 |
| Reacher / 3072 | 9/100 | 7/100；−2 [−10, +5]；0.791 | 15/100；+6 [−2, +14]；0.476 |
| Reacher / 4096 | 4/100 | 10/100；+6 [−1, +13]；0.146 | 12/100；+8 [+2, +14]；0.043 |

Reacher/4096 的 B-only 对 Joint 原始 McNemar p=0.0215，但对所有 8 项作 Holm 敏感性校正后为 0.1719；因此不作为跨 task/seed 的显著证据。P3 方向依 task/seed 变化，不能证明 B-only 或 LeWM 一贯优于 Joint。

PushT epoch-2 固定池方面，旧 LeWM artifact 与当前 B-only artifact 的 100×64 物理动作、候选噪声和 Joint cost 矩阵逐项完全相同；两份 LeWM checkpoint 的 SHA 也与本轮 P3 使用的快照权重相同。以当前 B-only 固定池回放的 6,400 条物理分支作为共同 outcomes，重算 Joint-B 与 LeWM 排序得到：

| PushT seed | Pairwise accuracy（Joint-B / LeWM）及差值 [95% CI] | Spearman（Joint-B / LeWM）及差值 [95% CI] | Top-1 成功（Joint-B / LeWM） | Regret（Joint-B / LeWM）；Joint-B−LeWM [95% CI] |
|---:|---|---|---:|---|
| 3072 | 0.552 / 0.466；+0.086 [+0.059, +0.115] | 0.147 / −0.099；+0.246 [+0.168, +0.326] | 0% / 1%（McNemar p=1.000） | 43.87 / 56.65；−12.86 [−20.18, −5.44] |
| 4096 | 0.548 / 0.484；+0.064 [+0.031, +0.097] | 0.133 / −0.047；+0.180 [+0.086, +0.274] | 2% / 0%（McNemar p=0.500） | 46.39 / 49.30；−2.76 [−9.88, +4.69] |

这些是未对多指标和 seed 扫描校正的中期 dev 排序结果；top-1 成功仍只有 0–2%。B-only 相对 LeWM 的 pairwise/Spearman 点估计也偏 B-only，但不是 Joint-B 的主比较。固定池重评分队列正在等待另外两个 Reacher B-only 物理回放完成，然后会在同一保存动作池上重算 Joint-B、B-only、LeWM 三组成本并使用同一份 outcomes。

重复物理回放审计发现：旧 LeWM 与当前 B-only 对相同 cohort、相同动作池的两次分支回放并未产生完全相同的环境记录。PushT/3072 的 6,400 条中，`success_by_25` 有 24 条不同、`valid_at_25` 有 19 条不同；PushT/4096 分别有 182 条和 171 条不同。故上述成对排序只使用当前 B-only 回放作为所有评分器共享的物理 outcomes，没有拼接两次回放。差异根因尚未定位；当前区间只对起点重采样，没有覆盖回放间变异，这是解释这些 dev 指标时的限制。

截至 10:50，八个 retry3 Recorded-control/B-only 均在 epoch 1：PushT/3072 为 3,850/13,796 与 6,700/13,796 步，PushT/4096 为 4,050/13,796 与 6,800/13,796；Reacher/3072 为 4,950/13,034 与 7,450/13,034；Reacher/4096 为 4,650/13,034 与 7,300/13,034。新终点权重 0/8。GPU0–3 仍约 40.3 GiB/卡、98–100%；GPU4–7 仅承载低显存评测。主机 `MemAvailable` 约 344 GiB，79 GiB swap 使用约 407 MiB；根卷余 75 GiB，数据卷余 59 GiB。

### 11:03 CST 训练与固定池进度

用户已将 swap 手动增加到约 80 GiB；`swapon` 确认约 80 GiB 已启用，当前使用约 407 MiB。主机 `MemAvailable` 约 341 GiB，说明目前没有明显内存压力，swap 主要提供峰值缓冲。

八个 retry3 Recorded-control/B-only 仍在 epoch 1，日志持续更新：PushT/3072 为 5,250/13,796 与 8,350/13,796 步；PushT/4096 为 5,450/13,796 与 8,500/13,796；Reacher/3072 为 6,400/13,034 与 9,150/13,034；Reacher/4096 为 6,100/13,034 与 9,000/13,034。epoch-10 终点权重仍为 0/8。GPU0–3 各使用约 40.3 GiB 显存、利用率 100%，空余约 8.0 GiB。

epoch-2 B-only 固定池评估的 PushT 两个 seed 已完成；Reacher/3072 与 /4096 均推进到 56/64 个候选分支。配对重评分队列仍等待四个固定池汇总文件齐全，当前为 2/4；此波仍是开发诊断。GPU4–7 上只运行低显存评估任务。

### 11:10 CST epoch-2 固定池三评分器同回放比较

Reacher 两个 B-only 固定池均完成 64/64 候选分支后，四个 task/seed 的 Joint-B、B-only、LeWM 重评分全部完成。每项使用同一批 100 个 dev 状态、64 个候选动作和同一份 6,400 分支物理 outcomes；候选动作与噪声逐项相同，LeWM 不重放环境。详细 provenance 与分数数组见 [`epoch-2 fixed-pool paired analyses`](../../../outputs/round5/phase1_7/epoch2_retry_ablation_fixed_pool_pair/)。

下表给出比较臂减 LeWM 的 pairwise 排序准确率差（百分点）和 Spearman 差；括号为按状态配对 bootstrap 的 95% CI。CI 未对四个 task/seed、两种评分器的八项比较作多重校正，也没有覆盖重放间变异。Top-1 列为 Joint-B / B-only / LeWM 的成功率。

| Task / seed | Joint-B vs LeWM：pairwise Δ pp；Spearman Δ | B-only vs LeWM：pairwise Δ pp；Spearman Δ | Top-1 成功率 J-B / B / LeWM |
|---|---|---|---|
| PushT / 3072 | +8.6 [+5.9, +11.5]；+0.246 [+0.168, +0.326] | +6.5 [+4.0, +9.0]；+0.189 [+0.116, +0.259] | 0% / 0% / 1% |
| PushT / 4096 | +6.4 [+3.1, +9.7]；+0.180 [+0.086, +0.274] | +4.1 [+1.2, +7.0]；+0.117 [+0.033, +0.200] | 2% / 0% / 0% |
| Reacher / 3072 | +8.5 [+6.2, +10.7]；+0.241 [+0.177, +0.302] | +14.1 [+12.1, +15.9]；+0.391 [+0.337, +0.442] | 5% / 10% / 5% |
| Reacher / 4096 | +2.8 [−1.1, +6.6]；+0.080 [−0.028, +0.184] | +12.8 [+8.5, +17.0]；+0.348 [+0.233, +0.461] | 3% / 7% / 8% |

这组早期 dev 排序诊断显示：Joint-B 的 pairwise 排序准确率在四个 strata 均高于 LeWM，三个 strata 的状态 bootstrap 区间高于零；B-only 的 pairwise 与 Spearman 差在四个 strata 的区间均高于零。但它没有形成稳定的 top-1 成功率优势，成功率仅 0–10%，且最佳评分器随 strata 改变。因此它支持“早期 checkpoint 的排序相关性可能强于单独 LeWM”，不足以声称 B 已改善实际选出的动作。该方向与 epoch-10 多 seed 固定池结果并不一致；主结论仍须等终点 checkpoint 与预注册 confirmation，不能用这组中期 dev 结果替代。

### 11:23 CST 训练与 Recorded-control 固定池进度

八个 retry3 训练日志仍在更新，均处于 epoch 1：PushT/3072 Recorded-control/B-only 为 7,400/13,796、11,000/13,796 步；PushT/4096 为 7,750/13,796、11,100/13,796；Reacher/3072 为 8,750/13,034、11,650/13,034；Reacher/4096 为 8,400/13,034、11,550/13,034。epoch-10 权重仍为 0/8。GPU0–3 每卡约用 40.3 GiB、利用率 100%；GPU4–7 继续承载低显存评估。主机 `MemAvailable` 约 345 GiB，79 GiB swap 仍只用约 407 MiB。

epoch-2 Recorded-control 固定池评估仍在运行：PushT/3072 与 /4096 均完成 24/64 个候选分支，Reacher 两个 seed 各完成 8/64；尚未生成汇总文件。评估队列和 Recorded-clean 队列继续等待各自训练终点 checkpoint，当前未启动额外终点任务。

### 11:29 CST 训练与 Recorded-control 固定池进度

主训练八臂继续推进且日志无错误，仍处于 epoch 1：PushT/3072 Recorded-control/B-only 为 8,050/13,796、11,750/13,796 步；PushT/4096 为 8,400/13,796、11,900/13,796；Reacher/3072 为 9,450/13,034、12,450/13,034；Reacher/4096 为 9,100/13,034、12,300/13,034。epoch-10 权重仍为 0/8。GPU0–3 仍各使用约 40.3 GiB、利用率 100%。

Recorded-control 固定池已有进展：PushT/3072 与 /4096 各 40/64 个分支，Reacher/3072 与 /4096 各 16/64；未发现错误或汇总文件。GPU4–7 仅承担低显存评估。主机 `MemAvailable` 约 339 GiB、swap 使用约 407 MiB；根卷余 75 GiB，数据卷余 59 GiB。

### 11:35 CST Recorded-control 固定池与主训练复查

Recorded-control 固定池继续推进：PushT 两个 seed 均为 48/64 分支，Reacher 两个 seed 均为 16/64；没有错误。主训练仍正常，epoch-10 权重 0/8：PushT/3072 Recorded-control/B-only 为 8,650/13,796、12,500/13,796 步；PushT/4096 为 9,000/13,796、12,600/13,796；Reacher/3072 为 10,100/13,034 与 B-only epoch 2 的 50/13,034；Reacher/4096 为 9,700/13,034、13,000/13,034。GPU0–3 利用率约 88–100%，各约用 40.3 GiB；GPU4–7 上为低显存诊断。`MemAvailable` 约 339 GiB，79 GiB swap 用约 407 MiB。

### 11:41 CST Recorded-control 固定池阶段完成情况

PushT/3072 与 /4096 的 Recorded-control 固定池均完成 64/64 分支并写出评估结果；Reacher 两个 seed 各完成 24/64。主训练继续正常：PushT/3072 Recorded-control/B-only 为 9,250/13,796、13,250/13,796 步；PushT/4096 为 9,650/13,796、13,400/13,796；Reacher/3072 Recorded-control 为 10,750/13,034，B-only 已进入 epoch 2（800/13,034）；Reacher/4096 为 10,400/13,034，B-only 已进入 epoch 2（650/13,034）。epoch-10 权重仍为 0/8。GPU0–3 基本满载；79 GiB swap 使用约 407 MiB，`MemAvailable` 约 345 GiB。

### 11:48 CST B-only epoch-2 与 Recorded-control 固定池

四个 B-only retry3 任务均已进入 epoch 2，Recorded-control 四项仍在 epoch 1；所有训练日志持续更新，epoch-10 权重 0/8。当前步数：PushT/3072 RC/B-only 10,000/13,796 与 300/13,796；PushT/4096 10,450/13,796 与 400/13,796；Reacher/3072 11,500/13,034 与 1,700/13,034；Reacher/4096 11,150/13,034 与 1,500/13,034。

Recorded-control 固定池的 PushT 两项已完成并有 summary；Reacher/3072 与 /4096 各完成 32/64 分支。GPU0–3 满载；GPU4–7 空余显存充足，仅有低显存评估占用。主机 `MemAvailable` 约 342 GiB，79 GiB swap 使用约 407 MiB。

### 11:53 CST 主训练与 Recorded-control 固定池

八个 retry3 训练继续更新，epoch-10 权重仍为 0/8。B-only 四项处于 epoch 2：PushT/3072、/4096 分别为 1,050/13,796、1,150/13,796 步；Reacher/3072、/4096 为 2,450/13,034、2,300/13,034。Recorded-control 四项仍处于 epoch 1：PushT/3072、/4096 为 10,650/13,796、11,100/13,796；Reacher/3072、/4096 为 12,150/13,034、11,800/13,034。

Recorded-control 固定池 PushT 两项已完成；Reacher 两个 seed 均到 40/64。GPU0–3 仍满载，GPU4–7 仅运行低显存评估。`MemAvailable` 约 344 GiB，swap 使用约 407 MiB。

### 11:59 CST Recorded-control 固定池接近完成

Recorded-control 固定池 Reacher/3072 与 /4096 均到 48/64 个分支，PushT 两个 seed 已完成。主训练仍无错误，epoch-10 权重 0/8：Recorded-control 的 PushT/3072、/4096、Reacher/3072、/4096 分别为 11,300/13,796、11,700/13,796、12,750/13,034、12,450/13,034；B-only 已在 epoch 2，分别为 1,750/13,796、1,900/13,796、3,150/13,034、3,000/13,034。GPU0–3 满载，`MemAvailable` 约 343 GiB，79 GiB swap 使用约 407 MiB。

### 12:05 CST B-only epoch-2 与固定池进度

B-only 四臂均在 epoch 2，Recorded-control 三臂仍在 epoch 1、Reacher/3072 已进入 epoch 2。当前步数：PushT/3072 RC/B-only 为 11,900/13,796 与 2,500/13,796；PushT/4096 为 12,350/13,796 与 2,600/13,796；Reacher/3072 为 RC epoch 2 的 300/13,034、B-only 3,950/13,034；Reacher/4096 为 13,000/13,034、3,750/13,034。日志正常更新，终点权重 0/8。

Recorded-control 固定池 PushT 两项已完成；Reacher 两项均为 48/64，尚未出 summary。GPU0–3 满载、GPU4–7 仅评估；主机 `MemAvailable` 约 342 GiB，79 GiB swap 使用约 407 MiB。

### 12:10 CST Recorded-control 固定池收尾

Reacher 两个 Recorded-control 固定池均到 56/64 分支，PushT 两项已完成；尚未生成 Reacher summary。主训练无错误：B-only 四臂都处于 epoch 2（PushT/3072、/4096 为 3,300/13,796、3,400/13,796；Reacher/3072、/4096 为 4,750/13,034、4,550/13,034）。Recorded-control 的 Reacher 两臂也已进入 epoch 2（1,000/13,034、650/13,034），PushT 两臂仍在 epoch 1（12,600/13,796、13,050/13,796）。终点权重 0/8；GPU0–3 满载，GPU4–7 仅承担低显存评估；`MemAvailable` 约 340 GiB，swap 使用约 407 MiB。

### 12:20 CST epoch-2 Recorded-control 固定池结果

四个 Recorded-control 固定池均已完成。其候选动作、物理动作、噪声和 Joint cost 与对应 B-only 固定池逐项相同；但两轮环境分支记录并不完全相同，所以这里只在每轮自己的 outcomes 内配对比较 Recorded-control score 与 Joint score，不跨两轮拼接 outcomes。CI 为 100 个 dev 状态配对 bootstrap，未校正多重比较。

| Task / seed | Pairwise Δ pp（RC−Joint） | Spearman Δ | Top-1 成功率 Δ pp | Regret Δ |
|---|---:|---:|---:|---:|
| PushT / 3072 | −1.9 [−4.3, +0.4] | −0.058 [−0.122, +0.008] | 0 [0, 0] | +3.76 [−2.62, +10.21] |
| PushT / 4096 | −0.8 [−3.0, +1.4] | −0.023 [−0.085, +0.040] | 0 [−3, +3] | +2.10 [−3.73, +7.95] |
| Reacher / 3072 | +9.8 [+7.9, +11.6] | +0.239 [+0.188, +0.291] | +4 [−1, +10] | −0.184 [−0.287, −0.078] |
| Reacher / 4096 | −0.4 [−2.7, +2.0] | −0.013 [−0.078, +0.055] | +4 [−1, +10] | −0.012 [−0.089, +0.065] |

Recorded-control 的排序提升集中在 Reacher/3072；PushT 两个 seed 与 Reacher/4096 均未见清楚优势，top-1 成功率差区间都包含零。这同样只是 epoch-2 dev 诊断，不能替代终点结论。

截至 12:20，八臂终点权重仍为 0/8。B-only 四臂均在 epoch 2：PushT/3072、/4096 为 4,400/13,796、4,550/13,796；Reacher/3072、/4096 为 5,850/13,034、5,650/13,034。Recorded-control 的 PushT/3072 为 epoch 1 的 13,550/13,796，另三臂在 epoch 2（PushT/4096 150/13,796；Reacher/3072、/4096 为 1,950/13,034、1,650/13,034）。GPU0–3 满载；低显存诊断已完成，GPU4–7 当前空闲。主机 `MemAvailable` 约 367 GiB，79 GiB swap 使用约 407 MiB。

### 12:25 CST Joint-B/B-only 对 LeWM 的双回放敏感性

为检验早期排序差异是否依赖某一次 PushT 环境回放，我将完全相同的 Joint-B、B-only、LeWM 分数分别与 B-only 和 Recorded-control 两次已保存回放配对。候选动作、物理动作、噪声、Joint cost 在两轮均逐项相同；回放 outcome 分开分析，不合并。完整哈希和指标见 [`two-replay fixed-pool analysis`](../../../outputs/round5/phase1_7/epoch2_retry_ablation_multi_replay/fixed_pool_multi_replay_analysis.json)。

| Task / seed | Joint-B−LeWM pairwise Δ pp：B-only / RC replay | B-only−LeWM pairwise Δ pp：B-only / RC replay |
|---|---|---|
| PushT / 3072 | +8.6 [+5.9, +11.5] / +9.2 [+6.5, +11.9] | +6.5 [+4.0, +9.0] / +6.5 [+3.9, +8.9] |
| PushT / 4096 | +6.4 [+3.1, +9.7] / +6.1 [+2.8, +9.3] | +4.1 [+1.2, +7.0] / +3.8 [+0.7, +6.8] |
| Reacher / 3072 | +8.5 [+6.2, +10.7] / +8.5 [+6.2, +10.7] | +14.1 [+12.1, +15.9] / +14.1 [+12.1, +15.9] |
| Reacher / 4096 | +2.8 [−1.1, +6.6] / +2.8 [−1.1, +6.6] | +12.8 [+8.5, +17.0] / +12.8 [+8.5, +17.0] |

PushT 的两个回放里，Joint-B 对 LeWM 的 pairwise 提升方向和量级接近；B-only 对 LeWM 的区间在四个 strata 的两个回放中均高于零。Reacher/4096 的 Joint-B 差异仍不确定。top-1 成功率差在各 strata 仍混合且区间较宽，所以更稳定的是候选两两排序相关性，不是实际选出的首个动作成功率。上述区间只按 100 个 dev 起点重采样，未作多重校正，也未覆盖环境回放总体的不确定性。

两次 PushT 回放确有环境差异：PushT/3072 的 6,400 分支中，成功标记有 85 条不同、valid 标记有 79 条不同，6,316 条可比较的终点距离不同，平均绝对差约 21.61；PushT/4096 分别为 91、78、6,323 条，平均绝对距离差约 19.59。两个 Reacher seed 的这两轮结果完全相同。因此将这组结果视为 early-dev 的回放敏感性复核，不将两次回放当作独立抽样来合并显著性。

### 12:26 CST 八臂均进入 epoch 2

八个 retry3 训练任务均已进入 epoch 2，日志正常更新，epoch-10 权重仍为 0/8。Recorded-control/B-only 当前步数：PushT/3072 为 550/13,796、5,350/13,796；PushT/4096 为 1,000/13,796、5,450/13,796；Reacher/3072 为 2,750/13,034、6,800/13,034；Reacher/4096 为 2,450/13,034、6,600/13,034。GPU0–3 仍约 40.3 GiB/卡、95–100% 利用率；GPU4–7 在 epoch-2 dev 诊断结束后空闲。主机 `MemAvailable` 约 366 GiB，79 GiB swap 使用约 407 MiB。

### 12:31 CST epoch-2 训练进度与存储

八个主训练均持续推进且日志无错误：Recorded-control 的 PushT/3072、/4096 为 1,150/13,796、1,600/13,796，Reacher/3072、/4096 为 3,400/13,034、3,050/13,034；B-only 对应步数为 6,050/13,796、6,200/13,796、7,550/13,034、7,350/13,034。终点权重仍为 0/8。GPU0–3 满载、GPU4–7 空闲；`MemAvailable` 约 365 GiB，79 GiB swap 使用约 407 MiB。根卷余 75 GiB，数据卷余 58 GiB。

### 12:36 CST epoch-2 训练继续推进

八个 retry3 训练日志继续更新，无报错，epoch-10 权重 0/8。Recorded-control 当前步数：PushT/3072、/4096 为 1,750/13,796、2,250/13,796；Reacher/3072、/4096 为 4,000/13,034、3,700/13,034。B-only 对应为 6,800/13,796、6,900/13,796、8,250/13,034、8,050/13,034。GPU0–3 约 40.3 GiB/卡且 100% 利用率；GPU4–7 空闲。主机 `MemAvailable` 约 366 GiB，79 GiB swap 使用约 407 MiB；数据卷余约 58 GiB。

### 12:41 CST epoch-2 训练进度

八个主训练均正常更新，终点权重仍 0/8。Recorded-control 当前为 PushT/3072、/4096 的 2,400/13,796、2,850/13,796 步，以及 Reacher/3072、/4096 的 4,650/13,034、4,300/13,034 步；B-only 分别为 7,500/13,796、7,600/13,796、8,950/13,034、8,750/13,034。GPU0–3 满载，GPU4–7 空闲。`MemAvailable` 约 369 GiB，79 GiB swap 使用约 407 MiB；数据卷余 58 GiB。

### 12:51 CST 主训练进度

八个 retry3 训练都在 epoch 2 并持续更新，epoch-10 权重 0/8。Recorded-control 的 PushT/3072、/4096、Reacher/3072、/4096 当前为 3,600/13,796、4,050/13,796、5,850/13,034、5,500/13,034 步；B-only 对应为 8,900/13,796、9,000/13,796、10,400/13,034、10,200/13,034。GPU0–3 满载，GPU4–7 空闲。主机 `MemAvailable` 约 368 GiB，79 GiB swap 使用约 407 MiB；数据卷余 58 GiB。

### 12:56 CST epoch-2 训练进度

八个训练继续稳定推进、尚无 epoch-10 权重。Recorded-control 当前步数为 PushT/3072、/4096 的 4,250/13,796、4,700/13,796 和 Reacher/3072、/4096 的 6,500/13,034、6,150/13,034；B-only 为 9,650/13,796、9,750/13,796、11,100/13,034、10,900/13,034。GPU0–3 满载，GPU4–7 空闲；`MemAvailable` 约 367 GiB，79 GiB swap 使用约 407 MiB，数据卷余 58 GiB。

### 13:01 CST epoch-2 训练进度

八个训练仍在 epoch 2、日志持续更新，epoch-10 权重 0/8。Recorded-control 的 PushT/3072、/4096 为 4,850/13,796、5,350/13,796，Reacher/3072、/4096 为 7,100/13,034、6,800/13,034；B-only 对应为 10,350/13,796、10,500/13,796、11,850/13,034、11,650/13,034。GPU0–3 利用率 96–100%，约 40.3 GiB/卡；GPU4–7 空闲。`MemAvailable` 约 367 GiB，79 GiB swap 使用约 407 MiB，数据卷余 58 GiB。

### 13:06 CST B-only epoch-2 进度

B-only 两个 Reacher 训练接近 epoch 2 末尾（12,600/13,034、12,400/13,034 步），B-only PushT 为 11,100/13,796、11,200/13,796。Recorded-control 四项仍在 epoch 2：PushT 为 5,500/13,796、5,950/13,796，Reacher 为 7,750/13,034、7,400/13,034。所有任务日志正常，epoch-10 权重 0/8；GPU0–3 满载，GPU4–7 空闲；主机 `MemAvailable` 约 366 GiB，79 GiB swap 使用约 407 MiB，数据卷余 58 GiB。

## Retry3 epoch-10 终点与本轮暂停（2026-10-02）

### 收尾核对

PushT/Reacher × seed 3072/4096 的八个 retry3 训练全部完成，八份 epoch-10 权重均存在；Recorded-control 与 B-only 的初始权重、数据划分、normalizer 和训练窗口匹配审计通过。八个 confirmation P3 结果（B-only 与 Recorded-control 各四项）均为 `status=ok`、每项 200 个起点；八个 dev 固定池摘要均为 100 个起点 × 64 个候选。预注册 P3 分析和固定池配对分析均已生成。最终检查没有发现本轮训练或评测进程仍在运行，也没有错误退出记录。

按用户要求，Recorded-clean 后续训练启动队列及其依赖评测等待器已停止，四个 Recorded-clean 模型没有启动。本轮结果文件和汇总分析：[`epoch10_retry3_primary_family_analysis.json`](../../../outputs/round5/phase1_7/epoch10_retry3_primary_family_analysis.json)、[`epoch10_retry3_fixed_pool_dev_pair_analysis.json`](../../../outputs/round5/phase1_7/epoch10_retry3_fixed_pool_dev_pair_analysis.json)。

### Confirmation P3：Joint-B、独立 LeWM 与 B-only

下表差值为比较臂减 Joint-B 的 25 步成功率；区间按 200 个配对起点 bootstrap。每个任务×训练 seed 的检验族含 LeWM 与 B-only 两个比较；最后一列的第二个 p 值是八项比较的敏感性 Holm 校正。

| Task / seed | Joint-B | LeWM：成功数；Δ pp [95% CI]；Holm p（2 项 / 8 项） | B-only：成功数；Δ pp [95% CI]；Holm p（2 项 / 8 项） |
|---|---:|---|---|
| PushT / 3072 | 181/200 | 173/200；−4.0 [−8.5, 0.0]；0.193 / 0.674 | 182/200；+0.5 [−3.0, +4.0]；1.000 / 1.000 |
| PushT / 4096 | 180/200 | 171/200；−4.5 [−8.0, −1.0]；0.045 / 0.180 | 184/200；+2.0 [−1.5, +5.5]；0.424 / 1.000 |
| Reacher / 3072 | 42/200 | 50/200；+4.0 [−4.0, +12.0]；0.763 / 1.000 | 42/200；0.0 [−7.0, +7.0]；1.000 / 1.000 |
| Reacher / 4096 | 52/200 | 44/200；−4.0 [−11.5, +3.5]；0.716 / 1.000 | 45/200；−3.5 [−10.5, +3.5]；0.716 / 1.000 |

PushT/4096 是唯一在任务×seed 两项 Holm 家族内显著的 LeWM 对比，方向有利于 Joint-B；扩展到八项比较的敏感性校正后 p=0.180。其余 LeWM 结果方向混合或区间较宽。四个 B-only 对比的区间均包含零，没有显示独立 B scorer 在闭环成功率上稳定优于 Joint-B。

### Dev 固定池：B-only 与 Recorded-control scorer 相对 Joint scorer

同一 Joint actor、同一 100 个 dev 起点和每状态 64 个候选下，B-only 与 Recorded-control 两次评分都复用了逐项完全相同的候选、物理动作、噪声和 Joint cost。下表报告 scorer−Joint 的 pairwise sign accuracy 差和 Spearman 差；区间按起点配对 bootstrap，未作多重校正。完整 top-1、regret 和逐状态指标见上述配对分析文件。

| Task / seed | B-only pairwise Δ pp [95% CI] | B-only Spearman Δ [95% CI] | Recorded-control pairwise Δ pp [95% CI] | Recorded-control Spearman Δ [95% CI] |
|---|---:|---:|---:|---:|
| PushT / 3072 | +2.3 [−1.4, +6.3] | +0.053 [−0.038, +0.147] | +1.9 [−0.1, +4.0] | +0.053 [−0.006, +0.115] |
| PushT / 4096 | +1.7 [−1.0, +4.3] | +0.051 [−0.018, +0.120] | +1.6 [−0.2, +3.5] | +0.048 [−0.004, +0.101] |
| Reacher / 3072 | +0.8 [−0.8, +2.3] | +0.021 [−0.025, +0.067] | +0.7 [−0.8, +2.2] | +0.023 [−0.020, +0.067] |
| Reacher / 4096 | −1.3 [−2.7, +0.2] | −0.034 [−0.076, +0.009] | −0.0 [−1.2, +1.1] | +0.001 [−0.034, +0.034] |

所有连续排序差的区间都包含零。B-only 的 top-1 成功率差依次为 −3、−2、−6、+5 pp，区间均包含零；Recorded-control 依次为 −5、+3、−4、+4 pp，其中 PushT/3072 的未校正 McNemar p=0.0625，其余方向混合。排序指标的正向点估计没有一致传递到实际首选动作的成功率。

两臂使用相同候选池，但 PushT 的两次物理分支回放并不相同：seed 3072 的 6,400 个候选结果中 6,324 行不同，seed 4096 中 6,329 行不同；Reacher 两个 seed 的 6,400 行结果均完全一致。因此本节对每个 scorer 只在其自身回放内与 Joint 配对，不把 PushT 的两次回放拼接成一个统计样本。

### 对论文主张的影响

完整 epoch-10 Joint-B/LeWM 固定池结果已见上文：PushT 两个 seed 的 pairwise sign accuracy 分别为 0.514/0.515 和 0.500/0.497，Reacher 为 0.508/0.499 和 0.502/0.499；绝对排序接近机会水平，Spearman 接近零，差值区间没有建立稳定优势。P3 中 PushT/4096 有一项支持 Joint-B 的任务×seed 结果，但八项敏感性校正后不显著；Reacher 方向相反且不确定。epoch-2 dev 的较大排序差没有在 epoch-10 固定池复现。

因此本轮证据**不支持声称 B 的动作排序普遍优于单独 LeWM**。更准确的表述是：当前两个任务、两个训练 seed 下没有建立稳定的排序优势；PushT/4096 出现一个局部的闭环收益信号，但没有跨任务复现。若论文保留 B 的贡献，应限定为任务/设置相关的闭环表现或其他已独立验证的系统贡献，不把它概括成普遍更准确的动作排序器。

本轮训练和已排定评测均已结束。目标按用户要求暂停；没有发起 Recorded-clean 或其他新训练。
