CoWM: Coupled Latent World Model for Fast Action Selection and Refinement

## Introduction
1 Recent JEPA-based world model怎么怎么样 （讲一下JEPA，确定研究范式和JEPA的价值）
2 Despite their efficiency （从JEPA的优势转移到缺点）
3 To this end, we propose CoWM （提出核心思想，解释设计依据）
4 In practice, CoWM comprises （根据核心思想来讲述技术机制）
5 During deployment, （说明模型的特性）
6 Our contributions are summarized as follows:

1 研究背景 ———— 轻量潜空间预测仍需要高效的动作决策
视觉目标条件决策需要预测动作后果，选择并调整action =》JEPA，潜空间高效规划，避免像素重建成本 =》闭环控制要求整个决策过程在有限时间内完成。

从视觉观测条件下的动作决策切入：智能体需要预测动作带来的后果，才能选择并调整行为。随后引入 JEPA-based latent world models：通过在紧凑表征空间预测未来状态，避免像素重建的成本，为规划提供有效的预测模型。最后，潜空间预测的计算效率为视觉规划提供了基础，但实际部署还要求模型能够在有限决策时间内选出有效动作

引出本文关注的两种用途：Action Selection and Refinement，评估动作和优化动作；引出有限决策时间问题。

2 核心问题 ———— 如何在有限时间内更有效地采样、评估和修正动作
1. 搜索式规划需要大量候选评估和多轮更新，产生决策开销。
2. 学习动作生成可以提供更有针对性的候选，减少在线搜索的负担。
3. 有限决策预算下，需要从少量候选中选出好动作，或者通过少量更新改善动作
4. 由于selection and refinement操作作用于候选action，因此候选生成与后果预测之间的配合之间关系到预算利用效率
5. 能否通过联合学习改善这种配合？
凝练的科学问题:
> 在有限决策预算下，联合学习动作生成与动作条件动力学，能否提高模型对候选动作的评估（rerank）和修正（Guidance，PO/GF）能力，从而减少对大规模搜索的依赖？

初稿：
搜索式规划通常需要反复采样动作、预测后果并更新候选分布，因此，即使采用轻量的潜空间世界模型，多轮搜索仍可能带来显著的决策开销。学习目标条件的动作生成器能够提供更有针对性的候选，从而减少在线搜索的负担。然而，在有限决策时间内，可评估的候选数量和可执行的修正步数均受到限制，系统需要有效判断生成候选的优劣，并利用少量更新改善其任务效果。**由于待评估和修正的动作由生成器提供，动力学模型对这些动作后果的预测，以及由此得到的排序和修正方向，成为连接候选生成与最终决策的关键。** 这引出一个问题：在有限决策预算下，联合学习动作生成与动作条件动力学，能否提高候选动作的评估与修正能力，从而减少对大规模搜索的依赖？


3 核心思想 ———— 提出耦合模型，解决上述问题
提出 CoWM，明确两项互补任务：
1. 动作生成根据当前观测和目标提出候选动作；
2. 动作条件动力学预测候选动作带来的潜空间后果。

随后解释设计直觉：
> 在有限预算下，动作生成应提供具有任务相关性的候选，而动力学应学习支持对这些候选的有效判断；CoWM 通过联合学习框架将两项任务联系起来，有望提高每次候选评估和动作更新的决策价值。

然后从概念上说明联系：共享表示与参数

4 关键机制 ———— 训练中的耦合 与 预测中的并行化
解释方案如何实现：
1. 共享视觉表征和 DiT，以不同模式完成目标条件动作生成与动作条件未来预测。
2. 将生成动作估计引入动力学训练，并允许动力学损失通过动作输入路径影响生成过程。
3. 基于动作前缀并行预测未来潜状态，减少逐时间步递归预测的串行开销。


5 有限预算下的决策 ———— 耦合模型如何实现 action selection 和 refinement
先说如何 action selection ：并行生成有限数量的候选动作，并行预测后果，依据cost进行reranking
再说如何 action refinement 利用可微预测得到的cost函数，在生成后进行post-opt ，或者在生成过程中进行Guidance Flow ，引导动作调整

强调候选数量、生成步数和修正次数提供了可调节的预算配置。论文关注的是：**在给定时间内获得什么决策效果，以及达到给定效果需要多少时间。**

最后概述实验结果，说明coupling的作用，以及模型对selection和refinement的支持。

6 贡献总结
1. 模型与思想贡献：提出CoWM，明确两项任务之间的训练联系
2. 有限预算下的决策机制：利用耦合模型的动作生成与可微动力学预测，构建支持候选选择及动作修正的决策流程，并通过候选规模和更新步数调节计算开销。
3. 实验贡献：验证决策成功率、决策机制与决策成本，并通过训练对照实验来验证耦合的作用。

## 实验

实验主线：系统效果 → 耦合归因 → 排序与修正机制 → 推理策略 → 成功率与决策时间。物理读出作为辅助分析，真机作为可选扩展。

以下为未填数据的设计模板，不代表实验已经完成，也不自动要求启动所有条件性扩展。`TBD` 表示待填写，`—` 表示不适用；未执行配置最终应明确标注未测，不能填为零。新增基线的具体版本与适用性需核对，沿用 DeWM 的其他基线在定稿时补全。

### 统一实验口径

- 主任务预设为 TwoRoom、Push-T、Reacher、Wall/Cube（最终使用仓库的正式任务名）；保持预设任务集合，不能按结果优劣选择性汇总。
- 固定并记录数据、训练 seed、评测起点、目标、动作范围、预测时域与每次执行动作长度。报告实际完成的训练 seed 数；推理 seed 和 episode 数不能替代训练重复。
- 主表报告闭环成功率及不确定性。配对实验使用相同评测起点，区间计算区分训练 seed 与 episode 层级；说明开发集与最终评测集的复用情况。
- 决策延迟覆盖一次完整 policy 调用：编码、动作生成、后果预测、排序以及需要的反向传播与更新。固定硬件、精度、batch=1 环境设置，预热并同步 GPU，隔离其他负载；报告 p50/p95。候选可在内部批量处理。
- 参数量明确是否包含视觉编码器、全部注册模块或仅活跃模块；训练成本与部署决策成本分开报告。训练时长须与实际数据量、更新步数及硬件对应。

### 表 1：主结果——闭环决策表现

目的：比较 CoWM 与潜空间世界模型/规划基线的整体效果。各任务的预算细节和延迟在表 5 展开；本表本身不等同于等时间预算比较。CoWM 的默认配置在开发阶段选定后固定。

| 方法 | 动作来源 / 决策方式 | 预算配置索引 | TwoRoom SR ↑ | Push-T SR ↑ | Reacher SR ↑ | Wall/Cube SR ↑ | 等任务宏平均 ↑ |
|---|---|---|---|---|---|---|---|
| LeWM | CEM | TBD | TBD | TBD | TBD | TBD | TBD |
| DeWM | 其对应规划器 | TBD | TBD | TBD | TBD | TBD | TBD |
| LeFlow | 按原方法配置 | TBD | TBD | TBD | TBD | TBD | TBD |
| DeWM 中其他适用基线（逐行补全） | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 相同 CoWM-A + 独立 LeWM | 相同候选 + rerank | TBD | TBD | TBD | TBD | TBD | TBD |
| CoWM | Selection，默认配置 | TBD | TBD | TBD | TBD | TBD | TBD |
| CoWM | Refinement，预设配置 | TBD | TBD | TBD | TBD | TBD | TBD |

注意：“相同 CoWM-A + LeWM”固定生成器，检验 scorer 替换的系统效果；它与两个任务完全独立训练的双 DiT 对照不同。Refinement 行需明确采用 PO、GF 或选择后修正，不按测试结果逐任务挑选最优方法。

### 表 2：耦合消融——哪些训练联系有用？

目的：检验耦合各部分的作用，不预设 Full 一定最好。以下为理想对照定义；现有 checkpoint 若不满足定义，应按实际配置命名，不直接填入。

**表 2a：训练配置定义**

| 配置 | DiT 参数共享 | B 训练动作输入 | B→动作→A 梯度 | 说明 |
|---|---|---|---|---|
| A-only（条件性扩展） | — | — | — | 独立动作生成；不能用联合模型 P0 替代 |
| B-only | — | 记录动作 | — | 干净动力学监督参考 |
| Separate-Recorded（条件性扩展） | 否 | 记录动作 | 无 | 两个独立 DiT；其余编码器共享状态需明确 |
| Shared-Recorded | 是 | 记录动作 | 无 | 与 Separate-Recorded 对比参数共享；与 B-only 对比加入 A 任务 |
| Shared-Pred-Detach（条件性扩展） | 是 | 记录动作 + A 动作估计 | 切断 | 保留参数共享，切断动作路径梯度 |
| Shared-Pred-Full | 是 | 记录动作 + A 动作估计 | 保留 | 完整耦合 |

关键配对：Shared-Recorded vs Shared-Pred-Detach 检验输入改变；Detach vs Full 检验动作路径梯度。上述配对需匹配 B 的 timestep 条件、loss 权重、混合比例及训练预算等其他设置。现有 Recorded-control/Recorded-clean 若同时改变这些因素，单独列明，不能归因为纯输入效应。Full vs Shared-Recorded 仅说明输入与梯度联合改变的效果。Separate-Recorded 与 Shared-Recorded 若编码器共享情况也不同，须承认同时改变了多项因素。

**表 2b：决策效果（每任务一组，填写配置与实际训练 seed）**

| 训练配置 | 任务 | Own-A P0 SR ↑ | Own-A + B Selection SR ↑ | Own-A + B PO SR ↑ | 固定候选池 regret ↓ | 固定动作修正收益 ↑ |
|---|---|---|---|---|---|---|
| A-only | TBD | TBD | — | — | — | — |
| B-only | TBD | — | — | — | TBD | TBD |
| Separate-Recorded | TBD | TBD | TBD | TBD | TBD | TBD |
| Shared-Recorded | TBD | TBD | TBD | TBD | TBD | TBD |
| Shared-Pred-Detach | TBD | TBD | TBD | TBD | TBD | TBD |
| Shared-Pred-Full | TBD | TBD | TBD | TBD | TBD | TBD |

Own-A 衡量整个训练系统的效果；固定候选池和固定初始动作衡量 scorer/修正能力，二者分开解释。固定池来源、N、生成步数及动作样本须完全一致；共同候选池不天然消除对某个生成器的偏好，可补充交叉生成器评估。

**表 2c：资源统计（共享与分离模型）**

| 配置 | 总参数 M ↓ | 活跃参数 M ↓ | 更新步数 / 数据量 | 每更新耗时 ms ↓ | 训练峰值显存 GiB ↓ | 总 GPU-hours ↓ | 匹配 N/S/K 的决策 p50 ms ↓ |
|---|---|---|---|---|---|---|---|
| Separate-Recorded | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Shared-Recorded | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Shared-Pred-Full | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

同宽同深双 DiT 与 shared DiT 的总容量不同，属于实际系统资源比较，不能称等参数实验。若要进一步归因于共享本身，可条件性补充等总参数/等计算对照。共享权重不自动减少 A/B 两次前向的计算量。

### 表 3：机制诊断——排序、修正与物理读出

**表 3a：同一候选池上的排序能力（每任务一组）**

| 任务 / 固定池来源 / N | 评分器 | 候选池成功覆盖率 ↑ | 选中动作成功率 ↑ | 物理 regret ↓ | 排序相关性 ↑ |
|---|---|---|---|---|---|
| TBD | 随机选择（期望或重复采样） | TBD | TBD | TBD | — |
| TBD | 独立 LeWM | TBD | TBD | TBD | TBD |
| TBD | CoWM-B | TBD | TBD | TBD | TBD |
| TBD | 池内物理 oracle（诊断上界） | TBD | TBD | 0 | — |

从同一恢复状态执行相同候选，以预先定义的任务物理代价 c 评价；regret = c(选中动作) − min c(池内动作)。预测代价不替代真实执行后果。覆盖率指池内至少一个成功候选的状态比例，在相同池中应相同；它与选中成功率分开报告。排序相关性明确选用 Spearman 等指标，并报告并列值、无效候选及终止处理。候选池诊断成功不等于闭环成功。

**表 3b：同一初始动作上的修正能力（每任务一组）**

| 任务 / 初始动作来源 | 代价模型 / 修正方式 | S / K | 实际动作位移 RMS | 真实代价下降 ↑ | 改善状态比例 ↑ | 预测改善但真实变差比例 ↓ | 对应闭环 SR ↑ |
|---|---|---|---|---|---|---|---|
| TBD | 不修正 | TBD / 0 | 0 | 0 | — | — | TBD |
| TBD | 等 RMS 随机扰动 | TBD | TBD | TBD | TBD | — | TBD |
| TBD | LeWM / PO | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | CoWM-B / PO | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | CoWM-B / GF | TBD | TBD | TBD | TBD | TBD | TBD |

真实代价下降定义为 c(原动作) − c(修正动作)，两者在相同恢复状态下执行；报告动作合法化之后的实际 RMS。GF 使用相同初始噪声和匹配的无引导生成过程作参考，不强行视为与 PO 完全相同的更新问题。S=1 时若 PO/GF 数学等价，不作为两个独立有效性证据。局部改善不直接外推为闭环收益。

**表 3c：物理信息读出（辅助分析）**

| 任务 | 物理量 / 单位 | 表征来源 | 读出器 | LeWM 误差 ↓ | CoWM 误差 ↓ |
|---|---|---|---|---|---|
| Wall/Cube | Block position / TBD | 观测 latent | Linear | TBD | TBD |
| Wall/Cube | Block position / TBD | 预测 latent | Linear | TBD | TBD |
| Wall/Cube | Block position / TBD | 观测 latent | MLP | TBD | TBD |
| Wall/Cube | Block position / TBD | 预测 latent | MLP | TBD | TBD |
| 其他预先选定物理量（条件性扩展） | TBD | TBD | TBD | TBD | TBD |

固定读出数据划分、样本量、预测时域和误差定义，记录 latent 维度与读出器容量；预测表征比较需使用相同动作。当前只确认 Cube block position 的读出范围，其他量按实际完成情况填写。可读出物理量不等于模型学会正确动力学，也不直接证明耦合带来决策收益。

### 表 4：固定模型下的推理策略

目的：回答“已训练的 CoWM 如何使用”。固定同一 checkpoint，区分该表的推理收益与表 2 的训练收益。每任务填写一组。

| 方法 | 初始化 / 候选来源 | 动力学评分器 | N 候选 | S 生成步 | I 搜索轮 | K 修正步 | 闭环 SR ↑ | 决策 p50 ms ↓ |
|---|---|---|---|---|---|---|---|---|
| P0 | A 单次生成 | — | 1 | TBD | 0 | 0 | TBD | TBD |
| Random-N | A 候选池，随机选一条 | — | TBD | TBD | 0 | 0 | TBD | TBD |
| P1 | 随机初始化 CEM | CoWM-B | TBD | 0 | TBD | 0 | TBD | TBD |
| P2 | A 初始化 CEM | CoWM-B | TBD | TBD | TBD | 0 | TBD | TBD |
| P3 | A 候选池 rerank | CoWM-B | TBD | TBD | 0 | 0 | TBD | TBD |
| P0-PO | A 单条候选后修正 | CoWM-B | 1 | TBD | 0 | TBD | TBD | TBD |
| P0-GF | A 生成过程中引导 | CoWM-B | 1 | TBD | 0 | TBD | TBD | TBD |
| P3→PO（可选） | 先选择，再修正选中动作 | CoWM-B | TBD | TBD | 0 | TBD | TBD | TBD |

Random-N 与 P3 复用相同候选，检验评分的作用；P0 是低成本直接执行参考。CEM 的 N 表示每轮候选数，其他方法表示候选池大小；GF 的 K 需说明每个生成步更新几次及总更新次数。若评估先修正全部候选再选择，应另列 P3-PO，不与 P3→PO 合并。

### 表 5：有限决策预算——成功率与延迟

目的：支撑标题中的 Fast。扫描 N/S/K 与 CEM 的候选数/迭代数；优先画各任务的成功率—延迟曲线，以下表格记录对应点。相同 N 不等于相同时间预算。

**表 5a：预算扫描与实际成本（每任务一组）**

| 方法 | 配置 N/S/I/K | 闭环 SR ↑ | p50 ms ↓ | p95 ms ↓ | 推理峰值显存 MiB ↓ | A/B 前向次数 | 反向次数 |
|---|---|---|---|---|---|---|---|
| LeWM + CEM，小预算 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| LeWM + CEM，中预算 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| LeWM + CEM，参考预算 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 相同 A + LeWM rerank | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CoWM P0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CoWM P3，小预算 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CoWM P3，中预算 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CoWM P3，大预算 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CoWM PO | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CoWM GF | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

前向次数按实际串行调用记录，同时给出每次调用的 batch/预测时域；次数不能替代实际 FLOPs 或延迟。20 ms 等数字必须标明具体方法和测量范围。若需将加速归因于并行未来预测，应补充递归/并行实现的对照；否则只报告系统级速度差异。

**表 5b：相同时间预算下的任务效果**

| 任务 | 决策时限 ms | LeWM+CEM SR ↑ | 相同 A+LeWM SR ↑ | CoWM Selection SR ↑ | CoWM Refinement SR ↑ | 超时处理 |
|---|---|---|---|---|---|---|
| TBD | 低预算：TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | 中预算：TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | 高预算：TBD | TBD | TBD | TBD | TBD | TBD |

时间预算预先确定；仅从开发集选择符合时限的配置，再在评测集测量。若用 p95≤时限筛选配置，必须称统计延迟约束，并报告实际超时比例；不能声称硬实时保证。也可补充达到预设成功率阈值的最小延迟，未达到时明确记为未达到。完整 rollout 时间可作为补充，并同时报告执行步数，避免提前成功或失败造成误读。

### 表 6：真机验证（可选，资源和任务明确后再确定）

目的：检验真实交互中的闭环效果与决策延迟；没有执行时不影响前五组模板使用，不将仿真结果填入。

| 任务 / 场景 | 方法 | 试验次数 | 成功次数 | SR及区间 ↑ | policy p50/p95 ms ↓ | 观测到指令发出 p95 ms ↓ | 任务完成时间 s ↓ |
|---|---|---|---|---|---|---|---|
| TBD | LeWM + CEM | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | CoWM Selection | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | CoWM Refinement | TBD | TBD | TBD | TBD | TBD | TBD |

统一任务初始化、控制频率与动作接口；任务完成时间注明仅成功试验统计，或明确失败的截断计时规则。该表中的系统延迟与仿真的纯 policy 延迟分开解释。

排版建议：上述为完整实验设计模板，不要求全部作为正文大表。正文优先放主结果、关键耦合对照、排序/修正证据和成功率—延迟曲线；完整配置、资源统计、物理 probe 及扩展推理结果按篇幅放入附录。

## 讨论

科学问题: 
在有限决策预算下，联合学习动作生成与动作条件动力学，能否提高模型对候选动作的评估（rerank）和修正（Guidance，PO/GF）能力，从而减少对大规模搜索的依赖？
1. 耦合模型，联合学习。
2. 并行评估action、使用cost梯度修正action。
3. 有限预算 / 推理速度 / fast

耦合： 
1. 参数耦合，使用同一个shared DiT 实现两个任务。
2. 梯度耦合，梯度回传。A=》B，B的latent loss回传到A。
3. 动作输入分布耦合。（特指B在训练的时候混入A的action，从而让B预先在A的输出action分布上进行优化，提升B对于A的action的rerank效果）

评估： 
1. 候选动作排序（P3 ， A输出action + B rerank ，速度快效果好）
2. 动作梯度修正方向（PO/GF，在reacher任务有提升）
3. 物理特性probe（复用lewm的实验，但重点关注cube任务中lewm做不好的几个物理量）

fast/硬件效率 ：
平均单次推理时间为 20ms 
闭环rollout后边测一下。

## 附录
放一下R4-AB和R4-ABCD的对比

## 删去

~~(旧版，删去)
先说明搜索式规划的计算开销来源：大规模候选评估、多轮搜索，反复采样候选动作、预测后果并更新搜索动作分布。
然后以lewm和cem solver为例，使用数据说明，即使单次推理仅需20ms，大规模候选评估与多轮搜索仍需要数十秒的时间才能输出最终的动作结果。
再引出高效动作采样、评估和修正的重要性
最后凝练研究问题：
> 在有限决策预算下，联合学习动作生成与动作条件动力学，能否提高模型对候选动作的评估（rerank）和修正（Guidance，PO/GF）能力，从而减少对大规模搜索的依赖？~~

~~(旧版，删去)
先明确两种用途的要求： action selection需要模型能够有效区分候选动作的后果，refinement需要模型能够为候选动作的优化调整提供有效依据。
然后说明现有方法（lewm）在这两个方面的建模不足，并且说明它是如何影响动作决策过程。
最后凝练研究问题~~

~~(旧版，删去)
有限预算下，既需要有用的候选，也需要能够对这些候选做出有效判断和修正的预测模型。CoWM 将二者置于联合学习框架中，研究这种联系是否能够改善决策过程。
并交代核心设计。
1. 耦合的具体对象是什么
2. 耦合的对象之间的关系是什么样的
3. 显示建模耦合关系，为什么能够缓解这一科学问题~~

~~(旧版，删去)
段末预告实验围绕三个问题展开：
1. 联合学习是否改善对候选的评估与修正？
2. 这些局部能力是否转化为闭环任务收益？
3. 相比搜索式规划和分离模型，成功率与延迟之间的关系如何？~~
