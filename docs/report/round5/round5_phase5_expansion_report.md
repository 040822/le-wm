# Round 5 Phase 5 扩展：五种原生动作结构报告

日期：2026-10-05。状态：PushT、Reacher 的五种结构训练与开发集评测产物齐全。本报告按 [Phase 5 扩展计划](../../plan/round5_phase5_expansion_plan.md) 的 C 阶段记录动作结构主矩阵；其余任务与后续实验留待用户决定。

## 执行范围与配置

本轮只训练 **PushT、Reacher**，训练 seed 为 **3072**。五种结构按 `k×H` 命名：`k` 是每个动作 token 覆盖的环境步数，`H` 是动作 token 数，模型预测跨度为 `kH` 步。

| 结构 | k | H | 预测跨度 | P3 评测的 execute / score（环境步） |
|---|---:|---:|---:|---|
| 5×5 | 5 | 5 | 25 | execute 1、5、10、25；score 5、10、25 |
| 1×25 | 1 | 25 | 25 | execute 1、5、10、25；score 1、5、10、25 |
| 5×1 | 5 | 1 | 5 | execute 1、5；score 5 |
| 1×5 | 1 | 5 | 5 | execute 1、5；score 1、5 |
| 1×1 | 1 | 1 | 1 | execute 1；score 1 |

10 个训练格均从头训练，没有复用原 5×5 checkpoint。训练固定 25 个环境步的有效窗口和 A 目标，按数据集 seed 3072 的 90/10 划分；每格训练 10 epochs、AdamW、lr=5e-5、有效 batch=128。实际 micro-batch 为 32，梯度累积 4 次。P3 推理使用 64 个候选、2-step Euler。

| 任务 | 五种结构实际训练 GPU | checkpoint |
|---|---|---|
| PushT | 5×5: GPU0；1×25: GPU2；5×1: GPU1；1×5: GPU1；1×1: GPU3 | 5/5 个 epoch-10 checkpoint 均存在，哈希记于 scheduler state |
| Reacher | 5×5: GPU1；1×25: GPU3；5×1: GPU0；1×5: GPU0；1×1: GPU3 | 5/5 个 epoch-10 checkpoint 均存在，哈希记于 scheduler state |

训练使用 GPU0–3；低显存评测分派使用 GPU0–7，符合本轮 GPU 授权。评测输出记录了每个 worker 的 GPU 和 checkpoint/cohort 哈希。

评测使用每任务 50 个固定开发起点，配置 seed=42、goal offset=25、闭环预算 50 步。PushT 使用 `round3_revised` 开发 cohort（SHA256 `5b3cd950…`），Reacher 使用 `legacy_50`（SHA256 `ff4f26ad…`）。全部结构统一使用物理动作裁剪；比较 P3、P0 和 Random-64。P0 不使用 verifier score。共 **122 个条件**，每个 50 episodes，即 6,100 条跨条件 rollout 记录；同一任务的条件复用相同 50 个起点，不能把 6,100 当作独立样本数。

## 主要结果

下表列出每个任务、每个结构的 P3 最高观测条件。`execute/score` 以环境步计；区间为该 50-episode 比例的 Wilson 95% 区间。每格是在多种时域组合中取最高值，因此适合筛选候选，不是无偏的最终估计。

| 任务 | 结构 | 最高观测条件 | 成功数 | 成功率 | Wilson 95% CI |
|---|---|---|---:|---:|---:|
| PushT | 5×5 | 25/10 | 41/50 | 82.0% | [69.2%, 90.2%] |
| PushT | 1×25 | 25/10（与 25/5 并列） | 42/50 | 84.0% | [71.5%, 91.7%] |
| PushT | 5×1 | 5/5 | 30/50 | 60.0% | [46.2%, 72.4%] |
| PushT | 1×5 | 5/5 | 32/50 | 64.0% | [50.1%, 75.9%] |
| PushT | 1×1 | 1/1 | 2/50 | 4.0% | [1.1%, 13.5%] |
| Reacher | 5×5 | 25/10（与 5/5 并列） | 19/50 | 38.0% | [25.9%, 51.8%] |
| Reacher | 1×25 | 1/1 | 25/50 | 50.0% | [36.6%, 63.4%] |
| Reacher | 5×1 | 5/5 | 48/50 | 96.0% | [86.5%, 98.9%] |
| Reacher | 1×5 | 5/5 | 49/50 | 98.0% | [89.5%, 99.6%] |
| Reacher | 1×1 | 1/1 | 27/50 | 54.0% | [40.4%, 67.0%] |

因此，当前结果支持用户观察：PushT 的强候选是 **1×25 与 5×5**，Reacher 的峰值是 **1×5、execute/score=5/5、P3**；Reacher 的 **5×1 5/5** 也只低一个 episode（96%）。相邻候选的区间明显重叠，50 个起点不足以证明这些名次稳定。

同条件的无选择对照显示 P3 的筛选作用：PushT 的 1×25、execute=25 时，P3 84%（score=5/10）对比 P0 78%、Random-64 74%；Reacher 的 1×5、execute/score=5/5 时，P3 98% 对比 P0 22%、Random-64 14%。这些仍是单 cohort、单训练 seed 的开发集结果。

## 与主表结果的差异

正式 Table 1 的标准 CoWM-Selection（P3）为 PushT **97.33 ± 1.03%**、Reacher **88.33 ± 4.97%**，统计六个评测 seed、每 seed 50 episodes。Phase 5 的直接 5×5、execute/score=25/25 对应结果为 PushT **78%**、Reacher **12%**。不过本轮最优结构/时域并非都低于主表：Reacher 1×5、5/5 达到 98%。这些数字不能作同协议的模型排名。

能确认的设置差异如下：

- **checkpoint 不同。** 主表 P3 使用 Round 4 的 seed-3072、epoch-10 checkpoint；Phase 5 的五种结构（包括 5×5）全部新训，权重、样本清单及 checkpoint 哈希均不同。
- **训练 micro-batch 不同。** 主表原始训练配置为 batch=128、梯度累积=1；Phase 5 用 micro-batch=32、累积=4，名义有效 batch 虽同为 128，但模型 projector 含 `BatchNorm1d`。累积梯度不会合并不同 micro-batch 的 BatchNorm 统计量，因此两种训练并不等价。这是值得优先审计的差异，但现有实验没有隔离它的因果影响。
- **动作处理不同。** 主表标准 P3 为 `action_bound_mode=none`；Phase 5 统一使用 `clip`。主表同一旧 checkpoint 的裁剪对照中，P3 的 PushT 均值从 97.33% 变为 97.00%，Reacher 仍为 88.33%，所以裁剪本身不足以解释本轮 5×5 Reacher 的大幅差距；该对照也不能排除裁剪与新 checkpoint 的交互。
- **样本协议不同。** PushT 主表使用 `pusht_legacy_50_v1`，Phase 5 用 `round3_revised` 开发 cohort，起点哈希不同。Reacher 两边的 `legacy_50` cohort SHA256 相同，因此 Reacher 的低 5×5/25/25 结果不能只归因于起点集合变化。
- **评测时域不同。** 主表标准行按 25/25 评测；Phase 5 同时改变结构、执行频率和评分窗口。只有同为 5×5、25/25 的 Phase 5 行接近主表协议；其余格子回答的是不同控制时域下的表现。
- **样本量与随机性不同。** 主表每任务 300 episodes、跨 6 个评测 seed；本轮每任务只有一个 50 起点开发 cohort。主表的误差条反映评测 seed 变化，不反映训练 seed 变化；本轮同样没有估计训练 seed 变异。

因此，目前能确定的是“两个实验不能直接比较”，尚不能把低分单独归因于某一项。尤其是 Reacher 在 cohort 相同的情况下，旧 checkpoint 主表 25/25 的 seed-42 结果为 88%，新训 5×5/25/25 为 12%；checkpoint、micro-batch/BatchNorm、裁剪和推理随机流均可能参与。应优先复核 5×5 新训练与旧 Round 4 checkpoint 的训练身份、BatchNorm 行为和复现差异，再决定是否把该落差解释为动作结构效应。

## 对正式方法选择的影响

本轮适合作为结构筛选：它把候选缩小到 PushT 的 1×25/5×5、Reacher 的 1×5/5×1，并暴露了明显的任务×结构交互。它不足以冻结正式主方法。PushT 最高两格只差一个 episode；Reacher 的 1×5 与 5×1 也只差一个 episode。再加上从 122 个条件里报告每任务峰值会产生选择偏差，不能把最高点估计当作确认性证据。

原计划要求四任务共同选择一个动作结构和主推理规则，只允许评分/执行时域按任务变化。本轮只覆盖两个任务，因此不能据此决定四任务统一结构，也不应直接把“PushT 用 1×25、Reacher 用 1×5”写成最终主方法。待其余任务开展后，应在预先固定的开发规则下共同选型，再对锁定方案做独立复核；若更新正式主表，至少按其六个评测 seed × 每 seed 50 episodes 的协议重跑。若要声称训练初始化稳健，还需要单独增加训练 seed。

评测目录记录的总运行秒数混有环境步进、结构带来的重规划次数和设备负载，不能替代隔离 GPU 上的单次推理 p50/p95；正式选型不能只按该总时长排序。

## 范围外与产物

本轮没有运行其他三项任务、A/B/D/E 阶段、近目标训练对照、新 100 起点复核或引导参数扫描；这些留待后续安排。

主要产物：

- 训练配置：[phase5_expansion.json](../../../config/round5/phase5_expansion.json)，评测配置：[phase5_expansion_eval.json](../../../config/round5/phase5_expansion_eval.json)
- 训练身份、数据集与样本清单：`outputs/round5_phase5_expansion_seed3072/` 及各结构 checkpoint 目录
- 训练状态：[scheduler_state.json](../../../outputs/round5_phase5_expansion_seed3072/scheduler_state.json)
- 122 条条件的汇总：[conditions.csv](../../../outputs/round5_phase5_expansion_seed3072_eval/analysis/conditions.csv)
- 全量条件表：[analysis/report.md](../../../outputs/round5_phase5_expansion_seed3072_eval/analysis/report.md)，机器可读统计：[analysis.json](../../../outputs/round5_phase5_expansion_seed3072_eval/analysis/analysis.json)

记录说明：`scheduler_state.json` 顶层仍标记 `running`，但 10 个训练 job 条目均标记 `complete`，并记录 epoch-10 checkpoint 路径和 SHA256；其中部分 supervisor 退出码不可用。此处的“完成”依据每个训练日志到达 10 epochs、最终 checkpoint 与身份记录存在，不代表调度器正常写入了最终汇总状态。
