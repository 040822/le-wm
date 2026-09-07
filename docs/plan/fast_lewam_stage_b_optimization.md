# Fast-LeWAM 第二轮：Stage B 性能调优实验计划

## 1. 目标与判定口径

第二轮实验的核心目标是解释并消除 Fast Stage B 相对本地 LeWM 的性能差距。

- 本地 LeWM 基线：Cube 78%，Push-T 98%。
- 新实验先使用训练种子 `3072` 筛选，任务顺序为 Cube → Push-T；是否补充其余训练种子由筛选结果决定。
- 候选方案满足以下任一条件时，进入人工复核并决定是否补种子：
  - Cube 和 Push-T 均追平 LeWM；
  - 两任务平均成功率超过 LeWM；
  - 某一任务超过对应 LeWM 基线 10 个百分点以上，另一任务没有明显退化。
- 主结果必须报告纯 Stage B、随机初始化 CEM 的表现。E6 有效后，额外报告 Stage A warm-start CEM，作为统一 WAM 的系统结果。
- 第二轮先追求成功率，但所有候选仍记录参数量和 planning latency，为后续恢复“Fast”的工程约束保留依据。
- 所有第二轮横向训练实验统一使用单卡、global `batch_size=128`。本地 E0 LeWM 基线也是 batch size 128，因此不使用默认 LeWM 配置中的 200，避免 batch size、optimizer update 数和 BatchNorm 统计成为混杂因素。

## 2. 第一阶段：无重训诊断

第一阶段先确定性能瓶颈来自 Stage B cost、CEM 搜索还是训练数据与目标定义。完成本阶段并复盘结果后，再执行后续训练实验。

### 2.1 完成 E6

- 在相同 checkpoint、相同评测 cohort 下比较随机初始化 CEM 与 Stage A warm-start CEM。
- 纯 Stage B 仍为主口径；warm-start 结果单独报告。
- 固定 CEM 候选数、迭代数、方差和随机种子，避免把 solver budget 变化混入 E6。

### 2.2 可视化训练窗口

在 Cube 和 Push-T 各抽取固定样本，展示：

- `z0...zH` 对应的原始图像；
- 25 步窗口终点 goal；
- 5 个 Action Block 及其内部原始动作曲线；
- 当前状态、窗口 goal 和任务成功状态之间的关系。

该检查用于确认训练 goal 与评测 goal 的对应方式，并判断 25 步窗口是否包含足够明确的动作后果。

### 2.3 Simulator-grounded Stage B 排序诊断

从同一可恢复 simulator state 构造以下候选动作：

- expert action；
- expert action 的局部扰动；
- Action Block 时间置换；
- 随机 CEM 候选；
- Stage A 输出及其邻域采样。

对每个候选真实执行 25 个环境步，并记录：

- Stage B predicted latent cost；
- 真实终点到 goal 的距离；
- predicted cost 与真实距离的 Spearman 相关性；
- top-k recall；
- best-candidate regret；
- 对动作局部扰动和时间置换的敏感性。

至少比较 E0 LeWM、E1、E3 和 E4。诊断结论按以下方式解释：

- 排序相关性低：优先修改 Stage B 模型或训练目标；
- 排序相关性高但成功率低：优先修改 CEM；
- expert latent MSE 低但候选排序差：说明当前 MSE 目标与 planning 需求不一致。

## 3. 第二阶段：优先解释性能缺口

### 3.1 同量级扩容的 E1 Stage-B-only

这是第二轮第一个重训实验。

- 保持 `frameskip=5`、`action_horizon=5`、`depth=6`、`heads=6`。
- 将 `model_dim` 从 192 调整为 384，`mlp_dim` 从 768 调整为 1024。
- 该配置下每个 attention head 为 64 维，且 384/1024 是更规整的硬件友好维度。
- Fast 非视觉部分参数量约为 14.14M，LeWM 非视觉部分约为 11.74M；Fast 高约 20%，仍处于同一量级。因此该实验定义为“同量级扩容”，而不是严格的参数量匹配。
- 该实验用于判断现有 Fast Stage B 的性能缺口是否主要来自容量不足；若结果超过 LeWM，只能说明容量充足的 Fast 架构可以达到或超过 LeWM，不能据此声称同参数量下优于 LeWM。
- 单卡训练，global `batch_size=128`；学习率保持 `5e-5`，不做 batch-size scaling。
- 其余数据、优化器、训练 epoch、loss、CEM 和评测 cohort 与现有 E1 一致。
- 先运行 Cube；若结果没有改善至少 4 个百分点且排序指标也没有改善，则暂停 Push-T，优先进入 3.4 的 serial control 诊断。
- 若 Cube 明显改善，则继续 Push-T；若同量级扩容 E1 成为更强基线，再用相同容量运行 E3。

#### 3.1.1 参数分配匹配的补充对照

384/1024 配置检验的是在固定 `depth=6` 和现有 attention 实现下进行宽化是否有效，不能单独排除“总参数量相近，但参数分配方式不同”这一混杂因素。尤其需要区分以下两点：

- Fast attention 的 QKV 内部宽度固定为 `model_dim`，`heads` 只把该宽度重新分组；在 `model_dim` 不变时单独增加 `heads` 基本不增加参数量，因此不把 heads-only sweep 视为扩容实验。
- LeWM 使用独立的 `dim_head=64`，attention 内部宽度为 `heads * dim_head = 1024`。因此 LeWM 的 16 heads 同时扩大了 QKV 内部宽度，而 Fast 当前的 6 heads 不具备相同的参数语义。

如果 384/1024 宽化没有带来稳定收益，但仍需更严格地检验“Fast 相对 LeWM 的参数量缺口是否导致性能差距”，只补充一个参数分配匹配对照，不展开 depth × width × heads 网格搜索：

- 为 Fast attention 增加独立的 `dim_head` 或 `attention_inner_dim` 配置，使 QKV 内部宽度与 token 的 `model_dim` 解耦。
- 候选配置为 `model_dim=192`、`depth=6`、`heads=16`、`dim_head=64`（`attention_inner_dim=1024`）、`mlp_dim=2048`，对齐 LeWM predictor 的主要 block 容量分配。
- 实现后先精确统计总参数与非视觉参数；目标是接近 LeWM 的非视觉参数量，而不是再次明显超出。encoder、projector、数据、训练预算、loss、CEM 和评测 cohort 保持与 E1 对照一致。
- 首轮只运行 E1 Stage-B-only，并优先用 Push-T 检查 final 成功率以及 simulator-grounded Spearman、top-k recall 和 regret；没有明确收益时不继续 E3 或多任务扩展。
- 若该配置改善成功率和局部排序，结论应是参数分配或 attention 内部秩可能重要，不能归因于总参数量本身；若相对 E1-192 和 E1-384 均无改善，则停止继续探索模型规模，进入 3.4 的 serial one-step control。

加深窄模型可作为更低优先级的结构消融，但 Fast 与 LeWM 当前均为 6 层，原始参数差距并非由深度不同造成；在 serial control 给出需要更长计算链的证据之前，不优先运行深度扩容。

### 3.2 修正 Stage B timestep

该实验只影响 E4/E5 的预测动作分支，对 E1/E3 无影响。

- Stage B 接收的是真值 clean action 或 Stage A 的 clean estimate，因此 Stage B AdaLN condition 始终使用 `t=1`。
- Stage A 生成 clean estimate 时的原始 flow timestep 记为 `source_t`。
- `source_t` 只用于 latent-loss 置信度权重，不再作为 Stage B 的模型条件。
- 首先基于 detached E4 运行；未显示收益前，不继续投入 cross-head E5。

本轮筛选固定使用 192/768 容量、训练种子 3072，并运行 Cube 与 Reacher：

- E4 保持 `detach_clean_action=true`、`latent_action_mix_epochs=10`；
- 只启用 `stage_b_timestep_mode=clean_action`，位置编码显式保持 `token_encoding=legacy`；
- 主比较分别使用第一轮 E4 final Stage-B：Cube 74、Reacher 80。

### 3.3 对齐 terminal planning 目标

在当前最强的纯 Stage B 配置上比较 uniform prefix loss 与：

\[
L_{\text{latent}}
=0.5\,\operatorname{mean}(L_{1:H})+0.5\,L_H
\]

- 保持 latent loss 的总体尺度近似不变。
- 重点观察 terminal ranking、CEM 每轮候选质量和最终成功率。
- 只有 simulator-grounded 诊断证明 expert/negative 标签可靠时，才增加 ranking loss；不直接用缺乏真实反事实后继的预测动作构造强排序监督。

### 3.4 同 backbone 的 serial one-step control

如果容量匹配和 terminal loss 仍无法追平 LeWM，建立以下诊断对照：

- 训练时使用真实 `(z_i, a_i) -> z_{i+1}` 一步转移；
- 推理时递归 rollout 5 次；
- encoder、model dimension、深度和参数预算与对应 parallel Stage B 匹配。

该实验只判断性能差距是否主要来自 causal-prefix 并行预测，不作为最终 Fast 方法。当前 Stage B mask 已保证预测看不到未来动作，但它学习的是
`z0 + action prefix -> z_k`，不是显式 Markov transition，因此不优先进行缺乏明确假设的 mask sweep。

### 3.5 A/B 梯度冲突诊断

- 在 E3 中分别记录 action loss 与 latent loss 对 encoder/projector、SharedDiT 的梯度 cosine similarity。
- 若 Push-T 上出现持续负相关，先测试 Stage A 对视觉 encoder/projector stop-gradient，同时保留 SharedDiT 的 action supervision。
- 只有 stop-gradient 仍不能缓解冲突时，再测试 Stage-B warmup → E3 joint schedule。

## 4. 第三阶段：表示结构实验

### 4.1 统一 learned physical-time position 与 type embedding

先保留可学习位置编码，只统一 A/B 的时间语义：

- 使用一套 `time_positions[0:H]`；
- `z0` 位于 `t=0`；
- `a_h` 位于 `t=h`；
- 预测 `z_{h+1}` 的 query 位于 `t=h+1`；
- `zg` 位于 `t=H`；
- 增加 `STATE`、`GOAL`、`ACTION`、`QUERY` type embedding；
- query 使用一个共享 content token，避免 `[H,D]` 的独立 query 参数继续隐式编码绝对位置。

Stage A 和 Stage B 中同一个 `a_h` 必须使用相同的 action projection、time position 和 action type。

本轮筛选固定使用 192/768 容量、训练种子 3072，并运行 Reacher 与 Push-T：

- E3 保持 expert-action joint training，使用 `latent_action_mix_epochs=1e9`；
- 只启用 `token_encoding=physical_time_type`，Stage-B timestep 显式保持 legacy；
- 主比较使用第一轮 E3 final：Reacher A/A-shuf/B=76/4/82，Push-T=96/4/90。

### 4.2 固定 sinusoidal physical-time position

在 4.1 的结果基础上，仅把 learned time table 替换为固定 sinusoidal encoding，其他结构保持不变。

- 时间使用 Action Block 单位 `0...H`；
- 不复用当前为 flow timestep 设计、包含 `×1000` 缩放的 embedding；
- type embedding 继续保持可学习。

### 4.3 Action midpoint ablation

仅当 sinusoidal physical-time position 有效时，比较：

- action 位于区间起点 `t=h`；
- action 位于区间中点 `t=h+0.5`。

该实验属于低优先级相位消融，不单独抢在 position/type 实验之前运行。

### 4.4 25 action + 25 latent query

该实验恢复原始环境动作的显式时间粒度：

- `frameskip=1`；
- `action_horizon=25`；
- `action_dim=env_action_dim`；
- Stage A token 为 `[z0,zg,a0...a24]`；
- Stage B token 为 `[z0,a0,q1,...,a24,q25]`；
- eval 使用 `horizon=25`、`action_block=1`，主比较保持每 25 个环境步重新规划。

为单独检验 action 拆分，首轮使用 legacy position，不同时引入 4.1/4.2。训练时：

- effective batch 固定为 128；
- 使用显存可容纳的最大 2 的幂作为 micro-batch；
- 通过梯度累积恢复 effective batch；
- optimizer update 数、物理预测跨度和评测 cohort 与原方案保持一致。

需要同时检查两个风险：相邻帧 latent 变化过小导致 identity shortcut，以及更长序列造成的计算和优化困难。

## 5. CEM、Stage C 与 Online 的位置

### 5.1 CEM 调优

- E6 之后先比较 `var_scale` 和候选数，再考虑更复杂 solver。
- 同时报告相同 CEM budget 与相同 latency budget。
- 如果 Stage B 的真实候选排序可靠但 CEM 失败，CEM 调优提升为高优先级；否则不通过增大搜索预算掩盖模型问题。

### 5.2 Stage C

Stage C 暂时保留为结构负对照，不投入主要训练资源。它当前没有 goal token，且 action token 受 causal mask 限制，不是公平的 Stage A+B 单次融合。

### 5.3 Online training

Online training 放在最后。启动条件为：

- 纯 Stage B 的真实候选排序已经可信；
- E6 是否有效已经明确；
- 离线表示和训练目标的主要混杂因素已经排除。

初始 online 版本冻结视觉 encoder，只更新 dynamics/latent head，混合 offline replay，并保存真实 `(z_t,a_t,z_{t+1})`。确认 World Model online update 有效后，再考虑联合更新 Action Head。

## 6. 实验优先级汇总

| 项目 | 必要性 | 可行性 | 优先级 |
| --- | --- | --- | --- |
| E6 与 simulator-grounded 排序诊断 | 高 | 高/中 | P0 |
| 同量级扩容 E1/E3（384/1024） | 高 | 高，仅需配置扩容 | P1 |
| Stage B condition 固定 `t=1` | 高，但仅影响 E4/E5 | 高 | P1 |
| Terminal-weighted latent loss | 高 | 高 | P1 |
| Serial one-step control | 高，属于关键归因实验 | 中 | P1 |
| A/B 梯度冲突与 stop-gradient | 中高 | 中 | P2 |
| 共享 learned position + type | 中高 | 高 | P2 |
| Learned → sinusoidal | 中 | 高 | P2 |
| 25 action + 25 query | 中 | 中低，计算成本高 | P3 |
| Action time `+0.5` | 低 | 高 | P4 |
| Online training | 长期高、当前低 | 低 | P5 |

## 7. 验证与结果记录

实现阶段需要补充以下自动化验证：

- H=5/H=25 的输入输出 shape 和 eval action 展开测试；
- 所有横向训练配置均为单卡、global `batch_size=128`，且未隐式启用梯度累积；
- Stage B 的所有 query 均不可读取未来动作；
- A/B 中相同 `a_h` 使用相同 physical-time position 和 action type；
- sinusoidal position 的整数和 `+0.5` 时间值测试；
- Stage B condition 恒为 `t=1`，`source_t` 只影响配置允许的 loss weighting；
- 纯 Stage B 与 actor warm-start 使用相同 cohort 和 CEM budget；
- smoke training 中 loss 有限、有效 batch 正确、梯度路径符合配置。

每个实验保存：

- final epoch 成功率及逐 episode 结果；
- held-out latent MSE 和已有 expert preference/top-1/margin；
- simulator-grounded Spearman、top-k recall 和 regret；
- 参数量、训练吞吐、显存和 planning latency；
- 完整配置、checkpoint 和随机种子。

主结论使用 final epoch，不使用测试成功率挑选中间 checkpoint。

### 7.1 本轮四卡执行与验收

| GPU | 实验 | 任务 | final 评测 |
| --- | --- | --- | --- |
| GPU0 | 3.2 E4 clean-action timestep | Cube | Stage B |
| GPU1 | 3.2 E4 clean-action timestep | Reacher | Stage B |
| GPU2 | 4.1 E3 physical-time/type | Reacher | Stage A、A-shuf、Stage B |
| GPU3 | 4.1 E3 physical-time/type | Push-T | Stage A、A-shuf、Stage B |

- 四个任务均为单卡、global batch 128、10 epochs、fresh initialization；训练时关闭 epoch eval。
- 每个训练命令与后续离线评测显式限制对应的 `CUDA_VISIBLE_DEVICES=0/1/2/3`，禁止 GPU4–7。
- 训练完成后在原 GPU 自动运行 epoch-10、固定 50 episode 的相关阶段评测；不运行 Stage C、warm-start 或 simulator-grounded 排序诊断。
- 两项改动使用正交配置开关且默认均为 legacy；任一实验只开启自己的目标开关，并保存到独立 run directory。
- 验收要求为 epoch-10 weights 与 `last.ckpt` 完整，日志无 OOM/NaN/traceback，final `result.json` 为 `status=ok`，并汇总成功率、validation diagnostics、显存和 wall-clock。
- 单种子结果只作为筛选信号；本轮不自动启动 E5 或补种子。
