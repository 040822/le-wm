# Round 5 Phase 5：Action horizon 探索实验

## 目标与已有证据

本阶段探索 FastLeWAM 从 5×5 action chunk 缩短到较短 horizon 时的闭环效果和推理性能。实验限定 Reacher、训练 seed 3072、推理 seed 42，每个条件使用同一组 50 个起点。分两阶段执行：先对现有 5×5 checkpoint 做推理实验；完成并汇报第一阶段后，再训练三个短 chunk 变体。用户要求使用 `gpt-6-luna`、reasoning effort `max`；当前会话接口不支持切换或核验实际运行模型，因此配置记录了请求，但本轮实际执行模型无法确认与该请求一致。

按“维度轴 token 数 × 时间轴 token 数”命名：5×5 使用 action_dim=10、action_horizon=5，覆盖 25 个环境步。Round 2 的 serial one-step control 保持 action_horizon=5、frameskip=5，每个 B transition 单步预测一个 5-step block，再递归预测；Reacher 结果从 84% 降至 72%，它是相关动力学对照，不是原生 5×1 actor 实验。Cube temporal-token pilot 将 5×5 改为 frameskip=1、horizon=25，初步成功率 95%→85%，但 grounding 未完成，只作为不完整历史参考。证据见 `docs/summary/round1_round2_summary.md` §8 和 §9.2。

## 冻结协议和实验矩阵

- 共用 Phase 1 Reacher `legacy_50` cohort、goal offset=25、闭环预算=50 primitive steps、原始成功判据；推理 seed=42，训练 seed=3072。
- 推理条件共10种：P0/P2/P3 各运行无 guidance、guided-flow（GF）、post-opt（PO）；P1 仅运行无 guidance，flow step 不适用。flow step=2、Euler、P3候选数64。P1/P2 使用 Reacher `cem-clip`、CEM 300/30/30、var_scale=1；P0/P3 不做额外动作裁剪，沿用 Phase 1 协议。
- GF/PO 沿用 step_size=0.01、last_steps=5、inner_steps=5、max_rms_offset=0.2。CEM、候选数、归一化、checkpoint、cohort 及成功判据均保持不变。

### 第一阶段：固定 5×5 checkpoint 的推理实验

始终生成完整的25步动作，B 评分及 CEM 搜索动作维度保持完整；只改变实际执行前缀长度。执行长度为25、10、5、1个环境步。评分时点分两组：所有执行长度都使用完整25步评分；执行10步另使用第10步评分；执行5步另使用第5步评分。B prefix cost 在 CEM、GF、PO 中一致使用。现有 B 只有每个5-step action block 后的 latent，不插值出第1步预测，因此1步执行只对应25步评分。

矩阵共60个逻辑单元：4种执行长度×10种推理条件，共40项完整评分；10步和5步各增加10项短评分。P0无引导不使用 B，重复的P0条件复用同执行长度的结果。兼容且身份校验通过的10项 Phase 1 基线直接复用。

### 第二阶段：原生训练短 chunk 变体

| 变体 | frameskip | action_horizon | Reacher action_dim | 预测及执行跨度 |
|---|---:|---:|---:|---:|
| 5×5（现有基线） | 5 | 5 | 10 | 25步 |
| 5×1 | 5 | 1 | 10 | 5步 |
| 1×5 | 1 | 5 | 2 | 5步 |
| 1×1 | 1 | 1 | 2 | 1步 |

三个新模型从头训练 R4-AB、stage_ab：统一10 epochs、effective batch=128、lr=5e-5、训练 seed=3072；其他模型、损失和数据划分参数继承现有配置，正式评测固定 epoch 10。显存不足时降低 micro-batch 并用梯度累积保持 effective batch。记录每臂训练窗口数、optimizer 更新数与训练耗时；相同 epochs 不代表相同更新预算。

训练目标使用各模型原生终点（5×1和1×5为5步，1×1为1步）；所有模型仍面对25步远的 Reacher 评测目标。每个模型运行同样10种推理条件，共30个条件；与现有5×5基线合并比较。

## 实现、执行与产出

- 所有新增结果写入独立 Round5 Phase5 目录，输出逐条件 JSON、逐 episode trace、汇总 CSV/JSON 和 `docs/report/round5/round5_phase5_report.md`。计划保存在本文件；不覆盖 Round 5 Phase 4 草稿或已有产物。
- policy/evaluator 增加以 primitive environment steps 为单位的 `execute_steps`；动作经 action-block 展开后只缓存指定前缀，耗尽后重新规划。新增以 action-block 为单位的 `score_horizon_blocks`；省略时保留末端评分语义，越界或不能对齐 block 的值报错。模型及评测规划配置必须显式匹配 action_horizon、action_dim 和 frameskip；旧实验默认值和 H=1 的 A/B 约束保持可用。Round4 D 路径仍要求 horizon≥2。
- 每条件记录 cohort/checkpoint/配置/代码身份、评分与执行 horizon、规划次数、A/B 前向次数、B 反向次数、成功率、首次成功步数、p50/p95 延迟、吞吐及峰值显存。
- 评测效率计时采用 batch 1 和 batch 50，各10次预热、50次同步测量、固定观测、指定 GPU；记录进程和负载。训练/评测启动前查询显存并保留峰值余量；GPU命令显式设置 `CUDA_VISIBLE_DEVICES`。优先 GPU0–3；用户已授权本实验 GPU0–7。
- 正式运行前进行语义 smoke：确认动作块展平顺序、prefix latent index、评分和执行前缀一致、重规划次数正确、CEM/GF/PO 使用相同评分时点、终止和状态恢复清空动作缓冲区。确认 Phase1 基线 checkpoint/cohort 哈希和 trace 身份后方可复用。
- 第一阶段分别报告执行长度与评分时点的闭环成功率和配对差、达到成功的步数、规划频率及完整推理延迟。第二阶段再报告训练出的原生 chunk 结果。第一阶段不将执行截断解释成模型计算量缩短；单训练 seed 结果只作探索性证据，并披露各臂训练更新预算差异。
