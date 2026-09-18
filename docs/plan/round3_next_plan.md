# Round3 后继实验计划：Online Supervision for Latent Prediction

## 总体目标

在已有 online latent-prediction B-MSE 基础上，加入并比较三类新的 online 监督信号：

- T3：grounded ranking，改善 Stage B 候选排序；
- T5：online hindsight Stage A，改善动作生成；
- T6：online planner distillation，模仿真实执行结果较好的动作。

实验严格限定为 Reacher 和 Push-T。E1/E2 是前置诊断；若 guidance 未通过用户判断，阶段二使用未引导的 A+B 采集器。所有门槛只作为参考统计，不自动停止、筛选或扩展实验，由用户根据结果决定下一步。

主线模型固定为已有 task-specific legacy E5 epoch-10 checkpoint：

- Reacher：`outputs/fast_lewam/reacher/0717_/checkpoints/fast_lewam_weights_epoch_10.pt`
- Push-T：`outputs/fast_lewam/pusht/0803_e5_cross_head_grad/checkpoints/fast_lewam_weights_epoch_10.pt`

所有 checkpoint、代码版本、数据和 cohort hash 写入独立 manifest，不修改 Round3 历史结果。

## 阶段一：冻结模型的 guidance 诊断

### E1：局部梯度真实性

每个任务从既有 online pool 中固定抽取 64 个非初始成功状态，使用 `seed=42`。这些 episode 从后续训练 replay、dev 和 final cohort 中排除。

每个状态使用同一模拟器快照、goal 和初始噪声，比较：

- 原始 actor action；
- 负 latent-cost gradient；
- 等幅正梯度；
- 等幅随机方向。

扰动在归一化动作空间执行，RMS 为 `0.01 / 0.03 / 0.10`，随后投影到动作边界。每个候选完整执行 5 个 action blocks，记录预测 latent cost、真实物理终点 cost、success、动作偏移和 latent error。

不设置自动通过/失败。报告每个任务的 paired improvement rate、平均真实 cost 差值和 bootstrap 区间，由用户判断是否进入 guidance 采集。

### E2：生成后优化与生成中 guidance

统一使用：

- 10 个 Stage-A flow steps；
- CEM：`samples=300, topk=30, iterations=30, var_scale=1.0`；
- 相同 checkpoint、起点、goal 和初始噪声；
- dev cohort：每任务 50 episodes。

比较：

- A：原始 Stage A，单候选；
- A+B：64 候选后由 Stage B 选优；
- Post-opt：生成完成后做 5 次 latent-cost 梯度更新；
- Guided-flow：最后 5 个 flow steps 启用 guidance；
- B+CEM；
- A+CEM。

Guided-flow 采用 motion-planning diffusion 的 prior-mean guidance 结构：先得到 Euler prior proposal，再对 proposal 做 cost-gradient 内循环、限制相对初始 proposal 的最大位移，最后进入下一次 flow update。参考：[Motion Planning Diffusion](https://arxiv.org/abs/2412.19948)。

本仓库的 flow 版定义为：

```text
dt = 1 / K
s_k = k / K
v_k = v_theta(x_k, s_k)
mu_0 = x_k + dt * v_k

for inner_step in 1..5:
    clean = clean_estimate(mu, s_k + dt)
    E = MSE(StageB(z_t, clean, clean_timestep=1), z_g)
    g = -grad_mu(E)
    mu = mu + eta * normalize_per_sample_RMS(g)
    mu = clip_displacement(mu, mu_0, max_rms_offset=0.20)

x_{k+1} = detach(mu)
```

其中：

- 梯度完整穿过 clean action estimate 和 Stage B；
- Stage B 始终接收 clean action estimate，并使用 clean timestep；
- 每个 inner step 重新计算梯度；
- guidance 只启用最后 5 个生成步骤；
- 零梯度跳过；
- 非有限梯度记录为失败，不静默切换方法；
- 单候选结果用于机制比较；
- 最佳 guidance 设置另做 64 候选+B 对照；
- Post-opt 使用相同的 5 次 backward、相同步长集合和 `0.20` 位移上限。

E1/E2 结束后冻结采集器：

- 用户判断 guidance 有效：使用用户选定的 guided-flow 或 Post-opt；
- 用户判断 guidance 无效：使用未引导 A+B；
- 不根据 final cohort 结果反向选择采集器。

## 阶段二：固定 replay，分离监督信号

### 固定数据

每个任务先采集一份固定 replay，共 20,000 个原始环境步：

- 16,000：冻结采集器的连续交互；
- 4,000：grounded candidate groups。

每个 grounded group 固定包含：

- 同一 simulator snapshot、起点、goal 和执行长度；
- 1 条 actor 基准动作；
- 3 条 RMS=0.03 的独立随机扰动动作；
- 所有候选完整执行 5 个 action blocks；
- 只保存真实执行动作及真实后继。

Replay 必须记录：

- 实际 goal；
- 起点和终点物理 cost；
- success 标记；
- episode/reset 边界；
- snapshot/group ID；
- 执行长度；
- continuous/grounded source；
- collector 和 checkpoint 标识。

所有 T1–T6 先使用同一份完整 replay，再开始训练；训练不能反过来改变固定数据。

### 训练矩阵

所有训练臂从同一 legacy E5 checkpoint、同一初始 optimizer 状态开始。T0 不更新参数。

每个臂首轮最多执行 200 次 optimizer update。200 update 是早期信号筛选预算，不作为收敛结论：

- AdamW；
- `lr=1e-5`；
- `weight_decay=1e-3`；
- `betas=(0.9, 0.999)`；
- `eps=1e-8`；
- gradient clip `1.0`；
- 无 scheduler；
- encoder/projector 始终冻结。

每次更新只执行一次 backward 和一次 optimizer step：

```text
L = L_B
  + lambda_rank * L_rank
  + lambda_offA * L_offA
  + lambda_hindsight * L_hindsight
  + lambda_distill * L_distill
```

所有逻辑 batch 独立取 mean。

| 臂 | B-MSE batch | 新增监督 |
|---|---|---|
| T1 | 32 offline anchor + 32 offline replacement | 无 |
| T2 | 32 offline + 24 continuous + 8 grounded | 无 |
| T3 | 与 T2 完全相同 | grounded ranking |
| T4 | 与 T2 完全相同 | offline Stage-A flow loss |
| T5 | 与 T2 完全相同 | offline-A + online hindsight-A |
| T6 | 与 T2 完全相同 | offline-A + online planner distillation-A |

共享的 batch manifest 预先生成并保存，保证 T1–T6 的对照可复现。

### 各监督信号

Ranking：

- 只使用同一 group 内候选；
- success 候选优于 failure 候选；
- success 相同时，真实终点物理 cost 更低者优先；
- 完全并列跳过；
- 每次最多 32 对；
- 不跨起点、goal 或执行长度；
- 使用现有 Stage-B latent cost 的 pairwise softplus loss；
- 无有效 pair 时 loss 为零并记录数量。

Hindsight：

- 每个完整 online 窗口均可使用，包括失败和低位移窗口；
- 使用真实执行终点 latent 作为新 goal；
- 动作标签保持真实执行动作；
- 使用原 Stage-A flow-matching loss；
- 不使用 TD、reward 或额外加权。

Planner distillation：

- 保留原始 goal；
- 仅使用完整真实执行窗口；
- eligible 条件为原目标成功，或终点真实物理 cost 小于起点 cost；
- 不使用未执行动作后缀；
- eligible 样本不足时有放回抽样；
- eligible 为空时该 loss 为零，不重标定其他 loss。

辅助 loss 权重只在初始固定 calibration manifest 上校准一次：

\[
\lambda_k =
0.1
\frac{\|\nabla_{P_{shared}}L_B\|_2}
{\|\nabla_{P_{shared}}L_k\|_2}
\]

其中 `P_shared` 是 Stage A/B 共同经过的、排除 encoder/projector 后的可训练参数集合。

校准要求：

- FP32；
- 不启用 AMP scaling；
- 不进行 weight decay；
- 不进行 gradient clipping；
- `lambda_offA` 在 T4/T5/T6 中共用；
- ranking、hindsight、distillation 分别校准；
- 后续 update、seed 和阶段三复用冻结权重；
- 梯度为零或非有限时记录 `calibration_invalid`，该辅助 loss 权重置零，不临时手调。

## 阶段三：closed-loop online train

阶段二完成后，将结果按任务分别报告给用户。T3/T5/T6 不自动丢弃；由用户决定哪些候选进入阶段三。

阶段三的候选对照固定为：

- 冻结模型；
- Offline-only；
- Online B-MSE；
- Online B-MSE + ranking；
- Online B-MSE + offline-A；
- Online B-MSE + offline-A + hindsight-A；
- Online B-MSE + offline-A + distillation-A。

所有臂从原始 E5 checkpoint 开始，不从阶段二 checkpoint 继续。

采集协议：

- 每 100 个环境步执行 1 次 optimizer update；
- 80% continuous、20% grounded 是环境采集预算，不是 loss 权重；
- 每 500 个原始环境步采用 `4×100 continuous + 1×100 grounded`；
- 每个 grounded group 完整收集后才能用于对应更新；
- 轨迹不足完整窗口时先积累，不伪造样本；
- 跳过 update 时记录原因，并让匹配 offline 对照使用相同有效 update 数；
- 每个臂使用自己的当前模型、replay 和 RNG；“同一采集器”只表示相同采集算法和超参数；
- 总预算首轮为 20,000 环境步、目标 200 个有效 optimizer update。

R4-AB 只作为用户决定后的模型起点复验：

- 仅 Reacher、Push-T；
- 沿用冻结的 guidance、loss 权重、采样和评测配置；
- 不使用 D/E；
- 不自动启动。

## Success-rate 曲线与评测

曲线横轴统一为 `optimizer_update_step`，不再使用环境步作为主横轴。

首轮 200 update 期间在以下点生成 checkpoint 并评测：

```text
0, 10, 20, ..., 200
```

每个曲线点：

- checkpoint 保存完成后再评测；
- 使用同一固定 `round3_revised` final cohort；
- 每任务 200 episodes；
- 实际执行为 4×50 batch；
- 评测交互不计入训练预算；
- y 轴为该 checkpoint 的 episode success rate；
- 不表示单个环境 transition 的即时 success；
- `environment_steps` 作为辅助字段记录，但不作为曲线主横轴。

曲线文件至少包含：

- `optimizer_update_step`；
- `environment_steps`；
- `success_rate`；
- `success_rate_percent`；
- `episodes`；
- checkpoint hash；
- cohort hash；
- status；
- 实际有效 update 数和跳过原因。

最终 u=200 点作为首轮主结果。所有任务分别报告，不用跨任务平均值作为门槛。配对 episode 差值和 bootstrap 区间作为辅助统计；所有门槛只记录，不自动推进。

200 update 不等同于收敛。若曲线在 u=200 仍明显上升，由用户决定是否延长：

- 固定 replay 延长：测试同一数据上的优化耐久性；
- closed-loop 延长：保持 100 环境步/update，500 update 约需 50k 环境步，1000 update 约需 100k 环境步；
- 延长实验不改变中途 loss 权重、采样器或评测配置。

## 接口、验证与交付

主要实现范围：

- Stage-A sampler 增加 `none / post_opt / guided_flow`，显式接收初始噪声、flow steps、guidance steps、步长和 offset cap；
- `none` 模式严格复现原采样路径；
- replay 增加 source、group、真实 cost、success、goal、checkpoint 和 collector metadata；
- 训练接口提供 B-MSE、ranking、offline-A、hindsight-A、distillation-A 独立开关；
- 每次 update 记录真实数据来源、loss 权重、梯度范数和有效样本数；
- 新建独立 manifest、checkpoint、curve 和 decision-record 输出目录；
- 清理旧计划文件开头重复的标题和 `Proposed Plan` 残留文本。

必须通过的检查：

- guidance 梯度有限差分和符号检查；
- `none` 采样路径回归一致；
- Stage B 只接收 clean action estimate；
- guidance 不更新模型参数；
- 多候选梯度互不串扰；
- offset clipping 和动作边界投影正确；
- replay 不跨 episode/reset；
- ranking 不跨 group、goal 或执行长度；
- hindsight 和 distillation 不使用未执行后缀；
- frozen encoder/projector 参数不变化；
- T1–T6 的 batch 来源和计数符合 manifest；
- 空 grounded/eligible pool 不伪造样本；
- checkpoint 恢复保留 optimizer、RNG、replay cursor、累计环境步和 optimizer update step；
- 曲线点使用 final 200-episode cohort；
- 所有 GPU/EGL 命令显式限制在 GPU0–3，禁止 GPU4–7。

最终交付：

- 完整实验 manifest；
- E1 梯度诊断结果；
- E2 guidance/Post-opt/A+B 对照；
- T0–T6 固定 replay 训练表；
- closed-loop training 曲线；
- 每个 checkpoint 的 final 200-episode success rate；
- latent error、ranking、hindsight、distillation 和梯度诊断；
- 每任务配对差值与 bootstrap 区间；
- 用户决定的阶段推进记录；
- 所有正负结果及其适用范围。

## 本轮执行记录

E1/E2 前置诊断已完成，实际数值、采集预算、artifact 路径和下一轮待决策事项
记录在 [Round3 Online Supervision 实验报告](../report/round3/online_supervision/round3_online_supervision_experiment_report.md)。
用户已确认 Reacher 使用 `post_opt`、Push-T 使用 `guided_flow`，下一步启动 fixed
replay、T0–T6 和 200 optimizer-update success-rate 曲线。
