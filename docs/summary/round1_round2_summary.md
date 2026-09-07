# Fast-LeWAM 项目实验总览与 RoboTwin 迁移建议

> 汇总日期：2026-08-23
>
> 覆盖范围：LeWM 本地基线、Fast-LeWAM E0–E6、Stage-A 推理加速、Stage-B 扩容与单因素实验、CEM/simulator-grounded 诊断、serial one-step、planner-transition alignment、block-causal 与 terminal-only full attention。
>
> 当前状态：原四任务上的主要探索已收口；下一阶段转向 RoboTwin。仓库目前尚无 RoboTwin 数据、配置或适配代码。

## 1. 执行摘要

项目至今最重要的结论不是某个单一 trick 胜出，而是：**Fast-LeWAM 的收益和失败模式具有很强的任务依赖，当前没有任何结构或训练改动能在 Cube、Push-T、Reacher、TwoRoom 上稳定同向改善。**

- Cube 是 Stage-A 辅助监督和 actor warm-start 最有利的任务：E3 相对 E1 的 Stage-B 三种子平均提升 `+11.3 pp`，E6 actor warm-start 相对 E5 提升 `+32 pp`。但 E3 仍未超过本地 LeWM E0，Cube actor 又长期处于 100% 饱和且 shuffled-goal 仍约 40%，它不是可靠的唯一算法判据。
- Push-T 的本地 LeWM E0 已达 98%，多数 Fast Stage-B 改动受天花板和 planning 缺口共同限制。serial one-step 从 canonical parallel replay 的 86% 提升到 96%，但 shared-panel 排序指标没有一致改善；该收益真实，但机制混合。
- Reacher 的普通 Fast Stage-B 达 82%–84%，高于本地 LeWM E0 的 72%；但 actor warm-start、serial one-step 和放宽 attention 在 final 上均无稳定收益。其 grounded 结果反复指向 dynamics prediction、局部排序和 CEM/replan 联合作用。
- TwoRoom 的 E5/E6 Stage-B 已达 100%，而 E1–E4 缺省。它只能证明当前配置能解该任务，不能用于机制归因。
- 扩大模型、clean-action timestep、physical-time/type token、serial one-step、offline/streaming alignment、block-causal、terminal-only full attention 都未形成跨任务默认值。
- 系统优化比若干算法 trick 更确定：修复 CEM 重复视觉编码后，受控 30-iteration CEM 从 4.908 s 降到 0.109 s（约 45.1×）；Stage-A 延迟预处理在 40 个配对 episode 上动作路径与成功标签完全一致，已落地。BF16 和减少 Euler 步数则没有通过质量门槛。

因此，转向 RoboTwin 是合理的：它可以减少机器人形态、动作维度、相机定义和任务接口之间的混杂，更适合判断改动是否真正具备任务泛化性。但“同一机器人平台”不等于“任务动力学相同”；接触类型、双臂协调、目标精度和时域仍必须作为分层变量。

## 2. 证据口径与可比性边界

### 2.1 正式结果口径

- 第一、二轮正式主结果使用预定 epoch 10/final checkpoint、固定 seed 42 的 50 个 evaluation episodes；成功率最小步长为 2 个百分点。
- 主结论不根据测试成功率挑选中间 checkpoint。e2/e4/e6/e8 只用于观察训练动态。
- `mean ± SD` 是现有训练种子间的样本统计，不是置信区间。真正的训练重复数通常只有 1–3。
- 不跨任务平均 raw physical cost；grounded 指标只在同任务、同状态、同候选 panel 内比较。
- 10/20-episode pilot 是筛选证据，不能与 50-episode正式结果混为一谈。pilot 中的 baseline 与 candidate 使用同一嵌套 cohort，只有配对差值有意义。

### 2.2 本地 LeWM 不是严格论文复现

[LeWorldModel 审计](lewm_paper_repository_audit.md)确认本地四个发布权重与作者 Hugging Face 文件逐字节一致，结构和非有限值检查通过；但论文文字、官方代码、本地 eval YAML 和本地数据之间存在协议差异，特别是 Push-T 数据统计、TwoRoom history 以及部分 CEM/eval 参数。因此本报告中的 E0 只表示**本地同协议 LeWM 基线**，不能写成论文数字的严格复现。

### 2.3 结果等级

| 等级 | 定义 | 本报告用法 |
|---|---|---|
| 正式 | 50 episodes、final、`status=ok`，必要时 trace 0 mismatch | 支撑主要性能结论 |
| Grounded | 同状态、同候选、simulator rollout 和交叉打分验收通过 | 支撑 failure-mode 归因 |
| Pilot | 10/20 episodes 或单种子逐级 promotion | 决定是否扩大，不作最终主张 |
| 不完整/无效 | 中断、reference mismatch、缺 summary 或旧语义实现 | 仅记录，不纳入结论 |

## 3. 项目与模型实验框架

Fast-LeWAM 使用共享视觉 encoder/projector/SharedDiT，并提供三个功能路径：

- Stage A：goal-conditioned flow actor，从当前/目标 latent 生成 action chunk；
- Stage B：给定当前 latent 和 Candidate Action Sequence，预测 future/terminal latent，并由 CEM 根据 latent cost 排序；
- Stage C：早期统一结构路径。它没有得到与目标定义匹配的有效训练/推理闭环，后续只作为结构负对照。

第一轮正式消融定义为：

| 编号 | 设置 | 检验问题 |
|---|---|---|
| E0 | 本地 LeWM + CEM | 本地同协议 planner 基线 |
| E1 | Fast Stage-B only | parallel Stage-B 本身是否有效 |
| E2 | Fast Stage-A only | goal-conditioned actor 基线 |
| E3 | Stage A+B；Stage B 始终用专家动作 | action auxiliary supervision 是否帮助 WM |
| E4 | E3 + 50% 预测动作，detach | action exposure 是否帮助 WM |
| E5 | E4 + 跨-head梯度 | latent/action 两个 head 是否双向增强 |
| E6 | E5 checkpoint + actor warm-start CEM | actor proposal 是否改善 planner 搜索 |

## 4. 第一轮 E0–E6：结果与结论

完整轨迹、多种子统计、配对翻转和产物索引见[第一轮报告](plan/round1_experiment_report.md)。下表为 seed 3072 的 final 主结果；`A-shuf` 表示 shuffled-goal actor，`B` 表示 Stage-B planner。

| 任务 | E0 B | E1 B | E2 A/A-shuf | E3 A/A-shuf/B | E4 A/A-shuf/B | E5 A/A-shuf/B | E6 warm B |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cube | 78 | 60 | 100/40 | 100/42/74 | 100/40/74 | 100/42/66 | **98** |
| Push-T | **98** | 88 | 92/8 | 96/4/90 | 90/10/94 | 98/10/86 | 84 |
| Reacher | 72 | 82 | 68/8 | 76/4/82 | 70/10/80 | 76/10/84 | 68 |
| TwoRoom | 86 | 缺省 | 缺省 | 缺省 | 缺省 | 96/38/**100** | **100** |

### 4.1 Stage-A supervision 对 Stage-B

| 任务 | Action Block 维度 | E3−E1 Stage-B | 结论 |
|---|---:|---:|---|
| Cube | 25 | `+11.3 ± 6.4 pp`，3/3 seeds 为正 | 当前最稳定的单向正迁移 |
| Push-T | 10 | `−2.7 ± 4.2 pp`，1/3 seeds 为正 | 明确不能复现 Cube 结论 |
| Reacher | 10 | `0 pp`，单 seed | 无可见增益 |
| TwoRoom | 10 | 缺省 | 无法判断 |

这组结果支持“高维 Action Block 可能更需要结构化动作表示”的相关性假设，但不能作因果归因：只有 Cube 一个 25 维任务，动作维度与任务先验、接触动力学、数据质量和难度完全混杂。

### 4.2 跨-head梯度

Push-T 的 E4→E5 多种子结果最清楚：Stage-A 从 `91.3 ± 1.2` 提升到 `96.0 ± 2.0`，三个种子同向；Stage-B 从 `88.7 ± 5.0` 到 `88.0 ± 2.0`，差值方向不一致。结论是：**World Model supervision 对 actor 的帮助比 actor/latent 跨-head梯度对 planner 的帮助稳定得多。**

### 4.3 Goal conditioning 与任务先验

Push-T/Reacher 的 shuffled-goal 成功率通常只有 4%–10%，goal conditioning 明确生效。Cube 仍有约 40% 的 shuffled-goal 成功率，TwoRoom E5 为 38%，说明任务先验和易解轨迹占比较高；它们的高 actor 成功率不能等同于强目标理解。

### 4.4 Actor warm-start CEM

| 任务 | E5 B | E6 warm B | 差值 | 配对解释 |
|---|---:|---:|---:|---|
| Cube | 66 | **98** | **+32 pp** | 16 个不一致 episode 全部改善 |
| Push-T | 86 | 84 | −2 pp | 仅 1/50 由成功翻为失败 |
| Reacher | **84** | 68 | **−16 pp** | 14 回落、6 改善；20 个不一致 |
| TwoRoom | 100 | 100 | 0 | 天花板 |

Reacher 的 14 个 E5 成功/E6 失败 episode 中，纯 actor 在 12 个上成功。因此失败不是“proposal 没有好候选”，而是 planner 会把部分可成功初始化优化坏。

### 4.5 早期 Stage-C 负对照

早期 Stage-C 快照在 Cube/Push-T/Reacher/TwoRoom 上分别为 `40/6/6/60%`，平均 28%。这些 run 早于统一 E0–E6 协议，不能用于正式横向比较，只支持停止当前 Stage-C 结构方向。

## 5. 工程效率实验

### 5.1 参数量与 Stage-B 重复视觉编码

- 第一轮 Fast-LeWAM 约 10.5M 参数，LeWM 约 18.03M，Fast 少约 41.8%。Stage A/B 是同一 checkpoint 的不同 mode，不是两个模型参数相加。
- 旧 Fast Stage-B 在 CEM candidate 维重复编码相同 current/goal 图像，遮蔽了 parallel latent prediction 的收益。
- 修复后，同 context 下 current/goal 只编码一次并缓存；candidate-specific observation 保留原语义。
- Cube E5 epoch-10 的受控测试中，单次 cost 从 `614.256 ms / 600 images` 降到 `16.796 ms / 2 images`；同 context 再调用为 `3.268 ms / 0 images`，最大 cost 误差 `7.15e-7`。完整 30-iteration CEM 从 4.908 s 降至 0.109 s，约 45.1×。

旧的四任务端到端 wall-clock 记录包含重复编码、环境仿真、视频、提前终止和机器负载，不能继续用来代表修复后的架构速度。

### 5.2 Stage-A 加速探索

完整结果见[Stage-A 推理加速报告](plan/stage_a_inference_speed_exploration.md)。

| 方向 | 受控结果 | 任务结果 | 决策 |
|---|---|---|---|
| BF16 10-step | 比 FP32 慢约 23%–30% | 有非零输出误差 | 不采用 |
| Euler 10→5/2/1 | sampler 约 2×/5×/9.7× | Reacher 10-episode success 90→80/70/70 | 不采用；最多只值得补 5-step 正式非劣实验 |
| 延迟图像预处理 | 移除 action buffer 非空时的无用处理 | 四任务共 40 个 episode 逐项完全一致 | 已落地 Stage A/C |

延迟预处理的历史端到端观察为 15.35×–32.83×，但受到共享 CPU/GPU 负载严重影响，只能证明预处理是主要系统瓶颈，不能作为稳定部署倍率。

## 6. 第二轮 Stage-B 结构与训练目标探索

完整设置与训练轨迹见[第二轮报告](plan/round2_experiment_report.md)。

### 6.1 同量级扩容：192/768 → 384/1024

| 任务 | E1-192 B | E1-384 B | E3-192 B | E3-384 B | LeWM E0 |
|---|---:|---:|---:|---:|---:|
| Push-T | 88 | 84 | 90 | 92 | **98** |
| Reacher | 82 | 82 | 82 | **84** | 72 |

非视觉参数约从 4.62M 增到 14.56M，训练显存峰值约 21 GiB。E1 扩容在 Push-T 退化 4 pp、Reacher 持平；E3 只在两任务各提高 2 pp。**容量不足不是纯 Stage-B 缺口的主因，继续宽化的收益/成本比很低。**

### 6.2 §3.2 clean-action timestep

Reacher E4 对照轨迹为 `20→84→76→86→80`，clean-action timestep 为 `40→50→66→84→78`。它只改善 e2，final 反而低 2 pp，因此不采用。

Cube 对应 run 只保存到 epoch 8，没有 epoch-10 checkpoint/eval/grounding，属于中断实验，不能补写结果或用于跨任务判断。

### 6.3 §4.1 learned physical-time/type token

| 任务 | 原 E3 B final | §4.1 B final | 中期现象 | Actor/A-shuf final |
|---|---:|---:|---|---:|
| Reacher | 82 | 74 | e4 40→80，e8 达 88，但 final 回落 | 74/6（原 76/4） |
| Push-T | 90 | 90 | e2/e6/e8 更快，final 持平 | 92/14（原 96/4） |

物理时间/type encoding 可能改善早期优化，但没有 final 收益；Push-T shuffled-goal 从 4% 升到 14%，goal 约束反而变弱。因此它只保留为实验开关，不设默认，也没有继续 §4.2 sinusoidal sweep。

### 6.4 离线 validation 不能选择 planner checkpoint

三个 e8→e10 回落 run 中，predicted terminal MSE、preference accuracy 或 top-1 validation 指标仍在改善，而环境 success 下降。结论是：平均 latent MSE 和 expert-vs-negative 排序不足以替代真实 CEM 局部排序；不能据此 early-stop。

## 7. e8/e10 simulator-grounded 配对诊断

[诊断 manifest](../config/diagnostics/fast_lewam_epoch_pair_ranking.yaml)对三个已完成 pair 固定 38 个 slots，trace 全部严格复现 reference success vector；同一 simulator state 上合并两 checkpoint 候选、交叉打分，simulator rollout 只执行一次。全局 [summary](../outputs/diagnostics/fast_lewam_epoch_pair_ranking/0812/summary.json) 为 `status=ok`。

### 7.1 总体 grounded 差值（e10−e8）

| Pair | slots/panels | Δ predicted↔physical Spearman | Δ top-1 success | Δ physical regret | 主要变化 |
|---|---:|---:|---:|---:|---|
| Reacher §3.2 | 15/126 | −0.0080 | −5.56 pp | +0.0057 | ranking failure 3→11 |
| Reacher §4.1 | 17/135 | +0.0012 | 0 | +0.0012 | ranking 3→12；replan regression 2→7 |
| Push-T §4.1 | 6/36 | −0.0265 | −5.56 pp | +8.9542 | coverage 2→4；ranking 1→4 |

Reacher 的 true-terminal-latent↔physical Spearman 约 0.96，说明 latent metric 本身较好地表示物理目标；问题更靠近 predicted dynamics 和顶端排序。Push-T 同指标约 0.26，latent metric 与物理目标的对齐明显更弱，不能沿用 Reacher 的单一修复策略。

### 7.2 归因结论

- 三个 pair 的完整候选都经 oracle/coverage 检查，不是所有失败都能归为 coverage。
- 后期 checkpoint 的离线 MSE 改善没有转化为 top-1 physical 选择改善。
- Reacher §4.1 中 replan regression 明显增加；Push-T §4.1 的 regret 恶化幅度更大。
- 三个 pair 当时均触发 §3.4 serial one-step control，而不是直接扩大模型或继续位置编码 sweep。

## 8. §3.4 serial one-step control

serial 模型训练时 teacher-force `(z_i,a_i)→z_{i+1}`，评估时递归 rollout；parallel/serial 均使用同一 192/768 backbone、seed 3072、canonical CEM 和 50 episodes。[诊断 manifest](../config/diagnostics/fast_lewam_serial_one_step.yaml)与最终 grounded summary 均通过严格 trace 门禁。

| 任务 | Parallel canonical replay | Serial final | 差值 | Grounded 状态 |
|---|---:|---:|---:|---|
| Reacher | 84 | 72 | **−12 pp** | 50 slots、436 scores、0 acceptance errors |
| Push-T | 86 | 96 | **+10 pp** | 50 slots、347 scores、0 acceptance errors |

### 8.1 Reacher

serial 在 regression、improvement-control、stable-success 和 stable-failure 四类中都降低 predicted↔physical Spearman，并提高 regret 与 predicted terminal latent MSE。ranking failure 从 8 增至 14，dynamics error 从 3 增至 9；pair 决策为 `causal_prefix_not_supported`。

### 8.2 Push-T

环境成功率提升 10 pp，失败归因中的 coverage、dynamics、CEM exploitation 和 replan regression 都减少；但占 41/50 的 stable-success 组反而出现轻微 Spearman/recall 下降和 regret 上升。pair 决策为 `serial_control_mixed`。

因此 serial one-step 不是全局默认：它证明 parallel causal-prefix 可能是某些任务的瓶颈，但也可能因 recursive error accumulation 和排序退化伤害另一些任务。

## 9. Planner-transition alignment 与 action-token pilots

这组 pilot 使用 20-episode Level 1、50-episode Level 2 和 6/18 个 grounded slots。offline alignment 在固定 replay 上适配；streaming online 在 slot group 间保留模型/优化器状态，使前一组更新真实影响后一组收集与打分。这里的 “online” 是受控 simulator-transition adaptation，不是完整部署式在线 RL。

### 9.1 有效 summary

`grounded regret reduction` 正值表示 regret 下降；所有 success delta 都是同 cohort 的百分点差。

| seed | 任务/方向 | Level | baseline→pilot | Δ Spearman | grounded regret reduction | 决策 |
|---:|---|---:|---:|---:|---:|---|
| 42 | Push-T offline | 1 | 95→95 | +0.0486 | −4.1% | stop |
| 43 | Push-T offline | 1 | 95→95 | +0.0928 | −9.9% | stop |
| 42 | Push-T streaming | 1 | 95→95 | +0.0822 | +12.9% | promote |
| 43 | Push-T streaming | 1 | 95→95 | +0.0805 | +48.4% | promote |
| 42 | Push-T streaming | 2 | 86→78 | +0.0904 | +28.1% | complete |
| 42 | Reacher offline | 1 | 85→75 | +0.1550 | +48.2% | stop |
| 43 | Reacher streaming | 1 | 85→75 | +0.1300 | +51.8% | stop |

最关键的反例是 Push-T streaming Level 2：局部 grounded Spearman 和 regret 都改善，但 50-episode success 下降 8 pp。Reacher 两个方向也能显著改善局部 ranking/regret，却同时下降 10 pp。**优化固定 panel 的局部排序不等于改善闭环 planner；遗忘、状态分布变化、第二次 replan 和被诊断 panel 之外的排序都会抵消收益。**

### 9.2 Cube 25 temporal action tokens

该实验把对照的 `frameskip=5, horizon=5, action_dim=25` 改为 `frameskip=1, horizon=25, action_dim=5`，保持 25 个物理步跨度和等 update budget。epoch-3 的 20-episode初步 eval 为 control 95%、variant 85%，但 grounding 因 Cube info 缺少 `privileged_block_0_pos` 触发 `KeyError`，没有生成有效 summary；seed43 也没有完成。因此该方向当前是**不完整且初步不利**，不能作正式 representation 结论。

旧的 `*.invalid_semantics_20260815` 目录来自 streaming feedback/配对协议修复前的实现，全部排除，不与上述有效 summary 混用。

## 10. Stage-B attention pilots

attention 实验只训练 Stage-B，严格保持数据、容量、CEM 和 baseline cohort；逐级使用 epoch 1/3/10 与 10/20/50 episodes。两种变体为：

- block-causal：每个 `(a_t,q_{t+1})` transition block 内双向可见，只能看当前和过去 block；
- terminal-only full：token 为 `[z0,a0,…,aH−1,qH]`，只监督一个 terminal latent，所有 token 全注意力。

### 10.1 Reacher

| 变体 | e1（10 ep） | e3（20 ep） | e10（50 ep） | e10 grounded Δ Spearman | e10 regret reduction |
|---|---:|---:|---:|---:|---:|
| block-causal | 20→80 | 55→75（+20 pp） | 84→82（−2 pp） | −0.1739 | −45.1% |
| terminal-only full | 20→90 | 55→80（+25 pp） | 84→62（−22 pp） | −0.0157 | −16.1% |

两个变体都显著加快 Reacher 的早期学习，却在 final 回落；terminal-only full 的退化尤其明显。block-causal e3 的 grounded Spearman 只提高 0.0397、regret 仅改善约 1.0%，与 +20 pp success 不成比例；早期成功率不能替代 final 机制证据。

### 10.2 Push-T

| 变体 | e1（10 ep） | e3（20 ep） | Grounding/决策 |
|---|---:|---:|---|
| block-causal | 60→60 | 80→75（−5 pp） | Δ Spearman −0.2053，regret 恶化 16.3%；stop |
| terminal-only full | 60→10 | 80→60（−20 pp） | 50-episode trace 出现 baseline slot 27、candidate slot 5 mismatch，fail-closed；负向 eval 已足以停止 |

### 10.3 Attention 结论

放宽 attention 不是统一解：它可能改善早期优化，但没有改善两任务 final。terminal-only full 丢失 prefix supervision，并把每个 action token 的表示改为全序列条件；这对 terminal cost 在定义上并非信息泄漏，因为完整候选动作在规划时已知，但当前结果显示这种归纳偏置的稳定性风险大于收益。block-causal 可以保留为 RoboTwin 上的低优先级单因素开关，但 strict causal 仍应是默认对照。

## 11. 不完整、无效与缺省实验总账

| 项目 | 当前状态 | 报告处理 |
|---|---|---|
| TwoRoom E1–E4 | 未运行 | 不参与任何消融均值或机制结论 |
| Cube §3.2 clean-action timestep | 训练只到 epoch 8 | 无 final，不纳入 §3.2 结论 |
| Cube 25 temporal action tokens | 两个 epoch-3 eval 完成，grounding `KeyError` | 仅记录 95→85 初步信号，不验收 |
| Push-T terminal-only full e3 grounding | reference trace 2 slots mismatch | fail-closed；不使用 grounded 指标 |
| parallel pilot `invalid_semantics` 目录 | 旧 online/offline 语义与配对实现 | 完全排除 |
| 早期 0714–0722 开发快照 | 配置/评测协议未统一 | 只作开发历史；正式结果从 E0–E6 开始 |
| 历史旧 reference/cache | canonical replay 或 manifest identity 不一致 | 保留或隔离，不覆盖新验收结果 |

## 12. 基础设施与实验治理结论

这些工程事件不属于模型效果，但直接决定结果是否可信：

- `CUDA_VISIBLE_DEVICES` 不能单独约束 dm_control/MuJoCo EGL 的物理卡选择；现已同时绑定 `MUJOCO_EGL_DEVICE_ID`，所有 runner 只允许物理 GPU0–3。
- 发生过 NVIDIA driver/library mismatch，NVML 失效并导致实验停止；服务器重启后恢复。该事件不是训练收敛或 OOM 结论。
- trace 必须逐 episode 复现 reference success vector；任何 mismatch 在 grounding 前 fail-closed。
- cache identity 现绑定 manifest、checkpoint、config、dataset stat、代码、prefix/state 和 candidate hash；产物原子写入，避免重启后混用旧 cache。
- Reacher/Push-T full grounding 曾各产生约 86/90 GiB 可重建 simulator cache。删除 cache 不影响最终 scores/summary，但会失去快速续跑点。
- 训练时 epoch eval 曾增加约 30%–55% wall-clock；后续默认关闭，训练结束后离线评估。

## 13. 跨任务综合结论

### 13.1 哪些结论最可靠

1. **不存在已验证的 universal tuning trick。** 所有重要结构改动至少在一个任务上无收益或反向。
2. **任务差异不只是动作维度。** Cube 的 25 维 Action Block 与其他任务的 10 维确实是混杂因素，但 Push-T/Reacher 同为 10 维，serial one-step 仍分别 `+10/−12 pp`，说明接触动力学、状态表示、成功阈值和 replan 结构同样关键。
3. **Stage-B 的主要瓶颈是闭环排序，不是单一 latent MSE。** 多组实验都出现 offline/grounded 局部指标改善而 end-to-end success 下降。
4. **候选覆盖与排序必须分开。** actor warm-start 可显著改善 coverage，但 Reacher 中 planner 会把好初始化优化坏。
5. **CEM/replan 是模型的一部分。** 最后一轮 elite mean、缺少 best-ever 保留、第二次 replan 状态漂移，都可能把 world-model 小误差放大成任务失败。
6. **final checkpoint 口径不可放弃。** physical-time、attention、E1-384 等多次出现中期峰值高于 final；按测试峰值选 checkpoint 会系统性夸大收益。
7. **确定性的系统优化优先级高。** latent context cache 和延迟预处理带来可验证收益，不引入新的任务特异超参数。

### 13.2 原四任务分别教会了什么

| 任务 | 最有价值的信号 | 主要限制 |
|---|---|---|
| Cube | Stage-A supervision、warm-start 均强正向 | actor 饱和、任务先验强、唯一高维动作，容易把任务特性误写成方法增益 |
| Push-T | serial one-step 正向；LeWM E0 很强 | latent metric↔physical 对齐弱、接触任务、天花板高；alignment/attention 未改善闭环 |
| Reacher | Fast parallel Stage-B 高于本地 LeWM；grounded 诊断最清晰 | 局部 ranking/dynamics/replan 脆弱；warm-start、serial、terminal full 均可能退化 |
| TwoRoom | E5/E6 达 100% | E1–E4 缺失且天花板，无法归因 |

### 13.3 当前应停止或降级的方向

- 停止无目标的宽度/深度扩容；
- 不把 serial one-step、physical-time/type、clean-action timestep 或 actor warm-start 设为全局默认；
- terminal-only full attention 不进入下一轮首批实验；
- block-causal 只保留为小规模、预注册的单因素对照；
- 不继续用固定 panel alignment 指标单独决定 promotion；
- 不补齐 TwoRoom 矩阵来追求表格完整性，除非它能回答 RoboTwin 迁移所需的具体机制问题。

### 13.4 应保留的资产

- strict-causal Fast Stage-B 与本地 LeWM 的公平对照协议；
- Stage A/A-shuf/B 分支和 paired success vector；
- shared-panel simulator grounding、coverage/ranking/dynamics/latent-metric/CEM/replan 归因；
- trace mismatch 门禁、cache identity、原子写入和 GPU0–3 guard；
- latent context cache、Stage-A 延迟预处理；
- 10→20→50 episodes 的分级筛选，但最终关键结论仍需多训练种子。

## 14. 转向 RoboTwin 的专业判断

### 14.1 为什么值得转

原任务同时改变机器人形态、动作维度、图像分布、接触性质、目标定义和成功判据。即使一个 trick 在四任务上表现不一致，也无法判断是方法缺陷还是任务混杂。RoboTwin 若能提供统一双臂机器人、统一 action/proprio schema、相近相机和统一数据生成接口，就能显著降低这些混杂，更适合研究“同一个算法改动是否跨任务有效”。

但仍需警惕：同一平台上的抓取、插入、倾倒、工具使用和长时序双臂协作，动力学与误差容忍度仍可能完全不同。RoboTwin 应用于**控制变量更好的任务族实验**，而不是假定所有任务同质。

### 14.2 迁移前必须先建立的新基线

仓库当前没有 RoboTwin 适配代码。首要工作不是把旧 trick 全部移植，而是建立以下可审计接口：

1. 固定 observation 字段、相机顺序、图像尺寸、proprio/action 归一化和 task condition；
2. 明确单步 action 语义、控制频率、frameskip、Action Block 和 25-step physical horizon 的对应关系；
3. 固定 reset/initial-state cohort、goal 定义、success predicate 和最大 episode budget；
4. 验证 dataset window 的 `(z_t,a_t,z_{t+1})` 对齐，特别是双臂 action 的左右臂排列与 gripper 维；
5. 为 simulator state snapshot/restore 和 privileged physical metric 建立任务适配层；
6. 将 dataset stat、任务资产、代码和 checkpoint 全部纳入 cache identity；
7. 所有 GPU 命令继续显式限制在 GPU0–3，并保持一张卡一个首轮进程。

### 14.3 建议的首轮四进程矩阵

先选两个任务，而不是一次铺开完整 RoboTwin：

- Task A：短时域、目标容差较宽、接触较少，用于验证数据/控制链；
- Task B：接触丰富或需要双臂协调，用于暴露 world-model/planner 排序问题；
- 两者必须使用完全相同的 action dimension、相机布局和训练/eval 接口。

| GPU | 进程 | 目的 |
|---:|---|---|
| GPU0 | Task A 本地 LeWM（或同容量 serial WM）基线 | 建立任务 A planner 下界/参考 |
| GPU1 | Task A Fast Stage-B strict-causal E1 | 测 parallel Fast 的真实差值 |
| GPU2 | Task B 本地 LeWM（或同容量 serial WM）基线 | 建立任务 B 参考 |
| GPU3 | Task B Fast Stage-B strict-causal E1 | 检验差值是否跨同平台任务一致 |

若 LeWM 无法在不改变输入/action 语义的情况下直接适配，应先使用一个简单、可复现的 serial one-step WM 作为 planner control，同时保留独立 behavior-cloning actor；不能用不同 observation 或不同 planning budget 的 baseline 充数。

### 14.4 分级实验协议

- Level 0：每任务 2–3 episodes，只验证环境、shape、state restore、动作逆归一化和 GPU 映射；不报告性能。
- Level 1：固定 10 episodes、1 个训练 seed，筛掉崩溃、明显负向或吞吐不可接受的方向。
- Level 2：固定配对 20 episodes，并 ground 6 个预注册 slots；只有 success 不退化超过 5 pp，且至少 success/Spearman/regret 之一达到预注册门槛才扩大。
- Level 3：50 episodes、18 个 grounded slots；关键方法至少 3 个训练 seeds，并报告 paired flips、mean±SD 和 GPU-hours。
- 泛化验证：在两个开发任务确定同一超参数后，加入第三个 holdout 任务，**不重新调参**。只有 holdout 仍非退化，才能称任务泛化。

### 14.5 RoboTwin 上的实验优先级

1. **基线与接口正确性**：strict-causal Fast vs serial/LeWM，对齐物理 horizon 和 CEM budget。这是最高优先级。
2. **统一 action representation**：同一任务内比较 action block 与 temporal action tokens，终于能在不改变机器人/任务的前提下检验动作表示；先修复 Cube pilot 暴露的 physical-metric adapter 缺口。
3. **Block-causal 单因素**：只在两个任务各跑一个 pilot；如果仍只提高早期而 final 回落，立即停止。
4. **Planner robustness**：best-ever 保留、early-stop 和 replan protocol 应在 shared-panel 排序未退化但闭环失败时触发，而不是默认全开。
5. **Stage-A proposal**：只有 baseline grounding 显示 coverage failure 主导时才启用 actor warm-start；ranking 主导时不要先改 proposal。
6. **多任务共享模型**：在单任务 adapter 和 baseline 稳定后，再比较 per-task model 与 task-conditioned shared model。这才是 RoboTwin 对“任务族泛化”的核心价值。

terminal-only full attention、继续扩宽、physical-time/type 和 streaming alignment 不进入 RoboTwin 首轮；它们只有在新任务上的 grounded failure mode 与其设计目标一致时才恢复。

## 15. 最终决策

本项目在原四任务上的阶段性研究已经足以否定“继续堆统一 trick 会自然得到通用提升”的路线。更合理的下一步是：

1. 冻结当前 strict-causal、serial control、diagnostic 和效率修复作为旧任务基线；
2. 不再为补表而运行缺省 TwoRoom/Cube 项；
3. 为 RoboTwin 建立统一数据与 simulator adapter；
4. 用两个任务、四个 GPU0–3 进程先做 baseline matrix；
5. 以后每个改动先由 grounded failure mode 触发，再用同平台两任务和第三个 holdout 任务验证泛化。

这一路线把研究问题从“哪个 trick 在杂乱任务集合上偶然更高”改为“在受控任务族中，哪类 failure mode 应由哪类机制修复”，更有希望得到可复现、可解释、可迁移的结论。

## 16. 主要证据与产物索引

- [第一轮 E0–E6 报告](plan/round1_experiment_report.md)
- [第二轮 Stage-B 报告](plan/round2_experiment_report.md)
- [Stage-A 推理加速报告](plan/stage_a_inference_speed_exploration.md)
- [Fast-LeWAM 实现审计](fast_lewam_audit_report.md)
- [LeWorldModel 论文/权重/协议审计](lewm_paper_repository_audit.md)
- [e8/e10 pair-ranking manifest](../config/diagnostics/fast_lewam_epoch_pair_ranking.yaml)
- [e8/e10 global summary](../outputs/diagnostics/fast_lewam_epoch_pair_ranking/0812/summary.json)
- [serial one-step manifest](../config/diagnostics/fast_lewam_serial_one_step.yaml)
- [parallel/alignment pilot manifest](../config/experiments/fast_lewam_parallel_pilots.yaml)
- [seed43 replication manifest](../config/experiments/fast_lewam_streaming_replication_seed43.yaml)
- [attention pilot manifest](../config/experiments/fast_lewam_attention_pilots.yaml)
- [attention pilot outputs](../outputs/experiments/fast_lewam_attention_pilots/0820/)
- [planner-transition pilot outputs](../outputs/experiments/fast_lewam_parallel_pilots/)
