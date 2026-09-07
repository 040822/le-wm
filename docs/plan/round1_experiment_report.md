# Fast-LeWAM 第一轮 E0–E6 实验报告

> 汇总日期：2026-08-10；效率分析更新：2026-08-11
> 状态：第一轮主体实验完成；TwoRoom E1–E4 作为缺省项保留
> 主口径：预定 epoch 10 / final checkpoint，50 个固定评估 episode

## 1. 本轮目标与范围

本轮实验用于回答两个核心问题：

1. Stage-A action supervision 是否能增强 Stage-B world model planning？
2. Stage-B latent supervision 是否能增强 Stage-A actor，以及 actor 能否进一步帮助 CEM planner？

报告覆盖正式消融 E0–E6、已完成的多训练种子复核、E6 配对评估，以及 Push-T/Reacher 的 simulator-grounded 排序诊断。早期结构开发快照不并入正式主表，避免不同配置和评测协议混杂。

## 2. 实验定义

| 实验 | 训练 / 推理设置 | 主要问题 |
|---|---|---|
| E0 | 本地 LeWM 复现，Stage-B planning | 同协议 planner 基线 |
| E1 | Fast Stage-B only；真实动作、latent loss | parallel Stage-B 本身是否有效 |
| E2 | Fast Stage-A only；`lambda_latent=0` | 纯 goal-conditioned actor 基线 |
| E3 | Stage-A + Stage-B；Stage-B 始终使用专家动作 | action auxiliary supervision 是否增强 WM |
| E4 | E3 + 50% Stage-A 预测动作；`detach_clean_action=true` | exposure 是否增强 WM |
| E5 | E4 + 跨-head梯度；`detach_clean_action=false` | latent gradient 是否产生双向增益 |
| E6 | E5 checkpoint + Stage-A actor warm-start CEM | actor proposal 与 planner 是否协同；无需重训 |

## 3. 统一协议与统计边界

- 主结果均为 50 个评估 episode 的成功率，最小步长为 2 个百分点。
- 横向比较使用固定评估 seed、episode、起点、goal offset 和预算；配对分析只比较 cohort 完全一致的结果。
- 主结论使用 epoch 10 / final，不根据测试成功率选择中间 checkpoint。
- 默认训练种子为 `3072`。Cube 的 E1/E3 有 3 个训练种子；Push-T 的 E1/E3/E4/E5 有 3 个训练种子；其余主结果为单训练种子。
- `mean ± SD` 中的 SD 是现有训练种子间样本标准差，不是总体置信区间。
- E6 与 E5 使用相同 checkpoint、cohort 和 CEM budget，仅改变 CEM 初始均值。
- Reacher E1–E4 的 final JSON 均为 `status=ok`，epoch 10 checkpoint 和 `last.ckpt` 完整；日志无 traceback、OOM 或任务失败。训练结束后的 `asyncio socket.send()` warning 不影响已保存结果。

## 4. 完成度

| 任务 | E0 | E1 | E2 | E3 | E4 | E5 | E6 |
|---|---|---|---|---|---|---|---|
| Cube | 完成 | 完成，3 seeds | 完成 | 完成，3 seeds | 完成 | 完成 | 完成 |
| Push-T | 完成 | 完成，3 seeds | 完成 | 完成，3 seeds | 完成，3 seeds | 完成，3 seeds | 完成 |
| Reacher | 完成 | 完成 | 完成 | 完成 | 完成 | 完成 | 完成 |
| TwoRoom | 完成 | **缺省（未运行）** | **缺省（未运行）** | **缺省（未运行）** | **缺省（未运行）** | 完成 | 完成 |

TwoRoom 的缺省项表示没有实验数据，不表示失败或成功率为 0。其 E5 Stage-B 已达 100%，因此未完成项的主要价值是因果归因，而不是提升绝对成功率。

## 5. Seed 3072 完整结果

### 5.1 Final 主结果总表

`A` 为 Stage-A，`A-shuf` 为 shuffled-goal Stage-A，`B` 为随机初始化 CEM Stage-B；“不适用”表示该实验按定义没有该分支。

| 任务 | E0 B | E1 B | E2 A / A-shuf | E3 A / A-shuf / B | E4 A / A-shuf / B | E5 A / A-shuf / B | E6 warm B |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cube | 78 | 60 | 100 / 40 | 100 / 42 / 74 | 100 / 40 / 74 | 100 / 42 / 66 | **98** |
| Push-T | 98 | 88 | 92 / 8 | 96 / 4 / 90 | 90 / 10 / 94 | 98 / 10 / 86 | 84 |
| Reacher | 72 | 82 | 68 / 8 | 76 / 4 / 82 | 70 / 10 / 80 | 76 / 10 / 84 | 68 |
| TwoRoom | 86 | 缺省 | 缺省 | 缺省 | 缺省 | 96 / 38 / 100 | 100 |

### 5.2 Cube 训练轨迹

| 实验 | 分支 | e2 | e4 | e6 | e8 | final |
|---|---|---:|---:|---:|---:|---:|
| E0 | B | — | — | — | — | **78** |
| E1 | B | 50 | 60 | 58 | 60 | 60 |
| E2 | A | 98 | 100 | 100 | 100 | 100 |
| E2 | A-shuf | 44 | 42 | 44 | 40 | 40 |
| E3 | A | 100 | 100 | 100 | 100 | 100 |
| E3 | A-shuf | 42 | 42 | 40 | 42 | 42 |
| E3 | B | 68 | 74 | 74 | 70 | **74** |
| E4 | A | 98 | 100 | 100 | 100 | 100 |
| E4 | A-shuf | 42 | 42 | 40 | 40 | 40 |
| E4 | B | 74 | 76 | 74 | 74 | **74** |
| E5 | A | 100 | 100 | 100 | 100 | 100 |
| E5 | A-shuf | 42 | 42 | 42 | 42 | 42 |
| E5 | B | 72 | 72 | 72 | 74 | **66** |

Cube 的 E3 相对 E1 在 seed 3072 上提升 14 点，且多种子复核均为正；这是本轮最稳定的“Stage-A supervision 增强 Stage-B”信号。但 E3 三种子的 Stage-B 均没有超过 E0=78。E4 与 E3 final 相同，预测动作混合没有额外收益；E5 反而从 74 降至 66。

Stage-A 在 E2–E5 均为 100%，但 shuffled-goal 仍有 40–42%，说明 actor 使用了 goal，同时 Cube 存在较强任务先验和饱和效应。

### 5.3 Push-T 训练轨迹

| 实验 | 分支 | e2 | e4 | e6 | e8 | final |
|---|---|---:|---:|---:|---:|---:|
| E0 | B | — | — | — | — | **98** |
| E1 | B | 76 | 82 | 86 | 86 | 88 |
| E2 | A | 56 | 80 | 86 | 92 | 92 |
| E2 | A-shuf | 10 | 8 | 8 | 8 | 8 |
| E3 | A | 66 | 88 | 92 | 94 | 96 |
| E3 | A-shuf | 4 | 6 | 4 | 6 | 4 |
| E3 | B | 52 | 78 | 82 | 90 | **90** |
| E4 | A | 60 | 86 | 88 | 92 | 90 |
| E4 | A-shuf | 6 | 6 | 8 | 10 | 10 |
| E4 | B | 46 | 64 | 86 | 92 | **94** |
| E5 | A | 74 | 86 | 94 | 98 | 98 |
| E5 | A-shuf | 8 | 8 | 8 | 12 | 10 |
| E5 | B | 72 | 84 | 86 | 84 | **86** |

Push-T 的 E3 多种子 Stage-B 平均低于 E1 2.7 点，因此没有复现 Cube 的正迁移。E4/E5 的多种子结果显示，跨-head梯度对 Stage-A 的改善较稳定，但 Stage-B 平均几乎不变。所有 Fast Stage-B 配置仍低于本地 E0=98；当前主要缺口在 planning，而不是 actor。

Stage-A final 为 90–98%，shuffled-goal 仅为 4–10%，说明 Push-T actor 对 goal 的依赖很强。

### 5.4 Reacher 训练轨迹

| 实验 | 分支 | e2 | e4 | e6 | e8 | final |
|---|---|---:|---:|---:|---:|---:|
| E0 | B | — | — | — | — | **72** |
| E1 | B | 38 | 50 | 82 | 72 | **82** |
| E2 | A | 52 | 62 | 70 | 58 | **68** |
| E2 | A-shuf | 10 | 8 | 6 | 6 | **8** |
| E3 | A | 32 | 42 | 78 | 64 | **76** |
| E3 | A-shuf | 12 | 8 | 6 | 4 | **4** |
| E3 | B | 36 | 40 | 88 | 82 | **82** |
| E4 | A | 30 | 64 | 60 | 74 | **70** |
| E4 | A-shuf | 12 | 10 | 4 | 6 | **10** |
| E4 | B | 20 | 84 | 76 | 86 | **80** |
| E5 | A | — | — | — | — | 76 |
| E5 | A-shuf | — | — | — | — | 10 |
| E5 | B | — | — | — | — | **84** |

Reacher E1/E3/E4/E5 Stage-B final=`82/82/80/84`，均高于本地 E0=72，但目前只有一个训练种子。E3 与 E1 完全持平，因此无法把提升归因于 Stage-A auxiliary supervision；E4 相对 E3 为 −2，E5 相对 E4 为 +4，均为弱差异。

此前引用的论文 Reacher 数字为 86，但本地同协议 E0 为 72；本报告只使用本地 E0 做正式横向比较，论文数字仅作为外部参考。

相同 cohort 的 Stage-B 配对翻转为：E1→E3 退化 6、改善 6；E3→E4 退化 6、改善 5；E4→E5 退化 5、改善 7。结果说明相邻配置间存在大量双向变化，不能只根据 2–4 点 aggregate 差异作机制结论。

E3 在 e6 达到 88 后回落到 final 82，E4 在 e8 达到 86 后回落到 final 80。该现象支持继续坚持预定 final 口径，不能用测试成功率挑选中间 checkpoint。

### 5.5 TwoRoom 缺省项

| 实验 | Stage-A | A-shuf | Stage-B | 状态 |
|---|---:|---:|---:|---|
| E0 | 不适用 | 不适用 | 86 | 完成 |
| E1 | — | — | — | **缺省（未运行）** |
| E2 | — | — | — | **缺省（未运行）** |
| E3 | — | — | — | **缺省（未运行）** |
| E4 | — | — | — | **缺省（未运行）** |
| E5 | 96 | 38 | **100** | 完成 |
| E6 | 不适用 | 不适用 | **100** | 完成，无重训 |

TwoRoom E5 Stage-B 已达到 100%，E6 保持 100%，存在明显天花板效应。E1–E4 的缺省使 TwoRoom 无法参与本轮机制归因，但不影响确认当前 E5/E6 系统在该任务上的最终性能。

### 5.6 补充：Stage-C 结构快照

Stage-C 不属于 E0–E6 的公平主对照：它不接收 goal token，action token 又受 causal mask 限制，推理时也没有利用预测 latent 做候选选择。早期统一模型保存的四任务快照如下，仅作为结构负对照保留。

| 任务 | Stage-C final | 结果来源 |
|---|---:|---|
| Cube | 40 | `outputs/fast_lewam/cube/0715_goal/` |
| Push-T | 6 | `outputs/fast_lewam/pusht/0717_/` |
| Reacher | 6 | `outputs/fast_lewam/reacher/0717_/` |
| TwoRoom | 60 | `outputs/fast_lewam/tworoom/0717_/` |
| **平均** | **28.0** | — |

这些 run 早于正式 E0–E6 矩阵，不能用于推断 E3/E4/E5 的因果效果。结果只支持将当前 Stage-C 降级为负对照，不继续投入本轮主要算力。


## 6. 多训练种子结果

### 6.1 Cube：E1 vs E3 Stage-B

| seed | E1 | E3 | E3−E1 |
|---:|---:|---:|---:|
| 3072 | 60 | 74 | +14 |
| 3073 | 58 | 74 | +16~ |
| 3074 | 62 | 66 | +4 |
| **mean ± SD** | **60.0 ± 2.0** | **71.3 ± 4.6** | **+11.3 ± 6.4** |

三个配对差值均为正，支持 Cube 上存在稳定的单向正迁移；但 E3 每个 seed 仍低于 E0=78。

### 6.2 Push-T：E1 vs E3 Stage-B

| seed | E1 | E3 | E3−E1 |
|---:|---:|---:|---:|
| 3072 | 88 | 90 | +2 |
| 3073 | 90 | 86 | −4 |
| 3074 | 94 | 88 | −6 |
| **mean ± SD** | **90.7 ± 3.1** | **88.0 ± 2.0** | **−2.7 ± 4.2** |

三个差值一正两负，明确否定“Cube 的正迁移可以直接跨任务泛化”。

### 6.3 Push-T：E4 vs E5

| seed | E4 A | E5 A | E5−E4 A | E4 B | E5 B | E5−E4 B |
|---:|---:|---:|---:|---:|---:|---:|
| 3072 | 90 | 98 | +8 | 94 | 86 | −8 |
| 3073 | 92 | 94 | +2 | 88 | 90 | +2 |
| 3074 | 92 | 96 | +4 | 84 | 88 | +4 |
| **mean ± SD** | **91.3 ± 1.2** | **96.0 ± 2.0** | **+4.7 ± 3.1** | **88.7 ± 5.0** | **88.0 ± 2.0** | **−0.7 ± 6.4** |

跨-head梯度对 actor 的提升在三个 seed 上同向，而 planner 差值一负两正、均值接近 0。本轮关于“双向增强”的最准确表述是：World Model supervision 对 actor 有较稳定帮助，但 actor/latent 跨-head梯度没有稳定增强 planner。

## 7. E6 Actor warm-start CEM

### 7.1 成功率与配对翻转

| 任务 | E5 B | E6 warm B | 差值 | 都成功 | E5→E6 退化 | E5→E6 改善 | 都失败 | exact p | 耗时（秒） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cube | 66 | **98** | **+32** | 33 | 0 | 16 | 1 | <0.001 | 417.6 |
| Push-T | **86** | 84 | −2 | 42 | 1 | 0 | 7 | 1.000 | 317.3 |
| Reacher | **84** | 68 | −16 | 28 | 14 | 6 | 2 | 0.115 | 930.8 |
| TwoRoom | **100** | **100** | 0 | 50 | 0 | 0 | 0 | — | 1150.0 |

Cube 的 16 个不一致配对全部朝改善方向，warm-start 收益明确。Push-T 的 −2 仅来自一个 episode 翻转，不能视为系统性退化。Reacher 有 20 个不一致配对、净退化 8 个 episode，下降信号明显强于 Push-T；但单训练种子和 50 个 episode 仍不足以估计稳定效应。

Reacher 的 14 个 E5 成功、E6 失败 episode 中，纯 Stage-A actor 在 12 个上成功。因此 Reacher 下降不能用“actor 本身太弱”解释；planner 会把部分本来可成功的 actor 初始化优化坏。

### 7.2 Simulator-grounded 排序诊断

诊断固定复用 Push-T/Reacher E5 Stage-B cohort 的 slot 0–7。每个状态构造 45 个候选，包括 expert、actor、zero、时间置换、随机候选、actor 邻域和 expert 邻域。Stage-B 先预测 latent cost，再从同一模拟器状态执行候选一个 planning horizon（25 个原始环境步），用真实物理终点距离和任务成功判据评价。

| 任务 | 物理代价 Spearman | top-5 recall | 预测 top-1 成功率 | oracle 成功率 |
|---|---:|---:|---:|---:|
| Push-T | 0.756 | 37.5% | 100% | 100% |
| Reacher | 0.732 | 25.0% | 62.5% | 100% |

| 任务 / 候选池 | top-1 成功率 | oracle 成功率 | 候选成功率 |
|---|---:|---:|---:|
| Push-T / zero-centered | 0% | 0% | 0% |
| Push-T / actor-centered | 100% | 100% | 19.1% |
| Push-T / expert-centered | 100% | 100% | 100% |
| Reacher / zero-centered | 12.5% | 50% | 5.1% |
| Reacher / actor-centered | 62.5% | 100% | 27.2% |
| Reacher / expert-centered | 75% | 100% | 80.6% |

Actor warm-start 确实改善候选覆盖：Reacher actor-centered pool 的 oracle 从 50% 提高到 100%，候选成功率从 5.1% 提高到 27.2%。问题集中在高质量候选的精细排序。Reacher 上模型预测 actor cost=`0.0326`、expert cost=`0.1070`，认为 actor 更优；模拟器真实终点距离却是 actor=`0.0952`、expert=`0.0342`，且单 horizon 成功率分别为 50% 和 100%。

当前最强机制解释是：actor 提高了初始覆盖，但 Stage-B 在数据流形附近存在局部错序；CEM 只返回最后一轮 elite mean、没有保存历史 best，后续迭代可能利用模型误差，把可成功的 actor 初始化推向失败区域。该结论仍属于机制性推断，因为当前诊断只执行首 horizon，没有记录完整 30 轮 CEM 和第二次 replan。

## 8. 动作维度与 Stage-A 增益假设

### 8.1 原始动作维度与 Action Block 维度

四个 HDF5 数据集的原始 `action` 列形状分别为 Cube `[N,5]`，其余任务 `[N,2]`。当前统一使用 `frameskip=5`；数据管线把连续 5 帧动作沿最后一个维度拼接成一个 Action Block，因此模型实际接收的单 token 动作维度为：

| 任务 | 原始单帧动作维度 | `frameskip` / `action_block` | 单个 Action Block 维度 | 5-block plan 总标量数 |
|---|---:|---:|---:|---:|
| Cube | 5 | 5 | **25** | **125** |
| Push-T | 2 | 5 | **10** | **50** |
| Reacher | 2 | 5 | **10** | **50** |
| TwoRoom | 2 | 5 | **10** | **50** |

这里的 5-block plan 覆盖 25 个原始环境动作步。评估代码在 `source/policy/fast_lewam_eval.py::_validate_action_dim` 中显式检查 `model.action_dim = action_block × env_action_dim`；实际训练配置也记录 Cube `action_dim=25`、其余任务 `action_dim=10`。

### 8.2 当前结果是否支持“高维动作收益更大”

用于检验“Stage-A supervision 是否增强 Stage-B 动作理解”的直接对照是 E3−E1，而不是 E6−E5：E3 只在训练时加入 Stage-A auxiliary loss，Stage-B 评估仍从随机 CEM 初始化；E6 则额外改变了搜索初始化机制。

| 任务 | Action Block 维度 | E3−E1 Stage-B | 证据强度 |
|---|---:|---:|---|
| Cube | 25 | **+11.3 ± 6.4**，3/3 seeds 为正 | 当前最强正证据 |
| Push-T | 10 | **−2.7 ± 4.2**，1/3 seeds 为正 | 不支持正增益 |
| Reacher | 10 | **0**，单 seed | 没有可见增益 |
| TwoRoom | 10 | 缺省 | 无法判断 |

E6 的搜索结果也呈相同方向的相关性：Cube 为 +32，Push-T 为 −2，Reacher 为 −16，TwoRoom 为 0（但 TwoRoom 的 E5 已经 100%，受天花板限制）。因此现有结果与以下假设一致：当每个 Action Block 从 10 维增至 25 维、整个规划向量从 50 个标量增至 125 个标量时，Stage-A 的结构化动作建模或 actor proposal 更可能帮助 Stage-B。

但目前只能写成“支持性相关证据”，不能写成因果结论，原因是：

- 只有 Cube 一个 25 维任务，动作维度与环境动力学、目标结构、数据质量和任务难度完全混杂；
- Cube 的 Stage-A 已饱和且 shuffled-goal 仍约 40%，任务先验与其他任务不同；
- E3 衡量 auxiliary supervision，E6 衡量搜索初始化，两者虽都在 Cube 上为正，却不是同一种机制；
- TwoRoom 的 E1–E4 缺省，低维组少了一个关键对照；
- Reacher E6 已证明，即使 actor 提高候选覆盖，Stage-B 的局部错序仍可抵消增益。

所以本轮最稳妥的表述是：**结果提示 Stage-A 对高维 Action Block 的帮助可能更大，Cube 提供了跨训练种子的一致证据；但当前任务数不足以证明增益由动作维度本身造成。** 真正确认该假设需要在同一任务内改变动作表示维度，同时固定物理预测跨度、模型容量、训练步数和 CEM budget。

## 9. 模型规模与推理时间

### 9.1 参数量

第一轮 Fast-LeWAM 的 Stage-A 与 Stage-B 不是两个独立模型，而是同一个 checkpoint 的两个 mode，共享视觉 encoder、projector 和六层 SharedDiT。因此 Stage-A、Stage-B 和 E6 的模型参数量相同。LeWM 没有对应的 Stage-A actor，只提供 world model + CEM planner。

| 任务组 | Fast-LeWAM 参数量 | LeWM 参数量 | Fast−LeWM | Fast / LeWM | Lightning FP32 参数体积估计 |
|---|---:|---:|---:|---:|---:|
| Cube（25 维 block） | 10,488,985 | 18,034,628 | −7,545,643 | 58.16% | 41.956 MB vs 72.139 MB |
| Push-T/Reacher/TwoRoom（10 维 block） | 10,483,210 | 18,034,478 | −7,551,268 | 58.13% | 41.933 MB vs 72.138 MB |

第一轮 Fast-LeWAM 比 LeWM 少约 **41.8% 参数**，或者说 LeWM 参数量约为 Fast-LeWAM 的 **1.72 倍**。Cube 只因更大的动作输入/输出层多约 5.8K 参数，对总体规模影响很小。表中的 MB 是 Lightning 按 FP32 参数计算的静态体积，不等于 bf16 推理时的峰值显存。

### 9.2 已保存评估的端到端 wall-clock

下表直接读取结果 JSON 的 `evaluation_seconds`。每项均为 50 episodes、相同任务 cohort、`eval_budget=50`；Stage-B 均使用 300 candidates、top-30、30 CEM iterations。Fast 使用 E5 seed 3072，LeWM 使用本地 E0。

| 任务 | LeWM Stage-B | Fast Stage-A | A / LeWM-B | Fast Stage-B | Fast-B / LeWM-B | Fast-B / Fast-A |
|---|---:|---:|---:|---:|---:|---:|
| Cube | 138.2 s | 551.9 s | 3.99× | 1285.5 s | 9.30× | 2.33× |
| Push-T | 57.4 s | 695.1 s | 12.11× | 999.3 s | 17.40× | 1.44× |
| Reacher | 159.3 s | 349.1 s | 2.19× | 369.6 s | 2.32× | 1.06× |
| TwoRoom | 68.2 s | 17.3 s | 0.25× | 229.9 s | 3.37× | 13.32× |
| **四任务合计** | **423.1 s** | **1613.4 s** | **3.81×** | **2884.3 s** | **6.82×** | **1.79×** |

这些数字是已有实验的端到端记录，不是受控的纯神经网络 latency benchmark。它们包含环境仿真、图像预处理、视频保存、成功后的提前终止、不同日期的 GPU 争用和缓存状态。四任务倍率跨度很大，而且相同 CEM budget 的 E5/E6 也出现过数倍 wall-clock 波动。因此可以据此得出“当前系统没有体现 Fast 的速度优势”，但不能把 `6.82×` 当成稳定的架构速度比。

### 9.3 当前 Fast Stage-B 为什么可能反而更慢

从实现看，Fast 的 latent transition 本身是并行的：一次 SharedDiT 前向同时输出 5 个 future latent；LeWM 则按时间递归 rollout 5 步。可是当前 Fast `get_cost()` 直接编码 CEM 展开后的 `[B,S,...]` 图像，`S=300`，导致相同 current/goal 图像在每个 CEM iteration 中被重复编码约 300 次。对应代码为 `source/model/fast_lewam/jepa.py:462-465`。

LeWM 在 `source/model/lewm/jepa.py:94-101` 中先取一个 sample、只编码一次当前观测，再把 latent 扩展给所有候选，然后在 `104-118` 行执行串行 latent rollout。也就是说，第一轮 Fast Stage-B 节省了 latent 时间维的递归，却引入了更昂贵的重复视觉编码；视觉 encoder 成本可能掩盖甚至超过并行预测收益。

Stage-A 则没有 CEM，但每次生成动作需要 10 次 Euler/flow 前向；代码位于 `source/model/fast_lewam/jepa.py:330-347`。因此“没有 CEM”不等于单次模型调用。其端到端耗时还会受到 replan 次数和环境提前终止影响。

当前效率结论是：

1. **模型大小达成 Fast：** 参数量比 LeWM 少约 41.8%。
2. **端到端速度尚未达成 Fast：** 已保存结果中 Fast Stage-B 四任务合计 wall-clock 是 LeWM 的 6.82 倍。
3. **Stage-A 通常比 Fast Stage-B 快，但并非稳定：** 四任务合计约快 1.79 倍，逐任务差异很大。
4. **不能据现有 wall-clock 宣称架构固有慢 6.82 倍：** 首先需要缓存每次 replan 的 `z0`/`goal_latent`，避免沿候选维重复视觉编码，然后在同一 GPU、无并发、关闭视频、固定 replan 次数下重新做 warm-up 后 latency benchmark。

#### 9.3.1 重复视觉编码修复（2026-08-11）

上述重复编码不是可选优化，而是违反“每次 planning context 只编码一次、随后在候选层级共享 latent”的分层等价错误，现已在正式 Stage-B 路径中修复：

- CEM 展开的 sample 维确为广播共享时，只编码一份 current/goal，并在所有 iteration 中复用 latent；
- E6 actor warm-start 的同一份 latent context 传给后续 CEM，不再二次编码；
- 缓存保存 current/goal 的精确快照，只有 observation 完全相等时才复用；替换或原地修改 observation 都会失效；
- candidate-specific observation 保留旧的逐候选编码语义；current/goal 仍分别编码，以保持混合 dtype 或不同空间形状时的旧语义。

回归测试覆盖普通 CEM、E6 warm-start、缓存与旧 cost 的数值等价、缓存失效、混合图像 dtype，以及 candidate-specific fallback。Cube E5 epoch-10、RTX 4090、`B=1/S=300/H=5` 的初始受控测量中，单次 cost 从 614.256 ms/600 张 encoder 输入降至 16.796 ms/2 张；同 context 再调用为 3.268 ms/0 张，最大绝对 cost 误差 `7.15e-7`。完整 30-iteration CEM 从 4.908 s 降至 0.109 s，约 45.1×。该数字隔离验证修复，不应直接外推为新的四任务 wall-clock。

Stage-A 加速与该错误修复分开处理：探索只加载已训练 checkpoint；经配对验证与用户确认后，仅延迟预处理已应用到正式 Stage A/C chunk policy，Euler 步数与精度未改变。协议与结果见 `docs/plan/stage_a_inference_speed_exploration.md`。

## 10. 本轮可支持的结论

1. **Cube 存在稳定的单向正迁移。** E3 相对 E1 三种子平均提升 11.3 点，但仍未超过 E0。
2. **Action supervision 对 planner 的增益不能跨任务泛化。** Push-T E3 相对 E1 平均下降 2.7 点，Reacher 单种子 E3 与 E1 持平。
3. **World Model supervision 对 actor 的帮助更稳定。** Push-T E5 相对 E4 的 Stage-A 三个 seed 全部提升，平均 +4.7 点。
4. **跨-head梯度没有稳定增强 planner。** Push-T E5 相对 E4 的 Stage-B 平均 −0.7 点且方向不一致。
5. **goal conditioning 确实生效。** Push-T/Reacher shuffled-goal 大幅下降；Cube 仍保留约 40% 成功率，说明任务先验较强。
6. **Actor warm-start 具有强任务依赖。** Cube 显著提升，Push-T 基本持平，Reacher 下降，TwoRoom 受天花板限制。
7. **E6 的关键瓶颈是顶端排序而不是候选覆盖。** Reacher 的 actor-centered pool 已包含成功轨迹，但 Stage-B/CEM 未稳定选中并保留它们。
8. **测试集峰值不能作为 checkpoint 选择依据。** Reacher E3/E4 中期峰值均高于 final，主报告必须坚持预定 final。
9. **动作维度假设获得支持性证据。** 唯一的 25 维 Action Block 任务 Cube 在 E3−E1 和 E6−E5 上均明显受益，而 10 维任务没有同类增益；但目前不足以作因果归因。
10. **参数效率成立；Stage-B 重复编码已修复。** 第一轮 Fast-LeWAM 比 LeWM 少约 41.8% 参数；旧端到端记录仍不能代表修复后的速度，新结论需重新评估。

## 11. 本轮不能支持的主张

- 不能声称 Action Head 在所有任务上增强 World Model。
- 不能声称 Fast Stage-B 已整体超过本地 LeWM；Cube 和 Push-T 仍存在差距。
- 不能把 E6 描述为跨任务稳定增益。
- 不能根据单训练种子和 50 个 episode 的 2–8 点差异作强统计结论。
- 不能使用 TwoRoom E1–E4 做任何消融归因，因为这些项缺省、没有运行。
- 不能根据当前首 horizon 诊断断言完整 CEM 的具体失败轮次；需要迭代轨迹才能确认。
- 不能声称 Cube 的正迁移由动作维度单独导致；当前只有一个高维任务，任务属性存在混杂。
- 不能仅凭旧 wall-clock 声称修复后的 Fast-LeWAM 快于 LeWM；重复视觉编码虽已修复，但尚未在同一受控条件下完成四任务 Stage-B 端到端重评。

## 12. 缺省项与本轮收口状态

本轮唯一成组缺省项为 TwoRoom E1–E4 seed 3072：

- E1 Stage-B only：缺省；
- E2 Stage-A only：缺省；
- E3 expert-action joint training：缺省；
- E4 detached predicted-action mixing：缺省。

缺省项保留在矩阵中，但不参与均值、趋势判断或机制结论。截至 2026-08-10，没有 Fast-LeWAM 训练或评估进程运行。本报告记录当前轮次的静态收口状态，不自动启动缺省实验或第二轮实验。

## 13. 结果与产物索引

### 13.1 主要训练目录

- E0 Cube：`outputs/lewm/cube/0803_e0_lewm_baseline_bs128/`
- E0 Push-T：`outputs/lewm/pusht/0803_e0_lewm_baseline_bs128/`
- E0 Reacher：`outputs/lewm/reacher/0721_/`
- E0 TwoRoom：`outputs/lewm/tworoom/0721_/`
- Cube E1：`outputs/fast_lewam/cube/0722_stage_b_only/`
- Cube E2：`outputs/fast_lewam/cube/0730_e2_stage_a_only/`
- Cube E3：`outputs/fast_lewam/cube/0802_e3_expert_actions/`
- Cube E4：`outputs/fast_lewam/cube/0803_e4_detach/`
- Cube E5：`outputs/fast_lewam/cube/0803_e5_cross_head_grad/`
- Push-T E1：`outputs/fast_lewam/pusht/0802_e1_stage_b_only/`
- Push-T E2：`outputs/fast_lewam/pusht/0730_e2_stage_a_only/`
- Push-T E3：`outputs/fast_lewam/pusht/0802_e3_expert_actions/`
- Push-T E4：`outputs/fast_lewam/pusht/0803_e4_detach/`
- Push-T E5：`outputs/fast_lewam/pusht/0803_e5_cross_head_grad/`
- Reacher E1：`outputs/fast_lewam/reacher/0809_e1_stage_b_only_seed3072_tmux/`
- Reacher E2：`outputs/fast_lewam/reacher/0809_e2_stage_a_only_seed3072_tmux/`
- Reacher E3：`outputs/fast_lewam/reacher/0809_e3_expert_actions_seed3072_tmux/`
- Reacher E4：`outputs/fast_lewam/reacher/0809_e4_detach_seed3072_tmux/`
- Reacher E5：`outputs/fast_lewam/reacher/0717_/`
- TwoRoom E5：`outputs/fast_lewam/tworoom/0717_/`

### 13.2 E6 与诊断目录

- Cube E6：`outputs/fast_lewam/cube/0803_e5_cross_head_grad/eval/epoch_10/stage_b_actor_warm_start/`
- Push-T E6：`outputs/fast_lewam/pusht/0803_e5_cross_head_grad/eval/epoch_10/stage_b_actor_warm_start/`
- Reacher E6：`outputs/fast_lewam/reacher/0717_/eval/epoch_10/stage_b_actor_warm_start/`
- TwoRoom E6：`outputs/fast_lewam/tworoom/0717_/eval/epoch_10/stage_b_actor_warm_start/`
- Push-T 排序诊断：`outputs/fast_lewam/pusht/0803_e5_cross_head_grad/diagnostics/stage_b_ranking/epoch_10/`
- Reacher 排序诊断：`outputs/fast_lewam/reacher/0717_/diagnostics/stage_b_ranking/epoch_10/`

## 14. 下一轮入口（仅记录，不执行）

恢复实验后，优先在 Reacher 的 E6 失败配对上记录 CEM 第 0/1/2/5/10/20/30 轮 mean、elite、预测 cost 和模拟器真实代价，并覆盖第二次 replan。若确认预测 cost 随迭代下降而真实代价上升，再比较 early-stop / best-ever 保留；之后再决定是否进入 `fast_lewam_stage_b_optimization.md` 中的 Stage-B timestep、反事实监督和训练目标修正。
