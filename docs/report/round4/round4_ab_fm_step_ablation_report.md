# Round 4-AB-FM-step-ablation 报告

## 1. 实验目的与范围

本实验是 Round 4 的后继探索，用同一组 R4-AB epoch-10 checkpoint 改变 action flow 的 Euler 采样步数，观察 `P0`、`P2`、`P3` 的闭环成功率和规划成本如何变化。它不是新的直接 action 回归训练，也没有重新训练模型。

实验登记配置为 [`config/round4/flow_step_ablation.json`](../../config/round4/flow_step_ablation.json)，运行入口为 [`scripts/run_round4_ab_flow_step_ablation.sh`](../../scripts/run_round4_ab_flow_step_ablation.sh)，分析入口为 [`scripts/round4_flow_steps.py`](../../scripts/round4_flow_steps.py)。核心实现提交为 `1763812`，分析日志目录初始化修复提交为 `f8c9fef`。

使用的 checkpoint 是已有的 R4-AB、seed 3072、epoch 10 权重：

| 任务 | checkpoint | SHA256 |
|---|---|---|
| Cube | `outputs/round4/ab_seed3072_cube/checkpoints/r4_ab_seed3072_weights_epoch_10.pt` | `2748151a2c43c7841246a5fb8ee1fd7915ebbc00bfc27aa1a32b0f9080d92956` |
| Push-T | `outputs/round4/ab_seed3072_pusht/checkpoints/r4_ab_seed3072_weights_epoch_10.pt` | `62d00096a34e9f0b6da4ceb6a3c61eb350701a7aade5f5e4d2c1ee528b06015e` |
| Reacher | `outputs/round4/ab_seed3072_reacher/checkpoints/r4_ab_seed3072_weights_epoch_10.pt` | `087991339c9499d10c1a58a28e72d07c7dfd819d5f072c67bfd0169acaa83553` |
| TwoRoom | `outputs/round4/ab_seed3072_tworoom/checkpoints/r4_ab_seed3072_weights_epoch_10.pt` | `48d1ed2520ec7142327969eacf1c37b144faec51cfd6c9ce72701a8421bf57c2` |

任务为 Cube、Push-T、Reacher 和 TwoRoom；action flow steps 为 `1/2/5/10/16`。每个任务均评测 `P0/P2/P3` 的 5 个步数和固定的 `P1`，覆盖 dev 50 episodes 与 final 200 episodes，共 `4 × (3 × 5 × 2 + 1 × 2) = 128` 个结果 artifact。

## 2. 与直接 action 生成的区别

这里的 `steps=1` 仍然使用训练好的 action flow-matching 网络、噪声初始化、timestep 和同一个 checkpoint；它只是从噪声到 action 的 Euler 积分只调用一次网络。`steps=2/5/10/16` 是同一 flow field 的不同推理离散化，不是五个不同模型。

因此，本实验不能回答“直接 action 回归是否优于 flow matching”。要回答那个问题，需要额外训练一个以 action MSE 为目标的独立 checkpoint；该实验不在本轮范围内。这里观察到的 1-step 结果也不能直接解释成“模型学会了直接生成 action”。

模式含义如下：

| 模式 | 推理方式 | action flow steps 的作用 |
|---|---|---|
| P0 | A 生成一条 action plan，直接执行 | 改变 A 的采样步数 |
| P1 | 随机初始化的标准 CEM | 不使用 A，因而对步数不变量；只运行一份 |
| P2 | A warm-start，再由 B/CEM 优化 | 只改变 warm-start action 的采样步数，CEM 仍为 300/30/30 |
| P3 | A 生成 64 条 action 候选，由 B 选最优 | 改变 64 条 action 候选的采样步数，候选数和 B verifier 不变 |

所有模式都使用 `round3_revised` 协议：seed 42、goal offset 25、budget 50、horizon 5、receding horizon 5、action block 5；P2 的 CEM 为 300 samples、topk 30、30 iterations、`var_scale=1.0`，P3 为 64 candidates、candidate batch 64、solver batch 1。所有 CUDA 评测只绑定 GPU0–3。

## 3. 结果完整性

最终产物目录为 [`outputs/round4/ab_flow_steps_seed3072`](../../outputs/round4/ab_flow_steps_seed3072)。

- 128/128 个 `result.json` 存在，且 `status=ok`。
- 汇总 [`analysis.json`](../../outputs/round4/ab_flow_steps_seed3072/analysis/analysis.json) 有 128 行；其中 120 行为 action-step 结果，8 行为 invariant P1。
- 所有 action-step 结果的 cohort ID、cohort SHA256、checkpoint、task/mode/cohort 身份均通过校验。
- 每个非 canonical step 都与 step=16 结果完成 episode-level 对齐和配对统计。
- 详细 CSV 为 [`analysis.csv`](../../outputs/round4/ab_flow_steps_seed3072/analysis/analysis.csv)。

需要保留一个协议限制：数据来自完整原始 dataset，final cohort 是固定的协议评测集，但不是严格 held-out 泛化集。因此下面的数字用于同 checkpoint、同 cohort 的推理对照，不应表述为未见 episode 上的泛化证明。

## 4. 开发集成功率（%）

表中列表示 action flow steps；P1 不随 steps 变化，单独列在最后。

### P0

| 任务 | 1 | 2 | 5 | 10 | 16 | P1 |
|---|---:|---:|---:|---:|---:|---:|
| Cube | 100 | 100 | 100 | 100 | 100 | 40 |
| Push-T | 96 | 100 | 96 | 94 | 92 | 90 |
| Reacher | 90 | 84 | 80 | 70 | 70 | 74 |
| TwoRoom | 98 | 98 | 98 | 96 | 96 | 98 |

### P2

| 任务 | 1 | 2 | 5 | 10 | 16 | P1 |
|---|---:|---:|---:|---:|---:|---:|
| Cube | 98 | 98 | 98 | 100 | 96 | 40 |
| Push-T | 94 | 92 | 94 | 92 | 92 | 90 |
| Reacher | 92 | 74 | 64 | 62 | 50 | 74 |
| TwoRoom | 100 | 100 | 96 | 96 | 100 | 98 |

### P3

| 任务 | 1 | 2 | 5 | 10 | 16 | P1 |
|---|---:|---:|---:|---:|---:|---:|
| Cube | 100 | 100 | 100 | 100 | 100 | 40 |
| Push-T | 98 | 100 | 98 | 96 | 96 | 90 |
| Reacher | 92 | 90 | 80 | 72 | 86 | 74 |
| TwoRoom | 98 | 100 | 100 | 100 | 100 | 98 |

宏平均如下：

| 模式 | 1 | 2 | 5 | 10 | 16 | P1 invariant |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 96.0 | 95.5 | 93.5 | 90.0 | 89.5 | 75.5 |
| P2 | 96.0 | 91.0 | 88.0 | 87.5 | 84.5 | 75.5 |
| P3 | 97.0 | 97.5 | 94.5 | 92.0 | 95.5 | 75.5 |

P1 的四任务 dev 成功率实际为 Cube 40%、Push-T 90%、Reacher 74%、TwoRoom 98%，宏平均 75.5%；表中为便于横向阅读重复列出。

## 5. 最终集成功率（%）

### P0

| 任务 | 1 | 2 | 5 | 10 | 16 | P1 |
|---|---:|---:|---:|---:|---:|---:|
| Cube | 100 | 100 | 100 | 99.5 | 99.5 | 49 |
| Push-T | 95 | 95 | 94 | 93 | 92.5 | 86 |
| Reacher | 80 | 83 | 81 | 78.5 | 75.5 | 86 |
| TwoRoom | 97 | 97 | 97.5 | 97 | 96 | 98.5 |

### P2

| 任务 | 1 | 2 | 5 | 10 | 16 | P1 |
|---|---:|---:|---:|---:|---:|---:|
| Cube | 96.5 | 97 | 94.5 | 94.5 | 97.5 | 49 |
| Push-T | 93 | 92.5 | 93.5 | 93 | 92.5 | 86 |
| Reacher | 84 | 81.5 | 74.5 | 67 | 66 | 86 |
| TwoRoom | 100 | 99 | 97.5 | 98.5 | 98 | 98.5 |

### P3

| 任务 | 1 | 2 | 5 | 10 | 16 | P1 |
|---|---:|---:|---:|---:|---:|---:|
| Cube | 100 | 100 | 100 | 100 | 100 | 49 |
| Push-T | 96.5 | 97.5 | 96 | 94.5 | 96 | 86 |
| Reacher | 88 | 84.5 | 84 | 84.5 | 84.5 | 86 |
| TwoRoom | 98 | 99.5 | 99 | 99 | 99.5 | 98.5 |

P1 的四任务 final 成功率实际为 Cube 49%、Push-T 86%、Reacher 86%、TwoRoom 98.5%，宏平均 79.875%。宏平均曲线为：

| 模式 | 1 | 2 | 5 | 10 | 16 | P1 invariant |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 93.000 | 93.750 | 93.125 | 92.000 | 90.875 | 79.875 |
| P2 | 93.375 | 92.500 | 90.000 | 88.250 | 88.500 | 79.875 |
| P3 | 95.625 | 95.375 | 94.750 | 94.500 | 95.000 | 79.875 |

## 6. 配对比较：相对 canonical step=16

分析器对相同 task/mode/cohort 的 episode ID 和 start step 做配对；下面是四任务宏平均的成功率差值，单位为百分点。它不是把不同 cohort 当成独立样本的比较。

### Dev：相对 step=16 的差值

| 模式 | step=1 | step=2 | step=5 | step=10 | step=16 |
|---|---:|---:|---:|---:|---:|
| P0 | +6.5 | +6.0 | +4.0 | +0.5 | 0.0 |
| P2 | +11.5 | +6.5 | +3.5 | +3.0 | 0.0 |
| P3 | +1.5 | +2.0 | -1.0 | -3.5 | 0.0 |

### Final：相对 step=16 的差值

| 模式 | step=1 | step=2 | step=5 | step=10 | step=16 |
|---|---:|---:|---:|---:|---:|
| P0 | +2.125 | +2.875 | +2.250 | +1.125 | 0.000 |
| P2 | +4.875 | +4.000 | +1.500 | -0.250 | 0.000 |
| P3 | +0.625 | +0.375 | -0.250 | -0.500 | 0.000 |

一些具有代表性的 final episode-level 配对结果如下；完整的每任务、每步改进数/退化数和 McNemar exact p-value 保存在 `analysis.json/csv`：

| 比较 | 任务 | 成功率差 | 改进 / 退化 | McNemar exact p |
|---|---|---:|---:|---:|
| P2 step=1 vs 16 | Cube | -1.0 | 4 / 6 | 0.7539 |
| P2 step=1 vs 16 | Push-T | +0.5 | 6 / 5 | 1.0000 |
| P2 step=1 vs 16 | Reacher | +18.0 | 54 / 18 | 0.0000257 |
| P2 step=1 vs 16 | TwoRoom | +2.0 | 4 / 0 | 0.1250 |
| P3 step=1 vs 16 | Cube | 0.0 | 0 / 0 | — |
| P3 step=1 vs 16 | Push-T | +0.5 | 3 / 2 | 1.0000 |
| P3 step=1 vs 16 | Reacher | +3.5 | 28 / 21 | 0.3916 |
| P3 step=1 vs 16 | TwoRoom | -1.5 | 1 / 4 | 0.3750 |

## 7. 规划耗时、forward 次数与显存

P0 没有 CEM verifier，因此其 `planning` timing 字段为空；P1/P2/P3 的时间和 forward count 均记录在结果 artifact 中。以下列出 final canonical step=16 的代表值，单位分别为秒、次和 MiB：

| 任务 | 模式 | 规划中位数 | p95 | forward 次数 | 峰值显存 |
|---|---|---:|---:|---:|---:|
| Cube | P1 | 17.017 | 21.503 | 9090 | 82.7 |
| Push-T | P1 | 13.374 | 21.553 | 7020 | 82.4 |
| Reacher | P1 | 18.332 | 21.939 | 9270 | 82.4 |
| TwoRoom | P1 | 13.173 | 23.761 | 6360 | 82.4 |
| Cube | P2 | 11.006 | 20.246 | 6150 | 732.0 |
| Push-T | P2 | 12.115 | 21.061 | 6600 | 731.5 |
| Reacher | P2 | 19.979 | 24.888 | 9960 | 731.5 |
| TwoRoom | P2 | 12.724 | 22.357 | 6540 | 731.5 |
| Cube | P3 | 1.825 | 1.825 | 216 | 1415.5 |
| Push-T | P3 | 0.945 | 1.695 | 243 | 1415.0 |
| Reacher | P3 | 1.273 | 1.784 | 330 | 1415.0 |
| TwoRoom | P3 | 0.976 | 1.783 | 236 | 1415.0 |

P3 的 flow steps 从 1 增至 16 时，proposal/flow forward 次数和规划时间整体上升；P2 的总时间主要由固定 CEM 主导，所以 action flow steps 对总 planning median 的影响小于对成功率的影响。所有步数、dev/final timing、candidate count、solver batch 和显存原始值可在 `analysis.csv` 查询。

## 8. 结论

1. **在这个 R4-AB checkpoint 上，增加 Euler flow steps 没有带来成功率收益。** Final 宏平均中，P0 从 step=1 的 93.0% 降至 step=16 的 90.875%，P2 从 93.375% 降至 88.5%；P3 则从 95.625% 变为 95.0%，变化较小但没有正收益。

2. **低步数在本实验中反而更好，尤其是 Reacher 的 P2。** Reacher final P2 从 step=16 的 66% 提升到 step=1 的 84%，配对结果为 54 个 episode 改进、18 个退化，McNemar exact p=0.0000257。这个结果说明当前 checkpoint 的有限步数采样偏差可能比 16-step 积分误差更有利，但不能把它解释成直接 action 回归优势。

3. **P3 对步数更稳健，也仍是主要方法候选。** P3 final 四任务宏平均在 94.5%–95.625% 之间；step=2 在 Push-T/TwoRoom 上最好，step=1 在 Reacher 上最好，step=16 不是普遍最优。若以成功率和低延迟共同考虑，P3 使用 step=1 或 step=2 值得作为后续默认候选，但是否固定为某个步数还应结合更多 seed。

4. **P1 的结果与 action flow steps 无关。** P1 final 四任务为 49/86/86/98.5%，宏平均 79.875%；它不是本次 flow-step sweep 的变量，不能用来判断 flow steps 的影响。

5. **本实验没有证明 Shared DiT 的 latent/action 核心假设。** 这是 action flow 的推理步数消融，不能替代 P3/P4 latent-vs-action 对照，也不能证明直接 action 生成优于 flow matching。它只说明在现有 R4-AB 权重上，16-step Euler 不是成功率最优的工作点。

6. **结论仍是单 checkpoint、单训练 seed 的探索性结论。** final cohort 不是严格 held-out 集；没有新的训练 seed，也没有重新训练 direct-action 模型。因此不据此宣称低步数的稳定提升或训练因果结论。

综合建议：后续如果继续使用 R4-AB，应优先把 P3 作为主规划器，并将 action flow steps=1/2 作为低成本候选推理设置；在扩大训练 seed 或开展独立 direct-action 训练之前，不应把本结果外推为普遍的 flow matching 结论。

## 9. 验证记录

- Round 4 相关定向测试：45 个测试全部通过。
- 结果 artifact：128/128，全部 `status=ok`。
- 分析器：成功生成 128 行 JSON/CSV 汇总，并完成 episode-level 配对。
- shell 语法、Python 编译和 `git diff --check`：通过。
- 全量 unittest：296 个测试，292 个通过，2 个跳过，2 个失败。两个失败均来自已有的 `tests/test_inspect_h5_dataset.py` episode-length 文本格式断言，不属于本实验改动范围。
