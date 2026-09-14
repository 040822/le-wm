# Round 4 第 10 轮 / seed-3072 报告

## 范围与结果来源

- 任务：Cube、Push-T、Reacher、TwoRoom。
- 训练：`stage_abde`，seed=`3072`，名义 epoch 10，包含 A/B/D/E 的 Shared DiT。
- 评测协议：`round3_revised`，seed=42；dev 50 个 episode，final 200 个 episode；goal offset=25，budget=50，horizon=5，receding horizon=5，action block=5。
- 基础结果目录：`outputs/round4/epoch10_seed3072`（P0/P1/P2 及原始 P3/P4 运行结果）。
- 符合协议的 best-of-N 目录：`outputs/round4/epoch10_seed3072_batch64`（P3/P4/P4-first，candidate batch=64）。
- CEM 计时附属结果：`outputs/round4/epoch10_seed3072_timed_peak`。
- 代码提交：`419c704`（R4 实现）、`245ad56`（权重-only continuation）、`96b254d`（CEM 计时信息）、`dd7af42`（CEM 峰值显存信息），以及最终报告和诊断提交。

最终 cohort 已固定且可复现，但不是严格的 held-out 泛化集：它来自完整原始数据集，并使用现有 window-level 切分协议生成。本结果仅用于探索，不能证明加入 D/E 对 A/B 带来因果提升。

Cube、Push-T、Reacher 的首轮训练曾中断。Cube 和 Push-T 从 epoch-9 权重快照继续，Reacher 从 epoch 8 继续。

这些 continuation 只恢复模型权重并重新初始化 optimizer，不恢复 optimizer state。TwoRoom 直接完成原始训练。每个 run 目录均保留了这些来源信息。

## R4 正式成功率（%）

下表的 P0/P1/P2 使用基础结果目录，P3/P4/P4-first 使用符合协议的 batch64 目录。P0-shuf 和 P4-first 仅用于 dev 诊断。更早的 candidate-batch-8 运行仅保留作审计，不作为 canonical best-of-N 结果。

### 开发集（Dev）

| 任务 | P0 | P1 | P2 | P3 | P4 | P0-shuf | P4-first |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cube | 100 | 58 | 98 | 100 | 100 | 8 | 100 |
| Push-T | 92 | 94 | 96 | 94 | 90 | 8 | 60 |
| Reacher | 74 | 86 | 68 | 78 | 84 | 4 | 58 |
| TwoRoom | 96 | 100 | 98 | 98 | 98 | 48 | 96 |

### 最终集（Final）

| 任务 | P0 | P1 | P2 | P3 | P4 |
|---|---:|---:|---:|---:|---:|
| Cube | 100 | 59.5 | 95.5 | 100 | 100 |
| Push-T | 89.5 | 86 | 92.5 | 96 | 89 |
| Reacher | 74 | 86.5 | 68.5 | 85 | 78 |
| TwoRoom | 94.5 | 97 | 98.5 | 99 | 99.5 |

episode 配对比较位于两个结果目录的 `analysis.json` 以及跨目录的 P4/P2 artifact 中。

在 final 上，batch64 P4 相对 P3 在 Cube、Push-T、Reacher、TwoRoom 上分别为 0.0、-7.0、-7.0、+0.5 个百分点。P4 相对 P2 分别为 +4.5、-3.5、+9.5、+1.0 个百分点。

P4-first 可用于观察 verifier 的作用。相对于直接执行第一条候选，P4 在 dev 上的提升为：Cube 0 个百分点、Push-T 30 个百分点、Reacher 26 个百分点、TwoRoom 2 个百分点。

## 规划耗时与资源

P3/P4 的 timing 保存在符合协议的 batch64 结果 payload 中。P1/P2 的 CEM timing 在不改变 solver 语义的前提下加入，并记录在最终的 `timed_peak` 附属结果中。

开发集规划中位耗时如下，单位为秒：

| 任务 | P1 CEM | P2 热启动 CEM | P3 action-64 | P4 latent-64 |
|---|---:|---:|---:|---:|
| Cube | 4.954 | 6.232 | 0.880 | 0.919 |
| Push-T | 6.675 | 5.728 | 0.360 | 0.363 |
| Reacher | 9.471 | 9.024 | 0.540 | 0.477 |
| TwoRoom | 6.385 | 2.913 | 0.381 | 0.375 |

P3/P4 使用 64 个候选、16 个 flow steps、solver batch=1，以及 candidate batch=64。`timed_peak` CEM 附属结果包含 p95 延迟、forward 次数和峰值分配显存。开发集资源审计如下：

| 任务 | 模式 | 中位数（秒） | p95（秒） | forward 次数 | 峰值分配显存（MiB） |
|---|---|---:|---:|---:|---:|
| Cube | P1 | 4.954 | 6.865 | 2160 | 82.5 |
| Cube | P2 | 6.232 | 11.437 | 1530 | 220.0 |
| Push-T | P1 | 6.675 | 10.989 | 1740 | 82.4 |
| Push-T | P2 | 5.728 | 9.976 | 1620 | 219.9 |
| Reacher | P1 | 9.471 | 11.386 | 2460 | 82.4 |
| Reacher | P2 | 9.024 | 11.119 | 2460 | 219.9 |
| TwoRoom | P1 | 6.385 | 6.385 | 1500 | 82.4 |
| TwoRoom | P2 | 2.913 | 4.947 | 1680 | 219.9 |

计时附属结果仅用于资源审计；正式成功率表仍以基础结果目录中的 canonical 结果为准。

## LeFlow 参考结果

四任务 artifact manifest 及其依赖的 LeWM checkpoint 均通过 SHA256 校验。梯度与来源审计记录在 `docs/plan/round4_leflow_audit.md`。修订协议下的 final 成功率如下：

| 任务 | LeFlow final |
|---|---:|
| Cube | 100% |
| Push-T | 96% |
| Reacher | 83.5% |
| TwoRoom | 100% |

对应的 dev 成功率依次为 100%、98%、74% 和 100%。这些结果是历史/参考比较，不是同一 checkpoint 下的 R4 消融结果。

## 训练诊断

Epoch-10 诊断指标（`loss`、`d_loss`、`d_path_variance`、`e_action_mse`）如下：

| 任务 | loss | D loss | D 路径方差 | E action MSE |
|---|---:|---:|---:|---:|
| Cube | 0.9276 | 0.2057 | 0.6592 | 0.0702 |
| Push-T | 1.1630 | 0.2598 | 0.6550 | 0.1244 |
| Reacher | 2.8209 | 0.1953 | 0.6337 | 0.8014 |
| TwoRoom | 2.7301 | 0.3406 | 0.6514 | 0.7604 |

各 run 的 `round4_diagnostics.jsonl` 在可用时保留 epoch 0、5、10 的记录；continuation run 还保留了继续训练前的 epoch 快照。

## 离线 IDM/路径诊断

`scripts/round4_offline_diagnostics.py` 对每个任务评估 16 个真实窗口，以及每个窗口的 8 条生成路径。

下表依次为：真实路径上的 E action MSE、E 解码真实路径后的 B verifier MSE、生成路径上的 B validation MSE，以及生成终点 MSE（由 D 的 endpoint contract 固定为 0）：

| 任务 | 真实路径 E MSE | 解码动作 B MSE | 生成路径 B MSE | 终点 MSE |
|---|---:|---:|---:|---:|
| Cube | 0.0696 | 0.0103 | 0.0775 | 0.0000 |
| Push-T | 0.0630 | 0.0322 | 0.1258 | 0.0000 |
| Reacher | 0.8149 | 0.0219 | 0.1428 | 0.0000 |
| TwoRoom | 0.7657 | 0.1160 | 0.2926 | 0.0000 |

这些诊断用于检查离线能力；P4 的实际选择仍然使用 B 对解码动作的评分，不使用被 D 固定的终点。

## R4-AB 与 R4-ABDE 对照

R4-AB 保留完整 `Round4FastLeWAM` 结构，只训练 A/B，D/E loss 权重为 0。四个任务均从头训练 seed 3072、10 个 epoch。R4-ABDE 使用本报告已有的 epoch-10 结果作为固定对照。

两组使用相同的 `round3_revised` cohort。主指标是 P3；P0、P1、P2 用于辅助定位 A/B 变化。R4-AB 不运行 P4/P4-first，因为其 D/E 参数没有训练。

### P3 主结果

| cohort | 任务 | R4-AB | R4-ABDE | ABDE−AB（百分点） | 配对改进/退化 |
|---|---|---:|---:|---:|---:|
| dev | Cube | 100 | 100 | 0 | 0/0 |
| dev | Push-T | 96 | 94 | -2 | 1/2 |
| dev | Reacher | 86 | 78 | -8 | 5/9 |
| dev | TwoRoom | 100 | 98 | -2 | 0/1 |
| dev 平均 | — | 95.5 | 92.5 | -3.0 | — |
| final | Cube | 100 | 100 | 0 | 0/0 |
| final | Push-T | 96 | 96 | 0 | 4/4 |
| final | Reacher | 84.5 | 85 | +0.5 | 23/22 |
| final | TwoRoom | 99.5 | 99 | -0.5 | 1/2 |
| final 平均 | — | 95.0 | 95.0 | 0.0 | — |

Final 上，ABDE 没有提高 P3 的宏平均成功率。逐任务差异也只有 Reacher 的 +0.5 个百分点，TwoRoom 反而下降 0.5 个百分点。Dev 上 ABDE 低于 AB 3 个百分点，主要来自 Reacher 和 Push-T。

### P0–P2 辅助结果

上一版只列出了四个任务的宏平均，下面展开列出 Cube、Push-T、Reacher 和 TwoRoom。差值均为 R4-ABDE 减 R4-AB，单位为百分点。

#### 开发集（Dev）

| 任务 | P0 AB | P0 ABDE | P0 差值 | P1 AB | P1 ABDE | P1 差值 | P2 AB | P2 ABDE | P2 差值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cube | 100 | 100 | 0 | 40 | 58 | +18 | 100 | 98 | -2 |
| Push-T | 92 | 92 | 0 | 90 | 94 | +4 | 92 | 96 | +4 |
| Reacher | 70 | 74 | +4 | 74 | 86 | +12 | 62 | 68 | +6 |
| TwoRoom | 96 | 96 | 0 | 98 | 100 | +2 | 96 | 98 | +2 |
| 四任务平均 | 89.5 | 90.5 | +1.0 | 75.5 | 84.5 | +9.0 | 87.5 | 90.0 | +2.5 |

#### 最终集（Final）

| 任务 | P0 AB | P0 ABDE | P0 差值 | P1 AB | P1 ABDE | P1 差值 | P2 AB | P2 ABDE | P2 差值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cube | 99.5 | 100 | +0.5 | 49 | 59.5 | +10.5 | 94.5 | 95.5 | +1.0 |
| Push-T | 92.5 | 89.5 | -3.0 | 86 | 86 | 0 | 93 | 92.5 | -0.5 |
| Reacher | 75.5 | 74 | -1.5 | 86 | 86.5 | +0.5 | 67 | 68.5 | +1.5 |
| TwoRoom | 96 | 94.5 | -1.5 | 98.5 | 97 | -1.5 | 98.5 | 98.5 | 0 |
| 四任务平均 | 90.875 | 89.5 | -1.375 | 79.875 | 82.25 | +2.375 | 88.25 | 88.75 | +0.5 |

ABDE 在 P1/P2 上的平均成功率较高，但 P0 final 下降，P3 final 持平。因此，D/E 联合训练对 A/B 的影响并不表现为所有推理模式都一致改善。

当前 seed 3072 的闭环结果没有显示 D/E 辅助训练给 P3 带来收益。这个结果支持继续使用 P3，并把 R4-AB 作为更简洁的训练方案；单个训练 seed 不能证明两种训练方式稳定等价。

本比较的完整配对结果位于 `outputs/round4/ab_vs_abde/analysis.json`，配置登记位于 `config/round4/ab_control.json`，详细对照报告位于 `docs/plan/round4_ab_vs_abde_report.md`。

R4-ABDE 的 Cube、Push-T、Reacher 首轮训练曾中断并进行了权重-only continuation，因此这里是描述性对照，不能作为严格的训练因果证明。

## 结论与解释

1. **本轮没有证明 latent best-of-N（P4）优于 action best-of-N（P3）。** Final 上，P4 相对 P3 在 Cube、Push-T、Reacher、TwoRoom 上分别为 0.0、-7.0、-7.0、+0.5 个百分点。

   Dev 上分别为 0、-4、+6、0 个百分点。

   Reacher 的 dev 改善没有延续到 final，Push-T 的下降则在两个 cohort 上都出现。

2. **按本轮成功率，P3 是更稳妥的 learned planner。** Final P3 为 100%、96%、85%、99%，P2 为 95.5%、92.5%、68.5%、98.5%，P4 为 100%、89%、78%、99.5%。

   P3 在四个任务上均不低于 P2，并在 Push-T、Reacher 上明显高于 P4；P4 仅在 TwoRoom 上比 P3 高 0.5 个百分点。

   由于 P3/P2 的结果来自独立重复评测，不能把这些成功率差异解释为 episode-level 因果效应。

3. **P4 相对 CEM warm-start（P2）的结果是任务依赖的。** Final P4-P2 为 Cube +4.5、Push-T -3.5、Reacher +9.5、TwoRoom +1.0 个百分点。

   Reacher 的提升较明显，但不足以支持跨任务的普遍优势；Push-T 的回退也直接阻止了本轮扩展。

4. **B verifier reranking 是 latent candidate 能否工作的关键步骤。** Dev 中，P4 相对直接执行第一条候选（P4-first）在 Push-T 和 Reacher 上分别提升 30 和 26 个百分点。

   Cube 和 TwoRoom 只提升 0 和 2 个百分点。

   因此，P4-first 不能替代 P4；候选质量与 verifier 选择必须分开分析。

5. **best-of-64 的主要确定性优势是效率，而不是已验证的成功率提升。** Dev 中 P3/P4 的规划中位耗时为 0.360–0.919 秒，P1/P2 CEM 为 2.913–9.471 秒。

   按对应任务和模式比较，best-of-64 约快 5.6–18.9 倍。P3/P4 不做 CEM 迭代，因此适合作为低延迟候选生成方案，但 P4 的 latent route 不能仅凭速度替代 P3。

6. **D→E 的离线链路仍有明显的路径分布差距。** 真实路径上的 E action MSE 在 Cube/Push-T 为 0.0696/0.0630，在 Reacher/TwoRoom 升至 0.8149/0.7657。

   生成路径的 B validation MSE 在四个任务上都高于 E 解码真实路径的 B MSE，分别为 0.0775>0.0103、0.1258>0.0322、0.1428>0.0219、0.2926>0.1160。

   D 的终点 MSE 为零是 endpoint contract 的结果，不等于生成路径已经符合 verifier 或 IDM 的有效动作流形。

7. **训练诊断不能替代闭环评测。** D path variance 在四个任务上都约为 0.63–0.66，说明 D 产生了非退化的噪声相关路径，但它与最终 P4 成功率没有显示出足够的跨任务对应关系。

   当前最可靠的判断仍是固定 cohort 上的闭环成功率、配对结果和规划耗时。

8. **P3 接近历史 LeFlow 参考水平，但该比较不具备同 checkpoint 的因果意义。** P3 final 相对历史 LeFlow final 在 Cube、Push-T、Reacher、TwoRoom 上分别为 0、0、+1.5、-1.0 个百分点。

   P4 则为 0、-7、-5.5、-0.5 个百分点。

   该结果支持继续研究 action candidate，但不能据此宣称 R4-ABDE 复现或超越 LeFlow。

综合来看，本轮支持的结论是：**Shared DiT 可以同时完成 A/B/D/E 的探索性训练。**

P3 action best-of-64 是当前更可靠的低延迟规划候选；P4 latent best-of-64 尚未显示跨任务收益，D/E 的路径—动作接口和 verifier 一致性仍是主要风险。

单 seed、R4-ABDE 的权重-only continuation、非严格 held-out cohort 以及两组训练随机过程不完全 bitwise 对齐，均限制了因果和稳定性结论。

如果需要更强的稳定性证据，应补跑多个训练 seed，并让 AB 与 ABDE 都使用连续、匹配的 optimizer 状态。当前结果已经足够支持 P3 作为主要推理方法。

latent-space CEM、online+D/E 和架构搜索不应在当前证据不足时提前展开。

## 扩展决策

预注册的 epoch-10 dev gate 为 `expand=false`，因此不运行 seed 3073 和 3074。

该 gate 未通过：相对于 P3，只有 Reacher 提升超过 5 个百分点，同时 Push-T 出现下降。

相对于 P2，Reacher 提升，但 Push-T 下降 6 个百分点。效率 gate 同样要求任何任务下降不超过 2 个百分点，Push-T 不满足该条件。

决策 artifact：`outputs/round4/epoch10_seed3072/expansion_decision.json`。

## Checkpoint 与哈希

| 任务 | checkpoint | SHA256 |
|---|---|---|
| Cube | `outputs/round4/resume3_seed3072_cube/checkpoints/r4_abde_seed3072_weights_epoch_10.pt` | `95a26194eb74e3b02f9d82aa4d15e32b746fdbf24bce974d9eed90550a1c5495` |
| Push-T | `outputs/round4/resume3_seed3072_pusht/checkpoints/r4_abde_seed3072_weights_epoch_10.pt` | `1543ccdffd17c68fb9d3f905621995d760809efdfc86d8cc6c5f1486cbbcb9f1` |
| Reacher | `outputs/round4/resume3_seed3072_reacher/checkpoints/r4_abde_seed3072_weights_epoch_10.pt` | `d56252fe27caa89914c9871514d1bfcfeda4466de6dd34494f64bee89d78a83c` |
| TwoRoom | `outputs/20260912_102425_147611/checkpoints/r4_abde_seed3072_weights_epoch_10.pt` | `153b6baa8b5dac40ef5bc12700a19e117397199750edf289670be3e1e81deec6` |

## 验证

- LeFlow artifact manifest 校验：四个任务均通过。
- Cohort 文件 SHA256：与 `config/round4/cohort_artifacts.json` 一致。
- batch64 P3/P4/P4-first 协议结果：20/20 个结果 artifact 存在且有效。
- R4-AB 结果：36/36 个 P0–P3/P0-shuf result artifact 存在且有效。
- R4-AB/ABDE 对照：32 行主比较结果及配对统计已生成。
- CEM 计时资源附属结果：16/16 个结果 artifact 存在，并包含 p95、forward count 和峰值显存字段。
- 离线 IDM/路径诊断：四个任务全部完成。
- Round 4 定向测试：通过，覆盖 mode/mask、梯度、checkpoint 往返、候选拆分、timing wrapper 和 continuation 行为。
- 全量 unittest：292 个测试，1 个跳过，2 个失败。两个失败均是既有的 `test_inspect_h5_dataset.py` 对 episode-length 文本的期望，而当前 inspector 不输出该文本；Round 4 测试没有失败。

## 延后实验

以下实验仍不属于本轮：独立 D/E 模型、Flow-IDM、latent-space CEM、新 Stage C、consistency/cycle loss、online+D/E，以及 token/DiT 架构搜索。
