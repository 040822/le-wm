# Round 5 Phase 6.1：B 的 latent flow matching 预实验

## 执行状态

完整评估矩阵已完成。每个任务、执行/评分预算、方法与 guidance 条件均保留独立 cohort 行；兼容的历史 Regression-B 行通过 `source_refs` 引用，没有把 25/25 结果当作 10/10 alias。

每臂 44 个任务条件、计划 2200 个 episode 结果；新增实际评估 episode 数为 4300。历史复用行数为 2。

## 预注册主要终点

主要终点为 P3-none、25/25 下 FM-B 相对 Regression-B 的 legacy_50 配对成功率差。每个任务结果按原始 episode 配对；Bootstrap 以 episode 为抽样单元，Wilson 区间用于各臂成功率。

| 任务 | Regression-B | FM-B | 配对差（FM−Regression） | 配对区间 |
|---|---:|---:|---:|---:|
| pusht | 0.980 | 0.980 | +0.000 | [-0.060, +0.060] |
| reacher | 0.900 | 0.760 | -0.140 | [-0.280, +0.000] |

## 训练诊断

Epoch 10 的 validation velocity loss 与从噪声积分后的 latent MSE 分开报告；FM velocity loss 与 Regression-B latent MSE 量纲不同，不能直接比较。正式训练 callback 直接记录 weighted velocity loss 和 integrated endpoint MSE；另用 epoch-10 checkpoint、已保存配置及 seed=3072 的同一 90/10 held-out split 独立重算，两者的绝对差列于下表。训练耗时按解析配置保存到 epoch-10 checkpoint 的文件时间跨度统计；显存列为定期采样到的峰值进程占用。新 FM 分支增加 148,224 个参数（按模型结构统计）。

训练恢复记录：attempt_001 因输出管道 BrokenPipeError 中断（PushT 完成 6/10 epoch、Reacher 完成 5/10 epoch），且没有可继续训练的完整 optimizer checkpoint；其日志与权重已归档但未用于最终模型。正式 attempt_002 从 seed=3072 全新启动并完成 10 epoch。另一次 Hydra 配置解析失败发生在训练进程启动前，没有执行训练更新。

训练日志审计：两个 `phase6_1_epoch_metrics.jsonl` 原始文件各有 11 行；最后一行是 `on_train_end` 回调重复写入的 epoch=11 记录，其 global step 和指标与 epoch 10 相同，并非第 11 个训练 epoch。汇总只计实际完成的 10 个 epoch，原始日志保留。

审计材料：[attempt_001 恢复清单](../../../outputs/round5/phase6_1/training/attempts/attempt_001_interrupted_pipe_failure/recovery_manifest.json)；Hydra 对 `checkpoint_interval_steps` 覆盖的拒绝记录：[PushT](../../../outputs/round5/phase6_1/training/fm_b_seed3072_pusht/phase6_1_launch_attempt_001.json)、[Reacher](../../../outputs/round5/phase6_1/training/fm_b_seed3072_reacher/phase6_1_launch_attempt_001.json)。

| 任务 | validation 加权 velocity loss | validation integrated latent MSE | endpoint callback/posthoc 绝对差 | velocity callback/posthoc 绝对差 | 更新数 | 训练样本 | 训练耗时（config→epoch10 checkpoint） | 抽样峰值进程显存 | 新增参数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pusht | 0.091648 | 0.099443 | 2.73e-05 | 1.31e-05 | 126200 | 16153600 | 10.94h | 19.77 GiB | 148224 |
| reacher | 0.124410 | 0.187604 | 0.000391 | 8.99e-05 | 120930 | 15479040 | 10.54h | 19.77 GiB | 148224 |

## 冻结训练配置与来源

FM 使用 historical R4-AB 的实际保存配置逐字段对照；PushT 与 Reacher 的冻结字段必须全部相同。历史配置未显式写出的 `stage_b_action_source` 按当时有效默认值 `joint` 比较。各任务的完整字段表、两份原始配置副本和 checkpoint/config/data/cohort 哈希见归档链接。

| 任务 | 冻结字段一致 | Regression checkpoint SHA256 | FM checkpoint SHA256 | Dataset SHA256 | Cohort SHA256 |
|---|---:|---|---|---|---|
| pusht | 36/36 | `62d00096a34e…` | `ac17ec68f44e…` | `b6ebd9ac94bb…` | `b110bee6afa2…` |
| reacher | 36/36 | `087991339c94…` | `b27c6bfa5a90…` | `85a7dddfa180…` | `ff4f26ad3fd8…` |

预先指定的差异只有将训练分支从 `stage_ab` 切换到 `stage_ab_fm`、启用 B 的 latent flow matching（K=2），以及新增 noisy-latent、flow-time 和 velocity-head 参数；A、encoder、优化器和其余冻结设置沿用历史配置。

训练启动时工作区有未提交修改。正式训练源码快照包含逐文件 SHA256 与修改时间；所列训练源码和配置文件的修改时间均早于 attempt_002 启动。

| 任务 | 配置差异与哈希归档 | 训练代码身份 | 训练源码快照 |
|---|---|---|---|
| pusht | [training_config_diff.json](../../../outputs/round5/phase6_1/provenance/pusht/training_config_diff.json) | [phase6_1_training_code_identity.json](../../../outputs/round5/phase6_1/training/fm_b_seed3072_pusht/phase6_1_training_code_identity.json) | [phase6_1_training_sources.zip](../../../outputs/round5/phase6_1/training/fm_b_seed3072_pusht/phase6_1_training_sources.zip) |
| reacher | [training_config_diff.json](../../../outputs/round5/phase6_1/provenance/reacher/training_config_diff.json) | [phase6_1_training_code_identity.json](../../../outputs/round5/phase6_1/training/fm_b_seed3072_reacher/phase6_1_training_code_identity.json) | [phase6_1_training_sources.zip](../../../outputs/round5/phase6_1/training/fm_b_seed3072_reacher/phase6_1_training_sources.zip) |

评测代码逐条件兼容性检查：
- pusht：20 个新评测条件的共享运行时代码逐文件哈希一致（9 个文件）；历史复用 2 行由 `source_refs` 审计。
- reacher：22 个新评测条件的共享运行时代码逐文件哈希一致（9 个文件）；历史复用 0 行由 `source_refs` 审计。

兼容性复核中发现初始 Regression 矩阵与 FM 矩阵的运行时代码哈希不同。初始 Regression 的 44 个结果目录及早期 Reacher FM P0-none 单元已逐文件计算 SHA256 后归档；最终版在统一运行时代码下重跑了 Regression 的 42 个新评测条件，并重跑该 Reacher FM 单元。两个兼容的 PushT 历史行继续由 `source_refs` 引用。归档清单：[运行时身份复核](../../../outputs/round5/phase6_1/archive/pre_runtime_code_mismatch_20261006_1056/manifest.json)。归档的旧结果不计入本报告的 4,300 个有效新增 episode 数。

## 接口 smoke 与旧模型兼容性

CPU batch=2 smoke 全部通过：legacy checkpoint strict-load、共享参数同 seed 对齐、future-token prefix 因果、goal latent 不进入 B、τ 与 source timestep 独立、K=2 的真实前向计数、对动作/A/未来目标的梯度路径，以及 Regression-B 目标梯度不变。旧 PushT/Reacher checkpoint 的 SHA256 与本轮基线配置一致。Smoke 中合成模型的 endpoint 数值不作为训练或任务结果。

Smoke 原件：[cpu_smoke_final.json](../../../outputs/round5/phase6_1/smoke/cpu_smoke_final.json)（SHA256 `bc35ad9183af23c646732660489b7c100abf8e36be18abfefa7939f848503d9c`）。相关回归与协议套件的 11 个 unittest 模块共 75 项测试通过。

## 固定候选池诊断

Regression-B 的原始 A actor 为两个 scorer 共同生成 64 候选；两个 scorer 分别用各自 encoder 编码同一输入，随后用相同的 simulator replay outcome 计算排名、top-1 regret 和动作效果。候选真值评估额外执行 3,200 个 25-step episode/任务，计入独立诊断工作量。

| 任务 | Regression top-1 成功率 | FM top-1 成功率 | Regression 平均物理 regret | FM 平均物理 regret | 配对 regret 差区间 |
|---|---:|---:|---:|---:|---:|
| pusht | 0.100 | 0.080 | 12.39490 | 11.96661 | [-2.79540, +2.31403] |
| reacher | 0.560 | 0.340 | 0.03858 | 0.05425 | [+0.00557, +0.04868] |

候选排名校准图（每个状态内分别转为百分位，避免横比两模型的绝对 latent cost）：

![PushT fixed-pool ranking](../../../outputs/round5/phase6_1/diagnostics/fixed_pool_ranking/pusht/cost_vs_true_distance.png)

![Reacher fixed-pool ranking](../../../outputs/round5/phase6_1/diagnostics/fixed_pool_ranking/reacher/cost_vs_true_distance.png)

## 固定输入推理计时

使用真实 cohort 起始观测，batch=1/50，各 10 次预热与 50 次同步 FP32 规划调用；环境 stepping 不计入。逐条件记录 p50/p95、吞吐、A/B 网络前向、candidate-level guidance 梯度评估数、实际 autograd 调用数和峰值显存。FM 的 B 前向数按 Euler 每个积分网络调用计数。

计时落盘审计发现 4 条初始 FM 记录与约 20 GiB 的外部 GPU 占用重叠；这些记录保留在归档中并在 GPU5/7 重测。下表和 88 条正式记录只使用重测结果；归档清单记录原始 SHA256、GPU 前后快照及原因：[受外部 GPU 负载影响的记录](../../../outputs/round5/phase6_1/analysis/timing_conditions/archive_concurrent_gpu_load_20261006_0925/manifest.json)。最终 88 条记录均通过样本数、计时值与身份校验。

| 任务 | 方法 / guidance | Regression p50/p95 (B=1) | FM p50/p95 (B=1) | Regression p50/p95 (B=50) | FM p50/p95 (B=50) |
|---|---|---:|---:|---:|---:|
| pusht | P0-none | 0.2472/0.3995s | 0.0488/0.0738s | 12.3376/18.3432s | 0.0979/0.1025s |
| pusht | P0-GF-L | 0.4015/0.4748s | 0.0545/0.0568s | 18.3102/34.2623s | 0.1451/0.2890s |
| pusht | P0-PO-L | 0.8872/1.1702s | 0.0446/0.1626s | 29.9243/36.6723s | 0.1324/0.3016s |
| pusht | P1-none | 1.2398/1.9134s | 0.6246/0.8457s | 39.0619/50.2342s | 15.0708/20.8161s |
| pusht | P2-none | 1.0624/1.7295s | 0.3525/0.4678s | 35.9743/45.9060s | 17.5610/27.3851s |
| pusht | P2-GF-L | 1.0662/1.6011s | 0.3978/0.5294s | 36.7320/45.1188s | 23.3974/29.3496s |
| pusht | P2-PO-L | 0.8995/1.5568s | 0.7595/0.9618s | 27.9784/47.2547s | 23.4716/26.3205s |
| pusht | P3-none | 0.4325/0.5668s | 0.3126/0.4396s | 16.6105/19.0030s | 4.4490/12.7293s |
| pusht | P3-GF-L | 0.4602/0.6584s | 0.1456/0.1985s | 14.7934/49.0430s | 1.3772/3.6766s |
| pusht | P3-PO-L | 0.2497/0.3463s | 0.1131/0.1587s | 24.1138/34.1779s | 0.8816/2.3534s |
| pusht | P3-PO-refine-L | 0.4035/0.5333s | 0.1079/0.1298s | 23.8848/34.7454s | 0.5534/1.6226s |
| reacher | P0-none | 0.3712/0.4829s | 0.0129/0.0136s | 28.7165/35.1402s | 0.5132/3.4902s |
| reacher | P0-GF-L | 0.9133/1.1386s | 0.0574/0.1419s | 24.4142/33.4169s | 0.3035/0.8133s |
| reacher | P0-PO-L | 0.6836/0.9777s | 0.0943/0.1777s | 18.4762/25.8159s | 0.2999/1.9403s |
| reacher | P1-none | 0.8777/1.6316s | 0.2992/0.4651s | 36.4504/44.9905s | 12.5354/18.6934s |
| reacher | P2-none | 0.9019/1.4839s | 0.3141/0.6256s | 40.2901/47.8143s | 10.7675/13.0176s |
| reacher | P2-GF-L | 0.7631/1.5545s | 0.3772/0.5653s | 25.7256/32.2102s | 11.8481/26.2840s |
| reacher | P2-PO-L | 0.7158/1.1287s | 0.3792/0.5453s | 36.6491/47.9379s | 23.5688/27.5272s |
| reacher | P3-none | 0.5942/0.7365s | 0.1362/0.2925s | 20.8653/28.3339s | 6.4771/11.9051s |
| reacher | P3-GF-L | 0.5449/0.7009s | 0.3211/0.4809s | 20.0990/28.2589s | 7.4331/11.3483s |
| reacher | P3-PO-L | 0.3966/0.9113s | 0.1891/0.3048s | 19.2150/26.4503s | 7.7626/12.1691s |
| reacher | P3-PO-refine-L | 0.4395/0.7394s | 0.2416/0.3841s | 22.1178/27.1262s | 10.7806/13.2280s |

## 解释边界

两臂各使用一个训练 seed（3072）和一个评测 seed（42），因此这是预实验，不估计训练随机性。A、encoder 与 B 联合更新，观察到的差异属于替换 B 目标后的整体系统效应。Velocity loss 与 Regression-B latent MSE 量纲不同，不直接比较数值。

GF/PO 预算固定为轻量档：GF 两个动作 Euler 步各一次更新，PO 在生成后两次更新，P3-PO-refine 只优化已选候选。A 与 B 的 flow 步数、实际网络前向/反向数和规划延迟见逐条件 JSON/trace。

## 逐条件结果

完整的两臂成功率、Wilson 区间、配对 bootstrap、训练身份与损失、88 条固定输入计时记录、固定候选池真值与 source_refs 保存在 `outputs/round5/phase6_1/analysis/summary.json`、CSV 与各条件 `result.json`。逐 episode 闭环 trace 与固定候选的 simulator trace 均按 episode 保存。
