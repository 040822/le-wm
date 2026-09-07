# Fast-LeWAM 第二轮 Stage-B 优化实验报告

> 汇总日期：2026-08-15
> 状态：§3.1、已完成的 §3.2/§4.1，以及 §3.4 Reacher + Push-T 均完成验收
> 主口径：预定 epoch 10 / final checkpoint，50 个固定评估 episode；主结论使用 final，不根据测试成功率挑选中间 checkpoint
> 后续章节：§3.2/§4.1 结果见 §10；§3.4 serial one-step control 与 simulator-grounded 归因见 §11

## 1. 本轮目标与范围

本轮执行计划 `docs/plan/fast_lewam_stage_b_optimization.md` §3.1「同量级扩容的 E1 Stage-B-only」及其扩展：

- 扩容配置：`latent_head_dim`（model_dim）192→384，`mlp_dim` 768→1024；`depth=6`、`heads=6`、`frameskip=5`、`action_horizon=5` 保持不变。
- 实验矩阵：E1-384（`train_mode=stage_b`）与 E3-384（`train_mode=stage_ab`，`latent_action_mix_epochs=1e9`，Stage B 始终使用专家动作）在 Reacher 与 Push-T 上并行训练；第一轮 192 维结果（E1-192 / E3-192）与 E0 LeWM 作为对照。
- 按用户指示，本轮同时运行 E1 与 E3，且任务范围改为 Reacher + Push-T；**Cube 与 TwoRoom 未运行**，偏离计划 §3.1 的 Cube-first 顺序。
- 单卡训练（GPU 0–3 各一卡），global `batch_size=128`，`lr=5e-5` 不做 batch-size scaling，`seed=3072`，`max_epochs=10`。

运行目录：

- `outputs/fast_lewam/reacher/0811_e1_stage_b_only_384/`
- `outputs/fast_lewam/reacher/0811_e3_expert_actions_384/`
- `outputs/fast_lewam/pusht/0811_e1_stage_b_only_384/`
- `outputs/fast_lewam/pusht/0811_e3_expert_actions_384/`

## 2. 实验定义

| 实验 | 训练 / 推理设置 | 主要问题 |
|---|---|---|
| E0 | 本地 LeWM 复现，Stage-B planning（batch 128） | 同协议 planner 基线 |
| E1-192 | 第一轮 Fast Stage-B only；真实动作、latent loss；192/768 | parallel Stage-B 本身是否有效（对照） |
| E3-192 | 第一轮 Stage-A + Stage-B；Stage-B 始终使用专家动作；192/768 | action auxiliary supervision 是否增强 WM（对照） |
| **E1-384** | E1 同量级扩容；`latent_head_dim=384`、`mlp_dim=1024` | 容量不足是否为 Fast Stage-B 性能缺口的主因 |
| **E3-384** | E3 同量级扩容；配置同 E3-192，仅扩容量 | 扩容对「A 辅助监督 + B」组合的增益 |

## 3. 统一协议与统计边界

- 主结果均为 50 个评估 episode 的成功率，最小步长为 2 个百分点。
- 横向比较使用固定评估 seed（42）、episode、起点、goal offset（25）和 CEM 预算；与第一轮 192 维结果同协议，可直接比较。
- 主结论使用 epoch 10 / final，不根据测试成功率选择中间 checkpoint。
- 本轮全部为单训练种子 `3072`。
- 完成口径：训练与 epoch eval 均在 tmux 会话 `lewam_r2_scale`（GPU 0–3 各一卡）中运行；GPU 4–7 未使用。

## 4. 完成度

| 任务 | E0 | E1-192 | E3-192 | E1-384 | E3-384 |
|---|---|---|---|---|---|
| Push-T | 完成 | 完成 | 完成 | 完成，final=84 | 完成，final=92 |
| Reacher | 完成 | 完成 | 完成 | 完成，final=82 | 完成，final=84 |

验收记录：4 个 run 的 final JSON 均为 `status=ok`、50 个 episode 完整、`fast_lewam_weights_epoch_10.pt` 与 `last.ckpt` 齐全；日志无 OOM 或任务失败。训练结束后的 `asyncio`/`TeeStream` atexit warning 不影响已保存结果（同第一轮）。

## 5. 结果统计

### 5.1 Final 主结果总表

`A` 为 Stage-A，`A-shuf` 为 shuffled-goal Stage-A，`B` 为随机初始化 CEM Stage-B。

| 任务 | E0 B | E1-192 B | E1-384 B | E3-192 A / A-shuf / B | E3-384 A / A-shuf / B |
|---|---:|---:|---:|---:|---:|
| Push-T | **98** | 88 | 84 | 96 / 4 / 90 | 96 / 8 / 92 |
| Reacher | 72 | 82 | 82 | 76 / 4 / 82 | 76 / 12 / **84** |

### 5.2 Push-T 训练轨迹

| 实验 | 分支 | e2 | e4 | e6 | e8 | final |
|---|---|---:|---:|---:|---:|---:|
| E0 | B | — | — | — | — | **98** |
| E1-192 | B | 76 | 82 | 86 | 86 | 88 |
| E1-384 | B | 64 | 80 | 88 | 88 | **84** |
| E3-192 | A | 66 | 88 | 92 | 94 | 96 |
| E3-192 | A-shuf | 4 | 6 | 4 | 6 | 4 |
| E3-192 | B | 52 | 78 | 82 | 90 | 90 |
| E3-384 | A | 84 | 96 | 96 | 96 | 96 |
| E3-384 | A-shuf | 4 | 6 | 6 | 8 | 8 |
| E3-384 | B | 68 | 70 | 94 | 92 | **92** |

E1-384 final 84 相对 E1-192（88）**退化 4 点**，且 e8 峰值 88 回落至 final 84，扩容对纯 Stage-B 无收益。E3-384 的 B 分支 e6 达峰 94 后稳定在 92，相对 E3-192（90）提升 2 点；A 分支与 192 持平（96），A-shuf 略升（8 vs 4）但保持低值，goal conditioning 正常。

### 5.3 Reacher 训练轨迹

| 实验 | 分支 | e2 | e4 | e6 | e8 | final |
|---|---|---:|---:|---:|---:|---:|
| E0 | B | — | — | — | — | 72 |
| E1-192 | B | 38 | 50 | 82 | 72 | 82 |
| E1-384 | B | 62 | 58 | 78 | 88 | **82** |
| E3-192 | A | 32 | 42 | 78 | 64 | 76 |
| E3-192 | A-shuf | 12 | 8 | 6 | 4 | 4 |
| E3-192 | B | 36 | 40 | 88 | 82 | 82 |
| E3-384 | A | 44 | 52 | 62 | 80 | **76** |
| E3-384 | A-shuf | 10 | 6 | 12 | 10 | **12** |
| E3-384 | B | 60 | 72 | 76 | 78 | **84** |

E1-384 final 82 与 E1-192 持平，e8 峰值 88 回落至 82，再次印证不能按测试峰值选 checkpoint；Reacher 上 192 维已达 82（超过 E0=72），扩容没有额外贡献。E3-384 的 B 分支从 e2 起持续爬升（60→72→76→78→**84**），final 为全轮最高、也是本轮唯一超过 E0（72）且高于 E1/E3-192（82）的 Reacher Stage-B 结果；A 分支 e8 达峰 80 后回落至 76，与 E3-192 持平；A-shuf final 12 为各配置最高，但整体仍为低值。

## 6. 工程指标

| 指标 | 192/768（第一轮） | 384/1024（本轮） |
|---|---|---|
| 参数量（总 / 非视觉） | 10.12M / 4.62M | 20.06M / **14.56M** |
| 训练吞吐（Reacher） | ~3.7 it/s | ~2.4–3.7 it/s（波动，后期恢复 3.7） |
| 训练吞吐（Push-T） | ~2.4–3.1 it/s | ~2.8–3.6 it/s |
| 训练显存峰值 | — | ~21 GB/卡 |
| final stage_b eval 耗时（50 eps） | — | Reacher E1 1019 s；Reacher E3 663 s；Push-T E1 3275 s；Push-T E3 733 s |
| 整轮 wall-clock（含训练时 eval） | — | ~16–19 h（E1 两任务 / Push-T E3）；Reacher E3 ~23 h |

- 非视觉参数 14.56M 相对 LeWM（~11.74M）高约 24%，符合计划 §3.1「同一量级」界定；384 维下每 head 64 维。
- 训练吞吐早期偏低（~2.4 it/s）与磁盘缓存预热及并发 eval 相关，后期回升至 3.7 it/s，与 192 相当；不视为显著工程退化。
- 本轮训练时 epoch eval 带来的 wall-clock 开销约 30–55%（4-stage eval 窗口 50–120 分钟 × 5 窗口）。报告收口后已把 `config/train/fast_lewam.yaml` 的 `epoch_eval.enabled` 默认改为 `false`，后续 run 纯训练、训练后离线评估（`eval_fast_lewam.py`），可在下一轮训练的同时并行评估上一轮结果。

## 7. 结论（基于 final 的正式口径）

1. **容量不足不是纯 Stage-B 性能缺口的主因（E1 口径明确）。** E1-384 在 Push-T 上 final 84，反而比 E1-192（88）退化 4 点，且低于 E0=98 达 14 点；Reacher 上与 E1-192 持平（82）。同量级扩容（非视觉参数 ×3.15）没有缩小与 LeWM 的差距。
2. **扩容对「A 辅助监督 + B」组合在 Reacher 上产生本轮唯一超过 LeWM 的 B 结果，Push-T 上收益有限。** Reacher E3-384 final 84，超过 E0（72）12 点、超过 E3-192/E1-192（82）2 点；Push-T E3-384 final 92，仅比 E3-192（90）高 2 点，仍低于 E0=98 达 6 点。A 分支两任务均与 192 持平（96 / 76）。
3. **E3-384 满足计划 §3.1 的复核条件之一，进入人工复核。** 两任务平均成功率 88（92+84）/2 超过 LeWM 平均 85；同时 Reacher 相对 LeWM 超 12 个百分点，但 Push-T 相对 LeWM 退化 6 点，是否属于「另一任务没有明显退化」需人工判断。复核决定是否需要补训练种子（3073/3074）验证该信号。
4. **E1-384 不满足任何复核条件**（两任务平均 83 < 85；Reacher +10 伴随 Push-T −14 明显退化），按计划 §3.1 该方向暂停。
5. **goal conditioning 保持正常。** 所有 A-shuf 均维持低值（4–12%），扩容未引入 goal 泄漏或任务先验依赖增强；Reacher E3-384 A-shuf（12）为本轮最高，需在补种子时留意是否上升。
6. **中期峰值不可作为 checkpoint 依据再次得到验证。** Push-T E1-384 e8=88 → final 84；Reacher E1-384 e8=88 → final 82；Push-T E3-384 e6=94 → final 92。反向案例：Reacher E3-384 final 84 为全轮最高，最终 epoch 无回落。

## 8. 运行事件记录

- **E3 启动崩溃与修复**：`latent_action_mix_epochs=.inf` 被 Hydra override parser 解析为字符串 `'.inf'`，触发 `mix_epochs <= 0` 的 `TypeError`。已改用 `latent_action_mix_epochs=1e9`（Hydra 解析为 float；10 epoch 内 mix 概率恒 0，语义与 `.inf` 等价）。
- **训练时 eval 开销与策略变更**：多 stage 的 epoch eval 每窗口 50–120 分钟，整轮额外 30–55% wall-clock；已确认 eval 耗时主要由 50 episode × 2 replan × 9000 候选（batch_size=1）的 CEM 前向与多 stage 重复 rollout 构成。收口后默认关闭训练时 eval（见 §6）。
- smoke 验证（fast_dev_run，batch 2，真实 384/1024 维度）通过，loss 有限，checkpoint 234 MB 与约 2× 参数量一致。

## 9. 后续安排

按计划 §3.1 决策门：

1. **E3-384 进入人工复核**（满足「两任务平均超过 LeWM」；Reacher 单项 +12 而 Push-T −6 待判）。若确认值得继续，补训练种子 3073/3074 验证 Reacher 84 与 Push-T 92 的稳定性；期间同步检查 Reacher A-shuf（12）是否继续上升。
2. **E1 扩容方向暂停**，优先进入 §3.4 serial one-step control：用与 parallel Stage-B 匹配的 backbone 训练真实 `(z_i, a_i) -> z_{i+1}` 一步转移、推理时递归 rollout 5 次，归因 causal-prefix 并行预测是否为性能差距主因。
3. **同步补充 §2.1/§2.3 诊断**：在 E1-384/E3-384 final checkpoint 上运行 simulator-grounded 排序诊断（Spearman / top-k recall / regret）与 E6 warm-start 对比，确认瓶颈在 Stage-B 排序还是 CEM；排序诊断可在下一轮训练的同时用离线入口并行执行。
4. **Cube 扩容实验**：E1 扩容已在两个任务上确认无收益；若后续要跑 Cube，建议只跑 E3-384（该方向信号更积极），并按新工作流关闭训练时 eval、final 后离线评估。

## 10. 0812 后续单因素实验：§3.2 与 §4.1

### 10.1 实验范围与验收状态

该轮固定使用 192/768、global batch 128、seed 3072、10 epochs。训练时关闭 epoch eval，完成后离线评估 e2/e4/e6/e8/e10，每项使用固定 seed 42 的同一批 50 个 episode。

| 计划项 | 任务 | 目标开关 | 对照保持 | 状态 |
|---|---|---|---|---|
| §3.2 E4 | Cube | `stage_b_timestep_mode=clean_action` | `token_encoding=legacy` | 训练中 |
| §3.2 E4 | Reacher | `stage_b_timestep_mode=clean_action` | `token_encoding=legacy` | 完成 |
| §4.1 E3 | Reacher | `token_encoding=physical_time_type` | Stage-B timestep legacy | 完成 |
| §4.1 E3 | Push-T | `token_encoding=physical_time_type` | Stage-B timestep legacy | 完成 |

三个已完成 run 均有 epoch-10 weights、`last.ckpt`、validation diagnostics 和逐 episode `result.json`。所有 final JSON 为 `status=ok`，日志无 OOM、NaN 或 traceback。

### 10.2 §3.2 Reacher：Stage-B condition 固定 clean-action timestep

| 配置 | e2 | e4 | e6 | e8 | final |
|---|---:|---:|---:|---:|---:|
| 原 E4 Stage B | 20 | **84** | **76** | **86** | **80** |
| §3.2 clean-action timestep | **40** | 50 | 66 | 84 | 78 |

§3.2 仅在 e2 提前收敛，e4 之后均未超过原 E4。final 78 比原 E4 低 2 点，e8 峰值 84 也低于原 E4 的 86，因此 Reacher 不支持采用该改动。

该 run 峰值显存 35,245 MiB，训练加 final eval 用时 64,964 秒。final predicted terminal MSE 为 0.0231，expert preference accuracy 为 80.44%，top-1 为 51.37%。

### 10.3 §4.1 Reacher：learned physical-time/type

| 分支 | e2 | e4 | e6 | e8 | final | 原 E3 final |
|---|---:|---:|---:|---:|---:|---:|
| A | 34 | 56 | 62 | 68 | 74 | **76** |
| A-shuf | 10 | 8 | 6 | 8 | 6 | **4** |
| B | 36 | 80 | **88** | **88** | 74 | **82** |

原 E3 Stage-B 轨迹为 36→40→88→82→82。§4.1 在 e4 提升 40 点、e8 提升 6 点，但 e6 持平，final 反而低 8 点；配对 e4 cohort 中有 26 个失败转成功、6 个成功转失败。

Stage-A final 低 2 点，A-shuf 高 2 点，未显示更强的 goal conditioning。该 run 峰值显存 34,859 MiB，总用时 65,024 秒。

### 10.4 §4.1 Push-T：learned physical-time/type

| 分支 | e2 | e4 | e6 | e8 | final | 原 E3 final |
|---|---:|---:|---:|---:|---:|---:|
| A | 78 | 78 | 90 | **96** | 92 | **96** |
| A-shuf | 12 | 14 | 10 | 12 | 14 | **4** |
| B | 78 | 76 | 90 | **94** | 90 | 90 |

原 E3 Stage-B 轨迹为 52→78→82→90→90。§4.1 在 e2/e6/e8 分别高 26/8/4 点，但 final 持平；A final 低 4 点，A-shuf 从 4 升至 14，说明正确 goal 对 actor 的约束变弱。

该 run 峰值显存 24,584 MiB，总用时 45,351 秒。final clean/predicted terminal MSE 为 0.0297/1.2965，preference accuracy 为 90.33%，top-1 为 73.73%。

### 10.5 Epoch-10 复跑核验

为排除评测随机波动，三个已完成 run 的 epoch 10 均在原 GPU 上重新评估。复跑使用同一 checkpoint、固定 50-episode cohort 和相同 stage 配置。

| 实验 | 分支 | 首次 | 复跑 | 逐 episode 翻转 |
|---|---|---:|---:|---:|
| Reacher §3.2 | B | 78 | 78 | 0/50 |
| Reacher §4.1 | A / A-shuf / B | 74 / 6 / 74 | 74 / 6 / 74 | 0/50 |
| Push-T §4.1 | A / A-shuf / B | 92 / 14 / 90 | 92 / 14 / 90 | 0/50 |

所有复跑均为 `status=ok`。成功率和每个 episode 的成功/失败标签完全一致，因此 e8→e10 的回落来自 checkpoint 本身，不是一次评测的随机波动。

### 10.6 Validation 与环境成功率不一致

Reacher §3.2 从 e8 到 e10 的 predicted terminal MSE 从 0.0272 降至 0.0231，preference accuracy 从 79.85% 升至 80.44%，但 Stage-B success 从 84 降至 78。

Reacher §4.1 的 preference accuracy 从 81.28% 升至 81.84%，top-1 从 53.13% 升至 53.91%，但 Stage-B success 从 88 降至 74。

Push-T §4.1 的 preference accuracy 从 89.81% 升至 90.33%，top-1 从 72.36% 升至 73.73%，但 Stage-B success 从 94 降至 90。

这些结果表明平均 latent MSE 和现有 expert ranking diagnostics 不能可靠选择 planner checkpoint。CEM 对候选局部排序敏感，而环境 success 是阈值指标，后期离线指标改善仍可能使成功候选排名下降。

### 10.7 当前结论与决策

1. **§3.2 暂不采用。** Reacher final 与中后期轨迹均未超过原 E4；待 Cube 完成后再关闭该实验项，不因单个任务提前终止仍在运行的 Cube。
2. **§4.1 保留为可选开关，不设为默认。** 两个任务均出现更快的中期 Stage-B 收敛，但 final 没有改善：Push-T 持平，Reacher 退化 8 点。
3. **暂不启动 §4.2。** Push-T 的 A-shuf 明显升高，且 §4.1 尚无 final 收益；当前证据不足以继续 learned→sinusoidal position。
4. **不按测试成功率选择 e8。** 中间 checkpoint 只用于分析训练动态；若要 early stopping，应新增与真实 planning 排序一致的 validation cohort，而不是事后查看环境测试成功率。
5. **优先补 simulator-grounded ranking。** 需要比较 e8/e10 的 Spearman、top-k recall、regret 和 CEM 候选翻转，解释离线 diagnostics 改善但环境 success 回落的原因。

## 11. §3.4 Serial one-step causal control

### 11.1 目标与实验定义

该实验检验 parallel multi-step latent dynamics 是否是 Stage-B 排序误差的主因。对照组保持原 parallel Stage-B；实验组改为 serial one-step dynamics。

serial 模型训练时以真实 `z_i` 和动作 `a_i` teacher-force 下一步 `z_{i+1}`。评估时不读取未来真实 latent，而是从预测结果递归 rollout，避免训练目标泄漏到规划结果。

两组均使用 192/768 backbone、seed 3072、10 epochs，并保持 canonical CEM：300 samples、30 iterations、top-k 30、seed 42、50 个固定 episode。

诊断 manifest 沿用通用 `e8`/`e10` 标签，但本节两者均为 epoch 10：`e8` 表示 parallel control，`e10` 表示 serial one-step intervention。

运行目录：

- Reacher parallel：`outputs/fast_lewam/reacher/0809_e1_stage_b_only_seed3072_tmux/`
- Reacher serial：`outputs/fast_lewam/reacher/0813_s34_e1_serial_one_step_192/`
- Push-T parallel：`outputs/fast_lewam/pusht/0802_e1_stage_b_only/`
- Push-T serial：`outputs/fast_lewam/pusht/0813_s34_e1_serial_one_step_192/`

### 11.2 训练轨迹与环境成功率

| 任务 / 配置 | e2 | e4 | e6 | e8 | final |
|---|---:|---:|---:|---:|---:|
| Reacher serial | 26 | 52 | 64 | 70 | 72 |
| Push-T serial | 64 | 86 | 92 | 92 | 96 |

为避免历史 reference 漂移，parallel final 使用独立 canonical replay。Reacher replay 为 84%，Push-T replay 为 86%；两者随后均被 instrumented trace 逐 episode 精确复现。

| 任务 | parallel replay | serial final | 变化 | 回落 | 改善 | 稳定成功 | 稳定失败 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reacher | 84 | 72 | **−12** | 10 | 4 | 32 | 4 |
| Push-T | 86 | 96 | **+10** | 2 | 7 | 41 | 0 |

同一改动在 Push-T 上提升 10 点，却在 Reacher 上退化 12 点。该差异远大于 50-episode 评估的 2 点最小步长，不能解释为单个 episode 的随机波动。

### 11.3 Simulator-grounded 诊断协议与验收

两组先严格复现完整 50-episode success vector。Reacher 与 Push-T 的 parallel/serial trace 均为 0 mismatch，任何 reference mismatch 都在 grounding 前被门禁阻止。

grounding 在相同 simulator 状态下合并两组候选，并由两个模型交叉打分。replan 0 使用共享初始状态；replan 1 分别恢复两组实际执行前缀形成的状态。

每个 panel 保存 iteration 0/5/29 的 elite、固定 non-elite、mean、iteration-best、best-ever 与 expert anchor。需要时扩展到两组完整候选集合。

simulator 对同一共享 panel 只 rollout 一次，物理距离与成功标签由两个模型共享。报告不跨任务平均 raw physical cost，只在 pair 和 episode 类别内计算 `serial − parallel`。

| 任务 | trace | grounded slots | score artifacts | acceptance errors | pair status |
|---|---|---:|---:|---:|---|
| Reacher | 0 mismatch | 50/50 | 436 | 0 | `ok` |
| Push-T | 0 mismatch | 50/50 | 347 | 0 | `ok` |

最终全局 `summary.json` 为 `status=ok`、`missing_pairs=[]`。所有指标有限，shared-panel candidate hash 与持久化 score artifact 一致。

### 11.4 Reacher grounding 结果

下表为 serial − parallel。Spearman、recall 为正表示改善；regret 与 latent MSE 为负表示改善。

| 类别 | slots | panels | Δ Spearman | Δ top-30 recall | Δ success recall@30 | Δ regret | Δ latent MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 回落 | 10 | 72 | **−0.1897** | −0.0745 | −0.0204 | **+0.0875** | +0.0789 |
| 改善对照 | 4 | 30 | −0.0948 | −0.0678 | −0.0363 | +0.0665 | +0.0446 |
| 稳定成功 | 32 | 195 | −0.1171 | −0.0542 | −0.0170 | +0.0523 | +0.0516 |
| 稳定失败 | 4 | 36 | −0.1496 | −0.0667 | +0.0501 | +0.0284 | +0.0560 |

serial 在 Reacher 的所有类别中都降低 predicted↔physical Spearman，并提高 best-candidate regret 和 predicted terminal latent MSE。退化不是只集中在最终失败 episode。

回落组的 true-terminal-latent↔physical Spearman 仅变化 −0.0097，但 predicted↔true-terminal-latent Spearman 下降 0.1566。

因此 Reacher 的主要问题更接近 serial dynamics prediction 与候选排序退化，而不是 latent metric 突然失去对物理目标的表达能力。

归因计数允许一个 slot 同时拥有多个标签，因此计数不可直接相加为 episode 数。

| 模型 | coverage | ranking | dynamics | latent metric | CEM exploitation | replan regression | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| parallel | 0 | 8 | 3 | 0 | 2 | 3 | 5 |
| serial | 2 | 14 | 9 | 1 | 2 | 2 | 12 |

serial 的 ranking failure 从 8 增至 14，dynamics error 从 3 增至 9。Reacher pair 决策为 `causal_prefix_not_supported`。

### 11.5 Push-T grounding 结果

| 类别 | slots | panels | Δ Spearman | Δ top-30 recall | Δ success recall@30 | Δ regret | Δ latent MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 回落 | 2 | 12 | −0.0549 | −0.0556 | −0.2550 | **−7.5277** | +0.0729 |
| 改善对照 | 7 | 45 | +0.0321 | +0.0333 | +0.0226 | **+7.8086** | +0.0336 |
| 稳定成功 | 41 | 141 | −0.0555 | −0.0206 | −0.0241 | **+8.7465** | +0.0397 |
| 稳定失败 | 0 | 0 | — | — | — | — | — |

Push-T 的环境成功率提高 10 点，但 shared-panel 排序没有一致改善。改善对照仅有轻微 Spearman/recall 增益，同时 regret 与 latent MSE 仍恶化。

稳定成功组占 41/50，却出现 Spearman、recall 与 regret 的一致小幅退化。成功率提升不能被解释为“serial 全面改善了共享候选排序”。

| 模型 | coverage | ranking | dynamics | latent metric | CEM exploitation | replan regression | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| parallel | 7 | 4 | 4 | 2 | 3 | 6 | 7 |
| serial | 1 | 2 | 1 | 2 | 0 | 0 | 2 |

serial 的失败归因总量明显减少，特别是 coverage、dynamics、CEM exploitation 和 replan regression；但其 shared-panel 排序增益不稳定，因此 pair 决策为 `serial_control_mixed`。

### 11.6 工程事件与可恢复性

历史 parallel reference 在当前 canonical 环境下发生漂移。Push-T 旧 reference 与 trace 仅在 slot 42 不同；独立 replay 得到 43/50，并与后续 instrumented trace 完全一致。

Reacher 同样使用独立 replay reference。旧结果没有被覆盖，stale trace 被单独保留；grounding 只在新 reference 取得 0 mismatch 后继续。

GPU 异常调查发现，仅设置 `CUDA_VISIBLE_DEVICES` 不能约束 dm_control EGL 的物理设备选择。已新增 `MUJOCO_EGL_DEVICE_ID` 绑定与 GPU0–3 guard，GPU4–7 被显式拒绝。

Reacher grounding 产生约 86 GiB 可重建 simulator cache。经明确授权删除后，436 个最终 scores、50 个 slot JSON、trace 和 summary 均保持完整。

Push-T 当前仍保留约 90 GiB simulator cache；最终 scores 约 257 MiB，slot JSON 约 2.9 MiB。删除 cache 不影响本报告，但会失去 simulator rollout 的快速续跑点。

manifest 更新曾使已验收 Reacher summary 的整文件哈希失效。现只允许忽略其他 pair 引起的 manifest hash 变化。

当前 pair 的 protocol、checkpoint、config、reference、候选/状态哈希和 slots 仍严格校验。

### 11.7 结论与决策

1. **不把 serial one-step 设为全局默认。** 它在 Push-T 上提升 10 点，却在 Reacher 上退化 12 点，缺乏任务泛化性。
2. **causal-prefix 不是统一主瓶颈。** Reacher 的 grounding 明确显示 dynamics、ranking、regret 和 latent MSE 同时恶化，全局决策为 `causal_prefix_not_supported`。
3. **Push-T 的正收益真实但机制混合。** strict replay 和 50-slot grounding 均通过，但 shared-panel 排序没有全面改善，不能把 +10 点全部归因于更准确的 one-step dynamics。
4. **后续调优应由 failure mode 触发。** coverage 主导时改 proposal/actor warm-start；ranking/dynamics 主导时改模型；CEM/replan 主导时改 best-ever、early-stop 或重规划协议。
5. **统一诊断框架可以保留，统一结构默认值暂不成立。** task-specific dynamics、latent objective 和 planner 配置应成为一等配置，而不是继续叠加全局 trick。
6. **Cube 是下一项关键判据。** 若 Cube 与 Push-T 同向，可检验“接触操作 vs 连续控制”的任务分界；若再次表现不同，则应完全转向 diagnosis-gated 配置。

本轮至此完成：Reacher 与 Push-T 的训练、canonical replay、strict trace、simulator grounding、pair summary 和 global summary 均已验收。
