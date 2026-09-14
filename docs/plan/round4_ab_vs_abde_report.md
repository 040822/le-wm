# R4-AB 与 R4-ABDE 对照报告

## 实验定义

本实验比较同一 `Round4FastLeWAM` 结构下的两种训练方式：R4-AB 只训练 A/B，R4-ABDE 同时训练 A/B/D/E。两者均使用 seed 3072 和 10 个 epoch。

R4-AB 是从头训练的四个任务 run，输出位于 `outputs/round4/ab_seed3072_<task>`。R4-ABDE 使用已完成的 epoch-10 结果作为固定对照。

Cube、Push-T、Reacher 的 R4-ABDE 首轮曾中断并进行了权重-only continuation，因此本比较是描述性对照，不能当作严格的训练因果实验。

## 评测协议

两组使用相同的 `round3_revised` cohort、评测 seed=42、dev 50 episodes 和 final 200 episodes。主指标是 P3：A 生成 64 条 action candidates，再由 B verifier 选优。

P0、P1、P2 用于辅助定位 A/B 能力变化。R4-AB 不运行 P4/P4-first，因为其 D/E 参数没有训练，无法解释 latent planner 结果。

## P3 主结果

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

因此，在当前 seed 3072 的闭环结果中，没有观察到 D/E 辅助训练给 P3 带来收益。这个结果支持把 AB 作为 P3 的简洁训练方案，但单个训练 seed 不足以证明稳定等价或稳定优于 ABDE。

## P0–P2 辅助结果

上一版只列出了四个任务的宏平均，下面展开列出 Cube、Push-T、Reacher 和 TwoRoom。差值均为 R4-ABDE 减 R4-AB，单位为百分点。

### 开发集（Dev）

| 任务 | P0 AB | P0 ABDE | P0 差值 | P1 AB | P1 ABDE | P1 差值 | P2 AB | P2 ABDE | P2 差值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cube | 100 | 100 | 0 | 40 | 58 | +18 | 100 | 98 | -2 |
| Push-T | 92 | 92 | 0 | 90 | 94 | +4 | 92 | 96 | +4 |
| Reacher | 70 | 74 | +4 | 74 | 86 | +12 | 62 | 68 | +6 |
| TwoRoom | 96 | 96 | 0 | 98 | 100 | +2 | 96 | 98 | +2 |
| 四任务平均 | 89.5 | 90.5 | +1.0 | 75.5 | 84.5 | +9.0 | 87.5 | 90.0 | +2.5 |

### 最终集（Final）

| 任务 | P0 AB | P0 ABDE | P0 差值 | P1 AB | P1 ABDE | P1 差值 | P2 AB | P2 ABDE | P2 差值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cube | 99.5 | 100 | +0.5 | 49 | 59.5 | +10.5 | 94.5 | 95.5 | +1.0 |
| Push-T | 92.5 | 89.5 | -3.0 | 86 | 86 | 0 | 93 | 92.5 | -0.5 |
| Reacher | 75.5 | 74 | -1.5 | 86 | 86.5 | +0.5 | 67 | 68.5 | +1.5 |
| TwoRoom | 96 | 94.5 | -1.5 | 98.5 | 97 | -1.5 | 98.5 | 98.5 | 0 |
| 四任务平均 | 90.875 | 89.5 | -1.375 | 79.875 | 82.25 | +2.375 | 88.25 | 88.75 | +0.5 |

ABDE 在 P1/P2 上的平均成功率较高，但 P0 final 下降，P3 final 持平。因此，D/E 联合训练对 A/B 的影响并不表现为所有推理模式都一致改善。

## 结论

当前最稳妥的结论是：P3 继续作为主要推理方法；在 P3 上，R4-ABDE 没有显示出相对 R4-AB 的实际收益。

D/E 可能改变 B/CEM 相关行为，但本轮数据不能区分这种差异来自辅助任务、训练随机性，还是 R4-ABDE 的 continuation provenance。

严格验证 D/E 是否帮助 P3，需要后续使用多个训练 seed，并让 AB 与 ABDE 具有一致的连续训练和 optimizer 状态。本轮不补跑额外 seed，也不把该单 seed 结果表述为因果证明。

## R4-AB checkpoint 哈希

| 任务 | checkpoint | SHA256 |
|---|---|---|
| Cube | `outputs/round4/ab_seed3072_cube/checkpoints/r4_ab_seed3072_weights_epoch_10.pt` | `2748151a2c43c7841246a5fb8ee1fd7915ebbc00bfc27aa1a32b0f9080d92956` |
| Push-T | `outputs/round4/ab_seed3072_pusht/checkpoints/r4_ab_seed3072_weights_epoch_10.pt` | `62d00096a34e9f0b6da4ceb6a3c61eb350701a7aade5f5e4d2c1ee528b06015e` |
| Reacher | `outputs/round4/ab_seed3072_reacher/checkpoints/r4_ab_seed3072_weights_epoch_10.pt` | `087991339c9499d10c1a58a28e72d07c7dfd819d5f072c67bfd0169acaa83553` |
| TwoRoom | `outputs/round4/ab_seed3072_tworoom/checkpoints/r4_ab_seed3072_weights_epoch_10.pt` | `48d1ed2520ec7142327969eacf1c37b144faec51cfd6c9ce72701a8421bf57c2` |

## 产物

- 比较 JSON：`outputs/round4/ab_vs_abde/analysis.json`
- 比较 CSV：`outputs/round4/ab_vs_abde/analysis.csv`
- 训练与评测配置：`config/round4/ab_control.json`
- R4-AB 评测目录：`outputs/round4/ab_seed3072_eval`
