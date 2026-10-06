# Shared DiT 加深影响归因分析（6 / 8 / 12 层）

## 目的与解释边界

本轮使用现有 epoch-10 checkpoint，不重新训练。候选池实验将动作生成与 Stage-B rerank 分开：同一 legacy50 起点和同一组 64 个初始噪声下，分别比较 6/8/12 层生成的物理候选，再让三个深度的 verifier 对每个候选池交叉评分。闭环部分补充 P0（直接动作生成）、P1（Stage-B/CEM）和 P3（动作候选加 Stage-B rerank）。

样本是固定 cohort 和单训练 seed 的配对结果。候选池的 oracle 是已采样的 64 个候选中的最好结果，只表示该池的上限，不代表部署策略能达到。bootstrap 区间按 state 重采样；它描述当前固定 cohort 的不确定性，不替代独立 seed。

## Checkpoint 与训练损失

| Task | Depth | 参数量 | Validation action loss（epoch 10） | Validation latent-prefix loss（epoch 10） | Checkpoint SHA256 |
|---|---:|---:|---:|---:|---|
| pusht | 6 | 10,638,164 | 0.414 | 0.030 | `62d00096a34e9f0b6da4ceb6a3c61eb350701a7aade5f5e4d2c1ee528b06015e` |
| pusht | 8 | 11,969,876 | 0.398 | 0.028 | `07f8cafd40560ab0bfaa4659058e7a7aa848358f95bd5b3b7f1bd4ac7210ceb7` |
| pusht | 12 | 14,633,300 | 0.382 | 0.024 | `77fded174fef263bc21879800cf585c5b6a3a0cb5ade03f710844e3f48db9d17` |
| reacher | 6 | 10,638,164 | 1.473 | 0.119 | `087991339c9499d10c1a58a28e72d07c7dfd819d5f072c67bfd0169acaa83553` |
| reacher | 8 | 11,969,876 | 1.472 | 0.119 | `dc4b7b5fc429a9b0116b275b87ac1adc050526596590f45f9cefc7924c95f8b6` |
| reacher | 12 | 14,633,300 | 1.470 | 0.119 | `b0142b6027b014630a15ff0955fda661de540f27212209a5320bd3816033cfaf` |

## 固定候选池：生成与 rerank 的独立贡献

每个 task 有 50 个状态、每个状态 64 个配对噪声候选。每个物理候选按相同的 25 个 primitive action replay，记录 success 和 normalized physical distance。下表按 generator × verifier 列出 verifier 选中候选的成功率、候选池 oracle 成功率、距离 regret 和成本排序相关性。

### pusht

| Generator | Verifier | Pool oracle success | Selected success | Random candidate success | Distance regret | Spearman(cost, distance) | Oracle best in predicted top-5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 6 | 0.180 [0.080, 0.300] | 0.160 [0.060, 0.260] | 0.108 [0.039, 0.193] | 0.222 [0.139, 0.323] | 0.191 [0.106, 0.279] | 0.040 [0.000, 0.100] |
| 6 | 8 | 0.180 [0.080, 0.300] | 0.140 [0.060, 0.240] | 0.108 [0.039, 0.189] | 0.236 [0.156, 0.329] | 0.204 [0.116, 0.290] | 0.140 [0.060, 0.240] |
| 6 | 12 | 0.180 [0.080, 0.300] | 0.140 [0.060, 0.240] | 0.108 [0.039, 0.190] | 0.414 [0.165, 0.843] | 0.145 [0.058, 0.233] | 0.080 [0.020, 0.160] |
| 8 | 6 | 0.960 [0.900, 1.000] | 0.140 [0.060, 0.240] | 0.133 [0.060, 0.220] | 3.046 [2.409, 3.709] | 0.155 [0.065, 0.245] | 0.120 [0.040, 0.220] |
| 8 | 8 | 0.960 [0.900, 1.000] | 0.120 [0.040, 0.220] | 0.133 [0.060, 0.222] | 3.118 [2.482, 3.779] | 0.196 [0.111, 0.285] | 0.080 [0.020, 0.160] |
| 8 | 12 | 0.960 [0.900, 1.000] | 0.120 [0.040, 0.220] | 0.133 [0.061, 0.219] | 3.189 [2.527, 3.879] | 0.222 [0.128, 0.312] | 0.080 [0.020, 0.160] |
| 12 | 6 | 0.400 [0.260, 0.540] | 0.120 [0.040, 0.220] | 0.135 [0.058, 0.227] | 1.764 [1.225, 2.333] | 0.116 [0.032, 0.198] | 0.100 [0.020, 0.180] |
| 12 | 8 | 0.400 [0.260, 0.540] | 0.160 [0.060, 0.260] | 0.135 [0.056, 0.224] | 1.729 [1.190, 2.312] | 0.110 [0.029, 0.187] | 0.080 [0.020, 0.160] |
| 12 | 12 | 0.400 [0.260, 0.540] | 0.140 [0.060, 0.240] | 0.135 [0.057, 0.227] | 1.661 [1.136, 2.237] | 0.160 [0.073, 0.241] | 0.100 [0.020, 0.200] |

候选生成摘要：

| Generator | Pairwise action RMS | Fixed candidate success | Mean physical distance |
|---:|---:|---:|---:|
| 6 | 0.3597 | 0.1084 | 3.9654 |
| 8 | 0.3450 | 0.1334 | 3.8445 |
| 12 | 0.3374 | 0.1353 | 3.8482 |

### reacher

| Generator | Verifier | Pool oracle success | Selected success | Random candidate success | Distance regret | Spearman(cost, distance) | Oracle best in predicted top-5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 6 | 1.000 [1.000, 1.000] | 0.640 [0.500, 0.780] | 0.545 [0.494, 0.595] | 1.452 [1.122, 1.813] | 0.255 [0.173, 0.336] | 0.120 [0.040, 0.220] |
| 6 | 8 | 1.000 [1.000, 1.000] | 0.680 [0.540, 0.800] | 0.545 [0.493, 0.596] | 1.400 [1.091, 1.742] | 0.241 [0.158, 0.324] | 0.040 [0.000, 0.100] |
| 6 | 12 | 1.000 [1.000, 1.000] | 0.720 [0.600, 0.840] | 0.545 [0.494, 0.595] | 1.291 [0.994, 1.633] | 0.239 [0.157, 0.319] | 0.140 [0.060, 0.240] |
| 8 | 6 | 1.000 [1.000, 1.000] | 0.600 [0.460, 0.740] | 0.546 [0.492, 0.599] | 1.370 [1.067, 1.709] | 0.252 [0.172, 0.331] | 0.060 [0.000, 0.140] |
| 8 | 8 | 1.000 [1.000, 1.000] | 0.540 [0.400, 0.680] | 0.546 [0.492, 0.600] | 1.632 [1.284, 2.011] | 0.243 [0.158, 0.325] | 0.020 [0.000, 0.060] |
| 8 | 12 | 1.000 [1.000, 1.000] | 0.520 [0.380, 0.660] | 0.546 [0.493, 0.599] | 1.510 [1.181, 1.875] | 0.240 [0.159, 0.320] | 0.020 [0.000, 0.060] |
| 12 | 6 | 1.000 [1.000, 1.000] | 0.680 [0.540, 0.800] | 0.562 [0.510, 0.613] | 1.344 [1.017, 1.698] | 0.237 [0.157, 0.317] | 0.060 [0.000, 0.140] |
| 12 | 8 | 1.000 [1.000, 1.000] | 0.580 [0.440, 0.720] | 0.562 [0.510, 0.613] | 1.517 [1.200, 1.867] | 0.223 [0.139, 0.304] | 0.080 [0.020, 0.160] |
| 12 | 12 | 1.000 [1.000, 1.000] | 0.640 [0.500, 0.780] | 0.562 [0.510, 0.611] | 1.432 [1.096, 1.810] | 0.224 [0.140, 0.304] | 0.140 [0.060, 0.240] |

候选生成摘要：

| Generator | Pairwise action RMS | Fixed candidate success | Mean physical distance |
|---:|---:|---:|---:|
| 6 | 1.3025 | 0.5453 | 2.0891 |
| 8 | 1.3029 | 0.5463 | 2.0914 |
| 12 | 1.3040 | 0.5616 | 2.0317 |

## 闭环成功率

### pusht — legacy (50 episodes)

| Depth | P0 success | P1 success | P3 diagonal success | P0 paired Δ vs 6L | P1 paired Δ vs 6L | P3 paired Δ vs 6L |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 94.0% | 90.0% | 96.0% | — | — | — |
| 8 | 94.0% | 90.0% | 96.0% | +0.0 pp | +0.0 pp | +0.0 pp |
| 12 | 96.0% | 84.0% | 100.0% | +2.0 pp | -6.0 pp | +4.0 pp |

P3 closed-loop success matrix (generator rows × verifier columns):

| Generator \ Verifier | 6 | 8 | 12 |
|---:|---:|---:|---:|
| 6 | 96.0% | 98.0% | 100.0% |
| 8 | 96.0% | 96.0% | 96.0% |
| 12 | 96.0% | 98.0% | 100.0% |

P3 verifier crossover, expressed as paired percentage-point change against the diagonal verifier at the same generator depth:

| Generator | Verifier change | Paired gain/loss | Net Δ (95% CI) | McNemar p |
|---:|---:|---:|---:|---:|
| 6 | 8 | 2/1 | +2.0 [-4.0, +8.0] pp | 1.000 |
| 6 | 12 | 2/0 | +4.0 [+0.0, +10.0] pp | 0.500 |
| 8 | 6 | 0/0 | +0.0 [+0.0, +0.0] pp | 1.000 |
| 8 | 12 | 0/0 | +0.0 [+0.0, +0.0] pp | 1.000 |
| 12 | 6 | 0/2 | -4.0 [-10.0, +0.0] pp | 0.500 |
| 12 | 8 | 0/1 | -2.0 [-6.0, +0.0] pp | 1.000 |

P3 generator crossover, expressed as paired percentage-point change against the diagonal generator at the same verifier depth:

| Verifier | Generator change | Paired gain/loss | Net Δ (95% CI) | McNemar p |
|---:|---:|---:|---:|---:|
| 6 | 8 | 1/1 | +0.0 [-6.0, +6.0] pp | 1.000 |
| 6 | 12 | 1/1 | +0.0 [-6.0, +6.0] pp | 1.000 |
| 8 | 6 | 1/0 | +2.0 [+0.0, +6.0] pp | 1.000 |
| 8 | 12 | 1/0 | +2.0 [+0.0, +6.0] pp | 1.000 |
| 12 | 6 | 0/0 | +0.0 [+0.0, +0.0] pp | 1.000 |
| 12 | 8 | 0/2 | -4.0 [-10.0, +0.0] pp | 0.500 |

P3 median planning time per replan, seconds:

| Generator | Verifier | Proposal | Verifier scoring | Encoding | Total planning |
|---:|---:|---:|---:|---:|---:|
| 6 | 6 | 0.110 | 0.078 | 0.014 | 0.203 |
| 6 | 8 | 0.110 | 0.106 | 0.027 | 0.242 |
| 6 | 12 | 0.171 | 0.271 | 0.045 | 0.487 |
| 8 | 6 | 0.146 | 0.079 | 0.027 | 0.252 |
| 8 | 8 | 0.147 | 0.098 | 0.013 | 0.259 |
| 8 | 12 | 0.146 | 0.138 | 0.026 | 0.310 |
| 12 | 6 | 0.213 | 0.080 | 0.027 | 0.320 |
| 12 | 8 | 0.211 | 0.096 | 0.027 | 0.333 |
| 12 | 12 | 0.339 | 0.259 | 0.023 | 0.620 |

Paired depth comparisons against 6 layers (exact two-sided McNemar; bootstrap CI is for paired success-rate difference):

| Mode | Depth | Gain/loss | Net Δ (95% bootstrap CI) | p |
|---|---:|---:|---:|---:|
| P0 | 8 | 1/1 | +0.0 pp [-6.0, +6.0] | 1.000 |
| P0 | 12 | 3/2 | +2.0 pp [-6.0, +10.0] | 1.000 |
| P1 | 8 | 2/2 | +0.0 pp [-8.0, +8.0] | 1.000 |
| P1 | 12 | 1/4 | -6.0 pp [-16.0, +2.0] | 0.375 |
| P3 diagonal | 8 | 1/1 | +0.0 pp [-6.0, +6.0] | 1.000 |
| P3 diagonal | 12 | 2/0 | +4.0 pp [+0.0, +10.0] | 0.500 |

### reacher — legacy (50 episodes)

| Depth | P0 success | P1 success | P3 diagonal success | P0 paired Δ vs 6L | P1 paired Δ vs 6L | P3 paired Δ vs 6L |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 74.0% | 72.0% | 78.0% | — | — | — |
| 8 | 74.0% | 88.0% | 74.0% | +0.0 pp | +16.0 pp | -4.0 pp |
| 12 | 80.0% | 86.0% | 80.0% | +6.0 pp | +14.0 pp | +2.0 pp |

P3 closed-loop success matrix (generator rows × verifier columns):

| Generator \ Verifier | 6 | 8 | 12 |
|---:|---:|---:|---:|
| 6 | 78.0% | 90.0% | 84.0% |
| 8 | 84.0% | 74.0% | 74.0% |
| 12 | 80.0% | 84.0% | 80.0% |

P3 verifier crossover, expressed as paired percentage-point change against the diagonal verifier at the same generator depth:

| Generator | Verifier change | Paired gain/loss | Net Δ (95% CI) | McNemar p |
|---:|---:|---:|---:|---:|
| 6 | 8 | 9/3 | +12.0 [-2.0, +26.0] pp | 0.146 |
| 6 | 12 | 8/5 | +6.0 [-8.0, +20.0] pp | 0.581 |
| 8 | 6 | 9/4 | +10.0 [-4.0, +24.0] pp | 0.267 |
| 8 | 12 | 7/7 | +0.0 [-14.0, +14.0] pp | 1.000 |
| 12 | 6 | 5/5 | +0.0 [-12.0, +12.0] pp | 1.000 |
| 12 | 8 | 6/4 | +4.0 [-8.0, +16.0] pp | 0.754 |

P3 generator crossover, expressed as paired percentage-point change against the diagonal generator at the same verifier depth:

| Verifier | Generator change | Paired gain/loss | Net Δ (95% CI) | McNemar p |
|---:|---:|---:|---:|---:|
| 6 | 8 | 8/5 | +6.0 [-8.0, +20.0] pp | 0.581 |
| 6 | 12 | 11/10 | +2.0 [-16.0, +20.0] pp | 1.000 |
| 8 | 6 | 10/2 | +16.0 [+4.0, +30.0] pp | 0.039 |
| 8 | 12 | 7/2 | +10.0 [-2.0, +22.0] pp | 0.180 |
| 12 | 6 | 7/5 | +4.0 [-10.0, +18.0] pp | 0.774 |
| 12 | 8 | 3/6 | -6.0 [-18.0, +6.0] pp | 0.508 |

P3 median planning time per replan, seconds:

| Generator | Verifier | Proposal | Verifier scoring | Encoding | Total planning |
|---:|---:|---:|---:|---:|---:|
| 6 | 6 | 0.136 | 0.115 | 0.018 | 0.270 |
| 6 | 8 | 0.135 | 0.135 | 0.034 | 0.305 |
| 6 | 12 | 0.136 | 0.196 | 0.036 | 0.369 |
| 8 | 6 | 0.172 | 0.106 | 0.034 | 0.312 |
| 8 | 8 | 0.188 | 0.152 | 0.018 | 0.358 |
| 8 | 12 | 0.180 | 0.195 | 0.035 | 0.411 |
| 12 | 6 | 0.256 | 0.110 | 0.034 | 0.401 |
| 12 | 8 | 0.257 | 0.137 | 0.034 | 0.428 |
| 12 | 12 | 0.269 | 0.206 | 0.018 | 0.494 |

Paired depth comparisons against 6 layers (exact two-sided McNemar; bootstrap CI is for paired success-rate difference):

| Mode | Depth | Gain/loss | Net Δ (95% bootstrap CI) | p |
|---|---:|---:|---:|---:|
| P0 | 8 | 6/6 | +0.0 pp [-14.0, +14.0] | 1.000 |
| P0 | 12 | 8/5 | +6.0 pp [-8.0, +20.0] | 0.581 |
| P1 | 8 | 10/2 | +16.0 pp [+4.0, +28.0] | 0.039 |
| P1 | 12 | 9/2 | +14.0 pp [+2.0, +26.0] | 0.065 |
| P3 diagonal | 8 | 5/7 | -4.0 pp [-18.0, +10.0] | 0.774 |
| P3 diagonal | 12 | 9/8 | +2.0 pp [-14.0, +18.0] | 1.000 |

### reacher — reacher_final (200 episodes)

| Depth | P0 success | P1 success | P3 diagonal success | P0 paired Δ vs 6L | P1 paired Δ vs 6L | P3 paired Δ vs 6L |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 75.5% | — | 84.5% | — | — | — |
| 8 | 72.5% | — | 80.5% | -3.0 pp | — | -4.0 pp |
| 12 | 75.5% | — | 81.5% | +0.0 pp | — | -3.0 pp |

P3 closed-loop success matrix (generator rows × verifier columns):

| Generator \ Verifier | 6 | 8 | 12 |
|---:|---:|---:|---:|
| 6 | 84.5% | 80.0% | 87.5% |
| 8 | 78.0% | 80.5% | 83.0% |
| 12 | 81.5% | 80.5% | 81.5% |

P3 verifier crossover, expressed as paired percentage-point change against the diagonal verifier at the same generator depth:

| Generator | Verifier change | Paired gain/loss | Net Δ (95% CI) | McNemar p |
|---:|---:|---:|---:|---:|
| 6 | 8 | 23/32 | -4.5 [-11.5, +2.5] pp | 0.281 |
| 6 | 12 | 19/13 | +3.0 [-2.5, +8.5] pp | 0.377 |
| 8 | 6 | 20/25 | -2.5 [-9.0, +4.0] pp | 0.551 |
| 8 | 12 | 24/19 | +2.5 [-4.0, +9.0] pp | 0.542 |
| 12 | 6 | 24/24 | +0.0 [-7.0, +7.0] pp | 1.000 |
| 12 | 8 | 26/28 | -1.0 [-8.0, +6.0] pp | 0.892 |

P3 generator crossover, expressed as paired percentage-point change against the diagonal generator at the same verifier depth:

| Verifier | Generator change | Paired gain/loss | Net Δ (95% CI) | McNemar p |
|---:|---:|---:|---:|---:|
| 6 | 8 | 18/31 | -6.5 [-13.5, +0.5] pp | 0.085 |
| 6 | 12 | 19/25 | -3.0 [-9.5, +3.5] pp | 0.451 |
| 8 | 6 | 26/27 | -0.5 [-7.5, +6.5] pp | 1.000 |
| 8 | 12 | 26/26 | +0.0 [-7.0, +7.0] pp | 1.000 |
| 12 | 6 | 31/19 | +6.0 [-1.0, +13.0] pp | 0.119 |
| 12 | 8 | 24/21 | +1.5 [-5.0, +8.0] pp | 0.766 |

P3 median planning time per replan, seconds:

| Generator | Verifier | Proposal | Verifier scoring | Encoding | Total planning |
|---:|---:|---:|---:|---:|---:|
| 6 | 6 | 0.532 | 0.493 | 0.065 | 1.091 |
| 6 | 8 | 0.581 | 0.626 | 0.140 | 1.347 |
| 6 | 12 | 0.547 | 0.849 | 0.131 | 1.527 |
| 8 | 6 | 0.720 | 0.486 | 0.128 | 1.334 |
| 8 | 8 | 0.721 | 0.599 | 0.064 | 1.384 |
| 8 | 12 | 0.720 | 0.854 | 0.130 | 1.704 |
| 12 | 6 | 1.083 | 0.486 | 0.131 | 1.700 |
| 12 | 8 | 1.070 | 0.600 | 0.128 | 1.798 |
| 12 | 12 | 1.067 | 0.848 | 0.065 | 1.980 |

Paired depth comparisons against 6 layers (exact two-sided McNemar; bootstrap CI is for paired success-rate difference):

| Mode | Depth | Gain/loss | Net Δ (95% bootstrap CI) | p |
|---|---:|---:|---:|---:|
| P0 | 8 | 19/25 | -3.0 pp [-9.5, +3.5] | 0.451 |
| P0 | 12 | 27/27 | +0.0 pp [-7.5, +7.0] | 1.000 |
| P3 diagonal | 8 | 21/29 | -4.0 pp [-11.0, +2.5] | 0.322 |
| P3 diagonal | 12 | 21/27 | -3.0 pp [-10.0, +3.5] | 0.471 |

## 如何判断影响来自生成还是 rerank

- P0 的深度变化和固定候选池 physical/oracle 指标一起读，反映动作生成器本身能否提出更好的动作。
- 固定同一个 generator 行，改变 verifier 列，反映 Stage-B 对相同物理候选排序的影响；generator 行变化时 verifier 不变，反映候选分布变化。
- P1 提供 verifier 深度在另一种候选生成机制（CEM 随机动作）下的参考；P3 闭环结果包含交互效应，不能把 P3 单独归因于某一个模块。
- Cube 沿用已有深度 sweep 的 dev/final 饱和结果，本次不重复跑候选池与闭环矩阵。

## 对 24 层实验的讨论

本组结果暂不支持直接做覆盖两项任务、同时扫 generator/verifier 的完整 24 层矩阵。Reacher 的 P1/CEM 在 8 层相对 6 层提升 16 pp（95% 配对 bootstrap CI [+4, +28] pp，McNemar p=0.039），12 层提升 14 pp（[+2, +26] pp，p=0.065）；但 Reacher final200 的 P0 在 6/12 层均为 75.5%，P3 对角线从 84.5% 降至 81.5%。固定候选池显示 12 层 Reacher generator 的平均距离有小幅改善，但 64 候选 oracle 对三个深度都达到 100%，P3 verifier 的优势随 generator 变化，未形成一致的深度趋势。PushT 闭环接近饱和，候选池表现以 8 层最好，12 层未继续改善。

如果仍要研究更大模型，优先做一项缩小的 Reacher verifier/Stage-B 深度对照：在相同候选集与规划预算上比较 8/12/24 层的 P1 和固定池排序；暂不同时扩展 24 层动作生成器。这样可以直接检验目前最明显的信号是否随 verifier 容量继续增长。只有当该收益在独立 seed 或 cohort 上复现，且 latency 成本可接受，再扩展到完整 P3 或更多任务。Reacher final200 中 P3 的 12/12 每次 replanning 中位时间为 1.980 s，6/6 为 1.091 s，提示深度增加已有明显推理成本。

这些比较来自单训练 seed，多个深度/模式的 p 值未做多重比较校正；50/200 episodes 也不能替代独立训练 seed。因此上述 24 层建议是下一步实验优先级判断，不是统计定论，也不自动决定 go/no-go。

机器可读汇总：`outputs/round4_shared_dit_depth_attribution/analysis.json`。原始条件、候选池和交叉评分均保存在同一 output root 下。
