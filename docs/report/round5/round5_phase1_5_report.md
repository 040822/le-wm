# Round5 Phase1.5：冻结模型的决策能力诊断与推理方案收敛

本报告由独立 Phase1.5 入口生成。结果只使用 R4-AB seed3072 epoch10 和 legacy_50，不修改 Phase1 冻结结果。闭环成功率是探索性证据，重复这 50 个起点不能替代最终泛化评测。

> **事后随机数审计（2026-09-29）：本报告中标为推理 seed 43/44 的 Phase1.5 条件，实际评估 seed 均为 42。** 扫描入口保存了声明 seed，但 `_compose()` 未将其写入 `cfg.seed`；任务评估配置默认值为 42。报告中依赖 seed 42/43/44 的稳定性、跨 seed 汇总及方法选择不能视为多 seed 证据，应撤回其确认性解释。原始结果保留用于追溯；按真实参数 seed=42 解释时，样本仍是反复使用的 `legacy_50`，且动作协议与 Phase1.6 不同。审计证据见 `outputs/round5/phase1_6_seed3072/audit/phase1_5_inference_seed_audit.json`。

- 配置：`/data/users/wenxin/pre-exp/le-wm/config/round5/phase1_5.json`
- 输出根目录：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase1_5_seed3072_legacy`
- 条件状态：`{"completed": 2600}`
- 历史复用审计：{"historical_path_candidate_pairs": 2544, "historical_path_candidate_pairs_are_hints_only": true, "reuse_reason_counts": {"current_artifact_already_available": 2600}, "source_counts": {"current": 2600}, "verified_historical_candidates": {"by_round": {"round4": 64, "round5": 36}, "by_task": {"cube": 25, "pusht": 25, "reacher": 25, "tworoom": 25}, "conditions": 100, "current_artifacts_present": {"True": 100}, "current_index_sources": {"current": 100}, "full_reuse": 100, "success_only": 0}}

## 主扫描验收

| group | planned | indexed | pending |
|---|---:|---:|---:|
| A1 | 144 | 144 | 0 |
| A2 | 432 | 432 | 0 |
| A3 | 864 | 864 | 0 |
| A4 | 432 | 432 | 0 |
| A5 | 96 | 96 | 0 |
| A6 | 240 | 240 | 0 |
| A7 | 216 | 216 | 0 |

主网格 2424 条条件；固定稳定性 96 条、自适应稳定性 80 条，合计 2600 条。每个完成条件覆盖原 50 个 cohort episode。
历史结果只有在 checkpoint、legacy cohort、normalizer、动作裁剪、精度、随机数和候选生成语义都一致时才复用；缺轨迹历史结果只进入 success-only 统计。
历史只读审计另外逐条件核验 checkpoint、cohort 与执行语义。其中 100 条可完整复用，0 条仅可复用成功率；已存在的 current 结果仍标记为 current，不计入历史复用。

## Seed 42 主扫描最高成功率条件（描述性）

| task | group | family | success | p50 planning (s) | source |
|---|---|---|---:|---:|---|
| cube | A3 | p0_guided_flow | 100.0% | 0.4652742031030357 | current |
| cube | A3 | p0_guided_flow | 100.0% | 0.17453392734751105 | current |
| cube | A6 | cem_budget | 100.0% | 0.801135943736881 | current |
| cube | A3 | p0_guided_flow | 100.0% | 0.08105652313679457 | current |
| tworoom | A3 | p0_guided_flow | 100.0% | 0.22433264693245292 | current |
| cube | A3 | p0_guided_flow | 100.0% | 0.34935761522501707 | current |
| cube | A1 | proposal_ranking | 100.0% | 0.21119897812604904 | current |
| cube | A2 | p0_post_opt | 100.0% | 0.10222758539021015 | current |
| tworoom | A7 | p2_guidance | 100.0% | 2.690777254058048 | current |
| tworoom | A6 | cem_budget | 100.0% | 2.1518935160711408 | current |
| pusht | A3 | p0_guided_flow | 100.0% | 0.06599557190202177 | current |
| cube | A5 | p3_guided_flow | 100.0% | 0.5643311040475965 | current |
| pusht | A3 | p0_guided_flow | 100.0% | 0.16962711745873094 | current |
| tworoom | A5 | p3_guided_flow | 100.0% | 0.6227761525660753 | current |
| cube | A2 | p0_post_opt | 100.0% | 0.06399708706885576 | current |
| cube | A3 | p0_guided_flow | 100.0% | 0.16736199613660574 | current |
| tworoom | A4 | p3_post_opt | 100.0% | 0.7774863061495125 | current |
| cube | A2 | p0_post_opt | 100.0% | 0.07804025989025831 | current |
| cube | A4 | p3_post_opt | 100.0% | 3.2565162028186023 | current |
| cube | A3 | p0_guided_flow | 100.0% | 0.18022423470392823 | current |

## 跨任务方法选择

候选来自固定稳定性网格与在四任务均可配对的自适应配置。主方案按三个推理 seed、四任务平均成功率选择；距最高值不超过 2 个百分点时优先选精测延迟更低者，并要求任何任务不低于该任务最佳配置 4 个百分点以上。另保留一个成功率—延迟 Pareto 备选。

| role | family / mode | aggregate success | batch1 p50 / p95 (s) | batch50 throughput (obs/s) |
|---|---|---:|---:|---:|
| 主推方案 | proposal_ranking/P3, S=2, N=64, K=None | 0.970 | 0.0749 / 0.1165 | 54.3 |
| Pareto 备选 | p3_post_opt/P3, S=2, N=64, K=5 | 0.975 | 0.1355 / 0.1724 | 23.6 |

### 精测延迟、吞吐与模型调用

调用数按一次 replanning 计；列顺序为 Stage A 前向、Stage B 前向、Stage B 引导反向。峰值为 batch1 的 PyTorch 已分配显存最大值。CEM 的 B 前向按每轮、每个 solver batch 的评分调用计；P2 warm start 计入 A/B/反向调用。

| role | method | success | batch1 p50 / p95 (s) | batch50 throughput (obs/s) | A / B / B-backward calls per decision | max batch1 allocated memory (GiB) |
|---|---|---:|---:|---:|---:|---:|
| 主推方案 | proposal_ranking/P3, S=2, N=64, K=None | 0.970 | 0.0749 / 0.1165 | 54.3 | 2.0 / 1.0 / 0.0 | 0.17 |
| Pareto 备选 | p3_post_opt/P3, S=2, N=64, K=5 | 0.975 | 0.1355 / 0.1724 | 23.6 | 2.0 / 6.0 / 5.0 | 0.21 |
| P1 CEM 成功率参照 | cem_budget/P1, S=None, N=300, K=None | 0.880 | 0.1375 / 0.1648 | 8.4 | 0.0 / 30.0 / 0.0 | 0.21 |
| P2 CEM 成功率参照 | cem_budget/P2, S=1, N=300, K=None | 0.960 | 0.1223 / 0.1429 | 9.3 | 1.0 / 30.0 / 0.0 | 0.21 |

主推方案 相对成功率最高的 P1 CEM 配置（cem_budget/P1, S=None, N=300, K=None）：成功率差 +9.0 个百分点，batch1 p50 延迟加速 1.84×。
主推方案 相对成功率最高的 P2 CEM 配置（cem_budget/P2, S=1, N=300, K=None）：成功率差 +1.0 个百分点，batch1 p50 延迟加速 1.63×。
Pareto 备选 相对成功率最高的 P1 CEM 配置（cem_budget/P1, S=None, N=300, K=None）：成功率差 +9.5 个百分点，batch1 p50 延迟加速 1.01×。
Pareto 备选 相对成功率最高的 P2 CEM 配置（cem_budget/P2, S=1, N=300, K=None）：成功率差 +1.5 个百分点，batch1 p50 延迟加速 0.90×。

### 跨任务成功率—延迟 Pareto 候选

| method | aggregate success | batch1 p50 (s) |
|---|---:|---:|
| p3_post_opt/P3, S=2, N=64, K=5 | 0.975 | 0.1355 |
| proposal_ranking/P3, S=2, N=64, K=None | 0.970 | 0.0749 |
| p3_refine/P3, S=2, N=64, K=5 | 0.955 | 0.0736 |
| proposal_ranking/P0, S=2, N=1, K=None | 0.930 | 0.0514 |

## 诊断与决策

### 物理 probe

按轨迹拆分 train/validation，并排除评测轨迹。下表误差使用真实未来图像编码；B 预测 future latent 的读出误差另存于 probe JSON。

| task | validation rows | Ridge MAE | random encoder Ridge MAE | MLP MAE mean (3 seeds) | real future-latent MAE | B predicted-latent MAE |
|---|---:|---:|---:|---:|---:|---:|
| cube | 20000 | 0.0083 | 0.0421 | 0.0029 | 0.0069 | 0.0533 |
| pusht | 19054 | 5.2897 | 26.3391 | 2.3495 | 6.9819 | 41.2238 |
| reacher | 20000 | 0.0082 | 0.3091 | 0.0062 | 0.0087 | 0.2459 |
| tworoom | 18651 | 0.7220 | 5.5321 | 0.2363 | 0.6529 | 20.7009 |

### 固定候选池

成功率和 selection regret 先按状态计算，置信区间按状态聚类 bootstrap。

| task | S | oracle success | B-selected success [95% CI] | random success [95% CI] | B−random success [paired 95% CI] | selection regret [95% CI] | cost-distance correlation [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| cube | 1 | 1.000 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.139 [0.099, 0.182] | -0.000 |
| cube | 2 | 1.000 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.155 [0.116, 0.196] | -0.005 |
| cube | 5 | 1.000 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.164 [0.123, 0.205] | -0.016 |
| cube | 10 | 1.000 | 1.000 [1.000, 1.000] | 0.999 [0.997, 1.000] | 0.001 [0.000, 0.003] | 0.194 [0.148, 0.244] | 0.008 |
| cube | 16 | 1.000 | 1.000 [1.000, 1.000] | 0.998 [0.994, 1.000] | 0.002 [0.000, 0.005] | 0.186 [0.139, 0.237] | 0.017 |
| cube | 32 | 1.000 | 1.000 [1.000, 1.000] | 0.999 [0.996, 1.000] | 0.001 [0.000, 0.004] | 0.196 [0.145, 0.248] | 0.023 |
| pusht | 1 | 0.900 | 0.220 [0.120, 0.340] | 0.167 [0.069, 0.267] | 0.053 [-0.006, 0.131] | 3.173 [2.434, 3.929] | 0.088 [0.029, 0.148] |
| pusht | 2 | 0.980 | 0.160 [0.060, 0.260] | 0.143 [0.074, 0.225] | 0.017 [-0.054, 0.094] | 3.205 [2.545, 3.898] | 0.022 [-0.009, 0.058] |
| pusht | 5 | 0.180 | 0.160 [0.060, 0.260] | 0.120 [0.046, 0.209] | 0.040 [-0.001, 0.096] | 0.271 [0.153, 0.436] | 0.203 [0.112, 0.295] |
| pusht | 10 | 0.180 | 0.140 [0.060, 0.240] | 0.113 [0.041, 0.196] | 0.027 [-0.029, 0.088] | 0.348 [0.193, 0.550] | 0.232 [0.144, 0.321] |
| pusht | 16 | 0.200 | 0.120 [0.040, 0.220] | 0.111 [0.040, 0.193] | 0.009 [-0.058, 0.078] | 0.400 [0.217, 0.631] | 0.236 [0.151, 0.320] |
| pusht | 32 | 0.200 | 0.120 [0.040, 0.220] | 0.107 [0.039, 0.190] | 0.013 [-0.052, 0.083] | 0.425 [0.233, 0.665] | 0.240 [0.158, 0.324] |
| reacher | 1 | 0.680 | 0.360 [0.220, 0.500] | 0.376 [0.261, 0.495] | -0.016 [-0.108, 0.074] | 0.557 [0.361, 0.784] | 0.327 [0.181, 0.470] |
| reacher | 2 | 1.000 | 0.520 [0.380, 0.660] | 0.492 [0.413, 0.572] | 0.028 [-0.103, 0.157] | 1.384 [1.085, 1.729] | 0.269 [0.185, 0.351] |
| reacher | 5 | 1.000 | 0.600 [0.460, 0.740] | 0.554 [0.490, 0.615] | 0.046 [-0.075, 0.168] | 1.435 [1.129, 1.779] | 0.295 [0.191, 0.397] |
| reacher | 10 | 1.000 | 0.600 [0.460, 0.740] | 0.554 [0.498, 0.608] | 0.046 [-0.073, 0.161] | 1.403 [1.107, 1.722] | 0.365 [0.275, 0.452] |
| reacher | 16 | 1.000 | 0.640 [0.500, 0.780] | 0.547 [0.491, 0.600] | 0.093 [-0.035, 0.217] | 1.616 [1.266, 2.017] | 0.393 [0.306, 0.475] |
| reacher | 32 | 1.000 | 0.580 [0.440, 0.720] | 0.527 [0.480, 0.575] | 0.052 [-0.078, 0.177] | 1.545 [1.197, 1.947] | 0.412 [0.329, 0.488] |
| tworoom | 1 | 1.000 | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | 0.000 [0.000, 0.001] | 0.072 [0.061, 0.083] | 0.073 [0.039, 0.108] |
| tworoom | 2 | 1.000 | 1.000 [1.000, 1.000] | 0.986 [0.971, 0.996] | 0.014 [0.004, 0.029] | 0.232 [0.208, 0.257] | 0.110 [0.051, 0.173] |
| tworoom | 5 | 1.000 | 1.000 [1.000, 1.000] | 0.968 [0.945, 0.987] | 0.032 [0.013, 0.055] | 0.265 [0.235, 0.293] | 0.124 [0.061, 0.193] |
| tworoom | 10 | 1.000 | 1.000 [1.000, 1.000] | 0.957 [0.928, 0.981] | 0.043 [0.019, 0.072] | 0.252 [0.219, 0.287] | 0.123 [0.061, 0.190] |
| tworoom | 16 | 1.000 | 1.000 [1.000, 1.000] | 0.953 [0.920, 0.981] | 0.047 [0.020, 0.080] | 0.288 [0.257, 0.319] | 0.129 [0.067, 0.198] |
| tworoom | 32 | 1.000 | 1.000 [1.000, 1.000] | 0.953 [0.919, 0.980] | 0.047 [0.020, 0.080] | 0.283 [0.248, 0.316] | 0.136 [0.073, 0.203] |
各 S 单独汇报，以免跨 S 汇总掩盖提案分布差异。oracle 反映 A 候选池的覆盖，B-selected 与随机选择的配对差更直接反映 B 的排序增益；两者需结合解读。

### 固定控制动作

物理零动作和归一化零动作作为不同控制分别统计；方向扰动按固定随机种子生成。

| task | control kind | states | success rate | mean normalized physical distance |
|---|---|---:|---:|---:|
| cube | anchor | 50 | 1.000 | 0.621 |
| cube | block_transform | 50 | 0.656 | 1.426 |
| cube | normalized_zero | 50 | 0.440 | 2.937 |
| cube | physical_zero | 50 | 0.440 | 2.939 |
| cube | rms_perturbation | 50 | 0.993 | 0.621 |
| cube | standard_gaussian | 50 | 0.499 | 2.378 |
| pusht | anchor | 50 | 0.120 | 3.931 |
| pusht | block_transform | 50 | 0.033 | 4.097 |
| pusht | normalized_zero | 50 | 0.000 | 7.618 |
| pusht | physical_zero | 50 | 0.000 | 7.585 |
| pusht | rms_perturbation | 50 | 0.063 | 4.019 |
| pusht | standard_gaussian | 50 | 0.000 | 7.996 |
| reacher | anchor | 50 | 0.613 | 1.635 |
| reacher | block_transform | 50 | 0.444 | 2.615 |
| reacher | normalized_zero | 50 | 0.020 | 12.987 |
| reacher | physical_zero | 50 | 0.020 | 12.988 |
| reacher | rms_perturbation | 50 | 0.511 | 2.164 |
| reacher | standard_gaussian | 50 | 0.075 | 16.056 |
| tworoom | anchor | 50 | 0.993 | 0.879 |
| tworoom | block_transform | 50 | 0.918 | 0.952 |
| tworoom | normalized_zero | 50 | 0.100 | 2.474 |
| tworoom | physical_zero | 50 | 0.080 | 2.490 |
| tworoom | rms_perturbation | 50 | 0.987 | 0.885 |
| tworoom | standard_gaussian | 50 | 0.227 | 2.607 |

### 同状态配对控制效应

控制效应为控制动作减去同一状态三个 anchor 的平均结果；该合成基线不是实际执行策略。成功率正值表示控制成功更多，距离正值表示控制距离更差。成功率按 `legacy_50` 全部状态配对；距离使用双方共同有效的 raw milestone（5/10/15/20/25）。95% 区间以状态为单位进行 10,000 次 bootstrap。每个动作/状态只有一次记录，因此区间不包含 rollout 随机性；固定方向与控制组比较属于探索性分析，未做多重比较校正。

| task | control action | success Δ [state-bootstrap 95% CI] | common-step distance Δ [state-bootstrap 95% CI] | coverage (success states; common-step action-state pairs; distance states) |
|---|---|---:|---:|---:|
| PushT | normalized Gaussian | -0.120 [-0.213, -0.040] | +3.877 [+3.141, +4.592] | 50/50 success states; 3,200/3,200 pairs; 50/50 distance states |
| PushT | physical zero | -0.120 [-0.213, -0.040] | +3.481 [+2.747, +4.220] | 50/50 success states; 50/50 pairs; 50/50 distance states |
| PushT | normalized zero | -0.120 [-0.213, -0.040] | +3.509 [+2.758, +4.255] | 50/50 success states; 50/50 pairs; 50/50 distance states |
| Reacher | normalized Gaussian | -0.538 [-0.616, -0.461] | +12.342 [+10.517, +14.246] | 50/50 success states; 3,063/3,200 pairs; 48/50 distance states |
| Reacher | physical zero | -0.593 [-0.680, -0.513] | +10.208 [+8.302, +12.326] | 50/50 success states; 48/50 pairs; 48/50 distance states |
| Reacher | normalized zero | -0.593 [-0.680, -0.513] | +10.206 [+8.202, +12.326] | 50/50 success states; 48/50 pairs; 48/50 distance states |
| TwoRoom | normalized Gaussian | -0.766 [-0.844, -0.681] | +1.476 [+1.199, +1.765] | 50/50 success states; 2,552/3,200 pairs; 40/50 distance states |
| TwoRoom | physical zero | -0.913 [-0.980, -0.833] | +1.395 [+1.127, +1.682] | 50/50 success states; 40/50 pairs; 40/50 distance states |
| TwoRoom | normalized zero | -0.893 [-0.973, -0.800] | +1.390 [+1.104, +1.690] | 50/50 success states; 40/50 pairs; 40/50 distance states |
| Cube | normalized Gaussian | -0.501 [-0.624, -0.378] | +2.262 [+1.626, +2.966] | 50/50 success states; 1,761/3,200 pairs; 28/50 distance states |
| Cube | physical zero | -0.560 [-0.700, -0.420] | +3.121 [+2.537, +3.730] | 50/50 success states; 28/50 pairs; 28/50 distance states |
| Cube | normalized zero | -0.560 [-0.700, -0.420] | +3.116 [+2.538, +3.728] | 50/50 success states; 28/50 pairs; 28/50 distance states |

### Cube 控制池迁移审计

Cube 控制池由 240 个源动作、12,000 条记录迁移为 222 个动作、11,100 条记录。逐动作核对：All 11,100 retained slot/action rows matched regenerated current actions elementwise in float64; maximum absolute difference 0; no tolerance-only matches; no missing rows.。
是否重跑环境结果：否；保留旧版逐动作文件 240 份。
SHA-256：源聚合 `a6a663fb00ff3d43c3ba66cb525dbe63c50cd6dcc5abc77cbcdc8a0af97a25ec`；当前聚合 `ca5ce5ca9b4a259ced4673740e66743615a42ddbf96c8f98c99b971d9b7b0345`。

### 配对梯度修正

Guidance 成功率沿用评估器记录的 `episode_successes`，表示 `eval_budget=50` 内出现成功事件；它不等同于统一第 25 个 primitive step 的终态成功率。物理距离改变量单独按每个起点最后一个共同有效 milestone（5/10/15/20/25）配对。两类指标分别解读，并同时报告共同 milestone 的步数分布。

每项修正与相同 RMS 位移的随机方向对照，按共同有效 milestone 配对；没有共同 milestone 的分支不进入配对均值。

| task | guidance | variants | physical paired advantage, mean [mean config CI] | predicted improvement | true latent improvement | true physical improvement | model exploitation |
|---|---|---:|---:|---:|---:|---:|---:|
| pusht | guided_flow | 36 | 2.4346 [1.8295, 3.0747] | 0.0168 | 7.9426 | 2.6084 | 0.073 |
| pusht | post_opt | 216 | 2.4407 [1.8496, 3.0657] | 0.0165 | 6.0198 | 1.9510 | 0.157 |
| reacher | guided_flow | 36 | 0.2094 [-0.1100, 0.5355] | 0.0104 | 0.1078 | 0.1189 | 0.355 |
| reacher | post_opt | 216 | 0.2730 [-0.0017, 0.5496] | 0.0195 | 0.3071 | 0.2656 | 0.302 |
| tworoom | guided_flow | 36 | 0.0435 [-0.0004, 0.0890] | 0.0655 | 0.1789 | 0.0257 | 0.304 |

### 共同物理 milestone 分布

下表计数单位是 Guidance 变体×起点记录；同一起点在多个配置中会重复出现，不能当作独立样本数。配对距离取每条记录最后一个共同有效 milestone。

| task | guidance | paired rows | unpaired rows | step 5 | step 10 | step 15 | step 20 | step 25 | paired rows missing step |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cube | post_opt | — | — | — | — | — | — | — | — |
| cube | guided_flow | — | — | — | — | — | — | — | — |
| pusht | post_opt | 10800 | 0 | 0 | 193 | 558 | 8030 | 2019 | 0 |
| pusht | guided_flow | 1800 | 0 | 0 | 28 | 76 | 1370 | 326 | 0 |
| reacher | post_opt | 10665 | 135 | 141 | 481 | 1087 | 3496 | 5460 | 0 |
| reacher | guided_flow | 1782 | 18 | 29 | 94 | 176 | 621 | 862 | 0 |
| tworoom | post_opt | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tworoom | guided_flow | 1520 | 280 | 117 | 460 | 591 | 346 | 6 | 0 |

下面的 pooled-milestone GF−PO 表按相同配置和起点匹配，但每条记录使用其最后一个共同有效步数；由于步数分布可能不同，该汇总和逐配置表仅作探索性描述。正数表示 GF 改善更多，区间按起点聚类。

| task | matched configurations | matched states | pooled-milestone GF−PO physical improvement (exploratory) [state-clustered 95% CI] |
|---|---:|---:|---:|
| cube | 0 | 0 | — |
| pusht | 36 | 50 | -0.0115 [-0.0381, 0.0115] |
| reacher | 36 | 50 | -0.0137 [-0.0611, 0.0346] |
| tworoom | 0 | 0 | — |

| task | S | candidate | K | η | R | states | pooled-milestone GF−PO physical improvement (exploratory) [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| pusht | 2 | 0 | 1 | 0.003 | 0.2 | 50 | 0.0034 [-0.0008, 0.0084] |
| pusht | 2 | 0 | 1 | 0.01 | 0.2 | 50 | 0.0021 [-0.0041, 0.0091] |
| pusht | 2 | 0 | 1 | 0.03 | 0.2 | 50 | 0.0026 [-0.0155, 0.0219] |
| pusht | 2 | 0 | 10 | 0.003 | 0.2 | 49 | -0.0001 [-0.0107, 0.0122] |
| pusht | 2 | 0 | 10 | 0.01 | 0.2 | 48 | 0.0069 [-0.0207, 0.0344] |
| pusht | 2 | 0 | 10 | 0.03 | 0.2 | 46 | -0.0268 [-0.1033, 0.0419] |
| pusht | 2 | 0 | 5 | 0.003 | 0.2 | 50 | 0.0019 [-0.0048, 0.0094] |
| pusht | 2 | 0 | 5 | 0.01 | 0.2 | 47 | -0.0149 [-0.0375, 0.0083] |
| pusht | 2 | 0 | 5 | 0.03 | 0.2 | 44 | -0.0293 [-0.0820, 0.0237] |
| pusht | 5 | 0 | 1 | 0.003 | 0.2 | 50 | 0.0008 [-0.0033, 0.0051] |
| pusht | 5 | 0 | 1 | 0.01 | 0.2 | 49 | 0.0014 [-0.0097, 0.0140] |
| pusht | 5 | 0 | 1 | 0.03 | 0.2 | 45 | 0.0915 [-0.0157, 0.2632] |
| pusht | 5 | 0 | 10 | 0.003 | 0.2 | 48 | 0.0386 [-0.0142, 0.1138] |
| pusht | 5 | 0 | 10 | 0.01 | 0.2 | 49 | -0.0990 [-0.2640, 0.0037] |
| pusht | 5 | 0 | 10 | 0.03 | 0.2 | 46 | -0.0374 [-0.1694, 0.0865] |
| pusht | 5 | 0 | 5 | 0.003 | 0.2 | 50 | 0.0019 [-0.0115, 0.0160] |
| pusht | 5 | 0 | 5 | 0.01 | 0.2 | 46 | 0.0017 [-0.0829, 0.0651] |
| pusht | 5 | 0 | 5 | 0.03 | 0.2 | 44 | -0.1326 [-0.3278, 0.0181] |
| pusht | 2 | 1 | 1 | 0.003 | 0.2 | 49 | -0.0005 [-0.0024, 0.0019] |
| pusht | 2 | 1 | 1 | 0.01 | 0.2 | 50 | -0.0018 [-0.0053, 0.0019] |
| pusht | 2 | 1 | 1 | 0.03 | 0.2 | 48 | -0.0058 [-0.0244, 0.0150] |
| pusht | 2 | 1 | 10 | 0.003 | 0.2 | 50 | -0.0009 [-0.0083, 0.0068] |
| pusht | 2 | 1 | 10 | 0.01 | 0.2 | 48 | -0.0164 [-0.0395, 0.0054] |
| pusht | 2 | 1 | 10 | 0.03 | 0.2 | 46 | -0.0055 [-0.0518, 0.0428] |
| pusht | 2 | 1 | 5 | 0.003 | 0.2 | 49 | -0.0022 [-0.0068, 0.0026] |
| pusht | 2 | 1 | 5 | 0.01 | 0.2 | 46 | 0.0090 [-0.0107, 0.0295] |
| pusht | 2 | 1 | 5 | 0.03 | 0.2 | 44 | 0.0242 [-0.0426, 0.0919] |
| pusht | 5 | 1 | 1 | 0.003 | 0.2 | 50 | -0.0002 [-0.0040, 0.0034] |
| pusht | 5 | 1 | 1 | 0.01 | 0.2 | 50 | 0.0001 [-0.0098, 0.0102] |
| pusht | 5 | 1 | 1 | 0.03 | 0.2 | 47 | 0.0069 [-0.0304, 0.0462] |
| pusht | 5 | 1 | 10 | 0.003 | 0.2 | 47 | 0.0028 [-0.0211, 0.0296] |
| pusht | 5 | 1 | 10 | 0.01 | 0.2 | 46 | -0.0104 [-0.0584, 0.0409] |
| pusht | 5 | 1 | 10 | 0.03 | 0.2 | 45 | -0.1683 [-0.4678, 0.0216] |
| pusht | 5 | 1 | 5 | 0.003 | 0.2 | 49 | 0.0004 [-0.0133, 0.0142] |
| pusht | 5 | 1 | 5 | 0.01 | 0.2 | 47 | -0.0084 [-0.1112, 0.0651] |
| pusht | 5 | 1 | 5 | 0.03 | 0.2 | 42 | -0.0213 [-0.0990, 0.0585] |
| reacher | 2 | 0 | 1 | 0.003 | 0.2 | 49 | 0.0065 [-0.0024, 0.0158] |
| reacher | 2 | 0 | 1 | 0.01 | 0.2 | 49 | 0.0181 [-0.0129, 0.0523] |
| reacher | 2 | 0 | 1 | 0.03 | 0.2 | 46 | -0.1117 [-0.2358, 0.0082] |
| reacher | 2 | 0 | 10 | 0.003 | 0.2 | 47 | -0.0053 [-0.0807, 0.0685] |
| reacher | 2 | 0 | 10 | 0.01 | 0.2 | 43 | -0.2072 [-0.3733, -0.0559] |
| reacher | 2 | 0 | 10 | 0.03 | 0.2 | 37 | -0.1548 [-0.3549, 0.0414] |
| reacher | 2 | 0 | 5 | 0.003 | 0.2 | 44 | -0.0212 [-0.0623, 0.0203] |
| reacher | 2 | 0 | 5 | 0.01 | 0.2 | 41 | -0.0685 [-0.1691, 0.0388] |
| reacher | 2 | 0 | 5 | 0.03 | 0.2 | 36 | -0.1341 [-0.5281, 0.2645] |
| reacher | 5 | 0 | 1 | 0.003 | 0.2 | 49 | 0.0081 [-0.0179, 0.0341] |
| reacher | 5 | 0 | 1 | 0.01 | 0.2 | 47 | 0.0101 [-0.0760, 0.0987] |
| reacher | 5 | 0 | 1 | 0.03 | 0.2 | 42 | 0.1102 [-0.2127, 0.4378] |
| reacher | 5 | 0 | 10 | 0.003 | 0.2 | 44 | 0.0061 [-0.1197, 0.1632] |
| reacher | 5 | 0 | 10 | 0.01 | 0.2 | 34 | -0.1446 [-0.3173, 0.0279] |
| reacher | 5 | 0 | 10 | 0.03 | 0.2 | 40 | -0.1931 [-0.4395, 0.0454] |
| reacher | 5 | 0 | 5 | 0.003 | 0.2 | 46 | 0.0528 [-0.0471, 0.1624] |
| reacher | 5 | 0 | 5 | 0.01 | 0.2 | 43 | -0.0108 [-0.1876, 0.1776] |
| reacher | 5 | 0 | 5 | 0.03 | 0.2 | 34 | -0.1618 [-0.5981, 0.2390] |
| reacher | 2 | 1 | 1 | 0.003 | 0.2 | 49 | 0.0220 [0.0024, 0.0525] |
| reacher | 2 | 1 | 1 | 0.01 | 0.2 | 49 | 0.0178 [-0.0237, 0.0568] |
| reacher | 2 | 1 | 1 | 0.03 | 0.2 | 47 | -0.0565 [-0.1792, 0.0578] |
| reacher | 2 | 1 | 10 | 0.003 | 0.2 | 48 | -0.0006 [-0.0753, 0.0782] |
| reacher | 2 | 1 | 10 | 0.01 | 0.2 | 43 | 0.1016 [-0.0256, 0.2347] |
| reacher | 2 | 1 | 10 | 0.03 | 0.2 | 43 | -0.0931 [-0.4234, 0.1818] |
| reacher | 2 | 1 | 5 | 0.003 | 0.2 | 48 | 0.0347 [-0.0070, 0.0787] |
| reacher | 2 | 1 | 5 | 0.01 | 0.2 | 45 | -0.0411 [-0.1635, 0.0865] |
| reacher | 2 | 1 | 5 | 0.03 | 0.2 | 37 | -0.1491 [-0.4504, 0.1560] |
| reacher | 5 | 1 | 1 | 0.003 | 0.2 | 50 | 0.0290 [-0.0015, 0.0594] |
| reacher | 5 | 1 | 1 | 0.01 | 0.2 | 48 | 0.0668 [-0.0353, 0.1702] |
| reacher | 5 | 1 | 1 | 0.03 | 0.2 | 41 | 0.1767 [-0.1703, 0.5203] |
| reacher | 5 | 1 | 10 | 0.003 | 0.2 | 47 | 0.1305 [0.0140, 0.2702] |
| reacher | 5 | 1 | 10 | 0.01 | 0.2 | 44 | -0.0334 [-0.1814, 0.1209] |
| reacher | 5 | 1 | 10 | 0.03 | 0.2 | 38 | 0.0792 [-0.1226, 0.2872] |
| reacher | 5 | 1 | 5 | 0.003 | 0.2 | 48 | 0.0686 [-0.0567, 0.1976] |
| reacher | 5 | 1 | 5 | 0.01 | 0.2 | 47 | 0.0571 [-0.0591, 0.1807] |
| reacher | 5 | 1 | 5 | 0.03 | 0.2 | 40 | 0.0047 [-0.3491, 0.3492] |

主要物理比较按完全相同的 `physical_comparison_step` 分层，再匹配 flow steps、候选索引、K、η、R 和起点真实物理代价。该步数是 baseline、guided、random 三条分支最后共同有效的 milestone，受终止与结果可用性影响；分层由结果后的可观测长度决定，各步可能对应不同起点/配置子集。因此这些差值是描述性诊断，不能解释为同一批起点随时间的变化，也不是全50起点的因果效应。配置在起点内先平均，bootstrap 以起点为单位；跨步数/配置检视未做多重比较校正。正数表示 GF 改善更多。

| task | common physical step | matched configurations | matched states | GF−PO true physical improvement [state-clustered 95% CI] |
|---|---:|---:|---:|---:|
| pusht | 10 | 28 | 1 | 0.0012 |
| pusht | 15 | 33 | 7 | -0.0377 [-0.0890, 0.0075] |
| pusht | 20 | 36 | 44 | -0.0142 [-0.0350, 0.0045] |
| pusht | 25 | 36 | 19 | -0.0453 [-0.2247, 0.0791] |
| reacher | 5 | 27 | 2 | 0.0379 [-0.0997, 0.1755] |
| reacher | 10 | 31 | 7 | -0.1925 [-0.2955, -0.0978] |
| reacher | 15 | 35 | 19 | -0.2519 [-0.5204, -0.0267] |
| reacher | 20 | 36 | 34 | -0.0700 [-0.1567, 0.0125] |
| reacher | 25 | 36 | 45 | 0.0573 [0.0132, 0.1066] |
| tworoom | 5 | 0 | 0 | — |
| tworoom | 10 | 0 | 0 | — |
| tworoom | 15 | 0 | 0 | — |
| tworoom | 20 | 0 | 0 | — |
| tworoom | 25 | 0 | 0 | — |

计时只纳入四任务精测均无干扰的匹配配置，延迟差为负表示 GF 更快。

| matched config | GF−PO mean batch1 p50 (s) | tasks with GF faster | all four faster |
|---|---:|---:|---|
| P0, S=2, N=1, K=5 | +0.0931 | 0/4 | 否 |

### 公平计时

完整策略推理在固定真实观测上同步计时，包含编码、提案、评分/梯度、CEM 更新和动作输出；环境运行时间单列。明细：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase1_5_seed3072_legacy/analysis/timing.json`。
单环境报告 batch1 p50/p95；并行报告 batch50 延迟和吞吐。细测测量窗口前后记录 GPU 外部计算进程，存在干扰的窗口应排除主要加速结论。

粗测敏感性按 seed42 主网格逐任务比较观测到的 batch1 前沿，以及仅保留 batch1 边界采样未观察到外部计算的前沿。边界采样不能证明整个测量窗口隔离；最终选型只使用精测两个窗口均未观察到外部计算的配置。

粗测窗口边界干扰状态计数：

| window | status | count |
|---|---|---:|
| batch1 | interference_flagged | 1664 |
| batch1 | no_external_compute_observed_at_boundaries | 760 |
| batch50 | interference_flagged | 1749 |
| batch50 | no_external_compute_observed_at_boundaries | 675 |

两种 batch1 前沿及隔离复测候选：

| task | observed frontier | batch1 boundary-unflagged frontier | union frontier | coarse interference-suspect retests |
|---|---:|---:|---:|---:|
| cube | 1 | 1 | 2 | 1 |
| pusht | 3 | 3 | 6 | 4 |
| reacher | 8 | 6 | 10 | 6 |
| tworoom | 1 | 3 | 4 | 2 |

全任务共保留 22 个前沿配置，并另保留 48 个固定锚点和 40 个自适应锚点；前沿中的受干扰/缺少快照候选有 13 个；其中已完成隔离精测 13/13 个，batch1 与 batch50 边界均未观察到外部计算 13/13 个。

1. P3：六个 S 等权的 B-selected−随机成功率差在跨任务平均上为 +0.025，1/4 个任务的状态聚类 95% CI 下界大于 0；跨任务平均 oracle−B-selected 归一化距离 regret 为 0.758；尚不支持跨任务稳定保留 P3；不过统一三 seed 成功率/精测延迟规则仍将 proposal_ranking/P3, S=2, N=64 选为工程主配置；该配置选择不等同于候选池诊断证明 P3 排序具有稳定的跨任务优势。
2. PO/GF 替代 CEM：当前跨任务主方案不是经过配对实测的单候选 PO/GF，尚无替代 CEM 的直接证据。
3. GF：GF−PO 状态配对真实物理改善差仅有 2/4 个任务的完整区间；0/1 个完整匹配配置在四任务均测得 GF 更低的单环境 p50 延迟，证据不足以决定是否单独保留 GF。
4. R4-AB 下一轮：下一轮 R4-AB 消融待诊断完成后确定。

## 产物

- 索引：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase1_5_seed3072_legacy/analysis/index.json`
- 历史只读审计：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase1_5_seed3072_legacy/analysis/history_reuse_audit.json`
- 条件 CSV：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase1_5_seed3072_legacy/analysis/conditions.csv`
- 计时：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase1_5_seed3072_legacy/analysis/timing.json`
- 细测选择：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase1_5_seed3072_legacy/analysis/fine_timing_selection.json`
- 跨任务方法选择：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase1_5_seed3072_legacy/analysis/method_selection.json`
- 本报告：`/data/users/wenxin/pre-exp/le-wm/docs/report/round5/round5_phase1_5_report.md`
