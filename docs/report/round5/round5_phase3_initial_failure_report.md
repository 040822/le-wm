# Round 5 Phase 3：剔除初始成功样本后的复评

本报告复用 Round 5 Phase 3 已有的训练 checkpoint（训练 seed 3072），在新的、排除初始成功起点的固定 cohort 上复评；评测 seed 为 42。结果是描述性单训练 seed 实验，不构成 DeWM 论文六种子结果的严格复现。

## 起点评定、成功标准与抽样

先以规划使用的同一未来 goal row（start_step + 25）对每个候选起点应用任务成功判据；起点已成功的 start-goal pair 不进入新 cohort。然后排除原 legacy cohort 中的 episode，在其余 episode 中每个任务固定抽取 50 个不同 episode，每个 episode 随机取一个符合条件的起点。所有方法和 FastLeWAM 条件复用同一任务 cohort；筛选不依据任何模型输出。运行成功率的分母为这 50 个起点未成功的 episode。

| 任务 | 起点排除判据 / episode 成功判据 | 原 legacy cohort 初始成功 | 新 cohort 初始成功 |
|---|---|---:|---:|
| scene | cube 位置 L2 ≤ 0.04 m；两个 button 状态均精确匹配；drawer 与 window 位置误差均 ≤ 0.04 m，五项同时满足 | 18/50 | 0/50 |
| finger | tip 到未来 goal target 的距离 ≤ 0.03 m（与 turn_hard runtime reward ≥ 1 等价） | 42/50 | 0/50 |
| humanoid | head height ≥ 1.4、torso upright ≥ 0.9、平面质心速度 ≥ 1.0，三项同时满足 | 8/50 | 0/50 |

执行阶段仍按原 Phase 3 runtime termination 判定：在目标设定后的最多 50 个控制步内，任意一步达到同一任务成功判据即记为成功。Scene 目标状态和规划图像都取未来 goal row；Finger 使用 DMControl `turn_hard` 奖励；Humanoid 使用上述三阈值 conjunction。DMC 数据集的 NaN `success` 列不参与计算。

新 cohort 是独立的补充协议；它回答起点本来未成功时的条件成功率，不能与 legacy 的 50-episode 总体成功率混成同一分数。legacy 结果保留在原报告中，仅用于协议对照。

| task | cohort id | sampling seed | eligible failed start-goal pairs | selected unique episodes | cohort SHA-256 |
|---|---|---:|---:|---:|---|
| scene | `scene_initial_failure_50_phase3_v1` | 43 | 1096383 | 50 | `d58624fd1c422007d11946309046ad07697b001496bd0a12b20ee482b864f5f4` |
| finger | `finger_initial_failure_50_phase3_v1` | 43 | 269928 | 50 | `65a5409ac3e7559fb8c827e4b68299d7d58e941d37e1a40a268a413703468d8c` |
| humanoid | `humanoid_initial_failure_50_phase3_v1` | 43 | 1486334 | 50 | `c3f0c4aff2eced8103778ee5afb67bdb5c8de4b477913f8b2b80ca7177cc3bf4` |

新 cohort 采用 `sampling_revised` manifest；候选 start-goal row 沿用 legacy 有效行规则和全局末候选排除，筛选后按 episode 均匀抽样，每个 episode 内随机选择一个初始失败起点。旧 legacy episode 整体排除，以保证两批 episode 不重叠。

## 原 legacy 主结果分解

下表对原报告的 50 个 legacy episode 使用同一初始成功判据做拆分；总体率是旧评测的原始值，右侧条件率只以起点未成功的 episode 为分母。这是对旧 cohort 的诊断性拆分，不替代下方独立新 cohort 的复评。Finger 的旧 cohort 仅有 8 个起点未成功，条件率不适合用于方法排名。

| task | method | legacy 总成功 | 起点已成功 | 起点未成功后成功 | 起点未成功条件率 |
|---|---|---:|---:|---:|---:|
| scene | fast_lewam | 22/50 (44.0%) | 18/50 | 4/32 | 12.5% |
| scene | lewm | 23/50 (46.0%) | 18/50 | 5/32 | 15.6% |
| scene | leflow | 20/50 (40.0%) | 18/50 | 2/32 | 6.2% |
| finger | fast_lewam | 45/50 (90.0%) | 42/50 | 3/8 | 37.5% |
| finger | lewm | 44/50 (88.0%) | 42/50 | 2/8 | 25.0% |
| finger | leflow | 44/50 (88.0%) | 42/50 | 2/8 | 25.0% |
| humanoid | fast_lewam | 10/50 (20.0%) | 8/50 | 2/42 | 4.8% |
| humanoid | lewm | 8/50 (16.0%) | 8/50 | 0/42 | 0.0% |
| humanoid | leflow | 11/50 (22.0%) | 8/50 | 3/42 | 7.1% |

## 主比较

主比较固定为 FastLeWAM R4-AB 的 P3 / action-flow step 1 / no guidance；LeWM 使用 CEM 300/30/30，LeFlow 使用 64 paths / 16 flow steps。

| task | FastLeWAM P3/s1/none | LeWM | LeFlow |
|---|---:|---:|---:|
| scene | 3/50 (6.0%, CI [2.1, 16.2]) | 9/50 (18.0%, CI [9.8, 30.8]) | 5/50 (10.0%, CI [4.3, 21.4]) |
| finger | 14/50 (28.0%, CI [17.5, 41.7]) | 16/50 (32.0%, CI [20.8, 45.8]) | 23/50 (46.0%, CI [33.0, 59.6]) |
| humanoid | 5/50 (10.0%, CI [4.3, 21.4]) | 1/50 (2.0%, CI [0.4, 10.5]) | 5/50 (10.0%, CI [4.3, 21.4]) |

| task | baseline | Δ pp (Fast − baseline) | improved | regressed | McNemar p |
|---|---|---:|---:|---:|---:|
| scene | lewm | -12.0 | 2 | 8 | 0.109375 |
| scene | leflow | -4.0 | 0 | 2 | 0.500000 |
| finger | lewm | -4.0 | 4 | 6 | 0.753906 |
| finger | leflow | -18.0 | 2 | 11 | 0.022461 |
| humanoid | lewm | 8.0 | 4 | 0 | 0.125000 |
| humanoid | leflow | 0.0 | 1 | 1 | 1.000000 |

## 完整参数矩阵成功率

下表逐项列出所有 FastLeWAM 参数条件及 LeWM、LeFlow 标准基线。FastLeWAM 的 condition 字段保留完整配置键（mode / protocol / guidance / step / sampler；不适用项按配置记为 `none` 或 `not_applicable`）；成功率以该任务固定 cohort 的全部 episode 为分母。

| task | method | condition（完整参数键） | 成功数 / episode 数 | 成功率 |
|---|---|---|---:|---:|
| scene | fast_lewam | `P0/not_applicable/guided_flow/step_1/euler` | 4/50 | 8.0% |
| scene | fast_lewam | `P0/not_applicable/guided_flow/step_10/euler` | 5/50 | 10.0% |
| scene | fast_lewam | `P0/not_applicable/guided_flow/step_16/euler` | 5/50 | 10.0% |
| scene | fast_lewam | `P0/not_applicable/guided_flow/step_2/euler` | 4/50 | 8.0% |
| scene | fast_lewam | `P0/not_applicable/guided_flow/step_32/euler` | 6/50 | 12.0% |
| scene | fast_lewam | `P0/not_applicable/guided_flow/step_5/euler` | 6/50 | 12.0% |
| scene | fast_lewam | `P0/not_applicable/none/step_1/euler` | 4/50 | 8.0% |
| scene | fast_lewam | `P0/not_applicable/none/step_10/euler` | 5/50 | 10.0% |
| scene | fast_lewam | `P0/not_applicable/none/step_16/euler` | 5/50 | 10.0% |
| scene | fast_lewam | `P0/not_applicable/none/step_2/euler` | 4/50 | 8.0% |
| scene | fast_lewam | `P0/not_applicable/none/step_32/euler` | 6/50 | 12.0% |
| scene | fast_lewam | `P0/not_applicable/none/step_5/euler` | 6/50 | 12.0% |
| scene | fast_lewam | `P0/not_applicable/post_opt/step_1/euler` | 4/50 | 8.0% |
| scene | fast_lewam | `P0/not_applicable/post_opt/step_10/euler` | 5/50 | 10.0% |
| scene | fast_lewam | `P0/not_applicable/post_opt/step_16/euler` | 5/50 | 10.0% |
| scene | fast_lewam | `P0/not_applicable/post_opt/step_2/euler` | 4/50 | 8.0% |
| scene | fast_lewam | `P0/not_applicable/post_opt/step_32/euler` | 6/50 | 12.0% |
| scene | fast_lewam | `P0/not_applicable/post_opt/step_5/euler` | 6/50 | 12.0% |
| scene | fast_lewam | `P1/legacy/none/invariant/not_applicable` | 6/50 | 12.0% |
| scene | fast_lewam | `P2/legacy/guided_flow/step_1/euler` | 7/50 | 14.0% |
| scene | fast_lewam | `P2/legacy/guided_flow/step_10/euler` | 5/50 | 10.0% |
| scene | fast_lewam | `P2/legacy/guided_flow/step_16/euler` | 6/50 | 12.0% |
| scene | fast_lewam | `P2/legacy/guided_flow/step_2/euler` | 6/50 | 12.0% |
| scene | fast_lewam | `P2/legacy/guided_flow/step_32/euler` | 8/50 | 16.0% |
| scene | fast_lewam | `P2/legacy/guided_flow/step_5/euler` | 7/50 | 14.0% |
| scene | fast_lewam | `P2/legacy/none/step_1/euler` | 7/50 | 14.0% |
| scene | fast_lewam | `P2/legacy/none/step_10/euler` | 5/50 | 10.0% |
| scene | fast_lewam | `P2/legacy/none/step_16/euler` | 6/50 | 12.0% |
| scene | fast_lewam | `P2/legacy/none/step_2/euler` | 6/50 | 12.0% |
| scene | fast_lewam | `P2/legacy/none/step_32/euler` | 8/50 | 16.0% |
| scene | fast_lewam | `P2/legacy/none/step_5/euler` | 7/50 | 14.0% |
| scene | fast_lewam | `P2/legacy/post_opt/step_1/euler` | 7/50 | 14.0% |
| scene | fast_lewam | `P2/legacy/post_opt/step_10/euler` | 5/50 | 10.0% |
| scene | fast_lewam | `P2/legacy/post_opt/step_16/euler` | 6/50 | 12.0% |
| scene | fast_lewam | `P2/legacy/post_opt/step_2/euler` | 6/50 | 12.0% |
| scene | fast_lewam | `P2/legacy/post_opt/step_32/euler` | 8/50 | 16.0% |
| scene | fast_lewam | `P2/legacy/post_opt/step_5/euler` | 7/50 | 14.0% |
| scene | fast_lewam | `P3/not_applicable/guided_flow/step_1/euler` | 3/50 | 6.0% |
| scene | fast_lewam | `P3/not_applicable/guided_flow/step_10/euler` | 4/50 | 8.0% |
| scene | fast_lewam | `P3/not_applicable/guided_flow/step_16/euler` | 6/50 | 12.0% |
| scene | fast_lewam | `P3/not_applicable/guided_flow/step_2/euler` | 3/50 | 6.0% |
| scene | fast_lewam | `P3/not_applicable/guided_flow/step_32/euler` | 5/50 | 10.0% |
| scene | fast_lewam | `P3/not_applicable/guided_flow/step_5/euler` | 5/50 | 10.0% |
| scene | fast_lewam | `P3/not_applicable/none/step_1/euler` | 3/50 | 6.0% |
| scene | fast_lewam | `P3/not_applicable/none/step_10/euler` | 4/50 | 8.0% |
| scene | fast_lewam | `P3/not_applicable/none/step_16/euler` | 6/50 | 12.0% |
| scene | fast_lewam | `P3/not_applicable/none/step_2/euler` | 3/50 | 6.0% |
| scene | fast_lewam | `P3/not_applicable/none/step_32/euler` | 5/50 | 10.0% |
| scene | fast_lewam | `P3/not_applicable/none/step_5/euler` | 5/50 | 10.0% |
| scene | fast_lewam | `P3/not_applicable/post_opt/step_1/euler` | 3/50 | 6.0% |
| scene | fast_lewam | `P3/not_applicable/post_opt/step_10/euler` | 4/50 | 8.0% |
| scene | fast_lewam | `P3/not_applicable/post_opt/step_16/euler` | 6/50 | 12.0% |
| scene | fast_lewam | `P3/not_applicable/post_opt/step_2/euler` | 3/50 | 6.0% |
| scene | fast_lewam | `P3/not_applicable/post_opt/step_32/euler` | 5/50 | 10.0% |
| scene | fast_lewam | `P3/not_applicable/post_opt/step_5/euler` | 5/50 | 10.0% |
| scene | fast_lewam | `P3/not_applicable/post_opt_refine/step_1/euler` | 3/50 | 6.0% |
| scene | fast_lewam | `P3/not_applicable/post_opt_refine/step_10/euler` | 4/50 | 8.0% |
| scene | fast_lewam | `P3/not_applicable/post_opt_refine/step_16/euler` | 6/50 | 12.0% |
| scene | fast_lewam | `P3/not_applicable/post_opt_refine/step_2/euler` | 3/50 | 6.0% |
| scene | fast_lewam | `P3/not_applicable/post_opt_refine/step_32/euler` | 5/50 | 10.0% |
| scene | fast_lewam | `P3/not_applicable/post_opt_refine/step_5/euler` | 5/50 | 10.0% |
| scene | lewm | `standard` | 9/50 | 18.0% |
| scene | leflow | `standard` | 5/50 | 10.0% |
| finger | fast_lewam | `P0/not_applicable/guided_flow/step_1/euler` | 14/50 | 28.0% |
| finger | fast_lewam | `P0/not_applicable/guided_flow/step_10/euler` | 18/50 | 36.0% |
| finger | fast_lewam | `P0/not_applicable/guided_flow/step_16/euler` | 15/50 | 30.0% |
| finger | fast_lewam | `P0/not_applicable/guided_flow/step_2/euler` | 17/50 | 34.0% |
| finger | fast_lewam | `P0/not_applicable/guided_flow/step_32/euler` | 17/50 | 34.0% |
| finger | fast_lewam | `P0/not_applicable/guided_flow/step_5/euler` | 17/50 | 34.0% |
| finger | fast_lewam | `P0/not_applicable/none/step_1/euler` | 14/50 | 28.0% |
| finger | fast_lewam | `P0/not_applicable/none/step_10/euler` | 16/50 | 32.0% |
| finger | fast_lewam | `P0/not_applicable/none/step_16/euler` | 17/50 | 34.0% |
| finger | fast_lewam | `P0/not_applicable/none/step_2/euler` | 11/50 | 22.0% |
| finger | fast_lewam | `P0/not_applicable/none/step_32/euler` | 17/50 | 34.0% |
| finger | fast_lewam | `P0/not_applicable/none/step_5/euler` | 13/50 | 26.0% |
| finger | fast_lewam | `P0/not_applicable/post_opt/step_1/euler` | 14/50 | 28.0% |
| finger | fast_lewam | `P0/not_applicable/post_opt/step_10/euler` | 16/50 | 32.0% |
| finger | fast_lewam | `P0/not_applicable/post_opt/step_16/euler` | 15/50 | 30.0% |
| finger | fast_lewam | `P0/not_applicable/post_opt/step_2/euler` | 13/50 | 26.0% |
| finger | fast_lewam | `P0/not_applicable/post_opt/step_32/euler` | 17/50 | 34.0% |
| finger | fast_lewam | `P0/not_applicable/post_opt/step_5/euler` | 13/50 | 26.0% |
| finger | fast_lewam | `P1/cem-clip/none/invariant/not_applicable` | 17/50 | 34.0% |
| finger | fast_lewam | `P2/cem-clip/guided_flow/step_1/euler` | 20/50 | 40.0% |
| finger | fast_lewam | `P2/cem-clip/guided_flow/step_10/euler` | 23/50 | 46.0% |
| finger | fast_lewam | `P2/cem-clip/guided_flow/step_16/euler` | 19/50 | 38.0% |
| finger | fast_lewam | `P2/cem-clip/guided_flow/step_2/euler` | 21/50 | 42.0% |
| finger | fast_lewam | `P2/cem-clip/guided_flow/step_32/euler` | 24/50 | 48.0% |
| finger | fast_lewam | `P2/cem-clip/guided_flow/step_5/euler` | 22/50 | 44.0% |
| finger | fast_lewam | `P2/cem-clip/none/step_1/euler` | 19/50 | 38.0% |
| finger | fast_lewam | `P2/cem-clip/none/step_10/euler` | 22/50 | 44.0% |
| finger | fast_lewam | `P2/cem-clip/none/step_16/euler` | 20/50 | 40.0% |
| finger | fast_lewam | `P2/cem-clip/none/step_2/euler` | 21/50 | 42.0% |
| finger | fast_lewam | `P2/cem-clip/none/step_32/euler` | 20/50 | 40.0% |
| finger | fast_lewam | `P2/cem-clip/none/step_5/euler` | 22/50 | 44.0% |
| finger | fast_lewam | `P2/cem-clip/post_opt/step_1/euler` | 20/50 | 40.0% |
| finger | fast_lewam | `P2/cem-clip/post_opt/step_10/euler` | 19/50 | 38.0% |
| finger | fast_lewam | `P2/cem-clip/post_opt/step_16/euler` | 21/50 | 42.0% |
| finger | fast_lewam | `P2/cem-clip/post_opt/step_2/euler` | 19/50 | 38.0% |
| finger | fast_lewam | `P2/cem-clip/post_opt/step_32/euler` | 20/50 | 40.0% |
| finger | fast_lewam | `P2/cem-clip/post_opt/step_5/euler` | 20/50 | 40.0% |
| finger | fast_lewam | `P3/not_applicable/guided_flow/step_1/euler` | 15/50 | 30.0% |
| finger | fast_lewam | `P3/not_applicable/guided_flow/step_10/euler` | 21/50 | 42.0% |
| finger | fast_lewam | `P3/not_applicable/guided_flow/step_16/euler` | 23/50 | 46.0% |
| finger | fast_lewam | `P3/not_applicable/guided_flow/step_2/euler` | 25/50 | 50.0% |
| finger | fast_lewam | `P3/not_applicable/guided_flow/step_32/euler` | 21/50 | 42.0% |
| finger | fast_lewam | `P3/not_applicable/guided_flow/step_5/euler` | 25/50 | 50.0% |
| finger | fast_lewam | `P3/not_applicable/none/step_1/euler` | 14/50 | 28.0% |
| finger | fast_lewam | `P3/not_applicable/none/step_10/euler` | 26/50 | 52.0% |
| finger | fast_lewam | `P3/not_applicable/none/step_16/euler` | 26/50 | 52.0% |
| finger | fast_lewam | `P3/not_applicable/none/step_2/euler` | 22/50 | 44.0% |
| finger | fast_lewam | `P3/not_applicable/none/step_32/euler` | 26/50 | 52.0% |
| finger | fast_lewam | `P3/not_applicable/none/step_5/euler` | 24/50 | 48.0% |
| finger | fast_lewam | `P3/not_applicable/post_opt/step_1/euler` | 15/50 | 30.0% |
| finger | fast_lewam | `P3/not_applicable/post_opt/step_10/euler` | 24/50 | 48.0% |
| finger | fast_lewam | `P3/not_applicable/post_opt/step_16/euler` | 25/50 | 50.0% |
| finger | fast_lewam | `P3/not_applicable/post_opt/step_2/euler` | 22/50 | 44.0% |
| finger | fast_lewam | `P3/not_applicable/post_opt/step_32/euler` | 24/50 | 48.0% |
| finger | fast_lewam | `P3/not_applicable/post_opt/step_5/euler` | 23/50 | 46.0% |
| finger | fast_lewam | `P3/not_applicable/post_opt_refine/step_1/euler` | 15/50 | 30.0% |
| finger | fast_lewam | `P3/not_applicable/post_opt_refine/step_10/euler` | 26/50 | 52.0% |
| finger | fast_lewam | `P3/not_applicable/post_opt_refine/step_16/euler` | 28/50 | 56.0% |
| finger | fast_lewam | `P3/not_applicable/post_opt_refine/step_2/euler` | 23/50 | 46.0% |
| finger | fast_lewam | `P3/not_applicable/post_opt_refine/step_32/euler` | 25/50 | 50.0% |
| finger | fast_lewam | `P3/not_applicable/post_opt_refine/step_5/euler` | 24/50 | 48.0% |
| finger | lewm | `standard` | 16/50 | 32.0% |
| finger | leflow | `standard` | 23/50 | 46.0% |
| humanoid | fast_lewam | `P0/not_applicable/guided_flow/step_1/euler` | 6/50 | 12.0% |
| humanoid | fast_lewam | `P0/not_applicable/guided_flow/step_10/euler` | 5/50 | 10.0% |
| humanoid | fast_lewam | `P0/not_applicable/guided_flow/step_16/euler` | 4/50 | 8.0% |
| humanoid | fast_lewam | `P0/not_applicable/guided_flow/step_2/euler` | 6/50 | 12.0% |
| humanoid | fast_lewam | `P0/not_applicable/guided_flow/step_32/euler` | 4/50 | 8.0% |
| humanoid | fast_lewam | `P0/not_applicable/guided_flow/step_5/euler` | 5/50 | 10.0% |
| humanoid | fast_lewam | `P0/not_applicable/none/step_1/euler` | 6/50 | 12.0% |
| humanoid | fast_lewam | `P0/not_applicable/none/step_10/euler` | 4/50 | 8.0% |
| humanoid | fast_lewam | `P0/not_applicable/none/step_16/euler` | 4/50 | 8.0% |
| humanoid | fast_lewam | `P0/not_applicable/none/step_2/euler` | 8/50 | 16.0% |
| humanoid | fast_lewam | `P0/not_applicable/none/step_32/euler` | 4/50 | 8.0% |
| humanoid | fast_lewam | `P0/not_applicable/none/step_5/euler` | 5/50 | 10.0% |
| humanoid | fast_lewam | `P0/not_applicable/post_opt/step_1/euler` | 6/50 | 12.0% |
| humanoid | fast_lewam | `P0/not_applicable/post_opt/step_10/euler` | 5/50 | 10.0% |
| humanoid | fast_lewam | `P0/not_applicable/post_opt/step_16/euler` | 4/50 | 8.0% |
| humanoid | fast_lewam | `P0/not_applicable/post_opt/step_2/euler` | 9/50 | 18.0% |
| humanoid | fast_lewam | `P0/not_applicable/post_opt/step_32/euler` | 5/50 | 10.0% |
| humanoid | fast_lewam | `P0/not_applicable/post_opt/step_5/euler` | 5/50 | 10.0% |
| humanoid | fast_lewam | `P1/cem-clip/none/invariant/not_applicable` | 2/50 | 4.0% |
| humanoid | fast_lewam | `P2/cem-clip/guided_flow/step_1/euler` | 4/50 | 8.0% |
| humanoid | fast_lewam | `P2/cem-clip/guided_flow/step_10/euler` | 2/50 | 4.0% |
| humanoid | fast_lewam | `P2/cem-clip/guided_flow/step_16/euler` | 4/50 | 8.0% |
| humanoid | fast_lewam | `P2/cem-clip/guided_flow/step_2/euler` | 3/50 | 6.0% |
| humanoid | fast_lewam | `P2/cem-clip/guided_flow/step_32/euler` | 4/50 | 8.0% |
| humanoid | fast_lewam | `P2/cem-clip/guided_flow/step_5/euler` | 3/50 | 6.0% |
| humanoid | fast_lewam | `P2/cem-clip/none/step_1/euler` | 5/50 | 10.0% |
| humanoid | fast_lewam | `P2/cem-clip/none/step_10/euler` | 4/50 | 8.0% |
| humanoid | fast_lewam | `P2/cem-clip/none/step_16/euler` | 4/50 | 8.0% |
| humanoid | fast_lewam | `P2/cem-clip/none/step_2/euler` | 5/50 | 10.0% |
| humanoid | fast_lewam | `P2/cem-clip/none/step_32/euler` | 4/50 | 8.0% |
| humanoid | fast_lewam | `P2/cem-clip/none/step_5/euler` | 3/50 | 6.0% |
| humanoid | fast_lewam | `P2/cem-clip/post_opt/step_1/euler` | 4/50 | 8.0% |
| humanoid | fast_lewam | `P2/cem-clip/post_opt/step_10/euler` | 3/50 | 6.0% |
| humanoid | fast_lewam | `P2/cem-clip/post_opt/step_16/euler` | 3/50 | 6.0% |
| humanoid | fast_lewam | `P2/cem-clip/post_opt/step_2/euler` | 4/50 | 8.0% |
| humanoid | fast_lewam | `P2/cem-clip/post_opt/step_32/euler` | 4/50 | 8.0% |
| humanoid | fast_lewam | `P2/cem-clip/post_opt/step_5/euler` | 3/50 | 6.0% |
| humanoid | fast_lewam | `P3/not_applicable/guided_flow/step_1/euler` | 5/50 | 10.0% |
| humanoid | fast_lewam | `P3/not_applicable/guided_flow/step_10/euler` | 6/50 | 12.0% |
| humanoid | fast_lewam | `P3/not_applicable/guided_flow/step_16/euler` | 7/50 | 14.0% |
| humanoid | fast_lewam | `P3/not_applicable/guided_flow/step_2/euler` | 6/50 | 12.0% |
| humanoid | fast_lewam | `P3/not_applicable/guided_flow/step_32/euler` | 8/50 | 16.0% |
| humanoid | fast_lewam | `P3/not_applicable/guided_flow/step_5/euler` | 7/50 | 14.0% |
| humanoid | fast_lewam | `P3/not_applicable/none/step_1/euler` | 5/50 | 10.0% |
| humanoid | fast_lewam | `P3/not_applicable/none/step_10/euler` | 6/50 | 12.0% |
| humanoid | fast_lewam | `P3/not_applicable/none/step_16/euler` | 6/50 | 12.0% |
| humanoid | fast_lewam | `P3/not_applicable/none/step_2/euler` | 6/50 | 12.0% |
| humanoid | fast_lewam | `P3/not_applicable/none/step_32/euler` | 7/50 | 14.0% |
| humanoid | fast_lewam | `P3/not_applicable/none/step_5/euler` | 6/50 | 12.0% |
| humanoid | fast_lewam | `P3/not_applicable/post_opt/step_1/euler` | 5/50 | 10.0% |
| humanoid | fast_lewam | `P3/not_applicable/post_opt/step_10/euler` | 9/50 | 18.0% |
| humanoid | fast_lewam | `P3/not_applicable/post_opt/step_16/euler` | 7/50 | 14.0% |
| humanoid | fast_lewam | `P3/not_applicable/post_opt/step_2/euler` | 5/50 | 10.0% |
| humanoid | fast_lewam | `P3/not_applicable/post_opt/step_32/euler` | 7/50 | 14.0% |
| humanoid | fast_lewam | `P3/not_applicable/post_opt/step_5/euler` | 6/50 | 12.0% |
| humanoid | fast_lewam | `P3/not_applicable/post_opt_refine/step_1/euler` | 5/50 | 10.0% |
| humanoid | fast_lewam | `P3/not_applicable/post_opt_refine/step_10/euler` | 6/50 | 12.0% |
| humanoid | fast_lewam | `P3/not_applicable/post_opt_refine/step_16/euler` | 6/50 | 12.0% |
| humanoid | fast_lewam | `P3/not_applicable/post_opt_refine/step_2/euler` | 5/50 | 10.0% |
| humanoid | fast_lewam | `P3/not_applicable/post_opt_refine/step_32/euler` | 7/50 | 14.0% |
| humanoid | fast_lewam | `P3/not_applicable/post_opt_refine/step_5/euler` | 6/50 | 12.0% |
| humanoid | lewm | `standard` | 1/50 | 2.0% |
| humanoid | leflow | `standard` | 5/50 | 10.0% |

## Flow-step 敏感性（无 guidance）

| task | mode | step | successes/50 | success rate |
|---|---|---:|---:|---:|
| scene | P0 | 1 | 4/50 | 8.0% |
| scene | P0 | 2 | 4/50 | 8.0% |
| scene | P0 | 5 | 6/50 | 12.0% |
| scene | P0 | 10 | 5/50 | 10.0% |
| scene | P0 | 16 | 5/50 | 10.0% |
| scene | P0 | 32 | 6/50 | 12.0% |
| scene | P2 | 1 | 7/50 | 14.0% |
| scene | P2 | 2 | 6/50 | 12.0% |
| scene | P2 | 5 | 7/50 | 14.0% |
| scene | P2 | 10 | 5/50 | 10.0% |
| scene | P2 | 16 | 6/50 | 12.0% |
| scene | P2 | 32 | 8/50 | 16.0% |
| scene | P3 | 1 | 3/50 | 6.0% |
| scene | P3 | 2 | 3/50 | 6.0% |
| scene | P3 | 5 | 5/50 | 10.0% |
| scene | P3 | 10 | 4/50 | 8.0% |
| scene | P3 | 16 | 6/50 | 12.0% |
| scene | P3 | 32 | 5/50 | 10.0% |
| finger | P0 | 1 | 14/50 | 28.0% |
| finger | P0 | 2 | 11/50 | 22.0% |
| finger | P0 | 5 | 13/50 | 26.0% |
| finger | P0 | 10 | 16/50 | 32.0% |
| finger | P0 | 16 | 17/50 | 34.0% |
| finger | P0 | 32 | 17/50 | 34.0% |
| finger | P2 | 1 | 19/50 | 38.0% |
| finger | P2 | 2 | 21/50 | 42.0% |
| finger | P2 | 5 | 22/50 | 44.0% |
| finger | P2 | 10 | 22/50 | 44.0% |
| finger | P2 | 16 | 20/50 | 40.0% |
| finger | P2 | 32 | 20/50 | 40.0% |
| finger | P3 | 1 | 14/50 | 28.0% |
| finger | P3 | 2 | 22/50 | 44.0% |
| finger | P3 | 5 | 24/50 | 48.0% |
| finger | P3 | 10 | 26/50 | 52.0% |
| finger | P3 | 16 | 26/50 | 52.0% |
| finger | P3 | 32 | 26/50 | 52.0% |
| humanoid | P0 | 1 | 6/50 | 12.0% |
| humanoid | P0 | 2 | 8/50 | 16.0% |
| humanoid | P0 | 5 | 5/50 | 10.0% |
| humanoid | P0 | 10 | 4/50 | 8.0% |
| humanoid | P0 | 16 | 4/50 | 8.0% |
| humanoid | P0 | 32 | 4/50 | 8.0% |
| humanoid | P2 | 1 | 5/50 | 10.0% |
| humanoid | P2 | 2 | 5/50 | 10.0% |
| humanoid | P2 | 5 | 3/50 | 6.0% |
| humanoid | P2 | 10 | 4/50 | 8.0% |
| humanoid | P2 | 16 | 4/50 | 8.0% |
| humanoid | P2 | 32 | 4/50 | 8.0% |
| humanoid | P3 | 1 | 5/50 | 10.0% |
| humanoid | P3 | 2 | 6/50 | 12.0% |
| humanoid | P3 | 5 | 6/50 | 12.0% |
| humanoid | P3 | 10 | 6/50 | 12.0% |
| humanoid | P3 | 16 | 6/50 | 12.0% |
| humanoid | P3 | 32 | 7/50 | 14.0% |

## Guidance 相对无 guidance 的变化

| task | mode | guidance | step | Δ pp |
|---|---|---|---:|---:|
| scene | P0 | guided_flow | 1 | +0.0 |
| scene | P0 | post_opt | 1 | +0.0 |
| scene | P0 | guided_flow | 2 | +0.0 |
| scene | P0 | post_opt | 2 | +0.0 |
| scene | P0 | guided_flow | 5 | +0.0 |
| scene | P0 | post_opt | 5 | +0.0 |
| scene | P0 | guided_flow | 10 | +0.0 |
| scene | P0 | post_opt | 10 | +0.0 |
| scene | P0 | guided_flow | 16 | +0.0 |
| scene | P0 | post_opt | 16 | +0.0 |
| scene | P0 | guided_flow | 32 | +0.0 |
| scene | P0 | post_opt | 32 | +0.0 |
| scene | P2 | guided_flow | 1 | +0.0 |
| scene | P2 | post_opt | 1 | +0.0 |
| scene | P2 | guided_flow | 2 | +0.0 |
| scene | P2 | post_opt | 2 | +0.0 |
| scene | P2 | guided_flow | 5 | +0.0 |
| scene | P2 | post_opt | 5 | +0.0 |
| scene | P2 | guided_flow | 10 | +0.0 |
| scene | P2 | post_opt | 10 | +0.0 |
| scene | P2 | guided_flow | 16 | +0.0 |
| scene | P2 | post_opt | 16 | +0.0 |
| scene | P2 | guided_flow | 32 | +0.0 |
| scene | P2 | post_opt | 32 | +0.0 |
| scene | P3 | guided_flow | 1 | +0.0 |
| scene | P3 | post_opt | 1 | +0.0 |
| scene | P3 | post_opt_refine | 1 | +0.0 |
| scene | P3 | guided_flow | 2 | +0.0 |
| scene | P3 | post_opt | 2 | +0.0 |
| scene | P3 | post_opt_refine | 2 | +0.0 |
| scene | P3 | guided_flow | 5 | +0.0 |
| scene | P3 | post_opt | 5 | +0.0 |
| scene | P3 | post_opt_refine | 5 | +0.0 |
| scene | P3 | guided_flow | 10 | +0.0 |
| scene | P3 | post_opt | 10 | +0.0 |
| scene | P3 | post_opt_refine | 10 | +0.0 |
| scene | P3 | guided_flow | 16 | +0.0 |
| scene | P3 | post_opt | 16 | +0.0 |
| scene | P3 | post_opt_refine | 16 | +0.0 |
| scene | P3 | guided_flow | 32 | +0.0 |
| scene | P3 | post_opt | 32 | +0.0 |
| scene | P3 | post_opt_refine | 32 | +0.0 |
| finger | P0 | guided_flow | 1 | +0.0 |
| finger | P0 | post_opt | 1 | +0.0 |
| finger | P0 | guided_flow | 2 | +12.0 |
| finger | P0 | post_opt | 2 | +4.0 |
| finger | P0 | guided_flow | 5 | +8.0 |
| finger | P0 | post_opt | 5 | +0.0 |
| finger | P0 | guided_flow | 10 | +4.0 |
| finger | P0 | post_opt | 10 | +0.0 |
| finger | P0 | guided_flow | 16 | -4.0 |
| finger | P0 | post_opt | 16 | -4.0 |
| finger | P0 | guided_flow | 32 | +0.0 |
| finger | P0 | post_opt | 32 | +0.0 |
| finger | P2 | guided_flow | 1 | +2.0 |
| finger | P2 | post_opt | 1 | +2.0 |
| finger | P2 | guided_flow | 2 | +0.0 |
| finger | P2 | post_opt | 2 | -4.0 |
| finger | P2 | guided_flow | 5 | +0.0 |
| finger | P2 | post_opt | 5 | -4.0 |
| finger | P2 | guided_flow | 10 | +2.0 |
| finger | P2 | post_opt | 10 | -6.0 |
| finger | P2 | guided_flow | 16 | -2.0 |
| finger | P2 | post_opt | 16 | +2.0 |
| finger | P2 | guided_flow | 32 | +8.0 |
| finger | P2 | post_opt | 32 | +0.0 |
| finger | P3 | guided_flow | 1 | +2.0 |
| finger | P3 | post_opt | 1 | +2.0 |
| finger | P3 | post_opt_refine | 1 | +2.0 |
| finger | P3 | guided_flow | 2 | +6.0 |
| finger | P3 | post_opt | 2 | +0.0 |
| finger | P3 | post_opt_refine | 2 | +2.0 |
| finger | P3 | guided_flow | 5 | +2.0 |
| finger | P3 | post_opt | 5 | -2.0 |
| finger | P3 | post_opt_refine | 5 | +0.0 |
| finger | P3 | guided_flow | 10 | -10.0 |
| finger | P3 | post_opt | 10 | -4.0 |
| finger | P3 | post_opt_refine | 10 | +0.0 |
| finger | P3 | guided_flow | 16 | -6.0 |
| finger | P3 | post_opt | 16 | -2.0 |
| finger | P3 | post_opt_refine | 16 | +4.0 |
| finger | P3 | guided_flow | 32 | -10.0 |
| finger | P3 | post_opt | 32 | -4.0 |
| finger | P3 | post_opt_refine | 32 | -2.0 |
| humanoid | P0 | guided_flow | 1 | +0.0 |
| humanoid | P0 | post_opt | 1 | +0.0 |
| humanoid | P0 | guided_flow | 2 | -4.0 |
| humanoid | P0 | post_opt | 2 | +2.0 |
| humanoid | P0 | guided_flow | 5 | +0.0 |
| humanoid | P0 | post_opt | 5 | +0.0 |
| humanoid | P0 | guided_flow | 10 | +2.0 |
| humanoid | P0 | post_opt | 10 | +2.0 |
| humanoid | P0 | guided_flow | 16 | +0.0 |
| humanoid | P0 | post_opt | 16 | +0.0 |
| humanoid | P0 | guided_flow | 32 | +0.0 |
| humanoid | P0 | post_opt | 32 | +2.0 |
| humanoid | P2 | guided_flow | 1 | -2.0 |
| humanoid | P2 | post_opt | 1 | -2.0 |
| humanoid | P2 | guided_flow | 2 | -4.0 |
| humanoid | P2 | post_opt | 2 | -2.0 |
| humanoid | P2 | guided_flow | 5 | +0.0 |
| humanoid | P2 | post_opt | 5 | +0.0 |
| humanoid | P2 | guided_flow | 10 | -4.0 |
| humanoid | P2 | post_opt | 10 | -2.0 |
| humanoid | P2 | guided_flow | 16 | +0.0 |
| humanoid | P2 | post_opt | 16 | -2.0 |
| humanoid | P2 | guided_flow | 32 | +0.0 |
| humanoid | P2 | post_opt | 32 | +0.0 |
| humanoid | P3 | guided_flow | 1 | +0.0 |
| humanoid | P3 | post_opt | 1 | +0.0 |
| humanoid | P3 | post_opt_refine | 1 | +0.0 |
| humanoid | P3 | guided_flow | 2 | +0.0 |
| humanoid | P3 | post_opt | 2 | -2.0 |
| humanoid | P3 | post_opt_refine | 2 | -2.0 |
| humanoid | P3 | guided_flow | 5 | +2.0 |
| humanoid | P3 | post_opt | 5 | +0.0 |
| humanoid | P3 | post_opt_refine | 5 | +0.0 |
| humanoid | P3 | guided_flow | 10 | +0.0 |
| humanoid | P3 | post_opt | 10 | +6.0 |
| humanoid | P3 | post_opt_refine | 10 | +0.0 |
| humanoid | P3 | guided_flow | 16 | +2.0 |
| humanoid | P3 | post_opt | 16 | +2.0 |
| humanoid | P3 | post_opt_refine | 16 | +0.0 |
| humanoid | P3 | guided_flow | 32 | +2.0 |
| humanoid | P3 | post_opt | 32 | +0.0 |
| humanoid | P3 | post_opt_refine | 32 | +0.0 |

## 条件矩阵验收

- 结果数：189 / 189。
- 每个结果：50 episodes；成功率来自 runtime termination。
- 三个任务共享各自固定 cohort；每项结果都必须携带与冻结 manifest 完全一致的 cohort hash。
- DMC 的 NaN `success` 列未被用作指标。

## Scene 目标行修正

Scene 成功目标与规划使用的未来 goal row 对齐；此前按 episode 级 privileged target 判定的旧结果已归档，不纳入本报告。分组标签仍取该 goal row 的 `privileged_target_task`，只用于分组统计。详见 `docs/report/round5/round5_phase3_scene_goalrow_recheck.md`。

## 训练配置与种子

所有结果均记录训练 seed 3072、评测 seed 42，并通过配置文件路径和 SHA-256 固定对应训练配置。
- finger / fast_lewam: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/training/fastlewam/finger/config.yaml, SHA-256 e481371bfe05d045cfa218e6842cb99e333753b37c7adaa49b44c9ae653149c8.
- finger / leflow: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/training/leflow/finger/config.json, SHA-256 414805a83731b26f2242dfce66426c7bbd4413c3c65833827f4ffa60f72e8a90.
- finger / lewm: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/training/lewm/finger/config.yaml, SHA-256 939a1d10c60254230aa97ed4a0a78980b468dd73c906d18343dceafb9b37b77a.
- humanoid / fast_lewam: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/training/fastlewam/humanoid/config.yaml, SHA-256 6562316d3f3ce63b56e7d00aa13d1b1b0cff7c367d97f237e925751c75930fb8.
- humanoid / leflow: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/training/leflow/humanoid/config.json, SHA-256 839aa932ffc863049654985f433d8f31bda1e19f4f69b5ac3ab6637cd1dfb520.
- humanoid / lewm: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/training/lewm/humanoid/config.yaml, SHA-256 5f44ce1021c6777728de1fb3c92def738a3635d09addcaea161609b9e1fca80c.
- scene / fast_lewam: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/training/fastlewam/scene/config.yaml, SHA-256 0646b50eaedf927431552198572b0ce603faa7233479d410468f7ee1f25929c1.
- scene / leflow: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/training/leflow/scene/config.json, SHA-256 202f34eb267d072a188ace5e0b8bc262b9561c57f7a946a51a0bbe2a4e442d15.
- scene / lewm: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/training/lewm/scene/config.yaml, SHA-256 f1780a3b3efb7dae078dd429684763052c7e6bc6738a6fcd3da2a027bda241fd.

## Epoch 10 checkpoint 验收

已核验 9 组训练的 epoch 10 checkpoint；LeFlow 还要求完成标记为 ok 且 epochs=10。
- scene / fastlewam: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/training/fastlewam/scene/checkpoints/r4_ab_weights_epoch_10.pt。
- scene / lewm: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/training/lewm/scene/checkpoints/lewm_weights_epoch_10.pt。
- scene / leflow: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/training/leflow/scene/latent_planner_epoch_10.pt。
- finger / fastlewam: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/training/fastlewam/finger/checkpoints/r4_ab_weights_epoch_10.pt。
- finger / lewm: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/training/lewm/finger/checkpoints/lewm_weights_epoch_10.pt。
- finger / leflow: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/training/leflow/finger/latent_planner_epoch_10.pt。
- humanoid / fastlewam: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/training/fastlewam/humanoid/checkpoints/r4_ab_weights_epoch_10.pt。
- humanoid / lewm: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/training/lewm/humanoid/checkpoints/lewm_weights_epoch_10.pt。
- humanoid / leflow: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/training/leflow/humanoid/latent_planner_epoch_10.pt。

## 代表视频验收

共检查 12 段视频；每段对应冻结 cohort 的首个 start–goal 对。
- scene / fastlewam / p1_step_1: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3_initial_fail/videos/fastlewam/scene/p1_step_1/videos/env_0.mp4 (736×288, 90565 bytes)。
- scene / fastlewam / p3_step_1: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3_initial_fail/videos/fastlewam/scene/p3_step_1/videos/env_0.mp4 (736×288, 72720 bytes)。
- scene / lewm / standard: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3_initial_fail/videos/lewm/scene/standard/videos/env_0.mp4 (736×288, 104428 bytes)。
- scene / leflow / standard: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3_initial_fail/videos/leflow/scene/standard/videos/env_0.mp4 (736×288, 97711 bytes)。
- finger / fastlewam / p1_step_1: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3_initial_fail/videos/fastlewam/finger/p1_step_1/videos/env_0.mp4 (736×288, 50915 bytes)。
- finger / fastlewam / p3_step_1: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3_initial_fail/videos/fastlewam/finger/p3_step_1/videos/env_0.mp4 (736×288, 53978 bytes)。
- finger / lewm / standard: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3_initial_fail/videos/lewm/finger/standard/videos/env_0.mp4 (736×288, 56946 bytes)。
- finger / leflow / standard: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3_initial_fail/videos/leflow/finger/standard/videos/env_0.mp4 (736×288, 52817 bytes)。
- humanoid / fastlewam / p1_step_1: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3_initial_fail/videos/fastlewam/humanoid/p1_step_1/videos/env_0.mp4 (736×288, 90525 bytes)。
- humanoid / fastlewam / p3_step_1: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3_initial_fail/videos/fastlewam/humanoid/p3_step_1/videos/env_0.mp4 (736×288, 110369 bytes)。
- humanoid / lewm / standard: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3_initial_fail/videos/lewm/humanoid/standard/videos/env_0.mp4 (736×288, 94644 bytes)。
- humanoid / leflow / standard: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3_initial_fail/videos/leflow/humanoid/standard/videos/env_0.mp4 (736×288, 104028 bytes)。

## LeWM 训练续训记录

以下列出本阶段 LeWM 训练中存在的续训元数据；未记录的字段不作推断。
- `scene`：source run `/home/wenxin/office/pre-exp/le-wm/outputs/round5_phase3_lewm_scene_resume_epoch6`；元数据记录完成至 epoch 10。元数据：`training/lewm/scene/resume_metadata.json`。
- `finger`：source run `/home/wenxin/office/pre-exp/le-wm/outputs/round5_phase3_lewm_finger_resume_epoch9_tmux`；optimizer state restored=false。元数据：`training/lewm/finger/resume_metadata.json`。
- `humanoid`：source run `/home/wenxin/office/pre-exp/le-wm/outputs/round5_phase3_lewm_humanoid_resume_epoch6`；epoch 6 接续至 epoch 10；seed=3072；optimizer state restored=false；续训方式 `model_weights_with_epoch_offset`。元数据：`training/lewm/humanoid/resume_metadata.json`。

## Scene target_task 分组

- `fast_lewam` / `P0/not_applicable/none/step_1/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=1/13 (7.7%)
- `fast_lewam` / `P0/not_applicable/none/step_2/euler`：button=0/11 (0.0%), cube=2/20 (10.0%), drawer=1/6 (16.7%), window=1/13 (7.7%)
- `fast_lewam` / `P0/not_applicable/none/step_5/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=1/6 (16.7%), window=2/13 (15.4%)
- `fast_lewam` / `P0/not_applicable/none/step_10/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=2/13 (15.4%)
- `fast_lewam` / `P0/not_applicable/none/step_16/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=2/13 (15.4%)
- `fast_lewam` / `P0/not_applicable/none/step_32/euler`：button=1/11 (9.1%), cube=3/20 (15.0%), drawer=1/6 (16.7%), window=1/13 (7.7%)
- `fast_lewam` / `P1/legacy/none/invariant/not_applicable`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=1/6 (16.7%), window=2/13 (15.4%)
- `fast_lewam` / `P2/legacy/none/step_1/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=3/6 (50.0%), window=1/13 (7.7%)
- `fast_lewam` / `P2/legacy/none/step_2/euler`：button=2/11 (18.2%), cube=0/20 (0.0%), drawer=2/6 (33.3%), window=2/13 (15.4%)
- `fast_lewam` / `P2/legacy/none/step_5/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=2/6 (33.3%), window=2/13 (15.4%)
- `fast_lewam` / `P2/legacy/none/step_10/euler`：button=1/11 (9.1%), cube=1/20 (5.0%), drawer=1/6 (16.7%), window=2/13 (15.4%)
- `fast_lewam` / `P2/legacy/none/step_16/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=1/6 (16.7%), window=2/13 (15.4%)
- `fast_lewam` / `P2/legacy/none/step_32/euler`：button=1/11 (9.1%), cube=3/20 (15.0%), drawer=2/6 (33.3%), window=2/13 (15.4%)
- `fast_lewam` / `P3/not_applicable/none/step_1/euler`：button=0/11 (0.0%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=1/13 (7.7%)
- `fast_lewam` / `P3/not_applicable/none/step_2/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=0/13 (0.0%)
- `fast_lewam` / `P3/not_applicable/none/step_5/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=1/6 (16.7%), window=1/13 (7.7%)
- `fast_lewam` / `P3/not_applicable/none/step_10/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=1/13 (7.7%)
- `fast_lewam` / `P3/not_applicable/none/step_16/euler`：button=0/11 (0.0%), cube=4/20 (20.0%), drawer=1/6 (16.7%), window=1/13 (7.7%)
- `fast_lewam` / `P3/not_applicable/none/step_32/euler`：button=0/11 (0.0%), cube=3/20 (15.0%), drawer=0/6 (0.0%), window=2/13 (15.4%)
- `fast_lewam` / `P0/not_applicable/guided_flow/step_1/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=1/13 (7.7%)
- `fast_lewam` / `P2/legacy/guided_flow/step_1/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=3/6 (50.0%), window=1/13 (7.7%)
- `fast_lewam` / `P0/not_applicable/post_opt/step_1/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=1/13 (7.7%)
- `fast_lewam` / `P2/legacy/post_opt/step_1/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=3/6 (50.0%), window=1/13 (7.7%)
- `fast_lewam` / `P3/not_applicable/guided_flow/step_1/euler`：button=0/11 (0.0%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=1/13 (7.7%)
- `fast_lewam` / `P3/not_applicable/post_opt/step_1/euler`：button=0/11 (0.0%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=1/13 (7.7%)
- `fast_lewam` / `P3/not_applicable/post_opt_refine/step_1/euler`：button=0/11 (0.0%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=1/13 (7.7%)
- `fast_lewam` / `P0/not_applicable/guided_flow/step_2/euler`：button=0/11 (0.0%), cube=2/20 (10.0%), drawer=1/6 (16.7%), window=1/13 (7.7%)
- `fast_lewam` / `P2/legacy/guided_flow/step_2/euler`：button=2/11 (18.2%), cube=0/20 (0.0%), drawer=2/6 (33.3%), window=2/13 (15.4%)
- `fast_lewam` / `P0/not_applicable/post_opt/step_2/euler`：button=0/11 (0.0%), cube=2/20 (10.0%), drawer=1/6 (16.7%), window=1/13 (7.7%)
- `fast_lewam` / `P2/legacy/post_opt/step_2/euler`：button=2/11 (18.2%), cube=0/20 (0.0%), drawer=2/6 (33.3%), window=2/13 (15.4%)
- `fast_lewam` / `P3/not_applicable/guided_flow/step_2/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=0/13 (0.0%)
- `fast_lewam` / `P3/not_applicable/post_opt/step_2/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=0/13 (0.0%)
- `fast_lewam` / `P3/not_applicable/post_opt_refine/step_2/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=0/13 (0.0%)
- `fast_lewam` / `P0/not_applicable/guided_flow/step_5/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=1/6 (16.7%), window=2/13 (15.4%)
- `fast_lewam` / `P2/legacy/guided_flow/step_5/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=2/6 (33.3%), window=2/13 (15.4%)
- `fast_lewam` / `P0/not_applicable/post_opt/step_5/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=1/6 (16.7%), window=2/13 (15.4%)
- `fast_lewam` / `P2/legacy/post_opt/step_5/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=2/6 (33.3%), window=2/13 (15.4%)
- `fast_lewam` / `P3/not_applicable/guided_flow/step_5/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=1/6 (16.7%), window=1/13 (7.7%)
- `fast_lewam` / `P3/not_applicable/post_opt/step_5/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=1/6 (16.7%), window=1/13 (7.7%)
- `fast_lewam` / `P3/not_applicable/post_opt_refine/step_5/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=1/6 (16.7%), window=1/13 (7.7%)
- `fast_lewam` / `P0/not_applicable/guided_flow/step_10/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=2/13 (15.4%)
- `fast_lewam` / `P2/legacy/guided_flow/step_10/euler`：button=1/11 (9.1%), cube=1/20 (5.0%), drawer=1/6 (16.7%), window=2/13 (15.4%)
- `fast_lewam` / `P0/not_applicable/post_opt/step_10/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=2/13 (15.4%)
- `fast_lewam` / `P2/legacy/post_opt/step_10/euler`：button=1/11 (9.1%), cube=1/20 (5.0%), drawer=1/6 (16.7%), window=2/13 (15.4%)
- `fast_lewam` / `P3/not_applicable/guided_flow/step_10/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=1/13 (7.7%)
- `fast_lewam` / `P3/not_applicable/post_opt/step_10/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=1/13 (7.7%)
- `fast_lewam` / `P3/not_applicable/post_opt_refine/step_10/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=1/13 (7.7%)
- `fast_lewam` / `P0/not_applicable/guided_flow/step_16/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=2/13 (15.4%)
- `fast_lewam` / `P2/legacy/guided_flow/step_16/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=1/6 (16.7%), window=2/13 (15.4%)
- `fast_lewam` / `P0/not_applicable/post_opt/step_16/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=2/13 (15.4%)
- `fast_lewam` / `P2/legacy/post_opt/step_16/euler`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=1/6 (16.7%), window=2/13 (15.4%)
- `fast_lewam` / `P3/not_applicable/guided_flow/step_16/euler`：button=0/11 (0.0%), cube=4/20 (20.0%), drawer=1/6 (16.7%), window=1/13 (7.7%)
- `fast_lewam` / `P3/not_applicable/post_opt/step_16/euler`：button=0/11 (0.0%), cube=4/20 (20.0%), drawer=1/6 (16.7%), window=1/13 (7.7%)
- `fast_lewam` / `P3/not_applicable/post_opt_refine/step_16/euler`：button=0/11 (0.0%), cube=4/20 (20.0%), drawer=1/6 (16.7%), window=1/13 (7.7%)
- `fast_lewam` / `P0/not_applicable/guided_flow/step_32/euler`：button=1/11 (9.1%), cube=3/20 (15.0%), drawer=1/6 (16.7%), window=1/13 (7.7%)
- `fast_lewam` / `P2/legacy/guided_flow/step_32/euler`：button=1/11 (9.1%), cube=3/20 (15.0%), drawer=2/6 (33.3%), window=2/13 (15.4%)
- `fast_lewam` / `P0/not_applicable/post_opt/step_32/euler`：button=1/11 (9.1%), cube=3/20 (15.0%), drawer=1/6 (16.7%), window=1/13 (7.7%)
- `fast_lewam` / `P2/legacy/post_opt/step_32/euler`：button=1/11 (9.1%), cube=3/20 (15.0%), drawer=2/6 (33.3%), window=2/13 (15.4%)
- `fast_lewam` / `P3/not_applicable/guided_flow/step_32/euler`：button=0/11 (0.0%), cube=3/20 (15.0%), drawer=0/6 (0.0%), window=2/13 (15.4%)
- `fast_lewam` / `P3/not_applicable/post_opt/step_32/euler`：button=0/11 (0.0%), cube=3/20 (15.0%), drawer=0/6 (0.0%), window=2/13 (15.4%)
- `fast_lewam` / `P3/not_applicable/post_opt_refine/step_32/euler`：button=0/11 (0.0%), cube=3/20 (15.0%), drawer=0/6 (0.0%), window=2/13 (15.4%)
- `lewm` / `standard`：button=1/11 (9.1%), cube=3/20 (15.0%), drawer=1/6 (16.7%), window=4/13 (30.8%)
- `leflow` / `standard`：button=1/11 (9.1%), cube=2/20 (10.0%), drawer=0/6 (0.0%), window=2/13 (15.4%)

## 产物

- 输出根目录：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3_initial_fail`
- 条件明细：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3_initial_fail/analysis/conditions.csv`
- 分析 JSON：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3_initial_fail/analysis/analysis.json`
- 代表视频：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3_initial_fail/videos`（每任务 FastLeWAM P1/P3 step 1、LeWM、LeFlow 各一段）
- 本报告：`/data/users/wenxin/pre-exp/le-wm/docs/report/round5/round5_phase3_initial_failure_report.md`
