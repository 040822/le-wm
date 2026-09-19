# Round 5 Phase 1：R4-AB guidance 与 P3 结合实验

## 总表（step1 / step2 成功率 %）

每格依次为 **step1 / step2** 的成功率（百分数）；`P1 (inv)` 不使用 action flow，对 step 不变，只列一个值。`P2` 在 reacher 使用 `cem-clip`，其余任务 `legacy`。
GF = guided-flow，PO = post-opt，refine = 先 B 选优再 post-opt 精修。

| 任务 | P0 | P1 (inv) | P2 | P3 | P0+GF | P0+PO | P2+GF | P2+PO | P3+GF | P3+PO | P3+refine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cube | 100.0 / 100.0 | 70.0 (inv) | 98.0 / 96.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 98.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |
| pusht | 96.0 / 100.0 | 90.0 (inv) | 98.0 / 96.0 | 96.0 / 98.0 | 98.0 / 100.0 | 98.0 / 98.0 | 98.0 / 98.0 | 98.0 / 96.0 | 98.0 / 98.0 | 98.0 / 96.0 | 100.0 / 100.0 |
| reacher | 72.0 / 76.0 | 92.0 (inv) | 88.0 / 86.0 | 74.0 / 90.0 | 88.0 / 90.0 | 88.0 / 88.0 | 96.0 / 90.0 | 96.0 / 86.0 | 92.0 / 92.0 | 92.0 / 94.0 | 90.0 / 82.0 |
| tworoom | 100.0 / 96.0 | 100.0 (inv) | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 98.0 | 100.0 / 98.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |

### step=1 总表（成功率 %）

| 任务 | P0 | P1 | P2 | P3 | P0+GF | P0+PO | P2+GF | P2+PO | P3+GF | P3+PO | P3+refine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cube | 100.0 | 70.0 | 98.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| pusht | 96.0 | 90.0 | 98.0 | 96.0 | 98.0 | 98.0 | 98.0 | 98.0 | 98.0 | 98.0 | 100.0 |
| reacher | 72.0 | 92.0 | 88.0 | 74.0 | 88.0 | 88.0 | 96.0 | 96.0 | 92.0 | 92.0 | 90.0 |
| tworoom | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

## 实验范围

本轮在 R4-AB seed 3072 epoch 10 的四个任务上，使用 canonical `legacy_50` 旧协议 cohort（LeWM 上游口径，50 episodes）。P0/P1/P2/P3 的无 guidance 结果直接引用 Phase 4.5-4；本轮新增 guided-flow 与 post-opt 条件。flow step 网格为 1/2/5/10/16/32，主要落点为 step 1/2。

- 配置：`/data/users/wenxin/pre-exp/le-wm/config/round5/phase1.json`
- 输出根目录：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase1_seed3072_legacy`
- baseline 来源：`/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev`（只读引用）
- 分析 JSON：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase1_seed3072_legacy/analysis/analysis.json`

## 验收

- 条件总数：244 / 244；baseline=76 / 76，new=168 / 168。
- `status=ok`：True；每条件 50 episodes。

## 条件定义

guidance 条件：`late_steps=5, inner_steps=5, step_size=0.01, max_rms_offset=0.2`；`flow_steps` 等于该条件的 action flow step。
P3 语义：`guided_flow`/`post_opt` 对每条候选生效；`post_opt_refine` 先生成普通候选并由 B 选优，再对选中动作做 post-opt。
P2 中 reacher 使用 cem-clip，其余任务 legacy；P0/P3 不使用 clip。

## cube

### 基线（无 guidance，复用 Phase 4.5-4）

| 模式 | protocol | step | success |
|---|---|---:|---:|
| P0 | not_applicable | 1 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P0 | not_applicable | 2 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P0 | not_applicable | 5 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P0 | not_applicable | 10 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P0 | not_applicable | 16 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P0 | not_applicable | 32 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P1 | legacy | inv | 35/50 (70.0%)<br>CI [56.2, 80.9]% |
| P2 | legacy | 1 | 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| P2 | legacy | 2 | 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| P2 | legacy | 5 | 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| P2 | legacy | 10 | 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| P2 | legacy | 16 | 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| P2 | legacy | 32 | 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| P3 | not_applicable | 1 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P3 | not_applicable | 2 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P3 | not_applicable | 5 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P3 | not_applicable | 10 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P3 | not_applicable | 16 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P3 | not_applicable | 32 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |

### 新增 guidance 条件成功率

| 模式 | protocol | guidance | step1 | step2 | step5 | step10 | step16 | step32 |
|---|---|---|---|---|---|---|---|---|
| P0 | not_applicable | guided_flow | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P0 | not_applicable | post_opt | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P2 | legacy | guided_flow | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 46/50 (92.0%)<br>CI [81.2, 96.8]% | 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| P2 | legacy | post_opt | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| P3 | not_applicable | guided_flow | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P3 | not_applicable | post_opt | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P3 | not_applicable | post_opt_refine | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% |

### guidance 相对无 guidance（paired）

| mode | guidance | step | none | guidance | Δ pp | improved | regressed | McNemar p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | guided_flow | 1 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P0 | guided_flow | 2 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P0 | guided_flow | 5 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P0 | guided_flow | 10 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P0 | guided_flow | 16 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P0 | guided_flow | 32 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P0 | post_opt | 1 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P0 | post_opt | 2 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P0 | post_opt | 5 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P0 | post_opt | 10 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P0 | post_opt | 16 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P0 | post_opt | 32 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P2 | guided_flow | 1 | 98.0% | 100.0% | 2.0 | 1 | 0 | 1.000000 |
| P2 | guided_flow | 2 | 96.0% | 98.0% | 2.0 | 2 | 1 | 1.000000 |
| P2 | guided_flow | 5 | 98.0% | 98.0% | 0.0 | 0 | 0 | — |
| P2 | guided_flow | 10 | 96.0% | 98.0% | 2.0 | 1 | 0 | 1.000000 |
| P2 | guided_flow | 16 | 98.0% | 92.0% | -6.0 | 0 | 3 | 0.250000 |
| P2 | guided_flow | 32 | 98.0% | 96.0% | -2.0 | 0 | 1 | 1.000000 |
| P2 | post_opt | 1 | 98.0% | 100.0% | 2.0 | 1 | 0 | 1.000000 |
| P2 | post_opt | 2 | 96.0% | 100.0% | 4.0 | 2 | 0 | 0.500000 |
| P2 | post_opt | 5 | 98.0% | 98.0% | 0.0 | 0 | 0 | — |
| P2 | post_opt | 10 | 96.0% | 98.0% | 2.0 | 1 | 0 | 1.000000 |
| P2 | post_opt | 16 | 98.0% | 98.0% | 0.0 | 0 | 0 | — |
| P2 | post_opt | 32 | 98.0% | 98.0% | 0.0 | 0 | 0 | — |
| P3 | guided_flow | 1 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | guided_flow | 2 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | guided_flow | 5 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | guided_flow | 10 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | guided_flow | 16 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | guided_flow | 32 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt | 1 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt | 2 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt | 5 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt | 10 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt | 16 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt | 32 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt_refine | 1 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt_refine | 2 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt_refine | 5 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt_refine | 10 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt_refine | 16 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt_refine | 32 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |

### P3 guidance 语义对照（paired）

| step | baseline | treatment | Δ pp | improved | regressed | McNemar p |
|---:|---|---|---:|---:|---:|---:|
| 1 | post_opt | post_opt_refine | 0.0 | 0 | 0 | — |
| 1 | guided_flow | post_opt_refine | 0.0 | 0 | 0 | — |
| 1 | guided_flow | post_opt | 0.0 | 0 | 0 | — |
| 2 | post_opt | post_opt_refine | 0.0 | 0 | 0 | — |
| 2 | guided_flow | post_opt_refine | 0.0 | 0 | 0 | — |
| 2 | guided_flow | post_opt | 0.0 | 0 | 0 | — |
| 5 | post_opt | post_opt_refine | 0.0 | 0 | 0 | — |
| 5 | guided_flow | post_opt_refine | 0.0 | 0 | 0 | — |
| 5 | guided_flow | post_opt | 0.0 | 0 | 0 | — |
| 10 | post_opt | post_opt_refine | 0.0 | 0 | 0 | — |
| 10 | guided_flow | post_opt_refine | 0.0 | 0 | 0 | — |
| 10 | guided_flow | post_opt | 0.0 | 0 | 0 | — |
| 16 | post_opt | post_opt_refine | 0.0 | 0 | 0 | — |
| 16 | guided_flow | post_opt_refine | 0.0 | 0 | 0 | — |
| 16 | guided_flow | post_opt | 0.0 | 0 | 0 | — |
| 32 | post_opt | post_opt_refine | 0.0 | 0 | 0 | — |
| 32 | guided_flow | post_opt_refine | 0.0 | 0 | 0 | — |
| 32 | guided_flow | post_opt | 0.0 | 0 | 0 | — |

### 学习 planner 相对 P1（随机初始化 CEM，paired）

| mode | step | P1 | planner | Δ pp | improved | regressed | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 1 | 70.0% | 100.0% | 30.0 | 15 | 0 | 0.000061 |
| P0 | 2 | 70.0% | 100.0% | 30.0 | 15 | 0 | 0.000061 |
| P0 | 5 | 70.0% | 100.0% | 30.0 | 15 | 0 | 0.000061 |
| P0 | 10 | 70.0% | 100.0% | 30.0 | 15 | 0 | 0.000061 |
| P0 | 16 | 70.0% | 100.0% | 30.0 | 15 | 0 | 0.000061 |
| P0 | 32 | 70.0% | 100.0% | 30.0 | 15 | 0 | 0.000061 |
| P2 | 1 | 70.0% | 98.0% | 28.0 | 14 | 0 | 0.000122 |
| P2 | 2 | 70.0% | 96.0% | 26.0 | 13 | 0 | 0.000244 |
| P2 | 5 | 70.0% | 98.0% | 28.0 | 14 | 0 | 0.000122 |
| P2 | 10 | 70.0% | 96.0% | 26.0 | 13 | 0 | 0.000244 |
| P2 | 16 | 70.0% | 98.0% | 28.0 | 14 | 0 | 0.000122 |
| P2 | 32 | 70.0% | 98.0% | 28.0 | 14 | 0 | 0.000122 |
| P3 | 1 | 70.0% | 100.0% | 30.0 | 15 | 0 | 0.000061 |
| P3 | 2 | 70.0% | 100.0% | 30.0 | 15 | 0 | 0.000061 |
| P3 | 5 | 70.0% | 100.0% | 30.0 | 15 | 0 | 0.000061 |
| P3 | 10 | 70.0% | 100.0% | 30.0 | 15 | 0 | 0.000061 |
| P3 | 16 | 70.0% | 100.0% | 30.0 | 15 | 0 | 0.000061 |
| P3 | 32 | 70.0% | 100.0% | 30.0 | 15 | 0 | 0.000061 |

## pusht

### 基线（无 guidance，复用 Phase 4.5-4）

| 模式 | protocol | step | success |
|---|---|---:|---:|
| P0 | not_applicable | 1 | 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| P0 | not_applicable | 2 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P0 | not_applicable | 5 | 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| P0 | not_applicable | 10 | 47/50 (94.0%)<br>CI [83.8, 97.9]% |
| P0 | not_applicable | 16 | 47/50 (94.0%)<br>CI [83.8, 97.9]% |
| P0 | not_applicable | 32 | 47/50 (94.0%)<br>CI [83.8, 97.9]% |
| P1 | legacy | inv | 45/50 (90.0%)<br>CI [78.6, 95.7]% |
| P2 | legacy | 1 | 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| P2 | legacy | 2 | 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| P2 | legacy | 5 | 47/50 (94.0%)<br>CI [83.8, 97.9]% |
| P2 | legacy | 10 | 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| P2 | legacy | 16 | 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| P2 | legacy | 32 | 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| P3 | not_applicable | 1 | 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| P3 | not_applicable | 2 | 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| P3 | not_applicable | 5 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P3 | not_applicable | 10 | 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| P3 | not_applicable | 16 | 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| P3 | not_applicable | 32 | 49/50 (98.0%)<br>CI [89.5, 99.6]% |

### 新增 guidance 条件成功率

| 模式 | protocol | guidance | step1 | step2 | step5 | step10 | step16 | step32 |
|---|---|---|---|---|---|---|---|---|
| P0 | not_applicable | guided_flow | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 48/50 (96.0%)<br>CI [86.5, 98.9]% | 48/50 (96.0%)<br>CI [86.5, 98.9]% | 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| P0 | not_applicable | post_opt | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 48/50 (96.0%)<br>CI [86.5, 98.9]% | 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| P2 | legacy | guided_flow | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 48/50 (96.0%)<br>CI [86.5, 98.9]% | 47/50 (94.0%)<br>CI [83.8, 97.9]% | 47/50 (94.0%)<br>CI [83.8, 97.9]% |
| P2 | legacy | post_opt | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 48/50 (96.0%)<br>CI [86.5, 98.9]% | 48/50 (96.0%)<br>CI [86.5, 98.9]% | 47/50 (94.0%)<br>CI [83.8, 97.9]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 47/50 (94.0%)<br>CI [83.8, 97.9]% |
| P3 | not_applicable | guided_flow | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 48/50 (96.0%)<br>CI [86.5, 98.9]% | 48/50 (96.0%)<br>CI [86.5, 98.9]% | 48/50 (96.0%)<br>CI [86.5, 98.9]% | 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| P3 | not_applicable | post_opt | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 48/50 (96.0%)<br>CI [86.5, 98.9]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 48/50 (96.0%)<br>CI [86.5, 98.9]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| P3 | not_applicable | post_opt_refine | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 48/50 (96.0%)<br>CI [86.5, 98.9]% | 48/50 (96.0%)<br>CI [86.5, 98.9]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% |

### guidance 相对无 guidance（paired）

| mode | guidance | step | none | guidance | Δ pp | improved | regressed | McNemar p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | guided_flow | 1 | 96.0% | 98.0% | 2.0 | 1 | 0 | 1.000000 |
| P0 | guided_flow | 2 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P0 | guided_flow | 5 | 98.0% | 98.0% | 0.0 | 1 | 1 | 1.000000 |
| P0 | guided_flow | 10 | 94.0% | 96.0% | 2.0 | 3 | 2 | 1.000000 |
| P0 | guided_flow | 16 | 94.0% | 96.0% | 2.0 | 3 | 2 | 1.000000 |
| P0 | guided_flow | 32 | 94.0% | 96.0% | 2.0 | 3 | 2 | 1.000000 |
| P0 | post_opt | 1 | 96.0% | 98.0% | 2.0 | 1 | 0 | 1.000000 |
| P0 | post_opt | 2 | 100.0% | 98.0% | -2.0 | 0 | 1 | 1.000000 |
| P0 | post_opt | 5 | 98.0% | 98.0% | 0.0 | 0 | 0 | — |
| P0 | post_opt | 10 | 94.0% | 98.0% | 4.0 | 2 | 0 | 0.500000 |
| P0 | post_opt | 16 | 94.0% | 96.0% | 2.0 | 2 | 1 | 1.000000 |
| P0 | post_opt | 32 | 94.0% | 96.0% | 2.0 | 2 | 1 | 1.000000 |
| P2 | guided_flow | 1 | 98.0% | 98.0% | 0.0 | 1 | 1 | 1.000000 |
| P2 | guided_flow | 2 | 96.0% | 98.0% | 2.0 | 1 | 0 | 1.000000 |
| P2 | guided_flow | 5 | 94.0% | 98.0% | 4.0 | 3 | 1 | 0.625000 |
| P2 | guided_flow | 10 | 96.0% | 96.0% | 0.0 | 1 | 1 | 1.000000 |
| P2 | guided_flow | 16 | 96.0% | 94.0% | -2.0 | 0 | 1 | 1.000000 |
| P2 | guided_flow | 32 | 96.0% | 94.0% | -2.0 | 1 | 2 | 1.000000 |
| P2 | post_opt | 1 | 98.0% | 98.0% | 0.0 | 1 | 1 | 1.000000 |
| P2 | post_opt | 2 | 96.0% | 96.0% | 0.0 | 1 | 1 | 1.000000 |
| P2 | post_opt | 5 | 94.0% | 96.0% | 2.0 | 1 | 0 | 1.000000 |
| P2 | post_opt | 10 | 96.0% | 94.0% | -2.0 | 0 | 1 | 1.000000 |
| P2 | post_opt | 16 | 96.0% | 98.0% | 2.0 | 1 | 0 | 1.000000 |
| P2 | post_opt | 32 | 96.0% | 94.0% | -2.0 | 0 | 1 | 1.000000 |
| P3 | guided_flow | 1 | 96.0% | 98.0% | 2.0 | 2 | 1 | 1.000000 |
| P3 | guided_flow | 2 | 98.0% | 98.0% | 0.0 | 1 | 1 | 1.000000 |
| P3 | guided_flow | 5 | 100.0% | 96.0% | -4.0 | 0 | 2 | 0.500000 |
| P3 | guided_flow | 10 | 98.0% | 96.0% | -2.0 | 1 | 2 | 1.000000 |
| P3 | guided_flow | 16 | 96.0% | 96.0% | 0.0 | 1 | 1 | 1.000000 |
| P3 | guided_flow | 32 | 98.0% | 96.0% | -2.0 | 1 | 2 | 1.000000 |
| P3 | post_opt | 1 | 96.0% | 98.0% | 2.0 | 2 | 1 | 1.000000 |
| P3 | post_opt | 2 | 98.0% | 96.0% | -2.0 | 0 | 1 | 1.000000 |
| P3 | post_opt | 5 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt | 10 | 98.0% | 96.0% | -2.0 | 1 | 2 | 1.000000 |
| P3 | post_opt | 16 | 96.0% | 98.0% | 2.0 | 1 | 0 | 1.000000 |
| P3 | post_opt | 32 | 98.0% | 98.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt_refine | 1 | 96.0% | 100.0% | 4.0 | 2 | 0 | 0.500000 |
| P3 | post_opt_refine | 2 | 98.0% | 100.0% | 2.0 | 1 | 0 | 1.000000 |
| P3 | post_opt_refine | 5 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt_refine | 10 | 98.0% | 96.0% | -2.0 | 0 | 1 | 1.000000 |
| P3 | post_opt_refine | 16 | 96.0% | 96.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt_refine | 32 | 98.0% | 98.0% | 0.0 | 0 | 0 | — |

### P3 guidance 语义对照（paired）

| step | baseline | treatment | Δ pp | improved | regressed | McNemar p |
|---:|---|---|---:|---:|---:|---:|
| 1 | post_opt | post_opt_refine | 2.0 | 1 | 0 | 1.000000 |
| 1 | guided_flow | post_opt_refine | 2.0 | 1 | 0 | 1.000000 |
| 1 | guided_flow | post_opt | 0.0 | 0 | 0 | — |
| 2 | post_opt | post_opt_refine | 4.0 | 2 | 0 | 0.500000 |
| 2 | guided_flow | post_opt_refine | 2.0 | 1 | 0 | 1.000000 |
| 2 | guided_flow | post_opt | -2.0 | 0 | 1 | 1.000000 |
| 5 | post_opt | post_opt_refine | 0.0 | 0 | 0 | — |
| 5 | guided_flow | post_opt_refine | 4.0 | 2 | 0 | 0.500000 |
| 5 | guided_flow | post_opt | 4.0 | 2 | 0 | 0.500000 |
| 10 | post_opt | post_opt_refine | 0.0 | 1 | 1 | 1.000000 |
| 10 | guided_flow | post_opt_refine | 0.0 | 2 | 2 | 1.000000 |
| 10 | guided_flow | post_opt | 0.0 | 1 | 1 | 1.000000 |
| 16 | post_opt | post_opt_refine | -2.0 | 0 | 1 | 1.000000 |
| 16 | guided_flow | post_opt_refine | 0.0 | 1 | 1 | 1.000000 |
| 16 | guided_flow | post_opt | 2.0 | 2 | 1 | 1.000000 |
| 32 | post_opt | post_opt_refine | 0.0 | 0 | 0 | — |
| 32 | guided_flow | post_opt_refine | 2.0 | 2 | 1 | 1.000000 |
| 32 | guided_flow | post_opt | 2.0 | 2 | 1 | 1.000000 |

### 学习 planner 相对 P1（随机初始化 CEM，paired）

| mode | step | P1 | planner | Δ pp | improved | regressed | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 1 | 90.0% | 96.0% | 6.0 | 5 | 2 | 0.453125 |
| P0 | 2 | 90.0% | 100.0% | 10.0 | 5 | 0 | 0.062500 |
| P0 | 5 | 90.0% | 98.0% | 8.0 | 4 | 0 | 0.125000 |
| P0 | 10 | 90.0% | 94.0% | 4.0 | 3 | 1 | 0.625000 |
| P0 | 16 | 90.0% | 94.0% | 4.0 | 3 | 1 | 0.625000 |
| P0 | 32 | 90.0% | 94.0% | 4.0 | 3 | 1 | 0.625000 |
| P2 | 1 | 90.0% | 98.0% | 8.0 | 4 | 0 | 0.125000 |
| P2 | 2 | 90.0% | 96.0% | 6.0 | 3 | 0 | 0.250000 |
| P2 | 5 | 90.0% | 94.0% | 4.0 | 2 | 0 | 0.500000 |
| P2 | 10 | 90.0% | 96.0% | 6.0 | 3 | 0 | 0.250000 |
| P2 | 16 | 90.0% | 96.0% | 6.0 | 3 | 0 | 0.250000 |
| P2 | 32 | 90.0% | 96.0% | 6.0 | 3 | 0 | 0.250000 |
| P3 | 1 | 90.0% | 96.0% | 6.0 | 4 | 1 | 0.375000 |
| P3 | 2 | 90.0% | 98.0% | 8.0 | 4 | 0 | 0.125000 |
| P3 | 5 | 90.0% | 100.0% | 10.0 | 5 | 0 | 0.062500 |
| P3 | 10 | 90.0% | 98.0% | 8.0 | 5 | 1 | 0.218750 |
| P3 | 16 | 90.0% | 96.0% | 6.0 | 4 | 1 | 0.375000 |
| P3 | 32 | 90.0% | 98.0% | 8.0 | 5 | 1 | 0.218750 |

## reacher

### 基线（无 guidance，复用 Phase 4.5-4）

| 模式 | protocol | step | success |
|---|---|---:|---:|
| P0 | not_applicable | 1 | 36/50 (72.0%)<br>CI [58.3, 82.5]% |
| P0 | not_applicable | 2 | 38/50 (76.0%)<br>CI [62.6, 85.7]% |
| P0 | not_applicable | 5 | 37/50 (74.0%)<br>CI [60.4, 84.1]% |
| P0 | not_applicable | 10 | 39/50 (78.0%)<br>CI [64.8, 87.2]% |
| P0 | not_applicable | 16 | 37/50 (74.0%)<br>CI [60.4, 84.1]% |
| P0 | not_applicable | 32 | 39/50 (78.0%)<br>CI [64.8, 87.2]% |
| P1 | cem-clip | inv | 46/50 (92.0%)<br>CI [81.2, 96.8]% |
| P2 | cem-clip | 1 | 44/50 (88.0%)<br>CI [76.2, 94.4]% |
| P2 | cem-clip | 2 | 43/50 (86.0%)<br>CI [73.8, 93.0]% |
| P2 | cem-clip | 5 | 45/50 (90.0%)<br>CI [78.6, 95.7]% |
| P2 | cem-clip | 10 | 44/50 (88.0%)<br>CI [76.2, 94.4]% |
| P2 | cem-clip | 16 | 42/50 (84.0%)<br>CI [71.5, 91.7]% |
| P2 | cem-clip | 32 | 40/50 (80.0%)<br>CI [67.0, 88.8]% |
| P3 | not_applicable | 1 | 37/50 (74.0%)<br>CI [60.4, 84.1]% |
| P3 | not_applicable | 2 | 45/50 (90.0%)<br>CI [78.6, 95.7]% |
| P3 | not_applicable | 5 | 42/50 (84.0%)<br>CI [71.5, 91.7]% |
| P3 | not_applicable | 10 | 43/50 (86.0%)<br>CI [73.8, 93.0]% |
| P3 | not_applicable | 16 | 39/50 (78.0%)<br>CI [64.8, 87.2]% |
| P3 | not_applicable | 32 | 42/50 (84.0%)<br>CI [71.5, 91.7]% |

### 新增 guidance 条件成功率

| 模式 | protocol | guidance | step1 | step2 | step5 | step10 | step16 | step32 |
|---|---|---|---|---|---|---|---|---|
| P0 | not_applicable | guided_flow | 44/50 (88.0%)<br>CI [76.2, 94.4]% | 45/50 (90.0%)<br>CI [78.6, 95.7]% | 39/50 (78.0%)<br>CI [64.8, 87.2]% | 41/50 (82.0%)<br>CI [69.2, 90.2]% | 44/50 (88.0%)<br>CI [76.2, 94.4]% | 42/50 (84.0%)<br>CI [71.5, 91.7]% |
| P0 | not_applicable | post_opt | 44/50 (88.0%)<br>CI [76.2, 94.4]% | 44/50 (88.0%)<br>CI [76.2, 94.4]% | 43/50 (86.0%)<br>CI [73.8, 93.0]% | 40/50 (80.0%)<br>CI [67.0, 88.8]% | 37/50 (74.0%)<br>CI [60.4, 84.1]% | 39/50 (78.0%)<br>CI [64.8, 87.2]% |
| P2 | cem-clip | guided_flow | 48/50 (96.0%)<br>CI [86.5, 98.9]% | 45/50 (90.0%)<br>CI [78.6, 95.7]% | 46/50 (92.0%)<br>CI [81.2, 96.8]% | 41/50 (82.0%)<br>CI [69.2, 90.2]% | 41/50 (82.0%)<br>CI [69.2, 90.2]% | 41/50 (82.0%)<br>CI [69.2, 90.2]% |
| P2 | cem-clip | post_opt | 48/50 (96.0%)<br>CI [86.5, 98.9]% | 43/50 (86.0%)<br>CI [73.8, 93.0]% | 45/50 (90.0%)<br>CI [78.6, 95.7]% | 42/50 (84.0%)<br>CI [71.5, 91.7]% | 46/50 (92.0%)<br>CI [81.2, 96.8]% | 40/50 (80.0%)<br>CI [67.0, 88.8]% |
| P3 | not_applicable | guided_flow | 46/50 (92.0%)<br>CI [81.2, 96.8]% | 46/50 (92.0%)<br>CI [81.2, 96.8]% | 40/50 (80.0%)<br>CI [67.0, 88.8]% | 44/50 (88.0%)<br>CI [76.2, 94.4]% | 39/50 (78.0%)<br>CI [64.8, 87.2]% | 41/50 (82.0%)<br>CI [69.2, 90.2]% |
| P3 | not_applicable | post_opt | 46/50 (92.0%)<br>CI [81.2, 96.8]% | 47/50 (94.0%)<br>CI [83.8, 97.9]% | 42/50 (84.0%)<br>CI [71.5, 91.7]% | 41/50 (82.0%)<br>CI [69.2, 90.2]% | 42/50 (84.0%)<br>CI [71.5, 91.7]% | 42/50 (84.0%)<br>CI [71.5, 91.7]% |
| P3 | not_applicable | post_opt_refine | 45/50 (90.0%)<br>CI [78.6, 95.7]% | 41/50 (82.0%)<br>CI [69.2, 90.2]% | 45/50 (90.0%)<br>CI [78.6, 95.7]% | 41/50 (82.0%)<br>CI [69.2, 90.2]% | 41/50 (82.0%)<br>CI [69.2, 90.2]% | 44/50 (88.0%)<br>CI [76.2, 94.4]% |

### guidance 相对无 guidance（paired）

| mode | guidance | step | none | guidance | Δ pp | improved | regressed | McNemar p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | guided_flow | 1 | 72.0% | 88.0% | 16.0 | 9 | 1 | 0.021484 |
| P0 | guided_flow | 2 | 76.0% | 90.0% | 14.0 | 10 | 3 | 0.092285 |
| P0 | guided_flow | 5 | 74.0% | 78.0% | 4.0 | 8 | 6 | 0.790527 |
| P0 | guided_flow | 10 | 78.0% | 82.0% | 4.0 | 6 | 4 | 0.753906 |
| P0 | guided_flow | 16 | 74.0% | 88.0% | 14.0 | 9 | 2 | 0.065430 |
| P0 | guided_flow | 32 | 78.0% | 84.0% | 6.0 | 9 | 6 | 0.607239 |
| P0 | post_opt | 1 | 72.0% | 88.0% | 16.0 | 9 | 1 | 0.021484 |
| P0 | post_opt | 2 | 76.0% | 88.0% | 12.0 | 8 | 2 | 0.109375 |
| P0 | post_opt | 5 | 74.0% | 86.0% | 12.0 | 8 | 2 | 0.109375 |
| P0 | post_opt | 10 | 78.0% | 80.0% | 2.0 | 4 | 3 | 1.000000 |
| P0 | post_opt | 16 | 74.0% | 74.0% | 0.0 | 7 | 7 | 1.000000 |
| P0 | post_opt | 32 | 78.0% | 78.0% | 0.0 | 9 | 9 | 1.000000 |
| P2 | guided_flow | 1 | 88.0% | 96.0% | 8.0 | 4 | 0 | 0.125000 |
| P2 | guided_flow | 2 | 86.0% | 90.0% | 4.0 | 4 | 2 | 0.687500 |
| P2 | guided_flow | 5 | 90.0% | 92.0% | 2.0 | 2 | 1 | 1.000000 |
| P2 | guided_flow | 10 | 88.0% | 82.0% | -6.0 | 2 | 5 | 0.453125 |
| P2 | guided_flow | 16 | 84.0% | 82.0% | -2.0 | 3 | 4 | 1.000000 |
| P2 | guided_flow | 32 | 80.0% | 82.0% | 2.0 | 5 | 4 | 1.000000 |
| P2 | post_opt | 1 | 88.0% | 96.0% | 8.0 | 4 | 0 | 0.125000 |
| P2 | post_opt | 2 | 86.0% | 86.0% | 0.0 | 3 | 3 | 1.000000 |
| P2 | post_opt | 5 | 90.0% | 90.0% | 0.0 | 3 | 3 | 1.000000 |
| P2 | post_opt | 10 | 88.0% | 84.0% | -4.0 | 1 | 3 | 0.625000 |
| P2 | post_opt | 16 | 84.0% | 92.0% | 8.0 | 4 | 0 | 0.125000 |
| P2 | post_opt | 32 | 80.0% | 80.0% | 0.0 | 4 | 4 | 1.000000 |
| P3 | guided_flow | 1 | 74.0% | 92.0% | 18.0 | 9 | 0 | 0.003906 |
| P3 | guided_flow | 2 | 90.0% | 92.0% | 2.0 | 5 | 4 | 1.000000 |
| P3 | guided_flow | 5 | 84.0% | 80.0% | -4.0 | 5 | 7 | 0.774414 |
| P3 | guided_flow | 10 | 86.0% | 88.0% | 2.0 | 5 | 4 | 1.000000 |
| P3 | guided_flow | 16 | 78.0% | 78.0% | 0.0 | 8 | 8 | 1.000000 |
| P3 | guided_flow | 32 | 84.0% | 82.0% | -2.0 | 8 | 9 | 1.000000 |
| P3 | post_opt | 1 | 74.0% | 92.0% | 18.0 | 9 | 0 | 0.003906 |
| P3 | post_opt | 2 | 90.0% | 94.0% | 4.0 | 5 | 3 | 0.726562 |
| P3 | post_opt | 5 | 84.0% | 84.0% | 0.0 | 5 | 5 | 1.000000 |
| P3 | post_opt | 10 | 86.0% | 82.0% | -4.0 | 6 | 8 | 0.790527 |
| P3 | post_opt | 16 | 78.0% | 84.0% | 6.0 | 8 | 5 | 0.581055 |
| P3 | post_opt | 32 | 84.0% | 84.0% | 0.0 | 7 | 7 | 1.000000 |
| P3 | post_opt_refine | 1 | 74.0% | 90.0% | 16.0 | 8 | 0 | 0.007812 |
| P3 | post_opt_refine | 2 | 90.0% | 82.0% | -8.0 | 1 | 5 | 0.218750 |
| P3 | post_opt_refine | 5 | 84.0% | 90.0% | 6.0 | 6 | 3 | 0.507812 |
| P3 | post_opt_refine | 10 | 86.0% | 82.0% | -4.0 | 3 | 5 | 0.726562 |
| P3 | post_opt_refine | 16 | 78.0% | 82.0% | 4.0 | 5 | 3 | 0.726562 |
| P3 | post_opt_refine | 32 | 84.0% | 88.0% | 4.0 | 7 | 5 | 0.774414 |

### P3 guidance 语义对照（paired）

| step | baseline | treatment | Δ pp | improved | regressed | McNemar p |
|---:|---|---|---:|---:|---:|---:|
| 1 | post_opt | post_opt_refine | -2.0 | 0 | 1 | 1.000000 |
| 1 | guided_flow | post_opt_refine | -2.0 | 0 | 1 | 1.000000 |
| 1 | guided_flow | post_opt | 0.0 | 0 | 0 | — |
| 2 | post_opt | post_opt_refine | -12.0 | 2 | 8 | 0.109375 |
| 2 | guided_flow | post_opt_refine | -10.0 | 2 | 7 | 0.179688 |
| 2 | guided_flow | post_opt | 2.0 | 3 | 2 | 1.000000 |
| 5 | post_opt | post_opt_refine | 6.0 | 8 | 5 | 0.581055 |
| 5 | guided_flow | post_opt_refine | 10.0 | 9 | 4 | 0.266846 |
| 5 | guided_flow | post_opt | 4.0 | 5 | 3 | 0.726562 |
| 10 | post_opt | post_opt_refine | 0.0 | 7 | 7 | 1.000000 |
| 10 | guided_flow | post_opt_refine | -6.0 | 3 | 6 | 0.507812 |
| 10 | guided_flow | post_opt | -6.0 | 3 | 6 | 0.507812 |
| 16 | post_opt | post_opt_refine | -2.0 | 5 | 6 | 1.000000 |
| 16 | guided_flow | post_opt_refine | 4.0 | 7 | 5 | 0.774414 |
| 16 | guided_flow | post_opt | 6.0 | 10 | 7 | 0.629059 |
| 32 | post_opt | post_opt_refine | 4.0 | 7 | 5 | 0.774414 |
| 32 | guided_flow | post_opt_refine | 6.0 | 8 | 5 | 0.581055 |
| 32 | guided_flow | post_opt | 2.0 | 7 | 6 | 1.000000 |

### 学习 planner 相对 P1（随机初始化 CEM，paired）

| mode | step | P1 | planner | Δ pp | improved | regressed | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 1 | 92.0% | 72.0% | -20.0 | 2 | 12 | 0.012939 |
| P0 | 2 | 92.0% | 76.0% | -16.0 | 3 | 11 | 0.057373 |
| P0 | 5 | 92.0% | 74.0% | -18.0 | 3 | 12 | 0.035156 |
| P0 | 10 | 92.0% | 78.0% | -14.0 | 3 | 10 | 0.092285 |
| P0 | 16 | 92.0% | 74.0% | -18.0 | 3 | 12 | 0.035156 |
| P0 | 32 | 92.0% | 78.0% | -14.0 | 3 | 10 | 0.092285 |
| P2 | 1 | 92.0% | 88.0% | -4.0 | 3 | 5 | 0.726562 |
| P2 | 2 | 92.0% | 86.0% | -6.0 | 3 | 6 | 0.507812 |
| P2 | 5 | 92.0% | 90.0% | -2.0 | 4 | 5 | 1.000000 |
| P2 | 10 | 92.0% | 88.0% | -4.0 | 4 | 6 | 0.753906 |
| P2 | 16 | 92.0% | 84.0% | -8.0 | 4 | 8 | 0.387695 |
| P2 | 32 | 92.0% | 80.0% | -12.0 | 3 | 9 | 0.145996 |
| P3 | 1 | 92.0% | 74.0% | -18.0 | 2 | 11 | 0.022461 |
| P3 | 2 | 92.0% | 90.0% | -2.0 | 3 | 4 | 1.000000 |
| P3 | 5 | 92.0% | 84.0% | -8.0 | 3 | 7 | 0.343750 |
| P3 | 10 | 92.0% | 86.0% | -6.0 | 3 | 6 | 0.507812 |
| P3 | 16 | 92.0% | 78.0% | -14.0 | 2 | 9 | 0.065430 |
| P3 | 32 | 92.0% | 84.0% | -8.0 | 2 | 6 | 0.289062 |

## tworoom

### 基线（无 guidance，复用 Phase 4.5-4）

| 模式 | protocol | step | success |
|---|---|---:|---:|
| P0 | not_applicable | 1 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P0 | not_applicable | 2 | 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| P0 | not_applicable | 5 | 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| P0 | not_applicable | 10 | 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| P0 | not_applicable | 16 | 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| P0 | not_applicable | 32 | 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| P1 | legacy | inv | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P2 | legacy | 1 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P2 | legacy | 2 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P2 | legacy | 5 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P2 | legacy | 10 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P2 | legacy | 16 | 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| P2 | legacy | 32 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P3 | not_applicable | 1 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P3 | not_applicable | 2 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P3 | not_applicable | 5 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P3 | not_applicable | 10 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P3 | not_applicable | 16 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P3 | not_applicable | 32 | 50/50 (100.0%)<br>CI [92.9, 100.0]% |

### 新增 guidance 条件成功率

| 模式 | protocol | guidance | step1 | step2 | step5 | step10 | step16 | step32 |
|---|---|---|---|---|---|---|---|---|
| P0 | not_applicable | guided_flow | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| P0 | not_applicable | post_opt | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| P2 | legacy | guided_flow | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P2 | legacy | post_opt | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| P3 | not_applicable | guided_flow | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| P3 | not_applicable | post_opt | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| P3 | not_applicable | post_opt_refine | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% | 50/50 (100.0%)<br>CI [92.9, 100.0]% |

### guidance 相对无 guidance（paired）

| mode | guidance | step | none | guidance | Δ pp | improved | regressed | McNemar p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | guided_flow | 1 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P0 | guided_flow | 2 | 96.0% | 98.0% | 2.0 | 1 | 0 | 1.000000 |
| P0 | guided_flow | 5 | 98.0% | 100.0% | 2.0 | 1 | 0 | 1.000000 |
| P0 | guided_flow | 10 | 98.0% | 98.0% | 0.0 | 0 | 0 | — |
| P0 | guided_flow | 16 | 98.0% | 98.0% | 0.0 | 0 | 0 | — |
| P0 | guided_flow | 32 | 96.0% | 98.0% | 2.0 | 1 | 0 | 1.000000 |
| P0 | post_opt | 1 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P0 | post_opt | 2 | 96.0% | 98.0% | 2.0 | 1 | 0 | 1.000000 |
| P0 | post_opt | 5 | 98.0% | 98.0% | 0.0 | 0 | 0 | — |
| P0 | post_opt | 10 | 98.0% | 98.0% | 0.0 | 0 | 0 | — |
| P0 | post_opt | 16 | 98.0% | 98.0% | 0.0 | 0 | 0 | — |
| P0 | post_opt | 32 | 96.0% | 98.0% | 2.0 | 1 | 0 | 1.000000 |
| P2 | guided_flow | 1 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P2 | guided_flow | 2 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P2 | guided_flow | 5 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P2 | guided_flow | 10 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P2 | guided_flow | 16 | 98.0% | 100.0% | 2.0 | 1 | 0 | 1.000000 |
| P2 | guided_flow | 32 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P2 | post_opt | 1 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P2 | post_opt | 2 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P2 | post_opt | 5 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P2 | post_opt | 10 | 100.0% | 98.0% | -2.0 | 0 | 1 | 1.000000 |
| P2 | post_opt | 16 | 98.0% | 100.0% | 2.0 | 1 | 0 | 1.000000 |
| P2 | post_opt | 32 | 100.0% | 98.0% | -2.0 | 0 | 1 | 1.000000 |
| P3 | guided_flow | 1 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | guided_flow | 2 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | guided_flow | 5 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | guided_flow | 10 | 100.0% | 98.0% | -2.0 | 0 | 1 | 1.000000 |
| P3 | guided_flow | 16 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | guided_flow | 32 | 100.0% | 98.0% | -2.0 | 0 | 1 | 1.000000 |
| P3 | post_opt | 1 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt | 2 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt | 5 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt | 10 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt | 16 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt | 32 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt_refine | 1 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt_refine | 2 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt_refine | 5 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt_refine | 10 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt_refine | 16 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | post_opt_refine | 32 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |

### P3 guidance 语义对照（paired）

| step | baseline | treatment | Δ pp | improved | regressed | McNemar p |
|---:|---|---|---:|---:|---:|---:|
| 1 | post_opt | post_opt_refine | 0.0 | 0 | 0 | — |
| 1 | guided_flow | post_opt_refine | 0.0 | 0 | 0 | — |
| 1 | guided_flow | post_opt | 0.0 | 0 | 0 | — |
| 2 | post_opt | post_opt_refine | 0.0 | 0 | 0 | — |
| 2 | guided_flow | post_opt_refine | 0.0 | 0 | 0 | — |
| 2 | guided_flow | post_opt | 0.0 | 0 | 0 | — |
| 5 | post_opt | post_opt_refine | 0.0 | 0 | 0 | — |
| 5 | guided_flow | post_opt_refine | 0.0 | 0 | 0 | — |
| 5 | guided_flow | post_opt | 0.0 | 0 | 0 | — |
| 10 | post_opt | post_opt_refine | 0.0 | 0 | 0 | — |
| 10 | guided_flow | post_opt_refine | 2.0 | 1 | 0 | 1.000000 |
| 10 | guided_flow | post_opt | 2.0 | 1 | 0 | 1.000000 |
| 16 | post_opt | post_opt_refine | 0.0 | 0 | 0 | — |
| 16 | guided_flow | post_opt_refine | 0.0 | 0 | 0 | — |
| 16 | guided_flow | post_opt | 0.0 | 0 | 0 | — |
| 32 | post_opt | post_opt_refine | 0.0 | 0 | 0 | — |
| 32 | guided_flow | post_opt_refine | 2.0 | 1 | 0 | 1.000000 |
| 32 | guided_flow | post_opt | 2.0 | 1 | 0 | 1.000000 |

### 学习 planner 相对 P1（随机初始化 CEM，paired）

| mode | step | P1 | planner | Δ pp | improved | regressed | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 1 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P0 | 2 | 100.0% | 96.0% | -4.0 | 0 | 2 | 0.500000 |
| P0 | 5 | 100.0% | 98.0% | -2.0 | 0 | 1 | 1.000000 |
| P0 | 10 | 100.0% | 98.0% | -2.0 | 0 | 1 | 1.000000 |
| P0 | 16 | 100.0% | 98.0% | -2.0 | 0 | 1 | 1.000000 |
| P0 | 32 | 100.0% | 96.0% | -4.0 | 0 | 2 | 0.500000 |
| P2 | 1 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P2 | 2 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P2 | 5 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P2 | 10 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P2 | 16 | 100.0% | 98.0% | -2.0 | 0 | 1 | 1.000000 |
| P2 | 32 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | 1 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | 2 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | 5 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | 10 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | 16 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |
| P3 | 32 | 100.0% | 100.0% | 0.0 | 0 | 0 | — |

## step 相对 step16（paired，含 guidance）

| task | mode | protocol | guidance | step | Δ pp | improved | regressed |
|---|---|---|---|---:|---:|---:|---:|
| cube | P0 | not_applicable | guided_flow | 1 | 0.0 | 0 | 0 |
| cube | P0 | not_applicable | guided_flow | 2 | 0.0 | 0 | 0 |
| cube | P0 | not_applicable | guided_flow | 5 | 0.0 | 0 | 0 |
| cube | P0 | not_applicable | guided_flow | 10 | 0.0 | 0 | 0 |
| cube | P0 | not_applicable | guided_flow | 32 | 0.0 | 0 | 0 |
| cube | P0 | not_applicable | none | 1 | 0.0 | 0 | 0 |
| cube | P0 | not_applicable | none | 2 | 0.0 | 0 | 0 |
| cube | P0 | not_applicable | none | 5 | 0.0 | 0 | 0 |
| cube | P0 | not_applicable | none | 10 | 0.0 | 0 | 0 |
| cube | P0 | not_applicable | none | 32 | 0.0 | 0 | 0 |
| cube | P0 | not_applicable | post_opt | 1 | 0.0 | 0 | 0 |
| cube | P0 | not_applicable | post_opt | 2 | 0.0 | 0 | 0 |
| cube | P0 | not_applicable | post_opt | 5 | 0.0 | 0 | 0 |
| cube | P0 | not_applicable | post_opt | 10 | 0.0 | 0 | 0 |
| cube | P0 | not_applicable | post_opt | 32 | 0.0 | 0 | 0 |
| cube | P2 | legacy | guided_flow | 1 | 8.0 | 4 | 0 |
| cube | P2 | legacy | guided_flow | 2 | 6.0 | 3 | 0 |
| cube | P2 | legacy | guided_flow | 5 | 6.0 | 3 | 0 |
| cube | P2 | legacy | guided_flow | 10 | 6.0 | 3 | 0 |
| cube | P2 | legacy | guided_flow | 32 | 4.0 | 2 | 0 |
| cube | P2 | legacy | none | 1 | 0.0 | 0 | 0 |
| cube | P2 | legacy | none | 2 | -2.0 | 0 | 1 |
| cube | P2 | legacy | none | 5 | 0.0 | 0 | 0 |
| cube | P2 | legacy | none | 10 | -2.0 | 0 | 1 |
| cube | P2 | legacy | none | 32 | 0.0 | 0 | 0 |
| cube | P2 | legacy | post_opt | 1 | 2.0 | 1 | 0 |
| cube | P2 | legacy | post_opt | 2 | 2.0 | 1 | 0 |
| cube | P2 | legacy | post_opt | 5 | 0.0 | 0 | 0 |
| cube | P2 | legacy | post_opt | 10 | 0.0 | 0 | 0 |
| cube | P2 | legacy | post_opt | 32 | 0.0 | 0 | 0 |
| cube | P3 | not_applicable | guided_flow | 1 | 0.0 | 0 | 0 |
| cube | P3 | not_applicable | guided_flow | 2 | 0.0 | 0 | 0 |
| cube | P3 | not_applicable | guided_flow | 5 | 0.0 | 0 | 0 |
| cube | P3 | not_applicable | guided_flow | 10 | 0.0 | 0 | 0 |
| cube | P3 | not_applicable | guided_flow | 32 | 0.0 | 0 | 0 |
| cube | P3 | not_applicable | none | 1 | 0.0 | 0 | 0 |
| cube | P3 | not_applicable | none | 2 | 0.0 | 0 | 0 |
| cube | P3 | not_applicable | none | 5 | 0.0 | 0 | 0 |
| cube | P3 | not_applicable | none | 10 | 0.0 | 0 | 0 |
| cube | P3 | not_applicable | none | 32 | 0.0 | 0 | 0 |
| cube | P3 | not_applicable | post_opt | 1 | 0.0 | 0 | 0 |
| cube | P3 | not_applicable | post_opt | 2 | 0.0 | 0 | 0 |
| cube | P3 | not_applicable | post_opt | 5 | 0.0 | 0 | 0 |
| cube | P3 | not_applicable | post_opt | 10 | 0.0 | 0 | 0 |
| cube | P3 | not_applicable | post_opt | 32 | 0.0 | 0 | 0 |
| cube | P3 | not_applicable | post_opt_refine | 1 | 0.0 | 0 | 0 |
| cube | P3 | not_applicable | post_opt_refine | 2 | 0.0 | 0 | 0 |
| cube | P3 | not_applicable | post_opt_refine | 5 | 0.0 | 0 | 0 |
| cube | P3 | not_applicable | post_opt_refine | 10 | 0.0 | 0 | 0 |
| cube | P3 | not_applicable | post_opt_refine | 32 | 0.0 | 0 | 0 |
| pusht | P0 | not_applicable | guided_flow | 1 | 2.0 | 2 | 1 |
| pusht | P0 | not_applicable | guided_flow | 2 | 4.0 | 2 | 0 |
| pusht | P0 | not_applicable | guided_flow | 5 | 2.0 | 1 | 0 |
| pusht | P0 | not_applicable | guided_flow | 10 | 0.0 | 0 | 0 |
| pusht | P0 | not_applicable | guided_flow | 32 | 0.0 | 0 | 0 |
| pusht | P0 | not_applicable | none | 1 | 2.0 | 3 | 2 |
| pusht | P0 | not_applicable | none | 2 | 6.0 | 3 | 0 |
| pusht | P0 | not_applicable | none | 5 | 4.0 | 2 | 0 |
| pusht | P0 | not_applicable | none | 10 | 0.0 | 0 | 0 |
| pusht | P0 | not_applicable | none | 32 | 0.0 | 0 | 0 |
| pusht | P0 | not_applicable | post_opt | 1 | 2.0 | 2 | 1 |
| pusht | P0 | not_applicable | post_opt | 2 | 2.0 | 2 | 1 |
| pusht | P0 | not_applicable | post_opt | 5 | 2.0 | 1 | 0 |
| pusht | P0 | not_applicable | post_opt | 10 | 2.0 | 1 | 0 |
| pusht | P0 | not_applicable | post_opt | 32 | 0.0 | 0 | 0 |
| pusht | P2 | legacy | guided_flow | 1 | 4.0 | 2 | 0 |
| pusht | P2 | legacy | guided_flow | 2 | 4.0 | 2 | 0 |
| pusht | P2 | legacy | guided_flow | 5 | 4.0 | 2 | 0 |
| pusht | P2 | legacy | guided_flow | 10 | 2.0 | 1 | 0 |
| pusht | P2 | legacy | guided_flow | 32 | 0.0 | 1 | 1 |
| pusht | P2 | legacy | none | 1 | 2.0 | 1 | 0 |
| pusht | P2 | legacy | none | 2 | 0.0 | 0 | 0 |
| pusht | P2 | legacy | none | 5 | -2.0 | 0 | 1 |
| pusht | P2 | legacy | none | 10 | 0.0 | 0 | 0 |
| pusht | P2 | legacy | none | 32 | 0.0 | 0 | 0 |
| pusht | P2 | legacy | post_opt | 1 | 0.0 | 0 | 0 |
| pusht | P2 | legacy | post_opt | 2 | -2.0 | 0 | 1 |
| pusht | P2 | legacy | post_opt | 5 | -2.0 | 0 | 1 |
| pusht | P2 | legacy | post_opt | 10 | -4.0 | 0 | 2 |
| pusht | P2 | legacy | post_opt | 32 | -4.0 | 0 | 2 |
| pusht | P3 | not_applicable | guided_flow | 1 | 2.0 | 1 | 0 |
| pusht | P3 | not_applicable | guided_flow | 2 | 2.0 | 1 | 0 |
| pusht | P3 | not_applicable | guided_flow | 5 | 0.0 | 0 | 0 |
| pusht | P3 | not_applicable | guided_flow | 10 | 0.0 | 0 | 0 |
| pusht | P3 | not_applicable | guided_flow | 32 | 0.0 | 0 | 0 |
| pusht | P3 | not_applicable | none | 1 | 0.0 | 2 | 2 |
| pusht | P3 | not_applicable | none | 2 | 2.0 | 2 | 1 |
| pusht | P3 | not_applicable | none | 5 | 4.0 | 2 | 0 |
| pusht | P3 | not_applicable | none | 10 | 2.0 | 1 | 0 |
| pusht | P3 | not_applicable | none | 32 | 2.0 | 1 | 0 |
| pusht | P3 | not_applicable | post_opt | 1 | 0.0 | 1 | 1 |
| pusht | P3 | not_applicable | post_opt | 2 | -2.0 | 1 | 2 |
| pusht | P3 | not_applicable | post_opt | 5 | 2.0 | 1 | 0 |
| pusht | P3 | not_applicable | post_opt | 10 | -2.0 | 1 | 2 |
| pusht | P3 | not_applicable | post_opt | 32 | 0.0 | 0 | 0 |
| pusht | P3 | not_applicable | post_opt_refine | 1 | 4.0 | 2 | 0 |
| pusht | P3 | not_applicable | post_opt_refine | 2 | 4.0 | 2 | 0 |
| pusht | P3 | not_applicable | post_opt_refine | 5 | 4.0 | 2 | 0 |
| pusht | P3 | not_applicable | post_opt_refine | 10 | 0.0 | 1 | 1 |
| pusht | P3 | not_applicable | post_opt_refine | 32 | 2.0 | 1 | 0 |
| reacher | P0 | not_applicable | guided_flow | 1 | 0.0 | 6 | 6 |
| reacher | P0 | not_applicable | guided_flow | 2 | 2.0 | 4 | 3 |
| reacher | P0 | not_applicable | guided_flow | 5 | -10.0 | 2 | 7 |
| reacher | P0 | not_applicable | guided_flow | 10 | -6.0 | 2 | 5 |
| reacher | P0 | not_applicable | guided_flow | 32 | -4.0 | 1 | 3 |
| reacher | P0 | not_applicable | none | 1 | -2.0 | 11 | 12 |
| reacher | P0 | not_applicable | none | 2 | 2.0 | 9 | 8 |
| reacher | P0 | not_applicable | none | 5 | 0.0 | 7 | 7 |
| reacher | P0 | not_applicable | none | 10 | 4.0 | 4 | 2 |
| reacher | P0 | not_applicable | none | 32 | 4.0 | 7 | 5 |
| reacher | P0 | not_applicable | post_opt | 1 | 14.0 | 12 | 5 |
| reacher | P0 | not_applicable | post_opt | 2 | 14.0 | 9 | 2 |
| reacher | P0 | not_applicable | post_opt | 5 | 12.0 | 7 | 1 |
| reacher | P0 | not_applicable | post_opt | 10 | 6.0 | 6 | 3 |
| reacher | P0 | not_applicable | post_opt | 32 | 4.0 | 3 | 1 |
| reacher | P2 | cem-clip | guided_flow | 1 | 14.0 | 8 | 1 |
| reacher | P2 | cem-clip | guided_flow | 2 | 8.0 | 7 | 3 |
| reacher | P2 | cem-clip | guided_flow | 5 | 10.0 | 7 | 2 |
| reacher | P2 | cem-clip | guided_flow | 10 | 0.0 | 4 | 4 |
| reacher | P2 | cem-clip | guided_flow | 32 | 0.0 | 4 | 4 |
| reacher | P2 | cem-clip | none | 1 | 4.0 | 6 | 4 |
| reacher | P2 | cem-clip | none | 2 | 2.0 | 4 | 3 |
| reacher | P2 | cem-clip | none | 5 | 6.0 | 4 | 1 |
| reacher | P2 | cem-clip | none | 10 | 4.0 | 4 | 2 |
| reacher | P2 | cem-clip | none | 32 | -4.0 | 2 | 4 |
| reacher | P2 | cem-clip | post_opt | 1 | 4.0 | 3 | 1 |
| reacher | P2 | cem-clip | post_opt | 2 | -6.0 | 2 | 5 |
| reacher | P2 | cem-clip | post_opt | 5 | -2.0 | 1 | 2 |
| reacher | P2 | cem-clip | post_opt | 10 | -8.0 | 0 | 4 |
| reacher | P2 | cem-clip | post_opt | 32 | -12.0 | 1 | 7 |
| reacher | P3 | not_applicable | guided_flow | 1 | 14.0 | 11 | 4 |
| reacher | P3 | not_applicable | guided_flow | 2 | 14.0 | 11 | 4 |
| reacher | P3 | not_applicable | guided_flow | 5 | 2.0 | 8 | 7 |
| reacher | P3 | not_applicable | guided_flow | 10 | 10.0 | 10 | 5 |
| reacher | P3 | not_applicable | guided_flow | 32 | 4.0 | 7 | 5 |
| reacher | P3 | not_applicable | none | 1 | -4.0 | 8 | 10 |
| reacher | P3 | not_applicable | none | 2 | 12.0 | 9 | 3 |
| reacher | P3 | not_applicable | none | 5 | 6.0 | 9 | 6 |
| reacher | P3 | not_applicable | none | 10 | 8.0 | 8 | 4 |
| reacher | P3 | not_applicable | none | 32 | 6.0 | 7 | 4 |
| reacher | P3 | not_applicable | post_opt | 1 | 8.0 | 8 | 4 |
| reacher | P3 | not_applicable | post_opt | 2 | 10.0 | 8 | 3 |
| reacher | P3 | not_applicable | post_opt | 5 | 0.0 | 5 | 5 |
| reacher | P3 | not_applicable | post_opt | 10 | -2.0 | 3 | 4 |
| reacher | P3 | not_applicable | post_opt | 32 | 0.0 | 4 | 4 |
| reacher | P3 | not_applicable | post_opt_refine | 1 | 8.0 | 7 | 3 |
| reacher | P3 | not_applicable | post_opt_refine | 2 | 0.0 | 7 | 7 |
| reacher | P3 | not_applicable | post_opt_refine | 5 | 8.0 | 7 | 3 |
| reacher | P3 | not_applicable | post_opt_refine | 10 | 0.0 | 4 | 4 |
| reacher | P3 | not_applicable | post_opt_refine | 32 | 6.0 | 5 | 2 |
| tworoom | P0 | not_applicable | guided_flow | 1 | 2.0 | 1 | 0 |
| tworoom | P0 | not_applicable | guided_flow | 2 | 0.0 | 1 | 1 |
| tworoom | P0 | not_applicable | guided_flow | 5 | 2.0 | 1 | 0 |
| tworoom | P0 | not_applicable | guided_flow | 10 | 0.0 | 0 | 0 |
| tworoom | P0 | not_applicable | guided_flow | 32 | 0.0 | 0 | 0 |
| tworoom | P0 | not_applicable | none | 1 | 2.0 | 1 | 0 |
| tworoom | P0 | not_applicable | none | 2 | -2.0 | 1 | 2 |
| tworoom | P0 | not_applicable | none | 5 | 0.0 | 0 | 0 |
| tworoom | P0 | not_applicable | none | 10 | 0.0 | 0 | 0 |
| tworoom | P0 | not_applicable | none | 32 | -2.0 | 0 | 1 |
| tworoom | P0 | not_applicable | post_opt | 1 | 2.0 | 1 | 0 |
| tworoom | P0 | not_applicable | post_opt | 2 | 0.0 | 1 | 1 |
| tworoom | P0 | not_applicable | post_opt | 5 | 0.0 | 0 | 0 |
| tworoom | P0 | not_applicable | post_opt | 10 | 0.0 | 0 | 0 |
| tworoom | P0 | not_applicable | post_opt | 32 | 0.0 | 0 | 0 |
| tworoom | P2 | legacy | guided_flow | 1 | 0.0 | 0 | 0 |
| tworoom | P2 | legacy | guided_flow | 2 | 0.0 | 0 | 0 |
| tworoom | P2 | legacy | guided_flow | 5 | 0.0 | 0 | 0 |
| tworoom | P2 | legacy | guided_flow | 10 | 0.0 | 0 | 0 |
| tworoom | P2 | legacy | guided_flow | 32 | 0.0 | 0 | 0 |
| tworoom | P2 | legacy | none | 1 | 2.0 | 1 | 0 |
| tworoom | P2 | legacy | none | 2 | 2.0 | 1 | 0 |
| tworoom | P2 | legacy | none | 5 | 2.0 | 1 | 0 |
| tworoom | P2 | legacy | none | 10 | 2.0 | 1 | 0 |
| tworoom | P2 | legacy | none | 32 | 2.0 | 1 | 0 |
| tworoom | P2 | legacy | post_opt | 1 | 0.0 | 0 | 0 |
| tworoom | P2 | legacy | post_opt | 2 | 0.0 | 0 | 0 |
| tworoom | P2 | legacy | post_opt | 5 | 0.0 | 0 | 0 |
| tworoom | P2 | legacy | post_opt | 10 | -2.0 | 0 | 1 |
| tworoom | P2 | legacy | post_opt | 32 | -2.0 | 0 | 1 |
| tworoom | P3 | not_applicable | guided_flow | 1 | 0.0 | 0 | 0 |
| tworoom | P3 | not_applicable | guided_flow | 2 | 0.0 | 0 | 0 |
| tworoom | P3 | not_applicable | guided_flow | 5 | 0.0 | 0 | 0 |
| tworoom | P3 | not_applicable | guided_flow | 10 | -2.0 | 0 | 1 |
| tworoom | P3 | not_applicable | guided_flow | 32 | -2.0 | 0 | 1 |
| tworoom | P3 | not_applicable | none | 1 | 0.0 | 0 | 0 |
| tworoom | P3 | not_applicable | none | 2 | 0.0 | 0 | 0 |
| tworoom | P3 | not_applicable | none | 5 | 0.0 | 0 | 0 |
| tworoom | P3 | not_applicable | none | 10 | 0.0 | 0 | 0 |
| tworoom | P3 | not_applicable | none | 32 | 0.0 | 0 | 0 |
| tworoom | P3 | not_applicable | post_opt | 1 | 0.0 | 0 | 0 |
| tworoom | P3 | not_applicable | post_opt | 2 | 0.0 | 0 | 0 |
| tworoom | P3 | not_applicable | post_opt | 5 | 0.0 | 0 | 0 |
| tworoom | P3 | not_applicable | post_opt | 10 | 0.0 | 0 | 0 |
| tworoom | P3 | not_applicable | post_opt | 32 | 0.0 | 0 | 0 |
| tworoom | P3 | not_applicable | post_opt_refine | 1 | 0.0 | 0 | 0 |
| tworoom | P3 | not_applicable | post_opt_refine | 2 | 0.0 | 0 | 0 |
| tworoom | P3 | not_applicable | post_opt_refine | 5 | 0.0 | 0 | 0 |
| tworoom | P3 | not_applicable | post_opt_refine | 10 | 0.0 | 0 | 0 |
| tworoom | P3 | not_applicable | post_opt_refine | 32 | 0.0 | 0 | 0 |

## 方向性结论

- 学习 planner 相对 P1：36 个正向、24 个负向、12 个持平（比较数 72）。
- guidance 相对无 guidance：56 个正向、25 个负向、87 个持平（比较数 168）。
- P3 guidance 语义互比：22 个正向、9 个负向、41 个持平（比较数 72）。
- step 相对 step16：73 个正向、26 个负向、101 个持平（比较数 200）。
- 所有比较均在同一 legacy cohort 的相同 episode identity 上配对。
- 单 checkpoint、单训练 seed、50 episodes 的描述性结果，不外推到 final 或其他 seed。

## 产物索引

- 分析 JSON：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase1_seed3072_legacy/analysis/analysis.json`
- 条件 CSV：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase1_seed3072_legacy/analysis/conditions.csv`
- 比较 CSV：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase1_seed3072_legacy/analysis/comparisons.csv`
- 本报告：`/data/users/wenxin/pre-exp/le-wm/docs/report/round5/round5_phase1_report.md`
- 代码 commit：`17e8a9938aa702c1428fce21368d7da49931caed`
