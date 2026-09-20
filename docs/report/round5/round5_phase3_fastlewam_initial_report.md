# Round 5 Phase 3：FastLeWAM 初版结果报告

> 状态：FastLeWAM 的三任务完整条件矩阵已完成；本报告暂不包含 LeWM/LeFlow 基线比较。
> 生成时间：2026-09-20T23:26:58+08:00

本报告整理 OGBench-Scene、DMC-Finger-turn_hard 和 DMC-Humanoid-walk 上的 FastLeWAM 结果。每个任务 61 个条件、每个条件 50 个 episode，共 183 个条件和 9,150 个 episode。主结果按照 Phase 3 方案固定为 P3、flow step=1、无 guidance；P1 invariant、flow-step 和 guidance 条件用于诊断。成功率为 episode 成功数除以 50，区间为 Wilson 95% 区间。

## 当前可读结论

- **Scene**：61 个条件全部为 0/50（0.0%）；flow step 或 guidance 未改变 all-component 主成功判据。
- **Finger**：全矩阵范围为 86.0%–92.0%；主条件为 45/50 (90.0%, [78.6, 95.7])。
- **Humanoid**：全矩阵范围为 16.0%–30.0%；主条件为 10/50 (20.0%, [11.2, 33.0])。
- 这些是固定 cohort、单训练 seed=3072 的描述性结果；在基线完成前，不据此做 FastLeWAM 相对 LeWM/LeFlow 的结论。

## 评测协议与模型

| 项目 | 设置 |
|---|---|
| 训练 | Round 5 Phase 1 同款 R4-AB FastLeWAM，epoch 10，训练 seed 3072 |
| 评测 | eval seed 42，50 个 start–goal 对，goal offset 25，execution budget 50，horizon/receding horizon 5，action block 5 |
| flow 条件 | P0/P2/P3 使用 Euler steps {1, 2, 5, 10, 16, 32}；P1 使用 invariant 条件 |
| guidance | P0/P2：guided_flow、post_opt；P3：guided_flow、post_opt、post_opt_refine；step size=0.01，late/inner steps=5，max RMS offset=0.2 |
| P2 协议 | Scene 使用 legacy CEM；Finger/Humanoid 使用 cem-clip；P0/P3 不使用 CEM clipping |
| 成功判据 | Scene 为 cube、button、drawer、window 五项同时满足；DMC 使用环境 termination 判据，未使用 NaN success 字段 |

## 数据完整性

| 任务 | 条件数 | episode 数 | 成功率范围 | cohort id | cohort SHA-256 |
|---|---:|---:|---:|---|---|
| OGBench-Scene | 61/61 | 3050 | 0.0%–0.0% | `scene_legacy_50_phase3_v1` | `ec30c7373119c070d40190f98bbdc43dea57cd767b1860cb1c070f6233d5f771` |
| DMC-Finger-turn_hard | 61/61 | 3050 | 86.0%–92.0% | `finger_legacy_50_phase3_v1` | `6f3a7722d466b666ba1af2d440a83157bc24edbde5c7ac449bfeb9696c58eb8c` |
| DMC-Humanoid-walk | 61/61 | 3050 | 16.0%–30.0% | `humanoid_legacy_50_phase3_v1` | `8fa169404c13365d19d894928923b4eae625504ee5d5fc988afd09a4c14262cc` |

- 183/183 条件状态为 `ok`，每项正好 50 episodes；没有缺条件、NaN 成功指标或 cohort hash 不一致。
- 任务级 checkpoint 均为 epoch 10；各条件的 `code_commit` 随执行期间兼容性修订而不同，完整列表见汇总 JSON。

## 主条件与 P1 invariant

主条件是 `P3/not_applicable/none/step_1/euler`。P1 在实现中没有 action-flow step，因此以 invariant 表示；视频目录名使用 `p1_step_1` 作为展示标签。

| 任务 | 主条件 P3/step1/none | P1 invariant |
|---|---:|---:|
| OGBench-Scene | 0/50 (0.0%, [0.0, 7.1]) | 0/50 (0.0%, [0.0, 7.1]) |
| DMC-Finger-turn_hard | 45/50 (90.0%, [78.6, 95.7]) | 44/50 (88.0%, [76.2, 94.4]) |
| DMC-Humanoid-walk | 10/50 (20.0%, [11.2, 33.0]) | 8/50 (16.0%, [8.3, 28.5]) |

## Flow-step 敏感性（无 guidance）

表中每个单元格为 `成功数/50（成功率）`。

| 任务 | mode | step=1 | step=2 | step=5 | step=10 | step=16 | step=32 |
|---|---|---:|---:|---:|---:|---:|---:|
| OGBench-Scene | P0 | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) |
| OGBench-Scene | P2 | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) |
| OGBench-Scene | P3 | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) |
| DMC-Finger-turn_hard | P0 | 44/50 (88.0%) | 44/50 (88.0%) | 44/50 (88.0%) | 44/50 (88.0%) | 44/50 (88.0%) | 44/50 (88.0%) |
| DMC-Finger-turn_hard | P2 | 43/50 (86.0%) | 44/50 (88.0%) | 45/50 (90.0%) | 44/50 (88.0%) | 44/50 (88.0%) | 45/50 (90.0%) |
| DMC-Finger-turn_hard | P3 | 45/50 (90.0%) | 45/50 (90.0%) | 45/50 (90.0%) | 44/50 (88.0%) | 45/50 (90.0%) | 45/50 (90.0%) |
| DMC-Humanoid-walk | P0 | 10/50 (20.0%) | 12/50 (24.0%) | 13/50 (26.0%) | 11/50 (22.0%) | 12/50 (24.0%) | 12/50 (24.0%) |
| DMC-Humanoid-walk | P2 | 8/50 (16.0%) | 8/50 (16.0%) | 9/50 (18.0%) | 9/50 (18.0%) | 9/50 (18.0%) | 9/50 (18.0%) |
| DMC-Humanoid-walk | P3 | 10/50 (20.0%) | 10/50 (20.0%) | 13/50 (26.0%) | 13/50 (26.0%) | 11/50 (22.0%) | 13/50 (26.0%) |

## Guidance 在 step=1 的变化

Δ 是同一 mode、同一 flow step 下相对无 guidance 的百分点变化。

| 任务 | mode | 无 guidance | guided_flow Δ | post_opt Δ | post_opt_refine Δ |
|---|---|---:|---:|---:|---:|
| OGBench-Scene | P0 | 0.0% | +0.0 pp | +0.0 pp | — |
| OGBench-Scene | P2 | 0.0% | +0.0 pp | +0.0 pp | — |
| OGBench-Scene | P3 | 0.0% | +0.0 pp | +0.0 pp | +0.0 pp |
| DMC-Finger-turn_hard | P0 | 88.0% | +0.0 pp | +0.0 pp | — |
| DMC-Finger-turn_hard | P2 | 86.0% | +0.0 pp | +0.0 pp | — |
| DMC-Finger-turn_hard | P3 | 90.0% | -4.0 pp | -4.0 pp | -4.0 pp |
| DMC-Humanoid-walk | P0 | 20.0% | +0.0 pp | +0.0 pp | — |
| DMC-Humanoid-walk | P2 | 16.0% | +2.0 pp | +2.0 pp | — |
| DMC-Humanoid-walk | P3 | 20.0% | +2.0 pp | +2.0 pp | +0.0 pp |

完整 guidance×flow-step 差值、每项 Wilson 区间以及 183 个结果的精确路径写入 [fastlewam_conditions.csv](../../../outputs/round5/phase3/analysis/fastlewam_conditions.csv)；可复现汇总写入 [fastlewam_summary.json](../../../outputs/round5/phase3/analysis/fastlewam_summary.json)。

## Scene target_task 分组

Scene 的 all-component 成功率仍是主指标；下面给出主条件和 P1 invariant 按冻结 cohort 中 target_task 的分组结果。分组样本数为 button=9, cube=19, drawer=8, window=14。

| 条件 | cube | button | drawer | window |
|---|---:|---:|---:|---:|
| P3/step1/none | 0/19 (0.0%) | 0/9 (0.0%) | 0/8 (0.0%) | 0/14 (0.0%) |
| P1/invariant | 0/19 (0.0%) | 0/9 (0.0%) | 0/8 (0.0%) | 0/14 (0.0%) |

完整 61 条 Scene 分组统计位于 `scene_target_task_results` 字段。

## 代表视频

每个视频来自冻结 cohort 的第一个 start–goal 对，仅用于可视化，不计入 50-episode 统计。

| 任务 | 条件 | 视频 |
|---|---|---|
| OGBench-Scene | `p1_step_1` | [env_0.mp4](../../../outputs/round5/phase3/videos/fastlewam/scene/p1_step_1/videos/env_0.mp4) |
| OGBench-Scene | `p3_step_1` | [env_0.mp4](../../../outputs/round5/phase3/videos/fastlewam/scene/p3_step_1/videos/env_0.mp4) |
| DMC-Finger-turn_hard | `p1_step_1` | [env_0.mp4](../../../outputs/round5/phase3/videos/fastlewam/finger/p1_step_1/videos/env_0.mp4) |
| DMC-Finger-turn_hard | `p3_step_1` | [env_0.mp4](../../../outputs/round5/phase3/videos/fastlewam/finger/p3_step_1/videos/env_0.mp4) |
| DMC-Humanoid-walk | `p1_step_1` | [env_0.mp4](../../../outputs/round5/phase3/videos/fastlewam/humanoid/p1_step_1/videos/env_0.mp4) |
| DMC-Humanoid-walk | `p3_step_1` | [env_0.mp4](../../../outputs/round5/phase3/videos/fastlewam/humanoid/p3_step_1/videos/env_0.mp4) |

## 解释边界与后续

- 这是单训练 seed=3072、固定 50-episode cohort 的初版整理，不能替代多 seed 统计。
- Scene 当前全条件为 0%，需要在基线完成后结合 LeWM/LeFlow、成功向量和视频共同检查；本报告不把它解释为模型能力的最终结论。
- LeWM/LeFlow 的训练、标准条件评测和对应视频仍在 Phase 3 后续步骤中；完成后再生成包含 189 行结果、paired comparison 和 McNemar 检验的总报告。

## 产物

- 初版报告：`docs/report/round5/round5_phase3_fastlewam_initial_report.md`
- 逐条件 CSV：`outputs/round5/phase3/analysis/fastlewam_conditions.csv`
- 汇总 JSON：`outputs/round5/phase3/analysis/fastlewam_summary.json`
- 代表视频目录：`outputs/round5/phase3/videos/fastlewam/`
