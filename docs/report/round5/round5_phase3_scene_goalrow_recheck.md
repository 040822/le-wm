# Round 5 Phase 3 Scene 目标行修正与重评

## 结论

旧 Scene 评测把规划目标图像取自 `start_step + 25` 的未来行，却把环境成功终止目标恢复为 episode 级 `privileged_target_*` 任务目标。cube、drawer、window 的目标因此经常不同；旧 all-component 条件下 FastLeWAM 61 项和 LeWM 主评测的 0% 不能用于判断模型是否完成了对应的未来目标。

修正后，规划目标图像和环境目标都来自同一个 future goal row。保留原有 Scene all-component 成功判据（cube、两个 button、drawer、window 同时达标；连续位置误差不超过 0.04，button 状态精确相等），沿用同一 50-entry cohort。FastLeWAM 的 61 项成功率为 40.0–56.0%，平均 44.92%。

## 主结果

| 方法 | 条件 | 成功数 | 成功率 | Wilson 95% 区间 |
|---|---|---:|---:|---:|
| FastLeWAM | P3 / flow step 1 / no guidance | 22/50 | 44.0% | 31.2–57.7% |
| LeWM | standard | 23/50 | 46.0% | 33.0–59.6% |

同 cohort 配对比较：FastLeWAM 相对 LeWM improved=1、regressed=2、discordant=3，双侧 exact McNemar p=1.0000。两种方法的点估计接近；本单种子、50 episode 结果不足以得出方法优劣结论。

Scene 任务类型分组使用未来 goal row 的 `privileged_target_task` 标签；分组仅用于统计，不改变 all-component 主成功条件。

| 目标任务 | Cohort 数 | FastLeWAM P3/step1 | LeWM standard |
|---|---:|---:|---:|
| button | 9 | 3/9 (33.3%) | 3/9 (33.3%) |
| cube | 19 | 8/19 (42.1%) | 8/19 (42.1%) |
| drawer | 8 | 5/8 (62.5%) | 6/8 (75.0%) |
| window | 14 | 6/14 (42.9%) | 6/14 (42.9%) |

## FastLeWAM 61 条件矩阵

| 模式 | Guidance | Flow step | 成功数 | 成功率 |
|---|---|---:|---:|---:|
| P0 | none | 1 | 22/50 | 44.0% |
| P0 | none | 2 | 21/50 | 42.0% |
| P0 | none | 5 | 21/50 | 42.0% |
| P0 | none | 10 | 22/50 | 44.0% |
| P0 | none | 16 | 23/50 | 46.0% |
| P0 | none | 32 | 21/50 | 42.0% |
| P1 | none | invariant | 23/50 | 46.0% |
| P2 | none | 1 | 23/50 | 46.0% |
| P2 | none | 2 | 22/50 | 44.0% |
| P2 | none | 5 | 24/50 | 48.0% |
| P2 | none | 10 | 28/50 | 56.0% |
| P2 | none | 16 | 21/50 | 42.0% |
| P2 | none | 32 | 28/50 | 56.0% |
| P3 | none | 1 | 22/50 | 44.0% |
| P3 | none | 2 | 22/50 | 44.0% |
| P3 | none | 5 | 21/50 | 42.0% |
| P3 | none | 10 | 21/50 | 42.0% |
| P3 | none | 16 | 22/50 | 44.0% |
| P3 | none | 32 | 23/50 | 46.0% |
| P0 | guided_flow | 1 | 22/50 | 44.0% |
| P2 | guided_flow | 1 | 23/50 | 46.0% |
| P0 | post_opt | 1 | 22/50 | 44.0% |
| P2 | post_opt | 1 | 23/50 | 46.0% |
| P3 | guided_flow | 1 | 22/50 | 44.0% |
| P3 | post_opt | 1 | 22/50 | 44.0% |
| P3 | post_opt_refine | 1 | 22/50 | 44.0% |
| P0 | guided_flow | 2 | 21/50 | 42.0% |
| P2 | guided_flow | 2 | 22/50 | 44.0% |
| P0 | post_opt | 2 | 21/50 | 42.0% |
| P2 | post_opt | 2 | 22/50 | 44.0% |
| P3 | guided_flow | 2 | 22/50 | 44.0% |
| P3 | post_opt | 2 | 22/50 | 44.0% |
| P3 | post_opt_refine | 2 | 22/50 | 44.0% |
| P0 | guided_flow | 5 | 21/50 | 42.0% |
| P2 | guided_flow | 5 | 24/50 | 48.0% |
| P0 | post_opt | 5 | 21/50 | 42.0% |
| P2 | post_opt | 5 | 24/50 | 48.0% |
| P3 | guided_flow | 5 | 20/50 | 40.0% |
| P3 | post_opt | 5 | 21/50 | 42.0% |
| P3 | post_opt_refine | 5 | 21/50 | 42.0% |
| P0 | guided_flow | 10 | 22/50 | 44.0% |
| P2 | guided_flow | 10 | 28/50 | 56.0% |
| P0 | post_opt | 10 | 22/50 | 44.0% |
| P2 | post_opt | 10 | 28/50 | 56.0% |
| P3 | guided_flow | 10 | 21/50 | 42.0% |
| P3 | post_opt | 10 | 21/50 | 42.0% |
| P3 | post_opt_refine | 10 | 21/50 | 42.0% |
| P0 | guided_flow | 16 | 23/50 | 46.0% |
| P2 | guided_flow | 16 | 21/50 | 42.0% |
| P0 | post_opt | 16 | 23/50 | 46.0% |
| P2 | post_opt | 16 | 21/50 | 42.0% |
| P3 | guided_flow | 16 | 22/50 | 44.0% |
| P3 | post_opt | 16 | 22/50 | 44.0% |
| P3 | post_opt_refine | 16 | 22/50 | 44.0% |
| P0 | guided_flow | 32 | 21/50 | 42.0% |
| P2 | guided_flow | 32 | 26/50 | 52.0% |
| P0 | post_opt | 32 | 21/50 | 42.0% |
| P2 | post_opt | 32 | 26/50 | 52.0% |
| P3 | guided_flow | 32 | 23/50 | 46.0% |
| P3 | post_opt | 32 | 23/50 | 46.0% |
| P3 | post_opt_refine | 32 | 23/50 | 46.0% |

| 模式 | 条件数 | 平均成功率 | 最低 | 最高 |
|---|---:|---:|---:|---:|
| P0 | 18 | 43.33% | 42.0% | 46.0% |
| P1 | 1 | 46.00% | 46.0% | 46.0% |
| P2 | 18 | 48.22% | 42.0% | 56.0% |
| P3 | 24 | 43.58% | 40.0% | 46.0% |

## 协议与产物

- Cohort：`scene_legacy_50_phase3_v1`，SHA-256 `ec30c7373119c070d40190f98bbdc43dea57cd767b1860cb1c070f6233d5f771`；eval seed 42、goal offset 25、50 episodes、预算 50。61/61 FastLeWAM 条件与 LeWM 均含 50 个 episode，episode 顺序一致且 cohort hash 相同。
- 目标字段来自 `goal_privileged_block_0_pos`、`goal_privileged_block_0_quat`、两个 `goal_privileged_button_*_state`、`goal_privileged_drawer_pos` 和 `goal_privileged_window_pos`。50 个目标行的上述字段均有限且形状符合预期。
- FastLeWAM checkpoint：`outputs/round5/phase3/training/fastlewam/scene/checkpoints/r4_ab_weights_epoch_10.pt`。LeWM checkpoint：`outputs/round5/phase3/training/lewm/scene/checkpoints/lewm_weights_epoch_10.pt`。未重新训练。
- LeFlow Scene 尚未训练；按用户要求，本次不补训或评测。
- 视频：
  - [FastLeWAM P1 invariant](../../../outputs/round5/phase3_scene_goalrow_v2/videos/fastlewam/scene/p1_step_1/videos/env_0.mp4)
  - [FastLeWAM P3 flow step 1](../../../outputs/round5/phase3_scene_goalrow_v2/videos/fastlewam/scene/p3_step_1/videos/env_0.mp4)
  - [LeWM standard](../../../outputs/round5/phase3_scene_goalrow_v2/videos/lewm/scene/standard/videos/env_0.mp4)

P1 在本阶段协议中是 invariant 条件，没有独立 flow-step 维度；视频目录保留历史命名 `p1_step_1`。

详细 `result.json` 与 episode 成功向量位于 `outputs/round5/phase3_scene_goalrow_v2/`。旧结果仍保留在 `outputs/round5/phase3/`，不应与修正后的结果混用。
