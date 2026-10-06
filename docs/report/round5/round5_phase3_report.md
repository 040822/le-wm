# Round 5 Phase 3：新增任务上的 FastLeWAM 与基线对比

本报告使用单训练 seed 3072、评测 seed 42 和固定 legacy 50-episode cohort。结果是描述性单种子实验，不构成 DeWM 论文六种子结果的严格复现。

## 主比较

主比较固定为 FastLeWAM R4-AB 的 P3 / action-flow step 1 / no guidance；LeWM 使用 CEM 300/30/30，LeFlow 使用 64 paths / 16 flow steps。

| task | FastLeWAM P3/s1/none | LeWM | LeFlow |
|---|---:|---:|---:|
| scene | 22/50 (44.0%, CI [31.2, 57.7]) | 23/50 (46.0%, CI [33.0, 59.6]) | 20/50 (40.0%, CI [27.6, 53.8]) |
| finger | 45/50 (90.0%, CI [78.6, 95.7]) | 44/50 (88.0%, CI [76.2, 94.4]) | 44/50 (88.0%, CI [76.2, 94.4]) |
| humanoid | 10/50 (20.0%, CI [11.2, 33.0]) | 8/50 (16.0%, CI [8.3, 28.5]) | 11/50 (22.0%, CI [12.8, 35.2]) |

| task | baseline | Δ pp (Fast − baseline) | improved | regressed | McNemar p |
|---|---|---:|---:|---:|---:|
| scene | lewm | -2.0 | 1 | 2 | 1.000000 |
| scene | leflow | 4.0 | 2 | 0 | 0.500000 |
| finger | lewm | 2.0 | 2 | 1 | 1.000000 |
| finger | leflow | 2.0 | 1 | 0 | 1.000000 |
| humanoid | lewm | 4.0 | 2 | 0 | 0.500000 |
| humanoid | leflow | -2.0 | 1 | 2 | 1.000000 |

## Flow-step 敏感性（无 guidance）

| task | mode | step | successes/50 | success rate |
|---|---|---:|---:|---:|
| scene | P0 | 1 | 22/50 | 44.0% |
| scene | P0 | 2 | 21/50 | 42.0% |
| scene | P0 | 5 | 21/50 | 42.0% |
| scene | P0 | 10 | 22/50 | 44.0% |
| scene | P0 | 16 | 23/50 | 46.0% |
| scene | P0 | 32 | 21/50 | 42.0% |
| scene | P2 | 1 | 23/50 | 46.0% |
| scene | P2 | 2 | 22/50 | 44.0% |
| scene | P2 | 5 | 24/50 | 48.0% |
| scene | P2 | 10 | 28/50 | 56.0% |
| scene | P2 | 16 | 21/50 | 42.0% |
| scene | P2 | 32 | 28/50 | 56.0% |
| scene | P3 | 1 | 22/50 | 44.0% |
| scene | P3 | 2 | 22/50 | 44.0% |
| scene | P3 | 5 | 21/50 | 42.0% |
| scene | P3 | 10 | 21/50 | 42.0% |
| scene | P3 | 16 | 22/50 | 44.0% |
| scene | P3 | 32 | 23/50 | 46.0% |
| finger | P0 | 1 | 44/50 | 88.0% |
| finger | P0 | 2 | 44/50 | 88.0% |
| finger | P0 | 5 | 44/50 | 88.0% |
| finger | P0 | 10 | 44/50 | 88.0% |
| finger | P0 | 16 | 44/50 | 88.0% |
| finger | P0 | 32 | 44/50 | 88.0% |
| finger | P2 | 1 | 43/50 | 86.0% |
| finger | P2 | 2 | 44/50 | 88.0% |
| finger | P2 | 5 | 45/50 | 90.0% |
| finger | P2 | 10 | 44/50 | 88.0% |
| finger | P2 | 16 | 44/50 | 88.0% |
| finger | P2 | 32 | 45/50 | 90.0% |
| finger | P3 | 1 | 45/50 | 90.0% |
| finger | P3 | 2 | 45/50 | 90.0% |
| finger | P3 | 5 | 45/50 | 90.0% |
| finger | P3 | 10 | 44/50 | 88.0% |
| finger | P3 | 16 | 45/50 | 90.0% |
| finger | P3 | 32 | 45/50 | 90.0% |
| humanoid | P0 | 1 | 10/50 | 20.0% |
| humanoid | P0 | 2 | 12/50 | 24.0% |
| humanoid | P0 | 5 | 13/50 | 26.0% |
| humanoid | P0 | 10 | 11/50 | 22.0% |
| humanoid | P0 | 16 | 12/50 | 24.0% |
| humanoid | P0 | 32 | 12/50 | 24.0% |
| humanoid | P2 | 1 | 8/50 | 16.0% |
| humanoid | P2 | 2 | 8/50 | 16.0% |
| humanoid | P2 | 5 | 9/50 | 18.0% |
| humanoid | P2 | 10 | 9/50 | 18.0% |
| humanoid | P2 | 16 | 9/50 | 18.0% |
| humanoid | P2 | 32 | 9/50 | 18.0% |
| humanoid | P3 | 1 | 10/50 | 20.0% |
| humanoid | P3 | 2 | 10/50 | 20.0% |
| humanoid | P3 | 5 | 13/50 | 26.0% |
| humanoid | P3 | 10 | 13/50 | 26.0% |
| humanoid | P3 | 16 | 11/50 | 22.0% |
| humanoid | P3 | 32 | 13/50 | 26.0% |

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
| scene | P2 | guided_flow | 32 | -4.0 |
| scene | P2 | post_opt | 32 | -4.0 |
| scene | P3 | guided_flow | 1 | +0.0 |
| scene | P3 | post_opt | 1 | +0.0 |
| scene | P3 | post_opt_refine | 1 | +0.0 |
| scene | P3 | guided_flow | 2 | +0.0 |
| scene | P3 | post_opt | 2 | +0.0 |
| scene | P3 | post_opt_refine | 2 | +0.0 |
| scene | P3 | guided_flow | 5 | -2.0 |
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
| finger | P0 | guided_flow | 2 | +2.0 |
| finger | P0 | post_opt | 2 | +0.0 |
| finger | P0 | guided_flow | 5 | -2.0 |
| finger | P0 | post_opt | 5 | +0.0 |
| finger | P0 | guided_flow | 10 | +2.0 |
| finger | P0 | post_opt | 10 | +0.0 |
| finger | P0 | guided_flow | 16 | +0.0 |
| finger | P0 | post_opt | 16 | +2.0 |
| finger | P0 | guided_flow | 32 | +2.0 |
| finger | P0 | post_opt | 32 | +2.0 |
| finger | P2 | guided_flow | 1 | +0.0 |
| finger | P2 | post_opt | 1 | +0.0 |
| finger | P2 | guided_flow | 2 | +0.0 |
| finger | P2 | post_opt | 2 | +0.0 |
| finger | P2 | guided_flow | 5 | +0.0 |
| finger | P2 | post_opt | 5 | +0.0 |
| finger | P2 | guided_flow | 10 | +0.0 |
| finger | P2 | post_opt | 10 | +0.0 |
| finger | P2 | guided_flow | 16 | +0.0 |
| finger | P2 | post_opt | 16 | +0.0 |
| finger | P2 | guided_flow | 32 | -2.0 |
| finger | P2 | post_opt | 32 | -2.0 |
| finger | P3 | guided_flow | 1 | -4.0 |
| finger | P3 | post_opt | 1 | -4.0 |
| finger | P3 | post_opt_refine | 1 | -4.0 |
| finger | P3 | guided_flow | 2 | -2.0 |
| finger | P3 | post_opt | 2 | +0.0 |
| finger | P3 | post_opt_refine | 2 | +0.0 |
| finger | P3 | guided_flow | 5 | +0.0 |
| finger | P3 | post_opt | 5 | +0.0 |
| finger | P3 | post_opt_refine | 5 | +0.0 |
| finger | P3 | guided_flow | 10 | -2.0 |
| finger | P3 | post_opt | 10 | +2.0 |
| finger | P3 | post_opt_refine | 10 | +0.0 |
| finger | P3 | guided_flow | 16 | -2.0 |
| finger | P3 | post_opt | 16 | +0.0 |
| finger | P3 | post_opt_refine | 16 | +0.0 |
| finger | P3 | guided_flow | 32 | +0.0 |
| finger | P3 | post_opt | 32 | +2.0 |
| finger | P3 | post_opt_refine | 32 | +0.0 |
| humanoid | P0 | guided_flow | 1 | +0.0 |
| humanoid | P0 | post_opt | 1 | +0.0 |
| humanoid | P0 | guided_flow | 2 | +2.0 |
| humanoid | P0 | post_opt | 2 | +0.0 |
| humanoid | P0 | guided_flow | 5 | +0.0 |
| humanoid | P0 | post_opt | 5 | +0.0 |
| humanoid | P0 | guided_flow | 10 | +2.0 |
| humanoid | P0 | post_opt | 10 | +2.0 |
| humanoid | P0 | guided_flow | 16 | +4.0 |
| humanoid | P0 | post_opt | 16 | +0.0 |
| humanoid | P0 | guided_flow | 32 | -2.0 |
| humanoid | P0 | post_opt | 32 | +0.0 |
| humanoid | P2 | guided_flow | 1 | +2.0 |
| humanoid | P2 | post_opt | 1 | +2.0 |
| humanoid | P2 | guided_flow | 2 | +0.0 |
| humanoid | P2 | post_opt | 2 | +0.0 |
| humanoid | P2 | guided_flow | 5 | +0.0 |
| humanoid | P2 | post_opt | 5 | +0.0 |
| humanoid | P2 | guided_flow | 10 | +2.0 |
| humanoid | P2 | post_opt | 10 | +0.0 |
| humanoid | P2 | guided_flow | 16 | +0.0 |
| humanoid | P2 | post_opt | 16 | +2.0 |
| humanoid | P2 | guided_flow | 32 | +0.0 |
| humanoid | P2 | post_opt | 32 | +0.0 |
| humanoid | P3 | guided_flow | 1 | +2.0 |
| humanoid | P3 | post_opt | 1 | +2.0 |
| humanoid | P3 | post_opt_refine | 1 | +0.0 |
| humanoid | P3 | guided_flow | 2 | +2.0 |
| humanoid | P3 | post_opt | 2 | +2.0 |
| humanoid | P3 | post_opt_refine | 2 | +0.0 |
| humanoid | P3 | guided_flow | 5 | -6.0 |
| humanoid | P3 | post_opt | 5 | -2.0 |
| humanoid | P3 | post_opt_refine | 5 | +2.0 |
| humanoid | P3 | guided_flow | 10 | +0.0 |
| humanoid | P3 | post_opt | 10 | +0.0 |
| humanoid | P3 | post_opt_refine | 10 | +4.0 |
| humanoid | P3 | guided_flow | 16 | +4.0 |
| humanoid | P3 | post_opt | 16 | +2.0 |
| humanoid | P3 | post_opt_refine | 16 | +0.0 |
| humanoid | P3 | guided_flow | 32 | -2.0 |
| humanoid | P3 | post_opt | 32 | +2.0 |
| humanoid | P3 | post_opt_refine | 32 | +2.0 |

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
- scene / fastlewam / p1_step_1: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/videos/fastlewam/scene/p1_step_1/videos/env_0.mp4 (736×288, 91309 bytes)。
- scene / fastlewam / p3_step_1: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/videos/fastlewam/scene/p3_step_1/videos/env_0.mp4 (736×288, 73937 bytes)。
- scene / lewm / standard: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/videos/lewm/scene/standard/videos/env_0.mp4 (736×288, 91921 bytes)。
- scene / leflow / standard: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/videos/leflow/scene/standard/videos/env_0.mp4 (736×288, 80965 bytes)。
- finger / fastlewam / p1_step_1: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/videos/fastlewam/finger/p1_step_1/videos/env_0.mp4 (736×288, 58132 bytes)。
- finger / fastlewam / p3_step_1: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/videos/fastlewam/finger/p3_step_1/videos/env_0.mp4 (736×288, 53126 bytes)。
- finger / lewm / standard: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/videos/lewm/finger/standard/videos/env_0.mp4 (736×288, 54040 bytes)。
- finger / leflow / standard: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/videos/leflow/finger/standard/videos/env_0.mp4 (736×288, 55299 bytes)。
- humanoid / fastlewam / p1_step_1: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/videos/fastlewam/humanoid/p1_step_1/videos/env_0.mp4 (736×288, 92738 bytes)。
- humanoid / fastlewam / p3_step_1: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/videos/fastlewam/humanoid/p3_step_1/videos/env_0.mp4 (736×288, 108179 bytes)。
- humanoid / lewm / standard: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/videos/lewm/humanoid/standard/videos/env_0.mp4 (736×288, 106500 bytes)。
- humanoid / leflow / standard: /data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/videos/leflow/humanoid/standard/videos/env_0.mp4 (736×288, 106594 bytes)。

## 环境适配验证时序

2026-09-25 的 CPU smoke 只创建并重置 Scene、Finger、Humanoid 环境，完成目标设置后执行一次零动作 step；它没有加载 checkpoint 或运行模型规划，因此不等同于计划要求的每任务单 episode 端到端 smoke。适配层边界测试也在批量实验启动后补做。两类检查均为后验验证，计划要求的批量实验前检查门槛未满足。
- 三环境 CPU smoke：`logs/phase3_environment_smoke_cpu.log`。
- Phase 3 单测输出：`logs/phase3_unit_tests.log`。

## LeWM 训练续训记录

以下列出本阶段 LeWM 训练中存在的续训元数据；未记录的字段不作推断。
- `scene`：source run `/home/wenxin/office/pre-exp/le-wm/outputs/round5_phase3_lewm_scene_resume_epoch6`；元数据记录完成至 epoch 10。元数据：`training/lewm/scene/resume_metadata.json`。
- `finger`：source run `/home/wenxin/office/pre-exp/le-wm/outputs/round5_phase3_lewm_finger_resume_epoch9_tmux`；optimizer state restored=false。元数据：`training/lewm/finger/resume_metadata.json`。
- `humanoid`：source run `/home/wenxin/office/pre-exp/le-wm/outputs/round5_phase3_lewm_humanoid_resume_epoch6`；epoch 6 接续至 epoch 10；seed=3072；optimizer state restored=false；续训方式 `model_weights_with_epoch_offset`。元数据：`training/lewm/humanoid/resume_metadata.json`。

## Scene target_task 分组

- `fast_lewam` / `P0/not_applicable/none/step_1/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=6/14 (42.9%)
- `fast_lewam` / `P0/not_applicable/none/step_2/euler`：button=3/9 (33.3%), cube=7/19 (36.8%), drawer=6/8 (75.0%), window=5/14 (35.7%)
- `fast_lewam` / `P0/not_applicable/none/step_5/euler`：button=3/9 (33.3%), cube=7/19 (36.8%), drawer=6/8 (75.0%), window=5/14 (35.7%)
- `fast_lewam` / `P0/not_applicable/none/step_10/euler`：button=3/9 (33.3%), cube=7/19 (36.8%), drawer=7/8 (87.5%), window=5/14 (35.7%)
- `fast_lewam` / `P0/not_applicable/none/step_16/euler`：button=4/9 (44.4%), cube=7/19 (36.8%), drawer=7/8 (87.5%), window=5/14 (35.7%)
- `fast_lewam` / `P0/not_applicable/none/step_32/euler`：button=3/9 (33.3%), cube=7/19 (36.8%), drawer=6/8 (75.0%), window=5/14 (35.7%)
- `fast_lewam` / `P1/legacy/none/invariant/not_applicable`：button=3/9 (33.3%), cube=7/19 (36.8%), drawer=6/8 (75.0%), window=7/14 (50.0%)
- `fast_lewam` / `P2/legacy/none/step_1/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=6/8 (75.0%), window=6/14 (42.9%)
- `fast_lewam` / `P2/legacy/none/step_2/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=6/14 (42.9%)
- `fast_lewam` / `P2/legacy/none/step_5/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=6/8 (75.0%), window=7/14 (50.0%)
- `fast_lewam` / `P2/legacy/none/step_10/euler`：button=5/9 (55.6%), cube=8/19 (42.1%), drawer=6/8 (75.0%), window=9/14 (64.3%)
- `fast_lewam` / `P2/legacy/none/step_16/euler`：button=3/9 (33.3%), cube=7/19 (36.8%), drawer=6/8 (75.0%), window=5/14 (35.7%)
- `fast_lewam` / `P2/legacy/none/step_32/euler`：button=4/9 (44.4%), cube=9/19 (47.4%), drawer=7/8 (87.5%), window=8/14 (57.1%)
- `fast_lewam` / `P3/not_applicable/none/step_1/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=6/14 (42.9%)
- `fast_lewam` / `P3/not_applicable/none/step_2/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=6/14 (42.9%)
- `fast_lewam` / `P3/not_applicable/none/step_5/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=5/14 (35.7%)
- `fast_lewam` / `P3/not_applicable/none/step_10/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=5/14 (35.7%)
- `fast_lewam` / `P3/not_applicable/none/step_16/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=6/8 (75.0%), window=5/14 (35.7%)
- `fast_lewam` / `P3/not_applicable/none/step_32/euler`：button=4/9 (44.4%), cube=8/19 (42.1%), drawer=6/8 (75.0%), window=5/14 (35.7%)
- `fast_lewam` / `P0/not_applicable/guided_flow/step_1/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=6/14 (42.9%)
- `fast_lewam` / `P2/legacy/guided_flow/step_1/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=6/8 (75.0%), window=6/14 (42.9%)
- `fast_lewam` / `P0/not_applicable/post_opt/step_1/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=6/14 (42.9%)
- `fast_lewam` / `P2/legacy/post_opt/step_1/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=6/8 (75.0%), window=6/14 (42.9%)
- `fast_lewam` / `P3/not_applicable/guided_flow/step_1/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=6/14 (42.9%)
- `fast_lewam` / `P3/not_applicable/post_opt/step_1/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=6/14 (42.9%)
- `fast_lewam` / `P3/not_applicable/post_opt_refine/step_1/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=6/14 (42.9%)
- `fast_lewam` / `P0/not_applicable/guided_flow/step_2/euler`：button=3/9 (33.3%), cube=7/19 (36.8%), drawer=6/8 (75.0%), window=5/14 (35.7%)
- `fast_lewam` / `P2/legacy/guided_flow/step_2/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=6/14 (42.9%)
- `fast_lewam` / `P0/not_applicable/post_opt/step_2/euler`：button=3/9 (33.3%), cube=7/19 (36.8%), drawer=6/8 (75.0%), window=5/14 (35.7%)
- `fast_lewam` / `P2/legacy/post_opt/step_2/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=6/14 (42.9%)
- `fast_lewam` / `P3/not_applicable/guided_flow/step_2/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=6/14 (42.9%)
- `fast_lewam` / `P3/not_applicable/post_opt/step_2/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=6/14 (42.9%)
- `fast_lewam` / `P3/not_applicable/post_opt_refine/step_2/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=6/14 (42.9%)
- `fast_lewam` / `P0/not_applicable/guided_flow/step_5/euler`：button=3/9 (33.3%), cube=7/19 (36.8%), drawer=6/8 (75.0%), window=5/14 (35.7%)
- `fast_lewam` / `P2/legacy/guided_flow/step_5/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=6/8 (75.0%), window=7/14 (50.0%)
- `fast_lewam` / `P0/not_applicable/post_opt/step_5/euler`：button=3/9 (33.3%), cube=7/19 (36.8%), drawer=6/8 (75.0%), window=5/14 (35.7%)
- `fast_lewam` / `P2/legacy/post_opt/step_5/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=6/8 (75.0%), window=7/14 (50.0%)
- `fast_lewam` / `P3/not_applicable/guided_flow/step_5/euler`：button=3/9 (33.3%), cube=7/19 (36.8%), drawer=5/8 (62.5%), window=5/14 (35.7%)
- `fast_lewam` / `P3/not_applicable/post_opt/step_5/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=5/14 (35.7%)
- `fast_lewam` / `P3/not_applicable/post_opt_refine/step_5/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=5/14 (35.7%)
- `fast_lewam` / `P0/not_applicable/guided_flow/step_10/euler`：button=3/9 (33.3%), cube=7/19 (36.8%), drawer=7/8 (87.5%), window=5/14 (35.7%)
- `fast_lewam` / `P2/legacy/guided_flow/step_10/euler`：button=5/9 (55.6%), cube=8/19 (42.1%), drawer=6/8 (75.0%), window=9/14 (64.3%)
- `fast_lewam` / `P0/not_applicable/post_opt/step_10/euler`：button=3/9 (33.3%), cube=7/19 (36.8%), drawer=7/8 (87.5%), window=5/14 (35.7%)
- `fast_lewam` / `P2/legacy/post_opt/step_10/euler`：button=5/9 (55.6%), cube=8/19 (42.1%), drawer=6/8 (75.0%), window=9/14 (64.3%)
- `fast_lewam` / `P3/not_applicable/guided_flow/step_10/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=5/14 (35.7%)
- `fast_lewam` / `P3/not_applicable/post_opt/step_10/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=5/14 (35.7%)
- `fast_lewam` / `P3/not_applicable/post_opt_refine/step_10/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=5/8 (62.5%), window=5/14 (35.7%)
- `fast_lewam` / `P0/not_applicable/guided_flow/step_16/euler`：button=4/9 (44.4%), cube=7/19 (36.8%), drawer=7/8 (87.5%), window=5/14 (35.7%)
- `fast_lewam` / `P2/legacy/guided_flow/step_16/euler`：button=3/9 (33.3%), cube=7/19 (36.8%), drawer=6/8 (75.0%), window=5/14 (35.7%)
- `fast_lewam` / `P0/not_applicable/post_opt/step_16/euler`：button=4/9 (44.4%), cube=7/19 (36.8%), drawer=7/8 (87.5%), window=5/14 (35.7%)
- `fast_lewam` / `P2/legacy/post_opt/step_16/euler`：button=3/9 (33.3%), cube=7/19 (36.8%), drawer=6/8 (75.0%), window=5/14 (35.7%)
- `fast_lewam` / `P3/not_applicable/guided_flow/step_16/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=6/8 (75.0%), window=5/14 (35.7%)
- `fast_lewam` / `P3/not_applicable/post_opt/step_16/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=6/8 (75.0%), window=5/14 (35.7%)
- `fast_lewam` / `P3/not_applicable/post_opt_refine/step_16/euler`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=6/8 (75.0%), window=5/14 (35.7%)
- `fast_lewam` / `P0/not_applicable/guided_flow/step_32/euler`：button=3/9 (33.3%), cube=7/19 (36.8%), drawer=6/8 (75.0%), window=5/14 (35.7%)
- `fast_lewam` / `P2/legacy/guided_flow/step_32/euler`：button=4/9 (44.4%), cube=9/19 (47.4%), drawer=6/8 (75.0%), window=7/14 (50.0%)
- `fast_lewam` / `P0/not_applicable/post_opt/step_32/euler`：button=3/9 (33.3%), cube=7/19 (36.8%), drawer=6/8 (75.0%), window=5/14 (35.7%)
- `fast_lewam` / `P2/legacy/post_opt/step_32/euler`：button=4/9 (44.4%), cube=9/19 (47.4%), drawer=6/8 (75.0%), window=7/14 (50.0%)
- `fast_lewam` / `P3/not_applicable/guided_flow/step_32/euler`：button=4/9 (44.4%), cube=8/19 (42.1%), drawer=6/8 (75.0%), window=5/14 (35.7%)
- `fast_lewam` / `P3/not_applicable/post_opt/step_32/euler`：button=4/9 (44.4%), cube=8/19 (42.1%), drawer=6/8 (75.0%), window=5/14 (35.7%)
- `fast_lewam` / `P3/not_applicable/post_opt_refine/step_32/euler`：button=4/9 (44.4%), cube=8/19 (42.1%), drawer=6/8 (75.0%), window=5/14 (35.7%)
- `lewm` / `standard`：button=3/9 (33.3%), cube=8/19 (42.1%), drawer=6/8 (75.0%), window=6/14 (42.9%)
- `leflow` / `standard`：button=3/9 (33.3%), cube=7/19 (36.8%), drawer=5/8 (62.5%), window=5/14 (35.7%)

## 产物

- 输出根目录：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3`
- 条件明细：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/analysis/conditions.csv`
- 分析 JSON：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/analysis/analysis.json`
- 代表视频：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase3/videos`（每任务 FastLeWAM P1/P3 step 1、LeWM、LeFlow 各一段）
- 本报告：`/data/users/wenxin/pre-exp/le-wm/docs/report/round5/round5_phase3_report.md`
