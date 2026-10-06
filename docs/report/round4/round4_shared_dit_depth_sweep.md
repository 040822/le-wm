# Shared DiT 深度实验报告：6 / 8 / 12 层

## 结论

本报告汇总 Cube、PushT 和 Reacher 上 Shared DiT 6、8、12 层的评测结果，包含 `round3_revised` 的 dev/final cohort，以及 canonical legacy 50-episode cohort。各深度在同一任务和 cohort 内使用相同 episode 列表进行配对比较；不同 cohort 的结果分开报告，不合并样本。

Cube 在 `round3_revised` 的 dev 和 final cohort 上三个深度均为 100%。Reacher 在 `round3_revised` 上，8/12 层相对 6 层的成功率略低；legacy cohort 上各深度结果有小幅起伏。PushT legacy cohort 上 12 层达到 100%，比 6 层高 4 个百分点，但配对检验不显著。所有配对检验均未达到统计显著水平，当前数据没有表明加深模型能稳定提升成功率。

规划耗时在 `round3_revised` 评测中随深度上升；legacy 评测中的耗时未呈现一致趋势，且受到 GPU 共享和设备差异影响，不能据此作严格的深度性能比较。每个任务和深度只有一个训练 seed，因此结果适合用于探索方向，仍需多个 seed 验证。

## 实验设置

- 模型深度为 6、8、12 层。6 层使用已有 R4-AB、seed 3072、epoch 10 checkpoint；8/12 层按相同训练配置训练 10 epochs，深度实验只改变 `policy.model.latent_head_layers`，run 名称和输出路径随之变化。Reacher legacy 评测复用本报告中的 Reacher checkpoint；PushT 8/12 层为本轮新增训练。
- 除模型深度外，各训练配置经过审计确认一致。每个任务和深度使用一个训练 seed（3072）。
- 两套评测 cohort 独立呈现：`round3_revised` 包含 Cube 与 Reacher 的 dev/final；canonical `legacy_50` 包含 PushT 与 Reacher。Reacher 在两套 cohort 上评测相同深度模型，但 cohort 与 runner 不同。
- `round3_revised` 使用 Round4-AB P3：evaluation seed 42、64 个候选、16 个 flow steps、solver batch size 50、Stage-B 选取。统一重跑的结果由代码 commit `93eb1792af10947eee340e74cd5ccd7a7e525bd1` 生成。
- `legacy_50` 使用 Phase 4.5-4 legacy runner 的 P3 step 16：evaluation seed 42、64 个候选、solver batch size 1、candidate batch size 64。六项结果均为 `ok`，由同一代码 commit `93eb1792af10947eee340e74cd5ccd7a7e525bd1` 生成。
- 8 层 Cube/Reacher 训练分别使用 GPU2/GPU3，12 层 Cube/Reacher 分别使用 GPU0/GPU1；PushT 8/12 层分别使用 GPU2/GPU3。

## Cohort 与成功率

| 任务 | Cohort | Episodes | 6 层 | 8 层 | 12 层 |
|---|---|---:|---:|---:|---:|
| Cube | `round3_revised` dev | 50 | 50/50 (100%) | 50/50 (100%) | 50/50 (100%) |
| Cube | `round3_revised` final | 200 | 200/200 (100%) | 200/200 (100%) | 200/200 (100%) |
| Reacher | `round3_revised` dev | 50 | 43/50 (86%) | 43/50 (86%) | 36/50 (72%) |
| Reacher | `round3_revised` final | 200 | 169/200 (84.5%) | 161/200 (80.5%) | 163/200 (81.5%) |
| PushT | `legacy_50` | 50 | 48/50 (96%) | 48/50 (96%) | 50/50 (100%) |
| Reacher | `legacy_50` | 50 | 39/50 (78%) | 37/50 (74%) | 40/50 (80%) |

各深度在每个 cohort 内逐 episode 配对比较。下表中的“改善 / 退化”表示相对 6 层的成功状态变化；净变化为成功率百分点差，p 值为双侧 exact McNemar 检验。

| 任务 | Cohort | 深度对比 | 改善 / 退化 | 净变化 | p 值 |
|---|---|---:|---:|---:|---:|
| Cube | `round3_revised` dev | 8 vs 6 | 0 / 0 | 0 pp | 1.000 |
| Cube | `round3_revised` dev | 12 vs 6 | 0 / 0 | 0 pp | 1.000 |
| Cube | `round3_revised` final | 8 vs 6 | 0 / 0 | 0 pp | 1.000 |
| Cube | `round3_revised` final | 12 vs 6 | 0 / 0 | 0 pp | 1.000 |
| Reacher | `round3_revised` dev | 8 vs 6 | 4 / 4 | 0 pp | 1.000 |
| Reacher | `round3_revised` dev | 12 vs 6 | 2 / 9 | -14 pp | 0.065 |
| Reacher | `round3_revised` final | 8 vs 6 | 21 / 29 | -4 pp | 0.322 |
| Reacher | `round3_revised` final | 12 vs 6 | 21 / 27 | -3 pp | 0.471 |
| PushT | `legacy_50` | 8 vs 6 | 1 / 1 | 0 pp | 1.000 |
| PushT | `legacy_50` | 12 vs 6 | 2 / 0 | +4 pp | 0.500 |
| Reacher | `legacy_50` | 8 vs 6 | 5 / 7 | -4 pp | 0.774 |
| Reacher | `legacy_50` | 12 vs 6 | 9 / 8 | +2 pp | 1.000 |

Cube 的两个 cohort 上所有 episode 在三个深度下均成功，因此没有配对状态变化。Reacher `round3_revised` 的 95% Wilson 区间较宽且重叠：dev 中 6/8 层均为 73.8–93.0%，12 层为 58.3–82.5%；final 中 6、8、12 层分别为 78.8–88.9%、74.5–85.4%、75.5–86.3%。PushT legacy 12 层比 6 层多成功 2 个 episode；在 50 个 episode 的样本上，差异未达到统计显著。整体上没有稳定、跨 cohort 的随深度改善趋势。

各 cohort 的 SHA256：

| 任务 | Cohort | SHA256 |
|---|---|---|
| Cube | `round3_revised` dev | `578f6ea52e56d77461cb5bb2aeb01a004a21b9f6659689596f6f55f25dc17d01` |
| Cube | `round3_revised` final | `26171642a8c6a3617b0c586c2d0f448f9d352aa433d6f8a169b8fffdecb0e030` |
| Reacher | `round3_revised` dev | `d18695b2377000ce7dd774fcad78e4bb2d75980fb3ccbd98cc83d09ef50f1d06` |
| Reacher | `round3_revised` final | `f81e4830dd2bb80ad1a8109b67873496fa86651ccba63d49ab6d8b0c78b87490` |
| PushT | `legacy_50` | `b110bee6afa2f7be372d57b3ef1247a44048c8b1e336491d74a53401ad479b0c` |
| Reacher | `legacy_50` | `ff4f26ad3fd809e59e1b748c7c93c33a8e51b48e13d2505e89e5909d6121bf45` |

## P3 规划耗时

以下为评测记录中的每 episode 规划中位耗时，单位为秒。`round3_revised` 与 `legacy_50` 使用不同 runner 和 solver batch 设置，分开列出，不直接跨协议比较。运行期间 GPU 与其他任务共享；legacy PushT 的 6 层评测设备也与 8/12 层不同，因此这些数据只用于描述本次观测，不作为隔离条件下的吞吐基准。

| 任务 | Cohort | 6 层 | 8 层 | 12 层 |
|---|---|---:|---:|---:|
| Cube | `round3_revised` dev | 0.511 | 0.589 | 0.685 |
| Cube | `round3_revised` final | 1.212 | 1.497 | 2.141 |
| Reacher | `round3_revised` dev | 0.362 | 0.466 | 0.551 |
| Reacher | `round3_revised` final | 0.860 | 1.201 | 1.594 |
| PushT | `legacy_50` | 0.561 | 0.404 | 0.929 |
| Reacher | `legacy_50` | 0.871 | 0.751 | 0.728 |

`round3_revised` 上各任务和 cohort 的观测耗时均随深度增加，12 层相对 6 层增加约 34–85%；legacy 结果没有同样的单调变化，符合其运行条件存在噪声的情况。

## Checkpoint 与评测产物

训练 epoch-10 checkpoint SHA256：

| 深度 | 任务 | Checkpoint | SHA256 |
|---:|---|---|---|
| 6 | Cube | `outputs/round4/ab_seed3072_cube/checkpoints/r4_ab_seed3072_weights_epoch_10.pt` | `2748151a2c43c7841246a5fb8ee1fd7915ebbc00bfc27aa1a32b0f9080d92956` |
| 6 | Reacher | `outputs/round4/ab_seed3072_reacher/checkpoints/r4_ab_seed3072_weights_epoch_10.pt` | `087991339c9499d10c1a58a28e72d07c7dfd819d5f072c67bfd0169acaa83553` |
| 6 | PushT | `outputs/round4/ab_seed3072_pusht/checkpoints/r4_ab_seed3072_weights_epoch_10.pt` | `62d00096a34e9f0b6da4ceb6a3c61eb350701a7aade5f5e4d2c1ee528b06015e` |
| 8 | Cube | `outputs/round4_shared_dit_depth_d8_cube_seed3072/checkpoints/r4_ab_d8_seed3072_cube_weights_epoch_10.pt` | `506a83245fe9830c86d7a68f1c8d152e640a69418f47040a007c965704c4a1f9` |
| 8 | Reacher | `outputs/round4_shared_dit_depth_d8_reacher_seed3072/checkpoints/r4_ab_d8_seed3072_reacher_weights_epoch_10.pt` | `dc4b7b5fc429a9b0116b275b87ac1adc050526596590f45f9cefc7924c95f8b6` |
| 8 | PushT | `outputs/round4_shared_dit_depth_d8_pusht_seed3072/checkpoints/r4_ab_d8_seed3072_pusht_weights_epoch_10.pt` | `07f8cafd40560ab0bfaa4659058e7a7aa848358f95bd5b3b7f1bd4ac7210ceb7` |
| 12 | Cube | `outputs/round4_shared_dit_depth_d12_cube_seed3072/checkpoints/r4_ab_d12_seed3072_cube_weights_epoch_10.pt` | `5e1779877cfe8ef321a596bbea48eea13ec7aa75340526543940552df587d83d` |
| 12 | Reacher | `outputs/round4_shared_dit_depth_d12_reacher_seed3072/checkpoints/r4_ab_d12_seed3072_reacher_weights_epoch_10.pt` | `b0142b6027b014630a15ff0955fda661de540f27212209a5320bd3816033cfaf` |
| 12 | PushT | `outputs/round4_shared_dit_depth_d12_pusht_seed3072/checkpoints/r4_ab_d12_seed3072_pusht_weights_epoch_10.pt` | `77fded174fef263bc21879800cf585c5b6a3a0cb5ade03f710844e3f48db9d17` |

评测 JSON、逐 episode trace 与日志位置：

- `round3_revised`：`outputs/round4_shared_dit_depth_sweep_eval/`；统一重跑的 8/12 层结果位于 `matched/d8/`、`matched/d12/`，6 层基线位于 `d6/`。
- `legacy_50`：`outputs/round4_shared_dit_depth_sweep_legacy_phase45_4/d6/`、`d8/`、`d12/`，各目录包含 PushT 与 Reacher 的结果。

两套评测结果均完成状态、代码 commit、评测参数及 cohort 对齐校验；`round3_revised` 的 cohort hash、代码 commit、候选数、flow steps 和 solver batch size 在各深度间一致，legacy cohort 的 episode IDs 也在各深度间对齐。
