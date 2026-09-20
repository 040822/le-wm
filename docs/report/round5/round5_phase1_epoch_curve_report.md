# Round 5 Phase 1：中间 epoch 推理曲线

本实验固定 R4-AB seed 3072、legacy_50 cohort 和 Phase 1 推理协议，比较 epoch 1–10 的成功率曲线。epoch 1–9 为新评测；epoch 10 只引用已经完成并校验的 Phase 4.5-4 legacy baseline，不重跑、不复制旧结果。

- 配置：`/data/users/wenxin/pre-exp/le-wm/config/round5/phase1_epoch_curve.json`
- 输出根目录：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase1_epoch_curve_seed3072_legacy`
- 条件点：`340/340`；新增评测 `306`，epoch10 复用 `34`。
- status=ok：`True`；episodes/point：`[50]`。

## 评测矩阵

| task | conditions/epoch | 条件定义 |
|---|---:|---|
| cube | 4 | P0/1/not_applicable, P1/inv/legacy, P2/1/legacy, P3/1/not_applicable |
| pusht | 7 | P0/1/not_applicable, P0/2/not_applicable, P1/inv/legacy, P2/1/legacy, P2/2/legacy, P3/1/not_applicable, P3/2/not_applicable |
| reacher | 19 | P0/1/not_applicable, P0/2/not_applicable, P0/5/not_applicable, P0/10/not_applicable, P0/16/not_applicable, P0/32/not_applicable, P1/inv/cem-clip, P2/1/cem-clip, P2/2/cem-clip, P2/5/cem-clip, P2/10/cem-clip, P2/16/cem-clip, P2/32/cem-clip, P3/1/not_applicable, P3/2/not_applicable, P3/5/not_applicable, P3/10/not_applicable, P3/16/not_applicable, P3/32/not_applicable |
| tworoom | 4 | P0/1/not_applicable, P1/inv/legacy, P2/1/legacy, P3/1/not_applicable |

## 验收身份

所有点均要求 checkpoint 路径、checkpoint SHA256、cohort id/SHA256、epoch、50 个 episode 的 `(episode_id, start_step, row_index)` 序列和条件 key 一致。

| task | cohort | cohort SHA256 | epoch checkpoint SHA256 |
|---|---|---|---|
| cube | `cube_legacy_50_v1` | `b9054ebb327665142befac1b3ec36b2500de5dcdadc4008da322e2d0e4c721bb` | epoch10 `2748151a2c43c7841246a5fb8ee1fd7915ebbc00bfc27aa1a32b0f9080d92956` |
| pusht | `pusht_legacy_50_v1` | `b110bee6afa2f7be372d57b3ef1247a44048c8b1e336491d74a53401ad479b0c` | epoch10 `62d00096a34e9f0b6da4ceb6a3c61eb350701a7aade5f5e4d2c1ee528b06015e` |
| reacher | `reacher_legacy_50_v1` | `ff4f26ad3fd809e59e1b748c7c93c33a8e51b48e13d2505e89e5909d6121bf45` | epoch10 `087991339c9499d10c1a58a28e72d07c7dfd819d5f072c67bfd0169acaa83553` |
| tworoom | `tworoom_legacy_50_v1` | `d452befec00361add056c880e09e9ef2cb68060e7ad599535d8ce83a875e9091` | epoch10 `48d1ed2520ec7142327969eacf1c37b144faec51cfd6c9ce72701a8421bf57c2` |

## 成功率数据

结果已写入 `epoch_curve.csv` 和 `epoch_curve.json`。下表列出每个曲线点：

| task | epoch | mode | protocol | step | success | Wilson 95% | source |
|---|---:|---|---|---:|---|---|---|
| cube | 1 | P0 | not_applicable | 1 | 41/50 (82.0%) | [69.2, 90.2]% | evaluated |
| cube | 1 | P1 | legacy | inv | 27/50 (54.0%) | [40.4, 67.0]% | evaluated |
| cube | 1 | P2 | legacy | 1 | 37/50 (74.0%) | [60.4, 84.1]% | evaluated |
| cube | 1 | P3 | not_applicable | 1 | 42/50 (84.0%) | [71.5, 91.7]% | evaluated |
| cube | 2 | P0 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| cube | 2 | P1 | legacy | inv | 35/50 (70.0%) | [56.2, 80.9]% | evaluated |
| cube | 2 | P2 | legacy | 1 | 49/50 (98.0%) | [89.5, 99.6]% | evaluated |
| cube | 2 | P3 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| cube | 3 | P0 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| cube | 3 | P1 | legacy | inv | 36/50 (72.0%) | [58.3, 82.5]% | evaluated |
| cube | 3 | P2 | legacy | 1 | 47/50 (94.0%) | [83.8, 97.9]% | evaluated |
| cube | 3 | P3 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| cube | 4 | P0 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| cube | 4 | P1 | legacy | inv | 37/50 (74.0%) | [60.4, 84.1]% | evaluated |
| cube | 4 | P2 | legacy | 1 | 48/50 (96.0%) | [86.5, 98.9]% | evaluated |
| cube | 4 | P3 | not_applicable | 1 | 49/50 (98.0%) | [89.5, 99.6]% | evaluated |
| cube | 5 | P0 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| cube | 5 | P1 | legacy | inv | 33/50 (66.0%) | [52.2, 77.6]% | evaluated |
| cube | 5 | P2 | legacy | 1 | 48/50 (96.0%) | [86.5, 98.9]% | evaluated |
| cube | 5 | P3 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| cube | 6 | P0 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| cube | 6 | P1 | legacy | inv | 33/50 (66.0%) | [52.2, 77.6]% | evaluated |
| cube | 6 | P2 | legacy | 1 | 47/50 (94.0%) | [83.8, 97.9]% | evaluated |
| cube | 6 | P3 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| cube | 7 | P0 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| cube | 7 | P1 | legacy | inv | 32/50 (64.0%) | [50.1, 75.9]% | evaluated |
| cube | 7 | P2 | legacy | 1 | 49/50 (98.0%) | [89.5, 99.6]% | evaluated |
| cube | 7 | P3 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| cube | 8 | P0 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| cube | 8 | P1 | legacy | inv | 34/50 (68.0%) | [54.2, 79.2]% | evaluated |
| cube | 8 | P2 | legacy | 1 | 48/50 (96.0%) | [86.5, 98.9]% | evaluated |
| cube | 8 | P3 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| cube | 9 | P0 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| cube | 9 | P1 | legacy | inv | 33/50 (66.0%) | [52.2, 77.6]% | evaluated |
| cube | 9 | P2 | legacy | 1 | 49/50 (98.0%) | [89.5, 99.6]% | evaluated |
| cube | 9 | P3 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| cube | 10 | P0 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | epoch10_reuse |
| cube | 10 | P1 | legacy | inv | 35/50 (70.0%) | [56.2, 80.9]% | epoch10_reuse |
| cube | 10 | P2 | legacy | 1 | 49/50 (98.0%) | [89.5, 99.6]% | epoch10_reuse |
| cube | 10 | P3 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | epoch10_reuse |
| pusht | 1 | P0 | not_applicable | 1 | 12/50 (24.0%) | [14.3, 37.4]% | evaluated |
| pusht | 1 | P0 | not_applicable | 2 | 16/50 (32.0%) | [20.8, 45.8]% | evaluated |
| pusht | 1 | P1 | legacy | inv | 6/50 (12.0%) | [5.6, 23.8]% | evaluated |
| pusht | 1 | P2 | legacy | 1 | 9/50 (18.0%) | [9.8, 30.8]% | evaluated |
| pusht | 1 | P2 | legacy | 2 | 11/50 (22.0%) | [12.8, 35.2]% | evaluated |
| pusht | 1 | P3 | not_applicable | 1 | 10/50 (20.0%) | [11.2, 33.0]% | evaluated |
| pusht | 1 | P3 | not_applicable | 2 | 15/50 (30.0%) | [19.1, 43.8]% | evaluated |
| pusht | 2 | P0 | not_applicable | 1 | 39/50 (78.0%) | [64.8, 87.2]% | evaluated |
| pusht | 2 | P0 | not_applicable | 2 | 38/50 (76.0%) | [62.6, 85.7]% | evaluated |
| pusht | 2 | P1 | legacy | inv | 36/50 (72.0%) | [58.3, 82.5]% | evaluated |
| pusht | 2 | P2 | legacy | 1 | 38/50 (76.0%) | [62.6, 85.7]% | evaluated |
| pusht | 2 | P2 | legacy | 2 | 39/50 (78.0%) | [64.8, 87.2]% | evaluated |
| pusht | 2 | P3 | not_applicable | 1 | 41/50 (82.0%) | [69.2, 90.2]% | evaluated |
| pusht | 2 | P3 | not_applicable | 2 | 42/50 (84.0%) | [71.5, 91.7]% | evaluated |
| pusht | 3 | P0 | not_applicable | 1 | 40/50 (80.0%) | [67.0, 88.8]% | evaluated |
| pusht | 3 | P0 | not_applicable | 2 | 44/50 (88.0%) | [76.2, 94.4]% | evaluated |
| pusht | 3 | P1 | legacy | inv | 35/50 (70.0%) | [56.2, 80.9]% | evaluated |
| pusht | 3 | P2 | legacy | 1 | 32/50 (64.0%) | [50.1, 75.9]% | evaluated |
| pusht | 3 | P2 | legacy | 2 | 31/50 (62.0%) | [48.2, 74.1]% | evaluated |
| pusht | 3 | P3 | not_applicable | 1 | 40/50 (80.0%) | [67.0, 88.8]% | evaluated |
| pusht | 3 | P3 | not_applicable | 2 | 41/50 (82.0%) | [69.2, 90.2]% | evaluated |
| pusht | 4 | P0 | not_applicable | 1 | 42/50 (84.0%) | [71.5, 91.7]% | evaluated |
| pusht | 4 | P0 | not_applicable | 2 | 46/50 (92.0%) | [81.2, 96.8]% | evaluated |
| pusht | 4 | P1 | legacy | inv | 36/50 (72.0%) | [58.3, 82.5]% | evaluated |
| pusht | 4 | P2 | legacy | 1 | 41/50 (82.0%) | [69.2, 90.2]% | evaluated |
| pusht | 4 | P2 | legacy | 2 | 41/50 (82.0%) | [69.2, 90.2]% | evaluated |
| pusht | 4 | P3 | not_applicable | 1 | 46/50 (92.0%) | [81.2, 96.8]% | evaluated |
| pusht | 4 | P3 | not_applicable | 2 | 48/50 (96.0%) | [86.5, 98.9]% | evaluated |
| pusht | 5 | P0 | not_applicable | 1 | 46/50 (92.0%) | [81.2, 96.8]% | evaluated |
| pusht | 5 | P0 | not_applicable | 2 | 44/50 (88.0%) | [76.2, 94.4]% | evaluated |
| pusht | 5 | P1 | legacy | inv | 38/50 (76.0%) | [62.6, 85.7]% | evaluated |
| pusht | 5 | P2 | legacy | 1 | 42/50 (84.0%) | [71.5, 91.7]% | evaluated |
| pusht | 5 | P2 | legacy | 2 | 45/50 (90.0%) | [78.6, 95.7]% | evaluated |
| pusht | 5 | P3 | not_applicable | 1 | 46/50 (92.0%) | [81.2, 96.8]% | evaluated |
| pusht | 5 | P3 | not_applicable | 2 | 49/50 (98.0%) | [89.5, 99.6]% | evaluated |
| pusht | 6 | P0 | not_applicable | 1 | 47/50 (94.0%) | [83.8, 97.9]% | evaluated |
| pusht | 6 | P0 | not_applicable | 2 | 47/50 (94.0%) | [83.8, 97.9]% | evaluated |
| pusht | 6 | P1 | legacy | inv | 44/50 (88.0%) | [76.2, 94.4]% | evaluated |
| pusht | 6 | P2 | legacy | 1 | 46/50 (92.0%) | [81.2, 96.8]% | evaluated |
| pusht | 6 | P2 | legacy | 2 | 45/50 (90.0%) | [78.6, 95.7]% | evaluated |
| pusht | 6 | P3 | not_applicable | 1 | 47/50 (94.0%) | [83.8, 97.9]% | evaluated |
| pusht | 6 | P3 | not_applicable | 2 | 49/50 (98.0%) | [89.5, 99.6]% | evaluated |
| pusht | 7 | P0 | not_applicable | 1 | 49/50 (98.0%) | [89.5, 99.6]% | evaluated |
| pusht | 7 | P0 | not_applicable | 2 | 48/50 (96.0%) | [86.5, 98.9]% | evaluated |
| pusht | 7 | P1 | legacy | inv | 45/50 (90.0%) | [78.6, 95.7]% | evaluated |
| pusht | 7 | P2 | legacy | 1 | 46/50 (92.0%) | [81.2, 96.8]% | evaluated |
| pusht | 7 | P2 | legacy | 2 | 46/50 (92.0%) | [81.2, 96.8]% | evaluated |
| pusht | 7 | P3 | not_applicable | 1 | 47/50 (94.0%) | [83.8, 97.9]% | evaluated |
| pusht | 7 | P3 | not_applicable | 2 | 48/50 (96.0%) | [86.5, 98.9]% | evaluated |
| pusht | 8 | P0 | not_applicable | 1 | 46/50 (92.0%) | [81.2, 96.8]% | evaluated |
| pusht | 8 | P0 | not_applicable | 2 | 47/50 (94.0%) | [83.8, 97.9]% | evaluated |
| pusht | 8 | P1 | legacy | inv | 46/50 (92.0%) | [81.2, 96.8]% | evaluated |
| pusht | 8 | P2 | legacy | 1 | 46/50 (92.0%) | [81.2, 96.8]% | evaluated |
| pusht | 8 | P2 | legacy | 2 | 47/50 (94.0%) | [83.8, 97.9]% | evaluated |
| pusht | 8 | P3 | not_applicable | 1 | 46/50 (92.0%) | [81.2, 96.8]% | evaluated |
| pusht | 8 | P3 | not_applicable | 2 | 49/50 (98.0%) | [89.5, 99.6]% | evaluated |
| pusht | 9 | P0 | not_applicable | 1 | 47/50 (94.0%) | [83.8, 97.9]% | evaluated |
| pusht | 9 | P0 | not_applicable | 2 | 49/50 (98.0%) | [89.5, 99.6]% | evaluated |
| pusht | 9 | P1 | legacy | inv | 43/50 (86.0%) | [73.8, 93.0]% | evaluated |
| pusht | 9 | P2 | legacy | 1 | 49/50 (98.0%) | [89.5, 99.6]% | evaluated |
| pusht | 9 | P2 | legacy | 2 | 48/50 (96.0%) | [86.5, 98.9]% | evaluated |
| pusht | 9 | P3 | not_applicable | 1 | 46/50 (92.0%) | [81.2, 96.8]% | evaluated |
| pusht | 9 | P3 | not_applicable | 2 | 49/50 (98.0%) | [89.5, 99.6]% | evaluated |
| pusht | 10 | P0 | not_applicable | 1 | 48/50 (96.0%) | [86.5, 98.9]% | epoch10_reuse |
| pusht | 10 | P0 | not_applicable | 2 | 50/50 (100.0%) | [92.9, 100.0]% | epoch10_reuse |
| pusht | 10 | P1 | legacy | inv | 45/50 (90.0%) | [78.6, 95.7]% | epoch10_reuse |
| pusht | 10 | P2 | legacy | 1 | 49/50 (98.0%) | [89.5, 99.6]% | epoch10_reuse |
| pusht | 10 | P2 | legacy | 2 | 48/50 (96.0%) | [86.5, 98.9]% | epoch10_reuse |
| pusht | 10 | P3 | not_applicable | 1 | 48/50 (96.0%) | [86.5, 98.9]% | epoch10_reuse |
| pusht | 10 | P3 | not_applicable | 2 | 49/50 (98.0%) | [89.5, 99.6]% | epoch10_reuse |
| reacher | 1 | P0 | not_applicable | 1 | 1/50 (2.0%) | [0.4, 10.5]% | evaluated |
| reacher | 1 | P0 | not_applicable | 2 | 5/50 (10.0%) | [4.3, 21.4]% | evaluated |
| reacher | 1 | P0 | not_applicable | 5 | 8/50 (16.0%) | [8.3, 28.5]% | evaluated |
| reacher | 1 | P0 | not_applicable | 10 | 10/50 (20.0%) | [11.2, 33.0]% | evaluated |
| reacher | 1 | P0 | not_applicable | 16 | 10/50 (20.0%) | [11.2, 33.0]% | evaluated |
| reacher | 1 | P0 | not_applicable | 32 | 10/50 (20.0%) | [11.2, 33.0]% | evaluated |
| reacher | 1 | P1 | cem-clip | inv | 7/50 (14.0%) | [7.0, 26.2]% | evaluated |
| reacher | 1 | P2 | cem-clip | 1 | 6/50 (12.0%) | [5.6, 23.8]% | evaluated |
| reacher | 1 | P2 | cem-clip | 2 | 5/50 (10.0%) | [4.3, 21.4]% | evaluated |
| reacher | 1 | P2 | cem-clip | 5 | 6/50 (12.0%) | [5.6, 23.8]% | evaluated |
| reacher | 1 | P2 | cem-clip | 10 | 10/50 (20.0%) | [11.2, 33.0]% | evaluated |
| reacher | 1 | P2 | cem-clip | 16 | 5/50 (10.0%) | [4.3, 21.4]% | evaluated |
| reacher | 1 | P2 | cem-clip | 32 | 8/50 (16.0%) | [8.3, 28.5]% | evaluated |
| reacher | 1 | P3 | not_applicable | 1 | 5/50 (10.0%) | [4.3, 21.4]% | evaluated |
| reacher | 1 | P3 | not_applicable | 2 | 3/50 (6.0%) | [2.1, 16.2]% | evaluated |
| reacher | 1 | P3 | not_applicable | 5 | 6/50 (12.0%) | [5.6, 23.8]% | evaluated |
| reacher | 1 | P3 | not_applicable | 10 | 9/50 (18.0%) | [9.8, 30.8]% | evaluated |
| reacher | 1 | P3 | not_applicable | 16 | 7/50 (14.0%) | [7.0, 26.2]% | evaluated |
| reacher | 1 | P3 | not_applicable | 32 | 6/50 (12.0%) | [5.6, 23.8]% | evaluated |
| reacher | 2 | P0 | not_applicable | 1 | 7/50 (14.0%) | [7.0, 26.2]% | evaluated |
| reacher | 2 | P0 | not_applicable | 2 | 9/50 (18.0%) | [9.8, 30.8]% | evaluated |
| reacher | 2 | P0 | not_applicable | 5 | 14/50 (28.0%) | [17.5, 41.7]% | evaluated |
| reacher | 2 | P0 | not_applicable | 10 | 15/50 (30.0%) | [19.1, 43.8]% | evaluated |
| reacher | 2 | P0 | not_applicable | 16 | 12/50 (24.0%) | [14.3, 37.4]% | evaluated |
| reacher | 2 | P0 | not_applicable | 32 | 12/50 (24.0%) | [14.3, 37.4]% | evaluated |
| reacher | 2 | P1 | cem-clip | inv | 10/50 (20.0%) | [11.2, 33.0]% | evaluated |
| reacher | 2 | P2 | cem-clip | 1 | 10/50 (20.0%) | [11.2, 33.0]% | evaluated |
| reacher | 2 | P2 | cem-clip | 2 | 14/50 (28.0%) | [17.5, 41.7]% | evaluated |
| reacher | 2 | P2 | cem-clip | 5 | 11/50 (22.0%) | [12.8, 35.2]% | evaluated |
| reacher | 2 | P2 | cem-clip | 10 | 10/50 (20.0%) | [11.2, 33.0]% | evaluated |
| reacher | 2 | P2 | cem-clip | 16 | 13/50 (26.0%) | [15.9, 39.6]% | evaluated |
| reacher | 2 | P2 | cem-clip | 32 | 12/50 (24.0%) | [14.3, 37.4]% | evaluated |
| reacher | 2 | P3 | not_applicable | 1 | 6/50 (12.0%) | [5.6, 23.8]% | evaluated |
| reacher | 2 | P3 | not_applicable | 2 | 13/50 (26.0%) | [15.9, 39.6]% | evaluated |
| reacher | 2 | P3 | not_applicable | 5 | 16/50 (32.0%) | [20.8, 45.8]% | evaluated |
| reacher | 2 | P3 | not_applicable | 10 | 15/50 (30.0%) | [19.1, 43.8]% | evaluated |
| reacher | 2 | P3 | not_applicable | 16 | 15/50 (30.0%) | [19.1, 43.8]% | evaluated |
| reacher | 2 | P3 | not_applicable | 32 | 15/50 (30.0%) | [19.1, 43.8]% | evaluated |
| reacher | 3 | P0 | not_applicable | 1 | 15/50 (30.0%) | [19.1, 43.8]% | evaluated |
| reacher | 3 | P0 | not_applicable | 2 | 22/50 (44.0%) | [31.2, 57.7]% | evaluated |
| reacher | 3 | P0 | not_applicable | 5 | 16/50 (32.0%) | [20.8, 45.8]% | evaluated |
| reacher | 3 | P0 | not_applicable | 10 | 14/50 (28.0%) | [17.5, 41.7]% | evaluated |
| reacher | 3 | P0 | not_applicable | 16 | 13/50 (26.0%) | [15.9, 39.6]% | evaluated |
| reacher | 3 | P0 | not_applicable | 32 | 12/50 (24.0%) | [14.3, 37.4]% | evaluated |
| reacher | 3 | P1 | cem-clip | inv | 33/50 (66.0%) | [52.2, 77.6]% | evaluated |
| reacher | 3 | P2 | cem-clip | 1 | 26/50 (52.0%) | [38.5, 65.2]% | evaluated |
| reacher | 3 | P2 | cem-clip | 2 | 33/50 (66.0%) | [52.2, 77.6]% | evaluated |
| reacher | 3 | P2 | cem-clip | 5 | 27/50 (54.0%) | [40.4, 67.0]% | evaluated |
| reacher | 3 | P2 | cem-clip | 10 | 28/50 (56.0%) | [42.3, 68.8]% | evaluated |
| reacher | 3 | P2 | cem-clip | 16 | 25/50 (50.0%) | [36.6, 63.4]% | evaluated |
| reacher | 3 | P2 | cem-clip | 32 | 24/50 (48.0%) | [34.8, 61.5]% | evaluated |
| reacher | 3 | P3 | not_applicable | 1 | 19/50 (38.0%) | [25.9, 51.8]% | evaluated |
| reacher | 3 | P3 | not_applicable | 2 | 28/50 (56.0%) | [42.3, 68.8]% | evaluated |
| reacher | 3 | P3 | not_applicable | 5 | 25/50 (50.0%) | [36.6, 63.4]% | evaluated |
| reacher | 3 | P3 | not_applicable | 10 | 28/50 (56.0%) | [42.3, 68.8]% | evaluated |
| reacher | 3 | P3 | not_applicable | 16 | 26/50 (52.0%) | [38.5, 65.2]% | evaluated |
| reacher | 3 | P3 | not_applicable | 32 | 26/50 (52.0%) | [38.5, 65.2]% | evaluated |
| reacher | 4 | P0 | not_applicable | 1 | 26/50 (52.0%) | [38.5, 65.2]% | evaluated |
| reacher | 4 | P0 | not_applicable | 2 | 33/50 (66.0%) | [52.2, 77.6]% | evaluated |
| reacher | 4 | P0 | not_applicable | 5 | 32/50 (64.0%) | [50.1, 75.9]% | evaluated |
| reacher | 4 | P0 | not_applicable | 10 | 32/50 (64.0%) | [50.1, 75.9]% | evaluated |
| reacher | 4 | P0 | not_applicable | 16 | 24/50 (48.0%) | [34.8, 61.5]% | evaluated |
| reacher | 4 | P0 | not_applicable | 32 | 21/50 (42.0%) | [29.4, 55.8]% | evaluated |
| reacher | 4 | P1 | cem-clip | inv | 35/50 (70.0%) | [56.2, 80.9]% | evaluated |
| reacher | 4 | P2 | cem-clip | 1 | 36/50 (72.0%) | [58.3, 82.5]% | evaluated |
| reacher | 4 | P2 | cem-clip | 2 | 37/50 (74.0%) | [60.4, 84.1]% | evaluated |
| reacher | 4 | P2 | cem-clip | 5 | 38/50 (76.0%) | [62.6, 85.7]% | evaluated |
| reacher | 4 | P2 | cem-clip | 10 | 37/50 (74.0%) | [60.4, 84.1]% | evaluated |
| reacher | 4 | P2 | cem-clip | 16 | 38/50 (76.0%) | [62.6, 85.7]% | evaluated |
| reacher | 4 | P2 | cem-clip | 32 | 41/50 (82.0%) | [69.2, 90.2]% | evaluated |
| reacher | 4 | P3 | not_applicable | 1 | 33/50 (66.0%) | [52.2, 77.6]% | evaluated |
| reacher | 4 | P3 | not_applicable | 2 | 39/50 (78.0%) | [64.8, 87.2]% | evaluated |
| reacher | 4 | P3 | not_applicable | 5 | 39/50 (78.0%) | [64.8, 87.2]% | evaluated |
| reacher | 4 | P3 | not_applicable | 10 | 36/50 (72.0%) | [58.3, 82.5]% | evaluated |
| reacher | 4 | P3 | not_applicable | 16 | 35/50 (70.0%) | [56.2, 80.9]% | evaluated |
| reacher | 4 | P3 | not_applicable | 32 | 38/50 (76.0%) | [62.6, 85.7]% | evaluated |
| reacher | 5 | P0 | not_applicable | 1 | 23/50 (46.0%) | [33.0, 59.6]% | evaluated |
| reacher | 5 | P0 | not_applicable | 2 | 29/50 (58.0%) | [44.2, 70.6]% | evaluated |
| reacher | 5 | P0 | not_applicable | 5 | 27/50 (54.0%) | [40.4, 67.0]% | evaluated |
| reacher | 5 | P0 | not_applicable | 10 | 25/50 (50.0%) | [36.6, 63.4]% | evaluated |
| reacher | 5 | P0 | not_applicable | 16 | 24/50 (48.0%) | [34.8, 61.5]% | evaluated |
| reacher | 5 | P0 | not_applicable | 32 | 23/50 (46.0%) | [33.0, 59.6]% | evaluated |
| reacher | 5 | P1 | cem-clip | inv | 39/50 (78.0%) | [64.8, 87.2]% | evaluated |
| reacher | 5 | P2 | cem-clip | 1 | 39/50 (78.0%) | [64.8, 87.2]% | evaluated |
| reacher | 5 | P2 | cem-clip | 2 | 36/50 (72.0%) | [58.3, 82.5]% | evaluated |
| reacher | 5 | P2 | cem-clip | 5 | 36/50 (72.0%) | [58.3, 82.5]% | evaluated |
| reacher | 5 | P2 | cem-clip | 10 | 37/50 (74.0%) | [60.4, 84.1]% | evaluated |
| reacher | 5 | P2 | cem-clip | 16 | 35/50 (70.0%) | [56.2, 80.9]% | evaluated |
| reacher | 5 | P2 | cem-clip | 32 | 34/50 (68.0%) | [54.2, 79.2]% | evaluated |
| reacher | 5 | P3 | not_applicable | 1 | 28/50 (56.0%) | [42.3, 68.8]% | evaluated |
| reacher | 5 | P3 | not_applicable | 2 | 37/50 (74.0%) | [60.4, 84.1]% | evaluated |
| reacher | 5 | P3 | not_applicable | 5 | 33/50 (66.0%) | [52.2, 77.6]% | evaluated |
| reacher | 5 | P3 | not_applicable | 10 | 33/50 (66.0%) | [52.2, 77.6]% | evaluated |
| reacher | 5 | P3 | not_applicable | 16 | 37/50 (74.0%) | [60.4, 84.1]% | evaluated |
| reacher | 5 | P3 | not_applicable | 32 | 32/50 (64.0%) | [50.1, 75.9]% | evaluated |
| reacher | 6 | P0 | not_applicable | 1 | 31/50 (62.0%) | [48.2, 74.1]% | evaluated |
| reacher | 6 | P0 | not_applicable | 2 | 38/50 (76.0%) | [62.6, 85.7]% | evaluated |
| reacher | 6 | P0 | not_applicable | 5 | 33/50 (66.0%) | [52.2, 77.6]% | evaluated |
| reacher | 6 | P0 | not_applicable | 10 | 35/50 (70.0%) | [56.2, 80.9]% | evaluated |
| reacher | 6 | P0 | not_applicable | 16 | 37/50 (74.0%) | [60.4, 84.1]% | evaluated |
| reacher | 6 | P0 | not_applicable | 32 | 34/50 (68.0%) | [54.2, 79.2]% | evaluated |
| reacher | 6 | P1 | cem-clip | inv | 34/50 (68.0%) | [54.2, 79.2]% | evaluated |
| reacher | 6 | P2 | cem-clip | 1 | 37/50 (74.0%) | [60.4, 84.1]% | evaluated |
| reacher | 6 | P2 | cem-clip | 2 | 34/50 (68.0%) | [54.2, 79.2]% | evaluated |
| reacher | 6 | P2 | cem-clip | 5 | 41/50 (82.0%) | [69.2, 90.2]% | evaluated |
| reacher | 6 | P2 | cem-clip | 10 | 37/50 (74.0%) | [60.4, 84.1]% | evaluated |
| reacher | 6 | P2 | cem-clip | 16 | 39/50 (78.0%) | [64.8, 87.2]% | evaluated |
| reacher | 6 | P2 | cem-clip | 32 | 39/50 (78.0%) | [64.8, 87.2]% | evaluated |
| reacher | 6 | P3 | not_applicable | 1 | 35/50 (70.0%) | [56.2, 80.9]% | evaluated |
| reacher | 6 | P3 | not_applicable | 2 | 40/50 (80.0%) | [67.0, 88.8]% | evaluated |
| reacher | 6 | P3 | not_applicable | 5 | 37/50 (74.0%) | [60.4, 84.1]% | evaluated |
| reacher | 6 | P3 | not_applicable | 10 | 40/50 (80.0%) | [67.0, 88.8]% | evaluated |
| reacher | 6 | P3 | not_applicable | 16 | 41/50 (82.0%) | [69.2, 90.2]% | evaluated |
| reacher | 6 | P3 | not_applicable | 32 | 35/50 (70.0%) | [56.2, 80.9]% | evaluated |
| reacher | 7 | P0 | not_applicable | 1 | 34/50 (68.0%) | [54.2, 79.2]% | evaluated |
| reacher | 7 | P0 | not_applicable | 2 | 36/50 (72.0%) | [58.3, 82.5]% | evaluated |
| reacher | 7 | P0 | not_applicable | 5 | 36/50 (72.0%) | [58.3, 82.5]% | evaluated |
| reacher | 7 | P0 | not_applicable | 10 | 38/50 (76.0%) | [62.6, 85.7]% | evaluated |
| reacher | 7 | P0 | not_applicable | 16 | 34/50 (68.0%) | [54.2, 79.2]% | evaluated |
| reacher | 7 | P0 | not_applicable | 32 | 34/50 (68.0%) | [54.2, 79.2]% | evaluated |
| reacher | 7 | P1 | cem-clip | inv | 43/50 (86.0%) | [73.8, 93.0]% | evaluated |
| reacher | 7 | P2 | cem-clip | 1 | 44/50 (88.0%) | [76.2, 94.4]% | evaluated |
| reacher | 7 | P2 | cem-clip | 2 | 44/50 (88.0%) | [76.2, 94.4]% | evaluated |
| reacher | 7 | P2 | cem-clip | 5 | 39/50 (78.0%) | [64.8, 87.2]% | evaluated |
| reacher | 7 | P2 | cem-clip | 10 | 39/50 (78.0%) | [64.8, 87.2]% | evaluated |
| reacher | 7 | P2 | cem-clip | 16 | 41/50 (82.0%) | [69.2, 90.2]% | evaluated |
| reacher | 7 | P2 | cem-clip | 32 | 39/50 (78.0%) | [64.8, 87.2]% | evaluated |
| reacher | 7 | P3 | not_applicable | 1 | 37/50 (74.0%) | [60.4, 84.1]% | evaluated |
| reacher | 7 | P3 | not_applicable | 2 | 42/50 (84.0%) | [71.5, 91.7]% | evaluated |
| reacher | 7 | P3 | not_applicable | 5 | 36/50 (72.0%) | [58.3, 82.5]% | evaluated |
| reacher | 7 | P3 | not_applicable | 10 | 35/50 (70.0%) | [56.2, 80.9]% | evaluated |
| reacher | 7 | P3 | not_applicable | 16 | 36/50 (72.0%) | [58.3, 82.5]% | evaluated |
| reacher | 7 | P3 | not_applicable | 32 | 38/50 (76.0%) | [62.6, 85.7]% | evaluated |
| reacher | 8 | P0 | not_applicable | 1 | 31/50 (62.0%) | [48.2, 74.1]% | evaluated |
| reacher | 8 | P0 | not_applicable | 2 | 39/50 (78.0%) | [64.8, 87.2]% | evaluated |
| reacher | 8 | P0 | not_applicable | 5 | 37/50 (74.0%) | [60.4, 84.1]% | evaluated |
| reacher | 8 | P0 | not_applicable | 10 | 34/50 (68.0%) | [54.2, 79.2]% | evaluated |
| reacher | 8 | P0 | not_applicable | 16 | 36/50 (72.0%) | [58.3, 82.5]% | evaluated |
| reacher | 8 | P0 | not_applicable | 32 | 35/50 (70.0%) | [56.2, 80.9]% | evaluated |
| reacher | 8 | P1 | cem-clip | inv | 44/50 (88.0%) | [76.2, 94.4]% | evaluated |
| reacher | 8 | P2 | cem-clip | 1 | 42/50 (84.0%) | [71.5, 91.7]% | evaluated |
| reacher | 8 | P2 | cem-clip | 2 | 43/50 (86.0%) | [73.8, 93.0]% | evaluated |
| reacher | 8 | P2 | cem-clip | 5 | 45/50 (90.0%) | [78.6, 95.7]% | evaluated |
| reacher | 8 | P2 | cem-clip | 10 | 42/50 (84.0%) | [71.5, 91.7]% | evaluated |
| reacher | 8 | P2 | cem-clip | 16 | 40/50 (80.0%) | [67.0, 88.8]% | evaluated |
| reacher | 8 | P2 | cem-clip | 32 | 44/50 (88.0%) | [76.2, 94.4]% | evaluated |
| reacher | 8 | P3 | not_applicable | 1 | 38/50 (76.0%) | [62.6, 85.7]% | evaluated |
| reacher | 8 | P3 | not_applicable | 2 | 47/50 (94.0%) | [83.8, 97.9]% | evaluated |
| reacher | 8 | P3 | not_applicable | 5 | 44/50 (88.0%) | [76.2, 94.4]% | evaluated |
| reacher | 8 | P3 | not_applicable | 10 | 42/50 (84.0%) | [71.5, 91.7]% | evaluated |
| reacher | 8 | P3 | not_applicable | 16 | 40/50 (80.0%) | [67.0, 88.8]% | evaluated |
| reacher | 8 | P3 | not_applicable | 32 | 39/50 (78.0%) | [64.8, 87.2]% | evaluated |
| reacher | 9 | P0 | not_applicable | 1 | 33/50 (66.0%) | [52.2, 77.6]% | evaluated |
| reacher | 9 | P0 | not_applicable | 2 | 42/50 (84.0%) | [71.5, 91.7]% | evaluated |
| reacher | 9 | P0 | not_applicable | 5 | 37/50 (74.0%) | [60.4, 84.1]% | evaluated |
| reacher | 9 | P0 | not_applicable | 10 | 40/50 (80.0%) | [67.0, 88.8]% | evaluated |
| reacher | 9 | P0 | not_applicable | 16 | 39/50 (78.0%) | [64.8, 87.2]% | evaluated |
| reacher | 9 | P0 | not_applicable | 32 | 39/50 (78.0%) | [64.8, 87.2]% | evaluated |
| reacher | 9 | P1 | cem-clip | inv | 47/50 (94.0%) | [83.8, 97.9]% | evaluated |
| reacher | 9 | P2 | cem-clip | 1 | 45/50 (90.0%) | [78.6, 95.7]% | evaluated |
| reacher | 9 | P2 | cem-clip | 2 | 44/50 (88.0%) | [76.2, 94.4]% | evaluated |
| reacher | 9 | P2 | cem-clip | 5 | 42/50 (84.0%) | [71.5, 91.7]% | evaluated |
| reacher | 9 | P2 | cem-clip | 10 | 45/50 (90.0%) | [78.6, 95.7]% | evaluated |
| reacher | 9 | P2 | cem-clip | 16 | 45/50 (90.0%) | [78.6, 95.7]% | evaluated |
| reacher | 9 | P2 | cem-clip | 32 | 44/50 (88.0%) | [76.2, 94.4]% | evaluated |
| reacher | 9 | P3 | not_applicable | 1 | 37/50 (74.0%) | [60.4, 84.1]% | evaluated |
| reacher | 9 | P3 | not_applicable | 2 | 48/50 (96.0%) | [86.5, 98.9]% | evaluated |
| reacher | 9 | P3 | not_applicable | 5 | 43/50 (86.0%) | [73.8, 93.0]% | evaluated |
| reacher | 9 | P3 | not_applicable | 10 | 43/50 (86.0%) | [73.8, 93.0]% | evaluated |
| reacher | 9 | P3 | not_applicable | 16 | 44/50 (88.0%) | [76.2, 94.4]% | evaluated |
| reacher | 9 | P3 | not_applicable | 32 | 39/50 (78.0%) | [64.8, 87.2]% | evaluated |
| reacher | 10 | P0 | not_applicable | 1 | 36/50 (72.0%) | [58.3, 82.5]% | epoch10_reuse |
| reacher | 10 | P0 | not_applicable | 2 | 38/50 (76.0%) | [62.6, 85.7]% | epoch10_reuse |
| reacher | 10 | P0 | not_applicable | 5 | 37/50 (74.0%) | [60.4, 84.1]% | epoch10_reuse |
| reacher | 10 | P0 | not_applicable | 10 | 39/50 (78.0%) | [64.8, 87.2]% | epoch10_reuse |
| reacher | 10 | P0 | not_applicable | 16 | 37/50 (74.0%) | [60.4, 84.1]% | epoch10_reuse |
| reacher | 10 | P0 | not_applicable | 32 | 39/50 (78.0%) | [64.8, 87.2]% | epoch10_reuse |
| reacher | 10 | P1 | cem-clip | inv | 46/50 (92.0%) | [81.2, 96.8]% | epoch10_reuse |
| reacher | 10 | P2 | cem-clip | 1 | 44/50 (88.0%) | [76.2, 94.4]% | epoch10_reuse |
| reacher | 10 | P2 | cem-clip | 2 | 43/50 (86.0%) | [73.8, 93.0]% | epoch10_reuse |
| reacher | 10 | P2 | cem-clip | 5 | 45/50 (90.0%) | [78.6, 95.7]% | epoch10_reuse |
| reacher | 10 | P2 | cem-clip | 10 | 44/50 (88.0%) | [76.2, 94.4]% | epoch10_reuse |
| reacher | 10 | P2 | cem-clip | 16 | 42/50 (84.0%) | [71.5, 91.7]% | epoch10_reuse |
| reacher | 10 | P2 | cem-clip | 32 | 40/50 (80.0%) | [67.0, 88.8]% | epoch10_reuse |
| reacher | 10 | P3 | not_applicable | 1 | 37/50 (74.0%) | [60.4, 84.1]% | epoch10_reuse |
| reacher | 10 | P3 | not_applicable | 2 | 45/50 (90.0%) | [78.6, 95.7]% | epoch10_reuse |
| reacher | 10 | P3 | not_applicable | 5 | 42/50 (84.0%) | [71.5, 91.7]% | epoch10_reuse |
| reacher | 10 | P3 | not_applicable | 10 | 43/50 (86.0%) | [73.8, 93.0]% | epoch10_reuse |
| reacher | 10 | P3 | not_applicable | 16 | 39/50 (78.0%) | [64.8, 87.2]% | epoch10_reuse |
| reacher | 10 | P3 | not_applicable | 32 | 42/50 (84.0%) | [71.5, 91.7]% | epoch10_reuse |
| tworoom | 1 | P0 | not_applicable | 1 | 16/50 (32.0%) | [20.8, 45.8]% | evaluated |
| tworoom | 1 | P1 | legacy | inv | 23/50 (46.0%) | [33.0, 59.6]% | evaluated |
| tworoom | 1 | P2 | legacy | 1 | 25/50 (50.0%) | [36.6, 63.4]% | evaluated |
| tworoom | 1 | P3 | not_applicable | 1 | 20/50 (40.0%) | [27.6, 53.8]% | evaluated |
| tworoom | 2 | P0 | not_applicable | 1 | 35/50 (70.0%) | [56.2, 80.9]% | evaluated |
| tworoom | 2 | P1 | legacy | inv | 38/50 (76.0%) | [62.6, 85.7]% | evaluated |
| tworoom | 2 | P2 | legacy | 1 | 39/50 (78.0%) | [64.8, 87.2]% | evaluated |
| tworoom | 2 | P3 | not_applicable | 1 | 38/50 (76.0%) | [62.6, 85.7]% | evaluated |
| tworoom | 3 | P0 | not_applicable | 1 | 37/50 (74.0%) | [60.4, 84.1]% | evaluated |
| tworoom | 3 | P1 | legacy | inv | 40/50 (80.0%) | [67.0, 88.8]% | evaluated |
| tworoom | 3 | P2 | legacy | 1 | 47/50 (94.0%) | [83.8, 97.9]% | evaluated |
| tworoom | 3 | P3 | not_applicable | 1 | 39/50 (78.0%) | [64.8, 87.2]% | evaluated |
| tworoom | 4 | P0 | not_applicable | 1 | 44/50 (88.0%) | [76.2, 94.4]% | evaluated |
| tworoom | 4 | P1 | legacy | inv | 47/50 (94.0%) | [83.8, 97.9]% | evaluated |
| tworoom | 4 | P2 | legacy | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| tworoom | 4 | P3 | not_applicable | 1 | 46/50 (92.0%) | [81.2, 96.8]% | evaluated |
| tworoom | 5 | P0 | not_applicable | 1 | 47/50 (94.0%) | [83.8, 97.9]% | evaluated |
| tworoom | 5 | P1 | legacy | inv | 46/50 (92.0%) | [81.2, 96.8]% | evaluated |
| tworoom | 5 | P2 | legacy | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| tworoom | 5 | P3 | not_applicable | 1 | 47/50 (94.0%) | [83.8, 97.9]% | evaluated |
| tworoom | 6 | P0 | not_applicable | 1 | 49/50 (98.0%) | [89.5, 99.6]% | evaluated |
| tworoom | 6 | P1 | legacy | inv | 48/50 (96.0%) | [86.5, 98.9]% | evaluated |
| tworoom | 6 | P2 | legacy | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| tworoom | 6 | P3 | not_applicable | 1 | 49/50 (98.0%) | [89.5, 99.6]% | evaluated |
| tworoom | 7 | P0 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| tworoom | 7 | P1 | legacy | inv | 49/50 (98.0%) | [89.5, 99.6]% | evaluated |
| tworoom | 7 | P2 | legacy | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| tworoom | 7 | P3 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| tworoom | 8 | P0 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| tworoom | 8 | P1 | legacy | inv | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| tworoom | 8 | P2 | legacy | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| tworoom | 8 | P3 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| tworoom | 9 | P0 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| tworoom | 9 | P1 | legacy | inv | 49/50 (98.0%) | [89.5, 99.6]% | evaluated |
| tworoom | 9 | P2 | legacy | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| tworoom | 9 | P3 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | evaluated |
| tworoom | 10 | P0 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | epoch10_reuse |
| tworoom | 10 | P1 | legacy | inv | 50/50 (100.0%) | [92.9, 100.0]% | epoch10_reuse |
| tworoom | 10 | P2 | legacy | 1 | 50/50 (100.0%) | [92.9, 100.0]% | epoch10_reuse |
| tworoom | 10 | P3 | not_applicable | 1 | 50/50 (100.0%) | [92.9, 100.0]% | epoch10_reuse |

## 曲线图

每个 task 一张 P0/P1/P2/P3 四分面图；PNG、SVG、PDF 三种格式均写入 `plots/`。

- `cube_success_rate_by_epoch.png` / `.svg` / `.pdf`
- `pusht_success_rate_by_epoch.png` / `.svg` / `.pdf`
- `reacher_success_rate_by_epoch.png` / `.svg` / `.pdf`
- `tworoom_success_rate_by_epoch.png` / `.svg` / `.pdf`

## 解释边界

这是单训练 seed、单 checkpoint 系列、单 legacy cohort 的描述性曲线；epoch10 是历史 artifact 复用点。曲线只覆盖无 guidance 的 P0–P3 条件，不改变冻结的 Phase 1 guidance 结论或报告。
