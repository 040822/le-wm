# Round 4 Phase 4.5-4：旧协议与 dev 对照实验

## 实验范围

本轮使用 R4-AB seed3072 的 epoch10 checkpoint，在四个 task 上比较两个独立的 50-episode cohort：canonical `legacy_50.json` 与已完成 Phase 4.5-3 的 `round3_revised` dev 结果。每个 cohort 共 96 个条件：cube 19 个、pusht 19 个、reacher 33 个、tworoom 25 个。只运行 Euler；flow step 网格为 1/2/5/10/16/32（step16 为参考点）。Reacher 额外包含 P1/P2 的 cem-clip 与 cem-scale，TwoRoom 额外包含 P2/cem-clip。不运行 Heun、final 或 LeWM。

- 配置：`/data/users/wenxin/pre-exp/le-wm/config/round4/phase45_4.json`
- 输出根目录：`/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev`
- dev 结果来源：`/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev`（直接引用，未重跑）
- 分析生成路径：`/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/analysis/analysis.json`

## 验收与 cohort 完整性

- 条件结果：192 / 192；legacy=96 / 96，dev=96 / 96。
- `status=ok`：True；无 Heun：True；flow step 在预注册网格内：True。
- 每个结果 50 episodes，episode identity 已按 manifest 校验：True。
- legacy 与 dev 使用不同 cohort identity；两者之间只做成功率、Wilson 区间和绝对差异的 unpaired descriptive comparison，明确禁止 paired McNemar 检验。

## Manifest、checkpoint 与 cohort hash

| cohort | task | protocol_variant | cohort ID | cohort SHA256 | entries | seed | goal offset | manifest/result source |
|---|---|---|---|---|---:|---:|---:|---|
| legacy | cube | `legacy` | `cube_legacy_50_v1` | `b9054ebb327665142befac1b3ec36b2500de5dcdadc4008da322e2d0e4c721bb` | 50 | 42 | 25 | /data/users/wenxin/pre-exp/le-wm/outputs/round3/phase1/cohorts/cube/legacy_50.json |
| legacy | pusht | `legacy` | `pusht_legacy_50_v1` | `b110bee6afa2f7be372d57b3ef1247a44048c8b1e336491d74a53401ad479b0c` | 50 | 42 | 25 | /data/users/wenxin/pre-exp/le-wm/outputs/round3/phase1/cohorts/pusht/legacy_50.json |
| legacy | reacher | `legacy` | `reacher_legacy_50_v1` | `ff4f26ad3fd809e59e1b748c7c93c33a8e51b48e13d2505e89e5909d6121bf45` | 50 | 42 | 25 | /data/users/wenxin/pre-exp/le-wm/outputs/round3/phase1/cohorts/reacher/legacy_50.json |
| legacy | tworoom | `legacy` | `tworoom_legacy_50_v1` | `d452befec00361add056c880e09e9ef2cb68060e7ad599535d8ce83a875e9091` | 50 | 42 | 25 | /data/users/wenxin/pre-exp/le-wm/outputs/round3/phase1/cohorts/tworoom/legacy_50.json |
| dev | cube | `round3_revised` | `cube_dev_round3_round3_revised` | `578f6ea52e56d77461cb5bb2aeb01a004a21b9f6659689596f6f55f25dc17d01` | 50 | 42 | 25 | /data/users/wenxin/pre-exp/le-wm/outputs/round3/phase1/cohorts/cube/dev_round3_revised.json |
| dev | pusht | `round3_revised` | `pusht_dev_round3_round3_revised` | `5b3cd95037c17bc6282bdd29cab4e550aff1a5205d051b2991263a1bac937560` | 50 | 42 | 25 | /data/users/wenxin/pre-exp/le-wm/outputs/round3/phase1/cohorts/pusht/dev_round3_revised.json |
| dev | reacher | `round3_revised` | `reacher_dev_round3_round3_revised` | `d18695b2377000ce7dd774fcad78e4bb2d75980fb3ccbd98cc83d09ef50f1d06` | 50 | 42 | 25 | /data/users/wenxin/pre-exp/le-wm/outputs/round3/phase1/cohorts/reacher/dev_round3_revised.json |
| dev | tworoom | `round3_revised` | `tworoom_dev_round3_round3_revised` | `367aefc82f4c0786992172f535f00b743750108a8614c2d282019e9a78abdc6b` | 50 | 42 | 25 | /data/users/wenxin/pre-exp/le-wm/outputs/round3/phase1/cohorts/tworoom/dev_round3_revised.json |

checkpoint paths are embedded in every condition row and validated against the config:

| task | checkpoint | SHA256 (config-time reference) |
|---|---|---|
| cube | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/ab_seed3072_cube/checkpoints/r4_ab_seed3072_weights_epoch_10.pt` | `2748151a2c43c7841246a5fb8ee1fd7915ebbc00bfc27aa1a32b0f9080d92956` |
| pusht | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/ab_seed3072_pusht/checkpoints/r4_ab_seed3072_weights_epoch_10.pt` | `62d00096a34e9f0b6da4ceb6a3c61eb350701a7aade5f5e4d2c1ee528b06015e` |
| reacher | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/ab_seed3072_reacher/checkpoints/r4_ab_seed3072_weights_epoch_10.pt` | `087991339c9499d10c1a58a28e72d07c7dfd819d5f072c67bfd0169acaa83553` |
| tworoom | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/ab_seed3072_tworoom/checkpoints/r4_ab_seed3072_weights_epoch_10.pt` | `48d1ed2520ec7142327969eacf1c37b144faec51cfd6c9ce72701a8421bf57c2` |

## 完整条件表

### cube

| cohort | mode/protocol | integrator | step | success | Wilson 95% | planning mean/median/P95 (s) | result |
|---|---|---|---:|---:|---|---:|---|
| dev | P0/not_applicable | euler | 1 | 50/50 (100.0%) | [92.9, 100.0]% | 0.49/0.49/0.49 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/cube/P0/not_applicable/step_1/euler/dev/result.json` |
| dev | P0/not_applicable | euler | 2 | 50/50 (100.0%) | [92.9, 100.0]% | 0.05/0.05/0.05 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/cube/P0/not_applicable/step_2/euler/dev/result.json` |
| dev | P0/not_applicable | euler | 5 | 50/50 (100.0%) | [92.9, 100.0]% | 0.08/0.08/0.08 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/cube/P0/not_applicable/step_5/euler/dev/result.json` |
| dev | P0/not_applicable | euler | 10 | 50/50 (100.0%) | [92.9, 100.0]% | 0.07/0.07/0.07 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/cube/P0/not_applicable/step_10/euler/dev/result.json` |
| dev | P0/not_applicable | euler | 16 | 50/50 (100.0%) | [92.9, 100.0]% | 0.11/0.11/0.11 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/cube/P0/not_applicable/step_16/euler/dev/result.json` |
| dev | P0/not_applicable | euler | 32 | 50/50 (100.0%) | [92.9, 100.0]% | 0.19/0.19/0.19 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/cube/P0/not_applicable/step_32/euler/dev/result.json` |
| dev | P1/legacy | not_applicable | inv | 20/50 (40.0%) | [27.6, 53.8]% | 15.42/15.42/18.78 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/cube/P1/legacy/invariant/not_applicable/dev/result.json` |
| dev | P2/legacy | euler | 1 | 49/50 (98.0%) | [89.5, 99.6]% | 11.66/11.66/21.75 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/cube/P2/legacy/step_1/euler/dev/result.json` |
| dev | P2/legacy | euler | 2 | 49/50 (98.0%) | [89.5, 99.6]% | 11.83/11.83/22.09 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/cube/P2/legacy/step_2/euler/dev/result.json` |
| dev | P2/legacy | euler | 5 | 49/50 (98.0%) | [89.5, 99.6]% | 11.26/11.26/20.74 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/cube/P2/legacy/step_5/euler/dev/result.json` |
| dev | P2/legacy | euler | 10 | 50/50 (100.0%) | [92.9, 100.0]% | 19.67/19.67/19.67 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/cube/P2/legacy/step_10/euler/dev/result.json` |
| dev | P2/legacy | euler | 16 | 48/50 (96.0%) | [86.5, 98.9]% | 10.55/10.55/18.84 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/cube/P2/legacy/step_16/euler/dev/result.json` |
| dev | P2/legacy | euler | 32 | 50/50 (100.0%) | [92.9, 100.0]% | 21.26/21.26/21.26 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/cube/P2/legacy/step_32/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 1 | 50/50 (100.0%) | [92.9, 100.0]% | 0.29/0.29/0.29 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/cube/P3/not_applicable/step_1/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 2 | 50/50 (100.0%) | [92.9, 100.0]% | 0.31/0.31/0.31 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/cube/P3/not_applicable/step_2/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 5 | 50/50 (100.0%) | [92.9, 100.0]% | 0.38/0.38/0.38 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/cube/P3/not_applicable/step_5/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 10 | 50/50 (100.0%) | [92.9, 100.0]% | 0.52/0.52/0.52 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/cube/P3/not_applicable/step_10/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 16 | 50/50 (100.0%) | [92.9, 100.0]% | 0.56/0.56/0.56 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/cube/P3/not_applicable/step_16/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 32 | 50/50 (100.0%) | [92.9, 100.0]% | 1.30/1.30/1.30 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/cube/P3/not_applicable/step_32/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 1 | 50/50 (100.0%) | [92.9, 100.0]% | 0.42/0.42/0.42 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/cube/P0/not_applicable/step_1/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 2 | 50/50 (100.0%) | [92.9, 100.0]% | 0.06/0.06/0.06 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/cube/P0/not_applicable/step_2/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 5 | 50/50 (100.0%) | [92.9, 100.0]% | 0.29/0.29/0.29 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/cube/P0/not_applicable/step_5/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 10 | 50/50 (100.0%) | [92.9, 100.0]% | 0.06/0.06/0.06 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/cube/P0/not_applicable/step_10/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 16 | 50/50 (100.0%) | [92.9, 100.0]% | 0.09/0.09/0.09 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/cube/P0/not_applicable/step_16/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 32 | 50/50 (100.0%) | [92.9, 100.0]% | 0.15/0.15/0.15 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/cube/P0/not_applicable/step_32/euler/dev/result.json` |
| legacy | P1/legacy | not_applicable | inv | 35/50 (70.0%) | [56.2, 80.9]% | 15.15/15.15/23.25 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/cube/P1/legacy/invariant/not_applicable/dev/result.json` |
| legacy | P2/legacy | euler | 1 | 49/50 (98.0%) | [89.5, 99.6]% | 3.57/3.57/6.51 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/cube/P2/legacy/step_1/euler/dev/result.json` |
| legacy | P2/legacy | euler | 2 | 48/50 (96.0%) | [86.5, 98.9]% | 4.28/4.28/7.86 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/cube/P2/legacy/step_2/euler/dev/result.json` |
| legacy | P2/legacy | euler | 5 | 49/50 (98.0%) | [89.5, 99.6]% | 3.21/3.21/5.77 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/cube/P2/legacy/step_5/euler/dev/result.json` |
| legacy | P2/legacy | euler | 10 | 48/50 (96.0%) | [86.5, 98.9]% | 4.87/4.87/8.96 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/cube/P2/legacy/step_10/euler/dev/result.json` |
| legacy | P2/legacy | euler | 16 | 49/50 (98.0%) | [89.5, 99.6]% | 3.00/3.00/5.52 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/cube/P2/legacy/step_16/euler/dev/result.json` |
| legacy | P2/legacy | euler | 32 | 49/50 (98.0%) | [89.5, 99.6]% | 3.73/3.73/6.75 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/cube/P2/legacy/step_32/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 1 | 50/50 (100.0%) | [92.9, 100.0]% | 0.69/0.69/0.69 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/cube/P3/not_applicable/step_1/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 2 | 50/50 (100.0%) | [92.9, 100.0]% | 0.25/0.25/0.25 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/cube/P3/not_applicable/step_2/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 5 | 50/50 (100.0%) | [92.9, 100.0]% | 0.24/0.24/0.24 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/cube/P3/not_applicable/step_5/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 10 | 50/50 (100.0%) | [92.9, 100.0]% | 0.30/0.30/0.30 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/cube/P3/not_applicable/step_10/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 16 | 50/50 (100.0%) | [92.9, 100.0]% | 0.36/0.36/0.36 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/cube/P3/not_applicable/step_16/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 32 | 50/50 (100.0%) | [92.9, 100.0]% | 0.54/0.54/0.54 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/cube/P3/not_applicable/step_32/euler/dev/result.json` |

#### step 子表

| cohort | mode/protocol | step1 | step2 | step5 | step10 | step16 | step32 |
|---|---|---|---|---|---|---|---|
| legacy | P0/not_applicable | 1: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 5: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 10: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 16: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 32: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| legacy | P2/legacy | 1: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 2: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 5: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 10: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 16: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 32: 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| legacy | P3/not_applicable | 1: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 5: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 10: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 16: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 32: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| dev | P0/not_applicable | 1: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 5: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 10: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 16: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 32: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| dev | P2/legacy | 1: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 2: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 5: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 10: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 16: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 32: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| dev | P3/not_applicable | 1: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 5: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 10: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 16: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 32: 50/50 (100.0%)<br>CI [92.9, 100.0]% |

#### mode/protocol 子表

| cohort | step | P0/not_applicable | P1/legacy | P1/cem-clip | P1/cem-scale | P2/legacy | P2/cem-clip | P2/cem-scale | P3/not_applicable |
|---|---|---|---|---|---|---|---|---|---|
| legacy | inv | — | inv: 35/50 (70.0%)<br>CI [56.2, 80.9]% | — | — | — | — | — | — |
| legacy | 1 | 1: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | — | — | 1: 49/50 (98.0%)<br>CI [89.5, 99.6]% | — | — | 1: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| legacy | 2 | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | — | — | 2: 48/50 (96.0%)<br>CI [86.5, 98.9]% | — | — | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| legacy | 5 | 5: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | — | — | 5: 49/50 (98.0%)<br>CI [89.5, 99.6]% | — | — | 5: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| legacy | 10 | 10: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | — | — | 10: 48/50 (96.0%)<br>CI [86.5, 98.9]% | — | — | 10: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| legacy | 16 | 16: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | — | — | 16: 49/50 (98.0%)<br>CI [89.5, 99.6]% | — | — | 16: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| legacy | 32 | 32: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | — | — | 32: 49/50 (98.0%)<br>CI [89.5, 99.6]% | — | — | 32: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| dev | inv | — | inv: 20/50 (40.0%)<br>CI [27.6, 53.8]% | — | — | — | — | — | — |
| dev | 1 | 1: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | — | — | 1: 49/50 (98.0%)<br>CI [89.5, 99.6]% | — | — | 1: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| dev | 2 | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | — | — | 2: 49/50 (98.0%)<br>CI [89.5, 99.6]% | — | — | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| dev | 5 | 5: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | — | — | 5: 49/50 (98.0%)<br>CI [89.5, 99.6]% | — | — | 5: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| dev | 10 | 10: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | — | — | 10: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | — | 10: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| dev | 16 | 16: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | — | — | 16: 48/50 (96.0%)<br>CI [86.5, 98.9]% | — | — | 16: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| dev | 32 | 32: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | — | — | 32: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | — | 32: 50/50 (100.0%)<br>CI [92.9, 100.0]% |

### pusht

| cohort | mode/protocol | integrator | step | success | Wilson 95% | planning mean/median/P95 (s) | result |
|---|---|---|---:|---:|---|---:|---|
| dev | P0/not_applicable | euler | 1 | 48/50 (96.0%) | [86.5, 98.9]% | 0.23/0.23/0.36 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/pusht/P0/not_applicable/step_1/euler/dev/result.json` |
| dev | P0/not_applicable | euler | 2 | 50/50 (100.0%) | [92.9, 100.0]% | 0.05/0.05/0.05 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/pusht/P0/not_applicable/step_2/euler/dev/result.json` |
| dev | P0/not_applicable | euler | 5 | 48/50 (96.0%) | [86.5, 98.9]% | 0.05/0.05/0.06 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/pusht/P0/not_applicable/step_5/euler/dev/result.json` |
| dev | P0/not_applicable | euler | 10 | 48/50 (96.0%) | [86.5, 98.9]% | 0.07/0.07/0.08 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/pusht/P0/not_applicable/step_10/euler/dev/result.json` |
| dev | P0/not_applicable | euler | 16 | 46/50 (92.0%) | [81.2, 96.8]% | 0.08/0.08/0.09 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/pusht/P0/not_applicable/step_16/euler/dev/result.json` |
| dev | P0/not_applicable | euler | 32 | 46/50 (92.0%) | [81.2, 96.8]% | 0.13/0.13/0.14 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/pusht/P0/not_applicable/step_32/euler/dev/result.json` |
| dev | P1/legacy | not_applicable | inv | 45/50 (90.0%) | [78.6, 95.7]% | 5.01/5.01/8.78 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/pusht/P1/legacy/invariant/not_applicable/dev/result.json` |
| dev | P2/legacy | euler | 1 | 47/50 (94.0%) | [83.8, 97.9]% | 3.45/3.45/6.10 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/pusht/P2/legacy/step_1/euler/dev/result.json` |
| dev | P2/legacy | euler | 2 | 46/50 (92.0%) | [81.2, 96.8]% | 4.26/4.26/7.49 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/pusht/P2/legacy/step_2/euler/dev/result.json` |
| dev | P2/legacy | euler | 5 | 47/50 (94.0%) | [83.8, 97.9]% | 8.58/8.58/15.08 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/pusht/P2/legacy/step_5/euler/dev/result.json` |
| dev | P2/legacy | euler | 10 | 46/50 (92.0%) | [81.2, 96.8]% | 9.21/9.21/15.68 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/pusht/P2/legacy/step_10/euler/dev/result.json` |
| dev | P2/legacy | euler | 16 | 46/50 (92.0%) | [81.2, 96.8]% | 9.71/9.71/17.09 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/pusht/P2/legacy/step_16/euler/dev/result.json` |
| dev | P2/legacy | euler | 32 | 46/50 (92.0%) | [81.2, 96.8]% | 9.49/9.49/16.19 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/pusht/P2/legacy/step_32/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 1 | 49/50 (98.0%) | [89.5, 99.6]% | 0.13/0.13/0.22 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/pusht/P3/not_applicable/step_1/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 2 | 50/50 (100.0%) | [92.9, 100.0]% | 0.26/0.26/0.26 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/pusht/P3/not_applicable/step_2/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 5 | 49/50 (98.0%) | [89.5, 99.6]% | 0.19/0.19/0.31 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/pusht/P3/not_applicable/step_5/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 10 | 48/50 (96.0%) | [86.5, 98.9]% | 0.19/0.19/0.31 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/pusht/P3/not_applicable/step_10/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 16 | 48/50 (96.0%) | [86.5, 98.9]% | 0.24/0.24/0.37 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/pusht/P3/not_applicable/step_16/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 32 | 48/50 (96.0%) | [86.5, 98.9]% | 0.37/0.37/0.59 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/pusht/P3/not_applicable/step_32/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 1 | 48/50 (96.0%) | [86.5, 98.9]% | 0.26/0.26/0.41 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/pusht/P0/not_applicable/step_1/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 2 | 50/50 (100.0%) | [92.9, 100.0]% | 0.04/0.04/0.06 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/pusht/P0/not_applicable/step_2/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 5 | 49/50 (98.0%) | [89.5, 99.6]% | 0.20/0.20/0.29 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/pusht/P0/not_applicable/step_5/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 10 | 47/50 (94.0%) | [83.8, 97.9]% | 0.05/0.05/0.06 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/pusht/P0/not_applicable/step_10/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 16 | 47/50 (94.0%) | [83.8, 97.9]% | 0.08/0.08/0.08 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/pusht/P0/not_applicable/step_16/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 32 | 47/50 (94.0%) | [83.8, 97.9]% | 0.11/0.11/0.12 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/pusht/P0/not_applicable/step_32/euler/dev/result.json` |
| legacy | P1/legacy | not_applicable | inv | 45/50 (90.0%) | [78.6, 95.7]% | 9.71/9.71/16.25 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/pusht/P1/legacy/invariant/not_applicable/dev/result.json` |
| legacy | P2/legacy | euler | 1 | 49/50 (98.0%) | [89.5, 99.6]% | 8.77/8.77/15.39 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/pusht/P2/legacy/step_1/euler/dev/result.json` |
| legacy | P2/legacy | euler | 2 | 48/50 (96.0%) | [86.5, 98.9]% | 10.26/10.26/18.06 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/pusht/P2/legacy/step_2/euler/dev/result.json` |
| legacy | P2/legacy | euler | 5 | 47/50 (94.0%) | [83.8, 97.9]% | 2.81/2.81/4.95 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/pusht/P2/legacy/step_5/euler/dev/result.json` |
| legacy | P2/legacy | euler | 10 | 48/50 (96.0%) | [86.5, 98.9]% | 2.68/2.68/4.78 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/pusht/P2/legacy/step_10/euler/dev/result.json` |
| legacy | P2/legacy | euler | 16 | 48/50 (96.0%) | [86.5, 98.9]% | 3.39/3.39/6.04 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/pusht/P2/legacy/step_16/euler/dev/result.json` |
| legacy | P2/legacy | euler | 32 | 48/50 (96.0%) | [86.5, 98.9]% | 3.09/3.09/5.30 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/pusht/P2/legacy/step_32/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 1 | 48/50 (96.0%) | [86.5, 98.9]% | 0.29/0.29/0.52 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/pusht/P3/not_applicable/step_1/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 2 | 49/50 (98.0%) | [89.5, 99.6]% | 0.13/0.13/0.23 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/pusht/P3/not_applicable/step_2/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 5 | 50/50 (100.0%) | [92.9, 100.0]% | 0.25/0.25/0.25 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/pusht/P3/not_applicable/step_5/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 10 | 49/50 (98.0%) | [89.5, 99.6]% | 0.19/0.19/0.31 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/pusht/P3/not_applicable/step_10/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 16 | 48/50 (96.0%) | [86.5, 98.9]% | 0.23/0.23/0.37 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/pusht/P3/not_applicable/step_16/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 32 | 49/50 (98.0%) | [89.5, 99.6]% | 0.34/0.34/0.54 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/pusht/P3/not_applicable/step_32/euler/dev/result.json` |

#### step 子表

| cohort | mode/protocol | step1 | step2 | step5 | step10 | step16 | step32 |
|---|---|---|---|---|---|---|---|
| legacy | P0/not_applicable | 1: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 5: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 10: 47/50 (94.0%)<br>CI [83.8, 97.9]% | 16: 47/50 (94.0%)<br>CI [83.8, 97.9]% | 32: 47/50 (94.0%)<br>CI [83.8, 97.9]% |
| legacy | P2/legacy | 1: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 2: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 5: 47/50 (94.0%)<br>CI [83.8, 97.9]% | 10: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 16: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 32: 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| legacy | P3/not_applicable | 1: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 2: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 5: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 10: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 16: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 32: 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| dev | P0/not_applicable | 1: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 5: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 10: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 16: 46/50 (92.0%)<br>CI [81.2, 96.8]% | 32: 46/50 (92.0%)<br>CI [81.2, 96.8]% |
| dev | P2/legacy | 1: 47/50 (94.0%)<br>CI [83.8, 97.9]% | 2: 46/50 (92.0%)<br>CI [81.2, 96.8]% | 5: 47/50 (94.0%)<br>CI [83.8, 97.9]% | 10: 46/50 (92.0%)<br>CI [81.2, 96.8]% | 16: 46/50 (92.0%)<br>CI [81.2, 96.8]% | 32: 46/50 (92.0%)<br>CI [81.2, 96.8]% |
| dev | P3/not_applicable | 1: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 5: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 10: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 16: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 32: 48/50 (96.0%)<br>CI [86.5, 98.9]% |

#### mode/protocol 子表

| cohort | step | P0/not_applicable | P1/legacy | P1/cem-clip | P1/cem-scale | P2/legacy | P2/cem-clip | P2/cem-scale | P3/not_applicable |
|---|---|---|---|---|---|---|---|---|---|
| legacy | inv | — | inv: 45/50 (90.0%)<br>CI [78.6, 95.7]% | — | — | — | — | — | — |
| legacy | 1 | 1: 48/50 (96.0%)<br>CI [86.5, 98.9]% | — | — | — | 1: 49/50 (98.0%)<br>CI [89.5, 99.6]% | — | — | 1: 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| legacy | 2 | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | — | — | 2: 48/50 (96.0%)<br>CI [86.5, 98.9]% | — | — | 2: 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| legacy | 5 | 5: 49/50 (98.0%)<br>CI [89.5, 99.6]% | — | — | — | 5: 47/50 (94.0%)<br>CI [83.8, 97.9]% | — | — | 5: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| legacy | 10 | 10: 47/50 (94.0%)<br>CI [83.8, 97.9]% | — | — | — | 10: 48/50 (96.0%)<br>CI [86.5, 98.9]% | — | — | 10: 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| legacy | 16 | 16: 47/50 (94.0%)<br>CI [83.8, 97.9]% | — | — | — | 16: 48/50 (96.0%)<br>CI [86.5, 98.9]% | — | — | 16: 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| legacy | 32 | 32: 47/50 (94.0%)<br>CI [83.8, 97.9]% | — | — | — | 32: 48/50 (96.0%)<br>CI [86.5, 98.9]% | — | — | 32: 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| dev | inv | — | inv: 45/50 (90.0%)<br>CI [78.6, 95.7]% | — | — | — | — | — | — |
| dev | 1 | 1: 48/50 (96.0%)<br>CI [86.5, 98.9]% | — | — | — | 1: 47/50 (94.0%)<br>CI [83.8, 97.9]% | — | — | 1: 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| dev | 2 | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | — | — | 2: 46/50 (92.0%)<br>CI [81.2, 96.8]% | — | — | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| dev | 5 | 5: 48/50 (96.0%)<br>CI [86.5, 98.9]% | — | — | — | 5: 47/50 (94.0%)<br>CI [83.8, 97.9]% | — | — | 5: 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| dev | 10 | 10: 48/50 (96.0%)<br>CI [86.5, 98.9]% | — | — | — | 10: 46/50 (92.0%)<br>CI [81.2, 96.8]% | — | — | 10: 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| dev | 16 | 16: 46/50 (92.0%)<br>CI [81.2, 96.8]% | — | — | — | 16: 46/50 (92.0%)<br>CI [81.2, 96.8]% | — | — | 16: 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| dev | 32 | 32: 46/50 (92.0%)<br>CI [81.2, 96.8]% | — | — | — | 32: 46/50 (92.0%)<br>CI [81.2, 96.8]% | — | — | 32: 48/50 (96.0%)<br>CI [86.5, 98.9]% |

### reacher

| cohort | mode/protocol | integrator | step | success | Wilson 95% | planning mean/median/P95 (s) | result |
|---|---|---|---:|---:|---|---:|---|
| dev | P0/not_applicable | euler | 1 | 45/50 (90.0%) | [78.6, 95.7]% | 0.22/0.22/0.39 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P0/not_applicable/step_1/euler/dev/result.json` |
| dev | P0/not_applicable | euler | 2 | 42/50 (84.0%) | [71.5, 91.7]% | 0.03/0.03/0.03 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P0/not_applicable/step_2/euler/dev/result.json` |
| dev | P0/not_applicable | euler | 5 | 40/50 (80.0%) | [67.0, 88.8]% | 0.05/0.05/0.06 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P0/not_applicable/step_5/euler/dev/result.json` |
| dev | P0/not_applicable | euler | 10 | 35/50 (70.0%) | [56.2, 80.9]% | 0.06/0.06/0.06 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P0/not_applicable/step_10/euler/dev/result.json` |
| dev | P0/not_applicable | euler | 16 | 35/50 (70.0%) | [56.2, 80.9]% | 0.09/0.09/0.09 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P0/not_applicable/step_16/euler/dev/result.json` |
| dev | P0/not_applicable | euler | 32 | 34/50 (68.0%) | [54.2, 79.2]% | 0.13/0.13/0.14 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P0/not_applicable/step_32/euler/dev/result.json` |
| dev | P1/cem-clip | not_applicable | inv | 44/50 (88.0%) | [76.2, 94.4]% | 9.01/9.01/11.63 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P1/cem-clip/invariant/not_applicable/dev/result.json` |
| dev | P1/cem-scale | not_applicable | inv | 49/50 (98.0%) | [89.5, 99.6]% | 28.47/28.47/31.29 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P1/cem-scale/invariant/not_applicable/dev/result.json` |
| dev | P1/legacy | not_applicable | inv | 37/50 (74.0%) | [60.4, 84.1]% | 6.66/6.66/7.73 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P1/legacy/invariant/not_applicable/dev/result.json` |
| dev | P2/cem-clip | euler | 1 | 40/50 (80.0%) | [67.0, 88.8]% | 8.85/8.85/11.59 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P2/cem-clip/step_1/euler/dev/result.json` |
| dev | P2/cem-clip | euler | 2 | 42/50 (84.0%) | [71.5, 91.7]% | 8.90/8.90/11.37 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P2/cem-clip/step_2/euler/dev/result.json` |
| dev | P2/cem-clip | euler | 5 | 42/50 (84.0%) | [71.5, 91.7]% | 9.55/9.55/11.49 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P2/cem-clip/step_5/euler/dev/result.json` |
| dev | P2/cem-clip | euler | 10 | 45/50 (90.0%) | [78.6, 95.7]% | 8.70/8.70/11.34 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P2/cem-clip/step_10/euler/dev/result.json` |
| dev | P2/cem-clip | euler | 16 | 40/50 (80.0%) | [67.0, 88.8]% | 9.20/9.20/11.44 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P2/cem-clip/step_16/euler/dev/result.json` |
| dev | P2/cem-clip | euler | 32 | 45/50 (90.0%) | [78.6, 95.7]% | 9.67/9.67/11.94 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P2/cem-clip/step_32/euler/dev/result.json` |
| dev | P2/cem-scale | euler | 1 | 46/50 (92.0%) | [81.2, 96.8]% | 39.18/39.18/45.38 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P2/cem-scale/step_1/euler/dev/result.json` |
| dev | P2/cem-scale | euler | 2 | 46/50 (92.0%) | [81.2, 96.8]% | 32.33/32.33/37.76 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P2/cem-scale/step_2/euler/dev/result.json` |
| dev | P2/cem-scale | euler | 5 | 45/50 (90.0%) | [78.6, 95.7]% | 26.10/26.10/31.68 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P2/cem-scale/step_5/euler/dev/result.json` |
| dev | P2/cem-scale | euler | 10 | 45/50 (90.0%) | [78.6, 95.7]% | 20.51/20.51/22.96 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P2/cem-scale/step_10/euler/dev/result.json` |
| dev | P2/cem-scale | euler | 16 | 46/50 (92.0%) | [81.2, 96.8]% | 18.03/18.03/22.11 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P2/cem-scale/step_16/euler/dev/result.json` |
| dev | P2/cem-scale | euler | 32 | 45/50 (90.0%) | [78.6, 95.7]% | 13.24/13.24/15.97 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P2/cem-scale/step_32/euler/dev/result.json` |
| dev | P2/legacy | euler | 1 | 46/50 (92.0%) | [81.2, 96.8]% | 5.88/5.88/7.33 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P2/legacy/step_1/euler/dev/result.json` |
| dev | P2/legacy | euler | 2 | 37/50 (74.0%) | [60.4, 84.1]% | 5.35/5.35/6.59 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P2/legacy/step_2/euler/dev/result.json` |
| dev | P2/legacy | euler | 5 | 32/50 (64.0%) | [50.1, 75.9]% | 5.64/5.64/6.31 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P2/legacy/step_5/euler/dev/result.json` |
| dev | P2/legacy | euler | 10 | 31/50 (62.0%) | [48.2, 74.1]% | 5.11/5.11/5.78 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P2/legacy/step_10/euler/dev/result.json` |
| dev | P2/legacy | euler | 16 | 25/50 (50.0%) | [36.6, 63.4]% | 5.57/5.57/6.06 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P2/legacy/step_16/euler/dev/result.json` |
| dev | P2/legacy | euler | 32 | 28/50 (56.0%) | [42.3, 68.8]% | 6.02/6.02/6.21 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P2/legacy/step_32/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 1 | 46/50 (92.0%) | [81.2, 96.8]% | 0.18/0.18/0.24 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P3/not_applicable/step_1/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 2 | 45/50 (90.0%) | [78.6, 95.7]% | 0.18/0.18/0.22 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P3/not_applicable/step_2/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 5 | 40/50 (80.0%) | [67.0, 88.8]% | 0.21/0.21/0.28 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P3/not_applicable/step_5/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 10 | 36/50 (72.0%) | [58.3, 82.5]% | 0.27/0.27/0.33 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P3/not_applicable/step_10/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 16 | 43/50 (86.0%) | [73.8, 93.0]% | 0.32/0.32/0.38 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P3/not_applicable/step_16/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 32 | 42/50 (84.0%) | [71.5, 91.7]% | 0.48/0.48/0.61 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P3/not_applicable/step_32/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 1 | 36/50 (72.0%) | [58.3, 82.5]% | 0.22/0.22/0.39 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P0/not_applicable/step_1/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 2 | 38/50 (76.0%) | [62.6, 85.7]% | 0.07/0.07/0.09 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P0/not_applicable/step_2/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 5 | 37/50 (74.0%) | [60.4, 84.1]% | 0.19/0.19/0.31 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P0/not_applicable/step_5/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 10 | 39/50 (78.0%) | [64.8, 87.2]% | 0.07/0.07/0.07 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P0/not_applicable/step_10/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 16 | 37/50 (74.0%) | [60.4, 84.1]% | 0.09/0.09/0.09 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P0/not_applicable/step_16/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 32 | 39/50 (78.0%) | [64.8, 87.2]% | 0.14/0.14/0.15 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P0/not_applicable/step_32/euler/dev/result.json` |
| legacy | P1/cem-clip | not_applicable | inv | 46/50 (92.0%) | [81.2, 96.8]% | 37.57/37.57/57.94 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P1/cem-clip/invariant/not_applicable/dev/result.json` |
| legacy | P1/cem-scale | not_applicable | inv | 42/50 (84.0%) | [71.5, 91.7]% | 7.07/7.07/8.87 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P1/cem-scale/invariant/not_applicable/dev/result.json` |
| legacy | P1/legacy | not_applicable | inv | 36/50 (72.0%) | [58.3, 82.5]% | 14.12/14.12/15.50 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P1/legacy/invariant/not_applicable/dev/result.json` |
| legacy | P2/cem-clip | euler | 1 | 44/50 (88.0%) | [76.2, 94.4]% | 49.54/49.54/61.02 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P2/cem-clip/step_1/euler/dev/result.json` |
| legacy | P2/cem-clip | euler | 2 | 43/50 (86.0%) | [73.8, 93.0]% | 47.37/47.37/61.27 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P2/cem-clip/step_2/euler/dev/result.json` |
| legacy | P2/cem-clip | euler | 5 | 45/50 (90.0%) | [78.6, 95.7]% | 15.10/15.10/20.77 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P2/cem-clip/step_5/euler/dev/result.json` |
| legacy | P2/cem-clip | euler | 10 | 44/50 (88.0%) | [76.2, 94.4]% | 15.89/15.89/20.76 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P2/cem-clip/step_10/euler/dev/result.json` |
| legacy | P2/cem-clip | euler | 16 | 42/50 (84.0%) | [71.5, 91.7]% | 15.86/15.86/21.94 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P2/cem-clip/step_16/euler/dev/result.json` |
| legacy | P2/cem-clip | euler | 32 | 40/50 (80.0%) | [67.0, 88.8]% | 17.54/17.54/23.06 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P2/cem-clip/step_32/euler/dev/result.json` |
| legacy | P2/cem-scale | euler | 1 | 43/50 (86.0%) | [73.8, 93.0]% | 6.65/6.65/8.05 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P2/cem-scale/step_1/euler/dev/result.json` |
| legacy | P2/cem-scale | euler | 2 | 44/50 (88.0%) | [76.2, 94.4]% | 6.59/6.59/8.25 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P2/cem-scale/step_2/euler/dev/result.json` |
| legacy | P2/cem-scale | euler | 5 | 41/50 (82.0%) | [69.2, 90.2]% | 20.57/20.57/26.76 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P2/cem-scale/step_5/euler/dev/result.json` |
| legacy | P2/cem-scale | euler | 10 | 43/50 (86.0%) | [73.8, 93.0]% | 21.77/21.77/27.58 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P2/cem-scale/step_10/euler/dev/result.json` |
| legacy | P2/cem-scale | euler | 16 | 41/50 (82.0%) | [69.2, 90.2]% | 22.30/22.30/27.00 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P2/cem-scale/step_16/euler/dev/result.json` |
| legacy | P2/cem-scale | euler | 32 | 44/50 (88.0%) | [76.2, 94.4]% | 22.10/22.10/27.12 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P2/cem-scale/step_32/euler/dev/result.json` |
| legacy | P2/legacy | euler | 1 | 40/50 (80.0%) | [67.0, 88.8]% | 13.58/13.58/16.13 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P2/legacy/step_1/euler/dev/result.json` |
| legacy | P2/legacy | euler | 2 | 39/50 (78.0%) | [64.8, 87.2]% | 13.62/13.62/16.60 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P2/legacy/step_2/euler/dev/result.json` |
| legacy | P2/legacy | euler | 5 | 33/50 (66.0%) | [52.2, 77.6]% | 7.43/7.43/9.66 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P2/legacy/step_5/euler/dev/result.json` |
| legacy | P2/legacy | euler | 10 | 36/50 (72.0%) | [58.3, 82.5]% | 10.09/10.09/12.37 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P2/legacy/step_10/euler/dev/result.json` |
| legacy | P2/legacy | euler | 16 | 33/50 (66.0%) | [52.2, 77.6]% | 7.14/7.14/8.21 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P2/legacy/step_16/euler/dev/result.json` |
| legacy | P2/legacy | euler | 32 | 32/50 (64.0%) | [50.1, 75.9]% | 7.75/7.75/8.92 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P2/legacy/step_32/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 1 | 37/50 (74.0%) | [60.4, 84.1]% | 0.19/0.19/0.24 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P3/not_applicable/step_1/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 2 | 45/50 (90.0%) | [78.6, 95.7]% | 0.18/0.18/0.25 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P3/not_applicable/step_2/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 5 | 42/50 (84.0%) | [71.5, 91.7]% | 0.19/0.19/0.25 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P3/not_applicable/step_5/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 10 | 43/50 (86.0%) | [73.8, 93.0]% | 0.23/0.23/0.31 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P3/not_applicable/step_10/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 16 | 39/50 (78.0%) | [64.8, 87.2]% | 0.32/0.32/0.42 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P3/not_applicable/step_16/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 32 | 42/50 (84.0%) | [71.5, 91.7]% | 0.74/0.74/0.96 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/reacher/P3/not_applicable/step_32/euler/dev/result.json` |

#### step 子表

| cohort | mode/protocol | step1 | step2 | step5 | step10 | step16 | step32 |
|---|---|---|---|---|---|---|---|
| legacy | P0/not_applicable | 1: 36/50 (72.0%)<br>CI [58.3, 82.5]% | 2: 38/50 (76.0%)<br>CI [62.6, 85.7]% | 5: 37/50 (74.0%)<br>CI [60.4, 84.1]% | 10: 39/50 (78.0%)<br>CI [64.8, 87.2]% | 16: 37/50 (74.0%)<br>CI [60.4, 84.1]% | 32: 39/50 (78.0%)<br>CI [64.8, 87.2]% |
| legacy | P2/legacy | 1: 40/50 (80.0%)<br>CI [67.0, 88.8]% | 2: 39/50 (78.0%)<br>CI [64.8, 87.2]% | 5: 33/50 (66.0%)<br>CI [52.2, 77.6]% | 10: 36/50 (72.0%)<br>CI [58.3, 82.5]% | 16: 33/50 (66.0%)<br>CI [52.2, 77.6]% | 32: 32/50 (64.0%)<br>CI [50.1, 75.9]% |
| legacy | P2/cem-clip | 1: 44/50 (88.0%)<br>CI [76.2, 94.4]% | 2: 43/50 (86.0%)<br>CI [73.8, 93.0]% | 5: 45/50 (90.0%)<br>CI [78.6, 95.7]% | 10: 44/50 (88.0%)<br>CI [76.2, 94.4]% | 16: 42/50 (84.0%)<br>CI [71.5, 91.7]% | 32: 40/50 (80.0%)<br>CI [67.0, 88.8]% |
| legacy | P2/cem-scale | 1: 43/50 (86.0%)<br>CI [73.8, 93.0]% | 2: 44/50 (88.0%)<br>CI [76.2, 94.4]% | 5: 41/50 (82.0%)<br>CI [69.2, 90.2]% | 10: 43/50 (86.0%)<br>CI [73.8, 93.0]% | 16: 41/50 (82.0%)<br>CI [69.2, 90.2]% | 32: 44/50 (88.0%)<br>CI [76.2, 94.4]% |
| legacy | P3/not_applicable | 1: 37/50 (74.0%)<br>CI [60.4, 84.1]% | 2: 45/50 (90.0%)<br>CI [78.6, 95.7]% | 5: 42/50 (84.0%)<br>CI [71.5, 91.7]% | 10: 43/50 (86.0%)<br>CI [73.8, 93.0]% | 16: 39/50 (78.0%)<br>CI [64.8, 87.2]% | 32: 42/50 (84.0%)<br>CI [71.5, 91.7]% |
| dev | P0/not_applicable | 1: 45/50 (90.0%)<br>CI [78.6, 95.7]% | 2: 42/50 (84.0%)<br>CI [71.5, 91.7]% | 5: 40/50 (80.0%)<br>CI [67.0, 88.8]% | 10: 35/50 (70.0%)<br>CI [56.2, 80.9]% | 16: 35/50 (70.0%)<br>CI [56.2, 80.9]% | 32: 34/50 (68.0%)<br>CI [54.2, 79.2]% |
| dev | P2/legacy | 1: 46/50 (92.0%)<br>CI [81.2, 96.8]% | 2: 37/50 (74.0%)<br>CI [60.4, 84.1]% | 5: 32/50 (64.0%)<br>CI [50.1, 75.9]% | 10: 31/50 (62.0%)<br>CI [48.2, 74.1]% | 16: 25/50 (50.0%)<br>CI [36.6, 63.4]% | 32: 28/50 (56.0%)<br>CI [42.3, 68.8]% |
| dev | P2/cem-clip | 1: 40/50 (80.0%)<br>CI [67.0, 88.8]% | 2: 42/50 (84.0%)<br>CI [71.5, 91.7]% | 5: 42/50 (84.0%)<br>CI [71.5, 91.7]% | 10: 45/50 (90.0%)<br>CI [78.6, 95.7]% | 16: 40/50 (80.0%)<br>CI [67.0, 88.8]% | 32: 45/50 (90.0%)<br>CI [78.6, 95.7]% |
| dev | P2/cem-scale | 1: 46/50 (92.0%)<br>CI [81.2, 96.8]% | 2: 46/50 (92.0%)<br>CI [81.2, 96.8]% | 5: 45/50 (90.0%)<br>CI [78.6, 95.7]% | 10: 45/50 (90.0%)<br>CI [78.6, 95.7]% | 16: 46/50 (92.0%)<br>CI [81.2, 96.8]% | 32: 45/50 (90.0%)<br>CI [78.6, 95.7]% |
| dev | P3/not_applicable | 1: 46/50 (92.0%)<br>CI [81.2, 96.8]% | 2: 45/50 (90.0%)<br>CI [78.6, 95.7]% | 5: 40/50 (80.0%)<br>CI [67.0, 88.8]% | 10: 36/50 (72.0%)<br>CI [58.3, 82.5]% | 16: 43/50 (86.0%)<br>CI [73.8, 93.0]% | 32: 42/50 (84.0%)<br>CI [71.5, 91.7]% |

#### mode/protocol 子表

| cohort | step | P0/not_applicable | P1/legacy | P1/cem-clip | P1/cem-scale | P2/legacy | P2/cem-clip | P2/cem-scale | P3/not_applicable |
|---|---|---|---|---|---|---|---|---|---|
| legacy | inv | — | inv: 36/50 (72.0%)<br>CI [58.3, 82.5]% | inv: 46/50 (92.0%)<br>CI [81.2, 96.8]% | inv: 42/50 (84.0%)<br>CI [71.5, 91.7]% | — | — | — | — |
| legacy | 1 | 1: 36/50 (72.0%)<br>CI [58.3, 82.5]% | — | — | — | 1: 40/50 (80.0%)<br>CI [67.0, 88.8]% | 1: 44/50 (88.0%)<br>CI [76.2, 94.4]% | 1: 43/50 (86.0%)<br>CI [73.8, 93.0]% | 1: 37/50 (74.0%)<br>CI [60.4, 84.1]% |
| legacy | 2 | 2: 38/50 (76.0%)<br>CI [62.6, 85.7]% | — | — | — | 2: 39/50 (78.0%)<br>CI [64.8, 87.2]% | 2: 43/50 (86.0%)<br>CI [73.8, 93.0]% | 2: 44/50 (88.0%)<br>CI [76.2, 94.4]% | 2: 45/50 (90.0%)<br>CI [78.6, 95.7]% |
| legacy | 5 | 5: 37/50 (74.0%)<br>CI [60.4, 84.1]% | — | — | — | 5: 33/50 (66.0%)<br>CI [52.2, 77.6]% | 5: 45/50 (90.0%)<br>CI [78.6, 95.7]% | 5: 41/50 (82.0%)<br>CI [69.2, 90.2]% | 5: 42/50 (84.0%)<br>CI [71.5, 91.7]% |
| legacy | 10 | 10: 39/50 (78.0%)<br>CI [64.8, 87.2]% | — | — | — | 10: 36/50 (72.0%)<br>CI [58.3, 82.5]% | 10: 44/50 (88.0%)<br>CI [76.2, 94.4]% | 10: 43/50 (86.0%)<br>CI [73.8, 93.0]% | 10: 43/50 (86.0%)<br>CI [73.8, 93.0]% |
| legacy | 16 | 16: 37/50 (74.0%)<br>CI [60.4, 84.1]% | — | — | — | 16: 33/50 (66.0%)<br>CI [52.2, 77.6]% | 16: 42/50 (84.0%)<br>CI [71.5, 91.7]% | 16: 41/50 (82.0%)<br>CI [69.2, 90.2]% | 16: 39/50 (78.0%)<br>CI [64.8, 87.2]% |
| legacy | 32 | 32: 39/50 (78.0%)<br>CI [64.8, 87.2]% | — | — | — | 32: 32/50 (64.0%)<br>CI [50.1, 75.9]% | 32: 40/50 (80.0%)<br>CI [67.0, 88.8]% | 32: 44/50 (88.0%)<br>CI [76.2, 94.4]% | 32: 42/50 (84.0%)<br>CI [71.5, 91.7]% |
| dev | inv | — | inv: 37/50 (74.0%)<br>CI [60.4, 84.1]% | inv: 44/50 (88.0%)<br>CI [76.2, 94.4]% | inv: 49/50 (98.0%)<br>CI [89.5, 99.6]% | — | — | — | — |
| dev | 1 | 1: 45/50 (90.0%)<br>CI [78.6, 95.7]% | — | — | — | 1: 46/50 (92.0%)<br>CI [81.2, 96.8]% | 1: 40/50 (80.0%)<br>CI [67.0, 88.8]% | 1: 46/50 (92.0%)<br>CI [81.2, 96.8]% | 1: 46/50 (92.0%)<br>CI [81.2, 96.8]% |
| dev | 2 | 2: 42/50 (84.0%)<br>CI [71.5, 91.7]% | — | — | — | 2: 37/50 (74.0%)<br>CI [60.4, 84.1]% | 2: 42/50 (84.0%)<br>CI [71.5, 91.7]% | 2: 46/50 (92.0%)<br>CI [81.2, 96.8]% | 2: 45/50 (90.0%)<br>CI [78.6, 95.7]% |
| dev | 5 | 5: 40/50 (80.0%)<br>CI [67.0, 88.8]% | — | — | — | 5: 32/50 (64.0%)<br>CI [50.1, 75.9]% | 5: 42/50 (84.0%)<br>CI [71.5, 91.7]% | 5: 45/50 (90.0%)<br>CI [78.6, 95.7]% | 5: 40/50 (80.0%)<br>CI [67.0, 88.8]% |
| dev | 10 | 10: 35/50 (70.0%)<br>CI [56.2, 80.9]% | — | — | — | 10: 31/50 (62.0%)<br>CI [48.2, 74.1]% | 10: 45/50 (90.0%)<br>CI [78.6, 95.7]% | 10: 45/50 (90.0%)<br>CI [78.6, 95.7]% | 10: 36/50 (72.0%)<br>CI [58.3, 82.5]% |
| dev | 16 | 16: 35/50 (70.0%)<br>CI [56.2, 80.9]% | — | — | — | 16: 25/50 (50.0%)<br>CI [36.6, 63.4]% | 16: 40/50 (80.0%)<br>CI [67.0, 88.8]% | 16: 46/50 (92.0%)<br>CI [81.2, 96.8]% | 16: 43/50 (86.0%)<br>CI [73.8, 93.0]% |
| dev | 32 | 32: 34/50 (68.0%)<br>CI [54.2, 79.2]% | — | — | — | 32: 28/50 (56.0%)<br>CI [42.3, 68.8]% | 32: 45/50 (90.0%)<br>CI [78.6, 95.7]% | 32: 45/50 (90.0%)<br>CI [78.6, 95.7]% | 32: 42/50 (84.0%)<br>CI [71.5, 91.7]% |

### tworoom

| cohort | mode/protocol | integrator | step | success | Wilson 95% | planning mean/median/P95 (s) | result |
|---|---|---|---:|---:|---|---:|---|
| dev | P0/not_applicable | euler | 1 | 49/50 (98.0%) | [89.5, 99.6]% | 0.19/0.19/0.29 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P0/not_applicable/step_1/euler/dev/result.json` |
| dev | P0/not_applicable | euler | 2 | 49/50 (98.0%) | [89.5, 99.6]% | 0.02/0.02/0.03 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P0/not_applicable/step_2/euler/dev/result.json` |
| dev | P0/not_applicable | euler | 5 | 49/50 (98.0%) | [89.5, 99.6]% | 0.03/0.03/0.04 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P0/not_applicable/step_5/euler/dev/result.json` |
| dev | P0/not_applicable | euler | 10 | 48/50 (96.0%) | [86.5, 98.9]% | 0.05/0.05/0.05 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P0/not_applicable/step_10/euler/dev/result.json` |
| dev | P0/not_applicable | euler | 16 | 48/50 (96.0%) | [86.5, 98.9]% | 0.07/0.07/0.07 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P0/not_applicable/step_16/euler/dev/result.json` |
| dev | P0/not_applicable | euler | 32 | 48/50 (96.0%) | [86.5, 98.9]% | 0.12/0.12/0.13 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P0/not_applicable/step_32/euler/dev/result.json` |
| dev | P1/legacy | not_applicable | inv | 49/50 (98.0%) | [89.5, 99.6]% | 4.38/4.38/7.47 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P1/legacy/invariant/not_applicable/dev/result.json` |
| dev | P2/cem-clip | euler | 1 | 50/50 (100.0%) | [92.9, 100.0]% | 14.92/14.92/14.92 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P2/cem-clip/step_1/euler/dev/result.json` |
| dev | P2/cem-clip | euler | 2 | 50/50 (100.0%) | [92.9, 100.0]% | 18.10/18.10/18.10 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P2/cem-clip/step_2/euler/dev/result.json` |
| dev | P2/cem-clip | euler | 5 | 50/50 (100.0%) | [92.9, 100.0]% | 12.07/12.07/22.55 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P2/cem-clip/step_5/euler/dev/result.json` |
| dev | P2/cem-clip | euler | 10 | 50/50 (100.0%) | [92.9, 100.0]% | 15.41/15.41/28.96 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P2/cem-clip/step_10/euler/dev/result.json` |
| dev | P2/cem-clip | euler | 16 | 50/50 (100.0%) | [92.9, 100.0]% | 11.41/11.41/21.08 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P2/cem-clip/step_16/euler/dev/result.json` |
| dev | P2/cem-clip | euler | 32 | 50/50 (100.0%) | [92.9, 100.0]% | 13.30/13.30/24.36 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P2/cem-clip/step_32/euler/dev/result.json` |
| dev | P2/legacy | euler | 1 | 50/50 (100.0%) | [92.9, 100.0]% | 17.39/17.39/17.39 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P2/legacy/step_1/euler/dev/result.json` |
| dev | P2/legacy | euler | 2 | 50/50 (100.0%) | [92.9, 100.0]% | 9.65/9.65/16.72 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P2/legacy/step_2/euler/dev/result.json` |
| dev | P2/legacy | euler | 5 | 48/50 (96.0%) | [86.5, 98.9]% | 6.40/6.40/11.00 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P2/legacy/step_5/euler/dev/result.json` |
| dev | P2/legacy | euler | 10 | 48/50 (96.0%) | [86.5, 98.9]% | 3.47/3.47/5.83 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P2/legacy/step_10/euler/dev/result.json` |
| dev | P2/legacy | euler | 16 | 50/50 (100.0%) | [92.9, 100.0]% | 4.26/4.26/7.19 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P2/legacy/step_16/euler/dev/result.json` |
| dev | P2/legacy | euler | 32 | 47/50 (94.0%) | [83.8, 97.9]% | 6.97/6.97/9.17 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P2/legacy/step_32/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 1 | 49/50 (98.0%) | [89.5, 99.6]% | 0.12/0.12/0.22 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P3/not_applicable/step_1/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 2 | 50/50 (100.0%) | [92.9, 100.0]% | 0.21/0.21/0.21 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P3/not_applicable/step_2/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 5 | 50/50 (100.0%) | [92.9, 100.0]% | 0.27/0.27/0.27 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P3/not_applicable/step_5/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 10 | 50/50 (100.0%) | [92.9, 100.0]% | 0.45/0.45/0.45 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P3/not_applicable/step_10/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 16 | 50/50 (100.0%) | [92.9, 100.0]% | 0.41/0.41/0.41 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P3/not_applicable/step_16/euler/dev/result.json` |
| dev | P3/not_applicable | euler | 32 | 50/50 (100.0%) | [92.9, 100.0]% | 0.60/0.60/0.60 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_3_seed3072_dev/conditions/tworoom/P3/not_applicable/step_32/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 1 | 50/50 (100.0%) | [92.9, 100.0]% | 0.40/0.40/0.40 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P0/not_applicable/step_1/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 2 | 48/50 (96.0%) | [86.5, 98.9]% | 0.07/0.07/0.08 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P0/not_applicable/step_2/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 5 | 49/50 (98.0%) | [89.5, 99.6]% | 0.21/0.21/0.32 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P0/not_applicable/step_5/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 10 | 49/50 (98.0%) | [89.5, 99.6]% | 0.05/0.05/0.05 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P0/not_applicable/step_10/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 16 | 49/50 (98.0%) | [89.5, 99.6]% | 0.06/0.06/0.08 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P0/not_applicable/step_16/euler/dev/result.json` |
| legacy | P0/not_applicable | euler | 32 | 48/50 (96.0%) | [86.5, 98.9]% | 0.11/0.11/0.12 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P0/not_applicable/step_32/euler/dev/result.json` |
| legacy | P1/legacy | not_applicable | inv | 50/50 (100.0%) | [92.9, 100.0]% | 8.52/8.52/15.83 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P1/legacy/invariant/not_applicable/dev/result.json` |
| legacy | P2/cem-clip | euler | 1 | 50/50 (100.0%) | [92.9, 100.0]% | 74.77/74.77/74.77 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P2/cem-clip/step_1/euler/dev/result.json` |
| legacy | P2/cem-clip | euler | 2 | 50/50 (100.0%) | [92.9, 100.0]% | 59.62/59.62/59.62 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P2/cem-clip/step_2/euler/dev/result.json` |
| legacy | P2/cem-clip | euler | 5 | 50/50 (100.0%) | [92.9, 100.0]% | 20.21/20.21/20.21 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P2/cem-clip/step_5/euler/dev/result.json` |
| legacy | P2/cem-clip | euler | 10 | 50/50 (100.0%) | [92.9, 100.0]% | 19.43/19.43/19.43 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P2/cem-clip/step_10/euler/dev/result.json` |
| legacy | P2/cem-clip | euler | 16 | 50/50 (100.0%) | [92.9, 100.0]% | 19.75/19.75/19.75 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P2/cem-clip/step_16/euler/dev/result.json` |
| legacy | P2/cem-clip | euler | 32 | 50/50 (100.0%) | [92.9, 100.0]% | 19.73/19.73/19.73 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P2/cem-clip/step_32/euler/dev/result.json` |
| legacy | P2/legacy | euler | 1 | 50/50 (100.0%) | [92.9, 100.0]% | 14.73/14.73/14.73 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P2/legacy/step_1/euler/dev/result.json` |
| legacy | P2/legacy | euler | 2 | 50/50 (100.0%) | [92.9, 100.0]% | 14.95/14.95/14.95 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P2/legacy/step_2/euler/dev/result.json` |
| legacy | P2/legacy | euler | 5 | 50/50 (100.0%) | [92.9, 100.0]% | 3.25/3.25/5.94 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P2/legacy/step_5/euler/dev/result.json` |
| legacy | P2/legacy | euler | 10 | 50/50 (100.0%) | [92.9, 100.0]% | 3.46/3.46/6.02 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P2/legacy/step_10/euler/dev/result.json` |
| legacy | P2/legacy | euler | 16 | 49/50 (98.0%) | [89.5, 99.6]% | 3.55/3.55/6.41 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P2/legacy/step_16/euler/dev/result.json` |
| legacy | P2/legacy | euler | 32 | 50/50 (100.0%) | [92.9, 100.0]% | 3.89/3.89/6.86 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P2/legacy/step_32/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 1 | 50/50 (100.0%) | [92.9, 100.0]% | 0.25/0.25/0.25 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P3/not_applicable/step_1/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 2 | 50/50 (100.0%) | [92.9, 100.0]% | 0.26/0.26/0.26 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P3/not_applicable/step_2/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 5 | 50/50 (100.0%) | [92.9, 100.0]% | 0.27/0.27/0.27 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P3/not_applicable/step_5/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 10 | 50/50 (100.0%) | [92.9, 100.0]% | 0.32/0.32/0.32 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P3/not_applicable/step_10/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 16 | 50/50 (100.0%) | [92.9, 100.0]% | 0.47/0.47/0.47 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P3/not_applicable/step_16/euler/dev/result.json` |
| legacy | P3/not_applicable | euler | 32 | 50/50 (100.0%) | [92.9, 100.0]% | 0.61/0.61/0.61 | `/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy/tworoom/P3/not_applicable/step_32/euler/dev/result.json` |

#### step 子表

| cohort | mode/protocol | step1 | step2 | step5 | step10 | step16 | step32 |
|---|---|---|---|---|---|---|---|
| legacy | P0/not_applicable | 1: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 2: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 5: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 10: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 16: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 32: 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| legacy | P2/legacy | 1: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 5: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 10: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 16: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 32: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| legacy | P2/cem-clip | 1: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 5: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 10: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 16: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 32: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| legacy | P3/not_applicable | 1: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 5: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 10: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 16: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 32: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| dev | P0/not_applicable | 1: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 2: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 5: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 10: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 16: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 32: 48/50 (96.0%)<br>CI [86.5, 98.9]% |
| dev | P2/legacy | 1: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 5: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 10: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 16: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 32: 47/50 (94.0%)<br>CI [83.8, 97.9]% |
| dev | P2/cem-clip | 1: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 5: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 10: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 16: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 32: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| dev | P3/not_applicable | 1: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 5: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 10: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 16: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 32: 50/50 (100.0%)<br>CI [92.9, 100.0]% |

#### mode/protocol 子表

| cohort | step | P0/not_applicable | P1/legacy | P1/cem-clip | P1/cem-scale | P2/legacy | P2/cem-clip | P2/cem-scale | P3/not_applicable |
|---|---|---|---|---|---|---|---|---|---|
| legacy | inv | — | inv: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | — | — | — | — | — |
| legacy | 1 | 1: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | — | — | 1: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 1: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | 1: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| legacy | 2 | 2: 48/50 (96.0%)<br>CI [86.5, 98.9]% | — | — | — | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| legacy | 5 | 5: 49/50 (98.0%)<br>CI [89.5, 99.6]% | — | — | — | 5: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 5: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | 5: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| legacy | 10 | 10: 49/50 (98.0%)<br>CI [89.5, 99.6]% | — | — | — | 10: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 10: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | 10: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| legacy | 16 | 16: 49/50 (98.0%)<br>CI [89.5, 99.6]% | — | — | — | 16: 49/50 (98.0%)<br>CI [89.5, 99.6]% | 16: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | 16: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| legacy | 32 | 32: 48/50 (96.0%)<br>CI [86.5, 98.9]% | — | — | — | 32: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 32: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | 32: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| dev | inv | — | inv: 49/50 (98.0%)<br>CI [89.5, 99.6]% | — | — | — | — | — | — |
| dev | 1 | 1: 49/50 (98.0%)<br>CI [89.5, 99.6]% | — | — | — | 1: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 1: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | 1: 49/50 (98.0%)<br>CI [89.5, 99.6]% |
| dev | 2 | 2: 49/50 (98.0%)<br>CI [89.5, 99.6]% | — | — | — | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | 2: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| dev | 5 | 5: 49/50 (98.0%)<br>CI [89.5, 99.6]% | — | — | — | 5: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 5: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | 5: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| dev | 10 | 10: 48/50 (96.0%)<br>CI [86.5, 98.9]% | — | — | — | 10: 48/50 (96.0%)<br>CI [86.5, 98.9]% | 10: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | 10: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| dev | 16 | 16: 48/50 (96.0%)<br>CI [86.5, 98.9]% | — | — | — | 16: 50/50 (100.0%)<br>CI [92.9, 100.0]% | 16: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | 16: 50/50 (100.0%)<br>CI [92.9, 100.0]% |
| dev | 32 | 32: 48/50 (96.0%)<br>CI [86.5, 98.9]% | — | — | — | 32: 47/50 (94.0%)<br>CI [83.8, 97.9]% | 32: 50/50 (100.0%)<br>CI [92.9, 100.0]% | — | 32: 50/50 (100.0%)<br>CI [92.9, 100.0]% |


## 四任务平均

本节把同一 condition 在注册了它的任务上做宏平均（每个任务等权），并给出合并成功数与 pooled Wilson 区间。
cem-clip 仅覆盖 Reacher/TwoRoom，cem-scale 仅覆盖 Reacher，因此不同行的参与任务数可能不同，`tasks` 列给出参与任务数供核对。
注意：四个任务的难度、成功谓词和动作维度不同，跨任务平均只作纵向汇总，绝对值不能与单任务或论文数字直接等同。

### 四任务平均总表

| cohort | mode/protocol | step | tasks | mean success | pooled success | pooled Wilson 95% |
|---|---|---:|---:|---:|---:|---|
| dev | P0/not_applicable | 1 | 4 | 96.0% | 192/200 | [92.3, 98.0]% |
| dev | P0/not_applicable | 2 | 4 | 95.5% | 191/200 | [91.7, 97.6]% |
| dev | P0/not_applicable | 5 | 4 | 93.5% | 187/200 | [89.2, 96.2]% |
| dev | P0/not_applicable | 10 | 4 | 90.5% | 181/200 | [85.6, 93.8]% |
| dev | P0/not_applicable | 16 | 4 | 89.5% | 179/200 | [84.5, 93.0]% |
| dev | P0/not_applicable | 32 | 4 | 89.0% | 178/200 | [83.9, 92.6]% |
| dev | P1/legacy | inv | 4 | 75.5% | 151/200 | [69.1, 80.9]% |
| dev | P1/cem-clip | inv | 1 | 88.0% | 44/50 | [76.2, 94.4]% |
| dev | P1/cem-scale | inv | 1 | 98.0% | 49/50 | [89.5, 99.6]% |
| dev | P2/legacy | 1 | 4 | 96.0% | 192/200 | [92.3, 98.0]% |
| dev | P2/legacy | 2 | 4 | 91.0% | 182/200 | [86.2, 94.2]% |
| dev | P2/legacy | 5 | 4 | 88.0% | 176/200 | [82.8, 91.8]% |
| dev | P2/legacy | 10 | 4 | 87.5% | 175/200 | [82.2, 91.4]% |
| dev | P2/legacy | 16 | 4 | 84.5% | 169/200 | [78.8, 88.9]% |
| dev | P2/legacy | 32 | 4 | 85.5% | 171/200 | [80.0, 89.7]% |
| dev | P2/cem-clip | 1 | 2 | 90.0% | 90/100 | [82.6, 94.5]% |
| dev | P2/cem-clip | 2 | 2 | 92.0% | 92/100 | [85.0, 95.9]% |
| dev | P2/cem-clip | 5 | 2 | 92.0% | 92/100 | [85.0, 95.9]% |
| dev | P2/cem-clip | 10 | 2 | 95.0% | 95/100 | [88.8, 97.8]% |
| dev | P2/cem-clip | 16 | 2 | 90.0% | 90/100 | [82.6, 94.5]% |
| dev | P2/cem-clip | 32 | 2 | 95.0% | 95/100 | [88.8, 97.8]% |
| dev | P2/cem-scale | 1 | 1 | 92.0% | 46/50 | [81.2, 96.8]% |
| dev | P2/cem-scale | 2 | 1 | 92.0% | 46/50 | [81.2, 96.8]% |
| dev | P2/cem-scale | 5 | 1 | 90.0% | 45/50 | [78.6, 95.7]% |
| dev | P2/cem-scale | 10 | 1 | 90.0% | 45/50 | [78.6, 95.7]% |
| dev | P2/cem-scale | 16 | 1 | 92.0% | 46/50 | [81.2, 96.8]% |
| dev | P2/cem-scale | 32 | 1 | 90.0% | 45/50 | [78.6, 95.7]% |
| dev | P3/not_applicable | 1 | 4 | 97.0% | 194/200 | [93.6, 98.6]% |
| dev | P3/not_applicable | 2 | 4 | 97.5% | 195/200 | [94.3, 98.9]% |
| dev | P3/not_applicable | 5 | 4 | 94.5% | 189/200 | [90.4, 96.9]% |
| dev | P3/not_applicable | 10 | 4 | 92.0% | 184/200 | [87.4, 95.0]% |
| dev | P3/not_applicable | 16 | 4 | 95.5% | 191/200 | [91.7, 97.6]% |
| dev | P3/not_applicable | 32 | 4 | 95.0% | 190/200 | [91.0, 97.3]% |
| legacy | P0/not_applicable | 1 | 4 | 92.0% | 184/200 | [87.4, 95.0]% |
| legacy | P0/not_applicable | 2 | 4 | 93.0% | 186/200 | [88.6, 95.8]% |
| legacy | P0/not_applicable | 5 | 4 | 92.5% | 185/200 | [88.0, 95.4]% |
| legacy | P0/not_applicable | 10 | 4 | 92.5% | 185/200 | [88.0, 95.4]% |
| legacy | P0/not_applicable | 16 | 4 | 91.5% | 183/200 | [86.8, 94.6]% |
| legacy | P0/not_applicable | 32 | 4 | 92.0% | 184/200 | [87.4, 95.0]% |
| legacy | P1/legacy | inv | 4 | 83.0% | 166/200 | [77.2, 87.6]% |
| legacy | P1/cem-clip | inv | 1 | 92.0% | 46/50 | [81.2, 96.8]% |
| legacy | P1/cem-scale | inv | 1 | 84.0% | 42/50 | [71.5, 91.7]% |
| legacy | P2/legacy | 1 | 4 | 94.0% | 188/200 | [89.8, 96.5]% |
| legacy | P2/legacy | 2 | 4 | 92.5% | 185/200 | [88.0, 95.4]% |
| legacy | P2/legacy | 5 | 4 | 89.5% | 179/200 | [84.5, 93.0]% |
| legacy | P2/legacy | 10 | 4 | 91.0% | 182/200 | [86.2, 94.2]% |
| legacy | P2/legacy | 16 | 4 | 89.5% | 179/200 | [84.5, 93.0]% |
| legacy | P2/legacy | 32 | 4 | 89.5% | 179/200 | [84.5, 93.0]% |
| legacy | P2/cem-clip | 1 | 2 | 94.0% | 94/100 | [87.5, 97.2]% |
| legacy | P2/cem-clip | 2 | 2 | 93.0% | 93/100 | [86.3, 96.6]% |
| legacy | P2/cem-clip | 5 | 2 | 95.0% | 95/100 | [88.8, 97.8]% |
| legacy | P2/cem-clip | 10 | 2 | 94.0% | 94/100 | [87.5, 97.2]% |
| legacy | P2/cem-clip | 16 | 2 | 92.0% | 92/100 | [85.0, 95.9]% |
| legacy | P2/cem-clip | 32 | 2 | 90.0% | 90/100 | [82.6, 94.5]% |
| legacy | P2/cem-scale | 1 | 1 | 86.0% | 43/50 | [73.8, 93.0]% |
| legacy | P2/cem-scale | 2 | 1 | 88.0% | 44/50 | [76.2, 94.4]% |
| legacy | P2/cem-scale | 5 | 1 | 82.0% | 41/50 | [69.2, 90.2]% |
| legacy | P2/cem-scale | 10 | 1 | 86.0% | 43/50 | [73.8, 93.0]% |
| legacy | P2/cem-scale | 16 | 1 | 82.0% | 41/50 | [69.2, 90.2]% |
| legacy | P2/cem-scale | 32 | 1 | 88.0% | 44/50 | [76.2, 94.4]% |
| legacy | P3/not_applicable | 1 | 4 | 92.5% | 185/200 | [88.0, 95.4]% |
| legacy | P3/not_applicable | 2 | 4 | 97.0% | 194/200 | [93.6, 98.6]% |
| legacy | P3/not_applicable | 5 | 4 | 96.0% | 192/200 | [92.3, 98.0]% |
| legacy | P3/not_applicable | 10 | 4 | 96.0% | 192/200 | [92.3, 98.0]% |
| legacy | P3/not_applicable | 16 | 4 | 93.5% | 187/200 | [89.2, 96.2]% |
| legacy | P3/not_applicable | 32 | 4 | 95.5% | 191/200 | [91.7, 97.6]% |

### 四任务平均 step 子表

| cohort | mode/protocol | tasks | step1 | step2 | step5 | step10 | step16 | step32 |
|---|---|---:|---|---|---|---|---|---|
| legacy | P0/not_applicable | 4 | 92.0% | 93.0% | 92.5% | 92.5% | 91.5% | 92.0% |
| legacy | P2/legacy | 4 | 94.0% | 92.5% | 89.5% | 91.0% | 89.5% | 89.5% |
| legacy | P2/cem-clip | 2 | 94.0% | 93.0% | 95.0% | 94.0% | 92.0% | 90.0% |
| legacy | P2/cem-scale | 1 | 86.0% | 88.0% | 82.0% | 86.0% | 82.0% | 88.0% |
| legacy | P3/not_applicable | 4 | 92.5% | 97.0% | 96.0% | 96.0% | 93.5% | 95.5% |
| dev | P0/not_applicable | 4 | 96.0% | 95.5% | 93.5% | 90.5% | 89.5% | 89.0% |
| dev | P2/legacy | 4 | 96.0% | 91.0% | 88.0% | 87.5% | 84.5% | 85.5% |
| dev | P2/cem-clip | 2 | 90.0% | 92.0% | 92.0% | 95.0% | 90.0% | 95.0% |
| dev | P2/cem-scale | 1 | 92.0% | 92.0% | 90.0% | 90.0% | 92.0% | 90.0% |
| dev | P3/not_applicable | 4 | 97.0% | 97.5% | 94.5% | 92.0% | 95.5% | 95.0% |

### 四任务平均 mode/protocol 子表

| cohort | step | P0/not_applicable | P1/legacy | P1/cem-clip | P1/cem-scale | P2/legacy | P2/cem-clip | P2/cem-scale | P3/not_applicable |
|---|---|---|---|---|---|---|---|---|---|
| legacy | inv | — | 83.0% | 92.0% | 84.0% | — | — | — | — |
| legacy | 1 | 92.0% | — | — | — | 94.0% | 94.0% | 86.0% | 92.5% |
| legacy | 2 | 93.0% | — | — | — | 92.5% | 93.0% | 88.0% | 97.0% |
| legacy | 5 | 92.5% | — | — | — | 89.5% | 95.0% | 82.0% | 96.0% |
| legacy | 10 | 92.5% | — | — | — | 91.0% | 94.0% | 86.0% | 96.0% |
| legacy | 16 | 91.5% | — | — | — | 89.5% | 92.0% | 82.0% | 93.5% |
| legacy | 32 | 92.0% | — | — | — | 89.5% | 90.0% | 88.0% | 95.5% |
| dev | inv | — | 75.5% | 88.0% | 98.0% | — | — | — | — |
| dev | 1 | 96.0% | — | — | — | 96.0% | 90.0% | 92.0% | 97.0% |
| dev | 2 | 95.5% | — | — | — | 91.0% | 92.0% | 92.0% | 97.5% |
| dev | 5 | 93.5% | — | — | — | 88.0% | 92.0% | 90.0% | 94.5% |
| dev | 10 | 90.5% | — | — | — | 87.5% | 95.0% | 90.0% | 92.0% |
| dev | 16 | 89.5% | — | — | — | 84.5% | 90.0% | 92.0% | 95.5% |
| dev | 32 | 89.0% | — | — | — | 85.5% | 95.0% | 90.0% | 95.0% |

## 同一 cohort 内 paired 比较

这些比较只在同一 cohort 的相同 episode identity 上进行，因此可以报告 exact McNemar；每张表按 category 排序，step_vs_step16 与 protocol 对照可分别筛读。

| category | comparison | Δ success (pp) | improved | regressed | net | exact McNemar p |
|---|---|---:|---:|---:|---:|---:|
| cem_clip_vs_legacy | `dev/reacher/P1/cem-clip_vs_legacy` | 14.0 | 10 | 3 | 7 | 0.092285 |
| cem_clip_vs_legacy | `dev/reacher/P2/cem_clip_vs_legacy/step_1` | -12.0 | 1 | 7 | -6 | 0.070312 |
| cem_clip_vs_legacy | `dev/reacher/P2/cem_clip_vs_legacy/step_10` | 28.0 | 16 | 2 | 14 | 0.001312 |
| cem_clip_vs_legacy | `dev/reacher/P2/cem_clip_vs_legacy/step_16` | 30.0 | 18 | 3 | 15 | 0.001490 |
| cem_clip_vs_legacy | `dev/reacher/P2/cem_clip_vs_legacy/step_2` | 10.0 | 9 | 4 | 5 | 0.266846 |
| cem_clip_vs_legacy | `dev/reacher/P2/cem_clip_vs_legacy/step_32` | 34.0 | 18 | 1 | 17 | 0.000076 |
| cem_clip_vs_legacy | `dev/reacher/P2/cem_clip_vs_legacy/step_5` | 20.0 | 16 | 6 | 10 | 0.052479 |
| cem_clip_vs_legacy | `dev/tworoom/P2/cem_clip_vs_legacy/step_1` | 0.0 | 0 | 0 | 0 | — |
| cem_clip_vs_legacy | `dev/tworoom/P2/cem_clip_vs_legacy/step_10` | 4.0 | 2 | 0 | 2 | 0.500000 |
| cem_clip_vs_legacy | `dev/tworoom/P2/cem_clip_vs_legacy/step_16` | 0.0 | 0 | 0 | 0 | — |
| cem_clip_vs_legacy | `dev/tworoom/P2/cem_clip_vs_legacy/step_2` | 0.0 | 0 | 0 | 0 | — |
| cem_clip_vs_legacy | `dev/tworoom/P2/cem_clip_vs_legacy/step_32` | 6.0 | 3 | 0 | 3 | 0.250000 |
| cem_clip_vs_legacy | `dev/tworoom/P2/cem_clip_vs_legacy/step_5` | 4.0 | 2 | 0 | 2 | 0.500000 |
| cem_clip_vs_legacy | `legacy/reacher/P1/cem-clip_vs_legacy` | 20.0 | 12 | 2 | 10 | 0.012939 |
| cem_clip_vs_legacy | `legacy/reacher/P2/cem_clip_vs_legacy/step_1` | 8.0 | 7 | 3 | 4 | 0.343750 |
| cem_clip_vs_legacy | `legacy/reacher/P2/cem_clip_vs_legacy/step_10` | 16.0 | 10 | 2 | 8 | 0.038574 |
| cem_clip_vs_legacy | `legacy/reacher/P2/cem_clip_vs_legacy/step_16` | 18.0 | 11 | 2 | 9 | 0.022461 |
| cem_clip_vs_legacy | `legacy/reacher/P2/cem_clip_vs_legacy/step_2` | 8.0 | 7 | 3 | 4 | 0.343750 |
| cem_clip_vs_legacy | `legacy/reacher/P2/cem_clip_vs_legacy/step_32` | 16.0 | 13 | 5 | 8 | 0.096252 |
| cem_clip_vs_legacy | `legacy/reacher/P2/cem_clip_vs_legacy/step_5` | 24.0 | 13 | 1 | 12 | 0.001831 |
| cem_clip_vs_legacy | `legacy/tworoom/P2/cem_clip_vs_legacy/step_1` | 0.0 | 0 | 0 | 0 | — |
| cem_clip_vs_legacy | `legacy/tworoom/P2/cem_clip_vs_legacy/step_10` | 0.0 | 0 | 0 | 0 | — |
| cem_clip_vs_legacy | `legacy/tworoom/P2/cem_clip_vs_legacy/step_16` | 2.0 | 1 | 0 | 1 | 1.000000 |
| cem_clip_vs_legacy | `legacy/tworoom/P2/cem_clip_vs_legacy/step_2` | 0.0 | 0 | 0 | 0 | — |
| cem_clip_vs_legacy | `legacy/tworoom/P2/cem_clip_vs_legacy/step_32` | 0.0 | 0 | 0 | 0 | — |
| cem_clip_vs_legacy | `legacy/tworoom/P2/cem_clip_vs_legacy/step_5` | 0.0 | 0 | 0 | 0 | — |
| cem_scale_vs_cem_clip | `dev/reacher/P1/cem_scale_vs_cem_clip` | 10.0 | 6 | 1 | 5 | 0.125000 |
| cem_scale_vs_cem_clip | `dev/reacher/P2/cem_scale_vs_cem_clip/step_1` | 12.0 | 7 | 1 | 6 | 0.070312 |
| cem_scale_vs_cem_clip | `dev/reacher/P2/cem_scale_vs_cem_clip/step_10` | 0.0 | 4 | 4 | 0 | 1.000000 |
| cem_scale_vs_cem_clip | `dev/reacher/P2/cem_scale_vs_cem_clip/step_16` | 12.0 | 8 | 2 | 6 | 0.109375 |
| cem_scale_vs_cem_clip | `dev/reacher/P2/cem_scale_vs_cem_clip/step_2` | 8.0 | 5 | 1 | 4 | 0.218750 |
| cem_scale_vs_cem_clip | `dev/reacher/P2/cem_scale_vs_cem_clip/step_32` | 0.0 | 4 | 4 | 0 | 1.000000 |
| cem_scale_vs_cem_clip | `dev/reacher/P2/cem_scale_vs_cem_clip/step_5` | 6.0 | 4 | 1 | 3 | 0.375000 |
| cem_scale_vs_cem_clip | `legacy/reacher/P1/cem_scale_vs_cem_clip` | -8.0 | 2 | 6 | -4 | 0.289062 |
| cem_scale_vs_cem_clip | `legacy/reacher/P2/cem_scale_vs_cem_clip/step_1` | -2.0 | 3 | 4 | -1 | 1.000000 |
| cem_scale_vs_cem_clip | `legacy/reacher/P2/cem_scale_vs_cem_clip/step_10` | -2.0 | 2 | 3 | -1 | 1.000000 |
| cem_scale_vs_cem_clip | `legacy/reacher/P2/cem_scale_vs_cem_clip/step_16` | -2.0 | 4 | 5 | -1 | 1.000000 |
| cem_scale_vs_cem_clip | `legacy/reacher/P2/cem_scale_vs_cem_clip/step_2` | 2.0 | 5 | 4 | 1 | 1.000000 |
| cem_scale_vs_cem_clip | `legacy/reacher/P2/cem_scale_vs_cem_clip/step_32` | 8.0 | 6 | 2 | 4 | 0.289062 |
| cem_scale_vs_cem_clip | `legacy/reacher/P2/cem_scale_vs_cem_clip/step_5` | -8.0 | 1 | 5 | -4 | 0.218750 |
| cem_scale_vs_legacy | `dev/reacher/P1/cem-scale_vs_legacy` | 24.0 | 13 | 1 | 12 | 0.001831 |
| cem_scale_vs_legacy | `dev/reacher/P2/cem_scale_vs_legacy/step_1` | 0.0 | 3 | 3 | 0 | 1.000000 |
| cem_scale_vs_legacy | `dev/reacher/P2/cem_scale_vs_legacy/step_10` | 28.0 | 16 | 2 | 14 | 0.001312 |
| cem_scale_vs_legacy | `dev/reacher/P2/cem_scale_vs_legacy/step_16` | 42.0 | 23 | 2 | 21 | 0.000019 |
| cem_scale_vs_legacy | `dev/reacher/P2/cem_scale_vs_legacy/step_2` | 18.0 | 10 | 1 | 9 | 0.011719 |
| cem_scale_vs_legacy | `dev/reacher/P2/cem_scale_vs_legacy/step_32` | 34.0 | 18 | 1 | 17 | 0.000076 |
| cem_scale_vs_legacy | `dev/reacher/P2/cem_scale_vs_legacy/step_5` | 26.0 | 15 | 2 | 13 | 0.002350 |
| cem_scale_vs_legacy | `legacy/reacher/P1/cem-scale_vs_legacy` | 12.0 | 11 | 5 | 6 | 0.210114 |
| cem_scale_vs_legacy | `legacy/reacher/P2/cem_scale_vs_legacy/step_1` | 6.0 | 8 | 5 | 3 | 0.581055 |
| cem_scale_vs_legacy | `legacy/reacher/P2/cem_scale_vs_legacy/step_10` | 14.0 | 9 | 2 | 7 | 0.065430 |
| cem_scale_vs_legacy | `legacy/reacher/P2/cem_scale_vs_legacy/step_16` | 16.0 | 13 | 5 | 8 | 0.096252 |
| cem_scale_vs_legacy | `legacy/reacher/P2/cem_scale_vs_legacy/step_2` | 10.0 | 9 | 4 | 5 | 0.266846 |
| cem_scale_vs_legacy | `legacy/reacher/P2/cem_scale_vs_legacy/step_32` | 24.0 | 14 | 2 | 12 | 0.004181 |
| cem_scale_vs_legacy | `legacy/reacher/P2/cem_scale_vs_legacy/step_5` | 16.0 | 12 | 4 | 8 | 0.076813 |
| step_vs_step16 | `dev/cube/P0/not_applicable/step_10_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/cube/P0/not_applicable/step_1_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/cube/P0/not_applicable/step_2_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/cube/P0/not_applicable/step_32_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/cube/P0/not_applicable/step_5_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/cube/P2/legacy/step_10_vs_step_16` | 4.0 | 2 | 0 | 2 | 0.500000 |
| step_vs_step16 | `dev/cube/P2/legacy/step_1_vs_step_16` | 2.0 | 2 | 1 | 1 | 1.000000 |
| step_vs_step16 | `dev/cube/P2/legacy/step_2_vs_step_16` | 2.0 | 2 | 1 | 1 | 1.000000 |
| step_vs_step16 | `dev/cube/P2/legacy/step_32_vs_step_16` | 4.0 | 2 | 0 | 2 | 0.500000 |
| step_vs_step16 | `dev/cube/P2/legacy/step_5_vs_step_16` | 2.0 | 1 | 0 | 1 | 1.000000 |
| step_vs_step16 | `dev/cube/P3/not_applicable/step_10_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/cube/P3/not_applicable/step_1_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/cube/P3/not_applicable/step_2_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/cube/P3/not_applicable/step_32_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/cube/P3/not_applicable/step_5_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/pusht/P0/not_applicable/step_10_vs_step_16` | 4.0 | 2 | 0 | 2 | 0.500000 |
| step_vs_step16 | `dev/pusht/P0/not_applicable/step_1_vs_step_16` | 4.0 | 3 | 1 | 2 | 0.625000 |
| step_vs_step16 | `dev/pusht/P0/not_applicable/step_2_vs_step_16` | 8.0 | 4 | 0 | 4 | 0.125000 |
| step_vs_step16 | `dev/pusht/P0/not_applicable/step_32_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/pusht/P0/not_applicable/step_5_vs_step_16` | 4.0 | 2 | 0 | 2 | 0.500000 |
| step_vs_step16 | `dev/pusht/P2/legacy/step_10_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/pusht/P2/legacy/step_1_vs_step_16` | 2.0 | 1 | 0 | 1 | 1.000000 |
| step_vs_step16 | `dev/pusht/P2/legacy/step_2_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/pusht/P2/legacy/step_32_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/pusht/P2/legacy/step_5_vs_step_16` | 2.0 | 1 | 0 | 1 | 1.000000 |
| step_vs_step16 | `dev/pusht/P3/not_applicable/step_10_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/pusht/P3/not_applicable/step_1_vs_step_16` | 2.0 | 2 | 1 | 1 | 1.000000 |
| step_vs_step16 | `dev/pusht/P3/not_applicable/step_2_vs_step_16` | 4.0 | 2 | 0 | 2 | 0.500000 |
| step_vs_step16 | `dev/pusht/P3/not_applicable/step_32_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/pusht/P3/not_applicable/step_5_vs_step_16` | 2.0 | 1 | 0 | 1 | 1.000000 |
| step_vs_step16 | `dev/reacher/P0/not_applicable/step_10_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/reacher/P0/not_applicable/step_1_vs_step_16` | 20.0 | 14 | 4 | 10 | 0.030884 |
| step_vs_step16 | `dev/reacher/P0/not_applicable/step_2_vs_step_16` | 14.0 | 11 | 4 | 7 | 0.118469 |
| step_vs_step16 | `dev/reacher/P0/not_applicable/step_32_vs_step_16` | -2.0 | 0 | 1 | -1 | 1.000000 |
| step_vs_step16 | `dev/reacher/P0/not_applicable/step_5_vs_step_16` | 10.0 | 7 | 2 | 5 | 0.179688 |
| step_vs_step16 | `dev/reacher/P2/cem-clip/step_10_vs_step_16` | 10.0 | 7 | 2 | 5 | 0.179688 |
| step_vs_step16 | `dev/reacher/P2/cem-clip/step_1_vs_step_16` | 0.0 | 7 | 7 | 0 | 1.000000 |
| step_vs_step16 | `dev/reacher/P2/cem-clip/step_2_vs_step_16` | 4.0 | 7 | 5 | 2 | 0.774414 |
| step_vs_step16 | `dev/reacher/P2/cem-clip/step_32_vs_step_16` | 10.0 | 7 | 2 | 5 | 0.179688 |
| step_vs_step16 | `dev/reacher/P2/cem-clip/step_5_vs_step_16` | 4.0 | 7 | 5 | 2 | 0.774414 |
| step_vs_step16 | `dev/reacher/P2/cem-scale/step_10_vs_step_16` | -2.0 | 1 | 2 | -1 | 1.000000 |
| step_vs_step16 | `dev/reacher/P2/cem-scale/step_1_vs_step_16` | 0.0 | 2 | 2 | 0 | 1.000000 |
| step_vs_step16 | `dev/reacher/P2/cem-scale/step_2_vs_step_16` | 0.0 | 1 | 1 | 0 | 1.000000 |
| step_vs_step16 | `dev/reacher/P2/cem-scale/step_32_vs_step_16` | -2.0 | 3 | 4 | -1 | 1.000000 |
| step_vs_step16 | `dev/reacher/P2/cem-scale/step_5_vs_step_16` | -2.0 | 3 | 4 | -1 | 1.000000 |
| step_vs_step16 | `dev/reacher/P2/legacy/step_10_vs_step_16` | 12.0 | 12 | 6 | 6 | 0.237885 |
| step_vs_step16 | `dev/reacher/P2/legacy/step_1_vs_step_16` | 42.0 | 24 | 3 | 21 | 0.000049 |
| step_vs_step16 | `dev/reacher/P2/legacy/step_2_vs_step_16` | 24.0 | 17 | 5 | 12 | 0.016901 |
| step_vs_step16 | `dev/reacher/P2/legacy/step_32_vs_step_16` | 6.0 | 12 | 9 | 3 | 0.663624 |
| step_vs_step16 | `dev/reacher/P2/legacy/step_5_vs_step_16` | 14.0 | 17 | 10 | 7 | 0.247789 |
| step_vs_step16 | `dev/reacher/P3/not_applicable/step_10_vs_step_16` | -14.0 | 2 | 9 | -7 | 0.065430 |
| step_vs_step16 | `dev/reacher/P3/not_applicable/step_1_vs_step_16` | 6.0 | 6 | 3 | 3 | 0.507812 |
| step_vs_step16 | `dev/reacher/P3/not_applicable/step_2_vs_step_16` | 4.0 | 6 | 4 | 2 | 0.753906 |
| step_vs_step16 | `dev/reacher/P3/not_applicable/step_32_vs_step_16` | -2.0 | 3 | 4 | -1 | 1.000000 |
| step_vs_step16 | `dev/reacher/P3/not_applicable/step_5_vs_step_16` | -6.0 | 4 | 7 | -3 | 0.548828 |
| step_vs_step16 | `dev/tworoom/P0/not_applicable/step_10_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/tworoom/P0/not_applicable/step_1_vs_step_16` | 2.0 | 1 | 0 | 1 | 1.000000 |
| step_vs_step16 | `dev/tworoom/P0/not_applicable/step_2_vs_step_16` | 2.0 | 1 | 0 | 1 | 1.000000 |
| step_vs_step16 | `dev/tworoom/P0/not_applicable/step_32_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/tworoom/P0/not_applicable/step_5_vs_step_16` | 2.0 | 1 | 0 | 1 | 1.000000 |
| step_vs_step16 | `dev/tworoom/P2/cem-clip/step_10_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/tworoom/P2/cem-clip/step_1_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/tworoom/P2/cem-clip/step_2_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/tworoom/P2/cem-clip/step_32_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/tworoom/P2/cem-clip/step_5_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/tworoom/P2/legacy/step_10_vs_step_16` | -4.0 | 0 | 2 | -2 | 0.500000 |
| step_vs_step16 | `dev/tworoom/P2/legacy/step_1_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/tworoom/P2/legacy/step_2_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/tworoom/P2/legacy/step_32_vs_step_16` | -6.0 | 0 | 3 | -3 | 0.250000 |
| step_vs_step16 | `dev/tworoom/P2/legacy/step_5_vs_step_16` | -4.0 | 0 | 2 | -2 | 0.500000 |
| step_vs_step16 | `dev/tworoom/P3/not_applicable/step_10_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/tworoom/P3/not_applicable/step_1_vs_step_16` | -2.0 | 0 | 1 | -1 | 1.000000 |
| step_vs_step16 | `dev/tworoom/P3/not_applicable/step_2_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/tworoom/P3/not_applicable/step_32_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `dev/tworoom/P3/not_applicable/step_5_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/cube/P0/not_applicable/step_10_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/cube/P0/not_applicable/step_1_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/cube/P0/not_applicable/step_2_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/cube/P0/not_applicable/step_32_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/cube/P0/not_applicable/step_5_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/cube/P2/legacy/step_10_vs_step_16` | -2.0 | 0 | 1 | -1 | 1.000000 |
| step_vs_step16 | `legacy/cube/P2/legacy/step_1_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/cube/P2/legacy/step_2_vs_step_16` | -2.0 | 0 | 1 | -1 | 1.000000 |
| step_vs_step16 | `legacy/cube/P2/legacy/step_32_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/cube/P2/legacy/step_5_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/cube/P3/not_applicable/step_10_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/cube/P3/not_applicable/step_1_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/cube/P3/not_applicable/step_2_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/cube/P3/not_applicable/step_32_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/cube/P3/not_applicable/step_5_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/pusht/P0/not_applicable/step_10_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/pusht/P0/not_applicable/step_1_vs_step_16` | 2.0 | 3 | 2 | 1 | 1.000000 |
| step_vs_step16 | `legacy/pusht/P0/not_applicable/step_2_vs_step_16` | 6.0 | 3 | 0 | 3 | 0.250000 |
| step_vs_step16 | `legacy/pusht/P0/not_applicable/step_32_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/pusht/P0/not_applicable/step_5_vs_step_16` | 4.0 | 2 | 0 | 2 | 0.500000 |
| step_vs_step16 | `legacy/pusht/P2/legacy/step_10_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/pusht/P2/legacy/step_1_vs_step_16` | 2.0 | 1 | 0 | 1 | 1.000000 |
| step_vs_step16 | `legacy/pusht/P2/legacy/step_2_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/pusht/P2/legacy/step_32_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/pusht/P2/legacy/step_5_vs_step_16` | -2.0 | 0 | 1 | -1 | 1.000000 |
| step_vs_step16 | `legacy/pusht/P3/not_applicable/step_10_vs_step_16` | 2.0 | 1 | 0 | 1 | 1.000000 |
| step_vs_step16 | `legacy/pusht/P3/not_applicable/step_1_vs_step_16` | 0.0 | 2 | 2 | 0 | 1.000000 |
| step_vs_step16 | `legacy/pusht/P3/not_applicable/step_2_vs_step_16` | 2.0 | 2 | 1 | 1 | 1.000000 |
| step_vs_step16 | `legacy/pusht/P3/not_applicable/step_32_vs_step_16` | 2.0 | 1 | 0 | 1 | 1.000000 |
| step_vs_step16 | `legacy/pusht/P3/not_applicable/step_5_vs_step_16` | 4.0 | 2 | 0 | 2 | 0.500000 |
| step_vs_step16 | `legacy/reacher/P0/not_applicable/step_10_vs_step_16` | 4.0 | 4 | 2 | 2 | 0.687500 |
| step_vs_step16 | `legacy/reacher/P0/not_applicable/step_1_vs_step_16` | -2.0 | 11 | 12 | -1 | 1.000000 |
| step_vs_step16 | `legacy/reacher/P0/not_applicable/step_2_vs_step_16` | 2.0 | 9 | 8 | 1 | 1.000000 |
| step_vs_step16 | `legacy/reacher/P0/not_applicable/step_32_vs_step_16` | 4.0 | 7 | 5 | 2 | 0.774414 |
| step_vs_step16 | `legacy/reacher/P0/not_applicable/step_5_vs_step_16` | 0.0 | 7 | 7 | 0 | 1.000000 |
| step_vs_step16 | `legacy/reacher/P2/cem-clip/step_10_vs_step_16` | 4.0 | 4 | 2 | 2 | 0.687500 |
| step_vs_step16 | `legacy/reacher/P2/cem-clip/step_1_vs_step_16` | 4.0 | 6 | 4 | 2 | 0.753906 |
| step_vs_step16 | `legacy/reacher/P2/cem-clip/step_2_vs_step_16` | 2.0 | 4 | 3 | 1 | 1.000000 |
| step_vs_step16 | `legacy/reacher/P2/cem-clip/step_32_vs_step_16` | -4.0 | 2 | 4 | -2 | 0.687500 |
| step_vs_step16 | `legacy/reacher/P2/cem-clip/step_5_vs_step_16` | 6.0 | 4 | 1 | 3 | 0.375000 |
| step_vs_step16 | `legacy/reacher/P2/cem-scale/step_10_vs_step_16` | 4.0 | 5 | 3 | 2 | 0.726562 |
| step_vs_step16 | `legacy/reacher/P2/cem-scale/step_1_vs_step_16` | 4.0 | 7 | 5 | 2 | 0.774414 |
| step_vs_step16 | `legacy/reacher/P2/cem-scale/step_2_vs_step_16` | 6.0 | 6 | 3 | 3 | 0.507812 |
| step_vs_step16 | `legacy/reacher/P2/cem-scale/step_32_vs_step_16` | 6.0 | 4 | 1 | 3 | 0.375000 |
| step_vs_step16 | `legacy/reacher/P2/cem-scale/step_5_vs_step_16` | 0.0 | 4 | 4 | 0 | 1.000000 |
| step_vs_step16 | `legacy/reacher/P2/legacy/step_10_vs_step_16` | 6.0 | 8 | 5 | 3 | 0.581055 |
| step_vs_step16 | `legacy/reacher/P2/legacy/step_1_vs_step_16` | 14.0 | 15 | 8 | 7 | 0.210040 |
| step_vs_step16 | `legacy/reacher/P2/legacy/step_2_vs_step_16` | 12.0 | 10 | 4 | 6 | 0.179565 |
| step_vs_step16 | `legacy/reacher/P2/legacy/step_32_vs_step_16` | -2.0 | 9 | 10 | -1 | 1.000000 |
| step_vs_step16 | `legacy/reacher/P2/legacy/step_5_vs_step_16` | 0.0 | 7 | 7 | 0 | 1.000000 |
| step_vs_step16 | `legacy/reacher/P3/not_applicable/step_10_vs_step_16` | 8.0 | 8 | 4 | 4 | 0.387695 |
| step_vs_step16 | `legacy/reacher/P3/not_applicable/step_1_vs_step_16` | -4.0 | 8 | 10 | -2 | 0.814529 |
| step_vs_step16 | `legacy/reacher/P3/not_applicable/step_2_vs_step_16` | 12.0 | 9 | 3 | 6 | 0.145996 |
| step_vs_step16 | `legacy/reacher/P3/not_applicable/step_32_vs_step_16` | 6.0 | 7 | 4 | 3 | 0.548828 |
| step_vs_step16 | `legacy/reacher/P3/not_applicable/step_5_vs_step_16` | 6.0 | 9 | 6 | 3 | 0.607239 |
| step_vs_step16 | `legacy/tworoom/P0/not_applicable/step_10_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/tworoom/P0/not_applicable/step_1_vs_step_16` | 2.0 | 1 | 0 | 1 | 1.000000 |
| step_vs_step16 | `legacy/tworoom/P0/not_applicable/step_2_vs_step_16` | -2.0 | 1 | 2 | -1 | 1.000000 |
| step_vs_step16 | `legacy/tworoom/P0/not_applicable/step_32_vs_step_16` | -2.0 | 0 | 1 | -1 | 1.000000 |
| step_vs_step16 | `legacy/tworoom/P0/not_applicable/step_5_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/tworoom/P2/cem-clip/step_10_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/tworoom/P2/cem-clip/step_1_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/tworoom/P2/cem-clip/step_2_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/tworoom/P2/cem-clip/step_32_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/tworoom/P2/cem-clip/step_5_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/tworoom/P2/legacy/step_10_vs_step_16` | 2.0 | 1 | 0 | 1 | 1.000000 |
| step_vs_step16 | `legacy/tworoom/P2/legacy/step_1_vs_step_16` | 2.0 | 1 | 0 | 1 | 1.000000 |
| step_vs_step16 | `legacy/tworoom/P2/legacy/step_2_vs_step_16` | 2.0 | 1 | 0 | 1 | 1.000000 |
| step_vs_step16 | `legacy/tworoom/P2/legacy/step_32_vs_step_16` | 2.0 | 1 | 0 | 1 | 1.000000 |
| step_vs_step16 | `legacy/tworoom/P2/legacy/step_5_vs_step_16` | 2.0 | 1 | 0 | 1 | 1.000000 |
| step_vs_step16 | `legacy/tworoom/P3/not_applicable/step_10_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/tworoom/P3/not_applicable/step_1_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/tworoom/P3/not_applicable/step_2_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/tworoom/P3/not_applicable/step_32_vs_step_16` | 0.0 | 0 | 0 | 0 | — |
| step_vs_step16 | `legacy/tworoom/P3/not_applicable/step_5_vs_step_16` | 0.0 | 0 | 0 | 0 | — |

## legacy 与 dev 的 unpaired 对照

两套 manifest 的 cohort ID/hash 不同；以下表格同时列出两边的成功率和 Wilson 区间，并给出 dev 相对 legacy 的 signed/absolute difference。禁止把这些行解释为 paired McNemar。

| task | condition | legacy success / Wilson | dev success / Wilson | dev−legacy (pp) | absolute difference (pp) |
|---|---|---|---|---:|---:|
| cube | `P0/not_applicable/1/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| cube | `P0/not_applicable/2/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| cube | `P0/not_applicable/5/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| cube | `P0/not_applicable/10/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| cube | `P0/not_applicable/16/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| cube | `P0/not_applicable/32/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| cube | `P1/legacy/None/not_applicable` | 70.0% [56.2, 80.9]% | 40.0% [27.6, 53.8]% | -30.0 | 30.0 |
| cube | `P2/legacy/1/euler` | 98.0% [89.5, 99.6]% | 98.0% [89.5, 99.6]% | 0.0 | 0.0 |
| cube | `P2/legacy/2/euler` | 96.0% [86.5, 98.9]% | 98.0% [89.5, 99.6]% | 2.0 | 2.0 |
| cube | `P2/legacy/5/euler` | 98.0% [89.5, 99.6]% | 98.0% [89.5, 99.6]% | 0.0 | 0.0 |
| cube | `P2/legacy/10/euler` | 96.0% [86.5, 98.9]% | 100.0% [92.9, 100.0]% | 4.0 | 4.0 |
| cube | `P2/legacy/16/euler` | 98.0% [89.5, 99.6]% | 96.0% [86.5, 98.9]% | -2.0 | 2.0 |
| cube | `P2/legacy/32/euler` | 98.0% [89.5, 99.6]% | 100.0% [92.9, 100.0]% | 2.0 | 2.0 |
| cube | `P3/not_applicable/1/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| cube | `P3/not_applicable/2/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| cube | `P3/not_applicable/5/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| cube | `P3/not_applicable/10/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| cube | `P3/not_applicable/16/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| cube | `P3/not_applicable/32/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| pusht | `P0/not_applicable/1/euler` | 96.0% [86.5, 98.9]% | 96.0% [86.5, 98.9]% | 0.0 | 0.0 |
| pusht | `P0/not_applicable/2/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| pusht | `P0/not_applicable/5/euler` | 98.0% [89.5, 99.6]% | 96.0% [86.5, 98.9]% | -2.0 | 2.0 |
| pusht | `P0/not_applicable/10/euler` | 94.0% [83.8, 97.9]% | 96.0% [86.5, 98.9]% | 2.0 | 2.0 |
| pusht | `P0/not_applicable/16/euler` | 94.0% [83.8, 97.9]% | 92.0% [81.2, 96.8]% | -2.0 | 2.0 |
| pusht | `P0/not_applicable/32/euler` | 94.0% [83.8, 97.9]% | 92.0% [81.2, 96.8]% | -2.0 | 2.0 |
| pusht | `P1/legacy/None/not_applicable` | 90.0% [78.6, 95.7]% | 90.0% [78.6, 95.7]% | 0.0 | 0.0 |
| pusht | `P2/legacy/1/euler` | 98.0% [89.5, 99.6]% | 94.0% [83.8, 97.9]% | -4.0 | 4.0 |
| pusht | `P2/legacy/2/euler` | 96.0% [86.5, 98.9]% | 92.0% [81.2, 96.8]% | -4.0 | 4.0 |
| pusht | `P2/legacy/5/euler` | 94.0% [83.8, 97.9]% | 94.0% [83.8, 97.9]% | 0.0 | 0.0 |
| pusht | `P2/legacy/10/euler` | 96.0% [86.5, 98.9]% | 92.0% [81.2, 96.8]% | -4.0 | 4.0 |
| pusht | `P2/legacy/16/euler` | 96.0% [86.5, 98.9]% | 92.0% [81.2, 96.8]% | -4.0 | 4.0 |
| pusht | `P2/legacy/32/euler` | 96.0% [86.5, 98.9]% | 92.0% [81.2, 96.8]% | -4.0 | 4.0 |
| pusht | `P3/not_applicable/1/euler` | 96.0% [86.5, 98.9]% | 98.0% [89.5, 99.6]% | 2.0 | 2.0 |
| pusht | `P3/not_applicable/2/euler` | 98.0% [89.5, 99.6]% | 100.0% [92.9, 100.0]% | 2.0 | 2.0 |
| pusht | `P3/not_applicable/5/euler` | 100.0% [92.9, 100.0]% | 98.0% [89.5, 99.6]% | -2.0 | 2.0 |
| pusht | `P3/not_applicable/10/euler` | 98.0% [89.5, 99.6]% | 96.0% [86.5, 98.9]% | -2.0 | 2.0 |
| pusht | `P3/not_applicable/16/euler` | 96.0% [86.5, 98.9]% | 96.0% [86.5, 98.9]% | 0.0 | 0.0 |
| pusht | `P3/not_applicable/32/euler` | 98.0% [89.5, 99.6]% | 96.0% [86.5, 98.9]% | -2.0 | 2.0 |
| reacher | `P0/not_applicable/1/euler` | 72.0% [58.3, 82.5]% | 90.0% [78.6, 95.7]% | 18.0 | 18.0 |
| reacher | `P0/not_applicable/2/euler` | 76.0% [62.6, 85.7]% | 84.0% [71.5, 91.7]% | 8.0 | 8.0 |
| reacher | `P0/not_applicable/5/euler` | 74.0% [60.4, 84.1]% | 80.0% [67.0, 88.8]% | 6.0 | 6.0 |
| reacher | `P0/not_applicable/10/euler` | 78.0% [64.8, 87.2]% | 70.0% [56.2, 80.9]% | -8.0 | 8.0 |
| reacher | `P0/not_applicable/16/euler` | 74.0% [60.4, 84.1]% | 70.0% [56.2, 80.9]% | -4.0 | 4.0 |
| reacher | `P0/not_applicable/32/euler` | 78.0% [64.8, 87.2]% | 68.0% [54.2, 79.2]% | -10.0 | 10.0 |
| reacher | `P1/legacy/None/not_applicable` | 72.0% [58.3, 82.5]% | 74.0% [60.4, 84.1]% | 2.0 | 2.0 |
| reacher | `P2/legacy/1/euler` | 80.0% [67.0, 88.8]% | 92.0% [81.2, 96.8]% | 12.0 | 12.0 |
| reacher | `P2/legacy/2/euler` | 78.0% [64.8, 87.2]% | 74.0% [60.4, 84.1]% | -4.0 | 4.0 |
| reacher | `P2/legacy/5/euler` | 66.0% [52.2, 77.6]% | 64.0% [50.1, 75.9]% | -2.0 | 2.0 |
| reacher | `P2/legacy/10/euler` | 72.0% [58.3, 82.5]% | 62.0% [48.2, 74.1]% | -10.0 | 10.0 |
| reacher | `P2/legacy/16/euler` | 66.0% [52.2, 77.6]% | 50.0% [36.6, 63.4]% | -16.0 | 16.0 |
| reacher | `P2/legacy/32/euler` | 64.0% [50.1, 75.9]% | 56.0% [42.3, 68.8]% | -8.0 | 8.0 |
| reacher | `P3/not_applicable/1/euler` | 74.0% [60.4, 84.1]% | 92.0% [81.2, 96.8]% | 18.0 | 18.0 |
| reacher | `P3/not_applicable/2/euler` | 90.0% [78.6, 95.7]% | 90.0% [78.6, 95.7]% | 0.0 | 0.0 |
| reacher | `P3/not_applicable/5/euler` | 84.0% [71.5, 91.7]% | 80.0% [67.0, 88.8]% | -4.0 | 4.0 |
| reacher | `P3/not_applicable/10/euler` | 86.0% [73.8, 93.0]% | 72.0% [58.3, 82.5]% | -14.0 | 14.0 |
| reacher | `P3/not_applicable/16/euler` | 78.0% [64.8, 87.2]% | 86.0% [73.8, 93.0]% | 8.0 | 8.0 |
| reacher | `P3/not_applicable/32/euler` | 84.0% [71.5, 91.7]% | 84.0% [71.5, 91.7]% | 0.0 | 0.0 |
| reacher | `P1/cem-clip/None/not_applicable` | 92.0% [81.2, 96.8]% | 88.0% [76.2, 94.4]% | -4.0 | 4.0 |
| reacher | `P2/cem-clip/1/euler` | 88.0% [76.2, 94.4]% | 80.0% [67.0, 88.8]% | -8.0 | 8.0 |
| reacher | `P2/cem-clip/2/euler` | 86.0% [73.8, 93.0]% | 84.0% [71.5, 91.7]% | -2.0 | 2.0 |
| reacher | `P2/cem-clip/5/euler` | 90.0% [78.6, 95.7]% | 84.0% [71.5, 91.7]% | -6.0 | 6.0 |
| reacher | `P2/cem-clip/10/euler` | 88.0% [76.2, 94.4]% | 90.0% [78.6, 95.7]% | 2.0 | 2.0 |
| reacher | `P2/cem-clip/16/euler` | 84.0% [71.5, 91.7]% | 80.0% [67.0, 88.8]% | -4.0 | 4.0 |
| reacher | `P2/cem-clip/32/euler` | 80.0% [67.0, 88.8]% | 90.0% [78.6, 95.7]% | 10.0 | 10.0 |
| reacher | `P1/cem-scale/None/not_applicable` | 84.0% [71.5, 91.7]% | 98.0% [89.5, 99.6]% | 14.0 | 14.0 |
| reacher | `P2/cem-scale/1/euler` | 86.0% [73.8, 93.0]% | 92.0% [81.2, 96.8]% | 6.0 | 6.0 |
| reacher | `P2/cem-scale/2/euler` | 88.0% [76.2, 94.4]% | 92.0% [81.2, 96.8]% | 4.0 | 4.0 |
| reacher | `P2/cem-scale/5/euler` | 82.0% [69.2, 90.2]% | 90.0% [78.6, 95.7]% | 8.0 | 8.0 |
| reacher | `P2/cem-scale/10/euler` | 86.0% [73.8, 93.0]% | 90.0% [78.6, 95.7]% | 4.0 | 4.0 |
| reacher | `P2/cem-scale/16/euler` | 82.0% [69.2, 90.2]% | 92.0% [81.2, 96.8]% | 10.0 | 10.0 |
| reacher | `P2/cem-scale/32/euler` | 88.0% [76.2, 94.4]% | 90.0% [78.6, 95.7]% | 2.0 | 2.0 |
| tworoom | `P0/not_applicable/1/euler` | 100.0% [92.9, 100.0]% | 98.0% [89.5, 99.6]% | -2.0 | 2.0 |
| tworoom | `P0/not_applicable/2/euler` | 96.0% [86.5, 98.9]% | 98.0% [89.5, 99.6]% | 2.0 | 2.0 |
| tworoom | `P0/not_applicable/5/euler` | 98.0% [89.5, 99.6]% | 98.0% [89.5, 99.6]% | 0.0 | 0.0 |
| tworoom | `P0/not_applicable/10/euler` | 98.0% [89.5, 99.6]% | 96.0% [86.5, 98.9]% | -2.0 | 2.0 |
| tworoom | `P0/not_applicable/16/euler` | 98.0% [89.5, 99.6]% | 96.0% [86.5, 98.9]% | -2.0 | 2.0 |
| tworoom | `P0/not_applicable/32/euler` | 96.0% [86.5, 98.9]% | 96.0% [86.5, 98.9]% | 0.0 | 0.0 |
| tworoom | `P1/legacy/None/not_applicable` | 100.0% [92.9, 100.0]% | 98.0% [89.5, 99.6]% | -2.0 | 2.0 |
| tworoom | `P2/legacy/1/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| tworoom | `P2/legacy/2/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| tworoom | `P2/legacy/5/euler` | 100.0% [92.9, 100.0]% | 96.0% [86.5, 98.9]% | -4.0 | 4.0 |
| tworoom | `P2/legacy/10/euler` | 100.0% [92.9, 100.0]% | 96.0% [86.5, 98.9]% | -4.0 | 4.0 |
| tworoom | `P2/legacy/16/euler` | 98.0% [89.5, 99.6]% | 100.0% [92.9, 100.0]% | 2.0 | 2.0 |
| tworoom | `P2/legacy/32/euler` | 100.0% [92.9, 100.0]% | 94.0% [83.8, 97.9]% | -6.0 | 6.0 |
| tworoom | `P3/not_applicable/1/euler` | 100.0% [92.9, 100.0]% | 98.0% [89.5, 99.6]% | -2.0 | 2.0 |
| tworoom | `P3/not_applicable/2/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| tworoom | `P3/not_applicable/5/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| tworoom | `P3/not_applicable/10/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| tworoom | `P3/not_applicable/16/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| tworoom | `P3/not_applicable/32/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| tworoom | `P2/cem-clip/1/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| tworoom | `P2/cem-clip/2/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| tworoom | `P2/cem-clip/5/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| tworoom | `P2/cem-clip/10/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| tworoom | `P2/cem-clip/16/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |
| tworoom | `P2/cem-clip/32/euler` | 100.0% [92.9, 100.0]% | 100.0% [92.9, 100.0]% | 0.0 | 0.0 |

## 方向性结论

- step 相对 step16：63 个正向、20 个负向、67 个持平（比较数 150）。
- cem-clip 相对 legacy：17 个正向、1 个负向、8 个持平（比较数 26）。
- cem-scale 相对 legacy：13 个正向、0 个负向、1 个持平（比较数 14）。
- cem-scale 相对 cem-clip：7 个正向、5 个负向、2 个持平（比较数 14）。
- dev 相对 legacy cohort：24 个正向、36 个负向、36 个持平（比较数 96）。
- 判断原则：只有当同一 task/mode/protocol 的 legacy 与 dev signed difference 改变方向时，才称为 cohort 改变了该方向性结论；本报告不把不同 cohort 的 episode-level 成败强行配对。
- 这些是固定 checkpoint、单 seed、每 cohort 50 episodes 的描述性结果，不外推到 final 或其他训练 seed。

## 产物索引

- 分析 JSON：`/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/analysis/analysis.json`
- 条件 CSV：`/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/analysis/conditions.csv`
- 比较 CSV：`/data/users/wenxin/pre-exp/le-wm/outputs/round4/phase45_4_seed3072_legacy_dev/analysis/comparisons.csv`
- 本报告：`/data/users/wenxin/pre-exp/le-wm/docs/report/round4/round4_phase45-4_legacy_protocol_comparison_report.md`
- 代码 commit（生成时）：`cc0b2ec5c7939b55b836c861b568e29685f75a9d`
