# Round 3 Phase 2 在线后训练实验报告

## 1. 实验定义

Phase 2 采用已经训练完成的 epoch-10 checkpoint 做 post-training fine-tuning，
不是从零训练。每 100 个真实环境步触发一次更新；每次更新使用固定 64 条
replay minibatch。Offline 臂使用 64 条离线样本，Online 臂使用 32 条离线样本
和 32 条已执行在线样本。所有在线 transition 都持久化到 append-only replay，
但不会在一次更新中全部消费。

三臂为：

1. `freeze`：不更新；
2. `offline_continue`：从 E0 epoch10 继续离线更新；
3. `online_adapt`：从同一 E0 epoch10 权重开始，以 1:1 混合离线/在线 replay 更新。

开发评测固定使用 `round3_revised` 的 50-episode cohort，cohort SHA256 为
`578f6ea52e56d77461cb5bb2aeb01a004a21b9f6659689596f6f55f25dc17d01`。

## 2. E0 Cube 结果

成功率如下，列为环境步数 `0 / 5,000 / 10,000 / 20,000`：

| 臂 | 0 | 5k | 10k | 20k |
|---|---:|---:|---:|---:|
| Freeze | 54% | 54% | 54% | 54% |
| Offline continue | 54% | 62% | 56% | 48% |
| Online adapt | 54% | 60% | 56% | 48% |

所有 12 个结果均为 `status=ok`，每项 50 episodes，使用同一 cohort。

20k 终点差值为：

- Online − Freeze：`−6.0 pp`；
- Online − Offline continue：`0.0 pp`。

因此 E0 的 Online 臂没有同时超过两个对照至少 5pp，不触发补种子、
200-episode 独立测试或第二任务扩展。该结果只说明在本预算、更新频率和
适应配置下未见 E0 在线收益，不外推为所有在线学习均无效。

运行验收记录：

- 环境步数：20,000；
- 更新边界：200；
- replay shard：200；
- 在线 replay 总行数：2,070；
- 最后一次更新：Offline 64 条、Online mixed 64 条（32 offline + 32 online）。

完整机器记录见
[`run_state.json`](../../../../outputs/round3/phase2/cube_e0_three_arms_100step/run_state.json)，
注册配置见 [`phase2_cube.json`](../../../../config/round3/phase2_cube.json)。

## 3. 后续 legacy E5 验证

由于 E0 未通过扩展门，按 Round 3 方案运行一次单任务、单种子的 legacy E5
验证。该实验直接从历史 E5 epoch-10 checkpoint 开始，使用 Fast-LeWAM 原生
Stage-B 几何（6 帧、5 个 25 维 action block），其余环境步预算、100-step
更新频率、batch64、评测节点和决策门保持一致。

实验注册见
[`phase2_fast_e5_legacy_cube.json`](../../../../config/round3/phase2_fast_e5_legacy_cube.json)。

## 4. Legacy E5 Cube 结果与决策

该实验已完成。它从历史 E5 epoch-10 checkpoint 开始，不是从零训练；三臂
继续使用同一 checkpoint，区别仅在于是否更新以及更新时的 replay 来源。

成功率如下，列为环境步数 `0 / 5,000 / 10,000 / 20,000`：

| 臂 | 0 | 5k | 10k | 20k |
|---|---:|---:|---:|---:|
| Freeze | 48% | 48% | 48% | 48% |
| Offline continue | 48% | 42% | 38% | 48% |
| Online adapt | 48% | 58% | 56% | 44% |

所有 12 个结果均为 `status=ok`，每项 50 episodes，并使用同一开发 cohort
（SHA256 为
`578f6ea52e56d77461cb5bb2aeb01a004a21b9f6659689596f6f55f25dc17d01`）。
这里的 SHA256 是评测结果中记录的 canonical cohort hash；注册配置中的
`cohorts.dev.sha256` 记录的是 cohort JSON 文件字节 hash，两者语义不同。

20k 终点差值为：

- Online − Freeze：`−4.0 pp`；
- Online − Offline continue：`−4.0 pp`。

因此 legacy E5 的 online 臂没有同时超过两个对照至少 5pp，未通过扩展门。
按计划不再补充 legacy E5 种子、200-episode 独立测试或第二个任务的 online
扩展。该结论限定为当前 Cube、单种子、20,000 环境步、每 100 步更新一次及
1:1 offline/online minibatch 配置，不外推为所有在线学习均无效。

运行验收记录：

- 环境步数：20,000；
- optimizer update：200；
- 有效 replay shard：174；
- 在线 replay 总行数：999；
- 26 个 100-step chunk 没有形成完整的 6-frame/5-action episode-local 窗口，
  未写入伪造 replay，但仍按精确环境步数推进并使用已积累 replay 完成更新。

完整机器记录见
[`run_state.json`](../../../../outputs/round3/phase2/cube_e5_legacy_three_arms_100step/run_state.json)，
评测结果位于
`../../outputs/round3/phase2/cube_e5_legacy_three_arms_100step/evals/`，
注册配置见
[`phase2_fast_e5_legacy_cube.json`](../../../../config/round3/phase2_fast_e5_legacy_cube.json)。

## 5. 1k 步进、200-episode 完整曲线

按后续实验请求，对 E0 与 legacy E5 的 `offline_continue` 和
`online_adapt` 中间 checkpoint，以每 1,000 个环境步为间隔，在冻结的
`round3_revised` final 200-episode cohort 上重新评测。步数包括
`0, 1k, 2k, ..., 20k`，共 21 个点、84 个 aggregate 结果；每个点实际以
4 个 50-episode batch 运行后合并，避免共享 GPU 上一次性创建 200 个
MuJoCo EGL renderer 导致 framebuffer 失败。所有 aggregate 结果均为
`status=ok`、200 episodes，使用同一 final cohort
（canonical SHA256 为
`26171642a8c6a3617b0c586c2d0f448f9d352aa433d6f8a169b8fffdecb0e030`）。

关键节点成功率如下：

| 实验 / 臂 | 0 | 5k | 10k | 20k |
|---|---:|---:|---:|---:|
| E0 Offline | 49.0% | 53.5% | 54.5% | 52.0% |
| E0 Online | 49.0% | 54.0% | 58.5% | 56.5% |
| E5 Offline | 47.0% | 49.5% | 46.0% | 48.0% |
| E5 Online | 47.0% | 58.0% | 57.0% | 60.0% |

在 final 200-episode endpoint：

- E0 Online − Offline：`+4.5 pp`；Online − 初始权重：`+7.5 pp`；
- E5 Online − Offline：`+12.0 pp`；Online − 初始权重：`+13.0 pp`。

这与前面的 50-episode dev 曲线不同：dev 集上 E5 Online 在 20k 为 44%，
而 final 200-episode cohort 上为 60%，没有观察到同样的后期退化。说明此前
“20k 退化”的现象具有明显 cohort/评测方差，不能直接概括为模型在 20k
过拟合。原先基于 50-episode dev 预注册的扩展门仍保留其历史语义；本节是
用户追加的 final 曲线诊断，不自动改写原决策门，也未据此启动额外 seed。

完整曲线文件：

- [E0 offline CSV](../../../../outputs/round3/phase2/curve_200ep_1k/e0/offline_continue/curve.csv)
- [E0 online CSV](../../../../outputs/round3/phase2/curve_200ep_1k/e0/online_adapt/curve.csv)
- [E5 offline CSV](../../../../outputs/round3/phase2/curve_200ep_1k/e5/offline_continue/curve.csv)
- [E5 online CSV](../../../../outputs/round3/phase2/curve_200ep_1k/e5/online_adapt/curve.csv)

每个曲线目录同时保留 aggregate `result.json`、4 个 batch 的原始结果、
cohort manifest 和 episode trace。

曲线图：

![Round 3 Phase 2 success curves](../../../../outputs/round3/phase2/curve_200ep_1k/round3_phase2_success_curves.png)

可下载矢量版本：[SVG](../../../../outputs/round3/phase2/curve_200ep_1k/round3_phase2_success_curves.svg)、
[PDF](../../../../outputs/round3/phase2/curve_200ep_1k/round3_phase2_success_curves.pdf)。

## 6. 剩余三任务：Reacher、Push-T、TwoRoom

根据后续实验请求，在 Cube 已完成的同一 Phase 2 协议上扩展其余三个任务。该扩展
保留原 Phase 1 冻结的任务判据、cohort、epoch-10 起点、100 环境步一次更新、
batch size 64、E0/E5 的 offline/online replay 语义和 20,000 环境步预算；每个
任务的每条曲线在同一冻结 final cohort 上按 1,000 步评测，共 21 个点，每点 200
episodes，实际以 4×50 batch 运行。

六个训练 run 均完成并通过验收：每个 run 为 `status=ok`、20,000 环境步、200
个 update boundary。TwoRoom legacy E5 的第一个 100-step chunk 没有形成完整的
6-frame/5-action online window，因此按“有效轨迹不足先积累、不伪造窗口”规则跳过
了该次 online optimizer update；该 chunk 仍保留精确环境步数和 checkpoint，之后
恢复正常 1:1 online/offline 更新。其余 run 没有跳过 online update。

### 6.1 关键节点成功率

每个单元格为 `0k → 5k → 10k → 20k`，单位为成功率百分比：

| 任务 | E0 Offline | E0 Online | Legacy E5 Offline | Legacy E5 Online |
|---|---:|---:|---:|---:|
| Reacher | 86.0 → 84.0 → 81.5 → 80.5 | 86.0 → 84.5 → 82.0 → 80.5 | 87.5 → 79.0 → 87.0 → 83.5 | 87.5 → 85.0 → 84.5 → 80.5 |
| Push-T | 93.5 → 91.5 → 92.5 → 93.5 | 93.5 → 70.0 → 83.5 → 94.0 | 88.5 → 83.5 → 87.5 → 88.0 | 89.0 → 85.0 → 85.5 → 84.0 |
| TwoRoom | 86.5 → 86.5 → 89.0 → 87.0 | 86.5 → 88.0 → 87.0 → 87.0 | 96.0 → 95.0 → 94.5 → 95.0 | 96.0 → 95.0 → 95.0 → 94.5 |

20k endpoint 的 `online − offline` 差值为：Reacher E0 `0.0pp`、E5 `−3.0pp`；
Push-T E0 `+0.5pp`、E5 `−4.0pp`；TwoRoom E0 `0.0pp`、E5 `−0.5pp`。
因此在这三个任务上，online adaptation 没有复现 Cube final cohort 中的 E5
endpoint 优势；Push-T E0 在 5k 出现明显中期下降后恢复到 94.0%。这些结果仍是
单训练种子、固定 final cohort 下的探索性曲线，不构成多种子稳定性结论。

### 6.2 曲线图与原始数据

合并图：

![Round 3 Phase 2 remaining-task success curves](../../../../outputs/round3/phase2/round3_phase2_remaining_tasks_success_curves.png)

矢量版本：[SVG](../../../../outputs/round3/phase2/round3_phase2_remaining_tasks_success_curves.svg)、
[PDF](../../../../outputs/round3/phase2/round3_phase2_remaining_tasks_success_curves.pdf)。

独立任务图：[Reacher PNG](../../../../outputs/round3/phase2/reacher_curve_200ep_1k/round3_phase2_success_curves.png)、
[Push-T PNG](../../../../outputs/round3/phase2/pusht_curve_200ep_1k/round3_phase2_success_curves.png)、
[TwoRoom PNG](../../../../outputs/round3/phase2/tworoom_curve_200ep_1k/round3_phase2_success_curves.png)。

CSV 原始曲线：

- Reacher：[E0 offline](../../../../outputs/round3/phase2/reacher_curve_200ep_1k/e0/offline_continue/curve.csv)、[E0 online](../../../../outputs/round3/phase2/reacher_curve_200ep_1k/e0/online_adapt/curve.csv)、[E5 offline](../../../../outputs/round3/phase2/reacher_curve_200ep_1k/e5/offline_continue/curve.csv)、[E5 online](../../../../outputs/round3/phase2/reacher_curve_200ep_1k/e5/online_adapt/curve.csv)
- Push-T：[E0 offline](../../../../outputs/round3/phase2/pusht_curve_200ep_1k/e0/offline_continue/curve.csv)、[E0 online](../../../../outputs/round3/phase2/pusht_curve_200ep_1k/e0/online_adapt/curve.csv)、[E5 offline](../../../../outputs/round3/phase2/pusht_curve_200ep_1k/e5/offline_continue/curve.csv)、[E5 online](../../../../outputs/round3/phase2/pusht_curve_200ep_1k/e5/online_adapt/curve.csv)
- TwoRoom：[E0 offline](../../../../outputs/round3/phase2/tworoom_curve_200ep_1k/e0/offline_continue/curve.csv)、[E0 online](../../../../outputs/round3/phase2/tworoom_curve_200ep_1k/e0/online_adapt/curve.csv)、[E5 offline](../../../../outputs/round3/phase2/tworoom_curve_200ep_1k/e5/offline_continue/curve.csv)、[E5 online](../../../../outputs/round3/phase2/tworoom_curve_200ep_1k/e5/online_adapt/curve.csv)

完整性验收脚本为
[`scripts/validate_round3_phase2_all.py`](../../../../scripts/validate_round3_phase2_all.py)，
本次验证结果为 `training_runs=6 curves=12 aggregate_results=252`，每个 aggregate
结果均为 200 episodes、`status=ok`，且每个任务内部使用唯一 final-cohort hash。
