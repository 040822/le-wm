# Round 3 Phase 1.5 决策与执行登记

## 决策

Phase 1 已完成。根据用户确认，Phase 1.5 冻结为：

1. Phase 1.5 的全局 Fast 基线使用 E5 的现有 legacy checkpoint。
2. `physical_time_type` 不在 Phase 1.5 提前训练；当时登记为 Phase 3 的候选表示，
   `stage_b_timestep_mode=legacy` 保持不变。
3. 补充训练的 TwoRoom E3 纳入分析，但不回写 Phase 1 canonical matrix。
4. Round 4 当前运行继续完成；Round 3 与 Round 4 并行，但 GPU 训练和 EGL
   评测不与 Round 4 抢占 GPU0–3。

## E3/E5 选择证据

Phase 1 canonical dev、Stage-B、`round3_revised`：

| Task | E3 | E5 | E5 − E3 |
| --- | ---: | ---: | ---: |
| Cube | 42% | 48% | +6 pp |
| Push-T | 86% | 94% | +8 pp |
| Reacher | 78% | 78% | 0 pp |
| 共同任务平均 | — | — | +4.67 pp |

TwoRoom 的 canonical E3 权重当时缺失。补充 E3 checkpoint 的
`round3_revised` final Stage-B 为 90.5%，但当前没有对应 canonical dev 结果，
所以它作为 supplemental evidence 记录，不能填入上面的 dev 平均，也不能
覆盖 Phase 1 的 `missing_weight` 行。

## Phase 3 后续决策更新

Phase 3 的 `physical_time_type` 四任务结果已经完成并纳入对比。相对历史 legacy E5
的 Stage-B 差值为 Cube `+2.0 pp`、Push-T `−3.5 pp`、Reacher `−1.5 pp`、TwoRoom
`+0.5 pp`，未显示跨任务统一收益。

因此后续 Round 3 Fast 主线继续采用历史 E5 legacy checkpoint；`physical_time_type`
保留为已完成的探索性证据，不再追加主线训练或评测。详细对照见
`outputs/round3/phase3/phase3_e5_physical_time_type_report.md`。

## Phase 1.5 权重边界

当前已有的 E5 legacy checkpoint 就是 Phase 1.5 的起点。现有
`physical_time_type` checkpoint 只有 E3 的 Push-T 和 Reacher；它们不参与
Phase 1.5 起点替换，也不能冒充 Phase 3 的 E5 主线。

## Phase 2 前置条件

Phase 2 按 E0 规则先做 Cube。开始 GPU 运行前必须完成：

- Cube Phase 2 三组（冻结、离线续训、在线适应）的 CPU/dry-run；
- 真实动作、后续观测、episode 边界、采集模型版本和 replay 哈希的记录；
- 0、5k、10k、20k 环境步评估输出目录与 checkpoint 命名冻结。

上述 CPU 前置条件、mock 主循环和 exact-target collection smoke 已完成。报告为
`outputs/round3/phase2/cpu_dry_run/cube/phase2_cpu_dry_run_report.json`：三臂均
完成至少一次预期更新，offline/online 混合臂实际采到 4:4 的 source-balanced
batch，encoder/projector 的前后哈希一致。当前尚未宣称 GPU 评测完成；本 shell
没有可用 CUDA runtime，且 Round 4 仍在占用既有资源窗口。

下一步是等待一个不打断 Round 4 的 GPU0–3 资源窗口，先跑单个 100 环境步
Cube chunk，确认完整 CEM（300 samples、30 iterations、top-k 30）和在线更新，
再按 0/5k/10k/20k 节点推进。采集入口使用
`collect-online --target-environment-steps 100 --pool-offset 0`，不以 episode
数近似环境步数；若单 chunk 运行正常，才扩展到完整 20k 单种子。

Round 4 继续运行期间，Round 3 先完成上述非 GPU 工作；Round 3 GPU 任务必须
显式设置 `CUDA_VISIBLE_DEVICES` 为 GPU0–3 中的设备，并单独记录资源窗口。

## Phase 3 后续执行边界

原 Phase 3 `physical_time_type` 训练、注册和评测已完成，注册表
`config/round3/phase3_e5_physical_time_type.json` 与 verifier
`scripts/verify_round3_phase3_e5.py` 作为历史审计材料保留。该分支已标记为
`completed_exploratory_not_mainline`，不再启动新的训练或评测。

后续 Fast 任务直接复用 `config/round3/phase1_5_selection.json` 中登记的四个历史
E5 legacy checkpoint；任何 GPU0–3 运行仍须等待不打断 Round 4 的资源窗口。

## Round 4 解释边界

当前 Round 4 运行不重启，按用户决定继续完成。由于其现有训练配置可能接触
Round 3 cohort，它的结果在最终报告中标记为探索性结果，不作为无数据暴露的
严格公平对照。
