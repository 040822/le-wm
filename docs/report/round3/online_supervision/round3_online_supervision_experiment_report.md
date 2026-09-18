# Round3 Online Supervision：本轮结果与下一轮决策

## 本轮范围与状态

本轮完成了两个严格限定任务 `reacher` 和 `pusht` 的前置 guidance 诊断：

- E1：每个任务 3 个 RMS（`0.01 / 0.03 / 0.10`），每档 64 个 paired
  groups，共 768 条候选记录；
- E2：每个任务使用同一份 50-episode `round3_revised` dev cohort，比较六种
  Stage-A/Stage-B 方法；
- 所有 E1/E2 collection report 均为 `ok`；
- E1 的 `Δphysical_cost` 定义为候选终点物理 cost 减去 actor anchor cost，负值
  表示物理 cost 下降；
- 阶段二 fixed replay 已启动但尚未完成；T0–T6 训练和 success-rate 曲线尚未启动。

GPU 最初在沙箱内不可见，改用沙箱外执行后确认节点有 8 张 RTX 4090。本轮实际
只使用物理 GPU0–3：E1 使用 GPU0/1，E2 使用 GPU2/3；GPU4–7 未使用。

## E1：局部梯度真实性

### Reacher

| 候选方向 | mean latent Δcost | mean physical Δcost | paired improvement rate |
|---|---:|---:|---:|
| actor anchor | 0.000000 | 0.000000 | 0.000000 |
| negative latent gradient | 0.089661 | 0.051350 | 0.468750 |
| positive latent gradient | 0.261661 | 0.114516 | 0.234375 |
| random direction | 0.007782 | 0.011182 | 0.416667 |

Reacher 的负 latent-cost gradient 没有带来平均物理 cost 下降；这说明局部 latent
方向与真实物理改善的一致性较弱。E1 本身不自动否决 guidance，最终结合 E2 和
用户判断。

| RMS | groups | environment steps | replay rows |
|---:|---:|---:|---:|
| 0.01 | 64 | 5,788 | 444 |
| 0.03 | 64 | 5,964 | 468 |
| 0.10 | 64 | 6,196 | 492 |

### Push-T

| 候选方向 | mean latent Δcost | mean physical Δcost | paired improvement rate |
|---|---:|---:|---:|
| actor anchor | 0.000000 | 0.000000 | 0.000000 |
| negative latent gradient | 0.020634 | -4.593762 | 0.526042 |
| positive latent gradient | 0.096851 | -3.453202 | 0.437500 |
| random direction | 0.002257 | 0.197666 | 0.390625 |

Push-T 的负 latent-cost gradient 与物理 cost 改善有较好的方向一致性，且优于正
梯度和随机方向的 paired improvement rate。

| RMS | groups | environment steps | replay rows |
|---:|---:|---:|---:|
| 0.01 | 64 | 5,648 | 452 |
| 0.03 | 64 | 5,944 | 472 |
| 0.10 | 64 | 6,352 | 504 |

## E2：生成方法对照

E2 每个数值均来自 50 个 dev episodes，不是用户要求的 final 200-episode
checkpoint success rate。

| method | Reacher | Push-T |
|---|---:|---:|
| A | 0.78 | 0.94 |
| A+B | 0.82 | 0.82 |
| Post-opt | **0.84** | 0.96 |
| Guided-flow | 0.82 | **0.98** |
| B-CEM | 0.78 | 0.94 |
| A-CEM | 0.70 | 0.94 |

E2 的完整报告：

- [Reacher E1](../../../../outputs/round3/online_supervision/reacher/e1/e1_report.json)
- [Reacher E2](../../../../outputs/round3/online_supervision/reacher/e2_v2/e2_report.json)
- [Push-T E1](../../../../outputs/round3/online_supervision/pusht/e1/e1_report.json)
- [Push-T E2](../../../../outputs/round3/online_supervision/pusht/e2_v2/e2_report.json)

本轮还修正了两个运行问题并保留在提交中：

- 可复用 collection helper 移到 `source/experiments/round3_online_collection.py`，
  source 层不再依赖 CLI script；
- E2 的 64-sample A+B solver budget 被作为显式诊断例外允许，其他 Round3
  协议校验仍保持严格。

相关提交为 `9e0cdd2` 和 `9a5131e`。

## guidance 决策记录（2026-09-16）

用户已确认按本报告的推荐方案进入下一轮：

- Reacher：`post_opt`；
- Push-T：`guided_flow`；
- T0–T6 全部保留；
- 首轮预算保持 200 次 optimizer update，是否延长由曲线结果决定。

E1 的原始 `.pt` replay 已在 report/manifest 完成核验后临时归档到
`/tmp/round3_e1_artifacts_20260916/`，原 `outputs` 路径保留 symlink；E1 report
和 group manifest 未移动。

## 阶段二执行记录（2026-09-16，进行中）

已按上述选择启动两个任务的 fixed replay 构造，并显式限制为允许的设备：
Reacher 使用物理 GPU0，Push-T 使用物理 GPU1。两项任务都复用已经生成并校验过的
offline replay；fixed replay 的临时根目录为
`/tmp/round3_online_supervision/{reacher,pusht}/fixed_replay_v1/`，workspace 下对应
路径保留 symlink，以避免再次占满 `/data`。

首次连续采集尝试在第一个 continuous action 前暴露了 Stage-B guidance 与安装版
CEM solver 的 `inference_mode` 冲突：CEM 的 actor warm-start 调用被包在
`inference_mode` 中，导致 `post_opt`/`guided_flow` 无法建立对 action 的 latent-cost
梯度。该次尝试没有产生 continuous 或 grounded 结果，但 offline replay 已完整保留。

修复为在 `ActorWarmStartModelView` 的 guidance actor 调用处局部退出
`inference_mode` 并显式开启 autograd；CEM 候选评分路径未改变。新增的 smoke test
已同时覆盖 `post_opt` 和 `guided_flow`，并验证两者均实际产生 backward。随后用
`--resume` 从既有 offline replay 重新启动 continuous collection；截至本记录时
Reacher 的 continuous 已完成精确 16,000 env steps，并已发布
`replay/continuous.pt`；Push-T 的 continuous 仍在运行。

Reacher 随后的 grounded 阶段在 `grounded-0001` 遇到一个过短 episode，recorder
没有形成完整 history window。group-0000 的 100 env steps 和 8 条 replay rows 已
保留；该情况本应被记录为 `ok_no_complete_transition_window` 并继续后续 groups，
但 fixed-replay orchestration 当时没有传入 collector 已支持的
`allow_empty_replay=True`，因此进程提前退出。已补上该参数；重新执行时会复用
continuous.pt 和 group-0000 shard，只从失败 group 继续。当前尚无完整
grounded/fixed-replay 汇总，因此当时尚未启动 T0–T6 训练，也没有 success-rate 曲线
或 final-200 结果可报告。

随后发现 v1 的 grounded shard 还存在独立的 shape contract 问题：旧 recorder 为了
在 5-block panel 内产出数据使用了 `history_size=action_horizon-2`，导致 grounded
行是 `observations=[4,...]`、`actions=[3,...]`，而 Fast-LeWAM 训练严格要求
`observations=[action_horizon+1,...]`、`actions=[action_horizon,...]`。v1 的
grounded/fixed 汇总因此被保留为诊断 artifact，不作为训练输入。

已修复 recorder：grounded 仍只执行 5 个 action blocks，在 rollout 结束后以
terminal observation 闭合最后一个完整 block，不调用环境、不增加 env step；完整
候选现在生成与 continuous 相同的 `6/5` 窗口，提前终止候选按 no-window 记录。CPU
回归测试已通过。正确 shape 的重建输出使用独立的
`/tmp/round3_online_supervision/{reacher,pusht}/fixed_replay_v2/` 根目录，复用 v1
的 offline/continuous artifact。v2 已完成并通过 fixed replay 拼接校验：

| task | guidance | continuous actual steps | grounded nominal/actual steps | fixed rows |
|---|---|---:|---:|---:|
| Reacher | `post_opt` | 16,000 | 4,000 / 3,856 | 1,832 |
| Push-T | `guided_flow` | 16,000 | 4,000 / 3,628 | 947 |

两份 v2 replay 的每行 shape 均为 `observations=[6,...]`、`actions=[5,...]`，并已
在 GPU0/GPU1 启动 T0–T6 fixed-replay 训练；训练完成前不填写非 T0 监督臂的
success-rate 结果。

截至当前已完成的 T0 baseline point（同一任务 final cohort 的 200 episodes，实际
均为 4×50）为：Reacher `0.875`，cohort hash 为
`f81e4830dd2bb80ad1a8109b67873496fa86651ccba63d49ab6d8b0c78b87490`；Push-T
`0.885`，cohort hash 为
`138501dd8c2d88990074da59039aa092d4269e841c7114fe68ae49fdd36cc47b`。这只是
update-000000 的冻结 baseline，不代表任何监督臂结果。

## 下一轮：固定 replay 与 200 optimizer updates

下一轮按已确认的 guidance，按任务分别执行：

1. 用选定的 collector 构造每任务一份不可变 fixed replay：
   `16,000 continuous + 4,000 grounded` 原始环境步；
2. T0–T6 全部保留，使用相同 initial E5 checkpoint、相同 replay、相同 batch
   manifest 和独立 optimizer；T0 保持冻结；
3. 每个训练臂执行首轮 200 次 optimizer update；这里的 step 严格指
   optimizer update，不是 env step；
4. 在 `0, 10, 20, ..., 200` 保存 checkpoint 并评测；
5. 每个点使用同一份 final 200-episode cohort，实际执行为 `4×50`，记录
   checkpoint 之后的 final success rate；评测不计入训练环境预算；
6. 训练结束后分别报告 Reacher 和 Push-T 的曲线、loss/gradient/source counts
   以及 final-200 success rate，不跨任务平均。

200 update 仍是首轮筛选预算，不代表收敛。曲线在 200 update 仍明显上升时，是否
延长到 500/1000 update，由用户根据结果决定。

## 阶段二中间验收快照（2026-09-16 21:19）

### 当前执行状态

截至本快照，Push-T 的 T0--T6 已全部完成 200 个 optimizer updates；Reacher 的
T0--T5 已完成，T6 `online_distill_a` 正在运行到 update 40/200。T6 的 update-40
checkpoint 已写入，final-200 评测正在进行，因此该点暂记为
`evaluation_pending`；Reacher 的 T6 完成后才算本阶段训练结束。当前 GPU 进程已用
提升权限启动，并通过 `CUDA_VISIBLE_DEVICES=0` 限制在物理 GPU0；Push-T 使用的
GPU1 任务已结束。

本次中断恢复过程也已核验：第一次 Reacher T5 运行在 update-50 评测期间被中断，
造成 `run_state` 停在 40、adapter state 到 50 的不一致。随后用同一 checkpoint、
replay、seed 重建了 T5 的 update-40 adapter state，并逐 tensor 验证其 model 参数
与原 update-40 checkpoint 完全一致，再从 update 40 继续。因此当前 T5 的 1--200
update history 是连续的。runner 同时增加了评测前先落盘 pending 状态，以及将
GPU 映射后的 RNG state 转回 CPU ByteTensor 的恢复保护；针对性测试为 14 个通过。

### 已完成曲线的 final-200 success rate（百分比）

下表列出 0/50/100/150/200 optimizer updates 五个检查点，完整 0, 10, ..., 200
曲线保存在各 arm 的 `curve.csv`/`curve.json`。每个数均为同一任务固定 final cohort
的 200 episodes（4×50），不是 env step；两任务不混合比较。

| task | arm | 0 | 50 | 100 | 150 | 200 |
|---|---|---:|---:|---:|---:|---:|
| Reacher | T0 frozen | 87.5 | 87.5 | 87.5 | 87.5 | 87.5 |
| Reacher | T1 offline-B MSE | 87.5 | 87.0 | 79.0 | 83.5 | 83.0 |
| Reacher | T2 online-B MSE | 87.5 | 79.5 | 81.0 | 78.0 | 77.5 |
| Reacher | T3 online ranking | 87.5 | 79.5 | 81.0 | 78.0 | 77.5 |
| Reacher | T4 online/offline-A | 87.5 | 79.5 | 78.0 | 80.0 | 70.5 |
| Reacher | T5 online hindsight-A | 87.5 | 78.5 | 82.5 | 79.5 | 76.0 |
| Reacher | T6 online distill-A | 87.5 | 79.0 | 76.5 | 78.5 | 79.5 |
| Push-T | T0 frozen | 88.5 | 88.5 | 89.0 | 88.5 | 89.0 |
| Push-T | T1 offline-B MSE | 88.5 | 88.5 | 85.5 | 83.5 | 85.0 |
| Push-T | T2 online-B MSE | 88.5 | 84.5 | 81.5 | 83.0 | 81.5 |
| Push-T | T3 online ranking | 89.0 | 84.5 | 82.5 | 83.0 | 81.5 |
| Push-T | T4 online/offline-A | 88.5 | 85.5 | 83.0 | 83.0 | 81.5 |
| Push-T | T5 online hindsight-A | 88.5 | 81.5 | 86.5 | 79.5 | 80.5 |
| Push-T | T6 online distill-A | 88.5 | 83.5 | 85.5 | 80.5 | 80.5 |

Reacher T6 当前已完成的曲线点为：update 0=`87.5%`、10=`82.5%`、20=`84.0%`、
30=`80.0%`；update 40 仍在评测。其余已完成曲线的原始结果：

- [Reacher fixed-replay curves](../../../../outputs/round3/online_supervision/reacher/fixed_train_v1/)
- [Push-T fixed-replay curves](../../../../outputs/round3/online_supervision/pusht/fixed_train_v1/)

### supervision 信号的中间诊断

- T3 的 ranking calibration 在 Reacher 和 Push-T 都报告
  `invalid_terms=[rank]`、auxiliary gradient norm=`0`、rank weight=`0`。所以当前
  T3 实际退化为 T2 的 B-MSE-only 更新；Reacher 两者曲线完全相同，Push-T 的训练
  更新和曲线也相同（update-0 的评测随机性造成的 0.5 个百分点差异不改变这一判断）。
- Reacher 的非零校准权重为 T4/T5/T6 的 offline-A=`0.0425672`、T5 hindsight
  A=`0.0261718`、T6 distill-A=`0.0232694`；Push-T 对应 offline-A=`0.00379493`、
  hindsight-A=`0.00877454`、distill-A=`0.00384556`。
- 已完成 T5 的 update-200 total loss / B-MSE 为 Reacher
  `0.17984 / 0.06981`、Push-T `0.03645 / 0.02069`；T5 的 hindsight 项在两任务
  均实际有 32 个样本。Push-T T6 的 update-200 total loss / B-MSE 为
  `0.03023 / 0.02064`，distill 项实际有 32 个样本。T6 Reacher 的 update-40
  loss snapshot 已记录在 run state，待完成后再纳入最终汇总。

### 请用户进行的中间验收

这版结果只供你判断，不设置自动门槛，也不会自动启动下一阶段。当前最值得确认
的是：

1. 是否接受 T3 ranking signal 在本实现/当前 replay 上校准为零，暂按 T2 的重复
   对照保留，还是在继续前要求单独修正 ranking loss/replay contract；
2. 在 Reacher T6 完成后，哪些 T3/T5/T6 arm 值得进入 closed-loop；
3. 是否先按 200-update 曲线做阶段三的 20k-env-step 复验，或延长固定 replay 到
   500/1000 update。这里不预设通过分数，最终由你决定。

## 阶段二最终结果（2026-09-16 22:25）

### 完成性与协议核验

Reacher 和 Push-T 的 fixed-replay 根结果均为 `status=ok`，T0--T6 共 14 个 arm
全部完成 200 个 optimizer updates。每个 arm 均有完整的
`0, 10, 20, ..., 200` 曲线、21 个 checkpoint 和 200 条 update history；曲线的
step unit 明确为 `optimizer_update_step`，不是 env step。

final evaluation 独立核验了 294 个 aggregate 结果（14 arms × 21 points）：每个
结果均为 200 episodes，由 4 个各 50 episodes 的 batch 组成，状态为 `ok`，并且
每个任务内部使用同一个 fixed final cohort。Reacher cohort hash 为
`f81e4830dd2bb80ad1a8109b67873496fa86651ccba63d49ab6d8b0c78b87490`，Push-T
cohort hash 为 `138501dd8c2d88990074da59039aa092d4269e841c7114fe68ae49fdd36cc47b`。
所有 final checkpoint hash、curve coverage、loss finite性和 source counts 均已通过
全量校验。

工程侧验证也已完成：Round3 定向测试 `14/14` 通过，3 个本轮修改文件通过
`py_compile`。全仓库共运行 343 个测试，其中 1 个 skip；仅有 2 个既有的
`inspect_h5_dataset` 输出格式断言失败，均不涉及本轮 Round3 文件或实验 artifacts，
因此没有将其混入本轮修复。

### update-200 final success rate

| task | T0 frozen | T1 offline-B | T2 online-B | T3 ranking | T4 offline-A | T5 hindsight-A | T6 distill-A |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reacher | 87.5% | 83.0% | 77.5% | 77.5% | 70.5% | 76.0% | 79.5% |
| Push-T | 89.0% | 85.0% | 81.5% | 81.5% | 81.5% | 80.5% | 80.5% |

按 update-200 的 final-200 endpoint 看，当前新增监督臂没有超过各任务的 T0
冻结 baseline；Reacher 的最高训练臂为 T1 的 83.0%，Push-T 的最高训练臂为 T1
的 85.0%。完整 21-point 曲线仍以以下 artifacts 为准：

- [Reacher fixed-train result](../../../../outputs/round3/online_supervision/reacher/fixed_train_v1/result.json)
- [Reacher T5 curve](../../../../outputs/round3/online_supervision/reacher/fixed_train_v1/t5_online_hindsight_a/curve/curve.csv)
- [Reacher T6 curve](../../../../outputs/round3/online_supervision/reacher/fixed_train_v1/t6_online_distill_a/curve/curve.csv)
- [Push-T fixed-train result](../../../../outputs/round3/online_supervision/pusht/fixed_train_v1/result.json)
- [Push-T T6 curve](../../../../outputs/round3/online_supervision/pusht/fixed_train_v1/t6_online_distill_a/curve/curve.csv)

### loss 与监督信号核验

以下为 200 次 update 的平均 total loss，以及 update-200 的 total loss / B-MSE；
带辅助项的 total loss 不应直接与 B-MSE-only 臂横向解释。

| task | arm | mean total loss | final total / B-MSE |
|---|---|---:|---:|
| Reacher | T1 | 0.05220 | 0.04460 / 0.04460 |
| Reacher | T2 | 0.07185 | 0.07202 / 0.07202 |
| Reacher | T3 | 0.07185 | 0.07202 / 0.07202 |
| Reacher | T4 | 0.14039 | 0.11447 / 0.05062 |
| Reacher | T5 | 0.18795 | 0.17984 / 0.06981 |
| Reacher | T6 | 0.18449 | 0.17682 / 0.06980 |
| Push-T | T1 | 0.01152 | 0.00905 / 0.00905 |
| Push-T | T2 | 0.02814 | 0.01799 / 0.01799 |
| Push-T | T3 | 0.02814 | 0.01799 / 0.01799 |
| Push-T | T4 | 0.03220 | 0.02742 / 0.02394 |
| Push-T | T5 | 0.04514 | 0.03645 / 0.02069 |
| Push-T | T6 | 0.03928 | 0.03023 / 0.02064 |

source/batch manifest 也完成了预期覆盖：每个非 T0 update 的 B-MSE batch 为
64 条（32 offline + 24 continuous + 8 grounded）；T1 的 offline replacement
为 32 条；T4/T5/T6 的 offline-A 各为 32 条；T5 hindsight-A 和 T6 distill-A
各为 32 条。200 updates 的累计辅助样本数为 T5 hindsight-A=6,400、T6
distill-A=6,400，均为实际参与训练的样本。

需要单独保留的结论是：两任务 T3 的 batch manifest 中 200 updates 的
`ranking_pairs` 累计均为 0，calibration 因而报告 auxiliary gradient norm=`0`
并将 rank weight 设为 `0`。这解释了 T3 与 T2 的训练 loss/曲线相同；它是当前
ranking replay/contract 未提供有效 pair 的诊断结果，不应被解读为 ranking
监督已经验证有效。

### 本轮实验结论与下一步决策

本轮 fixed-replay 筛选已完成，但结果不支持自动推进任何新增监督臂。下一步仍由
用户决定：

1. 是否把 T3 ranking 的 `ranking_pairs=0` 作为需要先修复的协议问题；
2. 若暂不修复，是否选择 Reacher/Push-T 各自的 T3/T5/T6 中任意 arm 进入
   20k-env-step closed-loop 复验；
3. 是否对某些 arm 延长固定 replay 到 500/1000 optimizer updates。

除非用户明确选择，否则不会自动启动 closed-loop、延长训练或 R4-AB。

## 后续仍需用户决定的内容

本轮 guidance gate 已完成，不再等待 collector 选择。下一轮 fixed-replay 训练
完成后，再由用户决定：

- 哪些 T3/T5/T6 结果进入阶段三 closed-loop；
- 是否启动阶段三的 20k-env-step closed-loop 复验；
- 是否延长固定 replay 或 closed-loop 到 500/1000 update；
- 是否另行启动 R4-AB。R4-AB 不会自动启动。

阶段二训练已按上述选择启动后，曲线结果仍只作为参考；不会自动根据某个分数
推进阶段三、延长训练或启动 R4-AB。
