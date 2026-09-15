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

- [Reacher E1](../../outputs/round3/online_supervision/reacher/e1/e1_report.json)
- [Reacher E2](../../outputs/round3/online_supervision/reacher/e2_v2/e2_report.json)
- [Push-T E1](../../outputs/round3/online_supervision/pusht/e1/e1_report.json)
- [Push-T E2](../../outputs/round3/online_supervision/pusht/e2_v2/e2_report.json)

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
grounded/fixed-replay 汇总，因此尚未启动 T0–T6 训练，也没有 success-rate 曲线或
final-200 结果可报告。

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

## 后续仍需用户决定的内容

本轮 guidance gate 已完成，不再等待 collector 选择。下一轮 fixed-replay 训练
完成后，再由用户决定：

- 哪些 T3/T5/T6 结果进入阶段三 closed-loop；
- 是否启动阶段三的 20k-env-step closed-loop 复验；
- 是否延长固定 replay 或 closed-loop 到 500/1000 update；
- 是否另行启动 R4-AB。R4-AB 不会自动启动。

阶段二训练已按上述选择启动后，曲线结果仍只作为参考；不会自动根据某个分数
推进阶段三、延长训练或启动 R4-AB。
