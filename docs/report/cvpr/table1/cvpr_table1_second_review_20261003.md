# Table 1 方案二次复查

日期：2026-10-03。对象：上一轮修改的 [主实验 plan](../../plan/cvpr_table1_plan.md) 与 [计时及范围说明](cvpr_table1_timing_and_scope_20261003.md)。本轮阅读实现、核对公开论文、读取已有结果；没有启动训练或评测，没有改动实验实现及主 plan。

## 判断

总体方向可保留：baseline 25/25、CoWM 主结果 none、裁剪作为测试、Reacher 增加一行且保留原 25/25 行、Table 2 对照不进入本轮。尚不应把全部计时细则视为可以直接执行的冻结协议。

主要修改点是 batch 计时的归属、Reacher 10/10 的选择依据，以及计时实现的统计开销。Sub-JEPA 的接入状态也已发生变化，需要同步计划。

## 1. 必须修正：batch=50 的耗时不能直接记作每 episode 独立规划时间

Plan 第 152 行要求从 batch=50 成功率评测得到每 episode 累计规划时间，但当前记录的事件耗时属于整个 active batch。一个批次可能包含多个 episode，且成功后 active batch 会缩小。

`source/common/round4_eval.py:230` 的 `replans=len(events)` 是整批规划调用数。`source/policy/round4.py:757` 的 `replan_indices` 才表示每次有哪些环境参与。原始 episode 记录中的 `planning` 还复制了整次运行摘要，不能把其中的 `replans` 当作该 episode 的调用数。

例如已有 Reacher 10/10 结果的 `replans=5`，并不表示每个 episode 都规划 5 次；由逐 episode 执行步数可推算平均为 2.24 次。

建议冻结三种不同字段：

- `batch_planning_wall_seconds`：整批各次完整规划调用耗时之和。
- `episode_replan_count`：按参与索引累计每个 episode 的实际调用次数。
- 若确需每 episode 成本分摊，显式定义 `sum(event_seconds / active_batch_size)`，命名为批处理摊销成本；它不能称为 batch=1 的独立规划延迟。

不能将整批时间完整加到每个参与者后，再求和声称总计算成本；这会重复计算。若报告每个 episode 参与事件的等待时长，应使用明确的等待时长名称。当前正式主指标仍可使用独立 batch=1 延迟，无需新增成功率矩阵。

## 2. Reacher 新行合理，但 10/10 的选择理由需要补全

核对 `config/round5/phase5_pre_report2.json` 与原始结果确认：已有 P3 短前缀使用同一 Round4 checkpoint、Euler=2、64 候选、`action_bound_mode=none`，与拟定正式主方法方向一致。B checkpoint 配置为 strict_causal，评分代码计算完整序列后读取指定前缀输出；没有发现把 score=10 误当作原生 10 步模型的问题。

两种短前缀在该探索 cohort 中都是 50/50，但原始轨迹还提供了上一轮未纳入选择理由的信息：

| Reacher P3 | 10/10 | 5/5 |
|---|---:|---:|
| 成功数 | 50/50 | 50/50 |
| 平均首次成功／执行步数 | 19.16 | 12.80 |
| 平均每 episode 规划次数（由执行步数推算） | 2.24 | 2.94 |
| 全部 50 个 episode 规划次数之和（推算） | 112 | 147 |
| 实际整批规划事件数 | 5 | 6 |
| 完整 50 步预算下的理论最大每 episode 调用数 | 5 | 10 |

来源目录：

- `outputs/round5/phase5_seed3072_legacy/conditions/exec_10/score_10/P3/not_applicable/none/step_2/euler/dev/`
- `outputs/round5/phase5_seed3072_legacy/conditions/exec_5/score_5/P3/not_applicable/none/step_2/euler/dev/`

成功率和首次成功步数取自 `result.json`；调用次数由 `trace/episodes.jsonl` 的 `steps_executed` 按 `ceil(steps_executed / execute_steps)` 推算，依据固定执行前缀、成功即停止的现有执行路径，不冒充新增事件实测。

10/10 减少规划调用，5/5 更早完成环境任务。因此“理论最大调用少一半”不能单独证明 10/10 的实际端到端时间更短。旧评测时间受共享设备负载及环境开销影响，也不能据其快慢选正式版本。

建议仍可预先选 10/10，但把理由写为“在探索成功数持平时优先降低规划调用数”，同时保留 5/5 更早到达目标的事实。不要写成 10/10 全面优于 5/5。该取舍不要求再增加一个正式主表版本。

新行仅改 Reacher，另外三任务 alias，保留原 Selection 行，统计上成立。必须在 caption 说明这是任务相关部署配置；它不能支持四任务统一时域设置或等重规划频率下的收益主张。复用数据不增加独立样本。

## 3. 正式计时入口还需消除不一致的统计开销

上一轮提出“关闭额外诊断”是正确方向，但当前代码关闭 diagnostic callback 仍会执行：

- `round4.py:635` 起的多次内部 CUDA 同步及分段计时；
- `round4.py:758,762` 的选择索引／候选 cost 转 CPU 和 Python list；
- `jepa.py:596` 为零梯度统计执行 `.item()`；
- `fast_lewam_eval.py:551` 的 CEM 计时打印（裁剪测试分支）。

其中部分会强制设备同步。若各方法携带不同程度的统计工作，外层完整 policy timer 会一并计入，尤其影响短延迟方法。

建议实现统一的计时执行模式：保留方法必需计算、动作输出、数值合法性检查和轻量调用计数；关闭或延后纯统计数据回传、分段 timer 同步和打印。不能盲目删除动作返回需要的同步或算法所需处理。数值检查若保留，应把它作为实际实现成本计入并在配置中固定。

计时与成功率运行需使用相同算法设置和精度开关，冻结 goal-cache 开关、history、CEM／候选分片和 proposal chunk。任何为去除统计而引入的分支，实施时应以固定输入／随机状态核对动作与候选选择一致，再产生正式延迟；不能将优化后的计时代码与另一套成功率算法拼接。

完整调用外层读取峰值显存、覆盖 refinement 并禁用内部重复峰值复位的建议正确。P3-PO-refine 的最终投影缺口也已由源码确认，修复仅作用于裁剪测试。

## 4. 计时定义合法，输入覆盖和论文表述可再改善

batch=1、CPU 原始输入到 CPU 动作、包含全部规划过程、前后 CUDA 同步、强制真实重规划、PO 保留动作梯度，均可作为正式主指标。

但“5 个状态各重复 50 次”只覆盖五个输入。LeWM 原始样本通常按数据行排序，取前五个不是额外随机抽样；重复测量主要降低系统计时噪声。它本身不构成错误，但不能把 250 次重复解释为 250 个独立状态，或把 P95 解释为整个部署状态分布的尾延迟。

建议在同样 250 次测量预算内，使用 seed=42 清单全部 50 个起点、每点重复 5 次；每任务／方法先按预设规则完成运行时预热。状态、噪声 seed 列表和方法执行顺序预先固定，减少输入选择和设备随时间变化的影响。这是本研究的协议选择，并非三篇论文规定。

初始决策延迟可以保留，但 caption 必须明确初始 history/padding 与无已有目标 cache。不能用它代表已有真实历史和缓存的所有 MPC 决策。此处不强制新增稳态计时实验；若论文需要端到端系统速度主张，可使用明确边界的批量评测耗时，并标注批量与归档开销。

## 5. 状态更新：Sub-JEPA 已接入

当前 [官方权重指南](../../guide/subjepa_official.md) 与 `outputs/subjepa_official_integration/integration_summary.json` 记录四任务官方权重转换、数值检查及 GPU 环境 smoke 已完成。本轮只核对已有记录和四个 `subjepa.pt` 文件存在，没有重跑这些检查。

因此 plan 第 10 行“用户提供 Sub-JEPA 实现后”已过时。可以保留分阶段调度，但第二阶段应改为“已有官方 Sub-JEPA 资产的正式归档与统一评测”，并继续使用 25/25。接入 smoke 不等于正式成功率结果。

924 单元／46,200 episodes 作为仅含 LeWM、LeFlow 的第一阶段计数仍正确；加入已经规划的 Sub-JEPA 六 seeds × 四任务后，完整总量为 **948 单元／47,400 episodes**。这是原有 baseline 的接入状态更新，不是建议扩大方法范围。

## 6. 文献与其它解释的核对结论

- [LeWM Figure 3](https://arxiv.org/html/2603.19312v1) 的 50 次 planning-time 平均与 Appendix D 的 L40S、300 samples、PushT 30／其它任务 10 iterations，上一轮引用准确。图的细粒度计时边界仍不能从论文充分确认。
- [LeFlow Table 2 与 §4.2](https://arxiv.org/html/2608.24855v1#S4.SS2) 的 50-episode 完整评测时间、五次运行平均、含 stepping/rendering，上一轮解释准确。
- [Sub-JEPA](https://arxiv.org/html/2605.09241v1) 没有给出可直接照搬的独立推理延迟协议，不能为其假定与 LeWM 图相同的口径。
- P0-PO2 的定义、flow 2 次与 PO 2 次分开计数、只优化动作不更新模型的说明正确。
- 保持 CoWM 主版本 none、只修复裁剪测试的最终投影、Table 2 对照与在线 rank_1.0 不进入本轮，均符合用户范围。

建议先落实第 1、3 项计时口径和实现要求，补全第 2 项取舍说明及第 5 项状态；第 4 项采样调整属于建议。上述修订不要求增加训练、评分器消融或新的时域方法行。
