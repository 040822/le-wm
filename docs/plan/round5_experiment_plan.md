# Round 5 实验计划（剩余阶段）

本轮目标：整合 round3 与 round4 的实验结果，确定实验最终方案。编排为三个阶段。

## 阶段状态总览

| 阶段 | 内容 | 状态 | 说明 / 链接 |
|---|---|---|---|
| Phase 1 | R4-AB 最终推理方案探索（flow step、P0~P3、guided-flow / post-opt） | **已执行并冻结** | 方案存档 `docs/plan/round5_phase1_plan_frozen.md`；报告 `docs/report/round5/round5_phase1_report.md`；决策 `docs/report/round5/round5_final_scheme_decision.md` |
| Phase 2 | online train 探索（检查是否提升 P3） | **冻结（暂不执行）** | 见下文「Phase 2（冻结）」，原因：四任务成功率基本饱和 |
| Phase 3 | 任务探索实验，增加新实验任务 | 占位 | 见下文「Phase 3 占位」 |

Phase 1 已按独立冻结文档拆分并归档，本文件不再重复其矩阵与实现细节，只保留阶段状态与后续规划。

## Phase 1（已执行 · 冻结）

- 结论：主推理方案定为 **P3（64 候选 + B 选优），flow step 1/2**；round3 的
  guided-flow / post-opt **可与 P3 结合**，收益任务相关，仅 Reacher 稳定为正；
  guidance 作为按任务可选增益，默认关闭。评测沿用 `legacy` 旧协议，训练用 R4-AB。
- 244 / 244 条件完成，全部 `status=ok`，每条件 50 episodes；产物见
  `outputs/round5/phase1_seed3072_legacy/`。
- 完整方案见冻结文档：`docs/plan/round5_phase1_plan_frozen.md`。

## Phase 2（冻结）

**状态：冻结，本轮不执行，待满足解冻条件后再评估。**

### 冻结原因

- 当前四任务成功率已基本饱和：cube / tworoom 接近 100%，pusht 约 96–100%，
  仅 reacher 未饱和（无 guidance 时 72–90%，加 guidance 后 92–94%）。
- 在饱和的成功率下，online train 缺乏可提升空间：主指标接近上限，无法区分
  在线续训带来的增益，投入产出比低。
- 因此不将 online train 纳入主方案；待任务扩展（Phase 3）或评测协议变化使
  成功率脱离饱和后，再重新评估。

### 解冻条件（满足其一即可重新评估）

1. Phase 3 引入足够多、且当前方法**未饱和**的新任务；
2. 评测协议 / episode 数变化（如 50→200）使现有四任务成功率明显下降；
3. 出现明确的在线适应需求（例如新任务的分布偏移或非平稳环境）。

### 原方案大纲（存档，解冻后按此细化）

- 起点 R4-AB seed3072 epoch10；评测策略 **P3（P0~P2 辅助）**。
- 方案：round3 T-arm 固定 replay（T0 frozen、T1 offline-B、T2 online-B、
  T3 ranking、T4 offline-A、T5 hindsight-A、T6 distill-A），四任务，
  200 optimizer updates，曲线 0,10,…,200。
- 待 Phase 1 冻结后确定评测协议与 guidance 采集器；需为 cube / tworoom 新建
  offline replay。
- 风险：round3 T3 ranking `ranking_pairs=0` 未生效，需先修协议或由用户决定是否
  保留该 arm。

## Phase 3 占位

任务扩展（AAAI DeWM 任务 + 若干扩展任务）、LeWM 上过饱和检查、成功判定。本轮不细化。

## 风险与边界

- 单 checkpoint、单训练 seed、legacy cohort 非严格 held-out，结论保持探索性表述。
- Phase 2 冻结基于当前四任务饱和这一观察；若主指标或任务集变化，需重新审视。
- guidance 在 step 1/2（全程引导）与 round3（10/5/5）是不同 regime，不得直接等同。
