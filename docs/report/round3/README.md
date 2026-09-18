# Round 3 报告归档

本目录汇总 Round 3 的实验报告（仅 Markdown），记录截至 2026-09-16 22:31 的结果。协议为 `round3_revised`，主指标为 Stage-B success rate。

## 导航

- [Round 3 实验总报告](round3_experiment_report.md) — 覆盖全部阶段的汇总入口
- Phase 1（新协议复评）
  - [新协议复核总报告](phase1/round3_new_protocol_experiment_report.md)
  - [Canonical Phase 1 报告](phase1/phase1_report.md)
  - [扩展评测报告](phase1/extended_eval_report.md)
  - [失败案例报告](phase1/failure_cases_report.md)
- Phase 1.5（Fast 起点选择）
  - [决策与执行登记](phase1.5/round3_phase1.5_decision.md)
- Phase 2（在线后训练）
  - [在线后训练报告](phase2/round3_phase2_experiment_report.md)
- Phase 3（physical_time_type 探索）
  - [physical-time/type 报告](phase3/phase3_e5_physical_time_type_report.md)
- Online Supervision
  - [E1/E2 诊断与 T0–T6 fixed-replay 总报告](online_supervision/round3_online_supervision_experiment_report.md)

## 阶段结论速览

| 阶段 | 结论 |
|---|---|
| Phase 1 | 高层结论保留，效应减弱；Reacher Fast 优势与 E3-384 超过 E0 被推翻 |
| Phase 1.5 | Fast 全局起点冻结为 E5 legacy checkpoint |
| Phase 2 | 当前配置下无稳定跨任务 online adaptation 收益 |
| Phase 3 | `physical_time_type` 无统一收益，封存为 `completed_exploratory_not_mainline` |
| Online Supervision | T0–T6 无监督臂超过 T0 frozen；T3 因 `ranking_pairs=0` 退化为 T2；尚未进入 closed-loop 20k 复验 |

## 来源说明

- `phase1/round3_new_protocol_experiment_report.md`、`phase1.5/round3_phase1.5_decision.md`、`phase2/round3_phase2_experiment_report.md`、`online_supervision/round3_online_supervision_experiment_report.md` 由 `docs/plan/` 移入归档。
- `phase1/phase1_report.md`、`phase1/extended_eval_report.md`、`phase1/failure_cases_report.md`、`phase3/phase3_e5_physical_time_type_report.md` 为 `outputs/round3/` 下报告的副本；原始产物仍保留在 `outputs/round3/`。
- Online Supervision 的 fixed replay / checkpoint 实际位于 `/tmp/round3_online_supervision/`，workspace 下通过 symlink 引用。
