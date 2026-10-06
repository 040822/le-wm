# Reacher：LEWM / LEFlow 预算 50→125 步扩展

## 协议

在原有 50 个 Reacher legacy 起点、推理 seed=42、目标偏移25步、25/25执行与评分条件下，将单回合环境步数上限从50提高到125。固定各方法原 checkpoint 和规划设置，不训练、不微调；两预算按相同 episode ID 与起始步配对。125步结果位于 `outputs/round5/phase5_baseline_horizons/budget_125/`。

LEWM 使用本地训练 checkpoint 与原生 CEM；LEFlow 使用发布 planner 及其冻结的 LEWM 参考权重。两方法的横向差异不作单变量因果解释。

## 配对结果

| 方法 | 50步成功 | 125步成功 | 配对差 pp [bootstrap 95% CI] | 改善/退化/不变 | McNemar p | Holm p | 规划时间 50→125 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| lewm | 36/50 (72%) | 48/50 (96%) | +24.0 [+12.0, +36.0] | 12/0/38 | 0.0004883 | 0.0009766 | 67.31→65.52 |
| leflow | 43/50 (86%) | 50/50 (100%) | +14.0 [+6.0, +24.0] | 7/0/43 | 0.01562 | 0.01562 | 4.40→3.56 |

成功率区间为逐预算 Wilson 95% 区间；配对差的区间通过按 episode 重采样 bootstrap 10,000 次计算。检验为双侧 exact McNemar，Holm 校正覆盖两项方法内比较。单 checkpoint、单 seed、50 个起点仍属探索性结果。

预算50结果沿用 `outputs/round5/phase5_baseline_horizons/` 中已完成的同条件基线；预算125结果保留逐 episode 成功记录、规划事件、checkpoint/cohort SHA、评分审计和 GPU 信息。两预算 runner 的源文件 hash 不同：本次 runner 改动增加预算参数、预算身份记录和对应输出路径，planner 与模型代码未改。

逐项身份及成本数据：`outputs/round5/phase5_baseline_horizons/budget_125/analysis/paired_budget_comparison.json`。
