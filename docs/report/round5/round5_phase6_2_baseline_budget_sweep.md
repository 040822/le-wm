# Reacher：CoWM、LeWM、LeFlow 的执行预算探索

## 协议

三种方法均固定 25/25 执行与评分，每个预算评测相同的 50 个 Reacher legacy 起点（cohort seed=42），目标偏移25步。LEWM/LEFlow 使用 evaluation/policy seed=42；CoWM 沿用 Table 1 的环境 seed=10042、policy seed=20042。模型和规划设置保持不变，只改变每回合环境步数上限：60、75、90、110。非25整数倍预算只执行剩余步数，不向上取整。

CoWM 使用 Table 1 的标准 CoWM-Selection（P3，64候选，Euler action-flow step=2）；LeWM 使用原生 CEM；LeFlow 使用原生 latent-flow sampler。三种方法分别与同 checkpoint/cohort 的预算50基线比较。各模型横向差异不作单变量因果解释。

## 成功率与预算50配对差

| 方法 | 上限 | 成功数/50 | 成功率 Wilson 95% CI | 相对50步差 pp [配对 bootstrap 95% CI] | 改善/退化/不变 | McNemar p | Holm p（12项） | 规划事件数 | 规划秒 | 评测秒 | GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CoWM-Selection | 60 | 44/50 | 88.0% [76.2%, 94.4%] | +0.0 [+0.0, +0.0] | 0/0/50 | 1 | 1 | 3 | 0.87 | 15.61 | 1 |
| CoWM-Selection | 75 | 46/50 | 92.0% [81.2%, 96.8%] | +4.0 [+0.0, +10.0] | 2/0/48 | 0.5 | 1 | 3 | 0.57 | 15.25 | 1 |
| CoWM-Selection | 90 | 47/50 | 94.0% [83.8%, 97.9%] | +6.0 [+0.0, +14.0] | 3/0/47 | 0.25 | 1 | 4 | 0.67 | 17.34 | 1 |
| CoWM-Selection | 110 | 50/50 | 100.0% [92.9%, 100.0%] | +12.0 [+4.0, +22.0] | 6/0/44 | 0.03125 | 0.2188 | 5 | 0.60 | 16.24 | 1 |
| LeWM | 60 | 42/50 | 84.0% [71.5%, 91.7%] | +12.0 [+4.0, +22.0] | 6/0/44 | 0.03125 | 0.2188 | 3 | 59.27 | 78.34 | 3 |
| LeWM | 75 | 47/50 | 94.0% [83.8%, 97.9%] | +22.0 [+10.0, +34.0] | 11/0/39 | 0.0009766 | 0.009766 | 3 | 59.08 | 79.82 | 3 |
| LeWM | 90 | 48/50 | 96.0% [86.5%, 98.9%] | +24.0 [+12.0, +36.0] | 12/0/38 | 0.0004883 | 0.005859 | 4 | 59.67 | 79.37 | 3 |
| LeWM | 110 | 48/50 | 96.0% [86.5%, 98.9%] | +24.0 [+12.0, +36.0] | 12/0/38 | 0.0004883 | 0.005859 | 5 | 59.94 | 81.00 | 3 |
| LeFlow | 60 | 46/50 | 92.0% [81.2%, 96.8%] | +6.0 [+0.0, +14.0] | 3/0/47 | 0.25 | 1 | 3 | 3.58 | 22.14 | 0 |
| LeFlow | 75 | 49/50 | 98.0% [89.5%, 99.6%] | +12.0 [+4.0, +22.0] | 6/0/44 | 0.03125 | 0.2188 | 3 | 3.63 | 22.52 | 0 |
| LeFlow | 90 | 50/50 | 100.0% [92.9%, 100.0%] | +14.0 [+6.0, +24.0] | 7/0/43 | 0.01562 | 0.1406 | 4 | 3.64 | 22.04 | 0 |
| LeFlow | 110 | 50/50 | 100.0% [92.9%, 100.0%] | +14.0 [+6.0, +24.0] | 7/0/43 | 0.01562 | 0.1406 | 4 | 3.52 | 22.02 | 0 |

差值定义为该预算成功指示减去预算50基线成功指示；区间按 episode 配对 bootstrap 10,000 次计算。McNemar 为双侧精确检验；Holm 校正覆盖3个方法×4个预算的12项探索性比较。相同起点用于配对，但本扫参未审计不同预算下逐步动作/状态前缀是否完全一致。

基线来源：CoWM 为 Table 1 Reacher seed=42、P3 main 的既有预算50结果；LeWM、LeFlow 为 Round 5 baseline horizons 的 25/25、预算50结果。三份基线和本次扩展均使用 cohort SHA256 `ff4f26ad3fd809e59e1b748c7c93c33a8e51b48e13d2505e89e5909d6121bf45`。

规划事件数是 evaluator 记录的批次级事件数，不等于跨 episode 求和的每 episode 规划调用总数。两轮初始 CoWM 结果因 seed 未正确传入 evaluator 而失效，分别归档于 `outputs/round5/phase6_2_baseline_budgets/invalidated/cowm_environment42_policy42/` 和 `outputs/round5/phase6_2_baseline_budgets/invalidated/cowm_environment10042_policy10042/`；本报告只纳入 evaluator 参数确认使用 environment/policy seed=10042/20042 的正式重跑。

冻结配置：`config/round5/phase6_2_baseline_budget_sweep.json`。JSON/CSV 明细：`outputs/round5/phase6_2_baseline_budgets/analysis/budget_curve.json`、`outputs/round5/phase6_2_baseline_budgets/analysis/budget_curve.csv`。
