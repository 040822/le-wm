# Reacher：LeWM / LeFlow 执行与评分前缀对照

固定已有 checkpoint；50 个 legacy 起点，与 Phase5 pre_report2 的 Reacher cohort SHA 一致；seed=42、目标偏移25步、评测预算50步。

所有条件仍生成5个5-step动作块（25步）；仅改变评分所用的前缀和执行前缀。LeWM 保留原生 CEM（300 samples、30 iterations、top-30，沿用默认 warm-start）；LeFlow 保留64条候选、16步flow及原生解码。未新增 candidate clipping，不能与 FastLeWAM 的 cem-clip 解释为仅模型不同的严格消融。

LeFlow 的5/5不是生成1-block潜在路径：其现有采样器要求至少2个block，这里始终生成5个block后按首个block的rollout结果评分。LeWM 使用历史本地训练checkpoint；LeFlow使用发布planner及其引用的冻结LeWM，不能将两者差异单独归因于规划算法。

| 方法 | execute/score | 成功数 | 规划调用数 | 总规划时间(s) | 总评测时间(s) |
|---|---|---:|---:|---:|---:|
| lewm | 25/25 | 36/50 | 2 | 67.31 | 86.23 |
| lewm | 10/10 | 44/50 | 5 | 50.89 | 68.08 |
| lewm | 5/5 | 45/50 | 10 | 53.10 | 68.62 |
| leflow | 25/25 | 43/50 | 2 | 4.40 | 22.29 |
| leflow | 10/10 | 49/50 | 5 | 4.95 | 21.71 |
| leflow | 5/5 | 47/50 | 10 | 7.96 | 24.24 |

差值为短前缀减同方法25/25；按起点配对bootstrap 10,000次。四个McNemar检验统一Holm校正。

| 方法 | execute/score | 差值 pp [95% CI] | 改善/退化起点 | 精确p | Holm p |
|---|---|---|---:|---:|---:|
| lewm | 10/10 | +16.0 [+4.0, +30.0] | 10/2 | 0.03857 | 0.1157 |
| lewm | 5/5 | +18.0 [+6.0, +32.0] | 11/2 | 0.02246 | 0.08984 |
| leflow | 10/10 | +12.0 [+2.0, +24.0] | 7/1 | 0.07031 | 0.1406 |
| leflow | 5/5 | +8.0 [-2.0, +20.0] | 6/2 | 0.2891 | 0.2891 |

本轮六项均完成，运行时确认所有条件生成5个block，并按配置评分5/2/1个block。LeWM和LeFlow在两个短前缀条件下的成功率点估计均高于自身25/25基线，但四项统一Holm校正后均未达到0.05。bootstrap区间与精确检验可能给出不同的零边界判断，显著性按精确检验及其校正解释。

这提示Reacher上的短前缀收益可能跨规划方法出现，并非FastLeWAM独有的现象；当前样本不足以确认稳定提升。相对25/25同时改变了执行和评分长度，不能分别归因于重规划频率或评分窗口。进一步拆分需要10/25、5/25等控制条件。

LeWM短前缀增加了规划调用次数，但总规划观测时间从67.31秒降至50.89/53.10秒；LeFlow则从4.40秒增至4.95/7.96秒。缩短评分不保证整段控制计算成本下降；这些共享设备观测不能用于精确效率归因。

单checkpoint、单cohort的探索性结果；不代表重新训练短horizon模型的效果。GPU0/1与其他作业共享，计时只作本次观测，不能作为隔离设备性能。逐条件result.json记录checkpoint/cohort/runner SHA、评分前缀审计和规划事件。

产物：`outputs/round5/phase5_baseline_horizons/`；运行入口：`scripts/round5_phase5_baseline_horizons.py`；统计入口：`scripts/round5_phase5_baseline_horizons_analysis.py`。
