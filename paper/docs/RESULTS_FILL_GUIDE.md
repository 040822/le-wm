# CoWM 结果填充指南

更新：2026-10-01。用途：在固定论文结构中填后续结果，不作为实验启动清单。

## 1. 六张主表

| 主表 | 问题 | 文件 | 字段前缀 |
|---|---|---|---|
| 1 | 系统相对规划基线如何？ | `tables/final_main.tex` | `main-*` |
| 2 | 完整联合训练有什么贡献？ | `tables/final_training.tex` | `train-*`、`training-*` |
| 3 | 可用候选能否被正确选择？ | `tables/final_ranking.tex` | `pool-*`、`rank-*`、`ranking-*` |
| 4 | 观测/预测 latent 中物理量是否可读？ | `tables/final_probe.tex` | `probe-*` |
| 5 | P0、随机选择、P3、PO/GF 有何差别？ | `tables/final_inference.tex` | `infer-*`、`inference-*` |
| 6 | 对应成功率下推理成本如何？ | `tables/final_efficiency.tex` | `eff-*`、`efficiency-*` |

Phase1/1.5/1.6 详细表、曲线和短训干预移入 `sec/7_reference_results.tex`，由补充材料引用。正文表 5 保留 Phase1.6 实测参考面板；新完整训练结果填独立面板，不合并统计。

Phase1.7 epoch-10 dev 与在线结果在 `sec/6_appendix.tex`：开发检查不填最终主表，在线数据改进不填原离线训练收益。

## 2. 统一填写入口

编辑 `paper/results.tex`，表格和正文通过同名字段取值：

```tex
% 百分比格只填数值，单位由表头/图注给出：
\SetPaperResult{train-joint-pusht-p3}{94.0}

% 正文效应和区间显式带单位。以下只是格式示例：
\SetPaperResult{training-joint-bonly-effect}{+3.0 pp}
\SetPaperResult{training-joint-bonly-ci}{[$-$1.0, +7.0] pp}
```

蓝色 `TBD` 表示等待同协议结果，绝不表示零。横线表示不适用、未测或不在已声明范围；B-only 的 Own-A P0、未测物理量等不填零。

摘要主要效应、延迟和引言效应引用主字段。只有来源协议相同才能保留别名；若摘要选了不同 cohort，先统一口径，不能靠多个字段写出不同结论。

`abstract-*`、`intro-*`、`discussion-*`、`conclusion-*` 仍有少量解释字段。它们填写与实际效应一致的短英文判断，并非新实验。

## 3. 最小填充顺序

1. 填 `protocol-completed-scope`：实际任务、模型、训练 seed、epoch、起点数、推理 seed、训练数据暴露及复用；填 `protocol-test-family` 的实际统计族。
2. 收齐已规划完整训练终点，填表 2 和同 A 的 Joint-B/LeWM 对照。不将短训、epoch-2、在线臂填入完整离线训练单元。
3. 填同一物理池的覆盖、top-1、regret 和连续指标，记录 actor/checkpoint、有效候选与终止处理。
4. 填已执行且同协议的主基线、读出、推理、计时。无法匹配的历史结果留在补充材料。
5. 根据效应和区间填写解释字段，更新摘要/结论，说明或删除未执行的条件性行，复核来源和编译。

未完成项明确原因；已完成负结果保留。不能因表现不佳从原声明宏平均任务集中删任务。每个任务/模型实际完成 seed 数分别报告，不默认所有单元都具有两个训练 seed。

## 4. 各表口径

**系统表现。** 表 1 保留笔记中的 LeWM/LeFlow/DeWM 基线位置及同 A + LeWM 对照；占位不意味着另开实验。仅填已获得同协议结果，未执行项说明缺失并收窄比较范围。记录训练次数、数据、动作范围、历史及搜索预算。P0 不等于独立 A-only。

**训练归因。** Joint vs Recorded-control 同时改变输入来源和动作路径梯度，只写联合改变的效果。Recorded-clean vs B-only 检验干净监督下加入 A 任务；Recorded-control vs Recorded-clean 同时改变 timestep/weight。

Pred-Detach、双 DiT、进一步匹配 A-only 为条件性扩展。`optional-*` 默认横线；无结果时限制相关主张，不为填表自动启动。双 DiT 的总参数和计算不同，不称完全等容量比较。

**排序与修正。** Regret 使用同池同状态配对差。不同 scorer 的有效状态集合可能不同，不能直接相减各自平均 regret 替代配对分析。Continuous 排序改善而 top-1/闭环未改善，分别陈述。候选、推理 seed、读出 seed 不是独立训练重复。

PO/GF 记录合法处理后的实际 RMS、有效状态数、真实物理变化、预测改善但执行变差比例。局部收益不自动填为新模型闭环收益。

**物理读出。** Cube 当前只有 block position，其他行默认横线，不强制补完。有新增量才填写字段、单位、读出类型和划分。各模型使用可比协议；原论文归一化 MSE 与本地 MAE 不直接比较。Observed/Predicted 分列。

**推理。** 表 5 新面板针对实际完成匹配训练任务；宏平均不与四任务参考面板混用。GF 未复验可留未测，不开新网格。主模型最终选取规则和对应训练配置写入 `inference-operating-point`。

**效率。** 表 6 Success 对应同模型/预算闭环，延迟对应隔离完整 policy 调用。全脚本时间、batch-200 耗时不当作 batch-1 延迟。填写搜索臂究竟使用 Joint-B 还是 LeWM、初始化和样本/迭代预算。

替代门槛沿用：宏平均单侧 95% 下界 >−2 pp，声明任务下界 >−4 pp，p50 至少降低 20%。未通过则报告速度事实及性能区间，不写成功替代。延迟字段不能由并发评测估算。

## 5. 解释字段的收尾

| 后续结果 | 允许结论 | 不可自动推出 |
|---|---|---|
| 匹配闭环有稳定收益 | 指定任务、模型和预算下有决策收益 | 三种耦合分别有益、普遍泛化 |
| 点估计好但区间跨零 | 收益未确定，报告效应/区间 | 等价、非劣或已证明无效 |
| 连续排序好、控制未改善 | 排序与控制收益分离 | 控制成功率已提高 |
| Recorded-control/clean 更好 | 较简单训练在该设置更合适 | Full 必然最佳 |
| 更快但性能门槛未过 | 更低延迟，替代未确认 | 以速度掩盖性能差距 |
| 多任务无稳定收益 | 假设未获支持，收窄主张 | 必须增加任务/seed 找到正结果 |

先填实际数值，再填短判断。`conclusion-supported-claim` 写当前证据支持什么，`conclusion-unresolved-scope` 写仍未回答什么；摘要通过别名保持一致。填数后仍需核对这一轮解释。

## 6. 来源和交付

历史来源在 `evidence_snapshot.json`、`phase1_6_evidence_snapshot.json`；本次引用在 `phase1_7_evidence_snapshot.json`。新增数据记录 source JSON、checkpoint/cohort hash 和统计脚本；不要把可变进度报告当唯一数值来源。

修改前 11 个文件备份在 `archive/pre_fillable_20261001.zip`。`main.tex` 和 `supplement.tex` 均加载 `results.tex`，分别编译。内部指南与证据账本不进入匿名 Overleaf 包。
