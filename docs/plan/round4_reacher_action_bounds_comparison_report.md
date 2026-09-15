# Round 4 Reacher 动作边界快速对照报告

## 结论摘要

本实验使用 R4-AB Reacher epoch10、训练 seed3072 checkpoint，在固定的 `round3_revised` dev cohort（50 episodes）上进行 inference-only 对照。没有重新训练，也没有评测 final cohort。

当前 Fast-LeWAM 已经使用基于完整数据集拟合的 z-score action normalizer。本实验没有替换 normalizer，只在归一化动作空间中加入物理 action-space 边界投影。

最重要的结果是：

- P2 baseline 为 **50.0%**；将 warm-start 和每轮 CEM candidate 都在送入 B verifier 前裁剪（`P2_candidate_clip`）后为 **80.0%**，提升 **30pp**。
- episode-level paired comparison 为 `improved=18`、`regressed=3`，McNemar exact two-sided `p=0.00149`。
- P2 的物理动作越界率从 **0.1278** 降到 **0**。
- 只裁剪 P2 warm-start 为 **50.0%**，global-scale warm-start 为 **52.0%**，均没有复现 candidate-clip 的收益。
- P0 的 clip 和 global-scale 都保持 **70.0%**；物理越界率从 **0.0009** 降到 **0**，但成功率不变。

因此，当前证据支持“约束 CEM candidate 搜索空间有益”，而不是简单支持“裁剪 actor warm-start 就能解决问题”。P2 candidate-clip 的收益也可能来自改变 B verifier 看到的候选分布和 CEM elite 更新轨迹，不能仅归因于消除环境越界。

## 动作归一化口径

Reacher 的物理 action space 是 `[-1, 1]`。现有 `StandardScaler` 的统计量为：

| 维度 | mean | scale | 归一化物理下界 | 归一化物理上界 |
|---|---:|---:|---:|---:|
| 0 | 0.0000357 | 0.5774141 | -1.7319213 | 1.7317975 |
| 1 | -0.0005056 | 0.5771827 | -1.7316777 | 1.7334297 |

所以历史诊断中使用的 `abs(action_norm)>1` 只能作为 legacy unit-threshold proxy，不能称为真实环境越界。本实验同时记录了 proxy、真实归一化边界越界和 inverse-transform 后的物理越界。

## 对照设置

| variant | 说明 |
|---|---|
| P0-none | A 直接执行，Euler-16，无投影；复用已有结果 |
| P0-clip | A 整条归一化动作路径逐元素裁剪到真实边界 |
| P0-scale | A 整条路径使用单一比例缩放到真实边界内 |
| P2-none | A warm-start + B/CEM，Euler-16，无投影；复用已有结果 |
| P2-warm-clip | 仅裁剪 A warm-start，CEM 样本保持原逻辑 |
| P2-candidate-clip | warm-start 与每轮 CEM candidate 均裁剪，B 评分和 elite 更新使用裁剪后 candidate |
| P2-scale | 仅对 A warm-start 做单一比例缩放 |
| P3-none | A best-of-64 + B 选优，Euler-16；复用已有结果 |

所有结果均使用相同 checkpoint、cohort、eval seed42、goal offset25、eval budget50、horizon5、receding horizon5、action block5；P2 CEM 为 300 samples、topk30、30 iterations、`solver_batch_size=1`。

## Dev 结果

| variant | 成功率 | 物理动作越界率 | warm-start 真实归一化越界率（前） | CEM 原始候选越界率 | planning median(s) |
|---|---:|---:|---:|---:|---:|
| P0-none | 70.0% | 0.0009 | — | — | — |
| P0-clip | 70.0% | 0 | 0.00105 | — | 0.226 |
| P0-scale | 70.0% | 0 | 0.00105 | — | 0.070 |
| P2-none | 50.0% | 0.1278 | — | — | 5.521 |
| P2-warm-clip | 50.0% | 0.1278 | 0.00093 | — | 4.814 |
| P2-candidate-clip | 80.0% | 0 | 0.00077 | 0.02255 | 5.789 |
| P2-scale | 52.0% | 0.1281 | 0.00093 | — | 4.985 |
| P3-none | 86.0% | 0.0007 | — | — | 0.441 |

## Paired 结果

| comparison | improved | regressed | net |
|---|---:|---:|---:|
| P0-clip vs P0-none | 0 | 0 | 0 |
| P0-scale vs P0-none | 0 | 0 | 0 |
| P2-warm-clip vs P2-none | 0 | 0 | 0 |
| P2-candidate-clip vs P2-none | 18 | 3 | +15 |
| P2-scale vs P2-none | 1 | 0 | +1 |

## 解释与限制

1. P0 的真实越界本来就很少，裁剪只消除了少量非法动作，没有改善闭环成功率。
2. P2 warm-start 的真实归一化越界也只有约 0.08%，单独裁剪 warm-start 没有收益；P2 baseline 的大量物理越界来自后续 CEM 生成/选择出的动作序列。
3. P2 candidate-clip 同时改变了 CEM 的搜索分布、B 的评分输入和 elite 更新，因此它是“有边界约束的 CEM”对照，不是纯粹的执行层 clamp。
4. 这是单 checkpoint、单训练 seed、50 条 dev episode 的探索性结果。`P2_candidate_clip` 可作为后续方法候选，但暂不覆盖默认 Round4 evaluator，也不据此宣称稳定性能提升。

## 可复现产物

- 对照配置：`config/round4/reacher_action_bounds_compare.json`
- standalone evaluator：`scripts/round4_reacher_action_bounds.py`
- 单元测试：`tests/test_round4_action_bounds.py`
- 结果汇总：`outputs/round4/reacher_action_bounds_compare_seed3072_dev/summary.json`
- 中文机器生成报告：`outputs/round4/reacher_action_bounds_compare_seed3072_dev/report.md`
- checkpoint SHA256：`087991339c9499d10c1a58a28e72d07c7dfd819d5f072c67bfd0169acaa83553`
- cohort SHA256：`d18695b2377000ce7dd774fcad78e4bb2d75980fb3ccbd98cc83d09ef50f1d06`
