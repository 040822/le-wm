# Reacher 动作边界快速对照报告

## 实验说明

本实验使用 R4-AB epoch10、训练 seed3072 checkpoint，在固定的 `round3_revised` Reacher dev cohort（50 episodes）上进行 inference-only 对照。没有重新训练，也没有评测 final cohort。

Fast-LeWAM 原有 z-score action normalizer 保持不变；新增操作只是在归一化动作空间中加入物理 action-space 边界投影。四个方法都使用各自的 `*_none` 作为 paired baseline：P0 直接执行 A，P1 为随机初始化 CEM，P2 为 A warm-start CEM，P3 为 A best-of-64 再由 B 选优。

`abs(action_norm)>1` 仅作为历史 proxy 记录。真实归一化边界由 Reacher 物理 action space 和现有 `StandardScaler` 计算，不能把 proxy 直接称为环境越界。

## 归一化边界

Reacher 的物理 action space 是 `[-1, 1]`。现有 scaler 映射后，两个基础 action 维度的真实边界约为：

| 维度 | mean | scale | 归一化物理下界 | 归一化物理上界 |
|---|---:|---:|---:|---:|
| 0 | 0.0000357 | 0.5774141 | -1.7319213 | 1.7317975 |
| 1 | -0.0005056 | 0.5771827 | -1.7316777 | 1.7334297 |

## 对照设置

| variant | 说明 |
|---|---|
| P0-none | A 直接执行，Euler-16，无投影；复用已有结果 |
| P0-clip | A 生成的整条归一化动作路径逐元素裁剪到真实边界 |
| P0-scale | A 生成的整条路径使用单一比例缩放到真实边界内 |
| P1-none | 随机初始化 CEM，300 samples、topk30、30 iterations；复用已有结果 |
| P1-candidate-clip | CEM 每轮的候选在 B 评分和 elite 更新前逐元素裁剪 |
| P2-none | A warm-start + B/CEM，Euler-16，无投影；复用已有结果 |
| P2-warm-clip | 仅裁剪 A warm-start，CEM 样本保持原逻辑 |
| P2-candidate-clip | warm-start 与每轮 CEM candidate 均裁剪，B 评分和 elite 更新使用裁剪后 candidate |
| P2-scale | 仅对 A warm-start 做单一比例缩放 |
| P3-none | A best-of-64 + B 选优，Euler-16；复用已有结果 |
| P3-clip | A 生成的 64 条动作候选逐元素裁剪后再交给 B verifier |
| P3-scale | A 生成的每条候选路径用单一比例缩放后再交给 B verifier |

所有结果均使用相同 checkpoint、cohort、eval seed42、goal offset25、eval budget50、horizon5、receding horizon5、action block5；P1/P2 CEM 为 300 samples、topk30、30 iterations，`solver_batch_size=1`；P3 候选数为64。

## Dev 结果

“投影输入越界率”对 P0/P2 warm-start 表示被投影的动作路径，对 P1/P2 candidate 表示原始 CEM candidate，对 P3 表示 A 的原始候选；“候选修改率”只对 candidate/path 投影有意义。

| variant | 方法 | 边界处理 | 成功率 | 物理动作越界率 | 投影输入真实越界率 | 原始候选越界率 | 候选修改率 | planning median(s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| P0-none | P0 | none | 70.0% | 0.0009 | — | — | — | — |
| P0-clip | P0 | clip | 70.0% | 0 | 0.001045 | — | — | 0.226 |
| P0-scale | P0 | global-scale | 70.0% | 0 | 0.001045 | — | — | 0.070 |
| P1-none | P1 | none | 74.0% | 0.0098 | — | — | — | 4.946 |
| P1-candidate-clip | P1 | candidate-clip | 88.0% | 0 | 0.012465 | 0.012465 | 0.012465 | 5.522 |
| P2-none | P2 | none | 50.0% | 0.1278 | — | — | — | 5.521 |
| P2-warm-clip | P2 | warm-start-clip | 50.0% | 0.1278 | 0.000926 | — | — | 4.814 |
| P2-candidate-clip | P2 | candidate-clip | 80.0% | 0 | 0.022548 | 0.022548 | 0.022548 | 5.789 |
| P2-scale | P2 | warm-start-scale | 52.0% | 0.1281 | 0.000926 | — | — | 4.985 |
| P3-none | P3 | none | 86.0% | 0.0007 | — | — | — | 0.441 |
| P3-clip | P3 | clip | 84.0% | 0 | 0.001338 | 0.001338 | 0.001338 | 0.304 |
| P3-scale | P3 | global-scale | 84.0% | 0 | 0.001338 | 0.001338 | 0.064662 | 0.310 |

## Episode-level paired 结果

| comparison | 成功率变化 | improved | regressed | net | McNemar exact two-sided p |
|---|---:|---:|---:|---:|---:|
| P0-clip vs P0-none | 0pp | 0 | 0 | 0 | — |
| P0-scale vs P0-none | 0pp | 0 | 0 | 0 | — |
| P1-candidate-clip vs P1-none | +14pp | 10 | 3 | +7 | 0.09229 |
| P2-warm-clip vs P2-none | 0pp | 0 | 0 | 0 | — |
| P2-candidate-clip vs P2-none | +30pp | 18 | 3 | +15 | 0.00149 |
| P2-scale vs P2-none | +2pp | 1 | 0 | +1 | 1.00000 |
| P3-clip vs P3-none | -2pp | 0 | 1 | -1 | 1.00000 |
| P3-scale vs P3-none | -2pp | 0 | 1 | -1 | 1.00000 |

## 结论

1. **P0：** clip 和 global-scale 都保持 70.0%。它们把物理动作越界率从 0.0009 降到0，但没有改变成功率；P0 的少量真实越界不是该 cohort 的主要失败原因。
2. **P1：** candidate-clip 从 74.0% 提升到 88.0%（+14pp），paired improved=10、regressed=3，但 50 episodes 下的 McNemar `p=0.09229`，应视为积极但尚未稳定的探索信号。CEM 原始候选真实归一化越界率为 1.246%，投影后为0，物理动作越界也为0。
3. **P2：** candidate-clip 从 50.0% 提升到 80.0%（+30pp），paired improved=18、regressed=3，`p=0.00149`；只裁剪 warm-start 为50.0%，global-scale 为52.0%。这再次说明关键作用位于 CEM candidate 生成、B 评分和 elite 更新的一致约束，而不是只处理 actor warm-start。
4. **P3：** clip 和 global-scale 都从 86.0% 降到84.0%，各自仅有1个 paired regression；没有观察到正收益。P3 baseline 的物理越界本来就只有0.0007%，因此边界投影消除极少量越界不足以改善闭环成功率。
5. **跨方法解释：** P1/P2 的 candidate-clip 会改变 CEM 的搜索分布、B 的评分输入和 elite 更新轨迹；P3 的投影会改变 B 的评分输入和最终 argmin。因此成功率变化不能简单归因于“执行阶段避免越界”。本轮结果支持继续优先研究 CEM candidate 约束，但不支持将边界投影默认加入 P3。

## 限制与后续

这是单 checkpoint、单训练 seed、50 条 dev episode 的探索性结果，不是 final 泛化证明，也不宣称动作边界处理带来稳定提升。下一步若要确认 P1 candidate-clip 的收益，应优先在保持同一 cohort/协议下补充训练或评测 seed；P3 的投影暂不值得扩展到更大候选预算，除非出现新的越界证据。

## 可复现产物

- 对照配置：[config/round4/reacher_action_bounds_compare.json](/data/users/wenxin/pre-exp/le-wm/config/round4/reacher_action_bounds_compare.json)
- standalone evaluator：[scripts/round4_reacher_action_bounds.py](/data/users/wenxin/pre-exp/le-wm/scripts/round4_reacher_action_bounds.py)
- 单元测试：[tests/test_round4_action_bounds.py](/data/users/wenxin/pre-exp/le-wm/tests/test_round4_action_bounds.py)
- 结果汇总：`outputs/round4/reacher_action_bounds_compare_seed3072_dev/summary.json`
- 机器生成报告：`outputs/round4/reacher_action_bounds_compare_seed3072_dev/report.md`
- checkpoint SHA256：`087991339c9499d10c1a58a28e72d07c7dfd819d5f072c67bfd0169acaa83553`
- cohort SHA256：`d18695b2377000ce7dd774fcad78e4bb2d75980fb3ccbd98cc83d09ef50f1d06`
