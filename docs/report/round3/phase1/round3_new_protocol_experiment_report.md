# 新评测协议（Round 3 Phase 1）实验报告：原有结论复核

**报告日期**：2026-09-11  
**报告状态**：已完成  
**评测协议**：round3_revised  
**主要指标**：最终 200 个 episode 的 success rate；Stage-B 是主指标，Stage-A 与 A-shuf 用于分析规划器和目标条件控制

## 1. 目标与范围

本报告复核 Round 1 和 Round 2 报告中的实验结论，回答一个具体问题：在冻结的新评测协议下，原有结论是否仍然成立。

本轮主体工作复核已保存权重，没有根据最终测试集挑选权重；另外补充完成了 TwoRoom Round 1 的 E1–E4 训练，并按同一新协议完成验收。覆盖范围如下：

- Round 1 的 E0–E6；
- Round 2 的 E1-384、E3-384；
- Round 2 的 serial one-step、physical-time/type、clean-action timestep 等跟进实验；
- TwoRoom Round 1 的 E1–E4 补充训练，以及对应的 `round3_revised` 评测；
- 四个任务：Cube、Push-T、Reacher、TwoRoom。

结论比较优先使用相同任务、相同权重种子下的配对结果。旧协议和新协议的绝对 success rate 还受到 episode cohort、成功谓词和 episode 数量同时变化的影响，因此旧新绝对差值只作为协议影响的描述，不能直接解释成模型能力变化。

## 2. 新旧评测协议

| 维度 | Round 1/2 旧协议 | Round 3 Phase 1 新协议 |
|---|---|---|
| 最终测试规模 | 每个配置 50 个固定测试 episode | 每个任务 200 个最终测试 episode；另有 50 个 dev episode |
| 测试 cohort | 旧报告使用固定测试集，但没有记录本轮所需的完整 cohort 审计信息 | final/dev/online pool episode-disjoint；按物理距离分层；冻结 manifest 和 SHA256；排除初始即成功的 episode；不允许模型选择样本 |
| 成功谓词 | 沿用旧实验中的任务实现 | 对运行时谓词、字段、单位和边界进行审计：Cube 位置 L2 ≤ 0.04 m；Reacher 每个关节误差 < 0.05 rad；Push-T 位置 < 20 px 且角度 < π/9；TwoRoom proprioception L2 < 16 px |
| 规划预算 | seed、goal offset、receding horizon、CEM 配置沿用旧实验 | 冻结为 seed=42、goal_offset_steps=25、eval_budget=50、horizon/action_block=5、CEM 300 samples、30 steps、topk=30、var_scale=1.0 |
| checkpoint 规则 | 使用最终 checkpoint，通常为 epoch10 | 仍使用最终 checkpoint；不使用 intermediate test peak |
| 观测与审计 | 主要记录 episode 级结果 | 额外记录 per-step state、goal、distance、actions、legality、termination，并保存 pairwise matched 的 episode_id/start_step |

因此，新协议同时提高了统计分辨率、改变了测试样本构成，并把成功判定和运行轨迹审计固定下来。它适合建立新的可复现基线，但不能把新旧 rate 的差值单独归因于协议中的某一个因素。

## 3. 数据完整性与统计口径

本轮扩展评测共包含 42 个 manifest entry、93 个 stage task。93/93 个 stage task 均完成，所有结果均为 200 episode，且 save_video=true，共产出 18,600 个 episode 视频。

原 Canonical Phase 1 矩阵包含 28 个配置，其中 TwoRoom 的 E3 权重缺失，所以原矩阵中 25 个配置有有效结果、3 个配置标记为 unavailable。扩展评测补充了 E1、E2、E4、E6 以及 Round 2 权重和跟进实验；本次 TwoRoom E1–E4 是在该历史矩阵之外新增的训练与评测结果，详见第 11 节。

统计约定如下：

1. 所有百分比均为 success rate，表内保留一位小数；差值使用百分点（pp）。
2. Stage-B 是跨实验结论的主要比较对象。
3. A-shuf 是目标条件控制，不与正常 A 的绝对值混合平均。
4. 三个权重种子的均值 ± 标准差使用样本标准差；单种子结果不伪装成多种子统计。
5. 不跨任务平均原始 success rate，因为四个任务的难度和成功谓词不同。

## 4. 新协议下的主结果

下表是 200 episode 的新协议结果。E1 使用 Stage-B，E2 使用 Stage-A / A-shuf；E2 没有 Stage-B。E3、E4、E5 统一写成 A / A-shuf / B，三元组的第三项就是对应实验的 Stage-B。主表使用 seed3072 对应的主权重；Push-T 其他 E4 种子见 4.3。E6 用于观察 warm-start 的任务依赖性。TwoRoom 的 E1–E4 取本轮补充训练结果，E0、E5、E6 保持原 Canonical Phase 1 的新协议结果。

| Task | E0 B | E1 B | E2 A / A-shuf | E3 A / A-shuf / B | E4 A / A-shuf / B | E5 A / A-shuf / B | E6 warm-B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cube | 48.0% | 36.0% | 100.0% / 11.0% | 99.5% / 12.0% / 46.0% | 99.5% / 10.0% / 53.5% | 99.5% / 9.0% / 49.5% | 97.5% |
| Push-T | 93.0% | 89.0% | 90.5% / 9.0% | 94.0% / 7.5% / 84.5% | 89.0% / 11.0% / 85.5% | 93.0% / 7.0% / 89.5% | 93.0% |
| Reacher | 84.5% | 84.0% | 69.5% / 9.0% | 70.5% / 6.5% / 84.5% | 73.0% / 7.5% / 84.5% | 79.5% / 9.0% / 84.5% | 68.5% |
| TwoRoom | 84.5% | 89.0% | 92.5% / 42.5% | 95.5% / 44.5% / 90.5% | 95.0% / 45.0% / 96.5% | 95.0% / 44.0% / 95.5% | 96.5% |

新协议下，E0 的 task-level baseline 为 Cube 48.0%、Push-T 93.0%、Reacher 84.5%、TwoRoom 84.5%。E5 Stage-B 相对 E0 的差值分别为 +1.5、−3.5、0.0 和 +11.0 pp，说明 Fast Stage-B 仍然没有跨任务的统一优势，但任务间的具体关系已经发生变化。

### 4.1 Round 1 多种子配对结果

下面的比较使用新协议下同一任务、同一权重种子的结果，避免用不同权重或不同任务的总体均值替代配对比较。

| 配对比较 | 旧协议均值差 | 新协议均值差 | 新协议逐种子差值 |
|---|---:|---:|---|
| Cube：E3 B − E1 B | +11.3 ± 6.4 pp | +7.8 ± 2.6 pp | +10.0、+8.5、+5.0 |
| Push-T：E3 B − E1 B | −2.7 ± 4.2 pp | −3.8 ± 0.6 pp | −4.5、−3.5、−3.5 |
| Push-T：E5 A − E4 A | +4.7 ± 3.1 pp | +2.2 ± 1.6 pp | +4.0、+1.5、+1.0 |
| Push-T：E5 B − E4 B | −0.7 ± 6.4 pp | +1.5 ± 1.7 pp | +3.5、+0.5、+0.5 |

Cube 的 E3 正迁移、Push-T 的 actor 受世界模型监督收益仍能复现；两者幅度都比旧协议小。Push-T 的 E5 Stage-B 相对 E4 则从混合方向变成了三个种子均为正，但效应仍只有 1.5 pp。

### 4.2 E6 warm-start 配对结果

| Task | 旧协议 E6 − E5 | 新协议 E6 − E5 | 新协议判断 |
|---|---:|---:|---|
| Cube | +32 pp | +48 pp | 明显提升 |
| Push-T | −2 pp | +4 pp | 方向由略降变为略升 |
| Reacher | −16 pp | −16 pp | 明显下降 |
| TwoRoom | 0 pp | +1 pp | 接近持平 |

E6 仍然表现出强烈的 task dependence。新协议改变了 Push-T 的方向，但没有把 warm-start 变成稳定的跨任务改进。

### 4.3 E4 覆盖范围与结果

正式 Round 1 E4 使用 detach-clean-action 配置。以下结果均来自新协议的 200-episode final cohort，列顺序为 A / A-shuf / B：

| Task | 权重/种子 | A | A-shuf | B |
|---|---|---:|---:|---:|
| Cube | 0803 | 99.5% | 10.0% | 53.5% |
| Push-T | 0803 / seed3072 | 89.0% | 11.0% | 85.5% |
| Push-T | 0806 / seed3073 | 90.5% | 8.5% | 86.5% |
| Push-T | 0806 / seed3074 | 91.5% | 9.0% | 86.5% |
| Reacher | 0809 / seed3072 | 73.0% | 7.5% | 84.5% |
| TwoRoom | 0911 / seed3072 | 95.0% | 45.0% | 96.5% |

原始正式 E4 matrix 当时覆盖 Cube、Push-T 和 Reacher，TwoRoom 因缺少对应保存权重没有运行；本轮补充后，表中的 E4 覆盖已填满四个任务。所有列出的 E4 结果均为 `status=ok`，每个配置 200 episode；TwoRoom 的训练、checkpoint 和评测验收记录见第 11 节。

Round 2 的 clean-action timestep follow-up 也按新协议完成：

| Task | checkpoint | A | A-shuf | B |
|---|---|---:|---:|---:|
| Cube | epoch8 latest | 99.5% | 9.5% | 50.0% |
| Reacher | epoch10 | 75.0% | 9.5% | 83.0% |

其中 Cube follow-up 使用 epoch8 latest，因为该运行没有 epoch10 权重；Reacher 使用 epoch10。clean-action 的 B 结果相对正式 E4 分别为 Cube −3.5 pp、Reacher −1.5 pp，这支持继续把该配置作为诊断或可选方案，而不是默认 timestep。

## 5. 新协议本身对绝对结果的影响

下表把旧 Round 1 报告的 seed3072 主表与新协议下对应结果并列。TwoRoom 旧 E0 使用旧报告主表中的 86%；Round 3 的 legacy 50-episode 行为 88%，说明旧结果源之间也存在 cohort 差异，所以这里的差值应视为描述性对照。

| Task | 旧 E0 B | 新 E0 B | Δ | 旧 E3 B | 新 E3 B | Δ | 旧 E5 B | 新 E5 B | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cube | 78.0% | 48.0% | −30.0 pp | 74.0% | 46.0% | −28.0 pp | 66.0% | 49.5% | −16.5 pp |
| Push-T | 98.0% | 93.0% | −5.0 pp | 90.0% | 84.5% | −5.5 pp | 86.0% | 89.5% | +3.5 pp |
| Reacher | 72.0% | 84.5% | +12.5 pp | 82.0% | 84.5% | +2.5 pp | 84.0% | 84.5% | +0.5 pp |
| TwoRoom | 86.0% | 84.5% | −1.5 pp | — | — | — | 100.0% | 95.5% | −4.5 pp |

主要观察有三点：

- Cube 的绝对 success rate 大幅降低，且 E0、E3、E5 同时降低。这更像是新 cohort、排除初始成功和成功判定共同带来的测量变化，不能解释成 Cube 模型突然退化。
- Reacher 的 E0 baseline 从 72.0% 上升到 84.5%，使旧报告中 Fast Stage-B 相对 LeWM 的明显优势消失。
- Push-T 和 TwoRoom 的变化较小，但仍足以改变几个边界结论。新协议下应使用新的 E0 baseline，而不是把旧协议的绝对阈值继续沿用。

A-shuf 的绝对值也会随 cohort 改变：例如 Cube 的 E3 A-shuf 从旧报告的约 42% 降到 12%，Push-T 的 E3 从 4% 变为 7.5%，Reacher 的 E3 从 4% 变为 6.5%，TwoRoom 的 E5 从 38% 变为 44%。控制实验的总体含义仍然成立，但 A-shuf 的具体百分比不能跨协议直接比较。

## 6. Round 1 结论复核

| Round 1 原结论 | 新协议证据 | 判定 | 更新后的表述 |
|---|---|---|---|
| Cube 存在稳定的 E3 相对 E1 的单向正迁移 | 三个种子均为正；均值 +7.8 ± 2.6 pp | 确认，但幅度减弱 | Cube 上的正迁移可复现，效应约为 8 pp，仍低于旧协议约 11 pp 的估计 |
| Action supervision 的 planner gain 不跨任务泛化 | Push-T −3.8 ± 0.6 pp；Reacher 单种子 +0.5 pp | 确认并略加强 | 没有证据支持 action supervision 带来普遍 Stage-B 收益 |
| 世界模型监督帮助 actor | Push-T 三个种子 E5 A 均高于 E4 A，+2.2 ± 1.6 pp | 确认，但幅度减弱 | Push-T 的 Stage-A 收益仍存在，但新协议下只有小幅收益 |
| Cross-head gradient 没有稳定 planner gain | Push-T 三个种子均为正，E5 B − E4 B 为 +1.5 ± 1.7 pp；Cube 为 −4 pp，Reacher为 0 | 局部改变 | Push-T 出现弱正向信号，但没有跨任务、足够大的稳定收益，暂不改成默认配置 |
| Goal conditioning 有效 | Cube、Push-T、Reacher 的 A-shuf 分别为 12.0%、7.5%、6.5%；TwoRoom E5 为 44.0% | 确认 | 目标打乱后成功率仍显著降低；Cube 和 TwoRoom 保留的非零值说明任务先验仍存在 |
| E6 actor warm-start 强烈依赖任务 | 新协议增量为 Cube +48、Push-T +4、Reacher −16、TwoRoom +1 pp | 确认 | E6 必须按任务验证，不能作为跨任务默认改进 |
| Fast Stage-B 没有总体超过 LeWM | E5 B 相对 E0 为 Cube +1.5、Push-T −3.5、Reacher 0、TwoRoom +11 pp | 总体确认，任务图改变 | 仍没有统一优势；Cube 从旧协议的劣势变为轻微优势，Reacher 从旧协议的优势变为持平 |
| 25D action 更可能从 transfer 获益，10D 任务不明显 | Cube E3 B − E1 B 为 +7.8 pp；Push-T 为 −3.8 pp；Reacher 为 +0.5 pp | 保留为支持性证据 | 结果与 action dimension 假设相容，但仍不是因果结论，且 E6 的 warm-start 不能直接用于该因果比较 |
| 不应按 intermediate test peak 选 checkpoint | 本轮只评测冻结的最终 checkpoint | 未由本轮独立检验 | 评测协议继续保持 final-checkpoint 规则；若要重新验证峰值选择问题，需要保留完整训练轨迹 |
| 参数和速度结论 | 本轮没有受控的旧新运行时基准 | 未检验 | 不从本轮结果更新参数效率或速度结论 |

Round 1 的核心结构仍然存在：Cube 是最清晰的正迁移任务，Push-T 的 actor 监督收益较小，E6 具有任务依赖性。变化集中在效应大小和 Push-T 的 cross-head gradient：后者有了弱正向趋势，但证据仍不足以支持默认化。

## 7. Round 2 结论复核

### 7.1 Expansion 与 E0 的比较

| Task | 新 E0 B | 新 E1-384 B | 新 E3-384 A / A-shuf / B | E1-384 − E1-192 | E3-384 − E3-192 | E3-384 B − E0 |
|---|---:|---:|---:|---:|---:|---:|
| Push-T | 93.0% | 85.5% | 91.0% / 8.5% / 88.5% | −3.5 pp | +4.0 pp | −4.5 pp |
| Reacher | 84.5% | 81.0% | 73.5% / 7.5% / 82.0% | −3.0 pp | −2.5 pp | −2.5 pp |

旧报告中 E1-384 相对 E1-192 的差值为 Push-T −4、Reacher 0 pp；E3-384 相对 E3-192 为 Push-T +2、Reacher +2 pp；Reacher E3-384 曾以 84% 高于旧 E0 的 72%。

新协议下得到的判断如下：

| Round 2 原结论 | 新协议证据 | 判定 | 更新后的表述 |
|---|---|---|---|
| 单纯增加 Stage-B expansion capacity 不是主要 gap | E1-384 相对旧 E1-192：Push-T −3.5 pp、Reacher −3.0 pp；仍低于新 E0 | 确认并加强 | 增加 capacity 没有修复纯 Stage-B gap，两个任务都没有得到收益 |
| E3-384 让 Reacher 成为唯一超过 E0 的配置 | Reacher E3-384 B=82.0%，低于新 E0=84.5%；Push-T=88.5%，也低于 E0=93.0% | 推翻 | 旧协议下的 Reacher +12 pp 优势不能在新协议中复现；当前没有证据表明 E3-384 超过 E0 |
| E3-384 相对 E3-192 有小幅收益 | Push-T +4.0 pp；Reacher −2.5 pp | 改为任务依赖 | expansion 对 E3 的影响方向依赖任务，不能总结成普遍收益 |
| A-shuf 在 4–12% 范围内，目标条件正常 | 新 E3-384 为 Push-T 8.5%、Reacher 7.5% | 确认 | 新 cohort 下绝对值略变，但目标打乱控制仍有效 |
| Reacher clean-action timestep 不应作为默认 | E4 clean-action B=83.0%，原 E4 B=84.5%，下降 1.5 pp | 确认 | 继续保留为可选诊断配置，不作为默认 timestep |
| Learned physical-time/type 只能作为 optional | Push-T 从 84.5% 到 89.5%，+5 pp；Reacher 从 84.5% 到 83.5%，−1 pp | 确认，但任务差异更清楚 | Push-T 有值得跟进的正向信号，Reacher 没有收益，不能全局切换 |
| Serial one-step 不应成为全局默认 | Push-T 从 89.0% 到 90.5%，+1.5 pp；Reacher 从 84.0% 到 68.5%，−15.5 pp | 确认并加强 | serial 的收益仍不统一，Reacher 的负面影响在新协议下更明显 |
| causal-prefix 不是统一 bottleneck | 本轮没有新增针对 causal-prefix 的因果干预 | 未独立检验 | 该结论仍沿用 Round 2 诊断，不能宣称由本轮最终评测重新证明 |
| 不按中间 test peak 选 expansion checkpoint | 本轮使用最终 checkpoint | 未由本轮独立检验 | 继续使用 final-checkpoint 规则，保留中间曲线以供后续诊断 |

Round 2 中最需要更新的是 Reacher E3-384 的结论：它原来是“唯一超过 E0”的正面结果，新协议下变成低于 E0。E1 capacity 的负面结论仍然稳定；E3 expansion 则从“两个任务都有小幅收益”变为明显的任务依赖。

### 7.2 Round 2 跟进实验

| 跟进配置 | 新协议参考值 | 新协议跟进值 | 新 Δ | 旧报告 Δ | 判断 |
|---|---:|---:|---:|---:|---|
| Push-T serial one-step vs E1-192 | 89.0% | 90.5% | +1.5 pp | +10 pp | 方向仍为正，但幅度大幅缩小 |
| Reacher serial one-step vs E1-192 | 84.0% | 68.5% | −15.5 pp | −12 pp | 负面影响仍存在，且更明显 |
| Push-T physical-time/type vs E3-192 | 84.5% | 89.5% | +5.0 pp | 0 pp | 新协议给出值得跟进的任务特定正向信号 |
| Reacher physical-time/type vs E3-192 | 84.5% | 83.5% | −1.0 pp | −8 pp | 没有形成通用收益 |
| Reacher clean-action timestep vs E4 | 84.5% | 83.0% | −1.5 pp | −2 pp | 继续不作为默认 |
| Cube clean-action timestep vs E4 | 53.5% | 50.0% | −3.5 pp | 无旧最终对照 | 新协议下没有正向迹象；该权重为 epoch8 latest follow-up |

## 8. 综合判断：改变评测协议后，原有结论是否变化

答案是：**核心方向大部分保持不变，但绝对基线和几个局部结论已经变化，旧报告中的具体百分点不能继续直接使用。**

| 状态 | 结论 |
|---|---|
| 基本不变 | 没有普遍的 action-supervision planner gain；单纯增大 E1 Stage-B capacity 不能解决 gap；目标条件控制有效；E6 具有明显 task dependence；serial one-step 不能全局默认 |
| 幅度变弱 | Cube E3 相对 E1 的正迁移从 +11.3 pp 降为 +7.8 pp；Push-T E5 相对 E4 的 actor 收益从 +4.7 pp 降为 +2.2 pp |
| 局部改变 | Push-T E5 相对 E4 的 Stage-B 差值从旧协议的混合方向变为 +1.5 pp 的弱一致正向；Push-T physical-time/type 在新协议中为 +5 pp |
| 结论被改写 | Reacher 的 Fast Stage-B 相对 E0 优势消失；E3-384 “唯一超过 E0”的 Reacher 结果不再成立；E3-384 的 expansion 收益不再是跨任务正向 |
| 尚不能由本轮检验 | intermediate peak 选 checkpoint、参数效率、运行速度、causal-prefix 的因果瓶颈结论 |

因此，如果要把旧报告更新成一句可继续使用的话，可以写成：

> 新评测协议保留了原有的高层结论：Fast-LeWAM 的收益具有任务依赖性，目标条件有效，单一 Stage-B capacity 或 serial rollout 不能普遍解决规划 gap。新协议同时显示，Cube 的正迁移和 Push-T 的 actor 收益较旧估计更小，Reacher 的 Fast 相对 E0 优势以及 E3-384 的 Reacher 优势不能复现；Push-T 的 cross-head gradient 和 physical-time/type 只提供了小幅、任务特定的后续研究信号。

## 9. 限制

1. 新协议把 episode 数量、cohort 构成、成功谓词和轨迹审计一起改变，所以旧新绝对差值不是单因素因果实验。
2. 没有在新协议中同时重跑旧 cohort，因此无法把“协议效应”和“模型在不同 episode 上的表现”完全分离。
3. 多种子覆盖不均匀。Cube 和 Push-T 的部分 Round 1 配对有三个权重种子；Reacher、TwoRoom 以及多数 Round 2 跟进实验主要是单种子。
4. 原 Canonical Phase 1 矩阵的 TwoRoom E3 权重曾缺失，无法完成历史矩阵中的 E3–E0/E5 全对照；本次补充训练已产生 E1–E4 权重，但这些结果属于新增的 Round 1 补充矩阵，不能回写原 Canonical Phase 1 统计。
5. Cube clean-action follow-up 没有 epoch10 权重，使用的是 epoch8 latest，不能与标准 final-checkpoint 结果做完全同质的 checkpoint 比较。
6. 本报告主体是评测复核，主体结论描述的是现有权重在新协议下的行为，不是新协议下重新优化后的最终模型上限；TwoRoom E1–E4 的新增训练和验收单独记录在第 11 节。

## 10. 后续采用建议

- 后续正式表格和新实验以 round3_revised 的 200-episode final 结果为 canonical baseline，同时保留旧协议结果用于历史追踪。
- 删除或改写“Reacher E3-384 唯一超过 E0”的表述，不再用它作为扩容方案的主要依据。
- 保留 Cube 的正迁移、Push-T 的 actor 监督收益和 E6 task dependence，但使用新协议下的效应大小。
- Push-T 的 cross-head gradient 与 physical-time/type 进入候选跟进列表，暂不切换为全局默认；需要多种子和同协议配对实验确认。
- E6、serial 和 timestep 相关配置继续按任务单独验证。
- 若要更新速度、参数量或 intermediate peak 结论，应补做受控 benchmark 或保存训练全过程，不能从本轮最终 success rate 推断。

## 11. TwoRoom Round 1 E1–E4 补充训练与验收

本节记录为补齐 TwoRoom Round 1 缺项而完成的四个实验。训练统一使用 seed=3072、`latent_head_dim=192`、MLP dim=768、batch size=128、10 个 epoch 和最终 epoch10 checkpoint；E1 为 Stage-B only，E2 为 Stage-A only，E3 为 expert-action supervision，E4 为 detach-clean-action。E2、E3、E4 在 GPU0、GPU2、GPU3 上并行训练，未使用 GPU4–7。

新协议使用 `round3_revised` 的 final cohort，每项 200 episode；B 表示 Stage-B，A-shuf 表示 shuffled-goal Stage-A。所有结果均为 `status=ok` 且包含 200 个 episode。

| 实验 | 新协议 A | 新协议 A-shuf | 新协议 B |
|---|---:|---:|---:|
| E1 | — | — | **89.0%** |
| E2 | **92.5%** | **42.5%** | — |
| E3 | **95.5%** | **44.5%** | **90.5%** |
| E4 | **95.0%** | **45.0%** | **96.5%** |

补充旧协议使用 `legacy_50` dev cohort，每项 50 episode；所有结果同样为 `status=ok`。

| 实验 | 旧协议 A | 旧协议 A-shuf | 旧协议 B |
|---|---:|---:|---:|
| E1 | — | — | **86.0%** |
| E2 | **94.0%** | **36.0%** | — |
| E3 | **98.0%** | **42.0%** | **90.0%** |
| E4 | **98.0%** | **40.0%** | **98.0%** |

验收项如下：

| 验收项 | 结果 |
|---|---|
| E1–E4 epoch10 checkpoint | 4/4 存在且可加载 |
| 新协议结果 | 9/9 `status=ok`，每项 200 episode |
| 旧协议补充结果 | 9/9 `status=ok`，每项 50 episode |
| GPU 限制 | 仅使用 GPU0–GPU3 |

训练产物位于 `outputs/fast_lewam/tworoom/`；评测产物位于 `outputs/round3/extended/evals/`。部分训练 wrapper 在训练完成后的 shell 收尾阶段报告了 `wandb.enabled=false: command not found`，但 epoch10 checkpoint 已完整保存，随后已直接复用 checkpoint 完成全部评测并通过上述验收。

## 12. 相关产物

- [Round 3 Phase 1 协议](../../../plan/round3_phase1_protocol.md)
- [Round 3 实验计划](../../../plan/round3_experiment_plan.md)
- [Canonical Phase 1 报告](../../../../outputs/round3/phase1/phase1_report.md)
- [Canonical Phase 1 结果 CSV](../../../../outputs/round3/phase1/phase1_final_matrix.csv)
- [扩展评测报告](../../../../outputs/round3/extended/extended_eval_report.md)
- [扩展评测汇总 CSV](../../../../outputs/round3/extended/extended_eval_summary.csv)
- [扩展评测 manifest](../../../../outputs/round3/extended/extended_eval_manifest.json)
- [Round 1 原报告](../../round1_experiment_report.md)
- [Round 2 原报告](../../round2_experiment_report.md)
