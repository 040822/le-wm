# Round5 Phase1.5 阶段性实验报告

**数据快照：2026-09-24 20:31（Asia/Shanghai）**  
**阶段性质：** 探索性中期总结；不构成最终方法选择或泛化结论。

## 1. 范围与冻结口径

本轮按 [Phase1.5 计划](../../plan/round5_phase1.5_plan.md) 使用四个任务的 R4-AB、训练 seed3072、epoch10 权重和 `legacy_50` 固定评测 cohort；闭环主扫描使用推理 seed42。报告使用计划中的 Phase1 legacy 协议称谓，不主张完全复现 LeWM 论文协议。所有结果仍来自同一批50个固定评测起点，条件搜索结果只能用于机制诊断和形成后续假设。

| 任务 | Cohort ID | Checkpoint SHA256 |
|---|---|---|
| Cube | `cube_legacy_50_v1` | `2748151a2c43c7841246a5fb8ee1fd7915ebbc00bfc27aa1a32b0f9080d92956` |
| PushT | `pusht_legacy_50_v1` | `62d00096a34e9f0b6da4ceb6a3c61eb350701a7aade5f5e4d2c1ee528b06015e` |
| Reacher | `reacher_legacy_50_v1` | `087991339c9499d10c1a58a28e72d07c7dfd819d5f072c67bfd0169acaa83553` |
| TwoRoom | `tworoom_legacy_50_v1` | `48d1ed2520ec7142327969eacf1c37b144faec51cfd6c9ce72701a8421bf57c2` |

## 2. 已完成工作与当前结果

### 2.1 闭环主扫描与稳定性追加

`analysis/index.json` 登记了 **2,600/2,600** 个条件，状态均为 `completed`：seed42主扫描2,424个条件，seed43和seed44各追加88个条件。主扫描分析表 `analysis/analysis.json` 覆盖A1–A7全部 **2,424** 个seed42条件，每个条件50个episode，当前状态为 `ready_for_diagnostic_review`。

下表是每组跨任务、跨配置的等权成功率均值；每个配置有相同的50个episode。它用于描述搜索结果分布，不是独立样本上的统计比较，也不代表最终选型。

| 计划组 | 条件数 | 等权成功率均值 | 单条件成功率范围 |
|---|---:|---:|---:|
| A1 P0/P3 | 144 | 94.2% | 70%–100% |
| A2 P0+PO | 432 | 93.9% | 66%–100% |
| A3 P0+GF | 864 | 93.8% | 66%–100% |
| A4 P3+PO/refine | 432 | 95.9% | 66%–100% |
| A5 P3+GF | 96 | 95.1% | 66%–100% |
| A6 P1/P2 CEM | 240 | 89.9% | 18%–100% |
| A7 guidance-P2 | 216 | 93.9% | 68%–100% |

按任务查看，Cube在A1–A5的组均值均为100%，但A6为91.4%；PushT的A6组均值为88.1%，其他组为94.1%–97.3%；Reacher各组为79.8%–86.3%；TwoRoom各组为98.3%–100%。这些差异尚未经过完整的跨推理seed、延迟前沿和诊断结果联合分析。

历史复用审计记录100个满足完整兼容检查的历史结果候选，没有仅复用成功率的记录；审计说明当前索引仍按 `current` 来源标记，没有把当前产物重新标记成历史复用。

### 2.2 已完成候选池与B排序诊断

PushT、Reacher和TwoRoom候选池的 manifest 均为 `completed`。每个任务覆盖50个起点、6个 flow steps（S=1/2/5/10/16/32）、每个S 256条候选，共 **76,800** 条已执行候选记录；当前 `records.jsonl` 行数均为76,800，manifest记录了跨S候选噪声哈希核验和环境状态及随机状态恢复口径。

Cube已有全部6个 flow steps 的 `candidate_noise_s*.npz`，以及6份proposal文件，每份12,800行（共76,800条已生成proposal）。真实环境分支也已经部分落盘：`branches/s1`、`s2`、`s5`、`s10` 各有256个 `candidate_*.jsonl` 文件，`branches/s16` 有5个，`s32` 暂无文件，共1,029/1,536个候选序号文件；抽查的分支文件各含50个状态行且结果标为 `completed`。剩余按文件数计507个候选序号文件（S16还差251个、S32还差256个）。目前Cube仍没有汇总用的根级 `records.jsonl`、`controls.jsonl` 或候选池 manifest；六个 `proposal_eval_s*` 目录当前为空。因此Cube有完整proposal/noise、有部分已执行分支，但还不是可用于最终分析的完整候选池。

下表的成功率为每个已完成任务在六个S上的等权均值；配对差和selection regret的95%区间来自按50个起点聚类、10,000次bootstrap（每任务300个 state×S 汇总行）。候选池是诊断性固定池，不能代替完整闭环不同N配置。

| 任务 | Oracle成功率 | B选中成功率 | 随机选中成功率 | B−随机成功率差（95% CI） | Selection regret（95% CI） |
|---|---:|---:|---:|---:|---:|
| PushT | 44.0% | 15.3% | 12.7% | +2.7pp（−1.3, +6.8） | 1.304（1.080, 1.540） |
| Reacher | 94.7% | 55.0% | 50.8% | +4.2pp（−3.1, +11.2） | 1.323（1.087, 1.583） |
| TwoRoom | 100.0% | 100.0% | 96.9% | +3.1pp（+1.3, +5.3） | 0.232（0.219, 0.244） |

当前描述性结果显示PushT和Reacher候选池中oracle与B-selected仍有明显差距，B相对随机选择的区间包含零；TwoRoom随机基线已接近成功率上限。它们支持继续检查候选排序和任务难度，不足以决定P3是否保留或推理方法能否替代CEM。

**控制动作协议状态：** PushT manifest记录222条控制动作并带有 `phase1_5_fixed_pool_controls_v1` 标记；Reacher和TwoRoom manifest虽写 `control_status=completed`，但各有240条且缺少协议版本标记，与计划要求的222条控制动作不符，需要按当前协议重跑。Cube控制动作待候选池完成后生成。

### 2.3 Probe与候选池未来latent读出

四个任务的 probe 结果状态均为 `ok`。冻结编码特征上的Ridge、三seed MLP和随机编码器Ridge结果已保存。分割清单中训练、验证和50条评测轨迹ID互不重叠。下表列的是各任务自身目标尺度下的验证MAE；不同任务的目标和单位不同，不能横向比较绝对误差。

| 任务 | Probe特征行数 | Ridge验证 MAE | MLP三seed平均 MAE | 随机编码器Ridge验证 MAE |
|---|---:|---:|---:|---:|
| Cube | 100,000 | 0.00828 | 0.00292 | 0.04210 |
| PushT | 94,582 | 5.28970 | 2.34954 | 26.33911 |
| Reacher | 100,000 | 0.00819 | 0.00623 | 0.30914 |
| TwoRoom | 92,509 | 0.72203 | 0.23626 | 5.53209 |

三个完整候选池已完成 probe attach，各覆盖76,800条未来latent。PushT真实未来latent与B预测未来latent的读出MAE分别为6.982和41.224；Reacher为0.00871和0.24587；TwoRoom为0.653和20.701。该比较把probe应用到B预测latent上，包含latent预测分布偏移等误差来源，不能单独据此断言B没有学到物理。Cube的候选池probe attach待其物理候选池完成。

### 2.4 Guidance variant 文件级覆盖

计划网格为每任务216个post-opt和36个guided-flow变体，共1,008个。按本快照的variant JSONL文件核验，已落盘 **430个** 文件级完整变体；每个文件都有50行、覆盖50个不同的state/slot，JSON均可解析。有效配对21,094条；另有406条标为 `short_termination_unpaired`，这些行保留作终止记录，但不应纳入成对物理效应比较。

| 任务 | Post-opt文件级完整数 | Guided-flow manifest | 文件级完整变体合计 | 有效配对 / 未配对短终止 |
|---|---:|---|---:|---:|
| Cube | 0/216 | 尚无 | 0/252 | 0 / 0 |
| PushT | 164/216 | completed，36/36 | 200/252 | 10,000 / 0 |
| Reacher | 158/216 | completed，36/36 | 194/252 | 9,574 / 126 |
| TwoRoom | 0/216 | completed，36/36 | 36/252 | 1,520 / 280 |
| **合计** | **322/864** | **108/144** | **430/1,008** | **21,094 / 406** |

这里的“文件级完整”指每个variant JSONL通过50个状态行、50个不同state/slot、JSON解析及逐行状态核验。当前文件级完整变体共430个、21,500行；其中21,094行配对有效并标为 `completed`，406行标为 `short_termination_unpaired` 且配对无效。汇总 manifest 尚未完全与文件一致：PushT post-opt manifest 缺失；Reacher post-opt manifest 标记 `partial` 且只登记1/216，而目录中已有158个50行variant文件。最终 guidance 汇总前需要刷新并核对这两个 manifest。未配对短终止是按协议显式保留的结果，不是丢失slot。

## 3. 尚未完成的计划工作

1. **Cube候选池物理执行与收尾：** 补齐剩余507个候选序号分支文件（S16还251个、S32还256个）；增加控制动作，生成根级records和manifest，并运行候选池诊断汇总及Cube候选probe attach。
2. **Guidance sweep：** 尚缺578个变体：Cube 252、PushT 52、Reacher 58、TwoRoom 216。继续任务完成后，核对文件级覆盖、有效配对数和manifest。
3. **修正控制池：** 按计划协议重跑Reacher和TwoRoom各222条控制动作；检查PushT控制协议并补Cube控制集。
4. **稳定性分析：** seed43/44共有176个追加条件已在索引中；当前 `analysis.json` 记录96个稳定性条件，需要把全部计划内固定与自适应追加结果纳入最终分析。
5. **公平计时：** 当前 `analysis/timing.json` 只有一个Reacher P0、S=1粗测条件（5次warm-up、10次同步测量），单环境p50/p95为9.13/9.40 ms，batch50吞吐528.1 states/s；没有完成全配置粗测，也没有记录锚点/Pareto配置的20 warm-up、100测量精测。因此现有计时不能支撑方法间加速结论。
6. **统一诊断与最终研究决策：** 将四任务主扫描、稳定性、候选池、probe、PO/GF真实修正收益和公平延迟合并；按计划给出P3、PO/GF相对CEM、GF独立价值和下一轮训练消融假设，并形成最终报告中的主方法与备选方法（如证据不足则报告尚未收敛）。

## 4. 证据文件与解释限制

- 主扫描与复用：`outputs/round5/phase1_5_seed3072_legacy/analysis/index.json`、`analysis/analysis.json`、`analysis/history_reuse_audit.json`。
- 候选池：`diagnostics/candidate_pool/{pusht,reacher,tworoom}/{manifest.json,records.jsonl,controls.jsonl}`；对应诊断为 `diagnostics/summary_{pusht,reacher,tworoom}.json`。Cube当前无pool manifest和执行records。
- Probe：`diagnostics/probe/{cube,pusht,reacher,tworoom}/result.json` 与 `diagnostics/probe/summary.json`。
- Guidance：`diagnostics/guidance_sweep/<task>/<post_opt|guided_flow>/variants/*.jsonl` 及已有 `manifest.json`。
- 计时：`analysis/timing.json`。

本报告是带快照时间的阶段记录，不替代计划要求的最终报告。所有成功率、排序和配对差异均基于固定 `legacy_50` cohort；在完成全量诊断、种子稳定性和隔离计时前，不应将搜索中观察到的高分视为无偏泛化表现，也不能据此推断联合训练相对独立训练的因果优势或R4-AB的必要性。

## 5. Astra / Sol 评审与续跑补充

**状态快照：2026-09-24 21:32（Asia/Shanghai）。** 本节补充不覆盖上文 20:31 的历史快照。

### 5.1 评审结论

- 已完成的 2,424 个主扫描条件和 176 个稳定性条件不重跑。当前 `analysis/index.json` 有 2,600/2,600 个 completed 条件；`analysis/analysis.json` 仍是旧版，只计入 96 个稳定性条件，需要刷新后再做最终结论。
- 不增加新 cohort、主模型训练或大规模新实验。评审认为现有记录可分别分析评测预算内成功事件和共同物理 milestone；没有证据要求另加时间结果面板。
- 继续当前已启动的 PushT / Reacher 评测。宿主负载约 116，高于计划启动阈值 96，负载下降前不新开 GPU worker。
- 资源恢复后优先完成少量剩余 PushT / Reacher post-opt 配置与隔离计时（固定锚点、领先 Pareto 候选、强 CEM 对照）。若资源持续紧张，可延期 Cube 高 S 分支、TwoRoom 全量 post-opt 和穷举粗计时；若永久删减，必须记录协议变更，并收窄高 S、跨任务 Guidance 和完整延迟前沿相关结论。计划要求的四任务统一方法若证据不收敛，应报告“尚未收敛”。

### 5.2 候选池按 S 分层

现有 `summary_{task}.json` 已包含逐 S 指标及按 50 个起点聚类的 10,000 次 bootstrap 区间。上文六个 S 的等权合并会掩盖候选分布差异：

- **PushT：** S1 / S2 的 oracle 成功率为 90% / 98%，但 B-selected 仅 22% / 16%；S5–S32 的 oracle 仅 18%–20%。六个 S 的 B−随机提升区间均包含 0。S1 / S2 暴露排序问题，S5 以上还同时暴露提案池覆盖不足。
- **Reacher：** S2–S32 的 oracle 均为 100%，B-selected 为 52%–64%；六个 S 的 B−随机区间均包含 0。候选池存在大量可成功动作，但当前 B 排序仍未稳定找出它们。
- **TwoRoom：** 所有 S 的 B-selected 均为 100%，随机基线为 95.3%–100%。S2–S32 的小幅 B−随机差区间高于 0，但成功率天花板限制其作为 B 排序强证据的价值。

因此，候选池结果支持按任务和 S 分层诊断，不足以单独证明 P3 应保留，也不足以替代闭环成功率与公平延迟比较。

### 5.3 Reacher / TwoRoom 控制池兼容迁移

Reacher、TwoRoom 原有控制记录各为 240×50。独立兼容审计确认当前 222 个控制动作对应的 11,100 条状态×动作记录都与重生成动作逐元素 float64 完全相等，最大差为 0；仅 18 个旧索引属于额外 shift 动作并被移除。9 个 block-transform 的旧 metadata 去除 `shift:0` 后与当前协议一致。迁移保留原始 outcome、state、episode、row index，并把原 240 条文件以硬链接保存在原目录；没有重跑环境。

| 任务 | 当前控制记录 | 原始备份 | 审计 |
|---|---|---|---|
| Reacher | `diagnostics/candidate_pool/reacher/controls.jsonl`：222×50，协议 `phase1_5_fixed_pool_controls_v1` | `controls_legacy240_1df1fb5e6d0f.jsonl` | `diagnostics/candidate_pool/reacher/control_compatibility_audit.json` |
| TwoRoom | `diagnostics/candidate_pool/tworoom/controls.jsonl`：222×50，协议 `phase1_5_fixed_pool_controls_v1` | `controls_legacy240_fbc92b2ca8e0.jsonl` | `diagnostics/candidate_pool/tworoom/control_compatibility_audit.json` |

这解决了上文 20:31 快照中的“需要重跑”问题；剩余 Cube 控制动作仍待其候选池完成后生成。

Reacher / TwoRoom 的候选池 manifest 已更新，但各任务 `summary_*.json` 中的控制统计仍需在队列结束后从新 222 动作文件重新汇总；上文 20:31 诊断表仍是迁移前快照。

### 5.4 21:32 执行进度

| 项目 | 文件级状态 |
|---|---:|
| 主扫描与稳定性索引 | 2,600/2,600 completed |
| Cube 候选分支 S1/S2/S5/S10 | 各 256/256 |
| Cube 候选分支 S16 / S32 | 43/256 / 0/256 |
| PushT post-opt / guided-flow | 194/216 / 36/36 |
| Reacher post-opt / guided-flow | 196/216 / 36/36 |
| TwoRoom post-opt / guided-flow | 0/216 / 36/36 |
| Cube post-opt / guided-flow | 0/216 / 0/36 |

全网格 Guidance 为 1,008 个变体，此快照文件级已完成 498 个；Guidance 汇总 manifest 和有效配对数仍待任务队列收尾后刷新。计时目前仍只有一条粗测记录，尚无隔离精测，因此不作方法间加速结论。

Guidance 的 `episode_successes` 表示 `eval_budget=50` 内出现成功事件，不等同于统一第 25 个 primitive step 的终态成功率。物理距离差按每个起点最后一个共同有效 milestone（5/10/15/20/25）配对；最终报告需将事件成功率、物理距离和共同 milestone 步数分布分开呈现。

### 5.5 22:41 续跑检查点

本节记录 **2026-09-24 22:41（Asia/Shanghai）** 的续跑状态；上文各表仍保留其原快照，不追写覆盖。

- `scripts/round5_phase1_5.py validate` 返回 `status=ok`；主扫描网格为 2,424 条件，固定稳定性网格为 96 条件。主索引仍为 2,600/2,600 completed（含 80 个自适应稳定性条件），但 `analysis/analysis.json` 尚未刷新，仍只计入96个稳定性条件。
- PushT、Reacher post-opt 已完成，各216/216；两任务 guided-flow 也各36/36。TwoRoom post-opt 仍为0/216、guided-flow为36/36；Cube post-opt与guided-flow仍为0/216、0/36。全网格文件级已完成540/1,008，尚缺468个变体。
- Cube 候选池 S1/S2/S5/S10 各256/256，S16为100/256，S32为0/256，总计1,124/1,536个分支文件，剩余412个（S16剩156、S32剩256）。Cube候选执行记录、控制池、根级manifest及probe attach尚未完成；GPU3日志与S16最新分支文件在本次检查时仍有更新。
- 其余三个任务的固定候选池及控制池已存在；Reacher和TwoRoom控制动作已迁移到当前222动作协议，原240动作文件保留为硬链接备份。迁移后的控制统计、Guidance manifest和四任务诊断汇总仍需最终刷新。
- 宿主机可用内存约278.9 GiB，128核负载比约0.44，低于0.75上限；但 SwapFree 仅96 KiB，低于1,024 MiB启动门槛。按只检查GPU0–3的查询，GPU0/1/2/3可用显存约43.9/40.0/36.5/47.7 GiB，其中GPU0和GPU2当时负载较高。故保留正在推进的Cube候选任务，暂不启动新的队列。

下一次阶段分析需等待Cube候选池与剩余Guidance任务结束；随后刷新四任务的controls、candidate-pool和paired-guidance汇总，补Cube probe attach，并更新全量稳定性分析、计时和方法选择。当前方法间延迟证据仍只有一条粗测，不能支撑加速结论。

### 5.6 PushT / Reacher 摘要刷新（22:47）

对已结束的 PushT、Reacher 队列分别重算诊断摘要，使用当前候选池记录、222动作控制记录、post-opt 和 guided-flow 聚合记录；两条命令均以默认10,000次状态聚类 bootstrap 完成，退出码为0。`summary_pusht.json` 与 `summary_reacher.json` 已包含全部252个已完成变体、控制统计和 `guidance_milestone_steps`。这更新了5.5检查点所述的PushT / Reacher控制统计待刷新状态；TwoRoom、Cube及最终四任务统一分析仍待收尾。

22:48的队列检查显示 Cube S16 已推进到106/256，S32仍为0/256；候选分支尚缺406个（S16剩150、S32剩256）。Guidance仍为540/1,008个变体完成，SwapFree为64 KiB，故继续不启动新队列。

| 任务 / 模式 | 有效配对行 | 未配对行 | 最后共同 milestone 分布（step: 行数） |
|---|---:|---:|---|
| PushT post-opt | 10,800 | 0 | 10:193，15:558，20:8,030，25:2,019 |
| PushT guided-flow | 1,800 | 0 | 10:28，15:76，20:1,370，25:326 |
| Reacher post-opt | 10,665 | 135 | 5:141，10:481，15:1,087，20:3,496，25:5,460 |
| Reacher guided-flow | 1,782 | 18 | 5:29，10:94，15:176，20:621，25:862 |

以上计数是变体×起点行数，不是独立样本数；Reacher 的未配对短终止行仍保留在原始记录中，但不进入paired指标。两个任务当前控制摘要各覆盖11,100条记录（222动作×50起点）。不同 milestone 的行数不同，最终物理距离比较必须按共同 step 分层解释。

### 5.7 同 milestone 的 GF−PO 物理差异（22:56）

摘要新增 `matched_guidance_by_milestone`：先按相同 `physical_comparison_step` 过滤，再匹配相同 flow steps、候选索引、K、η、R 和起点物理代价；每个起点先平均匹配配置，再做10,000次状态聚类 bootstrap。正值表示 GF 的真实物理改善更多。当前 PushT、Reacher 的结果为：

| 任务 | step | 匹配配置数 / 起点数 | GF−PO 改善差 [95% CI] |
|---|---:|---:|---:|
| PushT | 10 | 28 / 1 | +0.0012（起点数不足，未给区间） |
| PushT | 15 | 33 / 7 | −0.0377 [−0.0890, +0.0075] |
| PushT | 20 | 36 / 44 | −0.0142 [−0.0350, +0.0045] |
| PushT | 25 | 36 / 19 | −0.0453 [−0.2247, +0.0791] |
| Reacher | 5 | 27 / 2 | +0.0379 [−0.0997, +0.1755] |
| Reacher | 10 | 31 / 7 | −0.1925 [−0.2955, −0.0978] |
| Reacher | 15 | 35 / 19 | −0.2519 [−0.5204, −0.0267] |
| Reacher | 20 | 36 / 34 | −0.0700 [−0.1567, +0.0125] |
| Reacher | 25 | 36 / 45 | +0.0573 [+0.0132, +0.1066] |

PushT 在有足够起点的步数上没有清晰 GF−PO 差异。Reacher 不同步数子集的描述性差值符号不同：step10/15为负、step25为正，step5/20区间跨0；这不能解释为同一批起点上的时间趋势，也不支持跨步数的统一 GF 优势。`physical_comparison_step` 由 baseline、guided 和 random 三条分支最后共同有效的 milestone 决定，受终止与结果可用性影响；按它分层属于结果后的选择，各步所含的起点和有效配置可能不同。36个匹配配置是在每个起点内先平均，并非36个独立样本；表中的1–45个起点才是 bootstrap 单位，Reacher step5/10尤其只有2/7个起点。跨步数和配置的检视未做多重比较校正，因此 step25 的正区间与 step10/15 的负区间一样，都只是探索性诊断线索，不是全50起点的因果效应或确认性结论。原先合并不同 milestone 的 GF−PO 数值也只作探索性汇总，不作为同一 rollout 长度的直接比较。

### 5.8 23:00 续跑状态

Cube候选分支 S1/S2/S5/S10仍各256/256，S16为117/256，S32为0/256；剩余395个分支（S16剩139、S32剩256）。Guidance文件级仍为540/1,008。宿主可用内存约260.6 GiB，负载比约0.45，但SwapFree仅232 KiB；GPU0–2当时负载较高，GPU3可用显存约47.5 GiB。保留当前Cube任务，不启动新的GPU队列。

### 5.9 23:18 续跑状态

Cube候选分支 S1/S2/S5/S10仍各256/256，S16为133/256，S32为0/256；剩余379个分支（S16剩123、S32剩256）。Guidance文件级仍为540/1,008。宿主可用内存约236.4 GiB，但SwapFree仅72 KiB；GPU3可用显存约42.4 GiB、利用率约1%，GPU1利用率约95%。Swap启动门槛未通过，继续保留当前任务并等待，不启动新的GPU队列。

### 5.10 全量索引分析刷新（23:22）

基于现有完整索引重新生成 `analysis/analysis.json`、`analysis/conditions.csv` 和 `analysis/history_reuse_audit.json`，命令退出码为0：2,424个主条件、96个固定稳定性条件、80个自适应稳定性条件，共176个稳定性条件；索引2,600/2,600均为completed。报告草稿输出到 `/tmp/round5_phase1_5_20260924_partial.md`，没有覆盖仓库主报告。该草稿的诊断摘要仍不完整，不能作为最终报告；TwoRoom和Cube摘要、Cube候选池、剩余Guidance及公平计时完成后还需重新统一生成。

### 5.11 23:35 续跑状态

Cube候选分支 S1/S2/S5/S10仍各256/256，S16为150/256，S32为0/256；还剩362个分支（S16剩106、S32剩256）。Guidance仍为540/1,008个变体完成。宿主可用内存约184 GiB，但SwapFree仅4 KiB；GPU1利用率约100%，GPU3当时空闲。继续不启动新队列，保留正在写出结果的Cube候选任务。

宿主进程核验确认该Cube worker为PID 3459660，已运行约21小时，进程环境设置 `CUDA_VISIBLE_DEVICES=3`。它现有启动命令包含 `--min-swap-free-mib 0`，即启动时绕过默认Swap余量门槛；后续新任务仍必须通过默认1,024 MiB检查，不沿用该覆盖值。

### 5.12 23:54 Cube worker 状态核验

Cube 候选池 S1/S2/S5/S10 各256/256，S16为168/256，S32为0/256，共1,192/1,536个分支文件，尚缺344个（S16剩88、S32剩256）。S16最后一个文件 `candidate_0167.jsonl` 的修改时间为23:53:39，Cube GPU3日志最后更新时间为23:53:40；23:54核验时未发现对应 Python worker 仍在运行。日志末尾只有 Gymnasium 初始化与 dtype 警告，没有 Python traceback。

23:51宿主机可用内存约113 GiB，但 SwapFree仅208 KiB；GPU0–3可用显存约43.9/35.5/36.5/42.2 GiB，利用率约0/92/38/4%。23:54复查时 MemAvailable约93.5 GiB、SwapFree仅176 KiB，仍远低于1,024 MiB启动门槛。23:57的GPU复查显示GPU0–3可用显存约43.9/35.5/36.5/42.3 GiB，利用率约0/35/54/4%；23:59 SwapFree为224 KiB，仍不启动新GPU任务。

23:54本地进程表未显示Python worker，但这不能证明任务已终止。后续分支文件仍持续写出：`candidate_0169.jsonl` 至 `candidate_0172.jsonl` 分别更新至23:55:42、23:56:44、23:57:46、23:58:49，日志最后更新时间为23:58:52；23:59时S16为173/256，六个S合计1,197/1,536，剩余339（S16剩83、S32剩256）。因此按输出时间戳将该队列视为仍在推进，不重启、不重复计算。

日志末尾未见Python traceback；沙箱及提权查询均不能读取内核 dmesg，当前用户可见的 journal 也没有对应记录。现有证据不足以判定为 OOM；进程列表不可见不等同于任务终止。后续保留并监测已有输出；只有在启动新队列前才要求SwapFree达到1,024 MiB，并按默认资源检查启动条件。

### 5.13 TwoRoom 部分诊断摘要刷新（00:02）

用当前摘要代码重算 TwoRoom，覆盖完整候选池、迁移后的222动作控制池和已完成的 guided-flow 记录；10,000次 state-cluster bootstrap 完成，退出码为0。更新后的 `summary_tworoom.json` 包含候选池、controls、milestone 分布、配对 guided/random 指标与按物理步数分层的比较字段。该任务当前只有GF记录，因此GF−PO各 milestone 均为 `insufficient_matched_data`，不能得出GF相对PO的结论；未完成的post-opt结果尚未纳入。

- 候选池：76,800行（6个S × 256候选 × 50起点）；B-selected 成功率各S均为100%，随机成功率为95.3%–100%。跨S汇总的 B-selected−随机差为+3.06个百分点，95% CI [+1.30,+5.27]个百分点；平均 selection regret 为0.232（95% CI [0.219,0.244]）。受成功率天花板与固定50起点限制，该差异作为诊断结果，不单独证明排序方法具有普遍优势。
- 控制池：11,100行（222动作 × 50起点）。Guided-flow：1,800条源记录，其中1,520条有效配对记录、45个起点；其余280条因结果不可配对而排除。GF相对等RMS随机方向的真实物理改善差为+0.0415，95% CI [+0.0155,+0.0675]；仅代表当前GF配置与随机方向的比较，不替代GF−PO。

这次刷新修复了旧TwoRoom摘要缺少controls与guidance统计的问题，但最终四任务分析仍需等待TwoRoom post-opt及Cube候选池、控制池与Guidance实验完成后统一生成。

### 5.14 00:13 全量分析与 probe 报告修正

TwoRoom摘要刷新后再次生成全量索引分析与临时报告草稿，命令退出码为0：主条件2,424、固定稳定性96、自适应稳定性80，共2,600条，均为completed；`decision_status=ready_for_diagnostic_review`。诊断汇总当前覆盖PushT、Reacher、TwoRoom，Cube尚缺候选池摘要。草稿写入 `/tmp/round5_phase1_5_20260925_partial.md`，没有覆盖主报告。

检查草稿发现 probe 的 Ridge 验证指标被渲染成“—”：持久化结构将 MAE 与验证行数存于 `validation_metrics` 子对象，而渲染器此前读取顶层字段。已修正读取逻辑，兼容新旧字段布局；四任务当前均可显示验证行数、Ridge、随机编码器 Ridge、三 seed MLP 均值。Cube 的真实/预测 future-latent 误差仍空缺，因为 Cube 候选物理记录尚未完成。

针对嵌套 probe 结构新增的渲染回归测试通过；完整 `tests.test_round5_phase1_5` 模块的60项测试全部通过。修改文件的 `py_compile` 与 `git diff --check` 通过。临时报告已用最终映射代码重新生成并核验，显示 Cube 20,000、PushT 19,054、Reacher 20,000、TwoRoom 18,651 个验证样本及对应指标。

00:13续跑状态：Cube S1/S2/S5/S10仍各256/256，S16为186/256，S32为0/256，共1,210/1,536，剩余326（S16剩70、S32剩256）。Guidance仍为540/1,008：PushT、Reacher的post-opt和GF均完成，TwoRoom仅GF完成，Cube两组均未启动。GPU0–3可用显存约43.9/43.9/36.5/47.7 GiB；SwapFree仅108 KiB，不启动新队列。现有Cube输出继续推进。

### 5.15 00:18 续跑检查

Cube候选池S1/S2/S5/S10各256/256，S16为190/256，S32为0/256，共1,214/1,536，剩余322（S16剩66、S32剩256）；Cube日志和S16分支文件仍有新写入。Guidance仍为540/1,008，缺TwoRoom post-opt 216条与Cube post-opt/GF 252条。宿主 MemAvailable约57.2 GiB、SwapFree约1.74 MiB，仍远低于1,024 MiB门槛；GPU0–3可用显存约43.9/43.9/36.5/42.4 GiB，利用率约100/0/72/1%。保留现有Cube队列，不启动新GPU工作。

### 5.16 冻结输入与主扫描覆盖复核（00:21）

重新对照Phase1.5配置和当前磁盘文件：Cube、PushT、Reacher、TwoRoom四个R4-AB seed3072 epoch10 checkpoint的实际SHA256均与配置一致；四份cohort的实际SHA256也一致，且每份均有50条legacy评测起点。主扫描索引含2,600条completed条件，逐条核验均有50个episode的验证记录；无缺样本或静默丢弃条件。

Probe产物也通过逐轨迹审计：四任务结果均为`ok`，train/validation/evaluation轨迹分别为800/200/50且两两不相交，排除的evaluation轨迹与cohort一致；MLP使用seed0/1/2。实际probe帧数为Cube/Reacher各100,000、PushT 94,582、TwoRoom 92,509，后两者低于上限是因为可用帧较少，未重复采样补齐。

计时产物核验：`timing.json`目前只有Reacher P0/S=1/N=1的一条粗测（5次warm-up、10次同步测量），batch1 p50为9.13 ms，batch50 p50为95.02 ms；不含环境运行时间。精测字段为null，`method_selection.json`尚不存在，所以当前没有可用的跨方法公平延迟比较或主推/备选选择。旧 smoke 日志中的启动门槛包含 `--min-swap-free-mib 0`；这只是已有记录，后续新任务不复用该覆盖。

### 5.17 00:24 续跑检查

Cube候选池S1/S2/S5/S10各256/256，S16为196/256，S32为0/256，共1,220/1,536，剩余316（S16剩60、S32剩256）；23:59后分支文件和日志均继续更新。Guidance仍为540/1,008。宿主MemAvailable约46.9 GiB，但SwapFree仅136 KiB；GPU0–3可用显存约43.9/43.9/36.5/47.7 GiB，利用率约0/0/57/0%。继续保留现有Cube队列，不启动新的GPU工作。

### 5.18 00:40 P3 覆盖口径修正与续跑检查

对临时分析稿做覆盖核对时发现：Cube候选池摘要尚不存在，报告表实际只有PushT、Reacher、TwoRoom三项，但旧决策句把三项上的均值写作“四任务平均”，并将显著任务数显示成“1/4”。GPT-6 Sol high只读评审确认这是覆盖口径误导；可用结果中的数值本身没有算错。

已修正报告决策逻辑：分开标记 B−随机增益区间与 selection-regret 的任务覆盖；覆盖不足四任务时写明已完成任务数和待补任务，不给出跨任务 P3 结论，也不提前确定下一轮 R4-AB 消融建议。对 guidance 结果同样要求有完整任务覆盖后才输出否定判断。回归测试覆盖Cube摘要缺失、增益与regret覆盖数不一致、guidance仅部分任务有记录；针对性测试3项通过，完整 `tests.test_round5_phase1_5` 模块61项通过，`py_compile` 和 `git diff --check` 通过。

修正后重生成的临时稿 `/tmp/round5_phase1_5_20260925_partial.md` 显示：候选池增益区间覆盖3/4个任务，三任务平均差为+0.033，其中1/3已完成任务的95%区间下界大于0；regret覆盖3/4，已完成任务平均为0.953。Cube仍待补，故不作四任务结论，R4-AB消融建议继续待诊断完成。全量索引分析仍为2,600/2,600 completed；本次只更新临时稿，没有覆盖主报告。

00:40续跑快照：Cube候选池S1/S2/S5/S10各256/256，S16为211/256、S32为0/256，共1,235/1,536个分支，剩301个（S16剩45、S32剩256）；S16分支文件和GPU3日志仍有更新。Guidance仍为540/1,008，欠TwoRoom post-opt 216条及Cube post-opt/GF 252条。宿主MemAvailable约316 GiB，SwapFree约11 MiB，仍低于新任务所需的1,024 MiB；GPU0–3可用显存约43.9/43.9/36.5/44.9 GiB，GPU3空闲。继续保留当前Cube队列，不启动新GPU任务。计时仍仅有一条粗测，细测选择与方法选择文件均未生成。

### 5.19 00:46 续跑检查

Cube候选池S1/S2/S5/S10各256/256，S16为217/256、S32为0/256，共1,241/1,536个分支，剩295个（S16剩39、S32剩256）；S16分支文件及GPU3日志仍更新，保留原队列不重启。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约297 GiB、SwapFree约9.8 MiB；GPU0–3可用显存约7.4/18.6/35.7/22.2 GiB，利用率约100/100/53/100%。Swap门槛未满足，不启动新GPU任务。

### 5.20 00:51 续跑检查

Cube候选池S1/S2/S5/S10各256/256，S16为221/256、S32为0/256，共1,245/1,536个分支，剩291个（S16剩35、S32剩256）；`candidate_0220.jsonl` 与GPU3日志在本次检查前仍有更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约292.6 GiB、SwapFree约35.7 MiB；GPU0–3可用显存约6.9/18.6/35.7/16.5 GiB，利用率约100/100/46/100%。Swap门槛未满足，保留当前Cube队列，不启动新GPU任务。

### 5.21 00:56 续跑检查

Cube候选池S1/S2/S5/S10各256/256，S16为227/256、S32为0/256，共1,251/1,536个分支，剩285个（S16剩29、S32剩256）；`candidate_0226.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约290 GiB、SwapFree约5.3 MiB；GPU0–3可用显存约2.6/18.6/35.7/22.2 GiB，GPU0/1/3利用率为100%。继续保留当前队列，不启动新GPU任务。

### 5.22 01:02 续跑检查

Cube候选池S1/S2/S5/S10各256/256，S16为231/256、S32为0/256，共1,255/1,536个分支，剩281个（S16剩25、S32剩256）；`candidate_0230.jsonl` 与GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约290.6 GiB、SwapFree约2.3 MiB；GPU0–3可用显存约0.6/18.6/35.7/16.5 GiB，GPU0/1/3利用率为100%。Swap门槛未满足，不启动新GPU任务。

### 5.23 01:07 续跑检查

Cube候选池S1/S2/S5/S10各256/256，S16为236/256、S32为0/256，共1,260/1,536个分支，剩276个（S16剩20、S32剩256）；`candidate_0235.jsonl` 和GPU3日志仍更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约294.2 GiB、SwapFree约2.5 MiB；GPU0–3可用显存约7.9/18.6/35.7/16.7 GiB，GPU0/1/3利用率为100%。保留当前Cube队列，不启动新GPU任务。

### 5.24 01:13 续跑检查

Cube候选池S1/S2/S5/S10各256/256，S16为242/256、S32为0/256，共1,266/1,536个分支，剩270个（S16剩14、S32剩256）；`candidate_0241.jsonl` 和GPU3日志在本次检查前仍有更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约290.1 GiB、SwapFree约1.3 MiB；GPU0–3可用显存约6.4/18.6/35.7/19.9 GiB，GPU0/1/3利用率为100%。Swap门槛未满足，保留现有Cube队列，不启动新任务。

### 5.25 01:19 续跑检查

Cube候选池S1/S2/S5/S10各256/256，S16为248/256、S32为0/256，共1,272/1,536个分支，剩264个（S16剩8、S32剩256）；`candidate_0247.jsonl` 与GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约292.3 GiB、SwapFree约120 KiB；GPU0–3可用显存约4.1/18.6/35.7/22.2 GiB，GPU0/1/3利用率为100%。继续保留现有队列，不启动新GPU任务。

### 5.26 01:25 续跑检查

Cube候选池S1/S2/S5/S10各256/256，S16为254/256、S32为0/256，共1,278/1,536个分支，剩258个（S16剩2、S32剩256）；`candidate_0253.jsonl` 和GPU3日志仍更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约290.1 GiB、SwapFree约480 KiB；GPU0–3可用显存约3.4/18.6/35.7/22.2 GiB，GPU0/1/3利用率为100%。不启动新GPU任务。

### 5.27 01:31 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为3/256，共1,283/1,536个分支，剩253个；上一检查后的8条S16分支已收尾，原worker已开始S32并继续写出 `candidate_0002.jsonl` 与GPU3日志。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约289.1 GiB、SwapFree约64 KiB；GPU0–3可用显存约4.6/18.6/35.7/17.4 GiB，GPU0/1/3利用率为100%。不启动新GPU任务。

### 5.28 01:37 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为9/256，共1,289/1,536个分支，剩247个；`candidate_0008.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约297.1 GiB、SwapFree约180 KiB；GPU0–3可用显存约12.7/18.6/35.7/16.5 GiB，GPU0/1/3利用率为100%。继续保留当前队列，不启动新GPU任务。

### 5.29 01:43 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为16/256，共1,296/1,536个分支，剩240个；`candidate_0015.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约298.0 GiB、SwapFree约104 KiB；GPU0–3可用显存约12.7/18.6/35.7/16.5 GiB，GPU0/1/3利用率为100%。继续保留当前队列，不启动新GPU任务。

### 5.30 01:49 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为22/256，共1,302/1,536个分支，剩234个；`candidate_0021.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约295.1 GiB、SwapFree约104 KiB；GPU0–3可用显存约12.7/18.6/35.7/16.5 GiB，GPU0/1/3利用率为100%。不启动新GPU任务。

### 5.31 01:55 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为28/256，共1,308/1,536个分支，剩228个；`candidate_0027.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约292.1 GiB、SwapFree约376 KiB；GPU0–3可用显存约12.7/18.6/35.7/16.5 GiB，GPU0/1/3利用率为100%。继续保留现有队列，不启动新GPU任务。

### 5.32 02:01 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为34/256，共1,314/1,536个分支，剩222个；`candidate_0033.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约297.5 GiB、SwapFree约900 KiB；GPU0–3可用显存约12.7/18.6/35.7/22.2 GiB，GPU0/1/3利用率为100%。继续保留现有队列，不启动新GPU任务。

### 5.33 02:07 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为40/256，共1,320/1,536个分支，剩216个；`candidate_0039.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约293.8 GiB、SwapFree约1.2 MiB，仍远低于1,024 MiB门槛；GPU0–3可用显存约12.7/18.6/35.7/22.2 GiB，GPU0/1/3利用率为100%。不启动新GPU任务。

### 5.34 02:13 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为46/256，共1,326/1,536个分支，剩210个；`candidate_0045.jsonl` 与GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约294.6 GiB、SwapFree约1.7 MiB；GPU0–3可用显存约12.7/18.6/35.7/16.5 GiB，GPU0/1/3利用率为100%。新任务门槛未通过，保留当前Cube队列。

### 5.35 02:19 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为52/256，共1,332/1,536个分支，剩204个；`candidate_0051.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约295.8 GiB、SwapFree约1.9 MiB；GPU0–3可用显存约12.7/18.6/35.7/22.2 GiB，GPU0/1/3利用率为100%。保留当前队列，不启动新GPU任务。

### 5.36 02:25 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为58/256，共1,338/1,536个分支，剩198个；`candidate_0057.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约293.4 GiB、SwapFree约2.4 MiB；GPU0–3可用显存约12.7/18.6/35.7/22.2 GiB，GPU0/1/3利用率为100%。继续保留现有队列，不启动新GPU任务。

### 5.37 02:31 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为63/256，共1,343/1,536个分支，剩193个；`candidate_0062.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约291.5 GiB、SwapFree约208 KiB；GPU0–3可用显存约12.7/18.6/35.7/17.2 GiB，GPU2利用率约1%，但Swap门槛仍未达到。保留当前队列，不启动新GPU任务。

### 5.38 02:38 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为69/256，共1,349/1,536个分支，剩187个；`candidate_0068.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约269.6 GiB、SwapFree约76 KiB；GPU0–3可用显存约12.7/18.6/35.7/22.2 GiB，GPU0/1/3利用率为100%。保留当前队列，不启动新GPU任务。

### 5.39 02:44 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为75/256，共1,355/1,536个分支，剩181个；`candidate_0074.jsonl` 与GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约270.6 GiB、SwapFree约112 KiB；GPU0–3可用显存约12.7/18.6/35.7/22.2 GiB，GPU2利用率约24%，但Swap门槛仍未满足。保留当前队列，不启动新GPU任务。

### 5.40 02:51 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为81/256，共1,361/1,536个分支，剩175个；`candidate_0080.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约269.2 GiB、SwapFree约16 KiB；GPU0–3可用显存约12.7/18.6/35.7/22.2 GiB，GPU0–3均有负载。继续保留现有队列，不启动新GPU任务。

### 5.41 02:57 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为86/256，共1,366/1,536个分支，剩170个；`candidate_0085.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约276.7 GiB、SwapFree约640 KiB；GPU0–3可用显存约17.1/18.6/35.7/20.7 GiB，GPU0/1/3利用率为100%。保留现有队列，不启动新GPU任务。

### 5.42 03:03 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为92/256，共1,372/1,536个分支，剩164个；`candidate_0091.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约277.3 GiB、SwapFree约32.3 MiB，虽较上次回升仍远低于1,024 MiB；GPU0–3可用显存约17.1/18.6/35.7/22.2 GiB，GPU0/1/3利用率为100%。继续保留现有队列，不启动新GPU任务。

### 5.43 03:10 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为99/256，共1,379/1,536个分支，剩157个；`candidate_0098.jsonl` 与GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约284.8 GiB、SwapFree约32.8 MiB，仍低于1,024 MiB；GPU0–3可用显存约17.1/18.6/35.7/22.2 GiB，GPU0/1/3利用率为100%。保留当前队列，不启动新GPU任务。

### 5.44 03:17 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为105/256，共1,385/1,536个分支，剩151个；`candidate_0104.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约275.1 GiB、SwapFree约33.9 MiB，仍远低于1,024 MiB；GPU0–3可用显存约17.1/18.6/35.7/17.9 GiB，GPU0/1/3利用率为100%。保留现有队列，不启动新GPU任务。

### 5.45 03:24 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为112/256，共1,392/1,536个分支，剩144个；`candidate_0111.jsonl` 与GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约276.6 GiB、SwapFree约42.6 MiB，仍远低于1,024 MiB；GPU0–3可用显存约17.1/18.6/35.7/19.7 GiB，GPU0/1/3利用率为100%。保留现有队列，不启动新GPU任务。

### 5.46 03:30 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为118/256，共1,398/1,536个分支，剩138个；`candidate_0117.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约278.8 GiB、SwapFree约27.3 MiB；GPU0–3可用显存约17.1/18.6/35.7/22.2 GiB，GPU0/1/3利用率为100%。继续保留现有队列，不启动新GPU任务。

### 5.47 03:36 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为123/256，共1,403/1,536个分支，剩133个；`candidate_0122.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约272.5 GiB、SwapFree约32.2 MiB；GPU0–3可用显存约17.1/18.6/35.7/16.8 GiB，GPU0/1/3利用率为100%。继续保留现有队列，不启动新GPU任务。

### 5.48 03:44 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为130/256，共1,410/1,536个分支，剩126个；`candidate_0129.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约274.3 GiB、SwapFree约32.9 MiB；GPU0–3可用显存约17.1/18.6/35.7/22.2 GiB，GPU0/1/3利用率为100%。保留现有队列，不启动新GPU任务。

### 5.49 03:50 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为136/256，共1,416/1,536个分支，剩120个；`candidate_0135.jsonl` 与GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约276.2 GiB、SwapFree约32.4 MiB；GPU0–3可用显存约17.1/18.6/35.7/22.2 GiB，GPU2利用率约19%。保留当前队列，不启动新GPU任务。

### 5.50 03:57 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为142/256，共1,422/1,536个分支，剩114个；`candidate_0141.jsonl` 与GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约273.0 GiB、SwapFree约1,076 KiB（约1.05 MiB），仍远低于1,024 MiB；GPU0–3可用显存约17.1/18.6/35.7/17.9 GiB，GPU0/1/3利用率为100%。保留现有队列，不启动新GPU任务。

### 5.51 04:05 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为150/256，共1,430/1,536个分支，剩106个；`candidate_0149.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约276.3 GiB、SwapFree约3.7 MiB；GPU0–3可用显存约17.1/18.6/35.7/21.3 GiB，GPU0/1/3利用率为100%。继续保留现有队列，不启动新GPU任务。

### 5.52 04:12 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为157/256，共1,437/1,536个分支，剩99个；`candidate_0156.jsonl` 与GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约274.7 GiB、SwapFree约60.4 MiB，仍远低于1,024 MiB；GPU0–3可用显存约17.1/18.6/35.7/22.2 GiB，GPU0/1/3利用率为100%。继续保留现有队列，不启动新GPU任务。

### 5.53 04:19 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为163/256，共1,443/1,536个分支，剩93个；`candidate_0162.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约276.9 GiB、SwapFree约63.1 MiB，仍远低于1,024 MiB；GPU0–3可用显存约17.1/18.6/35.7/21.5 GiB，GPU0/1/3利用率为100%。继续保留现有队列，不启动新GPU任务。

### 5.54 04:25 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为169/256，共1,449/1,536个分支，剩87个；`candidate_0168.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约276.9 GiB、SwapFree约63.6 MiB；GPU0–3可用显存约17.1/18.6/35.7/22.2 GiB，GPU0–3均有负载。保留当前队列，不启动新GPU任务。

### 5.55 04:33 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为175/256，共1,455/1,536个分支，剩81个；`candidate_0174.jsonl` 和GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约276.6 GiB、SwapFree约63.8 MiB；GPU0–3可用显存约17.1/18.6/35.7/18.0 GiB，GPU0/1/3利用率为100%。继续保留当前队列，不启动新GPU任务。

### 5.56 04:40 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为182/256，共1,462/1,536个分支，剩74个；`candidate_0181.jsonl` 与GPU3日志继续更新。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约276.1 GiB、SwapFree约64.3 MiB；GPU0–3可用显存约17.1/18.6/35.7/22.2 GiB，GPU0/1/3利用率为100%。GPU3日志末尾为Gymnasium告警，全文件搜索未发现Traceback、OOM、Exception或Killed标记；继续保留现有队列，不启动新任务。

### 5.57 04:47 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为189/256，共1,469/1,536个分支，剩67个；`candidate_0188.jsonl` 与GPU3日志继续更新。GPU3日志尾部仍是Gymnasium告警，没有新的错误标记。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约276.8 GiB、SwapFree约64.8 MiB；GPU0–3可用显存约17.1/18.6/35.7/22.2 GiB，GPU0/1/3利用率为100%。继续保留当前队列，不启动新GPU任务。

### 5.58 04:55 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为196/256，共1,476/1,536个分支，剩60个；`candidate_0195.jsonl` 与GPU3日志继续更新，日志尾部仍为Gymnasium告警。Guidance仍为540/1,008，计时仍为1条粗测、0条细测。宿主MemAvailable约273.6 GiB、SwapFree约64.9 MiB；GPU0–3可用显存约17.1/18.6/35.7/22.2 GiB，GPU0/1/3利用率为100%。继续保留现有队列，不启动新GPU任务。

### 5.59 05:00 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为201/256，共1,481/1,536个分支，剩55个；最新`candidate_0200.jsonl`包含50条有效记录且均为completed，GPU3日志继续更新，尾部为Gymnasium告警，未检出Traceback、OOM、Exception或Killed。Guidance仍为540/1,008，其中PushT与Reacher各完成post-opt 216/216及GF 36/36，TwoRoom仅GF 36/36，Cube尚未开始。计时仍为1/2,600条粗测条件、0条细测，`fine_timing_selection.json`与`method_selection.json`均未生成。宿主MemAvailable约272.4 GiB、SwapFree约65.1 MiB；GPU0–3可用显存约17.1/18.6/35.7/22.2 GiB，GPU0/1/3利用率为100%、GPU2约33%。SwapFree仍低于1,024 MiB启动门槛，保留当前队列，不启动新GPU任务。

### 5.60 05:08 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为208/256，共1,488/1,536个分支，剩48个；GPU3日志时间戳更新至05:08，末尾仍是Gymnasium告警，未检出Traceback、OOM、Exception或Killed。Guidance仍为540/1,008，其中PushT、Reacher的post-opt各216/216、GF各36/36，TwoRoom GF为36/36，TwoRoom post-opt及Cube两类变体仍未开始。粗计时仍为1/2,600条、细计时0条，精测选择和最终方法选择文件仍未生成。宿主MemAvailable约281.6 GiB、SwapFree约65.3 MiB；GPU0–3可用显存约17.1/18.6/35.7/22.2 GiB，GPU0/1/3利用率为100%，GPU2约34%。SwapFree仍低于1,024 MiB门槛，继续保留现有队列，不启动新GPU任务。

### 5.61 05:14 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为214/256，共1,494/1,536个分支，剩42个；GPU3日志更新时间为05:14，末尾仍为Gymnasium告警，未检出Traceback、OOM、Exception或Killed。Guidance仍为540/1,008：PushT、Reacher各完成post-opt 216/216与GF 36/36，TwoRoom完成GF 36/36，TwoRoom post-opt与Cube两类变体未开始。计时仍为1/2,600条粗测、0条细测，尚无精测选择和方法选择文件。宿主MemAvailable约279.1 GiB、SwapFree约65.3 MiB；GPU0–3可用显存约17.1/18.6/35.7/22.2 GiB，GPU0/1/3利用率为100%、GPU2约26%。SwapFree仍低于1,024 MiB门槛，保留当前队列，不启动新GPU任务。

### 5.62 05:20 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为219/256，共1,499/1,536个分支，剩37个；GPU3日志更新时间为05:19，末尾是Gymnasium告警，未检出Traceback、OOM、Exception或Killed。Guidance仍为540/1,008：PushT、Reacher的post-opt各216/216、GF各36/36；TwoRoom GF 36/36，TwoRoom post-opt与Cube两类变体未开始。计时仍只有1/2,600条粗测，细测为0，精测选择和方法选择文件均未生成。宿主MemAvailable约276.7 GiB、SwapFree约48.0 MiB；GPU0–3可用显存约17.1/18.6/35.7/20.1 GiB，GPU0/1/3利用率100%、GPU2约41%。SwapFree仍远低于1,024 MiB门槛，保留现有队列，不启动新GPU任务。

### 5.63 05:26 续跑检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为225/256，共1,505/1,536个分支，剩31个；GPU3日志更新至05:26，末尾仍为Gymnasium告警，未检出Traceback、OOM、Exception或Killed。Guidance仍为540/1,008，完成分布不变；粗测仍为1/2,600条、细测0条，尚无精测选择和方法选择文件。宿主MemAvailable约278.4 GiB，但SwapFree降至约0.48 MiB；GPU0–3可用显存约17.1/18.6/35.7/22.2 GiB，GPU0/1/3利用率100%、GPU2约59%。SwapFree远低于1,024 MiB门槛，继续保留现有队列，不启动新GPU任务。

### 5.64 05:28 Swap压力只读复查

因05:26的SwapFree读数降至0.48 MiB，补做一次只读压力快照：`vmstat -w 1 2`第一行是启动以来平均si/so约66/172 KiB/s，唯一的1秒样本约4/0 KiB/s；memory PSI的avg10为0.00、avg60为0.08、avg300为0.06。该短样本没有显示持续换页，MemAvailable约278.3 GiB，SwapFree回升到约4.1 MiB。GPT‑6 Sol high只读评审认为，Swap余量低本身不能证明当前有OOM或需要终止正常推进的worker；将1,024 MiB设为新GPU任务准入门槛属于保守操作规则。继续保留Cube worker，暂不启动新任务；若后续出现持续si/so、PSI明显升高、MemAvailable快速下降、OOM日志或进度停滞，再提前排查。

### 5.65 05:33 续跑与内存压力检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为231/256，共1,511/1,536个分支，剩25个；GPU3日志更新至05:33，未检出Traceback、OOM、Exception或Killed。Guidance仍为540/1,008，计时仍为1/2,600条粗测、0条细测。宿主MemAvailable约278.5 GiB、SwapFree约8.25 MiB；`vmstat -w 1 2`第一行的si/so约66/172 KiB/s是启动以来平均值，唯一的1秒样本为约356/0 KiB/s；memory PSI avg10/avg60/avg300为0.32/0.11/0.02，近期avg10略升但整体数值仍低。GPU0–3可用显存约17.1/18.6/35.7/19.6 GiB，GPU0/1/3利用率100%、GPU2约51%。短样本有少量换入、没有换出，不足以证明持续内存压力；保留现有worker，不启动新GPU任务，并继续观察趋势。

### 5.66 05:40 续跑与内存压力检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为238/256，共1,518/1,536个分支，剩18个；GPU3日志更新至05:40，仍未检出Traceback、OOM、Exception或Killed。Guidance仍为540/1,008，计时仍为1/2,600条粗测、0条细测。宿主MemAvailable约288.6 GiB、SwapFree约39.2 MiB；`vmstat -w 1 2`第一行是启动以来平均值，唯一1秒样本si/so约4/0 KiB/s；memory PSI的avg10、avg60、avg300均为0.00。GPU0–3可用显存约17.1/18.6/35.7/22.2 GiB，GPU0/1/3利用率100%、GPU2约32%。当前短时压力指标平稳，但SwapFree仍低于1,024 MiB准入门槛；继续保留现有worker，不启动新GPU任务。

### 5.67 05:46 续跑与内存压力检查

Cube候选池S1/S2/S5/S10/S16均为256/256，S32为243/256，共1,523/1,536个分支，剩13个；GPU3日志更新至05:46，未检出Traceback、OOM、Exception或Killed。Guidance仍为540/1,008，计时仍为1/2,600条粗测、0条细测。宿主MemAvailable约283.6 GiB、SwapFree约53.1 MiB；`vmstat -w 1 2`第一行是启动以来平均值，唯一1秒样本si/so为0/0 KiB/s；memory PSI avg10/avg60/avg300均为0.00。GPU0–3可用显存约17.1/18.6/35.7/16.5 GiB，GPU0/1/3利用率100%、GPU2约67%。GPU3剩余显存较前次下降，但当前worker日志和进度正常；SwapFree仍低于1,024 MiB准入门槛，不启动新GPU任务。

### 5.68 05:56 候选池收尾检查

Cube候选池S1/S2/S5/S10/S16均已256/256，S32为253/256，共1,533/1,536个分支，剩3个；已落盘分支文件编号连续，尾部JSON记录均为completed。Cube日志更新时间为05:56，未检出Traceback、OOM、Exception或Killed；`manifest.json`和`controls.jsonl`尚未生成，继续等待当前收尾队列并避免重复启动。Guidance仍为540/1,008，粗测仍1/2,600、细测0。宿主MemAvailable约278.6 GiB、SwapFree约53.3 MiB；`vmstat`唯一1秒样本si/so约4/0 KiB/s，memory PSI三个窗口均为0。GPU0–3可用显存约17.1/18.6/35.7/16.8 GiB，GPU0/1/3利用率100%、GPU2约69%。SwapFree仍低于1,024 MiB门槛，不启动新GPU任务。

### 5.69 06:03 Cube候选池完成并进入controls

Cube候选池S1/S2/S5/S10/S16/S32的分支文件现均为256/256，共1,536/1,536；逐分支文件编号连续，尾记录均为completed。当前队列已转入controls执行：已落盘`control_0000`至`control_0002`，各50条记录、末条状态为completed（原始队列3/240；其中规范子集3/222）；`controls.jsonl`与最终`manifest.json`尚未生成，继续保留当前worker等待收尾。GPU3日志更新时间为06:03，未检出Traceback、OOM、Exception或Killed。Guidance仍为540/1,008，粗测仍1/2,600、细测0。宿主MemAvailable约280.3 GiB、SwapFree约1.9 MiB；`vmstat`唯一1秒样本si/so为0/0 KiB/s，memory PSI avg10/avg60/avg300均为0。GPU0–3可用显存约17.1/18.6/35.7/20.8 GiB，GPU0/1/3利用率100%、GPU2约60%。当前SwapFree仍低于1,024 MiB准入门槛，不追加其他GPU任务。

### 5.70 06:09 Cube controls续跑

Cube六个flow-step均有256个完整候选分支，`records.jsonl`已聚合生成（约2.18 GB）。controls原始队列推进到9/240，规范子集为9/222；9个文件各50条记录，均以completed结束。最终`controls.jsonl`和`manifest.json`仍待当前worker收尾。GPU3日志更新至06:09，无Traceback、OOM、Exception或Killed。Guidance仍540/1,008，计时粗测1/2,600、细测0。宿主MemAvailable约284.2 GiB、SwapFree约2.1 MiB，`vmstat`唯一1秒样本si/so为0/0 KiB/s，memory PSI三个窗口均为0。GPU0–3可用显存约17.1/18.6/35.7/21.5 GiB，GPU0/1/3利用率100%、GPU2约64%。不启动其他GPU任务，继续保留controls worker。

### 5.71 06:17 Cube controls续跑

Cube候选分支全量仍为1,536/1,536，`records.jsonl`约2.18 GB；controls原始队列到16/240，规范子集16/222，16个control文件各50条记录且均完整completed。`controls.jsonl`和最终`manifest.json`仍未生成，GPU3日志更新至06:17且未检出Traceback、OOM、Exception或Killed。Guidance仍540/1,008，计时粗测1/2,600、细测0。宿主MemAvailable约280.3 GiB、SwapFree约11.4 MiB；`vmstat`唯一1秒样本si/so为0/0 KiB/s，memory PSI各窗口均为0。GPU0–3可用显存约17.1/18.6/35.7/22.2 GiB，GPU0/1/3利用率100%、GPU2约33%。保留controls worker，不启动新的GPU任务。

### 5.72 06:22 Cube controls续跑

Cube候选分支全量为1,536/1,536，controls原始队列推进到20/240，规范子集20/222；20个control文件均有50条记录，末条状态为completed。`controls.jsonl`与最终`manifest.json`仍未生成，GPU3日志更新至06:21且未检出Traceback、OOM、Exception或Killed。Guidance仍540/1,008，粗测仍1/2,600、细测0。宿主MemAvailable约279.7 GiB、SwapFree仅约0.055 MiB；`vmstat`唯一1秒样本si/so为0/0 KiB/s，memory PSI三个窗口均为0。GPU0–3可用显存约17.1/18.6/35.7/20.7 GiB，GPU0/1/3利用率100%、GPU2约44%。当前worker继续推进；不启动其他GPU任务。

### 5.73 06:28 Cube controls续跑

Cube候选分支仍为1,536/1,536；controls原始队列推进到26/240，规范子集26/222，26个文件均为50条记录且完整completed，aggregate和最终manifest尚未生成。GPU3日志更新时间06:28，无Traceback、OOM、Exception或Killed。Guidance仍540/1,008，计时粗测1/2,600、细测0。宿主MemAvailable约279.6 GiB、SwapFree约6.4 MiB；`vmstat`唯一1秒样本si/so为0/0 KiB/s，memory PSI avg10为0、avg60为0.01、avg300为0。GPU0–3可用显存约17.1/18.6/35.7/22.2 GiB，GPU0/1/3利用率100%、GPU2约62%。继续保留当前worker，不启动其他GPU任务。

### 5.74 06:33 Cube controls续跑

Cube候选池维持1,536/1,536，controls原始队列推进到31/240，规范子集31/222；已落盘31个control文件均完整completed，aggregate与最终manifest仍未生成。GPU3日志更新至06:33，无Traceback、OOM、Exception或Killed。Guidance仍540/1,008，计时粗测1/2,600、细测0。宿主MemAvailable约276.2 GiB、SwapFree约6.6 MiB；`vmstat`唯一1秒样本si/so为4/0 KiB/s，memory PSI三个窗口均为0。GPU0–3可用显存约17.1/18.6/35.7/17.2 GiB，GPU0/1/3利用率100%、GPU2约75%。保留现有worker，不启动新的GPU任务。

### 5.75 06:40 Cube controls续跑

Cube候选池仍为1,536/1,536，controls原始队列推进到37/240，规范子集37/222，所有已生成control文件均有50条记录且完整completed；aggregate与最终manifest尚未生成。GPU3日志更新至06:40，未检出Traceback、OOM、Exception或Killed。Guidance仍540/1,008，计时粗测1/2,600、细测0。宿主MemAvailable约279.9 GiB、SwapFree约6.7 MiB；`vmstat`唯一1秒样本si/so为0/0 KiB/s，memory PSI三个窗口均为0。GPU0–3可用显存约17.1/18.6/35.7/22.2 GiB，GPU0/1/3利用率100%、GPU2约72%。保留controls worker，不启动其他GPU任务。

### 5.76 06:44 Cube候选池CPU阶段汇总

Cube的CPU汇总已完成：50个固定状态、6档flow step、每档256个候选，共76,800条candidate记录；摘要标记`complete=true`，并按状态做10,000次cluster bootstrap。所有flow step的oracle与B-selected成功率均为1.000，随机成功率为0.997–1.000；总体selected-minus-random成功率差为0.00078，95% CI为[0, 0.00216]。总体归一化selection regret为0.1723，95% CI为[0.1328, 0.2125]。各flow step的预测-真实物理距离Pearson相关性先在每个状态内计算；每档只有37/50个状态的相关性有定义，另13个状态因256个候选的真实距离恒定而无法计算，故均值不包含这13个状态。Pearson均值约为-0.016至0.023。初步看，Cube的二值成功率处于天花板，B-selected没有显示明确的成功率优势；同时存在正selection regret，平均状态内线性相关较低。结论限于本轮固定的50个状态与该候选池，不能外推为训练因果结论。controls仍需完成，以检验动作扰动、顺序操作和随机控制下的真实物理效应；截至06:40原始队列37/240、规范子集37/222，聚合和manifest尚未完成。

### 5.77 06:45 GPT-6 Sol high结果评审

评审建议按当时规范222动作口径完成剩余185条（37/222）并生成聚合/manifest，暂不增加新采样实验。后续审计发现正在运行的worker载入了旧240动作生成器：其额外执行每个锚点、每种重排下的shift=±1变体；详见5.84。总体成功率增益CI包含0，不能据此声称稳定的二值成功提升。selection regret定义为B-selected真实归一化距离减去候选池最小真实距离；其正值及95% CI支持B在这批固定候选中持续错过更近动作。评审当时指出，接近零的平均状态内Pearson相关性不能单独证明B失败，也可能掩盖状态间差异；当时尚未计算Spearman等秩相关。controls完成前仍不能判断动作局部灵敏度、顺序干预效应、随机/零动作基线或模型action-effect误差来源。后续先用现有数据做逐任务/逐S和状态内排序分析，并结合距离分布、成功阈值margin及配对controls效应；bootstrap只描述这50个状态的抽样不确定性，不支持新状态分布、新checkpoint或新任务的外推。

### 5.78 06:52 Cube候选排序分析

对Cube 76,800条候选记录完成流式CPU排序分析并通过记录数校验；每个flow step包含50个状态，每状态256个候选，按flow step分别进行10,000次状态级bootstrap。Pearson和Spearman均先在状态内计算，再对相关性有定义的状态取均值。每档均有37个定义状态；其余13个状态的真实距离在256个候选间完全相同，因此两种相关性均未定义。这13个状态每档均是同一组状态，不将未定义相关性填为0。

| Flow step | Pearson均值 [95% CI] | Spearman均值 [95% CI] | 定义状态regret均值 [95% CI] |
|---:|---:|---:|---:|
| 1 | -0.0002 [-0.0728, 0.0710] | 0.0029 [-0.0569, 0.0667] | 0.1872 [0.1424, 0.2346] |
| 2 | -0.0046 [-0.0333, 0.0228] | -0.0081 [-0.0462, 0.0282] | 0.2101 [0.1672, 0.2523] |
| 5 | -0.0159 [-0.0486, 0.0163] | -0.0074 [-0.0389, 0.0222] | 0.2215 [0.1799, 0.2652] |
| 10 | 0.0077 [-0.0181, 0.0316] | 0.0106 [-0.0147, 0.0333] | 0.2622 [0.2146, 0.3119] |
| 16 | 0.0175 [-0.0097, 0.0430] | 0.0207 [-0.0071, 0.0466] | 0.2518 [0.2035, 0.3065] |
| 32 | 0.0229 [-0.0048, 0.0479] | 0.0210 [-0.0083, 0.0485] | 0.2642 [0.2129, 0.3205] |

Pearson和Spearman的全部区间均跨过0，说明在这37个可排序状态中，当前汇总没有显示稳定的正或负平均相关；状态间仍有差异，均值不等于每个状态都无关联。定义状态上的regret均值均为正，而上述13个距离恒定状态的regret恰为0；因此不能只引用包含全部50个状态的regret均值来描述可排序状态中的选择差距。与前述二值成功率天花板结果合看，B-selected在这批固定候选中没有稳健的成功率优势，但对真实距离仍有可见的候选内选择遗憾。此证据描述固定50状态和既定候选池，不足以归因为模型结构或训练失败，也不支持跨任务/新状态外推。

Sol high评审建议完成既定controls与聚合/manifest，不增加新的候选采样实验。Cube controls仍在运行；后续将用其配对对照分析动作扰动、顺序操作及随机/零动作基线，再判断是否有必要设计额外实验。

### 5.79 06:59 Cube controls续跑

Cube controls原始旧240动作队列推进到54/240；按`shift=0`保留的规范子集为52/222，两个已完成的`shift=±1`动作不属于规范子集。逐文件核验54个文件均有50条记录且末条`outcome_status=completed`，较06:52快照新增6个原始动作。GPU3日志更新到06:59，末尾仅有Gymnasium初始化/类型警告，未检出Traceback、OOM、Exception或Killed；虽然当前会话的进程列表未显示对应worker，但文件与日志均在继续增长，因此保留现有任务而不重启。Cube `controls.jsonl`和根级`manifest.json`尚未生成。

宿主`MemAvailable`约307 GiB、`SwapFree`约2.6 GiB；`vmstat`本次1秒样本si/so为0/0 KiB/s，memory PSI avg10/avg60/avg300均为0。GPU0–3可用显存约17.1/18.6/47.4/22.2 GiB，利用率约100/100/0/100%。尽管SwapFree已超过新任务的1,024 MiB准入线，当前GPU负载仍高且Cube controls尚未收尾，继续等待现有队列，不追加GPU工作。

### 5.80 07:03 Cube候选距离与成功阈值margin

对同一76,800条候选记录补算了距离分布和成功阈值margin，产物为`analysis/candidate_distance_margin_cube.json`。Cube的真实距离已按成功阈值0.04米归一化，成功边界为1；定义margin = 1 − normalized distance，正值表示候选在阈值内。每档flow step上，B-selected在50/50个状态均成功，平均归一化距离为0.599–0.616，平均margin为0.384–0.401；按状态bootstrap的平均margin 95% CI均高于0。每档有8–11/50个状态的margin不超过0.1，即实际位置距0.04米成功边界不超过约0.004米；margin第10百分位为0.038–0.072。这补充了成功率天花板的解释：固定候选池上的B-selected成功率为满分，但部分状态接近成功边界，二值成功率看不出候选间的距离质量差异。区间只描述这50个固定状态的状态级不确定性。

| Flow step | B-selected平均距离 | 平均margin [95% CI] | margin P10 | margin≤0.1状态数 |
|---:|---:|---:|---:|---:|
| 1 | 0.616 | 0.384 [0.290, 0.483] | 0.038 | 11/50 |
| 2 | 0.613 | 0.387 [0.294, 0.486] | 0.054 | 11/50 |
| 5 | 0.599 | 0.401 [0.311, 0.497] | 0.057 | 8/50 |
| 10 | 0.616 | 0.384 [0.291, 0.482] | 0.050 | 10/50 |
| 16 | 0.602 | 0.398 [0.306, 0.497] | 0.072 | 8/50 |
| 32 | 0.606 | 0.394 [0.301, 0.493] | 0.049 | 10/50 |

### 5.81 07:05 Cube controls续跑

Cube controls原始队列到58/240，规范子集54/222；58个文件均完整包含50条记录并以`outcome_status=completed`结束。较06:59新增4个原始动作，其中规范子集净增2个。GPU3日志更新到07:05，未检出Traceback、OOM、Exception或Killed；聚合`controls.jsonl`和最终`manifest.json`仍未生成。宿主MemAvailable约306 GiB、SwapFree约2.6 GiB，memory PSI三个窗口均为0，`vmstat`本次1秒样本si/so为0/0 KiB/s。GPU0–3可用显存约17.1/18.6/47.4/16.5 GiB，利用率约100/100/0/100%。保留当前队列，继续等待controls完成后再生成汇总并执行配对效应分析。

### 5.82 07:08 Cube候选池物理量probe补充

已将冻结的图像训练Ridge probe应用到Cube的76,800条候选记录，并写回`diagnostics/probe/cube/result.json`。以同一物理位置目标和同一probe比较：真实future latent读出MAE/RMSE为0.00691/0.01065米；B预测future latent读出MAE/RMSE为0.05326/0.09029米，前者误差分别约为后者的1/7.7与1/8.5。两项均覆盖同一批候选记录，episode划分审计已确认probe训练/验证与评测轨迹不相交。

该差距表明B预测latent经过物理probe解码后的坐标误差明显高于真实latent参照；预测latent支路同时包含B的future-latent预测误差和probe在该表示上的读出误差，不能据此单独断言动力学预测是唯一来源，也不能替代基于真实环境状态的controls分析。

### 5.83 07:10 Cube controls续跑

Cube controls原始队列推进到63/240，其中规范子集57/222；所有63个文件各含50条记录并完整结束。较06:59原始增加9个动作，规范子集增加5个。GPU3日志和最新文件更新时间均为07:09，未发现Traceback、OOM、Exception或Killed。聚合文件和manifest仍待队列收尾。宿主MemAvailable约305 GiB、SwapFree约2.6 GiB，memory PSI三个窗口均为0，`vmstat`最近1秒样本si/so为0/0 KiB/s。GPU0–3可用显存约17.1/18.6/47.4/17.5 GiB，利用率约100/100/0/100%。沿用评审顺序，暂不并发启动新的GPU队列。

### 5.84 07:21 Cube controls运行协议兼容性审计

对照计划、运行中control文件metadata和工作树源码后，确认运行进程在02:04启动时载入了旧版240动作生成器；工作树于06:52更新为当前222动作生成器。旧版每个锚点执行3种block排列×3个shift（0、+1、−1），当前规范协议只保留每个排列的shift=0。因此旧队列包含计划的222条规范动作以及18条额外shift动作，属于可由原记录无损筛选的协议兼容问题，不需要重跑环境。

07:21时原始队列74/240，74个文件全部完整；其中保留shift=0后的规范子集为68/222，已完成6条额外shift动作。额外动作分布在三个锚点的block-transform组，每锚点各有6条。GPU3日志更新至07:21，无错误标记；controls聚合文件与manifest尚未生成。Sol high只读评审建议保留完整240条原始数据，等待队列结束后逐状态核对当前222条规范动作与旧记录的动作数组/metadata，移除18条shift≠0行并重编号，另存原始240条备份，再校正manifest；不得只根据总行数认定协议兼容。该处理沿用Reacher和TwoRoom已完成的240→222动作级审计流程。

本次宿主MemAvailable约276 GiB、SwapFree约2.5 GiB，memory PSI avg10/avg60均为0、avg300为0.04；`vmstat`唯一近期样本si/so为0/0 KiB/s。GPU0–3可用显存约17.1/18.6/35.5/19.0 GiB，利用率约100/100/1/100%。原始队列剩166条，按最近约0.9条/分钟估算约需3小时；该时间不含后续兼容迁移和分析。

### 5.85 07:35 Cube controls前缀兼容性审计

用提升权限的主机进程视图确认，GPU3 worker（PID 3459660）持有`.cube.lock`，GPU0 worker（PID 3008546）在同一锁上等待；两个进程没有并发写入。普通沙箱的`/proc`视图看不到宿主机进程，不能据此判断worker退出。GPU3仍在写control文件，GPU0–3当时可用显存约17.1/18.6/35.5/16.5 GiB；未启动新GPU任务。

对control文件动态快照0–84（85个原始动作）完成CPU兼容性审计：4,250条原始状态结果均有50个完整slot且状态为completed。按metadata移除`shift=±1`的源索引52、53、55、56、58、59后，得到79个规范动作、3,950条完整结果。使用每个状态0–2号control的anchor重新生成当前动作，并逐状态比较动作数组和metadata（旧metadata移除`shift`字段后）；79个动作全部精确相同，没有错位或数值差异。审计期间队列继续增长，因此GPU状态快照为83/240原始、77/222规范，而CPU审计快照已到85/240、79/222。物理零控制尚未生成，本次不覆盖该末尾动作；全量迁移仍须在worker结束后再做完整240→222校验，并保存原始数据及独立审计记录。

本轮排查未能取得内核OOM日志：当前权限下`journalctl -k`没有可见条目，`dmesg`也未返回可见的OOM匹配项；这不能作为“未发生OOM”的证据。应用日志无Python traceback或显式CUDA OOM文本，GPU3进程仍存活，因此当前观察到的是沙箱进程视图差异，而不是已确认的worker崩溃。

### 5.86 07:42 Cube controls续跑与压力复查

原始controls队列到91/240；其中排除已落盘的6个`shift=±1`动作后，当前规范前缀为85/222。91个文件均有50个completed状态结果，无缺失或格式异常。较07:35审计快照新增6个动作。GPU3 worker仍持有输出锁并继续推进，GPU0 worker仍等待同一锁；GPU3日志更新时间07:42，近期未检出traceback、OOM、Exception或Killed。GPU0–3可用显存约17.1/18.6/35.5/22.2 GiB，利用率100/100/20/100%；不启动新GPU任务。

宿主MemAvailable约282 GiB，SwapFree约1.0 GiB。memory PSI avg10/avg60/avg300为3.53/2.20/0.73，较此前快照升高；本次`vmstat`唯一1秒样本si/so约4/0 KiB/s。该瞬时样本没有显示持续换出，且可用内存仍充足，但压力趋势需要继续观察。现有信息仍不足以把先前未确认的进程不可见归因于OOM。

### 5.87 07:50 Cube controls续跑

原始controls队列推进到98/240，排除6个已完成的`shift=±1`变体后，规范前缀为92/222。98个动作文件均有50个completed状态结果，无格式、slot或状态异常；较07:42新增7个动作。GPU3 worker仍持有输出锁，GPU0等待；日志更新时间07:50，未检出traceback、OOM、Exception或Killed。GPU0–3可用显存约17.1/18.6/35.5/16.5 GiB，利用率100/100/59/100%；不启动额外GPU任务。

宿主MemAvailable约283 GiB，SwapFree约724 MiB；memory PSI avg10/avg60/avg300回落至0.29/0.11/0.20，`vmstat`唯一1秒样本si/so约8/0 KiB/s。当前内存压力较07:42缓和，但SwapFree仍低于新任务的1,024 MiB准入线。按07:42–07:50约0.88个原始动作/分钟估算，剩余142个原始动作约需2.7小时；该估算随吞吐和收尾开销变化。

### 5.88 07:57 Cube controls续跑与压力复查

原始controls队列到104/240，去掉6个已完成的`shift=±1`变体后规范前缀为98/222。104个文件均包含完整50个slot且状态为completed，无异常记录；较07:50新增6个动作。GPU3继续持有锁并运行，GPU0等待；GPU3日志更新时间07:57，没有traceback、OOM、Exception或Killed标记。GPU0–3可用显存约17.1/18.6/35.5/16.5 GiB，利用率100/100/22/100%；未启动新GPU任务。

宿主MemAvailable约278 GiB、SwapFree约748 MiB；memory PSI avg10/avg60/avg300为0.00/0.04/0.05，`vmstat`唯一1秒样本si/so为0/0 KiB/s，较07:50快照回落。按07:50–07:57约0.86个原始动作/分钟估算，剩余136个原始动作约需2.6小时，另留聚合、迁移及审计时间。

07:58 CPU动作级审计快照扩展至源索引0–104：105个原始动作全部完整；筛除6个`shift=±1`项后，99个规范动作共4,950条记录，动作数组和metadata与当前生成器逐项精确一致。审计期间新文件继续生成；物理零控制尚未进入快照。

### 5.89 08:04 Cube controls续跑与anchor1区块变换审计

08:04资源快照时原始controls队列为110/240，规范前缀103/222；其中7个`shift=±1`动作应剔除。110个文件均有完整50个slot和completed状态。GPU3持有`.cube.lock`继续运行，GPU0等待；GPU3日志更新时间08:04，无错误标记。GPU0–3可用显存约17.1/18.6/35.5/20.7 GiB，利用率100/100/66/100%。宿主MemAvailable约277 GiB、SwapFree约669 MiB，memory PSI avg10/avg60/avg300为0.01/0.16/0.14，`vmstat`唯一1秒样本si/so为0/0 KiB/s。保留当前队列，不启动新GPU任务。

08:05 CPU动作级审计推进至源索引0–110：111个原始动作均完整，其中8个为`shift=±1`；其余103个规范动作共5,150条结果的动作数组和metadata均与当前222动作生成器逐项精确一致。此次覆盖了anchor1的首个shift=0区块变换；随后新增的两个shift变体被排除，未增加规范动作计数。物理零控制尚未生成，最终全池审计仍待队列结束后完成。

### 5.90 08:11 anchor1区块变换组审计

08:11状态快照为115/240个原始control文件、105/222个规范动作；shift变体累计10个，文件均完整，GPU3日志继续更新且无异常标记。GPU3持有锁，GPU0等待。GPU0–3可用显存约17.1/18.6/35.5/20.1 GiB，利用率100/100/77/100%；宿主MemAvailable约278 GiB、SwapFree约620 MiB，memory PSI avg10/avg60/avg300为0.00/0.05/0.11，`vmstat`单秒样本si/so为0/0 KiB/s。保留已有队列，不启动新GPU任务。

08:12 CPU审计快照扩展至源索引0–115：116个原始动作均完整，11个`shift=±1`动作被筛除，105个规范动作共5,250条结果的动作数组和metadata全部精确匹配当前生成器。新增的源索引115是额外shift变体，因此本次没有增加规范动作数；anchor1的3个shift=0变换均已逐状态核对。物理零控制尚未生成。

### 5.91 GPU3任务级内存事件与Sol high评审

读取GPU3 worker专属cgroup后，`memory.events`中的low/high/max/oom/oom_kill/oom_group_kill均为0，`memory.max`与`memory.high`均未设上限；`memory.current`约20.85 GB、峰值约23.23 GB，cgroup `memory.swap.current`约61.5 MB，进程`VmSwap`约60 MB。最新memory PSI avg10/avg60/avg300为0.07/2.77/5.80，IO PSI接近0；`pgmajfault`约30,039为进程累计计数，单点累计值不能说明当前故障。与较早同一服务PSI avg10约24%的快照相比，最新10秒窗口已回落，但60/300秒窗口仍体现此前等待。

GPT‑6 Sol high只读评审建议保留GPU3 worker：当前没有cgroup OOM证据，宿主可用内存充足，未见持续换出，controls持续落盘；PSI反映等待时间，不单独构成停机条件。继续按至少5分钟间隔联合观察该cgroup PSI/current/events/缺页、宿主MemAvailable/SwapFree/si-so、GPU余量和文件产出速度。只有在PSI持续升高并伴随吞吐明显下降、换入/换出持续增加、宿主可用内存快速下降、OOM事件/日志出现或文件长期停写时再干预。

### 5.92 08:16–08:20 Cube controls续跑与120动作前缀审计

08:16资源快照时原始control文件推进到120/240（索引0–119），共6,000条记录；GPU3 worker仍运行，GPU0仍等待同一输出锁。GPU3日志更新至08:20:16，未发现Traceback、CUDA OOM、OOM、Exception或Killed标记。GPU0–3可用显存约17.1/18.6/35.5/22.2 GiB；宿主MemAvailable约278 GiB、SwapFree约221 MiB，memory PSI avg10/avg60/avg300为0.00/0.04/0.08，`vmstat`单秒样本si/so为0/0 KiB/s。GPU3 cgroup的memory.events中oom与oom_kill均为0，memory.current约19.5 GiB、memory.peak约21.6 GiB、memory.swap.current约58 MiB；08:19复查PSI为0.00/0.05/2.42。保持现有队列，不启动新GPU任务。

CPU审计逐行核对了120个动作文件：6,000条记录全部具有50个slot且`outcome_status=completed`。根据`control_metadata`移除旧协议的12个`shift=±1`动作（源索引52、53、55、56、58、59、109、110、112、113、115、116），余下108个规范动作、5,400条记录。对50个状态分别从前三个anchor重建当前确定性控制动作后，108个保留动作的数组及去掉旧`shift`字段后的metadata均逐项完全一致，差异数为0；最新覆盖到规范动作索引107。此前一次快速状态探针误查不存在的`status`/`metadata`字段，产生了“状态缺失”和“shift数为0”的假警报；已按实际记录字段`outcome_status`/`control_metadata`重做审计，数据本身没有异常。

按08:12到08:16约0.86个原始动作/分钟估算，剩余120个动作约需2.3小时，另需聚合、迁移和最终全量审计时间。原始240动作文件仍须完整保留；最后只在所有worker释放锁后执行240→222迁移及全池校验。

### 5.93 08:23 Cube controls续跑与126动作前缀审计

08:23状态快照时队列推进到126/240个原始动作（索引0–125），共6,300条完整记录。GPU3仍在运行、GPU0仍等待；两者打开同一个`.cube.lock`，GPU3日志更新时间08:23:38且没有Traceback、CUDA OOM、OOM、Exception或Killed标记。GPU0–3可用显存约17.1/18.6/35.5/22.2 GiB，利用率100/100/69/100%。宿主MemAvailable约279 GiB、SwapFree约227 MiB，memory PSI avg10/avg60/avg300均为0，`vmstat`单秒样本si/so为0/0 KiB/s；GPU3 cgroup memory.current约19.7 GB、memory.peak约23.2 GB、memory.swap.current约61 MB，memory PSI avg10/avg60/avg300为0/0/0.83，所有OOM事件计数仍为0。

CPU精确审计扩展到原始索引0–125：126个文件的6,300条记录全部完成。筛除12个`shift=±1`动作后，114个规范动作共5,700条记录；按50个状态重建当前222动作协议并逐项比较后，动作数组和metadata差异均为0，覆盖至规范动作索引113。08:16–08:23约新增6个原始动作，按约0.89动作/分钟估算，剩余114个动作约需2.1小时，另加收尾、迁移与审计时间。

### 5.94 08:36–08:43 配对controls分析复核与Cube进度

08:43资源快照时Cube原始controls为144/240（索引0–143），规范子集132/222；144个文件共7,200条记录，全部50-slot completed且索引连续。GPU3 worker继续更新日志（08:43:29），GPU0仍等待同一`.cube.lock`；GPU3日志没有Traceback、OOM、Exception或Killed。GPU0–3可用显存约17.1/18.6/35.5/16.5 GiB，利用率100/100/71/100%；宿主MemAvailable约278 GiB、SwapFree约229 MiB，memory PSI和`vmstat`短时si/so均为0。GPU3 cgroup memory.current约22.1 GB、memory.peak约23.2 GB、memory.swap.current约60 MB，memory.events中的所有OOM计数均为0。

对当前144个文件做逐状态协议审计：筛除旧`shift=±1`的12个动作后，132个规范动作共6,600条记录与50个状态各自重建的当前222动作生成器逐项完全一致，动作数组和metadata差异数均为0，覆盖至规范动作索引131。08:36–08:43新增7个原始动作，按约1个/分钟估算，剩余96个约需1.6小时，另加迁移与最终分析。

新增离线分析入口[`round5_phase1_5_control_analysis.py`](../../../scripts/round5_phase1_5_control_analysis.py)，并为PushT、Reacher、TwoRoom生成`control_paired_effects.json`。输入检查要求222×50条completed控制记录、每状态控制编号齐全，且18个RMS分组、9个block变换组、64个高斯方向和两个零动作均完整。每组先在状态内平均多个动作，再按状态做10,000次bootstrap；同anchor RMS/block以对应单个anchor为基线，高斯与零动作使用同状态三个anchor的平均结果作为复合基线（它不是一项实际策略）。高斯组明确标为RMS归一化高斯方向：每个N(0,1)样本整体缩放到RMS=1。

主要成功率差为控制减去同状态三个anchor平均成功率；成功定义仍是50步evaluation budget内出现成功事件。距离同时保留记录中“raw step≤25的最后状态距离”及严格同raw step的辅助配对分析。后者逐个状态/动作寻找双方都恰好记录在5/10/15/20/25步的最大公共步数；仅纳入有公共milestone的配对，先平均每个状态内有效动作，再bootstrap状态。正距离差表示控制更差。

| 任务 | 三个anchor成功率（data / S=1 / S=2） | RMS归一化高斯：成功率差及95% CI | 物理零：成功率差及95% CI | 高斯同步步数距离差及覆盖 |
|---|---:|---:|---:|---:|
| PushT | 0.12 / 0.10 / 0.14 | −0.120 [−0.213, −0.040] | −0.120 [−0.213, −0.040] | +3.877 [3.141, 4.592]；3,200/3,200动作配对，50/50状态 |
| Reacher | 1.00 / 0.38 / 0.46 | −0.538 [−0.616, −0.461] | −0.593 [−0.680, −0.513] | +12.342 [10.517, 14.246]；3,063/3,200动作配对，48/50状态 |
| TwoRoom | 1.00 / 1.00 / 0.98 | −0.766 [−0.844, −0.681] | −0.913 [−0.980, −0.833] | +1.476 [1.199, 1.765]；2,552/3,200动作配对，40/50状态 |

这些效应描述固定`legacy_50`状态和当前动作池，不表示新状态泛化、模型训练因果或某个确定策略相对其他策略的净收益。Reacher的数据anchor在50个状态都成功，其中19个episode在step25前结束；TwoRoom的anchor记录中149/150次成功在step1–23结束。控制行的endpoint距离因此可能来自step25，也可能来自更早的终止步；固定长度距离差不能单独解释为同一物理时长下的效应。同raw step辅助分析覆盖数已列在表中，但仅对存在共同milestone的动作/状态有效。状态bootstrap不包含rollout间波动；RMS方向和64个高斯方向固定，组间比较未作多重检验校正。

GPT‑6 Sol high只读评审确认三任务的状态配对、bootstrap单位、变化方向、anchor/复合基线及协议覆盖实现正确；没有建议新增采样实验。建议继续完成Cube当前稳定运行的剩余controls，再按同一动作级审计和配对流程收尾。评审也指出事件成功率和距离步数须分开解释；报告与JSON已按此补充。

### 5.95 08:53–08:57 全阶段进度复核与GPT‑6 Sol high建议

08:53使用独立报告输出路径运行离线`analyze`，刷新`analysis/index.json`与`analysis/conditions.csv`。重算确认2,424条主扫描、96条固定稳定性、80条自适应稳定性全部完成，共2,600/2,600；`analysis.json`仍为`ready_for_diagnostic_review`，`method_selection`为空。该命令把新生成的报告写入`/tmp/round5_phase1_5_generated_report.md`，没有覆盖工作树中已有修改的主报告。

阶段完成项包括：四任务probe结果及汇总均已存在，Cube候选池probe attach已覆盖76,800条记录；四任务候选池与主扫描结果已完成。待收尾项目是Cube controls旧240动作队列及其240→222全量审计、配对分析；Guidance已完成540/1,008项；`timing.json`仅有1条粗测记录，尚无细测选择和方法选择文件。08:56运行离线汇总时主报告仍是旧快照；随后已用新分析重生成主报告并补入三任务配对控制结果及当前未收敛状态，Cube controls与最终方法/计时选择仍待完成。

GPT‑6 Sol high对余下实验和收敛条件的建议为：

1. 保留当前Cube controls队列，等240/240后完整核验shift=±1旧变体，再筛选、重编号并聚合规范222项。
2. 现有Guidance缺468项中，优先补与既有GF可一一比较的108项：TwoRoom post-opt 36项、Cube post-opt 36项及Cube GF 36项。其余360项低优先级网格可延期；若采取此收窄，必须将Guidance阶段标为648/1,008，禁止写成“全网格完成”，并收窄对未测尺度、flow steps及四任务完整响应面的结论。
3. 全2,600条件粗测仍是原计划的完整Pareto验收要求。若改为候选集精测，只能先基于已完成的三推理seed成功率固定候选，再对候选和强P1/P2对照做隔离粗筛与细测；报告只能声称“所选配置间的success–latency比较”，不能声称全局Pareto或最快方法。若候选及强CEM对照没有公平细测，不能宣布“Fast”、CEM替代方案或主备方法，阶段判断应为“尚未收敛”。
4. 固定`legacy_50`探索性结果不需要增加rollout或cohort。若将来需要泛化或随机推理稳定性的论文级结论，应先冻结方法，再独立验证。

本阶段判断：评审建议与当前资源/证据相符，最有价值的后续是先收尾Cube controls，再完成108项配对诊断并做公平候选精测；不值得在缺少可识别增益的情况下盲目补满其余360项。不过该取舍把原计划完整Guidance网格改成优先级子集，最终报告必须显式标注“延后/未完成”，不得把计划收窄伪装为原始验收已通过。

范围校正：当前活跃目标仍要求完成计划原定范围，故本次执行不采纳将360项Guidance标记为延后的收窄方案。Cube controls完成后继续完成剩余468项Guidance，并完成原计划全部2,600条件粗测和Pareto/主备方法分析。评审提出的108项优先级只用于排序执行顺序，不能替代后续全部条件。

### 5.102 09:35 Guidance现有产物完整性复核

逐目录检查`guidance_sweep`当前已有变体文件：PushT post-opt 216、PushT GF 36、Reacher post-opt 216、Reacher GF 36、TwoRoom GF 36，共540项。每个变体文件均覆盖50条状态记录，manifest状态为completed且没有格式或outcome状态异常。按原计划4×216 post-opt + 4×36 GF =1,008项计算，缺项468：TwoRoom post-opt 216、Cube post-opt 216、Cube GF 36。后续按此完整缺项集合续跑。

### 5.103 09:38 Cube controls续跑与兼容性前缀

09:38联合资源快照显示GPU3 worker运行并更新日志至09:37:19，GPU0仍等待同一锁。原始control文件计数为191/240；CPU精确审计过程中又落盘一个文件，实际核验索引0–191共192个文件、9,600条完整completed记录。排除18个旧shift动作后，174个规范动作的动作数组和metadata均与当前生成器在50个状态上的输出完全一致，差异为0，覆盖规范索引0–173；剩48个原始/规范动作。

GPU0–3可用显存约17.1/18.6/35.5/17.8 GiB，利用率100/100/29/100%；宿主MemAvailable约277 GiB、SwapFree约168 MiB，memory PSI三个窗口为0，`vmstat`单秒si/so为0/0。GPU3 cgroup memory.current约19.7 GiB、peak约21.6 GiB、swap.current约54 MiB，memory PSI三个窗口为0，oom/oom_kill计数为0。队列继续写入且没有观察到内存压力增长；当前无新GPU并发工作。

### 5.104 09:43 Cube controls推进及动作核对

09:43快照时队列达196/240个原始动作文件；CPU动作级复核覆盖索引0–195，共9,800条completed记录，18个旧shift动作全部排除。178个规范动作在50个状态的动作数组和metadata与当前生成器逐项完全一致，差异均为0，覆盖至规范索引177；剩44个规范动作。

GPU3继续运行并更新日志至09:43:10；GPU0仍等待同一锁。GPU0–3可用显存约17.1/18.6/35.5/22.2 GiB，利用率100/100/19/100%；宿主MemAvailable约281 GiB、SwapFree约172 MiB，memory PSI三窗口为0，`vmstat`单秒si/so为0/0。GPU3 cgroup memory.current约18.7 GiB、peak约21.6 GiB、swap.current约54 MiB，memory PSI三窗口为0，oom/oom_kill为0。按最近约1个动作/分钟估算余下44项约44分钟，收尾和全量迁移审计时间另计。

### 5.105 Phase1.5单元与协议测试

虚拟环境未安装`pytest`，改用Python内置`unittest`。使用`CUDA_VISIBLE_DEVICES=''`运行完整`tests.test_round5_phase1_5`，61项全部通过（0.982秒）；其中覆盖网格计数、同状态/RNG恢复、GF与PO的S=1等价性、控制动作确定性、222项control协议、控制池复用校验及报告逻辑。此前单独选取的9个环境恢复/GF/controls测试也全部通过。`python scripts/round5_phase1_5.py validate`返回`status=ok`，`git diff --check`通过。GPU实验任务仍按原计划独立运行。

另运行`tests.test_round4_protocol`完整协议测试，9项全部通过（1.153秒）。

### 5.106 09:49 Cube controls队列与兼容性前缀

09:49资源快照和CPU审计时原始controls为201/240个文件，实际核验索引0–200共201文件、10,050条completed记录。筛除18个旧shift动作后，183个规范动作的50状态数组和metadata与当前生成器完全一致，动作与metadata差异均为0，覆盖规范索引0–182；剩39个规范动作。

GPU3 worker仍存活并更新日志至09:49:04，GPU0仍等待同一锁。GPU0–3可用显存约17.1/18.6/35.5/16.5 GiB，利用率100/100/61/100%；宿主MemAvailable约276 GiB、SwapFree约173 MiB，memory PSI avg10/avg60/avg300为0/0.01/0，`vmstat`单秒si/so均为0。GPU3 cgroup memory.current约19.8 GiB、peak约21.6 GiB、swap.current约54 MiB，memory PSI三个窗口为0，oom/oom_kill为0。队列继续写入，保持单任务运行。

### 5.107 09:54 Cube controls收尾前进度

09:54资源快照和CPU审计时原始controls为205/240个文件；索引0–204共10,250条completed记录。筛除18个旧shift动作后，187个规范动作的50状态数组和metadata与当前生成器完全一致，差异均为0，覆盖规范索引0–186；剩35个规范动作。

GPU3 worker仍运行并更新日志至09:53:44，GPU0仍等待同一锁。GPU0–3可用显存约17.1/18.6/35.5/20.7 GiB，利用率100/100/37/100%；宿主MemAvailable约282 GiB、SwapFree约211 MiB，memory PSI avg10/avg60/avg300为0，`vmstat`单秒si/so为0。GPU3 cgroup memory.current约19.6 GiB、peak约21.6 GiB、swap.current约54 MiB，memory PSI三个窗口为0，oom/oom_kill为0。剩余35项按近期约0.8项/分钟估算约44分钟，另需稳定全池审计、聚合与迁移。

08:56–08:57 Cube资源快照：GPU0–3可用显存约17.1/18.6/35.5/20.8 GiB，利用率100/100/68/100%；宿主MemAvailable约280 GiB、SwapFree约231 MiB，memory PSI avg10/avg60/avg300为0.46/0.14/0.03，`vmstat`单秒si/so均为0。GPU3 worker专属cgroup的memory.current约17.8 GiB、memory.peak约21.6 GiB、swap.current约55 MiB，memory PSI三个窗口为0，oom/oom_kill计数均为0。原始control文件索引0–156共157/240个，7,850条记录均有50个completed slot且无坏文件；12个旧shift变体被筛除后规范前缀为145/222。动作级精确兼容比较此前已覆盖至原始索引149（138个规范动作），索引150–156仍待在队列结束后纳入全池复核。GPU3 worker（PID 3459660）仍存活并更新日志，GPU0 worker（PID 3008546）继续等待同一锁；当前无可证实的OOM事件，也不启动并发GPU任务。

### 5.96 09:06 Cube controls进度与内存状态

队列推进至原始索引0–163，共164/240个文件。CPU复核这164个文件得到8,200条记录，均有50个completed slot且没有坏文件；按旧metadata排除12个`shift=±1`动作后，当前规范前缀152/222。新增加的索引150–163尚未纳入动作数组及metadata的逐状态精确兼容比较；完整240→222审计仍留待锁释放后统一执行。

GPU3 worker（PID 3459660）继续运行并持有同一输出任务；GPU0 worker（PID 3008546）继续等待。GPU0–3可用显存约17.1/18.6/35.5/22.2 GiB，利用率100/100/44/100%。宿主MemAvailable约282 GiB、SwapFree约232 MiB，memory PSI avg10/avg60/avg300为0.00/0.02/0.01，`vmstat`单秒si/so均为0。GPU3 cgroup memory.current约17.6 GiB、memory.peak约21.6 GiB、swap.current约55 MiB，memory PSI avg10/avg60/avg300为0.01/0.02/0.00，oom/oom_kill计数为0；09:05:42应用日志仍更新且没有新增错误。原始队列剩76项，按近期约0.88项/分钟估算约87分钟，另加最终聚合、全量协议审计、迁移和配对分析；SwapFree仍低于常规新任务准入线，不并发启动GPU任务。

### 5.97 09:08 Cube controls动作级前缀审计

在GPU队列持续运行期间取得CPU动作文件快照：索引0–167共168/240个文件、8,400条完整completed记录。较此前快照新增的原始索引166和167属于旧协议`shift=±1`变体。筛除这14个shift变体后，154个规范动作的动作数组及去除旧`shift`字段后的metadata，均与当前生成器对50个状态逐项完全一致；数组与metadata差异均为0，覆盖至规范动作索引153。此快照剩72个原始动作；队列结束后仍须进行一次稳定全池审计与迁移，并校验worker释放锁及聚合文件完整性。

### 5.98 09:14 Cube controls扩展审计与GPU状态

09:14合并快照时，GPU3 worker仍持有任务并更新日志，GPU0 worker仍等待同一锁；原始control文件已达171/240（索引0–170）。本轮CPU逐文件复核了8,550条记录，全部文件各有50个completed slot。筛除16个`shift=±1`旧协议项后，155个规范动作的50状态动作数组和metadata与当前222动作生成器逐项完全一致，差异均为0，覆盖至规范动作索引154。距离全量收尾尚余69个原始动作，其中规范动作67个。

GPU0–3可用显存约17.1/18.6/35.5/22.2 GiB，利用率100/100/57/100%。宿主MemAvailable约276 GiB、SwapFree约156 MiB；memory PSI avg10/avg60/avg300为0.00/0.18/0.40，`vmstat`单秒si/so均为0。GPU3 cgroup memory.current约18.7 GiB、peak约21.6 GiB、swap.current约55 MiB，memory PSI avg10/avg60/avg300为0.00/0.64/1.35；oom/oom_kill仍为0。PSI长窗口较09:06升高，但短窗口、宿主可用内存与换页样本没有显示持续恶化，队列仍持续落盘；保持现有任务并继续按至少5分钟间隔观察，不启动其他GPU工作。

另运行CPU-only `python scripts/round5_phase1_5.py validate`，返回`status=ok`：主条件2,424、固定稳定性96、最大条件数2,600、最多130,000个episode、规范控制动作222项，与计划预算一致。

### 5.99 09:20 Cube controls队列与动作精确审计

09:20资源快照的worker仍存活，GPU3日志更新至09:19:39，GPU0继续等待同一锁。原始动作文件计数为176；CPU精确审计期间队列又写入索引176，因此审计实际覆盖索引0–176的177/240个文件，共8,850条完整completed记录。旧shift动作现已全部出现并识别为18项（索引52、53、55、56、58、59、109、110、112、113、115、116、166、167、169、170、172、173）；剩余159个规范动作与当前生成器的动作数组和metadata对50个状态逐项完全一致，差异均为0，覆盖至规范动作索引158。此时剩余63个原始动作，均属于规范子集。

GPU0–3可用显存约17.1/18.6/35.5/20.0 GiB，利用率100/100/22/100%；宿主MemAvailable约279 GiB、SwapFree约157 MiB，memory PSI avg10/avg60/avg300为0.00/0.04/0.36。`vmstat`单秒样本si/so约12/0 KiB/s。GPU3 cgroup memory.current约19.4 GiB、peak约21.6 GiB、swap.current约55 MiB，memory PSI avg10/avg60/avg300为0.00/0.00/0.37，oom/oom_kill仍为0。长窗口等待轻微增加，但没有持续换出或停写；保留队列并继续监控，不追加GPU任务。

### 5.100 09:25 Cube controls队列推进

09:25主机与GPU快照显示GPU3 worker继续运行、GPU0等待同一锁，日志更新至09:25:31。GPU0–3可用显存约17.1/18.6/35.5/22.2 GiB，利用率100/100/59/100%；宿主MemAvailable约281 GiB、SwapFree约158 MiB，memory PSI avg10/avg60/avg300为0.00/0.02/0.11，`vmstat`单秒si/so均为0。GPU3 cgroup memory.current约17.0 GiB、peak约21.6 GiB、swap.current约55 MiB，memory PSI avg10/avg60/avg300为0.00/0.00/0.11，oom/oom_kill计数仍为0。

文件队列到181/240（索引0–180）。CPU精确审计逐文件检查9,050条completed记录：18个旧shift变体全部按metadata排除，163个规范动作在50个状态上的动作数组与metadata均和当前生成器完全一致，差异数均为0，覆盖规范索引0–162。剩59个原始动作，按近期约0.8项/分钟估算约74分钟，另需最终聚合、全池审计、迁移与配对分析。未见持续换出或停写，继续保留现有worker，不并发启动GPU工作。

### 5.101 09:31 Cube controls进度与兼容性前缀

09:31快照时已落盘186/240个原始动作文件（索引0–185）；CPU审计逐行确认9,300条结果均完整且completed。排除18个shift变体后，168个规范动作的50状态动作数组和metadata全部与当前生成器逐项相同，差异数为0，覆盖规范索引0–167；剩54个原始/规范动作。

GPU0–3可用显存约17.1/18.6/35.5/16.5 GiB，利用率100/100/74/100%。宿主MemAvailable约278 GiB、SwapFree约158 MiB；memory PSI avg10/avg60/avg300为0/0/0.01，`vmstat`单秒si/so为0/0。GPU3 cgroup memory.current约20.8 GiB、peak约21.6 GiB、swap.current约54 MiB，memory PSI三个窗口均为0，oom/oom_kill仍为0。GPU3日志更新时间09:31:25，worker持续写入；54项按近期约0.83项/分钟估算约65分钟，随后还需全量审计、迁移和配对分析。继续单独运行当前队列。

### 5.108 09:59–10:02 Cube controls队列更新

09:59只读资源快照确认GPU3 worker（PID 3459660）仍运行，GPU0 worker（PID 3008546）仍等待同一锁。GPU0–3可用显存约17.1/18.6/35.5/16.5 GiB，利用率100/100/74/100%；宿主MemAvailable约278 GiB、SwapFree约211 MiB，memory PSI三个窗口均为0，`vmstat`间隔样本si/so为0/0。GPU3 cgroup memory.current约19.6 GiB、memory.peak约21.6 GiB、swap.current约54 MiB，cgroup memory PSI三个窗口为0，oom/oom_kill为0。该次快照没有显示持续内存压力或OOM。

约10:02对controls目录作CPU完整性检查时，队列已推进至连续索引0–212，共213/240个文件、10,650条记录；每个文件均有50个slot，slot顺序为0–49且状态均为completed，JSON可解析，无缺号或坏文件。相对上次动作级精确兼容审计的索引0–204，本次新增索引205–212尚未做动作数组与metadata逐状态比较；已确认的规范动作精确匹配仍为187项。剩27个原始动作，队列结束后继续执行全量精确审计、240→222迁移、manifest修订及Cube配对分析。日志在10:00:49仍有更新；GPU0–3没有启动新任务。

### 5.109 10:06 Cube controls队列更新

10:06联合资源快照时GPU3 worker（PID 3459660）仍在运行，GPU0 worker（PID 3008546）等待同一锁。GPU0–3可用显存约17.1/18.6/35.5/18.8 GiB，利用率100/100/35/100%；宿主MemAvailable约282 GiB、SwapFree约212 MiB，memory PSI avg10/avg60/avg300约0.07/0.04/0，`vmstat`间隔样本si/so为0/0。GPU3 cgroup memory.current约19.6 GiB、memory.peak约21.6 GiB、swap.current约54 MiB，memory PSI三个窗口为0，oom/oom_kill为0。

队列已推进至连续原始索引0–214，共215/240个动作文件、10,750条记录。逐文件检查确认各文件均有50个slot（0–49），JSON均可解析，记录状态全部为completed，无缺号或坏文件。动作数组和metadata逐状态精确比较仍覆盖索引0–204中的187个规范动作、差异为0；新增索引205–214尚未做此比较。尚余25个原始动作；日志在10:05:30仍有更新。待worker退出并释放锁后进行完整动作兼容审计和聚合迁移，不追加GPU任务。

### 5.110 10:14 Cube controls接近收尾

10:14联合快照时GPU3 worker（PID 3459660）和GPU0等待进程（PID 3008546）仍在；GPU3日志mtime对应约10:13:52。GPU0–3可用显存约17.1/18.6/35.5/22.2 GiB，利用率100/100/20/100%；宿主MemAvailable约285 GiB、SwapFree约212 MiB，memory PSI三个窗口为0，`vmstat`间隔样本si/so为0/0。GPU3 cgroup memory.current约17.3 GiB、memory.peak约21.6 GiB、swap.current约54 MiB，memory PSI三个窗口为0，oom/oom_kill为0。

controls文件推进至连续索引0–221，共222/240个文件、11,100条记录；逐文件完整性检查再次确认每项均覆盖50个slot，全部completed且无缺号、坏文件。尚余18个原始动作。动作数组/metadata精确比较尚未扩展，已确认结果仍为索引0–204中的187个规范动作零差异。待全部240项结束且两个进程退出、锁释放后执行稳定全池动作审计、源聚合备份、规范222动作迁移、manifest更新和Cube配对分析。

### 5.111 10:21 Cube controls收尾进度与资源

10:21联合快照确认GPU3 worker（PID 3459660）仍运行、GPU0 worker（PID 3008546）仍等待同一锁；GPU3日志更新至约10:20:54。GPU0–3可用显存约17.1/18.6/35.5/16.5 GiB，利用率100/100/20/100%，GPU3余量约高于worker启动下限0.5 GiB，不并发安排新任务。宿主MemAvailable约282 GiB、SwapFree约183 MiB；memory PSI avg10/avg60/avg300约0.10/0.05/0.01，`vmstat`间隔样本si/so为0/0。GPU3 cgroup memory.current约20.2 GiB、peak约21.6 GiB、swap.current约54 MiB，cgroup memory PSI三个窗口为0，oom/oom_kill为0。

controls文件连续覆盖原始索引0–227，共228/240个文件、11,400条记录；全量已写文件仍均为50个slot、JSON可解析、状态completed，无缺号或坏文件。剩12个原始动作。动作数组和metadata的精确复核仍待全队列完成后扩展至完整240项；此时确认的规范动作匹配为索引0–204内187项、差异为0。worker退出、锁释放后再做全池审计和迁移。

### 5.112 10:26 Cube controls即将完成

10:26联合快照确认GPU3 worker（PID 3459660）仍运行并更新日志至约10:25:44；GPU0 worker（PID 3008546）继续等待同一锁。GPU0–3可用显存约17.1/18.6/35.5/16.5 GiB，利用率100/100/48/100%，GPU3显存余量约比worker启动线高0.5 GiB，保持不并发。宿主MemAvailable约280 GiB、SwapFree约184 MiB；memory PSI avg10/avg60/avg300为0/0/0.06，`vmstat`间隔样本si/so为0/0。GPU3 cgroup memory.current约19.7 GiB、peak约21.6 GiB、swap.current约54 MiB，memory PSI三个窗口为0，oom/oom_kill为0。

control文件连续覆盖原始索引0–231，共232/240个文件、11,600条记录，所有已落盘文件均有完整的50个slot且全部completed；没有缺号或坏文件。剩8个原始动作。动作数组和metadata精确比较仍待worker结束后覆盖全池；当前已核实187个规范动作零差异。按最近约1项/分钟估算，控制队列约8分钟可结束，之后须等待GPU0/GPU3进程及文件锁全部释放再迁移。

### 5.113 Cube 240项完成与重复GPU0任务处置

10:35左右GPU3 worker完成旧协议240项controls，生成12,000条完整记录及`control_actions=240`的completed manifest，随后退出。等待锁的GPU0 `candidate-pool` 进程获取同一`.cube.lock`后开始重聚合`records.jsonl`，并从control 51开始按现行222协议逐项重评；这会重复计算可由旧240项源结果映射复用的动作。按用户偏好使用GPT-6 Sol high只读复核后，确认control 54–57与旧control 60–63逐状态50/50动作相同、success相同、true_distance最大差约7.4e-11；建议停止重复进程并精确迁移，不直接截尾。

原始`controls.jsonl`（240×50，293,840,390 bytes）mtime保持10:35:06；在SIGTERM前复制为`controls_legacy240_a6a663fb00ff.jsonl`，SHA-256为`a6a663fb00ff3d43c3ba66cb525dbe63c50cd6dcc5abc77cbcdc8a0af97a25ec`。发送SIGTERM后确认GPU0进程退出、`.cube.lock`无持有者、源聚合哈希与备份完全一致，宿主及worker均未报告OOM。GPU0重写过单动作文件51–59，后续会依据已校验的240项聚合恢复原始per-action记录，再执行全量动作/metadata/状态映射审计和222项迁移；manifest目前仍标记旧240协议。

### 5.114 Cube controls迁移、配对统计与剩余任务

使用GPT‑6 Sol high只读复核迁移事务脚本并修复原子状态写入、目录持久化、失败回滚与重入检查后，CPU dry-run通过；随后应用240→222迁移。source聚合保持12,000行且SHA-256仍为`a6a663fb00ff3d43c3ba66cb525dbe63c50cd6dcc5abc77cbcdc8a0af97a25ec`。新聚合包含11,100行/222项控制动作，SHA-256为`ca5ce5ca9b4a259ced4673740e66743615a42ddbf96c8f98c99b971d9b7b0345`；manifest更新为现行222动作协议。旧shift控制源索引`52,53,55,56,58,59,109,110,112,113,115,116,166,167,169,170,172,173`被排除。对所有11,100个保留状态/动作行，float64动作数组逐元素相等、最大绝对差为0；213项metadata原样相同，剩余9项在移除旧`shift`字段后相同。未重跑任何环境rollout，结果直接复用SHA-256验证过的源记录。

新的`controls/`目录包含222个规范per-action文件；`controls_legacy240/`按源聚合精确重建240个旧文件，并与源聚合的SHA相同；迁移前可能受重复GPU0任务影响的240文件目录单独保存在`controls_pre_migration_mixed240/`。迁移审计为`outputs/round5/phase1_5_seed3072_legacy/diagnostics/candidate_pool/cube/control_compatibility_audit.json`。迁移后独立核验确认manifest、聚合、两套raw目录、行数和哈希全部匹配。

Cube控制配对分析完成，按10,000次状态聚类bootstrap统计。相对同状态三个anchor的平均参照，64个固定单位RMS高斯控制的成功率差为−0.501（95% CI [−0.624, −0.378]），共同里程碑距离差为+2.262（[+1.626, +2.966]）；物理零动作对应−0.560（[−0.700, −0.420]）和+3.121（[+2.537, +3.730]）。终点成功数据覆盖50/50个状态；共同里程碑距离只覆盖28/50个状态，因此距离结论应按该子集解释。配对分析文件为`outputs/round5/phase1_5_seed3072_legacy/diagnostics/candidate_pool/cube/control_paired_effects.json`，candidate-pool与control汇总也已合并重写到`diagnostics/summary_cube.json`。

Phase1.5仍未收敛：完整Guidance网格为540/1,008，尚缺468项；粗计时文件目前仅有1/2,600项，尚缺2,599项，后续仍需完成细测候选选择和跨任务方法选择。Cube控制迁移不缩减这些原计划工作。

### 5.115 下一批实验的GPU资源门槛

只读检查时GPU0–3空闲显存约18.5/18.1/10.5/22.6 GiB，GPU利用率99–100%；宿主MemAvailable约341 GiB、SwapFree约479 MiB，memory PSI avg10/avg60/avg300均为0。现有Phase1.5启动器默认要求GPU空闲显存至少3,500 MiB、宿主可用内存至少8,192 MiB、SwapFree至少1,024 MiB；当前swap低于启动门槛，且GPU均在满载。GPU1上仍有本仓库Phase3 LEFlow训练进程（PID 1770666），未对其作停止或修改。没有发现Phase1.5实验进程，因此本轮未启动新的GPU任务，也未使用GPU4–7。

根据各任务Guidance sweep manifest，已完成PushT post-opt 216项、PushT guided-flow 36项、Reacher post-opt 216项、Reacher guided-flow 36项及TwoRoom guided-flow 36项，共540/1,008项。剩余468项为Cube post-opt 216项、Cube guided-flow 36项、TwoRoom post-opt 216项。粗计时`timing.json`有1/2,600项；剩余2,599项以及后续隔离细测、方法选择仍按原计划执行。待GPU0–3资源与主机swap恢复到安全准入范围后继续排队，不降低实验范围。

### 5.116 全局条件与实现验证复核

CPU-only `scripts/round5_phase1_5.py analyze --allow-incomplete`核对当前索引：主条件2,424项、固定稳定性96项、自适应稳定性80项，总计2,600/2,600全部完成。该结果只证明闭环主扫描完整，不代表Guidance与计时已完成；`analysis.json`尚无方法选择结果。主计划validator返回`status=ok`，diagnostics validator也返回`status=ok`；内置`unittest`将`tests.test_round5_phase1_5`与`tests.test_round4_protocol`合并运行，70项全部通过。分析报告写到`/tmp/round5_phase1_5_generated_status.md`，没有覆盖主报告里的Cube配对控制结果。

再次检查GPU0–3时，空闲显存仍约18.5/18.1/10.5/22.6 GiB，利用率均为100%；宿主MemAvailable约350 GiB、SwapFree约481 MiB，memory PSI三个窗口为0。GPU1上次观察到的Phase3进程PID 1770666已不在进程表中，但GPU负载仍满；没有发现Phase1.5运行进程。默认1,024 MiB swap启动门槛仍未满足，因此没有启动新GPU任务；GPU4–7继续等待用户明确授权。

### 5.117 14:43 恢复Guidance实验

用户明确要求移除swap门槛并继续实验。GPU0复核为RTX 4090、空闲显存48,508 MiB、利用率0%；宿主MemAvailable约361.8 GiB，128核负载比0.237，memory PSI为0，SwapFree为29 MiB。三个新任务都显式传入`--min-swap-free-mib 0`，其余CPU、可用内存及GPU余量检查保留。首个Cube post-opt变体已启动；同一GPU0进程将依次完成Cube post-opt 216项、Cube guided-flow 36项和TwoRoom post-opt 216项。运行日志写入`diagnostics/guidance_sweep/runlogs/remaining_guidance_20260925.log`。

### 5.118 Guidance续跑进度

GPU0上的Cube post-opt已完成10/216个变体，每个已完成文件均含完整50状态记录；Cube guided-flow与TwoRoom post-opt仍排在当前队列之后。GPU0空闲显存约42.9 GiB；宿主MemAvailable约353.3 GiB、SwapFree约341 MiB，memory PSI三个窗口均为0。实验进程仍在运行，日志显示当前变体持续推进。

### 5.119 五分钟快照

Cube post-opt推进至19/216，已生成文件均为完整50状态且无无效文件；另外两个队列尚未开始。GPU0空闲显存约46.6 GiB，宿主MemAvailable约354.3 GiB、SwapFree约346 MiB，memory PSI三个窗口均为0；进程仍在运行。

### 5.120 Guidance续跑检查

Cube post-opt推进至25/216，全部已落盘文件均有完整50状态记录且无无效文件。GPU0空闲显存约41.7 GiB，利用率2%；宿主MemAvailable约349.4 GiB、SwapFree约347 MiB，memory PSI三个窗口均为0。进程持续运行，另两组Guidance仍按队列顺序等待。

### 5.121 Guidance续跑快照

Cube post-opt推进至31/216，完整50状态文件无无效项；当前正在运行下一变体。GPU0空闲显存约46.6 GiB、利用率0%，宿主MemAvailable约354.5 GiB、SwapFree约291 MiB，memory PSI三个窗口均为0。实验进程持续运行。

### 5.122 Guidance续跑快照

Cube post-opt推进至36/216，已落盘变体全部含完整50状态记录，无无效文件；当前进入候选1、K=5参数组。GPU0空闲显存约39.5 GiB，宿主MemAvailable约327.0 GiB、SwapFree约965 MiB，memory PSI三个窗口均为0；进程仍在运行。

### 5.123 Guidance续跑快照

Cube post-opt推进至43/216，全部已落盘文件均完整且无无效项。GPU0空闲显存约46.5 GiB；宿主MemAvailable约354.6 GiB、SwapFree约950 MiB，memory PSI近期窗口接近0。当前进程仍在执行候选1的K=5参数组。

### 5.124 Guidance续跑快照

Cube post-opt推进至46/216，已落盘记录完整且无无效文件。GPU0空闲显存约46.6 GiB；宿主MemAvailable约356.2 GiB、SwapFree约950 MiB，memory PSI三个窗口均为0。队列继续运行，尚未切换到后续两组。

### 5.125 Guidance续跑快照

Cube post-opt推进至49/216，所有已完成变体均有完整50状态记录且没有无效文件。GPU0空闲显存约46.6 GiB；宿主MemAvailable约359.2 GiB、SwapFree约951 MiB，memory PSI三个窗口均为0。运行进程继续推进。

### 5.126 Guidance续跑快照

Cube post-opt推进至52/216，已落盘文件均完整且无无效项。此前Guidance为540/1,008，计入本轮52项后为592/1,008，尚余416项。GPU0空闲显存约40.9 GiB；宿主MemAvailable约356.0 GiB、SwapFree约951 MiB，memory PSI三个窗口均为0。

### 5.127 Guidance续跑快照

Cube post-opt推进至55/216，已生成的55个变体均为完整50状态记录且无无效文件；本轮Guidance总量为595/1,008，尚余413项。GPU0空闲显存约41.6 GiB、利用率4%；宿主MemAvailable约353.2 GiB、SwapFree约951 MiB，memory PSI三个窗口均为0。

### 5.128 Guidance续跑快照

Cube post-opt推进至58/216，全部已完成文件仍通过50状态完整性检查。当前Guidance总量为598/1,008，尚余410项。GPU0空闲显存约44.8 GiB，宿主MemAvailable约357.5 GiB、SwapFree约952 MiB，memory PSI三个窗口均为0。

### 5.129 Guidance续跑快照

Cube post-opt推进至61/216，所有已完成文件均有效；本轮Guidance总量为601/1,008，尚余407项。GPU0空闲显存约46.6 GiB；宿主MemAvailable约351.2 GiB、SwapFree约952 MiB。memory PSI的avg10为0、avg60/avg300约0.06/0.05，仍处于低水平；实验进程继续运行。

### 5.130 Guidance续跑快照

Cube post-opt推进至64/216，本轮Guidance总量为604/1,008，尚余404项；已完成文件均完整且无无效项。GPU0空闲显存约46.6 GiB；宿主MemAvailable约354.4 GiB、SwapFree约952 MiB，memory PSI三个窗口均回到0，实验进程继续运行。

### 5.131 Guidance续跑快照

Cube post-opt推进至69/216，本轮Guidance总进度为609/1,008，尚余399项；当前进入S=2参数组。所有已落盘变体均完整有效。GPU0空闲显存约46.0 GiB；宿主MemAvailable约351.2 GiB、SwapFree约907 MiB，memory PSI仍接近0。

### 5.132 Guidance续跑快照

Cube post-opt推进至72/216，全部已完成变体完整有效。GPU0空闲显存约46.6 GiB；宿主MemAvailable约349.8 GiB、SwapFree约853 MiB，memory PSI三个窗口均为0。进程继续运行于S=2参数组。

### 5.133 Guidance续跑快照

Cube post-opt推进至75/216，本轮Guidance总量为615/1,008，尚余393项；已落盘结果均完整有效。GPU0空闲显存约41.7 GiB，主机MemAvailable约348.0 GiB、SwapFree约854 MiB，memory PSI三个窗口为0。

### 5.134 Guidance续跑快照

Cube post-opt推进至81/216，本轮Guidance总进度为621/1,008，尚余387项。已完成文件均完整有效；GPU0空闲显存约46.0 GiB，宿主MemAvailable约347.8 GiB、SwapFree约856 MiB，memory PSI三个窗口均为0。

### 5.135 Guidance续跑快照

Cube post-opt推进至91/216，本轮Guidance总进度为631/1,008，尚余377项；已完成文件均完整有效。GPU0空闲显存约40.9 GiB；宿主MemAvailable约344.3 GiB、SwapFree约857 MiB，memory PSI三个窗口均为0。

### 5.136 Guidance续跑快照

Cube post-opt推进至97/216，本轮Guidance总量为637/1,008，尚余371项；所有已落盘变体均完整有效。GPU0空闲显存约41.3 GiB、利用率22%；宿主MemAvailable约339.8 GiB、SwapFree约857 MiB，memory PSI三个窗口均为0。

### 5.137 Guidance续跑快照

Cube post-opt已完成100/216，所有已落盘变体完整有效。本轮Guidance总进度为640/1,008，尚余368项。GPU0空闲显存约46.6 GiB；宿主MemAvailable约350.5 GiB、SwapFree约857 MiB，memory PSI三个窗口均为0。

### 5.138 Guidance续跑快照

Cube post-opt推进至102/216，本轮Guidance总量为642/1,008，尚余366项；产物完整有效。GPU0空闲显存约44.3 GiB；尽管宿主SwapFree已降至0，MemAvailable仍约343.5 GiB且memory PSI三个窗口均为0，任务进程继续运行。

### 5.139 Guidance续跑快照

Cube post-opt推进至105/216，本轮Guidance总量为645/1,008，尚余363项；全部已落盘变体完整有效。SwapFree仍为0，但宿主MemAvailable约345.0 GiB且memory PSI三个窗口为0。GPU0空闲显存约45.4 GiB，进程继续运行。

### 5.140 Guidance续跑快照

Cube post-opt推进至107/216，本轮Guidance总量为647/1,008，尚余361项；已完成文件完整有效。SwapFree为0，宿主MemAvailable仍约344.6 GiB、memory PSI三个窗口均为0；GPU0空闲显存约41.2 GiB，进程继续运行。

### 5.141 Guidance续跑快照

Cube post-opt推进至110/216，本轮Guidance总量为650/1,008，尚余358项；已完成记录均完整有效。当前进入S=5参数组。SwapFree为0，但MemAvailable约342.4 GiB且memory PSI三个窗口为0；GPU0空闲显存约44.3 GiB。

### 5.142 Guidance续跑快照

Cube post-opt推进至116/216，本轮Guidance总量为656/1,008，尚余352项。已落盘变体均完整有效。SwapFree仍为0，MemAvailable约348.0 GiB；memory PSI的avg10/avg60/avg300为0.05/0.06/0.01，处于低水平。GPU0空闲显存约44.3 GiB。

### 5.143 Guidance续跑快照

Cube post-opt推进至121/216，本轮Guidance总量为661/1,008，尚余347项；已落盘文件全部完整有效。SwapFree约1 MiB，但宿主MemAvailable约348.1 GiB，memory PSI三个窗口均为0；GPU0空闲显存约44.3 GiB。

### 5.144 Guidance续跑快照

Cube post-opt推进至123/216，本轮Guidance总量为663/1,008，尚余345项。已完成变体均有效；SwapFree约1 MiB，MemAvailable约339.1 GiB，memory PSI三个窗口均为0；GPU0空闲显存约41.7 GiB。

### 5.145 Guidance续跑快照

Cube post-opt推进至126/216，本轮Guidance总量为666/1,008，尚余342项；已完成变体均完整有效。SwapFree约2 MiB，MemAvailable约349.7 GiB，memory PSI三个窗口均为0；GPU0空闲显存约40.9 GiB，进程继续运行。

### 5.146 Guidance续跑快照

Cube post-opt推进至132/216，本轮Guidance总量为672/1,008，尚余336项；无无效或不完整变体。GPU0空闲显存约44.3 GiB，MemAvailable约348.1 GiB，SwapFree约4 MiB，memory PSI三个窗口均为0。

### 5.147 Guidance续跑快照

Cube post-opt推进至134/216，全部结果文件完整有效；本轮Guidance总量为674/1,008，尚余334项。GPU0空闲显存约44.3 GiB，MemAvailable约349.4 GiB，SwapFree约4 MiB；memory PSI的avg60约0.01，其余窗口为0。

### 5.148 Guidance续跑快照

Cube post-opt推进至140/216，本轮Guidance总进度为680/1,008，尚余328项；已生成文件全部完整有效。GPU0空闲显存约39.6 GiB、利用率32%；宿主MemAvailable约349.2 GiB、SwapFree约5 MiB，memory PSI三个窗口均为0。

### 5.149 Guidance续跑快照

Cube post-opt推进至145/216，本轮Guidance总量为685/1,008，尚余323项；已完成变体均完整有效。GPU0空闲显存约38.3 GiB、利用率37%；宿主MemAvailable约345.4 GiB、SwapFree约5 MiB，memory PSI三个窗口均为0。

### 5.150 Guidance续跑快照

Cube post-opt推进至148/216，本轮Guidance总进度为688/1,008，尚余320项；已完成文件全部完整有效。GPU0空闲显存约46.6 GiB，MemAvailable约348.6 GiB、SwapFree约12 MiB，memory PSI三个窗口均为0。

### 5.151 Guidance续跑快照

Cube post-opt推进至154/216，本轮Guidance总进度为694/1,008，尚余314项；已完成变体均完整有效。GPU0空闲显存约46.6 GiB；宿主MemAvailable约350.8 GiB、SwapFree约15 MiB，memory PSI三个窗口均为0。

### 5.152 Guidance续跑快照

Cube post-opt推进至159/216，本轮Guidance总量为699/1,008，尚余309项。已完成变体均完整有效；GPU0空闲显存约44.3 GiB，MemAvailable约347.4 GiB、SwapFree约15 MiB，memory PSI三个窗口均为0。

### 5.153 Guidance续跑快照

Cube post-opt推进至162/216，本轮Guidance总量为702/1,008，尚余306项；已完成变体全部完整有效。GPU0空闲显存约46.6 GiB，MemAvailable约350.7 GiB、SwapFree约15 MiB，memory PSI三个窗口均为0。

### 5.154 Guidance续跑快照

Cube post-opt推进至165/216，本轮Guidance总量为705/1,008，尚余303项；全部已完成文件完整有效。当前进入S=16参数组。GPU0空闲显存约46.6 GiB，MemAvailable约353.6 GiB、SwapFree约15 MiB，memory PSI三个窗口均为0。

### 5.155 Guidance续跑快照

Cube post-opt推进至171/216，本轮Guidance总量为711/1,008，尚余297项；已完成变体均完整有效。GPU0空闲显存约46.6 GiB，MemAvailable约353.5 GiB、SwapFree约16 MiB，memory PSI三个窗口均为0。

### 5.156 Guidance续跑快照

Cube post-opt推进至176/216，本轮Guidance总进度为716/1,008，尚余292项；已完成变体均完整有效。GPU0空闲显存约44.3 GiB、利用率35%；宿主MemAvailable约353.3 GiB、SwapFree约16 MiB，memory PSI三个窗口均为0。

### 5.157 Guidance续跑快照

Cube post-opt推进至179/216，本轮Guidance总量为719/1,008，尚余289项。已完成文件完整有效；GPU0空闲显存约46.6 GiB，宿主MemAvailable约357.5 GiB、SwapFree约16 MiB，memory PSI三个窗口均为0。

### 5.158 Guidance续跑快照

Cube post-opt推进至182/216，本轮Guidance总量为722/1,008，尚余286项；已完成文件均完整有效。GPU0空闲显存约42.3 GiB、利用率36%；宿主MemAvailable约350.7 GiB、SwapFree约16 MiB，memory PSI三个窗口均为0。

### 5.159 Guidance续跑快照

Cube post-opt推进至185/216，本轮Guidance总量为725/1,008，尚余283项；已完成变体均完整有效。GPU0空闲显存约44.2 GiB；宿主MemAvailable约355.0 GiB、SwapFree约16 MiB，memory PSI三个窗口均为0。

### 5.160 Guidance续跑快照

Cube post-opt推进至189/216，本轮Guidance总量为729/1,008，尚余279项；当前所有已落盘变体均为50条状态记录且`outcome_status`有效。GPU0空闲显存约38.6 GiB、利用率11%；宿主MemAvailable约350.8 GiB、SwapFree约16 MiB，memory PSI三个窗口均为0。

### 5.161 Guidance续跑快照

Cube post-opt推进至191/216，本轮Guidance总量为731/1,008，尚余277项；所有已落盘文件仍均通过50条记录及状态字段完整性检查。GPU0空闲显存约43.1 GiB、利用率45%；宿主MemAvailable约348.0 GiB、SwapFree约17 MiB，memory PSI三个窗口均为0。

### 5.162 Guidance续跑快照

Cube post-opt推进至194/216，本轮Guidance总量为734/1,008，尚余274项；所有已落盘变体的行数和`outcome_status`均有效。GPU0空闲显存约45.3 GiB、利用率34%；宿主MemAvailable约343.5 GiB、SwapFree约17 MiB，memory PSI三个窗口均为0。

### 5.163 Guidance续跑快照

Cube post-opt推进至197/216，本轮Guidance总量为737/1,008，尚余271项；全部已落盘结果均有50条有效状态记录。GPU0空闲显存约44.5 GiB、利用率33%；宿主MemAvailable约343.4 GiB、SwapFree约17 MiB，memory PSI三个窗口均为0。

### 5.164 Guidance续跑快照

Cube post-opt推进至199/216，本轮Guidance总量为739/1,008，尚余269项；所有已落盘文件均通过状态记录完整性检查。GPU0空闲显存约38.9 GiB、利用率39%；宿主MemAvailable约346.8 GiB、SwapFree约17 MiB，memory PSI三个窗口均为0。

### 5.165 Guidance续跑快照

Cube post-opt推进至202/216，本轮Guidance总量为742/1,008，尚余266项；全部已落盘变体完整有效。GPU0空闲显存约43.8 GiB、利用率0%（采样瞬间）；宿主MemAvailable约353.9 GiB、SwapFree约17 MiB，memory PSI三个窗口均为0。

### 5.166 Guidance续跑快照

Cube post-opt推进至205/216，本轮Guidance总量为745/1,008，尚余263项；所有已落盘变体均有50条有效状态记录。GPU0空闲显存约44.2 GiB、利用率27%；宿主MemAvailable约364.8 GiB、SwapFree约50 MiB，memory PSI三个窗口均为0。

### 5.167 Guidance续跑快照

Cube post-opt推进至208/216，本轮Guidance总量为748/1,008，尚余260项；所有已落盘结果均完整有效。GPU0空闲显存约44.3 GiB、利用率25%；宿主MemAvailable约363.4 GiB、SwapFree约50 MiB，memory PSI三个窗口均为0。

### 5.168 Guidance续跑快照

Cube post-opt推进至211/216，本轮Guidance总量为751/1,008，尚余257项；已落盘变体完整性检查无异常。GPU0空闲显存约38.7 GiB、利用率30%；宿主MemAvailable约365.2 GiB、SwapFree约51 MiB，memory PSI三个窗口均为0。

### 5.169 Guidance续跑快照

Cube post-opt推进至214/216，本轮Guidance总量为754/1,008，尚余254项；全部已落盘文件完整有效。GPU0空闲显存约42.9 GiB、利用率34%；宿主MemAvailable约364.5 GiB、SwapFree约52 MiB，memory PSI三个窗口均为0。

### 5.170 Guidance续跑快照

Cube post-opt已完成216/216，Cube guided-flow已开始并完成首个变体；Guidance总量为757/1,008，尚余251项。当前所有已落盘文件均有50条有效状态记录。GPU0空闲显存约40.5 GiB、利用率18%；宿主MemAvailable约359.8 GiB、SwapFree约52 MiB，memory PSI三个窗口均为0。

### 5.171 Guidance续跑快照

Cube post-opt维持216/216；Cube guided-flow推进至4/36，Guidance总量为760/1,008，尚余248项。全部已落盘变体完整有效。GPU0空闲显存约44.2 GiB、利用率30%；宿主MemAvailable约358.4 GiB、SwapFree约137 MiB，memory PSI三个窗口均为0。

### 5.172 Guidance续跑快照

Cube post-opt维持216/216；Cube guided-flow推进至6/36，Guidance总量为762/1,008，尚余246项。所有已落盘文件均完整有效。GPU0空闲显存约41.0 GiB、利用率29%；宿主MemAvailable约354.2 GiB、SwapFree约272 MiB，memory PSI三个窗口均为0。

### 5.173 Guidance续跑快照

Cube post-opt维持216/216；Cube guided-flow推进至9/36，Guidance总量为765/1,008，尚余243项。所有已落盘变体完整有效。GPU0空闲显存约46.6 GiB、利用率0%（采样瞬间）；宿主MemAvailable约360.4 GiB、SwapFree约272 MiB，memory PSI三个窗口均为0。

### 5.174 Guidance续跑快照

Cube post-opt维持216/216；Cube guided-flow推进至12/36，Guidance总量为768/1,008，尚余240项。已落盘结果完整有效。GPU0空闲显存约46.6 GiB、利用率0%（采样瞬间）；宿主MemAvailable约367.4 GiB、SwapFree约272 MiB，memory PSI三个窗口均为0。

### 5.175 Guidance续跑快照

Cube post-opt维持216/216；Cube guided-flow推进至15/36，Guidance总量为771/1,008，尚余237项。所有已落盘变体完整有效。GPU0空闲显存约44.9 GiB、利用率1%；宿主MemAvailable约366.3 GiB、SwapFree约273 MiB，memory PSI三个窗口均为0。

### 5.176 Guidance续跑快照

Cube post-opt维持216/216；Cube guided-flow推进至18/36，Guidance总量为774/1,008，尚余234项。已落盘文件完整有效。GPU0空闲显存约38.1 GiB、利用率29%；宿主MemAvailable约370.4 GiB、SwapFree约273 MiB，memory PSI三个窗口均为0。

### 5.177 Guidance续跑快照

Cube post-opt维持216/216；Cube guided-flow推进至21/36，Guidance总量为777/1,008，尚余231项。全部已落盘变体完整有效。GPU0空闲显存约38.1 GiB、利用率30%；宿主MemAvailable约370.0 GiB、SwapFree约273 MiB，memory PSI三个窗口均为0。

### 5.178 Guidance续跑快照

Cube post-opt维持216/216；Cube guided-flow推进至24/36，Guidance总量为780/1,008，尚余228项。所有已落盘文件完整有效。GPU0空闲显存约38.1 GiB、利用率51%；宿主MemAvailable约369.7 GiB、SwapFree约273 MiB，memory PSI三个窗口均为0。

### 5.179 Guidance续跑快照

Cube post-opt维持216/216；Cube guided-flow推进至27/36，Guidance总量为783/1,008，尚余225项。所有已落盘变体完整有效。GPU0空闲显存约32.5 GiB、利用率69%；宿主MemAvailable约358.4 GiB、SwapFree约273 MiB，memory PSI avg60为0.01，其余采样窗口为0。

### 5.180 Guidance续跑快照

Cube post-opt维持216/216；Cube guided-flow推进至30/36，Guidance总量为786/1,008，尚余222项。全部已落盘变体完整有效。GPU0空闲显存约38.2 GiB、利用率33%；宿主MemAvailable约361.6 GiB、SwapFree约273 MiB，memory PSI三个窗口均为0。

### 5.181 Guidance续跑快照

Cube post-opt维持216/216；Cube guided-flow推进至33/36，Guidance总量为789/1,008，尚余219项。所有已落盘变体完整有效。GPU0空闲显存约38.2 GiB、利用率35%；宿主MemAvailable约361.9 GiB、SwapFree约273 MiB，memory PSI avg60为0.01，其余采样窗口为0。

### 5.182 Guidance续跑快照

Cube post-opt和Cube guided-flow均已完成（216/216、36/36）；TwoRoom post-opt已启动并推进至5/216。Guidance总量为797/1,008，尚余211项；所有已落盘结果完整有效。GPU0空闲显存约38.3 GiB、利用率36%；宿主MemAvailable约372.5 GiB、SwapFree约273 MiB，memory PSI三个窗口均为0。

### 5.183 Guidance续跑快照

Cube post-opt和Cube guided-flow保持完成；TwoRoom post-opt推进至42/216。Guidance总量为834/1,008，尚余174项；全部已落盘文件完整有效。GPU0空闲显存约46.6 GiB、利用率0%（采样瞬间）；宿主MemAvailable约376.4 GiB、SwapFree约273 MiB，memory PSI三个窗口均为0。

### 5.184 Guidance续跑快照

Cube两种Guidance均已完成；TwoRoom post-opt推进至79/216。Guidance总量为871/1,008，尚余137项；已落盘数据完整有效。GPU0空闲显存约38.3 GiB、利用率22%；宿主MemAvailable约371.5 GiB、SwapFree约274 MiB，memory PSI三个窗口均为0。

### 5.185 Guidance续跑快照

Cube两种Guidance均已完成；TwoRoom post-opt推进至117/216。Guidance总量为909/1,008，尚余99项；所有已落盘文件完整有效。GPU0空闲显存约38.3 GiB、利用率21%；宿主MemAvailable约371.1 GiB、SwapFree约274 MiB，memory PSI三个窗口均为0。

### 5.186 Guidance续跑快照

Cube两种Guidance均已完成；TwoRoom post-opt推进至154/216。Guidance总量为946/1,008，尚余62项；所有已落盘变体完整有效。GPU0空闲显存约46.6 GiB、利用率0%（采样瞬间）；宿主MemAvailable约376.5 GiB、SwapFree约274 MiB，memory PSI三个窗口均为0。

### 5.187 Guidance续跑快照

Cube两种Guidance均已完成；TwoRoom post-opt推进至190/216。Guidance总量为982/1,008，尚余26项；所有已落盘变体完整有效。GPU0空闲显存约38.3 GiB、利用率26%；宿主MemAvailable约376.8 GiB、SwapFree约274 MiB，memory PSI三个窗口均为0。

### 5.188 Guidance收尾与粗测启动

Guidance sweep已完成1,008/1,008个变体，完整性核验无异常。公平计时粗测已启动，固定稳定性配置按任务分到GPU0–3；此前直接共用正式输出目录的并发启动触发目录级全局锁，三个冲突进程在写入条件结果前退出。随后将Reacher、Cube、TwoRoom各自改写到独立临时输出目录，PushT继续写正式目录，避免共享锁冲突。当前已完成粗测记录为正式目录18条（Reacher 1、PushT 17）及TwoRoom临时目录1条，其他工作进程已通过GPU与宿主启动检查。SwapFree约278 MiB，但启动阈值为0；宿主MemAvailable约370.3 GiB，memory PSI avg60约0.03。

### 5.189 粗测运行快照

粗测仍在四张允许GPU上运行。正式目录已有20条完整记录（Reacher 1、PushT 19）；临时目录另有Reacher 1、Cube 1、TwoRoom 3条，其中Reacher首项与正式目录既有condition ID相同，合并时保留正式记录。当前独立条件共24/2,600条完整，另有若干正在写入的部分记录。GPU0–3空闲显存约33.8/46.4/29.2/44.0 GiB；宿主MemAvailable约356.5 GiB、SwapFree约278 MiB，memory PSI avg60约0.03。

### 5.190 粗测运行快照

四个粗测进程均持续运行。正式目录当前有22条完整记录（PushT 21、Reacher 1）；临时目录分别有Reacher 3、Cube 2、TwoRoom 4条，其中Reacher有1条与正式目录重复，故独立完成数为30/2,600。未发现当前临时工作进程报错。GPU0–3空闲显存约33.8/46.4/29.2/44.0 GiB；宿主MemAvailable约355.9 GiB、SwapFree约278 MiB，memory PSI avg60和avg300约0.03。

### 5.191 粗测运行快照

正式目录有23条完整记录（PushT 22、Reacher 1）；临时目录有Reacher 4、Cube 3、TwoRoom 5条，其中Reacher与正式目录有1个相同condition ID。当前独立完成数为34/2,600，剩余2,566项；四个粗测进程持续运行，临时工作进程没有错误。GPU0–3空闲显存约33.8/38.2/29.2/35.7 GiB；宿主MemAvailable约346.6 GiB、SwapFree约279 MiB，memory PSI三个窗口均为0。

### 5.192 粗测运行快照

正式目录有24条完整记录（PushT 23、Reacher 1）；临时目录有Reacher 5、Cube 4、TwoRoom 6条，其中Reacher与正式目录重复1条，故独立完成数为38/2,600，剩余2,562项。四个进程仍运行，临时工作进程无报错。GPU0–3空闲显存约43.4/38.2/29.2/16.6 GiB；GPU3上有其他计算进程，当前仍有约16.6 GiB空闲。宿主MemAvailable约349.0 GiB、SwapFree约279 MiB，memory PSI avg60约0.05。

### 5.193 粗测运行快照

正式目录有25条完整记录（PushT 24、Reacher 1）；临时目录有Reacher 6、Cube 5、TwoRoom 7条，其中Reacher重复1条，故独立完成数为42/2,600，剩余2,558项。四个粗测进程继续运行，临时工作进程未报错。GPU0–3空闲显存约32.8/46.4/29.2/44.0 GiB；宿主MemAvailable约354.0 GiB、SwapFree约279 MiB，memory PSI三个窗口均为0。

### 5.194 粗测运行快照

正式目录有26条完整记录（PushT 25、Reacher 1）；临时目录有Reacher 8、Cube 6、TwoRoom 7条，其中Reacher重复1条，故独立完成数为46/2,600，剩余2,554项。四个粗测进程仍运行，临时工作进程无错误。GPU0–3空闲显存约32.8/46.4/29.2/44.0 GiB；宿主MemAvailable约353.8 GiB、SwapFree约279 MiB，memory PSI三个窗口均为0。

### 5.195 粗测运行快照

正式目录有27条完整记录（PushT 26、Reacher 1）；临时目录有Reacher 9、Cube 7、TwoRoom 8条，其中Reacher重复1条，故独立完成数为50/2,600，剩余2,550项。四个粗测进程持续运行，临时工作进程无错误。GPU0–3空闲显存约32.8/46.4/30.5/44.0 GiB；宿主MemAvailable约349.8 GiB、SwapFree约279 MiB，memory PSI三个窗口均为0。

### 5.196 粗测运行快照

正式目录有28条完整记录（PushT 27、Reacher 1）；临时目录有Reacher 10、Cube 8、TwoRoom 8条，其中Reacher重复1条，故独立完成数为53/2,600，剩余2,547项。四个粗测进程继续运行，临时工作进程无错误。GPU0–3空闲显存约32.8/38.2/34.9/44.0 GiB；宿主MemAvailable约350.3 GiB、SwapFree约279 MiB，memory PSI三个窗口均为0。

### 5.197 粗测运行快照

正式目录有29条完整记录（PushT 28、Reacher 1）；临时目录有Reacher 11、Cube 9、TwoRoom 9条，其中Reacher重复1条，故独立完成数为57/2,600，剩余2,543项。四个工作进程仍运行且无新错误。GPU0–3空闲显存约41.0/38.2/34.9/44.0 GiB；宿主MemAvailable约355.7 GiB、SwapFree约279 MiB，memory PSI三个窗口均为0。

### 5.198 粗测运行快照

正式目录有30条完整记录（PushT 29、Reacher 1）；临时目录有Reacher 13、Cube 10、TwoRoom 10条，其中Reacher重复1条，故独立完成数为62/2,600，剩余2,538项。四个工作进程继续运行，临时目录无错误。GPU0–3空闲显存约41.0/46.4/32.4/44.0 GiB；宿主MemAvailable约357.3 GiB、SwapFree约279 MiB，memory PSI三个窗口均为0。

### 5.199 粗测运行快照

正式目录有31条完整记录（PushT 30、Reacher 1）；临时目录有Reacher 14、Cube 11、TwoRoom 11条，其中Reacher重复1条，故独立完成数为66/2,600，剩余2,534项。四个工作进程仍运行，临时目录未见错误。GPU0–3空闲显存约41.0/46.4/32.3/44.0 GiB；宿主MemAvailable约357.2 GiB、SwapFree约280 MiB，memory PSI三个窗口均为0。

### 5.200 粗测运行快照

正式目录有33条完整记录（PushT 32、Reacher 1）；临时目录分别有Reacher 16、Cube 12、TwoRoom 12条，其中Reacher重复1条，故独立完成数为72/2,600，剩余2,528项。四个粗测进程仍在运行，当前未发现OOM或新的条件锁错误。GPU0–3空闲显存约43.4/46.4/29.2/46.4 GiB；宿主MemAvailable约361.7 GiB，memory PSI三个窗口均为0；启动检查已设置为不要求swap余量。

### 5.201 粗测运行快照

正式目录有34条完整记录（PushT 33、Reacher 1）；临时目录分别有Reacher 18、Cube 14、TwoRoom 13条，其中Reacher重复1条，故独立完成数为78/2,600，剩余2,522项。四个粗测进程均仍在推进；日志中可见的条件锁异常来自此前共用正式目录的失败启动。GPU0–3空闲显存约42.1/46.4/34.9/46.4 GiB；宿主MemAvailable约363.9 GiB，memory PSI三个窗口均为0。按最近5分钟增加6项粗略外推，粗测尚需约35小时，耗时会随条件而变化。

### 5.202 粗测运行快照

正式目录有35条完整记录（PushT 34、Reacher 1）；临时目录分别有Reacher 20、Cube 15、TwoRoom 14条，其中Reacher重复1条，故独立完成数为83/2,600，剩余2,517项。四个粗测进程仍有进展。GPU0–3空闲显存约45.1/36.8/29.2/38.1 GiB，均有外部GPU负载；宿主MemAvailable约351.2 GiB，memory PSI三个窗口均为0。粗测按最近5分钟增5项外推约需42小时。Phase1.5 CLI 默认swap启动门槛已从1,024 MiB改为0，当前worker也一直显式设置为0。

### 5.203 粗测运行快照

正式目录有36条完整记录（PushT 35、Reacher 1）；临时目录分别有Reacher 21、Cube 16、TwoRoom 15条，其中Reacher重复1条，故独立完成数为87/2,600，剩余2,513项。四个任务进程均保持活跃。GPU0–3空闲显存约41.0/45.1/29.2/38.2 GiB，存在外部GPU工作负载；宿主MemAvailable约353.9 GiB，memory PSI三个窗口均为0。按最近5分钟增加4项外推，粗测尚需约52小时，实际耗时会随条件变化。

### 5.204 粗测运行快照

正式目录有38条完整记录（PushT 37、Reacher 1）；临时目录分别有Reacher 23、Cube 17、TwoRoom 16条，其中Reacher重复1条，故独立完成数为93/2,600，剩余2,507项。四个粗测会话均保持活跃并持续输出。GPU0–3空闲显存约41.0/35.7/29.2/38.2 GiB，GPU1外部负载较高；宿主MemAvailable约349.1 GiB，memory PSI三个窗口均为0。过去15分钟增加15项，粗测剩余时间按短期均速约42小时估算。

### 5.205 粗测运行快照

正式目录有39条完整记录（PushT 38、Reacher 1）；临时目录分别有Reacher 24、Cube 18、TwoRoom 17条，其中Reacher重复1条，故独立完成数为97/2,600，剩余2,503项。四个worker会话均活跃。GPU0–3空闲显存约33.9/44.0/29.2/45.1 GiB；宿主MemAvailable约352.1 GiB，memory PSI三个窗口均为0。近20分钟增加19项，当前粗略外推尚需约44小时。

### 5.206 粗测运行快照

正式目录有41条完整记录（PushT 40、Reacher 1）；临时目录分别有Reacher 26、Cube 20、TwoRoom 17条，其中Reacher重复1条，故独立完成数为103/2,600，剩余2,497项。四个worker会话都活跃。GPU0–3空闲显存约41.0/38.1/27.7/38.2 GiB，其中GPU2利用率97%；宿主MemAvailable约349.4 GiB，memory PSI三个窗口均为0。近20分钟增加20项，粗测按该均速暂估尚需约42小时。

### 5.207 粗测运行快照

正式目录有42条完整记录（PushT 41、Reacher 1）；临时目录分别有Reacher 27、Cube 21、TwoRoom 18条，其中Reacher重复1条，故独立完成数为107/2,600，剩余2,493项。四个粗测会话均活跃并写入新结果。GPU0–3空闲显存约33.9/38.1/27.7/42.3 GiB，GPU0、2、3利用率较高；宿主MemAvailable约344.7 GiB，memory PSI三个窗口均为0。近25分钟增加24项，粗测剩余时间按此均速估算约43小时。

### 5.208 粗测运行快照

正式目录有43条完整记录（PushT 42、Reacher 1）；临时目录分别有Reacher 29、Cube 22、TwoRoom 19条，其中Reacher重复1条，故独立完成数为112/2,600，剩余2,488项。四个任务均产出新记录。GPU0–3空闲显存约43.3/45.1/27.7/45.1 GiB，GPU2利用率约96%；宿主MemAvailable约350.5 GiB，memory PSI三个窗口均为0。近30分钟增加34项，按短期速度估算粗测尚需约37小时。

### 5.209 粗测运行快照

正式目录有45条完整记录（PushT 44、Reacher 1）；临时目录分别有Reacher 30、Cube 23、TwoRoom 20条，其中Reacher重复1条，故独立完成数为117/2,600，剩余2,483项。四个worker继续推进，TwoRoom短暂无新结果后已完成下一项。GPU0–3空闲显存约41.0/45.1/34.9/45.1 GiB；宿主MemAvailable约359.8 GiB，memory PSI avg60约0.01、avg10与avg300为0。按近35分钟趋势估算，粗测剩余约43小时。

### 5.210 粗测运行快照

正式目录有46条完整记录（PushT 45、Reacher 1）；临时目录分别有Reacher 31、Cube 23、TwoRoom 20条，其中Reacher重复1条，故独立完成数为119/2,600，剩余2,481项。四个粗测会话仍活跃。GPU0–3空闲显存约43.4/35.6/29.2/38.1 GiB；宿主MemAvailable约351.3 GiB，memory PSI三个窗口均为0。按近35分钟完成32项计算，粗测尚需约45小时。

### 5.211 粗测运行快照

正式目录有47条完整记录（PushT 46、Reacher 1）；临时目录分别有Reacher 33、Cube 24、TwoRoom 21条，其中Reacher重复1条，故独立完成数为124/2,600，剩余2,476项。四个worker均继续运行。GPU0–3空闲显存约36.2/35.7/22.1/36.8 GiB，本次粗测峰值模型显存约0.4 GiB；宿主MemAvailable约341.6 GiB，memory PSI三个窗口均为0。按近45分钟速度估算，粗测还需约45小时。

### 5.212 粗测运行快照

正式目录有49条完整记录（PushT 48、Reacher 1）；临时目录分别有Reacher 35、Cube 26、TwoRoom 22条，其中Reacher重复1条，故独立完成数为131/2,600，剩余2,469项。四个worker均有新结果。GPU0–3空闲显存约35.5/44.0/24.4/45.1 GiB，GPU0与GPU2利用率较高；宿主MemAvailable约346.7 GiB，memory PSI三个窗口均为0。近半小时增加28项，按该均速估算粗测尚需约44小时。

### 5.213 粗测运行快照

正式目录有51条完整记录（PushT 50、Reacher 1）；临时目录分别有Reacher 36、Cube 27、TwoRoom 23条，其中Reacher重复1条，故独立完成数为136/2,600，剩余2,464项。四个粗测进程均活跃并新增结果。GPU0–3空闲显存约36.1/45.8/26.3/45.1 GiB；宿主MemAvailable约352.8 GiB。memory PSI avg10为0.46%、avg60为0.19%、avg300为0.04%，压力短暂上升但可用内存充足。按近35分钟增加33项估算，粗测尚需约43小时。

### 5.214 粗测运行快照

正式目录有52条完整记录（PushT 51、Reacher 1）；临时目录分别有Reacher 38、Cube 28、TwoRoom 23条，其中Reacher重复1条，故独立完成数为140/2,600，剩余2,460项。TwoRoom已开始下一条件的batch50计时，临时文件仍在写入，尚不计入完整数。GPU0–3空闲显存约43.3/46.4/22.1/45.1 GiB；宿主MemAvailable约355.6 GiB，memory PSI三个窗口均为0。按近40分钟速度估算，粗测剩余约43小时。

### 5.215 粗测运行快照

正式目录有54条完整记录（PushT 53、Reacher 1）；临时目录分别有Reacher 40、Cube 29、TwoRoom 24条，其中Reacher重复1条，故独立完成数为146/2,600，剩余2,454项。TwoRoom刚完成先前在写条件；四个worker仍活跃。GPU0–3空闲显存约39.2/46.4/22.1/45.1 GiB，GPU2利用率100%；宿主MemAvailable约352.5 GiB，memory PSI三个窗口均为0。

### 5.216 粗测运行快照

正式目录有55条完整记录（PushT 54、Reacher 1）；临时目录分别有Reacher 41、Cube 31、TwoRoom 24条，其中Reacher重复1条，故独立完成数为150/2,600，剩余2,450项。四个worker会话均保持活跃。GPU0–3空闲显存约36.1/46.4/34.9/45.1 GiB；宿主MemAvailable约358.9 GiB，memory PSI三个窗口均为0。

### 5.217 粗测运行快照

正式目录有56条完整记录（PushT 55、Reacher 1）；临时目录分别有Reacher 43、Cube 32、TwoRoom 25条，其中Reacher重复1条，故独立完成数为155/2,600，剩余2,445项。四个任务本轮均有新增结果。GPU0–3空闲显存约36.1/46.4/22.1/45.1 GiB，GPU0与GPU2负载较高；宿主MemAvailable约352.1 GiB，memory PSI三个窗口均为0。近30分钟增加31项，粗测剩余时间按此均速估算约40小时。

### 5.218 粗测运行快照

正式目录有58条完整记录（PushT 57、Reacher 1）；临时目录分别有Reacher 44、Cube 33、TwoRoom 26条，其中Reacher重复1条，故独立完成数为160/2,600，剩余2,440项。四个任务本轮均有新结果。GPU0–3空闲显存约40.4/46.4/29.2/45.1 GiB；宿主MemAvailable约357.4 GiB，memory PSI三个窗口均为0。按近35分钟增加36项估算，粗测尚需约40小时。

### 5.219 粗测运行快照

正式目录有59条完整记录（PushT 58、Reacher 1）；临时目录分别有Reacher 46、Cube 34、TwoRoom 26条，其中Reacher重复1条，故独立完成数为164/2,600，剩余2,436项。PushT、Reacher和Cube本轮有新结果，TwoRoom会话仍活跃但本轮暂未完成新条件。GPU0–3空闲显存约36.1/46.4/22.0/45.1 GiB；宿主MemAvailable约350.4 GiB，memory PSI三个窗口均为0。按近40分钟增加40项估算，粗测尚需约41小时。

### 5.220 粗测运行快照

正式目录有60条完整记录（PushT 59、Reacher 1）；临时目录分别有Reacher 47、Cube 35、TwoRoom 27条，其中Reacher重复1条，故独立完成数为168/2,600，剩余2,432项。四个任务本轮均有新增结果。GPU0–3空闲显存约40.1/42.3/29.2/41.1 GiB；宿主MemAvailable约354.5 GiB，memory PSI三个窗口均为0。按近50分钟增加49项估算，粗测剩余约41小时。

### 5.221 粗测运行快照

正式目录有61条完整记录（PushT 60、Reacher 1）；临时目录分别有Reacher 49、Cube 36、TwoRoom 28条，其中Reacher重复1条，故独立完成数为173/2,600，剩余2,427项。四个任务本轮均完成新条件。GPU0–3空闲显存约43.3/42.3/22.0/45.1 GiB，GPU2利用率约95%；宿主MemAvailable约354.0 GiB，memory PSI三个窗口均为0。按近50分钟趋势估算，粗测剩余约41小时。

### 5.222 粗测运行快照

正式目录有63条完整记录（PushT 62、Reacher 1）；临时目录分别有Reacher 51、Cube 37、TwoRoom 29条，其中Reacher重复1条，故独立完成数为179/2,600，剩余2,421项。四个任务本轮都有新完成项。GPU0–3空闲显存约46.3/46.4/29.1/45.1 GiB；宿主MemAvailable约360.5 GiB，memory PSI三个窗口均为0。按最近一小时增加55项估算，粗测剩余约44小时。

### 5.223 粗测运行快照

正式目录有65条完整记录（PushT 64、Reacher 1）；临时目录分别有Reacher 52、Cube 39、TwoRoom 29条，其中Reacher重复1条，故独立完成数为184/2,600，剩余2,416项。TwoRoom已开始下一条件的batch1计时，临时项仍在执行；其他任务本轮均继续推进。GPU0–3空闲显存约43.3/46.4/29.1/45.1 GiB；宿主MemAvailable约360.9 GiB，memory PSI三个窗口均为0。按近65分钟增加65项估算，粗测剩余约40小时。

### 5.224 粗测运行快照

正式目录有66条完整记录（PushT 65、Reacher 1）；临时目录分别有Reacher 54、Cube 41、TwoRoom 30条，其中Reacher重复1条，故独立完成数为190/2,600，剩余2,410项。四个任务本轮均有新结果。GPU0–3空闲显存约43.3/46.0/29.1/41.7 GiB；宿主MemAvailable约358.3 GiB，memory PSI三个窗口均为0。按近55分钟增59项估算，粗测剩余约38小时。

### 5.225 粗测运行快照

正式目录有67条完整记录（PushT 66、Reacher 1）；临时目录分别有Reacher 56、Cube 43、TwoRoom 31条，其中Reacher重复1条，故独立完成数为196/2,600，剩余2,404项。四个任务均有新增结果。GPU0–3空闲显存约45.1/46.4/34.7/45.1 GiB；宿主MemAvailable约363.6 GiB，memory PSI三个窗口均为0。按过去一小时增加60项估算，粗测剩余约40小时。

### 5.226 粗测运行快照

正式目录有69条完整记录（PushT 68、Reacher 1）；临时目录分别有Reacher 57、Cube 44、TwoRoom 32条，其中Reacher重复1条，故独立完成数为201/2,600，剩余2,399项。四个任务本轮均有新增结果。GPU0–3空闲显存约43.3/46.4/29.6/45.1 GiB；宿主MemAvailable约359.8 GiB，memory PSI三个窗口均为0。按近65分钟均速估算，粗测剩余约40小时。

### 5.227 粗测运行快照

正式目录有70条完整记录（PushT 69、Reacher 1）；临时目录分别有Reacher 58、Cube 45、TwoRoom 32条，其中Reacher重复1条，故独立完成数为204/2,600，剩余2,396项。PushT、Reacher、Cube本轮有新结果；TwoRoom会话保持活跃但本轮未完成新项。GPU0–3空闲显存约46.3/46.4/29.1/45.1 GiB；宿主MemAvailable约359.6 GiB，memory PSI三个窗口均为0。整体近一小时均速约58项。

### 5.228 粗测运行快照

正式目录有72条完整记录（PushT 71、Reacher 1）；临时目录分别有Reacher 60、Cube 46、TwoRoom 33条，其中Reacher重复1条，故独立完成数为210/2,600，剩余2,390项。四个粗测会话均活跃。GPU0–3空闲显存约40.4/46.4/29.1/41.7 GiB；宿主MemAvailable约354.4 GiB，memory PSI三个窗口均为0。按近15分钟增加14项估算，粗测尚需约42小时。

### 5.229 粗测运行快照

正式目录有74条完整记录（PushT 73、Reacher 1）；临时目录分别有Reacher 62、Cube 48、TwoRoom 34条，其中Reacher重复1条，故独立完成数为217/2,600，剩余2,383项。四个任务本轮都有新记录。GPU0–3空闲显存约43.3/46.4/29.1/45.1 GiB；宿主MemAvailable约360.0 GiB，memory PSI三个窗口均为0。按过去约一小时的均速估算，粗测剩余约35小时。

### 5.230 粗测运行快照

正式目录有75条完整记录（PushT 74、Reacher 1）；临时目录分别有Reacher 64、Cube 50、TwoRoom 35条，其中Reacher重复1条，故独立完成数为223/2,600，剩余2,377项。四个任务本轮都有新增结果。GPU0–3空闲显存约35.0/46.4/34.8/45.1 GiB；宿主MemAvailable约355.5 GiB，memory PSI三个窗口均为0。按最近一小时约63项估算，粗测剩余约38小时。

### 5.231 粗测运行快照

正式目录有77条完整记录（PushT 76、Reacher 1）；临时目录分别有Reacher 66、Cube 51、TwoRoom 35条，其中Reacher重复1条，故独立完成数为228/2,600，剩余2,372项。PushT、Reacher、Cube本轮有新结果；TwoRoom会话仍活跃但本轮暂未提交新项。GPU0–3空闲显存约35.0/46.4/29.1/45.1 GiB；宿主MemAvailable约356.1 GiB，memory PSI三个窗口均为0。按近一小时均速估算，粗测剩余约37小时。

### 5.232 粗测运行快照

正式目录有78条完整记录（PushT 77、Reacher 1）；临时目录分别有Reacher 67、Cube 53、TwoRoom 36条，其中Reacher重复1条，故独立完成数为233/2,600，剩余2,367项。四个任务本轮均推进。GPU0–3空闲显存约35.0/46.4/34.8/45.1 GiB；宿主MemAvailable约356.2 GiB，memory PSI三个窗口均为0。按近65分钟增加65项估算，粗测剩余约40小时。

### 5.233 粗测运行快照

正式目录有80条完整记录（PushT 79、Reacher 1）；临时目录分别有Reacher 69、Cube 54、TwoRoom 38条，其中Reacher重复1条，故独立完成数为240/2,600，剩余2,360项。四个任务本轮均有进展。GPU0–3空闲显存约43.3/44.9/29.1/45.1 GiB；宿主MemAvailable约358.6 GiB，memory PSI三个窗口均为0。按近一小时增加72项估算，粗测剩余约33小时。

### 5.234 粗测运行快照

正式目录有81条完整记录（PushT 80、Reacher 1）；临时目录分别有Reacher 70、Cube 55、TwoRoom 39条，其中Reacher重复1条，故独立完成数为244/2,600，剩余2,356项。四个任务本轮均有新结果。GPU0–3空闲显存约43.3/46.4/32.9/45.1 GiB；宿主MemAvailable约359.4 GiB，memory PSI三个窗口均为0。按近75分钟趋势估算，粗测剩余约37小时。

### 5.235 粗测运行快照

正式目录有82条完整记录（PushT 81、Reacher 1）；临时目录分别有Reacher 72、Cube 57、TwoRoom 39条，其中Reacher重复1条，故独立完成数为249/2,600，剩余2,351项。PushT、Reacher和Cube本轮有新结果。GPU0–3空闲显存约32.1/46.4/34.8/45.1 GiB，GPU0利用率约96%；宿主MemAvailable约352.9 GiB，memory PSI三个窗口均为0。按近80分钟趋势估算，粗测剩余约37小时。

### 5.236 粗测运行快照

正式目录有84条完整记录（PushT 83、Reacher 1）；临时目录分别有Reacher 74、Cube 58、TwoRoom 40条，其中Reacher重复1条，故独立完成数为255/2,600，剩余2,345项。四个任务本轮均有新结果。GPU0–3空闲显存约32.4/46.4/29.1/45.1 GiB，GPU0利用率约92%；宿主MemAvailable约353.1 GiB，memory PSI三个窗口均为0。粗测剩余时间暂估约35小时。

### 5.237 粗测运行快照

正式目录有85条完整记录（PushT 84、Reacher 1）；临时目录分别有Reacher 75、Cube 60、TwoRoom 41条，其中Reacher重复1条，故独立完成数为260/2,600，剩余2,340项。四个任务本轮均有新结果。GPU0–3空闲显存约35.0/46.4/29.1/46.4 GiB；宿主MemAvailable约359.9 GiB，memory PSI三个窗口均为0。近期进度速度上升，粗测剩余时间暂估约30小时。

### 5.238 粗测运行快照

正式目录有87条完整记录（PushT 86、Reacher 1）；临时目录分别有Reacher 77、Cube 61、TwoRoom 42条，其中Reacher重复1条，故独立完成数为266/2,600，剩余2,334项。四个任务本轮均更新记录。GPU0–3空闲显存约43.3/46.4/33.1/46.4 GiB；宿主MemAvailable约360.8 GiB，memory PSI三个窗口均为0。按近70分钟均速约70项/小时估算，粗测剩余约33小时。

### 5.239 粗测运行快照

正式目录有88条完整记录（PushT 87、Reacher 1）；临时目录分别有Reacher 78、Cube 62、TwoRoom 43条，其中Reacher重复1条，故独立完成数为270/2,600，剩余2,330项。四个任务本轮均有新结果。GPU0–3空闲显存约39.2/46.4/29.1/46.4 GiB；宿主MemAvailable约359.5 GiB，memory PSI三个窗口均为0。按近70分钟增加74项估算，粗测剩余约37小时。

### 5.240 粗测运行快照

正式目录有90条完整记录（PushT 89、Reacher 1）；临时目录分别有Reacher 80、Cube 64、TwoRoom 44条，其中Reacher重复1条，故独立完成数为277/2,600，剩余2,323项。四个任务本轮均有新条件完成。GPU0–3空闲显存约39.2/43.4/29.1/46.4 GiB；宿主MemAvailable约355.1 GiB，memory PSI三个窗口均为0。按近75分钟增加76项估算，粗测剩余约38小时。

### 5.241 粗测运行快照

正式目录有91条完整记录（PushT 90、Reacher 1）；临时目录分别有Reacher 81、Cube 65、TwoRoom 46条，其中Reacher重复1条，故独立完成数为282/2,600，剩余2,318项。四个任务本轮均有新结果。GPU0–3空闲显存约43.3/46.4/29.1/46.4 GiB；宿主MemAvailable约361.7 GiB，memory PSI三个窗口均为0。按最近一小时增加78项估算，粗测尚需约30小时。

### 5.242 粗测运行快照

正式目录有93条完整记录（PushT 92、Reacher 1）；临时目录分别有Reacher 83、Cube 67、TwoRoom 47条，其中Reacher重复1条，故独立完成数为289/2,600，剩余2,311项。四个任务本轮均有新结果。GPU0–3空闲显存约39.2/46.4/29.1/46.4 GiB；宿主MemAvailable约358.3 GiB，memory PSI三个窗口均为0。按近55分钟增加66项估算，粗测剩余约32小时。

### 5.243 粗测运行快照

正式目录有94条完整记录（PushT 93、Reacher 1）；临时目录分别有Reacher 85、Cube 69、TwoRoom 48条，其中Reacher重复1条，故独立完成数为295/2,600，剩余2,305项。四个任务本轮均有新结果。GPU0–3空闲显存约42.3/38.0/29.1/43.4 GiB；宿主MemAvailable约354.2 GiB，memory PSI三个窗口均为0。按近55分钟增加72项估算，粗测剩余约30小时。

### 5.244 粗测运行快照

正式目录有96条完整记录（PushT 95、Reacher 1）；临时目录分别有Reacher 86、Cube 70、TwoRoom 49条，其中Reacher重复1条，故独立完成数为300/2,600，剩余2,300项。四个任务本轮均有新结果。GPU0–3空闲显存约39.2/46.4/31.9/46.4 GiB；宿主MemAvailable约358.6 GiB，memory PSI三个窗口均为0。粗测完成约11.5%，按近期均速估算尚需约32小时。

### 5.245 粗测运行快照

正式目录有97条完整记录（PushT 96、Reacher 1）；临时目录分别有Reacher 88、Cube 71、TwoRoom 50条，其中Reacher重复1条，故独立完成数为305/2,600，剩余2,295项。四个任务本轮均新增完整条件。GPU0–3空闲显存约39.2/46.4/29.1/46.4 GiB；宿主MemAvailable约359.4 GiB，memory PSI三个窗口均为0。按近55分钟增加61项估算，粗测剩余约35小时。

### 5.246 粗测运行快照

正式目录有99条完整记录（PushT 98、Reacher 1）；临时目录分别有Reacher 90、Cube 73、TwoRoom 51条，其中Reacher重复1条，故独立完成数为312/2,600，剩余2,288项。四个任务本轮均有新结果。GPU0–3空闲显存约39.2/38.0/29.1/46.4 GiB；宿主MemAvailable约353.9 GiB，memory PSI三个窗口均为0。粗测剩余时间暂估约35小时。

### 5.247 粗测运行快照

正式目录有100条完整记录（PushT 99、Reacher 1）；临时目录分别有Reacher 91、Cube 75、TwoRoom 52条，其中Reacher重复1条，故独立完成数为317/2,600，剩余2,283项。四个任务本轮均有新结果。GPU0–3空闲显存约39.2/46.4/29.1/46.4 GiB；宿主MemAvailable约359.7 GiB，memory PSI三个窗口均为0。按近65分钟趋势估算粗测剩余约34小时。

### 5.248 粗测运行快照

正式目录有102条完整记录（PushT 101、Reacher 1）；临时目录分别有Reacher 93、Cube 77、TwoRoom 53条，其中Reacher重复1条，故独立完成数为324/2,600，剩余2,276项。四个任务本轮均有新结果。GPU0–3空闲显存约35.2/46.4/31.6/46.4 GiB；宿主MemAvailable约356.9 GiB，memory PSI三个窗口均为0。按近65分钟增加64项估算，粗测剩余约39小时。

### 5.249 粗测运行快照

正式目录有104条完整记录（PushT 103、Reacher 1）；临时目录分别有Reacher 95、Cube 79、TwoRoom 54条，其中Reacher重复1条，故独立完成数为331/2,600，剩余2,269项。TwoRoom先前较慢的条件已完成，目前已开始下一项；四个任务会话均活跃。GPU0–3空闲显存约39.3/38.0/29.1/46.4 GiB；宿主MemAvailable约356.8 GiB，memory PSI三个窗口均为0。按近期均速估算，粗测剩余约36小时。

### 5.250 粗测运行快照

正式目录有105条完整记录（PushT 104、Reacher 1）；临时目录分别有Reacher 97、Cube 80、TwoRoom 54条，其中Reacher重复1条，故独立完成数为335/2,600，剩余2,265项。PushT、Reacher和Cube本轮有新结果；TwoRoom会话保持活跃。GPU0–3空闲显存约39.2/46.4/29.1/38.0 GiB；宿主MemAvailable约355.7 GiB，memory PSI三个窗口均为0。按近一小时速度估算，粗测剩余约38小时。

### 5.251 粗测运行快照

正式目录有107条完整记录（PushT 106、Reacher 1）；临时目录分别有Reacher 99、Cube 82、TwoRoom 55条，其中Reacher重复1条，故独立完成数为342/2,600，剩余2,258项。四个任务本轮均有新结果。GPU0–3空闲显存约43.3/46.4/29.1/46.4 GiB；宿主MemAvailable约362.1 GiB，memory PSI三个窗口均为0。按近35分钟增42项估算，粗测剩余约31小时。

### 5.252 粗测运行快照

正式目录有108条完整记录（PushT 107、Reacher 1）；临时目录分别有Reacher 100、Cube 84、TwoRoom 56条，其中Reacher重复1条，故独立完成数为347/2,600，剩余2,253项。四个任务本轮均有新增结果。GPU0–3空闲显存约39.2/43.4/34.8/43.4 GiB；宿主MemAvailable约351.2 GiB，memory PSI三个窗口均为0。按近70分钟趋势估算粗测剩余约30小时。

### 5.253 粗测运行快照

正式目录有110条完整记录（PushT 109、Reacher 1）；临时目录分别有Reacher 102、Cube 85、TwoRoom 57条，其中Reacher重复1条，故独立完成数为353/2,600，剩余2,247项。四个任务本轮均有新结果。GPU0–3空闲显存约39.2/43.5/29.1/43.5 GiB；宿主MemAvailable约353.7 GiB，memory PSI三个窗口均为0。按近55分钟速度估算粗测剩余约39小时。

### 5.254 粗测运行快照

正式目录有112条完整记录（PushT 111、Reacher 1）；临时目录分别有Reacher 104、Cube 87、TwoRoom 59条，其中Reacher重复1条，故独立完成数为361/2,600，剩余2,239项。四个任务本轮均有新增结果。GPU0–3空闲显存约43.3/46.4/29.1/46.4 GiB；宿主MemAvailable约365.5 GiB，memory PSI三个窗口均为0。按近65分钟增加79项估算，粗测剩余约31小时。

### 5.255 粗测运行快照

正式目录有114条完整记录（PushT 113、Reacher 1）；临时目录分别有Reacher 105、Cube 88、TwoRoom 60条，其中Reacher重复1条，故独立完成数为366/2,600，剩余2,234项。四个任务本轮均有新结果。GPU0–3空闲显存约35.2/38.0/29.1/38.0 GiB；宿主MemAvailable约349.7 GiB，memory PSI三个窗口均为0。按近70分钟增加66项估算，粗测剩余约40小时。

### 5.256 粗测运行快照

正式目录有115条完整记录（PushT 114、Reacher 1）；临时目录分别有Reacher 107、Cube 89、TwoRoom 61条，其中Reacher重复1条，故独立完成数为371/2,600，剩余2,229项。四个任务本轮均有新结果。GPU0–3空闲显存约42.9/38.0/29.1/38.0 GiB；宿主MemAvailable约353.6 GiB，memory PSI三个窗口均为0。按近50分钟增加59项估算，粗测剩余约32小时。

### 5.257 粗测运行快照

正式目录有117条完整记录（PushT 116、Reacher 1）；临时目录分别有Reacher 109、Cube 91、TwoRoom 62条，其中Reacher重复1条，故独立完成数为378/2,600，剩余2,222项。四个任务本轮均完成新条件。GPU0–3空闲显存约43.3/38.0/29.1/38.1 GiB；宿主MemAvailable约354.4 GiB，memory PSI三个窗口均为0。按近55分钟均速估算粗测剩余约31小时。

### 5.258 粗测运行快照

正式目录有118条完整记录（PushT 117、Reacher 1）；临时目录分别有Reacher 110、Cube 92、TwoRoom 63条，其中Reacher重复1条，故独立完成数为382/2,600，剩余2,218项。四个任务本轮均完成新条件。GPU0–3空闲显存约39.2/38.1/29.1/38.1 GiB；宿主MemAvailable约349.6 GiB，memory PSI三个窗口均为0。按最近一小时均速估算粗测剩余约39小时。

### 5.259 粗测运行快照

正式目录有120条完整记录（PushT 119、Reacher 1）；临时目录分别有Reacher 112、Cube 94、TwoRoom 64条，其中Reacher重复1条，故独立完成数为389/2,600，剩余2,211项。四个任务本轮均有新结果。GPU0–3空闲显存约43.3/38.1/32.2/38.1 GiB；宿主MemAvailable约351.6 GiB，memory PSI三个窗口均为0。按近期约72项/小时估算，粗测剩余约31小时。

### 5.260 粗测运行快照

正式目录有122条完整记录（PushT 121、Reacher 1）；临时目录分别有Reacher 114、Cube 96、TwoRoom 64条，其中Reacher重复1条，故独立完成数为395/2,600，剩余2,205项。PushT、Reacher和Cube本轮均有新结果；TwoRoom会话仍活跃。GPU0–3空闲显存约46.3/46.4/29.1/46.4 GiB；宿主MemAvailable约363.8 GiB，memory PSI三个窗口均为0。按近50分钟增加60项估算，粗测剩余约30小时。

### 5.261 粗测运行快照

正式目录有123条完整记录（PushT 122、Reacher 1）；临时目录分别有Reacher 115、Cube 97、TwoRoom 65条，其中Reacher重复1条，故独立完成数为399/2,600，剩余2,201项。四个任务本轮均完成新条件。GPU0–3空闲显存约43.3/46.4/29.1/46.4 GiB；宿主MemAvailable约360.7 GiB，memory PSI三个窗口均为0。近期速度约70项/小时，粗测剩余约31小时。

### 5.262 粗测运行快照

截至 2026-09-26 12:54（北京时间），正式目录有 235 条完整记录（PushT 234、Reacher 1）；临时目录有 Reacher 238、Cube 210、TwoRoom 147 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 829/2,600（31.9%），剩余 1,771 项。四个 worker 的输出和活动锁均有近期更新，未见退出或 OOM。GPU0–3 空闲显存约 43.3/46.3/30.5/44.0 GiB；宿主 MemAvailable 约 367.0 GiB，memory PSI 三个窗口均为 0。自上次快照约 6 小时新增 430 项，按约 71 项/小时估算，粗测还需约 25 小时。

### 5.263 粗测运行快照

截至 2026-09-26 13:01（北京时间），正式目录有 236 条完整记录（PushT 235、Reacher 1）；临时目录有 Reacher 240、Cube 212、TwoRoom 148 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 835/2,600（32.1%），剩余 1,765 项。四个 worker 在本轮均有新结果和活动锁更新。GPU0–3 空闲显存约 27.7/38.0/29.1/44.0 GiB；宿主 MemAvailable 约 345.6 GiB，memory PSI 三个窗口均为 0。近期速度约 72 项/小时，粗测还需约 24.5 小时。

### 5.264 粗测运行快照

截至 2026-09-26 13:07（北京时间），正式目录有 238 条完整记录（PushT 237、Reacher 1）；临时目录有 Reacher 241、Cube 213、TwoRoom 149 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 840/2,600（32.3%），剩余 1,760 项。四个 worker 会话仍存活，所有目录的结果和活动锁均在本轮更新。GPU0–3 空闲显存约 46.3/46.3/34.7/44.0 GiB；宿主 MemAvailable 约 357.5 GiB，memory PSI 三个窗口均为 0。近期速度约 72 项/小时，粗测还需约 24.5 小时。

### 5.265 粗测运行快照

截至 2026-09-26 13:13（北京时间），正式目录有 240 条完整记录（PushT 239、Reacher 1）；临时目录有 Reacher 243、Cube 215、TwoRoom 150 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 847/2,600（32.6%），剩余 1,753 项。四个 worker 会话仍存活，结果文件和活动锁均在更新。GPU0–3 空闲显存约 43.3/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 357.6 GiB，memory PSI 三个窗口均为 0。按近 6 小时约 71 项/小时的速度估算，粗测还需约 25 小时。

### 5.266 粗测运行快照

截至 2026-09-26 13:19（北京时间），正式目录有 241 条完整记录（PushT 240、Reacher 1）；临时目录有 Reacher 245、Cube 216、TwoRoom 152 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 853/2,600（32.8%），剩余 1,747 项。四个 worker 会话仍存活，所有条件目录均在本轮更新。GPU0–3 空闲显存约 43.3/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 357.5 GiB，memory PSI 三个窗口均为 0。按近期约 71 项/小时估算，粗测还需约 24.5 小时。

### 5.267 粗测运行快照

截至 2026-09-26 13:26（北京时间），正式目录有 244 条完整记录（PushT 243、Reacher 1）；临时目录有 Reacher 248、Cube 218、TwoRoom 155 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 864/2,600（33.2%），剩余 1,736 项。四个 worker 会话仍存活，各任务最近数分钟均有输出更新。GPU0–3 空闲显存约 35.9/46.3/34.8/44.0 GiB；宿主 MemAvailable 约 357.5 GiB，memory PSI 三个窗口均为 0。按近 6.5 小时约 71 项/小时估算，粗测还需约 24.5 小时。

### 5.268 粗测运行快照

截至 2026-09-26 13:31（北京时间），正式目录有 245 条完整记录（PushT 244、Reacher 1）；临时目录有 Reacher 249、Cube 219、TwoRoom 156 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 868/2,600（33.4%），剩余 1,732 项。四个 worker 会话仍存活，四个结果目录和活动锁均在最近几分钟更新。GPU0–3 空闲显存约 32.3/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 350.2 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 24.5 小时。

### 5.269 粗测运行快照

截至 2026-09-26 13:37（北京时间），正式目录有 246 条完整记录（PushT 245、Reacher 1）；临时目录有 Reacher 250、Cube 220、TwoRoom 157 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 872/2,600（33.5%），剩余 1,728 项。四个 worker 会话仍存活，所有任务最近数分钟均有新结果。GPU0–3 空闲显存约 35.9/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 353.2 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 24.5 小时。

### 5.270 粗测运行快照

截至 2026-09-26 13:44（北京时间），正式目录有 248 条完整记录（PushT 247、Reacher 1）；临时目录有 Reacher 252、Cube 222、TwoRoom 158 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 879/2,600（33.8%），剩余 1,721 项。四个 worker 会话仍存活；本轮 Cube 有新提交，其他目录的最近完整记录在数分钟前。GPU0–3 空闲显存约 24.0/38.1/29.1/44.0 GiB；宿主 MemAvailable 约 342.8 GiB，memory PSI 三个窗口均为 0。整体速度约 70 项/小时，粗测还需约 24.5 小时。

### 5.271 粗测运行快照

截至 2026-09-26 13:50（北京时间），正式目录有 249 条完整记录（PushT 248、Reacher 1）；临时目录有 Reacher 254、Cube 223、TwoRoom 160 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 885/2,600（34.0%），剩余 1,715 项。四个 worker 会话仍存活，四个任务都在最近几分钟提交了结果。GPU0–3 空闲显存约 36.0/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 353.2 GiB。memory PSI 短窗口约 0–0.01%，300 秒窗口 some/full 分别约 0.09%/0.06%，当前处于低水平。按整体约 70 项/小时估算，粗测还需约 24.5 小时。

### 5.272 粗测运行快照

截至 2026-09-26 13:56（北京时间），正式目录有 251 条完整记录（PushT 250、Reacher 1）；临时目录有 Reacher 256、Cube 224、TwoRoom 161 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 891/2,600（34.3%），剩余 1,709 项。四个 worker 会话仍存活，所有任务最近几分钟均有新结果。GPU0–3 空闲显存约 43.3/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 361.1 GiB。memory PSI 300 秒窗口 some/full 约 0.02%/0.01%，处于低水平。按约 71 项/小时估算，粗测还需约 24 小时。

### 5.273 粗测运行快照

截至 2026-09-26 14:02（北京时间），正式目录有 253 条完整记录（PushT 252、Reacher 1）；临时目录有 Reacher 259、Cube 226、TwoRoom 163 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 900/2,600（34.6%），剩余 1,700 项。四个 worker 会话仍存活，四个任务本轮均有新结果。GPU0–3 空闲显存约 43.3/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 360.8 GiB。memory PSI 的 10/60 秒窗口短暂升至约 0.02%/0.04%，300 秒窗口为 0%，仍处于低水平。按整体约 70 项/小时估算，粗测还需约 24 小时。

### 5.274 粗测运行快照

截至 2026-09-26 14:08（北京时间），正式目录有 254 条完整记录（PushT 253、Reacher 1）；临时目录有 Reacher 261、Cube 227、TwoRoom 164 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 905/2,600（34.8%），剩余 1,695 项。四个 worker 会话仍存活，最近几分钟各目录均有新提交。GPU0–3 空闲显存约 38.9/46.3/32.0/44.0 GiB；宿主 MemAvailable 约 356.6 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 24 小时。

### 5.275 粗测运行快照

截至 2026-09-26 14:15（北京时间），正式目录有 257 条完整记录（PushT 256、Reacher 1）；临时目录有 Reacher 263、Cube 229、TwoRoom 166 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 914/2,600（35.2%），剩余 1,686 项。四个 worker 会话仍存活；之前耗时较长的 TwoRoom 条件已完成并提交，其他任务也持续推进。GPU0–3 空闲显存约 38.9/46.3/32.0/44.0 GiB；宿主 MemAvailable 约 356.6 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 24 小时。

### 5.276 粗测运行快照

截至 2026-09-26 14:21（北京时间），正式目录有 259 条完整记录（PushT 258、Reacher 1）；临时目录有 Reacher 265、Cube 231、TwoRoom 167 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 921/2,600（35.4%），剩余 1,679 项。四个 worker 会话仍存活，各任务最近几分钟均有提交。GPU0–3 空闲显存约 43.3/46.3/29.1/40.5 GiB；宿主 MemAvailable 约 358.4 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 24 小时。

### 5.277 粗测运行快照

截至 2026-09-26 14:27（北京时间），正式目录有 261 条完整记录（PushT 260、Reacher 1）；临时目录有 Reacher 266、Cube 232、TwoRoom 169 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 927/2,600（35.7%），剩余 1,673 项。四个 worker 会话仍存活，四个任务最近几分钟均有提交。GPU0–3 空闲显存约 32.3/42.7/29.1/44.0 GiB；宿主 MemAvailable 约 351.9 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 24 小时。

### 5.278 粗测运行快照

截至 2026-09-26 14:33（北京时间），正式目录有 263 条完整记录（PushT 262、Reacher 1）；临时目录有 Reacher 269、Cube 234、TwoRoom 170 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 935/2,600（36.0%），剩余 1,665 项。四个 worker 会话仍存活，所有任务最近几分钟均有输出。GPU0–3 空闲显存约 46.3/46.3/30.0/44.0 GiB；宿主 MemAvailable 约 360.8 GiB，memory PSI 三个窗口均为 0。GPU2 利用率较高但显存余量充足。按整体约 71 项/小时估算，粗测还需约 23.5 小时。

### 5.279 粗测运行快照

截至 2026-09-26 14:38（北京时间），正式目录有 266 条完整记录（PushT 265、Reacher 1）；临时目录有 Reacher 270、Cube 235、TwoRoom 172 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 942/2,600（36.2%），剩余 1,658 项。四个 worker 会话仍存活，四个任务最近几分钟均有输出。GPU0–3 空闲显存约 43.3/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 362.2 GiB，memory PSI 三个窗口均为 0。按整体约 71 项/小时估算，粗测还需约 23.5 小时。

### 5.280 粗测运行快照

截至 2026-09-26 14:44（北京时间），正式目录有 268 条完整记录（PushT 267、Reacher 1）；临时目录有 Reacher 272、Cube 237、TwoRoom 174 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 950/2,600（36.5%），剩余 1,650 项。四个 worker 会话仍存活，四个任务目录均在本轮更新。GPU0–3 空闲显存约 36.0/46.3/33.0/44.0 GiB；宿主 MemAvailable 约 358.9 GiB，memory PSI 三个窗口均为 0。按整体约 71 项/小时估算，粗测还需约 23 小时。

### 5.281 粗测运行快照

截至 2026-09-26 14:50（北京时间），正式目录有 270 条完整记录（PushT 269、Reacher 1）；临时目录有 Reacher 274、Cube 238、TwoRoom 177 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 958/2,600（36.8%），剩余 1,642 项。四个 worker 会话仍存活，所有输出目录均在本轮更新。GPU0–3 空闲显存约 43.3/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 364.5 GiB，memory PSI 三个窗口均为 0。按整体约 71 项/小时估算，粗测还需约 23 小时。

### 5.282 粗测运行快照

截至 2026-09-26 14:56（北京时间），正式目录有 272 条完整记录（PushT 271、Reacher 1）；临时目录有 Reacher 276、Cube 240、TwoRoom 180 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 967/2,600（37.2%），剩余 1,633 项。四个 worker 会话仍存活，四个任务目录均在本轮更新。GPU0–3 空闲显存约 43.3/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 363.3 GiB，memory PSI 三个窗口均为 0。按整体约 71 项/小时估算，粗测还需约 23 小时。

### 5.283 粗测运行快照

截至 2026-09-26 15:02（北京时间），正式目录有 273 条完整记录（PushT 272、Reacher 1）；临时目录有 Reacher 278、Cube 241、TwoRoom 182 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 973/2,600（37.4%），剩余 1,627 项。四个 worker 会话仍存活，四个任务最近几分钟均有提交。GPU0–3 空闲显存约 39.5/42.2/34.7/44.0 GiB；宿主 MemAvailable 约 360.9 GiB，memory PSI 三个窗口均为 0。GPU2 利用率较高但 VRAM 余量充足。按整体约 71 项/小时估算，粗测还需约 23 小时。

### 5.284 粗测运行快照

截至 2026-09-26 15:08（北京时间），正式目录有 275 条完整记录（PushT 274、Reacher 1）；临时目录有 Reacher 280、Cube 243、TwoRoom 183 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 980/2,600（37.7%），剩余 1,620 项。四个 worker 会话仍存活，所有任务目录最近几分钟均有结果更新。GPU0–3 空闲显存约 42.5/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 363.6 GiB，memory PSI 三个窗口均为 0。按整体约 71 项/小时估算，粗测还需约 22.8 小时。

### 5.285 粗测运行快照

截至 2026-09-26 15:14（北京时间），正式目录有 277 条完整记录（PushT 276、Reacher 1）；临时目录有 Reacher 282、Cube 245、TwoRoom 185 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 988/2,600（38.0%），剩余 1,612 项。四个 worker 会话仍存活，所有任务最近几分钟均有输出更新。GPU0–3 空闲显存约 35.9/46.3/34.8/44.0 GiB；宿主 MemAvailable 约 361.7 GiB，memory PSI 三个窗口均为 0。按整体约 71 项/小时估算，粗测还需约 22.7 小时。

### 5.286 粗测运行快照

截至 2026-09-26 15:20（北京时间），正式目录有 278 条完整记录（PushT 277、Reacher 1）；临时目录有 Reacher 284、Cube 246、TwoRoom 186 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 993/2,600（38.2%），剩余 1,607 项。四个 worker 会话仍存活，四个任务最近几分钟均有输出。GPU0–3 空闲显存约 35.4/42.2/29.1/44.0 GiB；宿主 MemAvailable 约 356.3 GiB，memory PSI 三个窗口均为 0。按整体约 71 项/小时估算，粗测还需约 22.5 小时。

### 5.287 粗测运行快照

截至 2026-09-26 15:28（北京时间），正式目录有 281 条完整记录（PushT 280、Reacher 1）；临时目录有 Reacher 287、Cube 248、TwoRoom 188 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,003/2,600（38.6%），剩余 1,597 项。四个 Python worker 进程均在运行，所有任务目录最近数分钟有结果更新。GPU0–3 空闲显存约 43.3/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 363.9 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 22.7 小时。

### 5.288 粗测运行快照

截至 2026-09-26 15:35（北京时间），正式目录有 283 条完整记录（PushT 282、Reacher 1）；临时目录有 Reacher 289、Cube 249、TwoRoom 190 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,010/2,600（38.8%），剩余 1,590 项。四个 Python worker 进程均在运行，所有任务最近几分钟均有输出更新。GPU0–3 空闲显存约 43.3/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 361.5 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 22.7 小时。

### 5.289 粗测运行快照

截至 2026-09-26 15:41（北京时间），正式目录有 286 条完整记录（PushT 285、Reacher 1）；临时目录有 Reacher 293、Cube 251、TwoRoom 192 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,021/2,600（39.3%），剩余 1,579 项。四个 worker 进程仍运行，四个任务最近数分钟均有结果提交。GPU0–3 空闲显存约 46.3/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 363.2 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 22.4 小时。

### 5.290 粗测运行快照

截至 2026-09-26 15:47（北京时间），正式目录有 288 条完整记录（PushT 287、Reacher 1）；临时目录有 Reacher 295、Cube 253、TwoRoom 193 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,028/2,600（39.5%），剩余 1,572 项。四个 worker 进程仍运行，所有任务目录均在最近几分钟更新。GPU0–3 空闲显存约 35.6/43.2/34.8/44.0 GiB；宿主 MemAvailable 约 358.6 GiB，memory PSI 三个窗口均为 0。按整体约 71 项/小时估算，粗测还需约 22.3 小时。

### 5.291 粗测运行快照

截至 2026-09-26 15:53（北京时间），正式目录有 289 条完整记录（PushT 288、Reacher 1）；临时目录有 Reacher 297、Cube 254、TwoRoom 194 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,033/2,600（39.7%），剩余 1,567 项。四个 worker 进程仍运行，所有任务最近几分钟均有写入。GPU0–3 空闲显存约 39.2/38.1/29.1/44.0 GiB；宿主 MemAvailable 约 356.1 GiB。memory PSI 的 60/300 秒窗口约 0.02%/0.01%，水平很低。按整体约 70 项/小时估算，粗测还需约 22.4 小时。

### 5.292 粗测运行快照

截至 2026-09-26 15:59（北京时间），正式目录有 290 条完整记录（PushT 289、Reacher 1）；临时目录有 Reacher 299、Cube 256、TwoRoom 195 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,039/2,600（40.0%），剩余 1,561 项。四个 worker 进程仍运行，所有任务最近几分钟均有新结果。GPU0–3 空闲显存约 43.3/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 362.2 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 22 小时。

### 5.293 粗测运行快照

截至 2026-09-26 16:05（北京时间），正式目录有 292 条完整记录（PushT 291、Reacher 1）；临时目录有 Reacher 301、Cube 258、TwoRoom 196 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,046/2,600（40.2%），剩余 1,554 项。四个 worker 进程仍运行，所有任务最近几分钟均有结果更新。GPU0–3 空闲显存约 35.2/38.9/29.1/44.0 GiB；宿主 MemAvailable 约 353.9 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 22 小时。

### 5.294 粗测运行快照

截至 2026-09-26 16:11（北京时间），正式目录有 294 条完整记录（PushT 293、Reacher 1）；临时目录有 Reacher 302、Cube 259、TwoRoom 197 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,051/2,600（40.4%），剩余 1,549 项。四个 worker 进程仍运行，所有任务最近几分钟均有输出。GPU0–3 空闲显存约 35.1/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 357.4 GiB。memory PSI 的 60/300 秒窗口约 0.33%/0.24%，虽略有升高但仍很低。按整体约 70 项/小时估算，粗测还需约 22 小时。

### 5.295 粗测运行快照

截至 2026-09-26 16:17（北京时间），正式目录有 296 条完整记录（PushT 295、Reacher 1）；临时目录有 Reacher 304、Cube 261、TwoRoom 198 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,058/2,600（40.7%），剩余 1,542 项。四个 worker 进程仍运行，任务结果持续提交。GPU0–3 空闲显存约 43.3/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 362.9 GiB。memory PSI 的 300 秒窗口约 0.10%，仍处于低水平。按整体约 70 项/小时估算，粗测还需约 22 小时。

### 5.296 粗测运行快照

截至 2026-09-26 16:23（北京时间），正式目录有 297 条完整记录（PushT 296、Reacher 1）；临时目录有 Reacher 306、Cube 263、TwoRoom 199 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,064/2,600（40.9%），剩余 1,536 项。四个 worker 进程仍运行，所有任务目录最近几分钟均有结果更新。GPU0–3 空闲显存约 43.3/46.3/32.8/44.0 GiB；宿主 MemAvailable 约 361.5 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 22 小时。

### 5.297 粗测运行快照

截至 2026-09-26 16:29（北京时间），正式目录有 299 条完整记录（PushT 298、Reacher 1）；临时目录有 Reacher 308、Cube 264、TwoRoom 200 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,070/2,600（41.2%），剩余 1,530 项。四个 worker 进程仍运行；PushT、Reacher、Cube 近期有提交，TwoRoom 最近完整条件约 4 分钟前完成且进程仍活跃。GPU0–3 空闲显存约 35.2/42.6/29.1/44.0 GiB；宿主 MemAvailable 约 355.2 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 21.8 小时。

### 5.298 粗测运行快照

截至 2026-09-26 16:35（北京时间），正式目录有 300 条完整记录（PushT 299、Reacher 1）；临时目录有 Reacher 310、Cube 266、TwoRoom 201 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,076/2,600（41.4%），剩余 1,524 项。四个 worker 进程仍运行，PushT、Reacher、Cube 近期有提交；TwoRoom 最近提交约 3.5 分钟前且进程仍活跃。GPU0–3 空闲显存约 43.3/46.3/31.3/44.0 GiB；宿主 MemAvailable 约 362.7 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 21.7 小时。

### 5.299 粗测运行快照

截至 2026-09-26 16:41（北京时间），正式目录有 302 条完整记录（PushT 301、Reacher 1）；临时目录有 Reacher 312、Cube 267、TwoRoom 202 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,082/2,600（41.6%），剩余 1,518 项。四个 worker 进程仍运行，所有任务近期有结果。GPU0–3 空闲显存约 43.3/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 362.8 GiB。GPU2 利用率较高但显存余量充足；memory PSI 的 300 秒窗口约 0.06–0.07%，仍为低水平。按整体约 70 项/小时估算，粗测还需约 21.7 小时。

### 5.300 粗测运行快照

截至 2026-09-26 16:47（北京时间），正式目录有 304 条完整记录（PushT 303、Reacher 1）；临时目录有 Reacher 314、Cube 269、TwoRoom 204 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,090/2,600（41.9%），剩余 1,510 项。四个 worker 进程仍运行，所有任务最近数分钟均有提交。GPU0–3 空闲显存约 45.9/45.8/34.8/44.0 GiB；宿主 MemAvailable 约 362.7 GiB。memory PSI 短窗口约 0.01%，仍很低。按整体约 70 项/小时估算，粗测还需约 21.6 小时。

### 5.301 粗测运行快照

截至 2026-09-26 16:53（北京时间），正式目录有 305 条完整记录（PushT 304、Reacher 1）；临时目录有 Reacher 316、Cube 270、TwoRoom 205 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,095/2,600（42.1%），剩余 1,505 项。四个 worker 进程仍运行，所有任务最近几分钟均有新结果。GPU0–3 空闲显存约 43.3/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 363.1 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 21.5 小时。

### 5.302 粗测运行快照

截至 2026-09-26 16:59（北京时间），正式目录有 307 条完整记录（PushT 306、Reacher 1）；临时目录有 Reacher 318、Cube 272、TwoRoom 206 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,102/2,600（42.4%），剩余 1,498 项。四个 worker 进程仍运行，所有任务最近几分钟均有新记录。GPU0–3 空闲显存约 35.1/46.3/34.8/44.0 GiB；宿主 MemAvailable 约 357.6 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 21.4 小时。

### 5.303 粗测运行快照

截至 2026-09-26 17:05（北京时间），正式目录有 309 条完整记录（PushT 308、Reacher 1）；临时目录有 Reacher 321、Cube 273、TwoRoom 208 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,110/2,600（42.7%），剩余 1,490 项。四个 worker 进程仍运行，四个任务最近几分钟均有新结果。GPU0–3 空闲显存约 39.4/42.4/29.1/44.0 GiB；宿主 MemAvailable 约 358.4 GiB。memory PSI 的 60/300 秒窗口约 0.08%/0.07%，仍处于低水平。按整体约 70 项/小时估算，粗测还需约 21.3 小时。

### 5.304 粗测运行快照

截至 2026-09-26 17:11（北京时间），正式目录有 311 条完整记录（PushT 310、Reacher 1）；临时目录有 Reacher 322、Cube 274、TwoRoom 209 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,115/2,600（42.9%），剩余 1,485 项。四个 worker 进程仍运行，所有任务目录最近几分钟均有提交。GPU0–3 空闲显存约 38.1/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 357.5 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 21.2 小时。

### 5.305 粗测运行快照

截至 2026-09-26 17:18（北京时间），正式目录有 313 条完整记录（PushT 312、Reacher 1）；临时目录有 Reacher 325、Cube 276、TwoRoom 210 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,123/2,600（43.2%），剩余 1,477 项。四个 worker 进程仍运行；PushT、Reacher、Cube 最近几分钟有提交，TwoRoom 最近完整条件约 4.5 分钟前完成且进程仍活跃。GPU0–3 空闲显存约 34.7/46.0/29.1/44.0 GiB；宿主 MemAvailable 约 357.2 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 21 小时。

### 5.306 粗测运行快照

截至 2026-09-26 17:24（北京时间），正式目录有 315 条完整记录（PushT 314、Reacher 1）；临时目录有 Reacher 327、Cube 277、TwoRoom 212 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,130/2,600（43.5%），剩余 1,470 项。四个 worker 进程仍运行，所有任务最近几分钟均有新结果。GPU0–3 空闲显存约 46.3/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 362.8 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 21 小时。

### 5.307 粗测运行快照

截至 2026-09-26 17:29（北京时间），正式目录有 317 条完整记录（PushT 316、Reacher 1）；临时目录有 Reacher 329、Cube 279、TwoRoom 213 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,137/2,600（43.7%），剩余 1,463 项。四个 worker 进程仍运行，四个任务最近几分钟均有新结果。GPU0–3 空闲显存约 43.3/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 363.0 GiB，memory PSI 的 60 秒窗口约 0.01%，其余窗口为 0。按整体约 70 项/小时估算，粗测还需约 20.9 小时。

### 5.308 粗测运行快照

截至 2026-09-26 17:35（北京时间），正式目录有 319 条完整记录（PushT 318、Reacher 1）；临时目录有 Reacher 331、Cube 281、TwoRoom 214 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,144/2,600（44.0%），剩余 1,456 项。四个 worker 进程仍运行，最近几分钟均有输出提交。GPU0–3 空闲显存约 37.4/46.3/34.8/44.0 GiB；宿主 MemAvailable 约 361.7 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 20.8 小时。

### 5.309 粗测运行快照

截至 2026-09-26 17:41（北京时间），正式目录有 321 条完整记录（PushT 320、Reacher 1）；临时目录有 Reacher 333、Cube 282、TwoRoom 215 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,150/2,600（44.2%），剩余 1,450 项。四个 worker 进程仍运行，四个任务在最近五分钟内均有提交。GPU0–3 空闲显存约 35.2/46.3/29.1/44.0 GiB；宿主 MemAvailable 约 358.5 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 20.7 小时。

### 5.310 粗测运行快照

截至 2026-09-26 17:47（北京时间），正式目录有 323 条完整记录（PushT 322、Reacher 1）；临时目录有 Reacher 335、Cube 284、TwoRoom 217 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,158/2,600（44.5%），剩余 1,442 项。四个 worker 进程仍运行，所有任务最近几分钟均有结果提交。GPU0–3 空闲显存约 35.2/46.3/32.2/44.0 GiB；宿主 MemAvailable 约 357.3 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 20.6 小时。

### 5.311 粗测运行快照

截至 2026-09-26 17:53（北京时间），正式目录有 325 条完整记录（PushT 324、Reacher 1）；临时目录有 Reacher 337、Cube 286、TwoRoom 218 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,165/2,600（44.8%），剩余 1,435 项。四个 worker 进程仍运行，四个任务最近几分钟均有结果。GPU0–3 空闲显存约 42.6/46.3/34.8/46.3 GiB；宿主 MemAvailable 约 366.5 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 20.5 小时。

### 5.312 粗测运行快照

截至 2026-09-26 17:59（北京时间），正式目录有 328 条完整记录（PushT 327、Reacher 1）；临时目录有 Reacher 340、Cube 288、TwoRoom 220 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,175/2,600（45.2%），剩余 1,425 项。四个 worker 进程仍运行，所有任务最近几分钟均有新结果。GPU0–3 空闲显存约 39.6/46.3/31.1/46.3 GiB；宿主 MemAvailable 约 364.0 GiB。memory PSI 的 60/300 秒窗口升至约 1.29%/0.42%，但可用内存仍充足且任务正常推进，后续继续观察。按整体约 70 项/小时估算，粗测还需约 20.4 小时。

### 5.313 粗测运行快照

截至 2026-09-26 18:05（北京时间），正式目录有 330 条完整记录（PushT 329、Reacher 1）；临时目录有 Reacher 342、Cube 290、TwoRoom 222 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,183/2,600（45.5%），剩余 1,417 项。四个 worker 进程仍运行，结果持续更新。GPU0–3 空闲显存约 34.1/46.3/34.8/46.3 GiB；宿主 MemAvailable 约 360.0 GiB。此前出现的 memory PSI 升高已回落，当前 300 秒窗口约 0.08%。按整体约 70 项/小时估算，粗测还需约 20.2 小时。

### 5.314 粗测运行快照

截至 2026-09-26 18:12（北京时间），正式目录有 332 条完整记录（PushT 331、Reacher 1）；临时目录有 Reacher 344、Cube 291、TwoRoom 224 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,190/2,600（45.8%），剩余 1,410 项。四个 worker 进程仍运行，四个任务最近几分钟均有输出。GPU0–3 空闲显存约 31.1/46.3/29.1/46.3 GiB；宿主 MemAvailable 约 358.6 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 20.1 小时。

### 5.315 粗测运行快照

截至 2026-09-26 18:18（北京时间），正式目录有 334 条完整记录（PushT 333、Reacher 1）；临时目录有 Reacher 346、Cube 293、TwoRoom 226 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,198/2,600（46.1%），剩余 1,402 项。四个 worker 进程仍运行，所有任务最近数分钟均有输出。GPU0–3 空闲显存约 35.1/46.3/29.1/46.3 GiB；宿主 MemAvailable 约 359.4 GiB，memory PSI 短窗口约 0.01%。按整体约 70 项/小时估算，粗测还需约 20 小时。

### 5.316 粗测运行快照

截至 2026-09-26 18:24（北京时间），正式目录有 336 条完整记录（PushT 335、Reacher 1）；临时目录有 Reacher 349、Cube 295、TwoRoom 227 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,206/2,600（46.4%），剩余 1,394 项。四个 worker 进程仍运行，四个任务最近几分钟均有提交。GPU0–3 空闲显存约 35.1/46.3/34.7/46.3 GiB；宿主 MemAvailable 约 359.6 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 19.9 小时。

### 5.317 粗测运行快照

截至 2026-09-26 18:30（北京时间），正式目录有 338 条完整记录（PushT 337、Reacher 1）；临时目录有 Reacher 351、Cube 297、TwoRoom 229 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,214/2,600（46.7%），剩余 1,386 项。四个 worker 进程仍运行，所有任务最近几分钟均有提交。GPU0–3 空闲显存约 39.1/46.3/34.7/46.3 GiB；宿主 MemAvailable 约 360.2 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 19.8 小时。

### 5.318 粗测运行快照

截至 2026-09-26 18:36（北京时间），正式目录有 341 条完整记录（PushT 340、Reacher 1）；临时目录有 Reacher 353、Cube 300、TwoRoom 231 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,224/2,600（47.1%），剩余 1,376 项。四个 worker 进程仍运行，所有任务最近几分钟都有结果更新。GPU0–3 空闲显存约 43.3/46.3/31.2/46.3 GiB；宿主 MemAvailable 约 365.6 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 19.7 小时。

### 5.319 粗测运行快照

截至 2026-09-26 18:42（北京时间），正式目录有 343 条完整记录（PushT 342、Reacher 1）；临时目录有 Reacher 355、Cube 302、TwoRoom 236 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,235/2,600（47.5%），剩余 1,365 项。四个 worker 进程仍运行，本轮四个任务均有新提交。GPU0–3 空闲显存约 43.3/46.3/29.1/46.3 GiB；宿主 MemAvailable 约 365.2 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 19.5 小时。

### 5.320 粗测运行快照

截至 2026-09-26 18:48（北京时间），正式目录有 346 条完整记录（PushT 345、Reacher 1）；临时目录有 Reacher 357、Cube 304、TwoRoom 238 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,244/2,600（47.8%），剩余 1,356 项。四个 worker 进程仍运行，各任务均在最近几分钟有结果提交。GPU0–3 空闲显存约 46.3/46.3/34.8/46.3 GiB；宿主 MemAvailable 约 365.7 GiB，memory PSI 三个窗口均为 0。按整体约 70 项/小时估算，粗测还需约 19.4 小时。

### 5.321 粗测运行快照

截至 2026-09-26 18:54（北京时间），正式目录有 347 条完整记录（PushT 346、Reacher 1）；临时目录有 Reacher 358、Cube 306、TwoRoom 239 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,249/2,600（48.0%），剩余 1,351 项。四个 worker 进程均仍运行，四个任务最近几分钟有新结果。GPU上有其他计算进程共享显存；GPU0–3 空闲显存约 19.9/23.0/9.4/26.7 GiB，GPU2 当前余量最低但 worker 仍在推进。宿主 MemAvailable 约 305.6 GiB；memory PSI 的 60/300 秒窗口为 0.57%/0.83%。按整体约 71 项/小时估算，粗测还需约 19 小时。

### 5.322 粗测运行快照

截至 2026-09-26 19:01（北京时间），正式目录有 348 条完整记录（PushT 347、Reacher 1）；临时目录有 Reacher 359、Cube 307、TwoRoom 241 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,254/2,600（48.2%），剩余 1,346 项。四个 worker 进程仍运行，四个任务最近几分钟均有新结果。GPU上有共享计算负载，GPU0–3 空闲显存约 23.5/26.6/9.4/26.7 GiB，GPU2 余量最低。宿主 MemAvailable 约 304.0 GiB；memory PSI 的 60/300 秒窗口约 0.11%/0.55%。粗测仍在推进；后续精测按计划需隔离 GPU。按整体约 70 项/小时估算，粗测还需约 19.2 小时。

### 5.323 粗测运行快照

截至 2026-09-26 19:07（北京时间），正式目录有 349 条完整记录（PushT 348、Reacher 1）；临时目录有 Reacher 360、Cube 308、TwoRoom 242 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,258/2,600（48.4%），剩余 1,342 项。四个 worker 进程仍运行，本轮四个任务均有结果提交。GPU共享负载较高，GPU0–3 空闲显存约 23.5/26.6/9.4/26.7 GiB，其中 GPU2 余量最低。宿主 MemAvailable 约 313.1 GiB；memory PSI 的 60/300 秒窗口约 0.03%/0.28%。当前任务仍在推进，按整体约 70 项/小时估算，粗测还需约 19.2 小时。

### 5.324 粗测运行快照

截至 2026-09-26 19:13（北京时间），正式目录有 350 条完整记录（PushT 349、Reacher 1）；临时目录有 Reacher 361、Cube 309、TwoRoom 243 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,262/2,600（48.5%），剩余 1,338 项。四个 worker 进程仍运行且最近几分钟均有提交。GPU共享负载仍高，GPU0–3 空闲显存约 23.5/23.0/9.4/26.7 GiB。宿主 MemAvailable 约 312.4 GiB；memory PSI 的 10/60/300 秒窗口约 2.33%/0.96%/0.29%，300 秒压力仍低，当前未见 worker 退出或 OOM。按整体约 70 项/小时估算，粗测还需约 19 小时。

### 5.325 粗测运行快照

截至 2026-09-26 19:19（北京时间），正式目录有 351 条完整记录（PushT 350、Reacher 1）；临时目录有 Reacher 363、Cube 310、TwoRoom 244 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,267/2,600（48.7%），剩余 1,333 项。四个 worker 进程仍运行且持续有结果提交。GPU0–3 空闲显存约 19.0/22.1/9.4/25.7 GiB；宿主 MemAvailable 约 277.9 GiB。memory PSI 的 60/300 秒窗口约 0.69%/0.68%，压力有所增加但仍有充足可用内存，目前未见 OOM。按整体约 70 项/小时估算，粗测还需约 19 小时。

### 5.326 粗测运行快照

截至 2026-09-26 19:26（北京时间），正式目录有 352 条完整记录（PushT 351、Reacher 1）；临时目录有 Reacher 364、Cube 311、TwoRoom 245 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,271/2,600（48.9%），剩余 1,329 项。四个 worker 进程仍运行并有新结果。GPU共享负载持续较高，GPU0–3 空闲显存约 19.5/25.7/12.3/25.7 GiB。宿主 MemAvailable 约 282.7 GiB；memory PSI 的 60/300 秒窗口约 0.78%/0.85%。当前进度正常，持续观察资源压力。按整体约 70 项/小时估算，粗测还需约 19 小时。

### 5.327 粗测运行快照

截至 2026-09-26 19:33（北京时间），正式目录有 352 条完整记录（PushT 351、Reacher 1）；临时目录有 Reacher 364、Cube 311、TwoRoom 246 条，其中 Reacher 与正式目录重复 1 条，故独立完成数为 1,272/2,600（48.9%），剩余 1,328 项。四个 worker 进程均活跃；当前四个条件的 batch1 已完成 10 次，batch50 仍在运行，因此暂未计入完整记录。GPU0–3 空闲显存约 13.3/25.7/9.4/25.7 GiB，GPU 利用率较高。宿主 MemAvailable 约 282.7 GiB；memory PSI 的 10/60/300 秒窗口约 2.70%/1.23%/0.85%。最近半小时增加 23 项，对应约 45 项/小时；按此估算粗测还需约 30 小时，但最近十余分钟速度更慢，ETA 不稳定。

### 5.328 粗测运行快照与筛选评审

截至 2026-09-26 19:48（北京时间），正式目录有 353 条完整记录（PushT 352、Reacher 1）；临时目录有 Reacher 366、Cube 313、TwoRoom 247 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,278/2,600（49.2%），剩余 1,322 项。四个 worker 会话均未报告退出，Reacher 和 Cube 在最近日志轮询中仍有输出。GPU0–3 空闲显存约 10.4/26.3/9.4/26.4 GiB，利用率约 16%/39%/91%/3%；GPU2 余量最低。宿主 MemAvailable 约 273 GiB，memory PSI 的 10/60/300 秒窗口约 0.37%/0.50%/0.16%。SwapFree 几乎为零，但 worker 的 swap 门槛已设为 0，实验继续推进且未见 OOM。近 15 分钟约增加 6 个独立条件，按此估算剩余时间约 55 小时；短时吞吐受共享 GPU 负载影响较大，ETA 仅作粗略参考。

Sol High 只读评审确认粗测可继续作为筛选数据，无需整体重跑；但 `select-fine-timing` 当前未把粗测窗口的 GPU 干扰状态纳入 Pareto 筛选。粗测窗口中 1,655 个标记为 `interference_flagged`，903 个仅表示边界采样未观察到外部计算，后者也不能证明整段窗口完全隔离。进入精测候选选择前，应保留固定锚点，记录两个粗测批次的干扰状态，并同时比较观察到的 frontier 与未标记 batch1 的 frontier；对受干扰但可能影响 frontier 的候选安排隔离复测。最终加速比应基于候选和 P1/P2 基线的隔离精测。后续方法选择器已要求精测窗口状态满足边界无外部计算观测，但该状态的证据范围仍限于边界采样。

### 5.329 粗测运行快照

截至 2026-09-26 19:56（北京时间），正式目录有 354 条完整记录（PushT 353、Reacher 1）；临时目录有 Reacher 368、Cube 314、TwoRoom 248 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,283/2,600（49.3%），剩余 1,317 项。四个 worker 会话均未退出，四个任务最近轮询均有完成条件或新运行日志。GPU0–3 空闲显存约 12.0/25.7/15.1/25.7 GiB，利用率约 90%/87%/92%/74%，共享计算负载仍高。宿主 MemAvailable 约 279 GiB，memory PSI 的 10/60/300 秒窗口约 0.37%/0.84%/0.30%；SwapFree 仍几乎为零，但 swap 门槛为 0，未见 OOM。最近 8 分钟增加 5 项，按该短时速度估算剩余约 35 小时；吞吐仍会随共享 GPU 负载变化。

### 5.330 粗测运行快照

截至 2026-09-26 20:03（北京时间），正式目录有 354 条完整记录（PushT 353、Reacher 1）；临时目录有 Reacher 369、Cube 315、TwoRoom 248 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,285/2,600（49.4%），剩余 1,315 项。四个 worker 会话均未报告退出，Reacher 最近完成了新条件，其他任务仍在运行。GPU0–3 空闲显存约 13.2/25.3/15.1/25.7 GiB，利用率约 0%/80%/98%/0%；GPU2 仍承受较高负载。宿主 MemAvailable 约 266 GiB，memory PSI 的 10/60/300 秒窗口约 0.12%/0.05%/0.21%；SwapFree 几乎为零，swap 门槛为 0，未见 OOM。最近约 8 分钟新增 2 项，但此前短窗吞吐也明显波动，剩余时间估算暂取约 45–80 小时，需随共享负载变化修正。

### 5.331 粗测运行快照

截至 2026-09-26 20:10（北京时间），独立完整记录仍为 1,285/2,600，剩余 1,315 项；本轮约 6 分钟没有新增完整条件。四个 worker 会话仍打开，没有退出或 OOM 迹象。GPU0–3 空闲显存约 10.2/25.7/9.4/25.7 GiB，利用率约 41%/80%/97%/98%，多个设备处于高负载。宿主 MemAvailable 约 268 GiB，memory PSI 的 10/60/300 秒窗口均很低，300 秒约 0.03%；SwapFree 接近零但不再作为运行门槛。由于本轮无新增完成记录，ETA 暂不按这个短间隔外推，保留上一轮 45–80 小时的宽范围并等待后续状态。

### 5.332 粗测运行快照

截至 2026-09-26 20:16（北京时间），正式目录有 355 条完整记录（PushT 354、Reacher 1）；临时目录有 Reacher 370、Cube 316、TwoRoom 249 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,289/2,600（49.6%），剩余 1,311 项。四个 worker 会话仍打开，计数较上轮增加 4 项。GPU0–3 空闲显存约 10.2/25.7/9.4/25.7 GiB，利用率约 55%/84%/100%/2%；GPU2 计算负载满载但仍有约 9.4 GiB 显存余量。宿主 MemAvailable 约 272 GiB，memory PSI 的 10/60/300 秒窗口约 0.00%/0.05%/0.30%；SwapFree 接近零但不作为门槛，未见 OOM。近 28 分钟增加 11 项，按此估算剩余约 56 小时；短时速度有停顿和突发，ETA 仍不稳定。

### 5.333 粗测运行快照

截至 2026-09-26 20:22（北京时间），正式目录有 355 条完整记录（PushT 354、Reacher 1）；临时目录有 Reacher 371、Cube 317、TwoRoom 249 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,291/2,600（49.7%），剩余 1,309 项。四个 worker 会话仍打开，本轮比上一轮增加 2 项。GPU0–3 空闲显存约 10.2/25.7/9.4/25.7 GiB，利用率约 88%/83%/39%/21%；共享负载仍有波动。宿主 MemAvailable 约 274 GiB，memory PSI 的 10/60/300 秒窗口约 0.00%/0.06%/0.32%；SwapFree 约 1.1 MiB但阈值为 0，未见 OOM。最近 19 分钟增加 6 项，按该窗口估算剩余约 69 小时；结合前一阶段停顿和短时突发，当前 ETA 仍以 45–80 小时范围描述。

### 5.334 粗测运行快照

截至 2026-09-26 20:28（北京时间），正式目录有 356 条完整记录（PushT 355、Reacher 1）；临时目录有 Reacher 372、Cube 318、TwoRoom 250 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,295/2,600（49.8%），剩余 1,305 项。四个 worker 会话仍打开，本轮比上一轮增加 4 项。GPU0–3 空闲显存约 10.2/25.7/9.4/25.7 GiB，利用率约 0%/41%/46%/63%；GPU利用率较前几轮回落。宿主 MemAvailable 约 267 GiB，memory PSI 的 10/60/300 秒窗口均低，300 秒约 0.06%；SwapFree 约 1.5 MiB，门槛为 0，未见 OOM。近 25 分钟增加 10 项，按此估算剩余约 54 小时；受共享负载和间歇停顿影响，ETA 继续按粗略值看待。

### 5.335 粗测运行快照

截至 2026-09-26 20:34（北京时间），正式目录有 356 条完整记录（PushT 355、Reacher 1）；临时目录有 Reacher 373、Cube 319、TwoRoom 250 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,297/2,600（49.9%），剩余 1,303 项。四个 worker 会话仍打开，本轮比上一轮增加 2 项。GPU0–3 空闲显存约 10.2/25.7/9.4/25.7 GiB，利用率约 31%/82%/100%/54%；GPU2 满载，仍有约 9.4 GiB 显存。宿主 MemAvailable 约 263 GiB，memory PSI 的 10/60/300 秒窗口均为 0；SwapFree 约 1.7 MiB但阈值为 0，未见 OOM。最近 31 分钟增加 12 项，按该区间估算剩余约 56 小时；短时间吞吐不稳定，继续以约 50–70 小时作为宽松 ETA。

### 5.336 粗测运行快照

截至 2026-09-26 20:40（北京时间），正式目录有 357 条完整记录（PushT 356、Reacher 1）；临时目录有 Reacher 374、Cube 320、TwoRoom 250 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,300/2,600（50.0%），剩余 1,300 项。四个 worker 会话仍打开，本轮增加 3 项。GPU0–3 空闲显存约 13.3/25.7/9.4/25.7 GiB，利用率约 86%/100%/100%/45%；GPU1、GPU2 利用率满载，GPU2 显存余量最低。宿主 MemAvailable 约 275 GiB，memory PSI 的 10/60/300 秒窗口中短窗有所升高（some 2.46%、full 1.86%），300 秒窗口分别约 0.27%/0.20%，当前没有内存告警或 OOM。SwapFree 约 1.8 MiB，阈值为 0。近 37 分钟增加 15 项，粗略剩余时间约 54 小时；考虑共享负载变化，保留 45–70 小时范围。

### 5.337 粗测运行快照

截至 2026-09-26 20:46（北京时间），正式目录有 358 条完整记录（PushT 357、Reacher 1）；临时目录有 Reacher 375、Cube 321、TwoRoom 251 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,304/2,600（50.2%），剩余 1,296 项。四个 worker 会话仍打开，本轮新增 4 项。GPU0–3 空闲显存约 10.2/25.7/9.4/25.7 GiB，利用率约 100%/89%/16%/42%；GPU2 仍为显存余量最低的设备。宿主 MemAvailable 约 269 GiB，memory PSI 的 10/60/300 秒窗口较低（some 0.15%/0.33%/0.15%，full 0.13%/0.29%/0.13%）；SwapFree 约 2.4 MiB，门槛为 0，未见 OOM。近 36 分钟增加 19 项，按该窗口估算剩余约 41 小时；粗测速率会随共享负载波动。

### 5.338 粗测运行快照

截至 2026-09-26 20:52（北京时间），正式目录有 358 条完整记录（PushT 357、Reacher 1）；临时目录有 Reacher 376、Cube 322、TwoRoom 251 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,306/2,600（50.2%），剩余 1,294 项。四个 worker 会话仍打开，本轮新增 2 项。GPU0–3 空闲显存约 10.2/25.7/15.0/25.7 GiB，利用率约 100%/97%/98%/93%；共享计算负载在四卡上都较高。宿主 MemAvailable 约 266 GiB，memory PSI 的 10/60/300 秒窗口很低，300 秒约 0.02%/0.01%；SwapFree 约 5 MiB，阈值为 0，未见 OOM。近 49 分钟增加 21 项，按此区间估算剩余约 50 小时；最近短窗吞吐偏低，继续把 ETA 看作约 50–65 小时的粗估。

### 5.339 粗测运行快照

截至 2026-09-26 20:58（北京时间），正式目录有 359 条完整记录（PushT 358、Reacher 1）；临时目录有 Reacher 377、Cube 322、TwoRoom 252 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,309/2,600（50.3%），剩余 1,291 项。四个 worker 会话仍打开，本轮新增 3 项。GPU0–3 空闲显存约 13.2/25.7/9.4/25.7 GiB，利用率约 83%/86%/89%/96%；共享 GPU 利用率仍高，GPU2 显存余量最低。宿主 MemAvailable 约 272 GiB，memory PSI 的 10/60/300 秒窗口均为 0；SwapFree 约 8.6 MiB但运行门槛为 0，未见 OOM。近 55 分钟增加 24 项，按此窗口估算剩余约 49 小时；综合短时速度波动，ETA 暂按约 45–70 小时估计。

### 5.340 粗测运行快照

截至 2026-09-26 21:04（北京时间），正式目录有 359 条完整记录（PushT 358、Reacher 1）；临时目录有 Reacher 378、Cube 323、TwoRoom 252 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,311/2,600（50.4%），剩余 1,289 项。四个 worker 会话仍打开，本轮新增 2 项。GPU0–3 空闲显存约 12.6/25.7/9.4/25.7 GiB，利用率约 12%/75%/93%/92%；GPU2 仍有最低的显存余量。宿主 MemAvailable 约 261 GiB，memory PSI 的 10/60/300 秒窗口较低（some 0.00%/0.10%/0.13%，full 0.00%/0.09%/0.12%）。SwapFree 约 8.7 MiB而门槛为 0，未见 OOM。近 61 分钟增加 26 项，粗略剩余约 50 小时；短时吞吐持续变化。

### 5.341 粗测运行快照

截至 2026-09-26 21:10（北京时间），正式目录有 360 条完整记录（PushT 359、Reacher 1）；临时目录有 Reacher 378、Cube 323、TwoRoom 252 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,312/2,600（50.5%），剩余 1,288 项。四个 worker 会话仍打开，本轮新增 1 项。GPU0–3 空闲显存约 10.2/25.7/9.4/25.7 GiB，利用率约 94%/46%/87%/84%；GPU2 显存余量仍最低。宿主 MemAvailable 约 261 GiB，memory PSI 的 10/60/300 秒窗口约 0.00%/0.00%/0.01%；SwapFree 约 8.8 MiB，运行门槛为 0，未见 OOM。单个 6 分钟窗口只增加 1 项，ETA 暂以最近一小时约 50 小时量级为参考，不按短窗单独外推。

### 5.342 粗测运行快照

截至 2026-09-26 21:16（北京时间），正式目录有 360 条完整记录（PushT 359、Reacher 1）；临时目录有 Reacher 379、Cube 324、TwoRoom 253 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,315/2,600（50.6%），剩余 1,285 项。四个 worker 会话仍打开，本轮新增 3 项。GPU0–3 空闲显存约 10.2/25.7/9.4/25.7 GiB，利用率约 0%/78%/78%/58%；GPU2 显存余量最低。宿主 MemAvailable 约 262 GiB，memory PSI 的 10/60/300 秒窗口均为 0；SwapFree 约 8.8 MiB，门槛为 0，未见 OOM。最近约 73 分钟增加 30 项，按整体速度估算剩余约 52 小时；以约 45–65 小时作为当前粗略范围。

### 5.343 粗测运行快照

截至 2026-09-26 21:22（北京时间），正式目录有 361 条完整记录（PushT 360、Reacher 1）；临时目录有 Reacher 380、Cube 325、TwoRoom 253 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,318/2,600（50.7%），剩余 1,282 项。四个 worker 会话仍打开，本轮新增 3 项。GPU0–3 空闲显存约 10.5/25.7/15.0/25.7 GiB，利用率约 99%/37%/99%/0%；GPU0和GPU2计算利用率接近满载。宿主 MemAvailable 约 267 GiB，memory PSI 的 10/60/300 秒窗口较低，300 秒 some/full 约 0.19%/0.13%；SwapFree 约 8.9 MiB而运行门槛为 0，未见 OOM。最近约 79 分钟增加 33 项，剩余时间粗估约 51 小时，继续以约 45–70 小时范围看待。

### 5.344 粗测运行快照

截至 2026-09-26 21:28（北京时间），正式目录有 361 条完整记录（PushT 360、Reacher 1）；临时目录有 Reacher 381、Cube 325、TwoRoom 253 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,319/2,600（50.7%），剩余 1,281 项。四个 worker 会话仍打开，本轮新增 1 项。GPU0–3 空闲显存约 10.2/25.7/9.4/25.7 GiB，利用率约 34%/38%/95%/0%；GPU2仍处于高计算负载并有最低显存余量。宿主 MemAvailable 约 260 GiB，memory PSI 的 10/60/300 秒窗口均很低，300 秒 some/full 约 0.03%/0.01%；SwapFree 约 9.3 MiB，swap 门槛为 0，未见 OOM。近 85 分钟增加 34 项，按累计速度估算剩余约 50 小时；继续考虑共享负载波动。

### 5.345 粗测运行快照

截至 2026-09-26 21:34（北京时间），正式目录有 362 条完整记录（PushT 361、Reacher 1）；临时目录有 Reacher 382、Cube 326、TwoRoom 254 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,323/2,600（50.9%），剩余 1,277 项。四个 worker 会话仍打开，本轮新增 4 项。GPU0–3 空闲显存约 10.2/25.7/15.1/25.7 GiB，利用率约 95%/0%/97%/90%。宿主 MemAvailable 约 264 GiB；memory PSI 的 10/60/300 秒窗口升至约 0.05%/0.75%/0.73%（some），full 约 0.05%/0.73%/0.69%，仍远低于 1%。SwapFree 仅约 0.14 MiB，但门槛为 0，任务仍运行且未见 OOM。最近 30 分钟增加 12 项，剩余时间粗估约 53 小时，继续留有速度波动余量。

### 5.346 粗测运行快照

截至 2026-09-26 21:40（北京时间），独立完整记录仍为 1,323/2,600（50.9%），剩余 1,277 项；本轮约 6 分钟未新增完整条件。四个 worker 会话均仍打开。GPU0–3 空闲显存约 16.4/25.7/9.4/25.7 GiB，利用率约 91%/100%/100%/22%；GPU2仍有最低显存余量。宿主 MemAvailable 约 263 GiB，memory PSI 的 10/60/300 秒窗口为 0/0/0.24%（some）和 0/0/0.22%（full）；SwapFree 约 0.25 MiB但运行门槛为 0，未见 OOM。短时无新增时继续等待下个监控周期，不按该单窗外推 ETA。

### 5.347 粗测运行快照

截至 2026-09-26 21:46（北京时间），正式目录有 363 条完整记录（PushT 362、Reacher 1）；临时目录有 Reacher 383、Cube 327、TwoRoom 254 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,326/2,600（51.0%），剩余 1,274 项。四个 worker 会话仍打开，本轮新增 3 项。GPU0–3 空闲显存约 10.2/25.5/9.4/25.7 GiB，利用率约 96%/23%/100%/57%；GPU2仍有最低显存余量。宿主 MemAvailable 约 272 GiB，memory PSI 的 10/60/300 秒窗口中300秒约 0.41%（some）/0.40%（full）；SwapFree 约 0.34 MiB，阈值为 0，未见 OOM。近 30 分钟增加 11 项，按该窗口估算剩余约 58 小时，速度和 ETA 仍受共享负载影响。

### 5.348 粗测运行快照

截至 2026-09-26 21:52（北京时间），正式目录有 363 条完整记录（PushT 362、Reacher 1）；临时目录有 Reacher 384、Cube 328、TwoRoom 255 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,329/2,600（51.1%），剩余 1,271 项。四个 worker 会话仍打开，本轮新增 6 项。GPU0–3 空闲显存约 10.2/25.4/15.1/25.7 GiB，利用率约 0%/100%/91%/57%；共享负载仍有较大波动。宿主 MemAvailable 约 261 GiB，memory PSI 的 10/60/300 秒窗口较低，300 秒 some/full 约 0.26%/0.25%；SwapFree 约 0.33 MiB，运行门槛为 0，未见 OOM。近 48 分钟增加 18 项，估算剩余约 57 小时；继续将 ETA 视为粗略区间。

### 5.349 粗测运行快照

截至 2026-09-26 21:58（北京时间），正式目录有 364 条完整记录（PushT 363、Reacher 1）；临时目录有 Reacher 385、Cube 328、TwoRoom 255 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,331/2,600（51.2%），剩余 1,269 项。四个 worker 会话仍打开，本轮新增 2 项。GPU0–3 空闲显存约 13.3/25.4/9.4/25.7 GiB，利用率约 96%/5%/100%/78%；GPU0与GPU2负载较高，GPU2余量最低。宿主 MemAvailable 约 276 GiB；memory PSI 的 10/60/300 秒窗口有所升高，300秒 some/full 约 0.57%/0.44%，10秒 some/full 约 2.41%/1.83%，仍未见 OOM。SwapFree 仅约 8 KiB，但 swap 门槛为 0。近 42 分钟增加 16 项，粗略剩余约 55 小时；ETA 继续按约 50–65 小时理解。

### 5.350 粗测运行快照

截至 2026-09-26 22:04（北京时间），正式目录有 364 条完整记录（PushT 363、Reacher 1）；临时目录有 Reacher 386、Cube 329、TwoRoom 255 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,333/2,600（51.3%），剩余 1,267 项。四个 worker 会话仍打开，本轮新增 2 项。GPU0–3 空闲显存约 12.2/25.4/9.4/25.7 GiB，利用率约 29%/97%/33%/60%；GPU2仍为显存余量最低的设备。宿主 MemAvailable 约 263 GiB，memory PSI 的 10/60/300 秒窗口较低，300秒 some/full 约 0.35%/0.28%；SwapFree 约 43 MiB但不再限制任务，未见 OOM。近一小时增加 22 项，按此估算剩余约 58 小时，继续考虑共享负载波动。

### 5.351 粗测运行快照

截至 2026-09-26 22:10（北京时间），正式目录有 365 条完整记录（PushT 364、Reacher 1）；临时目录有 Reacher 386、Cube 330、TwoRoom 256 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,336/2,600（51.4%），剩余 1,264 项。四个 worker 会话仍打开，本轮新增 3 项。GPU0–3 空闲显存约 16.2/25.4/15.1/25.7 GiB，利用率约 0%/21%/96%/100%；GPU2和GPU3计算负载高，显存仍有余量。宿主 MemAvailable 约 270 GiB，memory PSI 的 10/60/300 秒窗口较低，300秒 some/full 约 0.11%/0.10%。SwapFree 约 43.5 MiB，已取消swap门槛，未见 OOM。近 54 分钟增加 25 项，按整体速度估算还需约 54 小时；此估值仍受共享负载波动影响。

### 5.352 粗测运行快照

截至 2026-09-26 22:16（北京时间），正式目录有 365 条完整记录（PushT 364、Reacher 1）；临时目录有 Reacher 387、Cube 330、TwoRoom 256 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,337/2,600（51.4%），剩余 1,263 项。四个 worker 会话仍打开，本轮新增 1 项。GPU0–3 空闲显存约 16.2/25.4/9.4/25.7 GiB，利用率约 0%/38%/97%/100%；GPU2和GPU3利用率高，但显存仍有余量。宿主 MemAvailable 约 264 GiB，memory PSI 的 10/60/300 秒窗口均为 0；SwapFree 约 173 MiB，门槛为 0，未见 OOM。近 72 分钟增加 26 项，按累计速度估算还需约 63 小时；短时吞吐仍有起伏。

### 5.353 粗测运行快照

截至 2026-09-26 22:22（北京时间），正式目录有 366 条完整记录（PushT 365、Reacher 1）；临时目录有 Reacher 388、Cube 331、TwoRoom 257 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,341/2,600（51.6%），剩余 1,259 项。四个 worker 会话仍打开，本轮新增 4 项。GPU0–3 空闲显存约 18.9/25.4/9.4/25.5 GiB，利用率约 87%/13%/62%/78%；GPU2仍是显存余量最低的设备。宿主 MemAvailable 约 274 GiB，memory PSI 的 10/60/300 秒窗口中300秒 some/full 约 0.48%/0.44%，短窗 some/full 约 1.91%/1.91%；SwapFree 约 78 MiB但不作为运行门槛，未见 OOM。近一小时增加 23 项，按此区间估算剩余约 55 小时，吞吐仍受共享负载影响。

### 5.354 粗测运行快照

截至 2026-09-26 22:28（北京时间），正式目录有 367 条完整记录（PushT 366、Reacher 1）；临时目录有 Reacher 389、Cube 332、TwoRoom 257 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,344/2,600（51.7%），剩余 1,256 项。四个 worker 会话仍打开，本轮新增 3 项。GPU0–3 空闲显存约 13.1/25.4/15.0/25.5 GiB，利用率约 57%/99%/96%/50%。宿主 MemAvailable 约 267 GiB；memory PSI 的 10/60/300 秒窗口有所上升，300秒 some/full 约 1.26%/1.10%，60秒约 2.13%/1.77%；SwapFree 约 73 MiB，门槛为 0，未见 OOM。近 24 分钟增加 11 项，估算剩余约 46 小时；主机仍有充足可用内存，后续继续观察压力曲线和worker状态。

### 5.355 粗测运行快照

截至 2026-09-26 22:35（北京时间），正式目录有 367 条完整记录（PushT 366、Reacher 1）；临时目录有 Reacher 389、Cube 332、TwoRoom 258 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,345/2,600（51.7%），剩余 1,255 项。四个 worker 会话仍打开，本轮新增 2 项。GPU0–3 空闲显存约 9.9/25.4/9.4/25.5 GiB，利用率约 8%/80%/95%/89%；GPU2仍为显存余量最低的设备。宿主 MemAvailable 约 263 GiB，memory PSI 的 10/60/300 秒窗口较低，300秒 some/full 约 0.53%/0.45%；SwapFree 约 73 MiB，swap 门槛为 0，未见 OOM。近 31 分钟增加 12 项，按累计速度估算剩余约 54 小时；继续留有共享负载造成的误差。

### 5.356 粗测运行快照

截至 2026-09-26 22:41（北京时间），正式目录有 368 条完整记录（PushT 367、Reacher 1）；临时目录有 Reacher 390、Cube 333、TwoRoom 258 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,348/2,600（51.8%），剩余 1,252 项。四个 worker 会话仍打开，本轮新增 3 项。GPU0–3 空闲显存约 16.1/25.4/9.4/25.5 GiB，利用率约 0%/64%/99%/0%；GPU2仍有最低显存余量。宿主 MemAvailable 约 267 GiB；memory PSI 的 10/60/300 秒窗口中300秒 some/full 约 1.33%/1.14%，60秒约 0.78%/0.58%，可用内存仍充足，未见 OOM。SwapFree 约 105 MiB但门槛为 0。最近 37 分钟增加 15 项，按整体速度估算剩余约 52 小时，ETA 仍是粗略值。

### 5.357 粗测运行快照

截至 2026-09-26 22:47（北京时间），正式目录有 369 条完整记录（PushT 368、Reacher 1）；临时目录有 Reacher 391、Cube 333、TwoRoom 259 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,351/2,600（52.0%），剩余 1,249 项。四个 worker 会话仍打开，本轮新增 3 项。GPU0–3 空闲显存约 9.9/25.4/9.4/25.5 GiB，利用率约 9%/100%/83%/6%；GPU2仍有最低显存余量。宿主 MemAvailable 约 264 GiB，但 memory PSI 短窗有明显尖峰：10秒 some/full 约 11.44%/10.22%，60秒约 2.63%/2.37%，300秒约 1.27%/1.14%。目前任务会话仍运行、未见 OOM；SwapFree 约 114 MiB且门槛为 0。近一小时增加 25 项，粗略剩余约 50 小时；下一轮重点观察 PSI 是否回落。

### 5.358 粗测运行快照

截至 2026-09-26 22:53（北京时间），正式目录有 369 条完整记录（PushT 368、Reacher 1）；临时目录有 Reacher 392、Cube 334、TwoRoom 259 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,353/2,600（52.0%），剩余 1,247 项。四个 worker 会话仍打开，本轮新增 2 项。GPU0–3 空闲显存约 9.9/25.4/9.4/25.5 GiB，利用率约 25%/19%/90%/78%；GPU2仍是显存余量最低的卡。宿主 MemAvailable 约 257 GiB，memory PSI 短窗已回落至 0，300秒 some/full 约 0.33%/0.29%。SwapFree 约 13.5 MiB，门槛为 0，未见 OOM。近 49 分钟增加 20 项，按整体速度估算剩余约 51 小时；ETA 仍会随共享负载变化。

### 5.359 粗测运行快照

截至 2026-09-26 22:59（北京时间），正式目录有 370 条完整记录（PushT 369、Reacher 1）；临时目录有 Reacher 392、Cube 334、TwoRoom 259 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,354/2,600（52.1%），剩余 1,246 项。四个 worker 会话仍打开，本轮新增 1 项。GPU0–3 空闲显存约 9.9/25.4/9.4/25.5 GiB，利用率约 85%/0%/96%/66%；GPU2仍有最低显存余量。宿主 MemAvailable 约 257 GiB，memory PSI 的 10/60/300 秒窗口回落至接近 0，300秒 some/full 约 0.06%/0.05%。SwapFree 约 3.5 MiB，门槛为 0，未见 OOM。近 55 分钟增加 21 项，粗略剩余约 55 小时，单个短窗只新增 1 项不改变整体估算。

### 5.360 粗测运行快照

截至 2026-09-26 23:08（北京时间），正式目录有 371 条完整记录（PushT 370、Reacher 1）；临时目录有 Reacher 394、Cube 335、TwoRoom 260 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,359/2,600（52.3%），剩余 1,241 项。四个 worker 会话仍打开，本轮增加 5 项。GPU0–3 空闲显存约 15.6/25.4/9.4/25.5 GiB，利用率约 76%/100%/91%/93%；GPU2仍为显存余量最低的设备。宿主 MemAvailable 约 268 GiB，但memory PSI短窗再次升高：10秒 some/full 约 9.20%/8.95%，60秒约 2.34%/2.28%，300秒回落在约 0.51%/0.50%。SwapFree 约 76 KiB，运行门槛为 0，未见 OOM。近 58 分钟增加 23 项，粗略剩余约 52 小时；继续观察内存压力是否保持短暂。

### 5.361 粗测运行快照

截至 2026-09-26 23:38（北京时间），正式目录有 374 条完整记录（PushT 373、Reacher 1）；临时目录有 Reacher 397、Cube 338、TwoRoom 262 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,370/2,600（52.7%），剩余 1,230 项。四个 worker 会话仍打开，最近约半小时增加 11 项。GPU0–3 空闲显存约 13.1/25.4/9.4/25.5 GiB，利用率约 0%/87%/93%/82%；GPU2仍有最低显存余量。宿主 MemAvailable 约 276 GiB，memory PSI 的10/60/300秒窗口回落为0；SwapFree约472 MiB，但swap门槛为0，未见OOM。粗测速窗干扰统计为1,839个 `interference_flagged` 和903个边界采样未观察到外部计算的窗口；后者不代表整段窗口已证明隔离。按最近半小时速度估算，剩余约54小时，仍受共享负载影响。

### 5.362 粗测运行快照

截至 2026-09-27 00:07（北京时间），正式目录有 377 条完整记录（PushT 376、Reacher 1）；临时目录有 Reacher 402、Cube 342、TwoRoom 263 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,383/2,600（53.2%），剩余 1,217 项。四个 worker 会话仍打开，最近约半小时增加 13 项。GPU0–3 空闲显存约 12.8/25.4/15.0/25.5 GiB，利用率约 15%/60%/99%/43%。宿主 MemAvailable 约 274 GiB，memory PSI 的10/60/300秒窗口均为0；SwapFree约312 MiB但门槛为0，未见OOM。当前粗测窗口统计为1,865个 `interference_flagged` 和903个边界采样未观察到外部计算。近半小时速度约26项/小时，估算剩余约47小时；ETA仍受共享负载变化影响。

### 5.363 粗测运行快照

截至 2026-09-27 00:37（北京时间），正式目录有 380 条完整记录（PushT 379、Reacher 1）；临时目录有 Reacher 406、Cube 344、TwoRoom 265 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,394/2,600（53.6%），剩余 1,206 项。四个 worker 会话仍打开，最近约半小时增加 11 项。GPU0–3 空闲显存约 13.1/25.4/9.4/25.5 GiB，利用率约 99%/75%/96%/5%。宿主 MemAvailable 约 272 GiB，memory PSI 的10/60/300秒窗口均很低，300秒约0.06%；SwapFree约76 MiB，运行门槛为0，未见OOM。粗测干扰状态计数为1,887个 `interference_flagged` 和903个边界采样未观察到外部计算的窗口。近半小时增加11项，按该速度粗估剩余约53小时。

### 5.364 粗测运行快照

截至 2026-09-27 01:12（北京时间），正式目录有 383 条完整记录（PushT 382、Reacher 1）；临时目录有 Reacher 410、Cube 348、TwoRoom 267 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,407/2,600（54.1%），剩余 1,193 项。四个 worker 会话仍打开，最近约35分钟增加13项。GPU0–3 空闲显存约 9.9/25.4/9.4/25.5 GiB，利用率约100%/3%/91%/84%。宿主 MemAvailable 约273 GiB，memory PSI 的10/60/300秒窗口为0；SwapFree仅约1.4 MiB但swap门槛为0，未见OOM。粗测干扰状态计数为1,913个 `interference_flagged` 和903个边界采样未观察到外部计算的窗口。按近35分钟吞吐估算剩余约54小时。

### 5.365 粗测运行快照

截至 2026-09-27 01:41（北京时间），正式目录有 386 条完整记录（PushT 385、Reacher 1）；临时目录有 Reacher 414、Cube 351、TwoRoom 269 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,419/2,600（54.6%），剩余 1,181 项。四个 worker 会话仍打开，最近约29分钟增加12项。GPU0–3 空闲显存约 13.0/26.6/9.4/25.5 GiB，利用率约0%/100%/96%/79%。宿主 MemAvailable 约283 GiB；memory PSI 300秒 some/full 约0.59%/0.48%，60秒约1.51%/1.23%。SwapFree仅约80 KiB，但运行门槛为0，未见OOM。粗测窗口状态有1,937个 `interference_flagged` 和903个边界采样未观察到外部计算窗口。近半小时增加12项，剩余时间粗估约48小时。

### 5.366 粗测运行快照

截至 2026-09-27 02:10（北京时间），正式目录有 389 条完整记录（PushT 388、Reacher 1）；临时目录有 Reacher 420、Cube 354、TwoRoom 271 条，其中 Reacher 与正式目录重复 1 条，独立完成数为 1,433/2,600（55.1%），剩余 1,167 项。四个 worker 会话仍打开，最近29分钟增加14项。GPU0–3 空闲显存约10.0/26.0/9.4/25.5 GiB，利用率约76%/58%/100%/25%；GPU2满载且显存余量最低。宿主 MemAvailable 约279 GiB，memory PSI 300秒 some/full 约0.26%/0.25%；SwapFree不足1 MiB但swap门槛为0，未见OOM。按最近约半小时速度，剩余粗估约40小时，受共享负载波动影响。

### 5.367 阶段完成项复核与粗测前沿修正

截至 2026-09-27 07:11（北京时间），2,600项计时粗测有1,649个唯一完整条件，剩余951项；四个worker最近一次检查仍在运行。自5.366快照以来增加216项。按这段约5小时的速度外推，粗测约还需22小时；该估计只针对粗测，之后仍需精测、跨任务选型和最终分析。

本次按当前产物复核，旧快照5.2节中“Guidance 540/1,008”的状态已过期：四任务 post-opt 共864/864、guided-flow 共144/144，全部manifest为completed，1,008个变体文件均恰有50行且JSON可解析。主扫描索引为2,600/2,600 completed；四个候选池均各有76,800条记录、11,100条控制记录，四任务probe状态均为ok。

为处理粗测共享GPU负载，`select-fine-timing` 已改为同时保留观测到的batch1前沿与“batch1边界采样未观察到外部计算”的敏感性前沿，并始终保留固定seed锚点。受干扰或缺少边界快照的前沿候选会被明确标记，以供隔离精测复测；该边界状态不等于证明完整测量窗口无干扰。新增 `--retry-interfered-fine` 可在首次精测窗口仍受干扰或缺少快照时重测batch1和batch50，并保留前次摘要。当前没有生成精测选择或启动精测；待全部粗测完成后再运行选择流程。

三条`/tmp` staging根的完整粗测记录已用新增的`merge-timing-staging`命令归档进正式Phase1.5输出目录；归档采用条件锁和原子写入，重复运行可识别已同步记录。重复的Reacher条件保留两次粗测样本，主窗口不被覆盖，并把batch50干扰状态合并为更保守的`interference_flagged`。staging worker仍在运行，粗测收尾时需再同步一次并核对全量覆盖。

验证：`tests.test_round5_phase1_5` 68项通过；主入口和diagnostics入口的`validate`均返回`status=ok`；Python编译、`git diff --check`以及staging归档命令执行通过。细测筛选和归档入口现在都要求粗测窗口恰为5次warm-up、10次同步测量。
