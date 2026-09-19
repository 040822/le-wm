# Round 5 Phase 1：最终方案决策书

## 结论速览

本轮在 R4-AB、seed 3072、epoch 10、canonical `legacy_50` 旧协议（LeWM 上游口径，
每任务 50 episodes）上，复用 Phase 4.5-4 的 76 条无 guidance 基线，新增 168 条
guidance 条件，回答了两件事：

1. **主推理方案**：继续以 **P3（A 生成 64 条动作候选 + B 选优）为主规划器**，
   action flow step 取 **1 或 2**。
2. **round3 方案能否与 P3 结合**：**可以**。guided-flow 与 post-opt 都已能在 P3
   上运行，并额外测了「先选后精修」（post_opt_refine）语义。收益**任务相关**：
   Reacher 明显受益，其余三个接近饱和的任务基本中性。

## 关键证据（legacy 50-episode，四任务宏平均，百分点）

| 方案 | step1 | step2 | step5 | step10 | step16 | step32 |
|---|---:|---:|---:|---:|---:|---:|
| P3 无 guidance | 92.5 | 97.0 | 96.0 | 96.0 | 93.5 | 95.5 |
| P3 + guided-flow（每候选） | 97.5 | 97.5 | 94.0 | 95.5 | 93.5 | 94.0 |
| P3 + post-opt（每候选） | 97.5 | 97.5 | 96.0 | 94.5 | 95.5 | 95.5 |
| P3 + post-opt_refine（先选后精修） | 97.5 | 95.5 | 97.5 | 94.5 | 94.5 | 96.5 |
| P2 无 guidance | 96.0 | 94.5 | 95.5 | 95.0 | 94.0 | 93.5 |
| P2 + guided-flow | 98.5 | 96.5 | 97.0 | 94.0 | 92.0 | 93.0 |
| P2 + post-opt | 98.5 | 95.5 | 96.0 | 93.5 | 97.0 | 92.5 |
| P0 无 guidance | 92.0 | 93.0 | 92.5 | 92.5 | 91.5 | 92.0 |
| P0 + guided-flow | 96.5 | 97.0 | 94.0 | 94.0 | 95.5 | 94.5 |

guidance 相对无 guidance 的逐任务平均差（6 个 step 平均）：

| 任务 | P0 GF | P0 PO | P2 GF | P2 PO | P3 GF | P3 PO | P3 refine |
|---|---:|---:|---:|---:|---:|---:|---:|
| cube | 0.0 | 0.0 | −0.3 | +1.3 | 0.0 | 0.0 | 0.0 |
| pusht | +1.3 | +1.3 | +0.3 | 0.0 | −1.0 | 0.0 | +0.7 |
| **reacher** | **+9.7** | **+7.0** | **+1.3** | **+2.0** | **+2.7** | **+4.0** | **+3.0** |
| tworoom | +1.0 | +0.7 | +0.3 | −0.3 | −0.7 | 0.0 | 0.0 |

- Reacher 是唯一有明确、跨 mode 一致收益的任务；P0 从 72–76% 提升到 88–90%。
- 最强的单点证据：Reacher **P3 step1** 加 guidance，相对无 guidance
  +18 pp（guided_flow / post_opt，9 改进 0 退化，McNemar exact p≈0.0039）；
  post_opt_refine +16 pp（8/0，p≈0.0078）。Reacher step2 及更高步数的方向不一致，
  因此 guidance 的收益集中在低步数。
- cube / tworoom 接近 100% 上限，guidance 无区分度；pusht 基本持平。
- 逐任务配对比较（440 条）中，guidance_vs_none：56 正向、25 负向、87 持平。

## P1（随机初始化 CEM）对照

P1 是标准 CEM planner（B + 随机初始化 CEM）。本轮按计划复用 Phase 4.5-4 的
legacy P1 结果（reacher 用 `cem-clip`），它是学习 planner 的重要参照：

| 任务 | P1 | P0 none | P2 none | P3 none | P3 + guidance |
|---|---:|---:|---:|---:|---:|
| cube | 70% | 100% | 96–98% | 100% | 100% |
| pusht | 90% | 96–100% | 94–98% | 96–98% | 96–100% |
| reacher | **92%** | 72–78% | 80–90% | 74–90% | **92–94%** |
| tworoom | 100% | 96–100% | 98–100% | 98–100% | 100% |

- cube：学习 planner 明显优于 P1（+26~30 pp，McNemar p≈1e-4）；P1 的随机 CEM
  在 cube 上偏弱。
- pusht / tworoom：学习 planner 不劣于或优于 P1。
- **reacher 是例外**：P1（92%）显著强于无 guidance 的 P0/P3（74–90%），甚至
  step1 的 P0/P3 明显低于 P1（−18~−20 pp，p≈0.01–0.02）。加入 guidance 后
  P3 才追平/略超 P1（92–94%）。
- 因此最终方案在 reacher 上应把 **P1（cem-clip）或 P3+guidance** 作为并列候选；
  其余任务以 P3 为主。

## 最终方案建议

1. **主规划器**：P3，`num_candidates=64`，Euler，flow step=2 为默认（step=1 作为
   低延迟备选）。两档在 P3 上并列最好。
2. **guidance 组合**：作为**可选增益**，默认关闭；对 Reacher 等未饱和任务启用。
   若启用，优先 **P3 + post-opt_refine**（先生成并 B 选优、再对选中动作做一次
   post-opt）：成本最低，且在 Reacher/Push-T 上表现稳定；其次 P3 + post-opt。
   对 Reacher，建议在 **step1** 启用（+16–18 pp，McNemar p<0.01）。
   `P3 + guided_flow` 在 pusht/tworoom 有轻微负向，不作为默认。
3. **训练模型**：R4-AB（只训 A/B），不启用 D/E；沿用 Phase 4.5-4 的 4 任务
   seed 3072 epoch 10 权重。
4. **评测协议**：主实验使用与 LeWM 论文可比的旧协议 `legacy`（`legacy_50`）；
   已确认 legacy cohort 基于 `select_eval_cohort` 的上游采样复刻。
5. **Online train**：本轮不纳入主方案；由 Phase 2 在 P3 评测下单独检验（见下）。

## 关于 round3 方案与 P3 的结合

- **可行**：已在 `Round4BestOfNPolicy` 中实现两种语义——(a) 每条候选生成时
  引导；(b) 先用普通候选做 B 选优、再对选中动作 post-opt 精修。
- **结论**：结合后的收益继承单候选 observed 规律——只在未饱和任务（Reacher）
  上稳定为正；在接近饱和任务上不劣化或轻微劣化。
- 因此最终主方案不强制 guidance，而是把它作为按任务可选的增益开关。

## 边界与限制

- 单 checkpoint、单训练 seed、每条件 50 episodes、旧协议非严格 held-out；
  以上是探索性描述性结论，不作统计显著性声明。
- 受共享 GPU/CPU 抢占影响，P2 条件实测耗时 13–33 分钟/条件；耗时数字不作为
  与 LeWM 的正式延迟比较（延迟比较见 Round 4 Phase 4.5-3）。
- D/E、P4、latent CEM、seed 扩展均不在本轮结论内。

## 产物

- 数据报告：`docs/report/round5/round5_phase1_report.md`
- 分析 JSON/CSV：`outputs/round5/phase1_seed3072_legacy/analysis/`
- 条件结果：`outputs/round5/phase1_seed3072_legacy/conditions/legacy/`
- 配置：`config/round5/phase1.json`；入口：`scripts/round5_phase1.py`
  `scripts/run_round5_phase1.sh`
