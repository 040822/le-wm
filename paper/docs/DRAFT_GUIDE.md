# CoWM 初稿写作与证据说明

更新：2026-09-24。目标：CVPR 2027 工作稿。此版本根据用户指定会话重写，正文是完整英文初稿，实验结论严格受现有证据限制。

## 1. 会话依据与本次取舍

来源会话：`01a0ba23-c649-7b90-8acc-96716f894f9f`（2026-09-19 起）。读取了本地保存的用户与助手消息，而非启动交互式 `codex resume`。会话中的用户决定依次包括：

- 2026-09-21：主模型收敛为 Shared DiT，主要任务为 pred action 与 parallel pred future；C/D/E 退居针对性消融；在线训练留给后续研究。
- 贡献围绕 coupled、fast、probe 收敛；需要解释为什么使用 R4-AB，比较 P3 与 P0-PO/GF 能否减少 CEM 搜索需求。
- 用户随后决定沿该方向推进，并委托文献研究；后续讨论明确将耦合拆为参数、动作输入、动作路径梯度三种联系。
- Phase1.5 定位为固定已有 checkpoint 的推理诊断，不能凭它证明训练耦合的因果收益；保留原有 50 个样本，并允许冻结表示上的读出器。

本次还对照了 [本地 CoWM 笔记](Couple_lewm.md)、[Phase1 报告](../../docs/report/round5/round5_phase1_report.md)、[Phase1.5 中期报告](../../docs/report/round5/round5_phase1_5_interim_report.md) 和实际实现。

暂定标题：**CoWM: Coupled Latent World–Action Modeling for Action Selection and Refinement**。
CoWM 来自用户现有笔记，仍是工作名。它不修改代码中的 Fast-LeWAM/R4-AB 名称，也不预先承诺“Fast”的比较结论。

核心问题：

> 训练时的动作—动力学耦合，能否改善有限决策预算下的动作选择或修正？其中哪些耦合必要，哪些反而有害？

论文的论证顺序：训练连接 → 决策接口 → 真实后果诊断 → 性能与延迟。三者的作用分别是方法、计算价值与机制证据；不把普通 batching 包装成新硬件算法，不把 probe 本身当成已经验证的贡献。

## 2. 阅读与修改顺序

| 文件 | 作用 |
|---|---|
| `main.tex` | 标题、匿名审稿设置、正文入口 |
| `sec/0_abstract.tex` | 科学问题、耦合模型、现有探索结果与待补证据 |
| `sec/1_intro.tex` | 控制问题 → 近邻边界 → 耦合假设 → 方法 → 证据范围 |
| `sec/2_related.tex` | latent WM、生成策略/规划、WAM、物理与决策 probe |
| `sec/3_method.tex` | A/B、三类耦合、P0/P3/PO/GF/CEM，含公式 |
| `sec/4_experiments.tex` | Phase1 主表、Phase1.5 排序诊断、正式对照与计时占位 |
| `sec/5_discussion.tex` | 适用范围、局限、结论 |
| `supplement.tex`、`sec/6_appendix.tex` | 单独编译的工作补充材料、对照设计和复现细节 |
| `fig/draft_architecture.tex` | 原生 LaTeX 结构示意，尚可继续美化 |
| `tables/*.tex` | 从结果 JSON 提取的固定数值 |
| `docs/evidence_snapshot.json` | 本次引用结果行、路径与源文件 SHA256；内部账本 |

旧稿已备份到 `docs/archive/pre_cowm_20260924.zip`，包括修改前已有的未提交稿件。旧稿围绕 ABDE/path proposal 的叙事未混进新正文；编号章节布局保持不变。保留原模板示例。

## 3. 主张—证据—缺口

| 主张/问题 | 当前正文如何写 | 尚需证据 |
|---|---|---|
| 共享 A/B 支持动作生成和并行未来预测 | 已实现的方法事实 | 正式版本资源统计 |
| 参数共享优于分离 predictor | 未宣称成立 | 同 encoder 双 DiT，控制容量/计算与训练条件 |
| 预测动作输入有益 | 待验证 | Shared-GT vs Pred-Detach；控制 timestep/weight |
| 动作路径梯度有益 | 待验证 | Pred-Detach vs Full，多训练 seed |
| B 在推理时有用 | Reacher 上观察到特定配置增益 | 独立评测、更多训练 seed、配对统计 |
| B 可以选中好动作 | 已有候选池表；Push-T/Reacher 与 oracle 有明显差距 | Cube 完整池、跨模型公共候选池、正式对照 |
| PO/GF 真正改善执行 | 不由预测 cost 或闭环提升直接推断机制 | 同状态同动作的真实修正、随机方向对照 |
| 物理 probe 解释耦合 | 当前仅说明某些物理信息可读、预测 latent 读出有差距 | 多训练变体、逐物理量单位、与决策指标关联 |
| 少量修正能替代 CEM | 保留为问题 | 同成功率延迟比较和同时间预算成功率比较 |
| fast/硬件效率 | 尚不写加速倍数 | 独占或低干扰计时、p50/p95、显存、训练成本 |

Full 不优于 Detach 时可以采用 Detach；Shared-GT 最好时可以采用更简单训练。当前初稿不预设 Full > Detach > Shared-GT 的单调顺序，也不预先声称“双方增强”。

## 4. 数值与来源

结果表直接读取 JSON 生成，未将对话中的数字当作唯一依据；来源哈希冻结于 `evidence_snapshot.json`。本次没有重跑训练或 GPU 评测。

| 内容 | 来源 | 范围 |
|---|---|---|
| 主表 11 配置 × 4 任务 | `outputs/round5/phase1_seed3072_legacy/analysis/analysis.json` 的唯一匹配行 | 单训练 seed3072，epoch10，每任务固定 50 个开发样本 |
| 三任务候选池表 | `outputs/round5/phase1_5_seed3072_legacy/diagnostics/summary_{task}.json` | 每任务 50 状态 × 6 flow steps × 256 候选；不是闭环主表 |
| 诊断完成度、probe MAE、计时缺口 | [2026-09-24 20:31 中期报告](../../docs/report/round5/round5_phase1_5_interim_report.md) | 带时间快照的阶段材料；后续变化不会自动改写稿件 |
| checkpoint/cohort/inference 参数 | `config/round5/phase1.json` | 固定历史 Phase1 口径 |
| A/B loss 与 mixing | `source/policy/fast_lewam.py`、`config/train/round4_ab.yaml` | 区分配置、实际训练历史与正式消融 |
| PO/GF | `source/model/fast_lewam/jepa.py` | RMS 梯度归一化、RMS 位移约束、单步等价 |
| 物理诊断距离 | `source/common/round5_phase1_5.py` | 按实际字段及阈值记录，不替换 success predicate |

解释边界：

- 旧稿的 revised 200-episode 历史表不与当前 legacy_50 合并。
- 数千超参数条件仍反复使用同一开发 cohort；最高成功率不是无偏 final 结果。
- 候选池表使用六种 S 的等权平均；300 个 state×S 行来自 50 个初始状态，bootstrap 按状态聚类，不把候选视为独立统计样本。
- 候选池成功聚合与闭环成功不同；不能把 Reacher 55% 与闭环 90% 直接解释为同一指标的变化。
- 候选池 oracle 是池内/诊断时域上界；不是长期最优策略。
- Phase1 Reacher CEM 与 P0/P3 的裁剪处理不同，正文已披露；正式比较需统一动作语义。
- MLP 三个读出器 seed 不代表三个世界模型训练 seed。
- Probe 数据划分排除评测轨迹不代表原主模型训练已经排除这些轨迹。
- 当前 checkpoint 类还分配未使用的 D/E 参数；不可只统计活跃头便宣称总参数节省。

## 5. 文献边界

2026-09-24 核对了以下原始 arXiv 页面：标题、作者、版本和摘要层面的主张。本文新写的近邻比较仅限这些可核实的大方向，**不是已完成全文 novelty audit**。

| 工作 | 本次读取的版本 | 用途 |
|---|---|---|
| [LeWorldModel](https://arxiv.org/abs/2603.19312) | v3；本地既有实现审计可能基于较早版本 | 联合表征/动力学、SIGReg、物理 probe |
| [Fast LeWorldModel](https://arxiv.org/abs/2606.26217) | v1 | action-prefix 并行预测已有先例 |
| [LeFlow](https://arxiv.org/abs/2608.24855) | v1 | latent proposal、IDM、frozen WM selection |
| [FlowMPC](https://arxiv.org/abs/2606.16286) | v1 | flow policy 与 WM/MPPI 组合已有先例 |
| [Fast-WAM](https://arxiv.org/abs/2603.16666) | v2 | 区分训练价值与测试时未来生成 |
| [Faster-WAM](https://arxiv.org/abs/2608.04404) | v1 | 保留未来条件并提高推理效率已有先例 |
| [tau0-WM](https://arxiv.org/abs/2606.01027) | v2 | 统一生成、模拟、评估和修正已有先例 |

Diffusion Policy、DINO-WM 与 Flow Matching 沿用已有 BibTeX；正式投稿前统一全文引用版本及发表信息。新文献作为 arXiv 预印本引用，没有编造会议录用状态。文献摘要可支持高层定位，具体机制差异和排他性创新表述必须待全文核验。

## 6. 接下来如何补成投稿稿

1. 完成 guidance 与 Cube 候选池验收，先解释排名与真实修正，避免仅凭 predicted cost 下降选型。
2. 补训练消融：A-only、B-only、Shared-GT、Pred-Detach、Full、共享 encoder 双 DiT；独立 A+B 复用单任务模型。
3. 冻结一个主推理方案及开发集选择规则；P3 与 P0-PO 优先比较，多步 GF 为对照，CEM 保留合理预算网格。
4. 完成独立 final、多训练 seed、非饱和条件与公平延迟，再填写两处正文 TODO。
5. 根据证据重写摘要末段、Introduction 最后一段和 Conclusion，贡献列表不能比实验更强。
6. 完成全文文献审计、清理匿名信息、使用目标届次正式模板、检查正文八页。

## 7. 编译与交付

`main.tex` 和 `supplement.tex` 分别编译正文与工作补充材料。目标年份已改为 2027，但现有 `cvpr.sty` 仍来自 2026 模板，不能据此声称完成 2027 模板合规检查。

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

本机未发现 latexmk/pdflatex/xelatex/tectonic，故不声称 PDF 编译成功或满足八页限制。当前进行静态依赖、引用、环境/括号与数值来源检查。Overleaf 压缩包只包含排版所需文件，不包含内部会话说明、绝对产物路径账本或旧稿备份；仍是带 TODO 的工作稿，不是投稿包。
