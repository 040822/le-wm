# CoWM 初稿写作与证据说明

更新：2026-10-01。目标：CVPR 2027 工作稿。本次改为可填结果的英文初稿：主线与六组实验问题固定，已有参考结果保留，未完成数值使用统一字段。填充操作见 [RESULTS_FILL_GUIDE.md](RESULTS_FILL_GUIDE.md)。

范围以现有证据和 Phase1.7 已规划主矩阵为基础。上一轮的 24–48 run 全矩阵不作为论文默认执行前提；Pred-Detach、双 DiT、进一步匹配 A-only 及额外物理量是条件性扩展。缺失时收窄主张，不因占位符自动新增实验。

## 当前初稿修订（2026-10-01，Fast CoWM）

本节覆盖下文旧稿的章节布局和命名说明；历史证据、来源及实验边界仍保留。修改前稿件备份见 `docs/archive/pre_fast_cowm_*.zip`。

- 标题统一为 **CoWM: Coupled Latent World Model for Fast Action Selection and Refinement**。
- Introduction 按最新讨论组织为五个正文段落及三条贡献：背景与时间预算、搜索与生成的联系、耦合思想、三类训练机制与并行预测、部署及预算评价。
- 本次使用 edit-article 的组织原则；采用用户确认的完整论文段落结构，覆盖技能中 240 字符短段落的通用建议，既定章节不再重复确认。
- 三类耦合统一为 parameter-sharing / action-input / action-mediated gradient coupling。Separate-Recorded 与 Shared-Recorded 的 Recorded 指 B 的记录动作输入；不再用易混淆的 GT。
- `sec/3_method.tex` 保留现有公式，增加决策预算定义与两条梯度路径。生成动作估计仍由记录未来 latent 监督，不宣称已有反事实监督或训练部署分布完全对齐。
- 主文实验组织成五组表：系统、耦合、排序/修正、推理、预算。物理读出、资源统计及可选真机模板位于 `sec/8_budget_extensions.tex`，由 `supplement.tex` 接入。
- 各诊断表是每任务实例的模板，不能把异质物理单位合并平均；最终按实际完成任务复制/排版，并给出 seed、counts、区间和配置。
- `draft-*` 是本轮新占位字段，仍集中定义在 `results.tex`。旧字段及历史数值未删除；主文不再用固定 checkpoint 历史结果替代最终比较。
- 当前没有声称 10M 总参数、训练更省、同预算普遍更优、所有任务替代 CEM 或完成真机。摘要和引言的结果句明确留有 TBD。
- 现有参考文献沿用已有稿件；本轮没有新增文献审计。DeWM 版本与引用仍需补齐，不能将匿名稿编造为已发表条目。
- 本环境未发现 TeX 引擎；源码一致性检查不等于 PDF 编译成功，页数及浮动体布局仍需在 Overleaf/TeX Live 检查。

## 1. 会话依据与本次取舍

来源会话：`01a0ba23-c649-7b90-8acc-96716f894f9f`（2026-09-19 起）。读取了本地保存的用户与助手消息，而非启动交互式 `codex resume`。会话中的用户决定依次包括：

- 2026-09-21：主模型收敛为 Shared DiT，主要任务为 pred action 与 parallel pred future；C/D/E 退居针对性消融；在线训练留给后续研究。
- 贡献围绕 coupled、fast、probe 收敛；需要解释为什么使用 R4-AB，比较 P3 与 P0-PO/GF 能否减少 CEM 搜索需求。
- 用户随后决定沿该方向推进，并委托文献研究；后续讨论明确将耦合拆为参数、动作输入、动作路径梯度三种联系。
- Phase1.5 定位为固定已有 checkpoint 的推理诊断，不能凭它证明训练耦合的因果收益；保留原有 50 个样本，并允许冻结表示上的读出器。

本次还对照了 [本地 CoWM 笔记](Couple_lewm.md)、[Phase1.6 报告](../../docs/report/round5/round5_phase1_6_report.md)、[Phase1.7 报告](../../docs/report/round5/round5_phase1_7_report.md)、[Phase1.7 计划](../../docs/plan/round5_phase1_7_plan.md) 和实际结果 JSON。

暂定标题：**CoWM: Coupled Latent World Model for Fast Action Selection and Refinement**。
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
| `sec/4_experiments.tex` | 六组固定问题：系统表现、训练贡献、排序/修正、物理读出、推理策略、成功率/延迟 |
| `sec/5_discussion.tex` | 适用范围、局限、结论 |
| `supplement.tex`、`sec/6_appendix.tex` | 单独编译的补充材料、对照范围、epoch-10 开发检查、在线扩展与复现细节 |
| `sec/7_reference_results.tex` | 从修改前实验章保留的完整 Phase1/1.5/1.6 参考结果 |
| `fig/draft_architecture.tex` | 原生 LaTeX 结构示意，尚可继续美化 |
| `results.tex`、`tables/final_*.tex` | 集中填写的结果字段和六张主表；未知为蓝色 TBD，未测条件扩展为横线 |
| `tables/phase1_7_snapshot.tex` | 从 epoch-10 配对 JSON 提取的开发检查，非 final 结果 |
| `docs/evidence_snapshot.json`、`docs/phase1_7_evidence_snapshot.json` | 历史与本次新增引用的路径、数值、SHA256；内部账本 |

旧稿已备份到 `docs/archive/pre_cowm_20260924.zip`。本次改稿前的 11 个文件另存 `docs/archive/pre_fillable_20261001.zip`，保留原有未提交内容。编号章节布局不变，详细历史实验转入单独编译的补充材料。

## 3. 主张—证据—缺口

| 主张/问题 | 当前正文如何写 | 尚需证据 |
|---|---|---|
| 共享 A/B 支持动作生成和并行未来预测 | 已实现的方法事实 | 正式版本资源统计 |
| 参数共享优于分离 predictor | 未宣称成立，条件性机制问题 | 双 DiT 未完成时不写共享收益；同时报告容量/计算差异 |
| 预测动作输入有益 | Joint vs Recorded-control 只检验输入与动作梯度的联合改变 | 单独归因须有完整 Pred-Detach 并控制 timestep/weight；缺失时只写联合效应 |
| 动作路径梯度有益 | 冻结表征短训未识别可靠收益，完整训练归因未完成 | 完整 Pred-Detach vs Full 若未完成，不写梯度收益 |
| B 在推理时有用 | Phase1.6 新起点上 P3 相对匹配 Random-64 的等任务宏均值高 2.0 pp，差异主要来自 Reacher；单训练 seed | 多训练 seed、真正独立的确认集与任务交互复验 |
| B 比独立 LeWM 更会排序 | 特定 epoch-2 共池连续指标偏向 B；epoch-10 seed4096 PushT dev 未复现，闭环优势未建立 | 收齐已规划主矩阵、共池 top-1/regret/覆盖和逐 seed 闭环；不跨 cohort 拼接 |
| PO/GF 真正改善执行 | 同状态物理执行和等 RMS 随机方向显示 Push-T 局部方向收益；Reacher 与跨任务闭环未确认普遍收益 | 多训练 seed 下测试局部收益是否稳定传到闭环控制 |
| 物理 probe 解释耦合 | Cube 当前只读 block position，不能说明速度或姿态改善 | 仅填已测同协议数据；其他物理量是条件性扩展，不作为成稿前提 |
| 少量修正能替代 CEM | Phase1.6 预设逐任务非劣效门槛未通过；宏均值过线不能替代 Reacher/Push-T 任务界 | 动作语义完全一致的预算扫描、同时间预算成功率和多 seed 评测 |
| fast/硬件效率 | Phase1.6 已测固定 checkpoint 下隔离 `policy.get_action` p50/p95 与 batch-50 吞吐；不能外推为端到端或跨硬件加速 | 不同硬件、含观测/环境/通信的端到端基准，以及完整训练成本 |

Full 不优于 Detach 时可以采用 Detach；Shared-GT 最好时可以采用更简单训练。当前初稿不预设 Full > Detach > Shared-GT 的单调顺序，也不预先声称“双方增强”。

## 4. 数值与来源

结果表直接读取 JSON 生成，未将对话中的数字当作唯一依据；Phase1.6 的来源哈希保存在 `phase1_6_evidence_snapshot.json`，并并入 `evidence_snapshot.json`。Phase1.6 使用一个固定训练 checkpoint；其短程训练臂各 1,000 次更新，不代表完整训练消融。

本次新增引用单独记录于 `phase1_7_evidence_snapshot.json`：epoch-10 dev P3、PushT 固定池及在线结果均核对实际 JSON。开发检查不填多 seed 最终表；在线结果单列补充材料。

| 内容 | 来源 | 范围 |
|---|---|---|
| Phase1.6 主闭环表和配对效应 | `outputs/round5/phase1_6_seed3072/analysis/paper_summary.json` 与 `conditions/` | 每任务 100 个新起点、两个推理 seed、9 种方法；每格 200 episodes |
| Phase1.6 动作后果/候选/引导诊断 | `outputs/round5/phase1_6_seed3072/mechanism/` | legacy cohort 的机制分析，不能与新起点闭环表合并 |
| Phase1.6 训练干预 | `outputs/round5/phase1_6_seed3072/short_train/` 与对应训练评测条件 | Push-T/Reacher；Continue、Detach、Recorded 各 1,000 updates；Frozen 复用确认集结果 |
| Phase1.6 推理计时 | `outputs/round5/phase1_6_seed3072/timing/summary.json` | 每方法每任务 batch-1/batch-50，记录 GPU 与主机负载边界；只用无干扰条件作速度主张 |
| Phase1.7 epoch-10 dev P3 | `outputs/round5/phase1_7/epoch10_main_dev_p3/analysis/{pusht,reacher}_paired.json` | seed4096，各 100 配对起点；区间跨零，非独立确认 |
| Phase1.7 epoch-10 PushT dev 固定池 | `outputs/round5/phase1_7/epoch10_main_dev_fixed_pool/pusht/pusht_phase1_7_dev_16027_v1/summary.json` | 100 状态 × 64 候选，连续排序未复现早期优势 |
| Phase1.7 在线锁定闭环 | `outputs/round5/phase1_7/locked_confirmation_closed_loop_v1/analysis.json` | epoch-2 actor，各 200 起点；新增真实分支数据，单列报告 |
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
- Phase1.7 confirmation 已用于 epoch-2/在线结果，后续沿用须注明复用。现有训练 episode 的新起点不能称训练未见。
- 只完成两个训练 seed 就报告两个；不足以判断的效果保留不确定性，不以第三 seed 作为所有写作的前提。

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

## 6. 有限范围内完成稿件

1. 已按 Phase1.6 的延迟—成功率结果将 P3 定为主推理候选、P0 定为低延迟备选；保留“未通过逐任务 CEM 非劣效门槛”的边界。
2. 优先收齐 Phase1.7 已规划主矩阵的完整终点和逐 seed 结果。Recorded-clean 未完成时明确缺失，不用短训替代。
3. 填主表时同步记录数据暴露、复用、动作语义和预算。只有确实独立的新评测才称独立确认；不为这一称呼自动启动全矩阵重训。
4. Pred-Detach/双 DiT/A-only、更多物理量、任务和真机不因占位自动启动。已有证据不足时限制对应主张；已规划主矩阵收齐后先判断继续或收尾。
5. 完成全文文献与 novelty 审计、匿名信息清理、CVPR 2027 正式模板检查及正文页数检查；现稿不等于投稿稿。

填数后仍需核对解释：正、零附近及负效果均允许。摘要和结论由实际效应、区间及对照范围决定，不能只换数字而保留预设正结论。详见 [RESULTS_FILL_GUIDE.md](RESULTS_FILL_GUIDE.md)。

## 7. 编译与交付

`main.tex` 和 `supplement.tex` 分别编译正文与工作补充材料。目标年份已改为 2027，但现有 `cvpr.sty` 仍来自 2026 模板，不能据此声称完成 2027 模板合规检查。

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

本机未发现 latexmk/pdflatex/xelatex/tectonic，故不声称 PDF 编译成功或满足八页限制。当前进行静态依赖、引用、环境/括号与数值来源检查。Overleaf 压缩包只包含排版所需文件，不包含内部会话说明、绝对产物路径账本或旧稿备份；仍是带 TODO 的工作稿，不是投稿包。
