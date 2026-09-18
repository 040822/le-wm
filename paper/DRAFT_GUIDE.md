# Fast-LeWAM 论文工作稿

版本：v0.1，2026-09-12。英文正文已经写入 LaTeX；这是可随实验更新的初稿，尚不是投稿稿。

## 1. 论文定位

暂定标题：**Fast-LeWAM: Shared Latent Modeling for Action Generation and Planning**

中文工作标题：**Fast-LeWAM：面向动作生成与规划的共享隐空间模型**

主问题：同一模型能够直接生成动作，也能够预测动作后果，那么增加 latent 路径生成与逆动力学解码后，何时能获得值得付出额外计算的规划收益？

叙事顺序：

1. 动作策略与世界模型分别提供直接控制和后果预测。
2. 已完成实验表明，联合训练和 actor warm-start 的收益依赖任务。
3. 这提示我们分开研究候选生成和候选选择。
4. 在共享 A/B 主干中加入 D/E，用同一 B 比较动作候选与路径解码候选。
5. 用最终闭环成功率和完整规划开销回答这种组合是否有价值。

这是暂定写作方向。若第四轮 final 无收益，保留负结果并重新评估主问题；不要为了维持标题改用中期峰值。

“Fast”沿用项目名称，不构成已验证的相对速度主张。在线适应在正文讨论中保留，尚不列为贡献，也没有因写作而修改任何实验计划。

## 2. 中文摘要对应稿

隐空间世界模型通过预测候选动作的后果支持视觉控制，生成式策略则直接根据观测和目标生成动作。将两种能力结合起来，引出了一个实际问题：共享模型能否同时支持直接控制与显式规划，以及何时规划能够改善模型自身生成的动作？

我们通过 Fast-LeWAM 研究这一问题。该模型共享视觉编码器和 Transformer，支持动作生成、因果未来预测、隐空间路径生成以及逆动力学解码四种模式。模型可以直接生成动作候选，也可以先生成连接当前状态与目标的隐空间路径，再将路径解码为动作。两种候选均由同一动力学预测器根据预测终点与目标之间的距离评分，从而允许在同一训练模型内比较不同候选空间。

已完成的早期动作—动力学模型实验表明，联合训练及 actor 初始化规划的收益具有明显任务依赖性。在修订后的评测协议下，动作辅助监督在三个训练种子上改善 Cube 的规划表现，却降低 Push-T 的表现。这些结果说明，评估共享世界动作模型时，应分别考察成功候选的生成和选择。

工作稿状态：四模式模型目前已有中期开发结果；最终性能、效率和匹配训练对照尚待完成。最终摘要需要据真实结果改写最后一段。

## 3. 文件与阅读顺序

| 文件 | 内容 | 当前完成度 |
|---|---|---|
| [main.tex](main.tex) | 主入口、标题、文献、章节组合 | 已接入真实正文 |
| [draft_abstract.tex](sec/draft_abstract.tex) | 英文摘要 | 已写；末段等待 final |
| [draft_intro.tex](sec/draft_intro.tex) | 动机、问题、方法概览、证据范围 | 已写 |
| [draft_related.tex](sec/draft_related.tex) | LeWM、DINO-WM、LeFlow、生成策略、WAM | 初版完成；需扩充全文比较 |
| [draft_method.tex](sec/draft_method.tex) | A/B/D/E、公式、mask、训练、推理 | 已与当前实现交叉核对 |
| [draft_experiments.tex](sec/draft_experiments.tex) | 历史结果、协议、final 表占位、实验问题 | 已写；所有未出结果标记 TBD |
| [draft_discussion.tex](sec/draft_discussion.tex) | 局限、在线学习位置、结论 | 已写 |
| [draft_appendix.tex](sec/draft_appendix.tex) | epoch 2 与后续附录材料 | 已写；默认不编入主稿 |
| [references.bib](references.bib) | 7 项初始参考文献 | 作者、标题与 arXiv 记录已核对 |
| [draft_architecture.tex](fig/draft_architecture.tex) | 两种 proposal 与共用 B 的结构示意 | 原生 LaTeX 图，已接入方法章节 |

原模板的 0_abstract.tex、1_intro.tex、2_formatting.tex、3_finalcopy.tex、main.bib 和示例 teaser 保留作模板参考，主稿不再引用它们。preamble.tex 中的红色 TODO 命令提示尚待填充的材料。

## 4. 可以写的主张与需要补的证据

| ID | 主张/问题 | 当前证据 | 还缺什么 | 正文位置 |
|---|---|---|---|---|
| C1 | 一个共享主干提供 A/B/D/E 接口 | 代码与 epoch 2 闭环运行 | final 能力验收、参数量、稳定训练诊断 | Method |
| C2 | A 辅助监督对 B 的影响依赖任务 | Cube/Push-T 三种子，新协议 | 可补更多任务；不能推广为普遍增益 | Historical study |
| C3 | latent 候选优于 action 候选 | 未确认 | 同 checkpoint 的 P4 vs P3，final 配对结果 | Four-mode study |
| C4 | 生成后选优比 CEM 更划算 | 未确认 | P4/P3 vs P2/P1 的完整计时与成功率 | Efficiency |
| C5 | 新增 D/E 训练改善 A/B | 当前研究不能识别 | 同数据、batch、训练预算的 AB vs ABDE | 后续新实验 |
| C6 | 参数共享优于独立模块 | 当前研究不能识别 | 匹配的独立 D/E 对照及成本记录 | 后续新实验 |
| C7 | B 选优能识别有效路径 | 未确认 | P4 vs P4-first，固定状态真实候选执行 | Diagnostics |
| C8 | 在线真实交互优于离线续训 | 正式结果未完成 | Round 3 三组对照及独立测试 | 暂放 Discussion |

C5/C6 是主张成立所需的后续建议，并不是当前 Round 4 已登记实验。现有计划明确先完成 ABDE 首轮。此次写作不授权或启动任何额外训练。

## 5. 结果与来源账本

优先级：可追溯原始结果与配对身份 > 汇总 CSV > 正式实验报告 > 早期讨论。协议、权重、代码或 cohort 不同的记录不能只按实验名称合并。

| 正文材料 | 来源 | 解释范围 |
|---|---|---|
| 新协议四任务历史主表 | [新协议报告 §4/§11](../docs/report/round3/phase1/round3_new_protocol_experiment_report.md)、[扩展 CSV](../outputs/round3/extended/extended_eval_summary.csv) | 早期 A/B 模型；不代表 ABDE |
| Cube/Push-T E3−E1 多种子 | 同报告 §4.1 | 样本 SD，不是 CI |
| E5−E4 Push-T 多种子 | 同报告 §4.1、扩展 CSV | 应使用同一扩展矩阵来源 |
| E6 比较 | 同报告 §4.2 与扩展矩阵 | 单主权重；初始化改变，未新增训练 |
| Reacher 候选 coverage/top-1 | [第一轮报告 §7.2](../docs/plan/round1_experiment_report.md) | 旧协议、小型 panel；不能充当新协议配对解释 |
| streaming pilot 86→78 | [7–8 月总报告 §5.11](../docs/summary/round1_round2_summary_v2.md) | 历史有限适配；不是完整在线学习 |
| ABDE 实现与预算 | [第四轮计划](../docs/plan/round4_experiment_plan.md)、[模型](../source/model/fast_lewam/round4.py)、[训练损失](../source/policy/fast_lewam.py)、[配置](../config/train/round4_abde.yaml) | 区分计划预算与实际完成 |
| ABDE epoch 2 | [中期报告](../docs/plan/round4_intermediate_epoch2_report.md) | 开发集、单种子，默认放工作附录 |
| 数据暴露解释 | [Phase 1.5 决策](../docs/report/round3/phase1.5/round3_phase1.5_decision.md) | 当前 R4 为探索性；不能宣称严格未见数据 |
| 参数与缓存修复 | [第一轮报告 §9](../docs/plan/round1_experiment_report.md) | 参数为旧 AB 模型；加速是修复前后比较 |

### 已发现的来源差异：Push-T E5 B

9 月 11 日报告主表列为 **89.5%**。扩展评测 CSV 及对应 result.json 则为 **89.0%**：

~~~text
outputs/round3/extended/evals/round1_pusht_E5_0803_e5_cross_head_grad/results/pusht/e5_fast/round3_revised/stage_b/final/138501dd8c2d88990074da59039aa092d4269e841c7114fe68ae49fdd36cc47b/result.json
~~~

初稿采用扩展矩阵的 89.0%，因此该来源下 E6−E5 为 93−89=4 pp，E5−E4 为 89−85.5=3.5 pp，与扩展多种子统计一致。正文保留显式 provenance TODO；这不是把原 canonical 结果判错或覆盖。

提交前需要检查两份运行的 cohort、checkpoint、代码与成功向量，解释差异，再固定论文引用的运行身份。本轮没有修改原报告或实验产物。

### 方法细节的写作边界

- R4 A/B 当前使用 legacy token/timestep，不能写成 Round 3 新选的 physical_time_type。
- R4 新增 learned mode embedding；不能声称与旧 E5 完全同一参数结构。
- E 的 action query 包含相邻状态与状态差特征，并能读取完整路径。
- D 的插值目标和 velocity 目标停止梯度，但干净起点/目标仍向 encoder 传梯度；E 的路径输入也保留梯度。
- E5 预测动作分支仍监督专家 future，不能写成真实反事实后继监督。
- B 的“验证”是共享模型的预测评分，不是独立物理验证或安全保证。
- 当前 5 个 block × 每 block 5 步，并且执行全部 5 个 block 才重规划。不能误写成每个环境步重规划。
- 旧 audit 和新版 runtime 对 Push-T state 字段的细节需保持可追溯；写物理阈值时同时记录实际谓词版本。

## 6. 边实验边写作的更新顺序

1. **先收口 final dev。** 完成 P0/P0-shuf/P1/P2/P3/P4 与开发专用 P4-first；输出同 checkpoint 的逐任务配对差。
2. **按既有 Round 4 门槛决定是否扩种子。** 不用 final test 决定是否补种子，不用 epoch 2 代替 epoch 10。
3. **完成 final 与统一计时。** 填主表，再写结果段。结果为负时，正常写负结果与适用边界。
4. **逐项回填主张。** P4 vs P3 只能支持候选路线比较；共享、训练迁移需要各自匹配对照。
5. **最后改摘要和结论。** 用真实主结果替换工作状态段，删除不成立的贡献；不要只删除 TODO 而保留隐含承诺。

每次加入结果，建议同时记录：run ID、checkpoint/epoch/hash、train seed、protocol/cohort/hash、数据暴露、成功数/episode 数、配对翻转、成本计时边界。

## 7. 图表安排

| 图表 | 应表达什么 | 需要的材料 |
|---|---|---|
| 方法图（结构示意已写） | 一套 encoder/DiT、四个 mode、两种 proposal、共用 B | 后续完善 token/mask 与梯度边界视觉表达 |
| 历史结果表（已有） | 联合训练/warm-start 的任务依赖 | 新协议矩阵 |
| ABDE 主表（TBD） | 同一 checkpoint 的推理方式差异 | epoch 10 200-episode final |
| 成功率—延迟图 | 额外 planning 是否值得 | 独占设备的统一计时 |
| 诊断图 | 好候选存在但选不出来，或生成路径不可执行 | 固定状态、真实 rollout、候选评分 |
| 训练对照表（条件后续） | D/E 监督、共享参数的独立作用 | 新登记的匹配训练组 |

epoch 2 不放主性能表，不把历史 AB 参数量当成 ABDE 参数量，不把缓存修复倍率当成相对 LeWM 的方法速度。

## 8. 文献与版本说明

本轮已核对以下原始记录，BibTeX 位于 references.bib：

- [LeWM](https://arxiv.org/abs/2603.19312)：本地 PDF 是 v1；线上已有 v3。当前引用预印本，具体实现审计仍以本地固定版本为准。
- [LeFlow](https://arxiv.org/abs/2608.24855)：最直接相关工作。路径生成、IDM 解码、world-model 评分不能写成本项目首次提出。
- [DINO-WM](https://arxiv.org/abs/2411.04983)。
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)：当前作者表对应扩展版本，投稿前统一版本与发表信息。
- [Flow Matching](https://arxiv.org/abs/2210.02747)。
- [Fast-WAM](https://arxiv.org/abs/2603.16666)。
- [Faster-WAM](https://arxiv.org/abs/2608.04404)：相关工作须覆盖高效 future conditioning，不能将保留未来信息本身称为新颖性。

这是初始文献集合，并非完整 novelty survey。正文已避免 first、SOTA、通用收益等未证实主张。

## 9. 编译与交付状态

在安装了 TeX Live 和 latexmk 的环境，进入 paper/：

~~~bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
~~~

需要工作附录时，取消 main.tex 末尾三行 \clearpage、\appendix、\input{sec/draft_appendix} 的注释。

本地当前没有 latexmk/pdflatex/xelatex/lualatex/tectonic，因此本轮只做静态引用、输入文件、环境与括号检查；尚未确认 PDF 编译、分页、溢出或页数。也可以将 paper/ 导入 Overleaf，以 main.tex 为主文件编译。

仓库现有样式来自 CVPR 2026 模板，保留其年份用于草稿排版；目标届次、paper ID、真实作者与投稿时的模板要求尚未填写。主稿暂用 Anonymous Authors。DRAFT ID、所有 TBD/TODO 和工作稿状态声明必须在投稿前处理。旧 .github 模板工作流未在本次修改或触发。
