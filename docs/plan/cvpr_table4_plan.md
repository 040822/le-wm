# CVPR Table 4：固定模型下的推理策略实验方案

## 依据与编号

编写日期：2026-10-05。按当前论文的 [实验设计](../../paper/docs/Couple_lewm.md)、[填写指南](../../paper/docs/RESULTS_FILL_GUIDE.md) 和 `paper/sec/4_experiments.tex` 的编号组织：Table 2 耦合、Table 3 排序/修正/物理量探测、Table 4 推理、Table 5 预算/效率。Table 3c 优先完成 Cube 全物理量探测；9 月 19 日旧版的 flow-step 消融列为 Table 5 低优先级可选附录。用户已确认采用当前论文版编号。用户已确认 Table 2 采用五个核心训练臂、三个训练 seeds；下述细化参数在实施前写入冻结配置。

结果依据：[Table 1 完整报告](../report/cvpr/cvpr_table1_report.md)、[审计报告](../report/cvpr/cvpr_table1_paper_audit_20261005.md)、[Table 1 计划](cvpr_table1_plan.md)。实际运行协议以 `outputs/cvpr/table1/v1/frozen_config.json` 为准，已完成 948 个成功率单元和 158 个计时条件。

## 复用决定与执行优先级

**可以直接复用 Table 1 的完整实验结果，首版 Table 4 不新增训练、闭环评测或计时。** 复用范围包括主表背后的18种 main 推理协议，而不只论文 Table 1 展示的六行。先输出已完成方法的成功率/计时对照，Random-64 与同 A+LeWM 在 Table 3 桥接完成后补入，不作为本表首版的前置条件。

同一批 GPU6 RTX4090 已完成的计时可直接组成内部一致的比较，无需为了换表编号重测。只有将新设备/新会话的新增条件并入同硬件比较时，才按下文规则补共同测时；不要求先把旧矩阵全部重跑。

## 目标与主要发现

固定 Table 1 的四任务 R4-AB、seed3072、epoch10，比较生成、搜索、选择及修正的不同用法。Table 1 已完成全部 18 种推理方式及裁剪测试，主体可以直接复用；本表不需要重新训练。

现有同跨度主版本宏平均：P0=94.17%、P1=82.42%、P2=91.67%、P3=96.25%、P0-GF-L=95.58%、P0-PO-L=94.50%、P3-PO-L=96.42%。P2 未稳定优于 P0，较大修正预算未稳定更好；因此保留全部预设方法，不围绕最优成绩重新命名。

## 主表矩阵

| 方法 | 候选/初始化 | S | CEM（samples×iterations, elites） | 梯度更新总数 | 来源 |
|---|---|---:|---|---:|---|
| P0 | A 单候选 | 2 | — | 0 | Table 1 |
| Random-64 | A 生成 64 条后均匀随机选 1 条 | 2 | — | 0 | Table 3 新增桥接 |
| P1 | 随机初始化，CoWM-B CEM | — | 300×30,30 | 0 | Table 1 |
| P2 | A 初始化，CoWM-B CEM | 2 | 300×30,30 | 0 | Table 1 |
| P3 / Selection | A 64 条，B rerank | 2 | — | 0 | Table 1 |
| P0-PO-L / Refinement | A 单候选后优化 | 2 | — | 2 | Table 1 |
| P0-GF-L | A 单候选生成中引导 | 2 | — | 2（每 Euler 步 1） | Table 1 |
| P3-PO-refine-L | 先选 1 条，再优化 | 2 | — | 2 | Table 1 |
| P3-PO-L | 64 条全部优化，再选 | 2 | — | 2/候选 | Table 1 |

P3-PO-L 与 P3-PO-refine-L 分开，不把“每个候选 2 次更新”写成与单候选 PO 等计算量。保留 Table 1 的步长=0.01、最大 RMS 偏移=0.2、CEM var_scale=1、P2 初始化缩放=1，以及 history/cache/chunk/solver batch 的实际设置。

主表列：方法、N/S/I/K、四任务 SR、宏平均 SR、四任务等权平均完整重规划 mean(ms)；逐任务 mean/P50/P95 和显存放补表。CEM 的 N 是每轮 samples，其余 N 是候选池大小。P1 的 S 为不适用。

同 A+LeWM 的桥接行可放主表下方独立 scorer 面板，使用 Table 3 同一批新结果；它回答更换 scorer 的系统效果，不能归因为参数共享本身。

## 完整补表及分析

完整展示 Table 1 的 18 种 main 配置与 18 种 crop 配置，保留轻量/较大预算。GF-H 总 K=10，PO-H 总 K=5，两者不称等预算。Selection† 单列 Reacher 10/10，不加入常规协议的主排名；其它三任务 alias 明示。

主要配对为 P3−Random-64、P0-PO-L−P0、P0-GF-L−P0、P2−P1；四任务共 16 项构成 Holm 族。先报告逐 seed 差、配对区间，再解释方向。Table 1 已经观察过的比较标注为事后扩展分析；新的 Random-64 在运行前固定抽样规则。P3−P0 同时改变生成候选数与选择，不能作为纯评分收益。

Random-64 保留生成全部 64 条的实际成本，均匀选择 RNG 独立于 proposal RNG；不能通过只生成一条动作来虚报 Random-64 的延迟。与 P3 在同一输入上的原始噪声和候选逐项核对；若旧 P3 未保存足够身份，做固定输入配对验收，不能直接声称旧闭环分支拥有相同池。

## 工作量、计时与验收

常规 main 18×24=432 个旧单元，crop 同为 432，短时域为 12 个旧 Reacher 单元；复用总共 876 个 CoWM 单元。主表只展示其中八种已有条件加 Random-64，不重跑已通过审计的数据。

后续桥接扩展的独立新增仅 Random-64 四任务×六 seeds=24 单元/1,200 episodes，由 Table 3 负责运行，本表引用；同 A+LeWM 的另外 24 单元也归 Table 3，禁止跨表重复派发。没有新增训练。计时每新增方法四任务×250 样本=1,000 次有效调用，外加每任务 10 次预热；同样由共用测时清单去重。

计时严格沿用 Table 1 的 batch=1 初始完整重规划边界：CPU 原始输入到 CPU 可执行动作，CUDA 同步，包含 encoder/生成/评分/反传/转换，排除环境、加载和归档。50 起点×5 重复，目标缓存/动作缓存/CEM 残余初始清空，参数固定但 PO/GF 开动作梯度。成功率/计时动作 parity 使用 atol=1e-6、rtol=1e-5。

Table 1 实际在 GPU6 RTX4090 计时；本轮默认 GPU0–3。新旧计时若设备或环境不同，主效率对比必须在同卡同会话补测所展示的方法，旧耗时只能标成历史参照。与 [Table 5](cvpr_table5_plan.md) 共用测时矩阵，不重复采集。

先建立旧结果与时延索引，直接汇总已有配对差及完整附表并交付首版；Random-64/同 A+LeWM 随 Table 3 后续桥接补齐。交付需能从原始 result/trace 重建所有单元，并区分“未测”“失败”“不适用”。本表固定旧主模型，不按 Table 2 新训练成绩更换 checkpoint。

## 公共协议与资产规则

- 四任务固定为 TwoRoom、PushT、Reacher、OGBench-Cube；闭环沿用 Table 1 的 24 份样本清单，评测 seeds 为 `42,100,2026,3407,1234,4444`，每 seed 50 episodes。起点、目标、环境成功判据和环境 RNG 均沿用冻结配置，不改用 Round5 的过滤协议。
- 默认目标 offset=25、总执行预算=50、生成/评分/执行=25/25/25，5 个 block，每 block 5 步。Euler、S=2；P3 默认 N=64；FP32，关闭 autocast/TF32/compile；无额外候选裁剪，保留环境原生动作处理。例外逐条件登记。
- Reacher 10/10 的 Selection† 仅为单列部署参考，不混入同跨度的机制比较。裁剪条件单列；不按结果为每个方法挑选裁剪或时域。
- 单 checkpoint 结果报告六个评测 seed 的 mean ± sample std（ddof=1）；宏平均先在同 seed 内平均四任务，再跨 seed 汇总。训练重复另外统计，不能把评测 seed 当训练 seed。
- Table 1 样本已用于开发观察，后续复用只能称同 benchmark 扩展，不能称未见确认集。配对差保留逐任务、逐 seed 结果；存在重复起点/同来源轨迹时，区间按来源 episode 聚类，重复出现项共同重采样，不把候选或重复测时当独立样本。
- 只有 checkpoint、样本清单、动作语义、算法参数、代码兼容性及原始结果哈希全部通过审计才复用；建立 `source_refs.json` 指向已有单元，明确 alias，不伪造新运行。旧 Round5 clip、过滤 cohort、短训结果只能作历史参考。
- 新资产放 `outputs/cvpr/tableN/v1`（N 为本表编号），报告放 `docs/report/cvpr/cvpr_tableN_report.md`。保存 frozen config、代码及差异、数据/权重/样本 SHA256、逐 episode 或逐候选记录、失败记录、统计脚本和报告来源；不覆盖已有结果。
- 此次交付为计划，不启动训练或评测。实施时训练仅用 GPU0–3，每条 GPU 命令显式设置 `CUDA_VISIBLE_DEVICES`，先检查空闲显存并保留运行余量。GPU4–7 须有适用的用户明确授权且仅用于低显存评测，10:00–23:00 北京时间尽量避开；Table 1 的 GPU6 使用记录不自动扩展为本轮授权。
