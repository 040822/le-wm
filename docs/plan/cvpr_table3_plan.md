# CVPR Table 3：排序、修正与物理量探测实验方案

## 依据与编号

编写日期：2026-10-05。按当前论文的 [实验设计](../../paper/docs/Couple_lewm.md)、[填写指南](../../paper/docs/RESULTS_FILL_GUIDE.md) 和 `paper/sec/4_experiments.tex` 的编号组织：Table 2 耦合、Table 3 排序/修正/物理量探测、Table 4 推理、Table 5 预算/效率。Table 3c 优先完成 Cube 全物理量探测；9 月 19 日旧版的 flow-step 消融列为 Table 5 低优先级可选附录。用户已确认采用当前论文版编号。用户已确认 Table 2 采用五个核心训练臂、三个训练 seeds；下述细化参数在实施前写入冻结配置。

结果依据：[Table 1 完整报告](../report/cvpr/cvpr_table1_report.md)、[审计报告](../report/cvpr/cvpr_table1_paper_audit_20261005.md)、[Table 1 计划](cvpr_table1_plan.md)。实际运行协议以 `outputs/cvpr/table1/v1/frozen_config.json` 为准，已完成 948 个成功率单元和 158 个计时条件。

## 本轮优先级（用户修订）

**首批新增实验优先做 3c 的 Cube 全物理量 probe**，不等待 Table 2 重训或 3a/3b 候选分支采集。随后补充其它三个任务的 probe，再推进 3a 排序与 3b 修正。Table 4/5 的既有结果整理可同步进行，不占新增训练预算。原“只保留已有 block position、不新增 probe”的范围已取消。

## 问题与既有证据

分开回答“候选池里是否有可成功动作”“B 能否选中它”“预测代价下降是否对应真实执行改善”。Table 1 中 P3 相对 P0 的四任务均值高 2.08 pp，Reacher 高 7.33 pp；P0-PO2 仅比 P0 高 0.33 pp，P3-PO2 比 P3 高 0.17 pp。这些描述性结果不足以证明评分更准或梯度方向更好，不能预设 refinement 必然有效。

主诊断固定 Table 1 的 R4-AB actor/B 和配套 LeWM；Table 2 的 B 臂训练完成后在同池上补充评分与修正，将训练归因与旧 checkpoint 的机制观察分开。

## 3a：同一物理候选池上的排序

每任务使用 Table 1 六个评测 seed 的全部 300 个样本出现项；按唯一来源状态 ID 保存快照，重复项保留映射与原有权重。A 使用 Euler S=2、N=64、生成 25 步，无额外投影。每个状态的动作池只生成一次，保存噪声、物理单位动作、normalizer、actor SHA 和候选索引。每条候选从同一恢复快照独立执行，最多 25 步，保存真实成功事件、有效步数及 5/10/25 步的有效后果。

| 选择器 | 作用 | 是否新增环境回放 |
|---|---|---|
| Uniform Random-64 期望 | 全池成功率/代价均值，避免一次抽样噪声 | 否 |
| Table 1 独立 LeWM | 相同 A、替换评分器的系统对照 | 否 |
| Table 1 CoWM-B | 主排序器 | 否 |
| 物理代价 oracle | 选真实代价最小候选，regret=0 | 否 |
| 成功 oracle | 池内存在成功候选即命中，表示覆盖率 | 否 |
| Table 2 各 B 臂 | 相同物理输入上的训练归因 | 否 |

LeWM 和各 B 使用各自 encoder、目标编码、原生 history 和动作归一化；交换接口是相同原始图像历史与物理动作。不能交换 latent，不能将 CoWM 归一化动作直接交给 LeWM。对同物理动作的归一化往返、第 25 步对齐、首帧 history padding 做数值验收。

主面板报告池覆盖率、选中动作在 25 步内的成功率、物理 regret、Spearman、有效状态/候选数。regret=c(选中)−min c(池中)；先逐状态算再汇总。物理 oracle 的成功率未必等于成功 oracle；覆盖率在相同完整池上对所有评分器相同。

物理代价冻结为任务可解释量：Cube 为 block 位置 L2（米）；Reacher 为关节差 L2（弧度，保持环境角度约定）；TwoRoom 为环境位置 L2（原环境单位）。PushT 分别报告环境成功判据对应的位置距离和环绕角度距离；综合排序代价建议为二者除以各自成功阈值后的最大值，须在运行前审计字段及阈值并写入配置，不直接把像素与弧度拼接求 L2。成功标签严格取 Table 1 环境原生判据，物理距离不替代成功。

主连续指标使用实际第 25 步后果。成功/终止早于 25 步且不能继续时，不复制终止状态冒充第 25 步：终止时成功及代价另报；第 25 步 regret/Spearman 主统计仅使用完整 64 候选均有合法第 25 步后果的状态，明确覆盖率和选择偏差；另外报告共同有效子池结果及候选数。执行失败与缺失不得当作失败任务标签，也不得悄悄删掉难状态。

在少量固定快照上验证恢复后重复回放一致性；确认噪声/RNG 与候选执行顺序不会影响真值。不支持精确恢复的环境须先修复或明确该任务诊断未完成。

四任务上限 1,200 个状态出现项 ×64=76,800 个分支、1,920,000 primitive steps；重复状态/相同池可去重，但统计保留清单权重。增加评分器只新增离线评分，不重复收集同一池的真实后果。候选不是 76,800 个独立评测状态。

## 3b：同一起点与初始动作的修正

每状态固定池中 index=0 的完整原动作，作为 PO 和随机扰动的共同起点；该候选必须来自冻结噪声，不能挑 best/worst。GF 以同一初始噪声的无引导 S=2 轨迹为参考。

| 方法 | 配置 | 对照 |
|---|---|---|
| 无修正 | 同一原动作 | 复用 3a 分支 |
| CoWM-B PO | K=2，步长 0.01，RMS 上限 0.2 | 原动作 |
| LeWM PO | 同 K/步长/共同动作坐标约束 | 原动作 |
| CoWM-B GF | S=2，每步 1 次，总 K=2 | 同噪声无引导动作 |
| 等 RMS 随机扰动 | 每种修正对应 5 个预设随机方向 | 匹配该状态的实际动作改变量 |

PO/GF 的优化坐标统一为 actor normalizer 坐标，scorer 内部可微转换到其自身坐标；保存完整原/修正动作与真实物理动作。主配置不额外裁剪；同时报告环境动作处理之后的实际位移。随机扰动按环境合法化后的 RMS 匹配（固定缩放求解规则/容差），不可匹配项明确标记，不放宽容差挑结果。

每个修正分支从相同快照执行；报告真实代价下降 c(original)−c(updated)、改善状态比例、预测改善但真实变差比例、动作 RMS、有效分母和早停比例。数值“改善”的容差在冻结配置中预设，Spearman 并列用平均秩、常数序列记 NA。不同评分器的 raw latent cost 不直接互比。

三种修正各 1 条、各自随机扰动 5 条，共新增最多 1,200×18=21,600 分支/540,000 环境步。加排序池最多 98,400 分支/2,460,000 步；不含 Table 2 新 B 的修正分支，每新增单任务 scorer checkpoint 上限再加 300×6=1,800 条（PO+5 个匹配随机方向）；Table 2 的四个 B 臂×三个训练 seeds×四任务共 48 个 checkpoint，上限另加 86,400 分支/2,160,000 环境步。

闭环 SR 是独立指标：CoWM P0/PO/GF 可以引用 Table 1；同 A+LeWM PO 若要填闭环列，需另做 24 单元/1,200 episodes，不能把单段修正成功率填入闭环列。该闭环扩展先列可选。

## 3c：物理量探测（最高新增实验优先级，Cube 为主）

### 对齐 LeWM 的目标与表格

已核对本地 `docs/paper/LeWorldModel.pdf` 第 24 页 Table 4，以及 [LeWM 论文附录 F.2](https://arxiv.org/html/2603.19312v2#A6.SS2)。该表包含下列 **八类物理量及 Overall 汇总**，每项均报告 Linear/MLP 的 MSE↓和 Pearson r↑。本轮完整覆盖八类，不按 LeWM 的强弱挑选；原稿只有 Block Position 四行的模板需在结果填表时扩展。

本地 `data/datasets/ogbench/cube_single.h5` 已做只读 schema 检查，以下字段均存在，每字段 2,010,000 行。字段名/维度是本地已核实的映射；原论文没有逐一公布字段选择，尤其 Gripper 的具体定义仍需在实施审计中核对，不把映射推断说成官方实现。

| 原论文属性 | 本地目标字段 | 维度 | 口径 |
|---|---|---:|---|
| Joint Position | `proprio_joint_pos` | 6 | 机器人关节位置 |
| Joint Velocity | `proprio_joint_vel` | 6 | 机器人关节速度 |
| End-Effector Position | `proprio_effector_pos` | 3 | 末端位置 |
| End-Effector Yaw | `proprio_effector_yaw` | 1 | 末端偏航角 |
| Gripper | `proprio_gripper_opening` | 1 | 暂按夹爪开度；另有 contact/vel 字段，不能不加说明拼入此项 |
| Block Position | `privileged_block_0_pos` | 3 | 方块位置 |
| Block Quaternion | `privileged_block_0_quat` | 4 | 方块四元数，环境约定 w/x/y/z |
| Block Yaw | `privileged_block_0_yaw` | 1 | 方块偏航角 |
| Overall | 以上八类汇总 | — | 本轮主报八类等权均值，另外保留各坐标指标 |

物理单位、坐标系、角度范围和 Gripper 定义写入 `target_schema.json`。不能以目标方块位置等 goal 字段替代当前物理状态。原论文的 Overall 聚合权重、MSE 归一化、± 的来源以及 probe 优化超参未从所查正文/附录确认；实施时先核对官方可用实现。无法取得时采用下述公开的本轮协议，两模型统一重测，论文原数值仅作参考，不宣称逐项数值复刻。

### 模型、输入与 probe

主比较使用 Table 1 归档的 LeWM 和 CoWM R4-AB checkpoint，模型保持 eval，encoder/projector/动力学全部冻结。输入为同一帧 RGB 所得到的、模型实际用于动力学的 latent；不把物理标签、动作、目标图像或 history 拼到 probe 输入，不用一个模型的 latent 训练另一个模型的读出器。记录实际 latent 层/维度及图像预处理。主测 **observed latent 的可读出性**，与 LeWM 表格的问题对齐。

- 每模型/物理量各训练独立线性读出和 MLP。建议 Linear=带截距 Ridge，alpha 网格 `{0,0.01,0.1,1,10,100}`，仅 validation 选值；MLP=`latent→256→256→target_dim`，ReLU，Adam lr=1e-3、batch=256、上限100 epochs、验证集 patience=10，读出 seeds=0/1/2。两模型使用相同搜索预算。这些是本轮默认超参，不冒称原论文给定。
- 特征和目标的均值/方差只由 probe-train 拟合；常数目标维度保留并标注，Pearson 记 NA，不用零方差除法制造数值。主报标准化坐标 MSE 与逐坐标 Pearson 后的属性均值，同时保存物理单位 MSE/MAE。
- Yaw/Quaternion 主列保留原始坐标定义以对应论文属性；周期角的 wrapped error、四元数 q/−q 不变的旋转误差作为另列补充，不暗中将主目标替换为 sin/cos 或改变 quaternion 符号约定。额外变换均写明，不与原论文坐标 MSE 混用。
- 同时报训练均值预测器；标签打乱控制用于发现泄漏。可以复用历史代码的特征缓存、Ridge/MLP 组件，但旧代码只对 Cube block position 输出 MAE/RMSE，不能直接填满本表。

### 数据划分与统计

从原始离线数据抽取，按来源 episode 分割为 probe-train/validation/test，同一 episode 的帧不跨集合。先排除已用于 Table 1 及已冻结 3a/3b 的评测 episode，再固定 RNG=20261005 选择最多1,000条可用轨迹，按80/10/10划分，每轨迹均匀取最多100个不同帧（最多100,000帧）。数量不足使用全部可用轨迹并报告实际分母，不重复补足；两模型、八类目标使用同一有效帧集合及索引哈希。

probe-test 相对读出器训练独立；对于已经训练好的世界模型，不能保证这些数据未被其预训练见过，需分别披露，不将 probe-test 称为世界模型训练未见数据。不会用闭环的六个评测 seeds 代替 probe 数据划分。

每属性报告 Linear/MLP 的标准化 MSE 和 r、样本数及坐标维度；MLP 报三个读出 seeds 的 mean±std，Linear 确定性拟合不伪造三个训练重复。配对区间以 test episode 为簇进行10,000次 bootstrap，保留同 episode 的帧及两模型预测配对。读出 seed 不代表三个世界模型 seed。若报告显著性，八属性×两读出器的 MSE 差构成16项 Holm族，r 和 Overall 为补充；完整逐属性结果保留，不仅呈现 Overall。

### 辅助任务与后续扩展

- TwoRoom：Agent Position（2D），对应原论文附录 Table 3。
- PushT：Agent Location、Block Location、Block Angle，对应原论文 Table 1；按本地 state schema 分离字段，不合并为一个误差。
- Reacher：先测关节位置；关节速度在字段/时点一致性审计后加入。LeWM F.2 没有 Reacher probe 表，明确这是我们的辅助扩展，不能称为复现原表。
- 以上三任务共用 Cube 的数据划分原则与读出评价规则，分别汇总，不与 Cube 八属性混成一个总体得分；先完成 Cube 的全部目标再扩任务。
- predicted latent 为后续机制扩展：将只在真实图像 latent 上训练好的 probe 固定，用相同记录动作预测第5/25 primitive step，与真实未来图像 latent 的读出误差成对比较；不重新训练 probe 去适应预测 latent。该扩展不阻塞首批 observed probe，也不把其误差单独归因于动力学。

### 实施与交付

先审计字段、标签时点、模型层和 split，再缓存两模型同帧 features，完成 Cube Linear 和 MLP 全量指标，之后扩其它任务。Cube 主面板为2模型×8目标×（1个最终Linear+3个MLP seeds）=64个最终 probe fits，网格验证另计；无需重训世界模型，也无需新增闭环或反事实候选回放。probe 的梯度训练在 GPU0–3 或 CPU 运行，不将其视为 GPU4–7 的评测授权。

输出 `outputs/cvpr/table3/v1/probe/{task}`，包含论文/代码来源、target schema、split/row manifest、normalizer、特征与模型哈希、probe config/checkpoint、逐样本真值/预测、各坐标/属性/Overall 指标和重建脚本。验收必须覆盖八类目标齐全、标签没有输入 encoder、episode 无泄漏、归一化只拟合训练集、两模型相同测试行以及 MSE/r 可从原始预测重算。已有旧 block-position 结果只作历史参考。

## 同 A 闭环桥接、统计与交付

新增两组闭环：Random-64、同 A+LeWM rerank，各 24 单元，共 48 单元/2,400 episodes；同时供 Table 2 的评分器说明及 Table 4/5 使用。闭环轨迹分叉后输入状态不同，只能共享状态索引化的随机流/噪声调度，不能宣称每次实际候选池相同。对相同观测的候选完全匹配由 3a 证明。

主要诊断：CoWM-B 对 LeWM 的选中成功率差和 regret 差、CoWM PO 对等 RMS 随机扰动的真实改善差，四任务共 12 项为一个 Holm 族；GF、连续分项和与随机选择的对比另列探索结果。配对 bootstrap 10,000 次、固定 RNG=20261005，以来源 episode 为簇；跨训练 seed 扩展时增加 seed 层。单 checkpoint 结论不外推训练稳定性。

从 `scripts/round5_phase1_7_fixed_pool.py`、`scripts/round5_phase1_6_mechanism.py` 等复用恢复/分支/指标工具；旧脚本包含 clip 和旧成功判据，必须接入本表冻结协议后才可用。保存 `pools`、`replays`、`scores`、`refinement`、`closed_loop` 和数据身份索引；做恢复一致性、动作接口和有效分母验收。

上述桥接与3a/3b在首批 Cube 3c 后推进；ABDE、在线微调和真机不在此范围。

## 公共协议与资产规则

- 以下闭环/候选协议适用于3a/3b及桥接；3c的数据、输入与统计遵循其专节，不套用300个闭环样本。四任务固定为 TwoRoom、PushT、Reacher、OGBench-Cube；闭环沿用 Table 1 的 24 份样本清单，评测 seeds 为 `42,100,2026,3407,1234,4444`，每 seed 50 episodes。起点、目标、环境成功判据和环境 RNG 均沿用冻结配置，不改用 Round5 的过滤协议。
- 默认目标 offset=25、总执行预算=50、生成/评分/执行=25/25/25，5 个 block，每 block 5 步。Euler、S=2；P3 默认 N=64；FP32，关闭 autocast/TF32/compile；无额外候选裁剪，保留环境原生动作处理。例外逐条件登记。
- Reacher 10/10 的 Selection† 仅为单列部署参考，不混入同跨度的机制比较。裁剪条件单列；不按结果为每个方法挑选裁剪或时域。
- 单 checkpoint 结果报告六个评测 seed 的 mean ± sample std（ddof=1）；宏平均先在同 seed 内平均四任务，再跨 seed 汇总。训练重复另外统计，不能把评测 seed 当训练 seed。
- Table 1 样本已用于开发观察，后续复用只能称同 benchmark 扩展，不能称未见确认集。配对差保留逐任务、逐 seed 结果；存在重复起点/同来源轨迹时，区间按来源 episode 聚类，重复出现项共同重采样，不把候选或重复测时当独立样本。
- 只有 checkpoint、样本清单、动作语义、算法参数、代码兼容性及原始结果哈希全部通过审计才复用；建立 `source_refs.json` 指向已有单元，明确 alias，不伪造新运行。旧 Round5 clip、过滤 cohort、短训结果只能作历史参考。
- 新资产放 `outputs/cvpr/tableN/v1`（N 为本表编号），报告放 `docs/report/cvpr/cvpr_tableN_report.md`。保存 frozen config、代码及差异、数据/权重/样本 SHA256、逐 episode 或逐候选记录、失败记录、统计脚本和报告来源；不覆盖已有结果。
- 此次交付为计划，不启动训练或评测。实施时训练仅用 GPU0–3，每条 GPU 命令显式设置 `CUDA_VISIBLE_DEVICES`，先检查空闲显存并保留运行余量。GPU4–7 须有适用的用户明确授权且仅用于低显存评测，10:00–23:00 北京时间尽量避开；Table 1 的 GPU6 使用记录不自动扩展为本轮授权。
