# Round5 Phase1.7：独立 LeWM 对照、B 排序瓶颈与改进验证

## 目标与判定

首要问题：固定动作生成器 A 时，联合训练的 B 是否比独立 LeWM 更会选择动作。其次判断联合训练是否优于同架构独立训练的 B，以及排序改进是否传递到闭环成功率。

Phase1.6 的 CEM 使用同一个 B，不能作为独立 LeWM 对照。历史 LeWM 结果存在架构、历史输入和训练设置差异，不能单独证明 B 优劣。Phase1.7 将区分排序、训练归因和推理效率三项贡献；允许实验得出 B 无优势或落后的结论。

## 实验矩阵

### 现有权重对照

六个任务 Cube、PushT、Reacher、TwoRoom、Scene、Humanoid 均使用相同的 A，Euler、S=2、N=64、无 guidance，评估随机选取、原 B、独立 LeWM；补充原 B 与 LeWM 各自的 300×30 CEM。固定池诊断复用逐候选相同的物理动作；闭环只在相同观测处匹配候选随机流。

物理动作是评分器交换接口。A 候选先反归一化并按物理范围裁剪，再各自转换到 B 和 LeWM 的输入规范。两模型分别编码相同原始观测和目标，不比较跨模型 latent 数值。LeWM 保留标准历史输入和自回归 rollout，历史只从真实轨迹/在线缓冲构建；显式审计首帧填充、动作拼接及第 25 primitive step 对齐。

### 配对训练

PushT/Reacher 各两个 seed（3072、4096），预算允许时各补 seed 5120。完整主矩阵每 seed 含四臂：

| 臂 | 训练定义 |
|---|---|
| Joint | 当前共享 A/B 联合训练和预测动作混合课程 |
| Recorded-control | 共享架构并保留 A loss/原 mask 与权重，只把 B 输入替换为记录动作 |
| B-only | 同架构 B 动力学训练，无 A loss，只用记录动作 |
| LeWM | 独立标准 LeWM，匹配数据和训练样本访问量 |

Joint、Recorded-control、B-only、Recorded-clean 在同一训练 seed 下使用同架构、同初始 A/B 权重、同 episode split、同窗口采样顺序及可配对噪声。各模型均允许表征训练。Recorded-control 使用原训练的 timestep/权重；B-only 使用干净动作条件 t=1。实现门槛：该消融组的 `phase1_7_metadata.json` 必须记录一致的 `initial_model_state_sha256`、episode split SHA、action normalizer 和 train window count；若任一不一致，则停止训练臂因果解释，并用同一冻结初始权重和数据划分重跑受影响的臂。独立 LeWM 是不同架构且单独随机初始化；它与 Joint 使用相同 episode split、normalizer、batch size、优化更新数和每轮样本访问数量，但因 `history_size` 和预测窗口不同，原始 train window count 及窗口 ID/顺序不完全相同。Joint-B 对 LeWM 因此是预算匹配的系统级排序比较，不用于单独归因联合训练机制。

Scene/Humanoid 各训练 Joint/LeWM 一个配对 seed，预算允许时各补第二 seed。Cube/TwoRoom 复用现有权重作覆盖和退化检查。默认 10 epochs、batch 128、AdamW、lr 5e-5、weight decay 1e-3、现有表征正则。固定最后 epoch 为终点，逐模型记录样本访问、优化步数、参数量和 GPU 小时；不把同 epoch 当作同计算量。

训练轨迹诊断：在资源允许且不阻塞训练时，可用已保存的 epoch 2 checkpoint 对 PushT/Reacher dev 100 起点提前运行 P3 闭环比较。Joint-B 与独立 LeWM 使用同一 Joint actor、同 seed cohort 和匹配更新数；输出只作中期开发信号，须记录真实 checkpoint epoch，不并入 epoch 10 主终点或 confirmation 结论。若同一 task/seed 的 Recorded-control 或 B-only 早期训练目录保留了完整 epoch 2 权重，且 `initial_model_state_sha256`、split SHA、normalizer、训练窗口数均与 Joint 一致，可追加相同 actor、相同 dev cohort 的 P3 与固定池 100 状态诊断。此追加诊断须标记为中断初试 checkpoint，只用于解释训练轨迹，不进入终点比较、confirmation 或主要显著性检验族；缺少任一匹配字段时不运行。

### 改进探索

离线主线在 PushT/Reacher 各两个 seed 添加 Recorded-clean：完整联合训练，B 始终使用记录动作、t=1、单位样本权重。与 Recorded-control 比较 timestep/权重效应，与 Joint 比较可部署方案是否改善排序。

在线分支仅在 PushT/Reacher 探索。每任务从训练 episode 抽 256 个状态，每状态 32 个候选分支（16 条 A 候选、16 条裁剪局部扰动），每分支最多执行 25 步，上限 16,384 分支和 409,600 环境步。逐条存储真实动作、时点观测、成功/终止和有效长度；恢复后审计重放一致性；不可继续的终止不伪造未来目标。B 与 LeWM 获得相同原始分支数据及划分。

在线微调臂为：等更新数离线续训、加入真实分支未来状态的预测训练、再加同状态动作对排序损失。固定 A 与图像表征，只微调评分器。排序目标用真实未来图像在各评分器冻结表征下的目标距离产生；不将物理距离或成功标签用于训练。损失为成对 logistic，λ∈{0.1,1.0}，2,000 更新、lr 1e-5、batch 128。开发集按闭环成功率、物理 regret、延迟依次选配置，确认集只用于一次锁定验证。新数据设定单独报告，不混入纯离线主结论。

在线训练入口为 `scripts/round5_phase1_7_online_train.py`。缓存阶段按 scorer 分别编码起始/目标图像和有效分支里程碑；5 个模型步各对应 5 个已执行 primitive action，早停分支只监督完整有效 block。训练阶段锁定图像 encoder/projector，Joint 使用 Stage-B 预测，LeWM 使用真实当前帧复制填充的标准历史与自回归预测。离线、分支预测、λ=0.1、λ=1.0 四臂复用各自已冻结 checkpoint；在线三臂共用预生成 batch/pair schedule，训练更新只读预测图像损失和真实图像 latent 距离，不读取成功标签或物理距离。GPU 特征缓存属于低显存评测/数据处理；梯度训练必须限于 GPU0–3，并在预计峰值后保留至少 6 GiB。

首次大规模分支采集作为探索性中期数据集单独锁定：PushT/Reacher 各从 train split 的 256 个互斥 episode 中各选一个有完整 25 步未来的状态；按固定随机种子先抽 episode，再在 episode 内均匀抽起点，不按模型分数或分支结果筛选。为尽早验证在线改进路线，首批固定使用 seed 3072 的 Joint actor epoch 2 snapshot 作为 A；其结果只对应该 actor 分布，后续若切换 epoch 10 actor，必须另存新 cohort 与 artifact，不覆盖首批数据。每状态用同一 A 生成 16 个 P3/S=2/Euler 候选，再以候选池均值为中心采 16 个局部高斯扰动；每维标准差为该状态 16 个 A 候选标准差的 0.25 倍，最小 0.05 个训练 normalizer 标准差，并裁剪到物理动作边界经 train normalizer 映射后的合法区间。动作池、初始噪声、候选来源和 actor/checkpoint hash 全部固定记录。每候选从相同环境快照独立回放，最多 25 个 primitive step；存储真实动作、状态/目标图像索引、5/10/15/20/25 步图像及终止图像、终止标志和有效长度。终止后的状态不补造。查看分支结果前，按状态 ID 固定 80/20 的训练/开发划分，Joint-B 与 LeWM 共用完全相同的划分。训练目标只从各自冻结表征下的真实未来图像距离生成；成功标签与物理距离只作审计/报告，不进入损失。

## Cohort、终点与统计

新训练采用 episode 级互斥 train/dev/confirmation 清单，归一化统计只从训练集拟合。训练 episode 供在线分支采集；评测 episode 与训练 episode 互斥。PushT/Reacher dev 100、确认 200 起点；Scene/Humanoid dev 50、确认 100；Cube/TwoRoom 覆盖评测 100。每 episode 最多一个起点，推理 seed 42/43。旧权重只称新评测起点，不宣称训练未见。若合格 episode 不足，按固定随机顺序使用可用集合、报告实际分母，不重复起点凑样本。

主要闭环比较：同 A 下 Joint-B 对 LeWM、Joint-B 对 B-only；两个比较组成 Holm 检验族。固定池排序报告 top-1 真实成功率、物理距离 regret、相对随机改善、候选池 oracle 与命中率、成对符号准确率、Spearman、oracle-best 的 top-5 覆盖。物理后果指标评估用，不跨模型比较 raw latent MSE。固定报告第 5/25 步、25 步成功事件、有效覆盖率。100 状态（扩展任务 50）固定池独立确认。

对起点和训练 seed 做配对/分层重采样；不把推理 seed 或候选当独立样本。每个训练 seed 单独呈现，避免两 seed 的汇总掩盖反向结果。任务间优劣差异须有直接交互检验。

- 排序优势：同池真实结果与闭环同方向，主闭环配对差经过校正支持正向效果。
- CEM 效率主张：对 A+LeWM 的宏平均单侧 95% 下界 >−2 pp、声明任务下界 >−4 pp，且完整推理 p50 至少降低 20%。
- 改进成立：锁定确认集优于等更新离线续训；使用新增分支数据的结论同时报告 LeWM 对照。
- 未显著优于不等于等价；未通过非劣性不表示已证明劣于。

## 实施、资源与交付

新增配置 `config/round5/phase1_7.json`、执行入口 `scripts/round5_phase1_7.py`、报告 `docs/report/round5/round5_phase1_7_report.md`。入口覆盖审计、数据冻结、训练、分支采集、固定池评分、闭环、计时和导出。评分器统一接收原始观测历史、目标和物理单位候选动作，返回候选代价及有效标记。

验收须覆盖动作归一化往返、clip/评分/执行一致、历史与第 25 步对齐、批处理和逐候选排序一致、候选 RNG、episode 互斥、状态恢复重放、参数冻结及 checkpoint 续训。先运行小规模端到端校准，再扩展。计时覆盖 A、全部图像编码、动作转换和评分；batch-1/50 报 p50/p95、吞吐、峰值显存和调用数。

72 小时目标：0–6 小时审计、清单、吞吐校准及现有权重对照；6–48 小时配对训练和分支采集；48–60 小时开发选择与锁定确认；60–72 小时补齐、隔离计时、统计与报告。训练矩阵不能在校准前承诺全部按时完成。按完成时间预估预留 25% 余量，并在查看性能结果前冻结队列。缩减时先删第三核心 seed、扩展任务第二 seed和 detach 扩展，再缩排序网格，再延后 Scene/Humanoid 新训练、Recorded-clean；始终优先保留 PushT/Reacher 两个 seed 的 Joint/LeWM 对照、同 A 排序对照和一个改进确认。不得用部分训练臂拼成完整比较。

训练每条命令显式设置 `CUDA_VISIBLE_DEVICES`，只在 GPU0–3 上运行，并按实际可用显存和峰值波动预留至少 6 GiB；评测预留至少 2 GiB；隔离计时与同卡训练错开。GPU4–7 仅用于低显存评测，即使显存充足或用户授权，也不用于训练；北京时间 10:00–23:00 如 GPU0–3 有可行空档，优先避开 GPU4–7。

最终报告回答：原 B 是否优于独立 LeWM、联合训练是否有额外排序收益、哪种改进复现、以及论文应如何界定贡献。若 A+LeWM 持平或更优，论文相应调整结论；只有公平比较支持时才主张联合 B 的排序优势。


## 在线epoch-2确认选择锁定（2026-09-30）

在查看confirmation结果前，PushT和Reacher均锁定Joint-B `rank_1.0`作为在线开发候选。PushT该臂在Joint-B臂中闭环成功率最高（22/100）；Reacher该臂在Joint-B臂中闭环成功率最高（12/100），并在共同动作池中取得最低regret。独立LeWM、原Joint-B、等更新offline续训及LeWM在线控制作为预先指定比较项。一次确认的cohort、checkpoint SHA、比较项和口径记录在 `outputs/round5/phase1_7/locked_confirmation_selection.json`；确认数据不得用于重新选臂。
