# CVPR Table 1 CoWM 提升实验方案

## 1. 目标与冻结协议

本计划优先提高 Table 1 中 CoWM 的 Reacher 成功率，并让正式结果可以替换 Table 1 的 CoWM 行。主结果保持 Table 1 的目标偏移、总预算、反馈频率、episode 样本和推理精度；允许仅 Reacher 使用不同评分、refinement 参数和续训 checkpoint。成功率没有硬阈值，按下述预先规定的规则选择结果，并完整报告负结果。

主表新增或替换为以下两种 CoWM 方法：

| 主表行 | 定义 |
|---|---|
| CoWM-Selection | P3：产生64个25步候选，按 CoWM-B 代价评分并选择，不作梯度优化。 |
| CoWM-Selection + Refinement | 在同一 Selection checkpoint 和代价函数上，Reacher 先选候选再作 PO/GF；另外三任务复用 Selection。表注明确 refinement 只用于 Reacher。 |

Table 1 原 P3、P0 refinement 与 Selection† 结果继续保留为历史和补充对照。此次结果写入独立版本，不覆盖 `outputs/cvpr/table1/v1`。

| 项目 | 设置 |
|---|---|
| 任务 | TwoRoom、PushT、Reacher、OGBench-Cube |
| 正式 cohorts | 复用 Table 1 冻结的任务×seed 样本清单和起点，不重新抽样 |
| 评测 seeds | `42, 100, 2026, 3407, 1234, 4444`；每 seed 每任务50 episodes |
| seed映射 | cohort seed=s；环境seed=s+10000；策略seed=s+20000 |
| 目标偏移 / 总执行预算 | 25 / 50 个环境步 |
| 生成跨度 / 执行跨度 | 25 / 25 步；5个action block，每个5步 |
| 候选数 / 动作生成器 | 64 / 两步Euler积分 |
| 动作边界 | `action_bound_mode=none`，沿用Table 1主版本和环境原生处理 |
| 评测批量 / 精度 | 50个环境；FP32；关闭autocast、TF32和compile |
| 复用权重 | 首选Table 1归档的R4-AB、seed3072、epoch10四任务权重 |

原六个评测seed直接参与本轮调参选择，这是按用户指定的公开benchmark优化。报告应说明六seed统计描述的是benchmark评测波动，不把选中结果表述为独立、未查看的泛化确认。

## 2. 复现与轨迹评分

### P3复现

先用当前正式入口在四任务×六seed上重跑原始P3，共1,200 episodes。逐单元核对checkpoint哈希、cohort哈希、三个seed、50条成功标记、初始候选和选中索引。原endpoint路径应与冻结Table 1结果匹配；任何差异先定位代码或随机数原因，再开展新评分实验。

### Reacher评分扫描

CoWM-B当前代价只使用第25步latent与目标latent的均方距离。新增不依赖真实环境状态的整轨迹聚合。令预测距离为：

\[
d_h=\operatorname{mean}\left[(\hat z_h-z_g)^2\right],\quad h\in\{1,2,3,4,5\}
\]

比较以下固定配置：

| 名称 | 终点权重α | 代价 |
|---|---:|---|
| endpoint | 1.00 | \(d_5\)，原Table 1评分 |
| minimum | 0.00 | \(\min_h d_h\) |
| mixed-025 | 0.25 | \(0.25d_5+0.75\min_h d_h\) |
| mixed-050 | 0.50 | \(0.50d_5+0.50\min_h d_h\) |

始终生成完整25步候选，读取第5、10、15、20、25步的预测，不把初始状态放入最小值；目标仍为25步后的目标。四个评分配置均使用相同64候选数、环境反馈频率和50步总预算。无引导endpoint复用P3复现单元，其他三种新增评分各跑六seed，共900个Reacher episodes。

endpoint是无改动回归路径。轨迹评分仍只使用模型预测，不把环境真实未来状态用于打分。min/mixed是否提高闭环成功率是待实验验证的假设。

## 3. Reacher小规模续训

从同一个Table 1归档Reacher checkpoint分别初始化两个训练范围，各运行两个续训随机seed（3072、4096），总计最多四个训练run。训练seed与Table 1评测seed独立记录。

| 训练臂 | 更新范围 | 约束 |
|---|---|---|
| B-specific | `latent_head`、`query_tokens`、`joint_positions`及mode embedding的B行 | encoder、projector、共享predictor和Stage-A使用的参数冻结，A输出保持原样 |
| Shared-preserve | Stage-A/Stage-B共享主干与两阶段专属参数 | 冻结视觉表示和BN统计；用冻结原模型的Stage-A动作预测作保持约束 |

两臂都冻结encoder、projector及其所有BN统计，冻结Round4 D/E专有参数，不新增部署参数。代码中A/B共用predictor、latent/action输入层和condition层；B-specific必须冻结这些共享参数。Stage A使用`action_positions`而Stage B使用`joint_positions`，因此B-specific只训练后者。mode embedding只训练B行，其余行须逐步保持原值；正确处理AdamW weight decay及动量状态，避免未训练行被修改。

**离线数据与训练设置：**

- 复用Round4原训练配置、图像处理及训练动作归一化。以原训练配置seed3072的90/10窗口划分为数据来源；在正式训练前，排除Table 1六个Reacher cohorts所含的episode。
- 固定抽取512个合格训练episode，每个episode无放回取16个窗口，共8,192个窗口。两臂与两个训练seed使用配对的缓存、窗口顺序、噪声、时间步和目标采样表。
- 每条样本只使用原始离线图像编码latent与记录动作。训练不采集新环境分支，不读取物理状态或成功标签。
- 训练batch=128，AdamW，lr=`1e-5`，weight decay=`0.001`，梯度裁剪1.0，FP32。每run固定2,000次更新，保存0/500/1000/2000更新checkpoint；只有2,000更新终点进入主结果选择，中间checkpoint用于诊断和故障恢复。

**训练损失：**

每个窗口均匀选择一个真实未来block latent作为目标，保持完整五步latent预测监督。两个训练臂使用：

\[
L_B=L_{latent}+0.1L_{cost}+0.1L_{temporal}
\]

- `L_latent`：完整五个预测latent对离线真实latent的均方误差。
- `L_cost`：预测与真实目标距离的SmoothL1误差。距离除以训练缓存中所有真实正距离的中位数，分母下限为`1e-6`。
- `L_temporal`：在同窗口、同目标下，对未来block距离做成对排序。仅保留真实归一化距离相差至少0.05的pair；每对损失为`softplus(-sign(真实距离差) × 预测距离差)`。无有效pair时该项为零。
- Shared-preserve额外加入系数1.0的A保持损失：在配对的起点、目标、噪声动作和时间步上，当前A与冻结原A的动作速度预测MSE。

B使用记录动作预测其真实未来latent；不把不同起点的离线动作误当成同一起点的反事实成功/失败排序样本。所有损失和数据窗口哈希都写入训练receipt。

四个2,000更新checkpoint各评估endpoint、minimum、mixed-025、mixed-050四种无引导P3，Reacher六seed。共16个checkpoint×评分单元、4,800 episodes。与原权重阶段A候选合并选择Selection模型和评分规则。

## 4. Refinement扫描与预先选择规则

Selection选择完成后锁定一个Reacher checkpoint与一个评分规则。PO、GF和候选选择全部使用同一评分代价。Refinement只在Reacher上运行，仍执行完整25步动作。最多评估以下12种配置，每种六seed、50 episodes：

| 方法 | 更新预算 | step size |
|---|---|---|
| P3-PO-refine：选中候选后优化 | K=2或5 | 0.001、0.003、0.01 |
| P3-PO：优化全部64候选后选择 | K=2或5 | 0.003、0.01 |
| P3-GF：两个Euler步均做引导 | 每个Euler步1次更新，共2次 | 0.003、0.01 |

所有PO/GF使用`max_rms_offset=0.2`，沿用当前归一化梯度和reference更新规则；不增加规划裁剪。

结果按以下顺序确定，不按未预先列出的指标追加搜索：

1. 只考虑配置完整、数值有效且通过延迟门槛的单元。
2. Selection取Reacher六seed累计成功episode数最高者；候选包括原checkpoint的四种评分和四个训练终点的四种评分。平分时先选六seed成功率标准差较小者，再选测得延迟较低者，再选原checkpoint，最后按固定配置ID排序。
3. Refinement在锁定的Selection checkpoint和代价函数上，取Reacher成功episode数最高且通过延迟门槛者。平分时按更新次数少、六seed标准差小、延迟低、配置ID排序。
4. 如没有PO/GF配置同时提高Selection成功数并通过延迟门槛，则不宣称获得Refinement提升；保留Selection主结果和完整refinement扫描结果。

选择是对既定六seed benchmark的优化结果。六seed mean±sample std照Table 1报告，不把它解释为训练seed方差或事后选择后的独立显著性检验。

## 5. 延迟、公平性、产物与验收

### 延迟门槛

Selection和Selection+Refinement两行的四任务等权平均初始重规划延迟都必须低于同设备、同口径重新测量的LeFlow。refinement行另外三任务复用原P3，所以只需在Reacher达到动态上限：

\[
T_{Reacher}<4T_{LeFlow,mean}-\sum_{task\ne Reacher}T_{P3,task}
\]

旧报告数据给出的Reacher估算上限约130ms，正式筛选必须使用本轮重测值，不使用旧估算替代验收。

沿用Table 1独立初始重规划计时：batch=1、FP32、无预填goal cache、原生history/padding、10次warmup、50个独特状态各重复5次。计时GPU空闲，记录均值、P50/P95、原始250次调用、A/B前向和反向数、峰值显存及模型加载后基线显存。计时和成功率使用同一算法设置。三项复用任务的P3时间以及四任务LeFlow时间同批重测；四任务mean等权平均比较。

正式选中的Selection与Refinement配置另记录episode累计规划调用和规划时间，说明初始决策延迟不代表整集总时长。

### 公平性与统计

- 各CoWM候选和原baseline使用相同Table 1 cohorts、目标偏移、环境seed、policy seed、精度和50步预算。
- 所有候选及训练模型都保持25步生成、25步执行；不采用10/10、5/5或更大的总预算。
- baseline继续使用Table 1冻结权重与官方配置，不新增baseline训练或轨迹评分扫描。论文和报告中披露CoWM的Reacher离线续训步数与方案搜索规模。
- 任务成功率报告六seed mean±sample std (`ddof=1`)；宏平均先对每seed的四任务成功率求平均，再跨seed汇总。保留逐seed和逐episode配对结果。
- 不重抽样处理失败，不删除超时或低分episodes；记录全部attempt和选择依据。

### 实现与交付

- 新配置与输出版本独立命名，例如`config/cvpr/table1_refine.json`及`outputs/cvpr/table1/refine_v1`。新报告为`docs/report/cvpr/table1/cvpr_table1_refine_report.md`。
- 新增的评分reduction默认为`endpoint`，支持`minimum`和带`terminal_weight`的`mixed`；旧配置未提供新字段时必须得到原终点代价。
- 统一评分函数供选择、PO与GF使用。记录五个前缀代价、聚合代价、被选候选、refinement位移、模型哈希、cohort哈希、配置哈希、seed、训练receipt和计时原始样本。
- Table 1主表提供Selection与Selection+Refinement两行，三任务复用标记为alias且不重复计数；同时生成原P3、原Refinement及配对差值对照。`docs/report/cvpr`报告与`outputs/cvpr/table1/refine_v1`原始产物足以重建汇总表。
- 所有checkpoint独立复制并校验SHA256；原`table1/v1`产物不覆盖。输出清晰标记缺失、失败、完成与重试。

### 验收检查

- 原endpoint无引导路径在冻结输入上复现原P3候选、选择与成功向量；默认设置不改变旧checkpoint推理行为。
- 用构造latent检查minimum/mixed的第1至5 block索引、系数、相等距离和有限梯度；确认初始latent不参与最小值。
- 同输入、seed及候选分片下，正常模式和计时模式选择一致，数值容差在配置中预先固定。
- 所有结果均为50 episodes/单元，seed/cohort/checkpoint身份与Table 1一致；缺失episode或哈希不匹配时不得进入汇总。
- 训练数据episode与正式评测episode无交集；B-specific的A输出不变；Shared-preserve记录A动作输出漂移；BN统计及视觉表示保持不变。
- 通过Table 1同口径延迟门槛；最终report能够从逐episode成功向量和原始计时样本独立重建。

GPU任务优先GPU0–3，GPU4–7亦可；每次实验启动前检查显存，留出至少6GiB余量，所有GPU命令显式设置`CUDA_VISIBLE_DEVICES`。计时只在空闲GPU运行。若进入goal模式监控训练，启动后确认一次，日常检查间隔遵守仓库AGENTS.md规定。
