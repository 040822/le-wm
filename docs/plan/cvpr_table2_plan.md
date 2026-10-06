# CVPR Table 2：训练耦合消融实验方案

## 依据与编号

编写日期：2026-10-05。按当前论文的 [实验设计](../../paper/docs/Couple_lewm.md)、[填写指南](../../paper/docs/RESULTS_FILL_GUIDE.md) 和 `paper/sec/4_experiments.tex` 的编号组织：Table 2 耦合、Table 3 排序/修正/物理量探测、Table 4 推理、Table 5 预算/效率。Table 3c 优先完成 Cube 全物理量探测；9 月 19 日旧版的 flow-step 消融列为 Table 5 低优先级可选附录。用户已确认采用当前论文版编号。用户已确认 Table 2 采用五个核心训练臂、三个训练 seeds；下述细化参数在实施前写入冻结配置。

结果依据：[Table 1 完整报告](../report/cvpr/cvpr_table1_report.md)、[审计报告](../report/cvpr/cvpr_table1_paper_audit_20261005.md)、[Table 1 计划](cvpr_table1_plan.md)。实际运行协议以 `outputs/cvpr/table1/v1/frozen_config.json` 为准，已完成 948 个成功率单元和 158 个计时条件。

## 目标与 Table 1 的启示

检验共享框架中加入动作生成任务、引入预测动作、保留 B→动作→A 梯度是否改善生成、评分和修正能力，不预设 Full 最优。Table 1 的 P3 为 96.25%，P0 为 94.17%，只能说明固定模型下使用 B 的效果，不能归因为联合训练。TwoRoom/Cube 已接近饱和，保留四任务闭环覆盖，并用 Table 3 的连续后果指标补充。

## 训练矩阵与可归因配对

用户确认五个核心臂、三个训练 seeds。Shared-Recorded 在本计划中明确指 Recorded-control：保留 Full 的 timestep/权重，只替换 B 的动作输入。B-only 使用现有 clean 定义；二者不能作为纯“加入 A 任务”对照。

| 臂 | A loss | B 动作输入 | B timestep / 样本权重 | 动作路径梯度 |
|---|---|---|---|---|
| A-only | 有 | 不训练 B | 不适用 | 不适用 |
| B-only-clean | 无 | 记录动作 | clean t=1 / 单位权重 | 无 |
| Shared-Recorded-control | 有 | 记录动作 | 与 Full 相同的 legacy timestep / noise weight | 无 |
| Shared-Pred-Detach | 有 | 与 Full 相同的记录/预测动作课程 | 与 Full 相同 | 对传入 B 的预测动作 detach |
| Shared-Pred-Full | 有 | 记录/预测动作课程 | 原 R4-AB 设置 | 保留 |

所有 Shared 臂均使用同一个共享 DiT；Detach 不切断 B 自身对共享参数/表征的梯度。B-only 不提供自己的 actor。A-only 不得用 Full 的 P0 代替。

预设主要配对：

1. Shared-Recorded-control − A-only：加入记录动作 B 监督对 Own-A P0 的影响，属于该联合训练方案的整体效应。
2. Shared-Pred-Detach − Shared-Recorded-control：匹配 timestep/权重/混合 mask 后的预测动作输入效应。
3. Shared-Pred-Full − Shared-Pred-Detach：动作路径梯度效应。

可选桥接臂 Shared-Recorded-clean 使用 clean t=1/单位权重，与 B-only-clean 比较才能隔离干净监督下加入 A 任务的效果；与 Recorded-control 比较则诊断 timestep/权重联合变化。该桥接臂不计入已确认核心预算，不自动启动。Full vs A-only/B-only 为系统比较。以上五臂不能单独证明“参数共享优于两个独立 DiT”；Separate-Recorded 为条件扩展，需固定编码器共享方式，并报告容量差异。ABDE/ABCDE、在线训练、额外任务不纳入核心矩阵；历史 E0–E5、R4-ABDE 不直接重命名成上述匹配臂。

## 训练预算与实现验收

四任务 × 五臂 × 三训练 seeds（3072/4096/5120），共 60 个完整训练 run。先完成 3072 的全部匹配臂以核验流程，再完成另两个 seeds；不能根据成功率决定保留哪些臂。若之后启用 Recorded-clean 桥接，额外增加 4×3=12 个 run，独立登记扩展范围。

训练超参数建议按当前 R4-AB 模板设置，实施前与归档实际配置逐项核对：10 epochs、batch 128、AdamW lr=5e-5、weight decay=1e-3、192 维、6 层 shared DiT、parallel_prefix/strict_causal、legacy token、latent weight=1、SIGReg=0.09、动作混合课程 10 epochs。训练精度沿用实际归档配置（当前模板 bf16），评测统一 FP32。固定最后 epoch=10，不选最佳测试 checkpoint。

同 task/seed 共享初始同名权重、数据 split、训练窗口、normalizer、样本顺序及可匹配的随机流。RNG 分流，避免某臂少一次前向导致数据/噪声错位。保存初始模型哈希及共有参数哈希、每轮样本数、优化步数和实际训练成本。同 epoch/更新数匹配不等于同 FLOPs；A-only/B-only 的表征正则、目标梯度路径也须明确匹配。

已知入口为 `config/train/round4_ab.yaml`、`config/train/policy/round4_ab.yaml` 和 `source/policy/fast_lewam.py`。实施前核验 stage_a/stage_b 的有效损失与目标构建；不能只改配置名就认为获得匹配消融。先用同一 minibatch 验证 recorded_control 只换动作、不换 t/weight；Detach 只断动作路径；A/B-only 无被禁用任务的有效梯度。

已有 R4-AB/Phase1.7 权重逐项审计再决定是否可复用。原 Table 1 R4-AB 作为历史 anchor；若无法重建与其他臂相同的初始化、数据与预算，Full 也需重训。Table 2 新 Full 不自动替换 Table 1/4/5 的固定 checkpoint。

## 评测面板与规模

**2a 训练定义**采用上表；**2b 结果**逐任务呈现以下指标，宏平均只用于有完整四任务的可比列：

- 四个有 A 的臂：Own-A P0；三个 Shared 臂：Own-A P3、Own-A P0-PO2。
- 四个有 B 的臂：同一固定 actor 的候选池 regret/选中成功率，以及同初始动作的 PO2 真实修正收益；固定 actor 采用 Table 1 R4-AB A，保持外部输入一致，沿用 [Table 3](cvpr_table3_plan.md) 的池与分支真值。此结果限于该 actor 分布，交叉 actor 池为补充。
- 四个有 B 的臂：固定 R4-AB actor + 各 B 的闭环 P3，作为训练对评分效用的主要闭环终点。

每训练 seed 有 4+3×2+4=14 个闭环条件，每条件 24 单元/1,200 episodes；三个训练 seeds 上限 1,008 单元/50,400 episodes，单 seed 336 单元/16,800 episodes。相同条件经身份核验可 alias，不能重复计数。固定池评分与局部修正分支另计；不把 B-only 的 Own-A 列填零。

**2c 资源补表**记录总/活跃参数（分别含/不含 encoder）、优化步数、GPU-hours、每更新耗时、峰值显存。没有双 DiT 时不声称共享节省了多少资源。

训练重复汇总先在每训练 seed 内平均六评测 seeds，再报告三个训练 seed 的 mean ± sample std；另附六评测 seeds 的原始结果。主要差值按训练 seed 配对，区间同时考虑训练 seed 与来源 episode，注明三个训练重复的统计精度有限。三个主要配对分别指定终点：Recorded-control−A-only 看 Own-A P0，Detach−Recorded-control 与 Full−Detach 看 fixed-actor P3；四任务共 12 个检验构成 Holm 校正族；其它指标为探索性。负结果保留，不用饱和任务掩盖 Reacher 的方向变化。

## 顺序、交付与待定项

先完成权重/代码/数据身份审计和五臂 CPU 梯度验收，再做完整训练、统一闭环和 Table 3 固定输入诊断，最后统计资源与配对差。Table 3 可先用 Table 1 权重推进，不必等待训练。

新增执行配置/入口应支持 audit、train、eval、diagnose、summarize；当前仅有历史脚本可借用，不宣称已有 `cvpr_table2` 完整入口。核心范围固定为五臂、三 seeds；Recorded-clean 桥接、Separate-Recorded 与 ABDE/ABCDE 均为可选扩展，未纳入 60 run 的执行矩阵。

## 公共协议与资产规则

- 四任务固定为 TwoRoom、PushT、Reacher、OGBench-Cube；闭环沿用 Table 1 的 24 份样本清单，评测 seeds 为 `42,100,2026,3407,1234,4444`，每 seed 50 episodes。起点、目标、环境成功判据和环境 RNG 均沿用冻结配置，不改用 Round5 的过滤协议。
- 默认目标 offset=25、总执行预算=50、生成/评分/执行=25/25/25，5 个 block，每 block 5 步。Euler、S=2；P3 默认 N=64；FP32，关闭 autocast/TF32/compile；无额外候选裁剪，保留环境原生动作处理。例外逐条件登记。
- Reacher 10/10 的 Selection† 仅为单列部署参考，不混入同跨度的机制比较。裁剪条件单列；不按结果为每个方法挑选裁剪或时域。
- 单 checkpoint 结果报告六个评测 seed 的 mean ± sample std（ddof=1）；宏平均先在同 seed 内平均四任务，再跨 seed 汇总。训练重复另外统计，不能把评测 seed 当训练 seed。
- Table 1 样本已用于开发观察，后续复用只能称同 benchmark 扩展，不能称未见确认集。配对差保留逐任务、逐 seed 结果；存在重复起点/同来源轨迹时，区间按来源 episode 聚类，重复出现项共同重采样，不把候选或重复测时当独立样本。
- 只有 checkpoint、样本清单、动作语义、算法参数、代码兼容性及原始结果哈希全部通过审计才复用；建立 `source_refs.json` 指向已有单元，明确 alias，不伪造新运行。旧 Round5 clip、过滤 cohort、短训结果只能作历史参考。
- 新资产放 `outputs/cvpr/tableN/v1`（N 为本表编号），报告放 `docs/report/cvpr/cvpr_tableN_report.md`。保存 frozen config、代码及差异、数据/权重/样本 SHA256、逐 episode 或逐候选记录、失败记录、统计脚本和报告来源；不覆盖已有结果。
- 此次交付为计划，不启动训练或评测。实施时训练仅用 GPU0–3，每条 GPU 命令显式设置 `CUDA_VISIBLE_DEVICES`，先检查空闲显存并保留运行余量。GPU4–7 须有适用的用户明确授权且仅用于低显存评测，10:00–23:00 北京时间尽量避开；Table 1 的 GPU6 使用记录不自动扩展为本轮授权。
