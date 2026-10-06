# Round 5 Phase 6.1：B 的 latent flow matching 预实验

编写日期：2026-10-05。状态：已完成（2026-10-06；FM-B 实现、训练、评测矩阵、诊断与报告均已完成）。编号按最初提出的两个想法排列；Phase 6.2 为复用权重的 Reacher 执行预算实验，可先执行。

## 目标与范围

在 PushT、Reacher 两个任务上，将 B 的未来 latent 点预测损失替换为 conditional flow matching，检验当前 A/B 联合训练体系下的候选评分、动作修正和闭环控制是否改善。每任务一个训练 seed=3072，固定 epoch 10；其余训练方式沿用当前 R4-AB。新增两个 FM 训练 run，不训练额外 seed，不加入 D/E，不改成冻结 encoder/A 的独立 B 训练。

主比较为现有 Regression-B R4-AB 与新 FM-B R4-AB。联合训练可能同时改变 encoder、A 和 B，因此主结果解释为“替换 B 目标后的整体系统效应”，不能单独归因于固定表征下 B 的分布建模。

## 训练协议

依据 `config/train/round4_ab.yaml`、`config/train/policy/round4_ab.yaml`、`source/policy/fast_lewam.py`，实施前导出现有权重的实际训练配置并保存差异表；不静默用后续修改过的模板替代历史有效配置。

| 项目 | 冻结设置 |
|---|---|
| 任务 / 数据 | PushT / pusht.h5；Reacher / dmcontrol/reacher.h5，沿用原数据划分、窗口、归一化和预处理 |
| 训练 | 从头联合训练 stage_ab，seed=3072，10 epochs，最后 epoch 10 评测 |
| 时域 | history=1，frameskip=5，action_horizon=5，覆盖 25 环境步 |
| 模型 | 当前 encoder/projector、192 维、6 层共享 DiT、6 heads、MLP 768、legacy token；保持 prefix 动作因果约束 |
| 优化 | effective batch=128，AdamW lr=5e-5，weight_decay=1e-3，gradient_clip=1，沿用 bf16 与 activation checkpointing |
| A | 原动作 FM 损失、goal token 注入、动作采样及训练 timestep 分布 |
| A→B | joint 动作输入，10-epoch 记录/预测动作混合课程，detach_clean_action=false，保留 B→动作→A 梯度 |
| 损失 | B weight=1，SIGReg weight=0.09、knots=17、num_proj=1024；原 source-timestep 权重及阈值 0.2 |

显存不足可减 micro-batch 并累积，保持 effective batch；记录实际更新数、样本数、峰值显存、训练耗时。为 latent 噪声与 latent flow 时间新增独立 RNG 流，避免改变已有数据顺序、动作噪声和动作混合 mask。共有参数初始化尽可能与原 seed 对齐，新增参数单独记录。基线有效配置或初始化无法完整重建时，披露历史基线限制，不宣称严格同初始化消融。

## B 的目标与接口

设现有监督目标为完整未来 latent 序列 Z=(z1,...,z5)，ε~N(0,I)，独立采样 τ~Uniform(0,1)：

    Zτ = (1-τ) ε + τ Z
    v_target = Z - ε
    v_pred = B(z0, action_input, source_timestep, Zτ, τ)
    L_B = mean_sample[w(source_timestep) * mean_token,dim((v_pred-v_target)^2)]
    w(t) = min(t / 0.2, 1)

τ 是 latent flow 时间，source_timestep 是现有动作来源/噪声条件，两者用独立输入编码，不能复用一个时间变量。A 的 loss、SIGReg、目标 encoder 的 detach/梯度规则、原始 reduction 和预测动作配原记录未来的监督语义保持原实现。τ 不代替原 source-timestep 权重。

在共享 DiT 上增加必要的 noisy-latent 输入、latent-time 条件和 velocity 输出接口，保持模型深度与宽度。新增参数量和 B 实际 FLOPs单列。对第 k 个 latent 的预测只允许依赖合法动作前缀及不越过 k 的 noisy-latent tokens；禁止由未来 noisy latent 旁路泄漏未来监督。B 不读取评测目标 latent；goal 仅在外部 cost 中使用。

训练只需一次随机 τ 的 velocity 回归；推理从 ε 积分获得完整 Z，再按原 prefix cost 读取对应 block。新接口须兼容 CEM、GF、PO；GF/PO 对动作的梯度穿过全部 latent 积分步骤，不能 detach 或用单步近似替代。

## 推理矩阵与采样约定

本轮方法组合采用 [CVPR Table 1](cvpr_table1_plan.md) 的四种无引导协议及七种引导/优化协议，仅使用轻量 GF/PO 预算，全部关闭规划内部的额外动作裁剪。每种协议均覆盖 25/25、10/10 两组执行/评分长度。其余训练、任务、权重来源、cohort、评测 seed、采样及计时设置保持本计划现有口径；`config/round5/phase5_pre_report2.json` 仅作为这些历史设置与基线资产的来源，不再原样沿用其完整推理矩阵。

| 方法 | guidance |
|---|---|
| P0 | none / GF-L / PO-L |
| P1 | none |
| P2 | none / GF-L / PO-L |
| P3 | none / GF-L / PO-L / PO-refine-L |

共 11 种协议。P0 为 A 单候选直接执行，P1 为随机初始化的 B-CEM，P2 为 A 初始化的 B-CEM，P3 为 A 生成 64 个候选、B 评分选择。P2-GF/PO 先引导或优化生成结果，再初始化 CEM；P3-GF/PO 对全部 64 个候选引导或优化后再选择；P3-PO-refine-L 先评分选出一个候选，再仅优化选中动作。

每个方法条件覆盖两组执行/评分环境步数：25/25、10/10。始终生成 25 步，目标 offset=25，总执行预算=50。两任务均用原 legacy_50 cohort，推理 seed=42，每单元 50 episodes。

- A：Euler、flow_steps=2；P3 N=64，candidate_batch=64、proposal_chunk=512。
- P1/P2：samples=300、topk=30、iterations=30、var_scale=1、solver_batch=1，P2 初始化缩放=1。所有协议均设置 action_bound_mode=none，不使用 cem-clip 或额外候选投影，保留环境原生动作处理；不新增裁剪对照。
- GF/PO 仅使用 CVPR 轻量档：GF 在两个 Euler 生成步各更新 1 次（总计 2 次），PO 在生成完成后更新 2 次；step_size=0.01、max_rms_offset=0.2。GF 配置 last_steps=2、inner_steps=1，PO 配置 inner_steps=2；验收实际更新次数，不能沿用旧 inner_steps=5。GF 每个 Euler 步使用局部 reference，PO 使用完整生成动作作为 reference；P3-PO-refine-L 的 reference 为选中的完整动作。GF 与 PO 不宣称等计算量；P3-GF/PO 的次数按每候选记录。
- 新 B 默认 Euler、latent_flow_steps=2、每候选 K=1。这是控制预实验成本的预先指定工作点，不按正式成功率选积分步数或 K；A/B flow steps 分别记录。
- 同一 episode/replan 内，所有候选共享同一 latent 初始噪声；CEM 迭代及 GF/PO 优化过程固定该噪声。噪声按 task/seed/episode/replan 派生，不能随 batch、chunk 或候选顺序变化；新的 replan 可以换噪声。
- K=1 时评分使用该预测路径的原有 goal cost。多样本 K>1 仅作后续扩展，使用平均 cost，而非最优样本 cost，不计入本轮核心预算。

FM 模型共 2 任务×2 组时域×11 种协议=44 个逻辑及独立评测单元、2,200 episodes。25/25 与 10/10 的执行长度不同，P0-none 也须分别评测，不存在仅评分长度不同的 alias。Regression 基线同样 44 个独立单元、2,200 episodes；两臂合计 88 个独立单元、4,400 episodes 的结果覆盖。FM 自己的 P0 必须新评，因为联合训练改变了 A。新增执行量扣除通过兼容性审计的历史基线，其余使用原权重重评。

## 对照、诊断与验收

基线权重、cohort 路径与 SHA256 取自 `config/round5/phase5_pre_report2.json` 的 pusht/reacher 条目。两臂使用本轮相同的 11 种协议及两组时域。复用须核验 checkpoint、cohort、动作处理、推理参数、成功语义与代码兼容性；建立 source_refs，不把 alias 当新样本。Phase5 的 P1/P2 cem-clip 结果及旧 GF/PO 较大预算结果不能直接充当本轮基线；CVPR 结果也不能仅因方法名相同而复用，须核对本轮 legacy_50 样本身份、RNG 设置和其余协议。若原结果不兼容或对应条件缺失，使用原权重重评，不默认增加训练 run。

1. CPU/小批 smoke：目标插值端点、shape、τ/source-timestep 独立性、prefix 因果性、无 goal 泄漏、梯度穿过 B 积分到动作及 A、原分支兼容；显式确认目标 encoder 梯度规则没有意外改变。
2. 训练 smoke 后完成两个完整训练 run。记录 velocity loss 与积分后的 latent MSE，不能把 FM velocity loss 数值直接与旧 latent MSE 比优劣。
3. 完整执行上述矩阵，不根据前几个条件成绩删减 GF/PO 或任务。主要闭环终点预设为各任务 P3-none、25/25 的成功率配对差；Reacher 10/10 及其余矩阵为探索性分析。
4. 固定输入诊断采用同一原 actor 的初始状态候选动作池，原/new 模型各用自己的 encoder 编码同一观测和目标；比较 simulator-grounded ranking、top-1 regret、选中动作真实效果及采样稳定性。不同 encoder 的 latent 距离绝对值不直接横比；该诊断也不能独立分离表征与 predictor 收益。优先复用身份匹配的候选真值，新增真值工作量单列。
5. 计时沿用 batch=1/50、预热10次、同步测量50次，FP32推理，记录单次规划 p50/p95、累计规划时间、A/B网络前向与反向次数、峰值显存。B 积分的每个网络调用都计数，不能把一次 sampler 调用计成一次前向。

逐 episode 保存 success、首次成功步数、实际规划次数和失败信息。成功率给 Wilson 区间；配对差按来源 episode 聚类 bootstrap，不能将候选、重复计时或相同来源起点视为独立样本。单训练 seed、单评测 seed 仅为预实验，不声称训练稳定性。若 FM 改善成功率但显著增加延迟，同时报告收益与代价。

## 产出与执行约束

计划文件为本文件；实施时新增 `config/round5/phase6_1.json`、相应 FM 训练配置及 runner。结果写 `outputs/round5/phase6_1/`，报告写 `docs/report/round5/round5_phase6_1_report.md`。归档 frozen config、代码差异、数据/权重/cohort 哈希、训练日志、逐 episode trace、source_refs、汇总 CSV/JSON 和诊断图。

执行记录：本计划已完成实施与实验。最终报告见 [round5_phase6_1_report.md](../report/round5/round5_phase6_1_report.md)，机器可读汇总位于 `outputs/round5/phase6_1/analysis/summary.json`。初始运行时身份不一致的结果保存在 `outputs/round5/phase6_1/archive/`；受外部 GPU 负载影响的计时记录保存在 `outputs/round5/phase6_1/analysis/timing_conditions/` 下的单独归档中。最终汇总使用统一运行时代码的重评结果和干净重测计时。

实施时训练及评测优先使用 GPU0–3，允许使用 GPU4–7。每个实验启动前检查所选 GPU 的剩余显存，并为运行时波动及其他任务保留合理余量；如果显存足够，允许在同一张显卡上同时运行多个实验，以充分利用显卡资源。每条 GPU 命令显式设置 CUDA_VISIBLE_DEVICES。
