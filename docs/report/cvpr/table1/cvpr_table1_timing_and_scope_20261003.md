# Table 1 范围修订、Reacher 新行与计时依据

日期：2026-10-03。对应 [正式计划](../../plan/cvpr_table1_plan.md)。本文落实用户对首轮复查的范围纠正；未启动正式实验，未修改规划实现。

## 1. 本轮范围

- Baseline 四任务均保持执行／评分 25/25，沿用已约定的其它设置。
- CoWM 主结果使用 `action_bound_mode=none`，保留环境原生动作处理；候选裁剪只作为测试。
- Phase5 只吸收 Reacher 短时域 Selection，不吸收其它任务／原生结构，不等待其它结构训练来更换本轮权重。
- 同 A+LeWM、Random-64 等评分器对照属于 Table 2，本轮不新增。
- 在线 `rank_1.0` 不进入本轮；PO5/GF10 原已列在完整推理矩阵，保留为测试，不新增主表方法行。

首轮复查中扩大评分器对照、调整 baseline 时域及将 clip 作为主版本的建议，由这些范围决策取代。

## 2. 裁剪实现问题的准确范围

当前 `source/policy/round4.py` 的 `post_opt_refine` 路径是：生成候选 → `_project_candidates` → B 评分选择 → `post_optimize_actions` → 写入执行缓存。优化后没有再次调用物理边界投影。因此 clip 模式仅保证优化前候选合法，不能保证最终动作合法。例如某维归一化合法上界假设为 1，动作 0.99 在优化后到 1.01，仍会进入缓存；优化前的投影统计不能证明最终动作合法。

裁剪测试须在 refinement 结束后再次投影，记录优化后的原始动作、最终动作与投影统计。主版本 `none` 不加此投影。

`source/model/fast_lewam/jepa.py` 的 `_latent_cost_from_clean_actions` 直接对当前动作做 B 预测，没有物理边界投影。RMS 偏移约束限制相对参考动作的修正幅度，不是动作合法区间约束。本轮裁剪测试保持原语义：对未投影动作的代价做梯度下降，随后投影，再最终选择／执行；不能声称每个内层梯度都优化投影动作的代价，也不为本轮另改梯度目标。

此外，当前 `round4.py` 的峰值显存读取在 `post_opt_refine` 前，可能漏记优化反传峰值。正式计时在完整规划调用外层复位、同步并读取，不直接采用该提前读数。

## 3. Reacher 的主表新增方式

新增完整四任务行 **CoWM-Selection†**，原 Selection 行保留。

| 配置 | TwoRoom | PushT | Reacher | Cube |
|---|---|---|---|---|
| LeWM / LeFlow / 后续 Sub-JEPA | 25/25 | 25/25 | 25/25 | 25/25 |
| CoWM-Selection | 25/25 | 25/25 | 25/25 | 25/25 |
| CoWM-Selection† | 复用 Selection | 复用 Selection | **10/10** | 复用 Selection |

数字为执行／评分环境步数。表注说明“† 仅 Reacher 使用 10/10，其它任务与原 Selection 相同”。复用单元有 alias 身份，不重复计入样本量。每 seed 的四任务平均重算，再跨六 seeds 求 mean±sample std。

Reacher 固定同一 Round4 AB checkpoint、5×5 输出、25 步生成、Euler=2、N=64、无候选裁剪、goal offset=25、eval budget=50。只修改 `execute_steps=10` 与 `score_horizon_blocks=2`。第 10 步预测 latent 仍与原第 25 步目标图像 latent 比较，并未将目标采样位置改为第 10 步。

已完成 Phase5 legacy cohort 的 P3：25/25 为 45/50；10/10、5/5 均为 50/50；10/25 为 17/50，5/25 为 15/50。值得加入的是评分与执行同时缩短，不能只改执行前缀。来源：[Phase5 多任务报告](../round5/round5_phase5_pre_report2.md)。正式主表重新运行六 seeds，不直接填旧探索值。

选择 10/10：已有成功数与 5/5 相同，但完整 50 步预算下最多重规划 5 次而非 10 次。在正式结果出现前确定，不再按正式成绩挑 5/5 或 10/10。当前 A 仍生成 25 步，B 仍生成完整预测后读取前缀；单次规划延迟未必显著减少。

此行比较预先声明的部署配置，Reacher 重规划频率不同于 baseline；原 25/25 行提供同频率参照。短时域收益不能只解释为模型质量提升。

主版本与裁剪测试各增加六个 Reacher 单元，共 12 单元／600 episodes。第一阶段总量为 924 独立执行单元／46,200 实际 episodes；不含独立计时与后续 Sub-JEPA。

## 4. P0-PO2 的定义

P0：A 只生成一条 25 步动作序列，无多候选选择，生成用两次 Euler 积分。PO2：生成后，用 B 的末端预测与目标 latent 的 MSE 对动作张量做两次梯度下降；步长 0.01，梯度 RMS 归一化，相对原生成动作的 RMS 偏移上限 0.2。

A/B 权重全程固定；只更新动作，不收集新的真实分支监督，不在线训练模型，不调用 CEM，也不生成 64 候选。P3-PO-refine 则先从 64 条中选一条再优化，属于另一个配置。flow step=2 与 PO update=2 是两个独立计数，不能把其成本仅写成“两次生成前向”。

## 5. 三篇论文与公开实现的计时证据

### LeWM

[论文 Figure 3 与 Appendix D](https://arxiv.org/html/2603.19312v1) 明确 planning time 是 50 次运行的平均；使用单 NVIDIA L40S、CEM 300 candidates、top-30、horizon=5、frame skip=5、完整执行 25 步；迭代数为 PushT 最多 30、其它任务最多 10。

论文未充分交代该图计时窗口、warmup、CUDA 同步、batch 与缓存状态，不能确认是否包含所有编码／预处理，更不能宣称本研究的完整 policy-call 协议逐项复刻该图。

公开 [CEM 默认配置](https://raw.githubusercontent.com/lucas-maes/le-wm/main/config/eval/solver/cem.yaml) 为 300×30、top-k=30、batch_size=1。本计划保留已约定的 300×30；论文应称公开代码默认配置下统一重评，不将其它任务 30 iterations 写成原论文 Appendix D 的逐项设置复现。25/25 不变。

### LeFlow

[论文 Table 2 与 Section 4.2](https://arxiv.org/html/2608.24855v1#S4.SS2) 的 Eval Time 单位为秒，覆盖完整 50-episode 评测，CEM 与 LeFlow 时间均取五次运行平均；正文明确包括环境 stepping/rendering。这不是 batch=1 单次重规划延迟。

[官方 eval.py](https://raw.githubusercontent.com/hsiangwei0903/LeFlow/main/eval.py) 在模型／数据集加载后，用 `time.time()` 包住 `world.evaluate_from_dataset(...)`，传入 `video_path`；返回后结束计时，结果文本写入在窗口外。该外层窗口没有单独显式 CUDA 同步或 warmup。视频相关具体写入边界依赖框架，不能仅凭入口确认所有内部细节。

### Sub-JEPA

[论文 Section 4](https://arxiv.org/html/2605.09241v1#S4) 报告六 seeds 的规划成功率 mean±std；未发现独立推理延迟表或 planning-time 测量协议。

[官方 README 的 Evaluation](https://github.com/intcomp/Sub-JEPA#evaluation) 调用 `le-wm/eval.py`，六 seeds 为 42、100、2026、3407、1234、4444。仓库 LeWM 子模块固定在 `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`；该版本 [eval.py](https://raw.githubusercontent.com/lucas-maes/le-wm/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/eval.py) 用 `time.time()` 包住整批 `world.evaluate(...)` 并传入视频目录，加载在窗口外。

这确认公开入口记录整批 evaluation_time，不能据此宣称论文报告了 LeWM Figure 3 同口径延迟。子空间正则化是训练损失，不在规划时重算。

## 6. 本研究的统一计时方案

主表列名建议 **Initial replanning latency (ms), batch=1**，或简写 Planning latency 并在 caption 中解释初始决策。全部方法使用自己的冻结权重与原生 history，测共同输入／输出边界。

| 内容 | 主延迟是否计入 |
|---|---|
| CPU 内存中的原始图像处理、数据转换与传到 GPU | 是 |
| 当前观测、目标与所需 history 的 encoder | 是 |
| A 生成、B rollout、CEM、PO/GF 反传、选择 | 是 |
| 动作反归一化、返回 CPU 可执行动作 | 是 |
| 权重加载、数据集打开、起点读取 | 否 |
| 环境 stepping/rendering、视频与报告写入 | 否 |

1. 同一空闲 GPU3 串行测所有方法；FP32，关闭 autocast、TF32、compile，CPU／BLAS 线程为 1。记录硬件、驱动、库与计时配置。
2. 每任务 seed=42 清单前五个状态。恢复方法原生初始 history/padding、空动作缓存、初始目标 cache、无残余 CEM warm-start；保留 baseline 的 warm-start 配置，不禁用其功能。
3. 每状态预热 10 次再测 50 次。恢复策略状态与 RNG 在窗口外；kernel／allocator 保持热态，不逐次清空 CUDA allocator。每个样本都触发一次真实 planning event。
4. 同步 CUDA → `perf_counter()` 起点 → 完整 policy call → 同步 CUDA → `perf_counter()` 终点。GPU event 可补充 kernel 时间，不能替代包含 CPU 工作的主指标。
5. PO/GF 保留动作梯度，模型参数固定。关闭额外诊断和序列化；完整动作统计另行采集。
6. 每任务／方法 250 样本。主报算术 mean，附报 P50/P95；四任务汇总等权平均四个任务 mean，不混淆 pooled percentile 与逐任务 percentile。
7. 完整调用外层测 allocated 显存峰值，记录加载后基线占用；禁用内部重复峰值复位，避免抹掉预处理等此前阶段的峰值，不复用 refine 前 event 字段。
8. 检测到 GPU／明显 CPU 干扰时保留记录并整批重测，以预定干扰规则采用批次，不按最快成绩挑选。

测量对象是运行时预热后的初始决策，目标尚未编码；不声称等于已有完整历史／目标缓存的 MPC 稳态延迟。实际稳态开销使用成功率运行中的实际事件记录补充，不能暗中让某方法使用缓存输入、另一个使用原始输入。

`stable_worldmodel/policy.py` 只在 action buffer 为空时启动 solver；连续调用同一 policy 50 次可能只有两次完整规划，其余取缓存。不能直接平均普通 `get_action` 循环。Phase1.6 的 `_needs_flush` 可强制重规划，正式方案还需恢复 history、goal cache 与 warm-start。

另外报告 batch=50 评测总耗时、实际环境步数、每 episode 规划次数／累计规划时间。该数据接近 LeFlow 的系统耗时概念，但示例视频／详细归档开销不同，不声称完全复现其边界。总耗时除以 50 是并行吞吐指标，不能当单环境规划延迟。

Reacher† 额外报告 `mean_latency / execute_steps` 与实际累计规划时间。若单次均为 20 ms，在无提前成功的 50 步预算下，25/25 最多约 40 ms 规划，10/10 最多约 100 ms；单次延迟相近不等于整段控制计算量相近。

## 7. 在线与大预算和 Table 1 的关系

在线 rank_1.0 使用新增真实环境分支监督，属于不同数据设定，与当前冻结离线模型的 Table 1 没有直接关系，本轮排除。推理时 PO 更新动作不等于在线训练模型。

PO5/GF10 仅与原计划的完整推理配置测试有关，不作为主表方法，不作为继续扩大主表的理由。主行固定 Selection=P3/S2/N64、Refinement=P0-PO2/S2，加上 Reacher†。

本轮不新增 CEM 预算扫描、评分器控制或训练机制实验。正式运行前仍需实现裁剪测试最终投影、完整计时与归档；本文只明确范围和协议。
