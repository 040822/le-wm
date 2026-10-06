# Round 5 Phase 6.3：CoWM 推理加速实验

日期：2026-10-06。状态：执行中；资产核验、首个 E0 组件 profiler 及 E1 单任务 parity 已完成。正式计时和完整闭环验收尚未完成，进度见 `docs/report/round5/round5_phase6_3_report.md`。

## 1. 目标与范围

在主表权重、样本、规划预算及成功判据下，评估四个加速方向：减少 refinement 同步、缓存固定条件计算、优化预处理与数据传输、仅 encoder/projector 使用 BF16。研究对象为 P0、P3、P3-PO-refine-L、P3-PO-refine-H。

保持 A 两步 Euler、P3 候选数 64、refinement 更新次数 L=2/H=5。主版本使用 `action_bound_mode=none`，不增加候选裁剪；不扩展到其它推理方式、Reacher† 或裁剪版本，不重新训练。

不开展增大环境评分 batch 的实验：单环境计时保持 batch=1，闭环环境 batch 保持 50，评分 `solver_batch_size=1` 不变。吞吐优化如后续开展，另立实验，不能将批量耗时除以环境数称作单环境延迟。

## 2. 与主表一致的冻结设置

协议依据为 `outputs/cvpr/table1/v1/frozen_config.json`，方法定义复用 `source/common/cvpr_table1.py`，评测及计时边界复用 `scripts/cvpr_table1.py`。旧 Round4 加速报告仅用于提出假设，不替代本轮基线测量。

| 项目 | 设置 |
|---|---|
| 任务 | TwoRoom、PushT、Reacher、OGBench-Cube |
| 评测 seeds | 42、100、2026、3407、1234、4444 |
| 每任务每 seed | 50 episodes |
| 样本 | 直接复用 `outputs/cvpr/table1/v1/cohorts/<task>/seed_<seed>.json`，不重新抽样或过滤初始成功样本 |
| seed 关系 | cohort=评测 seed；环境=seed+10000；策略=seed+20000 |
| 协议 | legacy，沿用主表环境成功判据及处理流程 |
| goal offset / 环境步预算 | 25 / 50 |
| horizon / receding horizon / action block | 5 / 5 / 5 |
| 生成、评分、执行跨度 | 25 / 25 / 25 个环境步 |
| 闭环环境 batch | 50 |
| A 生成 | Euler，2 步；P0 单候选，P3 系列 64 候选 |
| B 评分 | 原生 parallel_prefix / strict_causal，完整五块预测 |
| 候选执行分块 | solver_batch_size=1、candidate_batch_size=64、proposal_chunk_size=512 |
| refinement | 先选后优化；L=2、H=5；步长 0.01、最大 RMS 偏移 0.2 |
| 动作处理 | none，保留环境原生动作处理和主表反归一化 |
| 目标 latent cache | false，保留原生 history 与初始化方式 |
| 基线精度 | FP32，关闭 autocast、TF32、compile |

BF16 是本轮唯一预先声明的精度变体：只开启 encoder/projector 的 BF16 autocast，输出转回 FP32，A/B/refinement 保持 FP32；成功率与计时使用同一精度配置。其它设置与主表一致。

## 3. 复用主表权重及资产

直接加载主表归档的四任务 CoWM 权重：

```text
outputs/cvpr/table1/v1/assets/cowm/<task>/checkpoints/r4_ab_seed3072_weights_epoch_10.pt
outputs/cvpr/table1/v1/assets/cowm/<task>/config.yaml
```

运行前核验权重、配置及 cohort 的 SHA256，权重身份以主表 `provenance/assets_manifest.json` 及冻结记录为准。每任务只用该固定权重，不重新选 checkpoint、不替换 backbone、不重训。记录引用路径、哈希和本轮代码版本，主表目录只读，新实验资产写入独立目录。

## 4. 基线与实验矩阵

| 版本 | 唯一变化 | 适用方法 |
|---|---|---|
| E0 | 当前主表实现，新测本地基线 | 全部四种方法 |
| E1 | 减少 refinement 内部 CPU/GPU 同步 | P3-refine-L/H |
| E2 | 重规划内固定条件计算缓存 | 全部四种方法 |
| E3 | 预处理、传输、CPU 动作处理优化 | 全部四种方法 |
| E4 | encoder/projector BF16 | 全部四种方法 |
| C-FP32 | 组合通过验收且有效的 E1/E2/E3 | 全部四种方法 |
| C-BF16 | C-FP32 加 E4 | 全部四种方法 |

E1 在 P0/P3 上没有作用，记录为不适用，不重复计作实验。若某项没有收益，保留单变量结果，但不加入组合。若组合与已有版本实现相同，标记 alias，不重复运行或增加样本量。

组合纳入规则在正式计时接受样本前固定：按每个方法分别判断单变量收益，要求四任务等权的配对平均延迟差小于零，且按任务分层的配对 95% bootstrap 区间上界也小于零；四任务的正式条件须全部完整。E1 只对两个 refinement 方法判断。C-FP32 仅组合满足该规则的 E1/E2/E3；C-BF16 在此基础上，仅当 E4 也满足同一延迟规则且其四任务、六 seed 闭环验证完整时加入 E4。若组合无额外变化或等同已有单变量版本，记录 alias。闭环成功率差及区间照常报告；本计划不据此宣称非劣，也不另设事后非劣界值。

组件筛选只产生待测组合；组合本身完成相邻 E0 bracket 正式计时后，再用相同的四任务延迟准则判定是否有效。未通过的组合保留计时结果，不进入组合闭环矩阵。

### E1：减少 refinement 同步

定位每轮 `torch.isfinite(gradient).all()` 的 host 判断、入口 latent 检查及其它 host 标量读取。探索设备端累计检查、在返回动作前统一读取，保留非有限值检测和拒绝返回异常动作的行为。入口检查可在边界验证后复用，不能直接删除异常保护。

不改变 cost、梯度归一化、更新规则、RMS 约束或迭代次数。单独测试 NaN/Inf 输入和梯度异常路径。计时模式已经关闭的诊断不计作新增收益。

### E2：固定条件计算缓存

探索复用当前 latent 的条件投影、固定 timestep embedding、锚点投影以及可复用的 AdaLN 调制结果。每次重规划重新建立缓存，明确 A/B 的 timestep 和依赖；B 的候选 batch=64 与选后 refinement 的 batch=1 必须正确索引，不能错误广播或共享。

仅缓存与当前待优化动作无关的量，不切断动作梯度。保持候选 RNG 消费顺序和随机张量一致，不跨重规划缓存目标 latent。分别测缓存建立成本和复用收益，防止准备开销抵消收益。

### E3：预处理与数据传输

先测输入复制、图像转换/归一化、H2D、CPU 动作返回及缓存写入，再针对实际热点减少重复复制、转换、临时张量和传输。检查两个 policy 适配器的差异，不假设 P0/P3 使用同一实现。

不改变图像变换、动作归一化、输入所有权或环境动作缓存语义；不把工作移出计时窗口。不预设 pinned memory/nonblocking 一定有收益，涉及其准备成本时一并计入。正式基线和优化版本使用同一外层输入复制边界。

### E4：encoder/projector BF16

编码 current/goal 后转回 FP32，其余路径保持 FP32。作为会改变 latent、候选排序和动作的独立精度实验，不宣称逐位等价。所有参与该变体的编码调用必须采用相同精度开关，避免只优化计时路径。

## 5. 计时与组件分析

完整延迟复用主表口径：原始观测和目标已在 CPU 内存，到返回 CPU 可执行动作；包括输入复制、图像处理、传输、encoder、A、B、选择、refinement 和动作反归一化，排除加载、样本读取、环境 stepping/rendering、视频和文件写入。

- 每任务使用主表 seed=42 的 50 个状态；10 次预热，50 个状态各测 5 次，共 250 个正式样本。
- 复用 RNG base=910000 和主表采样次序、初始化/重置规则；每次恢复空动作缓存、原生初始 history、无已有目标 cache，保证每个样本触发真实规划。
- CUDA 同步 → perf_counter → 完整 policy call → CPU 输出 → CUDA 同步。CPU/BLAS 线程为 1，TF32、compile 关闭。
- 报告 mean、P50、P95、每任务和四任务等权平均、加载后基线显存与完整调用峰值显存；保留全部原始样本。
- 新测 E0 与优化版本必须在同一设备、相同资源条件下比较。采用预定的交错执行顺序；正式每版本独立遵循主表采样顺序，不按最快批次选结果。
- 主表历史计时实际使用 GPU6 的 RTX 4090，见 `provenance/timing_device_override.json`。优先同设备复测；若设备不同，完整记录覆盖设置，历史数值仅作参考，收益以本轮同设备 E0 为分母。

组件 profiler 单独运行，记录输入处理/H2D、encoder、A、B/选择、refinement 前向/反传/更新、CPU 返回，以及同步和 kernel launch 的时间线。不能把插入逐阶段同步或 profiler 后的耗时填入正式延迟表。

已有 E0 四任务均值参考：P0=24.94 ms、P3=29.88 ms、refine-L=50.03 ms、refine-H=72.93 ms。其长尾明显，且个别任务 P3 均值低于 P0；不同版本总时间差不等同于组件耗时。首先检查长尾来源，不预先归因为某个模型阶段。

## 6. 正确性与闭环验收

E1–E3 使用同一输入和噪声，对比候选动作、B cost、selected index、refinement 每步梯度及最终 CPU 动作。先沿用主表 parity 容差 atol=1e-6、rtol=1e-5，保存最大/平均误差及不一致状态；索引要求一致。不得事后放宽容差来隐藏差异。发生实质行为变化时，先排查原因，不能归类为保持行为的实现优化。

验证 E1 的 NaN/Inf 处理、E2 缓存失效和 batch 索引、E3 原输入不被意外修改与动作缓存一致性。模型参数冻结；refinement 保留动作梯度。

E4 记录 latent、cost、索引和动作差异，必须做完整闭环成功率验证。最终独立有效版本及组合均按四任务×六 seeds×50 episodes 验证，每个方法/版本为 1,200 episodes。相同实现 alias 不重复计数；早期微基准或 smoke 不作为正式成功率证据。

使用主表 cohort 做配对比较，报告逐任务成功率、六 seed 均值/样本标准差、四任务等权平均、配对成功/失败转换及差异区间。小样本下“未显著下降”不等于证明无损；不预设未经约定的非劣界值。另记录实际执行步数、规划次数和累计规划时间，避免把单次延迟当作整个 episode 计算量。

## 7. 执行顺序与资源

1. 核验主表资产、配置和 cohort；归档本轮冻结设置、代码身份及版本开关。
2. 新测 E0，完成组件 profiler 与长尾诊断。
3. E1 → E2 → E3 做单变量正确性与微基准；E4 独立验证数值和编码收益。
4. 有效版本做四任务正式计时，形成 C-FP32/C-BF16 并复验组合收益。
5. 完成最终独立版本和组合的完整闭环验证，输出报告；无收益或失败项保留证据。

组合正式计时对每个入选的 method-task 条件采用相邻 `E0 → combination → E0` 三段，复用同一批 50 状态、每状态 5 次、相同 RNG schedule、空 cache 与完整 policy-call 计时边界；三段须在同一空闲 GPU/驱动完成。组合比较用前后两段 E0 对应样本的均值作为配对基线，保留独立组合计时 driver 与原始样本。被标记为 alias 的组合不重复测量。

GPU0–3 优先，GPU4–7 可用；正式计时在同一空闲设备串行进行，不与训练或其它实验同 GPU 并发。每次启动前检查剩余 VRAM并保留至少主表 6 GiB 安全余量；所有 GPU 命令显式设置 `CUDA_VISIBLE_DEVICES`。sandbox 无法检查或访问 GPU 时，按仓库要求以同一设备限制申请升级权限后重试。

## 8. 产物与完成条件

预定输出：`outputs/round5/phase6_3/`；报告：`docs/report/round5/round5_phase6_3_report.md`。保存冻结配置、主表资产引用/哈希、硬件及软件环境、版本开关、parity 和异常验证结果、profiler、原始计时、逐回合轨迹、成功率与延迟汇总。

报告分别列出单变量和组合收益，区分 FP32 实现优化与 BF16 精度变化，并标记 alias/不适用/无收益项。完成条件为：四个方向均有可核查结论，最终有效版本完成同协议计时及闭环验证，所有收益相对本轮同设备 E0 计算。不预设加速倍数，也不通过改变统计口径、预算或样本提高结果。
