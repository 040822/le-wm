# Round 5 Phase 6.3 执行记录

2026-10-06，执行中。此文记录已经验证的事实，尚无正式加速或成功率结论。完成条件仍为原计划的四个方向、最终有效版本及组合的正式计时和完整闭环验证。

## 协议与资产

冻结设置直接读取 `outputs/cvpr/table1/v1/frozen_config.json`，方法注册与完整 policy 计时调用复用主表入口。P0、P3、P3-PO-refine-L/H 保持原规划预算、动作处理、样本、随机种子关系和成功判据。

已核验四任务的 8 个 CoWM 权重/配置文件：文件 SHA256 与主表资产清单中的 archive/source SHA256 一致。24 个 cohort 均为 50 episodes，语义 SHA256 与主表冻结诊断记录一致，并另外记录文件 SHA256。没有重新抽样。

独立产物为 `outputs/round5/phase6_3/`：`frozen_config.json`、`provenance/assets_and_cohorts.json`、`provenance/environment.json` 和按代码身份保存的 `provenance/code_snapshots/`。主表目录未写入。

## 已完成的诊断

E0 TwoRoom P3-PO-refine-H 的组件 profiler 使用 GPU6、batch=1，预热 10 次、采集 5 次完整调用，保留 CPU/CUDA 时间线与算子明细。该设备同时有外部驻留计算进程，且 profiler 本身引入开销，**这些数值不能进入正式延迟或加速表**。

证据路径：`profile/E0/cowm_p3_po_refine_h__main__tworoom__seed_42/attempt_001/` 下的 `timeline.json`、`operators.json`、`operators.txt`、`result.json`。

5 次调用累计有 35 次 `aten::isfinite`、35 次 `aten::item` 和 35 次标量 D2H。这与每次调用 2 个入口 latent 检查及 5 个 refinement 梯度检查一致，支持继续测量 E1 的假设，尚不能证明延迟收益。全调用另有 H2D、D2D、CPU 返回和同步；不同阶段的完整归因及长尾诊断仍待完成。

## E1 正确性

E1 通过实验模型实例的临时适配器启用：保留原入口 latent 检查，每步计算梯度有限性布尔值，在设备端保存，循环结束后一次读取；异常时抛出 `FloatingPointError`，不返回动作。归一化、更新、RMS 约束与次数保持原实现。诊断打开时原 zero-gradient 统计仍保留，未将诊断关闭计为收益。

固定 TwoRoom 权重、seed=42 的全部 50 个初始状态已完成 GPU parity。每状态使用与主表正式计时同一 seed schedule 中的第一个噪声 seed（910000 + state_index × 5），比较 encoder latent、全部候选生成输出、B cost、refinement 每步 cost 和动作梯度、selected index 及最终 CPU 动作。50/50 状态通过 atol=1e-6、rtol=1e-5，全部观测量最大绝对误差为 0，索引一致。

证据路径：`validate/E1/cowm_p3_po_refine_h__main__tworoom__seed_42/attempt_001/` 下的 `result.json` 与 `raw_parity.pt`。这只证明该任务和方法的初始状态 parity，不能替代四任务、多方法、闭环行为验证。

单元测试目前 7/7 通过，包括 E1 每步梯度和最终动作逐位相等、模型参数无梯度、NaN/±Inf latent 与注入异常梯度被拒绝、异常后可恢复调用，E2 RNG/cost/动作梯度、修改与编码边界失效、不同 batch 视图及冻结参数约束，E3 两个 policy 适配器 batch=1/50 的预处理与输入所有权，以及 E4 返回 FP32 latent 并恢复原编码方法。日志：`provenance/unit_tests.txt`；命令：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m unittest tests.test_round5_phase6_3 -v`。

## E2 / E3 实现与扩展验证

E2 缓存原生 `z_condition` 和 `latent_input` 的投影。缓存键包含底层存储地址、offset、shape、stride、dtype/device、修改计数和 autocast 状态，保存输入引用防止地址复用。需要梯度的输入绕过缓存，参数必须冻结。每次 encoder 调用前清空缓存，因此不跨重规划缓存任何 goal latent。固定 timestep 和 AdaLN 保留原生计算，尚未评价其它可复用方案。

真实权重下，E2 16个单元共800个初始状态 parity 全部通过；原始缓存轨迹共记录5,600次命中、3,200次首次计算、800次重规划建立，均值为每状态7次命中。这些次数只证明缓存被重用，不证明延迟收益；冷启动及命中路径的CPU/CUDA时间仍需单独分析和正式计时。

另对 TwoRoom P3-PO-refine-H 做了单独的E2缓存路径诊断：262次调用（2次 parity、10次预热、250次状态测量）对应262次缓存清空、1,310次投影 miss 和3,406次 hit。`z_condition` miss/hit 的CPU调用中位数分别为0.184/0.018 ms，`latent_input`为0.084/0.010 ms；miss的CUDA event投影中位数分别为0.078/0.047 ms，缓存清空中位数为0.027 ms。诊断插入了计时与CUDA event，不是端到端延迟证据，且只覆盖该代表单元；完整E2收益仍以正式交错计时为准。原始记录：`cache_diagnostic/cowm_p3_po_refine_h__main__tworoom__seed_42/attempt_001/cache_costs.json`及`cache_projection_events.jsonl`。

E3 实现原生 CPU 图像转换链的批量执行，消除逐帧 Python dispatch 与最终 stack 复制；ToImage、FP32/scale、Normalize、Resize 的顺序和参数不变。外层 deepcopy、H2D 及 CPU 动作返回仍为主表原路径，没有 pinned memory 或 nonblocking 假设。单测覆盖 P0/P3 的 batch=1/50、uint8/FP32、输入不被修改及返回图像与输入不共享可修改内容。真实权重 seed=42 parity 已完成16/16单元，所有审计比较量最大绝对误差为0；收益评价待正式计时。

E1→E2→E3→E4 的四任务初始状态验证及原始输出审计已完成56/56单元：E1为8/8个适用单元，E2/E3/E4各16/16。E1/E2/E3的候选、cost、动作梯度及CPU缓存动作通过容差，selected index变化为0；E3最大绝对误差为0。E4所有输出有限，但在800个方法/状态配对中有87个selected index变化，最大最终CPU动作绝对差为1.793，明确属于精度变化。审计保存在`summary/initial_state_validation.json`，包含50状态的原始tensor、梯度次数、固定RNG及数值差异。

E0 四任务×四方法的带阶段标签 profiler 已完成 16/16 个单元。算子明细含输入转换与 H2D、encoder/projector、A 生成、B cost/评分、refinement 前向/反向/归一化/RMS 更新、CPU 返回标签，并保留 CUDA 时间线。一个较早的首轮 refine-H profiler 使用代码栈采样，单独保存；阶段标签汇总针对其余16个目标单元。标签内的 CPU 累计时间是5次 profiler 调用总和，标签有包含关系，且设备与其它任务共享；仅用于定位热点，不进入正式延迟结果。

`summary/profile_stage_summary.json` 将最新一轮标签 profiler 汇总为每调用均值。输入变换CPU self time按方法在四任务间约为1.55–1.65 ms，CPU动作返回约0.034–0.048 ms；CUDA设备标签均值中，encoder/projector约8.3–8.6 ms、A约14.4–15.7 ms、B cost约7.5–7.8 ms，refinement-L/H约44.4/98.5 ms。范围仅用于定位热点；GPU共享、profiler插桩和嵌套标签使其不能相加或代表完整调用延迟。它说明E3主要触及的输入处理比A、encoder及refinement阶段小，最终价值仍由正式计时决定。

扩展验证曾发现：直接复用首次输入的 `_needs_flush` 标志会让第二次动作调用再次重规划。旧扩展批次虽数值一致，但不计作动作缓存验收；原始 attempt 保留。已修正为清除该标志，并断言第二次调用没有任何新增编码、候选、cost 或梯度计算；修正后的 E1 四任务验证另行启动，其余版本用修正后的入口继续执行。

## 闭环入口验证

已实现独立入口 `scripts/round5_phase6_3_closed_loop.py`，引用主表资产和 cohort，复用原生闭环 evaluator；每个单元严格 50 episodes，保存配置、身份、命令、结果和 `episodes.jsonl`。原始批量 planning wall time、逐回合执行步数、规划次数及 batch exposure 保留；amortized 字段只表示计算量分摊，不能称为单环境延迟。

E0 四任务四方法六 seeds 闭环矩阵已完成96/96，其中95个单元新运行，P0 TwoRoom seed=42 复用一个有效的历史同cohort结果。与冻结主表比较的是结果标签，不是完全相同代码的重放：两边Git commit相同，但主表记录24个源文件、Phase 6.3 冻结97个源文件，24个重叠文件中20个哈希相同、4个不同（`source/common/round4_eval.py`、`source/model/fast_lewam/jepa.py`、`source/policy/fast_lewam_eval.py`、`source/policy/round4.py`）。在这个限定下，95/96个单元的50个success标签完全一致；唯一差异在P3-PO-refine-H PushT seed=2026，episode `(dataset_episode=13352, start_step=116)` 主表成功而E0失败。主表运行在GPU2（记录利用率100%），E0在GPU5。该episode共同24步的动作完全相同，但首个记录状态最大绝对差为0.972；主表第24步成功终止，E0运行50步后失败。由于源文件哈希和设备均不同，不能将状态差异归因于单一原因或宣称精确复现。逐episode与源码指纹审计见`summary/baseline_closed_loop_parity.json`。

E4 四任务四方法六 seeds 闭环矩阵已完成96/96，共4,800 episodes；16/16个方法-任务组均通过episode键与trace校验。下表给出相对E0的四任务等权配对成功率差、按六个seed均值计算的样本标准差、分层episode配对bootstrap 95%区间，以及每回合执行步数、重规划数和规划时间暴露。E0在GPU5运行，E4在GPU4运行；这些运行时间字段只描述批量闭环工作量，不作为单次延迟结论。

| 方法 | 成功率差（六seed SD；95%区间） | 平均执行步数/回合 E0→E4 | 平均重规划数/回合 E0→E4 | 规划时间暴露秒/回合 E0→E4 |
|---|---:|---:|---:|---:|
| P0 | +0.67 pp（0.93 pp；−0.08至+1.50 pp） | 20.538→20.491 | 1.158→1.156 | 0.442→0.308 |
| P3 | +0.08 pp（1.46 pp；−0.75至+0.92 pp） | 19.704→19.764 | 1.134→1.134 | 0.528→0.456 |
| P3-PO-refine-L | +0.42 pp（0.66 pp；−0.42至+1.25 pp） | 19.851→19.723 | 1.137→1.135 | 0.619→0.463 |
| P3-PO-refine-H | +0.42 pp（1.02 pp；−0.42至+1.25 pp） | 19.800→19.703 | 1.138→1.137 | 0.686→0.424 |

四个成功率区间都包含零；这些结果没有显示可确认的成功率改善，也不能证明非劣。E4初始状态审计有87/800 selected index变化，因此仍按精度变体解释。原始逐seed统计及全部配对转换见`summary/closed_loop_paired.json`。完整矩阵日志为`logs/closed_loop_E4_gpu4.log`，各结果保存在`closed_loop/<version>/<cell_id>/attempt_001/`。日志末尾有MuJoCo EGL context析构阶段被Python忽略的`EGL_NOT_INITIALIZED`异常；96个结果均为有效状态，全部episode trace通过配对分析校验。

## 组合计时与选择门准备

单变量正式计时尚无接受样本，因此现在先固定组合纳入规则：对每个方法分别要求四任务等权配对平均延迟差小于零、按任务分层的配对95%区间上界也小于零；四任务条件必须全部完整。E1只适用于两个refinement方法。C-FP32组合通过此门的E1/E2/E3；C-BF16只有在E4也通过延迟门且现有E4四任务、六seed闭环记录完整时才加入E4。规则已补入实验计划，0/120 accepted timing samples时确定。

组合本身另走每个method-task的`E0→combination→E0`正式计时，状态、RNG、预热、测量次数和计时边界与主计时一致；组合也须满足同一四任务配对门，才进入完整闭环。`scripts/round5_phase6_3_select_combinations.py`冻结候选，`scripts/round5_phase6_3_combo_timing.py`与`_analysis.py`测量组合，`scripts/round5_phase6_3_finalize_combinations.py`判定组合是否有效；闭环入口会拒绝未通过该门的组合，并把alias标记为不需重跑。组合上下文顺序保证E4包住E2 encoder cache失效包装。

组合入口、结果身份/哈希校验和恢复路径已经就绪；新增4项组合单测与原7项Phase6.3单测共11/11通过，新增脚本均通过`py_compile`。尚未生成候选组合、未运行组合计时或闭环，也未改动冻结的模型源码与单变量计时driver。

## 待完成矩阵

| 项目 | 当前证据 | 尚需执行 |
|---|---|---|
| E0 | 四任务四方法 profiler 16/16；闭环矩阵96/96（95新运行、1有效复用）；跨源码快照对照主表success标签95/96单元一致 | 正式计时、长尾汇总；保留并解释1个主表/E0 PushT episode差异 |
| E1 | 异常单测；四任务 P3-refine-L/H 初始 parity 8/8 | 同设备正式计时、有效版本完整闭环 |
| E2 | 投影缓存及失效/RNG/batch/梯度检查；初始 parity 16/16；代表单元冷 miss/hit 诊断 | 同设备正式计时、有效时闭环 |
| E3 | 原生 CPU 图像链向量化；两个适配器及输入所有权单测；初始 parity 16/16 | CPU/H2D/动作返回/缓存写入热点分析、单变量计时、有效时闭环 |
| E4 | 全编码入口临时 BF16 适配器；初始状态16/16、87个选择变化；闭环矩阵96/96；配对分析16/16方法-任务组 | 正式计时及是否进入最终有效版本的判定 |
| C-FP32 / C-BF16 | 计时驱动、每方法的候选冻结与组合有效性门已实现；单变量正式结果尚未产生，未选候选 | 单变量计时完成后冻结候选，执行组合正式 bracket；仅对组合延迟有效项做闭环，alias不重跑 |
| 汇总 | 尚无正式结论 | 原始计时、成功率/配对转换/区间、步数/规划次数/累计时间与最终报告 |

正式计时入口 `scripts/round5_phase6_3_timing.py` 按冻结的 `E0→E1→E0→E2→E0→E3→E0→E4` 顺序执行，共120个条件，每个条件保留250个原始样本、端点VRAM及完整设备身份。启动前修正了 attempt 目录重复创建问题，并由97个冻结源文件、资产和parity gate复核通过；CLI首轮启动暴露了未定义的`args.version`读取，尚未加载模型或采集样本。现已改为直接使用冻结的版本顺序，语法编译、7项单测及全部版本gate通过。GPU7启动前无进程、0%利用率、47.37 GiB空闲；CPU load为31.80/128核（0.248）。首个E0 P0 TwoRoom seed=42条件运行后采集到250个原始样本，但结束检查发现GPU7已有外部计算进程（2.42 GiB驻留、2%利用率），因此整个条件被标记失败、0个样本被接受；原始值仅留作审计，不进入延迟分析。随后归一化CPU load升至1.243，也超过0.75阈值。21:32的资源复查仍发现GPU0–7均有外部计算进程或明显利用率，GPU7虽仅1%利用率但仍有3601 MiB外部显存驻留，CPU load为1.483/核。失败 attempt、原始样本和堆栈保留在`timing/runs/E0/cowm_p0__main__tworoom__seed_42/attempt_001/`，日志为`logs/formal_timing_gpu7_retry1.log`。该结果暴露了模型加载后到测量前缺少最后一次idle检查；已在计时driver中补上测量前检查，未改动冻结的推理/评估源文件，重新编译并通过全部版本gate。失败尝试driver SHA为`3a450b3d86b874c9575249bf4f81c88234b75e212b3ff66d4f1fc53fdb545421`，待资源恢复后重试的driver SHA为`e8f89ccb2ca0f5d96aa0890f41f5e02df99b9af2fe049c0713f3a479e76c0013`。21:39的等待器首轮检查仍无合格设备（CPU load 1.329/核，GPU7有4232 MiB外部显存驻留）。首个等待器因遗漏轮询间隔曾连续检查数分钟，未启动实验，现已停止。按用户要求恢复300秒轮询。旧session 16094已无活动进程且waiter锁可获取；当前等待器session 27069已通过提升权限完成首次GPU查询（15:15:50 UTC）：GPU0–6有外部计算进程，GPU7无进程、0%利用率、47.37 GiB空闲，但CPU load为1.457/核，仍高于0.75门槛，因此没有合格设备。当前正式计时仍为0/120 accepted；等待器要求同一GPU连续两次满足资源条件后才会从被拒绝的E0条件恢复，轮询间隔300秒。主表历史计时不替代本轮E0。

执行入口：`scripts/round5_phase6_3.py prepare|profile|validate|timing`，已接入E0–E4；正式交错计时为`scripts/round5_phase6_3_timing.py`，组合计时/分析/最终门分别为`scripts/round5_phase6_3_combo_timing.py`、`scripts/round5_phase6_3_combo_timing_analysis.py`、`scripts/round5_phase6_3_finalize_combinations.py`；组合闭环入口为`scripts/round5_phase6_3_closed_loop.py`，配对统计为`scripts/round5_phase6_3_closed_loop_analysis.py`，正确性审计为`scripts/round5_phase6_3_summary.py`。后续完成正式计时、收益筛选、有效项/组合闭环及最终报告。
