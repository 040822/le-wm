  # 评测协议修订与第三轮在线学习预实验计划

  ## 目标与原则

  保留“四阶段”流程，但将第1.5阶段视为 Fast-LeWAM 的准备工作，不阻塞 E0 在线预实验。E0 没有正收益时，先诊断，再允许一次有限预算的 Fast 验证。

  第三轮检验的核心假设：

  > 在固定视觉表征与规划协议下，真实交互数据能否比等更新次数的离线续训更有效地改善世界模型的闭环成功率？

  第一阶段不重训；后续只使用 GPU0–3，训练与 EGL 渲染均显式绑定允许的设备。暂不加入 RL、Stage C、新结构或新的 CEM trick。

  Round 3 与 Round 4 允许并行推进：Round 3 的协议登记、代码实现和 CPU/dry-run 可以立即并行；Round 3 的 GPU 训练与 EGL 评测必须等待 GPU0–3 有明确资源窗口，不与正在运行的 Round 4 争抢设备。Round 4 当前运行继续完成，并在报告中标记为探索性结果，不把它当作无数据暴露的严格对照。

  ## 第一阶段：利用已有权重修订四任务评测

  范围：Cube、Reacher、Push-T、TwoRoom 全部审计。

  1. 固定历史基准。 登记已有 E0、E3、E5 的完整 checkpoint、训练配置、数据及评测版本。使用预定 final 权重；缺失模型记为缺省，不补训练，不以中间测试峰值替
     代。保留原协议与历史结果，新增独立的修订协议版本。

  2. 分别审计两类问题。
      - 成功判据：查明实际代码、物理量、单位和容差，验证目标设置后 success 是否正确刷新。
      - 初始条件：记录初始成功比例、起点—目标距离、合法保持动作成功率及首次成功时间。
      - 保留原任务语义，例如 Reacher 仍为当前的 qpos_match；释放、稳定保持等先作辅助指标。

  3. 分开验证采样与容差修改。 比较“原协议”“仅修改采样”“仅修改容差”“两者均修改”。修订采样排除初始成功，依据任务对应的物理距离分层，不以模型成功或失败筛选
     episode。阈值根据任务物理语义和轨迹检查确定，不能根据哪个方法排名更高来选择。

  4. 建立开发与测试隔离。
      - 开发 cohort：每任务50个episode，供协议检查及有限模型选择使用。
      - 最终测试 cohort：每任务200个episode，协议和配置冻结后才运行。
      - 按原始轨迹划分，避免重叠窗口跨开发集、测试集和在线采集池；记录已有离线权重可能接触过的数据，不能据此声称完全未见数据泛化。
      - 同一协议内所有方法使用相同 cohort、执行预算和 CEM 设置。先保留25步 goal offset、50步执行预算；若不适用，单独报告，不能静默改变。

  交付与验收： 四任务的判据、容差、采样、预算和 cohort 清单；历史／修订结果及配对差值；初始成功与保持动作统计。所有具体容差必须在本阶段形成数值表并冻结，
  后续实施不得自行选择。无法确定合理判据的任务标为未验收，不进入在线实验。

  200 episodes 用于最终确认，不用于每次修改后的重复调参。

  ### 当前状态与转入条件

  Phase 1 已完成：四任务的 predicate、cohort、历史结果和 `round3_revised` final 结果已经冻结。Phase 1.5 的用户决策记录在 `docs/report/round3/phase1.5/round3_phase1.5_decision.md` 和 `config/round3/phase1_5_selection.json`。

  Round 4 已在运行，Round 3 后续先推进不占 GPU 的登记、实现和 CPU/dry-run。任何新的 GPU 训练或 EGL 评测都必须显式绑定 GPU0–3，并在 Round 4 释放资源后启动。

  ## 第1.5阶段：选择 Fast-LeWAM 在线起点

  基础候选仍限定为 192维、legacy timestep/token 的 E3 与 E5；仅比较与各自训练配置匹配的已有权重，不直接修改已有 checkpoint 的训练语义。用户已明确选择 E5 作为全局 Fast 基线：共同 canonical dev 任务的 E5 相对 E3 平均提升约 4.67 个百分点，虽略低于原定 5 pp 门槛，但视为接近门槛并接受该选择。

  - 主指标为修订开发集上的 Stage B 成功率；A、A-shuf 为辅助指标。
  - 在两者均有有效 checkpoint 的任务上，按任务等权比较，报告配对翻转及已有训练种子的差异；TwoRoom 等缺省项不进入选择均值。
  - 原有 5 pp 规则作为历史筛选依据保留；本轮按用户明确决策采用 E5，不将 4.67 pp 四舍五入成达标，也不把该选择表述为统计显著。

  - 不按任务分别挑 E3/E5，不扩展 timestep、token、宽度或 attention 搜索。Phase 1.5
    冻结现有 E5 legacy checkpoint；后续 Round 3 Fast 主线继续使用该 legacy E5。
    `physical_time_type` 的 Phase 3 结果保留为探索性证据，不再作为主线扩展。现有
    E3 的 Push-T/Reacher physical-time checkpoint 只作历史证据。
  - 首轮 Fast 在线实验使用随机初始化 CEM；E6 留作后续独立推理消融。

  交付： 一个全局 Fast 起点、对应权重清单及选择依据。补充训练的 TwoRoom E3 纳入 Phase 1.5 分析，但保留为 supplemental：它已有 `round3_revised` final 结果，尚未产生 canonical dev 结果，不回写 Phase 1 canonical matrix，也不伪装成 dev 统计。最终测试集不参与选择。

  ## 第二阶段：E0 LeWM post-training fine-tuning 在线预实验

  Phase 2 不是从零训练。三组对照都从已经训练好的 E0 LeWM epoch-10 checkpoint
  出发；“online train”在本轮语义中是对该 checkpoint 的后训练微调。冻结组只保留
  同一模型的推理结果，离线续训组和在线适应组从同一权重、同一 optimizer 初始状态
  开始更新。

  任务选择。 从已验收任务中，优先选择修订开发集上 E0 成功率处于20%–80%、具有改进空间的任务；多个任务满足时选最接近50%的任务，同分依次优先 Cube、Reacher、
  Push-T、TwoRoom。若没有合适任务，先补一个操作任务及其离线基线，作为独立准备阶段，不直接扩大在线训练。

  首轮三组对照：

   组别          设置
  ━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   E0冻结        不更新参数
  ────────────  ──────────────────────────────────────────────────
   E0离线续训    只使用原训练数据，匹配在线组更新次数
  ────────────  ──────────────────────────────────────────────────
   E0在线适应    当前模型+CEM采集真实轨迹，离线／在线回放混合更新

  - 冻结视觉 encoder/projector，包括其运行统计；仅更新动力学预测相关参数。
  - 以实际执行动作及其真实后续观测构造监督，不使用未执行的 CEM 候选后果，不跨 episode 拼接窗口。
  - 按 E0 原训练时序接口计算 latent prediction loss，冻结的编码器同时生成输入与监督目标。
  - replay 保留原始观测、实际动作、目标、episode边界和采集模型版本；在线失败轨迹正常保留。
  - 当前协议是“采集恰好100环境步→执行1次 optimizer update”。100 是环境交互步的
    更新间隔，不是 batch size=100。
  - 当前 optimizer minibatch size 为64。离线续训每次更新采样64条原始 replay；在线
    适应每次更新采样32条原始 replay和32条在线 replay。100步内产生的转移全部追加
    到 replay，供后续采样，但不要求在当前这次 update 中全部使用。
  - 有效轨迹不足时先积累，不伪造窗口。AdamW学习率 1e-5、weight decay 1e-3、梯度
    裁剪1.0；离线续训组与在线适应组匹配 optimizer update 次数。

  - 每采集种子最多20,000真实环境步，在0、5,000、10,000、20,000步做冻结参数的开发评估。5,000步检查用于排错和发现退化，不因无提升提前宣判失败。
  - 先跑一个在线采集／更新种子；20,000步开发成功率相对冻结组和离线续训组均提高至少5个百分点时，补至三个种子。所有重复从相同离线 checkpoint 出发，明确它们
    不是独立预训练种子。

  记录 success—环境步数曲线、更新次数、GPU-hours、离线能力遗忘及少量固定状态诊断。采集与测试分离；仿真分支采集若另行开展，全部计入环境步预算。

  决策门：

  - success 正向：补种子、进行200-episode独立测试，再扩展第二个任务。
  - 只有MSE／局部排序改善：不判定成功，不据此扩大计算。
  - success 退化：检查动作对齐、遗忘及闭环状态分布；发现实现错误后修复并重跑对照。
  - 验收后仍无收益：记录“该预算与配置下未见收益”，不外推为所有在线学习无效。

  当前执行状态：Cube 三臂的 CPU/dry-run、mock 主循环和 exact-target collection
  smoke 均已通过，E0 Phase 2 已完成；随后按本阶段决策门完成了一个单任务、
  单种子的 legacy E5 Fast 验证。离线窗口沿用 E0 原生
  `frameskip=5, num_steps=4`，在线 replay 只记录实际执行动作和真实 successor
  observation，checkpoint 命名冻结为
  `{task}_{arm}_envsteps_{environment_steps:06d}.ckpt`。E0 与 legacy E5 均未
  通过 online 扩展门，因此不再启动 Round 3 的额外 online 种子、200-episode
  独立 online 测试或第二任务 online 扩展；完整结果见
  `docs/report/round3/phase2/round3_phase2_experiment_report.md`。

  注意：历史草案曾出现过每1,000步或每50步更新一次的 schedule。当前已确定改为每
  100环境步更新一次：20,000步共200次 optimizer update。每50步会有400次、每1,000
  步只有20次；这些方案的更新预算不同，不能直接比较。100环境步仍不是 batch size，
  batch 语义保持固定的64条 replay 窗口。

  ## 第三阶段：Fast-LeWAM 在线预实验与最终判断

  后续 Fast 主线直接使用 Phase 1.5 选出的历史 E5 legacy epoch-10 checkpoint，不再
  训练新的 `physical_time_type` checkpoint。已完成的 Phase 3 `physical_time_type`
  四任务结果归档为探索性对照；相对历史 E5 的 Stage-B 差值为 Cube +2.0 pp、PushT
  −3.5 pp、Reacher −1.5 pp、TwoRoom +0.5 pp，未显示跨任务统一收益，因此不再投入
  该表示的后续主线算力。

  未来 Fast 在线实验复用 E0 的任务、cohort、环境步预算和评估节点，直接从对应历史
  E5 checkpoint 开始。报告中保留 A／A-shuf 变化，但不把 `physical_time_type` 的
  已完成结果删除或解释为严格的表示因果消融。

  - 冻结 encoder/projector；使用真实执行动作计算 B loss，关闭训练中的预测动作混合。
  - 首版不新增 A 在线训练。共享 DiT 可随 B loss 更新，因此不得声称 A 完全冻结；持续报告 A／A-shuf 的变化。
  - 各模型按自身原生时序目标训练；匹配真实环境步数与更新次数，同时报告计算量，不声称两种结构计算成本完全相等。
  - 若 E0 未见收益，排除实现问题后仍允许一次单任务、单种子、20,000步 legacy E5
    验证；无正向信号则停止扩展。
  - 若 Fast 正向，补种子并完成独立测试；不把“比自身离线起点提高”等同于“超过 E0”。

  实际结果：legacy E5 Cube 单种子已完成 20,000 环境步。20k 成功率为
  Freeze 48%、Offline continue 48%、Online adapt 44%；Online 相对两个对照
  均为 −4.0 pp，未通过扩展门。Round 3 的 online 主线至此停止，Round 4
  继续按原计划运行。

  最终同时回答三个问题：

  1. 真实交互是否优于继续离线训练？
  2. 收益能否在独立评测、多个在线种子和第二个任务上复现？
  3. Fast 的在线收益是否超过 E0 的在线收益，且最终绝对成功率是否更高？

  各阶段结论分别归档。评测修订、模型选择和在线收益分别形成证据，不用下一阶段的结果反向修改上一阶段的选择规则。

  ## 用户追加扩展：剩余三任务的完整训练—曲线评测

  上述“停止第二任务扩展”是基于 Cube 50-episode dev gate 的原始决策记录。随后
  用户明确要求使用相同训练和评测方式完成剩余的 Reacher、Push-T、TwoRoom 三个
  任务，因此本节记录该追加探索，不回写原 gate 的历史语义。

  三个任务均从 Phase 1 冻结的 E0 与 legacy E5 epoch-10 checkpoint 出发，各运行
  `offline_continue` 与 `online_adapt` 两臂，100 环境步更新一次，最多 20,000
  环境步；每个 1,000 步 checkpoint 在同一任务的冻结 final 200-episode cohort
  上评测，形成 0–20k 共 21 个点。每点采用 4×50 batch 聚合为 200 episodes。

  追加实验的训练、曲线和验收结果统一归档于
  `docs/report/round3/phase2/round3_phase2_experiment_report.md` 的第 6 节；当前已完成 6 个训练
  run、12 条曲线和 252 个 aggregate 结果，均通过完整性验证。
