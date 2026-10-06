# CoWM 主表协议记录

更新：2026-10-05。正式 Table 1 已完成并回填论文。以下历史决策与启动前建议保留用于溯源；实际执行以 `outputs/cvpr/table1/v1/frozen_config.json`、逐单元归档及计时设备覆盖记录为准。

## 已完成结果与论文回填

- 正式报告：`docs/report/cvpr/cvpr_table1_report.md`；复核记录：`docs/report/cvpr/cvpr_table1_paper_audit_20261005.md`。
- 948/948 独立评测单元，共 47,400 episodes；158/158 计时条件，共 39,500 正式计时样本。该数量包含推理矩阵和裁剪测试，不是六行主表的独立样本数。
- 主表为 LeWM、LeFlow、CoWM-Selection、CoWM-Refinement (P0-PO-L)、CoWM-Selection†、Sub-JEPA。DeWM、同 A+LeWM 不在本次主表。
- 六个评测 seeds，每 seed 50 episodes；单任务和逐 seed 四任务平均均报告 mean ± sample std (ddof=1)，不表示训练 seed 不确定性。
- Selection† 仅 Reacher 改为执行／评分 10/10，其余任务严格复用 Selection；常规行均为 25/25。
- 实际计时在 GPU6（RTX 4090）完成；原配置 GPU3 的覆盖和当时用户授权见 `outputs/cvpr/table1/v1/provenance/timing_device_override.json`。实际为 10 次预热、50 个状态各 5 次测量，不采用早期计划的 5 状态各 50 次。
- 初始完整重规划延迟与整段控制时间、稳态缓存延迟及等时间预算结果分别解释。论文尚未完成的机制和等预算实验保留 TBD。


## 用户已确认

- 四任务：TwoRoom、PushT、Reacher、OGBench-Cube。
- 每任务使用单个训练 seed 的 checkpoint，6 个评测 seed，每 seed 50 episodes。
- TwoRoom 采用公开代码默认：`goal_offset_steps=25`、`eval_budget=50`。
- CoWM 全量评测各种推理协议，动作生成 `inference_steps=2`。
- 论文命名：CoWM-Selection = P3；CoWM-Refinement = P0-PO。
- 先完成本仓库 CoWM、LeWM、LeFlow；Sub-JEPA、DeWM 后续接入。

## TwoRoom 来源核对

| 方法 | 论文文字 | 公开代码/其他证据 |
| --- | --- | --- |
| LeWM | 附录 F.1：100/150 | `config/eval/tworoom.yaml` 默认 25/50 |
| LeFlow | 附录 A：100/150 | `config/eval/tworoom.yaml` 默认 25/50 |
| Sub-JEPA | 未明确列出目标距离/执行预算数值 | README 评测入口使用锁定的 LeWM 子模块，其配置为 25/50 |
| DeWM | 本地 PDF 未明确列出这两个数值 | 用户记忆为 25/50，待仓库配置或运行日志确认 |

这里斜线前是目标距离，后是环境执行预算，均不代表生成模型去噪步数或 CEM 迭代次数。公开默认值不能证明作者生成论文表格时未通过命令行覆盖。主表应统一重评，不直接混入未经协议核对的论文原始数值。

来源：

- [LeFlow 附录 A](https://arxiv.org/html/2608.24855v1#A1)
- [LeFlow TwoRoom 配置](https://raw.githubusercontent.com/hsiangwei0903/LeFlow/main/config/eval/tworoom.yaml)
- [Sub-JEPA 论文](https://arxiv.org/html/2605.09241v1)
- [Sub-JEPA 仓库](https://github.com/intcomp/Sub-JEPA)
- [Sub-JEPA 锁定的 LeWM TwoRoom 配置](https://raw.githubusercontent.com/lucas-maes/le-wm/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/eval/tworoom.yaml)
- DeWM：`docs/paper/AAAI2027_Flow_JEPA.pdf`。

## checkpoint 复用建议与证据

- CoWM：统一复用 Round4 AB、训练 seed 3072、epoch 10 的四任务 checkpoint，即 Round5 phase1_6 使用的完整系列；四个文件哈希与已有清单一致。推理 step 改为 2 不需要重新训练。
- LeFlow：复用 `config/round4/leflow_artifacts.json` 指定的四任务 planner，必须同时使用清单内配套的冻结 LeWM。12 个权重/配置文件均存在且哈希一致；不能随意替换其 latent backbone。
- LeWM+CEM：建议使用上述 LeFlow 配套的 LeWM checkpoint，以固定 LeWM 与 LeFlow 的 backbone；如需比较本地重训版本，单独标注。
- 完整哈希见同目录 `main_table_checkpoint_audit.json`。这是文件完整性审计，不替代实际加载和评测。
- 优先复用权重并按新协议重新评测。旧结果只有在评测样本、配置、seed、实现版本全部匹配时才可复用。

## 启动前仍需写入运行配置的细节

1. 建议评测 seeds 使用 Sub-JEPA README 的 `[42, 100, 2026, 3407, 1234, 4444]`。各方法共享相同初始状态和目标清单，避免不同随机数消耗改变评测样本。
2. CoWM 全量矩阵建议包含 P0/P1/P2/P3、P0-GF/PO、P2-GF/PO、P3-GF/PO、P3-PO-refine；明确后者是先选择再优化，区别于逐候选优化。全部动作生成 step=2。
3. 显式固定候选数、PO 次数/步长/约束、GF 作用步数/每步更新次数，以及 P2 的 CEM 预算。不能沿用历史配置时无意保留 step=1。
4. CEM 也有论文/代码差异：LeWM 论文写 PushT 30 次、其他任务 10 次；已核对公开 solver 默认 300 samples × 30 iterations、top-k 30。建议主表以公开代码默认作为完整 CEM baseline，另做预算曲线。
5. LeFlow 采用其独立默认推理配置；CoWM 的 step=2 不自动施加到 LeFlow。冻结其候选数和 flow 步数。
6. 成功率报告六个 seed 成功率的 mean ± std，明确 std 的 ddof；这不代表训练 seed 的不确定性。
7. 时间测量固定 GPU、精度、环境批量、计时边界和 CUDA 同步；区分单决策延迟与批量评测吞吐。至少计入编码、生成、排序、优化；环境运行时间单独列出。
8. 保存完整 resolved config、checkpoint hash、代码版本及逐 episode 结果。评测 seed 更换不等于 episode-level held-out 泛化；该实验需独立数据协议。

补充纠正：LeFlow 论文附录 B/Table 6 写五个评测 seeds，因此不能将四篇基线全部概括为原文统一采用六个 seeds。这里六个 seeds 是本研究统一重评协议。
