# Fast-LeWAM 仓库审查报告

- 审查日期：2026-08-07
- 审查范围：`source/model/fast_lewam/`、`source/policy/fast_lewam.py`、`source/policy/fast_lewam_eval.py`、`source/common/epoch_eval.py`、`source/common/fast_lewam_validation.py`、`source/common/eval.py`（Fast 相关路径）、`train.py`/`eval_fast_lewam.py`、`config/train/fast_lewam.yaml`、`config/eval/*.yaml`、`scripts/`、`tests/`。lewm 与 value_jepa 基线不在审查范围。
- 审查方式：只读 + 在 `/tmp/opencode/` 运行独立验证脚本（未改动仓库任何文件）。
- 优先级排序：影响任务正确率的问题 > 影响训练/评测性能的问题 > 与论文 idea 违背的问题 > 其他。

---

## 一、高优先级：影响算法完成任务正确率/正确性的漏洞

### H1. Stage B 的 timestep 条件在训练与评测之间错位（与项目自身计划 §3.2 违背，直接削弱 CEM 排序能力）

- 训练（`source/policy/fast_lewam.py:143-149`）：预测动作分支的 Stage B AdaLN condition 使用 flow timestep `t`（随机采样自 [0,1)），真值动作分支使用 `t=1`。
- 评测（`source/model/fast_lewam/jepa.py:456-460` 的 `get_cost`、`:429-431` 的 `get_action`）：恒定使用 `t=1`。
- 后果：Stage B 模型在训练时把"clean estimate 动作 + 随机 t"作为输入，评测时却把所有 CEM 候选（原始高斯采样）当作"clean 动作（t=1）"来预测 latent。由于 t 被赋予了"动作噪声水平"语义，同一个动作在不同 t 下会产生不同的 latent 预测；评测分布（随机候选 @ t=1）在训练中从未出现，latent cost 的排序质量被直接削弱。
- `docs/plan/fast_lewam_stage_b_optimization.md §3.2` 已明确要求"Stage B AdaLN condition 始终使用 t=1，source_t 只用于 latent-loss 置信度权重"，当前代码并未落实该计划；`docs/plan/fast_lewam_change.md` 第 4 条也指向同一问题。
- 影响：Stage B planning 的核心信号（latent cost）在训练/评测间分布失配，是 E1 相对 E0（Cube 60 vs 78、Push-T 88 vs 98）差距的可疑来源之一。

### H2. CEM 候选动作分布与训练分布失配，且论文设想的缓解手段（warm-start CEM）未实现

- CEM 候选从 `N(0, var_scale=1)` 直接采样（`config/eval/solver/cem.yaml`），而 Stage B 训练输入只有两类：专家 clean 动作（t=1）与模型自身 clean estimate（t<1）。原始随机动作在 Stage B 训练中从未出现，`get_cost` 在 t=1 下对它们做 latent 预测属于外推，CEM 早期迭代的 top-k 精英选择建立在不可靠 cost 上。
- 论文 idea（LEDP 出发点 2、废案 2）明确把"训练-推理分布不匹配"列为要解决的问题，缓解手段是 Stage A warm-start CEM（talk 中 E6）。当前 `StageBModelView`（`source/policy/fast_lewam_eval.py:14-24`）屏蔽了 `get_action`，`prepare_init_action` 走 zero-padding 路径（`stable_worldmodel/solver/utils.py`），warm-start 完全未落地。
- 建议方向（本轮不改）：E6 落地；或在 `get_cost` 前对候选做 Stage A 去噪；或将 CEM 候选在训练时作为 Stage B 输入显式暴露。

### H3. 预测动作混合造成"错误监督"（反事实缺失）

- `stage_ab` 中预测动作分支的 latent 监督目标仍是专家轨迹的 future latent `z1:H`（`source/policy/fast_lewam.py:59,175`）。当 Stage A 的 clean estimate 偏离专家动作时，目标 latent 并不是该动作的真实后果，模型被迫把"自生成动作"映射到"专家未来"，产生不一致监督。
- 且默认 `detach_clean_action=false` 时，latent loss 的梯度经 clean estimate 直接回传 Action Head（`jepa.py:287-288`），把预测动作的后果偏差反向教给 action head（talk 中 E5 在 Cube 上 Stage B 66 vs E4 74 的负迁移与此一致）。
- 该问题 talk 已指出，属设计层面；但作为"影响正确率"的风险点应明确记录，且与 H4 相关。

### H4. 默认训练配置 = 已测变体中 Stage B 最差的 E5

- `config/train/fast_lewam.yaml:24` `detach_clean_action: false` + `latent_action_mix_epochs: 10`（上限 50% 预测动作），即 E5 配置。
- talk 数据：E5 Cube Stage B final 66（E4=74、E3=74），Push-T E5=86（E4=94）。默认配置在 Stage B 上系统性差于 detached 变体，同时论文主张的"跨 head 梯度双向增益"在当前证据下反而是负迁移。
- 影响：任何使用默认配置的新实验都会复现 E5 的次优结果；论文结论也需与默认配置脱钩。

### H5. validation 阶段 100% 使用预测动作，与训练课程（≤50%）不一致

- `_predicted_action_probability`（`source/policy/fast_lewam.py:19-21`）：`stage != "fit"` 恒返回 1.0。
- 后果：validate/loss 与 fit/loss 不可比；模型从未在 100% 预测动作 + 随机 t 下训练，validation 曲线系统性偏高且逐 batch 随机（t 随机），基于 val loss 的早停/选点不可靠。`fast_lewam_forward` 对 fit/validate 共用同一分支，没有区分处理。

### H6. 默认 epoch_eval 对 stage_ab 模型运行无意义的 stage_c 评测

- `config/train/fast_lewam.yaml:38` 默认 stages 含 `stage_c`。stage_c 的 action head 读 joint layout 的动作位置（`jepa.py:299`），该路径对 stage_ab 训练的模型从未被训练过，评测结果≈随机（talk 中 28%），且每次评测多消耗 10+ 分钟（见 P2）。
- `eval_fast_lewam.py:204` 默认 stages 同样包含 stage_c。

---

## 二、中优先级：影响训练/评测性能的漏洞

### P1. `get_cost` 每个 CEM 迭代重复编码 pixels 与 goal（实测占单次调用 ~99% 时间）

- `source/model/fast_lewam/jepa.py:449-452`：`get_cost` 每次调用都对当前帧与 goal 图像重新 `encode_pixels`，而 CEM 求解器（`stable_worldmodel/solver/cem.py:210`）在 30 个迭代中反复传入相同的 `info_dict`——同一批图像的编码被重复 30 次/环境。
- 实测（RTX 4090，真实 checkpoint `0803_e5_cross_head_grad/weights_epoch_10.pt`，batch=1、300 候选，验证脚本 `/tmp/opencode/fast_lewam_timing.py`）：
  - 单次 `get_cost`：128.5 ms
  - 其中编码 current+goal（2×300 张图）：126.9 ms（**99%**）
  - DiT Stage B 预测 + cost：仅 8.2 ms
- 估算：50 envs × 30 迭代，模型侧 193 s → 每 plan 只编码一次后约 19 s（**~10× 冗余**）。实测 stage_b 单次 epoch 评测 1285 s（`outputs/fast_lewam/cube/0803_e5_cross_head_grad/eval/epoch_10/stage_b/metrics.txt`），其中模型侧冗余约占 170 s。
- 影响：直接削弱论文"Fast（推理高效）"主张；epoch 内评测（P2）被进一步放大。修复方向：按 plan 缓存 z0/goal latent（z0、goal 在 30 个迭代内不变）。

### P2. epoch_eval 阻塞 DDP 训练

- `source/common/epoch_eval.py:95-136`：`on_train_epoch_end` 在 rank 0 上串行执行全部阶段评测（stage_b 单次 ~21 分钟，4 阶段合计 ~40 分钟/轮），其余 rank 在 `barrier` 上空等；`every_n_epochs: 2` 使 10 epoch 训练累计空等 ~100+ 分钟。
- 配合 P1 修复可显著缓解，但评测本身也应在不阻塞训练的前提下调度（例如移到训练循环外/独立进程）。

### P3. CEM 按 env 串行求解（batch_size=1）

- `config/eval/solver/cem.yaml` `batch_size: 1`：50 个 env 依次求解，每 env 30 次 `get_cost`。受显存约束（300 候选 × 224² 图像），可接受，但与 P1 合并考虑：每次 `get_cost` 的编码冗余是更大的开销来源。

### P4. 评测 goal 在 replan 时与训练分布不一致

- 评测 goal 固定在 `start + goal_offset_steps`（`stable_worldmodel/world/world.py:_evaluate_from_dataset`），第一次 replan 后（已执行 25 步）当前帧已偏离数据集 start+25 帧，goal 与当前状态的相对距离不再是"恰 1 个 chunk 之后"，与训练分布（goal=窗口末帧 z_H）不一致。
- 该协议与 LeWM 基线共享（非回归），但 Stage A 的"goal-conditioned BC"语义受此影响更大（二次 chunk 的目标可能是身后状态）。属协议级风险，应记录。

---

## 三、与论文 idea 违背/未落地的地方

### D1. Online Training 完全未实现

- LEDP 贡献 3 与 "Online Train" 设计（先预测 action → 执行 → 用真实后继 latent 在线更新）是论文三大贡献之一，但仓库中没有任何 online 训练循环、replay 缓冲或环境数据回流入口；训练设施（train.py、policy）只支持离线阶段。若论文按此主张投稿，需要补全或降级表述。

### D2. "并行预测提升推理效率"被评测路径的计算热点削弱

- 论文主张并行 latent 预测替代串行 rollout 以提升效率；但评测（P1）实际热点是重复图像编码而非 DiT 预测。Stage B 的 "Fast" 收益需在修复 P1 后才能成立。

### D3. Stage A 与论文定义的差异

- LEDP 定义 Stage A 为"根据 z0 预测 a1:aH"；实现为 goal-conditioned flow matching（z0+z_H 双锚点 + 噪声动作条件，`jepa.py:212-241`）。goal token 是合理扩展（shuffled-goal 消融证明其有效性），但与论文表述不一致，成文时需对齐描述。

### D4. 默认配置方向与论文核心主张相反

- 论文主张"Action Head 训练反向增强 World Model"；默认 E5 配置（H4）在 Cube/Push-T 上均呈现 Stage B 负迁移。当前实现中唯一能支撑该主张的是 E3（Stage B 恒用真值动作），而 E3 不是默认配置，且 E3 的 Stage A 增益也未超过 E0 基线。

---

## 四、其他问题

### O1. 合并残留文件

- `tests/test_fast_lewam_epoch_eval.py.orig` 为旧 API 版本（`modes`/`_load_eval_config`）的残留，应删除，避免混淆。

### O2. `FastLeWAM.get_action` warm-start 路径无调用者且语义存疑

- `jepa.py:406-433`：`get_action` 在给定 prefix 时先用 Stage B 预测 `z_{prefix_length}`，再从中采样一段**全新** chunk 并返回其前 `horizon` 步。该路径当前没有任何调用者（StageBModelView 屏蔽、chunk policy 直接调用 `sample_actions`），且与 `prepare_init_action` 的"tail 拼接"语义需要澄清（重新采样整段是否包含 prefix 步）。E6 落地前应明确或删除。

### O3. 重复/冗余代码路径

- `encode_pixels` 同时支持 uint8 设备侧归一化与 float 输入（`jepa.py:115-121`），训练（GPU 预处理）与评测（CPU transform）两条路径并存，行为一致性依赖约定而非断言，易漂移。

### O4. 统计强度与多 seed 设施

- 评测固定 50 episodes、单 eval seed（`config/eval/*.yaml` `seed: 42`），talk 已指出最小变化单位为 2%、结论需 3 训练 seed。训练侧多 seed 依赖手工修改 `seed`（train.sh 注释示例），无批量入口；`epoch_eval` 每个 run 使用独立 cohort 但跨 seed 无配对机制。

---

## 五、已确认无问题/正确的部分（供参考）

- **数据对齐**：窗口 6 帧（`num_steps = num_preds + history_size`）× frameskip=5 共 25 个原始环境步；`clean_actions[:, i]` 对应帧 i→i+1 的 Action Block（HDF5Dataset `_load_slice` 对 action 列保持稠密并按 block 重塑），训练目标 `z1:H`、训练 goal `z_H` 与评测 goal（`start+25` 帧）完全一致。
- **因果 mask 语义**：`causal_attention_mask` 下三角；`q_i` 只依赖 `z0,a1..a_i,q1..q_{i-1}`，测试 `test_causal_mask_and_predictions_hide_future_actions` 验证未来动作不影响早期预测。
- **Stage A mask**：锚点不读噪声动作、动作可读锚点（`stage_a_attention_mask`），与文档一致。
- **形状校验**：`_validate_inputs`、`get_cost`、`get_action`、`_validate_action_dim` 均有严格 shape 检查；solver 展开后的 `[B,S,1,C,H,W]` 经 `_last_frame` 取 T 维正确。
- **归一化空间一致性**：训练 z-score（`get_column_normalizer`）与评测 `StandardScaler`（`fit_eval_processors`）拟合自同一数据集且均剔除 NaN；CEM 在归一化空间采样、仅执行前逆变换，正确。
- **shuffled-goal 消融**：policy 内偏移 `1 + seed % (num_envs-1)` 与结果记录 `goal_source_slot` 完全一致。
- **epoch eval 状态恢复**：RNG、train/eval 模式、requires_grad 均正确恢复（`epoch_eval.py:98-135`），测试覆盖。
- **现有测试**：fast_lewam 相关 14 项测试全部通过（`python -m unittest tests.test_fast_lewam_model tests.test_fast_lewam_inference tests.test_fast_lewam_policy`）。

---

## 六、修复优先级建议（供后续排期，本轮未改任何文件）

1. **P1**（缓存 z0/goal latent，~10× 评测加速）——成本最低、收益最大，先做。
2. **H1**（Stage B 恒用 t=1，source_t 只用于权重）——按项目计划 §3.2 落实，成本低，直接影响 Stage B 排序。
3. **H4**（默认配置改为 detached 变体或明确标注 E5 为默认）——一行配置，避免新实验复现次优结果。
4. **H6/O4**（默认 stages 移除 stage_c；eval_fast_lewam 同）——减少无意义计算。
5. **H2/H3**（warm-start CEM、预测动作监督的反事实问题）——设计级，需实验计划配合（E6 及 simulator-grounded 排序诊断）。
6. **H5**（validate 与 fit 课程一致性）——影响监控可信度。
7. **D1**（Online Training）——论文贡献落地前需补齐或调整论文表述。

审查过程仅创建本报告；验证脚本位于 `/tmp/opencode/fast_lewam_timing.py`，仓库无其他改动。
