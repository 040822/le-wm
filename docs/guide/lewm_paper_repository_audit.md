# LeWorldModel 论文、官方代码、发布权重与本地评测环境审计

审计日期：2026-07-21  
论文：[LeWorldModel.pdf](./LeWorldModel.pdf)  
官方代码仓库：<https://github.com/lucas-maes/le-wm>  
审计所用官方提交：[`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`](https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac)

## 1. 目的与结论

本报告比较以下四类对象：

1. 论文正文及附录中的实验声明；
2. LeWorldModel 官方 GitHub 仓库的评测配置；
3. 作者账号 `quentinll` 发布的四个 Hugging Face 模型；
4. 当前本地仓库的评测配置和实际数据文件。

核心结论如下：

- 四个本地 `weights.pt` 与作者 Hugging Face 发布文件逐字节一致，文件完整，未发现 NaN/Inf，并能通过本仓库的兼容加载路径完整加载。
- 权重结构大部分符合论文，包括 ViT-Tiny、patch size 14、224×224 输入、192 维表示、从头训练以及主要 predictor 配置。
- 作者发布的 TwoRoom 权重配置使用 `num_frames=3`，而论文附录 D 声明 TwoRoom history length 为 1。
- 官方上游代码本身也有若干参数与论文文字不一致：非 PushT 任务的 CEM 迭代数，以及 TwoRoom 的评测预算和目标间隔。
- 当前本地评测 YAML 并非官方上游原样文件；数据集标识和输出设置均有修改。
- 当前本地 PushT 数据的 episode 数和平均长度明显不符合论文描述，因而不能直接用于严格论文复现。
- 按“评测参数与上游不一致则不评测”的门禁要求，本次审计没有启动 eval，也没有产生或覆盖评测结果。

因此，在实验前必须明确区分“复现官方代码结果”和“复现论文文字协议”。二者目前不是同一套实验条件。

## 2. 证据范围与检查方法

### 2.1 论文

检查了论文的模型架构、Implementation Details、Environment & Dataset、Evaluation Details、Figure 6 和 Table 5。

### 2.2 官方代码

从官方 GitHub 仓库读取并固定到提交 `8edfeb3`，重点比较：

- `config/eval/{pusht,cube,reacher,tworoom}.yaml`
- `config/eval/solver/cem.yaml`
- 官方 `eval.py` 的数据抽样和评测调用逻辑

### 2.3 发布权重

本地目录名是 `data/checkpoints/quetinll`，但作者的 Hugging Face 用户名是 `quentinll`。检查对象为：

- `lewm-cube`
- `lewm-pusht`
- `lewm-reacher`
- `lewm-tworooms`

对每个文件执行了 SHA-256、张量数量、参数数量、非有限值和兼容加载检查，并使用 Hugging Face LFS OID 验证远端身份。

### 2.4 本地环境

检查了当前 `config/eval`、统一后的 `eval.py`、实际数据路径及 HDF5 episode 统计。本地 `data/datasets` 是符号链接：

```text
data/datasets -> /data/users/majiahua/le-wm/data/datasets
```

## 3. 权重身份与完整性

| 模型 | 文件大小 | SHA-256 | 张量数 | 模型参数量 | 非有限值 | 远端一致 |
|---|---:|---|---:|---:|---:|---|
| Cube | 72,291,425 B | `2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89` | 303 | 18,034,628 | 0 | 是 |
| PushT | 72,290,721 B | `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607` | 303 | 18,034,478 | 0 | 是 |
| Reacher | 72,290,849 B | `eb70b1fd5409f8f81875d62f5ee5a20dd220a3128a477de66b5760f475f0f469` | 303 | 18,034,478 | 0 | 是 |
| TwoRoom | 72,290,849 B | `566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd` | 303 | 18,034,478 | 0 | 是 |

作者发布页：

- <https://huggingface.co/quentinll/lewm-cube>
- <https://huggingface.co/quentinll/lewm-pusht>
- <https://huggingface.co/quentinll/lewm-reacher>
- <https://huggingface.co/quentinll/lewm-tworooms>

四个权重都能经过 `source/common/remap.py` 完整加载，且没有 missing 或 unexpected keys。该 remap 只处理 Hugging Face Transformers 新旧版本间的 ViT 参数键名变化，例如旧版 `encoder.encoder.layer.*` 到新版 `encoder.layers.*`。直接用当前新版依赖中的原始类加载旧权重会报告键名不匹配，这不是权重损坏。

Cube 的本地 `config.json` 与远端配置语义相同，但本地文件少一个结尾换行，因此文本 blob 哈希不同；模型字段没有差异。其余三个本地配置的 Git blob 哈希与远端一致。

## 4. 模型结构与论文比较

### 4.1 一致项

| 项目 | 论文 | 发布权重配置 | 结论 |
|---|---|---|---|
| Encoder | ViT-Tiny | `size=tiny` | 一致 |
| Patch size | 14 | 14 | 一致 |
| 输入分辨率 | 224×224 | 224 | 一致 |
| Encoder hidden size | 192 | 192 | 一致 |
| Encoder layers | 12 | ViT-Tiny 对应 12 | 一致 |
| Encoder heads | 3 | ViT-Tiny 对应 3 | 一致 |
| 预训练 | 不使用预训练 | `pretrained=false` | 一致 |
| Predictor depth | 6 | 6 | 一致 |
| Predictor heads | 16 | 16 | 一致 |
| Predictor dropout | 0.1 | 0.1 | 一致 |
| PushT history | 3 | 3 | 一致 |
| Cube history | 3 | 3 | 一致 |

Cube 的 action encoder 输入维度为 25，即 5 个环境动作维度乘以 action block 5；其余三个任务为 10，即 2×5。这也解释了将 Cube 权重误用于 5 维动作环境或错误设置 action block 时出现的 `action_dim` 检查错误。

### 4.2 明确冲突

| 项目 | 论文附录 D | 作者 TwoRoom 权重 |
|---|---:|---:|
| History length | 1 | 3 |

该冲突存在于作者发布权重自身，并非本地兼容代码引入。

### 4.3 无法由权重证明的训练信息

发布目录只有模型 state dict 和结构配置，没有 optimizer state、训练日志、训练数据哈希或完整训练配置。因此无法仅凭这些文件确认：

- 是否确实训练 10 epochs；
- SIGReg 是否使用 `M=1024`、`lambda=0.1`；
- batch size 是否为 128；
- 是否严格使用论文所述数据版本；
- checkpoint 是最佳 epoch、最后 epoch，还是另一次筛选结果。

这些项目需要作者的完整 run config、日志或在相同数据上的重新训练实验才能确认。

## 5. 论文与官方上游评测配置比较

### 5.1 一致项

论文和官方上游都使用：

- CEM 300 个 candidate sequences；
- top 30 elites；
- initial variance 1；
- planning horizon 5；
- receding horizon 5；
- action block / frame skip 5；
- PushT budget 50、goal offset 25；
- Cube/Reacher budget 50、goal offset 25。

### 5.2 冲突项

| 任务/参数 | 论文 | 官方上游代码 | 当前本地配置 |
|---|---:|---:|---:|
| PushT CEM iterations | 30 | 30 | 30 |
| Cube CEM iterations | 10 | 30 | 30 |
| Reacher CEM iterations | 10 | 30 | 30 |
| TwoRoom CEM iterations | 10 | 30 | 30 |
| TwoRoom eval budget | 150 | 50 | 50 |
| TwoRoom goal offset | 100 | 25 | 25 |

官方上游 CEM 配置：<https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/eval/solver/cem.yaml>  
官方上游 TwoRoom 配置：<https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/eval/tworoom.yaml>

这意味着即使直接运行未修改的官方仓库，也不能同时声称严格遵循论文附录中的所有参数。

## 6. 当前本地仓库与官方上游比较

当前本地 YAML 保留了上游的主要数值参数，但配置文件不是上游原样：

| 任务 | 官方上游 `dataset_name` | 当前本地 `dataset_name` | 当前新增 benchmark 名称 |
|---|---|---|---|
| PushT | `pusht_expert_train` | `pusht` | `pusht_expert_train` |
| Cube | `ogbench/cube_single_expert` | `ogbench/cube_single` | `ogbench/cube_single_expert` |
| Reacher | `dmc/reacher_random` | `dmcontrol/reacher` | `dmc/reacher_random` |
| TwoRoom | `tworoom` | `tworoom` | `tworoom` |

本地配置还增加了：

- `eval.task_name`
- `eval.benchmark_dataset_name`
- `output.save_video=true`

同时移除了上游按任务设置的 `output.filename`，改由统一评测入口管理 txt、JSON 和视频输出。

这些改动未必都会改变成功率。例如输出路径和视频开关通常只影响记录行为；但是实际供 dataset loader 使用的 `dataset_name` 已发生变化，必须证明别名最终解析到与上游完全相同的数据，才能认为实验等价。

## 7. 数据集比较

### 7.1 论文声明

| 任务 | 论文数据规模 |
|---|---|
| TwoRoom | 10,000 episodes，平均 92 steps |
| PushT | 20,000 expert episodes，平均 196 steps |
| Cube | 10,000 episodes，每条 200 steps |
| Reacher | 10,000 episodes，每条 200 steps |

### 7.2 当前本地数据统计

| 任务 | 本地文件 | 本地统计 | 与论文关系 |
|---|---|---|---|
| TwoRoom | `tworoom.h5` | 10,000 episodes，平均约 92.08 | 基本一致 |
| PushT | `pusht.h5` | 18,685 episodes，平均约 125.06，范围 49–246 | 明显不一致 |
| Cube | `ogbench/cube_single.h5` | 10,000 episodes，读取长度约 201 | 可能是 200 transitions 加初始状态，需核实格式 |
| Reacher | `dmcontrol/reacher.h5` | 10,000 episodes，读取长度约 201 | 可能是 200 transitions 加初始状态，需核实格式 |

PushT 的差异会直接影响：

- 可抽样 episode 集合；
- 合法初始下标数量；
- 初始状态和目标状态分布；
- 与论文相同 seed 下实际抽到的轨迹；
- 最终成功率。

因此，在确认或替换 PushT 数据前，不应把本地评测称为论文复现。

## 8. 论文结果参考值

论文 Figure 6 中 LeWM 的成功率参考值为：

| 任务 | LeWM 成功率 |
|---|---:|
| TwoRoom | 87% |
| Reacher | 86% |
| PushT | 96% |
| OGBench-Cube | 74% |

论文 Table 5 还报告 PushT 在三个训练 seed 上为 `96.0 ± 2.83`，每次评测使用同一组 50 条轨迹。

这些数值只能在数据、轨迹抽样、环境版本、成功判定、模型历史长度、CEM 参数及预算一致时进行严格比较。50 episodes 下，每一个 episode 会改变 2 个百分点，因此单次评测的小幅差异也应结合逐 episode 结果检查。

## 9. 本次未运行评测的原因

本次遵守了以下门禁：

> 先使用当前仓库的评测参数；评测前确保参数未修改、与上游保持一致；如果不一致则不进行评测。

门禁未通过的具体原因：

1. 当前四个任务 YAML 与官方上游存在字段差异；
2. 当前 `eval.py` 和公共 eval 实现也处于本地修改状态；
3. 本地数据别名与官方上游标识不同，且尚无数据哈希等价证明；
4. PushT 实际数据统计与论文显著不同；
5. 官方上游参数与论文协议本身存在冲突。

因此本次没有启动 GPU eval，也没有生成或覆盖 video、JSON、txt 结果。

## 10. 建议的验证实验矩阵

为了判断“官方代码是否合理”以及“论文结论能否复现”，建议不要只跑一套参数，而是明确运行以下协议。

### 实验 A：官方代码复现

目的：回答“作者发布权重在官方代码原始配置下能否得到官方结果”。

- 使用固定提交 `8edfeb3`；
- 使用上游原始任务 YAML 和 CEM YAML；
- 使用作者权重；
- 使用官方数据名对应的原始数据；
- 保留 seed 42、官方抽样逻辑和 50 episodes；
- 记录环境及依赖版本。

该实验复现的是代码，而不是论文附录的所有文字参数。

### 实验 B：论文协议复现

目的：回答“严格按论文附录参数能否得到 Figure 6 结果”。

- PushT：CEM 30 iterations，budget 50，offset 25；
- Cube/Reacher：CEM 10 iterations，budget 50，offset 25；
- TwoRoom：CEM 10 iterations，budget 150，offset 100；
- TwoRoom history length 按论文设为 1。

TwoRoom 的作者权重按 `num_frames=3` 训练/发布，不能直接证明与 history 1 兼容。因此应额外区分：

- B1：权重保持 history 3，只修改规划和控制参数；
- B2：使用 history 1 重新训练的模型，严格复现论文声明。

### 实验 C：单因素消融

目的：解释论文与代码差异对成功率的实际影响。

建议固定轨迹集合，分别改变：

1. CEM 10 vs 30 iterations；
2. TwoRoom 50/25 vs 150/100；
3. TwoRoom history 1 vs 3；
4. 官方数据 vs 当前本地数据；
5. 上游 eval 入口 vs 当前统一 eval 入口。

每个实验应保存同一批 episode ID、逐 episode 成功标记、总成功率、规划耗时和总耗时。只有固定 episode 集合后，才能把差异归因于单个参数。

## 11. 建议的结果记录字段

每次实验至少记录：

```text
protocol_name
git_commit
checkpoint_sha256
checkpoint_config
dataset_path
dataset_sha256_or_manifest
dataset_episode_count
dataset_length_statistics
environment/package_versions
seed
sampled_episode_ids
sampled_start_indices
goal_offset_steps
eval_budget
history_length
action_block
planning_horizon
receding_horizon
cem_num_samples
cem_n_steps
cem_topk
per_episode_success
success_rate
per_episode_elapsed_seconds
total_elapsed_seconds
```

视频、JSON 和 txt 应放在同一实验子目录。JSON 作为机器可读真值，txt 用于人工快速审查，视频用于诊断失败 episode。

## 12. 审查时需要优先裁定的问题

1. 最终基准应以论文文字还是官方代码为准？建议两套协议都保留并明确命名。
2. 作者用于 Figure 6 的实际 TwoRoom history length 是 1 还是 3？
3. 作者用于非 PushT 任务的实际 CEM iterations 是 10 还是 30？
4. Figure 6 的 TwoRoom 是否使用 150/100，还是上游代码中的 50/25？
5. 官方 PushT 数据的确切文件版本和哈希是什么？
6. 作者发布 checkpoint 对应哪个训练 seed 和 epoch？

在这些问题得到实验或作者元数据支持之前，应将“官方代码复现结果”和“论文协议复现结果”分开报告，不能直接混合比较。
