# Round3 Online Supervision Runbook

本 runbook 是 `round3_next_plan.md` 的可执行入口。任务严格限定为
`reacher` 和 `pusht`；所有 GPU 命令必须显式设置 `CUDA_VISIBLE_DEVICES`，且只
能包含 `0,1,2,3`。

## 0. 前置检查

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/round3_online_supervision.py \
  dry-run --task reacher --device cuda:0
```

如果使用 CPU 做接口检查，可将 `--device` 改为 `cpu`。该命令不启动训练或
环境评测。

## 1. E1/E2 guidance 诊断

先为每个任务生成冻结 manifest：

```bash
python scripts/round3_guidance.py manifest \
  --task reacher \
  --output outputs/round3/online_supervision/reacher/guidance_manifest.json
```

有限差分检查：

```bash
python scripts/round3_guidance.py gradient-check \
  --task reacher --device cpu \
  --output outputs/round3/online_supervision/reacher/gradient_check.json
```

E1 的 64 个 paired 状态会分别收集 RMS=.01/.03/.10 的 actor、负梯度、正梯度
和随机方向，并将这些状态标记为排除数据：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/round3_guidance.py e1 \
  --task reacher --device cuda:0 \
  --pool outputs/round3/phase2/reacher_online_pool.json \
  --output-dir outputs/round3/online_supervision/reacher/e1
```

E2 使用 50-episode `round3_revised` dev cohort，比较 A、A+B、Post-opt、
Guided-flow、B+CEM、A+CEM；它只生成 report，不自动决定是否通过：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/round3_guidance.py e2 \
  --task reacher --device cuda:0 \
  --output-dir outputs/round3/online_supervision/reacher/e2
```

根据 E1/E2 结果由用户选择 `none`、`post_opt` 或 `guided_flow`。如果不采用
guidance，后续命令使用 `--guidance-mode none`；没有自动 gate。

## 2. 构造固定 replay

下面命令为每个任务构造同一份固定 replay：16,000 continuous steps 加
4,000 grounded steps。grounded 部分为 40 组，每组 4 个候选、每个候选执行
5 个 action blocks，因此每组名义上消耗 100 个环境步。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/round3_online_supervision.py \
  prepare-fixed-replay \
  --task reacher --device cuda:0 \
  --guidance-mode none \
  --output-root outputs/round3/online_supervision/reacher/fixed_replay_v1
```

Push-T 运行同一命令，将 `reacher` 替换为 `pusht`。准备完成后检查：

- `fixed_replay.pt` 的 `data_kinds` 同时包含 `offline`、`continuous`、
  `grounded`；
- continuous/grounded 都有 success、起点物理 cost 和终点物理 cost；
- `manifest.json` 中两个环境预算分别为 16,000 和 4,000；
- replay hash、checkpoint hash、collector version 和 group manifest 均已记录。

## 3. T0–T6 固定 replay 训练

首轮预算是 200 次 optimizer update，曲线点为
`0,10,...,200`；这里的 step 不是 env step。每个非 T0 臂使用同一 replay、同一
初始 checkpoint 和预生成 batch manifest。默认每 10 个 update 评测一次 final
200-episode cohort，实际评测为 4×50。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/round3_online_supervision.py \
  train-fixed-replay \
  --task reacher --device cuda:0 \
  --replay outputs/round3/online_supervision/reacher/fixed_replay_v1/fixed_replay.pt \
  --output-root outputs/round3/online_supervision/reacher/fixed_train_v1
```

若只做训练 artifact 检查而暂不运行环境评测，增加 `--no-eval`；此时曲线状态
为 `evaluation_pending`，不能当作 success-rate 结果。

## 4. closed-loop online train

阶段三复用阶段二冻结的 loss weights；`--fixed-training-root` 应指向阶段二训练
输出。每五个 update 中有四个 100-step continuous event 和一个 100-step
grounded event，即环境预算 80/20；optimizer update 仍为每 100 环境步一次。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/round3_online_supervision.py \
  train-closed-loop \
  --task reacher --device cuda:0 \
  --pool outputs/round3/phase2/reacher_online_pool.json \
  --offline-replay outputs/round3/online_supervision/reacher/fixed_replay_v1/replay/offline.pt \
  --fixed-training-root outputs/round3/online_supervision/reacher/fixed_train_v1 \
  --output-root outputs/round3/online_supervision/reacher/closed_loop_v1
```

T0 是冻结曲线，不更新参数；T1 是 offline-only；T2–T6 使用当前模型采集自己
的 online replay。R4-AB 不在这些命令中自动启动。

## 5. 恢复与结果解释

训练和采集入口都使用 append-only/atomic artifact。恢复时必须保持 task、
checkpoint、replay、seed、update schedule 和 GPU device 一致：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/round3_online_supervision.py \
  train-fixed-replay ... --resume
```

最终查看每个臂的：

- `curve/curve.json` 和 `curve/curve.csv`：主横轴是 `optimizer_update_step`；
- `run_state.json`：每次有效 update 的 batch 来源、loss、权重、梯度范数和
  effective sample counts；
- `batch_manifest.json`：固定 replay 的采样 manifest；
- `final_eval/update_000200/result.json`：同一 final 200-episode cohort 的最终
  评测。

200 update 只是首轮筛选预算，不代表收敛。是否通过门槛、保留 T3/T5/T6 或延长
到 500/1000 update，由用户根据两个任务各自的结果决定。
