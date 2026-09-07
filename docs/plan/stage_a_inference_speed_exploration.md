# Stage A 推理加速探索与延迟预处理落地

## 1. 边界与目的

本实验只加载第一轮已经训练好的 E5 checkpoint，在运行时改变 Stage A 的推理路径，回答三个问题：能否加速、能加速多少、是否降低任务性能。探索阶段未修改正式实现；延迟预处理在配对验证通过并经用户确认后，已应用到正式 Stage A 与 Stage C chunk policy。

## 2. Checkpoint 与硬件

- Cube：`outputs/fast_lewam/cube/0803_e5_cross_head_grad/checkpoints/fast_lewam_weights_epoch_10.pt`
- Push-T：`outputs/fast_lewam/pusht/0803_e5_cross_head_grad/checkpoints/fast_lewam_weights_epoch_10.pt`
- Reacher：`outputs/fast_lewam/reacher/0717_/checkpoints/fast_lewam_weights_epoch_10.pt`
- TwoRoom：`outputs/fast_lewam/tworoom/0717_/checkpoints/fast_lewam_weights_epoch_10.pt`
- 受控 sampler microbenchmark：RTX 4090，CUDA，同一输入 latent 和同一初始噪声，warm-up 后报告 p50。
- 端到端筛选：每任务固定 10-episode cohort，关闭视频，只使用 GPU0–3。最初的 50-episode 尝试因共享机器预处理耗时过长而停止，因此此处只用于候选筛选；性能结论优先看逐 episode success，速度趋势同时参考受控 microbenchmark。

## 3. 候选优化

1. 将 Euler/flow 步数从 10 降到 5、2 或 1。
2. 仅在 Stage A sampler 内使用 BF16 autocast，视觉 latent 和动作累计保持 FP32。
3. 延迟图像预处理：策略已有动作 chunk 可直接消费时，不执行 `_prepare_info()`；只有需要 replan 的环境子批次才预处理 current/goal 图像。该优化先以进程内 monkeypatch 完成探索，现已合入正式 `FastLeWAMChunkPolicy`。

## 4. 受控 sampler microbenchmark

表中加速比均相对同 batch 的 FP32 10-step；RMSE 使用相同初始噪声，相对 FP32 10-step 输出计算。

| Batch | 精度 | Euler steps | p50 latency | 加速比 | action RMSE |
|---:|---|---:|---:|---:|---:|
| 1 | FP32 | 10 | 28.589 ms | 1.000× | 0 |
| 1 | FP32 | 5 | 14.076 ms | 2.031× | 0.0472 |
| 1 | FP32 | 2 | 5.652 ms | 5.058× | 0.1721 |
| 1 | FP32 | 1 | 2.925 ms | 9.773× | 0.2992 |
| 1 | BF16 | 10 | 37.153 ms | 0.769× | 0.0029 |
| 10 | FP32 | 10 | 28.523 ms | 1.000× | 0 |
| 10 | FP32 | 5 | 14.282 ms | 1.997× | 0.0994 |
| 10 | FP32 | 2 | 5.747 ms | 4.963× | 0.3103 |
| 10 | FP32 | 1 | 2.934 ms | 9.723× | 0.5099 |
| 10 | BF16 | 10 | 36.740 ms | 0.776× | 0.0056 |
| 50 | FP32 | 10 | 28.121 ms | 1.000× | 0 |
| 50 | FP32 | 5 | 14.125 ms | 1.991× | 0.1633 |
| 50 | FP32 | 2 | 5.691 ms | 4.941× | 0.4480 |
| 50 | FP32 | 1 | 2.946 ms | 9.545× | 0.6021 |
| 50 | BF16 | 10 | 36.442 ms | 0.772× | 0.0182 |

结果表明，减少 Euler 步数能近似线性降低 sampler latency，但输出偏差随步数减少和 batch 增大而明显增加，不能只凭 microbenchmark 落地。BF16 在该 RTX 4090 和小型 DiT 上反而慢约 23%–30%，同时引入非零输出误差，因此不进入端到端候选。

## 5. 固定 cohort 端到端结果

比较包括原始 FP32 10-step，以及仅运行时启用延迟预处理后的 10/5/2/1-step。`deferred-10` 分离延迟预处理本身，其他项同时改变 Euler 步数。速度比相对各任务原始 10-step；绝对时间受共享 GPU/CPU 负载影响很大。

| 任务 | 原始 10-step | deferred-10 | deferred-5 | deferred-2 | deferred-1 |
|---|---:|---:|---:|---:|---:|
| Cube success | 100% | 100% | 100% | 100% | 100% |
| Cube time / speedup | 306.60 s / 1.0× | 12.47 s / 24.58× | 17.16 s / 17.87× | 8.81 s / 34.80× | 10.43 s / 29.39× |
| Push-T success | 100% | 100% | 100% | 100% | 100% |
| Push-T time / speedup | 197.60 s / 1.0× | 10.02 s / 19.73× | 16.56 s / 11.93× | 14.82 s / 13.34× | 14.19 s / 13.92× |
| Reacher success | 90% | 90% | 80% | 70% | 70% |
| Reacher time / speedup | 433.62 s / 1.0× | 13.21 s / 32.83× | 12.11 s / 35.80× | 11.70 s / 37.05× | 12.90 s / 33.60× |
| TwoRoom success | 100% | 100% | 100% | 100% | 100% |
| TwoRoom time / speedup | 597.14 s / 1.0× | 38.91 s / 15.35× | 12.92 s / 46.22× | 8.61 s / 69.38× | 7.87 s / 75.92× |

延迟预处理的 10-step 与原路径在四任务共 40 个 episode 上逐项完全一致（0 个 regression、0 个 improvement），说明该候选没有改变动作数值路径，只移除了 action buffer 非空时无用的图像变换。端到端观察到 15.35×–32.83×，但这个倍率受到严重共享负载影响；它证明预处理是当前主要系统瓶颈，不应被当作稳定部署倍率。

减少 Euler 步数并非无损。Cube、Push-T、TwoRoom 的小样本均已饱和，无法排除性能下降；Reacher 则从 10-step 的 90% 降至 5-step 的 80%，2/1-step 均为 70%。配对上分别新增 1、2、3 个 baseline-success regression（1-step 同时有 1 个 improvement）。这与 sampler RMSE 随降步增大的受控结果一致。

正式合入后使用 GPU0–3 重跑相同四任务 10-episode cohort：production 10-step 的 success vector 与探索阶段 deferred-10 在四任务上逐 episode 完全一致；成功率仍为 Cube 100%、Push-T 100%、Reacher 90%、TwoRoom 100%。对应本次 `evaluation_seconds` 为 5.86、1.27、2.86、1.04 秒；绝对值受运行负载影响，不与历史倍率直接比较。

## 6. 结论与落地建议

- BF16：当前硬件和模型形状下没有速度收益，不建议继续。
- 减少 Euler 步数：sampler 有明确的 2×–9.8× 潜在加速，但 Reacher 已出现 10–20 点回退，**不建议直接应用**。如果继续，只值得对 5-step 做正式 50-episode 非劣验证。
- 延迟预处理：40 个配对 episode 完全一致，且四任务均大幅减少端到端耗时；现已应用到正式 Stage A/C 策略，仍建议后续补充正式 50-episode 非劣验证。
- 端到端倍率不用于容量规划；正式验证应在空闲 GPU0–3、固定 CPU 负载下重复，并报告 preprocessing/replan/sampler 分段计时。
- 正式实现只应用延迟预处理；Euler 步数与精度保持不变。
