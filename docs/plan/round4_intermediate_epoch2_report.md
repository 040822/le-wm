# Round 4 epoch2 intermediate diagnostic

日期：2026-09-12

这是一份训练尚未完成时的开发集诊断，不是 Round 4 的正式 final 结果，也不用于判断种子扩展门槛。四个任务均使用 seed `3072` 的共同 `epoch 2` 权重和 `round3_revised` dev cohort（每任务 50 episodes）。

## 结果

| task | P0 | P3 | P4 |
|---|---:|---:|---:|
| Cube | 49/50 (98%) | 49/50 (98%) | 47/50 (94%) |
| Push-T | 40/50 (80%) | 34/50 (68%) | 39/50 (78%) |
| Reacher | 14/50 (28%) | 22/50 (44%) | 16/50 (32%) |
| TwoRoom | 23/50 (46%) | 21/50 (42%) | 28/50 (56%) |

P3/P4 均使用 64 candidates、16 flow steps、`solver_batch_size=1`；为避免与训练共享显存压力，候选验证使用 `candidate_batch_size=8`。候选总数和选择规则没有改变。

## 可追溯结果

- Cube：`outputs/round4/intermediate_epoch2/cube/{P0,P3,P4}/dev/result.json`
- Push-T：`outputs/round4/intermediate_epoch2/pusht/{P0,P3,P4}/dev/result.json`
- Reacher：`outputs/round4/intermediate_epoch2/reacher/{P0,P3,P4}/dev/result.json`
- TwoRoom：`outputs/round4/intermediate_epoch2/tworoom/{P0,P3,P4}/dev/result.json`

所有上述 `result.json` 的状态为 `ok`，并包含 cohort-bound trace。对应 checkpoint 为各任务 run 目录下的 `checkpoints/r4_abde_seed3072_weights_epoch_2.pt`。

## 解释边界

- 这些结果只说明 epoch2 闭环和 P4 实现已经可运行；不能替代 epoch10 的正式 dev/final 矩阵。
- P4 相对 P3 的早期差异在四任务上并不一致，不能据此宣称 latent candidate 优于 action candidate。
- 训练仍继续进行；正式评测仍需使用最终固定的 epoch10 checkpoint，并补齐 P0、P0-shuf、P1、P2、P3、P4、P4-first 及 final cohort。
