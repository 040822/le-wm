# Round 4 Reacher action-flow step 诊断报告

## 结论摘要

本报告基于 R4-AB epoch10、训练 seed 3072、`round3_revised` Reacher dev cohort 的 50 个 episode。它是单 checkpoint、单 seed 的诊断实验，不是 final 评测，也不是严格 held-out 泛化证明。

最强信号来自 P2 actor warm-start 的动作幅度，而不是单纯的 Euler 步数：标准 P2 step16 的 dev 成功率为 **50.0%**，将同一 warm-start 缩放为 `alpha=0.5` 后为 **80.0%**；step1 为 **92.0%**，P1 随机 CEM 为 **74.0%**。step16 相对 alpha=0.5 的 paired 结果为 improved=17、regressed=2。

Heun 能部分改变 P2 的闭环结果，但不是完整解释：P2-Heun-16 为 **68.0%**，相对 P2-Euler-16 的 paired 结果为 improved=14、regressed=5；P0 和 P3 使用 Heun 后分别为 **64.0%** 和 **78.0%**，均未优于对应 Euler-16 基线。

## 已有 step sweep

| 方法 | step1 | step2 | step5 | step10 | step16 | step32 |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 90.0% | 84.0% | 80.0% | 70.0% | 70.0% | — |
| P2 | 92.0% | 74.0% | 64.0% | 62.0% | 50.0% | 56.0% |
| P3 | 92.0% | 90.0% | 80.0% | 72.0% | 86.0% | — |
| P1 random CEM | — | — | — | — | 74.0% | — |

P2 step1 到 step16 的变化为 92% 到 50%，step32 回升到 56%，因此影响不是随 step 单调平滑变化的数值积分误差。

## 固定噪声 sampler 诊断

| sampler | median mean_abs | median 越界比例 | median path distance to Euler-16 |
|---|---:|---:|---:|
| Euler-1 | 0.142 | 0.000 | 6.587 |
| Euler-16 | 0.825 | 0.390 | 0.000 |
| Euler-32 | 0.852 | 0.420 | 0.207 |
| Heun-16 | 0.878 | 0.440 | 0.417 |

Euler step 增大使动作幅度和 normalized action 越界比例持续上升；Heun-16 的 path difference 也达到约 0.417，但其幅度并未回到 step1，而是更高。因此 Heun 不是“修复 step1 优势”的直接替代方案。

## CEM warm-start 诊断

| 初始化 | median 初始 cost | median 初始越界比例 | median 最终 elite cost |
|---|---:|---:|---:|
| P1 / no-warm | 0.8836 | 0.000 | 0.0088 |
| P2 Euler-1 | 0.0349 | 0.000 | 0.0084 |
| P2 Euler-16 | 0.0147 | 0.390 | 0.0086 |
| P2 Euler-16, alpha=0.5 | 0.3154 | 0.000 | 0.0088 |

alpha=0.5 使初始越界比例从 0.390 降到 0.000，并在闭环中恢复 30 个百分点。CEM 最终 elite cost 的差异很小，说明问题主要发生在初始动作 basin 与真实环境执行，而不是 CEM 最终 latent cost 无法收敛。

## Heun 闭环 gate

| 方法 | 成功率 | median planning seconds | action-flow integrator |
|---|---:|---:|---|
| P0 Euler-16 | 70.0% | — | Euler |
| P0 Heun-16 | 64.0% | — | Heun |
| P2 Euler-16 | 50.0% | 5.521 | Euler |
| P2 Heun-16 | 68.0% | 3.998 | Heun |
| P3 Euler-16 | 86.0% | 0.441 | Euler |
| P3 Heun-16 | 78.0% | 0.556 | Heun |

Heun 对 P2 有一定帮助，但 P0/P3 反而下降，不能据此把积分器替换为主要方法。P2 的 Heun 增益更像是改变了 actor warm-start 的动作轨迹，仍受同一幅度/执行问题影响。

## 失败边界与限制

P2 Euler-16 dev 失败 episode 中，有 1/25 个 terminal margin 落在 `[-0.01, 0)`，说明 Reacher 严格的逐关节 0.05 rad 判据会放大很小的动作差异；但这只是放大器，不足以解释动作越界比例和 alpha 对照的恢复效果。

本轮没有重新训练、没有运行 final、没有增加第二个训练 seed。结果应解释为：在固定 checkpoint 和固定 dev cohort 上，P2 的 step 敏感性主要与 actor warm-start 的动作尺度/越界及 Reacher 阈值交互有关，Euler/Heun 离散化是次要影响因素。后续优先检查 action normalizer、warm-start 输出裁剪策略和 actor flow 的训练/校准；不建议仅通过增加 Euler steps 选择 P2 的默认配置。

## 可复现实验产物

- 诊断配置：[reacher_step_diagnosis.json](/data/users/wenxin/pre-exp/le-wm/config/round4/reacher_step_diagnosis.json)
- 诊断入口：[round4_reacher_step_diagnosis.py](/data/users/wenxin/pre-exp/le-wm/scripts/round4_reacher_step_diagnosis.py)
- 诊断输出目录：`outputs/round4/reacher_step_diagnosis_seed3072_dev/`

## 版本与复现信息

- 诊断代码 commit：见输出目录 `summary.json` 的 `code.code_commit` 字段
- 诊断配置 SHA256：`11a146fab07c38aae9a27eac985ad31bcc786b6ed7489cdc914ef11168517759`
