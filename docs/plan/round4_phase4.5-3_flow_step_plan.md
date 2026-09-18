# Round 4 Phase 4.5-3：CEM-clip、candidate-scale、flow step 与 Heun 验证

## Summary

固定 R4-AB、seed 3072、epoch 10 权重，在四任务 `round3_revised` dev cohort（每任务 50 episodes）上验证：

- `cem-clip` 下 flow step 是否仍影响成功率；
- `cem-clip` 下 Euler/Heun 是否仍有影响；
- candidate-clip 是否对其他任务有帮助；
- candidate-scale 是否比 candidate-clip 更适合不同 CEM 模式；
- action 越界与归一化边界的关系；
- 各配置 planning latency 与 LeWM 的差异。

不重新训练，不评测 final 集。

## 1. 冻结实验矩阵

固定设置：`seed=42`、goal offset 25、eval budget 50、horizon 5、receding horizon 5、action block 5、CEM `300 samples / top-30 / 30 iterations / var_scale=1.0`。

| 模式 | 协议 | step/integrator | 条件数 | 用途 |
|---|---|---:|---:|---|
| P0 | legacy direct-control | 6 × Euler/Heun | 48 | 非 CEM flow-step/积分器控制 |
| P1 | `cem-clip` | invariant | 4 | 随机 CEM 的 candidate-clip 跨任务对照 |
| P1 | `cem-scale` | invariant | 4 | 随机 CEM 的 candidate-scale 跨任务对照 |
| P2 | `cem-clip` | 6 × Euler/Heun | 48 | 主要 flow-step/Heun 实验 |
| P2 | `cem-scale` | 6 × Euler/Heun | 48 | scale 与 clip 的完整对照 |
| P3 | legacy best-of-N | 6 × Euler/Heun | 48 | 非 CEM best-of-N 控制 |

step 为 `1, 2, 5, 10, 16, 32`。P1 不使用 action flow，因此不展开 step 或 integrator。

P0/P3 保持原始逻辑，不启用 clip 或 scale。P1/P2 分别运行 candidate-clip 和 candidate-scale 两种 CEM 协议。

现有 Euler P0/P3 的 step 1/2/5/10/16 结果、P1 legacy 结果以及 Reacher 已完成的 clip/scale 结果，在 checkpoint、cohort、配置和 trace 一致时复用；缺失的 step32、Heun、其他任务 scale 结果补跑。

## 2. CEM-clip 与 candidate-scale 定义

协议 metadata 使用以下值：

- `legacy`：raw candidate 直接进入 B 评分；
- `cem-clip`：逐坐标 clip；
- `cem-scale`：逐 candidate path 使用一个全局非负缩放因子；
- `not_applicable`：P0/P3。

内部保留 `action_bound_mode`：

- `none`
- `candidate_clip`
- `candidate_scale`

### CEM-clip

P1/P2 中：

- P2 actor warm-start 初始 mean 先 clip；
- 每轮 raw CEM candidate 在 B 评分前 clip；
- elite、mean、variance 均由 clip 后 candidate 更新；
- 最终 CEM mean 保持合法；
- B 永远只接收 clip 后 action。

### Candidate-scale

P1/P2 中：

- P1 对每条 CEM candidate path 使用一个全局缩放因子；
- P2 对 actor warm-start 和每条 CEM candidate path 都使用同样的 global-scale 规则；
- 缩放以零为中心，尽量保持整条 action path 的相对形状；
- B 评分、elite 更新和最终输出均使用 scale 后 action；
- candidate-scale 与 candidate-clip 互斥，不叠加使用。

## 3. 四任务 action 边界

四个环境当前物理 action space 均为 `[-1, 1]`，但模型输出是 z-score normalized action。真实归一化边界按：

\[
a_{\mathrm{norm}}=(a_{\mathrm{physical}}-\mu)/\sigma
\]

逐维计算。

| 任务 | 物理 action 维度 | 模型 action 维度 | 归一化边界 |
|---|---:|---:|---|
| Cube | 5 | 25 | `[-3.492797, 3.417580]`, `[-2.531917, 2.547874]`, `[-1.558995, 1.550765]`, `[-2.546894, 2.544735]`, `[-4.631316, 3.358860]` |
| Push-T | 2 | 10 | `[-4.759436, 4.834388]`, `[-4.869975, 4.803608]` |
| Reacher | 2 | 10 | `[-1.731921, 1.731798]`, `[-1.731678, 1.733430]` |
| TwoRoom | 2 | 10 | `[-1.156241, 1.149109]`, `[-1.090232, 1.212222]` |

模型 action 维度按 action block 重复 5 次。运行时保存完整精度的 physical bounds、normalizer mean/scale 和 normalized bounds；表中数值仅作可读摘要。

## 4. 越界统计

每个 task/mode/step/integrator 分别统计：

- normalized true-bound violation fraction；
- normalized action-vector violation fraction；
- normalized trajectory violation fraction；
- inverse-transform 后 physical action violation fraction；
- 最大越界幅度；
- `abs(action_norm)>1` legacy proxy，单独记录，不能称为真实越界；
- raw CEM candidate violation；
- projected candidate violation；
- candidate changed fraction；
- P2 warm-start 投影前后越界率；
- candidate-scale 的缩放因子分布；
- 最终实际执行动作的归一化和物理越界率。

真实归一化越界使用逐坐标判断：

```text
a_norm < normalized_low - 1e-6
or
a_norm > normalized_high + 1e-6
```

CEM-clip 和 candidate-scale 的 projected candidate violation 预期均为 0；若不为 0，视为实现错误。P0/P3 虽不启用边界处理，也必须记录其 raw action path 和最终执行动作越界情况。

## 5. 统计比较

每个条件报告成功率、成功数、Wilson 95% 区间、paired improved/regressed/net 和双侧 exact McNemar p 值。

主要比较：

- P2-clip：各 integrator 下不同 step 对 step16；
- P2-scale：各 integrator 下不同 step 对 step16；
- Euler vs Heun：同任务、同模式、同 step；
- P1-clip vs P1-legacy；
- P1-scale vs P1-clip 及 P1-legacy；
- P2-clip vs P2-legacy；
- P2-scale vs P2-clip 及 P2-legacy；
- candidate-clip/candidate-scale 在四任务上的迁移差异。

本阶段不设置自动通过阈值、不自动选择最佳 step、不根据 dev 结果扩充训练种子。结论分为“支持 / 不支持 / 证据不足”，不使用跨任务平均成功率替代逐任务结论。

## 6. 计时与 LeWM 对照

按照 LeWM 论文 Figure 3 的“50 次 planning 平均耗时”口径：

- 每任务固定 dev cohort 的 50 个 state-goal context；
- 每个 context 执行一次 planning；
- 预热不计入；
- CUDA 前后同步；
- 排除模型加载、进程启动、数据读取、环境 reset、视频和环境执行；
- 包含 encode、flow/CEM proposal、B scoring 和选择；
- 所有 P0/P1/P2/P3 的主条件均记录 mean、median、P95；
- candidate-clip 与 candidate-scale 额外比较投影开销；
- LeWM 使用现有 epoch10 checkpoint，在同一 dev context 和 R4 CEM budget 下重新计时；
- LeWM 主比较使用统一 30 iterations，避免与本轮任务间 CEM budget 不一致；
- 不跨任务平均 latency。

已有 LeWM `evaluation_seconds` 只作为历史参考，不直接与 planning latency 比较。

## 7. 实现、执行与交付

增加显式的 `cem_protocol` 配置和 CLI 参数，并写入每个结果的 `round4_planning` metadata。结果必须区分：

- `cem_protocol`
- `action_bound_mode`
- `action_flow_steps`
- `action_flow_integrator`
- `action_bounds`
- projection/scale 统计
- timing 统计
- forward count
- peak memory

使用独立配置和可恢复 runner，输出保存到新的 `outputs/round4/` 子目录，不覆盖旧 CEM 结果。

四个任务分别绑定 GPU0、GPU1、GPU2、GPU3 并行执行。每张 GPU 同时只运行一个任务，所有命令显式设置 `CUDA_VISIBLE_DEVICES`，禁止使用 GPU4–GPU7。正式计时阶段保证对应设备无其他任务。

## 8. 测试与验收

- 验证四任务 physical bounds 与 normalized bounds 的 round-trip；
- 验证 clip 逐坐标限制和 scale 单路径因子限制；
- 验证 CEM-clip/CEM-scale 的 B 输入、elite 和最终输出均合法；
- 验证 P2 warm-start 使用正确的 clip/scale 语义；
- 验证 P0/P3 不意外启用边界处理；
- 验证 Euler forward count 为 `step`，Heun 为 `2×step`；
- 验证所有 paired comparison 的 episode identity、start index 和 cohort hash 一致；
- 验证每个结果包含 50 episodes、完整越界统计和计时样本；
- 验证复用结果的 checkpoint、cohort、模式、step、integrator metadata；
- 验证所有主条件为 `status=ok`。

## Assumptions

- 将已有 Reacher 结果解释为：P1-candidate-scale 劣于 P1-candidate-clip，P2-candidate-scale 优于 P2-candidate-clip。
- P1 candidate-scale 只运行固定 CEM 条件；P2 candidate-scale 与 P2-candidate-clip 一样展开全部 6 step × 2 integrator。
- P0/P3 继续作为 legacy 非 CEM 控制，不启用 clip 或 scale。
- action 边界以当前本地数据集拟合的 StandardScaler 为准；normalizer 改变时必须重新计算。
- 只使用 dev 集，不新增训练、不跑 final、不设置自动扩展门槛。
