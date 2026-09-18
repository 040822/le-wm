# Round 4 P3 推理延迟探索与加速报告

## 0. 一句话结论

P3（A best-of-64 + B rerank）的规划延迟瓶颈**不在 A、也不在 encoder 的算力**，而在 Stage-B 打分对环境维的串行循环（`solver_batch_size=1`）。把它改成一次性批量后，B 快约 4×、动作逐位不变，已固化为默认值 50。其余手段里，**bf16 encoder 有效（1.36–2.1×）**、**flow step 2→1 有效（A 减半）**，而 **bf16 A/B 与 `torch.compile` 均无效甚至更慢**，goal-latent 预编码/缓存收益可忽略。

与 LeWM 的对照（同一 phase4.5-3 dev run）：同 CEM budget 下 FastLeWAM P1/P2 快 **2–9×**；P3 step2 快 **约 85–183×**，且四任务成功率均不低于 LeWM。固化 #1 后再叠加 P2/P3 的**预计**规划时间约 **44–55 ms**（reacher，g 待安静 GPU 实测），约为 LeWM 的 550–680×（见 3.9、3.10）。

---

## 1. 背景与动机

P3 的模式定义（`source/common/round4_protocol.py`）：

- A：对每个环境生成 64 条 action 候选（Stage-A action flow，Euler，flow-steps 默认 16，本探索用 2）；
- B：用 Stage-B 动力学预测每条候选的终点 latent，取到 goal latent 的 MSE 作为代价，`argmin` 选最优；
- 冻结参数：`candidate_count=64`、`flow_steps=16`（`config/round4/protocol.json`）。

起因是问：既然 A、B 都是 batch 并行，P3 是否只是"一次 A + 一次 B"，与 P0 相比没有明显增长、且远快于 P1/P2？

### 1.1 评测拓扑（关键前提）

`source/common/eval.py:53` 强制 `world.num_envs == eval.num_eval`。Round 4 dev 是 50 个 episode，因此：

- **所有 50 个环境在一个向量化 world 里并行**，`get_action` 每次拿到 batch=50；
- 全程只有 **2 次 replan**（receding horizon 5 blocks × action_block 5 = 25 步/plan，eval_budget 50）；
- 于是顶层 `round4_planning` 的 planning 统计**只有 2 个样本**，端到端延迟极易受 GPU 争用影响，**不能用来判定优化收益**；本报告所有延迟结论以隔离微基准（≥100 次重复取中位数）为准，端到端只用于核对成功率与动作等价性。

### 1.2 代码路径

| 阶段 | 位置 |
|---|---|
| A：候选生成 | `source/policy/round4.py::Round4BestOfNPolicy._propose`（action 分支一次 `sample_actions`，batch = env × 64） |
| B：候选打分 | `source/policy/round4.py::score_candidates_in_chunks` → `FastLeWAM.get_cost_from_latents`（`source/model/fast_lewam/jepa.py:1098`） |
| 选择/执行 | `Round4BestOfNPolicy.get_action`：`costs.argmin(1)`，取前 `keep_blocks` 块压入 per-env buffer |
| encode | `Round4BestOfNPolicy._encode_context` → `FastLeWAM.encode_pixels`（`jepa.py:183`） |

---

## 2. 瓶颈分析（优化前）

### 2.1 安静的冻结结果（reacher P3 dev，flow-step=2）

来源：`outputs/round4/phase45_3_seed3072_dev/conditions/reacher/P3/not_applicable/step_2/euler/dev/result.json`（`round3_revised` dev cohort，checkpoint `outputs/round4/ab_seed3072_reacher/checkpoints/r4_ab_seed3072_weights_epoch_10.pt`）。

| 组件 | 时间 | 调用形态 |
|---|---:|---|
| encode | 0.0196 s | 1 次，ViT 编 current+goal（2×50 帧） |
| A proposal | 0.0204 s | 1 次，batch 3200 × 2 步 |
| **B verifier** | **0.1361 s** | **50 次**，每个环境一行 × 64 候选 |
| 合计 | 0.1761 s | |

B 占约 **77%**，是安静时的主瓶颈。

### 2.2 B 为什么是 50 次：`solver_batch_size=1`

`score_candidates_in_chunks` 按环境维以 `solver_batch_size` 为步长循环，默认 1，即逐行调用 Stage-B（`source/policy/round4.py:64-78`）。每次只处理 1×64 的极小 batch，单次约 3 ms，但 GPU 数学只有微秒级——是 **launch/dispatch/Python 开销受限**。

微基准（50 环境 × 64 候选，150 次重复）：

| solver_batch_size | 中位数 | 每行 |
|---:|---:|---:|
| 1（原默认） | 158.1 ms | 3.16 ms |
| 5 | 49.7 ms | 9.9 ms |
| 25 | 23.9 ms | — |
| **50** | **23.1 ms** | — |

- **加速 6.85×**；
- `argmin` 完全一致（代价最大差 3.6e-7），与函数 docstring 的"切分不变"承诺一致。

### 2.3 encode 不随帧数扩展

微基准（100 次中位数）：

| | 中位数 |
|---|---:|
| encode current only（50 帧） | 30.5 ms |
| encode current+goal（100 帧） | 33.5 ms |

帧数翻倍只涨 10%，说明 encode 基本是 per-call 固定开销（ViT kernel launch + H2D + 归一化），不是 batch 算力受限。

### 2.4 小 DiT 上 bf16 反而更慢

微基准（batch 64，150 次中位数）：

| | fp32 | bf16 autocast |
|---|---:|---:|
| B verifier | 4.16 ms | 4.29 ms |
| A 2-step flow | 7.34 / 7.27 ms | 8.36 / 8.30 ms |
| 单次 stage_A forward | 3.62 ms | 4.26 ms |

模型太小（`latent_dim=192`、6 层 DiT、hidden 192），bf16 的权重 cast 与分发开销超过任何 tensor-core 收益。

### 2.5 调整后的图景（#1 之后）

安静时每 replan 变为 B ≈ 23 ms、A ≈ 20 ms、encode ≈ 20 ms，三者量级相当。端到端里看到的 `A≈0.10 / encode≈0.16` 是共享 GPU 争用放大的结果。

---

## 3. 优化实验

环境：8×RTX 4090 全部被他人工负载（`dexCG`、`isaac-sim`）占满；GPU4–7 经用户显式授权用于低显存 eval。所有延迟取多次重复中位数以抗争用。

### 3.1 #1 `solver_batch_size = 50`（已固化，✅ 有效）

端到端（reacher P3 step2 dev，50 episodes，各跑两次）：

| variant | success | total | A | B | encode |
|---|---:|---:|---:|---:|---:|
| base_sb1 | 0.90 | 0.397 | 0.101 | 0.136 | 0.161 |
| base_sb1_2 | 0.90 | 0.384 | 0.094 | 0.129 | 0.161 |
| **sb50** | 0.90 | **0.290** | 0.095 | **0.034** | 0.161 |
| **sb50_2** | 0.90 | **0.282** | 0.091 | **0.032** | 0.158 |

- B ≈ **4×** 更快，总规划 ≈ 1.36× 更快；
- 峰值显存 410 MB → 460 MB；
- **success 0.90 不变，执行动作与基线逐位相同**（对比 trace 前 6 步 action，`identical=True`）。

**已固化为默认值**（见第 4 节）。

### 3.2 #2 goal-latent 缓存 / 预编码（❌ 无效）

- 实现：`_goal_latent_cache` 按 env 缓存 goal latent，`_needs_flush` 时失效；首次 replan 编 `cat(current,goal)` 并缓存，后续 replan 只编 current。
- 与"评测前一次性预编码"**总工作量相同**：预编码 = `50(全部goal)+2×50(current)=150` 帧；懒缓存 = `100+50=150` 帧。
- 隔离微基准（2 次 replan）：无缓存 67.1 ms，有缓存 64.1 ms，**仅省 3.0 ms（4.5%）**；
- 端到端 encode 无可分辨变化（2 样本 + 争用）。
- 结论：收益可忽略，不作为默认。

### 3.3 #3 proposal 噪声/reshape 微优化（❌ 无效）

- 实现：跳过预生成 noise + clone，直接让采样器用同一 `generator` 产噪声（`--optimize-proposal`）。
- 微基准：预生成+clone 7.336 ms vs 采样器内生成 7.270 ms，**无差别**。
- 之前观察到的 "A 有 0.05–0.11 s 开销" 实为 A 的 GPU 计算（`proposal_seconds` 在 `_sync` 后测、`flow_seconds` 只测 launch），非可回收的 host 张量开销。
- 结论：放弃。

### 3.4 bf16 A/B verifier（❌ 更慢）

端到端（probe1）：

| variant | success | total | A | B | encode |
|---|---:|---:|---:|---:|---:|
| baseline | 0.90 | 0.390 | 0.098 | 0.127 | 0.165 |
| bf16ver | 0.92 | 0.535 | 0.096 | 0.276 | 0.163 |
| bf16ver2 | 0.92 | 0.434 | 0.076 | 0.223 | 0.134 |
| bf16both | 0.92 | 0.422 | 0.130 | 0.158 | 0.134 |
| bf16both2 | 0.92 | 0.459 | 0.137 | 0.161 | 0.162 |

端到端 bf16ver 看似慢 2×，但 bf16both 并不慢，且微基准只有 ~3% 差异 → 端到端差异是争用噪声。可信结论来自 2.4：**小 DiT 上 bf16 不划算**。成功率 0.92 vs 0.90 无损失。结论：不采用（保留为诊断开关）。

### 3.5 P3：bf16 encoder（✅ 有效）

ViT 比小 DiT 计算重，bf16 真正吃到 tensor core：

| 输入形态 | fp32 | bf16 | 倍数 |
|---|---:|---:|---:|
| float（已在 GPU） | 33.5 ms | **15.7 ms** | 2.1× |
| uint8（E2E 真实路径） | 35.5 ms | **26.0 ms** | 1.36× |

端到端（probe3，flow-step=2）：

| variant | success | total | encode |
|---|---:|---:|---:|
| base_s2 / base_s2_2 | 0.90 / 0.90 | 0.299 / 0.279 | 0.163 / 0.153 |
| bf16enc_s2 / _2 | 0.94 / 0.94 | 0.309 / 0.364 | 0.173 / 0.204 |

- 成功率 0.94 vs 0.90（+2 episode，仍在 Wilson 噪声内），无损失；
- 端到端 encode 时间被争用主导，未能复现隔离收益；以微基准为准；
- 数值影响：与基线动作逐位不同（`max|Δaction|≈1.39`），因为它改变了编码 latent，属**方法级数值改动**，需多 seed / final 复核。
- 实现：`_encode_context` 内 autocast，输出 `.float()` 转回 fp32，A/B dtype 不变（`--bf16-encode`）。

### 3.6 P2：flow step 2→1（✅ 有效，属方法层）

| | A 中位数 |
|---|---:|
| A 2-step eager | 33.0 ms |
| **A 1-step eager** | **17.3 ms** |

- E2E 成功率 0.92/0.92 vs 基线 0.90/0.90，无损失（reacher 在 `phase45_4` 中本就是 step1 最好，92%）；
- 改变 P3 采样轨迹，属方法级改动，需 final 复核；
- 开关：`--action-flow-steps 1`。

### 3.7 P1：`torch.compile` / CUDA graph（❌ 更慢，放弃）

噪声预生成后（公平设定），150 次中位数：

| | eager | compile default | compile reduce-overhead |
|---|---:|---:|---:|
| A 1-step | 17.3 ms | 25.3 ms | 25.2 ms |
| A 2-step | 33.0 ms | 49.3 ms | 49.3 ms |
| encode | 33.5 ms | 31.6 ms | 38.6 ms |

- A 的 compile 稳定**更慢**（~50%）；encode default 仅快 6% 且需 ~10 s 编译期 + 动态 shape 风险。
- 早先一版"compile 更快"是把 `randn` 放进被编译函数导致的假象（输出 `max_diff=2.5` 即随机噪声不同）。
- 结论：**不引入 compile / CUDA graph**。

### 3.8 组合效果

P2 + P3（flow-step=1 + bf16 encode）对 A+encode 的隔离中位数：

| | A | encode | A+encode |
|---|---:|---:|---:|
| 现状（step2, fp32） | 33.0 | 33.5 | 66.5 ms |
| P2+P3（step1, bf16） | 17.3 | 15.7 | **33.0 ms（~2×）** |

加上 #1 已固化的 B（≈23 ms），安静 GPU 下规划可由 ≈89 ms 降至 ≈56 ms。

### 3.9 与 LeWM 的规划速度对比

数据来源：同一轮 `round4 phase4.5-3` dev run（`outputs/round4/phase45_3_seed3072_dev`），同一个 `round3_revised` 50-episode cohort、同一 R4-AB seed3072 epoch10 checkpoint、同一 GPU 条件。LeWM 使用统一的 300/30/30 CEM budget（`docs/report/round4/round4_phase45-3_flow_step_report.md` 的 "LeWM latency reference"）。

Planning 中位数（秒）：

| 任务 | LeWM (CEM 300/30/30) | P1 随机 CEM | P2 热启动 CEM | P3 step2 | P3 step16 |
|---|---:|---:|---:|---:|---:|
| cube | **31.73** | 15.42 | 10.55 | 0.310 | 0.556 |
| pusht | **22.25** | 5.01 | 3.45 | 0.262 | 0.243 |
| reacher | **29.98** | 6.66 | 5.35 | 0.176 | 0.323 |
| tworoom | **38.38** | 4.38 | 4.26 | 0.209 | 0.408 |

相对 LeWM 的加速倍数：

| 任务 | P1（同 CEM budget） | P2（同 CEM budget） | P3 step2 | P3 step16 |
|---|---:|---:|---:|---:|
| cube | 2.1× | 3.0× | **102×** | 57× |
| pusht | 4.4× | 6.5× | **85×** | 92× |
| reacher | 4.5× | 5.6× | **170×** | 93× |
| tworoom | 8.8× | 9.0× | **183×** | 94× |

成功率对照：

| 任务 | LeWM | P3 step2 |
|---|---:|---:|
| cube | 54% | 100% |
| pusht | 94% | 100% |
| reacher | 82% | 90% |
| tworoom | 90% | 100% |

结论：

- **同算法口径（都是 CEM 300/30/30）**：FastLeWAM P1/P2 比 LeWM 快 **2–9×**，且 forward count 量级相近（LeWM 1680–2430，P1/P2 1500–2640），说明差距来自"每次 forward 更便宜"，不是少算。
- **不同算法口径（P3 best-of-64 generate+select）**：比 LeWM 快 **约 50–180×**，同时四任务成功率均不低于 LeWM。
- 论文中应把这两类结论分开陈述：P1/P2 是同一 CEM 算法的实现加速，P3 是方法级替换。
- 历史背景：`docs/report/round1_experiment_report.md` 曾记录 Fast Stage-B 四任务 wall-clock 是 LeWM 的 **6.82×（更慢）**；到 P3 已反转为快约两个数量级。

口径警告：LeWM 与 FastLeWAM 的 `forward_count` 由各自 timing wrapper 定义，**不要跨方法比 forward count**；只有 latency 可比。以上均为 dev / 单 seed / 单 checkpoint 的描述性结果，planning 统计仅 2 个样本且含争用噪声，但两个数量级的差距是稳健的。

### 3.10 加速策略的预计时间（★待安静 GPU 实测）

以下为**估计值**：以安静时的冻结基线（`phase45_3` reacher P3 step2：encode 20 ms + A 20 ms + B 136 ms = 176 ms）为基准，套用本报告测得的隔离倍数外推（B 6.85×、A step1 2×、uint8 encode bf16 1.36×）。**真实数字待 GPU 空闲后按第 7 节的安静基线流程实测并回填。**

reacher P3 step2，单次 replan，单位 ms：

| 配置 | encode | A | B | 合计 | 相对基线 |
|---|---:|---:|---:|---:|---:|
| 基线（step2, fp32, sb=1） | 20 | 20 | 136 | **176** | 1.00× |
| #1 `sb=50`（已固化） | 20 | 20 | ~20–30 | **~60–70** | ~2.6–2.9× |
| #1 + P2 `step1` | 20 | ~10 | ~20–30 | **~50–60** | ~3.0–3.5× |
| #1 + P3 `bf16-encode` | ~14–15 | 20 | ~20–30 | **~54–65** | ~2.7–3.3× |
| #1 + P2 + P3 | ~14–15 | ~10 | ~20–30 | **~44–55** | ~3.2–4.0× |

对 LeWM（reacher 29983 ms）的预计加速：基线 170× → 全优化后 **约 550–680×**（估计）。

注：
- B 的 20–30 ms 区间来自争用下的微基准（23 ms）与端到端实测（34 ms）；安静时可能更低。
- P2（step1）与 P3（bf16 encode）都会改变数值/轨迹，属方法级改动，估计收益需与 final cohort 成功率一起复核后才能固化。
- encode 的 bf16 收益在 uint8 真实路径上为 1.36×（float 输入为 2.1×），上表按 1.36× 保守估计。

### 3.11 P0 与 P3 的对照（step 1/2）

P0 是**直接生成单条动作、无 B rerank**（`proposal_source=action`, `candidate_count=1`），而 P3 是 best-of-64 + B。同一 phase45_3 dev run 的 planning 中位数（秒）与成功率：

| 任务 | P0 step1 | P0 step2 | P0 step16 | P3 step1 | P3 step2 | LeWM |
|---|---:|---:|---:|---:|---:|---:|
| cube | 0.488² | 0.054 | 0.112 | 0.288 | 0.310 | 31.73 |
| pusht | 0.231 | 0.045 | 0.082 | 0.130 | 0.262 | 22.25 |
| reacher | 0.221 | 0.028 | 0.090 | 0.183 | 0.176 | 29.98 |
| tworoom | 0.193 | 0.022 | 0.068 | 0.124 | 0.209 | 38.38 |

² P0 step1 的 cube 0.488 s 是 **warmup 假象**：planning 只有 2 个样本，step1 首次 replan 含 CUDA 初始化/自动调优，中位数被抬高；P0 step2 与 step16 不受影响，才是真实水平。

成功率：

| 任务 | P0 step1 | P0 step2 | P3 step1 | P3 step2 | LeWM |
|---|---:|---:|---:|---:|---:|
| cube | 100% | 100% | 100% | 100% | 54% |
| pusht | 96% | 100% | 98% | 100% | 94% |
| reacher | 90% | 84% | 92% | 90% | 82% |
| tworoom | 98% | 98% | 98% | 100% | 90% |

相对 LeWM 的加速倍率（P0 step2 / P3 step1）：

| 任务 | P0 step2 vs LeWM | P3 step1 vs LeWM | P3 step1 / P0 step2 |
|---|---:|---:|---:|
| cube | 585× | 110× | 5.3× |
| pusht | 492× | 171× | 2.9× |
| reacher | 1063× | 164× | 6.5× |
| tworoom | 1721× | 309× | 5.6× |

结论：

- **P0 比 P3 便宜约 3–7×**，因为 P0 只有 1 条候选、且没有 B 打分；P3 的 64 候选 A + B 是这部分代价的来源。
- 代价换来了精度：reacher 上 P0 step2 只有 84%，P3 step1/step2 为 92%/90%；pusht P0 step2 100% 而 P3 step1 98%——多数任务 P0 已够好，rerank 的净收益主要体现在 reacher。
- 上表 P3 仍用的是 `solver_batch_size=1`；**固化 #1 后 P3 的 B 大幅下降**，P3 step1 预计降到 P0 step2 的 ~2×（reacher 约 55 vs 28 ms，见 3.10）。
- P0 step2 已经是"接近下限"的规划时间（encode 占大头），进一步压缩空间有限。

---

## 4. 已固化改动：#1 solver_batch_size 默认 50

| 文件 | 改动 |
|---|---|
| `source/common/round4_protocol.py` | `ROUND4_DEFAULTS["best_of_n"]["solver_batch_size"] = 50` + 说明注释 |
| `config/round4/protocol.json` | `best_of_n.solver_batch_size: 50` |
| `source/common/round4_eval.py` | `run_round4_evaluation(..., solver_batch_size: int \| None = None)`，`None` 时解析为冻结默认，并校验为正 |
| `scripts/round4.py` | `--solver-batch-size` 默认 `None`（回落冻结默认），仍可显式覆盖 |
| `tests/test_round4_protocol.py` | 新增 `test_best_of_n_solver_batch_size_is_frozen_at_the_dev_world_size` |

验证（driver 不传 `--solver-batch-size`）：解析为 50，planning 0.302 s，B 0.038 s，success 0.90。

兼容性：历史/诊断脚本均显式传 `solver_batch_size=1`（`round4_phase45_3.py`、`round4_phase45_4.py`、`round4_reacher_action_bounds.py`、两个 shell），可复现性不受影响；`config/round4/ab_control.json` 等实验专属配置保持原值不动。

**注意**：final cohort 是 200 个环境，50 会拆成 4 个 row batch（原为 200 次）。若想让 final 也一轮过完，跑 final 时显式传 `--solver-batch-size 200`（显存约多几百 MB）。

---

## 5. 诊断开关（全部默认关闭，供后续复核）

均通过 CLI → `run_round4_evaluation` → `make_round4_policy` → `Round4BestOfNPolicy` 贯通：

| flag | 作用 | 结论 |
|---|---|---|
| `--bf16-proposal` | A flow 用 bf16 autocast | 无效 |
| `--bf16-verifier` | B 打分用 bf16 autocast | 无效 |
| `--optimize-proposal` | 采样器内产噪声，省一次 clone | 无效 |
| `--cache-goal-latent` | 跨 replan 缓存 goal latent | 收益可忽略 |
| `--bf16-encode` | encoder 用 bf16 autocast，输出转回 fp32 | **有效，待 final 复核** |
| `--action-flow-steps 1` | A flow 步数（已有） | **有效，属方法层** |

新增字段均记录进 `round4_planning` 与 `parameters`。

---

## 6. 数据与复现

- 探索输出根：`outputs/round4/perf_probe_reacher/<variant>/reacher/P3/dev/`
  - `result.json`（含 `round4_planning` 与逐 episode 记录）
  - `trace/episodes.jsonl`（含逐 step action）
  - 日志：`outputs/round4/perf_probe_reacher/logs/*.log`
- driver：`/tmp/opencode/run_p3_perf_probe.py`（GPU7，显式 `CUDA_VISIBLE_DEVICES`）
- 微基准脚本：`/tmp/opencode/bench_ab.py`、`bench_rows.py`、`bench_encode.py`、`bench_opt.py`、`bench_opt2.py`
- 基准 checkpoint：`outputs/round4/ab_seed3072_reacher/checkpoints/r4_ab_seed3072_weights_epoch_10.pt`
- cohort：`outputs/round3/phase1/cohorts/reacher/dev_round3_revised.json`（SHA256 `d18695b2...f1d06`）
- 冻结对照：`outputs/round4/phase45_3_seed3072_dev/...`、`outputs/round4/phase45_4_seed3072_legacy_dev/...`

复现命令模板（低显存 eval，仅 GPU4–7 且需授权）：

```bash
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -u /tmp/opencode/run_p3_perf_probe.py <variant> \
  --gpu 7 [--action-flow-steps 1] [--bf16-encode] [--solver-batch-size N]
```

---

## 7. 局限与后续

1. **延迟只在共享 GPU 上测得**，端到端 planning 仅 2 样本、被争用污染；结论依赖隔离微基准。正式实验前应在**安静窗口重测**一次真实基线。
2. **仅 reacher、单 seed（3072 epoch10）、dev cohort**。P2（step1）与 P3（bf16 encode）都会改变数值/轨迹，需在 **final cohort + 多个 seed + 四任务**上复核成功率与延迟，才能作为方法级默认。
3. `torch.compile`/CUDA graph 在本模型规模下无收益；后续若模型变大或 batch 增大可重新评估。
4. 尚未做的结构性空间：encode 的 H2D/归一化（`jepa.py:195-201`）融合、把 `bf16` 与 `solver_batch_size` 等执行参数统一到一个显式的 `execution` 配置块。

### 待实测清单（GPU 空闲后）

- [ ] 在**安静窗口**重测 reacher P3 step2 的真实组件拆分（encode / A / B），回填 3.10 的预计值。
- [ ] 在同一安静条件下测 `#1`、`#1+P2`、`#1+P3`、`#1+P2+P3` 四种配置的真实 planning 时间。
- [ ] 对 P2（step1）与 P3（bf16 encode）在 **final cohort + 多 seed + 四任务**上复核成功率与延迟。
- [ ] final（200 env）下确认 `solver_batch_size` 取值（50 拆 4 批，或显式 200 一轮过完）。
- [ ] 回填 LeWM 对比的最终加速倍数，并把 3.9 的结论写进论文实验节。

---

## 附：核心数字速查

- B 打分微基准：`solver_batch_size` 1→50：158.1 → 23.1 ms（6.85×，argmin 一致）。
- A flow：2-step 33.0 ms → 1-step 17.3 ms（1.9×）。
- encode：fp32 33.5 → bf16 15.7 ms（float 输入 2.1×）；uint8 路径 35.5 → 26.0 ms（1.36×）。
- 小 DiT bf16：B +3%，A +14%（更慢）。
- `torch.compile`：A 33→49 ms（更慢）。
- goal 缓存：2 replan 省 3.0 ms（4.5%）。
- 已固化默认后：reacher P3 step2 dev planning 0.39 → 0.30 s（B 0.136 → 0.034 s），动作逐位一致，success 0.90。
- LeWM planning（phase45_3 dev）：cube 31.73 / pusht 22.25 / reacher 29.98 / tworoom 38.38 s。
- 同 CEM budget：FastLeWAM P1/P2 比 LeWM 快 2–9×；P3 step2 快 85–183×。
- P0 step2（单候选、无 rerank）比 LeWM 快 492–1721×；P3 step1 是 P0 step2 的 2.9–6.5×。
- **预计（待实测）**：reacher P3 step2 全优化后 ~44–55 ms，约 LeWM 550–680×。
