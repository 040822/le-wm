# Round 5 Phase 1：实验方案（冻结 · 已执行）

> **状态：已执行并冻结（EXECUTED / FROZEN）。**
> 本文件是 Phase 1 的方案存档，记录冻结时的设计、实现改动与执行入口。
> 执行已完成，所有结果以报告与产物为准，**本文件不再修改**。
>
> | 项 | 值 |
> |---|---|
> | 执行状态 | 完成，条件总数 244 / 244（baseline 76 + new 168），全部 `status=ok` |
> | 结果报告 | `docs/report/round5/round5_phase1_report.md` |
> | 方案决策 | `docs/report/round5/round5_final_scheme_decision.md` |
> | 结果整合 | `docs/report/round5/round3_round4_consolidation.md` |
> | 产物根目录 | `outputs/round5/phase1_seed3072_legacy/` |
> | 配置 | `config/round5/phase1.json` |
> | 入口 | `scripts/round5_phase1.py`、`scripts/run_round5_phase1.sh` |
>
> **执行结论（摘要）**：主推理方案定为 **P3（64 候选 + B 选优）、flow step 1/2**；
> round3 的 guided-flow / post-opt **可与 P3 结合**，收益任务相关，仅 Reacher 稳定为正；
> guidance 作为按任务可选增益，默认关闭。详见 `round5_final_scheme_decision.md`。

本轮目标：整合 round3 与 round4 的实验结果，确定实验最终方案。Phase 1 为 R4-AB 的
最终推理方案探索，使用旧评测方案，回答「round3 的方案（guided-flow / post-opt）
是否能与 P3 结合」。

## 1. 冻结基准

| 项 | 值 |
|---|---|
| 模型 | R4-AB，seed 3072，epoch 10（四任务，不重训） |
| checkpoint | `outputs/round4/ab_seed3072_{cube,pusht,reacher,tworoom}/checkpoints/r4_ab_seed3072_weights_epoch_10.pt`（SHA256 见 phase45_4 报告） |
| 协议 | `legacy`（LeWM 上游口径），`legacy_50.json`，每任务 50 episodes |
| 评测 | seed 42，goal_offset 25，budget 50，horizon/receding/action_block 5，Euler，CEM 300/30/30/var_scale 1.0 |
| 代码边界 | 复用 phase45_4 / phase45_3，不改写历史结果；新入口输出到独立目录 |

Legacy cohort 已核验：`build_legacy_manifest` 是 `source.common.eval.select_eval_cohort`
（`LEWM_EVAL_PROTOCOL="lewm_upstream_v1"`）的精确复刻，保留 global-last-row
exclusion、不做 episode 去重、不过滤初始成功；四任务均为 `seed=42`、
`goal_offset_steps=25`、`n=50`，id 为 `*_legacy_50_v1`。评测配置
`config/eval/*.yaml` 与上游一致（num_eval 50、goal_offset 25、budget 50、
horizon/receding/action_block 5、CEM 300/30/30/var_scale 1.0），因此可作为与
LeWM 论文可比的旧协议基准。可比较性仍要求模型/窗口(frameskip)/数据配置一致；
成功判据取自安装版 `stable-worldmodel` 运行时（`round3_runtime_predicates_v1`）。

## 2. 实验矩阵（旧协议，四任务）

### 2.1 复用（不重跑）

`outputs/round4/phase45_4_seed3072_legacy_dev` 中已存在，校验 cohort hash /
checkpoint / episode identity 后直接引用：

- P0（step 1/2/5/10/16/32，无 bound）
- P1（invariant；**reacher 使用 cem-clip**，其余任务 legacy）
- P2（step 1/2/5/10/16/32；**reacher 使用 cem-clip**，其余任务 legacy）
- P3（step 1/2/5/10/16/32，无 bound）

不含 P0-shuf、P4、cem-scale、Heun。

### 2.2 新增 guidance 条件（7 × 6 = 42 条件/任务，四任务共 168）

guidance 超参统一：`late_steps=5, inner_steps=5, step_size=0.01,
max_rms_offset=0.2`；**flow_steps = 动作流 step 网格 {1,2,5,10,16,32}**
（step=10 即 round3 原设置；step≤5 时退化为全程引导）。

| 新条件 | 含义 | 基座设置 |
|---|---|---|
| P0+GF | 单候选 Stage-A，guided-flow | bound none |
| P0+PO | 单候选采样后 post-opt 精修 | bound none |
| P2+GF | guidance 作用于 warm-start，再由 CEM 精修 | reacher cem-clip，其余 legacy |
| P2+PO | 同上，post-opt warm-start | 同上 |
| P3+GF | 每条候选生成时 guided-flow，B 选优 | N=64，bound none |
| P3+PO | 每条候选生成后 post-opt，B 选优 | N=64 |
| P3+PO-refine | 64 条普通候选先 B 选优，再对选中动作 post-opt | N=64 |

`P1+guidance` 不做（无 action flow，无意义）。

**主要分析落点为 step 1/2**，完整网格作为探索性消融。

### 2.3 「三个 step」的定义与关联

原始代码位于 `source/model/fast_lewam/jepa.py` 的 `_euler_sample` /
`_apply_guided_flow_step` / `_apply_post_opt_guidance`：

| 名称 | 代码参数 | 含义 |
|---|---|---|
| 动作流步数 | `num_steps` / `action_flow_steps`（phase1 的 1/2/…） | 动作 flow-matching ODE 的 Euler 积分步数，每步 1 次 Stage-A forward；P0/P2/P3 的可变量，P1 不用 |
| guidance 总步数 | guidance `flow_steps` | 带 guidance 的采样循环总 Euler 步数，与动作流步数是同一循环，故等于该条件的动作流步数 |
| guidance 晚启步数 | `guidance_last_steps`（round3=5） | 只在最后 N 步启用 guidance，`start = max(0, steps - late_steps)` |
| guidance 内步数 | `guidance_inner_steps`（round3=5） | 每个启用 guidance 的流步里，对 latent-cost 做几次梯度下降迭代 |

关联：被引导流步数 = `min(late_steps, flow_steps)`；每候选 backward 次数
`guided_flow = 被引导流步数 × inner_steps`，`post_opt = inner_steps`（在整条采样
结束后做一次，与 `late_steps` 无关）。round3 的 `10/5/5` 表示后 5 步被引导、每
候选 25 次内层更新；step≤5 时全部步被引导，step>5 时只引导最后 5 步。报告中必须
显式区分这两个 regime。

## 3. 实现改动

1. `make_round4_policy`：为 P0/P2 线程化
   `guidance_mode / guidance_step_size / guidance_last_steps /
   guidance_inner_steps / guidance_max_rms_offset`（P2 已由
   `ActorWarmStartModelView` 支持，只需透传）。
2. `Round4BestOfNPolicy`：新增 P3 guidance 支持
   - per-candidate：在 `_propose` 的 action 路线为 `sample_actions` 传 guidance；
     post-opt 对每条候选调用模型 guidance。
   - select-then-refine：verifier 选优后对选中候选调用 post-opt。
   - 候选生成支持分块（guidance 带 autograd 图，64 候选一次生成可能 OOM）。
3. 新配置 `config/round5/phase1.json`、runner `scripts/round5_phase1.py`，输出
   `outputs/round5/phase1_seed3072_legacy/`；每条结果写入 guidance / step /
   protocol / bound / timing / peak-memory / forward-count metadata。
4. 显式 `allowed_protocol_variants=["legacy"]`，复用 `round4_eval` 机制。
5. 复用结果通过 cohort hash / checkpoint / episode identity 校验，不覆盖旧产物。

## 4. 执行与 GPU

- 先做 cost / VRAM pre-flight：单任务单条件（P3+GF step10、P2+GF step10）实测
  延迟、峰值显存、forward/backward 次数，确定分块大小后再批量。
- 四任务各绑一张卡，优先 GPU0–3，每卡同时只跑一个任务；所有命令显式
  `CUDA_VISIBLE_DEVICES`。
- 仅在用户授权且为低显存评测时使用 GPU4–7，且北京时间 10:00–23:00 尽量不用。
- 支持 resume / shard，逐条件落盘。

## 5. 分析

- 每条件：成功率、Wilson 95%、配对 improved / regressed / net、exact McNemar p
  （同一 legacy cohort 内配对）。
- 核心对照：
  1. guidance vs 对应无 guidance（P0+GF/P0+PO vs P0；P2+GF/P2+PO vs P2；
     P3+GF/P3+PO/P3+PO-refine vs P3），逐 step 并重点看 step 1/2。
  2. step 1/2 vs step 16，检查 guidance 条件下是否仍低步更优。
  3. P3 三种 guidance 语义互比（per-candidate vs select-then-refine）。
  4. 与 round3 E2 的 guided-flow / post-opt 结果为历史参考（不同 checkpoint，
     不作因果比较）。
  5. 延迟 / 显存 / forward 次数对照，给出最终方案的成本画像。
- 结论形式：支持 / 不支持 / 证据不足，逐任务报告；不以跨任务平均代替结论。

## 6. 交付物（已完成）

- `docs/report/round5/round5_phase1_report.md`：完整条件表 + 配对比较 + timing。
- `docs/report/round5/round3_round4_consolidation.md`：round3 + round4 结果整合。
- `docs/report/round5/round5_final_scheme_decision.md`：最终方案决策书，确定主方法
  （P3 + step 1/2 ± 选定 guidance）、评测协议，以及是否需要 online train。
- 明确回答：round3 两个方案能否与 P3 结合、以何种语义结合。

## 7. 执行结论（冻结）

- 244 / 244 条件完成，全部 `status=ok`，每条件 50 episodes。
- 最终方案：**P3（64 候选）为主规划器，flow step 2 为默认（step 1 低延迟备选）；
  guidance 可选，默认关闭，对 Reacher 等未饱和任务启用**；训练用 R4-AB；
  评测用 `legacy` 旧协议。
- 结合结论：guided-flow / post-opt 均可在 P3 上运行；`post_opt_refine`
  （先选后精修）成本最低且稳定。Reacher P3 step1 加 guidance 相对无 guidance
  +16~18 pp（McNemar exact p<0.01）；cube / tworoom 接近饱和无区分度。
- 保留边界：单 checkpoint、单训练 seed、legacy cohort 非严格 held-out，
  结论为探索性描述。
