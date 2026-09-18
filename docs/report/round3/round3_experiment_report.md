# Round 3 实验总报告

**归档日期**：2026-09-19  
**实验截止记录**：2026-09-16 22:31  
**评测协议**：`round3_revised`（dev 50 episodes，final 200 episodes；CEM 300 samples / 30 iterations / top-k 30）  
**主指标**：Stage-B success rate；Stage-A 与 A-shuf 为辅助指标  
**任务范围**：Cube、Push-T、Reacher、TwoRoom

本报告是 Round 3 的汇总入口，覆盖 Phase 1（新协议复评）、Phase 1.5（Fast 起点选择）、Phase 2（在线后训练）、Phase 3（physical_time_type 探索）以及 Online Supervision（E1/E2 guidance 诊断、T0–T6 fixed-replay 训练）。各阶段细节见同目录下的分阶段报告，索引见 [`README.md`](README.md)。

---

## 0. 执行摘要

| 阶段 | 实验内容 | 当前状态 |
|---|---|---|
| Phase 1 | `round3_revised` 新评测协议审计与复评，覆盖 Cube、Push-T、Reacher、TwoRoom | 已完成 |
| Phase 1.5 | E3/E5 Fast 起点选择 | 已冻结为 E5 legacy |
| Phase 2 | E0 LeWM、E5 Fast 的 offline continuation vs online adaptation | Cube 主实验完成，其余三任务曲线完成后补全 |
| Phase 3 | E5 `physical_time_type` 表示探索 | 四任务完成，已封存为 exploratory |
| Online Supervision E1/E2 | 局部梯度真实性、Post-opt / Guided-flow 采样方法诊断 | 已完成 |
| Online Supervision T0–T6 | 固定 replay 上比较不同在线监督信号 | Reacher、Push-T 均完成 200 optimizer updates |

**核心结论**：

1. 新协议保留原有高层结论：Fast-LeWAM 收益具有任务依赖性，目标条件控制有效，单一 Stage-B capacity 或 serial rollout 不能普遍解决规划 gap。
2. 新协议下效应幅度普遍减弱，且部分局部结论被改写：Reacher 的 Fast 优势与 E3-384 超过 E0 的结论不再成立。
3. 在当前预算、更新频率和 replay 配置下，没有稳定的跨任务 online adaptation 收益。
4. `physical_time_type` 没有跨任务统一收益，已标记为 `completed_exploratory_not_mainline`。
5. T0–T6 fixed-replay 筛选中，没有任何新增监督臂超过 T0 frozen baseline；T3 ranking 因 `ranking_pairs=0` 实际退化为 T2。
6. 新的 Online Supervision 尚未进入 closed-loop 20k 环境步复验阶段。

---

## 1. 评测协议与统计口径

新协议 `round3_revised` 相对 Round 1/2 旧协议的主要变化：

| 维度 | Round 1/2 旧协议 | Round 3 新协议 |
|---|---|---|
| 最终测试规模 | 每配置 50 个固定测试 episode | 每任务 200 个 final episode，另 50 个 dev episode |
| 测试 cohort | 固定测试集，无完整 cohort 审计 | final/dev/online pool episode-disjoint；按物理距离分层；冻结 manifest 与 SHA256；排除初始即成功 episode；不允许模型选样 |
| 成功谓词 | 沿用旧实现 | 运行时审计：Cube 位置 L2 ≤ 0.04 m；Reacher 每关节误差 < 0.05 rad；Push-T 位置 < 20 px 且角度 < π/9；TwoRoom proprioception L2 < 16 px |
| 规划预算 | 沿用旧实验 | seed=42、goal_offset_steps=25、eval_budget=50、horizon/action_block=5、CEM 300 samples / 30 steps / topk=30 / var_scale=1.0 |
| checkpoint 规则 | 通常 epoch10 最终 checkpoint | 仍用最终 checkpoint；不使用 intermediate test peak |
| 观测与审计 | episode 级结果为主 | 额外记录 per-step state、goal、distance、actions、legality、termination，保存 pairwise matched episode_id/start_step |

统计约定：所有百分比为 success rate；差值使用百分点（pp）；Stage-B 为主比较对象；A-shuf 不与正常 A 混合平均；多种子使用样本标准差；不跨任务平均原始 success rate。

---

## 2. Phase 1：新协议复评

### 2.1 Final 200-episode 主结果

下表为 seed3072 主权重；E1 使用 Stage-B，E2 使用 Stage-A / A-shuf，E3/E4/E5 为 A / A-shuf / B 三元组。TwoRoom 的 E1–E4 为本轮补充训练结果（supplemental），不回写原始 canonical matrix。

| 任务 | E0 B | E1 B | E2 A / A-shuf | E3 A / A-shuf / B | E4 A / A-shuf / B | E5 A / A-shuf / B | E6 warm-B |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cube | 48.0% | 36.0% | 100.0% / 11.0% | 99.5% / 12.0% / 46.0% | 99.5% / 10.0% / 53.5% | 99.5% / 9.0% / 49.5% | 97.5% |
| Push-T | 93.0% | 89.0% | 90.5% / 9.0% | 94.0% / 7.5% / 84.5% | 89.0% / 11.0% / 85.5% | 93.0% / 7.0% / 89.5% | 93.0% |
| Reacher | 84.5% | 84.0% | 69.5% / 9.0% | 70.5% / 6.5% / 84.5% | 73.0% / 7.5% / 84.5% | 79.5% / 9.0% / 84.5% | 68.5% |
| TwoRoom | 84.5% | 89.0%* | 92.5% / 42.5%* | 95.5% / 44.5% / 90.5%* | 95.0% / 45.0% / 96.5%* | 95.0% / 44.0% / 95.5% | 96.5% |

\* TwoRoom E1–E4 是后补训练，属于 supplemental，不回写原始 canonical matrix。

新协议下 E0 task-level baseline 为 Cube 48.0%、Push-T 93.0%、Reacher 84.5%、TwoRoom 84.5%。E5 Stage-B 相对 E0 分别为 +1.5、−3.5、0.0、+11.0 pp，说明 Fast Stage-B 仍没有跨任务统一优势，但任务间关系已发生变化。

### 2.2 Round 1 多种子配对

| 配对比较 | 旧协议均值差 | 新协议均值差 | 新协议逐种子差值 |
|---|---:|---:|---|
| Cube：E3 B − E1 B | +11.3 ± 6.4 pp | +7.8 ± 2.6 pp | +10.0、+8.5、+5.0 |
| Push-T：E3 B − E1 B | −2.7 ± 4.2 pp | −3.8 ± 0.6 pp | −4.5、−3.5、−3.5 |
| Push-T：E5 A − E4 A | +4.7 ± 3.1 pp | +2.2 ± 1.6 pp | +4.0、+1.5、+1.0 |
| Push-T：E5 B − E4 B | −0.7 ± 6.4 pp | +1.5 ± 1.7 pp | +3.5、+0.5、+0.5 |

Cube 的 E3 正迁移、Push-T 的 actor 受世界模型监督收益仍能复现，但幅度都小于旧协议。

### 2.3 E6 warm-start 配对

| 任务 | 旧协议 E6 − E5 | 新协议 E6 − E5 | 判断 |
|---|---:|---:|---|
| Cube | +32 pp | +48 pp | 明显提升 |
| Push-T | −2 pp | +4 pp | 方向由略降变为略升 |
| Reacher | −16 pp | −16 pp | 明显下降 |
| TwoRoom | 0 pp | +1 pp | 接近持平 |

E6 仍表现出强烈任务依赖，不能作为跨任务默认改进。

### 2.4 Round 1 / Round 2 结论更新

| 原结论 | 新协议证据 | 判定 |
|---|---|---|
| Cube 存在稳定 E3 相对 E1 的单向正迁移 | 三种子均为正，+7.8 ± 2.6 pp | 确认，幅度减弱 |
| Action supervision 的 planner gain 不跨任务泛化 | Push-T −3.8 ± 0.6 pp；Reacher +0.5 pp | 确认并略加强 |
| 世界模型监督帮助 actor | Push-T E5 A − E4 A = +2.2 ± 1.6 pp | 确认，幅度减弱 |
| Cross-head gradient 没有稳定 planner gain | Push-T E5 B − E4 B = +1.5 ± 1.7 pp；Cube −4 pp；Reacher 0 | 局部弱正向，未默认化 |
| Goal conditioning 有效 | Cube/Push-T/Reacher A-shuf 12.0% / 7.5% / 6.5%；TwoRoom E5 44.0% | 确认 |
| E6 actor warm-start 强烈依赖任务 | Cube +48、Push-T +4、Reacher −16、TwoRoom +1 pp | 确认 |
| Fast Stage-B 没有总体超过 LeWM | E5 B − E0：Cube +1.5、Push-T −3.5、Reacher 0、TwoRoom +11 pp | 总体确认，任务图改变 |
| E1-384 不能修复 Stage-B gap | Push-T −3.5 pp、Reacher −3.0 pp，均低于新 E0 | 确认并加强 |
| Reacher E3-384 是唯一超过 E0 的配置 | Reacher E3-384 B=82.0% < E0=84.5%；Push-T 88.5% < 93.0% | **推翻** |
| E3-384 相对 E3-192 有小幅收益 | Push-T +4.0 pp、Reacher −2.5 pp | 改为任务依赖 |
| Learned physical-time/type 只作 optional | Push-T +5.0 pp、Reacher −1.0 pp | 确认，任务差异更清楚 |
| Serial one-step 不应成为全局默认 | Push-T +1.5 pp、Reacher −15.5 pp | 确认并加强 |

### 2.5 数据完整性与失败案例

- 扩展评测共 **42 个 manifest entry、93 个 stage task**，全部完成，均为 200 episode 且 `save_video=true`，共产出 **18,600 个 episode 视频**，完整性校验全部通过。
- 原 Canonical Phase 1 矩阵 28 个配置中，25 个有效、3 个 `missing_weight`（TwoRoom E3）。
- 失败案例统计：**338 个至少出现一次失败的 episode、649 个失败事件**。

| 任务 | 可用条件 | 至少一次失败 episode | 失败率 | 失败事件 |
|---|---|---:|---:|---:|
| Cube | E0B+E3A+E3B+E5A+E5B | 137 | 68.5% | 315 |
| Push-T | E0B+E3A+E3B+E5A+E5B | 50 | 25.0% | 91 |
| Reacher | E0B+E3A+E3B+E5A+E5B | 113 | 56.5% | 193 |
| TwoRoom | E0B+E5A+E5B（E3 不可用） | 38 | 19.0% | 50 |

Cube 与 Reacher 的失败率明显高于 Push-T 与 TwoRoom；Cube 的主要失败集中在 Stage-B 和 invalid action。

### 2.6 TwoRoom Round 1 E1–E4 补充训练

补充训练统一 seed=3072、`latent_head_dim=192`、MLP dim=768、batch size=128、10 epoch、epoch10 checkpoint。

| 实验 | 新协议 A | 新协议 A-shuf | 新协议 B | 旧协议 B |
|---|---:|---:|---:|---:|
| E1 | — | — | 89.0% | 86.0% |
| E2 | 92.5% | 42.5% | — | — |
| E3 | 95.5% | 44.5% | 90.5% | 90.0% |
| E4 | 95.0% | 45.0% | 96.5% | 98.0% |

验收：E1–E4 epoch10 checkpoint 4/4 存在；新协议 9/9 `status=ok`（每项 200 episode）；旧协议 9/9 `status=ok`（每项 50 episode）；仅使用 GPU0–GPU3。

### 2.7 新协议对绝对结果的影响

| 任务 | 旧 E0 B | 新 E0 B | Δ | 旧 E3 B | 新 E3 B | Δ | 旧 E5 B | 新 E5 B | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cube | 78.0% | 48.0% | −30.0 pp | 74.0% | 46.0% | −28.0 pp | 66.0% | 49.5% | −16.5 pp |
| Push-T | 98.0% | 93.0% | −5.0 pp | 90.0% | 84.5% | −5.5 pp | 86.0% | 89.5% | +3.5 pp |
| Reacher | 72.0% | 84.5% | +12.5 pp | 82.0% | 84.5% | +2.5 pp | 84.0% | 84.5% | +0.5 pp |
| TwoRoom | 86.0% | 84.5% | −1.5 pp | — | — | — | 100.0% | 95.5% | −4.5 pp |

Cube 绝对 success rate 大幅降低是 cohort、排除初始成功与成功判定共同带来的测量变化；Reacher E0 baseline 上升到 84.5%，使旧报告 Fast Stage-B 优势消失。新协议下应使用新 E0 baseline。

---

## 3. Phase 1.5：Fast 起点选择

比较 canonical dev Stage-B：

| 任务 | E3 | E5 | E5 − E3 |
|---|---:|---:|---:|
| Cube | 42% | 48% | +6 pp |
| Push-T | 86% | 94% | +8 pp |
| Reacher | 78% | 78% | 0 pp |
| 共同任务平均 | — | — | +4.67 pp |

最终选择：

- 全局 Fast 起点：**E5 legacy checkpoint**。
- 不采用 capacity-384、serial one-step、clean-action timestep 作为默认配置。
- `physical_time_type` 作为探索性分支保留。

TwoRoom 的 canonical E3 权重当时缺失；补充 E3 checkpoint 的 final Stage-B 为 90.5%，作为 supplemental evidence 记录。

---

## 4. Phase 2：在线后训练

实验臂：`freeze`、`offline_continue`、`online_adapt`。每 100 个真实环境步更新一次，20k 环境步，共 200 次更新。Offline 臂用 64 条离线样本，Online 臂用 32 离线 + 32 在线样本。

### 4.1 原始 Cube dev gate（50-episode cohort，0/5k/10k/20k）

E0 Cube：

| 臂 | 0 | 5k | 10k | 20k |
|---|---:|---:|---:|---:|
| Freeze | 54% | 54% | 54% | 54% |
| Offline continue | 54% | 62% | 56% | 48% |
| Online adapt | 54% | 60% | 56% | 48% |

E5 Cube：

| 臂 | 0 | 5k | 10k | 20k |
|---|---:|---:|---:|---:|
| Freeze | 48% | 48% | 48% | 48% |
| Offline continue | 48% | 42% | 38% | 48% |
| Online adapt | 48% | 58% | 56% | 44% |

两者都没有通过“online 同时超过 freeze 和 offline 至少 5 pp”的扩展门，因此没有按原规则继续增加 online seed。

### 4.2 Final-200 曲线诊断（20k endpoint）

| 任务 / 模型 | Offline | Online | Online − Offline |
|---|---:|---:|---:|
| Cube E0 | 52.0% | 56.5% | +4.5 pp |
| Cube E5 | 48.0% | 60.0% | +12.0 pp |
| Reacher E0 | 80.5% | 80.5% | 0 pp |
| Reacher E5 | 83.5% | 80.5% | −3.0 pp |
| Push-T E0 | 93.5% | 94.0% | +0.5 pp |
| Push-T E5 | 88.0% | 84.0% | −4.0 pp |
| TwoRoom E0 | 87.0% | 87.0% | 0 pp |
| TwoRoom E5 | 95.0% | 94.5% | −0.5 pp |

曲线说明结果对 cohort 和评测规模较敏感（例如 E5 Cube dev 20k 为 44%，final 200 为 60%）。最终判断：在当前预算、更新频率和 replay 配置下，没有稳定的跨任务 online adaptation 收益。

---

## 5. Phase 3：physical_time_type 探索

四任务均随机初始化、seed 3072、192 维模型。

| 任务 | Stage-A | A-shuf | Stage-B | 相对历史 E5 legacy B |
|---|---:|---:|---:|---:|
| Cube | 100.0% | 8.5% | 51.5% | +2.0 pp |
| Push-T | 93.5% | 12.5% | 86.0% | −3.5 pp |
| Reacher | 73.0% | 6.0% | 83.0% | −1.5 pp |
| TwoRoom | 93.5% | 45.5% | 96.0% | +0.5 pp |

没有跨任务统一收益，已标记为 `completed_exploratory_not_mainline`；Round 3 Fast 主线继续采用历史 E5 legacy checkpoint。

---

## 6. Online Supervision：E1/E2 guidance 诊断

实验限定 Reacher 和 Push-T。

### 6.1 E1：局部梯度真实性

每任务 3 个 RMS（0.01/0.03/0.10），每档 64 个 paired groups，总计 768 条候选记录。

| 任务 | 候选方向 | mean latent Δcost | mean physical Δcost | paired improvement rate |
|---|---|---:|---:|---:|
| Reacher | negative latent gradient | 0.089661 | 0.051350 | 0.468750 |
| Reacher | random direction | 0.007782 | 0.011182 | 0.416667 |
| Push-T | negative latent gradient | 0.020634 | −4.593762 | 0.526042 |
| Push-T | random direction | 0.002257 | 0.197666 | 0.390625 |

- Reacher：负 latent-cost gradient 没有带来平均物理 cost 改善。
- Push-T：负 latent-cost gradient 物理 cost 平均改善约 −4.59，paired improvement rate 52.6%，优于随机方向。

### 6.2 E2：生成方法对照（dev 50 episodes）

| 方法 | Reacher | Push-T |
|---|---:|---:|
| A | 78% | 94% |
| A+B | 82% | 82% |
| Post-opt | **84%** | 96% |
| Guided-flow | 82% | **98%** |
| B-CEM | 78% | 94% |
| A-CEM | 70% | 94% |

用户最终选择：Reacher 使用 `post_opt`；Push-T 使用 `guided_flow`。

---

## 7. Online Supervision：T0–T6 fixed-replay 训练

固定 replay：

| 任务 | guidance | continuous 实际步数 | grounded 实际步数 | fixed rows |
|---|---|---:|---:|---:|
| Reacher | post-opt | 16,000 | 3,856 | 1,832 |
| Push-T | guided-flow | 16,000 | 3,628 | 947 |

T0–T6 共 14 个训练臂全部完成：200 optimizer updates；`0,10,...,200` 共 21 个 checkpoint；294 个 aggregate final evaluation；每个评测点 200 episodes；全部 `status=ok`。

### 7.1 update-200 final success rate

| 任务 | T0 frozen | T1 offline-B | T2 online-B | T3 ranking | T4 offline-A | T5 hindsight-A | T6 distill-A |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reacher | 87.5% | 83.0% | 77.5% | 77.5% | 70.5% | 76.0% | 79.5% |
| Push-T | 89.0% | 85.0% | 81.5% | 81.5% | 81.5% | 80.5% | 80.5% |

结论：

- 没有任何新增监督臂超过 T0 frozen baseline。
- T1 是两个任务中表现最好的训练臂，但仍低于冻结模型。
- T3 的 `ranking_pairs=0`，ranking weight 被置为 0，T3 实际退化为 T2，不能视为 ranking supervision 已验证有效。
- 尚未启动这些 arm 的 closed-loop 20k 环境步复验，也未延长到 500/1000 updates。

### 7.2 监督信号核验

- 两任务 T3 的 200 updates `ranking_pairs` 累计均为 0，calibration 报告 auxiliary gradient norm=0、rank weight=0，解释了 T3 与 T2 曲线相同。
- T5 hindsight-A 与 T6 distill-A 各累计 6,400 个实际参与训练的辅助样本。
- 非 T0 update 的 B-MSE batch 为 64 条（32 offline + 24 continuous + 8 grounded）。

---

## 8. 总体结论与未决事项

### 8.1 已确立的结论

1. 新协议保留高层结论，但绝对基线与部分局部结论已变化，旧报告具体百分点不能直接沿用。
2. 没有普遍 action-supervision planner gain；单纯增大 E1 capacity 不能解决 gap；目标条件控制有效；E6 task-dependent；serial one-step 不能全局默认。
3. 当前在线后训练配置没有稳定跨任务收益。
4. `physical_time_type` 无跨任务统一收益，退回探索性分支。
5. fixed-replay 上新增监督信号（B-MSE / ranking / offline-A / hindsight-A / distill-A）均未超过冻结基线。

### 8.2 未决事项（待用户决定）

- 是否把 T3 `ranking_pairs=0` 作为需先修复的协议问题，还是在继续前单独修正 ranking loss/replay contract。
- 是否选择 Reacher/Push-T 的 T3/T5/T6 中任一 arm 进入 20k-env-step closed-loop 复验。
- 是否将固定 replay 延长到 500/1000 optimizer updates。
- 是否另行启动 R4-AB（不会自动启动）。

---

## 9. 归档文件索引

| 阶段 | 报告 | 位置 |
|---|---|---|
| 总报告 | Round 3 实验总报告（本文件） | `docs/report/round3/round3_experiment_report.md` |
| Phase 1 | 新协议复核总报告 | [`phase1/round3_new_protocol_experiment_report.md`](phase1/round3_new_protocol_experiment_report.md) |
| Phase 1 | Canonical Phase 1 报告 | [`phase1/phase1_report.md`](phase1/phase1_report.md) |
| Phase 1 | 扩展评测报告 | [`phase1/extended_eval_report.md`](phase1/extended_eval_report.md) |
| Phase 1 | 失败案例报告 | [`phase1/failure_cases_report.md`](phase1/failure_cases_report.md) |
| Phase 1.5 | 决策与执行登记 | [`phase1.5/round3_phase1.5_decision.md`](phase1.5/round3_phase1.5_decision.md) |
| Phase 2 | 在线后训练报告 | [`phase2/round3_phase2_experiment_report.md`](phase2/round3_phase2_experiment_report.md) |
| Phase 3 | physical-time/type 报告 | [`phase3/phase3_e5_physical_time_type_report.md`](phase3/phase3_e5_physical_time_type_report.md) |
| Online Supervision | 总报告 | [`online_supervision/round3_online_supervision_experiment_report.md`](online_supervision/round3_online_supervision_experiment_report.md) |

原始产物（CSV、JSON、checkpoint、视频、曲线图）仍位于 `outputs/round3/`，本目录只归档 Markdown 报告。Online Supervision 的 fixed replay 与训练结果目录通过 symlink 指向 `/tmp/round3_online_supervision/`。
