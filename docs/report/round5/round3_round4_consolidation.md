# Round 3 + Round 4 结果整合（Round 5 Phase 1 交付）

本文件把 Round 3、Round 4 与 Round 5 Phase 1 的结论合并，作为最终主实验方案
的证据基础。所有数字均来自对应报告；本文件不新增实验。

## 1. 评测协议

| 项 | 结论 |
|---|---|
| 旧协议（`legacy`） | `legacy_50.json` 是 `source.common.eval.select_eval_cohort`（`LEWM_EVAL_PROTOCOL="lewm_upstream_v1"`）的精确复刻：保留 global-last-row exclusion、不去重、不过滤初始成功；seed 42、goal offset 25、50 episodes。评测配置与 LeWM 上游一致（budget 50、horizon/receding/action_block 5、CEM 300/30/30/var 1.0）。**用于与 LeWM 论文可比的主实验。** |
| 修订协议（`round3_revised`） | 剔除初始成功、按物理距离分层、每原始 episode 至多一个起点，200-episode final。用于更严格的内部对照。 |
| 成功判据 | 四个任务与安装版 `stable-worldmodel` 运行时一致（`round3_runtime_predicates_v1`）。 |

## 2. Round 3 结论

- **Phase 1 / 1.5**：冻结四任务 predicate 与 cohort；选定 legacy E5 为 Fast 起点
  （dev 上相对 E3 平均 +4.67 pp，接近门槛并接受）。
- **Phase 2（E0、legacy E5 online）**：Cube 20k env steps 下 Freeze 48%、
  Offline 48%、Online 44%（−4 pp）；所有任务/臂均未通过扩展门，online 主线停止。
- **Online Supervision（Reacher/Push-T，fixed replay，T0–T6，200 updates）**：
  update-200 上没有任何新增监督臂超过 T0 冻结 baseline（Reacher T0 87.5%，最好
  训练臂 T1 83.0%；Push-T T0 89.0%，最好 T1 85.0%）。T3 ranking 因
  `ranking_pairs=0` 未生效。
- **guidance E1/E2（单候选）**：Reacher 负梯度与物理改善一致性弱；Push-T 较好。
  E2 成功率：Reacher post-opt 0.84 > A+B 0.82/A 0.78；Push-T guided-flow 0.98 >
  post-opt 0.96 > A 0.94 > A+B 0.82。用户据此选 Reacher=post_opt、
  Push-T=guided_flow。

**Round 3 净结论**：online train 在本预算/配置下未见收益；guidance 在单候选下
任务相关、值得进一步与最佳 planner 结合。

## 3. Round 4 结论

- **R4-AB vs R4-ABDE**：final P3 宏平均 95.0 vs 95.0；D/E 辅助训练未显示收益。
  支持用更简洁的 **R4-AB** 作为 P3 训练方案（单 seed，非因果结论）。
- **P3 vs P4**：final P4−P3 = Cube 0、Push-T −7、Reacher −7、TwoRoom +0.5 pp；
  P4 未显示跨任务收益。**P3 是更稳妥的低延迟 learned planner。**
- **Flow step（R4-AB）**：final 宏平均 P3 step1 95.625、step2 95.375、step16
  95.0；P2 step1 明显优于 step16（Reacher 54/18 改进，p≈2.6e-5）。低步数不劣于
  16 步，且更省时。**推荐 step 1/2。**
- **旧协议对照**：legacy 与 dev 的 step/mode 方向性一致；Reacher 的 P1/P2
  `cem-clip` 优于 legacy（P1 +20 pp，P2 各 step 多为正）。**Reacher 的 P2 采用
  cem-clip。**
- **速度**：best-of-64 P3/P4 规划中位 0.36–0.92 s，比 P1/P2 CEM 快约 5.6–18.9 倍。

## 4. Round 5 Phase 1（本轮）

在 R4-AB + legacy 50 上新增 guidance 与 P3 结合条件，见
`round5_phase1_report.md` 与 `round5_final_scheme_decision.md`：

- 168 条新条件全部 `status=ok`；与 76 条基线配对（含复用的 P1：reacher
  `cem-clip` 92%、其余任务 legacy 70/90/100%）。
- Reacher 是唯一稳定受益任务（P3+post_opt 宏平均 86.7 vs 无 guidance 82.7；
  P0 72→88%）；且无 guidance 的 P0/P3 在 reacher step1 明显低于 P1（−18~−20 pp），
  加入 guidance 后才追平 P1。其余任务接近饱和，guidance 中性或学习 planner 已优于 P1。
- P3 三种 guidance 语义均可用；`post_opt_refine`（先选后精修）成本最低且稳定。
- **最终方案**：P3 + step 2（step 1 备选）；guidance 作为按任务可选增益；
  训练用 R4-AB；评测用 legacy 旧协议。

## 5. 尚未解决 / 下一阶段

- **Online train 是否提升 P3**：Round 3 的 online 方案评测在 CEM(stage_b) 上，
  未在 P3 上验证；列为 Round 5 Phase 2，从 R4-AB 出发、以 P3 评测、复用
  T-arm fixed replay（四任务，200 updates），但 T3 ranking 需先修协议。
- **任务扩展**：AAAI DeWM 任务 + 扩展任务、LeWM 过饱和检查、episode 50→200，
  列为 Round 5 Phase 3（占位）。
- **稳定性**：单训练 seed；D/E 与 P4 未纳入主方案。

## 6. 证据索引

- Round 3：`docs/report/round3/`（phase1、phase1.5、phase2、phase3、
  online_supervision）。
- Round 4：`docs/report/round4/`（epoch10、ab_vs_abde、phase45-3、phase45-4、
  ab_fm_step_ablation）。
- Round 5 Phase 1：`docs/report/round5/round5_phase1_report.md`、
  `docs/report/round5/round5_final_scheme_decision.md`。
