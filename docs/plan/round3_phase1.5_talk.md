# Round 3 Phase 1.5 决策记录

本文件记录 Phase 1 完成后的正式用户决策；详细的证据、路径和未完成项见
`docs/report/round3/phase1.5/round3_phase1.5_decision.md`。

- 全局 Fast 基线选择 **E5**。共同 canonical dev Stage-B 的 E5 相对 E3
  在 Cube、Push-T、Reacher 上分别为 `+6、+8、0 pp`，平均约 `+4.67 pp`。
  原计划的 5 pp 只是筛选门槛；用户确认 4.67 pp 已足够接近，接受 E5。
- Phase 1.5 只使用现有 E5 legacy checkpoint 作为 Fast 起点。
- `physical_time_type` 延后到 Phase 3 做候选探索，Stage-B timestep 保持
  `legacy`。该探索已完成但没有跨任务统一 Stage-B 收益；后续主线继续使用历史
  E5 legacy checkpoint，physical-time/type 只作历史证据。
- 不采用 capacity-384、serial one-step 或 clean-action timestep 作为主线。
- 补充训练的 TwoRoom E3 纳入 Phase 1.5 分析，但保留 supplemental 身份：
  已有 `round3_revised` final Stage-B=90.5%，尚未有 canonical dev 结果，
  不回写 Phase 1 canonical matrix，也不伪装成 dev 结果。
- Round 4 当前运行继续完成。Round 3 与 Round 4 并行推进，但新的 Round 3
  GPU 训练/EGL 评测须等待 GPU0–3 有明确资源窗口。
