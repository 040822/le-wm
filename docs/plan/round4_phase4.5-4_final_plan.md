# Round 4 Phase 4.5-4：R4-AB 旧协议对照实验

  ## Summary

  使用 R4-AB seed3072 的 epoch10 checkpoint，运行 4 个 task、两个 50-episode cohort：

  - 旧协议：legacy_50.json
  - Dev：复用已有 round3_revised dev 结果，不重跑

  每个 cohort 共 96 个条件（flow step 网格扩展为 1/2/5/10/16/32，Euler，step16 为参考点）：

  - 全任务：P0 六个 step、P1/legacy、P2/legacy 六个 step、P3 六个 step，共 19 个/task；
  - Reacher 追加 P1/cem-clip、P2/cem-clip 六个 step，以及 P1/cem-scale、P2/cem-scale 六个 step，共 33 个；
  - TwoRoom 追加 P2/cem-clip 六个 step，共 25 个。

  不运行 Heun、final 集或 LeWM，不使用 goal。本轮在 legacy cohort 上补测 flow step 的完整影响，dev 侧由 phase45_3 提供同一网格。

  ## Implementation Changes

  - 新增 Phase 4.5-4 配置和 runner，输出到独立目录：
    outputs/round4/phase45_4_seed3072_legacy_dev/。

  - 在 source/common/round4_eval.py 增加显式 allowed_protocol_variants 参数：
      - 默认只允许 round3_revised，保持旧入口行为；
      - Phase 4.5-4 明确允许 legacy 和 round3_revised；
      - 结果仍原样保存真实 protocol_variant、cohort ID 和 SHA256。

  - 新 runner 读取 config/round4/phase45_4.json，只执行缺失的 legacy 条件；已有 dev 结果直接引用 outputs/round4/phase45_3_seed3072_dev。
  - 每个 task 一个 worker、一个物理 GPU；启动前检查 VRAM 和占用情况。优先 GPU0–3，必要时使用用户已授权的 GPU4–7；每条命令显式设置 CUDA_VISIBLE_DEVICES。

  ## Analysis and Report

  生成新报告：

  docs/report/round4/round4_phase45-4_legacy_protocol_comparison_report.md

  报告包含：

  - 旧协议与 dev 的完整条件表；
  - 按 task 拆分的 step、mode/protocol 子表；
  - 同一 cohort 内每个 step 相对 step16、以及 cem-scale vs legacy / cem-clip 的配对比较；
  - 旧协议与 dev 的成功率、Wilson 区间及绝对差异；
  - 明确禁止对不同 cohort 做 paired McNemar 检验；
  - checkpoint、cohort hash、结果路径和 50 episode 完整性；
  - 结论重点判断：旧协议是否改变 step、mode 或 protocol 的方向性结论。

  旧有 Phase 4.5-3 报告及 without_heun 报告保持不变。

  ## Test and Acceptance

  - 验证 96 个条件矩阵无遗漏、无 Heun、flow step 均在预注册网格内；
  - 验证 legacy manifest、checkpoint、seed、goal offset 和 episode 数量；
  - 验证 dev 复用结果均为 status=ok 且每项 50 episodes；
  - 验证每个 legacy 结果的 protocol_variant=legacy；
  - 验证同 cohort 内 episode identity 一致；
  - 验证 4 个 task 的 legacy 结果全部完成后再生成报告；
  - 验证报告同时列出旧协议和 dev，且不覆盖既有报告。

  ## Assumptions

  - “旧协议”采用 canonical legacy_50.json；它与 dev_legacy.json 的 50 个 entries 相同，但 cohort identity 不同。
  - “dev”采用已有 phase45_3_seed3072_dev 的 round3_revised 结果。
  - P1/legacy 的 step 固定为 invariant；P2 的 step 指 actor warm-start action flow steps。
  - success rate 是主指标，planning latency 仅作辅助信息并注明首调用热身影响。