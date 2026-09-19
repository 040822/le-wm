# Round 5 Phase 3：新增任务上的 FastLeWAM 与基线对比

  ## 概要

  Phase 3 只研究三个新增任务，不重跑原有四任务：

  - OGBench-Scene：data/datasets/ogbench/scene.h5
  - DMC-Finger：data/datasets/dmcontrol/finger_turn_hard.h5
  - DMC-Humanoid：data/datasets/dmcontrol/humanoid_walk.h5

  实验分为两步：

  1. 按 Round 5 Phase 1 的 R4-AB 训练和评测设置训练 FastLeWAM，并运行完整 P0–P3、flow-step、
     guidance 矩阵。

  2. 独立训练 LeWM 和 LeFlow，并在完全相同的任务、数据和评测 cohort 上评测。

  所有方法暂用训练 seed 3072。该阶段属于单种子探索实验，不将结果表述为对 DeWM 论文 (docs/
  paper/AAAI2027_Flow_JEPA.pdf) 六随机种子结果的严格复现。

  ## 公共协议与环境准备

  - 三个任务统一使用论文评测协议：评测 seed 42、50 个 start–goal 对、goal offset 25、执行预算
    50、规划 horizon 5、receding horizon 5、action block 5。

  - 按 legacy 算法一次性生成并冻结每个任务的 cohort，包括 episode、start step、goal step、全
    局数据索引和 cohort hash；所有方法和条件必须复用同一 cohort。

  - 本仓库当前环境缺少论文版本中的部分成功判据和目标恢复接口。实现时建立仓库内受控的兼容层，
    不依赖学长个人虚拟环境：
      - Scene：恢复 cube、两个 button、drawer、window 的目标；cube、drawer、window 误差分别不
        超过 0.04，button 状态精确匹配，五项同时满足才成功。

      - Finger-turn_hard：增加 set_target_position 等价接口，使用 goal row 的
        target_position；DMC reward ≥ 1.0 时成功。

      - Humanoid-walk：head height ≥ 1.4、torso upright ≥ 0.9、平面质心速度 ≥ 1.0 时成功。

  - 成功率定义为 50 个 episode 中，在执行预算内任一时刻满足成功条件的比例。DMC 数据中的 NaN
    success 字段不得作为指标来源。

  - Scene 同时按数据中的 privileged_target_task 输出 cube、button、drawer、window 分组结果；
    分组不改变 all-component 主成功判据。

  - GPU 仅使用 GPU0–3。每次启动前检查显存并显式设置 CUDA_VISIBLE_DEVICES。

  ## Step 1：FastLeWAM

  - 三个任务分别从头训练 R4-AB：
      - train_mode=stage_ab
      - seed 3072，10 epochs，bf16，batch size 128
      - AdamW，learning rate 5e-5，weight decay 1e-3
      - image size 224，embed dim 192，history size 1，horizon 5
      - 90/10 train split，并沿用 Phase 1 的其余数据增强、checkpoint 和优化设置

  - 每个任务评测 61 个条件：
      - 无 guidance：P0、P2、P3 分别运行 flow step {1,2,5,10,16,32}，P1 运行其 invariant 条
        件，共 19 个。

      - guidance：P0、P2 分别运行 guided_flow、post_opt × 6 个 flow step；P3 运行
        guided_flow、post_opt、post_opt_refine × 6 个 flow step，共 42 个。

      - guidance 参数固定为 step_size=0.01、late_steps=5、inner_steps=5、max_rms_offset=0.2；
        积分器使用 Euler；P3 使用 64 个候选。

      - P2 对 DMC-Finger 和 DMC-Humanoid使用 cem-clip，Scene 使用 legacy CEM；P0/P3 不使用
        CEM clipping。

  - 三任务合计 183 个 FastLeWAM 评测条件，每个条件 50 episodes。
  - FastLeWAM 主结果固定为 P3、flow step 1、无 guidance；其余条件用于分析模块贡献、flow-step
    敏感性和 guidance 收益。

  ## Step 2：LeWM 与 LeFlow

  - 不复用学长已有 checkpoint 或实验结果，在三个任务上独立训练。
  - 共享预算固定为 seed 3072、10 epochs、相同训练数据、90/10 split、图像尺寸和数据预处理；
    history、表示维度等结构参数使用各方法在本仓库中的标准配置。

  - LeWM：
      - 使用标准 LeWM 训练设置。
      - 评测采用 CEM：300 candidates、top-30、30 iterations、初始方差 1。
      - 每个任务只运行一个标准推理条件。

  - LeFlow：
      - 使用对应任务中新训练的 LeWM checkpoint 作为冻结基础模型，再按本仓库标准 LeFlow 流程训
        练 10 epochs。

      - 标准推理使用 64 paths 和 16 flow steps。
      - 每个任务只运行一个标准推理条件。

  - Step 2 共增加 6 个评测结果；Phase 3 总计 189 个方法–条件–任务结果。

  ## 验证、报告与验收

  - 在批量实验前完成环境级测试：
      - Scene 分别检查各组件阈值、五项 conjunction 以及目标恢复。
      - Finger 检查目标位置恢复和 reward 1.0 边界。
      - Humanoid 检查三个阈值的边界组合。
      - 每个任务先运行一个 episode 的端到端 smoke test。

  - 每个结果必须记录 checkpoint、训练配置、训练 seed、评测 seed、cohort hash、episode 成功向
    量、成功数、成功率和运行状态；Scene 额外记录四个 target_task 分组。

  - 主表比较 FastLeWAM P3/step1/no-guidance、LeWM、LeFlow。报告成功数、成功率、Wilson 95% 区
    间，以及同 cohort 的 paired improved/regressed 数和 McNemar 检验。

  - 辅助分析依次报告 P0–P3 差异、flow-step 敏感性、guidance 相对无 guidance 的变化。只呈现数
    据和统计量，最终方法结论由用户决定。

  - 验收条件：
      - 三个训练任务均产生 epoch 10 checkpoint，配置和 seed 可追溯。
      - 183 个 FastLeWAM 条件和 6 个基线条件全部完成，每项正好 50 episodes。
      - 所有方法的 cohort hash 完全一致，无 NaN 成功指标或遗漏 episode。
      - DMC 成功率来自环境 termination，Scene 主结果采用论文 all-component 判据。

  - 将本方案替换 docs/plan/round5_experiment_plan.md 中的 Phase 3 占位；执行产物统一写入
    outputs/round5/phase3，最终报告写入 docs/report/round5/
    round5_phase3_report.md。