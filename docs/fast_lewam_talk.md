1. check model设计，尤其是attention mask的合理性（stage A mask z0=》a0~aH-1 ，stage A是否要添加zg， stage B的大causal mask真的能建模zt+at=>zt+1的转移过程吗？——
2. check train设计，目前是简单的 stage A => stage B, 能不能更复杂一些，比如 stage B=>stage A或者stage A => stage B => stage A，或者训练的时候加入stage C。
3. 如何证明stage A 对 stage B的增益？
4. 当前benchmark 实际上是不能证明stage A的有效性的，因为目前的目标只是goal，而没有一个明确的task goal，因此目前没有goal latent的stage A是理论不能完成的。
5. 可视化训练数据，看一下训练数据到底是在训什么，尤其是goal相关数据。
6. timestep embedding在stage B的train和eval中应该怎么定。
7. 如何提升CEM solver的性能？

• 我读完了 docs/LEDP.md，也核对了当前 Fast-LeWAM 的训练和评测实现。我的总体判断是：

  > 这个方向值得继续，而且 Stage A 已经显示出很强的可行性；但目前结果只能证明“统一模型能同时学出不错的 action policy 和可用的 world model”，还不能证
  > 明“Action Head 反向增强了 World Model”，也不能与论文表格直接做 SOTA 比较。

  ## 一、如何解读当前结果

  10 epoch 的四任务平均成功率为：

   模式                     平均成功率    主要含义
  ━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Stage A                       92.0%    goal-conditioned 动作生成
  ───────────────────────  ────────────  ────────────────────────────
   Stage B                       84.5%    World Model + CEM planning
  ───────────────────────  ────────────  ────────────────────────────
   Stage C                       28.0%    单次 causal pass 联合生成
  ───────────────────────  ────────────  ────────────────────────────
   Stage A shuffled goal         25.0%    错误目标下的动作生成

  ### 1. Stage A 是目前最强、最稳定的结果

  Stage A 达到 96/76/96/100，说明：

  - 共享 DiT 可以学会完整 action chunk；
  - goal latent 对动作生成确实有用；
  - Fast-LeWAM 至少已经是一个很强的 goal-conditioned policy。

  shuffled goal 后下降到 38/10/12/40，平均下降约 67 个百分点。这是一个很好的信号，说明 Stage A 并不是简单记忆任务先验，而是在真正读取目标。

  但 Stage A 本质上更接近 goal-conditioned behavior cloning/flow policy，不是 planner。它高于 Stage B 并不表示 World Model 没用，因为两者解决问题的方式不
  同。

  ### 2. Stage B 可用，但没有表现出稳定提升

  相对 LeWM 论文结果，Stage B 的变化为：

  - Two Room：87 → 100，提升明显
  - Reacher：86 → 84，基本持平
  - Push-T：96 → 84，明显下降
  - Cube：74 → 70，下降

  整体平均几乎与 LeWM 论文持平。因此目前不能声称 Action Head 增强了 World Model。更准确的结论是：

  > 联合训练没有破坏所有任务上的 planning 能力，但正迁移具有明显任务依赖性。

  Push-T 和 Cube 正好是下一步最值得诊断的任务：它们的 Stage A 很强，而 Stage B 明显更弱，可以帮助判断问题究竟来自 world model、CEM，还是联合训练冲突。

  ### 3. Stage C 的失败不意外，建议暂时降级

  当前 Stage C：

  - 不接收 goal token；
  - action token 受到 causal mask 限制；
  - 推理最终只执行动作，没有利用预测 latent 做选择。

  因此它和 goal-conditioned Stage A 不是公平对照，也不是真正的“Stage A+B 单次融合”。其 28% 左右的结果更像无目标行为先验。建议把 Stage C 保留为负结果或结构消
  融，不要继续投入主要算力。

  ### 4. 5→10 epoch 表明部分任务尚未收敛

  Stage A：

  - Reacher：46 → 76
  - Push-T：86 → 96

  Stage B：

  - Reacher：66 → 84
  - Two Room：96 → 100

  说明 10 epoch 对 Reacher 尤其可能还不够。但不建议马上只把当前配置训练到 50/100 epoch，因为首先需要解决因果消融和评测公平性，否则只是把不确定性训练得更久。

  ### 5. 目前的数值统计强度不够

  每项只有 50 个 eval episode，所以成功率最小变化单位是 2%。单次训练、单个 eval seed 下：

  - 2–8 个百分点的差异通常不足以下结论；
  - 10 个百分点左右也需要配对结果或多个训练 seed；
  - Two Room、Cube 的高成功率还存在饱和效应。

  最终至少需要 3 个训练 seed，并保存逐 episode 成败，用相同初始状态做 paired bootstrap 或配对统计。

  ## 二、目前最大的科学问题

  你的核心贡献需要拆成两个方向分别验证：

  1. Action supervision 是否增强 World Model？
  2. World Model supervision 是否增强 Action Head？

  当前模型同时加入了共享参数、action loss、latent loss、预测动作混合和跨 head 梯度，缺少消融，所以无法知道提升来自哪里。

  还有一个更重要的问题：当前 Stage B 最多使用约 50% 的 Stage A 预测动作，但仍用专家轨迹的 future latent 作为监督目标。如果预测动作偏离专家动作，那么这个
  future latent 严格来说不是该动作真正造成的未来。这可能会迫使 Stage B 忽略动作差异，损害 CEM 所需要的候选动作排序能力。

  因此不要直接把“加入更多随机/预测动作”当作解决分布偏移的方法。离线数据没有对应的反事实未来；真正正确的解决方式是：

  - planning 时让 CEM 候选靠近 Stage A 的数据分布；
  - 或在线执行动作并收集其真实后继状态，再更新 World Model。

  ## 三、下一步最应该做什么

  建议先在 Push-T 和 Cube 上运行下面这组最小消融，固定数据、全局 batch size、训练步数、模型宽度和评测初始状态：

   实验    训练方式                                             回答的问题
  ━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   E0      本地复现原始 LeWM                                    同协议 planning 基线
  ──────  ───────────────────────────────────────────────────  ─────────────────────────────────
   E1      Parallel Stage B only，只有真实动作和 latent loss    并行化本身是否有效
  ──────  ───────────────────────────────────────────────────  ─────────────────────────────────
   E2      Stage A only，lambda_latent=0                        纯动作模型基线
  ──────  ───────────────────────────────────────────────────  ─────────────────────────────────
   E3      A+B 共享，但 Stage B 始终使用真实动作                action 辅助任务是否增强 WM
  ──────  ───────────────────────────────────────────────────  ─────────────────────────────────
   E4      A+B，50% 预测动作，detach=true                       exposure 是否帮助 WM
  ──────  ───────────────────────────────────────────────────  ─────────────────────────────────
   E5      当前配置，50% 预测动作，detach=false                 latent 梯度回传是否产生双向增益
  ──────  ───────────────────────────────────────────────────  ─────────────────────────────────
   E6      E5 + Stage A warm-start CEM                          actor 与 planner 是否真正协同

  ### 当前 E0–E6 消融结果（2026-08-10）

  统一口径：表中数值均为 50 个评估 episode 的成功率（%）。`mean ± SD` 中的 SD 是训练种子间样本标准差。当前资源策略已改为优先单种子覆盖多任务。

  结果完整性如下：

  - Cube 的 E0–E5 已完成；E1/E3 有 3 个训练种子。
  - Push-T 的 E0–E5 已完成；E1/E3/E4/E5 有 3 个训练种子。
  - Reacher 的 E0–E5 seed 3072 已全部完成；E1–E4 均有 epoch 10 评估和 `last.ckpt`。
  - TwoRoom 目前只有 E0、E5；E1–E4 尚未运行。
  - E6 已完成四任务 seed 3072 评估；它复用 E5 epoch 10 checkpoint，不需要重新训练。

  #### Cube：seed 3072 完整轨迹

  | 实验 | 分支 / 评估设置 | e2 | e4 | e6 | e8 | e10 / final |
  |---|---|---:|---:|---:|---:|---:|
  | E0 | LeWM baseline | — | — | — | — | **78** |
  | E1 | Stage-B | 50 | 60 | 58 | 60 | 60 |
  | E2 | Stage-A | 98 | 100 | 100 | 100 | 100 |
  | E2 | Stage-A, shuffled goal | 44 | 42 | 44 | 40 | 40 |
  | E3 | Stage-A | 100 | 100 | 100 | 100 | 100 |
  | E3 | Stage-A, shuffled goal | 42 | 42 | 40 | 42 | 42 |
  | E3 | Stage-B | 68 | 74 | 74 | 70 | **74** |
  | E4 | Stage-A | 98 | 100 | 100 | 100 | 100 |
  | E4 | Stage-A, shuffled goal | 42 | 42 | 40 | 40 | 40 |
  | E4 | Stage-B | 74 | 76 | 74 | 74 | **74** |
  | E5 | Stage-A | 100 | 100 | 100 | 100 | 100 |
  | E5 | Stage-A, shuffled goal | 42 | 42 | 42 | 42 | 42 |
  | E5 | Stage-B | 72 | 72 | 72 | 74 | **66** |

  #### Cube：E1 与 E3 的配对多种子结果

  | 训练种子 | E1 Stage-B final | E3 Stage-B final | E3 − E1 |
  |---:|---:|---:|---:|
  | 3072 | 60 | 74 | +14 |
  | 3073 | 58 | 74 | +16 |
  | 3074 | 62 | 66 | +4 |
  | **mean ± SD** | **60.0 ± 2.0** | **71.3 ± 4.6** | **+11.3 ± 6.4** |

  三个配对差值都为正，说明 Cube 上的 Stage-A auxiliary supervision 能稳定改善 Fast Stage-B。不过 E0 为 78，仍高于 E3 的均值和每个训练种子，因此不能声称 Fast 超过原 LeWM。

  E3 的 Stage-A final 为 `100.0 ± 0.0`，shuffled-goal 为 `41.3 ± 1.2`。Action Head 明确使用 goal，但约 40% 的残余成功率也说明 Cube 存在较强任务先验。

  #### Push-T：seed 3072 完整轨迹

  | 实验 | 分支 / 评估设置 | e2 | e4 | e6 | e8 | e10 / final |
  |---|---|---:|---:|---:|---:|---:|
  | E0 | LeWM baseline | — | — | — | — | **98** |
  | E1 | Stage-B | 76 | 82 | 86 | 86 | 88 |
  | E2 | Stage-A | 56 | 80 | 86 | 92 | 92 |
  | E2 | Stage-A, shuffled goal | 10 | 8 | 8 | 8 | 8 |
  | E3 | Stage-A | 66 | 88 | 92 | 94 | 96 |
  | E3 | Stage-A, shuffled goal | 4 | 6 | 4 | 6 | 4 |
  | E3 | Stage-B | 52 | 78 | 82 | 90 | **90** |
  | E4 | Stage-A | 60 | 86 | 88 | 92 | 90 |
  | E4 | Stage-A, shuffled goal | 6 | 6 | 8 | 10 | 10 |
  | E4 | Stage-B | 46 | 64 | 86 | 92 | **94** |
  | E5 | Stage-A | 74 | 86 | 94 | 98 | 98 |
  | E5 | Stage-A, shuffled goal | 8 | 8 | 8 | 12 | 10 |
  | E5 | Stage-B | 72 | 84 | 86 | 84 | **86** |

  #### Push-T：E1 与 E3 的配对多种子结果

  | 训练种子 | E1 Stage-B final | E3 Stage-B final | E3 − E1 |
  |---:|---:|---:|---:|
  | 3072 | 88 | 90 | +2 |
  | 3073 | 90 | 86 | −4 |
  | 3074 | 94 | 88 | −6 |
  | **mean ± SD** | **90.7 ± 3.1** | **88.0 ± 2.0** | **−2.7 ± 4.2** |

  三个差值中只有一个为正。Push-T 不仅没有复现 Cube 的正迁移，E3 的平均 final 还低于 E1 2.7 个百分点；当前不支持跨任务的 “Action Head 增强 WM” 主张。

  E3 的 Stage-A final 为 `95.3 ± 1.2`，shuffled-goal 为 `6.0 ± 2.0`。Push-T 的 goal dependence 很强，但强 actor 没有转化为更强的随机初始化 CEM planner。

  #### Push-T：E4 与 E5 的配对多种子结果

  | seed | E4 Stage-A | E5 Stage-A | E5−E4 | E4 Stage-B | E5 Stage-B | E5−E4 |
  |---:|---:|---:|---:|---:|---:|---:|
  | 3072 | 90 | 98 | +8 | 94 | 86 | −8 |
  | 3073 | 92 | 94 | +2 | 88 | 90 | +2 |
  | 3074 | 92 | 96 | +4 | 84 | 88 | +4 |
  | **mean ± SD** | **91.3 ± 1.2** | **96.0 ± 2.0** | **+4.7 ± 3.1** | **88.7 ± 5.0** | **88.0 ± 2.0** | **−0.7 ± 6.4** |

  E5 对 Stage-A 的提升在三个种子上都为正，平均为 4.7 点。Stage-B 差值则一负两正、均值接近 0，因此跨-head梯度改善 actor 的证据强于改善 planner 的证据。

  E0 的 final 为 98，高于 E1/E3/E4/E5 的 Stage-B 三种子均值。Push-T 的核心缺口仍是 Stage-B planning，而不是 Stage-A 动作生成。

  #### Reacher：seed 3072 完整轨迹

  | 实验 | 分支 / 评估设置 | e2 | e4 | e6 | e8 | e10 / final |
  |---|---|---:|---:|---:|---:|---:|
  | E0 | LeWM baseline | — | — | — | — | **72** |
  | E1 | Stage-B | 38 | 50 | 82 | 72 | **82** |
  | E2 | Stage-A | 52 | 62 | 70 | 58 | **68** |
  | E2 | Stage-A, shuffled goal | 10 | 8 | 6 | 6 | **8** |
  | E3 | Stage-A | 32 | 42 | 78 | 64 | **76** |
  | E3 | Stage-A, shuffled goal | 12 | 8 | 6 | 4 | **4** |
  | E3 | Stage-B | 36 | 40 | 88 | 82 | **82** |
  | E4 | Stage-A | 30 | 64 | 60 | 74 | **70** |
  | E4 | Stage-A, shuffled goal | 12 | 10 | 4 | 6 | **10** |
  | E4 | Stage-B | 20 | 84 | 76 | 86 | **80** |
  | E5 | Stage-A | — | — | — | — | 76 |
  | E5 | Stage-A, shuffled goal | — | — | — | — | 10 |
  | E5 | Stage-B | — | — | — | — | 84 |

  E1–E4 已正常完成，四个目录均存在 epoch 10 评估和 `last.ckpt`；结果 JSON 均为 `status=ok`，训练日志中没有 traceback、OOM 或任务失败。日志末尾的 `asyncio socket.send()` warning 出现在评估和 checkpoint 已保存之后，属于 W&B/网络连接关闭阶段，不影响实验完整性。

  Reacher 的 Stage-B final 为 E1/E3/E4/E5=`82/82/80/84`，都高于本地同协议 E0=72，但差距只有 8–12 点且目前均为单训练种子。E3 与 E1 final 完全相同，说明 Reacher 不支持“加入 Stage-A auxiliary supervision 必然增强 WM”；E4 相对 E3 下降 2 点，预测动作混合并未显示收益；E5 相对 E4 上升 4 点，只能视为弱正信号。E3 在 e6 达到 88 后回落到 final 82，E4 从 e8 的 86 回落到 final 80，也再次说明不能按测试集峰值选 checkpoint。

  相同 cohort 的 Stage-B 配对翻转进一步表明这些小差异不稳定：E1→E3 为 6 个退化、6 个改善；E3→E4 为 6 个退化、5 个改善；E4→E5 为 5 个退化、7 个改善。Stage-A final 则为 E2/E3/E4/E5=`68/76/70/76`，shuffled-goal=`8/4/10/10`，说明 actor 明确依赖 goal；但各配置间仍有较多双向 episode 翻转，单种子差值不宜外推。

  Reacher E0 的 72 低于此前引用的论文数字 86，因此横向比较应以本地同协议 E0=72 为主，并把论文数字只作为外部参考。

  #### TwoRoom：当前覆盖

  | 实验 | Stage-A | shuffled goal | Stage-B | Stage-C |
  |---|---:|---:|---:|---:|
  | E0 LeWM | — | — | **86** | — |
  | E5 | 96 | 38 | **100** | 60 |

  TwoRoom 的 E1–E4 尚未运行。E5 的 Stage-B 已达到 100%，后续消融主要用于因果归因，而不是继续追求更高成功率。

  #### E6：Actor warm-start CEM

  E6 复用各任务 E5 seed 3072 的 epoch 10 checkpoint。除用 Action Head 生成 CEM 初始均值外，E5 Stage-B 与 E6 使用相同的 50 个 episode、起点、goal offset、随机种子和 CEM budget；Action Head 推理步数为 10。

  | 任务 | E5 Stage-B | E6 warm-start | E6 − E5 | E6 评估耗时（秒） |
  |---|---:|---:|---:|---:|
  | Cube | 66 | **98** | **+32** | 417.6 |
  | Push-T | **86** | 84 | −2 | 317.3 |
  | Reacher | **84** | 68 | −16 | 930.8 |
  | TwoRoom | **100** | **100** | 0 | 1150.0 |

  Actor warm-start 在 Cube 上带来 32 点提升，但在 Push-T 上基本持平、在 Reacher 上下降 16 点。TwoRoom 的 E5 已达到 100%，存在天花板效应。因而 E6 证明 actor proposal 在部分任务上能显著帮助 planner，但当前不能支持跨任务稳定增益。

  相同 cohort 上的逐 episode 配对翻转如下。`E5 成功→E6 失败` 与反方向的翻转数，比成功率差值更能说明变化规模；`exact p` 是只使用不一致配对的双侧精确二项检验。

  | 任务 | 两者都成功 | E5 成功→E6 失败 | E5 失败→E6 成功 | 两者都失败 | exact p |
  |---|---:|---:|---:|---:|---:|
  | Cube | 33 | 0 | 16 | 1 | <0.001 |
  | Push-T | 42 | 1 | 0 | 7 | 1.000 |
  | Reacher | 28 | 14 | 6 | 2 | 0.115 |
  | TwoRoom | 50 | 0 | 0 | 0 | — |

  因此 Push-T 的 −2 实际只来自 1/50 个 episode 的反向翻转，更符合有限样本波动，不能作为系统性退化的证据。Reacher 则有 20 个不一致配对，净损失 8 个 episode，下降信号比 Push-T 实质得多；但在单训练种子、50 个 episode 下，双侧精确检验仍未达到 0.05。进一步核对还发现，Reacher 的 14 个 `E5 成功→E6 失败` episode 中，纯 Stage-A actor 在 12 个上成功，所以“Reacher actor 本身太弱”不能解释 E6 的下降；planner 能把原本可成功的 actor 起点优化坏。

  ##### Simulator-grounded Stage-B 排序诊断（2026-08-09）

  为区分候选覆盖与代价排序问题，在 Push-T 和 Reacher 的 E5 epoch 10 checkpoint 上固定复用 Stage-B cohort 的 slot 0–7。每个初始状态构造 45 个候选：expert、actor、zero、时间反转/平移，以及 16 个零中心随机候选、16 个 actor 邻域候选和 8 个 expert 邻域候选。先用 Stage-B latent cost 排序，再从完全相同的模拟器状态分别执行每个候选一个规划 horizon（5 个 action block，共 25 个原始环境步），用任务物理状态到目标的真实距离和成功判据评价。该实验只诊断第一次规划的局部排序，不等同于完整 50-step E6 rollout。

  | 任务 | 物理代价 Spearman | top-5 recall | 预测 top-1 成功率 | 候选 oracle 成功率 |
  |---|---:|---:|---:|---:|
  | Push-T | 0.756 | 37.5% | 100% | 100% |
  | Reacher | 0.732 | 25.0% | 62.5% | 100% |

  两个任务的全候选 Spearman 接近，因此“Reacher 的全局排序整体崩坏”不成立；差异集中在最优候选附近。Reacher 的候选集合在 8/8 个状态里都包含可成功轨迹，但 Stage-B 只在 5/8 个状态选中成功的 top-1，top-5 与物理真值 top-5 的平均重合率也只有 25%。

  | 任务 / 候选池 | 预测 top-1 成功率 | oracle 成功率 | 池内候选成功率 |
  |---|---:|---:|---:|
  | Push-T / zero-centered | 0% | 0% | 0% |
  | Push-T / actor-centered | 100% | 100% | 19.1% |
  | Push-T / expert-centered | 100% | 100% | 100% |
  | Reacher / zero-centered | 12.5% | 50% | 5.1% |
  | Reacher / actor-centered | 62.5% | 100% | 27.2% |
  | Reacher / expert-centered | 75% | 100% | 80.6% |

  Actor warm-start 的确改善了初始候选覆盖：Reacher 的 actor-centered 池 oracle 成功率从 zero-centered 的 50% 提高到 100%，池内候选成功率从 5.1% 提高到 27.2%。问题不是“actor proposal 没带来好候选”，而是 Stage-B 未能稳定选出已有的好候选。最直接的排序反例是：Reacher 上模型认为 actor 的平均 latent cost `0.0326` 优于 expert 的 `0.1070`，但模拟器真实终点距离反而是 actor `0.0952`、expert `0.0342`，对应单 horizon 成功率分别为 50% 和 100%。这说明模型在靠近数据流形的高质量候选之间存在局部错序。

  Push-T 的首 horizon 诊断中 actor、expert 均为 8/8 成功；即使单独检查完整 E6 中唯一失败的 slot 15，首 horizon 的预测 top-1 仍成功。因此该 −2 无法在第一次 25-step 规划中复现，更可能来自第二次 replan、30 轮 CEM 的后期轨迹或随机采样波动。

  当前最强解释是：actor warm-start 提高了候选覆盖，但 Reacher 的 Stage-B 局部代价不够可靠；CEM 又只返回最后一轮 elite mean、没有保留历史 best，后续迭代可能利用模型误差，把一个本来可成功的 actor 初始化推向失败区域。该结论仍是机制性推断，因为本轮只执行了首 horizon 的静态候选，而没有记录完整 CEM 迭代轨迹。

  下一步无重训诊断应在 Reacher 的失败配对上记录 CEM 第 0/1/2/5/10/20/30 轮的 mean、elite 和预测 cost，并在模拟器中从同一状态执行这些中间解；同时检查第二次 replan。若确认随着迭代次数增加，预测 cost 下降而真实物理代价上升，再比较 early-stop / best-ever 保留，并优先修正 Stage-B 训练时随机 timestep 与评估固定 `t=1` 的错配，以及预测动作仍配专家 future latent 的反事实监督问题。

  原始诊断结果保存在：

  - `outputs/fast_lewam/pusht/0803_e5_cross_head_grad/diagnostics/stage_b_ranking/epoch_10/`
  - `outputs/fast_lewam/reacher/0717_/diagnostics/stage_b_ranking/epoch_10/`

  这组 E6 和排序诊断都只有一个训练种子。它们适合定位机制，不足以估计稳定的跨种子收益。

  #### 当前可支持的结论

  1. **Cube 有稳定的单向正迁移。** E3 相对 E1 三种子平均提升 11.3 点，但仍未超过 E0。
  2. **正迁移不能跨任务泛化。** Push-T 的 E3 相对 E1 三种子平均下降 2.7 点，Reacher 单种子 E3 与 E1 final 持平。
  3. **跨-head梯度主要帮助 actor。** Push-T 的 E5 Stage-A 三种子都高于 E4，但 Stage-B 平均差接近 0。
  4. **Reacher 的本地 Stage-B 有正信号，但消融归因仍弱。** E1/E3/E4/E5 final 都高于本地 E0=72，但 E1=E3、E4 略低、E5 仅高 2–4 点；当前单种子结果不能证明是哪一种联合训练机制带来收益。
  5. **goal conditioning 确实生效。** 正确 goal 与 shuffled goal 差距大，但 Cube/TwoRoom 仍保留较强任务先验。
  6. **Actor 与 planner 的协同具有明显任务依赖。** E6 在 Cube 上大幅提升，在 Push-T 上近似持平，在 Reacher 上明显下降，TwoRoom 则受 100% 天花板限制。
  7. **E6 的主要新瓶颈是顶端排序而不是候选覆盖。** Actor-centered proposal 在 Push-T/Reacher 都提高了首 horizon 的成功候选覆盖；Reacher 下降更符合 Stage-B 对高质量候选的局部错序，以及 CEM 后续迭代利用模型误差。

  #### 统计与解释边界

  每次评估只有 50 个 episode，成功率最小步长为 2 点。训练种子数少，`mean ± SD` 只描述现有训练样本波动，不能当作稳定置信区间。

  Reacher E1–E4 已有 final；其中 E3/E4 的中期峰值高于 final，报告只使用预定 epoch 10 结果，不按测试成功率选择 checkpoint。

  当前优先单种子覆盖多任务。多种子只用于解释冲突或验证关键主张，不再作为补齐第一轮矩阵的默认要求。

  #### 第一轮剩余项与第二轮入口

  1. 第一轮只剩 TwoRoom E1–E4 seed 3072，尚未运行。
  2. E6 的首轮无重训排序诊断已经完成；恢复实验后，优先追踪 Reacher 失败 episode 的完整 CEM 迭代和第二次 replan。
  3. 根据该轨迹诊断再决定是否测试 early-stop / best-ever 保留，或进入 `docs/plan/fast_lewam_stage_b_optimization.md` 中的训练目标修正。

  截至 2026-08-10 检查时，没有 Fast-LeWAM 训练或评估进程在运行；按当前要求暂不启动 TwoRoom 或下一步诊断实验。

  ### 历史快照：E1–E4（2026-08-03）

  > 以下内容保留为当时的阶段记录；当前判断以此前的 2026-08-10 汇总为准。

  以下均为 50 个 eval episodes 上的成功率（%）。E1–E3 已完成 10 epoch；E4 仍在训练，因此 E4 只能作为中期结果。

  #### Cube

  | 实验 | Stage | Epoch 2 | Epoch 4 | Epoch 6 | Epoch 8 | Epoch 10 |
  | --- | --- | ---: | ---: | ---: | ---: | ---: |
  | E1：Stage B only | B | 50 | 60 | 58 | 60 | 60 |
  | E2：Stage A only | A | 98 | 100 | 100 | 100 | 100 |
  |  | shuffled goal | 44 | 42 | 44 | 40 | 40 |
  | E3：A+B，Stage B 使用真实动作 | A | 100 | 100 | 100 | 100 | 100 |
  |  | shuffled goal | 42 | 42 | 40 | 42 | 42 |
  |  | B | 68 | 74 | 74 | 70 | 74 |
  | E4：A+B，预测动作混合，detach | A | 98 | 100 | 100 | — | — |
  |  | shuffled goal | 42 | 42 | 40 | — | — |
  |  | B | 74 | 76 | 74 | — | — |

  Cube E4 已完成到 epoch 6 的评估并继续训练。对应输出目录为 `outputs/fast_lewam/cube/0803_e4_detach`。

  #### Push-T

  | 实验 | Stage | Epoch 2 | Epoch 4 | Epoch 6 | Epoch 8 | Epoch 10 |
  | --- | --- | ---: | ---: | ---: | ---: | ---: |
  | E1：Stage B only | B | 76 | 82 | 86 | 86 | 88 |
  | E2：Stage A only | A | 56 | 80 | 86 | 92 | 92 |
  |  | shuffled goal | 10 | 8 | 8 | 8 | 8 |
  | E3：A+B，Stage B 使用真实动作 | A | 66 | 88 | 92 | 94 | 96 |
  |  | shuffled goal | 4 | 6 | 4 | 6 | 4 |
  |  | B | 52 | 78 | 82 | 90 | 90 |

  Push-T E4 已启动，但尚未到首次 epoch 2 评估。对应输出目录为 `outputs/fast_lewam/pusht/0803_e4_detach`。

  #### 阶段性结论

  - E1 vs E3：Cube 的 Stage B 从 60% 提升到 74%，是目前最强的正迁移信号；Push-T 仅从 88% 提升到 90%，不足以支持稳定增益结论。
  - E2 vs E3：Cube 的 Stage A 最终均为 100%；Push-T 从 92% 提升到 96%，但 4 个百分点仍需多 seed 验证。
  - E3 vs E4：Cube 在 epoch 2/4/6 的 Stage B 分别从 68/74/74 变为 74/76/74，暂未显示预测动作混合带来持续增益。
  - shuffled-goal 最终为 Cube 42%、Push-T 4%，明显低于正确目标下的 Stage A，说明 Action Head 确实依赖目标条件。
  - 当前只能声称 Cube 单 seed 上存在 Action Head 增强 World Model 的迹象，不能泛化为跨任务结论。

  每项评估只有 50 episodes，最小变化单位为 2%。2–8 个百分点的差异应视为弱信号；最终结论至少需要 3 个训练 seed 和配对评估。

  当前 E0 尚未完成 Cube/Push-T 本地同协议基线，E5 和 E6 尚未启动。应先等待 E4 完整结果，再决定是否进入 E5。

  关键比较：

  - E1 vs E3：Action Head 是否增强 World Model
  - E2 vs E3/E5：World Model 是否增强 Action Head
  - E4 vs E5：跨 head 梯度是否有价值
  - E3 vs E4：预测动作混合是提高鲁棒性还是制造错误监督
  - E5 vs E6：统一 WAM 是否在推理阶段产生实际协同

  其中 E6 很可能是最快产生正结果的实验。不要让 CEM 完全从随机动作开始，可以：

  - 一部分候选直接来自 Stage A；
  - 一部分在 Stage A 动作附近加噪声；
  - 少量候选保持全局随机探索。

  这样 Action Head 不只是通过共享权重间接帮助 World Model，还能直接缓解 CEM 的训练—推理分布偏移。

  ## 四、必须补充的诊断指标

  成功率只能告诉你 planner 最终失败了，不能告诉你为什么失败。建议在一组可恢复 simulator state 上采样候选动作，真实执行这些动作，然后比较：

  - predicted latent cost 与真实 goal distance 的 Spearman 相关性；
  - 最优候选 top-k recall；
  - 专家动作相对扰动动作的 preference accuracy；
  - 不同 horizon 的 latent prediction error；
  - CEM 每轮候选质量是否实际提高；
  - Stage B 对动作置换、缩放和局部扰动是否敏感。

  仓库已经有 clean terminal MSE、expert preference、top-1 和 margin 等诊断，可以先用起来。但最终最好加入“在模拟器里真实执行候选动作”的排序校准，因为 latent
  MSE 低不等于适合 planning。

  ## 五、未来研究路线

  ### 阶段 1：建立可信离线结论

  - Cube、Push-T 的 E0–E5 已完成；Push-T 的 E1/E3/E4/E5 三种子结果已齐。
  - Reacher E0/E5 已完成，E1–E4 seed 3072 正在运行。
  - TwoRoom E0/E5 已完成，E1–E4 seed 3072 尚待运行。
  - 单独测量 LeWM 串行 rollout 与 Fast Stage B 的延迟、吞吐、显存和端到端 planning 时间。

  完成后才能决定第一篇工作的主结论是：

  - “Action/World 双向正迁移”，或者
  - “共享模型中的快速策略 + 并行显式规划”。

  当前 E1 vs E3 只在 Cube 上稳定提升，Push-T 尚未复现。因此主张应限定为任务相关的单向正迁移，不能升级为普遍的 “Action Head 增强 WM”。

  ### 阶段 2：做好 actor-guided planning

  比较：

  - Stage A only
  - 随机初始化 CEM
  - Stage A warm-start CEM
  - Stage A 周围局部搜索
  - 不同候选数量下的成功率—延迟曲线

  这一阶段最可能同时体现实时策略和显式 planning，也是当前 idea 中最实际的价值点。

  ### 阶段 3：再进入 online training

  先采用保守版本：

  - 初期冻结视觉 encoder；
  - 只在线更新 dynamics/latent head；
  - 保留 offline replay，混合 offline 与 online 数据；
  - 每次执行后保存真实 (z_t, a_t, z_{t+1})；
  - 监控旧任务成功率和 held-out latent ranking，防止遗忘。

  对照至少包括：

  - 不在线更新
  - 只更新 World Model
  - 联合更新 World Model + Action Head
  - 无 replay 的 naive online fine-tuning

  等 World Model online update 明确有效后，再尝试 Action Head 联合更新和跨任务持续学习。否则三个贡献同时推进，定位失败原因会非常困难。

  ## 最终建议

  现在最优先的不是继续扩大模型或马上做 online learning，而是：

  1. 等待 Reacher E1–E4 完成并汇总 final；
  2. 运行 TwoRoom E1–E4 seed 3072，补齐四任务单种子矩阵；
  3. 分析 E6 的任务差异，检查 actor proposal 与随机初始化的候选覆盖；
  4. 检查 Stage B 的 simulator-grounded 候选排序能力；
  5. 测量 LeWM 与 Fast 的延迟、吞吐、显存和 planning 时间；
  6. 按第二轮调优计划决定扩容、timestep 和 terminal loss 实验。

  当前最稳妥的正式表述是：“在 Cube 上，Action Head 辅助监督能提高 Fast Stage-B；该增益尚未在 Push-T 复现，且尚未超过原 LeWM 基线。”
