# Round 3 Phase 3：E5 physical-time/type 汇总

## 验收状态

- 严格校验命令：`PYTHONPATH=. .venv/bin/python scripts/verify_round3_phase3_e5.py --strict`
- 严格校验结果：`status=ok`
- 协议：`round3_revised`
- 开发集：每任务 50 episodes；四任务 cohort hash 与注册表一致
- 最终测试集：每任务 200 episodes；四任务 final cohort hash 与结果一致
- 评测阶段：`stage_a`、`stage_a_shuffled_goal`、`stage_b`
- 评测预算：goal offset 25 steps、eval budget 50、horizon 5、action block 5
- CEM：300 samples、30 iterations、top-k 30

## 表示与训练配置

- `token_encoding=physical_time_type`
- `stage_b_timestep_mode=legacy`
- `stage_b_dynamics=parallel_prefix`
- `stage_b_attention_mode=strict_causal`
- `train_mode=stage_ab`
- `stage_a_goal_injection=token`
- `seed=3072`
- `latent_head_dim=192`
- `latent_action_mix_epochs=10`
- `detach_clean_action=false`
- `serial_activation_checkpointing=false`
- `inference_steps=10`
- 初始化：随机初始化；未进行 legacy E5 checkpoint 的 partial transfer

## Dev 结果

| Task | Stage A | A-shuf | Stage B | Checkpoint SHA256 | Dev cohort SHA256 |
|---|---:|---:|---:|---|---|
| Cube | 100% | 8% | 44% | `a5d70ea91eccc25d950cd6a448ef5fae579354ceea04e8cb3e11dde2c93480ec` | `578f6ea52e56d77461cb5bb2aeb01a004a21b9f6659689596f6f55f25dc17d01` |
| PushT | 94% | 18% | 80% | `89de1f134800753a925bc7b86424782cb4948b1fd3f62bb19418ece3cee2023d` | `5b3cd95037c17bc6282bdd29cab4e550aff1a5205d051b2991263a1bac937560` |
| Reacher | 68% | 6% | 82% | `e178b7d80adba649d169483d1f97a2ce1b0cb332b585caf7bdf554ec5e9448fc` | `d18695b2377000ce7dd774fcad78e4bb2d75980fb3ccbd98cc83d09ef50f1d06` |
| TwoRoom | 96% | 46% | 96% | `2a1a11d52484b26aa576a7674357ab4e502d16ea17829e5d4248d83e54eb7847` | `367aefc82f4c0786992172f535f00b743750108a8614c2d282019e9a78abdc6b` |

## Final 结果

| Task | Stage A | A-shuf | Stage B | Episodes | Final cohort SHA256 |
|---|---:|---:|---:|---:|---|
| Cube | 100.0% | 8.5% | 51.5% | 200 | `26171642a8c6a3617b0c586c2d0f448f9d352aa433d6f8a169b8fffdecb0e030` |
| PushT | 93.5% | 12.5% | 86.0% | 200 | `138501dd8c2d88990074da59039aa092d4269e841c7114fe68ae49fdd36cc47b` |
| Reacher | 73.0% | 6.0% | 83.0% | 200 | `f81e4830dd2bb80ad1a8109b67873496fa86651ccba63d49ab6d8b0c78b87490` |
| TwoRoom | 93.5% | 45.5% | 96.0% | 200 | `d981c8872d168ca951450d01f713315df269bb8ee61c6c268f6c61956c3b9421` |

## 与历史 E5 对比

以下历史 E5 数值直接取自 `docs/report/round3/phase1/round3_new_protocol_experiment_report.md` §4 的
200-episode 新协议主表。历史 E5 使用 legacy token 语义；本报告的 Phase 3 E5 使用
`physical_time_type`，因此这里是描述性对照，不是只改变 token 编码的严格因果消融。

| Task | 历史 E5 legacy A | 历史 A-shuf | 历史 B | Phase 3 E5 physical-time/type A | Phase 3 A-shuf | Phase 3 B | Δ A / A-shuf / B (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cube | 99.5% | 9.0% | 49.5% | 100.0% | 8.5% | 51.5% | +0.5 / −0.5 / +2.0 |
| PushT | 93.0% | 7.0% | 89.5% | 93.5% | 12.5% | 86.0% | +0.5 / +5.5 / −3.5 |
| Reacher | 79.5% | 9.0% | 84.5% | 73.0% | 6.0% | 83.0% | −6.5 / −3.0 / −1.5 |
| TwoRoom | 95.0% | 44.0% | 95.5% | 93.5% | 45.5% | 96.0% | −1.5 / +1.5 / +0.5 |

Phase 3 的 Cube 和 TwoRoom Stage-B 略高于历史 E5，PushT、Reacher 略低；四任务
不能据此得出 `physical_time_type` 的统一收益。两边虽都报告为 200 episodes，仍应
保留 checkpoint、训练表示和 cohort provenance 的差异，不能把该表当作同权重迁移或
同 cohort 的严格 ablation。

## 证据边界

本报告覆盖四个 E5 physical-time/type epoch-10 checkpoint、冻结 dev cohort 的三阶段评测，以及冻结 final cohort 的 200-episode 三阶段评测，并纳入历史 E5 的描述性对照。它不代表 E0/Fast 在线适应或多种子复现已经完成；后续若继续 Round 3，仍需单独登记在线适应实验和匹配离线对照，不能用本表替代这些证据。
