# Round5 Phase1.5：冻结模型的决策能力诊断与推理方案收敛

本报告由独立 Phase1.5 入口生成。结果只使用 R4-AB seed3072 epoch10 和 legacy_50，不修改 Phase1 冻结结果。闭环成功率是探索性证据，重复这 50 个起点不能替代最终泛化评测。

- 配置：`/data/users/wenxin/pre-exp/le-wm/config/round5/phase1_5.json`
- 输出根目录：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase1_5_seed3072_legacy`
- 条件状态：`{"completed": 483, "pending": 1941}`

## 主扫描验收

| group | planned | indexed | pending |
|---|---:|---:|---:|
| A1 | 144 | 110 | 34 |
| A2 | 432 | 77 | 355 |
| A3 | 864 | 66 | 798 |
| A4 | 432 | 75 | 357 |
| A5 | 96 | 76 | 20 |
| A6 | 240 | 52 | 188 |
| A7 | 216 | 27 | 189 |

主扫描网格为 2,424 条条件、121,200 个 episode；固定 seed43/44 稳定性扩展最多增加 176 条条件。
历史结果只有在 checkpoint、legacy cohort、normalizer、动作裁剪、精度、随机数和候选生成语义都一致时才复用；缺轨迹历史结果只进入 success-only 统计。

## 当前最高成功率条件（描述性）

| task | group | family | success | p50 planning (s) | source |
|---|---|---|---:|---:|---|
| cube | A3 | p0_guided_flow | 100.0% | 0.17453392734751105 | current |
| cube | A6 | cem_budget | 100.0% | 0.801135943736881 | current |
| cube | A3 | p0_guided_flow | 100.0% | 0.08105652313679457 | current |
| cube | A3 | p0_guided_flow | 100.0% | 0.34935761522501707 | current |
| cube | A1 | proposal_ranking | 100.0% | 0.21119897812604904 | current |
| cube | A2 | p0_post_opt | 100.0% | 0.10222758539021015 | current |
| cube | A5 | p3_guided_flow | 100.0% | 0.5643311040475965 | current |
| cube | A2 | p0_post_opt | 100.0% | 0.06399708706885576 | current |
| cube | A3 | p0_guided_flow | 100.0% | 0.16736199613660574 | current |
| tworoom | A4 | p3_post_opt | 100.0% | 1.023237818852067 | history |
| cube | A4 | p3_post_opt | 100.0% | 3.2565162028186023 | current |
| cube | A3 | p0_guided_flow | 100.0% | 0.18022423470392823 | current |
| cube | A6 | cem_budget | 100.0% | 1.0599323119968176 | current |
| cube | A4 | p3_post_opt | 100.0% | 0.3081235936842859 | current |
| cube | A3 | p0_guided_flow | 100.0% | 0.10032219812273979 | current |
| cube | A4 | p3_post_opt | 100.0% | 2.7060805475339293 | current |
| cube | A4 | p3_post_opt | 100.0% | 0.9021547930315137 | current |
| cube | A4 | p3_refine | 100.0% | 0.38333469722419977 | current |
| cube | A2 | p0_post_opt | 100.0% | 0.1953987660817802 | history |
| tworoom | A6 | cem_budget | 100.0% | 14.729640062432736 | history |

## 诊断与决策

- 候选池：报告 oracle、B-selected、selection regret，并按状态聚类 bootstrap；常量候选池相关系数保持 undefined。
- 梯度修正：同时报告预测 latent cost、真实 latent cost 和物理距离改善，以及 model exploitation。
- Probe：按轨迹拆分 train/validation，排除评测轨迹；真实未来图像与 B 预测 future latent 分开报告。
- 计时：包含编码、提案、评分/梯度、CEM 更新和动作输出；环境时间单独报告。

本报告仍处于扫描阶段，条件尚未齐全；P3、PO/GF、CEM 的最终取舍暂记为尚未收敛。

## 产物

- 索引：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase1_5_seed3072_legacy/analysis/index.json`
- 条件 CSV：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase1_5_seed3072_legacy/analysis/conditions.csv`
- 本报告：`/data/users/wenxin/pre-exp/le-wm/docs/report/round5/round5_phase1_5_report.md`
