# Round5 Phase1.5：冻结模型的决策能力诊断与推理方案收敛

本报告由独立 Phase1.5 入口生成。结果只使用 R4-AB seed3072 epoch10 和 legacy_50，不修改 Phase1 冻结结果。闭环成功率是探索性证据，重复这 50 个起点不能替代最终泛化评测。

- 配置：`/data/users/wenxin/pre-exp/le-wm/config/round5/phase1_5.json`
- 输出根目录：`/data/users/wenxin/pre-exp/le-wm/outputs/round5/phase1_5_seed3072_legacy`
- 条件状态：`{"completed": 100, "pending": 2324}`

## 主扫描验收

| group | planned | indexed | pending |
|---|---:|---:|---:|
| A1 | 144 | 48 | 96 |
| A2 | 432 | 16 | 416 |
| A3 | 864 | 4 | 860 |
| A4 | 432 | 16 | 416 |
| A5 | 96 | 0 | 96 |
| A6 | 240 | 16 | 224 |
| A7 | 216 | 0 | 216 |

主扫描网格为 2,424 条条件、121,200 个 episode；固定 seed43/44 稳定性扩展最多增加 176 条条件。
历史结果只有在 checkpoint、legacy cohort、normalizer、动作裁剪、精度、随机数和候选生成语义都一致时才复用；缺轨迹历史结果只进入 success-only 统计。

## 当前最高成功率条件（描述性）

| task | group | family | success | p50 planning (s) | source |
|---|---|---|---:|---:|---|
| tworoom | A4 | p3_post_opt | 100.0% | 1.023237818852067 | history |
| cube | A2 | p0_post_opt | 100.0% | 0.1953987660817802 | history |
| tworoom | A6 | cem_budget | 100.0% | 14.729640062432736 | history |
| cube | A3 | p0_guided_flow | 100.0% | 0.5832880977541208 | history |
| tworoom | A6 | cem_budget | 100.0% | 3.2451155183371156 | history |
| cube | A4 | p3_refine | 100.0% | 0.348144163377583 | history |
| tworoom | A4 | p3_refine | 100.0% | 1.2689756895415485 | history |
| cube | A1 | proposal_ranking | 100.0% | 0.2923399251885712 | history |
| cube | A1 | proposal_ranking | 100.0% | 0.29689727863296866 | history |
| tworoom | A1 | proposal_ranking | 100.0% | 0.6107347696088254 | history |
| cube | A2 | p0_post_opt | 100.0% | 0.24421118991449475 | history |
| cube | A1 | proposal_ranking | 100.0% | 0.5383123108185828 | history |
| cube | A4 | p3_refine | 100.0% | 0.3450439455918968 | history |
| cube | A4 | p3_post_opt | 100.0% | 1.6279823179356754 | history |
| cube | A1 | proposal_ranking | 100.0% | 0.2542917304672301 | history |
| cube | A4 | p3_post_opt | 100.0% | 0.7936307601630688 | history |
| tworoom | A1 | proposal_ranking | 100.0% | 0.4673698083497584 | history |
| cube | A1 | proposal_ranking | 100.0% | 0.42275777040049434 | history |
| tworoom | A4 | p3_refine | 100.0% | 0.3273179279640317 | history |
| cube | A1 | proposal_ranking | 100.0% | 0.36359437135979533 | history |

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
