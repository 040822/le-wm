# CVPR Table 1 复核与论文更新

日期：2026-10-05。仅读取既有实验产物；未启动训练或 GPU 评测。

## 核验结果

- `summary/all_independent_cells.csv` 的 948 个单元均 complete；逐一读取引用的 result.json，确认每单元 50 episodes、成功率与 CSV 一致，并验证 episodes.jsonl 的 SHA256 与 result 中 trace_sha256 一致，共 47,400 episodes。
- 按 task/evaluation seed 分组，各方法 cohort_sha256 一致。
- 从 `table1_six_rows_per_seed.csv` 独立重算全部六行四任务 mean ± sample std，以及先按 seed 求四任务平均再计算的 mean ± sample std，与报告逐项一致。
- `timing_summary.json` 的 158 个条件均 complete，每条件 250 样本／50 状态；独立重算全部 mean_ms，一致。全部动作和选择 parity 均通过（单候选的选择项为不适用）。归档前后均无其他 GPU compute process。
- 主表六行的四任务平均计时重算一致：625.6725、51.4348、29.8782、45.2316、29.5301、663.9982 ms。
- 计时实际 GPU6，RTX 4090；GPU3→GPU6 的变更和原用户授权已归档于 `provenance/timing_device_override.json`。早期计划的五状态方案被实际冻结的 50 状态 × 5 次替代。

## 论文更新

- `paper/tables/final_main.tex`：六行正式结果、四任务平均和初始重规划延迟。
- `paper/results.tex`：回填数值、完成范围、计时协议和结论；保留尚无对应证据的占位值。
- `paper/sec/4_experiments.tex`：真实 baseline 配置、样本数、标准差定义、时域例外和结果解释。
- `paper/sec/0_abstract.tex`：加入已完成成功率和计时结果；明确等预算与训练机制研究待完成。
- `paper/docs/MAIN_TABLE_PROTOCOL.md`、`paper/README.md`：更新完成状态及溯源。

## 解释边界

Selection 为 96.25 ± 1.41%，LeFlow 为 96.17 ± 1.75%；0.08 个百分点差异不能据此认定显著优势。Selection 的初始延迟分别为 LeFlow 和 LeWM 的约 1/1.72、1/20.94，这些比率基于四任务平均延迟，不是逐任务加速比的平均。Refinement 为 94.50 ± 1.26%，45.23 ms，当前工作点不优于 Selection。Selection† 为 99.17 ± 0.41%，但 Reacher 改用 10/10 时域，不能当作同一时域比较；其他三任务复用不增加样本量。误差条不覆盖训练 seed 变化；本次主表不证明训练耦合的因果收益或等时间预算优势。

## 文稿检查

结果键无重复，更新表格与正文引用均有定义，更新源文件花括号平衡。未安装 latexmk、pdflatex 或 tectonic，未编译 PDF，尚未核验分页和排版。已有其它工作区修改保留。
