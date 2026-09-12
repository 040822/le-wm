# Round 4 TwoRoom epoch10 report

日期：2026-09-12

TwoRoom 的 R4-ABDE seed `3072` 已完成 10 个 epoch。正式评测使用 `round3_revised` cohort：dev 50 episodes、final 200 episodes、seed 42、goal offset 25、eval budget 50、horizon/receding horizon 5、action block 5。

## LeFlow reference

LeFlow final 复现通过，成功率为 `200/200 = 100%`：

`outputs/round4/leflow_revised_final/tworoom/leflow/final/result.json`

## Round4 planning matrix

| mode | dev | final |
|---|---:|---:|
| P0 | 48/50 (96%) | 189/200 (94.5%) |
| P0-shuf | 24/50 (48%) | — |
| P1 | 50/50 (100%) | 194/200 (97%) |
| P2 | 49/50 (98%) | 197/200 (98.5%) |
| P3 | 49/50 (98%) | 198/200 (99%) |
| P4 | 49/50 (98%) | 199/200 (99.5%) |
| P4-first | 48/50 (96%) | — |

P3/P4/P4-first 使用 64 candidates、16 flow steps、`solver_batch_size=1`、`candidate_batch_size=8`。候选总数保持为 64；candidate batch 只是显存控制，不改变候选集合或 argmin 规则。

## Paired comparisons

- P4 vs P3：dev `0 pp`，final `+0.5 pp`。
- P4 vs P2：dev `0 pp`，final `+1.0 pp`。
- P4-first vs P4：dev `-2 pp`，说明 B verifier/reranking 在该任务上有可测作用。

正式结果和配对 trace 已由 standalone analyzer 写入：

- `outputs/round4/epoch10_seed3072/analysis.csv`
- `outputs/round4/epoch10_seed3072/analysis.json`
- `outputs/round4/epoch10_seed3072/tworoom/`

## Checkpoint identity

```text
checkpoint: outputs/20260912_102425_147611/checkpoints/r4_abde_seed3072_weights_epoch_10.pt
sha256:     153b6baa8b5dac40ef5bc12700a19e117397199750edf289670be3e1e81deec6
```

该结果只代表 seed `3072`，不能作为跨种子稳定性结论；其他三个任务仍在训练中。
