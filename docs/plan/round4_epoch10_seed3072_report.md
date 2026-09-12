# Round 4 epoch-10 / seed-3072 report

## Scope and provenance

- Tasks: Cube, Push-T, Reacher, TwoRoom.
- Training: `stage_abde`, seed `3072`, nominal epoch 10, Shared DiT with A/B/D/E.
- Evaluation protocol: `round3_revised`, seed 42, dev 50 episodes, final 200 episodes, goal offset 25, budget 50, horizon 5, receding horizon 5, action block 5.
- Main result root: `outputs/round4/epoch10_seed3072`.
- CEM timing sidecar: `outputs/round4/epoch10_seed3072_timed`.
- Code commits: `419c704` (R4 implementation), `245ad56` (weight-only continuation), `96b254d` (CEM timing metadata).

The final cohort is fixed and reproducible, but it is not a strict held-out generalization set: it is drawn from the complete original dataset under the existing window-level split protocol. These results are exploratory and do not establish a causal improvement of A/B from adding D/E.

The first Cube/Push-T/Reacher runs were interrupted. Cube and Push-T were continued from their epoch-9 weight snapshots; Reacher was continued from epoch 8. These continuations restore model weights only and reinitialize the optimizer; they do not restore optimizer state. TwoRoom completed the original run directly. This provenance is retained in each run directory.

## Formal R4 success rates (%)

The main table below uses the non-overwritten formal result root. P0-shuf and P4-first are dev-only diagnostics.

### Dev

| task | P0 | P1 | P2 | P3 | P4 | P0-shuf | P4-first |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cube | 100 | 58 | 98 | 100 | 100 | 8 | 100 |
| Push-T | 92 | 94 | 96 | 94 | 90 | 8 | 60 |
| Reacher | 74 | 86 | 68 | 78 | 84 | 4 | 58 |
| TwoRoom | 96 | 100 | 98 | 98 | 98 | 48 | 96 |

### Final

| task | P0 | P1 | P2 | P3 | P4 |
|---|---:|---:|---:|---:|---:|
| Cube | 100 | 59.5 | 95.5 | 100 | 100 |
| Push-T | 89.5 | 86 | 92.5 | 96 | 89 |
| Reacher | 74 | 86.5 | 68.5 | 85 | 78 |
| TwoRoom | 94.5 | 97 | 98.5 | 99 | 99.5 |

Episode-paired comparisons are in `analysis.json`. On final, P4 vs P3 is 0.0 pp / -7.0 pp / -7.0 pp / +0.5 pp for Cube / Push-T / Reacher / TwoRoom. P4 vs P2 is +4.5 pp / -3.5 pp / +9.5 pp / +1.0 pp.

P4-first shows the verifier effect: relative to first-candidate execution, P4 gains 0 pp on Cube, 30 pp on Push-T, 26 pp on Reacher, and 2 pp on TwoRoom in dev.

## Planning timing and resources

P3/P4 timing is in the formal result payloads. P1/P2 CEM timing was added without changing solver semantics and recorded in the timed sidecar. Dev median planning seconds were:

| task | P1 CEM | P2 warm-start CEM | P3 action-64 | P4 latent-64 |
|---|---:|---:|---:|---:|
| Cube | 6.312 | 4.614 | 1.762 | 1.755 |
| Push-T | 5.332 | 4.809 | 1.104 | 1.003 |
| Reacher | 7.740 | 7.417 | 1.445 | 1.365 |
| TwoRoom | 8.644 | 5.507 | 1.014 | 0.956 |

P3/P4 use 64 candidates, 16 flow steps, solver batch 1, and candidate batch 8. The timed CEM sidecar contains p95 latency and forward counts. Its success vectors match the formal P1/P2 vectors except for one Push-T P2 final episode on the repeated environment run (92.0% sidecar versus 92.5% formal); the formal result remains canonical.

## LeFlow reference

The four-task artifact manifest and dependent LeWM checkpoints passed SHA256 validation. Revised final success rates are:

| task | LeFlow final |
|---|---:|
| Cube | 100% |
| Push-T | 96% |
| Reacher | 83.5% |
| TwoRoom | 100% |

The corresponding dev rates are 100%, 98%, 74%, and 100%. These are historical/reference comparisons, not same-checkpoint R4 ablations.

## Training diagnostics

Epoch-10 diagnostics (`loss`, `d_loss`, `d_path_variance`, `e_action_mse`) are:

| task | loss | D loss | D path variance | E action MSE |
|---|---:|---:|---:|---:|
| Cube | 0.9276 | 0.2057 | 0.6592 | 0.0702 |
| Push-T | 1.1630 | 0.2598 | 0.6550 | 0.1244 |
| Reacher | 2.8209 | 0.1953 | 0.6337 | 0.8014 |
| TwoRoom | 2.7301 | 0.3406 | 0.6514 | 0.7604 |

The per-run `round4_diagnostics.jsonl` files retain epoch 0/5/10 where available; resumed runs also retain the pre-resume epoch snapshots.

## Expansion decision

The pre-registered epoch-10 dev gate is `expand=false`; seeds 3073 and 3074 are not run. The gate fails because P4 is not at least +5 pp on two tasks without a large regression: versus P3, only Reacher clears +5 pp and Push-T declines; versus P2, Reacher improves but Push-T declines by 6 pp. The efficiency gate also requires no task to decline by more than 2 pp, which Push-T violates.

Decision artifact: `outputs/round4/epoch10_seed3072/expansion_decision.json`.

## Checkpoints and hashes

| task | checkpoint | SHA256 |
|---|---|---|
| Cube | `outputs/round4/resume3_seed3072_cube/checkpoints/r4_abde_seed3072_weights_epoch_10.pt` | `95a26194eb74e3b02f9d82aa4d15e32b746fdbf24bce974d9eed90550a1c5495` |
| Push-T | `outputs/round4/resume3_seed3072_pusht/checkpoints/r4_abde_seed3072_weights_epoch_10.pt` | `1543ccdffd17c68fb9d3f905621995d760809efdfc86d8cc6c5f1486cbbcb9f1` |
| Reacher | `outputs/round4/resume3_seed3072_reacher/checkpoints/r4_abde_seed3072_weights_epoch_10.pt` | `d56252fe27caa89914c9871514d1bfcfeda4466de6dd34494f64bee89d78a83c` |
| TwoRoom | `outputs/20260912_102425_659562/checkpoints/r4_abde_seed3072_weights_epoch_10.pt` | `153b6baa8b5dac40ef5bc12700a19e117397199750edf289670be3e1e81deec6` |

## Verification

- LeFlow manifest validation: accepted for all four tasks.
- Cohort file SHA256s match `config/round4/cohort_artifacts.json`.
- Targeted R4 tests: pass, including mode/mask, gradients, checkpoint round-trip, candidate splitting, timing wrapper, and continuation behavior.
- Full unittest: 287 tests, 2 skipped, 2 failures. Both failures are pre-existing `test_inspect_h5_dataset.py` expectations for episode-length text that the current inspector omits; no Round 4 test failed.

Deferred experiments remain outside this round: R4-AB, independent D/E models, Flow-IDM, latent-space CEM, new Stage C, consistency/cycle losses, online+D/E, and token/DiT architecture searches.
