# Round 3 Phase 1 protocol

This repository implements Phase 1 as an audit/re-evaluation workflow. It
does not train a model, discover an intermediate checkpoint, or replace a
missing weight with another experiment.

The frozen defaults are in
[`config/round3/phase1_protocol.json`](../../config/round3/phase1_protocol.json).
They are `seed=42`, `goal_offset_steps=25`, `eval_budget=50`,
`horizon=receding_horizon=action_block=5`, and CEM
`num_samples=300`, `n_steps=30`, `topk=30`, `var_scale=1.0`. Final testing is
200 episodes per task and uses only `round3_revised`.

## Workflow

Run the CPU-only audit and registry first:

```bash
python -m scripts.round3_phase1 audit --runtime-refresh --output outputs/round3/phase1
python -m scripts.round3_phase1 registry --output outputs/round3/phase1
```

The registry is generated from
[`config/round3/phase1_artifacts.json`](../../config/round3/phase1_artifacts.json).
It records absolute checkpoint paths, SHA256, final epoch, training config
fields, dataset names, historical result status, dependency versions, code
commit, and possible training/evaluation data exposure. Entries with missing
or invalid assets have `participates=false` and are never substituted.

Create the deterministic data-only manifests after the local dataset is
available:

```bash
python -m scripts.round3_phase1 cohort cube --output outputs/round3/phase1
python -m scripts.round3_phase1 dev-protocols cube --output outputs/round3/phase1
```

This writes `cohorts/<task>/dev.json` and `final.json`. Each manifest stores
the raw episode split, future goal row, start distance, five fixed strata,
candidate counts, and a SHA256 over its canonical payload. Development and
final starts are episode-disjoint from each other and from the online pool.
The legacy manifest command preserves the upstream global-last-row exclusion
and records duplicate episode IDs and initial-success starts for diagnosis.

For a permitted online evaluation, use an explicit manifest and final weight:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.round3_phase1 evaluate cube e5_fast stage_b \
  --protocol-variant round3_revised \
  --cohort outputs/round3/phase1/cohorts/cube/final.json \
  --checkpoint outputs/fast_lewam/cube/0803_e5_cross_head_grad/checkpoints/fast_lewam_weights_epoch_10.pt \
  --device cuda
```

Every GPU invocation must set `CUDA_VISIBLE_DEVICES` to a subset of physical
GPUs 0, 1, 2, and 3. The evaluator rejects an unset variable and any value
containing GPU 4--7.

## Predicate audit

`source/common/round3_protocol.py` records the installed runtime source hashes
and the exact field/unit/boundary contract:

| Task | Current | Goal | Success |
|---|---|---|---|
| Cube | `privileged_block_0_pos` | `goal_privileged_block_0_pos` | position L2 `<= 0.04 m` |
| Reacher | `qpos` | `goal_qpos` | every joint absolute error `< 0.05 rad` (`qpos_match`) |
| Push-T | `state` | `goal_state` | position L2 `< 20 px` and circular angle error `< pi/9` |
| TwoRoom | `proprio` | `goal_proprio` | L2 `< 16 px` |

The comparison operators are intentional. The generic goal-refresh adapter
requires ten deterministic cases and verifies old/new goal readback and
success against the new goal. The static audit reports
`goal_refresh.status=not_run` only when the environment adapter was not requested; the runtime-refresh command records ten installed-runtime checks; the
synthetic option is marked as a synthetic adapter and is not presented as an
environment result.

## Results and analysis

`source/common/round3_eval.py` observes the installed evaluator's environment
pool and writes per-step states, goals, physical distance, success, actions,
legality, termination, and episode summaries to `episodes.jsonl`. Result JSON
adds `protocol_variant`, `cohort_id`, `cohort_sha256`, `predicate_version`,
`code_commit`, and trace schema metadata. `source/common/round3_analysis.py`
can relabel the same trace for tolerance-only comparisons without collecting
a second action sequence. `summarize_episodes` reports success vectors, Wilson
intervals, initial success, distance quantiles, hold-window success, first
success step, terminal distance, and rollout diagnostics. Pairwise changes use
matched `(episode_id, start_step)` keys and report improved, regressed, stable
success/failure, and percentage-point deltas.

After results are present, generate the descriptive matrix and report:

```bash
python -m scripts.round3_phase1 analyze --output outputs/round3/phase1
```

The final report does not average raw distances or success rates across the
four tasks and does not use final-test outcomes to modify a frozen predicate,
cohort, budget, or model choice.
