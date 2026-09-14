#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/users/wenxin/pre-exp/le-wm"
OUTPUT_ROOT="${ROOT}/outputs/round4/ab_flow_steps_seed3072"
CHECKPOINT_ROOT="${ROOT}/outputs/round4"

declare -A CHECKPOINTS=(
  [cube]="${CHECKPOINT_ROOT}/ab_seed3072_cube/checkpoints/r4_ab_seed3072_weights_epoch_10.pt"
  [pusht]="${CHECKPOINT_ROOT}/ab_seed3072_pusht/checkpoints/r4_ab_seed3072_weights_epoch_10.pt"
  [reacher]="${CHECKPOINT_ROOT}/ab_seed3072_reacher/checkpoints/r4_ab_seed3072_weights_epoch_10.pt"
  [tworoom]="${CHECKPOINT_ROOT}/ab_seed3072_tworoom/checkpoints/r4_ab_seed3072_weights_epoch_10.pt"
)

declare -A GPUS=(
  [cube]=0
  [pusht]=1
  [reacher]=2
  [tworoom]=3
)

declare -A COHORTS_DEV=(
  [cube]="${ROOT}/outputs/round3/phase1/cohorts/cube/dev_round3_revised.json"
  [pusht]="${ROOT}/outputs/round3/phase1/cohorts/pusht/dev_round3_revised.json"
  [reacher]="${ROOT}/outputs/round3/phase1/cohorts/reacher/dev_round3_revised.json"
  [tworoom]="${ROOT}/outputs/round3/phase1/cohorts/tworoom/dev_round3_revised.json"
)

declare -A COHORTS_FINAL=(
  [cube]="${ROOT}/outputs/round3/phase1/cohorts/cube/final.json"
  [pusht]="${ROOT}/outputs/round3/phase1/cohorts/pusht/final.json"
  [reacher]="${ROOT}/outputs/round3/phase1/cohorts/reacher/final.json"
  [tworoom]="${ROOT}/outputs/round3/phase1/cohorts/tworoom/final.json"
)

STEPS=(1 2 5 10 16)

for task in cube pusht reacher tworoom; do
  if [[ ! -s "${CHECKPOINTS[$task]}" ]]; then
    echo "missing checkpoint for ${task}: ${CHECKPOINTS[$task]}" >&2
    exit 1
  fi
done

run_mode() {
  local task="$1"
  local mode="$2"
  local cohort_kind="$3"
  local output_root="$4"
  local action_flow_steps="${5:-}"
  local cohort
  if [[ "$cohort_kind" == "dev" ]]; then
    cohort="${COHORTS_DEV[$task]}"
  else
    cohort="${COHORTS_FINAL[$task]}"
  fi
  mkdir -p "${output_root}/logs"
  local log_name="${task}_${mode}_${cohort_kind}"
  if [[ -n "$action_flow_steps" ]]; then
    log_name="steps_${action_flow_steps}_${log_name}"
  else
    log_name="invariant_${log_name}"
  fi
  local step_args=()
  if [[ -n "$action_flow_steps" ]]; then
    step_args=(--action-flow-steps "$action_flow_steps")
  fi
  CUDA_VISIBLE_DEVICES="${GPUS[$task]}" .venv/bin/python -u scripts/round4.py evaluate \
    "$task" "$mode" \
    --cohort "$cohort" \
    --checkpoint "${CHECKPOINTS[$task]}" \
    --epoch 10 \
    --device cuda \
    --gpu "${GPUS[$task]}" \
    --candidate-count 64 \
    --flow-steps 16 \
    "${step_args[@]}" \
    --solver-batch-size 1 \
    --candidate-batch-size 64 \
    --output "$output_root" \
    > "${output_root}/logs/${log_name}.log" 2>&1
}

run_task() {
  local task="$1"
  for step in "${STEPS[@]}"; do
    run_mode "$task" P0 dev "${OUTPUT_ROOT}/steps_${step}" "$step"
    run_mode "$task" P0 final "${OUTPUT_ROOT}/steps_${step}" "$step"
    run_mode "$task" P2 dev "${OUTPUT_ROOT}/steps_${step}" "$step"
    run_mode "$task" P2 final "${OUTPUT_ROOT}/steps_${step}" "$step"
    run_mode "$task" P3 dev "${OUTPUT_ROOT}/steps_${step}" "$step"
    run_mode "$task" P3 final "${OUTPUT_ROOT}/steps_${step}" "$step"
  done
  run_mode "$task" P1 dev "${OUTPUT_ROOT}/invariant_p1"
  run_mode "$task" P1 final "${OUTPUT_ROOT}/invariant_p1"
}

run_task cube & cube_pid=$!
run_task pusht & pusht_pid=$!
run_task reacher & reacher_pid=$!
run_task tworoom & tworoom_pid=$!

wait "$cube_pid"
wait "$pusht_pid"
wait "$reacher_pid"
wait "$tworoom_pid"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/round4_flow_steps.py \
  --results-root "$OUTPUT_ROOT" \
  --invariant-root "${OUTPUT_ROOT}/invariant_p1" \
  --output "${OUTPUT_ROOT}/analysis" \
  > "${OUTPUT_ROOT}/logs/analysis.log" 2>&1
