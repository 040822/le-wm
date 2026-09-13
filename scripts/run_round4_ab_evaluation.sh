#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/users/wenxin/pre-exp/le-wm"
EVAL_ROOT="${ROOT}/outputs/round4/ab_seed3072_eval"
ABDE_ROOT="${ROOT}/outputs/round4/epoch10_seed3072"
ABDE_P3_ROOT="${ROOT}/outputs/round4/epoch10_seed3072_batch64"

mkdir -p "${EVAL_ROOT}/logs"

declare -A CHECKPOINTS=(
  [cube]="${ROOT}/outputs/round4/ab_seed3072_cube/checkpoints/r4_ab_seed3072_weights_epoch_10.pt"
  [pusht]="${ROOT}/outputs/round4/ab_seed3072_pusht/checkpoints/r4_ab_seed3072_weights_epoch_10.pt"
  [reacher]="${ROOT}/outputs/round4/ab_seed3072_reacher/checkpoints/r4_ab_seed3072_weights_epoch_10.pt"
  [tworoom]="${ROOT}/outputs/round4/ab_seed3072_tworoom/checkpoints/r4_ab_seed3072_weights_epoch_10.pt"
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

wait_for_training() {
  while true; do
    ready=1
    for task in cube pusht reacher tworoom; do
      if [[ ! -s "${CHECKPOINTS[$task]}" ]]; then
        ready=0
      fi
    done
    if [[ "$ready" -eq 1 ]]; then
      return
    fi
    sleep 60
  done
}

run_mode() {
  local task="$1"
  local gpu="$2"
  local mode="$3"
  local cohort_kind="$4"
  local cohort
  if [[ "$cohort_kind" == "dev" ]]; then
    cohort="${COHORTS_DEV[$task]}"
  else
    cohort="${COHORTS_FINAL[$task]}"
  fi
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u scripts/round4.py evaluate \
    "$task" "$mode" \
    --cohort "$cohort" \
    --checkpoint "${CHECKPOINTS[$task]}" \
    --epoch 10 \
    --device cuda \
    --gpu "$gpu" \
    --candidate-count 64 \
    --flow-steps 16 \
    --solver-batch-size 1 \
    --candidate-batch-size 64 \
    --output "$EVAL_ROOT" \
    > "${EVAL_ROOT}/logs/${task}_${mode}_${cohort_kind}.log" 2>&1
}

run_task() {
  local task="$1"
  local gpu="$2"
  run_mode "$task" "$gpu" P0 dev
  run_mode "$task" "$gpu" P0 final
  run_mode "$task" "$gpu" P0-shuf dev
  run_mode "$task" "$gpu" P1 dev
  run_mode "$task" "$gpu" P1 final
  run_mode "$task" "$gpu" P2 dev
  run_mode "$task" "$gpu" P2 final
  run_mode "$task" "$gpu" P3 dev
  run_mode "$task" "$gpu" P3 final
}

wait_for_training
run_task cube 2 &
cube_pid=$!
run_task pusht 3 &
pusht_pid=$!
run_task reacher 1 &
reacher_pid=$!
run_task tworoom 0 &
tworoom_pid=$!
wait "$cube_pid" "$pusht_pid" "$reacher_pid" "$tworoom_pid"

.venv/bin/python scripts/round4_compare.py \
  --ab-root "$EVAL_ROOT" \
  --abde-root "$ABDE_ROOT" \
  --abde-p3-root "$ABDE_P3_ROOT" \
  --output "${ROOT}/outputs/round4/ab_vs_abde"
