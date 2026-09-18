#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/users/wenxin/pre-exp/le-wm"
PYTHON="${ROOT}/.venv/bin/python"
OUTPUT_ROOT="${ROOT}/outputs/round4/phase45_3_seed3072_dev"
LOG_ROOT="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_ROOT}"

LATE_INDEX_ARGS=(
  --condition-index 50
  --condition-index 51
  --condition-index 52
  --condition-index 53
  --condition-index 54
  --condition-index 55
  --condition-index 56
  --condition-index 57
  --condition-index 58
  --condition-index 59
  --condition-index 60
  --condition-index 61
  --condition-index 62
)

run_task() {
  local task="$1"
  local gpu="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${ROOT}/scripts/round4_phase45_3.py" \
    --config "${ROOT}/config/round4/phase45_3.json" \
    --output-root "${OUTPUT_ROOT}" \
    --task "${task}" \
    --device cuda \
    --gpu "${gpu}" \
    --skip-lewm \
    "${LATE_INDEX_ARGS[@]}" \
    > "${LOG_ROOT}/late_${task}.log" 2>&1
}

run_task cube 5 & cube_pid=$!
run_task pusht 7 & pusht_pid=$!
run_task reacher 6 & reacher_pid=$!
run_task tworoom 4 & tworoom_pid=$!

wait "${cube_pid}"
wait "${pusht_pid}"
wait "${reacher_pid}"
wait "${tworoom_pid}"
