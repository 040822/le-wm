#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/users/wenxin/pre-exp/le-wm"
PYTHON="${ROOT}/.venv/bin/python"
OUTPUT_ROOT="${ROOT}/outputs/round4/phase45_3_seed3072_dev"
LOG_ROOT="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_ROOT}"

run_task() {
  local task="$1"
  local gpu="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${ROOT}/scripts/round4_phase45_3.py" \
    --config "${ROOT}/config/round4/phase45_3.json" \
    --output-root "${OUTPUT_ROOT}" \
    --task "${task}" \
    --device cuda \
    --gpu "${gpu}" \
    > "${LOG_ROOT}/${task}.log" 2>&1
}

run_task cube 0 & cube_pid=$!
run_task pusht 1 & pusht_pid=$!
run_task reacher 2 & reacher_pid=$!
run_task tworoom 3 & tworoom_pid=$!

wait "${cube_pid}"
wait "${pusht_pid}"
wait "${reacher_pid}"
wait "${tworoom_pid}"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u "${ROOT}/scripts/round4_phase45_3.py" \
  --config "${ROOT}/config/round4/phase45_3.json" \
  --output-root "${OUTPUT_ROOT}" \
  --device cpu \
  --analyze-only \
  > "${LOG_ROOT}/analysis.log" 2>&1
