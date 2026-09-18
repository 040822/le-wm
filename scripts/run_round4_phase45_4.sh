#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/users/wenxin/pre-exp/le-wm"
PYTHON="${ROOT}/.venv/bin/python"
CONFIG="${ROOT}/config/round4/phase45_4.json"
OUTPUT_ROOT="${ROOT}/outputs/round4/phase45_4_seed3072_legacy_dev"
LOG_ROOT="${OUTPUT_ROOT}/logs"
MIN_FREE_MIB=4096

mkdir -p "${LOG_ROOT}"

check_gpu() {
  local gpu="$1"
  CUDA_VISIBLE_DEVICES="${gpu}" nvidia-smi \
    --id "${gpu}" \
    --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
}

for gpu in 4 1 2 3; do
  check_gpu "${gpu}" > "${LOG_ROOT}/gpu${gpu}_preflight.log"
done

run_task() {
  local task="$1"
  local gpu="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${ROOT}/scripts/round4_phase45_4.py" \
    --config "${CONFIG}" \
    --output-root "${OUTPUT_ROOT}" \
    --task "${task}" \
    --device cuda \
    --gpu "${gpu}" \
    --min-free-mib "${MIN_FREE_MIB}" \
    > "${LOG_ROOT}/${task}.log" 2>&1
}

run_task cube 4 & cube_pid=$!
run_task pusht 1 & pusht_pid=$!
run_task reacher 2 & reacher_pid=$!
run_task tworoom 3 & tworoom_pid=$!

wait "${cube_pid}"
wait "${pusht_pid}"
wait "${reacher_pid}"
wait "${tworoom_pid}"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u "${ROOT}/scripts/round4_phase45_4.py" \
  --config "${CONFIG}" \
  --output-root "${OUTPUT_ROOT}" \
  --report-output "${ROOT}/docs/report/round4/round4_phase45-4_legacy_protocol_comparison_report.md" \
  --device cpu \
  --analyze-only \
  > "${LOG_ROOT}/analysis.log" 2>&1
