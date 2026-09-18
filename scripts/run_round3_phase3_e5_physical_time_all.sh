#!/usr/bin/env bash
set -euo pipefail

# Launch the four registered Phase 3 E5 runs only after the caller confirms
# that Round 4 has released the selected GPUs.  Each child gets exactly one
# physical GPU from 0--3 and maps it to logical cuda:0.

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <cube_gpu> <pusht_gpu> <reacher_gpu> <tworoom_gpu>" >&2
    exit 2
fi
if [[ "${ROUND3_RESOURCE_WINDOW_CONFIRMED:-}" != "1" ]]; then
    echo "Set ROUND3_RESOURCE_WINDOW_CONFIRMED=1 only after verifying Round4 is not using these GPUs." >&2
    exit 2
fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASKS=(cube pusht reacher tworoom)
GPUS=("$1" "$2" "$3" "$4")
for gpu in "${GPUS[@]}"; do
    case "${gpu}" in
        0|1|2|3) ;;
        *) echo "GPU ${gpu} is prohibited; only GPU0-GPU3 are allowed." >&2; exit 2 ;;
    esac
done
if [[ "${GPUS[0]}" == "${GPUS[1]}" || "${GPUS[0]}" == "${GPUS[2]}" || "${GPUS[0]}" == "${GPUS[3]}" || \
      "${GPUS[1]}" == "${GPUS[2]}" || "${GPUS[1]}" == "${GPUS[3]}" || "${GPUS[2]}" == "${GPUS[3]}" ]]; then
    echo "Phase3 task GPU assignments must be distinct." >&2
    exit 2
fi

LOG_ROOT="${ROUND3_PHASE3_LOG_ROOT:-${ROOT}/outputs/round3/phase3/logs}"
mkdir -p "${LOG_ROOT}"
"${ROOT}/.venv/bin/python" scripts/round3_resource_preflight.py --gpus "${GPUS[@]}"
pids=()
for index in "${!TASKS[@]}"; do
    task="${TASKS[$index]}"
    gpu="${GPUS[$index]}"
    log="${LOG_ROOT}/${task}_e5_physical_time_type.log"
    echo "[$(date '+%F %T')] Starting ${task} on physical GPU${gpu}" | tee "${log}"
    "${ROOT}/scripts/run_round3_phase3_e5_physical_time.sh" "${task}" "${gpu}" \
        >>"${log}" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"
