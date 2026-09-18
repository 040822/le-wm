#!/usr/bin/env bash
set -euo pipefail

# Evaluate the four completed Phase 3 E5 physical-time/type checkpoints on
# their frozen round3_revised dev cohorts. Training and evaluation are separate
# resource windows; this script refuses to run until the registry is complete.

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <cube_gpu> <pusht_gpu> <reacher_gpu> <tworoom_gpu>" >&2
    exit 2
fi
if [[ "${ROUND3_RESOURCE_WINDOW_CONFIRMED:-}" != "1" ]]; then
    echo "Set ROUND3_RESOURCE_WINDOW_CONFIRMED=1 only after verifying Round4 is not using these GPUs." >&2
    exit 2
fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
GPUS=("$1" "$2" "$3" "$4")
TASKS=(cube pusht reacher tworoom)
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

"${ROOT}/.venv/bin/python" scripts/verify_round3_phase3_e5.py --strict

"${ROOT}/.venv/bin/python" scripts/round3_resource_preflight.py --gpus "${GPUS[@]}"

LOG_ROOT="${ROUND3_PHASE3_EVAL_LOG_ROOT:-${ROOT}/outputs/round3/phase3/eval_logs}"
EVAL_ROOT="${ROUND3_PHASE3_EVAL_ROOT:-${ROOT}/outputs/round3/phase3/evals}"
mkdir -p "${LOG_ROOT}" "${EVAL_ROOT}"
pids=()
for index in "${!TASKS[@]}"; do
    task="${TASKS[$index]}"
    gpu="${GPUS[$index]}"
    checkpoint="outputs/round3/phase3/fast_lewam/${task}/e5_physical_time_type_192_seed3072/checkpoints/fast_lewam_weights_epoch_10.pt"
    cohort="outputs/round3/phase1/cohorts/${task}/dev_round3_revised.json"
    log="${LOG_ROOT}/${task}_stages.log"
    echo "[$(date '+%F %T')] Evaluating ${task} on physical GPU${gpu}" | tee "${log}"
    (
        for stage in stage_a stage_a_shuffled_goal stage_b; do
            echo "[$(date '+%F %T')] ${task} ${stage}" >>"${log}"
            CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_EGL_DEVICE_ID="${gpu}" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
                "${ROOT}/.venv/bin/python" -u scripts/round3_phase1.py evaluate \
                "${task}" e5_fast "${stage}" \
                --protocol-variant round3_revised \
                --cohort "${cohort}" \
                --checkpoint "${checkpoint}" \
                --epoch 10 \
                --device cuda:0 \
                --output "${EVAL_ROOT}" \
                >>"${log}" 2>&1
        done
    ) &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"
