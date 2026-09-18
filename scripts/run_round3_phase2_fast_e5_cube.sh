#!/usr/bin/env bash
set -euo pipefail

# One-seed Round 3 legacy E5 online validation.  The caller chooses one
# allow-listed physical GPU; the process sees it as logical cuda:0.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:?usage: $0 <GPU_ID 0-3> [max_environment_steps] [output_root]}"
MAX_STEPS="${2:-20000}"
OUTPUT_ROOT="${3:-outputs/round3/phase2/cube_e5_legacy_three_arms_100step}"
MIN_FREE_MIB="${ROUND3_E5_MINIMUM_FREE_MIB:-16000}"
RESUME_MODE="${4:-}"

case "${GPU_ID}" in
    0|1|2|3) ;;
    *)
        echo "GPU ${GPU_ID} is prohibited; only GPU0-GPU3 are allowed." >&2
        exit 2
        ;;
esac

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_EGL_DEVICE_ID="${GPU_ID}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export STABLEWM_HOME="${STABLEWM_HOME:-${ROOT}/data}"

"${ROOT}/.venv/bin/python" scripts/round3_resource_preflight.py \
    --gpus "${GPU_ID}" \
    --minimum-free-mib "${MIN_FREE_MIB}" \
    --allow-existing-compute

RESUME_ARGS=()
if [[ "${RESUME_MODE}" == "resume" ]]; then
    RESUME_ARGS+=(--resume)
elif [[ -n "${RESUME_MODE}" ]]; then
    echo "fourth argument must be 'resume' when provided" >&2
    exit 2
fi

exec "${ROOT}/.venv/bin/python" -u scripts/round3_phase2_fast.py run-phase2-fast \
    --device cuda:0 \
    --max-environment-steps "${MAX_STEPS}" \
    --update-interval 100 \
    --output-root "${OUTPUT_ROOT}" \
    "${RESUME_ARGS[@]}"
