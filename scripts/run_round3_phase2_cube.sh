#!/usr/bin/env bash
set -euo pipefail

# One-seed Round 3 Phase 2 Cube runner.  The caller must choose an idle,
# allow-listed physical GPU; this wrapper maps it to logical cuda:0 and never
# touches Round 4 processes.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:?usage: $0 <GPU_ID 0-3> [max_environment_steps] [output_root]}"
MAX_STEPS="${2:-20000}"
OUTPUT_ROOT="${3:-outputs/round3/phase2/cube_e0_three_arms_100step}"

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
    --minimum-free-mib 24000 \
    --allow-existing-compute

exec "${ROOT}/.venv/bin/python" -u scripts/round3_phase2.py run-phase2 \
    --device cuda:0 \
    --max-environment-steps "${MAX_STEPS}" \
    --update-interval 100 \
    --output-root "${OUTPUT_ROOT}"
