#!/usr/bin/env bash
set -euo pipefail

# Round 3 Phase 3 mainline: E5 training semantics with physical-time/type
# token encoding. This script intentionally does not launch evaluation; the
# frozen round3_revised cohort must be evaluated in a separate GPU window.

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <cube|pusht|reacher|tworoom> <gpu_id>" >&2
    exit 2
fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

task=$1
gpu_id=$2

case "${task}" in
    cube|pusht|reacher|tworoom) ;;
    *)
        echo "Unsupported task: ${task}" >&2
        exit 2
        ;;
esac

case "${gpu_id}" in
    0|1|2|3) ;;
    *)
        echo "GPU ${gpu_id} is prohibited; only GPU0-GPU3 are allowed." >&2
        exit 2
        ;;
esac

run_root=${ROUND3_PHASE3_ROOT:-outputs/round3/phase3}
run_dir="${run_root}/fast_lewam/${task}/e5_physical_time_type_192_seed3072"

if [[ -e "${run_dir}" ]]; then
    echo "Refusing to reuse existing run directory: ${run_dir}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export MUJOCO_EGL_DEVICE_ID="${gpu_id}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1
export STABLEWM_HOME="${STABLEWM_HOME:-${ROOT}/data}"

mkdir -p "${run_dir}"

"${ROOT}/.venv/bin/python" -u train.py \
    --config-name=fast_lewam \
    data="${task}" \
    hydra.run.dir="${run_dir}" \
    info=round3_phase3_e5_physical_time_type \
    seed=3072 \
    latent_head_dim=192 \
    policy.model.mlp_dim=768 \
    loader.batch_size=128 \
    trainer.max_epochs=10 \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    epoch_eval.enabled=false \
    validation_diagnostics.enabled=false \
    wandb.enabled=false \
    train_mode=stage_ab \
    stage_a_goal_injection=token \
    detach_clean_action=false \
    latent_action_mix_epochs=10 \
    stage_b_timestep_mode=legacy \
    stage_b_dynamics=parallel_prefix \
    stage_b_attention_mode=strict_causal \
    serial_activation_checkpointing=false \
    inference_steps=10 \
    token_encoding=physical_time_type

checkpoint="${run_dir}/checkpoints/fast_lewam_weights_epoch_10.pt"
if [[ ! -f "${checkpoint}" ]]; then
    echo "Training exited without the required epoch-10 checkpoint: ${checkpoint}" >&2
    exit 1
fi

config_path="${run_dir}/config.yaml"
if [[ ! -f "${config_path}" ]]; then
    config_path="${run_dir}/.hydra/config.yaml"
fi
if [[ ! -f "${config_path}" ]]; then
    echo "Training exited without a persisted Hydra config: ${run_dir}" >&2
    exit 1
fi
for expected in \
    "^token_encoding: physical_time_type$" \
    "^stage_b_timestep_mode: legacy$" \
    "^seed: 3072$" \
    "^latent_head_dim: 192$" \
    "^stage_a_goal_injection: token$" \
    "^latent_action_mix_epochs: 10$" \
    "^detach_clean_action: false$" \
    "^stage_b_dynamics: parallel_prefix$" \
    "^stage_b_attention_mode: strict_causal$" \
    "^serial_activation_checkpointing: false$" \
    "^inference_steps: 10$"; do
    if ! rg -q "${expected}" "${config_path}"; then
        echo "Persisted Phase 3 config violates E5 physical-time/type registration: ${expected}" >&2
        exit 1
    fi
done

echo "checkpoint=${checkpoint}"
