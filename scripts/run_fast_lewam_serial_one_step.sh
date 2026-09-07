#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <reacher|pusht> <gpu_id> [--preflight-only]" >&2
    exit 2
fi

task=$1
gpu_id=$2
mode=${3:-}

case "${task}" in
    reacher|pusht) ;;
    *)
        echo "Unsupported task: ${task}; expected reacher or pusht." >&2
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
case "${task}:${gpu_id}" in
    reacher:1|pusht:2) ;;
    *)
        echo "§3.4 requires Reacher on GPU1 and Push-T on GPU2." >&2
        exit 2
        ;;
esac
if [[ -n "${mode}" && "${mode}" != "--preflight-only" ]]; then
    echo "Unknown option: ${mode}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export MUJOCO_EGL_DEVICE_ID="${gpu_id}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1
export STABLEWM_HOME="${STABLEWM_HOME:-$PWD/data}"

free_mib=$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if (( free_mib < 4096 )); then
    echo "GPU${gpu_id} has only ${free_mib} MiB free; at least 4096 MiB is required." >&2
    exit 1
fi

run_date=${RUN_DATE:-$(date +"%m%d")}
if [[ ! "${run_date}" =~ ^[0-9]{4}$ ]]; then
    echo "RUN_DATE must contain exactly four digits, got: ${run_date}" >&2
    exit 2
fi
run_dir="outputs/fast_lewam/${task}/${run_date}_s34_e1_serial_one_step_192"
if [[ -e "${run_dir}" ]]; then
    echo "Refusing to reuse existing run directory: ${run_dir}" >&2
    exit 2
fi

preflight_root=$(mktemp -d "/tmp/fast_lewam_s34_${task}_XXXXXX")
preflight_samples="${preflight_root}/gpu_memory_samples.csv"
echo "timestamp,memory_used_mib" > "${preflight_samples}"

monitor_memory() {
    local process_id=$1
    local samples=$2
    local memory_used
    while kill -0 "${process_id}" 2>/dev/null; do
        memory_used=$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits)
        memory_used=${memory_used// /}
        echo "$(date +%s),${memory_used}" >> "${samples}"
        sleep 1
    done
}

run_preflight() {
    local activation_checkpointing=$1
    local leaf="${preflight_root}/checkpoint_${activation_checkpointing}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python -u train.py \
        --config-name=fast_lewam \
        "data=${task}" \
        train_mode=stage_b \
        stage_b_dynamics=serial_one_step \
        "serial_activation_checkpointing=${activation_checkpointing}" \
        seed=3072 \
        latent_head_dim=192 \
        policy.model.mlp_dim=768 \
        loader.batch_size=128 \
        loader.num_workers=0 \
        loader.persistent_workers=false \
        '~loader.prefetch_factor' \
        +trainer.fast_dev_run=true \
        trainer.devices=1 \
        epoch_eval.enabled=false \
        wandb.enabled=false \
        +policy.scheduler.type=ConstantLR \
        +policy.scheduler.factor=1.0 \
        +policy.scheduler.total_iters=1 \
        "hydra.run.dir=${leaf}"
}

checkpointing=false
set +e
run_preflight false &
preflight_pid=$!
monitor_memory "${preflight_pid}" "${preflight_samples}" &
monitor_pid=$!
wait "${preflight_pid}"
preflight_status=$?
wait "${monitor_pid}"
set -e

total_mib=$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
peak_mib=$(awk -F, 'NR > 1 && $2 > peak {peak=$2} END {print peak+0}' "${preflight_samples}")
threshold_mib=$((total_mib * 90 / 100))
if (( preflight_status != 0 || peak_mib == 0 || peak_mib >= threshold_mib )); then
    checkpointing=true
    echo "Retrying preflight with serial activation checkpointing."
    echo "timestamp,memory_used_mib" > "${preflight_samples}"
    set +e
    run_preflight true &
    preflight_pid=$!
    monitor_memory "${preflight_pid}" "${preflight_samples}" &
    monitor_pid=$!
    wait "${preflight_pid}"
    preflight_status=$?
    wait "${monitor_pid}"
    set -e
    peak_mib=$(awk -F, 'NR > 1 && $2 > peak {peak=$2} END {print peak+0}' "${preflight_samples}")
fi
if (( preflight_status != 0 || peak_mib == 0 || peak_mib >= threshold_mib )); then
    echo "Serial preflight is unsafe: status=${preflight_status}, peak=${peak_mib} MiB, threshold=${threshold_mib} MiB." >&2
    exit 1
fi
echo "Preflight passed on GPU${gpu_id}: peak=${peak_mib} MiB, checkpointing=${checkpointing}."
if [[ "${mode}" == "--preflight-only" ]]; then
    exit 0
fi

mkdir -p "${run_dir}"
memory_samples="${run_dir}/gpu_memory_samples.csv"
echo "timestamp,memory_used_mib" > "${memory_samples}"

train_command=(
    python -u train.py
    --config-name=fast_lewam
    "data=${task}"
    "hydra.run.dir=${run_dir}"
    info=s34_e1_serial_one_step_192
    seed=3072
    train_mode=stage_b
    stage_b_dynamics=serial_one_step
    "serial_activation_checkpointing=${checkpointing}"
    latent_head_dim=192
    policy.model.mlp_dim=768
    loader.batch_size=128
    trainer.max_epochs=10
    trainer.devices=1
    epoch_eval.enabled=false
    wandb.enabled=false
)

start_seconds=$(date +%s)
"${train_command[@]}" &
train_pid=$!
monitor_memory "${train_pid}" "${memory_samples}" &
monitor_pid=$!
set +e
wait "${train_pid}"
train_status=$?
wait "${monitor_pid}"
set -e
if (( train_status != 0 )); then
    exit "${train_status}"
fi
training_peak_mib=$(awk -F, 'NR > 1 && $2 > peak {peak=$2} END {print peak+0}' "${memory_samples}")
if (( training_peak_mib == 0 || training_peak_mib >= threshold_mib )); then
    echo "Training memory acceptance failed: peak=${training_peak_mib} MiB, threshold=${threshold_mib} MiB." >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${gpu_id}" python -u eval_fast_lewam.py \
    "${run_dir}/checkpoints" \
    --epochs 2 4 6 8 10 \
    --stages stage_b

elapsed_seconds=$(($(date +%s) - start_seconds))
python -u scripts/summarize_fast_lewam_experiment.py \
    "${run_dir}" \
    --elapsed-seconds "${elapsed_seconds}" \
    --stages stage_b

echo "Completed §3.4 serial control: ${run_dir}"
