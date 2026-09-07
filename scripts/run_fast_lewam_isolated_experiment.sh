#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <s32_e4|s41_e3> <task> <gpu_id>" >&2
    exit 2
fi

variant=$1
task=$2
gpu_id=$3

case "${gpu_id}" in
    0|1|2|3) ;;
    *)
        echo "GPU ${gpu_id} is prohibited; only GPU0-GPU3 are allowed." >&2
        exit 2
        ;;
esac

case "${variant}:${task}" in
    s32_e4:cube|s32_e4:reacher)
        info="s32_e4_clean_t_192"
        eval_stages=(stage_b)
        experiment_overrides=(
            "train_mode=stage_ab"
            "detach_clean_action=true"
            "latent_action_mix_epochs=10"
            "stage_b_timestep_mode=clean_action"
            "token_encoding=legacy"
        )
        ;;
    s41_e3:reacher|s41_e3:pusht)
        info="s41_e3_physical_time_type_192"
        eval_stages=(stage_a stage_a_shuffled_goal stage_b)
        experiment_overrides=(
            "train_mode=stage_ab"
            "detach_clean_action=false"
            "latent_action_mix_epochs=1e9"
            "stage_b_timestep_mode=legacy"
            "token_encoding=physical_time_type"
        )
        ;;
    *)
        echo "Unsupported experiment pair: ${variant}:${task}" >&2
        exit 2
        ;;
esac

run_date=${RUN_DATE:-$(date +"%m%d")}
if [[ ! "${run_date}" =~ ^[0-9]{4}$ ]]; then
    echo "RUN_DATE must contain exactly four digits, got: ${run_date}" >&2
    exit 2
fi
run_name="${run_date}_${info}"
run_dir="outputs/fast_lewam/${task}/${run_name}"

if [[ -e "${run_dir}" ]]; then
    echo "Refusing to reuse existing run directory: ${run_dir}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export HYDRA_FULL_ERROR=1
export STABLEWM_HOME="${STABLEWM_HOME:-$PWD/data}"
mkdir -p "${run_dir}"

memory_samples="${run_dir}/gpu_memory_samples.csv"
echo "timestamp,phase,memory_used_mib" > "${memory_samples}"

monitor_gpu_memory() {
    local process_id=$1
    local phase=$2
    local memory_used
    while kill -0 "${process_id}" 2>/dev/null; do
        memory_used=$(CUDA_VISIBLE_DEVICES="${gpu_id}" nvidia-smi \
            -i "${gpu_id}" \
            --query-gpu=memory.used \
            --format=csv,noheader,nounits)
        memory_used=${memory_used// /}
        echo "$(date +%s),${phase},${memory_used}" >> "${memory_samples}"
        sleep 30
    done
}

run_monitored() {
    local phase=$1
    shift
    "$@" &
    local process_id=$!
    monitor_gpu_memory "${process_id}" "${phase}" &
    local monitor_id=$!
    set +e
    wait "${process_id}"
    local status=$?
    wait "${monitor_id}"
    local monitor_status=$?
    set -e
    if [[ ${status} -ne 0 ]]; then
        return "${status}"
    fi
    if [[ ${monitor_status} -ne 0 ]]; then
        echo "GPU memory monitoring failed during ${phase}" >&2
        return "${monitor_status}"
    fi
}

common_overrides=(
    "data=${task}"
    "hydra.run.dir=${run_dir}"
    "info=${info}"
    "seed=3072"
    "latent_head_dim=192"
    "policy.model.mlp_dim=768"
    "loader.batch_size=128"
    "trainer.max_epochs=10"
    "trainer.devices=1"
    "epoch_eval.enabled=false"
    "wandb.enabled=false"
)

train_command=(
    python -u train.py
    --config-name=fast_lewam
    "${common_overrides[@]}"
    "${experiment_overrides[@]}"
)
eval_command=(
    python -u eval_fast_lewam.py
    "${run_dir}/checkpoints"
    --epochs 10
    --stages "${eval_stages[@]}"
)
start_seconds=$(date +%s)

echo "Launching ${variant}:${task} on visible GPU ${gpu_id}"
echo "Run directory: ${run_dir}"

run_monitored train "${train_command[@]}"
run_monitored eval "${eval_command[@]}"

elapsed_seconds=$(($(date +%s) - start_seconds))
python -u scripts/summarize_fast_lewam_experiment.py \
    "${run_dir}" \
    --elapsed-seconds "${elapsed_seconds}" \
    --stages "${eval_stages[@]}"

echo "Completed training and final evaluation: ${run_dir}"
