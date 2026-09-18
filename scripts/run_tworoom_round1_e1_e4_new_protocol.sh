#!/usr/bin/env bash
set -euo pipefail

# Round 1 TwoRoom E1-E4 training followed by the frozen Round 3 final test.
# The caller must provide one of the permitted physical GPUs; default to the
# currently idle GPU1.  With CUDA_VISIBLE_DEVICES=1, torch device cuda:0 is
# physical GPU1.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU_ID="${GPU_ID:-1}"
case "${GPU_ID}" in
    0|1|2|3) ;;
    *)
        echo "GPU ${GPU_ID} is prohibited; only GPU0-GPU3 are allowed." >&2
        exit 2
        ;;
esac

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export STABLEWM_HOME="${STABLEWM_HOME:-${ROOT}/data}"

DATE_TAG="${RUN_DATE:-$(date +%m%d)}"
RUN_PREFIX="${DATE_TAG}_round1_tworoom"
LOG_DIR="${ROOT}/outputs/round3/tworoom_e1_e4/logs"
EVAL_ROOT="${ROOT}/outputs/round3/extended/evals"
COHORT="${ROOT}/outputs/round3/phase1/cohorts/tworoom/final.json"
mkdir -p "${LOG_DIR}" "${EVAL_ROOT}"

if [[ ! -f "${COHORT}" ]]; then
    echo "Missing frozen final cohort: ${COHORT}" >&2
    exit 2
fi

run_logged() {
    local log_file="$1"
    shift
    echo "[$(date '+%F %T')] RUN $*" | tee -a "${log_file}" >&2
    "$@" 2>&1 | tee -a "${log_file}" >&2
}

result_is_complete() {
    local eval_dir="$1"
    local method="$2"
    local stage="$3"
    local result_file
    result_file="$(find "${eval_dir}/results/tworoom/${method}/round3_revised/${stage}/final" \
        -type f -name result.json -print -quit 2>/dev/null || true)"
    [[ -n "${result_file}" ]] || return 1
    jq -e '.status == "ok" and (.episodes | length) == 200' "${result_file}" >/dev/null
}

train_if_needed() {
    local experiment="$1"
    local run_name="$2"
    shift 2
    local run_dir="${ROOT}/outputs/fast_lewam/tworoom/${run_name}"
    local checkpoint="${run_dir}/checkpoints/fast_lewam_weights_epoch_10.pt"
    local log_file="${LOG_DIR}/${experiment}_${run_name}_train.log"

    if [[ -e "${run_dir}" && ! -f "${checkpoint}" ]]; then
        echo "Refusing to reuse incomplete run directory: ${run_dir}" >&2
        exit 2
    fi
    if [[ ! -f "${checkpoint}" ]]; then
        run_logged "${log_file}" \
            "${ROOT}/.venv/bin/python" -u train.py \
            --config-name=fast_lewam \
            data=tworoom \
            "hydra.run.dir=${run_dir}" \
            "info=round1_${experiment}" \
            seed=3072 \
            latent_head_dim=192 \
            policy.model.mlp_dim=768 \
            loader.batch_size=128 \
            trainer.max_epochs=10 \
            trainer.devices=1 \
            trainer.accelerator=gpu \
            epoch_eval.enabled=false \
            wandb.enabled=false \
            "$@"
    else
        echo "[$(date '+%F %T')] REUSE completed checkpoint ${checkpoint}" | tee -a "${log_file}" >&2
    fi
    printf '%s\n' "${checkpoint}"
}

evaluate_stage() {
    local experiment="$1"
    local run_name="$2"
    local method="$3"
    local stage="$4"
    local checkpoint="$5"
    local eval_dir="${EVAL_ROOT}/round1_tworoom_${experiment}_${run_name}"
    local log_file="${LOG_DIR}/${experiment}_${run_name}_${stage}_new_protocol.log"

    if result_is_complete "${eval_dir}" "${method}" "${stage}"; then
        echo "[$(date '+%F %T')] REUSE complete evaluation ${experiment} ${stage}" | tee -a "${log_file}"
        return 0
    fi
    run_logged "${log_file}" \
        env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        "${ROOT}/.venv/bin/python" -u -m scripts.round3_phase1 evaluate \
        tworoom "${method}" "${stage}" \
        --protocol-variant round3_revised \
        --cohort "${COHORT}" \
        --checkpoint "${checkpoint}" \
        --epoch 10 \
        --device cuda:0 \
        --output "${eval_dir}"
}

run_experiment() {
    local experiment="$1"
    local suffix="$2"
    local method="$3"
    local stages="$4"
    shift 4
    local run_name="${RUN_PREFIX}_${suffix}"
    local checkpoint
    checkpoint="$(train_if_needed "${experiment}" "${run_name}" "$@")"
    for stage in ${stages}; do
        evaluate_stage "${experiment}" "${run_name}" "${method}" "${stage}" "${checkpoint}"
    done
}

echo "[$(date '+%F %T')] Starting TwoRoom E1-E4 on physical GPU${GPU_ID}"

run_experiment E1 e1_stage_b_only e1_fast "stage_b" \
    train_mode=stage_b \
    latent_action_mix_epochs=10 \
    detach_clean_action=false \
    stage_b_timestep_mode=legacy \
    token_encoding=legacy

run_experiment E2 e2_stage_a_only e2_fast "stage_a stage_a_shuffled_goal" \
    train_mode=stage_ab \
    loss.latent.weight=0 \
    latent_action_mix_epochs=10 \
    detach_clean_action=false \
    stage_b_timestep_mode=legacy \
    token_encoding=legacy

run_experiment E3 e3_expert_actions e3_fast "stage_a stage_a_shuffled_goal stage_b" \
    train_mode=stage_ab \
    latent_action_mix_epochs=1e9 \
    detach_clean_action=false \
    stage_b_timestep_mode=legacy \
    token_encoding=legacy

run_experiment E4 e4_detach e4_fast "stage_a stage_a_shuffled_goal stage_b" \
    train_mode=stage_ab \
    latent_action_mix_epochs=10 \
    detach_clean_action=true \
    stage_b_timestep_mode=legacy \
    token_encoding=legacy

echo "[$(date '+%F %T')] Completed TwoRoom E1-E4 training and round3_revised evaluation"
