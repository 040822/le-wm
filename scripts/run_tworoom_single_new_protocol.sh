#!/usr/bin/env bash
set -euo pipefail

# Run one TwoRoom Round 1 experiment and its round3_revised evaluations.
# The caller supplies one permitted physical GPU through GPU_ID.  With
# CUDA_VISIBLE_DEVICES set to that GPU, the evaluator's cuda:0 is physical
# GPU_ID.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

EXPERIMENT="${1:?usage: GPU_ID=<0-3> $0 E2|E3|E4}"
GPU_ID="${GPU_ID:?GPU_ID must be set to 0, 1, 2, or 3}"

case "${GPU_ID}" in
    0|1|2|3) ;;
    *)
        echo "GPU ${GPU_ID} is prohibited; only GPU0-GPU3 are allowed." >&2
        exit 2
        ;;
esac

case "${EXPERIMENT}" in
    E2)
        SUFFIX="e2_stage_a_only"
        METHOD="e2_fast"
        STAGES=(stage_a stage_a_shuffled_goal)
        TRAIN_ARGS=(
            train_mode=stage_ab
            loss.latent.weight=0
            latent_action_mix_epochs=10
            detach_clean_action=false
            stage_b_timestep_mode=legacy
            token_encoding=legacy
        )
        ;;
    E3)
        SUFFIX="e3_expert_actions"
        METHOD="e3_fast"
        STAGES=(stage_a stage_a_shuffled_goal stage_b)
        TRAIN_ARGS=(
            train_mode=stage_ab
            # Hydra passes .inf as a string; use a very large finite value so
            # the model's numeric comparison keeps expert-action mixing on.
            latent_action_mix_epochs=1e9
            detach_clean_action=false
            stage_b_timestep_mode=legacy
            token_encoding=legacy
        )
        ;;
    E4)
        SUFFIX="e4_detach"
        METHOD="e4_fast"
        STAGES=(stage_a stage_a_shuffled_goal stage_b)
        TRAIN_ARGS=(
            train_mode=stage_ab
            latent_action_mix_epochs=10
            detach_clean_action=true
            stage_b_timestep_mode=legacy
            token_encoding=legacy
        )
        ;;
    *)
        echo "Only E2, E3, and E4 can be launched by this parallel runner." >&2
        exit 2
        ;;
esac

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export STABLEWM_HOME="${STABLEWM_HOME:-${ROOT}/data}"

# Keep the same run-date namespace as the already running E1 job unless the
# caller explicitly overrides it.
DATE_TAG="${RUN_DATE:-0910}"
RUN_NAME="${DATE_TAG}_round1_tworoom_${SUFFIX}"
RUN_DIR="${ROOT}/outputs/fast_lewam/tworoom/${RUN_NAME}"
LOG_DIR="${ROOT}/outputs/round3/tworoom_e1_e4/logs"
EVAL_ROOT="${ROOT}/outputs/round3/extended/evals"
COHORT="${ROOT}/outputs/round3/phase1/cohorts/tworoom/final.json"
CHECKPOINT="${RUN_DIR}/checkpoints/fast_lewam_weights_epoch_10.pt"
TRAIN_LOG="${LOG_DIR}/${EXPERIMENT}_${RUN_NAME}_train.log"

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
    local stage="$2"
    local result_file
    result_file="$(find "${eval_dir}/results/tworoom/${METHOD}/round3_revised/${stage}/final" \
        -type f -name result.json -print -quit 2>/dev/null || true)"
    [[ -n "${result_file}" ]] || return 1
    jq -e '.status == "ok" and (.episodes | length) == 200' "${result_file}" >/dev/null
}

if [[ -e "${RUN_DIR}" && ! -f "${CHECKPOINT}" ]]; then
    echo "Refusing to reuse incomplete run directory: ${RUN_DIR}" >&2
    exit 2
fi

echo "[$(date '+%F %T')] Starting ${EXPERIMENT} on physical GPU${GPU_ID}" | tee -a "${TRAIN_LOG}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    run_logged "${TRAIN_LOG}" \
        "${ROOT}/.venv/bin/python" -u train.py \
        --config-name=fast_lewam \
        data=tworoom \
        "hydra.run.dir=${RUN_DIR}" \
        "info=round1_${EXPERIMENT}" \
        seed=3072 \
        latent_head_dim=192 \
        policy.model.mlp_dim=768 \
        loader.batch_size=128 \
        trainer.max_epochs=10 \
        trainer.devices=1 \
        trainer.accelerator=gpu \
        epoch_eval.enabled=false \
        wandb.enabled=false \
        "${TRAIN_ARGS[@]}"
else
    echo "[$(date '+%F %T')] REUSE completed checkpoint ${CHECKPOINT}" | tee -a "${TRAIN_LOG}"
fi

EVAL_DIR="${EVAL_ROOT}/round1_tworoom_${EXPERIMENT}_${RUN_NAME}"
for STAGE in "${STAGES[@]}"; do
    EVAL_LOG="${LOG_DIR}/${EXPERIMENT}_${RUN_NAME}_${STAGE}_new_protocol.log"
    if result_is_complete "${EVAL_DIR}" "${STAGE}"; then
        echo "[$(date '+%F %T')] REUSE complete evaluation ${EXPERIMENT} ${STAGE}" | tee -a "${EVAL_LOG}"
        continue
    fi
    run_logged "${EVAL_LOG}" \
        env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        "${ROOT}/.venv/bin/python" -u -m scripts.round3_phase1 evaluate \
        tworoom "${METHOD}" "${STAGE}" \
        --protocol-variant round3_revised \
        --cohort "${COHORT}" \
        --checkpoint "${CHECKPOINT}" \
        --epoch 10 \
        --device cuda:0 \
        --output "${EVAL_DIR}"
done

echo "[$(date '+%F %T')] Completed ${EXPERIMENT} training and round3_revised evaluation" | tee -a "${TRAIN_LOG}"
