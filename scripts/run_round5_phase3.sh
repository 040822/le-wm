#!/usr/bin/env bash
# Launch, monitor and analyze the Round 5 Phase 3 matrix.
#
# Training is deliberately restricted to physical GPU0--3.  Evaluation is
# launched by round5_phase3.py and may use GPU0--7 when its preflight passes.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/round5/phase3}"
TRAIN_ROOT="$OUTPUT_ROOT/training"
LOG_ROOT="$OUTPUT_ROOT/logs"
MIN_TRAIN_FREE_MIB="${MIN_TRAIN_FREE_MIB:-12000}"

# The current host has unrelated workloads on all eight cards.  These are the
# Phase 3 training assignments; start() checks live free memory before each
# launch and leaves a task pending when the card is too full.
TASKS=(scene finger humanoid)
GPUS=(2 0 1)

gpu_free_mib() {
  local gpu="$1"
  nvidia-smi --id="$gpu" \
    --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]'
}

launch() {
  local kind="$1" task="$2" gpu="$3" command="$4" log="$5"
  local free
  free="$(gpu_free_mib "$gpu")" || {
    echo "cannot inspect GPU${gpu}; skipping ${kind}/${task}" >&2
    return 1
  }
  if [ "$free" -lt "$MIN_TRAIN_FREE_MIB" ]; then
    echo "GPU${gpu} has ${free} MiB free (< ${MIN_TRAIN_FREE_MIB}); pending ${kind}/${task}"
    return 0
  fi
  if pgrep -f -- "$command" >/dev/null 2>&1; then
    echo "already running: ${kind}/${task} on GPU${gpu}"
    return 0
  fi
  mkdir -p "$(dirname "$log")"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" MPLCONFIGDIR="/tmp/mpl-phase3-${task}" \
    nohup bash -lc "$command" >"$log" 2>&1 &
  echo "started: ${kind}/${task} GPU${gpu} pid=$! log=${log} free_mib=${free}"
}

start_fast() {
  for index in "${!TASKS[@]}"; do
    local task="${TASKS[$index]}" gpu="${GPUS[$index]}"
    local run_dir="$TRAIN_ROOT/fastlewam/$task"
    local checkpoint="$run_dir/checkpoints/r4_ab_weights_epoch_10.pt"
    local log="$LOG_ROOT/train_fastlewam_${task}_gpu${gpu}.log"
    if [ -f "$checkpoint" ]; then
      echo "reused: FastLeWAM/${task} checkpoint=$checkpoint"
      continue
    fi
    local command="cd '$ROOT' && '$PYTHON' -u train.py --config-name=round4_ab data=$task wandb.enabled=false hydra.run.dir='$run_dir' hydra.job.chdir=false"
    launch fastlewam "$task" "$gpu" "$command" "$log"
  done
}

start_lewm() {
  for index in "${!TASKS[@]}"; do
    local task="${TASKS[$index]}" gpu="${GPUS[$index]}"
    local run_dir="$TRAIN_ROOT/lewm/$task"
    local checkpoint="$run_dir/checkpoints/lewm_weights_epoch_10.pt"
    local log="$LOG_ROOT/train_lewm_${task}_gpu${gpu}.log"
    if [ -f "$checkpoint" ]; then
      echo "reused: LeWM/${task} checkpoint=$checkpoint"
      continue
    fi
    local command="cd '$ROOT' && '$PYTHON' -u train.py --config-name=lewm data=$task wandb.enabled=false hydra.run.dir='$run_dir' hydra.job.chdir=false"
    launch lewm "$task" "$gpu" "$command" "$log"
  done
}

start_leflow() {
  for index in "${!TASKS[@]}"; do
    local task="${TASKS[$index]}" gpu="${GPUS[$index]}"
    local lewm="$TRAIN_ROOT/lewm/$task/checkpoints/lewm_weights_epoch_10.pt"
    local out="$TRAIN_ROOT/leflow/$task"
    local checkpoint="$out/latent_planner.pt"
    local log="$LOG_ROOT/train_leflow_${task}_gpu${gpu}.log"
    if [ ! -f "$lewm" ]; then
      echo "pending: LeWM/${task} checkpoint missing: $lewm"
      continue
    fi
    if [ -f "$checkpoint" ]; then
      echo "reused: LeFlow/${task} checkpoint=$checkpoint"
      continue
    fi
    local command="cd '$ROOT' && '$PYTHON' -u scripts/train_round5_phase3_leflow.py '$task' --lewm-checkpoint '$lewm' --output '$out' --device cuda"
    launch leflow "$task" "$gpu" "$command" "$log"
  done
}

status() {
  echo "== Phase 3 processes =="
  pgrep -af 'train.py --config-name=(round4_ab|lewm)|train_round5_phase3_leflow.py' || true
  echo
  echo "== checkpoints =="
  for task in "${TASKS[@]}"; do
    printf '%-10s Fast=%s LeWM=%s LeFlow=%s\n' "$task" \
      "$(test -f "$TRAIN_ROOT/fastlewam/$task/checkpoints/r4_ab_weights_epoch_10.pt" && echo yes || echo no)" \
      "$(test -f "$TRAIN_ROOT/lewm/$task/checkpoints/lewm_weights_epoch_10.pt" && echo yes || echo no)" \
      "$(test -f "$TRAIN_ROOT/leflow/$task/latent_planner.pt" && echo yes || echo no)"
  done
  echo
  CUDA_VISIBLE_DEVICES=0,1,2,3 nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true
}

analyze() {
  "$PYTHON" scripts/round5_phase3.py --output-root "$OUTPUT_ROOT" analyze
}

case "${1:-status}" in
  prepare) "$PYTHON" scripts/round5_phase3.py --output-root "$OUTPUT_ROOT" prepare ;;
  start-fast) start_fast ;;
  start-lewm) start_lewm ;;
  start-leflow) start_leflow ;;
  status) status ;;
  analyze) analyze ;;
  *) echo "usage: $0 {prepare|start-fast|start-lewm|start-leflow|status|analyze}" >&2; exit 1 ;;
esac
