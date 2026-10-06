#!/usr/bin/env bash
# Wait for the independently trained Phase 3 Finger LeWM model and run its
# standard baseline evaluation plus the requested representative video.
#
# This watcher intentionally handles only Finger.  The remaining Phase 3
# tasks stay untouched so the goal can be paused after this evaluation and
# resumed manually later.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
OUTPUT_ROOT="$ROOT/outputs/round5/phase3"
RAW_RUN="$ROOT/outputs/20260920_120937_078324_lewm_data_datasets_dmcontrol_finger_turn_hard_2853076"
RESUME_RUN="$ROOT/outputs/round5_phase3_lewm_finger_resume_epoch9_tmux"
ARCHIVE="$OUTPUT_ROOT/training/lewm/finger"
LEWM_CHECKPOINT="$ARCHIVE/checkpoints/lewm_weights_epoch_10.pt"
LOG_ROOT="$OUTPUT_ROOT/logs"
WATCH_LOG="$LOG_ROOT/watch_finger_eval.log"
DONE_MARKER="$LOG_ROOT/finger_eval_complete.json"
FAIL_MARKER="$LOG_ROOT/finger_eval_failed.json"
POLL_SECONDS="${POLL_SECONDS:-300}"
MIN_EVAL_FREE_MIB="${MIN_EVAL_FREE_MIB:-10000}"

mkdir -p "$LOG_ROOT"
exec >>"$WATCH_LOG" 2>&1

echo "[$(date -Iseconds)] watcher started; raw_run=$RAW_RUN"
if [[ -f "$DONE_MARKER" ]]; then
  echo "[$(date -Iseconds)] completion marker already exists; exiting"
  exit 0
fi

SOURCE_RUN=""
while :; do
  if [[ -f "$RAW_RUN/checkpoints/lewm_weights_epoch_10.pt" ]]; then
    SOURCE_RUN="$RAW_RUN"
    break
  fi
  if [[ -f "$RESUME_RUN/checkpoints/lewm_weights_epoch_10.pt" ]]; then
    SOURCE_RUN="$RESUME_RUN"
    break
  fi
  echo "[$(date -Iseconds)] waiting for Finger LeWM epoch10 checkpoint"
  sleep "$POLL_SECONDS"
done

echo "[$(date -Iseconds)] epoch10 checkpoint found in $SOURCE_RUN"

# Do not compete with the final validation/checkpoint writer for the training
# card.  The checkpoint file can become visible before Lightning releases the
# CUDA context.
while pgrep -f -- "[p]hase3_finger_resume_epoch9.py" >/dev/null 2>&1; do
  echo "[$(date -Iseconds)] epoch10 exists but resume process is still releasing resources"
  sleep 60
done

# The raw run is written outside the stable Phase 3 layout.  Archive it only
# after epoch10 exists, then evaluate the archived checkpoint for traceability.
mkdir -p "$ARCHIVE"
cp -a "$SOURCE_RUN/." "$ARCHIVE/"
# Keep the original run directory discoverable by the existing Phase 3
# launchers as well.  The resume run is explicitly recorded in the archive's
# metadata and does not overwrite earlier epoch files.
if [[ "$SOURCE_RUN" != "$RAW_RUN" ]]; then
  cp -f "$SOURCE_RUN/checkpoints/lewm_weights_epoch_10.pt" "$RAW_RUN/checkpoints/"
  for artifact in lewm_policy.ckpt lewm_object.ckpt; do
    if [[ -f "$SOURCE_RUN/checkpoints/$artifact" ]]; then
      cp -f "$SOURCE_RUN/checkpoints/$artifact" "$RAW_RUN/checkpoints/$artifact"
    fi
  done
  printf '{"source_run":"%s","initial_epoch":9,"optimizer_state_restored":false}\n' "$SOURCE_RUN" >"$ARCHIVE/resume_metadata.json"
fi
test -f "$LEWM_CHECKPOINT"
echo "[$(date -Iseconds)] archived checkpoint to $LEWM_CHECKPOINT"

choose_gpu() {
  # Prefer a card outside the current training assignments.  Every nvidia-smi
  # invocation is explicitly restricted to the candidate physical device.
  local gpu free
  for gpu in 3 2 7 6 5 1 0 4; do
    free="$(CUDA_VISIBLE_DEVICES="$gpu" nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= MIN_EVAL_FREE_MIB )); then
      echo "$gpu"
      return 0
    fi
    echo "[$(date -Iseconds)] GPU${gpu} has ${free:-unknown} MiB free; below evaluation threshold" >&2
  done
  return 1
}

GPU=""
while [[ -z "$GPU" ]]; do
  GPU="$(choose_gpu || true)"
  if [[ -z "$GPU" ]]; then
    echo "[$(date -Iseconds)] no evaluation GPU has ${MIN_EVAL_FREE_MIB} MiB free; retrying"
    sleep "$POLL_SECONDS"
  fi
done
echo "[$(date -Iseconds)] selected evaluation GPU${GPU}"

run_eval() {
  echo "[$(date -Iseconds)] starting LeWM Finger standard evaluation on GPU${GPU}"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u scripts/round5_phase3.py \
    --output-root "$OUTPUT_ROOT" eval-baseline finger lewm \
    --checkpoint "$LEWM_CHECKPOINT" --device cuda --gpu "$GPU"
  echo "[$(date -Iseconds)] LeWM Finger standard evaluation finished"

  echo "[$(date -Iseconds)] starting LeWM Finger representative video on GPU${GPU}"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u scripts/round5_phase3.py \
    --output-root "$OUTPUT_ROOT" eval-video finger lewm \
    --condition standard --checkpoint "$LEWM_CHECKPOINT" \
    --device cuda --gpu "$GPU"
  echo "[$(date -Iseconds)] LeWM Finger representative video finished"
}

if run_eval; then
  :
else
  status=$?
  printf '{"status":"failed","task":"finger","method":"lewm","gpu":%s,"time":"%s"}\n' \
    "$GPU" "$(date -Iseconds)" >"$FAIL_MARKER"
  echo "[$(date -Iseconds)] evaluation failed with status $status"
  exit "$status"
fi

"$PYTHON" - "$OUTPUT_ROOT" "$GPU" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
gpu = int(sys.argv[2])
fast = root / "fastlewam/finger/P3/not_applicable/none/step_1/euler/result.json"
lewm = root / "baselines/lewm/finger/result.json"
payload = {
    "status": "ok",
    "task": "finger",
    "method": "lewm",
    "gpu": gpu,
    "fastlewam_primary_result": str(fast),
    "lewm_result": str(lewm),
}
if fast.is_file():
    fast_data = json.loads(fast.read_text())
    payload["fastlewam_successes"] = sum(bool(row.get("success")) for row in fast_data.get("episodes", []))
if lewm.is_file():
    lewm_data = json.loads(lewm.read_text())
    payload["lewm_successes"] = sum(bool(row.get("success")) for row in lewm_data.get("episodes", []))
(root / "logs/finger_eval_complete.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "[$(date -Iseconds)] Finger evaluation and video complete; goal may now be paused"
