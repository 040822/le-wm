#!/usr/bin/env bash
# Launch and monitor the independent Round 5 Phase 1 epoch-curve experiment.
#
# Usage:
#   scripts/run_round5_phase1_epoch_curve.sh dry-run
#   scripts/run_round5_phase1_epoch_curve.sh smoke
#   scripts/run_round5_phase1_epoch_curve.sh start
#   scripts/run_round5_phase1_epoch_curve.sh status
#   scripts/run_round5_phase1_epoch_curve.sh analyze
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-config/round5/phase1_epoch_curve.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/round5/phase1_epoch_curve_seed3072_legacy}"
REPORT_OUTPUT="${REPORT_OUTPUT:-docs/report/round5/round5_phase1_epoch_curve_report.md}"
LOG_DIR="$OUTPUT_ROOT/logs"
MIN_FREE_MIB="${MIN_FREE_MIB:-3500}"
DEVICE="${DEVICE:-cuda}"

# One worker per preferred GPU0-3.  The worker performs its own nvidia-smi
# preflight before importing/loading an R4-AB checkpoint.
WORKERS=(
  "cube:0:0"
  "pusht:0:1"
  "reacher:0:2"
  "tworoom:0:3"
)

dry_run() {
  python scripts/round5_phase1_epoch_curve.py dry-run \
    --config "$CONFIG" \
    --output-root "$OUTPUT_ROOT"
}

smoke() {
  mkdir -p "$LOG_DIR"
  for worker in "${WORKERS[@]}"; do
    IFS=":" read -r task shard gpu <<< "$worker"
    log="$LOG_DIR/${task}_smoke_gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/round5_phase1_epoch_curve.py worker \
      --config "$CONFIG" \
      --output-root "$OUTPUT_ROOT" \
      --task "$task" \
      --shard-index "$shard" \
      --num-shards 1 \
      --device "$DEVICE" \
      --gpu "$gpu" \
      --min-free-mib "$MIN_FREE_MIB" \
      --smoke \
      2>&1 | tee "$log"
  done
}

start() {
  mkdir -p "$LOG_DIR"
  for worker in "${WORKERS[@]}"; do
    IFS=":" read -r task shard gpu <<< "$worker"
    log="$LOG_DIR/${task}_s${shard}_gpu${gpu}.log"
    if pgrep -f "round5_phase1_epoch_curve.py worker .*--task ${task} .*--gpu ${gpu}" >/dev/null 2>&1; then
      echo "already running: $task shard=$shard gpu=$gpu"
      continue
    fi
    CUDA_VISIBLE_DEVICES="$gpu" nohup setsid python scripts/round5_phase1_epoch_curve.py worker \
      --config "$CONFIG" \
      --output-root "$OUTPUT_ROOT" \
      --task "$task" \
      --shard-index "$shard" \
      --num-shards 1 \
      --device "$DEVICE" \
      --gpu "$gpu" \
      --min-free-mib "$MIN_FREE_MIB" \
      </dev/null \
      >"$log" 2>&1 &
    echo "started: $task shard=$shard gpu=$gpu pid=$! log=$log"
  done
}

status() {
  echo "== running epoch-curve workers =="
  pgrep -af "round5_phase1_epoch_curve.py worker" || echo "(none)"
  echo
  echo "== result counts =="
  python - "$OUTPUT_ROOT" "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path
from source.common.round5_phase1_epoch_curve import job_specs

root = Path(sys.argv[1])
try:
    config = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    jobs = job_specs(config)
except Exception as exc:
    print(f"cannot read job manifest: {exc}")
    raise SystemExit(1)
results = list(root.glob("conditions/legacy/*/epoch_*/**/result.json"))
references = list(root.glob("conditions/legacy/*/epoch_10/**/reference.json"))
failures = list(root.glob("conditions/legacy/*/epoch_*/**/failure.json"))
print(f"evaluated result.json: {len(results)} / {sum(j['epoch'] != 10 for j in jobs)}")
print(f"epoch10 references:    {len(references)} / {sum(j['epoch'] == 10 for j in jobs)}")
print(f"failures:              {len(failures)}")
print(f"total completed:       {len(results) + len(references)} / {len(jobs)}")
PY
  echo
  echo "== GPU snapshots (preferred GPUs only) =="
  for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$gpu" nvidia-smi --id "$gpu" \
      --query-gpu=index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null || echo "GPU${gpu}: unavailable"
  done
}

analyze() {
  python scripts/round5_phase1_epoch_curve.py analyze \
    --config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --report-output "$REPORT_OUTPUT"
}

case "${1:-start}" in
  dry-run) dry_run ;;
  smoke) smoke ;;
  start) start ;;
  status) status ;;
  analyze) analyze ;;
  *) echo "usage: $0 {dry-run|smoke|start|status|analyze}" >&2; exit 1 ;;
esac
