#!/usr/bin/env bash
# Launch, monitor and analyze the Round 5 Phase 1 guidance experiments.
#
# Usage:
#   scripts/run_round5_phase1.sh start      # launch all shards on their GPUs
#   scripts/run_round5_phase1.sh status     # show running workers and result counts
#   scripts/run_round5_phase1.sh analyze    # regenerate the report from all results
#
# The runner is resume-safe: an already-complete condition is reused and never
# recomputed, so `start` can be called again at any time.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-config/round5/phase1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/round5/phase1_seed3072_legacy}"
LOG_DIR="$OUTPUT_ROOT/logs"
MIN_FREE_MIB="${MIN_FREE_MIB:-5000}"
NUM_SHARDS="${NUM_SHARDS:-2}"

# task:shard:physical_gpu  (two workers share GPU6 and GPU7; they are low-VRAM)
WORKERS=(
  "cube:0:7" "cube:1:6"
  "pusht:0:6" "pusht:1:4"
  "reacher:0:3" "reacher:1:2"
  "tworoom:0:0" "tworoom:1:7"
)

start() {
  mkdir -p "$LOG_DIR"
  for worker in "${WORKERS[@]}"; do
    IFS=":" read -r task shard gpu <<< "$worker"
    log="$LOG_DIR/${task}_s${shard}_gpu${gpu}.log"
    if pgrep -f "round5_phase1.py .*--task ${task} --shard-index ${shard} " >/dev/null 2>&1; then
      echo "already running: $task shard=$shard gpu=$gpu"
      continue
    fi
    CUDA_VISIBLE_DEVICES="$gpu" nohup python scripts/round5_phase1.py \
      --config "$CONFIG" \
      --output-root "$OUTPUT_ROOT" \
      --task "$task" \
      --shard-index "$shard" \
      --num-shards "$NUM_SHARDS" \
      --gpu "$gpu" \
      --min-free-mib "$MIN_FREE_MIB" \
      >"$log" 2>&1 &
    echo "started: $task shard=$shard gpu=$gpu pid=$! log=$log"
  done
}

status() {
  echo "== running round5 workers =="
  pgrep -af "round5_phase1.py" || echo "(none)"
  echo
  echo "== result counts =="
  local new_count baseline_count
  new_count=$(find "$OUTPUT_ROOT/conditions" -name result.json 2>/dev/null | wc -l)
  baseline_count=$(find "outputs/round4/phase45_4_seed3072_legacy_dev/conditions/legacy" -name result.json 2>/dev/null | wc -l)
  echo "new guidance results:   $new_count / 168"
  echo "reused baseline rows:   $baseline_count (phase45_4 legacy)"
  echo
  echo "== tail of newest logs =="
  ls -t "$LOG_DIR"/*.log 2>/dev/null | head -3 | while read -r log; do
    echo "--- $log"
    grep -E '"status": "(ok|reused)"|Traceback|RuntimeError|CUDA out of memory' "$log" | tail -2
  done
}

analyze() {
  python scripts/round5_phase1.py \
    --config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --analyze-only
}

case "${1:-start}" in
  start) start ;;
  status) status ;;
  analyze) analyze ;;
  *) echo "usage: $0 {start|status|analyze}" >&2; exit 1 ;;
esac
