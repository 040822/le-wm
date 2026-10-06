#!/usr/bin/env bash
# Finish Round 5 Phase 3 with training limited to GPU0/1.
set -euo pipefail
ROOT="$(cd "$(dirname "$BASH_SOURCE")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
OUTPUT_ROOT="$ROOT/outputs/round5/phase3"
LOG_ROOT="$OUTPUT_ROOT/logs"
POLL_SECONDS=300
MIN_TRAIN_FREE_MIB=12000
MIN_EVAL_FREE_MIB=5000
WATCH_LOG="$LOG_ROOT/baseline_pipeline_watch.log"
DONE_MARKER="$LOG_ROOT/baseline_pipeline_complete.json"
SCENE_RECHECK_ROOT="$ROOT/outputs/round5/phase3_scene_goalrow_v2"
mkdir -p "$LOG_ROOT"
exec >>"$WATCH_LOG" 2>&1
timestamp() { date -Iseconds; }
echo "[$(timestamp)] baseline pipeline started"
if [[ -f "$DONE_MARKER" ]]; then echo "[$(timestamp)] already complete"; exit 0; fi

complete() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json,sys
from pathlib import Path
try: d=json.loads(Path(sys.argv[1]).read_text())
except (OSError,json.JSONDecodeError): raise SystemExit(1)
raise SystemExit(0 if d.get("status")=="ok" and len(d.get("episodes",[]))==int(sys.argv[2]) else 1)
PY
}
free_mib() {
  CUDA_VISIBLE_DEVICES="$1" nvidia-smi --id="$1" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]'
}
eval_gpu() {
  local gpu free
  for gpu in 2 3 0 1 4 5 6 7; do
    free="$(free_mib "$gpu" 2>/dev/null || true)"
    if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= MIN_EVAL_FREE_MIB )); then echo "$gpu"; return 0; fi
  done
  return 1
}
train_gpu() {
  local gpu free
  for gpu in 0 1; do
    free="$(free_mib "$gpu" 2>/dev/null || true)"
    if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= MIN_TRAIN_FREE_MIB )); then echo "$gpu"; return 0; fi
  done
  return 1
}
preserve_partial() {
  local target="$1" tag="$2"
  [[ -d "$target" ]] || return 0
  [[ -n "$(find "$target" -mindepth 1 -maxdepth 1 -print -quit)" ]] || return 0
  local archive="$OUTPUT_ROOT/incomplete_archive/"$tag"_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$(dirname "$archive")"
  mv "$target" "$archive"
  echo "[$(timestamp)] preserved partial artifact: $archive"
}
sync_scene() {
  local source="$SCENE_RECHECK_ROOT" marker="$LOG_ROOT/scene_goalrow_sync.json"
  [[ -f "$marker" ]] && return 0
  "$PYTHON" - "$OUTPUT_ROOT" "$source" <<'PY'
import json,sys
from pathlib import Path
main,source=map(Path,sys.argv[1:])
def h(root):
 d=json.loads((root/"cohorts/scene/legacy_50.json").read_text())
 return d.get("cohort_sha256") or d.get("manifest_sha256")
assert h(main)==h(source),"Scene cohort mismatch"
results=list((source/"fastlewam/scene").glob("**/result.json"))
assert len(results)==61,f"expected 61 corrected Fast results, found {len(results)}"
for p in results:
 d=json.loads(p.read_text()); assert d.get("status")=="ok" and len(d.get("episodes",[]))==50,p
d=json.loads((source/"baselines/lewm/scene/result.json").read_text())
assert d.get("status")=="ok" and len(d.get("episodes",[]))==50
for p in ("videos/fastlewam/scene/p1_step_1/videos/env_0.mp4","videos/fastlewam/scene/p3_step_1/videos/env_0.mp4","videos/lewm/scene/standard/videos/env_0.mp4"):
 assert (source/p).is_file(),p
PY
  local archive="$OUTPUT_ROOT/legacy_scene_target_mismatch" rel
  for rel in fastlewam/scene baselines/lewm/scene videos/fastlewam/scene videos/lewm/scene; do
    if [[ -e "$OUTPUT_ROOT/$rel" && ! -e "$archive/$rel" ]]; then
      mkdir -p "$archive/$(dirname "$rel")"; mv "$OUTPUT_ROOT/$rel" "$archive/$rel"
    fi
  done
  mkdir -p "$OUTPUT_ROOT/fastlewam" "$OUTPUT_ROOT/baselines/lewm" "$OUTPUT_ROOT/videos/fastlewam" "$OUTPUT_ROOT/videos/lewm"
  cp -a "$source/fastlewam/scene" "$OUTPUT_ROOT/fastlewam/"
  cp -a "$source/baselines/lewm/scene" "$OUTPUT_ROOT/baselines/lewm/"
  cp -a "$source/videos/fastlewam/scene" "$OUTPUT_ROOT/videos/fastlewam/"
  cp -a "$source/videos/lewm/scene" "$OUTPUT_ROOT/videos/lewm/"
  "$PYTHON" - "$marker" "$source" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
marker,source=map(Path,sys.argv[1:])
d=json.loads((source/"cohorts/scene/legacy_50.json").read_text())
marker.write_text(json.dumps({"status":"ok","source":str(source),"cohort_sha256":d.get("cohort_sha256") or d.get("manifest_sha256"),"legacy_scene_preserved_at":str(source.parent/"phase3/legacy_scene_target_mismatch"),"synchronized_at":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")
PY
  echo "[$(timestamp)] synchronized corrected Scene results and videos; prior outputs archived"
}
video() {
  local task="$1" method="$2" condition="$3" checkpoint="$4"
  local target="$OUTPUT_ROOT/videos/$method/$task/$condition" path="$OUTPUT_ROOT/videos/$method/$task/$condition/videos/env_0.mp4" gpu
  if [[ -s "$path" ]] && complete "$target/result.json" 1; then return 0; fi
  preserve_partial "$target" "video_"$method"_"$task"_"$condition
  gpu="$(eval_gpu || true)"
  while [[ -z "$gpu" ]]; do echo "[$(timestamp)] waiting for GPU for video $method/$task"; sleep "$POLL_SECONDS"; gpu="$(eval_gpu || true)"; done
  echo "[$(timestamp)] video $method/$task/$condition on GPU$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$ROOT/scripts/round5_phase3.py" --output-root "$OUTPUT_ROOT" eval-video "$task" "$method" --condition "$condition" --checkpoint "$checkpoint" --device cuda --gpu "$gpu"
}
baseline() {
  local task="$1" method="$2" checkpoint="$3"
  local target="$OUTPUT_ROOT/baselines/$method/$task" gpu
  if ! complete "$target/result.json" 50; then
    preserve_partial "$target" "baseline_"$method"_"$task
    gpu="$(eval_gpu || true)"
    while [[ -z "$gpu" ]]; do echo "[$(timestamp)] waiting for GPU for $method/$task"; sleep "$POLL_SECONDS"; gpu="$(eval_gpu || true)"; done
    echo "[$(timestamp)] eval $method/$task on GPU$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$ROOT/scripts/round5_phase3.py" --output-root "$OUTPUT_ROOT" eval-baseline "$task" "$method" --checkpoint "$checkpoint" --device cuda --gpu "$gpu"
  fi
  video "$task" "$method" standard "$checkpoint"
}
archive_humanoid_lewm() {
  local source="$ROOT/outputs/round5_phase3_lewm_humanoid_resume_epoch6"
  local target="$OUTPUT_ROOT/training/lewm/humanoid"
  local cp="$source/checkpoints/lewm_weights_epoch_10.pt"
  local pattern='[t]rain.py.*round5_phase3_lewm_humanoid_resume_epoch6'
  while [[ ! -f "$cp" ]]; do
    if ! pgrep -f -- "$pattern" >/dev/null 2>&1; then
      echo "[$(timestamp)] requesting Humanoid LeWM continuation on GPU0/1"
      "$ROOT/scripts/run_round5_phase3.sh" resume-lewm-humanoid || true
      sleep 30
    fi
    echo "[$(timestamp)] waiting for LeWM/Humanoid epoch10"
    sleep "$POLL_SECONDS"
  done
  while pgrep -f -- "$pattern" >/dev/null 2>&1; do echo "[$(timestamp)] waiting for trainer cleanup"; sleep 60; done
  mkdir -p "$target/checkpoints" "$target/resume_epoch6"
  if [[ ! -f "$target/resume_epoch6/.archived" ]]; then cp -a "$source/." "$target/resume_epoch6/"; date -Iseconds >"$target/resume_epoch6/.archived"; fi
  for epoch in 7 8 9 10; do cp -f "$source/checkpoints/lewm_weights_epoch_"$epoch".pt" "$target/checkpoints/"; done
  for artifact in lewm_policy.ckpt lewm_object.ckpt; do [[ ! -f "$source/checkpoints/$artifact" ]] || cp -f "$source/checkpoints/$artifact" "$target/checkpoints/$artifact"; done
  cp -f "$cp" "$target/checkpoints/lewm_weights_epoch_10.pt"
  "$PYTHON" - "$target/resume_metadata.json" "$source" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
path,source=map(Path,sys.argv[1:])
d={"source_run":str(source),"initial_epoch":6,"completed_epoch":10,"optimizer_state_restored":False,"resume_method":"model_weights_with_epoch_offset","seed":3072,"archived_at":datetime.now(timezone.utc).isoformat()}
path.write_text(json.dumps(d,indent=2)+"\n")
PY
  echo "[$(timestamp)] archived Humanoid LeWM epoch10"
}
train_leflow() {
  local task="$1"
  local lewm="$OUTPUT_ROOT/training/lewm/$task/checkpoints/lewm_weights_epoch_10.pt"
  local out="$OUTPUT_ROOT/training/leflow/$task" cp="$OUTPUT_ROOT/training/leflow/$task/latent_planner.pt" done="$OUTPUT_ROOT/training/leflow/$task/training_complete.json" gpu log
  local running_pattern="[t]rain_round5_phase3_leflow.py.*[[:space:]]${task}([[:space:]]|$)"
  [[ -f "$lewm" ]] || { echo "missing LeWM checkpoint: $lewm" >&2; return 1; }
  if [[ -f "$cp" && -f "$done" ]] && "$PYTHON" - "$done" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if d.get("status")=="ok" and d.get("epochs")==10 else 1)
PY
  then return 0; fi
  if pgrep -f -- "$running_pattern" >/dev/null 2>&1; then
    echo "[$(timestamp)] waiting for existing LeFlow/$task trainer"
    while pgrep -f -- "$running_pattern" >/dev/null 2>&1; do sleep "$POLL_SECONDS"; done
    if [[ -f "$cp" && -f "$done" ]] && "$PYTHON" - "$done" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if d.get("status")=="ok" and d.get("epochs")==10 else 1)
PY
    then return 0; fi
  fi
  mkdir -p "$out"
  gpu="$(train_gpu || true)"
  while [[ -z "$gpu" ]]; do echo "[$(timestamp)] waiting for GPU0/1 with 12 GiB free for LeFlow/$task"; sleep "$POLL_SECONDS"; gpu="$(train_gpu || true)"; done
  log="$LOG_ROOT/train_leflow_"$task"_gpu"$gpu".log"
  echo "[$(timestamp)] training LeFlow/$task on GPU$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" MPLCONFIGDIR="/tmp/mpl-phase3-leflow-$task" "$PYTHON" -u "$ROOT/scripts/train_round5_phase3_leflow.py" "$task" --lewm-checkpoint "$lewm" --output "$out" --device cuda >"$log" 2>&1
  [[ -f "$cp" && -f "$done" ]] || { echo "LeFlow/$task missing final checkpoint; see $log" >&2; return 1; }
}
sync_scene
archive_humanoid_lewm
baseline humanoid lewm "$OUTPUT_ROOT/training/lewm/humanoid/checkpoints/lewm_weights_epoch_10.pt"
for task in scene finger humanoid; do
  cp="$OUTPUT_ROOT/training/fastlewam/$task/checkpoints/r4_ab_weights_epoch_10.pt"
  video "$task" fastlewam p1_step_1 "$cp"
  video "$task" fastlewam p3_step_1 "$cp"
done
baseline scene lewm "$OUTPUT_ROOT/training/lewm/scene/checkpoints/lewm_weights_epoch_10.pt"
baseline finger lewm "$OUTPUT_ROOT/training/lewm/finger/checkpoints/lewm_weights_epoch_10.pt"
for task in scene finger humanoid; do
  train_leflow "$task"
  baseline "$task" leflow "$OUTPUT_ROOT/training/leflow/$task/latent_planner.pt"
done
echo "[$(timestamp)] running full Phase 3 analysis"
"$PYTHON" "$ROOT/scripts/round5_phase3.py" --output-root "$OUTPUT_ROOT" analyze
for task in scene finger humanoid; do
  for rel in "fastlewam/$task/p1_step_1" "fastlewam/$task/p3_step_1" "lewm/$task/standard" "leflow/$task/standard"; do
    [[ -s "$OUTPUT_ROOT/videos/$rel/videos/env_0.mp4" ]] || { echo "missing video $rel" >&2; exit 1; }
  done
done
printf '{"status":"ok","completed_at":"%s","result_count":189}\n' "$(timestamp)" >"$DONE_MARKER"
echo "[$(timestamp)] complete: 189 result cells and 12 videos"
