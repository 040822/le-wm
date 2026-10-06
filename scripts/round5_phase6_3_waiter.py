#!/usr/bin/env python3
"""Wait for a quiet, sufficiently large GPU and resume frozen Phase 6.3 timing."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/round5/phase6_3"
STATE = OUT / "timing/resource_waiter.json"
LOG_DIR = OUT / "logs"
GPU_ORDER = tuple(range(8))
INTERVAL_SECONDS = 300
REQUIRED_CONSECUTIVE_CHECKS = 2
MIN_FREE_GIB = 10
MAX_CPU_LOAD_PER_CPU = 0.75


def gpu_snapshot(gpu: int) -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    query = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free,memory.total,utilization.gpu,uuid,name,driver_version",
         "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()
    fields = [item.strip() for item in query.split(",")]
    if len(fields) != 6:
        raise RuntimeError(f"unexpected nvidia-smi GPU row for {gpu}: {query!r}")
    process_query = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()
    processes = []
    for line in process_query.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if parts and parts[0].isdigit():
            processes.append(parts)
    free_mib, total_mib, utilization = map(int, fields[:3])
    return {
        "gpu": gpu,
        "free_vram_gib": free_mib / 1024,
        "total_vram_gib": total_mib / 1024,
        "gpu_utilization_percent": utilization,
        "gpu_uuid": fields[3],
        "gpu_name": fields[4],
        "driver_version": fields[5],
        "compute_processes": processes,
    }


def cpu_snapshot() -> dict:
    load_1, load_5, load_15 = os.getloadavg()
    count = max(1, os.cpu_count() or 1)
    return {
        "cpu_count": count,
        "cpu_load_1m": load_1,
        "cpu_load_5m": load_5,
        "cpu_load_15m": load_15,
        "cpu_load_1m_per_cpu": load_1 / count,
    }


def gpu_is_eligible(snapshot: dict, cpu: dict) -> tuple[bool, list[str]]:
    reasons = []
    if snapshot["free_vram_gib"] < MIN_FREE_GIB:
        reasons.append("insufficient_free_vram")
    if snapshot["compute_processes"]:
        reasons.append("external_compute_process")
    if not snapshot["compute_processes"] and snapshot["gpu_utilization_percent"] > 5:
        reasons.append("gpu_utilization_above_5_percent")
    if cpu["cpu_load_1m_per_cpu"] > MAX_CPU_LOAD_PER_CPU:
        reasons.append("cpu_load_above_0.75_per_cpu")
    if _free_disk_gib() < 20:
        reasons.append("free_disk_below_20_gib")
    return not reasons, reasons


def _free_disk_gib() -> float:
    result = subprocess.run(
        ["df", "-Pk", str(ROOT)], check=True, capture_output=True, text=True
    )
    fields = result.stdout.splitlines()[-1].split()
    return int(fields[3]) / (1024 * 1024)


def write_state(**values) -> None:
    code_path = Path(__file__).resolve()
    digest = hashlib.sha256(code_path.read_bytes()).hexdigest()
    state = {
        "status": "waiting_for_resources",
        "pid": os.getpid(),
        "started_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "poll_interval_seconds": INTERVAL_SECONDS,
        "required_consecutive_checks": REQUIRED_CONSECUTIVE_CHECKS,
        "waiter_code_sha256": digest,
        **values,
    }
    OUT.joinpath("timing").mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(STATE)


def completed_gpu_pin() -> str | None:
    """Keep retries on the device already used by any accepted formal sample."""
    found = set()
    for result_path in (OUT / "timing/runs").glob("*/**/attempt_*/result.json"):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("formal_latency") is True and result.get("status") == "complete":
            uuid = result.get("gpu_before", {}).get("gpu_uuid")
            if uuid:
                found.add(uuid)
    if len(found) > 1:
        raise RuntimeError(f"formal timing already spans multiple GPU identities: {sorted(found)}")
    return next(iter(found), None)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = OUT / "timing/resource_waiter.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another Phase 6.3 resource waiter already holds the lock", flush=True)
            return 2

        pinned_uuid = completed_gpu_pin()
        previous_gpu = None
        consecutive = 0
        checks = 0
        while True:
            checks += 1
            cpu = cpu_snapshot()
            candidate = None
            snapshots = []
            for gpu in GPU_ORDER:
                try:
                    snapshot = gpu_snapshot(gpu)
                    eligible, reasons = gpu_is_eligible(snapshot, cpu)
                except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
                    snapshot = {"gpu": gpu, "error": f"{type(exc).__name__}: {exc}"}
                    eligible, reasons = False, ["gpu_probe_failed"]
                snapshots.append({**snapshot, "eligible": eligible, "reasons": reasons})
                if eligible and (pinned_uuid is None or snapshot["gpu_uuid"] == pinned_uuid):
                    candidate = snapshot
                    break

            same_candidate = (
                candidate is not None
                and previous_gpu is not None
                and candidate["gpu"] == previous_gpu["gpu"]
                and candidate["gpu_uuid"] == previous_gpu["gpu_uuid"]
            )
            consecutive = consecutive + 1 if same_candidate else (1 if candidate else 0)
            previous_gpu = candidate
            write_state(
                checks=checks,
                consecutive_eligible_checks=consecutive,
                pinned_gpu_uuid=pinned_uuid,
                candidate=candidate,
                cpu=cpu,
                gpu_observations=snapshots,
                last_check_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            print(json.dumps({
                "check": checks,
                "candidate": None if candidate is None else candidate["gpu"],
                "consecutive": consecutive,
                "pinned_gpu_uuid": pinned_uuid,
                "cpu_load_1m_per_cpu": cpu["cpu_load_1m_per_cpu"],
                "eligible": [(item["gpu"], item.get("gpu_uuid")) for item in snapshots
                             if item.get("eligible")],
            }), flush=True)

            if consecutive >= REQUIRED_CONSECUTIVE_CHECKS and candidate is not None:
                gpu = int(candidate["gpu"])
                log_path = LOG_DIR / f"formal_timing_gpu{gpu}_resume.log"
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                command = [sys.executable, "scripts/round5_phase6_3_timing.py", "--gpu", str(gpu)]
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] launch: {command!r}\n")
                    log.flush()
                    write_state(
                        status="timing_running", checks=checks,
                        consecutive_eligible_checks=consecutive,
                        pinned_gpu_uuid=candidate["gpu_uuid"], candidate=candidate,
                        cpu=cpu, log=str(log_path.relative_to(ROOT)),
                        command=command,
                    )
                    child = subprocess.Popen(
                        command, cwd=ROOT, env=env, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True,
                    )
                    write_state(
                        status="timing_running", checks=checks,
                        consecutive_eligible_checks=consecutive,
                        pinned_gpu_uuid=candidate["gpu_uuid"], candidate=candidate,
                        cpu=cpu, log=str(log_path.relative_to(ROOT)),
                        command=command, timing_pid=child.pid,
                    )
                    print(json.dumps({"status": "timing_started", "gpu": gpu,
                                      "pid": child.pid, "log": str(log_path.relative_to(ROOT))}),
                          flush=True)
                    return_code = child.wait()
                    log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] exit={return_code}\n")
                    log.flush()

                progress_path = OUT / "timing/formal_progress.json"
                progress = {}
                if progress_path.is_file():
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                if return_code == 0 and progress.get("status") == "complete":
                    write_state(
                        status="complete", checks=checks, pinned_gpu_uuid=candidate["gpu_uuid"],
                        candidate=candidate, cpu=cpu, log=str(log_path.relative_to(ROOT)),
                        timing_pid=child.pid, timing_exit_code=return_code,
                        completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    )
                    print('{"status":"complete"}', flush=True)
                    return 0

                last_failure = None
                failures = sorted((OUT / "timing/runs").glob("*/**/attempt_*/failure.json"))
                if failures:
                    try:
                        last_failure = json.loads(failures[-1].read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        pass
                error = str((last_failure or {}).get("error", ""))
                retryable = any(token in error.lower() for token in (
                    "timing requires idle gpu", "timing requires a quiet cpu",
                    "timing interference", "external gpu compute process",
                    "gpu utilization is",
                ))
                if not retryable:
                    write_state(
                        status="failed", checks=checks, pinned_gpu_uuid=candidate["gpu_uuid"],
                        candidate=candidate, cpu=cpu, log=str(log_path.relative_to(ROOT)),
                        timing_pid=child.pid, timing_exit_code=return_code,
                        last_failure=last_failure, progress=progress,
                    )
                    print(json.dumps({"status": "failed", "exit_code": return_code,
                                      "last_failure": last_failure}), flush=True)
                    return return_code or 1

                pinned_uuid = candidate["gpu_uuid"]
                previous_gpu = None
                consecutive = 0
                write_state(
                    status="waiting_for_resources", checks=checks,
                    consecutive_eligible_checks=0, pinned_gpu_uuid=pinned_uuid,
                    last_timing_exit_code=return_code, last_failure=last_failure,
                    progress=progress, retry_same_gpu=True,
                    last_check_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                print(json.dumps({"status": "retry_wait", "gpu_uuid": pinned_uuid,
                                  "exit_code": return_code, "error": error}), flush=True)

            time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
