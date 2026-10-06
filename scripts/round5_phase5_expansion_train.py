#!/usr/bin/env python3
"""Calibrate and run the Round 5 Phase 5 native-action training matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "round5_phase5_expansion_seed3072"
HASH_CACHE = OUTPUT_ROOT / "dataset_hashes.json"
STATE_PATH = OUTPUT_ROOT / "scheduler_state.json"
PID_PATH = OUTPUT_ROOT / "scheduler.pid"
SCHEDULER_LOG = OUTPUT_ROOT / "scheduler.log"
GPU_IDS = (0, 1, 2, 3)
TASKS = ("pusht", "reacher")
STRUCTURES = (
    ("5x5", 5, 5),
    ("1x25", 1, 25),
    ("5x1", 5, 1),
    ("1x5", 1, 5),
    ("1x1", 1, 1),
)


def _dataset_path(task: str) -> Path:
    if task == "pusht":
        return ROOT / "data" / "datasets" / "pusht.h5"
    if task == "reacher":
        return ROOT / "data" / "datasets" / "dmcontrol" / "reacher.h5"
    raise ValueError(f"Unsupported Phase5 task: {task}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_hashes(tasks=TASKS) -> dict[str, str]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        cache = json.loads(HASH_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cache = {}

    pending: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for task in tasks:
        path = _dataset_path(task)
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        cached = cache.get(task, {})
        if (
            int(cached.get("size_bytes", -1)) == stat.st_size
            and int(cached.get("mtime_ns", -1)) == stat.st_mtime_ns
            and cached.get("sha256")
        ):
            hashes[task] = str(cached["sha256"])
        else:
            pending[task] = path

    def hash_task(item):
        task, path = item
        print(f"Hashing {task} dataset: {path}", flush=True)
        return task, _sha256_file(path)

    with ThreadPoolExecutor(max_workers=max(1, len(pending))) as executor:
        for task, digest in executor.map(hash_task, pending.items()):
            hashes[task] = digest
            stat = _dataset_path(task).stat()
            cache[task] = {
                "path": str(_dataset_path(task)),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": digest,
            }
            HASH_CACHE.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"{task} dataset SHA256: {digest}", flush=True)
    return hashes


def _job(task: str, structure: tuple[str, int, int]) -> dict[str, Any]:
    label, action_block_steps, action_horizon = structure
    return {
        "id": f"{task}_{label}",
        "task": task,
        "structure": label,
        "action_block_steps": action_block_steps,
        "action_horizon": action_horizon,
        "goal_index": 25 // action_block_steps,
    }


def _all_jobs() -> list[dict[str, Any]]:
    # Interleave tasks so both datasets start early and the four initial jobs
    # cover 5x5 and 1x25 on every device.
    return [_job(task, structure) for structure in STRUCTURES for task in TASKS]


def _command(job: dict[str, Any], dataset_sha256: str, *, micro_batch: int, max_steps=-1, subdir=None):
    action_block_steps = int(job["action_block_steps"])
    action_horizon = int(job["action_horizon"])
    goal_index = int(job["goal_index"])
    task = str(job["task"])
    structure = str(job["structure"])
    accumulation = 128 // int(micro_batch)
    output_model_name = f"r5_phase5_seed3072_{task}_{structure}"
    output_subdir = subdir or f"round5_phase5_expansion_seed3072_{task}_{structure}"
    return [
        sys.executable,
        "train.py",
        "--config-name=round5_phase5_expansion",
        f"data={task}",
        "seed=3072",
        f"action_horizon={action_horizon}",
        f"data.dataset.frameskip={action_block_steps}",
        f"data.dataset.num_steps={goal_index + 1}",
        f"policy.stage_a_goal_index={goal_index}",
        f"loader.batch_size={micro_batch}",
        f"trainer.accumulate_grad_batches={accumulation}",
        f"trainer.max_steps={int(max_steps)}",
        f"round5_phase5_expansion.task={task}",
        f"round5_phase5_expansion.structure={structure}",
        f"round5_phase5_expansion.action_block_steps={action_block_steps}",
        f"round5_phase5_expansion.stage_a_goal_index={goal_index}",
        f"round5_phase5_expansion.dataset_sha256={dataset_sha256}",
        f"subdir={output_subdir}",
        f"output_model_name={output_model_name}",
        "wandb.enabled=false",
        "round4_diagnostics.enabled=false",
    ]


def _run_calibration_job(
    job: dict[str, Any], dataset_sha256: str, micro_batch: int, gpu_id: int
) -> dict[str, Any]:
    calibration_dir = OUTPUT_ROOT / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    output_subdir = (
        f"round5_phase5_calibration_seed3072_{job['task']}_"
        f"{job['structure']}_mb{micro_batch}_s2"
    )
    command = _command(
        job,
        dataset_sha256,
        micro_batch=micro_batch,
        max_steps=2,
        subdir=output_subdir,
    )
    log_path = calibration_dir / f"{job['id']}_mb{micro_batch}_s2.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    baseline = _gpu_memory_snapshot(gpu_id)
    peak_mib = baseline["used_mib"] if baseline else None
    minimum_free_mib = baseline["free_mib"] if baseline else None
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_stream:
        log_stream.write("COMMAND: " + " ".join(command) + "\n")
        log_stream.write(f"CUDA_VISIBLE_DEVICES={gpu_id}\n")
        log_stream.flush()
        if baseline is None or baseline["free_mib"] < 6144:
            log_stream.write("SKIPPED: selected GPU has less than 6 GiB free\n")
            return {
                "job": job["id"],
                "micro_batch_size": micro_batch,
                "gradient_accumulation": 128 // micro_batch,
                "gpu": gpu_id,
                "return_code": 125,
                "oom": False,
                "resource_blocked": True,
                "vram_margin_ok": False,
                "elapsed_seconds": 0.0,
                "gpu_memory_baseline_mib": None if baseline is None else baseline["used_mib"],
                "gpu_memory_peak_observed_mib": peak_mib,
                "gpu_memory_peak_delta_mib": None,
                "gpu_memory_minimum_free_mib": minimum_free_mib,
                "log_path": str(log_path),
            }
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            snapshot = _gpu_memory_snapshot(gpu_id)
            if snapshot is not None:
                peak_mib = max(peak_mib or 0, snapshot["used_mib"])
                minimum_free_mib = min(
                    minimum_free_mib or snapshot["free_mib"],
                    snapshot["free_mib"],
                )
            time.sleep(1)
        return_code = int(process.returncode)
    final_snapshot = _gpu_memory_snapshot(gpu_id)
    if final_snapshot is not None:
        peak_mib = max(peak_mib or 0, final_snapshot["used_mib"])
        minimum_free_mib = min(
            minimum_free_mib or final_snapshot["free_mib"],
            final_snapshot["free_mib"],
        )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    is_oom = "out of memory" in log_text.lower() or "cuda oom" in log_text.lower()
    return {
        "job": job["id"],
        "micro_batch_size": micro_batch,
        "gradient_accumulation": 128 // micro_batch,
        "gpu": gpu_id,
        "return_code": return_code,
        "oom": is_oom,
        "resource_blocked": False,
        "vram_margin_ok": minimum_free_mib is not None and minimum_free_mib >= 6144,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "gpu_memory_baseline_mib": None if baseline is None else baseline["used_mib"],
        "gpu_memory_peak_observed_mib": peak_mib,
        "gpu_memory_peak_delta_mib": (
            None
            if baseline is None or peak_mib is None
            else max(0, peak_mib - baseline["used_mib"])
        ),
        "gpu_memory_minimum_free_mib": minimum_free_mib,
        "log_path": str(log_path),
    }


def _gpu_memory_snapshot(gpu_id: int) -> dict[str, int] | None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu_id),
                "--query-gpu=memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        used_mib, total_mib, free_mib = map(
            int, completed.stdout.strip().splitlines()[0].split(",")
        )
        return {"used_mib": used_mib, "total_mib": total_mib, "free_mib": free_mib}
    except (OSError, ValueError, subprocess.CalledProcessError, IndexError):
        return None


def _best_available_gpu() -> tuple[int, dict[str, int]] | None:
    available = []
    for gpu_id in GPU_IDS:
        snapshot = _gpu_memory_snapshot(gpu_id)
        if snapshot is not None and snapshot["free_mib"] >= 6144:
            available.append((gpu_id, snapshot))
    return max(available, key=lambda item: item[1]["free_mib"]) if available else None


def calibrate() -> int:
    hashes = _dataset_hashes(tasks=("pusht",))
    jobs = [_job("pusht", structure) for structure in STRUCTURES]
    calibration_results: list[dict[str, Any]] = []
    chosen_micro_batch = None
    for micro_batch in (32, 16, 8):
        print(
            f"Calibrating all five structures with micro-batch {micro_batch} "
            f"and accumulation {128 // micro_batch}",
            flush=True,
        )
        pass_results = []
        should_retry = False
        for job in jobs:
            selected = _best_available_gpu()
            while selected is None:
                print("Waiting for a permitted GPU with at least 6 GiB free", flush=True)
                time.sleep(30)
                selected = _best_available_gpu()
            gpu_id, _ = selected
            result = _run_calibration_job(
                job, hashes["pusht"], micro_batch, gpu_id
            )
            if result["resource_blocked"]:
                print(
                    f"GPU{gpu_id} lost its VRAM margin before {job['id']}; retrying selection",
                    flush=True,
                )
                continue
            calibration_results.append(result)
            pass_results.append(result)
            print(
                f"{result['job']} mb={micro_batch}: rc={result['return_code']}, "
                f"peak={result['gpu_memory_peak_observed_mib']} MiB, "
                f"min-free={result['gpu_memory_minimum_free_mib']} MiB, "
                f"elapsed={result['elapsed_seconds']}s",
                flush=True,
            )
            if result["return_code"] != 0 or not result["vram_margin_ok"]:
                if result["oom"] or not result["vram_margin_ok"]:
                    should_retry = True
                    break
                log_tail = Path(result["log_path"]).read_text(
                    encoding="utf-8", errors="replace"
                )[-5000:]
                print(log_tail, file=sys.stderr, flush=True)
                return 1
        if not should_retry and len(pass_results) == len(jobs):
            chosen_micro_batch = micro_batch
            break
        if micro_batch == 8:
            break

    result_path = OUTPUT_ROOT / "calibration" / "results.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tasks_calibrated": ["pusht"],
                "gpu_ids_used": sorted(
                    {record["gpu"] for record in calibration_results}
                ),
                "effective_batch_size": 128,
                "chosen_micro_batch_size": chosen_micro_batch,
                "calibrations": calibration_results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if chosen_micro_batch is None:
        print(f"No safe micro-batch found; see {result_path}", file=sys.stderr)
        return 1
    print(f"Selected shared micro-batch size: {chosen_micro_batch}")
    print(f"Calibration record: {result_path}")
    return 0


def _atomic_state_write(state: dict[str, Any]) -> None:
    temporary = STATE_PATH.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(STATE_PATH)


def _read_state() -> dict[str, Any]:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "not_started", "jobs": {}}


def start() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    calibration_path = OUTPUT_ROOT / "calibration" / "results.json"
    if not calibration_path.is_file():
        print("Run the five-structure calibration first: calibrate", file=sys.stderr)
        return 1
    if PID_PATH.exists():
        try:
            pid = int(PID_PATH.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
            command_line = Path(f"/proc/{pid}/cmdline").read_bytes().decode(
                "utf-8", errors="replace"
            )
            if str(Path(__file__).resolve()) in command_line and "schedule" in command_line:
                print(f"Scheduler already running with PID {pid}")
                return 0
            PID_PATH.unlink(missing_ok=True)
        except (ValueError, OSError):
            PID_PATH.unlink(missing_ok=True)
    log_stream = SCHEDULER_LOG.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "schedule"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_stream.close()
    PID_PATH.write_text(f"{process.pid}\n", encoding="utf-8")
    print(f"Started Phase5 scheduler PID {process.pid}; status: status")
    print(f"Scheduler log: {SCHEDULER_LOG}")
    return 0


def _make_job_command(job, dataset_hashes, micro_batch):
    return _command(job, dataset_hashes[job["task"]], micro_batch=micro_batch)


def _job_output_dir(job):
    return (
        ROOT
        / "outputs"
        / f"round5_phase5_expansion_seed3072_{job['task']}_{job['structure']}"
    )


def _existing_output(job, dataset_sha256, micro_batch):
    run_dir = _job_output_dir(job)
    if not run_dir.exists() or not any(run_dir.iterdir()):
        return "new", None
    identity_path = run_dir / "round5_phase5_training_identity.json"
    if not identity_path.is_file():
        return "collision", "existing output has no Phase5 training identity"
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return "collision", f"cannot read existing training identity: {error}"
    expected = {
        "task": job["task"],
        "structure": job["structure"],
        "seed": 3072,
        "action_block_steps": int(job["action_block_steps"]),
        "action_horizon": int(job["action_horizon"]),
        "stage_a_goal_index": int(job["goal_index"]),
        "effective_window_steps": 25,
        "clip_storage_steps": 30,
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            return "collision", f"existing identity differs at {key}"
    if identity.get("dataset", {}).get("sha256") != dataset_sha256:
        return "collision", "existing identity has a different dataset SHA256"
    split = identity.get("split", {})
    if (
        split.get("effective_batch_size") != 128
        or split.get("micro_batch_size") != micro_batch
        or split.get("gradient_accumulation") != 128 // micro_batch
        or split.get("epochs") != 10
        or split.get("max_steps") != -1
    ):
        return "collision", "existing identity has a different batch or training budget"
    resolved = identity.get("resolved_config", {})
    expected_config = {
        "seed": 3072,
        "train_mode": "stage_ab",
        "train_split": 0.9,
        "action_horizon": int(job["action_horizon"]),
        "trainer.max_epochs": 10,
        "trainer.max_steps": -1,
        "trainer.precision": "bf16",
        "trainer.accumulate_grad_batches": 128 // micro_batch,
        "loader.batch_size": micro_batch,
        "optimizer.lr": 5e-5,
        "loss.latent.weight": 1.0,
        "loss.sigreg.weight": 0.09,
        "loss.d.lambda": 0.0,
        "loss.e.lambda": 0.0,
        "data.dataset.frameskip": int(job["action_block_steps"]),
        "data.dataset.num_steps": int(job["goal_index"]) + 1,
        "policy.stage_a_goal_index": int(job["goal_index"]),
        "wandb.enabled": False,
        "round4_diagnostics.enabled": False,
        "round5_phase5_expansion.enabled": True,
        "round5_phase5_expansion.task": job["task"],
        "round5_phase5_expansion.structure": job["structure"],
        "round5_phase5_expansion.effective_window_steps": 25,
        "round5_phase5_expansion.clip_storage_steps": 30,
        "round5_phase5_expansion.effective_batch_size": 128,
    }
    for dotted_key, expected_value in expected_config.items():
        value = resolved
        for component in dotted_key.split("."):
            value = value.get(component) if isinstance(value, dict) else None
        if value != expected_value:
            return "collision", f"existing resolved config differs at {dotted_key}"
    code_hashes = identity.get("code_sha256", {})
    if len(code_hashes) < 8:
        return "collision", "existing identity lacks complete code hashes"
    for path, expected_hash in code_hashes.items():
        code_path = ROOT / path
        if not code_path.is_file() or _sha256_file(code_path) != expected_hash:
            return "collision", f"training source changed since existing run: {path}"
    checkpoint = (
        run_dir
        / "checkpoints"
        / f"r5_phase5_seed3072_{job['task']}_{job['structure']}_weights_epoch_10.pt"
    )
    if checkpoint.is_file():
        return "complete", str(checkpoint)
    return "resume", None


def _schedule() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "hashing_datasets",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_ids": list(GPU_IDS),
        "dataset_hashes": {},
        "jobs": {},
    }
    _atomic_state_write(state)
    calibration_path = OUTPUT_ROOT / "calibration" / "results.json"
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        micro_batch = int(calibration["chosen_micro_batch_size"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        state["status"] = "blocked_no_calibration"
        state["error"] = "Run the five-structure calibration before training."
        _atomic_state_write(state)
        return 1
    successful_calibrations = {
        record.get("job")
        for record in calibration.get("calibrations", [])
        if record.get("micro_batch_size") == micro_batch
        and record.get("return_code") == 0
        and record.get("vram_margin_ok") is True
    }
    expected_calibrations = {f"pusht_{label}" for label, _, _ in STRUCTURES}
    if (
        micro_batch not in {8, 16, 32}
        or calibration.get("effective_batch_size") != 128
        or not expected_calibrations.issubset(successful_calibrations)
    ):
        state["status"] = "blocked_invalid_calibration"
        state["error"] = "Calibration must pass all five structures with >=6 GiB free."
        _atomic_state_write(state)
        return 1

    try:
        dataset_hashes = _dataset_hashes()
    except Exception as error:
        state["status"] = "failed_dataset_identity"
        state["error"] = f"Could not identify training datasets: {error}"
        _atomic_state_write(state)
        raise
    state["dataset_hashes"] = dataset_hashes
    state["status"] = "running"
    _atomic_state_write(state)

    pending = _all_jobs()
    active: dict[int, tuple[dict[str, Any], subprocess.Popen, Any]] = {}
    for job in pending:
        state["jobs"][job["id"]] = {"status": "queued", **job}
    _atomic_state_write(state)

    next_index = 0
    while next_index < len(pending) or active:
        free_gpus = [gpu_id for gpu_id in GPU_IDS if gpu_id not in active]
        for gpu_id in free_gpus:
            if next_index >= len(pending):
                break
            job = pending[next_index]
            record = state["jobs"][job["id"]]
            gpu_snapshot = _gpu_memory_snapshot(gpu_id)
            if gpu_snapshot is None or gpu_snapshot["free_mib"] < 6144:
                record.update(
                    status="waiting_for_vram_margin",
                    gpu=gpu_id,
                    free_memory_mib=(
                        None if gpu_snapshot is None else gpu_snapshot["free_mib"]
                    ),
                )
                continue

            output_status, output_detail = _existing_output(
                job, dataset_hashes[job["task"]], micro_batch
            )
            if output_status == "collision":
                record.update(status="failed_existing_output", error=output_detail)
                print(f"Refusing to overwrite {job['id']}: {output_detail}", flush=True)
                next_index += 1
                continue
            if output_status == "complete":
                record.update(
                    status="complete",
                    gpu=gpu_id,
                    checkpoint=output_detail,
                    checkpoint_sha256=_sha256_file(Path(output_detail)),
                    reused_existing_output=True,
                )
                next_index += 1
                continue

            command = _make_job_command(job, dataset_hashes, micro_batch)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            log_path = OUTPUT_ROOT / f"{job['id']}.log"
            log_stream = log_path.open("a", encoding="utf-8")
            log_stream.write(
                "\nCOMMAND: " + " ".join(command) + f"\nGPU: {gpu_id}\n"
            )
            log_stream.flush()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record.update(
                status="running",
                pid=process.pid,
                gpu=gpu_id,
                log_path=str(log_path),
                started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                micro_batch_size=micro_batch,
                gradient_accumulation=128 // micro_batch,
                resuming_existing_output=output_status == "resume",
            )
            active[gpu_id] = (job, process, log_stream)
            next_index += 1
            print(
                f"Started {job['id']} on GPU{gpu_id}, PID {process.pid}",
                flush=True,
            )
        for gpu_id, (job, process, log_stream) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_stream.close()
            record = state["jobs"][job["id"]]
            checkpoint = _job_output_dir(job) / "checkpoints" / (
                f"r5_phase5_seed3072_{job['task']}_{job['structure']}_weights_epoch_10.pt"
            )
            if return_code == 0 and checkpoint.is_file():
                record.update(
                    status="complete",
                    completed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    return_code=0,
                    checkpoint=str(checkpoint),
                    checkpoint_sha256=_sha256_file(checkpoint),
                )
                print(f"Completed {job['id']} on GPU{gpu_id}", flush=True)
            else:
                record.update(
                    status="failed",
                    completed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    return_code=int(return_code),
                    checkpoint_exists=checkpoint.is_file(),
                )
                print(
                    f"Failed {job['id']} on GPU{gpu_id}: return code {return_code}",
                    flush=True,
                )
            del active[gpu_id]
        _atomic_state_write(state)
        if active or next_index < len(pending):
            time.sleep(10)

    failures = [
        job_id
        for job_id, record in state["jobs"].items()
        if str(record.get("status", "")).startswith("failed")
    ]
    state["status"] = "failed" if failures else "complete"
    state["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["failures"] = failures
    _atomic_state_write(state)
    print(f"Phase5 scheduler finished with status {state['status']}", flush=True)
    return 1 if failures else 0


def status() -> int:
    state = _read_state()
    print(f"Scheduler: {state.get('status', 'not_started')}")
    print(f"GPUs: {','.join(map(str, GPU_IDS))}")
    for job_id, record in state.get("jobs", {}).items():
        details = [str(record.get("status", "unknown"))]
        if record.get("gpu") is not None:
            details.append(f"GPU{record['gpu']}")
        if record.get("pid") is not None:
            details.append(f"PID {record['pid']}")
        if record.get("return_code") is not None:
            details.append(f"exit {record['return_code']}")
        print(f"{job_id}: " + ", ".join(details))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPU_IDS))
    try:
        gpu_status = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                ",".join(map(str, GPU_IDS)),
                "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader",
            ],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        print("GPU status:")
        print(gpu_status.stdout.rstrip())
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Could not query GPU status: {error}")
        return 1
    return 1 if state.get("status", "").startswith("failed") else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("calibrate", "start", "schedule", "status"))
    command = parser.parse_args().command
    if command == "calibrate":
        return calibrate()
    if command == "start":
        return start()
    if command == "schedule":
        return _schedule()
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
