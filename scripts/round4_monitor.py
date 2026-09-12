#!/usr/bin/env python3
"""Resident, read-only monitor for the Round 4 training/evaluation phase.

The monitor records progress and resource snapshots.  It deliberately never
restarts a run, changes a configuration, or sends a signal to a training
process.  GPU inspection is restricted to physical GPU0--3.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any


_PROGRESS_RE = re.compile(
    r"\[Epoch\s+(?P<epoch>\d+)/(?P<epochs>\d+)\]\s+"
    r"step\s+(?P<step>\d+)/(?P<steps>\d+)\s+"
    r"\((?P<rate>[^)]+)\)"
)
_ERROR_MARKERS = (
    "CUDA out of memory",
    "Traceback (most recent call last):",
    "RuntimeError:",
    "KeyboardInterrupt",
)


def _parse_run(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("run must have the form task=/absolute/run_dir")
    return name, Path(path).expanduser()


def _read_tail(path: Path, limit: int = 128 * 1024) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            return stream.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _run_snapshot(name: str, run_dir: Path) -> dict[str, Any]:
    log_candidates = [
        run_dir / f"{name}_single.log",
        run_dir / f"{name}.log",
        run_dir / "train.log",
    ]
    log_path = next((path for path in log_candidates if path.is_file()), None)
    tail = _read_tail(log_path) if log_path else ""
    matches = list(_PROGRESS_RE.finditer(tail))
    progress = None
    if matches:
        match = matches[-1]
        progress = {
            "epoch": int(match.group("epoch")),
            "epochs": int(match.group("epochs")),
            "step": int(match.group("step")),
            "steps": int(match.group("steps")),
            "rate": match.group("rate"),
        }
    checkpoint_dir = run_dir / "checkpoints"
    checkpoints = []
    if checkpoint_dir.is_dir():
        for path in sorted(checkpoint_dir.iterdir()):
            if path.is_file():
                stat = path.stat()
                checkpoints.append(
                    {
                        "name": path.name,
                        "bytes": stat.st_size,
                        "mtime": datetime.fromtimestamp(
                            stat.st_mtime, tz=timezone.utc
                        ).isoformat(),
                    }
                )
    errors = [marker for marker in _ERROR_MARKERS if marker in tail]
    return {
        "task": name,
        "run_dir": str(run_dir),
        "log": str(log_path) if log_path else None,
        "log_mtime": (
            datetime.fromtimestamp(log_path.stat().st_mtime, tz=timezone.utc).isoformat()
            if log_path
            else None
        ),
        "progress": progress,
        "checkpoints": checkpoints,
        "alerts": errors,
    }


def _gpu_snapshot() -> dict[str, Any]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    command = [
        "nvidia-smi",
        "-i",
        "0,1,2,3",
        "--query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "unavailable", "error": str(error)}
    devices = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        try:
            index = int(fields[0])
            if index not in range(4):
                continue
            devices.append(
                {
                    "index": index,
                    "memory_used_mib": int(fields[1]),
                    "memory_free_mib": int(fields[2]),
                    "memory_total_mib": int(fields[3]),
                    "utilization_percent": int(fields[4]),
                }
            )
        except ValueError:
            continue
    return {"status": "ok", "devices": devices}


def _result_snapshot(result_root: Path) -> list[dict[str, Any]]:
    results = []
    if not result_root.is_dir():
        return results
    for path in sorted(result_root.glob("**/result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            results.append({"path": str(path), "error": str(error)})
            continue
        results.append(
            {
                "path": str(path),
                "task": payload.get("task"),
                "mode": payload.get("round4_mode", payload.get("stage")),
                "cohort_kind": payload.get("cohort_kind"),
                "success_rate": payload.get("success_rate"),
                "status": payload.get("status"),
            }
        )
    return results


def snapshot(runs: list[tuple[str, Path]], result_root: Path) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runs": [_run_snapshot(name, path) for name, path in runs],
        "gpu": _gpu_snapshot(),
        "results": _result_snapshot(result_root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run,
        required=True,
        help="training run mapping, e.g. cube=/path/to/run_dir; repeat per task",
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=300.0)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    runs = list(args.run)
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    latest = args.output.with_suffix(".latest.json")
    while True:
        payload = snapshot(runs, args.result_root)
        line = json.dumps(payload, sort_keys=True)
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        latest.write_text(line + "\n", encoding="utf-8")
        if args.once or (args.stop_file is not None and args.stop_file.exists()):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
