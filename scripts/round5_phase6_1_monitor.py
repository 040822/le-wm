#!/usr/bin/env python3
"""Append one consolidated GPU and Phase 6.1 training-process snapshot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def _query(gpu: int, query: str) -> list[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            str(gpu),
            query,
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="2,4")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus or len(set(gpus)) != len(gpus) or any(value not in range(8) for value in gpus):
        raise ValueError("--gpus must list distinct physical GPU IDs 0..7")
    records = []
    for gpu in gpus:
        fields = _query(
            gpu,
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
        )
        if len(fields) != 1:
            raise RuntimeError(f"expected one GPU{gpu} status row, got {fields!r}")
        values = [item.strip() for item in fields[0].split(",")]
        if len(values) != 7:
            raise RuntimeError(f"invalid GPU{gpu} status row: {fields[0]!r}")
        process_rows = _query(gpu, "--query-compute-apps=pid,used_memory")
        processes = []
        for row in process_rows:
            pieces = [item.strip() for item in row.split(",")]
            if len(pieces) != 2:
                continue
            pid = int(pieces[0])
            command = subprocess.run(
                ["ps", "-p", str(pid), "-o", "etime=,args="],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            is_phase61 = "round5_phase6_1" in command
            if not is_phase61:
                command = "other compute process"
            processes.append(
                {
                    "pid": pid if is_phase61 else None,
                    "used_mib": pieces[1],
                    "process": command or None,
                }
            )
        records.append(
            {
                "gpu": int(values[0]),
                "uuid": values[1],
                "name": values[2],
                "memory_total_mib": int(values[3]),
                "memory_used_mib": int(values[4]),
                "memory_free_mib": int(values[5]),
                "utilization_percent": int(values[6]),
                "compute_processes": processes,
            }
        )
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpus": records,
    }
    output = ROOT / "outputs/round5/phase6_1/training/gpu_monitoring.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(snapshot, sort_keys=True) + "\n")
    print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
