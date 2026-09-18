"""Fail-closed, read-only GPU resource preflight for Round 3 launchers.

Existing compute processes are rejected by default.  A caller that has an
explicit shared-GPU allocation may opt in, while the free-memory guard still
applies.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Any


ALLOWED_GPUS = frozenset({0, 1, 2, 3})


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in command_gpus)
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
        env=environment,
    )


def _parse_processes(output: str, gpu: int) -> list[str]:
    processes = []
    for line in output.splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#") or fields[0] == "-":
            continue
        if len(fields) < 3 or fields[1] == "-":
            continue
        if int(fields[0]) != int(gpu):
            raise RuntimeError(f"pmon returned physical GPU{fields[0]} while querying GPU{gpu}")
        processes.append(" ".join(fields))
    return processes


def preflight(
    gpus: list[int],
    *,
    minimum_free_mib: int = 12000,
    allow_existing_compute: bool = False,
) -> dict[str, Any]:
    global command_gpus
    command_gpus = list(gpus)
    invalid = sorted(set(gpus) - ALLOWED_GPUS)
    if invalid or not gpus or len(set(gpus)) != len(gpus):
        raise ValueError(f"GPU list must be distinct and limited to 0-3; got {gpus}")
    memory: dict[str, int] = {}
    processes: dict[str, list[str]] = {}
    for gpu in gpus:
        proc = _run(["nvidia-smi", "pmon", "-i", str(gpu), "-c", "1"])
        rows = _parse_processes(proc.stdout, gpu)
        processes[str(gpu)] = rows
        if rows and not allow_existing_compute:
            raise RuntimeError(f"GPU{gpu} has active compute processes: {rows}")
        mem = _run([
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ])
        free_mib = int(mem.stdout.strip())
        memory[str(gpu)] = free_mib
        if free_mib < int(minimum_free_mib):
            raise RuntimeError(
                f"GPU{gpu} has only {free_mib} MiB free; "
                f"minimum is {int(minimum_free_mib)} MiB"
            )
    return {
        "status": "ok",
        "gpus": gpus,
        "free_memory_mib": memory,
        "compute_processes": processes,
        "shared_compute_allowed": bool(allow_existing_compute),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--minimum-free-mib", type=int, default=12000)
    parser.add_argument(
        "--allow-existing-compute",
        action="store_true",
        help="allow explicitly shared GPUs; the free-memory guard remains active",
    )
    args = parser.parse_args(argv)
    print(json.dumps(preflight(
        args.gpus,
        minimum_free_mib=args.minimum_free_mib,
        allow_existing_compute=args.allow_existing_compute,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
