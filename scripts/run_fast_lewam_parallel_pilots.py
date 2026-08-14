"""Launch and monitor the manifest-defined Fast-LeWAM four-way pilots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from source.experiments.fast_lewam_parallel_pilots import (
    build_launch_specs,
    load_pilot_manifest,
    valid_summary_payload,
)


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _pid_matches(pid, command):
    path = Path(f"/proc/{int(pid)}/cmdline")
    if not path.is_file():
        return False
    actual = tuple(value.decode() for value in path.read_bytes().split(b"\0") if value)
    expected = tuple(command)
    return len(actual) >= len(expected) and actual[-len(expected):] == expected


def _gpu_compute_processes(gpu, environment):
    result = subprocess.run(
        [
            "nvidia-smi", "-i", str(gpu),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True, text=True, capture_output=True,
        env={**os.environ, **environment},
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _valid_summary(path, manifest, pilot, level):
    if not path.is_file():
        return False
    value = json.loads(path.read_text(encoding="utf-8"))
    return valid_summary_payload(value, manifest, pilot, level)


def _free_memory(gpu, environment):
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, **environment},
    )
    return int(result.stdout.strip())


def _preflight(manifest, specs):
    errors = []
    selected = {spec.pilot for spec in specs}
    for pilot in manifest.pilots.values():
        if pilot.name not in selected:
            continue
        required = [pilot.baseline_run_config, pilot.baseline_checkpoint]
        if pilot.direction != "action_tokens_25":
            required.extend((pilot.actor_run_config, pilot.actor_checkpoint, pilot.trace))
        errors.extend(str(path) for path in required if path is None or not path.is_file())
    if errors:
        raise FileNotFoundError(f"missing pilot inputs: {errors}")
    memory = {}
    for spec in specs:
        processes = _gpu_compute_processes(spec.gpu, spec.environment)
        if processes:
            raise RuntimeError(
                f"GPU{spec.gpu} already has compute processes: {processes}"
            )
        memory[spec.gpu] = _free_memory(spec.gpu, spec.environment)
        if memory[spec.gpu] < 12000:
            raise RuntimeError(
                f"GPU{spec.gpu} has only {memory[spec.gpu]} MiB free; 12000 MiB required"
            )
    return memory


def _status(manifest, level):
    rows = {}
    for pilot in manifest.pilots.values():
        root = pilot.output_dir / f"level_{level}"
        state_path = root / "process.json"
        summary = root / "summary.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        pid = state.get("pid")
        expected_command = state.get("command", ())
        running = bool(pid and _pid_matches(pid, expected_command))
        checkpoints = sorted((root / "train" / "checkpoints").glob("*_epoch_*.pt"))
        rows[pilot.name] = {
            "gpu": pilot.gpu,
            "running": running,
            "pid": pid,
            "checkpoint": checkpoints[-1].name if checkpoints else None,
            "complete": _valid_summary(summary, manifest, pilot, level),
        }
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--level", type=int, choices=(1, 2), default=1)
    parser.add_argument("--pilots", nargs="+", help="optional manifest pilot subset")
    parser.add_argument(
        "--phase", choices=("dry-run", "preflight", "launch", "status"), required=True
    )
    args = parser.parse_args(argv)
    manifest = load_pilot_manifest(args.manifest)
    specs = build_launch_specs(manifest, level=args.level)
    if args.pilots:
        unknown = sorted(set(args.pilots) - set(manifest.pilots))
        if unknown:
            raise ValueError(f"unknown pilot(s): {unknown}")
        requested = set(args.pilots)
        specs = tuple(spec for spec in specs if spec.pilot in requested)
    if args.phase == "dry-run":
        print(json.dumps([
            {
                "pilot": spec.pilot,
                "gpu": spec.gpu,
                "command": list(spec.command),
                "manifest_sha256": manifest.sha256,
                "environment": dict(spec.environment),
            }
            for spec in specs
        ], indent=2))
        return
    if args.phase == "status":
        print(json.dumps(_status(manifest, args.level), indent=2))
        return
    memory = _preflight(manifest, specs)
    if args.phase == "preflight":
        print(json.dumps({"status": "ok", "free_memory_mib": memory}, indent=2))
        return
    for spec in specs:
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        state_path = spec.output_dir / "process.json"
        summary_path = spec.output_dir / "summary.json"
        pilot = manifest.pilots[spec.pilot]
        if _valid_summary(summary_path, manifest, pilot, args.level):
            raise RuntimeError(f"pilot already completed: {spec.pilot}")
        if args.level == 2:
            level_one = pilot.output_dir / "level_1" / "summary.json"
            if not _valid_summary(level_one, manifest, pilot, 1):
                raise RuntimeError(f"pilot has no valid Level-1 result: {spec.pilot}")
            decision = json.loads(level_one.read_text(encoding="utf-8")).get("decision")
            if decision != "promote":
                raise RuntimeError(f"pilot was not promoted: {spec.pilot}")
        if state_path.is_file():
            state = json.loads(state_path.read_text())
            if state.get("pid") and _pid_matches(state["pid"], state.get("command", ())):
                raise RuntimeError(f"pilot already running: {spec.pilot}")
        log = (spec.output_dir / "worker.log").open("a", encoding="utf-8")
        process = subprocess.Popen(
            spec.command,
            cwd=manifest.repository_root,
            env={**os.environ, **spec.environment},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _atomic_json(
            state_path,
            {
                "pid": process.pid,
                "pilot": spec.pilot,
                "gpu": spec.gpu,
                "started_at": time.time(),
                "command": list(spec.command),
                "manifest_sha256": manifest.sha256,
            },
        )
        print(f"started {spec.pilot} pid={process.pid} GPU{spec.gpu}")


if __name__ == "__main__":
    main()
