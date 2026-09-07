"""CLI launcher for the four-way Fast-LeWAM attention pilots.

The launcher is intentionally conservative: dry-run and status are read-only,
preflight never changes the requested batch, and launch refuses an existing
or identity-mismatched output instead of silently rerunning it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Callable, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from source.experiments.fast_lewam_attention_pilots import (
    LaunchSpec,
    MemorySnapshot,
    PilotManifest,
    StaleArtifactError,
    atomic_write_json,
    build_launch_specs,
    expected_environment,
    load_pilot_manifest,
    memory_guard_ok,
    parse_pmon_processes,
    path_identity,
    validate_manifest_inputs,
    validate_memory_guard,
)

# Kept as a module-level alias for tests and for callers migrating from the
# older parallel-pilot launcher.
_parse_pmon_processes = parse_pmon_processes


def build_fast_dev_run_command(spec: LaunchSpec) -> tuple[str, ...]:
    """Build the one-batch fast-dev preflight command for a pilot card."""
    return (
        *spec.command,
        "--fast-dev-run",
    )


def _run_fast_dev(spec: LaunchSpec, manifest: PilotManifest) -> dict[str, object]:
    command = build_fast_dev_run_command(spec)
    log_path = spec.output_dir / "preflight" / "fast_dev_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, int]] = []
    stop = threading.Event()

    def monitor() -> None:
        while not stop.is_set():
            try:
                sample = dict(_query_gpu_memory(spec.gpu, spec.environment))
                samples.append({
                    "timestamp_ns": time.time_ns(),
                    "total_mib": int(sample["total_mib"]),
                    "used_mib": int(sample["used_mib"]),
                    "free_mib": int(sample["free_mib"]),
                })
            except (OSError, RuntimeError, subprocess.SubprocessError, ValueError):
                # The preflight result remains fail-closed if no valid sample
                # is available; transient nvidia-smi startup errors are retried.
                pass
            stop.wait(0.25)

    thread = threading.Thread(target=monitor, name=f"fast-dev-gpu{spec.gpu}", daemon=True)
    thread.start()
    try:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write("COMMAND " + " ".join(command) + "\n")
            stream.flush()
            subprocess.run(
                list(command),
                cwd=manifest.repository_root,
                env={**os.environ, **dict(spec.environment)},
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
    finally:
        stop.set()
        thread.join(timeout=5.0)
    if not samples:
        raise RuntimeError(f"GPU{spec.gpu} fast-dev preflight collected no memory samples")
    peak = max(samples, key=lambda row: row["used_mib"])
    final = samples[-1]
    atomic_write_json(
        spec.output_dir / "preflight" / "memory_samples.json",
        {"gpu": spec.gpu, "samples": samples, "peak": peak, "post_test": final},
    )
    return {"peak": peak, "post_test": final, "samples": samples}


def _pid_matches(pid: int, command: Sequence[str]) -> bool:
    if not command:
        return False
    path = Path(f"/proc/{int(pid)}/cmdline")
    try:
        actual = tuple(value.decode(errors="replace") for value in path.read_bytes().split(b"\0") if value)
    except (OSError, ValueError):
        return False
    expected = tuple(str(value) for value in command)
    return len(actual) >= len(expected) and actual[-len(expected):] == expected


def _query_gpu_memory(gpu: int, environment: Mapping[str, str]) -> dict[str, int]:
    result = subprocess.run(
        [
            "nvidia-smi", "-i", str(int(gpu)),
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, **dict(environment)},
    )
    fields = [part.strip() for part in result.stdout.strip().split(",")]
    if len(fields) != 3:
        raise RuntimeError(f"unexpected nvidia-smi memory output for GPU{gpu}: {result.stdout!r}")
    try:
        total, used, free = (int(part) for part in fields)
    except ValueError as exc:
        raise RuntimeError(f"non-numeric nvidia-smi memory output: {result.stdout!r}") from exc
    return {"total_mib": total, "used_mib": used, "free_mib": free}


def _free_memory(gpu: int, environment: Mapping[str, str]) -> int:
    return _query_gpu_memory(gpu, environment)["free_mib"]


def _gpu_compute_processes(gpu: int, environment: Mapping[str, str]) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "pmon", "-i", str(int(gpu)), "-c", "1"],
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, **dict(environment)},
    )
    return _parse_pmon_processes(result.stdout, gpu)


def _preflight(
    manifest: PilotManifest,
    specs: Sequence[LaunchSpec],
    *,
    allow_existing_compute: bool = False,
    pmon_reader: Callable[[int, Mapping[str, str]], Sequence[str]] | None = None,
    memory_reader: Callable[[int, Mapping[str, str]], Mapping[str, int]] | None = None,
    fast_dev_runner: Callable[[LaunchSpec, PilotManifest], Mapping[str, object] | None] | None = None,
) -> dict[int, MemorySnapshot]:
    """Check inputs, ownership, and memory without starting a worker."""
    validate_manifest_inputs(manifest)
    pmon_reader = pmon_reader or _gpu_compute_processes
    memory_reader = memory_reader or _query_gpu_memory
    fast_dev_runner = fast_dev_runner or _run_fast_dev
    snapshots: dict[int, MemorySnapshot] = {}
    for spec in specs:
        if spec.gpu not in {0, 1, 2, 3}:
            raise RuntimeError(f"pilot {spec.pilot} selected prohibited GPU{spec.gpu}")
        processes = list(pmon_reader(spec.gpu, spec.environment))
        if processes and not allow_existing_compute:
            raise RuntimeError(
                f"GPU{spec.gpu} already has compute processes; pass "
                "--allow-existing-compute explicitly to continue: "
                + ", ".join(processes)
            )
        before = dict(memory_reader(spec.gpu, spec.environment))
        # Exercise exactly one fast-dev batch with the canonical pilot command.
        # This is a preflight only; the real worker is not started here.
        fast_dev = fast_dev_runner(spec, manifest)
        # A second sample is the post-test guard. We do not launch or mutate
        # batch settings when either the initial or final safety check fails.
        after = dict(memory_reader(spec.gpu, spec.environment))
        peak_used = None
        if isinstance(fast_dev, Mapping):
            peak = fast_dev.get("peak")
            if isinstance(peak, Mapping) and "used_mib" in peak:
                peak_used = int(peak["used_mib"])
        snapshots[spec.gpu] = validate_memory_guard(
            spec.gpu,
            total_mib=int(before["total_mib"]),
            used_mib=int(before["used_mib"]),
            free_mib=int(before["free_mib"]),
            post_test_free_mib=int(after["free_mib"]),
            peak_used_mib=peak_used,
        )
    return snapshots


def _read_json_if_file(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise StaleArtifactError(f"invalid JSON state: {path}") from exc
    if not isinstance(value, dict):
        raise StaleArtifactError(f"JSON state must be an object: {path}")
    return value


def _status(manifest: PilotManifest, *, through_epoch: int = 10) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for pilot in sorted(manifest.pilots.values(), key=lambda item: item.gpu):
        process_path = pilot.output_dir / "process.json"
        process = _read_json_if_file(process_path)
        command = process.get("command", ())
        pid = process.get("pid")
        running = bool(
            pid
            and process.get("manifest_sha256") == manifest.sha256
            and _pid_matches(int(pid), command)
        )
        markers = {}
        for epoch in (1, 3, 10):
            marker = pilot.output_dir / "phases" / f"epoch_{epoch}.json"
            value = _read_json_if_file(marker)
            if value:
                markers[str(epoch)] = {
                    "status": value.get("status"),
                    "identity": value.get("identity"),
                }
        rows[pilot.name] = {
            "pilot": pilot.name,
            "gpu": pilot.gpu,
            "task": pilot.task,
            "attention_variant": pilot.attention_variant,
            "pid": pid,
            "running": running,
            "command_matches": bool(pid and _pid_matches(int(pid), command)),
            "process_identity_matches": process.get("manifest_sha256") == manifest.sha256,
            "through_epoch": int(through_epoch),
            "phases": markers,
            "output_dir": str(pilot.output_dir),
        }
    return rows


def _summary(manifest: PilotManifest, *, through_epoch: int = 10) -> dict:
    through_epoch = int(through_epoch)
    if through_epoch not in (1, 3, 10):
        raise ValueError("through_epoch must be one of 1, 3, or 10")
    rows = _status(manifest, through_epoch=through_epoch)
    for name, row in rows.items():
        pilot = manifest.pilots[name]
        row["summary"] = {}
        for epoch in (1, 3, 10):
            path = pilot.output_dir / "summaries" / f"epoch_{epoch}.json"
            if path.is_file():
                row["summary"][str(epoch)] = _read_json_if_file(path)

    def terminal(row: Mapping[str, object]) -> bool:
        summaries = row.get("summary", {})
        if not isinstance(summaries, Mapping):
            return False
        if through_epoch == 1:
            value = summaries.get("1")
            return isinstance(value, Mapping) and value.get("status") == "ok"
        if through_epoch == 3:
            value = summaries.get("3")
            return isinstance(value, Mapping) and value.get("status") == "ok"
        epoch3 = summaries.get("3")
        if isinstance(epoch3, Mapping) and epoch3.get("decision") == "stop":
            return True
        epoch10 = summaries.get("10")
        return isinstance(epoch10, Mapping) and epoch10.get("status") == "ok"

    all_terminal = all(terminal(row) for row in rows.values())
    any_running = any(bool(row.get("running")) for row in rows.values())
    status = "ok" if all_terminal else ("running" if any_running else "incomplete")
    summary = {
        "schema_version": 1,
        "status": status,
        "manifest_sha256": manifest.sha256,
        "through_epoch": through_epoch,
        "pilots": rows,
    }
    atomic_write_json(manifest.output_root / "summary.json", summary)
    return summary


def _launch_one(spec: LaunchSpec, manifest: PilotManifest) -> dict:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    process_path = spec.output_dir / "process.json"
    existing = _read_json_if_file(process_path)
    if existing:
        if existing.get("manifest_sha256") != manifest.sha256:
            raise StaleArtifactError(f"process identity mismatch: {process_path}")
        if existing.get("pid") and _pid_matches(existing["pid"], existing.get("command", ())):
            raise RuntimeError(f"pilot is already running: {spec.pilot}")
        if existing.get("pid") and not existing.get("finished_at"):
            # Reconcile a naturally exited worker before deciding whether a
            # subsequent explicit launch is allowed.
            _finish_process_marker(
                manifest,
                spec.pilot,
                returncode=existing.get("returncode", 0),
            )
            existing = _read_json_if_file(process_path)
        if not existing.get("finished_at"):
            raise RuntimeError(
                f"pilot has an unfinished process marker; archive it explicitly before retry: {process_path}"
            )
    log_path = spec.output_dir / "worker.log"
    log = log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            list(spec.command),
            cwd=manifest.repository_root,
            env={**os.environ, **dict(spec.environment)},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        log.close()
        raise
    atomic_write_json(
        process_path,
        {
            "schema_version": 1,
            "pid": process.pid,
            "pilot": spec.pilot,
            "gpu": spec.gpu,
            "started_at": time.time(),
            "manifest_sha256": manifest.sha256,
            "command": list(spec.command),
            "environment": dict(spec.environment),
            "through_epoch": spec.through_epoch,
        },
    )
    return {"pilot": spec.pilot, "gpu": spec.gpu, "pid": process.pid}


def _finish_process_marker(manifest: PilotManifest, pilot_name: str, *, returncode: int) -> None:
    pilot = manifest.pilots[pilot_name]
    path = pilot.output_dir / "process.json"
    value = _read_json_if_file(path)
    if not value:
        return
    if value.get("manifest_sha256") != manifest.sha256:
        raise StaleArtifactError(f"process identity mismatch: {path}")
    value["finished_at"] = time.time()
    value["returncode"] = int(returncode)
    atomic_write_json(path, value)


def _select_pilots(args) -> tuple[str, ...] | None:
    selected = args.pilots or args.pilot
    if selected is None:
        return None
    if isinstance(selected, str):
        selected = [selected]
    return tuple(selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--phase", required=True,
        choices=("dry-run", "preflight", "launch", "status", "summarize"),
    )
    parser.add_argument(
        "--through-epoch", "--epoch", "--level",
        type=int, choices=(1, 3, 10), default=10,
    )
    parser.add_argument("--pilots", nargs="+")
    parser.add_argument("--pilot")
    parser.add_argument(
        "--allow-existing-compute",
        action="store_true",
        help="explicitly permit existing pmon compute processes; memory guards still apply",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_pilot_manifest(args.manifest)
    selected = _select_pilots(args)
    specs = build_launch_specs(
        manifest,
        through_epoch=args.through_epoch,
        pilots=selected,
    )
    if args.phase == "dry-run":
        print(json.dumps([
            {
                "pilot": spec.pilot,
                "gpu": spec.gpu,
                "through_epoch": spec.through_epoch,
                "command": list(spec.command),
                "environment": dict(spec.environment),
                "manifest_sha256": manifest.sha256,
            }
            for spec in specs
        ], indent=2))
        return 0
    if args.phase == "status":
        print(json.dumps(_status(manifest, through_epoch=args.through_epoch), indent=2))
        return 0
    if args.phase == "summarize":
        print(json.dumps(_summary(manifest, through_epoch=args.through_epoch), indent=2))
        return 0
    snapshots = _preflight(
        manifest,
        specs,
        allow_existing_compute=args.allow_existing_compute,
    )
    if args.phase == "preflight":
        print(json.dumps({
            "status": "ok",
            "manifest_sha256": manifest.sha256,
            "memory": {str(gpu): snapshot.__dict__ for gpu, snapshot in snapshots.items()},
        }, indent=2))
        return 0
    launched = [_launch_one(spec, manifest) for spec in specs]
    print(json.dumps({
        "status": "launched",
        "manifest_sha256": manifest.sha256,
        "pilots": launched,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
