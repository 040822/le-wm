"""Prune redundant Round 3 Phase 2 checkpoints after a disk-full stop.

Only checkpoint files directly under each explicitly supplied ``checkpoints``
directory are considered.  One checkpoint per 1,000 environment steps and the
latest available checkpoint are retained; optimizer sidecars follow the same
rule.  Replay shards, run state, evaluations, and traces are never touched.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


CHECKPOINT_RE = re.compile(r".+_envsteps_(\d{6})\.ckpt(?:\.state\.pt)?$")


def plan(root: Path) -> tuple[list[Path], list[Path]]:
    checkpoint_dir = root / "checkpoints"
    files = sorted(path for path in checkpoint_dir.iterdir() if path.is_file())
    checkpoint_steps: dict[Path, int] = {}
    for path in files:
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match is None:
            continue
        checkpoint_steps[path] = int(match.group(1))
    if not checkpoint_steps:
        return [], [path for path in files if path.name.endswith(".tmp")]
    latest = max(checkpoint_steps.values())
    remove = [
        path
        for path, step in checkpoint_steps.items()
        if step % 1000 != 0 and step != latest
    ]
    remove.extend(path for path in files if path.name.endswith(".tmp"))
    return sorted(set(remove)), []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    total_bytes = 0
    total_files = 0
    for root in args.roots:
        remove, _ = plan(root)
        size = sum(path.stat().st_size for path in remove if path.is_file())
        total_files += len(remove)
        total_bytes += size
        print(f"{root}: files={len(remove)} bytes={size} apply={args.apply}")
        if args.apply:
            for path in remove:
                if path.exists():
                    path.unlink()
    print(f"total_files={total_files} total_bytes={total_bytes} apply={args.apply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
