"""Clean the explicitly approved old outputs; dry run unless --apply is supplied.

Remove verified duplicate Phase1.5 branch records, intermediate epoch weights
from Round3 and earlier runs, fine-grained Round3 Phase2 checkpoints, and all
Round3 replay shards and optimizer sidecars. Keep final weights, aggregates,
and the seven explicitly retained attention/ranking diagnostic checkpoints.
Uses only the Python standard library; can be run from any working directory.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"
EPOCH = re.compile(r"(.+)_weights_epoch_(\d+)\.pt$")
ENVSTEP = re.compile(r"(.+)_envsteps_(\d{6})\.ckpt$")
# User-selected diagnostic baselines. Historical epoch-5 checkpoints remain
# eligible for deletion under the normal intermediate-epoch rule.
KEEP_DIAGNOSTIC_WEIGHTS = {
    OUTPUTS / "fast_lewam" / run / "checkpoints" / f"fast_lewam_weights_epoch_{epoch}.pt"
    for run, epochs in (
        ("reacher/0809_e1_stage_b_only_seed3072_tmux", (1, 3)),
        ("pusht/0802_e1_stage_b_only", (1, 3)),
        ("reacher/0812_s32_e4_clean_t_192", (8,)),
        ("reacher/0812_s41_e3_physical_time_type_192", (8,)),
        ("pusht/0812_s41_e3_physical_time_type_192", (8,)),
    )
    for epoch in epochs
}


def walk_files(root: Path):
    """Yield regular files without traversing symbolic links."""
    if root.is_symlink() or not root.is_dir():
        return
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(base) / name).is_symlink()]
        for name in files:
            p = Path(base) / name
            if stat.S_ISREG(p.lstat().st_mode):
                yield p


def digest(paths: list[Path]) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    for p in paths:
        with p.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                h.update(block)
                size += len(block)
    return h.hexdigest(), size


def plan() -> dict[str, list[Path]]:
    protected = set()
    for root in (OUTPUTS, REPO / "data/checkpoints"):
        for base, dirs, files in os.walk(root, followlinks=False):
            for name in dirs + files:
                p = Path(base) / name
                if p.is_symlink():
                    protected.add(p.resolve())

    result: dict[str, list[Path]] = defaultdict(list)
    selected: set[Path] = set()

    def add(category: str, p: Path):
        # All categories protect link targets, hard links, and paths outside outputs.
        if (
            p in selected
            or p in KEEP_DIAGNOSTIC_WEIGHTS
            or not p.resolve().is_relative_to(OUTPUTS.resolve())
            or p.resolve() in protected
            or not stat.S_ISREG(p.lstat().st_mode)
            or p.stat().st_nlink != 1
        ):
            return
        selected.add(p)
        result[category].append(p)

    pool = OUTPUTS / "round5/phase1_5_seed3072_legacy/diagnostics/candidate_pool"
    for task in ("cube", "pusht", "reacher", "tworoom"):
        root = pool / task
        branches = sorted(
            p for p in walk_files(root / "branches")
            if p.parent.parent == root / "branches"
            and p.parent.name.startswith("s")
            and re.fullmatch(r"candidate_\d+\.jsonl", p.name)
        )
        if not branches:
            continue
        aggregate = root / "records.jsonl"
        if aggregate.is_symlink() or not aggregate.is_file():
            raise RuntimeError(f"Missing regular aggregate: {aggregate}")
        if digest(branches) != digest([aggregate]):
            raise RuntimeError(f"Branch contents differ from aggregate: {root}")
        print(f"Verified duplicate branches: {task} ({len(branches)} files)", flush=True)
        for p in branches:
            add("duplicate_branches", p)

    groups: dict[tuple[Path, str], list[tuple[int, Path]]] = defaultdict(list)
    for name in ("fast_lewam", "lewm", "experiments", "round3"):
        for p in walk_files(OUTPUTS / name):
            match = EPOCH.fullmatch(p.name)
            if match and p.parent.name == "checkpoints":
                groups[p.parent, match[1]].append((int(match[2]), p))
    for snapshots in groups.values():
        latest = max(epoch for epoch, _ in snapshots)
        for epoch, p in snapshots:
            if epoch < latest and epoch != 10:
                add("old_epoch_weights", p)

    envsteps: dict[tuple[Path, str], list[tuple[int, Path]]] = defaultdict(list)
    for p in walk_files(OUTPUTS / "round3"):
        if "replay_shards" in p.relative_to(OUTPUTS / "round3").parts[:-1]:
            add("round3_replay_shards", p)
        elif p.name.endswith(".ckpt.state.pt"):
            add("round3_optimizer_sidecars", p)
        elif p.is_relative_to(OUTPUTS / "round3/phase2") and p.parent.name == "checkpoints":
            match = ENVSTEP.fullmatch(p.name)
            if match:
                envsteps[p.parent, match[1]].append((int(match[2]), p))
    for snapshots in envsteps.values():
        latest = max(step for step, _ in snapshots)
        for step, p in snapshots:
            if step % 1000 and step != latest:
                add("round3_intermediate_checkpoints", p)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Actually unlink the planned files")
    parser.add_argument("--list", action="store_true", help="Print every planned file")
    args = parser.parse_args()
    if OUTPUTS.is_symlink() or not OUTPUTS.is_dir():
        parser.error("outputs must be a regular directory in this repository")
    # Finish every duplicate check before performing any deletion.
    targets = plan()
    identities = {}
    total = 0
    for category, paths in sorted(targets.items()):
        size = 0
        for p in sorted(paths):
            s = p.lstat()
            identities[p] = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_nlink)
            size += s.st_blocks * 512
            if args.list:
                print(f"{category}: {p.relative_to(REPO)}")
        total += size
        print(f"{category}: {len(paths)} files, {size / 2**30:.3f} GiB")
    print(f"TOTAL: {len(identities)} files, {total / 2**30:.3f} GiB; apply={args.apply}", flush=True)
    if args.apply:
        # Detect changed or replaced candidates before starting deletion.
        for p, identity in identities.items():
            s = p.lstat()
            if not stat.S_ISREG(s.st_mode) or identity != (
                s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_nlink
            ):
                raise RuntimeError(f"Candidate changed during planning: {p}")
        for p in identities:
            p.unlink()
        print(f"Deleted {len(identities)} files; empty directories retained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
