#!/usr/bin/env python3
"""Print structural and episode-level information for HDF5 datasets."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Iterable

import h5py
import numpy as np


H5_SUFFIXES = {".h5", ".hdf5"}
EPISODE_INDEX_KEYS = ("ep_idx", "episode_idx")


def find_h5_files(path: Path) -> list[Path]:
    """Resolve one HDF5 file or recursively find all HDF5 files in a directory."""
    if path.is_file():
        return [path] if path.suffix.lower() in H5_SUFFIXES else []
    if path.is_dir():
        return sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in H5_SUFFIXES
        )
    return []


def _lengths_from_episode_indices(dataset: h5py.Dataset) -> np.ndarray:
    """Count episode-index occurrences without loading a large index fully."""
    counts: Counter[int] = Counter()
    chunk_size = 1_000_000
    for start in range(0, dataset.shape[0], chunk_size):
        values = np.asarray(dataset[start : start + chunk_size]).reshape(-1)
        unique, frequency = np.unique(values, return_counts=True)
        counts.update(
            {int(index): int(count) for index, count in zip(unique, frequency)}
        )
    return np.asarray([counts[index] for index in sorted(counts)], dtype=np.int64)


def episode_lengths(handle: h5py.File) -> tuple[np.ndarray | None, str | None]:
    """Return episode lengths and the metadata source used to infer them."""
    if "ep_len" in handle and isinstance(handle["ep_len"], h5py.Dataset):
        raw_lengths = np.asarray(handle["ep_len"][...])
        if raw_lengths.ndim != 1:
            raise ValueError(
                f"/ep_len must be one-dimensional, got shape {raw_lengths.shape}"
            )
        if np.any(raw_lengths <= 0):
            raise ValueError("/ep_len must contain only positive values")
        lengths = np.asarray(raw_lengths, dtype=np.int64)
        return lengths, "/ep_len"

    for key in EPISODE_INDEX_KEYS:
        if key in handle and isinstance(handle[key], h5py.Dataset):
            return _lengths_from_episode_indices(handle[key]), f"/{key}"

    return None, None


def _format_shape(shape: tuple[int, ...]) -> str:
    return "(" + ", ".join(map(str, shape)) + ("," if len(shape) == 1 else "") + ")"


def _format_bytes(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}" if unit != "B" else f"{size} B"
        value /= 1024
    raise AssertionError("unreachable")


def _dataset_role(
    dataset: h5py.Dataset, episode_count: int | None, total_steps: int | None
) -> str:
    if dataset.ndim == 0:
        return "scalar"
    if total_steps is not None and dataset.shape[0] == total_steps:
        return "per-step"
    if episode_count is not None and dataset.shape[0] == episode_count:
        return "per-episode"
    return "other"


def _iter_tree(group: h5py.Group, prefix: str = "") -> Iterable[tuple[str, object]]:
    for name in sorted(group.keys()):
        value = group[name]
        path = f"{prefix}/{name}"
        yield path, value
        if isinstance(value, h5py.Group):
            yield from _iter_tree(value, path)


def _print_episode_summary(lengths: np.ndarray, summary_only: bool) -> None:
    count = int(lengths.size)
    total = int(lengths.sum())
    print(f"Episodes: {count:,}")
    print(f"Total steps: {total:,}")
    print(
        "Episode length: "
        f"min={int(lengths.min()):,}, max={int(lengths.max()):,}, "
        f"mean={float(lengths.mean()):.2f}, median={float(np.median(lengths)):.2f}"
    )

    distribution = Counter(int(value) for value in lengths)
    distribution_text = ", ".join(
        f"{length:,} x {frequency:,}"
        for length, frequency in sorted(distribution.items())
    )
    print(f"Length distribution (length x episodes): {distribution_text}")

    print("Per-episode lengths:")
    if summary_only and count > 20:
        indices = list(range(10)) + list(range(count - 10, count))
    else:
        indices = range(count)
    for index in indices:
        print(f"  episode {index:>6}: {int(lengths[index]):,}")
    if summary_only and count > 20:
        print(
            f"  ... {count - 20:,} episodes omitted; "
            "run without --summary-only to print every episode"
        )


def inspect_file(path: Path, summary_only: bool = False) -> None:
    """Print metadata, episode statistics, and the complete HDF5 tree."""
    print(f"\n{'=' * 80}\nFile: {path}\nSize: {_format_bytes(path.stat().st_size)}")
    with h5py.File(path, "r") as handle:
        lengths, source = episode_lengths(handle)
        episode_count = int(lengths.size) if lengths is not None else None
        total_steps = int(lengths.sum()) if lengths is not None else None

        if lengths is None or lengths.size == 0:
            print("Episodes: unknown (no ep_len or episode index found)")
        else:
            print(f"Episode metadata source: {source}")
            _print_episode_summary(lengths, summary_only)

        root_kind = "flat time-major datasets" if all(
            isinstance(value, h5py.Dataset) for value in handle.values()
        ) else "hierarchical groups/datasets"
        print(f"Organization: {root_kind}")

        if handle.attrs:
            print("Root attributes:")
            for key, value in sorted(handle.attrs.items()):
                print(f"  @{key}: {value!r}")

        print("Data tree (path: kind, shape, dtype, role, storage):")
        for item_path, value in _iter_tree(handle):
            depth = item_path.count("/") - 1
            indent = "  " * depth
            if isinstance(value, h5py.Group):
                print(f"  {indent}{item_path}/: group")
                continue
            role = _dataset_role(value, episode_count, total_steps)
            storage = []
            if value.chunks is not None:
                storage.append(f"chunks={_format_shape(value.chunks)}")
            if value.compression is not None:
                storage.append(f"compression={value.compression}")
            storage_text = ", ".join(storage) if storage else "contiguous"
            print(
                f"  {indent}{item_path}: dataset, "
                f"shape={_format_shape(value.shape)}, dtype={value.dtype}, "
                f"role={role}, storage={storage_text}"
            )
            for key, attr_value in sorted(value.attrs.items()):
                print(f"    {indent}@{key}: {attr_value!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one HDF5 dataset or all .h5/.hdf5 files under a directory."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("data/datasets"),
        help="HDF5 file or directory (default: data/datasets)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="preview only the first and last 10 episode lengths",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = find_h5_files(args.path)
    if not files:
        print(f"No .h5 or .hdf5 files found at: {args.path}", file=sys.stderr)
        return 2

    failures = 0
    for path in files:
        try:
            inspect_file(path, args.summary_only)
        except (OSError, ValueError) as error:
            failures += 1
            print(f"Failed to inspect {path}: {error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
