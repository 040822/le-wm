"""Reusable optimizer-update success-rate curve artifacts for Round 3."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def optimizer_update_steps(max_updates: int = 200, interval: int = 10) -> tuple[int, ...]:
    """Return checkpoint boundaries for the Round3 update-step curve."""
    maximum = int(max_updates)
    stride = int(interval)
    if maximum < 0 or stride < 1:
        raise ValueError("max_updates must be non-negative and interval must be positive")
    if maximum % stride:
        raise ValueError("max_updates must be divisible by interval")
    return tuple(range(0, maximum + 1, stride))


def make_curve_row(
    *,
    optimizer_update_step: int,
    environment_steps: int,
    success_rate: float,
    episodes: int,
    checkpoint: str,
    checkpoint_sha256: str | None,
    cohort_sha256: str,
    status: str = "ok",
    skipped_reason: str | None = None,
) -> dict[str, Any]:
    """Normalize one evaluation result to the update-step curve schema."""
    rate = float(success_rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError("success_rate must be in [0, 1]")
    if int(optimizer_update_step) < 0 or int(environment_steps) < 0:
        raise ValueError("curve steps must be non-negative")
    if int(episodes) < 1:
        raise ValueError("episodes must be positive")
    if not cohort_sha256:
        raise ValueError("cohort_sha256 is required")
    return {
        "optimizer_update_step": int(optimizer_update_step),
        "environment_steps": int(environment_steps),
        "success_rate": rate,
        "success_rate_percent": rate * 100.0,
        "episodes": int(episodes),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "cohort_sha256": str(cohort_sha256),
        "status": str(status),
        "skipped_reason": skipped_reason,
    }


def validate_curve_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_steps: Iterable[int] = optimizer_update_steps(),
    expected_episodes: int = 200,
    cohort_sha256: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Validate ordering, coverage, episode count, and cohort identity."""
    normalized = tuple(dict(row) for row in rows)
    expected = tuple(int(step) for step in expected_steps)
    actual = tuple(int(row["optimizer_update_step"]) for row in normalized)
    if actual != expected:
        raise ValueError(f"curve update steps {actual} do not match expected {expected}")
    for row in normalized:
        if int(row.get("episodes", -1)) != int(expected_episodes):
            raise ValueError("curve point has the wrong final-cohort episode count")
        if row.get("status") != "ok":
            raise ValueError("curve contains a non-ok point")
        if cohort_sha256 is not None and row.get("cohort_sha256") != cohort_sha256:
            raise ValueError("curve points use different final cohort hashes")
    return normalized


def write_update_curve(
    output_dir: str | Path,
    *,
    task: str,
    experiment: str,
    arm: str,
    rows: Iterable[Mapping[str, Any]],
    cohort: Mapping[str, Any],
    visible_physical_gpus: Iterable[int | str],
    max_updates: int = 200,
    interval: int = 10,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Atomically publish JSON/CSV update-step curve artifacts."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    rows = tuple(dict(row) for row in rows)
    expected = optimizer_update_steps(max_updates, interval)
    complete = (
        len(rows) == len(expected)
        and tuple(int(row["optimizer_update_step"]) for row in rows) == expected
        and all(row.get("status") == "ok" for row in rows)
    )
    if require_complete:
        validate_curve_rows(
            rows,
            expected_steps=expected,
            expected_episodes=int(cohort.get("episodes", 200)),
            cohort_sha256=str(cohort["canonical_sha256"]),
        )
    metadata = {
        "schema_version": 2,
        "status": "ok" if complete else "running",
        "task": str(task),
        "experiment": str(experiment),
        "arm": str(arm),
        "step_unit": "optimizer_update_step",
        "step_increment": int(interval),
        "max_optimizer_updates": int(max_updates),
        "evaluation_steps": list(expected),
        "cohort": dict(cohort),
        "visible_physical_gpus": [str(value) for value in visible_physical_gpus],
        "evaluation_batching": {
            "batch_size": 50,
            "batch_count": 4,
            "aggregate_episode_count": 200,
            "reason": "avoid EGL framebuffer failure from 200 simultaneous renderers",
        },
        "results": list(rows),
    }
    json_tmp = target / "curve.json.tmp"
    json_tmp.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    json_tmp.replace(target / "curve.json")
    csv_tmp = target / "curve.csv.tmp"
    fields = (
        "optimizer_update_step",
        "environment_steps",
        "success_rate",
        "success_rate_percent",
        "episodes",
        "checkpoint",
        "checkpoint_sha256",
        "cohort_sha256",
        "status",
        "skipped_reason",
    )
    with csv_tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    csv_tmp.replace(target / "curve.csv")
    return metadata


__all__ = [
    "make_curve_row",
    "optimizer_update_steps",
    "validate_curve_rows",
    "write_update_curve",
]
