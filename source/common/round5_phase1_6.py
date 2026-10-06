"""Protocol, sampling, and paired-analysis helpers for Round 5 Phase1.6."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from . import round3_phase1 as phase1


SCHEMA_VERSION = "round5_phase1_6_v1"
TASKS = ("cube", "pusht", "reacher", "tworoom")
MECHANISM_TASKS = ("pusht", "reacher")


def canonical_json(value: Any) -> bytes:
    """Serialize JSON values deterministically for stable IDs and hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def stable_hash(value: Any) -> str:
    """Return the SHA-256 digest of canonical JSON bytes."""

    return hashlib.sha256(canonical_json(value)).hexdigest()


def stable_seed(*parts: Any) -> int:
    """Derive a process-independent 64-bit seed from a logical identity."""

    digest = hashlib.sha256(canonical_json(parts)).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def rng_for(*parts: Any) -> np.random.Generator:
    """Create an RNG whose stream is independent of workers and shard order."""

    return np.random.default_rng(stable_seed(*parts))


def atomic_json(path: str | Path, value: Any) -> Path:
    """Atomically replace a JSON artifact in its destination directory."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def build_confirmation_cohort(
    dataset: Any,
    *,
    task: str,
    count: int,
    seed: int,
    goal_offset_steps: int,
    excluded_episode_ids: Sequence[Any],
) -> phase1.CohortManifest:
    """Sample at most one initially-failing start per non-historical episode."""

    episode_column = phase1._episode_column(dataset)
    all_episode_ids = np.unique(phase1._column(dataset, episode_column))
    candidates, diagnostics = phase1._collect_candidates(
        dataset,
        task=task,
        episode_ids=all_episode_ids,
        seed=int(seed) + 17,
        goal_offset_steps=int(goal_offset_steps),
    )
    excluded_keys = {phase1._episode_key(value) for value in excluded_episode_ids}
    eligible = [
        item for item in candidates
        if phase1._episode_key(item.episode_id) not in excluded_keys
    ]
    rng = rng_for("phase1_6_confirmation", task, int(seed))
    order = rng.permutation(len(eligible))
    chosen = [eligible[int(index)] for index in order[: int(count)]]
    if len(chosen) != int(count):
        raise ValueError(
            f"{task} has only {len(chosen)} eligible non-historical episodes; "
            f"requested {count}"
        )
    entries = tuple(
        phase1.CohortEntry(
            row_index=int(item.row_index),
            episode_id=item.episode_id,
            start_step=int(item.start_step),
            goal_row_index=int(item.goal_row_index),
            goal_step=int(item.goal_step),
            start_distance=float(item.distance),
            initially_successful=False,
            stratum=None,
            start_state=tuple(float(value) for value in item.start_state),
            goal_state=tuple(float(value) for value in item.goal_state),
        )
        for item in chosen
    )
    selected_keys = {phase1._episode_key(item.episode_id) for item in chosen}
    excluded_rows = [
        phase1._jsonable(value)
        for value in all_episode_ids
        if phase1._episode_key(value) in excluded_keys
    ]
    online_rows = [
        phase1._jsonable(value)
        for value in all_episode_ids
        if phase1._episode_key(value) not in selected_keys | excluded_keys
    ]
    diagnostics = {
        **diagnostics,
        "raw_episode_count": int(len(all_episode_ids)),
        "historical_episode_exclusions": int(len(excluded_rows)),
        "eligible_nonhistorical_episode_count": int(len(eligible)),
        "selected_episode_count": int(len(entries)),
    }
    return phase1.CohortManifest(
        task=task,
        cohort_id=f"{task}_phase1_6_confirm_{int(seed)}_v1",
        cohort_kind="final",
        protocol_variant="round3_revised",
        seed=int(seed),
        goal_offset_steps=int(goal_offset_steps),
        entries=entries,
        episode_split={
            "dev": (),
            "final": tuple(item.episode_id for item in chosen),
            "online": tuple(online_rows),
            "excluded_historical": tuple(excluded_rows),
        },
        strata_boundaries=(),
        candidate_counts=(len(eligible),),
        selected_counts=(len(entries),),
        sampling_rule={
            "algorithm": "one_valid_start_per_raw_episode_then_seeded_uniform_episode_sample",
            "sampling_seed": int(seed),
            "max_one_start_per_episode": True,
            "exclude_initial_success": True,
            "excluded_historical_episode_count": len(excluded_rows),
            "model_independent": True,
        },
        diagnostics=diagnostics,
    )


def state_cluster_bootstrap(
    differences_by_state: Mapping[str, Sequence[float]],
    *,
    samples: int = 10_000,
    seed: int = 20_260_501,
) -> dict[str, Any]:
    """Bootstrap state means; repeated rows within a state remain clustered."""

    state_ids = sorted(differences_by_state)
    values = [np.asarray(differences_by_state[key], dtype=np.float64) for key in state_ids]
    valid = [array[np.isfinite(array)] for array in values]
    keep = [(key, array) for key, array in zip(state_ids, valid) if len(array)]
    if not keep:
        return {
            "estimate": None,
            "ci95": [None, None],
            "states": 0,
            "rows": 0,
            "bootstrap_samples": int(samples),
            "seed": int(seed),
        }
    means = np.asarray([array.mean() for _, array in keep], dtype=np.float64)
    rng = rng_for("state_cluster_bootstrap", int(seed))
    indices = rng.integers(0, len(means), size=(int(samples), len(means)))
    draws = means[indices].mean(axis=1)
    return {
        "estimate": float(means.mean()),
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "states": int(len(means)),
        "rows": int(sum(len(array) for _, array in keep)),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
    }


def paired_state_metrics(
    left: Mapping[str, Sequence[float]],
    right: Mapping[str, Sequence[float]],
    *,
    samples: int = 10_000,
    seed: int = 20_260_501,
) -> dict[str, Any]:
    """Summarize paired left-minus-right rows after within-state aggregation."""

    differences: dict[str, list[float]] = {}
    for state in sorted(set(left) & set(right)):
        a = np.asarray(left[state], dtype=np.float64).reshape(-1)
        b = np.asarray(right[state], dtype=np.float64).reshape(-1)
        count = min(len(a), len(b))
        valid = np.isfinite(a[:count]) & np.isfinite(b[:count])
        if np.any(valid):
            differences[state] = (a[:count][valid] - b[:count][valid]).tolist()
    return state_cluster_bootstrap(differences, samples=samples, seed=seed)


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Atomically write a JSONL shard."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_json(dict(row)) + b"\n" for row in rows)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return target
