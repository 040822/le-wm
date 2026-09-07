"""Round 3 Phase 1 protocol, cohort, trace, and analysis utilities.

The existing evaluator remains the compatibility entry point for the historic
LeWM protocol.  This module owns the new, auditable surface used by Phase 1:
cohort manifests, artifact registration, predicate-aware trace summaries, and
paired analysis.  It intentionally keeps model loading and environment
construction lazy so the deterministic parts can be tested without a GPU.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .round3_protocol import (
    ARTIFACT_REGISTRY_SCHEMA_VERSION,
    COHORT_SCHEMA_VERSION,
    FAST_STAGES,
    METHODS,
    PREDICATE_VERSION,
    PROTOCOL_VARIANTS,
    ROUND3_EVAL_DEFAULTS,
    ROUND3_PROTOCOL,
    TASKS,
    TRACE_SCHEMA_VERSION,
    evaluate_success,
    physical_distance,
    resolve_task_field,
)


PHASE1_DEV_EPISODES = 50
PHASE1_FINAL_EPISODES = 200
PHASE1_STRATA = 5
PHASE1_STRATUM_TARGETS = {50: 10, 200: 40}
_WEIGHT_EPOCH_RE = re.compile(r".+_weights_epoch_(\d+)\.pt$")


def _jsonable(value: Any) -> Any:
    """Convert numpy/path values into stable JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Hash a JSON value using the same canonical representation as manifests."""
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading a checkpoint into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _episode_key(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)


def _episode_sort_key(value: Any) -> tuple[int, Any]:
    value = _jsonable(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 0, value
    return 1, _episode_key(value)


def _same_value(left: Any, right: Any) -> bool:
    try:
        return bool(np.asarray(left == right).all())
    except (TypeError, ValueError):
        return _episode_key(left) == _episode_key(right)


def _episode_column(dataset: Any) -> str:
    columns = tuple(str(column) for column in dataset.column_names)
    for name in ("episode_idx", "ep_idx"):
        if name in columns:
            return name
    raise ValueError("dataset must expose episode_idx or ep_idx")


def _column(dataset: Any, name: str) -> np.ndarray:
    try:
        values = dataset.get_col_data(name)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"dataset column {name!r} is not available") from exc
    values = np.asarray(values)
    if values.ndim == 0:
        raise ValueError(f"dataset column {name!r} must have a row dimension")
    return values


def _row_value(values: np.ndarray, row: int) -> np.ndarray:
    value = np.asarray(values[int(row)])
    return value.astype(np.float64, copy=False)


def _state_columns(dataset: Any, task: str) -> tuple[str, str | None]:
    columns = tuple(str(column) for column in dataset.column_names)
    current = resolve_task_field(columns, task)
    goal = resolve_task_field(columns, task, goal=True)
    if current is None:
        predicate = task.lower().replace("push-t", "pusht").replace("two_room", "tworoom")
        raise ValueError(
            f"dataset for {predicate!r} has no field for the audited current state"
        )
    return current, goal


def _row_lookup(episode_ids: np.ndarray, steps: np.ndarray) -> dict[tuple[str, int], int]:
    return {
        (_episode_key(episode), int(step)): index
        for index, (episode, step) in enumerate(zip(episode_ids, steps))
    }


@dataclass(frozen=True)
class CohortEntry:
    """One selected start and the future row used as its deterministic goal."""

    row_index: int
    episode_id: Any
    start_step: int
    goal_row_index: int | None
    goal_step: int | None
    start_distance: float | None
    initially_successful: bool | None
    stratum: int | None = None
    start_state: tuple[float, ...] | None = None
    goal_state: tuple[float, ...] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_index": int(self.row_index),
            "episode_id": _jsonable(self.episode_id),
            "start_step": int(self.start_step),
            "goal_row_index": (
                int(self.goal_row_index) if self.goal_row_index is not None else None
            ),
            "goal_step": int(self.goal_step) if self.goal_step is not None else None,
            "start_distance": (
                float(self.start_distance) if self.start_distance is not None else None
            ),
            "initially_successful": self.initially_successful,
            "stratum": int(self.stratum) if self.stratum is not None else None,
            "start_state": _jsonable(self.start_state),
            "goal_state": _jsonable(self.goal_state),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CohortEntry":
        start_state = value.get("start_state")
        goal_state = value.get("goal_state")
        return cls(
            row_index=int(value["row_index"]),
            episode_id=value["episode_id"],
            start_step=int(value["start_step"]),
            goal_row_index=(
                int(value["goal_row_index"])
                if value.get("goal_row_index") is not None
                else None
            ),
            goal_step=(
                int(value["goal_step"]) if value.get("goal_step") is not None else None
            ),
            start_distance=(
                float(value["start_distance"])
                if value.get("start_distance") is not None
                else None
            ),
            initially_successful=(
                bool(value["initially_successful"])
                if value.get("initially_successful") is not None
                else None
            ),
            stratum=(int(value["stratum"]) if value.get("stratum") is not None else None),
            start_state=(tuple(float(item) for item in start_state) if start_state is not None else None),
            goal_state=(tuple(float(item) for item in goal_state) if goal_state is not None else None),
        )


@dataclass(frozen=True)
class CohortManifest:
    """Self-hashing, serializable cohort selection record."""

    task: str
    cohort_id: str
    cohort_kind: str
    protocol_variant: str
    seed: int
    goal_offset_steps: int
    entries: tuple[CohortEntry, ...]
    episode_split: Mapping[str, tuple[Any, ...]]
    strata_boundaries: tuple[float, ...] = ()
    candidate_counts: tuple[int, ...] = ()
    selected_counts: tuple[int, ...] = ()
    sampling_rule: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = COHORT_SCHEMA_VERSION
    cohort_sha256: str | None = None

    def __post_init__(self):
        if self.cohort_sha256 is None:
            object.__setattr__(self, "cohort_sha256", self.computed_sha256)


    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "protocol": ROUND3_PROTOCOL,
            "task": self.task,
            "cohort_id": self.cohort_id,
            "cohort_kind": self.cohort_kind,
            "protocol_variant": self.protocol_variant,
            "seed": int(self.seed),
            "goal_offset_steps": int(self.goal_offset_steps),
            "entries": [entry.as_dict() for entry in self.entries],
            "episode_split": {
                str(key): [_jsonable(item) for item in value]
                for key, value in sorted(self.episode_split.items())
            },
            "strata_boundaries": [float(item) for item in self.strata_boundaries],
            "candidate_counts": [int(item) for item in self.candidate_counts],
            "selected_counts": [int(item) for item in self.selected_counts],
            "sampling_rule": _jsonable(dict(self.sampling_rule)),
            "diagnostics": _jsonable(dict(self.diagnostics)),
        }

    @property
    def computed_sha256(self) -> str:
        return sha256_json(self._payload())

    def as_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["cohort_sha256"] = self.computed_sha256
        return payload

    def to_evaluation_cohort(self):
        """Convert to the legacy evaluator's lightweight cohort type."""
        from .eval import EvaluationCohort

        return EvaluationCohort(
            row_indices=np.asarray([entry.row_index for entry in self.entries], dtype=np.int64),
            episode_ids=np.asarray([entry.episode_id for entry in self.entries]),
            start_steps=np.asarray([entry.start_step for entry in self.entries], dtype=np.int64),
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        return target

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, verify_hash: bool = True) -> "CohortManifest":
        if value.get("protocol") != ROUND3_PROTOCOL:
            raise ValueError("cohort does not use the Round 3 Phase 1 protocol")
        entries = tuple(CohortEntry.from_dict(item) for item in value.get("entries", ()))
        split = {
            str(key): tuple(items) for key, items in value.get("episode_split", {}).items()
        }
        manifest = cls(
            task=str(value["task"]),
            cohort_id=str(value["cohort_id"]),
            cohort_kind=str(value["cohort_kind"]),
            protocol_variant=str(value["protocol_variant"]),
            seed=int(value["seed"]),
            goal_offset_steps=int(value["goal_offset_steps"]),
            entries=entries,
            episode_split=split,
            strata_boundaries=tuple(float(item) for item in value.get("strata_boundaries", ())),
            candidate_counts=tuple(int(item) for item in value.get("candidate_counts", ())),
            selected_counts=tuple(int(item) for item in value.get("selected_counts", ())),
            sampling_rule=dict(value.get("sampling_rule", {})),
            diagnostics=dict(value.get("diagnostics", {})),
            schema_version=int(value.get("schema_version", COHORT_SCHEMA_VERSION)),
            cohort_sha256=value.get("cohort_sha256"),
        )
        if verify_hash and manifest.cohort_sha256 != manifest.computed_sha256:
            raise ValueError(
                f"cohort hash mismatch for {manifest.cohort_id}: "
                f"stored={manifest.cohort_sha256!r} computed={manifest.computed_sha256!r}"
            )
        return manifest

    @classmethod
    def load(cls, path: str | Path, *, verify_hash: bool = True) -> "CohortManifest":
        target = Path(path)
        return cls.from_dict(json.loads(target.read_text(encoding="utf-8")), verify_hash=verify_hash)


@dataclass(frozen=True)
class _Candidate:
    row_index: int
    episode_id: Any
    start_step: int
    goal_row_index: int
    goal_step: int
    distance: float
    initially_successful: bool
    start_state: tuple[float, ...]
    goal_state: tuple[float, ...]


def split_episode_ids(
    episode_ids: Iterable[Any],
    *,
    seed: int,
    dev_count: int = PHASE1_DEV_EPISODES,
    final_count: int = PHASE1_FINAL_EPISODES,
    online_count: int | None = None,
) -> dict[str, tuple[Any, ...]]:
    """Split raw episode IDs once, before any model-dependent operation."""
    unique = {_episode_key(value): value for value in episode_ids}
    ordered = [unique[key] for key in sorted(unique, key=lambda key: _episode_sort_key(unique[key]))]
    dev_count = int(dev_count)
    final_count = int(final_count)
    if dev_count < 0 or final_count < 0:
        raise ValueError("dev_count and final_count must be non-negative")
    remainder = len(ordered) - dev_count - final_count
    if online_count is None:
        online_count = remainder
    online_count = int(online_count)
    if online_count < 0 or dev_count + final_count + online_count > len(ordered):
        raise ValueError(
            f"episode split needs {dev_count + final_count + online_count} episodes, "
            f"but dataset contains {len(ordered)}"
        )
    permutation = np.random.default_rng(int(seed)).permutation(len(ordered))
    shuffled = [ordered[int(index)] for index in permutation]
    return {
        "dev": tuple(shuffled[:dev_count]),
        "final": tuple(shuffled[dev_count : dev_count + final_count]),
        "online": tuple(shuffled[dev_count + final_count : dev_count + final_count + online_count]),
    }


def _episode_rng(seed: int, episode_id: Any) -> np.random.Generator:
    digest = hashlib.sha256(f"{int(seed)}:{_episode_key(episode_id)}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(value)


def _collect_candidates(
    dataset: Any,
    *,
    task: str,
    episode_ids: Sequence[Any],
    goal_offset_steps: int,
    seed: int,
) -> tuple[list[_Candidate], dict[str, int]]:
    """Choose at most one valid start per raw episode, independent of models."""
    episode_column = _episode_column(dataset)
    episodes = _column(dataset, episode_column)
    steps = _column(dataset, "step_idx")
    if episodes.ndim != 1 or steps.ndim != 1 or len(episodes) != len(steps):
        raise ValueError("episode_idx/step_idx columns must be aligned one-dimensional arrays")
    current_column, _ = _state_columns(dataset, task)
    current_values = _column(dataset, current_column)
    lookup = _row_lookup(episodes, steps)
    rows_by_episode: dict[str, list[int]] = {}
    for index, episode in enumerate(episodes):
        rows_by_episode.setdefault(_episode_key(episode), []).append(index)
    candidates: list[_Candidate] = []
    diagnostics = {
        "episode_count": len(episode_ids),
        "episodes_without_valid_future": 0,
        "initial_success_excluded": 0,
        "invalid_state_excluded": 0,
    }
    offset = int(goal_offset_steps)
    for episode_id in episode_ids:
        rows = np.asarray(rows_by_episode.get(_episode_key(episode_id), ()), dtype=np.int64)
        if len(rows) == 0:
            diagnostics["episodes_without_valid_future"] += 1
            continue
        episode_steps = np.asarray(steps[rows], dtype=np.int64)
        episode_length = int(np.max(episode_steps)) + 1
        valid_rows = [
            int(row)
            for row, step in zip(rows, episode_steps)
            if int(step) + offset < episode_length
            and (_episode_key(episode_id), int(step) + offset) in lookup
        ]
        if not valid_rows:
            diagnostics["episodes_without_valid_future"] += 1
            continue
        row_index = int(_episode_rng(seed, episode_id).choice(valid_rows))
        start_step = int(steps[row_index])
        goal_row_index = lookup[(_episode_key(episode_id), start_step + offset)]
        try:
            start_state = _row_value(current_values, row_index)
            goal_state = _row_value(current_values, goal_row_index)
            if start_state.ndim != 1 or goal_state.ndim != 1:
                raise ValueError("state columns must contain one-dimensional per-row vectors")
            if not np.all(np.isfinite(start_state)) or not np.all(np.isfinite(goal_state)):
                raise ValueError("state contains non-finite values")
            initial_success = bool(evaluate_success(task, start_state, goal_state))
            distance = float(physical_distance(task, start_state, goal_state))
        except (TypeError, ValueError, IndexError):
            diagnostics["invalid_state_excluded"] += 1
            continue
        if initial_success:
            diagnostics["initial_success_excluded"] += 1
            continue
        candidates.append(
            _Candidate(
                row_index=row_index,
                episode_id=_jsonable(episode_id),
                start_step=start_step,
                goal_row_index=goal_row_index,
                goal_step=start_step + offset,
                distance=distance,
                initially_successful=initial_success,
                start_state=tuple(float(item) for item in start_state),
                goal_state=tuple(float(item) for item in goal_state),
            )
        )
    return candidates, diagnostics


def fixed_strata_boundaries(distances: Sequence[float]) -> tuple[float, ...]:
    """Freeze five distance strata using deterministic empirical quantiles."""
    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 1 or len(values) < PHASE1_STRATA:
        raise ValueError("at least five finite candidate distances are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("candidate distances must be finite")
    return tuple(float(item) for item in np.quantile(values, np.linspace(0.0, 1.0, 6)))


def _stratum(distance: float, boundaries: Sequence[float]) -> int:
    if len(boundaries) != PHASE1_STRATA + 1:
        raise ValueError("five strata require six boundaries")
    return int(np.searchsorted(np.asarray(boundaries[1:-1]), float(distance), side="right"))


def _select_candidates(
    candidates: Sequence[_Candidate],
    *,
    num_eval: int,
    boundaries: Sequence[float],
    seed: int,
    strata_by_row: Mapping[int, int] | None = None,
) -> tuple[list[CohortEntry], tuple[int, ...], tuple[int, ...]]:
    num_eval = int(num_eval)
    if num_eval < 1 or num_eval % PHASE1_STRATA:
        raise ValueError("Phase 1 cohort size must be a positive multiple of five")
    groups: list[list[_Candidate]] = [[] for _ in range(PHASE1_STRATA)]
    for candidate in candidates:
        group_id = strata_by_row.get(candidate.row_index) if strata_by_row is not None else None
        groups[_stratum(candidate.distance, boundaries) if group_id is None else int(group_id)].append(candidate)
    target = num_eval // PHASE1_STRATA
    counts = tuple(len(group) for group in groups)
    if any(count < target for count in counts):
        raise ValueError(
            f"not enough candidates in a fixed distance stratum: counts={counts}, target={target}"
        )
    selected: list[CohortEntry] = []
    for stratum, group in enumerate(groups):
        group = sorted(group, key=lambda item: (item.distance, _episode_sort_key(item.episode_id), item.start_step, item.row_index))
        rng = np.random.default_rng(int(seed) + 1009 * (stratum + 1))
        positions = np.sort(rng.choice(len(group), size=target, replace=False))
        for position in positions:
            candidate = group[int(position)]
            selected.append(
                CohortEntry(
                    row_index=candidate.row_index,
                    episode_id=candidate.episode_id,
                    start_step=candidate.start_step,
                    goal_row_index=candidate.goal_row_index,
                    goal_step=candidate.goal_step,
                    start_distance=candidate.distance,
                    initially_successful=candidate.initially_successful,
                    stratum=stratum,
                    start_state=candidate.start_state,
                    goal_state=candidate.goal_state,
                )
            )
    selected.sort(key=lambda item: item.row_index)
    return selected, counts, tuple(target for _ in range(PHASE1_STRATA))


def _make_revised_manifest(
    *,
    task: str,
    cohort_kind: str,
    entries: Sequence[CohortEntry],
    split: Mapping[str, tuple[Any, ...]],
    seed: int,
    goal_offset_steps: int,
    boundaries: Sequence[float],
    candidate_counts: Sequence[int],
    selected_counts: Sequence[int],
    diagnostics: Mapping[str, Any],
) -> CohortManifest:
    return CohortManifest(
        task=task,
        cohort_id=f"{task}_{cohort_kind}_round3_v1",
        cohort_kind=cohort_kind,
        protocol_variant="round3_revised",
        seed=int(seed),
        goal_offset_steps=int(goal_offset_steps),
        entries=tuple(entries),
        episode_split=split,
        strata_boundaries=tuple(float(item) for item in boundaries),
        candidate_counts=tuple(int(item) for item in candidate_counts),
        selected_counts=tuple(int(item) for item in selected_counts),
        sampling_rule={
            "original_episode_id_split_first": True,
            "max_one_start_per_raw_episode": True,
            "eligibility": "start_step + goal_offset_steps < episode_length",
            "initial_success_excluded": True,
            "strata": PHASE1_STRATA,
            "strata_method": "empirical_quantile_boundaries",
            "strata_tie_break": "stable distance, episode_id, start_step, row_index rank",
            "model_dependent_selection": False,
            "rng": "numpy.default_rng; episode-local deterministic seed",
        },
        diagnostics=dict(diagnostics),
    )




def _legacy_cohort_rows(
    dataset: Any, *, goal_offset_steps: int, num_eval: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Exact copy of the historic selection arithmetic, including its quirks."""
    episode_column = _episode_column(dataset)
    episodes = _column(dataset, episode_column)
    steps = _column(dataset, "step_idx")
    unique_episodes, inverse = np.unique(episodes, return_inverse=True)
    try:
        step_values = np.asarray(steps, dtype=np.int64)
        max_steps = np.full(len(unique_episodes), np.iinfo(np.int64).min, dtype=np.int64)
        np.maximum.at(max_steps, inverse, step_values)
        per_row_limit = max_steps[inverse] - int(goal_offset_steps)
        valid_indices = np.flatnonzero(step_values <= per_row_limit).astype(np.int64)
    except (TypeError, ValueError):
        limits: dict[str, int] = {}
        for episode in unique_episodes:
            values = steps[np.asarray([_same_value(item, episode) for item in episodes])]
            limits[_episode_key(episode)] = int(np.max(values)) - int(goal_offset_steps)
        valid_indices = np.asarray(
            [
                index
                for index, (episode, step) in enumerate(zip(episodes, steps))
                if int(step) <= limits[_episode_key(episode)]
            ],
            dtype=np.int64,
        )
    candidate_count = len(valid_indices) - 1
    if candidate_count < int(num_eval):
        raise ValueError(
            f"legacy cohort needs {num_eval} starts, only {max(candidate_count, 0)} candidates available"
        )
    positions = np.random.default_rng(int(seed)).choice(
        candidate_count, size=int(num_eval), replace=False
    )
    rows = np.sort(valid_indices[positions])
    selected_episodes = episodes[rows]
    duplicate_keys = [key for key in {_episode_key(item) for item in selected_episodes} if sum(_episode_key(value) == key for value in selected_episodes) > 1]
    return rows, selected_episodes, {
        "candidate_count": int(candidate_count),
        "valid_row_count": int(len(valid_indices)),
        "global_last_row_excluded": int(valid_indices[-1]) if len(valid_indices) else None,
        "duplicate_episode_ids": [json.loads(key) for key in sorted(duplicate_keys)],
        "duplicate_episode_count": len(duplicate_keys),
        "selected_initial_success_not_filtered": True,
    }


def _entry_for_row(
    dataset: Any,
    *,
    task: str,
    row_index: int,
    goal_offset_steps: int,
    episode_column: str,
    episodes: np.ndarray,
    steps: np.ndarray,
    lookup: Mapping[tuple[str, int], int],
    state_values: np.ndarray,
) -> CohortEntry:
    episode_id = _jsonable(episodes[row_index])
    start_step = int(steps[row_index])
    goal_row = lookup.get((_episode_key(episode_id), start_step + int(goal_offset_steps)))
    start_distance = None
    initial_success = None
    start_state = None
    goal_state = None
    if goal_row is not None:
        try:
            start = _row_value(state_values, row_index)
            goal = _row_value(state_values, goal_row)
            if np.all(np.isfinite(start)) and np.all(np.isfinite(goal)):
                start_distance = float(physical_distance(task, start, goal))
                initial_success = bool(evaluate_success(task, start, goal))
                start_state = tuple(float(item) for item in start)
                goal_state = tuple(float(item) for item in goal)
        except (TypeError, ValueError, IndexError):
            pass
    return CohortEntry(
        row_index=int(row_index),
        episode_id=episode_id,
        start_step=start_step,
        goal_row_index=int(goal_row) if goal_row is not None else None,
        goal_step=start_step + int(goal_offset_steps) if goal_row is not None else None,
        start_distance=start_distance,
        initially_successful=initial_success,
        start_state=start_state,
        goal_state=goal_state,
    )


def build_legacy_manifest(
    dataset: Any,
    *,
    task: str,
    seed: int = int(ROUND3_EVAL_DEFAULTS["seed"]),
    goal_offset_steps: int = int(ROUND3_EVAL_DEFAULTS["goal_offset_steps"]),
    num_eval: int = PHASE1_DEV_EPISODES,
) -> CohortManifest:
    """Capture the historic cohort and explicitly record its known quirks."""
    rows, selected_episodes, diagnostics = _legacy_cohort_rows(
        dataset,
        goal_offset_steps=int(goal_offset_steps),
        num_eval=int(num_eval),
        seed=int(seed),
    )
    episode_column = _episode_column(dataset)
    episodes = _column(dataset, episode_column)
    steps = _column(dataset, "step_idx")
    state_column, _ = _state_columns(dataset, task)
    state_values = _column(dataset, state_column)
    lookup = _row_lookup(episodes, steps)
    entries = tuple(
        _entry_for_row(
            dataset,
            task=task,
            row_index=int(row),
            goal_offset_steps=int(goal_offset_steps),
            episode_column=episode_column,
            episodes=episodes,
            steps=steps,
            lookup=lookup,
            state_values=state_values,
        )
        for row in rows
    )
    initial_success_rows = [
        entry.row_index for entry in entries if entry.initially_successful is True
    ]
    diagnostics = {
        **diagnostics,
        "selected_initial_success_count": len(initial_success_rows),
        "selected_initial_success_rows": initial_success_rows,
    }
    return CohortManifest(
        task=task,
        cohort_id=f"{task}_legacy_{int(num_eval)}_v1",
        cohort_kind="dev" if int(num_eval) == PHASE1_DEV_EPISODES else "custom",
        protocol_variant="legacy",
        seed=int(seed),
        goal_offset_steps=int(goal_offset_steps),
        entries=entries,
        episode_split={"selected": tuple(_jsonable(item) for item in selected_episodes)},
        sampling_rule={
            "implementation": "source.common.eval.select_eval_cohort compatibility arithmetic",
            "global_last_row_excluded": True,
            "episode_level_deduplication": False,
            "initial_success_exclusion": False,
        },
        diagnostics=diagnostics,
    )


def make_goal_refresh_cases(task: str, *, count: int = 10, seed: int = 42) -> tuple[dict[str, Any], ...]:
    """Create deterministic old/new targets that exercise goal replacement."""
    key = task.lower().replace("push-t", "pusht").replace("two_room", "tworoom")
    dimensions = {"cube": 3, "reacher": 2, "pusht": 7, "tworoom": 2}
    scales = {"cube": 0.2, "reacher": 0.5, "pusht": 50.0, "tworoom": 40.0}
    if key not in dimensions:
        raise ValueError(f"unsupported task {task!r}")
    rng = np.random.default_rng(int(seed))
    cases = []
    for index in range(int(count)):
        old_goal = rng.normal(0.0, scales[key], size=dimensions[key])
        new_goal = rng.normal(0.0, scales[key], size=dimensions[key])
        current = np.array(new_goal, copy=True)
        if key == "pusht":
            current[4] = new_goal[4]
        cases.append(
            {
                "case": index,
                "old_goal": old_goal.tolist(),
                "new_goal": new_goal.tolist(),
                "current": current.tolist(),
            }
        )
    return tuple(cases)


def run_goal_refresh_checks(
    task: str,
    cases: Iterable[Mapping[str, Any]],
    *,
    set_goal,
    read_goal,
    read_success,
) -> dict[str, Any]:
    """Check goal field refresh and new-goal success through a runtime adapter.

    The adapter is deliberately tiny: callers can bind it to an environment
    pool, while tests can use an in-memory task double.  Every case first sets
    an old goal and then a new goal, so a stale success snapshot is observable.
    """
    checks = []
    for index, case in enumerate(cases):
        old_goal = np.asarray(case["old_goal"], dtype=np.float64)
        new_goal = np.asarray(case["new_goal"], dtype=np.float64)
        current = np.asarray(case["current"], dtype=np.float64)
        set_goal(old_goal)
        old_observed = np.asarray(read_goal(), dtype=np.float64)
        old_success = bool(np.asarray(read_success()).all())
        set_goal(new_goal)
        new_observed = np.asarray(read_goal(), dtype=np.float64)
        new_success = bool(np.asarray(read_success()).all())
        expected_old = bool(evaluate_success(task, current, old_goal))
        expected_new = bool(evaluate_success(task, current, new_goal))
        checks.append(
            {
                "case": int(case.get("case", index)),
                "goal_field_refreshed": bool(np.array_equal(new_observed, new_goal)),
                "old_goal_readback": bool(np.array_equal(old_observed, old_goal)),
                "old_success_matches_predicate": old_success == expected_old,
                "new_success_matches_predicate": new_success == expected_new,
                "success_uses_new_goal": new_success == expected_new,
                "old_success": old_success,
                "new_success": new_success,
                "expected_old_success": expected_old,
                "expected_new_success": expected_new,
            }
        )
    passed = all(
        item["goal_field_refreshed"]
        and item["old_goal_readback"]
        and item["old_success_matches_predicate"]
        and item["new_success_matches_predicate"]
        for item in checks
    )
    return {
        "status": "accepted" if passed and len(checks) == 10 else "unaccepted",
        "checks_expected": 10,
        "checks_run": len(checks),
        "checks": checks,
        "continuous_hold": False,
        "reacher_qpos_match_preserved": task.lower() == "reacher",
    }


def run_synthetic_goal_refresh_checks(
    task: str, *, count: int = 10, seed: int = 42
) -> dict[str, Any]:
    """Exercise the adapter contract without pretending it is an env audit."""
    state: dict[str, np.ndarray] = {"goal": np.zeros(1), "current": np.zeros(1)}
    cases = make_goal_refresh_cases(task, count=count, seed=seed)
    state["current"] = np.asarray(cases[0]["current"], dtype=np.float64)

    current_by_goal = {
        tuple(float(item) for item in np.asarray(goal).reshape(-1)): np.asarray(case["current"], dtype=np.float64)
        for case in cases
        for goal in (case["old_goal"], case["new_goal"])
    }

    def set_goal(value):
        value_array = np.asarray(value, dtype=np.float64).reshape(-1)
        state["goal"] = np.array(value_array, dtype=np.float64, copy=True)
        state["current"] = current_by_goal.get(tuple(float(item) for item in value_array), state["current"])
    def read_goal():
        return state["goal"]

    def read_success():
        return evaluate_success(task, state["current"], state["goal"])

    # Each case must update the current state too; this mirrors a reset/goal
    # setter pair and avoids a false pass caused by reusing the first state.
    wrapped_cases = []
    for case in cases:
        state["current"] = np.asarray(case["current"], dtype=np.float64)
        wrapped_cases.append(case)
    result = run_goal_refresh_checks(
        task,
        wrapped_cases,
        set_goal=set_goal,
        read_goal=read_goal,
        read_success=read_success,
    )
    result["adapter"] = "synthetic_predicate_adapter"
    return result


def wilson_interval(successes: Iterable[bool] | int, n: int | None = None, *, confidence: float = 0.95) -> tuple[float, float]:
    """Return a two-sided Wilson interval for a Bernoulli proportion."""
    if isinstance(successes, (int, np.integer)):
        if n is None:
            raise ValueError("n is required when successes is a count")
        count = int(successes)
        total = int(n)
    else:
        values = np.asarray(list(successes), dtype=bool)
        count = int(values.sum())
        total = int(values.size)
    if total <= 0 or count < 0 or count > total:
        raise ValueError("success count and sample size are invalid")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = count / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _episode_value(episode: Any, key: str, default: Any = None) -> Any:
    if isinstance(episode, Mapping):
        return episode.get(key, default)
    return getattr(episode, key, default)


def _finite_quantiles(values: Iterable[Any]) -> dict[str, float | None]:
    numeric = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if len(numeric) == 0:
        return {name: None for name in ("p10", "p25", "median", "p75", "p90")}
    quantiles = np.quantile(numeric, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        name: float(value)
        for name, value in zip(("p10", "p25", "median", "p75", "p90"), quantiles)
    }


def summarize_episodes(episodes: Sequence[Any]) -> dict[str, Any]:
    """Compute the required descriptive metrics from episode-level records."""
    if not episodes:
        raise ValueError("at least one episode is required")
    successes = np.asarray([bool(_episode_value(item, "success", False)) for item in episodes], dtype=bool)
    initial = [
        bool(_episode_value(item, "initial_success", False))
        for item in episodes
        if _episode_value(item, "initial_success", None) is not None
    ]
    hold = [
        bool(_episode_value(item, "hold_success", False))
        for item in episodes
        if _episode_value(item, "hold_success", None) is not None
    ]
    start_distances = [_episode_value(item, "start_distance") for item in episodes]
    terminal_distances = [_episode_value(item, "terminal_distance") for item in episodes]
    first_success = [_episode_value(item, "first_success_step") for item in episodes]
    finite_first = [value for value in first_success if value is not None and np.isfinite(value)]
    low, high = wilson_interval(successes)
    result = {
        "num_episodes": int(len(episodes)),
        "successes": int(successes.sum()),
        "success_vector": successes.astype(bool).tolist(),
        "success_rate": float(successes.mean()),
        "success_rate_percent": float(successes.mean() * 100.0),
        "wilson_95": {"low": float(low), "high": float(high)},
        "wilson_95_percent": {"low": float(low * 100.0), "high": float(high * 100.0)},
        "initial_success_rate": float(np.mean(initial)) if initial else None,
        "initial_success_count": int(sum(initial)) if initial else 0,
        "distance_quantiles": _finite_quantiles(start_distances),
        "terminal_distance_quantiles": _finite_quantiles(terminal_distances),
        "hold_success_rate": float(np.mean(hold)) if hold else None,
        "hold_success_count": int(sum(hold)) if hold else 0,
        "first_success_step_mean": float(np.mean(finite_first)) if finite_first else None,
        "first_success_step_median": float(np.median(finite_first)) if finite_first else None,
        "rollout_failures": int(sum(bool(_episode_value(item, "rollout_failed", False)) for item in episodes)),
        "missing_fields": int(sum(int(_episode_value(item, "missing_field_count", 0) or 0) for item in episodes)),
        "invalid_actions": int(sum(int(_episode_value(item, "invalid_action_count", 0) or 0) for item in episodes)),
        "early_terminations": int(sum(bool(_episode_value(item, "early_terminated", False)) for item in episodes)),
    }
    return result


def _paired_key(episode: Any) -> tuple[str, int]:
    return (
        _episode_key(_episode_value(episode, "episode_id", _episode_value(episode, "dataset_episode"))),
        int(_episode_value(episode, "start_step", 0)),
    )


def _exact_mcnemar_p(discordant_left: int, discordant_right: int) -> float | None:
    total = int(discordant_left) + int(discordant_right)
    if total == 0:
        return None
    smaller = min(int(discordant_left), int(discordant_right))
    tail = sum(math.comb(total, index) for index in range(smaller + 1)) / (2.0 ** total)
    return float(min(1.0, 2.0 * tail))


def paired_comparison(baseline: Sequence[Any], candidate: Sequence[Any]) -> dict[str, Any]:
    """Compare matched starts without pooling tasks or changing episode IDs."""
    base = {_paired_key(item): bool(_episode_value(item, "success", False)) for item in baseline}
    cand = {_paired_key(item): bool(_episode_value(item, "success", False)) for item in candidate}
    if len(base) != len(baseline) or len(cand) != len(candidate):
        raise ValueError("paired episodes must have unique episode_id/start_step keys")
    if set(base) != set(cand):
        raise ValueError("baseline and candidate do not contain the same starts")
    improved = sum(not base[key] and cand[key] for key in base)
    regressed = sum(base[key] and not cand[key] for key in base)
    stable_success = sum(base[key] and cand[key] for key in base)
    stable_failure = sum(not base[key] and not cand[key] for key in base)
    total = len(base)
    return {
        "n": int(total),
        "improved": int(improved),
        "regressed": int(regressed),
        "stable_success": int(stable_success),
        "stable_failure": int(stable_failure),
        "delta_pp": float((improved - regressed) / total * 100.0) if total else None,
        "mcnemar_exact_two_sided_p": _exact_mcnemar_p(improved, regressed),
    }


def protocol_sensitivity(
    legacy: Sequence[Any],
    sampling_revised: Sequence[Any],
    tolerance_revised: Sequence[Any],
    round3_revised: Sequence[Any],
) -> dict[str, Any]:
    def compare(baseline: Sequence[Any], candidate: Sequence[Any]) -> dict[str, Any]:
        """Compare protocol outcomes, pairing only when starts are identical."""
        baseline_values = [bool(_episode_value(item, "success", False)) for item in baseline]
        candidate_values = [bool(_episode_value(item, "success", False)) for item in candidate]
        baseline_rate = float(np.mean(baseline_values)) if baseline_values else None
        candidate_rate = float(np.mean(candidate_values)) if candidate_values else None
        baseline_keys = {_paired_key(item) for item in baseline}
        candidate_keys = {_paired_key(item) for item in candidate}
        result = {
            "comparison_type": "unpaired_descriptive",
            "baseline_n": int(len(baseline)),
            "candidate_n": int(len(candidate)),
            "baseline_successes": int(sum(baseline_values)),
            "candidate_successes": int(sum(candidate_values)),
            "baseline_success_rate": baseline_rate,
            "candidate_success_rate": candidate_rate,
            "delta_pp": (
                float((candidate_rate - baseline_rate) * 100.0)
                if baseline_rate is not None and candidate_rate is not None
                else None
            ),
            "paired": None,
        }
        if baseline_keys == candidate_keys and len(baseline_keys) == len(baseline):
            result["comparison_type"] = "paired"
            result["paired"] = paired_comparison(baseline, candidate)
        return result

    return {
        "legacy_to_sampling_revised": compare(legacy, sampling_revised),
        "legacy_to_tolerance_revised": compare(legacy, tolerance_revised),
        "legacy_to_round3_revised": compare(legacy, round3_revised),
    }


def _info_value(infos: Any, names: Sequence[str], slot: int) -> Any:
    """Read one slot from EnvPool dicts, list-of-dicts, or scalar info maps."""
    if isinstance(infos, Mapping):
        for name in names:
            if name in infos:
                value = infos[name]
                try:
                    array = np.asarray(value)
                    if array.ndim > 0 and array.shape[0] > slot:
                        value = array[slot]
                except (TypeError, ValueError):
                    pass
                value = np.asarray(value)
                while value.ndim > 1 and value.shape[0] == 1:
                    value = value[0]
                return value
        return None
    if isinstance(infos, (list, tuple)) and len(infos) > slot:
        item = infos[slot]
        if isinstance(item, Mapping):
            return _info_value(item, names, 0)
    return None


def _batch_action(actions: Any, slot: int) -> np.ndarray | None:
    if actions is None:
        return None
    array = np.asarray(actions)
    if array.ndim == 0:
        return array.reshape(1).astype(np.float64)
    if array.ndim >= 2 and array.shape[0] > slot:
        return np.asarray(array[slot], dtype=np.float64).reshape(-1)
    if slot == 0:
        return np.asarray(array, dtype=np.float64).reshape(-1)
    return None


def _boolean_info(value: Any) -> bool | None:
    """Convert an info flag while treating floating NaN as unavailable."""
    if value is None:
        return None
    try:
        array = np.asarray(value)
        if array.size == 0:
            return None
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            return None
        return bool(array.all())
    except (TypeError, ValueError):
        return None



def action_is_legal(action: Any, action_space: Any = None) -> bool | None:
    if action is None:
        return None
    array = np.asarray(action, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return False
    if action_space is None:
        return True
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is None or high is None:
        return True
    try:
        return bool(np.all(array >= np.asarray(low)) and np.all(array <= np.asarray(high)))
    except (TypeError, ValueError):
        return None


class Round3TraceCollector:
    """Collect lightweight per-step traces and episode summaries.

    It accepts the information returned by the installed environment pool, so
    it can be attached by a wrapper without modifying stable-worldmodel.
    """

    def __init__(self, task: str, manifest: CohortManifest, *, action_block: int = 5, action_space: Any = None, neutral_action: Any = None):
        self.task = task
        self.manifest = manifest
        self.action_block = int(action_block)
        self.action_space = action_space
        self.neutral_action = None if neutral_action is None else np.asarray(neutral_action, dtype=np.float64).reshape(-1)
        self._steps: list[list[dict[str, Any]]] = [[] for _ in manifest.entries]

    def record_step(self, actions: Any, infos: Any, *, raw_env_step: int) -> None:
        predicate = next(item for item in TASKS if item == self.task.lower().replace("push-t", "pusht").replace("two_room", "tworoom"))
        from .round3_protocol import TASK_PREDICATES

        definition = TASK_PREDICATES[predicate]
        for slot in range(len(self._steps)):
            action = _batch_action(actions, slot)
            current = _info_value(infos, (definition.current_field, *definition.field_aliases), slot)
            goal = _info_value(infos, (definition.goal_field, *definition.goal_aliases), slot)
            success_info = _info_value(infos, ("success", "is_success"), slot)
            terminated = _info_value(infos, ("terminated", "done"), slot)
            truncated = _info_value(infos, ("truncated",), slot)
            step_info = _info_value(infos, ("step_idx",), slot)
            try:
                step_array = np.asarray(step_info, dtype=np.float64).reshape(-1)
                step_value = int(step_array[0]) if step_array.size and np.isfinite(step_array[0]) else int(raw_env_step)
            except (TypeError, ValueError, IndexError):
                step_value = int(raw_env_step)
            if self._steps[slot] and step_value == self._steps[slot][-1]["raw_env_step"]:
                continue
            current_array = None if current is None else np.asarray(current, dtype=np.float64).reshape(-1)
            goal_array = None if goal is None else np.asarray(goal, dtype=np.float64).reshape(-1)
            distance = None
            predicate_success = None
            if current_array is not None and goal_array is not None:
                try:
                    distance = float(physical_distance(self.task, current_array, goal_array))
                    predicate_success = bool(evaluate_success(self.task, current_array, goal_array))
                except (TypeError, ValueError):
                    distance = None
            env_success = (
                _boolean_info(success_info)
                if success_info is not None
                else predicate_success
            )
            self._steps[slot].append(
                {
                    "raw_env_step": step_value,
                    "current": _jsonable(current_array),
                    "goal": _jsonable(goal_array),
                    "distance": distance,
                    "predicate_success": predicate_success,
                    "env_success": env_success,
                    "action": _jsonable(action),
                    "action_legal": action_is_legal(action, self.action_space),
                    "neutral_action": bool(action is not None and np.allclose(action, 0.0 if self.neutral_action is None else self.neutral_action)),
                    "neutral_action_source": "zero" if self.neutral_action is None else "declared",
                    "terminated": _boolean_info(terminated) or False,
                    "truncated": _boolean_info(truncated) or False,
                }
            )

    def finalize(self, successes: Sequence[bool], *, eval_budget: int) -> list[dict[str, Any]]:
        if len(successes) != len(self._steps):
            raise ValueError("success vector and trace slot count differ")
        records = []
        for slot, (entry, steps, success) in enumerate(zip(self.manifest.entries, self._steps, successes)):
            success_steps = [
                int(item["raw_env_step"])
                for item in steps
                if item.get("predicate_success") is True or item.get("env_success") is True
            ]
            valid_hold_windows = []
            for start in range(max(0, len(steps) - self.action_block + 1)):
                window = steps[start : start + self.action_block]
                if len(window) != self.action_block:
                    continue
                legal = all(item.get("action_legal") is True for item in window)
                neutral = all(item.get("neutral_action") is True for item in window)
                held = all(item.get("predicate_success") is True for item in window)
                if legal:
                    valid_hold_windows.append({"start_step": window[0]["raw_env_step"], "success": bool(neutral and held)})
            current_values = [item["distance"] for item in steps if item.get("distance") is not None]
            terminal_distance = current_values[-1] if current_values else None
            missing_fields = sum(item.get("current") is None or item.get("goal") is None for item in steps)
            invalid_actions = sum(item.get("action_legal") is False for item in steps)
            early = any(
                (item.get("terminated") or item.get("truncated")) and int(item["raw_env_step"]) < int(eval_budget)
                for item in steps
            )
            records.append(
                {
                    "slot": int(slot),
                    "episode_id": _jsonable(entry.episode_id),
                    "dataset_episode": _jsonable(entry.episode_id),
                    "start_step": int(entry.start_step),
                    "row_index": int(entry.row_index),
                    "goal_row_index": entry.goal_row_index,
                    "success": bool(success),
                    "initial_success": entry.initially_successful,
                    "start_distance": entry.start_distance,
                    "terminal_distance": terminal_distance,
                    "first_success_step": min(success_steps) if success_steps else None,
                    "hold_success": bool(any(item["success"] for item in valid_hold_windows)),
                    "hold_windows": valid_hold_windows,
                    "rollout_failed": len(steps) == 0,
                    "missing_field_count": int(missing_fields),
                    "invalid_action_count": int(invalid_actions),
                    "early_terminated": bool(early),
                    "steps_executed": int(len(steps)),
                    "steps": steps,
                }
            )
        return records


class _PolicyTap:
    """Delegate a world policy while retaining the action passed to the env."""

    def __init__(self, policy: Any):
        self.policy = policy
        self.last_action = None

    def set_env(self, env):
        return self.policy.set_env(env)

    def get_action(self, *args, **kwargs):
        self.last_action = self.policy.get_action(*args, **kwargs)
        return self.last_action

    def __call__(self, *args, **kwargs):
        self.last_action = self.policy(*args, **kwargs)
        return self.last_action

    def __getattr__(self, name):
        return getattr(self.policy, name)


def write_trace_jsonl(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(target)
    return target


def _code_commit(repo_root: str | Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root or Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def write_round3_result(
    payload: Mapping[str, Any],
    output_dir: str | Path,
    *,
    trace_records: Sequence[Mapping[str, Any]] | None = None,
    trace_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write the Phase 1 result schema, preserving an optional trace file."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    result = dict(_jsonable(payload))
    result.setdefault("schema_version", 2)
    result.setdefault("protocol", ROUND3_PROTOCOL)
    result.setdefault("status", "ok")
    if trace_records is not None:
        trace_dir = Path(trace_output_dir) if trace_output_dir is not None else target_dir
        trace_path = trace_dir / "episodes.jsonl"
        write_trace_jsonl(trace_records, trace_path)
        result["trace_path"] = str(trace_path)
        result["trace_schema_version"] = TRACE_SCHEMA_VERSION
    temporary = target_dir / "result.json.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target_dir / "result.json")
    summary = result.get("summary", {})
    lines = [
        f"protocol: {result.get('protocol')}",
        f"protocol_variant: {result.get('protocol_variant')}",
        f"status: {result.get('status')}",
        f"success_rate: {summary.get('success_rate', result.get('success_rate'))}",
        f"cohort_sha256: {result.get('cohort_sha256')}",
    ]
    (target_dir / "metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def enrich_result_payload(
    base_result: Any,
    *,
    manifest: CohortManifest,
    protocol_variant: str,
    predicate_version: str = PREDICATE_VERSION,
    trace_records: Sequence[Mapping[str, Any]] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Attach frozen protocol identity to a legacy ``EvaluationResult``."""
    if hasattr(base_result, "__dataclass_fields__"):
        from dataclasses import asdict

        payload = asdict(base_result)
    elif isinstance(base_result, Mapping):
        payload = dict(base_result)
    else:
        raise TypeError("base_result must be a mapping or dataclass result")
    payload.update(
        {
            "protocol": ROUND3_PROTOCOL,
            "protocol_variant": str(protocol_variant),
            "cohort_id": manifest.cohort_id,
            "cohort_sha256": manifest.computed_sha256,
            "predicate_version": predicate_version,
            "code_commit": _code_commit(repo_root),
            "cohort_schema_version": manifest.schema_version,
            "trace_schema_version": TRACE_SCHEMA_VERSION if trace_records is not None else None,
            "cohort": manifest.as_dict(),
        }
    )
    if trace_records is not None:
        payload["episodes"] = list(trace_records)
        payload["summary"] = summarize_episodes(trace_records)
        payload["success_rate"] = payload["summary"]["success_rate"]
    return _jsonable(payload)


def _dependency_versions() -> dict[str, str | None]:
    names = ("stable-worldmodel", "stable-pretraining", "torch", "numpy", "omegaconf")
    result = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _find_epoch(path: Path) -> int | None:
    match = _WEIGHT_EPOCH_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def _read_config_values(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "missing train config"
    try:
        from omegaconf import OmegaConf

        config = OmegaConf.load(path)
        try:
            value = OmegaConf.to_container(config, resolve=True)
        except Exception:
            value = OmegaConf.to_container(config, resolve=False)
        return value if isinstance(value, dict) else {"value": value}, None
    except Exception as exc:  # noqa: BLE001 - registry records the failure
        return None, f"could not parse train config: {type(exc).__name__}: {exc}"


def _nested_get(value: Any, path: Sequence[str]) -> Any:
    current = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _historical_result_status(path: Path | None) -> tuple[str, str | None]:
    if path is None:
        return "missing", "historical result path is not registered"
    if not path.is_file():
        return "missing", f"historical result not found: {path}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "invalid", f"historical result is not JSON: {exc}"
    status = str(value.get("status", "ok"))
    if status != "ok":
        return "invalid", f"historical result status={status}"
    return "ok", None


def build_artifact_registry(
    spec_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve the fixed matrix without substituting missing checkpoints."""
    root = Path(repo_root or Path(spec_path).resolve().parents[2]).resolve()
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    entries = []
    for group in spec.get("entries", []):
        stages = group.get("stages", [None])
        for stage in stages:
            checkpoint_value = group.get("checkpoint")
            checkpoint = None if checkpoint_value in (None, "") else (root / checkpoint_value).resolve() if not Path(checkpoint_value).is_absolute() else Path(checkpoint_value).resolve()
            config_value = group.get("train_config")
            train_config = None if not config_value else (root / config_value).resolve() if not Path(config_value).is_absolute() else Path(config_value).resolve()
            historical_value = group.get("historical_result")
            historical = None
            if historical_value:
                rendered = str(historical_value).replace("{stage}", str(stage)).replace("{epoch}", str(group.get("final_epoch", "")))
                historical = (root / rendered).resolve() if not Path(rendered).is_absolute() else Path(rendered).resolve()
            reasons = []
            checkpoint_hash = None
            if checkpoint is None or not checkpoint.is_file():
                reasons.append("checkpoint missing")
            else:
                checkpoint_hash = sha256_file(checkpoint)
            config_values, config_error = _read_config_values(train_config) if train_config else (None, "train config is not registered")
            if config_error:
                reasons.append(config_error)
            config_expectations = dict(group.get("config_expectations", {}))
            config_match = config_error is None
            observed = {}
            for name, expected in config_expectations.items():
                path = tuple(str(part) for part in name.split("."))
                actual = _nested_get(config_values, path) if config_values is not None else None
                observed[name] = actual
                if actual != expected:
                    config_match = False
                    reasons.append(f"config mismatch {name}: expected {expected!r}, got {actual!r}")
            if not config_match and "config mismatch" not in " ".join(reasons) and config_error is None:
                reasons.append("train config did not match registered expectations")
            historical_status, historical_reason = _historical_result_status(historical)
            if historical_reason:
                reasons.append(historical_reason)
            if reasons:
                status = "invalid" if historical_status == "invalid" or any("mismatch" in reason or "invalid" in reason or "could not parse" in reason for reason in reasons) else "missing"
            else:
                status = "ok"
            final_epoch = group.get("final_epoch")
            if final_epoch is None and checkpoint is not None:
                final_epoch = _find_epoch(checkpoint)
            entries.append(
                {
                    "task": str(group["task"]),
                    "method": str(group["method"]),
                    "stage": stage,
                    "checkpoint": str(checkpoint) if checkpoint is not None else None,
                    "checkpoint_sha256": checkpoint_hash,
                    "final_epoch": int(final_epoch) if final_epoch is not None else None,
                    "train_config": str(train_config) if train_config is not None else None,
                    "config_expectations": config_expectations,
                    "config_observed": observed,
                    "config_match": bool(config_match),
                    "local_dataset": group.get("local_dataset"),
                    "benchmark_dataset": group.get("benchmark_dataset"),
                    "entrypoint": group.get("entrypoint"),
                    "code_commit": group.get("code_commit") or _code_commit(root),
                    "dependencies": _dependency_versions(),
                    "historical_result": str(historical) if historical is not None else None,
                    "historical_result_status": historical_status,
                    "data_exposure": group.get("data_exposure", {"eval_data_seen_during_training": "unknown"}),
                    "status": status,
                    "participates": status == "ok",
                    "reasons": reasons,
                }
            )
    return {
        "schema_version": ARTIFACT_REGISTRY_SCHEMA_VERSION,
        "protocol": ROUND3_PROTOCOL,
        "matrix": "phase1_fixed",
        "fixed_eval": dict(ROUND3_EVAL_DEFAULTS),
        "code_commit": _code_commit(root),
        "entries": entries,
    }


def write_artifact_registry(
    spec_path: str | Path, output_path: str | Path, *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    payload = build_artifact_registry(spec_path, repo_root=repo_root)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return payload


def load_result(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"result at {path} is not a JSON object")
    return value


PHASE1_MATRIX: dict[str, dict[str, tuple[str, ...]]] = {
    "cube": {"e0_lewm": ("stage_b",), "e3_fast": FAST_STAGES, "e5_fast": FAST_STAGES},
    "pusht": {"e0_lewm": ("stage_b",), "e3_fast": FAST_STAGES, "e5_fast": FAST_STAGES},
    "reacher": {"e0_lewm": ("stage_b",), "e3_fast": FAST_STAGES, "e5_fast": FAST_STAGES},
    "tworoom": {"e0_lewm": ("stage_b",), "e3_fast": FAST_STAGES, "e5_fast": FAST_STAGES},
}


def iter_phase1_matrix() -> Iterable[tuple[str, str, str]]:
    for task in TASKS:
        for method, stages in PHASE1_MATRIX[task].items():
            for stage in stages:
                yield task, method, stage


def write_phase1_matrix(results_root: str | Path, output_csv: str | Path) -> list[dict[str, Any]]:
    root = Path(results_root)
    rows = []
    for task, method, stage in iter_phase1_matrix():
        for variant in PROTOCOL_VARIANTS:
            path = root / task / method / variant / stage / "result.json"
            if not path.is_file():
                value = {"status": "missing"}
            else:
                try:
                    value = load_result(path)
                except (OSError, json.JSONDecodeError, ValueError):
                    value = {"status": "invalid"}
            status = str(value.get("status", "ok"))
            if status not in {"ok", "missing", "invalid"}:
                status = "invalid"
            summary = value.get("summary", {})
            if not isinstance(summary, Mapping):
                summary = {}
            rows.append(
                {
                    "task": task,
                    "method": method,
                    "stage": stage,
                    "protocol_variant": variant,
                    "status": status,
                    "num_episodes": summary.get("num_episodes", len(value.get("episodes", []))) if status == "ok" else None,
                    "success_rate": summary.get("success_rate", value.get("success_rate")) if status == "ok" else None,
                    "wilson_low": summary.get("wilson_95", {}).get("low") if status == "ok" else None,
                    "wilson_high": summary.get("wilson_95", {}).get("high") if status == "ok" else None,
                    "cohort_sha256": value.get("cohort_sha256"),
                    "predicate_version": value.get("predicate_version"),
                }
            )
    target = Path(output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task", "method", "stage", "protocol_variant", "status", "num_episodes",
        "success_rate", "wilson_low", "wilson_high", "cohort_sha256", "predicate_version",
    ]
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)
    return rows


def write_phase1_report(
    rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    registry: Mapping[str, Any] | None = None,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Round 3 Phase 1 report",
        "",
        "This report is descriptive. Stage B is the primary metric; Stage A and A-shuffled are auxiliary.",
        "Final testing uses only the frozen `round3_revised` cohort and is not used to tune predicates or sampling.",
        "",
        "| Task | Method | Stage | Protocol | n | Success | Wilson 95% | Status |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        low = row.get("wilson_low")
        high = row.get("wilson_high")
        interval = "—" if low is None or high is None else f"{100 * low:.1f}%–{100 * high:.1f}%"
        rate = row.get("success_rate")
        rate_text = "—" if rate is None else f"{100 * float(rate):.1f}%"
        lines.append(
            f"| {row.get('task')} | {row.get('method')} | {row.get('stage')} | "
            f"{row.get('protocol_variant')} | {row.get('num_episodes')} | {rate_text} | {interval} | {row.get('status')} |"
        )
    if registry is not None:
        entries = registry.get("entries", [])
        missing = sum(item.get("status") == "missing" for item in entries)
        invalid = sum(item.get("status") == "invalid" for item in entries)
        lines.extend(["", f"Registered artifacts: {len(entries)}; missing: {missing}; invalid: {invalid}."])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


__all__ = [
    "CohortEntry",
    "CohortManifest",
    "PHASE1_DEV_EPISODES",
    "PHASE1_FINAL_EPISODES",
    "PHASE1_MATRIX",
    "Round3TraceCollector",
    "action_is_legal",
    "build_artifact_registry",
    "build_legacy_manifest",
    "build_revised_cohorts",
    "enrich_result_payload",
    "fixed_strata_boundaries",
    "iter_phase1_matrix",
    "load_result",
    "make_goal_refresh_cases",
    "paired_comparison",
    "protocol_sensitivity",
    "run_goal_refresh_checks",
    "run_synthetic_goal_refresh_checks",
    "sha256_file",
    "sha256_json",
    "split_episode_ids",
    "summarize_episodes",
    "wilson_interval",
    "write_artifact_registry",
    "write_phase1_matrix",
    "write_phase1_report",
    "write_round3_result",
    "write_trace_jsonl",
]
# The stratified allocator is kept in a lazy module to avoid an import cycle.
def build_revised_cohorts(
    dataset: Any,
    *,
    task: str,
    seed: int = int(ROUND3_EVAL_DEFAULTS["seed"]),
    goal_offset_steps: int = int(ROUND3_EVAL_DEFAULTS["goal_offset_steps"]),
    dev_count: int = PHASE1_DEV_EPISODES,
    final_count: int = PHASE1_FINAL_EPISODES,
    online_count: int | None = None,
) -> dict[str, CohortManifest]:
    from .round3_sampling import build_revised_cohorts as implementation

    return implementation(
        dataset,
        task=task,
        seed=seed,
        goal_offset_steps=goal_offset_steps,
        dev_count=dev_count,
        final_count=final_count,
        online_count=online_count,
    )
