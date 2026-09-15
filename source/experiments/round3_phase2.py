"""CPU-testable replay and Stage-B adaptation primitives for Round 3 Phase 2.

The module deliberately stops at the transition-update seam. Environment
collection, cohort evaluation, and process orchestration belong to the Phase 2
runner and can use these primitives without changing the E0 model semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _update_digest(digest: "hashlib._Hash", value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
        return
    digest.update(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


@dataclass(frozen=True)
class TransitionReplay:
    """Episode-local observation/action windows from executed transitions.

    ``observations`` contains ``history_size + 1`` frames and ``actions``
    contains ``history_size`` actions. Every row must come from one episode;
    the final observation is therefore the real successor target for the last
    action in the window. The raw tensors are retained so the frozen encoder
    can regenerate latent inputs after a checkpoint reload.
    """

    observations: torch.Tensor
    actions: torch.Tensor
    episode_ids: tuple[tuple[Any, ...], ...]
    start_steps: torch.Tensor
    sources: tuple[str, ...]
    model_versions: tuple[str, ...]
    goals: torch.Tensor | None = None
    raw_observations: torch.Tensor | None = None
    raw_actions: torch.Tensor | None = None
    data_kinds: tuple[str, ...] | None = None
    group_ids: tuple[Any, ...] | None = None
    snapshot_ids: tuple[Any, ...] | None = None
    collector_versions: tuple[str, ...] | None = None
    successes: torch.Tensor | None = None
    physical_start_costs: torch.Tensor | None = None
    physical_terminal_costs: torch.Tensor | None = None
    execution_lengths: torch.Tensor | None = None
    goal_latents: torch.Tensor | None = None

    def __post_init__(self) -> None:
        observations = self.observations
        actions = self.actions
        if not isinstance(observations, torch.Tensor) or not isinstance(actions, torch.Tensor):
            raise TypeError("observations and actions must be torch tensors")
        if observations.ndim < 3 or actions.ndim < 3:
            raise ValueError("replay tensors must include batch, time, and feature dimensions")
        count = int(observations.shape[0])
        if int(actions.shape[0]) != count:
            raise ValueError("observations and actions have different batch sizes")
        if int(observations.shape[1]) != int(actions.shape[1]) + 1:
            raise ValueError("a replay row must contain one successor observation")
        if count < 1:
            raise ValueError("replay cannot be empty")
        for name, tensor in (("observations", observations), ("actions", actions)):
            if not torch.is_floating_point(tensor):
                raise TypeError(f"replay {name} must be floating point")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"replay {name} contains non-finite values")
        if self.goals is not None:
            if not isinstance(self.goals, torch.Tensor) or int(self.goals.shape[0]) != count:
                raise ValueError("goals must have one row per replay transition")
            if not torch.isfinite(self.goals).all():
                raise ValueError("replay goals contain non-finite values")
        for name, value in (
            ("successes", self.successes),
            ("physical_start_costs", self.physical_start_costs),
            ("physical_terminal_costs", self.physical_terminal_costs),
            ("execution_lengths", self.execution_lengths),
        ):
            if value is None:
                continue
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != (count,):
                raise ValueError(f"{name} must have shape [{count}]")
            if not torch.isfinite(value.float()).all():
                raise ValueError(f"{name} contains non-finite values")
        if self.goal_latents is not None:
            if not isinstance(self.goal_latents, torch.Tensor) or int(self.goal_latents.shape[0]) != count:
                raise ValueError("goal_latents must have one row per replay transition")
            if self.goal_latents.ndim != 2 or not torch.isfinite(self.goal_latents).all():
                raise ValueError("goal_latents must be a finite [N,D] tensor")
        if self.raw_observations is not None:
            if tuple(self.raw_observations.shape) != tuple(observations.shape):
                raise ValueError("raw_observations must match observations shape")
            if not torch.isfinite(self.raw_observations.float()).all():
                raise ValueError("raw_observations contain non-finite values")
        if self.raw_actions is not None:
            if tuple(self.raw_actions.shape) != tuple(actions.shape):
                raise ValueError("raw_actions must match actions shape")
            if not torch.isfinite(self.raw_actions.float()).all():
                raise ValueError("raw_actions contain non-finite values")
        if len(self.episode_ids) != count:
            raise ValueError("episode_ids must have one entry per replay transition")
        for window in self.episode_ids:
            if len(window) != int(observations.shape[1]):
                raise ValueError("each episode window must cover every observation")
            if len({_jsonable(value).__repr__() for value in window}) != 1:
                raise ValueError("a replay window crosses an episode boundary")
        if len(self.start_steps) != count:
            raise ValueError("start_steps must have one entry per replay transition")
        if len(self.sources) != count or len(self.model_versions) != count:
            raise ValueError("replay provenance length differs from transition count")
        kinds = self.resolved_data_kinds
        if len(kinds) != count:
            raise ValueError("data_kinds must have one entry per transition")
        if any(kind not in {"offline", "continuous", "grounded"} for kind in kinds):
            raise ValueError("data_kinds must be offline, continuous, or grounded")
        for source, kind in zip(self.sources, kinds):
            if source == "offline" and kind != "offline":
                raise ValueError("offline replay rows must have data_kind='offline'")
            if source == "online" and kind == "offline":
                raise ValueError("online replay rows cannot have data_kind='offline'")
        for name, values in (
            ("group_ids", self.group_ids),
            ("snapshot_ids", self.snapshot_ids),
            ("collector_versions", self.collector_versions),
        ):
            if values is not None and len(values) != count:
                raise ValueError(f"{name} must have one entry per transition")
        if not torch.is_floating_point(self.start_steps):
            starts = self.start_steps
        else:
            starts = self.start_steps
        if starts.ndim != 1 or not torch.isfinite(starts.float()).all():
            raise ValueError("start_steps must be a finite one-dimensional tensor")
        if any(source not in {"offline", "online"} for source in self.sources):
            raise ValueError("replay sources must be 'offline' or 'online'")
        if any(not version for version in self.model_versions):
            raise ValueError("model_versions must be non-empty")

    @property
    def count(self) -> int:
        return int(self.observations.shape[0])

    @property
    def history_size(self) -> int:
        return int(self.actions.shape[1])

    @property
    def resolved_data_kinds(self) -> tuple[str, ...]:
        """Return explicit buckets while preserving legacy replay files."""
        if self.data_kinds is not None:
            return tuple(str(value) for value in self.data_kinds)
        return tuple("offline" if source == "offline" else "continuous" for source in self.sources)

    @property
    def resolved_group_ids(self) -> tuple[Any, ...]:
        """Return group IDs, using row-local IDs only for ungrouped legacy rows."""
        if self.group_ids is not None:
            return tuple(self.group_ids)
        return tuple(range(self.count))

    def content_sha256(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.observations,
            self.actions,
            self.goals,
            self.raw_observations,
            self.raw_actions,
            self.episode_ids,
            self.start_steps,
            self.sources,
            self.model_versions,
            self.resolved_data_kinds,
            self.group_ids,
            self.snapshot_ids,
            self.collector_versions,
            self.successes,
            self.physical_start_costs,
            self.physical_terminal_costs,
            self.execution_lengths,
            self.goal_latents,
        ):
            _update_digest(digest, value)
        return digest.hexdigest()

    def legacy_content_sha256(self) -> str:
        """Hash schema-1 fields so historical replay files remain loadable."""
        digest = hashlib.sha256()
        for value in (
            self.observations,
            self.actions,
            self.goals,
            self.raw_observations,
            self.raw_actions,
            self.episode_ids,
            self.start_steps,
            self.sources,
            self.model_versions,
        ):
            _update_digest(digest, value)
        return digest.hexdigest()

    def subset(self, indices: Iterable[int]) -> "TransitionReplay":
        selected = torch.as_tensor(list(indices), dtype=torch.long)
        if selected.numel() == 0:
            raise ValueError("replay subset cannot be empty")
        positions = selected.tolist()
        return TransitionReplay(
            observations=self.observations[selected],
            actions=self.actions[selected],
            episode_ids=tuple(self.episode_ids[index] for index in positions),
            start_steps=self.start_steps[selected],
            sources=tuple(self.sources[index] for index in positions),
            model_versions=tuple(self.model_versions[index] for index in positions),
            goals=None if self.goals is None else self.goals[selected],
            raw_observations=(
                None if self.raw_observations is None else self.raw_observations[selected]
            ),
            raw_actions=None if self.raw_actions is None else self.raw_actions[selected],
            data_kinds=tuple(self.resolved_data_kinds[index] for index in positions),
            group_ids=(
                None if self.group_ids is None else tuple(self.group_ids[index] for index in positions)
            ),
            snapshot_ids=(
                None if self.snapshot_ids is None else tuple(self.snapshot_ids[index] for index in positions)
            ),
            collector_versions=(
                None
                if self.collector_versions is None
                else tuple(self.collector_versions[index] for index in positions)
            ),
            successes=None if self.successes is None else self.successes[selected],
            physical_start_costs=(
                None
                if self.physical_start_costs is None
                else self.physical_start_costs[selected]
            ),
            physical_terminal_costs=(
                None
                if self.physical_terminal_costs is None
                else self.physical_terminal_costs[selected]
            ),
            execution_lengths=(
                None
                if self.execution_lengths is None
                else self.execution_lengths[selected]
            ),
            goal_latents=None if self.goal_latents is None else self.goal_latents[selected],
        )


def concatenate_replays(replays: Iterable[TransitionReplay]) -> TransitionReplay:
    values = list(replays)
    if not values:
        raise ValueError("cannot concatenate an empty replay collection")
    has_goals = [value.goals is not None for value in values]

    def optional_tensor(name: str) -> torch.Tensor | None:
        payloads = [getattr(value, name) for value in values]
        if not all(item is not None for item in payloads):
            return None
        return torch.cat(payloads, dim=0)

    def optional_numeric_tensor(name: str) -> torch.Tensor | None:
        """Preserve row-aligned diagnostics when offline rows lack them.

        The online collectors are required to populate these fields before a
        fixed-replay manifest is accepted.  Zero-fill is only for unrelated
        offline rows so a single mixed replay can still carry the online
        metadata needed by ranking/distillation.
        """
        payloads = [getattr(value, name) for value in values]
        if all(item is None for item in payloads):
            return None
        reference = next(item for item in payloads if item is not None)
        filled = [
            item
            if item is not None
            else torch.zeros(
                (value.count, *reference.shape[1:]),
                dtype=reference.dtype,
                device=reference.device,
            )
            for value, item in zip(values, payloads)
        ]
        return torch.cat(filled, dim=0)

    def optional_sequence(name: str) -> tuple[Any, ...] | None:
        payloads = [getattr(value, name) for value in values]
        if not all(item is not None for item in payloads):
            return None
        return tuple(item for payload in payloads for item in payload)

    return TransitionReplay(
        observations=torch.cat([value.observations for value in values], dim=0),
        actions=torch.cat([value.actions for value in values], dim=0),
        episode_ids=tuple(item for value in values for item in value.episode_ids),
        start_steps=torch.cat([value.start_steps for value in values], dim=0),
        sources=tuple(item for value in values for item in value.sources),
        model_versions=tuple(item for value in values for item in value.model_versions),
        # Goal payloads are optional metadata and may use different native
        # encodings (the environment collector receives goal pixels, while
        # the HDF5 training set exposes state targets). Never fabricate a
        # cross-source tensor merely to make concatenation possible.
        goals=(None if not all(has_goals) else torch.cat([value.goals for value in values], dim=0)),
        raw_observations=(
            None
            if not all(value.raw_observations is not None for value in values)
            else torch.cat([value.raw_observations for value in values], dim=0)
        ),
        raw_actions=(
            None
            if not all(value.raw_actions is not None for value in values)
            else torch.cat([value.raw_actions for value in values], dim=0)
        ),
        data_kinds=tuple(
            item for value in values for item in value.resolved_data_kinds
        ),
        group_ids=optional_sequence("group_ids"),
        snapshot_ids=optional_sequence("snapshot_ids"),
        collector_versions=optional_sequence("collector_versions"),
        successes=optional_numeric_tensor("successes"),
        physical_start_costs=optional_numeric_tensor("physical_start_costs"),
        physical_terminal_costs=optional_numeric_tensor("physical_terminal_costs"),
        execution_lengths=optional_numeric_tensor("execution_lengths"),
        goal_latents=optional_tensor("goal_latents"),
    )


def build_offline_window_manifest(
    dataset: Any,
    *,
    episode_ids: Sequence[Any] | None = None,
    count: int,
    seed: int = 3072,
    frameskip: int = 5,
    num_steps: int = 4,
) -> tuple[dict[str, Any], ...]:
    """Choose episode-local windows using the native LeWM data geometry.

    The E0 loader consumes ``num_steps`` observations sampled every
    ``frameskip`` environment steps and keeps the dense action block for each
    observation.  This helper records only episode-local ``(episode, start)``
    pairs; pixels are loaded lazily by :func:`load_offline_replay` so a full
    Phase 2 run does not materialize the original HDF5 dataset in RAM.
    """
    if int(count) < 1:
        raise ValueError("count must be positive")
    if int(frameskip) < 1 or int(num_steps) < 2:
        raise ValueError("frameskip must be positive and num_steps must be at least two")
    if not hasattr(dataset, "lengths") or not hasattr(dataset, "offsets"):
        raise TypeError("dataset must expose episode lengths and offsets")

    episode_column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    raw_episode_ids = np.asarray(dataset.get_col_data(episode_column))
    unique_episode_ids = np.unique(raw_episode_ids)
    if len(unique_episode_ids) != len(dataset.lengths):
        raise ValueError("dataset episode IDs do not match dataset episode lengths")
    id_to_index = {
        _jsonable(episode_id).__repr__(): index
        for index, episode_id in enumerate(unique_episode_ids.tolist())
    }
    requested = unique_episode_ids.tolist() if episode_ids is None else list(episode_ids)
    if not requested:
        raise ValueError("episode_ids cannot be empty")
    episode_indices: list[tuple[Any, int, int]] = []
    span = int(frameskip) * int(num_steps)
    for episode_id in requested:
        key = _jsonable(episode_id).__repr__()
        if key not in id_to_index:
            raise ValueError(f"episode ID is not present in dataset: {episode_id!r}")
        episode_index = id_to_index[key]
        length = int(dataset.lengths[episode_index])
        max_start = length - span
        if max_start < 0:
            continue
        episode_indices.extend(
            (episode_id, episode_index, start_step)
            for start_step in range(max_start + 1)
        )
    if len(episode_indices) < int(count):
        raise ValueError(
            f"requested {count} offline windows but only {len(episode_indices)} "
            "episode-local windows are available"
        )
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(len(episode_indices), size=int(count), replace=False)
    selected.sort()
    return tuple(
        {
            "episode_id": _jsonable(episode_indices[int(index)][0]),
            "episode_index": int(episode_indices[int(index)][1]),
            "start_step": int(episode_indices[int(index)][2]),
            "frameskip": int(frameskip),
            "num_steps": int(num_steps),
        }
        for index in selected
    )


def load_offline_replay(
    dataset: Any,
    manifest: Sequence[Mapping[str, Any]],
    *,
    process: Mapping[str, Any],
    observation_transform: Any,
    model_version: str,
    history_size: int = 3,
    frameskip: int = 5,
    num_steps: int = 4,
    goal_offset_steps: int | None = None,
) -> TransitionReplay:
    """Load a bounded offline batch with the same layout as executed replay."""
    entries = list(manifest)
    if not entries:
        raise ValueError("offline manifest cannot be empty")
    if int(num_steps) != int(history_size) + 1:
        raise ValueError("num_steps must equal history_size + 1")
    if not hasattr(dataset, "load_chunk"):
        raise TypeError("dataset must expose load_chunk")

    episodes = np.asarray([int(entry["episode_index"]) for entry in entries], dtype=np.int64)
    starts = np.asarray([int(entry["start_step"]) for entry in entries], dtype=np.int64)
    if np.any(starts < 0):
        raise ValueError("offline start steps must be non-negative")
    previous_frameskip = getattr(dataset, "frameskip", None)
    previous_num_steps = getattr(dataset, "num_steps", None)
    dataset.frameskip = int(frameskip)
    dataset.num_steps = int(num_steps)
    try:
        chunks = dataset.load_chunk(
            episodes,
            starts,
            starts + int(frameskip) * int(num_steps),
        )
    finally:
        if previous_frameskip is not None:
            dataset.frameskip = previous_frameskip
        if previous_num_steps is not None:
            dataset.num_steps = previous_num_steps

    processed_observations: list[torch.Tensor] = []
    raw_observations: list[torch.Tensor] = []
    normalized_actions: list[torch.Tensor] = []
    raw_actions: list[torch.Tensor] = []
    episode_windows: list[tuple[Any, ...]] = []
    for entry, chunk in zip(entries, chunks):
        pixels = torch.as_tensor(chunk["pixels"]).detach().cpu()
        if pixels.ndim != 4:
            raise ValueError(f"offline pixels must have shape [T,C,H,W], got {tuple(pixels.shape)}")
        raw_observations.append(pixels.clone())
        processed_observations.append(
            torch.stack([observation_transform(frame) for frame in pixels], dim=0).float()
        )
        actions = torch.as_tensor(chunk["action"]).detach().cpu().float()
        if actions.ndim != 2 or int(actions.shape[0]) != int(num_steps):
            raise ValueError("offline actions do not match native LeWM time dimension")
        raw_actions.append(actions.clone())
        action_width = int(actions.shape[-1])
        if action_width % int(frameskip) != 0:
            raise ValueError("offline action width is not divisible by frameskip")
        flat_actions = actions.reshape(-1, action_width // int(frameskip))
        normalized = torch.as_tensor(
            process["action"].transform(flat_actions.numpy())
        ).float().reshape_as(actions)
        normalized_actions.append(normalized[: int(history_size)])
        raw_actions[-1] = raw_actions[-1][: int(history_size)]
        episode_id = _jsonable(entry["episode_id"])
        episode_windows.append((episode_id,) * int(num_steps))

    goal_pixels: torch.Tensor | None = None
    if goal_offset_steps is not None:
        if int(goal_offset_steps) < 0:
            raise ValueError("goal_offset_steps must be non-negative")
        previous_frameskip = getattr(dataset, "frameskip", None)
        previous_num_steps = getattr(dataset, "num_steps", None)
        dataset.frameskip = 1
        dataset.num_steps = 1
        try:
            goal_chunks = dataset.load_chunk(
                episodes,
                starts + int(goal_offset_steps),
                starts + int(goal_offset_steps) + 1,
            )
        finally:
            if previous_frameskip is not None:
                dataset.frameskip = previous_frameskip
            if previous_num_steps is not None:
                dataset.num_steps = previous_num_steps
        goals: list[torch.Tensor] = []
        for chunk in goal_chunks:
            pixels = torch.as_tensor(chunk["pixels"]).detach().cpu()
            if pixels.ndim != 4 or int(pixels.shape[0]) < 1:
                raise ValueError("offline goal pixels must contain one [C,H,W] frame")
            goals.append(pixels[0].clone())
        if len(goals) != len(entries):
            raise ValueError("offline goal loader returned the wrong number of rows")
        goal_pixels = torch.stack(goals, dim=0)

    return TransitionReplay(
        observations=torch.stack(processed_observations, dim=0),
        actions=torch.stack(normalized_actions, dim=0),
        episode_ids=tuple(episode_windows),
        start_steps=torch.as_tensor(starts, dtype=torch.long),
        sources=("offline",) * len(entries),
        model_versions=(str(model_version),) * len(entries),
        raw_observations=torch.stack(raw_observations, dim=0),
        raw_actions=torch.stack(raw_actions, dim=0),
        goals=goal_pixels,
    )


def save_transition_replay(path: str | Path, replay: TransitionReplay) -> None:
    """Atomically save a replay together with its content hash."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "content_sha256": replay.content_sha256(),
        "observations": replay.observations.cpu(),
        "actions": replay.actions.cpu(),
        "goals": None if replay.goals is None else replay.goals.cpu(),
        "raw_observations": None if replay.raw_observations is None else replay.raw_observations.cpu(),
        "raw_actions": None if replay.raw_actions is None else replay.raw_actions.cpu(),
        "episode_ids": replay.episode_ids,
        "start_steps": replay.start_steps.cpu(),
        "sources": replay.sources,
        "model_versions": replay.model_versions,
        "data_kinds": replay.resolved_data_kinds,
        "group_ids": replay.group_ids,
        "snapshot_ids": replay.snapshot_ids,
        "collector_versions": replay.collector_versions,
        "successes": None if replay.successes is None else replay.successes.cpu(),
        "physical_start_costs": (
            None
            if replay.physical_start_costs is None
            else replay.physical_start_costs.cpu()
        ),
        "physical_terminal_costs": (
            None
            if replay.physical_terminal_costs is None
            else replay.physical_terminal_costs.cpu()
        ),
        "execution_lengths": (
            None
            if replay.execution_lengths is None
            else replay.execution_lengths.cpu()
        ),
        "goal_latents": None if replay.goal_latents is None else replay.goal_latents.cpu(),
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def load_transition_replay(path: str | Path) -> TransitionReplay:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    schema_version = int(payload.get("schema_version", 0))
    if schema_version not in {1, 2}:
        raise ValueError("unsupported transition replay schema")
    replay = TransitionReplay(
        observations=payload["observations"],
        actions=payload["actions"],
        episode_ids=tuple(tuple(item) for item in payload["episode_ids"]),
        start_steps=payload["start_steps"],
        sources=tuple(payload["sources"]),
        model_versions=tuple(payload["model_versions"]),
        goals=payload.get("goals"),
        raw_observations=payload.get("raw_observations"),
        raw_actions=payload.get("raw_actions"),
        data_kinds=payload.get("data_kinds"),
        group_ids=payload.get("group_ids"),
        snapshot_ids=payload.get("snapshot_ids"),
        collector_versions=payload.get("collector_versions"),
        successes=payload.get("successes"),
        physical_start_costs=payload.get("physical_start_costs"),
        physical_terminal_costs=payload.get("physical_terminal_costs"),
        execution_lengths=payload.get("execution_lengths"),
        goal_latents=payload.get("goal_latents"),
    )
    actual_hash = (
        replay.legacy_content_sha256()
        if schema_version == 1
        else replay.content_sha256()
    )
    if actual_hash != payload.get("content_sha256"):
        raise ValueError(f"transition replay hash mismatch: {path}")
    return replay


@dataclass(frozen=True)
class ReplayShard:
    """Metadata for one append-only replay shard on disk."""

    path: str
    count: int
    content_sha256: str


class ReplayShardPool:
    """Sample an append-only replay without concatenating all shards in RAM."""

    def __init__(self, *, seed: int = 3072):
        self.seed = int(seed)
        self._generator = np.random.default_rng(self.seed)
        self._shards: list[ReplayShard] = []

    @property
    def shards(self) -> tuple[ReplayShard, ...]:
        return tuple(self._shards)

    @property
    def count(self) -> int:
        return sum(shard.count for shard in self._shards)

    def add(self, path: str | Path, *, count: int, content_sha256: str) -> ReplayShard:
        if int(count) < 1:
            raise ValueError("replay shard count must be positive")
        shard = ReplayShard(
            path=str(Path(path)),
            count=int(count),
            content_sha256=str(content_sha256),
        )
        self._shards.append(shard)
        return shard

    def sample(self, count: int) -> TransitionReplay:
        """Draw rows uniformly over all appended shards and load only them."""
        if int(count) < 1:
            raise ValueError("sample count must be positive")
        if not self._shards:
            raise ValueError("cannot sample an empty replay pool")
        total = self.count
        replace = total < int(count)
        global_indices = self._generator.choice(total, size=int(count), replace=replace)
        offsets = np.cumsum([0, *[shard.count for shard in self._shards]])
        selected_by_shard: dict[int, list[int]] = {}
        for global_index in np.asarray(global_indices).tolist():
            shard_index = int(np.searchsorted(offsets[1:], int(global_index), side="right"))
            local_index = int(global_index) - int(offsets[shard_index])
            selected_by_shard.setdefault(shard_index, []).append(local_index)

        replays: list[TransitionReplay] = []
        for shard_index, local_indices in selected_by_shard.items():
            shard = self._shards[shard_index]
            replay = load_transition_replay(shard.path)
            if replay.count != shard.count or replay.content_sha256() != shard.content_sha256:
                raise ValueError(f"replay shard metadata mismatch: {shard.path}")
            replays.append(replay.subset(local_indices))
        sampled = concatenate_replays(replays)
        if sampled.count != int(count):
            raise AssertionError("replay shard sampler returned the wrong sample count")
        return sampled


def phase2_update_steps(
    max_environment_steps: int = 20000,
    update_interval: int = 100,
) -> tuple[int, ...]:
    """Return nominal chunk boundaries for the frozen Phase 2 schedule."""
    maximum = int(max_environment_steps)
    interval = int(update_interval)
    if maximum < 1 or interval < 1 or maximum % interval:
        raise ValueError("max_environment_steps must be divisible by a positive update_interval")
    return tuple(range(interval, maximum + 1, interval))


def freeze_lewm_encoder_projector(model: nn.Module) -> tuple[str, ...]:
    """Freeze visual representation modules and return trainable parameter names."""
    for name in ("encoder", "projector"):
        module = getattr(model, name, None)
        if module is None:
            raise AttributeError(f"LeWM model is missing required module: {name}")
        module.requires_grad_(False)
        module.eval()

    trainable_modules = ("predictor", "action_encoder", "pred_proj")
    names: list[str] = []
    for name in trainable_modules:
        module = getattr(model, name, None)
        if module is None:
            raise AttributeError(f"LeWM model is missing required module: {name}")
        module.requires_grad_(True)
        names.extend(f"{name}.{parameter_name}" for parameter_name, _ in module.named_parameters())
    return tuple(names)


def set_lewm_adaptation_mode(model: nn.Module) -> None:
    """Keep frozen representation statistics fixed while training dynamics."""
    model.train()
    for name in ("encoder", "projector"):
        getattr(model, name).eval()
    for name in ("predictor", "action_encoder", "pred_proj"):
        getattr(model, name).train()


def freeze_fast_encoder_projector(model: nn.Module) -> tuple[str, ...]:
    """Freeze Fast-LeWAM's visual representation and expose Stage-B parameters.

    Fast-LeWAM has a shared DiT rather than LeWM's ``action_encoder`` /
    ``pred_proj`` pair.  Stage-B post-training therefore freezes only the
    visual encoder/projector and leaves the latent dynamics path (including
    the shared DiT, latent head, action input, and positional parameters)
    trainable.
    """
    model.requires_grad_(True)
    for name in ("encoder", "projector"):
        module = getattr(model, name, None)
        if module is None:
            raise AttributeError(f"Fast-LeWAM model is missing required module: {name}")
        module.requires_grad_(False)
        module.eval()
    names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if not names:
        raise ValueError("no trainable Fast-LeWAM Stage-B parameters remain")
    return names


def set_fast_adaptation_mode(model: nn.Module) -> None:
    """Restore Fast-LeWAM's train/eval split after environment evaluation."""
    model.train()
    model.encoder.eval()
    model.projector.eval()


class FastStageBOnlineAdapter:
    """Update Fast-LeWAM Stage-B dynamics from executed transition windows."""

    def __init__(
        self,
        model: nn.Module,
        *,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-3,
        gradient_clip_norm: float = 1.0,
        seed: int = 3072,
        device: str | torch.device = "cpu",
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self._trainable_names = freeze_fast_encoder_projector(self.model)
        parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.sampler_seed = int(seed)
        self.optimizer_steps = 0

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return self._trainable_names

    def prepare_for_update(self) -> None:
        freeze_fast_encoder_projector(self.model)
        set_fast_adaptation_mode(self.model)

    def _loss(self, replay: TransitionReplay, indices: torch.Tensor) -> torch.Tensor:
        observations = replay.observations[indices].to(self.device)
        actions = replay.actions[indices].to(self.device)
        if int(observations.shape[1]) != int(self.model.action_horizon) + 1:
            raise ValueError(
                "Fast-LeWAM replay must contain action_horizon + 1 observations; "
                f"got {observations.shape[1]} for horizon {self.model.action_horizon}"
            )
        if int(actions.shape[1]) != int(self.model.action_horizon):
            raise ValueError(
                "Fast-LeWAM replay action history must equal action_horizon; "
                f"got {actions.shape[1]} for horizon {self.model.action_horizon}"
            )
        with torch.no_grad():
            embeddings = self.model.encode_pixels(observations)
            states = embeddings[:, :-1]
            target = embeddings[:, 1:].detach()
        predicted = self.model.predict_training_latents(states, actions)
        return (predicted - target).square().mean()

    def update(
        self,
        replay: TransitionReplay,
        *,
        batch_size: int = 64,
        updates: int = 1,
        mixed: bool = True,
        source: str | None = None,
    ) -> UpdateReport:
        if int(updates) < 1:
            raise ValueError("updates must be positive")
        sampler = MixedReplaySampler(replay, seed=self.sampler_seed + self.optimizer_steps)
        losses: list[float] = []
        source_counts = {"offline": 0, "online": 0}
        self.prepare_for_update()
        for _ in range(int(updates)):
            indices, counts = sampler.sample(int(batch_size), mixed=mixed, source=source)
            if indices.numel() == 0:
                break
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._loss(replay, indices)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                self.gradient_clip_norm,
            )
            self.optimizer.step()
            self.optimizer_steps += 1
            losses.append(float(loss.detach().cpu()))
            source_counts["offline"] += counts["offline"]
            source_counts["online"] += counts["online"]
        return UpdateReport(
            loss=None if not losses else float(np.mean(losses)),
            sample_count=source_counts["offline"] + source_counts["online"],
            offline_count=source_counts["offline"],
            online_count=source_counts["online"],
            optimizer_steps=self.optimizer_steps,
        )


class MixedReplaySampler:
    """Deterministic source-balanced sampler without fabricated online rows."""

    def __init__(self, replay: TransitionReplay, *, seed: int = 3072):
        self.replay = replay
        self.seed = int(seed)
        self._generator = torch.Generator(device="cpu").manual_seed(self.seed)

    def sample(
        self,
        batch_size: int,
        *,
        mixed: bool,
        source: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        if source not in {None, "offline", "online"}:
            raise ValueError("source must be None, 'offline', or 'online'")
        groups = {
            "offline": [i for i, source in enumerate(self.replay.sources) if source == "offline"],
            "online": [i for i, source in enumerate(self.replay.sources) if source == "online"],
        }
        if source is not None:
            available = groups[source]
            if not available:
                return torch.empty(0, dtype=torch.long), {"offline": 0, "online": 0}
            count = min(int(batch_size), len(available))
            order = torch.randperm(len(available), generator=self._generator)[:count]
            indices = torch.tensor([available[index] for index in order.tolist()], dtype=torch.long)
            return indices, {
                "offline": count if source == "offline" else 0,
                "online": count if source == "online" else 0,
            }

        if not mixed:
            # Preserve the historical offline-only arm when no explicit source
            # is supplied. The online-only arm must opt in explicitly so that a
            # caller cannot accidentally turn an offline update into online
            # adaptation merely by passing a replay with a different source.
            return self.sample(int(batch_size), mixed=False, source="offline")

        if not groups["offline"] or not groups["online"]:
            return torch.empty(0, dtype=torch.long), {"offline": 0, "online": 0}
        half = max(1, int(batch_size) // 2)
        pair_count = min(half, len(groups["offline"]), len(groups["online"]))
        if pair_count == 0:
            return torch.empty(0, dtype=torch.long), {"offline": 0, "online": 0}
        offline_count = online_count = pair_count
        def draw(values: list[int], count: int) -> list[int]:
            order = torch.randperm(len(values), generator=self._generator)[:count]
            return [values[index] for index in order.tolist()]
        selected = draw(groups["offline"], offline_count) + draw(groups["online"], online_count)
        return torch.tensor(selected, dtype=torch.long), {
            "offline": offline_count,
            "online": online_count,
        }


@dataclass(frozen=True)
class UpdateReport:
    loss: float | None
    sample_count: int
    offline_count: int
    online_count: int
    optimizer_steps: int


class StageBOnlineAdapter:
    """Update only LeWM dynamics modules from executed transition windows."""

    def __init__(
        self,
        model: nn.Module,
        *,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-3,
        gradient_clip_norm: float = 1.0,
        seed: int = 3072,
        device: str | torch.device = "cpu",
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        freeze_lewm_encoder_projector(self.model)
        self._trainable_names = tuple(
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        )
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("no trainable LeWM dynamics parameters remain")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.sampler_seed = int(seed)
        self.optimizer_steps = 0

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return self._trainable_names

    def prepare_for_update(self) -> None:
        """Restore optimizer-facing requires-grad flags after evaluation.

        ``make_world_policy_from_model`` puts the model in inference mode and
        disables every parameter.  Phase 2 evaluates the same arm at 0/5k/
        10k/20k boundaries, so the next update must explicitly restore the
        frozen-representation/trainable-dynamics split.
        """
        freeze_lewm_encoder_projector(self.model)
        set_lewm_adaptation_mode(self.model)

    def _loss(self, replay: TransitionReplay, indices: torch.Tensor) -> torch.Tensor:
        observations = replay.observations[indices].to(self.device)
        actions = replay.actions[indices].to(self.device)
        with torch.no_grad():
            encoded = self.model.encode({"pixels": observations})["emb"]
            context = encoded[:, : replay.history_size]
            target = encoded[:, 1:].detach()
        action_embedding = self.model.action_encoder(actions)
        prediction = self.model.predict(context, action_embedding[:, : replay.history_size])
        return (prediction - target).square().mean()

    def update(
        self,
        replay: TransitionReplay,
        *,
        batch_size: int = 64,
        updates: int = 1,
        mixed: bool = True,
        source: str | None = None,
    ) -> UpdateReport:
        if int(updates) < 1:
            raise ValueError("updates must be positive")
        sampler = MixedReplaySampler(replay, seed=self.sampler_seed + self.optimizer_steps)
        losses: list[float] = []
        source_counts = {"offline": 0, "online": 0}
        self.prepare_for_update()
        for _ in range(int(updates)):
            indices, counts = sampler.sample(int(batch_size), mixed=mixed, source=source)
            if indices.numel() == 0:
                break
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._loss(replay, indices)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                self.gradient_clip_norm,
            )
            self.optimizer.step()
            self.optimizer_steps += 1
            losses.append(float(loss.detach().cpu()))
            source_counts["offline"] += counts["offline"]
            source_counts["online"] += counts["online"]
        return UpdateReport(
            loss=None if not losses else float(np.mean(losses)),
            sample_count=source_counts["offline"] + source_counts["online"],
            offline_count=source_counts["offline"],
            online_count=source_counts["online"],
            optimizer_steps=self.optimizer_steps,
        )


__all__ = [
    "FastStageBOnlineAdapter",
    "MixedReplaySampler",
    "ReplayShard",
    "ReplayShardPool",
    "StageBOnlineAdapter",
    "TransitionReplay",
    "UpdateReport",
    "build_offline_window_manifest",
    "concatenate_replays",
    "freeze_lewm_encoder_projector",
    "freeze_fast_encoder_projector",
    "load_offline_replay",
    "load_transition_replay",
    "save_transition_replay",
    "set_lewm_adaptation_mode",
    "set_fast_adaptation_mode",
    "phase2_update_steps",
]
