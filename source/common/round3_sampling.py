"""Stratified revised-cohort implementation.

The public helpers in ``round3_phase1`` expose the data model and hashing
rules.  This module supplies the allocation step that reserves exactly the
requested number of starts in every distance stratum while keeping raw
episode IDs disjoint across dev, final, and online pools.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import round3_phase1 as p


def _balanced_episode_split(
    raw_episode_ids: Sequence[Any],
    candidates: Sequence[Any],
    *,
    boundaries: Sequence[float],
    seed: int,
    dev_count: int,
    final_count: int,
    online_count: int | None,
) -> dict[str, tuple[Any, ...]]:
    if dev_count % p.PHASE1_STRATA or final_count % p.PHASE1_STRATA:
        raise ValueError("dev_count and final_count must be divisible by five")
    by_stratum: list[list[Any]] = [[] for _ in range(p.PHASE1_STRATA)]
    ranked = sorted(
        candidates,
        key=lambda item: (item.distance, p._episode_sort_key(item.episode_id), item.start_step, item.row_index),
    )
    strata_by_row = {}
    for rank, positions in enumerate(np.array_split(np.arange(len(ranked)), p.PHASE1_STRATA)):
        for position in positions:
            strata_by_row[ranked[int(position)].row_index] = rank
    for candidate in candidates:
        by_stratum[strata_by_row[candidate.row_index]].append(candidate)
    # Use one deterministic permutation for all raw IDs.  Allocation then
    # walks that permutation inside each physical-distance stratum, so it is
    # still an episode-ID split and never a model-result split.
    order = p.split_episode_ids(
        raw_episode_ids,
        seed=int(seed),
        dev_count=0,
        final_count=0,
        online_count=len(raw_episode_ids),
    )["online"]
    order_index = {p._episode_key(value): index for index, value in enumerate(order)}
    for group in by_stratum:
        group.sort(key=lambda item: order_index.get(p._episode_key(item.episode_id), len(order)))

    dev_target = dev_count // p.PHASE1_STRATA
    final_target = final_count // p.PHASE1_STRATA
    dev_ids: list[Any] = []
    final_ids: list[Any] = []
    for rank, group in enumerate(by_stratum):
        required = dev_target + final_target
        if len(group) < required:
            raise ValueError(
                f"not enough valid raw episodes in stratum {rank}: "
                f"{len(group)} available, {required} required"
            )
        dev_ids.extend(item.episode_id for item in group[:dev_target])
        final_ids.extend(item.episode_id for item in group[dev_target:required])
    selected = {p._episode_key(value) for value in dev_ids + final_ids}
    online_ids = [value for value in order if p._episode_key(value) not in selected]
    online_target = len(online_ids) if online_count is None else int(online_count)
    if online_target < 0 or online_target > len(online_ids):
        raise ValueError(
            f"online_count={online_target} exceeds the unallocated raw episode pool "
            f"of {len(online_ids)}"
        )
    return {
        "dev": tuple(dev_ids),
        "final": tuple(final_ids),
        "online": tuple(online_ids[:online_target]),
    }

def build_revised_cohorts(
    dataset: Any,
    *,
    task: str,
    seed: int = int(p.ROUND3_EVAL_DEFAULTS["seed"]),
    goal_offset_steps: int = int(p.ROUND3_EVAL_DEFAULTS["goal_offset_steps"]),
    dev_count: int = p.PHASE1_DEV_EPISODES,
    final_count: int = p.PHASE1_FINAL_EPISODES,
    online_count: int | None = None,
) -> dict[str, p.CohortManifest]:
    """Build cohorts with fixed bins and exact per-bin dev/final quotas."""
    episode_column = p._episode_column(dataset)
    raw_episode_ids = np.unique(p._column(dataset, episode_column))
    all_candidates, all_diagnostics = p._collect_candidates(
        dataset,
        task=task,
        episode_ids=raw_episode_ids,
        seed=int(seed) + 17,
        goal_offset_steps=int(goal_offset_steps),
    )
    boundaries = p.fixed_strata_boundaries([item.distance for item in all_candidates])
    split = _balanced_episode_split(
        raw_episode_ids,
        all_candidates,
        boundaries=boundaries,
        online_count=online_count,
        seed=int(seed),
        dev_count=int(dev_count),
        final_count=int(final_count),
    )
    strata_by_row = {}
    ranked = sorted(
        all_candidates,
        key=lambda item: (item.distance, p._episode_sort_key(item.episode_id), item.start_step, item.row_index),
    )
    for rank, positions in enumerate(np.array_split(np.arange(len(ranked)), p.PHASE1_STRATA)):
        for position in positions:
            strata_by_row[ranked[int(position)].row_index] = rank
    candidate_by_episode = {p._episode_key(item.episode_id): item for item in all_candidates}
    manifests = {}
    requested = {"dev": int(dev_count), "final": int(final_count)}
    for name in ("dev", "final"):
        if requested[name] == 0:
            continue
        candidates = [candidate_by_episode[p._episode_key(item)] for item in split[name]]
        entries, counts, selected_counts = p._select_candidates(
            candidates,
            num_eval=requested[name],
            boundaries=boundaries,
            seed=int(seed) + (101 if name == "dev" else 202),
            strata_by_row=strata_by_row,
        )
        diagnostics = {
            "episode_count": len(split[name]),
            "candidate_count": len(candidates),
            "candidate_pool_count": len(all_candidates),
            "candidate_pool_diagnostics": all_diagnostics,
            "candidate_count_by_stratum": list(counts),
            "selected_count_by_stratum": list(selected_counts),
            "raw_episode_count": len(split[name]),
            "online_episode_count": len(split["online"]),
            "split_is_stratified_by_audited_distance": True,
        }
        manifests[name] = p._make_revised_manifest(
            task=task,
            cohort_kind=name,
            entries=entries,
            split=split,
            seed=int(seed),
            goal_offset_steps=int(goal_offset_steps),
            boundaries=boundaries,
            candidate_counts=counts,
            selected_counts=selected_counts,
            diagnostics=diagnostics,
        )
    return manifests


__all__ = ["build_revised_cohorts"]
