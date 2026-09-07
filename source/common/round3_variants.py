"""Explicit cohort identities for the four Phase 1 development protocols."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .round3_phase1 import (
    CohortManifest,
    PHASE1_DEV_EPISODES,
    build_legacy_manifest,
    build_revised_cohorts,
)


def with_protocol_variant(
    manifest: CohortManifest, protocol_variant: str
) -> CohortManifest:
    """Clone a manifest while changing only its protocol identity/hash."""
    if protocol_variant not in {"legacy", "sampling_revised", "tolerance_revised", "round3_revised"}:
        raise ValueError(f"unknown protocol variant {protocol_variant!r}")
    cohort_id = manifest.cohort_id.rsplit("_", 1)[0] + f"_{protocol_variant}"
    return replace(
        manifest,
        cohort_id=cohort_id,
        protocol_variant=protocol_variant,
        cohort_sha256=None,
    )


def build_development_protocol_manifests(
    dataset: Any,
    *,
    task: str,
    seed: int = 42,
    goal_offset_steps: int = 25,
    num_eval: int = PHASE1_DEV_EPISODES,
) -> dict[str, CohortManifest]:
    """Return legacy/sampling/tolerance/joint manifests for development.

    ``sampling_revised`` and ``round3_revised`` share the same data-only
    revised cohort. ``legacy`` and ``tolerance_revised`` share the exact
    upstream cohort. Label changes are applied later to the same trace by
    :func:`source.common.round3_analysis.relabel_trace`.
    """
    if int(num_eval) != PHASE1_DEV_EPISODES:
        raise ValueError("the Phase 1 development protocol is fixed at 50 episodes")
    legacy = build_legacy_manifest(
        dataset,
        task=task,
        seed=seed,
        goal_offset_steps=goal_offset_steps,
        num_eval=num_eval,
    )
    revised = build_revised_cohorts(
        dataset,
        task=task,
        seed=seed,
        goal_offset_steps=goal_offset_steps,
        dev_count=num_eval,
        final_count=0,
        online_count=None,
    )["dev"]
    return {
        "legacy": with_protocol_variant(legacy, "legacy"),
        "sampling_revised": with_protocol_variant(revised, "sampling_revised"),
        "tolerance_revised": with_protocol_variant(legacy, "tolerance_revised"),
        "round3_revised": with_protocol_variant(revised, "round3_revised"),
    }


__all__ = ["build_development_protocol_manifests", "with_protocol_variant"]
