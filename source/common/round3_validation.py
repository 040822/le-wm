"""Acceptance checks for Phase 1 manifests and result artifacts."""

from __future__ import annotations

from typing import Any, Mapping

from .round3_phase1 import CohortManifest
from .round3_protocol import PREDICATE_VERSION, ROUND3_PROTOCOL


def validate_cohort_manifest(
    manifest: CohortManifest,
    *,
    expected_count: int | None = None,
    required_kind: str | None = None,
) -> None:
    if manifest.protocol_variant not in {"legacy", "sampling_revised", "tolerance_revised", "round3_revised"}:
        raise ValueError(f"unknown protocol variant {manifest.protocol_variant!r}")
    if expected_count is not None and len(manifest.entries) != int(expected_count):
        raise ValueError(
            f"manifest {manifest.cohort_id} has {len(manifest.entries)} entries; "
            f"expected {expected_count}"
        )
    if required_kind is not None and manifest.cohort_kind != required_kind:
        raise ValueError(
            f"manifest {manifest.cohort_id} is {manifest.cohort_kind!r}; "
            f"expected {required_kind!r}"
        )
    rows = [entry.row_index for entry in manifest.entries]
    if len(rows) != len(set(rows)):
        raise ValueError("cohort row indices are not unique")
    episodes = [str(entry.episode_id) for entry in manifest.entries]
    if manifest.protocol_variant != "legacy" and len(episodes) != len(set(episodes)):
        raise ValueError("revised cohort raw episode IDs are not unique")
    if manifest.cohort_sha256 != manifest.computed_sha256:
        raise ValueError("cohort SHA256 does not match its canonical payload")


def validate_result_payload(
    payload: Mapping[str, Any], *, expected_count: int | None = None
) -> None:
    """Reject incomplete result JSON before it can enter the phase matrix."""
    if payload.get("protocol") != ROUND3_PROTOCOL:
        raise ValueError("result does not use the Round 3 Phase 1 protocol")
    if payload.get("status") != "ok":
        raise ValueError(f"result status is not ok: {payload.get('status')!r}")
    if payload.get("predicate_version") != PREDICATE_VERSION:
        raise ValueError("result predicate version is missing or stale")
    cohort_hash = payload.get("cohort_sha256")
    if not isinstance(cohort_hash, str) or len(cohort_hash) != 64:
        raise ValueError("result cohort_sha256 is missing or malformed")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("result episodes must be a list")
    if expected_count is not None and len(episodes) != int(expected_count):
        raise ValueError("result episode count does not match the frozen cohort")
    success_vector = payload.get("summary", {}).get("success_vector")
    if success_vector is not None and len(success_vector) != len(episodes):
        raise ValueError("summary success_vector and episodes have different lengths")


__all__ = ["validate_cohort_manifest", "validate_result_payload"]
