"""Publish sampling-revised results from an identical audited rollout trace."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .round3_phase1 import (
    CohortManifest,
    enrich_result_payload,
    load_result,
    write_round3_result,
)


def derive_sampling_identity_result(
    source_result: Mapping[str, Any],
    *,
    manifest: CohortManifest,
) -> dict[str, Any]:
    """Copy a round3 trace when only the protocol's sampling label changes.

    ``sampling_revised`` and ``round3_revised`` use the same audited runtime
    predicate.  Reusing the immutable trace keeps their comparison free of
    additional rollout noise while the target manifest still records the
    distinct cohort and protocol identity.
    """
    if source_result.get("status") != "ok":
        raise ValueError("source result must have status=ok")
    if source_result.get("protocol_variant") != "round3_revised":
        raise ValueError("sampling identity source must be round3_revised")
    source_episodes = source_result.get("episodes")
    if not isinstance(source_episodes, list):
        raise ValueError("source result must contain a list of episode traces")
    expected = {(entry.episode_id, entry.start_step) for entry in manifest.entries}
    observed = {(record.get("episode_id"), record.get("start_step")) for record in source_episodes}
    if observed != expected:
        raise ValueError("source trace does not match sampling_revised manifest")

    payload = deepcopy(dict(source_result))
    payload["episodes"] = deepcopy(source_episodes)
    payload["derived_from"] = {
        "kind": "sampling_identity",
        "source_cohort_sha256": source_result.get("cohort_sha256"),
        "source_protocol_variant": source_result.get("protocol_variant"),
        "source_trace_path": source_result.get("trace_path"),
    }
    for key in ("summary", "cohort", "cohort_id", "cohort_sha256", "trace_path"):
        payload.pop(key, None)
    return payload


def derive_sampling_identity_result_file(
    source_path: str | Path,
    target_path: str | Path,
    *,
    manifest_path: str | Path,
    task: str,
    trace_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Read a round3 result and publish it under sampling_revised metadata."""
    source = load_result(source_path)
    manifest = CohortManifest.load(manifest_path)
    if manifest.task != task:
        raise ValueError("manifest task differs from requested derivation task")
    if manifest.protocol_variant != "sampling_revised":
        raise ValueError("manifest protocol must be sampling_revised")
    payload = derive_sampling_identity_result(source, manifest=manifest)
    payload = enrich_result_payload(
        payload,
        manifest=manifest,
        protocol_variant="sampling_revised",
        trace_records=payload["episodes"],
    )
    return write_round3_result(
        payload,
        target_path,
        trace_records=payload["episodes"],
        trace_output_dir=trace_output_dir,
    )


__all__ = [
    "derive_sampling_identity_result",
    "derive_sampling_identity_result_file",
]
