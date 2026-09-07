"""Derive predicate-only Phase 1 protocol results from an existing trace."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .round3_analysis import relabel_trace
from .round3_phase1 import (
    CohortManifest,
    load_result,
    write_round3_result,
)


def derive_relabelled_result(
    source_result: Mapping[str, Any],
    *,
    manifest: CohortManifest,
    task: str,
    protocol_variant: str,
    action_block: int = 5,
) -> dict[str, Any]:
    """Copy one result while recomputing labels from its immutable rollout trace.

    This is used when only the success predicate changes.  Actions, current and
    goal states, episode IDs, and start rows are retained; the manifest and
    protocol identity are replaced with the requested frozen variant.
    """
    if protocol_variant not in {"tolerance_revised", "round3_revised"}:
        raise ValueError(
            "predicate-only derivation is only valid for tolerance_revised or "
            "round3_revised"
        )
    source_episodes = source_result.get("episodes")
    if not isinstance(source_episodes, list):
        raise ValueError("source result must contain a list of episode traces")
    records = relabel_trace(source_episodes, task=task, action_block=action_block)
    payload = deepcopy(dict(source_result))
    payload["episodes"] = records
    payload["status"] = "ok"
    payload["derived_from"] = {
        "kind": "predicate_relabel",
        "source_cohort_sha256": source_result.get("cohort_sha256"),
        "source_protocol_variant": source_result.get("protocol_variant"),
        "source_trace_path": source_result.get("trace_path"),
    }
    # This field came from the environment's original termination label.  It
    # is not a substitute for the newly recomputed predicate label.
    payload.pop("runtime_success_rate_percent", None)
    payload.pop("summary", None)
    payload.pop("cohort", None)
    payload.pop("cohort_id", None)
    payload.pop("cohort_sha256", None)
    payload.pop("trace_path", None)
    return payload


def derive_protocol_result_file(
    source_path: str | Path,
    target_path: str | Path,
    *,
    manifest_path: str | Path,
    task: str,
    protocol_variant: str,
    action_block: int = 5,
    trace_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Read a source result, relabel it, and atomically publish the result."""
    source = load_result(source_path)
    manifest = CohortManifest.load(manifest_path)
    if manifest.task != task:
        raise ValueError("manifest task differs from requested derivation task")
    if manifest.protocol_variant != protocol_variant:
        raise ValueError("manifest protocol differs from requested derivation variant")
    payload = derive_relabelled_result(
        source,
        manifest=manifest,
        task=task,
        protocol_variant=protocol_variant,
        action_block=action_block,
    )
    from .round3_phase1 import enrich_result_payload

    payload = enrich_result_payload(
        payload,
        manifest=manifest,
        protocol_variant=protocol_variant,
        trace_records=payload["episodes"],
    )
    return write_round3_result(
        payload,
        target_path,
        trace_records=payload["episodes"],
        trace_output_dir=trace_output_dir,
    )


__all__ = ["derive_relabelled_result", "derive_protocol_result_file"]
