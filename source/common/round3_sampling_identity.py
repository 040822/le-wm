"""Publish sampling-revised results from an identical audited rollout trace."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .round3_derivation import (
    _coerce_manifest,
    _derived_from,
    _immutable_trace_signature,
    _prepare_target_payload,
    _raw_termination_signature,
    _resolve_source_trace,
    _validate_episode_order,
    _validate_frozen_source,
    preflight_derived_publish,
    validate_derivation_source,
)
from .round3_phase1 import (
    CohortManifest,
    load_result,
    sha256_json,
    summarize_episodes,
    validate_trace_contract,
    write_round3_result,
)
from .round3_protocol import ROUND3_PROTOCOL


def derive_sampling_identity_result(
    source_result: Mapping[str, Any],
    *,
    manifest: CohortManifest,
    source_manifest: CohortManifest | Mapping[str, Any] | None = None,
    source_result_path: str | Path | None = None,
    source_trace_path: str | Path | None = None,
) -> dict[str, Any]:
    """Copy a round3 trace when only the sampling identity changes."""
    if source_result.get("status") != "ok":
        raise ValueError("source result must have status=ok")
    if source_result.get("task") != manifest.task:
        raise ValueError("source/target task differs for sampling identity derivation")
    if manifest.protocol_variant != "sampling_revised":
        raise ValueError("target manifest protocol must be sampling_revised")
    if source_result.get("protocol_variant") != "round3_revised":
        raise ValueError("sampling identity source must be round3_revised")
    source_episodes = source_result.get("episodes")
    if not isinstance(source_episodes, list) or not source_episodes:
        raise ValueError("source result must contain non-empty episode traces")
    source_manifest_value = source_manifest
    if source_manifest_value is None and isinstance(source_result.get("cohort"), Mapping):
        source_manifest_value = source_result["cohort"]
    if source_manifest_value is None:
        raise ValueError("source result is missing its embedded cohort manifest")
    if source_trace_path is None and source_result.get("trace_path") not in (None, ""):
        declared_trace = Path(str(source_result["trace_path"]))
        source_parent = (
            Path(source_result_path).parent
            if source_result_path is not None
            else Path.cwd()
        )
        for candidate in (
            (declared_trace,)
            if declared_trace.is_absolute()
            else (declared_trace, source_parent / declared_trace)
        ):
            if candidate.is_file():
                source_trace_path = candidate
                break
    _validate_episode_order(source_episodes, source_manifest_value)
    validate_trace_contract(source_episodes)
    if source_trace_path is not None:
        validate_derivation_source(
            source_result,
            manifest=source_manifest_value,
            trace_path=source_trace_path,
            task=manifest.task,
            source_result_path=source_result_path,
        )
    else:
        _validate_frozen_source(
            source_result,
            source_manifest=_coerce_manifest(source_manifest_value),
            source_trace=Path(source_result.get("trace_path"))
            if source_result.get("trace_path") is not None
            else Path("__missing_source_trace__.jsonl"),
        )
    source_termination = _raw_termination_signature(source_episodes)
    source_immutable_trace = _immutable_trace_signature(source_episodes)
    payload = deepcopy(dict(source_result))
    payload["protocol"] = ROUND3_PROTOCOL
    payload["episodes"] = deepcopy(source_episodes)
    if _raw_termination_signature(payload["episodes"]) != source_termination:
        raise ValueError("sampling identity derivation changed raw termination provenance")
    if _immutable_trace_signature(payload["episodes"]) != source_immutable_trace:
        raise ValueError("sampling identity derivation changed immutable trace provenance")
    payload["summary"] = summarize_episodes(payload["episodes"])
    payload["success_rate"] = payload["summary"]["success_rate"]
    payload["success_vector_sha256"] = sha256_json(payload["summary"]["success_vector"])
    payload["summary_sha256"] = sha256_json(payload["summary"])
    payload["derived_from"] = _derived_from(
        source_result,
        kind="sampling_identity",
        source_manifest=source_manifest_value,
        source_result_path=source_result_path,
        source_trace_path=source_trace_path,
    )
    payload["derived_from"]["mapping"]["target_protocol_variant"] = "sampling_revised"
    for key in ("cohort", "cohort_id", "cohort_sha256", "trace_path"):
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
    """Strictly validate a round3 result and publish sampling identity."""
    source_file = Path(source_path)
    source = load_result(source_file)
    manifest = CohortManifest.load(manifest_path)
    if manifest.task != task:
        raise ValueError("manifest task differs from requested derivation task")
    if manifest.protocol_variant != "sampling_revised":
        raise ValueError("manifest protocol must be sampling_revised")
    source_manifest = _coerce_manifest(source.get("cohort"))
    if source_manifest.task != task:
        raise ValueError("source cohort task differs from requested task")
    source_trace = _resolve_source_trace(source, source_file)
    _validate_frozen_source(source, source_manifest=source_manifest, source_trace=source_trace)
    validate_derivation_source(
        source,
        manifest=source_manifest,
        trace_path=source_trace,
        task=task,
        source_result_path=source_file,
    )
    payload = derive_sampling_identity_result(
        source,
        manifest=manifest,
        source_manifest=source_manifest,
        source_result_path=source_file,
        source_trace_path=source_trace,
    )
    target_dir = Path(target_path)
    trace_dir = Path(trace_output_dir) if trace_output_dir is not None else target_dir
    result_dir, trace_dir, target_trace = preflight_derived_publish(
        source_result_path=source_file,
        source_trace_path=source_trace,
        target_result_dir=target_dir,
        target_trace_dir=trace_dir,
    )
    candidate = _prepare_target_payload(
        payload,
        manifest=manifest,
        protocol_variant="sampling_revised",
        target_trace_path=target_trace,
    )
    return write_round3_result(
        candidate,
        result_dir,
        trace_records=candidate["episodes"],
        trace_output_dir=trace_dir,
    )


__all__ = [
    "derive_sampling_identity_result",
    "derive_sampling_identity_result_file",
]
