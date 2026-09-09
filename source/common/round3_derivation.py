"""Strict predicate-only Phase 1 derivation from immutable result traces."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .round3_analysis import relabel_trace
from .round3_phase1 import (
    CohortManifest,
    _result_directory,
    _trace_directory,
    enrich_result_payload,
    load_result,
    sha256_json,
    sha256_file,
    summarize_episodes,
    validate_trace_contract,
    write_round3_result,
)
from .round3_protocol import ROUND3_PROTOCOL
from .round3_validation import trace_content_sha256, trace_summary_sha256


_IDENTITY_FIELDS = (
    "slot",
    "episode_id",
    "dataset_episode",
    "start_step",
    "row_index",
    "goal_row_index",
    "goal_step",
)
_ALLOWED_PREDICATE_MAPPINGS = {
    ("legacy", "tolerance_revised"),
    ("sampling_revised", "round3_revised"),
}


def _json_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, ensure_ascii=False, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _manifest_entry_value(entry: Any, field: str, index: int) -> Any:
    if isinstance(entry, Mapping):
        value = entry.get(field)
    else:
        value = getattr(entry, field, None)
    if field == "slot":
        return index
    if field == "dataset_episode":
        if value is None:
            value = (
                entry.get("episode_id")
                if isinstance(entry, Mapping)
                else getattr(entry, "episode_id", None)
            )
    return value


def _expected_identity(manifest: CohortManifest | Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = manifest.entries if isinstance(manifest, CohortManifest) else manifest.get("entries", ())
    return [
        {
            field: _manifest_entry_value(entry, field, index)
            for field in _IDENTITY_FIELDS
        }
        for index, entry in enumerate(entries)
    ]


def _observed_identity(
    records: Sequence[Mapping[str, Any]],
    *,
    manifest: CohortManifest | Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    observed = []
    expected = _expected_identity(manifest) if manifest is not None else None
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"episode {index} must be a mapping")
        missing = []
        for field in _IDENTITY_FIELDS:
            if field in record:
                continue
            optional_none = (
                expected is not None
                and index < len(expected)
                and expected[index][field] is None
                and field in {"goal_row_index", "goal_step"}
            )
            if optional_none:
                continue
            missing.append(field)
        if missing:
            raise ValueError(
                f"episode {index} is missing ordered identity fields: {', '.join(missing)}"
            )
        observed.append(
            {
                field: record.get(field)
                if field in record
                else None
                for field in _IDENTITY_FIELDS
            }
        )
    return observed


def _validate_episode_order(
    records: Sequence[Mapping[str, Any]], manifest: CohortManifest | Mapping[str, Any]
) -> None:
    expected = _expected_identity(manifest)
    observed = _observed_identity(records, manifest=manifest)
    if len(observed) != len(expected):
        raise ValueError(
            f"source trace has {len(observed)} episodes; manifest has {len(expected)}"
        )
    for index, (actual, wanted) in enumerate(zip(observed, expected)):
        if not _json_equal(actual, wanted):
            raise ValueError(
                f"episode order/identity mismatch at index {index}: "
                f"actual={actual!r} expected={wanted!r}"
            )


def _read_trace_records(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    try:
        raw = path.read_bytes()
        lines = raw.decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"source trace cannot be read: {path}") from exc
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("source trace must contain non-blank JSONL rows")
    records = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"source trace row {index} is not valid JSON") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"source trace row {index} must be an object")
        records.append(dict(value))
    return raw, records


def _coerce_manifest(value: CohortManifest | Mapping[str, Any] | None) -> CohortManifest:
    if isinstance(value, CohortManifest):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("source result is missing its embedded cohort manifest")
    try:
        return CohortManifest.from_dict(value, verify_hash=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("source embedded cohort manifest is invalid") from exc


def validate_derivation_source(
    source_result: Mapping[str, Any],
    *,
    manifest: CohortManifest | Mapping[str, Any],
    trace_path: str | Path,
    task: str,
    source_result_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate source result, actual JSONL, raw provenance, and ordered identity."""
    if not isinstance(source_result, Mapping):
        raise ValueError("source result must be a mapping")
    if source_result.get("status") != "ok":
        raise ValueError("source result must have status=ok")
    if source_result.get("protocol") != ROUND3_PROTOCOL:
        raise ValueError("source result protocol is stale or missing")
    if source_result.get("task") != task:
        raise ValueError("source result task differs from requested task")
    source_variant = source_result.get("protocol_variant")
    if source_variant not in {"legacy", "sampling_revised", "round3_revised"}:
        raise ValueError(
            "source result protocol variant is not a permitted derivation source"
        )
    records = source_result.get("episodes")
    if not isinstance(records, list) or not records:
        raise ValueError("source result must contain non-empty episode traces")
    parsed_manifest = _coerce_manifest(manifest)
    embedded_manifest = _coerce_manifest(source_result.get("cohort"))
    if embedded_manifest.as_dict() != parsed_manifest.as_dict():
        raise ValueError("source result cohort differs from its supplied manifest")
    if embedded_manifest.task != task:
        raise ValueError("source cohort task differs from requested task")
    trace = Path(trace_path)
    if not trace.is_file():
        raise ValueError(f"source trace is absent: {trace}")
    declared_trace_path = source_result.get("trace_path")
    if declared_trace_path not in (None, ""):
        declared = Path(str(declared_trace_path))
        source_parent = (
            Path(source_result_path).parent
            if source_result_path is not None
            else Path.cwd()
        )
        declared_candidates = (
            (declared,)
            if declared.is_absolute()
            else (declared, source_parent / declared)
        )
        if not any(candidate.resolve() == trace.resolve() for candidate in declared_candidates):
            raise ValueError("source result trace_path differs from the supplied trace")
    raw, file_records = _read_trace_records(trace)
    if len(file_records) != len(records) or any(
        not _json_equal(left, right) for left, right in zip(file_records, records)
    ):
        raise ValueError("source JSONL trace differs from embedded episodes")
    actual_trace_sha = sha256_file(trace)
    declared_trace = source_result.get("trace_content_sha256")
    if declared_trace is None:
        declared_trace = source_result.get("trace_sha256")
    if declared_trace is None:
        raise ValueError("source result is missing the actual JSONL trace hash")
    if declared_trace != actual_trace_sha:
        raise ValueError(
            f"source trace hash mismatch: declared={declared_trace!r} actual={actual_trace_sha!r}"
        )
    validate_trace_contract(records)
    _validate_episode_order(records, parsed_manifest)
    from .round3_validation import validate_result_payload

    validate_result_payload(
        source_result,
        manifest=parsed_manifest,
        expected_count=len(parsed_manifest.entries),
        trace_path=trace,
        require_integrity_hashes=True,
    )
    source_result_sha = None
    if source_result_path is not None:
        source_file = Path(source_result_path)
        if not source_file.is_file():
            raise ValueError(f"source result is absent: {source_file}")
        try:
            stored_source = load_result(source_file)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("source result file cannot be read as JSON") from exc
        if not _json_equal(stored_source, source_result):
            raise ValueError("source result mapping differs from its immutable file")
        source_result_sha = sha256_file(source_file)
    source_identity = _observed_identity(records, manifest=parsed_manifest)
    return {
        "records": deepcopy(records),
        "source_result_path": (
            str(Path(source_result_path).resolve())
            if source_result_path is not None
            else None
        ),
        "source_result_sha256": source_result_sha,
        "source_trace_path": str(trace.resolve()),
        "source_trace_sha256": actual_trace_sha,
        "source_trace_content_sha256": actual_trace_sha,
        "source_episode_identity": source_identity,
        "source_cohort_identity": {
            "task": parsed_manifest.task,
            "cohort_id": parsed_manifest.cohort_id,
            "cohort_kind": parsed_manifest.cohort_kind,
            "cohort_sha256": parsed_manifest.computed_sha256,
        },
        "source_cohort_sha256": parsed_manifest.computed_sha256,
        "raw_trace_bytes": len(raw),
    }


def _resolve_source_trace(source_result: Mapping[str, Any], source_path: Path) -> Path:
    declared = source_result.get("trace_path")
    if declared is None:
        raise ValueError("source result is missing trace_path")
    value = Path(str(declared))
    if value.is_absolute() and value.is_file():
        return value
    candidates = (value, source_path.parent / value)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"source trace is absent: {declared}")


def _validate_frozen_source(
    source: Mapping[str, Any],
    *,
    source_manifest: CohortManifest,
    source_trace: Path,
) -> None:
    from .round3_validation import validate_result_payload

    validate_result_payload(
        source,
        manifest=source_manifest,
        expected_count=len(source_manifest.entries),
        trace_path=source_trace,
        require_integrity_hashes=True,
    )


def _derived_from(
    source_result: Mapping[str, Any],
    *,
    kind: str,
    source_manifest: CohortManifest | Mapping[str, Any] | None,
    source_result_path: str | Path | None,
    source_trace_path: str | Path | None,
) -> dict[str, Any]:
    records = source_result.get("episodes", [])
    result_path = Path(source_result_path) if source_result_path is not None else None
    trace_path = Path(source_trace_path) if source_trace_path is not None else None
    source_identity = _observed_identity(records, manifest=source_manifest)
    source_cohort_identity: dict[str, Any] | None = None
    if isinstance(source_manifest, CohortManifest):
        source_cohort_identity = {
            "task": source_manifest.task,
            "cohort_id": source_manifest.cohort_id,
            "cohort_kind": source_manifest.cohort_kind,
            "cohort_sha256": source_manifest.computed_sha256,
        }
    elif isinstance(source_manifest, Mapping):
        source_cohort_identity = {
            "task": source_manifest.get("task"),
            "cohort_id": source_manifest.get("cohort_id"),
            "cohort_kind": source_manifest.get("cohort_kind"),
            "cohort_sha256": source_manifest.get("cohort_sha256"),
        }
    return {
        "kind": kind,
        "mapping": {
            "source_protocol_variant": source_result.get("protocol_variant"),
            "target_protocol_variant": None,
        },
        "source_protocol_variant": source_result.get("protocol_variant"),
        "source_cohort_sha256": (
            source_manifest.computed_sha256
            if isinstance(source_manifest, CohortManifest)
            else source_manifest.get("cohort_sha256") if isinstance(source_manifest, Mapping) else source_result.get("cohort_sha256")
        ),
        "source_cohort_identity": source_cohort_identity,
        "source_result_path": str(result_path.resolve()) if result_path is not None else None,
        "source_result_sha256": (
            sha256_file(result_path)
            if result_path is not None and result_path.is_file()
            else None
        ),
        "source_trace_path": (
            str(trace_path.resolve())
            if trace_path is not None
            else source_result.get("trace_path")
        ),
        "source_trace_sha256": (
            sha256_file(trace_path)
            if trace_path is not None and trace_path.is_file()
            else None
        ),
        "source_trace_content_sha256": (
            sha256_file(trace_path)
            if trace_path is not None and trace_path.is_file()
            else trace_content_sha256(records)
        ),
        "source_episode_identity": source_identity,
    }


def _raw_termination_signature(
    records: Sequence[Mapping[str, Any]],
) -> list[list[tuple[Any, Any, Any, Any]]]:
    return [
        [
            (
                step.get("raw_env_step"),
                step.get("terminated"),
                step.get("truncated"),
                step.get("termination_reason"),
            )
            for step in record.get("steps", [])
        ]
        for record in records
    ]


def _immutable_trace_signature(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Capture source trace fields that a predicate derivation may not mutate."""
    signature: list[dict[str, Any]] = []
    for record in records:
        copied = deepcopy(dict(record))
        copied.pop("success", None)
        copied.pop("first_success_step", None)
        copied.pop("terminal_distance", None)
        copied.pop("hold_windows", None)
        copied.pop("hold_success", None)
        copied.pop("relabel_status", None)
        copied.pop("relabel_missing_field_count", None)
        copied["steps"] = [
            {
                key: deepcopy(value)
                for key, value in step.items()
                if not str(key).startswith("relabelled_")
            }
            for step in copied.get("steps", [])
        ]
        signature.append(copied)
    return signature


def _assert_mapping(source: Mapping[str, Any], target_variant: str) -> None:
    mapping = (str(source.get("protocol_variant")), str(target_variant))
    if mapping not in _ALLOWED_PREDICATE_MAPPINGS:
        raise ValueError(
            f"unsupported derivation mapping {mapping[0]!r} -> {mapping[1]!r}"
        )


def derive_relabelled_result(
    source_result: Mapping[str, Any],
    *,
    manifest: CohortManifest,
    task: str,
    protocol_variant: str,
    action_block: int = 5,
    source_manifest: CohortManifest | Mapping[str, Any] | None = None,
    source_result_path: str | Path | None = None,
    source_trace_path: str | Path | None = None,
) -> dict[str, Any]:
    """Copy one result while recomputing only predicate-derived labels."""
    _assert_mapping(source_result, protocol_variant)
    if source_result.get("task") != task or manifest.task != task:
        raise ValueError("source/target task differs from requested derivation task")
    if manifest.protocol_variant != protocol_variant:
        raise ValueError("target manifest protocol differs from requested derivation variant")
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
            task=task,
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
    records = relabel_trace(source_episodes, task=task, action_block=action_block)
    if _raw_termination_signature(records) != source_termination:
        raise ValueError("predicate derivation changed raw termination provenance")
    if _immutable_trace_signature(records) != source_immutable_trace:
        raise ValueError("predicate derivation changed immutable trace provenance")
    _validate_episode_order(records, manifest)
    validate_trace_contract(records)
    payload = deepcopy(dict(source_result))
    payload["episodes"] = records
    payload["status"] = "ok"
    payload["protocol"] = ROUND3_PROTOCOL
    payload["summary"] = summarize_episodes(records)
    payload["success_rate"] = payload["summary"]["success_rate"]
    payload["success_vector_sha256"] = sha256_json(payload["summary"]["success_vector"])
    payload["summary_sha256"] = sha256_json(payload["summary"])
    derived = _derived_from(
        source_result,
        kind="predicate_relabel",
        source_manifest=source_manifest_value,
        source_result_path=source_result_path,
        source_trace_path=source_trace_path,
    )
    derived["mapping"]["target_protocol_variant"] = protocol_variant
    payload["derived_from"] = derived
    payload.pop("runtime_success_rate_percent", None)
    for key in ("cohort", "cohort_id", "cohort_sha256", "trace_path"):
        payload.pop(key, None)
    return payload


def preflight_derived_publish(
    *,
    source_result_path: str | Path,
    source_trace_path: str | Path | None = None,
    target_result_dir: str | Path,
    target_trace_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Reject source/target collisions and any existing output before writes."""
    source = Path(source_result_path).resolve()
    result_dir = _result_directory(target_result_dir)
    trace_dir = _trace_directory(target_trace_dir)
    result_dir = result_dir.resolve()
    trace_dir = trace_dir.resolve()
    target_result = result_dir / "result.json"
    target_metrics = result_dir / "metrics.txt"
    target_trace = trace_dir / "episodes.jsonl"
    source_paths = {source}
    source_paths.add(source.parent / "metrics.txt")
    if source_trace_path is not None:
        source_paths.add(Path(source_trace_path).resolve())
    if source_paths.intersection(
        {target_result.resolve(), target_metrics.resolve(), target_trace.resolve()}
    ):
        raise ValueError("source result and derived target collide")
    targets = (target_result, target_metrics, target_trace)
    existing = [path for path in targets if path.exists()]
    existing += [path.with_suffix(path.suffix + ".tmp") for path in targets if path.with_suffix(path.suffix + ".tmp").exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite derived artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    return result_dir, trace_dir, target_trace


def _prepare_target_payload(
    payload: Mapping[str, Any],
    *,
    manifest: CohortManifest,
    protocol_variant: str,
    target_trace_path: Path,
) -> dict[str, Any]:
    result = enrich_result_payload(
        payload,
        manifest=manifest,
        protocol_variant=protocol_variant,
        trace_records=payload["episodes"],
    )
    result["trace_path"] = str(target_trace_path)
    result["trace_content_sha256"] = trace_content_sha256(result["episodes"])
    result["trace_sha256"] = result["trace_content_sha256"]
    result["trace_summary_sha256"] = trace_summary_sha256(result["summary"])
    from .round3_validation import validate_result_payload

    validate_result_payload(
        result,
        manifest=manifest,
        expected_count=len(manifest.entries),
        trace_path=target_trace_path,
        trace_records=result["episodes"],
        require_integrity_hashes=True,
    )
    validate_trace_contract(result["episodes"])
    return result


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
    """Strictly validate a source and publish one predicate-relabelled result."""
    source_file = Path(source_path).resolve()
    source = load_result(source_file)
    manifest = CohortManifest.load(manifest_path)
    if manifest.task != task:
        raise ValueError("manifest task differs from requested derivation task")
    if manifest.protocol_variant != protocol_variant:
        raise ValueError("manifest protocol differs from requested derivation variant")
    _assert_mapping(source, protocol_variant)
    source_manifest = _coerce_manifest(source.get("cohort"))
    if source_manifest.task != task:
        raise ValueError("source cohort task differs from requested derivation task")
    source_trace = _resolve_source_trace(source, source_file)
    _validate_frozen_source(source, source_manifest=source_manifest, source_trace=source_trace)
    validate_derivation_source(
        source,
        manifest=source_manifest,
        trace_path=source_trace,
        task=task,
        source_result_path=source_file,
    )
    payload = derive_relabelled_result(
        source,
        manifest=manifest,
        task=task,
        protocol_variant=protocol_variant,
        action_block=action_block,
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
        protocol_variant=protocol_variant,
        target_trace_path=target_trace,
    )
    return write_round3_result(
        candidate,
        result_dir,
        trace_records=candidate["episodes"],
        trace_output_dir=trace_dir,
    )


__all__ = [
    "derive_relabelled_result",
    "derive_protocol_result_file",
    "preflight_derived_publish",
    "validate_derivation_source",
]
