"""Strict integrity contracts for Round 3 manifests and result artifacts.

The evaluator produces two independently useful records: a frozen cohort
manifest and a result payload.  This module binds them together at the
publication boundary.  It deliberately accepts an in-memory trace so callers
can validate a result before the optional JSONL file is written.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .round3_phase1 import (
    CohortManifest,
    _jsonable,
    sha256_json,
    summarize_episodes,
)
from .round3_protocol import (
    COHORT_SCHEMA_VERSION,
    PREDICATE_VERSION,
    PROTOCOL_VARIANTS,
    ROUND3_PROTOCOL,
    PHASE3_TASKS,
    TASKS,
    TRACE_SCHEMA_VERSION,
)


RESULT_SCHEMA_VERSION = 2
_COHORT_KINDS = frozenset({"dev", "final", "online", "custom"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not a finite JSON-compatible value") from exc


def _value_key(value: Any) -> bytes:
    return _canonical_bytes(value)


def _same_value(left: Any, right: Any) -> bool:
    return _value_key(left) == _value_key(right)


def success_vector_sha256(success_vector: Sequence[Any]) -> str:
    """Hash an explicitly boolean success vector in a stable representation."""
    if not isinstance(success_vector, (list, tuple)):
        raise ValueError("success vector must be a list or tuple")
    values: list[bool] = []
    for index, value in enumerate(success_vector):
        if type(value) is not bool:
            raise ValueError(f"success vector item {index} must be a boolean")
        values.append(value)
    return sha256_json(values)


def summary_sha256(summary: Mapping[str, Any]) -> str:
    """Hash the canonical episode summary without adding a self-referential field."""
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be a mapping")
    return sha256_json(summary)


def _declared_hash(payload: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return None


def _validate_summary_hashes(
    payload: Mapping[str, Any],
    expected_summary: Mapping[str, Any],
    *,
    required: bool = False,
) -> None:
    expected_vector_hash = success_vector_sha256(expected_summary["success_vector"])
    expected_summary_hash = summary_sha256(expected_summary)
    declared_vector_hash = _declared_hash(
        payload, ("success_vector_sha256", "success_vector_hash")
    )
    declared_summary_hash = _declared_hash(
        payload, ("summary_sha256", "summary_hash", "trace_summary_sha256")
    )
    if required and declared_vector_hash is None:
        raise ValueError("result success vector hash is missing")
    if required and declared_summary_hash is None:
        raise ValueError("result summary hash is missing")
    if declared_vector_hash is not None and _require_hash(
        declared_vector_hash, field="success vector hash"
    ) != expected_vector_hash:
        raise ValueError("success vector hash does not match embedded episodes")
    if declared_summary_hash is not None and _require_hash(
        declared_summary_hash, field="summary hash"
    ) != expected_summary_hash:
        raise ValueError("summary hash does not match embedded episodes")


def _normalise_task(task: Any) -> str:
    return str(task).lower().replace("push-t", "pusht").replace("two_room", "tworoom")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_hash(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} is missing or malformed")
    return value


def _coerce_manifest(value: CohortManifest | Mapping[str, Any]) -> CohortManifest:
    if isinstance(value, CohortManifest):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("manifest must be a CohortManifest or mapping")
    required = (
        "protocol",
        "schema_version",
        "task",
        "cohort_id",
        "cohort_kind",
        "protocol_variant",
        "seed",
        "goal_offset_steps",
        "entries",
        "episode_split",
        "cohort_sha256",
    )
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"manifest is missing required fields: {', '.join(missing)}")
    try:
        return CohortManifest.from_dict(value, verify_hash=True)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("manifest is not a valid Round 3 cohort") from exc


def _split_key(value: Any) -> bytes:
    return _value_key(value)


def validate_cohort_manifest(
    manifest: CohortManifest | Mapping[str, Any],
    *,
    expected_count: int | None = None,
    required_kind: str | None = None,
    task: str | None = None,
) -> CohortManifest:
    """Validate all frozen cohort identity and split-ownership invariants."""

    parsed = _coerce_manifest(manifest)
    if parsed.schema_version != COHORT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported cohort schema version {parsed.schema_version!r}; "
            f"expected {COHORT_SCHEMA_VERSION}"
        )
    if _normalise_task(parsed.task) not in (*TASKS, *PHASE3_TASKS):
        raise ValueError(f"unknown cohort task {parsed.task!r}")
    if task is not None and _normalise_task(parsed.task) != _normalise_task(task):
        raise ValueError(
            f"cohort task {parsed.task!r} does not match requested task {task!r}"
        )
    if parsed.protocol_variant not in PROTOCOL_VARIANTS:
        raise ValueError(f"unknown protocol variant {parsed.protocol_variant!r}")
    if parsed.cohort_kind not in _COHORT_KINDS:
        raise ValueError(f"unknown cohort kind {parsed.cohort_kind!r}")
    if required_kind is not None and parsed.cohort_kind != required_kind:
        raise ValueError(
            f"manifest {parsed.cohort_id} is {parsed.cohort_kind!r}; "
            f"expected {required_kind!r}"
        )
    if not isinstance(parsed.cohort_id, str) or not parsed.cohort_id:
        raise ValueError("cohort_id must be a non-empty string")
    if not _is_int(parsed.schema_version) or not _is_int(parsed.seed):
        raise ValueError("cohort schema_version and seed must be integers")
    if not _is_int(parsed.goal_offset_steps) or parsed.goal_offset_steps < 0:
        raise ValueError("cohort goal_offset_steps must be a non-negative integer")
    if expected_count is not None:
        if not _is_int(expected_count) or expected_count < 0:
            raise ValueError("expected_count must be a non-negative integer")
        if len(parsed.entries) != expected_count:
            raise ValueError(
                f"manifest {parsed.cohort_id} has {len(parsed.entries)} entries; "
                f"expected {expected_count}"
            )

    stored_hash = _require_hash(parsed.cohort_sha256, field="cohort_sha256")
    try:
        computed_hash = parsed.computed_sha256
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("cohort canonical payload is not hashable") from exc
    if stored_hash != computed_hash:
        raise ValueError(
            f"cohort SHA256 does not match its canonical payload: "
            f"stored={stored_hash!r} computed={computed_hash!r}"
        )

    entries = parsed.entries
    if not isinstance(entries, (tuple, list)):
        raise ValueError("cohort entries must be a sequence")
    rows: list[int] = []
    entry_episode_keys: list[bytes] = []
    for entry in entries:
        if not _is_int(entry.row_index) or entry.row_index < 0:
            raise ValueError("cohort row_index must be a non-negative integer")
        if not _is_int(entry.start_step) or entry.start_step < 0:
            raise ValueError("cohort start_step must be a non-negative integer")
        if entry.episode_id is None:
            raise ValueError("cohort episode_id cannot be null")
        rows.append(entry.row_index)
        entry_episode_keys.append(_split_key(entry.episode_id))
        if entry.goal_row_index is not None and (
            not _is_int(entry.goal_row_index) or entry.goal_row_index < 0
        ):
            raise ValueError("cohort goal_row_index must be a non-negative integer")
        if entry.goal_step is not None:
            if not _is_int(entry.goal_step) or entry.goal_step < 0:
                raise ValueError("cohort goal_step must be a non-negative integer")
            if entry.goal_row_index is None:
                raise ValueError("cohort goal_step requires goal_row_index")
            if entry.goal_step != entry.start_step + parsed.goal_offset_steps:
                raise ValueError("cohort goal_step does not match the frozen offset")
        for state_name in ("start_state", "goal_state"):
            state = getattr(entry, state_name)
            if state is not None:
                _canonical_bytes(state)

    if len(rows) != len(set(rows)):
        raise ValueError("cohort row indices are not unique")
    if parsed.protocol_variant != "legacy" and len(entry_episode_keys) != len(set(entry_episode_keys)):
        raise ValueError("revised cohort raw episode IDs are not unique")

    split = parsed.episode_split
    if not isinstance(split, Mapping) or not split:
        raise ValueError("episode_split must be a non-empty mapping")
    split_membership: dict[bytes, str] = {}
    for split_name, values in split.items():
        if not isinstance(split_name, str) or not split_name:
            raise ValueError("episode_split names must be non-empty strings")
        if not isinstance(values, (tuple, list)):
            raise ValueError(f"episode_split[{split_name!r}] must be a sequence")
        local_keys: set[bytes] = set()
        for episode_id in values:
            key = _split_key(episode_id)
            if key in local_keys and parsed.protocol_variant != "legacy":
                raise ValueError(f"episode_split[{split_name!r}] contains duplicate episode IDs")
            local_keys.add(key)
            if key in split_membership and split_membership[key] != split_name:
                raise ValueError(
                    f"episode ID belongs to both {split_membership[key]!r} and {split_name!r}"
                )
            split_membership[key] = split_name

    owner_name = parsed.cohort_kind
    if owner_name not in split and parsed.protocol_variant == "legacy" and "selected" in split:
        owner_name = "selected"
    if owner_name not in split:
        raise ValueError(
            f"episode_split has no ownership bucket for cohort kind {parsed.cohort_kind!r}"
        )
    owner_keys = {_split_key(item) for item in split[owner_name]}
    if parsed.protocol_variant == "legacy":
        owner_sequence = [_split_key(item) for item in split[owner_name]]
        if owner_sequence != entry_episode_keys:
            raise ValueError(
                "legacy selected episode IDs do not match entries in order "
                "and multiplicity"
            )
    for key in entry_episode_keys:
        if key not in split_membership:
            raise ValueError("cohort entry episode is absent from episode_split")
        if key not in owner_keys:
            raise ValueError(
                f"cohort entry episode is owned by {split_membership[key]!r}, "
                f"not {owner_name!r}"
            )

    return parsed


def _manifest_identity_fields(manifest: CohortManifest) -> dict[str, Any]:
    return {
        "protocol": ROUND3_PROTOCOL,
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": manifest.task,
        "cohort_id": manifest.cohort_id,
        "cohort_kind": manifest.cohort_kind,
        "protocol_variant": manifest.protocol_variant,
        "cohort_sha256": manifest.computed_sha256,
    }


def _episode_identity_error(index: int, field: str, actual: Any, expected: Any) -> ValueError:
    return ValueError(
        f"episode slot {index} field {field!r} does not match manifest: "
        f"actual={actual!r} expected={expected!r}"
    )


def _validate_episode_records(
    episodes: Sequence[Any],
    *,
    manifest: CohortManifest | None,
) -> None:
    seen_identity_keys: set[tuple[bytes, bytes, bytes]] = set()
    expected_entries = manifest.entries if manifest is not None else None
    if expected_entries is not None and len(episodes) != len(expected_entries):
        raise ValueError("result episode count does not match the frozen cohort")
    for index, record in enumerate(episodes):
        if not isinstance(record, Mapping):
            raise ValueError(f"episode slot {index} must be a mapping")
        if "slot" not in record or not _is_int(record["slot"]):
            raise ValueError(f"episode slot {index} is missing an integer slot")
        if record["slot"] != index:
            raise ValueError(
                f"episode order/slot mismatch at index {index}: "
                f"record slot={record['slot']!r}"
            )
        if "episode_id" not in record:
            raise ValueError(f"episode slot {index} is missing episode_id")
        for field in ("start_step", "row_index"):
            if field not in record or not _is_int(record[field]):
                raise ValueError(f"episode slot {index} is missing an integer {field}")
        identity = (
            _split_key(record["episode_id"]),
            _split_key(record["start_step"]),
            _split_key(record["row_index"]),
        )
        if identity in seen_identity_keys:
            raise ValueError(f"duplicate result episode identity at slot {index}")
        seen_identity_keys.add(identity)

        if manifest is None:
            if "dataset_episode" in record and not _same_value(
                record["dataset_episode"], record["episode_id"]
            ):
                raise _episode_identity_error(
                    index, "dataset_episode", record["dataset_episode"], record["episode_id"]
                )
            continue

        entry = expected_entries[index]
        required_fields = {
            "episode_id": entry.episode_id,
            "start_step": entry.start_step,
            "row_index": entry.row_index,
            "goal_row_index": entry.goal_row_index,
        }
        for field, expected in required_fields.items():
            if expected is None:
                if field in record and record[field] is not None:
                    raise _episode_identity_error(index, field, record[field], expected)
                continue
            if field not in record:
                raise ValueError(f"episode slot {index} is missing {field}")
            if not _same_value(record[field], expected):
                raise _episode_identity_error(index, field, record[field], expected)

        for field in ("goal_step", "goal_state"):
            if field in record:
                expected = getattr(entry, field)
                if expected is None:
                    if record[field] is not None:
                        raise _episode_identity_error(index, field, record[field], expected)
                elif not _same_value(record[field], expected):
                    raise _episode_identity_error(index, field, record[field], expected)

        if "dataset_episode" in record and not _same_value(
            record["dataset_episode"], entry.episode_id
        ):
            raise _episode_identity_error(
                index, "dataset_episode", record["dataset_episode"], entry.episode_id
            )

        steps = record.get("steps")
        if steps is not None:
            if not isinstance(steps, list):
                raise ValueError(f"episode slot {index} steps must be a list")
            if entry.goal_state is not None:
                for step_index, step in enumerate(steps):
                    if not isinstance(step, Mapping):
                        raise ValueError(
                            f"episode slot {index} step {step_index} must be a mapping"
                        )
                    if "goal" in step and step["goal"] is not None and not _same_value(
                        step["goal"], entry.goal_state
                    ):
                        raise ValueError(
                            f"episode slot {index} step {step_index} goal "
                            "does not match the manifest goal state"
                        )


def _trace_jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    lines: list[bytes] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"trace row {index} must be a mapping")
        try:
            line = json.dumps(
                _jsonable(record),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"trace row {index} is not JSON-compatible") from exc
        lines.append(line + b"\n")
    return b"".join(lines)


def trace_content_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Return the hash of the JSONL bytes emitted by the Phase 1 trace writer."""

    return hashlib.sha256(_trace_jsonl_bytes(records)).hexdigest()


def trace_summary_sha256(summary: Mapping[str, Any]) -> str:
    """Return the hash of the canonical episode summary."""

    return sha256_json(summary)


def _trace_metadata(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
    nested = payload.get("trace")
    return (nested if isinstance(nested, Mapping) else {}, nested)


def _first_declared(
    payload: Mapping[str, Any],
    nested: Mapping[str, Any],
    names: Sequence[str],
) -> Any:
    for source in (payload, nested):
        for name in names:
            if name in source and source[name] is not None:
                return source[name]
    return None


def _load_trace_file(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"trace JSONL cannot be read: {path}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"trace JSONL is not UTF-8: {path}") from exc
    lines = text.splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError(f"trace JSONL contains a blank row: {path}")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"trace JSONL row {index} is invalid JSON") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"trace JSONL row {index} must be a mapping")
        records.append(dict(value))
    return raw, records


def _validate_trace(
    payload: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    manifest: CohortManifest | None,
    *,
    trace_path: str | Path | None,
    trace_records: Sequence[Mapping[str, Any]] | None,
    expected_summary: Mapping[str, Any],
) -> None:
    nested, nested_value = _trace_metadata(payload)
    declared_path = trace_path
    if declared_path is None:
        declared_path = payload.get("trace_path")
    if declared_path is None and isinstance(nested_value, Mapping):
        declared_path = nested_value.get("path")

    records: list[Mapping[str, Any]] | None = (
        list(trace_records) if trace_records is not None else None
    )
    if records is None and isinstance(nested_value, list):
        records = list(nested_value)

    has_trace = (
        declared_path is not None
        or records is not None
        or bool(nested)
        or payload.get("trace_schema_version") is not None
    )
    if not has_trace:
        return

    schema_version = _first_declared(
        payload,
        nested,
        ("trace_schema_version", "schema_version"),
    )
    if schema_version is not None and schema_version != TRACE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported trace schema version {schema_version!r}; "
            f"expected {TRACE_SCHEMA_VERSION}"
        )

    raw: bytes | None = None
    file_records: list[dict[str, Any]] | None = None
    if declared_path is not None:
        path = Path(declared_path)
        try:
            raw, file_records = _load_trace_file(path)
        except ValueError:
            if records is None:
                raise
            # During pre-publication validation the writer has not created the
            # optional file yet.  The in-memory records remain authoritative.
            if path.exists():
                raise

    if records is None:
        records = file_records
    if records is None:
        raise ValueError("trace metadata is present but no trace records are available")
    _validate_episode_records(records, manifest=manifest)
    if len(records) != len(episodes):
        raise ValueError("trace row count does not match embedded episodes")
    for index, (trace_record, episode) in enumerate(zip(records, episodes)):
        if not _same_value(trace_record, episode):
            raise ValueError(f"trace row {index} differs from embedded episode")

    if file_records is not None:
        if len(file_records) != len(records):
            raise ValueError("trace JSONL row count differs from in-memory trace")
        for index, (file_record, record) in enumerate(zip(file_records, records)):
            if not _same_value(file_record, record):
                raise ValueError(f"trace JSONL row {index} differs from embedded trace")

    content_hash = hashlib.sha256(raw).hexdigest() if raw is not None else trace_content_sha256(records)
    declared_content_hash = _first_declared(
        payload,
        nested,
        (
            "trace_content_sha256",
            "trace_sha256",
            "trace_hash",
            "content_sha256",
            "content_hash",
            "sha256",
        ),
    )
    if declared_content_hash is not None:
        if _require_hash(declared_content_hash, field="trace content hash") != content_hash:
            raise ValueError("trace content hash does not match JSONL content")

    declared_summary_hash = _first_declared(
        payload,
        nested,
        (
            "trace_summary_sha256",
            "summary_sha256",
            "summary_hash",
        ),
    )
    if declared_summary_hash is not None:
        if _require_hash(declared_summary_hash, field="trace summary hash") != trace_summary_sha256(expected_summary):
            raise ValueError("trace summary hash does not match embedded episodes")

    if isinstance(nested_value, Mapping) and "summary" in nested_value:
        if not isinstance(nested_value["summary"], Mapping) or not _same_value(
            nested_value["summary"], expected_summary
        ):
            raise ValueError("trace summary does not match embedded episodes")


def validate_result_payload(
    payload: Mapping[str, Any],
    *,
    manifest: CohortManifest | Mapping[str, Any] | None = None,
    expected_count: int | None = None,
    trace_path: str | Path | None = None,
    trace_records: Sequence[Mapping[str, Any]] | None = None,
    require_integrity_hashes: bool = False,
) -> None:
    """Reject a result unless its cohort, episodes, summary, and trace agree."""

    if not isinstance(payload, Mapping):
        raise TypeError("result payload must be a mapping")
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported result schema version {payload.get('schema_version')!r}; "
            f"expected {RESULT_SCHEMA_VERSION}"
        )
    if payload.get("protocol") != ROUND3_PROTOCOL:
        raise ValueError("result does not use the Round 3 Phase 1 protocol")
    if payload.get("status") != "ok":
        raise ValueError(f"result status is not ok: {payload.get('status')!r}")
    if payload.get("predicate_version") != PREDICATE_VERSION:
        raise ValueError("result predicate version is missing or stale")

    cohort_hash = _require_hash(payload.get("cohort_sha256"), field="result cohort_sha256")
    parsed_manifest: CohortManifest | None = None
    if manifest is not None:
        parsed_manifest = validate_cohort_manifest(manifest, expected_count=expected_count)
    embedded_value = payload.get("cohort")
    embedded_manifest: CohortManifest | None = None
    if embedded_value is not None:
        embedded_manifest = validate_cohort_manifest(embedded_value)
    if parsed_manifest is None:
        parsed_manifest = embedded_manifest
    elif embedded_manifest is None:
        raise ValueError("result is missing its embedded cohort manifest")
    if parsed_manifest is not None:
        if cohort_hash != parsed_manifest.computed_sha256:
            raise ValueError("result cohort_sha256 does not match the frozen manifest")
        if embedded_manifest is not None and embedded_manifest.as_dict() != parsed_manifest.as_dict():
            raise ValueError("embedded cohort manifest differs from the supplied manifest")
        for field, expected in _manifest_identity_fields(parsed_manifest).items():
            if field == "schema_version":
                continue
            if payload.get(field) != expected:
                raise ValueError(
                    f"result field {field!r} does not match the frozen manifest"
                )
    if expected_count is not None:
        if not _is_int(expected_count) or expected_count < 0:
            raise ValueError("expected_count must be a non-negative integer")
        episodes = payload.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != expected_count:
            raise ValueError("result episode count does not match the frozen cohort")

    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("result episodes must be a list")
    _validate_episode_records(episodes, manifest=parsed_manifest)
    if not episodes:
        raise ValueError("result must contain at least one episode")

    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("result summary must be a mapping")
    try:
        expected_summary = summarize_episodes(episodes)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("result episodes cannot be summarized") from exc
    if not _same_value(summary, expected_summary):
        raise ValueError("result summary is not an exact recomputation of episodes")
    if "success_rate" in payload and not _same_value(
        payload["success_rate"], expected_summary["success_rate"]
    ):
        raise ValueError("result success_rate does not match summary")
    _validate_summary_hashes(
        payload,
        expected_summary,
        required=require_integrity_hashes,
    )

    _validate_trace(
        payload,
        episodes,
        parsed_manifest,
        trace_path=trace_path,
        trace_records=trace_records,
        expected_summary=expected_summary,
    )


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "success_vector_sha256",
    "summary_sha256",
    "trace_content_sha256",
    "trace_summary_sha256",
    "validate_cohort_manifest",
    "validate_result_payload",
]
