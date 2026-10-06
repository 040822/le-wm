"""Validation and lookup helpers for Phase 6.3 selected combinations."""
from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

from scripts.round5_phase6_3 import METHODS, OUT

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("E1", "E2", "E3", "E4")
COMBO_VERSIONS = ("C-FP32", "C-BF16")
SELECTION_PATH = OUT / "summary/selected_combinations.json"
FINAL_SELECTION_PATH = OUT / "summary/final_combinations.json"


def validate_selection(value: dict) -> dict:
    if value.get("schema_version") != 1:
        raise ValueError("unsupported selected-combinations schema")
    versions = value.get("versions")
    if not isinstance(versions, dict) or set(versions) != set(COMBO_VERSIONS):
        raise ValueError("selection must define C-FP32 and C-BF16")

    normalized = {}
    for version in COMBO_VERSIONS:
        entries = versions[version]
        if not isinstance(entries, dict) or set(entries) != set(METHODS):
            raise ValueError(f"{version} must define every frozen method")
        normalized[version] = {}
        for method in METHODS:
            entry = entries[method]
            status = entry.get("status")
            components = entry.get("components")
            if status not in {"selected", "alias"} or not isinstance(components, list):
                raise ValueError(f"invalid {version}/{method} selection entry")
            if len(components) != len(set(components)) or any(c not in COMPONENTS for c in components):
                raise ValueError(f"invalid components for {version}/{method}")
            canonical = [c for c in COMPONENTS if c in components]
            if components != canonical:
                raise ValueError(f"components are not in canonical order for {version}/{method}")
            if method in {"P0", "P3"} and "E1" in components:
                raise ValueError(f"E1 does not apply to {method}")
            if version == "C-FP32" and "E4" in components:
                raise ValueError("C-FP32 cannot include E4")
            if status == "selected" and not components:
                raise ValueError(f"selected {version}/{method} must change at least one component")
            if version == "C-BF16" and status == "selected" and "E4" not in components:
                raise ValueError(f"selected C-BF16 must include E4 for {method}")
            if status == "selected" and len(components) == 1:
                raise ValueError(f"single-component {version}/{method} must be recorded as an alias")
            if status == "alias":
                alias_of = entry.get("alias_of")
                if alias_of not in {"E0", "E1", "E2", "E3", "E4", "C-FP32"}:
                    raise ValueError(f"invalid alias target for {version}/{method}")
                if alias_of == version:
                    raise ValueError(f"self alias in {version}/{method}")
                if version == "C-FP32":
                    if len(components) > 1:
                        raise ValueError(f"combined C-FP32 implementation cannot be an alias for {method}")
                    expected_alias = components[0] if components else "E0"
                    if alias_of != expected_alias:
                        raise ValueError(f"incorrect C-FP32 alias target for {method}")
            normalized[version][method] = {**entry, "components": components}

    for method in METHODS:
        fp32 = normalized["C-FP32"][method]
        bf16 = normalized["C-BF16"][method]
        expected = fp32["components"] + ([] if "E4" in fp32["components"] else ["E4"])
        expected = [c for c in COMPONENTS if c in expected]
        if bf16["status"] == "selected" and "E4" not in bf16["components"]:
            raise ValueError(f"selected C-BF16 must include E4 for {method}")
        if bf16["components"] not in (fp32["components"], expected):
            raise ValueError(f"C-BF16 must equal C-FP32 or add E4 for {method}")
        if bf16["status"] == "alias" and bf16["components"] == fp32["components"]:
            if bf16["alias_of"] != "C-FP32":
                raise ValueError(f"C-BF16 without E4 must alias C-FP32 for {method}")
        elif bf16["status"] == "alias" and len(bf16["components"]) == 1:
            if bf16["alias_of"] != bf16["components"][0]:
                raise ValueError(f"incorrect single-version C-BF16 alias for {method}")
        elif bf16["status"] == "alias":
            raise ValueError(f"combined C-BF16 implementation cannot be an alias for {method}")
    return {**value, "versions": normalized}


def load_selection(path: Path = SELECTION_PATH) -> tuple[dict, str]:
    if not path.is_file():
        raise FileNotFoundError(f"combination selection has not been frozen: {path}")
    raw = path.read_bytes()
    selection = validate_selection(json.loads(raw))
    current_code = {
        "selection_script_sha256": hashlib.sha256(
            (ROOT / "scripts/round5_phase6_3_select_combinations.py").read_bytes()).hexdigest(),
        "combination_helpers_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for key, digest in current_code.items():
        if selection.get(key) != digest:
            raise ValueError(f"combination selection code changed after freezing: {key}")
    return selection, hashlib.sha256(raw).hexdigest()


def load_final_selection() -> tuple[dict, str]:
    selection, selection_sha256 = load_selection()
    if not FINAL_SELECTION_PATH.is_file():
        raise FileNotFoundError(f"combination timing gate is not complete: {FINAL_SELECTION_PATH}")
    raw = FINAL_SELECTION_PATH.read_bytes()
    final = json.loads(raw)
    if final.get("schema_version") != 1 or final.get("combination_selection_sha256") != selection_sha256:
        raise ValueError("final combination gate does not match the frozen candidates")
    evidence = (
        (OUT / "summary/combo_formal_timing_analysis.json", final.get("timing_analysis_sha256")),
        (OUT / "summary/formal_timing_analysis.json", selection.get("timing_analysis_sha256")),
    )
    for path, expected in evidence:
        if not path.is_file() or not expected or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"combination evidence changed after selection: {path}")
    timing_analysis = json.loads(evidence[0][0].read_text(encoding="utf-8"))
    timing_analysis_script = ROOT / "scripts/round5_phase6_3_combo_timing_analysis.py"
    if timing_analysis.get("analysis_sha256") != hashlib.sha256(timing_analysis_script.read_bytes()).hexdigest():
        raise ValueError("combination timing analysis code changed after freezing")
    driver_sha = timing_analysis.get("timing_driver_sha256")
    if driver_sha:
        driver_archive = OUT / "provenance/combo_timing_driver" / driver_sha / "round5_phase6_3_combo_timing.py"
        if not driver_archive.is_file() or hashlib.sha256(driver_archive.read_bytes()).hexdigest() != driver_sha:
            raise ValueError("combination timing driver archive is missing or changed")
    finalizer = ROOT / "scripts/round5_phase6_3_finalize_combinations.py"
    helpers = Path(__file__)
    if (hashlib.sha256(finalizer.read_bytes()).hexdigest() != final.get("finalizer_code_sha256")
            or hashlib.sha256(helpers.read_bytes()).hexdigest() != final.get("combination_helpers_sha256")):
        raise ValueError("final combination gate code changed after freezing")
    e4_path = OUT / "summary/closed_loop_paired.json"
    if (not e4_path.is_file()
            or closed_loop_e4_fingerprint(e4_path) != selection.get("E4_closed_loop_evidence_sha256")):
        raise ValueError(f"E4 closed-loop evidence changed after selection: {e4_path}")
    versions = final.get("versions", {})
    for version in COMBO_VERSIONS:
        if set(versions.get(version, {})) != set(METHODS):
            raise ValueError(f"final gate must define every {version} method")
        for method in METHODS:
            status = versions[version][method].get("status")
            if status not in {"effective", "not_effective", "alias"}:
                raise ValueError(f"invalid final status for {version}/{method}")
            candidate_status = selection["versions"][version][method]["status"]
            if candidate_status == "alias" and status != "alias":
                raise ValueError(f"candidate alias changed status for {version}/{method}")
            if candidate_status == "selected" and status == "alias":
                raise ValueError(f"measured candidate cannot become an alias for {version}/{method}")
    return final, hashlib.sha256(raw).hexdigest()


def closed_loop_e4_fingerprint(path: Path) -> str:
    report = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "task_equal_average": report.get("task_equal_average", {}).get("E4", {}),
        "conditions": [row for row in report.get("conditions", []) if row.get("version") == "E4"],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def entry_for(selection: dict, version: str, method: str) -> dict:
    if version not in COMBO_VERSIONS:
        raise ValueError(f"not a combination version: {version}")
    return selection["versions"][version][method]


def enter_components(
    stack: ExitStack,
    components: list[str],
    *,
    preprocess,
    encoder,
    cache,
    deferred_checks,
) -> None:
    """Enter compatible component scopes; BF16 must wrap cache invalidation."""
    factories = {
        "E3": preprocess,
        "E4": encoder,
        "E2": cache,
        "E1": deferred_checks,
    }
    for component in ("E3", "E4", "E2", "E1"):
        if component in components:
            stack.enter_context(factories[component]())
