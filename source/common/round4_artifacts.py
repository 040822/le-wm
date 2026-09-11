"""Immutable artifact checks used before Round 4 LeFlow evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        while True:
            block = stream.read(int(chunk_size))
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_artifact_manifest(
    manifest: Mapping[str, Any] | str | Path,
    *,
    require_files: bool = True,
    required_tasks: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Validate planner files, dependency references, and declared SHA256s.

    The manifest intentionally does not guess checkpoint locations.  A missing
    file is reported as an explicit preflight failure instead of silently
    falling back to another task or checkpoint.
    """
    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest_path = path.parent
    else:
        payload = dict(manifest)
        manifest_path = Path(str(payload.get("root", Path.cwd())))
    tasks = payload.get("tasks")
    if not isinstance(tasks, Mapping):
        raise ValueError("Round 4 artifact manifest must contain a tasks mapping")
    checked = {}
    errors = []
    if required_tasks is not None:
        for task in required_tasks:
            if task not in tasks:
                errors.append(f"missing required task {task}")
    for task, item in tasks.items():
        if not isinstance(item, Mapping):
            errors.append(f"{task}: task entry must be an object")
            continue
        task_result = {}
        for name in ("planner_checkpoint", "lewm_checkpoint", "config"):
            value = item.get(name)
            declared = item.get(f"{name}_sha256")
            if value is None:
                errors.append(f"{task}: missing {name}")
                continue
            target = Path(str(value)).expanduser()
            if not target.is_absolute():
                target = manifest_path / target
            task_result[name] = {"path": str(target), "sha256": declared}
            if require_files and not target.is_file():
                errors.append(f"{task}: missing {name} at {target}")
                continue
            if declared is None:
                errors.append(f"{task}: missing declared {name}_sha256")
                continue
            actual = sha256_file(target)
            task_result[name]["actual_sha256"] = actual
            if actual != str(declared):
                errors.append(
                    f"{task}: {name} SHA256 mismatch; expected {declared}, got {actual}"
                )
        checked[str(task)] = task_result
    if errors:
        raise ValueError("Round 4 artifact manifest rejected:\n" + "\n".join(errors))
    return {"status": "accepted", "tasks": checked}


def validate_leflow_manifest(
    manifest: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Validate the four LeFlow payloads and their frozen planner contract."""
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload.setdefault("root", str(manifest_path.parent))
    else:
        payload = dict(manifest)
    checked = validate_artifact_manifest(
        payload,
        require_files=True,
        required_tasks=("cube", "pusht", "reacher", "tworoom"),
    )
    expected_action_dims = {"cube": 25, "pusht": 10, "reacher": 10, "tworoom": 10}
    reports: dict[str, Any] = {}
    for task, files in checked["tasks"].items():
        item = payload["tasks"][task]
        planner_path = Path(files["planner_checkpoint"]["path"])
        config_path = Path(files["config"]["path"])
        planner = torch.load(planner_path, map_location="cpu", weights_only=False)
        if not isinstance(planner, Mapping):
            raise ValueError(f"{task}: planner checkpoint is not a mapping payload")
        required = {
            "arch",
            "flow_state_dict",
            "inverse_dynamics_state_dict",
            "lewm_checkpoint",
        }
        missing = sorted(required - set(planner))
        if missing:
            raise ValueError(f"{task}: planner payload missing keys {missing}")
        arch = planner["arch"]
        inverse = arch.get("inverse_dynamics", {})
        if int(planner.get("action_block", -1)) != 5:
            raise ValueError(f"{task}: planner action_block must be 5")
        if int(inverse.get("action_dim", -1)) != expected_action_dims[task]:
            raise ValueError(
                f"{task}: inverse dynamics action_dim must be "
                f"{expected_action_dims[task]}"
            )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        planner_config = config.get("planner", {})
        if {
            "horizon": int(planner_config.get("horizon", -1)),
            "action_block": int(planner_config.get("action_block", -1)),
            "num_samples_eval": int(planner_config.get("num_samples_eval", -1)),
            "flow_steps_eval": int(planner_config.get("flow_steps_eval", -1)),
        } != {
            "horizon": 5,
            "action_block": 5,
            "num_samples_eval": 64,
            "flow_steps_eval": 16,
        }:
            raise ValueError(f"{task}: LeFlow planner config drifts from Round 4 contract")
        expected_reference = str(item.get("lewm_reference", planner["lewm_checkpoint"]))
        if str(planner["lewm_checkpoint"]) != expected_reference:
            raise ValueError(f"{task}: planner LeWM dependency reference mismatch")
        reports[task] = {
            "planner_checkpoint": files["planner_checkpoint"],
            "lewm_checkpoint": files["lewm_checkpoint"],
            "config": files["config"],
            "lewm_reference": str(planner["lewm_checkpoint"]),
            "action_block": 5,
            "action_dim": int(inverse["action_dim"]),
            "path_contract": payload.get("path_contract"),
            "reranking_contract": payload.get("reranking_contract"),
            "action_normalization": payload.get("action_normalization"),
        }
    if payload.get("protocol") != "round3_revised":
        raise ValueError("LeFlow artifact manifest must bind to round3_revised")
    for key in ("path_contract", "reranking_contract", "action_normalization"):
        if not payload.get(key):
            raise ValueError(f"LeFlow artifact manifest is missing {key}")
    return {"status": "accepted", "protocol": payload["protocol"], "tasks": reports}


__all__ = ["sha256_file", "validate_artifact_manifest", "validate_leflow_manifest"]
