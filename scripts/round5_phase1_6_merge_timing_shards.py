#!/usr/bin/env python3
"""Merge completed, task-isolated Phase1.6 timing summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/round5/phase1_6.json")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = _load(config_path)
    expected_config_hash = _sha256(config_path)
    output_root = (ROOT / config["paths"]["output_root"]).resolve()
    timing_dir = output_root / "timing"

    methods = [str(item["id"]) for item in config["evaluation"]["methods"]]
    inference_seed = int(config["evaluation"]["seed"])
    expected_timing_script_hash = _sha256(ROOT / "scripts/round5_phase1_6_timing.py")
    conditions: dict[tuple[str, str, int], dict[str, Any]] = {}
    shards: dict[str, dict[str, Any]] = {}
    timing_code: dict[str, Any] | None = None
    thread_policy: dict[str, Any] | None = None
    for task in config["evaluation"]["tasks"]:
        shard_path = timing_dir / "shards" / f"{task}.json"
        if not shard_path.is_file():
            raise FileNotFoundError(f"missing timing shard: {shard_path}")
        shard = _load(shard_path)
        if shard.get("schema_version") != "round5_phase1_6_timing_v1":
            raise ValueError(f"unexpected shard schema: {shard_path}")
        if shard.get("config_sha256") != expected_config_hash:
            raise ValueError(f"config hash mismatch in {shard_path}")
        gpu_id = int(shard["gpu"])
        shard_conditions = shard.get("conditions", [])
        shard_keys = set()
        for condition in shard_conditions:
            if str(condition.get("task")) != task:
                raise ValueError(f"task mismatch in {shard_path}: {condition.get('task')}")
            key = (task, str(condition["method"]), int(condition["seed"]))
            if key in conditions or key in shard_keys:
                raise ValueError(f"duplicate timing condition: {key}")
            shard_keys.add(key)
            conditions[key] = condition
            current_code = condition.get("timing_code")
            if current_code.get("script_sha256") != expected_timing_script_hash:
                raise ValueError(f"timing script hash mismatch in {shard_path}")
            if timing_code is None:
                timing_code = current_code
            elif current_code != timing_code:
                raise ValueError(f"timing code identity differs in {shard_path}")
        current_thread_policy = shard.get("cpu_thread_policy")
        if thread_policy is None:
            thread_policy = current_thread_policy
        elif current_thread_policy != thread_policy:
            raise ValueError(f"CPU thread policy differs in {shard_path}")
        shards[task] = {
            "summary_path": str(shard_path.relative_to(ROOT)),
            "gpu": gpu_id,
            "condition_count": len(shard_conditions),
            "pending": shard.get("pending", []),
            "host_preflight": shard.get("host_preflight"),
            "gpu_preflight": shard.get("gpu_preflight"),
        }

    expected = {
        (task, method, inference_seed)
        for task in config["evaluation"]["tasks"]
        for method in methods
    }
    pending = sorted(expected - set(conditions))
    unexpected = set(conditions) - expected
    if unexpected:
        raise ValueError(f"unexpected timing conditions: {sorted(unexpected)}")
    if pending:
        raise RuntimeError(f"timing shards are incomplete; pending={pending}")

    merged = {
        "schema_version": "round5_phase1_6_timing_v1",
        "config_sha256": expected_config_hash,
        "timing_code": timing_code,
        "cpu_thread_policy": thread_policy,
        "device_policy": "one task worker per GPU; each condition retains GPU and host boundary checks",
        "gpu_by_task": {task: shard["gpu"] for task, shard in shards.items()},
        "host_preflight_by_task": {task: shard["host_preflight"] for task, shard in shards.items()},
        "gpu_preflight_by_task": {task: shard["gpu_preflight"] for task, shard in shards.items()},
        "shards": shards,
        "conditions": [conditions[key] for key in sorted(conditions)],
        "pending": [],
    }
    target = timing_dir / "summary.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    flagged = sum(not bool(c.get("interference_eligible_for_speedup_claim")) for c in merged["conditions"])
    print(json.dumps({"summary": str(target), "conditions": len(conditions), "pending": 0, "flagged": flagged}, sort_keys=True))


if __name__ == "__main__":
    main()
