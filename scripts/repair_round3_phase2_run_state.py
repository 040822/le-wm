"""Repair one Round 3 Phase 2 state after an incomplete checkpoint write."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


SHARD_RE = re.compile(r"online_envsteps_(\d+)\.pt$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()

    state_path = args.state
    target = int(args.step)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    backup = state_path.with_suffix(state_path.suffix + ".before_repair")
    shutil.copy2(state_path, backup)

    def record_step(record: dict) -> int:
        if "update" in record:
            return int(record["update"]["environment_steps"])
        return int(record["environment_steps"])

    records = [record for record in payload["records"] if record_step(record) <= target]
    updates = [record["update"] for record in records if "update" in record]
    if not updates or int(updates[-1]["environment_steps"]) != target:
        raise ValueError(f"no complete update record at target step {target}")

    shards = []
    for shard in payload.get("replay_shards", []):
        match = SHARD_RE.search(str(shard["path"]))
        if match is None or int(match.group(1)) <= target:
            shards.append(shard)

    payload["status"] = "running"
    payload["current_environment_steps"] = target
    payload["pool_offset"] = int(updates[-1]["collect"]["next_pool_offset"])
    payload["records"] = records
    payload["replay_shards"] = shards
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(state_path)
    print(json.dumps({
        "state": str(state_path),
        "backup": str(backup),
        "target_step": target,
        "records": len(records),
        "replay_shards": len(shards),
        "pool_offset": payload["pool_offset"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
