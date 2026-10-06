#!/usr/bin/env python3
"""Audit current-revision initial-state evidence without claiming goal completion."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.round5_phase6_3 import OUT, cells
from scripts import cvpr_table1 as table


def summarize():
    import torch
    frozen = table._read_json(OUT / "frozen_config.json")
    revision = frozen["source_revision"]
    rows = []
    for version in ("E1", "E2", "E3", "E4"):
        for cell in cells():
            if version == "E1" and cell["guidance_mode"] != "post_opt_refine":
                continue
            parent = OUT / "validate" / version / cell["cell_id"]
            result = None
            path = None
            for candidate in sorted(parent.glob("attempt_*/result.json")):
                value = table._read_json(candidate)
                if value.get("identity", {}).get("source_revision") == revision:
                    result, path = value, candidate
            row = {"version": version, "method": cell["id"], "task": cell["task"],
                   "condition_id": cell["cell_id"], "status": "missing"}
            if result is not None:
                raw_path = path.parent / "raw_parity.pt"
                raw = torch.load(raw_path, map_location="cpu", weights_only=True) if raw_path.exists() else []
                states = result.get("states", [])
                expected_keys = {"encode_pixels", "sample_actions", "cpu_action", "buffered_cpu_action"}
                if cell["mode"] == "P3":
                    expected_keys.add("get_cost_from_latents")
                steps = int(cell.get("guidance_inner_steps") or 0)
                if cell["guidance_mode"] == "post_opt_refine":
                    expected_keys.update(("gradient/0", "_latent_cost_from_clean_actions"))
                valid = (result.get("status") == "complete" and result.get("states_tested") == 50
                         and [s["state_index"] for s in states] == list(range(50))
                         and [s["state_index"] for s in raw] == list(range(50))
                         and result.get("tolerance") == {"atol": 1e-6, "rtol": 1e-5})
                for index, (state, tensors) in enumerate(zip(states, raw)):
                    valid &= state["rng_seed"] == 910000 + index * 5
                    for side in ("baseline", "variant"):
                        valid &= expected_keys.issubset(tensors[side])
                        if "gradient/0" in expected_keys:
                            valid &= len(tensors[side]["gradient/0"]) == steps
                        valid &= all(torch.isfinite(t).all().item() for ts in tensors[side].values() for t in ts)
                numerical_equal = bool(result.get("all_equal"))
                row.update({"status": "complete" if valid else "invalid_evidence",
                            "initial_parity_passed": numerical_equal if version != "E4" else None,
                            "precision_change": version == "E4", "states": len(states),
                            "selection_changes": sum(not s["selection_equal"] for s in states),
                            "result": str(path.relative_to(ROOT)),
                            "raw_parity_sha256": table._sha256(raw_path) if raw_path.exists() else None,
                            "numeric_differences": {}})
                for name in sorted(expected_keys):
                    differences = [s["differences"].get(name, {}) for s in states]
                    row["numeric_differences"][name] = {
                        "max_abs_error": max((d.get("max_abs_error", float("inf")) for d in differences), default=None),
                        "mean_state_mean_abs_error": (sum(d.get("mean_abs_error", float("inf")) for d in differences)/len(differences)
                                                      if differences else None),
                        "states_within_tolerance": sum(d.get("equal", False) for d in differences)}
                if version != "E4" and not numerical_equal:
                    row["status"] = "parity_failed"
            rows.append(row)
    summary = {"source_revision": revision, "scope": "seed42 initial states; no closed-loop or timing claim",
               "conditions_target": len(rows),
               "conditions_complete": sum(r["status"] == "complete" for r in rows),
               "conditions": rows}
    table._write_json(OUT / "summary/initial_state_validation.json", summary)
    print(json.dumps({"target": len(rows), "complete": summary["conditions_complete"],
                      "versions": {v: sum(r["status"] == "complete" and r["version"] == v for r in rows)
                                   for v in ("E1", "E2", "E3", "E4")}}))


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("read-only CPU audit requires CUDA_VISIBLE_DEVICES=''")
    summarize()
