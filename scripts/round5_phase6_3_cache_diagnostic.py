#!/usr/bin/env python3
"""Non-formal diagnostic of E2 projection cold misses, hits, and invalidations."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cvpr_table1 as table
from scripts import round5_phase6_3 as phase
from source.common.round5_phase6_3 import FixedProjectionCache, fixed_condition_cache


def _stats(values):
    import numpy as np
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {"count": int(array.size), "mean_ms": float(array.mean()),
            "p50_ms": float(np.percentile(array, 50)),
            "p95_ms": float(np.percentile(array, 95))}


def run(args):
    import numpy as np
    import torch
    from source.common.eval import get_dataset
    from source.common.round3_phase1 import CohortManifest

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise RuntimeError(f"launch with CUDA_VISIBLE_DEVICES={args.gpu}")
    frozen = table._read_json(phase.OUT / "frozen_config.json")
    if phase.source_hashes() != frozen["source_hashes"]:
        raise RuntimeError("current Phase 6.3 source differs from its frozen snapshot")
    cell = next(c for c in phase.cells() if c["task"] == args.task and c["id"] == args.method)
    references = table._read_json(phase.OUT / "provenance/assets_and_cohorts.json")
    selected_refs = [r for r in references if r["task"] == args.task and
                     (r["kind"] != "cohort" or r["seed"] == 42)]
    for ref in selected_refs:
        if table._sha256(Path(ref["path"])) != ref["sha256"]:
            raise RuntimeError(f"frozen input changed: {ref['path']}")

    config = copy.deepcopy(frozen["main_table"])
    config["timing"]["gpu"] = args.gpu
    config["resources"]["permitted_gpus"] = list(range(8))
    config["outputs"]["root"] = str(phase.OUT)
    prior_config = table._load_config
    prior_preflight = table._check_dispatch_resources
    prior_idle = table._check_timing_idle
    table._load_config = lambda: config
    table._check_dispatch_resources = phase.preflight
    resource = phase.preflight(args.gpu, config)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    model = dataset = None
    projection_records = []
    clear_records = []
    cuda_events = []
    original_project = FixedProjectionCache.project
    original_clear = FixedProjectionCache.clear

    def measured_project(cache, name, original, value):
        started = time.perf_counter_ns()
        before_hits, before_misses = cache.hits, cache.misses
        pending = []

        def measured_original(tensor):
            if tensor.device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                output = original(tensor)
                end_event.record()
                pending.append((start_event, end_event))
                return output
            return original(tensor)

        output = original_project(cache, name, measured_original, value)
        if cache.hits > before_hits:
            kind = "hit"
        elif cache.misses > before_misses:
            kind = "miss"
        else:
            kind = "bypass"
        row = {"projector": name, "kind": kind,
               "cpu_call_ms": (time.perf_counter_ns() - started) / 1_000_000}
        if pending:
            row["cuda_event_index"] = len(cuda_events)
            cuda_events.extend((row, start, end) for start, end in pending)
        projection_records.append(row)
        return output

    def measured_clear(cache):
        started = time.perf_counter_ns()
        entries_cleared = len(cache.values)
        original_clear(cache)
        clear_records.append({"cpu_call_ms": (time.perf_counter_ns() - started) / 1_000_000,
                              "entries_cleared": entries_cleared})

    script_sha = table._sha256(Path(__file__))
    identity = {"cell": cell, "version": "E2", "diagnostic": "projection cache miss/hit",
                "source_revision": frozen["source_revision"], "diagnostic_script_sha256": script_sha,
                "assets": [{"path": r["path"], "sha256": r["sha256"]} for r in selected_refs],
                "device": {"gpu": args.gpu, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}}
    identity_sha = table.identity_hash(identity)
    parent = phase.OUT / "cache_diagnostic" / cell["cell_id"]
    attempts = sorted(parent.glob("attempt_*")) if parent.exists() else []
    attempt = parent / f"attempt_{len(attempts)+1:03d}"

    try:
        loaded = table._load_models(phase.MAIN, args.task, int(cell["stage"]), "cuda:0")
        model = loaded["cowm"].eval().requires_grad_(False)
        cfg = table._cell_config(cell)
        cfg.output.save_video = False
        dataset = get_dataset(cfg, cfg.eval.dataset_name)
        manifest = CohortManifest.load(phase.MAIN / "cohorts" / args.task / "seed_42.json")
        table._check_timing_idle = lambda gpu, _config: {
            **phase.preflight(gpu, config), "diagnostic_shared_device": True,
            "other_compute_processes": "not required to be idle for this non-formal diagnostic"}
        with patch.object(FixedProjectionCache, "project", measured_project), \
             patch.object(FixedProjectionCache, "clear", measured_clear), \
             fixed_condition_cache(model):
            result = table._capture_and_measure_condition(
                root=phase.OUT, cell=cell, identity_sha256=identity_sha,
                config=config, model=model, dataset=dataset, manifest=manifest,
                attempt_dir=attempt)
        torch.cuda.synchronize(device=0)
        for row, start_event, end_event in cuda_events:
            row["cuda_projection_ms"] = float(start_event.elapsed_time(end_event))

        grouped = {}
        for name in ("z_condition", "latent_input"):
            grouped[name] = {}
            for kind in ("miss", "hit", "bypass"):
                selected = [r for r in projection_records if r["projector"] == name and r["kind"] == kind]
                grouped[name][kind] = {
                    "cpu_call": _stats([r["cpu_call_ms"] for r in selected]),
                    "cuda_projection": _stats([r["cuda_projection_ms"] for r in selected
                                                if "cuda_projection_ms" in r]),
                }
        summary = {
            "formal_latency": False,
            "scope": "instrumented E2 cache diagnostic; timings are not policy latency",
            "cell": cell, "identity": identity, "identity_sha256": identity_sha,
            "gpu_preflight": resource,
            "measurement": {"unique_states": result["unique_states"],
                            "repeats_per_state": result["repeats_per_state"],
                            "full_policy_call_ms_excluded": True},
            "projection_costs": grouped,
            "cache_clear_cpu_call": _stats([r["cpu_call_ms"] for r in clear_records]),
            "cache_counters": {name: sum(r["projector"] == name and r["kind"] == "hit"
                                          for r in projection_records)
                               for name in ("z_condition", "latent_input")},
            "projection_records": len(projection_records),
            "clear_count": len(clear_records),
        }
        table._write_json(attempt / "cache_costs.json", summary)
        with (attempt / "cache_projection_events.jsonl").open("w", encoding="utf-8") as stream:
            for row in projection_records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        result.update({"formal_latency": False, "cache_diagnostic": summary})
        table._write_json(attempt / "result.json", result)
        print(json.dumps({"attempt": str(attempt.relative_to(ROOT)),
                          "hit_counts": summary["cache_counters"],
                          "projection_costs": grouped,
                          "clear_count": len(clear_records)}), flush=True)
    except BaseException as exc:
        attempt.mkdir(parents=True, exist_ok=True)
        table._write_json(attempt / "failure.json", {"error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        table._load_config = prior_config
        table._check_dispatch_resources = prior_preflight
        table._check_timing_idle = prior_idle
        if model is not None:
            del model
        if dataset is not None:
            del dataset
        if "loaded" in locals():
            del loaded
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--task", choices=("tworoom", "pusht", "reacher", "cube"), default="tworoom")
    parser.add_argument("--method", choices=phase.METHODS, default="P3-PO-refine-H")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
