#!/usr/bin/env python3
"""Cohort-paired Phase 6.3 closed-loop evaluation using the Table 1 runtime."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cvpr_table1 as table
from scripts.round5_phase6_3 import MAIN, OUT, METHODS, preflight, source_hashes, preprocess_factory
from scripts.round5_phase6_3_combos import (
    COMBO_VERSIONS,
    entry_for,
    enter_components,
    load_final_selection,
    load_selection,
)
from source.common.cvpr_table1 import TASKS, SEEDS, evaluation_cells, identity_hash


def run(args):
    import torch
    from source.common.checkpoint import load_policy_or_model
    from source.common.eval import get_dataset
    from source.common.round5_phase6_3 import deferred_refinement_checks, fixed_condition_cache, encoder_bf16
    selection = selection_sha256 = final_selection = final_selection_sha256 = None
    if args.version in COMBO_VERSIONS:
        selection, selection_sha256 = load_selection()
        final_selection, final_selection_sha256 = load_final_selection()
    frozen = table._read_json(OUT / "frozen_config.json")
    if source_hashes() != frozen["source_hashes"]:
        raise RuntimeError("source differs from current Phase6.3 snapshot")
    refs = table._read_json(OUT / "provenance/assets_and_cohorts.json")
    for ref in refs:
        if table._sha256(Path(ref["path"])) != ref["sha256"]:
            raise RuntimeError(f"changed reference: {ref['path']}")
    config = copy.deepcopy(frozen["main_table"])
    config["resources"]["permitted_gpus"] = list(range(8))
    table._load_config = lambda: config
    table._check_dispatch_resources = preflight
    native_config = table._cell_config

    def cell_config(cell):
        cfg = native_config(cell)
        cfg.output.save_video = False
        return cfg

    table._cell_config = cell_config
    selected = [c for c in evaluation_cells() if c["family"] == "cowm" and c["id"] in METHODS
                and c["method_id"].endswith("__main")
                and (not args.task or c["task"] == args.task)
                and (not args.method or c["id"] == args.method)
                and (args.seed is None or c["evaluation_seed"] == args.seed)]
    if args.version == "E1":
        selected = [c for c in selected if c["guidance_mode"] == "post_opt_refine"]
    if args.version in COMBO_VERSIONS:
        selected = [c for c in selected
                    if entry_for(selection, args.version, c["id"])["status"] == "selected"
                    and final_selection["versions"][args.version][c["id"]]["status"] == "effective"]
    if not selected:
        raise ValueError("no applicable evaluation cells")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    script = Path(__file__).resolve()
    script_digest = table._sha256(script)
    archive = OUT / "provenance/closed_loop_code" / script_digest / script.name
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(script, archive)
    table._write_json(OUT / "closed_loop" / args.version / f"dispatch_gpu{args.gpu}.json", {
        "preflight": preflight(args.gpu, config), "source_revision": frozen["source_revision"],
        "runner_sha256": script_digest, "condition_order": [c["cell_id"] for c in selected],
        "timing_scope": "native batch planning wall time and per-episode workload allocation; not batch=1 latency",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    model = dataset = None
    current_task = None
    for cell in selected:
        task = cell["task"]
        components = []
        if args.version in COMBO_VERSIONS:
            choice = entry_for(selection, args.version, cell["id"])
            if choice["status"] != "selected":
                raise RuntimeError(f"refusing to run alias {args.version}/{cell['id']}")
            components = list(choice["components"])
        identity = {"cell": cell, "version": args.version, "components": components,
                    "combination_selection_sha256": selection_sha256,
                    "final_combination_gate_sha256": final_selection_sha256,
                    "source_revision": frozen["source_revision"],
                    "runner_sha256": script_digest,
                    "asset_sha256": [{"path": ref["path"], "sha256": ref["sha256"]}
                                     for ref in refs if ref["task"] == task and ref["kind"] != "cohort"],
                    "cohort_file_sha256": next(ref["sha256"] for ref in refs if ref["task"] == task
                                                and ref["kind"] == "cohort" and ref["seed"] == cell["evaluation_seed"])}
        digest = identity_hash(identity)
        parent = OUT / "closed_loop" / args.version / cell["cell_id"]
        attempts = sorted(parent.glob("attempt_*"))
        if any(table._result_is_complete(p, digest) for p in attempts):
            print(f"skip complete {args.version} {cell['cell_id']}", flush=True)
            continue
        preflight(args.gpu, config)
        if task != current_task:
            del model, dataset
            torch.cuda.empty_cache()
            checkpoint = MAIN / "assets/cowm" / task / "checkpoints/r4_ab_seed3072_weights_epoch_10.pt"
            loaded, _ = load_policy_or_model(str(checkpoint), cache_dir=str(ROOT / "data"))
            model = getattr(loaded, "model", loaded).eval().requires_grad_(False)
            del loaded
            cfg = cell_config(cell)
            dataset = get_dataset(cfg, cfg.eval.dataset_name)
            current_task = task
        attempt = parent / f"attempt_{len(attempts)+1:03d}"
        attempt.mkdir(parents=True, exist_ok=False)
        if args.version in COMBO_VERSIONS:
            from contextlib import ExitStack

            variants = ExitStack()
            enter_components(
                variants, components,
                preprocess=preprocess_factory,
                encoder=lambda: encoder_bf16(model),
                cache=lambda: fixed_condition_cache(model),
                deferred_checks=lambda: deferred_refinement_checks(model),
            )
            variant = variants
        else:
            variant = {"E0": lambda: nullcontext(), "E1": lambda: deferred_refinement_checks(model),
                       "E2": lambda: fixed_condition_cache(model), "E3": preprocess_factory,
                       "E4": lambda: encoder_bf16(model)}[args.version]()
        with variant:
            result = table._run_one_cell(root=MAIN, cell=cell, attempt_dir=attempt,
                identity=identity, identity_sha256=digest, gpu=args.gpu,
                models={"cowm": model}, dataset=dataset)
        result["phase6_3"] = {"identity": identity, "identity_sha256": digest,
                               "components": components,
                               "combination_selection_sha256": selection_sha256,
                               "final_combination_gate_sha256": final_selection_sha256,
                               "formal_batch1_latency": False, "devices_shared": True}
        events = result.get("planning_events", [])
        for episode in result["episodes"]:
            slot = int(episode["slot"])
            episode["episode_planning_batch_exposure_seconds"] = sum(
                float(event["wall_seconds"]) for event in events if slot in event["replan_indices"])
        table._write_json(attempt / "result.json", result)
        print(f"complete closed_loop {args.version} {cell['cell_id']} episodes={len(result['episodes'])}", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--version", choices=("E0", "E1", "E2", "E3", "E4", *COMBO_VERSIONS), required=True)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
