#!/usr/bin/env python3
"""Isolated Phase 6.3 provenance, baseline timing, and component profiling.

Table 1 assets/cohorts are referenced read-only. Formal timing deliberately
refuses GPUs with other compute processes, including apparently idle jobs.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cvpr_table1 as table
from source.common.cvpr_table1 import SEEDS, TASKS, timing_conditions

MAIN = ROOT / "outputs/cvpr/table1/v1"
OUT = ROOT / "outputs/round5/phase6_3"
METHODS = ("P0", "P3", "P3-PO-refine-L", "P3-PO-refine-H")
VERSIONS = {
    "E0": {"refinement_deferred_checks": False, "condition_cache": False,
           "preprocess_optimized": False, "encoder_bf16": False},
    "E1": {"refinement_deferred_checks": True},
    "E2": {"condition_cache": True, "cached_projections": ["z_condition", "latent_input"],
           "invalidation": "clear before every encoder invocation", "timestep_adaln": "native"},
    "E3": {"preprocess_optimized": True, "implementation": "native CPU image transform vectorization",
           "h2d": "native pageable blocking", "cpu_action_return": "native"},
    "E4": {"encoder_bf16": True},
}


def cells():
    return [c for c in timing_conditions() if c["family"] == "cowm"
            and c["id"] in METHODS and c["method_id"].endswith("__main")]


def source_hashes():
    paths = set(table.SOURCE_SNAPSHOT_PATHS)
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "source").rglob("*.py"))
    paths.add("scripts/round5_phase6_3.py")
    return {p: table._sha256(ROOT / p) for p in sorted(paths) if (ROOT / p).is_file()}


def prepare():
    from source.common.round3_phase1 import CohortManifest
    config = table._read_json(MAIN / "frozen_config.json")
    assets = table._read_json(MAIN / "provenance/assets_manifest.json")
    diagnostics = table._read_json(MAIN / "cohorts/cohort_diagnostics.json")
    references = []
    for task in TASKS:
        for key in ("cowm_checkpoint", "cowm_config"):
            record = assets["tasks"][task][key]
            path = table._repo_path(record["archive"])
            digest = table._sha256(path)
            if digest != record["archive_sha256"] or digest != record["source_sha256"]:
                raise RuntimeError(f"asset identity mismatch: {path}")
            references.append({"kind": key, "task": task, "path": str(path), "sha256": digest})
        for seed in SEEDS:
            path = MAIN / "cohorts" / task / f"seed_{seed}.json"
            cohort = CohortManifest.load(path)
            expected = diagnostics["task_seed_cohort_sha256"][task][str(seed)]
            if cohort.computed_sha256 != expected or len(cohort.entries) != 50:
                raise RuntimeError(f"cohort identity mismatch: {path}")
            references.append({"kind": "cohort", "task": task, "seed": seed,
                               "path": str(path), "sha256": table._sha256(path),
                               "cohort_sha256": cohort.computed_sha256})
    hashes = source_hashes()
    revision = table._code_revision_id(hashes)
    snapshot = OUT / "provenance/code_snapshots" / revision
    for relative in hashes:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    versions = {}
    for package in ("torch", "numpy", "transformers", "gymnasium", "omegaconf"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    table._write_json(OUT / "frozen_config.json", {
        "main_table": config, "methods": METHODS, "versions": VERSIONS,
        "combinations": "select only after measured correctness and benefit; aliases are not rerun",
        "source_revision": revision, "source_hashes": hashes,
        "main_table_frozen_config_sha256": table._sha256(MAIN / "frozen_config.json"),
        "precision": "FP32 except E4 encoder/projector BF16 with FP32 output",
        "formal_version_order": ["E0", "E1", "E0", "E2", "E0", "E3", "E0", "E4"],
        "formal_state_order": "state ascending, repeat ascending within each independent version",
        "E1_not_applicable": ["P0", "P3"],
    })
    table._write_json(OUT / "provenance/assets_and_cohorts.json", references)
    table._write_json(OUT / "provenance/environment.json", {
        "python": sys.version, "platform": platform.platform(), "packages": versions,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
    })
    print(json.dumps({"verified_references": len(references), "revision": revision}), flush=True)


def preflight(gpu, config):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError(f"requires CUDA_VISIBLE_DEVICES={gpu}")
    if gpu not in range(8):
        raise ValueError("GPU must be 0..7")
    output = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=memory.free,memory.total,utilization.gpu,uuid,name,driver_version",
        "--format=csv,noheader,nounits"], text=True).strip()
    f = [s.strip() for s in output.split(",")]
    free = int(f[0]) / 1024
    # At least 6 GiB remains after the anticipated 4 GiB complete-call peak.
    if free < 10:
        raise RuntimeError(f"requires >=10 GiB free before launch, got {free}")
    if table._free_disk_gib(ROOT) < config["resources"]["stop_dispatch_free_disk_gib"]:
        raise RuntimeError("insufficient disk space")
    return {"gpu": gpu, "free_vram_gib": free, "total_vram_gib": int(f[1])/1024,
            "gpu_utilization_percent": int(f[2]), "gpu_uuid": f[3],
            "gpu_name": f[4], "driver_version": f[5], "free_disk_gib": table._free_disk_gib(ROOT)}


def validate_capture(policy, info, *, model, version, attempt, identity):
    import numpy as np
    import torch
    from unittest.mock import patch
    from source.common.round5_phase6_3 import deferred_refinement_checks, encoder_bf16, fixed_condition_cache, preprocess_optimized
    proxy = table._SingleSlotTimingEnv(policy.env.single_action_space)
    slots = policy.env.num_envs
    input_info = table._cpu_copy(info)
    table._set_timing_mode(policy)

    def record_run(slot, seed, context):
        captured = {}

        def append(name, value):
            if torch.is_tensor(value):
                captured.setdefault(name, []).append(value.detach().float().cpu().clone())
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    append(f"{name}/{index}", item)
            elif isinstance(value, dict):
                for key, item in value.items():
                    append(f"{name}/{key}", item)

        def wrap(name, original):
            def call(*args, **kwargs):
                result = original(*args, **kwargs)
                append(name, result)
                return result
            return call

        with context as variant_state, ExitStack() as stack:
            for name in ("encode_pixels", "sample_actions", "get_cost_from_latents",
                         "_latent_cost_from_clean_actions"):
                stack.enter_context(patch.object(model, name, wrap(name, getattr(model, name))))
            stack.enter_context(patch("torch.autograd.grad", wrap("gradient", torch.autograd.grad)))
            action, _, _ = table._timed_policy_call(policy, slot, seed, proxy)
            buffer_input = copy.deepcopy(slot)
            if buffer_input.get("_needs_flush") is not None:
                buffer_input["_needs_flush"] = np.zeros(proxy.num_envs, dtype=bool)
            counts = {key: len(value) for key, value in captured.items()}
            buffered_action = table._cpu_action(policy.get_action(buffer_input))
            if counts != {key: len(value) for key, value in captured.items()}:
                raise RuntimeError("buffered action call unexpectedly ran model computation")
        captured["cpu_action"] = [torch.as_tensor(np.array(action, copy=True)).float()]
        captured["buffered_cpu_action"] = [torch.as_tensor(np.array(buffered_action, copy=True)).float()]
        selection = table._policy_selection(policy, timing_mode=True)
        statistics = ({"hits": variant_state.hits, "misses": variant_state.misses,
                       "replans": variant_state.replans}
                      if hasattr(variant_state, "hits") else None)
        return captured, selection, statistics

    states = []
    raw = []
    for index in range(slots):
        slot = table._info_slot(input_info, index, slots)
        seed = 910000 + index * 5
        baseline, base_selection, _ = record_run(slot, seed, nullcontext())
        context = (preprocess_optimized(policy) if version == "E3" else
                   {"E1": deferred_refinement_checks, "E2": fixed_condition_cache,
                    "E4": encoder_bf16}[version](model))
        optimized, opt_selection, statistics = record_run(slot, seed, context)
        differences = {}
        for name in sorted(set(baseline) | set(optimized)):
            left, right = baseline.get(name, []), optimized.get(name, [])
            comparable = len(left) == len(right) and all(a.shape == b.shape for a, b in zip(left, right))
            if not comparable:
                differences[name] = {"equal": False, "shape_mismatch": True}
                continue
            errors = torch.cat([(a-b).abs().flatten() for a, b in zip(left, right)])
            differences[name] = {"equal": all(torch.allclose(a, b, atol=1e-6, rtol=1e-5)
                                            for a, b in zip(left, right)),
                                 "max_abs_error": errors.max().item(), "mean_abs_error": errors.mean().item()}
        row = {"state_index": index, "rng_seed": seed, "differences": differences,
               "selection_equal": base_selection == opt_selection,
               "baseline_selection": base_selection, "variant_selection": opt_selection,
               "cache_statistics": statistics}
        row["equal"] = row["selection_equal"] and all(d["equal"] for d in differences.values())
        states.append(row)
        raw.append({"state_index": index, "baseline": baseline, "variant": optimized})
        table._write_json(attempt / "parity_progress.json", {"states": states})
    torch.save(raw, attempt / "raw_parity.pt")
    result = {"status": "complete", "formal_latency": False, "identity": identity,
              "tolerance": {"atol": 1e-6, "rtol": 1e-5}, "states": states,
              "states_tested": slots, "all_equal": all(s["equal"] for s in states),
              "bf16_is_precision_change": version == "E4"}
    if version in {"E1", "E2", "E3"} and not result["all_equal"]:
        table._write_json(attempt / "failed_parity.json", result)
        raise RuntimeError(f"{version} behavior parity failed; see failed_parity.json")
    return result


def component_scopes(policy, model):
    """Profiler-only labels; no stage synchronization or formal timing reuse."""
    import torch
    from unittest.mock import patch
    stack = ExitStack()

    def labeled(label, original):
        def call(*args, **kwargs):
            with torch.profiler.record_function("phase6_3/" + label):
                return original(*args, **kwargs)
        return call

    for owner, name, label in (
            (policy, "_prepare_info", "input_transform"),
            (policy, "_prepare", "input_transform_and_h2d"),
            (model, "encode_pixels", "encoder_projector"),
            (model, "sample_actions", "A_generation"),
            (policy, "score_candidates", "B_scoring"),
            (model, "get_cost_from_latents", "B_cost"),
            (model, "post_optimize_actions", "refinement"),
            (model, "_latent_cost_from_clean_actions", "refinement_forward"),
            (model, "_normalize_guidance_gradient", "refinement_normalize"),
            (model, "_clip_rms_displacement", "refinement_rms_constraint"),
            (torch.autograd, "grad", "refinement_backward"),
            (table, "_cpu_action", "cpu_return")):
        if hasattr(owner, name):
            stack.enter_context(patch.object(owner, name, labeled(label, getattr(owner, name))))
    return stack


@contextmanager
def preprocess_factory():
    import source.common.round4_eval as evaluation
    from unittest.mock import patch
    from source.common.round5_phase6_3 import preprocess_optimized
    original = evaluation.make_round4_policy
    with ExitStack() as stack:
        def factory(*args, **kwargs):
            policy = original(*args, **kwargs)
            stack.enter_context(preprocess_optimized(policy))
            return policy
        with patch.object(evaluation, "make_round4_policy", factory):
            yield


def run(args):
    import torch
    from source.common.checkpoint import load_policy_or_model
    from source.common.eval import get_dataset
    from source.common.round3_phase1 import CohortManifest
    from source.common.round5_phase6_3 import deferred_refinement_checks, encoder_bf16, fixed_condition_cache
    frozen = table._read_json(OUT / "frozen_config.json")
    if not frozen:
        raise RuntimeError("run prepare first")
    if source_hashes() != frozen["source_hashes"]:
        raise RuntimeError("source changed: refresh provenance before running")
    for ref in table._read_json(OUT / "provenance/assets_and_cohorts.json"):
        if table._sha256(Path(ref["path"])) != ref["sha256"]:
            raise RuntimeError(f"asset/cohort changed: {ref['path']}")
    config = copy.deepcopy(frozen["main_table"])
    config["timing"]["gpu"] = args.gpu
    config["resources"]["permitted_gpus"] = list(range(8))
    table._load_config = lambda: config
    table._check_dispatch_resources = preflight
    resource = preflight(args.gpu, config)
    if args.command == "timing":
        resource = table._check_timing_idle(args.gpu, config)
    selected = [c for c in cells() if (not args.task or c["task"] == args.task)
                and (not args.method or c["id"] == args.method)]
    if args.version == "E1":
        selected = [c for c in selected if c["guidance_mode"] == "post_opt_refine"]
    if not selected:
        raise ValueError("no applicable cells selected")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    table._write_json(OUT / args.command / "device.json", {
        "preflight": resource, "historical_gpu": 6, "selected_gpu": args.gpu,
        "history_is_reference_only": True, "formal": args.command == "timing"})
    for cell in selected:
        preflight(args.gpu, config)
        if args.command == "timing":
            table._check_timing_idle(args.gpu, config)
        task = cell["task"]
        checkpoint = MAIN / "assets/cowm" / task / "checkpoints/r4_ab_seed3072_weights_epoch_10.pt"
        loaded, _ = load_policy_or_model(str(checkpoint), cache_dir=str(ROOT / "data"))
        model = getattr(loaded, "model", loaded).eval().requires_grad_(False)
        cfg = table._cell_config(cell)
        dataset = get_dataset(cfg, cfg.eval.dataset_name)
        manifest = CohortManifest.load(MAIN / "cohorts" / task / "seed_42.json")
        parent = OUT / args.command / args.version / cell["cell_id"]
        attempts = sorted(parent.glob("attempt_*"))
        attempt = parent / f"attempt_{len(attempts)+1:03d}"
        identity = {"cell": cell, "version": args.version, "source_revision": frozen["source_revision"]}
        if args.command in {"profile", "validate"}:
            import source.common.round4_eval as evaluation
            original = evaluation.run_round4_evaluation

            def profile_eval(*positional, **kwargs):
                def capture(policy, info):
                    if args.command == "validate":
                        return validate_capture(policy, info, model=model, version=args.version,
                                                attempt=attempt, identity=identity)
                    proxy = table._SingleSlotTimingEnv(policy.env.single_action_space)
                    slot = table._info_slot(table._cpu_copy(info), 0, policy.env.num_envs)
                    table._set_timing_mode(policy)
                    for index in range(10):
                        table._timed_policy_call(policy, slot, 960000+index, proxy)
                    with component_scopes(policy, model), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA], record_shapes=True,
                            profile_memory=True, with_stack=False) as prof:
                        for index in range(5):
                            table._timed_policy_call(policy, slot, 910000+index, proxy)
                            prof.step()
                    prof.export_chrome_trace(str(attempt / "timeline.json"))
                    (attempt / "operators.txt").write_text(prof.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=80))
                    table._write_json(attempt / "operators.json", [
                        {"key": e.key, "count": e.count, "self_cpu_time_us": e.self_cpu_time_total,
                         "self_device_time_us": e.self_device_time_total} for e in prof.key_averages()])
                    return {"status": "complete", "formal_latency": False, "profiled_calls": 5,
                            "warmups": 10, "identity": identity}
                kwargs["timing_capture_callback"] = capture
                return original(*positional, **kwargs)
            evaluation.run_round4_evaluation = profile_eval
        variant = {"E0": lambda: nullcontext(), "E1": lambda: deferred_refinement_checks(model),
                   "E2": lambda: fixed_condition_cache(model),
                   "E3": preprocess_factory,
                   "E4": lambda: encoder_bf16(model)}[args.version]
        try:
            with (nullcontext() if args.command == "validate" else variant()):
                table._capture_and_measure_condition(root=OUT, cell=cell,
                    identity_sha256=table.identity_hash(identity), config=config,
                    model=model, dataset=dataset, manifest=manifest, attempt_dir=attempt)
        finally:
            if args.command in {"profile", "validate"}:
                evaluation.run_round4_evaluation = original
        print(f"complete {args.command} {args.version} {cell['cell_id']}", flush=True)
        del model, loaded, dataset
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("prepare", "timing", "profile", "validate"))
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--version", choices=("E0", "E1", "E2", "E3", "E4"), default="E0")
    args = parser.parse_args()
    if args.command == "validate" and args.version == "E0":
        parser.error("validate requires an experimental version")
    if args.command == "prepare":
        prepare()
    else:
        run(args)


if __name__ == "__main__":
    main()
