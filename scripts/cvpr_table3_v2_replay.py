#!/usr/bin/env python3
"""Replay accepted v1 action pools and retain auditable v2 endpoint traces."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.cvpr_table3_fixed_pool as pool
import scripts.round5_phase1_5_diagnostics as p15
from source.common.checkpoint import load_policy_or_model
from source.common.eval import DatasetEvaluationSession, get_dataset
from source.common.round3_phase1 import CohortManifest
from source.common.round3_protocol import TASK_PREDICATES
from scripts.round5_phase1_5 import _gpu_preflight

TASKS = pool.TASKS
SEEDS = pool.SEEDS
DEFAULT_OUTPUT = ROOT / "outputs/cvpr/table3/v2/3a"
MIN_FREE_MIB = 12_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _validate_physical_protocol(task: str, global_frozen: dict[str, Any]) -> dict[str, Any]:
    declared = global_frozen.get("task_physical_definitions", {}).get(task)
    if not isinstance(declared, dict):
        raise RuntimeError(f"frozen physical-state definition is missing for {task}")
    predicate = TASK_PREDICATES[task].as_dict()
    checks = {
        "current_field": predicate["current_field"],
        "goal_field": predicate["goal_field"],
        "field_aliases": predicate["field_aliases"],
        "goal_aliases": predicate["goal_aliases"],
        "state_unit": predicate["unit"],
        "success_predicate": predicate["formula"],
        "success_thresholds": predicate["thresholds"],
        "success_comparison": predicate["comparison"],
    }
    for key, expected in checks.items():
        actual = declared.get(key)
        if isinstance(expected, dict):
            expected = {k: float(v) for k, v in expected.items()}
            actual = {k: float(v) for k, v in actual.items()} if isinstance(actual, dict) else actual
        if actual != expected:
            raise RuntimeError(f"{task} frozen {key} differs from the runtime Round3 predicate")
    if task == "pusht":
        components = pool._physical_cost(task, [3.0, 4.0, 0.0, 0.0, math.pi / 18.0], [0.0] * 5)
        expected_cost = 0.5
    else:
        components = pool._physical_cost(task, [3.0, 4.0], [0.0, 0.0])
        expected_cost = 5.0
    if not math.isclose(float(components["selection_cost"]), expected_cost, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError(f"{task} frozen selection-cost formula failed its synthetic geometry check")
    return {
        "runtime_predicate": predicate,
        "selection_cost_formula": declared["selection_cost"],
        "selection_cost_unit": declared["selection_cost_unit"],
        "synthetic_formula_check": "pass",
    }


def _load_source(task: str, seed: int) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    source = pool.OUTPUT_ROOT / task / f"seed_{seed}"
    acceptance = _read_json(source / "acceptance.json")
    summary = _read_json(source / "summary.json")
    frozen = _read_json(source / "frozen_config.json")
    if acceptance.get("status") != "pass" or acceptance.get("errors") != []:
        raise RuntimeError(f"{task}/seed_{seed}: v1 source acceptance failed")
    if acceptance.get("summary_sha256") != _sha256(source / "summary.json"):
        raise RuntimeError(f"{task}/seed_{seed}: v1 summary hash differs")
    if acceptance.get("fixed_pool_sha256") != _sha256(source / "fixed_pool.npz"):
        raise RuntimeError(f"{task}/seed_{seed}: v1 fixed pool hash differs")
    if summary.get("fixed_pool_sha256") != _sha256(source / "fixed_pool.npz"):
        raise RuntimeError(f"{task}/seed_{seed}: v1 fixed pool provenance differs")
    data = np.load(source / "fixed_pool.npz")
    arrays = {key: data[key] for key in data.files}
    required = {"candidates", "physical_actions", "cowm_b_costs", "lewm_costs", "candidate_noise"}
    if not required.issubset(arrays):
        raise RuntimeError(f"{task}/seed_{seed}: v1 fixed pool lacks {sorted(required-set(arrays))}")
    expected_shape = (50, 64)
    if arrays["candidates"].shape[:2] != expected_shape:
        raise RuntimeError(f"{task}/seed_{seed}: candidate shape changed")
    if arrays["cowm_b_costs"].shape != expected_shape or arrays["lewm_costs"].shape != expected_shape:
        raise RuntimeError(f"{task}/seed_{seed}: scorer matrices do not match the candidate pool")
    for name in ("candidate_0000.jsonl", *[f"candidate_{i:04d}.jsonl" for i in range(1, 64)]):
        branch = source / "branches" / name
        expected_hash = summary.get("branch_file_sha256", {}).get(name)
        if not branch.is_file() or _sha256(branch) != expected_hash:
            raise RuntimeError(f"{task}/seed_{seed}: v1 branch integrity failed: {name}")
    return source, acceptance, summary, frozen, arrays


def _trace_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        key: step.get(key)
        for key in (
            "raw_env_step", "current", "goal", "action", "terminated", "truncated",
            "env_success", "predicate_success", "termination_reason", "termination_reason_source",
        )
    }


def _compare_replays(reference: list[dict[str, Any]], replayed: list[dict[str, Any]]) -> dict[str, Any]:
    left = {int(row["slot"]): row for row in reference}
    right = {int(row["slot"]): row for row in replayed}
    if set(left) != set(right) or len(left) != 50:
        raise RuntimeError("replay-order check did not cover the exact 50 state slots")
    max_abs = 0.0
    comparisons = 0
    for slot in range(50):
        a, b = left[slot], right[slot]
        if bool(a["success"]) != bool(b["success"]) or len(a["steps"]) != len(b["steps"]):
            raise RuntimeError(f"replay order changed success/length at slot={slot}")
        for sa, sb in zip(a["steps"], b["steps"]):
            for key in ("terminated", "truncated", "env_success", "predicate_success"):
                if sa.get(key) != sb.get(key):
                    raise RuntimeError(f"replay order changed {key} at slot={slot}")
            for key in ("current", "goal", "action"):
                va, vb = sa.get(key), sb.get(key)
                if va is None or vb is None:
                    if va != vb:
                        raise RuntimeError(f"replay order changed {key} at slot={slot}")
                    continue
                xa, xb = np.asarray(va, dtype=np.float64), np.asarray(vb, dtype=np.float64)
                if xa.shape != xb.shape:
                    raise RuntimeError(f"replay order changed {key} shape at slot={slot}")
                error = float(np.max(np.abs(xa-xb))) if xa.size else 0.0
                max_abs = max(max_abs, error)
                comparisons += 1
                if not np.allclose(xa, xb, atol=1e-6, rtol=1e-6):
                    raise RuntimeError(f"replay order changed {key} at slot={slot}: max_abs={error}")
    return {"status": "pass", "checked_states": 50, "array_comparisons": comparisons, "max_absolute_difference": max_abs}


def _attach_v2(
    expected: list[dict[str, Any]], episodes: list[dict[str, Any]], *, task: str,
    candidate_index: int, trace_stream: Any,
) -> list[dict[str, Any]]:
    clipped_episodes = []
    for episode in episodes:
        clipped = dict(episode)
        clipped["steps"] = pool._first_episode_steps(list(episode.get("steps", [])))
        clipped_episodes.append(clipped)
    by_slot = {int(row["slot"]): row for row in clipped_episodes}
    result = pool._attach_milestones(expected, clipped_episodes, task=task)
    for row in result:
        slot = int(row["slot"])
        episode = by_slot[slot]
        steps = episode.get("steps", [])
        if not steps:
            raise RuntimeError(f"empty replay trace for slot={slot}, candidate={candidate_index}")
        trace = [_trace_step(step) for step in steps]
        final = trace[-1]
        if final.get("current") is None or final.get("goal") is None:
            raise RuntimeError(f"endpoint vectors missing for slot={slot}, candidate={candidate_index}")
        terminated = bool(final.get("terminated", False))
        truncated = bool(final.get("truncated", False))
        if terminated and truncated:
            raise RuntimeError(f"endpoint cannot be both terminated and truncated, slot={slot}, candidate={candidate_index}")
        if any(bool(step.get("terminated")) or bool(step.get("truncated")) for step in trace[:-1]):
            raise RuntimeError(f"trace contains steps after its first native terminal event, slot={slot}, candidate={candidate_index}")
        if len(trace) < 25 and not (terminated or truncated):
            raise RuntimeError(
                f"short nonterminal replay ({len(trace)} steps), slot={slot}, candidate={candidate_index}"
            )
        if len(trace) > 25:
            raise RuntimeError(f"replay exceeded 25 steps, slot={slot}, candidate={candidate_index}")
        raw_steps = [int(step["raw_env_step"]) for step in trace]
        if any(b <= a for a, b in zip(raw_steps, raw_steps[1:])):
            raise RuntimeError(f"trace raw environment steps are not strictly increasing, slot={slot}, candidate={candidate_index}")
        if not (terminated or truncated) and len(trace) == 25 and final.get("termination_reason") != "running":
            raise RuntimeError(f"budget endpoint has unexpected termination reason, slot={slot}, candidate={candidate_index}")
        endpoint_type = (
            "terminated" if terminated else "truncated" if truncated else "budget_25"
        )
        components = pool._physical_cost(task, final["current"], final["goal"])
        row.update(
            {
                "endpoint_type": endpoint_type,
                "endpoint_current": final["current"],
                "endpoint_goal": final["goal"],
                "endpoint_raw_env_step": final.get("raw_env_step"),
                "endpoint_terminated": terminated,
                "endpoint_truncated": truncated,
                "endpoint_env_success": final.get("env_success"),
                "endpoint_predicate_success": final.get("predicate_success"),
                "endpoint_termination_reason": final.get("termination_reason"),
                "endpoint_termination_reason_source": final.get("termination_reason_source"),
                "endpoint_cost_recomputed": float(components["selection_cost"]),
                "endpoint_cost_components_recomputed": components,
                "trajectory_step_count": len(trace),
                "trajectory_trace_sha256": _canonical_sha(trace),
                "trajectory_trace_file": f"candidate_{candidate_index:04d}.jsonl.gz",
            }
        )
        trace_row = {
            "task": task,
            "candidate_index": candidate_index,
            "slot": slot,
            "episode_id": row["episode_id"],
            "state_id": row["state_id"],
            "steps": trace,
        }
        trace_stream.write(
            json.dumps(trace_row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        )
    return result


def run(args: argparse.Namespace) -> Path:
    task, seed, gpu = args.task, int(args.seed), str(args.gpu)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != gpu:
        raise ValueError("CUDA_VISIBLE_DEVICES must equal the single selected physical GPU")
    if args.device != "cuda:0":
        raise ValueError("Use logical cuda:0 with CUDA_VISIBLE_DEVICES selecting one physical GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    preflight = _gpu_preflight(gpu, MIN_FREE_MIB)

    freeze_root = Path(args.freeze_root).resolve()
    global_frozen = _read_json(freeze_root / "frozen_config.json")
    global_refs_path = freeze_root / "source_refs.json"
    if _sha256(global_refs_path) != global_frozen.get("source_refs_sha256"):
        raise RuntimeError("global frozen source_refs hash mismatch")
    if _sha256(ROOT / global_frozen["plan_path"]) != global_frozen.get("plan_sha256"):
        raise RuntimeError("v2 plan changed after protocol freeze")
    for code_path, expected_hash in global_frozen.get("implementation_code_sha256", {}).items():
        if _sha256(ROOT / code_path) != expected_hash:
            raise RuntimeError(f"implementation changed after freeze: {code_path}")
    physical_protocol_check = _validate_physical_protocol(task, global_frozen)
    global_refs = _read_json(global_refs_path)

    source, source_acceptance, source_summary, source_frozen, arrays = _load_source(task, seed)
    source_ref = global_refs["cells"][f"{task}/seed_{seed}"]["3a"]
    if source_ref.get("acceptance_sha256") != _sha256(source / "acceptance.json"):
        raise RuntimeError("global frozen references do not match this 3a acceptance")
    if source_ref.get("fixed_pool_sha256") != _sha256(source / "fixed_pool.npz"):
        raise RuntimeError("global frozen references do not match this 3a pool")
    if source_ref.get("frozen_config_sha256") != _sha256(source / "frozen_config.json"):
        raise RuntimeError("global frozen references do not match this 3a frozen config")
    source_frozen = _read_json(source / "frozen_config.json")
    cell = Path(args.output_root).resolve() / task / f"seed_{seed}"
    attempt = cell / args.attempt
    if attempt.exists():
        raise FileExistsError(f"refusing to overwrite existing v2 attempt: {attempt}")

    cohort_path = pool.TABLE1 / f"cohorts/{task}/seed_{seed}.json"
    manifest = CohortManifest.load(cohort_path)
    if len(manifest.entries) != 50 or manifest.computed_sha256 != source_summary.get("cohort_sha256"):
        raise RuntimeError(f"{task}/seed_{seed}: source cohort identity changed")
    if source_ref.get("cohort_sha256") != manifest.computed_sha256:
        raise RuntimeError("global frozen references do not match the loaded cohort")
    actor_path = pool._table1_actor_path(task)
    actor, loaded_path = load_policy_or_model(str(actor_path), cache_dir=str(ROOT / "data"))
    if loaded_path is None:
        raise RuntimeError(f"could not resolve archived actor: {actor_path}")
    actor_path = Path(loaded_path).resolve()
    actual_actor_sha = _sha256(actor_path)
    if actual_actor_sha != source_summary.get("actor_checkpoint_sha256") or actual_actor_sha != source_frozen["models"]["cowm_r4_ab"]["sha256"]:
        raise RuntimeError(f"loaded actor checkpoint hash changed: {actor_path}")
    actor_model = getattr(actor, "model", actor).to(args.device).eval()
    cfg = pool._resolved_config(task, seed, args.device)
    dataset_path = Path(cfg.eval.dataset_name).resolve()
    expected_dataset = source_frozen["dataset"]
    dataset_ref = source_ref.get("dataset", {})
    expected_dataset_path = (ROOT / dataset_ref.get("path", "")).resolve()
    if not dataset_ref.get("path") or dataset_path != expected_dataset_path:
        raise RuntimeError("resolved runtime dataset path differs from the audited frozen dataset path")
    stat = dataset_path.stat()
    if stat.st_size != int(expected_dataset["file_size_bytes"]):
        raise RuntimeError(f"dataset file size changed: {dataset_path}")
    if dataset_ref.get("sha256") != expected_dataset["sha256"] or dataset_ref.get("file_size_bytes") != stat.st_size:
        raise RuntimeError("dataset identity does not match the audited frozen source")
    if dataset_ref.get("mtime_ns") != stat.st_mtime_ns:
        raise RuntimeError("dataset changed after the source hash audit")
    lewm_path = ROOT / source_frozen["models"]["table1_lewm"]["path"]
    actual_lewm_sha = _sha256(lewm_path)
    if actual_lewm_sha != source_summary.get("verifier_checkpoint_sha256") or actual_lewm_sha != source_frozen["models"]["table1_lewm"]["sha256"]:
        raise RuntimeError(f"LeWM checkpoint hash changed: {lewm_path}")
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    session = DatasetEvaluationSession(cfg, task=task, dataset=dataset, cohort=manifest.to_evaluation_cohort())
    if task == "pusht":
        session.world_factory = pool._seeded_world_factory(session.world_factory, int(cfg.seed))
    action_processor = session.process.get("action")
    normalizer = pool._action_normalizer_record(action_processor)
    if normalizer["sha256"] != source_summary.get("action_normalizer_sha256"):
        raise RuntimeError(f"{task}/seed_{seed}: action normalizer differs from source pool")
    attempt.mkdir(parents=True)
    interop = pool._action_interop_audit(
        arrays["candidates"], action_processor, int(cfg.plan_config.action_block)
    )
    physical = pool._physical_actions(
        arrays["candidates"], action_processor, int(cfg.plan_config.action_block)
    )
    if not np.allclose(physical, arrays["physical_actions"], rtol=0, atol=1e-7):
        raise RuntimeError(f"{task}/seed_{seed}: archived physical actions do not roundtrip")

    branch_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.update(branch_cfg, "eval.eval_budget", 25, force_add=True)
    OmegaConf.update(branch_cfg, "world.max_episode_steps", 50, force_add=True)
    reusable_world = None
    if task != "pusht":
        world_cfg = OmegaConf.to_container(branch_cfg.world, resolve=True)
        world_cfg["max_episode_steps"] = 50
        reusable_world = session.world_factory(**world_cfg, image_shape=(224, 224))
    branch_state: dict[str, Any] = {}
    branch_root = attempt / "branches"
    branch_root.mkdir()
    trace_root = attempt / "traces"
    trace_root.mkdir()
    joint = {"candidates": arrays["candidates"], "costs": arrays["cowm_b_costs"], "noise_sha256": source_summary["candidate_noise_sha256"]}
    lewm = {"costs": arrays["lewm_costs"]}
    reverse_first = pool._replay_candidate(
        cfg=branch_cfg,
        task=task,
        manifest=manifest,
        actions=arrays["candidates"][:, 63],
        session=session,
        actor_model=actor_model,
        device=args.device,
        output_dir=attempt / "replay_work" / "candidate_0063_first",
        branch_state=branch_state,
        reusable_world=reusable_world,
    )
    baseline_first = pool._replay_candidate(
        cfg=branch_cfg,
        task=task,
        manifest=manifest,
        actions=arrays["candidates"][:, 0],
        session=session,
        actor_model=actor_model,
        device=args.device,
        output_dir=attempt / "replay_work" / "candidate_0000_first",
        branch_state=branch_state,
        reusable_world=reusable_world,
    )
    baseline_repeat = pool._replay_candidate(
        cfg=branch_cfg,
        task=task,
        manifest=manifest,
        actions=arrays["candidates"][:, 0],
        session=session,
        actor_model=actor_model,
        device=args.device,
        output_dir=attempt / "replay_work" / "candidate_0000_repeat",
        branch_state=branch_state,
        reusable_world=reusable_world,
    )
    restore_check = pool._assert_restore_repeatable(
        baseline_first,
        baseline_repeat,
        state_limit=50 if task == "pusht" else 5,
    )
    if task == "pusht":
        restore_check.update({
            "replay_strategy": "seeded_dataset_reset_with_fresh_branch_snapshot",
            "environment_seed": int(cfg.seed),
            "rng_restored": False,
            "dataset_reset_seed_applied": True,
        })
    else:
        restore_check["replay_strategy"] = "captured_environment_and_rng_snapshot"
    states_index = []
    for slot, entry in enumerate(manifest.entries):
        states_index.append({
            "slot": slot,
            "state_id": p15._candidate_state_id(entry),
            "episode_id": entry.episode_id,
            "start_step": int(entry.start_step),
            "row_index": int(entry.row_index),
            "goal_row_index": int(entry.goal_row_index),
        })
    _write_json(attempt / "state_index.json", {"task": task, "seed": seed, "entries": states_index})

    replay_summary = {"candidate_count": 64, "branch_record_count": 0, "trace_record_count": 0, "endpoint_types": {}}
    for candidate_index in range(64):
        old_rows = _read_jsonl(source / "branches" / f"candidate_{candidate_index:04d}.jsonl")
        if len(old_rows) != 50:
            raise RuntimeError(f"v1 branch has {len(old_rows)} rows, expected 50")
        expected = pool._expected_rows(
            task=task, manifest=manifest, candidate_index=candidate_index,
            joint=joint, lewm=lewm, physical=physical,
        )
        for old, new in zip(old_rows, expected):
            if old["slot"] != new["slot"] or old["state_id"] != new["state_id"]:
                raise RuntimeError("source branch order/state identity differs from frozen cohort")
            for key in ("predicted_cost_cowm_b", "predicted_cost_lewm"):
                if not math.isclose(float(old[key]), float(new[key]), rel_tol=0, abs_tol=1e-12):
                    raise RuntimeError(f"source predicted score mismatch: {key}, candidate={candidate_index}")

        replay_started = time.time()
        episodes = (
            baseline_first if candidate_index == 0 else reverse_first if candidate_index == 63 else pool._replay_candidate(
                cfg=branch_cfg,
                task=task,
                manifest=manifest,
                actions=arrays["candidates"][:, candidate_index],
                session=session,
                actor_model=actor_model,
                device=args.device,
                output_dir=attempt / "replay_work" / f"candidate_{candidate_index:04d}",
                branch_state=branch_state,
                reusable_world=reusable_world,
            )
        )
        trace_path = trace_root / f"candidate_{candidate_index:04d}.jsonl.gz"
        trace_temp = trace_path.with_suffix(trace_path.suffix + ".tmp")
        with trace_temp.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
                import io
                text_stream = io.TextIOWrapper(zipped, encoding="utf-8", write_through=True)
                rows = _attach_v2(
                    expected, episodes, task=task, candidate_index=candidate_index,
                    trace_stream=text_stream,
                )
                text_stream.flush()
                text_stream.detach()
        trace_temp.replace(trace_path)
        for old, new in zip(old_rows, rows):
            for key in ("effective_valid_length", "success_by_5", "success_by_25", "terminal_step"):
                if old.get(key) != new.get(key):
                    raise RuntimeError(
                        f"v1/v2 replay changed {key} at candidate={candidate_index}, slot={new['slot']}"
                    )
            old_cost = old.get("terminal_selection_cost")
            if old_cost is None or not math.isclose(
                float(old_cost), float(new["endpoint_cost_recomputed"]), rel_tol=0, abs_tol=1e-6
            ):
                raise RuntimeError(f"v1/v2 endpoint cost changed at candidate={candidate_index}, slot={new['slot']}")
            if old.get("native_success_after_budget") != new.get("native_success_after_budget"):
                raise RuntimeError(f"v1/v2 terminal success changed at candidate={candidate_index}, slot={new['slot']}")
        branch_path = branch_root / f"candidate_{candidate_index:04d}.jsonl"
        p15._write_jsonl(branch_path, rows)
        replay_summary["branch_record_count"] += len(rows)
        replay_summary["trace_record_count"] += len(episodes)
        for row in rows:
            replay_summary["endpoint_types"][row["endpoint_type"]] = replay_summary["endpoint_types"].get(row["endpoint_type"], 0) + 1
        print(
            f"{task} seed={seed} candidate={candidate_index+1}/64 rows=50 "
            f"elapsed={time.time()-replay_started:.1f}s",
            flush=True,
        )

    reverse_repeat = pool._replay_candidate(
        cfg=branch_cfg,
        task=task,
        manifest=manifest,
        actions=arrays["candidates"][:, 63],
        session=session,
        actor_model=actor_model,
        device=args.device,
        output_dir=attempt / "replay_work" / "candidate_0063_after_full_order",
        branch_state=branch_state,
        reusable_world=reusable_world,
    )
    order_check = _compare_replays(reverse_first, reverse_repeat)

    if reusable_world is not None:
        reusable_world.close()
    fixed_pool_path = attempt / "fixed_pool.npz"
    shutil.copyfile(source / "fixed_pool.npz", fixed_pool_path)
    source_ref = {
        "source_cell": str(source.relative_to(ROOT)),
        "source_acceptance_sha256": _sha256(source / "acceptance.json"),
        "source_summary_sha256": _sha256(source / "summary.json"),
        "source_fixed_pool_sha256": _sha256(source / "fixed_pool.npz"),
        "source_cohort_sha256": source_summary["cohort_sha256"],
        "source_branch_hashes": source_summary["branch_file_sha256"],
        "reuse_policy": "reuse exact v1 candidate actions and both score matrices; replay all candidates only to retain endpoint vectors, termination flags, and state/action traces",
    }
    _write_json(attempt / "source_refs.json", source_ref)
    frozen = {
        "schema_version": 1,
        "experiment": "cvpr_table3_v2_endpoint_replay",
        "task": task,
        "evaluation_seed": seed,
        "gpu_preflight": preflight,
        "source_refs_sha256": _sha256(attempt / "source_refs.json"),
        "global_frozen_config_sha256": _sha256(freeze_root / "frozen_config.json"),
        "global_source_refs_sha256": _sha256(global_refs_path),
        "source_fixed_pool_sha256": source_ref["source_fixed_pool_sha256"],
        "new_fixed_pool_sha256": _sha256(fixed_pool_path),
        "cohort_sha256": manifest.computed_sha256,
        "actor_checkpoint_sha256": source_summary["actor_checkpoint_sha256"],
        "loaded_actor_checkpoint_sha256": actual_actor_sha,
        "lewm_checkpoint_sha256": source_summary["verifier_checkpoint_sha256"],
        "loaded_lewm_checkpoint_sha256": actual_lewm_sha,
        "dataset_sha256": dataset_ref["sha256"],
        "dataset_file_signature": {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "action_normalizer": normalizer,
        "protocol": {
            "candidate_count": 64,
            "candidate_source": "accepted CVPR Table 3a v1 fixed pool; no candidate regeneration",
            "replay_budget": 25,
            "endpoint": "last observed physical state at native terminated/truncated event or budget step 25",
            "early_terminal_state_repeated": False,
            "termination_event_fields": ["terminated", "truncated", "env_success", "predicate_success"],
            "endpoint_tie_definition": "exact equality to the minimum float64 recorded physical cost; all exact minimizers form the oracle set",
            "spearman_ties": "average ranks; constant/nonfinite sequence is NA",
            "environment_action_processing": "native Table 1 action normalizer and environment clipping",
            "autocast": False,
            "tf32": False,
            "torch_compile": False,
            "precision": "FP32",
            "action_interop_audit": interop,
            "trace_fields": ["raw_env_step", "current", "goal", "action", "terminated", "truncated", "env_success", "predicate_success", "termination_reason", "termination_reason_source"],
            "improvement_tolerance_for_3b": 1e-6,
            "task_physical_definition": physical_protocol_check,
        },
        "branch_record_count": replay_summary["branch_record_count"],
        "trajectory_trace_record_count": replay_summary["trace_record_count"],
        "endpoint_types": replay_summary["endpoint_types"],
        "branch_file_sha256": {
            p.name: _sha256(p) for p in sorted(branch_root.glob("*.jsonl"))
        },
        "trace_file_sha256": {
            p.name: _sha256(p) for p in sorted(trace_root.glob("*.jsonl.gz"))
        },
        "restore_check": restore_check,
        "candidate_order_check": order_check,
        "runtime": {"cuda_visible_devices": visible, "logical_device": args.device},
    }
    _write_json(attempt / "frozen_config.json", frozen)
    checks = {
        "source_v1_cell_passed": True,
        "exact_source_candidate_pool_reused": _sha256(fixed_pool_path) == source_ref["source_fixed_pool_sha256"],
        "all_64_candidates_replayed": replay_summary["branch_record_count"] == 50 * 64,
        "endpoint_vectors_and_flags_stored": True,
        "all_64_trace_files_stored_and_hashed": len(frozen["trace_file_sha256"]) == 64,
        "all_24_states_by_64_candidates_identity_verified": replay_summary["branch_record_count"] == 3200,
        "candidate_order_independent_truth": order_check["status"] == "pass",
        "old_success_and_effective_length_match": True,
        "terminal_cost_recomputed_from_saved_vectors": True,
        "source_endpoint_cost_matches_replay": True,
        "loaded_checkpoint_and_dataset_identity": True,
        "action_normalizer_match": True,
        "action_interop_pass": interop["status"] == "pass",
    }
    if not all(checks.values()):
        raise RuntimeError(f"v2 acceptance checks failed: {checks}")
    summary = {
        "schema_version": 1,
        "task": task,
        "evaluation_seed": seed,
        "source_refs": source_ref,
        "replay": replay_summary,
        "checks": checks,
        "branch_file_sha256": frozen["branch_file_sha256"],
        "trace_file_sha256": frozen["trace_file_sha256"],
        "frozen_config_sha256": _sha256(attempt / "frozen_config.json"),
    }
    _write_json(attempt / "summary.json", summary)
    _write_json(attempt / "acceptance.json", {"status": "pass", "errors": [], "checks": checks, "summary_sha256": _sha256(attempt / "summary.json")})
    print(json.dumps({"status": "ok", "attempt": str(attempt), "branch_rows": replay_summary["branch_record_count"], "trace_rows": replay_summary["trace_record_count"]}, sort_keys=True), flush=True)
    return attempt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--gpu", required=True, choices=tuple(str(i) for i in range(4)))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--freeze-root", default=str(ROOT / "outputs/cvpr/table3/v2"))
    parser.add_argument("--attempt", default="attempt_001")
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
