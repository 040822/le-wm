#!/usr/bin/env python3
"""Bound-aware Phase1.6 PO/GF direction checks on the legacy 50-state cohort."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from scripts import round5_phase1_5_diagnostics as diag
from source.common.round3_phase1 import CohortManifest
from source.common.round3_protocol import evaluate_success
from source.common.round4_action_bounds import (
    compute_normalized_action_bounds,
    project_normalized_actions,
)
from source.common.round5_phase1_6 import atomic_json, stable_hash


DEFAULT_CONFIG = ROOT / "config/round5/phase1_5.json"
DEFAULT_POOL_ROOT = ROOT / "outputs/round5/phase1_5_seed3072_legacy/diagnostics/candidate_pool"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/round5/phase1_6_seed3072"
DEFAULT_CLIP_AUDIT = ROOT / "outputs/round5/phase1_6_seed3072/audit/phase1_5_pool_clip_audit.json"
TASKS = ("pusht", "reacher")
DEVICE = "cuda"


def _physical_gpu(value: str) -> str:
    if not str(value).isdigit() or int(value) not in range(8):
        raise argparse.ArgumentTypeError("--gpu must select one physical GPU0-7")
    return str(value)


def _seed(*parts: Any) -> int:
    data = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _submanifest(manifest: CohortManifest, keep_slots: Sequence[int]) -> CohortManifest:
    entries = tuple(manifest.entries[int(slot)] for slot in keep_slots)
    return replace(
        manifest,
        cohort_id=f"{manifest.cohort_id}_phase1_6_bound_compatible",
        cohort_kind="custom",
        entries=entries,
        episode_split={**manifest.episode_split, "custom": tuple(entry.episode_id for entry in entries)},
        cohort_sha256=None,
    )


def _load_baseline(pool_root: Path, task: str, candidate: int) -> list[dict[str, Any]]:
    path = pool_root / task / "branches" / "s2" / f"candidate_{candidate:04d}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 50 or {int(row["slot"]) for row in rows} != set(range(50)):
        raise RuntimeError(f"candidate branch must contain the 50 fixed slots: {path}")
    return sorted(rows, key=lambda row: int(row["slot"]))


def _changed_slots(audit_path: Path, task: str) -> tuple[set[int], dict[str, Any]]:
    audit = _read_json(audit_path)
    matches = [row for row in audit.get("tasks", []) if row.get("task") == task]
    if len(matches) != 1:
        raise RuntimeError(f"action bound audit must contain exactly one {task} record")
    record = matches[0]
    slots = set()
    for candidate, pairs in record.get("pairs_by_candidate", {}).items():
        if int(candidate) >= 64:
            continue
        slots.update(int(pair["slot"]) for pair in pairs)
    return slots, record


def _clip(actions: np.ndarray, bounds: Any) -> np.ndarray:
    result = project_normalized_actions(
        np.asarray(actions, dtype=np.float32), bounds, mode="clip"
    )
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("clipped action path contains a non-finite value")
    return result


def _rms_delta(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(delta))))


def _matched_bounded_random(
    baseline: np.ndarray,
    guided: np.ndarray,
    bounds: Any,
    *,
    seed: int,
    max_directions: int = 32,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    baseline = np.asarray(baseline, dtype=np.float32)
    guided = np.asarray(guided, dtype=np.float32)
    if baseline.shape != guided.shape or baseline.ndim != 3:
        raise ValueError("baseline and guided actions must both have shape [B,H,A]")
    rng = np.random.default_rng(int(seed))
    random_actions = baseline.copy()
    records: list[dict[str, Any]] = []
    for row_index in range(len(baseline)):
        target = _rms_delta(guided[row_index], baseline[row_index])
        if target <= 1e-10:
            records.append({"valid": False, "attempts": 0, "target_rms": target, "actual_rms": 0.0, "reason": "zero_guidance_displacement"})
            continue
        matched = False
        last_rms = 0.0
        for attempt in range(1, int(max_directions) + 1):
            direction = rng.normal(size=baseline[row_index].shape)
            norm = float(np.sqrt(np.mean(np.square(direction))))
            if norm <= 1e-12:
                continue
            direction = direction / norm

            low, high = 0.0, max(target, 1e-4)
            candidate_high = _clip((baseline[row_index] + high * direction)[None], bounds)[0]
            last_rms = _rms_delta(candidate_high, baseline[row_index])
            while last_rms < target and high < 64.0:
                high *= 2.0
                candidate_high = _clip((baseline[row_index] + high * direction)[None], bounds)[0]
                last_rms = _rms_delta(candidate_high, baseline[row_index])
            if last_rms + max(1e-5, target * 1e-3) < target:
                continue
            best = candidate_high
            for _ in range(32):
                middle = (low + high) * 0.5
                candidate = _clip((baseline[row_index] + middle * direction)[None], bounds)[0]
                actual = _rms_delta(candidate, baseline[row_index])
                if abs(actual - target) < abs(_rms_delta(best, baseline[row_index]) - target):
                    best = candidate
                if actual < target:
                    low = middle
                else:
                    high = middle
            actual = _rms_delta(best, baseline[row_index])
            if abs(actual - target) <= max(1e-5, target * 1e-3):
                random_actions[row_index] = best
                records.append({"valid": True, "attempts": attempt, "target_rms": target, "actual_rms": actual, "reason": None})
                matched = True
                break
        if not matched:
            records.append({"valid": False, "attempts": int(max_directions), "target_rms": target, "actual_rms": _rms_delta(random_actions[row_index], baseline[row_index]), "reason": "no_clipped_direction_matched_rms"})
    return random_actions, records


def _score(model: Any, starts: torch.Tensor, goals: torch.Tensor, actions: np.ndarray) -> np.ndarray:
    action_tensor = torch.as_tensor(actions, device=starts.device, dtype=torch.float32)
    with torch.inference_mode():
        values = model.get_cost_from_latents(starts, goals, action_tensor[:, None])[:, 0]
    return values.detach().float().cpu().numpy()


def _milestone(row: Mapping[str, Any], step: int = 25) -> Mapping[str, Any] | None:
    value = row.get("milestones", {}).get(str(step))
    return value if isinstance(value, Mapping) else None


def _event(task: str, row: Mapping[str, Any], step: int = 25) -> bool | None:
    milestone = _milestone(row, step)
    if milestone is None or milestone.get("current") is None or milestone.get("goal") is None:
        return None
    return bool(evaluate_success(task, milestone["current"], milestone["goal"]))


def _episode_metrics(task: str, baseline: Mapping[str, Any], guided: Mapping[str, Any], random: Mapping[str, Any]) -> dict[str, Any]:
    rows = {"baseline": baseline, "guided": guided, "random": random}
    milestones = {key: _milestone(row) for key, row in rows.items()}
    valid = all(
        value is not None
        and value.get("distance") is not None
        and value.get("future_latent_cost") is not None
        and value.get("current") is not None
        and value.get("goal") is not None
        for value in milestones.values()
    )
    if not valid:
        return {"paired_25_valid": False, "physical_distance_25": None, "latent_cost_25": None, "success_by_25": None}
    physical = {key: float(value["distance"]) for key, value in milestones.items()}
    latent = {key: float(value["future_latent_cost"]) for key, value in milestones.items()}
    success = {key: bool(evaluate_success(task, value["current"], value["goal"])) for key, value in milestones.items()}
    return {
        "paired_25_valid": True,
        "physical_distance_25": physical,
        "latent_cost_25": latent,
        "success_by_25": success,
        "guided_physical_improvement_vs_baseline": physical["baseline"] - physical["guided"],
        "random_physical_improvement_vs_baseline": physical["baseline"] - physical["random"],
        "guided_minus_random_physical_improvement": physical["random"] - physical["guided"],
        "guided_latent_improvement_vs_baseline": latent["baseline"] - latent["guided"],
        "random_latent_improvement_vs_baseline": latent["baseline"] - latent["random"],
    }


def _guidance_actions(
    model: Any,
    mode: str,
    starts: torch.Tensor,
    goals: torch.Tensor,
    baseline: np.ndarray,
    noise: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    baseline_tensor = torch.as_tensor(baseline, device=starts.device, dtype=torch.float32)
    if mode == "post_opt":
        guided = model.post_optimize_actions(
            starts,
            goals,
            baseline_tensor,
            step_size=0.01,
            inner_steps=2,
            max_rms_offset=0.2,
        )
    elif mode == "guided_flow":
        guided = model.sample_actions(
            starts,
            noise=torch.as_tensor(noise, device=starts.device, dtype=torch.float32),
            num_steps=2,
            goal_latent=goals,
            integrator="euler",
            guidance_mode="guided_flow",
            guidance_step_size=0.01,
            guidance_last_steps=2,
            guidance_inner_steps=1,
            guidance_max_rms_offset=0.2,
        )
    else:
        raise ValueError(mode)
    stats = dict(getattr(model, "last_guidance_stats", {}))
    return guided.detach().float().cpu().numpy(), stats


def _one_task(task: str, args: argparse.Namespace) -> dict[str, Any]:
    gpu = str(args.gpu)
    diag._configure_device(
        DEVICE,
        gpu,
        minimum_free_mib=int(args.min_free_mib),
        max_load_per_cpu=float(args.max_load_per_cpu),
        minimum_available_mib=int(args.min_available_mib),
        minimum_swap_free_mib=int(args.min_swap_free_mib),
    )
    if int(gpu) >= 4:
        memory_budget = _read_json(args.output_root / "analysis/eval_memory_budget.json")
        if not memory_budget.get("eligible_for_gpu4_7") or float(memory_budget.get("maximum_peak_allocated_gib", 99)) > 8.0:
            raise RuntimeError("GPU4-7 may only run this evaluation after calibration below the 8 GiB cap")

    config = diag._load_config(diag._resolve(args.config))
    manifest = diag._manifest(config, task)
    checkpoint, checkpoint_hash = diag._checkpoint(config, task)
    task_pool = args.pool_root / task
    pool_manifest = _read_json(task_pool / "manifest.json")
    if pool_manifest.get("checkpoint_sha256") != checkpoint_hash or pool_manifest.get("cohort_sha256") != manifest.computed_sha256:
        raise RuntimeError(f"legacy candidate pool identity mismatch for {task}")
    clip_slots, clip_audit = _changed_slots(args.clip_audit, task)
    kept_slots = [slot for slot in range(50) if slot not in clip_slots]
    if len(kept_slots) < 40:
        raise RuntimeError(f"clip compatibility audit left too few states for {task}: {len(kept_slots)}")
    run_manifest = _submanifest(manifest, kept_slots)

    model, resolved = diag.load_policy_or_model(str(checkpoint))
    if resolved is not None and Path(resolved).resolve() != checkpoint.resolve():
        raise RuntimeError("checkpoint resolver changed the requested checkpoint")
    model = getattr(model, "model", model).to(DEVICE).eval()
    cfg = diag.compose_eval_config(
        task,
        overrides=[
            f"eval.num_eval={len(run_manifest.entries)}",
            "eval.goal_offset_steps=25",
            "eval.eval_budget=50",
            "plan_config.horizon=5",
            "plan_config.receding_horizon=5",
            "plan_config.action_block=5",
            "output.save_video=false",
            f"solver.device={DEVICE}",
        ],
    )
    session = diag.DatasetEvaluationSession(cfg, task=task, cohort=run_manifest.to_evaluation_cohort())
    process = session.process
    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    world_cfg["max_episode_steps"] = 2 * int(cfg.eval.eval_budget)
    probe_world = session.world_factory(**world_cfg, image_shape=(224, 224))
    try:
        envs = getattr(probe_world, "envs", None)
        if envs is None:
            raise RuntimeError("environment factory did not expose envs")
        bounds = compute_normalized_action_bounds(
            envs.single_action_space,
            process.get("action"),
            action_block=int(cfg.plan_config.action_block),
        )
    finally:
        if hasattr(probe_world, "close"):
            probe_world.close()
    bound_metadata = bounds.metadata(action_dim=int(model.action_dim))
    starts_all = diag._encode_start_latents(session.dataset, run_manifest, session.transform["pixels"], model, DEVICE)
    goals_all = diag._encode_goal_latents(session.dataset, run_manifest, session.transform["pixels"], model, DEVICE)
    starts = torch.as_tensor(starts_all, device=DEVICE, dtype=torch.float32)
    goals = torch.as_tensor(goals_all, device=DEVICE, dtype=torch.float32)

    baseline_rows_by_candidate: dict[int, list[dict[str, Any]]] = {}
    baseline_actions_by_candidate: dict[int, np.ndarray] = {}
    for candidate in (0, 1):
        source_rows = _load_baseline(args.pool_root, task, candidate)
        filtered_rows = [dict(source_rows[slot]) for slot in kept_slots]
        source_actions = np.asarray([row["action"] for row in filtered_rows], dtype=np.float32)
        legal_actions = _clip(source_actions, bounds)
        if not np.allclose(source_actions, legal_actions, rtol=0.0, atol=1e-6):
            raise RuntimeError(f"unlisted legacy action changed under the Phase1.6 bound audit: {task}, candidate={candidate}")
        baseline_rows_by_candidate[candidate] = []
        for new_slot, (original_slot, row) in enumerate(zip(kept_slots, filtered_rows)):
            row["source_slot"] = int(original_slot)
            row["slot"] = int(new_slot)
            baseline_rows_by_candidate[candidate].append(row)
        baseline_actions_by_candidate[candidate] = legal_actions

    noise_path = task_pool / "candidate_noise_s2.npz"
    with np.load(noise_path) as archive:
        noise_all = np.asarray(archive["candidate_noise"], dtype=np.float32)
    output_dir = args.output_root / "mechanism/guidance" / task
    variants_dir = output_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    completed: list[str] = []
    for mode in ("post_opt", "guided_flow"):
        for candidate in (0, 1):
            target = variants_dir / f"{mode}_s2_c{candidate}.jsonl"
            if target.is_file():
                existing = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line]
                if len(existing) == len(kept_slots) and all(row.get("action_bound_sha256") == stable_hash(bound_metadata) for row in existing):
                    records.extend(existing)
                    completed.append(target.name)
                    continue
            baseline_rows = baseline_rows_by_candidate[candidate]
            baseline_actions = baseline_actions_by_candidate[candidate]
            noise = noise_all[kept_slots, candidate]
            # The stored flow proposals are required to reconstruct the matching S=2 candidate.
            if mode == "guided_flow":
                with torch.inference_mode():
                    regenerated = model.sample_actions(
                        starts,
                        noise=torch.as_tensor(noise, device=DEVICE, dtype=torch.float32),
                        num_steps=2,
                        goal_latent=goals,
                        integrator="euler",
                    ).detach().float().cpu().numpy()
                if not np.allclose(regenerated, np.asarray([row["action"] for row in baseline_rows]), rtol=2e-5, atol=1e-5):
                    raise RuntimeError(f"S=2 noise no longer reproduces baseline candidate {candidate} for {task}")

            raw_guided, guidance_stats = _guidance_actions(model, mode, starts, goals, baseline_actions, noise)
            guided_actions = _clip(raw_guided, bounds)
            guided_rms = np.asarray([_rms_delta(a, b) for a, b in zip(guided_actions, baseline_actions)], dtype=np.float64)
            random_actions, random_match = _matched_bounded_random(
                baseline_actions,
                guided_actions,
                bounds,
                seed=_seed("round5_phase1_6_bounded_random", task, mode, candidate, 16026),
                max_directions=32,
            )
            random_rms = np.asarray([_rms_delta(a, b) for a, b in zip(random_actions, baseline_actions)], dtype=np.float64)
            before_cost = _score(model, starts, goals, baseline_actions)
            after_cost = _score(model, starts, goals, guided_actions)
            random_cost = _score(model, starts, goals, random_actions)

            replay_state: dict[str, Any] = {}
            guided_episodes = diag._run_fixed_candidate(
                cfg=cfg,
                task=task,
                manifest=run_manifest,
                normalized_actions=guided_actions,
                process=process,
                model=model,
                transform=session.transform["pixels"],
                device=DEVICE,
                output_dir=variants_dir / f"{mode}_s2_c{candidate}" / "guided",
                dataset=session.dataset,
                branch_state=replay_state,
                goal_latents=goals_all,
            )
            random_episodes = diag._run_fixed_candidate(
                cfg=cfg,
                task=task,
                manifest=run_manifest,
                normalized_actions=random_actions,
                process=process,
                model=model,
                transform=session.transform["pixels"],
                device=DEVICE,
                output_dir=variants_dir / f"{mode}_s2_c{candidate}" / "random",
                dataset=session.dataset,
                branch_state=replay_state,
                goal_latents=goals_all,
            )
            guided_by_slot = diag._attach_guidance_branch_outcomes(
                baseline_rows, guided_episodes, guided_actions, task=task
            )
            random_by_slot = diag._attach_guidance_branch_outcomes(
                baseline_rows, random_episodes, random_actions, task=task
            )
            variant_rows: list[dict[str, Any]] = []
            for index, baseline in enumerate(baseline_rows):
                slot = int(baseline["slot"])
                guided_row = guided_by_slot[slot]
                random_row = random_by_slot[slot]
                metrics = _episode_metrics(task, baseline, guided_row, random_row)
                rms = random_match[index]
                variant_rows.append({
                    "schema_version": "round5_phase1_6_guidance_v1",
                    "task": task,
                    "cohort_id": manifest.cohort_id,
                    "cohort_sha256": manifest.computed_sha256,
                    "source_slot": int(baseline["source_slot"]),
                    "state_id": baseline["state_id"],
                    "episode_id": baseline["episode_id"],
                    "start_step": baseline["start_step"],
                    "flow_steps": 2,
                    "candidate_index": candidate,
                    "guidance": mode,
                    "guidance_inner_steps": 2 if mode == "post_opt" else 1,
                    "guidance_last_steps": 2 if mode == "guided_flow" else 0,
                    "guidance_step_size": 0.01,
                    "max_rms_offset": 0.2,
                    "checkpoint_sha256": checkpoint_hash,
                    "action_bound_sha256": stable_hash(bound_metadata),
                    "clip_audit_sha256": diag._sha256_file(args.clip_audit),
                    "baseline_action_sha256": stable_hash(baseline_actions[index].tolist()),
                    "guided_action_sha256": stable_hash(guided_actions[index].tolist()),
                    "random_action_sha256": stable_hash(random_actions[index].tolist()),
                    "baseline_predicted_cost": float(before_cost[index]),
                    "guided_predicted_cost": float(after_cost[index]),
                    "random_predicted_cost": float(random_cost[index]),
                    "guided_action_rms_displacement_after_clip": float(guided_rms[index]),
                    "random_action_rms_displacement_after_clip": float(random_rms[index]),
                    "random_rms_match_valid": bool(rms["valid"]),
                    "random_rms_match_attempts": int(rms["attempts"]),
                    "random_rms_match_error": float(abs(rms["actual_rms"] - rms["target_rms"])),
                    "guidance_stats": guidance_stats,
                    **metrics,
                    "baseline_outcome_status": baseline.get("outcome_status"),
                    "guided_outcome_status": guided_row.get("outcome_status"),
                    "random_outcome_status": random_row.get("outcome_status"),
                })
            _write_jsonl(target, variant_rows)
            records.extend(variant_rows)
            completed.append(target.name)
            print(json.dumps({"task": task, "guidance": mode, "candidate": candidate, "states": len(variant_rows), "paired_25": sum(bool(row["paired_25_valid"]) and bool(row["random_rms_match_valid"]) for row in variant_rows)}, sort_keys=True), flush=True)
            atomic_json(output_dir / "manifest.json", {
                "schema_version": "round5_phase1_6_guidance_manifest_v1",
                "status": "partial",
                "task": task,
                "checkpoint_sha256": checkpoint_hash,
                "source_cohort_sha256": manifest.computed_sha256,
                "source_pool_manifest_sha256": diag._sha256_file(task_pool / "manifest.json"),
                "action_bound_metadata": bound_metadata,
                "action_bound_sha256": stable_hash(bound_metadata),
                "clip_audit_sha256": diag._sha256_file(args.clip_audit),
                "excluded_clip_incompatible_slots": sorted(clip_slots),
                "retained_slots": kept_slots,
                "candidate_indices": [0, 1],
                "flow_steps": 2,
                "post_opt": {"inner_steps": 2, "step_size": 0.01, "max_rms_offset": 0.2},
                "guided_flow": {"inner_steps_per_last_step": 1, "last_steps": 2, "step_size": 0.01, "max_rms_offset": 0.2},
                "random_control": "clip then match per-state final action RMS; up to 32 Gaussian directions",
                "completed_variants": sorted(set(completed)),
                "status_note": clip_audit.get("outcome_reuse_status"),
            })

    reverse_path = output_dir / "candidate0_block_reverse.jsonl"
    if not reverse_path.is_file() or args.overwrite_reverse:
        baseline_rows = baseline_rows_by_candidate[0]
        baseline_actions = baseline_actions_by_candidate[0]
        reversed_actions = _clip(np.flip(baseline_actions, axis=1).copy(), bounds)
        replay_state = {}
        reverse_episodes = diag._run_fixed_candidate(
            cfg=cfg,
            task=task,
            manifest=run_manifest,
            normalized_actions=reversed_actions,
            process=process,
            model=model,
            transform=session.transform["pixels"],
            device=DEVICE,
            output_dir=output_dir / "candidate0_block_reverse",
            dataset=session.dataset,
            branch_state=replay_state,
            goal_latents=goals_all,
        )
        reverse_by_slot = diag._attach_guidance_branch_outcomes(
            baseline_rows, reverse_episodes, reversed_actions, task=task
        )
        reverse_cost = _score(model, starts, goals, reversed_actions)
        reverse_rows = []
        for index, baseline in enumerate(baseline_rows):
            reverse = reverse_by_slot[int(baseline["slot"])]
            baseline_m = _milestone(baseline)
            reverse_m = _milestone(reverse)
            valid = baseline_m is not None and reverse_m is not None and baseline_m.get("distance") is not None and reverse_m.get("distance") is not None
            reverse_rows.append({
                "schema_version": "round5_phase1_6_reverse_control_v1",
                "task": task,
                "cohort_sha256": manifest.computed_sha256,
                "source_slot": int(baseline["source_slot"]),
                "state_id": baseline["state_id"],
                "candidate_index": 0,
                "control": "reverse_action_block_order",
                "action_bound_sha256": stable_hash(bound_metadata),
                "baseline_action_sha256": stable_hash(baseline_actions[index].tolist()),
                "reversed_action_sha256": stable_hash(reversed_actions[index].tolist()),
                "predicted_cost_reversed": float(reverse_cost[index]),
                "paired_25_valid": bool(valid),
                "baseline_distance_25": None if not valid else float(baseline_m["distance"]),
                "reversed_distance_25": None if not valid else float(reverse_m["distance"]),
                "baseline_latent_cost_25": None if not valid else baseline_m.get("future_latent_cost"),
                "reversed_latent_cost_25": None if not valid else reverse_m.get("future_latent_cost"),
                "baseline_success_by_25": _event(task, baseline),
                "reversed_success_by_25": _event(task, reverse),
            })
        _write_jsonl(reverse_path, reverse_rows)

    summary = {
        "schema_version": "round5_phase1_6_guidance_task_summary_v1",
        "task": task,
        "status": "completed",
        "checkpoint_sha256": checkpoint_hash,
        "cohort_sha256": manifest.computed_sha256,
        "source_pool_manifest_sha256": diag._sha256_file(task_pool / "manifest.json"),
        "action_bound_metadata": bound_metadata,
        "action_bound_sha256": stable_hash(bound_metadata),
        "clip_audit_sha256": diag._sha256_file(args.clip_audit),
        "clip_compatibility": clip_audit,
        "retained_slots": kept_slots,
        "completed_variants": sorted(set(completed)),
        "guidance_records": str(output_dir / "variants"),
        "reverse_control_records": str(reverse_path),
        "direction_branch_count": sum(len([json.loads(line) for line in (variants_dir / name).read_text(encoding="utf-8").splitlines() if line]) for name in sorted(set(completed))),
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pool-root", type=Path, default=DEFAULT_POOL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clip-audit", type=Path, default=DEFAULT_CLIP_AUDIT)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--gpu", type=_physical_gpu, required=True)
    parser.add_argument("--min-free-mib", type=int, default=3500)
    parser.add_argument("--min-available-mib", type=int, default=64 * 1024)
    parser.add_argument("--min-swap-free-mib", type=int, default=1024)
    parser.add_argument("--max-load-per-cpu", type=float, default=0.75)
    parser.add_argument("--overwrite-reverse", action="store_true")
    args = parser.parse_args()
    for field in ("config", "pool_root", "output_root", "clip_audit"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    print(json.dumps(_one_task(args.task, args), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
