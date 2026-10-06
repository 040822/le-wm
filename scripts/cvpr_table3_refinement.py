#!/usr/bin/env python3
"""Run the matched-action PO/GF diagnostics for CVPR Table 3b."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import cvpr_table3_fixed_pool as pool
import scripts.round5_phase1_5_diagnostics as p15
from source.common.round3_phase1 import CohortManifest
from source.common.eval import DatasetEvaluationSession, get_dataset
from source.common.round4_action_bounds import compute_normalized_action_bounds


POOL_ROOT = ROOT / "outputs/cvpr/table3/v1/3a"
OUTPUT_ROOT = ROOT / "outputs/cvpr/table3/v1/3b"
TASKS = pool.TASKS
SEEDS = pool.SEEDS
TOLERANCE = 1e-6
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(*parts: Any) -> int:
    value = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    p15._write_jsonl(path, rows)


def _rms(value: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(value))))


def _native_legalize(
    actions: np.ndarray,
    processor: Any,
    bounds: Any,
    *,
    action_block: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply only the environment's declared physical Box bounds."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 3 or actions.shape[-1] % int(action_block):
        raise ValueError("actions must be packed [state, block, action_dim]")
    base_dim = int(actions.shape[-1]) // int(action_block)
    physical = processor.inverse_transform(actions.reshape(-1, base_dim)).reshape(
        actions.shape[0], -1, base_dim
    )
    legalized = np.clip(physical, bounds.physical_low, bounds.physical_high)
    normalized = processor.transform(legalized.reshape(-1, base_dim)).reshape(actions.shape)
    return normalized.astype(np.float32), legalized.astype(np.float32)


def _rms_trust_region(
    value: torch.Tensor, reference: torch.Tensor, radius: float
) -> torch.Tensor:
    delta = value - reference
    rms = delta.float().square().mean(dim=(1, 2), keepdim=True).sqrt()
    scale = (float(radius) / rms.clamp_min(float(radius))).clamp(max=1.0)
    return reference + delta * scale.to(dtype=delta.dtype)


def _score_cowm(model: Any, starts: torch.Tensor, goals: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    candidates = actions[:, None]
    return model.get_cost_from_latents(
        starts, goals, candidates, score_horizon_blocks=int(model.action_horizon)
    )[:, 0]


def _score_lewm(
    verifier: Any,
    history_latents: torch.Tensor,
    goal_latent: torch.Tensor,
    history_actions: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    context = {"_phase17_history_actions": history_actions}
    return verifier.score_action_candidates(
        history_latents,
        goal_latent,
        actions[:, None],
        context=context,
        candidate_batch_size=1,
        solver_batch_size=10,
    )[:, 0]


def _lewm_post_opt(
    model_for_gradient: Any,
    verifier: Any,
    history_latents: torch.Tensor,
    goal_latent: torch.Tensor,
    history_actions: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    reference = actions.detach()
    current = reference
    zero_gradients = 0
    for _ in range(2):
        current = current.detach().requires_grad_(True)
        costs = _score_lewm(
            verifier, history_latents, goal_latent, history_actions, current
        )
        gradient = torch.autograd.grad(costs.sum(), current)[0]
        normalized, zero = model_for_gradient._normalize_guidance_gradient(
            gradient, count_zero=True
        )
        zero_gradients += int(zero)
        current = _rms_trust_region(
            current.detach() - 0.01 * normalized, reference, 0.20
        ).detach()
    displacement = current - reference
    rms = displacement.float().square().mean(dim=(1, 2)).sqrt()
    return current, {
        "mode": "post_opt",
        "inner_steps": 2,
        "step_size": 0.01,
        "max_rms_offset": 0.20,
        "gradient_normalization": "per-state RMS one",
        "zero_gradient_state_updates": zero_gradients,
        "raw_action_displacement_rms_mean": float(rms.mean().cpu()),
        "raw_action_displacement_rms_max": float(rms.max().cpu()),
    }


def _matched_random_direction(
    baseline_legal: np.ndarray,
    target_legal: np.ndarray,
    processor: Any,
    bounds: Any,
    *,
    seed: int,
    action_block: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Test one predeclared direction using a fixed bisection and tolerance."""
    target_rms = _rms(target_legal - baseline_legal)
    record: dict[str, Any] = {
        "seed": int(seed),
        "target_rms_normalized_after_native_handling": target_rms,
        "match_tolerance": max(1e-5, target_rms * 1e-3),
        "bisection_steps": 48,
        "status": "unmatchable",
        "actual_rms_normalized_after_native_handling": 0.0,
    }
    if target_rms <= 1e-10:
        record["reason"] = "zero_refinement_displacement"
        return None, record

    rng = np.random.default_rng(seed)
    direction = rng.normal(size=baseline_legal.shape)
    direction_rms = _rms(direction)
    if not np.isfinite(direction_rms) or direction_rms <= 1e-12:
        record["reason"] = "degenerate_predeclared_direction"
        return None, record
    direction = direction / direction_rms

    def legalize_scale(scale: float) -> tuple[np.ndarray, np.ndarray]:
        proposed = baseline_legal + float(scale) * direction
        normalized, physical = _native_legalize(
            proposed[None], processor, bounds, action_block=action_block
        )
        return normalized[0], physical[0]

    low = 0.0
    high = max(target_rms, 1e-4)
    high_norm, high_physical = legalize_scale(high)
    high_actual = _rms(high_norm - baseline_legal)
    while high_actual < target_rms and high < 64.0:
        high *= 2.0
        high_norm, high_physical = legalize_scale(high)
        high_actual = _rms(high_norm - baseline_legal)
    if high_actual + record["match_tolerance"] < target_rms:
        record["reason"] = "predeclared_direction_cannot_reach_target_after_action_bounds"
        return None, record

    best_norm, best_physical = high_norm, high_physical
    best_actual = high_actual
    for _ in range(48):
        middle = (low + high) * 0.5
        candidate_norm, candidate_physical = legalize_scale(middle)
        actual = _rms(candidate_norm - baseline_legal)
        if abs(actual - target_rms) < abs(best_actual - target_rms):
            best_norm, best_physical, best_actual = (
                candidate_norm,
                candidate_physical,
                actual,
            )
        if actual < target_rms:
            low = middle
        else:
            high = middle
    record["actual_rms_normalized_after_native_handling"] = float(best_actual)
    record["actual_rms_physical"] = _rms(
        best_physical
        - _native_legalize(
            baseline_legal[None], processor, bounds, action_block=action_block
        )[1][0]
    )
    if abs(best_actual - target_rms) <= record["match_tolerance"]:
        record["status"] = "matched"
        record["reason"] = None
        return best_norm, record
    record["reason"] = "fixed_bisection_did_not_meet_tolerance"
    return None, record


def _bootstrap_mean(values: np.ndarray, episode_ids: list[Any], seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) != len(episode_ids):
        raise ValueError("bootstrap values and episode IDs differ")
    grouped: dict[str, list[float]] = {}
    for value, episode in zip(values, episode_ids):
        if np.isfinite(value):
            key = json.dumps(episode, ensure_ascii=False, sort_keys=True)
            grouped.setdefault(key, []).append(float(value))
    if not grouped:
        return {"mean": None, "ci95": None, "source_episode_count": 0, "replicates": 10_000}
    clusters = list(grouped.values())
    point = float(np.mean([value for cluster in clusters for value in cluster]))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(clusters), size=(10_000, len(clusters)))
    draws = np.asarray(
        [np.mean([value for index in row for value in clusters[int(index)]]) for row in picks],
        dtype=np.float64,
    )
    return {
        "mean": point,
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "source_episode_count": len(clusters),
        "state_count": int(sum(map(len, clusters))),
        "replicates": 10_000,
    }


def _compare_baseline(reference: list[dict[str, Any]], replayed: list[dict[str, Any]]) -> dict[str, Any]:
    left = {int(row["slot"]): row for row in reference}
    right = {int(row["slot"]): row for row in replayed}
    if left.keys() != right.keys() or len(left) != 50:
        raise RuntimeError("3a candidate-0 replay does not contain the exact 50-state cohort")
    max_abs = 0.0
    comparisons = 0
    for slot in range(50):
        a, b = left[slot], right[slot]
        for key in (
            "native_success_after_budget",
            "success_by_5",
            "success_by_25",
            "effective_valid_length",
        ):
            if a.get(key) != b.get(key):
                raise RuntimeError(f"candidate-0 repeat differs for {key}, slot={slot}")
        if len(a.get("steps", [])) != len(b.get("steps", [])):
            raise RuntimeError(f"candidate-0 repeat length differs at slot {slot}")
        for step_a, step_b in zip(a.get("steps", []), b.get("steps", [])):
            for key in ("terminated", "truncated", "env_success", "predicate_success"):
                if step_a.get(key) != step_b.get(key):
                    raise RuntimeError(
                        f"candidate-0 repeat differs for {key}, slot={slot}"
                    )
            for key in ("current", "goal", "action"):
                va, vb = step_a.get(key), step_b.get(key)
                if va is None or vb is None:
                    if va != vb:
                        raise RuntimeError(
                            f"candidate-0 repeat differs for {key}, slot={slot}"
                        )
                    continue
                xa, xb = np.asarray(va, dtype=np.float64), np.asarray(vb, dtype=np.float64)
                if xa.shape != xb.shape:
                    raise RuntimeError(f"candidate-0 repeat {key} shape differs")
                error = float(np.max(np.abs(xa - xb))) if xa.size else 0.0
                max_abs = max(max_abs, error)
                comparisons += 1
                if not np.allclose(xa, xb, atol=1e-6, rtol=1e-6):
                    raise RuntimeError(
                        f"candidate-0 repeat diverged for {key}, slot={slot}: {error}"
                    )
    return {
        "status": "pass",
        "checked_states": 50,
        "state_array_comparisons": comparisons,
        "max_absolute_difference": max_abs,
        "atol": 1e-6,
        "rtol": 1e-6,
    }


def _replay_candidate(
    *,
    world: Any,
    cfg: Any,
    task: str,
    manifest: CohortManifest,
    actions: np.ndarray,
    session: DatasetEvaluationSession,
    actor_model: Any,
    device: str,
    output_dir: Path,
    branch_state: dict[str, Any],
) -> list[dict[str, Any]]:
    if task == "pusht":
        branch_state = {}
    return p15._run_fixed_candidate(
        cfg=cfg,
        task=task,
        manifest=manifest,
        normalized_actions=actions,
        process=session.process,
        model=actor_model,
        transform=session.transform["pixels"],
        device=device,
        output_dir=output_dir,
        dataset=session.dataset,
        branch_state=branch_state,
        capture_future_latents=False,
        evaluation_session=session,
        reusable_world=world,
    )


def _variant_metrics(
    method: str,
    rows: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    predicted_before: np.ndarray,
    predicted_after: np.ndarray,
    action_displacements: np.ndarray,
    episode_ids: list[Any],
    *,
    seed: int,
    match_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    baseline_by_slot = {int(row["slot"]): row for row in baseline}
    variant_by_slot = {int(row["slot"]): row for row in rows}
    valid_slots = []
    real_improvements = np.full(50, np.nan, dtype=np.float64)
    selected_success = np.full(50, np.nan, dtype=np.float64)
    baseline_success = np.full(50, np.nan, dtype=np.float64)
    misleading = np.zeros(50, dtype=bool)
    pred_improvement = predicted_before - predicted_after
    for slot in range(50):
        a, b = baseline_by_slot[slot], variant_by_slot[slot]
        matched = match_rows is None or bool(match_rows[slot]["status"] == "matched")
        valid = (
            matched
            and a.get("valid_at_25") is True
            and b.get("valid_at_25") is True
            and a.get("physical_cost_at_25") is not None
            and b.get("physical_cost_at_25") is not None
            and np.isfinite(float(a["physical_cost_at_25"]))
            and np.isfinite(float(b["physical_cost_at_25"]))
        )
        if valid:
            valid_slots.append(slot)
            real_improvements[slot] = float(a["physical_cost_at_25"]) - float(
                b["physical_cost_at_25"]
            )
            selected_success[slot] = float(bool(b.get("success_by_25")))
            baseline_success[slot] = float(bool(a.get("success_by_25")))
            misleading[slot] = bool(
                pred_improvement[slot] > TOLERANCE
                and real_improvements[slot] < -TOLERANCE
            )
    paired_difference = real_improvements[valid_slots]
    improvement = _bootstrap_mean(
        paired_difference,
        [episode_ids[slot] for slot in valid_slots],
        seed,
    )
    success_difference = selected_success[valid_slots] - baseline_success[valid_slots]
    return {
        "method": method,
        "matched_control": match_rows is not None,
        "valid_25_state_count": int(len(valid_slots)),
        "unmatched_state_count": (
            int(sum(row["status"] != "matched" for row in match_rows))
            if match_rows is not None
            else 0
        ),
        "early_terminal_count": int(
            sum(not bool(variant_by_slot[slot].get("valid_at_25")) for slot in range(50))
        ),
        "true_cost_improvement_original_minus_updated": improvement,
        "improved_state_fraction": (
            float(np.mean(paired_difference > TOLERANCE))
            if len(paired_difference)
            else None
        ),
        "predicted_improvement_mean": (
            float(np.nanmean(pred_improvement)) if np.isfinite(pred_improvement).any() else None
        ),
        "misleading_predicted_improvement_worsened_true_cost_fraction": (
            float(np.mean(misleading[valid_slots])) if valid_slots else None
        ),
        "success_rate_change": (
            _bootstrap_mean(success_difference, [episode_ids[slot] for slot in valid_slots], seed + 1)
            if valid_slots
            else None
        ),
        "action_displacement_rms_normalized_mean": (
            float(np.mean(action_displacements)) if len(action_displacements) else None
        ),
        "valid_slots": valid_slots,
        "per_state_true_cost_improvement": [
            float(value) if np.isfinite(value) else None for value in real_improvements
        ],
        "per_state_predicted_cost_improvement": [
            float(value) if np.isfinite(value) else None for value in pred_improvement
        ],
    }


def run(args: argparse.Namespace) -> Path:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.isdigit() or int(visible) not in range(4):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one preflighted physical GPU0-3.")
    if args.device != "cuda:0":
        raise ValueError("Use logical cuda:0 with CUDA_VISIBLE_DEVICES selecting one physical GPU0-3.")
    task, seed = args.task, int(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_root = POOL_ROOT / task / f"seed_{seed}"
    source_acceptance = _read_json(source_root / "acceptance.json")
    if source_acceptance.get("status") != "pass":
        raise RuntimeError(f"3a source pool has not passed acceptance: {source_root}")
    source_summary = _read_json(source_root / "summary.json")
    source_frozen = _read_json(source_root / "frozen_config.json")
    pool_path = source_root / "fixed_pool.npz"
    if _sha256(pool_path) != source_summary.get("fixed_pool_sha256"):
        raise RuntimeError("accepted 3a fixed-pool hash does not match the saved archive")
    dataset_record = source_frozen["dataset"]
    dataset_path = ROOT / dataset_record["path"]
    if (
        not dataset_path.is_file()
        or dataset_path.stat().st_size != int(dataset_record["file_size_bytes"])
        or _sha256(dataset_path) != dataset_record["sha256"]
    ):
        raise RuntimeError("local dataset no longer matches the accepted 3a source identity")
    cohort_record = source_frozen["cohort"]
    cohort_path = ROOT / cohort_record["path"]
    if _sha256(cohort_path) != cohort_record["sha256"]:
        raise RuntimeError("Table 1 cohort file changed after the accepted 3a run")
    for model_key in ("cowm_r4_ab", "table1_lewm"):
        model_record = source_frozen["models"][model_key]
        model_path = ROOT / model_record["path"]
        if not model_path.is_file() or _sha256(model_path) != model_record["sha256"]:
            raise RuntimeError(f"{model_key} checkpoint changed after the accepted 3a run")
    root = Path(args.output_root).resolve() / task / f"seed_{seed}"
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(
            f"refinement output already has artifacts; preserve it and use a new output root: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)

    actor_path = pool._table1_actor_path(task)
    actor, resolved_actor = pool.load_policy_or_model(str(actor_path), cache_dir=str(ROOT / "data"))
    if resolved_actor is None:
        raise RuntimeError(f"could not resolve Table 1 CoWM checkpoint: {actor_path}")
    expected_actor_path = (ROOT / source_frozen["models"]["cowm_r4_ab"]["path"]).resolve()
    if Path(resolved_actor).resolve() != expected_actor_path:
        raise RuntimeError("loaded CoWM checkpoint path differs from the accepted 3a source")
    actor_model = getattr(actor, "model", actor).to(args.device).eval().requires_grad_(False)
    verifier, lewm_path, history_size = pool._load_table1_lewm_verifier(
        task, actor_model, args.device
    )
    expected_lewm_path = (ROOT / source_frozen["models"]["table1_lewm"]["path"]).resolve()
    if Path(lewm_path).resolve() != expected_lewm_path:
        raise RuntimeError("loaded LeWM checkpoint path differs from the accepted 3a source")
    cfg = pool._resolved_config(task, seed, args.device)
    if getattr(actor_model, "task_condition_dim", None) not in (None, 0):
        raise RuntimeError("3b currently requires the frozen Table 1 R4-AB task_condition_dim=null.")
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    manifest = CohortManifest.load(pool.TABLE1 / f"cohorts/{task}/seed_{seed}.json")
    if len(manifest.entries) != 50:
        raise RuntimeError("Table 3b requires the registered 50-state Table 1 cohort.")
    if manifest.computed_sha256 != source_frozen["cohort"]["cohort_sha256"]:
        raise RuntimeError("loaded Table 1 cohort differs from the accepted 3a source identity")
    session = DatasetEvaluationSession(
        cfg, task=task, dataset=dataset, cohort=manifest.to_evaluation_cohort()
    )
    action_processor = session.process.get("action")
    if action_processor is None:
        raise RuntimeError("Table 3b requires the Table 1 evaluation action StandardScaler.")
    normalizer_record = pool._action_normalizer_record(action_processor)
    if normalizer_record["sha256"] != source_frozen["action_normalizer"]["sha256"]:
        raise RuntimeError("Table 1 action normalizer changed after the accepted 3a run")

    branch_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.update(branch_cfg, "eval.eval_budget", 25, force_add=True)
    OmegaConf.update(branch_cfg, "world.max_episode_steps", 50, force_add=True)
    world_cfg = OmegaConf.to_container(branch_cfg.world, resolve=True)
    world_cfg["max_episode_steps"] = 50
    replay_world = session.world_factory(**world_cfg, image_shape=(224, 224))
    initial_state_strategy = source_frozen["protocol"].get(
        "branch_initial_state_strategy", "captured_environment_and_rng_snapshot"
    )
    if initial_state_strategy == "seeded_dataset_reset_with_fresh_branch_snapshot":
        if task != "pusht":
            raise RuntimeError("seeded dataset reset strategy is only supported for PushT")
        environment_seed = int(source_frozen["environment_seed"])
        if environment_seed != int(cfg.seed):
            raise RuntimeError("PushT dataset reset seed differs from the accepted 3a source")
        pool._install_dataset_reset_seed(replay_world, environment_seed)
    envs = getattr(replay_world, "envs", None)
    if envs is None:
        raise RuntimeError("Table 3b environment does not expose action bounds.")
    bounds = compute_normalized_action_bounds(
        envs.single_action_space,
        action_processor,
        action_block=int(branch_cfg.plan_config.action_block),
    )

    with np.load(pool_path, allow_pickle=False) as archive:
        saved = {key: np.asarray(archive[key]) for key in archive.files}
    candidates = saved["candidates"].astype(np.float32, copy=False)
    candidate_noise = saved["candidate_noise"].astype(np.float32, copy=False)
    physical = saved["physical_actions"].astype(np.float32, copy=False)
    if candidates.shape != (50, 64, 5, int(actor_model.action_dim)):
        raise RuntimeError(f"unexpected frozen proposal pool shape: {candidates.shape}")
    if candidate_noise.shape != candidates.shape:
        raise RuntimeError(
            "frozen candidate noise and proposal pool shapes differ: "
            f"{candidate_noise.shape} != {candidates.shape}"
        )
    base = candidates[:, 0]
    z_start = torch.as_tensor(saved["cowm_b_z_start"], device=args.device, dtype=torch.float32)
    z_goal = torch.as_tensor(saved["cowm_b_z_goal"], device=args.device, dtype=torch.float32)
    hist_latents = torch.as_tensor(saved["lewm_history_latents"], device=args.device, dtype=torch.float32)
    lewm_goal = torch.as_tensor(saved["lewm_goal_latent"], device=args.device, dtype=torch.float32)
    hist_actions = torch.as_tensor(saved["lewm_history_actions"], device=args.device, dtype=torch.float32)
    if hist_latents.shape[1] != history_size:
        raise RuntimeError("saved LeWM history and archived Table 1 history length differ.")

    frozen_config = {
        "schema_version": 1,
        "experiment": "cvpr_table3_refinement_v1",
        "task": task,
        "evaluation_seed": seed,
        "source_3a_root": str(source_root.relative_to(ROOT)),
        "source_3a_acceptance_sha256": _sha256(source_root / "acceptance.json"),
        "source_fixed_pool_sha256": _sha256(pool_path),
        "source_candidate_noise_sha256": source_summary["candidate_noise_sha256"],
        "dataset_sha256": source_frozen["dataset"]["sha256"],
        "actor_checkpoint_sha256": source_frozen["models"]["cowm_r4_ab"]["sha256"],
        "lewm_checkpoint_sha256": source_frozen["models"]["table1_lewm"]["sha256"],
        "action_normalizer_sha256": source_frozen["action_normalizer"]["sha256"],
        "cohort_sha256": source_frozen["cohort"]["cohort_sha256"],
        "code": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": _sha256(Path(__file__).resolve()),
            "source_fixed_pool_script_sha256": source_frozen["code"]["sha256"],
            "current_fixed_pool_script_sha256": _sha256(ROOT / "scripts/cvpr_table3_fixed_pool.py"),
        },
        "runtime": {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "cuda_visible_devices": visible,
            "physical_gpu_id": int(visible),
            "logical_device": args.device,
        },
        "protocol": {
            "branch_initial_state_strategy": initial_state_strategy,
            "dataset_reset_seed": (
                int(source_frozen["environment_seed"])
                if initial_state_strategy == "seeded_dataset_reset_with_fresh_branch_snapshot"
                else None
            ),
            "source_candidate_index": 0,
            "candidate_regeneration": {
                "candidate_batch_shape": [50, 64],
                "noise_shape": list(candidate_noise.shape),
                "verification_scope": "all 64 candidates in the original flattened 50x64 batch",
                "rtol": 2e-5,
                "atol": 1e-5,
            },
            "coWM_po": {"inner_steps": 2, "step_size": 0.01, "rms_trust_region": 0.20},
            "leWM_po": {"inner_steps": 2, "step_size": 0.01, "rms_trust_region": 0.20},
            "coWM_gf": {"flow_steps": 2, "integrator": "euler", "one_update_per_step": True, "step_size": 0.01, "rms_trust_region": 0.20},
            "extra_candidate_projection": False,
            "random_controls_per_method": 5,
            "random_direction_selection": "one predeclared seeded Gaussian direction per control and state; no resampling",
            "random_rms_coordinate": "Table 1 normalized action coordinates after physical Box legalisation",
            "random_match_tolerance": "max(1e-5, target_rms*1e-3)",
            "random_scale_solver": "48-step bisection with upper bracket doubled from max(target_rms,1e-4), capped at 64",
            "improvement_tolerance": TOLERANCE,
            "paired_bootstrap": {"unit": "source episode", "replicates": 10_000, "seed_base": 20261005},
        },
        "action_bounds": bounds.metadata(action_dim=int(actor_model.action_dim)),
    }
    pool.write_json(root / "frozen_config.json", frozen_config)

    candidates_tensor = torch.as_tensor(candidates, device=args.device, dtype=torch.float32)
    base_tensor = candidates_tensor[:, 0]
    candidate_noise_tensor = torch.as_tensor(
        candidate_noise, device=args.device, dtype=torch.float32
    )
    noise_tensor = candidate_noise_tensor[:, 0]
    flat_start = (
        z_start[:, None, :]
        .expand(50, 64, z_start.shape[-1])
        .reshape(50 * 64, z_start.shape[-1])
    )
    flat_goal = (
        z_goal[:, None, :]
        .expand(50, 64, z_goal.shape[-1])
        .reshape(50 * 64, z_goal.shape[-1])
    )
    with torch.no_grad():
        regenerated_pool = actor_model.sample_actions(
            flat_start,
            noise=candidate_noise_tensor.reshape(
                50 * 64, candidate_noise_tensor.shape[-2], candidate_noise_tensor.shape[-1]
            ),
            num_steps=2,
            goal_latent=flat_goal,
            integrator="euler",
        )
        regenerated_pool = regenerated_pool.reshape_as(candidates_tensor)
        regeneration_max_abs = float(
            (regenerated_pool - candidates_tensor).abs().max().detach().cpu()
        )
        regeneration_exact = torch.allclose(
            regenerated_pool, candidates_tensor, rtol=2e-5, atol=1e-5
        )
        if not regeneration_exact:
            raise RuntimeError(
                "the 50x64 proposal pool does not reproduce from its frozen initial noise; "
                f"max absolute difference={regeneration_max_abs:.9g}"
            )
        cowm_before = _score_cowm(actor_model, z_start, z_goal, base_tensor).cpu().numpy()
        lewm_before = _score_lewm(
            verifier, hist_latents, lewm_goal, hist_actions, base_tensor
        ).detach().cpu().numpy()
    if not np.allclose(cowm_before, saved["cowm_b_costs"][:, 0], rtol=2e-4, atol=2e-5):
        raise RuntimeError("CoWM-B candidate-0 rescoring differs from the fixed-pool score.")
    if not np.allclose(lewm_before, saved["lewm_costs"][:, 0], rtol=2e-4, atol=2e-5):
        raise RuntimeError("LeWM candidate-0 rescoring differs from the fixed-pool score.")

    with torch.enable_grad():
        cowm_po_tensor = actor_model.post_optimize_actions(
            z_start,
            z_goal,
            base_tensor,
            step_size=0.01,
            inner_steps=2,
            max_rms_offset=0.20,
            score_horizon_blocks=int(actor_model.action_horizon),
        )
        lewm_po_tensor, lewm_po_stats = _lewm_post_opt(
            actor_model, verifier, hist_latents, lewm_goal, hist_actions, base_tensor
        )
        cowm_gf_tensor = actor_model.sample_actions(
            z_start,
            noise=noise_tensor,
            num_steps=2,
            goal_latent=z_goal,
            integrator="euler",
            guidance_mode="guided_flow",
            guidance_step_size=0.01,
            guidance_last_steps=2,
            guidance_inner_steps=1,
            guidance_max_rms_offset=0.20,
            score_horizon_blocks=int(actor_model.action_horizon),
        )
    guided = {
        "cowm_po": cowm_po_tensor.detach().float().cpu().numpy(),
        "lewm_po": lewm_po_tensor.detach().float().cpu().numpy(),
        "cowm_gf": cowm_gf_tensor.detach().float().cpu().numpy(),
    }
    methods = {
        "cowm_po": "cowm_b",
        "lewm_po": "lewm",
        "cowm_gf": "cowm_b",
    }
    raw_rms = {
        name: np.asarray([_rms(row - base[index]) for index, row in enumerate(actions)])
        for name, actions in guided.items()
    }
    baseline_legal, baseline_physical_legal = _native_legalize(
        base, action_processor, bounds, action_block=int(branch_cfg.plan_config.action_block)
    )
    guided_legal, guided_physical = {}, {}
    guided_physical_raw = {}
    for name, actions in guided.items():
        packed_action_dim = actions.shape[-1] // int(branch_cfg.plan_config.action_block)
        guided_physical_raw[name] = action_processor.inverse_transform(
            actions.reshape(-1, packed_action_dim)
        ).reshape(50, -1, packed_action_dim).astype(np.float32)
        guided_legal[name], guided_physical[name] = _native_legalize(
            actions, action_processor, bounds, action_block=int(branch_cfg.plan_config.action_block)
        )

    random_actions: dict[str, np.ndarray] = {}
    random_match: dict[str, list[dict[str, Any]]] = {}
    for name in ("cowm_po", "lewm_po", "cowm_gf"):
        for control_index in range(5):
            key = f"{name}_random_{control_index}"
            batch = baseline_legal.copy()
            records = []
            for slot in range(50):
                direction_seed = _seed(
                    "cvpr_table3_equal_rms_random",
                    task,
                    seed,
                    name,
                    slot,
                    control_index,
                )
                action, record = _matched_random_direction(
                    baseline_legal[slot],
                    guided_legal[name][slot],
                    action_processor,
                    bounds,
                    seed=direction_seed,
                    action_block=int(branch_cfg.plan_config.action_block),
                )
                if action is not None:
                    batch[slot] = action
                record.update(
                    task=task,
                    evaluation_seed=seed,
                    method=name,
                    control_index=control_index,
                    slot=slot,
                    episode_id=manifest.entries[slot].episode_id,
                )
                records.append(record)
            random_actions[key] = batch
            random_match[key] = records

    metrics: dict[str, Any] = {}
    updated_costs: dict[str, np.ndarray] = {}
    for name, actions in guided.items():
        tensor = torch.as_tensor(actions, device=args.device, dtype=torch.float32)
        with torch.no_grad():
            costs = (
                _score_cowm(actor_model, z_start, z_goal, tensor)
                if methods[name] == "cowm_b"
                else _score_lewm(verifier, hist_latents, lewm_goal, hist_actions, tensor)
            )
        updated_costs[name] = costs.cpu().numpy()
        metrics[name] = {
            "predicted_cost_before_mean": float(np.mean(cowm_before if methods[name] == "cowm_b" else lewm_before)),
            "predicted_cost_after_mean": float(np.mean(updated_costs[name])),
            "predicted_cost_decrease_mean": float(np.mean((cowm_before if methods[name] == "cowm_b" else lewm_before) - updated_costs[name])),
            "raw_action_displacement_rms_mean": float(np.mean(raw_rms[name])),
            "environment_legalized_action_displacement_rms_normalized_mean": float(
                np.mean([_rms(guided_legal[name][i] - baseline_legal[i]) for i in range(50)])
            ),
            "environment_legalized_action_displacement_rms_physical_mean": float(
                np.mean(
                    [
                        _rms(guided_physical[name][i] - baseline_physical_legal[i])
                        for i in range(50)
                    ]
                )
            ),
        }
    for name, actions in random_actions.items():
        scorer_name = name.rsplit("_random_", 1)[0]
        scorer = methods[scorer_name]
        tensor = torch.as_tensor(actions, device=args.device, dtype=torch.float32)
        with torch.no_grad():
            costs = (
                _score_cowm(actor_model, z_start, z_goal, tensor)
                if scorer == "cowm_b"
                else _score_lewm(verifier, hist_latents, lewm_goal, hist_actions, tensor)
            )
        updated_costs[name] = costs.cpu().numpy()
        score_before = cowm_before if scorer == "cowm_b" else lewm_before
        metrics[name] = {
            "predicted_cost_before_mean": float(np.mean(score_before)),
            "predicted_cost_after_mean": float(np.mean(updated_costs[name])),
            "predicted_cost_decrease_mean": float(np.mean(score_before - updated_costs[name])),
        }

    source_joint = {
        "candidates": candidates,
        "costs": saved["cowm_b_costs"],
        "noise_sha256": source_summary["candidate_noise_sha256"],
    }
    source_lewm = {
        "costs": saved["lewm_costs"],
        "noise_sha256": source_summary["candidate_noise_sha256"],
    }
    expected = pool._expected_rows(
        task=task,
        manifest=manifest,
        candidate_index=0,
        joint=source_joint,
        lewm=source_lewm,
        physical=physical,
    )
    branch_state: dict[str, Any] = {}
    baseline_episodes = _replay_candidate(
        world=replay_world,
        cfg=branch_cfg,
        task=task,
        manifest=manifest,
        actions=base,
        session=session,
        actor_model=actor_model,
        device=args.device,
        output_dir=root / "branches" / "baseline_replay_check",
        branch_state=branch_state,
    )
    baseline = pool._attach_milestones(expected, baseline_episodes, task=task)
    reference_baseline = _read_jsonl(source_root / "branches" / "candidate_0000.jsonl")
    restore_check = _compare_baseline(reference_baseline, baseline)
    restore_check["replay_strategy"] = initial_state_strategy
    if initial_state_strategy == "seeded_dataset_reset_with_fresh_branch_snapshot":
        restore_check.update(
            {
                "environment_seed": int(source_frozen["environment_seed"]),
                "captured_snapshot_reused": False,
                "same_start_state_replayability": True,
                "dataset_reset_seed_applied": True,
                "rng_restored": False,
            }
        )
    _write_jsonl(root / "baseline_replay_check.jsonl", baseline)
    print(
        f"{task} seed={seed}: candidate-0 restore check passed "
        f"(max abs {restore_check['max_absolute_difference']:.3g})",
        flush=True,
    )

    episode_ids = [entry.episode_id for entry in manifest.entries]
    branch_files: dict[str, str] = {}
    action_archive: dict[str, np.ndarray] = {
        "candidate0_original_normalized": base,
        "candidate0_original_physical": physical[:, 0],
        "candidate0_environment_legalized_normalized": baseline_legal,
        "candidate0_environment_legalized_physical": baseline_physical_legal,
        "cowm_b_predicted_cost_before": cowm_before,
        "lewm_predicted_cost_before": lewm_before,
        "candidate0_noise": candidate_noise[:, 0],
    }
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    branch_state = branch_state or {}
    all_names = list(guided) + list(random_actions)
    for name in all_names:
        print(f"{task} seed={seed}: replaying {name}", flush=True)
        if name in guided:
            actions = guided[name]
            match_rows = None
            action_archive[f"{name}_updated_normalized"] = actions
            action_archive[f"{name}_updated_physical_raw"] = guided_physical_raw[name]
            action_archive[f"{name}_environment_legalized_normalized"] = guided_legal[name]
            action_archive[f"{name}_environment_legalized_physical"] = guided_physical[name]
        else:
            actions = random_actions[name]
            match_rows = random_match[name]
            action_archive[f"{name}_normalized"] = actions
            packed_action_dim = actions.shape[-1] // int(branch_cfg.plan_config.action_block)
            action_archive[f"{name}_physical"] = action_processor.inverse_transform(
                actions.reshape(-1, packed_action_dim)
            ).reshape(50, -1, packed_action_dim).astype(np.float32)
            action_archive[f"{name}_match_status"] = np.asarray(
                [row["status"] for row in match_rows], dtype="U16"
            )
        variant_path = root / "branches" / f"{name}.jsonl"
        episodes = _replay_candidate(
            world=replay_world,
            cfg=branch_cfg,
            task=task,
            manifest=manifest,
            actions=actions,
            session=session,
            actor_model=actor_model,
            device=args.device,
            output_dir=root / "branches" / name,
            branch_state=branch_state,
        )
        records = pool._attach_milestones(expected, episodes, task=task)
        for record in records:
            record["variant"] = name
            record["variant_action_archive_key"] = (
                f"{name}_updated_normalized" if name in guided else f"{name}_normalized"
            )
            record["variant_action_sha256"] = hashlib.sha256(
                np.ascontiguousarray(actions[int(record["slot"])]).tobytes()
            ).hexdigest()
            if match_rows is not None:
                match = match_rows[int(record["slot"])]
                record["random_match"] = match
                record["eligible_for_equal_rms_control"] = match["status"] == "matched"
            else:
                record["eligible_for_equal_rms_control"] = True
        _write_jsonl(variant_path, records)
        branch_files[name] = _sha256(variant_path)
        rows_by_variant[name] = records
        print(f"{task} seed={seed}: completed {name}", flush=True)

        score_before = cowm_before if methods[name.split("_random_")[0]] == "cowm_b" else lewm_before
        score_after = updated_costs[name]
        displacement = (
            raw_rms[name]
            if name in raw_rms
            else np.asarray(
                [
                    _rms(actions[i] - baseline_legal[i])
                    for i in range(len(actions))
                    if match_rows[i]["status"] == "matched"
                ],
                dtype=np.float64,
            )
        )
        metrics[name].update(
            _variant_metrics(
                name,
                records,
                baseline,
                score_before,
                score_after,
                displacement,
                episode_ids,
                seed=20261005 + seed + _seed(task, name) % 10_000,
                match_rows=match_rows,
            )
        )
        if match_rows is not None:
            matched_rows = [row for row in match_rows if row["status"] == "matched"]
            metrics[name]["random_target_rms_normalized_mean"] = (
                float(np.mean([row["target_rms_normalized_after_native_handling"] for row in matched_rows]))
                if matched_rows
                else None
            )
            metrics[name]["random_actual_rms_normalized_mean"] = (
                float(np.mean([row["actual_rms_normalized_after_native_handling"] for row in matched_rows]))
                if matched_rows
                else None
            )
            metrics[name]["random_actual_rms_physical_mean"] = (
                float(np.mean([row["actual_rms_physical"] for row in matched_rows]))
                if matched_rows
                else None
            )

    for method in ("cowm_po", "lewm_po", "cowm_gf"):
        guide_rows = rows_by_variant[method]
        guide_by_slot = {int(row["slot"]): row for row in guide_rows}
        control_names = [f"{method}_random_{index}" for index in range(5)]
        control_by_name = [
            {int(row["slot"]): row for row in rows_by_variant[name]}
            for name in control_names
        ]
        deltas = []
        delta_episodes = []
        directions_per_eligible_state = []
        for slot in range(50):
            guide = guide_by_slot[slot]
            if not (
                guide.get("valid_at_25") is True
                and guide.get("physical_cost_at_25") is not None
                and np.isfinite(float(guide["physical_cost_at_25"]))
            ):
                continue
            random_improvements = []
            baseline_cost = baseline[slot].get("physical_cost_at_25")
            if baseline[slot].get("valid_at_25") is not True or baseline_cost is None:
                continue
            for name, rows in zip(control_names, control_by_name):
                record = rows[slot]
                match = random_match[name][slot]
                cost = record.get("physical_cost_at_25")
                if match["status"] == "matched" and record.get("valid_at_25") is True and cost is not None:
                    random_improvements.append(float(baseline_cost) - float(cost))
            directions_per_eligible_state.append(len(random_improvements))
            if random_improvements:
                guide_improvement = float(baseline_cost) - float(guide["physical_cost_at_25"])
                deltas.append(guide_improvement - float(np.mean(random_improvements)))
                delta_episodes.append(episode_ids[slot])
        direction_count_histogram = {
            str(count): directions_per_eligible_state.count(count)
            for count in range(6)
        }
        metrics[f"{method}_versus_equal_rms_random"] = {
            "matched_state_count": len(deltas),
            "eligible_state_count": len(directions_per_eligible_state),
            "matched_valid_random_directions_per_state": direction_count_histogram,
            "mean_improvement_difference_method_minus_random": _bootstrap_mean(
                np.asarray(deltas), delta_episodes, 20261005 + seed + _seed(task, method) % 10_000
            ),
            "control_direction_count": 5,
            "unmatchable_control_count": int(
                sum(
                    row["status"] != "matched"
                    for control_name in control_names
                    for row in random_match[control_name]
                )
            ),
        }

    action_path = root / "actions.npz"
    with action_path.open("wb") as stream:
        np.savez_compressed(stream, **action_archive)
    action_path_sha = _sha256(action_path)
    summary = {
        "schema_version": 1,
        "task": task,
        "evaluation_seed": seed,
        "candidate_regeneration_check": {
            "status": "pass",
            "candidate_batch_shape": [50, 64],
            "verified_candidate_count": 50 * 64,
            "max_absolute_difference": regeneration_max_abs,
            "rtol": 2e-5,
            "atol": 1e-5,
        },
        "restore_check": restore_check,
        "action_archive_sha256": action_path_sha,
        "branch_file_sha256": branch_files,
        "metrics": metrics,
        "lewm_post_opt_diagnostics": lewm_po_stats,
        "cowm_guidance_diagnostics": dict(getattr(actor_model, "last_guidance_stats", {})),
        "branch_replay_count": len(branch_files),
        "random_controls_requested": 15,
        "random_controls_unmatchable_state_count": int(
            sum(row["status"] != "matched" for values in random_match.values() for row in values)
        ),
    }
    pool.write_json(root / "summary.json", summary)
    pool.write_json(
        root / "acceptance.json",
        {
            "status": "pass",
            "errors": [],
            "checks": {
                "source_3a_pool_passed": True,
                "candidate_0_reproduced_from_frozen_noise": True,
                "coWM_candidate0_cost_recomputed": True,
                "leWM_candidate0_cost_recomputed": True,
                "candidate0_branch_matches_3a": restore_check["status"] == "pass",
                "all_three_corrections_replayed": all(name in branch_files for name in guided),
                "all_15_random_controls_replayed": all(name in branch_files for name in random_actions),
                "each_control_has_50_predeclared_direction_records": all(
                    len(rows) == 50 for rows in random_match.values()
                ),
                "unmatchable_controls_preserved": True,
                "action_archive_sha256_recorded": action_path_sha == _sha256(action_path),
                "extra_candidate_projection_disabled": True,
                "branch_trace_hashes_recorded": len(branch_files) == 18,
                "paired_bootstrap_10000": True,
            },
            "action_archive_sha256": action_path_sha,
            "frozen_config_sha256": _sha256(root / "frozen_config.json"),
            "summary_sha256": _sha256(root / "summary.json"),
        },
    )
    if hasattr(replay_world, "close"):
        replay_world.close()
    return root / "frozen_config.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", choices=SEEDS, required=True, type=int)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        root = Path(args.output_root).resolve() / args.task / f"seed_{args.seed}"
        root.mkdir(parents=True, exist_ok=True)
        with (root / "failures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        raise


if __name__ == "__main__":
    main()
