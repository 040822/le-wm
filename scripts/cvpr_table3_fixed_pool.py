#!/usr/bin/env python3
"""Capture and replay the frozen Table 1 action pools for CVPR Table 3a."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.round5_phase1_5_diagnostics as p15
from source.common.checkpoint import load_policy_or_model
from source.common.remap import load_pretrained_remapped
from source.common.eval import (
    DatasetEvaluationSession,
    EvaluationIdentity,
    get_dataset,
)
from source.common.round3_phase1 import CohortManifest
from source.common.round4_eval import _DiagnosticCaptureComplete, run_round4_evaluation
from source.policy.round5_phase1_7 import LeWMStageBVerifier

TABLE1 = ROOT / "outputs/cvpr/table1/v1"
TASKS = ("tworoom", "pusht", "reacher", "cube")
SEEDS = (42, 100, 2026, 3407, 1234, 4444)
OUTPUT_ROOT = ROOT / "outputs/cvpr/table3/v1/3a"
PROVENANCE_PATH = TABLE1 / "provenance/input_datasets.json"
ACTOR_ASSET_ROOT = TABLE1 / "assets/cowm"
LEWM_ASSET_ROOT = TABLE1 / "assets/lewm"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cohort_identity_counts(entries: tuple[Any, ...]) -> dict[str, int]:
    state_ids = [p15._candidate_state_id(item) for item in entries]
    episode_ids = [
        json.dumps(item.episode_id, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for item in entries
    ]
    return {
        "appearance_count": len(entries),
        "unique_source_state_count": len(set(state_ids)),
        "unique_source_episode_count": len(set(episode_ids)),
        "duplicate_source_episode_appearances": len(episode_ids) - len(set(episode_ids)),
    }


def _install_dataset_reset_seed(world: Any, environment_seed: int) -> Any:
    """Replace dataset resets without an HDF5 seed column by the frozen seed."""
    original_reset = world.reset

    def reset_with_frozen_seed(seed=None, options=None):
        selected_seed = int(environment_seed) if seed is None else seed
        return original_reset(seed=selected_seed, options=options)

    world.reset = reset_with_frozen_seed
    return world


def _seeded_world_factory(world_factory: Any, environment_seed: int):
    def create_world(*args, **kwargs):
        return _install_dataset_reset_seed(
            world_factory(*args, **kwargs), environment_seed
        )

    return create_world


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _capture_pool(
    *,
    task: str,
    name: str,
    actor: object,
    verifier: object | None,
    actor_path: Path,
    manifest: CohortManifest,
    cfg: object,
    dataset: object,
    output_dir: Path,
    device: str,
    candidate_count: int,
    flow_steps: int,
    training_epoch: int,
) -> dict:
    actor_model = getattr(actor, "model", actor)
    count = len(manifest.entries)
    captured: dict = {}

    def callback(event: dict) -> None:
        slots = tuple(int(value) for value in event["replan_indices"])
        candidates = np.asarray(event["candidates"], dtype=np.float32)
        costs = np.asarray(event["costs"], dtype=np.float64)
        noise = event.get("candidate_noise")
        if noise is None:
            raise RuntimeError("candidate pool capture did not expose its initial noise")
        noise = np.asarray(noise, dtype=np.float32).reshape(candidates.shape)
        if candidates.shape != (len(slots), candidate_count, actor_model.action_horizon, actor_model.action_dim):
            raise ValueError(f"unexpected candidate shape: {candidates.shape}")
        if costs.shape != (len(slots), candidate_count):
            raise ValueError(f"unexpected score shape: {costs.shape}")
        if set(slots) != set(range(count)):
            raise ValueError(f"first replan did not include every cohort slot: {slots}")
        ordered_candidates = np.empty((count, *candidates.shape[1:]), dtype=np.float32)
        ordered_costs = np.empty((count, candidate_count), dtype=np.float64)
        ordered_noise = np.empty_like(ordered_candidates)
        context_arrays = {}
        context_values = {}
        for name in ("z_start", "z_goal", "verifier_z_start", "verifier_z_goal"):
            value = event.get(name)
            if value is None:
                continue
            if torch.is_tensor(value):
                value = value.detach().float().cpu().numpy()
            value = np.asarray(value, dtype=np.float32)
            if value.shape[0] != len(slots):
                raise ValueError(f"unexpected {name} batch shape: {value.shape}")
            context_values[name] = value
            context_arrays[name] = np.empty((count, *value.shape[1:]), dtype=np.float32)
        history_actions = None
        history_pixels = None
        context = event.get("context")
        if isinstance(context, dict) and context.get("_phase17_history_actions") is not None:
            value = context["_phase17_history_actions"]
            if torch.is_tensor(value):
                value = value.detach().float().cpu().numpy()
            value = np.asarray(value, dtype=np.float32)
            if value.shape[0] != len(slots):
                raise ValueError(f"unexpected history action batch shape: {value.shape}")
            history_actions = np.empty((count, *value.shape[1:]), dtype=np.float32)
        if isinstance(context, dict) and context.get("_phase17_history_pixels") is not None:
            value = context["_phase17_history_pixels"]
            if torch.is_tensor(value):
                value = value.detach().float().cpu().numpy()
            value = np.asarray(value, dtype=np.float32)
            if value.shape[0] != len(slots):
                raise ValueError(f"unexpected history pixel batch shape: {value.shape}")
            history_pixels = np.empty((count, *value.shape[1:]), dtype=np.float32)
        for row, slot in enumerate(slots):
            ordered_candidates[slot] = candidates[row]
            ordered_costs[slot] = costs[row]
            ordered_noise[slot] = noise[row]
            for name, value in context_values.items():
                context_arrays[name][slot] = value[row]
            if history_actions is not None:
                value = context["_phase17_history_actions"]
                if torch.is_tensor(value):
                    value = value.detach().float().cpu().numpy()
                history_actions[slot] = value[row]
            if history_pixels is not None:
                value = context["_phase17_history_pixels"]
                if torch.is_tensor(value):
                    value = value.detach().float().cpu().numpy()
                history_pixels[slot] = value[row]
        metadata = event.get("event", {})
        captured.update(
            candidates=ordered_candidates,
            costs=ordered_costs,
            noise=ordered_noise,
            noise_sha256=metadata.get("candidate_noise_sha256"),
            slots=list(slots),
            **context_arrays,
            history_actions=history_actions,
            history_pixels=history_pixels,
        )
        raise _DiagnosticCaptureComplete(
            {"scorer": name, "states": count, "candidate_count": candidate_count}
        )

    result = run_round4_evaluation(
        cfg,
        task=task,
        policy_or_model=actor,
        verifier_policy_or_model=verifier,
        verifier_metadata=(
            None
            if verifier is None
            else {
                "kind": (
                    "independent_lewm"
                    if name == "lewm"
                    else f"matched_fast_stage_b_{name}"
                ),
                "checkpoint": str(verifier),
            }
        ),
        mode="P3",
        identity=EvaluationIdentity(
            entrypoint="cvpr_table3_fixed_pool",
            policy_kind=f"fixed_pool_{name}",
            checkpoint=str(actor_path),
            epoch=training_epoch,
            stage=name,
        ),
        manifest=manifest,
        output_dir=output_dir,
        dataset=dataset,
        device=device,
        trace=False,
        candidate_count=candidate_count,
        flow_steps=flow_steps,
        action_flow_steps=flow_steps,
        solver_batch_size=1,
        candidate_batch_size=candidate_count,
        action_flow_integrator="euler",
        action_bound_mode="none",
        allowed_protocol_variants=("legacy",),
        allow_solver_config_override=True,
        diagnostic_callback=callback,
        allow_cohort_seed_mismatch=True,
        allow_evaluation_seed_override=True,
        policy_seed=int(cfg.eval.policy_seed),
    )
    if result.get("status") != "diagnostic_capture" or "candidates" not in captured:
        raise RuntimeError(f"failed to capture {name} candidate pool: {result.get('status')}")
    return captured


def _physical_actions(candidates: np.ndarray, processor, action_block: int) -> np.ndarray:
    count, pool_size, horizon, packed_dim = candidates.shape
    if packed_dim % action_block:
        raise ValueError("packed action width is not divisible by action_block")
    action_dim = packed_dim // action_block
    rows = candidates.reshape(-1, action_dim)
    physical = processor.inverse_transform(rows)
    # Each latent-planner token packs `action_block` primitive environment
    # actions. Preserve both the packed proposal and the replayable primitive
    # action stream.
    return np.asarray(physical, dtype=np.float32).reshape(
        count, pool_size, horizon * action_block, action_dim
    )


def _encode_goal_latents_in_order(dataset, manifest, transform, model, device: str) -> np.ndarray:
    """Read HDF5 goal rows in increasing order, then restore cohort order."""
    rows = np.asarray([int(entry.goal_row_index) for entry in manifest.entries], dtype=np.int64)
    if np.any(rows < 0):
        raise ValueError("fixed-pool cohort contains an invalid goal row")
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order].tolist()
    raw_sorted = dataset.get_row_data(sorted_rows)
    inverse = np.argsort(order, kind="stable")
    pixels = np.asarray(raw_sorted["pixels"])[inverse]
    image_batch = torch.stack([transform(pixel) for pixel in pixels]).to(device)
    model = getattr(model, "model", model).to(device).eval()
    with torch.inference_mode():
        return model.encode_pixels(image_batch).float().detach().cpu().numpy()


def _physical_cost(task: str, current: Any, goal: Any) -> dict[str, float]:
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    goal = np.asarray(goal, dtype=np.float64).reshape(-1)
    if current.shape != goal.shape or not np.isfinite(current).all() or not np.isfinite(goal).all():
        raise ValueError(f"{task}: invalid physical state at the requested milestone")
    delta = current - goal
    if task == "pusht":
        if delta.size < 5:
            raise ValueError("PushT state lacks XY and angle fields")
        position = float(np.linalg.norm(delta[:4]))
        angle = float(abs((delta[4] + np.pi) % (2.0 * np.pi) - np.pi))
        combined = max(position / 20.0, angle / (np.pi / 9.0))
        return {
            "position_l2_pixels": position,
            "wrapped_angle_radians": angle,
            "selection_cost": float(combined),
        }
    physical = float(np.linalg.norm(delta))
    units = {
        "cube": "meters",
        "reacher": "radians",
        "tworoom": "environment_position_units",
    }.get(task, "task_physical_units")
    return {"physical_l2": physical, "selection_cost": physical, "selection_cost_units": units}


def _first_episode_steps(steps: list[dict]) -> list[dict]:
    """Keep only the initial rollout, stopping at its first native terminal event."""
    kept = []
    for step in steps:
        kept.append(step)
        if bool(step.get("terminated", False)) or bool(step.get("truncated", False)):
            break
    return kept


def _attach_milestones(records: list[dict], episodes: list[dict], *, task: str) -> list[dict]:
    by_slot = {int(item["slot"]): item for item in episodes}
    attached = []
    for record in records:
        episode = by_slot[int(record["slot"])]
        steps = _first_episode_steps(list(episode.get("steps", [])))
        events = [
            item.get("env_success") if item.get("env_success") is not None else item.get("predicate_success")
            for item in steps
        ]
        success_events = [value is not None and bool(value) for value in events]
        output = dict(record)
        output["effective_valid_length"] = len(steps)
        output["native_success_after_budget"] = bool(episode.get("success", False))
        output["first_native_success_step"] = next((i + 1 for i, value in enumerate(success_events) if value), None)
        output["success_by_5"] = bool(any(success_events[:5]))
        output["success_by_25"] = bool(any(success_events[:25]))
        output["terminal_step"] = len(steps) if steps and (steps[-1].get("terminated") or steps[-1].get("truncated")) else None
        if steps:
            endpoint = steps[-1]
            output["endpoint_current"] = endpoint.get("current")
            output["endpoint_goal"] = endpoint.get("goal")
            output["endpoint_raw_env_step"] = endpoint.get("raw_env_step")
            output["endpoint_terminated"] = bool(endpoint.get("terminated", False))
            output["endpoint_truncated"] = bool(endpoint.get("truncated", False))
            output["endpoint_env_success"] = endpoint.get("env_success")
            output["endpoint_predicate_success"] = endpoint.get("predicate_success")
            output["endpoint_type"] = (
                "terminated" if output["endpoint_terminated"]
                else "truncated" if output["endpoint_truncated"]
                else "budget_25" if len(steps) == 25
                else "incomplete_nonterminal"
            )
        else:
            output["endpoint_current"] = None
            output["endpoint_goal"] = None
            output["endpoint_raw_env_step"] = None
            output["endpoint_terminated"] = False
            output["endpoint_truncated"] = False
            output["endpoint_env_success"] = None
            output["endpoint_predicate_success"] = None
            output["endpoint_type"] = "missing"
        if steps:
            terminal_components = _physical_cost(task, steps[-1].get("current"), steps[-1].get("goal"))
            output["terminal_selection_cost"] = terminal_components["selection_cost"]
            output["terminal_cost_components"] = terminal_components
        else:
            output["terminal_selection_cost"] = None
            output["terminal_cost_components"] = None
        for target in (5, 25):
            output[f"valid_at_{target}"] = len(steps) >= target
            milestone = steps[target - 1] if len(steps) >= target else None
            if milestone is None:
                output[f"physical_cost_at_{target}"] = None
                output[f"physical_cost_components_at_{target}"] = None
            else:
                components = _physical_cost(task, milestone.get("current"), milestone.get("goal"))
                output[f"physical_cost_at_{target}"] = components["selection_cost"]
                output[f"physical_cost_components_at_{target}"] = components
                output[f"milestone_state_at_{target}"] = {
                    "local_primitive_step": target,
                    "source_raw_env_step": milestone.get("raw_env_step"),
                    "current": milestone.get("current"),
                    "goal": milestone.get("goal"),
                }
        output["outcome_status"] = "completed"
        attached.append(output)
    return attached


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if len(left) < 2 or len(right) != len(left) or not np.isfinite(left).all() or not np.isfinite(right).all():
        return None
    x, y = rankdata(left, method="average"), rankdata(right, method="average")
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return None if denominator <= 0.0 else float(np.dot(x, y) / denominator)


def _rank_metrics(costs: np.ndarray, outcomes: list[dict]) -> dict:
    count, pool_size = costs.shape
    if len(outcomes) != count * pool_size:
        raise ValueError("candidate outcomes do not match the captured pool")
    by_slot: list[list[dict]] = [[] for _ in range(count)]
    for item in outcomes:
        by_slot[int(item["slot"])].append(item)
    rows = []
    for slot in range(count):
        group = sorted(by_slot[slot], key=lambda row: int(row["candidate_index"]))
        if len(group) != pool_size or [int(row["candidate_index"]) for row in group] != list(range(pool_size)):
            raise ValueError(f"slot {slot} has {len(group)} outcomes, expected {pool_size}")
        distance = np.asarray([np.nan if row.get("physical_cost_at_25") is None else row["physical_cost_at_25"] for row in group], dtype=np.float64)
        success25 = np.asarray([row["success_by_25"] for row in group], dtype=bool)
        success5 = np.asarray([row["success_by_5"] for row in group], dtype=bool)
        valid25 = np.asarray([row["valid_at_25"] for row in group], dtype=bool)
        valid5 = np.asarray([row["valid_at_5"] for row in group], dtype=bool)
        order = np.argsort(costs[slot], kind="stable")
        top = int(order[0])
        valid_indices = np.flatnonzero(valid25 & np.isfinite(distance))
        complete = len(valid_indices) == pool_size
        oracle = int(valid_indices[np.argmin(distance[valid_indices])]) if len(valid_indices) else None
        common_top = int(valid_indices[np.argmin(costs[slot, valid_indices])]) if len(valid_indices) else None
        rho = _spearman(costs[slot], distance) if complete else None
        common_rho = _spearman(costs[slot, valid_indices], distance[valid_indices]) if len(valid_indices) > 1 else None
        source_episode = group[0]["episode_id"]
        rows.append(
            {
                "episode_id": source_episode,
                "complete_64_valid_at_25": bool(complete),
                "valid_candidate_count_at_25": int(len(valid_indices)),
                "top1_success_5": float(success5[top]),
                "top1_success_25": float(success25[top]),
                "top1_cost_at_25": float(distance[top]) if valid25[top] and np.isfinite(distance[top]) else None,
                "pool_success_5": float(success5.mean()),
                "pool_success_25": float(success25.mean()),
                "success_oracle_pool_25": bool(success25.any()),
                "physical_oracle_success_25": bool(success25[oracle]) if oracle is not None else None,
                "uniform_random_expected_success_25": float(success25.mean()),
                "uniform_random_expected_cost_at_25": float(np.mean(distance)) if complete else None,
                "complete_64_selected_physical_regret_at_25": float(distance[top] - distance[oracle]) if complete and valid25[top] and oracle is not None else None,
                "complete_64_spearman_cost_vs_physical_at_25": rho,
                "common_valid_subpool_candidate_count": int(len(valid_indices)),
                "common_valid_subpool_selected_success_25": bool(success25[common_top]) if common_top is not None else None,
                "common_valid_subpool_expected_success_25": float(success25[valid_indices].mean()) if len(valid_indices) else None,
                "common_valid_subpool_selected_physical_regret_at_25": float(distance[common_top] - distance[oracle]) if common_top is not None and oracle is not None else None,
                "common_valid_subpool_spearman_cost_vs_physical_at_25": common_rho,
                "uniform_random_expected_cost_in_valid_subpool_at_25": float(np.mean(distance[valid_indices])) if len(valid_indices) else None,
                "valid_fraction_5": float(valid5.mean()),
                "valid_fraction_25": float(valid25.mean()),
            }
        )
    keys = tuple(key for key in rows[0] if key not in {"episode_id", "complete_64_valid_at_25"})
    means = {key: (float(np.mean(values)) if len(values := [float(row[key]) for row in rows if isinstance(row.get(key), (int, float, np.number)) and np.isfinite(row[key])]) else None) for key in keys}
    means["complete_64_valid_state_count"] = int(sum(row["complete_64_valid_at_25"] for row in rows))
    means["complete_64_valid_state_fraction"] = float(means["complete_64_valid_state_count"] / count)
    return {"state_count": count, "candidate_count": pool_size, "mean": means, "by_state": rows}


def _paired_cluster_bootstrap(left: list[dict], right: list[dict], key: str, seed: int, replicates: int = 10_000) -> dict:
    if len(left) != len(right) or any(a["episode_id"] != b["episode_id"] for a, b in zip(left, right)):
        raise ValueError("paired cluster bootstrap requires the same ordered source episodes")
    per_episode: dict[str, list[float]] = {}
    for a, b in zip(left, right):
        x, y = a.get(key), b.get(key)
        if isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number)) and np.isfinite(x) and np.isfinite(y):
            per_episode.setdefault(json.dumps(a["episode_id"], sort_keys=True), []).append(float(x) - float(y))
    if not per_episode:
        return {"n_source_episodes": 0, "n_state_rows": 0, "mean_difference_left_minus_right": None, "ci95": None, "replicates": replicates}
    clusters = list(per_episode.values())
    point = float(np.mean([value for cluster in clusters for value in cluster]))
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
    means = np.empty(replicates, dtype=np.float64)
    for index, draw in enumerate(draw_indices):
        sampled = [value for cluster_index in draw for value in clusters[int(cluster_index)]]
        means[index] = np.mean(sampled)
    return {
        "n_source_episodes": len(clusters),
        "n_state_rows": sum(len(cluster) for cluster in clusters),
        "mean_difference_left_minus_right": point,
        "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "replicates": replicates,
    }


def _load_table1_lewm_verifier(task: str, actor_model: Any, device: str):
    folder = TABLE1 / "assets" / "lewm" / task
    config = json.loads((folder / "config.json").read_text())
    history_size = int(config["predictor"]["num_frames"])
    world_model = load_pretrained_remapped(str(folder), cache_dir=str(ROOT / "data"))
    world_model.to(device).eval().requires_grad_(False)
    verifier = LeWMStageBVerifier(
        world_model,
        history_size=history_size,
        action_dim=int(actor_model.action_dim),
        action_horizon=int(actor_model.action_horizon),
        latent_dim=int(actor_model.latent_dim),
    ).to(device).eval()
    if int(config["action_encoder"]["input_dim"]) != int(actor_model.action_dim):
        raise ValueError(f"{task}: LeWM native action width differs from the Table 1 actor")
    return verifier, folder / "weights.pt", history_size


def _resolved_config(task: str, seed: int, device: str):
    template = TABLE1 / "runs" / f"cowm_p3__main__{task}__seed_42" / "attempt_001" / "resolved_config.yaml"
    if not template.is_file():
        raise FileNotFoundError(f"Table 1 resolved configuration is missing: {template}")
    cfg = OmegaConf.load(template)
    policy_seed = int(seed) + 20_000
    environment_seed = int(seed) + 10_000
    OmegaConf.update(cfg, "eval.num_eval", 50, force_add=True)
    OmegaConf.update(cfg, "eval.goal_offset_steps", 25, force_add=True)
    OmegaConf.update(cfg, "eval.eval_budget", 50, force_add=True)
    OmegaConf.update(cfg, "eval.policy_seed", policy_seed, force_add=True)
    OmegaConf.update(cfg, "seed", environment_seed, force_add=True)
    OmegaConf.update(cfg, "world.num_envs", 50, force_add=True)
    OmegaConf.update(cfg, "world.max_episode_steps", 100, force_add=True)
    OmegaConf.update(cfg, "plan_config.horizon", 5, force_add=True)
    OmegaConf.update(cfg, "plan_config.receding_horizon", 5, force_add=True)
    OmegaConf.update(cfg, "plan_config.action_block", 5, force_add=True)
    OmegaConf.update(cfg, "solver.device", device, force_add=True)
    OmegaConf.update(cfg, "solver.seed", policy_seed, force_add=True)
    OmegaConf.update(cfg, "output.save_video", False, force_add=True)
    dataset_path = ROOT / json.loads((TABLE1 / "provenance/input_datasets.json").read_text())["tasks"][task]["configured_path"]
    OmegaConf.update(cfg, "eval.dataset_name", str(dataset_path), force_add=True)
    OmegaConf.update(cfg, "eval.benchmark_dataset_name", str(dataset_path.relative_to(ROOT)), force_add=True)
    return cfg


def _table1_actor_path(task: str) -> Path:
    matches = sorted((ACTOR_ASSET_ROOT / task / "checkpoints").glob("*.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{task}: expected one archived Table 1 actor checkpoint, found {matches}")
    return matches[0]


def _action_normalizer_record(processor: Any) -> dict[str, Any]:
    if processor is None or not hasattr(processor, "mean_") or not hasattr(processor, "scale_"):
        raise TypeError("Table 1 action preprocessing must be a fitted StandardScaler")
    payload = {
        "class": f"{type(processor).__module__}.{type(processor).__qualname__}",
        "fit_source": "full task action column via DatasetEvaluationSession/fit_eval_processors, matching the archived Table 1 evaluation path and official LeWM eval.py",
        "source_audit": {
            "official_lewm_eval": "https://github.com/lucas-maes/le-wm/blob/main/eval.py",
            "official_lewm_train": "https://github.com/lucas-maes/le-wm/blob/main/train.py",
            "finding": "The official evaluation path fits sklearn StandardScaler on the evaluation dataset action column; the official training helper computes z-score statistics, so this fixed-pool scorer deliberately follows the released evaluation/archived Table 1 contract and records its exact fitted statistics.",
            "local_implementation": "source/common/eval.py::fit_eval_processors",
        },
        "sample_count": int(np.asarray(processor.n_samples_seen_).max()),
        "mean": np.asarray(processor.mean_, dtype=np.float64).tolist(),
        "scale": np.asarray(processor.scale_, dtype=np.float64).tolist(),
        "variance": np.asarray(processor.var_, dtype=np.float64).tolist(),
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _action_interop_audit(candidates: np.ndarray, processor: Any, action_block: int) -> dict[str, Any]:
    packed_dim = int(candidates.shape[-1])
    if packed_dim % int(action_block):
        raise ValueError("packed action width does not match action_block")
    base_dim = packed_dim // int(action_block)
    normalized = candidates.reshape(-1, base_dim).astype(np.float64)
    physical = processor.inverse_transform(normalized)
    restored = processor.transform(physical)
    maximum_error = float(np.max(np.abs(restored - normalized)))
    if maximum_error > 1e-6:
        raise RuntimeError(f"action normalizer round-trip failed: max error {maximum_error}")
    return {
        "status": "pass",
        "candidate_coordinate": "Table 1 dataset StandardScaler coordinates",
        "shared_evaluation_processor_for_proposal_replay_and_LeWM_history": True,
        "physical_action_roundtrip_max_abs_error": maximum_error,
        "action_dim": base_dim,
        "packed_action_width": packed_dim,
        "action_block": int(action_block),
        "native_action_space": "same physical actions; transform applied by the archived Table 1 evaluation processor",
    }


def _freeze_run(
    root: Path,
    *,
    task: str,
    seed: int,
    manifest: CohortManifest,
    actor_path: Path,
    lewm_path: Path,
    history_size: int,
    normalizer: dict[str, Any],
) -> dict[str, Any]:
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))["tasks"][task]
    data_path = ROOT / provenance["configured_path"]
    if not data_path.is_file() or data_path.stat().st_size != int(provenance["bytes"]):
        raise RuntimeError(f"{task}: local Table 1 dataset identity differs from frozen provenance")
    cohort_path = TABLE1 / f"cohorts/{task}/seed_{seed}.json"
    identity_counts = _cohort_identity_counts(manifest.entries)
    payload = {
        "schema_version": 1,
        "experiment": "cvpr_table3_fixed_pool_v1",
        "task": task,
        "evaluation_seed": int(seed),
        "environment_seed": int(seed) + 10_000,
        "policy_seed": int(seed) + 20_000,
        "cohort": {
            "path": str(cohort_path.relative_to(ROOT)),
            "sha256": _sha256(cohort_path),
            "cohort_id": manifest.cohort_id,
            "cohort_sha256": manifest.computed_sha256,
            "entry_count": len(manifest.entries),
            **identity_counts,
            "duplicate_appearance_policy": "retain each Table 1 cohort slot and its original weight; cluster paired bootstrap by source episode",
            "entries": [
                {
                    "slot": slot,
                    "source_state_id": p15._candidate_state_id(item),
                    "episode_id": item.episode_id,
                    "start_step": int(item.start_step),
                    "row_index": int(item.row_index),
                    "goal_row_index": int(item.goal_row_index),
                }
                for slot, item in enumerate(manifest.entries)
            ],
        },
        "dataset": {
            "path": provenance["configured_path"],
            "sha256": provenance["sha256"],
            "file_size_bytes": int(provenance["bytes"]),
            "source_manifest": str(PROVENANCE_PATH.relative_to(ROOT)),
        },
        "models": {
            "cowm_r4_ab": {"path": str(actor_path.relative_to(ROOT)), "sha256": _sha256(actor_path)},
            "table1_lewm": {
                "path": str(lewm_path.relative_to(ROOT)),
                "sha256": _sha256(lewm_path),
                "history_size": int(history_size),
            },
        },
        "action_normalizer": {
            "sha256": normalizer["sha256"],
            "fit_source": normalizer["fit_source"],
            "sample_count": normalizer["sample_count"],
        },
        "protocol": {
            "candidate_count": 64,
            "flow_steps": 2,
            "integrator": "euler",
            "action_bound_mode": "none",
            "primitive_replay_horizon": 25,
            "capture_budget": 50,
            "replay_budget": 25,
            "complete_64_valid25_is_primary_denominator": True,
            "early_success_is_counted_without_fabricating_step25": True,
            "common_valid_subpool_is_reported_separately": True,
            "branch_initial_state_strategy": (
                "seeded_dataset_reset_with_fresh_branch_snapshot"
                if task == "pusht"
                else "captured_environment_and_rng_snapshot"
            ),
            "branch_world_lifecycle": (
                "fresh_world_with_seeded_dataset_reset"
                if task == "pusht"
                else "persistent_world_with_snapshot_restore"
            ),
            "dataset_reset_seed": int(seed) + 10_000 if task == "pusht" else None,
            "success_source": "Round3TraceCollector native env success or frozen Table 1 task predicate",
            "precision": "FP32",
            "autocast": False,
            "tf32": False,
            "torch_compile": False,
        },
        "code": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": _sha256(Path(__file__))},
        "runtime": {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_gpu_id": int(os.environ["CUDA_VISIBLE_DEVICES"]),
            "logical_device": "cuda:0",
        },
    }
    path = root / "frozen_config.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise RuntimeError("frozen fixed-pool configuration changed; use a new output root")
    write_json(path, payload)
    write_json(root / "action_normalizer.json", normalizer)
    return payload


def _expected_rows(
    *, task: str, manifest: CohortManifest, candidate_index: int, joint: dict,
    lewm: dict, physical: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for slot, entry in enumerate(manifest.entries):
        rows.append(
            {
                "task": task,
                "state_id": p15._candidate_state_id(entry),
                "slot": slot,
                "episode_id": entry.episode_id,
                "start_step": int(entry.start_step),
                "row_index": int(entry.row_index),
                "flow_steps": 2,
                "candidate_index": candidate_index,
                "predicted_cost_cowm_b": float(joint["costs"][slot, candidate_index]),
                "predicted_cost_lewm": float(lewm["costs"][slot, candidate_index]),
                "action": joint["candidates"][slot, candidate_index].tolist(),
                "physical_action": physical[slot, candidate_index].tolist(),
                "candidate_noise_sha256": joint["noise_sha256"],
                "outcome_status": "pending",
            }
        )
    return rows


def _replay_candidate(
    *,
    cfg: Any,
    task: str,
    manifest: CohortManifest,
    actions: np.ndarray,
    session: DatasetEvaluationSession,
    actor_model: Any,
    device: str,
    output_dir: Path,
    branch_state: dict[str, Any],
    reusable_world: Any | None = None,
) -> list[dict[str, Any]]:
    # PushT's Pymunk snapshot omits solver state. Dataset resets have no seed
    # column, so use the frozen environment seed and capture a fresh initial
    # snapshot for each branch instead of restoring a prior snapshot.
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
        reusable_world=reusable_world,
    )


def _assert_restore_repeatable(first: list[dict], repeated: list[dict], *, state_limit: int = 5) -> dict[str, Any]:
    left = {int(row["slot"]): row for row in first if int(row["slot"]) < state_limit}
    right = {int(row["slot"]): row for row in repeated if int(row["slot"]) < state_limit}
    if left.keys() != right.keys() or len(left) != state_limit:
        raise RuntimeError("restore repeat check did not include the fixed first states")
    max_abs = 0.0
    comparisons = 0
    for slot in range(state_limit):
        a, b = left[slot], right[slot]
        if a["success"] != b["success"] or len(a["steps"]) != len(b["steps"]):
            raise RuntimeError(f"restore repeat changed success or valid length at slot {slot}")
        for sa, sb in zip(a["steps"], b["steps"]):
            for key in ("terminated", "truncated", "predicate_success", "env_success"):
                if sa.get(key) != sb.get(key):
                    raise RuntimeError(f"restore repeat changed {key} at slot {slot}")
            for key in ("current", "goal", "action"):
                va, vb = sa.get(key), sb.get(key)
                if va is None or vb is None:
                    if va != vb:
                        raise RuntimeError(f"restore repeat changed {key} at slot {slot}")
                    continue
                aa, bb = np.asarray(va, dtype=np.float64), np.asarray(vb, dtype=np.float64)
                if aa.shape != bb.shape:
                    raise RuntimeError(f"restore repeat changed {key} shape at slot {slot}")
                error = float(np.max(np.abs(aa - bb))) if aa.size else 0.0
                max_abs = max(max_abs, error)
                comparisons += 1
                if not np.allclose(aa, bb, atol=1e-6, rtol=1e-6):
                    raise RuntimeError(f"restore repeat diverged for {key} at slot {slot}: {error}")
    return {
        "status": "pass",
        "checked_states": state_limit,
        "checked_primitive_steps": int(sum(len(left[i]["steps"]) for i in left)),
        "state_array_comparisons": comparisons,
        "max_absolute_difference": max_abs,
        "atol": 1e-6,
        "rtol": 1e-6,
        "rng_restored": True,
    }


def run(args: argparse.Namespace) -> Path:
    task, seed = args.task, int(args.seed)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (
        visible_devices is None
        or not visible_devices.isdigit()
        or int(visible_devices) not in range(4)
    ):
        raise ValueError(
            "Table 3 fixed-pool evaluation requires CUDA_VISIBLE_DEVICES to select exactly one preflighted GPU0-3"
        )
    if args.device != "cuda:0":
        raise ValueError("Use logical cuda:0 with CUDA_VISIBLE_DEVICES set to one physical GPU0-3")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cohort_path = TABLE1 / f"cohorts/{task}/seed_{seed}.json"
    manifest = CohortManifest.load(cohort_path)
    if manifest.task != task or manifest.seed != seed or len(manifest.entries) != 50:
        raise ValueError("Table 1 cohort identity/count differs from the frozen fixed-pool protocol")
    identity_counts = _cohort_identity_counts(manifest.entries)
    if identity_counts["unique_source_state_count"] != 50:
        raise ValueError("Table 1 fixed-pool seed must contain 50 unique source states")
    if manifest.protocol_variant != "legacy" or manifest.goal_offset_steps != 25:
        raise ValueError("Table 3 fixed pool requires the registered legacy Table 1 cohort with offset 25")
    actor_path = _table1_actor_path(task)
    lewm_folder = LEWM_ASSET_ROOT / task
    lewm_path = lewm_folder / "weights.pt"
    if not lewm_path.is_file():
        raise FileNotFoundError(lewm_path)

    actor, actor_loaded_path = load_policy_or_model(str(actor_path), cache_dir=str(ROOT / "data"))
    if actor_loaded_path is None:
        raise RuntimeError(f"could not resolve archived Table 1 actor checkpoint {actor_path}")
    actor_path = Path(actor_loaded_path).resolve()
    actor_model = getattr(actor, "model", actor)
    if not all(hasattr(actor_model, name) for name in ("action_horizon", "action_dim", "latent_dim")):
        raise TypeError("Table 1 R4-AB checkpoint did not resolve to the expected CoWM model")
    verifier, resolved_lewm_path, history_size = _load_table1_lewm_verifier(task, actor_model, args.device)
    if Path(resolved_lewm_path).resolve() != lewm_path.resolve():
        raise RuntimeError("loaded LeWM weights differ from the archived Table 1 asset")

    cfg = _resolved_config(task, seed, args.device)
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    root = Path(args.output_root).resolve() / task / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "state_index.csv"
    with identity_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("slot", "source_state_id", "episode_id", "start_step", "row_index", "goal_row_index"))
        for slot, item in enumerate(manifest.entries):
            writer.writerow((slot, p15._candidate_state_id(item), item.episode_id, item.start_step, item.row_index, item.goal_row_index))

    session = DatasetEvaluationSession(
        cfg, task=task, dataset=dataset, cohort=manifest.to_evaluation_cohort()
    )
    if task == "pusht":
        session.world_factory = _seeded_world_factory(
            session.world_factory, int(cfg.seed)
        )
    action_processor = session.process.get("action")
    if action_processor is None:
        raise ValueError("fixed-pool branch replay needs the action processor")
    normalizer = _action_normalizer_record(action_processor)
    frozen = _freeze_run(
        root,
        task=task,
        seed=seed,
        manifest=manifest,
        actor_path=actor_path,
        lewm_path=lewm_path,
        history_size=history_size,
        normalizer=normalizer,
    )

    cowm_b = _capture_pool(
        task=task,
        name="cowm_b",
        actor=actor,
        verifier=None,
        actor_path=actor_path,
        manifest=manifest,
        cfg=cfg,
        dataset=dataset,
        output_dir=root / "cowm_b_capture",
        device=args.device,
        candidate_count=64,
        flow_steps=2,
        training_epoch=10,
    )
    lewm = _capture_pool(
        task=task,
        name="lewm",
        actor=actor,
        verifier=verifier,
        actor_path=actor_path,
        manifest=manifest,
        cfg=cfg,
        dataset=dataset,
        output_dir=root / "lewm_capture",
        device=args.device,
        candidate_count=64,
        flow_steps=2,
        training_epoch=10,
    )
    if not np.array_equal(cowm_b["candidates"], lewm["candidates"]):
        raise RuntimeError("CoWM-B and LeWM did not receive an identical generated action pool")
    if not np.array_equal(cowm_b["noise"], lewm["noise"]):
        raise RuntimeError("CoWM-B and LeWM candidate noise differs")
    if cowm_b["noise_sha256"] != lewm["noise_sha256"]:
        raise RuntimeError("candidate-noise provenance hash differs across scorers")
    if "history_actions" not in lewm or lewm["history_actions"] is None:
        raise RuntimeError("LeWM first-replan history actions were not captured")

    interop = _action_interop_audit(
        cowm_b["candidates"], action_processor, int(cfg.plan_config.action_block)
    )
    physical = _physical_actions(
        cowm_b["candidates"], action_processor, int(cfg.plan_config.action_block)
    )
    write_json(root / "action_interop_audit.json", interop)
    pool_path = root / "fixed_pool.npz"
    temporary = pool_path.with_name(f".{pool_path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            candidates=cowm_b["candidates"],
            physical_actions=physical,
            cowm_b_costs=cowm_b["costs"],
            lewm_costs=lewm["costs"],
            candidate_noise=cowm_b["noise"],
            cowm_b_z_start=cowm_b["z_start"],
            cowm_b_z_goal=cowm_b["z_goal"],
            lewm_history_latents=lewm["verifier_z_start"],
            lewm_goal_latent=lewm["verifier_z_goal"],
            lewm_history_actions=lewm["history_actions"],
            lewm_history_pixels=lewm["history_pixels"],
        )
    temporary.replace(pool_path)

    branch_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.update(branch_cfg, "eval.eval_budget", 25, force_add=True)
    OmegaConf.update(branch_cfg, "world.max_episode_steps", 50, force_add=True)
    replay_world = None
    if task != "pusht":
        world_cfg = OmegaConf.to_container(branch_cfg.world, resolve=True)
        world_cfg["max_episode_steps"] = 50
        replay_world = session.world_factory(**world_cfg, image_shape=(224, 224))
    branch_state: dict[str, Any] = {}
    outcome_root = root / "branches"
    outcome_root.mkdir(parents=True, exist_ok=True)
    branch_files: list[Path] = []
    all_outcomes: list[dict[str, Any]] = []
    failures_path = root / "failures.jsonl"

    # Candidate 0 always runs first to capture the one shared simulator/RNG
    # snapshot, even when a previous attempt already has a completed JSONL.
    first_expected = _expected_rows(
        task=task, manifest=manifest, candidate_index=0, joint=cowm_b, lewm=lewm, physical=physical
    )
    started = time.time()
    try:
        first_episodes = _replay_candidate(
            cfg=branch_cfg,
            task=task,
            manifest=manifest,
            actions=cowm_b["candidates"][:, 0],
            session=session,
            actor_model=actor_model,
            device=args.device,
            output_dir=outcome_root / "candidate_0000_first",
            branch_state=branch_state,
            reusable_world=replay_world,
        )
    except Exception as exc:
        with failures_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"candidate_index": 0, "phase": "initial_snapshot", "error": repr(exc), "traceback": traceback.format_exc()}, ensure_ascii=False) + "\n")
        raise
    repeated_episodes = _replay_candidate(
        cfg=branch_cfg,
        task=task,
        manifest=manifest,
        actions=cowm_b["candidates"][:, 0],
        session=session,
        actor_model=actor_model,
        device=args.device,
        output_dir=outcome_root / "candidate_0000_repeat",
        branch_state=branch_state,
        reusable_world=replay_world,
    )
    restore = _assert_restore_repeatable(
        first_episodes,
        repeated_episodes,
        state_limit=50 if task == "pusht" else 5,
    )
    if task == "pusht":
        restore.update(
            {
                "replay_strategy": "seeded_dataset_reset_with_fresh_branch_snapshot",
                "environment_seed": int(cfg.seed),
                "captured_snapshot_reused": False,
                "same_start_state_replayability": True,
                "dataset_reset_seed_applied": True,
                "rng_restored": False,
            }
        )
    else:
        restore["replay_strategy"] = "captured_environment_and_rng_snapshot"
        restore["branch_world_lifecycle"] = frozen["protocol"]["branch_world_lifecycle"]
    write_json(root / "restore_check.json", restore)
    first_records = _attach_milestones(first_expected, first_episodes, task=task)
    first_path = outcome_root / "candidate_0000.jsonl"
    p15._write_jsonl(first_path, first_records)
    all_outcomes.extend(first_records)
    branch_files.append(first_path)
    print(f"{task} seed={seed}: candidate 1/64 and replayability check passed in {time.time()-started:.1f}s", flush=True)

    for candidate_index in range(1, 64):
        expected = _expected_rows(
            task=task,
            manifest=manifest,
            candidate_index=candidate_index,
            joint=cowm_b,
            lewm=lewm,
            physical=physical,
        )
        branch_path = outcome_root / f"candidate_{candidate_index:04d}.jsonl"
        if p15._branch_file_matches(branch_path, expected):
            records = p15._read_records(branch_path)
            all_outcomes.extend(records)
            branch_files.append(branch_path)
            continue
        started = time.time()
        try:
            episodes = _replay_candidate(
                cfg=branch_cfg,
                task=task,
                manifest=manifest,
                actions=cowm_b["candidates"][:, candidate_index],
                session=session,
                actor_model=actor_model,
                device=args.device,
                output_dir=outcome_root / branch_path.stem,
                branch_state=branch_state,
                reusable_world=replay_world,
            )
            records = _attach_milestones(expected, episodes, task=task)
            p15._write_jsonl(branch_path, records)
        except Exception as exc:
            with failures_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"candidate_index": candidate_index, "phase": "branch_replay", "error": repr(exc), "traceback": traceback.format_exc()}, ensure_ascii=False) + "\n")
            raise
        all_outcomes.extend(records)
        branch_files.append(branch_path)
        if candidate_index % 4 == 3:
            print(f"{task} seed={seed}: completed {candidate_index + 1}/64 candidate branches (last {time.time()-started:.1f}s)", flush=True)

    if len(all_outcomes) != 50 * 64 or len(branch_files) != 64:
        raise RuntimeError(f"incomplete fixed-pool branch set: records={len(all_outcomes)}, files={len(branch_files)}")
    if replay_world is not None:
        replay_world.close()
        replay_world = None
    metrics = {
        "cowm_b": _rank_metrics(cowm_b["costs"], all_outcomes),
        "lewm": _rank_metrics(lewm["costs"], all_outcomes),
    }
    paired = {
        key: _paired_cluster_bootstrap(
            metrics["lewm"]["by_state"],
            metrics["cowm_b"]["by_state"],
            key,
            seed=20261005 + seed,
        )
        for key in (
            "top1_success_25",
            "complete_64_selected_physical_regret_at_25",
            "complete_64_spearman_cost_vs_physical_at_25",
            "common_valid_subpool_selected_physical_regret_at_25",
        )
    }
    branch_sha = {path.name: _sha256(path) for path in branch_files}
    if any(row.get("outcome_status") != "completed" for row in all_outcomes):
        raise RuntimeError("fixed-pool artifact contains a non-completed branch row")
    summary = {
        "schema_version": 1,
        "task": task,
        "evaluation_seed": seed,
        "cohort_id": manifest.cohort_id,
        "cohort_sha256": manifest.computed_sha256,
        "state_count": 50,
        "cohort_identity": {
            **identity_counts,
            "duplicate_appearance_policy": "retain every Table 1 cohort slot at its original weight; paired bootstrap clusters by source episode",
        },
        "candidate_count": 64,
        "flow_steps": 2,
        "integrator": "euler",
        "action_bound_mode": "none",
        "replay_budget": 25,
        "actor_checkpoint": str(actor_path.relative_to(ROOT)),
        "actor_checkpoint_sha256": frozen["models"]["cowm_r4_ab"]["sha256"],
        "scorer": "table1_lewm",
        "verifier_checkpoint": str(lewm_path.relative_to(ROOT)),
        "verifier_checkpoint_sha256": frozen["models"]["table1_lewm"]["sha256"],
        "verifier_history_size": history_size,
        "same_candidate_pool": True,
        "candidate_noise_sha256": cowm_b["noise_sha256"],
        "action_normalizer_sha256": normalizer["sha256"],
        "action_interop_audit": interop,
        "fixed_pool_sha256": _sha256(pool_path),
        "fixed_pool_artifact": str(pool_path.relative_to(ROOT)),
        "branch_file_sha256": branch_sha,
        "outcome_record_count": len(all_outcomes),
        "restore_check": restore,
        "branch_initial_state_strategy": frozen["protocol"]["branch_initial_state_strategy"],
        "branch_world_lifecycle": frozen["protocol"]["branch_world_lifecycle"],
        "metrics": metrics,
        "paired_lewm_minus_cowm_b": paired,
    }
    write_json(root / "summary.json", summary)
    write_json(
        root / "acceptance.json",
        {
            "status": "pass",
            "errors": [],
            "checks": {
                "table1_50_state_appearances": len(manifest.entries) == 50,
                "unique_source_state_ids": identity_counts["unique_source_state_count"] == 50,
                "duplicate_source_episode_appearances_preserved": True,
                "episode_cluster_bootstrap": True,
                "candidate_pool_identical_for_cowm_and_lewm": True,
                "candidate_noise_identical_for_cowm_and_lewm": True,
                "all_64_candidates_replayed_from_shared_snapshot": (
                    task != "pusht" and len(all_outcomes) == 50 * 64
                ),
                "all_64_candidates_replayed_from_seeded_start_state": (
                    task == "pusht" and len(all_outcomes) == 50 * 64
                ),
                "same_start_state_replayability": (
                    task == "pusht" and restore.get("same_start_state_replayability") is True
                ),
                "dataset_reset_seed_applied": (
                    task == "pusht" and restore.get("dataset_reset_seed_applied") is True
                ),
                "every_state_has_candidate_indices_0_through_63": True,
                "restore_repeatability": restore["status"] == "pass",
                "world_lifecycle_matches_protocol": (
                    frozen["protocol"]["branch_world_lifecycle"]
                    == ("fresh_world_with_seeded_dataset_reset" if task == "pusht" else "persistent_world_with_snapshot_restore")
                ),
                "action_coordinate_roundtrip": interop["status"] == "pass",
                "branch_files_sha256_recorded": len(branch_sha) == 64,
                "replay_budget": 25,
                "no_candidate_projection": frozen["protocol"]["action_bound_mode"] == "none",
            },
            "fixed_pool_sha256": _sha256(pool_path),
            "summary_sha256": _sha256(root / "summary.json"),
        },
    )
    print(json.dumps({"status": "ok", "summary": str(root / "summary.json"), "metrics": {name: value["mean"] for name, value in metrics.items()}}, ensure_ascii=False, sort_keys=True), flush=True)
    return root / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
