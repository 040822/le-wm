#!/usr/bin/env python3
"""Compare Joint-B and independent LeWM on one simulator-grounded action pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import binomtest, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.round5_phase1_5_diagnostics as p15
from scripts.round5_phase1_7 import (
    CONFIG_PATH,
    _load_lewm_verifier,
    _load_training_metadata,
    _phase_config,
)
from source.common.checkpoint import load_policy_or_model
from source.common.eval import (
    DatasetEvaluationSession,
    EvaluationIdentity,
    compose_eval_config,
    get_dataset,
)
from source.common.round3_phase1 import CohortManifest
from source.common.round4_eval import _DiagnosticCaptureComplete, run_round4_evaluation


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    action_bound_mode: str = "clip",
    latent_noise_schedule=None,
    allowed_protocol_variants: tuple[str, ...] = ("round3_revised",),
    allow_eval_budget_override: bool = False,
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
        context = event.get("context")
        if isinstance(context, dict) and context.get("_phase17_history_actions") is not None:
            value = context["_phase17_history_actions"]
            if torch.is_tensor(value):
                value = value.detach().float().cpu().numpy()
            value = np.asarray(value, dtype=np.float32)
            if value.shape[0] != len(slots):
                raise ValueError(f"unexpected history action batch shape: {value.shape}")
            history_actions = np.empty((count, *value.shape[1:]), dtype=np.float32)
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
        metadata = event.get("event", {})
        captured.update(
            candidates=ordered_candidates,
            costs=ordered_costs,
            noise=ordered_noise,
            noise_sha256=metadata.get("candidate_noise_sha256"),
            event_metadata=dict(metadata),
            slots=list(slots),
            **context_arrays,
            history_actions=history_actions,
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
            entrypoint="round5_phase1_7_fixed_pool",
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
        action_bound_mode=action_bound_mode,
        diagnostic_callback=callback,
        latent_noise_schedule=latent_noise_schedule,
        allowed_protocol_variants=allowed_protocol_variants,
        allow_cohort_seed_mismatch=False,
        allow_evaluation_seed_override=(manifest.seed != 42),
        allow_eval_budget_override=allow_eval_budget_override,
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


def _attach_milestones(records: list[dict], episodes: list[dict]) -> list[dict]:
    by_slot = {int(item["slot"]): item for item in episodes}
    attached = []
    for record in records:
        episode = by_slot[int(record["slot"])]
        steps = episode.get("steps", [])
        output = dict(record)
        output["effective_valid_length"] = len(steps)
        for target in (5, 25):
            step = p15._milestone_step(steps, target)
            reached = step is not None and int(step.get("raw_env_step", -1)) >= target
            output[f"success_by_{target}"] = bool(
                any(
                    item.get("predicate_success") is True
                    for item in steps
                    if int(item.get("raw_env_step", -1)) <= target
                )
            )
            output[f"distance_at_{target}"] = (
                None if not reached else step.get("distance")
            )
            output[f"valid_at_{target}"] = bool(reached)
        attached.append(output)
    return attached


def _rank_metrics(costs: np.ndarray, outcomes: list[dict], *, seed: int) -> dict:
    count, pool_size = costs.shape
    if len(outcomes) != count * pool_size:
        raise ValueError("candidate outcomes do not match the captured pool")
    by_slot: list[list[dict]] = [[] for _ in range(count)]
    for item in outcomes:
        by_slot[int(item["slot"])].append(item)
    rows = []
    rng = np.random.default_rng(seed)
    random_indices = rng.integers(0, pool_size, size=count)
    for slot in range(count):
        group = sorted(by_slot[slot], key=lambda row: int(row["candidate_index"]))
        if len(group) != pool_size:
            raise ValueError(f"slot {slot} has {len(group)} outcomes, expected {pool_size}")
        distance = np.asarray(
            [np.nan if row.get("distance_at_25") is None else row["distance_at_25"] for row in group],
            dtype=np.float64,
        )
        success25 = np.asarray([row["success_by_25"] for row in group], dtype=bool)
        success5 = np.asarray([row["success_by_5"] for row in group], dtype=bool)
        valid25 = np.asarray([row["valid_at_25"] for row in group], dtype=bool)
        valid5 = np.asarray([row["valid_at_5"] for row in group], dtype=bool)
        finite = np.isfinite(distance) & valid25
        order = np.argsort(costs[slot], kind="stable")
        top = int(order[0])
        topk = order[: min(5, pool_size)]
        pair_correct = []
        for left in range(pool_size):
            for right in range(left + 1, pool_size):
                if not (finite[left] and finite[right]):
                    continue
                predicted_delta = costs[slot, left] - costs[slot, right]
                true_delta = distance[left] - distance[right]
                if predicted_delta == 0 or true_delta == 0:
                    continue
                pair_correct.append((predicted_delta > 0) == (true_delta > 0))
        rho = spearmanr(costs[slot, finite], distance[finite]).statistic if finite.sum() > 1 else np.nan
        oracle = int(np.nanargmin(distance)) if finite.any() else None
        distance5 = np.asarray(
            [np.nan if row.get("distance_at_5") is None else row["distance_at_5"] for row in group],
            dtype=np.float64,
        )
        finite5 = np.isfinite(distance5) & valid5
        rows.append(
            {
                "top1_success_5": float(success5[top]),
                "top1_success_25": float(success25[top]),
                "top1_distance_5": (
                    float(distance5[top]) if np.isfinite(distance5[top]) and valid5[top] else None
                ),
                "top1_distance_25": (
                    float(distance[top]) if np.isfinite(distance[top]) and valid25[top] else None
                ),
                "pool_success_5": float(success5.mean()),
                "pool_success_25": float(success25.mean()),
                "oracle_success_25": float(success25.any()),
                "physical_regret_25": (
                    float(distance[top] - distance[oracle])
                    if oracle is not None and valid25[top] and np.isfinite(distance[top])
                    else None
                ),
                "random_distance_25": (
                    float(distance[random_indices[slot]])
                    if valid25[random_indices[slot]] and np.isfinite(distance[random_indices[slot]])
                    else None
                ),
                "random_success_25": float(success25[random_indices[slot]]),
                "oracle_best_top5": float(oracle in topk) if oracle is not None else None,
                "spearman_cost_distance": float(rho) if np.isfinite(rho) else None,
                "pairwise_sign_accuracy": (
                    float(np.mean(pair_correct)) if pair_correct else None
                ),
                "valid_fraction_5": float(valid5.mean()),
                "valid_fraction_25": float(valid25.mean()),
            }
        )
    keys = tuple(rows[0])
    means = {}
    for key in keys:
        values = np.asarray(
            [np.nan if row[key] is None else row[key] for row in rows], dtype=np.float64
        )
        means[key] = float(np.nanmean(values)) if np.isfinite(values).any() else None
    return {"state_count": count, "candidate_count": pool_size, "mean": means, "by_state": rows}


def _paired_bootstrap(left: np.ndarray, right: np.ndarray, seed: int) -> dict:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if not len(left):
        return {"n": 0, "mean_difference": None, "ci95": None}
    difference = right - left
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(difference), size=(10_000, len(difference)))
    means = difference[draws].mean(axis=1)
    return {
        "n": len(difference),
        "mean_difference": float(difference.mean()),
        "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
    }


def run(args: argparse.Namespace) -> Path:
    phase = _phase_config()
    task = args.task
    if args.training_epoch < 1:
        raise ValueError("--training-epoch must be positive")
    manifest = CohortManifest.load(Path(args.cohort))
    if manifest.task != task:
        raise ValueError("cohort task does not match --task")
    if args.count < 1 or args.count > len(manifest.entries):
        raise ValueError("--count must select a non-empty prefix of the cohort")
    if args.count != len(manifest.entries):
        from dataclasses import replace

        manifest = replace(
            manifest,
            entries=manifest.entries[: args.count],
            cohort_id=f"{manifest.cohort_id}_fixed_pool_{args.count}",
            cohort_sha256=None,
        )

    checkpoint_cfg = phase["checkpoints"]
    actor_path = Path(args.actor_checkpoint or checkpoint_cfg["actor"][task]).resolve()
    scorer_name = args.scorer
    scorer_argument = args.verifier_checkpoint or args.lewm_checkpoint
    if scorer_name == "lewm":
        scorer_path = Path(scorer_argument or checkpoint_cfg["lewm"][task]).resolve()
    else:
        if not scorer_argument:
            raise ValueError(f"--verifier-checkpoint is required for scorer={scorer_name}")
        scorer_path = Path(scorer_argument).resolve()
    actor, resolved_actor = load_policy_or_model(str(actor_path))
    actor_model = getattr(actor, "model", actor)
    if scorer_name == "lewm":
        verifier, resolved_scorer, history_size = _load_lewm_verifier(
            scorer_path, actor_model
        )
    else:
        verifier_policy, resolved_scorer = load_policy_or_model(str(scorer_path))
        verifier = getattr(verifier_policy, "model", verifier_policy)
        if not callable(getattr(verifier, "get_cost_from_latents", None)):
            raise TypeError(f"checkpoint is not a Fast-LeWAM Stage-B scorer: {resolved_scorer}")
        for name in ("latent_dim", "action_dim", "action_horizon"):
            if getattr(verifier, name, None) != getattr(actor_model, name, None):
                raise ValueError(f"actor and scorer must share {name}")
        history_size = None
    actor_meta, actor_meta_path = _load_training_metadata(Path(resolved_actor))
    scorer_meta, scorer_meta_path = _load_training_metadata(Path(resolved_scorer))
    actor_stats = None if actor_meta is None else actor_meta.get("normalizers", {}).get("action")
    scorer_stats = None if scorer_meta is None else scorer_meta.get("normalizers", {}).get("action")
    if (actor_stats is None) != (scorer_stats is None):
        raise ValueError("cannot fairly score one fresh model without both action normalizers")
    if actor_stats is not None and actor_stats != scorer_stats:
        raise ValueError("Joint and verifier train-only action normalizers differ")

    cfg = compose_eval_config(
        task,
        [
            f"eval.num_eval={len(manifest.entries)}",
            f"world.num_envs={len(manifest.entries)}",
            "output.save_video=false",
            f"seed={manifest.seed}",
        ],
    )
    if actor_stats is not None:
        OmegaConf.update(cfg, "eval.action_normalizer_stats", {"action": actor_stats}, force_add=True)
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    root = Path(args.output_root).resolve() / task / manifest.cohort_id
    if scorer_name != "lewm":
        root /= scorer_name
    root.mkdir(parents=True, exist_ok=True)

    joint = _capture_pool(
        task=task,
        name="joint",
        actor=actor,
        verifier=None,
        actor_path=Path(resolved_actor),
        manifest=manifest,
        cfg=cfg,
        dataset=dataset,
        output_dir=root / "joint_capture",
        device=args.device,
        candidate_count=64,
        flow_steps=2,
        training_epoch=args.training_epoch,
    )
    scored = _capture_pool(
        task=task,
        name=scorer_name,
        actor=actor,
        verifier=verifier,
        actor_path=Path(resolved_actor),
        manifest=manifest,
        cfg=cfg,
        dataset=dataset,
        output_dir=root / f"{scorer_name}_capture",
        device=args.device,
        candidate_count=64,
        flow_steps=2,
        training_epoch=args.training_epoch,
    )
    if not np.array_equal(joint["candidates"], scored["candidates"]):
        raise RuntimeError(f"Joint-B and {scorer_name} were not scored on an identical action pool")
    if not np.array_equal(joint["noise"], scored["noise"]):
        raise RuntimeError(f"Joint-B and {scorer_name} candidate noise did not match")
    if joint["noise_sha256"] != scored["noise_sha256"]:
        raise RuntimeError("candidate noise provenance hash changed across scorers")

    session = DatasetEvaluationSession(
        cfg, task=task, dataset=dataset, cohort=manifest.to_evaluation_cohort()
    )
    action_processor = session.process.get("action")
    if action_processor is None:
        raise ValueError("fixed-pool branch replay needs the action processor")
    physical = _physical_actions(
        joint["candidates"], action_processor, int(cfg.plan_config.action_block)
    )
    goal_latents = _encode_goal_latents_in_order(
        dataset,
        manifest,
        session.transform["pixels"],
        actor_model,
        args.device,
    )
    branch_state: dict = {}
    all_outcomes: list[dict] = []
    outcome_root = root / "branches"
    outcome_root.mkdir(parents=True, exist_ok=True)
    for candidate_index in range(64):
        expected = []
        for slot, entry in enumerate(manifest.entries):
            expected.append(
                {
                    "task": task,
                    "state_id": p15._candidate_state_id(entry),
                    "slot": slot,
                    "episode_id": entry.episode_id,
                    "start_step": int(entry.start_step),
                    "row_index": int(entry.row_index),
                    "flow_steps": 2,
                    "candidate_index": candidate_index,
                    "predicted_cost_joint": float(joint["costs"][slot, candidate_index]),
                    f"predicted_cost_{scorer_name}": float(scored["costs"][slot, candidate_index]),
                    "action": joint["candidates"][slot, candidate_index].tolist(),
                    "physical_action": physical[slot, candidate_index].tolist(),
                    "candidate_noise_sha256": joint["noise_sha256"],
                    "outcome_status": "pending",
                }
            )
        branch_path = outcome_root / f"candidate_{candidate_index:04d}.jsonl"
        if p15._branch_file_matches(branch_path, expected):
            records = p15._read_records(branch_path)
            all_outcomes.extend(records)
            continue
        episodes = p15._run_fixed_candidate(
            cfg=cfg,
            task=task,
            manifest=manifest,
            normalized_actions=joint["candidates"][:, candidate_index],
            process=session.process,
            model=actor_model,
            transform=session.transform["pixels"],
            device=args.device,
            output_dir=branch_path.parent / branch_path.stem,
            dataset=dataset,
            branch_state=branch_state,
            goal_latents=goal_latents,
        )
        records = p15._attach_candidate_outcomes(expected, episodes, task=task)
        records = _attach_milestones(records, episodes)
        p15._write_jsonl(branch_path, records)
        all_outcomes.extend(records)
        if (candidate_index + 1) % 8 == 0:
            print(f"completed {candidate_index + 1}/64 candidate branches", flush=True)

    metrics = {
        name: _rank_metrics(capture["costs"], all_outcomes, seed=16027)
        for name, capture in (("joint", joint), (scorer_name, scored))
    }
    jrows = metrics["joint"]["by_state"]
    scorer_rows = metrics[scorer_name]["by_state"]
    paired = {}
    for key in ("top1_success_5", "top1_success_25", "top1_distance_25", "physical_regret_25"):
        jvalues = np.asarray([row[key] for row in jrows], dtype=np.float64)
        scorer_values = np.asarray([row[key] for row in scorer_rows], dtype=np.float64)
        paired[key] = _paired_bootstrap(jvalues, scorer_values, seed=16028)
    joint_success = np.asarray([row["top1_success_25"] for row in jrows], dtype=bool)
    scorer_success = np.asarray([row["top1_success_25"] for row in scorer_rows], dtype=bool)
    gains = int(np.sum(~joint_success & scorer_success))
    losses = int(np.sum(joint_success & ~scorer_success))
    paired["top1_success_25_exact_mcnemar"] = {
        f"{scorer_name}_only_success": gains,
        "joint_only_success": losses,
        "two_sided_p": float(
            binomtest(min(gains, losses), gains + losses, 0.5).pvalue
            if gains + losses
            else 1.0
        ),
    }

    pool_path = root / "fixed_pool.npz"
    temp_pool = pool_path.with_name(f".{pool_path.name}.tmp")
    with temp_pool.open("wb") as stream:
        np.savez_compressed(
            stream,
            candidates=joint["candidates"],
            physical_actions=physical,
            joint_costs=joint["costs"],
            **{f"{scorer_name}_costs": scored["costs"]},
            candidate_noise=joint["noise"],
            joint_z_start=joint["z_start"],
            joint_z_goal=joint["z_goal"],
            verifier_z_start=scored["verifier_z_start"],
            verifier_z_goal=scored["verifier_z_goal"],
            history_actions=(
                np.empty((len(manifest.entries), 0, 0), dtype=np.float32)
                if scored.get("history_actions") is None
                else scored["history_actions"]
            ),
        )
    temp_pool.replace(pool_path)
    summary_path = root / "summary.json"
    summary = {
        "schema_version": 1,
        "task": task,
        "cohort_id": manifest.cohort_id,
        "cohort_sha256": manifest.computed_sha256,
        "state_count": len(manifest.entries),
        "candidate_count": 64,
        "flow_steps": 2,
        "integrator": "euler",
        "action_bound_mode": "clip",
        "actor_checkpoint": str(resolved_actor),
        "actor_checkpoint_sha256": _sha256(Path(resolved_actor)),
        "scorer": scorer_name,
        "verifier_checkpoint": str(resolved_scorer),
        "verifier_checkpoint_sha256": _sha256(Path(resolved_scorer)),
        "actor_metadata": None if actor_meta_path is None else str(actor_meta_path),
        "verifier_metadata": None if scorer_meta_path is None else str(scorer_meta_path),
        "verifier_history_size": history_size,
        "same_candidate_pool": True,
        "candidate_noise_sha256": joint["noise_sha256"],
        "fixed_pool_sha256": _sha256(pool_path),
        "fixed_pool_artifact": str(pool_path),
        "outcome_record_count": len(all_outcomes),
        "metrics": metrics,
        f"paired_comparison_{scorer_name}_minus_joint": paired,
        **({"paired_comparison_lewm_minus_joint": paired} if scorer_name == "lewm" else {}),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "ok", "summary": str(summary_path), "metrics": {name: value["mean"] for name, value in metrics.items()}}, ensure_ascii=False, sort_keys=True), flush=True)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("pusht", "reacher"))
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--training-epoch", type=int, default=10)
    parser.add_argument("--actor-checkpoint")
    parser.add_argument("--lewm-checkpoint")
    parser.add_argument(
        "--scorer",
        choices=("lewm", "recorded_control", "b_only", "recorded_clean", "online_joint"),
        default="lewm",
    )
    parser.add_argument(
        "--verifier-checkpoint",
        help="Fast-LeWAM Stage-B checkpoint, including the online_joint scorer",
    )
    parser.add_argument("--output-root", default="outputs/round5/phase1_7/fixed_pool")
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
