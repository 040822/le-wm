#!/usr/bin/env python3
"""Round 5 Phase1.7 fair action-ranking comparisons."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.checkpoint import load_policy_or_model
from source.common.eval import EvaluationIdentity, compose_eval_config, get_dataset
from source.common.round3_phase1 import CohortEntry, CohortManifest
from source.common.round5_phase1_6 import build_confirmation_cohort
from source.common.round4_eval import run_round4_evaluation
from source.policy.round5_phase1_7 import LeWMStageBVerifier


CONFIG_PATH = ROOT / "config/round5/phase1_7.json"
COHORT_CONFIG_PATH = ROOT / "config/round4/cohort_artifacts.json"


def _phase_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _registered_dev_manifest(task: str) -> Path:
    artifacts = json.loads(COHORT_CONFIG_PATH.read_text(encoding="utf-8"))
    if task in artifacts["tasks"]:
        item = artifacts["tasks"][task]["dev"]
        return (COHORT_CONFIG_PATH.parent / item["path"]).resolve()
    legacy = ROOT / "outputs/round5/phase3_diagnostics" / f"{task}_single_task/cohorts/{task}/legacy_50.json"
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(
        f"no registered cohort for {task}; prepare a fresh Phase1.7 cohort first"
    )


def _fresh_manifest_path(task: str, split: str) -> Path:
    return ROOT / "outputs/round5/phase1_7/cohorts" / task / f"{split}.json"


def _historical_episode_ids(task: str) -> list:
    roots = [
        ROOT / "outputs/round3/phase1/cohorts" / task,
        ROOT / "outputs/round5/phase1_5/cohorts" / task,
        ROOT / "outputs/round5/phase1_6/cohorts" / task,
        ROOT / "outputs/round5/phase3/cohorts" / task,
        ROOT / "outputs/round5/phase3_diagnostics" / f"{task}_single_task/cohorts" / task,
    ]
    ids = {}
    paths = []
    try:
        paths.append(_registered_dev_manifest(task))
    except FileNotFoundError:
        pass
    for root in roots:
        if root.exists():
            paths.extend(root.glob("*.json"))
    for path in paths:
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("task") not in (None, task):
            continue
        for entry in value.get("entries", ()):
            episode = entry.get("episode_id")
            if episode is not None:
                ids[json.dumps(episode, sort_keys=True)] = episode
    return list(ids.values())


def _build_extended_cohort(
    dataset,
    *,
    task: str,
    count: int,
    seed: int,
    goal_offset_steps: int,
    excluded_episode_ids: list,
    split: str,
) -> CohortManifest:
    """Sample episode-disjoint image-goal starts for Phase3 task schemas."""
    episode_column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_ids = np.asarray(dataset.get_col_data(episode_column))
    step_column = next(
        (name for name in ("step_idx", "timestep", "time_idx") if name in dataset.column_names),
        None,
    )
    if step_column is None:
        raise ValueError(f"{task} dataset has no step index column for fixed-offset goals")
    steps = np.asarray(dataset.get_col_data(step_column)).reshape(-1)
    if len(steps) != len(episode_ids):
        raise ValueError("episode and step columns have different lengths")
    rows_by_episode: dict[str, dict[int, int]] = {}
    episode_values: dict[str, object] = {}
    for row, (episode, step) in enumerate(zip(episode_ids, steps)):
        episode_value = episode.item() if isinstance(episode, np.generic) else episode
        step_value = int(step)
        key = json.dumps(episode_value, sort_keys=True)
        rows_by_episode.setdefault(key, {})[step_value] = int(row)
        episode_values[key] = episode_value
    excluded = {json.dumps(value, sort_keys=True) for value in excluded_episode_ids}
    eligible = []
    for key, rows in rows_by_episode.items():
        if key in excluded:
            continue
        valid_starts = [step for step in rows if step + int(goal_offset_steps) in rows]
        if valid_starts:
            eligible.append((key, valid_starts))
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(eligible))
    chosen = []
    for index in order[: int(count)]:
        key, valid_starts = eligible[int(index)]
        start_step = int(valid_starts[int(rng.integers(len(valid_starts)))])
        goal_step = start_step + int(goal_offset_steps)
        chosen.append(
            CohortEntry(
                row_index=rows_by_episode[key][start_step],
                episode_id=episode_values[key],
                start_step=start_step,
                goal_row_index=rows_by_episode[key][goal_step],
                goal_step=goal_step,
                start_distance=None,
                initially_successful=None,
                stratum=None,
                start_state=None,
                goal_state=None,
            )
        )
    if len(chosen) != int(count):
        raise ValueError(f"{task} has only {len(chosen)} eligible episodes; requested {count}")
    return CohortManifest(
        task=task,
        cohort_id=f"{task}_phase1_7_{split}_{int(seed)}_v1",
        cohort_kind="final" if split == "confirmation" else "dev",
        protocol_variant="round3_revised",
        seed=int(seed),
        goal_offset_steps=int(goal_offset_steps),
        entries=tuple(chosen),
        episode_split={
            "final" if split == "confirmation" else "dev": tuple(item.episode_id for item in chosen)
        },
        candidate_counts=(len(eligible),),
        selected_counts=(len(chosen),),
        sampling_rule={
            "algorithm": "uniform_nonhistorical_episode_then_uniform_valid_start",
            "max_one_start_per_episode": True,
            "goal_offset_steps": int(goal_offset_steps),
            "seed": int(seed),
            "historical_episode_exclusion": True,
            "state_distance_fields": "not_available_for_phase3_task_schema",
        },
        diagnostics={
            "raw_episode_count": len(rows_by_episode),
            "excluded_historical_episode_count": len(excluded),
            "eligible_nonhistorical_episode_count": len(eligible),
            "selected_episode_count": len(chosen),
        },
    )


def prepare_cohorts(*, task: str, seed: int, dev_count: int, confirmation_count: int) -> Path:
    if task not in {"cube", "pusht", "reacher", "tworoom", "scene", "humanoid"}:
        raise ValueError("fresh revised cohorts are supported for the six Phase1.7 tasks")
    config = _phase_config()
    cfg = compose_eval_config(
        task,
        ["eval.num_eval=1", "world.num_envs=1", "output.save_video=false"],
    )
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    historical = _historical_episode_ids(task)
    if task in {"cube", "pusht", "reacher", "tworoom"}:
        build = build_confirmation_cohort
        build_kwargs = {
            "goal_offset_steps": int(config["evaluation"]["goal_offset_steps"]),
            "excluded_episode_ids": historical,
        }
    else:
        build = _build_extended_cohort
        build_kwargs = {
            "goal_offset_steps": int(config["evaluation"]["goal_offset_steps"]),
            "excluded_episode_ids": historical,
            "split": "dev",
        }
    dev = build(
        dataset,
        task=task,
        count=int(dev_count),
        seed=int(seed),
        **build_kwargs,
    )
    dev = replace(
        dev,
        cohort_id=f"{task}_phase1_7_dev_{seed}_v1",
        cohort_kind="dev",
        episode_split={"dev": tuple(item.episode_id for item in dev.entries)},
        sampling_rule={
            **dict(dev.sampling_rule),
            "new_phase": "round5_phase1_7",
            "historical_episode_exclusion": True,
        },
        cohort_sha256=None,
    )
    confirmation_exclusions = [*historical, *(item.episode_id for item in dev.entries)]
    confirmation_kwargs = {
        "goal_offset_steps": int(config["evaluation"]["goal_offset_steps"]),
        "excluded_episode_ids": confirmation_exclusions,
    }
    if task not in {"cube", "pusht", "reacher", "tworoom"}:
        confirmation_kwargs["split"] = "confirmation"
    confirmation = build(
        dataset,
        task=task,
        count=int(confirmation_count),
        seed=int(seed) + 1,
        **confirmation_kwargs,
    )
    confirmation = replace(
        confirmation,
        cohort_id=f"{task}_phase1_7_confirmation_{seed + 1}_v1",
        cohort_kind="final",
        # The shared Round3 validator recognizes the locked holdout bucket as
        # ``final``; the Phase1.7 artifact path and report call it confirmation.
        episode_split={"final": tuple(item.episode_id for item in confirmation.entries)},
        sampling_rule={
            **dict(confirmation.sampling_rule),
            "new_phase": "round5_phase1_7",
            "historical_episode_exclusion": True,
            "dev_episode_exclusion": True,
        },
        cohort_sha256=None,
    )
    target = _fresh_manifest_path(task, "dev")
    dev.save(target)
    confirmation.save(_fresh_manifest_path(task, "confirmation"))
    episode_column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    all_episode_ids = np.unique(dataset.get_col_data(episode_column)).tolist()
    held_out = {
        json.dumps(item.episode_id, sort_keys=True)
        for item in (*dev.entries, *confirmation.entries)
    }
    split = {
        "schema_version": 1,
        "task": task,
        "seed": int(seed),
        "dev_cohort": target.as_posix(),
        "dev_sha256": dev.computed_sha256,
        "confirmation_cohort": _fresh_manifest_path(task, "confirmation").as_posix(),
        "confirmation_sha256": confirmation.computed_sha256,
        "historical_episode_count": len(historical),
        "train_episode_ids": [
            item for item in all_episode_ids
            if json.dumps(item, sort_keys=True) not in held_out
        ],
        "dev_episode_ids": [item.episode_id for item in dev.entries],
        "confirmation_episode_ids": [item.episode_id for item in confirmation.entries],
    }
    split_path = target.parent / "episode_split.json"
    split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "task": task,
                "dev_count": len(dev.entries),
                "confirmation_count": len(confirmation.entries),
                "train_episode_count": len(split["train_episode_ids"]),
                "historical_episode_count": len(historical),
                "dev_sha256": dev.computed_sha256,
                "confirmation_sha256": confirmation.computed_sha256,
                "output": str(target.parent),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return target


def _subset_manifest(path: Path, *, count: int) -> CohortManifest:
    original = CohortManifest.load(path)
    if not 1 <= count <= len(original.entries):
        raise ValueError(f"requested {count} starts from a cohort of {len(original.entries)}")
    if count == len(original.entries):
        return original
    return replace(
        original,
        cohort_id=f"{original.cohort_id}_phase1_7_calibration_{count}",
        entries=original.entries[:count],
        cohort_sha256=None,
    )


def _load_lewm_verifier(path: Path, actor_model):
    policy, resolved_path = load_policy_or_model(str(path))
    model = getattr(policy, "model", policy)
    run_dir = resolved_path.parent.parent if resolved_path.parent.name == "checkpoints" else resolved_path.parent
    train_cfg_path = run_dir / "config.yaml"
    if not train_cfg_path.is_file():
        raise FileNotFoundError(f"LeWM training config is missing: {train_cfg_path}")
    train_cfg = OmegaConf.load(train_cfg_path)
    history_size = int(train_cfg.history_size)
    verifier = LeWMStageBVerifier(
        model,
        history_size=history_size,
        action_dim=int(actor_model.action_dim),
        action_horizon=int(actor_model.action_horizon),
        latent_dim=int(actor_model.latent_dim),
    )
    return verifier, resolved_path, history_size


def _load_fast_stage_b_verifier(path: Path, actor_model):
    """Load a matched Fast-LeWAM checkpoint as a separate Stage-B scorer."""
    policy, resolved_path = load_policy_or_model(str(path))
    verifier = getattr(policy, "model", policy)
    if not callable(getattr(verifier, "get_cost_from_latents", None)):
        raise TypeError(f"checkpoint is not a Fast-LeWAM Stage-B scorer: {resolved_path}")
    for name in ("latent_dim", "action_dim", "action_horizon"):
        if getattr(verifier, name, None) != getattr(actor_model, name, None):
            raise ValueError(f"actor and verifier must share {name}")
    return verifier, Path(resolved_path)


def _load_training_metadata(checkpoint_path: Path):
    run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    path = run_dir / "phase1_7_metadata.json"
    if not path.is_file():
        return None, None
    return json.loads(path.read_text(encoding="utf-8")), path


def run_evaluation(args) -> Path:
    config = _phase_config()
    task = args.task
    use_cem = bool(args.cem_300x30)
    cem_config = config["evaluation"]["cem"]
    eval_mode = str(cem_config["mode"]) if use_cem else "P3"
    comparison_id = str(cem_config["id"]) if use_cem else "p3"
    candidate_count = (
        int(cem_config["samples"])
        if use_cem
        else int(config["evaluation"]["candidate_count"])
    )
    flow_steps = (
        int(cem_config["flow_steps"])
        if use_cem
        else int(config["evaluation"]["action_flow_steps"])
    )
    action_flow_steps = flow_steps
    action_bound_mode = (
        str(cem_config["action_bound_mode"])
        if use_cem
        else str(config["evaluation"]["action_bound_mode"])
    )
    cem_protocol = str(cem_config["protocol"]) if use_cem else "not_applicable"
    if use_cem and eval_mode != "P2":
        raise ValueError("Phase1.7 CEM comparison is locked to actor-warm-started P2")
    fast_stage_b_comparisons = {
        "recorded_control",
        "b_only",
        "recorded_clean",
        "online_joint",
    }
    if use_cem and args.comparison not in {"joint", "lewm"}:
        raise ValueError("the Phase1.7 P2 CEM comparison is defined for Joint-B and LeWM")
    if task not in config["evaluation"]["tasks"]:
        raise ValueError(f"unsupported Phase1.7 task: {task}")
    if task in {"scene", "humanoid"}:
        # These task wrappers need the same upstream compatibility patches
        # used by the Phase3 evaluation entry point (scene button state and
        # Phase3 goal/termination fields).
        from source.common.phase3_compat import patch_phase3_environments

        patch_phase3_environments()
    checkpoint_cfg = config["checkpoints"]
    actor_path = Path(args.actor_checkpoint or checkpoint_cfg["actor"][task])
    verifier_argument = args.verifier_checkpoint or args.lewm_checkpoint
    if args.comparison == "lewm":
        verifier_path = Path(verifier_argument or checkpoint_cfg["lewm"][task])
    elif args.comparison in fast_stage_b_comparisons:
        if not verifier_argument:
            raise ValueError(f"--verifier-checkpoint is required for {args.comparison}")
        verifier_path = Path(verifier_argument)
    else:
        verifier_path = None
    for path in (actor_path,):
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.cohort:
        cohort_path = Path(args.cohort)
    elif _fresh_manifest_path(task, "dev").is_file():
        cohort_path = _fresh_manifest_path(task, "dev")
    else:
        cohort_path = _registered_dev_manifest(task)
    manifest = _subset_manifest(cohort_path, count=args.count)
    if manifest.task != task:
        raise ValueError(f"cohort task {manifest.task!r} does not match {task!r}")
    allowed_protocol_variants = ("round3_revised",)
    if args.allow_legacy_humanoid_cohort_audit:
        expected_legacy_path = (
            ROOT / "outputs/round5/phase3/cohorts/humanoid/legacy_50.json"
        ).resolve()
        if (
            task != "humanoid"
            or args.comparison not in {"joint", "lewm"}
            or cohort_path.resolve() != expected_legacy_path
            or manifest.cohort_id != "humanoid_legacy_50_phase3_v1"
            or manifest.protocol_variant != "legacy"
            or len(manifest.entries) != 50
            or args.count != 50
        ):
            raise ValueError(
                "legacy cohort audit is restricted to the registered 50-start "
                "Humanoid Phase3 cohort with its original legacy protocol label"
            )
        allowed_protocol_variants = ("round3_revised", "legacy")

    cfg = compose_eval_config(
        task,
        [
            f"eval.num_eval={args.count}",
            f"world.num_envs={args.count}",
            "output.save_video=false",
            f"seed={manifest.seed}",
        ],
    )
    if use_cem:
        for path, value in (
            ("solver.num_samples", int(cem_config["samples"])),
            ("solver.n_steps", int(cem_config["iterations"])),
            ("solver.topk", int(cem_config["topk"])),
            ("solver.var_scale", float(cem_config["var_scale"])),
        ):
            OmegaConf.update(cfg, path, value, force_add=True)
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    actor, resolved_actor = load_policy_or_model(str(actor_path))
    actor_model = getattr(actor, "model", actor)
    actor_training_metadata, actor_metadata_path = _load_training_metadata(resolved_actor)
    verifier = None
    verifier_history_size = None
    selection_rule = None
    if args.comparison in {"lewm", *fast_stage_b_comparisons}:
        if verifier_path is None or not verifier_path.is_file():
            raise FileNotFoundError(verifier_path)
        if args.comparison == "lewm":
            verifier, verifier_path, verifier_history_size = _load_lewm_verifier(
                verifier_path, actor_model
            )
        else:
            verifier, verifier_path = _load_fast_stage_b_verifier(
                verifier_path, actor_model
            )
            verifier_history_size = None
        verifier_training_metadata, verifier_metadata_path = _load_training_metadata(
            verifier_path
        )
    elif args.comparison == "random":
        selection_rule = "random"
        verifier_training_metadata, verifier_metadata_path = None, None
    else:
        verifier_training_metadata, verifier_metadata_path = None, None

    actor_action_stats = (
        None
        if actor_training_metadata is None
        else actor_training_metadata.get("normalizers", {}).get("action")
    )
    verifier_action_stats = (
        None
        if verifier_training_metadata is None
        else verifier_training_metadata.get("normalizers", {}).get("action")
    )
    if args.comparison in {"lewm", *fast_stage_b_comparisons} and (
        actor_action_stats is not None or verifier_action_stats is not None
    ):
        if actor_action_stats is None or verifier_action_stats is None:
            raise ValueError(
                "a fresh Phase1.7 model cannot be paired with a checkpoint whose "
                "action normalization metadata is unavailable"
            )
        if actor_action_stats != verifier_action_stats:
            raise ValueError(
                "Joint actor and independent LeWM use different train-only action "
                "normalizers; run through the explicit cross-normalizer adapter"
            )
    action_stats = actor_action_stats or verifier_action_stats
    if action_stats is not None:
        OmegaConf.update(
            cfg,
            "eval.action_normalizer_stats",
            {"action": action_stats},
            force_add=True,
        )

    output_dir = Path(args.output_root) / task / manifest.cohort_id
    if use_cem:
        output_dir /= comparison_id
    output_dir /= args.comparison
    output_dir /= f"seed_{manifest.seed}"
    verifier_metadata = None
    if verifier is not None:
        verifier_metadata = {
            "kind": (
                "independent_lewm"
                if args.comparison == "lewm"
                else (
                    "online_finetuned_joint_stage_b"
                    if args.comparison == "online_joint"
                    else f"matched_fast_stage_b_{args.comparison}"
                )
            ),
            "checkpoint": str(verifier_path),
            "history_size": verifier_history_size,
            "candidate_action_space": (
                "actor_warm_started_cem_physical_actions_restandardized"
                if use_cem
                else "shared_actor_pool_physical_actions_restandardized"
            ),
            "training_metadata": (
                None if verifier_metadata_path is None else str(verifier_metadata_path)
            ),
        }
    identity = EvaluationIdentity(
        entrypoint="round5_phase1_7",
        policy_kind=f"{comparison_id}_{args.comparison}",
        checkpoint=str(resolved_actor),
        epoch=args.training_epoch,
        stage=(
            f"{comparison_id}_joint_stage_b"
            if args.comparison == "joint"
            else f"{comparison_id}_{args.comparison}"
        ),
    )
    result = run_round4_evaluation(
        cfg,
        task=task,
        policy_or_model=actor,
        verifier_policy_or_model=verifier,
        verifier_metadata=verifier_metadata,
        mode=eval_mode,
        identity=identity,
        manifest=manifest,
        output_dir=output_dir,
        dataset=dataset,
        device=args.device,
        # Round3's physical predicate/trace schema covers the four core tasks.
        # Phase3 extensions report environment success without those fields.
        trace=task not in {"scene", "humanoid"},
        candidate_count=candidate_count,
        flow_steps=flow_steps,
        action_flow_steps=action_flow_steps,
        solver_batch_size=int(config["evaluation"]["solver_batch_size"]),
        candidate_batch_size=int(config["evaluation"]["candidate_batch_size"]),
        action_flow_integrator=str(config["evaluation"]["integrator"]),
        action_bound_mode=action_bound_mode,
        cem_protocol=cem_protocol,
        proposal_chunk_size=(
            int(cem_config["proposal_chunk_size"]) if use_cem else None
        ),
        allowed_protocol_variants=allowed_protocol_variants,
        selection_rule=selection_rule,
        allow_cohort_seed_mismatch=False,
        allow_evaluation_seed_override=(manifest.seed != 42),
    )
    result_path = output_dir / "result.json"
    context = {
        "schema_version": 1,
        "task": task,
        "comparison": args.comparison,
        "mode": eval_mode,
        "candidate_count": candidate_count,
        "cem_protocol": cem_protocol,
        "cem_settings": (
            {
                "iterations": int(cem_config["iterations"]),
                "topk": int(cem_config["topk"]),
                "var_scale": float(cem_config["var_scale"]),
                "proposal_chunk_size": int(cem_config["proposal_chunk_size"]),
            }
            if use_cem
            else None
        ),
        "actor_checkpoint": str(resolved_actor),
        "training_epoch": args.training_epoch,
        "actor_training_metadata": (
            None if actor_metadata_path is None else str(actor_metadata_path)
        ),
        "verifier_checkpoint": None if verifier_path is None else str(verifier_path),
        "verifier_training_metadata": (
            None if verifier_metadata_path is None else str(verifier_metadata_path)
        ),
        "action_normalizer_sha256": (
            None
            if action_stats is None
            else hashlib.sha256(
                json.dumps(action_stats, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        ),
    }
    (output_dir / "phase1_7_evaluation_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "task": task,
                "comparison": args.comparison,
                "mode": eval_mode,
                "cohort": str(cohort_path),
                "cohort_sha256": manifest.computed_sha256,
                "count": len(manifest.entries),
                "success_rate": result.get("success_rate"),
                "evaluation_seconds": result.get("evaluation_seconds"),
                "output": str(result_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("cube", "pusht", "reacher", "tworoom", "scene", "humanoid"))
    parser.add_argument(
        "--comparison",
        choices=("joint", "lewm", "random", "recorded_control", "b_only", "recorded_clean", "online_joint"),
    )
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--training-epoch", type=int, default=10)
    parser.add_argument("--cohort")
    parser.add_argument("--actor-checkpoint")
    parser.add_argument("--lewm-checkpoint")
    parser.add_argument(
        "--verifier-checkpoint",
        help="checkpoint for a matched Fast-LeWAM Stage-B scorer, including online_joint",
    )
    parser.add_argument("--output-root", default="outputs/round5/phase1_7/calibration")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-cohorts", action="store_true")
    parser.add_argument("--split-seed", type=int, default=16027)
    parser.add_argument("--dev-count", type=int, default=100)
    parser.add_argument("--confirmation-count", type=int, default=200)
    parser.add_argument(
        "--cem-300x30",
        action="store_true",
        help="run P2 actor-warm-started CEM with 300 samples, 30 iterations, top-30, and cem-clip",
    )
    parser.add_argument(
        "--allow-legacy-humanoid-cohort-audit",
        action="store_true",
        help="allow only the registered Phase3 legacy_50 Humanoid cohort for compatibility auditing",
    )
    args = parser.parse_args()
    if args.training_epoch < 1:
        parser.error("--training-epoch must be positive")
    if args.prepare_cohorts:
        prepare_cohorts(
            task=args.task,
            seed=args.split_seed,
            dev_count=args.dev_count,
            confirmation_count=args.confirmation_count,
        )
        return
    if args.comparison is None:
        parser.error("--comparison is required unless --prepare-cohorts is set")
    run_evaluation(args)


if __name__ == "__main__":
    main()
