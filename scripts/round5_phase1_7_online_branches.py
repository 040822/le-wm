#!/usr/bin/env python3
"""Collect reproducible, train-split branch rollouts for Phase1.7 exploration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.checkpoint import load_policy_or_model
from source.common.eval import (
    DatasetEvaluationSession,
    EvaluationIdentity,
    compose_eval_config,
    get_dataset,
)
from source.common.round3_phase1 import CohortEntry, CohortManifest
from source.common import round3_phase1 as phase1_cohorts
from source.common.round4_action_bounds import compute_normalized_action_bounds
from source.common.round4_eval import run_round4_evaluation
from source.common.round3_validation import validate_cohort_manifest
from scripts import round5_phase1_5_diagnostics as phase15


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_training_metadata(checkpoint: Path) -> tuple[dict[str, Any], Path]:
    run_dir = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    path = run_dir / "phase1_7_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint has no Phase1.7 training metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8")), path


def _build_online_manifest(
    *,
    dataset: Any,
    task: str,
    train_episode_ids: list[Any],
    count: int,
    seed: int,
    cohort_id: str,
) -> tuple[CohortManifest, int]:
    episode_column = phase1_cohorts._episode_column(dataset)
    episode_values = phase1_cohorts._column(dataset, episode_column)
    step_values = np.asarray(phase1_cohorts._column(dataset, "step_idx"), dtype=np.int64).reshape(-1)
    if len(episode_values) != len(step_values):
        raise ValueError("episode and step columns have different lengths")
    state_column, _ = phase1_cohorts._state_columns(dataset, task)
    state_values = phase1_cohorts._column(dataset, state_column)
    lookup = phase1_cohorts._row_lookup(episode_values, step_values)
    train_keys = {phase1_cohorts._episode_key(item) for item in train_episode_ids}
    episode_rows: dict[str, list[int]] = {}
    episode_by_key: dict[str, Any] = {}
    for row, raw_episode in enumerate(episode_values):
        episode_id = phase1_cohorts._jsonable(raw_episode)
        key = phase1_cohorts._episode_key(episode_id)
        if key in train_keys:
            episode_rows.setdefault(key, []).append(int(row))
            episode_by_key[key] = episode_id

    eligible: list[tuple[str, list[int]]] = []
    for key, rows in episode_rows.items():
        valid_rows = [
            row
            for row in rows
            if (key, int(step_values[row]) + 25) in lookup
        ]
        if valid_rows:
            eligible.append((key, valid_rows))
    if len(eligible) < int(count):
        raise ValueError(f"only {len(eligible)} train episodes have a complete 25-step future")

    rng = np.random.default_rng(int(seed))
    episode_order = rng.permutation(len(eligible))[: int(count)]
    entries: list[CohortEntry] = []
    for position in episode_order:
        key, rows = eligible[int(position)]
        row_index = int(rows[int(rng.integers(len(rows)))])
        entry = phase1_cohorts._entry_for_row(
            dataset,
            task=task,
            row_index=row_index,
            goal_offset_steps=25,
            episode_column=episode_column,
            episodes=episode_values,
            steps=step_values,
            lookup=lookup,
            state_values=state_values,
        )
        if entry.goal_row_index is None or entry.goal_step != entry.start_step + 25:
            raise RuntimeError("sampled online state has no exact 25-step future row")
        entries.append(entry)
    entries.sort(key=lambda item: (int(item.row_index), str(item.episode_id)))
    selected_ids = [item.episode_id for item in entries]
    if len({phase1_cohorts._episode_key(item) for item in selected_ids}) != len(selected_ids):
        raise RuntimeError("online cohort repeated a raw training episode")
    manifest = CohortManifest(
        task=task,
        cohort_id=cohort_id,
        cohort_kind="online",
        protocol_variant="round3_revised",
        seed=int(seed),
        goal_offset_steps=25,
        entries=tuple(entries),
        episode_split={"online": tuple(train_episode_ids)},
        sampling_rule={
            "algorithm": "uniform_train_episode_then_uniform_valid_25_step_start",
            "train_episode_split_sha256_verified": True,
            "one_start_per_raw_episode": True,
            "model_or_outcome_dependent_selection": False,
            "goal_offset_steps": 25,
            "seed": int(seed),
        },
        diagnostics={
            "eligible_train_episode_count": len(eligible),
            "selected_episode_count": len(entries),
        },
    )
    validate_cohort_manifest(manifest, task=task, expected_count=int(count))
    return manifest, len(eligible)


def _capture_actor_pool(
    *,
    args: argparse.Namespace,
    task: str,
    manifest: CohortManifest,
    cfg: Any,
    dataset: Any,
    actor: Any,
    checkpoint: Path,
    checkpoint_sha256: str,
) -> tuple[np.ndarray, np.ndarray]:
    captured: dict[str, np.ndarray] = {}

    def capture(event: dict[str, Any]) -> None:
        candidates = torch.as_tensor(event["candidates"]).detach().cpu().numpy()
        noise = event.get("candidate_noise")
        if noise is None:
            raise RuntimeError("actor candidate capture omitted its initial noise")
        noise_array = torch.as_tensor(noise).detach().cpu().numpy()
        if candidates.ndim != 4 or candidates.shape[:2] != (len(manifest.entries), args.actor_candidates):
            raise ValueError(f"unexpected candidate tensor shape: {candidates.shape}")
        if noise_array.shape != (
            candidates.shape[0] * candidates.shape[1],
            candidates.shape[2],
            candidates.shape[3],
        ):
            raise ValueError(f"candidate noise does not align with actions: {noise_array.shape}")
        captured["actor_candidates"] = candidates.astype(np.float32, copy=True)
        captured["candidate_noise"] = noise_array.reshape(candidates.shape).astype(np.float32, copy=True)
        raise phase15._DiagnosticCaptureComplete(
            {"task": task, "states": len(manifest.entries), "candidates": args.actor_candidates}
        )

    result = run_round4_evaluation(
        cfg,
        task=task,
        policy_or_model=actor,
        mode="P3",
        identity=EvaluationIdentity(
            entrypoint="round5_phase1_7_online_branch_capture",
            policy_kind="round4_shared_dit",
            checkpoint=str(checkpoint.resolve()),
            epoch=int(args.training_epoch),
            stage="P3",
        ),
        manifest=manifest,
        output_dir=args.output_dir / "proposal_capture",
        dataset=dataset,
        device=args.device,
        trace=False,
        candidate_count=int(args.actor_candidates),
        flow_steps=2,
        action_flow_steps=2,
        solver_batch_size=1,
        candidate_batch_size=min(32, int(args.actor_candidates)),
        action_flow_integrator="euler",
        action_bound_mode="clip",
        proposal_chunk_size=128,
        allowed_protocol_variants=("round3_revised",),
        allow_variable_candidate_count=True,
        allow_solver_config_override=True,
        allow_evaluation_seed_override=True,
        diagnostic_callback=capture,
    )
    if result.get("status") != "diagnostic_capture" or "actor_candidates" not in captured:
        raise RuntimeError(f"actor proposal capture failed: {result.get('status')}")
    return captured["actor_candidates"], captured["candidate_noise"]


def _action_bounds(
    *,
    cfg: Any,
    task: str,
    dataset: Any,
    manifest: CohortManifest,
    device: str,
) -> tuple[dict[str, Any], Any, Any]:
    session = DatasetEvaluationSession(
        cfg, task=task, dataset=dataset, cohort=manifest.to_evaluation_cohort()
    )
    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    world_cfg["max_episode_steps"] = 2 * int(cfg.eval.eval_budget)
    world = session.world_factory(**world_cfg, image_shape=(224, 224))
    try:
        envs = getattr(world, "envs", None)
        if envs is None:
            raise RuntimeError("online branch world does not expose its action space")
        bounds = compute_normalized_action_bounds(
            envs.single_action_space,
            session.process["action"],
            action_block=int(cfg.plan_config.action_block),
        )
    finally:
        if hasattr(world, "close"):
            world.close()
    return bounds.metadata(), session.process, session.transform["pixels"]


def _make_candidate_pool(
    actor_candidates: np.ndarray,
    candidate_noise: np.ndarray,
    *,
    args: argparse.Namespace,
    normalized_low: np.ndarray,
    normalized_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    values = np.asarray(actor_candidates, dtype=np.float32)
    if values.ndim != 4 or values.shape[1] != int(args.actor_candidates):
        raise ValueError("actor candidate pool does not match --actor-candidates")
    action_block = 5
    action_dim = values.shape[-1] // action_block
    if action_dim < 1 or values.shape[-1] != action_dim * action_block:
        raise ValueError("packed actor action width is not divisible by action_block=5")
    packed_low = np.tile(np.asarray(normalized_low, dtype=np.float32), action_block)
    packed_high = np.tile(np.asarray(normalized_high, dtype=np.float32), action_block)
    values = np.clip(values, packed_low, packed_high)
    center = np.mean(values, axis=1, dtype=np.float64).astype(np.float32)
    spread = np.std(values, axis=1, ddof=1, dtype=np.float64).astype(np.float32)
    local_scale = np.maximum(0.25 * spread, 0.05).astype(np.float32)
    rng = np.random.default_rng(int(args.seed) + 2)
    perturbation_noise = rng.standard_normal(
        (values.shape[0], int(args.local_perturbations), *values.shape[2:])
    ).astype(np.float32)
    local = center[:, None, ...] + perturbation_noise * local_scale[:, None, ...]
    local = np.clip(local, packed_low, packed_high).astype(np.float32)
    pool = np.concatenate((values, local), axis=1)
    source = np.asarray(
        ["actor"] * values.shape[1] + ["local_perturbation"] * local.shape[1],
        dtype="U20",
    )
    pool_sha256 = _sha256_arrays(pool, candidate_noise, perturbation_noise, local_scale)
    return pool, source, perturbation_noise, pool_sha256


def _branch_records_match(
    path: Path,
    future_pixels_path: Path,
    expected: list[dict[str, Any]],
) -> bool:
    if not path.is_file() or not future_pixels_path.is_file():
        return False
    try:
        actual = phase15._read_records(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if len(actual) != len(expected):
        return False
    pixels_sha256 = _sha256_file(future_pixels_path)
    try:
        with np.load(future_pixels_path, allow_pickle=False) as archive:
            if not {"slot", "raw_env_step", "pixels"}.issubset(archive.files):
                return False
            if len(archive["pixels"]) != len(archive["slot"]) or len(archive["pixels"]) == 0:
                return False
    except (OSError, ValueError):
        return False
    by_slot = {int(item.get("slot", -1)): item for item in actual}
    if len(by_slot) != len(actual):
        return False
    for row in expected:
        existing = by_slot.get(int(row["slot"]))
        if existing is None or existing.get("outcome_status") != "completed":
            return False
        for key in ("state_id", "episode_id", "start_step", "row_index", "candidate_index", "candidate_type", "actor_checkpoint_sha256", "candidate_pool_sha256"):
            if existing.get(key) != row.get(key):
                return False
        if existing.get("future_pixels_sha256") != pixels_sha256:
            return False
        if not np.array_equal(
            np.asarray(existing.get("action"), dtype=np.float32),
            np.asarray(row["action"], dtype=np.float32),
        ):
            return False
    return True


def _truncate_episode(episode: dict[str, Any], *, limit: int = 25) -> dict[str, Any]:
    steps = list(episode.get("steps", ()))[: int(limit)]
    valid_length = len(steps)
    for index, step in enumerate(steps):
        if step.get("terminated") or step.get("truncated"):
            valid_length = index + 1
            steps = steps[:valid_length]
            break
    result = dict(episode)
    result["steps"] = steps
    result["steps_executed"] = valid_length
    result["valid_length"] = valid_length
    result["success"] = bool(
        any(step.get("predicate_success") is True or step.get("env_success") is True for step in steps)
    )
    result["termination_reason"] = (
        steps[-1].get("termination_reason") if steps else "no_step"
    )
    if steps:
        result["terminal_distance"] = steps[-1].get("distance")
    result["future_latents"] = {
        key: value for key, value in result.get("future_latents", {}).items() if int(key) <= valid_length
    }
    result["future_latent_costs"] = {
        key: value for key, value in result.get("future_latent_costs", {}).items() if int(key) <= valid_length
    }
    return result


def _expected_rows(
    *,
    manifest: CohortManifest,
    candidates: np.ndarray,
    candidate_index: int,
    candidate_type: str,
    actor_checkpoint_sha256: str,
    pool_sha256: str,
) -> list[dict[str, Any]]:
    rows = []
    for slot, entry in enumerate(manifest.entries):
        rows.append(
            {
                "task": manifest.task,
                "cohort_id": manifest.cohort_id,
                "cohort_sha256": manifest.computed_sha256,
                "state_id": phase15._candidate_state_id(entry),
                "slot": slot,
                "episode_id": entry.episode_id,
                "start_step": int(entry.start_step),
                "row_index": int(entry.row_index),
                "goal_row_index": int(entry.goal_row_index),
                "candidate_index": int(candidate_index),
                "candidate_type": candidate_type,
                "actor_checkpoint_sha256": actor_checkpoint_sha256,
                "candidate_pool_sha256": pool_sha256,
                "action": candidates[slot, candidate_index].astype(np.float32).tolist(),
                "outcome_status": "pending",
            }
        )
    return rows


def _run_candidate(
    *,
    expected: list[dict[str, Any]],
    cfg: Any,
    task: str,
    manifest: CohortManifest,
    dataset: Any,
    process: Any,
    transform: Any,
    actor: Any,
    device: str,
    output_dir: Path,
    branch_state: dict[str, Any],
    future_pixels_path: Path,
    evaluation_session: DatasetEvaluationSession,
    reusable_world: Any,
) -> list[dict[str, Any]]:
    candidate_index = int(expected[0]["candidate_index"])
    actions = np.stack([np.asarray(item["action"], dtype=np.float32) for item in expected])
    episodes = phase15._run_fixed_candidate(
        cfg=cfg,
        task=task,
        manifest=manifest,
        normalized_actions=actions,
        process=process,
        model=actor,
        transform=transform,
        device=device,
        output_dir=output_dir,
        dataset=dataset,
        branch_state=branch_state,
        future_pixels_path=future_pixels_path,
        capture_future_latents=False,
        evaluation_session=evaluation_session,
        reusable_world=reusable_world,
    )
    episode_by_slot = {int(item["slot"]): _truncate_episode(dict(item)) for item in episodes}
    with np.load(future_pixels_path, allow_pickle=False) as archive:
        pixel_slots = np.asarray(archive["slot"], dtype=np.int64)
        pixel_steps = np.asarray(archive["raw_env_step"], dtype=np.int64)
    pixel_steps_by_slot: dict[int, set[int]] = {}
    for slot, step in zip(pixel_slots, pixel_steps):
        pixel_steps_by_slot.setdefault(int(slot), set()).add(int(step))
    records = []
    for row in expected:
        episode = episode_by_slot[int(row["slot"])]
        steps = episode["steps"]
        valid_length = int(episode["valid_length"])
        if valid_length and valid_length not in pixel_steps_by_slot.get(int(row["slot"]), set()):
            raise RuntimeError(
                "branch endpoint image is missing for "
                f"slot={row['slot']} endpoint={valid_length}"
            )
        success = bool(episode["success"])
        terminal = steps[-1] if steps else {}
        records.append(
            {
                **row,
                "success": success,
                "valid_length": valid_length,
                "termination_reason": episode["termination_reason"],
                "terminal_distance": episode.get("terminal_distance"),
                "physical_state": terminal.get("current"),
                "goal_state": terminal.get("goal"),
                "future_pixels_path": str(future_pixels_path.resolve()),
                "future_pixels_sha256": _sha256_file(future_pixels_path),
                "steps": steps,
                "outcome_status": "completed",
            }
        )
    return records


def collect(args: argparse.Namespace) -> Path:
    checkpoint = args.actor_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    metadata, metadata_path = _load_training_metadata(checkpoint)
    if metadata.get("task") != args.task:
        raise ValueError("actor metadata task does not match --task")
    if int(metadata.get("seed", -1)) != 3072:
        raise ValueError("the exploratory first collection is locked to Joint actor seed 3072")
    split_path = Path(metadata["episode_split_manifest"]).expanduser().resolve()
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    split_sha256 = _sha256_file(split_path)
    if split_sha256 != metadata.get("episode_split_sha256"):
        raise ValueError("actor training metadata and episode split manifest SHA256 disagree")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("task") != args.task:
        raise ValueError("episode split manifest task does not match --task")
    train_ids = [phase1_cohorts._jsonable(item) for item in split["train_episode_ids"]]
    train_keys = {phase1_cohorts._episode_key(item) for item in train_ids}
    dev_keys = {phase1_cohorts._episode_key(item) for item in split.get("dev_episode_ids", ())}
    confirmation_keys = {
        phase1_cohorts._episode_key(item) for item in split.get("confirmation_episode_ids", ())
    }
    if train_keys & (dev_keys | confirmation_keys):
        raise ValueError("episode split manifest has train/dev/confirmation overlap")

    checkpoint_sha256 = _sha256_file(checkpoint)
    action_stats = metadata.get("normalizers", {}).get("action")
    if not action_stats:
        raise ValueError("actor metadata has no train-only action normalizer")
    args.output_dir = args.output_root.expanduser().resolve() / args.task / (
        f"actor_seed3072_epoch{int(args.training_epoch)}_{checkpoint_sha256[:12]}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with phase15._exclusive_file_lock(args.output_dir / ".collection.lock"):
        device = phase15._configure_device(
            args.device,
            str(args.gpu),
            minimum_free_mib=int(args.min_free_mib),
            max_load_per_cpu=float(args.max_load_per_cpu),
            minimum_available_mib=int(args.min_available_mib),
            minimum_swap_free_mib=int(args.min_swap_free_mib),
        )
        cohort_id = f"{args.task}_phase1_7_online_actor3072_ep{int(args.training_epoch)}_seed{int(args.seed)}_v1"
        cfg = compose_eval_config(
            args.task,
            overrides=[
                f"eval.num_eval={int(args.count)}",
                f"world.num_envs={int(args.count)}",
                "eval.goal_offset_steps=25",
                "eval.eval_budget=50",
                "plan_config.horizon=5",
                "plan_config.receding_horizon=5",
                "plan_config.action_block=5",
                "output.save_video=false",
                f"seed={int(args.seed)}",
                f"solver.device={args.device}",
            ],
        )
        OmegaConf.update(
            cfg,
            "eval.action_normalizer_stats",
            {"action": action_stats},
            force_add=True,
        )
        dataset = get_dataset(cfg, cfg.eval.dataset_name)
        manifest, eligible_episode_count = _build_online_manifest(
            dataset=dataset,
            task=args.task,
            train_episode_ids=train_ids,
            count=int(args.count),
            seed=int(args.seed),
            cohort_id=cohort_id,
        )
        cohort_path = args.output_dir / "online_train_cohort.json"
        if cohort_path.is_file():
            existing = CohortManifest.load(cohort_path)
            if existing.computed_sha256 != manifest.computed_sha256:
                raise ValueError("existing online train cohort differs from this run")
            manifest = existing
        else:
            manifest.save(cohort_path)

        actor, resolved_actor = load_policy_or_model(str(checkpoint))
        resolved_actor = Path(resolved_actor or checkpoint).resolve()
        if resolved_actor != checkpoint:
            raise ValueError(f"checkpoint resolver changed actor path: {resolved_actor}")
        actor_model = getattr(actor, "model", actor)
        actor_pool_path = args.output_dir / "candidate_pool.npz"
        pool_identity_path = args.output_dir / "candidate_pool_identity.json"
        if actor_pool_path.is_file() and pool_identity_path.is_file():
            identity = json.loads(pool_identity_path.read_text(encoding="utf-8"))
            if identity.get("actor_checkpoint_sha256") != checkpoint_sha256 or identity.get("cohort_sha256") != manifest.computed_sha256:
                raise ValueError("existing candidate pool belongs to a different actor or cohort")
            with np.load(actor_pool_path, allow_pickle=False) as archive:
                actor_candidates = np.asarray(archive["actor_candidates"], dtype=np.float32)
                candidate_noise = np.asarray(archive["candidate_noise"], dtype=np.float32)
        else:
            actor_candidates, candidate_noise = _capture_actor_pool(
                args=args,
                task=args.task,
                manifest=manifest,
                cfg=cfg,
                dataset=dataset,
                actor=actor,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
            )

        bounds_dict, process, transform = _action_bounds(
            cfg=cfg,
            task=args.task,
            dataset=dataset,
            manifest=manifest,
            device=str(device),
        )
        pool, candidate_types, perturbation_noise, pool_sha256 = _make_candidate_pool(
            actor_candidates,
            candidate_noise,
            args=args,
            normalized_low=np.asarray(bounds_dict["normalized_low"], dtype=np.float32),
            normalized_high=np.asarray(bounds_dict["normalized_high"], dtype=np.float32),
        )
        if actor_pool_path.is_file() and pool_identity_path.is_file():
            identity = json.loads(pool_identity_path.read_text(encoding="utf-8"))
            if identity.get("candidate_pool_sha256") != pool_sha256:
                raise ValueError("reconstructed candidate pool hash differs from saved artifact")
        else:
            temporary = actor_pool_path.with_name(f".{actor_pool_path.name}.{os.getpid()}.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    actor_candidates=actor_candidates,
                    candidate_noise=candidate_noise,
                    candidate_pool=pool,
                    candidate_type=candidate_types,
                    perturbation_noise=perturbation_noise,
                    normalized_low=np.asarray(bounds_dict["normalized_low"], dtype=np.float32),
                    normalized_high=np.asarray(bounds_dict["normalized_high"], dtype=np.float32),
                )
            temporary.replace(actor_pool_path)
            _atomic_json(
                pool_identity_path,
                {
                    "actor_checkpoint": str(checkpoint),
                    "actor_checkpoint_sha256": checkpoint_sha256,
                    "cohort_id": manifest.cohort_id,
                    "cohort_sha256": manifest.computed_sha256,
                    "candidate_pool_sha256": pool_sha256,
                    "actor_candidates": int(args.actor_candidates),
                    "local_perturbations": int(args.local_perturbations),
                    "flow_steps": 2,
                    "integrator": "euler",
                    "perturbation_center": "per_state_mean_of_actor_candidates",
                    "perturbation_std_scale": 0.25,
                    "perturbation_std_floor": 0.05,
                    "clip_bounds": bounds_dict,
                    "generation_seed": int(args.seed),
                },
            )

        state_ids = [phase15._candidate_state_id(entry) for entry in manifest.entries]
        split_rng = np.random.default_rng(int(args.seed) + 3)
        split_order = split_rng.permutation(len(state_ids))
        train_state_count = int(round(0.8 * len(state_ids)))
        branch_data_split = {
            "schema_version": 1,
            "seed": int(args.seed) + 3,
            "train_state_ids": [state_ids[int(i)] for i in split_order[:train_state_count]],
            "dev_state_ids": [state_ids[int(i)] for i in split_order[train_state_count:]],
            "selection_before_outcome_inspection": True,
            "joint_and_lewm_share_identical_split": True,
        }
        split_output = args.output_dir / "branch_data_split.json"
        if split_output.is_file():
            existing_split = json.loads(split_output.read_text(encoding="utf-8"))
            if existing_split != branch_data_split:
                raise ValueError("existing train/dev branch split differs from this run")
        else:
            _atomic_json(split_output, branch_data_split)

        branch_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        branch_cfg.eval.eval_budget = 25
        branch_cfg.world.max_episode_steps = 50
        branch_session = DatasetEvaluationSession(
            branch_cfg, task=args.task, dataset=dataset, cohort=manifest.to_evaluation_cohort()
        )
        branch_state: dict[str, Any] = {}
        branch_root = args.output_dir / "branches"
        branch_root.mkdir(parents=True, exist_ok=True)
        manifest_output = args.output_dir / "collection_manifest.json"
        expected_by_candidate: dict[int, tuple[list[dict[str, Any]], Path, Path]] = {}
        completed = 0
        for candidate_index in range(pool.shape[1]):
            candidate_type = str(candidate_types[candidate_index])
            expected = _expected_rows(
                manifest=manifest,
                candidates=pool,
                candidate_index=candidate_index,
                candidate_type=candidate_type,
                actor_checkpoint_sha256=checkpoint_sha256,
                pool_sha256=pool_sha256,
            )
            branch_path = branch_root / f"candidate_{candidate_index:04d}.jsonl"
            pixel_path = branch_root / "future_pixels" / f"candidate_{candidate_index:04d}.npz"
            if _branch_records_match(branch_path, pixel_path, expected):
                completed += 1
            else:
                expected_by_candidate[candidate_index] = (expected, branch_path, pixel_path)

        branch_world = None
        if expected_by_candidate:
            branch_world_cfg = OmegaConf.to_container(branch_cfg.world, resolve=True)
            branch_world_cfg["max_episode_steps"] = 2 * int(branch_cfg.eval.eval_budget)
            branch_world = branch_session.world_factory(**branch_world_cfg, image_shape=(224, 224))
        for candidate_index, (expected, branch_path, pixel_path) in expected_by_candidate.items():
            candidate_type = str(candidate_types[candidate_index])
            records = _run_candidate(
                expected=expected,
                cfg=branch_cfg,
                task=args.task,
                manifest=manifest,
                dataset=dataset,
                process=branch_session.process,
                transform=branch_session.transform["pixels"],
                actor=actor_model,
                device=str(device),
                output_dir=branch_root / f"candidate_{candidate_index:04d}_trace",
                branch_state=branch_state,
                future_pixels_path=pixel_path,
                evaluation_session=branch_session,
                reusable_world=branch_world,
            )
            phase15._write_jsonl(branch_path, records)
            completed += 1
            _atomic_json(
                manifest_output,
                {
                    "schema_version": 1,
                    "status": "partial" if completed < pool.shape[1] else "completed",
                    "task": args.task,
                    "training_split_manifest": str(split_path),
                    "training_split_sha256": split_sha256,
                    "training_metadata": str(metadata_path),
                    "actor_checkpoint": str(checkpoint),
                    "actor_checkpoint_sha256": checkpoint_sha256,
                    "training_epoch": int(args.training_epoch),
                    "cohort": str(cohort_path),
                    "cohort_sha256": manifest.computed_sha256,
                    "eligible_train_episode_count": int(eligible_episode_count),
                    "state_count": len(manifest.entries),
                    "candidate_count": int(pool.shape[1]),
                    "actor_candidate_count": int(args.actor_candidates),
                    "local_perturbation_count": int(args.local_perturbations),
                    "candidate_pool_sha256": pool_sha256,
                    "branch_data_split": str(split_output),
                    "branch_data_train_state_count": len(branch_data_split["train_state_ids"]),
                    "branch_data_dev_state_count": len(branch_data_split["dev_state_ids"]),
                    "max_primitive_steps": 25,
                    "future_pixel_milestones": [5, 10, 15, 20, 25],
                    "completed_candidate_branches": completed,
                    "candidate_type_counts": {
                        "actor": int(np.sum(candidate_types == "actor")),
                        "local_perturbation": int(np.sum(candidate_types == "local_perturbation")),
                    },
                },
            )
            print(
                json.dumps(
                    {"task": args.task, "candidate_branches_completed": completed, "total": int(pool.shape[1])},
                    sort_keys=True,
                ),
                flush=True,
            )
        if branch_world is not None and hasattr(branch_world, "close"):
            branch_world.close()
        if completed == pool.shape[1]:
            _atomic_json(
                manifest_output,
                {
                    "schema_version": 1,
                    "status": "completed",
                    "task": args.task,
                    "training_split_manifest": str(split_path),
                    "training_split_sha256": split_sha256,
                    "training_metadata": str(metadata_path),
                    "actor_checkpoint": str(checkpoint),
                    "actor_checkpoint_sha256": checkpoint_sha256,
                    "training_epoch": int(args.training_epoch),
                    "cohort": str(cohort_path),
                    "cohort_sha256": manifest.computed_sha256,
                    "eligible_train_episode_count": int(eligible_episode_count),
                    "state_count": len(manifest.entries),
                    "candidate_count": int(pool.shape[1]),
                    "candidate_pool_sha256": pool_sha256,
                    "branch_data_split": str(split_output),
                    "branch_data_train_state_count": len(branch_data_split["train_state_ids"]),
                    "branch_data_dev_state_count": len(branch_data_split["dev_state_ids"]),
                    "max_primitive_steps": 25,
                    "future_pixel_milestones": [5, 10, 15, 20, 25],
                    "completed_candidate_branches": completed,
                },
            )
        return manifest_output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("pusht", "reacher"))
    parser.add_argument("--actor-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/round5/phase1_7/online_branches_epoch2_v1")
    parser.add_argument("--training-epoch", type=int, default=2)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--actor-candidates", type=int, default=16)
    parser.add_argument("--local-perturbations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=16045)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", required=True, type=phase15._gpu)
    parser.add_argument("--min-free-mib", type=int, default=2048)
    parser.add_argument("--max-load-per-cpu", type=float, default=phase15.DEFAULT_MAX_LOAD_PER_CPU)
    parser.add_argument("--min-available-mib", type=int, default=phase15.DEFAULT_MIN_AVAILABLE_MIB)
    parser.add_argument("--min-swap-free-mib", type=int, default=phase15.DEFAULT_MIN_SWAP_FREE_MIB)
    args = parser.parse_args(argv)
    if int(args.training_epoch) < 1 or int(args.count) < 1:
        parser.error("--training-epoch and --count must be positive")
    if int(args.actor_candidates) != 16 or int(args.local_perturbations) != 16:
        parser.error("the Phase1.7 branch matrix is locked to 16 actor + 16 local candidates")
    return args


def main(argv: list[str] | None = None) -> None:
    output = collect(parse_args(argv))
    print(json.dumps({"status": "completed", "manifest": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
