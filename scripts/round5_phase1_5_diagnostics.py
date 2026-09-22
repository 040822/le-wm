#!/usr/bin/env python3
"""Collect the non-closed-loop Round 5 Phase1.5 diagnostics.

The main Phase1.5 entry point owns the 2,424-condition closed-loop scan.  This
entry point owns diagnostics whose data must not be inferred from that scan:
trajectory-disjoint physical probes and small deterministic protocol checks.
Probe artifacts are written below the independent Phase1.5 output root.

Examples::

    python scripts/round5_phase1_5_diagnostics.py validate
    CUDA_VISIBLE_DEVICES=0 python scripts/round5_phase1_5_diagnostics.py probe \
        --task cube --gpu 0 --device cuda
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.checkpoint import load_policy_or_model
from source.common.eval import (
    DatasetEvaluationSession,
    compose_eval_config,
    evaluate_from_dataset_compat,
    get_dataset,
    img_transform,
)
from source.common.eval import EvaluationIdentity
from source.common.round3_phase1 import CohortManifest, Round3TraceCollector
from source.common.round4_eval import run_round4_evaluation, validate_gpu_visibility
from scripts.round5_phase1_5 import (
    DEFAULT_MAX_LOAD_PER_CPU,
    DEFAULT_MIN_AVAILABLE_MIB,
    DEFAULT_MIN_FREE_MIB,
    _gpu_preflight,
    _host_preflight,
)
from source.common.round5_phase1_5 import (
    PHASE15_PROTOCOL_VARIANT,
    PHASE15_TASKS,
    atomic_write_json,
    candidate_pool_metrics,
    guidance_effect_metrics,
    fit_ridge_probe,
    make_probe_split,
    normalized_physical_distance,
    paired_guidance_metrics,
)


DEFAULT_CONFIG = ROOT / "config" / "round5" / "phase1_5.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "round5" / "phase1_5_seed3072_legacy"


class FixedCandidatePolicy(swm.policy.BasePolicy):
    """Replay one captured candidate action sequence in every env slot."""

    def __init__(self, normalized_actions: np.ndarray, *, process: Mapping[str, Any], action_block: int):
        super().__init__()
        actions = np.asarray(normalized_actions, dtype=np.float64)
        if actions.ndim != 3 or actions.shape[0] < 1 or actions.shape[1] < 1:
            raise ValueError("normalized_actions must have shape [env, horizon, action_dim]")
        if int(action_block) < 1 or actions.shape[-1] % int(action_block):
            raise ValueError("candidate action_dim must be divisible by action_block")
        processor = process.get("action") if process is not None else None
        if processor is None or not hasattr(processor, "inverse_transform"):
            raise ValueError("candidate replay needs the fitted action processor")
        base_action_dim = actions.shape[-1] // int(action_block)
        primitive = actions.reshape(actions.shape[0], -1, base_action_dim)
        physical = processor.inverse_transform(primitive.reshape(-1, base_action_dim))
        self._physical_actions = np.asarray(
            physical.reshape(actions.shape[0], primitive.shape[1], base_action_dim),
            dtype=np.float64,
        )
        self._action_block = int(action_block)
        self._queues: list[deque[np.ndarray]] | None = None
        self._done: np.ndarray | None = None
        self.type = "round5_phase1_5_fixed_candidate"

    def set_env(self, env: Any) -> None:
        if int(env.num_envs) != int(self._physical_actions.shape[0]):
            raise ValueError("candidate replay env count differs from captured pool")
        self.env = env
        self._queues = [deque(row.copy() for row in self._physical_actions[index]) for index in range(env.num_envs)]
        self._done = np.zeros(env.num_envs, dtype=bool)

    def get_action(self, info_dict: Mapping[str, Any], **kwargs: Any) -> np.ndarray:
        del kwargs
        if self._queues is None or self._done is None:
            raise RuntimeError("set_env must be called before get_action")
        terminated = np.asarray(info_dict.get("terminated", np.zeros(len(self._queues))), dtype=bool).reshape(-1)
        truncated = np.asarray(info_dict.get("truncated", np.zeros(len(self._queues))), dtype=bool).reshape(-1)
        if len(terminated) != len(self._queues) or len(truncated) != len(self._queues):
            raise ValueError("termination vectors do not match candidate replay env count")
        self._done |= terminated | truncated
        action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        for index, queue in enumerate(self._queues):
            if not self._done[index] and queue:
                action[index] = queue.popleft()
            elif not queue:
                self._done[index] = True
        return action

    def metadata(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "action_block": self._action_block,
            "environment_count": int(self._physical_actions.shape[0]),
            "primitive_action_steps": int(self._physical_actions.shape[1]),
        }


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"diagnostic config must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu(value: str | None) -> str | None:
    if value is None:
        return None
    if not str(value).isdigit() or int(value) not in range(8):
        raise argparse.ArgumentTypeError("--gpu must select one physical GPU0-7")
    return str(value)


def _configure_device(
    device: str,
    gpu: str | None,
    *,
    minimum_free_mib: int,
    max_load_per_cpu: float,
    minimum_available_mib: int,
) -> torch.device:
    if str(device).startswith("cuda"):
        if gpu is None:
            raise ValueError("CUDA diagnostics require --gpu")
        _host_preflight(
            max_load_per_cpu=max_load_per_cpu,
            minimum_available_mib=minimum_available_mib,
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        validate_gpu_visibility(device)
        _gpu_preflight(str(gpu), minimum_free_mib)
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {gpu} is unavailable")
    return torch.device(device)


def _manifest(config: Mapping[str, Any], task: str) -> CohortManifest:
    path = _resolve(config["cohort"]["paths"][task])
    value = CohortManifest.load(path)
    if value.protocol_variant != PHASE15_PROTOCOL_VARIANT or value.cohort_kind != "dev" or len(value.entries) != 50:
        raise ValueError(f"{task} is not the required legacy_50/dev cohort: {path}")
    expected = config["cohort"].get("sha256", {}).get(task)
    if expected is not None and value.computed_sha256 != expected:
        raise ValueError(f"{task} cohort hash changed")
    return value


def _checkpoint(config: Mapping[str, Any], task: str) -> tuple[Path, str]:
    path = _resolve(config["training"]["checkpoints"][task])
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = _sha256_file(path)
    expected = config["training"].get("checkpoint_sha256", {}).get(task)
    if expected is not None and digest != expected:
        raise ValueError(f"{task} checkpoint hash changed: {digest} != {expected}")
    return path, digest


def _candidate_state_id(entry: Any) -> str:
    return f"episode={entry.episode_id};start={int(entry.start_step)};row={int(entry.row_index)}"


def _write_jsonl(path: str | Path, records: Sequence[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False))
            stream.write("\n")
    temporary.replace(target)
    return target


def _first_replan_pool_records(
    event: Mapping[str, Any],
    *,
    task: str,
    flow_steps: int,
    manifest: CohortManifest,
    seen_slots: set[int],
) -> list[dict[str, Any]]:
    candidates = torch.as_tensor(event["candidates"]).detach().cpu().numpy()
    costs = event.get("costs")
    if costs is None:
        raise ValueError("candidate pool capture requires verifier costs")
    costs_array = torch.as_tensor(costs).detach().cpu().numpy()
    predicted = event.get("predicted_latents")
    predicted_array = None if predicted is None else torch.as_tensor(predicted).detach().cpu().numpy()
    slots = tuple(int(value) for value in event["replan_indices"])
    if candidates.ndim != 4 or costs_array.shape != candidates.shape[:2]:
        raise ValueError("candidate callback returned incompatible candidate/cost shapes")
    if predicted_array is None or predicted_array.shape[:3] != candidates.shape[:3]:
        raise ValueError("candidate callback did not return predicted future latents")
    result: list[dict[str, Any]] = []
    for row, slot in enumerate(slots):
        if slot in seen_slots:
            continue
        if slot < 0 or slot >= len(manifest.entries):
            raise ValueError(f"candidate callback returned invalid slot {slot}")
        seen_slots.add(slot)
        entry = manifest.entries[slot]
        state_id = _candidate_state_id(entry)
        for candidate_index in range(candidates.shape[1]):
            result.append(
                {
                    "task": task,
                    "state_id": state_id,
                    "slot": slot,
                    "episode_id": entry.episode_id,
                    "start_step": int(entry.start_step),
                    "row_index": int(entry.row_index),
                    "flow_steps": int(flow_steps),
                    "candidate_index": int(candidate_index),
                    "predicted_cost": float(costs_array[row, candidate_index]),
                    "predicted_future_latent": predicted_array[row, candidate_index, -1].tolist(),
                    "action": candidates[row, candidate_index].tolist(),
                    "outcome_status": "pending",
                }
            )
    return result


def _capture_candidate_proposals(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    task: str,
    manifest: CohortManifest,
    checkpoint: Path,
    checkpoint_sha256: str,
    model: Any,
    flow_steps: Sequence[int],
    output_root: Path,
) -> list[dict[str, Any]]:
    """Capture the fixed 256-action proposal pool at each requested flow step."""
    all_records: list[dict[str, Any]] = []
    for flow_step in flow_steps:
        target = output_root / f"proposals_s{int(flow_step)}.jsonl"
        if target.is_file():
            all_records.extend(_read_records(target))
            continue
        captured: list[dict[str, Any]] = []
        seen_slots: set[int] = set()

        def callback(event: dict[str, Any]) -> None:
            captured.extend(
                _first_replan_pool_records(
                    event,
                    task=task,
                    flow_steps=int(flow_step),
                    manifest=manifest,
                    seen_slots=seen_slots,
                )
            )

        cfg = compose_eval_config(
            task,
            overrides=[
                "eval.num_eval=50",
                "eval.goal_offset_steps=25",
                "eval.eval_budget=50",
                "plan_config.horizon=5",
                "plan_config.receding_horizon=5",
                "plan_config.action_block=5",
                "output.save_video=false",
                f"solver.device={args.device}",
            ],
        )
        run_round4_evaluation(
            cfg,
            task=task,
            policy_or_model=model,
            mode="P3",
            identity=EvaluationIdentity(
                entrypoint="round5_phase1_5_candidate_pool",
                policy_kind="round4_shared_dit",
                checkpoint=str(checkpoint.resolve()),
                epoch=int(config["training"]["epoch"]),
                stage="P3",
            ),
            manifest=manifest,
            output_dir=output_root / f"proposal_eval_s{int(flow_step)}",
            trace_output_dir=output_root / f"proposal_eval_s{int(flow_step)}" / "trace",
            device=args.device,
            trace=True,
            candidate_count=256,
            flow_steps=int(flow_step),
            action_flow_steps=int(flow_step),
            solver_batch_size=int(config["evaluation"]["solver_batch_size"]),
            candidate_batch_size=int(config["evaluation"]["candidate_batch_size"]),
            action_flow_integrator="euler",
            action_bound_mode="none",
            cem_protocol="not_applicable",
            guidance_mode="none",
            proposal_chunk_size=int(config["evaluation"]["proposal_chunk_size"]),
            allowed_protocol_variants=("legacy",),
            allow_variable_candidate_count=True,
            allow_solver_config_override=True,
            diagnostic_callback=callback,
        )
        if seen_slots != set(range(len(manifest.entries))):
            missing = sorted(set(range(len(manifest.entries))) - seen_slots)
            raise RuntimeError(f"candidate pool missed initial replans for slots {missing}")
        _write_jsonl(target, captured)
        all_records.extend(captured)
    return all_records


def _milestone_step(steps: Sequence[Mapping[str, Any]], target: int) -> Mapping[str, Any] | None:
    eligible = [item for item in steps if int(item.get("raw_env_step", -1)) <= int(target)]
    return eligible[-1] if eligible else None


def _extract_pixel_batch(value: Any, expected_count: int) -> Any | None:
    """Find a batched pixel observation in a vector-env observation."""
    if isinstance(value, Mapping):
        for key in ("pixels", "image", "observation", "obs"):
            if key in value:
                found = _extract_pixel_batch(value[key], expected_count)
                if found is not None:
                    return found
        for child in value.values():
            found = _extract_pixel_batch(child, expected_count)
            if found is not None:
                return found
        return None
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 4 or int(shape[0]) != int(expected_count):
        return None
    return value


def _run_fixed_candidate(
    *,
    cfg: Any,
    task: str,
    manifest: CohortManifest,
    normalized_actions: np.ndarray,
    process: Mapping[str, Any],
    model: Any,
    transform: Any,
    device: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Execute one captured candidate for all 50 states and retain true outcomes."""
    session = DatasetEvaluationSession(
        cfg,
        task=task,
        cohort=manifest.to_evaluation_cohort(),
    )
    policy = FixedCandidatePolicy(
        normalized_actions,
        process=process,
        action_block=int(cfg.plan_config.action_block),
    )
    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    world_cfg["max_episode_steps"] = 2 * int(cfg.eval.eval_budget)
    output_dir.mkdir(parents=True, exist_ok=True)
    world = session.world_factory(**world_cfg, image_shape=(224, 224))
    envs = getattr(world, "envs", None)
    if envs is None:
        raise RuntimeError("candidate branch world does not expose envs")
    action_processor = session.process.get("action")
    collector = Round3TraceCollector(
        task,
        manifest,
        action_block=int(cfg.plan_config.action_block),
        action_space=getattr(envs, "single_action_space", None),
        action_processor=action_processor,
    )
    original_step = envs.step
    step_counter = {"value": 0}
    future_latents: dict[int, dict[str, list[float]]] = {index: {} for index in range(len(manifest.entries))}
    model = getattr(model, "model", model).to(device).eval()

    def traced_step(actions: Any, *args: Any, **kwargs: Any):
        result = original_step(actions, *args, **kwargs)
        step_counter["value"] += 1
        if step_counter["value"] in {5, 10, 15, 20, 25}:
            pixels = _extract_pixel_batch(result[0] if isinstance(result, tuple) else result, len(manifest.entries))
            if pixels is not None:
                transformed = []
                for index in range(len(manifest.entries)):
                    row = pixels[index]
                    if hasattr(row, "detach"):
                        row = row.detach().cpu().numpy()
                    row = np.asarray(row)
                    if row.ndim >= 4:
                        row = row[-1]
                    transformed.append(transform(row))
                image_batch = torch.stack(transformed).to(device)
                with torch.inference_mode():
                    encoded = model.encode_pixels(image_batch).float().detach().cpu().numpy()
                for index, latent in enumerate(encoded):
                    future_latents[index][str(step_counter["value"])] = latent.tolist()
        infos = result[-1] if isinstance(result, tuple) and result else result
        collector.record_step(actions, infos, raw_env_step=step_counter["value"])
        return result

    envs.step = traced_step
    try:
        world.set_policy(policy)
        metrics = evaluate_from_dataset_compat(
            world=world,
            dataset=session.dataset,
            eval_start_idx=session.cohort.start_steps,
            eval_episodes=session.cohort.episode_ids,
            cfg=cfg,
            video_path=output_dir / "videos",
            save_video=False,
        )
    finally:
        envs.step = original_step
        if hasattr(world, "close"):
            world.close()
    successes = np.asarray(metrics["episode_successes"], dtype=bool).reshape(-1)
    episodes = collector.finalize(successes.tolist(), eval_budget=int(cfg.eval.eval_budget))
    for index, episode in enumerate(episodes):
        episode["future_latents"] = future_latents[index]
    return episodes


def _attach_candidate_outcomes(
    proposals: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    *,
    task: str,
) -> list[dict[str, Any]]:
    """Join branch traces to proposal rows without inventing post-terminal state."""
    by_slot = {int(row["slot"]): row for row in episodes}
    result: list[dict[str, Any]] = []
    for proposal in proposals:
        slot = int(proposal["slot"])
        episode = by_slot[slot]
        steps = episode.get("steps", [])
        terminal = _milestone_step(steps, 25)
        if terminal is None:
            terminal = steps[-1] if steps else None
        current = None if terminal is None else terminal.get("current")
        goal = None if terminal is None else terminal.get("goal")
        if current is not None and goal is not None:
            true_distance = float(normalized_physical_distance(task, current, goal))
        else:
            true_distance = None
        milestones: dict[str, Any] = {}
        for target in (5, 10, 15, 20, 25):
            step = _milestone_step(steps, target)
            if step is None:
                continue
            raw_step = int(step["raw_env_step"])
            milestone = {
                "raw_env_step": raw_step,
                "current": step.get("current"),
                "goal": step.get("goal"),
                "distance": step.get("distance"),
                "future_latent": episode.get("future_latents", {}).get(
                    str(raw_step)
                ),
            }
            if milestone["future_latent"] is None:
                raise RuntimeError(
                    "candidate branch did not expose a future pixel encoding at "
                    f"raw_env_step={milestone['raw_env_step']}"
                )
            milestones[str(target)] = milestone
        result.append(
            {
                **dict(proposal),
                "success": bool(episode.get("success", False)),
                "true_distance": true_distance,
                "physical_state": current,
                "goal_state": goal,
                "milestones": milestones,
                "valid_length": int(len(steps)),
                "termination_reason": (
                    steps[-1].get("termination_reason") if steps else "no_step"
                ),
                "outcome_status": "completed",
            }
        )
    return result


def _first_guidance_records(
    event: Mapping[str, Any],
    *,
    task: str,
    flow_steps: int,
    guidance: str,
    manifest: CohortManifest,
    seen_slots: set[int],
    seed: int,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    """Capture paired unguided, guided, and same-RMS random actions."""
    guided = torch.as_tensor(event["guided_actions"]).detach().cpu().numpy()
    baseline = torch.as_tensor(event["unguided_actions"]).detach().cpu().numpy()
    before = torch.as_tensor(event["predicted_cost_before"]).detach().cpu().numpy().reshape(-1)
    after = torch.as_tensor(event["predicted_cost_after"]).detach().cpu().numpy().reshape(-1)
    slots = tuple(int(value) for value in event["replan_indices"])
    if guided.shape != baseline.shape or guided.ndim != 3:
        raise ValueError("guidance callback returned incompatible action shapes")
    if guided.shape[0] != len(slots) or before.shape != (len(slots),) or after.shape != (len(slots),):
        raise ValueError("guidance callback returned incompatible cost or slot shapes")
    random_actions = np.empty_like(guided)
    rows: list[dict[str, Any]] = []
    new_indices: list[int] = []
    for row, slot in enumerate(slots):
        if slot in seen_slots:
            continue
        if slot < 0 or slot >= len(manifest.entries):
            raise ValueError(f"guidance callback returned invalid slot {slot}")
        seen_slots.add(slot)
        new_indices.append(row)
        delta = np.asarray(guided[row] - baseline[row], dtype=np.float64)
        rms = float(np.sqrt(np.mean(np.square(delta))))
        rng = np.random.default_rng(int(seed) + slot)
        direction = rng.normal(size=delta.shape)
        direction_rms = float(np.sqrt(np.mean(np.square(direction))))
        if direction_rms <= 0.0:
            raise RuntimeError("random guidance direction has zero RMS")
        random_actions[row] = baseline[row] + direction * (rms / direction_rms)
        entry = manifest.entries[slot]
        state_id = _candidate_state_id(entry)
        rows.append(
            {
                "task": task,
                "state_id": state_id,
                "slot": slot,
                "episode_id": entry.episode_id,
                "start_step": int(entry.start_step),
                "row_index": int(entry.row_index),
                "flow_steps": int(flow_steps),
                "guidance": str(guidance),
                "guided_predicted_cost_before": float(before[row]),
                "guided_predicted_cost_after": float(after[row]),
                "guided_action_rms_displacement": rms,
                "random_action_rms_displacement": rms,
            }
        )
    indices = np.asarray(new_indices, dtype=np.int64)
    return rows, baseline[indices], guided[indices], random_actions[indices]


def guidance_pool(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    """Run paired guidance branches against same-RMS random controls."""
    if args.task == "all":
        raise ValueError("guidance-pool requires one task")
    if args.guidance not in {"post_opt", "guided_flow"}:
        raise ValueError("--guidance must be post_opt or guided_flow")
    if args.device.startswith("cuda") and args.gpu is None:
        raise ValueError("guidance-pool CUDA runs require --gpu")
    _configure_device(
        args.device,
        args.gpu,
        minimum_free_mib=args.min_free_mib,
        max_load_per_cpu=args.max_load_per_cpu,
        minimum_available_mib=args.min_available_mib,
    )
    manifest = _manifest(config, args.task)
    checkpoint, checkpoint_sha256 = _checkpoint(config, args.task)
    model, resolved = load_policy_or_model(str(checkpoint))
    if resolved is not None and Path(resolved).resolve() != checkpoint.resolve():
        raise ValueError(f"checkpoint resolver changed requested path: {checkpoint}")
    diagnostics = config.get("diagnostics", {})
    default_flow_steps = (1, 2, 5, 16) if args.guidance == "post_opt" else (2, 5)
    flow_steps = tuple(args.flow_steps or default_flow_steps)
    output_root = _resolve(args.output_root) / "diagnostics" / "guidance" / args.task
    output_root.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    for flow_step in flow_steps:
        target = output_root / f"{args.guidance}_s{int(flow_step)}.jsonl"
        if target.is_file():
            all_records.extend(_read_records(target))
            continue
        captured: list[dict[str, Any]] = []
        action_batches: dict[str, list[np.ndarray]] = {
            "baseline": [],
            "guided": [],
            "random": [],
        }
        seen_slots: set[int] = set()

        def callback(event: dict[str, Any]) -> None:
            rows, baseline, guided, random_actions = _first_guidance_records(
                event,
                task=args.task,
                flow_steps=int(flow_step),
                guidance=args.guidance,
                manifest=manifest,
                seen_slots=seen_slots,
                seed=int(config.get("diagnostics", {}).get("pool_seed", 2026)),
            )
            captured.extend(rows)
            if rows:
                action_batches["baseline"].append(baseline)
                action_batches["guided"].append(guided)
                action_batches["random"].append(random_actions)

        cfg = compose_eval_config(
            args.task,
            overrides=[
                "eval.num_eval=50",
                "eval.goal_offset_steps=25",
                "eval.eval_budget=50",
                "plan_config.horizon=5",
                "plan_config.receding_horizon=5",
                "plan_config.action_block=5",
                "output.save_video=false",
                f"solver.device={args.device}",
            ],
        )
        run_round4_evaluation(
            cfg,
            task=args.task,
            policy_or_model=model,
            mode="P0",
            identity=EvaluationIdentity(
                entrypoint="round5_phase1_5_guidance_pool",
                policy_kind="round4_shared_dit",
                checkpoint=str(checkpoint.resolve()),
                epoch=int(config["training"]["epoch"]),
                stage="P0",
            ),
            manifest=manifest,
            output_dir=output_root / f"proposal_eval_s{int(flow_step)}",
            device=args.device,
            trace=False,
            candidate_count=1,
            flow_steps=int(flow_step),
            action_flow_steps=int(flow_step),
            action_bound_mode="none",
            cem_protocol="not_applicable",
            guidance_mode=args.guidance,
            guidance_step_size=float(args.guidance_step_size),
            guidance_last_steps=int(args.guidance_last_steps or flow_step),
            guidance_inner_steps=int(args.guidance_inner_steps),
            guidance_max_rms_offset=float(args.max_rms_offset),
            allowed_protocol_variants=("legacy",),
            allow_variable_candidate_count=True,
            allow_solver_config_override=True,
            diagnostic_callback=callback,
        )
        if seen_slots != set(range(len(manifest.entries))):
            missing = sorted(set(range(len(manifest.entries))) - seen_slots)
            raise RuntimeError(f"guidance pool missed initial replans for slots {missing}")
        action_arrays = {
            kind: np.concatenate(batches, axis=0)
            for kind, batches in action_batches.items()
            if batches
        }
        if set(action_arrays) != {"baseline", "guided", "random"}:
            raise RuntimeError("guidance callback did not capture all paired action batches")
        def proposal_rows(kind: str, cost_field: str) -> list[dict[str, Any]]:
            return [
                {
                    **row,
                    "candidate_index": 0,
                    "predicted_cost": float(row[cost_field]),
                    "action": action_arrays[kind][index].tolist(),
                }
                for index, row in enumerate(captured)
            ]
        baseline_rows = proposal_rows("baseline", "guided_predicted_cost_before")
        guided_rows = proposal_rows("guided", "guided_predicted_cost_after")
        random_rows = proposal_rows("random", "guided_predicted_cost_before")
        branch_session = DatasetEvaluationSession(
            cfg, task=args.task, cohort=manifest.to_evaluation_cohort()
        )
        branches = {
            kind: _run_fixed_candidate(
                cfg=cfg,
                task=args.task,
                manifest=manifest,
                normalized_actions=action_arrays[kind],
                process=branch_session.process,
                model=model,
                transform=branch_session.transform["pixels"],
                device=args.device,
                output_dir=output_root / f"{args.guidance}_s{int(flow_step)}" / kind,
            )
            for kind in ("baseline", "guided", "random")
        }
        outcomes = {
            kind: {
                int(row["slot"]): row
                for row in _attach_candidate_outcomes(
                    proposal_rows(kind, "guided_predicted_cost_before" if kind != "guided" else "guided_predicted_cost_after"),
                    branches[kind],
                    task=args.task,
                )
            }
            for kind in ("baseline", "guided", "random")
        }
        completed: list[dict[str, Any]] = []
        for row in captured:
            slot = int(row["slot"])
            base = outcomes["baseline"][slot]
            guided = outcomes["guided"][slot]
            random = outcomes["random"][slot]
            if (
                base["true_distance"] is None
                or guided["true_distance"] is None
                or random["true_distance"] is None
            ):
                raise RuntimeError(f"guidance branch missed a physical outcome for slot {slot}")
            completed.append(
                {
                    **row,
                    "guided_true_cost_before": float(base["true_distance"]),
                    "guided_true_cost_after": float(guided["true_distance"]),
                    "random_true_cost_before": float(base["true_distance"]),
                    "random_true_cost_after": float(random["true_distance"]),
                    "outcome_status": "completed",
                }
            )
        _write_jsonl(target, completed)
        all_records.extend(completed)
    aggregate = output_root / "records.jsonl"
    _write_jsonl(aggregate, all_records)
    atomic_write_json(
        output_root / "manifest.json",
        {
            "schema_version": "round5_phase1_5_guidance_pool_v1",
            "task": args.task,
            "guidance": args.guidance,
            "flow_steps": list(flow_steps),
            "records": str(aggregate),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
        },
    )
    print(json.dumps({"guidance_pool": str(output_root), "records": str(aggregate)}, ensure_ascii=False, sort_keys=True))


def _episode_column(dataset: Any) -> str:
    if "episode_idx" in dataset.column_names:
        return "episode_idx"
    if "ep_idx" in dataset.column_names:
        return "ep_idx"
    raise ValueError("dataset has no episode index column")


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _probe_target(task: str, values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode cyclic physical fields as sin/cos while preserving the schema."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"probe targets must be [rows, dimensions], got {values.shape}")
    if task == "reacher":
        return np.concatenate((np.sin(values), np.cos(values)), axis=1), {
            "source": "qpos",
            "cyclic_dimensions": list(range(values.shape[1])),
        }
    if task == "pusht":
        if values.shape[1] < 5:
            raise ValueError("PushT state must contain an angle in column 4")
        return np.concatenate((values[:, :4], np.sin(values[:, 4:5]), np.cos(values[:, 4:5])), axis=1), {
            "source": "state",
            "cyclic_dimensions": [4],
        }
    return values, {"source": "privileged_block_0_pos" if task == "cube" else "proprio", "cyclic_dimensions": []}


def _probe_rows(
    dataset: Any,
    *,
    task: str,
    selected_trajectories: Sequence[Any],
    frames_per_trajectory: int,
) -> tuple[list[int], list[Any]]:
    episodes = np.asarray(dataset.get_col_data(_episode_column(dataset)))
    steps = np.asarray(dataset.get_col_data("step_idx"), dtype=np.int64)
    selected = {_scalar(item) for item in selected_trajectories}
    rows: list[int] = []
    ids: list[Any] = []
    for episode in selected_trajectories:
        episode_value = _scalar(episode)
        positions = np.flatnonzero(episodes == episode_value)
        if len(positions) == 0:
            continue
        count = min(int(frames_per_trajectory), len(positions))
        picked = np.linspace(0, len(positions) - 1, count, dtype=np.int64)
        for position in positions[picked]:
            rows.append(int(position))
            ids.append(episode_value)
    if not rows:
        raise ValueError(f"no probe rows found for {task}; selected={sorted(selected)!r}")
    return rows, ids


def _target_column(task: str) -> str:
    return {
        "cube": "privileged_block_0_pos",
        "pusht": "state",
        "reacher": "qpos",
        "tworoom": "proprio",
    }[task]


def _jsonable_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in {
            "model",
            "x_scaler",
            "y_scaler",
            # The complete arrays are kept in features_targets.npz.  Keeping
            # them in result.json would make the human-readable manifest
            # needlessly huge and would duplicate the probe data.
            "validation_prediction",
            "validation_target",
        }:
            continue
        if isinstance(item, np.ndarray):
            result[key] = item.tolist()
        elif isinstance(item, Mapping):
            result[key] = _jsonable_metrics(item)
        elif isinstance(item, (np.generic,)):
            result[key] = item.item()
        else:
            result[key] = item
    return result


def _probe_candidate_predictions(
    records: Sequence[Mapping[str, Any]],
    *,
    task: str,
    ridge: Mapping[str, Any],
    milestone: int = 5,
) -> dict[str, Any]:
    """Apply a probe fitted on real images to real and predicted future latents.

    Candidate records contain the Stage-B endpoint prediction and the matching
    environment trace.  The comparison uses the same physical target and the
    same fitted Ridge readout for both features, so the second error includes
    the model's future-latent prediction error while the first is the
    representation/readout reference.
    """
    model = ridge.get("model")
    x_scaler = ridge.get("x_scaler")
    y_scaler = ridge.get("y_scaler")
    if model is None or x_scaler is None or y_scaler is None:
        raise ValueError("ridge probe artifacts do not contain a fitted model")
    predicted: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    for record in records:
        milestone_record = record.get("milestones", {}).get(str(int(milestone)))
        if not isinstance(milestone_record, Mapping):
            continue
        future = milestone_record.get("future_latent")
        predicted_future = record.get("predicted_future_latent")
        current = milestone_record.get("current")
        if future is None or predicted_future is None or current is None:
            continue
        observed.append(np.asarray(future, dtype=np.float64).reshape(-1))
        predicted.append(np.asarray(predicted_future, dtype=np.float64).reshape(-1))
        raw_targets.append(np.asarray(current, dtype=np.float64).reshape(-1))
    if not predicted:
        raise ValueError(
            f"candidate records contain no complete milestone-{milestone} "
            "future-latent rows"
        )
    targets, target_schema = _probe_target(task, np.stack(raw_targets, axis=0))
    observed_features = np.stack(observed, axis=0)
    predicted_features = np.stack(predicted, axis=0)

    def error_metrics(prediction: np.ndarray) -> dict[str, Any]:
        error = np.asarray(prediction, dtype=np.float64) - targets
        return {
            "count": int(len(targets)),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "per_dimension_mae": np.mean(np.abs(error), axis=0).tolist(),
        }

    def decode(features: np.ndarray) -> np.ndarray:
        scaled = x_scaler.transform(features)
        return y_scaler.inverse_transform(model.predict(scaled))

    observed_prediction = decode(observed_features)
    predicted_prediction = decode(predicted_features)
    return {
        "milestone": int(milestone),
        "rows": int(len(targets)),
        "target_schema": target_schema,
        "real_future_latent": error_metrics(observed_prediction),
        "predicted_future_latent": error_metrics(predicted_prediction),
    }


def _candidate_records_path(value: str | Path, task: str) -> Path:
    """Resolve either one task's JSONL file or a candidate-pool root."""
    path = _resolve(value)
    if path.is_dir():
        candidates = (
            path / task / "records.jsonl",
            path / "candidate_pool" / task / "records.jsonl",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return path


def _collect_features(
    dataset: Any,
    model: Any,
    transform: Any,
    *,
    rows: Sequence[int],
    target_column: str,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), int(batch_size)):
            batch_rows = list(rows[start : start + int(batch_size)])
            raw = dataset.get_row_data(batch_rows)
            pixels = np.asarray(raw["pixels"])
            image_batch = torch.stack([transform(pixel) for pixel in pixels]).to(device)
            latent = model.encode_pixels(image_batch).float().detach().cpu().numpy()
            features.append(np.asarray(latent, dtype=np.float64))
            targets.append(np.asarray(raw[target_column], dtype=np.float64))
    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


def _fit_mlp(
    features: np.ndarray,
    targets: np.ndarray,
    trajectory_ids: Sequence[Any],
    split: Mapping[str, Sequence[Any]],
    *,
    seeds: Sequence[int] = (0, 1, 2),
    epochs: int = 50,
    patience: int = 5,
) -> list[dict[str, Any]]:
    """Fit the plan's supplementary one-hidden-layer probe."""
    from torch.utils.data import DataLoader, TensorDataset

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    ids = np.asarray(trajectory_ids, dtype=object)
    train_ids, validation_ids = set(split["train"]), set(split["validation"])
    train_mask = np.asarray([item in train_ids for item in ids])
    validation_mask = np.asarray([item in validation_ids for item in ids])
    x_mean, x_std = x[train_mask].mean(0), x[train_mask].std(0)
    y_mean, y_std = y[train_mask].mean(0), y[train_mask].std(0)
    x_std[x_std < 1e-6] = 1.0
    y_std[y_std < 1e-6] = 1.0
    x_train = torch.from_numpy((x[train_mask] - x_mean) / x_std)
    y_train = torch.from_numpy((y[train_mask] - y_mean) / y_std)
    x_validation = torch.from_numpy((x[validation_mask] - x_mean) / x_std)
    y_validation = y[validation_mask]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        model = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, y.shape[1]),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(int(seed)))
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        for _ in range(int(epochs)):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                val = model(x_validation).numpy() * y_std + y_mean
            val_loss = float(np.mean(np.square(val - y_validation)))
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= int(patience):
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        with torch.inference_mode():
            prediction = model(x_validation).numpy() * y_std + y_mean
        error = prediction - y_validation
        rows.append({
            "seed": int(seed),
            "epochs_run": int(_ + 1),
            "validation_count": int(len(y_validation)),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "per_dimension_mae": np.mean(np.abs(error), axis=0).tolist(),
        })
    return rows


def probe(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    tasks = PHASE15_TASKS if args.task == "all" else (args.task,)
    device = _configure_device(
        args.device,
        args.gpu,
        minimum_free_mib=args.min_free_mib,
        max_load_per_cpu=args.max_load_per_cpu,
        minimum_available_mib=args.min_available_mib,
    )
    output_root = _resolve(args.output_root) / "diagnostics" / "probe"
    output_root.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    for task in tasks:
        manifest = _manifest(config, task)
        checkpoint, checkpoint_sha256 = _checkpoint(config, task)
        cfg = compose_eval_config(task, overrides=["dataset.keys_to_cache=[action]"])
        dataset = get_dataset(cfg, cfg.eval.dataset_name)
        episode_column = _episode_column(dataset)
        all_episodes = list(dict.fromkeys(_scalar(item) for item in np.asarray(dataset.get_col_data(episode_column))))
        excluded = {_scalar(entry.episode_id) for entry in manifest.entries}
        candidates = [item for item in all_episodes if item not in excluded]
        split = make_probe_split(
            candidates,
            eval_trajectory_ids=excluded,
            seed=int(args.seed),
            max_trajectories=int(args.max_trajectories),
        )
        chosen = tuple(split["train"]) + tuple(split["validation"])
        rows, trajectory_ids = _probe_rows(
            dataset,
            task=task,
            selected_trajectories=chosen,
            frames_per_trajectory=int(args.frames_per_trajectory),
        )
        model, resolved = load_policy_or_model(str(checkpoint))
        if resolved is not None and Path(resolved).resolve() != checkpoint.resolve():
            raise ValueError(f"checkpoint resolver changed requested path: {checkpoint}")
        model = getattr(model, "model", model).to(device).eval()
        transform = img_transform(cfg)
        features, raw_targets = _collect_features(
            dataset,
            model,
            transform,
            rows=rows,
            target_column=_target_column(task),
            device=device,
            batch_size=int(args.batch_size),
        )
        targets, target_schema = _probe_target(task, raw_targets)
        ridge = fit_ridge_probe(features, targets, trajectory_ids, split=split, seed=int(args.seed))
        result = {
            "schema_version": "round5_phase1_5_probe_v1",
            "status": "ok",
            "task": task,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "cohort_id": manifest.cohort_id,
            "cohort_sha256": manifest.computed_sha256,
            "seed": int(args.seed),
            "excluded_evaluation_trajectories": sorted(excluded, key=str),
            "trajectory_split": {key: list(value) for key, value in split.items()},
            "rows": len(rows),
            "feature_dimension": int(features.shape[1]),
            "target_dimension": int(targets.shape[1]),
            "target_schema": target_schema,
            "ridge": _jsonable_metrics(ridge),
        }
        if args.candidate_records is not None:
            candidate_path = _candidate_records_path(args.candidate_records, task)
            candidate_records = _read_records(candidate_path)
            result["candidate_pool_probe"] = _probe_candidate_predictions(
                candidate_records,
                task=task,
                ridge=ridge,
                milestone=5,
            )
            result["candidate_pool_probe"]["records"] = str(candidate_path)
        if args.mlp:
            result["mlp"] = _fit_mlp(features, targets, trajectory_ids, split, seeds=(0, 1, 2))
        task_root = output_root / task
        task_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(task_root / "features_targets.npz", features=features, targets=targets, trajectory_ids=np.asarray(trajectory_ids, dtype=object))
        atomic_write_json(task_root / "result.json", result)
        all_results.append(result)
        print(json.dumps({"task": task, "rows": len(rows), "result": str(task_root / "result.json")}, ensure_ascii=False, sort_keys=True), flush=True)
    atomic_write_json(output_root / "summary.json", {"schema_version": "round5_phase1_5_probe_summary_v1", "results": all_results})


def validate(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    manifests = {task: _manifest(config, task) for task in PHASE15_TASKS}
    for task, manifest in manifests.items():
        if len(manifest.entries) != 50 or manifest.protocol_variant != "legacy":
            raise AssertionError(f"invalid {task} manifest")
    print(json.dumps({
        "status": "ok",
        "tasks": list(PHASE15_TASKS),
        "trajectory_split_seed": 2026,
        "max_trajectories": 1000,
        "max_frames_per_trajectory": 100,
        "excluded_cohort_entries": 50,
    }, ensure_ascii=False, sort_keys=True))


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    target = _resolve(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    if target.suffix == ".jsonl":
        records = []
        for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"record {line_number} in {target} is not an object")
            records.append(dict(value))
        return records
    value = json.loads(target.read_text(encoding="utf-8"))
    if isinstance(value, Mapping):
        value = value.get("records", value.get("rows"))
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"diagnostic records must be a list or JSONL object stream: {target}")
    return [dict(item) for item in value]


def summarize(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    """Analyze persisted candidate-pool and paired-guidance records."""
    if args.candidate_records is not None and args.task == "all":
        raise ValueError(
            "summarize candidate records requires --task so physical-state "
            "normalization is unambiguous"
        )
    result: dict[str, Any] = {
        "schema_version": "round5_phase1_5_diagnostics_summary_v1",
        "task": args.task,
        "inputs": {},
    }
    if args.candidate_records is not None:
        records = _read_records(args.candidate_records)
        diagnostics = config.get("diagnostics", {})
        flow_steps = tuple(args.flow_steps or diagnostics.get("pool_flow_steps", (1, 2, 5, 10, 16, 32)))
        candidate_count = int(
            args.candidates_per_flow_step
            or diagnostics.get("pool_candidates_per_flow_step", 256)
        )
        result["inputs"]["candidate_records"] = str(_resolve(args.candidate_records))
        result["candidate_pool"] = candidate_pool_metrics(
            records,
            task=args.task,
            flow_steps=flow_steps,
            candidates_per_flow_step=candidate_count,
            bootstrap_samples=int(args.bootstrap_samples),
        )
    if args.guidance_records is not None:
        records = _read_records(args.guidance_records)
        result["inputs"]["guidance_records"] = str(_resolve(args.guidance_records))
        result["guidance"] = guidance_effect_metrics(records)
        result["paired_guidance"] = paired_guidance_metrics(
            records,
            bootstrap_samples=int(args.bootstrap_samples),
        )
    if len(result["inputs"]) == 0:
        raise ValueError("summarize requires --candidate-records or --guidance-records")
    output = _resolve(args.output_root) / "diagnostics" / "summary.json"
    atomic_write_json(output, result)
    print(json.dumps({"summary": str(output), "inputs": result["inputs"]}, ensure_ascii=False, sort_keys=True))


def candidate_pool(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    """Capture and execute the fixed candidate pool for one task."""
    if args.task == "all":
        raise ValueError("candidate-pool requires one task; run it once per task")
    if args.device.startswith("cuda") and args.gpu is None:
        raise ValueError("candidate-pool CUDA runs require --gpu")
    _configure_device(
        args.device,
        args.gpu,
        minimum_free_mib=args.min_free_mib,
        max_load_per_cpu=args.max_load_per_cpu,
        minimum_available_mib=args.min_available_mib,
    )
    manifest = _manifest(config, args.task)
    checkpoint, checkpoint_sha256 = _checkpoint(config, args.task)
    model, resolved = load_policy_or_model(str(checkpoint))
    if resolved is not None and Path(resolved).resolve() != checkpoint.resolve():
        raise ValueError(f"checkpoint resolver changed requested path: {checkpoint}")
    diagnostics = config.get("diagnostics", {})
    flow_steps = tuple(args.flow_steps or diagnostics.get("pool_flow_steps", (1, 2, 5, 10, 16, 32)))
    task_root = _resolve(args.output_root) / "diagnostics" / "candidate_pool" / args.task
    task_root.mkdir(parents=True, exist_ok=True)
    proposals = _capture_candidate_proposals(
        args=args,
        config=config,
        task=args.task,
        manifest=manifest,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        model=model,
        flow_steps=flow_steps,
        output_root=task_root,
    )
    proposal_count = int(diagnostics.get("pool_candidates_per_flow_step", 256))
    if args.proposal_only:
        atomic_write_json(
            task_root / "manifest.json",
            {
                "schema_version": "round5_phase1_5_candidate_pool_manifest_v1",
                "status": "proposal_only",
                "task": args.task,
                "flow_steps": list(flow_steps),
                "candidates_per_flow_step": proposal_count,
                "states": len(manifest.entries),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "cohort_id": manifest.cohort_id,
                "cohort_sha256": manifest.computed_sha256,
            },
        )
        print(json.dumps({"candidate_pool": str(task_root), "status": "proposal_only"}, ensure_ascii=False, sort_keys=True))
        return

    cfg = compose_eval_config(
        args.task,
        overrides=[
            "eval.num_eval=50",
            "eval.goal_offset_steps=25",
            "eval.eval_budget=50",
            "plan_config.horizon=5",
            "plan_config.receding_horizon=5",
            "plan_config.action_block=5",
            "output.save_video=false",
            f"solver.device={args.device}",
        ],
    )
    session = DatasetEvaluationSession(cfg, task=args.task, cohort=manifest.to_evaluation_cohort())
    process = session.process
    completed_files: list[Path] = []
    branch_limit = proposal_count if args.limit_candidates is None else min(proposal_count, int(args.limit_candidates))
    if branch_limit <= 0:
        raise ValueError("--limit-candidates must be positive")
    for flow_step in flow_steps:
        flow_proposals = [row for row in proposals if int(row["flow_steps"]) == int(flow_step)]
        by_candidate: dict[int, list[dict[str, Any]]] = {}
        for row in flow_proposals:
            by_candidate.setdefault(int(row["candidate_index"]), []).append(row)
        for candidate_index in range(branch_limit):
            branch_path = task_root / "branches" / f"s{int(flow_step)}" / f"candidate_{candidate_index:04d}.jsonl"
            if branch_path.is_file():
                completed_files.append(branch_path)
                continue
            rows = sorted(by_candidate.get(candidate_index, []), key=lambda item: int(item["slot"]))
            if len(rows) != len(manifest.entries):
                raise RuntimeError(
                    f"candidate {candidate_index} at S={flow_step} has {len(rows)} states; "
                    f"expected {len(manifest.entries)}"
                )
            actions = np.asarray([row["action"] for row in rows], dtype=np.float64)
            episodes = _run_fixed_candidate(
                cfg=cfg,
                task=args.task,
                manifest=manifest,
                normalized_actions=actions,
                process=process,
                model=model,
                transform=session.transform["pixels"],
                device=args.device,
                output_dir=branch_path.parent / f"candidate_{candidate_index:04d}",
            )
            outcomes = _attach_candidate_outcomes(rows, episodes, task=args.task)
            _write_jsonl(branch_path, outcomes)
            completed_files.append(branch_path)
    aggregate = task_root / "records.jsonl"
    temporary = aggregate.with_name(f".{aggregate.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for path in sorted(completed_files):
            stream.write(path.read_text(encoding="utf-8"))
    temporary.replace(aggregate)
    complete = branch_limit == proposal_count
    atomic_write_json(
        task_root / "manifest.json",
        {
            "schema_version": "round5_phase1_5_candidate_pool_manifest_v1",
            "status": "completed" if complete else "partial",
            "task": args.task,
            "flow_steps": list(flow_steps),
            "candidates_per_flow_step": proposal_count,
            "executed_candidates_per_flow_step": branch_limit,
            "states": len(manifest.entries),
            "records": str(aggregate),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "cohort_id": manifest.cohort_id,
            "cohort_sha256": manifest.computed_sha256,
        },
    )
    print(json.dumps({"candidate_pool": str(task_root), "records": str(aggregate), "status": "completed" if complete else "partial"}, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("validate", "probe", "summarize", "candidate-pool", "guidance-pool"),
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--task", choices=(*PHASE15_TASKS, "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=_gpu)
    parser.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB)
    parser.add_argument("--min-available-mib", type=int, default=DEFAULT_MIN_AVAILABLE_MIB)
    parser.add_argument("--max-load-per-cpu", type=float, default=DEFAULT_MAX_LOAD_PER_CPU)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-trajectories", type=int, default=1000)
    parser.add_argument("--frames-per-trajectory", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mlp", action="store_true")
    parser.add_argument("--candidate-records")
    parser.add_argument("--guidance-records")
    parser.add_argument("--flow-steps", type=int, nargs="+")
    parser.add_argument("--candidates-per-flow-step", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--proposal-only", action="store_true")
    parser.add_argument("--limit-candidates", type=int)
    parser.add_argument("--guidance", choices=("post_opt", "guided_flow"), default="post_opt")
    parser.add_argument("--guidance-step-size", type=float, default=0.01)
    parser.add_argument("--guidance-last-steps", type=int)
    parser.add_argument("--guidance-inner-steps", type=int, default=5)
    parser.add_argument("--max-rms-offset", type=float, default=0.2)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _load_config(_resolve(args.config))
    if args.command == "validate":
        validate(args, config)
    elif args.command == "probe":
        probe(args, config)
    elif args.command == "candidate-pool":
        candidate_pool(args, config)
    elif args.command == "guidance-pool":
        guidance_pool(args, config)
    else:
        summarize(args, config)


if __name__ == "__main__":
    main()
