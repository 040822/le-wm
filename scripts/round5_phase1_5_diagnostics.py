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
    CUDA_VISIBLE_DEVICES=0 python scripts/round5_phase1_5_diagnostics.py guidance-sweep \
        --task cube --guidance post_opt --gpu 0 --device cuda
"""

from __future__ import annotations

import argparse
from collections import deque
import copy
from contextlib import contextmanager
import fcntl
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
from source.common.round4_eval import (
    _DiagnosticCaptureComplete,
    run_round4_evaluation,
    validate_gpu_visibility,
)
from scripts.round5_phase1_5 import (
    DEFAULT_MAX_LOAD_PER_CPU,
    DEFAULT_MIN_AVAILABLE_MIB,
    DEFAULT_MIN_FREE_MIB,
    DEFAULT_MIN_SWAP_FREE_MIB,
    _gpu_preflight,
    _host_preflight,
)
from source.common.round5_phase1_5 import (
    PHASE15_PROTOCOL_VARIANT,
    PHASE15_TASKS,
    atomic_write_json,
    capture_environment_state,
    candidate_pool_metrics,
    control_action_metrics,
    make_control_actions,
    guidance_effect_metrics,
    fit_ridge_probe,
    matched_guidance_mode_metrics,
    matched_guidance_mode_metrics_by_milestone,
    make_probe_split,
    normalized_physical_distance,
    paired_guidance_metrics,
    capture_rng_state,
    restore_environment_state,
    restore_rng_state,
)


DEFAULT_CONFIG = ROOT / "config" / "round5" / "phase1_5.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "round5" / "phase1_5_seed3072_legacy"


@contextmanager
def _exclusive_file_lock(path: Path):
    """Serialize writers for an artifact set across processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"[candidate-pool] waiting for output lock: {path}", flush=True)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


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
    minimum_swap_free_mib: int,
) -> torch.device:
    if str(device).startswith("cuda"):
        if gpu is None:
            raise ValueError("CUDA diagnostics require --gpu")
        _host_preflight(
            max_load_per_cpu=max_load_per_cpu,
            minimum_available_mib=minimum_available_mib,
            minimum_swap_free_mib=minimum_swap_free_mib,
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
    planning_event = event.get("event", {})
    if not isinstance(planning_event, Mapping):
        planning_event = {}
    candidate_noise_sha256 = planning_event.get("candidate_noise_sha256")
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
                    "candidate_noise_sha256": candidate_noise_sha256,
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
    """Capture proposals and their shared initial noise at each flow step."""
    all_records: list[dict[str, Any]] = []
    noise_by_flow: dict[int, np.ndarray] = {}
    for flow_step in flow_steps:
        target = output_root / f"proposals_s{int(flow_step)}.jsonl"
        noise_target = output_root / f"candidate_noise_s{int(flow_step)}.npz"
        if target.is_file() and noise_target.is_file():
            existing = _read_records(target)
            if existing and all(item.get("candidate_noise_sha256") for item in existing):
                all_records.extend(existing)
                with np.load(noise_target) as archive:
                    noise_by_flow[int(flow_step)] = np.asarray(
                        archive["candidate_noise"], dtype=np.float32
                    )
                continue
        captured: list[dict[str, Any]] = []
        seen_slots: set[int] = set()
        noise_by_slot: dict[int, np.ndarray] = {}

        def callback(event: dict[str, Any]) -> None:
            candidates = torch.as_tensor(event["candidates"]).detach().cpu().numpy()
            candidate_noise = event.get("candidate_noise")
            if candidate_noise is None:
                raise RuntimeError("candidate pool did not expose the sampled initial noise")
            candidate_noise = torch.as_tensor(candidate_noise).detach().cpu().numpy()
            if candidates.ndim != 4 or candidate_noise.shape != (
                candidates.shape[0] * candidates.shape[1],
                candidates.shape[2],
                candidates.shape[3],
            ):
                raise ValueError(
                    "candidate noise does not align with the candidate callback: "
                    f"candidates={candidates.shape}, noise={candidate_noise.shape}"
                )
            noise_rows = candidate_noise.reshape(candidates.shape).astype(np.float32, copy=False)
            fresh_slots = set(int(slot) for slot in event["replan_indices"]) - seen_slots
            for row, slot in enumerate(event["replan_indices"]):
                slot = int(slot)
                if slot in fresh_slots:
                    noise_by_slot[slot] = noise_rows[row].copy()
            captured.extend(
                _first_replan_pool_records(
                    event,
                    task=task,
                    flow_steps=int(flow_step),
                    manifest=manifest,
                    seen_slots=seen_slots,
                )
            )
            if len(seen_slots) == len(manifest.entries):
                raise _DiagnosticCaptureComplete(
                    {"task": task, "flow_steps": int(flow_step), "states": len(seen_slots)}
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
            device=args.device,
            trace=False,
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
        if set(noise_by_slot) != set(range(len(manifest.entries))):
            missing = sorted(set(range(len(manifest.entries))) - set(noise_by_slot))
            raise RuntimeError(f"candidate pool missed initial noise for slots {missing}")
        candidate_noise_array = np.stack(
            [noise_by_slot[slot] for slot in range(len(manifest.entries))], axis=0
        ).astype(np.float32, copy=False)
        noise_temporary = noise_target.with_name(
            f".{noise_target.name}.{os.getpid()}.tmp"
        )
        with noise_temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                candidate_noise=candidate_noise_array,
                flow_steps=np.asarray([int(flow_step)], dtype=np.int64),
            )
        noise_temporary.replace(noise_target)
        noise_by_flow[int(flow_step)] = candidate_noise_array
        _write_jsonl(target, captured)
        all_records.extend(captured)
    if any(int(flow) not in noise_by_flow for flow in flow_steps):
        raise RuntimeError("candidate pool did not persist initial noise for every flow step")
    reference_noise = noise_by_flow[int(flow_steps[0])]
    if any(
        not np.array_equal(noise_by_flow[int(flow)], reference_noise)
        for flow in flow_steps
    ):
        raise RuntimeError(
            "candidate pool flow steps did not reuse the same initial candidate noise"
        )
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


def _extract_step_pixel_batch(result: Any, expected_count: int) -> Any | None:
    """Extract pixels from the info component returned by a vector env step."""
    value = result[-1] if isinstance(result, tuple) and len(result) >= 5 else result
    return _extract_pixel_batch(value, expected_count)


def _terminal_env_slots(result: Any, expected_count: int) -> list[int]:
    """Return vector-env slots that terminated or truncated on this step."""
    if not isinstance(result, tuple) or len(result) < 4:
        return []
    done = np.zeros(int(expected_count), dtype=bool)
    for value in result[2:4]:
        try:
            flags = np.asarray(value, dtype=bool).reshape(-1)
        except (TypeError, ValueError):
            continue
        if len(flags) == int(expected_count):
            done |= flags
    return np.flatnonzero(done).tolist()


def _encode_goal_latents(
    dataset: Any,
    manifest: CohortManifest,
    transform: Any,
    model: Any,
    device: str,
) -> np.ndarray:
    """Encode the fixed dataset goals used as true image-latent references."""
    if any(entry.goal_row_index is None for entry in manifest.entries):
        raise ValueError("candidate diagnostics require a goal row for every cohort entry")
    rows = [int(entry.goal_row_index) for entry in manifest.entries]
    if any(row < 0 for row in rows):
        raise ValueError("candidate diagnostics require non-negative goal rows")
    raw = dataset.get_row_data(rows)
    pixels = np.asarray(raw["pixels"])
    if len(pixels) != len(rows):
        raise ValueError("goal pixel batch does not match the cohort")
    image_batch = torch.stack([transform(pixel) for pixel in pixels]).to(device)
    model = getattr(model, "model", model).to(device).eval()
    with torch.inference_mode():
        return model.encode_pixels(image_batch).float().detach().cpu().numpy()


def _encode_start_latents(
    dataset: Any,
    manifest: CohortManifest,
    transform: Any,
    model: Any,
    device: str,
) -> np.ndarray:
    """Encode the fixed cohort observations used to create candidate pools."""
    rows = [int(entry.row_index) for entry in manifest.entries]
    raw = dataset.get_row_data(rows)
    pixels = np.asarray(raw["pixels"])
    if len(pixels) != len(rows):
        raise ValueError("start pixel batch does not match the cohort")
    image_batch = torch.stack([transform(pixel) for pixel in pixels]).to(device)
    model = getattr(model, "model", model).to(device).eval()
    with torch.inference_mode():
        return model.encode_pixels(image_batch).float().detach().cpu().numpy()


def _same_rms_random_actions(
    baseline: np.ndarray,
    guided: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic random directions with each guided RMS displacement."""
    baseline = np.asarray(baseline, dtype=np.float32)
    guided = np.asarray(guided, dtype=np.float32)
    if baseline.shape != guided.shape or baseline.ndim != 3:
        raise ValueError("baseline and guided actions must share [state, horizon, action] shape")
    delta = guided.astype(np.float64) - baseline.astype(np.float64)
    target_rms = np.sqrt(np.mean(np.square(delta), axis=(1, 2), keepdims=True))
    rng = np.random.default_rng(int(seed))
    direction = rng.normal(size=delta.shape)
    direction_rms = np.sqrt(np.mean(np.square(direction), axis=(1, 2), keepdims=True))
    random_delta = direction * (target_rms / np.maximum(direction_rms, np.finfo(np.float64).tiny))
    random_delta[target_rms[:, 0, 0] == 0.0] = 0.0
    random_actions = (baseline.astype(np.float64) + random_delta).astype(np.float32)
    actual_rms = np.sqrt(
        np.mean(
            np.square(random_actions.astype(np.float64) - baseline.astype(np.float64)),
            axis=(1, 2),
        )
    )
    return random_actions, actual_rms


def _action_saturation_fraction(actions: np.ndarray) -> float:
    actions = np.asarray(actions, dtype=np.float64)
    return float(np.mean(np.abs(actions) >= 0.99))


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
    dataset: Any | None = None,
    branch_state: dict[str, Any] | None = None,
    goal_latents: np.ndarray | None = None,
    future_pixels_path: Path | None = None,
    capture_future_latents: bool = True,
    evaluation_session: DatasetEvaluationSession | None = None,
    reusable_world: Any | None = None,
) -> list[dict[str, Any]]:
    """Execute one captured candidate and retain true outcomes."""
    session = evaluation_session or DatasetEvaluationSession(
        cfg, task=task, dataset=dataset, cohort=manifest.to_evaluation_cohort()
    )
    expected_cohort = manifest.to_evaluation_cohort()
    if (
        session.task != task
        or not np.array_equal(session.cohort.episode_ids, expected_cohort.episode_ids)
        or not np.array_equal(session.cohort.start_steps, expected_cohort.start_steps)
    ):
        raise ValueError("reused evaluation session does not match the branch cohort")
    policy = FixedCandidatePolicy(
        normalized_actions,
        process=process,
        action_block=int(cfg.plan_config.action_block),
    )
    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    world_cfg["max_episode_steps"] = 2 * int(cfg.eval.eval_budget)
    output_dir.mkdir(parents=True, exist_ok=True)
    owns_world = reusable_world is None
    world = (
        reusable_world
        if reusable_world is not None
        else session.world_factory(**world_cfg, image_shape=(224, 224))
    )
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
    future_latent_costs: dict[int, dict[str, float]] = {index: {} for index in range(len(manifest.entries))}
    future_pixels: dict[int, dict[str, np.ndarray]] = {index: {} for index in range(len(manifest.entries))}
    terminal_steps: dict[int, int] = {}
    model = getattr(model, "model", model)
    if capture_future_latents:
        model = model.to(device).eval()
    elif goal_latents is not None:
        raise ValueError("goal_latents require capture_future_latents=True")
    if goal_latents is not None:
        goal_latents = np.asarray(goal_latents, dtype=np.float64)
        if goal_latents.ndim != 2 or goal_latents.shape[0] != len(manifest.entries):
            raise ValueError("goal_latents do not match the diagnostic cohort")
    replay_state = branch_state if branch_state is not None else {}

    def replay_initial_state() -> None:
        env_slots = getattr(envs, "envs", None)
        if env_slots is None or len(env_slots) != len(manifest.entries):
            raise RuntimeError("diagnostic branch cannot enumerate simulator environments")
        if not replay_state:
            replay_state["snapshots"] = [
                {
                    "environment": capture_environment_state(env),
                    "rng": capture_rng_state(env),
                }
                for env in env_slots
            ]
            replay_state["source"] = "captured_before_first_primitive_step"
        else:
            snapshots = replay_state.get("snapshots")
            if not isinstance(snapshots, list) or len(snapshots) != len(env_slots):
                raise RuntimeError("diagnostic branch replay snapshot count changed")
            for env, snapshot in zip(env_slots, snapshots):
                restore_environment_state(env, snapshot["environment"])
                restore_rng_state(env, snapshot.get("rng"))

    def traced_step(actions: Any, *args: Any, **kwargs: Any):
        if step_counter["value"] == 0:
            replay_initial_state()
        result = original_step(actions, *args, **kwargs)
        step_counter["value"] += 1
        milestone_step = step_counter["value"] in {5, 10, 15, 20, 25}
        terminal_slots = _terminal_env_slots(result, len(manifest.entries))
        for index in terminal_slots:
            terminal_steps.setdefault(int(index), int(step_counter["value"]))
        current_terminal_slots = [
            index
            for index in terminal_slots
            if terminal_steps.get(int(index)) == int(step_counter["value"])
        ]
        if milestone_step or current_terminal_slots:
            pixels = _extract_step_pixel_batch(result, len(manifest.entries))
            encode_indices = (
                [
                    index
                    for index in range(len(manifest.entries))
                    if int(index) not in terminal_steps
                    or terminal_steps[int(index)] == int(step_counter["value"])
                ]
                if milestone_step
                else current_terminal_slots
            )
            if pixels is not None and encode_indices:
                transformed = []
                for index in encode_indices:
                    row = pixels[index]
                    if hasattr(row, "detach"):
                        row = row.detach().cpu().numpy()
                    row = np.asarray(row)
                    if row.ndim >= 4:
                        row = row[-1]
                    if future_pixels_path is not None:
                        future_pixels[index][str(step_counter["value"])] = np.ascontiguousarray(row)
                    if capture_future_latents:
                        transformed.append(transform(row))
                if capture_future_latents:
                    image_batch = torch.stack(transformed).to(device)
                    with torch.inference_mode():
                        encoded = model.encode_pixels(image_batch).float().detach().cpu().numpy()
                    for index, latent in zip(encode_indices, encoded):
                        future_latents[index][str(step_counter["value"])] = latent.tolist()
                        if goal_latents is not None:
                            future_latent_costs[index][str(step_counter["value"])] = float(
                                np.linalg.norm(
                                    np.asarray(latent, dtype=np.float64) - goal_latents[index]
                                )
                            )
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
        if owns_world and hasattr(world, "close"):
            world.close()
    successes = np.asarray(metrics["episode_successes"], dtype=bool).reshape(-1)
    episodes = collector.finalize(successes.tolist(), eval_budget=int(cfg.eval.eval_budget))
    for index, episode in enumerate(episodes):
        episode["future_latents"] = future_latents[index]
        episode["future_latent_costs"] = future_latent_costs[index]
    if future_pixels_path is not None:
        records = [
            (slot, int(step), pixels)
            for slot, step_map in future_pixels.items()
            for step, pixels in step_map.items()
        ]
        if records:
            shapes = {tuple(np.asarray(item[2]).shape) for item in records}
            if len(shapes) != 1:
                raise ValueError(f"future pixel shapes changed within a branch: {sorted(shapes)}")
            future_pixels_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = future_pixels_path.with_name(
                f".{future_pixels_path.name}.{os.getpid()}.tmp"
            )
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    slot=np.asarray([item[0] for item in records], dtype=np.int32),
                    raw_env_step=np.asarray([item[1] for item in records], dtype=np.int16),
                    pixels=np.stack([np.asarray(item[2]) for item in records], axis=0),
                )
            temporary.replace(future_pixels_path)
        else:
            raise RuntimeError("future pixel capture was requested but no pixels were observed")
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
                "future_latent_cost": episode.get("future_latent_costs", {}).get(
                    str(raw_step)
                ),
            }
            if milestone["future_latent"] is None:
                raise RuntimeError(
                    "candidate branch did not expose a future pixel encoding at "
                    f"raw_env_step={milestone['raw_env_step']}"
                )
            if episode.get("future_latent_costs") and milestone["future_latent_cost"] is None:
                raise RuntimeError(
                    "candidate branch did not expose a true image-latent cost at "
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


def _attach_guidance_branch_outcomes(
    baseline_rows: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    normalized_actions: np.ndarray,
    *,
    task: str,
) -> dict[int, dict[str, Any]]:
    """Attach persisted physical-trace fields to raw guidance branch episodes."""
    actions = np.asarray(normalized_actions, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[0] != len(baseline_rows):
        raise ValueError(
            "guidance actions must have one rank-3 action batch per baseline row"
        )
    proposals: list[dict[str, Any]] = []
    for index, baseline in enumerate(baseline_rows):
        proposal = {
            key: baseline[key]
            for key in (
                "task",
                "state_id",
                "slot",
                "episode_id",
                "start_step",
                "row_index",
                "flow_steps",
                "candidate_index",
                "candidate_noise_sha256",
            )
            if key in baseline
        }
        proposal["action"] = actions[index].tolist()
        proposals.append(proposal)
    attached = _attach_candidate_outcomes(proposals, episodes, task=task)
    by_slot = {int(row["slot"]): row for row in attached}
    if len(by_slot) != len(baseline_rows):
        raise RuntimeError("guidance branch outcomes do not uniquely cover baseline slots")
    return by_slot


def _dataset_anchor_actions(
    dataset: Any,
    manifest: CohortManifest,
    process: Mapping[str, Any],
    *,
    primitive_steps: int = 25,
) -> np.ndarray:
    """Read and normalize the data-trajectory anchor for every cohort state."""
    processor = process.get("action")
    if processor is None or not hasattr(processor, "transform"):
        raise ValueError("control pool requires a fitted action processor")
    rows: list[np.ndarray] = []
    episode_column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_values = np.asarray(dataset.get_col_data(episode_column))
    for entry in manifest.entries:
        indices = np.arange(
            int(entry.row_index), int(entry.row_index) + int(primitive_steps), dtype=np.int64
        )
        if int(indices[-1]) >= len(episode_values):
            raise ValueError(f"data anchor runs past dataset end for slot {entry.row_index}")
        if not np.all(episode_values[indices] == entry.episode_id):
            raise ValueError(
                "data anchor crosses an episode boundary for "
                f"episode={entry.episode_id}, row={entry.row_index}"
            )
        raw = np.asarray(dataset.get_row_data(indices)["action"], dtype=np.float64)
        if raw.ndim != 2 or raw.shape[0] != int(primitive_steps):
            raise ValueError(f"data action rows have unexpected shape {raw.shape}")
        normalized = np.asarray(processor.transform(raw), dtype=np.float64)
        rows.append(normalized)
    return np.stack(rows, axis=0)


def _proposal_action_as_primitives(
    action: Any,
    primitive_shape: tuple[int, ...],
    *,
    action_block: int,
) -> np.ndarray:
    """Convert one packed proposal into the dataset's primitive-action layout."""
    values = np.asarray(action, dtype=np.float64)
    if values.shape == primitive_shape:
        return values
    if (
        values.ndim == 2
        and len(primitive_shape) == 2
        and int(action_block) > 0
        and values.shape[0] * int(action_block) == primitive_shape[0]
        and values.shape[1] == primitive_shape[1] * int(action_block)
    ):
        return values.reshape(primitive_shape)
    raise ValueError(
        "proposal action cannot be reblocked to the data anchor shape: "
        f"proposal={values.shape}, data={primitive_shape}, action_block={action_block}"
    )


def _pack_primitive_actions_for_replay(
    actions: np.ndarray,
    *,
    action_block: int,
    horizon: int,
) -> np.ndarray:
    """Pack primitive normalized actions into the planner's action-token layout."""
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] < 1:
        raise ValueError("primitive actions must have shape [env, steps, action_dim]")
    if int(action_block) < 1 or int(horizon) < 1:
        raise ValueError("action_block and horizon must be positive")
    expected_steps = int(action_block) * int(horizon)
    if values.shape[1] != expected_steps:
        raise ValueError(
            "primitive action length does not match the replay horizon: "
            f"steps={values.shape[1]}, action_block={action_block}, horizon={horizon}"
        )
    return values.reshape(
        values.shape[0], int(horizon), int(action_block) * values.shape[-1]
    )


def _control_pool_actions(
    *,
    dataset: Any,
    manifest: CohortManifest,
    process: Mapping[str, Any],
    proposals: Sequence[Mapping[str, Any]],
    seed: int,
    action_block: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build per-state controls from data, S=1, and S=2 proposal anchors."""
    data = _dataset_anchor_actions(dataset, manifest, process)
    by_flow: dict[int, dict[int, Mapping[str, Any]]] = {1: {}, 2: {}}
    for row in proposals:
        flow = int(row["flow_steps"])
        if flow in by_flow and int(row["candidate_index"]) == 0:
            by_flow[flow][int(row["slot"])] = row
    missing = {
        flow: sorted(set(range(len(manifest.entries))) - set(rows))
        for flow, rows in by_flow.items()
        if len(rows) != len(manifest.entries)
    }
    if missing:
        raise RuntimeError(f"control pool is missing first proposals: {missing}")
    anchors: list[np.ndarray] = []
    for slot in range(len(manifest.entries)):
        first_s1 = _proposal_action_as_primitives(
            by_flow[1][slot]["action"],
            data[slot].shape,
            action_block=int(action_block),
        )
        first_s2 = _proposal_action_as_primitives(
            by_flow[2][slot]["action"],
            data[slot].shape,
            action_block=int(action_block),
        )
        anchors.append(np.stack((data[slot], first_s1, first_s2), axis=0))
    values: list[np.ndarray] = []
    metadata: list[dict[str, Any]] | None = None
    for slot_anchors in anchors:
        physical_zero = np.asarray(
            process["action"].transform(np.zeros((1, slot_anchors.shape[-1]), dtype=np.float64)),
            dtype=np.float64,
        )
        physical_zero = np.broadcast_to(physical_zero, slot_anchors.shape[1:]).copy()
        actions, current_metadata = make_control_actions(
            slot_anchors,
            seed=int(seed),
            action_block=int(action_block),
            physical_zero=physical_zero,
        )
        if metadata is None:
            metadata = current_metadata
        elif current_metadata != metadata:
            raise AssertionError("control metadata changed between cohort states")
        values.append(actions)
    if metadata is None:
        raise RuntimeError("control pool did not produce metadata")
    return np.stack(values, axis=1), metadata


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
        minimum_swap_free_mib=args.min_swap_free_mib,
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
            if len(seen_slots) == len(manifest.entries):
                raise _DiagnosticCaptureComplete(
                    {"task": args.task, "flow_steps": int(flow_step), "states": len(seen_slots)}
                )

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
        goal_latents = _encode_goal_latents(
            branch_session.dataset,
            manifest,
            branch_session.transform["pixels"],
            model,
            args.device,
        )
        branch_replay: dict[str, Any] = {}
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
                dataset=branch_session.dataset,
                branch_state=branch_replay,
                goal_latents=goal_latents,
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
            def latent_cost(outcome: Mapping[str, Any]) -> float:
                milestone = outcome.get("milestones", {}).get("25", {})
                value = milestone.get("future_latent_cost")
                if value is None:
                    raise RuntimeError(
                        f"guidance branch missed a true image-latent cost for slot {slot}"
                    )
                return float(value)

            baseline_latent = latent_cost(base)
            guided_latent = latent_cost(guided)
            random_latent = latent_cost(random)
            completed.append(
                {
                    **row,
                    "guided_true_cost_before": float(base["true_distance"]),
                    "guided_true_cost_after": float(guided["true_distance"]),
                    "random_true_cost_before": float(base["true_distance"]),
                    "random_true_cost_after": float(random["true_distance"]),
                    "guided_true_latent_cost_before": baseline_latent,
                    "guided_true_latent_cost_after": guided_latent,
                    "random_true_latent_cost_before": baseline_latent,
                    "random_true_latent_cost_after": random_latent,
                    "true_latent_improvement": baseline_latent - guided_latent,
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
            "state_replay": "captured_before_first_primitive_step_and_restored_with_rng",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
        },
    )
    print(json.dumps({"guidance_pool": str(output_root), "records": str(aggregate)}, ensure_ascii=False, sort_keys=True))


def _fixed_pool_actions_match_replay(
    replayed: np.ndarray,
    fixed_pool: np.ndarray,
    *,
    rtol: float = 2e-5,
    atol: float = 1e-5,
) -> bool:
    """Allow small floating-point drift when replaying persisted actions."""
    actual = np.asarray(replayed)
    expected = np.asarray(fixed_pool)
    return actual.shape == expected.shape and bool(
        np.allclose(actual, expected, rtol=float(rtol), atol=float(atol))
    )


def _float_token(value: float) -> str:
    return format(float(value), ".6g").replace("-", "m").replace(".", "p")


def _common_guidance_outcome(
    task: str,
    baseline: Mapping[str, Any],
    guided: Mapping[str, Any],
    random: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return physical and latent outcomes at the latest shared real milestone."""
    rows = (baseline, guided, random)
    for target in (25, 20, 15, 10, 5):
        milestones = [row.get("milestones", {}).get(str(target)) for row in rows]
        if any(not isinstance(item, Mapping) for item in milestones):
            continue
        if any(int(item.get("raw_env_step", -1)) != target for item in milestones):
            continue
        if any(
            item.get("current") is None
            or item.get("goal") is None
            or item.get("future_latent_cost") is None
            for item in milestones
        ):
            continue
        return {
            "step": int(target),
            "physical_costs": [
                float(normalized_physical_distance(task, item["current"], item["goal"]))
                for item in milestones
            ],
            "latent_costs": [float(item["future_latent_cost"]) for item in milestones],
        }
    return None


def guidance_sweep(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    """Run the plan's paired correction sweep on the first two fixed-pool candidates."""
    if args.task == "all":
        raise ValueError("guidance-sweep requires one task")
    if args.guidance not in {"post_opt", "guided_flow"}:
        raise ValueError("--guidance must be post_opt or guided_flow")
    _configure_device(
        args.device,
        args.gpu,
        minimum_free_mib=args.min_free_mib,
        max_load_per_cpu=args.max_load_per_cpu,
        minimum_available_mib=args.min_available_mib,
        minimum_swap_free_mib=args.min_swap_free_mib,
    )
    task = str(args.task)
    mode = str(args.guidance)
    diagnostics = config.get("diagnostics", {})
    all_flow_steps = (1, 2, 5, 16) if mode == "post_opt" else (2, 5)
    flow_steps = tuple(int(value) for value in (args.flow_steps or all_flow_steps))
    if not flow_steps or any(value not in all_flow_steps for value in flow_steps):
        raise ValueError(f"unsupported {mode} flow steps: {flow_steps}")
    inner_steps_grid = (1, 5, 10)
    step_size_grid = (0.003, 0.01, 0.03)
    if mode == "post_opt":
        max_rms_grid = (0.05, 0.2, 0.5)
    else:
        max_rms_grid = (0.2,)

    output_root = _resolve(args.output_root)
    pool_root = output_root / "diagnostics" / "candidate_pool" / task
    pool_manifest_path = pool_root / "manifest.json"
    if not pool_manifest_path.is_file():
        raise FileNotFoundError(f"fixed candidate pool manifest is missing: {pool_manifest_path}")
    pool_manifest = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
    expected_candidates = int(diagnostics.get("pool_candidates_per_flow_step", 256))
    if (
        pool_manifest.get("status") != "completed"
        or int(pool_manifest.get("states", -1)) != 50
        or int(pool_manifest.get("candidates_per_flow_step", -1)) != expected_candidates
        or int(pool_manifest.get("executed_candidates_per_flow_step", -1)) != expected_candidates
        or not set(flow_steps).issubset(
            {int(value) for value in pool_manifest.get("flow_steps", ())}
        )
    ):
        raise ValueError("guidance-sweep requires a completed 50-state fixed candidate pool")
    checkpoint, checkpoint_sha256 = _checkpoint(config, task)
    if pool_manifest.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("candidate pool checkpoint does not match the frozen checkpoint")
    manifest = _manifest(config, task)
    if pool_manifest.get("cohort_sha256") != manifest.computed_sha256:
        raise ValueError("candidate pool cohort does not match the frozen legacy_50 cohort")

    model, resolved = load_policy_or_model(str(checkpoint))
    if resolved is not None and Path(resolved).resolve() != checkpoint.resolve():
        raise ValueError(f"checkpoint resolver changed requested path: {checkpoint}")
    model = getattr(model, "model", model).to(args.device).eval()
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
    session = DatasetEvaluationSession(cfg, task=task, cohort=manifest.to_evaluation_cohort())
    starts = _encode_start_latents(
        session.dataset, manifest, session.transform["pixels"], model, args.device
    )
    goals = _encode_goal_latents(
        session.dataset, manifest, session.transform["pixels"], model, args.device
    )
    if starts.shape != goals.shape or starts.shape[0] != len(manifest.entries):
        raise ValueError("encoded cohort start and goal latents are incompatible")

    task_root = output_root / "diagnostics" / "guidance_sweep" / task / mode
    variants_root = task_root / "variants"
    variants_root.mkdir(parents=True, exist_ok=True)
    expected_variant_count = (
        len(flow_steps)
        * 2
        * len(inner_steps_grid)
        * len(step_size_grid)
        * len(max_rms_grid)
    )
    if args.limit_guidance_variants is not None and int(args.limit_guidance_variants) < 1:
        raise ValueError("--limit-guidance-variants must be positive")
    selected_variant_count = (
        expected_variant_count
        if args.limit_guidance_variants is None
        else min(expected_variant_count, int(args.limit_guidance_variants))
    )
    completed_variants = 0
    variant_ordinal = 0
    for flow_step in flow_steps:
        noise_path = pool_root / f"candidate_noise_s{flow_step}.npz"
        if not noise_path.is_file():
            raise FileNotFoundError(f"candidate pool initial noise is missing: {noise_path}")
        with np.load(noise_path) as archive:
            candidate_noise = np.asarray(archive["candidate_noise"], dtype=np.float32)
        if (
            candidate_noise.ndim != 4
            or candidate_noise.shape[0] != len(manifest.entries)
            or candidate_noise.shape[1] < 2
        ):
            raise ValueError(f"candidate noise has an invalid shape: {candidate_noise.shape}")
        for candidate_index in (0, 1):
            branch_path = pool_root / "branches" / f"s{flow_step}" / f"candidate_{candidate_index:04d}.jsonl"
            if not branch_path.is_file():
                raise FileNotFoundError(f"candidate physical outcomes are missing: {branch_path}")
            branch_rows = _read_records(branch_path)
            baseline_by_slot = {int(row["slot"]): row for row in branch_rows}
            if len(branch_rows) != len(manifest.entries) or set(baseline_by_slot) != set(range(len(manifest.entries))):
                raise ValueError(f"candidate branch does not cover all 50 states: {branch_path}")
            ordered_baseline = [baseline_by_slot[slot] for slot in range(len(manifest.entries))]
            baseline_actions = np.asarray(
                [row["action"] for row in ordered_baseline], dtype=np.float32
            )
            if baseline_actions.shape != candidate_noise[:, candidate_index].shape:
                raise ValueError(
                    "candidate actions and initial noise have different shapes: "
                    f"actions={baseline_actions.shape}, noise={candidate_noise[:, candidate_index].shape}"
                )
            noise_batch = candidate_noise[:, candidate_index]

            if mode == "guided_flow":
                with torch.inference_mode():
                    regenerated = model.sample_actions(
                        torch.as_tensor(starts, device=args.device, dtype=torch.float32),
                        noise=torch.as_tensor(noise_batch, device=args.device, dtype=torch.float32),
                        num_steps=flow_step,
                        goal_latent=torch.as_tensor(goals, device=args.device, dtype=torch.float32),
                        integrator="euler",
                    ).detach().cpu().numpy()
                reproduction_error = float(
                    np.max(np.abs(regenerated - baseline_actions))
                )
                if not _fixed_pool_actions_match_replay(
                    regenerated, baseline_actions
                ):
                    raise RuntimeError(
                        "fixed-pool action could not be reproduced from its stored noise: "
                        f"task={task}, S={flow_step}, candidate={candidate_index}, "
                        f"max_abs_error={reproduction_error}"
                    )
            else:
                reproduction_error = None

            for inner_steps in inner_steps_grid:
                for step_size in step_size_grid:
                    for max_rms_offset in max_rms_grid:
                        key = (
                            f"s{flow_step}_c{candidate_index}_k{inner_steps}_"
                            f"eta{_float_token(step_size)}_r{_float_token(max_rms_offset)}.jsonl"
                        )
                        variant_ordinal += 1
                        if variant_ordinal > selected_variant_count:
                            continue
                        target = variants_root / key
                        if target.is_file():
                            existing = _read_records(target)
                            if len(existing) != len(manifest.entries):
                                raise ValueError(f"stored guidance variant is incomplete: {target}")
                            completed_variants += 1
                            continue
                        print(
                            "[guidance_sweep] "
                            f"task={task} mode={mode} S={flow_step} candidate={candidate_index} "
                            f"K={inner_steps} eta={step_size} R={max_rms_offset}",
                            flush=True,
                        )
                        start_tensor = torch.as_tensor(starts, device=args.device, dtype=torch.float32)
                        goal_tensor = torch.as_tensor(goals, device=args.device, dtype=torch.float32)
                        baseline_tensor = torch.as_tensor(
                            baseline_actions, device=args.device, dtype=torch.float32
                        )
                        if mode == "post_opt":
                            guided_tensor = model.post_optimize_actions(
                                start_tensor,
                                goal_tensor,
                                baseline_tensor,
                                step_size=step_size,
                                inner_steps=inner_steps,
                                max_rms_offset=max_rms_offset,
                            )
                        else:
                            guided_tensor = model.sample_actions(
                                start_tensor,
                                noise=torch.as_tensor(noise_batch, device=args.device, dtype=torch.float32),
                                num_steps=flow_step,
                                goal_latent=goal_tensor,
                                integrator="euler",
                                guidance_mode="guided_flow",
                                guidance_step_size=step_size,
                                guidance_last_steps=min(3, flow_step),
                                guidance_inner_steps=inner_steps,
                                guidance_max_rms_offset=max_rms_offset,
                            )
                        guidance_stats = dict(getattr(model, "last_guidance_stats", {}))
                        guided_actions = guided_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
                        random_seed_bytes = hashlib.sha256(
                            f"{task}|{mode}|{flow_step}|{candidate_index}|{inner_steps}|{step_size}|{max_rms_offset}|2026".encode()
                        ).digest()[:8]
                        random_actions, random_rms = _same_rms_random_actions(
                            baseline_actions,
                            guided_actions,
                            seed=int.from_bytes(random_seed_bytes, "little"),
                        )
                        guided_rms = np.sqrt(
                            np.mean(
                                np.square(
                                    guided_actions.astype(np.float64)
                                    - baseline_actions.astype(np.float64)
                                ),
                                axis=(1, 2),
                            )
                        )
                        if not np.allclose(guided_rms, random_rms, rtol=0.0, atol=1e-6):
                            raise RuntimeError("random direction did not match the guidance RMS displacement")
                        with torch.inference_mode():
                            before_costs = model.get_cost_from_latents(
                                start_tensor, goal_tensor, baseline_tensor[:, None]
                            )[:, 0].detach().cpu().numpy()
                            after_costs = model.get_cost_from_latents(
                                start_tensor,
                                goal_tensor,
                                torch.as_tensor(guided_actions, device=args.device, dtype=torch.float32)[:, None],
                            )[:, 0].detach().cpu().numpy()

                        branch_state: dict[str, Any] = {}
                        variant_output = target.with_suffix("")
                        guided_outcomes = _run_fixed_candidate(
                            cfg=cfg,
                            task=task,
                            manifest=manifest,
                            normalized_actions=guided_actions,
                            process=session.process,
                            model=model,
                            transform=session.transform["pixels"],
                            device=args.device,
                            output_dir=variant_output / "guided",
                            dataset=session.dataset,
                            branch_state=branch_state,
                            goal_latents=goals,
                        )
                        random_outcomes = _run_fixed_candidate(
                            cfg=cfg,
                            task=task,
                            manifest=manifest,
                            normalized_actions=random_actions,
                            process=session.process,
                            model=model,
                            transform=session.transform["pixels"],
                            device=args.device,
                            output_dir=variant_output / "random",
                            dataset=session.dataset,
                            branch_state=branch_state,
                            goal_latents=goals,
                        )
                        guided_by_slot = _attach_guidance_branch_outcomes(
                            ordered_baseline,
                            guided_outcomes,
                            guided_actions,
                            task=task,
                        )
                        random_by_slot = _attach_guidance_branch_outcomes(
                            ordered_baseline,
                            random_outcomes,
                            random_actions,
                            task=task,
                        )
                        if (
                            set(guided_by_slot) != set(baseline_by_slot)
                            or set(random_by_slot) != set(baseline_by_slot)
                        ):
                            raise RuntimeError("guided or random physical branches missed cohort states")
                        records: list[dict[str, Any]] = []
                        for slot, baseline_row in enumerate(ordered_baseline):
                            guided_row = guided_by_slot[slot]
                            random_row = random_by_slot[slot]
                            common = _common_guidance_outcome(
                                task, baseline_row, guided_row, random_row
                            )
                            physical = None if common is None else common["physical_costs"]
                            latent = None if common is None else common["latent_costs"]
                            records.append(
                                {
                                    "schema_version": "round5_phase1_5_guidance_sweep_v1",
                                    "task": task,
                                    "guidance": mode,
                                    "flow_steps": flow_step,
                                    "candidate_index": candidate_index,
                                    "guidance_inner_steps": inner_steps,
                                    "guidance_step_size": step_size,
                                    "max_rms_offset": max_rms_offset,
                                    "guidance_last_steps": min(3, flow_step) if mode == "guided_flow" else 0,
                                    "state_id": baseline_row["state_id"],
                                    "slot": slot,
                                    "episode_id": baseline_row["episode_id"],
                                    "predicted_cost_before": float(before_costs[slot]),
                                    "predicted_cost_after": float(after_costs[slot]),
                                    "baseline_pool_predicted_cost": float(baseline_row["predicted_cost"]),
                                    "guided_predicted_cost_before": float(before_costs[slot]),
                                    "guided_predicted_cost_after": float(after_costs[slot]),
                                    "true_cost_before": None if physical is None else float(physical[0]),
                                    "true_cost_after": None if physical is None else float(physical[1]),
                                    "random_true_cost_before": None if physical is None else float(physical[0]),
                                    "random_true_cost_after": None if physical is None else float(physical[2]),
                                    "guided_true_cost_before": None if physical is None else float(physical[0]),
                                    "guided_true_cost_after": None if physical is None else float(physical[1]),
                                    "guided_true_latent_cost_before": None if latent is None else float(latent[0]),
                                    "guided_true_latent_cost_after": None if latent is None else float(latent[1]),
                                    "random_true_latent_cost_before": None if latent is None else float(latent[0]),
                                    "random_true_latent_cost_after": None if latent is None else float(latent[2]),
                                    "true_latent_improvement": None if latent is None else float(latent[0] - latent[1]),
                                    "guided_action_rms_displacement": float(guided_rms[slot]),
                                    "random_action_rms_displacement": float(random_rms[slot]),
                                    "action_rms_displacement": float(guided_rms[slot]),
                                    "action_saturation_fraction": _action_saturation_fraction(guided_actions[slot]),
                                    "guided_success": bool(guided_row["success"]),
                                    "random_success": bool(random_row["success"]),
                                    "baseline_success": bool(baseline_row["success"]),
                                    "baseline_valid_length": int(baseline_row["valid_length"]),
                                    "guided_valid_length": int(guided_row["valid_length"]),
                                    "random_valid_length": int(random_row["valid_length"]),
                                    "paired_outcome_valid": common is not None,
                                    "physical_comparison_step": None if common is None else int(common["step"]),
                                    "baseline_noise_reproduction_max_abs_error": reproduction_error,
                                    "guidance_stats": guidance_stats,
                                    "outcome_status": "completed" if common is not None else "short_termination_unpaired",
                                }
                            )
                        _write_jsonl(target, records)
                        completed_variants += 1
                        print(
                            f"[guidance_sweep] completed variant={key} states={len(records)}",
                            flush=True,
                        )

    variant_files = sorted(variants_root.glob("*.jsonl"))
    all_records = [record for path in variant_files for record in _read_records(path)]
    aggregate = task_root / "records.jsonl"
    _write_jsonl(aggregate, all_records)
    atomic_write_json(
        task_root / "manifest.json",
        {
            "schema_version": "round5_phase1_5_guidance_sweep_manifest_v1",
            "status": "completed" if completed_variants == expected_variant_count else "partial",
            "task": task,
            "guidance": mode,
            "flow_steps": list(flow_steps),
            "candidate_indices": [0, 1],
            "guidance_inner_steps": list(inner_steps_grid),
            "guidance_step_sizes": list(step_size_grid),
            "max_rms_offsets": list(max_rms_grid),
            "guidance_last_steps_rule": "min(3,S) for guided_flow; not applicable for post_opt",
            "expected_variants": expected_variant_count,
            "selected_variants": selected_variant_count,
            "completed_variants": completed_variants,
            "paired_outcome_valid_records": sum(bool(row["paired_outcome_valid"]) for row in all_records),
            "paired_outcome_invalid_records": sum(not bool(row["paired_outcome_valid"]) for row in all_records),
            "records": str(aggregate),
            "candidate_pool_manifest": str(pool_manifest_path),
            "state_replay": "shared simulator state and RNG snapshots for guided/random branches",
            "random_control": "per-state Gaussian direction scaled to the exact guided-action RMS displacement",
            "action_saturation_definition": "fraction of normalized action values with absolute value >= 0.99",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "cohort_id": manifest.cohort_id,
            "cohort_sha256": manifest.computed_sha256,
        },
    )
    print(
        json.dumps(
            {
                "guidance_sweep": str(task_root),
                "records": str(aggregate),
                "variants": completed_variants,
                "expected_variants": expected_variant_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


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


def _branch_file_matches(path: Path, expected_rows: Sequence[Mapping[str, Any]]) -> bool:
    """Reuse a completed branch only when it still matches its proposed actions."""
    if not path.is_file():
        return False
    try:
        actual_rows = _read_records(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if len(actual_rows) != len(expected_rows):
        return False
    actual_by_slot = {int(row.get("slot", -1)): row for row in actual_rows}
    if len(actual_by_slot) != len(actual_rows):
        return False
    for expected in expected_rows:
        slot = int(expected["slot"])
        actual = actual_by_slot.get(slot)
        if actual is None or actual.get("outcome_status") != "completed":
            return False
        for field in (
            "task",
            "state_id",
            "slot",
            "episode_id",
            "start_step",
            "row_index",
            "flow_steps",
            "candidate_index",
            "control_index",
            "control_kind",
            "control_metadata",
        ):
            if field in expected and actual.get(field) != expected[field]:
                return False
        expected_action = np.asarray(expected.get("action"), dtype=np.float64)
        actual_action = np.asarray(actual.get("action"), dtype=np.float64)
        if (
            expected_action.shape != actual_action.shape
            or not np.array_equal(expected_action, actual_action)
        ):
            return False
        if expected.get("predicted_cost") is not None and not np.isclose(
            float(actual.get("predicted_cost", np.nan)),
            float(expected["predicted_cost"]),
            rtol=0.0,
            atol=1e-8,
        ):
            return False
    return True


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
    # The trajectory sampler returns rows in trajectory order, which makes
    # each small h5py read a scattered read over the dataset.  Sort the full
    # request once so adjacent batches follow dataset order, then restore the
    # sampler's order before fitting the trajectory-disjoint probe.
    requested_rows = np.asarray(rows, dtype=np.int64)
    read_order = np.argsort(requested_rows, kind="stable")
    sorted_rows = requested_rows[read_order]
    # Keep HDF5 reads bounded to the sampled rows.  Reading the full span
    # between sparse frames can decompress most of a long episode and retain
    # several GiB in the process/page cache on large datasets.
    h5_file = getattr(dataset, "h5_file", None)
    if h5_file is not None and "pixels" in h5_file:
        pixel_store = h5_file["pixels"]
        target_store = h5_file[target_column]
        episode_values = np.asarray(dataset.get_col_data(_episode_column(dataset)))
        sorted_episode_values = episode_values[sorted_rows]
        group_breaks = np.flatnonzero(sorted_episode_values[1:] != sorted_episode_values[:-1]) + 1
        group_starts = np.concatenate(([0], group_breaks))
        group_stops = np.concatenate((group_breaks, [len(sorted_rows)]))
        row_groups = [
            sorted_rows[start:stop]
            for start, stop in zip(group_starts, group_stops)
        ]
        prefetch = True
    else:
        raw = dataset.get_row_data(sorted_rows.tolist())
        all_pixels = np.asarray(raw["pixels"])
        all_targets = np.asarray(raw[target_column], dtype=np.float64)
        prefetch = False
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        if prefetch:
            batches = (
                (
                    np.asarray(pixel_store[group]),
                    np.asarray(target_store[group], dtype=np.float64),
                )
                for group in row_groups
            )
        else:
            batches = ((all_pixels, all_targets),)
        rows_encoded = 0
        report_interval = max(1_000, int(batch_size) * 2)
        next_report = report_interval
        total_groups = len(row_groups) if prefetch else 1
        for group_index, (pixel_group, target_group) in enumerate(batches, start=1):
            for start in range(0, len(pixel_group), int(batch_size)):
                stop = start + int(batch_size)
                batch_pixels = pixel_group[start:stop]
                if batch_pixels.ndim == 4 and batch_pixels.shape[-1] in (1, 3, 4):
                    # torchvision v2 transforms operate on batched BCHW tensors;
                    # avoid invoking the transform once per frame on the CPU.
                    image_batch = transform(
                        torch.from_numpy(np.ascontiguousarray(batch_pixels)).permute(0, 3, 1, 2)
                    ).to(device)
                else:
                    image_batch = torch.stack([transform(pixel) for pixel in batch_pixels]).to(device)
                latent = model.encode_pixels(image_batch).float().detach().cpu().numpy()
                features.append(np.asarray(latent, dtype=np.float32))
                targets.append(np.asarray(target_group[start:stop], dtype=np.float64))
                rows_encoded += len(batch_pixels)
                if rows_encoded >= next_report or rows_encoded == len(sorted_rows):
                    print(
                        "[probe_features] "
                        f"rows={rows_encoded}/{len(sorted_rows)} "
                        f"trajectory_groups={group_index}/{total_groups}",
                        flush=True,
                    )
                    while next_report <= rows_encoded:
                        next_report += report_interval
    inverse_order = np.argsort(read_order, kind="stable")
    return (
        np.concatenate(features, axis=0)[inverse_order],
        np.concatenate(targets, axis=0)[inverse_order],
    )


def _random_encoder(model: Any, *, seed: int, device: torch.device) -> Any:
    """Make a deterministic, frozen random-weight encoder baseline."""
    baseline = copy.deepcopy(model).to(device)
    cuda_devices = []
    if device.type == "cuda" and device.index is not None:
        cuda_devices = [int(device.index)]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        for module in baseline.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
    baseline.requires_grad_(False)
    return baseline.eval()


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


def _write_probe_summary(output_root: Path) -> None:
    """Publish all completed per-task probe results, including resumed runs."""
    results = []
    for task in PHASE15_TASKS:
        result_path = output_root / task / "result.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") == "ok" and result.get("task") == task:
            results.append(result)
    atomic_write_json(
        output_root / "summary.json",
        {"schema_version": "round5_phase1_5_probe_summary_v1", "results": results},
    )


def probe(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    tasks = PHASE15_TASKS if args.task == "all" else (args.task,)
    device = _configure_device(
        args.device,
        args.gpu,
        minimum_free_mib=args.min_free_mib,
        max_load_per_cpu=args.max_load_per_cpu,
        minimum_available_mib=args.min_available_mib,
        minimum_swap_free_mib=args.min_swap_free_mib,
    )
    output_root = _resolve(args.output_root) / "diagnostics" / "probe"
    output_root.mkdir(parents=True, exist_ok=True)
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
        print(f"[probe] task={task} stage=ridge_fit rows={len(rows)}", flush=True)
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
        print(f"[probe] task={task} stage=random_encoder_encode", flush=True)
        random_model = _random_encoder(model, seed=int(args.seed), device=device)
        random_features, _ = _collect_features(
            dataset,
            random_model,
            transform,
            rows=rows,
            target_column=_target_column(task),
            device=device,
            batch_size=int(args.batch_size),
        )
        print(f"[probe] task={task} stage=random_encoder_ridge_fit", flush=True)
        random_ridge = fit_ridge_probe(
            random_features,
            targets,
            trajectory_ids,
            split=split,
            seed=int(args.seed),
        )
        result["random_encoder_ridge"] = _jsonable_metrics(random_ridge)
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
            print(f"[probe] task={task} stage=mlp_fit seeds=0,1,2", flush=True)
            result["mlp"] = _fit_mlp(features, targets, trajectory_ids, split, seeds=(0, 1, 2))
        task_root = output_root / task
        task_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(task_root / "features_targets.npz", features=features, targets=targets, trajectory_ids=np.asarray(trajectory_ids, dtype=object))
        atomic_write_json(task_root / "result.json", result)
        _write_probe_summary(output_root)
        print(f"[probe] task={task} stage=artifacts_written", flush=True)
        print(json.dumps({"task": task, "rows": len(rows), "result": str(task_root / "result.json")}, ensure_ascii=False, sort_keys=True), flush=True)
    _write_probe_summary(output_root)


def attach_candidate_probe(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    """Apply the fitted image-trained Ridge probe to stored candidate predictions."""
    tasks = PHASE15_TASKS if args.task == "all" else (str(args.task),)
    output_root = _resolve(args.output_root) / "diagnostics" / "probe"
    candidate_root = (
        args.candidate_records
        if args.candidate_records is not None
        else _resolve(args.output_root) / "diagnostics" / "candidate_pool"
    )
    diagnostics = config.get("diagnostics", {})
    expected_rows = (
        len(tuple(diagnostics.get("pool_flow_steps", (1, 2, 5, 10, 16, 32))))
        * int(diagnostics.get("pool_candidates_per_flow_step", 256))
        * 50
    )
    for task in tasks:
        task_root = output_root / task
        result_path = task_root / "result.json"
        feature_path = task_root / "features_targets.npz"
        if not result_path.is_file() or not feature_path.is_file():
            raise FileNotFoundError(
                f"completed probe artifacts are missing for {task}: {result_path}, {feature_path}"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "ok" or result.get("task") != task:
            raise ValueError(f"probe result is not complete for {task}: {result_path}")
        candidate_path = _candidate_records_path(candidate_root, task)
        candidate_records = _read_records(candidate_path)
        if len(candidate_records) != expected_rows:
            raise ValueError(
                f"candidate pool for {task} has {len(candidate_records)} rows; expected {expected_rows}"
            )
        with np.load(feature_path, allow_pickle=True) as archive:
            features = np.asarray(archive["features"], dtype=np.float32)
            targets = np.asarray(archive["targets"], dtype=np.float32)
            trajectory_ids = np.asarray(archive["trajectory_ids"], dtype=object)
        split = {
            key: tuple(values)
            for key, values in result["trajectory_split"].items()
            if key in {"train", "validation"}
        }
        ridge = fit_ridge_probe(
            features,
            targets,
            trajectory_ids,
            split=split,
            seed=int(result.get("seed", 2026)),
        )
        result["candidate_pool_probe"] = _probe_candidate_predictions(
            candidate_records,
            task=task,
            ridge=ridge,
            milestone=5,
        )
        result["candidate_pool_probe"]["records"] = str(candidate_path)
        result["candidate_pool_probe"]["probe"] = str(result_path)
        atomic_write_json(result_path, result)
        print(
            json.dumps(
                {
                    "task": task,
                    "candidate_pool_probe": result["candidate_pool_probe"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    _write_probe_summary(output_root)


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


def guidance_milestone_step_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Count paired guidance outcomes by their last shared physical milestone."""
    result: dict[str, Any] = {}
    expected_steps = (5, 10, 15, 20, 25)
    for mode in ("post_opt", "guided_flow"):
        selected = [record for record in records if record.get("guidance") == mode]
        counts = {str(step): 0 for step in expected_steps}
        unexpected: dict[str, int] = {}
        paired = 0
        unpaired = 0
        missing_step = 0
        for record in selected:
            if not bool(record.get("paired_outcome_valid", True)):
                unpaired += 1
                continue
            paired += 1
            raw_step = record.get("physical_comparison_step")
            if raw_step is None:
                missing_step += 1
                continue
            step = int(raw_step)
            if step in expected_steps:
                counts[str(step)] += 1
            else:
                key = str(step)
                unexpected[key] = unexpected.get(key, 0) + 1
        result[mode] = {
            "source_records": len(selected),
            "paired_records": paired,
            "unpaired_records": unpaired,
            "paired_records_missing_step": missing_step,
            "step_counts": counts,
            "unexpected_step_counts": dict(sorted(unexpected.items())),
        }
    return result


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
    if args.guidance_records:
        guidance_paths = list(args.guidance_records)
        records = [
            record
            for path in guidance_paths
            for record in _read_records(path)
        ]
        result["inputs"]["guidance_records"] = [
            str(_resolve(path)) for path in guidance_paths
        ]
        result["guidance_milestone_steps"] = guidance_milestone_step_summary(records)
        result["guidance"] = guidance_effect_metrics(records)
        result["paired_guidance"] = paired_guidance_metrics(
            records,
            bootstrap_samples=int(args.bootstrap_samples),
        )
        variant_fields = (
            "guidance",
            "flow_steps",
            "candidate_index",
            "guidance_inner_steps",
            "guidance_step_size",
            "max_rms_offset",
        )
        grouped_guidance: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for record in records:
            key = tuple(record.get(field) for field in variant_fields)
            grouped_guidance.setdefault(key, []).append(record)
        result["guidance_by_variant"] = [
            {
                "condition": dict(zip(variant_fields, key)),
                "guidance": guidance_effect_metrics(group_records),
                "paired_guidance": paired_guidance_metrics(
                    group_records,
                    bootstrap_samples=int(args.bootstrap_samples),
                    seed=2026 + index,
                ),
            }
            for index, (key, group_records) in enumerate(
                sorted(grouped_guidance.items(), key=lambda item: repr(item[0]))
            )
        ]
        result["matched_guidance_comparison"] = matched_guidance_mode_metrics(
            records,
            bootstrap_samples=int(args.bootstrap_samples),
        )
        result["matched_guidance_by_milestone"] = (
            matched_guidance_mode_metrics_by_milestone(
                records,
                bootstrap_samples=int(args.bootstrap_samples),
            )
        )
    if args.control_records is not None:
        records = _read_records(args.control_records)
        result["inputs"]["control_records"] = str(_resolve(args.control_records))
        result["controls"] = control_action_metrics(records)
    if len(result["inputs"]) == 0:
        raise ValueError(
            "summarize requires --candidate-records, --guidance-records, or "
            "--control-records"
        )
    output_name = "summary.json" if args.task == "all" else f"summary_{args.task}.json"
    output = _resolve(args.output_root) / "diagnostics" / output_name
    atomic_write_json(output, result)
    print(json.dumps({"summary": str(output), "inputs": result["inputs"]}, ensure_ascii=False, sort_keys=True))


def candidate_pool(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    if args.task == "all":
        raise ValueError("candidate-pool requires one task; run it once per task")
    lock_path = (
        _resolve(args.output_root)
        / "diagnostics"
        / "candidate_pool"
        / f".{args.task}.lock"
    )
    with _exclusive_file_lock(lock_path):
        _candidate_pool_locked(args, config)


def _current_control_artifacts_match(
    task_root: Path,
    *,
    task: str,
    flow_steps: Sequence[int],
    proposal_count: int,
    expected_states: int,
    checkpoint: Path,
    checkpoint_sha256: str,
    cohort: CohortManifest,
) -> bool:
    """Return whether controls already satisfy the exact planned protocol."""
    manifest_path = task_root / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    expected = {
        "schema_version": "round5_phase1_5_candidate_pool_manifest_v1",
        "status": "completed",
        "task": task,
        "flow_steps": [int(value) for value in flow_steps],
        "candidates_per_flow_step": int(proposal_count),
        "executed_candidates_per_flow_step": int(proposal_count),
        "states": int(expected_states),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "cohort_id": cohort.cohort_id,
        "cohort_sha256": cohort.computed_sha256,
        "control_actions": 222,
        "control_protocol_version": "phase1_5_fixed_pool_controls_v1",
        "control_status": "completed",
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return False

    records_value = payload.get("records")
    controls_value = payload.get("controls_records")
    if not isinstance(records_value, str) or not isinstance(controls_value, str):
        return False
    records_path = Path(records_value).resolve()
    controls_path = Path(controls_value).resolve()
    if records_path != (task_root / "records.jsonl").resolve():
        return False
    if controls_path != (task_root / "controls.jsonl").resolve():
        return False
    expected_candidate_rows = len(flow_steps) * int(proposal_count) * int(expected_states)
    expected_control_rows = 222 * int(expected_states)
    try:
        with records_path.open("rb") as stream:
            candidate_rows = sum(1 for _ in stream)
        with controls_path.open("rb") as stream:
            control_rows = sum(1 for _ in stream)
    except OSError:
        return False
    return candidate_rows == expected_candidate_rows and control_rows == expected_control_rows


def _candidate_pool_locked(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
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
        minimum_swap_free_mib=args.min_swap_free_mib,
    )
    manifest = _manifest(config, args.task)
    checkpoint, checkpoint_sha256 = _checkpoint(config, args.task)
    diagnostics = config.get("diagnostics", {})
    flow_steps = tuple(args.flow_steps or diagnostics.get("pool_flow_steps", (1, 2, 5, 10, 16, 32)))
    task_root = _resolve(args.output_root) / "diagnostics" / "candidate_pool" / args.task
    task_root.mkdir(parents=True, exist_ok=True)
    proposal_count = int(diagnostics.get("pool_candidates_per_flow_step", 256))
    if args.controls_only and _current_control_artifacts_match(
        task_root,
        task=args.task,
        flow_steps=flow_steps,
        proposal_count=proposal_count,
        expected_states=len(manifest.entries),
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        cohort=manifest,
    ):
        print(json.dumps(
            {
                "candidate_pool": str(task_root),
                "controls": str(task_root / "controls.jsonl"),
                "status": "completed",
                "controls_reused": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        ))
        return
    model, resolved = load_policy_or_model(str(checkpoint))
    if resolved is not None and Path(resolved).resolve() != checkpoint.resolve():
        raise ValueError(f"checkpoint resolver changed requested path: {resolved}")
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
                "candidate_noise_records": {
                    str(flow): str(task_root / f"candidate_noise_s{int(flow)}.npz")
                    for flow in flow_steps
                },
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
    goal_latents = _encode_goal_latents(
        session.dataset,
        manifest,
        session.transform["pixels"],
        model,
        args.device,
    )
    completed_files: list[Path] = []
    branch_limit = proposal_count if args.limit_candidates is None else min(proposal_count, int(args.limit_candidates))
    if branch_limit <= 0:
        raise ValueError("--limit-candidates must be positive")
    branch_replay: dict[str, Any] = {}
    if not args.controls_only:
        for flow_step in flow_steps:
            flow_proposals = [row for row in proposals if int(row["flow_steps"]) == int(flow_step)]
            by_candidate: dict[int, list[dict[str, Any]]] = {}
            for row in flow_proposals:
                by_candidate.setdefault(int(row["candidate_index"]), []).append(row)
            for candidate_index in range(branch_limit):
                branch_path = task_root / "branches" / f"s{int(flow_step)}" / f"candidate_{candidate_index:04d}.jsonl"
                rows = sorted(by_candidate.get(candidate_index, []), key=lambda item: int(item["slot"]))
                if len(rows) != len(manifest.entries):
                    raise RuntimeError(
                        f"candidate {candidate_index} at S={flow_step} has {len(rows)} states; "
                        f"expected {len(manifest.entries)}"
                    )
                if _branch_file_matches(branch_path, rows):
                    completed_files.append(branch_path)
                    continue
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
                    dataset=session.dataset,
                    branch_state=branch_replay,
                    goal_latents=goal_latents,
                )
                outcomes = _attach_candidate_outcomes(rows, episodes, task=args.task)
                _write_jsonl(branch_path, outcomes)
                completed_files.append(branch_path)
    aggregate = task_root / "records.jsonl"
    if args.controls_only:
        previous_manifest_path = task_root / "manifest.json"
        if not previous_manifest_path.is_file():
            raise RuntimeError("--controls-only requires a completed candidate-pool manifest")
        previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        if (
            previous_manifest.get("status") != "completed"
            or int(previous_manifest.get("executed_candidates_per_flow_step", -1)) != proposal_count
            or {int(value) for value in previous_manifest.get("flow_steps", [])}
            != {int(value) for value in flow_steps}
        ):
            raise RuntimeError("--controls-only requires all candidate branches to be complete")
    else:
        temporary = aggregate.with_name(f".{aggregate.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for path in sorted(completed_files):
                stream.write(path.read_text(encoding="utf-8"))
        temporary.replace(aggregate)

    if not args.controls_only and not {1, 2}.issubset({int(value) for value in flow_steps}):
        raise ValueError("control-pool execution requires flow steps 1 and 2")
    control_files: list[Path] = []
    control_metadata: list[dict[str, Any]] = []
    control_aggregate = task_root / "controls.jsonl"
    if not args.proposal_only:
        control_actions, control_metadata = _control_pool_actions(
            dataset=session.dataset,
            manifest=manifest,
            process=process,
            proposals=proposals,
            seed=int(config.get("diagnostics", {}).get("pool_seed", 2026)),
            action_block=int(cfg.plan_config.action_block),
        )
        control_root = task_root / "controls"
        for control_index, metadata in enumerate(control_metadata):
            control_path = control_root / f"control_{control_index:04d}.jsonl"
            rows = []
            for slot, action in enumerate(control_actions[control_index]):
                entry = manifest.entries[slot]
                rows.append(
                    {
                        "task": args.task,
                        "state_id": _candidate_state_id(entry),
                        "slot": slot,
                        "episode_id": entry.episode_id,
                        "start_step": int(entry.start_step),
                        "row_index": int(entry.row_index),
                        "flow_steps": "control",
                        "candidate_index": None,
                        "control_index": control_index,
                        "control_kind": metadata["kind"],
                        "control_metadata": metadata,
                        "predicted_cost": None,
                        "action": np.asarray(action, dtype=np.float64).tolist(),
                        "outcome_status": "pending",
                    }
                )
            if _branch_file_matches(control_path, rows):
                control_files.append(control_path)
                continue
            episodes = _run_fixed_candidate(
                cfg=cfg,
                task=args.task,
                manifest=manifest,
                normalized_actions=_pack_primitive_actions_for_replay(
                    control_actions[control_index],
                    action_block=int(cfg.plan_config.action_block),
                    horizon=int(cfg.plan_config.horizon),
                ),
                process=process,
                model=model,
                transform=session.transform["pixels"],
                device=args.device,
                output_dir=control_path.parent / control_path.stem,
                dataset=session.dataset,
                branch_state=branch_replay,
                goal_latents=goal_latents,
            )
            _write_jsonl(control_path, _attach_candidate_outcomes(rows, episodes, task=args.task))
            control_files.append(control_path)
        control_temp = control_aggregate.with_name(f".{control_aggregate.name}.{os.getpid()}.tmp")
        with control_temp.open("w", encoding="utf-8") as stream:
            for path in sorted(control_files):
                stream.write(path.read_text(encoding="utf-8"))
        control_temp.replace(control_aggregate)
    candidate_complete = args.controls_only or branch_limit == proposal_count
    controls_complete = args.proposal_only or len(control_files) == len(control_metadata) == 222
    complete = candidate_complete and controls_complete
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
            "candidate_noise_records": {
                str(flow): str(task_root / f"candidate_noise_s{int(flow)}.npz")
                for flow in flow_steps
            },
            "controls_records": None if args.proposal_only else str(control_aggregate),
            "control_actions": len(control_metadata),
            "control_protocol_version": "phase1_5_fixed_pool_controls_v1",
            "control_status": "completed" if controls_complete else "partial",
            "state_replay": "captured_before_first_primitive_step_and_restored_with_rng",
            "candidate_noise_provenance": "sha256_checked_across_flow_steps",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "cohort_id": manifest.cohort_id,
            "cohort_sha256": manifest.computed_sha256,
        },
    )
    print(json.dumps({"candidate_pool": str(task_root), "records": str(aggregate), "controls": None if args.proposal_only else str(control_aggregate), "status": "completed" if complete else "partial"}, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "validate",
            "probe",
            "attach-candidate-probe",
            "summarize",
            "candidate-pool",
            "guidance-pool",
            "guidance-sweep",
        ),
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--task", choices=(*PHASE15_TASKS, "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=_gpu)
    parser.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB)
    parser.add_argument("--min-available-mib", type=int, default=DEFAULT_MIN_AVAILABLE_MIB)
    parser.add_argument("--min-swap-free-mib", type=int, default=DEFAULT_MIN_SWAP_FREE_MIB)
    parser.add_argument("--max-load-per-cpu", type=float, default=DEFAULT_MAX_LOAD_PER_CPU)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-trajectories", type=int, default=1000)
    parser.add_argument("--frames-per-trajectory", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mlp", action="store_true")
    parser.add_argument("--candidate-records")
    parser.add_argument("--control-records")
    parser.add_argument("--guidance-records", action="append")
    parser.add_argument("--flow-steps", type=int, nargs="+")
    parser.add_argument("--candidates-per-flow-step", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--proposal-only", action="store_true")
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument("--limit-candidates", type=int)
    parser.add_argument("--guidance", choices=("post_opt", "guided_flow"), default="post_opt")
    parser.add_argument("--guidance-step-size", type=float, default=0.01)
    parser.add_argument("--guidance-last-steps", type=int)
    parser.add_argument("--guidance-inner-steps", type=int, default=5)
    parser.add_argument("--max-rms-offset", type=float, default=0.2)
    parser.add_argument("--limit-guidance-variants", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _load_config(_resolve(args.config))
    if args.command == "validate":
        validate(args, config)
    elif args.command == "probe":
        probe(args, config)
    elif args.command == "attach-candidate-probe":
        attach_candidate_probe(args, config)
    elif args.command == "candidate-pool":
        candidate_pool(args, config)
    elif args.command == "guidance-pool":
        guidance_pool(args, config)
    elif args.command == "guidance-sweep":
        guidance_sweep(args, config)
    else:
        summarize(args, config)


if __name__ == "__main__":
    main()
