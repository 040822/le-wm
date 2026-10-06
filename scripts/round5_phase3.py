#!/usr/bin/env python3
"""Round 5 Phase 3 preparation, evaluation, and analysis driver."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.checkpoint import load_policy_or_model
from source.common.eval import DatasetEvaluationSession, EvaluationIdentity, compose_eval_config
from source.common.phase3_compat import (
    PHASE3_DATASETS,
    PHASE3_TASKS,
    _phase3_initial_success_flags,
    build_phase3_initial_failure_manifest,
    build_phase3_legacy_manifest,
    patch_phase3_environments,
    scene_target_group,
)
from source.common.round3_phase1 import CohortManifest, wilson_interval
from source.common.round4_eval import run_round4_evaluation


OUTPUT_ROOT = ROOT / "outputs" / "round5" / "phase3"
REPORT_PATH = ROOT / "docs" / "report" / "round5" / "round5_phase3_report.md"
INITIAL_FAILURE_REPORT_PATH = ROOT / "docs" / "report" / "round5" / "round5_phase3_initial_failure_report.md"
SEED = 3072
EVAL_SEED = 42
NUM_EVAL = 50
GOAL_OFFSET = 25
EVAL_BUDGET = 50
MIN_FREE_MIB = 5000
COHORT_NAMES = ("legacy_50", "initial_fail_50")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_keys(task: str) -> list[str]:
    # Evaluation validation needs the episode/step columns in addition to the
    # model inputs; the frozen cohort uses these same identifiers to verify
    # every sampled start/goal pair.
    common = ["ep_idx", "step_idx", "pixels", "action", "qpos", "qvel"]
    if task == "scene":
        return common + [
            "button_states",
            "privileged_target_task",
            "privileged_block_0_pos",
            "privileged_block_0_quat",
            "privileged_button_0_state",
            "privileged_button_1_state",
            "privileged_drawer_pos",
            "privileged_window_pos",
        ]
    if task == "finger":
        return common + ["target_position"]
    if task == "humanoid":
        return common
    raise ValueError(task)


def _load_dataset(
    task: str, *, cohort_only: bool = False, initial_filter_only: bool = False
):
    from stable_worldmodel.data.formats.hdf5 import HDF5Dataset

    if cohort_only and initial_filter_only:
        raise ValueError("cohort_only and initial_filter_only are mutually exclusive")
    if cohort_only:
        keys = ["ep_idx", "step_idx"]
    elif initial_filter_only:
        keys = {
            "scene": [
                "ep_idx", "step_idx", "privileged_block_0_pos",
                "privileged_button_0_state", "privileged_button_1_state",
                "privileged_drawer_pos", "privileged_window_pos",
            ],
            "finger": ["ep_idx", "step_idx", "tip_position", "target_position"],
            "humanoid": ["ep_idx", "step_idx", "head_height", "torso_upright", "speed"],
        }[task]
    else:
        keys = _dataset_keys(task)
    return HDF5Dataset(path=PHASE3_DATASETS[task], keys_to_load=keys)


def _manifest_path(
    task: str, output_root: Path = OUTPUT_ROOT, cohort_name: str = "legacy_50"
) -> Path:
    if cohort_name not in COHORT_NAMES:
        raise ValueError(f"unsupported Phase 3 cohort name: {cohort_name!r}")
    return output_root / "cohorts" / task / f"{cohort_name}.json"


def _ensure_manifest(
    task: str,
    output_root: Path = OUTPUT_ROOT,
    cohort_name: str = "legacy_50",
) -> CohortManifest:
    path = _manifest_path(task, output_root, cohort_name)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing frozen cohort {path}; run the matching Phase 3 prepare command first"
        )
    manifest = CohortManifest.load(path)
    expected_variant = "legacy" if cohort_name == "legacy_50" else "sampling_revised"
    if manifest.task != task or manifest.protocol_variant != expected_variant or len(manifest.entries) != NUM_EVAL:
        raise ValueError(f"invalid Phase 3 cohort: {path}")
    if cohort_name == "initial_fail_50" and any(
        entry.initially_successful is not False for entry in manifest.entries
    ):
        raise ValueError(f"initial-failure cohort contains an unverified start: {path}")
    return manifest


def _compose(task: str, *, num_eval: int = NUM_EVAL, save_video: bool = False):
    return compose_eval_config(
        task,
        overrides=[
            f"eval.num_eval={int(num_eval)}",
            f"eval.goal_offset_steps={GOAL_OFFSET}",
            f"eval.eval_budget={EVAL_BUDGET}",
            f"output.save_video={'true' if save_video else 'false'}",
            "solver.num_samples=300",
            "solver.topk=30",
            "solver.n_steps=30",
            "solver.var_scale=1.0",
        ],
    )


def _gpu(value: str | None) -> str | None:
    if value is None:
        return None
    if not value.isdigit() or int(value) not in range(8):
        raise argparse.ArgumentTypeError("--gpu must select one physical GPU0-7")
    return value


def _gpu_preflight(gpu: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            str(gpu),
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [item.strip() for item in completed.stdout.splitlines()[0].split(",")]
    if len(fields) < 6:
        raise RuntimeError(f"invalid nvidia-smi output for GPU{gpu}: {completed.stdout!r}")
    snapshot = {
        "gpu": int(gpu),
        "name": fields[1],
        "memory_total_mib": int(fields[2]),
        "memory_used_mib": int(fields[3]),
        "memory_free_mib": int(fields[4]),
        "utilization_percent": int(fields[5]),
    }
    if snapshot["memory_free_mib"] < MIN_FREE_MIB:
        raise RuntimeError(
            f"GPU{gpu} has {snapshot['memory_free_mib']} MiB free; "
            f"Phase 3 requires at least {MIN_FREE_MIB} MiB"
        )
    print(json.dumps({"gpu_preflight": snapshot}, sort_keys=True), flush=True)
    return snapshot


def _configure_gpu(gpu: str | None, device: str) -> None:
    if str(device).startswith("cuda"):
        if gpu is None:
            raise ValueError("CUDA evaluation requires --gpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        _gpu_preflight(gpu)
        # configure_mujoco_egl_device is called by the shared evaluator after
        # this process has selected exactly one physical GPU.


def _condition_name(spec: Mapping[str, Any]) -> str:
    step = "invariant" if spec["step"] is None else f"step_{int(spec['step'])}"
    return "/".join(
        (
            str(spec["mode"]),
            str(spec["protocol"]),
            str(spec["guidance"]),
            step,
            "euler" if spec["step"] is not None else "not_applicable",
        )
    )


def p2_protocol(task: str) -> str:
    return "legacy" if task == "scene" else "cem-clip"


def fast_conditions(task: str) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    steps = (1, 2, 5, 10, 16, 32)
    protocol = p2_protocol(task)
    for step in steps:
        conditions.append({"task": task, "mode": "P0", "protocol": "not_applicable", "step": step, "guidance": "none"})
    conditions.append({"task": task, "mode": "P1", "protocol": protocol, "step": None, "guidance": "none"})
    for step in steps:
        conditions.append({"task": task, "mode": "P2", "protocol": protocol, "step": step, "guidance": "none"})
    for step in steps:
        conditions.append({"task": task, "mode": "P3", "protocol": "not_applicable", "step": step, "guidance": "none"})
    for step in steps:
        for guidance in ("guided_flow", "post_opt"):
            conditions.append({"task": task, "mode": "P0", "protocol": "not_applicable", "step": step, "guidance": guidance})
            conditions.append({"task": task, "mode": "P2", "protocol": protocol, "step": step, "guidance": guidance})
        for guidance in ("guided_flow", "post_opt", "post_opt_refine"):
            conditions.append({"task": task, "mode": "P3", "protocol": "not_applicable", "step": step, "guidance": guidance})
    return conditions


def _condition_dir(output_root: Path, task: str, spec: Mapping[str, Any]) -> Path:
    return output_root / "fastlewam" / task / _condition_name(spec)


def _video_manifest(manifest: CohortManifest) -> CohortManifest:
    """Make a one-entry view of the frozen cohort for representative videos."""
    entry = manifest.entries[0]
    owner_bucket = "selected" if manifest.protocol_variant == "legacy" else "dev"
    return replace(
        manifest,
        cohort_id=f"{manifest.cohort_id}_video_0",
        entries=(entry,),
        episode_split={owner_bucket: (entry.episode_id,)},
        candidate_counts=(1,),
        selected_counts=(1,),
        sampling_rule={
            **dict(manifest.sampling_rule),
            "video_source_cohort_sha256": manifest.computed_sha256,
            "video_entry_index": 0,
        },
        diagnostics={
            **dict(manifest.diagnostics),
            "video_source_cohort_sha256": manifest.computed_sha256,
        },
        cohort_sha256=None,
    )


def _video_dir(output_root: Path, method: str, task: str, condition: str) -> Path:
    return output_root / "videos" / method / task / condition


def _result_complete(path: Path) -> bool:
    result = path / "result.json"
    if not result.is_file():
        return False
    try:
        payload = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "ok" and len(payload.get("episodes", [])) == NUM_EVAL


def _training_config_path(method: str, task: str, output_root: Path) -> Path:
    training_root = output_root / "training" / method / task
    candidates = [training_root / "config.json", training_root / "config.yaml", training_root / "config.yml"]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"expected exactly one training config for {method}/{task} under {training_root}; "
            f"found {[str(path) for path in existing]}"
        )
    return existing[0]


def _training_metadata(method: str, task: str, output_root: Path) -> dict[str, Any]:
    config_path = _training_config_path(method, task, output_root)
    if config_path.suffix == ".json":
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or config.get("seed") is None:
        raise ValueError(f"training config has no seed: {config_path}")
    seed = int(config["seed"])
    if seed != SEED:
        raise ValueError(f"unexpected training seed in {config_path}: {seed} != {SEED}")
    metadata = {
        "seed": seed,
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
    }
    resume_path = output_root / "training" / method / task / "resume_metadata.json"
    if resume_path.is_file():
        metadata["resume_metadata_path"] = str(resume_path.resolve())
        metadata["resume_metadata_sha256"] = _sha256(resume_path)
    return metadata


def _validate_training_outputs(output_root: Path) -> list[dict[str, Any]]:
    checkpoint_names = {
        "fastlewam": "checkpoints/r4_ab_weights_epoch_10.pt",
        "lewm": "checkpoints/lewm_weights_epoch_10.pt",
        "leflow": "latent_planner_epoch_10.pt",
    }
    outputs = []
    for task in PHASE3_TASKS:
        for method, relative_checkpoint in checkpoint_names.items():
            training_root = output_root / "training" / method / task
            checkpoint = training_root / relative_checkpoint
            if not checkpoint.is_file():
                raise FileNotFoundError(f"missing epoch 10 checkpoint: {checkpoint}")
            metadata = _training_metadata(method, task, output_root)
            if method == "leflow":
                completion_path = training_root / "training_complete.json"
                if not completion_path.is_file():
                    raise FileNotFoundError(f"missing LeFlow completion marker: {completion_path}")
                completion = json.loads(completion_path.read_text(encoding="utf-8"))
                if completion.get("status") != "ok" or completion.get("epochs") != 10:
                    raise ValueError(f"LeFlow training did not complete 10 epochs: {completion_path}")
                final_checkpoint = training_root / "latent_planner.pt"
                if not final_checkpoint.is_file():
                    raise FileNotFoundError(f"missing final LeFlow checkpoint: {final_checkpoint}")
            outputs.append(
                {
                    "method": method,
                    "task": task,
                    "epoch": 10,
                    "checkpoint": str(checkpoint.resolve()),
                    "training_seed": metadata["seed"],
                    "training_config_path": metadata["config_path"],
                    "training_config_sha256": metadata["config_sha256"],
                }
            )
    return outputs


def _attach_result_metadata(
    payload: dict[str, Any], *, method: str, task: str, output_root: Path
) -> None:
    episodes = payload.get("episodes", [])
    if not episodes:
        raise ValueError("cannot attach Phase 3 metadata to a result without episodes")
    parameters = payload.get("parameters")
    evaluation_seed = parameters.get("seed") if isinstance(parameters, Mapping) else None
    if evaluation_seed is None or int(evaluation_seed) != EVAL_SEED:
        raise ValueError(
            f"unexpected evaluation seed in {task}/{method}: {evaluation_seed} != {EVAL_SEED}"
        )
    success_vector = [bool(item.get("success", False)) for item in episodes]
    payload.update(
        {
            "training_seed": SEED,
            "training_config": _training_metadata(method, task, output_root),
            "evaluation_seed": EVAL_SEED,
            "episode_success_vector": success_vector,
            "success_count": int(sum(success_vector)),
            "success_rate_percent": float(np.mean(success_vector) * 100.0),
        }
    )


def _annotate_result_file(
    path: Path, *, method: str, task: str, output_root: Path
) -> dict[str, Any]:
    result_path = path / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    _attach_result_metadata(payload, method=method, task=task, output_root=output_root)
    temporary_path = result_path.with_suffix(".json.metadata.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_path, result_path)
    return payload


def _validate_video_result(
    payload: Mapping[str, Any], *, manifest: CohortManifest, result_path: Path
) -> str:
    if payload.get("status") != "ok" or len(payload.get("episodes", [])) != 1:
        raise ValueError(f"invalid representative video result: {result_path}")
    metadata = payload.get("phase3_video")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"missing representative video metadata: {result_path}")
    if metadata.get("source_cohort_sha256") != manifest.computed_sha256:
        raise ValueError(f"representative video cohort mismatch: {result_path}")
    if (
        metadata.get("source_cohort_id") != manifest.cohort_id
        or metadata.get("source_entry_index") != 0
    ):
        raise ValueError(
            f"representative video does not identify the first frozen cohort entry: "
            f"{result_path}"
        )
    first_entry = manifest.entries[0]
    episode = payload["episodes"][0]
    if (
        episode.get("dataset_episode") != first_entry.episode_id
        or episode.get("start_step") != first_entry.start_step
    ):
        raise ValueError(
            f"representative video start does not match the first frozen cohort entry: "
            f"{result_path}"
        )
    for field in ("row_index", "goal_row_index"):
        actual = episode.get(field)
        expected = getattr(first_entry, field)
        if actual is not None and actual != expected:
            raise ValueError(
                f"representative video {field} does not match the first frozen "
                f"cohort entry: {result_path}"
            )
    return str(metadata["source_cohort_sha256"])


def _validate_video_file(video_path: Path) -> dict[str, int]:
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty representative video: {video_path}")
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type,width,height",
            "-of",
            "json",
            str(video_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise ValueError(f"ffprobe could not read {video_path}: {probe.stderr.strip()}")
    streams = json.loads(probe.stdout).get("streams", [])
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    if not video_streams:
        raise ValueError(f"representative video has no decodable video stream: {video_path}")
    width = int(video_streams[0].get("width", 0))
    height = int(video_streams[0].get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"representative video has invalid dimensions: {video_path}")
    return {"width": width, "height": height}


def command_prepare(args: argparse.Namespace) -> None:
    if args.cohort_name != "legacy_50":
        raise ValueError("prepare creates only the legacy_50 cohort")
    output_root = Path(args.output_root).resolve()
    for task in PHASE3_TASKS:
        dataset = _load_dataset(task, cohort_only=True)
        manifest = build_phase3_legacy_manifest(
            dataset,
            task=task,
            seed=EVAL_SEED,
            goal_offset_steps=GOAL_OFFSET,
            num_eval=NUM_EVAL,
        )
        path = _manifest_path(task, output_root)
        manifest.save(path)
        print(json.dumps({"task": task, "path": str(path), "cohort_sha256": manifest.computed_sha256}, sort_keys=True))


def command_prepare_initial_fail(args: argparse.Namespace) -> None:
    if args.cohort_name != "initial_fail_50":
        raise ValueError("prepare-initial-fail requires --cohort-name initial_fail_50")
    output_root = Path(args.output_root).resolve()
    reference_root = Path(args.reference_output_root).resolve()
    for task in PHASE3_TASKS:
        reference_manifest = _ensure_manifest(task, reference_root, "legacy_50")
        dataset = _load_dataset(task, initial_filter_only=True)
        manifest = build_phase3_initial_failure_manifest(
            dataset,
            task=task,
            reference_manifest=reference_manifest,
            seed=EVAL_SEED,
            selection_seed=args.selection_seed,
            goal_offset_steps=GOAL_OFFSET,
            num_eval=NUM_EVAL,
        )
        path = _manifest_path(task, output_root, args.cohort_name)
        manifest.save(path)
        print(
            json.dumps(
                {
                    "task": task,
                    "path": str(path),
                    "cohort_id": manifest.cohort_id,
                    "cohort_sha256": manifest.computed_sha256,
                    "sampling_rule": manifest.sampling_rule,
                    "diagnostics": manifest.diagnostics,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def command_smoke(args: argparse.Namespace) -> None:
    import gymnasium as gym

    patch_phase3_environments()
    _configure_gpu(args.gpu, args.device)
    env_names = {
        "scene": "swm/OGBScene-v0",
        "finger": "swm/FingerDMControl-v0",
        "humanoid": "swm/HumanoidDMControl-v0",
    }
    for task in PHASE3_TASKS:
        if task == "scene":
            env = gym.make(env_names[task], max_episode_steps=100, ob_type="states", terminate_at_goal=True, visualize_info=False)
        elif task == "finger":
            env = gym.make(env_names[task], max_episode_steps=100, task="turn_hard")
        else:
            env = gym.make(env_names[task], max_episode_steps=100, task="walk")
        try:
            _, info = env.reset(seed=0)
            raw = env.unwrapped
            if task == "scene":
                qpos = np.asarray(raw._data.qpos).copy()
                qvel = np.asarray(raw._data.qvel).copy()
                raw.set_state(qpos, qvel, button_states=np.asarray([0, 0]))
                cube_qpos = np.asarray(raw._data.joint("object_joint_0").qpos).copy()
                drawer_pos = float(raw._data.joint("drawer_slide").qpos[0])
                window_pos = float(raw._data.joint("window_slide").qpos[0])
                raw.set_phase3_scene_goal(
                    cube_qpos[:3], cube_qpos[3:], 0, 0, drawer_pos, window_pos
                )
            elif task == "finger":
                raw.set_target_position(np.asarray(raw.env.physics.target_position()).copy())
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            env.step(action)
            print(json.dumps({"task": task, "status": "ok", "info_keys": sorted(str(k) for k in info)[:8]}, sort_keys=True))
        finally:
            env.close()


def command_eval_fast(args: argparse.Namespace) -> None:
    task = args.task
    output_root = Path(args.output_root).resolve()
    manifest = _ensure_manifest(task, output_root, args.cohort_name)
    patch_phase3_environments()
    _configure_gpu(args.gpu, args.device)
    cfg = _compose(task)
    dataset = _load_dataset(task)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model, resolved = load_policy_or_model(str(checkpoint))
    conditions = fast_conditions(task)
    if args.condition_index is not None:
        conditions = [conditions[int(index)] for index in args.condition_index]
    for spec in conditions:
        target = _condition_dir(output_root, task, spec)
        if _result_complete(target):
            print(json.dumps({"status": "reused", "path": str(target / "result.json")}, sort_keys=True), flush=True)
            continue
        target.mkdir(parents=True, exist_ok=True)
        if any(target.iterdir()):
            raise FileExistsError(f"refusing to overwrite partial condition: {target}")
        mode = str(spec["mode"])
        result = run_round4_evaluation(
            cfg,
            task=task,
            policy_or_model=model,
            mode=mode,
            identity={
                "entrypoint": "round5_phase3",
                "policy_kind": "fast_lewam",
                "checkpoint": str(resolved or checkpoint),
                "epoch": 10,
                "stage": mode,
                "guidance_mode": str(spec["guidance"]),
            },
            manifest=manifest,
            output_dir=target,
            dataset=dataset,
            device=args.device,
            trace=False,
            candidate_count=64,
            flow_steps=16,
            action_flow_steps=spec["step"],
            action_flow_integrator="euler",
            cem_protocol=spec["protocol"],
            action_bound_mode=None,
            guidance_mode=str(spec["guidance"]),
            guidance_step_size=0.01,
            guidance_last_steps=5,
            guidance_inner_steps=5,
            guidance_max_rms_offset=0.20,
            proposal_chunk_size=512,
            allowed_protocol_variants=(manifest.protocol_variant,),
        )
        payload = json.loads((target / "result.json").read_text(encoding="utf-8"))
        payload.update(
            {
                "phase3_method": "fast_lewam",
                "phase3_condition": dict(spec),
                "success_rate_percent": float(np.mean([bool(item["success"]) for item in payload["episodes"]]) * 100.0),
            }
        )
        _attach_result_metadata(
            payload, method="fastlewam", task=task, output_root=output_root
        )
        (target / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "ok", "task": task, "condition": _condition_name(spec), "success_rate": result["success_rate"]}, sort_keys=True), flush=True)


def command_eval_video(args: argparse.Namespace) -> None:
    """Generate one representative video while preserving the 50-episode evals."""
    task = args.task
    method = args.method
    condition = args.condition
    if method == "fastlewam" and condition not in {"p1_step_1", "p3_step_1"}:
        raise ValueError("FastLeWAM video condition must be p1_step_1 or p3_step_1")
    if method != "fastlewam" and condition != "standard":
        raise ValueError("LeWM and LeFlow video condition must be standard")

    output_root = Path(args.output_root).resolve()
    source_manifest = _ensure_manifest(task, output_root, args.cohort_name)
    source_hash = source_manifest.computed_sha256
    manifest = _video_manifest(source_manifest)
    target = _video_dir(output_root, method, task, condition)
    video_path = target / "videos" / "env_0.mp4"
    result_path = target / "result.json"
    if result_path.is_file() and video_path.is_file():
        try:
            video_result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            video_result = {}
        if (
            video_result.get("status") == "ok"
            and len(video_result.get("episodes", [])) == 1
        ):
            print(json.dumps({"status": "reused", "path": str(video_path)}, sort_keys=True), flush=True)
            return
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"refusing to overwrite partial video artifact: {target}")

    patch_phase3_environments()
    _configure_gpu(args.gpu, args.device)
    cfg = _compose(task, num_eval=1, save_video=True)
    dataset = _load_dataset(task)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    policy_or_model, resolved = load_policy_or_model(str(checkpoint))

    if method == "fastlewam":
        mode = "P1" if condition == "p1_step_1" else "P3"
        result = run_round4_evaluation(
            cfg,
            task=task,
            policy_or_model=policy_or_model,
            mode=mode,
            identity={
                "entrypoint": "round5_phase3_video",
                "policy_kind": "fast_lewam",
                "checkpoint": str(resolved or checkpoint),
                "epoch": 10,
                "stage": mode,
                "guidance_mode": "none",
            },
            manifest=manifest,
            output_dir=target,
            dataset=dataset,
            device=args.device,
            trace=False,
            candidate_count=64,
            flow_steps=16,
            action_flow_steps=1 if mode == "P3" else None,
            action_flow_integrator="euler",
            cem_protocol=p2_protocol(task) if mode == "P1" else "not_applicable",
            action_bound_mode=None,
            guidance_mode="none",
            proposal_chunk_size=512,
            allowed_protocol_variants=(manifest.protocol_variant,),
        )
    else:
        identity = EvaluationIdentity(
            entrypoint="round5_phase3_video",
            policy_kind=method,
            checkpoint=str(resolved or checkpoint),
            epoch=10,
            stage=None,
        )
        session = DatasetEvaluationSession(
            cfg,
            task=task,
            dataset=dataset,
            cohort=manifest.to_evaluation_cohort(),
        )
        result = session.evaluate(
            policy_or_model,
            identity=identity,
            output_dir=target,
            device=args.device,
        )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["phase3_video"] = {
        "method": method,
        "task": task,
        "condition": condition,
        "source_cohort_sha256": source_hash,
        "source_cohort_id": source_manifest.cohort_id,
        "source_entry_index": 0,
        "video": str(video_path.relative_to(target)),
    }
    _attach_result_metadata(
        payload,
        method="fastlewam" if method == "fastlewam" else method,
        task=task,
        output_root=output_root,
    )
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "task": task, "method": method, "condition": condition, "video": str(video_path)}, sort_keys=True), flush=True)


def _baseline_dir(output_root: Path, method: str, task: str) -> Path:
    return output_root / "baselines" / method / task


def command_eval_baseline(args: argparse.Namespace) -> None:
    task = args.task
    method = args.method
    output_root = Path(args.output_root).resolve()
    manifest = _ensure_manifest(task, output_root, args.cohort_name)
    target = _baseline_dir(output_root, method, task)
    if _result_complete(target):
        print(json.dumps({"status": "reused", "path": str(target / "result.json")}, sort_keys=True), flush=True)
        return
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"refusing to overwrite partial baseline: {target}")
    patch_phase3_environments()
    _configure_gpu(args.gpu, args.device)
    cfg = _compose(task)
    dataset = _load_dataset(task)
    policy_or_model, resolved = load_policy_or_model(str(Path(args.checkpoint).resolve()))
    identity = EvaluationIdentity(
        entrypoint="round5_phase3",
        policy_kind=method,
        checkpoint=str(resolved or args.checkpoint),
        epoch=10,
        stage=None,
    )
    session = DatasetEvaluationSession(
        cfg,
        task=task,
        dataset=dataset,
        cohort=manifest.to_evaluation_cohort(),
    )
    result = session.evaluate(
        policy_or_model,
        identity=identity,
        output_dir=target,
        device=args.device,
    )
    result_path = target / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    successes = [bool(item["success"]) for item in payload["episodes"]]
    payload.update(
        {
            "phase3_method": method,
            "phase3_condition": {
                "mode": "standard",
                "cem_protocol": "legacy",
                "num_samples": 300 if method == "lewm" else 64,
                "flow_steps": None if method == "lewm" else 16,
            },
            "cohort_sha256": manifest.computed_sha256,
            "cohort_id": manifest.cohort_id,
            "success_rate_percent": float(np.mean(successes) * 100.0),
        }
    )
    _attach_result_metadata(
        payload, method=method, task=task, output_root=output_root
    )
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "task": task, "method": method, "success_rate_percent": float(result.success_rate)}, sort_keys=True), flush=True)


def _successes(payload: Mapping[str, Any]) -> list[bool]:
    episodes = payload.get("episodes", [])
    if len(episodes) != NUM_EVAL:
        raise ValueError("Phase 3 result must contain exactly 50 episodes")
    values = []
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("each Phase 3 episode must be a mapping")
        value = episode.get("success")
        if type(value) is not bool:
            raise ValueError("each Phase 3 episode must contain a boolean runtime success value")
        values.append(value)
    return values


def _mcnemar(left: Sequence[bool], right: Sequence[bool]) -> dict[str, Any]:
    improved = sum((not a) and b for a, b in zip(left, right))
    regressed = sum(a and (not b) for a, b in zip(left, right))
    discordant = improved + regressed
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1)) / (2.0 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"improved": int(improved), "regressed": int(regressed), "mcnemar_exact_two_sided_p": float(p_value)}


def _legacy_primary_decomposition() -> list[dict[str, Any]]:
    """Split the original primary results by whether their start already passed."""

    analysis_path = OUTPUT_ROOT / "analysis" / "analysis.json"
    if not analysis_path.is_file():
        raise FileNotFoundError(f"missing original Phase 3 analysis: {analysis_path}")
    legacy_analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    decomposition = []
    for task in PHASE3_TASKS:
        manifest = _ensure_manifest(task, OUTPUT_ROOT, "legacy_50")
        dataset = _load_dataset(task, initial_filter_only=True)
        start_rows = np.asarray([entry.row_index for entry in manifest.entries], dtype=np.int64)
        goal_rows = np.asarray(
            [entry.goal_row_index for entry in manifest.entries], dtype=np.int64
        )
        initially_successful = _phase3_initial_success_flags(
            dataset,
            task=task,
            start_rows=start_rows,
            goal_rows=goal_rows,
        )
        for method in ("fast_lewam", "lewm", "leflow"):
            row = next(
                item
                for item in legacy_analysis["primary"]
                if item["task"] == task and item["method"] == method
            )
            payload = json.loads(Path(row["path"]).read_text(encoding="utf-8"))
            if payload.get("cohort_sha256") != manifest.computed_sha256:
                raise ValueError(
                    f"legacy primary result cohort mismatch: {row['path']}"
                )
            episodes = payload.get("episodes", [])
            if len(episodes) != NUM_EVAL:
                raise ValueError(f"legacy primary result must contain 50 episodes: {row['path']}")
            for index, episode in enumerate(episodes):
                episode_id = episode.get("episode_id", episode.get("dataset_episode"))
                if (
                    episode_id != manifest.entries[index].episode_id
                    or int(episode.get("start_step", -1))
                    != manifest.entries[index].start_step
                ):
                    raise ValueError(
                        f"legacy primary result order does not match its manifest: {row['path']}"
                    )
            successes = np.asarray(
                [bool(episode.get("success", False)) for episode in episodes],
                dtype=bool,
            )
            initial_failures = ~initially_successful
            conditional_successes = int(np.logical_and(successes, initial_failures).sum())
            denominator = int(initial_failures.sum())
            decomposition.append(
                {
                    "task": task,
                    "method": method,
                    "legacy_successes": int(successes.sum()),
                    "legacy_episodes": len(successes),
                    "legacy_success_rate_percent": float(successes.mean() * 100.0),
                    "initial_success_starts": int(initially_successful.sum()),
                    "initial_failure_episodes": denominator,
                    "successes_from_initial_failure_starts": conditional_successes,
                    "initial_failure_success_rate_percent": (
                        float(conditional_successes / denominator * 100.0)
                        if denominator
                        else None
                    ),
                }
            )
    return decomposition


def _read_result(path: Path, *, method: str, task: str, condition: str) -> dict[str, Any]:
    payload = json.loads((path / "result.json").read_text(encoding="utf-8"))
    successes = _successes(payload)
    if payload.get("status") != "ok":
        raise ValueError(f"result is not successful: {path / 'result.json'}")
    if payload.get("training_seed") != SEED or payload.get("evaluation_seed") != EVAL_SEED:
        raise ValueError(f"missing or mismatched seeds in {path / 'result.json'}")
    training_config = payload.get("training_config")
    if not isinstance(training_config, Mapping):
        raise ValueError(f"missing training config metadata in {path / 'result.json'}")
    if training_config.get("seed") != SEED:
        raise ValueError(f"missing or mismatched training config seed in {path / 'result.json'}")
    config_path = Path(str(training_config.get("config_path", "")))
    if not config_path.is_file() or _sha256(config_path) != training_config.get("config_sha256"):
        raise ValueError(f"missing or mismatched training config in {path / 'result.json'}")
    resume_path = training_config.get("resume_metadata_path")
    if resume_path is not None:
        resume_file = Path(str(resume_path))
        if (
            not resume_file.is_file()
            or _sha256(resume_file) != training_config.get("resume_metadata_sha256")
        ):
            raise ValueError(f"missing or mismatched resume metadata in {path / 'result.json'}")
    checkpoint = payload.get("checkpoint")
    if not checkpoint:
        raise ValueError(f"missing checkpoint in {path / 'result.json'}")
    checkpoint_path = Path(str(checkpoint))
    if not checkpoint_path.is_file() or payload.get("epoch") != 10:
        raise ValueError(f"missing epoch 10 checkpoint in {path / 'result.json'}")
    success_vector = [bool(item.get("success", False)) for item in payload["episodes"]]
    if payload.get("episode_success_vector") != success_vector:
        raise ValueError(f"success vector mismatch in {path / 'result.json'}")
    if payload.get("success_count") != sum(success_vector):
        raise ValueError(f"success count mismatch in {path / 'result.json'}")
    expected_rate = float(np.mean(success_vector) * 100.0)
    if not math.isclose(
        float(payload.get("success_rate_percent", float("nan"))),
        expected_rate,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"success rate mismatch in {path / 'result.json'}")
    low, high = wilson_interval(successes)
    return {
        "task": task,
        "method": method,
        "condition": condition,
        "path": str(path / "result.json"),
        "cohort_sha256": payload.get("cohort_sha256") or payload.get("parameters", {}).get("cohort_sha256"),
        "successes": int(sum(successes)),
        "episodes": len(successes),
        "success_rate": float(np.mean(successes)),
        "success_rate_percent": float(np.mean(successes) * 100.0),
        "training_seed": int(payload["training_seed"]),
        "training_config_path": str(config_path),
        "training_config_sha256": str(training_config["config_sha256"]),
        "evaluation_seed": int(payload["evaluation_seed"]),
        "checkpoint": str(checkpoint_path),
        "wilson_95_percent_low": float(low * 100.0),
        "wilson_95_percent_high": float(high * 100.0),
        "success_vector": successes,
    }


def command_analyze(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    training_outputs = _validate_training_outputs(output_root)
    rows: list[dict[str, Any]] = []
    manifests = {
        task: _ensure_manifest(task, output_root, args.cohort_name)
        for task in PHASE3_TASKS
    }
    for task in PHASE3_TASKS:
        for spec in fast_conditions(task):
            target = _condition_dir(output_root, task, spec)
            if not _result_complete(target):
                raise FileNotFoundError(f"missing FastLeWAM result: {target / 'result.json'}")
            _annotate_result_file(
                target, method="fastlewam", task=task, output_root=output_root
            )
            rows.append(_read_result(target, method="fast_lewam", task=task, condition=_condition_name(spec)))
        for method in ("lewm", "leflow"):
            target = _baseline_dir(output_root, method, task)
            if not _result_complete(target):
                raise FileNotFoundError(f"missing baseline result: {target / 'result.json'}")
            _annotate_result_file(
                target, method=method, task=task, output_root=output_root
            )
            rows.append(_read_result(target, method=method, task=task, condition="standard"))

    expected = len(PHASE3_TASKS) * (61 + 2)
    if len(rows) != expected:
        raise ValueError(f"Phase 3 result count {len(rows)} != {expected}")
    expected_hashes = {task: manifests[task].computed_sha256 for task in PHASE3_TASKS}
    for row in rows:
        if row["cohort_sha256"] != expected_hashes[row["task"]]:
            raise ValueError(
                f"missing or mismatched cohort hash in {row['path']}: "
                f"{row['cohort_sha256']} != {expected_hashes[row['task']]}"
            )

    videos: list[dict[str, Any]] = []
    video_conditions = (
        ("fastlewam", "fastlewam", ("p1_step_1", "p3_step_1")),
        ("lewm", "lewm", ("standard",)),
        ("leflow", "leflow", ("standard",)),
    )
    for task in PHASE3_TASKS:
        for video_method, training_method, conditions in video_conditions:
            for condition in conditions:
                target = _video_dir(output_root, video_method, task, condition)
                video_path = target / "videos" / "env_0.mp4"
                result_path = target / "result.json"
                video_info = _validate_video_file(video_path)
                if not result_path.is_file():
                    raise FileNotFoundError(f"missing representative video result: {result_path}")
                video_payload = json.loads(result_path.read_text(encoding="utf-8"))
                video_cohort_hash = _validate_video_result(
                    video_payload, manifest=manifests[task], result_path=result_path
                )
                _annotate_result_file(
                    target,
                    method=training_method,
                    task=task,
                    output_root=output_root,
                )
                videos.append(
                    {
                        "task": task,
                        "method": video_method,
                        "condition": condition,
                        "path": str(video_path),
                        "size_bytes": video_path.stat().st_size,
                        **video_info,
                        "source_cohort_sha256": video_cohort_hash,
                    }
                )

    primary: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for task in PHASE3_TASKS:
        fast = next(row for row in rows if row["task"] == task and row["method"] == "fast_lewam" and row["condition"].startswith("P3/not_applicable/none/step_1"))
        for method in ("lewm", "leflow"):
            baseline = next(row for row in rows if row["task"] == task and row["method"] == method)
            paired = _mcnemar(baseline["success_vector"], fast["success_vector"])
            comparisons.append({"task": task, "baseline": method, "treatment": "fast_lewam_p3_step1_none", "delta_pp": (fast["success_rate"] - baseline["success_rate"]) * 100.0, **paired})
        primary.extend([fast] + [row for row in rows if row["task"] == task and row["method"] in {"lewm", "leflow"}])

    scene_groups = []
    scene_manifest = manifests["scene"]
    scene_dataset = _load_dataset("scene")
    groups = scene_target_group(scene_dataset, scene_manifest)
    for row in rows:
        if row["task"] != "scene":
            continue
        grouped = {}
        for group in sorted(set(groups)):
            values = [value for value, label in zip(row["success_vector"], groups) if label == group]
            grouped[group] = {"episodes": len(values), "successes": int(sum(values)), "success_rate_percent": float(np.mean(values) * 100.0) if values else None}
        scene_groups.append({"method": row["method"], "condition": row["condition"], "groups": grouped})

    fast_rows = {
        (row["task"], row["condition"]): row
        for row in rows
        if row["method"] == "fast_lewam"
    }
    flow_step_sensitivity = []
    guidance_deltas = []
    for task in PHASE3_TASKS:
        protocol = p2_protocol(task)
        for mode in ("P0", "P2", "P3"):
            protocol_name = "not_applicable" if mode in {"P0", "P3"} else protocol
            for step in (1, 2, 5, 10, 16, 32):
                baseline_name = f"{mode}/{protocol_name}/none/step_{step}/euler"
                baseline = fast_rows[(task, baseline_name)]
                flow_step_sensitivity.append(
                    {
                        "task": task,
                        "mode": mode,
                        "guidance": "none",
                        "step": step,
                        "success_rate_percent": baseline["success_rate_percent"],
                        "successes": baseline["successes"],
                    }
                )
                guidance_names = (
                    ("guided_flow", "post_opt")
                    if mode in {"P0", "P2"}
                    else ("guided_flow", "post_opt", "post_opt_refine")
                )
                for guidance in guidance_names:
                    guided_name = f"{mode}/{protocol_name}/{guidance}/step_{step}/euler"
                    guided = fast_rows[(task, guided_name)]
                    guidance_deltas.append(
                        {
                            "task": task,
                            "mode": mode,
                            "guidance": guidance,
                            "step": step,
                            "delta_pp": guided["success_rate_percent"] - baseline["success_rate_percent"],
                        }
                    )

    analysis = {
        "experiment": "Round 5 Phase 3",
        "cohort_name": args.cohort_name,
        "seed": SEED,
        "evaluation_seed": EVAL_SEED,
        "tasks": list(PHASE3_TASKS),
        "expected_result_count": expected,
        "result_count": len(rows),
        "cohort_hashes": expected_hashes,
        "cohort_details": {
            task: {
                "cohort_id": manifests[task].cohort_id,
                "protocol_variant": manifests[task].protocol_variant,
                "cohort_sha256": manifests[task].computed_sha256,
                "selection_rule": dict(manifests[task].sampling_rule),
                "diagnostics": dict(manifests[task].diagnostics),
            }
            for task in PHASE3_TASKS
        },
        "legacy_primary_decomposition": _legacy_primary_decomposition(),
        "training_outputs": training_outputs,
        "videos": videos,
        "primary": [
            {key: value for key, value in row.items() if key != "success_vector"}
            for row in primary
        ],
        "primary_comparisons": comparisons,
        "scene_target_task_groups": scene_groups,
        "flow_step_sensitivity": flow_step_sensitivity,
        "guidance_deltas": guidance_deltas,
        "rows": [{key: value for key, value in row.items() if key != "success_vector"} for row in rows],
        "code_commit": _git_commit(),
    }
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [key for key in analysis["rows"][0] if key not in {"payload"}]
    with (analysis_dir / "conditions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(analysis["rows"])
    default_report_path = (
        INITIAL_FAILURE_REPORT_PATH
        if args.cohort_name == "initial_fail_50"
        else REPORT_PATH
    )
    report_path = (
        Path(args.report_path).resolve()
        if args.report_path is not None
        else default_report_path
    )
    report = _render_report(analysis, report_path, output_root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"status": "ok", "result_count": len(rows), "analysis": str(analysis_dir / "analysis.json"), "report": str(report_path)}, sort_keys=True))


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _render_report(analysis: Mapping[str, Any], report_path: Path, output_root: Path) -> str:
    rows = analysis["rows"]
    by_key = {(row["task"], row["method"], row["condition"]): row for row in rows}
    is_initial_filtered = analysis.get("cohort_name") == "initial_fail_50"
    title = (
        "# Round 5 Phase 3：剔除初始成功样本后的复评"
        if is_initial_filtered
        else "# Round 5 Phase 3：新增任务上的 FastLeWAM 与基线对比"
    )
    intro = (
        "本报告复用 Round 5 Phase 3 已有的训练 checkpoint（训练 seed 3072），在新的、排除初始成功起点的固定 cohort 上复评；评测 seed 为 42。结果是描述性单训练 seed 实验，不构成 DeWM 论文六种子结果的严格复现。"
        if is_initial_filtered
        else "本报告使用单训练 seed 3072、评测 seed 42 和固定 legacy 50-episode cohort。结果是描述性单种子实验，不构成 DeWM 论文六种子结果的严格复现。"
    )
    lines = [title, "", intro, ""]
    if is_initial_filtered:
        lines.extend(
            [
                "## 起点评定、成功标准与抽样",
                "",
                "先以规划使用的同一未来 goal row（start_step + 25）对每个候选起点应用任务成功判据；起点已成功的 start-goal pair 不进入新 cohort。然后排除原 legacy cohort 中的 episode，在其余 episode 中每个任务固定抽取 50 个不同 episode，每个 episode 随机取一个符合条件的起点。所有方法和 FastLeWAM 条件复用同一任务 cohort；筛选不依据任何模型输出。运行成功率的分母为这 50 个起点未成功的 episode。",
                "",
                "| 任务 | 起点排除判据 / episode 成功判据 | 原 legacy cohort 初始成功 | 新 cohort 初始成功 |",
                "|---|---|---:|---:|",
            ]
        )
        criteria = {
            "scene": "cube 位置 L2 ≤ 0.04 m；两个 button 状态均精确匹配；drawer 与 window 位置误差均 ≤ 0.04 m，五项同时满足",
            "finger": "tip 到未来 goal target 的距离 ≤ 0.03 m（与 turn_hard runtime reward ≥ 1 等价）",
            "humanoid": "head height ≥ 1.4、torso upright ≥ 0.9、平面质心速度 ≥ 1.0，三项同时满足",
        }
        for task in PHASE3_TASKS:
            details = analysis["cohort_details"][task]
            diagnostic = details["diagnostics"]
            legacy_initial = diagnostic.get("reference_initial_success_count")
            legacy_count = diagnostic.get("reference_episode_count_excluded")
            selected_initial = diagnostic.get("selected_initial_success_count")
            lines.append(
                f"| {task} | {criteria[task]} | {legacy_initial}/{legacy_count} | "
                f"{selected_initial}/{NUM_EVAL} |"
            )
        lines.extend(
            [
                "",
                "执行阶段仍按原 Phase 3 runtime termination 判定：在目标设定后的最多 50 个控制步内，任意一步达到同一任务成功判据即记为成功。Scene 目标状态和规划图像都取未来 goal row；Finger 使用 DMControl `turn_hard` 奖励；Humanoid 使用上述三阈值 conjunction。DMC 数据集的 NaN `success` 列不参与计算。",
                "",
                "新 cohort 是独立的补充协议；它回答起点本来未成功时的条件成功率，不能与 legacy 的 50-episode 总体成功率混成同一分数。legacy 结果保留在原报告中，仅用于协议对照。",
                "",
                "| task | cohort id | sampling seed | eligible failed start-goal pairs | selected unique episodes | cohort SHA-256 |",
                "|---|---|---:|---:|---:|---|",
            ]
        )
        for task in PHASE3_TASKS:
            details = analysis["cohort_details"][task]
            rule = details["selection_rule"]
            diagnostic = details["diagnostics"]
            lines.append(
                f"| {task} | `{details['cohort_id']}` | {rule['selection_seed']} | "
                f"{diagnostic['fresh_initial_failure_candidate_count']} | {NUM_EVAL} | "
                f"`{details['cohort_sha256']}` |"
            )
        lines.extend(["", "新 cohort 采用 `sampling_revised` manifest；候选 start-goal row 沿用 legacy 有效行规则和全局末候选排除，筛选后按 episode 均匀抽样，每个 episode 内随机选择一个初始失败起点。旧 legacy episode 整体排除，以保证两批 episode 不重叠。", ""])
        lines.extend(
            [
                "## 原 legacy 主结果分解",
                "",
                "下表对原报告的 50 个 legacy episode 使用同一初始成功判据做拆分；总体率是旧评测的原始值，右侧条件率只以起点未成功的 episode 为分母。这是对旧 cohort 的诊断性拆分，不替代下方独立新 cohort 的复评。Finger 的旧 cohort 仅有 8 个起点未成功，条件率不适合用于方法排名。",
                "",
                "| task | method | legacy 总成功 | 起点已成功 | 起点未成功后成功 | 起点未成功条件率 |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for item in analysis["legacy_primary_decomposition"]:
            lines.append(
                f"| {item['task']} | {item['method']} | "
                f"{item['legacy_successes']}/{item['legacy_episodes']} "
                f"({item['legacy_success_rate_percent']:.1f}%) | "
                f"{item['initial_success_starts']}/50 | "
                f"{item['successes_from_initial_failure_starts']}/"
                f"{item['initial_failure_episodes']} | "
                f"{item['initial_failure_success_rate_percent']:.1f}% |"
            )
        lines.append("")
    else:
        lines.extend(
            [
                "本报告使用固定 legacy 50-episode cohort。",
                "",
            ]
        )
    lines.extend([
        "## 主比较",
        "",
        "主比较固定为 FastLeWAM R4-AB 的 P3 / action-flow step 1 / no guidance；LeWM 使用 CEM 300/30/30，LeFlow 使用 64 paths / 16 flow steps。",
        "",
        "| task | FastLeWAM P3/s1/none | LeWM | LeFlow |",
        "|---|---:|---:|---:|",
    ])
    for task in PHASE3_TASKS:
        key = next(key for key in by_key if key[0] == task and key[1] == "fast_lewam" and key[2].startswith("P3/not_applicable/none/step_1"))
        cells = [by_key[key], by_key[(task, "lewm", "standard")], by_key[(task, "leflow", "standard")]]
        lines.append("| {} | {} | {} | {} |".format(task, *[f"{item['successes']}/50 ({item['success_rate_percent']:.1f}%, CI [{item['wilson_95_percent_low']:.1f}, {item['wilson_95_percent_high']:.1f}])" for item in cells]))
    lines.extend(["", "| task | baseline | Δ pp (Fast − baseline) | improved | regressed | McNemar p |", "|---|---|---:|---:|---:|---:|"])
    for item in analysis["primary_comparisons"]:
        lines.append(f"| {item['task']} | {item['baseline']} | {item['delta_pp']:.1f} | {item['improved']} | {item['regressed']} | {item['mcnemar_exact_two_sided_p']:.6f} |")
    method_order = {"fast_lewam": 0, "lewm": 1, "leflow": 2}
    task_order = {task: index for index, task in enumerate(PHASE3_TASKS)}
    matrix_rows = sorted(
        rows,
        key=lambda row: (
            task_order[row["task"]],
            method_order.get(row["method"], len(method_order)),
            row["condition"],
        ),
    )
    lines.extend(
        [
            "",
            "## 完整参数矩阵成功率",
            "",
            "下表逐项列出所有 FastLeWAM 参数条件及 LeWM、LeFlow 标准基线。FastLeWAM 的 condition 字段保留完整配置键（mode / protocol / guidance / step / sampler；不适用项按配置记为 `none` 或 `not_applicable`）；成功率以该任务固定 cohort 的全部 episode 为分母。",
            "",
            "| task | method | condition（完整参数键） | 成功数 / episode 数 | 成功率 |",
            "|---|---|---|---:|---:|",
        ]
    )
    for item in matrix_rows:
        lines.append(
            f"| {item['task']} | {item['method']} | `{item['condition']}` | "
            f"{item['successes']}/{item['episodes']} | "
            f"{item['success_rate_percent']:.1f}% |"
        )
    lines.extend(["", "## Flow-step 敏感性（无 guidance）", "", "| task | mode | step | successes/50 | success rate |", "|---|---|---:|---:|---:|"])
    for item in analysis["flow_step_sensitivity"]:
        lines.append(f"| {item['task']} | {item['mode']} | {item['step']} | {item['successes']}/50 | {item['success_rate_percent']:.1f}% |")
    lines.extend(["", "## Guidance 相对无 guidance 的变化", "", "| task | mode | guidance | step | Δ pp |", "|---|---|---|---:|---:|"])
    for item in analysis["guidance_deltas"]:
        lines.append(f"| {item['task']} | {item['mode']} | {item['guidance']} | {item['step']} | {item['delta_pp']:+.1f} |")
    lines.extend(["", "## 条件矩阵验收", "", f"- 结果数：{analysis['result_count']} / {analysis['expected_result_count']}。", "- 每个结果：50 episodes；成功率来自 runtime termination。", "- 三个任务共享各自固定 cohort；每项结果都必须携带与冻结 manifest 完全一致的 cohort hash。", "- DMC 的 NaN `success` 列未被用作指标。", "", "## Scene 目标行修正", "", "Scene 成功目标与规划使用的未来 goal row 对齐；此前按 episode 级 privileged target 判定的旧结果已归档，不纳入本报告。分组标签仍取该 goal row 的 `privileged_target_task`，只用于分组统计。详见 `docs/report/round5/round5_phase3_scene_goalrow_recheck.md`。"])
    training_configs = {
        (row["task"], row["method"], row["training_config_path"], row["training_config_sha256"])
        for row in rows
    }
    lines.extend(
        [
            "",
            "## 训练配置与种子",
            "",
            "所有结果均记录训练 seed 3072、评测 seed 42，并通过配置文件路径和 SHA-256 固定对应训练配置。",
        ]
    )
    for task, method, config_path, config_hash in sorted(training_configs):
        lines.append(
            f"- {task} / {method}: {config_path}, SHA-256 {config_hash}."
        )

    training_outputs = analysis.get("training_outputs", [])
    if training_outputs:
        lines.extend(
            [
                "",
                "## Epoch 10 checkpoint 验收",
                "",
                f"已核验 {len(training_outputs)} 组训练的 epoch 10 checkpoint；LeFlow 还要求完成标记为 ok 且 epochs=10。",
            ]
        )
        for item in training_outputs:
            lines.append(
                f"- {item['task']} / {item['method']}: {item['checkpoint']}。"
            )

    video_rows = analysis.get("videos", [])
    if video_rows:
        lines.extend(
            [
                "",
                "## 代表视频验收",
                "",
                f"共检查 {len(video_rows)} 段视频；每段对应冻结 cohort 的首个 start–goal 对。",
            ]
        )
        for video in video_rows:
            lines.append(
                f"- {video['task']} / {video['method']} / {video['condition']}: "
                f"{video['path']} ({video['width']}×{video['height']}, "
                f"{video['size_bytes']} bytes)。"
            )

    validation_log_root = output_root / "logs"
    smoke_log = validation_log_root / "phase3_environment_smoke_cpu.log"
    unit_log = validation_log_root / "phase3_unit_tests.log"
    gate_path = validation_log_root / "pre_evaluation_gate.json"
    if gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("status") != "ok":
            raise ValueError(f"invalid pre-evaluation gate marker: {gate_path}")
        lines.extend([
            "",
            "## 环境适配验证与复评时序",
            "",
            "本报告对应的完整复评矩阵是在 Phase 3 单测、三环境 CPU 适配 smoke 和三个任务各一条模型端到端 smoke 全部通过后启动的。更早的首轮批量实验先于这些检查；该历史时序偏差仍如实记录，但本报告所列复评结果来自门槛通过后的新批次。",
            f"- 单测：{gate['unit_tests']['status']}（{gate['unit_tests']['count']} 项），日志 `{gate['unit_tests']['log']}`。",
            f"- 三环境 CPU 适配 smoke：{gate['environment_smoke']['status']}，日志 `{gate['environment_smoke']['log']}`。",
            f"- 模型端到端 smoke：{gate['end_to_end_smoke']['status']}，方法 `{gate['end_to_end_smoke']['method']}` / 条件 `{gate['end_to_end_smoke']['condition']}`，Scene、Finger、Humanoid 各 {gate['end_to_end_smoke']['episodes_per_task']} episode。",
        ])
    elif smoke_log.is_file() or unit_log.is_file():
        lines.extend([
            "",
            "## 环境适配验证时序",
            "",
            "2026-09-25 的 CPU smoke 只创建并重置 Scene、Finger、Humanoid 环境，完成目标设置后执行一次零动作 step；它没有加载 checkpoint 或运行模型规划，因此不等同于计划要求的每任务单 episode 端到端 smoke。适配层边界测试也在批量实验启动后补做。两类检查均为后验验证，计划要求的批量实验前检查门槛未满足。",
        ])
        if smoke_log.is_file():
            lines.append(f"- 三环境 CPU smoke：`{smoke_log.relative_to(output_root)}`。")
        if unit_log.is_file():
            lines.append(f"- Phase 3 单测输出：`{unit_log.relative_to(output_root)}`。")
    resume_records = []
    for task in PHASE3_TASKS:
        resume_path = output_root / "training" / "lewm" / task / "resume_metadata.json"
        if resume_path.is_file():
            resume_records.append(
                (task, resume_path, json.loads(resume_path.read_text(encoding="utf-8")))
            )
    if resume_records:
        lines.extend([
            "",
            "## LeWM 训练续训记录",
            "",
            "以下列出本阶段 LeWM 训练中存在的续训元数据；未记录的字段不作推断。",
        ])
        for task, resume_path, resume in resume_records:
            details = [f"source run `{resume['source_run']}`"]
            initial_epoch = resume.get("initial_epoch")
            completed_epoch = resume.get("completed_epoch", resume.get("epoch"))
            if initial_epoch is not None and completed_epoch is not None:
                details.append(f"epoch {initial_epoch} 接续至 epoch {completed_epoch}")
            elif completed_epoch is not None:
                details.append(f"元数据记录完成至 epoch {completed_epoch}")
            if "seed" in resume:
                details.append(f"seed={resume['seed']}")
            if "optimizer_state_restored" in resume:
                details.append(
                    "optimizer state restored="
                    + str(resume["optimizer_state_restored"]).lower()
                )
            if "resume_method" in resume:
                details.append(f"续训方式 `{resume['resume_method']}`")
            lines.append(
                f"- `{task}`：" + "；".join(details)
                + f"。元数据：`{resume_path.relative_to(output_root)}`。"
            )
    lines.extend(["", "## Scene target_task 分组", ""])
    for group in analysis["scene_target_task_groups"]:
        lines.append(f"- `{group['method']}` / `{group['condition']}`：" + ", ".join(f"{name}={value['successes']}/{value['episodes']} ({value['success_rate_percent']:.1f}%)" for name, value in sorted(group["groups"].items())))
    lines.extend(["", "## 产物", "", f"- 输出根目录：`{output_root}`", f"- 条件明细：`{output_root / 'analysis' / 'conditions.csv'}`", f"- 分析 JSON：`{output_root / 'analysis' / 'analysis.json'}`", f"- 代表视频：`{output_root / 'videos'}`（每任务 FastLeWAM P1/P3 step 1、LeWM、LeFlow 各一段）", f"- 本报告：`{report_path}`", ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--cohort-name", choices=COHORT_NAMES, default="legacy_50")
    parser.add_argument("--report-path")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.set_defaults(function=command_prepare)
    prepare_initial = sub.add_parser("prepare-initial-fail")
    prepare_initial.add_argument("--reference-output-root", default=str(OUTPUT_ROOT))
    prepare_initial.add_argument("--selection-seed", type=int, default=43)
    prepare_initial.set_defaults(function=command_prepare_initial_fail)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--device", default="cuda")
    smoke.add_argument("--gpu", type=_gpu)
    smoke.set_defaults(function=command_smoke)
    evaluate = sub.add_parser("eval-fast")
    evaluate.add_argument("task", choices=PHASE3_TASKS)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--gpu", type=_gpu)
    evaluate.add_argument("--condition-index", type=int, action="append")
    evaluate.set_defaults(function=command_eval_fast)
    video = sub.add_parser("eval-video")
    video.add_argument("task", choices=PHASE3_TASKS)
    video.add_argument("method", choices=("fastlewam", "lewm", "leflow"))
    video.add_argument("--condition", required=True, choices=("p1_step_1", "p3_step_1", "standard"))
    video.add_argument("--checkpoint", required=True)
    video.add_argument("--device", default="cuda")
    video.add_argument("--gpu", type=_gpu)
    video.set_defaults(function=command_eval_video)
    baseline = sub.add_parser("eval-baseline")
    baseline.add_argument("task", choices=PHASE3_TASKS)
    baseline.add_argument("method", choices=("lewm", "leflow"))
    baseline.add_argument("--checkpoint", required=True)
    baseline.add_argument("--device", default="cuda")
    baseline.add_argument("--gpu", type=_gpu)
    baseline.set_defaults(function=command_eval_baseline)
    analyze = sub.add_parser("analyze")
    analyze.set_defaults(function=command_analyze)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
