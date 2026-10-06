#!/usr/bin/env python3
"""Multi-task Round 5 Phase 5 execution/score horizon exploration.

The checkpoint always predicts the original five-by-five action chunk. This
runner varies the number of environment steps executed before replanning and
the number of five-step latent/action blocks used for candidate scoring.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.checkpoint import load_policy_or_model
from source.common.eval import EvaluationIdentity, compose_eval_config
from source.common.phase3_compat import patch_phase3_environments
from source.common.round3_phase1 import CohortManifest, paired_comparison, wilson_interval
from source.common.round3_validation import validate_result_payload
from source.common.round4_eval import run_round4_evaluation, validate_gpu_visibility
from source.common.round4_protocol import resolve_cem_protocol


DEFAULT_CONFIG = ROOT / "config" / "round5" / "phase5_pre_report2.json"
DEFAULT_MIN_FREE_MIB = 3500
BOOTSTRAP_SAMPLES = 10_000
TIMING_WARMUP = 10
TIMING_RUNS = 50
MODES = (
    ("P0", "none"),
    ("P0", "guided_flow"),
    ("P0", "post_opt"),
    ("P1", "none"),
    ("P2", "none"),
    ("P2", "guided_flow"),
    ("P2", "post_opt"),
    ("P3", "none"),
    ("P3", "guided_flow"),
    ("P3", "post_opt"),
)
TASK_ORDER = ("reacher", "cube", "pusht", "tworoom", "scene", "finger", "humanoid")


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"experiment config must be a JSON object: {path}")
    return value


def _load_task_inputs(config: Mapping[str, Any], task: str):
    item = config["tasks"][task]
    checkpoint = _resolve(item["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = _sha256_file(checkpoint)
    if checkpoint_sha != str(item["checkpoint_sha256"]):
        raise ValueError(f"{task} checkpoint SHA256 changed: {checkpoint_sha}")
    manifest_path = _resolve(item["cohort"])
    manifest = CohortManifest.load(manifest_path)
    if manifest.task != task:
        raise ValueError(f"{task} cohort contains task={manifest.task!r}")
    if manifest.cohort_kind != "dev" or manifest.protocol_variant != "legacy":
        raise ValueError(f"{task} requires a legacy/dev cohort")
    evaluation = config["evaluation"]
    if len(manifest.entries) != int(evaluation["num_eval"]):
        raise ValueError(f"{task} cohort has {len(manifest.entries)} entries")
    if manifest.computed_sha256 != str(item["cohort_sha256"]):
        raise ValueError(f"{task} cohort SHA256 changed: {manifest.computed_sha256}")
    if (
        manifest.seed != int(evaluation["seed"])
        or manifest.goal_offset_steps != int(evaluation["goal_offset_steps"])
    ):
        raise ValueError(f"{task} cohort seed/goal offset differs from the frozen protocol")
    return manifest, manifest_path, checkpoint, checkpoint_sha


def _expected_condition(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: spec[key]
        for key in (
            "execute_steps",
            "score_steps_env",
            "score_horizon_blocks",
            "mode",
            "guidance",
            "cem_protocol",
            "action_flow_steps",
            "action_flow_integrator",
        )
    }


def _condition_specs(config: Mapping[str, Any], task: str) -> list[dict[str, Any]]:
    score_map = config["evaluation"]["score_steps_by_execute"]
    result: list[dict[str, Any]] = []
    for execute_steps in config["evaluation"]["execute_steps"]:
        scores = [int(value) for value in score_map[str(execute_steps)]]
        for score_steps in scores:
            for mode, guidance in MODES:
                if score_steps < 25 and mode == "P0" and guidance == "none":
                    continue
                protocol = "cem-clip" if mode in {"P1", "P2"} else "not_applicable"
                result.append(
                    {
                        "execute_steps": int(execute_steps),
                        "score_steps_env": score_steps,
                        "score_horizon_blocks": score_steps // 5,
                        "mode": mode,
                        "guidance": guidance,
                        "cem_protocol": protocol,
                        "action_flow_steps": None if mode == "P1" else int(config["evaluation"]["flow_steps"]),
                        "action_flow_integrator": (
                            "not_applicable" if mode == "P1" else str(config["evaluation"]["integrator"])
                        ),
                    }
                )
    if len(result) != 76:
        raise AssertionError(f"expected 76 unique conditions for {task}, got {len(result)}")
    return result


def _logical_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    score_map = config["evaluation"]["score_steps_by_execute"]
    for execute_steps in config["evaluation"]["execute_steps"]:
        for score_steps in score_map[str(execute_steps)]:
            score_steps = int(score_steps)
            for mode, guidance in MODES:
                alias = score_steps < 25 and mode == "P0" and guidance == "none"
                result.append(
                    {
                        "execute_steps": int(execute_steps),
                        "score_steps_env": score_steps,
                        "storage_score_steps_env": 25 if alias else score_steps,
                        "score_alias": alias,
                        "mode": mode,
                        "guidance": guidance,
                    }
                )
    if len(result) != 80:
        raise AssertionError(f"expected 80 logical conditions, got {len(result)}")
    return result


def _timing_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Unique fixed-input plans across score horizons; execute does not alter one call."""
    result: list[dict[str, Any]] = []
    scores = sorted({int(score) for values in config["evaluation"]["score_steps_by_execute"].values() for score in values})
    for score_steps in scores:
        for mode, guidance in MODES:
            if score_steps < 25 and mode == "P0" and guidance == "none":
                continue
            result.append(
                {
                    "execute_steps": 25,
                    "score_steps_env": score_steps,
                    "score_horizon_blocks": score_steps // 5,
                    "mode": mode,
                    "guidance": guidance,
                    "cem_protocol": "cem-clip" if mode in {"P1", "P2"} else "not_applicable",
                    "action_flow_steps": None if mode == "P1" else int(config["evaluation"]["flow_steps"]),
                    "action_flow_integrator": (
                        "not_applicable" if mode == "P1" else str(config["evaluation"]["integrator"])
                    ),
                }
            )
    if len(result) != 28:
        raise AssertionError(f"expected 28 unique fixed-input timing specs, got {len(result)}")
    return result


def _condition_path(output_root: Path, task: str, spec: Mapping[str, Any]) -> Path:
    guidance = str(spec["guidance"])
    return (
        output_root
        / "conditions"
        / task
        / f"execute_{int(spec['execute_steps'])}"
        / f"score_{int(spec['score_steps_env'])}"
        / str(spec["mode"])
        / guidance
        / "result.json"
    )


def _timing_path(output_root: Path, task: str, spec: Mapping[str, Any], batch: int) -> Path:
    return (
        output_root
        / "analysis"
        / "timing_conditions"
        / task
        / f"score_{int(spec['score_steps_env'])}"
        / f"{spec['mode']}_{spec['guidance']}_batch{int(batch)}.json"
    )


def _gpu(value: str) -> str:
    if not value.isdigit() or int(value) not in range(8):
        raise argparse.ArgumentTypeError("GPU must be a physical ID in 0..7")
    return value


def _gpu_snapshot(gpu: str) -> dict[str, Any]:
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
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [item.strip() for item in completed.stdout.splitlines()[0].split(",")]
    if len(fields) != 6:
        raise RuntimeError(f"invalid GPU{gpu} snapshot: {completed.stdout!r}")
    return {
        "gpu": int(gpu),
        "name": fields[1],
        "memory_total_mib": int(fields[2]),
        "memory_used_mib": int(fields[3]),
        "memory_free_mib": int(fields[4]),
        "utilization_percent": int(fields[5]),
    }


def _gpu_preflight(gpu: str, minimum_free_mib: int) -> dict[str, Any]:
    snapshot = _gpu_snapshot(gpu)
    snapshot["minimum_free_mib"] = int(minimum_free_mib)
    if snapshot["memory_free_mib"] < int(minimum_free_mib):
        raise RuntimeError(
            f"GPU{gpu} has {snapshot['memory_free_mib']} MiB free; need {minimum_free_mib} MiB"
        )
    print(json.dumps({"gpu_preflight": snapshot}, sort_keys=True), flush=True)
    return snapshot


def _code_identity(config_path: Path) -> dict[str, Any]:
    files = (
        config_path,
        ROOT / "scripts" / "round5_phase5_pre_report2.py",
        ROOT / "source" / "common" / "round4_eval.py",
        ROOT / "source" / "common" / "round4_protocol.py",
        ROOT / "source" / "policy" / "round4.py",
        ROOT / "source" / "policy" / "fast_lewam_eval.py",
        ROOT / "source" / "model" / "fast_lewam" / "jepa.py",
        ROOT / "source" / "model" / "fast_lewam" / "round4.py",
    )
    hashes = {str(path.relative_to(ROOT)): _sha256_file(path) for path in files}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "git_commit": commit,
        "file_sha256": hashes,
        "bundle_sha256": _canonical_sha256(hashes),
        "config_sha256": _sha256_file(config_path),
    }


def _compose(task: str, manifest: CohortManifest, device: str, config: Mapping[str, Any]):
    ev = config["evaluation"]
    cfg = compose_eval_config(
        task,
        overrides=[
            f"eval.num_eval={len(manifest.entries)}",
            "output.save_video=false",
            f"solver.device={device}",
            f"solver.num_samples={int(ev['cem_num_samples'])}",
            f"solver.topk={int(ev['cem_topk'])}",
            f"solver.n_steps={int(ev['cem_iterations'])}",
            f"solver.var_scale={float(ev['cem_var_scale'])}",
        ],
    )
    cfg.solver.device = device
    if (
        int(cfg.plan_config.horizon) != int(ev["action_horizon_blocks"])
        or int(cfg.plan_config.action_block) != int(ev["action_block_env_steps"])
    ):
        raise ValueError(f"{task} eval config does not match the frozen 5x5 protocol")
    return cfg


def _validate_result(
    payload: Mapping[str, Any],
    *,
    task: str,
    config: Mapping[str, Any],
    manifest: CohortManifest,
    checkpoint: Path,
    spec: Mapping[str, Any],
    require_metadata: bool,
) -> None:
    if payload.get("status") != "ok" or len(payload.get("episodes", [])) != len(manifest.entries):
        raise ValueError(f"{task} result is incomplete")
    if payload.get("task") != task:
        raise ValueError(f"result task mismatch: {payload.get('task')!r} != {task!r}")
    if str(Path(str(payload.get("checkpoint", ""))).resolve()) != str(checkpoint.resolve()):
        raise ValueError(f"{task} result checkpoint mismatch")
    params = payload.get("parameters", {})
    expected = {
        "seed": int(config["evaluation"]["seed"]),
        "num_eval": int(config["evaluation"]["num_eval"]),
        "goal_offset_steps": int(config["evaluation"]["goal_offset_steps"]),
        "eval_budget": int(config["evaluation"]["eval_budget"]),
        "horizon": int(config["evaluation"]["action_horizon_blocks"]),
        "receding_horizon": int(config["evaluation"]["action_horizon_blocks"]),
        "action_block": int(config["evaluation"]["action_block_env_steps"]),
        "execute_steps": int(spec["execute_steps"]),
        "score_horizon_blocks": int(spec["score_horizon_blocks"]),
        "cohort_id": manifest.cohort_id,
        "cohort_sha256": manifest.computed_sha256,
    }
    for key, value in expected.items():
        if params.get(key) != value:
            raise ValueError(f"{task} result protocol mismatch {key}: {params.get(key)!r} != {value!r}")
    if payload.get("round4_mode") != spec["mode"]:
        raise ValueError(f"{task} result mode mismatch")
    planning = payload.get("round4_planning", {})
    expected_fields = {
        "guidance_mode": spec["guidance"],
        "cem_protocol": spec["cem_protocol"],
        "action_flow_steps": spec["action_flow_steps"],
        "action_flow_integrator": spec["action_flow_integrator"],
    }
    for key, value in expected_fields.items():
        actual = planning.get(key)
        if key == "guidance_mode" and actual is None:
            actual = "none"
        if actual != value:
            raise ValueError(f"{task} result condition mismatch {key}: {actual!r} != {value!r}")
    if spec["mode"] in {"P1", "P2"} and planning.get("action_bound_mode") != "candidate_clip":
        raise ValueError(f"{task} P1/P2 result must use cem-clip")
    trace_enabled = task not in {"scene", "finger", "humanoid"}
    if trace_enabled and payload.get("trace_content_sha256") is None and payload.get("trace_sha256") is None:
        raise ValueError(f"{task} result does not have a trace integrity hash")
    expected_entries = [(str(e.episode_id), int(e.start_step)) for e in manifest.entries]
    actual_entries = [
        (str(e.get("episode_id", e.get("dataset_episode"))), int(e.get("start_step", 0)))
        for e in payload["episodes"]
    ]
    if actual_entries != expected_entries:
        raise ValueError(f"{task} result episode identities/order differ from its cohort")
    if require_metadata:
        metadata = payload.get("round5_phase5_pre_report2")
        if not isinstance(metadata, Mapping) or metadata.get("condition") != _expected_condition(spec):
            raise ValueError(f"{task} result lacks matching experiment metadata")
    validate_result_payload(payload, manifest=manifest, expected_count=len(manifest.entries))


def _run_one(
    *,
    task: str,
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    gpu: str,
    device: str,
    model: Any,
    manifest: CohortManifest,
    checkpoint: Path,
    checkpoint_sha: str,
    code_identity: Mapping[str, Any],
) -> dict[str, Any]:
    result_path = _condition_path(output_root, task, spec)
    target = result_path.parent
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if "round5_phase5_pre_report2" not in payload:
            # The evaluator commits its validated trace/result before experiment metadata.
            _validate_result(
                payload,
                task=task,
                config=config,
                manifest=manifest,
                checkpoint=checkpoint,
                spec=spec,
                require_metadata=False,
            )
            payload["round5_phase5_pre_report2"] = _result_metadata(
                config, code_identity, manifest, checkpoint_sha, spec, gpu
            )
            _atomic_write_json(result_path, payload)
        _validate_result(
            payload,
            task=task,
            config=config,
            manifest=manifest,
            checkpoint=checkpoint,
            spec=spec,
            require_metadata=True,
        )
        print(json.dumps({"status": "resumed", "task": task, "condition": _expected_condition(spec)}), flush=True)
        return payload
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"partially populated condition directory: {target}")

    ev = config["evaluation"]
    guidance = config["guidance"]
    cfg = _compose(task, manifest, device, config)
    identity = EvaluationIdentity(
        entrypoint="round5_phase5_pre_report2",
        policy_kind="round4_shared_dit",
        checkpoint=str(checkpoint.resolve()),
        epoch=10,
        stage=str(spec["mode"]),
    )
    payload = run_round4_evaluation(
        cfg,
        task=task,
        policy_or_model=model,
        mode=str(spec["mode"]),
        identity=identity,
        manifest=manifest,
        output_dir=target,
        trace_output_dir=target / "trace",
        device=device,
        trace=task not in {"scene", "finger", "humanoid"},
        candidate_count=int(ev["best_of_n_candidates"]),
        flow_steps=16,
        action_flow_steps=spec["action_flow_steps"],
        solver_batch_size=int(ev["solver_batch_size"]),
        candidate_batch_size=int(ev["candidate_batch_size"]),
        action_flow_integrator=str(spec["action_flow_integrator"]),
        cem_protocol=str(spec["cem_protocol"]),
        guidance_mode=str(spec["guidance"]),
        guidance_step_size=float(guidance["step_size"]),
        guidance_last_steps=int(guidance["last_steps"]),
        guidance_inner_steps=int(guidance["inner_steps"]),
        guidance_max_rms_offset=float(guidance["max_rms_offset"]),
        proposal_chunk_size=int(guidance["p3_proposal_chunk_size"]),
        allowed_protocol_variants=("legacy",),
        execute_steps=int(spec["execute_steps"]),
        score_horizon_blocks=int(spec["score_horizon_blocks"]),
    )
    if payload.get("status") != "ok":
        raise RuntimeError(f"{task} evaluator returned unexpected status {payload.get('status')!r}")
    payload["round5_phase5_pre_report2"] = _result_metadata(
        config, code_identity, manifest, checkpoint_sha, spec, gpu
    )
    _atomic_write_json(result_path, payload)
    _validate_result(
        payload,
        task=task,
        config=config,
        manifest=manifest,
        checkpoint=checkpoint,
        spec=spec,
        require_metadata=True,
    )
    print(json.dumps({"status": "ok", "task": task, "condition": _expected_condition(spec), "result": str(result_path)}, sort_keys=True), flush=True)
    return payload


def _result_metadata(
    config: Mapping[str, Any],
    code_identity: Mapping[str, Any],
    manifest: CohortManifest,
    checkpoint_sha: str,
    spec: Mapping[str, Any],
    gpu: str,
) -> dict[str, Any]:
    condition = _expected_condition(spec)
    return {
        "experiment": "Round 5 Phase 5 multi-task action-horizon exploration",
        "condition": condition,
        "condition_sha256": _canonical_sha256(condition),
        "cohort_id": manifest.cohort_id,
        "cohort_sha256": manifest.computed_sha256,
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": code_identity["config_sha256"],
        "code_identity": dict(code_identity),
        "gpu": int(gpu),
        "requested_runner_model": config.get("requested_runner_model"),
        "requested_reasoning_effort": config.get("requested_reasoning_effort"),
        "step_trace_enabled": manifest.task not in {"scene", "finger", "humanoid"},
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    summary: dict[str, Any] = {}
    for task in TASK_ORDER:
        manifest, manifest_path, checkpoint, checkpoint_sha = _load_task_inputs(config, task)
        model, resolved = load_policy_or_model(str(checkpoint))
        if resolved is not None and Path(resolved).resolve() != checkpoint.resolve():
            raise ValueError(f"{task} checkpoint loader resolved different weights")
        action_dim = int(getattr(model, "action_dim", -1))
        horizon = int(getattr(model, "action_horizon", -1))
        expected_dim = int(config["tasks"][task]["action_dim"])
        if action_dim != expected_dim or horizon != int(config["evaluation"]["action_horizon_blocks"]):
            raise ValueError(f"{task} checkpoint model shape {action_dim}x{horizon} differs from config")
        cfg = _compose(task, manifest, "cuda:0", config)
        if int(cfg.eval.eval_budget) != int(config["evaluation"]["eval_budget"]):
            raise ValueError(f"{task} evaluation budget differs from frozen protocol")
        diagnostics = dict(manifest.diagnostics)
        summary[task] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "cohort": str(manifest_path),
            "cohort_id": manifest.cohort_id,
            "cohort_sha256": manifest.computed_sha256,
            "episodes": len(manifest.entries),
            "action_dim": action_dim,
            "action_horizon_blocks": horizon,
            "protocol_variant": manifest.protocol_variant,
            "initial_success_count": diagnostics.get("selected_initial_success_count", "not recorded"),
            "unique_conditions": len(_condition_specs(config, task)),
            "logical_conditions": len(_logical_specs(config)),
        }
        del model
    result = {
        "status": "ok",
        "config": str(config_path.resolve()),
        "config_sha256": _sha256_file(config_path),
        "tasks": summary,
        "timing_specs_per_task": len(_timing_specs(config)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run_worker(args: argparse.Namespace) -> None:
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    task = str(args.task)
    if task in {"scene", "finger", "humanoid"}:
        patch_phase3_environments()
    manifest, _, checkpoint, checkpoint_sha = _load_task_inputs(config, task)
    all_specs = _condition_specs(config, task)
    indices = [int(value) for value in args.indices.split(",") if value]
    if len(set(indices)) != len(indices) or any(value < 0 or value >= len(all_specs) for value in indices):
        raise ValueError("worker indices are duplicate or outside the condition list")
    specs = [all_specs[index] for index in indices]
    device = "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    validate_gpu_visibility(device)
    _gpu_preflight(args.gpu, int(args.min_free_mib))
    model, resolved = load_policy_or_model(str(checkpoint))
    if resolved is not None and Path(resolved).resolve() != checkpoint.resolve():
        raise ValueError(f"{task} checkpoint loader resolved different weights")
    if (
        int(getattr(model, "action_horizon", -1)) != int(config["evaluation"]["action_horizon_blocks"])
        or int(getattr(model, "action_dim", -1)) != int(config["tasks"][task]["action_dim"])
    ):
        raise ValueError(f"{task} loaded model has an unexpected action shape")
    output_root = _resolve(args.output_root or config["output_root"])
    identity = _code_identity(config_path)
    for spec in specs:
        _run_one(
            task=task,
            spec=spec,
            config=config,
            config_path=config_path,
            output_root=output_root,
            gpu=args.gpu,
            device=device,
            model=model,
            manifest=manifest,
            checkpoint=checkpoint,
            checkpoint_sha=checkpoint_sha,
            code_identity=identity,
        )


def _timing_manifest(manifest: CohortManifest, batch_size: int) -> CohortManifest:
    if int(batch_size) == 50:
        return manifest
    if int(batch_size) != 1:
        raise ValueError("fixed-input benchmark supports batch sizes 1 and 50")
    entry = manifest.entries[0]
    result = replace(
        manifest,
        cohort_id=f"{manifest.cohort_id}_phase5_timing1",
        cohort_kind="custom",
        entries=(entry,),
        episode_split={**manifest.episode_split, "selected": (entry.episode_id,)},
    )
    return replace(result, cohort_sha256=result.computed_sha256)


def _timing_one(
    *,
    task: str,
    spec: Mapping[str, Any],
    batch_size: int,
    config: Mapping[str, Any],
    output_root: Path,
    gpu: str,
    device: str,
    model: Any,
    manifest: CohortManifest,
    checkpoint: Path,
    checkpoint_sha: str,
    code_identity: Mapping[str, Any],
) -> dict[str, Any]:
    target = _timing_path(output_root, task, spec, batch_size)
    condition = _expected_condition(spec)
    if target.is_file():
        saved = json.loads(target.read_text(encoding="utf-8"))
        if (
            saved.get("condition") != condition
            or saved.get("batch_size") != int(batch_size)
            or saved.get("checkpoint_sha256") != checkpoint_sha
            or saved.get("cohort_sha256") != manifest.computed_sha256
            or saved.get("warmup") != TIMING_WARMUP
            or saved.get("runs") != TIMING_RUNS
        ):
            raise ValueError(f"timing record identity mismatch: {target}")
        return saved

    cfg = _compose(task, manifest, device, config)
    cfg.eval.num_eval = int(batch_size)
    cfg.world.num_envs = int(batch_size)
    timing_manifest = _timing_manifest(manifest, batch_size)
    ev = config["evaluation"]
    guidance = config["guidance"]
    identity = EvaluationIdentity(
        entrypoint="round5_phase5_pre_report2_fixed_input_timing",
        policy_kind="round4_shared_dit",
        checkpoint=str(checkpoint.resolve()),
        epoch=10,
        stage=str(spec["mode"]),
    )

    def capture(policy, *call_args, **call_kwargs):
        info = call_kwargs.get("info_dict")
        if info is None and call_args:
            info = call_args[0]
        if not isinstance(info, Mapping):
            raise TypeError("fixed-input benchmark requires the live policy info mapping")
        environment_batch_size = int(getattr(getattr(policy, "env", None), "num_envs", batch_size))

        def infer_once():
            replay_info = dict(info)
            replay_info["_needs_flush"] = np.ones(environment_batch_size, dtype=bool)
            return policy.get_action(replay_info)

        import torch

        sync = lambda: torch.cuda.synchronize(device) if str(device).startswith("cuda") else None
        for _ in range(TIMING_WARMUP):
            infer_once()
        sync()
        samples = []
        for _ in range(TIMING_RUNS):
            start = time.perf_counter()
            infer_once()
            sync()
            samples.append(time.perf_counter() - start)
        events = list(getattr(policy, "planning_events", ())) [-TIMING_RUNS:]
        if len(events) != TIMING_RUNS:
            raise RuntimeError(f"expected {TIMING_RUNS} planning events, saw {len(events)}")
        if spec["mode"] == "P0":
            stage_a = sum(int(e.get("guidance_stats", {}).get("stage_a_forward_count", e.get("forward_count", 0))) for e in events)
            stage_b = sum(int(e.get("guidance_stats", {}).get("stage_b_forward_count", 0)) for e in events)
        elif spec["mode"] in {"P1", "P2"}:
            stage_a = sum(int(e.get("stage_a_forward_count", 0)) for e in events)
            stage_b = sum(int(e.get("stage_b_forward_count", e.get("forward_count", 0))) for e in events)
        else:
            stage_a = sum(int(e.get("stage_a_forward_count", e.get("proposal_forward_count", 0))) for e in events)
            stage_b = sum(int(e.get("stage_b_forward_count", 0)) + int(e.get("verifier_forward_count", 0)) for e in events)
        backward = sum(int(e.get("guidance_backward_count", e.get("guidance_stats", {}).get("backward_count", 0))) for e in events)
        values = np.asarray(samples, dtype=np.float64)
        return {
            "environment_batch_size": environment_batch_size,
            "samples_seconds": values.tolist(),
            "p50_seconds": float(np.quantile(values, 0.50)),
            "p95_seconds": float(np.quantile(values, 0.95)),
            "mean_seconds": float(values.mean()),
            "throughput_per_second": float(environment_batch_size / values.mean()),
            "stage_a_forwards_per_call": float(stage_a / TIMING_RUNS),
            "stage_b_forwards_per_call": float(stage_b / TIMING_RUNS),
            "guidance_backwards_per_call": float(backward / TIMING_RUNS),
            "timed_planning_events": len(events),
        }

    started_at = time.time()
    result = run_round4_evaluation(
        cfg,
        task=task,
        policy_or_model=model,
        mode=str(spec["mode"]),
        identity=identity,
        manifest=timing_manifest,
        output_dir=output_root / "timing" / "scratch" / task / f"score_{spec['score_steps_env']}" / f"{spec['mode']}_{spec['guidance']}" / f"batch_{batch_size}",
        device=device,
        trace=False,
        candidate_count=int(ev["best_of_n_candidates"]),
        flow_steps=16,
        action_flow_steps=spec["action_flow_steps"],
        solver_batch_size=int(ev["solver_batch_size"]),
        candidate_batch_size=int(ev["candidate_batch_size"]),
        action_flow_integrator=str(spec["action_flow_integrator"]),
        cem_protocol=str(spec["cem_protocol"]),
        guidance_mode=str(spec["guidance"]),
        guidance_step_size=float(guidance["step_size"]),
        guidance_last_steps=int(guidance["last_steps"]),
        guidance_inner_steps=int(guidance["inner_steps"]),
        guidance_max_rms_offset=float(guidance["max_rms_offset"]),
        proposal_chunk_size=int(guidance["p3_proposal_chunk_size"]),
        allowed_protocol_variants=("legacy",),
        execute_steps=25,
        score_horizon_blocks=int(spec["score_horizon_blocks"]),
        timing_capture_callback=capture,
    )
    if result.get("status") != "timing_capture":
        raise RuntimeError(f"{task} timing callback did not capture a call: {result.get('status')!r}")
    captured = dict(result["timing_capture"])
    record = {
        "schema_version": "round5_phase5_pre_report2_timing_v1",
        "task": task,
        "condition": condition,
        "condition_sha256": _canonical_sha256(condition),
        "batch_size": int(batch_size),
        "warmup": TIMING_WARMUP,
        "runs": TIMING_RUNS,
        "source": "fixed live legacy_50 observation; policy inference only; environment stepping excluded",
        "execute_steps_for_timing_call": 25,
        "score_steps_env": int(spec["score_steps_env"]),
        "score_horizon_blocks": int(spec["score_horizon_blocks"]),
        "cohort_id": manifest.cohort_id,
        "cohort_sha256": manifest.computed_sha256,
        "timing_observation_cohort_id": timing_manifest.cohort_id,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "gpu": int(gpu),
        "started_at_unix": started_at,
        "logical_cpu_count": os.cpu_count(),
        "code_identity": dict(code_identity),
        **captured,
    }
    _atomic_write_json(target, record)
    print(json.dumps({"status": "timed", "task": task, "condition": condition, "batch_size": batch_size, "p50_seconds": record["p50_seconds"], "result": str(target)}, sort_keys=True), flush=True)
    return record


def timing_worker(args: argparse.Namespace) -> None:
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    task = str(args.task)
    if task in {"scene", "finger", "humanoid"}:
        patch_phase3_environments()
    manifest, _, checkpoint, checkpoint_sha = _load_task_inputs(config, task)
    all_specs = _timing_specs(config)
    indexes = [int(value) for value in args.indices.split(",") if value]
    if len(set(indexes)) != len(indexes) or any(value < 0 or value >= len(all_specs) for value in indexes):
        raise ValueError("timing indices are duplicate or outside the timing spec list")
    specs = [all_specs[index] for index in indexes]
    device = "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    validate_gpu_visibility(device)
    _gpu_preflight(args.gpu, int(args.min_free_mib))
    model, resolved = load_policy_or_model(str(checkpoint))
    if resolved is not None and Path(resolved).resolve() != checkpoint.resolve():
        raise ValueError(f"{task} checkpoint loader resolved different weights")
    if int(getattr(model, "action_horizon", -1)) != int(config["evaluation"]["action_horizon_blocks"]):
        raise ValueError(f"{task} model action horizon mismatch")
    output_root = _resolve(args.output_root or config["output_root"])
    identity = _code_identity(config_path)
    for spec in specs:
        for batch in config["timing"]["batch_sizes"]:
            _timing_one(
                task=task,
                spec=spec,
                batch_size=int(batch),
                config=config,
                output_root=output_root,
                gpu=args.gpu,
                device=device,
                model=model,
                manifest=manifest,
                checkpoint=checkpoint,
                checkpoint_sha=checkpoint_sha,
                code_identity=identity,
            )


def _run_worker_wave(
    *,
    config_path: Path,
    output_root: Path,
    task: str,
    gpus: Sequence[str],
    indices: Sequence[int],
    timing: bool,
    min_free_mib: int,
) -> None:
    for gpu in gpus:
        _gpu_preflight(gpu, min_free_mib)
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    processes = []
    script = ROOT / "scripts" / "round5_phase5_pre_report2.py"
    for shard, gpu in enumerate(gpus):
        owned = [value for pos, value in enumerate(indices) if pos % len(gpus) == shard]
        if not owned:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        command = [
            sys.executable,
            str(script),
            "timing-worker" if timing else "worker",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--task",
            task,
            "--gpu",
            gpu,
            "--min-free-mib",
            str(min_free_mib),
            "--indices",
            ",".join(map(str, owned)),
        ]
        log = (log_root / f"{'timing_' if timing else ''}{task}_gpu{gpu}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((gpu, process, log))
        print(json.dumps({"worker_started": task, "gpu": int(gpu), "timing": timing, "indices": owned}), flush=True)
    failures = []
    for gpu, process, log in processes:
        code = process.wait()
        log.close()
        print(json.dumps({"worker_finished": task, "gpu": int(gpu), "exit_code": code, "timing": timing}), flush=True)
        if code:
            failures.append((gpu, code))
    if failures:
        raise RuntimeError(f"workers failed for {task}: {failures}; inspect {log_root}")


def run_matrix(args: argparse.Namespace, *, timing: bool = False) -> None:
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    output_root = _resolve(args.output_root or config["output_root"])
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in TASK_ORDER for task in tasks):
        raise ValueError(f"tasks must be distinct items from {TASK_ORDER}")
    gpus = [_gpu(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("--gpus must list distinct physical IDs")
    for task in tasks:
        if timing:
            specs = _timing_specs(config)
            if task == "reacher":
                # The original Phase5 analysis already contains validated
                # full-score timings for all ten policies and both batches.
                indices = [index for index, spec in enumerate(specs) if int(spec["score_steps_env"]) < 25]
            else:
                indices = list(range(len(specs)))
        else:
            specs = _condition_specs(config, task)
            if task == "reacher":
                indices = [
                    index
                    for index, spec in enumerate(specs)
                    if (int(spec["execute_steps"]), int(spec["score_steps_env"])) in {(5, 10), (1, 5)}
                ]
            else:
                indices = list(range(len(specs)))
        if not indices:
            continue
        print(json.dumps({"task_wave": task, "timing": timing, "selected_conditions": len(indices), "gpus": [int(x) for x in gpus]}), flush=True)
        _run_worker_wave(
            config_path=config_path,
            output_root=output_root,
            task=task,
            gpus=gpus,
            indices=indices,
            timing=timing,
            min_free_mib=int(args.min_free_mib),
        )
    print(json.dumps({"status": "timing_matrix_complete" if timing else "evaluation_matrix_complete", "tasks": tasks, "output_root": str(output_root)}), flush=True)


def _reacher_existing(config: Mapping[str, Any], output_root: Path):
    """Validate and load the original 60-cell Reacher Phase5 matrix and timings."""
    import round5_phase5 as old

    old_config_path = ROOT / "config" / "round5" / "phase5.json"
    old_config = old._load_config(old_config_path)
    manifest, _, checkpoint, checkpoint_sha = old._load_inputs(old_config)
    if (
        checkpoint_sha != config["tasks"]["reacher"]["checkpoint_sha256"]
        or manifest.computed_sha256 != config["tasks"]["reacher"]["cohort_sha256"]
    ):
        raise ValueError("existing Reacher Phase5 artifacts do not match report2 inputs")
    analysis_path = _resolve(config["tasks"]["reacher"]["existing_analysis"])
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    by_key: dict[tuple[int, int, str, str], tuple[dict[str, Any], str]] = {}
    base_specs = old._condition_specs()
    for row in analysis["rows"]:
        key = (
            int(row["execute_steps"]),
            int(row["score_steps_applied_env"]),
            str(row["mode"]),
            str(row["guidance"]),
        )
        path = _resolve(row["result_path"])
        if key in by_key:
            if by_key[key][1] != str(path.resolve()):
                raise ValueError(f"Reacher Phase5 alias points to a different result: {key}")
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec = next(
            item
            for item in base_specs
            if (int(item["execute_steps"]), int(item["score_steps_env"]), item["mode"], item["guidance"])
            == key
        )
        old._validate_result(
            payload,
            config=old_config,
            manifest=manifest,
            checkpoint=checkpoint,
            spec=spec,
            require_phase5_metadata=key[0] != 25,
        )
        by_key[key] = (payload, str(path.resolve()))
    if len(by_key) != 58:
        raise ValueError(f"expected 58 validated Reacher Phase5 stored results, found {len(by_key)}")
    return by_key, analysis


def _paired_bootstrap(left: Sequence[Any], right: Sequence[Any], seed: int) -> dict[str, Any]:
    comparison = paired_comparison(left, right)
    a = np.asarray([bool(item["success"]) for item in left], dtype=np.float64)
    b = np.asarray([bool(item["success"]) for item in right], dtype=np.float64)
    if a.shape != b.shape or not a.size:
        raise ValueError("paired bootstrap requires same-sized non-empty episode outcomes")
    delta = b - a
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(delta), size=(BOOTSTRAP_SAMPLES, len(delta)))].mean(axis=1) * 100.0
    comparison["bootstrap_95_ci_pp"] = [float(value) for value in np.quantile(draws, [0.025, 0.975])]
    comparison["bootstrap_samples"] = BOOTSTRAP_SAMPLES
    return comparison


def _result_row(task: str, logical: Mapping[str, Any], payload: Mapping[str, Any], path: str, source: str) -> dict[str, Any]:
    episodes = list(payload["episodes"])
    successes = int(sum(bool(item.get("success", False)) for item in episodes))
    lo, hi = wilson_interval(successes, len(episodes))
    planning = payload.get("round4_planning", {})
    samples = [float(value) for value in (planning.get("planning_samples_seconds") or []) if value is not None]
    p50 = float(np.quantile(samples, 0.50)) if samples else planning.get("planning_median_seconds")
    p95 = float(np.quantile(samples, 0.95)) if samples else planning.get("planning_p95_seconds")
    total = float(sum(samples)) if samples else (float(p50) * int(planning.get("replans") or 0) if p50 is not None else None)
    first_success = [float(item["first_success_step"]) for item in episodes if item.get("first_success_step") is not None]
    return {
        "task": task,
        "execute_steps": int(logical["execute_steps"]),
        "score_steps_env": int(logical["score_steps_env"]),
        "score_steps_applied_env": int(logical["storage_score_steps_env"]),
        "score_horizon_blocks": int(logical["storage_score_steps_env"]) // 5,
        "score_alias": bool(logical["score_alias"]),
        "mode": str(logical["mode"]),
        "guidance": str(logical["guidance"]),
        "n": len(episodes),
        "successes": successes,
        "success_rate_percent": 100.0 * successes / len(episodes),
        "wilson_95_low_percent": 100.0 * lo,
        "wilson_95_high_percent": 100.0 * hi,
        "first_success_step_mean": float(np.mean(first_success)) if first_success else None,
        "first_success_step_median": float(np.median(first_success)) if first_success else None,
        "replans": planning.get("replans"),
        "planning_p50_seconds": p50,
        "planning_p95_seconds": p95,
        "planning_total_seconds": total,
        "evaluation_seconds": payload.get("evaluation_seconds"),
        "episodes_per_second": (len(episodes) / float(payload["evaluation_seconds"]) if float(payload.get("evaluation_seconds") or 0) > 0 else None),
        "forward_count": planning.get("forward_count"),
        "stage_a_forward_count": planning.get("stage_a_forward_count"),
        "stage_b_forward_count": planning.get("stage_b_forward_count"),
        "guidance_backward_count": planning.get("guidance_backward_count"),
        "result_source": source,
        "result_path": path,
        "checkpoint_sha256": payload.get("round5_phase5_pre_report2", {}).get("checkpoint_sha256") or payload.get("round5_phase5", {}).get("checkpoint_sha256"),
        "cohort_sha256": payload.get("parameters", {}).get("cohort_sha256"),
        "trace_sha256": payload.get("trace_content_sha256", payload.get("trace_sha256")),
    }


def _load_timing_records(config: Mapping[str, Any], task: str, output_root: Path, checkpoint_sha: str, cohort_sha: str, reacher_existing: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    specs = _timing_specs(config)
    batches = [int(value) for value in config["timing"]["batch_sizes"]]
    records: list[dict[str, Any]] = []
    existing_by_key: dict[tuple[int, str, str, int], dict[str, Any]] = {}
    if task == "reacher":
        for record in (reacher_existing or {}).get("fixed_input_timing", []):
            condition = record.get("condition", {})
            key = (
                int(condition.get("score_steps_env", 25)),
                str(condition.get("mode")),
                str(condition.get("guidance")),
                int(record.get("batch_size", 0)),
            )
            existing_by_key[key] = dict(record)
    for spec in specs:
        for batch in batches:
            key = (int(spec["score_steps_env"]), str(spec["mode"]), str(spec["guidance"]), batch)
            if task == "reacher" and key in existing_by_key:
                record = existing_by_key[key]
                record["task"] = task
                record["source"] = "reused_existing_round5_phase5_full_horizon_timing"
            else:
                path = _timing_path(output_root, task, spec, batch)
                if not path.is_file():
                    raise FileNotFoundError(f"missing timing record: {path}")
                record = json.loads(path.read_text(encoding="utf-8"))
            if (
                int(record.get("batch_size", -1)) != batch
                or int(record.get("warmup", -1)) != TIMING_WARMUP
                or int(record.get("runs", -1)) != TIMING_RUNS
                or len(record.get("samples_seconds", [])) != TIMING_RUNS
                or int(record.get("timed_planning_events", -1)) != TIMING_RUNS
                or record.get("checkpoint_sha256") != checkpoint_sha
                or record.get("cohort_sha256") != cohort_sha
                or int(record.get("score_steps_env", record.get("condition", {}).get("score_steps_env", -1))) != int(spec["score_steps_env"])
            ):
                raise ValueError(f"invalid timing record for {task}: {key}")
            record["task"] = task
            records.append(record)
    if len(records) != len(specs) * len(batches):
        raise ValueError(f"{task} timing matrix is incomplete")
    return records


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    output_root = _resolve(args.output_root or config["output_root"])
    logical_specs = _logical_specs(config)
    per_task_rows: dict[str, list[dict[str, Any]]] = {}
    comparisons: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    reacher_map, reacher_old_analysis = _reacher_existing(config, output_root)

    for task_index, task in enumerate(TASK_ORDER):
        manifest, _, checkpoint, checkpoint_sha = _load_task_inputs(config, task)
        metadata[task] = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "cohort_id": manifest.cohort_id,
            "cohort_sha256": manifest.computed_sha256,
            "cohort_source": config["tasks"][task]["cohort_source"],
            "protocol_variant": manifest.protocol_variant,
            "initial_success_count": manifest.diagnostics.get("selected_initial_success_count", "not recorded"),
            "initial_success_filtering": manifest.diagnostics.get("selected_initial_success_not_filtered", "not recorded"),
        }
        rows: list[dict[str, Any]] = []
        indexed: dict[tuple[int, int, str, str], dict[str, Any]] = {}
        for logical in logical_specs:
            execute = int(logical["execute_steps"])
            score = int(logical["score_steps_env"])
            storage_score = int(logical["storage_score_steps_env"])
            mode = str(logical["mode"])
            guidance = str(logical["guidance"])
            key = (execute, storage_score, mode, guidance)
            if task == "reacher" and key in reacher_map:
                payload, path = reacher_map[key]
                source = "validated_existing_round5_phase5"
            else:
                spec = next(
                    item
                    for item in _condition_specs(config, task)
                    if (int(item["execute_steps"]), int(item["score_steps_env"]), item["mode"], item["guidance"])
                    == key
                )
                path_obj = _condition_path(output_root, task, spec)
                if not path_obj.is_file():
                    raise FileNotFoundError(f"missing {task} evaluation condition: {path_obj}")
                payload = json.loads(path_obj.read_text(encoding="utf-8"))
                _validate_result(
                    payload,
                    task=task,
                    config=config,
                    manifest=manifest,
                    checkpoint=checkpoint,
                    spec=spec,
                    require_metadata=True,
                )
                path = str(path_obj.resolve())
                source = "new_round5_phase5_pre_report2"
            spec = next(
                item for item in _condition_specs(config, task)
                if (int(item["execute_steps"]), int(item["score_steps_env"]), item["mode"], item["guidance"]) == key
            )
            if source != "validated_existing_round5_phase5":
                # The old Reacher Phase5 matrix has already been validated
                # against its original protocol by _reacher_existing(). Its
                # result payload predates the explicit execute_steps field, so
                # revalidating it with this phase's protocol would reject a
                # valid reused row (None != 25) even though the stored row key
                # and checkpoint/cohort identity were verified above.
                _validate_result(
                    payload,
                    task=task,
                    config=config,
                    manifest=manifest,
                    checkpoint=checkpoint,
                    spec=spec,
                    require_metadata=True,
                )
            row = _result_row(task, logical, payload, path, source)
            rows.append(row)
            indexed[(execute, score, mode, guidance)] = payload
        if len(rows) != 80:
            raise ValueError(f"{task} analysis has {len(rows)} logical rows; expected 80")
        per_task_rows[task] = rows

        for mode, guidance in MODES:
            baseline = indexed[(25, 25, mode, guidance)]["episodes"]
            for execute in (10, 5, 1):
                candidate = indexed[(execute, 25, mode, guidance)]["episodes"]
                comparisons.append(
                    {
                        "task": task,
                        "comparison": "execute_prefix_vs_execute25_at_score25",
                        "mode": mode,
                        "guidance": guidance,
                        "execute_steps": execute,
                        "score_steps_env": 25,
                        **_paired_bootstrap(baseline, candidate, 40_000 + task_index * 100 + execute),
                    }
                )
        for execute, score in ((10, 10), (5, 10), (5, 5), (1, 5)):
            for mode, guidance in MODES:
                if mode == "P0" and guidance == "none":
                    continue
                baseline = indexed[(execute, 25, mode, guidance)]["episodes"]
                candidate = indexed[(execute, score, mode, guidance)]["episodes"]
                comparisons.append(
                    {
                        "task": task,
                        "comparison": "short_score_vs_score25",
                        "mode": mode,
                        "guidance": guidance,
                        "execute_steps": execute,
                        "score_steps_env": score,
                        **_paired_bootstrap(baseline, candidate, 50_000 + task_index * 100 + execute * 10 + score),
                    }
                )

        task_timing = _load_timing_records(
            config,
            task,
            output_root,
            checkpoint_sha,
            manifest.computed_sha256,
            reacher_old_analysis if task == "reacher" else None,
        )
        timings.extend(task_timing)

    analysis = {
        "experiment": "Round 5 Phase 5 multi-task action-horizon exploration",
        "stage": "inference_action_horizon_matrix",
        "config": str(config_path.resolve()),
        "config_sha256": _sha256_file(config_path),
        "output_root": str(output_root.resolve()),
        "requested_runner_model": config.get("requested_runner_model"),
        "requested_reasoning_effort": config.get("requested_reasoning_effort"),
        "task_metadata": metadata,
        "unique_conditions_per_task": 76,
        "logical_cells_per_task": 80,
        "rows": [row for task in TASK_ORDER for row in per_task_rows[task]],
        "comparisons": comparisons,
        "fixed_input_timing": timings,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_unit": "paired start episode within the task's frozen legacy_50 cohort",
        "reacher_existing_source_analysis": str(_resolve(config["tasks"]["reacher"]["existing_analysis"])),
    }
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(analysis_dir / "analysis.json", analysis)
    for filename, values in (("conditions.csv", analysis["rows"]), ("comparisons.csv", comparisons), ("timing.csv", timings)):
        with (analysis_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(values)
    report = _render_report(analysis)
    report_path = _resolve(config["report_output"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"status": "complete", "conditions": len(analysis["rows"]), "comparisons": len(comparisons), "timings": len(timings), "report": str(report_path.resolve()), "analysis": str((analysis_dir / "analysis.json").resolve())}, ensure_ascii=False), flush=True)
    return analysis


def _policy_label(mode: str, guidance: str) -> str:
    short = {"none": "none", "guided_flow": "GF", "post_opt": "PO"}[guidance]
    return f"{mode}+{short}"


def _render_report(analysis: Mapping[str, Any]) -> str:
    rows = analysis["rows"]
    timings = analysis["fixed_input_timing"]
    metadata = analysis["task_metadata"]
    lines = [
        "# Round 5 Phase 5：多任务 action horizon 与推理性能探索",
        "",
        "本实验固定使用各任务已有的 5×5 FastLeWAM checkpoint：模型仍预测 5 个 latent/action block，每个 block 覆盖 5 个环境步，共 25 步。这里测试的是推理期的执行前缀和评分长度，不是重新训练 5×1、1×5 或 1×1 结构。`execute` 是动作送入环境后到下一次重规划的环境步数；`score` 是候选动作由 verifier 评分的环境步数，并按完整 5-step latent/action block 对齐。因此 `execute=1, score=5` 表示每个环境步后重规划、以首个 5 步 block 评分；`execute=5, score=10` 表示执行一个 5 步 chunk、以首两个 block 评分。",
        "",
        "P0/P2/P3 的 flow-based 设置使用 step=2、Euler；各自比较 none、GF（guided-flow）、PO（post-opt），P1 使用 none。P1/P2 使用 cem-clip，CEM 为 300 samples、top-k 30、30 iterations。每个任务 50 个固定 cohort 起点、单 checkpoint。短评分条件的 P0+none 与同一执行长度下的 score=25 完全复用，因为它不运行候选评分；80 个逻辑单元因此对应 76 个独立闭环评测。",
        "",
        f"- 配置：`{analysis['config']}`（SHA256 `{analysis['config_sha256']}`）。",
        f"- 用户请求的 runner 是 `{analysis['requested_runner_model']}/{analysis['requested_reasoning_effort']}`；本会话无法验证实际底层运行模型，因此报告记录为请求值，不将其表述为已验证身份。",
        "- 成功率区间为 95% Wilson；差值使用同一 cohort 起点配对 bootstrap 10,000 次，单位为百分点。单 seed、单 checkpoint、50 个 episode 仅用于探索性判断。",
        "- 固定输入计时在真实 cohort observation 上预热 10 次、同步测量 50 次，报告单次 policy inference 的 p50/p95 和 batch throughput；环境 stepping 不计入。score=25/10/5 的规划评分长度分别对应 5/2/1 个 block。",
        "- 执行步数会改变闭环重规划频率；每个闭环结果同时报告 replans、每次规划延迟和整次 50-episode evaluation 时间。由此可观察更短执行前缀是否以更多规划调用换取响应速度。",
        "",
        "## 数据来源与 cohort",
        "",
        "| task | checkpoint SHA256 | cohort | cohort SHA256 | initial-success 起点 | 初始成功是否过滤 |",
        "|---|---|---|---|---:|---|",
    ]
    for task in TASK_ORDER:
        item = metadata[task]
        count = item["initial_success_count"]
        filtered = item["initial_success_filtering"]
        lines.append(f"| {task} | `{item['checkpoint_sha256'][:12]}…` | `{item['cohort_id']}` | `{item['cohort_sha256'][:12]}…` | {count} | {filtered} |")
    lines += [
        "",
        "Cube 与 Tworoom legacy cohort 分别包含 16 和 4 个初始成功起点，且构建诊断说明未按初始成功状态过滤；这会影响其绝对成功率解释。Scene/Finger/Humanoid 的 cohort 文件没有记录初始成功数量，报告保留为 not recorded。Scene 使用 Phase3 固定的 scene-goal callable。",
        "",
        "## 固定输入推理性能",
        "",
        "每个任务下表按 score 长度展示同一策略的单次推理计时；`batch=1/50` 分别表示单环境和完整 50 环境并行 batch。执行长度不改变单次调用，但会改变每个 episode 的规划调用数量；因此需结合后面的闭环 replans/总时长阅读。Reacher 的 score=25 计时复用既有 Phase5 记录，score=10/5 为本轮补测。",
        "",
    ]
    for task in TASK_ORDER:
        lines += [
            f"### {task}",
            "",
            "| score | policy | batch=1 p50/p95 ms | batch=1 calls/s | batch=50 p50/p95 ms | batch=50 env/s |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        subset = [item for item in timings if item["task"] == task]
        keys = sorted({(int(item.get("score_steps_env", item.get("condition", {}).get("score_steps_env", 25))), item.get("condition", {}).get("mode"), item.get("condition", {}).get("guidance")) for item in subset})
        for score, mode, guidance in keys:
            pair = {int(item["batch_size"]): item for item in subset if int(item.get("score_steps_env", item.get("condition", {}).get("score_steps_env", 25))) == score and item.get("condition", {}).get("mode") == mode and item.get("condition", {}).get("guidance") == guidance}
            one = pair[1]
            many = pair[50]
            lines.append(
                f"| {score} | {_policy_label(str(mode), str(guidance))} | {1000*one['p50_seconds']:.2f}/{1000*one['p95_seconds']:.2f} | {one['throughput_per_second']:.2f} | {1000*many['p50_seconds']:.2f}/{1000*many['p95_seconds']:.2f} | {many['throughput_per_second']:.2f} |"
            )
        lines.append("")

    lines += [
        "## 闭环成功率横向对比",
        "",
        "表格单元为成功数/50（成功率）；短评分的 P0+none 标为 `=`，表示它复用同执行长度下的 score=25 结果。",
        "",
    ]
    columns = [_policy_label(mode, guidance) for mode, guidance in MODES]
    lines.append("| task | execute/score | " + " | ".join(columns) + " |")
    lines.append("|---|---:|" + "---:|" * len(columns))
    for task in TASK_ORDER:
        task_rows = [item for item in rows if item["task"] == task]
        for execute, score in ((25, 25), (10, 25), (10, 10), (5, 25), (5, 10), (5, 5), (1, 25), (1, 5)):
            lookup = {(item["mode"], item["guidance"]): item for item in task_rows if item["execute_steps"] == execute and item["score_steps_env"] == score}
            cells = []
            for mode, guidance in MODES:
                row = lookup[(mode, guidance)]
                cells.append("=" if row["score_alias"] else f"{row['successes']}/{row['n']} ({row['success_rate_percent']:.1f}%)")
            lines.append(f"| {task} | {execute}/{score} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += [
        "## 重规划成本与对照文件",
        "",
        "每个条件的 replans、planning p50/p95、planning total、evaluation wall time 和 episodes/s 均保存在逐条件表中。总规划时间会随执行前缀长度变化；固定输入表则用于隔离评分长度对单次推理的影响。",
        "",
        "配对差值完整结果见 `analysis/comparisons.csv`，逐条件原始汇总见 `analysis/conditions.csv`，固定输入计时见 `analysis/timing.csv`；每个条件的 episode trace 与结果 JSON 位于配置所列 output root。Reacher 的旧 60 个逻辑单元经重新校验后与本轮新增的 `5/10`、`1/5` 组合合并；其它任务为本轮完整 80 个逻辑单元。",
        "",
        "## 解读边界",
        "",
        "本矩阵保持模型输出维度 5×5 不变，以推理中的 score horizon 和 execute/replan cadence 为因素。它能回答缩短评分窗口、缩短动作执行前缀对闭环成功率与推理成本的影响；不能单独证明将训练目标或模型输出张量改成 5×1、1×5 或 1×1 的效果。更短的 execute 让控制器更快获得新观测，但增加规划调用；更短的 score 减少 verifier 评估跨度，作用可能依策略而异。结果需结合 50 episode 的区间、初始 cohort 分布和固定输入延迟一起看。",
        "",
    ]
    return "\n".join(lines)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("validate", help="CPU-validate all frozen inputs and checkpoints")
    _add_common_arguments(p)
    p = commands.add_parser("worker", help="run one task condition shard")
    _add_common_arguments(p)
    p.add_argument("--task", required=True, choices=TASK_ORDER)
    p.add_argument("--gpu", required=True, type=_gpu)
    p.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB)
    p.add_argument("--indices", required=True)
    p = commands.add_parser("timing-worker", help="run one task fixed-input timing shard")
    _add_common_arguments(p)
    p.add_argument("--task", required=True, choices=TASK_ORDER)
    p.add_argument("--gpu", required=True, type=_gpu)
    p.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB)
    p.add_argument("--indices", required=True)
    p = commands.add_parser("run-matrix", help="run the closed-loop condition matrix")
    _add_common_arguments(p)
    p.add_argument("--tasks", default="cube,pusht,tworoom,scene,finger,humanoid,reacher")
    p.add_argument("--gpus", default="4,5,6,7")
    p.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB)
    p = commands.add_parser("timing-matrix", help="run fixed-input inference timing")
    _add_common_arguments(p)
    p.add_argument("--tasks", default="cube,pusht,tworoom,scene,finger,humanoid,reacher")
    p.add_argument("--gpus", default="4,5,6,7")
    p.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB)
    p = commands.add_parser("analyze", help="validate, merge, and render results")
    _add_common_arguments(p)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        validate(args)
    elif args.command == "worker":
        run_worker(args)
    elif args.command == "timing-worker":
        timing_worker(args)
    elif args.command == "run-matrix":
        run_matrix(args, timing=False)
    elif args.command == "timing-matrix":
        run_matrix(args, timing=True)
    elif args.command == "analyze":
        analyze(args)


if __name__ == "__main__":
    main()
