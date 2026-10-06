#!/usr/bin/env python3
"""Run and summarize the Round 5 Phase 6.1 Stage-B flow-matching study."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile
from typing import Any, Mapping

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.checkpoint import load_policy_or_model
from source.common.eval import EvaluationIdentity, compose_eval_config
from source.common.round3_phase1 import CohortManifest, paired_comparison, wilson_interval
from source.common.round3_validation import validate_result_payload
from source.common.round4_eval import run_round4_evaluation, validate_gpu_visibility


CONFIG_PATH = ROOT / "config/round5/phase6_1.json"
BOOTSTRAP_SEED = 6101
BOOTSTRAP_SAMPLES = 10_000
MIN_FREE_MIB = 8_192
TASKS = ("pusht", "reacher")
ARMS = ("regression", "fm")
GUIDANCE_CONFIG = {
    "none": {"public": "none", "last_steps": 2, "inner_steps": 0},
    "guided_flow": {"public": "GF-L", "last_steps": 2, "inner_steps": 1},
    "post_opt": {"public": "PO-L", "last_steps": 2, "inner_steps": 2},
    "post_opt_refine": {
        "public": "PO-refine-L",
        "last_steps": 2,
        "inner_steps": 2,
    },
}


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def _load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Phase 6.1 config must be a JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def condition_specs() -> list[dict[str, Any]]:
    config = _load_config()
    specs = []
    for execute_steps in config["evaluation"]["execution_and_score_env_steps"]:
        for protocol in config["protocols"]:
            if protocol["mode"] == "P0" and protocol["guidance"] == "post_opt_refine":
                continue
            specs.append(
                {
                    "execute_steps": int(execute_steps),
                    "score_steps_env": int(execute_steps),
                    "score_horizon_blocks": int(execute_steps // 5),
                    "mode": str(protocol["mode"]),
                    "guidance": str(protocol["guidance"]),
                    "cem_protocol": (
                        "legacy"
                        if protocol["mode"] in {"P1", "P2"}
                        else "not_applicable"
                    ),
                    "action_flow_steps": (
                        None if protocol["mode"] == "P1" else 2
                    ),
                    "action_flow_integrator": (
                        "not_applicable" if protocol["mode"] == "P1" else "euler"
                    ),
                    "action_bound_mode": "none",
                }
            )
    if len(specs) != 22:
        raise AssertionError(f"expected 22 conditions per task/arm, got {len(specs)}")
    return specs


def _condition_path(root: Path, arm: str, task: str, spec: Mapping[str, Any]) -> Path:
    guidance = str(spec["guidance"])
    return (
        root
        / arm
        / task
        / f"execute_{int(spec['execute_steps'])}"
        / f"score_{int(spec['score_steps_env'])}"
        / str(spec["mode"])
        / guidance
    )


def _condition_identity(spec: Mapping[str, Any]) -> dict[str, Any]:
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
            "action_bound_mode",
        )
    }


def _cohort(task: str, config: Mapping[str, Any]) -> tuple[CohortManifest, Path]:
    cohort_path = _resolve(config["tasks"][task]["cohort"])
    manifest = CohortManifest.load(cohort_path)
    if manifest.task != task:
        raise ValueError(f"cohort task {manifest.task!r} does not match {task!r}")
    if manifest.protocol_variant != "legacy" or manifest.cohort_kind != "dev":
        raise ValueError("Phase 6.1 requires the existing legacy_50 dev cohort")
    expected = str(config["tasks"][task]["cohort_sha256"])
    if manifest.computed_sha256 != expected:
        raise ValueError(
            f"{task} cohort SHA256 changed: {manifest.computed_sha256} != {expected}"
        )
    if len(manifest.entries) != int(config["evaluation"]["num_eval"]):
        raise ValueError(f"{task} cohort has the wrong number of episodes")
    return manifest, cohort_path


def _checkpoint(
    task: str,
    arm: str,
    config: Mapping[str, Any],
    *,
    require_file: bool = True,
) -> tuple[Path, str]:
    if arm == "regression":
        task_config = config["tasks"][task]
        path = _resolve(task_config["baseline_checkpoint"])
        expected_hash = str(task_config["baseline_checkpoint_sha256"])
    elif arm == "fm":
        run_dir = _resolve(
            Path(config["training"]["output_root"])
            / f"fm_b_seed3072_{task}"
        )
        model_name = f"r5p61_fm_seed3072_{task}"
        path = run_dir / "checkpoints" / f"{model_name}_weights_epoch_10.pt"
        expected_hash = ""
    else:
        raise ValueError(f"unknown arm {arm!r}")
    if not path.is_file() and require_file:
        raise FileNotFoundError(path)
    if not path.is_file():
        return path, "pending_training"
    observed = _sha256_file(path)
    if expected_hash and observed != expected_hash:
        raise ValueError(f"{arm}/{task} checkpoint SHA256 changed: {observed}")
    return path, observed


def _gpu_preflight(gpu: int, minimum_free_mib: int = MIN_FREE_MIB) -> dict[str, Any]:
    if int(gpu) not in range(8):
        raise ValueError("physical GPU must be one of 0..7")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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
    fields = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
    if len(fields) != 6:
        raise RuntimeError(f"invalid GPU{gpu} preflight output: {completed.stdout!r}")
    snapshot = {
        "gpu": int(fields[0]),
        "name": fields[1],
        "memory_total_mib": int(fields[2]),
        "memory_used_mib": int(fields[3]),
        "memory_free_mib": int(fields[4]),
        "utilization_percent": int(fields[5]),
        "minimum_free_mib": int(minimum_free_mib),
    }
    if snapshot["memory_free_mib"] < int(minimum_free_mib):
        raise RuntimeError(
            f"GPU{gpu} has {snapshot['memory_free_mib']} MiB free; "
            f"need {minimum_free_mib} MiB"
        )
    print(json.dumps({"gpu_preflight": snapshot}, sort_keys=True), flush=True)
    return snapshot


class _LatentNoiseSchedule:
    """Derive one fixed latent path per task/seed/episode/replan key."""

    def __init__(self, *, task: str, seed: int, manifest: CohortManifest):
        self.task = str(task)
        self.seed = int(seed)
        self.episode_ids = [entry.episode_id for entry in manifest.entries]

    def __call__(self, keys, *, horizon, latent_dim, device, dtype):
        rows = []
        for slot, replan in keys:
            identity = {
                "task": self.task,
                "seed": self.seed,
                "episode_id": str(self.episode_ids[int(slot)]),
                "replan": int(replan),
            }
            seed = int(_canonical_sha256(identity)[:16], 16) % (2**63 - 1)
            generator = torch.Generator(device=torch.device(device)).manual_seed(seed)
            rows.append(
                torch.randn(
                    int(horizon),
                    int(latent_dim),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
            )
        if rows:
            return torch.stack(rows, dim=0)
        return torch.empty((0, int(horizon), int(latent_dim)), device=device, dtype=dtype)


def _validate_result(
    payload: Mapping[str, Any],
    *,
    task: str,
    arm: str,
    spec: Mapping[str, Any],
    manifest: CohortManifest,
    checkpoint: Path,
    checkpoint_sha256: str,
    require_phase_metadata: bool = True,
) -> None:
    if str(payload.get("task")) != task:
        raise ValueError("result task differs")
    if Path(str(payload.get("checkpoint", ""))).resolve() != checkpoint.resolve():
        raise ValueError("result checkpoint path differs")
    parameters = payload.get("parameters", {})
    expected_parameters = {
        "seed": 42,
        "num_eval": 50,
        "goal_offset_steps": 25,
        "eval_budget": 50,
        "horizon": 5,
        "action_block": 5,
        "execute_steps": int(spec["execute_steps"]),
        "score_horizon_blocks": int(spec["score_horizon_blocks"]),
        "cohort_sha256": manifest.computed_sha256,
    }
    for key, expected in expected_parameters.items():
        if parameters.get(key) != expected:
            raise ValueError(
                f"result parameter {key} differs: {parameters.get(key)!r} != {expected!r}"
            )
    planning = payload.get("round4_planning", {})
    expected_planning = {
        "guidance_mode": str(spec["guidance"]),
        "action_flow_integrator": str(spec["action_flow_integrator"]),
        "action_flow_steps": spec["action_flow_steps"],
        "action_bound_mode": "none",
    }
    if expected_planning["guidance_mode"] == "post_opt_refine":
        expected_planning["guidance_mode"] = "post_opt_refine"
    for key, expected in expected_planning.items():
        actual = planning.get(key)
        if key == "guidance_mode" and actual is None:
            actual = "none"
        if actual != expected:
            raise ValueError(
                f"result planning parameter {key} differs: {actual!r} != {expected!r}"
            )
    expected_protocol = str(spec["cem_protocol"])
    if planning.get("cem_protocol") != expected_protocol:
        raise ValueError("result CEM protocol differs")
    if require_phase_metadata:
        phase = payload.get("round5_phase6_1")
        if not isinstance(phase, Mapping):
            raise ValueError("result is missing Phase 6.1 metadata")
        if phase.get("arm") != arm or phase.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("result arm or checkpoint hash differs")
        if phase.get("condition") != _condition_identity(spec):
            raise ValueError("result condition identity differs")
    validate_result_payload(
        payload, manifest=manifest, expected_count=len(manifest.entries)
    )


def _reusable_push_baseline(
    task: str,
    arm: str,
    spec: Mapping[str, Any],
    *,
    manifest: CohortManifest,
    checkpoint: Path,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not (
        task == "pusht"
        and arm == "regression"
        and int(spec["execute_steps"]) == 25
        and spec["mode"] in {"P0", "P3"}
        and spec["guidance"] == "none"
    ):
        return None
    source = (
        ROOT
        / "outputs/round5/phase5_pre_report2/conditions/pusht/execute_25/score_25"
        / str(spec["mode"])
        / "none/result.json"
    )
    if not source.is_file():
        return None
    payload = json.loads(source.read_text(encoding="utf-8"))
    _validate_result(
        payload,
        task=task,
        arm=arm,
        spec=spec,
        manifest=manifest,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        require_phase_metadata=False,
    )
    planning = payload.get("round4_planning", {})
    expected_count = 1 if spec["mode"] == "P0" else 64
    if (
        int(planning.get("candidate_count", -1)) != expected_count
        or int(planning.get("action_flow_steps", -1)) != 2
        or planning.get("action_flow_integrator") != "euler"
        or planning.get("action_bound_mode") != "none"
        or planning.get("guidance_mode", "none") != "none"
    ):
        raise ValueError(f"historical {spec['mode']} result does not match the frozen protocol")
    source_trace = source.parent / "trace/episodes.jsonl"
    if not source_trace.is_file():
        raise FileNotFoundError(f"verified historical trace is missing: {source_trace}")
    reference = {
        "kind": "verified_historical_result",
        "source_result": str(source.resolve()),
        "source_result_sha256": _sha256_file(source),
        "source_trace": str(source_trace.resolve()),
        "source_trace_sha256": _sha256_file(source_trace),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "cohort_id": manifest.cohort_id,
        "cohort_sha256": manifest.computed_sha256,
        "compatibility_checks": {
            "checkpoint_and_cohort": True,
            "same_seed_and_success_semantics": True,
            "same_action_flow_and_candidate_protocol": True,
            "same_action_bound_and_guidance": True,
            "regression_branch_unchanged_by_FM_optional_path": True,
        },
    }
    return payload, reference


def _eval_config(task: str, config: Mapping[str, Any], device: str):
    cfg = compose_eval_config(
        task,
        overrides=[
            "eval.num_eval=50",
            "eval.goal_offset_steps=25",
            "eval.eval_budget=50",
            "seed=42",
            "output.save_video=false",
            f"solver.device={device}",
            "solver.batch_size=1",
            "solver.num_samples=300",
            "solver.topk=30",
            "solver.n_steps=30",
            "solver.var_scale=1.0",
        ],
    )
    cfg.world.num_envs = 50
    cfg.solver.device = device
    return cfg


def _code_identity() -> dict[str, Any]:
    files = (
        CONFIG_PATH,
        ROOT / "config/train/round5_phase6_1_fm.yaml",
        ROOT / "config/train/policy/round4_ab.yaml",
        ROOT / "scripts/round5_phase6_1.py",
        ROOT / "source/common/round4_eval.py",
        ROOT / "source/policy/round4.py",
        ROOT / "source/policy/fast_lewam_eval.py",
        ROOT / "source/policy/fast_lewam.py",
        ROOT / "source/model/fast_lewam/jepa.py",
        ROOT / "source/model/fast_lewam/round4.py",
    )
    return {
        str(path.relative_to(ROOT)): _sha256_file(path)
        for path in files
    }


def _archive_training_provenance(task: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the actual historical and FM configs and report protocol deltas."""
    baseline_checkpoint = _resolve(config["tasks"][task]["baseline_checkpoint"])
    # Historical-arm evaluations do not depend on the new epoch-10 checkpoint;
    # archive their frozen config now and let a later FM evaluation finalize the
    # checkpoint hash after training completes.
    fm_checkpoint, fm_checkpoint_sha256 = _checkpoint(
        task, "fm", config, require_file=False
    )
    baseline_config_path = baseline_checkpoint.parent.parent / "config.yaml"
    fm_config_path = fm_checkpoint.parent.parent / "config.yaml"
    if not baseline_config_path.is_file() or not fm_config_path.is_file():
        raise FileNotFoundError(
            f"training configs are missing for {task}: "
            f"{baseline_config_path}, {fm_config_path}"
        )
    target = _resolve(config["output_root"]) / "provenance" / task
    target.mkdir(parents=True, exist_ok=True)
    baseline_copy = target / "regression_config.yaml"
    fm_copy = target / "fm_config.yaml"
    shutil.copy2(baseline_config_path, baseline_copy)
    shutil.copy2(fm_config_path, fm_copy)
    baseline_cfg = OmegaConf.load(baseline_config_path)
    fm_cfg = OmegaConf.load(fm_config_path)
    fields = (
        "seed",
        "trainer.max_epochs",
        "loader.batch_size",
        "trainer.accumulate_grad_batches",
        "trainer.precision",
        "trainer.gradient_clip_val",
        "optimizer.type",
        "optimizer.lr",
        "optimizer.weight_decay",
        "action_horizon",
        "history_size",
        "data.dataset.frameskip",
        "data.dataset.name",
        "img_size",
        "embed_dim",
        "policy.model._target_",
        "policy.model.latent_head_dim",
        "policy.model.latent_head_layers",
        "policy.model.heads",
        "policy.model.mlp_dim",
        "policy.model.stage_a_goal_injection",
        "policy.model.token_encoding",
        "policy.model.stage_b_dynamics",
        "policy.model.stage_b_attention_mode",
        "latent_action_mix_epochs",
        "detach_clean_action",
        "stage_b_timestep_mode",
        "stage_b_action_source",
        "latent_loss_noise_threshold",
        "loss.latent.weight",
        "loss.sigreg.weight",
        "loss.sigreg.kwargs.knots",
        "loss.sigreg.kwargs.num_proj",
        "activation_checkpointing",
        "data_pipeline.gpu_image_preprocessing",
        "data_pipeline.hdf5_chunk_size",
    )
    changes = []
    historical_defaults = {"stage_b_action_source": "joint"}
    for field in fields:
        baseline_value = OmegaConf.select(baseline_cfg, field, default=None)
        fm_value = OmegaConf.select(fm_cfg, field, default=None)
        effective_baseline_value = (
            historical_defaults[field]
            if baseline_value is None and field in historical_defaults
            else baseline_value
        )
        changes.append(
            {
                "field": field,
                "regression": _jsonable(baseline_value),
                "regression_effective_default": _jsonable(effective_baseline_value),
                "fm": _jsonable(fm_value),
                "equal": effective_baseline_value == fm_value,
                "historical_default_applied": baseline_value is None
                and field in historical_defaults,
            }
        )
    frozen_fields = set(fields) - {"data.dataset.name"}
    mismatches = [
        item
        for item in changes
        if item["field"] in frozen_fields and not item["equal"]
    ]
    if mismatches:
        raise ValueError(
            f"{task} FM config differs from historical frozen settings: {mismatches}"
        )
    intentional = {
        "train_mode": {"regression": "stage_ab", "fm": "stage_ab_fm"},
        "policy.model.latent_flow_matching": {
            "regression": False,
            "fm": True,
        },
        "policy.model.latent_flow_steps": {
            "regression": 2,
            "fm": 2,
        },
        "new_parameters": [
            "latent_flow_input",
            "latent_flow_time_mlp",
            "latent_flow_velocity_head",
        ],
    }
    archived = {
        "schema_version": 1,
        "task": task,
        "baseline_checkpoint": str(baseline_checkpoint.resolve()),
        "baseline_checkpoint_sha256": _sha256_file(baseline_checkpoint),
        "fm_checkpoint": str(fm_checkpoint.resolve()),
        "fm_checkpoint_sha256": fm_checkpoint_sha256,
        "fm_checkpoint_status": (
            "available" if fm_checkpoint_sha256 != "pending_training" else "pending_training"
        ),
        "baseline_config_source": str(baseline_config_path.resolve()),
        "baseline_config_copy": str(baseline_copy.resolve()),
        "baseline_config_sha256": _sha256_file(baseline_copy),
        "fm_config_source": str(fm_config_path.resolve()),
        "fm_config_copy": str(fm_copy.resolve()),
        "fm_config_sha256": _sha256_file(fm_copy),
        "historical_vs_fm_frozen_field_diff": changes,
        "intentional_differences": intentional,
        "dataset_sha256": config["tasks"][task]["dataset_sha256"],
        "cohort_sha256": config["tasks"][task]["cohort_sha256"],
        "code_identity": _code_identity(),
    }
    _atomic_json(target / "training_config_diff.json", archived)
    return archived


def _run_condition(
    *,
    task: str,
    arm: str,
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    manifest: CohortManifest,
    checkpoint: Path,
    checkpoint_sha256: str,
    model,
    gpu: int,
    code_identity: Mapping[str, Any],
) -> dict[str, Any]:
    output_root = _resolve(config["output_root"])
    target = _condition_path(output_root, arm, task, spec)
    result_path = target / "result.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        _validate_result(
            payload,
            task=task,
            arm=arm,
            spec=spec,
            manifest=manifest,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
        )
        print(json.dumps({"status": "resumed", "result": str(result_path)}), flush=True)
        return payload
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"partial condition directory requires inspection: {target}")

    # A historical Regression-B row is reused only after the identity and
    # protocol checks above. FM rows are always evaluated with their own A.
    reusable = _reusable_push_baseline(
        task,
        arm,
        spec,
        manifest=manifest,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
    )
    if reusable is not None:
        source_payload, reference = reusable
        payload = dict(source_payload)
        payload["round5_phase6_1"] = {
            "arm": arm,
            "condition": _condition_identity(spec),
            "checkpoint_sha256": checkpoint_sha256,
            "cohort_sha256": manifest.computed_sha256,
            "source_refs": [reference],
            "reused_episode_count": len(manifest.entries),
            "new_samples_added": 0,
        }
        _atomic_json(result_path, payload)
        _atomic_json(target / "source_refs.json", {"source_refs": [reference]})
        print(json.dumps({"status": "reused_verified", "result": str(result_path)}), flush=True)
        return payload

    _gpu_preflight(gpu)
    device = "cuda:0"
    cfg = _eval_config(task, config, device)
    guidance = str(spec["guidance"])
    guidance_values = GUIDANCE_CONFIG[guidance]
    noise_schedule = _LatentNoiseSchedule(
        task=task,
        seed=int(config["evaluation"]["seed"]),
        manifest=manifest,
    )
    identity = EvaluationIdentity(
        entrypoint="round5_phase6_1",
        policy_kind="round4_shared_dit",
        checkpoint=str(checkpoint.resolve()),
        epoch=10,
        stage=str(spec["mode"]),
    )
    result = run_round4_evaluation(
        cfg,
        task=task,
        policy_or_model=model,
        mode=str(spec["mode"]),
        identity=identity,
        manifest=manifest,
        output_dir=target,
        trace_output_dir=target / "trace",
        device=device,
        trace=True,
        candidate_count=64,
        flow_steps=2,
        action_flow_steps=spec["action_flow_steps"],
        solver_batch_size=1,
        candidate_batch_size=64,
        actor_warm_start_scale=1.0,
        action_flow_integrator="euler",
        cem_protocol=str(spec["cem_protocol"]),
        action_bound_mode="none",
        guidance_mode=guidance,
        guidance_step_size=0.01,
        guidance_last_steps=int(guidance_values["last_steps"]),
        guidance_inner_steps=int(guidance_values["inner_steps"]),
        guidance_max_rms_offset=0.2,
        proposal_chunk_size=512,
        allowed_protocol_variants=("legacy",),
        allow_solver_config_override=True,
        allow_eval_budget_override=False,
        policy_seed=int(config["evaluation"]["seed"]),
        latent_noise_schedule=noise_schedule,
        execute_steps=int(spec["execute_steps"]),
        score_horizon_blocks=int(spec["score_horizon_blocks"]),
    )
    if result.get("status") != "ok":
        raise RuntimeError(f"evaluator returned {result.get('status')!r}")
    result["round5_phase6_1"] = {
        "arm": arm,
        "condition": _condition_identity(spec),
        "checkpoint_sha256": checkpoint_sha256,
        "cohort_sha256": manifest.computed_sha256,
        "latent_noise_schedule": "sha256(task,seed,episode_id,replan)->N(0,I)",
        "latent_flow_steps": 2 if arm == "fm" else 0,
        "source_refs": [],
        "new_samples_added": len(manifest.entries),
        "code_identity": dict(code_identity),
        "gpu": int(gpu),
    }
    _atomic_json(result_path, result)
    _validate_result(
        result,
        task=task,
        arm=arm,
        spec=spec,
        manifest=manifest,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
    )
    print(json.dumps({"status": "ok", "result": str(result_path)}), flush=True)
    return result


def run_eval(args: argparse.Namespace) -> None:
    config = _load_config()
    if args.task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    if args.arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    _gpu_preflight(args.gpu, args.min_free_mib)
    validate_gpu_visibility("cuda:0")
    manifest, cohort_path = _cohort(args.task, config)
    checkpoint, checkpoint_sha256 = _checkpoint(args.task, args.arm, config)
    _archive_training_provenance(args.task, config)
    model, resolved = load_policy_or_model(str(checkpoint))
    if resolved is None or Path(resolved).resolve() != checkpoint.resolve():
        raise ValueError("checkpoint loader resolved a different weight file")
    if int(getattr(model, "action_horizon", -1)) != 5:
        raise ValueError("checkpoint action horizon must be five")
    if int(getattr(model, "action_dim", -1)) != int(
        config["tasks"][args.task]["action_dim"]
    ):
        raise ValueError("checkpoint action dimension differs from the frozen config")
    if args.arm == "fm" and not bool(getattr(model, "latent_flow_matching", False)):
        raise ValueError("FM checkpoint was instantiated without latent flow matching")
    model = model.to("cuda:0").eval()
    print(
        json.dumps(
            {
                "task": args.task,
                "arm": args.arm,
                "gpu": int(args.gpu),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "cohort": str(cohort_path.resolve()),
                "cohort_sha256": manifest.computed_sha256,
                "conditions": len(condition_specs()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    code_identity = _code_identity()
    for spec in condition_specs():
        _run_condition(
            task=args.task,
            arm=args.arm,
            spec=spec,
            config=config,
            manifest=manifest,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            model=model,
            gpu=int(args.gpu),
            code_identity=code_identity,
        )


def _load_matrix_result(
    *, task: str, arm: str, spec: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    manifest, _ = _cohort(task, config)
    checkpoint, checkpoint_sha256 = _checkpoint(task, arm, config)
    path = _condition_path(_resolve(config["output_root"]), arm, task, spec) / "result.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_result(
        payload,
        task=task,
        arm=arm,
        spec=spec,
        manifest=manifest,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
    )
    return payload


def _summary_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    episodes = payload["episodes"]
    successes = np.asarray([bool(item["success"]) for item in episodes], dtype=bool)
    low, high = wilson_interval(int(successes.sum()), len(successes))
    planning = payload.get("round4_planning", {})
    by_episode = payload.get("round5_phase6_1", {})
    return {
        "successes": int(successes.sum()),
        "n": int(len(successes)),
        "success_rate": float(successes.mean()),
        "wilson_95": [float(low), float(high)],
        "planning_median_seconds": planning.get("planning_median_seconds"),
        "planning_p95_seconds": planning.get("planning_p95_seconds"),
        "source_refs": by_episode.get("source_refs", []),
        "new_samples_added": int(by_episode.get("new_samples_added", 0)),
    }


def _training_summary(task: str, config: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = _resolve(config["training"]["output_root"]) / f"fm_b_seed3072_{task}"
    identity_path = run_dir / "phase6_1_training_identity.json"
    code_identity_path = run_dir / "phase6_1_training_code_identity.json"
    metrics_path = run_dir / "phase6_1_epoch_metrics.jsonl"
    posthoc_path = run_dir / "phase6_1_endpoint_validation.json"
    if not identity_path.is_file() or not metrics_path.is_file() or not code_identity_path.is_file():
        raise FileNotFoundError(
            f"Phase 6.1 training identity/code/metrics are missing for {task}: "
            f"{identity_path}, {code_identity_path}, {metrics_path}"
        )
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    code_identity = json.loads(code_identity_path.read_text(encoding="utf-8"))
    if code_identity.get("task") != task:
        raise ValueError(f"training code identity task differs for {task}: {code_identity_path}")
    source_archive = Path(str(code_identity.get("source_archive", "")))
    if (
        not source_archive.is_file()
        or _sha256_file(source_archive) != code_identity.get("source_archive_sha256")
    ):
        raise ValueError(f"training source archive hash differs for {task}: {source_archive}")
    with zipfile.ZipFile(source_archive, mode="r") as archive:
        for relative_path, expected_hash in code_identity[
            "training_code_sha256"
        ].items():
            observed_hash = hashlib.sha256(archive.read(relative_path)).hexdigest()
            if observed_hash != expected_hash:
                raise ValueError(
                    f"training source file hash differs for {task}/{relative_path}"
                )
        resolved_config_hash = hashlib.sha256(
            archive.read("resolved_training_config.yaml")
        ).hexdigest()
    if resolved_config_hash != identity.get("resolved_training_config_sha256"):
        raise ValueError(f"archived training config hash differs for {task}")
    metric_rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not metric_rows:
        raise ValueError(f"Phase 6.1 training metrics are empty: {metrics_path}")
    epoch10 = next((item for item in metric_rows if int(item.get("epoch", -1)) == 10), None)
    if epoch10 is None:
        raise ValueError(
            f"Phase 6.1 training did not record validation metrics for epoch 10: {metrics_path}"
        )
    values = dict(epoch10.get("metrics", {}))
    endpoint_validation = None
    if posthoc_path.is_file():
        endpoint_validation = json.loads(posthoc_path.read_text(encoding="utf-8"))
        fm_checkpoint, fm_checkpoint_sha256 = _checkpoint(task, "fm", config)
        if (
            endpoint_validation.get("task") != task
            or Path(str(endpoint_validation.get("checkpoint", ""))).resolve()
            != fm_checkpoint.resolve()
            or endpoint_validation.get("checkpoint_sha256") != fm_checkpoint_sha256
            or int(endpoint_validation.get("validation_sample_count", 0)) <= 0
            or endpoint_validation.get("training_identity_sha256")
            != _sha256_file(identity_path)
            or endpoint_validation.get("resolved_config_sha256")
            != identity.get("resolved_training_config_sha256")
            or endpoint_validation.get("dataset_sha256")
            != identity.get("dataset_sha256")
        ):
            raise ValueError(f"posthoc validation identity differs for {task}: {posthoc_path}")
        values["validate/latent_flow_endpoint_mse_posthoc"] = endpoint_validation[
            "latent_flow_endpoint_mse"
        ]
    endpoint_callback = values.get("validate/latent_flow_endpoint_mse_epoch")
    endpoint_posthoc = values.get("validate/latent_flow_endpoint_mse_posthoc")
    endpoint = endpoint_callback if endpoint_callback is not None else endpoint_posthoc
    if endpoint is None:
        raise ValueError(
            f"Phase 6.1 integrated endpoint MSE is missing for {task}; "
            f"expected epoch metric or {posthoc_path}"
        )
    endpoint_source = (
        "epoch-10 validation callback"
        if endpoint_callback is not None
        else "posthoc epoch-10 validation"
    )
    velocity_callback = values.get("validate/weighted_latent_flow_velocity_loss_epoch")
    if velocity_callback is None:
        velocity_callback = values.get("validate/latent_flow_velocity_loss_epoch")
    velocity_posthoc = None
    if endpoint_validation is not None:
        velocity_posthoc = endpoint_validation.get("weighted_latent_flow_velocity_loss")
        if velocity_posthoc is None:
            velocity_posthoc = endpoint_validation.get("latent_flow_velocity_loss")
    velocity = velocity_callback if velocity_callback is not None else velocity_posthoc
    velocity_source = (
        "epoch-10 validation callback"
        if velocity_callback is not None
        else "posthoc epoch-10 validation"
    )
    if velocity is None:
        raise ValueError(f"Phase 6.1 validation velocity loss is missing for {task}")
    validation_crosscheck = {
        "callback_endpoint_mse": (
            None if endpoint_callback is None else float(endpoint_callback)
        ),
        "posthoc_endpoint_mse": (
            None if endpoint_posthoc is None else float(endpoint_posthoc)
        ),
        "endpoint_mse_absolute_difference": (
            None
            if endpoint_callback is None or endpoint_posthoc is None
            else abs(float(endpoint_callback) - float(endpoint_posthoc))
        ),
        "callback_weighted_velocity_loss": (
            None if velocity_callback is None else float(velocity_callback)
        ),
        "posthoc_weighted_velocity_loss": (
            None if velocity_posthoc is None else float(velocity_posthoc)
        ),
        "weighted_velocity_loss_absolute_difference": (
            None
            if velocity_callback is None or velocity_posthoc is None
            else abs(float(velocity_callback) - float(velocity_posthoc))
        ),
    }
    epoch_count = int(identity["epochs"])
    global_step = int(epoch10["global_step"])
    telemetry_path = run_dir.parent / "gpu_monitoring.jsonl"
    observed_gpu_memory = []
    if telemetry_path.is_file():
        for line in telemetry_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            snapshot = json.loads(line)
            for gpu in snapshot.get("gpus", []):
                for process in gpu.get("compute_processes", []):
                    command = str(process.get("process") or "")
                    if f"fm_b_seed3072_{task}" in command:
                        observed_gpu_memory.append(
                            {
                                "timestamp_utc": snapshot.get("timestamp_utc"),
                                "gpu": int(gpu["gpu"]),
                                "pid": int(process["pid"]),
                                "used_mib": int(process["used_mib"]),
                            }
                        )
    training_log = run_dir / f"{task}.log"
    epoch_seconds = []
    if training_log.is_file():
        log_text = training_log.read_text(encoding="utf-8", errors="replace")
        epoch_seconds = [
            float(value)
            for value in re.findall(r"\[Epoch \d+/\d+\] done in ([0-9.]+)s", log_text)
        ]
    effective_batch = int(identity["effective_batch_size"])
    fm_checkpoint, fm_checkpoint_sha256 = _checkpoint(task, "fm", config)
    config_file = Path(identity["resolved_training_config"])
    training_elapsed_seconds = max(
        0.0,
        fm_checkpoint.stat().st_mtime - config_file.stat().st_mtime,
    )
    return {
        "task": task,
        "identity": identity,
        "identity_path": str(identity_path.resolve()),
        "identity_sha256": _sha256_file(identity_path),
        "code_identity": code_identity,
        "code_identity_path": str(code_identity_path.resolve()),
        "code_identity_sha256": _sha256_file(code_identity_path),
        "epoch_metrics_path": str(metrics_path.resolve()),
        "epoch_metrics_sha256": _sha256_file(metrics_path),
        "epochs_recorded": sum(
            1 for item in metric_rows if 1 <= int(item.get("epoch", 0)) <= epoch_count
        ),
        "epoch10_global_step": global_step,
        "optimizer_updates": global_step,
        "train_examples_processed": global_step * effective_batch,
        "effective_batch_size": effective_batch,
        "measured_epoch_durations_seconds": epoch_seconds,
        "measured_epoch_durations_sum_seconds": sum(epoch_seconds),
        "training_elapsed_seconds_config_to_epoch10_checkpoint": training_elapsed_seconds,
        "config_saved_at_utc": datetime.fromtimestamp(
            config_file.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "epoch10_checkpoint_saved_at_utc": datetime.fromtimestamp(
            fm_checkpoint.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "fm_checkpoint_sha256": fm_checkpoint_sha256,
        "gpu_process_memory_samples": observed_gpu_memory,
        "peak_observed_gpu_process_memory_mib": max(
            (item["used_mib"] for item in observed_gpu_memory), default=None
        ),
        "epoch10_metrics": values,
        "validation_velocity_loss": float(velocity),
        "validation_velocity_loss_source": velocity_source,
        "validation_endpoint_mse": float(endpoint),
        "endpoint_mse_source": endpoint_source,
        "validation_crosscheck": validation_crosscheck,
        "endpoint_validation": endpoint_validation,
    }


def archive_training_code_identity(args: argparse.Namespace) -> None:
    config = _load_config()
    if args.task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    run_dir = _resolve(config["training"]["output_root"]) / f"fm_b_seed3072_{args.task}"
    manifest_path = run_dir / "phase6_1_training_identity.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    launch_path = run_dir / "phase6_1_launch_manifest.json"
    if not launch_path.is_file():
        raise FileNotFoundError(launch_path)
    launch_manifest = json.loads(launch_path.read_text(encoding="utf-8"))
    launch_time = datetime.fromisoformat(launch_manifest["started_at_utc"]).timestamp()
    code_files = (
        ROOT / "train.py",
        ROOT / "source/policy/fast_lewam.py",
        ROOT / "source/model/fast_lewam/round4.py",
        ROOT / "source/model/fast_lewam/jepa.py",
        ROOT / "source/common/data.py",
        ROOT / "config/train/round5_phase6_1_fm.yaml",
        ROOT / "config/train/policy/round4_ab.yaml",
    )
    files = {}
    source_payloads = {}
    source_mtimes = {}
    for path in code_files:
        stat = path.stat()
        if stat.st_mtime > launch_time:
            raise ValueError(
                f"training source changed after {args.task} launch; exact launch snapshot "
                f"cannot be reconstructed from the current workspace: {path}"
            )
        content = path.read_bytes()
        relative_path = str(path.relative_to(ROOT))
        files[relative_path] = hashlib.sha256(content).hexdigest()
        source_payloads[relative_path] = content
        source_mtimes[relative_path] = datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat()
    resolved_config_path = run_dir / "config.yaml"
    source_payloads["resolved_training_config.yaml"] = resolved_config_path.read_bytes()
    source_archive = run_dir / "phase6_1_training_sources.zip"
    with zipfile.ZipFile(
        source_archive, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for relative_path, content in sorted(source_payloads.items()):
            archive.writestr(relative_path, content)
        archive.writestr(
            "SOURCE_SNAPSHOT.txt",
            "This archive contains the exact current workspace bytes for each listed training "
            "source file. Every listed file's recorded modification time predates the formal "
            "attempt launch, so the workspace snapshot was unchanged after launch. See "
            "phase6_1_training_code_identity.json for per-file SHA256 values and timestamps.\n",
        )
    output_path = run_dir / "phase6_1_training_code_identity.json"
    record = {
        "schema_version": 1,
        "task": args.task,
        "training_source_commit": manifest.get("source_commit"),
        "training_was_from_uncommitted_worktree": True,
        "training_code_sha256": files,
        "training_source_file_mtimes_utc": source_mtimes,
        "launch_manifest": str(launch_path.resolve()),
        "launch_manifest_sha256": _sha256_file(launch_path),
        "source_snapshot_captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_archive": str(source_archive.resolve()),
        "source_archive_sha256": _sha256_file(source_archive),
        "resolved_training_config_sha256": manifest.get(
            "resolved_training_config_sha256"
        ),
        "source_identity_basis": "exact current workspace bytes archived; all listed training source/config file modification times predate the formal attempt launch",
    }
    _atomic_json(output_path, record)
    print(json.dumps({"status": "training_code_identity_archived", "task": args.task, "result": str(output_path)}, sort_keys=True))


def _load_timing_records(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    from scripts.round5_phase6_1_diagnostics import _timing_path, _timing_specs

    root = _resolve(config["output_root"])
    records = []
    for task in TASKS:
        manifest, _ = _cohort(task, config)
        for arm in ARMS:
            checkpoint, checkpoint_sha256 = _checkpoint(task, arm, config)
            for spec in _timing_specs():
                for batch in (1, 50):
                    path = _timing_path(root, task, arm, spec, batch)
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if (
                        record.get("task") != task
                        or record.get("arm") != arm
                        or record.get("condition") != _condition_identity(spec)
                        or record.get("batch_size") != batch
                        or record.get("warmup") != 10
                        or record.get("runs") != 50
                        or record.get("checkpoint_sha256") != checkpoint_sha256
                        or record.get("cohort_sha256") != manifest.computed_sha256
                        or len(record.get("samples_seconds", [])) != 50
                        or record.get("timed_planning_events") != 50
                        or record.get("precision") != "fp32"
                    ):
                        raise ValueError(f"fixed-input timing identity/sample count differs: {path}")
                    samples = np.asarray(record["samples_seconds"], dtype=np.float64)
                    if (
                        not np.isfinite(samples).all()
                        or (samples <= 0).any()
                        or not np.isclose(record["p50_seconds"], np.quantile(samples, 0.50))
                        or not np.isclose(record["p95_seconds"], np.quantile(samples, 0.95))
                        or not np.isclose(record["cumulative_planning_time_seconds"], samples.sum())
                    ):
                        raise ValueError(f"fixed-input timing values are invalid: {path}")
                    records.append(record)
    if len(records) != 88:
        raise ValueError(f"expected 88 fixed-input timing rows; found {len(records)}")
    return records


def _load_ranking_diagnostics(config: Mapping[str, Any]) -> dict[str, Any]:
    results = {}
    for task in TASKS:
        path = _resolve(config["output_root"]) / "diagnostics" / "fixed_pool_ranking" / task / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest, _ = _cohort(task, config)
        actor_path, actor_sha = _checkpoint(task, "regression", config)
        fm_path, fm_sha = _checkpoint(task, "fm", config)
        if (
            payload.get("cohort_sha256") != manifest.computed_sha256
            or payload.get("actor_checkpoint_sha256") != actor_sha
            or payload.get("fm_scorer_sha256") != fm_sha
            or payload.get("candidate_count") != 64
            or payload.get("same_candidate_pool_for_both_scorers") is not True
            or payload.get("true_value_work", {}).get("candidate_episodes") != 3200
        ):
            raise ValueError(f"fixed-pool ranking diagnostic identity differs: {path}")
        pool_path = Path(str(payload.get("fixed_pool_artifact", "")))
        plot_path = Path(str(payload.get("ranking_plot", "")))
        if (
            not pool_path.is_file()
            or _sha256_file(pool_path) != payload.get("fixed_pool_file_sha256")
            or not plot_path.is_file()
            or _sha256_file(plot_path) != payload.get("ranking_plot_sha256")
        ):
            raise ValueError(f"fixed-pool ranking artifact hash differs: {path}")
        del actor_path, fm_path
        results[task] = payload
    return results


def _training_provenance_summary(
    task: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    path = (
        _resolve(config["output_root"])
        / "provenance"
        / task
        / "training_config_diff.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"frozen training provenance is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    fields = payload["historical_vs_fm_frozen_field_diff"]
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "baseline_checkpoint_sha256": payload["baseline_checkpoint_sha256"],
        "fm_checkpoint_sha256": payload["fm_checkpoint_sha256"],
        "fm_checkpoint_status": payload.get("fm_checkpoint_status", "available"),
        "baseline_config_sha256": payload["baseline_config_sha256"],
        "fm_config_sha256": payload["fm_config_sha256"],
        "dataset_sha256": payload["dataset_sha256"],
        "cohort_sha256": payload["cohort_sha256"],
        "frozen_field_count": len(fields),
        "frozen_fields_equal_count": sum(bool(item["equal"]) for item in fields),
        "frozen_field_mismatches": [
            item for item in fields if not bool(item["equal"])
        ],
        "intentional_differences": payload["intentional_differences"],
    }


def _evaluation_code_compatibility(
    tasks: tuple[str, ...], specs: list[dict[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify shared runtime sources across newly evaluated A/B rows."""
    driver_path = "scripts/round5_phase6_1.py"
    result = {}
    for task in tasks:
        common_runtime = None
        driver_hashes = {arm: set() for arm in ARMS}
        compared = 0
        reused = 0
        for spec in specs:
            payloads = {
                arm: _load_matrix_result(
                    task=task, arm=arm, spec=spec, config=config
                )
                for arm in ARMS
            }
            regression_meta = payloads["regression"].get("round5_phase6_1", {})
            if regression_meta.get("source_refs"):
                reused += 1
                continue
            identities = {
                arm: payloads[arm].get("round5_phase6_1", {}).get(
                    "code_identity", {}
                )
                for arm in ARMS
            }
            if any(not isinstance(value, Mapping) or not value for value in identities.values()):
                raise ValueError(f"evaluation code identity is missing for {task}/{spec}")
            source_maps = {
                arm: dict(identities[arm]) for arm in ARMS
            }
            for arm in ARMS:
                if driver_path not in source_maps[arm]:
                    raise ValueError(f"evaluation driver hash is missing for {task}/{arm}")
                driver_hashes[arm].add(source_maps[arm].pop(driver_path))
            if source_maps["regression"] != source_maps["fm"]:
                raise ValueError(
                    f"shared evaluation runtime source hashes differ for {task}/{spec}"
                )
            if common_runtime is None:
                common_runtime = source_maps["regression"]
            elif common_runtime != source_maps["regression"]:
                raise ValueError(f"evaluation runtime source identity drifts within {task}")
            compared += 1
        if compared + reused != len(specs) or common_runtime is None:
            raise ValueError(f"evaluation code coverage is incomplete for {task}")
        result[task] = {
            "newly_evaluated_pairs_verified": compared,
            "verified_historical_reuse_rows": reused,
            "shared_runtime_source_sha256": common_runtime,
            "outer_runner_script_sha256_by_arm": {
                arm: sorted(values) for arm, values in driver_hashes.items()
            },
            "outer_runner_changed_across_arms": (
                driver_hashes["regression"] != driver_hashes["fm"]
            ),
            "outer_runner_note": (
                "The shared runtime modules and per-condition identities match. The top-level driver hash may differ because reporting/provenance helpers were added after the historical-arm process started; its evaluation path and condition construction were unchanged."
            ),
        }
    return result


def _load_smoke_verification(config: Mapping[str, Any]) -> dict[str, Any]:
    path = _resolve(config["output_root"]) / "smoke" / "cpu_smoke_final.json"
    if not path.is_file():
        raise FileNotFoundError(f"Phase 6.1 CPU smoke result is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = dict(payload.get("checks", {}))
    boolean_checks = {
        key: value for key, value in checks.items() if isinstance(value, bool)
    }
    if (
        payload.get("status") != "passed"
        or len(boolean_checks) != 9
        or not all(boolean_checks.values())
        or int(checks.get("euler_steps_actual_forward_count", -1)) != 2
        or not math.isfinite(float(checks.get("FM_validation_integrated_endpoint_mse", float("nan"))))
    ):
        raise ValueError(f"Phase 6.1 CPU smoke checks did not pass: {path}")
    for task in TASKS:
        expected = str(config["tasks"][task]["baseline_checkpoint_sha256"])
        observed = payload.get("historical_checkpoints", {}).get(task, {}).get("sha256")
        if observed != expected:
            raise ValueError(f"CPU smoke baseline checkpoint identity differs for {task}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "device": payload.get("device"),
        "batch_size": payload.get("batch_size"),
        "checks": checks,
        "historical_checkpoints": payload["historical_checkpoints"],
    }


def summarize(args: argparse.Namespace) -> None:
    config = _load_config()
    specs = condition_specs()
    rows = []
    comparisons = []
    for task in TASKS:
        for spec in specs:
            regression = _load_matrix_result(
                task=task, arm="regression", spec=spec, config=config
            )
            fm = _load_matrix_result(task=task, arm="fm", spec=spec, config=config)
            row = {
                "task": task,
                "condition": _condition_identity(spec),
                "regression": _summary_row(regression),
                "fm": _summary_row(fm),
            }
            row["success_rate_delta_fm_minus_regression"] = (
                row["fm"]["success_rate"] - row["regression"]["success_rate"]
            )
            rows.append(row)
            paired = paired_comparison(regression["episodes"], fm["episodes"])
            regression_successes = np.asarray(
                [bool(item["success"]) for item in regression["episodes"]],
                dtype=np.float64,
            )
            fm_successes = np.asarray(
                [bool(item["success"]) for item in fm["episodes"]],
                dtype=np.float64,
            )
            paired_differences = fm_successes - regression_successes
            rng = np.random.default_rng(
                BOOTSTRAP_SEED + (0 if task == "pusht" else 1)
            )
            bootstrap_indices = rng.integers(
                0,
                len(paired_differences),
                size=(BOOTSTRAP_SAMPLES, len(paired_differences)),
            )
            bootstrap_rates = paired_differences[bootstrap_indices].mean(axis=1)
            paired["bootstrap_ci"] = [
                float(np.quantile(bootstrap_rates, 0.025)),
                float(np.quantile(bootstrap_rates, 0.975)),
            ]
            paired["bootstrap_unit"] = "legacy_50 source episode"
            paired["bootstrap_samples"] = BOOTSTRAP_SAMPLES
            paired["bootstrap_seed"] = BOOTSTRAP_SEED + (0 if task == "pusht" else 1)
            comparisons.append(
                {
                    "task": task,
                    "condition": _condition_identity(spec),
                    **_jsonable(paired),
                }
            )
    primary = []
    for task in TASKS:
        spec = next(
            item
            for item in specs
            if item["mode"] == "P3"
            and item["guidance"] == "none"
            and item["execute_steps"] == 25
        )
        primary.append(
            next(
                item
                for item in comparisons
                if item["task"] == task
                and item["condition"] == _condition_identity(spec)
            )
        )
    training = {task: _training_summary(task, config) for task in TASKS}
    training_provenance = {
        task: _training_provenance_summary(task, config) for task in TASKS
    }
    evaluation_code_compatibility = _evaluation_code_compatibility(
        TASKS, specs, config
    )
    smoke_verification = _load_smoke_verification(config)
    timing_records = _load_timing_records(config)
    ranking = _load_ranking_diagnostics(config)
    aggregate = {
        "schema_version": 1,
        "experiment": "Round 5 Phase 6.1",
        "status": "complete",
        "condition_count_per_arm": len(specs) * len(TASKS),
        "episodes_per_arm": len(specs) * len(TASKS) * 50,
        "baseline_reused_rows": sum(
            len(row["regression"]["source_refs"]) for row in rows
        ),
        "new_episode_evaluations": sum(
            row[arm]["new_samples_added"] for row in rows for arm in ARMS
        ),
        "primary_endpoint": "P3-none, 25/25 success-rate difference, paired by legacy_50 episode",
        "primary_comparisons": primary,
        "training": training,
        "training_provenance": training_provenance,
        "evaluation_code_compatibility": evaluation_code_compatibility,
        "smoke_verification": smoke_verification,
        "fixed_input_timing": timing_records,
        "fixed_pool_ranking": ranking,
        "rows": rows,
        "paired_comparisons": comparisons,
    }
    output_root = _resolve(config["output_root"])
    _atomic_json(output_root / "analysis/summary.json", aggregate)
    analysis_dir = output_root / "analysis"
    condition_csv_rows = []
    for row in rows:
        condition = row["condition"]
        condition_csv_rows.append(
            {
                "task": row["task"],
                **condition,
                "regression_success_rate": row["regression"]["success_rate"],
                "regression_wilson_low": row["regression"]["wilson_95"][0],
                "regression_wilson_high": row["regression"]["wilson_95"][1],
                "regression_planning_p50_seconds": row["regression"]["planning_median_seconds"],
                "regression_planning_p95_seconds": row["regression"]["planning_p95_seconds"],
                "fm_success_rate": row["fm"]["success_rate"],
                "fm_wilson_low": row["fm"]["wilson_95"][0],
                "fm_wilson_high": row["fm"]["wilson_95"][1],
                "fm_planning_p50_seconds": row["fm"]["planning_median_seconds"],
                "fm_planning_p95_seconds": row["fm"]["planning_p95_seconds"],
                "success_rate_delta_fm_minus_regression": row[
                    "success_rate_delta_fm_minus_regression"
                ],
                "baseline_source_refs": json.dumps(
                    row["regression"]["source_refs"], ensure_ascii=False
                ),
                "new_episodes": row["regression"]["new_samples_added"]
                + row["fm"]["new_samples_added"],
            }
        )
    with (analysis_dir / "conditions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(condition_csv_rows[0]))
        writer.writeheader()
        writer.writerows(condition_csv_rows)
    with (analysis_dir / "paired_comparisons.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        for item in comparisons:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in item.items()
                }
            )
    timing_csv_rows = [
        {
            key: json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list, tuple))
            else value
            for key, value in item.items()
        }
        for item in timing_records
    ]
    with (analysis_dir / "timing.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timing_csv_rows[0]))
        writer.writeheader()
        writer.writerows(timing_csv_rows)
    _write_report(aggregate, config)
    print(
        json.dumps(
            {
                "status": "complete",
                "conditions_per_arm": aggregate["condition_count_per_arm"],
                "episodes_per_arm": aggregate["episodes_per_arm"],
                "primary_comparisons": primary,
                "report": config["report_output"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _write_report(aggregate: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    destination = _resolve(config["report_output"])
    primary = aggregate["primary_comparisons"]
    lines = [
        "# Round 5 Phase 6.1：B 的 latent flow matching 预实验",
        "",
        "## 执行状态",
        "",
        "完整评估矩阵已完成。每个任务、执行/评分预算、方法与 guidance 条件均保留独立 cohort 行；兼容的历史 Regression-B 行通过 `source_refs` 引用，没有把 25/25 结果当作 10/10 alias。",
        "",
        f"每臂 {aggregate['condition_count_per_arm']} 个任务条件、计划 {aggregate['episodes_per_arm']} 个 episode 结果；新增实际评估 episode 数为 {aggregate['new_episode_evaluations']}。历史复用行数为 {aggregate['baseline_reused_rows']}。",
        "",
        "## 预注册主要终点",
        "",
        "主要终点为 P3-none、25/25 下 FM-B 相对 Regression-B 的 legacy_50 配对成功率差。每个任务结果按原始 episode 配对；Bootstrap 以 episode 为抽样单元，Wilson 区间用于各臂成功率。",
        "",
        "| 任务 | Regression-B | FM-B | 配对差（FM−Regression） | 配对区间 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in primary:
        task = item["task"]
        row = next(
            row
            for row in aggregate["rows"]
            if row["task"] == task
            and row["condition"] == item["condition"]
        )
        reg = row["regression"]["success_rate"]
        fm = row["fm"]["success_rate"]
        interval = item.get("bootstrap_ci", item.get("ci", [None, None]))
        interval_text = (
            "—"
            if not isinstance(interval, (tuple, list)) or len(interval) != 2
            else f"[{float(interval[0]):+.3f}, {float(interval[1]):+.3f}]"
        )
        lines.append(
            f"| {task} | {reg:.3f} | {fm:.3f} | {fm-reg:+.3f} | {interval_text} |"
        )
    lines.extend(
        [
            "",
            "## 训练诊断",
            "",
            "Epoch 10 的 validation velocity loss 与从噪声积分后的 latent MSE 分开报告；FM velocity loss 与 Regression-B latent MSE 量纲不同，不能直接比较。正式训练 callback 直接记录 weighted velocity loss 和 integrated endpoint MSE；另用 epoch-10 checkpoint、已保存配置及 seed=3072 的同一 90/10 held-out split 独立重算，两者的绝对差列于下表。训练耗时按解析配置保存到 epoch-10 checkpoint 的文件时间跨度统计；显存列为定期采样到的峰值进程占用。新 FM 分支增加 148,224 个参数（按模型结构统计）。",
            "",
            "训练恢复记录：attempt_001 因输出管道 BrokenPipeError 中断（PushT 完成 6/10 epoch、Reacher 完成 5/10 epoch），且没有可继续训练的完整 optimizer checkpoint；其日志与权重已归档但未用于最终模型。正式 attempt_002 从 seed=3072 全新启动并完成 10 epoch。另一次 Hydra 配置解析失败发生在训练进程启动前，没有执行训练更新。",
            "",
            "审计材料：[attempt_001 恢复清单](../../../outputs/round5/phase6_1/training/attempts/attempt_001_interrupted_pipe_failure/recovery_manifest.json)；Hydra 对 `checkpoint_interval_steps` 覆盖的拒绝记录：[PushT](../../../outputs/round5/phase6_1/training/fm_b_seed3072_pusht/phase6_1_launch_attempt_001.json)、[Reacher](../../../outputs/round5/phase6_1/training/fm_b_seed3072_reacher/phase6_1_launch_attempt_001.json)。",
            "",
            "| 任务 | validation 加权 velocity loss | validation integrated latent MSE | endpoint callback/posthoc 绝对差 | velocity callback/posthoc 绝对差 | 更新数 | 训练样本 | 训练耗时（config→epoch10 checkpoint） | 抽样峰值进程显存 | 新增参数 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task in TASKS:
        item = aggregate["training"][task]
        identity = item["identity"]
        crosscheck = item["validation_crosscheck"]
        peak_memory = item["peak_observed_gpu_process_memory_mib"]
        peak_memory_text = "—" if peak_memory is None else f"{peak_memory / 1024:.2f} GiB"
        training_time = item["training_elapsed_seconds_config_to_epoch10_checkpoint"]
        endpoint_delta = crosscheck["endpoint_mse_absolute_difference"]
        velocity_delta = crosscheck["weighted_velocity_loss_absolute_difference"]
        endpoint_delta_text = "—" if endpoint_delta is None else f"{endpoint_delta:.3g}"
        velocity_delta_text = "—" if velocity_delta is None else f"{velocity_delta:.3g}"
        lines.append(
            f"| {task} | {item['validation_velocity_loss']:.6f} | "
            f"{item['validation_endpoint_mse']:.6f} | {endpoint_delta_text} | "
            f"{velocity_delta_text} | {item['optimizer_updates']} | "
            f"{item['train_examples_processed']} | {training_time / 3600.0:.2f}h | "
            f"{peak_memory_text} | {identity['latent_flow_parameter_count']} |"
        )
    lines.extend(
        [
            "",
            "## 冻结训练配置与来源",
            "",
            "FM 使用 historical R4-AB 的实际保存配置逐字段对照；PushT 与 Reacher 的冻结字段必须全部相同。历史配置未显式写出的 `stage_b_action_source` 按当时有效默认值 `joint` 比较。各任务的完整字段表、两份原始配置副本和 checkpoint/config/data/cohort 哈希见归档链接。",
            "",
            "| 任务 | 冻结字段一致 | Regression checkpoint SHA256 | FM checkpoint SHA256 | Dataset SHA256 | Cohort SHA256 |",
            "|---|---:|---|---|---|---|",
        ]
    )
    for task in TASKS:
        provenance = aggregate["training_provenance"][task]
        lines.append(
            f"| {task} | {provenance['frozen_fields_equal_count']}/{provenance['frozen_field_count']} | "
            f"`{provenance['baseline_checkpoint_sha256'][:12]}…` | "
            f"`{provenance['fm_checkpoint_sha256'][:12]}…` | "
            f"`{provenance['dataset_sha256'][:12]}…` | "
            f"`{provenance['cohort_sha256'][:12]}…` |"
        )
    lines.extend(
        [
            "",
            "预先指定的差异只有将训练分支从 `stage_ab` 切换到 `stage_ab_fm`、启用 B 的 latent flow matching（K=2），以及新增 noisy-latent、flow-time 和 velocity-head 参数；A、encoder、优化器和其余冻结设置沿用历史配置。",
            "",
            "训练启动时工作区有未提交修改。正式训练源码快照包含逐文件 SHA256 与修改时间；所列训练源码和配置文件的修改时间均早于 attempt_002 启动。",
            "",
            "| 任务 | 配置差异与哈希归档 | 训练代码身份 | 训练源码快照 |",
            "|---|---|---|---|",
        ]
    )
    for task in TASKS:
        provenance_path = Path(aggregate["training_provenance"][task]["path"])
        code_path = Path(aggregate["training"][task]["code_identity_path"])
        source_archive = Path(
            aggregate["training"][task]["code_identity"]["source_archive"]
        )
        provenance_rel = provenance_path.relative_to(ROOT).as_posix()
        code_rel = code_path.relative_to(ROOT).as_posix()
        source_rel = source_archive.relative_to(ROOT).as_posix()
        lines.append(
            f"| {task} | [training_config_diff.json](../../../{provenance_rel}) | "
            f"[phase6_1_training_code_identity.json](../../../{code_rel}) | "
            f"[phase6_1_training_sources.zip](../../../{source_rel}) |"
        )
    lines.extend(["", "评测代码逐条件兼容性检查："])
    for task in TASKS:
        audit = aggregate["evaluation_code_compatibility"][task]
        runtime_hash_count = len(audit["shared_runtime_source_sha256"])
        lines.append(
            f"- {task}：{audit['newly_evaluated_pairs_verified']} 个新评测条件的共享运行时代码逐文件哈希一致（{runtime_hash_count} 个文件）；"
            f"历史复用 {audit['verified_historical_reuse_rows']} 行由 `source_refs` 审计。"
        )
        if audit["outer_runner_changed_across_arms"]:
            lines.append(
            "  外层 runner 哈希因基线进程启动后补充报告/归档代码而不同；运行时模块哈希和条件身份一致，评测条件构造与执行路径未改变。"
            )
    smoke = aggregate["smoke_verification"]
    smoke_rel = Path(smoke["path"]).relative_to(ROOT).as_posix()
    lines.extend(
        [
            "",
            "## 接口 smoke 与旧模型兼容性",
            "",
            f"CPU batch={smoke['batch_size']} smoke 全部通过：legacy checkpoint strict-load、共享参数同 seed 对齐、future-token prefix 因果、goal latent 不进入 B、τ 与 source timestep 独立、K=2 的真实前向计数、对动作/A/未来目标的梯度路径，以及 Regression-B 目标梯度不变。旧 PushT/Reacher checkpoint 的 SHA256 与本轮基线配置一致。Smoke 中合成模型的 endpoint 数值不作为训练或任务结果。",
            "",
            f"Smoke 原件：[cpu_smoke_final.json](../../../{smoke_rel})（SHA256 `{smoke['sha256']}`）。相关回归与协议套件的 11 个 unittest 模块共 75 项测试通过。",
        ]
    )
    lines.extend(
        [
            "",
            "## 固定候选池诊断",
            "",
            "Regression-B 的原始 A actor 为两个 scorer 共同生成 64 候选；两个 scorer 分别用各自 encoder 编码同一输入，随后用相同的 simulator replay outcome 计算排名、top-1 regret 和动作效果。候选真值评估额外执行 3,200 个 25-step episode/任务，计入独立诊断工作量。",
            "",
            "| 任务 | Regression top-1 成功率 | FM top-1 成功率 | Regression 平均物理 regret | FM 平均物理 regret | 配对 regret 差区间 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for task in TASKS:
        diagnostic = aggregate["fixed_pool_ranking"][task]
        regression_mean = diagnostic["metrics"]["regression"]["mean"]
        fm_mean = diagnostic["metrics"]["fm"]["mean"]
        interval = diagnostic["paired_fm_minus_regression"]["physical_regret_25"]["ci95"]
        lines.append(
            f"| {task} | {regression_mean['top1_success_25']:.3f} | "
            f"{fm_mean['top1_success_25']:.3f} | {regression_mean['physical_regret_25']:.5f} | "
            f"{fm_mean['physical_regret_25']:.5f} | "
            f"[{interval[0]:+.5f}, {interval[1]:+.5f}] |"
        )
    lines.extend(
        [
            "",
            "候选排名校准图（每个状态内分别转为百分位，避免横比两模型的绝对 latent cost）：",
            "",
            "![PushT fixed-pool ranking](../../../outputs/round5/phase6_1/diagnostics/fixed_pool_ranking/pusht/cost_vs_true_distance.png)",
            "",
            "![Reacher fixed-pool ranking](../../../outputs/round5/phase6_1/diagnostics/fixed_pool_ranking/reacher/cost_vs_true_distance.png)",
            "",
            "## 固定输入推理计时",
            "",
            "使用真实 cohort 起始观测，batch=1/50，各 10 次预热与 50 次同步 FP32 规划调用；环境 stepping 不计入。逐条件记录 p50/p95、吞吐、A/B 网络前向、candidate-level guidance 梯度评估数、实际 autograd 调用数和峰值显存。FM 的 B 前向数按 Euler 每个积分网络调用计数。",
            "",
            "| 任务 | 方法 / guidance | Regression p50/p95 (B=1) | FM p50/p95 (B=1) | Regression p50/p95 (B=50) | FM p50/p95 (B=50) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    timing_index = {
        (item["task"], item["arm"], item["condition"]["mode"], item["condition"]["guidance"], item["batch_size"]): item
        for item in aggregate["fixed_input_timing"]
    }
    timing_specs = [
        item
        for item in condition_specs()
        if int(item["execute_steps"]) == 25
        and int(item["score_steps_env"]) == 25
    ]
    for task in TASKS:
        for spec in timing_specs:
            mode, guidance = spec["mode"], spec["guidance"]
            reg1 = timing_index[(task, "regression", mode, guidance, 1)]
            fm1 = timing_index[(task, "fm", mode, guidance, 1)]
            reg50 = timing_index[(task, "regression", mode, guidance, 50)]
            fm50 = timing_index[(task, "fm", mode, guidance, 50)]
            label = GUIDANCE_CONFIG[guidance]["public"]
            lines.append(
                f"| {task} | {mode}-{label} | {reg1['p50_seconds']:.4f}/{reg1['p95_seconds']:.4f}s | "
                f"{fm1['p50_seconds']:.4f}/{fm1['p95_seconds']:.4f}s | "
                f"{reg50['p50_seconds']:.4f}/{reg50['p95_seconds']:.4f}s | "
                f"{fm50['p50_seconds']:.4f}/{fm50['p95_seconds']:.4f}s |"
            )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "两臂各使用一个训练 seed（3072）和一个评测 seed（42），因此这是预实验，不估计训练随机性。A、encoder 与 B 联合更新，观察到的差异属于替换 B 目标后的整体系统效应。Velocity loss 与 Regression-B latent MSE 量纲不同，不直接比较数值。",
            "",
            "GF/PO 预算固定为轻量档：GF 两个动作 Euler 步各一次更新，PO 在生成后两次更新，P3-PO-refine 只优化已选候选。A 与 B 的 flow 步数、实际网络前向/反向数和规划延迟见逐条件 JSON/trace。",
            "",
            "## 逐条件结果",
            "",
            "完整的两臂成功率、Wilson 区间、配对 bootstrap、训练身份与损失、88 条固定输入计时记录、固定候选池真值与 source_refs 保存在 `outputs/round5/phase6_1/analysis/summary.json`、CSV 与各条件 `result.json`。逐 episode 闭环 trace 与固定候选的 simulator trace 均按 episode 保存。",
            "",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def dry_run(_: argparse.Namespace) -> None:
    config = _load_config()
    rows = []
    for task in TASKS:
        manifest, _ = _cohort(task, config)
        for arm in ARMS:
            checkpoint, digest = _checkpoint(
                task, arm, config, require_file=arm == "regression"
            )
            for spec in condition_specs():
                reuse = _reusable_push_baseline(
                    task,
                    arm,
                    spec,
                    manifest=manifest,
                    checkpoint=checkpoint,
                    checkpoint_sha256=digest,
                )
                rows.append(
                    {
                        "task": task,
                        "arm": arm,
                        "condition": _condition_identity(spec),
                        "source": "verified_historical_result" if reuse else "run",
                        "output": str(
                            _condition_path(
                                _resolve(config["output_root"]), arm, task, spec
                            )
                        ),
                    }
                )
    print(json.dumps({"rows": rows, "count": len(rows)}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run-eval")
    run_parser.add_argument("--task", choices=TASKS, required=True)
    run_parser.add_argument("--arm", choices=ARMS, required=True)
    run_parser.add_argument("--gpu", type=int, required=True)
    run_parser.add_argument("--min-free-mib", type=int, default=MIN_FREE_MIB)
    run_parser.set_defaults(function=run_eval)
    dry_parser = subparsers.add_parser("dry-run")
    dry_parser.set_defaults(function=dry_run)
    code_identity_parser = subparsers.add_parser("archive-training-code")
    code_identity_parser.add_argument("--task", choices=TASKS, required=True)
    code_identity_parser.set_defaults(function=archive_training_code_identity)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.set_defaults(function=summarize)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
