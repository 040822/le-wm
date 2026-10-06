#!/usr/bin/env python3
"""Run and analyze the Round 5 Phase 5 action-horizon inference experiment.

The first-stage matrix reuses only identity-verified 5x5 baseline artifacts;
all shortened execution/score conditions are written to a new output root.
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
from source.common.round3_phase1 import CohortManifest, paired_comparison, wilson_interval
from source.common.round4_eval import run_round4_evaluation, validate_gpu_visibility
from source.common.round3_validation import validate_result_payload


DEFAULT_CONFIG = ROOT / "config" / "round5" / "phase5.json"
DEFAULT_MIN_FREE_MIB = 3500
BOOTSTRAP_SAMPLES = 10_000
TIMING_WARMUP = 10
TIMING_RUNS = 50
EXECUTE_STEPS = (25, 10, 5, 1)
SCORE_STEPS = (25, 10, 5)
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


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Phase5 config must be a JSON object: {path}")
    return value


def _load_inputs(config: Mapping[str, Any]):
    cohort_config = config["cohort"]
    manifest_path = _resolve(cohort_config["path"])
    manifest = CohortManifest.load(manifest_path)
    if manifest.cohort_kind != "dev" or manifest.protocol_variant != "legacy":
        raise ValueError("Phase5 requires a legacy/dev Reacher cohort")
    if len(manifest.entries) != int(config["evaluation"]["num_eval"]):
        raise ValueError("Phase5 Reacher cohort has the wrong number of episodes")
    observed_cohort_sha = manifest.computed_sha256
    if observed_cohort_sha != str(cohort_config["sha256"]):
        raise ValueError(
            f"Reacher cohort SHA256 changed: {observed_cohort_sha} != "
            f"{cohort_config['sha256']}"
        )

    training = config["training"]
    checkpoint = _resolve(training["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    observed_checkpoint_sha = _sha256_file(checkpoint)
    if observed_checkpoint_sha != str(training["checkpoint_sha256"]):
        raise ValueError(
            f"R4-AB checkpoint SHA256 changed: {observed_checkpoint_sha} != "
            f"{training['checkpoint_sha256']}"
        )
    return manifest, manifest_path, checkpoint, observed_checkpoint_sha


def _condition_specs() -> list[dict[str, Any]]:
    """Return 58 unique cells; P0/no-guidance score aliases are deduplicated."""
    specs: list[dict[str, Any]] = []
    for execute_steps in EXECUTE_STEPS:
        score_steps = [25]
        if execute_steps in {10, 5}:
            score_steps.append(execute_steps)
        for score_step in score_steps:
            for mode, guidance in MODES:
                if score_step < 25 and mode == "P0" and guidance == "none":
                    continue
                specs.append(
                    {
                        "execute_steps": int(execute_steps),
                        "score_steps_env": int(score_step),
                        "score_horizon_blocks": int(score_step // 5),
                        "mode": mode,
                        "guidance": guidance,
                        "cem_protocol": (
                            "cem-clip" if mode in {"P1", "P2"} else "not_applicable"
                        ),
                        "action_flow_steps": None if mode == "P1" else 2,
                        "action_flow_integrator": (
                            "not_applicable" if mode == "P1" else "euler"
                        ),
                    }
                )
    if len(specs) != 58:
        raise AssertionError(f"expected 58 unique Phase5 conditions, got {len(specs)}")
    return specs


def _logical_specs() -> list[dict[str, Any]]:
    """Expand the 58 stored cells to the full 60-cell reporting matrix."""
    result: list[dict[str, Any]] = []
    for execute_steps in EXECUTE_STEPS:
        scores = [25]
        if execute_steps in {10, 5}:
            scores.append(execute_steps)
        for score_step in scores:
            for mode, guidance in MODES:
                logical = {
                    "execute_steps": int(execute_steps),
                    "score_steps_env": int(score_step),
                    "mode": mode,
                    "guidance": guidance,
                }
                if score_step < 25 and mode == "P0" and guidance == "none":
                    logical["storage_score_steps_env"] = 25
                    logical["score_alias"] = True
                else:
                    logical["storage_score_steps_env"] = int(score_step)
                    logical["score_alias"] = False
                result.append(logical)
    if len(result) != 60:
        raise AssertionError(f"expected 60 logical Phase5 cells, got {len(result)}")
    return result


def _condition_label(spec: Mapping[str, Any]) -> str:
    guidance = str(spec["guidance"])
    return f"{spec['mode']}+{'none' if guidance == 'none' else ('GF' if guidance == 'guided_flow' else 'PO')}"


def _condition_path(root: Path, spec: Mapping[str, Any]) -> Path:
    mode = str(spec["mode"])
    protocol = str(spec["cem_protocol"])
    guidance = str(spec["guidance"])
    step = "invariant" if mode == "P1" else f"step_{int(spec['action_flow_steps'])}"
    integrator = str(spec["action_flow_integrator"])
    return (
        root
        / "conditions"
        / f"exec_{int(spec['execute_steps'])}"
        / f"score_{int(spec['score_steps_env'])}"
        / mode
        / protocol
        / guidance
        / step
        / integrator
        / "dev"
    )


def _baseline_path(config: Mapping[str, Any], spec: Mapping[str, Any]) -> Path:
    cohort = config["cohort"]
    mode = str(spec["mode"])
    protocol = str(spec["cem_protocol"])
    guidance = str(spec["guidance"])
    step = "invariant" if mode == "P1" else f"step_{int(spec['action_flow_steps'])}"
    integrator = str(spec["action_flow_integrator"])
    if guidance == "none":
        root = _resolve(cohort["baseline_phase4_root"])
        components = (mode, protocol, step, integrator, "dev")
    else:
        root = _resolve(cohort["baseline_phase1_root"])
        components = (mode, protocol, guidance, step, integrator, "dev")
    return root / "conditions" / "legacy" / "reacher" / Path(*components) / "result.json"


def _expected_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
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


def _code_identity(config_path: Path) -> dict[str, Any]:
    files = (
        config_path,
        ROOT / "scripts" / "round5_phase5.py",
        ROOT / "source" / "common" / "round4_eval.py",
        ROOT / "source" / "policy" / "round4.py",
        ROOT / "source" / "policy" / "fast_lewam_eval.py",
        ROOT / "source" / "model" / "fast_lewam" / "jepa.py",
        ROOT / "source" / "model" / "fast_lewam" / "round4.py",
    )
    hashes = {str(path.relative_to(ROOT)): _sha256_file(path) for path in files}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "git_commit": commit,
        "tracked_and_worktree_sha256": hashes,
        "bundle_sha256": _canonical_sha256(hashes),
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256_file(config_path),
    }


def _validate_result(
    payload: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    manifest: CohortManifest,
    checkpoint: Path,
    spec: Mapping[str, Any],
    require_phase5_metadata: bool,
) -> None:
    if payload.get("status") != "ok" or len(payload.get("episodes", [])) != 50:
        raise ValueError("result is incomplete or not successful")
    if str(Path(str(payload.get("checkpoint", ""))).resolve()) != str(checkpoint.resolve()):
        raise ValueError("result checkpoint does not match the frozen R4-AB checkpoint")
    parameters = payload.get("parameters", {})
    expected_protocol = {
        "horizon": 5,
        "receding_horizon": 5,
        "action_block": 5,
        "seed": int(config["evaluation"]["seed"]),
        "goal_offset_steps": int(config["evaluation"]["goal_offset_steps"]),
        "eval_budget": int(config["evaluation"]["eval_budget"]),
        "cohort_id": manifest.cohort_id,
        "cohort_sha256": manifest.computed_sha256,
    }
    for key, value in expected_protocol.items():
        if parameters.get(key) != value:
            raise ValueError(
                f"result protocol mismatch for {key}: {parameters.get(key)!r} != {value!r}"
            )
    if payload.get("trace_content_sha256") is None and payload.get("trace_sha256") is None:
        raise ValueError("result lacks an integrity-checked episode trace")
    entries = [
        (str(entry.episode_id), int(entry.start_step)) for entry in manifest.entries
    ]
    records = payload["episodes"]
    observed = [
        (
            str(item.get("episode_id", item.get("dataset_episode"))),
            int(item.get("start_step", 0)),
        )
        for item in records
    ]
    if observed != entries:
        raise ValueError("result episode identities/order differ from legacy_50")

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
            raise ValueError(
                f"result condition mismatch for {key}: {actual!r} != {value!r}"
            )
    if spec["mode"] in {"P1", "P2"} and planning.get("action_bound_mode") != "candidate_clip":
        raise ValueError("P1/P2 Reacher results must use cem-clip")
    if require_phase5_metadata:
        phase5 = payload.get("round5_phase5")
        if not isinstance(phase5, Mapping) or phase5.get("condition") != _expected_spec(spec):
            raise ValueError("result is missing matching Round5 Phase5 condition metadata")
        if (
            parameters.get("execute_steps") != int(spec["execute_steps"])
            or parameters.get("score_horizon_blocks")
            != int(spec["score_horizon_blocks"])
        ):
            raise ValueError("result execution or scoring horizon metadata differs")
    validate_result_payload(
        payload,
        manifest=manifest,
        expected_count=len(manifest.entries),
    )


def _read_baseline(
    config: Mapping[str, Any],
    manifest: CohortManifest,
    checkpoint: Path,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    path = _baseline_path(config, spec)
    if not path.is_file():
        raise FileNotFoundError(f"expected reusable 5x5 baseline is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_result(
        payload,
        config=config,
        manifest=manifest,
        checkpoint=checkpoint,
        spec=spec,
        require_phase5_metadata=False,
    )
    if int(payload["parameters"].get("receding_horizon", 0)) * int(
        payload["parameters"].get("action_block", 0)
    ) != 25:
        raise ValueError(f"baseline does not execute the full 25-step plan: {path}")
    payload["_phase5_result_path"] = str(path.resolve())
    return payload


def _gpu(value: str) -> str:
    if not value.isdigit() or int(value) not in range(8):
        raise argparse.ArgumentTypeError("GPU must be one physical ID in 0..7")
    return value


def _gpu_preflight(gpu: str, minimum_free_mib: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [
        "nvidia-smi",
        "--id",
        str(gpu),
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [item.strip() for item in completed.stdout.splitlines()[0].split(",")]
    if len(fields) != 6:
        raise RuntimeError(f"invalid GPU{gpu} nvidia-smi response: {completed.stdout!r}")
    snapshot = {
        "gpu": int(gpu),
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
    print(json.dumps({"gpu_preflight": snapshot}, ensure_ascii=False, sort_keys=True))
    return snapshot


def _configure_device(device: str, gpu: str, minimum_free_mib: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    validate_gpu_visibility(device)
    _gpu_preflight(gpu, minimum_free_mib)


def _compose(manifest: CohortManifest, device: str):
    cfg = compose_eval_config(
        "reacher",
        overrides=[
            f"eval.num_eval={len(manifest.entries)}",
            "output.save_video=false",
            f"solver.device={device}",
            "solver.num_samples=300",
            "solver.topk=30",
            "solver.n_steps=30",
            "solver.var_scale=1.0",
        ],
    )
    cfg.solver.device = device
    return cfg


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _phase5_metadata(
    *,
    config: Mapping[str, Any],
    code_identity: Mapping[str, Any],
    manifest: CohortManifest,
    checkpoint_sha256: str,
    spec: Mapping[str, Any],
    gpu: str,
) -> dict[str, Any]:
    return {
        "experiment": "Round 5 Phase 5",
        "stage": "inference_action_horizon_matrix",
        "condition": _expected_spec(spec),
        "condition_sha256": _canonical_sha256(_expected_spec(spec)),
        "cohort_id": manifest.cohort_id,
        "cohort_sha256": manifest.computed_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": code_identity["config_sha256"],
        "code_identity": dict(code_identity),
        "gpu": int(gpu),
        "requested_runner_model": config.get("requested_runner_model"),
        "requested_reasoning_effort": config.get("requested_reasoning_effort"),
        "created_by": "scripts/round5_phase5.py",
    }


def _run_one(
    *,
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    device: str,
    gpu: str,
    model: Any,
    manifest: CohortManifest,
    checkpoint: Path,
    checkpoint_sha256: str,
    code_identity: Mapping[str, Any],
) -> dict[str, Any]:
    target = _condition_path(output_root, spec)
    result_path = target / "result.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        _validate_result(
            payload,
            config=config,
            manifest=manifest,
            checkpoint=checkpoint,
            spec=spec,
            require_phase5_metadata=True,
        )
        print(json.dumps({"status": "resumed", "result": str(result_path)}))
        return payload
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"condition directory is partially populated: {target}")

    evaluation = config["evaluation"]
    guidance = config["guidance"]
    cfg = _compose(manifest, device)
    identity = EvaluationIdentity(
        entrypoint="round5_phase5",
        policy_kind="round4_shared_dit",
        checkpoint=str(checkpoint.resolve()),
        epoch=int(config["training"]["epoch"]),
        stage=str(spec["mode"]),
    )
    payload = run_round4_evaluation(
        cfg,
        task="reacher",
        policy_or_model=model,
        mode=str(spec["mode"]),
        identity=identity,
        manifest=manifest,
        output_dir=target,
        trace_output_dir=target / "trace",
        device=device,
        trace=True,
        candidate_count=int(evaluation["best_of_n_candidates"]),
        flow_steps=16,
        action_flow_steps=spec["action_flow_steps"],
        solver_batch_size=int(evaluation["solver_batch_size"]),
        candidate_batch_size=int(evaluation["candidate_batch_size"]),
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
        raise RuntimeError(f"Round4 evaluator returned a non-result: {payload.get('status')}")
    payload["round5_phase5"] = _phase5_metadata(
        config=config,
        code_identity=code_identity,
        manifest=manifest,
        checkpoint_sha256=checkpoint_sha256,
        spec=spec,
        gpu=gpu,
    )
    _atomic_write_json(result_path, payload)
    _validate_result(
        payload,
        config=config,
        manifest=manifest,
        checkpoint=checkpoint,
        spec=spec,
        require_phase5_metadata=True,
    )
    print(json.dumps({"status": "ok", "result": str(result_path)}))
    return payload


def validate(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    manifest, manifest_path, checkpoint, checkpoint_sha = _load_inputs(config)
    baseline_rows = []
    baseline_specs = [
        spec
        for spec in _condition_specs()
        if int(spec["execute_steps"]) == 25
    ]
    for spec in baseline_specs:
        payload = _read_baseline(config, manifest, checkpoint, spec)
        baseline_rows.append(
            {
                "mode": spec["mode"],
                "guidance": spec["guidance"],
                "result": payload["_phase5_result_path"],
                "successes": int(sum(bool(item["success"]) for item in payload["episodes"])),
            }
        )
    result = {
        "status": "ok",
        "cohort": str(manifest_path.resolve()),
        "cohort_id": manifest.cohort_id,
        "cohort_sha256": manifest.computed_sha256,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "unique_conditions": len(_condition_specs()),
        "logical_conditions": len(_logical_specs()),
        "reused_baselines": baseline_rows,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def dry_run(args: argparse.Namespace) -> None:
    config = _load_config(_resolve(args.config))
    root = _resolve(args.output_root or config["output_root"])
    baselines = {
        (
            int(spec["execute_steps"]),
            int(spec["score_steps_env"]),
            spec["mode"],
            spec["guidance"],
        )
        for spec in _condition_specs()
        if int(spec["execute_steps"]) == 25
        and int(spec["score_steps_env"]) == 25
    }
    rows = []
    for index, spec in enumerate(_condition_specs()):
        is_baseline = int(spec["execute_steps"]) == 25
        rows.append(
            {
                "index": index,
                "condition": _expected_spec(spec),
                "source": "verified_historical_baseline" if is_baseline else "run",
                "output": str(
                    (_baseline_path(config, spec) if is_baseline else _condition_path(root, spec))
                ),
            }
        )
    aliases = [
        row for row in _logical_specs() if row.get("score_alias")
    ]
    print(
        json.dumps(
            {"unique_conditions": rows, "logical_aliases": aliases},
            ensure_ascii=False,
            indent=2,
        )
    )


def run_worker(args: argparse.Namespace) -> None:
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    manifest, _, checkpoint, checkpoint_sha = _load_inputs(config)
    specs = _condition_specs()
    indexes = [int(value) for value in args.indices.split(",") if value]
    if len(set(indexes)) != len(indexes) or any(
        value < 0 or value >= len(specs) for value in indexes
    ):
        raise ValueError("--indices contains duplicate or out-of-range condition indexes")
    selected = [specs[index] for index in indexes]
    output_root = _resolve(args.output_root or config["output_root"])

    # Historical baseline-only shards perform read-only validation and do not
    # allocate a GPU or deserialize model weights.
    for spec in selected:
        if int(spec["execute_steps"]) == 25:
            _read_baseline(config, manifest, checkpoint, spec)
            continue
        existing_result = _condition_path(output_root, spec) / "result.json"
        if existing_result.is_file():
            existing_payload = json.loads(existing_result.read_text(encoding="utf-8"))
            _validate_result(
                existing_payload,
                config=config,
                manifest=manifest,
                checkpoint=checkpoint,
                spec=spec,
                require_phase5_metadata=True,
            )
    pending = [
        spec
        for spec in selected
        if int(spec["execute_steps"]) != 25
        and not (_condition_path(output_root, spec) / "result.json").is_file()
    ]
    if not pending:
        print(json.dumps({"status": "validated_baseline_only", "conditions": len(selected)}))
        return

    device = f"cuda:0"
    _configure_device(device, args.gpu, int(args.min_free_mib))
    model, resolved_checkpoint = load_policy_or_model(str(checkpoint))
    if resolved_checkpoint is not None and Path(resolved_checkpoint).resolve() != checkpoint.resolve():
        raise ValueError("checkpoint loader resolved a different weights file")
    if int(getattr(model, "action_horizon", -1)) != int(config["evaluation"]["action_horizon_blocks"]):
        raise ValueError("checkpoint action_horizon does not match the Phase5 5x5 model")
    if int(getattr(model, "action_dim", -1)) != 10:
        raise ValueError("checkpoint action_dim does not match Reacher 5x5 (10)")
    code_identity = _code_identity(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "task": "reacher",
                "gpu": int(args.gpu),
                "pending_conditions": len(pending),
                "condition_indices": indexes,
                "checkpoint_sha256": checkpoint_sha,
                "cohort_sha256": manifest.computed_sha256,
            },
            sort_keys=True,
        )
    )
    for spec in selected:
        if spec not in pending:
            continue
        if int(spec["execute_steps"]) == 25:
            continue
        _run_one(
            spec=spec,
            config=config,
            config_path=config_path,
            output_root=output_root,
            device=device,
            gpu=args.gpu,
            model=model,
            manifest=manifest,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            code_identity=code_identity,
        )


def run_matrix(args: argparse.Namespace) -> None:
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    # Verify all frozen local identities and the 10 reusable baseline rows
    # before consuming any GPU time.
    validate(argparse.Namespace(config=str(config_path)))
    gpu_values = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_values or len(set(gpu_values)) != len(gpu_values):
        raise ValueError("--gpus must list distinct physical GPU IDs")
    gpus = [_gpu(value) for value in gpu_values]
    for gpu in gpus:
        _gpu_preflight(gpu, int(args.min_free_mib))

    specs = _condition_specs()
    output_root = _resolve(args.output_root or config["output_root"])
    processes = []
    for shard, gpu in enumerate(gpus):
        indexes = list(range(shard, len(specs), len(gpus)))
        if not indexes:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "round5_phase5.py"),
            "run-worker",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--gpu",
            gpu,
            "--min-free-mib",
            str(args.min_free_mib),
            "--indices",
            ",".join(map(str, indexes)),
        ]
        print(json.dumps({"launch_gpu": int(gpu), "conditions": len(indexes)}))
        processes.append((gpu, subprocess.Popen(command, cwd=ROOT, env=env)))
    results = [(gpu, process.wait()) for gpu, process in processes]
    failures = [(gpu, code) for gpu, code in results if code]
    if failures:
        raise SystemExit(f"Phase5 evaluation workers failed: {failures}")
    print(json.dumps({"status": "matrix_complete", "unique_conditions": len(specs)}))


def _timing_specs() -> list[dict[str, Any]]:
    """Use the 10 frozen mode/guidance policies at full 25-step scoring."""
    return [
        spec
        for spec in _condition_specs()
        if int(spec["execute_steps"]) == 25
        and int(spec["score_steps_env"]) == 25
    ]


def _timing_result_path(output_root: Path, spec: Mapping[str, Any], batch_size: int) -> Path:
    mode = str(spec["mode"])
    guidance = str(spec["guidance"])
    return (
        output_root
        / "analysis"
        / "timing_conditions"
        / f"exec25_score25_{mode}_{guidance}_batch{int(batch_size)}.json"
    )


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


def _timing_manifest(manifest: CohortManifest, batch_size: int) -> CohortManifest:
    if int(batch_size) == 50:
        return manifest
    if int(batch_size) != 1:
        raise ValueError("Phase5 fixed-input timing supports batch sizes 1 and 50")
    result = replace(
        manifest,
        cohort_id=f"{manifest.cohort_id}_phase5_timing1",
        cohort_kind="custom",
        entries=(manifest.entries[0],),
        episode_split={
            **manifest.episode_split,
            "selected": (manifest.entries[0].episode_id,),
        },
    )
    return replace(result, cohort_sha256=result.computed_sha256)


def _benchmark_one(
    *,
    spec: Mapping[str, Any],
    batch_size: int,
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    device: str,
    gpu: str,
    model: Any,
    manifest: CohortManifest,
    checkpoint: Path,
    checkpoint_sha256: str,
    code_identity: Mapping[str, Any],
) -> dict[str, Any]:
    target = _timing_result_path(output_root, spec, batch_size)
    expected_condition = _expected_spec(spec)
    if target.is_file():
        saved = json.loads(target.read_text(encoding="utf-8"))
        if (
            saved.get("condition") != expected_condition
            or saved.get("batch_size") != int(batch_size)
            or saved.get("warmup") != TIMING_WARMUP
            or saved.get("runs") != TIMING_RUNS
            or saved.get("checkpoint_sha256") != checkpoint_sha256
            or saved.get("cohort_sha256") != manifest.computed_sha256
        ):
            raise ValueError(f"existing timing record has incompatible identity: {target}")
        return saved

    cfg = _compose(manifest, device)
    cfg.eval.num_eval = int(batch_size)
    cfg.world.num_envs = int(batch_size)
    timing_manifest = _timing_manifest(manifest, batch_size)
    identity = EvaluationIdentity(
        entrypoint="round5_phase5_fixed_input_timing",
        policy_kind="round4_shared_dit",
        checkpoint=str(checkpoint.resolve()),
        epoch=int(config["training"]["epoch"]),
        stage=str(spec["mode"]),
    )

    def capture(policy, *call_args, **call_kwargs):
        info = call_kwargs.get("info_dict")
        if info is None and call_args:
            info = call_args[0]
        if not isinstance(info, Mapping):
            raise TypeError("fixed-input benchmark expected the real policy info mapping")
        environment_batch_size = int(
            getattr(getattr(policy, "env", None), "num_envs", batch_size)
        )

        def infer_fixed_observation():
            replay_info = dict(info)
            replay_info["_needs_flush"] = np.ones(environment_batch_size, dtype=bool)
            return policy.get_action(replay_info)

        import torch

        synchronize = (
            (lambda: torch.cuda.synchronize(device))
            if str(device).startswith("cuda")
            else (lambda: None)
        )
        for _ in range(TIMING_WARMUP):
            infer_fixed_observation()
        synchronize()
        samples: list[float] = []
        for _ in range(TIMING_RUNS):
            started = time.perf_counter()
            infer_fixed_observation()
            synchronize()
            samples.append(time.perf_counter() - started)

        events = list(getattr(policy, "planning_events", ()))
        timed_events = events[-TIMING_RUNS:]
        if len(timed_events) != TIMING_RUNS:
            raise RuntimeError(
                f"expected {TIMING_RUNS} planning events, observed {len(timed_events)}"
            )
        mode = str(spec["mode"])
        if mode == "P0":
            stage_a = sum(
                int(
                    event.get("guidance_stats", {}).get(
                        "stage_a_forward_count", event.get("forward_count", 0)
                    )
                )
                for event in timed_events
            )
            stage_b = sum(
                int(event.get("guidance_stats", {}).get("stage_b_forward_count", 0))
                for event in timed_events
            )
        elif mode in {"P1", "P2"}:
            stage_a = sum(int(event.get("stage_a_forward_count", 0)) for event in timed_events)
            stage_b = sum(
                int(event.get("stage_b_forward_count", event.get("forward_count", 0)))
                for event in timed_events
            )
        else:
            stage_a = sum(
                int(event.get("stage_a_forward_count", event.get("proposal_forward_count", 0)))
                for event in timed_events
            )
            stage_b = sum(
                int(event.get("stage_b_forward_count", 0))
                + int(event.get("verifier_forward_count", 0))
                for event in timed_events
            )
        backward = sum(
            int(
                event.get(
                    "guidance_backward_count",
                    event.get("guidance_stats", {}).get("backward_count", 0),
                )
            )
            for event in timed_events
        )
        peak_values = [
            int(event["peak_memory_bytes"])
            for event in timed_events
            if event.get("peak_memory_bytes") is not None
        ]
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
            "timed_planning_events": len(timed_events),
            "peak_memory_bytes": max(peak_values) if peak_values else None,
        }

    gpu_before = _gpu_snapshot(gpu)
    cpu_load_before = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    result = run_round4_evaluation(
        cfg,
        task="reacher",
        policy_or_model=model,
        mode=str(spec["mode"]),
        identity=identity,
        manifest=timing_manifest,
        output_dir=output_root / "timing" / "scratch" / f"{spec['mode']}_{spec['guidance']}" / f"batch_{batch_size}",
        device=device,
        trace=False,
        candidate_count=int(config["evaluation"]["best_of_n_candidates"]),
        flow_steps=16,
        action_flow_steps=spec["action_flow_steps"],
        solver_batch_size=int(config["evaluation"]["solver_batch_size"]),
        candidate_batch_size=int(config["evaluation"]["candidate_batch_size"]),
        action_flow_integrator=str(spec["action_flow_integrator"]),
        cem_protocol=str(spec["cem_protocol"]),
        guidance_mode=str(spec["guidance"]),
        guidance_step_size=float(config["guidance"]["step_size"]),
        guidance_last_steps=int(config["guidance"]["last_steps"]),
        guidance_inner_steps=int(config["guidance"]["inner_steps"]),
        guidance_max_rms_offset=float(config["guidance"]["max_rms_offset"]),
        proposal_chunk_size=int(config["guidance"]["p3_proposal_chunk_size"]),
        allowed_protocol_variants=("legacy",),
        execute_steps=25,
        score_horizon_blocks=5,
        timing_capture_callback=capture,
    )
    if result.get("status") != "timing_capture":
        raise RuntimeError(f"fixed-input timing did not capture a policy call: {result.get('status')}")
    gpu_after = _gpu_snapshot(gpu)
    cpu_load_after = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    captured = dict(result["timing_capture"])
    record = {
        "schema_version": "round5_phase5_timing_v1",
        "condition": expected_condition,
        "condition_sha256": _canonical_sha256(expected_condition),
        "batch_size": int(batch_size),
        "warmup": TIMING_WARMUP,
        "runs": TIMING_RUNS,
        "source": "fixed real Reacher legacy_50 observation; policy inference only; environment stepping excluded",
        "execution_semantics": "25 primitive environment steps; full 25-step planning horizon",
        "cohort_id": manifest.cohort_id,
        "cohort_sha256": manifest.computed_sha256,
        "timing_observation_cohort_id": timing_manifest.cohort_id,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "process_id": os.getpid(),
        "logical_cpu_count": os.cpu_count(),
        "system_load_average_before": cpu_load_before,
        "system_load_average_after": cpu_load_after,
        "config_sha256": code_identity["config_sha256"],
        "code_identity": dict(code_identity),
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        **captured,
    }
    _atomic_write_json(target, record)
    print(
        json.dumps(
            {
                "status": "timed",
                "mode": spec["mode"],
                "guidance": spec["guidance"],
                "batch_size": batch_size,
                "p50_seconds": record["p50_seconds"],
                "result": str(target),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return record


def benchmark_worker(args: argparse.Namespace) -> None:
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    manifest, _, checkpoint, checkpoint_sha = _load_inputs(config)
    specs = _timing_specs()
    indexes = [int(value) for value in args.indices.split(",") if value]
    if len(set(indexes)) != len(indexes) or any(
        value < 0 or value >= len(specs) for value in indexes
    ):
        raise ValueError("timing --indices contains duplicate or out-of-range indexes")
    selected = [specs[index] for index in indexes]
    output_root = _resolve(args.output_root or config["output_root"])
    targets = [
        _timing_result_path(output_root, spec, batch)
        for spec in selected
        for batch in (1, 50)
    ]
    if all(path.is_file() for path in targets):
        print(json.dumps({"status": "timing_resumed", "conditions": len(selected)}))
        return
    device = "cuda:0"
    _configure_device(device, args.gpu, int(args.min_free_mib))
    model, resolved_checkpoint = load_policy_or_model(str(checkpoint))
    if resolved_checkpoint is not None and Path(resolved_checkpoint).resolve() != checkpoint.resolve():
        raise ValueError("checkpoint loader resolved a different weights file")
    if int(getattr(model, "action_horizon", -1)) != 5 or int(getattr(model, "action_dim", -1)) != 10:
        raise ValueError("fixed-input timing requires the verified 5x5 Reacher checkpoint")
    code_identity = _code_identity(config_path)
    for spec in selected:
        for batch_size in (1, 50):
            target = _timing_result_path(output_root, spec, batch_size)
            if target.is_file():
                saved = json.loads(target.read_text(encoding="utf-8"))
                if (
                    saved.get("condition") != _expected_spec(spec)
                    or saved.get("batch_size") != batch_size
                    or saved.get("runs") != TIMING_RUNS
                    or saved.get("checkpoint_sha256") != checkpoint_sha
                ):
                    raise ValueError(f"existing timing result identity mismatch: {target}")
                continue
            _benchmark_one(
                spec=spec,
                batch_size=batch_size,
                config=config,
                config_path=config_path,
                output_root=output_root,
                device=device,
                gpu=args.gpu,
                model=model,
                manifest=manifest,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha,
                code_identity=code_identity,
            )


def benchmark_matrix(args: argparse.Namespace) -> None:
    config_path = _resolve(args.config)
    validate(argparse.Namespace(config=str(config_path)))
    gpu_values = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_values or len(set(gpu_values)) != len(gpu_values):
        raise ValueError("--gpus must list distinct physical GPU IDs")
    gpus = [_gpu(value) for value in gpu_values]
    for gpu in gpus:
        _gpu_preflight(gpu, int(args.min_free_mib))
    specs = _timing_specs()
    config = _load_config(config_path)
    output_root = _resolve(args.output_root or config["output_root"])
    processes = []
    for shard, gpu in enumerate(gpus):
        indexes = list(range(shard, len(specs), len(gpus)))
        if not indexes:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "round5_phase5.py"),
            "benchmark-worker",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--gpu",
            gpu,
            "--min-free-mib",
            str(args.min_free_mib),
            "--indices",
            ",".join(map(str, indexes)),
        ]
        print(json.dumps({"benchmark_gpu": int(gpu), "conditions": indexes}))
        processes.append((gpu, subprocess.Popen(command, cwd=ROOT, env=env)))
    results = [(gpu, process.wait()) for gpu, process in processes]
    failures = [(gpu, code) for gpu, code in results if code]
    if failures:
        raise SystemExit(f"Phase5 fixed-input timing workers failed: {failures}")
    print(json.dumps({"status": "fixed_input_timing_complete", "conditions": len(specs)}))


def _result_for_logical(
    config: Mapping[str, Any],
    output_root: Path,
    logical: Mapping[str, Any],
    *,
    manifest: CohortManifest,
    checkpoint: Path,
) -> tuple[dict[str, Any], str]:
    execute_steps = int(logical["execute_steps"])
    score_steps = int(logical["storage_score_steps_env"])
    mode = str(logical["mode"])
    guidance = str(logical["guidance"])
    spec = next(
        item
        for item in _condition_specs()
        if int(item["execute_steps"]) == execute_steps
        and int(item["score_steps_env"]) == score_steps
        and item["mode"] == mode
        and item["guidance"] == guidance
    )
    if execute_steps == 25:
        payload = _read_baseline(config, manifest, checkpoint, spec)
        return payload, str(payload["_phase5_result_path"])
    path = _condition_path(output_root, spec) / "result.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing Phase5 result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_result(
        payload,
        config=config,
        manifest=manifest,
        checkpoint=checkpoint,
        spec=spec,
        require_phase5_metadata=True,
    )
    return payload, str(path.resolve())


def _paired_bootstrap(baseline: Sequence[Any], candidate: Sequence[Any], seed: int) -> dict[str, Any]:
    comparison = paired_comparison(baseline, candidate)
    left = np.asarray([bool(item["success"]) for item in baseline], dtype=np.float64)
    right = np.asarray([bool(item["success"]) for item in candidate], dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("paired bootstrap requires equally sized non-empty outcomes")
    delta = right - left
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(BOOTSTRAP_SAMPLES, len(delta)))
    draws = delta[indices].mean(axis=1) * 100.0
    comparison["bootstrap_95_ci_pp"] = [float(value) for value in np.quantile(draws, [0.025, 0.975])]
    comparison["bootstrap_samples"] = BOOTSTRAP_SAMPLES
    return comparison


def _load_timing_records(
    output_root: Path,
    *,
    checkpoint_sha256: str,
    cohort_sha256: str,
) -> list[dict[str, Any]]:
    timing_dir = output_root / "analysis" / "timing_conditions"
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(timing_dir.glob("exec25_score25_*_batch*.json"))
    ]
    expected_count = len(_timing_specs()) * 2
    if len(records) != expected_count:
        raise ValueError(
            f"fixed-input timing is incomplete: expected {expected_count} records, "
            f"found {len(records)} in {timing_dir}"
        )
    seen = set()
    for item in records:
        key = (item.get("condition_sha256"), item.get("batch_size"))
        if key in seen:
            raise ValueError(f"duplicate fixed-input timing record: {key}")
        seen.add(key)
        if (
            item.get("schema_version") != "round5_phase5_timing_v1"
            or item.get("warmup") != TIMING_WARMUP
            or item.get("runs") != TIMING_RUNS
            or item.get("checkpoint_sha256") != checkpoint_sha256
            or item.get("cohort_sha256") != cohort_sha256
            or len(item.get("samples_seconds", [])) != TIMING_RUNS
            or item.get("timed_planning_events") != TIMING_RUNS
        ):
            raise ValueError("fixed-input timing record failed identity or sample validation")
    return records


def _result_row(
    logical: Mapping[str, Any], payload: Mapping[str, Any], path: str, source: str
) -> dict[str, Any]:
    episodes = list(payload["episodes"])
    successes = int(sum(bool(item.get("success", False)) for item in episodes))
    low, high = wilson_interval(successes, len(episodes))
    planning = payload.get("round4_planning", {})
    times = planning.get("planning_samples_seconds") or []
    times = [float(value) for value in times if value is not None]
    p50 = float(np.quantile(times, 0.50)) if times else planning.get("planning_median_seconds")
    p95 = float(np.quantile(times, 0.95)) if times else planning.get("planning_p95_seconds")
    planning_total = (
        float(sum(times))
        if times
        else (
            float(p50) * int(planning.get("replans") or 0)
            if p50 is not None
            else None
        )
    )
    first_success = [
        float(item["first_success_step"])
        for item in episodes
        if item.get("first_success_step") is not None
    ]
    eval_seconds = payload.get("evaluation_seconds")
    stage_a = planning.get("stage_a_forward_count")
    stage_b = planning.get("stage_b_forward_count")
    backward = planning.get("guidance_backward_count")
    if stage_a is None and logical["mode"] == "P0" and logical["guidance"] == "none":
        stage_a, stage_b, backward = planning.get("forward_count"), 0, 0
    if stage_b is None and logical["mode"] == "P1" and logical["guidance"] == "none":
        stage_a, stage_b, backward = 0, planning.get("forward_count"), 0
    if (
        stage_a is None
        and logical["mode"] == "P0"
        and logical["guidance"] in {"guided_flow", "post_opt"}
    ):
        replans = int(planning.get("replans") or 0)
        flow_steps = int(planning.get("action_flow_steps") or 0)
        inner_steps = int(planning.get("guidance_inner_steps") or 0)
        stage_a = flow_steps * replans * (
            1 + inner_steps if logical["guidance"] == "guided_flow" else 1
        )
        stage_b = (
            flow_steps * inner_steps * replans
            if logical["guidance"] == "guided_flow"
            else inner_steps * replans
        )
        backward = stage_b
    return {
        "execute_steps": int(logical["execute_steps"]),
        "score_steps_env": int(logical["score_steps_env"]),
        "score_steps_applied_env": int(logical["storage_score_steps_env"]),
        "score_alias": bool(logical.get("score_alias", False)),
        "mode": str(logical["mode"]),
        "guidance": str(logical["guidance"]),
        "condition": _condition_label(logical),
        "result_source": source,
        "result_path": path,
        "n": len(episodes),
        "successes": successes,
        "success_rate_percent": 100.0 * successes / len(episodes),
        "wilson_95_low_percent": 100.0 * low,
        "wilson_95_high_percent": 100.0 * high,
        "first_success_step_mean": float(np.mean(first_success)) if first_success else None,
        "first_success_step_median": float(np.median(first_success)) if first_success else None,
        "replans": planning.get("replans"),
        "planning_p50_seconds": p50,
        "planning_p95_seconds": p95,
        "planning_total_seconds": planning_total,
        "evaluation_seconds": eval_seconds,
        "episodes_per_second": (
            float(len(episodes) / eval_seconds)
            if eval_seconds is not None and float(eval_seconds) > 0
            else None
        ),
        "forward_count": planning.get("forward_count"),
        "stage_a_forward_count": stage_a,
        "stage_b_forward_count": stage_b,
        "guidance_backward_count": backward,
        "peak_memory_bytes": planning.get("peak_memory_bytes"),
        "trace_sha256": payload.get("trace_content_sha256", payload.get("trace_sha256")),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    manifest, _, checkpoint, checkpoint_sha = _load_inputs(config)
    output_root = _resolve(args.output_root or config["output_root"])
    logical_specs = _logical_specs()
    indexed: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for logical in logical_specs:
        payload, path = _result_for_logical(
            config,
            output_root,
            logical,
            manifest=manifest,
            checkpoint=checkpoint,
        )
        source = "historical_baseline" if int(logical["execute_steps"]) == 25 else "phase5"
        row = _result_row(logical, payload, path, source)
        rows.append(row)
        key = (
            int(logical["execute_steps"]),
            int(logical["score_steps_env"]),
            str(logical["mode"]),
            str(logical["guidance"]),
        )
        indexed[key] = payload

    comparisons: list[dict[str, Any]] = []
    for mode, guidance in MODES:
        baseline = indexed[(25, 25, mode, guidance)]["episodes"]
        for execute_steps in (10, 5, 1):
            candidate = indexed[(execute_steps, 25, mode, guidance)]["episodes"]
            comparisons.append(
                {
                    "comparison": "execute_prefix_vs_25",
                    "mode": mode,
                    "guidance": guidance,
                    "execute_steps": execute_steps,
                    "score_steps_env": 25,
                    **_paired_bootstrap(baseline, candidate, 50_000 + execute_steps),
                }
            )
    for execute_steps, score_steps in ((10, 10), (5, 5)):
        for mode, guidance in MODES:
            if mode == "P0" and guidance == "none":
                continue
            baseline = indexed[(execute_steps, 25, mode, guidance)]["episodes"]
            candidate = indexed[(execute_steps, score_steps, mode, guidance)]["episodes"]
            comparisons.append(
                {
                    "comparison": "short_score_vs_25",
                    "mode": mode,
                    "guidance": guidance,
                    "execute_steps": execute_steps,
                    "score_steps_env": score_steps,
                    **_paired_bootstrap(baseline, candidate, 60_000 + score_steps),
                }
            )

    fixed_input_timing = _load_timing_records(
        output_root,
        checkpoint_sha256=checkpoint_sha,
        cohort_sha256=manifest.computed_sha256,
    )
    analysis = {
        "experiment": "Round 5 Phase 5",
        "stage": "inference_action_horizon_matrix",
        "config": str(config_path.resolve()),
        "config_sha256": _sha256_file(config_path),
        "output_root": str(output_root.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "cohort_id": manifest.cohort_id,
        "cohort_sha256": manifest.computed_sha256,
        "unique_result_count": len(_condition_specs()),
        "logical_cell_count": len(rows),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_unit": "paired legacy_50 episode start",
        "fixed_input_timing": fixed_input_timing,
        "rows": rows,
        "comparisons": comparisons,
    }
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(analysis_dir / "analysis.json", analysis)
    fields = list(rows[0].keys())
    with (analysis_dir / "conditions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    comparison_fields = list(comparisons[0].keys())
    with (analysis_dir / "comparisons.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparisons)
    timing_fields = list(fixed_input_timing[0].keys())
    with (analysis_dir / "timing.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=timing_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(fixed_input_timing)
    report_path = _resolve(config["report_output"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(analysis),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "unique_results": len(_condition_specs()),
                "logical_cells": len(rows),
                "report": str(report_path.resolve()),
                "analysis": str((analysis_dir / "analysis.json").resolve()),
            },
            ensure_ascii=False,
        )
    )
    return analysis


def _render_report(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# Round 5 Phase 5：Action horizon 第一阶段推理实验",
        "",
        "本报告使用固定的 R4-AB 5×5 checkpoint，在 Reacher `legacy_50` 上比较执行前缀长度与 B 评分时点。每个格子为同一组 50 个起点上的闭环成功率；95% Wilson 区间描述单条件比例，差值区间通过配对 episode bootstrap（10,000 次）计算。结果为单 checkpoint、单 cohort 的探索性证据。",
        "",
        f"- Checkpoint SHA256：`{analysis['checkpoint_sha256']}`",
        f"- Cohort：`{analysis['cohort_id']}` / `{analysis['cohort_sha256']}`",
        "- 用户指定 runner 为 `gpt-6-luna/max`，配置已记录该请求；当前会话接口不能切换或核验实际运行模型，因此无法确认本轮执行模型与请求一致。",
        f"- 独立存储结果：{analysis['unique_result_count']}；逻辑矩阵单元：{analysis['logical_cell_count']}。P0 无引导不调用 B，其短评分单元复用同一执行长度的完整评分结果。",
        "- 规划始终生成完整 25 环境步；`execute_steps` 只截断环境执行前缀，动作耗尽后重规划。评分时点按 5-step action block 对齐；1-step 执行只报告 25-step 评分。",
        "- GF = guided-flow；PO = post-opt。P1/P2 使用 `cem-clip` 和 300/30/30 CEM；其余参数见逐条件 JSON 与配置。",
        "- A/B 前向与 guidance 反向计数从 planning event 汇总；历史 P2/P3 基线没有分支计数记录，故这些计数留空，原始总 forward 数和规划延迟仍可用。",
        "- 执行25步的 success rows 复用 Round4/Phase1 历史结果；它们的在线 latency 可能来自不同 GPU/系统负载，不作严格横向性能结论。固定观测同步计时表是本轮同协议的直接推理吞吐对照。",
        "",
        "## 全部条件",
        "",
        "| execute | score | condition | success | Wilson 95% | first success median | replans | plan p50 / p95 / total (s) | A/B forwards / B backward | episodes/s | source |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in analysis["rows"]:
        first = row["first_success_step_median"]
        replans = row["replans"]
        p50 = row["planning_p50_seconds"]
        p95 = row["planning_p95_seconds"]
        planning_total = row["planning_total_seconds"]
        throughput = row["episodes_per_second"]
        forward_counts = (
            f"{_fmt(row['stage_a_forward_count'], 0)} / "
            f"{_fmt(row['stage_b_forward_count'], 0)} / "
            f"{_fmt(row['guidance_backward_count'], 0)}"
        )
        ci = f"[{row['wilson_95_low_percent']:.1f}, {row['wilson_95_high_percent']:.1f}]"
        lines.append(
            f"| {row['execute_steps']} | {row['score_steps_env']} | {row['condition']} | "
            f"{row['successes']}/50 ({row['success_rate_percent']:.1f}%) | {ci} | "
            f"{_fmt(first)} | {_fmt(replans, 0)} | {_fmt(p50, 3)} / {_fmt(p95, 3)} / {_fmt(planning_total, 2)} | "
            f"{forward_counts} | {_fmt(throughput, 2)} | {row['result_source']} |"
        )
    horizon_combinations = [(25, 25), (10, 25), (10, 10), (5, 25), (5, 5), (1, 25)]
    rows_by_mode: dict[str, dict[tuple[int, int, str], Mapping[str, Any]]] = {}
    for row in analysis["rows"]:
        rows_by_mode.setdefault(row["mode"], {})[
            (int(row["execute_steps"]), int(row["score_steps_env"]), row["guidance"])
        ] = row
    lines.extend(
        [
            "",
            "## 各推理模式在不同 execute/score 组合下的成功率",
            "",
            "每格为成功数/样本数（成功率）；GF = guided-flow，PO = post-opt。",
        ]
    )
    mode_tables = [
        (
            "P0",
            ("none", "guided_flow", "post_opt"),
            "P0 不使用 B；无引导条件在 score=10/5 的格子复用同一 execute 长度的 score=25 结果。",
        ),
        (
            "P1",
            ("none",),
            "P1 本矩阵只运行无引导配置，GF/PO 未测。",
        ),
        ("P2", ("none", "guided_flow", "post_opt"), ""),
        (
            "P3",
            ("none", "guided_flow", "post_opt"),
            "execute=1 仅评 score=25：B 的预测 latent 位于每 5 步一个 action block 的边界，没有第 1 步 latent 可用于短评分。",
        ),
    ]
    for mode, guidances, note in mode_tables:
        guidance_headers = [
            f"{mode} 无引导" if guidance == "none" else f"{mode}+{'GF' if guidance == 'guided_flow' else 'PO'}"
            for guidance in guidances
        ]
        lines.extend(
            [
                "",
                f"### {mode}",
                "",
                "| execute | score | " + " | ".join(guidance_headers) + " |",
                "|---:|---:|" + "|".join("---:" for _ in guidances) + "|",
            ]
        )
        for execute_steps, score_steps in horizon_combinations:
            cells = []
            for guidance in guidances:
                row = rows_by_mode.get(mode, {}).get(
                    (execute_steps, score_steps, guidance)
                )
                if row is None:
                    cells.append("—")
                else:
                    cells.append(
                        f"{row['successes']}/{row['n']} ({row['success_rate_percent']:.1f}%)"
                    )
            lines.append(
                f"| {execute_steps} | {score_steps} | " + " | ".join(cells) + " |"
            )
        if note:
            lines.extend(["", note])
    lines.extend(
        [
            "",
            "## 固定观测同步计时",
            "",
            "对每种推理策略在固定的真实 Reacher cohort 观测上分别测 batch=1 和 batch=50；每组先预热 10 次，再同步测量 50 次，只计完整 policy inference，不计环境 step。计时使用完整 25-step action proposal 和第 5 个 block 评分；评分前缀只选取已预测的 latent，不改变模型前向形状。每条记录保留 GPU 使用/显存前后快照和系统 load average。",
            "",
            "| condition | batch=1 p50 / p95 (s) | batch=50 p50 / p95 (s) | batch=50 episodes/s | A/B forwards / backward per call | peak MiB | GPU |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    timing_by_condition = {
        (item["condition"]["mode"], item["condition"]["guidance"], item["batch_size"]): item
        for item in analysis["fixed_input_timing"]
    }
    for spec in _timing_specs():
        one = timing_by_condition[(spec["mode"], spec["guidance"], 1)]
        fifty = timing_by_condition[(spec["mode"], spec["guidance"], 50)]
        label = f"{spec['mode']}+{'none' if spec['guidance'] == 'none' else ('GF' if spec['guidance'] == 'guided_flow' else 'PO')}"
        counts = (
            f"{_fmt(fifty['stage_a_forwards_per_call'], 1)} / "
            f"{_fmt(fifty['stage_b_forwards_per_call'], 1)} / "
            f"{_fmt(fifty['guidance_backwards_per_call'], 1)}"
        )
        peak_mib = (
            None
            if fifty.get("peak_memory_bytes") is None
            else float(fifty["peak_memory_bytes"]) / (1024.0 * 1024.0)
        )
        lines.append(
            f"| {label} | {_fmt(one['p50_seconds'], 4)} / {_fmt(one['p95_seconds'], 4)} | "
            f"{_fmt(fifty['p50_seconds'], 4)} / {_fmt(fifty['p95_seconds'], 4)} | "
            f"{_fmt(fifty['throughput_per_second'], 2)} | {counts} | "
            f"{_fmt(peak_mib, 0)} | {fifty['gpu_after']['gpu']} |"
        )
    lines.extend(
        [
            "",
            "## 配对比较",
            "",
            "正的 Δ 表示候选条件成功更多。CI 对 50 个 episode 起点进行配对重采样；它不覆盖训练 seed、checkpoint 或 rollout RNG 的总体变异。",
            "",
            "| comparison | mode | guidance | execute | score | Δ success (pp) | paired improved / regressed | bootstrap 95% CI |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in analysis["comparisons"]:
        ci = item["bootstrap_95_ci_pp"]
        lines.append(
            f"| {item['comparison']} | {item['mode']} | {item['guidance']} | "
            f"{item['execute_steps']} | {item['score_steps_env']} | {item['delta_pp']:.1f} | "
            f"{item['improved']} / {item['regressed']} | [{ci[0]:.1f}, {ci[1]:.1f}] |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "更短的执行前缀会提高重规划频率，但本阶段仍为每次规划生成完整 25-step 候选，因此闭环收益和推理耗时变化不能解释为更短的模型预测 horizon。B 评分通过选择第 1、2 或 5 个预测 block 的 latent 实现，不在 block 内插值。GPU 并发可能影响墙钟耗时；规划 p50/p95 是本轮在线 rollout 内的实测值。",
            "",
            "## 产物",
            "",
            f"- 配置：`{analysis['config']}`",
            f"- 条件 CSV：`{analysis['output_root']}/analysis/conditions.csv`",
            f"- 比较 CSV：`{analysis['output_root']}/analysis/comparisons.csv`",
            f"- 固定观测计时 CSV：`{analysis['output_root']}/analysis/timing.csv`",
            f"- 分析 JSON：`{analysis['output_root']}/analysis/analysis.json`",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="verify inputs and historical baselines")
    validate_parser.add_argument("--config", default=str(DEFAULT_CONFIG))

    dry_parser = subparsers.add_parser("dry-run", help="print stored and logical condition matrices")
    dry_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    dry_parser.add_argument("--output-root")

    worker_parser = subparsers.add_parser("run-worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    worker_parser.add_argument("--output-root")
    worker_parser.add_argument("--gpu", required=True, type=_gpu)
    worker_parser.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB)
    worker_parser.add_argument("--indices", required=True)

    matrix_parser = subparsers.add_parser("run-matrix", help="run/resume the full inference matrix")
    matrix_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    matrix_parser.add_argument("--output-root")
    matrix_parser.add_argument("--gpus", default="7", help="comma-separated physical GPU IDs")
    matrix_parser.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB)

    timing_worker_parser = subparsers.add_parser("benchmark-worker", help=argparse.SUPPRESS)
    timing_worker_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    timing_worker_parser.add_argument("--output-root")
    timing_worker_parser.add_argument("--gpu", required=True, type=_gpu)
    timing_worker_parser.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB)
    timing_worker_parser.add_argument("--indices", required=True)

    timing_matrix_parser = subparsers.add_parser(
        "benchmark-matrix", help="run/resume fixed-input inference timing matrix"
    )
    timing_matrix_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    timing_matrix_parser.add_argument("--output-root")
    timing_matrix_parser.add_argument("--gpus", default="7", help="comma-separated physical GPU IDs")
    timing_matrix_parser.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB)

    analysis_parser = subparsers.add_parser("analyze", help="validate complete matrix and write report")
    analysis_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    analysis_parser.add_argument("--output-root")

    args = parser.parse_args()
    if args.command == "validate":
        validate(args)
    elif args.command == "dry-run":
        dry_run(args)
    elif args.command == "run-worker":
        run_worker(args)
    elif args.command == "run-matrix":
        run_matrix(args)
    elif args.command == "benchmark-worker":
        benchmark_worker(args)
    elif args.command == "benchmark-matrix":
        benchmark_matrix(args)
    elif args.command == "analyze":
        analyze(args)


if __name__ == "__main__":
    main()
