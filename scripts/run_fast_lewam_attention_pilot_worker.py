"""Run one attention pilot with identity-bound staged continuation.

The worker owns orchestration only. It never fabricates grounding metrics:
when the grounding producer is unavailable, it writes a pending schema marker
and leaves the epoch-3 decision at pending_grounding.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from source.experiments.fast_lewam_attention_pilots import (
    ATTENTION_VARIANTS,
    COHORT_SIZES,
    CANONICAL_TRAINING,
    Pilot,
    PilotIdentityError,
    PilotManifest,
    StaleArtifactError,
    atomic_write_json,
    artifact_identity,
    canonical_train_overrides,
    cohort_for_level,
    cohort_identity,
    decide_epoch3_promotion,
    expected_environment,
    file_sha256,
    load_pilot_manifest,
    path_identity,
    phase_identity,
    read_json,
    read_phase_marker,
    reference_cohorts,
    validate_grounding_slots,
    validate_manifest_inputs,
    write_phase_marker,
)


def _configure_isolated_gpu_environment(expected_gpu: int | None = None) -> int:
    """Require one permitted physical GPU and expose it as logical cuda:0."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"0", "1", "2", "3"}:
        raise RuntimeError("worker requires exactly one physical GPU from GPU0-GPU3")
    physical = int(visible)
    if expected_gpu is not None and physical != int(expected_gpu):
        raise RuntimeError(
            f"worker environment GPU{physical} does not match requested GPU{expected_gpu}"
        )
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(physical)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["EGL_DEVICE_ID"] = str(physical)
    return physical


def _stage_epochs(through_epoch: int) -> tuple[int, ...]:
    through_epoch = int(through_epoch)
    if through_epoch not in COHORT_SIZES:
        raise ValueError("through_epoch must be one of 1, 3, or 10")
    return tuple(epoch for epoch in (1, 3, 10) if epoch <= through_epoch)


def _candidate_root(pilot: Pilot) -> Path:
    return pilot.output_dir / "candidate"


def _stage_root(pilot: Pilot, epoch: int) -> Path:
    return _candidate_root(pilot) / f"epoch_{int(epoch)}"


def _full_checkpoint_for_epoch(pilot: Pilot, epoch: int) -> Path:
    return _stage_root(pilot, epoch) / "checkpoints" / "last.ckpt"


def _weight_checkpoint_for_epoch(pilot: Pilot, epoch: int) -> Path:
    return _stage_root(pilot, epoch) / "checkpoints" / f"fast_lewam_weights_epoch_{int(epoch)}.pt"


def _fast_dev_train_overrides() -> tuple[str, ...]:
    """Use a one-step constant scheduler only for fast-dev preflight."""
    return (
        "+trainer.fast_dev_run=true",
        "trainer.max_epochs=1",
        "+policy.scheduler.type=ConstantLR",
        "+policy.scheduler.factor=1.0",
        "+policy.scheduler.total_iters=1",
    )



def _build_train_command(
    manifest: PilotManifest,
    pilot: Pilot,
    *,
    epoch: int,
    run_dir: str | Path,
    parent_checkpoint: str | Path | None = None,
    python: str | None = None,
) -> tuple[str, ...]:
    """Build the canonical one-GPU train command for one continuation stage."""
    overrides = canonical_train_overrides(
        epoch=epoch,
        task=pilot.task,
        attention_variant=pilot.attention_variant,
        run_dir=run_dir,
        parent_checkpoint=parent_checkpoint,
    )
    return (
        python or sys.executable,
        "-u",
        "train.py",
        "--config-name=fast_lewam",
        *overrides,
    )


def _run(command: Sequence[str], *, cwd: Path, log_path: Path, env: Mapping[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("COMMAND " + " ".join(str(value) for value in command) + "\n")
        stream.flush()
        subprocess.run(
            list(command),
            cwd=cwd,
            env={**os.environ, **(dict(env) if env is not None else {})},
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _attention_implementation_ready(manifest: PilotManifest, pilot: Pilot) -> bool:
    """Return readiness declared by the model owner, never infer it from a tag."""
    implementations = manifest.raw.get("attention_implementations", {})
    value = implementations.get(pilot.attention_variant) if isinstance(implementations, Mapping) else None
    if isinstance(value, Mapping):
        return bool(value.get("ready", False))
    return bool(value is True)


def _write_readiness_marker(manifest: PilotManifest, pilot: Pilot, *, reason: str) -> Path:
    path = pilot.output_dir / "readiness.json"
    identity = {
        "schema_version": 1,
        "manifest_sha256": manifest.sha256,
        "pilot": pilot.name,
        "attention_variant": pilot.attention_variant,
        "canonical_training": CANONICAL_TRAINING,
    }
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "status": "pending_variant_implementation",
            "reason": reason,
            "identity": identity,
            "expected_environment": expected_environment(pilot.gpu),
            "no_metrics_written": True,
        },
    )
    return path


def _phase_marker_path(pilot: Pilot, epoch: int) -> Path:
    return pilot.output_dir / "phases" / f"epoch_{int(epoch)}.json"


def _load_completed_stage(
    manifest: PilotManifest,
    pilot: Pilot,
    *,
    epoch: int,
    parent_checkpoint: Path | None,
) -> bool:
    cohort = cohort_for_level(manifest, pilot, epoch)
    identity = phase_identity(
        manifest,
        pilot,
        epoch=epoch,
        phase="train",
        parent_checkpoint=parent_checkpoint,
        cohort=cohort,
    )
    marker = _phase_marker_path(pilot, epoch)
    if not marker.is_file():
        return False
    read_phase_marker(marker, identity)
    return True


def _assert_stage_retriable(stage_root: Path) -> None:
    if stage_root.exists() and any(stage_root.iterdir()):
        raise RuntimeError(
            f"incomplete stage output exists; archive it explicitly before retry: {stage_root}"
        )


def _stage_artifacts(pilot: Pilot, epoch: int) -> tuple[Path, ...]:
    return (
        _stage_root(pilot, epoch) / "config.yaml",
        _full_checkpoint_for_epoch(pilot, epoch),
        _weight_checkpoint_for_epoch(pilot, epoch),
    )


def _validate_stage_artifacts(pilot: Pilot, epoch: int) -> dict[str, Any]:
    paths = _stage_artifacts(pilot, epoch)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"training epoch {epoch} did not produce full checkpoint artifacts: {missing}"
        )
    return artifact_identity(paths)


def _train_stage(
    manifest: PilotManifest,
    pilot: Pilot,
    *,
    epoch: int,
    parent_checkpoint: Path | None,
    device: str,
) -> Path:
    if device != "cuda:0":
        raise RuntimeError("attention pilot worker must use logical device cuda:0")
    stage_root = _stage_root(pilot, epoch)
    cohort = cohort_for_level(manifest, pilot, epoch)
    identity = phase_identity(
        manifest,
        pilot,
        epoch=epoch,
        phase="train",
        parent_checkpoint=parent_checkpoint,
        cohort=cohort,
    )
    marker = _phase_marker_path(pilot, epoch)
    if marker.is_file():
        read_phase_marker(marker, identity)
        return _full_checkpoint_for_epoch(pilot, epoch)
    stage_identity_path = stage_root / "stage_identity.json"
    resume_checkpoint = None
    if stage_root.exists() and any(stage_root.iterdir()):
        if not stage_identity_path.is_file():
            raise RuntimeError(
                f"incomplete stage has no identity marker; archive it explicitly: {stage_root}"
            )
        stage_state = read_json(stage_identity_path)
        if stage_state.get("identity") != identity:
            raise StaleArtifactError(f"stage identity mismatch: {stage_identity_path}")
        resume_checkpoint = _full_checkpoint_for_epoch(pilot, epoch)
        if not resume_checkpoint.is_file():
            raise RuntimeError(
                f"stage crashed before a full checkpoint was written; archive it explicitly: {stage_root}"
            )
    stage_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        stage_identity_path,
        {
            "schema_version": 1,
            "status": "running",
            "identity": identity,
            "parent_checkpoint": path_identity(parent_checkpoint) if parent_checkpoint else None,
        },
    )
    command = _build_train_command(
        manifest,
        pilot,
        epoch=epoch,
        run_dir=stage_root,
        parent_checkpoint=resume_checkpoint or parent_checkpoint,
    )
    environment = expected_environment(pilot.gpu)
    environment["FAST_LEWAM_ATTENTION_VARIANT"] = pilot.attention_variant
    _run(command, cwd=manifest.repository_root, log_path=stage_root / "train.log", env=environment)
    artifacts = _validate_stage_artifacts(pilot, epoch)
    write_phase_marker(
        marker,
        identity,
        artifacts,
        status="complete",
        command=list(command),
        device=device,
        attention_variant=pilot.attention_variant,
    )
    atomic_write_json(
        stage_identity_path,
        {"schema_version": 1, "status": "complete", "identity": identity, "command": list(command)},
    )
    return _full_checkpoint_for_epoch(pilot, epoch)


def _eval_leaf(pilot: Pilot, epoch: int) -> Path:
    return pilot.output_dir / "eval" / f"epoch_{int(epoch)}" / "stage_b"


def _baseline_eval_leaf(pilot: Pilot, epoch: int) -> Path:
    return pilot.output_dir / "baseline_eval" / f"epoch_{int(epoch)}" / "stage_b"


def _result_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = read_json(path)
    if value.get("status", "ok") != "ok":
        raise PilotIdentityError(f"evaluation is not successful: {path}")
    if not isinstance(value.get("episodes"), list):
        raise PilotIdentityError(f"evaluation has no episode provenance: {path}")
    return value


def _validate_result_cohort(result: Mapping[str, Any], expected) -> None:
    records = result.get("episodes", ())
    actual = tuple((row.get("dataset_episode"), row.get("start_step")) for row in records)
    wanted = tuple(
        (episode, int(step))
        for episode, step in zip(expected.episode_ids.tolist(), expected.start_steps.tolist())
    )
    if actual != wanted:
        raise PilotIdentityError("evaluation result does not use the exact requested cohort")


def _evaluation_cache_identity(
    manifest: PilotManifest,
    pilot: Pilot,
    *,
    epoch: int,
    checkpoint: Path,
    run_config: Path,
    cohort,
) -> dict[str, Any]:
    """Bind an evaluation result to every input that can change its meaning."""
    checkpoint_identity = path_identity(checkpoint)
    config_identity = path_identity(run_config)
    for label, value in (("checkpoint", checkpoint_identity), ("run_config", config_identity)):
        if not value.get("exists") or value.get("is_dir"):
            raise FileNotFoundError(f"{label} is missing or not a file: {value.get('path')}")
    return {
        "schema_version": 1,
        "manifest_sha256": manifest.sha256,
        "pilot": pilot.name,
        "epoch": int(epoch),
        "checkpoint": checkpoint_identity,
        "run_config": config_identity,
        "cohort_sha256": cohort_identity(cohort),
    }


def _evaluation_identity_path(output_leaf: Path) -> Path:
    return output_leaf / "identity.json"


def _validate_evaluation_cache(
    result_path: Path,
    identity_path: Path,
    expected_identity: Mapping[str, Any],
) -> None:
    """Reject a result unless its sidecar and result bytes still match."""
    if not identity_path.is_file():
        raise StaleArtifactError(f"evaluation result has no identity sidecar: {identity_path}")
    try:
        sidecar = read_json(identity_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise StaleArtifactError(f"invalid evaluation identity sidecar: {identity_path}") from exc
    if sidecar.get("identity") != dict(expected_identity):
        raise StaleArtifactError(f"evaluation identity mismatch: {identity_path}")
    expected_result_sha256 = sidecar.get("result_sha256")
    if not isinstance(expected_result_sha256, str):
        raise StaleArtifactError(f"evaluation sidecar lacks result hash: {identity_path}")
    try:
        actual_result_sha256 = file_sha256(result_path)
    except OSError as exc:
        raise StaleArtifactError(f"evaluation result disappeared: {result_path}") from exc
    if actual_result_sha256 != expected_result_sha256:
        raise StaleArtifactError(f"evaluation result bytes changed: {result_path}")


def _evaluate_checkpoint(
    manifest: PilotManifest,
    pilot: Pilot,
    *,
    epoch: int,
    checkpoint: Path,
    run_config: Path,
    output_leaf: Path,
    device: str,
    cohort=None,
) -> Path:
    """Evaluate with DatasetEvaluationSession(cohort=...) in a worker process."""
    expected = cohort if cohort is not None else cohort_for_level(manifest, pilot, epoch)
    expected_identity = _evaluation_cache_identity(
        manifest,
        pilot,
        epoch=epoch,
        checkpoint=checkpoint,
        run_config=run_config,
        cohort=expected,
    )
    result_path = output_leaf / "result.json"
    identity_path = _evaluation_identity_path(output_leaf)
    if result_path.is_file():
        _validate_evaluation_cache(result_path, identity_path, expected_identity)
        _validate_result_cohort(_result_payload(result_path), expected)
        return result_path
    if identity_path.exists():
        raise StaleArtifactError(
            f"evaluation identity exists without result; refusing reuse: {identity_path}"
        )
    output_leaf.mkdir(parents=True, exist_ok=True)
    # Imports stay inside the execution path so dry-run/status only inspect
    # manifests and never initialize CUDA or MuJoCo.
    import torch
    from omegaconf import OmegaConf
    from eval_fast_lewam import load_model_from_weights
    from source.common.eval import DatasetEvaluationSession, EvaluationIdentity, compose_eval_config

    cfg = compose_eval_config(
        pilot.task,
        overrides=(
            f"eval.num_eval={len(expected.row_indices)}",
            f"world.num_envs={len(expected.row_indices)}",
            "output.save_video=false",
            f"solver.device={device}",
            "seed=42",
        ),
    )
    session = DatasetEvaluationSession(
        cfg,
        task=pilot.task,
        cohort=expected,
    )
    run_cfg = OmegaConf.load(run_config)
    model = load_model_from_weights(run_cfg, checkpoint, device)
    identity = EvaluationIdentity(
        entrypoint="fast_lewam_attention_pilot",
        policy_kind="fast_lewam_attention",
        checkpoint=str(checkpoint),
        epoch=int(epoch),
        stage="stage_b",
    )
    try:
        session.evaluate(model, identity=identity, output_dir=output_leaf, device=device)
    finally:
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    _validate_result_cohort(_result_payload(result_path), expected)
    atomic_write_json(
        identity_path,
        {
            "schema_version": 1,
            "identity": expected_identity,
            "result_sha256": file_sha256(result_path),
        },
    )
    return result_path


def _baseline_checkpoint(pilot: Pilot, epoch: int) -> Path:
    try:
        return pilot.baseline_checkpoints[int(epoch)]
    except KeyError as exc:
        raise FileNotFoundError(f"baseline checkpoint for epoch {epoch} is not declared") from exc


def _write_grounding_readiness(
    manifest: PilotManifest,
    pilot: Pilot,
    *,
    epoch: int,
    candidate_result: Path,
    baseline_result: Path,
) -> Path:
    slots = validate_grounding_slots(pilot.task, epoch, pilot.grounding_slots[epoch])
    path = pilot.output_dir / "grounding" / f"epoch_{int(epoch)}.json"
    cohort = cohort_for_level(manifest, pilot, epoch)
    identity = phase_identity(
        manifest,
        pilot,
        epoch=epoch,
        phase="grounding",
        cohort=cohort,
    )
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "status": "pending_grounding",
            "identity": identity,
            "task": pilot.task,
            "epoch": int(epoch),
            "slots": list(slots),
            "cohort_sha256": identity["cohort_sha256"],
            "baseline_result": str(baseline_result),
            "candidate_result": str(candidate_result),
            "required_metrics": [
                "predicted_physical_spearman",
                "physical_best_regret",
            ],
            "no_synthetic_metrics": True,
        },
    )
    return path


def _grounding_input_paths(
    pilot: Pilot,
    *,
    epoch: int,
    baseline_result: Path | None = None,
    candidate_result: Path | None = None,
    candidate_config: Path | None = None,
    candidate_checkpoint: Path | None = None,
) -> dict[str, Path]:
    """Resolve the exact all-50 artifacts consumed by paired grounding."""
    epoch = int(epoch)
    if epoch not in (3, 10):
        raise ValueError("grounding inputs are defined for epoch 3 and epoch 10")
    if epoch == 3:
        default_baseline_result = (
            pilot.output_dir / "baseline_eval" / "epoch_3_grounding_50"
            / "stage_b" / "result.json"
        )
        default_candidate_result = (
            pilot.output_dir / "eval" / "epoch_3_grounding_50"
            / "stage_b" / "result.json"
        )
    else:
        default_baseline_result = _baseline_eval_leaf(pilot, 10) / "result.json"
        default_candidate_result = _eval_leaf(pilot, 10) / "result.json"
    return {
        "baseline_config": pilot.baseline_config,
        "baseline_checkpoint": _baseline_checkpoint(pilot, epoch),
        "baseline_result": baseline_result or default_baseline_result,
        "candidate_config": candidate_config or (_stage_root(pilot, epoch) / "config.yaml"),
        "candidate_checkpoint": candidate_checkpoint or _weight_checkpoint_for_epoch(pilot, epoch),
        "candidate_result": candidate_result or default_candidate_result,
    }


def _diagnostic_manifest_path(pilot: Pilot, epoch: int) -> Path:
    return pilot.output_dir / "grounding" / f"diagnostic_manifest_epoch_{int(epoch)}.yaml"


def _diagnostic_pair_label(pilot: Pilot, epoch: int) -> str:
    return f"{pilot.name}_epoch_{int(epoch)}"


def _diagnostic_summary_path(pilot: Pilot, epoch: int) -> Path:
    return (
        pilot.output_dir / "grounding" / "diagnostic"
        / _diagnostic_pair_label(pilot, epoch) / "summary.json"
    )


def _grounding_cache_identity(
    manifest: PilotManifest,
    pilot: Pilot,
    *,
    epoch: int,
    baseline_result: Path | None = None,
    candidate_result: Path | None = None,
    candidate_config: Path | None = None,
    candidate_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Bind wrapper grounding metrics to diagnostic and model artifacts."""
    inputs = _grounding_input_paths(
        pilot,
        epoch=epoch,
        baseline_result=baseline_result,
        candidate_result=candidate_result,
        candidate_config=candidate_config,
        candidate_checkpoint=candidate_checkpoint,
    )
    identity = phase_identity(
        manifest,
        pilot,
        epoch=epoch,
        phase="grounding",
        cohort=cohort_for_level(manifest, pilot, epoch),
    )
    identity["grounding_cohort_sha256"] = cohort_identity(
        cohort_for_level(manifest, pilot, 10)
    )
    identity["input_artifacts"] = {
        name: path_identity(path) for name, path in inputs.items()
    }
    manifest_path = _diagnostic_manifest_path(pilot, epoch)
    summary_path = _diagnostic_summary_path(pilot, epoch)
    identity["diagnostic_manifest_artifact"] = path_identity(manifest_path)
    identity["diagnostic_summary_artifact"] = path_identity(summary_path)
    identity["diagnostic_pair"] = _diagnostic_pair_label(pilot, epoch)
    return identity


def _validate_diagnostic_summary(
    summary_path: Path,
    manifest_path: Path,
    pair_label: str,
    *,
    epoch: int,
    inputs: Mapping[str, Path],
) -> dict[str, Any]:
    """Validate the diagnostic summary and its nested epoch input hashes."""
    if not summary_path.is_file():
        raise StaleArtifactError(f"diagnostic summary is missing: {summary_path}")
    try:
        summary = read_json(summary_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise StaleArtifactError(f"invalid diagnostic summary: {summary_path}") from exc
    if summary.get("status") != "ok":
        raise PilotIdentityError(f"grounding diagnostic is incomplete: {summary_path}")
    identity = summary.get("identity")
    if not isinstance(identity, Mapping):
        raise StaleArtifactError(f"diagnostic summary has no identity: {summary_path}")
    if identity.get("manifest_sha256") != file_sha256(manifest_path):
        raise StaleArtifactError(f"diagnostic manifest changed: {manifest_path}")
    if identity.get("pair") != pair_label or identity.get("phase") != "summarize":
        raise StaleArtifactError(f"diagnostic summary pair/phase mismatch: {summary_path}")
    expected_epochs = {
        "e8": {
            "epoch": int(epoch),
            "config_sha256": file_sha256(inputs["baseline_config"]),
            "checkpoint_sha256": file_sha256(inputs["baseline_checkpoint"]),
            "reference_result_sha256": file_sha256(inputs["baseline_result"]),
        },
        "e10": {
            "epoch": int(epoch),
            "config_sha256": file_sha256(inputs["candidate_config"]),
            "checkpoint_sha256": file_sha256(inputs["candidate_checkpoint"]),
            "reference_result_sha256": file_sha256(inputs["candidate_result"]),
        },
    }
    epochs = identity.get("epochs")
    if not isinstance(epochs, Mapping):
        raise StaleArtifactError(f"diagnostic summary has no identity.epochs: {summary_path}")
    for label, expected in expected_epochs.items():
        actual = epochs.get(label)
        if not isinstance(actual, Mapping):
            raise StaleArtifactError(f"diagnostic summary lacks identity.epochs.{label}: {summary_path}")
        for key, wanted in expected.items():
            if actual.get(key) != wanted:
                raise StaleArtifactError(
                    f"diagnostic summary identity.epochs.{label}.{key} is stale: {summary_path}"
                )
    if identity.get("config_sha256") != file_sha256(inputs["baseline_config"]):
        raise StaleArtifactError(f"diagnostic summary root config is stale: {summary_path}")
    return summary


def _diagnostic_manifest(
    manifest: PilotManifest,
    pilot: Pilot,
    *,
    epoch: int,
    baseline_result: Path,
    candidate_result: Path,
    candidate_config: Path,
    candidate_checkpoint: Path,
) -> tuple[Path, dict[str, Any]]:
    """Create the pair manifest consumed by the existing trace/ground code."""
    if int(epoch) not in (3, 10):
        raise ValueError("paired grounding is defined for epoch 3 and epoch 10")
    baseline_payload = _result_payload(baseline_result)
    candidate_payload = _result_payload(candidate_result)
    if len(baseline_payload["episodes"]) != 50 or len(candidate_payload["episodes"]) != 50:
        raise PilotIdentityError("grounding reference evaluations must contain all 50 cohort rows")
    baseline_success = [bool(row["success"]) for row in baseline_payload["episodes"]]
    candidate_success = [bool(row["success"]) for row in candidate_payload["episodes"]]
    if len(baseline_success) != len(candidate_success):
        raise PilotIdentityError("grounding baseline/candidate success vectors differ")
    selected = validate_grounding_slots(pilot.task, epoch, pilot.grounding_slots[epoch])
    categories = {
        "regression": [],
        "improvement_control": [],
        "stable_success": [],
        "stable_failure": [],
    }
    for slot in selected:
        if baseline_success[slot] and not candidate_success[slot]:
            category = "regression"
        elif not baseline_success[slot] and candidate_success[slot]:
            category = "improvement_control"
        elif baseline_success[slot]:
            category = "stable_success"
        else:
            category = "stable_failure"
        categories[category].append(int(slot))
    baseline_parameters = baseline_payload["parameters"]
    protocol = {
        "seed": int(baseline_parameters["seed"]),
        "num_eval": 50,
        "goal_offset_steps": int(baseline_parameters["goal_offset_steps"]),
        "eval_budget": int(baseline_parameters["eval_budget"]),
        "horizon": int(baseline_parameters["plan_config"]["horizon"]),
        "receding_horizon": int(baseline_parameters["plan_config"]["receding_horizon"]),
        "action_block": int(baseline_parameters["plan_config"]["action_block"]),
        "num_samples": int(baseline_parameters["solver"]["num_samples"]),
        "n_steps": int(baseline_parameters["solver"]["n_steps"]),
        "topk": int(baseline_parameters["solver"]["topk"]),
        "detailed_iterations": [0, 5, 29],
        "fixed_non_elites": 10,
    }
    pair_label = f"{pilot.name}_epoch_{int(epoch)}"
    output_root = pilot.output_dir / "grounding" / "diagnostic"
    raw = {
        "version": 1,
        "repository_root": str(manifest.repository_root),
        "output_root": str(output_root),
        "protocol": protocol,
        "pairs": {
            pair_label: {
                "task": pilot.task,
                "section": "attention_pilot",
                "preferred_gpu": int(pilot.gpu),
                "run_config": str(pilot.baseline_config),
                "epochs": {
                    "e8": {
                        "epoch": int(epoch),
                        "run_config": str(pilot.baseline_config),
                        "checkpoint": str(_baseline_checkpoint(pilot, epoch)),
                        "reference_result": str(baseline_result),
                    },
                    "e10": {
                        "epoch": int(epoch),
                        "run_config": str(candidate_config),
                        "checkpoint": str(candidate_checkpoint),
                        "reference_result": str(candidate_result),
                    },
                },
                "slots": categories,
            }
        },
    }
    identity = phase_identity(
        manifest,
        pilot,
        epoch=epoch,
        phase="grounding",
        cohort=cohort_for_level(manifest, pilot, epoch),
    )
    inputs = _grounding_input_paths(
        pilot,
        epoch=epoch,
        baseline_result=baseline_result,
        candidate_result=candidate_result,
        candidate_config=candidate_config,
        candidate_checkpoint=candidate_checkpoint,
    )
    identity["grounding_cohort_sha256"] = cohort_identity(
        cohort_for_level(manifest, pilot, 10)
    )
    identity["input_artifacts"] = {
        name: path_identity(path) for name, path in inputs.items()
    }
    raw["identity"] = identity
    raw["grounding_cohort_sha256"] = identity["grounding_cohort_sha256"]
    raw["input_artifacts"] = identity["input_artifacts"]
    path = _diagnostic_manifest_path(pilot, epoch)
    atomic_write_json(path, raw)
    return path, raw


def _run_grounding_diagnostic(
    manifest: PilotManifest,
    pilot: Pilot,
    *,
    epoch: int,
    baseline_result: Path,
    candidate_result: Path,
    candidate_config: Path,
    candidate_checkpoint: Path,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run and validate real simulator grounding for the fixed pilot slots."""
    if int(epoch) not in (3, 10):
        raise ValueError("grounding diagnostic is defined for epoch 3 and epoch 10")
    all_fifty = cohort_for_level(manifest, pilot, 10)
    if int(epoch) == 3:
        # The level-3 gate evaluates 20 rows. The existing paired diagnostic
        # requires the strict 50-row protocol, so produce one separately
        # identity-bound all-50 result for its six selected slots.
        baseline_50 = pilot.output_dir / "baseline_eval" / "epoch_3_grounding_50" / "stage_b"
        candidate_50 = pilot.output_dir / "eval" / "epoch_3_grounding_50" / "stage_b"
        baseline_50 = _evaluate_checkpoint(
            manifest, pilot, epoch=epoch,
            checkpoint=_baseline_checkpoint(pilot, epoch),
            run_config=pilot.baseline_config,
            output_leaf=baseline_50,
            device=device,
            cohort=all_fifty,
        )
        candidate_50 = _evaluate_checkpoint(
            manifest, pilot, epoch=epoch,
            checkpoint=candidate_checkpoint,
            run_config=candidate_config,
            output_leaf=candidate_50,
            device=device,
            cohort=all_fifty,
        )
    else:
        # Epoch-10 evaluation already uses all 50 rows; do not duplicate it.
        baseline_50 = baseline_result
        candidate_50 = candidate_result
    manifest_path, diagnostic_raw = _diagnostic_manifest(
        manifest, pilot, epoch=epoch,
        baseline_result=baseline_50,
        candidate_result=candidate_50,
        candidate_config=candidate_config,
        candidate_checkpoint=candidate_checkpoint,
    )
    pair_label = next(iter(diagnostic_raw["pairs"]))
    summary_path = _diagnostic_summary_path(pilot, epoch)
    # Always invoke the diagnostic entrypoint: its own identity-bound caches
    # either reuse valid trace/ground artifacts or reject stale ones.
    command = (
        sys.executable, "-u",
        "scripts/diagnose_fast_lewam_epoch_pair_ranking.py",
        "--manifest", str(manifest_path),
        "--pair", pair_label,
        "--phase", "all",
        "--device", device,
    )
    environment = expected_environment(pilot.gpu)
    _run(
        command,
        cwd=manifest.repository_root,
        log_path=pilot.output_dir / "grounding" / "diagnostic.log",
        env=environment,
    )
    diagnostic_inputs = _grounding_input_paths(
        pilot,
        epoch=epoch,
        baseline_result=baseline_50,
        candidate_result=candidate_50,
        candidate_config=candidate_config,
        candidate_checkpoint=candidate_checkpoint,
    )
    summary = _validate_diagnostic_summary(
        summary_path,
        manifest_path,
        pair_label,
        epoch=epoch,
        inputs=diagnostic_inputs,
    )
    overall = summary.get("overall", {})
    metrics = overall.get("metrics", {})
    baseline_metrics = metrics.get("e8", {})
    candidate_metrics = metrics.get("e10", {})
    required = ("predicted_physical_spearman", "physical_best_candidate_regret")
    if any(key not in baseline_metrics or key not in candidate_metrics for key in required):
        raise PilotIdentityError("grounding summary lacks promotion metrics")
    ground_identity = _grounding_cache_identity(
        manifest,
        pilot,
        epoch=epoch,
        baseline_result=baseline_50,
        candidate_result=candidate_50,
        candidate_config=candidate_config,
        candidate_checkpoint=candidate_checkpoint,
    )
    path = pilot.output_dir / "grounding" / f"epoch_{int(epoch)}_metrics.json"
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "status": "ok",
            "epoch": int(epoch),
            "identity": ground_identity,
            "diagnostic_manifest": str(manifest_path),
            "diagnostic_summary": str(summary_path),
            "baseline": {
                "predicted_physical_spearman": baseline_metrics["predicted_physical_spearman"],
                "physical_best_regret": baseline_metrics["physical_best_candidate_regret"],
            },
            "candidate": {
                "predicted_physical_spearman": candidate_metrics["predicted_physical_spearman"],
                "physical_best_regret": candidate_metrics["physical_best_candidate_regret"],
            },
        },
    )
    return (
        {
            "predicted_physical_spearman": baseline_metrics["predicted_physical_spearman"],
            "physical_best_regret": baseline_metrics["physical_best_candidate_regret"],
        },
        {
            "predicted_physical_spearman": candidate_metrics["predicted_physical_spearman"],
            "physical_best_regret": candidate_metrics["physical_best_candidate_regret"],
        },
    )


def _load_grounding_metrics(
    manifest: PilotManifest,
    pilot: Pilot,
    *,
    epoch: int,
    baseline_result: Path | None = None,
    candidate_result: Path | None = None,
    candidate_config: Path | None = None,
    candidate_checkpoint: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    path = pilot.output_dir / "grounding" / f"epoch_{int(epoch)}_metrics.json"
    if not path.is_file():
        return None
    inputs = _grounding_input_paths(
        pilot,
        epoch=epoch,
        baseline_result=baseline_result,
        candidate_result=candidate_result,
        candidate_config=candidate_config,
        candidate_checkpoint=candidate_checkpoint,
    )
    manifest_path = _diagnostic_manifest_path(pilot, epoch)
    summary_path = _diagnostic_summary_path(pilot, epoch)
    pair_label = _diagnostic_pair_label(pilot, epoch)
    _validate_diagnostic_summary(
        summary_path,
        manifest_path,
        pair_label,
        epoch=epoch,
        inputs=inputs,
    )
    value = read_json(path)
    expected = _grounding_cache_identity(
        manifest,
        pilot,
        epoch=epoch,
        baseline_result=inputs["baseline_result"],
        candidate_result=inputs["candidate_result"],
        candidate_config=inputs["candidate_config"],
        candidate_checkpoint=inputs["candidate_checkpoint"],
    )
    if value.get("identity") != expected:
        raise StaleArtifactError(f"grounding metric identity mismatch: {path}")
    baseline = value.get("baseline")
    candidate = value.get("candidate")
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        raise PilotIdentityError("grounding metrics must contain baseline and candidate objects")
    for metrics in (baseline, candidate):
        if any(key not in metrics for key in ("predicted_physical_spearman", "physical_best_regret")):
            raise PilotIdentityError("grounding metrics are incomplete")
    return dict(baseline), dict(candidate)


def _summarize_epoch(
    manifest: PilotManifest,
    pilot: Pilot,
    *,
    epoch: int,
    baseline_result: Path,
    candidate_result: Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    baseline = _result_payload(baseline_result)
    candidate = _result_payload(candidate_result)
    expected = cohort_for_level(manifest, pilot, epoch)
    _validate_result_cohort(baseline, expected)
    _validate_result_cohort(candidate, expected)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "ok",
        "manifest_sha256": manifest.sha256,
        "pilot": pilot.name,
        "task": pilot.task,
        "attention_variant": pilot.attention_variant,
        "epoch": int(epoch),
        "cohort_sha256": phase_identity(
            manifest, pilot, epoch=epoch, phase="summary", cohort=expected
        )["cohort_sha256"],
        "baseline_result": str(baseline_result),
        "candidate_result": str(candidate_result),
    }
    if epoch in (3, 10):
        candidate_config = _stage_root(pilot, epoch) / "config.yaml"
        candidate_checkpoint = _weight_checkpoint_for_epoch(pilot, epoch)
        grounding = _load_grounding_metrics(
            manifest,
            pilot,
            epoch=epoch,
            baseline_result=baseline_result,
            candidate_result=candidate_result,
            candidate_config=candidate_config,
            candidate_checkpoint=candidate_checkpoint,
        )
        if grounding is None:
            grounding = _run_grounding_diagnostic(
                manifest,
                pilot,
                epoch=epoch,
                baseline_result=baseline_result,
                candidate_result=candidate_result,
                candidate_config=candidate_config,
                candidate_checkpoint=candidate_checkpoint,
                device=device,
            )
        base_ground, candidate_ground = grounding
        from source.experiments.fast_lewam_attention_pilots import compute_promotion_metrics
        metrics = compute_promotion_metrics(
            baseline, candidate,
            baseline_grounding=base_ground,
            candidate_grounding=candidate_ground,
        )
        summary.update({
            "decision": (
                decide_epoch3_promotion(
                    baseline, candidate,
                    baseline_grounding=base_ground,
                    candidate_grounding=candidate_ground,
                    thresholds=manifest.promotion,
                )
                if epoch == 3 else "complete"
            ),
            "metrics": metrics,
            "grounding_status": "complete",
        })
    elif epoch == 1:
        summary.update({"decision": "continue", "metrics": None})
    else:
        summary.update({"decision": "complete", "metrics": None})
    path = pilot.output_dir / "summaries" / f"epoch_{int(epoch)}.json"
    atomic_write_json(path, summary)
    return summary


def run_pilot(
    manifest: PilotManifest,
    pilot: Pilot,
    *,
    through_epoch: int,
    device: str = "cuda:0",
    fast_dev_run: bool = False,
) -> dict[str, Any]:
    physical = _configure_isolated_gpu_environment(pilot.gpu)
    if device != "cuda:0":
        raise RuntimeError("only logical cuda:0 is permitted")
    if physical != pilot.gpu:
        raise RuntimeError("worker physical GPU does not match manifest mapping")
    validate_grounding_slots(pilot.task, 3, pilot.grounding_slots[3])
    validate_grounding_slots(pilot.task, 10, pilot.grounding_slots[10])
    validate_manifest_inputs(manifest)
    if fast_dev_run:
        preflight_root = pilot.output_dir / "preflight" / "fast_dev_run"
        command = _build_train_command(
            manifest,
            pilot,
            epoch=1,
            run_dir=preflight_root,
            python=sys.executable,
        ) + _fast_dev_train_overrides()
        environment = expected_environment(pilot.gpu)
        environment["FAST_LEWAM_ATTENTION_VARIANT"] = pilot.attention_variant
        _run(
            command,
            cwd=manifest.repository_root,
            log_path=pilot.output_dir / "preflight" / "fast_dev_run.log",
            env=environment,
        )
        return {
            "status": "fast_dev_complete",
            "pilot": pilot.name,
            "gpu": pilot.gpu,
            "command": list(command),
        }
    if not _attention_implementation_ready(manifest, pilot):
        path = _write_readiness_marker(
            manifest,
            pilot,
            reason=(
                f"{pilot.attention_variant} is not declared ready by the model "
                "implementation; no training or metrics are started"
            ),
        )
        return {"status": "pending_variant_implementation", "readiness": str(path)}
    previous_checkpoint = None
    for epoch in _stage_epochs(through_epoch):
        checkpoint = _train_stage(
            manifest,
            pilot,
            epoch=epoch,
            parent_checkpoint=previous_checkpoint,
            device=device,
        )
        # Baseline evaluation is also run through the explicit cohort path;
        # the existing strict e10 artifact is only the cohort reference.
        baseline = _evaluate_checkpoint(
            manifest, pilot,
            epoch=epoch,
            checkpoint=_baseline_checkpoint(pilot, epoch),
            run_config=pilot.baseline_config,
            output_leaf=_baseline_eval_leaf(pilot, epoch),
            device=device,
        )
        candidate = _evaluate_checkpoint(
            manifest, pilot,
            epoch=epoch,
            checkpoint=_weight_checkpoint_for_epoch(pilot, epoch),
            run_config=_stage_root(pilot, epoch) / "config.yaml",
            output_leaf=_eval_leaf(pilot, epoch),
            device=device,
        )
        summary = _summarize_epoch(
            manifest,
            pilot,
            epoch=epoch,
            baseline_result=baseline,
            candidate_result=candidate,
            device=device,
        )
        if epoch == 3 and summary.get("decision") != "promote":
            # pending_grounding and stop both fail closed; do not continue to
            # epoch 10 before a real grounded result has passed the gate.
            return summary
        previous_checkpoint = checkpoint
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pilot", required=True)
    parser.add_argument("--through-epoch", "--epoch", "--level", type=int, choices=(1, 3, 10), default=10)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_pilot_manifest(args.manifest)
    if args.pilot not in manifest.pilots:
        raise ValueError(f"unknown pilot {args.pilot}")
    pilot = manifest.pilots[args.pilot]
    if int(args.physical_gpu) != pilot.gpu:
        raise RuntimeError("physical GPU argument does not match manifest")
    try:
        result = run_pilot(
            manifest,
            pilot,
            through_epoch=args.through_epoch,
            device=args.device,
            fast_dev_run=args.fast_dev_run,
        )
    except Exception as exc:
        error_path = pilot.output_dir / "worker_failure.json"
        atomic_write_json(
            error_path,
            {
                "schema_version": 1,
                "status": "failed",
                "manifest_sha256": manifest.sha256,
                "pilot": pilot.name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
