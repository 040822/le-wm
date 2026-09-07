"""Exact-replay Stage-B failures and collect side-effect-free CEM traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if __name__ == "__main__":
    from source.common.gpu_environment import configure_mujoco_egl_device

    configure_mujoco_egl_device()

from omegaconf import OmegaConf

from eval_fast_lewam import load_model_from_weights
from source.common.checkpoint import load_policy_or_model
from source.common.eval import (
    DatasetEvaluationSession,
    EvaluationIdentity,
    compose_eval_config,
)
from source.diagnostics.stage_b_episode_failures import (
    CEMTraceCollector,
    TracedCEMSolver,
    TracedPolicy,
    load_diagnostic_manifest,
    select_diagnostic_slots,
    write_trace_artifacts,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_device(device: str) -> None:
    if device == "cpu":
        return
    if device != "cuda:0":
        raise ValueError(
            "GPU diagnostics must use internal device 'cuda:0'; select the "
            "physical GPU with CUDA_VISIBLE_DEVICES"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        raise EnvironmentError(
            "CUDA_VISIBLE_DEVICES must explicitly select only GPU0-3"
        )
    try:
        selected = [int(token.strip()) for token in visible.split(",")]
    except ValueError as exc:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must be a comma-separated list of GPU IDs"
        ) from exc
    if not selected or any(device_id not in {0, 1, 2, 3} for device_id in selected):
        raise ValueError(
            f"prohibited CUDA_VISIBLE_DEVICES={visible!r}; only GPU0-3 are allowed"
        )


def _load_subject(run, device: str):
    if run.model_kind == "fast_lewam":
        run_cfg = OmegaConf.load(run.run_config)
        return load_model_from_weights(run_cfg, run.checkpoint, device)
    if run.model_kind == "lewm":
        subject, resolved = load_policy_or_model(str(run.checkpoint))
        if resolved is None or resolved.resolve() != run.checkpoint:
            raise RuntimeError(
                f"resolved checkpoint {resolved} differs from manifest "
                f"{run.checkpoint}"
            )
        return subject
    raise ValueError(f"unsupported model_kind={run.model_kind!r}")


def run_trace(*, manifest_path, run_label, device, output_dir):
    _validate_device(str(device))
    manifest = load_diagnostic_manifest(manifest_path)
    if run_label not in manifest.runs:
        raise KeyError(
            f"unknown run label {run_label!r}; choose one of "
            f"{sorted(manifest.runs)}"
        )
    run = manifest.runs[run_label]
    protocol = manifest.protocol
    output_dir = Path(output_dir).expanduser().resolve() / run_label
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = json.loads(run.reference_result.read_text(encoding="utf-8"))
    expected = [bool(episode["success"]) for episode in reference["episodes"]]
    selection_vectors = {}
    for candidate in manifest.runs.values():
        if candidate.task != run.task or candidate.family not in {
            "e0",
            "e1_384",
            "e3_384",
        }:
            continue
        candidate_result = json.loads(
            candidate.reference_result.read_text(encoding="utf-8")
        )
        selection_vectors[candidate.family] = [
            bool(episode["success"])
            for episode in candidate_result["episodes"]
        ]
    slot_selection = select_diagnostic_slots(selection_vectors)

    cfg = compose_eval_config(
        run.task,
        (
            f"solver.device={device}",
            "output.save_video=false",
        ),
    )
    session = DatasetEvaluationSession(cfg, task=run.task)
    subject = _load_subject(run, str(device))
    identity = EvaluationIdentity(
        entrypoint="diagnose_stage_b_episode_failures",
        policy_kind=run.model_kind,
        checkpoint=str(run.checkpoint),
        epoch=run.epoch if run.model_kind == "fast_lewam" else None,
        stage="stage_b" if run.model_kind == "fast_lewam" else None,
    )
    base_policy = session._build_policy(subject, identity, str(device))
    collector = CEMTraceCollector(
        detailed_steps={
            int(step) for step in protocol["detailed_iterations"]
        }
    )
    base_policy.solver = TracedCEMSolver(base_policy.solver, collector)
    traced_policy = TracedPolicy(base_policy, collector)

    try:
        result = session.evaluate(
            traced_policy,
            identity=identity,
            output_dir=output_dir / "evaluation",
            device=str(device),
        )
        actual = [bool(episode.success) for episode in result.episodes]
        metadata = {
            "task": run.task,
            "family": run.family,
            "model_kind": run.model_kind,
            "epoch": run.epoch,
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "checkpoint": str(run.checkpoint),
            "checkpoint_sha256": _sha256(run.checkpoint),
            "run_config": str(run.run_config),
            "run_config_sha256": _sha256(run.run_config),
            "reference_result": str(run.reference_result),
            "reference_result_sha256": _sha256(run.reference_result),
            "manifest": str(manifest.path),
            "manifest_sha256": _sha256(manifest.path),
            "slot_selection": slot_selection,
            **{
                key: protocol[key]
                for key in (
                    "seed",
                    "num_eval",
                    "goal_offset_steps",
                    "eval_budget",
                    "horizon",
                    "receding_horizon",
                    "action_block",
                    "num_samples",
                    "n_steps",
                    "topk",
                )
            },
        }
        artifact = write_trace_artifacts(
            output_dir,
            run_label=run_label,
            collector=collector,
            actual_successes=actual,
            expected_successes=expected,
            metadata=metadata,
        )
        expected_summary_rows = len(collector.plans) * int(protocol["n_steps"])
        expected_detail_rows = len(collector.plans) * len(
            protocol["detailed_iterations"]
        )
        if len(collector.summaries) != expected_summary_rows:
            raise RuntimeError(
                "incomplete CEM summary trace: "
                f"{len(collector.summaries)} != {expected_summary_rows}"
            )
        if len(collector.details) != expected_detail_rows:
            raise RuntimeError(
                "incomplete detailed CEM trace: "
                f"{len(collector.details)} != {expected_detail_rows}"
            )
        (output_dir / "failure.json").unlink(missing_ok=True)
        print(
            f"{run_label}: status={artifact['status']} "
            f"mismatch_slots={artifact['mismatch_slots']} "
            f"trace={output_dir / 'trace.pt'}"
        )
        return artifact
    except Exception as exc:
        (output_dir / "failure.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    run_trace(
        manifest_path=args.manifest,
        run_label=args.run_label,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
