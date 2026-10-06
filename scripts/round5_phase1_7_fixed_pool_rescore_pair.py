#!/usr/bin/env python3
"""Rescore a saved Phase1.7 fixed action pool with LeWM and compare rankers.

The reference summary must be a completed B-only fixed-pool run. This command
captures LeWM's current-task latent context, scores the saved action candidates,
and reuses the reference run's physical outcomes for all rank metrics. It does
not replay the environment branches.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.round5_phase1_7 import (
    _load_lewm_verifier,
    _load_training_metadata,
    _phase_config,
)
from scripts.round5_phase1_7_fixed_pool import (
    _capture_pool,
    _rank_metrics,
)
from scripts.round5_phase1_7_rescore import (
    _load_outcomes,
    _load_lewm_scores,
    _metric_comparison,
)
from source.common.checkpoint import load_policy_or_model
from source.common.eval import compose_eval_config, get_dataset
from source.common.round3_phase1 import CohortManifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _branch_tree_sha256(summary_path: Path) -> str:
    branch_dir = summary_path.parent / "branches"
    paths = sorted(branch_dir.glob("candidate_*.jsonl"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_reference(summary_path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("scorer") != "b_only":
        raise ValueError("reference summary must come from the B-only scorer")
    if summary.get("task") not in {"pusht", "reacher"}:
        raise ValueError("reference summary must be a PushT or Reacher fixed-pool run")
    pool_path = Path(summary["fixed_pool_artifact"]).resolve()
    if _sha256(pool_path) != summary.get("fixed_pool_sha256"):
        raise ValueError("fixed_pool.npz checksum differs from its summary")
    with np.load(pool_path) as archive:
        pool = {name: archive[name].copy() for name in archive.files}
    required = {"candidates", "physical_actions", "joint_costs", "b_only_costs", "candidate_noise"}
    missing = required.difference(pool)
    if missing:
        raise ValueError(f"fixed pool is missing required arrays: {sorted(missing)}")
    if pool["candidates"].shape[:2] != (int(summary["state_count"]), int(summary["candidate_count"])):
        raise ValueError("candidate array shape differs from fixed-pool summary")
    if pool["joint_costs"].shape != pool["b_only_costs"].shape:
        raise ValueError("Joint-B and B-only score matrices differ in shape")
    noise_sha = hashlib.sha256(pool["candidate_noise"].tobytes()).hexdigest()
    if noise_sha != summary.get("candidate_noise_sha256"):
        raise ValueError("candidate noise checksum differs from fixed-pool summary")
    return summary, pool


def run(args: argparse.Namespace) -> Path:
    if not str(args.device).startswith("cuda"):
        raise ValueError("LeWM rescoring must run on an explicitly selected CUDA device")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"4", "5", "6", "7"}:
        raise ValueError("this low-VRAM evaluation must expose exactly one GPU among 4–7")

    summary_path = Path(args.reference_summary).resolve()
    summary, reference = _load_reference(summary_path)
    task = str(summary["task"])
    actor_path = Path(args.actor_checkpoint or summary["actor_checkpoint"]).resolve()
    lewm_path = Path(args.lewm_checkpoint).resolve()
    if str(actor_path) != str(Path(summary["actor_checkpoint"]).resolve()):
        raise ValueError("actor checkpoint differs from the B-only fixed-pool reference")
    if _sha256(actor_path) != summary.get("actor_checkpoint_sha256"):
        raise ValueError("actor checkpoint checksum differs from the B-only summary")

    actor_metadata, actor_metadata_path = _load_training_metadata(actor_path)
    lewm_metadata, lewm_metadata_path = _load_training_metadata(lewm_path)
    if actor_metadata is None or lewm_metadata is None:
        raise ValueError("actor and LeWM training metadata are required")
    seed = int(actor_metadata["seed"])
    if actor_metadata.get("task") != task or lewm_metadata.get("task") != task:
        raise ValueError("actor/LeWM training metadata task differs from the reference")
    if int(lewm_metadata.get("seed", -1)) != seed:
        raise ValueError("LeWM and actor training seeds differ")
    actor_action_stats = actor_metadata.get("normalizers", {}).get("action")
    lewm_action_stats = lewm_metadata.get("normalizers", {}).get("action")
    if actor_action_stats is None or actor_action_stats != lewm_action_stats:
        raise ValueError("Joint actor and LeWM action normalizers differ")

    manifest = CohortManifest.load(Path(args.cohort).resolve())
    if manifest.task != task or manifest.computed_sha256 != summary.get("cohort_sha256"):
        raise ValueError("cohort differs from the B-only fixed-pool reference")
    if len(manifest.entries) != int(summary["state_count"]):
        raise ValueError("cohort count differs from the B-only fixed-pool reference")

    output_path = Path(args.output).resolve()
    scores_path = output_path.with_name(f"{output_path.stem}.scores.npz")
    capture_dir = output_path.parent / "lewm_diagnostic_capture"
    if output_path.exists() or scores_path.exists() or capture_dir.exists():
        raise FileExistsError(f"rescore output already exists: {output_path.parent}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    phase = _phase_config()
    cfg = compose_eval_config(
        task,
        [
            f"eval.num_eval={len(manifest.entries)}",
            f"world.num_envs={len(manifest.entries)}",
            "output.save_video=false",
            f"seed={manifest.seed}",
        ],
    )
    OmegaConf.update(
        cfg,
        "eval.action_normalizer_stats",
        {"action": actor_action_stats},
        force_add=True,
    )
    dataset = get_dataset(cfg, cfg.eval.dataset_name)

    actor_policy, resolved_actor = load_policy_or_model(str(actor_path))
    actor_model = getattr(actor_policy, "model", actor_policy)
    device = torch.device(args.device)
    actor_model.to(device).eval()
    verifier, resolved_lewm, history_size = _load_lewm_verifier(lewm_path, actor_model)
    verifier.to(device).eval()

    captured = _capture_pool(
        task=task,
        name="lewm",
        actor=actor_policy,
        verifier=verifier,
        actor_path=Path(resolved_actor),
        manifest=manifest,
        cfg=cfg,
        dataset=dataset,
        output_dir=capture_dir,
        device=args.device,
        candidate_count=int(summary["candidate_count"]),
        flow_steps=int(summary["flow_steps"]),
        training_epoch=2,
    )
    if captured.get("history_actions") is None or captured["history_actions"].shape[1] < 1:
        raise ValueError("LeWM capture did not preserve real history actions")
    if captured["verifier_z_start"].shape[0] != len(manifest.entries):
        raise ValueError("LeWM latent context count differs from reference pool")

    candidates_match = np.array_equal(captured["candidates"], reference["candidates"])
    noise_match = np.array_equal(captured["noise"], reference["candidate_noise"])
    if candidates_match:
        lewm_costs = captured["costs"].astype(np.float64, copy=False)
        score_source = "same-call-captured-candidates"
    else:
        lewm_costs = _load_lewm_scores(
            Path(resolved_lewm),
            actor_model,
            device,
            {
                "candidates": reference["candidates"],
                "verifier_z_start": captured["verifier_z_start"],
                "verifier_z_goal": captured["verifier_z_goal"],
                "history_actions": captured["history_actions"],
            },
            int(phase["evaluation"]["candidate_batch_size"]),
        )
        score_source = "rescored-reference-candidates"
    if lewm_costs.shape != reference["joint_costs"].shape or not np.isfinite(lewm_costs).all():
        raise ValueError("LeWM produced an invalid score matrix")

    branch_files = list((summary_path.parent / "branches").glob("candidate_*.jsonl"))
    if len(branch_files) != int(summary["candidate_count"]):
        raise ValueError("physical branch file count differs from candidate count")
    outcomes = _load_outcomes(summary_path, reference["candidates"])
    if len(outcomes) != int(summary["outcome_record_count"]):
        raise ValueError("loaded physical outcome count differs from B-only summary")
    outcome_keys = {
        (int(row["slot"]), int(row["candidate_index"])) for row in outcomes
    }
    expected_outcome_keys = {
        (slot, candidate)
        for slot in range(int(summary["state_count"]))
        for candidate in range(int(summary["candidate_count"]))
    }
    if len(outcome_keys) != len(outcomes) or outcome_keys != expected_outcome_keys:
        raise ValueError("physical outcomes contain duplicates or missing state/candidate pairs")
    joint_metrics = _rank_metrics(reference["joint_costs"], outcomes, seed=16027)
    b_only_metrics = _rank_metrics(reference["b_only_costs"], outcomes, seed=16027)
    lewm_metrics = _rank_metrics(lewm_costs, outcomes, seed=16027)
    comparisons = {
        "joint_b_minus_lewm": _metric_comparison(lewm_metrics, joint_metrics),
        "b_only_minus_lewm": _metric_comparison(lewm_metrics, b_only_metrics),
    }

    temp_scores = scores_path.with_suffix(scores_path.suffix + ".tmp")
    with temp_scores.open("wb") as stream:
        np.savez_compressed(
            stream,
            candidates=reference["candidates"],
            physical_actions=reference["physical_actions"],
            candidate_noise=reference["candidate_noise"],
            joint_costs=reference["joint_costs"],
            b_only_costs=reference["b_only_costs"],
            lewm_costs=lewm_costs,
            verifier_z_start=captured["verifier_z_start"],
            verifier_z_goal=captured["verifier_z_goal"],
            history_actions=captured["history_actions"],
        )
    temp_scores.replace(scores_path)

    payload = {
        "schema_version": 1,
        "classification": "epoch-2 dev fixed-pool diagnostic; same physical outcomes and action pool; not terminal or confirmation",
        "task": task,
        "training_seed": seed,
        "training_epoch": 2,
        "cohort_id": summary["cohort_id"],
        "cohort_sha256": summary["cohort_sha256"],
        "state_count": int(summary["state_count"]),
        "candidate_count": int(summary["candidate_count"]),
        "physical_outcomes_reused_from": str(summary_path),
        "environment_replayed_for_lewm": False,
        "candidate_pool_audit": {
            "candidate_actions_exact": candidates_match,
            "candidate_noise_exact": noise_match,
            "reference_candidate_noise_sha256": summary["candidate_noise_sha256"],
            "shared_pool_sha256": _array_sha256(
                reference["candidates"],
                reference["physical_actions"],
                reference["candidate_noise"],
            ),
            "lewm_score_source": score_source,
        },
        "models": {
            "joint_actor_checkpoint": str(actor_path),
            "joint_actor_checkpoint_sha256": _sha256(actor_path),
            "joint_actor_training_metadata": str(actor_metadata_path),
            "lewm_checkpoint": str(resolved_lewm),
            "lewm_checkpoint_sha256": _sha256(Path(resolved_lewm)),
            "lewm_training_metadata": str(lewm_metadata_path),
            "lewm_history_size": history_size,
        },
        "source_artifacts": {
            "b_only_summary": str(summary_path),
            "b_only_summary_sha256": _sha256(summary_path),
            "b_only_fixed_pool_sha256": _sha256(Path(summary["fixed_pool_artifact"])),
            "b_only_branch_tree_sha256": _branch_tree_sha256(summary_path),
            "scored_arrays": str(scores_path),
            "scored_arrays_sha256": _sha256(scores_path),
        },
        "mean_metrics": {
            "joint_b": joint_metrics["mean"],
            "b_only": b_only_metrics["mean"],
            "lewm": lewm_metrics["mean"],
        },
        "paired_comparisons": comparisons,
        "intervals_adjusted_for_metric_scan": False,
        "bootstrap_unit": "starting state",
    }
    temp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp_output.replace(output_path)
    print(json.dumps({"status": "ok", "output": str(output_path), "task": task, "training_seed": seed}, sort_keys=True))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-summary", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--lewm-checkpoint", required=True, type=Path)
    parser.add_argument("--actor-checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
