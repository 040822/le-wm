#!/usr/bin/env python3
"""Score all Phase1.7 online checkpoints on one captured action pool."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.round5_phase1_7 import (
    _load_fast_stage_b_verifier,
    _load_lewm_verifier,
    _load_training_metadata,
    _phase_config,
)
from scripts.round5_phase1_7_fixed_pool import _paired_bootstrap, _rank_metrics
from source.common.checkpoint import load_policy_or_model
from source.common.round3_phase1 import CohortManifest


ARMS = ("offline", "branches", "rank_0.1", "rank_1.0")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_outcomes(summary_path: Path, candidates: np.ndarray) -> list[dict]:
    branch_root = summary_path.parent / "branches"
    outcomes = []
    expected_count = candidates.shape[0] * candidates.shape[1]
    for candidate_index in range(candidates.shape[1]):
        path = branch_root / f"candidate_{candidate_index:04d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                slot = int(row["slot"])
                actual = np.asarray(row["action"], dtype=np.float32)
                expected = candidates[slot, candidate_index]
                if not np.array_equal(actual, expected):
                    raise ValueError(
                        f"branch action mismatch in {path}:{line_number} "
                        f"(slot={slot}, candidate={candidate_index})"
                    )
                outcomes.append(row)
    if len(outcomes) != expected_count:
        raise ValueError(
            f"loaded {len(outcomes)} outcomes, expected {expected_count}"
        )
    return outcomes


def _exact_mcnemar(left: np.ndarray, right: np.ndarray) -> dict:
    from scipy.stats import binomtest

    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    gains = int(np.sum(~left & right))
    losses = int(np.sum(left & ~right))
    pvalue = (
        float(binomtest(min(gains, losses), gains + losses, 0.5).pvalue)
        if gains + losses
        else 1.0
    )
    return {"right_only_success": gains, "left_only_success": losses, "two_sided_p": pvalue}


def _metric_comparison(left: dict, right: dict) -> dict:
    result = {}
    for key in (
        "top1_success_5",
        "top1_success_25",
        "top1_distance_25",
        "physical_regret_25",
        "pairwise_sign_accuracy",
        "spearman_cost_distance",
    ):
        a = np.asarray([row[key] for row in left["by_state"]], dtype=np.float64)
        b = np.asarray([row[key] for row in right["by_state"]], dtype=np.float64)
        result[key] = _paired_bootstrap(a, b, seed=16028)
    a_success = np.asarray(
        [row["top1_success_25"] for row in left["by_state"]], dtype=bool
    )
    b_success = np.asarray(
        [row["top1_success_25"] for row in right["by_state"]], dtype=bool
    )
    result["top1_success_25_exact_mcnemar"] = _exact_mcnemar(a_success, b_success)
    return result


def _load_joint_scores(path: Path, actor_model, device: torch.device, pool: dict, batch_size: int) -> np.ndarray:
    del batch_size
    verifier, _ = _load_fast_stage_b_verifier(path, actor_model)
    verifier.to(device).eval()
    starts = torch.as_tensor(pool["joint_z_start"], dtype=torch.float32, device=device)
    goals = torch.as_tensor(pool["joint_z_goal"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(pool["candidates"], dtype=torch.float32, device=device)
    with torch.inference_mode():
        scores = verifier.get_cost_from_latents(starts, goals, actions)
    return scores.detach().float().cpu().numpy().astype(np.float64)


def _load_lewm_scores(path: Path, actor_model, device: torch.device, pool: dict, batch_size: int) -> np.ndarray:
    verifier, _, _ = _load_lewm_verifier(path, actor_model)
    verifier.to(device).eval()
    history = torch.as_tensor(pool["verifier_z_start"], dtype=torch.float32, device=device)
    goals = torch.as_tensor(pool["verifier_z_goal"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(pool["candidates"], dtype=torch.float32, device=device)
    history_actions = torch.as_tensor(pool["history_actions"], dtype=torch.float32, device=device)
    with torch.inference_mode():
        scores = verifier.score_action_candidates(
            history,
            goals,
            actions,
            context={"_phase17_history_actions": history_actions},
            candidate_batch_size=batch_size,
            solver_batch_size=1,
        )
    return scores.detach().float().cpu().numpy().astype(np.float64)


def run(args: argparse.Namespace) -> Path:
    if not str(args.device).startswith("cuda"):
        raise ValueError("the locked scorer matrix must run on an explicitly selected CUDA device")
    task = args.task
    summary_path = Path(args.reference_summary).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("task") != task or summary.get("scorer") != "lewm":
        raise ValueError("reference summary must be a LeWM fixed-pool capture for --task")
    pool_path = Path(summary["fixed_pool_artifact"]).resolve()
    with np.load(pool_path) as archive:
        pool = {name: archive[name].copy() for name in archive.files}
    required = {
        "candidates",
        "physical_actions",
        "joint_costs",
        "lewm_costs",
        "candidate_noise",
        "joint_z_start",
        "joint_z_goal",
        "verifier_z_start",
        "verifier_z_goal",
        "history_actions",
    }
    missing = required.difference(pool)
    if missing:
        raise ValueError(f"reference pool is missing scorer context fields: {sorted(missing)}")
    if pool["history_actions"].ndim != 3 or pool["history_actions"].shape[1] < 1:
        raise ValueError("reference LeWM capture did not preserve real history actions")
    noise_sha = hashlib.sha256(pool["candidate_noise"].tobytes()).hexdigest()
    if noise_sha != summary.get("candidate_noise_sha256"):
        raise ValueError("reference candidate-noise hash does not match its summary")

    manifest = CohortManifest.load(Path(args.cohort).resolve())
    if args.count < 1 or args.count > len(manifest.entries):
        raise ValueError("--count must select a non-empty prefix of the cohort")
    if args.count != len(manifest.entries):
        manifest = replace(
            manifest,
            entries=manifest.entries[: args.count],
            cohort_id=f"{manifest.cohort_id}_fixed_pool_{args.count}",
            cohort_sha256=None,
        )
    if manifest.task != task or manifest.computed_sha256 != summary.get("cohort_sha256"):
        raise ValueError("reference pool and requested cohort do not match")
    if len(manifest.entries) != pool["candidates"].shape[0]:
        raise ValueError("reference pool row count does not match the cohort")
    outcomes = _load_outcomes(summary_path, pool["candidates"])

    actor_path = Path(args.actor_checkpoint).resolve()
    actor_policy, resolved_actor = load_policy_or_model(str(actor_path))
    actor_model = getattr(actor_policy, "model", actor_policy)
    device = torch.device(args.device)
    actor_model.to(device).eval()
    actor_metadata, _ = _load_training_metadata(Path(resolved_actor))
    action_stats = None if actor_metadata is None else actor_metadata.get("normalizers", {}).get("action")
    if action_stats is None:
        raise ValueError("actor metadata must contain the train-only action normalizer")

    config = _phase_config()
    candidate_batch_size = int(config["evaluation"]["candidate_batch_size"])
    online_root = Path(args.online_root)
    joint_root = online_root / task / "joint" / "arms"
    lewm_name = "lewm_epoch2" if task == "reacher" else "lewm"
    lewm_root = online_root / task / lewm_name / "arms"

    base_joint = {
        "state_count": len(manifest.entries),
        "candidate_count": pool["candidates"].shape[1],
        "mean": summary["metrics"]["joint"]["mean"],
        "by_state": summary["metrics"]["joint"]["by_state"],
    }
    base_lewm = {
        "state_count": len(manifest.entries),
        "candidate_count": pool["candidates"].shape[1],
        "mean": summary["metrics"]["lewm"]["mean"],
        "by_state": summary["metrics"]["lewm"]["by_state"],
    }
    scorer_results = {"joint/base": base_joint, "lewm/base": base_lewm}
    identities = {}
    for family, directory, scorer in (
        ("joint", joint_root, _load_joint_scores),
        ("lewm", lewm_root, _load_lewm_scores),
    ):
        for arm in ARMS:
            checkpoint = (directory / arm / "checkpoints" / f"{arm}_policy.ckpt").resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            metadata, _ = _load_training_metadata(checkpoint)
            if metadata is None or metadata.get("normalizers", {}).get("action") != action_stats:
                raise ValueError(f"action normalizer mismatch for {family}/{arm}")
            online = metadata.get("online_finetune", {})
            if online.get("representation_frozen_bitwise") is not True:
                raise ValueError(f"representation was not frozen for {family}/{arm}")
            if online.get("success_labels_used_for_training") is not False or online.get("physical_distances_used_for_training") is not False:
                raise ValueError(f"training target audit failed for {family}/{arm}")
            scores = scorer(checkpoint, actor_model, device, pool, candidate_batch_size)
            if scores.shape != pool["joint_costs"].shape or not np.isfinite(scores).all():
                raise ValueError(f"invalid score matrix for {family}/{arm}: {scores.shape}")
            scorer_results[f"{family}/{arm}"] = _rank_metrics(
                scores, outcomes, seed=16027
            )
            identities[f"{family}/{arm}"] = {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
                "base_checkpoint": online.get("base_checkpoint"),
                "base_checkpoint_sha256": online.get("base_checkpoint_sha256"),
                "updates": online.get("updates"),
                "rank_loss_weight": online.get("rank_loss_weight"),
            }

    comparisons = {}
    for name, result in scorer_results.items():
        if name in {"joint/base", "lewm/base"}:
            continue
        comparisons[name] = {
            "vs_joint_base": _metric_comparison(base_joint, result),
            "vs_independent_lewm_base": _metric_comparison(base_lewm, result),
        }

    payload = {
        "schema_version": 1,
        "task": task,
        "cohort_id": manifest.cohort_id,
        "cohort_sha256": manifest.computed_sha256,
        "state_count": len(manifest.entries),
        "candidate_count": int(pool["candidates"].shape[1]),
        "same_candidate_pool_for_every_scorer": True,
        "candidate_noise_sha256": noise_sha,
        "fixed_pool_artifact": str(pool_path),
        "fixed_pool_sha256": summary.get("fixed_pool_sha256"),
        "actor_checkpoint": str(resolved_actor),
        "actor_checkpoint_sha256": summary.get("actor_checkpoint_sha256"),
        "scorer_identities": identities,
        "metrics": {name: value["mean"] for name, value in scorer_results.items()},
        "comparisons": comparisons,
    }
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(target), "metrics": payload["metrics"]}, ensure_ascii=False, sort_keys=True), flush=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("pusht", "reacher"))
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--actor-checkpoint", required=True)
    parser.add_argument("--reference-summary", required=True)
    parser.add_argument("--online-root", default="outputs/round5/phase1_7/online_training")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
