#!/usr/bin/env python3
"""Analyze a paired Phase1.7 Joint-B versus LeWM P3 comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(run_dir: Path, *, expected_epoch: int) -> tuple[dict, dict[str, bool]]:
    run_dir = run_dir.resolve()
    result_path = run_dir / "result.json"
    context_path = run_dir / "phase1_7_evaluation_context.json"
    episodes_path = run_dir / "episodes.jsonl"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    context = json.loads(context_path.read_text(encoding="utf-8"))
    if result.get("status") != "ok":
        raise ValueError(f"evaluation did not complete successfully: {run_dir}")
    if int(result.get("epoch", -1)) != expected_epoch:
        raise ValueError(f"result epoch does not match {expected_epoch}: {run_dir}")
    if int(context.get("training_epoch", -1)) != expected_epoch:
        raise ValueError(f"evaluation context epoch does not match {expected_epoch}: {run_dir}")

    episodes: dict[str, bool] = {}
    for line_number, line in enumerate(episodes_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        episode_id = str(row["episode_id"])
        if episode_id in episodes:
            raise ValueError(f"duplicate episode_id {episode_id} in {episodes_path}:{line_number}")
        episodes[episode_id] = bool(row["success"])
    if len(episodes) != int(result.get("cohort", {}).get("count", len(episodes))):
        raise ValueError(f"episode record count differs from result cohort: {run_dir}")
    return {
        "run_dir": str(run_dir),
        "result": result,
        "context": context,
        "result_sha256": _sha256(result_path),
        "episodes_sha256": _sha256(episodes_path),
        "context_sha256": _sha256(context_path),
    }, episodes


def analyze(
    *,
    joint_dir: Path,
    lewm_dir: Path,
    output: Path,
    expected_epoch: int,
    seed: int,
    draws: int,
    classification: str = "interim dev P3 comparison; not a confirmation result",
) -> Path:
    if expected_epoch < 1 or draws < 1:
        raise ValueError("expected_epoch and draws must be positive")
    joint_meta, joint = _load(joint_dir, expected_epoch=expected_epoch)
    lewm_meta, lewm = _load(lewm_dir, expected_epoch=expected_epoch)
    jr, lr = joint_meta["result"], lewm_meta["result"]
    jc, lc = joint_meta["context"], lewm_meta["context"]
    if jr.get("task") != lr.get("task"):
        raise ValueError("Joint and LeWM task names differ")
    if jr.get("cohort_id") != lr.get("cohort_id") or jr.get("cohort_sha256") != lr.get("cohort_sha256"):
        raise ValueError("Joint and LeWM must use the same cohort")
    if jc.get("actor_checkpoint") != lc.get("actor_checkpoint"):
        raise ValueError("Joint and LeWM must use the same actor checkpoint")
    if not lc.get("verifier_checkpoint"):
        raise ValueError("LeWM evaluation context has no verifier checkpoint")
    if set(joint) != set(lewm):
        raise ValueError("Joint and LeWM episode IDs do not match")

    episode_ids = sorted(joint)
    joint_success = np.asarray([joint[key] for key in episode_ids], dtype=np.float64)
    lewm_success = np.asarray([lewm[key] for key in episode_ids], dtype=np.float64)
    difference = lewm_success - joint_success
    lewm_only = int(np.sum((lewm_success == 1) & (joint_success == 0)))
    joint_only = int(np.sum((joint_success == 1) & (lewm_success == 0)))
    discordant = lewm_only + joint_only
    p_value = float(
        binomtest(min(lewm_only, joint_only), discordant, 0.5, alternative="two-sided").pvalue
        if discordant
        else 1.0
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(draws, len(difference)))
    bootstrap_means = difference[indices].mean(axis=1)
    ci95 = np.quantile(bootstrap_means, [0.025, 0.975]) * 100.0

    joint_rate = float(joint_success.mean())
    lewm_rate = float(lewm_success.mean())
    joint_seconds = float(jr.get("evaluation_seconds", 0.0))
    lewm_seconds = float(lr.get("evaluation_seconds", 0.0))
    payload = {
        "schema_version": 1,
        "classification": classification,
        "task": jr["task"],
        "training_epoch": expected_epoch,
        "cohort_id": jr["cohort_id"],
        "cohort_sha256": jr["cohort_sha256"],
        "paired_episode_count": len(episode_ids),
        "same_actor_checkpoint": jc["actor_checkpoint"],
        "joint": {
            "run_dir": joint_meta["run_dir"],
            "success_count": int(joint_success.sum()),
            "success_rate": joint_rate,
            "evaluation_seconds": joint_seconds,
            "result_sha256": joint_meta["result_sha256"],
            "episodes_sha256": joint_meta["episodes_sha256"],
        },
        "lewm": {
            "run_dir": lewm_meta["run_dir"],
            "verifier_checkpoint": lc["verifier_checkpoint"],
            "success_count": int(lewm_success.sum()),
            "success_rate": lewm_rate,
            "evaluation_seconds": lewm_seconds,
            "result_sha256": lewm_meta["result_sha256"],
            "episodes_sha256": lewm_meta["episodes_sha256"],
        },
        "paired_comparison_lewm_minus_joint": {
            "difference_pp": float(difference.mean() * 100.0),
            "bootstrap_ci95_pp": [float(value) for value in ci95],
            "bootstrap_seed": seed,
            "bootstrap_draws": draws,
            "discordant_episodes": {
                "lewm_only_success": lewm_only,
                "joint_only_success": joint_only,
            },
            "exact_mcnemar_two_sided_p_unadjusted": p_value,
        },
        "lewm_over_joint_evaluation_time_ratio": (
            lewm_seconds / joint_seconds if joint_seconds > 0 else None
        ),
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"status": "ok", "output": str(output), "task": jr["task"], "paired_episode_count": len(episode_ids), "lewm_minus_joint_pp": payload["paired_comparison_lewm_minus_joint"]["difference_pp"], "mcnemar_p": p_value}, ensure_ascii=False, sort_keys=True))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-dir", type=Path, required=True)
    parser.add_argument("--lewm-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument(
        "--classification",
        default="interim dev P3 comparison; not a confirmation result",
        help="descriptive label saved with the paired analysis",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=16041)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    analyze(
        joint_dir=args.joint_dir,
        lewm_dir=args.lewm_dir,
        output=args.output,
        expected_epoch=args.expected_epoch,
        classification=args.classification,
        seed=args.bootstrap_seed,
        draws=args.bootstrap_draws,
    )


if __name__ == "__main__":
    main()
