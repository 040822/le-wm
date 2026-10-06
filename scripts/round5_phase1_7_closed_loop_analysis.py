#!/usr/bin/env python3
"""Analyze paired Phase1.7 closed-loop confirmation results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_run(run_dir: Path) -> tuple[dict, dict[str, dict]]:
    run_dir = run_dir.resolve()
    result_path = run_dir / "result.json"
    episodes_path = run_dir / "episodes.jsonl"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "ok":
        raise ValueError(f"closed-loop run is incomplete: {result_path}")
    episodes = {}
    with episodes_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = json.dumps(row["episode_id"], sort_keys=True)
            if key in episodes:
                raise ValueError(f"duplicate episode_id in {episodes_path}: {key}")
            if not isinstance(row.get("success"), bool):
                raise ValueError(f"episode success must be boolean at {episodes_path}:{line_number}")
            episodes[key] = row
    if len(episodes) != int(result["summary"]["num_episodes"]):
        raise ValueError(f"episode row count differs from result summary: {episodes_path}")
    return result, episodes


def _paired_comparison(joint: dict, lewm: dict, *, seed: int, draws: int) -> dict:
    if joint.get("task") != lewm.get("task"):
        raise ValueError("paired runs belong to different tasks")
    if joint.get("cohort_id") != lewm.get("cohort_id"):
        raise ValueError("paired runs use different cohort IDs")
    if joint.get("cohort_sha256") != lewm.get("cohort_sha256"):
        raise ValueError("paired runs use different cohort manifests")
    if joint.get("protocol_variant") != lewm.get("protocol_variant"):
        raise ValueError("paired runs use different protocol variants")
    return {
        "task": joint["task"],
        "cohort_id": joint["cohort_id"],
        "cohort_sha256": joint["cohort_sha256"],
        "joint_successes": int(np.sum(joint["summary"]["success_vector"])),
        "joint_rate": float(joint["success_rate"]),
        "lewm_successes": int(np.sum(lewm["summary"]["success_vector"])),
        "lewm_rate": float(lewm["success_rate"]),
    }


def analyze(
    *,
    pusht_joint_dir: Path,
    pusht_lewm_dir: Path,
    reacher_joint_dir: Path,
    reacher_lewm_dir: Path,
    output: Path,
    locked_confirmation: Path,
    seed: int = 16041,
    draws: int = 10_000,
) -> Path:
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    inputs = {
        "pusht_joint": pusht_joint_dir,
        "pusht_lewm": pusht_lewm_dir,
        "reacher_joint": reacher_joint_dir,
        "reacher_lewm": reacher_lewm_dir,
    }
    loaded = {key: _load_run(path) for key, path in inputs.items()}
    by_task = {
        "pusht": (loaded["pusht_joint"], loaded["pusht_lewm"]),
        "reacher": (loaded["reacher_joint"], loaded["reacher_lewm"]),
    }
    comparisons = {}
    nominal_p = {}
    for offset, (task, (joint_pair, lewm_pair)) in enumerate(by_task.items()):
        joint_result, joint_episodes = joint_pair
        lewm_result, lewm_episodes = lewm_pair
        base = _paired_comparison(
            joint_result, lewm_result, seed=seed + offset, draws=draws
        )
        if joint_episodes.keys() != lewm_episodes.keys():
            raise ValueError(f"episode IDs differ between scorers for {task}")
        joint_success = np.asarray(
            [row["success"] for row in joint_episodes.values()], dtype=bool
        )
        lewm_success = np.asarray(
            [lewm_episodes[key]["success"] for key in joint_episodes], dtype=bool
        )
        lewm_only = int(np.sum(~joint_success & lewm_success))
        joint_only = int(np.sum(joint_success & ~lewm_success))
        discordant = lewm_only + joint_only
        p_value = float(
            binomtest(min(lewm_only, joint_only), discordant, 0.5).pvalue
            if discordant
            else 1.0
        )
        difference = lewm_success.astype(np.float64) - joint_success.astype(np.float64)
        rng = np.random.default_rng(seed + offset)
        indices = rng.integers(0, len(difference), size=(draws, len(difference)))
        bootstrap_means = difference[indices].mean(axis=1)
        base.update(
            {
                "paired_episode_count": len(difference),
                "lewm_minus_joint_pp": float(difference.mean() * 100.0),
                "discordant_episodes": {
                    "lewm_only_success": lewm_only,
                    "joint_only_success": joint_only,
                },
                "exact_mcnemar_two_sided_p": p_value,
                "paired_bootstrap_ci95_pp": [
                    float(value * 100.0)
                    for value in np.quantile(bootstrap_means, [0.025, 0.975])
                ],
                "bootstrap_seed": seed + offset,
                "bootstrap_draws": draws,
            }
        )
        comparisons[task] = base
        nominal_p[task] = p_value

    ordered = sorted(nominal_p, key=nominal_p.get)
    holm = {}
    previous = 0.0
    for rank, task in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - rank) * nominal_p[task])
        previous = max(previous, adjusted)
        holm[task] = previous
    for task in comparisons:
        comparisons[task]["holm_adjusted_p_two_task_family"] = holm[task]

    lock_path = locked_confirmation.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    locked_matches = {}
    for task, (joint_pair, lewm_pair) in by_task.items():
        task_lock = lock["confirmation_result_artifacts"][task]
        matched = True
        for scorer, (result, _) in (("joint", joint_pair), ("lewm", lewm_pair)):
            expected = task_lock["comparisons"][scorer]
            matched = matched and (
                result.get("cohort_sha256") == task_lock["cohort_sha256"]
                and result.get("success_vector_sha256")
                == expected["success_vector_sha256"]
            )
        locked_matches[task] = bool(matched)

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "comparison": "legacy Joint-B vs independent LeWM; P3 closed-loop",
        "classification": "rerun of locked confirmation cohort; not an independent replication",
        "matches_locked_confirmation_artifacts": locked_matches,
        "locked_confirmation_manifest": str(lock_path),
        "independent_confirmation_of_p3": False,
        "old_checkpoint_training_unseen_guaranteed": False,
        "multiple_testing_family": "PushT and Reacher exact McNemar tests; Holm adjusted",
        "inputs": {
            key: {
                "run_dir": str(path.resolve()),
                "result_sha256": _sha256(path.resolve() / "result.json"),
                "episodes_sha256": _sha256(path.resolve() / "episodes.jsonl"),
            }
            for key, path in inputs.items()
        },
        "comparisons": comparisons,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"status": "ok", "output": str(output), "comparisons": comparisons}, ensure_ascii=False, sort_keys=True))
    return output


def main() -> None:
    root = Path("outputs/round5/phase1_7/independent_confirmation")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pusht-joint", type=Path, default=root / "pusht/pusht_phase1_7_confirmation_16028_v1/joint/seed_16028")
    parser.add_argument("--pusht-lewm", type=Path, default=root / "pusht/pusht_phase1_7_confirmation_16028_v1/lewm/seed_16028")
    parser.add_argument("--reacher-joint", type=Path, default=root / "reacher/reacher_phase1_7_confirmation_16028_v1/joint/seed_16028")
    parser.add_argument("--reacher-lewm", type=Path, default=root / "reacher/reacher_phase1_7_confirmation_16028_v1/lewm/seed_16028")
    parser.add_argument("--output", type=Path, default=root / "analysis/closed_loop_pair_analysis.json")
    parser.add_argument("--locked-confirmation", type=Path, default=Path("outputs/round5/phase1_7/analysis_lock.json"))
    parser.add_argument("--bootstrap-seed", type=int, default=16041)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    analyze(
        pusht_joint_dir=args.pusht_joint,
        pusht_lewm_dir=args.pusht_lewm,
        reacher_joint_dir=args.reacher_joint,
        reacher_lewm_dir=args.reacher_lewm,
        output=args.output,
        locked_confirmation=args.locked_confirmation,
        seed=args.bootstrap_seed,
        draws=args.bootstrap_draws,
    )


if __name__ == "__main__":
    main()
