#!/usr/bin/env python3
"""Analyze the pre-registered Phase1.7 online confirmation evaluations."""

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

from source.common.round3_phase1 import CohortManifest


LABELS = (
    "selected_joint_rank_1_0",
    "joint_base",
    "lewm_base",
    "joint_offline",
    "lewm_rank_1_0",
    "lewm_offline",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_result(root: Path, task: str, label: str) -> Path:
    search_root = root / task / label
    paths = sorted(search_root.rglob("result.json")) if search_root.is_dir() else []
    if len(paths) != 1:
        raise FileNotFoundError(
            f"expected one result.json under {search_root}, found {len(paths)}"
        )
    return paths[0]


def _load_run(
    root: Path,
    task: str,
    label: str,
    *,
    cohort_id: str,
    cohort_sha256: str,
    actor_path: Path,
    expected_verifier: Path | None,
) -> tuple[dict, dict[str, bool]]:
    result_path = _find_result(root, task, label)
    run_dir = result_path.parent
    context_path = run_dir / "phase1_7_evaluation_context.json"
    episodes_path = run_dir / "episodes.jsonl"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    context = json.loads(context_path.read_text(encoding="utf-8"))
    if (
        result.get("status") != "ok"
        or result.get("round4_mode") != "P3"
        or context.get("mode") != "P3"
    ):
        raise ValueError(f"run is incomplete or is not P3: {run_dir}")
    if int(result.get("epoch", -1)) != 2:
        raise ValueError(f"run result does not record the locked epoch-2 actor: {run_dir}")
    if result.get("task") != task:
        raise ValueError(f"run task differs from locked task: {run_dir}")
    if result.get("cohort_id") != cohort_id or result.get("cohort_sha256") != cohort_sha256:
        raise ValueError(f"run does not use the locked confirmation cohort: {run_dir}")
    if int(context.get("training_epoch", -1)) != 2:
        raise ValueError(f"run did not use the locked epoch-2 actor: {run_dir}")
    if Path(context["actor_checkpoint"]).resolve() != actor_path.resolve():
        raise ValueError(f"run actor checkpoint differs from lock: {run_dir}")
    expected_comparison = (
        "joint" if label == "joint_base" else
        "lewm" if label in {"lewm_base", "lewm_rank_1_0", "lewm_offline"} else
        "online_joint"
    )
    if context.get("comparison") != expected_comparison:
        raise ValueError(f"run comparison kind differs from locked arm: {run_dir}")
    actual_verifier = context.get("verifier_checkpoint")
    if expected_verifier is None:
        if actual_verifier is not None:
            raise ValueError(f"Joint baseline unexpectedly has a separate verifier: {run_dir}")
    elif actual_verifier is None or Path(actual_verifier).resolve() != expected_verifier.resolve():
        raise ValueError(f"run verifier checkpoint differs from lock: {run_dir}")

    episodes: dict[str, bool] = {}
    with episodes_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = json.dumps(row["episode_id"], sort_keys=True)
            if key in episodes:
                raise ValueError(f"duplicate episode ID at {episodes_path}:{line_number}")
            if not isinstance(row.get("success"), bool):
                raise ValueError(f"episode success is not boolean at {episodes_path}:{line_number}")
            episodes[key] = row["success"]
    expected_count = int(result.get("summary", {}).get("num_episodes", -1))
    if len(episodes) != expected_count or expected_count != 200:
        raise ValueError(f"expected 200 episode rows, found {len(episodes)} at {episodes_path}")
    observed_rate = sum(episodes.values()) / len(episodes)
    if not np.isclose(float(result.get("success_rate", np.nan)), observed_rate):
        raise ValueError(f"result success rate differs from episode rows: {run_dir}")

    return (
        {
            "label": label,
            "run_dir": str(run_dir.resolve()),
            "result_sha256": _sha256(result_path),
            "episodes_sha256": _sha256(episodes_path),
            "context_sha256": _sha256(context_path),
            "comparison": context.get("comparison"),
            "successes": int(sum(episodes.values())),
            "success_rate": float(observed_rate),
            "evaluation_seconds": result.get("evaluation_seconds"),
        },
        episodes,
    )


def _paired_stats(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    baseline_name: str,
    seed: int,
    draws: int,
) -> dict:
    gains = int(np.sum(candidate & ~baseline))
    losses = int(np.sum(~candidate & baseline))
    discordant = gains + losses
    p_value = (
        float(binomtest(min(gains, losses), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    difference = candidate.astype(np.float64) - baseline.astype(np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(difference), size=(draws, len(difference)))
    means = difference[sampled].mean(axis=1)
    return {
        f"candidate_minus_{baseline_name}_pp": float(difference.mean() * 100.0),
        "paired_bootstrap_ci95_pp": [
            float(value * 100.0) for value in np.quantile(means, [0.025, 0.975])
        ],
        "discordant_starts": {
            "candidate_only_success": gains,
            "lewm_only_success": losses,
        },
        "exact_mcnemar_two_sided_p": p_value,
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
    }


def analyze(*, lock_path: Path, result_root: Path, output: Path, seed: int, draws: int) -> Path:
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    lock_path = lock_path.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    result_root = result_root.resolve()
    tasks = {}
    primary_p: dict[str, float] = {}

    for task_index, task in enumerate(("pusht", "reacher")):
        task_lock = lock["tasks"][task]
        if task_lock.get("selected_joint_b_arm") != "rank_1.0":
            raise ValueError(f"locked Joint-B arm differs from rank_1.0 for {task}")
        confirmation = lock["confirmation"][task]
        actor = Path(task_lock["actor_epoch2"]["path"]).resolve()
        if _sha256(actor) != task_lock["actor_epoch2"]["sha256"]:
            raise ValueError(f"locked actor checkpoint hash changed for {task}")
        cohort_path = Path(confirmation["path"]).resolve()
        cohort = CohortManifest.load(cohort_path)
        if cohort.cohort_id != confirmation["cohort_id"]:
            raise ValueError(f"confirmation cohort ID changed for {task}")
        if cohort.computed_sha256 != confirmation["sha256"]:
            raise ValueError(f"confirmation cohort hash changed for {task}")
        if len(cohort.entries) != int(confirmation["count"]):
            raise ValueError(f"confirmation cohort count changed for {task}")

        runs = {}
        vectors = {}
        for label in LABELS:
            checkpoint_key = {
                "selected_joint_rank_1_0": "selected_joint_rank_1.0",
                "joint_base": None,
                "lewm_base": "independent_lewm_epoch2",
                "joint_offline": "joint_offline",
                "lewm_rank_1_0": "lewm_rank_1.0",
                "lewm_offline": "lewm_offline",
            }[label]
            checkpoint_info = (
                None
                if checkpoint_key is None
                else (
                    task_lock[checkpoint_key]
                    if checkpoint_key == "independent_lewm_epoch2"
                    else task_lock["checkpoints"][checkpoint_key]
                )
            )
            expected_verifier = (
                None
                if checkpoint_info is None
                else Path(checkpoint_info["path"]).resolve()
            )
            if expected_verifier is not None and not expected_verifier.is_file():
                raise FileNotFoundError(expected_verifier)
            if expected_verifier is not None and _sha256(expected_verifier) != checkpoint_info["sha256"]:
                raise ValueError(f"locked verifier checkpoint hash changed for {task}/{label}")
            run, episodes = _load_run(
                result_root,
                task,
                label,
                cohort_id=confirmation["cohort_id"],
                cohort_sha256=confirmation["sha256"],
                actor_path=actor,
                expected_verifier=expected_verifier,
            )
            runs[label] = run
            vectors[label] = episodes

        baseline = vectors["lewm_base"]
        comparisons = {}
        baseline_order = sorted(baseline)
        baseline_vector = np.asarray([baseline[key] for key in baseline_order], dtype=bool)
        for offset, label in enumerate(LABELS):
            if label == "lewm_base":
                continue
            current = vectors[label]
            if current.keys() != baseline.keys():
                raise ValueError(f"paired episode IDs differ for {task}/{label}")
            vector = np.asarray([current[key] for key in baseline_order], dtype=bool)
            stats = _paired_stats(
                vector,
                baseline_vector,
                baseline_name="independent_lewm",
                seed=seed + task_index * len(LABELS) + offset,
                draws=draws,
            )
            comparisons[label] = stats
            if label == "selected_joint_rank_1_0":
                primary_p[task] = stats["exact_mcnemar_two_sided_p"]

        selected_vector = np.asarray(
            [vectors["selected_joint_rank_1_0"][key] for key in baseline_order],
            dtype=bool,
        )
        selected_vs_secondary = {}
        for offset, label in enumerate(
            ("joint_base", "joint_offline", "lewm_rank_1_0", "lewm_offline")
        ):
            secondary_vector = np.asarray(
                [vectors[label][key] for key in baseline_order], dtype=bool
            )
            selected_vs_secondary[label] = _paired_stats(
                selected_vector,
                secondary_vector,
                baseline_name=label,
                seed=seed + 100 + task_index * len(LABELS) + offset,
                draws=draws,
            )

        tasks[task] = {
            "cohort_id": confirmation["cohort_id"],
            "cohort_sha256": confirmation["sha256"],
            "count": len(baseline_order),
            "runs": runs,
            "comparisons_vs_independent_lewm": comparisons,
            "selected_rank_1_0_vs_predeclared_secondary_arms": selected_vs_secondary,
        }

    order = sorted(primary_p, key=primary_p.get)
    adjusted: dict[str, float] = {}
    previous = 0.0
    for rank, task in enumerate(order):
        current = min(1.0, (len(order) - rank) * primary_p[task])
        previous = max(previous, current)
        adjusted[task] = previous
    for task in tasks:
        tasks[task]["comparisons_vs_independent_lewm"]["selected_joint_rank_1_0"][
            "holm_adjusted_primary_p_two_task_family"
        ] = adjusted[task]

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "classification": "one-time locked confirmation; not an independent replication",
        "primary_comparison": "Joint-B rank_1.0 vs independent LeWM epoch-2 baseline",
        "primary_multiple_testing_family": "PushT and Reacher exact McNemar tests; Holm adjusted",
        "secondary_comparisons": "reported as exploratory paired estimates without confirmatory claims",
        "locked_selection_sha256": _sha256(lock_path),
        "locked_selection": str(lock_path),
        "result_root": str(result_root),
        "tasks": tasks,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"status": "ok", "output": str(output), "tasks": tasks}, ensure_ascii=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("outputs/round5/phase1_7/locked_confirmation_selection.json"),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("outputs/round5/phase1_7/locked_confirmation_closed_loop_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/round5/phase1_7/locked_confirmation_closed_loop_v1/analysis.json"),
    )
    parser.add_argument("--bootstrap-seed", type=int, default=16041)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    analyze(
        lock_path=args.lock,
        result_root=args.result_root,
        output=args.output,
        seed=args.bootstrap_seed,
        draws=args.bootstrap_draws,
    )


if __name__ == "__main__":
    main()
