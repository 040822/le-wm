#!/usr/bin/env python3
"""Analyze epoch-matched Joint-B P3 comparisons against LeWM and B-only.

The input JSON manifest has a ``pairs`` list. Each row names one task/seed
stratum and its two run directories, for example::

    {"family_group": "seed3072/pusht", "training_seed": 3072,
     "comparison": "lewm", "joint_dir": ".../joint/seed_16028",
     "comparison_dir": ".../lewm/seed_16028"}

Each family group must contain exactly one ``lewm`` row and one ``b_only``
row. P values are Holm-adjusted within each task/seed family; a global
adjustment over all supplied rows is also recorded as a sensitivity result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_run(run_dir: Path, *, expected_epoch: int) -> tuple[dict, dict, dict[str, bool], dict]:
    run_dir = run_dir.resolve()
    result_path = run_dir / "result.json"
    context_path = run_dir / "phase1_7_evaluation_context.json"
    episodes_path = run_dir / "episodes.jsonl"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    context = json.loads(context_path.read_text(encoding="utf-8"))
    if result.get("status") != "ok":
        raise ValueError(f"P3 evaluation is incomplete: {result_path}")
    if context.get("mode") != "P3":
        raise ValueError(f"expected P3 evaluation: {context_path}")
    if int(result.get("epoch", -1)) != expected_epoch:
        raise ValueError(f"result epoch differs from {expected_epoch}: {result_path}")
    if int(context.get("training_epoch", -1)) != expected_epoch:
        raise ValueError(f"context epoch differs from {expected_epoch}: {context_path}")
    if result.get("task") != context.get("task"):
        raise ValueError(f"task differs between result and context: {run_dir}")

    episodes: dict[str, bool] = {}
    with episodes_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
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
    if len(episodes) != expected_count:
        raise ValueError(f"episode count differs from result summary: {episodes_path}")

    metadata = {
        "run_dir": str(run_dir),
        "result_sha256": _sha256(result_path),
        "context_sha256": _sha256(context_path),
        "episodes_sha256": _sha256(episodes_path),
    }
    return result, context, episodes, metadata


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    order = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    previous = 0.0
    count = len(order)
    for rank, key in enumerate(order):
        previous = max(previous, min(1.0, (count - rank) * p_values[key]))
        adjusted[key] = previous
    return adjusted


def _load_training_metadata(path: str | None, *, role: str, task: str, seed: int) -> dict:
    if not path:
        raise ValueError(f"missing {role} training metadata for {task}/seed{seed}")
    metadata_path = Path(path).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("task") != task or int(metadata.get("seed", -1)) != seed:
        raise ValueError(f"{role} metadata task/seed mismatch: {metadata_path}")
    return {
        "path": str(metadata_path),
        "sha256": _sha256(metadata_path),
        "data": metadata,
    }


def _compare(pair: dict, *, expected_epoch: int, bootstrap_seed: int, draws: int) -> dict:
    label = str(pair["comparison"])
    if label not in {"lewm", "b_only"}:
        raise ValueError(f"unsupported primary comparison: {label}")
    joint_result, joint_context, joint_rows, joint_metadata = _load_run(
        Path(pair["joint_dir"]), expected_epoch=expected_epoch
    )
    other_result, other_context, other_rows, other_metadata = _load_run(
        Path(pair["comparison_dir"]), expected_epoch=expected_epoch
    )
    if joint_context.get("comparison") != "joint":
        raise ValueError(f"baseline is not Joint-B: {joint_context}")
    if other_context.get("comparison") != label:
        raise ValueError(f"comparison label does not match evaluation: {other_context}")
    for key in ("task", "actor_checkpoint", "candidate_count", "mode", "training_epoch", "action_normalizer_sha256"):
        if joint_context.get(key) != other_context.get(key):
            raise ValueError(f"Joint-B and {label} differ in {key}")
    expected_task = str(joint_context["task"])
    expected_seed = int(pair["training_seed"])
    joint_training = _load_training_metadata(
        joint_context.get("actor_training_metadata"),
        role="Joint actor",
        task=expected_task,
        seed=expected_seed,
    )
    comparison_training = _load_training_metadata(
        other_context.get("verifier_training_metadata"),
        role=label,
        task=expected_task,
        seed=expected_seed,
    )
    joint_meta = joint_training["data"]
    comparison_meta = comparison_training["data"]
    for key in ("episode_split_sha256", "normalizers"):
        if joint_meta.get(key) != comparison_meta.get(key):
            raise ValueError(f"Joint-B and {label} training metadata differ in {key}")
    if label == "b_only":
        for key in ("initial_model_state_sha256", "train_window_count"):
            if joint_meta.get(key) != comparison_meta.get(key):
                raise ValueError(f"Joint-B and B-only metadata differ in {key}")
    for key in ("task", "cohort_id", "cohort_sha256", "protocol_variant"):
        joint_value = joint_result.get(key)
        other_value = other_result.get(key)
        if key in {"cohort_id", "cohort_sha256"}:
            joint_value = joint_value or joint_result.get("cohort", {}).get(key)
            other_value = other_value or other_result.get("cohort", {}).get(key)
        if joint_value != other_value:
            raise ValueError(f"Joint-B and {label} differ in result {key}")

    joint_ids = set(joint_rows)
    other_ids = set(other_rows)
    if joint_ids != other_ids:
        raise ValueError("paired P3 episode IDs differ")
    episode_ids = sorted(joint_ids)
    joint_success = np.asarray([joint_rows[key] for key in episode_ids], dtype=bool)
    other_success = np.asarray([other_rows[key] for key in episode_ids], dtype=bool)
    challenger_only = int(np.sum(~joint_success & other_success))
    joint_only = int(np.sum(joint_success & ~other_success))
    discordant = challenger_only + joint_only
    p_value = float(
        binomtest(min(challenger_only, joint_only), discordant, 0.5).pvalue
        if discordant
        else 1.0
    )
    differences = other_success.astype(np.float64) - joint_success.astype(np.float64)
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(differences), size=(draws, len(differences)))
    ci = np.quantile(differences[indices].mean(axis=1), [0.025, 0.975]) * 100.0
    return {
        "family_group": str(pair["family_group"]),
        "task": joint_context["task"],
        "training_seed": int(pair["training_seed"]),
        "comparison": label,
        "classification": str(pair.get("classification", "final epoch-10 P3 confirmation")),
        "paired_episode_count": len(episode_ids),
        "cohort_id": joint_result.get("cohort_id", joint_result.get("cohort", {}).get("cohort_id")),
        "cohort_sha256": joint_result.get("cohort_sha256", joint_result.get("cohort", {}).get("cohort_sha256")),
        "actor_checkpoint": joint_context["actor_checkpoint"],
        "matched_training_metadata": {
            "joint_actor": {"path": joint_training["path"], "sha256": joint_training["sha256"]},
            label: {"path": comparison_training["path"], "sha256": comparison_training["sha256"]},
        },
        "joint_success_count": int(joint_success.sum()),
        "comparison_success_count": int(other_success.sum()),
        "comparison_minus_joint_pp": float(differences.mean() * 100.0),
        "paired_bootstrap_ci95_pp": [float(value) for value in ci],
        "discordant_episodes": {
            "comparison_only_success": challenger_only,
            "joint_only_success": joint_only,
        },
        "exact_mcnemar_two_sided_p_unadjusted": p_value,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_draws": draws,
        "joint_run": joint_metadata,
        "comparison_run": other_metadata,
    }


def analyze(manifest_path: Path, output_path: Path, *, expected_epoch: int, seed: int, draws: int) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("manifest must contain a non-empty pairs list")
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    rows = []
    for offset, pair in enumerate(pairs):
        rows.append(_compare(pair, expected_epoch=expected_epoch, bootstrap_seed=seed + offset, draws=draws))

    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[row["family_group"]].append(row)
    per_group_adjusted = {}
    for group, group_rows in by_group.items():
        labels = {row["comparison"] for row in group_rows}
        if labels != {"lewm", "b_only"} or len(group_rows) != 2:
            raise ValueError(f"{group} must include exactly one LeWM and one B-only test")
        adjusted = _holm({row["comparison"]: row["exact_mcnemar_two_sided_p_unadjusted"] for row in group_rows})
        per_group_adjusted[group] = adjusted
        for row in group_rows:
            row["holm_adjusted_p_within_task_seed_family"] = adjusted[row["comparison"]]

    global_adjusted = _holm({
        f"{row['family_group']}/{row['comparison']}": row["exact_mcnemar_two_sided_p_unadjusted"]
        for row in rows
    })
    for row in rows:
        row["holm_adjusted_p_all_strata_sensitivity"] = global_adjusted[f"{row['family_group']}/{row['comparison']}"]

    payload = {
        "schema_version": 1,
        "classification": manifest.get("classification", "final matched-checkpoint P3 confirmation"),
        "expected_epoch": expected_epoch,
        "multiple_testing_family": "Joint-B vs LeWM and Joint-B vs B-only within each task and training seed",
        "global_eight_test_holm_values_are_sensitivity_only": True,
        "cohort_independent_of_prior_analyses": False,
        "family_results": rows,
        "holm_adjustments_by_task_seed": per_group_adjusted,
        "holm_adjustments_all_strata_sensitivity": global_adjusted,
        "input_manifest": str(manifest_path.resolve()),
        "input_manifest_sha256": _sha256(manifest_path.resolve()),
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps({"status": "ok", "output": str(output_path), "families": len(by_group), "comparisons": len(rows)}, sort_keys=True))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-epoch", type=int, default=10)
    parser.add_argument("--bootstrap-seed", type=int, default=16041)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    analyze(args.manifest, args.output, expected_epoch=args.expected_epoch, seed=args.bootstrap_seed, draws=args.bootstrap_draws)


if __name__ == "__main__":
    main()
