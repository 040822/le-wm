#!/usr/bin/env python3
"""Analyze compatible Phase1.5 candidate replays for Phase1.6 mechanisms.

The source candidate pool is reused only where its checkpoint, S=2 proposal,
candidate action, and physical bound audit agree with the Phase1.6 protocol.
For PushT, the one affected start state is removed from every N curve so each
selector is compared on the same complete nested pool.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from source.common.round3_protocol import evaluate_success
from source.common.round5_phase1_6 import state_cluster_bootstrap, stable_hash


DEFAULT_POOL_ROOT = ROOT / "outputs/round5/phase1_5_seed3072_legacy/diagnostics/candidate_pool"
DEFAULT_CLIP_AUDIT = ROOT / "outputs/round5/phase1_6_seed3072/audit/phase1_5_pool_clip_audit.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/round5/phase1_6_seed3072"
TASKS = ("pusht", "reacher")
COUNTS = (1, 4, 16, 64)
PAIR_BUDGET = 128
EPS = 1e-10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(name: str, seed: int) -> int:
    digest = hashlib.sha256(f"{name}|{int(seed)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _load_rows(pool_root: Path, task: str) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, Any]]:
    task_root = pool_root / task
    manifest_path = task_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "completed" or int(manifest.get("executed_candidates_per_flow_step", -1)) != 256:
        raise RuntimeError(f"candidate pool is incomplete for {task}: {manifest_path}")
    if 2 not in {int(value) for value in manifest.get("flow_steps", [])}:
        raise RuntimeError(f"candidate pool has no S=2 branch for {task}")

    proposal_path = task_root / "proposals_s2.jsonl"
    proposals: dict[tuple[str, int], dict[str, Any]] = {}
    with proposal_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            candidate = int(row["candidate_index"])
            if int(row["flow_steps"]) != 2 or candidate >= 64:
                continue
            key = (str(row["state_id"]), candidate)
            if key in proposals:
                raise RuntimeError(f"duplicate proposal key {key} in {proposal_path}")
            proposals[key] = row

    outcomes: dict[str, dict[int, dict[str, Any]]] = {}
    branch_hashes: dict[str, str] = {}
    for candidate in range(64):
        path = task_root / "branches" / "s2" / f"candidate_{candidate:04d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        branch_hashes[str(candidate)] = _sha256(path)
        seen_slots: set[int] = set()
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if int(row["flow_steps"]) != 2 or int(row["candidate_index"]) != candidate:
                    raise RuntimeError(f"branch identity mismatch in {path}")
                key = (str(row["state_id"]), candidate)
                proposal = proposals.get(key)
                if proposal is None:
                    raise RuntimeError(f"branch has no matching proposal: {key}")
                if row.get("candidate_noise_sha256") != proposal.get("candidate_noise_sha256"):
                    raise RuntimeError(f"proposal noise hash mismatch for {key}")
                if row.get("action") != proposal.get("action"):
                    raise RuntimeError(f"proposal action differs from replayed action for {key}")
                slot = int(row["slot"])
                if slot in seen_slots:
                    raise RuntimeError(f"duplicate slot {slot} in {path}")
                seen_slots.add(slot)
                state = str(row["state_id"])
                if candidate in outcomes.setdefault(state, {}):
                    raise RuntimeError(f"duplicate candidate {candidate} for {state}")
                outcomes[state][candidate] = row
        if len(seen_slots) != 50:
            raise RuntimeError(f"candidate {candidate} has {len(seen_slots)} starts; expected 50")

    if len(proposals) != 50 * 64:
        raise RuntimeError(f"S=2 proposal subset has {len(proposals)} rows; expected 3200")
    if len(outcomes) != 50 or any(len(rows) != 64 for rows in outcomes.values()):
        raise RuntimeError(f"S=2 replay subset is not a complete 50x64 pool for {task}")
    source = {
        "pool_manifest": str(manifest_path),
        "pool_manifest_sha256": _sha256(manifest_path),
        "proposal_file": str(proposal_path),
        "proposal_file_sha256": _sha256(proposal_path),
        "candidate_branches_sha256": stable_hash(branch_hashes),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "cohort_sha256": manifest.get("cohort_sha256"),
        "cohort_id": manifest.get("cohort_id"),
        "branch_hashes": branch_hashes,
    }
    return outcomes, source


def _clip_compatibility(clip_audit_path: Path, task: str) -> tuple[set[int], dict[str, Any]]:
    audit = _read_json(clip_audit_path)
    matches = [row for row in audit.get("tasks", []) if row.get("task") == task]
    if len(matches) != 1:
        raise RuntimeError(f"clip audit must contain exactly one record for {task}")
    record = matches[0]
    pairs = record.get("pairs_by_candidate", {})
    affected_slots: set[int] = set()
    affected_pairs = 0
    for candidate, rows in pairs.items():
        if int(candidate) >= 64:
            continue
        for row in rows:
            affected_slots.add(int(row["slot"]))
            affected_pairs += 1
    return affected_slots, {
        "clip_audit_path": str(clip_audit_path),
        "clip_audit_sha256": _sha256(clip_audit_path),
        "confirmation_action_bound_sha256": record.get("confirmation_action_bound_sha256"),
        "first_64_action_state_pairs_requiring_clip": affected_pairs,
        "excluded_slots_for_common_pool": sorted(affected_slots),
        "outcome_reuse_status": record.get("outcome_reuse_status"),
    }


def _metric_summary(values: Mapping[str, Sequence[float]], *, name: str, seed: int, samples: int) -> dict[str, Any]:
    return state_cluster_bootstrap(values, samples=samples, seed=_seed(name, seed))


def _cosines(pred: np.ndarray, truth: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    delta_pred = pred[i] - pred[j]
    delta_truth = truth[i] - truth[j]
    denominator = np.linalg.norm(delta_pred, axis=1) * np.linalg.norm(delta_truth, axis=1)
    numerator = np.einsum("ij,ij->i", delta_pred, delta_truth)
    result = np.full(len(i), np.nan, dtype=np.float64)
    valid = denominator > EPS
    result[valid] = numerator[valid] / denominator[valid]
    return np.clip(result, -1.0, 1.0)


def _sign_matches(pred: np.ndarray, truth: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    delta_pred = pred[i] - pred[j]
    delta_truth = truth[i] - truth[j]
    valid = (np.abs(delta_pred) > EPS) & (np.abs(delta_truth) > EPS)
    result = np.full(len(i), np.nan, dtype=np.float64)
    result[valid] = (np.sign(delta_pred[valid]) == np.sign(delta_truth[valid])).astype(np.float64)
    return result


def _sample_pairs(state_id: str, candidate_ids: Sequence[int], *, seed: int) -> np.ndarray:
    all_pairs = np.asarray(list(itertools.combinations(range(len(candidate_ids)), 2)), dtype=np.int64)
    if len(all_pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    count = min(PAIR_BUDGET, len(all_pairs))
    rng = np.random.default_rng(_seed(f"round5_phase1_6_pairs|{state_id}", seed))
    selected = rng.choice(len(all_pairs), size=count, replace=False)
    return all_pairs[np.sort(selected)]


def _consequence_analysis(
    task: str,
    rows_by_state: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    excluded_slots: set[int],
    seed: int,
    bootstrap_samples: int,
    permutation_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    latent_cos_by_state: dict[str, list[float]] = {}
    cost_sign_by_state: dict[str, list[float]] = {}
    latent_sign_by_state: dict[str, list[float]] = {}
    null_cos = np.full(permutation_samples, np.nan, dtype=np.float64)
    null_cost_sign = np.full(permutation_samples, np.nan, dtype=np.float64)
    state_null_cos: list[np.ndarray] = []
    state_null_cost_sign: list[np.ndarray] = []
    pair_manifest: list[dict[str, Any]] = []
    total_pairs = 0
    latent_identifiable = 0
    cost_identifiable = 0
    analyzed_states = 0

    for state_id in sorted(rows_by_state):
        all_rows = rows_by_state[state_id]
        slot = int(next(iter(all_rows.values()))["slot"])
        if slot in excluded_slots:
            continue
        valid_ids: list[int] = []
        for candidate in range(64):
            row = all_rows[candidate]
            milestone = row.get("milestones", {}).get("25")
            if not isinstance(milestone, Mapping) or milestone.get("future_latent") is None:
                continue
            if len(row.get("predicted_future_latent", [])) != 192 or len(milestone["future_latent"]) != 192:
                continue
            if not math.isfinite(float(row.get("predicted_cost", math.nan))):
                continue
            if not math.isfinite(float(milestone.get("distance", math.nan))):
                continue
            if not math.isfinite(float(milestone.get("future_latent_cost", math.nan))):
                continue
            if not np.isfinite(np.asarray(row["predicted_future_latent"], dtype=np.float64)).all():
                continue
            if not np.isfinite(np.asarray(milestone["future_latent"], dtype=np.float64)).all():
                continue
            valid_ids.append(candidate)
        if len(valid_ids) < 2:
            continue
        pred = np.asarray([all_rows[c]["predicted_future_latent"] for c in valid_ids], dtype=np.float64)
        truth = np.asarray([all_rows[c]["milestones"]["25"]["future_latent"] for c in valid_ids], dtype=np.float64)
        pred_cost = np.asarray([all_rows[c]["predicted_cost"] for c in valid_ids], dtype=np.float64)
        physical_cost = np.asarray([all_rows[c]["milestones"]["25"]["distance"] for c in valid_ids], dtype=np.float64)
        pairs = _sample_pairs(state_id, valid_ids, seed=seed)
        i, j = pairs[:, 0], pairs[:, 1]
        cosine = _cosines(pred, truth, i, j)
        cost_match = _sign_matches(pred_cost, physical_cost, i, j)
        latent_match = _sign_matches(
            np.asarray([all_rows[c]["predicted_cost"] for c in valid_ids], dtype=np.float64),
            np.asarray([all_rows[c]["milestones"]["25"]["future_latent_cost"] for c in valid_ids], dtype=np.float64),
            i,
            j,
        )
        cosine_valid = np.isfinite(cosine)
        cost_valid = np.isfinite(cost_match)
        latent_valid = np.isfinite(latent_match)
        if np.any(cosine_valid):
            latent_cos_by_state[state_id] = cosine[cosine_valid].tolist()
        if np.any(cost_valid):
            cost_sign_by_state[state_id] = cost_match[cost_valid].tolist()
        if np.any(latent_valid):
            latent_sign_by_state[state_id] = latent_match[latent_valid].tolist()
        total_pairs += len(pairs)
        latent_identifiable += int(cosine_valid.sum())
        cost_identifiable += int(cost_valid.sum())
        analyzed_states += 1
        pair_manifest.extend({
            "task": task,
            "state_id": state_id,
            "candidate_i": int(valid_ids[int(a)]),
            "candidate_j": int(valid_ids[int(b)]),
        } for a, b in pairs)

        rng = np.random.default_rng(_seed(f"round5_phase1_6_shuffle|{task}|{state_id}", seed))
        state_cos_null = np.full(permutation_samples, np.nan, dtype=np.float64)
        state_cost_null = np.full(permutation_samples, np.nan, dtype=np.float64)
        pred_delta = pred[i] - pred[j]
        pred_delta_norm = np.linalg.norm(pred_delta, axis=1)
        pred_cost_delta = pred_cost[i] - pred_cost[j]
        for permutation in range(permutation_samples):
            assignment = rng.permutation(len(valid_ids))
            shuffled_truth = truth[assignment]
            truth_delta = shuffled_truth[i] - shuffled_truth[j]
            denom = pred_delta_norm * np.linalg.norm(truth_delta, axis=1)
            ok = denom > EPS
            if np.any(ok):
                vals = np.einsum("ij,ij->i", pred_delta[ok], truth_delta[ok]) / denom[ok]
                state_cos_null[permutation] = float(np.mean(np.clip(vals, -1.0, 1.0)))
            shuffled_cost = physical_cost[assignment]
            cost_delta = shuffled_cost[i] - shuffled_cost[j]
            ok_sign = (np.abs(pred_cost_delta) > EPS) & (np.abs(cost_delta) > EPS)
            if np.any(ok_sign):
                state_cost_null[permutation] = float(np.mean(np.sign(pred_cost_delta[ok_sign]) == np.sign(cost_delta[ok_sign])))
        if np.any(cosine_valid):
            state_null_cos.append(state_cos_null)
        if np.any(cost_valid):
            state_null_cost_sign.append(state_cost_null)

    if not latent_cos_by_state:
        raise RuntimeError(f"no identifiable prediction/outcome pairs for {task}")
    null_cos_matrix = np.asarray(state_null_cos, dtype=np.float64)
    null_cost_matrix = np.asarray(state_null_cost_sign, dtype=np.float64)
    for permutation in range(permutation_samples):
        cos_values = null_cos_matrix[:, permutation] if null_cos_matrix.ndim == 2 else np.asarray([], dtype=np.float64)
        if np.isfinite(cos_values).any():
            null_cos[permutation] = float(np.nanmean(cos_values))
        cost_values = null_cost_matrix[:, permutation] if null_cost_matrix.ndim == 2 else np.asarray([], dtype=np.float64)
        if np.isfinite(cost_values).any():
            null_cost_sign[permutation] = float(np.nanmean(cost_values))

    observed_cos = _metric_summary(latent_cos_by_state, name=f"{task}_latent_cos", seed=seed, samples=bootstrap_samples)
    observed_cost = _metric_summary(cost_sign_by_state, name=f"{task}_cost_sign", seed=seed, samples=bootstrap_samples)
    observed_latent_cost = _metric_summary(latent_sign_by_state, name=f"{task}_latent_cost_sign", seed=seed, samples=bootstrap_samples)
    finite_cos_null = null_cos[np.isfinite(null_cos)]
    finite_cost_null = null_cost_sign[np.isfinite(null_cost_sign)]
    observed_cos_estimate = observed_cos["estimate"]
    observed_cost_estimate = observed_cost["estimate"]
    summary = {
        "task": task,
        "source_states": len(rows_by_state),
        "excluded_clip_incompatible_states": len(excluded_slots),
        "analyzed_states": analyzed_states,
        "candidate_pairs_preselected": total_pairs,
        "candidate_pairs_latent_identifiable": latent_identifiable,
        "candidate_pairs_cost_identifiable": cost_identifiable,
        "latent_identifiability_coverage": latent_identifiable / total_pairs if total_pairs else None,
        "cost_sign_identifiability_coverage": cost_identifiable / total_pairs if total_pairs else None,
        "pair_sampling": {"max_pairs_per_state": PAIR_BUDGET, "seed": int(seed), "algorithm": "uniform without replacement over unordered pairs among valid S=2 candidates 0..63"},
        "predicted_vs_true_latent_delta_cosine": observed_cos,
        "predicted_cost_vs_true_physical_distance_sign_accuracy": observed_cost,
        "predicted_cost_vs_true_latent_cost_sign_accuracy": observed_latent_cost,
        "within_state_shuffle_negative_control": {
            "permutations": int(permutation_samples),
            "latent_delta_cosine_null_mean": float(np.mean(finite_cos_null)) if len(finite_cos_null) else None,
            "latent_delta_cosine_null_ci95": [float(x) for x in np.quantile(finite_cos_null, [0.025, 0.975])] if len(finite_cos_null) else [None, None],
            "latent_delta_cosine_empirical_p_one_sided": (
                float((1 + np.count_nonzero(finite_cos_null >= observed_cos_estimate)) / (len(finite_cos_null) + 1))
                if len(finite_cos_null) and observed_cos_estimate is not None else None
            ),
            "cost_sign_null_mean": float(np.mean(finite_cost_null)) if len(finite_cost_null) else None,
            "cost_sign_null_ci95": [float(x) for x in np.quantile(finite_cost_null, [0.025, 0.975])] if len(finite_cost_null) else [None, None],
            "cost_sign_empirical_p_one_sided": (
                float((1 + np.count_nonzero(finite_cost_null >= observed_cost_estimate)) / (len(finite_cost_null) + 1))
                if len(finite_cost_null) and observed_cost_estimate is not None else None
            ),
        },
    }
    return summary, pair_manifest


def _selection_analysis(
    task: str,
    rows_by_state: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    excluded_slots: set[int],
    seed: int,
    bootstrap_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    long_rows: list[dict[str, Any]] = []
    state_records: list[dict[str, Any]] = []
    for n in COUNTS:
        by_selector: dict[str, dict[str, list[float]]] = {
            key: {} for key in ("b", "random_uniform", "latent_oracle", "physical_oracle")
        }
        b_random_success: dict[str, list[float]] = {}
        b_random_distance_improvement: dict[str, list[float]] = {}
        b_physical_regret: dict[str, list[float]] = {}
        b_latent_regret: dict[str, list[float]] = {}
        coverages: list[float] = []
        state_count = 0
        for state_id in sorted(rows_by_state):
            rows = rows_by_state[state_id]
            slot = int(next(iter(rows.values()))["slot"])
            if slot in excluded_slots:
                continue
            valid: list[tuple[int, Mapping[str, Any], Mapping[str, Any]]] = []
            for candidate in range(n):
                row = rows[candidate]
                milestone = row.get("milestones", {}).get("25")
                if not isinstance(milestone, Mapping) or milestone.get("future_latent") is None:
                    continue
                if milestone.get("future_latent_cost") is None or milestone.get("distance") is None:
                    continue
                current, goal = milestone.get("current"), milestone.get("goal")
                if current is None or goal is None:
                    continue
                success = bool(evaluate_success(task, current, goal))
                valid.append((candidate, row, {**milestone, "success_by_25": success}))
            coverage = len(valid) / n
            coverages.append(coverage)
            if not valid:
                continue
            state_count += 1
            candidates = [item[0] for item in valid]
            predictions = np.asarray([float(item[1]["predicted_cost"]) for item in valid], dtype=np.float64)
            distances = np.asarray([float(item[2]["distance"]) for item in valid], dtype=np.float64)
            latent_costs = np.asarray([float(item[2]["future_latent_cost"]) for item in valid], dtype=np.float64)
            successes = np.asarray([float(item[2]["success_by_25"]) for item in valid], dtype=np.float64)
            b_pos = int(np.argmin(predictions))
            latent_pos = int(np.argmin(latent_costs))
            physical_pos = int(np.argmin(distances))
            b_row = valid[b_pos]
            picked = {
                "b": b_pos,
                "random_uniform": None,
                "latent_oracle": latent_pos,
                "physical_oracle": physical_pos,
            }
            for selector, position in picked.items():
                if position is None:
                    continue
                by_selector[selector].setdefault(state_id, []).append(float(successes[position]))
                # The random selector is the exact expected outcome under uniform sampling,
                # not a single extra Monte Carlo draw.
            by_selector["random_uniform"].setdefault(state_id, []).append(float(successes.mean()))
            b_random_success[state_id] = [float(successes[b_pos] - successes.mean())]
            b_random_distance_improvement[state_id] = [float(distances.mean() - distances[b_pos])]
            b_physical_regret[state_id] = [float(distances[b_pos] - distances[physical_pos])]
            b_latent_regret[state_id] = [float(latent_costs[b_pos] - latent_costs[latent_pos])]
            state_records.append({
                "task": task,
                "n": n,
                "state_id": state_id,
                "slot": slot,
                "candidate_coverage": coverage,
                "valid_candidate_indices": candidates,
                "b_candidate_index": int(valid[b_pos][0]),
                "latent_oracle_candidate_index": int(valid[latent_pos][0]),
                "physical_oracle_candidate_index": int(valid[physical_pos][0]),
                "success_by_25": {
                    "b": float(successes[b_pos]),
                    "random_uniform_expected": float(successes.mean()),
                    "latent_oracle": float(successes[latent_pos]),
                    "physical_oracle": float(successes[physical_pos]),
                },
                "distance_at_25": {
                    "b": float(distances[b_pos]),
                    "random_uniform_expected": float(distances.mean()),
                    "latent_oracle": float(distances[latent_pos]),
                    "physical_oracle": float(distances[physical_pos]),
                },
                "predicted_cost_b": float(predictions[b_pos]),
                "true_latent_cost_b": float(latent_costs[b_pos]),
                "physical_regret_b": float(distances[b_pos] - distances[physical_pos]),
                "latent_regret_b": float(latent_costs[b_pos] - latent_costs[latent_pos]),
            })

        def add_metric(selector: str, metric: str, values: Mapping[str, Sequence[float]]) -> None:
            result = _metric_summary(values, name=f"{task}_N{n}_{selector}_{metric}", seed=seed, samples=bootstrap_samples)
            long_rows.append({
                "task": task,
                "n": n,
                "selector": selector,
                "metric": metric,
                "estimate": result["estimate"],
                "ci95_low": result["ci95"][0],
                "ci95_high": result["ci95"][1],
                "states": result["states"],
                "candidate_coverage_mean": float(np.mean(coverages)) if coverages else None,
                "candidate_coverage_min": float(np.min(coverages)) if coverages else None,
            })

        for selector, values in by_selector.items():
            add_metric(selector, "success_by_25", values)
        # Distance is analyzed on the same state and candidate intersection as success.
        for selector in by_selector:
            distance_values: dict[str, list[float]] = {}
            distance_key = "random_uniform_expected" if selector == "random_uniform" else selector
            for row in state_records:
                if row["task"] == task and row["n"] == n:
                    distance_values.setdefault(row["state_id"], []).append(row["distance_at_25"][distance_key])
            add_metric(selector, "distance_at_25", distance_values)
        add_metric("b_minus_random", "success_by_25", b_random_success)
        add_metric("random_minus_b", "distance_at_25_improvement", b_random_distance_improvement)
        add_metric("b", "physical_distance_regret", b_physical_regret)
        add_metric("b", "true_latent_cost_regret", b_latent_regret)
        long_rows.append({
            "task": task,
            "n": n,
            "selector": "coverage",
            "metric": "states_with_at_least_one_common_valid_candidate",
            "estimate": state_count,
            "ci95_low": None,
            "ci95_high": None,
            "states": state_count,
            "candidate_coverage_mean": float(np.mean(coverages)) if coverages else None,
            "candidate_coverage_min": float(np.min(coverages)) if coverages else None,
        })
    return long_rows, state_records


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap_samples <= 0 or args.permutation_samples <= 0:
        raise ValueError("bootstrap and permutation sample counts must be positive")
    pool_root = Path(args.pool_root).expanduser().resolve()
    clip_audit_path = Path(args.clip_audit).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    results: dict[str, Any] = {}
    consequence_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    state_records: list[dict[str, Any]] = []
    for task in args.tasks:
        rows_by_state, source = _load_rows(pool_root, task)
        excluded_slots, compatibility = _clip_compatibility(clip_audit_path, task)
        expected_checkpoint = args.checkpoint_sha256.get(task)
        if expected_checkpoint and source.get("checkpoint_sha256") != expected_checkpoint:
            raise RuntimeError(f"candidate-pool checkpoint hash mismatch for {task}")
        consequence, task_pairs = _consequence_analysis(
            task,
            rows_by_state,
            excluded_slots=excluded_slots,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
        )
        selectors, task_states = _selection_analysis(
            task,
            rows_by_state,
            excluded_slots=excluded_slots,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
        )
        results[task] = {
            "source": source,
            "clip_compatibility": compatibility,
            "candidate_consequence": consequence,
            "candidate_selection_n_curve": selectors,
        }
        consequence_rows.append({
            "task": task,
            "states": consequence["analyzed_states"],
            "pairs_preselected": consequence["candidate_pairs_preselected"],
            "latent_identifiable_coverage": consequence["latent_identifiability_coverage"],
            "latent_delta_cosine": consequence["predicted_vs_true_latent_delta_cosine"]["estimate"],
            "latent_delta_cosine_ci95_low": consequence["predicted_vs_true_latent_delta_cosine"]["ci95"][0],
            "latent_delta_cosine_ci95_high": consequence["predicted_vs_true_latent_delta_cosine"]["ci95"][1],
            "physical_cost_sign_accuracy": consequence["predicted_cost_vs_true_physical_distance_sign_accuracy"]["estimate"],
            "physical_cost_sign_ci95_low": consequence["predicted_cost_vs_true_physical_distance_sign_accuracy"]["ci95"][0],
            "physical_cost_sign_ci95_high": consequence["predicted_cost_vs_true_physical_distance_sign_accuracy"]["ci95"][1],
            "shuffle_latent_cosine_null_mean": consequence["within_state_shuffle_negative_control"]["latent_delta_cosine_null_mean"],
            "shuffle_latent_cosine_p": consequence["within_state_shuffle_negative_control"]["latent_delta_cosine_empirical_p_one_sided"],
            "shuffle_cost_sign_null_mean": consequence["within_state_shuffle_negative_control"]["cost_sign_null_mean"],
            "shuffle_cost_sign_p": consequence["within_state_shuffle_negative_control"]["cost_sign_empirical_p_one_sided"],
        })
        selection_rows.extend(selectors)
        pair_rows.extend(task_pairs)
        state_records.extend(task_states)

    output_dir = output_root / "mechanism"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "round5_phase1_6_candidate_mechanism_v1",
        "source_protocol": "Round5 Phase1.5 legacy_50 S=2 candidate pool",
        "reuse_rule": "reuse only exact proposal-to-replay matches whose actions are unchanged by the Phase1.6 physical bound audit; exclude any affected state from all N curves",
        "seed": int(args.seed),
        "bootstrap_samples": int(args.bootstrap_samples),
        "permutation_samples": int(args.permutation_samples),
        "tasks": results,
    }
    payload["analysis_sha256"] = stable_hash(payload)
    json_path = output_dir / "candidate_consequences.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "candidate_consequences.csv", consequence_rows, list(consequence_rows[0]))
    _write_csv(output_dir / "candidate_selection_n_curve.csv", selection_rows, list(selection_rows[0]))
    _write_csv(output_dir / "candidate_pair_manifest.csv", pair_rows, ["task", "state_id", "candidate_i", "candidate_j"])
    _write_csv(output_dir / "candidate_selection_state_records.csv", state_records, list(state_records[0]))
    paper_root = ROOT / "paper/tables/phase1_6"
    _write_csv(paper_root / "action_consequence.csv", consequence_rows, list(consequence_rows[0]))
    _write_csv(paper_root / "candidate_selection_n_curve.csv", selection_rows, list(selection_rows[0]))
    return {"json": str(json_path), "analysis_sha256": payload["analysis_sha256"], "tasks": {task: {"states": data["candidate_consequence"]["analyzed_states"], "pairs": data["candidate_consequence"]["candidate_pairs_preselected"]} for task, data in results.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", default=str(DEFAULT_POOL_ROOT))
    parser.add_argument("--clip-audit", default=str(DEFAULT_CLIP_AUDIT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--seed", type=int, default=16026)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutation-samples", type=int, default=1000)
    parser.add_argument("--checkpoint-sha256", type=json.loads, default={
        "pusht": "62d00096a34e9f0b6da4ceb6a3c61eb350701a7aade5f5e4d2c1ee528b06015e",
        "reacher": "087991339c9499d10c1a58a28e72d07c7dfd819d5f072c67bfd0169acaa83553",
    })
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
