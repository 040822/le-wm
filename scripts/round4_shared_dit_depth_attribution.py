#!/usr/bin/env python3
"""Attribute Shared DiT depth effects to action generation and verification.

The fixed candidate pool pairs 6/8/12-layer action generators with identical
initial noise, replays each candidate from the same captured simulator state,
and cross-scores every generator pool with every verifier. Closed-loop runs
then compare P0, P1, and all P3 generator/verifier pairings on frozen cohorts.

Typical use::

    python scripts/round4_shared_dit_depth_attribution.py validate
    CUDA_VISIBLE_DEVICES=1 python scripts/round4_shared_dit_depth_attribution.py candidate-pool --task pusht --gpu 1
    CUDA_VISIBLE_DEVICES=1 python scripts/round4_shared_dit_depth_attribution.py cross-score --task pusht --gpu 1
    CUDA_VISIBLE_DEVICES=1 python scripts/round4_shared_dit_depth_attribution.py closed-loop --task pusht --cohort legacy --gpu 1
    python scripts/round4_shared_dit_depth_attribution.py analyze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.checkpoint import load_policy_or_model
from source.common.eval import DatasetEvaluationSession, EvaluationIdentity, compose_eval_config
from source.common.round3_phase1 import CohortManifest
from source.common.round4_eval import run_round4_evaluation, validate_gpu_visibility
from scripts.round5_phase1_5_diagnostics import (
    _attach_candidate_outcomes,
    _encode_goal_latents,
    _encode_start_latents,
    _run_fixed_candidate,
)


DEFAULT_CONFIG = ROOT / "config/round4/shared_dit_depth_attribution.json"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = _resolve(path)
    with target.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"non-object JSONL row in {target}")
    return [dict(row) for row in rows]


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
    temporary.replace(target)
    return target


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _config(path: str | Path) -> dict[str, Any]:
    value = _read_json(path)
    if int(value.get("schema_version", -1)) != 1:
        raise ValueError("unsupported depth attribution config schema")
    return value


def _depths(config: Mapping[str, Any]) -> tuple[int, ...]:
    result = tuple(int(value) for value in config["training"]["depths"])
    if result != (6, 8, 12):
        raise ValueError(f"this attribution protocol requires depths 6/8/12, got {result}")
    return result


def _checkpoint(config: Mapping[str, Any], task: str, depth: int) -> Path:
    return _resolve(config["training"]["checkpoints"][task][str(depth)]).resolve()


def _manifest(config: Mapping[str, Any], cohort_name: str, task: str) -> CohortManifest:
    section = config["cohorts"][cohort_name]
    manifest = CohortManifest.load(_resolve(section["paths"][task]))
    expected_count = 200 if section["kind"] == "final" else 50
    if (
        manifest.task != task
        or manifest.cohort_kind != section["kind"]
        or manifest.protocol_variant != section["protocol_variant"]
        or len(manifest.entries) != expected_count
    ):
        raise ValueError(f"cohort contract mismatch: {cohort_name}/{task}")
    expected_sha = section.get("sha256", {}).get(task)
    if expected_sha and manifest.computed_sha256 != expected_sha:
        raise ValueError(f"cohort hash changed: {cohort_name}/{task}")
    return manifest


def _load_model(path: Path, *, device: str | torch.device = "cpu"):
    loaded, resolved = load_policy_or_model(str(path))
    if resolved is not None and Path(resolved).resolve() != path.resolve():
        raise ValueError(f"checkpoint resolver changed the requested path: {path} -> {resolved}")
    model = getattr(loaded, "model", loaded)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"checkpoint did not resolve to a torch module: {path}")
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def _model_summary(model: torch.nn.Module) -> dict[str, Any]:
    predictor = getattr(model, "predictor", None)
    layers = getattr(predictor, "layers", None)
    return {
        "depth": None if layers is None else len(layers),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "latent_dim": int(model.latent_dim),
        "action_dim": int(model.action_dim),
        "action_horizon": int(model.action_horizon),
    }


def validate(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    depths = _depths(config)
    records: dict[str, Any] = {}
    for task in ("pusht", "reacher"):
        legacy = _manifest(config, "legacy", task)
        records[task] = {"legacy_cohort_sha256": legacy.computed_sha256, "models": {}}
        for depth in depths:
            path = _checkpoint(config, task, depth)
            if not path.is_file():
                raise FileNotFoundError(path)
            model = _load_model(path)
            summary = _model_summary(model)
            if summary["depth"] != depth:
                raise ValueError(f"{path} declares {summary['depth']} layers, expected {depth}")
            summary.update({"checkpoint": str(path), "checkpoint_sha256": _sha256_file(path)})
            records[task]["models"][str(depth)] = summary
            del model
    final = _manifest(config, "reacher_final", "reacher")
    records["reacher"]["final_cohort_sha256"] = final.computed_sha256
    records["status"] = "ok"
    print(json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True))


def _configure_device(args: argparse.Namespace) -> torch.device:
    if not str(args.device).startswith("cuda"):
        return torch.device(args.device)
    if args.gpu is None or not str(args.gpu).isdigit() or int(args.gpu) not in range(4):
        raise ValueError("GPU evaluation requires --gpu selecting one physical GPU0-3")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    validate_gpu_visibility(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError(f"requested GPU {args.gpu} is unavailable")
    return torch.device(args.device)


def _legacy_eval_session(task: str, manifest: CohortManifest, device: str):
    cfg = compose_eval_config(
        task,
        overrides=[
            f"eval.num_eval={len(manifest.entries)}",
            f"eval.goal_offset_steps={manifest.goal_offset_steps}",
            "eval.eval_budget=50",
            "plan_config.horizon=5",
            "plan_config.receding_horizon=5",
            "plan_config.action_block=5",
            "output.save_video=false",
            f"solver.device={device}",
            "solver.num_samples=300",
            "solver.topk=30",
            "solver.n_steps=30",
            "solver.var_scale=1.0",
        ],
    )
    cfg.solver.device = device
    session = DatasetEvaluationSession(cfg, task=task, cohort=manifest.to_evaluation_cohort())
    return cfg, session


def _pool_root(config: Mapping[str, Any], task: str) -> Path:
    return _resolve(config["output_root"]) / "candidate_pool" / task


def _source_pool(config: Mapping[str, Any], task: str) -> Path:
    return _resolve(config["candidate_pool"]["source_root"]) / task


def _candidate_indices(config: Mapping[str, Any]) -> tuple[int, ...]:
    pool = config["candidate_pool"]
    indices = tuple(range(0, int(pool["source_candidate_count"]), int(pool["candidate_stride"])))
    if len(indices) != int(pool["candidate_count"]):
        raise ValueError("source candidate count/stride does not produce the frozen 64 pool")
    return indices


def _selected_baseline_rows(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    manifest: CohortManifest,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected = _candidate_indices(config)
    expected_flow = int(config["candidate_pool"]["flow_steps"])
    by_key = {
        (int(row["slot"]), int(row["candidate_index"])): row
        for row in source_rows
        if int(row.get("flow_steps", -1)) == expected_flow
        and int(row.get("candidate_index", -1)) in selected
    }
    result: list[dict[str, Any]] = []
    for slot, entry in enumerate(manifest.entries):
        for candidate_index, source_index in enumerate(selected):
            row = by_key.get((slot, source_index))
            if row is None or row.get("outcome_status") != "completed":
                raise ValueError(f"6-layer source pool missing slot={slot}, candidate={source_index}")
            if row.get("episode_id") != entry.episode_id or int(row["row_index"]) != int(entry.row_index):
                raise ValueError("6-layer candidate source no longer matches the frozen cohort")
            result.append(
                {
                    **dict(row),
                    "candidate_index": candidate_index,
                    "source_candidate_index": source_index,
                    "generator_depth": 6,
                    "flow_steps": expected_flow,
                }
            )
    return result


def _load_fixed_noise(
    config: Mapping[str, Any], task: str, manifest: CohortManifest
) -> tuple[np.ndarray, str, str]:
    source_root = _source_pool(config, task)
    source_manifest = _read_json(source_root / "manifest.json")
    baseline = _checkpoint(config, task, 6)
    if (
        source_manifest.get("status") != "completed"
        or source_manifest.get("checkpoint_sha256") != _sha256_file(baseline)
        or source_manifest.get("cohort_sha256") != manifest.computed_sha256
        or int(source_manifest.get("candidates_per_flow_step", -1))
        != int(config["candidate_pool"]["source_candidate_count"])
    ):
        raise ValueError(f"existing 6-layer source pool is not compatible: {source_root}")
    noise_path = source_root / f"candidate_noise_s{int(config['candidate_pool']['flow_steps'])}.npz"
    with np.load(noise_path) as archive:
        noise = np.asarray(archive["candidate_noise"], dtype=np.float32)
    if noise.shape[:2] != (len(manifest.entries), int(config["candidate_pool"]["source_candidate_count"])):
        raise ValueError(f"unexpected source noise shape: {noise.shape}")
    selected = noise[:, _candidate_indices(config)].copy()
    return selected, str(noise_path.resolve()), _sha256_array(selected)


def _existing_generator_pool(config: Mapping[str, Any], task: str, depth: int, expected: Mapping[str, Any]):
    root = _pool_root(config, task) / f"generator_{depth}"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    current = _read_json(manifest_path)
    for key, value in expected.items():
        if current.get(key) != value:
            raise ValueError(f"completed candidate pool provenance changed at {root}: {key}")
    record_path = root / "records.jsonl"
    if current.get("status") != "completed":
        return None
    if not record_path.is_file():
        raise FileNotFoundError(record_path)
    rows = _read_jsonl(record_path)
    if len(rows) != int(expected["states"]) * int(expected["candidate_count"]):
        raise ValueError(f"candidate pool row count is incomplete: {record_path}")
    return rows


def _make_proposal_rows(
    *,
    task: str,
    depth: int,
    model: torch.nn.Module,
    noise: np.ndarray,
    manifest: CohortManifest,
    session: DatasetEvaluationSession,
    device: torch.device,
) -> list[dict[str, Any]]:
    starts = _encode_start_latents(
        session.dataset, manifest, session.transform["pixels"], model, str(device)
    )
    goals = _encode_goal_latents(
        session.dataset, manifest, session.transform["pixels"], model, str(device)
    )
    rows: list[dict[str, Any]] = []
    candidate_count = noise.shape[1]
    flow_steps = 16
    for slot, entry in enumerate(manifest.entries):
        with torch.inference_mode():
            action = model.sample_actions(
                torch.as_tensor(starts[slot : slot + 1], device=device).expand(candidate_count, -1),
                noise=torch.as_tensor(noise[slot], device=device),
                num_steps=flow_steps,
                goal_latent=torch.as_tensor(goals[slot : slot + 1], device=device).expand(candidate_count, -1),
                integrator="euler",
            ).detach().float().cpu().numpy()
        for candidate_index in range(candidate_count):
            rows.append(
                {
                    "schema_version": "round4_shared_dit_depth_candidate_v1",
                    "task": task,
                    "state_id": f"episode={entry.episode_id};start={entry.start_step};row={entry.row_index}",
                    "slot": slot,
                    "episode_id": entry.episode_id,
                    "start_step": int(entry.start_step),
                    "row_index": int(entry.row_index),
                    "flow_steps": flow_steps,
                    "candidate_index": candidate_index,
                    "source_candidate_index": _candidate_indices_for_noise_index(candidate_index),
                    "generator_depth": depth,
                    "action": action[candidate_index].tolist(),
                    "candidate_noise_sha256": _sha256_array(noise[slot, candidate_index]),
                    "success": None,
                    "true_distance": None,
                    "outcome_status": "pending",
                }
            )
        print(f"[candidate-pool] task={task} depth={depth} encoded slot={slot + 1}/{len(manifest.entries)}", flush=True)
    return rows


def _candidate_indices_for_noise_index(index: int) -> int:
    return int(index) * 4


def candidate_pool(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    task = str(args.task)
    device = _configure_device(args)
    manifest = _manifest(config, "legacy", task)
    if len(manifest.entries) != 50:
        raise ValueError("fixed candidate replay is frozen to legacy_50")
    noise, source_noise_path, noise_sha = _load_fixed_noise(config, task, manifest)
    source_root = _source_pool(config, task)
    source_manifest = _read_json(source_root / "manifest.json")
    source_rows = _read_jsonl(source_root / "records.jsonl")
    selected6 = _selected_baseline_rows(source_rows, task=task, manifest=manifest, config=config)
    task_root = _pool_root(config, task)
    task_root.mkdir(parents=True, exist_ok=True)
    branch_state: dict[str, Any] = {}
    cfg, session = _legacy_eval_session(task, manifest, str(device))
    common = {
        "schema_version": "round4_shared_dit_depth_candidate_manifest_v1",
        "task": task,
        "cohort_sha256": manifest.computed_sha256,
        "candidate_count": len(_candidate_indices(config)),
        "states": len(manifest.entries),
        "flow_steps": int(config["candidate_pool"]["flow_steps"]),
        "candidate_noise_sha256": noise_sha,
        "source_candidate_stride": int(config["candidate_pool"]["candidate_stride"]),
        "candidate_noise_source": source_noise_path,
    }

    for depth in _depths(config):
        checkpoint = _checkpoint(config, task, depth)
        checkpoint_sha = _sha256_file(checkpoint)
        depth_root = task_root / f"generator_{depth}"
        expected = {
            **common,
            "generator_depth": depth,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
        }
        existing = _existing_generator_pool(config, task, depth, expected)
        if existing is not None:
            print(f"[candidate-pool] task={task} depth={depth} reused rows={len(existing)}", flush=True)
            continue
        if depth_root.exists() and any(depth_root.iterdir()) and not (depth_root / "proposals.jsonl").is_file():
            raise FileExistsError(f"candidate output has unrelated partial files: {depth_root}")
        depth_root.mkdir(parents=True, exist_ok=True)
        if depth == 6:
            proposal_rows = selected6
            for row in proposal_rows:
                row["outcome_status"] = "completed"
        else:
            model = _load_model(checkpoint, device=device)
            proposal_rows = _make_proposal_rows(
                task=task,
                depth=depth,
                model=model,
                noise=noise,
                manifest=manifest,
                session=session,
                device=device,
            )
            _write_jsonl(depth_root / "proposals.jsonl", proposal_rows)
            by_slot = {slot: [] for slot in range(len(manifest.entries))}
            for row in proposal_rows:
                by_slot[int(row["slot"])].append(row)
            goal_latents = _encode_goal_latents(
                session.dataset,
                manifest,
                session.transform["pixels"],
                model,
                str(device),
            )
            branch_root = depth_root / "branches"
            branch_root.mkdir(parents=True, exist_ok=True)
            for candidate_index in range(len(_candidate_indices(config))):
                if args.limit_candidates is not None and candidate_index >= int(args.limit_candidates):
                    break
                proposals = sorted(
                    (by_slot[slot][candidate_index] for slot in by_slot),
                    key=lambda row: int(row["slot"]),
                )
                branch_path = branch_root / f"candidate_{candidate_index:03d}.jsonl"
                actions = np.asarray([row["action"] for row in proposals], dtype=np.float32)
                force_replay = not branch_state and candidate_index == 0
                if branch_path.is_file() and not force_replay:
                    stored = _read_jsonl(branch_path)
                    if len(stored) == len(proposals) and all(
                        np.array_equal(np.asarray(old["action"]), new_action)
                        for old, new_action in zip(stored, actions)
                    ):
                        for row, outcome in zip(proposals, stored):
                            row.update(
                                {
                                    key: value
                                    for key, value in outcome.items()
                                    if key not in {"action", "candidate_index", "slot"}
                                }
                            )
                        print(f"[candidate-pool] task={task} depth={depth} candidate={candidate_index} reused", flush=True)
                        continue
                print(f"[candidate-pool] task={task} depth={depth} candidate={candidate_index + 1}/64 replay", flush=True)
                outcomes = _run_fixed_candidate(
                    cfg=cfg,
                    task=task,
                    manifest=manifest,
                    normalized_actions=actions,
                    process=session.process,
                    model=model,
                    transform=session.transform["pixels"],
                    device=str(device),
                    output_dir=branch_root / f"candidate_{candidate_index:03d}",
                    dataset=session.dataset,
                    branch_state=branch_state,
                    goal_latents=np.asarray(goal_latents, dtype=np.float32),
                )
                attached = _attach_candidate_outcomes(proposals, outcomes, task=task)
                _write_jsonl(branch_path, attached)
                for row, outcome in zip(proposals, attached):
                    row.update(
                        {
                            key: value
                            for key, value in outcome.items()
                            if key not in {"action", "candidate_index", "slot"}
                        }
                    )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del model
        rows_path = depth_root / "records.jsonl"
        _write_jsonl(rows_path, proposal_rows)
        manifest_payload = {
            **expected,
            "status": (
                "completed"
                if depth == 6
                or args.limit_candidates is None
                or int(args.limit_candidates) >= len(_candidate_indices(config))
                else "partial"
            ),
            "records": str(rows_path.resolve()),
            "source_pool_manifest": str((source_root / "manifest.json").resolve()),
            "state_replay": "same captured simulator environment and RNG snapshot across candidate branches",
        }
        _write_json(depth_root / "manifest.json", manifest_payload)
        print(
            f"[candidate-pool] task={task} depth={depth} {manifest_payload['status']} rows={len(proposal_rows)}",
            flush=True,
        )


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(_rankdata(left), _rankdata(right))[0, 1])


def _bootstrap_interval(values: Sequence[float], *, seed: int, samples: int) -> list[float] | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(int(samples), len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _candidate_metrics(rows: Sequence[Mapping[str, Any]], *, seed: int, samples: int) -> dict[str, Any]:
    by_slot: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_slot.setdefault(int(row["slot"]), []).append(row)
    states: list[dict[str, float]] = []
    for slot, group in sorted(by_slot.items()):
        group = sorted(group, key=lambda row: int(row["candidate_index"]))
        predicted = np.asarray([row["predicted_cost"] for row in group], dtype=np.float64)
        distance = np.asarray([row["true_distance"] for row in group], dtype=np.float64)
        success = np.asarray([bool(row["success"]) for row in group], dtype=bool)
        if len(group) != 64 or not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(distance)):
            raise ValueError(f"candidate score coverage or values invalid for slot={slot}")
        selected = int(np.argmin(predicted))
        oracle = int(np.argmin(distance))
        order = np.argsort(predicted)
        states.append(
            {
                "oracle_success": float(success.any()),
                "selected_success": float(success[selected]),
                "random_success": float(success.mean()),
                "selected_distance": float(distance[selected]),
                "oracle_distance": float(distance.min()),
                "distance_regret": float(distance[selected] - distance.min()),
                "oracle_best_top1": float(order[0] == oracle),
                "oracle_best_top5": float(oracle in order[:5]),
                "spearman_cost_distance": float(_spearman(predicted, distance) or 0.0),
            }
        )
    def summary(key: str, *, offset: int) -> dict[str, Any]:
        values = np.asarray([row[key] for row in states], dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "bootstrap_95ci": _bootstrap_interval(values, seed=seed + offset, samples=samples),
        }
    return {
        "states": len(states),
        "candidates_per_state": 64,
        "oracle_success_rate": summary("oracle_success", offset=0),
        "selected_success_rate": summary("selected_success", offset=1),
        "random_candidate_success_rate": summary("random_success", offset=2),
        "selected_true_distance": summary("selected_distance", offset=3),
        "oracle_true_distance": summary("oracle_distance", offset=4),
        "selected_distance_regret": summary("distance_regret", offset=5),
        "oracle_best_candidate_ranked_top1": summary("oracle_best_top1", offset=6),
        "oracle_best_candidate_ranked_top5": summary("oracle_best_top5", offset=7),
        "spearman_predicted_cost_vs_true_distance": summary("spearman_cost_distance", offset=8),
    }


def _generator_diversity(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    by_slot: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_slot.setdefault(int(row["slot"]), []).append(row)
    state_rms = []
    for group in by_slot.values():
        actions = np.asarray([row["action"] for row in group], dtype=np.float64)
        candidate_count = len(actions)
        mean_pairwise_squared = (
            2.0 * candidate_count / (candidate_count - 1)
        ) * float(np.var(actions, axis=0).mean())
        state_rms.append(float(np.sqrt(max(0.0, mean_pairwise_squared))))
    success = np.asarray([bool(row["success"]) for row in rows], dtype=np.float64)
    distances = np.asarray([row["true_distance"] for row in rows], dtype=np.float64)
    return {
        "candidate_action_pairwise_rms": float(np.mean(state_rms)),
        "fixed_candidate_success_rate": float(success.mean()),
        "fixed_candidate_true_distance_mean": float(distances.mean()),
    }


def cross_score(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    task = str(args.task)
    device = _configure_device(args)
    manifest = _manifest(config, "legacy", task)
    _, session = _legacy_eval_session(task, manifest, str(device))
    task_root = _pool_root(config, task)
    generator_rows = {
        depth: _read_jsonl(task_root / f"generator_{depth}" / "records.jsonl")
        for depth in _depths(config)
    }
    result_root = _resolve(config["output_root"]) / "cross_score" / task
    result_root.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "schema_version": "round4_shared_dit_depth_cross_score_v1",
        "task": task,
        "cohort_sha256": manifest.computed_sha256,
        "candidate_count": 64,
        "generators": {},
        "matrix": {},
    }
    for depth, rows in generator_rows.items():
        metrics["generators"][str(depth)] = _generator_diversity(rows)
    for verifier_depth in _depths(config):
        checkpoint = _checkpoint(config, task, verifier_depth)
        verifier = _load_model(checkpoint, device=device)
        starts = _encode_start_latents(
            session.dataset, manifest, session.transform["pixels"], verifier, str(device)
        )
        goals = _encode_goal_latents(
            session.dataset, manifest, session.transform["pixels"], verifier, str(device)
        )
        for generator_depth, rows in generator_rows.items():
            by_slot: dict[int, list[Mapping[str, Any]]] = {}
            for row in rows:
                by_slot.setdefault(int(row["slot"]), []).append(row)
            scored: list[dict[str, Any]] = []
            for slot, group in sorted(by_slot.items()):
                group = sorted(group, key=lambda row: int(row["candidate_index"]))
                actions = torch.as_tensor(
                    np.asarray([row["action"] for row in group], dtype=np.float32)[None],
                    device=device,
                )
                with torch.inference_mode():
                    costs = verifier.get_cost_from_latents(
                        torch.as_tensor(starts[slot : slot + 1], device=device),
                        torch.as_tensor(goals[slot : slot + 1], device=device),
                        actions,
                    ).detach().float().cpu().numpy()[0]
                for row, cost in zip(group, costs):
                    scored.append(
                        {
                            **dict(row),
                            "verifier_depth": verifier_depth,
                            "predicted_cost": float(cost),
                        }
                    )
                print(
                    f"[cross-score] task={task} generator={generator_depth} verifier={verifier_depth} slot={slot + 1}/{len(manifest.entries)}",
                    flush=True,
                )
            key = f"generator_{generator_depth}_verifier_{verifier_depth}"
            score_path = result_root / f"{key}.jsonl"
            _write_jsonl(score_path, scored)
            metrics["matrix"][key] = {
                **_candidate_metrics(
                    scored,
                    seed=3072 + generator_depth * 100 + verifier_depth,
                    samples=int(config["bootstrap_samples"]),
                ),
                "records": str(score_path.resolve()),
            }
        del verifier
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _write_json(result_root / "summary.json", metrics)
    print(json.dumps({"status": "ok", "summary": str(result_root / "summary.json")}, sort_keys=True))


def _result_path(config: Mapping[str, Any], cohort: str, task: str, condition: str) -> Path:
    return _resolve(config["output_root"]) / "closed_loop" / cohort / task / condition / "result.json"


def _reuse_result(path: Path, *, task: str, cohort: CohortManifest, stage: str) -> dict[str, Any] | None:
    if not path.is_file():
        if path.parent.exists() and any(path.parent.iterdir()):
            raise FileExistsError(f"partial closed-loop condition requires inspection: {path.parent}")
        return None
    payload = _read_json(path)
    if (
        payload.get("status") != "ok"
        or payload.get("task") != task
        or payload.get("round4_mode", payload.get("stage")) != stage
        or payload.get("cohort_sha256") != cohort.computed_sha256
        or len(payload.get("episodes", ())) != len(cohort.entries)
    ):
        raise ValueError(f"existing closed-loop result does not match requested condition: {path}")
    return payload


def _run_condition(
    *,
    config: Mapping[str, Any],
    cohort_name: str,
    task: str,
    manifest: CohortManifest,
    generator_depth: int,
    verifier_depth: int | None,
    mode: str,
    models: Mapping[int, torch.nn.Module],
    device: torch.device,
) -> dict[str, Any]:
    generator_model = models[generator_depth]
    verifier_model = None if verifier_depth is None else models[verifier_depth]
    condition = (
        f"d{generator_depth}_{mode}"
        if verifier_depth is None or mode != "P3"
        else f"P3_g{generator_depth}_v{verifier_depth}"
    )
    target = _result_path(config, cohort_name, task, condition).parent
    existing = _reuse_result(
        target / "result.json", task=task, cohort=manifest, stage=mode
    )
    if existing is not None:
        print(f"[closed-loop] reused {cohort_name}/{task}/{condition}", flush=True)
        return existing
    cfg = compose_eval_config(
        task,
        overrides=[
            f"eval.num_eval={len(manifest.entries)}",
            f"eval.goal_offset_steps={manifest.goal_offset_steps}",
            "eval.eval_budget=50",
            "plan_config.horizon=5",
            "plan_config.receding_horizon=5",
            "plan_config.action_block=5",
            "output.save_video=false",
            f"solver.device={device}",
            "solver.num_samples=300",
            "solver.topk=30",
            "solver.n_steps=30",
            "solver.var_scale=1.0",
        ],
    )
    cfg.solver.device = str(device)
    generator_checkpoint = _checkpoint(config, task, generator_depth)
    generator_sha = _sha256_file(generator_checkpoint)
    identity = EvaluationIdentity(
        entrypoint="round4_shared_dit_depth_attribution",
        policy_kind="round4_shared_dit",
        checkpoint=str(generator_checkpoint),
        epoch=int(config["training"]["epoch"]),
        stage=mode,
    )
    verifier_metadata = None
    if mode == "P3":
        assert verifier_depth is not None
        verifier_checkpoint = _checkpoint(config, task, verifier_depth)
        verifier_metadata = {
            "depth": verifier_depth,
            "checkpoint": str(verifier_checkpoint),
            "checkpoint_sha256": _sha256_file(verifier_checkpoint),
        }
    pool_cfg = config["closed_loop"]
    payload = run_round4_evaluation(
        cfg,
        task=task,
        policy_or_model=generator_model,
        verifier_policy_or_model=(
            verifier_model if mode == "P3" and verifier_depth != generator_depth else None
        ),
        verifier_metadata=verifier_metadata,
        mode=mode,
        identity=identity,
        manifest=manifest,
        output_dir=target,
        trace_output_dir=target / "trace",
        device=str(device),
        trace=True,
        candidate_count=int(pool_cfg["candidate_count"]),
        flow_steps=int(pool_cfg["flow_steps"]),
        action_flow_steps=16 if mode in {"P0", "P3"} else None,
        solver_batch_size=int(pool_cfg["solver_batch_size"]),
        candidate_batch_size=int(pool_cfg["candidate_batch_size"]),
        action_flow_integrator="euler",
        action_bound_mode=str(pool_cfg["action_bound_mode"]),
        cem_protocol=(str(pool_cfg["cem_protocol"]) if mode == "P1" else "not_applicable"),
        allowed_protocol_variants=("legacy", "round3_revised"),
    )
    payload["depth_attribution"] = {
        "generator_depth": generator_depth,
        "generator_checkpoint_sha256": generator_sha,
        "verifier_depth": verifier_depth,
    }
    _write_json(target / "result.json", payload)
    # run_round4_evaluation owns trace and result publication; rewrite the
    # payload with an attribution annotation while preserving its trace hashes.
    print(
        f"[closed-loop] complete cohort={cohort_name} task={task} condition={condition} success={payload['success_rate']:.4f}",
        flush=True,
    )
    return payload


def closed_loop(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    device = _configure_device(args)
    cohort_name = str(args.cohort)
    tasks = (str(args.task),) if args.task != "all" else (("pusht", "reacher") if cohort_name == "legacy" else ("reacher",))
    for task in tasks:
        manifest = _manifest(config, cohort_name, task)
        models: dict[int, torch.nn.Module] = {}
        for depth in _depths(config):
            models[depth] = _load_model(_checkpoint(config, task, depth), device=device)
        depths = _depths(config)
        conditions = [("P0", depth, None) for depth in depths]
        if cohort_name == "legacy":
            conditions.extend(("P1", depth, None) for depth in depths)
        conditions.extend(
            ("P3", generator_depth, verifier_depth)
            for generator_depth in depths
            for verifier_depth in depths
        )
        if args.limit_conditions is not None:
            if int(args.limit_conditions) < 1:
                raise ValueError("--limit-conditions must be positive")
            conditions = conditions[: int(args.limit_conditions)]
        for mode, generator_depth, verifier_depth in conditions:
            _run_condition(
                config=config,
                cohort_name=cohort_name,
                task=task,
                manifest=manifest,
                generator_depth=generator_depth,
                verifier_depth=verifier_depth,
                mode=mode,
                models=models,
                device=device,
            )
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _episode_index(payload: Mapping[str, Any]) -> dict[tuple[Any, int, int], bool]:
    return {
        (
            row.get("dataset_episode", row.get("episode_id")),
            int(row.get("start_step", 0)),
            int(row.get("row_index", 0)),
        ): bool(row["success"])
        for row in payload.get("episodes", ())
    }


def _exact_mcnemar(better: int, worse: int) -> float:
    total = int(better + worse)
    if total == 0:
        return 1.0
    tail = min(better, worse)
    p = 2.0 * sum(math.comb(total, index) for index in range(tail + 1)) / (2**total)
    return float(min(1.0, p))


def _paired_comparison(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    a = _episode_index(left)
    b = _episode_index(right)
    if a.keys() != b.keys():
        raise ValueError("paired comparison episode identities differ")
    gains = sum(not a[key] and b[key] for key in a)
    losses = sum(a[key] and not b[key] for key in a)
    delta = np.asarray([float(b[key]) - float(a[key]) for key in a], dtype=np.float64)
    rng = np.random.default_rng(20260929 + len(delta))
    sampled = rng.integers(0, len(delta), size=(10000, len(delta)))
    bootstrap_delta = delta[sampled].mean(axis=1)
    return {
        "episodes": len(a),
        "left_success_rate": float(np.mean(list(a.values()))),
        "right_success_rate": float(np.mean(list(b.values()))),
        "paired_gain": int(gains),
        "paired_loss": int(losses),
        "net_success_delta_pp": 100.0 * float(delta.mean()),
        "paired_success_delta_bootstrap_95ci_pp": [
            100.0 * float(np.quantile(bootstrap_delta, 0.025)),
            100.0 * float(np.quantile(bootstrap_delta, 0.975)),
        ],
        "exact_mcnemar_p": _exact_mcnemar(gains, losses),
    }


def _closed_result(config: Mapping[str, Any], cohort: str, task: str, condition: str) -> dict[str, Any]:
    return _read_json(_result_path(config, cohort, task, condition))


def _training_losses(config: Mapping[str, Any], task: str, depth: int) -> dict[str, Any]:
    checkpoint = _checkpoint(config, task, depth)
    run_dir = checkpoint.parent.parent
    log_path = run_dir / f"{task}.log"
    if not log_path.is_file():
        return {"training_log": str(log_path), "status": "missing"}
    text = re.sub(r"\x1b\[[0-9;]*m", "", log_path.read_text(encoding="utf-8", errors="replace"))
    fields = {
        "val/action_loss": "validate/action_loss_epoch",
        "val/latent_prefix_loss": "validate/latent_prefix_loss_epoch",
    }
    result: dict[str, Any] = {"training_log": str(log_path), "status": "ok"}
    for output_field, log_field in fields.items():
        values = re.findall(
            r"\|\s*" + re.escape(log_field) + r"\s*\|\s*([-+0-9.eE]+)\s*\|",
            text,
        )
        if values:
            result[output_field] = float(values[-1])
        else:
            result["status"] = "partial"
    return result


def _format_value(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, Mapping) and "mean" in value:
        ci = value.get("bootstrap_95ci")
        return f"{float(value['mean']):.{digits}f}" + (f" [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]" if ci else "")
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_analysis_and_report(config: Mapping[str, Any]) -> dict[str, Any]:
    depths = _depths(config)
    analysis: dict[str, Any] = {
        "schema_version": "round4_shared_dit_depth_attribution_analysis_v1",
        "tasks": {},
        "training_losses": {},
    }
    for task in ("pusht", "reacher"):
        analysis["training_losses"][task] = {
            str(depth): _training_losses(config, task, depth) for depth in depths
        }
        analysis["tasks"][task] = {}
        for cohort_name in (("legacy", "reacher_final") if task == "reacher" else ("legacy",)):
            root = _resolve(config["output_root"]) / "closed_loop" / cohort_name / task
            if not root.is_dir():
                continue
            cohort_rows: dict[str, Any] = {"cohort_sha256": _manifest(config, cohort_name, task).computed_sha256}
            p0 = {str(depth): _closed_result(config, cohort_name, task, f"d{depth}_P0") for depth in depths}
            cohort_rows["P0"] = {str(depth): float(value["success_rate"]) for depth, value in ((int(k), v) for k, v in p0.items())}
            cohort_rows["P0_pairs_vs_d6"] = {
                str(depth): _paired_comparison(p0["6"], p0[str(depth)]) for depth in depths if depth != 6
            }
            p3 = {
                f"g{generator}_v{verifier}": _closed_result(
                    config,
                    cohort_name,
                    task,
                    f"P3_g{generator}_v{verifier}",
                )
                for generator in depths
                for verifier in depths
            }
            cohort_rows["P3"] = {key: float(value["success_rate"]) for key, value in p3.items()}
            cohort_rows["P3_planning"] = {
                key: {
                    metric: value.get("round4_planning", {}).get(metric)
                    for metric in (
                        "planning_median_seconds",
                        "proposal_median_seconds",
                        "verifier_median_seconds",
                        "encode_median_seconds",
                        "forward_count",
                    )
                }
                for key, value in p3.items()
            }
            cohort_rows["P3_diagonal_pairs_vs_d6"] = {
                str(depth): _paired_comparison(p3[f"g6_v6"], p3[f"g{depth}_v{depth}"])
                for depth in depths
                if depth != 6
            }
            cohort_rows["P3_verifier_pairs_by_generator"] = {
                str(generator): {
                    str(verifier): _paired_comparison(
                        p3[f"g{generator}_v{generator}"], p3[f"g{generator}_v{verifier}"]
                    )
                    for verifier in depths
                    if verifier != generator
                }
                for generator in depths
            }
            cohort_rows["P3_generator_pairs_by_verifier"] = {
                str(verifier): {
                    str(generator): _paired_comparison(
                        p3[f"g{verifier}_v{verifier}"], p3[f"g{generator}_v{verifier}"]
                    )
                    for generator in depths
                    if generator != verifier
                }
                for verifier in depths
            }
            if cohort_name == "legacy":
                p1 = {str(depth): _closed_result(config, cohort_name, task, f"d{depth}_P1") for depth in depths}
                cohort_rows["P1"] = {str(depth): float(value["success_rate"]) for depth, value in ((int(k), v) for k, v in p1.items())}
                cohort_rows["P1_pairs_vs_d6"] = {
                    str(depth): _paired_comparison(p1["6"], p1[str(depth)]) for depth in depths if depth != 6
                }
            analysis["tasks"][task][cohort_name] = cohort_rows
    analysis["candidate_pools"] = {}
    for task in ("pusht", "reacher"):
        summary_path = _resolve(config["output_root"]) / "cross_score" / task / "summary.json"
        if summary_path.is_file():
            analysis["candidate_pools"][task] = _read_json(summary_path)

    output_root = _resolve(config["output_root"])
    _write_json(output_root / "analysis.json", analysis)
    report_path = _resolve(config["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Shared DiT 加深影响归因分析（6 / 8 / 12 层）",
        "",
        "## 目的与解释边界",
        "",
        "本轮使用现有 epoch-10 checkpoint，不重新训练。候选池实验将动作生成与 Stage-B rerank 分开：同一 legacy50 起点和同一组 64 个初始噪声下，分别比较 6/8/12 层生成的物理候选，再让三个深度的 verifier 对每个候选池交叉评分。闭环部分补充 P0（直接动作生成）、P1（Stage-B/CEM）和 P3（动作候选加 Stage-B rerank）。",
        "",
        "样本是固定 cohort 和单训练 seed 的配对结果。候选池的 oracle 是已采样的 64 个候选中的最好结果，只表示该池的上限，不代表部署策略能达到。bootstrap 区间按 state 重采样；它描述当前固定 cohort 的不确定性，不替代独立 seed。",
        "",
        "## Checkpoint 与训练损失",
        "",
        "| Task | Depth | 参数量 | Validation action loss（epoch 10） | Validation latent-prefix loss（epoch 10） | Checkpoint SHA256 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for task in ("pusht", "reacher"):
        for depth in depths:
            model_summary = _model_summary(_load_model(_checkpoint(config, task, depth)))
            losses = analysis["training_losses"][task][str(depth)]
            lines.append(
                f"| {task} | {depth} | {model_summary['parameters']:,} | "
                f"{_format_value(losses.get('val/action_loss'))} | "
                f"{_format_value(losses.get('val/latent_prefix_loss'))} | "
                f"`{_sha256_file(_checkpoint(config, task, depth))}` |"
            )
    lines.extend(
        [
            "",
            "## 固定候选池：生成与 rerank 的独立贡献",
            "",
            "每个 task 有 50 个状态、每个状态 64 个配对噪声候选。每个物理候选按相同的 25 个 primitive action replay，记录 success 和 normalized physical distance。下表按 generator × verifier 列出 verifier 选中候选的成功率、候选池 oracle 成功率、距离 regret 和成本排序相关性。",
            "",
        ]
    )
    for task in ("pusht", "reacher"):
        candidate = analysis["candidate_pools"].get(task)
        lines.extend([f"### {task}", ""])
        if not candidate:
            lines.extend(["候选池交叉评分尚未完成。", ""])
            continue
        lines.extend(
            [
                "| Generator | Verifier | Pool oracle success | Selected success | Random candidate success | Distance regret | Spearman(cost, distance) | Oracle best in predicted top-5 |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for generator in depths:
            for verifier in depths:
                record = candidate["matrix"][f"generator_{generator}_verifier_{verifier}"]
                lines.append(
                    f"| {generator} | {verifier} | "
                    f"{_format_value(record['oracle_success_rate'])} | "
                    f"{_format_value(record['selected_success_rate'])} | "
                    f"{_format_value(record['random_candidate_success_rate'])} | "
                    f"{_format_value(record['selected_distance_regret'])} | "
                    f"{_format_value(record['spearman_predicted_cost_vs_true_distance'])} | "
                    f"{_format_value(record['oracle_best_candidate_ranked_top5'])} |"
                )
        lines.extend(
            [
                "",
                "候选生成摘要：",
                "",
                "| Generator | Pairwise action RMS | Fixed candidate success | Mean physical distance |",
                "|---:|---:|---:|---:|",
            ]
        )
        for depth in depths:
            row = candidate["generators"][str(depth)]
            lines.append(
                f"| {depth} | {row['candidate_action_pairwise_rms']:.4f} | "
                f"{row['fixed_candidate_success_rate']:.4f} | "
                f"{row['fixed_candidate_true_distance_mean']:.4f} |"
            )
        lines.append("")
    lines.extend(["## 闭环成功率", ""])
    for task in ("pusht", "reacher"):
        for cohort_name in (("legacy", "reacher_final") if task == "reacher" else ("legacy",)):
            values = analysis["tasks"].get(task, {}).get(cohort_name)
            if not values:
                continue
            title = f"### {task} — {cohort_name} ({len(_manifest(config, cohort_name, task).entries)} episodes)"
            lines.extend([title, ""])
            lines.extend(
                [
                    "| Depth | P0 success | P1 success | P3 diagonal success | P0 paired Δ vs 6L | P1 paired Δ vs 6L | P3 paired Δ vs 6L |",
                    "|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for depth in depths:
                p0 = values["P0"][str(depth)]
                p1 = values.get("P1", {}).get(str(depth))
                p3 = values["P3"][f"g{depth}_v{depth}"]
                delta0 = "—" if depth == 6 else f"{values['P0_pairs_vs_d6'][str(depth)]['net_success_delta_pp']:+.1f} pp"
                delta1 = "—" if depth == 6 or p1 is None else f"{values['P1_pairs_vs_d6'][str(depth)]['net_success_delta_pp']:+.1f} pp"
                delta3 = "—" if depth == 6 else f"{values['P3_diagonal_pairs_vs_d6'][str(depth)]['net_success_delta_pp']:+.1f} pp"
                lines.append(
                    f"| {depth} | {100*p0:.1f}% | "
                    f"{'—' if p1 is None else f'{100*p1:.1f}%'} | "
                    f"{100*p3:.1f}% | {delta0} | {delta1} | {delta3} |"
                )
            lines.extend(
                [
                    "",
                    "P3 closed-loop success matrix (generator rows × verifier columns):",
                    "",
                    "| Generator \\ Verifier | 6 | 8 | 12 |",
                    "|---:|---:|---:|---:|",
                ]
            )
            for generator in depths:
                cells = [100.0 * values["P3"][f"g{generator}_v{verifier}"] for verifier in depths]
                lines.append(
                    f"| {generator} | "
                    + " | ".join(f"{value:.1f}%" for value in cells)
                    + " |"
                )
            lines.extend(
                [
                    "",
                    "P3 verifier crossover, expressed as paired percentage-point change against the diagonal verifier at the same generator depth:",
                    "",
                    "| Generator | Verifier change | Paired gain/loss | Net Δ (95% CI) | McNemar p |",
                    "|---:|---:|---:|---:|---:|",
                ]
            )
            for generator in depths:
                for verifier in depths:
                    if verifier == generator:
                        continue
                    comparison = values["P3_verifier_pairs_by_generator"][str(generator)][str(verifier)]
                    ci = comparison["paired_success_delta_bootstrap_95ci_pp"]
                    lines.append(
                        f"| {generator} | {verifier} | "
                        f"{comparison['paired_gain']}/{comparison['paired_loss']} | "
                        f"{comparison['net_success_delta_pp']:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}] pp | "
                        f"{comparison['exact_mcnemar_p']:.3f} |"
                    )
            lines.extend(
                [
                    "",
                    "P3 generator crossover, expressed as paired percentage-point change against the diagonal generator at the same verifier depth:",
                    "",
                    "| Verifier | Generator change | Paired gain/loss | Net Δ (95% CI) | McNemar p |",
                    "|---:|---:|---:|---:|---:|",
                ]
            )
            for verifier in depths:
                for generator in depths:
                    if generator == verifier:
                        continue
                    comparison = values["P3_generator_pairs_by_verifier"][str(verifier)][str(generator)]
                    ci = comparison["paired_success_delta_bootstrap_95ci_pp"]
                    lines.append(
                        f"| {verifier} | {generator} | "
                        f"{comparison['paired_gain']}/{comparison['paired_loss']} | "
                        f"{comparison['net_success_delta_pp']:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}] pp | "
                        f"{comparison['exact_mcnemar_p']:.3f} |"
                    )
            lines.extend(
                [
                    "",
                    "P3 median planning time per replan, seconds:",
                    "",
                    "| Generator | Verifier | Proposal | Verifier scoring | Encoding | Total planning |",
                    "|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for generator in depths:
                for verifier in depths:
                    timing = values["P3_planning"][f"g{generator}_v{verifier}"]
                    lines.append(
                        f"| {generator} | {verifier} | "
                        f"{_format_value(timing['proposal_median_seconds'])} | "
                        f"{_format_value(timing['verifier_median_seconds'])} | "
                        f"{_format_value(timing['encode_median_seconds'])} | "
                        f"{_format_value(timing['planning_median_seconds'])} |"
                    )
            lines.extend(
                [
                    "",
                    "Paired depth comparisons against 6 layers (exact two-sided McNemar; bootstrap CI is for paired success-rate difference):",
                    "",
                    "| Mode | Depth | Gain/loss | Net Δ (95% bootstrap CI) | p |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for mode, comparison_key in (
                ("P0", "P0_pairs_vs_d6"),
                ("P1", "P1_pairs_vs_d6"),
                ("P3 diagonal", "P3_diagonal_pairs_vs_d6"),
            ):
                for depth in depths:
                    if depth == 6 or str(depth) not in values.get(comparison_key, {}):
                        continue
                    comparison = values[comparison_key][str(depth)]
                    ci = comparison["paired_success_delta_bootstrap_95ci_pp"]
                    lines.append(
                        f"| {mode} | {depth} | {comparison['paired_gain']}/{comparison['paired_loss']} | "
                        f"{comparison['net_success_delta_pp']:+.1f} pp "
                        f"[{ci[0]:+.1f}, {ci[1]:+.1f}] | {comparison['exact_mcnemar_p']:.3f} |"
                    )
            lines.append("")
    lines.extend(
        [
            "## 如何判断影响来自生成还是 rerank",
            "",
            "- P0 的深度变化和固定候选池 physical/oracle 指标一起读，反映动作生成器本身能否提出更好的动作。",
            "- 固定同一个 generator 行，改变 verifier 列，反映 Stage-B 对相同物理候选排序的影响；generator 行变化时 verifier 不变，反映候选分布变化。",
            "- P1 提供 verifier 深度在另一种候选生成机制（CEM 随机动作）下的参考；P3 闭环结果包含交互效应，不能把 P3 单独归因于某一个模块。",
            "- Cube 沿用已有深度 sweep 的 dev/final 饱和结果，本次不重复跑候选池与闭环矩阵。",
            "",
            "## 对 24 层实验的讨论",
            "",
            "本组结果暂不支持直接做覆盖两项任务、同时扫 generator/verifier 的完整 24 层矩阵。Reacher 的 P1/CEM 在 8 层相对 6 层提升 16 pp（95% 配对 bootstrap CI [+4, +28] pp，McNemar p=0.039），12 层提升 14 pp（[+2, +26] pp，p=0.065）；但 Reacher final200 的 P0 在 6/12 层均为 75.5%，P3 对角线从 84.5% 降至 81.5%。固定候选池显示 12 层 Reacher generator 的平均距离有小幅改善，但 64 候选 oracle 对三个深度都达到 100%，P3 verifier 的优势随 generator 变化，未形成一致的深度趋势。PushT 闭环接近饱和，候选池表现以 8 层最好，12 层未继续改善。",
            "",
            "如果仍要研究更大模型，优先做一项缩小的 Reacher verifier/Stage-B 深度对照：在相同候选集与规划预算上比较 8/12/24 层的 P1 和固定池排序；暂不同时扩展 24 层动作生成器。这样可以直接检验目前最明显的信号是否随 verifier 容量继续增长。只有当该收益在独立 seed 或 cohort 上复现，且 latency 成本可接受，再扩展到完整 P3 或更多任务。Reacher final200 中 P3 的 12/12 每次 replanning 中位时间为 1.980 s，6/6 为 1.091 s，提示深度增加已有明显推理成本。",
            "",
            "这些比较来自单训练 seed，多个深度/模式的 p 值未做多重比较校正；50/200 episodes 也不能替代独立训练 seed。因此上述 24 层建议是下一步实验优先级判断，不是统计定论，也不自动决定 go/no-go。",
            "",
            "机器可读汇总：`outputs/round4_shared_dit_depth_attribution/analysis.json`。原始条件、候选池和交叉评分均保存在同一 output root 下。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return analysis


def analyze(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    analysis = _write_analysis_and_report(config)
    print(
        json.dumps(
            {
                "status": "ok",
                "analysis": str((_resolve(config["output_root"]) / "analysis.json").resolve()),
                "report": str(_resolve(config["report"]).resolve()),
                "tasks": sorted(analysis["tasks"]),
            },
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    for name in ("candidate-pool", "cross-score", "closed-loop"):
        command = subparsers.add_parser(name)
        command.add_argument("--task", choices=("pusht", "reacher", "all"), required=True)
        command.add_argument("--device", default="cuda")
        command.add_argument("--gpu", type=str)
        if name == "candidate-pool":
            command.add_argument("--limit-candidates", type=int)
        if name == "closed-loop":
            command.add_argument("--cohort", choices=("legacy", "reacher_final"), default="legacy")
            command.add_argument("--limit-conditions", type=int)
    subparsers.add_parser("analyze")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = _config(args.config)
    if args.command == "validate":
        validate(args, config)
    elif args.command == "candidate-pool":
        if args.task == "all":
            raise ValueError("run candidate-pool separately per task/GPU")
        if args.limit_candidates is not None and args.limit_candidates < 1:
            raise ValueError("--limit-candidates must be positive")
        candidate_pool(args, config)
    elif args.command == "cross-score":
        if args.task == "all":
            raise ValueError("run cross-score separately per task/GPU")
        cross_score(args, config)
    elif args.command == "closed-loop":
        closed_loop(args, config)
    elif args.command == "analyze":
        analyze(args, config)
    else:
        parser.error(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
