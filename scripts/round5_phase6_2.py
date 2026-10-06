#!/usr/bin/env python3
"""Run and analyze the Round 5 Phase 6.2 Reacher budget experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import round5_phase5_pre_report2 as phase5
from source.common.checkpoint import load_policy_or_model
from source.common.eval import EvaluationIdentity
from source.common.round3_phase1 import CohortManifest, wilson_interval
from source.common.round4_eval import run_round4_evaluation, validate_gpu_visibility


DEFAULT_CONFIG = ROOT / "config" / "round5" / "phase6_2.json"
DEFAULT_MIN_FREE_MIB = 12_000
BOOTSTRAP_SAMPLES = 10_000
TIMING_SOURCE_CONFIG = ROOT / "config" / "round5" / "phase5_pre_report2.json"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"experiment config must be a JSON object: {path}")
    return value


def _load_inputs(config: Mapping[str, Any]):
    return phase5._load_task_inputs(config, "reacher")


def _condition_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs = phase5._condition_specs(config, "reacher")
    if len(specs) != 76:
        raise ValueError(f"expected 76 unique Reacher conditions, got {len(specs)}")
    return specs


def _condition_key(spec: Mapping[str, Any]) -> tuple[int, int, str, str]:
    return (
        int(spec["execute_steps"]),
        int(spec["score_steps_env"]),
        str(spec["mode"]),
        str(spec["guidance"]),
    )


def _condition_path(output_root: Path, budget: int, spec: Mapping[str, Any]) -> Path:
    return (
        output_root
        / "conditions"
        / f"budget_{int(budget)}"
        / f"execute_{int(spec['execute_steps'])}"
        / f"score_{int(spec['score_steps_env'])}"
        / str(spec["mode"])
        / str(spec["guidance"])
        / "result.json"
    )


def _budget_config(config: Mapping[str, Any], budget: int) -> dict[str, Any]:
    result = dict(config)
    result["evaluation"] = {
        **dict(config["evaluation"]),
        "eval_budget": int(budget),
    }
    return result


def _code_identity(config_path: Path) -> dict[str, Any]:
    files = (
        config_path,
        ROOT / "scripts" / "round5_phase6_2.py",
        ROOT / "source" / "common" / "round4_eval.py",
        ROOT / "source" / "common" / "round3_eval.py",
        ROOT / "source" / "common" / "round3_phase1.py",
        ROOT / "source" / "common" / "round3_validation.py",
        ROOT / "source" / "common" / "round4_protocol.py",
        ROOT / "source" / "common" / "round4_action_bounds.py",
        ROOT / "source" / "common" / "eval.py",
        ROOT / "source" / "policy" / "round4.py",
        ROOT / "source" / "policy" / "fast_lewam_eval.py",
        ROOT / "source" / "model" / "fast_lewam" / "jepa.py",
        ROOT / "source" / "model" / "fast_lewam" / "round4.py",
        ROOT / "config" / "eval" / "reacher.yaml",
        ROOT / "config" / "eval" / "launcher" / "local.yaml",
        ROOT / "config" / "eval" / "solver" / "cem.yaml",
    )
    hashes = {str(path.relative_to(ROOT)): _sha256_file(path) for path in files}
    packages = ("torch", "stable-worldmodel", "stable-pretraining", "gymnasium", "mujoco")
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    identity = {
        "file_sha256": hashes,
        "runtime_versions": {"python": sys.version.split()[0], **versions},
    }
    return {
        **identity,
        "bundle_sha256": _canonical_sha256(identity),
    }


def _result_metadata(
    config: Mapping[str, Any],
    config_path: Path,
    manifest: CohortManifest,
    checkpoint_sha: str,
    budget: int,
    spec: Mapping[str, Any],
    gpu: str,
    gpu_before: Mapping[str, Any],
    gpu_after: Mapping[str, Any],
) -> dict[str, Any]:
    condition = phase5._expected_condition(spec)
    return {
        "schema_version": "round5_phase6_2_v1",
        "budget": int(budget),
        "condition": condition,
        "condition_sha256": _canonical_sha256(condition),
        "checkpoint_sha256": checkpoint_sha,
        "cohort_id": manifest.cohort_id,
        "cohort_sha256": manifest.computed_sha256,
        "config_sha256": _sha256_file(config_path),
        "code_identity": _code_identity(config_path),
        "seed_protocol": {
            "environment_seed": int(config["evaluation"]["seed"]),
            "policy_seed": int(config["evaluation"]["seed"]),
            "policy_rng": "Round4 seeded generator; same batch size and condition across budgets",
            "prefix_validation": "Compare per-episode action and state traces through step 50; require unchanged first success for successful episodes.",
        },
        "gpu": {
            "physical_id": int(gpu),
            "before": dict(gpu_before),
            "after": dict(gpu_after),
        },
        "source_refs": {
            "plan": "docs/plan/round5_phase6_2_plan.md",
            "config": str(config_path.relative_to(ROOT)),
            "checkpoint": str(_resolve(config["tasks"]["reacher"]["checkpoint"]).relative_to(ROOT)),
            "cohort": str(_resolve(config["tasks"]["reacher"]["cohort"]).relative_to(ROOT)),
            "round4_evaluator": "source/common/round4_eval.py",
        },
    }


def _validate_phase6_result(
    payload: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    manifest: CohortManifest,
    checkpoint: Path,
    budget: int,
    spec: Mapping[str, Any],
    require_metadata: bool,
) -> None:
    phase5._validate_result(
        payload,
        task="reacher",
        config=_budget_config(config, budget),
        manifest=manifest,
        checkpoint=checkpoint,
        spec=spec,
        require_metadata=False,
    )
    metadata = payload.get("round5_phase6_2")
    if require_metadata:
        if not isinstance(metadata, Mapping):
            raise ValueError("Phase 6.2 result is missing experiment metadata")
        if metadata.get("schema_version") != "round5_phase6_2_v1":
            raise ValueError("Phase 6.2 result has an unknown metadata schema")
        if int(metadata.get("budget", -1)) != int(budget):
            raise ValueError("Phase 6.2 result budget metadata differs")
        if metadata.get("condition") != phase5._expected_condition(spec):
            raise ValueError("Phase 6.2 result condition metadata differs")
        if metadata.get("checkpoint_sha256") != config["tasks"]["reacher"]["checkpoint_sha256"]:
            raise ValueError("Phase 6.2 result checkpoint metadata differs")
        if metadata.get("cohort_sha256") != manifest.computed_sha256:
            raise ValueError("Phase 6.2 result cohort metadata differs")


def _run_one(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    manifest: CohortManifest,
    checkpoint: Path,
    checkpoint_sha: str,
    model: Any,
    gpu: str,
    device: str,
    budget: int,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    result_path = _condition_path(output_root, budget, spec)
    target = result_path.parent
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        saved_metadata = payload.get("round5_phase6_2")
        current_bundle = _code_identity(config_path)["bundle_sha256"]
        saved_bundle = (
            saved_metadata.get("code_identity", {}).get("bundle_sha256")
            if isinstance(saved_metadata, Mapping)
            else None
        )
        if saved_bundle != current_bundle:
            print(
                json.dumps(
                    {
                        "status": "recompute_stale_code_identity",
                        "budget": budget,
                        "condition": phase5._expected_condition(spec),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            shutil.rmtree(target)
        else:
            _validate_phase6_result(
                payload,
                config=config,
                manifest=manifest,
                checkpoint=checkpoint,
                budget=budget,
                spec=spec,
                require_metadata=True,
            )
            print(json.dumps({"status": "resumed", "budget": budget, "condition": phase5._expected_condition(spec)}), flush=True)
            return payload
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"partially populated condition directory: {target}")

    gpu_before = phase5._gpu_preflight(gpu, DEFAULT_MIN_FREE_MIB)
    cfg = phase5._compose("reacher", manifest, device, config)
    cfg.eval.eval_budget = int(budget)
    cfg.world.max_episode_steps = 2 * int(budget)
    ev = config["evaluation"]
    guidance = config["guidance"]
    identity = EvaluationIdentity(
        entrypoint="round5_phase6_2",
        policy_kind="round4_shared_dit",
        checkpoint=str(checkpoint.resolve()),
        epoch=10,
        stage=str(spec["mode"]),
    )
    payload = run_round4_evaluation(
        cfg,
        task="reacher",
        policy_or_model=model,
        mode=str(spec["mode"]),
        identity=identity,
        manifest=manifest,
        output_dir=target,
        trace_output_dir=target / "trace",
        device=device,
        trace=True,
        candidate_count=int(ev["best_of_n_candidates"]),
        flow_steps=16,
        action_flow_steps=spec["action_flow_steps"],
        solver_batch_size=int(ev["solver_batch_size"]),
        candidate_batch_size=int(ev["candidate_batch_size"]),
        action_flow_integrator=str(spec["action_flow_integrator"]),
        cem_protocol=str(spec["cem_protocol"]),
        guidance_mode=str(spec["guidance"]),
        guidance_step_size=float(guidance["step_size"]),
        guidance_last_steps=int(guidance["last_steps"]),
        guidance_inner_steps=int(guidance["inner_steps"]),
        guidance_max_rms_offset=float(guidance["max_rms_offset"]),
        proposal_chunk_size=int(guidance["p3_proposal_chunk_size"]),
        allowed_protocol_variants=("legacy",),
        execute_steps=int(spec["execute_steps"]),
        score_horizon_blocks=int(spec["score_horizon_blocks"]),
        allow_eval_budget_override=True,
    )
    if payload.get("status") != "ok":
        raise RuntimeError(f"evaluator returned unexpected status {payload.get('status')!r}")
    gpu_after = phase5._gpu_snapshot(gpu)
    payload["round5_phase6_2"] = _result_metadata(
        config,
        config_path,
        manifest,
        checkpoint_sha,
        budget,
        spec,
        gpu,
        gpu_before,
        gpu_after,
    )
    phase5._atomic_write_json(result_path, payload)
    _validate_phase6_result(
        payload,
        config=config,
        manifest=manifest,
        checkpoint=checkpoint,
        budget=budget,
        spec=spec,
        require_metadata=True,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "budget": budget,
                "condition": phase5._expected_condition(spec),
                "success_rate": payload.get("success_rate"),
                "evaluation_seconds": payload.get("evaluation_seconds"),
                "result": str(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


def validate(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    manifest, manifest_path, checkpoint, checkpoint_sha = _load_inputs(config)
    if checkpoint_sha != config["tasks"]["reacher"]["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA256 differs from frozen config")
    specs = _condition_specs(config)
    output_root = _resolve(config["output_root"])
    reused = _load_phase5_budget50(config, output_root, specs, manifest, checkpoint)
    reference_index = _budget50_reference_index(config, output_root, specs, reused)
    _write_json(output_root / "analysis" / "budget50_sources.json", {"conditions": reference_index})
    timings = _load_phase5_timings(config)
    result = {
        "status": "validated",
        "task": "reacher",
        "checkpoint_sha256": checkpoint_sha,
        "cohort_sha256": manifest.computed_sha256,
        "cohort_entries": len(manifest.entries),
        "unique_conditions_per_budget": len(specs),
        "budget50_reused_conditions": len(reused),
        "fixed_input_timing_records": len(timings),
        "config_sha256": _sha256_file(config_path),
        "cohort_path": str(manifest_path),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def _budget50_reference_index(
    config: Mapping[str, Any],
    output_root: Path,
    specs: Sequence[Mapping[str, Any]],
    validated: Mapping[tuple[int, int, str, str], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a compact map of the validated budget-50 source artifacts."""
    if validated is None:
        index_path = output_root / "analysis" / "budget50_sources.json"
        if index_path.is_file():
            value = json.loads(index_path.read_text(encoding="utf-8"))
            rows = value.get("conditions", [])
            if len(rows) == 76:
                return rows
        manifest, _, checkpoint, _ = _load_inputs(config)
        validated = _load_phase5_budget50(config, output_root, specs, manifest, checkpoint)
    result = []
    for spec in specs:
        key = _condition_key(spec)
        item = validated[key]
        payload = item["payload"]
        result.append(
            {
                "execute_steps": key[0],
                "score_steps_env": key[1],
                "mode": key[2],
                "guidance": key[3],
                "source_path": item["source_path"],
                "source_kind": item["source_kind"],
                "trace_sha256": payload.get("trace_content_sha256", payload.get("trace_sha256")),
            }
        )
    return result


def run_worker(args: argparse.Namespace) -> None:
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    manifest, _, checkpoint, checkpoint_sha = _load_inputs(config)
    specs = _condition_specs(config)
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    if len(set(indices)) != len(indices) or any(index < 0 or index >= len(specs) for index in indices):
        raise ValueError("worker indices are duplicate or outside the condition list")
    if int(args.budget) not in {int(value) for value in config["evaluation"]["budgets"]}:
        raise ValueError(f"budget {args.budget} is not frozen in the config")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"
    validate_gpu_visibility(device)
    phase5._gpu_preflight(str(args.gpu), int(args.min_free_mib))
    model, resolved = load_policy_or_model(str(checkpoint))
    if resolved is not None and Path(resolved).resolve() != checkpoint.resolve():
        raise ValueError("checkpoint loader resolved different weights")
    if int(getattr(model, "action_horizon", -1)) != 5 or int(getattr(model, "action_dim", -1)) != 10:
        raise ValueError("loaded Reacher model has an unexpected action shape")
    output_root = _resolve(args.output_root or config["output_root"])
    for index in indices:
        _run_one(
            config=config,
            config_path=config_path,
            output_root=output_root,
            manifest=manifest,
            checkpoint=checkpoint,
            checkpoint_sha=checkpoint_sha,
            model=model,
            gpu=str(args.gpu),
            device=device,
            budget=int(args.budget),
            spec=specs[index],
        )


def _load_phase5_budget50(
    config: Mapping[str, Any],
    output_root: Path,
    specs: Sequence[Mapping[str, Any]],
    manifest: CohortManifest,
    checkpoint: Path,
) -> dict[tuple[int, int, str, str], dict[str, Any]]:
    """Validate the legacy 58-cell Phase5 matrix plus the 18-cell expansion."""
    prior_root = _resolve(config["tasks"]["reacher"]["existing_output_root"])
    old_by_key, old_analysis = phase5._reacher_existing(config, output_root)
    old_sources = {}
    for row in old_analysis["rows"]:
        key = (
            int(row["execute_steps"]),
            int(row["score_steps_applied_env"]),
            str(row["mode"]),
            str(row["guidance"]),
        )
        old_sources.setdefault(key, str(row.get("result_source", "phase5")))
    reused: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for spec in specs:
        key = _condition_key(spec)
        if key in old_by_key:
            payload, source_path = old_by_key[key]
            source_kind = old_sources.get(key, "phase5")
        else:
            source_path_obj = phase5._condition_path(prior_root, "reacher", spec)
            if not source_path_obj.is_file():
                raise FileNotFoundError(f"missing budget-50 Phase5 condition: {source_path_obj}")
            payload = json.loads(source_path_obj.read_text(encoding="utf-8"))
            phase5._validate_result(
                payload,
                task="reacher",
                config=_budget_config(config, 50),
                manifest=manifest,
                checkpoint=checkpoint,
                spec=spec,
                require_metadata=True,
            )
            source_path = str(source_path_obj.resolve())
            source_kind = "phase5_pre_report2_expansion"
        reused[key] = {
            "payload": payload,
            "source_path": source_path,
            "source_kind": source_kind,
        }
    if len(reused) != 76:
        raise ValueError(f"expected 76 validated budget-50 conditions, found {len(reused)}")
    return reused


def _timing_key(record: Mapping[str, Any]) -> tuple[int, str, str, int]:
    condition = record.get("condition", {})
    return (
        int(condition.get("score_steps_env", record.get("score_steps_env", -1))),
        str(condition.get("mode", "")),
        str(condition.get("guidance", "")),
        int(record.get("batch_size", -1)),
    )


def _load_phase5_timings(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Reuse identity-matched 10-warmup/50-run fixed-input measurements."""
    analysis_path = _resolve(config["tasks"]["reacher"]["existing_analysis"])
    old_analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    old = [dict(item) for item in old_analysis.get("fixed_input_timing", [])]
    timing_config = phase5._load_config(TIMING_SOURCE_CONFIG)
    prior_root = _resolve(timing_config["output_root"])
    records = {}
    expected_specs = phase5._timing_specs(config)
    for item in old:
        if item.get("task", "reacher") != "reacher":
            continue
        key = _timing_key(item)
        if key[0] != 25:
            continue
        records[key] = {**item, "source": "phase5_existing_full_score_timing"}
    for spec in expected_specs:
        for batch_size in (1, 50):
            key = (
                int(spec["score_steps_env"]),
                str(spec["mode"]),
                str(spec["guidance"]),
                int(batch_size),
            )
            if key in records:
                continue
            path = phase5._timing_path(prior_root, "reacher", spec, batch_size)
            if not path.is_file():
                raise FileNotFoundError(f"missing identity-matched timing record: {path}")
            record = json.loads(path.read_text(encoding="utf-8"))
            if (
                record.get("condition") != phase5._expected_condition(spec)
                or int(record.get("batch_size", -1)) != int(batch_size)
                or int(record.get("warmup", -1)) != 10
                or int(record.get("runs", -1)) != 50
                or record.get("checkpoint_sha256") != config["tasks"]["reacher"]["checkpoint_sha256"]
                or record.get("cohort_sha256") != config["tasks"]["reacher"]["cohort_sha256"]
            ):
                raise ValueError(f"fixed-input timing identity mismatch: {path}")
            records[key] = {**record, "source": "phase5_pre_report2_fixed_input_timing"}
    expected_count = 56
    if len(records) != expected_count:
        raise ValueError(f"expected {expected_count} fixed-input timing records, found {len(records)}")
    return [records[key] for key in sorted(records)]


def _trace_prefix_mismatches(
    short: Mapping[str, Any], long: Mapping[str, Any], *, prefix_steps: int = 50
) -> list[dict[str, Any]]:
    """Compare per-episode action/state prefixes and preserve first-success time."""
    short_eps = short.get("episodes", [])
    long_eps = long.get("episodes", [])
    if len(short_eps) != len(long_eps):
        return [{"reason": "episode_count_mismatch"}]
    mismatches = []
    for left, right in zip(short_eps, long_eps):
        episode = str(left.get("episode_id", left.get("dataset_episode")))
        if str(right.get("episode_id", right.get("dataset_episode"))) != episode:
            mismatches.append({"episode_id": episode, "reason": "episode_identity_mismatch"})
            continue
        left_success = left.get("first_success_step")
        right_success = right.get("first_success_step")
        if bool(left.get("success")) and left_success != right_success:
            mismatches.append(
                {
                    "episode_id": episode,
                    "reason": "first_success_step_mismatch",
                    "budget50": left_success,
                    "budget125": right_success,
                }
            )
        steps_to_compare = int(prefix_steps)
        if bool(left.get("success")) and left_success is not None:
            steps_to_compare = min(steps_to_compare, int(left_success))
        left_steps = left.get("steps", [])[:steps_to_compare]
        right_steps = right.get("steps", [])[:steps_to_compare]
        if len(left_steps) != len(right_steps):
            mismatches.append(
                {
                    "episode_id": episode,
                    "reason": "prefix_length_mismatch",
                    "budget50_steps": len(left_steps),
                    "budget125_steps": len(right_steps),
                }
            )
            continue
        for step_index, (left_step, right_step) in enumerate(zip(left_steps, right_steps), start=1):
            for field in ("current", "goal", "action"):
                a = left_step.get(field)
                b = right_step.get(field)
                if a is None or b is None:
                    if a != b:
                        mismatches.append(
                            {"episode_id": episode, "step": step_index, "field": field, "reason": "missing_value_mismatch"}
                        )
                        break
                else:
                    aa = np.asarray(a, dtype=np.float64)
                    bb = np.asarray(b, dtype=np.float64)
                    if aa.shape != bb.shape or not np.allclose(aa, bb, rtol=1e-7, atol=1e-7, equal_nan=True):
                        mismatches.append(
                            {"episode_id": episode, "step": step_index, "field": field, "reason": "prefix_value_mismatch"}
                        )
                        break
            if mismatches and mismatches[-1].get("episode_id") == episode:
                break
    return mismatches


def _paired_cluster_bootstrap(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    left_by_key = {
        (str(item.get("episode_id", item.get("dataset_episode"))), int(item.get("start_step", 0))): bool(item.get("success"))
        for item in left
    }
    right_by_key = {
        (str(item.get("episode_id", item.get("dataset_episode"))), int(item.get("start_step", 0))): bool(item.get("success"))
        for item in right
    }
    if set(left_by_key) != set(right_by_key):
        raise ValueError("paired bootstrap inputs have different source episodes")
    keys = sorted(left_by_key)
    differences = np.asarray(
        [float(right_by_key[key]) - float(left_by_key[key]) for key in keys],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(differences), size=(BOOTSTRAP_SAMPLES, len(differences)))
    draws = differences[indices].mean(axis=1)
    return {
        "left_minus_right": False,
        "estimate": float(differences.mean()),
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "pairs": int(len(differences)),
        "discordant_left_fail_right_success": int(np.sum(differences > 0)),
        "discordant_left_success_right_fail": int(np.sum(differences < 0)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": int(seed),
    }


def _first_success_step(episode: Mapping[str, Any]) -> int | None:
    value = episode.get("first_success_step")
    if value is not None:
        return int(value)
    observed = [
        int(step["raw_env_step"])
        for step in episode.get("steps", [])
        if step.get("predicate_success") is True or step.get("env_success") is True
    ]
    return min(observed) if observed else None


def _episode_replan_count(episode: Mapping[str, Any], execute_steps: int) -> int:
    value = episode.get("episode_replan_count")
    if value is not None:
        return int(value)
    # Legacy Phase5 traces predate per-slot event counts. One call starts each
    # action block, including the initial plan, so the trace length determines
    # the exact count when every environment step uses the frozen block size.
    steps = int(episode.get("steps_executed", 0))
    return int(math.ceil(steps / int(execute_steps))) if steps else 0


def _episode_planning_value(episode: Mapping[str, Any], field: str) -> float | None:
    if episode.get(field) is not None:
        return float(episode[field])
    # Legacy Phase5 traces copied the batch-wide planning summary into every
    # episode record. Do not mistake those repeated aggregates for per-episode
    # observations; they remain missing until the Phase 6.2 runner records them.
    return None


def _mean_optional(values: Sequence[float | None]) -> float | None:
    observed = [float(value) for value in values if value is not None]
    return float(np.mean(observed)) if observed else None


def _matrix_rows(
    config: Mapping[str, Any],
    results: Mapping[tuple[int, int, int, str, str], Mapping[str, Any]],
    timing_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    timing_by_key = {_timing_key(item): item for item in timing_records}
    rows = []
    episode_rows = []
    for (budget, execute, score, mode, guidance), item in sorted(results.items()):
        payload = item["payload"]
        episodes = payload["episodes"]
        successes = [bool(ep.get("success")) for ep in episodes]
        count = int(sum(successes))
        low, high = wilson_interval(count, len(episodes))
        evaluation_seconds = float(payload.get("evaluation_seconds", payload.get("evaluation_time_seconds", 0.0)))
        replan_counts = [_episode_replan_count(ep, execute) for ep in episodes]
        success_steps = [_first_success_step(ep) for ep in episodes if ep.get("success")]
        final_distances = [
            float(ep["terminal_distance"])
            for ep in episodes
            if not ep.get("success") and ep.get("terminal_distance") is not None
        ]
        amortized_times = [_episode_planning_value(ep, "episode_amortized_planning_seconds") for ep in episodes]
        exposed_times = [_episode_planning_value(ep, "episode_planning_wall_seconds_exposed") for ep in episodes]
        stage_a = [int(_episode_planning_value(ep, "episode_stage_a_forward_count")) for ep in episodes]
        stage_b = [int(_episode_planning_value(ep, "episode_stage_b_forward_count")) for ep in episodes]
        timing = timing_by_key.get((score, mode, guidance, 50))
        matrix_row = {
            "budget": budget,
            "execute_steps": execute,
            "score_steps_env": score,
            "mode": mode,
            "guidance": guidance,
            "n": len(episodes),
            "successes": count,
            "success_rate": count / len(episodes),
            "wilson_95_low": float(low),
            "wilson_95_high": float(high),
            "evaluation_seconds": evaluation_seconds,
            "episode_steps_median": float(np.median([int(ep.get("steps_executed", 0)) for ep in episodes])),
            "episode_replan_calls_mean": float(np.mean(replan_counts)),
            "planning_batch_wall_seconds": float(payload.get("batch_planning_wall_seconds", 0.0)),
            "episode_planning_seconds_amortized_mean": _mean_optional(amortized_times),
            "episode_planning_seconds_exposed_mean": _mean_optional(exposed_times),
            "episode_stage_a_forward_count_mean": _mean_optional(stage_a),
            "episode_stage_b_forward_count_mean": _mean_optional(stage_b),
            "success_first_step_median": float(np.median(success_steps)) if success_steps else None,
            "failed_terminal_distance_median": float(np.median(final_distances)) if final_distances else None,
            "fixed_input_p50_batch50_seconds": None if timing is None else float(timing["p50_seconds"]),
            "source": item["source_kind"],
            "result_path": item["source_path"],
            "trace_sha256": payload.get("trace_content_sha256", payload.get("trace_sha256")),
        }
        rows.append(matrix_row)
        for episode in episodes:
            key = (str(episode.get("episode_id", episode.get("dataset_episode"))), int(episode.get("start_step", 0)))
            first_success = _first_success_step(episode)
            max_calls = int(math.ceil(budget / execute))
            replan_count = _episode_replan_count(episode, execute)
            reason = episode.get("episode_termination_reason")
            if reason is None:
                last_step = (episode.get("steps") or [{}])[-1]
                if last_step.get("terminated"):
                    reason = "environment_terminal"
                elif last_step.get("truncated"):
                    reason = "environment_truncated"
                elif int(episode.get("steps_executed", 0)) >= budget:
                    reason = "budget_exhausted"
                else:
                    reason = "rollout_incomplete"
            episode_rows.append(
                {
                    "budget": budget,
                    "execute_steps": execute,
                    "score_steps_env": score,
                    "mode": mode,
                    "guidance": guidance,
                    "episode_id": key[0],
                    "start_step": key[1],
                    "success": bool(episode.get("success")),
                    "first_success_step": first_success,
                    "first_success_planning_call": (
                        None if first_success is None else max(1, int(math.ceil(first_success / execute)))
                    ),
                    "planning_calls_observed": replan_count,
                    "planning_calls_possible": max_calls,
                    "planning_seconds_amortized": _episode_planning_value(episode, "episode_amortized_planning_seconds"),
                    "planning_seconds_exposed": _episode_planning_value(episode, "episode_planning_wall_seconds_exposed"),
                    "stage_a_forward_count": (
                        None
                        if _episode_planning_value(episode, "episode_stage_a_forward_count") is None
                        else int(_episode_planning_value(episode, "episode_stage_a_forward_count"))
                    ),
                    "stage_b_forward_count": (
                        None
                        if _episode_planning_value(episode, "episode_stage_b_forward_count") is None
                        else int(_episode_planning_value(episode, "episode_stage_b_forward_count"))
                    ),
                    "environment_terminated": bool(episode.get("environment_terminated", reason == "environment_terminal")),
                    "environment_truncated": bool(episode.get("environment_truncated", reason == "environment_truncated")),
                    "budget_exhausted": bool(episode.get("budget_exhausted", reason == "budget_exhausted")),
                    "termination_reason": reason,
                    "steps_executed": int(episode.get("steps_executed", 0)),
                    "terminal_distance": episode.get("terminal_distance"),
                    "source_path": item["source_path"],
                }
            )
    return rows, episode_rows


def _comparison(
    results: Mapping[tuple[int, int, int, str, str], Mapping[str, Any]],
    left_key: tuple[int, int, int, str, str],
    right_key: tuple[int, int, int, str, str],
    *,
    comparison: str,
    seed: int,
) -> dict[str, Any]:
    left = results[left_key]["payload"]["episodes"]
    right = results[right_key]["payload"]["episodes"]
    stats = _paired_cluster_bootstrap(left, right, seed=seed)
    return {
        "comparison": comparison,
        "left": {"budget": left_key[0], "execute_steps": left_key[1], "score_steps_env": left_key[2], "mode": left_key[3], "guidance": left_key[4]},
        "right": {"budget": right_key[0], "execute_steps": right_key[1], "score_steps_env": right_key[2], "mode": right_key[3], "guidance": right_key[4]},
        "left_success_rate": float(np.mean([bool(ep["success"]) for ep in left])),
        "right_success_rate": float(np.mean([bool(ep["success"]) for ep in right])),
        "paired_difference_right_minus_left": stats["estimate"],
        "paired_bootstrap_ci95": stats["ci95"],
        "pairs": stats["pairs"],
        "discordant_left_fail_right_success": stats["discordant_left_fail_right_success"],
        "discordant_left_success_right_fail": stats["discordant_left_success_right_fail"],
        "bootstrap_samples": stats["bootstrap_samples"],
        "seed": stats["seed"],
    }


def _make_comparisons(
    results: Mapping[tuple[int, int, int, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    comparisons = []
    p3_none = ("P3", "none")
    primary_left = (50, 25, 25, *p3_none)
    primary_right = (125, 25, 25, *p3_none)
    reference_left = (50, 10, 10, *p3_none)
    if primary_left in results and primary_right in results:
        comparisons.append(_comparison(results, primary_left, primary_right, comparison="primary_P3_none_25_25_budget125_vs_budget50", seed=62001))
    if reference_left in results and primary_right in results:
        comparisons.append(_comparison(results, reference_left, primary_right, comparison="key_reference_P3_none_25_25_budget125_vs_10_10_budget50", seed=62002))

    for budget in (50, 125):
        for execute, scores in ((10, (10, 25)), (5, (5, 10, 25)), (1, (5, 25))):
            for mode, guidance in phase5.MODES:
                if execute == 5 and mode == "P0" and guidance == "none":
                    continue
                if execute == 1 and mode == "P0" and guidance == "none":
                    continue
                for left_score, right_score in zip(scores, scores[1:]):
                    left_key = (budget, execute, left_score, mode, guidance)
                    right_key = (budget, execute, right_score, mode, guidance)
                    if left_key in results and right_key in results:
                        comparisons.append(
                            _comparison(
                                results,
                                left_key,
                                right_key,
                                comparison=f"fixed_execute_score_{budget}_{execute}_{left_score}_vs_{right_score}_{mode}_{guidance}",
                                seed=62100 + budget + execute + left_score + right_score,
                            )
                        )
        score_values = sorted({key[2] for key in results if key[0] == budget})
        for score in score_values:
            by_policy: dict[tuple[str, str], list[int]] = {}
            for key in results:
                if key[0] == budget and key[2] == score:
                    by_policy.setdefault((key[3], key[4]), []).append(key[1])
            for (mode, guidance), executes in by_policy.items():
                executes = sorted(set(executes))
                for left_execute, right_execute in zip(executes, executes[1:]):
                    left_key = (budget, left_execute, score, mode, guidance)
                    right_key = (budget, right_execute, score, mode, guidance)
                    comparisons.append(
                        _comparison(
                            results,
                            left_key,
                            right_key,
                            comparison=f"fixed_score_execute_{budget}_{score}_{left_execute}_vs_{right_execute}_{mode}_{guidance}",
                            seed=62200 + budget + score + left_execute + right_execute,
                        )
                    )
    return comparisons


def _curves(
    results: Mapping[tuple[int, int, int, str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    env_curve = []
    call_curve = []
    for key, item in sorted(results.items()):
        budget, execute, score, mode, guidance = key
        episodes = item["payload"]["episodes"]
        for step in (10, 25, 50, 75, 100, 125):
            if step > budget:
                continue
            successes = sum(
                bool(ep.get("success"))
                and _first_success_step(ep) is not None
                and int(_first_success_step(ep)) <= step
                for ep in episodes
            )
            env_curve.append(
                {
                    "budget": budget,
                    "execute_steps": execute,
                    "score_steps_env": score,
                    "mode": mode,
                    "guidance": guidance,
                    "environment_steps": step,
                    "successes": int(successes),
                    "n": len(episodes),
                    "cumulative_success_rate": float(successes / len(episodes)),
                }
            )
        max_calls = int(math.ceil(budget / execute))
        for call_index in range(1, max_calls + 1):
            successes = 0
            censored = 0
            for episode in episodes:
                first_step = _first_success_step(episode)
                first_call = None if first_step is None else max(1, int(math.ceil(first_step / execute)))
                if first_call is not None and first_call <= call_index:
                    successes += 1
                elif _episode_replan_count(episode, execute) < call_index:
                    censored += 1
            call_curve.append(
                {
                    "budget": budget,
                    "execute_steps": execute,
                    "score_steps_env": score,
                    "mode": mode,
                    "guidance": guidance,
                    "planning_call_index": call_index,
                    "maximum_planning_calls": max_calls,
                    "successes_by_call": int(successes),
                    "n_full_cohort": len(episodes),
                    "cumulative_success_rate_full_cohort": float(successes / len(episodes)),
                    "censored_before_this_call": int(censored),
                }
            )
    return env_curve, call_curve


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value
                    for key, value in row.items()
                }
            )


def _write_json(path: Path, payload: Any) -> None:
    phase5._atomic_write_json(path, payload if isinstance(payload, Mapping) else {"rows": payload})


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    manifest, _, checkpoint, _ = _load_inputs(config)
    specs = _condition_specs(config)
    output_root = _resolve(args.output_root or config["output_root"])
    results: dict[tuple[int, int, int, str, str], dict[str, Any]] = {}
    missing = []
    for spec in specs:
        condition = _condition_key(spec)
        budget50_path = _condition_path(output_root, 50, spec)
        if not budget50_path.is_file():
            missing.append({"budget": 50, "condition": phase5._expected_condition(spec), "path": str(budget50_path)})
        else:
            payload = json.loads(budget50_path.read_text(encoding="utf-8"))
            _validate_phase6_result(
                payload,
                config=config,
                manifest=manifest,
                checkpoint=checkpoint,
                budget=50,
                spec=spec,
                require_metadata=True,
            )
            results[(50, *condition)] = {
                "payload": payload,
                "source_path": str(budget50_path.resolve()),
                "source_kind": "phase6_2_budget50_re_evaluation",
            }

        budget125_path = _condition_path(output_root, 125, spec)
        if not budget125_path.is_file():
            missing.append({"budget": 125, "condition": phase5._expected_condition(spec), "path": str(budget125_path)})
            continue
        payload125 = json.loads(budget125_path.read_text(encoding="utf-8"))
        _validate_phase6_result(
            payload125,
            config=config,
            manifest=manifest,
            checkpoint=checkpoint,
            budget=125,
            spec=spec,
            require_metadata=True,
        )
        results[(125, *condition)] = {
            "payload": payload125,
            "source_path": str(budget125_path.resolve()),
            "source_kind": "phase6_2_budget125",
        }
    analysis_dir = output_root / "analysis"
    _write_json(analysis_dir / "missing_conditions.json", {"missing": missing})
    if missing:
        print(json.dumps({"status": "incomplete", "missing_conditions": len(missing)}, flush=True))
        return {"status": "incomplete", "missing": missing}

    prefix_audit = []
    mismatched_conditions = []
    for spec in specs:
        key = _condition_key(spec)
        mismatches = _trace_prefix_mismatches(
            results[(50, *key)]["payload"], results[(125, *key)]["payload"]
        )
        item = {
            "condition": phase5._expected_condition(spec),
            "prefix_steps": 50,
            "match": not mismatches,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:20],
            "budget50_source": results[(50, *key)]["source_path"],
            "budget125_source": results[(125, *key)]["source_path"],
        }
        prefix_audit.append(item)
        if mismatches and results[(50, *key)]["source_kind"] != "phase6_2_budget50_re_evaluation":
            mismatched_conditions.append({"index": specs.index(spec), **item})
    _write_json(analysis_dir / "prefix_audit.json", {"conditions": prefix_audit})
    if mismatched_conditions:
        _write_json(analysis_dir / "budget50_reruns_required.json", {"conditions": mismatched_conditions})
        print(json.dumps({"status": "budget50_rerun_required", "conditions": len(mismatched_conditions)}, flush=True))
        return {"status": "budget50_rerun_required", "conditions": mismatched_conditions}

    timing_records = _load_phase5_timings(config)
    matrix, episode_costs = _matrix_rows(config, results, timing_records)
    comparisons = _make_comparisons(results)
    env_curve, call_curve = _curves(results)
    _write_json(analysis_dir / "matrix.json", {"rows": matrix})
    _write_csv(analysis_dir / "matrix.csv", matrix)
    _write_json(analysis_dir / "episode_costs.json", {"episodes": episode_costs})
    _write_csv(analysis_dir / "episode_costs.csv", episode_costs)
    _write_json(analysis_dir / "comparisons.json", {"comparisons": comparisons})
    _write_csv(analysis_dir / "comparisons.csv", comparisons)
    _write_json(analysis_dir / "environment_step_curves.json", {"curves": env_curve})
    _write_csv(analysis_dir / "environment_step_curves.csv", env_curve)
    _write_json(analysis_dir / "planning_call_curves.json", {"curves": call_curve})
    _write_csv(analysis_dir / "planning_call_curves.csv", call_curve)
    _write_json(analysis_dir / "fixed_input_timing.json", {"records": timing_records})
    _write_csv(analysis_dir / "fixed_input_timing.csv", timing_records)
    report = _render_report(config, matrix, comparisons, prefix_audit, timing_records)
    report_path = _resolve(config["report_output"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    result = {
        "status": "complete",
        "unique_conditions_per_budget": len(specs),
        "logical_cells": len(matrix),
        "episodes_per_budget": sum(row["n"] for row in matrix if row["budget"] == 50),
        "new_budget125_episodes": sum(row["n"] for row in matrix if row["budget"] == 125),
        "prefix_conditions_matched": sum(bool(item["match"]) for item in prefix_audit),
        "comparisons": len(comparisons),
        "fixed_input_timing_records": len(timing_records),
        "report": str(report_path.resolve()),
    }
    _write_json(analysis_dir / "analysis.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def _render_report(
    config: Mapping[str, Any],
    matrix: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    prefix_audit: Sequence[Mapping[str, Any]],
    timings: Sequence[Mapping[str, Any]],
) -> str:
    key_by_name = {item["comparison"]: item for item in comparisons}
    primary = key_by_name.get("primary_P3_none_25_25_budget125_vs_budget50")
    reference = key_by_name.get("key_reference_P3_none_25_25_budget125_vs_10_10_budget50")
    lines = [
        "# Round 5 Phase 6.2：Reacher 总执行预算与重规划机会预实验",
        "",
        "## 协议与身份",
        "",
        f"- 权重：`{config['tasks']['reacher']['checkpoint']}`，SHA256 `{config['tasks']['reacher']['checkpoint_sha256']}`。",
        f"- Cohort：`{config['tasks']['reacher']['cohort']}`，规范化 SHA256 `{config['tasks']['reacher']['cohort_sha256']}`，50 个固定起点。",
        "- 推理 seed=42；旧 Phase5 结果经权重/cohort/配置核验后作为前缀参照。旧 trace 把批次规划汇总复制到每个 episode，不能提供逐 episode 规划成本，因此预算50与125均由 Phase 6.2 runner 重评。",
        f"- 前缀审计：{sum(bool(item['match']) for item in prefix_audit)}/{len(prefix_audit)} 个条件在前50步动作和状态一致；成功 episode 首次成功步也保持一致。",
        "- 两个预算的 world max_episode_steps 分别设为预算的两倍，实际 evaluator eval_budget 分别为50和125；非整除尾块只执行剩余环境步数。",
        "- 固定输入计时为 Phase5 中身份匹配的 batch=1/50、预热10次、测量50次记录；环境执行时间和闭环累计规划成本使用本轮每个 episode trace。",
        "",
        "## 预注册主终点",
        "",
    ]
    if primary:
        lines.append(
            f"P3-none、25/25 的成功率从预算50的 {primary['left_success_rate']:.1%} "
            f"变为预算125的 {primary['right_success_rate']:.1%}；配对差值 "
            f"{primary['paired_difference_right_minus_left'] * 100:+.1f} 个百分点，"
            f"来源 episode 配对 bootstrap 95% CI "
            f"[{primary['paired_bootstrap_ci95'][0] * 100:+.1f}, {primary['paired_bootstrap_ci95'][1] * 100:+.1f}]。"
        )
    if reference:
        lines.append(
            f"关键参照：P3-none、25/25预算125 {reference['right_success_rate']:.1%}，"
            f"对比 P3-none、10/10预算50 {reference['left_success_rate']:.1%}；"
            f"配对差 {reference['paired_difference_right_minus_left'] * 100:+.1f} 个百分点，"
            f"95% CI [{reference['paired_bootstrap_ci95'][0] * 100:+.1f}, {reference['paired_bootstrap_ci95'][1] * 100:+.1f}]。"
        )
    lines.extend(
        [
            "",
            "## 完整矩阵",
            "",
            "| 预算 | 执行步/评分步 | 方法 | 成功数/50 | 成功率 Wilson 95% CI | 环境秒 | 平均实际规划调用 | 失败末端距离中位数 |",
            "|---:|:---:|:---|---:|:---:|---:|---:|---:|",
        ]
    )
    labels = {"none": "none", "guided_flow": "GF", "post_opt": "PO"}
    for row in matrix:
        low, high = row["wilson_95_low"], row["wilson_95_high"]
        distance = row["failed_terminal_distance_median"]
        lines.append(
            f"| {row['budget']} | {row['execute_steps']}/{row['score_steps_env']} | "
            f"{row['mode']}-{labels.get(row['guidance'], row['guidance'])} | "
            f"{row['successes']}/50 | {row['success_rate']:.1%} [{low:.1%}, {high:.1%}] | "
            f"{row['evaluation_seconds']:.1f} | {row['episode_replan_calls_mean']:.2f} | "
            f"{'' if distance is None else f'{distance:.4f}'} |"
        )
    lines.extend(
        [
            "",
            "## 累计成功曲线与规划成本",
            "",
            "环境步曲线列于 `outputs/round5/phase6_2/analysis/environment_step_curves.csv`，规划调用曲线列于 `outputs/round5/phase6_2/analysis/planning_call_curves.csv`。两类曲线均以完整50个 cohort 起点为分母；已成功 episode 作为吸收事件，未观察到后续规划机会的未成功 episode 单独计入截尾数。",
            "成本表 `outputs/round5/phase6_2/analysis/episode_costs.csv` 逐 episode 保存实际调用数、规划 wall-time 暴露、摊销规划时间、Stage A/B forward 数、终止原因和末端距离。批量并行推理中的 episode 暴露时间会重复计入各活跃 slot；摊销时间按本次调用的活跃 slot 数均分，因此两者含义不同。",
            "固定输入单次规划 p50/p95 见 `outputs/round5/phase6_2/analysis/fixed_input_timing.csv`；它不含环境 stepping，不能代替闭环运行时。",
            "",
            "## 解释边界",
            "",
            "预算125若帮助25/25追上10/10预算50，只能说明额外执行时间和规划机会能够补偿部分差距；25步反馈间隔和25步评分跨度没有改变。若未追上，结果只说明在预算125内补偿不足。评分跨度与执行频率比较属于矩阵内的探索性配对，不是完整析因设计。单 checkpoint、单推理 seed 和50个起点仍是探索性证据。",
            "",
        "预算50与125各运行76个独立条件，每条件50个 episode，共7,600个 episode 运行。所有条件结果、trace、source identity 与失败状态保留在配置所指的输出目录；矩阵和配对 bootstrap 明细见 `analysis/` 下 JSON/CSV。",
            "",
        ]
    )
    return "\n".join(lines)


def _gpu_id(value: str) -> str:
    if not value.isdigit() or int(value) not in range(8):
        raise argparse.ArgumentTypeError("GPU must be a physical ID in 0..7")
    return value


def audit_prefix(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    manifest, _, checkpoint, _ = _load_inputs(config)
    specs = _condition_specs(config)
    output_root = _resolve(args.output_root or config["output_root"])
    reference_rows = _budget50_reference_index(config, output_root, specs)
    references = {
        (
            int(row["execute_steps"]),
            int(row["score_steps_env"]),
            str(row["mode"]),
            str(row["guidance"]),
        ): row
        for row in reference_rows
    }
    requested = None
    if args.indices:
        requested = {int(value) for value in args.indices.split(",") if value.strip()}
        if any(index < 0 or index >= len(specs) for index in requested):
            raise ValueError("prefix-audit index is outside the condition list")
    rows = []
    reruns = []
    for index, spec in enumerate(specs):
        if requested is not None and index not in requested:
            continue
        path125 = _condition_path(output_root, 125, spec)
        if not path125.is_file():
            continue
        key = _condition_key(spec)
        own50 = _condition_path(output_root, 50, spec)
        if own50.is_file():
            path50 = own50
            source_kind = "phase6_2_budget50_re_evaluation"
        else:
            reference = references[key]
            path50 = Path(str(reference["source_path"]))
            if not path50.is_absolute():
                path50 = ROOT / path50
            source_kind = str(reference["source_kind"])
        left = json.loads(path50.read_text(encoding="utf-8"))
        right = json.loads(path125.read_text(encoding="utf-8"))
        params = left.get("parameters", {})
        if (
            int(params.get("eval_budget", -1)) != 50
            or params.get("cohort_sha256") != manifest.computed_sha256
            or str(Path(str(left.get("checkpoint", ""))).resolve()) != str(checkpoint.resolve())
        ):
            raise ValueError(f"budget-50 prefix source identity mismatch: {path50}")
        _validate_phase6_result(
            right,
            config=config,
            manifest=manifest,
            checkpoint=checkpoint,
            budget=125,
            spec=spec,
            require_metadata=True,
        )
        mismatches = _trace_prefix_mismatches(left, right)
        row = {
            "index": index,
            "condition": phase5._expected_condition(spec),
            "match": not mismatches,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:20],
            "budget50_source": str(path50.resolve()),
            "budget50_source_kind": source_kind,
            "budget125_source": str(path125.resolve()),
        }
        rows.append(row)
        if mismatches and source_kind != "phase6_2_budget50_re_evaluation":
            reruns.append(row)
    analysis_dir = output_root / "analysis"
    _write_json(analysis_dir / "prefix_smoke_audit.json", {"conditions": rows})
    _write_json(analysis_dir / "budget50_reruns_required.json", {"conditions": reruns})
    result = {
        "status": "prefix_match" if not reruns and all(row["match"] for row in rows) else "budget50_rerun_required",
        "conditions_checked": len(rows),
        "conditions_matched": sum(bool(row["match"]) for row in rows),
        "budget50_reruns_required": len(reruns),
        "mismatched_conditions": [row["index"] for row in reruns],
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def run_matrix(args: argparse.Namespace) -> None:
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    specs = _condition_specs(config)
    selected = (
        list(range(len(specs)))
        if not args.indices
        else [int(value) for value in args.indices.split(",") if value.strip()]
    )
    if len(set(selected)) != len(selected) or any(index < 0 or index >= len(specs) for index in selected):
        raise ValueError("matrix indices are duplicate or outside the condition list")
    gpu = str(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    phase5._gpu_preflight(gpu, int(args.min_free_mib))
    worker_count = max(1, min(int(args.workers), len(selected)))
    shards = [
        [selected[int(position)] for position in chunk]
        for chunk in np.array_split(np.arange(len(selected)), worker_count)
        if len(chunk)
    ]
    output_root = _resolve(args.output_root or config["output_root"])
    log_root = output_root / "logs" / f"budget_{int(args.budget)}"
    log_root.mkdir(parents=True, exist_ok=True)
    children = []
    for worker_index, shard in enumerate(shards):
        log_path = log_root / f"worker_{worker_index}_gpu{gpu}.log"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "round5_phase6_2.py"),
            "worker",
            "--config",
            str(config_path),
            "--budget",
            str(args.budget),
            "--gpu",
            gpu,
            "--indices",
            ",".join(str(index) for index in shard),
            "--output-root",
            str(output_root),
            "--min-free-mib",
            str(args.min_free_mib),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        children.append((worker_index, process, log_path, log))
        print(
            json.dumps(
                {"status": "worker_started", "worker": worker_index, "gpu": int(gpu), "indices": shard, "log": str(log_path)},
                sort_keys=True,
            ),
            flush=True,
        )
    failures = []
    for worker_index, process, log_path, log in children:
        code = process.wait()
        log.close()
        print(
            json.dumps(
                {"status": "worker_finished", "worker": worker_index, "gpu": int(gpu), "exit_code": code, "log": str(log_path)},
                sort_keys=True,
            ),
            flush=True,
        )
        if code != 0:
            failures.append({"worker": worker_index, "exit_code": code, "log": str(log_path)})
    if failures:
        raise RuntimeError(f"matrix workers failed: {failures}")
    print(json.dumps({"status": "matrix_complete", "budget": int(args.budget), "conditions": len(selected), "workers": worker_count}, sort_keys=True), flush=True)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate", help="audit all budget-50 references and fixed-input timings")
    _common(check)
    worker = commands.add_parser("worker", help="run one budget shard of the condition matrix")
    _common(worker)
    worker.add_argument("--budget", type=int, required=True)
    worker.add_argument("--gpu", type=_gpu_id, required=True)
    worker.add_argument("--indices", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75")
    worker.add_argument("--output-root")
    worker.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB)
    analyze_parser = commands.add_parser("analyze", help="validate completed traces and write analyses/report")
    _common(analyze_parser)
    analyze_parser.add_argument("--output-root")
    prefix = commands.add_parser("audit-prefix", help="compare budget-50 and budget-125 action/state prefixes")
    _common(prefix)
    prefix.add_argument("--output-root")
    prefix.add_argument("--indices", default="")
    matrix = commands.add_parser("matrix", help="run concurrent condition shards on one selected GPU")
    _common(matrix)
    matrix.add_argument("--budget", type=int, required=True)
    matrix.add_argument("--gpu", type=_gpu_id, required=True)
    matrix.add_argument("--workers", type=int, default=2)
    matrix.add_argument("--indices", default="")
    matrix.add_argument("--output-root")
    matrix.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        validate(args)
    elif args.command == "worker":
        run_worker(args)
    elif args.command == "analyze":
        analyze(args)
    elif args.command == "audit-prefix":
        audit_prefix(args)
    elif args.command == "matrix":
        run_matrix(args)


if __name__ == "__main__":
    main()
