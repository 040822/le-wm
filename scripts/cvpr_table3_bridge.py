#!/usr/bin/env python3
"""Run the Table 3 same-actor closed-loop Random-64 / LeWM bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.cvpr_table3_bridge import make_state_indexed_schedules
from source.common.eval import EvaluationIdentity, get_dataset
from source.common.round3_phase1 import CohortManifest
from source.common.round4_eval import run_round4_evaluation
from scripts import cvpr_table1


TASKS = ("tworoom", "pusht", "reacher", "cube")
SEEDS = (42, 100, 2026, 3407, 1234, 4444)
VARIANTS = ("random64", "lewm_rerank")
TABLE1_ROOT = ROOT / "outputs/cvpr/table1/v1"
TABLE3_ROOT = ROOT / "outputs/cvpr/table3/v1"
OUTPUT_ROOT = TABLE3_ROOT / "bridge"
SCHEDULE_NAME = "cvpr_table3_state_indexed_schedule_v1"
BOOTSTRAP_SEED = 20261005


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _preflight(gpu: int) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != str(gpu):
        raise RuntimeError(
            f"GPU {gpu} task requires CUDA_VISIBLE_DEVICES={gpu}, got {visible!r}"
        )
    if gpu not in {0, 1, 2, 3}:
        raise RuntimeError(f"GPU {gpu} is not approved for this task")
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.free,memory.total,utilization.gpu,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    fields = [item.strip() for item in result.split(",")]
    if len(fields) != 5:
        raise RuntimeError(f"unexpected nvidia-smi response: {result!r}")
    free_gib = int(fields[0]) / 1024
    if free_gib < 6.0:
        raise RuntimeError(f"GPU {gpu} has only {free_gib:.2f} GiB free")
    return {
        "gpu": gpu,
        "uuid": fields[3],
        "name": fields[4],
        "free_vram_gib": free_gib,
        "total_vram_gib": int(fields[1]) / 1024,
        "utilization_percent": int(fields[2]),
    }


def _actor_and_verifier(task: str, device: str):
    from source.common.checkpoint import load_policy_or_model
    from scripts.cvpr_table3_fixed_pool import _load_table1_lewm_verifier

    actor_dir = TABLE1_ROOT / "assets/cowm" / task / "checkpoints"
    actor_paths = sorted(actor_dir.glob("*.pt"))
    if len(actor_paths) != 1:
        raise RuntimeError(f"expected one frozen CoWM checkpoint in {actor_dir}")
    actor_wrapper, _resolved = load_policy_or_model(
        str(actor_paths[0]), cache_dir=str(ROOT / "data")
    )
    actor = getattr(actor_wrapper, "model", actor_wrapper).eval().requires_grad_(False)
    verifier, verifier_path, history_size = _load_table1_lewm_verifier(
        task, actor, device
    )
    return actor, verifier, actor_paths[0], verifier_path, history_size


def _source_refs(task: str, seed: int) -> dict[str, Any]:
    p3_parent = TABLE1_ROOT / "runs" / f"cowm_p3__main__{task}__seed_{seed}"
    references = []
    for attempt in sorted(p3_parent.glob("attempt_*")) if p3_parent.exists() else []:
        result_path = attempt / "result.json"
        trace_path = attempt / "episodes.jsonl"
        if not result_path.is_file() or not trace_path.is_file():
            continue
        result = _json(result_path)
        if result.get("status") == "ok" and len(result.get("episodes", [])) == 50:
            table_identity = result.get("cvpr_table1", {})
            if table_identity.get("trace_sha256") != _sha256(trace_path):
                continue
            references.append((attempt, result_path, trace_path, result))
    if not references:
        raise RuntimeError(f"accepted Table 1 P3 reference is missing for {task}/seed_{seed}")
    attempt, result_path, trace_path, result = references[-1]

    three_a = TABLE3_ROOT / "3a" / task / f"seed_{seed}"
    acceptance_path = three_a / "acceptance.json"
    if not acceptance_path.is_file() or _json(acceptance_path).get("status") != "pass":
        raise RuntimeError(f"accepted Table 3a source is missing for {task}/seed_{seed}")
    return {
        "table1_cowm_p3": {
            "attempt": str(attempt.relative_to(ROOT)),
            "result_sha256": _sha256(result_path),
            "trace_sha256": _sha256(trace_path),
            "success_rate": float(result["success_rate"]),
        },
        "table3a_source": {
            "root": str(three_a.relative_to(ROOT)),
            "acceptance_sha256": _sha256(acceptance_path),
            "fixed_pool_sha256": _json(three_a / "summary.json").get("fixed_pool_sha256"),
        },
    }


def _cell_config(task: str, seed: int):
    cell = {
        "task": task,
        "evaluation_seed": int(seed),
        "environment_seed": int(seed) + 10_000,
        "policy_seed": int(seed) + 20_000,
        "method_id": "cowm_p3__main",
        "cell_id": f"cowm_p3__main__{task}__seed_{seed}",
    }
    cfg = cvpr_table1._cell_config(cell)
    cfg.output.save_video = False
    return cell, cfg


def _next_attempt(parent: Path) -> Path:
    existing = []
    if parent.exists():
        for path in parent.glob("attempt_*"):
            try:
                existing.append(int(path.name.removeprefix("attempt_")))
            except ValueError:
                continue
    return parent / f"attempt_{max(existing, default=0) + 1:03d}"


def _capture_schedule_rows(variant: str, action_dim: int):
    rows: list[dict[str, Any]] = []

    def capture(payload: dict[str, Any]) -> None:
        keys = payload["event"]["state_indexed_schedule_keys"]
        candidates = payload["candidates"].detach().cpu().numpy()
        candidate_noise = payload["candidate_noise"].detach().cpu().numpy().reshape(
            len(keys), 64, candidates.shape[-2], action_dim
        )
        selected = payload["selected_indices"].detach().cpu().numpy().reshape(-1)
        if candidates.shape[:2] != (len(keys), 64) or len(selected) != len(keys):
            raise RuntimeError("bridge schedule audit received an unexpected batch shape")
        for row, (slot, replan_index) in enumerate(keys):
            pool = np.ascontiguousarray(candidates[row])
            noise = np.ascontiguousarray(candidate_noise[row])
            rows.append(
                {
                    "variant": variant,
                    "cohort_slot": int(slot),
                    "replan_index": int(replan_index),
                    "candidate_noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
                    "candidate_pool_sha256": hashlib.sha256(pool.tobytes()).hexdigest(),
                    "selected_index": int(selected[row]),
                }
            )

    return capture, rows


def _run_variant(
    *,
    variant: str,
    task: str,
    seed: int,
    gpu: int,
    cfg,
    actor,
    verifier,
    actor_path: Path,
    verifier_path: Path,
    verifier_history_size: int,
    dataset,
    manifest,
    noise_schedule,
    selection_schedule,
    pair_root: Path,
    common_identity: dict[str, Any],
):
    attempt = _next_attempt(pair_root / "runs" / variant)
    attempt.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, attempt / "resolved_config.yaml", resolve=True)
    capture, schedule_rows = _capture_schedule_rows(variant, int(actor.action_dim))
    lewm_variant = variant == "lewm_rerank"
    verifier_metadata = (
        {
            "name": "Table 1 LeWM scorer",
            "checkpoint": str(verifier_path.relative_to(ROOT)),
            "checkpoint_sha256": _sha256(verifier_path),
            "history_size": int(verifier_history_size),
            "selection_rule": "argmin_verifier",
        }
        if lewm_variant
        else None
    )
    identity = EvaluationIdentity(
        entrypoint="cvpr_table3_bridge",
        policy_kind="fast_lewam",
        checkpoint=str(actor_path),
        epoch=10,
        stage="stage_b",
        guidance_mode="none",
    )
    _write_json(
        attempt / "bridge_identity.json",
        {
            **common_identity,
            "variant": variant,
            "verifier": verifier_metadata,
            "preflight": _preflight(gpu),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "argv": sys.argv,
        },
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)
    torch.manual_seed(int(seed) + 20_000)
    np.random.seed((int(seed) + 20_000) % (2**32 - 1))
    result = run_round4_evaluation(
        cfg,
        task=task,
        policy_or_model=actor,
        verifier_policy_or_model=verifier if lewm_variant else None,
        verifier_metadata=verifier_metadata,
        mode="P3",
        identity=identity,
        manifest=manifest,
        output_dir=attempt,
        trace_output_dir=attempt,
        dataset=dataset,
        device="cuda:0",
        trace=True,
        candidate_count=64,
        flow_steps=2,
        action_flow_steps=2,
        solver_batch_size=1,
        candidate_batch_size=64,
        action_flow_integrator="euler",
        action_bound_mode="none",
        guidance_mode="none",
        proposal_chunk_size=512,
        allowed_protocol_variants=("legacy",),
        allow_solver_config_override=True,
        allow_evaluation_seed_override=True,
        allow_cohort_seed_mismatch=True,
        policy_seed=int(seed) + 20_000,
        diagnostic_callback=capture,
        candidate_noise_schedule=noise_schedule,
        selection_index_schedule=selection_schedule if variant == "random64" else None,
        selection_rule="random" if variant == "random64" else "argmin_verifier",
        execute_steps=25,
        score_horizon_blocks=5,
    )
    result_path = attempt / "result.json"
    trace_path = attempt / "episodes.jsonl"
    if not result_path.is_file() or not trace_path.is_file():
        raise RuntimeError(f"bridge evaluator did not publish result and trace: {attempt}")
    published = _json(result_path)
    if published.get("status") != "ok" or len(published.get("episodes", [])) != 50:
        raise RuntimeError(f"bridge result is incomplete: {attempt}")
    _write_jsonl(attempt / "schedule_audit.jsonl", schedule_rows)
    _write_json(
        attempt / "acceptance.json",
        {
            "status": "pass",
            "errors": [],
            "result_sha256": _sha256(result_path),
            "trace_sha256": _sha256(trace_path),
            "schedule_audit_sha256": _sha256(attempt / "schedule_audit.jsonl"),
            "episode_count": 50,
            "replan_record_count": len(schedule_rows),
        },
    )
    return attempt, published, schedule_rows


def run_pair(args) -> Path:
    task, seed, gpu = args.task, int(args.seed), int(args.gpu)
    preflight = _preflight(gpu)
    pair_root = OUTPUT_ROOT / task / f"seed_{seed}"
    pair_root.mkdir(parents=True, exist_ok=True)
    pair_summary_path = pair_root / "pair_summary.json"
    if pair_summary_path.is_file() and _json(pair_summary_path).get("status") == "pass":
        print(f"already accepted: {pair_root}", flush=True)
        return pair_summary_path

    common_identity = _source_refs(task, seed)
    cell, cfg = _cell_config(task, seed)
    cohort_path = TABLE1_ROOT / "cohorts" / task / f"seed_{seed}.json"
    manifest = CohortManifest.load(cohort_path)
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    actor, verifier, actor_path, verifier_path, verifier_history_size = (
        _actor_and_verifier(task, device="cuda:0")
    )
    policy_seed = int(seed) + 20_000
    noise_schedule, selection_schedule, schedule_metadata = make_state_indexed_schedules(
        task=task, evaluation_seed=seed, policy_seed=policy_seed
    )
    source_dataset = _json(TABLE3_ROOT / "3a" / task / f"seed_{seed}" / "frozen_config.json")
    cell_frozen = {
        "schema_version": 1,
        "experiment": "cvpr_table3_closed_loop_bridge_v1",
        "task": task,
        "evaluation_seed": seed,
        "environment_seed": int(cell["environment_seed"]),
        "policy_seed": policy_seed,
        "cohort_sha256": _sha256(cohort_path),
        "dataset_sha256": source_dataset["dataset"]["sha256"],
        "actor_checkpoint_sha256": _sha256(actor_path),
        "verifier_checkpoint_sha256": _sha256(verifier_path),
        "source_refs": common_identity,
        "protocol": {
            "candidate_count": 64,
            "action_flow_steps": 2,
            "integrator": "euler",
            "action_bound_mode": "none",
            "plan_horizon_blocks": 5,
            "action_block": 5,
            "execute_steps": 25,
            "eval_budget": 50,
            "variants": list(VARIANTS),
            "random_selection": "one uniform index per cohort slot and replan index",
            "candidate_noise_schedule": schedule_metadata,
            "lewm_history_size": int(verifier_history_size),
            "paired_noise_identity": "task, evaluation seed, policy seed, cohort slot, replan index",
            "trajectory_divergence": "candidate pools may differ after state divergence; shared noise schedule only",
            "first_replan_pool_equality_required": True,
        },
        "code_sha256": {
            "runner": _sha256(Path(__file__).resolve()),
            "schedule": _sha256(ROOT / "source/common/cvpr_table3_bridge.py"),
            "policy": _sha256(ROOT / "source/policy/round4.py"),
            "evaluator": _sha256(ROOT / "source/common/round4_eval.py"),
            "table3_fixed_pool_loader": _sha256(ROOT / "scripts/cvpr_table3_fixed_pool.py"),
            "lewm_stage_b_verifier": _sha256(ROOT / "source/policy/round5_phase1_7.py"),
        },
        "runtime": {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "logical_device": "cuda:0",
        },
    }
    frozen_path = pair_root / "frozen_config.json"
    if frozen_path.is_file() and _json(frozen_path) != cell_frozen:
        if pair_summary_path.is_file() and _json(pair_summary_path).get("status") == "pass":
            raise RuntimeError(f"accepted bridge frozen configuration changed: {frozen_path}")
        history_dir = pair_root / "frozen_config_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        previous = sorted(history_dir.glob("attempt_*.json"))
        archive_path = history_dir / f"attempt_{len(previous) + 1:03d}.json"
        archive_path.write_bytes(frozen_path.read_bytes())
        print(f"archived previous frozen config: {archive_path}", flush=True)
        _write_json(frozen_path, cell_frozen)
    if not frozen_path.is_file():
        _write_json(frozen_path, cell_frozen)

    attempted = []
    try:
        for variant in VARIANTS:
            attempt, _result, _schedule_rows = _run_variant(
                variant=variant,
                task=task,
                seed=seed,
                gpu=gpu,
                cfg=cfg,
                actor=actor,
                verifier=verifier,
                actor_path=actor_path,
                verifier_path=verifier_path,
                verifier_history_size=verifier_history_size,
                dataset=dataset,
                manifest=manifest,
                noise_schedule=noise_schedule,
                selection_schedule=selection_schedule,
                pair_root=pair_root,
                common_identity=common_identity,
            )
            attempted.append((variant, attempt, _result, _schedule_rows))

        by_variant = {
            variant: {
                (int(row["cohort_slot"]), int(row["replan_index"])): row
                for row in rows
            }
            for variant, _attempt, _result, rows in attempted
        }
        random_rows, lewm_rows = by_variant["random64"], by_variant["lewm_rerank"]
        shared_keys = sorted(set(random_rows) & set(lewm_rows))
        if not shared_keys:
            raise RuntimeError("the paired bridge variants have no shared scheduled replans")
        noise_mismatches = [
            key
            for key in shared_keys
            if random_rows[key]["candidate_noise_sha256"]
            != lewm_rows[key]["candidate_noise_sha256"]
        ]
        if noise_mismatches:
            raise RuntimeError(f"paired candidate-noise schedules differ: {noise_mismatches[:5]}")
        unexpected_random_indices = []
        for key, row in random_rows.items():
            expected_index = int(
                selection_schedule(
                    (key,), num_candidates=64, device="cpu"
                )[0]
            )
            if int(row["selected_index"]) != expected_index:
                unexpected_random_indices.append(key)
        if unexpected_random_indices:
            raise RuntimeError(
                "Random-64 did not use the frozen state-indexed selections: "
                f"{unexpected_random_indices[:5]}"
            )
        first_keys = [key for key in shared_keys if key[1] == 0]
        pool_mismatches = [
            key
            for key in first_keys
            if random_rows[key]["candidate_pool_sha256"]
            != lewm_rows[key]["candidate_pool_sha256"]
        ]
        if {key[0] for key in first_keys} != set(range(50)) or pool_mismatches:
            raise RuntimeError(
                "the two bridge policies did not reproduce the same first-replan pool: "
                f"{pool_mismatches[:5]}"
            )
        outcomes = {}
        for variant, _attempt, result, rows in attempted:
            outcomes[variant] = {
                "success_rate": float(result["success_rate"]),
                "episode_count": len(result["episodes"]),
                "attempt": str(_attempt.relative_to(ROOT)),
                "result_sha256": _sha256(_attempt / "result.json"),
                "trace_sha256": _sha256(_attempt / "episodes.jsonl"),
                "schedule_audit_sha256": _sha256(_attempt / "schedule_audit.jsonl"),
                "state_replan_count": len(rows),
            }
        _write_json(
            pair_summary_path,
            {
                "status": "pass",
                "task": task,
                "evaluation_seed": seed,
                "cohort_sha256": _sha256(cohort_path),
                "shared_schedule_key_count": len(shared_keys),
                "candidate_noise_mismatch_count": 0,
                "first_replan_state_count": len(first_keys),
                "first_replan_pool_mismatch_count": 0,
                "variants": outcomes,
                "source_refs": common_identity,
            },
        )
        return pair_summary_path
    except Exception as exc:
        failures = pair_root / "failures.jsonl"
        with failures.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"error": repr(exc), "traceback": traceback.format_exc()},
                    ensure_ascii=False,
                )
                + "\n"
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", choices=SEEDS, required=True, type=int)
    parser.add_argument("--gpu", choices=(0, 1, 2, 3), required=True, type=int)
    args = parser.parse_args()
    try:
        path = run_pair(args)
    except Exception as exc:
        pair_root = OUTPUT_ROOT / args.task / f"seed_{args.seed}"
        pair_root.mkdir(parents=True, exist_ok=True)
        failures = pair_root / "failures.jsonl"
        already_recorded = False
        if failures.is_file():
            lines = [line for line in failures.read_text(encoding="utf-8").splitlines() if line]
            if lines:
                already_recorded = json.loads(lines[-1]).get("error") == repr(exc)
        if not already_recorded:
            with failures.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {"error": repr(exc), "traceback": traceback.format_exc()},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        raise
    print(path, flush=True)


if __name__ == "__main__":
    main()
