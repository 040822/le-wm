#!/usr/bin/env python3
"""Audit, train, evaluate, diagnose, and summarize CVPR Table 2."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.cvpr_table2 import (
    ARMS,
    RNG_STREAMS,
    SEEDS,
    TASKS,
    arm_config,
    json_bytes,
    matrix_rows,
    run_id,
    sha256_file,
    sha256_value,
    write_json_atomic,
)
from source.common.cvpr_table2_diagnostics import (
    candidate_truth_record,
    fixed_pool_rank_metrics,
    source_state_id,
)


CONFIG_PATH = ROOT / "config/cvpr/table2.json"
TABLE1_CONFIG_PATH = ROOT / "config/cvpr/table1.json"
TABLE1_FREEZE_PATH = ROOT / "outputs/cvpr/table1/v1/freeze.json"
TABLE1_ROUND4_MANIFEST = ROOT / "config/round5/phase1_6.json"
OUTPUT_ROOT = ROOT / "outputs/cvpr/table2/v1"
FIXED_POOL_ROOT = OUTPUT_ROOT / "fixed_pool"
DATASETS = {
    "tworoom": "data/datasets/tworoom.h5",
    "pusht": "data/datasets/pusht.h5",
    "reacher": "data/datasets/dmcontrol/reacher.h5",
    "cube": "data/datasets/ogbench/cube_single.h5",
}
CODE_PATHS = (
    "config/cvpr/table2.json",
    "config/cvpr/table1.json",
    "config/round5/phase1_6.json",
    "config/train/round4_ab.yaml",
    "config/train/policy/round4_ab.yaml",
    "config/train/data/tworoom.yaml",
    "config/train/data/pusht.yaml",
    "config/train/data/reacher.yaml",
    "config/train/data/cube.yaml",
    "source/common/cvpr_table2.py",
    "source/common/cvpr_table2_diagnostics.py",
    "source/common/cvpr_table1.py",
    "source/common/logging.py",
    "source/common/eval.py",
    "source/common/round4_eval.py",
    "source/common/round3_eval.py",
    "source/common/round3_phase1.py",
    "source/common/round3_protocol.py",
    "source/common/round4_action_bounds.py",
    "source/common/round3_validation.py",
    "source/common/round5_phase1_5.py",
    "source/common/checkpoint.py",
    "source/policy/fast_lewam.py",
    "source/policy/round4.py",
    "source/model/fast_lewam/jepa.py",
    "source/model/fast_lewam/round4.py",
    "train.py",
    "scripts/cvpr_table2.py",
    "scripts/round5_phase1_5.py",
    "scripts/round5_phase1_5_diagnostics.py",
    "tests/test_cvpr_table2.py",
    "docs/plan/cvpr_table2_plan.md",
    "docs/plan/cvpr_table3_plan.md",
)
_STATUS_LOCK = threading.Lock()


def _read_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_static_config() -> dict[str, Any]:
    value = _read_json(CONFIG_PATH)
    if value is None:
        raise FileNotFoundError(CONFIG_PATH)
    return value


def _load_frozen() -> dict[str, Any]:
    path = OUTPUT_ROOT / "frozen_config.json"
    value = _read_json(path)
    if value is None:
        raise RuntimeError("Table 2 is not frozen yet; run `audit` first")
    return value


def _dataset_hashes() -> dict[str, dict[str, Any]]:
    table1_freeze = _read_json(TABLE1_FREEZE_PATH)
    if not isinstance(table1_freeze, dict) or table1_freeze.get("status") != "frozen":
        raise RuntimeError("Table 1 dataset identity is not in frozen state")
    expected_hashes = table1_freeze.get("dataset_sha256", {})
    result = {}
    for task, relative in DATASETS.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Table 2 dataset is missing: {path}")
        digest = sha256_file(path)
        if digest != expected_hashes.get(task):
            raise RuntimeError(
                f"{task} dataset hash differs from frozen Table 1: "
                f"expected={expected_hashes.get(task)!r}, actual={digest}"
            )
        result[task] = {
            "path": relative,
            "bytes": int(path.stat().st_size),
            "sha256": digest,
            "table1_frozen_sha256": expected_hashes[task],
        }
    return result


def _audit_table1_protocol(static: dict[str, Any]) -> dict[str, Any]:
    table1 = _read_json(TABLE1_CONFIG_PATH)
    table1_frozen = _read_json(ROOT / "outputs/cvpr/table1/v1/frozen_config.json")
    table1_freeze = _read_json(TABLE1_FREEZE_PATH)
    if not isinstance(table1, dict) or not isinstance(table1_frozen, dict):
        raise RuntimeError("Table 1 protocol configuration or frozen snapshot is missing")
    if not isinstance(table1_freeze, dict) or table1_freeze.get("status") != "frozen":
        raise RuntimeError("Table 1 is not frozen")
    from source.common.cvpr_table2 import json_bytes

    config_digest = hashlib.sha256(json_bytes(table1)).hexdigest()
    if config_digest != table1_freeze.get("configuration_sha256"):
        raise RuntimeError("Table 1 configuration no longer matches its freeze record")
    if table1_frozen != table1:
        raise RuntimeError("Table 1 frozen protocol snapshot differs from its current config")

    protocol = table1.get("protocol", {})
    cowm = protocol.get("cowm", {})
    table2_eval = static.get("evaluation", {})
    checks = {
        "tasks": (table1.get("tasks"), static.get("tasks")),
        "evaluation_seeds": (table1.get("seeds"), static.get("evaluation_seeds")),
        "evaluation.protocol_seeds": (table1.get("seeds"), table2_eval.get("evaluation_seeds")),
        "episodes_per_seed": (protocol.get("episodes_per_seed"), table2_eval.get("episodes_per_seed")),
        "goal_offset_steps": (protocol.get("goal_offset_steps"), table2_eval.get("goal_offset_steps")),
        "total_execution_budget": (protocol.get("eval_budget"), table2_eval.get("total_execution_budget")),
        "horizon": (protocol.get("horizon"), table2_eval.get("horizon")),
        "receding_horizon": (protocol.get("receding_horizon"), table2_eval.get("receding_horizon")),
        "action_block": (protocol.get("action_block"), table2_eval.get("action_block")),
        "candidate_count": (cowm.get("candidate_count"), static.get("diagnostics", {}).get("candidate_count")),
        "action_flow_steps": (cowm.get("action_flow_steps"), table2_eval.get("action_flow_steps")),
        "action_flow_integrator": (cowm.get("integrator"), table2_eval.get("action_flow_integrator")),
        "cohort_source": (table1.get("outputs", {}).get("root", "") + "/cohorts", table2_eval.get("cohort_source")),
    }
    mismatches = {
        key: {"table1": left, "table2": right}
        for key, (left, right) in checks.items()
        if left != right
    }
    if mismatches:
        raise RuntimeError(f"Table 2 protocol differs from frozen Table 1 identity: {mismatches}")
    if protocol.get("variant") != "legacy":
        raise RuntimeError("Table 2 must use the frozen Table 1 legacy evaluation protocol")
    for key in ("precision", "autocast", "tf32", "compile"):
        if table2_eval.get(key) != protocol.get(key):
            raise RuntimeError(f"Table 2 evaluation {key} differs from frozen Table 1")
    if table2_eval.get("action_bound_mode") != "none":
        raise RuntimeError("Table 2 fixed-pool evaluation must preserve native unprojected actions")
    if table1.get("assets", {}).get("cowm_manifest") != str(TABLE1_ROUND4_MANIFEST.relative_to(ROOT)):
        raise RuntimeError("Table 1 fixed actor manifest does not match the frozen Table 2 source")
    expected_fixed_pool_states = int(protocol["episodes_per_seed"]) * len(table1["seeds"])
    if static.get("diagnostics", {}).get("fixed_pool_protocol", {}).get("states_per_task") != expected_fixed_pool_states:
        raise RuntimeError("Table 2 fixed-pool state count does not cover every frozen Table 1 cohort")
    return {
        "table1_config_sha256": config_digest,
        "table1_freeze_sha256": sha256_file(TABLE1_FREEZE_PATH),
        "protocol_variant": protocol["variant"],
        "evaluation_seeds": list(table1["seeds"]),
        "cohort_states_per_task": expected_fixed_pool_states,
        "matched_fields": list(checks),
    }


def _audit_actor_training_config(config_path: Path, task: str, manifest: dict[str, Any]) -> dict[str, Any]:
    import yaml

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise RuntimeError(f"fixed actor config is not a YAML mapping: {config_path}")
    data_name = Path(DATASETS[task]).relative_to("data/datasets").as_posix()
    checks = {
        "seed": (config.get("seed"), int(manifest["training"]["seed"])),
        "train_mode": (config.get("train_mode"), "stage_ab"),
        "embed_dim": (config.get("embed_dim"), 192),
        "action_horizon": (config.get("action_horizon"), 5),
        "history_size": (config.get("history_size"), 1),
        "latent_head_layers": (config.get("latent_head_layers"), 6),
        "latent_head_dim": (config.get("latent_head_dim"), 192),
        "token_encoding": (config.get("token_encoding"), "legacy"),
        "stage_b_dynamics": (config.get("stage_b_dynamics"), "parallel_prefix"),
        "stage_b_attention_mode": (config.get("stage_b_attention_mode"), "strict_causal"),
        "trainer.max_epochs": (config.get("trainer", {}).get("max_epochs"), int(manifest["training"]["epoch"])),
        "trainer.precision": (config.get("trainer", {}).get("precision"), "bf16"),
        "loader.batch_size": (config.get("loader", {}).get("batch_size"), 128),
        "optimizer.type": (config.get("optimizer", {}).get("type"), "AdamW"),
        "optimizer.lr": (float(config.get("optimizer", {}).get("lr", -1)), 5e-5),
        "optimizer.weight_decay": (float(config.get("optimizer", {}).get("weight_decay", -1)), 1e-3),
        "loss.latent.weight": (config.get("loss", {}).get("latent", {}).get("weight"), 1.0),
        "loss.sigreg.weight": (config.get("loss", {}).get("sigreg", {}).get("weight"), 0.09),
        "data.dataset.name": (config.get("data", {}).get("dataset", {}).get("name"), data_name),
        "data_pipeline.gpu_image_preprocessing": (config.get("data_pipeline", {}).get("gpu_image_preprocessing"), True),
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in checks.items()
        if actual != expected
    }
    if mismatches:
        raise RuntimeError(f"Table 1 fixed actor training config mismatch for {task}: {mismatches}")
    return {"matched_fields": list(checks), "sha256": sha256_file(config_path)}


def _fixed_actor_audit() -> dict[str, dict[str, Any]]:
    training_manifest = _read_json(TABLE1_ROUND4_MANIFEST)
    if training_manifest is None:
        raise FileNotFoundError(TABLE1_ROUND4_MANIFEST)
    training = training_manifest.get("training", {})
    if (
        training.get("model") != "R4-AB"
        or int(training.get("seed", -1)) != 3072
        or int(training.get("epoch", -1)) != 10
    ):
        raise RuntimeError("Table 1 fixed actor manifest must identify R4-AB seed 3072 epoch 10")
    actor_records = {}
    for task in TASKS:
        source = ROOT / training_manifest["training"]["checkpoints"][task]
        source_config = source.parent.parent / "config.yaml"
        archive_root = ROOT / "outputs/cvpr/table1/v1/assets/cowm" / task
        archived = archive_root / "checkpoints" / source.name
        archived_config = archive_root / "config.yaml"
        if not source.is_file() or not archived.is_file() or not source_config.is_file():
            raise FileNotFoundError(
                f"Table 1 fixed R4-AB actor assets are incomplete for {task}"
            )
        source_digest = sha256_file(source)
        expected_digest = training_manifest["training"]["checkpoint_sha256"][task]
        archive_digest = sha256_file(archived)
        if source_digest != expected_digest or archive_digest != expected_digest:
            raise RuntimeError(f"R4-AB fixed actor hash audit failed for {task}")
        if sha256_file(source_config) != sha256_file(archived_config):
            raise RuntimeError(f"R4-AB fixed actor config differs from Table 1 archive: {task}")
        config_audit = _audit_actor_training_config(source_config, task, training_manifest)
        actor_records[task] = {
            "training_seed": 3072,
            "epoch": 10,
            "source_checkpoint": str(source.relative_to(ROOT)),
            "source_sha256": source_digest,
            "archived_checkpoint": str(archived.relative_to(ROOT)),
            "archived_sha256": archive_digest,
            "source_config": str(source_config.relative_to(ROOT)),
            "source_config_sha256": sha256_file(source_config),
            "training_config_audit": config_audit,
            "role": "fixed actor only; not an initialization for Table 2 core arms",
        }
    return actor_records


def _cohort_audit() -> dict[str, dict[str, Any]]:
    root = ROOT / "outputs/cvpr/table1/v1/cohorts"
    records = {}
    source_state_ids: dict[str, set[str]] = {task: set() for task in TASKS}
    for task in TASKS:
        for seed in _load_static_config()["evaluation_seeds"]:
            relative = Path("outputs/cvpr/table1/v1/cohorts") / task / f"seed_{seed}.json"
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(f"frozen Table 1 cohort is missing: {path}")
            cohort = _read_json(path)
            entries = cohort.get("entries", [])
            if len(entries) != 50 or int(cohort.get("seed", -1)) != int(seed):
                raise RuntimeError(f"frozen Table 1 cohort has the wrong identity: {path}")
            if int(cohort.get("goal_offset_steps", -1)) != 25:
                raise RuntimeError(f"frozen Table 1 cohort has a different goal offset: {path}")
            if cohort.get("protocol_variant") != "legacy":
                raise RuntimeError(f"fixed Table 2 pools require legacy Table 1 cohorts: {path}")
            state_ids = [source_state_id(task, entry) for entry in entries]
            if len(set(state_ids)) != len(state_ids):
                raise RuntimeError(f"frozen Table 1 cohort repeats a source state: {path}")
            overlap = source_state_ids[task].intersection(state_ids)
            if overlap:
                raise RuntimeError(
                    f"Table 1 cohorts reuse {len(overlap)} source states for {task}; "
                    "Table 2 fixed pool requires 300 unique appearances"
                )
            source_state_ids[task].update(state_ids)
            key = f"{task}/seed_{seed}"
            records[key] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "episode_count": len(entries),
                "cohort_id": cohort["cohort_id"],
                "protocol_variant": cohort["protocol_variant"],
                "goal_offset_steps": cohort["goal_offset_steps"],
            }
    if len(records) != 24:
        raise RuntimeError(f"expected 24 frozen evaluation cohorts; found {len(records)}")
    if any(len(values) != 300 for values in source_state_ids.values()):
        raise RuntimeError("the 24 frozen Table 1 cohorts must provide 300 distinct source states per task")
    return records


def _write_code_snapshot(code_hashes: dict[str, str]) -> None:
    snapshot_root = OUTPUT_ROOT / "provenance/code_snapshot"
    for relative in CODE_PATHS:
        source = ROOT / relative
        target = snapshot_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_file(target) != code_hashes[relative]:
                raise FileExistsError(f"Table 2 code snapshot is immutable: {target}")
            continue
        shutil.copy2(source, target)
        if sha256_file(target) != code_hashes[relative]:
            raise IOError(f"Table 2 code snapshot hash mismatch: {relative}")


def command_audit(_args) -> None:
    static = _load_static_config()
    for key, expected in (("tasks", list(TASKS)), ("training_seeds", list(SEEDS))):
        if static.get(key) != expected:
            raise RuntimeError(f"config/cvpr/table2.json has unexpected {key}")
    if len(ARMS) != 5:
        raise RuntimeError("the core Table 2 matrix must contain exactly five arms")
    table1_protocol = _audit_table1_protocol(static)
    dataset_records = _dataset_hashes()
    actor_records = _fixed_actor_audit()
    cohort_records = _cohort_audit()
    code_hashes = {}
    for relative in CODE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Table 2 source file is missing: {path}")
        code_hashes[relative] = sha256_file(path)

    frozen = {
        **static,
        "identity": {
            "config_sha256": sha256_file(CONFIG_PATH),
            "table1_freeze_sha256": sha256_file(TABLE1_FREEZE_PATH),
            "datasets": dataset_records,
            "table1_protocol_audit": table1_protocol,
            "fixed_actor_checkpoints": actor_records,
            "evaluation_cohorts": cohort_records,
            "code_sha256": code_hashes,
            "core_training_runs": len(matrix_rows()),
            "closed_loop_conditions_per_training_seed": 4 * 14,
            "closed_loop_independent_units": 1008,
            "closed_loop_episodes": 50400,
            "fixed_pool_state_appearances": 4 * 300,
            "fixed_pool_candidate_branches": 4 * 300 * 64,
            "fixed_pool_trained_b_checkpoints": 4 * 3 * 4,
            "fixed_pool_po2_branches": 4 * 3 * 4 * 300,
            "fixed_pool_action_bound_mode": "none",
            "initialization_rule": "identical fresh model initialization within each task/seed across all five arms",
            "checkpoint_rule": "use epoch 10; no test-based checkpoint selection",
        },
    }
    frozen_path = OUTPUT_ROOT / "frozen_config.json"
    previous = _read_json(frozen_path)
    if previous is not None and previous != frozen:
        raise FileExistsError(
            f"frozen Table 2 identity already exists and differs: {frozen_path}"
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(frozen_path, frozen)
    _write_code_snapshot(code_hashes)
    write_json_atomic(
        OUTPUT_ROOT / "provenance/source_refs.json",
        {
            "table1_report": "docs/report/cvpr/cvpr_table1_report.md",
            "table1_audit": "docs/report/cvpr/cvpr_table1_paper_audit_20261005.md",
            "table1_plan": "docs/plan/cvpr_table1_plan.md",
            "table3_plan": "docs/plan/cvpr_table3_plan.md",
            "table1_frozen_config": "outputs/cvpr/table1/v1/frozen_config.json",
            "table1_fixed_actors": actor_records,
            "table1_evaluation_cohorts": cohort_records,
            "reuse_classification": "same benchmark expansion; previously observed Table 1 starts are not an unseen confirmation set",
        },
    )
    write_json_atomic(
        OUTPUT_ROOT / "provenance/code_manifest.json",
        {"schema_version": 1, "files": code_hashes, "snapshot_root": "provenance/code_snapshot"},
    )
    matrix = matrix_rows()
    write_json_atomic(
        OUTPUT_ROOT / "matrix/training_runs.json",
        {
            "schema_version": 1,
            "run_count": len(matrix),
            "required_seed_order": list(SEEDS),
            "runs": matrix,
        },
    )
    write_json_atomic(
        OUTPUT_ROOT / "provenance/audit.json",
        {
            "schema_version": 1,
            "status": "passed",
            "assets": "passed",
            "data": "passed",
            "cohorts": "passed",
            "fixed_actor_role": "fixed evaluation actor only; no weight reuse for core training",
            "five_arm_matrix": "passed",
            "closed_loop_matrix": "14 conditions per task and training seed",
            "closed_loop_independent_units": 1008,
            "closed_loop_episodes": 50400,
            "fixed_pool_state_appearances": 1200,
            "fixed_pool_candidate_branches": 76800,
            "fixed_pool_po2_branches": 14400,
            "fixed_pool_action_bound_mode": "none",
            "gpu_tasks_started": 0,
        },
    )
    print(json.dumps({"status": "frozen", "runs": len(matrix), "output": str(frozen_path)}, indent=2))


def command_amend_code_snapshot(args) -> None:
    """Version a narrowly scoped implementation correction after dispatch."""
    frozen_path = OUTPUT_ROOT / "frozen_config.json"
    frozen = _load_frozen()
    manifest_path = OUTPUT_ROOT / "provenance/code_manifest.json"
    code_manifest = _read_json(manifest_path)
    if not isinstance(code_manifest, dict):
        raise RuntimeError("frozen Table 2 code manifest is missing")
    changed = []
    previous_hashes = frozen.get("identity", {}).get("code_sha256", {})
    for relative in CODE_PATHS:
        source = ROOT / relative
        current_hash = sha256_file(source)
        previous_hash = previous_hashes.get(relative)
        if previous_hash == current_hash:
            continue
        snapshot = OUTPUT_ROOT / "provenance/code_snapshot" / relative
        if not snapshot.is_file() or sha256_file(snapshot) != previous_hash:
            raise RuntimeError(f"previous frozen Table 2 snapshot is missing or changed: {relative}")
        history_path = (
            OUTPUT_ROOT / "provenance/code_history" / str(previous_hash) / relative
        )
        history_path.parent.mkdir(parents=True, exist_ok=True)
        if history_path.exists() and sha256_file(history_path) != previous_hash:
            raise RuntimeError(f"Table 2 code history artifact is immutable: {history_path}")
        if not history_path.exists():
            shutil.copy2(snapshot, history_path)
        temporary = snapshot.with_name(f".{snapshot.name}.{os.getpid()}.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(snapshot)
        previous_hashes[relative] = current_hash
        code_manifest.setdefault("files", {})[relative] = current_hash
        changed.append(
            {
                "path": relative,
                "previous_sha256": previous_hash,
                "current_sha256": current_hash,
                "previous_snapshot": str(history_path.relative_to(OUTPUT_ROOT)),
            }
        )
    if not changed:
        print(json.dumps({"status": "unchanged", "paths": []}, ensure_ascii=False))
        return
    amendment = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": args.reason,
        "changes": changed,
        "training_identity_code_paths_unchanged": True,
    }
    amendments_path = OUTPUT_ROOT / "provenance/code_amendments.json"
    amendments = _read_json(amendments_path, {"schema_version": 1, "amendments": []})
    amendments["amendments"].append(amendment)
    write_json_atomic(frozen_path, frozen)
    code_manifest["snapshot_root"] = "provenance/code_snapshot"
    write_json_atomic(manifest_path, code_manifest)
    write_json_atomic(amendments_path, amendments)
    audit_path = OUTPUT_ROOT / "provenance/audit.json"
    audit = _read_json(audit_path, {})
    audit["code_snapshot_amendments"] = len(amendments["amendments"])
    write_json_atomic(audit_path, audit)
    print(json.dumps({"status": "amended", "changes": changed}, indent=2, ensure_ascii=False))


def _table2_run_dir(task: str, arm: str, seed: int) -> Path:
    return OUTPUT_ROOT / "runs" / task / f"seed_{seed}" / arm


def _latest_training_attempt(task: str, arm: str, seed: int) -> Path:
    root = _table2_run_dir(task, arm, seed)
    attempts = sorted(
        [path for path in root.glob("attempt_*") if path.is_dir()]
    )
    if attempts:
        completed = [
            path for path in attempts
            if _read_json(path / "table2_training_progress.json", {}).get("status") == "complete"
        ]
        return max(completed or attempts, key=lambda path: path.name)
    return root


def _training_checkpoint_path(task: str, arm: str, seed: int) -> Path:
    path = (
        _latest_training_attempt(task, arm, seed)
        / "checkpoints"
        / f"cvpr_table2_{task}_{arm}_seed{seed}_weights_epoch_10.pt"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Table 2 epoch-10 checkpoint is missing: {path}")
    progress = _read_json(path.parent.parent / "table2_training_progress.json", {})
    if progress.get("status") != "complete" or len(progress.get("epochs", [])) != 10:
        raise RuntimeError(f"Table 2 checkpoint attempt is not a complete ten-epoch run: {path}")
    return path


def _check_gpu_vram(gpus: list[int], margin_gib: float) -> None:
    if not gpus or any(gpu not in range(4) for gpu in gpus):
        raise ValueError("Table 2 training may use only explicit GPU0–3 IDs")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    by_id = {}
    for line in completed.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) == 3:
            by_id[int(parts[0])] = (int(parts[1]), int(parts[2]))
    for gpu in gpus:
        if gpu not in by_id:
            raise RuntimeError(f"GPU{gpu} is not visible to nvidia-smi")
        free_mib, utilization = by_id[gpu]
        if free_mib < int(margin_gib * 1024):
            raise RuntimeError(
                f"GPU{gpu} has only {free_mib / 1024:.2f} GiB free; "
                f"Table 2 requires a {margin_gib:.1f} GiB safety margin"
            )
        print(
            f"GPU{gpu}: free={free_mib / 1024:.2f} GiB, "
            f"utilization={utilization}%",
            flush=True,
        )


def _training_overrides(
    task: str, arm: str, seed: int, attempt_number: int, frozen: dict[str, Any]
) -> list[str]:
    definition = arm_config(arm)
    values = [
        f"data={task}",
        f"output_model_name=cvpr_table2_{task}_{arm}_seed{seed}",
        f"subdir=cvpr/table2/v1/runs/{task}/seed_{seed}/{arm}/attempt_{attempt_number:02d}",
        "subdir_preserve_path=true",
        f"seed={seed}",
        f"train_mode={definition['train_mode']}",
        f"stage_b_action_source={definition['stage_b_action_source']}",
        f"detach_clean_action={str(definition['detach_clean_action']).lower()}",
        f"loss.latent.weight={definition['lambda_latent']}",
        f"loss.sigreg.weight={definition['lambda_sigreg']}",
        "loss.d.lambda=0.0",
        "loss.e.lambda=0.0",
        "train_split=0.9",
        "history_size=1",
        "action_horizon=5",
        "num_preds=5",
        "img_size=224",
        "embed_dim=192",
        "latent_head_layers=6",
        "latent_head_dim=192",
        "stage_a_goal_injection=token",
        "latent_action_mix_epochs=10",
        "latent_loss_noise_threshold=0.2",
        "stage_b_timestep_mode=legacy",
        "stage_b_dynamics=parallel_prefix",
        "stage_b_attention_mode=strict_causal",
        "token_encoding=legacy",
        "trainer.max_epochs=10",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "trainer.precision=bf16",
        "trainer.gradient_clip_val=1.0",
        "loader.batch_size=128",
        "num_workers=8",
        "optimizer.type=AdamW",
        "optimizer.lr=0.00005",
        "optimizer.weight_decay=0.001",
        "data_pipeline.gpu_image_preprocessing=true",
        "data_pipeline.hdf5_chunk_size=100",
        "init_weights=null",
        "resume_ckpt=null",
        "initial_epoch=0",
        "episode_split_manifest=null",
        "wandb.enabled=false",
        "round4_diagnostics.enabled=true",
        "cvpr_table2.enabled=true",
        f"cvpr_table2.task={task}",
        f"cvpr_table2.arm={arm}",
        f"cvpr_table2.dataset_sha256.{task}={frozen['identity']['datasets'][task]['sha256']}",
    ]
    return values


def _run_one_training(run: dict[str, Any], gpu: int, frozen: dict[str, Any]) -> dict[str, Any]:
    task, arm, seed = run["task"], run["arm"], int(run["seed"])
    run_root = _table2_run_dir(task, arm, seed)
    run_root.mkdir(parents=True, exist_ok=True)
    prior_attempts = sorted(run_root.glob("attempt_*") )
    complete = []
    for prior_dir in prior_attempts:
        prior_progress = _read_json(prior_dir / "table2_training_progress.json", {})
        prior_checkpoint = prior_dir / "checkpoints" / f"cvpr_table2_{task}_{arm}_seed{seed}_weights_epoch_10.pt"
        if prior_progress.get("status") == "complete" and prior_checkpoint.is_file():
            identity = _read_json(prior_dir / "table2_training_identity.json", {})
            if identity.get("run_id") != run["run_id"]:
                raise RuntimeError(f"completed Table 2 run identity mismatch: {run['run_id']}")
            complete.append((prior_dir, prior_checkpoint))
        invocation = _read_json(prior_dir / "training_invocation.json", {})
        if invocation.get("status") == "running":
            pid = invocation.get("pid")
            if pid:
                try:
                    os.kill(int(pid), 0)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    raise RuntimeError(f"Table 2 training attempt is still active: {prior_dir}")
                else:
                    raise RuntimeError(f"Table 2 training attempt is still active: {prior_dir}")
    if complete:
        attempt_dir, checkpoint = max(complete, key=lambda item: item[0].name)
        return {**run, "status": "already_complete", "checkpoint": str(checkpoint), "attempt": attempt_dir.name}
    attempt_number = max(
        (int(path.name.removeprefix("attempt_")) for path in prior_attempts if path.name.removeprefix("attempt_").isdigit()),
        default=0,
    ) + 1
    run_dir = run_root / f"attempt_{attempt_number:02d}"
    progress_path = run_dir / "table2_training_progress.json"
    identity_path = run_dir / "table2_training_identity.json"
    checkpoint = run_dir / "checkpoints" / f"cvpr_table2_{task}_{arm}_seed{seed}_weights_epoch_10.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    overrides = _training_overrides(task, arm, seed, attempt_number, frozen)
    command = [sys.executable, "train.py", "--config-name=round4_ab", *overrides]
    invocation = {
        **run,
        "status": "running",
        "gpu": int(gpu),
        "attempt": run_dir.name,
        "cuda_visible_devices": str(gpu),
        "command": command,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_atomic(run_dir / "training_invocation.json", invocation)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    log_path = run_dir / "launcher.log"
    started = time.perf_counter()
    with log_path.open("ab", buffering=0) as log_stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )
        invocation["pid"] = int(process.pid)
        write_json_atomic(run_dir / "training_invocation.json", invocation)
        return_code = process.wait()
    elapsed = time.perf_counter() - started
    invocation.update(
        status="postprocess_pending" if return_code == 0 else "failed",
        return_code=int(return_code),
        wall_seconds=float(elapsed),
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        checkpoint_sha256=sha256_file(checkpoint) if checkpoint.is_file() else None,
    )
    write_json_atomic(run_dir / "training_invocation.json", invocation)
    result = {
        **run,
        "status": invocation["status"],
        "return_code": int(return_code),
        "wall_seconds": float(elapsed),
        "gpu": int(gpu),
        "attempt": run_dir.name,
        "checkpoint": str(checkpoint) if checkpoint.is_file() else None,
    }
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if not checkpoint.is_file():
        raise RuntimeError(f"Table 2 training completed without epoch-10 checkpoint: {run_dir}")
    progress = _read_json(progress_path, {})
    if progress.get("status") != "complete" or len(progress.get("epochs", [])) != 10:
        raise RuntimeError(f"Table 2 training did not record ten completed epochs: {run_dir}")
    identity = _read_json(identity_path, {})
    if identity.get("run_id") != run["run_id"]:
        raise RuntimeError(f"Table 2 training identity was not written correctly: {run_dir}")
    invocation.update(status="complete")
    write_json_atomic(run_dir / "training_invocation.json", invocation)
    result["status"] = "complete"
    return result


def _save_training_status(results: list[dict[str, Any]], current_seed: int, complete: bool) -> None:
    with _STATUS_LOCK:
        path = OUTPUT_ROOT / "summary/training_status.json"
        current = _read_json(path, {"schema_version": 1, "runs": []})
        by_id = {item["run_id"]: item for item in current.get("runs", [])}
        for item in results:
            by_id[item["run_id"]] = item
        write_json_atomic(
            path,
            {
                "schema_version": 1,
                "current_seed_wave": int(current_seed),
                "seed_wave_complete": bool(complete),
                "runs": [by_id[key] for key in sorted(by_id)],
            },
        )


def _record_training_result(
    result: dict[str, Any], wave_results: list[dict[str, Any]], seed: int
) -> None:
    """Append one worker result, then persist it under the status writer's lock."""
    with _STATUS_LOCK:
        wave_results.append(result)
    _save_training_status(wave_results, seed, False)


def _validate_training_wave_identities(tasks: tuple[str, ...], arms: tuple[str, ...], seed: int) -> None:
    compare_paths = (
        ("initial_model_sha256", ("initial_model_sha256",)),
        ("initial_shared_representation_sha256", ("initial_shared_representation_sha256",)),
        ("train_indices_sha256", ("split", "train_indices_sha256")),
        ("validation_indices_sha256", ("split", "validation_indices_sha256")),
        ("sample_order_sha256_by_epoch", ("split", "sample_order_sha256_by_epoch")),
        ("normalizer_sha256", ("normalizer", "sha256")),
        ("updates_per_epoch", ("optimization", "updates_per_epoch")),
    )
    audited = []
    for task in tasks:
        records = {}
        for arm in arms:
            attempt = _latest_training_attempt(task, arm, seed)
            identity = _read_json(attempt / "table2_training_identity.json")
            progress = _read_json(attempt / "table2_training_progress.json", {})
            checkpoint = attempt / "checkpoints" / f"cvpr_table2_{task}_{arm}_seed{seed}_weights_epoch_10.pt"
            if not checkpoint.is_file() or progress.get("status") != "complete" or len(progress.get("epochs", [])) != 10:
                raise RuntimeError(f"Table 2 seed wave has an incomplete run: {task}/{seed}/{arm}")
            records[arm] = identity
        reference_arm = arms[0]
        reference = records[reference_arm]
        for arm in arms[1:]:
            for label, path in compare_paths:
                left = reference
                right = records[arm]
                for component in path:
                    left = left[component]
                    right = right[component]
                if left != right:
                    raise RuntimeError(
                        f"Table 2 matched identity failed for {task}/seed_{seed}: "
                        f"{reference_arm} vs {arm} differ in {label}"
                    )
        audited.append(
            {
                "task": task,
                "training_seed": int(seed),
                "arms": list(arms),
                "matched_fields": [label for label, _ in compare_paths],
                "initial_model_sha256": reference["initial_model_sha256"],
                "shared_representation_sha256": reference["initial_shared_representation_sha256"],
                "train_indices_sha256": reference["split"]["train_indices_sha256"],
                "validation_indices_sha256": reference["split"]["validation_indices_sha256"],
                "sample_order_sha256_by_epoch": reference["split"]["sample_order_sha256_by_epoch"],
                "normalizer_sha256": reference["normalizer"]["sha256"],
                "updates_per_epoch": reference["optimization"]["updates_per_epoch"],
                "status": "matched",
            }
        )
    audit_path = OUTPUT_ROOT / "provenance/training_identity_matches.json"
    existing = _read_json(audit_path, {"schema_version": 1, "waves": []})
    waves = [item for item in existing.get("waves", []) if int(item["training_seed"]) != int(seed) or item["task"] not in tasks]
    waves.extend(audited)
    write_json_atomic(audit_path, {"schema_version": 1, "waves": waves})


def command_train(args) -> None:
    frozen = _load_frozen()
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds or any(seed not in SEEDS for seed in seeds):
        raise ValueError(f"--seeds must be a non-empty subset of {SEEDS}")
    if tuple(sorted(seeds, key=SEEDS.index)) != seeds:
        raise ValueError("training seed waves must run in the frozen order 3072, 4096, 5120")
    tasks = tuple(args.tasks or TASKS)
    arms = tuple(args.arms or tuple(ARMS))
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"invalid Table 2 task {task!r}")
    for arm in arms:
        arm_config(arm)
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    _check_gpu_vram(gpus, float(frozen["resources"]["training_vram_margin_gib"]))
    free_disk_gib = shutil.disk_usage(OUTPUT_ROOT.parent).free / (1024**3)
    if free_disk_gib < float(frozen["resources"]["stop_dispatch_free_disk_gib"]):
        raise RuntimeError(
            f"only {free_disk_gib:.1f} GiB disk space is free; "
            "Table 2 training dispatch threshold is 20 GiB"
        )
    print(f"free disk: {free_disk_gib:.1f} GiB", flush=True)

    all_results = []
    for seed in seeds:
        jobs = [
            row
            for row in matrix_rows(tasks=tasks, seeds=(seed,))
            if row["arm"] in arms
        ]
        worker_jobs = [jobs[index:: len(gpus)] for index in range(len(gpus))]
        wave_results: list[dict[str, Any]] = []
        errors: list[tuple[str, str]] = []

        def worker(gpu: int, assigned: list[dict[str, Any]]):
            local_results = []
            for run in assigned:
                try:
                    result = _run_one_training(run, gpu, frozen)
                    local_results.append(result)
                    _record_training_result(result, wave_results, seed)
                except Exception as exc:
                    rendered = traceback.format_exc()
                    with _STATUS_LOCK:
                        errors.append((run["run_id"], rendered))
                        failure_path = OUTPUT_ROOT / "summary/training_failures.json"
                        existing = _read_json(failure_path, {"failures": []})
                        existing["failures"] = [
                            item for item in existing.get("failures", []) if item["run_id"] != run["run_id"]
                        ] + [{"run_id": run["run_id"], "error": rendered}]
                        write_json_atomic(failure_path, existing)
            return local_results

        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [
                executor.submit(worker, gpu, assigned)
                for gpu, assigned in zip(gpus, worker_jobs)
            ]
            for future in futures:
                all_results.extend(future.result())
        _validate_training_wave_identities(tasks, arms, seed)
        _save_training_status(wave_results, seed, not errors)
        if errors:
            raise RuntimeError(
                f"seed {seed} had {len(errors)} failed runs; inspect "
                f"{OUTPUT_ROOT / 'summary/training_failures.json'}"
            )
        print(
            json.dumps(
                {"seed": seed, "completed_runs": len(wave_results), "next_seed_unblocked": True},
                ensure_ascii=False,
            ),
            flush=True,
        )
    print(json.dumps({"training_runs_returned": len(all_results)}, ensure_ascii=False))


def _gradient_norm(module) -> float:
    import torch

    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.sqrt(torch.stack(values).sum()).item())


def _cpu_gradient_diagnostic(task: str, seed: int, batch_size: int):
    import hydra
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    from source.common.data import load_dataset
    from source.common.cvpr_table2 import RNG_STREAMS
    from source.policy.fast_lewam import fast_lewam_forward

    data_paths = DATASETS
    torch.set_num_threads(1)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config/train")):
        cfg = compose(
            config_name="round4_ab",
            overrides=[f"data={task}", "wandb.enabled=false"],
        )
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_cfg.pop("name")
    dataset = load_dataset(str(ROOT / data_paths[task]), **dataset_cfg)
    action_dim = int(dataset.get_dim("action") * dataset.frameskip)
    with open_dict(cfg):
        cfg.policy.model.action_dim = action_dim

    torch.manual_seed(seed)
    model = hydra.utils.instantiate(cfg.policy.model).cpu()
    sigreg = hydra.utils.instantiate(cfg.policy.sigreg).cpu()
    initial_state = copy.deepcopy(model.state_dict())
    action_horizon = int(cfg.action_horizon)
    generator = torch.Generator().manual_seed(seed + 9901)
    batch = {
        "pixels": torch.randint(
            0,
            256,
            (batch_size, action_horizon + 1, 3, 224, 224),
            dtype=torch.uint8,
            generator=generator,
        ),
        "action": torch.randn(
            batch_size,
            action_horizon,
            action_dim,
            generator=generator,
        ),
    }

    class Harness:
        def __init__(self, train_mode: str, detach: bool):
            self.model = model
            self.sigreg = sigreg
            self.current_epoch = 9
            self._round4_epoch_offset = 0
            self.rng_seed = seed
            self._fast_lewam_rng_streams = {}
            self.train_mode = train_mode
            self.detach_clean_action = detach
            self.stage_b_action_source = "joint"

        @staticmethod
        def log_dict(*_args, **_kwargs):
            return None

    def run_arm(arm: str, *, latent_only: bool = False):
        definition = arm_config(arm)
        model.load_state_dict(initial_state, strict=True)
        model.zero_grad(set_to_none=True)
        sigreg.zero_grad(set_to_none=True)
        model.train()
        harness = Harness(
            str(definition["train_mode"]), bool(definition["detach_clean_action"])
        )
        captured = {"stage_a": [], "stage_b": []}

        def hook(_module, args, kwargs):
            mode = kwargs.get("mode")
            if mode == "stage_a":
                captured["stage_a"].append(
                    {"actions": args[1].detach().clone(), "timestep": args[2].detach().clone()}
                )
            elif mode == "stage_b":
                captured["stage_b"].append(
                    {"actions": args[1].detach().clone(), "timestep": args[2].detach().clone()}
                )

        hook_handle = model.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            output = fast_lewam_forward(
                harness,
                batch=batch,
                stage="fit",
                action_horizon=action_horizon,
                train_mode=str(definition["train_mode"]),
                lambda_latent=float(definition["lambda_latent"]),
                lambda_sigreg=float(definition["lambda_sigreg"]),
                detach_clean_action=bool(definition["detach_clean_action"]),
                latent_loss_noise_threshold=0.2,
                latent_action_mix_epochs=10,
                stage_b_timestep_mode="legacy",
                stage_b_action_source=str(definition["stage_b_action_source"]),
                stage_a_goal_index=None,
            )
            selected_loss = output["weighted_latent_prefix_loss"] if latent_only else output["loss"]
            selected_loss.backward()
        finally:
            hook_handle.remove()
        return output, captured

    a_only_output, a_only_capture = run_arm("a_only")
    a_only_action_head_grad = _gradient_norm(model.action_head)
    if a_only_capture["stage_b"]:
        raise RuntimeError("A-only unexpectedly ran the B forward")
    if not a_only_action_head_grad > 0.0:
        raise RuntimeError("A-only did not train the action generator")
    model.load_state_dict(initial_state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.action_head.requires_grad_(False)
    if model.action_positions is not None:
        model.action_positions.requires_grad_(False)
    if model.goal_token_embedding is not None:
        model.goal_token_embedding.requires_grad_(False)
    b_only_output, b_only_capture = run_arm("b_only_clean")
    b_only_action_head_grad = _gradient_norm(model.action_head)
    if "action_loss" in b_only_output or b_only_capture["stage_a"]:
        raise RuntimeError("B-only unexpectedly trained Stage A")
    b_only_latent_head_grad = _gradient_norm(model.latent_head)
    if not b_only_latent_head_grad > 0.0:
        raise RuntimeError("B-only did not train the latent predictor")

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    recorded_output, recorded_capture = run_arm("shared_recorded_control")
    full_output, full_capture = run_arm("shared_pred_full")
    if len(recorded_capture["stage_b"]) != 1 or len(full_capture["stage_b"]) != 1:
        raise RuntimeError("shared A+B arms must perform exactly one Stage-B forward")
    recorded_b = recorded_capture["stage_b"][0]
    full_b = full_capture["stage_b"][0]
    if not torch.equal(recorded_b["timestep"], full_b["timestep"]):
        raise RuntimeError("Recorded-control and Pred-Full changed Stage-B timesteps")
    if not torch.equal(recorded_output["noise_weight"], full_output["noise_weight"]):
        raise RuntimeError("Recorded-control and Pred-Full changed latent sample weights")
    if not torch.equal(recorded_b["actions"], batch["action"][:, :action_horizon]):
        raise RuntimeError("Recorded-control did not pass recorded actions to Stage B")
    changed_rows = (
        (full_b["actions"] - recorded_b["actions"]).abs().amax(dim=(1, 2)) > 1e-7
    )
    if not bool(changed_rows.any()):
        raise RuntimeError("Pred-Full diagnostic minibatch contained no predicted actions")

    _, detach_capture = run_arm("shared_pred_detach", latent_only=True)
    if not torch.equal(
        detach_capture["stage_b"][0]["actions"], full_b["actions"]
    ):
        raise RuntimeError("Detach changed the Stage-B action values")
    if not torch.equal(detach_capture["stage_b"][0]["timestep"], full_b["timestep"]):
        raise RuntimeError("Detach changed the Stage-B timestep values")
    detach_action_path_grad = _gradient_norm(model.action_head)
    _, _ = run_arm("shared_pred_full", latent_only=True)
    full_action_path_grad = _gradient_norm(model.action_head)
    full_shared_predictor_grad = _gradient_norm(model.predictor)
    if detach_action_path_grad != 0.0:
        raise RuntimeError("Pred-Detach retained a Stage-A action-path gradient from B loss")
    if not full_action_path_grad > 0.0:
        raise RuntimeError("Pred-Full did not propagate B loss into predicted actions")
    if not full_shared_predictor_grad > 0.0:
        raise RuntimeError("Pred-Detach did not train the shared predictor from B loss")

    return {
        "task": task,
        "training_seed_for_rng": seed,
        "batch_size": int(batch_size),
        "action_dim": action_dim,
        "synthetic_minibatch_sha256": hashlib.sha256(
            batch["pixels"].numpy().tobytes() + batch["action"].numpy().tobytes()
        ).hexdigest(),
        "recorded_control_vs_full": {
            "same_stage_b_timestep": True,
            "same_mean_latent_sample_weight": True,
            "recorded_control_uses_recorded_actions": True,
            "pred_full_rows_using_predicted_actions": int(changed_rows.sum().item()),
        },
        "detach_vs_full": {
            "same_stage_b_action_values": True,
            "same_stage_b_timesteps": True,
            "pred_detach_action_head_gradient_norm": detach_action_path_grad,
            "pred_full_action_head_gradient_norm": full_action_path_grad,
            "pred_detach_shared_predictor_gradient_norm": full_shared_predictor_grad,
        },
        "a_only": {
            "stage_b_forward_count": len(a_only_capture["stage_b"]),
            "b_loss_present": "weighted_latent_prefix_loss" in a_only_output,
            "action_head_gradient_norm": a_only_action_head_grad,
        },
        "b_only_clean": {
            "stage_a_forward_count": len(b_only_capture["stage_a"]),
            "has_action_loss": "action_loss" in b_only_output,
            "action_head_gradient_norm": b_only_action_head_grad,
            "latent_head_gradient_norm": b_only_latent_head_grad,
        },
    }


def command_diagnose(args) -> None:
    _load_frozen()
    result = {
        "schema_version": 1,
        "status": "passed",
        "device": "cpu",
        "checks": [
            _cpu_gradient_diagnostic(task, int(args.seed), int(args.batch_size))
            for task in args.tasks or ("pusht",)
        ],
    }
    write_json_atomic(OUTPUT_ROOT / "diagnostics/cpu_gradient_acceptance.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _pool_seed_root(task: str, evaluation_seed: int) -> Path:
    return FIXED_POOL_ROOT / task / f"evaluation_seed_{int(evaluation_seed)}"


def _table2_artifact_identity(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise RuntimeError(f"missing Table 2 identity file: {path}")
    return value


def _write_npz_atomic(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        import numpy as np

        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _make_fixed_pool_eval_config(task: str, evaluation_seed: int):
    from source.common.eval import compose_eval_config

    environment_seed = int(evaluation_seed) + 10000
    policy_seed = int(evaluation_seed) + 20000
    cfg = compose_eval_config(
        task,
        [
            "eval.num_eval=50",
            "eval.goal_offset_steps=25",
            "eval.eval_budget=50",
            "plan_config.horizon=5",
            "plan_config.receding_horizon=5",
            "plan_config.action_block=5",
            "output.save_video=false",
            f"seed={environment_seed}",
            f"eval.policy_seed={policy_seed}",
            "solver.device=cuda:0",
            "solver.batch_size=1",
            "solver.num_samples=300",
            "solver.n_steps=30",
            "solver.topk=30",
            "solver.var_scale=1.0",
        ],
    )
    cfg.eval.dataset_name = str((ROOT / DATASETS[task]).resolve())
    cfg.eval.benchmark_dataset_name = DATASETS[task]
    cfg.world.num_envs = 50
    cfg.world.max_episode_steps = 100
    return cfg


def _capture_table1_fixed_pool(
    *, task: str, evaluation_seed: int, actor, actor_path: Path, manifest,
    cfg, dataset, output_dir: Path,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from source.common.eval import EvaluationIdentity
    from source.common.round4_eval import _DiagnosticCaptureComplete, run_round4_evaluation

    model = getattr(actor, "model", actor)
    captured: dict[str, Any] = {}

    def callback(event: dict[str, Any]) -> None:
        slots = tuple(int(value) for value in event["replan_indices"])
        if set(slots) != set(range(len(manifest.entries))):
            raise RuntimeError("fixed actor pool did not cover the full frozen cohort")
        candidates = np.asarray(event["candidates"], dtype=np.float32)
        costs = np.asarray(event["costs"], dtype=np.float64)
        noise = event.get("candidate_noise")
        if noise is None:
            raise RuntimeError("fixed actor pool omitted the exact candidate noise")
        if torch.is_tensor(noise):
            noise = noise.detach().float().cpu().numpy()
        noise = np.asarray(noise, dtype=np.float32).reshape(candidates.shape)
        expected = (
            len(manifest.entries), 64, int(model.action_horizon), int(model.action_dim)
        )
        if candidates.shape != expected or noise.shape != expected:
            raise ValueError(f"fixed actor candidate shape {candidates.shape}; expected {expected}")
        if costs.shape != expected[:2] or not np.isfinite(costs).all():
            raise ValueError("fixed actor scores are missing or non-finite")
        metadata = event.get("event", {})
        if metadata.get("action_bound_mode") != "none":
            raise RuntimeError("fixed actor pool must retain native, unprojected actions")
        captured.update(
            candidates=candidates,
            candidate_noise=noise,
            table1_fixed_actor_costs=costs,
            candidate_noise_sha256=metadata.get("candidate_noise_sha256"),
            scorer_z_start=np.asarray(event["verifier_z_start"], dtype=np.float32),
            scorer_z_goal=np.asarray(event["verifier_z_goal"], dtype=np.float32),
        )
        raise _DiagnosticCaptureComplete(
            {"states": len(manifest.entries), "candidate_count": 64}
        )

    torch.manual_seed(int(evaluation_seed) + 20000)
    np.random.seed((int(evaluation_seed) + 20000) % (2**32 - 1))
    result = run_round4_evaluation(
        cfg,
        task=task,
        policy_or_model=actor,
        mode="P3",
        identity=EvaluationIdentity(
            entrypoint="cvpr_table2_fixed_pool",
            policy_kind="table1_fixed_r4_ab_actor",
            checkpoint=str(actor_path.resolve()),
            epoch=10,
            stage="stage_ab",
        ),
        manifest=manifest,
        output_dir=output_dir,
        device="cuda:0",
        trace=False,
        candidate_count=64,
        flow_steps=2,
        action_flow_steps=2,
        solver_batch_size=1,
        candidate_batch_size=64,
        action_flow_integrator="euler",
        action_bound_mode="none",
        bf16_proposal=False,
        bf16_verifier=False,
        optimize_proposal=False,
        cache_goal_latent=False,
        bf16_encode=False,
        allowed_protocol_variants=("legacy",),
        allow_solver_config_override=True,
        allow_evaluation_seed_override=True,
        allow_cohort_seed_mismatch=True,
        policy_seed=int(evaluation_seed) + 20000,
        execute_steps=25,
        score_horizon_blocks=5,
        video_slots=0,
        diagnostic_callback=callback,
        dataset=dataset,
    )
    if result.get("status") != "diagnostic_capture" or "candidates" not in captured:
        raise RuntimeError(f"could not capture Table 1 fixed actor pool: {result.get('status')}")
    return captured


def _pool_physical_actions(candidates: Any, action_processor: Any, action_block: int):
    import numpy as np

    state_count, candidate_count, horizon, packed_dim = candidates.shape
    base_action_dim = packed_dim // int(action_block)
    if packed_dim % int(action_block):
        raise ValueError("packed action width is not divisible by the frozen action block")
    physical = action_processor.inverse_transform(
        np.asarray(candidates).reshape(-1, base_action_dim)
    )
    return np.asarray(physical, dtype=np.float32).reshape(
        state_count, candidate_count, horizon * int(action_block), base_action_dim
    )


def _pool_identity(*, task: str, evaluation_seed: int, actor_path: Path, manifest, frozen):
    return {
        "schema_version": 1,
        "experiment": "cvpr_table2_v1_fixed_pool",
        "task": task,
        "evaluation_seed": int(evaluation_seed),
        "environment_seed": int(evaluation_seed) + 10000,
        "policy_seed": int(evaluation_seed) + 20000,
        "dataset_sha256": frozen["identity"]["datasets"][task]["sha256"],
        "cohort_sha256": manifest.computed_sha256,
        "actor_checkpoint": str(actor_path.resolve()),
        "actor_checkpoint_sha256": sha256_file(actor_path),
        "protocol": {
            "candidate_count": 64,
            "flow_steps": 2,
            "integrator": "euler",
            "action_bound_mode": "none",
            "execution_steps": 25,
            "precision": "float32",
        },
    }


def _complete_pool_artifact(pool_root: Path, identity: dict[str, Any]):
    metadata_path = pool_root / "pool_metadata.json"
    artifact_path = pool_root / "fixed_pool.npz"
    metadata = _read_json(metadata_path)
    if metadata is None or not artifact_path.is_file():
        return None
    identity_sha = sha256_value(identity)
    if metadata.get("identity_sha256") != identity_sha or metadata.get("identity") != identity:
        raise RuntimeError(f"fixed candidate pool identity changed: {pool_root}")
    if metadata.get("artifact_sha256") != sha256_file(artifact_path):
        raise RuntimeError(f"fixed candidate pool artifact hash mismatch: {artifact_path}")
    return artifact_path


def _load_or_make_fixed_pool(
    *, task: str, evaluation_seed: int, actor, actor_path: Path, manifest, cfg, dataset, frozen,
):
    import numpy as np

    from source.common.eval import DatasetEvaluationSession

    pool_root = _pool_seed_root(task, evaluation_seed)
    pool_root.mkdir(parents=True, exist_ok=True)
    pool_path = _complete_pool_artifact(
        pool_root, _pool_identity(
            task=task, evaluation_seed=evaluation_seed, actor_path=actor_path,
            manifest=manifest, frozen=frozen,
        )
    )
    if pool_path is not None:
        with np.load(pool_path, allow_pickle=False) as stored:
            return {key: stored[key] for key in stored.files}, pool_path

    session = DatasetEvaluationSession(
        cfg, task=task, dataset=dataset, cohort=manifest.to_evaluation_cohort()
    )
    captured = _capture_table1_fixed_pool(
        task=task,
        evaluation_seed=evaluation_seed,
        actor=actor,
        actor_path=actor_path,
        manifest=manifest,
        cfg=cfg,
        dataset=dataset,
        output_dir=pool_root / "capture_work",
    )
    physical = _pool_physical_actions(
        captured["candidates"],
        session.process["action"],
        int(cfg.plan_config.action_block),
    )
    arrays = {
        "candidates": captured["candidates"],
        "physical_actions": physical,
        "table1_fixed_actor_costs": captured["table1_fixed_actor_costs"],
        "candidate_noise": captured["candidate_noise"],
        "table1_fixed_actor_z_start": captured["scorer_z_start"],
        "table1_fixed_actor_z_goal": captured["scorer_z_goal"],
        "action_processor_mean": np.asarray(session.process["action"].mean_, dtype=np.float64),
        "action_processor_scale": np.asarray(session.process["action"].scale_, dtype=np.float64),
    }
    identity = _pool_identity(
        task=task, evaluation_seed=evaluation_seed, actor_path=actor_path,
        manifest=manifest, frozen=frozen,
    )
    identity_sha = sha256_value(identity)
    existing_pool = pool_root / "fixed_pool.npz"
    if existing_pool.exists():
        # A crash after the atomic array publish but before its sidecar is
        # recoverable only if deterministic recapture reproduces the same bytes.
        temporary = pool_root / f".recaptured_{os.getpid()}.npz"
        _write_npz_atomic(temporary, **arrays)
        with np.load(temporary, allow_pickle=False) as fresh, np.load(existing_pool, allow_pickle=False) as existing:
            matches = set(fresh.files) == set(existing.files) and all(
                np.array_equal(fresh[key], existing[key]) for key in fresh.files
            )
        temporary.unlink()
        if not matches:
            raise RuntimeError(f"unregistered fixed pool differs from deterministic recapture: {existing_pool}")
    else:
        _write_npz_atomic(existing_pool, **arrays)
    write_json_atomic(
        pool_root / "pool_metadata.json",
        {
            "schema_version": 1,
            "status": "complete",
            "identity": identity,
            "identity_sha256": identity_sha,
            "candidate_noise_sha256": captured["candidate_noise_sha256"],
            "artifact": str(existing_pool.relative_to(ROOT)),
            "artifact_sha256": sha256_file(existing_pool),
            "candidate_shape": list(captured["candidates"].shape),
            "action_bound_mode": "none",
            "environment_action_handling": "native environment processing after inverse dataset normalizer",
            "evaluation_action_normalizer": {
                "kind": "StandardScaler fitted over full evaluation dataset",
                "mean": np.asarray(session.process["action"].mean_, dtype=np.float64).tolist(),
                "scale": np.asarray(session.process["action"].scale_, dtype=np.float64).tolist(),
            },
        },
    )
    return arrays, existing_pool


def _candidate_branch_identity(
    *, task: str, evaluation_seed: int, candidate_index: int, pool_path: Path,
    manifest, frozen,
):
    return {
        "schema_version": 1,
        "task": task,
        "evaluation_seed": int(evaluation_seed),
        "candidate_index": int(candidate_index),
        "pool_sha256": sha256_file(pool_path),
        "dataset_sha256": frozen["identity"]["datasets"][task]["sha256"],
        "cohort_sha256": manifest.computed_sha256,
        "environment_seed": int(evaluation_seed) + 10000,
        "action_bound_mode": "none",
        "branch_horizon_primitive_steps": 25,
        "success_source": "native environment success event",
    }


def _read_complete_branch(path: Path, identity: dict[str, Any]):
    sidecar = _read_json(path.with_suffix(".meta.json"))
    if sidecar is None or not path.is_file():
        return None
    if sidecar.get("identity") != identity or sidecar.get("identity_sha256") != sha256_value(identity):
        raise RuntimeError(f"candidate branch identity changed: {path}")
    if sidecar.get("artifact_sha256") != sha256_file(path):
        raise RuntimeError(f"candidate branch hash mismatch: {path}")
    rows = _read_jsonl(path)
    if len(rows) != 50:
        raise RuntimeError(f"candidate branch has {len(rows)} outcomes, expected 50: {path}")
    return rows


def _execute_fixed_candidate_branch(
    *, task: str, evaluation_seed: int, candidate_index: int, normalized_actions,
    manifest, session, branch_cfg, dataset, actor_model, branch_state,
    output_dir: Path,
):
    import scripts.round5_phase1_5_diagnostics as p15

    episodes = p15._run_fixed_candidate(
        cfg=branch_cfg,
        task=task,
        manifest=manifest,
        normalized_actions=normalized_actions,
        process=session.process,
        model=actor_model,
        transform=session.transform["pixels"],
        device="cuda:0",
        output_dir=output_dir,
        dataset=dataset,
        branch_state=branch_state,
        goal_latents=None,
        capture_future_latents=False,
        evaluation_session=session,
    )
    by_slot = {int(item["slot"]): item for item in episodes}
    if set(by_slot) != set(range(len(manifest.entries))):
        raise RuntimeError("candidate branch did not return one episode per frozen state")
    return [
        candidate_truth_record(
            task=task,
            evaluation_seed=evaluation_seed,
            entry=manifest.entries[slot].as_dict(),
            slot=slot,
            candidate_index=candidate_index,
            episode=by_slot[slot],
        )
        for slot in range(len(manifest.entries))
    ]


def _ensure_candidate_branch_truth(
    *, task: str, evaluation_seed: int, manifest, cfg, dataset, actor_model,
    pool: dict[str, Any], pool_path: Path, frozen,
):
    import copy
    import numpy as np

    from source.common.eval import DatasetEvaluationSession

    pool_root = _pool_seed_root(task, evaluation_seed)
    branches_root = pool_root / "branches"
    branches_root.mkdir(parents=True, exist_ok=True)
    branch_cfg = copy.deepcopy(cfg)
    branch_cfg.eval.eval_budget = 25
    branch_cfg.world.max_episode_steps = 50
    session = DatasetEvaluationSession(
        branch_cfg, task=task, dataset=dataset, cohort=manifest.to_evaluation_cohort()
    )
    branch_state: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    branch_hashes = []
    for candidate_index in range(64):
        path = branches_root / f"candidate_{candidate_index:04d}.jsonl"
        identity = _candidate_branch_identity(
            task=task, evaluation_seed=evaluation_seed,
            candidate_index=candidate_index, pool_path=pool_path,
            manifest=manifest, frozen=frozen,
        )
        rows = _read_complete_branch(path, identity)
        if rows is None:
            rows = _execute_fixed_candidate_branch(
                task=task,
                evaluation_seed=evaluation_seed,
                candidate_index=candidate_index,
                normalized_actions=np.asarray(pool["candidates"][:, candidate_index]),
                manifest=manifest,
                session=session,
                branch_cfg=branch_cfg,
                dataset=dataset,
                actor_model=actor_model,
                branch_state=branch_state,
                output_dir=branches_root / "replay_work",
            )
            if path.exists():
                if _read_jsonl(path) != rows:
                    raise RuntimeError(f"unregistered candidate branch differs from deterministic replay: {path}")
            else:
                _write_jsonl_atomic(path, rows)
            write_json_atomic(
                path.with_suffix(".meta.json"),
                {
                    "status": "complete",
                    "identity": identity,
                    "identity_sha256": sha256_value(identity),
                    "artifact_sha256": sha256_file(path),
                },
            )
        all_rows.extend(rows)
        branch_hashes.append(sha256_file(path))
        if (candidate_index + 1) % 8 == 0:
            print(
                f"[{task}/{evaluation_seed}] native candidate branches {candidate_index + 1}/64",
                flush=True,
            )

    replay = _execute_fixed_candidate_branch(
        task=task,
        evaluation_seed=evaluation_seed,
        candidate_index=0,
        normalized_actions=np.asarray(pool["candidates"][:, 0]),
        manifest=manifest,
        session=session,
        branch_cfg=branch_cfg,
        dataset=dataset,
        actor_model=actor_model,
        branch_state=branch_state,
        output_dir=branches_root / "replay_repeat_work",
    )
    original = _read_complete_branch(
        branches_root / "candidate_0000.jsonl",
        _candidate_branch_identity(
            task=task, evaluation_seed=evaluation_seed,
            candidate_index=0, pool_path=pool_path,
            manifest=manifest, frozen=frozen,
        ),
    )
    repeat_hash = sha256_value(replay)
    original_hash = sha256_value(original)
    if repeat_hash != original_hash:
        raise RuntimeError(
            f"fixed-pool environment reset/replay was not deterministic: {task}/{evaluation_seed}"
        )
    determinism_path = pool_root / "branch_determinism.json"
    determinism = {
        "schema_version": 1,
        "task": task,
        "evaluation_seed": int(evaluation_seed),
        "candidate_zero_original_sha256": original_hash,
        "candidate_zero_repeat_sha256": repeat_hash,
        "identical": True,
        "branch_action_bound_mode": "none",
    }
    previous = _read_json(determinism_path)
    if previous is not None and previous != determinism:
        raise RuntimeError(f"fixed-pool replay proof changed: {determinism_path}")
    write_json_atomic(determinism_path, determinism)

    truth_path = pool_root / "branch_truth.jsonl"
    truth_identity = {
        "schema_version": 1,
        "task": task,
        "evaluation_seed": int(evaluation_seed),
        "pool_sha256": sha256_file(pool_path),
        "candidate_branch_sha256": branch_hashes,
        "row_count": len(all_rows),
        "determinism_sha256": sha256_value(determinism),
    }
    truth_meta_path = pool_root / "branch_truth.meta.json"
    if truth_path.exists():
        if _read_jsonl(truth_path) != all_rows:
            raise RuntimeError(f"fixed branch truth aggregate changed: {truth_path}")
        previous_identity = _read_json(truth_meta_path)
        if previous_identity is not None and (
            previous_identity.get("identity") != truth_identity
            or previous_identity.get("artifact_sha256") != sha256_file(truth_path)
        ):
            raise RuntimeError(f"fixed branch truth aggregate metadata changed: {truth_path}")
    else:
        _write_jsonl_atomic(truth_path, all_rows)
    if not truth_meta_path.exists():
        write_json_atomic(
            truth_meta_path,
            {
                "identity": truth_identity,
                "identity_sha256": sha256_value(truth_identity),
                "artifact_sha256": sha256_file(truth_path),
            },
        )
    if len(all_rows) != len(manifest.entries) * 64:
        raise RuntimeError("fixed-pool branch truth has the wrong number of rows")
    return all_rows, truth_path


def _encode_manifest_observations(model, dataset, transform, manifest, device: str):
    import numpy as np
    import torch

    starts = np.asarray([int(entry.row_index) for entry in manifest.entries], dtype=np.int64)
    goals = np.asarray(
        [int(entry.goal_row_index) for entry in manifest.entries], dtype=np.int64
    )
    if np.any(starts < 0) or np.any(goals < 0):
        raise ValueError("fixed-pool states require valid start and goal dataset rows")
    all_rows = np.concatenate((starts, goals))
    unique_rows, inverse = np.unique(all_rows, return_inverse=True)
    raw = dataset.get_row_data(unique_rows.tolist())
    pixels = np.asarray(raw["pixels"])
    if pixels.shape[0] != len(unique_rows):
        raise ValueError("HDF5 observation rows do not match the fixed pool cohort")
    model = getattr(model, "model", model).to(device).eval()
    encoded_parts = []
    batch_size = 8
    with torch.inference_mode():
        for start in range(0, len(pixels), batch_size):
            transformed = torch.stack(
                [transform(pixel) for pixel in pixels[start : start + batch_size]]
            ).to(device)
            encoded_parts.append(model.encode_pixels(transformed).float().cpu().numpy())
    unique_latents = np.concatenate(encoded_parts, axis=0)
    ordered = unique_latents[inverse]
    count = len(manifest.entries)
    return ordered[:count], ordered[count:]


def _verify_training_action_normalizer(dataset, identity: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    stats = identity.get("normalizer", {}).get("stats", {}).get("action")
    if not isinstance(stats, dict):
        raise RuntimeError("Table 2 training identity is missing its action normalizer")
    actions = np.asarray(dataset.get_col_data("action"), dtype=np.float64)
    actions = actions[~np.isnan(actions).any(axis=1)]
    actual_mean = np.mean(actions, axis=0)
    actual_std = np.std(actions, axis=0, ddof=1)
    recorded_mean = np.asarray(stats.get("mean", ()), dtype=np.float64)
    recorded_std = np.asarray(stats.get("std", ()), dtype=np.float64)
    if recorded_mean.shape != actual_mean.shape or recorded_std.shape != actual_std.shape:
        raise RuntimeError("Table 2 action normalizer width differs from the raw dataset")
    if not np.allclose(recorded_mean, actual_mean, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Table 2 training action mean is not the full-dataset mean")
    if not np.allclose(recorded_std, actual_std, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Table 2 training action scale is not the full-dataset sample std")
    if int(stats.get("std_correction", -1)) != 1:
        raise RuntimeError("Table 2 training action normalizer must use sample standard deviation")
    return {
        "source": "full_original_dataset",
        "std_correction": 1,
        "sample_count": int(stats.get("sample_count", len(actions))),
        "sha256": sha256_value(stats),
    }


def _score_candidate_actions(model, z_start, z_goal, candidates, *, chunk_states: int = 4):
    import numpy as np
    import torch

    scores = []
    with torch.inference_mode():
        for start in range(0, len(candidates), chunk_states):
            end = min(start + chunk_states, len(candidates))
            actions = torch.as_tensor(candidates[start:end], device="cuda:0", dtype=torch.float32)
            current = torch.as_tensor(z_start[start:end], device="cuda:0", dtype=torch.float32)
            goal = torch.as_tensor(z_goal[start:end], device="cuda:0", dtype=torch.float32)
            cost = model.get_cost_from_latents(
                current, goal, actions, score_horizon_blocks=int(model.action_horizon)
            )
            cost = cost.reshape(end - start, candidates.shape[1]).float().cpu().numpy()
            if not np.isfinite(cost).all():
                raise RuntimeError("a Table 2 B scorer returned non-finite fixed-pool costs")
            scores.append(cost)
    return np.concatenate(scores, axis=0)


def _score_artifact_paths(task: str, seed: int, arm: str, evaluation_seed: int):
    root = (
        FIXED_POOL_ROOT / task / "scorers" / f"training_seed_{int(seed)}"
        / arm / f"evaluation_seed_{int(evaluation_seed)}"
    )
    return root, root / "candidate_costs.npz", root / "rank_metrics.json"


def _immutable_npz(path: Path, identity: dict[str, Any], **arrays) -> Path:
    import numpy as np

    sidecar_path = path.with_suffix(".meta.json")
    previous = _read_json(sidecar_path)
    if path.exists() and previous is not None:
        if previous.get("identity") != identity or previous.get("identity_sha256") != sha256_value(identity):
            raise RuntimeError(f"immutable Table 2 diagnostic identity changed: {path}")
        if previous.get("artifact_sha256") != sha256_file(path):
            raise RuntimeError(f"immutable Table 2 diagnostic artifact changed: {path}")
        return path
    if previous is not None and not path.exists():
        raise RuntimeError(f"Table 2 diagnostic sidecar has no artifact: {path}")
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays):
                raise RuntimeError(f"unregistered Table 2 diagnostic array set changed: {path}")
            if any(not np.array_equal(existing[key], arrays[key]) for key in arrays):
                raise RuntimeError(f"unregistered Table 2 diagnostic arrays changed: {path}")
    else:
        _write_npz_atomic(path, **arrays)
    write_json_atomic(
        sidecar_path,
        {
            "identity": identity,
            "identity_sha256": sha256_value(identity),
            "artifact_sha256": sha256_file(path),
        },
    )
    return path


def _load_immutable_npz(path: Path, identity: dict[str, Any]):
    import numpy as np

    sidecar = _read_json(path.with_suffix(".meta.json"))
    if sidecar is None:
        if path.exists():
            return None
        return None
    if not path.is_file():
        raise RuntimeError(f"Table 2 diagnostic sidecar has no artifact: {path}")
    if sidecar.get("identity") != identity or sidecar.get("identity_sha256") != sha256_value(identity):
        raise RuntimeError(f"immutable Table 2 diagnostic identity changed: {path}")
    if sidecar.get("artifact_sha256") != sha256_file(path):
        raise RuntimeError(f"immutable Table 2 diagnostic artifact changed: {path}")
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def _immutable_jsonl(path: Path, identity: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    sidecar_path = path.with_suffix(".meta.json")
    previous = _read_json(sidecar_path)
    if path.exists() and previous is not None:
        if previous.get("identity") != identity or previous.get("identity_sha256") != sha256_value(identity):
            raise RuntimeError(f"immutable Table 2 diagnostic identity changed: {path}")
        if previous.get("artifact_sha256") != sha256_file(path):
            raise RuntimeError(f"immutable Table 2 diagnostic artifact changed: {path}")
        return path
    if previous is not None and not path.exists():
        raise RuntimeError(f"Table 2 diagnostic sidecar has no artifact: {path}")
    if path.exists():
        if _read_jsonl(path) != rows:
            raise RuntimeError(f"unregistered Table 2 diagnostic rows changed: {path}")
    else:
        _write_jsonl_atomic(path, rows)
    write_json_atomic(
        sidecar_path,
        {
            "identity": identity,
            "identity_sha256": sha256_value(identity),
            "artifact_sha256": sha256_file(path),
        },
    )
    return path


def _rank_saved_pool(*, costs, truth_rows, manifest, scorer_label):
    import numpy as np

    result = fixed_pool_rank_metrics(
        costs,
        truth_rows,
        state_count=len(manifest.entries),
        candidate_count=64,
    )
    for row, entry in zip(result["by_state"], manifest.entries):
        if row["state_id"] != source_state_id(manifest.task, entry.as_dict()):
            raise RuntimeError("fixed-pool truth state order differs from its frozen cohort")
    return result


def _po2_identity(*, task, seed, arm, evaluation_seed, checkpoint, pool_path, frozen):
    return {
        "schema_version": 1,
        "experiment": "cvpr_table2_fixed_pool_po2",
        "task": task,
        "training_seed": int(seed),
        "scorer_arm": arm,
        "evaluation_seed": int(evaluation_seed),
        "scorer_checkpoint": str(checkpoint.resolve()),
        "scorer_checkpoint_sha256": sha256_file(checkpoint),
        "pool_sha256": sha256_file(pool_path),
        "dataset_sha256": frozen["identity"]["datasets"][task]["sha256"],
        "baseline_candidate_index": 0,
        "algorithm": {
            "name": "Stage-B post-opt action correction",
            "inner_steps": 2,
            "step_size": 0.01,
            "max_rms_offset": 0.2,
            "score_horizon_blocks": 5,
            "action_coordinates": "common full-dataset training normalizer coordinates",
            "action_bound_mode": "none",
        },
    }


def _refine_first_pool_candidate(model, z_start, z_goal, candidates):
    import numpy as np
    import torch

    baseline = np.asarray(candidates[:, 0], dtype=np.float32)
    refined_parts = []
    before_parts = []
    after_parts = []
    for start in range(0, len(baseline), 8):
        end = min(start + 8, len(baseline))
        current = torch.as_tensor(z_start[start:end], device="cuda:0", dtype=torch.float32)
        goal = torch.as_tensor(z_goal[start:end], device="cuda:0", dtype=torch.float32)
        actions = torch.as_tensor(baseline[start:end], device="cuda:0", dtype=torch.float32)
        with torch.no_grad():
            before = model.get_cost_from_latents(
                current, goal, actions[:, None], score_horizon_blocks=5
            ).reshape(-1)
        refined = model.post_optimize_actions(
            current,
            goal,
            actions,
            step_size=0.01,
            inner_steps=2,
            max_rms_offset=0.2,
            score_horizon_blocks=5,
            collect_guidance_diagnostics=True,
        )
        with torch.no_grad():
            after = model.get_cost_from_latents(
                current, goal, refined[:, None], score_horizon_blocks=5
            ).reshape(-1)
        refined_parts.append(refined.detach().float().cpu().numpy())
        before_parts.append(before.float().cpu().numpy())
        after_parts.append(after.float().cpu().numpy())
    updated = np.concatenate(refined_parts, axis=0)
    before = np.concatenate(before_parts, axis=0)
    after = np.concatenate(after_parts, axis=0)
    rms = np.sqrt(np.mean(np.square(updated.astype(np.float64) - baseline.astype(np.float64)), axis=(1, 2)))
    if not np.isfinite(updated).all() or not np.isfinite(before).all() or not np.isfinite(after).all():
        raise RuntimeError("Table 2 PO2 produced non-finite actions or scores")
    if np.any(rms > 0.2 + 1e-6):
        raise RuntimeError("Table 2 PO2 exceeded its frozen RMS action displacement bound")
    return baseline, updated, before, after, rms


def _ensure_po2_truth(
    *, task: str, seed: int, arm: str, evaluation_seed: int, checkpoint: Path,
    scorer, baseline_actions, refined_actions, predicted_before, predicted_after,
    action_rms, pool, pool_path, truth_rows, manifest, cfg, dataset, frozen,
):
    import copy
    import numpy as np

    from source.common.eval import DatasetEvaluationSession

    root = FIXED_POOL_ROOT / task / "scorers" / f"training_seed_{seed}" / arm / f"evaluation_seed_{evaluation_seed}"
    identity = _po2_identity(
        task=task, seed=seed, arm=arm, evaluation_seed=evaluation_seed,
        checkpoint=checkpoint, pool_path=pool_path, frozen=frozen,
    )
    existing = _read_json(root / "po2_summary.json")
    if existing is not None:
        rows_path = root / "po2_outcomes.jsonl"
        actions_path = root / "po2_actions.npz"
        if existing.get("identity") != identity:
            raise RuntimeError(f"Table 2 PO2 identity changed: {root}")
        if (
            existing.get("outcomes_sha256") != sha256_file(rows_path)
            or existing.get("actions_sha256") != sha256_file(actions_path)
        ):
            raise RuntimeError(f"Table 2 PO2 artifact hash mismatch: {root}")
        return _read_jsonl(rows_path)

    branch_cfg = copy.deepcopy(cfg)
    branch_cfg.eval.eval_budget = 25
    branch_cfg.world.max_episode_steps = 50
    session = DatasetEvaluationSession(
        branch_cfg, task=task, dataset=dataset, cohort=manifest.to_evaluation_cohort()
    )
    episodes = _execute_fixed_candidate_branch(
        task=task,
        evaluation_seed=evaluation_seed,
        candidate_index=64,
        normalized_actions=refined_actions,
        manifest=manifest,
        session=session,
        branch_cfg=branch_cfg,
        dataset=dataset,
        actor_model=scorer,
        branch_state={},
        output_dir=root / "po2_replay_work",
    )
    baseline_by_slot = {
        int(row["slot"]): row for row in truth_rows if int(row["candidate_index"]) == 0
    }
    rows = []
    for slot, updated in enumerate(episodes):
        baseline = baseline_by_slot[slot]
        both_valid = bool(baseline["valid_at_25"] and updated["valid_at_25"])
        physical_improvement = (
            float(baseline["physical_cost_at_25"] - updated["physical_cost_at_25"])
            if both_valid
            else None
        )
        predicted_improvement = float(predicted_before[slot] - predicted_after[slot])
        rows.append(
            {
                **{key: baseline[key] for key in (
                    "task", "evaluation_seed", "state_id", "slot", "episode_id",
                    "row_index", "start_step",
                )},
                "scorer_arm": arm,
                "training_seed": int(seed),
                "baseline_candidate_index": 0,
                "baseline_success_by_25": bool(baseline["success_by_25"]),
                "po2_success_by_25": bool(updated["success_by_25"]),
                "success_delta": float(bool(updated["success_by_25"]) - bool(baseline["success_by_25"])),
                "baseline_physical_cost_at_25": baseline["physical_cost_at_25"],
                "po2_physical_cost_at_25": updated["physical_cost_at_25"],
                "both_valid_at_25": both_valid,
                "physical_improvement_at_25": physical_improvement,
                "predicted_cost_before": float(predicted_before[slot]),
                "predicted_cost_after": float(predicted_after[slot]),
                "predicted_improvement": predicted_improvement,
                "predicted_improved_true_worse": bool(
                    predicted_improvement > 1e-8
                    and physical_improvement is not None
                    and physical_improvement < -1e-8
                ),
                "normalized_action_rms": float(action_rms[slot]),
                "po2_valid_at_25": bool(updated["valid_at_25"]),
            }
        )
    rows_path = _immutable_jsonl(root / "po2_outcomes.jsonl", identity, rows)
    physical = _pool_physical_actions(
        refined_actions[:, None], session.process["action"], int(branch_cfg.plan_config.action_block)
    )[:, 0]
    arrays_path = _immutable_npz(
        root / "po2_actions.npz",
        identity,
        baseline_normalized=baseline_actions,
        refined_normalized=refined_actions,
        refined_physical=physical,
        predicted_cost_before=predicted_before,
        predicted_cost_after=predicted_after,
        normalized_action_rms=action_rms,
    )
    valid_improvements = [row["physical_improvement_at_25"] for row in rows if row["physical_improvement_at_25"] is not None]
    write_json_atomic(
        root / "po2_summary.json",
        {
            "identity": identity,
            "identity_sha256": sha256_value(identity),
            "outcomes": str(rows_path.relative_to(ROOT)),
            "outcomes_sha256": sha256_file(rows_path),
            "actions": str(arrays_path.relative_to(ROOT)),
            "actions_sha256": sha256_file(arrays_path),
            "state_count": len(rows),
            "paired_valid_at_25_count": len(valid_improvements),
            "mean_success_delta": float(np.mean([row["success_delta"] for row in rows])),
            "mean_physical_improvement_at_25": (
                None if not valid_improvements else float(np.mean(valid_improvements))
            ),
            "mean_predicted_improvement": float(np.mean([row["predicted_improvement"] for row in rows])),
            "predicted_improved_true_worse_count": sum(row["predicted_improved_true_worse"] for row in rows),
            "mean_normalized_action_rms": float(np.mean(action_rms)),
            "action_bound_mode": "none",
        },
    )
    return rows


def _run_table2_scorer_group(
    *, task: str, seed: int, arm: str, dataset, cfg_by_seed, manifests,
    pools, pool_paths, truths, frozen,
):
    import gc
    import numpy as np
    import torch

    from source.common.checkpoint import load_policy_or_model
    from source.common.eval import DatasetEvaluationSession

    checkpoint = _training_checkpoint_path(task, arm, seed)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Table 2 scorer checkpoint is missing: {checkpoint}")
    identity_path = checkpoint.parent.parent / "table2_training_identity.json"
    train_identity = _table2_artifact_identity(identity_path)
    if train_identity.get("task") != task or train_identity.get("arm") != arm or int(train_identity.get("training_seed", -1)) != int(seed):
        raise RuntimeError(f"Table 2 scorer training identity mismatch: {identity_path}")
    normalizer_audit = _verify_training_action_normalizer(dataset, train_identity)
    policy, resolved = load_policy_or_model(str(checkpoint.resolve()))
    if resolved is None:
        raise RuntimeError(f"could not load Table 2 scorer checkpoint: {checkpoint}")
    model = getattr(policy, "model", policy).eval().float().to("cuda:0")
    model.requires_grad_(False)
    summaries = []
    all_rank_rows = []
    all_po2_rows = []
    for evaluation_seed in _load_static_config()["evaluation_seeds"]:
        manifest = manifests[int(evaluation_seed)]
        pool = pools[int(evaluation_seed)]
        pool_path = pool_paths[int(evaluation_seed)]
        truth_rows = truths[int(evaluation_seed)]
        cfg = cfg_by_seed[int(evaluation_seed)]
        session = DatasetEvaluationSession(
            cfg, task=task, dataset=dataset, cohort=manifest.to_evaluation_cohort()
        )
        z_start, z_goal = _encode_manifest_observations(
            model, dataset, session.transform["pixels"], manifest, "cuda:0"
        )
        score_identity = {
            "schema_version": 1,
            "task": task,
            "training_seed": int(seed),
            "arm": arm,
            "checkpoint_sha256": sha256_file(checkpoint),
            "training_identity_sha256": sha256_file(identity_path),
            "action_normalizer_sha256": normalizer_audit["sha256"],
            "pool_sha256": sha256_file(pool_path),
            "cohort_sha256": manifest.computed_sha256,
            "score_horizon_blocks": int(model.action_horizon),
            "precision": "float32",
        }
        root, costs_path, rank_path = _score_artifact_paths(
            task, seed, arm, int(evaluation_seed)
        )
        root.mkdir(parents=True, exist_ok=True)
        stored_costs = _load_immutable_npz(costs_path, score_identity)
        if stored_costs is None:
            costs = _score_candidate_actions(
                model, z_start, z_goal, np.asarray(pool["candidates"]), chunk_states=4
            )
            _immutable_npz(costs_path, score_identity, candidate_costs=costs)
        else:
            costs = stored_costs["candidate_costs"]
        ranking = _rank_saved_pool(
            costs=costs,
            truth_rows=truth_rows,
            manifest=manifest,
            scorer_label=arm,
        )
        rank_artifact = {
            "identity": score_identity,
            "identity_sha256": sha256_value(score_identity),
            **{key: value for key, value in ranking.items() if key != "by_state"},
        }
        previous_rank = _read_json(rank_path)
        if previous_rank is not None and previous_rank != rank_artifact:
            raise RuntimeError(f"fixed-pool rank summary changed: {rank_path}")
        write_json_atomic(rank_path, rank_artifact)
        score_rows = [
            {
                "task": task,
                "training_seed": int(seed),
                "scorer_arm": arm,
                "evaluation_seed": int(evaluation_seed),
                **row,
            }
            for row in ranking["by_state"]
        ]
        _immutable_jsonl(root / "rank_by_state.jsonl", score_identity, score_rows)
        baseline, refined, before, after, action_rms = _refine_first_pool_candidate(
            model, z_start, z_goal, np.asarray(pool["candidates"])
        )
        po2_rows = _ensure_po2_truth(
            task=task,
            seed=seed,
            arm=arm,
            evaluation_seed=int(evaluation_seed),
            checkpoint=checkpoint,
            scorer=model,
            baseline_actions=baseline,
            refined_actions=refined,
            predicted_before=before,
            predicted_after=after,
            action_rms=action_rms,
            pool=pool,
            pool_path=pool_path,
            truth_rows=truth_rows,
            manifest=manifest,
            cfg=cfg,
            dataset=dataset,
            frozen=frozen,
        )
        summaries.append(
            {
                "evaluation_seed": int(evaluation_seed),
                "rank": {key: value for key, value in ranking.items() if key != "by_state"},
                "po2": _read_json(root / "po2_summary.json"),
            }
        )
        all_rank_rows.extend(score_rows)
        all_po2_rows.extend(po2_rows)
        print(
            f"[{task}/{seed}/{arm}/{evaluation_seed}] rescored 50 x 64 candidates; PO2 branches 50/50",
            flush=True,
        )

    rank_by_state = all_rank_rows
    po2_by_state = all_po2_rows
    valid_po2 = [row for row in po2_by_state if row["physical_improvement_at_25"] is not None]
    scorer_summary = {
        "schema_version": 1,
        "status": "complete",
        "task": task,
        "training_seed": int(seed),
        "arm": arm,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_identity_sha256": sha256_file(identity_path),
        "action_normalizer_audit": normalizer_audit,
        "state_count": len(rank_by_state),
        "candidate_count": 64,
        "candidate_pool": {
            "complete_pool_state_count": sum(int(item["rank"]["complete_pool_state_count"]) for item in summaries),
            **{
                key: (
                    float(np.mean([item["rank"][key] for item in summaries if item["rank"][key] is not None]))
                    if any(item["rank"][key] is not None for item in summaries)
                    else None
                )
                for key in (
                    "mean_selected_success_by_25",
                    "mean_pool_success_fraction_25",
                    "mean_pool_has_success_by_25",
                    "mean_valid_candidate_count_25",
                    "mean_complete_pool_regret_at_25",
                    "mean_valid_subpool_regret_at_25",
                    "mean_valid_subpool_spearman",
                )
            }
        },
        "po2": {
            "state_count": len(po2_by_state),
            "paired_valid_at_25_count": len(valid_po2),
            "mean_success_delta": float(np.mean([row["success_delta"] for row in po2_by_state])),
            "mean_physical_improvement_at_25": (
                None if not valid_po2 else float(np.mean([row["physical_improvement_at_25"] for row in valid_po2]))
            ),
            "mean_predicted_improvement": float(np.mean([row["predicted_improvement"] for row in po2_by_state])),
            "predicted_improved_true_worse_count": sum(row["predicted_improved_true_worse"] for row in po2_by_state),
            "mean_normalized_action_rms": float(np.mean([row["normalized_action_rms"] for row in po2_by_state])),
        },
        "per_evaluation_seed": summaries,
    }
    summary_root = FIXED_POOL_ROOT / task / "scorers" / f"training_seed_{seed}" / arm
    summary_root.mkdir(parents=True, exist_ok=True)
    summary_identity = {
        "task": task,
        "training_seed": int(seed),
        "arm": arm,
        "checkpoint_sha256": sha256_file(checkpoint),
        "evaluation_seed_metrics": summaries,
    }
    previous_summary = _read_json(summary_root / "summary.json")
    if previous_summary is not None and previous_summary != scorer_summary:
        raise RuntimeError(f"fixed-pool scorer summary changed: {summary_root}")
    write_json_atomic(summary_root / "summary.json", scorer_summary)
    _immutable_jsonl(summary_root / "rank_all_states.jsonl", summary_identity, rank_by_state)
    _immutable_jsonl(summary_root / "po2_all_states.jsonl", summary_identity, po2_by_state)
    model.to("cpu")
    del model, policy
    gc.collect()
    torch.cuda.empty_cache()
    return scorer_summary


def _fixed_pool_group(task: str, training_seeds: tuple[int, ...]) -> dict[str, Any]:
    import copy
    import gc
    import numpy as np
    import torch

    from source.common.checkpoint import load_policy_or_model
    from source.common.eval import DatasetEvaluationSession, get_dataset
    from source.common.round3_phase1 import CohortManifest

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids = [item.strip() for item in visible.split(",") if item.strip()]
    if len(ids) != 1 or not ids[0].isdigit() or int(ids[0]) not in range(4):
        raise RuntimeError(f"fixed-pool diagnostics require one explicit GPU0–3; got {visible!r}")
    gpu = int(ids[0])
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    frozen = _load_frozen()
    static = _load_static_config()
    shared_action_normalizer = None
    for seed in training_seeds:
        for arm in ("b_only_clean", "shared_recorded_control", "shared_pred_detach", "shared_pred_full"):
            checkpoint = _training_checkpoint_path(task, arm, int(seed))
            if not checkpoint.is_file():
                raise FileNotFoundError(f"fixed-pool scorer checkpoint is missing: {checkpoint}")
            identity = _table2_artifact_identity(checkpoint.parent.parent / "table2_training_identity.json")
            progress = _read_json(checkpoint.parent.parent / "table2_training_progress.json", {})
            if progress.get("status") != "complete":
                raise RuntimeError(f"fixed-pool scorer training is not complete: {checkpoint}")
            if identity.get("run_id") != run_id(task, arm, int(seed)):
                raise RuntimeError(f"fixed-pool scorer has the wrong Table 2 run identity: {checkpoint}")
            action_stats = identity.get("normalizer", {}).get("stats", {}).get("action")
            if not isinstance(action_stats, dict):
                raise RuntimeError(f"Table 2 B scorer is missing action normalizer stats: {checkpoint}")
            if shared_action_normalizer is None:
                shared_action_normalizer = action_stats
            elif action_stats != shared_action_normalizer:
                raise RuntimeError(
                    f"Table 2 B scorers do not share the common task normalizer: {task}/{seed}/{arm}"
                )

    dataset_path = ROOT / DATASETS[task]
    base_cfg = _make_fixed_pool_eval_config(task, int(static["evaluation_seeds"][0]))
    dataset = get_dataset(base_cfg, str(dataset_path.resolve()))
    actor_path = ROOT / frozen["identity"]["fixed_actor_checkpoints"][task]["archived_checkpoint"]
    actor, resolved = load_policy_or_model(str(actor_path.resolve()))
    if resolved is None:
        raise RuntimeError(f"could not load fixed Table 1 actor: {actor_path}")
    actor_model = getattr(actor, "model", actor).eval().float().to("cuda:0")
    actor_model.requires_grad_(False)

    manifests: dict[int, Any] = {}
    cfg_by_seed: dict[int, Any] = {}
    pools: dict[int, Any] = {}
    pool_paths: dict[int, Path] = {}
    truths: dict[int, list[dict[str, Any]]] = {}
    fixed_actor_state_rows = []
    for evaluation_seed in static["evaluation_seeds"]:
        evaluation_seed = int(evaluation_seed)
        cohort_info = frozen["identity"]["evaluation_cohorts"][f"{task}/seed_{evaluation_seed}"]
        manifest = CohortManifest.load(ROOT / cohort_info["path"])
        if manifest.computed_sha256 != cohort_info["sha256"]:
            raise RuntimeError(f"Table 1 frozen cohort hash changed: {task}/{evaluation_seed}")
        manifest_entries = [entry.as_dict() for entry in manifest.entries]
        state_ids = [source_state_id(task, entry) for entry in manifest_entries]
        if len(set(state_ids)) != len(state_ids):
            raise RuntimeError(f"frozen cohort contains duplicate source states: {task}/{evaluation_seed}")
        cfg = _make_fixed_pool_eval_config(task, evaluation_seed)
        session = DatasetEvaluationSession(
            cfg, task=task, dataset=dataset, cohort=manifest.to_evaluation_cohort()
        )
        pool, pool_path = _load_or_make_fixed_pool(
            task=task,
            evaluation_seed=evaluation_seed,
            actor=actor_model,
            actor_path=actor_path,
            manifest=manifest,
            cfg=cfg,
            dataset=dataset,
            frozen=frozen,
        )
        branch_rows, _ = _ensure_candidate_branch_truth(
            task=task,
            evaluation_seed=evaluation_seed,
            manifest=manifest,
            cfg=cfg,
            dataset=dataset,
            actor_model=actor_model,
            pool=pool,
            pool_path=pool_path,
            frozen=frozen,
        )
        fixed_actor_rank = _rank_saved_pool(
            costs=np.asarray(pool["table1_fixed_actor_costs"]),
            truth_rows=branch_rows,
            manifest=manifest,
            scorer_label="table1_fixed_actor",
        )
        fixed_actor_state_rows.extend(
            {
                "task": task,
                "scorer": "table1_fixed_actor",
                "training_seed": None,
                "evaluation_seed": evaluation_seed,
                **row,
            }
            for row in fixed_actor_rank["by_state"]
        )
        manifests[evaluation_seed] = manifest
        cfg_by_seed[evaluation_seed] = cfg
        pools[evaluation_seed] = pool
        pool_paths[evaluation_seed] = pool_path
        truths[evaluation_seed] = branch_rows
        print(
            f"[{task}/{evaluation_seed}] fixed Table 1 pool and 3,200 native outcomes complete",
            flush=True,
        )
        del session

    global_truths = []
    global_costs = []
    for cohort_index, evaluation_seed in enumerate(static["evaluation_seeds"]):
        evaluation_seed = int(evaluation_seed)
        global_costs.append(np.asarray(pools[evaluation_seed]["table1_fixed_actor_costs"]))
        global_truths.extend(
            {**row, "slot": cohort_index * 50 + int(row["slot"])}
            for row in truths[evaluation_seed]
        )
    fixed_actor_rank_summary = fixed_pool_rank_metrics(
        np.concatenate(global_costs, axis=0),
        global_truths,
        state_count=300,
        candidate_count=64,
    )
    fixed_actor_root = FIXED_POOL_ROOT / task / "fixed_actor"
    fixed_actor_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        fixed_actor_root / "summary.json",
        {
            "schema_version": 1,
            "task": task,
            "scorer": "table1_fixed_actor",
            "status": "complete",
            "pool_state_count": 300,
            "candidate_count": 64,
            "metrics": {key: value for key, value in fixed_actor_rank_summary.items() if key != "by_state"},
        },
    )
    _write_jsonl_atomic(fixed_actor_root / "rank_by_state.jsonl", fixed_actor_state_rows)

    scorer_summaries = []
    for seed in training_seeds:
        for arm in ("b_only_clean", "shared_recorded_control", "shared_pred_detach", "shared_pred_full"):
            scorer_summaries.append(
                _run_table2_scorer_group(
                    task=task,
                    seed=int(seed),
                    arm=arm,
                    dataset=dataset,
                    cfg_by_seed=cfg_by_seed,
                    manifests=manifests,
                    pools=pools,
                    pool_paths=pool_paths,
                    truths=truths,
                    frozen=frozen,
                )
            )
    result = {
        "schema_version": 1,
        "task": task,
        "gpu": gpu,
        "training_seeds": list(training_seeds),
        "status": "complete",
        "pool_states": 300,
        "candidate_branches": 300 * 64,
        "po2_branches": len(training_seeds) * 4 * 300,
        "action_bound_mode": "none",
        "table2_action_normalizer_sha256": sha256_value(shared_action_normalizer),
        "fixed_actor_normalizer_rule": "Table 1 fixed actor was trained from the same full original dataset with sample std; Table 2 B checkpoint stats are rechecked against that raw dataset",
        "candidate_pool_noise_policy": "one deterministic frozen-actor pool per Table 1 evaluation seed; all B scorers use these saved actions",
        "fixed_actor_metrics": {
            key: value for key, value in fixed_actor_rank_summary.items() if key != "by_state"
        },
        "scorer_groups_complete": len(scorer_summaries),
    }
    task_root = FIXED_POOL_ROOT / task
    write_json_atomic(task_root / "group_status.json", result)
    write_json_atomic(task_root / "summary.json", result)
    actor_model.to("cpu")
    del actor_model, actor
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _fixed_pool_worker(gpu: int, tasks: list[str], training_seeds: tuple[int, ...]):
    results = []
    for task in tasks:
        command = [
            sys.executable,
            "scripts/cvpr_table2.py",
            "fixed-pool-group",
            "--task",
            task,
            "--training-seeds",
            *[str(seed) for seed in training_seeds],
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        log_path = FIXED_POOL_ROOT / task / "launcher.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with log_path.open("ab", buffering=0) as log_stream:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
            code = process.wait()
        entry = {
            "task": task,
            "gpu": int(gpu),
            "training_seeds": list(training_seeds),
            "return_code": int(code),
            "wall_seconds": time.perf_counter() - started,
        }
        results.append(entry)
        if code != 0:
            write_json_atomic(FIXED_POOL_ROOT / task / "launcher_failure.json", entry)
            break
    return results


def command_fixed_pool(args) -> None:
    frozen = _load_frozen()
    tasks = tuple(args.tasks or TASKS)
    seeds = tuple(int(value) for value in args.training_seeds)
    if not seeds or any(value not in SEEDS for value in seeds):
        raise ValueError(f"--training-seeds must be a non-empty subset of {SEEDS}")
    if tuple(sorted(seeds, key=SEEDS.index)) != seeds:
        raise ValueError("fixed-pool training seeds must follow the frozen seed order")
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    _check_gpu_vram(gpus, float(frozen["resources"]["training_vram_margin_gib"]))
    free_disk_gib = shutil.disk_usage(OUTPUT_ROOT.parent).free / (1024**3)
    if free_disk_gib < float(frozen["resources"]["stop_dispatch_free_disk_gib"]):
        raise RuntimeError(f"only {free_disk_gib:.1f} GiB disk space is free; refusing fixed-pool dispatch")
    assignments = [tasks[index::len(gpus)] for index in range(len(gpus))]
    dispatch = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(_fixed_pool_worker, gpu, assigned, seeds)
            for gpu, assigned in zip(gpus, assignments)
        ]
        for future in futures:
            dispatch.extend(future.result())
    write_json_atomic(
        FIXED_POOL_ROOT / "dispatch.json",
        {
            "schema_version": 1,
            "training_seeds": list(seeds),
            "groups": dispatch,
            "status": "failed" if any(row["return_code"] for row in dispatch) else "complete",
        },
    )
    if len(dispatch) != len(tasks) or any(row["return_code"] for row in dispatch):
        raise RuntimeError("one or more fixed-pool diagnostic task groups failed")


def command_fixed_pool_group(args) -> None:
    seeds = tuple(int(value) for value in args.training_seeds)
    result = _fixed_pool_group(args.task, seeds)
    print(json.dumps(result, ensure_ascii=False))


def _evaluation_conditions() -> list[dict[str, Any]]:
    shared = (
        "shared_recorded_control",
        "shared_pred_detach",
        "shared_pred_full",
    )
    conditions = [
        {
            "condition_id": f"own_a_p0__{arm}",
            "panel": "own_a_p0",
            "actor_arm": arm,
            "scorer_arm": None,
            "mode": "P0",
            "guidance_mode": "none",
        }
        for arm in ("a_only", *shared)
    ]
    conditions.extend(
        {
            "condition_id": f"own_a_p3__{arm}",
            "panel": "own_shared_p3",
            "actor_arm": arm,
            "scorer_arm": None,
            "mode": "P3",
            "guidance_mode": "none",
        }
        for arm in shared
    )
    conditions.extend(
        {
            "condition_id": f"own_a_p0_po2__{arm}",
            "panel": "own_shared_p0_po2",
            "actor_arm": arm,
            "scorer_arm": None,
            "mode": "P0",
            "guidance_mode": "post_opt",
        }
        for arm in shared
    )
    conditions.extend(
        {
            "condition_id": f"fixed_actor_p3__{arm}",
            "panel": "fixed_actor_trained_b_p3",
            "actor_arm": "table1_fixed_actor",
            "scorer_arm": arm,
            "mode": "P3",
            "guidance_mode": "none",
        }
        for arm in ("b_only_clean", *shared)
    )
    if len(conditions) != 14 or len({item["condition_id"] for item in conditions}) != 14:
        raise RuntimeError("Table 2 closed-loop condition matrix must contain 14 unique cells")
    return conditions


def _eval_cell_root(training_seed: int, task: str, evaluation_seed: int, condition_id: str) -> Path:
    return (
        OUTPUT_ROOT
        / "evals"
        / f"training_seed_{training_seed}"
        / task
        / f"evaluation_seed_{evaluation_seed}"
        / condition_id
    )


def _eval_cell_identity(
    *, task: str, training_seed: int, evaluation_seed: int,
    condition: dict[str, Any], frozen: dict[str, Any],
) -> dict[str, Any]:
    actor_arm = condition["actor_arm"]
    scorer_arm = condition["scorer_arm"]
    actor_path = (
        Path(frozen["identity"]["fixed_actor_checkpoints"][task]["archived_checkpoint"])
        if actor_arm == "table1_fixed_actor"
        else _training_checkpoint_path(task, actor_arm, training_seed)
    )
    scorer_path = None
    if scorer_arm is not None:
        scorer_path = _training_checkpoint_path(task, scorer_arm, training_seed)
    elif condition["mode"] == "P3" or condition["guidance_mode"] != "none":
        scorer_path = actor_path
    for path in (actor_path, scorer_path):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"Table 2 evaluation checkpoint is missing: {path}")
    cohort_path = ROOT / frozen["identity"]["evaluation_cohorts"][
        f"{task}/seed_{evaluation_seed}"
    ]["path"]
    return {
        "schema_version": 1,
        "experiment": "cvpr_table2_v1",
        "training_seed": int(training_seed),
        "task": task,
        "evaluation_seed": int(evaluation_seed),
        "environment_seed": int(evaluation_seed) + 10000,
        "policy_seed": int(evaluation_seed) + 20000,
        "condition": dict(condition),
        "protocol": {
            "goal_offset_steps": 25,
            "episodes": 50,
            "eval_budget": 50,
            "generation_scoring_execution_steps": [25, 25, 25],
            "candidate_count": 64,
            "action_flow_steps": 2,
            "integrator": "euler",
            "action_bound_mode": "none",
            "precision": "float32",
            "autocast": False,
            "tf32": False,
            "compile": False,
        },
        "actor_checkpoint": str(actor_path.resolve()),
        "actor_checkpoint_sha256": sha256_file(actor_path),
        "scorer_checkpoint": None if scorer_path is None else str(scorer_path.resolve()),
        "scorer_checkpoint_sha256": None
        if scorer_path is None
        else sha256_file(scorer_path),
        "cohort_path": str(cohort_path.resolve()),
        "cohort_sha256": sha256_file(cohort_path),
    }


def _complete_eval_attempt(cell_root: Path, identity_sha256: str) -> Path | None:
    for attempt in sorted(cell_root.glob("attempt_*")):
        result_path = attempt / "result.json"
        trace_path = attempt / "episodes.jsonl"
        if not result_path.is_file() or not trace_path.is_file():
            continue
        try:
            result = _read_json(result_path)
        except (OSError, json.JSONDecodeError):
            continue
        metadata = result.get("cvpr_table2", {})
        if (
            result.get("status") == "ok"
            and len(result.get("episodes", [])) == 50
            and metadata.get("identity_sha256") == identity_sha256
            and metadata.get("trace_sha256") == sha256_file(trace_path)
        ):
            return attempt
    return None


def _run_evaluation_group(task: str, training_seed: int) -> dict[str, Any]:
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from source.common.checkpoint import load_policy_or_model
    from source.common.eval import EvaluationIdentity, compose_eval_config, get_dataset
    from source.common.round3_phase1 import CohortManifest
    from source.common.round4_eval import run_round4_evaluation

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_ids = [item.strip() for item in visible.split(",") if item.strip()]
    if len(visible_ids) != 1 or not visible_ids[0].isdigit() or int(visible_ids[0]) not in range(4):
        raise RuntimeError(f"Table 2 evaluation requires one explicit GPU0–3; got {visible!r}")
    gpu = int(visible_ids[0])
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    frozen = _load_frozen()
    static = _load_static_config()
    model_paths = {
        arm: _training_checkpoint_path(task, arm, training_seed)
        for arm in ARMS
    }
    models = {}
    for arm, checkpoint_path in model_paths.items():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Table 2 checkpoint is missing: {checkpoint_path}")
        model, resolved = load_policy_or_model(str(checkpoint_path.resolve()))
        if resolved is None:
            raise RuntimeError(f"could not resolve Table 2 checkpoint: {checkpoint_path}")
        model = getattr(model, "model", model).eval().float()
        model.requires_grad_(False)
        models[arm] = model
    fixed_path = ROOT / frozen["identity"]["fixed_actor_checkpoints"][task]["archived_checkpoint"]
    fixed_actor, resolved = load_policy_or_model(str(fixed_path.resolve()))
    if resolved is None:
        raise RuntimeError(f"could not resolve frozen Table 1 actor: {fixed_path}")
    fixed_actor = getattr(fixed_actor, "model", fixed_actor).eval().float()
    fixed_actor.requires_grad_(False)
    models["table1_fixed_actor"] = fixed_actor

    dataset_path = ROOT / DATASETS[task]
    base_cfg = compose_eval_config(task)
    base_cfg.eval.dataset_name = str(dataset_path.resolve())
    base_cfg.eval.benchmark_dataset_name = str(DATASETS[task])
    base_cfg.eval.num_eval = 50
    base_cfg.eval.goal_offset_steps = 25
    base_cfg.eval.eval_budget = 50
    base_cfg.world.num_envs = 50
    base_cfg.world.max_episode_steps = 100
    base_cfg.plan_config.horizon = 5
    base_cfg.plan_config.receding_horizon = 5
    base_cfg.plan_config.action_block = 5
    base_cfg.output.save_video = False
    base_cfg.solver.device = "cuda:0"
    base_cfg.solver.batch_size = 1
    base_cfg.solver.num_samples = 300
    base_cfg.solver.n_steps = 30
    base_cfg.solver.topk = 30
    base_cfg.solver.var_scale = 1.0
    dataset = get_dataset(base_cfg, base_cfg.eval.dataset_name)
    group_root = OUTPUT_ROOT / "evals/groups" / f"training_seed_{training_seed}" / task
    group_root.mkdir(parents=True, exist_ok=True)
    conditions = _evaluation_conditions()
    entries = []
    errors = []

    for evaluation_seed in static["evaluation_seeds"]:
        manifest_path = ROOT / frozen["identity"]["evaluation_cohorts"][
            f"{task}/seed_{evaluation_seed}"
        ]["path"]
        manifest = CohortManifest.load(manifest_path)
        for condition in conditions:
            cell_identity = _eval_cell_identity(
                task=task,
                training_seed=training_seed,
                evaluation_seed=int(evaluation_seed),
                condition=condition,
                frozen=frozen,
            )
            identity_sha256 = sha256_value(cell_identity)
            cell_root = _eval_cell_root(
                training_seed, task, int(evaluation_seed), condition["condition_id"]
            )
            previous = _complete_eval_attempt(cell_root, identity_sha256)
            if previous is not None:
                entries.append(
                    {
                        "evaluation_seed": int(evaluation_seed),
                        "condition_id": condition["condition_id"],
                        "status": "already_complete",
                        "attempt": previous.name,
                    }
                )
                continue
            prior_attempts = [
                int(path.name.removeprefix("attempt_"))
                for path in cell_root.glob("attempt_*")
                if path.name.removeprefix("attempt_").isdigit()
            ]
            attempt_number = max(prior_attempts, default=0) + 1
            attempt_dir = cell_root / f"attempt_{attempt_number:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            write_json_atomic(
                attempt_dir / "cell_identity.json",
                {"identity": cell_identity, "identity_sha256": identity_sha256},
            )

            cfg = copy.deepcopy(base_cfg)
            cfg.seed = int(evaluation_seed) + 10000
            cfg.eval.policy_seed = int(evaluation_seed) + 20000
            cfg.solver.seed = int(evaluation_seed) + 20000
            OmegaConf.save(cfg, attempt_dir / "resolved_config.yaml", resolve=True)
            torch.manual_seed(int(evaluation_seed) + 20000)
            np.random.seed((int(evaluation_seed) + 20000) % (2**32 - 1))
            actor = models[condition["actor_arm"]]
            scorer = (
                None
                if condition["scorer_arm"] is None
                else models[condition["scorer_arm"]]
            )
            actor_path = Path(cell_identity["actor_checkpoint"])
            actor_stage = (
                "stage_ab"
                if condition["mode"] == "P3" or condition["guidance_mode"] != "none"
                else "stage_a"
            )
            evaluation_identity = EvaluationIdentity(
                entrypoint="cvpr_table2",
                policy_kind="fast_lewam",
                checkpoint=str(actor_path),
                epoch=10,
                stage=actor_stage,
                guidance_mode=str(condition["guidance_mode"]),
                guidance_step_size=0.01,
                guidance_last_steps=2,
                guidance_inner_steps=2 if condition["guidance_mode"] == "post_opt" else 1,
                guidance_max_rms_offset=0.2,
            )
            verifier_metadata = None
            if scorer is not None:
                verifier_metadata = {
                    "kind": "table2_trained_b",
                    "training_seed": int(training_seed),
                    "arm": str(condition["scorer_arm"]),
                    "checkpoint": cell_identity["scorer_checkpoint"],
                    "checkpoint_sha256": cell_identity["scorer_checkpoint_sha256"],
                }
            try:
                run_round4_evaluation(
                    cfg,
                    task=task,
                    policy_or_model=actor,
                    verifier_policy_or_model=scorer,
                    verifier_metadata=verifier_metadata,
                    mode=str(condition["mode"]),
                    identity=evaluation_identity,
                    manifest=manifest,
                    output_dir=attempt_dir,
                    trace_output_dir=attempt_dir,
                    device="cuda:0",
                    trace=True,
                    candidate_count=64,
                    flow_steps=2,
                    action_flow_steps=2,
                    solver_batch_size=1,
                    candidate_batch_size=64,
                    action_flow_integrator="euler",
                    action_bound_mode="none",
                    bf16_proposal=False,
                    bf16_verifier=False,
                    optimize_proposal=False,
                    cache_goal_latent=False,
                    bf16_encode=False,
                    guidance_mode=str(condition["guidance_mode"]),
                    guidance_step_size=0.01,
                    guidance_last_steps=2,
                    guidance_inner_steps=2 if condition["guidance_mode"] == "post_opt" else 1,
                    guidance_max_rms_offset=0.2,
                    allowed_protocol_variants=("legacy",),
                    allow_solver_config_override=True,
                    allow_evaluation_seed_override=True,
                    allow_cohort_seed_mismatch=True,
                    policy_seed=int(evaluation_seed) + 20000,
                    execute_steps=25,
                    score_horizon_blocks=5,
                    video_slots=0,
                    dataset=dataset,
                )
                result_path = attempt_dir / "result.json"
                trace_path = attempt_dir / "episodes.jsonl"
                result = _read_json(result_path)
                if result.get("status") != "ok" or len(result.get("episodes", [])) != 50:
                    raise RuntimeError("Table 2 evaluator did not produce 50 successful episodes")
                if not trace_path.is_file():
                    raise RuntimeError("Table 2 evaluator did not produce episodes.jsonl")
                result["cvpr_table2"] = {
                    "identity": cell_identity,
                    "identity_sha256": identity_sha256,
                    "trace_sha256": sha256_file(trace_path),
                    "gpu": gpu,
                    "cuda_visible_devices": visible,
                    "evaluator_return_status": "ok",
                }
                write_json_atomic(result_path, result)
                entry = {
                    "evaluation_seed": int(evaluation_seed),
                    "condition_id": condition["condition_id"],
                    "status": "complete",
                    "attempt": attempt_dir.name,
                    "success_rate": float(result["success_rate"]),
                    "episodes": len(result["episodes"]),
                }
            except Exception as exc:
                failure = {
                    "status": "failed",
                    "identity_sha256": identity_sha256,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                write_json_atomic(attempt_dir / "failure.json", failure)
                entry = {
                    "evaluation_seed": int(evaluation_seed),
                    "condition_id": condition["condition_id"],
                    "status": "failed",
                    "attempt": attempt_dir.name,
                    "error": repr(exc),
                }
                errors.append(entry)
            entries.append(entry)
            write_json_atomic(
                group_root / "group_status.json",
                {
                    "schema_version": 1,
                    "training_seed": int(training_seed),
                    "task": task,
                    "gpu": gpu,
                    "status": "running",
                    "completed_cells": sum(item["status"] in {"complete", "already_complete"} for item in entries),
                    "failed_cells": len(errors),
                    "expected_cells": 84,
                    "cells": entries,
                },
            )
    final = {
        "schema_version": 1,
        "training_seed": int(training_seed),
        "task": task,
        "gpu": gpu,
        "status": "failed" if errors else "complete",
        "completed_cells": sum(item["status"] in {"complete", "already_complete"} for item in entries),
        "failed_cells": len(errors),
        "expected_cells": 84,
        "cells": entries,
        "failures": errors,
    }
    write_json_atomic(group_root / "group_status.json", final)
    if errors:
        raise RuntimeError(f"Table 2 eval group {training_seed}/{task} had {len(errors)} failures")
    return final


def _eval_worker(gpu: int, groups: list[tuple[str, int]]) -> list[dict[str, Any]]:
    output = []
    for task, training_seed in groups:
        command = [
            sys.executable,
            "scripts/cvpr_table2.py",
            "eval-group",
            "--task",
            task,
            "--training-seed",
            str(training_seed),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        log_path = OUTPUT_ROOT / "evals/groups" / f"training_seed_{training_seed}" / task / "launcher.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as log_stream:
            started = time.perf_counter()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
            code = process.wait()
        item = {
            "task": task,
            "training_seed": int(training_seed),
            "gpu": int(gpu),
            "return_code": int(code),
            "wall_seconds": time.perf_counter() - started,
        }
        output.append(item)
        if code != 0:
            write_json_atomic(
                OUTPUT_ROOT / "evals/groups" / f"training_seed_{training_seed}" / task / "launcher_failure.json",
                item,
            )
    return output


def command_eval(args) -> None:
    frozen = _load_frozen()
    tasks = tuple(args.tasks or TASKS)
    seeds = tuple(int(value) for value in args.training_seeds)
    if not seeds or any(value not in SEEDS for value in seeds):
        raise ValueError(f"--training-seeds must be a non-empty subset of {SEEDS}")
    if tuple(sorted(seeds, key=SEEDS.index)) != seeds:
        raise ValueError("Table 2 training seeds must be evaluated in the frozen order")
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    _check_gpu_vram(gpus, float(frozen["resources"]["training_vram_margin_gib"]))
    free_disk_gib = shutil.disk_usage(OUTPUT_ROOT.parent).free / (1024**3)
    if free_disk_gib < float(frozen["resources"]["stop_dispatch_free_disk_gib"]):
        raise RuntimeError(f"only {free_disk_gib:.1f} GiB disk space is free; refusing evaluation dispatch")
    groups = [(task, seed) for seed in seeds for task in tasks]
    assignments = [groups[index:: len(gpus)] for index in range(len(gpus))]
    results = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(_eval_worker, gpu, assigned)
            for gpu, assigned in zip(gpus, assignments)
        ]
        for future in futures:
            results.extend(future.result())
    write_json_atomic(OUTPUT_ROOT / "summary/evaluation_dispatch.json", {
        "schema_version": 1,
        "groups": results,
        "status": "failed" if any(item["return_code"] != 0 for item in results) else "complete",
    })
    if any(item["return_code"] != 0 for item in results):
        raise RuntimeError("one or more Table 2 evaluation groups failed; inspect group_status.json")
    print(json.dumps({"evaluation_groups": len(results), "cells_per_group": 84}, ensure_ascii=False))


def command_eval_group(args) -> None:
    if args.training_seed not in SEEDS or args.task not in TASKS:
        raise ValueError("evaluation group is outside the frozen Table 2 matrix")
    result = _run_evaluation_group(args.task, int(args.training_seed))
    print(json.dumps({"status": result["status"], "group": f"{args.training_seed}/{args.task}"}))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_complete_eval_cell(task, training_seed, evaluation_seed, condition, frozen):
    import numpy as np

    identity = _eval_cell_identity(
        task=task,
        training_seed=training_seed,
        evaluation_seed=evaluation_seed,
        condition=condition,
        frozen=frozen,
    )
    identity_sha = sha256_value(identity)
    cell_root = _eval_cell_root(
        training_seed, task, evaluation_seed, condition["condition_id"]
    )
    attempt = _complete_eval_attempt(cell_root, identity_sha)
    if attempt is None:
        return None
    result = _read_json(attempt / "result.json")
    trace_path = attempt / "episodes.jsonl"
    cohort_path = ROOT / frozen["identity"]["evaluation_cohorts"][
        f"{task}/seed_{evaluation_seed}"
    ]["path"]
    cohort = _read_json(cohort_path)
    episodes = sorted(result["episodes"], key=lambda item: int(item["slot"]))
    expected_starts = [
        (entry["episode_id"], int(entry["start_step"]))
        for entry in cohort["entries"]
    ]
    actual_starts = [
        (episode["dataset_episode"], int(episode["start_step"]))
        for episode in episodes
    ]
    if _json_key(expected_starts) != _json_key(actual_starts):
        raise RuntimeError(
            f"evaluation starts differ from frozen cohort: "
            f"{training_seed}/{task}/{evaluation_seed}/{condition['condition_id']}"
        )
    rebuilt_rate = float(np.mean([bool(episode["success"]) for episode in episodes]))
    if not math.isclose(rebuilt_rate, float(result["success_rate"]), abs_tol=1e-12):
        raise RuntimeError("Table 2 success rate cannot be rebuilt from its episode records")
    return {
        "identity": identity,
        "identity_sha256": identity_sha,
        "attempt": attempt,
        "result": result,
        "episodes": episodes,
        "success_rate": rebuilt_rate,
        "trace_sha256": sha256_file(trace_path),
    }


def _summary_endpoints() -> list[dict[str, str]]:
    endpoints = []
    for arm in ("a_only", "shared_recorded_control", "shared_pred_detach", "shared_pred_full"):
        endpoints.append({"arm": arm, "endpoint": "own_a_p0", "condition_id": f"own_a_p0__{arm}"})
    for arm in ("shared_recorded_control", "shared_pred_detach", "shared_pred_full"):
        endpoints.append({"arm": arm, "endpoint": "own_a_p3", "condition_id": f"own_a_p3__{arm}"})
        endpoints.append({"arm": arm, "endpoint": "own_a_p0_po2", "condition_id": f"own_a_p0_po2__{arm}"})
    for arm in ("b_only_clean", "shared_recorded_control", "shared_pred_detach", "shared_pred_full"):
        endpoints.append({"arm": arm, "endpoint": "fixed_actor_p3", "condition_id": f"fixed_actor_p3__{arm}"})
    return endpoints


def _hierarchical_cluster_bootstrap(cluster_values, *, seed: int, draws: int = 10000):
    import numpy as np

    training_seeds = sorted(cluster_values)
    point = np.mean(
        [
            np.mean([value for values in cluster_values[training_seed].values() for value in values])
            for training_seed in training_seeds
        ]
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected_training_seeds = rng.choice(training_seeds, size=len(training_seeds), replace=True)
        seed_means = []
        for training_seed in selected_training_seeds:
            clusters = cluster_values[int(training_seed)]
            names = list(clusters)
            selected_clusters = rng.choice(names, size=len(names), replace=True)
            sampled = [value for name in selected_clusters for value in clusters[str(name)]]
            seed_means.append(float(np.mean(sampled)))
        samples[draw] = float(np.mean(seed_means))
    low, high = np.percentile(samples, [2.5, 97.5])
    two_sided_p = min(1.0, 2.0 * min(float(np.mean(samples <= 0)), float(np.mean(samples >= 0))))
    return float(point), float(low), float(high), float(two_sided_p)


def _holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [1.0] * len(p_values)
    previous = 0.0
    for rank, (index, p_value) in enumerate(sorted(enumerate(p_values), key=lambda item: item[1])):
        current = min(1.0, (len(p_values) - rank) * float(p_value))
        previous = max(previous, current)
        adjusted[index] = previous
    return adjusted


def _training_resource_rows() -> list[dict[str, Any]]:
    rows = []
    for run in matrix_rows():
        run_dir = _latest_training_attempt(run["task"], run["arm"], int(run["seed"]))
        identity = _read_json(run_dir / "table2_training_identity.json", {})
        progress = _read_json(run_dir / "table2_training_progress.json", {})
        invocation = _read_json(run_dir / "training_invocation.json", {})
        parameter_counts = identity.get("model_parameters", {})
        epochs = progress.get("epochs", [])
        updates = sum(int(item.get("optimizer_steps_global_step_delta", 0)) for item in epochs)
        if not updates:
            updates = int(progress.get("global_step", 0))
        training_seconds = sum(float(item.get("wall_seconds", 0.0)) for item in epochs)
        allocated = [item.get("peak_cuda_allocated_bytes") for item in epochs if item.get("peak_cuda_allocated_bytes") is not None]
        reserved = [item.get("peak_cuda_reserved_bytes") for item in epochs if item.get("peak_cuda_reserved_bytes") is not None]
        rows.append(
            {
                "run_id": run["run_id"],
                "task": run["task"],
                "arm": run["arm"],
                "training_seed": int(run["seed"]),
                "status": progress.get("status", invocation.get("status", "missing")),
                "total_parameters_including_encoder": parameter_counts.get("total_including_encoder", "NA"),
                "total_parameters_excluding_encoder": parameter_counts.get("total_excluding_encoder", "NA"),
                "trainable_parameters_including_encoder": parameter_counts.get("trainable_including_encoder", "NA"),
                "trainable_parameters_excluding_encoder": parameter_counts.get("trainable_excluding_encoder", "NA"),
                "optimizer_updates": updates,
                "expected_optimizer_updates": identity.get("optimization", {}).get("expected_total_updates", "NA"),
                "training_epoch_wall_seconds": training_seconds,
                "wall_ms_per_update": "NA" if not updates else training_seconds * 1000.0 / updates,
                "gpu_hours_wallclock": float(progress.get("gpu_hours_wallclock", 0.0)),
                "peak_cuda_allocated_gib": "NA" if not allocated else max(allocated) / (1024**3),
                "peak_cuda_reserved_gib": "NA" if not reserved else max(reserved) / (1024**3),
                "run_wall_seconds": invocation.get("wall_seconds", "NA"),
                "initial_model_sha256": identity.get("initial_model_sha256", "NA"),
                "initial_shared_representation_sha256": identity.get("initial_shared_representation_sha256", "NA"),
                "train_indices_sha256": identity.get("split", {}).get("train_indices_sha256", "NA"),
                "normalizer_sha256": identity.get("normalizer", {}).get("sha256", "NA"),
                "sample_order_sha256_by_epoch": _json_key(identity.get("split", {}).get("sample_order_sha256_by_epoch", {})),
            }
        )
    return rows


def _markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(str(row[index])) for row in rows) for index in range(len(rows[0]))]
    lines = [
        "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |"
        for row in rows
    ]
    lines.insert(1, "| " + " | ".join("-" * width for width in widths) + " |")
    return "\n".join(lines)


def command_summarize(args) -> None:
    import numpy as np

    frozen = _load_frozen()
    conditions = _evaluation_conditions()
    evaluation_seeds = _load_static_config()["evaluation_seeds"]
    result_index = {}
    cell_rows = []
    episode_rows = []
    missing = []
    for training_seed in SEEDS:
        for task in TASKS:
            for evaluation_seed in evaluation_seeds:
                for condition in conditions:
                    value = _load_complete_eval_cell(
                        task=task,
                        training_seed=int(training_seed),
                        evaluation_seed=int(evaluation_seed),
                        condition=condition,
                        frozen=frozen,
                    )
                    key = (int(training_seed), task, int(evaluation_seed), condition["condition_id"])
                    if value is None:
                        missing.append(key)
                        cell_rows.append(
                            {
                                "training_seed": training_seed,
                                "task": task,
                                "evaluation_seed": evaluation_seed,
                                "condition_id": condition["condition_id"],
                                "panel": condition["panel"],
                                "arm": condition["scorer_arm"] or condition["actor_arm"],
                                "status": "missing_or_incomplete",
                                "success_rate": "NA",
                                "episode_count": 0,
                            }
                        )
                        continue
                    result_index[key] = value
                    cell_rows.append(
                        {
                            "training_seed": training_seed,
                            "task": task,
                            "evaluation_seed": evaluation_seed,
                            "condition_id": condition["condition_id"],
                            "panel": condition["panel"],
                            "arm": condition["scorer_arm"] or condition["actor_arm"],
                            "status": "complete",
                            "success_rate": value["success_rate"],
                            "episode_count": len(value["episodes"]),
                            "identity_sha256": value["identity_sha256"],
                            "trace_sha256": value["trace_sha256"],
                            "result_path": str((value["attempt"] / "result.json").relative_to(ROOT)),
                        }
                    )
                    for episode in value["episodes"]:
                        episode_rows.append(
                            {
                                "training_seed": training_seed,
                                "task": task,
                                "evaluation_seed": evaluation_seed,
                                "condition_id": condition["condition_id"],
                                "panel": condition["panel"],
                                "arm": condition["scorer_arm"] or condition["actor_arm"],
                                "slot": int(episode["slot"]),
                                "dataset_episode": episode["dataset_episode"],
                                "start_step": int(episode["start_step"]),
                                "success": bool(episode["success"]),
                                "result_path": str((value["attempt"] / "result.json").relative_to(ROOT)),
                            }
                        )
    _write_csv(OUTPUT_ROOT / "summary/all_independent_cells.csv", cell_rows)
    _write_csv(OUTPUT_ROOT / "summary/evaluation_episodes.csv", episode_rows)

    endpoints = _summary_endpoints()
    by_seed = []
    for endpoint in endpoints:
        for task in TASKS:
            for training_seed in SEEDS:
                rates = [
                    result_index[(training_seed, task, int(evaluation_seed), endpoint["condition_id"])]["success_rate"]
                    for evaluation_seed in evaluation_seeds
                    if (training_seed, task, int(evaluation_seed), endpoint["condition_id"]) in result_index
                ]
                if not rates:
                    continue
                by_seed.append(
                    {
                        "training_seed": training_seed,
                        "arm": endpoint["arm"],
                        "endpoint": endpoint["endpoint"],
                        "task": task,
                        "evaluation_seed_count": len(rates),
                        "episodes": len(rates) * 50,
                        "mean_success_rate": float(np.mean(rates)),
                        "evaluation_seed_sample_std": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
                    }
                )
    _write_csv(OUTPUT_ROOT / "summary/by_training_seed.csv", by_seed)

    aggregated = []
    for endpoint in endpoints:
        for task in (*TASKS, "four_task_macro"):
            seed_rates = []
            for training_seed in SEEDS:
                task_rows = [
                    row["mean_success_rate"]
                    for row in by_seed
                    if row["training_seed"] == training_seed
                    and row["arm"] == endpoint["arm"]
                    and row["endpoint"] == endpoint["endpoint"]
                    and (task == "four_task_macro" or row["task"] == task)
                ]
                if task == "four_task_macro":
                    if len(task_rows) == len(TASKS):
                        seed_rates.append(float(np.mean(task_rows)))
                elif task_rows:
                    seed_rates.append(float(task_rows[0]))
            aggregated.append(
                {
                    "arm": endpoint["arm"],
                    "endpoint": endpoint["endpoint"],
                    "task": task,
                    "training_seed_count": len(seed_rates),
                    "mean_success_rate": float(np.mean(seed_rates)) if seed_rates else "NA",
                    "training_seed_sample_std": float(np.std(seed_rates, ddof=1)) if len(seed_rates) > 1 else "NA",
                }
            )
    _write_csv(OUTPUT_ROOT / "summary/by_arm_endpoint.csv", aggregated)

    pair_specs = (
        ("Shared-Recorded-control - A-only", "own_a_p0__shared_recorded_control", "own_a_p0__a_only"),
        ("Shared-Pred-Detach - Shared-Recorded-control", "fixed_actor_p3__shared_pred_detach", "fixed_actor_p3__shared_recorded_control"),
        ("Shared-Pred-Full - Shared-Pred-Detach", "fixed_actor_p3__shared_pred_full", "fixed_actor_p3__shared_pred_detach"),
    )
    paired = []
    for pair_index, (contrast, left_id, right_id) in enumerate(pair_specs):
        for task_index, task in enumerate(TASKS):
            seed_clusters = {}
            paired_occurrences = 0
            for training_seed in SEEDS:
                clusters = {}
                for evaluation_seed in evaluation_seeds:
                    left = result_index.get((training_seed, task, int(evaluation_seed), left_id))
                    right = result_index.get((training_seed, task, int(evaluation_seed), right_id))
                    if left is None or right is None:
                        continue
                    for le, re in zip(left["episodes"], right["episodes"]):
                        left_key = (le["slot"], _json_key(le["dataset_episode"]), le["start_step"])
                        right_key = (re["slot"], _json_key(re["dataset_episode"]), re["start_step"])
                        if left_key != right_key:
                            raise RuntimeError(f"paired evaluation cohort mismatch for {contrast}/{task}")
                        cluster = _json_key(le["dataset_episode"])
                        delta = float(bool(le["success"])) - float(bool(re["success"]))
                        clusters.setdefault(cluster, []).append(delta)
                        paired_occurrences += 1
                if clusters:
                    seed_clusters[training_seed] = clusters
            if len(seed_clusters) == len(SEEDS):
                estimate, low, high, p_value = _hierarchical_cluster_bootstrap(
                    seed_clusters,
                    seed=20261005 + pair_index * 100 + task_index,
                )
            else:
                estimate = low = high = p_value = float("nan")
            paired.append(
                {
                    "contrast": contrast,
                    "task": task,
                    "training_seed_count": len(seed_clusters),
                    "paired_episode_occurrences": paired_occurrences,
                    "mean_difference": estimate,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "bootstrap_p_two_sided": p_value,
                    "holm_adjusted_p": "pending",
                }
            )
    finite = [float(row["bootstrap_p_two_sided"]) for row in paired if math.isfinite(float(row["bootstrap_p_two_sided"]))]
    if len(finite) == 12:
        for row, value in zip(paired, _holm_adjust(finite)):
            row["holm_adjusted_p"] = value
    _write_csv(OUTPUT_ROOT / "summary/primary_paired_differences.csv", paired)

    resources = _training_resource_rows()
    _write_csv(OUTPUT_ROOT / "summary/training_resources.csv", resources)
    fixed_pool_rows = []
    fixed_pool_missing = []
    fixed_pool_task_status = {}
    fixed_pool_aggregated = []
    for task in TASKS:
        task_status = _read_json(FIXED_POOL_ROOT / task / "group_status.json", {})
        fixed_pool_task_status[task] = task_status
        if (
            task_status.get("status") != "complete"
            or task_status.get("training_seeds") != list(SEEDS)
            or int(task_status.get("scorer_groups_complete", 0)) != len(SEEDS) * 4
        ):
            fixed_pool_missing.append(f"{task}/group_status")
        anchor = _read_json(FIXED_POOL_ROOT / task / "fixed_actor/summary.json")
        if anchor is None:
            fixed_pool_missing.append(f"{task}/table1_fixed_actor")
        else:
            metrics = anchor.get("metrics", {})
            fixed_pool_rows.append(
                {
                    "task": task,
                    "scorer": "table1_fixed_actor",
                    "training_seed": "NA",
                    "status": anchor.get("status", "missing"),
                    "state_count": anchor.get("pool_state_count", 0),
                    "candidate_count": anchor.get("candidate_count", 64),
                    **metrics,
                    "po2_mean_success_delta": "NA",
                    "po2_mean_physical_improvement_at_25": "NA",
                    "po2_paired_valid_state_count": "NA",
                }
            )
        for arm in ("b_only_clean", "shared_recorded_control", "shared_pred_detach", "shared_pred_full"):
            seed_summaries = []
            for seed in SEEDS:
                path = FIXED_POOL_ROOT / task / "scorers" / f"training_seed_{seed}" / arm / "summary.json"
                value = _read_json(path)
                if value is None or value.get("status") != "complete":
                    fixed_pool_missing.append(f"{task}/{seed}/{arm}")
                    continue
                seed_summaries.append(value)
                rank = value.get("candidate_pool", {})
                po2 = value.get("po2", {})
                fixed_pool_rows.append(
                    {
                        "task": task,
                        "scorer": arm,
                        "training_seed": int(seed),
                        "status": value.get("status"),
                        "state_count": value.get("state_count", 0),
                        "candidate_count": value.get("candidate_count", 64),
                        **rank,
                        "po2_mean_success_delta": po2.get("mean_success_delta", "NA"),
                        "po2_mean_physical_improvement_at_25": po2.get("mean_physical_improvement_at_25", "NA"),
                        "po2_paired_valid_state_count": po2.get("paired_valid_at_25_count", 0),
                        "po2_predicted_improved_true_worse_count": po2.get("predicted_improved_true_worse_count", "NA"),
                        "po2_mean_predicted_improvement": po2.get("mean_predicted_improvement", "NA"),
                    }
                )
            for scorer, values in ((arm, seed_summaries),):
                for metric in (
                    "mean_selected_success_by_25",
                    "mean_pool_success_fraction_25",
                    "mean_pool_has_success_by_25",
                    "mean_complete_pool_regret_at_25",
                    "mean_valid_subpool_regret_at_25",
                    "mean_valid_subpool_spearman",
                ):
                    samples = [value["candidate_pool"].get(metric) for value in values]
                    samples = [float(item) for item in samples if item is not None]
                    fixed_pool_aggregated.append(
                        {
                            "task": task,
                            "scorer": scorer,
                            "metric": metric,
                            "training_seed_count": len(samples),
                            "mean": float(np.mean(samples)) if samples else "NA",
                            "training_seed_sample_std": float(np.std(samples, ddof=1)) if len(samples) > 1 else "NA",
                        }
                    )
                for metric in (
                    "mean_success_delta",
                    "mean_physical_improvement_at_25",
                    "mean_predicted_improvement",
                ):
                    samples = [value["po2"].get(metric) for value in values]
                    samples = [float(item) for item in samples if item is not None]
                    fixed_pool_aggregated.append(
                        {
                            "task": task,
                            "scorer": scorer,
                            "metric": f"po2_{metric}",
                            "training_seed_count": len(samples),
                            "mean": float(np.mean(samples)) if samples else "NA",
                            "training_seed_sample_std": float(np.std(samples, ddof=1)) if len(samples) > 1 else "NA",
                        }
                    )
    _write_csv(OUTPUT_ROOT / "summary/fixed_pool_by_checkpoint.csv", fixed_pool_rows)
    _write_csv(OUTPUT_ROOT / "summary/fixed_pool_by_task_arm.csv", fixed_pool_aggregated)
    missing_count = len(missing)
    if missing_count and not args.allow_incomplete:
        raise RuntimeError(f"Table 2 evaluation matrix is incomplete: {missing_count}/1008 cells")
    if fixed_pool_missing and not args.allow_incomplete:
        raise RuntimeError(f"Table 2 fixed-pool diagnostics are incomplete: {len(fixed_pool_missing)} items")
    complete = (
        missing_count == 0
        and not fixed_pool_missing
        and all(row["status"] == "complete" for row in resources)
    )

    def endpoint_value(arm, endpoint, task):
        row = next(
            item for item in aggregated
            if item["arm"] == arm and item["endpoint"] == endpoint and item["task"] == task
        )
        if row["training_seed_count"] != 3:
            return "NA"
        return (
            f"{float(row['mean_success_rate']) * 100:.2f} ± "
            f"{float(row['training_seed_sample_std']) * 100:.2f}"
        )

    def pool_value(task, scorer, metric, *, percent=False):
        if scorer == "table1_fixed_actor":
            summary = _read_json(FIXED_POOL_ROOT / task / "fixed_actor/summary.json", {})
            value = summary.get("metrics", {}).get(metric)
            return "NA" if value is None else (f"{float(value) * 100:.2f}" if percent else f"{float(value):.4f}")
        row = next(
            (item for item in fixed_pool_aggregated if item["task"] == task and item["scorer"] == scorer and item["metric"] == metric),
            None,
        )
        if row is None or row["training_seed_count"] != 3:
            return "NA"
        scale = 100.0 if percent else 1.0
        return (
            f"{float(row['mean']) * scale:.2f} ± "
            f"{float(row['training_seed_sample_std']) * scale:.2f}"
        )

    def po2_value(task, scorer, metric, *, percent=False):
        row = next(
            (item for item in fixed_pool_aggregated if item["task"] == task and item["scorer"] == scorer and item["metric"] == f"po2_{metric}"),
            None,
        )
        if row is None or row["training_seed_count"] != 3:
            return "NA"
        scale = 100.0 if percent else 1.0
        return (
            f"{float(row['mean']) * scale:.2f} ± "
            f"{float(row['training_seed_sample_std']) * scale:.2f}"
        )

    own_table = [["Task", "A-only P0", "Recorded P0", "Detach P0", "Full P0", "Recorded P3", "Detach P3", "Full P3", "Recorded P0-PO2", "Detach P0-PO2", "Full P0-PO2"]]
    for task in TASKS:
        own_table.append(
            [task]
            + [endpoint_value(arm, endpoint, task) for arm, endpoint in (
                ("a_only", "own_a_p0"),
                ("shared_recorded_control", "own_a_p0"),
                ("shared_pred_detach", "own_a_p0"),
                ("shared_pred_full", "own_a_p0"),
                ("shared_recorded_control", "own_a_p3"),
                ("shared_pred_detach", "own_a_p3"),
                ("shared_pred_full", "own_a_p3"),
                ("shared_recorded_control", "own_a_p0_po2"),
                ("shared_pred_detach", "own_a_p0_po2"),
                ("shared_pred_full", "own_a_p0_po2"),
            )]
        )
    if complete:
        own_table.append(
            ["Four-task macro"]
            + [endpoint_value(arm, endpoint, "four_task_macro") for arm, endpoint in (
                ("a_only", "own_a_p0"),
                ("shared_recorded_control", "own_a_p0"),
                ("shared_pred_detach", "own_a_p0"),
                ("shared_pred_full", "own_a_p0"),
                ("shared_recorded_control", "own_a_p3"),
                ("shared_pred_detach", "own_a_p3"),
                ("shared_pred_full", "own_a_p3"),
                ("shared_recorded_control", "own_a_p0_po2"),
                ("shared_pred_detach", "own_a_p0_po2"),
                ("shared_pred_full", "own_a_p0_po2"),
            )]
        )
    fixed_table = [["Task", "B-only clean", "Recorded-control B", "Pred-Detach B", "Pred-Full B"]]
    for task in TASKS:
        fixed_table.append(
            [task]
            + [endpoint_value(arm, "fixed_actor_p3", task) for arm in (
                "b_only_clean", "shared_recorded_control", "shared_pred_detach", "shared_pred_full"
            )]
        )
    if complete:
        fixed_table.append(
            ["Four-task macro"]
            + [endpoint_value(arm, "fixed_actor_p3", "four_task_macro") for arm in (
                "b_only_clean", "shared_recorded_control", "shared_pred_detach", "shared_pred_full"
            )]
        )
    pool_table = [[
        "Task", "Scorer", "Selected success@25 (%)", "Pool coverage@25 (%)",
        "Complete-pool regret@25", "Complete-pool states / 300",
    ]]
    for task in TASKS:
        scorers = ("table1_fixed_actor", "b_only_clean", "shared_recorded_control", "shared_pred_detach", "shared_pred_full")
        for scorer in scorers:
            complete_states = "NA"
            if scorer == "table1_fixed_actor":
                anchor = _read_json(FIXED_POOL_ROOT / task / "fixed_actor/summary.json", {})
                complete_states = anchor.get("metrics", {}).get("complete_pool_state_count", "NA")
            else:
                values = [
                    _read_json(FIXED_POOL_ROOT / task / "scorers" / f"training_seed_{seed}" / scorer / "summary.json", {})
                    for seed in SEEDS
                ]
                counts = [int(value.get("candidate_pool", {}).get("complete_pool_state_count", 0)) for value in values if value.get("status") == "complete"]
                complete_states = f"{float(np.mean(counts)):.1f} ± {float(np.std(counts, ddof=1)):.1f}" if len(counts) == 3 else "NA"
            pool_table.append(
                [
                    task,
                    scorer,
                    pool_value(task, scorer, "mean_selected_success_by_25", percent=True),
                    pool_value(task, scorer, "mean_pool_has_success_by_25", percent=True),
                    pool_value(task, scorer, "mean_complete_pool_regret_at_25"),
                    str(complete_states),
                ]
            )
    po2_table = [[
        "Task", "B scorer", "Success change@25 (pp)", "True physical-cost improvement@25",
        "Predicted-cost improvement", "PO2 paired-valid states / 300",
    ]]
    for task in TASKS:
        for scorer in ("b_only_clean", "shared_recorded_control", "shared_pred_detach", "shared_pred_full"):
            valid_counts = [
                int((_read_json(FIXED_POOL_ROOT / task / "scorers" / f"training_seed_{seed}" / scorer / "summary.json", {}).get("po2", {})).get("paired_valid_at_25_count", 0))
                for seed in SEEDS
            ]
            valid_value = f"{float(np.mean(valid_counts)):.1f} ± {float(np.std(valid_counts, ddof=1)):.1f}" if len(valid_counts) == 3 else "NA"
            po2_table.append(
                [
                    task,
                    scorer,
                    po2_value(task, scorer, "mean_success_delta", percent=True),
                    po2_value(task, scorer, "mean_physical_improvement_at_25"),
                    po2_value(task, scorer, "mean_predicted_improvement"),
                    valid_value,
                ]
            )
    paired_table = [["Contrast", "Task", "Difference (pp)", "95% CI (pp)", "bootstrap p", "Holm p"]]
    for row in paired:
        if math.isfinite(float(row["mean_difference"])):
            paired_table.append(
                [
                    row["contrast"], row["task"], f"{float(row['mean_difference']) * 100:.2f}",
                    f"[{float(row['bootstrap_ci_low']) * 100:.2f}, {float(row['bootstrap_ci_high']) * 100:.2f}]",
                    f"{float(row['bootstrap_p_two_sided']):.4f}",
                    "pending" if row["holm_adjusted_p"] == "pending" else f"{float(row['holm_adjusted_p']):.4f}",
                ]
            )
        else:
            paired_table.append([row["contrast"], row["task"], "NA", "NA", "NA", "NA"])
    report = [
        "# CVPR Table 2：训练耦合消融结果",
        "",
        f"执行状态：{'完成' if complete else '未完成'}；闭环评测 {1008 - missing_count}/1008 个独立单元。",
        f"固定池诊断状态：{'完成' if not fixed_pool_missing else f'未完成（{len(fixed_pool_missing)} 项待完成）'}；每任务 300 个状态、64 候选，原生真值最多 76,800 条，PO2 最多 14,400 条。",
        "",
        "训练矩阵为五臂 × 四任务 × 三个训练 seeds；各训练 seed 内先对六个评测 seeds 求均值，再以三个独立训练重复报告 mean ± sample std。评测 seed 不计作训练重复。所有臂使用同一 task/seed 的初始化、数据 split、normalizer 和逐 epoch 样本顺序。",
        "",
        "Table 1 R4-AB 权重只作为 fixed actor 诊断输入，不用于核心臂初始化。Table 1 的样本已用于开发观察，当前结果属于同 benchmark 扩展。",
        "",
        "## Own-A 闭环结果（%）",
        "",
        _markdown_table(own_table),
        "",
        "## 固定 R4-AB actor + 新训练 B 的闭环 P3（%）",
        "",
        _markdown_table(fixed_table),
        "",
        "## 固定候选池评分与真实 25 步后果",
        "",
        "同一 Table 1 R4-AB actor 候选池只生成一次并由 Table 1 固定 B 与各 Table 2 B checkpoint 离线评分。没有在规划器中裁剪动作，分支保留环境原生动作处理。成功率采用冻结原生环境事件；25 步物理 regret 的主口径仅使用 64 条分支都获得合法 25 步状态的样本，完整池覆盖不足时另报有效子池但不冒充完整池 regret。PushT cost 为 max(position L2 / 20 px, circular angle error / (π/9)); 其它任务按对应物理状态 L2 和任务单位报告。Table 2 B 的数字为三个训练 seeds 的 mean ± sample std。",
        "",
        _markdown_table(pool_table),
        "",
        "## 固定初始动作 PO2 的真实修正结果",
        "",
        "每个 B scorer 对同一噪声池的 candidate index 0 做两步 post-optimization；原始候选分支复用上表共享真值，修正动作独立从同一冻结起点执行。物理改善只在原动作与修正动作都有合法 25 步状态时计算。",
        "",
        _markdown_table(po2_table),
        "",
        "## 预设配对（百分点）",
        "",
        "95% 区间按训练 seed 外层和来源 episode 内层成簇重采样；重复来源 episode 共同抽样。四任务 × 三个主要配对为一个 Holm 校正族。三次训练重复的统计精度有限。",
        "",
        _markdown_table(paired_table),
        "",
        "## 训练成本与资源",
        "",
        "每个 run 的总/活跃参数（含/不含 encoder）、优化步数、GPU-hours、每更新耗时和峰值显存见 `outputs/cvpr/table2/v1/summary/training_resources.csv`。本实验没有双 DiT 对照，因此不将参数量解释为共享带来的资源节省。",
        "",
        "## 复现资产",
        "",
        "- 冻结配置、数据/权重/样本/代码身份：`outputs/cvpr/table2/v1/frozen_config.json` 与 `provenance/`。",
        "- 每训练 run 的共同初始化、split、normalizer、样本顺序：`runs/{task}/seed_{seed}/{arm}/table2_training_identity.json`。",
        "- 逐回合数据与身份：`evals/`；重建 CSV：`summary/all_independent_cells.csv` 和 `summary/evaluation_episodes.csv`。",
        "- 五臂 CPU 梯度验收：`diagnostics/cpu_gradient_acceptance.json`。",
        "",
    ]
    report_path = ROOT / _load_static_config()["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    write_json_atomic(
        OUTPUT_ROOT / "summary/summary_status.json",
        {
            "schema_version": 1,
            "status": "complete" if complete else "incomplete",
            "expected_evaluation_cells": 1008,
            "complete_evaluation_cells": 1008 - missing_count,
            "missing_evaluation_cells": missing_count,
            "fixed_pool_diagnostics_complete": not fixed_pool_missing,
            "fixed_pool_missing_count": len(fixed_pool_missing),
            "fixed_pool_task_status": fixed_pool_task_status,
            "training_runs_complete": sum(row["status"] == "complete" for row in resources),
            "training_runs_expected": len(resources),
            "report": str(report_path.relative_to(ROOT)),
        },
    )
    print(json.dumps({"status": "complete" if complete else "incomplete", "evaluations": 1008 - missing_count, "report": str(report_path)}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="audit and freeze inputs, code, and matrix")
    audit.set_defaults(func=command_audit)
    amend = subparsers.add_parser(
        "amend-code-snapshot",
        help="version a post-freeze code correction while preserving the prior snapshot",
    )
    amend.add_argument("--reason", required=True)
    amend.set_defaults(func=command_amend_code_snapshot)
    train = subparsers.add_parser("train", help="run the frozen training matrix by seed wave")
    train.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    train.add_argument("--tasks", nargs="+", choices=TASKS)
    train.add_argument("--arms", nargs="+", choices=tuple(ARMS))
    train.add_argument("--gpus", default="0,1,3")
    train.set_defaults(func=command_train)
    diagnose = subparsers.add_parser("diagnose", help="run the CPU five-arm gradient acceptance")
    diagnose.add_argument("--tasks", nargs="+", choices=TASKS)
    diagnose.add_argument("--seed", type=int, default=3072)
    diagnose.add_argument("--batch-size", type=int, default=32)
    diagnose.set_defaults(func=command_diagnose)
    evaluate = subparsers.add_parser("eval", help="run the frozen Table 2 closed-loop panel")
    evaluate.add_argument("--training-seeds", nargs="+", type=int, default=list(SEEDS))
    evaluate.add_argument("--tasks", nargs="+", choices=TASKS)
    evaluate.add_argument("--gpus", default="0,1,3")
    evaluate.set_defaults(func=command_eval)
    eval_group = subparsers.add_parser("eval-group", help=argparse.SUPPRESS)
    eval_group.add_argument("--task", choices=TASKS, required=True)
    eval_group.add_argument("--training-seed", type=int, required=True)
    eval_group.set_defaults(func=command_eval_group)
    fixed_pool = subparsers.add_parser(
        "fixed-pool", help="capture the Table 1 pool, native branch truths, and all Table 2 B/PO2 diagnostics"
    )
    fixed_pool.add_argument("--training-seeds", nargs="+", type=int, default=list(SEEDS))
    fixed_pool.add_argument("--tasks", nargs="+", choices=TASKS)
    fixed_pool.add_argument("--gpus", default="0,1,3")
    fixed_pool.set_defaults(func=command_fixed_pool)
    fixed_pool_group = subparsers.add_parser("fixed-pool-group", help=argparse.SUPPRESS)
    fixed_pool_group.add_argument("--task", choices=TASKS, required=True)
    fixed_pool_group.add_argument("--training-seeds", nargs="+", type=int, required=True)
    fixed_pool_group.set_defaults(func=command_fixed_pool_group)
    summarize = subparsers.add_parser("summarize", help="rebuild paired summaries and report")
    summarize.add_argument("--allow-incomplete", action="store_true")
    summarize.set_defaults(func=command_summarize)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.func(arguments)
