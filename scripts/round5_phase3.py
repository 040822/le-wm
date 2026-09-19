#!/usr/bin/env python3
"""Round 5 Phase 3 preparation, evaluation, and analysis driver."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.checkpoint import load_policy_or_model
from source.common.eval import DatasetEvaluationSession, EvaluationIdentity, compose_eval_config
from source.common.phase3_compat import (
    PHASE3_DATASETS,
    PHASE3_TASKS,
    build_phase3_legacy_manifest,
    patch_phase3_environments,
    scene_target_group,
)
from source.common.round3_phase1 import CohortManifest, wilson_interval
from source.common.round4_eval import run_round4_evaluation


OUTPUT_ROOT = ROOT / "outputs" / "round5" / "phase3"
REPORT_PATH = ROOT / "docs" / "report" / "round5" / "round5_phase3_report.md"
SEED = 3072
EVAL_SEED = 42
NUM_EVAL = 50
GOAL_OFFSET = 25
EVAL_BUDGET = 50
MIN_FREE_MIB = 5000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_keys(task: str) -> list[str]:
    common = ["pixels", "action", "qpos", "qvel"]
    if task == "scene":
        return common + [
            "button_states",
            "privileged_target_task",
            "privileged_target_block_pos",
            "privileged_target_button",
            "privileged_target_button_state",
            "privileged_target_drawer_pos",
            "privileged_target_window_pos",
        ]
    if task == "finger":
        return common + ["target_position"]
    if task == "humanoid":
        return common
    raise ValueError(task)


def _load_dataset(task: str, *, cohort_only: bool = False):
    from stable_worldmodel.data.formats.hdf5 import HDF5Dataset

    keys = ["ep_idx", "step_idx"] if cohort_only else _dataset_keys(task)
    return HDF5Dataset(path=PHASE3_DATASETS[task], keys_to_load=keys)


def _manifest_path(task: str, output_root: Path = OUTPUT_ROOT) -> Path:
    return output_root / "cohorts" / task / "legacy_50.json"


def _ensure_manifest(task: str, output_root: Path = OUTPUT_ROOT) -> CohortManifest:
    path = _manifest_path(task, output_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing frozen cohort {path}; run `round5_phase3.py prepare` first"
        )
    manifest = CohortManifest.load(path)
    if manifest.task != task or manifest.protocol_variant != "legacy" or len(manifest.entries) != 50:
        raise ValueError(f"invalid Phase 3 cohort: {path}")
    return manifest


def _compose(task: str):
    return compose_eval_config(
        task,
        overrides=[
            f"eval.num_eval={NUM_EVAL}",
            f"eval.goal_offset_steps={GOAL_OFFSET}",
            f"eval.eval_budget={EVAL_BUDGET}",
            "output.save_video=false",
            "solver.num_samples=300",
            "solver.topk=30",
            "solver.n_steps=30",
            "solver.var_scale=1.0",
        ],
    )


def _gpu(value: str | None) -> str | None:
    if value is None:
        return None
    if not value.isdigit() or int(value) not in range(8):
        raise argparse.ArgumentTypeError("--gpu must select one physical GPU0-7")
    return value


def _gpu_preflight(gpu: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            str(gpu),
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [item.strip() for item in completed.stdout.splitlines()[0].split(",")]
    if len(fields) < 6:
        raise RuntimeError(f"invalid nvidia-smi output for GPU{gpu}: {completed.stdout!r}")
    snapshot = {
        "gpu": int(gpu),
        "name": fields[1],
        "memory_total_mib": int(fields[2]),
        "memory_used_mib": int(fields[3]),
        "memory_free_mib": int(fields[4]),
        "utilization_percent": int(fields[5]),
    }
    if snapshot["memory_free_mib"] < MIN_FREE_MIB:
        raise RuntimeError(
            f"GPU{gpu} has {snapshot['memory_free_mib']} MiB free; "
            f"Phase 3 requires at least {MIN_FREE_MIB} MiB"
        )
    print(json.dumps({"gpu_preflight": snapshot}, sort_keys=True), flush=True)
    return snapshot


def _configure_gpu(gpu: str | None, device: str) -> None:
    if str(device).startswith("cuda"):
        if gpu is None:
            raise ValueError("CUDA evaluation requires --gpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        _gpu_preflight(gpu)
        # configure_mujoco_egl_device is called by the shared evaluator after
        # this process has selected exactly one physical GPU.


def _condition_name(spec: Mapping[str, Any]) -> str:
    step = "invariant" if spec["step"] is None else f"step_{int(spec['step'])}"
    return "/".join(
        (
            str(spec["mode"]),
            str(spec["protocol"]),
            str(spec["guidance"]),
            step,
            "euler" if spec["step"] is not None else "not_applicable",
        )
    )


def p2_protocol(task: str) -> str:
    return "legacy" if task == "scene" else "cem-clip"


def fast_conditions(task: str) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    steps = (1, 2, 5, 10, 16, 32)
    protocol = p2_protocol(task)
    for step in steps:
        conditions.append({"task": task, "mode": "P0", "protocol": "not_applicable", "step": step, "guidance": "none"})
    conditions.append({"task": task, "mode": "P1", "protocol": protocol, "step": None, "guidance": "none"})
    for step in steps:
        conditions.append({"task": task, "mode": "P2", "protocol": protocol, "step": step, "guidance": "none"})
    for step in steps:
        conditions.append({"task": task, "mode": "P3", "protocol": "not_applicable", "step": step, "guidance": "none"})
    for step in steps:
        for guidance in ("guided_flow", "post_opt"):
            conditions.append({"task": task, "mode": "P0", "protocol": "not_applicable", "step": step, "guidance": guidance})
            conditions.append({"task": task, "mode": "P2", "protocol": protocol, "step": step, "guidance": guidance})
        for guidance in ("guided_flow", "post_opt", "post_opt_refine"):
            conditions.append({"task": task, "mode": "P3", "protocol": "not_applicable", "step": step, "guidance": guidance})
    return conditions


def _condition_dir(output_root: Path, task: str, spec: Mapping[str, Any]) -> Path:
    return output_root / "fastlewam" / task / _condition_name(spec)


def _result_complete(path: Path) -> bool:
    result = path / "result.json"
    if not result.is_file():
        return False
    try:
        payload = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "ok" and len(payload.get("episodes", [])) == NUM_EVAL


def command_prepare(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    for task in PHASE3_TASKS:
        dataset = _load_dataset(task, cohort_only=True)
        manifest = build_phase3_legacy_manifest(
            dataset,
            task=task,
            seed=EVAL_SEED,
            goal_offset_steps=GOAL_OFFSET,
            num_eval=NUM_EVAL,
        )
        path = _manifest_path(task, output_root)
        manifest.save(path)
        print(json.dumps({"task": task, "path": str(path), "cohort_sha256": manifest.computed_sha256}, sort_keys=True))


def command_smoke(args: argparse.Namespace) -> None:
    import gymnasium as gym

    patch_phase3_environments()
    _configure_gpu(args.gpu, args.device)
    env_names = {
        "scene": "swm/OGBScene-v0",
        "finger": "swm/FingerDMControl-v0",
        "humanoid": "swm/HumanoidDMControl-v0",
    }
    for task in PHASE3_TASKS:
        if task == "scene":
            env = gym.make(env_names[task], max_episode_steps=100, ob_type="states", terminate_at_goal=True, visualize_info=False)
        elif task == "finger":
            env = gym.make(env_names[task], max_episode_steps=100, task="turn_hard")
        else:
            env = gym.make(env_names[task], max_episode_steps=100, task="walk")
        try:
            _, info = env.reset(seed=0)
            raw = env.unwrapped
            if task == "scene":
                qpos = np.asarray(raw._data.qpos).copy()
                qvel = np.asarray(raw._data.qvel).copy()
                raw.set_state(qpos, qvel, button_states=np.asarray([0, 0]))
                raw.set_phase3_scene_goal("drawer", np.asarray([0.45, 0.0, 0.02]), 0, 0, np.asarray([0.0]), np.asarray([0.0]))
            elif task == "finger":
                raw.set_target_position(np.asarray(raw.env.physics.target_position()).copy())
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            env.step(action)
            print(json.dumps({"task": task, "status": "ok", "info_keys": sorted(str(k) for k in info)[:8]}, sort_keys=True))
        finally:
            env.close()


def command_eval_fast(args: argparse.Namespace) -> None:
    task = args.task
    output_root = Path(args.output_root).resolve()
    manifest = _ensure_manifest(task, output_root)
    patch_phase3_environments()
    _configure_gpu(args.gpu, args.device)
    cfg = _compose(task)
    dataset = _load_dataset(task)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model, resolved = load_policy_or_model(str(checkpoint))
    conditions = fast_conditions(task)
    if args.condition_index is not None:
        conditions = [conditions[int(index)] for index in args.condition_index]
    for spec in conditions:
        target = _condition_dir(output_root, task, spec)
        if _result_complete(target):
            print(json.dumps({"status": "reused", "path": str(target / "result.json")}, sort_keys=True), flush=True)
            continue
        target.mkdir(parents=True, exist_ok=True)
        if any(target.iterdir()):
            raise FileExistsError(f"refusing to overwrite partial condition: {target}")
        mode = str(spec["mode"])
        result = run_round4_evaluation(
            cfg,
            task=task,
            policy_or_model=model,
            mode=mode,
            identity={
                "entrypoint": "round5_phase3",
                "policy_kind": "fast_lewam",
                "checkpoint": str(resolved or checkpoint),
                "epoch": 10,
                "stage": mode,
                "guidance_mode": str(spec["guidance"]),
            },
            manifest=manifest,
            output_dir=target,
            dataset=dataset,
            device=args.device,
            trace=False,
            candidate_count=64,
            flow_steps=16,
            action_flow_steps=spec["step"],
            action_flow_integrator="euler",
            cem_protocol=spec["protocol"],
            action_bound_mode=None,
            guidance_mode=str(spec["guidance"]),
            guidance_step_size=0.01,
            guidance_last_steps=5,
            guidance_inner_steps=5,
            guidance_max_rms_offset=0.20,
            proposal_chunk_size=512,
            allowed_protocol_variants=("legacy",),
        )
        payload = json.loads((target / "result.json").read_text(encoding="utf-8"))
        payload.update(
            {
                "phase3_method": "fast_lewam",
                "phase3_condition": dict(spec),
                "success_rate_percent": float(np.mean([bool(item["success"]) for item in payload["episodes"]]) * 100.0),
            }
        )
        (target / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "ok", "task": task, "condition": _condition_name(spec), "success_rate": result["success_rate"]}, sort_keys=True), flush=True)


def _baseline_dir(output_root: Path, method: str, task: str) -> Path:
    return output_root / "baselines" / method / task


def command_eval_baseline(args: argparse.Namespace) -> None:
    task = args.task
    method = args.method
    output_root = Path(args.output_root).resolve()
    manifest = _ensure_manifest(task, output_root)
    target = _baseline_dir(output_root, method, task)
    if _result_complete(target):
        print(json.dumps({"status": "reused", "path": str(target / "result.json")}, sort_keys=True), flush=True)
        return
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"refusing to overwrite partial baseline: {target}")
    patch_phase3_environments()
    _configure_gpu(args.gpu, args.device)
    cfg = _compose(task)
    dataset = _load_dataset(task)
    policy_or_model, resolved = load_policy_or_model(str(Path(args.checkpoint).resolve()))
    identity = EvaluationIdentity(
        entrypoint="round5_phase3",
        policy_kind=method,
        checkpoint=str(resolved or args.checkpoint),
        epoch=10,
        stage=None,
    )
    session = DatasetEvaluationSession(
        cfg,
        task=task,
        dataset=dataset,
        cohort=manifest.to_evaluation_cohort(),
    )
    result = session.evaluate(
        policy_or_model,
        identity=identity,
        output_dir=target,
        device=args.device,
    )
    result_path = target / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    successes = [bool(item["success"]) for item in payload["episodes"]]
    payload.update(
        {
            "phase3_method": method,
            "phase3_condition": {
                "mode": "standard",
                "cem_protocol": "legacy",
                "num_samples": 300 if method == "lewm" else 64,
                "flow_steps": None if method == "lewm" else 16,
            },
            "cohort_sha256": manifest.computed_sha256,
            "cohort_id": manifest.cohort_id,
            "success_rate_percent": float(np.mean(successes) * 100.0),
        }
    )
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "task": task, "method": method, "success_rate_percent": float(result.success_rate)}, sort_keys=True), flush=True)


def _successes(payload: Mapping[str, Any]) -> list[bool]:
    values = [bool(item.get("success", False)) for item in payload.get("episodes", [])]
    if len(values) != NUM_EVAL:
        raise ValueError("Phase 3 result must contain exactly 50 episodes")
    return values


def _mcnemar(left: Sequence[bool], right: Sequence[bool]) -> dict[str, Any]:
    improved = sum((not a) and b for a, b in zip(left, right))
    regressed = sum(a and (not b) for a, b in zip(left, right))
    discordant = improved + regressed
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1)) / (2.0 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"improved": int(improved), "regressed": int(regressed), "mcnemar_exact_two_sided_p": float(p_value)}


def _read_result(path: Path, *, method: str, task: str, condition: str) -> dict[str, Any]:
    payload = json.loads((path / "result.json").read_text(encoding="utf-8"))
    successes = _successes(payload)
    low, high = wilson_interval(successes)
    return {
        "task": task,
        "method": method,
        "condition": condition,
        "path": str(path / "result.json"),
        "cohort_sha256": payload.get("cohort_sha256") or payload.get("parameters", {}).get("cohort_sha256"),
        "successes": int(sum(successes)),
        "episodes": len(successes),
        "success_rate": float(np.mean(successes)),
        "success_rate_percent": float(np.mean(successes) * 100.0),
        "wilson_95_percent_low": float(low * 100.0),
        "wilson_95_percent_high": float(high * 100.0),
        "success_vector": successes,
        "payload": payload,
    }


def command_analyze(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    rows: list[dict[str, Any]] = []
    manifests = {task: _ensure_manifest(task, output_root) for task in PHASE3_TASKS}
    for task in PHASE3_TASKS:
        for spec in fast_conditions(task):
            target = _condition_dir(output_root, task, spec)
            if not _result_complete(target):
                raise FileNotFoundError(f"missing FastLeWAM result: {target / 'result.json'}")
            rows.append(_read_result(target, method="fast_lewam", task=task, condition=_condition_name(spec)))
        for method in ("lewm", "leflow"):
            target = _baseline_dir(output_root, method, task)
            if not _result_complete(target):
                raise FileNotFoundError(f"missing baseline result: {target / 'result.json'}")
            rows.append(_read_result(target, method=method, task=task, condition="standard"))

    expected = len(PHASE3_TASKS) * (61 + 2)
    if len(rows) != expected:
        raise ValueError(f"Phase 3 result count {len(rows)} != {expected}")
    expected_hashes = {task: manifests[task].computed_sha256 for task in PHASE3_TASKS}
    for row in rows:
        if row["cohort_sha256"] not in {expected_hashes[row["task"]], None}:
            raise ValueError(f"cohort hash mismatch in {row['path']}")

    primary: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for task in PHASE3_TASKS:
        fast = next(row for row in rows if row["task"] == task and row["method"] == "fast_lewam" and row["condition"].startswith("P3/not_applicable/none/step_1"))
        for method in ("lewm", "leflow"):
            baseline = next(row for row in rows if row["task"] == task and row["method"] == method)
            paired = _mcnemar(baseline["success_vector"], fast["success_vector"])
            comparisons.append({"task": task, "baseline": method, "treatment": "fast_lewam_p3_step1_none", "delta_pp": (fast["success_rate"] - baseline["success_rate"]) * 100.0, **paired})
        primary.extend([fast] + [row for row in rows if row["task"] == task and row["method"] in {"lewm", "leflow"}])

    scene_groups = []
    scene_manifest = manifests["scene"]
    scene_dataset = _load_dataset("scene")
    groups = scene_target_group(scene_dataset, scene_manifest)
    for row in rows:
        if row["task"] != "scene":
            continue
        grouped = {}
        for group in sorted(set(groups)):
            values = [value for value, label in zip(row["success_vector"], groups) if label == group]
            grouped[group] = {"episodes": len(values), "successes": int(sum(values)), "success_rate_percent": float(np.mean(values) * 100.0) if values else None}
        scene_groups.append({"method": row["method"], "condition": row["condition"], "groups": grouped})

    fast_rows = {
        (row["task"], row["condition"]): row
        for row in rows
        if row["method"] == "fast_lewam"
    }
    flow_step_sensitivity = []
    guidance_deltas = []
    for task in PHASE3_TASKS:
        protocol = p2_protocol(task)
        for mode in ("P0", "P2", "P3"):
            protocol_name = "not_applicable" if mode in {"P0", "P3"} else protocol
            for step in (1, 2, 5, 10, 16, 32):
                baseline_name = f"{mode}/{protocol_name}/none/step_{step}/euler"
                baseline = fast_rows[(task, baseline_name)]
                flow_step_sensitivity.append(
                    {
                        "task": task,
                        "mode": mode,
                        "guidance": "none",
                        "step": step,
                        "success_rate_percent": baseline["success_rate_percent"],
                        "successes": baseline["successes"],
                    }
                )
                guidance_names = (
                    ("guided_flow", "post_opt")
                    if mode in {"P0", "P2"}
                    else ("guided_flow", "post_opt", "post_opt_refine")
                )
                for guidance in guidance_names:
                    guided_name = f"{mode}/{protocol_name}/{guidance}/step_{step}/euler"
                    guided = fast_rows[(task, guided_name)]
                    guidance_deltas.append(
                        {
                            "task": task,
                            "mode": mode,
                            "guidance": guidance,
                            "step": step,
                            "delta_pp": guided["success_rate_percent"] - baseline["success_rate_percent"],
                        }
                    )

    analysis = {
        "experiment": "Round 5 Phase 3",
        "seed": SEED,
        "evaluation_seed": EVAL_SEED,
        "tasks": list(PHASE3_TASKS),
        "expected_result_count": expected,
        "result_count": len(rows),
        "cohort_hashes": expected_hashes,
        "primary": primary,
        "primary_comparisons": comparisons,
        "scene_target_task_groups": scene_groups,
        "flow_step_sensitivity": flow_step_sensitivity,
        "guidance_deltas": guidance_deltas,
        "rows": [{key: value for key, value in row.items() if key not in {"payload", "success_vector"}} for row in rows],
        "code_commit": _git_commit(),
    }
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [key for key in analysis["rows"][0] if key not in {"payload"}]
    with (analysis_dir / "conditions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(analysis["rows"])
    report = _render_report(analysis, REPORT_PATH, output_root)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(json.dumps({"status": "ok", "result_count": len(rows), "analysis": str(analysis_dir / "analysis.json"), "report": str(REPORT_PATH)}, sort_keys=True))


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _render_report(analysis: Mapping[str, Any], report_path: Path, output_root: Path) -> str:
    rows = analysis["rows"]
    by_key = {(row["task"], row["method"], row["condition"]): row for row in rows}
    lines = [
        "# Round 5 Phase 3：新增任务上的 FastLeWAM 与基线对比",
        "",
        "本报告使用单训练 seed 3072、评测 seed 42 和固定 legacy 50-episode cohort。结果是描述性单种子实验，不构成 DeWM 论文六种子结果的严格复现。",
        "",
        "## 主比较",
        "",
        "主比较固定为 FastLeWAM R4-AB 的 P3 / action-flow step 1 / no guidance；LeWM 使用 CEM 300/30/30，LeFlow 使用 64 paths / 16 flow steps。",
        "",
        "| task | FastLeWAM P3/s1/none | LeWM | LeFlow |",
        "|---|---:|---:|---:|",
    ]
    for task in PHASE3_TASKS:
        key = next(key for key in by_key if key[0] == task and key[1] == "fast_lewam" and key[2].startswith("P3/not_applicable/none/step_1"))
        cells = [by_key[key], by_key[(task, "lewm", "standard")], by_key[(task, "leflow", "standard")]]
        lines.append("| {} | {} | {} | {} |".format(task, *[f"{item['successes']}/50 ({item['success_rate_percent']:.1f}%, CI [{item['wilson_95_percent_low']:.1f}, {item['wilson_95_percent_high']:.1f}])" for item in cells]))
    lines.extend(["", "| task | baseline | Δ pp (Fast − baseline) | improved | regressed | McNemar p |", "|---|---|---:|---:|---:|---:|"])
    for item in analysis["primary_comparisons"]:
        lines.append(f"| {item['task']} | {item['baseline']} | {item['delta_pp']:.1f} | {item['improved']} | {item['regressed']} | {item['mcnemar_exact_two_sided_p']:.6f} |")
    lines.extend(["", "## Flow-step 敏感性（无 guidance）", "", "| task | mode | step | successes/50 | success rate |", "|---|---|---:|---:|---:|"])
    for item in analysis["flow_step_sensitivity"]:
        lines.append(f"| {item['task']} | {item['mode']} | {item['step']} | {item['successes']}/50 | {item['success_rate_percent']:.1f}% |")
    lines.extend(["", "## Guidance 相对无 guidance 的变化", "", "| task | mode | guidance | step | Δ pp |", "|---|---|---|---:|---:|"])
    for item in analysis["guidance_deltas"]:
        lines.append(f"| {item['task']} | {item['mode']} | {item['guidance']} | {item['step']} | {item['delta_pp']:+.1f} |")
    lines.extend(["", "## 条件矩阵验收", "", f"- 结果数：{analysis['result_count']} / {analysis['expected_result_count']}。", "- 每个结果：50 episodes；成功率来自 runtime termination。", "- 三个任务共享各自固定 cohort；cohort hash 见 `analysis/analysis.json`。", "- DMC 的 NaN `success` 列未被用作指标。", "", "## Scene target_task 分组", ""])
    for group in analysis["scene_target_task_groups"]:
        lines.append(f"- `{group['method']}` / `{group['condition']}`：" + ", ".join(f"{name}={value['successes']}/{value['episodes']} ({value['success_rate_percent']:.1f}%)" for name, value in sorted(group["groups"].items())))
    lines.extend(["", "## 产物", "", f"- 输出根目录：`{output_root}`", f"- 条件明细：`{output_root / 'analysis' / 'conditions.csv'}`", f"- 分析 JSON：`{output_root / 'analysis' / 'analysis.json'}`", f"- 本报告：`{report_path}`", ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.set_defaults(function=command_prepare)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--device", default="cuda")
    smoke.add_argument("--gpu", type=_gpu)
    smoke.set_defaults(function=command_smoke)
    evaluate = sub.add_parser("eval-fast")
    evaluate.add_argument("task", choices=PHASE3_TASKS)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--gpu", type=_gpu)
    evaluate.add_argument("--condition-index", type=int, action="append")
    evaluate.set_defaults(function=command_eval_fast)
    baseline = sub.add_parser("eval-baseline")
    baseline.add_argument("task", choices=PHASE3_TASKS)
    baseline.add_argument("method", choices=("lewm", "leflow"))
    baseline.add_argument("--checkpoint", required=True)
    baseline.add_argument("--device", default="cuda")
    baseline.add_argument("--gpu", type=_gpu)
    baseline.set_defaults(function=command_eval_baseline)
    analyze = sub.add_parser("analyze")
    analyze.set_defaults(function=command_analyze)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
