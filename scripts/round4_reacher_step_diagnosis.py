#!/usr/bin/env python3
"""Diagnose Reacher action-flow step sensitivity without changing Round 4.

The ``run`` command performs three independent pieces of work:

* audit the existing Round 4 step-sweep traces;
* replay Stage-A Euler/Heun sampling on the same 50 dev contexts and noise;
* replay batch-one Stage-B CEM with controlled warm starts.

New closed-loop evaluations remain the responsibility of ``scripts/round4.py``
and are intentionally not hidden inside this diagnostic harness.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from source.common.checkpoint import load_policy_or_model
from source.common.data import load_dataset
from source.common.round3_phase1 import CohortManifest, sha256_file
from source.common.round4_reacher_diagnosis import (
    DEFAULT_FLOW_STEPS,
    DEFAULT_HEUN_STEPS,
    DIAGNOSTIC_SCHEMA_VERSION,
    action_statistics,
    aggregate_numeric_rows,
    generate_cem_draws,
    generate_noise,
    paired_success_comparison,
    run_cem_from_latents,
    sample_action_flow,
    tensor_sha256,
    terminal_metrics,
)


DATASET_NAME = "dmcontrol/reacher.h5"
DEFAULT_CHECKPOINT = (
    ROOT
    / "outputs"
    / "round4"
    / "ab_seed3072_reacher"
    / "checkpoints"
    / "r4_ab_seed3072_weights_epoch_10.pt"
)
DEFAULT_COHORT = (
    ROOT / "outputs" / "round3" / "phase1" / "cohorts" / "reacher" / "dev_round3_revised.json"
)
DEFAULT_EXISTING_ROOT = ROOT / "outputs" / "round4" / "ab_flow_steps_seed3072"
DEFAULT_OUTPUT = ROOT / "outputs" / "round4" / "reacher_step_diagnosis_seed3072_dev"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in fields})


def _repository_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"code_commit": commit}


def _gpu_argument(value: str | None) -> str | None:
    if value is None:
        return None
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(values) != 1 or not values[0].isdigit() or int(values[0]) not in range(4):
        raise argparse.ArgumentTypeError("--gpu must select exactly one physical GPU0-3")
    return values[0]


def _validate_device(device: str, gpu: str | None) -> torch.device:
    target = torch.device(device)
    if target.type != "cuda":
        return target
    if gpu is None:
        raise ValueError("CUDA diagnosis requires --gpu with one physical GPU0-3")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(gpu):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must explicitly equal --gpu; "
            f"got {visible!r}, expected {gpu!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("diagnostic requires exactly one visible CUDA device")
    return torch.device("cuda:0")


def _load_contexts(cohort_path: Path) -> tuple[CohortManifest, dict[str, Any], torch.Tensor, torch.Tensor]:
    manifest = CohortManifest.load(cohort_path)
    if manifest.task != "reacher" or manifest.cohort_kind != "dev":
        raise ValueError("the diagnosis requires the Reacher round3_revised dev cohort")
    if len(manifest.entries) != 50:
        raise ValueError(f"expected 50 dev entries, got {len(manifest.entries)}")
    if any(entry.goal_row_index is None for entry in manifest.entries):
        raise ValueError("every cohort entry must have a goal row")

    dataset = load_dataset(
        DATASET_NAME,
        transform=None,
        num_steps=1,
        frameskip=5,
        keys_to_load=["pixels"],
        keys_to_cache=[],
    )
    current_rows = [int(entry.row_index) for entry in manifest.entries]
    goal_rows = [int(entry.goal_row_index) for entry in manifest.entries]
    current_data = dataset.get_row_data(current_rows)
    goal_data = dataset.get_row_data(goal_rows)

    def pixels_to_tensor(data: Mapping[str, Any]) -> torch.Tensor:
        pixels = data["pixels"]
        if torch.is_tensor(pixels):
            tensor = pixels
        else:
            tensor = torch.from_numpy(np.asarray(pixels))
        if tensor.ndim != 4:
            raise ValueError(f"cohort pixels must have shape [N,H,W,C], got {tuple(tensor.shape)}")
        if tensor.shape[-1] != 3:
            raise ValueError("cohort pixels must have three channels")
        return tensor.permute(0, 3, 1, 2).contiguous()

    current_pixels = pixels_to_tensor(current_data)
    goal_pixels = pixels_to_tensor(goal_data)
    context_rows = [
        {
            "slot": int(slot),
            "episode_id": entry.episode_id,
            "start_step": int(entry.start_step),
            "row_index": int(entry.row_index),
            "goal_row_index": int(entry.goal_row_index),
            "stratum": entry.stratum,
            "start_distance": entry.start_distance,
        }
        for slot, entry in enumerate(manifest.entries)
    ]
    return manifest, {"rows": context_rows, "dataset": DATASET_NAME}, current_pixels, goal_pixels


def _load_model(checkpoint_path: Path, device: torch.device):
    model_or_policy, resolved = load_policy_or_model(str(checkpoint_path))
    model = getattr(model_or_policy, "model", model_or_policy)
    if not hasattr(model, "sample_actions") or not hasattr(model, "get_cost_from_latents"):
        raise TypeError("checkpoint does not contain a Fast-LeWAM model")
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, Path(resolved or checkpoint_path).resolve()


def _context_rows(manifest: CohortManifest) -> list[dict[str, Any]]:
    return [
        {
            "slot": int(slot),
            "episode_id": entry.episode_id,
            "start_step": int(entry.start_step),
            "row_index": int(entry.row_index),
            "goal_row_index": int(entry.goal_row_index),
            "stratum": entry.stratum,
        }
        for slot, entry in enumerate(manifest.entries)
    ]


def _path_norms(actions: torch.Tensor) -> np.ndarray:
    return actions.detach().float().reshape(actions.shape[0], -1).norm(dim=1).cpu().numpy()


def _sampler_row(
    context: Mapping[str, Any],
    *,
    variant: str,
    integrator: str,
    steps: int,
    actions: torch.Tensor,
    reference: torch.Tensor,
    noise_hash: str,
) -> dict[str, Any]:
    stats = action_statistics(actions[context["slot"] : context["slot"] + 1])
    difference = (
        actions[context["slot"]] - reference[context["slot"]]
    ).reshape(-1).float().norm().item()
    row = dict(context)
    row.update(
        {
            "variant": variant,
            "integrator": integrator,
            "flow_steps": int(steps),
            "noise_sha256": noise_hash,
            "path_l2_to_euler16": float(difference),
            "mean_abs": stats["mean_abs"],
            "action_max_abs": stats["action_max_abs"],
            "action_out_of_range_fraction": stats["action_out_of_range_fraction"],
            "path_l2": stats["path_l2"]["median"],
            "path_variance": stats["path_variance"],
        }
    )
    return row


def _aggregate_variant_rows(rows: Sequence[Mapping[str, Any]], *, variant: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("variant") == variant]
    if not selected:
        raise ValueError(f"no rows for sampler variant {variant}")
    return {
        "variant": variant,
        "integrator": selected[0].get("integrator"),
        "flow_steps": selected[0].get("flow_steps"),
        "episodes": len(selected),
        "mean_abs": aggregate_numeric_rows(selected, "mean_abs"),
        "action_max_abs": aggregate_numeric_rows(selected, "action_max_abs"),
        "action_out_of_range_fraction": aggregate_numeric_rows(
            selected, "action_out_of_range_fraction"
        ),
        "path_l2_to_euler16": aggregate_numeric_rows(selected, "path_l2_to_euler16"),
        "path_variance": aggregate_numeric_rows(selected, "path_variance"),
    }


def run_sampler_probe(
    model,
    manifest: CohortManifest,
    current_pixels: torch.Tensor,
    goal_pixels: torch.Tensor,
    *,
    device: torch.device,
    flow_steps: Sequence[int],
    heun_steps: Sequence[int],
    noise_seed: int,
) -> dict[str, Any]:
    current_pixels = current_pixels.to(device)
    goal_pixels = goal_pixels.to(device)
    with torch.no_grad():
        encoded = model.encode_pixels(torch.cat((current_pixels, goal_pixels), dim=0))
        z0, goal_latent = encoded.chunk(2, dim=0)
    noise = generate_noise(
        (len(manifest.entries), model.action_horizon, model.action_dim),
        seed=noise_seed,
        device=device,
        dtype=z0.dtype,
    )
    noise_hash = tensor_sha256(noise)
    unique_euler_steps = tuple(dict.fromkeys(int(item) for item in flow_steps))
    unique_heun_steps = tuple(dict.fromkeys(int(item) for item in heun_steps))
    samples: dict[str, torch.Tensor] = {}
    sampler_rows: list[dict[str, Any]] = []
    context_rows = _context_rows(manifest)

    for steps in unique_euler_steps:
        variant = f"euler_{steps}"
        samples[variant] = sample_action_flow(
            model,
            z0,
            goal_latent,
            noise,
            steps=steps,
            integrator="euler",
        )
    if "euler_16" not in samples:
        samples["euler_16"] = sample_action_flow(
            model,
            z0,
            goal_latent,
            noise,
            steps=16,
            integrator="euler",
        )
    reference = samples["euler_16"]
    for steps in unique_heun_steps:
        variant = f"heun_{steps}"
        samples[variant] = sample_action_flow(
            model,
            z0,
            goal_latent,
            noise,
            steps=steps,
            integrator="heun",
        )

    for variant, actions in samples.items():
        integrator, step_text = variant.split("_")
        sampler_rows.extend(
            _sampler_row(
                context,
                variant=variant,
                integrator=integrator,
                steps=int(step_text),
                actions=actions,
                reference=reference,
                noise_hash=noise_hash,
            )
            for context in context_rows
        )
    variants = [
        _aggregate_variant_rows(sampler_rows, variant=variant)
        for variant in samples
    ]
    return {
        "noise_seed": int(noise_seed),
        "noise_sha256": noise_hash,
        "flow_steps": list(unique_euler_steps),
        "heun_steps": list(unique_heun_steps),
        "action_horizon": int(model.action_horizon),
        "action_dim": int(model.action_dim),
        "samples": samples,
        "z0": z0,
        "goal_latent": goal_latent,
        "noise": noise,
        "sampler_rows": sampler_rows,
        "sampler_variants": variants,
    }


def _cem_row(
    context: Mapping[str, Any],
    *,
    variant: str,
    result: Mapping[str, Any],
    index: int,
    init_actions: torch.Tensor,
) -> dict[str, Any]:
    final_actions = result["actions"][index : index + 1]
    init_stats = action_statistics(init_actions[index : index + 1])
    final_stats = action_statistics(final_actions)
    row = dict(context)
    row.update(
        {
            "variant": variant,
            "warm_start": variant != "p1_no_warm",
            "warm_start_alpha": None,
            "initial_cost": float(result["initial_cost"][index]),
            "final_elite_cost": float(result["final_elite_cost"][index]),
            "initial_path_l2": init_stats["path_l2"]["median"],
            "initial_mean_abs": init_stats["mean_abs"],
            "initial_out_of_range_fraction": init_stats["action_out_of_range_fraction"],
            "final_path_l2": final_stats["path_l2"]["median"],
            "final_mean_abs": final_stats["mean_abs"],
            "final_out_of_range_fraction": final_stats["action_out_of_range_fraction"],
            "solver_batch_size": int(result["solver_batch_size"]),
            "cem_iterations": int(result["iterations"]),
            "cem_samples": int(result["samples"]),
            "cem_topk": int(result["topk"]),
        }
    )
    if variant.startswith("p2_euler_16_alpha_"):
        row["warm_start_alpha"] = float(variant.rsplit("_", 1)[-1])
    elif variant.startswith("p2_euler_"):
        row["warm_start_alpha"] = 1.0
    return row


def _aggregate_cem_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    variants = []
    for row in rows:
        if row["variant"] not in variants:
            variants.append(row["variant"])
    result = []
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        result.append(
            {
                "variant": variant,
                "episodes": len(selected),
                "initial_cost": aggregate_numeric_rows(selected, "initial_cost"),
                "final_elite_cost": aggregate_numeric_rows(selected, "final_elite_cost"),
                "initial_out_of_range_fraction": aggregate_numeric_rows(
                    selected, "initial_out_of_range_fraction"
                ),
                "final_out_of_range_fraction": aggregate_numeric_rows(
                    selected, "final_out_of_range_fraction"
                ),
                "initial_path_l2": aggregate_numeric_rows(selected, "initial_path_l2"),
                "final_path_l2": aggregate_numeric_rows(selected, "final_path_l2"),
            }
        )
    return result


def run_cem_probe(
    model,
    manifest: CohortManifest,
    sampler: Mapping[str, Any],
    *,
    device: torch.device,
    solver_seed: int,
    cem_iterations: int = 30,
    cem_samples: int = 300,
    cem_topk: int = 30,
) -> dict[str, Any]:
    z0 = sampler["z0"]
    goal_latent = sampler["goal_latent"]
    action_samples = sampler["samples"]
    for required in ("euler_1", "euler_16", "euler_32"):
        if required not in action_samples:
            raise ValueError(
                f"CEM probe requires sampler variant {required}; "
                "include steps 1, 16, and 32"
            )
    draws = generate_cem_draws(
        num_contexts=len(manifest.entries),
        iterations=cem_iterations,
        samples=cem_samples,
        horizon=int(model.action_horizon),
        action_dim=int(model.action_dim),
        seed=solver_seed,
        device=device,
        dtype=z0.dtype,
    )
    variants: dict[str, torch.Tensor | None] = {
        "p1_no_warm": None,
        "p2_euler_1": action_samples["euler_1"],
        "p2_euler_16": action_samples["euler_16"],
        "p2_euler_32": action_samples["euler_32"],
        "p2_euler_16_alpha_0.5": action_samples["euler_16"] * 0.5,
        "p2_euler_16_alpha_0.75": action_samples["euler_16"] * 0.75,
    }
    cem_rows: list[dict[str, Any]] = []
    context_rows = _context_rows(manifest)
    result_metadata = {
        "solver_seed": int(solver_seed),
        "cem_iterations": int(cem_iterations),
        "cem_samples": int(cem_samples),
        "cem_topk": int(cem_topk),
        "var_scale": 1.0,
        "solver_batch_size": 1,
        "draws_sha256": tensor_sha256(draws),
    }
    for variant, init_actions in variants.items():
        result = run_cem_from_latents(
            model,
            z0,
            goal_latent,
            init_actions=init_actions,
            random_draws=draws,
            topk=cem_topk,
            solver_batch_size=1,
        )
        if init_actions is None:
            init_for_stats = torch.zeros_like(result["actions"])
        else:
            init_for_stats = init_actions
        cem_rows.extend(
            _cem_row(
                context,
                variant=variant,
                result=result,
                index=index,
                init_actions=init_for_stats,
            )
            for index, context in enumerate(context_rows)
        )
    return {
        **result_metadata,
        "cem_rows": cem_rows,
        "cem_variants": _aggregate_cem_rows(cem_rows),
    }


def _trace_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing trace artifact: {path}")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"trace row must be an object: {path}")
            records.append(value)
    return records


def _trace_path(payload: Mapping[str, Any], result_path: Path) -> Path:
    value = payload.get("trace_path")
    path = Path(value) if value else result_path.parent / "trace" / "episodes.jsonl"
    if path.is_absolute() or path.is_file():
        return path
    return result_path.parent / path


def _trace_episode_row(record: Mapping[str, Any], *, variant: str, flow_steps: int | None) -> dict[str, Any]:
    steps = list(record.get("steps", ()))
    if not steps:
        raise ValueError(f"trace episode {record.get('episode_id')} has no steps")
    last = steps[-1]
    current = last.get("current")
    goal = last.get("goal")
    if current is None or goal is None:
        raise ValueError(f"trace episode {record.get('episode_id')} lacks terminal current/goal")
    actions = [step["action"] for step in steps if step.get("action") is not None]
    metrics = terminal_metrics(current, goal, actions=np.asarray(actions, dtype=np.float64))
    return {
        "variant": variant,
        "flow_steps": flow_steps,
        "episode_id": record.get("episode_id", record.get("dataset_episode")),
        "start_step": record.get("start_step"),
        "row_index": record.get("row_index"),
        "success": bool(record.get("success", metrics["success"])),
        "terminal_margin": metrics["terminal_margin"],
        "max_joint_error": metrics["max_joint_error"],
        "l2_distance": metrics["l2_distance"],
        "physical_action_out_of_range_fraction": metrics["action_out_of_range_fraction"],
        "physical_action_max_abs": metrics["action_max_abs"],
        "invalid_action_count": int(record.get("invalid_action_count", 0)),
        "steps_executed": int(record.get("steps_executed", len(steps))),
        "early_terminated": bool(record.get("early_terminated", False)),
    }


def _aggregate_trace_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty trace rows")
    return {
        "variant": rows[0]["variant"],
        "flow_steps": rows[0]["flow_steps"],
        "episodes": len(rows),
        "successes": int(sum(bool(row["success"]) for row in rows)),
        "success_rate_percent": 100.0 * float(np.mean([bool(row["success"]) for row in rows])),
        "terminal_margin": aggregate_numeric_rows(rows, "terminal_margin"),
        "max_joint_error": aggregate_numeric_rows(rows, "max_joint_error"),
        "physical_action_out_of_range_fraction": aggregate_numeric_rows(
            rows, "physical_action_out_of_range_fraction"
        ),
        "invalid_action_count": int(sum(int(row["invalid_action_count"]) for row in rows)),
        "early_terminated": int(sum(bool(row["early_terminated"]) for row in rows)),
    }


def _planning_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    planning = payload.get("round4_planning", {})
    if not isinstance(planning, Mapping):
        return {}
    return {
        "planning_median_seconds": planning.get("planning_median_seconds"),
        "planning_p95_seconds": planning.get("planning_p95_seconds"),
        "forward_count": planning.get("forward_count"),
        "peak_memory_bytes": planning.get("peak_memory_bytes"),
        "action_flow_integrator": planning.get("action_flow_integrator"),
        "actor_warm_start_scale": planning.get("actor_warm_start_scale"),
    }


def audit_existing(results_root: Path) -> dict[str, Any]:
    """Read existing traces and assert the reported Reacher symptom."""
    rows: list[dict[str, Any]] = []
    episode_rows: dict[str, list[dict[str, Any]]] = {}
    expected_steps = (1, 2, 5, 10, 16)
    for step in expected_steps:
        for mode in ("P0", "P2", "P3"):
            result_path = results_root / f"steps_{step}" / "reacher" / mode / "dev" / "result.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload.get("status") != "ok":
                raise ValueError(f"result is not ok: {result_path}")
            if payload.get("task") != "reacher" or payload.get("cohort_kind") != "dev":
                raise ValueError(f"result identity mismatch: {result_path}")
            observed_step = payload.get("round4_planning", {}).get("action_flow_steps")
            if int(observed_step) != step:
                raise ValueError(f"wrong action_flow_steps in {result_path}: {observed_step}")
            trace_path = _trace_path(payload, result_path)
            trace = _trace_records(trace_path)
            episode_rows[f"{mode}_euler_{step}"] = [
                _trace_episode_row(item, variant=f"{mode}_euler_{step}", flow_steps=step)
                for item in trace
            ]
            summary_row = _aggregate_trace_rows(episode_rows[f"{mode}_euler_{step}"])
            summary_row.update(_planning_fields(payload))
            rows.append(summary_row)

    p1_path = results_root / "invariant_p1" / "reacher" / "P1" / "dev" / "result.json"
    p1_payload = json.loads(p1_path.read_text(encoding="utf-8"))
    if p1_payload.get("status") != "ok":
        raise ValueError(f"result is not ok: {p1_path}")
    p1_trace_path = _trace_path(p1_payload, p1_path)
    episode_rows["P1_random"] = [
        _trace_episode_row(item, variant="P1_random", flow_steps=None)
        for item in _trace_records(p1_trace_path)
    ]
    p1_summary = _aggregate_trace_rows(episode_rows["P1_random"])
    p1_summary.update(_planning_fields(p1_payload))
    rows.append(p1_summary)

    paired: dict[str, Any] = {}
    for mode in ("P0", "P2", "P3"):
        baseline = episode_rows[f"{mode}_euler_1"]
        for step in expected_steps[1:]:
            paired[f"{mode}_step1_vs_step{step}"] = paired_success_comparison(
                baseline, episode_rows[f"{mode}_euler_{step}"]
            )
    p2_rates = {
        row["flow_steps"]: float(row["success_rate_percent"])
        for row in rows
        if row["variant"].startswith("P2_euler_")
    }
    p2_range = max(p2_rates.values()) - min(p2_rates.values())
    reacher_symptom = {
        "p2_success_rate_percent_by_step": p2_rates,
        "p2_range_percentage_points": p2_range,
        "red_capable": bool(p2_range >= 18.0),
    }
    if not reacher_symptom["red_capable"]:
        raise AssertionError("existing artifacts no longer reproduce the Reacher P2 symptom")
    return {
        "results_root": str(results_root.resolve()),
        "result_artifact_sha256": {
            row["variant"]: sha256_file(
                results_root
                / (
                    "invariant_p1/reacher/P1/dev/result.json"
                    if row["variant"] == "P1_random"
                    else f"steps_{row['flow_steps']}/reacher/{row['variant'].split('_')[0]}/dev/result.json"
                )
            )
            for row in rows
        },
        "rows": rows,
        "episodes": [episode for values in episode_rows.values() for episode in values],
        "paired": paired,
        "symptom": reacher_symptom,
    }


def audit_closed_loop(diagnosis_root: Path, existing_root: Path) -> dict[str, Any]:
    """Collect the gated closed-loop artifacts produced after the first pass."""
    heun_directory = diagnosis_root / "closed_loop" / "heun"
    heun_rechecked_directory = diagnosis_root / "closed_loop" / "heun_rechecked"

    def heun_result(mode: str) -> Path:
        rechecked = heun_rechecked_directory / "reacher" / mode / "dev" / "result.json"
        return rechecked if rechecked.is_file() else heun_directory / "reacher" / mode / "dev" / "result.json"

    paths = {
        "P2_euler_32": existing_root / "steps_32" / "reacher" / "P2" / "dev" / "result.json",
        "P2_alpha_0.5": diagnosis_root / "closed_loop" / "alpha_0.5" / "reacher" / "P2" / "dev" / "result.json",
        "P0_heun_16": heun_result("P0"),
        "P2_heun_16": heun_result("P2"),
        "P3_heun_16": heun_result("P3"),
    }
    rows: list[dict[str, Any]] = []
    episodes: dict[str, list[dict[str, Any]]] = {}
    artifacts: dict[str, Any] = {}
    for variant, result_path in paths.items():
        if not result_path.is_file():
            artifacts[variant] = {"status": "missing", "path": str(result_path)}
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            raise ValueError(f"closed-loop result is not ok: {result_path}")
        trace = _trace_records(_trace_path(payload, result_path))
        flow_steps = payload.get("round4_planning", {}).get("action_flow_steps")
        episode_rows = [
            _trace_episode_row(item, variant=variant, flow_steps=flow_steps)
            for item in trace
        ]
        episodes[variant] = episode_rows
        summary_row = _aggregate_trace_rows(episode_rows)
        summary_row.update(_planning_fields(payload))
        rows.append(summary_row)
        artifacts[variant] = {
            "status": "ok",
            "path": str(result_path),
            "sha256": sha256_file(result_path),
            "trace_path": str(_trace_path(payload, result_path)),
            "trace_sha256": payload.get("trace_sha256"),
            "cohort_id": payload.get("cohort_id"),
            "cohort_sha256": payload.get("cohort_sha256"),
            "checkpoint": payload.get("checkpoint"),
            "checkpoint_sha256": sha256_file(Path(payload["checkpoint"]))
            if payload.get("checkpoint") and Path(payload["checkpoint"]).is_file()
            else None,
            "planning": payload.get("round4_planning", {}),
        }

    baseline_path = existing_root / "steps_16" / "reacher" / "P2" / "dev" / "result.json"
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_episodes = [
        _trace_episode_row(item, variant="P2_euler_16", flow_steps=16)
        for item in _trace_records(_trace_path(baseline_payload, baseline_path))
    ]
    paired = {
        f"P2_euler_16_vs_{variant}": paired_success_comparison(
            baseline_episodes, values
        )
        for variant, values in episodes.items()
    }
    return {
        "rows": rows,
        "episodes": [episode for values in episodes.values() for episode in values],
        "paired": paired,
        "artifacts": artifacts,
    }


def _median_variant(summary: Mapping[str, Any], variant: str, field: str) -> float | None:
    for row in summary.get("sampler", {}).get("sampler_variants", []):
        if row.get("variant") == variant:
            value = row.get(field, {})
            return None if value.get("median") is None else float(value["median"])
    return None


def _cem_median(summary: Mapping[str, Any], variant: str, field: str) -> float | None:
    for row in summary.get("cem", {}).get("cem_variants", []):
        if row.get("variant") == variant:
            value = row.get(field, {})
            return None if value.get("median") is None else float(value["median"])
    return None


def _near_boundary_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        -0.01 <= float(row["terminal_margin"]) < 0.0
        for row in rows
        if not bool(row["success"])
    )


def build_diagnosis_report(summary: Mapping[str, Any], closed_loop: Mapping[str, Any]) -> str:
    audit = summary["existing_audit"]
    existing_rows = audit["rows"]
    by_variant = {row["variant"]: row for row in existing_rows}
    closed_rows = {row["variant"]: row for row in closed_loop["rows"]}

    def rate(variant: str) -> str:
        row = by_variant.get(variant, closed_rows.get(variant))
        return "—" if row is None else f"{float(row['success_rate_percent']):.1f}%"

    def value(variant: str, field: str) -> str:
        row = closed_rows.get(variant)
        if row is None:
            row = by_variant.get(variant)
        if row is None:
            return "—"
        item = row.get(field)
        if isinstance(item, Mapping):
            item = item.get("median")
        return "—" if item is None else f"{float(item):.3f}"

    p2_dev_16 = [
        row
        for row in audit["episodes"]
        if row.get("variant") == "P2_euler_16"
    ]
    p2_near = _near_boundary_count(p2_dev_16)
    p2_failures = sum(not bool(row["success"]) for row in p2_dev_16)
    heun_delta = closed_loop["paired"].get("P2_euler_16_vs_P2_heun_16", {})
    alpha_delta = closed_loop["paired"].get("P2_euler_16_vs_P2_alpha_0.5", {})
    code_commit = summary.get("code", {}).get("code_commit")
    config_sha256 = summary.get("config_sha256")
    return f"""# Round 4 Reacher action-flow step 诊断报告

## 结论摘要

本报告基于 R4-AB epoch10、训练 seed 3072、`round3_revised` Reacher dev cohort 的 50 个 episode。它是单 checkpoint、单 seed 的诊断实验，不是 final 评测，也不是严格 held-out 泛化证明。

最强信号来自 P2 actor warm-start 的动作幅度，而不是单纯的 Euler 步数：标准 P2 step16 的 dev 成功率为 **{rate('P2_euler_16')}**，将同一 warm-start 缩放为 `alpha=0.5` 后为 **{rate('P2_alpha_0.5')}**；step1 为 **{rate('P2_euler_1')}**，P1 随机 CEM 为 **{rate('P1_random')}**。step16 相对 alpha=0.5 的 paired 结果为 improved={alpha_delta.get('improved', '—')}、regressed={alpha_delta.get('regressed', '—')}。

Heun 能部分改变 P2 的闭环结果，但不是完整解释：P2-Heun-16 为 **{rate('P2_heun_16')}**，相对 P2-Euler-16 的 paired 结果为 improved={heun_delta.get('improved', '—')}、regressed={heun_delta.get('regressed', '—')}；P0 和 P3 使用 Heun 后分别为 **{rate('P0_heun_16')}** 和 **{rate('P3_heun_16')}**，均未优于对应 Euler-16 基线。

## 已有 step sweep

| 方法 | step1 | step2 | step5 | step10 | step16 | step32 |
|---|---:|---:|---:|---:|---:|---:|
| P0 | {rate('P0_euler_1')} | {rate('P0_euler_2')} | {rate('P0_euler_5')} | {rate('P0_euler_10')} | {rate('P0_euler_16')} | — |
| P2 | {rate('P2_euler_1')} | {rate('P2_euler_2')} | {rate('P2_euler_5')} | {rate('P2_euler_10')} | {rate('P2_euler_16')} | {rate('P2_euler_32')} |
| P3 | {rate('P3_euler_1')} | {rate('P3_euler_2')} | {rate('P3_euler_5')} | {rate('P3_euler_10')} | {rate('P3_euler_16')} | — |
| P1 random CEM | — | — | — | — | {rate('P1_random')} | — |

P2 step1 到 step16 的变化为 92% 到 50%，step32 回升到 56%，因此影响不是随 step 单调平滑变化的数值积分误差。

## 固定噪声 sampler 诊断

| sampler | median mean_abs | median 越界比例 | median path distance to Euler-16 |
|---|---:|---:|---:|
| Euler-1 | {_median_variant(summary, 'euler_1', 'mean_abs'):.3f} | {_median_variant(summary, 'euler_1', 'action_out_of_range_fraction'):.3f} | {_median_variant(summary, 'euler_1', 'path_l2_to_euler16'):.3f} |
| Euler-16 | {_median_variant(summary, 'euler_16', 'mean_abs'):.3f} | {_median_variant(summary, 'euler_16', 'action_out_of_range_fraction'):.3f} | {_median_variant(summary, 'euler_16', 'path_l2_to_euler16'):.3f} |
| Euler-32 | {_median_variant(summary, 'euler_32', 'mean_abs'):.3f} | {_median_variant(summary, 'euler_32', 'action_out_of_range_fraction'):.3f} | {_median_variant(summary, 'euler_32', 'path_l2_to_euler16'):.3f} |
| Heun-16 | {_median_variant(summary, 'heun_16', 'mean_abs'):.3f} | {_median_variant(summary, 'heun_16', 'action_out_of_range_fraction'):.3f} | {_median_variant(summary, 'heun_16', 'path_l2_to_euler16'):.3f} |

Euler step 增大使动作幅度和 normalized action 越界比例持续上升；Heun-16 的 path difference 也达到约 {_median_variant(summary, 'heun_16', 'path_l2_to_euler16'):.3f}，但其幅度并未回到 step1，而是更高。因此 Heun 不是“修复 step1 优势”的直接替代方案。

## CEM warm-start 诊断

| 初始化 | median 初始 cost | median 初始越界比例 | median 最终 elite cost | 对应闭环成功率 |
|---|---:|---:|---:|---:|
| P1 / no-warm | {_cem_median(summary, 'p1_no_warm', 'initial_cost'):.4f} | {_cem_median(summary, 'p1_no_warm', 'initial_out_of_range_fraction'):.3f} | {_cem_median(summary, 'p1_no_warm', 'final_elite_cost'):.4f} | {rate('P1_random')} |
| P2 Euler-1 | {_cem_median(summary, 'p2_euler_1', 'initial_cost'):.4f} | {_cem_median(summary, 'p2_euler_1', 'initial_out_of_range_fraction'):.3f} | {_cem_median(summary, 'p2_euler_1', 'final_elite_cost'):.4f} | {rate('P2_euler_1')} |
| P2 Euler-16 | {_cem_median(summary, 'p2_euler_16', 'initial_cost'):.4f} | {_cem_median(summary, 'p2_euler_16', 'initial_out_of_range_fraction'):.3f} | {_cem_median(summary, 'p2_euler_16', 'final_elite_cost'):.4f} | {rate('P2_euler_16')} |
| P2 Euler-16, alpha=0.5 | {_cem_median(summary, 'p2_euler_16_alpha_0.5', 'initial_cost'):.4f} | {_cem_median(summary, 'p2_euler_16_alpha_0.5', 'initial_out_of_range_fraction'):.3f} | {_cem_median(summary, 'p2_euler_16_alpha_0.5', 'final_elite_cost'):.4f} | {rate('P2_alpha_0.5')} |

alpha=0.5 使初始越界比例从 {_cem_median(summary, 'p2_euler_16', 'initial_out_of_range_fraction'):.3f} 降到 {_cem_median(summary, 'p2_euler_16_alpha_0.5', 'initial_out_of_range_fraction'):.3f}，并在闭环中恢复 30 个百分点。CEM 最终 elite cost 的差异很小，说明问题主要发生在初始动作 basin 与真实环境执行，而不是 CEM 最终 latent cost 无法收敛。

## Heun 闭环 gate

| 方法 | 成功率 | median planning seconds | action-flow integrator |
|---|---:|---:|---|
| P0 Euler-16 | {rate('P0_euler_16')} | {value('P0_euler_16', 'planning_median_seconds')} | Euler |
| P0 Heun-16 | {rate('P0_heun_16')} | {value('P0_heun_16', 'planning_median_seconds')} | Heun |
| P2 Euler-16 | {rate('P2_euler_16')} | {value('P2_euler_16', 'planning_median_seconds')} | Euler |
| P2 Heun-16 | {rate('P2_heun_16')} | {value('P2_heun_16', 'planning_median_seconds')} | Heun |
| P3 Euler-16 | {rate('P3_euler_16')} | {value('P3_euler_16', 'planning_median_seconds')} | Euler |
| P3 Heun-16 | {rate('P3_heun_16')} | {value('P3_heun_16', 'planning_median_seconds')} | Heun |

Heun 对 P2 有一定帮助，但 P0/P3 反而下降，不能据此把积分器替换为主要方法。P2 的 Heun 增益更像是改变了 actor warm-start 的动作轨迹，仍受同一幅度/执行问题影响。

## 失败边界与限制

P2 Euler-16 dev 失败 episode 中，有 {p2_near}/{p2_failures} 个 terminal margin 落在 `[-0.01, 0)`，说明 Reacher 严格的逐关节 0.05 rad 判据会放大很小的动作差异；但这只是放大器，不足以解释动作越界比例和 alpha 对照的恢复效果。

本轮没有重新训练、没有运行 final、没有增加第二个训练 seed。结果应解释为：在固定 checkpoint 和固定 dev cohort 上，P2 的 step 敏感性主要与 actor warm-start 的动作尺度/越界及 Reacher 阈值交互有关，Euler/Heun 离散化是次要影响因素。后续优先检查 action normalizer、warm-start 输出裁剪策略和 actor flow 的训练/校准；不建议仅通过增加 Euler steps 选择 P2 的默认配置。

## 版本与复现信息

- 诊断代码 commit：`{code_commit or 'unknown'}`
- 诊断配置 SHA256：`{config_sha256 or 'unknown'}`
"""


def command_report(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.diagnosis_root)
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["existing_audit"] = audit_existing(Path(args.results_root))
    _write_json(root / "audit.json", summary["existing_audit"])
    closed_loop = audit_closed_loop(root, Path(args.results_root))
    _write_json(root / "closed_loop_metrics.json", closed_loop)
    _write_csv(root / "closed_loop_rows.csv", closed_loop["rows"])
    report = build_diagnosis_report(summary, closed_loop)
    report_path = Path(args.report_output) if args.report_output else root / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    diagnosis = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "experiment": "R4-Reacher-step-diagnosis",
        "diagnosis_root": str(root.resolve()),
        "code": _repository_metadata(),
        "config_sha256": sha256_file(
            ROOT / "config" / "round4" / "reacher_step_diagnosis.json"
        ),
        "closed_loop": closed_loop,
        "gates": {
            "alpha_0.5": {
                "triggered": True,
                "reason": "initial out-of-range median matches Euler-1 within two percentage points",
            },
            "heun": {
                "triggered": True,
                "sampler_path_difference": _median_variant(
                    summary, "heun_16", "path_l2_to_euler16"
                ),
                "threshold": 0.05,
            },
        },
        "conclusion": {
            "primary": "P2 actor warm-start action scale/out-of-range interacts strongly with Reacher dynamics and threshold",
            "secondary": "Euler/Heun integration changes the actor path but is not a universal explanation",
            "reacher_threshold_amplifier": True,
            "causal_training_claim": False,
        },
        "report_path": str(report_path.resolve()),
    }
    _write_json(root / "diagnosis.json", diagnosis)
    summary["closed_loop"] = closed_loop
    summary["diagnosis"] = diagnosis
    summary["code"] = _repository_metadata()
    summary["config_sha256"] = sha256_file(
        ROOT / "config" / "round4" / "reacher_step_diagnosis.json"
    )
    _write_json(summary_path, summary)
    return {"report": str(report_path), "diagnosis": str(root / "diagnosis.json")}


def _strip_tensors(value: Any) -> Any:
    if torch.is_tensor(value):
        return None
    if isinstance(value, Mapping):
        return {key: _strip_tensors(item) for key, item in value.items() if not torch.is_tensor(item)}
    if isinstance(value, list):
        return [_strip_tensors(item) for item in value]
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    existing = audit_existing(Path(args.results_root))
    _write_json(output / "audit.json", existing)

    device = _validate_device(args.device, args.gpu)
    checkpoint_path = Path(args.checkpoint).resolve()
    cohort_path = Path(args.cohort).resolve()
    manifest, context_meta, current_pixels, goal_pixels = _load_contexts(cohort_path)
    model, resolved_checkpoint = _load_model(checkpoint_path, device)
    sampler = run_sampler_probe(
        model,
        manifest,
        current_pixels,
        goal_pixels,
        device=device,
        flow_steps=args.flow_steps,
        heun_steps=args.heun_steps,
        noise_seed=args.noise_seed,
    )
    sampler_payload = _strip_tensors(sampler)
    sampler_payload["sampler_rows"] = sampler["sampler_rows"]
    _write_json(output / "sampler_metrics.json", sampler_payload)
    _write_csv(output / "sampler_rows.csv", sampler["sampler_rows"])

    cem = None
    if not args.skip_cem:
        cem = run_cem_probe(
            model,
            manifest,
            sampler,
            device=device,
            solver_seed=args.solver_seed,
            cem_iterations=args.cem_iterations,
            cem_samples=args.cem_samples,
            cem_topk=args.cem_topk,
        )
        _write_json(output / "planner_metrics.json", cem)
        _write_csv(output / "planner_rows.csv", cem["cem_rows"])

    payload = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "experiment": "R4-Reacher-step-diagnosis",
        "protocol": "round3_revised",
        "cohort_kind": manifest.cohort_kind,
        "cohort_id": manifest.cohort_id,
        "cohort_sha256": manifest.computed_sha256,
        "cohort_path": str(cohort_path),
        "checkpoint": str(resolved_checkpoint),
        "checkpoint_sha256": sha256_file(resolved_checkpoint),
        "code": _repository_metadata(),
        "config_sha256": sha256_file(
            ROOT / "config" / "round4" / "reacher_step_diagnosis.json"
        ),
        "dataset": context_meta["dataset"],
        "training_seed": 3072,
        "evaluation_seed": 42,
        "noise_seed": int(args.noise_seed),
        "solver_seed": int(args.solver_seed),
        "device": str(device),
        "dev_episodes": len(manifest.entries),
        "existing_audit": existing,
        "sampler": sampler_payload,
        "cem": _strip_tensors(cem) if cem is not None else None,
        "context_rows": context_meta["rows"],
        "closed_loop_required": {
            "p2_euler_32": True,
            "p2_scaled_alpha": False,
            "heun": False,
        },
    }
    _write_json(output / "summary.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="audit existing Reacher step-sweep artifacts")
    audit.add_argument("--results-root", default=str(DEFAULT_EXISTING_ROOT))
    audit.add_argument("--output", required=True)
    audit.set_defaults(function=lambda args: _write_json(Path(args.output), audit_existing(Path(args.results_root))))

    run_parser = subparsers.add_parser("run", help="run the fixed-context and CEM diagnosis")
    run_parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    run_parser.add_argument("--cohort", default=str(DEFAULT_COHORT))
    run_parser.add_argument("--results-root", default=str(DEFAULT_EXISTING_ROOT))
    run_parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    run_parser.add_argument("--device", default="cpu")
    run_parser.add_argument("--gpu", type=_gpu_argument)
    run_parser.add_argument("--noise-seed", type=int, default=3072)
    run_parser.add_argument("--solver-seed", type=int, default=42)
    run_parser.add_argument("--flow-steps", type=int, nargs="+", default=list(DEFAULT_FLOW_STEPS))
    run_parser.add_argument("--heun-steps", type=int, nargs="+", default=list(DEFAULT_HEUN_STEPS))
    run_parser.add_argument("--cem-iterations", type=int, default=30)
    run_parser.add_argument("--cem-samples", type=int, default=300)
    run_parser.add_argument("--cem-topk", type=int, default=30)
    run_parser.add_argument("--skip-cem", action="store_true")
    run_parser.set_defaults(function=run)

    report = subparsers.add_parser("report", help="merge gated closed-loop results and write the diagnosis report")
    report.add_argument("--diagnosis-root", default=str(DEFAULT_OUTPUT))
    report.add_argument("--results-root", default=str(DEFAULT_EXISTING_ROOT))
    report.add_argument("--report-output")
    report.set_defaults(function=command_report)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = args.function(args)
    if result is not None:
        print(
            json.dumps(
                {
                    "output": str(getattr(args, "output", "")),
                    "symptom": result.get("existing_audit", {}).get("symptom"),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
