"""Canonical four-by-fifty final-cohort evaluation for Round 3 curves."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any


EVALUATION_BATCH_SIZE = 50
SUPPORTED_TASKS = ("reacher", "pusht")


def _batch_manifest(parent: Any, *, batch_index: int, entries: tuple[Any, ...]) -> Any:
    from source.common.round3_phase1 import CohortManifest

    return CohortManifest(
        task=parent.task,
        cohort_id=f"{parent.cohort_id}_batch_{int(batch_index):02d}",
        cohort_kind="custom",
        protocol_variant=parent.protocol_variant,
        seed=int(parent.seed),
        goal_offset_steps=int(parent.goal_offset_steps),
        entries=entries,
        episode_split={"custom": tuple(entry.episode_id for entry in entries)},
        strata_boundaries=(),
        candidate_counts=(),
        selected_counts=(),
        sampling_rule={
            "derived_from_cohort_sha256": parent.computed_sha256,
            "partition": "contiguous_parent_entry_order",
            "batch_index": int(batch_index),
            "batch_size": len(entries),
        },
        diagnostics={
            "parent_cohort_id": parent.cohort_id,
            "parent_cohort_sha256": parent.computed_sha256,
            "parent_cohort_kind": parent.cohort_kind,
        },
    )


def _valid_result(path: Path, cohort_sha256: str, count: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("status") == "ok"
        and payload.get("cohort_sha256") == cohort_sha256
        and int(payload.get("parameters", {}).get("num_eval", -1)) == int(count)
        and len(payload.get("episodes", ())) == int(count)
    )


def evaluate_final_checkpoint(
    *,
    checkpoint: str | Path,
    task: str,
    output_dir: str | Path,
    device: str,
    final_cohort_path: str | Path | None = None,
    trace: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Evaluate one Fast-LeWAM checkpoint on the frozen 200-episode cohort.

    The world is instantiated four times with 50 environments.  This keeps
    the cohort identity fixed while avoiding the EGL framebuffer pressure of a
    single 200-environment renderer.
    """
    if str(task) not in SUPPORTED_TASKS:
        raise ValueError(f"final Round3 evaluation only supports {SUPPORTED_TASKS}")
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    from omegaconf import OmegaConf

    from source.common.checkpoint import load_policy_or_model
    from source.common.eval import EvaluationIdentity, compose_eval_config, get_dataset
    from source.common.gpu_environment import configure_mujoco_egl_device
    from source.common.round3_eval import run_round3_evaluation, validate_gpu_visibility
    from source.common.round3_phase1 import CohortManifest, enrich_result_payload, write_round3_result

    validate_gpu_visibility(device)
    if str(device).startswith("cuda"):
        configure_mujoco_egl_device()
    cohort_path = (
        Path(final_cohort_path).resolve()
        if final_cohort_path is not None
        else Path("outputs/round3/phase1/cohorts") / str(task) / "final.json"
    )
    if not cohort_path.is_absolute():
        cohort_path = (Path(__file__).resolve().parents[2] / cohort_path).resolve()
    manifest = CohortManifest.load(cohort_path)
    if manifest.task != str(task) or manifest.cohort_kind != "final":
        raise ValueError("final evaluator requires the task's final cohort manifest")
    if len(manifest.entries) != 200:
        raise ValueError("final evaluator requires exactly 200 cohort entries")

    cfg = compose_eval_config(
        str(task),
        (f"solver.device={device}", "output.save_video=false"),
    )
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    batch_cfg = OmegaConf.merge(
        cfg,
        {
            "eval": {"num_eval": EVALUATION_BATCH_SIZE},
            "solver": {"device": str(device)},
            "output": {"save_video": False},
            "world": {"num_envs": EVALUATION_BATCH_SIZE},
        },
    )

    target = output_dir / "result.json"
    if resume and _valid_result(target, manifest.computed_sha256, 200):
        return json.loads(target.read_text(encoding="utf-8"))
    batch_results: list[dict[str, Any]] = []
    batch_paths: list[str] = []
    for batch_index, start in enumerate(range(0, 200, EVALUATION_BATCH_SIZE)):
        entries = tuple(manifest.entries[start : start + EVALUATION_BATCH_SIZE])
        batch_manifest = _batch_manifest(
            manifest,
            batch_index=batch_index,
            entries=entries,
        )
        batch_dir = output_dir / "batches" / f"batch_{batch_index:02d}"
        batch_result_path = batch_dir / "result.json"
        if resume and _valid_result(
            batch_result_path,
            batch_manifest.computed_sha256,
            EVALUATION_BATCH_SIZE,
        ):
            batch_result = json.loads(batch_result_path.read_text(encoding="utf-8"))
        else:
            batch_dir.mkdir(parents=True, exist_ok=True)
            batch_manifest.save(batch_dir / "cohort.json")
            model, resolved = load_policy_or_model(str(checkpoint))
            identity = EvaluationIdentity(
                entrypoint="round3_online_supervision",
                policy_kind="e5_fast",
                checkpoint=str(resolved or checkpoint),
                epoch=10,
                stage="stage_b",
            )
            batch_result = run_round3_evaluation(
                batch_cfg,
                task=str(task),
                policy_or_model=model,
                identity=identity,
                manifest=batch_manifest,
                dataset=dataset,
                output_dir=batch_dir,
                trace_output_dir=batch_dir / "trace" if trace else None,
                device=device,
                trace=bool(trace),
            )
            del model
            gc.collect()
            if str(device).startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
        batch_results.append(batch_result)
        batch_paths.append(str(batch_result_path))

    episodes: list[dict[str, Any]] = []
    for batch_index, result in enumerate(batch_results):
        rows = result.get("episodes", ())
        if len(rows) != EVALUATION_BATCH_SIZE:
            raise ValueError(f"final evaluation batch {batch_index} has the wrong size")
        for row in rows:
            bound = dict(row)
            bound["slot"] = len(episodes)
            episodes.append(bound)
    if len(episodes) != 200:
        raise ValueError("final evaluation aggregate does not contain 200 episodes")
    base = {
        "task": str(task),
        "local_dataset": str(cfg.eval.dataset_name),
        "benchmark_dataset": str(cfg.eval.get("benchmark_dataset_name", cfg.eval.dataset_name)),
        "entrypoint": "round3_online_supervision",
        "policy_kind": "e5_fast",
        "checkpoint": str(checkpoint),
        "epoch": 10,
        "stage": "stage_b",
        "parameters": {
            "num_eval": 200,
            "evaluation_batch_size": EVALUATION_BATCH_SIZE,
            "evaluation_batch_count": 4,
            "save_video": False,
            "solver": OmegaConf.to_container(cfg.solver, resolve=True),
            "plan_config": OmegaConf.to_container(cfg.plan_config, resolve=True),
        },
        "evaluation_seconds": float(
            sum(float(item.get("evaluation_seconds", 0.0)) for item in batch_results)
        ),
        "success_rate": float(sum(bool(item.get("success")) for item in episodes) / 200.0),
        "episodes": episodes,
        "evaluation_batches": [
            {
                "index": int(index),
                "path": path,
                "cohort_sha256": result.get("cohort_sha256"),
                "success_rate": float(result.get("success_rate", 0.0)),
                "episodes": len(result.get("episodes", ())),
            }
            for index, (result, path) in enumerate(zip(batch_results, batch_paths))
        ],
    }
    payload = enrich_result_payload(
        base,
        manifest=manifest,
        protocol_variant=manifest.protocol_variant,
        trace_records=None,
    )
    payload["evaluation_batch_size"] = EVALUATION_BATCH_SIZE
    payload["evaluation_batch_count"] = 4
    return write_round3_result(payload, output_dir)


__all__ = ["EVALUATION_BATCH_SIZE", "evaluate_final_checkpoint"]
