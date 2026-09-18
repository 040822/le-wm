"""Evaluate Round 3 Phase 2 checkpoints on a fixed 200-episode cohort.

The runner evaluates one E0/E5 arm at 1,000-environment-step intervals and
publishes an auditable result for every checkpoint.  It is deliberately
resumable: a valid result is never overwritten, and the curve index is
rewritten atomically after each completed point.

GPU callers must expose only one physical GPU from the repository allowlist
through ``CUDA_VISIBLE_DEVICES`` and should pass the corresponding logical
device (normally ``cuda:0``).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEPS = tuple(range(0, 20_001, 1_000))
TASK = "cube"
COHORT_PATH = ROOT / "outputs/round3/phase1/cohorts/cube/final.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/round3/phase2/curve_200ep_1k"
EVALUATION_BATCH_SIZE = 50

TASK_SPECS: dict[str, dict[str, str]] = {
    "cube": {
        "e0_checkpoint": "outputs/lewm/cube/0803_e0_lewm_baseline_bs128/checkpoints/lewm_weights_epoch_10.pt",
        "e5_checkpoint": "outputs/fast_lewam/cube/0803_e5_cross_head_grad/checkpoints/fast_lewam_weights_epoch_10.pt",
    },
    "pusht": {
        "e0_checkpoint": "outputs/lewm/pusht/0803_e0_lewm_baseline_bs128/checkpoints/lewm_weights_epoch_10.pt",
        "e5_checkpoint": "outputs/fast_lewam/pusht/0803_e5_cross_head_grad/checkpoints/fast_lewam_weights_epoch_10.pt",
    },
    "reacher": {
        "e0_checkpoint": "outputs/lewm/reacher/0721_/checkpoints/lewm_weights_epoch_10.pt",
        "e5_checkpoint": "outputs/fast_lewam/reacher/0717_/checkpoints/fast_lewam_weights_epoch_10.pt",
    },
    "tworoom": {
        "e0_checkpoint": "outputs/lewm/tworoom/0721_/checkpoints/lewm_weights_epoch_10.pt",
        "e5_checkpoint": "outputs/fast_lewam/tworoom/0717_/checkpoints/fast_lewam_weights_epoch_10.pt",
    },
}

EXPERIMENTS: dict[str, dict[str, str]] = {}
ARMS = ("offline_continue", "online_adapt")


def _configure_task(task: str) -> None:
    if task not in TASK_SPECS:
        raise ValueError(f"unsupported task: {task}; expected one of {tuple(TASK_SPECS)}")
    spec = TASK_SPECS[task]
    global TASK, COHORT_PATH, DEFAULT_OUTPUT_ROOT, EXPERIMENTS
    TASK = task
    COHORT_PATH = ROOT / f"outputs/round3/phase1/cohorts/{task}/final.json"
    DEFAULT_OUTPUT_ROOT = ROOT / f"outputs/round3/phase2/{task}_curve_200ep_1k"
    EXPERIMENTS = {
        "e0": {
            "method": "e0_lewm",
            "policy_kind": "lewm",
            "base_checkpoint": spec["e0_checkpoint"],
            "checkpoint_prefix": f"{task}_",
            "train_output_root": f"outputs/round3/phase2/{task}_e0_three_arms_100step_threads4_retry1",
        },
        "e5": {
            "method": "e5_fast",
            "policy_kind": "e5_fast",
            "base_checkpoint": spec["e5_checkpoint"],
            "checkpoint_prefix": f"{task}_e5_legacy_",
            "train_output_root": f"outputs/round3/phase2/{task}_e5_legacy_three_arms_100step_threads4_retry1",
        },
    }


_configure_task(TASK)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("environment_steps", "success_rate", "success_rate_percent", "episodes", "status"),
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validate_gpu_visibility(device: str) -> tuple[str, ...]:
    if not str(device).startswith("cuda"):
        raise ValueError("curve evaluation requires an explicitly selected CUDA device")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    values = tuple(item.strip() for item in visible.split(",") if item.strip())
    if not values or any(not item.isdigit() or int(item) not in range(4) for item in values):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must explicitly contain only physical GPUs 0,1,2,3; "
            f"got {visible!r}"
        )
    return values


def _checkpoint_path(experiment: str, arm: str, step: int) -> Path:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unsupported experiment: {experiment}")
    if arm not in ARMS:
        raise ValueError(f"unsupported arm: {arm}")
    if int(step) not in STEPS:
        raise ValueError(f"step must be one of {STEPS}, got {step}")
    spec = EXPERIMENTS[experiment]
    if int(step) == 0:
        return ROOT / spec["base_checkpoint"]
    return (
        ROOT
        / EXPERIMENTS[experiment]["train_output_root"]
        / "checkpoints"
        / f"{spec['checkpoint_prefix']}{arm}_envsteps_{int(step):06d}.ckpt"
    )


def _result_is_valid(path: Path, cohort_sha256: str, expected_count: int = 200) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("status") == "ok"
        and int(payload.get("parameters", {}).get("num_eval", -1)) == int(expected_count)
        and len(payload.get("episodes", ())) == int(expected_count)
        and payload.get("cohort_sha256") == cohort_sha256
    )


def _curve_rows(series_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in STEPS:
        result_path = series_dir / f"envsteps_{step:06d}" / "result.json"
        if not result_path.is_file():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        success_rate = float(payload["success_rate"])
        rows.append(
            {
                "environment_steps": int(step),
                "success_rate": success_rate,
                "success_rate_percent": success_rate * 100.0,
                "episodes": len(payload.get("episodes", ())),
                "status": str(payload.get("status", "unknown")),
            }
        )
    return rows


def _publish_curve(
    series_dir: Path,
    *,
    experiment: str,
    arm: str,
    cohort_sha256: str,
    visible_gpus: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = _curve_rows(series_dir)
    complete = len(rows) == len(STEPS) and all(row["status"] == "ok" for row in rows)
    metadata = {
        "schema_version": 1,
        "status": "ok" if complete else "running",
        "task": TASK,
        "experiment": experiment,
        "arm": arm,
        "method": EXPERIMENTS[experiment]["method"],
        "cohort": {
            "path": str(COHORT_PATH),
            "kind": "final",
            "episodes": 200,
            "canonical_sha256": cohort_sha256,
        },
        "step_unit": "environment_steps",
        "step_increment": 1000,
        "evaluation_steps": list(STEPS),
        "save_video": False,
        "evaluation_batching": {
            "batch_size": EVALUATION_BATCH_SIZE,
            "batch_count": 200 // EVALUATION_BATCH_SIZE,
            "aggregate_episode_count": 200,
            "reason": "avoid EGL framebuffer failure from 200 simultaneous renderers",
        },
        "visible_physical_gpus": list(visible_gpus),
        "results": rows,
    }
    _write_json_atomic(series_dir / "curve.json", metadata)
    _write_csv_atomic(series_dir / "curve.csv", rows)
    return rows


def _make_batch_manifest(parent: Any, *, batch_index: int, entries: tuple[Any, ...]) -> Any:
    """Derive a self-hashing custom manifest for one fixed final-cohort batch."""
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


def _aggregate_batches(
    *,
    parent_manifest: Any,
    batch_results: list[dict[str, Any]],
    batch_paths: list[str],
    output_dir: Path,
    experiment: str,
    arm: str,
    checkpoint: Path,
    save_video: bool,
    trace: bool,
) -> dict[str, Any]:
    """Publish one canonical 200-episode result from four 50-episode runs."""
    from source.common.round3_phase1 import enrich_result_payload, write_round3_result

    if len(batch_results) != 200 // EVALUATION_BATCH_SIZE:
        raise ValueError("the curve aggregate requires exactly four 50-episode batches")
    episodes: list[dict[str, Any]] = []
    for batch_index, result in enumerate(batch_results):
        batch_episodes = result.get("episodes", ())
        if len(batch_episodes) != EVALUATION_BATCH_SIZE:
            raise ValueError(f"batch {batch_index} does not contain 50 episodes")
        for record in batch_episodes:
            bound = dict(record)
            bound["slot"] = len(episodes)
            episodes.append(bound)
    if len(episodes) != len(parent_manifest.entries):
        raise ValueError("aggregate episode count does not match the parent final cohort")

    first = batch_results[0]
    parameters = dict(first.get("parameters", {}))
    parameters.update(
        {
            "num_eval": len(parent_manifest.entries),
            "start_rows": [entry.row_index for entry in parent_manifest.entries],
            "episode_ids": [entry.episode_id for entry in parent_manifest.entries],
            "start_steps": [entry.start_step for entry in parent_manifest.entries],
            "cohort_id": parent_manifest.cohort_id,
            "cohort_sha256": parent_manifest.computed_sha256,
            "save_video": bool(save_video),
            "evaluation_batch_size": EVALUATION_BATCH_SIZE,
            "evaluation_batch_count": len(batch_results),
        }
    )
    base = {
        "task": TASK,
        "local_dataset": first.get("local_dataset"),
        "benchmark_dataset": first.get("benchmark_dataset"),
        "entrypoint": "round3_phase2_eval_curve",
        "policy_kind": EXPERIMENTS[experiment]["policy_kind"],
        "checkpoint": str(checkpoint),
        "epoch": None if experiment == "e0" else 10,
        "stage": None if experiment == "e0" else "stage_b",
        "parameters": parameters,
        "evaluation_seconds": float(sum(float(item.get("evaluation_seconds", 0.0)) for item in batch_results)),
        "success_rate": float(sum(bool(item.get("success")) for item in episodes) / len(episodes)),
        "episodes": episodes,
        "evaluation_batches": [
            {
                "index": int(index),
                "path": str(path),
                "cohort_sha256": result.get("cohort_sha256"),
                "success_rate": float(result.get("success_rate", 0.0)),
                "episodes": len(result.get("episodes", ())),
            }
            for index, (result, path) in enumerate(zip(batch_results, batch_paths))
        ],
    }
    trace_records = episodes if trace else None
    payload = enrich_result_payload(
        base,
        manifest=parent_manifest,
        protocol_variant=parent_manifest.protocol_variant,
        trace_records=trace_records,
    )
    payload["evaluation_batch_size"] = EVALUATION_BATCH_SIZE
    payload["evaluation_batch_count"] = len(batch_results)
    return write_round3_result(
        payload,
        output_dir,
        trace_records=trace_records,
        trace_output_dir=output_dir if trace else None,
    )


def run_curve(
    *,
    experiment: str,
    arm: str,
    output_root: Path,
    device: str,
    save_video: bool,
    trace: bool,
    task: str = "cube",
) -> dict[str, Any]:
    _configure_task(task)
    visible_gpus = _validate_gpu_visibility(device)
    # Bind MuJoCo/EGL before importing the dataset/evaluation stack.  CUDA
    # visibility alone is not sufficient when other jobs share the host.
    from source.common.gpu_environment import configure_mujoco_egl_device

    configure_mujoco_egl_device()
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unsupported experiment: {experiment}")
    if arm not in ARMS:
        raise ValueError(f"unsupported arm: {arm}")
    if save_video:
        raise ValueError(
            "save_video=True is intentionally disabled for the 84-point curve run; "
            "use a separate targeted evaluation for videos"
        )

    from omegaconf import OmegaConf

    from scripts.round3_phase1 import _load_dataset
    from source.common.checkpoint import load_policy_or_model
    from source.common.eval import EvaluationIdentity
    from source.common.round3_eval import run_round3_evaluation
    from source.common.round3_phase1 import CohortManifest

    manifest = CohortManifest.load(COHORT_PATH)
    if len(manifest.entries) != 200 or manifest.cohort_kind != "final":
        raise ValueError("the curve runner requires the frozen 200-episode final cohort")

    cfg, dataset = _load_dataset(TASK, None, None, len(manifest.entries))
    batch_cfg = OmegaConf.merge(
        cfg,
        {
            "eval": {"num_eval": EVALUATION_BATCH_SIZE},
            "solver": {"device": str(device)},
            "output": {"save_video": False},
            "world": {"num_envs": EVALUATION_BATCH_SIZE},
        },
    )

    series_dir = output_root / experiment / arm
    series_dir.mkdir(parents=True, exist_ok=True)
    for step in STEPS:
        checkpoint = _checkpoint_path(experiment, arm, step)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing checkpoint for {experiment}/{arm}/{step}: {checkpoint}")
        step_dir = series_dir / f"envsteps_{step:06d}"
        result_path = step_dir / "result.json"
        if _result_is_valid(result_path, manifest.computed_sha256, len(manifest.entries)):
            print(f"skip experiment={experiment} arm={arm} envsteps={step} existing=ok", flush=True)
            continue

        print(
            f"start experiment={experiment} arm={arm} envsteps={step} "
            f"checkpoint={checkpoint}",
            flush=True,
        )
        batch_results: list[dict[str, Any]] = []
        batch_paths: list[str] = []
        for batch_index, start in enumerate(range(0, len(manifest.entries), EVALUATION_BATCH_SIZE)):
            entries = tuple(manifest.entries[start : start + EVALUATION_BATCH_SIZE])
            if len(entries) != EVALUATION_BATCH_SIZE:
                raise AssertionError("final cohort size must be divisible by the evaluation batch size")
            batch_manifest = _make_batch_manifest(
                manifest,
                batch_index=batch_index,
                entries=entries,
            )
            batch_dir = step_dir / "batches" / f"batch_{batch_index:02d}"
            batch_result_path = batch_dir / "result.json"
            if _result_is_valid(
                batch_result_path,
                batch_manifest.computed_sha256,
                EVALUATION_BATCH_SIZE,
            ):
                batch_result = json.loads(batch_result_path.read_text(encoding="utf-8"))
                print(
                    f"skip experiment={experiment} arm={arm} envsteps={step} "
                    f"batch={batch_index} existing=ok",
                    flush=True,
                )
            else:
                if batch_dir.exists():
                    unexpected = [
                        path
                        for path in batch_dir.iterdir()
                        if path.name != "cohort.json"
                    ]
                    if unexpected:
                        raise FileExistsError(
                            f"refusing to overwrite incomplete curve batch: {batch_dir}"
                        )
                batch_dir.mkdir(parents=True, exist_ok=True)
                batch_manifest.save(batch_dir / "cohort.json")
                print(
                    f"start experiment={experiment} arm={arm} envsteps={step} "
                    f"batch={batch_index} episodes={EVALUATION_BATCH_SIZE}",
                    flush=True,
                )
                model, resolved_checkpoint = load_policy_or_model(str(checkpoint))
                identity = EvaluationIdentity(
                    entrypoint="round3_phase2_eval_curve",
                    policy_kind=EXPERIMENTS[experiment]["policy_kind"],
                    checkpoint=str(resolved_checkpoint or checkpoint),
                    epoch=None if experiment == "e0" else 10,
                    stage=None if experiment == "e0" else "stage_b",
                )
                batch_result = run_round3_evaluation(
                    batch_cfg,
                    task=TASK,
                    policy_or_model=model,
                    identity=identity,
                    manifest=batch_manifest,
                    dataset=dataset,
                    output_dir=batch_dir,
                    trace_output_dir=batch_dir / "trace" if trace else None,
                    device=device,
                    trace=trace,
                )
                print(
                    f"done experiment={experiment} arm={arm} envsteps={step} "
                    f"batch={batch_index} success_rate={float(batch_result['success_rate']):.4f}",
                    flush=True,
                )
                del model
                gc.collect()
                if str(device).startswith("cuda"):
                    import torch

                    torch.cuda.empty_cache()
            batch_results.append(batch_result)
            batch_paths.append(str(batch_result_path))

        result = _aggregate_batches(
            parent_manifest=manifest,
            batch_results=batch_results,
            batch_paths=batch_paths,
            output_dir=step_dir,
            experiment=experiment,
            arm=arm,
            checkpoint=checkpoint,
            save_video=False,
            trace=trace,
        )
        print(
            f"done experiment={experiment} arm={arm} envsteps={step} "
            f"success_rate={float(result['success_rate']):.4f}",
            flush=True,
        )
        _publish_curve(
            series_dir,
            experiment=experiment,
            arm=arm,
            cohort_sha256=manifest.computed_sha256,
            visible_gpus=visible_gpus,
        )

    rows = _publish_curve(
        series_dir,
        experiment=experiment,
        arm=arm,
        cohort_sha256=manifest.computed_sha256,
        visible_gpus=visible_gpus,
    )
    return {
        "status": "ok" if len(rows) == len(STEPS) else "running",
        "experiment": experiment,
        "arm": arm,
        "completed_points": len(rows),
        "total_points": len(STEPS),
        "output": str(series_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=tuple(EXPERIMENTS))
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("--task", choices=tuple(TASK_SPECS), default="cube")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--no-trace", dest="trace", action="store_false")
    parser.set_defaults(trace=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    _configure_task(args.task)
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else DEFAULT_OUTPUT_ROOT
    )

    if args.dry_run:
        visible = _validate_gpu_visibility(args.device) if str(args.device).startswith("cuda") else ()
        print(
            json.dumps(
                {
                    "experiment": args.experiment,
                    "arm": args.arm,
                    "task": args.task,
                    "steps": list(STEPS),
                    "checkpoints": {
                        str(step): str(_checkpoint_path(args.experiment, args.arm, step))
                        for step in STEPS
                    },
                    "visible_physical_gpus": list(visible),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    result = run_curve(
        experiment=args.experiment,
        arm=args.arm,
        output_root=output_root,
        device=args.device,
        save_video=bool(args.save_video),
        trace=bool(args.trace),
        task=args.task,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
