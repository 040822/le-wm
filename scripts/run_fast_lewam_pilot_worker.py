"""Run one isolated Fast-LeWAM pilot through training and Level evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

def _configure_isolated_gpu_environment():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"0", "1", "2", "3"}:
        raise RuntimeError(
            "pilot worker requires exactly one physical GPU from GPU0-GPU3"
        )
    os.environ["MUJOCO_EGL_DEVICE_ID"] = visible
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


_configure_isolated_gpu_environment()

import numpy as np
from omegaconf import OmegaConf, open_dict
from scipy.stats import spearmanr
import torch

from eval_fast_lewam import load_model_from_weights
from source.experiments.fast_lewam_parallel_pilots import (
    decide_level_one,
    file_sha256,
    load_pilot_manifest,
    paired_success_delta_pp,
    valid_summary_payload,
    validate_action_training_budget,
    validate_artifact_mapping,
)
from source.experiments.planner_transition_collection import (
    build_replay_identity,
    collect_action_token_grounding,
    collect_true_transition_replay,
)
from source.experiments.planner_transition_replay import (
    PlannerTransitionReplay,
    StageBReplayTrainer,
    save_replay,
    split_train_holdout,
)


def _require_level_one_summary(manifest, pilot):
    path = pilot.output_dir / "level_1" / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Level-1 summary missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    valid = valid_summary_payload(
        value, manifest, pilot, 1, require_promote=True
    )
    if not valid:
        raise RuntimeError(f"Level-1 summary or continuation artifacts are invalid: {path}")
    return value


def _arm_identity(manifest, pilot, level_number, level, action_tokens):
    identity = {
        "schema_version": 1,
        "manifest_sha256": manifest.sha256,
        "pilot": pilot.name,
        "level": int(level_number),
        "max_epochs": int(level.max_epochs),
        "action_tokens": bool(action_tokens),
    }
    if int(level_number) == 2:
        previous_name = "train" if action_tokens else "control_train"
        parent = pilot.output_dir / "level_1" / previous_name / "checkpoints" / "last.ckpt"
        if not parent.is_file():
            raise FileNotFoundError(
                f"Level-2 continuation checkpoint missing: {parent}"
            )
        identity["parent_last_ckpt"] = {
            "path": str(parent),
            "sha256": file_sha256(parent),
        }
    return identity


def _load_completed_arm(marker, expected_identity):
    if not marker.is_file():
        return False
    value = json.loads(marker.read_text(encoding="utf-8"))
    if value.get("identity") != expected_identity:
        raise RuntimeError(f"action arm identity mismatch: {marker}")
    if not validate_artifact_mapping(value.get("artifacts", {})):
        raise RuntimeError(f"action arm artifacts are invalid: {marker}")
    return True


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run(command, *, cwd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("COMMAND " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            env=os.environ.copy(),
        )


def _evaluation_result(run_dir: Path, epoch: int):
    path = run_dir / "eval" / f"epoch_{epoch}" / "stage_b" / "result.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _replay_groups(replay: PlannerTransitionReplay):
    for group in dict.fromkeys(replay.groups):
        indices = [index for index, value in enumerate(replay.groups) if value == group]
        yield PlannerTransitionReplay(
            states=replay.states[indices],
            actions=replay.actions[indices],
            targets=replay.targets[indices],
            sources=tuple(replay.sources[index] for index in indices),
            groups=tuple(replay.groups[index] for index in indices),
            identity={"parent": dict(replay.identity), "group": int(group)},
            goal_latents=replay.goal_latents[indices],
            physical_costs=replay.physical_costs[indices],
            successes=replay.successes[indices],
        )


@torch.no_grad()
def _grounded_metrics(model, replay, device):
    model = model.to(device).eval()
    rows = []
    for group in _replay_groups(replay):
        predicted = model.predict_training_latents(
            group.states.to(device), group.actions.to(device)
        )
        costs = (predicted[:, -1] - group.goal_latents.to(device)).square().mean(-1)
        costs = costs.detach().cpu().numpy()
        physical = group.physical_costs.numpy()
        success = group.successes.numpy().astype(bool)
        correlation = spearmanr(costs, physical).statistic
        selected = int(np.argmin(costs))
        k = min(30, len(costs))
        predicted_topk = np.argsort(costs)[:k]
        physical_topk = np.argsort(physical)[:k]
        successful = np.flatnonzero(success)
        rows.append(
            {
                "group": int(group.groups[0]),
                "candidate_count": len(costs),
                "spearman": None if not np.isfinite(correlation) else float(correlation),
                "regret": float(physical[selected] - physical.min()),
                "top1_success": bool(success[selected]),
                "oracle_success": bool(success.any()),
                "candidate_success_rate": float(success.mean()),
                "physical_top30_recall": float(
                    np.intersect1d(predicted_topk, physical_topk).size / k
                ),
                "successful_recall_at_30": (
                    None
                    if len(successful) == 0
                    else float(
                        np.intersect1d(predicted_topk, successful).size
                        / len(successful)
                    )
                ),
            }
        )
    finite_spearman = [row["spearman"] for row in rows if row["spearman"] is not None]
    finite_success_recall = [
        row["successful_recall_at_30"]
        for row in rows
        if row["successful_recall_at_30"] is not None
    ]
    return {
        "groups": rows,
        "spearman": float(np.mean(finite_spearman)) if finite_spearman else None,
        "regret": float(np.mean([row["regret"] for row in rows])),
        "top1_success": float(np.mean([row["top1_success"] for row in rows])),
        "oracle_success": float(np.mean([row["oracle_success"] for row in rows])),
        "candidate_success_rate": float(
            np.mean([row["candidate_success_rate"] for row in rows])
        ),
        "physical_top30_recall": float(
            np.mean([row["physical_top30_recall"] for row in rows])
        ),
        "successful_recall_at_30": (
            float(np.mean(finite_success_recall)) if finite_success_recall else None
        ),
    }


def _action_grounded_metrics(rows, cost_key):
    metrics = []
    for value in rows:
        costs = np.asarray(value[cost_key], dtype=np.float64)
        physical = np.asarray(value["physical"], dtype=np.float64)
        success = np.asarray(value["success"], dtype=bool)
        correlation = spearmanr(costs, physical).statistic
        selected = int(np.argmin(costs))
        k = min(30, len(costs))
        predicted_topk = np.argsort(costs)[:k]
        physical_topk = np.argsort(physical)[:k]
        successful = np.flatnonzero(success)
        metrics.append({
            "group": int(value["slot"]),
            "candidate_count": len(costs),
            "spearman": None if not np.isfinite(correlation) else float(correlation),
            "regret": float(physical[selected] - physical.min()),
            "top1_success": bool(success[selected]),
            "oracle_success": bool(success.any()),
            "candidate_success_rate": float(success.mean()),
            "physical_top30_recall": float(
                np.intersect1d(predicted_topk, physical_topk).size / k
            ),
            "successful_recall_at_30": (
                None if len(successful) == 0 else float(
                    np.intersect1d(predicted_topk, successful).size / len(successful)
                )
            ),
        })
    finite_spearman = [row["spearman"] for row in metrics if row["spearman"] is not None]
    finite_success_recall = [
        row["successful_recall_at_30"] for row in metrics
        if row["successful_recall_at_30"] is not None
    ]
    return {
        "groups": metrics,
        "spearman": float(np.mean(finite_spearman)) if finite_spearman else None,
        "regret": float(np.mean([row["regret"] for row in metrics])),
        "top1_success": float(np.mean([row["top1_success"] for row in metrics])),
        "oracle_success": float(np.mean([row["oracle_success"] for row in metrics])),
        "candidate_success_rate": float(np.mean([
            row["candidate_success_rate"] for row in metrics
        ])),
        "physical_top30_recall": float(np.mean([
            row["physical_top30_recall"] for row in metrics
        ])),
        "successful_recall_at_30": (
            float(np.mean(finite_success_recall)) if finite_success_recall else None
        ),
    }


def _training_transition_count(run_dir, batch_size):
    payload = torch.load(
        run_dir / "checkpoints" / "last.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    return int(payload["global_step"]) * int(batch_size)


def _save_adapted_run(pilot, level, trainer, output_dir, report):
    run_dir = output_dir / "train"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    weight_path = checkpoints / f"fast_lewam_weights_epoch_{level.max_epochs}.pt"
    temporary = weight_path.with_suffix(".pt.tmp")
    torch.save(trainer.model.state_dict(), temporary)
    temporary.replace(weight_path)
    state_path = run_dir / "adaptation_state.pt"
    state_temporary = state_path.with_suffix(".pt.tmp")
    torch.save(trainer.state_dict(), state_temporary)
    state_temporary.replace(state_path)
    cfg = OmegaConf.load(pilot.baseline_run_config)
    with open_dict(cfg):
        cfg.subdir = run_dir.name
        cfg.trainer.max_epochs = level.max_epochs
        cfg.trainer.default_root_dir = str(run_dir)
        cfg.pilot = report
    OmegaConf.save(cfg, run_dir / "config.yaml")
    return run_dir


def _eval_command(run_dir, *, task, epoch, num_eval, action_tokens):
    command = [
        "python",
        "-u",
        "eval_fast_lewam.py",
        str(run_dir / "checkpoints"),
        "--epochs",
        str(epoch),
        "--stages",
        "stage_b",
        f"eval.num_eval={num_eval}",
        f"world.num_envs={num_eval}",
        "output.save_video=false",
    ]
    if action_tokens:
        command.extend(
            (
                "plan_config.horizon=25",
                "plan_config.receding_horizon=25",
                "plan_config.action_block=1",
            )
        )
    return command


def _prepare_baseline_eval_run(pilot, output_dir):
    run_dir = output_dir / "baseline_eval"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pilot.baseline_run_config, run_dir / "config.yaml")
    target = checkpoints / pilot.baseline_checkpoint.name
    if target.is_symlink() and target.resolve() != pilot.baseline_checkpoint.resolve():
        raise RuntimeError(f"baseline checkpoint symlink mismatch: {target}")
    if not target.exists():
        target.symlink_to(pilot.baseline_checkpoint.resolve())
    return run_dir


def _train_action_arm(
    manifest, pilot, level_number, level, output_dir, *, action_tokens
):
    name = "train" if action_tokens else "control_train"
    run_dir = output_dir / name
    marker = output_dir / f"{name}_complete.json"
    arm_identity = _arm_identity(
        manifest, pilot, level_number, level, action_tokens
    )
    if _load_completed_arm(marker, arm_identity):
        return run_dir
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"incomplete action arm requires explicit archive before retry: {run_dir}"
        )
    baseline_cfg = OmegaConf.load(pilot.baseline_run_config)
    command = [
        "python", "-u", "train.py", "--config-name=fast_lewam",
        "data=cube", "train_mode=stage_b",
        "loader.batch_size=32", "trainer.devices=1",
        f"trainer.max_epochs={level.max_epochs}",
        "epoch_eval.enabled=false", "validation_diagnostics.enabled=false",
        "wandb.enabled=false", f"seed={int(baseline_cfg.seed)}",
        f"info=parallel_pilot_{'action_tokens_25' if action_tokens else 'control_5x10'}",
        f"hydra.run.dir={run_dir}",
    ]
    if action_tokens:
        command.extend((
            "action_horizon=25", "num_preds=25", "data.dataset.frameskip=1",
        ))
    if level_number == 2:
        previous_name = "train" if action_tokens else "control_train"
        checkpoint = pilot.output_dir / "level_1" / previous_name / "checkpoints" / "last.ckpt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Level-2 continuation checkpoint missing: {checkpoint}")
        command.append(f"resume_ckpt={checkpoint}")
    _run(command, cwd=manifest.repository_root, log_path=output_dir / f"{name}.log")
    _run(
        _eval_command(
            run_dir, task=pilot.task, epoch=level.max_epochs,
            num_eval=level.num_eval, action_tokens=action_tokens,
        ),
        cwd=manifest.repository_root,
        log_path=output_dir / f"{name}_eval.log",
    )
    artifact_paths = (
        run_dir / "config.yaml",
        run_dir / "checkpoints" / "last.ckpt",
        run_dir / "checkpoints" / f"fast_lewam_weights_epoch_{level.max_epochs}.pt",
        run_dir / "eval" / f"epoch_{level.max_epochs}" / "stage_b" / "result.json",
    )
    _atomic_json(marker, {
        "identity": arm_identity,
        "artifacts": {str(path): file_sha256(path) for path in artifact_paths},
    })
    return run_dir


def _run_action_tokens(manifest, pilot, level_number, level, output_dir, device):
    control_run = _train_action_arm(
        manifest, pilot, level_number, level, output_dir, action_tokens=False
    )
    variant_run = _train_action_arm(
        manifest, pilot, level_number, level, output_dir, action_tokens=True
    )
    control_model = load_model_from_weights(
        OmegaConf.load(control_run / "config.yaml"),
        control_run / "checkpoints" / f"fast_lewam_weights_epoch_{level.max_epochs}.pt",
        device,
    )
    variant_model = load_model_from_weights(
        OmegaConf.load(variant_run / "config.yaml"),
        variant_run / "checkpoints" / f"fast_lewam_weights_epoch_{level.max_epochs}.pt",
        device,
    )
    grounded = collect_action_token_grounding(
        manifest,
        pilot,
        level=level_number,
        control_model=control_model,
        variant_model=variant_model,
        device=device,
    )
    before = _action_grounded_metrics(grounded, "control_cost")
    after = _action_grounded_metrics(grounded, "variant_cost")
    control_examples = _training_transition_count(control_run, batch_size=32)
    variant_examples = _training_transition_count(variant_run, batch_size=32)
    control_cfg = OmegaConf.load(control_run / "config.yaml")
    variant_cfg = OmegaConf.load(variant_run / "config.yaml")
    control_span = int(control_cfg.action_horizon) * int(control_cfg.data.dataset.frameskip)
    variant_span = int(variant_cfg.action_horizon) * int(variant_cfg.data.dataset.frameskip)
    simulator_transition_count = sum(
        len(row["physical"]) * variant_span for row in grounded
    )
    training_transition_updates = variant_examples * variant_span
    budget = validate_action_training_budget(
        control_examples, variant_examples, control_span, variant_span
    )
    return (
        variant_run, control_run, before, after,
        simulator_transition_count, training_transition_updates, budget,
    )


def _run_replay(manifest, pilot, level_number, level, output_dir, device):
    run_cfg = OmegaConf.load(pilot.baseline_run_config)
    model = load_model_from_weights(run_cfg, pilot.baseline_checkpoint, device)
    replay_cfg = manifest.replay
    trainer = StageBReplayTrainer(
        model,
        learning_rate=float(replay_cfg["learning_rate"]),
        weight_decay=float(replay_cfg["weight_decay"]),
        expert_fraction=float(replay_cfg["expert_fraction"]),
        seed=manifest.seed,
        device=device,
    )
    if level_number == 2:
        state_path = pilot.output_dir / "level_1" / "train" / "adaptation_state.pt"
        summary_path = pilot.output_dir / "level_1" / "summary.json"
        if not state_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError("Level-2 requires a completed Level-1 adaptation")
        level_one_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if level_one_summary.get("decision") != "promote":
            raise RuntimeError("Level-2 launch rejected: Level-1 was not promoted")
        trainer.load_state_dict(
            torch.load(state_path, map_location="cpu", weights_only=False)
        )
    level_one_slots = manifest.levels[1].grounded_slots
    epoch_delta = level.max_epochs - (manifest.levels[1].max_epochs if level_number == 2 else 0)
    callback_index = {"value": 0}

    def update(group):
        group_number = callback_index["value"]
        callback_index["value"] += 1
        epochs = (
            epoch_delta
            if level_number == 1 or group_number < level_one_slots
            else level.max_epochs
        )
        train_group, _ = split_train_holdout(group)
        trainer.train(
            train_group,
            epochs=epochs,
            batch_size=int(replay_cfg["train_batch_size"]),
        )

    callback = update if pilot.direction == "streaming_online" else None
    replay = collect_true_transition_replay(
        manifest,
        pilot,
        level=level_number,
        device=device,
        on_group=callback,
        ranking_model=(trainer.model if pilot.direction == "streaming_online" else None),
        cache_dir=pilot.output_dir / "replay_groups",
    )
    save_replay(output_dir / "replay.pt", replay)
    train_replay, holdout_replay = split_train_holdout(replay)
    baseline_model = load_model_from_weights(run_cfg, pilot.baseline_checkpoint, device)
    grounded_before = _grounded_metrics(baseline_model, holdout_replay, device)
    if pilot.direction == "offline_alignment":
        for group_number, group in enumerate(_replay_groups(replay)):
            epochs = (
                epoch_delta
                if level_number == 1 or group_number < level_one_slots
                else level.max_epochs
            )
            train_group, _ = split_train_holdout(group)
            trainer.train(
                train_group,
                epochs=epochs,
                batch_size=int(replay_cfg["train_batch_size"]),
            )
    train_report = trainer.report()
    grounded_after = _grounded_metrics(trainer.model, holdout_replay, device)
    run_dir = _save_adapted_run(
        pilot,
        level,
        trainer,
        output_dir,
        {
            "direction": pilot.direction,
            "replay_identity": build_replay_identity(manifest, pilot, level_number),
            "train_report": train_report,
            "grounded_split": "alternating_planner_holdout",
        },
    )
    baseline_run = _prepare_baseline_eval_run(pilot, output_dir)
    for eval_run, epoch, log_name in (
        (baseline_run, 10, "baseline_eval.log"),
        (run_dir, level.max_epochs, "eval.log"),
    ):
        _run(
            _eval_command(
                eval_run, task=pilot.task, epoch=epoch,
                num_eval=level.num_eval, action_tokens=False,
            ),
            cwd=manifest.repository_root,
            log_path=output_dir / log_name,
        )
    action_block = int(run_cfg.data.dataset.frameskip)
    simulator_transition_count = int(
        replay.actions.shape[0] * replay.actions.shape[1] * action_block
    )
    training_transition_updates = int(
        sum(trainer.source_updates.values()) * replay.actions.shape[1]
    )
    return (
        run_dir, baseline_run, grounded_before, grounded_after,
        simulator_transition_count, training_transition_updates,
        {
            "collected_candidates": int(replay.actions.shape[0]),
            "action_block": action_block,
            "optimizer_steps": trainer.optimizer_steps,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pilot", required=True)
    parser.add_argument("--level", required=True, type=int, choices=(1, 2))
    parser.add_argument("--direction", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args(argv)
    if args.device != "cuda:0":
        raise ValueError("isolated pilot workers must use logical cuda:0")
    manifest = load_pilot_manifest(args.manifest)
    pilot = manifest.pilots[args.pilot]
    if args.direction != pilot.direction:
        raise ValueError("CLI direction differs from the manifest")
    visible = int(os.environ["CUDA_VISIBLE_DEVICES"])
    if visible != pilot.gpu:
        raise ValueError(
            f"pilot {pilot.name} requires physical GPU{pilot.gpu}, got GPU{visible}"
        )
    level = manifest.levels[args.level]
    output_dir = pilot.output_dir / f"level_{args.level}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.level == 2:
        _require_level_one_summary(manifest, pilot)
    existing_summary = output_dir / "summary.json"
    if existing_summary.is_file():
        raise RuntimeError(
            f"pilot output already contains a summary; use status or archive it: {existing_summary}"
        )
    started = time.time()
    if pilot.direction == "action_tokens_25":
        (
            run_dir, baseline_run, grounded_before, grounded_after,
            simulator_transition_count, training_transition_updates,
            budget_report,
        ) = _run_action_tokens(
            manifest, pilot, args.level, level, output_dir, args.device
        )
        baseline_epoch = level.max_epochs
    else:
        (
            run_dir, baseline_run, grounded_before, grounded_after,
            simulator_transition_count, training_transition_updates,
            budget_report,
        ) = _run_replay(
            manifest, pilot, args.level, level, output_dir, args.device
        )
        baseline_epoch = 10
    result = _evaluation_result(run_dir, level.max_epochs)
    baseline_result = _evaluation_result(baseline_run, baseline_epoch)
    baseline, success, paired_delta = paired_success_delta_pp(
        baseline_result, result
    )
    artifact_paths = (
        run_dir / "config.yaml",
        run_dir / "checkpoints" / f"fast_lewam_weights_epoch_{level.max_epochs}.pt",
        run_dir / "eval" / f"epoch_{level.max_epochs}" / "stage_b" / "result.json",
        baseline_run / "config.yaml",
        baseline_run / "checkpoints" / f"fast_lewam_weights_epoch_{baseline_epoch}.pt",
        baseline_run / "eval" / f"epoch_{baseline_epoch}" / "stage_b" / "result.json",
    )
    if pilot.direction == "action_tokens_25":
        artifact_paths = artifact_paths + (
            run_dir / "checkpoints" / "last.ckpt",
            baseline_run / "checkpoints" / "last.ckpt",
        )
    else:
        artifact_paths = artifact_paths + (run_dir / "adaptation_state.pt",)
    artifact_identity = {
        str(path): file_sha256(path) for path in artifact_paths
    }
    if grounded_before is None:
        spearman_delta = 0.0
        regret_delta = 0.0
    else:
        before_spearman = grounded_before["spearman"]
        after_spearman = grounded_after["spearman"]
        spearman_delta = (
            0.0
            if before_spearman is None or after_spearman is None
            else after_spearman - before_spearman
        )
        denominator = max(abs(grounded_before["regret"]), 1e-8)
        regret_delta = (grounded_after["regret"] - grounded_before["regret"]) / denominator
    deltas = {
        "success_delta_pp": paired_delta,
        "spearman_delta": float(spearman_delta),
        "regret_relative_delta": float(regret_delta),
    }
    decision = decide_level_one(deltas, manifest.promotion) if args.level == 1 else "complete"
    summary = {
        "schema_version": 2,
        "status": "ok",
        "pilot": pilot.name,
        "direction": pilot.direction,
        "task": pilot.task,
        "level": args.level,
        "manifest_sha256": manifest.sha256,
        "artifact_identity": artifact_identity,
        "elapsed_seconds": time.time() - started,
        "gpu_hours": (time.time() - started) / 3600.0,
        "transition_count": simulator_transition_count,
        "simulator_transition_count": simulator_transition_count,
        "training_transition_updates": training_transition_updates,
        "budget": budget_report,
        "baseline_success_rate": baseline,
        "pilot_success_rate": success,
        "grounded_before": grounded_before,
        "grounded_after": grounded_after,
        "deltas": deltas,
        "forgetting_pp": max(0.0, -deltas["success_delta_pp"]),
        "success_gain_pp_per_1k_simulator_transitions": (
            None if simulator_transition_count == 0
            else deltas["success_delta_pp"]
            / (simulator_transition_count / 1000.0)
        ),
        "decision": decision,
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
