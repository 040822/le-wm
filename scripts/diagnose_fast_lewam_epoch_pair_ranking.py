"""Trace, ground, and summarize Fast-LeWAM epoch-pair CEM ranking."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from omegaconf import OmegaConf
import stable_worldmodel as swm
import torch
from stable_worldmodel.world.world import _extract_init_goal

from eval_fast_lewam import load_model_from_weights
from scripts.ground_stage_b_episode_failures import (
    _predict_costs,
    _predict_terminal_latents,
    _terminal_latents,
)
from source.common.eval import (
    DatasetEvaluationSession,
    EvaluationIdentity,
    compose_eval_config,
    fit_eval_processors,
    get_dataset,
    img_transform,
)
from source.diagnostics.stage_b_episode_failures import (
    CEMTraceCollector,
    FixedRawSequencePolicy,
    TracedCEMSolver,
    TracedPolicy,
    build_prefix_replay_actions,
    normalize_expert_plan,
    prepare_image_info,
)
from source.diagnostics.stage_b_epoch_pair_ranking import (
    CandidatePanel,
    build_cross_epoch_panel,
    build_full_cross_epoch_panel,
    build_ground_identity,
    build_pair_identity,
    build_summary_identity,
    candidate_actions_sha256,
    cem_candidate_indices,
    classify_epoch_pair_failure,
    compute_pair_physical_terminal_cost,
    compute_ranking_metrics,
    cross_score_costs,
    grounded_pair_acceptance_errors,
    load_epoch_pair_manifest,
    planning_state_sha256,
    require_identity,
    require_matching_cache_identity,
    summarize_grounded_pair,
    tensor_bytes_sha256,
    validate_pair_device,
    write_atomic_json,
    write_atomic_text,
    write_atomic_torch,
    write_pair_trace_artifacts,
)


EPOCH_LABELS = ("e8", "e10")


class PairSimulatorGrounder:
    """Execute one shared panel and retain model-neutral terminal images."""

    def __init__(self, *, cfg, dataset, scaler, device):
        self.cfg = cfg
        self.dataset = dataset
        self.scaler = scaler
        self.device = device
        self.transform = {
            "pixels": img_transform(cfg),
            "goal": img_transform(cfg),
        }

    def run(self, *, task, episode, start, candidates, prefix_raw):
        raw = build_prefix_replay_actions(
            candidates,
            scaler=self.scaler,
            action_block=int(self.cfg.plan_config.action_block),
            prefix_raw=prefix_raw,
        )
        policy = FixedRawSequencePolicy(raw)
        world_cfg = OmegaConf.to_container(self.cfg.world, resolve=True)
        world_cfg["num_envs"] = len(raw)
        world_cfg["max_episode_steps"] = 2 * int(self.cfg.eval.eval_budget)
        world = swm.World(**world_cfg, image_shape=(224, 224))
        world.set_policy(policy)
        try:
            metrics = world.evaluate(
                dataset=self.dataset,
                episodes_idx=[episode] * len(raw),
                start_steps=[start] * len(raw),
                goal_offset=int(self.cfg.eval.goal_offset_steps),
                eval_budget=int(raw.shape[1]),
                callables=OmegaConf.to_container(
                    self.cfg.eval.callables, resolve=True
                ),
                reset_mode="wait",
            )
            physical = compute_pair_physical_terminal_cost(task, world.infos)
            terminal = prepare_image_info(
                {
                    "pixels": world.infos["pixels"],
                    "goal": world.infos["goal"],
                },
                self.transform,
                self.device,
            )
            success = np.asarray(
                metrics["episode_successes"], dtype=bool
            ).reshape(-1)
        finally:
            world.close()
        return {
            "physical_cost": physical,
            "terminal_pixels": terminal["pixels"].detach().cpu(),
            "terminal_goal": terminal["goal"].detach().cpu(),
            "success": success,
        }


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _subject_model(subject: Any, device: str) -> Any:
    return getattr(subject, "model", subject).to(device).eval()


def _load_subject(pair: Any, epoch_label: str, device: str) -> Any:
    run_cfg = OmegaConf.load(pair.run_config)
    return load_model_from_weights(
        run_cfg, pair.epochs[epoch_label].checkpoint, device
    )


def _trace_maps(trace: dict[str, Any]):
    plans = {
        (int(row["slot"]), int(row["replan"])): row
        for row in trace["plans"]
    }
    details = {
        (
            int(row["slot"]),
            int(row["replan"]),
            int(row["iteration"]),
        ): row
        for row in trace["details"]
    }
    return plans, details


@contextmanager
def _artifact_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def run_trace(*, manifest_path: str | Path, pair_label: str, device: str):
    manifest = load_epoch_pair_manifest(manifest_path)
    pair = manifest.pairs[pair_label]
    validate_pair_device(pair, device)
    pair_root = manifest.output_root / pair.label
    results = {}
    for epoch_label in EPOCH_LABELS:
        epoch = pair.epochs[epoch_label]
        output_dir = pair_root / "trace" / epoch_label
        reproduction_path = output_dir / "reproduction.json"
        identity = build_pair_identity(
            manifest, pair, phase="trace", epoch_label=epoch_label
        )
        if require_matching_cache_identity(reproduction_path, identity):
            cached = _read_json(reproduction_path)
            trace_path = output_dir / "trace.pt"
            if not trace_path.is_file():
                raise RuntimeError(f"trace cache is incomplete: {trace_path}")
            trace = torch.load(trace_path, map_location="cpu", weights_only=False)
            require_identity(trace.get("identity", {}), identity, source=trace_path)
            results[epoch_label] = cached
            print(f"{pair.label}/{epoch_label}: reuse trace", flush=True)
            continue

        reference = _read_json(epoch.reference_result)
        expected = [bool(row["success"]) for row in reference["episodes"]]
        cfg = compose_eval_config(
            pair.task,
            (f"solver.device={device}", "output.save_video=false"),
        )
        session = DatasetEvaluationSession(cfg, task=pair.task)
        subject = _load_subject(pair, epoch_label, device)
        evaluation_identity = EvaluationIdentity(
            entrypoint="diagnose_fast_lewam_epoch_pair_ranking",
            policy_kind="fast_lewam",
            checkpoint=str(epoch.checkpoint),
            epoch=epoch.epoch,
            stage="stage_b",
        )
        base_policy = session._build_policy(
            subject, evaluation_identity, str(device)
        )
        collector = CEMTraceCollector(
            detailed_steps={
                int(step) for step in manifest.protocol["detailed_iterations"]
            }
        )
        base_policy.solver = TracedCEMSolver(base_policy.solver, collector)
        traced_policy = TracedPolicy(base_policy, collector)
        evaluation = session.evaluate(
            traced_policy,
            identity=evaluation_identity,
            output_dir=output_dir / "evaluation",
            device=str(device),
        )
        actual = [bool(row.success) for row in evaluation.episodes]
        results[epoch_label] = write_pair_trace_artifacts(
            output_dir,
            collector=collector,
            selected_slots=pair.slots,
            actual_successes=actual,
            expected_successes=expected,
            identity=identity,
            protocol=manifest.protocol,
        )
        print(
            f"{pair.label}/{epoch_label}: "
            f"status={results[epoch_label]['status']} "
            f"mismatches={results[epoch_label]['mismatch_slots']}",
            flush=True,
        )
        del traced_policy, base_policy, subject
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pair_identity = build_pair_identity(manifest, pair, phase="trace")
    result = {
        "status": (
            "reproducible"
            if all(row["status"] == "reproducible" for row in results.values())
            else "non_reproducible"
        ),
        "pair": pair.label,
        "identity": pair_identity,
        "epochs": {
            label: {
                "status": row["status"],
                "mismatch_slots": row["mismatch_slots"],
                "success_vector": row["success_vector"],
            }
            for label, row in results.items()
        },
    }
    write_atomic_json(pair_root / "trace" / "result.json", result)
    return result


def _ground_cache(
    *,
    path: Path,
    identity: dict[str, Any],
    grounder: PairSimulatorGrounder,
    task: str,
    episode: int,
    start: int,
    panel: CandidatePanel,
    prefix: np.ndarray,
) -> dict[str, Any]:
    with _artifact_lock(path):
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            require_identity(payload.get("identity", {}), identity, source=path)
            if candidate_actions_sha256(payload["actions"]) != panel.sha256:
                raise RuntimeError(f"cached candidate bytes differ: {path}")
            return payload
        grounded = grounder.run(
            task=task,
            episode=episode,
            start=start,
            candidates=panel.actions,
            prefix_raw=prefix if len(prefix) else None,
        )
        payload = {
            "identity": identity,
            "actions": panel.actions,
            "sources": panel.sources,
            "source_aliases": panel.source_aliases,
            "physical_cost": np.asarray(grounded["physical_cost"]),
            "success": np.asarray(grounded["success"], dtype=bool),
            "terminal_pixels": grounded["terminal_pixels"],
            "terminal_goal": grounded["terminal_goal"],
        }
        write_atomic_torch(path, payload)
        return payload


def _source_outcomes(
    source_aliases: tuple[tuple[str, ...], ...],
    physical: np.ndarray,
    success: np.ndarray,
) -> dict[str, dict[str, Any]]:
    wanted = ("elite_mean", "iteration_best", "best_ever")
    return {
        source: {
            "physical_cost": float(physical[index]),
            "success": bool(success[index]),
        }
        for index, aliases in enumerate(source_aliases)
        for source in aliases
        if source.endswith(wanted)
    }


def _score_and_summarize_panel(
    *,
    models: dict[str, Any],
    planning_info: dict[str, Any],
    device: str,
    panel: CandidatePanel,
    grounded: dict[str, Any],
    score_path: Path,
    score_identity: dict[str, Any],
    topk: int,
) -> dict[str, Any]:
    if score_path.is_file():
        cached_scores = torch.load(
            score_path, map_location="cpu", weights_only=False
        )
        require_identity(
            cached_scores.get("identity", {}),
            score_identity,
            source=score_path,
        )
    predicted_costs = cross_score_costs(
        models,
        planning_info,
        panel.actions,
        scorer=lambda model, info, actions: _predict_costs(
            model, info, actions, device
        ),
    )
    physical = np.asarray(grounded["physical_cost"], dtype=np.float64)
    success = np.asarray(grounded["success"], dtype=bool)
    cem_indices = cem_candidate_indices(panel)
    model_payload = {}
    model_results = {}
    for epoch_label, model in models.items():
        predicted_terminal = _predict_terminal_latents(
            model, planning_info, panel.actions, device
        )
        true_terminal, latent_cost = _terminal_latents(
            model,
            {
                "pixels": grounded["terminal_pixels"].to(device),
                "goal": grounded["terminal_goal"].to(device),
            },
        )
        metrics = compute_ranking_metrics(
            predicted_cost=predicted_costs[epoch_label],
            true_terminal_latent_cost=latent_cost,
            physical_cost=physical,
            successes=success,
            predicted_terminal_latents=predicted_terminal,
            true_terminal_latents=true_terminal,
            topk=topk,
        )
        elite_by_source = {}
        for source_epoch in EPOCH_LABELS:
            indices = [
                index
                for index, aliases in enumerate(panel.source_aliases)
                if any(
                    source.startswith(f"{source_epoch}:elite_")
                    for source in aliases
                )
            ]
            if indices:
                elite_by_source[source_epoch] = {
                    "predicted_cost_mean": float(
                        np.mean(predicted_costs[epoch_label][indices])
                    ),
                    "physical_cost_mean": float(np.mean(physical[indices])),
                }
        model_results[epoch_label] = {
            "metrics": metrics,
            "elite_by_source_epoch": elite_by_source,
        }
        model_payload[epoch_label] = {
            "predicted_cost": predicted_costs[epoch_label],
            "true_terminal_latent_cost": latent_cost,
            "predicted_terminal_latent": predicted_terminal,
            "true_terminal_latent": true_terminal,
        }
    write_atomic_torch(
        score_path,
        {
            "identity": score_identity,
            "candidate_sha256": panel.sha256,
            "actions": panel.actions,
            "sources": panel.sources,
            "source_aliases": panel.source_aliases,
            "physical_cost": physical,
            "success": success,
            "models": model_payload,
        },
    )
    return {
        "candidate_count": len(panel.actions),
        "candidate_sha256": panel.sha256,
        "candidate_successes": success.tolist(),
        "cem_candidate_successes": success[list(cem_indices)].tolist(),
        "physical_cost_min": float(physical.min()),
        "selected_true_outcomes": _source_outcomes(
            panel.source_aliases, physical, success
        ),
        "models": model_results,
        "scores_file": str(score_path),
    }


def _expansion_reason(panel_result: dict[str, Any]) -> str | None:
    if not any(panel_result["cem_candidate_successes"]):
        return "no_success_in_sampled_panel"
    for model in panel_result["models"].values():
        metrics = model["metrics"]
        correlations = (
            metrics["predicted_true_terminal_latent_spearman"],
            metrics["true_terminal_latent_physical_spearman"],
        )
        if any(value is None or abs(value) < 0.2 for value in correlations):
            return "ambiguous_error_decomposition"
    return None


def _evaluate_ground_panel(
    *,
    base_identity: dict[str, Any],
    cache_root: Path,
    score_root: Path,
    grounder: PairSimulatorGrounder,
    models: dict[str, Any],
    planning_info: dict[str, Any],
    task: str,
    episode: int,
    start: int,
    prefix: np.ndarray,
    panel: CandidatePanel,
    cache_name: str,
    device: str,
    topk: int,
) -> dict[str, Any]:
    prefix_sha = tensor_bytes_sha256(prefix)
    identity = {
        **base_identity,
        "panel_prefix_sha256": prefix_sha,
        "panel_candidate_sha256": panel.sha256,
        "cache_name": cache_name,
    }
    cache_path = cache_root / f"{cache_name}.pt"
    grounded = _ground_cache(
        path=cache_path,
        identity=identity,
        grounder=grounder,
        task=task,
        episode=episode,
        start=start,
        panel=panel,
        prefix=prefix,
    )
    return _score_and_summarize_panel(
        models=models,
        planning_info=planning_info,
        device=device,
        panel=panel,
        grounded=grounded,
        score_path=score_root / f"{cache_name}.pt",
        score_identity=identity,
        topk=topk,
    )


def _initial_physical_distance(
    *, task: str, dataset: Any, episode: int, start: int, goal_offset: int
) -> float:
    initial, goal, _ = _extract_init_goal(
        dataset, [episode], [start], int(goal_offset)
    )
    if task == "pusht":
        left, right = initial["state"], goal["goal_state"]
    elif task == "reacher":
        left, right = initial["qpos"], goal["goal_qpos"]
    elif task == "cube":
        left = initial["privileged_block_0_pos"]
        right = goal["goal_privileged_block_0_pos"]
    else:
        raise ValueError(f"unsupported diagnostic task {task!r}")
    return float(
        np.linalg.norm(
            np.asarray(left).reshape(1, -1)
            - np.asarray(right).reshape(1, -1)
        )
    )


def _attributions_for_slot(
    *, slot_result: dict[str, Any], final_success: dict[str, bool]
) -> tuple[dict[str, list[str]], dict[str, list[float] | None]]:
    attributions = {}
    replan_distances = {}
    panels = slot_result["panels"]
    for epoch_label in EPOCH_LABELS:
        if final_success[epoch_label]:
            attributions[epoch_label] = []
            replan_distances[epoch_label] = None
            continue
        primary: set[str] = set()
        relevant = [
            panel
            for panel in panels
            if panel["state_epoch"] in {"shared", epoch_label}
        ]
        for panel in relevant:
            metrics = panel["models"][epoch_label]["metrics"]
            labels = classify_epoch_pair_failure(
                {
                    "all_candidates_grounded": bool(
                        panel["expanded_to_all_candidates"]
                    ),
                    "candidate_successes": panel[
                        "cem_candidate_successes"
                    ],
                    "selected_success": metrics["top1_success"],
                    "predicted_true_terminal_latent_spearman": metrics[
                        "predicted_true_terminal_latent_spearman"
                    ],
                    "true_terminal_latent_physical_spearman": metrics[
                        "true_terminal_latent_physical_spearman"
                    ],
                }
            )
            primary.update(label for label in labels if label != "mixed_or_undetermined")
        for replan, state_epoch in ((0, "shared"), (1, epoch_label)):
            curve = sorted(
                [
                    panel
                    for panel in relevant
                    if panel["replan"] == replan
                    and panel["state_epoch"] == state_epoch
                ],
                key=lambda row: row["iteration"],
            )
            if [row["iteration"] for row in curve] != [0, 5, 29]:
                continue
            predicted_curve = []
            physical_curve = []
            for row in curve:
                elite = row["sampled_elite_by_model"][epoch_label].get(
                    epoch_label
                )
                if elite is None:
                    break
                predicted_curve.append(elite["predicted_cost_mean"])
                physical_curve.append(elite["physical_cost_mean"])
            if len(predicted_curve) == 3:
                final_mean = curve[-1]["sampled_selected_true_outcomes"].get(
                    f"{epoch_label}:elite_mean"
                )
                labels = classify_epoch_pair_failure(
                    {
                        "predicted_elite_cost_0_5_29": predicted_curve,
                        "physical_elite_cost_0_5_29": physical_curve,
                        "earlier_best_physical_cost": min(
                            row["sampled_physical_cost_min"] for row in curve
                        ),
                        "final_mean_physical_cost": (
                            final_mean["physical_cost"] if final_mean else None
                        ),
                    }
                )
                primary.update(
                    label for label in labels if label != "mixed_or_undetermined"
                )
        first = next(
            (
                panel
                for panel in panels
                if panel["replan"] == 0
                and panel["iteration"] == 29
                and panel["state_epoch"] == "shared"
            ),
            None,
        )
        second = next(
            (
                panel
                for panel in panels
                if panel["replan"] == 1
                and panel["iteration"] == 29
                and panel["state_epoch"] == epoch_label
            ),
            None,
        )
        if first is not None and second is not None:
            first_outcome = first["sampled_selected_true_outcomes"].get(
                f"{epoch_label}:elite_mean"
            )
            second_outcome = second["sampled_selected_true_outcomes"].get(
                f"{epoch_label}:elite_mean"
            )
            if first_outcome and second_outcome:
                distances = [
                    slot_result["initial_physical_distance"],
                    first_outcome["physical_cost"],
                    second_outcome["physical_cost"],
                ]
                replan_distances[epoch_label] = distances
                labels = classify_epoch_pair_failure(
                    {"physical_distance_start_after_first_final": distances}
                )
                primary.update(
                    label for label in labels if label != "mixed_or_undetermined"
                )
            else:
                replan_distances[epoch_label] = None
        else:
            replan_distances[epoch_label] = None
        labels = sorted(primary)
        if len(labels) != 1:
            labels.append("mixed_or_undetermined")
        attributions[epoch_label] = labels
    return attributions, replan_distances


def _ground_artifact_errors(
    *, pair_root: Path, pair: Any, result: dict[str, Any], iterations: Any
) -> list[str]:
    errors = grounded_pair_acceptance_errors(
        pair, result, detailed_iterations=iterations
    )
    score_root = (pair_root / "ground" / "scores").resolve()
    for slot in result.get("slots", []):
        for panel in slot.get("panels", []):
            label = (
                f"slot {slot.get('slot')}/replan {panel.get('replan')}/"
                f"state {panel.get('state_epoch')}/iteration "
                f"{panel.get('iteration')}"
            )
            raw_path = panel.get("scores_file")
            if not raw_path:
                errors.append(f"{label} has no score artifact")
                continue
            path = Path(raw_path)
            try:
                path.resolve().relative_to(score_root)
            except ValueError:
                errors.append(f"{label} score artifact is outside pair output")
                continue
            if not path.is_file():
                errors.append(f"{label} score artifact is missing")
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            candidate_hash = panel.get("candidate_sha256")
            if payload.get("candidate_sha256") != candidate_hash:
                errors.append(f"{label} score candidate hash mismatch")
            if (
                candidate_actions_sha256(payload.get("actions"))
                != candidate_hash
            ):
                errors.append(f"{label} score action bytes mismatch")
            identity = payload.get("identity", {})
            for key, value in result.get("identity", {}).items():
                if identity.get(key) != value:
                    errors.append(f"{label} score identity mismatch for {key}")
            if identity.get("panel_prefix_sha256") != panel.get(
                "prefix_sha256"
            ):
                errors.append(f"{label} score prefix hash mismatch")
            if identity.get("panel_candidate_sha256") != candidate_hash:
                errors.append(f"{label} score identity candidate hash mismatch")
            count = int(panel.get("candidate_count", 0))
            model_payloads = payload.get("models", {})
            if set(model_payloads) != set(EPOCH_LABELS):
                errors.append(f"{label} score artifact lacks both models")
                continue
            for epoch_label in EPOCH_LABELS:
                for key in (
                    "predicted_cost",
                    "true_terminal_latent_cost",
                    "predicted_terminal_latent",
                    "true_terminal_latent",
                ):
                    values = torch.as_tensor(
                        model_payloads[epoch_label].get(key)
                    )
                    if len(values) != count or not torch.isfinite(values).all():
                        errors.append(
                            f"{label}/{epoch_label} invalid score tensor {key}"
                        )
    return errors


def run_ground(*, manifest_path: str | Path, pair_label: str, device: str):
    manifest = load_epoch_pair_manifest(manifest_path)
    pair = manifest.pairs[pair_label]
    validate_pair_device(pair, device)
    pair_root = manifest.output_root / pair.label
    trace_index = _read_json(pair_root / "trace" / "result.json")
    if trace_index["status"] != "reproducible":
        raise RuntimeError(
            f"{pair.label} trace is {trace_index['status']}; grounding prohibited"
        )
    traces = {}
    reproductions = {}
    for epoch_label in EPOCH_LABELS:
        trace_dir = pair_root / "trace" / epoch_label
        reproduction = _read_json(trace_dir / "reproduction.json")
        expected_identity = build_pair_identity(
            manifest, pair, phase="trace", epoch_label=epoch_label
        )
        require_identity(
            reproduction.get("identity", {}),
            expected_identity,
            source=trace_dir / "reproduction.json",
        )
        if reproduction["status"] != "reproducible":
            raise RuntimeError(f"{pair.label}/{epoch_label} has reproduction mismatch")
        trace = torch.load(
            trace_dir / "trace.pt", map_location="cpu", weights_only=False
        )
        require_identity(
            trace.get("identity", {}),
            expected_identity,
            source=trace_dir / "trace.pt",
        )
        traces[epoch_label] = trace
        reproductions[epoch_label] = reproduction

    base_identity = build_ground_identity(manifest, pair, traces)
    result_path = pair_root / "ground" / "result.json"
    if require_matching_cache_identity(result_path, base_identity):
        cached = _read_json(result_path)
        cache_errors = _ground_artifact_errors(
            pair_root=pair_root,
            pair=pair,
            result=cached,
            iterations=manifest.protocol["detailed_iterations"],
        )
        if cached.get("status") != "ok" or cache_errors:
            raise RuntimeError(
                f"cached ground result failed acceptance: {cache_errors}"
            )
        print(f"{pair.label}: reuse ground result", flush=True)
        return cached

    cfg = compose_eval_config(
        pair.task,
        (f"solver.device={device}", "output.save_video=false"),
    )
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    scaler = fit_eval_processors(dataset, ["action"])["action"]
    subjects = {
        label: _load_subject(pair, label, device) for label in EPOCH_LABELS
    }
    models = {
        label: _subject_model(subjects[label], device) for label in EPOCH_LABELS
    }
    grounder = PairSimulatorGrounder(
        cfg=cfg,
        dataset=dataset,
        scaler=scaler,
        device=device,
    )
    maps = {label: _trace_maps(traces[label]) for label in EPOCH_LABELS}
    reference = _read_json(pair.epochs["e8"].reference_result)
    episode_ids = reference["parameters"]["episode_ids"]
    start_steps = reference["parameters"]["start_steps"]
    result_slots = []
    for slot in pair.slots:
        episode = int(episode_ids[slot])
        start = int(start_steps[slot])
        panels = []
        skipped_states = []
        plans = {label: maps[label][0] for label in EPOCH_LABELS}
        details = {label: maps[label][1] for label in EPOCH_LABELS}
        state_specs = []
        terminal_before_replan_1 = []
        has_shared_replan = all(
            (slot, 0) in plans[label] for label in EPOCH_LABELS
        ) and all(
            (slot, 0, int(iteration)) in details[label]
            for label in EPOCH_LABELS
            for iteration in manifest.protocol["detailed_iterations"]
        )
        if has_shared_replan:
            hashes = {
                label: planning_state_sha256(
                    plans[label][(slot, 0)]["planning_info"]
                )
                for label in EPOCH_LABELS
            }
            prefixes = {
                label: tensor_bytes_sha256(
                    np.asarray(plans[label][(slot, 0)]["executed_prefix_before"])
                )
                for label in EPOCH_LABELS
            }
            if len(set(hashes.values())) != 1 or len(set(prefixes.values())) != 1:
                raise RuntimeError(
                    f"slot {slot} replan 0 is not a shared simulator state"
                )
            state_specs.append(
                ("shared", 0, plans["e8"][(slot, 0)], EPOCH_LABELS)
            )
        else:
            skipped_states.append("missing_shared_replan_0")
        for state_epoch in EPOCH_LABELS:
            if (slot, 1) not in plans[state_epoch]:
                if reproductions[state_epoch]["success_vector"][slot]:
                    terminal_before_replan_1.append(state_epoch)
                else:
                    skipped_states.append(f"missing_{state_epoch}_replan_1")
                continue
            available_labels = tuple(
                label
                for label in EPOCH_LABELS
                if all(
                    (slot, 1, int(iteration)) in details[label]
                    for iteration in manifest.protocol["detailed_iterations"]
                )
            )
            if not available_labels:
                skipped_states.append(
                    f"no_candidates_for_{state_epoch}_replan_1"
                )
                continue
            state_specs.append(
                (
                    state_epoch,
                    1,
                    plans[state_epoch][(slot, 1)],
                    available_labels,
                )
            )
        for state_epoch, replan, state_plan, available_labels in state_specs:
            prefix = np.asarray(
                state_plan["executed_prefix_before"], dtype=np.float32
            )
            chunk = dataset.load_chunk(
                np.asarray([episode]),
                np.asarray([start + len(prefix)]),
                np.asarray(
                    [
                        start
                        + len(prefix)
                        + int(cfg.plan_config.horizon)
                        * int(cfg.plan_config.action_block)
                    ]
                ),
            )[0]
            expert = normalize_expert_plan(
                chunk,
                scaler=scaler,
                horizon=int(cfg.plan_config.horizon),
                action_block=int(cfg.plan_config.action_block),
            )
            planning_info = state_plan["planning_info"]
            state_hash = planning_state_sha256(planning_info)
            for iteration in manifest.protocol["detailed_iterations"]:
                epoch_details = {
                    label: details[label][(slot, replan, int(iteration))]
                    for label in available_labels
                }
                seed = (
                    int(manifest.protocol["seed"])
                    + 1009 * int(slot)
                    + 97 * int(replan)
                    + int(iteration)
                )
                sampled_panel = build_cross_epoch_panel(
                    epoch_details,
                    expert=expert,
                    non_elite_count=int(
                        manifest.protocol["fixed_non_elites"]
                    ),
                    seed=seed,
                )
                name = (
                    f"slot_{slot:02d}_replan_{replan}_state_{state_epoch}_"
                    f"iteration_{int(iteration):02d}_sampled"
                )
                sampled = _evaluate_ground_panel(
                    base_identity=base_identity,
                    cache_root=pair_root / "ground" / "cache",
                    score_root=pair_root / "ground" / "scores",
                    grounder=grounder,
                    models=models,
                    planning_info=planning_info,
                    task=pair.task,
                    episode=episode,
                    start=start,
                    prefix=prefix,
                    panel=sampled_panel,
                    cache_name=name,
                    device=device,
                    topk=int(manifest.protocol["topk"]),
                )
                expansion_reason = _expansion_reason(sampled)
                final = sampled
                expanded = False
                if expansion_reason is not None:
                    full_panel = build_full_cross_epoch_panel(
                        epoch_details, expert=expert
                    )
                    expanded = True
                    final = _evaluate_ground_panel(
                        base_identity=base_identity,
                        cache_root=pair_root / "ground" / "cache",
                        score_root=pair_root / "ground" / "scores",
                        grounder=grounder,
                        models=models,
                        planning_info=planning_info,
                        task=pair.task,
                        episode=episode,
                        start=start,
                        prefix=prefix,
                        panel=full_panel,
                        cache_name=name.replace("sampled", "full"),
                        device=device,
                        topk=int(manifest.protocol["topk"]),
                    )
                panels.append(
                    {
                        "replan": replan,
                        "state_epoch": state_epoch,
                        "iteration": int(iteration),
                        "planning_state_sha256": state_hash,
                        "prefix_sha256": tensor_bytes_sha256(prefix),
                        "sampled_candidate_sha256": sampled[
                            "candidate_sha256"
                        ],
                        "candidate_sha256": final["candidate_sha256"],
                        "candidate_count": final["candidate_count"],
                        "candidate_successes": final[
                            "candidate_successes"
                        ],
                        "cem_candidate_successes": final[
                            "cem_candidate_successes"
                        ],
                        "expanded_to_all_candidates": expanded,
                        "expansion_reason": expansion_reason,
                        "models": final["models"],
                        "scores_file": final["scores_file"],
                        "sampled_elite_by_model": {
                            label: sampled["models"][label][
                                "elite_by_source_epoch"
                            ]
                            for label in EPOCH_LABELS
                        },
                        "sampled_selected_true_outcomes": sampled[
                            "selected_true_outcomes"
                        ],
                        "sampled_physical_cost_min": sampled[
                            "physical_cost_min"
                        ],
                    }
                )
        slot_result = {
            "slot": slot,
            "category": pair.category_for(slot),
            "dataset_episode": episode,
            "start_step": start,
            "initial_physical_distance": _initial_physical_distance(
                task=pair.task,
                dataset=dataset,
                episode=episode,
                start=start,
                goal_offset=int(manifest.protocol["goal_offset_steps"]),
            ),
            "final_success": {
                label: bool(reproductions[label]["success_vector"][slot])
                for label in EPOCH_LABELS
            },
            "skipped_states": skipped_states,
            "terminal_before_replan_1": terminal_before_replan_1,
            "panels": panels,
        }
        shared_iterations = {
            panel["iteration"]
            for panel in panels
            if panel["replan"] == 0 and panel["state_epoch"] == "shared"
        }
        if shared_iterations != {
            int(value) for value in manifest.protocol["detailed_iterations"]
        }:
            raise RuntimeError(
                f"slot {slot} has incomplete shared replan-0 panels: "
                f"{sorted(shared_iterations)}"
            )
        slot_result["attributions"], slot_result["replan_distances"] = (
            _attributions_for_slot(
                slot_result=slot_result,
                final_success=slot_result["final_success"],
            )
        )
        write_atomic_json(
            pair_root / "ground" / "slots" / f"slot_{slot:02d}.json",
            slot_result,
        )
        result_slots.append(slot_result)
        print(
            f"{pair.label}: grounded slot={slot} panels={len(panels)} "
            f"skipped={skipped_states}",
            flush=True,
        )
    result = {
        "schema_version": 1,
        "status": "pending_acceptance",
        "pair": pair.label,
        "identity": base_identity,
        "diagnostic_slots": list(pair.slots),
        "slots": result_slots,
    }
    result["acceptance_errors"] = _ground_artifact_errors(
        pair_root=pair_root,
        pair=pair,
        result=result,
        iterations=manifest.protocol["detailed_iterations"],
    )
    result["status"] = (
        "ok" if not result["acceptance_errors"] else "incomplete"
    )
    write_atomic_json(result_path, result)
    return result


def _format_metric(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _pair_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary['pair']} epoch-pair ranking diagnostic",
        "",
        f"- status: `{summary['status']}`",
        f"- task: `{summary['task']}`",
        f"- section: §{summary['section']}",
        f"- decision: `{summary['decision']}`",
        "",
        "| category | slots | panels | Δ Spearman | Δ top-30 recall | Δ success recall@30 | Δ regret | Δ latent MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category, row in summary["categories"].items():
        delta = row["e10_minus_e8"]
        lines.append(
            f"| {category} | {row['slot_count']} | {row['panel_count']} | "
            f"{_format_metric(delta['predicted_physical_spearman'])} | "
            f"{_format_metric(delta['physical_topk_recall'])} | "
            f"{_format_metric(delta['successful_candidate_recall_at_k'])} | "
            f"{_format_metric(delta['physical_best_candidate_regret'])} | "
            f"{_format_metric(delta['predicted_terminal_latent_mse'])} |"
        )
    lines.extend(["", "## Attribution counts", ""])
    for epoch_label, counts in summary["overall"][
        "attribution_counts"
    ].items():
        rendered = ", ".join(
            f"{label}={count}" for label, count in counts.items()
        ) or "none"
        lines.append(f"- {epoch_label}: {rendered}")
    return "\n".join(lines) + "\n"


def _global_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Fast-LeWAM epoch-pair ranking diagnostics",
        "",
        f"- status: `{summary['status']}`",
        "",
        "| pair | task | status | decision |",
        "|---|---|---|---|",
    ]
    for label, row in summary["pairs"].items():
        lines.append(
            f"| {label} | {row['task']} | {row['status']} | "
            f"{row['decision']} |"
        )
    if summary["missing_pairs"]:
        lines.extend(
            ["", f"Missing pairs: `{summary['missing_pairs']}`"]
        )
    return "\n".join(lines) + "\n"


def run_summarize(*, manifest_path: str | Path, pair_label: str):
    manifest = load_epoch_pair_manifest(manifest_path)
    pair = manifest.pairs[pair_label]
    pair_root = manifest.output_root / pair.label
    grounded = _read_json(pair_root / "ground" / "result.json")
    traces = {
        label: torch.load(
            pair_root / "trace" / label / "trace.pt",
            map_location="cpu",
            weights_only=False,
        )
        for label in EPOCH_LABELS
    }
    ground_identity = build_ground_identity(manifest, pair, traces)
    require_identity(
        grounded.get("identity", {}),
        ground_identity,
        source=pair_root / "ground" / "result.json",
    )
    artifact_errors = _ground_artifact_errors(
        pair_root=pair_root,
        pair=pair,
        result=grounded,
        iterations=manifest.protocol["detailed_iterations"],
    )
    grounded = dict(grounded)
    grounded["acceptance_errors"] = artifact_errors
    if artifact_errors:
        grounded["status"] = "incomplete"
    summary = summarize_grounded_pair(pair, grounded)
    summary["identity"] = build_summary_identity(
        manifest, pair, ground_identity
    )
    write_atomic_json(pair_root / "summary.json", summary)
    write_atomic_text(pair_root / "summary.md", _pair_markdown(summary))

    pairs = {}
    missing = []
    for label, configured_pair in manifest.pairs.items():
        path = manifest.output_root / label / "summary.json"
        if not path.is_file():
            missing.append(label)
            continue
        row = _read_json(path)
        configured_traces = {
            epoch_label: torch.load(
                manifest.output_root
                / label
                / "trace"
                / epoch_label
                / "trace.pt",
                map_location="cpu",
                weights_only=False,
            )
            for epoch_label in EPOCH_LABELS
        }
        configured_ground_identity = build_ground_identity(
            manifest, configured_pair, configured_traces
        )
        expected = build_summary_identity(
            manifest, configured_pair, configured_ground_identity
        )
        require_identity(row.get("identity", {}), expected, source=path)
        pairs[label] = {
            "task": row["task"],
            "section": row["section"],
            "status": row["status"],
            "decision": row["decision"],
            "categories": row["categories"],
            "overall": row["overall"],
        }
    global_summary = {
        "status": (
            "ok"
            if not missing
            and all(row["status"] == "ok" for row in pairs.values())
            else "incomplete"
        ),
        "manifest": str(manifest.path),
        "pairs": pairs,
        "missing_pairs": missing,
        "note": "Physical costs are summarized within pair only; no cross-task raw-cost average is computed.",
    }
    write_atomic_json(manifest.output_root / "summary.json", global_summary)
    write_atomic_text(
        manifest.output_root / "summary.md",
        _global_markdown(global_summary),
    )
    print(
        f"{pair.label}: summary={summary['status']} "
        f"global={global_summary['status']}",
        flush=True,
    )
    return summary


def run_phase(
    *, manifest_path: str | Path, pair_label: str, phase: str, device: str
):
    if phase == "trace":
        return run_trace(
            manifest_path=manifest_path, pair_label=pair_label, device=device
        )
    if phase == "ground":
        return run_ground(
            manifest_path=manifest_path, pair_label=pair_label, device=device
        )
    if phase == "summarize":
        return run_summarize(
            manifest_path=manifest_path, pair_label=pair_label
        )
    if phase == "all":
        run_trace(
            manifest_path=manifest_path, pair_label=pair_label, device=device
        )
        run_ground(
            manifest_path=manifest_path, pair_label=pair_label, device=device
        )
        return run_summarize(
            manifest_path=manifest_path, pair_label=pair_label
        )
    raise ValueError(f"unsupported phase {phase!r}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--pair",
        required=True,
        help="Pair label declared by the manifest",
    )
    parser.add_argument(
        "--phase", required=True, choices=("trace", "ground", "summarize", "all")
    )
    parser.add_argument("--device", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    run_phase(
        manifest_path=args.manifest,
        pair_label=args.pair,
        phase=args.phase,
        device=args.device,
    )


if __name__ == "__main__":
    main()
