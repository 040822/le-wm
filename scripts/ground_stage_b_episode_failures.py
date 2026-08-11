"""Ground saved CEM panels in the simulator after exact replay succeeds."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from omegaconf import OmegaConf
from scipy.stats import spearmanr
import stable_worldmodel as swm
import torch

from stable_worldmodel.world.world import _extract_init_goal
from scripts.diagnose_stage_b_episode_failures import (
    _load_subject,
    _sha256,
    _validate_device,
)
from source.common.eval import (
    compose_eval_config,
    fit_eval_processors,
    get_dataset,
    img_transform,
)
from source.diagnostics.stage_b_episode_failures import (
    FixedRawSequencePolicy,
    build_prefix_replay_actions,
    build_shared_initial_panel,
    classify_failure,
    compute_physical_terminal_cost,
    load_diagnostic_manifest,
    normalize_expert_plan,
    prepare_image_info,
    select_grounding_panel,
    should_expand_grounding,
)


@torch.inference_mode()
def _predict_costs(model, planning_info, candidates, device):
    candidates = candidates.to(device)
    count = len(candidates)
    expanded = {}
    for key, value in planning_info.items():
        if torch.is_tensor(value):
            value = value.to(device)
            expanded[key] = value[:, None].expand(
                value.shape[0], count, *value.shape[1:]
            )
        elif isinstance(value, np.ndarray):
            expanded[key] = np.repeat(value[:, None], count, axis=1)
    return (
        model.get_cost(expanded, candidates.unsqueeze(0))[0]
        .detach()
        .cpu()
        .numpy()
    )


@torch.inference_mode()
def _predict_terminal_latents(model, planning_info, candidates, device):
    candidates = candidates.to(device)
    pixels = planning_info["pixels"].to(device)
    if hasattr(model, "encode_pixels"):
        z0 = model.encode_pixels(model._last_frame(pixels))
        z0 = z0.expand(len(candidates), -1)
        timestep = z0.new_ones(len(candidates))
        return (
            model(
                z0,
                candidates,
                timestep,
                mode="stage_b",
            )["predicted_latents"][:, -1]
            .detach()
            .cpu()
        )
    count = len(candidates)
    expanded = {}
    for key, value in planning_info.items():
        if torch.is_tensor(value):
            value = value.to(device)
            expanded[key] = value[:, None].expand(
                value.shape[0], count, *value.shape[1:]
            )
    return (
        model.rollout(expanded, candidates.unsqueeze(0))["predicted_emb"][
            0, :, -1
        ]
        .detach()
        .cpu()
    )


@torch.inference_mode()
def _terminal_latents(model, prepared):
    pixels = prepared["pixels"]
    goal = prepared["goal"]
    if hasattr(model, "encode_pixels"):
        current = model.encode_pixels(model._last_frame(pixels))
        target = model.encode_pixels(model._last_frame(goal))
    else:
        current = model.encode({"pixels": pixels})["emb"][:, -1]
        target = model.encode({"pixels": goal})["emb"][:, -1]
    cost = (current - target).square().mean(dim=-1)
    return current.detach().cpu(), cost.detach().cpu().numpy()


def _finite_spearman(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


class SimulatorGrounder:
    """Execute normalized candidate plans from one saved cohort state."""

    def __init__(self, *, cfg, dataset, scaler, model, device):
        self.cfg = cfg
        self.dataset = dataset
        self.scaler = scaler
        self.model = model
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
        count = len(raw)
        world_policy = FixedRawSequencePolicy(raw)
        world_cfg = OmegaConf.to_container(self.cfg.world, resolve=True)
        world_cfg["num_envs"] = count
        world_cfg["max_episode_steps"] = 2 * int(self.cfg.eval.eval_budget)
        world = swm.World(**world_cfg, image_shape=(224, 224))
        world.set_policy(world_policy)
        try:
            metrics = world.evaluate(
                dataset=self.dataset,
                episodes_idx=[episode] * count,
                start_steps=[start] * count,
                goal_offset=int(self.cfg.eval.goal_offset_steps),
                eval_budget=int(raw.shape[1]),
                callables=OmegaConf.to_container(
                    self.cfg.eval.callables, resolve=True
                ),
                reset_mode="wait",
            )
            physical = compute_physical_terminal_cost(task, world.infos)
            terminal = prepare_image_info(
                {
                    "pixels": world.infos["pixels"],
                    "goal": world.infos["goal"],
                },
                self.transform,
                self.device,
            )
            terminal_latent, latent = _terminal_latents(self.model, terminal)
            success = np.asarray(
                metrics["episode_successes"], dtype=bool
            ).reshape(-1)
        finally:
            world.close()
        return {
            "physical_cost": physical,
            "true_terminal_latent_cost": latent,
            "terminal_latent": terminal_latent,
            "terminal_pixels": terminal["pixels"].detach().cpu(),
            "terminal_goal": terminal["goal"].detach().cpu(),
            "success": success,
        }


def _candidate_rows(
    actions,
    sources,
    predicted,
    predicted_terminal_latents,
    grounded,
):
    rows = []
    for index, source in enumerate(sources):
        action = actions[index]
        rows.append(
            {
                "panel_index": index,
                "source": source,
                "predicted_cost": float(predicted[index]),
                "true_terminal_latent_cost": float(
                    grounded["true_terminal_latent_cost"][index]
                ),
                "predicted_terminal_latent_mse": float(
                    (
                        predicted_terminal_latents[index]
                        - grounded["terminal_latent"][index]
                    )
                    .square()
                    .mean()
                    .item()
                ),
                "physical_cost": float(grounded["physical_cost"][index]),
                "success": bool(grounded["success"][index]),
                "action_out_of_bounds_fraction": float(
                    (action.abs() > 1).float().mean().item()
                ),
                "normalized_action_norm": float(action.norm().item()),
            }
        )
    return rows


def _panel_evidence(
    rows,
    *,
    all_candidates_grounded,
    final_selection_success,
):
    predicted = [row["predicted_cost"] for row in rows]
    latent = [row["true_terminal_latent_cost"] for row in rows]
    physical = [row["physical_cost"] for row in rows]
    successes = [row["success"] for row in rows]
    predicted_latent = _finite_spearman(predicted, latent)
    latent_physical = _finite_spearman(latent, physical)
    predicted_physical = _finite_spearman(predicted, physical)
    success_indices = [index for index, value in enumerate(successes) if value]
    order = np.argsort(np.asarray(predicted), kind="stable")
    topk = min(30, len(rows))
    topk_success_recall = (
        float(sum(successes[index] for index in order[:topk]) / len(success_indices))
        if success_indices
        else None
    )
    best = int(order[0])
    evidence = {
        "all_candidates_grounded": bool(all_candidates_grounded),
        "candidate_successes": successes,
        "final_success": bool(final_selection_success),
        "final_selection_success": bool(final_selection_success),
        "predicted_true_latent_spearman": predicted_latent,
        "true_latent_physical_spearman": latent_physical,
    }
    return {
        "predicted_physical_spearman": predicted_physical,
        "predicted_true_latent_spearman": predicted_latent,
        "true_latent_physical_spearman": latent_physical,
        "topk_success_recall": topk_success_recall,
        "regret": float(physical[best] - min(physical)),
        "oracle_success": bool(any(successes)),
        "labels": classify_failure(evidence),
    }


def run_grounding(*, manifest_path, run_label, device, output_dir):
    _validate_device(device)
    manifest = load_diagnostic_manifest(manifest_path)
    run = manifest.runs[run_label]
    run_output = Path(output_dir).expanduser().resolve() / run_label
    reproduction = json.loads(
        (run_output / "reproduction.json").read_text(encoding="utf-8")
    )
    if reproduction["status"] != "reproducible":
        raise RuntimeError(
            f"{run_label} is {reproduction['status']}; grounding is prohibited"
        )
    trace = torch.load(
        run_output / "trace.pt",
        map_location="cpu",
        weights_only=False,
    )
    reference = json.loads(run.reference_result.read_text(encoding="utf-8"))
    cfg = compose_eval_config(
        run.task,
        (f"solver.device={device}", "output.save_video=false"),
    )
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    scaler = fit_eval_processors(dataset, ["action"])["action"]
    subject = _load_subject(run, device)
    model = getattr(subject, "model", subject).to(device).eval()
    actor_run = next(
        candidate
        for candidate in manifest.runs.values()
        if candidate.task == run.task and candidate.family == "e3_384"
    )
    actor_checkpoint_sha256 = _sha256(actor_run.checkpoint)
    if actor_run.checkpoint == run.checkpoint and hasattr(model, "get_action"):
        actor_model = model
    else:
        actor_subject = _load_subject(actor_run, device)
        actor_model = getattr(actor_subject, "model", actor_subject).to(device).eval()
    grounder = SimulatorGrounder(
        cfg=cfg,
        dataset=dataset,
        scaler=scaler,
        model=model,
        device=device,
    )
    plans = {
        (int(item["slot"]), int(item["replan"])): item
        for item in trace["plans"]
    }
    details = {
        (
            int(item["slot"]),
            int(item["replan"]),
            int(item["iteration"]),
        ): item
        for item in trace["details"]
    }
    failed_slots = [
        slot
        for slot, success in enumerate(reproduction["success_vector"])
        if not success
    ]
    diagnostic_slots = [
        int(slot)
        for slot in reproduction["metadata"]["slot_selection"]["slots"]
    ]
    grounded_root = run_output / "grounded"
    grounded_root.mkdir(parents=True, exist_ok=True)
    result_slots = []
    for slot in diagnostic_slots:
        final_success = bool(reproduction["success_vector"][slot])
        episode = reference["parameters"]["episode_ids"][slot]
        start = int(reference["parameters"]["start_steps"][slot])
        slot_panels = []
        slot_labels = set()
        for replan in (0, 1):
            plan = plans.get((slot, replan))
            if plan is None:
                continue
            prefix = np.asarray(plan["executed_prefix_before"], dtype=np.float32)
            expert_chunk = dataset.load_chunk(
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
                expert_chunk,
                scaler=scaler,
                horizon=int(cfg.plan_config.horizon),
                action_block=int(cfg.plan_config.action_block),
            )
            if replan == 0:
                actor_info = {
                    key: value.to(device)
                    for key, value in plan["planning_info"].items()
                    if torch.is_tensor(value)
                }
                actor_generator = torch.Generator(device=device).manual_seed(
                    int(manifest.protocol["seed"]) + 1009 * slot
                )
                with torch.inference_mode():
                    actor = actor_model.get_action(
                        actor_info,
                        horizon=int(cfg.plan_config.horizon),
                        generator=actor_generator,
                        num_steps=int(actor_model.inference_steps),
                    )[0].detach().cpu()
                shared_panel = build_shared_initial_panel(
                    expert,
                    actor,
                    random_candidates=16,
                    actor_neighbors=16,
                    actor_noise_std=0.2,
                    generator=torch.Generator().manual_seed(
                        int(manifest.protocol["seed"]) + 1009 * slot
                    ),
                )
                shared_predicted = _predict_costs(
                    model,
                    plan["planning_info"],
                    shared_panel.actions,
                    device,
                )
                shared_predicted_terminal = _predict_terminal_latents(
                    model,
                    plan["planning_info"],
                    shared_panel.actions,
                    device,
                )
                shared_cache = (
                    Path(output_dir).expanduser().resolve()
                    / "_shared_initial"
                    / run.task
                    / f"slot_{slot:02d}.pt"
                )
                cache_identity = {
                    "task": run.task,
                    "slot": slot,
                    "episode": int(episode),
                    "start": start,
                    "seed": int(manifest.protocol["seed"]),
                    "goal_offset_steps": int(
                        manifest.protocol["goal_offset_steps"]
                    ),
                    "horizon": int(manifest.protocol["horizon"]),
                    "action_block": int(manifest.protocol["action_block"]),
                    "manifest_sha256": reproduction["metadata"][
                        "manifest_sha256"
                    ],
                    "actor_checkpoint_sha256": actor_checkpoint_sha256,
                }
                shared_cache.parent.mkdir(parents=True, exist_ok=True)
                with shared_cache.with_suffix(".lock").open("a+b") as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                    if shared_cache.is_file():
                        cached = torch.load(
                            shared_cache,
                            map_location="cpu",
                            weights_only=False,
                        )
                        if cached.get("identity") != cache_identity:
                            raise RuntimeError(
                                f"stale shared cache identity: {shared_cache}"
                            )
                        torch.testing.assert_close(
                            cached["actions"],
                            shared_panel.actions,
                            rtol=0,
                            atol=0,
                        )
                    else:
                        shared_grounded = grounder.run(
                            task=run.task,
                            episode=episode,
                            start=start,
                            candidates=shared_panel.actions,
                            prefix_raw=None,
                        )
                        payload = {
                            "identity": cache_identity,
                            "actions": shared_panel.actions,
                            "sources": shared_panel.sources,
                            "physical_cost": shared_grounded[
                                "physical_cost"
                            ],
                            "success": shared_grounded["success"],
                            "terminal_pixels": shared_grounded[
                                "terminal_pixels"
                            ],
                            "terminal_goal": shared_grounded[
                                "terminal_goal"
                            ],
                        }
                        with tempfile.NamedTemporaryFile(
                            dir=shared_cache.parent,
                            prefix=f".{shared_cache.name}.",
                            delete=False,
                        ) as temporary:
                            temporary_path = Path(temporary.name)
                        try:
                            torch.save(payload, temporary_path)
                            os.replace(temporary_path, shared_cache)
                        finally:
                            temporary_path.unlink(missing_ok=True)
                        cached = payload
                    terminal_latent, terminal_cost = _terminal_latents(
                        model,
                        {
                            "pixels": cached["terminal_pixels"].to(device),
                            "goal": cached["terminal_goal"].to(device),
                        },
                    )
                    shared_grounded = {
                        "physical_cost": cached["physical_cost"],
                        "true_terminal_latent_cost": terminal_cost,
                        "terminal_latent": terminal_latent,
                        "success": cached["success"],
                    }
                shared_rows = _candidate_rows(
                    shared_panel.actions,
                    shared_panel.sources,
                    shared_predicted,
                    shared_predicted_terminal,
                    shared_grounded,
                )
                shared_summary = _panel_evidence(
                    shared_rows,
                    all_candidates_grounded=False,
                    final_selection_success=final_success,
                )
                slot_panels.append(
                    {
                        "kind": "shared_initial",
                        "replan": 0,
                        "iteration": None,
                        "expanded_to_all_candidates": False,
                        "expansion_reason": None,
                        "summary": shared_summary,
                        "candidates": shared_rows,
                    }
                )
                if not final_success:
                    slot_labels.update(shared_summary["labels"])
            for iteration in (0, 5, 29):
                detail = details[(slot, replan, iteration)]
                panel = select_grounding_panel(
                    candidates=detail["candidates"],
                    costs=detail["costs"],
                    topk_indices=detail["topk_indices"],
                    previous_mean=detail["previous_mean"],
                    best_ever=detail.get(
                        "best_ever",
                        detail["candidates"][
                            int(torch.argmin(detail["costs"]).item())
                        ],
                    ),
                    anchors={"expert": expert},
                    non_elite_count=10,
                    seed=(
                        int(manifest.protocol["seed"])
                        + 1009 * slot
                        + 97 * replan
                        + iteration
                    ),
                )
                predicted = _predict_costs(
                    model,
                    plan["planning_info"],
                    panel.actions,
                    device,
                )
                predicted_terminal = _predict_terminal_latents(
                    model,
                    plan["planning_info"],
                    panel.actions,
                    device,
                )
                grounded = grounder.run(
                    task=run.task,
                    episode=episode,
                    start=start,
                    candidates=panel.actions,
                    prefix_raw=prefix if len(prefix) else None,
                )
                rows = _candidate_rows(
                    panel.actions,
                    panel.sources,
                    predicted,
                    predicted_terminal,
                    grounded,
                )
                elite_physical = [
                    row["physical_cost"]
                    for row in rows
                    if row["source"].startswith("elite_")
                ]
                updated_mean_physical = next(
                    (
                        row["physical_cost"]
                        for row in rows
                        if row["source"] == "elite_mean"
                    ),
                    None,
                )
                reference_actions = [
                    row
                    for row in rows
                    if row["success"] or row["source"] == "anchor_expert"
                ]
                selection_rows = [
                    row for row in rows if row["source"] == "elite_mean"
                ]
                if not selection_rows:
                    selection_rows = [
                        row
                        for row in rows
                        if row["source"] == "previous_mean"
                    ]
                selection_success = selection_rows[0]["success"]
                predicted_latent = _finite_spearman(
                    [row["predicted_cost"] for row in rows],
                    [row["true_terminal_latent_cost"] for row in rows],
                )
                latent_physical = _finite_spearman(
                    [row["true_terminal_latent_cost"] for row in rows],
                    [row["physical_cost"] for row in rows],
                )
                error_decomposition_clear = (
                    predicted_latent is not None
                    and latent_physical is not None
                    and abs(predicted_latent) >= 0.2
                    and abs(latent_physical) >= 0.2
                )
                expansion_reason = should_expand_grounding(
                    panel_successes=grounded["success"],
                    final_selection_success=selection_success,
                    error_decomposition_clear=error_decomposition_clear,
                )
                expanded = expansion_reason is not None
                if expanded:
                    expanded = True
                    full_actions = detail["candidates"]
                    full_predicted = detail["costs"].numpy()
                    full_predicted_terminal = _predict_terminal_latents(
                        model,
                        plan["planning_info"],
                        full_actions,
                        device,
                    )
                    full_grounded = grounder.run(
                        task=run.task,
                        episode=episode,
                        start=start,
                        candidates=full_actions,
                        prefix_raw=prefix if len(prefix) else None,
                    )
                    rows = _candidate_rows(
                        full_actions,
                        tuple(
                            f"candidate_{index}"
                            for index in range(len(full_actions))
                        ),
                        full_predicted,
                        full_predicted_terminal,
                        full_grounded,
                    )
                summary = _panel_evidence(
                    rows,
                    all_candidates_grounded=expanded,
                    final_selection_success=selection_success,
                )
                summary["elite_physical_cost_mean"] = (
                    float(np.mean(elite_physical))
                    if elite_physical
                    else None
                )
                summary["updated_mean_physical_cost"] = updated_mean_physical
                summary["reference_action_norm_mean"] = (
                    float(
                        np.mean(
                            [
                                row["normalized_action_norm"]
                                for row in reference_actions
                            ]
                        )
                    )
                    if reference_actions
                    else None
                )
                summary["reference_action_out_of_bounds_fraction_mean"] = (
                    float(
                        np.mean(
                            [
                                row["action_out_of_bounds_fraction"]
                                for row in reference_actions
                            ]
                        )
                    )
                    if reference_actions
                    else None
                )
                if not final_success:
                    slot_labels.update(summary["labels"])
                slot_panels.append(
                    {
                        "kind": "cem",
                        "replan": replan,
                        "iteration": iteration,
                        "expanded_to_all_candidates": expanded,
                        "expansion_reason": expansion_reason,
                        "summary": summary,
                        "candidates": rows,
                    }
                )
            replan_panels = [
                panel
                for panel in slot_panels
                if panel["kind"] == "cem" and panel["replan"] == replan
            ]
            if len(replan_panels) == 3 and not final_success:
                predicted_curve = [
                    float(
                        details[
                            (slot, replan, panel["iteration"])
                        ]["costs"][
                            details[
                                (slot, replan, panel["iteration"])
                            ]["topk_indices"]
                        ].mean().item()
                    )
                    for panel in replan_panels
                ]
                physical_curve = [
                    panel["summary"]["elite_physical_cost_mean"]
                    for panel in replan_panels
                ]
                if all(value is not None for value in physical_curve):
                    exploitation = classify_failure(
                        {
                            "final_success": False,
                            "predicted_elite_cost_0_5_29": predicted_curve,
                            "physical_cost_0_5_29": physical_curve,
                            "earlier_best_physical_cost": min(
                                row["physical_cost"]
                                for panel in replan_panels
                                for row in panel["candidates"]
                            ),
                            "final_mean_physical_cost": replan_panels[-1][
                                "summary"
                            ]["updated_mean_physical_cost"],
                        }
                    )
                    slot_labels.update(exploitation)

        if (
            not final_success
            and (slot, 0) in plans
            and (slot, 1) in plans
        ):
            init_state, goal_state, _ = _extract_init_goal(
                dataset,
                [episode],
                [start],
                int(cfg.eval.goal_offset_steps),
            )
            if run.task == "pusht":
                initial_distance = float(
                    np.linalg.norm(
                        np.asarray(init_state["state"]).reshape(1, -1)
                        - np.asarray(goal_state["goal_state"]).reshape(1, -1)
                    )
                )
            else:
                initial_distance = float(
                    np.linalg.norm(
                        np.asarray(init_state["qpos"]).reshape(1, -1)
                        - np.asarray(goal_state["goal_qpos"]).reshape(1, -1)
                    )
                )
            first_grounded = grounder.run(
                task=run.task,
                episode=episode,
                start=start,
                candidates=plans[(slot, 0)]["normalized_plan"].unsqueeze(0),
                prefix_raw=None,
            )
            second_grounded = grounder.run(
                task=run.task,
                episode=episode,
                start=start,
                candidates=plans[(slot, 1)]["normalized_plan"].unsqueeze(0),
                prefix_raw=np.asarray(
                    plans[(slot, 1)]["executed_prefix_before"],
                    dtype=np.float32,
                ),
            )
            distances = [
                initial_distance,
                float(first_grounded["physical_cost"][0]),
                float(second_grounded["physical_cost"][0]),
            ]
            slot_labels.update(
                classify_failure(
                    {
                        "final_success": False,
                        "physical_distance_start_after_first_final": distances,
                    }
                )
            )
            final_plan = plans[(slot, 1)]["normalized_plan"]
            reference_norms = [
                panel["summary"]["reference_action_norm_mean"]
                for panel in slot_panels
                if panel["summary"]["reference_action_norm_mean"] is not None
            ]
            reference_oob = [
                panel["summary"][
                    "reference_action_out_of_bounds_fraction_mean"
                ]
                for panel in slot_panels
                if panel["summary"][
                    "reference_action_out_of_bounds_fraction_mean"
                ]
                is not None
            ]
            if reference_norms and reference_oob:
                slot_labels.update(
                    classify_failure(
                        {
                            "final_success": False,
                            "failed_action_out_of_bounds_fraction": float(
                                (final_plan.abs() > 1).float().mean().item()
                            ),
                            "reference_action_out_of_bounds_fraction": float(
                                np.mean(reference_oob)
                            ),
                            "failed_action_norm": float(final_plan.norm().item()),
                            "reference_action_norm": float(
                                np.mean(reference_norms)
                            ),
                        }
                    )
                )
        else:
            distances = None
        slot_result = {
            "slot": slot,
            "dataset_episode": episode,
            "start_step": start,
            "labels": sorted(slot_labels) or ["mixed_or_undetermined"],
            "physical_distance_start_after_first_final": distances,
            "panels": slot_panels,
        }
        (grounded_root / f"slot_{slot:02d}.json").write_text(
            json.dumps(
                slot_result,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        result_slots.append(slot_result)
    result = {
        "status": "ok",
        "run_label": run_label,
        "failed_slots": failed_slots,
        "diagnostic_slots": diagnostic_slots,
        "slots": result_slots,
    }
    (grounded_root / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"grounded={grounded_root / 'result.json'}")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    run_grounding(
        manifest_path=args.manifest,
        run_label=args.run_label,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
