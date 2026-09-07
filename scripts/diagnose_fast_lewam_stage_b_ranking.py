"""Measure Fast-LeWAM Stage-B ranking against simulator rollouts.

The diagnostic restores the canonical evaluation states, constructs expert,
random, and actor-centred candidate action sequences, scores them with the
Stage-B latent cost, then executes every candidate for one planning horizon.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from stable_worldmodel.world.world import _extract_init_goal

from eval_fast_lewam import load_model_from_weights
from source.common.eval import (
    compose_eval_config,
    fit_eval_processors,
    get_dataset,
    img_transform,
)


def build_candidate_set(
    expert,
    actor,
    *,
    random_candidates,
    actor_neighbors,
    expert_neighbors,
    expert_noise_std,
    actor_noise_std,
    generator,
):
    """Build anchored candidate families in normalized action space."""
    expert = torch.as_tensor(expert).detach().cpu().float()
    actor = torch.as_tensor(actor).detach().cpu().float()
    if expert.ndim != 2 or tuple(actor.shape) != tuple(expert.shape):
        raise ValueError("expert and actor must share shape [H,A]")
    counts = (random_candidates, actor_neighbors, expert_neighbors)
    if any(int(count) < 0 for count in counts):
        raise ValueError("candidate counts must be non-negative")
    if expert_noise_std <= 0 or actor_noise_std <= 0:
        raise ValueError("candidate noise scales must be positive")

    candidates = [
        expert,
        actor,
        torch.zeros_like(expert),
        expert.flip(0),
        expert.roll(shifts=1, dims=0),
    ]
    families = [
        "expert",
        "actor",
        "zero",
        "time_reverse",
        "time_roll",
    ]

    def append_noise_family(center, count, scale, family):
        if count == 0:
            return
        noise = torch.randn(
            int(count),
            *expert.shape,
            generator=generator,
            dtype=expert.dtype,
        )
        candidates.extend(center.unsqueeze(0) + float(scale) * noise)
        families.extend([family] * int(count))

    append_noise_family(
        torch.zeros_like(expert),
        int(random_candidates),
        1.0,
        "random",
    )
    append_noise_family(
        actor,
        int(actor_neighbors),
        actor_noise_std,
        "actor_neighbor",
    )
    append_noise_family(
        expert,
        int(expert_neighbors),
        expert_noise_std,
        "expert_neighbor",
    )
    return torch.stack(candidates), families


def summarize_ranking(
    *, predicted_cost, true_cost, successes, families, topk=5
):
    """Summarize one state's predicted ranking against simulator outcomes."""
    predicted_cost = np.asarray(predicted_cost, dtype=np.float64).reshape(-1)
    true_cost = np.asarray(true_cost, dtype=np.float64).reshape(-1)
    successes = np.asarray(successes, dtype=bool).reshape(-1)
    families = list(families)
    size = len(predicted_cost)
    if not (size == len(true_cost) == len(successes) == len(families)):
        raise ValueError("ranking arrays and families must have equal length")
    if size < 1 or not np.isfinite(predicted_cost).all():
        raise ValueError("predicted costs must be finite and non-empty")
    if not np.isfinite(true_cost).all():
        raise ValueError("true costs must be finite")

    if size > 1 and np.ptp(predicted_cost) > 0 and np.ptp(true_cost) > 0:
        spearman = float(spearmanr(predicted_cost, true_cost).statistic)
    else:
        spearman = float("nan")
    k = min(max(int(topk), 1), size)
    predicted_order = np.argsort(predicted_cost, kind="stable")
    true_order = np.argsort(true_cost, kind="stable")
    predicted_top = set(predicted_order[:k].tolist())
    true_top = set(true_order[:k].tolist())
    predicted_best = int(predicted_order[0])

    by_family = {}
    family_indices = defaultdict(list)
    for index, family in enumerate(families):
        family_indices[family].append(index)
    for family, indices in family_indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        by_family[family] = {
            "count": int(len(idx)),
            "predicted_cost_mean": float(predicted_cost[idx].mean()),
            "true_cost_mean": float(true_cost[idx].mean()),
            "true_cost_min": float(true_cost[idx].min()),
            "success_rate": float(successes[idx].mean()),
        }

    return {
        "candidate_count": size,
        "spearman": spearman,
        "topk": k,
        "topk_recall": float(len(predicted_top & true_top) / k),
        "predicted_top1_index": predicted_best,
        "predicted_top1_family": families[predicted_best],
        "predicted_top1_success": float(successes[predicted_best]),
        "oracle_success": float(successes.any()),
        "predicted_top1_true_cost": float(true_cost[predicted_best]),
        "oracle_true_cost": float(true_cost.min()),
        "best_candidate_regret": float(
            true_cost[predicted_best] - true_cost.min()
        ),
        "families": by_family,
    }


def summarize_candidate_pools(
    *, predicted_cost, true_cost, successes, families, topk=5
):
    """Compare the zero-centred, actor-centred, and expert-centred pools."""
    definitions = {
        "zero_centered": {"zero", "random"},
        "actor_centered": {"actor", "actor_neighbor"},
        "expert_centered": {"expert", "expert_neighbor"},
    }
    predicted_cost = np.asarray(predicted_cost)
    true_cost = np.asarray(true_cost)
    successes = np.asarray(successes)
    families = np.asarray(families, dtype=object)
    summaries = {}
    for name, allowed in definitions.items():
        indices = np.flatnonzero(np.isin(families, list(allowed)))
        summaries[name] = summarize_ranking(
            predicted_cost=predicted_cost[indices],
            true_cost=true_cost[indices],
            successes=successes[indices],
            families=families[indices].tolist(),
            topk=topk,
        )
        summaries[name]["candidate_success_rate"] = float(
            np.asarray(successes[indices], dtype=bool).mean()
        )
    return summaries


def compute_physical_terminal_cost(task, info):
    """Return the environment task's terminal state distance to its goal."""
    if task == "pusht":
        current_key, goal_key = "state", "goal_state"
    elif task == "reacher":
        current_key, goal_key = "qpos", "goal_qpos"
    else:
        raise ValueError(f"unsupported diagnostic task {task!r}")
    if current_key not in info or goal_key not in info:
        raise KeyError(f"terminal info needs {current_key!r} and {goal_key!r}")
    current = np.asarray(info[current_key]).reshape(len(info[current_key]), -1)
    goal = np.asarray(info[goal_key]).reshape(len(info[goal_key]), -1)
    return np.linalg.norm(current - goal, axis=1)


class FixedSequencePolicy:
    """Execute one precomputed raw action sequence per simulator environment."""

    def __init__(self, actions):
        self.set_actions(actions)

    def set_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 3:
            raise ValueError("fixed actions must have shape [N,T,A]")
        self.actions = actions
        self.step = 0

    def set_env(self, env):
        self.env = env

    def get_action(self, info):
        index = min(self.step, self.actions.shape[1] - 1)
        self.step += 1
        return self.actions[:, index]


def _prepare_image_info(raw_info, transform, device):
    preprocessor = swm.policy.BasePolicy()
    preprocessor.process = {}
    preprocessor.transform = transform
    prepared = preprocessor._prepare_info(raw_info)
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in prepared.items()
    }


@torch.inference_mode()
def _predict_candidate_costs(model, info, candidates, device):
    candidates = candidates.to(device)
    count = candidates.shape[0]
    expanded = {}
    for key in ("pixels", "goal"):
        value = info[key]
        expanded[key] = value[:, None].expand(
            value.shape[0], count, *value.shape[1:]
        )
    return (
        model.get_cost(expanded, candidates.unsqueeze(0))[0]
        .detach()
        .cpu()
        .numpy()
    )


@torch.inference_mode()
def _simulator_latent_costs(model, info, device):
    current = model.encode_pixels(model._last_frame(info["pixels"]))
    goal = model.encode_pixels(model._last_frame(info["goal"]))
    return (
        (current - goal)
        .square()
        .mean(dim=-1)
        .detach()
        .cpu()
        .numpy()
    )


def _normalized_expert_plan(chunk, scaler, horizon, action_block):
    raw = np.asarray(chunk["action"], dtype=np.float32)
    plan_steps = int(horizon) * int(action_block)
    raw = raw[:plan_steps]
    if raw.shape[0] != plan_steps:
        raise ValueError(
            f"expert chunk has {raw.shape[0]} actions, expected {plan_steps}"
        )
    normalized = scaler.transform(np.nan_to_num(raw, nan=0.0))
    return torch.from_numpy(normalized.reshape(int(horizon), -1)).float()


def _raw_action_sequences(candidates, scaler, action_block):
    candidates = candidates.detach().cpu().numpy()
    count, horizon, action_dim = candidates.shape
    base_dim = len(scaler.mean_)
    expected = int(action_block) * base_dim
    if action_dim != expected:
        raise ValueError(
            f"candidate action_dim={action_dim}, expected {expected}"
        )
    flattened = candidates.reshape(count * horizon * int(action_block), base_dim)
    raw = scaler.inverse_transform(flattened)
    return raw.reshape(count, horizon * int(action_block), base_dim)


def _finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def _aggregate_state_results(states):
    physical = [state["physical_ranking"] for state in states]
    latent = [state["simulator_latent_ranking"] for state in states]
    aggregate = {
        "states": len(states),
        "physical_spearman_mean": _finite_mean(
            [item["spearman"] for item in physical]
        ),
        "physical_topk_recall_mean": _finite_mean(
            [item["topk_recall"] for item in physical]
        ),
        "physical_regret_mean": _finite_mean(
            [item["best_candidate_regret"] for item in physical]
        ),
        "simulator_latent_spearman_mean": _finite_mean(
            [item["spearman"] for item in latent]
        ),
        "simulator_latent_topk_recall_mean": _finite_mean(
            [item["topk_recall"] for item in latent]
        ),
        "predicted_top1_success_rate": _finite_mean(
            [item["predicted_top1_success"] for item in physical]
        ),
        "oracle_success_rate": _finite_mean(
            [item["oracle_success"] for item in physical]
        ),
    }
    family_rows = defaultdict(lambda: defaultdict(list))
    for state in physical:
        for family, metrics in state["families"].items():
            for key, value in metrics.items():
                if key != "count":
                    family_rows[family][key].append(value)
    aggregate["families"] = {
        family: {
            key: _finite_mean(values)
            for key, values in metrics.items()
        }
        for family, metrics in family_rows.items()
    }
    aggregate["pools"] = {}
    for pool in states[0]["physical_pools"]:
        pool_rows = [state["physical_pools"][pool] for state in states]
        aggregate["pools"][pool] = {
            key: _finite_mean([row[key] for row in pool_rows])
            for key in (
                "spearman",
                "topk_recall",
                "predicted_top1_success",
                "oracle_success",
                "candidate_success_rate",
                "best_candidate_regret",
            )
        }
    return aggregate


def _write_markdown(result, path):
    aggregate = result["aggregate"]
    lines = [
        f"# {result['task']} Stage-B simulator-grounded ranking",
        "",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- slots: `{result['slots']}`",
        f"- candidates per state: {result['parameters']['candidate_count']}",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "physical_spearman_mean",
        "physical_topk_recall_mean",
        "physical_regret_mean",
        "simulator_latent_spearman_mean",
        "simulator_latent_topk_recall_mean",
        "predicted_top1_success_rate",
        "oracle_success_rate",
    ):
        value = aggregate[key]
        lines.append(f"| {key} | {value:.4f} |" if value is not None else f"| {key} | n/a |")
    lines.extend(
        [
            "",
            "| family | predicted cost | physical cost | success rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for family, values in aggregate["families"].items():
        lines.append(
            f"| {family} | {values['predicted_cost_mean']:.4f} | "
            f"{values['true_cost_mean']:.4f} | {values['success_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| pool | Spearman | top-k recall | top-1 success | oracle success | candidate success | regret |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pool, values in aggregate["pools"].items():
        lines.append(
            f"| {pool} | {values['spearman']:.4f} | "
            f"{values['topk_recall']:.4f} | "
            f"{values['predicted_top1_success']:.4f} | "
            f"{values['oracle_success']:.4f} | "
            f"{values['candidate_success_rate']:.4f} | "
            f"{values['best_candidate_regret']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostic(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint = (
        run_dir
        / "checkpoints"
        / f"fast_lewam_weights_epoch_{int(args.epoch)}.pt"
    )
    run_config = run_dir / "config.yaml"
    cohort_path = run_dir / "eval" / f"epoch_{int(args.epoch)}" / "stage_b" / "result.json"
    for path in (checkpoint, run_config, cohort_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    slots = [int(slot) for slot in args.slots]
    if not slots or min(slots) < 0 or max(slots) >= len(cohort["episodes"]):
        raise ValueError("slots must index the saved Stage-B evaluation cohort")

    cfg = compose_eval_config(
        args.config_name,
        ("dataset.keys_to_cache=[action]",),
    )
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    process = fit_eval_processors(dataset, ["action"])
    scaler = process["action"]
    run_cfg = OmegaConf.load(run_config)
    model = load_model_from_weights(run_cfg, checkpoint, args.device)
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    episodes = [cohort["parameters"]["episode_ids"][slot] for slot in slots]
    starts = [cohort["parameters"]["start_steps"][slot] for slot in slots]
    horizon = int(cfg.plan_config.horizon)
    action_block = int(cfg.plan_config.action_block)
    plan_steps = horizon * action_block
    init_state, goal_state, _ = _extract_init_goal(
        dataset,
        episodes,
        starts,
        int(cfg.eval.goal_offset_steps),
    )
    chunks = dataset.load_chunk(
        np.asarray(episodes),
        np.asarray(starts),
        np.asarray(starts) + plan_steps,
    )
    actor_info = _prepare_image_info(
        {
            "pixels": init_state["pixels"][:, None],
            "goal": goal_state["goal"][:, None],
        },
        transform,
        args.device,
    )
    actor_generator = torch.Generator(device=args.device).manual_seed(int(args.seed))
    with torch.inference_mode():
        actor_plans = model.get_action(
            actor_info,
            horizon=horizon,
            generator=actor_generator,
            num_steps=int(model.inference_steps),
        ).detach().cpu()

    candidate_count = 5 + int(args.random_candidates) + int(args.actor_neighbors) + int(args.expert_neighbors)
    fixed_policy = FixedSequencePolicy(
        np.zeros(
            (candidate_count, plan_steps, len(scaler.mean_)),
            dtype=np.float32,
        )
    )
    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    world_cfg["num_envs"] = candidate_count
    world_cfg["max_episode_steps"] = 2 * int(cfg.eval.eval_budget)
    world = swm.World(**world_cfg, image_shape=(224, 224))
    world.set_policy(fixed_policy)
    callables = OmegaConf.to_container(cfg.eval.callables, resolve=True)
    states = []
    started = time.perf_counter()
    try:
        for position, slot in enumerate(slots):
            expert = _normalized_expert_plan(
                chunks[position], scaler, horizon, action_block
            )
            candidate_generator = torch.Generator().manual_seed(
                int(args.seed) + 1009 * int(slot)
            )
            candidates, families = build_candidate_set(
                expert,
                actor_plans[position],
                random_candidates=int(args.random_candidates),
                actor_neighbors=int(args.actor_neighbors),
                expert_neighbors=int(args.expert_neighbors),
                expert_noise_std=float(args.expert_noise_std),
                actor_noise_std=float(args.actor_noise_std),
                generator=candidate_generator,
            )
            state_info = {
                key: value[position : position + 1]
                for key, value in actor_info.items()
            }
            predicted_cost = _predict_candidate_costs(
                model, state_info, candidates, args.device
            )
            fixed_policy.set_actions(
                _raw_action_sequences(candidates, scaler, action_block)
            )
            metrics = world.evaluate(
                dataset=dataset,
                episodes_idx=[episodes[position]] * candidate_count,
                start_steps=[starts[position]] * candidate_count,
                goal_offset=int(cfg.eval.goal_offset_steps),
                eval_budget=plan_steps,
                callables=callables,
                reset_mode="wait",
            )
            physical_cost = compute_physical_terminal_cost(
                args.config_name, world.infos
            )
            terminal_info = _prepare_image_info(
                {
                    "pixels": world.infos["pixels"],
                    "goal": world.infos["goal"],
                },
                transform,
                args.device,
            )
            simulator_latent_cost = _simulator_latent_costs(
                model, terminal_info, args.device
            )
            successes = np.asarray(metrics["episode_successes"], dtype=bool)
            physical_ranking = summarize_ranking(
                predicted_cost=predicted_cost,
                true_cost=physical_cost,
                successes=successes,
                families=families,
                topk=int(args.topk),
            )
            physical_pools = summarize_candidate_pools(
                predicted_cost=predicted_cost,
                true_cost=physical_cost,
                successes=successes,
                families=families,
                topk=int(args.topk),
            )
            latent_ranking = summarize_ranking(
                predicted_cost=predicted_cost,
                true_cost=simulator_latent_cost,
                successes=successes,
                families=families,
                topk=int(args.topk),
            )
            states.append(
                {
                    "slot": slot,
                    "dataset_episode": episodes[position],
                    "start_step": starts[position],
                    "physical_ranking": physical_ranking,
                    "physical_pools": physical_pools,
                    "simulator_latent_ranking": latent_ranking,
                    "candidates": [
                        {
                            "family": family,
                            "predicted_cost": float(predicted_cost[index]),
                            "physical_cost": float(physical_cost[index]),
                            "simulator_latent_cost": float(
                                simulator_latent_cost[index]
                            ),
                            "success": bool(successes[index]),
                        }
                        for index, family in enumerate(families)
                    ],
                }
            )
            print(
                f"slot={slot} spearman={physical_ranking['spearman']:.3f} "
                f"topk={physical_ranking['topk_recall']:.3f} "
                f"regret={physical_ranking['best_candidate_regret']:.3f} "
                f"top1_success={physical_ranking['predicted_top1_success']:.0f} "
                f"oracle_success={physical_ranking['oracle_success']:.0f}",
                flush=True,
            )
    finally:
        world.close()

    result = {
        "status": "ok",
        "task": args.config_name,
        "checkpoint": str(checkpoint),
        "epoch": int(args.epoch),
        "seed": int(args.seed),
        "slots": slots,
        "parameters": {
            "candidate_count": candidate_count,
            "random_candidates": int(args.random_candidates),
            "actor_neighbors": int(args.actor_neighbors),
            "expert_neighbors": int(args.expert_neighbors),
            "expert_noise_std": float(args.expert_noise_std),
            "actor_noise_std": float(args.actor_noise_std),
            "topk": int(args.topk),
            "horizon": horizon,
            "action_block": action_block,
            "goal_offset_steps": int(cfg.eval.goal_offset_steps),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "aggregate": _aggregate_state_results(states),
        "states": states,
    }
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else run_dir / "diagnostics" / "stage_b_ranking" / f"epoch_{int(args.epoch)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(result, output_dir / "metrics.md")
    print(f"result={result_path}")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--config-name", required=True, choices=("pusht", "reacher"))
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--slots", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--random-candidates", type=int, default=16)
    parser.add_argument("--actor-neighbors", type=int, default=16)
    parser.add_argument("--expert-neighbors", type=int, default=8)
    parser.add_argument("--expert-noise-std", type=float, default=0.1)
    parser.add_argument("--actor-noise-std", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir")
    return parser


def main():
    run_diagnostic(build_parser().parse_args())


if __name__ == "__main__":
    main()
