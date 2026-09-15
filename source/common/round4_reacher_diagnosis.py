"""Deterministic, standalone diagnostics for the Round 4 Reacher step sweep.

The helpers in this module deliberately do not change the default Round 4
evaluator.  They expose the two seams that matter for the diagnosis:
action-flow integration with a fixed initial noise tensor and the existing
batch-one CEM update rule with a controlled initial mean.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .round3_protocol import evaluate_success


DIAGNOSTIC_SCHEMA_VERSION = 1
DEFAULT_FLOW_STEPS = (1, 2, 4, 5, 8, 10, 16, 32)
DEFAULT_HEUN_STEPS = (4, 8, 16, 32)
REACHER_THRESHOLD = 0.05


def _as_float_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return array


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "median": None, "p10": None, "p90": None}
    if not np.all(np.isfinite(array)):
        raise ValueError("diagnostic values must be finite")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


def action_statistics(actions: torch.Tensor | np.ndarray) -> dict[str, Any]:
    """Summarize normalized action paths without silently clipping them."""
    if torch.is_tensor(actions):
        array = actions.detach().float().cpu().numpy()
    else:
        array = np.asarray(actions, dtype=np.float64)
    if array.ndim < 2 or not np.all(np.isfinite(array)):
        raise ValueError("actions must be a finite tensor with path dimensions")
    flattened = array.reshape(-1, array.shape[-1])
    path_values = array.reshape(array.shape[0], -1)
    path_norm = np.linalg.norm(path_values, axis=1)
    return {
        "shape": [int(item) for item in array.shape],
        "element_count": int(flattened.size),
        "mean_abs": float(np.mean(np.abs(array))),
        "action_max_abs": float(np.max(np.abs(array))),
        "action_out_of_range_fraction": float(np.mean(np.abs(array) > 1.0)),
        "path_l2": _quantiles(path_norm),
        "path_variance": float(np.var(array)),
    }


def terminal_metrics(
    current: Sequence[float] | np.ndarray,
    goal: Sequence[float] | np.ndarray,
    *,
    actions: torch.Tensor | np.ndarray | None = None,
) -> dict[str, Any]:
    """Return the frozen Reacher predicate and its per-joint margin."""
    current_array = _as_float_array(current, name="current")
    goal_array = _as_float_array(goal, name="goal")
    if current_array.shape != goal_array.shape:
        raise ValueError("current and goal must have the same shape")
    error = np.abs(current_array - goal_array)
    max_joint_error = float(np.max(error))
    metrics: dict[str, Any] = {
        "max_joint_error": max_joint_error,
        "terminal_margin": float(REACHER_THRESHOLD - max_joint_error),
        "success": bool(evaluate_success("reacher", current_array, goal_array)),
        "l2_distance": float(np.linalg.norm(current_array - goal_array)),
    }
    if actions is not None:
        metrics.update(action_statistics(actions))
    return metrics


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash a tensor's CPU bytes together with shape and dtype metadata."""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def generate_noise(
    shape: Sequence[int], *, seed: int, device: torch.device | str = "cpu", dtype=torch.float32
) -> torch.Tensor:
    """Create one reproducible noise tensor to be reused by all samplers."""
    if any(int(item) < 1 for item in shape):
        raise ValueError("noise shape dimensions must be positive")
    target = torch.device(device)
    generator = torch.Generator(device=target).manual_seed(int(seed))
    return torch.randn(tuple(int(item) for item in shape), generator=generator, device=target, dtype=dtype)


@torch.no_grad()
def sample_action_flow(
    model,
    z0: torch.Tensor,
    goal_latent: torch.Tensor | None,
    noise: torch.Tensor,
    *,
    steps: int,
    integrator: str = "euler",
) -> torch.Tensor:
    """Sample Stage A with Euler or Heun while preserving the exact noise."""
    steps = int(steps)
    integrator = str(integrator).lower()
    if steps < 1:
        raise ValueError("steps must be positive")
    if integrator not in {"euler", "heun"}:
        raise ValueError("integrator must be 'euler' or 'heun'")
    expected = (z0.shape[0], int(model.action_horizon), int(model.action_dim))
    if tuple(noise.shape) != expected:
        raise ValueError(f"noise must have shape {expected}, got {tuple(noise.shape)}")
    noise = noise.to(device=z0.device, dtype=z0.dtype)
    if integrator == "euler":
        return model.sample_actions(
            z0,
            noise=noise,
            num_steps=steps,
            goal_latent=goal_latent,
        )

    actions = noise.clone()
    dt = 1.0 / float(steps)
    for step in range(steps):
        timestep = z0.new_full((z0.shape[0],), step / steps)
        first_velocity = model(
            z0,
            actions,
            timestep,
            mode="stage_a",
            goal_latent=goal_latent,
        )["action_velocity"]
        predictor = actions + dt * first_velocity
        next_timestep = z0.new_full((z0.shape[0],), (step + 1) / steps)
        second_velocity = model(
            z0,
            predictor,
            next_timestep,
            mode="stage_a",
            goal_latent=goal_latent,
        )["action_velocity"]
        actions = actions + 0.5 * dt * (first_velocity + second_velocity)
    return actions


def generate_cem_draws(
    *,
    num_contexts: int,
    iterations: int,
    samples: int,
    horizon: int,
    action_dim: int,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate paired CEM standard-normal draws for every context."""
    shape = (int(num_contexts), int(iterations), int(samples), int(horizon), int(action_dim))
    if any(item < 1 for item in shape):
        raise ValueError("CEM draw dimensions must be positive")
    target = torch.device(device)
    generator = torch.Generator(device=target).manual_seed(int(seed))
    return torch.randn(shape, generator=generator, device=target, dtype=dtype)


@torch.no_grad()
def run_cem_from_latents(
    model,
    z0: torch.Tensor,
    goal_latent: torch.Tensor,
    *,
    init_actions: torch.Tensor | None,
    random_draws: torch.Tensor,
    topk: int = 30,
    solver_batch_size: int = 1,
) -> dict[str, Any]:
    """Replay the installed CEM update using fixed draws and latent costs.

    The implementation mirrors ``stable_worldmodel.solver.CEMSolver`` for the
    configured Round 4 case: one environment per solver batch, the current
    mean as candidate zero, standard deviation updates, and latent Stage B
    costs.  It intentionally returns the initial and final elite costs needed
    to diagnose the warm-start basin.
    """
    if int(solver_batch_size) != 1:
        raise ValueError("the Reacher diagnosis requires solver_batch_size=1")
    if z0.ndim != 2 or goal_latent.shape != z0.shape:
        raise ValueError("z0 and goal_latent must have shape [B, latent_dim]")
    if random_draws.ndim != 5 or random_draws.shape[0] != z0.shape[0]:
        raise ValueError("random_draws must have shape [B,I,S,H,A]")
    batch, iterations, samples, horizon, action_dim = random_draws.shape
    expected = (batch, horizon, action_dim)
    if init_actions is None:
        init_actions = torch.zeros(expected, device=z0.device, dtype=z0.dtype)
    else:
        if tuple(init_actions.shape) != expected:
            raise ValueError(
                f"init_actions must have shape {expected}, got {tuple(init_actions.shape)}"
            )
        init_actions = init_actions.to(device=z0.device, dtype=z0.dtype)
    if not 1 <= int(topk) <= int(samples):
        raise ValueError("topk must be between one and the number of samples")

    mean = init_actions.clone()
    variance = torch.ones_like(mean)
    initial_cost: list[float] = []
    final_elite_cost: list[float] = []
    history: list[list[dict[str, float]]] = [[] for _ in range(batch)]

    for context in range(batch):
        context_mean = mean[context]
        context_variance = variance[context]
        context_initial = None
        context_final = None
        for iteration in range(iterations):
            candidates = (
                random_draws[context, iteration] * context_variance.unsqueeze(0)
                + context_mean.unsqueeze(0)
            )
            candidates[0] = context_mean
            costs = model.get_cost_from_latents(
                z0[context : context + 1],
                goal_latent[context : context + 1],
                candidates.unsqueeze(0),
            )[0]
            if context_initial is None:
                context_initial = float(costs[0].detach().cpu())
            topk_values, topk_indices = torch.topk(costs, k=int(topk), largest=False)
            elite = candidates[topk_indices]
            context_mean = elite.mean(dim=0)
            context_variance = elite.std(dim=0)
            context_final = float(topk_values.mean().detach().cpu())
            history[context].append(
                {
                    "iteration": float(iteration),
                    "initial_candidate_cost": float(costs[0].detach().cpu()),
                    "elite_cost": context_final,
                    "mean_norm": float(context_mean.norm().detach().cpu()),
                    "std_mean": float(context_variance.mean().detach().cpu()),
                }
            )
        initial_cost.append(float(context_initial))
        final_elite_cost.append(float(context_final))
        mean[context] = context_mean
        variance[context] = context_variance

    return {
        "actions": mean,
        "variance": variance,
        "initial_cost": initial_cost,
        "final_elite_cost": final_elite_cost,
        "history": history,
        "solver_batch_size": int(solver_batch_size),
        "iterations": int(iterations),
        "samples": int(samples),
        "topk": int(topk),
    }


def paired_success_comparison(
    baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare episode-level success vectors by stable episode/start keys."""
    def key(row: Mapping[str, Any]) -> tuple[Any, Any]:
        return row.get("episode_id", row.get("dataset_episode")), row.get("start_step")

    left = {key(row): bool(row["success"]) for row in baseline}
    right = {key(row): bool(row["success"]) for row in candidate}
    if set(left) != set(right):
        raise ValueError("paired episodes do not have identical episode_id/start_step keys")
    improved = sum(not left[item] and right[item] for item in left)
    regressed = sum(left[item] and not right[item] for item in left)
    unchanged_success = sum(left[item] and right[item] for item in left)
    unchanged_failure = sum(not left[item] and not right[item] for item in left)
    return {
        "episodes": len(left),
        "improved": int(improved),
        "regressed": int(regressed),
        "unchanged_success": int(unchanged_success),
        "unchanged_failure": int(unchanged_failure),
        "net_success_delta": int(improved - regressed),
    }


def aggregate_numeric_rows(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float | None]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return _quantiles(values)


__all__ = [
    "DEFAULT_FLOW_STEPS",
    "DEFAULT_HEUN_STEPS",
    "DIAGNOSTIC_SCHEMA_VERSION",
    "REACHER_THRESHOLD",
    "action_statistics",
    "aggregate_numeric_rows",
    "generate_cem_draws",
    "generate_noise",
    "paired_success_comparison",
    "run_cem_from_latents",
    "sample_action_flow",
    "tensor_sha256",
    "terminal_metrics",
]
