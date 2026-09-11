"""Environment-facing adapter for the integrated LeFlow solver."""

from __future__ import annotations

from pathlib import Path

import stable_worldmodel as swm
from omegaconf import OmegaConf

from source.model.leflow.latent_planner import (
    LatentPlannerRuntime,
    LearnedLatentPathSolver,
)
from source.policy.lewm import _to_container


_DEFAULTS = {
    "batch_size": 1,
    "num_samples": 64,
    "flow_steps": 16,
    "score_mode": "rollout_goal",
    "goal_weight": 1.0,
    "consistency_weight": 0.0,
    "smoothness_weight": 0.0,
    "history_size": 3,
}


def _solver_kwargs(solver_cfg):
    values = dict(_DEFAULTS)
    if solver_cfg is not None:
        if OmegaConf.is_config(solver_cfg):
            configured = OmegaConf.to_container(solver_cfg, resolve=False)
        else:
            configured = _to_container(solver_cfg)
        if configured:
            configured.pop("_target_", None)
            configured.pop("checkpoint", None)
            configured.pop("device", None)
            configured.pop("seed", None)
            values.update(configured)
    return values


def make_leflow_policy(
    policy_or_model,
    *,
    solver_cfg=None,
    plan_config,
    process=None,
    transform=None,
    device="cuda",
    checkpoint: str | Path | None = None,
):
    """Build a stable-worldmodel policy backed by a LeFlow runtime.

    ``policy_or_model`` may be an already-loaded :class:`LatentPlannerRuntime`
    or a checkpoint path.  Keeping this seam independent of Hydra lets both
    the legacy evaluator and the auditable Round 3 evaluator use the same
    adapter.
    """
    runtime = policy_or_model
    if isinstance(runtime, str | Path):
        checkpoint = checkpoint or runtime
        runtime = None
    if hasattr(runtime, "model") and isinstance(runtime.model, LatentPlannerRuntime):
        runtime = runtime.model
    if runtime is not None and not isinstance(runtime, LatentPlannerRuntime):
        raise TypeError(
            "LeFlow policy construction expects a LatentPlannerRuntime or checkpoint path"
        )

    solver = LearnedLatentPathSolver(
        checkpoint=checkpoint,
        model=runtime,
        device=device,
        **_solver_kwargs(solver_cfg),
    )
    config_values = _to_container(plan_config)
    config = (
        plan_config
        if isinstance(plan_config, swm.PlanConfig)
        else swm.PlanConfig(**config_values)
    )
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
    )


__all__ = ["make_leflow_policy"]
