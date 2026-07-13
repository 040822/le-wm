"""Environment-facing adapters for Fast-LeWAM Stage A, B, and C modes."""

from collections import deque

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from torch import nn

from source.policy.lewm import _to_container


class StageBModelView(nn.Module):
    """Cost-only view that prevents solver actor warm-start in Stage B mode."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_cost(self, info_dict, action_candidates):
        return self.model.get_cost(info_dict, action_candidates)


class FastLeWAMChunkPolicy(swm.policy.BasePolicy):
    """Execute a configurable number of generated action blocks before replanning."""

    def __init__(
        self,
        model,
        process=None,
        transform=None,
        mode="stage_a",
        action_block=1,
        receding_horizon_blocks=1,
        inference_steps=None,
        seed=0,
    ):
        super().__init__()
        if mode not in {"stage_a", "stage_c"}:
            raise ValueError("chunk policy mode must be stage_a or stage_c")
        if action_block < 1 or receding_horizon_blocks < 1:
            raise ValueError("action_block and receding_horizon_blocks must be positive")
        self.type = "fast_lewam"
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.process = process or {}
        self.transform = transform or {}
        self.mode = mode
        self.action_block = action_block
        self.receding_horizon_blocks = receding_horizon_blocks
        self.inference_steps = inference_steps
        self.seed = seed
        self._action_buffer = None

    def set_env(self, env):
        self.env = env
        self._action_buffer = [deque() for _ in range(env.num_envs)]
        base_action_dim = int(np.prod(env.single_action_space.shape))
        expected = self.action_block * base_action_dim
        if self.model.action_dim != expected:
            raise ValueError(
                f"model action_dim={self.model.action_dim} does not match "
                f"action_block({self.action_block}) * env_action_dim({base_action_dim})={expected}"
            )

    def _slice_info(self, info, indices):
        sliced = {}
        for key, value in info.items():
            if torch.is_tensor(value):
                sliced[key] = value[indices]
            elif isinstance(value, np.ndarray):
                sliced[key] = value[indices]
            elif isinstance(value, list):
                sliced[key] = [value[index] for index in indices]
            else:
                sliced[key] = value
        return sliced

    def get_action(self, info_dict, **kwargs):
        if self._action_buffer is None:
            raise RuntimeError("set_env must be called before get_action")
        info = self._prepare_info(info_dict)
        needs_flush = info.pop("_needs_flush", None)
        if needs_flush is not None:
            for index, flush in enumerate(needs_flush):
                if flush:
                    self._action_buffer[index].clear()

        terminated = info.get("terminated")
        dead = (
            np.asarray(terminated, dtype=bool)
            if terminated is not None
            else np.zeros(self.env.num_envs, dtype=bool)
        )
        replan = [
            index
            for index in range(self.env.num_envs)
            if not dead[index] and not self._action_buffer[index]
        ]
        if replan:
            selected = self._slice_info(info, replan)
            device = next(self.model.parameters()).device
            for key, value in selected.items():
                if torch.is_tensor(value):
                    selected[key] = value.to(device)
            z0 = self.model.encode_pixels(
                self.model._last_frame(selected["pixels"])
            )
            with torch.no_grad():
                if self.mode == "stage_a":
                    chunk = self.model.sample_actions(
                        z0, num_steps=self.inference_steps
                    )
                else:
                    chunk = self.model.sample_joint(
                        z0, num_steps=self.inference_steps
                    )["actions"]

            keep_blocks = min(
                self.receding_horizon_blocks, self.model.action_horizon
            )
            base_action_dim = self.model.action_dim // self.action_block
            plan = chunk[:, :keep_blocks].reshape(
                len(replan), keep_blocks * self.action_block, base_action_dim
            )
            for row, env_index in enumerate(replan):
                self._action_buffer[env_index].extend(plan[row].cpu())

        base_action_dim = int(np.prod(self.env.single_action_space.shape))
        action = torch.full((self.env.num_envs, base_action_dim), float("nan"))
        for index in range(self.env.num_envs):
            if not dead[index]:
                action[index] = self._action_buffer[index].popleft()
        result = action.reshape(*self.env.action_space.shape).numpy()
        if "action" in self.process:
            result = self.process["action"].inverse_transform(result)
        return result


def make_fast_lewam_policy(
    policy_or_model,
    solver_cfg,
    plan_config,
    process,
    transform,
    device="cuda",
    mode="stage_a",
    direct_receding_horizon=1,
    inference_steps=None,
):
    """Build a direct-action or solver-backed environment policy."""
    model = getattr(policy_or_model, "model", policy_or_model)
    if device is not None:
        model = model.to(device)
    model = model.eval()
    model.requires_grad_(False)
    plan_kwargs = _to_container(plan_config)
    config = (
        plan_config
        if isinstance(plan_config, swm.PlanConfig)
        else swm.PlanConfig(**plan_kwargs)
    )
    if mode == "stage_b":
        solver_model = StageBModelView(model)
        solver = hydra.utils.instantiate(solver_cfg, model=solver_model)
        return swm.policy.WorldModelPolicy(
            solver=solver,
            config=config,
            process=process,
            transform=transform,
        )
    if mode not in {"stage_a", "stage_c"}:
        raise ValueError("Fast-LeWAM policy mode must be stage_a, stage_b, or stage_c")
    return FastLeWAMChunkPolicy(
        model=model,
        process=process,
        transform=transform,
        mode=mode,
        action_block=config.action_block,
        receding_horizon_blocks=direct_receding_horizon,
        inference_steps=inference_steps,
    )


__all__ = ["FastLeWAMChunkPolicy", "StageBModelView", "make_fast_lewam_policy"]
