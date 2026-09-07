"""面向环境的 Fast-LeWAM Stage A、B、C 推理 policy 适配器。"""

from collections import deque

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from torch import nn

from source.policy.lewm import _to_container

_STAGE_B_Z0 = "_fast_lewam_stage_b_z0"
_STAGE_B_GOAL = "_fast_lewam_stage_b_goal"
_STAGE_B_CONTEXT_PIXELS = "_fast_lewam_stage_b_context_pixels"
_STAGE_B_CONTEXT_GOAL = "_fast_lewam_stage_b_context_goal"


class StageBModelView(nn.Module):
    """仅暴露 latent cost 的 Stage B 模型视图，阻止 solver 调用 actor warm start。"""

    def __init__(self, model):
        """保存底层 FastLeWAM 模型，但不转发其动作生成接口。"""
        super().__init__()
        self.model = model

    def get_cost(self, info_dict, action_candidates):
        """Reuse one exactly-matching visual context across CEM iterations."""
        if action_candidates.ndim != 4:
            return self.model.get_cost(info_dict, action_candidates)
        batch, samples = action_candidates.shape[:2]
        pixels = info_dict.get("pixels")
        goal = info_dict.get("goal")
        shared_observations = (
            torch.is_tensor(pixels)
            and torch.is_tensor(goal)
            and pixels.ndim >= 5
            and goal.ndim >= 5
            and pixels.shape[:2] == (batch, samples)
            and goal.shape[:2] == (batch, samples)
            and pixels.stride(1) == 0
            and goal.stride(1) == 0
        )
        if not shared_observations:
            return self.model.get_cost(info_dict, action_candidates)

        current = self.model._last_frame(pixels[:, 0])
        goal_frame = self.model._last_frame(goal[:, 0])
        cached_current = info_dict.get(_STAGE_B_CONTEXT_PIXELS)
        cached_goal = info_dict.get(_STAGE_B_CONTEXT_GOAL)
        if torch.is_tensor(cached_current) and cached_current.ndim == current.ndim + 1:
            cached_current = cached_current[:, 0]
        if torch.is_tensor(cached_goal) and cached_goal.ndim == goal_frame.ndim + 1:
            cached_goal = cached_goal[:, 0]
        valid_cache = (
            _STAGE_B_Z0 in info_dict
            and _STAGE_B_GOAL in info_dict
            and torch.is_tensor(cached_current)
            and torch.is_tensor(cached_goal)
            and torch.equal(cached_current, current)
            and torch.equal(cached_goal, goal_frame)
        )
        if valid_cache:
            z0 = info_dict[_STAGE_B_Z0]
            goal_latent = info_dict[_STAGE_B_GOAL]
            if z0.ndim == 3:
                z0 = z0[:, 0]
            if goal_latent.ndim == 3:
                goal_latent = goal_latent[:, 0]
            return self.model.get_cost_from_latents(
                z0, goal_latent, action_candidates
            )

        z0 = self.model.encode_pixels(current)
        goal_latent = self.model.encode_pixels(goal_frame)

        info_dict[_STAGE_B_Z0] = z0
        info_dict[_STAGE_B_GOAL] = goal_latent
        info_dict[_STAGE_B_CONTEXT_PIXELS] = current.detach().clone()
        info_dict[_STAGE_B_CONTEXT_GOAL] = goal_frame.detach().clone()
        return self.model.get_cost_from_latents(z0, goal_latent, action_candidates)


class ActorWarmStartModelView(StageBModelView):
    """为 Stage B planner 同时暴露 latent cost 和确定性的 Stage A 初始化。"""

    def __init__(self, model, *, seed=0, inference_steps=None):
        super().__init__(model)
        self.seed = int(seed)
        self.inference_steps = inference_steps
        self._generators = {}

    def _generator(self, device):
        key = str(device)
        if key not in self._generators:
            self._generators[key] = torch.Generator(device=device).manual_seed(
                self.seed
            )
        return self._generators[key]

    def get_action(self, info, horizon=1, prefix_actions=None):
        device = next(self.model.parameters()).device
        device_info = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in info.items()
        }
        current = self.model._last_frame(device_info["pixels"])
        goal = self.model._last_frame(device_info["goal"])
        z0 = self.model.encode_pixels(current)
        goal_latent = self.model.encode_pixels(goal)

        info[_STAGE_B_Z0] = z0
        info[_STAGE_B_GOAL] = goal_latent
        info[_STAGE_B_CONTEXT_PIXELS] = current.detach().clone()
        info[_STAGE_B_CONTEXT_GOAL] = goal.detach().clone()
        return self.model.get_action_from_latents(
            z0,
            goal_latent,
            horizon=horizon,
            prefix_actions=prefix_actions,
            generator=self._generator(device),
            num_steps=self.inference_steps,
        )



def _validate_action_dim(model, env, action_block):
    base_action_dim = int(np.prod(env.single_action_space.shape))
    expected = int(action_block) * base_action_dim
    if model.action_dim != expected:
        raise ValueError(
            f"model action_dim={model.action_dim} does not match "
            f"action_block({action_block}) * "
            f"env_action_dim({base_action_dim})={expected}"
        )


class FastLeWAMStageBPolicy(swm.policy.WorldModelPolicy):
    """Solver-backed Stage B policy with Fast model shape validation."""

    def __init__(self, *args, fast_model, action_block, **kwargs):
        super().__init__(*args, **kwargs)
        self.fast_model = fast_model
        self.fast_action_block = int(action_block)

    def set_env(self, env):
        _validate_action_dim(self.fast_model, env, self.fast_action_block)
        super().set_env(env)


class FastLeWAMChunkPolicy(swm.policy.BasePolicy):
    """缓存生成的动作块，并按 receding horizon 执行若干步后重新规划。"""

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
        goal_mode="correct",
    ):
        """构造直接动作 policy，并冻结模型、记录动作块尺寸和采样参数。"""
        super().__init__()
        if mode not in {"stage_a", "stage_c"}:
            raise ValueError("chunk policy mode must be stage_a or stage_c")
        if goal_mode not in {"correct", "cyclic_shift"}:
            raise ValueError("goal_mode must be correct or cyclic_shift")
        if action_block < 1 or receding_horizon_blocks < 1:
            raise ValueError(
                "action_block and receding_horizon_blocks must be positive"
            )
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
        self.goal_mode = goal_mode
        self._goal_indices = None
        self._action_buffer = None
        self._generators = {}

    def _generator(self, device):
        """按设备懒创建可复现的 torch.Generator，并在多次规划间保留随机状态。"""
        key = str(device)
        if key not in self._generators:
            self._generators[key] = torch.Generator(device=device).manual_seed(self.seed)
        return self._generators[key]

    def set_seed(self, seed):
        """更新采样种子，并清空各设备上已推进过状态的随机数生成器。"""
        self.seed = seed
        self._generators.clear()
        if self._goal_indices is not None and self.goal_mode == "cyclic_shift":
            num_envs = self.env.num_envs
            if num_envs < 2:
                raise ValueError("cyclic_shift goal ablation requires num_envs >= 2")
            offset = 1 + self.seed % (num_envs - 1)
            self._goal_indices = (
                np.arange(num_envs, dtype=np.int64) + offset
            ) % num_envs

    def set_env(self, env):
        """绑定向量环境、初始化每个环境的动作队列，并校验动作维度。"""
        self.env = env
        self._action_buffer = [deque() for _ in range(env.num_envs)]
        if self.goal_mode == "cyclic_shift":
            if env.num_envs < 2:
                raise ValueError("cyclic_shift goal ablation requires num_envs >= 2")
            offset = 1 + self.seed % (env.num_envs - 1)
            self._goal_indices = (
                np.arange(env.num_envs, dtype=np.int64) + offset
            ) % env.num_envs
        else:
            self._goal_indices = np.arange(env.num_envs, dtype=np.int64)
        _validate_action_dim(self.model, env, self.action_block)

    def _slice_info(self, info, indices):
        """只截取需要重新规划的环境信息，同时兼容 tensor、数组和列表。"""
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
        """为动作队列已空的环境批量规划，并从每个存活环境队列弹出一步动作。"""
        if self._action_buffer is None:
            raise RuntimeError("set_env must be called before get_action")
        needs_flush = info_dict.get("_needs_flush")
        if needs_flush is not None:
            for index, flush in enumerate(needs_flush):
                if flush:
                    self._action_buffer[index].clear()

        terminated = info_dict.get("terminated")
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
            selected_raw = self._slice_info(info_dict, replan)
            token_goal = (
                self.mode == "stage_a"
                and self.model.stage_a_goal_injection == "token"
            )
            if token_goal:
                if "goal" not in info_dict:
                    raise ValueError(
                        "goal observations are required for Stage A token mode"
                    )
                goal_indices = self._goal_indices[np.asarray(replan)]
                selected_raw["goal"] = self._slice_info(
                    info_dict, goal_indices
                )["goal"]
            selected = self._prepare_info(selected_raw)
            device = next(self.model.parameters()).device
            for key, value in selected.items():
                if torch.is_tensor(value):
                    selected[key] = value.to(device)
            current_frames = self.model._last_frame(selected["pixels"])
            goal_latent = None
            if token_goal:
                goal_frames = self.model._last_frame(selected["goal"])
                encoded = self.model.encode_pixels(
                    torch.cat((current_frames, goal_frames))
                )
                z0, goal_latent = encoded.split(len(replan))
            else:
                z0 = self.model.encode_pixels(current_frames)
            generator = self._generator(device)
            with torch.no_grad():
                if self.mode == "stage_a":
                    chunk = self.model.sample_actions(
                        z0,
                        num_steps=self.inference_steps,
                        generator=generator,
                        goal_latent=goal_latent,
                    )
                else:
                    chunk = self.model.sample_joint(
                        z0,
                        num_steps=self.inference_steps,
                        generator=generator,
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
    inference_steps=None,
    seed=0,
    goal_mode="correct",
    actor_warm_start=False,
):
    """按 mode 创建 Stage A/C 直接动作 policy 或 Stage B solver-backed policy。"""
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
        solver_model = (
            ActorWarmStartModelView(
                model,
                seed=seed,
                inference_steps=inference_steps,
            )
            if actor_warm_start
            else StageBModelView(model)
        )
        solver = hydra.utils.instantiate(solver_cfg, model=solver_model)
        return FastLeWAMStageBPolicy(
            solver=solver,
            config=config,
            process=process,
            transform=transform,
            fast_model=model,
            action_block=config.action_block,
        )
    if mode not in {"stage_a", "stage_c"}:
        raise ValueError("Fast-LeWAM policy mode must be stage_a, stage_b, or stage_c")
    return FastLeWAMChunkPolicy(
        model=model,
        process=process,
        transform=transform,
        mode=mode,
        action_block=config.action_block,
        receding_horizon_blocks=config.receding_horizon,
        inference_steps=inference_steps,
        seed=seed,
        goal_mode=goal_mode,
    )


__all__ = [
    "ActorWarmStartModelView",
    "FastLeWAMChunkPolicy",
    "FastLeWAMStageBPolicy",
    "StageBModelView",
    "make_fast_lewam_policy",
]
