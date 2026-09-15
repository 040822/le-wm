"""面向环境的 Fast-LeWAM Stage A、B、C 推理 policy 适配器。"""

from collections import deque
import time

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from torch import nn
from stable_worldmodel.solver.cem import prepare_init_action

from source.common.round4_action_bounds import (
    NormalizedActionBounds,
    compute_normalized_action_bounds,
    normalized_action_stats,
    project_normalized_actions,
)
from source.policy.lewm import _to_container

_STAGE_B_Z0 = "_fast_lewam_stage_b_z0"
_STAGE_B_GOAL = "_fast_lewam_stage_b_goal"
_STAGE_B_CONTEXT_PIXELS = "_fast_lewam_stage_b_context_pixels"
_STAGE_B_CONTEXT_GOAL = "_fast_lewam_stage_b_context_goal"


def _validate_action_bound_mode(mode, *, allowed):
    value = str(mode).lower()
    if value not in set(allowed):
        raise ValueError(
            f"action_bound_mode must be one of {tuple(allowed)!r}, got {mode!r}"
        )
    return value


def _projection_summary(
    before: torch.Tensor,
    after: torch.Tensor,
    bounds: NormalizedActionBounds,
) -> dict:
    """Summarize one normalized-action projection without retaining tensors."""
    if before.shape != after.shape:
        raise ValueError("projection inputs must have identical shapes")
    delta = (after - before).detach().float().abs()
    before_stats = normalized_action_stats(before, bounds)
    after_stats = normalized_action_stats(after, bounds)
    return {
        "before": before_stats,
        "after": after_stats,
        "changed_fraction": float((delta > 1e-6).float().mean().cpu()),
        "mean_abs_delta": float(delta.mean().cpu()),
        "max_abs_delta": float(delta.max().cpu()),
    }


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


class ProjectedCEMSolver:
    """CEM adapter that projects candidates before scoring and updating elites."""

    def __init__(self, solver, *, projection_mode="clip"):
        if str(projection_mode).lower() != "clip":
            raise ValueError("ProjectedCEMSolver currently supports clip only")
        self._solver = solver
        self.model = solver.model
        self.projection_mode = "clip"
        self.action_bounds = None
        self.last_action_bound_projection = None

    def __getattr__(self, name):
        return getattr(self._solver, name)

    @property
    def action_dim(self):
        return self._solver.action_dim

    @property
    def n_envs(self):
        return self._solver.n_envs

    @property
    def horizon(self):
        return self._solver.horizon

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def configure(self, *, action_space, n_envs, config):
        self._solver.configure(
            action_space=action_space,
            n_envs=n_envs,
            config=config,
        )

    def set_action_bounds(self, bounds: NormalizedActionBounds):
        if not isinstance(bounds, NormalizedActionBounds):
            raise TypeError("bounds must be a NormalizedActionBounds instance")
        bounds.for_action_dim(self.action_dim)
        self.action_bounds = bounds

    def _require_bounds(self) -> NormalizedActionBounds:
        if self.action_bounds is None:
            raise RuntimeError("set_env must configure action bounds before solving")
        return self.action_bounds

    @staticmethod
    def _candidate_bound_mask(
        candidates: torch.Tensor,
        bounds: NormalizedActionBounds,
    ) -> torch.Tensor:
        low, high = bounds.for_action_dim(int(candidates.shape[-1]))
        low_tensor = torch.as_tensor(
            low,
            device=candidates.device,
            dtype=candidates.dtype,
        ).reshape(1, 1, 1, -1)
        high_tensor = torch.as_tensor(
            high,
            device=candidates.device,
            dtype=candidates.dtype,
        ).reshape(1, 1, 1, -1)
        return (candidates < low_tensor - 1e-6) | (
            candidates > high_tensor + 1e-6
        )

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        """Run the installed CEM algorithm with projected candidate elites."""
        bounds = self._require_bounds()
        base = self._solver
        start_time = time.time()
        outputs = {"costs": [], "mean": [], "var": []}
        if hasattr(base.model, "last_action_bound_projection"):
            base.model.last_action_bound_projection = None

        total_envs = len(next(iter(info_dict.values())))
        prepared_init = prepare_init_action(
            base.model,
            info_dict,
            init_action,
            base.horizon,
            n_envs=total_envs,
            action_dim=base.action_dim,
        )
        projected_init = project_normalized_actions(
            prepared_init,
            bounds,
            mode="clip",
        )
        initial_projected_init = projected_init.detach().clone()
        mean, var = base.init_action_distrib(total_envs, projected_init)
        mean = mean.to(base.device)
        var = var.to(base.device)

        for cb in base.callbacks:
            cb.reset()

        raw_violation_count = torch.zeros((), device=mean.device, dtype=torch.float64)
        projected_violation_count = torch.zeros(
            (), device=mean.device, dtype=torch.float64
        )
        changed_count = torch.zeros((), device=mean.device, dtype=torch.float64)
        candidate_count = 0

        for start_idx in range(0, total_envs, base.batch_size):
            end_idx = min(start_idx + base.batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]

            expanded_infos = {}
            for key, value in info_dict.items():
                value_batch = value[start_idx:end_idx]
                if torch.is_tensor(value_batch):
                    target_dtype = (
                        base.dtype if value_batch.is_floating_point() else None
                    )
                    value_batch = value_batch.to(
                        device=base.device,
                        dtype=target_dtype,
                    ).unsqueeze(1).expand(
                        current_bs,
                        base.num_samples,
                        *value_batch.shape[1:],
                    )
                elif isinstance(value_batch, np.ndarray):
                    value_batch = np.repeat(
                        value_batch[:, None, ...],
                        base.num_samples,
                        axis=1,
                    )
                expanded_infos[key] = value_batch

            final_batch_cost = None
            for cb in base.callbacks:
                cb.start_batch()

            for step in range(base.n_steps):
                raw_candidates = torch.randn(
                    current_bs,
                    base.num_samples,
                    base.horizon,
                    base.action_dim,
                    generator=base.torch_gen,
                    device=base.device,
                    dtype=base.dtype,
                )
                raw_candidates = (
                    raw_candidates * batch_var.unsqueeze(1)
                    + batch_mean.unsqueeze(1)
                )
                raw_candidates[:, 0] = batch_mean
                candidates = project_normalized_actions(
                    raw_candidates,
                    bounds,
                    mode="clip",
                )
                candidates[:, 0] = project_normalized_actions(
                    batch_mean,
                    bounds,
                    mode="clip",
                )

                raw_mask = self._candidate_bound_mask(raw_candidates, bounds)
                projected_mask = self._candidate_bound_mask(candidates, bounds)
                raw_violation_count += raw_mask.sum()
                projected_violation_count += projected_mask.sum()
                changed_count += (raw_candidates - candidates).abs().gt(1e-6).sum()
                candidate_count += raw_candidates.numel()

                costs = base.model.get_cost(expanded_infos, candidates)
                if not isinstance(costs, torch.Tensor):
                    raise TypeError("CEM model must return a torch.Tensor of costs")
                if costs.ndim != 2 or costs.shape != (
                    current_bs,
                    base.num_samples,
                ):
                    raise ValueError(
                        "CEM model returned costs with unexpected shape "
                        f"{tuple(costs.shape)}"
                    )

                topk_values, topk_indices = torch.topk(
                    costs,
                    k=base.topk,
                    dim=1,
                    largest=False,
                )
                batch_indices = torch.arange(
                    current_bs,
                    device=base.device,
                ).unsqueeze(1).expand(-1, base.topk)
                topk_candidates = candidates[batch_indices, topk_indices]
                prev_mean = batch_mean
                prev_var = batch_var
                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1)

                for cb in base.callbacks:
                    cb(
                        step=step,
                        candidates=candidates,
                        costs=costs,
                        topk_vals=topk_values,
                        topk_inds=topk_indices,
                        topk_candidates=topk_candidates,
                        mean=batch_mean,
                        var=batch_var,
                        prev_mean=prev_mean,
                        prev_var=prev_var,
                    )
                final_batch_cost = topk_values.mean(dim=1).cpu().tolist()

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(final_batch_cost)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        if base.callbacks:
            outputs["callbacks"] = {}
            for cb in base.callbacks:
                cb.end_solve()
                outputs["callbacks"][cb.output_key] = cb.history

        model_projection = getattr(
            base.model,
            "last_action_bound_projection",
            None,
        )
        projection = {
            "mode": self.projection_mode,
            "bounds": bounds.metadata(action_dim=base.action_dim),
            "warm_start": (
                model_projection
                if model_projection is not None
                else _projection_summary(
                    prepared_init,
                    initial_projected_init,
                    bounds,
                )
            ),
            "raw_candidate_violation_fraction": float(
                (raw_violation_count / max(1, candidate_count)).cpu()
            ),
            "projected_candidate_violation_fraction": float(
                (projected_violation_count / max(1, candidate_count)).cpu()
            ),
            "candidate_changed_fraction": float(
                (changed_count / max(1, candidate_count)).cpu()
            ),
            "final_actions": normalized_action_stats(mean, bounds),
        }
        self.last_action_bound_projection = projection
        outputs["action_bound_projection"] = projection
        print(f"CEM solve time: {time.time() - start_time:.4f} seconds")
        return outputs


class ActorWarmStartModelView(StageBModelView):
    """为 Stage B planner 同时暴露 latent cost 和确定性的 Stage A 初始化。"""

    def __init__(
        self,
        model,
        *,
        seed=0,
        inference_steps=None,
        action_scale=1.0,
        integrator="euler",
        action_projection="none",
    ):
        super().__init__(model)
        self.seed = int(seed)
        self.inference_steps = inference_steps
        self.action_scale = float(action_scale)
        if not np.isfinite(self.action_scale) or self.action_scale <= 0.0:
            raise ValueError("action_scale must be a finite positive number")
        self.integrator = str(integrator).lower()
        if self.integrator not in {"euler", "heun"}:
            raise ValueError("integrator must be 'euler' or 'heun'")
        self.action_projection = _validate_action_bound_mode(
            action_projection,
            allowed=("none", "clip", "global_scale"),
        )
        self.action_bounds = None
        self.last_action_bound_projection = None
        self._generators = {}

    def set_action_bounds(self, bounds: NormalizedActionBounds):
        if not isinstance(bounds, NormalizedActionBounds):
            raise TypeError("bounds must be a NormalizedActionBounds instance")
        bounds.for_action_dim(self.model.action_dim)
        self.action_bounds = bounds

    def _project_actions(self, actions):
        if self.action_projection == "none":
            self.last_action_bound_projection = None
            return actions
        if self.action_bounds is None:
            raise RuntimeError(
                "set_env must configure action bounds before generating actions"
            )
        projected = project_normalized_actions(
            actions,
            self.action_bounds,
            mode=self.action_projection,
        )
        self.last_action_bound_projection = _projection_summary(
            actions,
            projected,
            self.action_bounds,
        )
        return projected

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
        actions = self.model.get_action_from_latents(
            z0,
            goal_latent,
            horizon=horizon,
            prefix_actions=prefix_actions,
            generator=self._generator(device),
            num_steps=self.inference_steps,
            integrator=self.integrator,
        )
        return self._project_actions(actions * self.action_scale)



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

    def __init__(
        self,
        *args,
        fast_model,
        action_block,
        action_bound_mode="none",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fast_model = fast_model
        self.fast_action_block = int(action_block)
        self.action_bound_mode = _validate_action_bound_mode(
            action_bound_mode,
            allowed=("none", "warm_start_clip", "candidate_clip", "warm_start_scale"),
        )
        self.action_bounds = None

    def set_env(self, env):
        _validate_action_dim(self.fast_model, env, self.fast_action_block)
        super().set_env(env)
        if self.action_bound_mode == "none":
            return
        processor = self.process.get("action") if self.process else None
        if processor is None:
            raise ValueError(
                "action-bound projection requires the evaluation action processor"
            )
        self.action_bounds = compute_normalized_action_bounds(
            env.single_action_space,
            processor,
            action_block=self.fast_action_block,
        )
        if hasattr(self.solver, "set_action_bounds"):
            self.solver.set_action_bounds(self.action_bounds)
        solver_model = getattr(self.solver, "model", None)
        if hasattr(solver_model, "set_action_bounds"):
            solver_model.set_action_bounds(self.action_bounds)


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
        integrator="euler",
        action_bound_mode="none",
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
        self.integrator = str(integrator).lower()
        if self.integrator not in {"euler", "heun"}:
            raise ValueError("integrator must be 'euler' or 'heun'")
        self.action_bound_mode = _validate_action_bound_mode(
            action_bound_mode,
            allowed=("none", "clip", "global_scale"),
        )
        self.action_bounds = None
        self._goal_indices = None
        self._action_buffer = None
        self._generators = {}
        self.planning_events: list[dict] = []

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
        if self.action_bound_mode != "none":
            processor = self.process.get("action") if self.process else None
            if processor is None:
                raise ValueError(
                    "action-bound projection requires the evaluation action processor"
                )
            self.action_bounds = compute_normalized_action_bounds(
                env.single_action_space,
                processor,
                action_block=self.action_block,
            )

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
            timed_projection = self.action_bound_mode != "none"
            encode_started = time.perf_counter() if timed_projection else None
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
            if encode_started is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                encode_seconds = time.perf_counter() - encode_started
            else:
                encode_seconds = 0.0
            generator = self._generator(device)
            proposal_started = time.perf_counter() if timed_projection else None
            with torch.no_grad():
                if self.mode == "stage_a":
                    chunk = self.model.sample_actions(
                        z0,
                        num_steps=self.inference_steps,
                        generator=generator,
                        goal_latent=goal_latent,
                        integrator=self.integrator,
                    )
                else:
                    chunk = self.model.sample_joint(
                        z0,
                        num_steps=self.inference_steps,
                        generator=generator,
                        integrator=self.integrator,
                    )["actions"]

            if proposal_started is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                proposal_seconds = time.perf_counter() - proposal_started
            else:
                proposal_seconds = 0.0

            raw_chunk = chunk
            if self.action_bound_mode != "none":
                if self.action_bounds is None:
                    raise RuntimeError(
                        "set_env must configure action bounds before planning"
                    )
                chunk = project_normalized_actions(
                    raw_chunk,
                    self.action_bounds,
                    mode=self.action_bound_mode,
                )
                projection = _projection_summary(
                    raw_chunk,
                    chunk,
                    self.action_bounds,
                )
            else:
                projection = None

            keep_blocks = min(
                self.receding_horizon_blocks, self.model.action_horizon
            )
            base_action_dim = self.model.action_dim // self.action_block
            plan = chunk[:, :keep_blocks].reshape(
                len(replan), keep_blocks * self.action_block, base_action_dim
            )
            if projection is not None:
                flow_steps = int(
                    16
                    if self.inference_steps is None
                    else self.inference_steps
                )
                self.planning_events.append(
                    {
                        "proposal_source": self.mode,
                        "candidate_count": 1,
                        "flow_steps": flow_steps,
                        "action_flow_steps": flow_steps,
                        "action_flow_integrator": self.integrator,
                        "verifier": "none",
                        "selection_rule": "direct",
                        "environment_batch_size": len(replan),
                        "forward_count": int(
                            flow_steps
                            * (2 if self.integrator == "heun" else 1)
                        ),
                        "action_bound_mode": self.action_bound_mode,
                        "action_bounds": self.action_bounds.metadata(
                            action_dim=self.model.action_dim
                        ),
                        "action_bound_projection": projection,
                        "encode_seconds": float(encode_seconds),
                        "proposal_seconds": float(proposal_seconds),
                        "flow_seconds": float(proposal_seconds),
                        "verify_seconds": 0.0,
                        "peak_memory_bytes": None,
                    }
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
    actor_warm_start_scale=1.0,
    action_flow_integrator="euler",
    action_bound_mode="none",
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
        action_bound_mode = _validate_action_bound_mode(
            action_bound_mode,
            allowed=("none", "warm_start_clip", "candidate_clip", "warm_start_scale"),
        )
        if action_bound_mode != "none" and not actor_warm_start:
            raise ValueError(
                "action-bound variants for stage_b require actor_warm_start=True"
            )
        warm_start_projection = {
            "none": "none",
            "warm_start_clip": "clip",
            "candidate_clip": "clip",
            "warm_start_scale": "global_scale",
        }[action_bound_mode]
        solver_model = (
            ActorWarmStartModelView(
                model,
                seed=seed,
                inference_steps=inference_steps,
                action_scale=actor_warm_start_scale,
                integrator=action_flow_integrator,
                action_projection=warm_start_projection,
            )
            if actor_warm_start
            else StageBModelView(model)
        )
        solver = hydra.utils.instantiate(solver_cfg, model=solver_model)
        if action_bound_mode == "candidate_clip":
            solver = ProjectedCEMSolver(solver, projection_mode="clip")
        return FastLeWAMStageBPolicy(
            solver=solver,
            config=config,
            process=process,
            transform=transform,
            fast_model=model,
            action_block=config.action_block,
            action_bound_mode=action_bound_mode,
        )
    if mode not in {"stage_a", "stage_c"}:
        raise ValueError("Fast-LeWAM policy mode must be stage_a, stage_b, or stage_c")
    action_bound_mode = _validate_action_bound_mode(
        action_bound_mode,
        allowed=("none", "clip", "global_scale"),
    )
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
        integrator=action_flow_integrator,
        action_bound_mode=action_bound_mode,
    )


__all__ = [
    "ActorWarmStartModelView",
    "FastLeWAMChunkPolicy",
    "FastLeWAMStageBPolicy",
    "ProjectedCEMSolver",
    "StageBModelView",
    "make_fast_lewam_policy",
]
