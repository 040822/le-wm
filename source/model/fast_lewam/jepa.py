"""Fast-LeWAM 原型共用的 Action/World DiT 模型。"""

from __future__ import annotations

import torch
from torch import nn

from source.model.fast_lewam.modules import (
    SharedDiT,
    causal_attention_mask,
    timestep_embedding,
)


class FastLeWAM(nn.Module):
    """复用 LeWM 视觉编码器、并可切换 Stage A/B/AB/C 的共享 DiT 预测器。"""

    def __init__(
        self,
        encoder,
        projector,
        latent_dim: int,
        action_dim: int,
        action_horizon: int = 5,
        latent_head_dim: int = 192,
        latent_head_layers: int = 6,
        heads: int = 6,
        mlp_dim: int = 768,
        dropout: float = 0.0,
        task_condition_dim: int | None = None,
        inference_steps: int = 10,
    ):
        """构造视觉编码器接口、共享 DiT、动作头和 causal-prefix latent 头。"""
        super().__init__()
        if action_horizon < 1 or action_dim < 1 or inference_steps < 1:
            raise ValueError(
                "action_horizon, action_dim, and inference_steps must be positive"
            )

        self.encoder = encoder
        self.projector = projector
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.model_dim = latent_head_dim
        self.inference_steps = inference_steps
        self.task_condition_dim = task_condition_dim

        self.action_input = nn.Linear(action_dim, latent_head_dim)
        self.latent_input = nn.Linear(latent_dim, latent_head_dim)
        self.z_condition = nn.Linear(latent_dim, latent_head_dim)
        self.task_condition = (
            nn.Linear(task_condition_dim, latent_head_dim)
            if task_condition_dim is not None
            else None
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(latent_head_dim, latent_head_dim),
            nn.SiLU(),
            nn.Linear(latent_head_dim, latent_head_dim),
        )
        self.action_positions = nn.Parameter(
            torch.randn(1, action_horizon, latent_head_dim) * 0.02
        )
        self.joint_positions = nn.Parameter(
            torch.randn(1, 1 + 2 * action_horizon, latent_head_dim) * 0.02
        )
        self.query_tokens = nn.Parameter(
            torch.randn(1, action_horizon, latent_head_dim) * 0.02
        )
        self.predictor = SharedDiT(
            dim=latent_head_dim,
            depth=latent_head_layers,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.action_head = nn.Linear(latent_head_dim, action_dim)
        self.latent_head = nn.Linear(latent_head_dim, latent_dim)
        self.register_buffer(
            "causal_mask", causal_attention_mask(action_horizon), persistent=False
        )

    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """将任意前导维的 ``[...,C,H,W]`` 图像编码为 ``[...,latent_dim]``。"""
        if pixels.ndim < 4:
            raise ValueError(
                f"pixels must end in [C,H,W], got shape {tuple(pixels.shape)}"
            )
        leading = pixels.shape[:-3]
        flat = pixels.reshape(-1, *pixels.shape[-3:])
        # uint8 训练数据保持原格式完成 H2D，随后才在当前设备上归一化。
        if flat.dtype == torch.uint8:
            flat = flat.float().div_(255.0)
            mean = flat.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
            std = flat.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
            flat = flat.sub_(mean).div_(std)
        else:
            flat = flat.float()
        encoded = self.encoder(flat, interpolate_pos_encoding=True)
        latent = self.projector(encoded.last_hidden_state[:, 0])
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"encoder/projector produced dim {latent.shape[-1]}, "
                f"expected {self.latent_dim}"
            )
        return latent.reshape(*leading, self.latent_dim)

    def _validate_inputs(self, z0, actions, timestep):
        """检查当前 latent、动作 chunk 和流时间步形状，并规范时间步设备与 dtype。"""
        if z0.ndim != 2 or z0.shape[-1] != self.latent_dim:
            raise ValueError(
                f"z0 must have shape [B,{self.latent_dim}], got {tuple(z0.shape)}"
            )
        expected = (z0.shape[0], self.action_horizon, self.action_dim)
        if tuple(actions.shape) != expected:
            raise ValueError(
                f"actions must have shape {expected}, got {tuple(actions.shape)}"
            )
        if not torch.is_tensor(timestep):
            timestep = torch.as_tensor(timestep, device=z0.device, dtype=z0.dtype)
        timestep = timestep.to(device=z0.device, dtype=z0.dtype)
        if timestep.ndim == 0:
            timestep = timestep.expand(z0.shape[0])
        if tuple(timestep.shape) != (z0.shape[0],):
            raise ValueError(
                f"timestep must have shape [{z0.shape[0]}], "
                f"got {tuple(timestep.shape)}"
            )
        return timestep

    def _condition(self, z0, timestep, task_condition):
        """融合当前 latent、流时间步以及可选任务条件，生成 AdaLN 条件向量。"""
        condition = self.z_condition(z0) + self.time_mlp(
            timestep_embedding(timestep, self.model_dim).to(z0.dtype)
        )
        if task_condition is not None:
            if self.task_condition is None:
                raise ValueError(
                    "task_condition was provided, but task_condition_dim is disabled"
                )
            expected = (z0.shape[0], self.task_condition_dim)
            if tuple(task_condition.shape) != expected:
                raise ValueError(
                    f"task_condition must have shape {expected}, "
                    f"got {tuple(task_condition.shape)}"
                )
            condition = condition + self.task_condition(task_condition)
        return condition

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """加载 checkpoint，并兼容旧版在禁用任务条件时仍保存的投影参数。"""
        if self.task_condition is None:
            state_dict.pop(prefix + "task_condition.weight", None)
            state_dict.pop(prefix + "task_condition.bias", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _stage_a(self, z0, noisy_actions, timestep, task_condition=None):
        """以双向动作 self-attention 预测整段 noisy action chunk 的流速度。"""
        condition = self._condition(z0, timestep, task_condition)
        tokens = self.action_input(noisy_actions) + self.action_positions
        hidden = self.predictor(tokens, condition, attention_mask=None)
        return self.action_head(hidden)

    def _joint_hidden(self, z0, actions, timestep, task_condition=None):
        """按 ``[z0,a1,q1,...,aH,qH]`` 排列 token，并计算严格因果隐藏状态。"""
        condition = self._condition(z0, timestep, task_condition)
        batch = z0.shape[0]
        tokens = z0.new_empty(batch, 1 + 2 * self.action_horizon, self.model_dim)
        tokens[:, 0] = self.latent_input(z0)
        tokens[:, 1::2] = self.action_input(actions)
        tokens[:, 2::2] = self.query_tokens.expand(batch, -1, -1)
        tokens = tokens + self.joint_positions
        return self.predictor(tokens, condition, attention_mask=self.causal_mask)

    def forward(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        mode: str = "stage_a",
        task_condition: torch.Tensor | None = None,
        detach_clean_action: bool = False,
    ) -> dict[str, torch.Tensor]:
        """按 mode 执行 Stage A、B、A+B 或 C，并返回对应动作/latent 预测。"""
        timestep = self._validate_inputs(z0, actions, timestep)
        if mode == "stage_a":
            velocity = self._stage_a(z0, actions, timestep, task_condition)
            return {"action_velocity": velocity}
        if mode == "stage_b":
            hidden = self._joint_hidden(z0, actions, timestep, task_condition)
            return {"predicted_latents": self.latent_head(hidden[:, 2::2])}
        if mode == "stage_ab":
            velocity = self._stage_a(z0, actions, timestep, task_condition)
            clean = actions + (1.0 - timestep[:, None, None]) * velocity
            stage_b_actions = clean.detach() if detach_clean_action else clean
            hidden = self._joint_hidden(z0, stage_b_actions, timestep, task_condition)
            return {
                "action_velocity": velocity,
                "clean_action": clean,
                "predicted_latents": self.latent_head(hidden[:, 2::2]),
            }
        if mode == "stage_c":
            hidden = self._joint_hidden(z0, actions, timestep, task_condition)
            return {
                "action_velocity": self.action_head(hidden[:, 1::2]),
                "predicted_latents": self.latent_head(hidden[:, 2::2]),
            }
        raise ValueError(
            f"unknown mode {mode!r}; expected stage_a, stage_b, stage_ab, or stage_c"
        )

    def _initial_noise(self, z0, noise, generator):
        """创建或校验 Euler flow 采样的初始高斯动作噪声。"""
        shape = (z0.shape[0], self.action_horizon, self.action_dim)
        if noise is None:
            return torch.randn(
                shape, device=z0.device, dtype=z0.dtype, generator=generator
            )
        if tuple(noise.shape) != shape:
            raise ValueError(f"noise must have shape {shape}, got {tuple(noise.shape)}")
        return noise.to(device=z0.device, dtype=z0.dtype).clone()

    def _euler_sample(
        self,
        z0,
        *,
        mode,
        noise=None,
        num_steps=None,
        generator=None,
        task_condition=None,
    ):
        """从噪声出发，用固定步长显式 Euler 积分生成归一化动作 chunk。"""
        steps = self.inference_steps if num_steps is None else num_steps
        if steps < 1:
            raise ValueError("num_steps must be positive")
        actions = self._initial_noise(z0, noise, generator)
        dt = 1.0 / steps
        for step in range(steps):
            timestep = z0.new_full((z0.shape[0],), step / steps)
            velocity = self(
                z0, actions, timestep, mode=mode, task_condition=task_condition
            )["action_velocity"]
            actions = actions + dt * velocity
        return actions

    @torch.no_grad()
    def sample_actions(
        self,
        z0,
        *,
        noise=None,
        num_steps=None,
        generator=None,
        task_condition=None,
    ):
        """只运行 Stage A flow，采样一整段归一化动作 chunk。"""
        return self._euler_sample(
            z0,
            mode="stage_a",
            noise=noise,
            num_steps=num_steps,
            generator=generator,
            task_condition=task_condition,
        )

    @torch.no_grad()
    def sample_joint(
        self,
        z0,
        *,
        noise=None,
        num_steps=None,
        generator=None,
        task_condition=None,
    ):
        """用 Stage C 采样动作，并在 clean endpoint 读取并行 latent query。"""
        actions = self._euler_sample(
            z0,
            mode="stage_c",
            noise=noise,
            num_steps=num_steps,
            generator=generator,
            task_condition=task_condition,
        )
        endpoint = self(
            z0,
            actions,
            torch.ones(z0.shape[0], device=z0.device, dtype=z0.dtype),
            mode="stage_c",
            task_condition=task_condition,
        )
        return {"actions": actions, "predicted_latents": endpoint["predicted_latents"]}

    @staticmethod
    def _last_frame(pixels):
        """从可带时间维的像素张量中取得最后一帧，并保留 batch 维。"""
        if pixels.ndim < 4:
            raise ValueError(f"pixels must end in [C,H,W], got {tuple(pixels.shape)}")
        return pixels if pixels.ndim == 4 else pixels.select(dim=-4, index=-1)

    def get_action(self, info, horizon=1, prefix_actions=None):
        """实现 solver warm start 使用的 Actionable 接口，并支持已有动作前缀。"""
        if not 1 <= horizon <= self.action_horizon:
            raise ValueError(
                f"horizon must be in [1,{self.action_horizon}], got {horizon}"
            )
        z0 = self.encode_pixels(self._last_frame(info["pixels"]))
        if z0.ndim != 2:
            raise ValueError("get_action expects pixels with leading [B,T] dimensions")
        if prefix_actions is not None and prefix_actions.shape[1] > 0:
            prefix_length = prefix_actions.shape[1]
            if prefix_length > self.action_horizon:
                raise ValueError("prefix_actions exceed the configured action horizon")
            padded = z0.new_zeros(z0.shape[0], self.action_horizon, self.action_dim)
            padded[:, :prefix_length] = prefix_actions.to(z0)
            predicted = self(
                z0,
                padded,
                torch.ones(z0.shape[0], device=z0.device, dtype=z0.dtype),
                mode="stage_b",
            )["predicted_latents"]
            z0 = predicted[:, prefix_length - 1]
        return self.sample_actions(z0)[:, :horizon]

    def get_cost(self, info_dict, action_candidates):
        """用一次并行 Stage B 因果预测计算候选动作终点到目标 latent 的代价。"""
        if action_candidates.ndim != 4:
            raise ValueError(
                "action_candidates must have shape [B,S,H,A], got "
                f"{tuple(action_candidates.shape)}"
            )
        batch, samples, horizon, action_dim = action_candidates.shape
        if horizon != self.action_horizon or action_dim != self.action_dim:
            raise ValueError(
                "candidate shape must match configured horizon/action_dim; got "
                f"{(horizon, action_dim)}, expected "
                f"{(self.action_horizon, self.action_dim)}"
            )
        current = self._last_frame(info_dict["pixels"])
        goal = self._last_frame(info_dict["goal"])
        z0 = self.encode_pixels(current).reshape(batch * samples, self.latent_dim)
        goal_latent = self.encode_pixels(goal).reshape(batch * samples, self.latent_dim)
        candidates = action_candidates.reshape(
            batch * samples, self.action_horizon, self.action_dim
        )
        predicted = self(
            z0,
            candidates,
            torch.ones(batch * samples, device=z0.device, dtype=z0.dtype),
            mode="stage_b",
        )["predicted_latents"]
        cost = (predicted[:, -1] - goal_latent).square().mean(dim=-1)
        return cost.reshape(batch, samples)


__all__ = ["FastLeWAM"]
