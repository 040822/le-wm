"""Fast-LeWAM 原型共用的 Action/World DiT 模型。"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from source.model.fast_lewam.modules import (
    SharedDiT,
    block_causal_attention_mask,
    causal_attention_mask,
    stage_a_attention_mask,
    terminal_full_attention_mask,
    timestep_embedding,
)


class FastLeWAM(nn.Module):
    """复用 LeWM 视觉编码器、并可切换 Stage A/B/AB/C 的共享 DiT 预测器。"""

    STATE_TYPE = 0
    GOAL_TYPE = 1
    ACTION_TYPE = 2
    QUERY_TYPE = 3

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
        stage_a_goal_injection: str = "none",
        inference_steps: int = 10,
        token_encoding: str = "legacy",
        stage_b_dynamics: str = "parallel_prefix",
        stage_b_attention_mode: str = "strict_causal",
        serial_activation_checkpointing: bool = False,
    ):
        """构造视觉编码器接口、共享 DiT、动作头和 causal-prefix latent 头。"""
        super().__init__()
        if action_horizon < 1 or action_dim < 1 or inference_steps < 1:
            raise ValueError(
                "action_horizon, action_dim, and inference_steps must be positive"
            )
        if stage_a_goal_injection not in {"none", "token"}:
            raise ValueError("stage_a_goal_injection must be 'none' or 'token'")
        if token_encoding not in {"legacy", "physical_time_type"}:
            raise ValueError("token_encoding must be 'legacy' or 'physical_time_type'")
        if stage_b_dynamics not in {"parallel_prefix", "serial_one_step"}:
            raise ValueError(
                "stage_b_dynamics must be 'parallel_prefix' or 'serial_one_step'"
            )
        if stage_b_attention_mode not in {
            "strict_causal",
            "block_causal",
            "terminal_full",
        }:
            raise ValueError(
                "stage_b_attention_mode must be 'strict_causal', 'block_causal', "
                "or 'terminal_full'"
            )
        if (
            stage_b_attention_mode != "strict_causal"
            and stage_b_dynamics != "parallel_prefix"
        ):
            raise ValueError(
                "non-strict stage_b_attention_mode requires "
                "stage_b_dynamics='parallel_prefix'"
            )

        self.encoder = encoder
        self.projector = projector
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.model_dim = latent_head_dim
        self.inference_steps = inference_steps
        self.task_condition_dim = task_condition_dim
        self.stage_a_goal_injection = stage_a_goal_injection
        self.token_encoding = token_encoding
        self.stage_b_dynamics = stage_b_dynamics
        self.stage_b_attention_mode = stage_b_attention_mode
        self.serial_activation_checkpointing = bool(serial_activation_checkpointing)

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
        if token_encoding == "legacy":
            self.action_positions = nn.Parameter(
                torch.randn(1, action_horizon, latent_head_dim) * 0.02
            )
            self.goal_token_embedding = (
                nn.Parameter(torch.zeros(1, 1, latent_head_dim))
                if stage_a_goal_injection == "token"
                else None
            )
            self.joint_positions = nn.Parameter(
                torch.randn(1, 1 + 2 * action_horizon, latent_head_dim) * 0.02
            )
            self.query_tokens = nn.Parameter(
                torch.randn(1, action_horizon, latent_head_dim) * 0.02
            )
            self.time_positions = None
            self.type_embeddings = None
            self.query_content = None
        else:
            self.action_positions = None
            self.goal_token_embedding = None
            self.joint_positions = None
            self.query_tokens = None
            self.time_positions = nn.Parameter(
                torch.randn(1, action_horizon + 1, latent_head_dim) * 0.02
            )
            self.type_embeddings = nn.Embedding(4, latent_head_dim)
            nn.init.normal_(self.type_embeddings.weight, std=0.02)
            self.query_content = nn.Parameter(
                torch.randn(1, 1, latent_head_dim) * 0.02
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
        self.register_buffer(
            "block_causal_mask",
            block_causal_attention_mask(action_horizon),
            persistent=False,
        )
        self.register_buffer(
            "terminal_full_mask",
            terminal_full_attention_mask(action_horizon),
            persistent=False,
        )
        self.register_buffer(
            "stage_a_mask",
            stage_a_attention_mask(
                action_horizon,
                num_anchor_tokens=2 if stage_a_goal_injection == "token" else 1,
            ),
            persistent=False,
        )
        self.last_guidance_stats = {
            "mode": "none",
            "flow_steps": int(inference_steps),
            "guidance_last_steps": 0,
            "guidance_inner_steps": 0,
            "guidance_step_size": 0.0,
            "guidance_max_rms_offset": 0.0,
            "forward_count": 0,
            "stage_a_forward_count": 0,
            "stage_b_forward_count": 0,
            "backward_count": 0,
            "zero_gradient_count": 0,
        }
        
    # ========= input processing  ============

    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """
        编码image为latent。
        将任意前导维的 ``[...,C,H,W]`` 图像编码为 ``[...,latent_dim]``。
        """
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
        """
        检查当前 latent、动作 chunk 和流时间步形状，并规范时间步设备与 dtype。
        z0: [B,latent_dim]
        actions: [B,H,action_dim]
        timestep: [B] or scalar => [B]
        
        """
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
        """
        融合当前 latent、流时间步以及可选任务条件，生成 AdaLN 条件向量。
        z0: [B,latent_dim] => [B,latent_head_dim]
        timestep: [B] => [B,latent_head_dim]
        condition: [B,latent_head_dim] + [B,latent_head_dim] +( [B,latent_head_dim] if task_condition is not None )=> [B,latent_head_dim]
        """
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


    # ========= forward ============

    def _state_token(self, z0):
        token = self.latent_input(z0).unsqueeze(1)
        if self.token_encoding == "legacy":
            return token
        return (
            token
            + self.time_positions[:, :1]
            + self.type_embeddings.weight[self.STATE_TYPE]
        )

    def _action_tokens(self, actions):
        tokens = self.action_input(actions)
        if self.token_encoding == "legacy":
            return tokens + self.action_positions
        return (
            tokens
            + self.time_positions[:, : self.action_horizon]
            + self.type_embeddings.weight[self.ACTION_TYPE]
        )

    def _stage_a(
        self, z0, noisy_actions, timestep, task_condition=None, goal_latent=None
    ):
        """
        用受保护的 z0/zg 锚点和双向动作 attention 预测动作流。
        """
        condition = self._condition(z0, timestep, task_condition) # [B, latent_head_dim]
        state_token = self._state_token(z0)
        anchors = [state_token]
        
        # goal_latent 注入方式为token时，锚点构造为 [z0+zg]
        if self.stage_a_goal_injection == "token":
            if goal_latent is None:
                raise ValueError(
                    "goal_latent is required when stage_a_goal_injection='token'"
                )
            if tuple(goal_latent.shape) != tuple(z0.shape):
                raise ValueError(
                    f"goal_latent must have shape {tuple(z0.shape)}, "
                    f"got {tuple(goal_latent.shape)}"
                )
            goal_token = self.latent_input(goal_latent).unsqueeze(1)
            if self.token_encoding == "legacy":
                goal_token = goal_token + self.goal_token_embedding
            else:
                goal_token = (
                    goal_token
                    + self.time_positions[
                        :, self.action_horizon : self.action_horizon + 1
                    ]
                    + self.type_embeddings.weight[self.GOAL_TYPE]
                )
            anchors.append(goal_token)
        
        action_tokens = self._action_tokens(noisy_actions)
        tokens = torch.cat((*anchors, action_tokens), dim=1) # [z0.zg.a1.a2...aH]
        hidden = self.predictor(tokens, condition, attention_mask=self.stage_a_mask)
        return self.action_head(hidden[:, len(anchors):])

    def _joint_hidden(self, z0, actions, timestep, task_condition=None):
        """
        stage b + stage c
        按 ``[z0,a1,q1,...,aH,qH]`` 排列 token，并计算严格因果隐藏状态。
        """
        condition = self._condition(z0, timestep, task_condition)
        batch = z0.shape[0]
        tokens = z0.new_empty(batch, 1 + 2 * self.action_horizon, self.model_dim)
        if self.token_encoding == "legacy":
            tokens[:, 0] = self.latent_input(z0)
            tokens[:, 1::2] = self.action_input(actions)
            tokens[:, 2::2] = self.query_tokens.expand(batch, -1, -1)
            tokens = tokens + self.joint_positions
        else:
            tokens[:, 0] = self._state_token(z0).squeeze(1)
            tokens[:, 1::2] = self._action_tokens(actions)
            tokens[:, 2::2] = (
                self.query_content
                + self.time_positions[:, 1 : self.action_horizon + 1]
                + self.type_embeddings.weight[self.QUERY_TYPE]
            )
        if self.stage_b_attention_mode == "terminal_full":
            raise ValueError("terminal_full uses the terminal Stage-B token path")
        attention_mask = (
            self.causal_mask
            if self.stage_b_attention_mode == "strict_causal"
            else self.block_causal_mask
        )
        return self.predictor(tokens, condition, attention_mask=attention_mask)

    def _terminal_hidden(self, z0, actions, timestep, task_condition=None):
        """Encode all actions and one terminal query with full attention."""
        condition = self._condition(z0, timestep, task_condition)
        batch = z0.shape[0]
        state_token = self._state_token(z0)
        if self.token_encoding == "legacy":
            state_token = state_token + self.joint_positions[:, :1]
            action_tokens = self.action_input(actions) + self.joint_positions[:, 1::2]
            query_token = self.query_tokens[:, -1:].expand(batch, -1, -1)
            query_token = query_token + self.joint_positions[:, -1:]
        else:
            action_tokens = self._action_tokens(actions)
            query_token = (
                self.query_content
                + self.time_positions[:, self.action_horizon : self.action_horizon + 1]
                + self.type_embeddings.weight[self.QUERY_TYPE]
            ).expand(batch, -1, -1)
        tokens = torch.cat((state_token, action_tokens, query_token), dim=1)
        return self.predictor(
            tokens, condition, attention_mask=self.terminal_full_mask
        )

    def predict_one_step(self, state, action, task_condition=None):
        """Predict one latent transition from a true or recursively predicted state."""
        if state.ndim != 2 or state.shape[-1] != self.latent_dim:
            raise ValueError(
                f"state must have shape [B,{self.latent_dim}], got {tuple(state.shape)}"
            )
        expected_action_shape = (state.shape[0], self.action_dim)
        if tuple(action.shape) != expected_action_shape:
            raise ValueError(
                f"action must have shape {expected_action_shape}, got {tuple(action.shape)}"
            )
        condition = self._condition(
            state,
            torch.ones(state.shape[0], device=state.device, dtype=state.dtype),
            task_condition,
        )
        if self.token_encoding == "legacy":
            tokens = torch.cat(
                (
                    self.latent_input(state).unsqueeze(1),
                    self.action_input(action).unsqueeze(1),
                    self.query_tokens[:, :1].expand(state.shape[0], -1, -1),
                ),
                dim=1,
            )
            tokens = tokens + self.joint_positions[:, :3]
        else:
            tokens = torch.cat(
                (
                    self._state_token(state),
                    (
                        self.action_input(action).unsqueeze(1)
                        + self.time_positions[:, :1]
                        + self.type_embeddings.weight[self.ACTION_TYPE]
                    ),
                    (
                        self.query_content
                        + self.time_positions[:, 1:2]
                        + self.type_embeddings.weight[self.QUERY_TYPE]
                    ).expand(state.shape[0], -1, -1),
                ),
                dim=1,
            )
        use_checkpoint = (
            self.serial_activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )
        if use_checkpoint:
            hidden = checkpoint(
                self.predictor,
                tokens,
                condition,
                use_reentrant=False,
            )
        else:
            hidden = self.predictor(tokens, condition)
        return self.latent_head(hidden[:, -1])

    def rollout_latents(self, z0, actions, timestep, task_condition=None):
        """Predict a latent trajectory with the configured Stage-B dynamics."""
        timestep = self._validate_inputs(z0, actions, timestep)
        if self.stage_b_dynamics == "parallel_prefix":
            if self.stage_b_attention_mode == "terminal_full":
                hidden = self._terminal_hidden(z0, actions, timestep, task_condition)
                return self.latent_head(hidden[:, -1:])
            hidden = self._joint_hidden(z0, actions, timestep, task_condition)
            return self.latent_head(hidden[:, 2::2])
        state = z0
        predicted = []
        for index in range(self.action_horizon):
            state = self.predict_one_step(
                state,
                actions[:, index],
                task_condition=task_condition,
            )
            predicted.append(state)
        return torch.stack(predicted, dim=1)

    def predict_training_latents(self, states, actions, task_condition=None):
        """Predict teacher-forced transitions for Stage-B training."""
        expected_states = (actions.shape[0], self.action_horizon, self.latent_dim)
        expected_actions = (actions.shape[0], self.action_horizon, self.action_dim)
        if tuple(states.shape) != expected_states:
            raise ValueError(
                f"states must have shape {expected_states}, got {tuple(states.shape)}"
            )
        if tuple(actions.shape) != expected_actions:
            raise ValueError(
                f"actions must have shape {expected_actions}, got {tuple(actions.shape)}"
            )
        if self.stage_b_dynamics == "parallel_prefix":
            return self.rollout_latents(
                states[:, 0],
                actions,
                torch.ones(
                    actions.shape[0], device=states.device, dtype=states.dtype
                ),
                task_condition,
            )
        batch = states.shape[0]
        flat_task_condition = None
        if task_condition is not None:
            flat_task_condition = (
                task_condition[:, None]
                .expand(-1, self.action_horizon, *task_condition.shape[1:])
                .reshape(batch * self.action_horizon, *task_condition.shape[1:])
            )
        predicted = self.predict_one_step(
            states.reshape(batch * self.action_horizon, self.latent_dim),
            actions.reshape(batch * self.action_horizon, self.action_dim),
            task_condition=flat_task_condition,
        )
        return predicted.reshape(batch, self.action_horizon, self.latent_dim)

    def forward(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        mode: str = "stage_a",
        task_condition: torch.Tensor | None = None,
        goal_latent: torch.Tensor | None = None,
        detach_clean_action: bool = False,
    ) -> dict[str, torch.Tensor]:
        """按 mode 执行 Stage A、B、A+B 或 C，并返回对应动作/latent 预测。"""
        timestep = self._validate_inputs(z0, actions, timestep) # 检查shape，并规范设备和dtype
        
        # 不同stage mode
        if mode == "stage_a":
            # Stage A 预测动作速度
            velocity = self._stage_a(
                z0, actions, timestep, task_condition, goal_latent
            )
            return {"action_velocity": velocity}
        if mode == "stage_b":
            # Stage B 使用 Stage A 预测动作或真值动作 进行 latent 监督。
            return {
                "predicted_latents": self.rollout_latents(
                    z0, actions, timestep, task_condition
                )
            }
        if mode in {"stage_ab", "stage_c"} and self.stage_b_attention_mode != "strict_causal":
            raise ValueError(
                "stage_ab and stage_c require stage_b_attention_mode='strict_causal'"
            )
        if mode == "stage_ab":
            # Stage A 预测动作速度，Stage B 使用 Stage A 预测动作或真值动作 进行 latent 监督。
            velocity = self._stage_a(
                z0, actions, timestep, task_condition, goal_latent
            )
            clean = actions + (1.0 - timestep[:, None, None]) * velocity
            stage_b_actions = clean.detach() if detach_clean_action else clean
            hidden = self._joint_hidden(z0, stage_b_actions, timestep, task_condition)
            return {
                "action_velocity": velocity,
                "clean_action": clean,
                "predicted_latents": self.latent_head(hidden[:, 2::2]),
            }
        if mode == "stage_c":
            # Stage C 直接同时预测动作速度和未来 latent，使用真值动作进行监督。
            hidden = self._joint_hidden(z0, actions, timestep, task_condition)
            return {
                "action_velocity": self.action_head(hidden[:, 1::2]),
                "predicted_latents": self.latent_head(hidden[:, 2::2]),
            }
        raise ValueError(
            f"unknown mode {mode!r}; expected stage_a, stage_b, stage_ab, or stage_c"
        )

    # ========= inference ============
        
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

    @staticmethod
    def _normalize_guidance_gradient(gradient: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Normalize each candidate independently and count zero gradients."""
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("guidance gradient contains a non-finite value")
        reduce_dims = tuple(range(1, gradient.ndim))
        rms = gradient.float().square().mean(dim=reduce_dims, keepdim=True).sqrt()
        zero = rms <= torch.finfo(rms.dtype).eps
        normalized = gradient.float() / rms.clamp_min(torch.finfo(rms.dtype).eps)
        normalized = torch.where(zero, torch.zeros_like(normalized), normalized)
        return normalized.to(dtype=gradient.dtype), int(zero.sum().item())

    @staticmethod
    def _clip_rms_displacement(
        value: torch.Tensor,
        reference: torch.Tensor,
        max_rms_offset: float,
    ) -> torch.Tensor:
        """Keep every candidate inside an RMS trust region around its proposal."""
        delta = value - reference
        rms = delta.float().square().mean(dim=(1, 2), keepdim=True).sqrt()
        limit = float(max_rms_offset)
        scale = (limit / rms.clamp_min(limit)).clamp(max=1.0)
        return reference + delta * scale.to(dtype=delta.dtype)

    def _validate_guidance_options(
        self,
        z0: torch.Tensor,
        goal_latent: torch.Tensor | None,
        *,
        guidance_mode: str,
        guidance_step_size: float,
        guidance_last_steps: int,
        guidance_inner_steps: int,
        guidance_max_rms_offset: float,
        integrator: str,
    ) -> str:
        mode = str(guidance_mode).lower()
        if mode not in {"none", "post_opt", "guided_flow"}:
            raise ValueError(
                "guidance_mode must be one of 'none', 'post_opt', or 'guided_flow'"
            )
        if mode != "none":
            if goal_latent is None:
                raise ValueError("goal_latent is required when guidance is enabled")
            if tuple(goal_latent.shape) != tuple(z0.shape):
                raise ValueError(
                    f"goal_latent must have shape {tuple(z0.shape)}, "
                    f"got {tuple(goal_latent.shape)}"
                )
            if not torch.isfinite(goal_latent).all():
                raise ValueError("goal_latent contains a non-finite value")
            if not torch.isfinite(z0).all():
                raise ValueError("z0 contains a non-finite value")
            if not torch.isfinite(z0.new_tensor(float(guidance_step_size))):
                raise ValueError("guidance_step_size must be finite")
            if float(guidance_step_size) <= 0.0:
                raise ValueError("guidance_step_size must be positive")
            if int(guidance_last_steps) < 1:
                raise ValueError("guidance_last_steps must be positive")
            if int(guidance_inner_steps) < 1:
                raise ValueError("guidance_inner_steps must be positive")
            if not torch.isfinite(z0.new_tensor(float(guidance_max_rms_offset))):
                raise ValueError("guidance_max_rms_offset must be finite")
            if float(guidance_max_rms_offset) <= 0.0:
                raise ValueError("guidance_max_rms_offset must be positive")
            if str(integrator).lower() != "euler":
                raise ValueError("guidance is currently defined only for Euler integration")
        return mode

    def _latent_cost_from_clean_actions(
        self,
        z0: torch.Tensor,
        goal_latent: torch.Tensor,
        clean_actions: torch.Tensor,
        task_condition: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return one Stage-B terminal latent cost per action candidate."""
        predicted = self.rollout_latents(
            z0,
            clean_actions,
            torch.ones(z0.shape[0], device=z0.device, dtype=z0.dtype),
            task_condition,
        )
        return (predicted[:, -1] - goal_latent).square().mean(dim=-1)

    def _apply_post_opt_guidance(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor,
        goal_latent: torch.Tensor,
        *,
        task_condition: torch.Tensor | None,
        step_size: float,
        inner_steps: int,
        max_rms_offset: float,
        stats: dict[str, int],
    ) -> torch.Tensor:
        """Apply cost descent after flow sampling inside a bounded trust region."""
        reference = actions.detach()
        current = reference
        for _ in range(int(inner_steps)):
            current = current.detach().requires_grad_(True)
            cost = self._latent_cost_from_clean_actions(
                z0, goal_latent, current, task_condition
            )
            gradient = torch.autograd.grad(cost.sum(), current)[0]
            stats["stage_b_forward_count"] += 1
            stats["backward_count"] += 1
            normalized, zero_count = self._normalize_guidance_gradient(gradient)
            stats["zero_gradient_count"] += zero_count
            current = current.detach() - float(step_size) * normalized
            current = self._clip_rms_displacement(
                current, reference, float(max_rms_offset)
            ).detach()
        return current

    def _apply_guided_flow_step(
        self,
        z0: torch.Tensor,
        proposal: torch.Tensor,
        goal_latent: torch.Tensor,
        *,
        next_timestep: torch.Tensor,
        task_condition: torch.Tensor | None,
        step_size: float,
        inner_steps: int,
        max_rms_offset: float,
        stats: dict[str, int],
    ) -> torch.Tensor:
        """Guide one Euler proposal using clean-estimate Stage-B cost gradients."""
        reference = proposal.detach()
        current = reference
        for _ in range(int(inner_steps)):
            current = current.detach().requires_grad_(True)
            velocity = self(
                z0,
                current,
                next_timestep,
                mode="stage_a",
                task_condition=task_condition,
                goal_latent=goal_latent,
            )["action_velocity"]
            clean = current + (
                1.0 - next_timestep[:, None, None]
            ) * velocity
            cost = self._latent_cost_from_clean_actions(
                z0, goal_latent, clean, task_condition
            )
            gradient = torch.autograd.grad(cost.sum(), current)[0]
            stats["stage_a_forward_count"] += 1
            stats["stage_b_forward_count"] += 1
            stats["backward_count"] += 1
            normalized, zero_count = self._normalize_guidance_gradient(gradient)
            stats["zero_gradient_count"] += zero_count
            current = current.detach() - float(step_size) * normalized
            current = self._clip_rms_displacement(
                current, reference, float(max_rms_offset)
            ).detach()
        return current

    def _euler_sample_base(
        self,
        z0,
        *,
        mode,
        noise=None,
        num_steps=None,
        generator=None,
        task_condition=None,
        goal_latent=None,
        integrator="euler",
    ) -> tuple[torch.Tensor, int]:
        """Run the original unguided Euler/Heun path and count model forwards."""
        steps = self.inference_steps if num_steps is None else int(num_steps)
        if steps < 1:
            raise ValueError("num_steps must be positive")
        integrator = str(integrator).lower()
        if integrator not in {"euler", "heun"}:
            raise ValueError("integrator must be 'euler' or 'heun'")
        actions = self._initial_noise(z0, noise, generator)
        dt = 1.0 / steps
        forward_count = 0
        for step in range(steps):
            timestep = z0.new_full((z0.shape[0],), step / steps)
            velocity = self(
                z0,
                actions,
                timestep,
                mode=mode,
                task_condition=task_condition,
                goal_latent=goal_latent,
            )["action_velocity"]
            forward_count += 1
            if integrator == "euler":
                actions = actions + dt * velocity
                continue
            predictor = actions + dt * velocity
            next_timestep = z0.new_full((z0.shape[0],), (step + 1) / steps)
            next_velocity = self(
                z0,
                predictor,
                next_timestep,
                mode=mode,
                task_condition=task_condition,
                goal_latent=goal_latent,
            )["action_velocity"]
            forward_count += 1
            actions = actions + 0.5 * dt * (velocity + next_velocity)
        return actions, forward_count

    def _euler_sample(
        self,
        z0,
        *,
        mode,
        noise=None,
        num_steps=None,
        generator=None,
        task_condition=None,
        goal_latent=None,
        integrator="euler",
        guidance_mode="none",
        guidance_step_size=0.01,
        guidance_last_steps=5,
        guidance_inner_steps=5,
        guidance_max_rms_offset=0.20,
    ):
        """Sample an action chunk with optional bounded latent-cost guidance."""
        steps = self.inference_steps if num_steps is None else int(num_steps)
        if steps < 1:
            raise ValueError("num_steps must be positive")
        integrator = str(integrator).lower()
        mode_name = self._validate_guidance_options(
            z0,
            goal_latent,
            guidance_mode=guidance_mode,
            guidance_step_size=guidance_step_size,
            guidance_last_steps=guidance_last_steps,
            guidance_inner_steps=guidance_inner_steps,
            guidance_max_rms_offset=guidance_max_rms_offset,
            integrator=integrator,
        )
        stats: dict[str, int | float | str] = {
            "mode": mode_name,
            "flow_steps": int(steps),
            "guidance_last_steps": int(guidance_last_steps) if mode_name != "none" else 0,
            "guidance_inner_steps": int(guidance_inner_steps) if mode_name != "none" else 0,
            "guidance_step_size": float(guidance_step_size) if mode_name != "none" else 0.0,
            "guidance_max_rms_offset": (
                float(guidance_max_rms_offset) if mode_name != "none" else 0.0
            ),
            "forward_count": 0,
            "stage_a_forward_count": 0,
            "stage_b_forward_count": 0,
            "backward_count": 0,
            "zero_gradient_count": 0,
        }

        if mode_name == "none":
            with torch.no_grad():
                actions, forward_count = self._euler_sample_base(
                    z0,
                    mode=mode,
                    noise=noise,
                    num_steps=steps,
                    generator=generator,
                    task_condition=task_condition,
                    goal_latent=goal_latent,
                    integrator=integrator,
                )
            stats["forward_count"] = int(forward_count)
            stats["stage_a_forward_count"] = int(forward_count)
            self.last_guidance_stats = dict(stats)
            return actions.detach()

        if mode_name == "post_opt":
            with torch.no_grad():
                actions, forward_count = self._euler_sample_base(
                    z0,
                    mode=mode,
                    noise=noise,
                    num_steps=steps,
                    generator=generator,
                    task_condition=task_condition,
                    goal_latent=goal_latent,
                    integrator=integrator,
                )
            stats["forward_count"] = int(forward_count)
            stats["stage_a_forward_count"] = int(forward_count)
            with torch.enable_grad():
                actions = self._apply_post_opt_guidance(
                    z0,
                    actions,
                    goal_latent,
                    task_condition=task_condition,
                    step_size=float(guidance_step_size),
                    inner_steps=int(guidance_inner_steps),
                    max_rms_offset=float(guidance_max_rms_offset),
                    stats=stats,
                )
            stats["forward_count"] = int(
                stats["stage_a_forward_count"] + stats["stage_b_forward_count"]
            )
            self.last_guidance_stats = dict(stats)
            return actions.detach()

        if mode != "stage_a":
            raise ValueError("guidance is only supported for Stage A action sampling")
        with torch.enable_grad():
            actions = self._initial_noise(z0, noise, generator).detach()
            dt = 1.0 / steps
            start_guidance = max(0, steps - int(guidance_last_steps))
            for step in range(steps):
                timestep = z0.new_full((z0.shape[0],), step / steps)
                with torch.no_grad():
                    velocity = self(
                        z0,
                        actions,
                        timestep,
                        mode=mode,
                        task_condition=task_condition,
                        goal_latent=goal_latent,
                    )["action_velocity"]
                stats["stage_a_forward_count"] += 1
                proposal = (actions + dt * velocity).detach()
                if step >= start_guidance:
                    next_timestep = z0.new_full(
                        (z0.shape[0],), (step + 1) / steps
                    )
                    actions = self._apply_guided_flow_step(
                        z0,
                        proposal,
                        goal_latent,
                        next_timestep=next_timestep,
                        task_condition=task_condition,
                        step_size=float(guidance_step_size),
                        inner_steps=int(guidance_inner_steps),
                        max_rms_offset=float(guidance_max_rms_offset),
                        stats=stats,
                    )
                else:
                    actions = proposal
        stats["forward_count"] = int(
            stats["stage_a_forward_count"] + stats["stage_b_forward_count"]
        )
        self.last_guidance_stats = dict(stats)
        return actions.detach()

    def sample_actions(
        self,
        z0,
        *,
        noise=None,
        num_steps=None,
        generator=None,
        task_condition=None,
        goal_latent=None,
        integrator="euler",
        guidance_mode="none",
        guidance_step_size=0.01,
        guidance_last_steps=5,
        guidance_inner_steps=5,
        guidance_max_rms_offset=0.20,
    ):
        """Sample a Stage-A action chunk with optional latent-cost guidance."""
        return self._euler_sample(
            z0,
            mode="stage_a",
            noise=noise,
            num_steps=num_steps,
            generator=generator,
            task_condition=task_condition,
            goal_latent=goal_latent,
            integrator=integrator,
            guidance_mode=guidance_mode,
            guidance_step_size=guidance_step_size,
            guidance_last_steps=guidance_last_steps,
            guidance_inner_steps=guidance_inner_steps,
            guidance_max_rms_offset=guidance_max_rms_offset,
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
        integrator="euler",
    ):
        """用 Stage C 采样动作，并在 clean endpoint 读取并行 latent query。"""
        actions = self._euler_sample(
            z0,
            mode="stage_c",
            noise=noise,
            num_steps=num_steps,
            generator=generator,
            task_condition=task_condition,
            integrator=integrator,
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

    def get_action_from_latents(
        self,
        z0,
        goal_latent,
        horizon=1,
        prefix_actions=None,
        *,
        generator=None,
        num_steps=None,
        integrator="euler",
        guidance_mode="none",
        guidance_step_size=0.01,
        guidance_last_steps=5,
        guidance_inner_steps=5,
        guidance_max_rms_offset=0.20,
    ):
        """Generate an actor warm start from an encoded planning context."""
        if not 1 <= horizon <= self.action_horizon:
            raise ValueError(
                f"horizon must be in [1,{self.action_horizon}], got {horizon}"
            )
        if z0.ndim != 2:
            raise ValueError("z0 must have leading batch and latent dimensions")
        if self.stage_a_goal_injection == "token" and goal_latent is None:
            raise ValueError("goal_latent is required in Stage A token mode")
        if prefix_actions is not None and prefix_actions.shape[1] > 0:
            if self.stage_b_attention_mode == "terminal_full":
                raise ValueError(
                    "prefix_actions are unsupported when "
                    "stage_b_attention_mode='terminal_full'"
                )
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
        return self.sample_actions(
            z0,
            generator=generator,
            num_steps=num_steps,
            goal_latent=goal_latent,
            integrator=integrator,
            guidance_mode=guidance_mode,
            guidance_step_size=guidance_step_size,
            guidance_last_steps=guidance_last_steps,
            guidance_inner_steps=guidance_inner_steps,
            guidance_max_rms_offset=guidance_max_rms_offset,
        )[:, :horizon]

    def get_action(
        self,
        info,
        horizon=1,
        prefix_actions=None,
        *,
        generator=None,
        num_steps=None,
        integrator="euler",
        guidance_mode="none",
        guidance_step_size=0.01,
        guidance_last_steps=5,
        guidance_inner_steps=5,
        guidance_max_rms_offset=0.20,
    ):
        """Implement the Actionable interface from raw current/goal images."""
        z0 = self.encode_pixels(self._last_frame(info["pixels"]))
        goal_latent = None
        if self.stage_a_goal_injection == "token" or str(guidance_mode).lower() != "none":
            if "goal" not in info:
                raise ValueError("goal observations are required in Stage-A goal conditioning")
            goal_latent = self.encode_pixels(self._last_frame(info["goal"]))
        return self.get_action_from_latents(
            z0,
            goal_latent,
            horizon=horizon,
            prefix_actions=prefix_actions,
            generator=generator,
            num_steps=num_steps,
            integrator=integrator,
            guidance_mode=guidance_mode,
            guidance_step_size=guidance_step_size,
            guidance_last_steps=guidance_last_steps,
            guidance_inner_steps=guidance_inner_steps,
            guidance_max_rms_offset=guidance_max_rms_offset,
        )

    def get_cost_from_latents(self, z0, goal_latent, action_candidates):
        """Score candidate actions from one encoded current/goal context."""
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
        expected_context_shape = (batch, self.latent_dim)
        if tuple(z0.shape) != expected_context_shape:
            raise ValueError(
                f"z0 must have shape {expected_context_shape}, got {tuple(z0.shape)}"
            )
        if tuple(goal_latent.shape) != expected_context_shape:
            raise ValueError(
                "goal_latent must have shape "
                f"{expected_context_shape}, got {tuple(goal_latent.shape)}"
            )
        z0 = z0[:, None].expand(batch, samples, self.latent_dim).reshape(
            batch * samples, self.latent_dim
        )
        goal_latent = goal_latent[:, None].expand(
            batch, samples, self.latent_dim
        ).reshape(batch * samples, self.latent_dim)
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
