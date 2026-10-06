"""Round 4 shared-DiT model with latent planning and inverse dynamics.

The original :class:`FastLeWAM` remains the compatibility surface for Round 1
through Round 3 checkpoints.  ``Round4FastLeWAM`` is intentionally a separate
class: its learned mode embedding and D/E parameters make strict loading of an
old A/B checkpoint fail loudly instead of silently changing the experiment.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

from source.model.fast_lewam.jepa import FastLeWAM
from source.model.fast_lewam.modules import (
    inverse_dynamics_attention_mask,
    latent_path_attention_mask,
)


MODE_IDS = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def d_attention_mask(action_horizon: int, device=None) -> torch.Tensor:
    """Return the Stage-D mask for ``[z0, zg, x1, ..., x{H-1}]``."""
    return latent_path_attention_mask(action_horizon, device=device)


def e_attention_mask(action_horizon: int, device=None) -> torch.Tensor:
    """Return the Stage-E mask for states followed by action queries."""
    return inverse_dynamics_attention_mask(action_horizon, device=device)


class Round4FastLeWAM(FastLeWAM):
    """Shared A/B/C/D/E DiT used by the exploratory Round 4 experiment."""

    MODE_A = MODE_IDS["A"]
    MODE_B = MODE_IDS["B"]
    MODE_C = MODE_IDS["C"]
    MODE_D = MODE_IDS["D"]
    MODE_E = MODE_IDS["E"]

    def __init__(
        self,
        *args,
        d_flow_steps: int = 16,
        latent_flow_matching: bool = False,
        latent_flow_steps: int = 2,
        activation_checkpointing: bool = True,
        **kwargs,
    ):
        if int(d_flow_steps) < 1:
            raise ValueError("d_flow_steps must be positive")
        if int(latent_flow_steps) < 1:
            raise ValueError("latent_flow_steps must be positive")
        super().__init__(*args, **kwargs)
        # Stage A/B support one-step action chunks.  Only the optional latent-
        # path (D) branch needs an interior token and therefore a horizon >=2.

        self.d_flow_steps = int(d_flow_steps)
        self.latent_flow_matching = bool(latent_flow_matching)
        self.latent_flow_steps = int(latent_flow_steps)
        self.activation_checkpointing = bool(activation_checkpointing)

        # One condition embedding is shared by every branch.  A/B/C use the
        # same token layout as the existing model; only their AdaLN condition
        # gains this mode signal in the Round 4 class.
        self.mode_embedding = nn.Embedding(len(MODE_IDS), self.model_dim)
        nn.init.normal_(self.mode_embedding.weight, std=0.02)

        # D-specific positions are deliberately separate from legacy A/B
        # positions because the endpoint/interior layout is different.
        self.d_positions = nn.Parameter(
            torch.randn(self.action_horizon + 1, self.model_dim) * 0.02
        )
        self.d_velocity_head = nn.Linear(self.model_dim, self.latent_dim)

        # E has state positions, query positions, and an explicit adjacent
        # state transition feature [z_t, z_{t+1}, z_{t+1}-z_t].
        self.e_state_positions = nn.Parameter(
            torch.randn(self.action_horizon + 1, self.model_dim) * 0.02
        )
        self.e_query_positions = nn.Parameter(
            torch.randn(self.action_horizon, self.model_dim) * 0.02
        )
        self.e_query_content = nn.Parameter(
            torch.randn(self.action_horizon, self.model_dim) * 0.02
        )
        self.e_adjacent_input = nn.Linear(3 * self.latent_dim, self.model_dim)
        self.inverse_dynamics_head = nn.Linear(self.model_dim, self.action_dim)

        # Keep the historical Regression-B checkpoint schema unchanged.  The
        # conditional latent-flow interface exists only in Phase 6.1 models;
        # constructing the old default model therefore consumes the same
        # initialization stream and still loads old checkpoints strictly.
        if self.latent_flow_matching:
            self.latent_flow_input = nn.Linear(self.latent_dim, self.model_dim)
            self.latent_flow_time_mlp = nn.Sequential(
                nn.Linear(self.model_dim, self.model_dim),
                nn.SiLU(),
                nn.Linear(self.model_dim, self.model_dim),
            )
            self.latent_flow_velocity_head = nn.Linear(
                self.model_dim, self.latent_dim
            )

        self.register_buffer(
            "d_mask",
            d_attention_mask(self.action_horizon)
            if self.action_horizon >= 2
            else None,
            persistent=False,
        )
        self.register_buffer(
            "e_mask", e_attention_mask(self.action_horizon), persistent=False
        )

        # SharedDiT owns the switch so all A/B/D/E calls use the same behavior.
        self.predictor.activation_checkpointing = self.activation_checkpointing
        self._active_mode_id = self.MODE_A
        self._joint_mode_id = self.MODE_B

    @property
    def mode_ids(self) -> dict[str, int]:
        """Return a copy of the frozen mode-ID contract."""
        return dict(MODE_IDS)

    @property
    def d_attention_mask(self) -> torch.Tensor:
        """Named alias matching the Stage-D protocol terminology."""
        return self.d_mask

    @property
    def e_attention_mask(self) -> torch.Tensor:
        """Named alias matching the Stage-E protocol terminology."""
        return self.e_mask

    @property
    def latent_velocity_head(self) -> nn.Module:
        """Named alias for D's latent velocity output head."""
        return self.d_velocity_head

    @contextmanager
    def _mode(self, mode_id: int):
        previous = self._active_mode_id
        self._active_mode_id = int(mode_id)
        try:
            yield
        finally:
            self._active_mode_id = previous

    @contextmanager
    def _joint_mode(self, mode_id: int):
        previous = self._joint_mode_id
        self._joint_mode_id = int(mode_id)
        try:
            yield
        finally:
            self._joint_mode_id = previous

    def _condition(self, z0, timestep, task_condition=None):
        """Add the active A/B/C mode embedding to the AdaLN condition."""
        if timestep is None:
            raise ValueError("A/B/C condition requires a timestep")
        condition = self.z_condition(z0) + self.time_mlp(
            self._time_embedding(timestep, z0)
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
        mode = torch.full(
            (z0.shape[0],),
            int(self._active_mode_id),
            device=z0.device,
            dtype=torch.long,
        )
        return condition + self.mode_embedding(mode)

    def _time_embedding(self, timestep, reference):
        # Keep the helper local so the subclass uses exactly the legacy
        # timestep implementation while allowing explicit no-time E calls.
        from source.model.fast_lewam.modules import timestep_embedding

        if not torch.is_tensor(timestep):
            timestep = torch.as_tensor(
                timestep, device=reference.device, dtype=reference.dtype
            )
        timestep = timestep.to(device=reference.device, dtype=reference.dtype)
        if timestep.ndim == 0:
            timestep = timestep.expand(reference.shape[0])
        if tuple(timestep.shape) != (reference.shape[0],):
            raise ValueError(
                f"timestep must have shape [{reference.shape[0]}], "
                f"got {tuple(timestep.shape)}"
            )
        return timestep_embedding(timestep, self.model_dim).to(reference.dtype)

    def _round4_condition(
        self,
        reference: torch.Tensor,
        timestep: torch.Tensor | None,
        mode_id: int,
        task_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build a condition for D/E without routing through A/B state."""
        condition = self.z_condition(reference)
        if timestep is not None:
            condition = condition + self.time_mlp(
                self._time_embedding(timestep, reference)
            )
        if task_condition is not None:
            if self.task_condition is None:
                raise ValueError(
                    "task_condition was provided, but task_condition_dim is disabled"
                )
            expected = (reference.shape[0], self.task_condition_dim)
            if tuple(task_condition.shape) != expected:
                raise ValueError(
                    f"task_condition must have shape {expected}, "
                    f"got {tuple(task_condition.shape)}"
                )
            condition = condition + self.task_condition(task_condition)
        mode = torch.full(
            (reference.shape[0],),
            int(mode_id),
            device=reference.device,
            dtype=torch.long,
        )
        return condition + self.mode_embedding(mode)

    def _stage_a(self, *args, **kwargs):
        with self._mode(self.MODE_A):
            return super()._stage_a(*args, **kwargs)

    def _joint_hidden(self, *args, **kwargs):
        with self._mode(self._joint_mode_id):
            return super()._joint_hidden(*args, **kwargs)

    def _terminal_hidden(self, *args, **kwargs):
        with self._mode(self._joint_mode_id):
            return super()._terminal_hidden(*args, **kwargs)

    def predict_one_step(self, *args, **kwargs):
        with self._mode(self.MODE_B):
            return super().predict_one_step(*args, **kwargs)

    def _validate_path_inputs(self, z_start, z_goal, interior):
        if self.action_horizon < 2:
            raise ValueError(
                "Round4 D latent-path operations require action_horizon >= 2"
            )
        if z_start.ndim != 2 or tuple(z_start.shape) != tuple(z_goal.shape):
            raise ValueError(
                "z_start and z_goal must both have shape [B, latent_dim]"
            )
        if z_start.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latent inputs must have final dimension {self.latent_dim}"
            )
        expected = (z_start.shape[0], self.action_horizon - 1, self.latent_dim)
        if tuple(interior.shape) != expected:
            raise ValueError(f"interior must have shape {expected}, got {tuple(interior.shape)}")

    def predict_latent_velocity(
        self,
        z_start: torch.Tensor,
        z_goal: torch.Tensor,
        noisy_interior: torch.Tensor,
        timestep: torch.Tensor,
        task_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict D's interior velocity while keeping endpoint anchors fixed."""
        self._validate_path_inputs(z_start, z_goal, noisy_interior)
        condition = self._round4_condition(
            z_start, timestep, self.MODE_D, task_condition=task_condition
        )
        anchors = torch.stack((self.latent_input(z_start), self.latent_input(z_goal)), dim=1)
        interior = self.latent_input(noisy_interior)
        tokens = torch.cat((anchors, interior), dim=1)
        tokens = tokens + self.d_positions.to(device=tokens.device, dtype=tokens.dtype)
        hidden = self.predictor(tokens, condition, attention_mask=self.d_mask)
        return self.d_velocity_head(hidden[:, 2:])

    def predict_latent_flow(self, *args, **kwargs) -> torch.Tensor:
        """Compatibility alias for callers that name D's output a flow."""
        return self.predict_latent_velocity(*args, **kwargs)

    def predict_future_latent_velocity(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor,
        source_timestep: torch.Tensor,
        noisy_latents: torch.Tensor,
        latent_flow_timestep: torch.Tensor,
        task_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict causal prefix velocities for a noisy future latent path.

        The token sequence is ``[z0, a1, z1_tau, ..., aH, zH_tau]``.  The
        existing strict lower-triangular mask means query k can read only
        actions through k and noisy latents through k.
        """
        if not self.latent_flow_matching:
            raise RuntimeError("conditional latent flow is disabled for this model")
        if self.stage_b_attention_mode != "strict_causal":
            raise ValueError(
                "conditional latent flow requires strict_causal Stage-B attention"
            )
        source_timestep = self._validate_inputs(z0, actions, source_timestep)
        expected_latents = (
            z0.shape[0], self.action_horizon, self.latent_dim
        )
        if tuple(noisy_latents.shape) != expected_latents:
            raise ValueError(
                f"noisy_latents must have shape {expected_latents}, "
                f"got {tuple(noisy_latents.shape)}"
            )
        if not torch.is_tensor(latent_flow_timestep):
            latent_flow_timestep = torch.as_tensor(
                latent_flow_timestep, device=z0.device, dtype=z0.dtype
            )
        latent_flow_timestep = latent_flow_timestep.to(
            device=z0.device, dtype=z0.dtype
        )
        if latent_flow_timestep.ndim == 0:
            latent_flow_timestep = latent_flow_timestep.expand(z0.shape[0])
        if tuple(latent_flow_timestep.shape) != (z0.shape[0],):
            raise ValueError(
                "latent_flow_timestep must have shape "
                f"[{z0.shape[0]}], got {tuple(latent_flow_timestep.shape)}"
            )

        with self._mode(self.MODE_B), self._joint_mode(self.MODE_B):
            condition = self._condition(z0, source_timestep, task_condition)
            flow_condition = self.latent_flow_time_mlp(
                self._time_embedding(latent_flow_timestep, z0)
            )
            condition = condition + flow_condition
            batch = z0.shape[0]
            tokens = z0.new_empty(
                batch, 1 + 2 * self.action_horizon, self.model_dim
            )
            if self.token_encoding == "legacy":
                tokens[:, 0] = self.latent_input(z0)
                tokens[:, 1::2] = self.action_input(actions)
                tokens[:, 2::2] = self.query_tokens + self.latent_flow_input(
                    noisy_latents
                )
                tokens = tokens + self.joint_positions
            else:
                tokens[:, 0] = self._state_token(z0).squeeze(1)
                tokens[:, 1::2] = self._action_tokens(actions)
                tokens[:, 2::2] = (
                    self.query_content
                    + self.latent_flow_input(noisy_latents)
                    + self.time_positions[:, 1 : self.action_horizon + 1]
                    + self.type_embeddings.weight[self.QUERY_TYPE]
                )
            hidden = self.predictor(
                tokens, condition, attention_mask=self.causal_mask
            )
        return self.latent_flow_velocity_head(hidden[:, 2::2])

    def sample_future_latents(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor,
        source_timestep: torch.Tensor,
        *,
        initial_noise: torch.Tensor | None = None,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
        task_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate the conditional latent velocity from noise with Euler."""
        if not self.latent_flow_matching:
            raise RuntimeError("conditional latent flow is disabled for this model")
        steps = self.latent_flow_steps if num_steps is None else int(num_steps)
        if steps < 1:
            raise ValueError("num_steps must be positive")
        expected = (z0.shape[0], self.action_horizon, self.latent_dim)
        if initial_noise is None:
            initial_noise = torch.randn(
                expected,
                device=z0.device,
                dtype=z0.dtype,
                generator=generator,
            )
        elif tuple(initial_noise.shape) != expected:
            raise ValueError(
                f"initial_noise must have shape {expected}, "
                f"got {tuple(initial_noise.shape)}"
            )
        latents = initial_noise.to(device=z0.device, dtype=z0.dtype)
        dt = 1.0 / float(steps)
        for step in range(steps):
            flow_timestep = z0.new_full((z0.shape[0],), step * dt)
            velocity = self.predict_future_latent_velocity(
                z0,
                actions,
                source_timestep,
                latents,
                flow_timestep,
                task_condition=task_condition,
            )
            latents = latents + dt * velocity
        self.last_latent_flow_forward_count = steps
        return latents

    @torch.no_grad()
    def sample_latent_paths(
        self,
        z_start: torch.Tensor,
        z_goal: torch.Tensor,
        *,
        horizon: int | None = None,
        num_samples: int = 64,
        num_steps: int | None = None,
        flow_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate ``[B,S,H+1,D]`` paths with exact fixed endpoints."""
        if self.action_horizon < 2:
            raise ValueError(
                "Round4 D latent-path operations require action_horizon >= 2"
            )
        if flow_steps is not None:
            if num_steps is not None and int(num_steps) != int(flow_steps):
                raise ValueError("num_steps and flow_steps disagree")
            num_steps = flow_steps
        if horizon is not None and int(horizon) != self.action_horizon:
            raise ValueError(
                f"horizon={horizon} does not match action_horizon={self.action_horizon}"
            )
        num_steps = self.d_flow_steps if num_steps is None else int(num_steps)
        num_samples = int(num_samples)
        if num_samples < 1 or num_steps < 1:
            raise ValueError("num_samples and num_steps must be positive")
        if z_start.ndim != 2 or tuple(z_start.shape) != tuple(z_goal.shape):
            raise ValueError("z_start and z_goal must have identical [B, latent_dim] shapes")
        batch = z_start.shape[0]
        total = batch * num_samples
        start = z_start[:, None].expand(batch, num_samples, -1).reshape(total, -1)
        goal = z_goal[:, None].expand(batch, num_samples, -1).reshape(total, -1)
        interior = torch.randn(
            total,
            self.action_horizon - 1,
            self.latent_dim,
            device=z_start.device,
            dtype=z_start.dtype,
            generator=generator,
        )
        dt = 1.0 / float(num_steps)
        for step in range(num_steps):
            timestep = z_start.new_full((total,), step * dt)
            velocity = self.predict_latent_velocity(start, goal, interior, timestep)
            interior = interior + dt * velocity
        paths = torch.cat((start[:, None], interior, goal[:, None]), dim=1)
        return paths.reshape(batch, num_samples, self.action_horizon + 1, self.latent_dim)

    def _validate_path(self, path: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if path.ndim not in {3, 4}:
            raise ValueError("z_path must have shape [B,H+1,D] or [B,S,H+1,D]")
        if path.shape[-2:] != (self.action_horizon + 1, self.latent_dim):
            raise ValueError(
                "z_path must end in "
                f"[{self.action_horizon + 1}, {self.latent_dim}], got {tuple(path.shape)}"
            )
        leading = tuple(path.shape[:-2])
        return path.reshape(-1, self.action_horizon + 1, self.latent_dim), leading

    def predict_inverse_dynamics(
        self, z_path: torch.Tensor, task_condition: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Decode a clean latent path to normalized action blocks in one pass."""
        path, leading = self._validate_path(z_path)
        state_tokens = self.latent_input(path)
        transitions = torch.cat(
            (path[:, :-1], path[:, 1:], path[:, 1:] - path[:, :-1]), dim=-1
        )
        query_tokens = self.e_adjacent_input(transitions)
        query_tokens = query_tokens + self.e_query_content + self.e_query_positions
        state_tokens = state_tokens + self.e_state_positions
        tokens = torch.cat((state_tokens, query_tokens), dim=1)
        condition = self._round4_condition(
            path[:, 0], None, self.MODE_E, task_condition=task_condition
        )
        hidden = self.predictor(tokens, condition, attention_mask=self.e_mask)
        actions = self.inverse_dynamics_head(hidden[:, self.action_horizon + 1 :])
        return actions.reshape(*leading, self.action_horizon, self.action_dim)

    def decode_latent_paths(self, z_path: torch.Tensor) -> torch.Tensor:
        """Public E alias used by the Round 4 planner."""
        return self.predict_inverse_dynamics(z_path)

    def forward(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        *,
        mode: str = "stage_a",
        task_condition: torch.Tensor | None = None,
        goal_latent: torch.Tensor | None = None,
        detach_clean_action: bool = False,
        latent_goal: torch.Tensor | None = None,
        noisy_latents: torch.Tensor | None = None,
        latent_flow_timestep: torch.Tensor | None = None,
        latent_path: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extend the legacy forward contract with explicit D and E modes."""
        if mode == "stage_b_fm":
            if noisy_latents is None or latent_flow_timestep is None:
                raise ValueError(
                    "stage_b_fm requires noisy_latents and latent_flow_timestep"
                )
            return {
                "latent_velocity": self.predict_future_latent_velocity(
                    z0,
                    actions,
                    timestep,
                    noisy_latents,
                    latent_flow_timestep,
                    task_condition=task_condition,
                )
            }
        if mode == "stage_d":
            if latent_goal is None or noisy_latents is None:
                raise ValueError("stage_d requires latent_goal and noisy_latents")
            return {
                "latent_velocity": self.predict_latent_velocity(
                    z0,
                    latent_goal,
                    noisy_latents,
                    timestep,
                    task_condition=task_condition,
                )
            }
        if mode == "stage_e":
            if latent_path is None:
                raise ValueError("stage_e requires latent_path")
            return {"actions": self.predict_inverse_dynamics(latent_path, task_condition)}
        if mode == "stage_c":
            with self._joint_mode(self.MODE_C):
                return super().forward(
                    z0,
                    actions,
                    timestep,
                    mode=mode,
                    task_condition=task_condition,
                    goal_latent=goal_latent,
                    detach_clean_action=detach_clean_action,
                )
        if mode == "stage_ab":
            with self._joint_mode(self.MODE_B):
                return super().forward(
                    z0,
                    actions,
                    timestep,
                    mode=mode,
                    task_condition=task_condition,
                    goal_latent=goal_latent,
                    detach_clean_action=detach_clean_action,
                )
        return super().forward(
            z0,
            actions,
            timestep,
            mode=mode,
            task_condition=task_condition,
            goal_latent=goal_latent,
            detach_clean_action=detach_clean_action,
        )


SharedDiTABDE = Round4FastLeWAM


__all__ = [
    "MODE_IDS",
    "Round4FastLeWAM",
    "SharedDiTABDE",
    "d_attention_mask",
    "e_attention_mask",
]
