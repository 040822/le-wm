"""LeFlow latent-path planner integrated with this repository's LeWM.

The public runtime surface is deliberately small: a frozen LeWM, a latent path
flow model, an inverse-dynamics decoder, and a solver implementing
``stable_worldmodel``'s solver protocol.  The implementation follows the
official LeFlow checkpoint format while using this repository's checkpoint
remapping layer for the installed Transformers version.

The planner implementation is adapted from
``hsiangwei0903/LeFlow`` (MIT License; see the repository LICENSE).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from gymnasium.spaces import Box
from torch import nn


_LEWM_ALIASES = {
    "tworoom/lewm": "quentinll/lewm-tworooms",
    "tworooms/lewm": "quentinll/lewm-tworooms",
    "pusht/lewm": "quentinll/lewm-pusht",
    "reacher/lewm": "quentinll/lewm-reacher",
    "cube/lewm": "quentinll/lewm-cube",
    "ogbench/cube/lewm": "quentinll/lewm-cube",
}
_LEFLOW_PAYLOAD_KEYS = frozenset(
    {"arch", "flow_state_dict", "inverse_dynamics_state_dict", "lewm_checkpoint"}
)


def _cache_dir(sub_folder: str | None = None) -> Path:
    """Resolve the stable-worldmodel cache across supported API versions."""
    try:
        return Path(swm.data.utils.get_cache_dir(sub_folder=sub_folder))
    except TypeError:
        root = Path(swm.data.utils.get_cache_dir())
        if sub_folder is None:
            return root
        return root / sub_folder


def resolve_leflow_checkpoint(path_or_name: str | Path) -> Path:
    """Resolve a LeFlow checkpoint path or stable-worldmodel cache name."""
    path = Path(path_or_name).expanduser()
    if path.is_file():
        return path

    roots = (_cache_dir(sub_folder="checkpoints"), _cache_dir())
    candidates = []
    for root in roots:
        candidates.append(root / path)
        if path.suffix != ".pt":
            candidates.append(root / f"{path}.pt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    for root in roots:
        directory = root / path
        if directory.is_dir():
            checkpoints = sorted(
                directory.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True
            )
            if checkpoints:
                return checkpoints[0]
    raise FileNotFoundError(
        f"Could not resolve LeFlow checkpoint {path_or_name!r}; tried "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _torch_load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location="cpu")


def is_leflow_checkpoint(path: str | Path) -> bool:
    """Return whether a file has the official LeFlow planner payload shape."""
    try:
        payload = _torch_load(path)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return isinstance(payload, dict) and _LEFLOW_PAYLOAD_KEYS.issubset(payload)


def _load_local_lewm(reference: str | Path) -> nn.Module:
    """Load an official LeWM reference through the repository remapper."""
    from source.common.remap import load_pretrained_remapped

    reference_text = str(reference)
    candidates = [reference_text]
    alias = _LEWM_ALIASES.get(reference_text.lower())
    if alias is not None:
        candidates.insert(0, alias)

    errors = []
    for candidate in candidates:
        try:
            model = load_pretrained_remapped(candidate)
            if not hasattr(model, "encode") or not hasattr(model, "predict"):
                raise TypeError(f"LeWM reference {candidate!r} is not a JEPA model")
            return model.eval()
        except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"{candidate}: {exc}")

    # This fallback keeps locally saved object checkpoints usable without
    # making the normal LeFlow import path depend on checkpoint internals.
    try:
        from source.common.checkpoint import load_policy_or_model

        model, _ = load_policy_or_model(reference_text)
        if hasattr(model, "model"):
            model = model.model
        if hasattr(model, "encode") and hasattr(model, "predict"):
            return model.eval()
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(f"generic loader: {exc}")

    raise FileNotFoundError(
        f"Could not load frozen LeWM reference {reference!r}. "
        + " | ".join(errors)
    )


def load_lewm(lewm_checkpoint: str | Path) -> nn.Module:
    """Load and freeze a LeWM checkpoint referenced by a LeFlow payload."""
    model = _load_local_lewm(lewm_checkpoint)
    model.requires_grad_(False)
    return model.eval()


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=t.device))
            * torch.arange(half, device=t.device)
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None]
        embedding = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class LatentPathFlow(nn.Module):
    """Rectified-flow velocity model for latent path interiors."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 512,
        depth: int = 4,
        max_horizon: int = 20,
        time_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_horizon = max_horizon
        self.depth = depth
        self.time_dim = time_dim
        self.dropout = dropout

        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_horizon - 1, hidden_dim) * 0.02
        )
        self.token_proj = nn.Linear(latent_dim, hidden_dim)
        self.start_proj = nn.Linear(latent_dim, hidden_dim)
        self.goal_proj = nn.Linear(latent_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(layer, num_layers=depth)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim)
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z_start: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> torch.Tensor:
        n_tokens = x_t.size(1)
        if n_tokens > self.max_horizon - 1:
            raise ValueError(
                f"Requested {n_tokens + 1} horizon, but max_horizon={self.max_horizon}"
            )
        condition = (
            self.start_proj(z_start)
            + self.goal_proj(z_goal)
            + self.time_embed(t.float())
        )
        tokens = self.token_proj(x_t)
        tokens = tokens + self.pos_embedding[:, :n_tokens] + condition[:, None]
        return self.out(self.net(tokens))


class InverseDynamics(nn.Module):
    """Decode adjacent latent states into one normalized action chunk."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        depth: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout

        layers: list[nn.Module] = []
        in_dim = latent_dim * 3
        for index in range(depth):
            layers.append(nn.Linear(in_dim if index == 0 else hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat([z_t, z_next, z_next - z_t], dim=-1)
        return self.net(tokens)


class LatentPlannerRuntime(nn.Module):
    """Frozen-LeWM verification plus flow sampling and action decoding."""

    def __init__(
        self,
        lewm: nn.Module,
        flow: LatentPathFlow,
        inverse_dynamics: InverseDynamics,
        action_block: int = 5,
    ):
        super().__init__()
        if action_block < 1:
            raise ValueError("action_block must be positive")
        self.lewm = lewm.eval()
        self.flow = flow
        self.inverse_dynamics = inverse_dynamics
        self.action_block = int(action_block)
        self.rollout_count = 0
        self.timing_mode = False
        self.last_selected_indices = None
        self.lewm.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        device: str | torch.device = "cpu",
        lewm_model: nn.Module | None = None,
    ) -> "LatentPlannerRuntime":
        path = resolve_leflow_checkpoint(checkpoint)
        payload = _torch_load(path)
        if not isinstance(payload, dict) or not _LEFLOW_PAYLOAD_KEYS.issubset(payload):
            raise TypeError(
                "LeFlow checkpoints must contain arch, flow_state_dict, "
                "inverse_dynamics_state_dict, and lewm_checkpoint"
            )
        arch = payload["arch"]
        lewm = (
            load_lewm(payload["lewm_checkpoint"])
            if lewm_model is None
            else lewm_model.eval().requires_grad_(False)
        )
        flow = LatentPathFlow(**arch["flow"])
        inverse_dynamics = InverseDynamics(**arch["inverse_dynamics"])
        flow.load_state_dict(payload["flow_state_dict"], strict=True)
        inverse_dynamics.load_state_dict(
            payload["inverse_dynamics_state_dict"], strict=True
        )
        model = cls(
            lewm=lewm,
            flow=flow,
            inverse_dynamics=inverse_dynamics,
            action_block=int(payload.get("action_block", 1)),
        )
        model.checkpoint = str(path)
        return model.to(device).eval()

    @torch.no_grad()
    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim == 4:
            pixels = pixels[:, None]
        if pixels.ndim != 5:
            raise ValueError(
                f"pixels must have shape [B,T,C,H,W] or [B,C,H,W], got {tuple(pixels.shape)}"
            )
        return self.lewm.encode({"pixels": pixels.to(self.device)})["emb"]

    @torch.no_grad()
    def encode_current_and_goal(
        self, info_dict: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "pixels" not in info_dict or "goal" not in info_dict:
            raise ValueError("LeFlow planning requires pixels and goal observations")
        z_start = self.encode_pixels(info_dict["pixels"])[:, -1]
        z_goal = self.encode_pixels(info_dict["goal"])[:, -1]
        return z_start, z_goal

    @torch.no_grad()
    def sample_paths(
        self,
        z_start: torch.Tensor,
        z_goal: torch.Tensor,
        *,
        horizon: int,
        num_samples: int,
        flow_steps: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if horizon < 2:
            raise ValueError("LeFlow horizon must be at least 2")
        if num_samples < 1 or flow_steps < 1:
            raise ValueError("num_samples and flow_steps must be positive")
        batch, latent_dim = z_start.shape
        if z_goal.shape != z_start.shape:
            raise ValueError("z_start and z_goal must have identical shapes")
        total = batch * num_samples
        start = z_start[:, None].expand(batch, num_samples, latent_dim).reshape(total, latent_dim)
        goal = z_goal[:, None].expand(batch, num_samples, latent_dim).reshape(total, latent_dim)
        interior = torch.randn(
            total,
            horizon - 1,
            latent_dim,
            device=z_start.device,
            dtype=z_start.dtype,
            generator=generator,
        )
        dt = 1.0 / float(flow_steps)
        for index in range(flow_steps):
            timestep = torch.full(
                (total,), index * dt, device=z_start.device, dtype=z_start.dtype
            )
            interior = interior + self.flow(interior, timestep, start, goal) * dt
        path = torch.cat([start[:, None], interior, goal[:, None]], dim=1)
        return path.reshape(batch, num_samples, horizon + 1, latent_dim)

    def decode_actions(self, paths: torch.Tensor) -> torch.Tensor:
        if paths.ndim != 4:
            raise ValueError("paths must have shape [B,S,H+1,D]")
        z_t = paths[..., :-1, :]
        z_next = paths[..., 1:, :]
        actions = self.inverse_dynamics(
            z_t.reshape(-1, z_t.size(-1)), z_next.reshape(-1, z_next.size(-1))
        )
        return actions.reshape(*z_t.shape[:-1], -1)

    @torch.no_grad()
    def rollout_final_latent(
        self,
        z_start: torch.Tensor,
        actions: torch.Tensor,
        history_size: int = 3,
    ) -> torch.Tensor:
        if history_size < 1:
            raise ValueError("history_size must be positive")
        batch, samples, horizon = actions.shape[:3]
        state = (
            z_start[:, None, None]
            .expand(batch, samples, 1, -1)
            .reshape(batch * samples, 1, -1)
            .clone()
        )
        flat_actions = actions.reshape(batch * samples, horizon, -1).to(self.device)
        for index in range(horizon):
            action_history = flat_actions[:, max(0, index - history_size + 1) : index + 1]
            state_history = state[:, -action_history.size(1) :]
            action_embeddings = self.lewm.action_encoder(action_history)
            prediction = self.lewm.predict(state_history, action_embeddings)[:, -1:]
            state = torch.cat([state, prediction], dim=1)
        self.rollout_count += batch * samples
        return state[:, -1].reshape(batch, samples, -1)

    @torch.no_grad()
    def rollout_paths(
        self,
        z_start: torch.Tensor,
        actions: torch.Tensor,
        history_size: int = 3,
    ) -> torch.Tensor:
        batch, samples, horizon = actions.shape[:3]
        state = (
            z_start[:, None, None]
            .expand(batch, samples, 1, -1)
            .reshape(batch * samples, 1, -1)
            .clone()
        )
        flat_actions = actions.reshape(batch * samples, horizon, -1).to(self.device)
        for index in range(horizon):
            action_history = flat_actions[:, max(0, index - history_size + 1) : index + 1]
            state_history = state[:, -action_history.size(1) :]
            action_embeddings = self.lewm.action_encoder(action_history)
            prediction = self.lewm.predict(state_history, action_embeddings)[:, -1:]
            state = torch.cat([state, prediction], dim=1)
        return state.reshape(batch, samples, horizon + 1, -1)

    @torch.no_grad()
    def plan(
        self,
        info_dict: dict[str, Any],
        *,
        horizon: int,
        num_samples: int,
        flow_steps: int,
        score_mode: str = "rollout_goal",
        goal_weight: float = 1.0,
        consistency_weight: float = 0.0,
        smoothness_weight: float = 0.0,
        history_size: int = 3,
        generator: torch.Generator | None = None,
        capture_diagnostics: bool = True,
    ) -> dict[str, torch.Tensor]:
        z_start, z_goal = self.encode_current_and_goal(info_dict)
        paths = self.sample_paths(
            z_start,
            z_goal,
            horizon=horizon,
            num_samples=num_samples,
            flow_steps=flow_steps,
            generator=generator,
        )
        actions = self.decode_actions(paths)
        mode = score_mode.lower()
        if mode == "rollout_goal":
            rollout_final = self.rollout_final_latent(z_start, actions, history_size)
            goal_cost = F.mse_loss(
                rollout_final,
                z_goal[:, None].expand_as(rollout_final),
                reduction="none",
            ).mean(dim=-1)
            cost = goal_weight * goal_cost
        elif mode == "first":
            goal_cost = torch.zeros(
                actions.shape[:2], device=actions.device, dtype=actions.dtype
            )
            cost = goal_cost.clone()
        elif mode == "path_smoothness":
            goal_cost = torch.zeros(
                actions.shape[:2], device=actions.device, dtype=actions.dtype
            )
            acceleration = paths[:, :, 2:] - 2 * paths[:, :, 1:-1] + paths[:, :, :-2]
            cost = acceleration.square().mean(dim=(-1, -2))
        else:
            raise ValueError(
                f"Unknown LeFlow score_mode={score_mode!r}; expected "
                "rollout_goal, first, or path_smoothness"
            )

        if consistency_weight:
            rollout = self.rollout_paths(z_start, actions, history_size)
            consistency = (rollout - paths).square().mean(dim=(-1, -2))
            cost = cost + consistency_weight * consistency
        if smoothness_weight:
            acceleration = paths[:, :, 2:] - 2 * paths[:, :, 1:-1] + paths[:, :, :-2]
            cost = cost + smoothness_weight * acceleration.square().mean(dim=(-1, -2))

        best = cost.argmin(dim=1)
        self.last_selected_indices = best.detach()
        batch_indices = torch.arange(actions.size(0), device=actions.device)
        result = {"actions": actions[batch_indices, best].detach().cpu()}
        if capture_diagnostics:
            result.update(
                {
                    "costs": cost[batch_indices, best].detach().cpu(),
                    "all_costs": cost.detach().cpu(),
                    "goal_costs": goal_cost.detach().cpu(),
                    "selected_indices": best.detach().cpu(),
                    "selected_action_sequences": actions[
                        batch_indices, best
                    ].detach().cpu(),
                }
            )
        return result


class LearnedLatentPathSolver:
    """``stable_worldmodel`` solver adapter for a LeFlow runtime."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        batch_size: int = 1,
        num_samples: int = 64,
        flow_steps: int = 16,
        score_mode: str = "rollout_goal",
        goal_weight: float = 1.0,
        consistency_weight: float = 0.0,
        smoothness_weight: float = 0.0,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        history_size: int = 3,
        model: LatentPlannerRuntime | None = None,
    ):
        if model is None and checkpoint is None:
            raise ValueError("either checkpoint or model must be provided")
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.batch_size = int(batch_size)
        self.num_samples = int(num_samples)
        self.flow_steps = int(flow_steps)
        self.score_mode = score_mode
        self.goal_weight = float(goal_weight)
        self.consistency_weight = float(consistency_weight)
        self.smoothness_weight = float(smoothness_weight)
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")
        self.device = requested_device
        self.history_size = int(history_size)
        self.timing_mode = False
        self.capture_timing_selection = False
        self.last_timing_selection = None
        self.planning_events: list[dict[str, Any]] = []
        self.model = model or LatentPlannerRuntime.from_checkpoint(
            checkpoint, device=self.device
        )
        self.model.to(self.device).eval()
        self.torch_gen = torch.Generator(device=self.device).manual_seed(int(seed))

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        self._action_space = action_space
        self._n_envs = int(n_envs)
        self._config = config
        if not isinstance(action_space, Box):
            raise TypeError(
                f"LearnedLatentPathSolver expects Box action space, got {type(action_space)}"
            )
        self._action_dim = int(np.prod(action_space.shape[1:]))
        expected = self._action_dim * int(config.action_block)
        actual = int(self.model.inverse_dynamics.action_dim)
        if expected != actual:
            raise ValueError(
                f"LeFlow action dimension mismatch: checkpoint={actual}, "
                f"environment={expected} ({self._action_dim} * action_block {config.action_block})"
            )

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * int(self._config.action_block)

    @property
    def horizon(self) -> int:
        return int(self._config.horizon)

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.solve(*args, **kwargs)

    @torch.inference_mode()
    def solve(
        self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None
    ) -> dict[str, Any]:
        del init_action
        total_envs = len(next(iter(info_dict.values())))
        actions = []
        costs = []
        goal_costs = []
        for start in range(0, total_envs, self.batch_size):
            end = min(start + self.batch_size, total_envs)
            batch = {
                key: value[start:end] if hasattr(value, "__getitem__") else value
                for key, value in info_dict.items()
            }
            output = self.model.plan(
                batch,
                horizon=self.horizon,
                num_samples=self.num_samples,
                flow_steps=self.flow_steps,
                score_mode=self.score_mode,
                goal_weight=self.goal_weight,
                consistency_weight=self.consistency_weight,
                smoothness_weight=self.smoothness_weight,
                history_size=self.history_size,
                generator=self.torch_gen,
                capture_diagnostics=not self.timing_mode,
            )
            actions.append(output["actions"])
            if self.timing_mode:
                if self.capture_timing_selection:
                    self.last_timing_selection = self.model.last_selected_indices.detach()
            else:
                costs.append(output["costs"])
                goal_costs.append(output["goal_costs"])
                self.planning_events.append(
                    {
                        "selected_indices": output["selected_indices"].tolist(),
                        "candidate_costs": output["all_costs"].tolist(),
                        "goal_costs": output["goal_costs"].tolist(),
                        "selected_action_sequences": output[
                            "selected_action_sequences"
                        ].tolist(),
                        "candidate_count": int(self.num_samples),
                        "flow_steps": int(self.flow_steps),
                        "rollout_count": int(self.model.rollout_count),
                    }
                )
        result = {
            "actions": torch.cat(actions, dim=0),
            "rollout_count": self.model.rollout_count,
        }
        if not self.timing_mode:
            result["costs"] = torch.cat(costs, dim=0).tolist()
            result["goal_costs"] = torch.cat(goal_costs, dim=0)
        return result


def flow_matching_loss(
    flow: LatentPathFlow,
    z_path: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Train flow velocity on the interior latent path points."""
    z_start = z_path[:, 0]
    z_goal = z_path[:, -1]
    target = z_path[:, 1:-1]
    noise = torch.randn(
        target.shape,
        device=target.device,
        dtype=target.dtype,
        generator=generator,
    )
    timestep = torch.rand(
        z_path.size(0), device=z_path.device, dtype=z_path.dtype, generator=generator
    )
    x_t = (1 - timestep[:, None, None]) * noise + timestep[:, None, None] * target
    predicted_velocity = flow(x_t, timestep, z_start, z_goal)
    return F.mse_loss(predicted_velocity, target - noise)


def inverse_dynamics_loss(
    inverse_dynamics: InverseDynamics,
    z_path: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train inverse dynamics on every adjacent latent transition."""
    predicted = inverse_dynamics(
        z_path[:, :-1].reshape(-1, z_path.size(-1)),
        z_path[:, 1:].reshape(-1, z_path.size(-1)),
    )
    predicted = predicted.reshape(z_path.size(0), z_path.size(1) - 1, -1)
    return F.mse_loss(predicted, actions), predicted


def lewm_consistency_loss(
    lewm: nn.Module,
    z_path: torch.Tensor,
    predicted_actions: torch.Tensor,
    history_size: int = 3,
) -> torch.Tensor:
    """Measure whether decoded actions reproduce the training latent path."""
    losses = []
    for index in range(predicted_actions.size(1)):
        start = max(0, index - history_size + 1)
        embedding_history = z_path[:, start : index + 1]
        action_history = predicted_actions[:, start : index + 1]
        action_embeddings = lewm.action_encoder(action_history)
        predicted = lewm.predict(embedding_history, action_embeddings)[:, -1]
        losses.append(F.mse_loss(predicted, z_path[:, index + 1]))
    return torch.stack(losses).mean()


def smoothness_loss(z_path: torch.Tensor) -> torch.Tensor:
    if z_path.size(1) < 3:
        return z_path.new_tensor(0.0)
    acceleration = z_path[:, 2:] - 2 * z_path[:, 1:-1] + z_path[:, :-2]
    return acceleration.square().mean()


def checkpoint_payload(
    *,
    lewm_checkpoint: str,
    action_block: int,
    flow: LatentPathFlow,
    inverse_dynamics: InverseDynamics,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Build the official-compatible lightweight LeFlow checkpoint payload."""
    return {
        "lewm_checkpoint": lewm_checkpoint,
        "action_block": int(action_block),
        "arch": {
            "flow": {
                "latent_dim": flow.latent_dim,
                "hidden_dim": flow.hidden_dim,
                "depth": flow.depth,
                "max_horizon": flow.max_horizon,
                "time_dim": flow.time_dim,
                "dropout": flow.dropout,
            },
            "inverse_dynamics": {
                "latent_dim": inverse_dynamics.latent_dim,
                "action_dim": inverse_dynamics.action_dim,
                "hidden_dim": inverse_dynamics.hidden_dim,
                "depth": inverse_dynamics.depth,
                "dropout": inverse_dynamics.dropout,
            },
        },
        "config": cfg,
        "flow_state_dict": flow.state_dict(),
        "inverse_dynamics_state_dict": inverse_dynamics.state_dict(),
    }


__all__ = [
    "InverseDynamics",
    "LatentPathFlow",
    "LatentPlannerRuntime",
    "LearnedLatentPathSolver",
    "checkpoint_payload",
    "flow_matching_loss",
    "inverse_dynamics_loss",
    "is_leflow_checkpoint",
    "lewm_consistency_loss",
    "load_lewm",
    "resolve_leflow_checkpoint",
    "smoothness_loss",
]
