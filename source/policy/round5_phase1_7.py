"""Independent LeWM scorer adapter for Round 5 Phase1.7."""

from __future__ import annotations

from collections import deque
from math import ceil

import numpy as np
import torch
from torch import nn


class LeWMStageBVerifier(nn.Module):
    """Score an action proposal with a separately trained, autoregressive LeWM.

    The caller supplies the real image/action history at the model's training
    frame skip. Candidate actions are already transformed by the shared eval
    action normalizer and clipped in physical action space.
    """

    def __init__(
        self,
        world_model: nn.Module,
        *,
        history_size: int,
        action_dim: int,
        action_horizon: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        if history_size < 1 or action_dim < 1 or action_horizon < 1 or latent_dim < 1:
            raise ValueError("LeWM verifier dimensions must be positive")
        self.world_model = world_model
        self.history_size = int(history_size)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.latent_dim = int(latent_dim)
        self.last_forward_count = 0

    @staticmethod
    def _last_frame(pixels: torch.Tensor) -> torch.Tensor:
        return pixels if pixels.ndim == 4 else pixels.select(dim=-4, index=-1)

    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.world_model.encode({"pixels": pixels})["emb"]

    def encode_planner_context(self, info: dict):
        """Encode buffered real history and goal in LeWM's own latent space."""
        if "_phase17_history_pixels" not in info or "_phase17_history_actions" not in info:
            raise ValueError("Phase1.7 LeWM scorer requires its real history buffer")
        history = info["_phase17_history_pixels"].float()
        goal_pixels = self._last_frame(info["goal"]).float()
        history_latents = self.encode_pixels(history)
        goal_latent = self.encode_pixels(goal_pixels[:, None])[:, 0]
        return history_latents, goal_latent

    def score_action_candidates(
        self,
        history_latents: torch.Tensor,
        goal_latent: torch.Tensor,
        candidates: torch.Tensor,
        *,
        context: dict,
        candidate_batch_size: int,
        solver_batch_size: int,
    ) -> torch.Tensor:
        """Autoregressively predict every candidate's terminal latent distance."""
        if context is None or "_phase17_history_actions" not in context:
            raise ValueError("LeWM candidate scoring requires buffered history actions")
        if candidates.ndim != 4:
            raise ValueError("candidates must have shape [B,N,T,A]")
        batch, count, horizon, action_dim = candidates.shape
        if action_dim != self.action_dim or horizon != self.action_horizon:
            raise ValueError(
                "candidate shape does not match the LeWM verifier: "
                f"{tuple(candidates.shape)}"
            )
        if history_latents.shape[:2] != (batch, self.history_size):
            raise ValueError("encoded LeWM history has the wrong shape")
        history_actions = context["_phase17_history_actions"].to(
            device=candidates.device, dtype=candidates.dtype
        )
        if tuple(history_actions.shape) != (
            batch,
            self.history_size - 1,
            self.action_dim,
        ):
            raise ValueError("LeWM history actions have the wrong shape")
        candidate_chunk = max(1, int(candidate_batch_size))
        row_chunk = max(1, int(solver_batch_size))
        cost_rows = []
        chunks_per_row = ceil(count / candidate_chunk)
        for row_start in range(0, batch, row_chunk):
            row_end = min(batch, row_start + row_chunk)
            row_costs = []
            for start in range(0, count, candidate_chunk):
                actions = candidates[row_start:row_end, start : start + candidate_chunk]
                local_batch, local_count = actions.shape[:2]
                past = history_actions[row_start:row_end, None].expand(
                    -1, local_count, -1, -1
                )
                sequence_actions = torch.cat((past, actions), dim=2)
                embeddings = history_latents[row_start:row_end, None].expand(
                    -1, local_count, -1, -1
                ).reshape(
                    local_batch * local_count, self.history_size, self.latent_dim
                )
                sequence_actions = sequence_actions.reshape(
                    local_batch * local_count,
                    self.history_size - 1 + horizon,
                    self.action_dim,
                )
                for step in range(horizon):
                    action_window = sequence_actions[:, step : step + self.history_size]
                    action_embeddings = self.world_model.action_encoder(action_window)
                    predicted = self.world_model.predict(
                        embeddings[:, -self.history_size :], action_embeddings
                    )[:, -1:]
                    embeddings = torch.cat((embeddings, predicted), dim=1)
                terminal = embeddings[:, -1].reshape(
                    local_batch, local_count, self.latent_dim
                )
                row_costs.append(
                    (terminal - goal_latent[row_start:row_end, None]).square().sum(dim=-1)
                )
            cost_rows.append(torch.cat(row_costs, dim=1))
        self.last_forward_count = ceil(batch / row_chunk) * chunks_per_row * (horizon + 2)
        return torch.cat(cost_rows, dim=0)


class _ActorWarmStartLeWMCEMModelView(nn.Module):
    """Keep Joint actor warm-start proposals while LeWM scores CEM candidates."""

    _CACHE_HISTORY = "_phase17_cem_history_latents"
    _CACHE_GOAL = "_phase17_cem_goal_latent"
    _CACHE_ACTIONS = "_phase17_cem_history_actions"

    def __init__(self, actor_view, verifier, *, candidate_batch_size, solver_batch_size):
        super().__init__()
        self.actor_view = actor_view
        self.verifier = verifier
        self.candidate_batch_size = int(candidate_batch_size)
        self.solver_batch_size = int(solver_batch_size)
        if self.candidate_batch_size < 1 or self.solver_batch_size < 1:
            raise ValueError("CEM verifier batch sizes must be positive")
        for name in ("latent_dim", "action_dim", "action_horizon"):
            if getattr(actor_view.model, name, None) != getattr(verifier, name, None):
                raise ValueError(f"CEM actor and LeWM must share {name}")

    def get_action(self, *args, **kwargs):
        return self.actor_view.get_action(*args, **kwargs)

    def set_action_bounds(self, bounds):
        return self.actor_view.set_action_bounds(bounds)

    @staticmethod
    def _first_candidate(value, candidates):
        if (
            torch.is_tensor(value)
            and value.ndim >= 2
            and value.shape[1] == candidates.shape[1]
            and value.stride(1) == 0
        ):
            return value[:, 0]
        return value

    def get_cost(self, info_dict, action_candidates):
        if action_candidates.ndim != 4:
            raise ValueError("CEM action candidates must have shape [B,N,H,A]")
        if self._CACHE_HISTORY not in info_dict:
            context = {
                "_phase17_history_pixels": self._first_candidate(
                    info_dict.get("_phase17_history_pixels"), action_candidates
                ),
                "_phase17_history_actions": self._first_candidate(
                    info_dict.get("_phase17_history_actions"), action_candidates
                ),
                "goal": self._first_candidate(info_dict.get("goal"), action_candidates),
            }
            if any(value is None for value in context.values()):
                raise ValueError("LeWM CEM scoring requires real history and goal inputs")
            history_latents, goal_latent = self.verifier.encode_planner_context(context)
            info_dict[self._CACHE_HISTORY] = history_latents
            info_dict[self._CACHE_GOAL] = goal_latent
            info_dict[self._CACHE_ACTIONS] = context["_phase17_history_actions"]
        return self.verifier.score_action_candidates(
            info_dict[self._CACHE_HISTORY],
            info_dict[self._CACHE_GOAL],
            action_candidates,
            context={
                "_phase17_history_actions": info_dict[self._CACHE_ACTIONS]
            },
            candidate_batch_size=self.candidate_batch_size,
            solver_batch_size=self.solver_batch_size,
        )


class Phase17LeWMCEMPolicy:
    """Attach a real LeWM history stream to the standard actor-warm-started CEM."""

    def __init__(
        self,
        policy,
        verifier,
        *,
        transform,
        candidate_batch_size,
        solver_batch_size,
        device,
    ):
        object.__setattr__(self, "inner", policy)
        object.__setattr__(self, "_verifier", verifier.to(device).eval())
        object.__setattr__(self, "_transform", transform or {})
        object.__setattr__(self, "_candidate_batch_size", int(candidate_batch_size))
        object.__setattr__(self, "_solver_batch_size", int(solver_batch_size))
        object.__setattr__(self, "_history_groups", None)
        object.__setattr__(self, "_history_steps", None)
        object.__setattr__(self, "_previous_actions", None)
        solver = policy.solver
        base_solver = getattr(solver, "_solver", solver)
        actor_view = getattr(base_solver, "model", None)
        if actor_view is None or not hasattr(actor_view, "get_action"):
            raise TypeError("P2 CEM solver does not expose its actor warm-start view")
        scorer_view = _ActorWarmStartLeWMCEMModelView(
            actor_view,
            self._verifier,
            candidate_batch_size=self._candidate_batch_size,
            solver_batch_size=self._solver_batch_size,
        )
        base_solver.model = scorer_view
        if solver is not base_solver and hasattr(solver, "model"):
            solver.model = scorer_view

    def __getattr__(self, name):
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def __setattr__(self, name, value):
        if name == "inner" or name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        setattr(self.inner, name, value)

    def set_env(self, env):
        self.inner.set_env(env)
        size = int(self._verifier.history_size)
        object.__setattr__(self, "_history_groups", [deque(maxlen=size) for _ in range(env.num_envs)])
        object.__setattr__(self, "_history_steps", [0 for _ in range(env.num_envs)])
        object.__setattr__(self, "_previous_actions", [None for _ in range(env.num_envs)])

    def _observe_real_history(self, info_dict):
        if "pixels" not in info_dict:
            raise ValueError("LeWM CEM evaluation requires pixel observations")
        block = int(self.inner.fast_action_block)
        needs_flush = info_dict.get("_needs_flush")
        if needs_flush is not None:
            flags = np.asarray(needs_flush, dtype=bool).reshape(-1)
            for index, flush in enumerate(flags):
                if flush:
                    self._history_groups[index].clear()
                    self._history_steps[index] = 0
                    self._previous_actions[index] = None
        pixels = info_dict["pixels"]
        for index, groups in enumerate(self._history_groups):
            previous = self._previous_actions[index]
            if previous is not None and groups:
                groups[-1]["actions"].append(previous)
            step = self._history_steps[index]
            if step % block == 0:
                image = np.asarray(pixels[index])
                if image.ndim == 4:
                    image = image[-1]
                if image.ndim != 3:
                    raise ValueError(
                        "LeWM history pixels must reduce to [H,W,C], "
                        f"got {tuple(image.shape)}"
                    )
                groups.append({"pixels": image.copy(), "actions": []})
            self._history_steps[index] += 1

    def _history_context(self, *, active_rows):
        from torchvision import tv_tensors

        history_size = int(self._verifier.history_size)
        action_width = int(self.inner.fast_model.action_dim)
        block = int(self.inner.fast_action_block)
        pixel_rows = []
        action_rows = []
        active_rows = set(int(value) for value in active_rows)
        for row_index, groups_buffer in enumerate(self._history_groups):
            groups = list(groups_buffer)[-history_size:]
            if not groups:
                raise RuntimeError("LeWM CEM history buffer has no current observation")
            padding = history_size - len(groups)
            first = groups[0]["pixels"]
            pixels = [first.copy() for _ in range(padding)]
            pixels.extend(group["pixels"] for group in groups)
            actions = [np.zeros(action_width, dtype=np.float32) for _ in range(padding)]
            for group in groups[:-1]:
                values = list(group["actions"])
                if len(values) != block and row_index in active_rows:
                    raise RuntimeError(
                        "LeWM CEM history action block is incomplete: "
                        f"expected {block}, found {len(values)}"
                    )
                if len(values) > block:
                    raise RuntimeError("LeWM CEM history action block is too long")
                values.extend(
                    np.zeros(action_width // block, dtype=np.float32)
                    for _ in range(block - len(values))
                )
                actions.append(np.concatenate(values, axis=0).astype(np.float32))
            if len(actions) != history_size - 1:
                raise RuntimeError("LeWM CEM history action prefix has the wrong length")
            transformed = []
            for image in pixels:
                tensor = torch.from_numpy(np.ascontiguousarray(image))
                if tensor.ndim != 3:
                    raise ValueError("LeWM history image must be rank 3")
                if tensor.shape[-1] in (1, 3, 4):
                    tensor = tensor.permute(2, 0, 1)
                transformed.append(
                    self._transform["pixels"](tv_tensors.Image(tensor))
                )
            pixel_rows.append(torch.stack(transformed))
            action_rows.append(np.stack(actions))
        return torch.stack(pixel_rows), torch.from_numpy(np.stack(action_rows))

    def get_action(self, info_dict, **kwargs):
        self._observe_real_history(info_dict)
        count = int(self.inner.env.num_envs)
        terminated = np.asarray(
            info_dict.get("terminated", np.zeros(count, dtype=bool)), dtype=bool
        ).reshape(-1)
        flush = np.asarray(
            info_dict.get("_needs_flush", np.zeros(count, dtype=bool)), dtype=bool
        ).reshape(-1)
        active_rows = [
            index
            for index in range(count)
            if not terminated[index]
            and (flush[index] or not self.inner._action_buffer[index])
        ]
        history_pixels, history_actions = self._history_context(
            active_rows=active_rows
        )
        enriched = dict(info_dict)
        enriched["_phase17_history_pixels"] = history_pixels
        enriched["_phase17_history_actions"] = history_actions
        action = self.inner.get_action(enriched, **kwargs)
        rows = np.asarray(action).reshape(self.inner.env.num_envs, -1)
        normalized = self.inner.process["action"].transform(rows)
        self._previous_actions = [
            row.astype(np.float32, copy=True) if np.isfinite(row).all() else None
            for row in normalized
        ]
        return action

    def metadata(self):
        return {
            "separate_verifier_model": True,
            "verifier_history_size": int(self._verifier.history_size),
            "verifier_candidate_batch_size": self._candidate_batch_size,
            "verifier_solver_batch_size": self._solver_batch_size,
            "verifier_action_space": "actor_cem_candidates_in_shared_eval_normalization",
        }

    def __call__(self, *args, **kwargs):
        return self.get_action(*args, **kwargs)


def attach_lewm_cem_verifier(
    policy,
    verifier,
    *,
    transform,
    candidate_batch_size,
    solver_batch_size,
    device,
):
    """Use actor warm starts and independently trained LeWM CEM costs."""
    return Phase17LeWMCEMPolicy(
        policy,
        verifier,
        transform=transform,
        candidate_batch_size=candidate_batch_size,
        solver_batch_size=solver_batch_size,
        device=device,
    )
