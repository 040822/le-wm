"""Round 4 planning policies and best-of-N candidate selection."""

from __future__ import annotations

from collections import deque
import time
from collections.abc import Mapping

import numpy as np
import stable_worldmodel as swm
import torch

from source.policy.fast_lewam_eval import _validate_action_dim, make_fast_lewam_policy
from source.policy.lewm import _to_container


ROUND4_MODES = ("P0", "P0-shuf", "P1", "P2", "P3", "P4", "P4-first")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def score_candidates_in_chunks(
    verifier,
    z_start: torch.Tensor,
    z_goal: torch.Tensor,
    candidates: torch.Tensor,
    *,
    solver_batch_size: int = 1,
    candidate_batch_size: int | None = None,
) -> torch.Tensor:
    """Score action candidates with B without exposing latent proposal paths.

    Splitting either the environment batch or candidate dimension is an
    execution detail.  The returned ``[B,S]`` costs and their argmin are
    therefore invariant to the split sizes.
    """
    if candidates.ndim != 4:
        raise ValueError("candidates must have shape [B,S,H,A]")
    if z_start.ndim != 2 or z_goal.shape != z_start.shape:
        raise ValueError("z_start and z_goal must have identical [B,D] shapes")
    if candidates.shape[0] != z_start.shape[0]:
        raise ValueError("candidate and context batch sizes differ")
    solver_batch_size = int(solver_batch_size)
    candidate_batch_size = (
        candidates.shape[1]
        if candidate_batch_size is None
        else int(candidate_batch_size)
    )
    if solver_batch_size < 1 or candidate_batch_size < 1:
        raise ValueError("solver_batch_size and candidate_batch_size must be positive")
    rows = []
    with torch.no_grad():
        for row_start in range(0, candidates.shape[0], solver_batch_size):
            row_end = min(row_start + solver_batch_size, candidates.shape[0])
            row_costs = []
            for col_start in range(0, candidates.shape[1], candidate_batch_size):
                col_end = min(col_start + candidate_batch_size, candidates.shape[1])
                row_costs.append(
                    verifier(
                        z_start[row_start:row_end],
                        z_goal[row_start:row_end],
                        candidates[row_start:row_end, col_start:col_end],
                    )
                )
            rows.append(torch.cat(row_costs, dim=1))
    return torch.cat(rows, dim=0)


class Round4BestOfNPolicy(swm.policy.BasePolicy):
    """Environment-facing P3/P4 policy with a fixed candidate budget."""

    def __init__(
        self,
        model,
        *,
        proposal_source: str,
        num_candidates: int = 64,
        flow_steps: int = 16,
        action_flow_steps: int | None = None,
        action_block: int = 5,
        receding_horizon_blocks: int = 5,
        solver_batch_size: int = 1,
        candidate_batch_size: int | None = None,
        verifier: str = "stage_b",
        selection_rule: str = "argmin_verifier",
        process=None,
        transform=None,
        seed: int = 42,
    ):
        super().__init__()
        if proposal_source not in {"action", "latent"}:
            raise ValueError("proposal_source must be 'action' or 'latent'")
        if int(num_candidates) < 1 or int(flow_steps) < 1:
            raise ValueError("num_candidates and flow_steps must be positive")
        if verifier not in {"stage_b", "none"}:
            raise ValueError("verifier must be 'stage_b' or 'none'")
        if selection_rule not in {"argmin_verifier", "first"}:
            raise ValueError("selection_rule must be 'argmin_verifier' or 'first'")
        if selection_rule == "first" and verifier != "none":
            raise ValueError("first selection must disable the verifier")
        if action_block < 1 or receding_horizon_blocks < 1:
            raise ValueError("action_block and receding_horizon_blocks must be positive")
        self.type = "round4_best_of_n"
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.proposal_source = proposal_source
        self.num_candidates = int(num_candidates)
        self.flow_steps = int(flow_steps)
        self.action_flow_steps = int(
            flow_steps if action_flow_steps is None else action_flow_steps
        )
        self.action_block = int(action_block)
        self.receding_horizon_blocks = int(receding_horizon_blocks)
        self.solver_batch_size = int(solver_batch_size)
        self.candidate_batch_size = candidate_batch_size
        self.verifier = verifier
        self.selection_rule = selection_rule
        self.process = process or {}
        self.transform = transform or {}
        self.seed = int(seed)
        self._generators = {}
        self._action_buffer = None
        self.planning_events: list[dict] = []

    def _generator(self, device):
        key = str(device)
        if key not in self._generators:
            self._generators[key] = torch.Generator(device=device).manual_seed(self.seed)
        return self._generators[key]

    def set_seed(self, seed):
        self.seed = int(seed)
        self._generators.clear()

    def set_env(self, env):
        self.env = env
        self._action_buffer = [deque() for _ in range(env.num_envs)]
        _validate_action_dim(self.model, env, self.action_block)

    @staticmethod
    def _slice_info(info, indices):
        result = {}
        for key, value in info.items():
            if torch.is_tensor(value):
                result[key] = value[indices]
            elif isinstance(value, np.ndarray):
                result[key] = value[indices]
            elif isinstance(value, list):
                result[key] = [value[index] for index in indices]
            else:
                result[key] = value
        return result

    def _prepare(self, info):
        selected = self._prepare_info(info)
        device = next(self.model.parameters()).device
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in selected.items()
        }

    def _encode_context(self, info):
        current = self.model._last_frame(info["pixels"])
        goal = self.model._last_frame(info["goal"])
        encoded = self.model.encode_pixels(torch.cat((current, goal), dim=0))
        return encoded[: current.shape[0]], encoded[current.shape[0] :]

    def _propose(self, z_start, z_goal, generator):
        batch = z_start.shape[0]
        if self.proposal_source == "action":
            flat_start = z_start[:, None].expand(batch, self.num_candidates, -1).reshape(-1, z_start.shape[-1])
            flat_goal = z_goal[:, None].expand(batch, self.num_candidates, -1).reshape(-1, z_goal.shape[-1])
            noise = torch.randn(
                batch * self.num_candidates,
                self.model.action_horizon,
                self.model.action_dim,
                device=z_start.device,
                dtype=z_start.dtype,
                generator=generator,
            )
            started = time.perf_counter()
            actions = self.model.sample_actions(
                flat_start,
                noise=noise,
                num_steps=self.action_flow_steps,
                goal_latent=flat_goal,
            )
            flow_seconds = time.perf_counter() - started
            return actions.reshape(
                batch, self.num_candidates, self.model.action_horizon, self.model.action_dim
            ), {
                "flow_steps": self.action_flow_steps,
                "idm_type": "none",
                "forward_count": self.action_flow_steps,
                "flow_seconds": float(flow_seconds),
                "idm_seconds": 0.0,
            }
        started = time.perf_counter()
        paths = self.model.sample_latent_paths(
            z_start,
            z_goal,
            num_samples=self.num_candidates,
            num_steps=self.flow_steps,
            generator=generator,
        )
        flow_seconds = time.perf_counter() - started
        started = time.perf_counter()
        actions = self.model.decode_latent_paths(paths)
        idm_seconds = time.perf_counter() - started
        return actions, {
            "flow_steps": self.flow_steps,
            "idm_type": "direct_shared_dit",
            "forward_count": self.flow_steps + 1,
            "flow_seconds": float(flow_seconds),
            "idm_seconds": float(idm_seconds),
        }

    def score_candidates(self, z_start, z_goal, candidates):
        return score_candidates_in_chunks(
            lambda start, goal, actions: self.model.get_cost_from_latents(
                start, goal, actions
            ),
            z_start,
            z_goal,
            candidates,
            solver_batch_size=self.solver_batch_size,
            candidate_batch_size=self.candidate_batch_size,
        )

    def metadata(self) -> dict:
        return {
            "proposal_source": self.proposal_source,
            "candidate_count": self.num_candidates,
            "flow_steps": self.flow_steps,
            "action_flow_steps": self.action_flow_steps,
            "idm_type": "direct_shared_dit" if self.proposal_source == "latent" else "none",
            "verifier": self.verifier,
            "selection_rule": self.selection_rule,
            "solver_batch_size": self.solver_batch_size,
            "candidate_batch_size": self.candidate_batch_size,
        }

    def get_action(self, info_dict, **kwargs):
        del kwargs
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
            selected = self._prepare(self._slice_info(info_dict, replan))
            device = next(self.model.parameters()).device
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            generator = self._generator(device)
            _sync(device)
            started = time.perf_counter()
            z_start, z_goal = self._encode_context(selected)
            _sync(device)
            encode_seconds = time.perf_counter() - started

            started = time.perf_counter()
            with torch.no_grad():
                candidates, proposal_meta = self._propose(z_start, z_goal, generator)
            _sync(device)
            proposal_seconds = time.perf_counter() - started

            costs = None
            if self.verifier == "stage_b":
                started = time.perf_counter()
                with torch.no_grad():
                    costs = self.score_candidates(z_start, z_goal, candidates)
                _sync(device)
                verify_seconds = time.perf_counter() - started
                selected_indices = costs.argmin(dim=1)
                row_batches = (len(replan) + self.solver_batch_size - 1) // self.solver_batch_size
                candidate_batches = (
                    self.num_candidates + (self.candidate_batch_size or self.num_candidates) - 1
                ) // (self.candidate_batch_size or self.num_candidates)
                verifier_forward_count = row_batches * candidate_batches
            else:
                verify_seconds = 0.0
                selected_indices = torch.zeros(
                    len(replan), dtype=torch.long, device=candidates.device
                )
                verifier_forward_count = 0
            peak_memory = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            )
            selected_actions = candidates[
                torch.arange(len(replan), device=candidates.device), selected_indices
            ]
            keep_blocks = min(self.receding_horizon_blocks, self.model.action_horizon)
            base_action_dim = self.model.action_dim // self.action_block
            plan = selected_actions[:, :keep_blocks].reshape(
                len(replan), keep_blocks * self.action_block, base_action_dim
            )
            event = {
                **self.metadata(),
                **proposal_meta,
                "environment_batch_size": len(replan),
                "selected_indices": selected_indices.detach().cpu().tolist(),
                "verify_seconds": float(verify_seconds),
                "proposal_seconds": float(proposal_seconds),
                "encode_seconds": float(encode_seconds),
                "costs": None if costs is None else costs.detach().cpu().tolist(),
                "forward_count": int(proposal_meta["forward_count"] + verifier_forward_count),
                "peak_memory_bytes": peak_memory,
            }
            self.planning_events.append(event)
            for row, env_index in enumerate(replan):
                self._action_buffer[env_index].extend(plan[row].detach().cpu())

        base_action_dim = int(np.prod(self.env.single_action_space.shape))
        action = torch.full((self.env.num_envs, base_action_dim), float("nan"))
        for index in range(self.env.num_envs):
            if not dead[index]:
                action[index] = self._action_buffer[index].popleft()
        result = action.reshape(*self.env.action_space.shape).numpy()
        if "action" in self.process:
            result = self.process["action"].inverse_transform(result)
        return result


def make_round4_policy(
    policy_or_model,
    *,
    mode: str,
    solver_cfg=None,
    plan_config,
    process=None,
    transform=None,
    device="cuda",
    seed: int = 42,
    candidate_count: int = 64,
    flow_steps: int = 16,
    action_flow_steps: int | None = None,
    solver_batch_size: int = 1,
    candidate_batch_size: int | None = None,
):
    """Build one of the frozen Round 4 P0--P4 policy variants."""
    if mode not in ROUND4_MODES:
        raise ValueError(f"unknown Round 4 mode {mode!r}; expected {ROUND4_MODES}")
    if action_flow_steps is not None and int(action_flow_steps) < 1:
        raise ValueError("action_flow_steps must be positive")
    if mode in {"P4", "P4-first"} and action_flow_steps is not None:
        raise ValueError("action_flow_steps is only valid for action proposals")
    resolved_action_flow_steps = (
        None if action_flow_steps is None else int(action_flow_steps)
    )
    if mode == "P0":
        return make_fast_lewam_policy(
            policy_or_model,
            solver_cfg=None,
            plan_config=plan_config,
            process=process or {},
            transform=transform or {},
            device=device,
            mode="stage_a",
            inference_steps=(
                16
                if resolved_action_flow_steps is None
                else resolved_action_flow_steps
            ),
            seed=seed,
        )
    if mode == "P0-shuf":
        return make_fast_lewam_policy(
            policy_or_model,
            solver_cfg=None,
            plan_config=plan_config,
            process=process or {},
            transform=transform or {},
            device=device,
            mode="stage_a",
            inference_steps=(
                16
                if resolved_action_flow_steps is None
                else resolved_action_flow_steps
            ),
            seed=seed,
            goal_mode="cyclic_shift",
        )
    if mode in {"P1", "P2"}:
        return make_fast_lewam_policy(
            policy_or_model,
            solver_cfg=solver_cfg,
            plan_config=plan_config,
            process=process or {},
            transform=transform or {},
            device=device,
            mode="stage_b",
            inference_steps=(
                None
                if mode == "P1"
                else (
                    10
                    if resolved_action_flow_steps is None
                    else resolved_action_flow_steps
                )
            ),
            seed=seed,
            actor_warm_start=mode == "P2",
        )
    model = getattr(policy_or_model, "model", policy_or_model)
    model = model.to(device).eval() if device is not None else model.eval()
    plan_values = _to_container(plan_config)

    def plan_value(name):
        if isinstance(plan_values, Mapping):
            return plan_values[name]
        return getattr(plan_values, name)

    proposal_flow_steps = (
        int(flow_steps)
        if resolved_action_flow_steps is None
        else resolved_action_flow_steps
    )
    return Round4BestOfNPolicy(
        model,
        proposal_source="action" if mode == "P3" else "latent",
        num_candidates=candidate_count,
        flow_steps=proposal_flow_steps if mode == "P3" else flow_steps,
        action_flow_steps=proposal_flow_steps if mode == "P3" else None,
        action_block=int(plan_value("action_block")),
        receding_horizon_blocks=int(plan_value("receding_horizon")),
        solver_batch_size=solver_batch_size,
        candidate_batch_size=candidate_batch_size,
        verifier="none" if mode == "P4-first" else "stage_b",
        selection_rule="first" if mode == "P4-first" else "argmin_verifier",
        process=process,
        transform=transform,
        seed=seed,
    )


__all__ = [
    "ROUND4_MODES",
    "Round4BestOfNPolicy",
    "make_round4_policy",
    "score_candidates_in_chunks",
]
