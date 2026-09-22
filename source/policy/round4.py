"""Round 4 planning policies and best-of-N candidate selection."""

from __future__ import annotations

from collections import deque
import time
from collections.abc import Callable, Mapping

import numpy as np
import stable_worldmodel as swm
import torch

from source.common.round4_action_bounds import (
    NormalizedActionBounds,
    compute_normalized_action_bounds,
    normalized_action_stats,
    project_normalized_actions,
)
from source.policy.fast_lewam_eval import (
    _validate_action_bound_mode,
    _validate_action_dim,
    make_fast_lewam_policy,
)
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
        action_flow_integrator: str = "euler",
        verifier: str = "stage_b",
        selection_rule: str = "argmin_verifier",
        process=None,
        transform=None,
        seed: int = 42,
        action_bound_mode: str = "none",
        bf16_proposal: bool = False,
        bf16_verifier: bool = False,
        optimize_proposal: bool = False,
        cache_goal_latent: bool = False,
        bf16_encode: bool = False,
        guidance_mode: str = "none",
        guidance_step_size: float = 0.01,
        guidance_last_steps: int = 5,
        guidance_inner_steps: int = 5,
        guidance_max_rms_offset: float = 0.20,
        proposal_chunk_size: int | None = None,
        diagnostic_callback: Callable[[dict], None] | None = None,
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
        self.guidance_mode = str(guidance_mode).lower()
        if self.guidance_mode not in {
            "none",
            "guided_flow",
            "post_opt",
            "post_opt_refine",
        }:
            raise ValueError(
                "guidance_mode must be 'none', 'guided_flow', 'post_opt', or "
                "'post_opt_refine'"
            )
        if self.guidance_mode != "none" and proposal_source != "action":
            raise ValueError("guidance is only defined for action proposals")
        if self.guidance_mode == "post_opt_refine" and verifier == "none":
            raise ValueError("post_opt_refine requires the Stage-B verifier")
        self.guidance_step_size = float(guidance_step_size)
        self.guidance_last_steps = int(guidance_last_steps)
        self.guidance_inner_steps = int(guidance_inner_steps)
        self.guidance_max_rms_offset = float(guidance_max_rms_offset)
        if self.guidance_mode != "none":
            if self.guidance_step_size <= 0.0:
                raise ValueError("guidance_step_size must be positive")
            if self.guidance_last_steps < 1:
                raise ValueError("guidance_last_steps must be positive")
            if self.guidance_inner_steps < 1:
                raise ValueError("guidance_inner_steps must be positive")
            if self.guidance_max_rms_offset <= 0.0:
                raise ValueError("guidance_max_rms_offset must be positive")
        self.proposal_chunk_size = (
            None if proposal_chunk_size is None else int(proposal_chunk_size)
        )
        if self.proposal_chunk_size is not None and self.proposal_chunk_size < 1:
            raise ValueError("proposal_chunk_size must be positive")
        self.type = "round4_best_of_n"
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.proposal_source = proposal_source
        self.num_candidates = int(num_candidates)
        self.flow_steps = int(flow_steps)
        self.action_flow_steps = int(
            flow_steps if action_flow_steps is None else action_flow_steps
        )
        self.action_flow_integrator = str(action_flow_integrator).lower()
        if self.action_flow_integrator not in {"euler", "heun"}:
            raise ValueError("action_flow_integrator must be 'euler' or 'heun'")
        self.action_block = int(action_block)
        self.receding_horizon_blocks = int(receding_horizon_blocks)
        self.solver_batch_size = int(solver_batch_size)
        self.candidate_batch_size = candidate_batch_size
        self.verifier = verifier
        self.selection_rule = selection_rule
        self.process = process or {}
        self.transform = transform or {}
        self.seed = int(seed)
        self.action_bound_mode = _validate_action_bound_mode(
            action_bound_mode,
            allowed=("none", "clip", "global_scale"),
        )
        self.action_bounds: NormalizedActionBounds | None = None
        self.bf16_proposal = bool(bf16_proposal)
        self.bf16_verifier = bool(bf16_verifier)
        self.optimize_proposal = bool(optimize_proposal)
        self.cache_goal_latent = bool(cache_goal_latent)
        self.bf16_encode = bool(bf16_encode)
        self._goal_latent_cache = None
        self._generators = {}
        self._action_buffer = None
        self.planning_events: list[dict] = []
        self.diagnostic_callback = diagnostic_callback

    def _autocast(self, enabled: bool):
        device = next(self.model.parameters()).device
        return torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=bool(enabled) and device.type == "cuda",
        )

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
        self._goal_latent_cache = (
            [None] * env.num_envs if self.cache_goal_latent else None
        )
        _validate_action_dim(self.model, env, self.action_block)
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
            action_block=self.action_block,
        )

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

    def _encode_context(self, info, indices=None):
        with self._autocast(self.bf16_encode):
            current = self.model._last_frame(info["pixels"])
            cache = self._goal_latent_cache
            if cache is not None and indices is not None and all(
                cache[index] is not None for index in indices
            ):
                current_latent = self.model.encode_pixels(current).float()
                goal_latent = torch.stack(
                    [cache[index] for index in indices], dim=0
                ).float()
                return current_latent, goal_latent
            goal = self.model._last_frame(info["goal"])
            encoded = self.model.encode_pixels(torch.cat((current, goal), dim=0))
        # Cast back to fp32 so Stage A/B always see a stable dtype regardless of
        # whether the encoder ran under bf16 autocast.
        current_latent = encoded[: current.shape[0]].float()
        goal_latent = encoded[current.shape[0] :].float()
        if cache is not None and indices is not None:
            for row, index in enumerate(indices):
                cache[index] = goal_latent[row].detach()
        return current_latent, goal_latent

    def _sample_action_chunk(self, flat_start, flat_goal, noise, generator):
        """Sample one action-proposal chunk, applying guidance when enabled."""
        if self.guidance_mode in {"guided_flow", "post_opt"}:
            return self.model.sample_actions(
                flat_start,
                noise=noise,
                num_steps=self.action_flow_steps,
                goal_latent=flat_goal,
                integrator=self.action_flow_integrator,
                guidance_mode=self.guidance_mode,
                guidance_step_size=self.guidance_step_size,
                guidance_last_steps=self.guidance_last_steps,
                guidance_inner_steps=self.guidance_inner_steps,
                guidance_max_rms_offset=self.guidance_max_rms_offset,
            )
        return self.model.sample_actions(
            flat_start,
            noise=noise,
            num_steps=self.action_flow_steps,
            goal_latent=flat_goal,
            integrator=self.action_flow_integrator,
        )

    def _propose_actions(self, z_start, z_goal, generator):
        """Generate ``[B,S,H,A]`` action candidates with optional chunking.

        The full candidate noise tensor is drawn up front, so execution
        chunking is purely a memory detail: it never changes the candidate set
        or the comparison against the unguided baseline.  Guidance builds an
        autograd graph per candidate, hence the flattened dimension is split
        into ``proposal_chunk_size`` chunks to bound peak memory.
        """
        batch = z_start.shape[0]
        total = batch * self.num_candidates
        flat_start = (
            z_start[:, None]
            .expand(batch, self.num_candidates, -1)
            .reshape(-1, z_start.shape[-1])
        )
        flat_goal = (
            z_goal[:, None]
            .expand(batch, self.num_candidates, -1)
            .reshape(-1, z_goal.shape[-1])
        )
        started = time.perf_counter()
        with self._autocast(self.bf16_proposal):
            if self.optimize_proposal and self.guidance_mode == "none":
                actions = self.model.sample_actions(
                    flat_start,
                    num_steps=self.action_flow_steps,
                    goal_latent=flat_goal,
                    integrator=self.action_flow_integrator,
                    generator=generator,
                )
            else:
                # Draw the whole candidate noise tensor first so execution
                # chunking never changes the candidate set.  Chunking is a
                # memory detail, not part of the condition identity.
                all_noise = torch.randn(
                    total,
                    self.model.action_horizon,
                    self.model.action_dim,
                    device=z_start.device,
                    dtype=z_start.dtype,
                    generator=generator,
                )
                chunk = (
                    total
                    if self.proposal_chunk_size is None
                    else self.proposal_chunk_size
                )
                pieces = []
                for start in range(0, total, chunk):
                    end = min(start + chunk, total)
                    rows = (
                        torch.arange(start, end, device=z_start.device)
                        // self.num_candidates
                    )
                    pieces.append(
                        self._sample_action_chunk(
                            flat_start[start:end],
                            flat_goal[start:end],
                            all_noise[start:end],
                            generator,
                        )
                    )
                actions = torch.cat(pieces, dim=0)
        flow_seconds = time.perf_counter() - started
        guided_steps = (
            min(self.guidance_last_steps, self.action_flow_steps)
            if self.guidance_mode == "guided_flow"
            else 0
        )
        inner = self.guidance_inner_steps if self.guidance_mode != "none" else 0
        per_candidate_backward = (
            int(guided_steps * inner)
            if self.guidance_mode == "guided_flow"
            else (int(inner) if self.guidance_mode == "post_opt" else 0)
        )
        return actions.reshape(
            batch, self.num_candidates, self.model.action_horizon, self.model.action_dim
        ), {
            "flow_steps": self.action_flow_steps,
            "idm_type": "none",
            "forward_count": self.action_flow_steps * (
                2 if self.action_flow_integrator == "heun" else 1
            ),
            "flow_seconds": float(flow_seconds),
            "idm_seconds": 0.0,
            "guidance_mode": self.guidance_mode,
            "guidance_backward_count": int(per_candidate_backward * total),
            "proposal_chunk_size": self.proposal_chunk_size,
        }

    def _propose(self, z_start, z_goal, generator):
        batch = z_start.shape[0]
        if self.proposal_source == "action":
            return self._propose_actions(z_start, z_goal, generator)
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
        with self._autocast(self.bf16_verifier):
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

    def _project_candidates(self, candidates):
        if self.action_bound_mode == "none":
            return candidates, None
        if self.action_bounds is None:
            raise RuntimeError(
                "set_env must configure action bounds before planning"
            )
        projected = project_normalized_actions(
            candidates,
            self.action_bounds,
            mode=self.action_bound_mode,
        )
        before = normalized_action_stats(candidates, self.action_bounds)
        after = normalized_action_stats(projected, self.action_bounds)
        delta = (projected - candidates).detach().float().abs()
        changed_fraction = float((delta > 1e-6).float().mean().cpu())
        return projected, {
            "mode": self.action_bound_mode,
            "bounds": self.action_bounds.metadata(
                action_dim=self.model.action_dim
            ),
            "before": before,
            "after": after,
            "changed_fraction": changed_fraction,
            "mean_abs_delta": float(delta.mean().cpu()),
            "max_abs_delta": float(delta.max().cpu()),
            "raw_candidate_violation_fraction": float(
                before["true_normalized_bound_violation_fraction"]
            ),
            "projected_candidate_violation_fraction": float(
                after["true_normalized_bound_violation_fraction"]
            ),
            "candidate_changed_fraction": changed_fraction,
        }

    def metadata(self) -> dict:
        metadata = {
            "proposal_source": self.proposal_source,
            "candidate_count": self.num_candidates,
            "flow_steps": self.flow_steps,
            "action_flow_steps": self.action_flow_steps,
            "action_flow_integrator": self.action_flow_integrator,
            "idm_type": "direct_shared_dit" if self.proposal_source == "latent" else "none",
            "verifier": self.verifier,
            "selection_rule": self.selection_rule,
            "solver_batch_size": self.solver_batch_size,
            "candidate_batch_size": self.candidate_batch_size,
            "action_bound_mode": self.action_bound_mode,
            "bf16_proposal": self.bf16_proposal,
            "bf16_verifier": self.bf16_verifier,
            "optimize_proposal": self.optimize_proposal,
            "cache_goal_latent": self.cache_goal_latent,
            "bf16_encode": self.bf16_encode,
            "guidance_mode": self.guidance_mode,
            "guidance_step_size": self.guidance_step_size,
            "guidance_last_steps": self.guidance_last_steps,
            "guidance_inner_steps": self.guidance_inner_steps,
            "guidance_max_rms_offset": self.guidance_max_rms_offset,
            "proposal_chunk_size": self.proposal_chunk_size,
        }
        if self.action_bounds is not None:
            metadata["action_bounds"] = self.action_bounds.metadata(
                action_dim=self.model.action_dim
            )
        return metadata

    def get_action(self, info_dict, **kwargs):
        del kwargs
        if self._action_buffer is None:
            raise RuntimeError("set_env must be called before get_action")
        needs_flush = info_dict.get("_needs_flush")
        if needs_flush is not None:
            for index, flush in enumerate(needs_flush):
                if flush:
                    self._action_buffer[index].clear()
                    if self._goal_latent_cache is not None:
                        self._goal_latent_cache[index] = None
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
            z_start, z_goal = self._encode_context(selected, replan)
            _sync(device)
            encode_seconds = time.perf_counter() - started

            started = time.perf_counter()
            with torch.no_grad():
                raw_candidates, proposal_meta = self._propose(
                    z_start, z_goal, generator
                )
                candidates, projection = self._project_candidates(raw_candidates)
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
            refine_seconds = 0.0
            if self.guidance_mode == "post_opt_refine":
                started = time.perf_counter()
                selected_actions = self.model.post_optimize_actions(
                    z_start,
                    z_goal,
                    selected_actions,
                    step_size=self.guidance_step_size,
                    inner_steps=self.guidance_inner_steps,
                    max_rms_offset=self.guidance_max_rms_offset,
                )
                _sync(device)
                refine_seconds = time.perf_counter() - started
                proposal_meta = {
                    **proposal_meta,
                    "guidance_backward_count": int(
                        self.guidance_inner_steps * len(replan)
                    ),
                    "guidance_refine_seconds": float(refine_seconds),
                }
            keep_blocks = min(self.receding_horizon_blocks, self.model.action_horizon)
            base_action_dim = self.model.action_dim // self.action_block
            plan = selected_actions[:, :keep_blocks].reshape(
                len(replan), keep_blocks * self.action_block, base_action_dim
            )
            event = {
                **self.metadata(),
                **proposal_meta,
                "environment_batch_size": len(replan),
                "replan_indices": [int(index) for index in replan],
                "selected_indices": selected_indices.detach().cpu().tolist(),
                "verify_seconds": float(verify_seconds),
                "proposal_seconds": float(proposal_seconds + refine_seconds),
                "encode_seconds": float(encode_seconds),
                "refine_seconds": float(refine_seconds),
                "costs": None if costs is None else costs.detach().cpu().tolist(),
                "forward_count": int(proposal_meta["forward_count"] + verifier_forward_count),
                "peak_memory_bytes": peak_memory,
            }
            if projection is not None:
                event["action_bound_projection"] = projection
            self.planning_events.append(event)
            if self.diagnostic_callback is not None:
                predicted_latents = self.model(
                    z_start[:, None]
                    .expand(-1, candidates.shape[1], -1)
                    .reshape(-1, z_start.shape[-1]),
                    candidates.reshape(-1, candidates.shape[-2], candidates.shape[-1]),
                    torch.ones(
                        len(replan) * candidates.shape[1],
                        device=candidates.device,
                        dtype=candidates.dtype,
                    ),
                    mode="stage_b",
                )["predicted_latents"].reshape(
                    len(replan), candidates.shape[1], candidates.shape[-2], -1
                )
                self.diagnostic_callback(
                    {
                        "z_start": z_start.detach().cpu(),
                        "z_goal": z_goal.detach().cpu(),
                        "candidates": candidates.detach().cpu(),
                        "costs": None if costs is None else costs.detach().cpu(),
                        "predicted_latents": predicted_latents.detach().cpu(),
                        "selected_indices": selected_indices.detach().cpu(),
                        "replan_indices": tuple(int(index) for index in replan),
                        "event": dict(event),
                    }
                )
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
    actor_warm_start_scale: float = 1.0,
    action_flow_integrator: str = "euler",
    action_bound_mode: str = "none",
    bf16_proposal: bool = False,
    bf16_verifier: bool = False,
    optimize_proposal: bool = False,
    cache_goal_latent: bool = False,
    bf16_encode: bool = False,
    guidance_mode: str = "none",
    guidance_step_size: float = 0.01,
    guidance_last_steps: int = 5,
    guidance_inner_steps: int = 5,
    guidance_max_rms_offset: float = 0.20,
    proposal_chunk_size: int | None = None,
    diagnostic_callback: Callable[[dict], None] | None = None,
):
    """Build one of the frozen Round 4 P0--P4 policy variants."""
    if mode not in ROUND4_MODES:
        raise ValueError(f"unknown Round 4 mode {mode!r}; expected {ROUND4_MODES}")
    if action_flow_steps is not None and int(action_flow_steps) < 1:
        raise ValueError("action_flow_steps must be positive")
    if mode in {"P4", "P4-first"} and action_flow_steps is not None:
        raise ValueError("action_flow_steps is only valid for action proposals")
    guidance_mode = str(guidance_mode).lower()
    if guidance_mode not in {"none", "guided_flow", "post_opt", "post_opt_refine"}:
        raise ValueError(
            "guidance_mode must be 'none', 'guided_flow', 'post_opt', or "
            "'post_opt_refine'"
        )
    if guidance_mode == "post_opt_refine" and mode != "P3":
        raise ValueError("post_opt_refine guidance is only defined for P3")
    if guidance_mode != "none" and mode in {"P0-shuf", "P1"}:
        raise ValueError(f"guidance is not defined for {mode}")
    guidance_kwargs = dict(
        guidance_mode=guidance_mode,
        guidance_step_size=guidance_step_size,
        guidance_last_steps=guidance_last_steps,
        guidance_inner_steps=guidance_inner_steps,
        guidance_max_rms_offset=guidance_max_rms_offset,
    )
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
            action_flow_integrator=action_flow_integrator,
            action_bound_mode=action_bound_mode,
            **guidance_kwargs,
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
            action_flow_integrator=action_flow_integrator,
            action_bound_mode=action_bound_mode,
        )
    if mode in {"P1", "P2"}:
        policy = make_fast_lewam_policy(
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
            actor_warm_start_scale=actor_warm_start_scale,
            action_flow_integrator=action_flow_integrator,
            action_bound_mode=action_bound_mode,
            **guidance_kwargs,
        )
        if mode == "P2":
            policy.actor_warm_start_scale = float(actor_warm_start_scale)
        return policy
    model = getattr(policy_or_model, "model", policy_or_model)
    model = model.to(device).eval() if device is not None else model.eval()
    plan_values = _to_container(plan_config)
    if mode in {"P4", "P4-first"} and str(action_bound_mode).lower() != "none":
        raise ValueError(
            "action-bound variants are not defined for latent P4 proposals"
        )

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
        action_flow_integrator=action_flow_integrator,
        verifier="none" if mode == "P4-first" else "stage_b",
        selection_rule="first" if mode == "P4-first" else "argmin_verifier",
        process=process,
        transform=transform,
        seed=seed,
        action_bound_mode=action_bound_mode,
        bf16_proposal=bf16_proposal,
        bf16_verifier=bf16_verifier,
        optimize_proposal=optimize_proposal,
        cache_goal_latent=cache_goal_latent,
        bf16_encode=bf16_encode,
        guidance_mode=guidance_mode,
        guidance_step_size=guidance_step_size,
        guidance_last_steps=guidance_last_steps,
        guidance_inner_steps=guidance_inner_steps,
        guidance_max_rms_offset=guidance_max_rms_offset,
        proposal_chunk_size=proposal_chunk_size,
        diagnostic_callback=diagnostic_callback,
    )


__all__ = [
    "ROUND4_MODES",
    "Round4BestOfNPolicy",
    "make_round4_policy",
    "score_candidates_in_chunks",
]
