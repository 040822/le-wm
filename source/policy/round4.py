"""Round 4 planning policies and best-of-N candidate selection."""

from __future__ import annotations

from collections import deque
import hashlib
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
    latent_noise: torch.Tensor | None = None,
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
            row_noise = (
                None
                if latent_noise is None
                else latent_noise[row_start:row_end]
            )
            row_costs = []
            for col_start in range(0, candidates.shape[1], candidate_batch_size):
                col_end = min(col_start + candidate_batch_size, candidates.shape[1])
                arguments = (
                    z_start[row_start:row_end],
                    z_goal[row_start:row_end],
                    candidates[row_start:row_end, col_start:col_end],
                )
                if row_noise is None:
                    row_costs.append(verifier(*arguments))
                else:
                    row_costs.append(
                        verifier(*arguments, latent_noise=row_noise)
                    )
            rows.append(torch.cat(row_costs, dim=1))
    return torch.cat(rows, dim=0)


class Round4BestOfNPolicy(swm.policy.BasePolicy):
    """Environment-facing P3/P4 policy with a fixed candidate budget."""

    def __init__(
        self,
        model,
        *,
        verifier_model=None,
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
        candidate_noise_schedule: Callable[..., torch.Tensor] | None = None,
        latent_noise_schedule: Callable[..., torch.Tensor] | None = None,
        selection_index_schedule: Callable[..., torch.Tensor] | None = None,
        execute_steps: int | None = None,
        score_horizon_blocks: int | None = None,
        score_reduction: str = "endpoint",
        terminal_weight: float = 1.0,
        timing_mode: bool = False,
    ):
        super().__init__()
        if proposal_source not in {"action", "latent"}:
            raise ValueError("proposal_source must be 'action' or 'latent'")
        if int(num_candidates) < 1 or int(flow_steps) < 1:
            raise ValueError("num_candidates and flow_steps must be positive")
        if verifier not in {"stage_b", "none"}:
            raise ValueError("verifier must be 'stage_b' or 'none'")
        if selection_rule not in {"argmin_verifier", "first", "random"}:
            raise ValueError(
                "selection_rule must be 'argmin_verifier', 'first', or 'random'"
            )
        if selection_rule in {"first", "random"} and verifier != "none":
            raise ValueError(f"{selection_rule} selection must disable the verifier")
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
        self.verifier_model = (self.model if verifier_model is None else verifier_model).eval()
        self.verifier_model.requires_grad_(False)
        self._shared_verifier = self.verifier_model is self.model
        for name in ("latent_dim", "action_dim", "action_horizon"):
            proposal_value = getattr(self.model, name, None)
            verifier_value = getattr(self.verifier_model, name, None)
            if proposal_value != verifier_value:
                raise ValueError(
                    "proposal and verifier models must share "
                    f"{name}: {proposal_value!r} != {verifier_value!r}"
                )
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
        self.execute_steps = None if execute_steps is None else int(execute_steps)
        max_execute_steps = int(self.model.action_horizon) * self.action_block
        if self.execute_steps is not None and not 1 <= self.execute_steps <= max_execute_steps:
            raise ValueError(f"execute_steps must be in [1,{max_execute_steps}]")
        self.score_horizon_blocks = (
            None if score_horizon_blocks is None else int(score_horizon_blocks)
        )
        if self.score_horizon_blocks is not None and not 1 <= self.score_horizon_blocks <= int(self.model.action_horizon):
            raise ValueError(
                "score_horizon_blocks must be in "
                f"[1,{int(self.model.action_horizon)}]"
            )
        self.score_reduction = str(score_reduction).lower()
        if self.score_reduction not in {"endpoint", "minimum", "mixed"}:
            raise ValueError("score_reduction must be endpoint, minimum, or mixed")
        self.terminal_weight = float(terminal_weight)
        if not 0.0 <= self.terminal_weight <= 1.0:
            raise ValueError("terminal_weight must be in [0, 1]")
        if self.score_reduction != "endpoint" and not callable(
            getattr(self.verifier_model, "reduce_goal_distance_costs", None)
        ):
            raise ValueError("trajectory scoring requires a CoWM Stage-B verifier")
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
        self._selection_generators = {}
        self._action_buffer = None
        self.planning_events: list[dict] = []
        self.diagnostic_callback = diagnostic_callback
        self.candidate_noise_schedule = candidate_noise_schedule
        self.latent_noise_schedule = latent_noise_schedule
        self.selection_index_schedule = selection_index_schedule
        self.timing_mode = bool(timing_mode)
        self.capture_timing_selection = False
        self.last_timing_selection = None
        self._candidate_noise_for_diagnostics = None
        self._active_latent_noise = None
        self._active_latent_noise_sha256 = None
        self._phase17_history_groups = None
        self._phase17_history_steps = None
        self._phase17_previous_actions = None
        self._replan_counts = None
        self._active_schedule_keys: tuple[tuple[int, int], ...] = ()

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
        self._selection_generators.clear()

    def _selection_generator(self, device):
        """Draw selection indices without consuming proposal RNG state."""
        key = str(device)
        if key not in self._selection_generators:
            self._selection_generators[key] = torch.Generator(device=device).manual_seed(
                self.seed ^ 0x51E1EC7
            )
        return self._selection_generators[key]

    def set_env(self, env):
        self.env = env
        self._action_buffer = [deque() for _ in range(env.num_envs)]
        self._replan_counts = [0 for _ in range(env.num_envs)]
        if hasattr(self.verifier_model, "history_size"):
            self._phase17_history_groups = [
                deque(maxlen=int(self.verifier_model.history_size))
                for _ in range(env.num_envs)
            ]
            self._phase17_history_steps = [0 for _ in range(env.num_envs)]
            self._phase17_previous_actions = [None for _ in range(env.num_envs)]
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

    def _encode_verifier_context(self, info):
        """Encode the current/goal observations in the verifier's latent space."""
        if self._shared_verifier:
            return self._encode_context(info)
        model = self.verifier_model
        if hasattr(model, "encode_planner_context"):
            return model.encode_planner_context(info)
        with self._autocast(self.bf16_encode):
            current = model._last_frame(info["pixels"])
            goal = model._last_frame(info["goal"])
            encoded = model.encode_pixels(torch.cat((current, goal), dim=0))
        current_latent = encoded[: current.shape[0]].float()
        goal_latent = encoded[current.shape[0] :].float()
        return current_latent, goal_latent

    def _sample_action_chunk(
        self, flat_start, flat_goal, noise, generator, latent_noise=None
    ):
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
                score_horizon_blocks=self.score_horizon_blocks,
                score_reduction=self.score_reduction,
                terminal_weight=self.terminal_weight,
                latent_noise=latent_noise,
            )
        return self.model.sample_actions(
            flat_start,
            noise=noise,
            num_steps=self.action_flow_steps,
            goal_latent=flat_goal,
                integrator=self.action_flow_integrator,
                score_horizon_blocks=self.score_horizon_blocks,
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
        started = None if self.timing_mode else time.perf_counter()
        candidate_noise_sha256 = None
        self._candidate_noise_for_diagnostics = None
        proposal_stage_a_forwards = 0
        proposal_stage_b_forwards = 0
        proposal_backward_calls = 0
        proposal_backward_invocations = 0
        guidance_displacement_sum = 0.0
        guidance_displacement_count = 0
        guidance_displacement_max = 0.0
        expanded_latent_noise = None
        if self._active_latent_noise is not None:
            expanded_latent_noise = self._active_latent_noise[:, None].expand(
                batch,
                self.num_candidates,
                self.model.action_horizon,
                self.model.latent_dim,
            ).reshape(
                total, self.model.action_horizon, self.model.latent_dim
            )

        def record_guidance_counts(candidate_count: int):
            nonlocal proposal_stage_a_forwards
            nonlocal proposal_stage_b_forwards
            nonlocal proposal_backward_calls
            nonlocal proposal_backward_invocations
            nonlocal guidance_displacement_sum
            nonlocal guidance_displacement_count
            nonlocal guidance_displacement_max
            stats = getattr(self.model, "last_guidance_stats", {})
            proposal_stage_a_forwards += int(stats.get("stage_a_forward_count", 0))
            proposal_stage_b_forwards += int(stats.get("stage_b_forward_count", 0))
            backward_count = int(stats.get("backward_count", 0))
            # Report the number of candidate-gradient evaluations. Each batched
            # autograd call updates every candidate in its chunk independently.
            proposal_backward_calls += backward_count * int(candidate_count)
            proposal_backward_invocations += backward_count
            guidance_displacement_sum += float(
                stats.get("guidance_displacement_rms_sum", 0.0)
            )
            guidance_displacement_count += int(
                stats.get("guidance_displacement_count", 0)
            )
            guidance_displacement_max = max(
                guidance_displacement_max,
                float(stats.get("guidance_displacement_rms_max", 0.0)),
            )

        with self._autocast(self.bf16_proposal):
            if self.optimize_proposal and self.guidance_mode == "none":
                actions = self.model.sample_actions(
                    flat_start,
                    num_steps=self.action_flow_steps,
                    goal_latent=flat_goal,
                    integrator=self.action_flow_integrator,
                    generator=generator,
                    collect_guidance_diagnostics=not self.timing_mode,
                )
                record_guidance_counts(total)
            else:
                # Draw the whole candidate noise tensor first so execution
                # chunking never changes the candidate set.  Chunking is a
                # memory detail, not part of the condition identity.
                if self.candidate_noise_schedule is None:
                    all_noise = torch.randn(
                        total,
                        self.model.action_horizon,
                        self.model.action_dim,
                        device=z_start.device,
                        dtype=z_start.dtype,
                        generator=generator,
                    )
                else:
                    scheduled_noise = torch.as_tensor(
                        self.candidate_noise_schedule(
                            self._active_schedule_keys,
                            num_candidates=self.num_candidates,
                            action_horizon=self.model.action_horizon,
                            action_dim=self.model.action_dim,
                            device=z_start.device,
                            dtype=z_start.dtype,
                        ),
                        device=z_start.device,
                        dtype=z_start.dtype,
                    )
                    expected_shape = (
                        batch,
                        self.num_candidates,
                        self.model.action_horizon,
                        self.model.action_dim,
                    )
                    if tuple(scheduled_noise.shape) != expected_shape:
                        raise RuntimeError(
                            "state-indexed candidate noise schedule returned the wrong shape: "
                            f"{tuple(scheduled_noise.shape)} != {expected_shape}"
                        )
                    all_noise = scheduled_noise.reshape(
                        total,
                        self.model.action_horizon,
                        self.model.action_dim,
                    )
                if self.diagnostic_callback is not None:
                    # Retain the exact draw only for the synchronous diagnostic
                    # callback.  It is deliberately kept out of planning_events
                    # and the normal evaluation artifact, which contain compact
                    # scalar provenance only.
                    self._candidate_noise_for_diagnostics = all_noise.detach().cpu()
                    candidate_noise_sha256 = hashlib.sha256(
                        all_noise.detach().cpu().numpy().tobytes()
                    ).hexdigest()
                chunk = (
                    total
                    if self.proposal_chunk_size is None
                    else self.proposal_chunk_size
                )
                pieces = []
                for start in range(0, total, chunk):
                    end = min(start + chunk, total)
                    pieces.append(
                        self._sample_action_chunk(
                            flat_start[start:end],
                            flat_goal[start:end],
                            all_noise[start:end],
                            generator,
                            None
                            if expanded_latent_noise is None
                            else expanded_latent_noise[start:end],
                        )
                    )
                    record_guidance_counts(end - start)
                actions = torch.cat(pieces, dim=0)
        flow_seconds = 0.0 if started is None else time.perf_counter() - started
        return actions.reshape(
            batch, self.num_candidates, self.model.action_horizon, self.model.action_dim
        ), {
            "flow_steps": self.action_flow_steps,
            "idm_type": "none",
            "stage_a_forward_count": int(proposal_stage_a_forwards),
            "stage_b_forward_count": int(proposal_stage_b_forwards),
            "forward_count": int(
                proposal_stage_a_forwards + proposal_stage_b_forwards
            ),
            "flow_seconds": float(flow_seconds),
            "idm_seconds": 0.0,
            "guidance_mode": self.guidance_mode,
            "guidance_backward_count": int(proposal_backward_calls),
            "guidance_backward_invocation_count": int(
                proposal_backward_invocations
            ),
            "guidance_displacement_rms_mean": (
                float(guidance_displacement_sum / guidance_displacement_count)
                if guidance_displacement_count
                else 0.0
            ),
            "guidance_displacement_rms_max": float(guidance_displacement_max),
            "proposal_chunk_size": self.proposal_chunk_size,
            "candidate_noise_sha256": candidate_noise_sha256,
        }

    def _propose(self, z_start, z_goal, generator):
        batch = z_start.shape[0]
        if self.proposal_source == "action":
            return self._propose_actions(z_start, z_goal, generator)
        started = None if self.timing_mode else time.perf_counter()
        paths = self.model.sample_latent_paths(
            z_start,
            z_goal,
            num_samples=self.num_candidates,
            num_steps=self.flow_steps,
            generator=generator,
        )
        flow_seconds = 0.0 if started is None else time.perf_counter() - started
        started = None if self.timing_mode else time.perf_counter()
        actions = self.model.decode_latent_paths(paths)
        idm_seconds = 0.0 if started is None else time.perf_counter() - started
        return actions, {
            "flow_steps": self.flow_steps,
            "idm_type": "direct_shared_dit",
            "forward_count": self.flow_steps + 1,
            "flow_seconds": float(flow_seconds),
            "idm_seconds": float(idm_seconds),
        }

    def score_candidates(
        self, z_start, z_goal, candidates, context=None, latent_noise=None
    ):
        if hasattr(self.verifier_model, "score_action_candidates"):
            return self.verifier_model.score_action_candidates(
                z_start,
                z_goal,
                candidates,
                context=context,
                candidate_batch_size=self.candidate_batch_size or self.num_candidates,
                solver_batch_size=self.solver_batch_size,
            )
        with self._autocast(self.bf16_verifier):
            is_fm = bool(
                getattr(self.verifier_model, "latent_flow_matching", False)
            )
            return score_candidates_in_chunks(
                lambda start, goal, actions, latent_noise=None: self.verifier_model.get_cost_from_latents(
                    start,
                    goal,
                    actions,
                    score_horizon_blocks=self.score_horizon_blocks,
                    latent_noise=latent_noise if is_fm else None,
                    score_reduction=self.score_reduction,
                    terminal_weight=self.terminal_weight,
                ),
                z_start,
                z_goal,
                candidates,
                solver_batch_size=self.solver_batch_size,
                candidate_batch_size=self.candidate_batch_size,
                latent_noise=latent_noise if is_fm else None,
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
        if self.timing_mode:
            return projected, None
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
            "seed": int(self.seed),
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
            "separate_verifier_model": not self._shared_verifier,
            "candidate_noise_schedule": (
                "state_indexed" if self.candidate_noise_schedule is not None else None
            ),
            "latent_noise_schedule": (
                "state_indexed" if self.latent_noise_schedule is not None else None
            ),
            "selection_index_schedule": (
                "state_indexed" if self.selection_index_schedule is not None else None
            ),
            "execute_steps": self.execute_steps,
            "score_horizon_blocks": self.score_horizon_blocks,
            "score_reduction": self.score_reduction,
            "terminal_weight": self.terminal_weight,
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
                    if self._replan_counts is not None:
                        self._replan_counts[index] = 0
                    if self._goal_latent_cache is not None:
                        self._goal_latent_cache[index] = None
                    if self._phase17_history_groups is not None:
                        self._phase17_history_groups[index].clear()
                        self._phase17_history_steps[index] = 0
                        self._phase17_previous_actions[index] = None
        if self._phase17_history_groups is not None:
            self._observe_phase17_history(info_dict)
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
            self._active_schedule_keys = tuple(
                (int(index), int(self._replan_counts[index]))
                for index in replan
            )
            self._active_latent_noise = None
            self._active_latent_noise_sha256 = None
            latent_model = (
                self.verifier_model
                if self.verifier == "stage_b"
                else self.model
            )
            if (
                bool(getattr(latent_model, "latent_flow_matching", False))
                and self.latent_noise_schedule is not None
            ):
                self._active_latent_noise = self.latent_noise_schedule(
                    self._active_schedule_keys,
                    horizon=int(latent_model.action_horizon),
                    latent_dim=int(latent_model.latent_dim),
                    device=next(latent_model.parameters()).device,
                    dtype=torch.float32,
                )
                expected_noise_shape = (
                    len(replan),
                    int(self.model.action_horizon),
                    int(self.model.latent_dim),
                )
                if tuple(self._active_latent_noise.shape) != expected_noise_shape:
                    raise ValueError(
                        "latent noise schedule must return "
                        f"{expected_noise_shape}, got "
                        f"{tuple(self._active_latent_noise.shape)}"
                    )
                self._active_latent_noise_sha256 = hashlib.sha256(
                    self._active_latent_noise.detach()
                    .cpu()
                    .contiguous()
                    .numpy()
                    .tobytes()
                ).hexdigest()
            selected = self._prepare(self._slice_info(info_dict, replan))
            device = next(self.model.parameters()).device
            if self._phase17_history_groups is not None:
                history_pixels, history_actions = self._phase17_history_context(replan)
                selected["_phase17_history_pixels"] = history_pixels.to(device)
                selected["_phase17_history_actions"] = history_actions.to(device)
            if device.type == "cuda" and not self.timing_mode:
                torch.cuda.reset_peak_memory_stats(device)
            generator = self._generator(device)
            if not self.timing_mode:
                _sync(device)
            started = None if self.timing_mode else time.perf_counter()
            z_start, z_goal = self._encode_context(selected, replan)
            if started is not None:
                _sync(device)
            proposal_encode_seconds = (
                0.0 if started is None else time.perf_counter() - started
            )
            if self.verifier == "stage_b" and not self._shared_verifier:
                verifier_started = None if self.timing_mode else time.perf_counter()
                verifier_z_start, verifier_z_goal = self._encode_verifier_context(
                    selected
                )
                if verifier_started is not None:
                    _sync(device)
                verifier_encode_seconds = (
                    0.0
                    if verifier_started is None
                    else time.perf_counter() - verifier_started
                )
            else:
                verifier_z_start, verifier_z_goal = z_start, z_goal
                verifier_encode_seconds = 0.0
            encode_seconds = proposal_encode_seconds + verifier_encode_seconds
            if not self.timing_mode:
                _sync(device)

            started = None if self.timing_mode else time.perf_counter()
            with torch.no_grad():
                raw_candidates, proposal_meta = self._propose(
                    z_start, z_goal, generator
                )
                candidates, projection = self._project_candidates(raw_candidates)
            if started is not None:
                _sync(device)
            proposal_seconds = 0.0 if started is None else time.perf_counter() - started

            costs = None
            if self.verifier == "stage_b":
                started = None if self.timing_mode else time.perf_counter()
                if bool(
                    getattr(self.verifier_model, "latent_flow_matching", False)
                ):
                    self.verifier_model.last_forward_count = 0
                with torch.no_grad():
                    costs = self.score_candidates(
                        verifier_z_start,
                        verifier_z_goal,
                        candidates,
                        context=selected,
                        latent_noise=self._active_latent_noise,
                    )
                if started is not None:
                    _sync(device)
                verify_seconds = 0.0 if started is None else time.perf_counter() - started
                selected_indices = costs.argmin(dim=1)
                row_batches = (len(replan) + self.solver_batch_size - 1) // self.solver_batch_size
                candidate_batches = (
                    self.num_candidates + (self.candidate_batch_size or self.num_candidates) - 1
                ) // (self.candidate_batch_size or self.num_candidates)
                verifier_forward_count = int(
                    getattr(
                        self.verifier_model,
                        "last_forward_count",
                        row_batches * candidate_batches,
                    )
                )
            elif self.selection_rule == "first":
                verify_seconds = 0.0
                selected_indices = torch.zeros(
                    len(replan), dtype=torch.long, device=candidates.device
                )
                verifier_forward_count = 0
            else:
                verify_seconds = 0.0
                if self.selection_index_schedule is None:
                    selected_indices = torch.randint(
                        self.num_candidates,
                        (len(replan),),
                        device=candidates.device,
                        generator=self._selection_generator(candidates.device),
                    )
                else:
                    selected_indices = torch.as_tensor(
                        self.selection_index_schedule(
                            self._active_schedule_keys,
                            num_candidates=self.num_candidates,
                            device=candidates.device,
                        ),
                        dtype=torch.long,
                        device=candidates.device,
                    )
                    if selected_indices.shape != (len(replan),):
                        raise RuntimeError(
                            "state-indexed selection schedule returned the wrong shape: "
                            f"{tuple(selected_indices.shape)} != {(len(replan),)}"
                        )
                    if bool(
                        ((selected_indices < 0) | (selected_indices >= self.num_candidates))
                        .any()
                        .item()
                    ):
                        raise RuntimeError(
                            "state-indexed selection schedule returned an out-of-range index"
                        )
                verifier_forward_count = 0
            peak_memory = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" and not self.timing_mode
                else None
            )
            selected_actions = candidates[
                torch.arange(len(replan), device=candidates.device), selected_indices
            ]
            if self.capture_timing_selection:
                self.last_timing_selection = selected_indices.detach().clone()
            selected_actions_before_refine = selected_actions
            refine_seconds = 0.0
            post_refine_projection = None
            if self.guidance_mode == "post_opt_refine":
                started = None if self.timing_mode else time.perf_counter()
                selected_actions = self.model.post_optimize_actions(
                    z_start,
                    z_goal,
                    selected_actions,
                    step_size=self.guidance_step_size,
                    inner_steps=self.guidance_inner_steps,
                    max_rms_offset=self.guidance_max_rms_offset,
                    score_horizon_blocks=self.score_horizon_blocks,
                    score_reduction=self.score_reduction,
                    terminal_weight=self.terminal_weight,
                    collect_guidance_diagnostics=not self.timing_mode,
                    latent_noise=self._active_latent_noise,
                )
                if started is not None:
                    _sync(device)
                refine_seconds = 0.0 if started is None else time.perf_counter() - started
                refine_stats = getattr(self.model, "last_guidance_stats", {})
                refine_stage_b_forwards = int(
                    refine_stats.get("stage_b_forward_count", self.guidance_inner_steps)
                )
                refine_backward_calls = int(
                    refine_stats.get("backward_count", self.guidance_inner_steps)
                )
                refine_backward_invocations = refine_backward_calls
                proposal_meta = {
                    **proposal_meta,
                    "stage_b_forward_count": int(
                        proposal_meta.get("stage_b_forward_count", 0)
                        + refine_stage_b_forwards
                    ),
                    "forward_count": int(
                        proposal_meta.get("forward_count", 0)
                        + refine_stage_b_forwards
                    ),
                    "guidance_backward_count": int(
                        proposal_meta.get("guidance_backward_count", 0)
                        + refine_backward_calls
                    ),
                    "guidance_backward_invocation_count": int(
                        proposal_meta.get(
                            "guidance_backward_invocation_count", 0
                        )
                        + refine_backward_invocations
                    ),
                    "guidance_refine_seconds": float(refine_seconds),
                }
                for key in (
                    "guidance_displacement_rms_sum",
                    "guidance_displacement_count",
                ):
                    proposal_meta[key] = (
                        float(proposal_meta.get(key, 0.0))
                        + float(refine_stats.get(key, 0.0))
                    )
                proposal_meta["guidance_displacement_rms_max"] = max(
                    float(proposal_meta.get("guidance_displacement_rms_max", 0.0)),
                    float(refine_stats.get("guidance_displacement_rms_max", 0.0)),
                )
                if self.action_bound_mode != "none":
                    if self.action_bounds is None:
                        raise RuntimeError(
                            "set_env must configure action bounds before planning"
                        )
                    pre_projection = selected_actions
                    selected_actions = project_normalized_actions(
                        pre_projection,
                        self.action_bounds,
                        mode=self.action_bound_mode,
                    )
                    if not self.timing_mode:
                        before_stats = normalized_action_stats(
                            pre_projection, self.action_bounds
                        )
                        after_stats = normalized_action_stats(
                            selected_actions, self.action_bounds
                        )
                        delta = (selected_actions - pre_projection).detach().float().abs()
                        post_refine_projection = {
                            "mode": self.action_bound_mode,
                            "bounds": self.action_bounds.metadata(
                                action_dim=self.model.action_dim
                            ),
                            "before": before_stats,
                            "after": after_stats,
                            "changed_fraction": float(
                                (delta > 1e-6).float().mean().cpu()
                            ),
                            "mean_abs_delta": float(delta.mean().cpu()),
                            "max_abs_delta": float(delta.max().cpu()),
                        }
            keep_blocks = min(self.receding_horizon_blocks, self.model.action_horizon)
            base_action_dim = self.model.action_dim // self.action_block
            plan = selected_actions[:, :keep_blocks].reshape(
                len(replan), keep_blocks * self.action_block, base_action_dim
            )
            if self.execute_steps is not None:
                plan = plan[:, : self.execute_steps]
            if not self.timing_mode:
                event = {
                    **self.metadata(),
                    **proposal_meta,
                    "environment_batch_size": len(replan),
                    "replan_indices": [int(index) for index in replan],
                    "selected_indices": selected_indices.detach().cpu().tolist(),
                    "state_indexed_schedule_keys": [
                        [slot, replan_index]
                        for slot, replan_index in self._active_schedule_keys
                    ],
                    "verify_seconds": float(verify_seconds),
                    "proposal_seconds": float(proposal_seconds + refine_seconds),
                    "encode_seconds": float(encode_seconds),
                    "refine_seconds": float(refine_seconds),
                    "costs": None if costs is None else costs.detach().cpu().tolist(),
                    "proposal_encode_seconds": float(proposal_encode_seconds),
                    "verifier_encode_seconds": float(verifier_encode_seconds),
                    "proposal_forward_count": int(proposal_meta["forward_count"]),
                    "stage_a_forward_count": int(
                        proposal_meta.get("stage_a_forward_count", 0)
                    ),
                    "stage_b_forward_count": int(
                        proposal_meta.get("stage_b_forward_count", 0)
                    ),
                    "verifier_forward_count": int(verifier_forward_count),
                    "guidance_backward_invocation_count": int(
                        proposal_meta.get("guidance_backward_invocation_count", 0)
                    ),
                    "forward_count": int(
                        proposal_meta["forward_count"] + verifier_forward_count
                    ),
                    "peak_memory_bytes": peak_memory,
                    "selected_action_sequences_before_refine": (
                        selected_actions_before_refine.detach().cpu().tolist()
                    ),
                    "selected_action_sequences": selected_actions.detach().cpu().tolist(),
                    "latent_flow_steps": int(
                        getattr(self.model, "latent_flow_steps", 0)
                    ),
                    "latent_noise_sha256": self._active_latent_noise_sha256,
                }
                if projection is not None:
                    event["action_bound_projection"] = projection
                if post_refine_projection is not None:
                    event["post_refine_action_bound_projection"] = post_refine_projection
                self.planning_events.append(event)
            if self.diagnostic_callback is not None and not self.timing_mode:
                predicted_latents = None
                # External scorers such as LeWM have their own context shape
                # and candidate-scoring interface. Their ranking costs are
                # already in the event; calling the Fast-LeWAM Stage-B
                # prediction API here would pass [B,H,D] history latents to
                # an API that expects [B,D] and corrupt diagnostic capture.
                if not hasattr(self.verifier_model, "score_action_candidates"):
                    predicted_latents = self.verifier_model(
                        verifier_z_start[:, None]
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
                        "verifier_z_start": verifier_z_start.detach().cpu(),
                        "verifier_z_goal": verifier_z_goal.detach().cpu(),
                        "z_goal": z_goal.detach().cpu(),
                        # Diagnostics that compare several scorers on one
                        # captured action pool need the exact first-replan
                        # history used by LeWM. The callback is synchronous
                        # and may stop evaluation immediately after capture.
                        "context": selected,
                        "candidates": candidates.detach().cpu(),
                        "candidate_noise": self._candidate_noise_for_diagnostics,
                        "costs": None if costs is None else costs.detach().cpu(),
                        "predicted_latents": (
                            None
                            if predicted_latents is None
                            else predicted_latents.detach().cpu()
                        ),
                        "selected_indices": selected_indices.detach().cpu(),
                        "replan_indices": tuple(int(index) for index in replan),
                        "event": dict(event),
                    }
                )
                self._candidate_noise_for_diagnostics = None
            for row, env_index in enumerate(replan):
                self._action_buffer[env_index].extend(plan[row].detach().cpu())
                if self._replan_counts is not None:
                    self._replan_counts[env_index] += 1

        base_action_dim = int(np.prod(self.env.single_action_space.shape))
        action = torch.full((self.env.num_envs, base_action_dim), float("nan"))
        for index in range(self.env.num_envs):
            if not dead[index]:
                action[index] = self._action_buffer[index].popleft()
        result = action.reshape(*self.env.action_space.shape).numpy()
        if "action" in self.process:
            result = self.process["action"].inverse_transform(result)
        if self._phase17_history_groups is not None:
            normalized = self.process["action"].transform(
                result.reshape(self.env.num_envs, -1)
            )
            self._phase17_previous_actions = [
                row.astype(np.float32, copy=True)
                if np.isfinite(row).all()
                else None
                for row in normalized
            ]
        return result

    def _observe_phase17_history(self, info_dict):
        """Retain real observations and executed action blocks for a LeWM verifier."""
        if "pixels" not in info_dict:
            raise ValueError("LeWM verification requires pixel observations")
        pixels = info_dict["pixels"]
        block = int(self.action_block)
        for index, groups in enumerate(self._phase17_history_groups):
            previous_action = self._phase17_previous_actions[index]
            if previous_action is not None and groups:
                groups[-1]["actions"].append(previous_action)
            step = self._phase17_history_steps[index]
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
            self._phase17_history_steps[index] += 1

    def _phase17_history_context(self, indices):
        """Build the history-size pixel/action prefix expected by standard LeWM."""
        import torch
        from torchvision import tv_tensors

        history_size = int(self.verifier_model.history_size)
        action_width = int(self.model.action_dim)
        block = int(self.action_block)
        pixel_rows = []
        action_rows = []
        for index in indices:
            groups = list(self._phase17_history_groups[index])[-history_size:]
            if not groups:
                raise RuntimeError("LeWM history buffer has no current observation")
            padding = history_size - len(groups)
            first = groups[0]["pixels"]
            padded_pixels = [first.copy() for _ in range(padding)]
            padded_pixels.extend(group["pixels"] for group in groups)
            chunks = [np.zeros(action_width, dtype=np.float32) for _ in range(padding)]
            for group in groups[:-1]:
                values = group["actions"]
                if len(values) != block:
                    raise RuntimeError(
                        "LeWM history action block is incomplete: "
                        f"expected {block}, found {len(values)}"
                    )
                chunks.append(np.concatenate(values, axis=0).astype(np.float32))
            if len(chunks) != history_size - 1:
                raise RuntimeError("LeWM history action prefix has the wrong length")
            action_rows.append(np.stack(chunks))
            transformed = []
            for image in padded_pixels:
                tensor = torch.from_numpy(np.ascontiguousarray(image))
                if tensor.ndim != 3:
                    raise ValueError("LeWM history image must be rank 3")
                if tensor.shape[-1] in (1, 3, 4):
                    tensor = tensor.permute(2, 0, 1)
                transformed.append(self.transform["pixels"](tv_tensors.Image(tensor)))
            pixel_rows.append(torch.stack(transformed))
        return torch.stack(pixel_rows), torch.from_numpy(np.stack(action_rows))


def make_round4_policy(
    policy_or_model,
    *,
    verifier_policy_or_model=None,
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
    candidate_noise_schedule: Callable[..., torch.Tensor] | None = None,
    latent_noise_schedule: Callable[..., torch.Tensor] | None = None,
    selection_index_schedule: Callable[..., torch.Tensor] | None = None,
    selection_rule: str | None = None,
    execute_steps: int | None = None,
    score_horizon_blocks: int | None = None,
    score_reduction: str = "endpoint",
    terminal_weight: float = 1.0,
):
    """Build one of the frozen Round 4 P0--P4 policy variants."""
    if mode not in ROUND4_MODES:
        raise ValueError(f"unknown Round 4 mode {mode!r}; expected {ROUND4_MODES}")
    if action_flow_steps is not None and int(action_flow_steps) < 1:
        raise ValueError("action_flow_steps must be positive")
    if mode in {"P4", "P4-first"} and action_flow_steps is not None:
        raise ValueError("action_flow_steps is only valid for action proposals")
    guidance_mode = str(guidance_mode).lower()
    if score_reduction != "endpoint" and mode != "P3":
        raise ValueError("trajectory score reductions are currently defined for P3")
    if guidance_mode not in {"none", "guided_flow", "post_opt", "post_opt_refine"}:
        raise ValueError(
            "guidance_mode must be 'none', 'guided_flow', 'post_opt', or "
            "'post_opt_refine'"
        )
    if guidance_mode == "post_opt_refine" and mode != "P3":
        raise ValueError("post_opt_refine guidance is only defined for P3")
    if verifier_policy_or_model is not None and mode not in {"P2", "P3"}:
        raise ValueError("a separate verifier model is only supported for P2 or P3")
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
            diagnostic_callback=diagnostic_callback,
            execute_steps=execute_steps,
            score_horizon_blocks=score_horizon_blocks,
            latent_noise_schedule=latent_noise_schedule,
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
            execute_steps=execute_steps,
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
            score_horizon_blocks=score_horizon_blocks,
            execute_steps=execute_steps,
            latent_noise_schedule=latent_noise_schedule,
            **guidance_kwargs,
        )
        if mode == "P2":
            policy.actor_warm_start_scale = float(actor_warm_start_scale)
        if verifier_policy_or_model is not None:
            if mode != "P2" or not hasattr(
                verifier_policy_or_model, "score_action_candidates"
            ):
                raise ValueError(
                    "a separate CEM verifier requires the Phase1.7 P2 LeWM scorer"
                )
            from source.policy.round5_phase1_7 import attach_lewm_cem_verifier

            policy = attach_lewm_cem_verifier(
                policy,
                verifier_policy_or_model,
                transform=transform,
                candidate_batch_size=candidate_batch_size or candidate_count,
                solver_batch_size=solver_batch_size,
                device=device,
            )
        return policy
    model = getattr(policy_or_model, "model", policy_or_model)
    model = model.to(device).eval() if device is not None else model.eval()
    verifier_model = (
        None
        if verifier_policy_or_model is None
        else getattr(verifier_policy_or_model, "model", verifier_policy_or_model)
    )
    if verifier_model is not None and device is not None:
        verifier_model = verifier_model.to(device).eval()
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
        verifier_model=verifier_model,
        proposal_source="action" if mode == "P3" else "latent",
        num_candidates=candidate_count,
        flow_steps=proposal_flow_steps if mode == "P3" else flow_steps,
        action_flow_steps=proposal_flow_steps if mode == "P3" else None,
        action_block=int(plan_value("action_block")),
        receding_horizon_blocks=int(plan_value("receding_horizon")),
        execute_steps=execute_steps,
        score_horizon_blocks=score_horizon_blocks,
        score_reduction=score_reduction,
        terminal_weight=terminal_weight,
        solver_batch_size=solver_batch_size,
        candidate_batch_size=candidate_batch_size,
        action_flow_integrator=action_flow_integrator,
        verifier=(
            "none"
            if mode == "P4-first" or selection_rule in {"first", "random"}
            else "stage_b"
        ),
        selection_rule=(
            "first"
            if mode == "P4-first" and selection_rule is None
            else (selection_rule or "argmin_verifier")
        ),
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
        candidate_noise_schedule=candidate_noise_schedule,
        latent_noise_schedule=latent_noise_schedule,
        selection_index_schedule=selection_index_schedule,
    )


__all__ = [
    "ROUND4_MODES",
    "Round4BestOfNPolicy",
    "make_round4_policy",
    "score_candidates_in_chunks",
]
