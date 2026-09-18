"""Round 3 T0--T6 supervision and deterministic batch planning.

The module keeps experiment policy at one seam: callers provide a
``TransitionReplay`` and an arm name, and receive one reproducible update
batch plus a single combined loss update.  Environment collection and final
evaluation remain outside this CPU-testable training module.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from source.experiments.round3_phase2 import (
    TransitionReplay,
    freeze_fast_encoder_projector,
    set_fast_adaptation_mode,
)


TRAINING_ARMS = (
    "t0_frozen",
    "t1_offline_b_mse",
    "t2_online_b_mse",
    "t3_online_ranking",
    "t4_online_offline_a",
    "t5_online_hindsight_a",
    "t6_online_distill_a",
)


def _as_indices(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


@dataclass(frozen=True)
class SupervisionBatch:
    """The complete row/pair manifest consumed by one optimizer update."""

    b_indices: tuple[int, ...]
    offline_anchor_indices: tuple[int, ...] = ()
    offline_replacement_indices: tuple[int, ...] = ()
    offline_a_indices: tuple[int, ...] = ()
    ranking_pairs: tuple[tuple[int, int], ...] = ()
    hindsight_indices: tuple[int, ...] = ()
    distill_indices: tuple[int, ...] = ()
    counts: Mapping[str, int] = None

    def __post_init__(self) -> None:
        if not self.b_indices:
            raise ValueError("a supervision batch must contain B-MSE rows")
        for name in (
            "b_indices",
            "offline_anchor_indices",
            "offline_replacement_indices",
            "offline_a_indices",
            "hindsight_indices",
            "distill_indices",
        ):
            values = getattr(self, name)
            if any(int(value) < 0 for value in values):
                raise ValueError(f"{name} contains a negative row index")
        for pair in self.ranking_pairs:
            if len(pair) != 2 or any(int(value) < 0 for value in pair):
                raise ValueError("ranking pairs must contain two non-negative indices")
        if self.counts is None:
            object.__setattr__(
                self,
                "counts",
                {
                    "b_mse": len(self.b_indices),
                    "offline_anchor": len(self.offline_anchor_indices),
                    "offline_replacement": len(self.offline_replacement_indices),
                    "offline_a": len(self.offline_a_indices),
                    "ranking_pairs": len(self.ranking_pairs),
                    "hindsight": len(self.hindsight_indices),
                    "distill": len(self.distill_indices),
                },
            )
        else:
            object.__setattr__(
                self,
                "counts",
                {str(key): int(value) for key, value in self.counts.items()},
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SupervisionBatchSampler:
    """Sample fixed, source-balanced T1--T6 update manifests."""

    def __init__(self, replay: TransitionReplay, *, seed: int = 3072):
        self.replay = replay
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._kinds = replay.resolved_data_kinds
        self._groups = replay.resolved_group_ids

    def _pool(self, kind: str) -> list[int]:
        return [index for index, value in enumerate(self._kinds) if value == kind]

    def _draw(self, pool: Iterable[int], count: int) -> tuple[int, ...]:
        values = np.asarray(tuple(int(value) for value in pool), dtype=np.int64)
        if int(count) < 0:
            raise ValueError("sample count must be non-negative")
        if int(count) == 0:
            return ()
        if values.size == 0:
            return ()
        selected = self._rng.choice(
            values,
            size=int(count),
            replace=values.size < int(count),
        )
        return _as_indices(np.asarray(selected).reshape(-1).tolist())

    def _grounded_pairs(self, limit: int) -> tuple[tuple[int, int], ...]:
        if self.replay.successes is None or self.replay.physical_terminal_costs is None:
            return ()
        lengths = self.replay.execution_lengths
        grouped: dict[Any, list[int]] = defaultdict(list)
        for index, kind in enumerate(self._kinds):
            if kind != "grounded":
                continue
            grouped[self._groups[index]].append(index)
        by_group: dict[Any, list[tuple[int, int]]] = {}
        successes = self.replay.successes.detach().cpu().reshape(-1)
        costs = self.replay.physical_terminal_costs.detach().cpu().reshape(-1)
        for group, indices in grouped.items():
            pairs: list[tuple[int, int]] = []
            for left, right in combinations(indices, 2):
                if lengths is not None and lengths[left].item() != lengths[right].item():
                    continue
                if (
                    self.replay.snapshot_ids is not None
                    and self.replay.snapshot_ids[left] != self.replay.snapshot_ids[right]
                ):
                    continue
                if self.replay.goal_latents is not None and not torch.equal(
                    self.replay.goal_latents[left], self.replay.goal_latents[right]
                ):
                    continue
                if self.replay.goal_latents is None and self.replay.goals is not None:
                    if not torch.equal(self.replay.goals[left], self.replay.goals[right]):
                        continue
                left_success = bool(successes[left].item())
                right_success = bool(successes[right].item())
                if left_success == right_success:
                    left_cost = float(costs[left].item())
                    right_cost = float(costs[right].item())
                    if left_cost == right_cost:
                        continue
                    preferred, rejected = (
                        (left, right) if left_cost < right_cost else (right, left)
                    )
                else:
                    preferred, rejected = (
                        (left, right) if left_success else (right, left)
                    )
                pairs.append((int(preferred), int(rejected)))
            if pairs:
                order = self._rng.permutation(len(pairs))
                by_group[group] = [pairs[int(index)] for index in order]
        if not by_group or int(limit) == 0:
            return ()
        group_keys = list(by_group)
        group_order = self._rng.permutation(len(group_keys)).tolist()
        cursors = {group: 0 for group in group_keys}
        selected: list[tuple[int, int]] = []
        while len(selected) < int(limit):
            progressed = False
            for position in group_order:
                group = group_keys[int(position)]
                cursor = cursors[group]
                if cursor >= len(by_group[group]):
                    continue
                selected.append(by_group[group][cursor])
                cursors[group] += 1
                progressed = True
                if len(selected) >= int(limit):
                    break
            if not progressed:
                break
        return tuple(selected)

    def _eligible_distill(self) -> tuple[int, ...]:
        if (
            self.replay.successes is None
            or self.replay.physical_start_costs is None
            or self.replay.physical_terminal_costs is None
        ):
            return ()
        eligible: list[int] = []
        successes = self.replay.successes.detach().cpu().reshape(-1)
        starts = self.replay.physical_start_costs.detach().cpu().reshape(-1)
        terminals = self.replay.physical_terminal_costs.detach().cpu().reshape(-1)
        for index, kind in enumerate(self._kinds):
            if kind in {"continuous", "grounded"} and (
                bool(successes[index].item())
                or float(terminals[index].item()) < float(starts[index].item())
            ):
                eligible.append(index)
        return tuple(eligible)

    def sample(
        self,
        arm: str,
        *,
        batch_size: int = 64,
        auxiliary_batch_size: int = 32,
        require_grounded: bool = False,
    ) -> SupervisionBatch:
        """Return one batch manifest; shortages use replacement, never fabricate rows."""
        arm = str(arm)
        if arm not in TRAINING_ARMS:
            raise ValueError(f"unknown Round3 training arm: {arm}")
        if int(batch_size) < 2 or int(batch_size) % 2:
            raise ValueError("batch_size must be an even positive number")
        if int(auxiliary_batch_size) < 1:
            raise ValueError("auxiliary_batch_size must be positive")
        offline = self._pool("offline")
        if arm in {"t0_frozen", "t1_offline_b_mse"}:
            anchor = self._draw(offline, int(batch_size) // 2)
            replacement = self._draw(offline, int(batch_size) // 2)
            b_indices = anchor + replacement
            kwargs: dict[str, Any] = {
                "b_indices": b_indices,
                "offline_anchor_indices": anchor,
                "offline_replacement_indices": replacement,
                "counts": {
                    "b_mse": len(b_indices),
                    "b_offline": len(b_indices),
                    "b_continuous": 0,
                    "b_grounded": 0,
                    "offline_anchor": len(anchor),
                    "offline_replacement": len(replacement),
                },
            }
        else:
            continuous = self._pool("continuous")
            grounded = self._pool("grounded")
            if require_grounded and not grounded:
                raise RuntimeError("grounded replay is required but empty")
            online_continuous = self._draw(continuous, int(batch_size) // 2 - 8)
            online_grounded = self._draw(grounded, 8)
            if len(online_grounded) < 8:
                online_continuous += self._draw(
                    continuous,
                    8 - len(online_grounded),
                )
            b_indices = self._draw(offline, int(batch_size) // 2) + online_continuous + online_grounded
            if len(b_indices) != int(batch_size):
                raise RuntimeError(
                    "unable to construct the requested 32:24:8 B-MSE batch; "
                    f"offline={len(offline)} continuous={len(continuous)} grounded={len(grounded)}"
                )
            kwargs = {
                "b_indices": b_indices,
                "counts": {
                    "b_mse": len(b_indices),
                    "b_offline": sum(
                        self._kinds[index] == "offline" for index in b_indices
                    ),
                    "b_continuous": sum(
                        self._kinds[index] == "continuous" for index in b_indices
                    ),
                    "b_grounded": sum(
                        self._kinds[index] == "grounded" for index in b_indices
                    ),
                },
            }
        if arm in {"t4_online_offline_a", "t5_online_hindsight_a", "t6_online_distill_a"}:
            kwargs["offline_a_indices"] = self._draw(offline, int(auxiliary_batch_size))
        if arm == "t3_online_ranking":
            kwargs["ranking_pairs"] = self._grounded_pairs(int(auxiliary_batch_size))
        if arm == "t5_online_hindsight_a":
            kwargs["hindsight_indices"] = self._draw(
                self._pool("continuous") + self._pool("grounded"),
                int(auxiliary_batch_size),
            )
        if arm == "t6_online_distill_a":
            kwargs["distill_indices"] = self._draw(
                self._eligible_distill(),
                int(auxiliary_batch_size),
            )
        counts = dict(kwargs.get("counts", {}))
        counts.update(
            {
                "offline_a": len(kwargs.get("offline_a_indices", ())),
                "ranking_pairs": len(kwargs.get("ranking_pairs", ())),
                "hindsight": len(kwargs.get("hindsight_indices", ())),
                "distill": len(kwargs.get("distill_indices", ())),
            }
        )
        for field in (
            "offline_a_indices",
            "hindsight_indices",
            "distill_indices",
        ):
            values = tuple(int(value) for value in kwargs.get(field, ()))
            label = field.removesuffix("_indices")
            for kind in ("offline", "continuous", "grounded"):
                counts[f"{label}_{kind}"] = sum(
                    self._kinds[index] == kind for index in values
                )
        kwargs["counts"] = counts
        return SupervisionBatch(**kwargs)


@dataclass(frozen=True)
class LossWeights:
    """Frozen coefficients for the auxiliary objectives."""

    rank: float = 0.0
    offline_a: float = 0.0
    hindsight: float = 0.0
    distill: float = 0.0

    def __post_init__(self) -> None:
        for name in ("rank", "offline_a", "hindsight", "distill"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"loss weight {name} must be finite and non-negative")


@dataclass(frozen=True)
class SupervisionUpdateReport:
    arm: str
    optimizer_step: int
    loss: float | None
    b_mse: float | None
    rank: float | None
    offline_a: float | None
    hindsight: float | None
    distill: float | None
    weights: LossWeights
    batch: Mapping[str, Any]
    gradient_norm_before_clip: float | None = None
    gradient_norm_after_clip: float | None = None
    skipped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_scalar(value: torch.Tensor) -> float:
    result = float(value.detach().cpu())
    if not np.isfinite(result):
        raise FloatingPointError("training loss became non-finite")
    return result


class FastOnlineSupervisionAdapter:
    """Train one Round3 arm with a single combined backward pass per update."""

    def __init__(
        self,
        model: nn.Module,
        *,
        arm: str,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-3,
        gradient_clip_norm: float = 1.0,
        seed: int = 3072,
        device: str | torch.device = "cpu",
        loss_weights: LossWeights | None = None,
    ):
        if str(arm) not in TRAINING_ARMS:
            raise ValueError(f"unknown Round3 training arm: {arm}")
        self.arm = str(arm)
        self.model = model.to(device)
        self.device = torch.device(device)
        freeze_fast_encoder_projector(self.model)
        # Round3 deliberately excludes D/E if an R4-compatible model is passed.
        for name, parameter in self.model.named_parameters():
            if name.startswith("d_") or name.startswith("e_"):
                parameter.requires_grad_(False)
        self._parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not self._parameters:
            raise ValueError("no trainable Fast-LeWAM parameters remain")
        self._trainable_names = tuple(
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        )
        shared_prefixes = (
            "predictor",
            "action_input",
            "latent_input",
            "z_condition",
            "time_mlp",
            "task_condition",
            "action_positions",
            "goal_token_embedding",
            "joint_positions",
            "query_tokens",
            "time_positions",
            "type_embeddings",
            "query_content",
        )
        self._shared_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
            and any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in shared_prefixes
            )
        ]
        self.optimizer = torch.optim.AdamW(
            self._parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.sampler_seed = int(seed)
        self.optimizer_steps = 0
        self.loss_weights = loss_weights or LossWeights()
        self.calibration: dict[str, Any] | None = (
            {
                "status": "reused",
                "weights": asdict(self.loss_weights),
            }
            if loss_weights is not None
            else None
        )
        self._torch_generator = torch.Generator(device=self.device).manual_seed(int(seed))

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return self._trainable_names

    def prepare_for_update(self) -> None:
        freeze_fast_encoder_projector(self.model)
        for name, parameter in self.model.named_parameters():
            if name.startswith("d_") or name.startswith("e_"):
                parameter.requires_grad_(False)
        set_fast_adaptation_mode(self.model)

    def _encode(self, replay: TransitionReplay, indices: Iterable[int]):
        selected = torch.as_tensor(tuple(int(value) for value in indices), dtype=torch.long)
        if selected.numel() == 0:
            raise ValueError("cannot encode an empty supervision batch")
        observations = replay.observations[selected].to(self.device)
        actions = replay.actions[selected].to(self.device)
        if int(observations.shape[1]) != int(self.model.action_horizon) + 1:
            raise ValueError("replay observation length does not match Fast-LeWAM horizon")
        if int(actions.shape[1]) != int(self.model.action_horizon):
            raise ValueError("replay action length does not match Fast-LeWAM horizon")
        with torch.no_grad():
            embeddings = self.model.encode_pixels(observations)
        return embeddings, actions, selected

    def _goal_latents(
        self,
        replay: TransitionReplay,
        selected: torch.Tensor,
        embeddings: torch.Tensor,
        *,
        require_original: bool,
    ) -> torch.Tensor:
        if replay.goal_latents is not None:
            return replay.goal_latents[selected].to(self.device).float()
        if replay.goals is not None:
            goals = replay.goals[selected].to(self.device)
            if goals.ndim == 2 and int(goals.shape[-1]) == int(self.model.latent_dim):
                return goals.float()
            if goals.ndim >= 4:
                with torch.no_grad():
                    return self.model.encode_pixels(goals).reshape(len(selected), -1)
        if require_original:
            raise ValueError("original goal latent is required for this supervision loss")
        return embeddings[:, -1].detach()

    def _stage_a_goal(self, goal_latents: torch.Tensor | None) -> torch.Tensor | None:
        if getattr(self.model, "stage_a_goal_injection", "none") == "token":
            if goal_latents is None:
                raise ValueError("goal latent is required by Stage-A token conditioning")
            return goal_latents
        return None

    def _b_mse(self, replay: TransitionReplay, indices: Iterable[int]) -> torch.Tensor:
        embeddings, actions, _ = self._encode(replay, indices)
        states = embeddings[:, :-1]
        target = embeddings[:, 1:].detach()
        predicted = self.model.predict_training_latents(states, actions)
        return F.mse_loss(predicted, target)

    def _stage_a_flow_loss(
        self,
        replay: TransitionReplay,
        indices: Iterable[int],
        *,
        hindsight: bool,
    ) -> torch.Tensor:
        embeddings, clean_actions, selected = self._encode(replay, indices)
        goal_latents = (
            embeddings[:, -1].detach()
            if hindsight
            else (
                self._goal_latents(
                    replay,
                    selected,
                    embeddings,
                    require_original=True,
                )
                if getattr(self.model, "stage_a_goal_injection", "none") == "token"
                else None
            )
        )
        timestep = torch.rand(
            (len(selected),),
            device=self.device,
            dtype=clean_actions.dtype,
            generator=self._torch_generator,
        )
        noise = torch.randn(
            clean_actions.shape,
            device=self.device,
            dtype=clean_actions.dtype,
            generator=self._torch_generator,
        )
        noisy = (
            (1.0 - timestep[:, None, None]) * noise
            + timestep[:, None, None] * clean_actions
        )
        output = self.model(
            embeddings[:, 0],
            noisy,
            timestep,
            mode="stage_a",
            goal_latent=self._stage_a_goal(goal_latents),
        )
        return F.mse_loss(output["action_velocity"], clean_actions - noise)

    def _ranking_loss(
        self,
        replay: TransitionReplay,
        pairs: Iterable[tuple[int, int]],
    ) -> torch.Tensor:
        pair_values = tuple(pairs)
        if not pair_values:
            return self._zero_loss()
        unique = tuple(dict.fromkeys(index for pair in pair_values for index in pair))
        embeddings, actions, selected = self._encode(replay, unique)
        goal_latents = self._goal_latents(
            replay,
            selected,
            embeddings,
            require_original=True,
        )
        predicted = self.model.predict_training_latents(embeddings[:, :-1], actions)
        costs = (predicted[:, -1] - goal_latents).square().mean(dim=-1)
        positions = {int(index): position for position, index in enumerate(unique)}
        values = []
        for preferred, rejected in pair_values:
            values.append(torch.logaddexp(
                costs.new_zeros(()),
                costs[positions[int(preferred)]] - costs[positions[int(rejected)]],
            ))
        return torch.stack(values).mean()

    def _zero_loss(self) -> torch.Tensor:
        return self._parameters[0].sum() * 0.0

    def _loss_terms(
        self,
        replay: TransitionReplay,
        batch: SupervisionBatch,
    ) -> dict[str, torch.Tensor]:
        terms = {
            "b_mse": self._b_mse(replay, batch.b_indices),
            "rank": self._zero_loss(),
            "offline_a": self._zero_loss(),
            "hindsight": self._zero_loss(),
            "distill": self._zero_loss(),
        }
        if self.arm == "t3_online_ranking":
            terms["rank"] = self._ranking_loss(replay, batch.ranking_pairs)
        elif self.arm == "t4_online_offline_a":
            terms["offline_a"] = self._stage_a_flow_loss(
                replay, batch.offline_a_indices, hindsight=False
            )
        elif self.arm == "t5_online_hindsight_a":
            terms["offline_a"] = self._stage_a_flow_loss(
                replay, batch.offline_a_indices, hindsight=False
            )
            if batch.hindsight_indices:
                terms["hindsight"] = self._stage_a_flow_loss(
                    replay, batch.hindsight_indices, hindsight=True
                )
        elif self.arm == "t6_online_distill_a":
            terms["offline_a"] = self._stage_a_flow_loss(
                replay, batch.offline_a_indices, hindsight=False
            )
            if batch.distill_indices:
                terms["distill"] = self._stage_a_flow_loss(
                    replay, batch.distill_indices, hindsight=False
                )
        return terms

    def _shared_norms(
        self,
        base: torch.Tensor,
        auxiliary: torch.Tensor,
    ) -> tuple[float, float]:
        base_grads = torch.autograd.grad(
            base,
            self._shared_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        auxiliary_grads = torch.autograd.grad(
            auxiliary,
            self._shared_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        base_values = []
        auxiliary_values = []
        for base_grad, auxiliary_grad in zip(base_grads, auxiliary_grads):
            if base_grad is not None:
                base_values.append(base_grad.float().reshape(-1))
            if auxiliary_grad is not None:
                auxiliary_values.append(auxiliary_grad.float().reshape(-1))
        if not base_values or not auxiliary_values:
            return 0.0, 0.0
        return (
            float(torch.cat(base_values).norm().detach().cpu()),
            float(torch.cat(auxiliary_values).norm().detach().cpu()),
        )

    def calibrate(self, replay: TransitionReplay, batch: SupervisionBatch) -> LossWeights:
        """Calibrate active auxiliary gradients once on a fixed manifest."""
        self.prepare_for_update()
        generator_state = self._torch_generator.get_state()
        terms = self._loss_terms(replay, batch)
        values: dict[str, float] = {}
        invalid: list[str] = []
        gradient_norms: dict[str, dict[str, float]] = {}
        for name in ("rank", "offline_a", "hindsight", "distill"):
            if name == "rank" and self.arm != "t3_online_ranking":
                values[name] = 0.0
                continue
            if name == "offline_a" and self.arm not in {
                "t4_online_offline_a",
                "t5_online_hindsight_a",
                "t6_online_distill_a",
            }:
                values[name] = 0.0
                continue
            if name == "hindsight" and self.arm != "t5_online_hindsight_a":
                values[name] = 0.0
                continue
            if name == "distill" and self.arm != "t6_online_distill_a":
                values[name] = 0.0
                continue
            base_norm, auxiliary_norm = self._shared_norms(
                terms["b_mse"], terms[name]
            )
            if (
                not np.isfinite(base_norm)
                or not np.isfinite(auxiliary_norm)
                or auxiliary_norm <= np.finfo(np.float64).eps
            ):
                values[name] = 0.0
                invalid.append(name)
                gradient_norms[name] = {
                    "b_mse": float(base_norm),
                    "auxiliary": float(auxiliary_norm),
                }
            else:
                values[name] = 0.1 * base_norm / auxiliary_norm
                gradient_norms[name] = {
                    "b_mse": float(base_norm),
                    "auxiliary": float(auxiliary_norm),
                }
        self.loss_weights = LossWeights(
            rank=values["rank"],
            offline_a=values["offline_a"],
            hindsight=values["hindsight"],
            distill=values["distill"],
        )
        self.calibration = {
            "status": "invalid" if invalid else "ok",
            "invalid_terms": invalid,
            "batch": batch.to_dict(),
            "weights": asdict(self.loss_weights),
            "gradient_norms": gradient_norms,
        }
        # Calibration is a diagnostic pass, not an additional training
        # sample.  Restore the flow RNG so the first optimizer update is
        # reproducible independently of whether calibration was recomputed.
        self._torch_generator.set_state(generator_state)
        return self.loss_weights

    def _total_loss(self, terms: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return (
            terms["b_mse"]
            + float(self.loss_weights.rank) * terms["rank"]
            + float(self.loss_weights.offline_a) * terms["offline_a"]
            + float(self.loss_weights.hindsight) * terms["hindsight"]
            + float(self.loss_weights.distill) * terms["distill"]
        )

    def update(
        self,
        replay: TransitionReplay,
        *,
        batch: SupervisionBatch | None = None,
        batch_size: int = 64,
        auxiliary_batch_size: int = 32,
        require_grounded: bool = False,
    ) -> SupervisionUpdateReport:
        self.prepare_for_update()
        if batch is None:
            batch = SupervisionBatchSampler(
                replay,
                seed=self.sampler_seed + self.optimizer_steps,
            ).sample(
                self.arm,
                batch_size=int(batch_size),
                auxiliary_batch_size=int(auxiliary_batch_size),
                require_grounded=require_grounded,
            )
        if self.arm == "t0_frozen":
            raise RuntimeError("T0 is a frozen evaluation arm and does not update")
        if self.arm != "t1_offline_b_mse" and self.calibration is None:
            self.calibrate(replay, batch)
        self.optimizer.zero_grad(set_to_none=True)
        terms = self._loss_terms(replay, batch)
        total = self._total_loss(terms)
        if not torch.isfinite(total):
            raise FloatingPointError("combined Round3 supervision loss is non-finite")
        total.backward()
        gradient_norm_before_clip = torch.nn.utils.clip_grad_norm_(
            self._parameters, self.gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm_before_clip):
            raise FloatingPointError("combined Round3 gradient is non-finite")
        gradient_norm_after_clip = torch.linalg.vector_norm(
            torch.cat(
                [
                    parameter.grad.detach().float().reshape(-1)
                    for parameter in self._parameters
                    if parameter.grad is not None
                ]
            )
        )
        self.optimizer.step()
        self.optimizer_steps += 1
        return SupervisionUpdateReport(
            arm=self.arm,
            optimizer_step=int(self.optimizer_steps),
            loss=_finite_scalar(total),
            b_mse=_finite_scalar(terms["b_mse"]),
            rank=_finite_scalar(terms["rank"]),
            offline_a=_finite_scalar(terms["offline_a"]),
            hindsight=_finite_scalar(terms["hindsight"]),
            distill=_finite_scalar(terms["distill"]),
            weights=self.loss_weights,
            batch=batch.to_dict(),
            gradient_norm_before_clip=_finite_scalar(gradient_norm_before_clip),
            gradient_norm_after_clip=_finite_scalar(gradient_norm_after_clip),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_steps": int(self.optimizer_steps),
            "loss_weights": asdict(self.loss_weights),
            "calibration": self.calibration,
            "torch_generator": self._torch_generator.get_state(),
            "trainable_parameter_names": self.trainable_parameter_names,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.optimizer_steps = int(state["optimizer_steps"])
        self.loss_weights = LossWeights(**state.get("loss_weights", {}))
        self.calibration = state.get("calibration")
        # ``torch.load(..., map_location=device)`` maps every tensor in the
        # adapter state, including this CPU-owned RNG state, to CUDA.  The
        # Generator API only accepts a CPU ByteTensor here.
        generator_state = state["torch_generator"]
        if isinstance(generator_state, torch.Tensor):
            generator_state = generator_state.detach().cpu()
        self._torch_generator.set_state(generator_state)


__all__ = [
    "FastOnlineSupervisionAdapter",
    "LossWeights",
    "SupervisionBatch",
    "SupervisionBatchSampler",
    "SupervisionUpdateReport",
    "TRAINING_ARMS",
]
