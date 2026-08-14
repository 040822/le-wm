"""True-transition replay and Stage-B-only adaptation for Fast-LeWAM pilots."""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PlannerTransitionReplay:
    """Encoded true simulator transitions grouped by collection context."""

    states: torch.Tensor
    actions: torch.Tensor
    targets: torch.Tensor
    sources: tuple[str, ...]
    groups: tuple[int, ...]
    identity: Mapping[str, Any]
    goal_latents: torch.Tensor | None = None
    physical_costs: torch.Tensor | None = None
    successes: torch.Tensor | None = None

    def __post_init__(self):
        count = int(self.actions.shape[0])
        if self.states.ndim != 3 or self.actions.ndim != 3 or self.targets.ndim != 3:
            raise ValueError("replay tensors must have shape [N,H,D]")
        if self.states.shape[:2] != self.actions.shape[:2]:
            raise ValueError("state and action replay horizons differ")
        if self.states.shape != self.targets.shape:
            raise ValueError("state and target replay shapes differ")
        if len(self.sources) != count or len(self.groups) != count:
            raise ValueError("replay provenance length differs from tensor count")
        if count < 1:
            raise ValueError("transition replay is empty")
        for name, value in (
            ("states", self.states),
            ("actions", self.actions),
            ("targets", self.targets),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"replay {name} contain non-finite values")
        optional = (self.goal_latents, self.physical_costs, self.successes)
        if any(value is not None for value in optional):
            if any(value is None for value in optional):
                raise ValueError("grounded replay fields must be provided together")
            if tuple(self.goal_latents.shape) != (count, self.states.shape[-1]):
                raise ValueError("goal latent shape differs from replay count")
            if tuple(self.physical_costs.shape) != (count,):
                raise ValueError("physical cost shape differs from replay count")
            if tuple(self.successes.shape) != (count,):
                raise ValueError("success shape differs from replay count")
            if not torch.isfinite(self.goal_latents).all():
                raise ValueError("goal latents contain non-finite values")
            if not torch.isfinite(self.physical_costs).all():
                raise ValueError("physical costs contain non-finite values")


def _stage_b_trainable_parameters(model):
    model.requires_grad_(False)
    modules = (
        model.action_input,
        model.latent_input,
        model.z_condition,
        model.time_mlp,
        model.predictor,
        model.latent_head,
    )
    for module in modules:
        module.requires_grad_(True)
    positional = (
        "action_positions",
        "joint_positions",
        "query_tokens",
        "time_positions",
        "type_embeddings",
        "query_content",
    )
    for name in positional:
        value = getattr(model, name, None)
        if isinstance(value, torch.nn.Module):
            value.requires_grad_(True)
        elif isinstance(value, torch.Tensor):
            value.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _balanced_epoch_indices(replay, *, expert_fraction, generator):
    expert = torch.tensor(
        [index for index, value in enumerate(replay.sources) if value == "expert"],
        dtype=torch.long,
    )
    planner = torch.tensor(
        [index for index, value in enumerate(replay.sources) if value != "expert"],
        dtype=torch.long,
    )
    if len(expert) == 0 or len(planner) == 0:
        raise ValueError("replay must contain both expert and planner transitions")
    total = len(replay.sources)
    expert_count = min(max(round(total * float(expert_fraction)), 1), total - 1)
    planner_count = total - expert_count

    def draw(pool, count):
        selections = []
        while len(selections) < count:
            order = pool[torch.randperm(len(pool), generator=generator)].tolist()
            selections.extend(order)
        return selections[:count]

    merged = draw(expert, expert_count) + draw(planner, planner_count)
    permutation = torch.randperm(len(merged), generator=generator).tolist()
    return [merged[index] for index in permutation]


class StageBReplayTrainer:
    """Stateful optimizer shared by offline and streaming replay schedules."""

    def __init__(
        self,
        model,
        *,
        learning_rate: float,
        weight_decay: float,
        expert_fraction: float,
        seed: int,
        device: str,
    ):
        if not 0 < expert_fraction < 1:
            raise ValueError("expert_fraction must be in (0, 1)")
        self.model = model.to(device).train()
        self.device = device
        self.expert_fraction = float(expert_fraction)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.parameters = _stage_b_trainable_parameters(self.model)
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        self.optimizer_steps = 0
        self.losses: list[float] = []
        self.source_updates = Counter()

    def train(self, replay: PlannerTransitionReplay, *, epochs: int, batch_size: int):
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        expected = (
            self.model.action_horizon,
            self.model.latent_dim,
            self.model.action_dim,
        )
        actual = (
            replay.states.shape[1],
            replay.states.shape[2],
            replay.actions.shape[2],
        )
        if actual != expected:
            raise ValueError(f"replay shape {actual} does not match model {expected}")
        for _ in range(int(epochs)):
            indices = _balanced_epoch_indices(
                replay,
                expert_fraction=self.expert_fraction,
                generator=self.generator,
            )
            for offset in range(0, len(indices), int(batch_size)):
                selected = indices[offset : offset + int(batch_size)]
                states = replay.states[selected].to(self.device)
                actions = replay.actions[selected].to(self.device)
                targets = replay.targets[selected].to(self.device)
                predicted = self.model.predict_training_latents(states, actions)
                loss = F.mse_loss(predicted, targets)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
                self.optimizer.step()
                self.optimizer_steps += 1
                self.losses.append(float(loss.detach().cpu()))
                self.source_updates.update(replay.sources[index] for index in selected)

    def state_dict(self):
        return {
            "model": copy.deepcopy(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "generator": self.generator.get_state(),
            "optimizer_steps": self.optimizer_steps,
            "losses": list(self.losses),
            "source_updates": dict(self.source_updates),
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"])
        self.optimizer_steps = int(state["optimizer_steps"])
        self.losses = [float(value) for value in state["losses"]]
        self.source_updates = Counter(state["source_updates"])

    def report(self):
        self.model.eval()
        return {
            "optimizer_steps": self.optimizer_steps,
            "loss_first": self.losses[0],
            "loss_last": self.losses[-1],
            "source_updates": dict(sorted(self.source_updates.items())),
        }


def fine_tune_stage_b(
    model,
    replay: PlannerTransitionReplay,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    expert_fraction: float,
    seed: int,
    device: str,
):
    """Adapt Stage B on true transitions while preserving the visual representation."""
    trainer = StageBReplayTrainer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        expert_fraction=expert_fraction,
        seed=seed,
        device=device,
    )
    trainer.train(replay, epochs=epochs, batch_size=batch_size)
    return trainer.report()


def replay_sha256(replay: PlannerTransitionReplay) -> str:
    digest = hashlib.sha256()
    tensors = (
        replay.states,
        replay.actions,
        replay.targets,
        replay.goal_latents,
        replay.physical_costs,
        replay.successes,
    )
    for value in tensors:
        if value is None:
            digest.update(b"none")
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes())
    digest.update(json.dumps(list(replay.sources)).encode())
    digest.update(json.dumps(list(replay.groups)).encode())
    digest.update(json.dumps(dict(replay.identity), sort_keys=True).encode())
    return digest.hexdigest()


def subset_replay(replay: PlannerTransitionReplay, indices) -> PlannerTransitionReplay:
    indices = [int(index) for index in indices]
    if not indices:
        raise ValueError("replay subset is empty")
    return PlannerTransitionReplay(
        states=replay.states[indices],
        actions=replay.actions[indices],
        targets=replay.targets[indices],
        sources=tuple(replay.sources[index] for index in indices),
        groups=tuple(replay.groups[index] for index in indices),
        identity={"parent_sha256": replay_sha256(replay), "indices": indices},
        goal_latents=None if replay.goal_latents is None else replay.goal_latents[indices],
        physical_costs=None if replay.physical_costs is None else replay.physical_costs[indices],
        successes=None if replay.successes is None else replay.successes[indices],
    )


def split_train_holdout(replay: PlannerTransitionReplay):
    """Keep expert anchors for training and hold out alternating planner actions."""
    train = []
    holdout = []
    per_group_planner = Counter()
    for index, (source, group) in enumerate(zip(replay.sources, replay.groups)):
        if source == "expert":
            train.append(index)
            continue
        ordinal = per_group_planner[int(group)]
        per_group_planner[int(group)] += 1
        (holdout if ordinal % 2 == 0 else train).append(index)
    if not holdout:
        raise ValueError("grounded replay has no planner holdout candidates")
    return subset_replay(replay, train), subset_replay(replay, holdout)


def save_replay(path: str | Path, replay: PlannerTransitionReplay) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 2,
            "sha256": replay_sha256(replay),
            "states": replay.states,
            "actions": replay.actions,
            "targets": replay.targets,
            "sources": replay.sources,
            "groups": replay.groups,
            "identity": dict(replay.identity),
            "goal_latents": replay.goal_latents,
            "physical_costs": replay.physical_costs,
            "successes": replay.successes,
        },
        temporary,
    )
    temporary.replace(path)


def load_replay(path: str | Path, *, expected_identity=None) -> PlannerTransitionReplay:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 2:
        raise RuntimeError(f"unsupported replay schema: {path}")
    replay = PlannerTransitionReplay(
        states=payload["states"],
        actions=payload["actions"],
        targets=payload["targets"],
        sources=tuple(payload["sources"]),
        groups=tuple(int(value) for value in payload["groups"]),
        identity=dict(payload["identity"]),
        goal_latents=payload.get("goal_latents"),
        physical_costs=payload.get("physical_costs"),
        successes=payload.get("successes"),
    )
    if payload.get("sha256") != replay_sha256(replay):
        raise RuntimeError(f"replay content hash mismatch: {path}")
    if expected_identity is not None and dict(replay.identity) != dict(expected_identity):
        raise RuntimeError(f"replay identity mismatch: {path}")
    return replay


__all__ = [
    "PlannerTransitionReplay",
    "StageBReplayTrainer",
    "fine_tune_stage_b",
    "load_replay",
    "replay_sha256",
    "save_replay",
    "split_train_holdout",
    "subset_replay",
]
