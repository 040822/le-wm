"""Record executed Cube actions at LeWM action-block boundaries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from .round3_phase2 import TransitionReplay


def _batch_pixels(value: Any) -> torch.Tensor:
    pixels = torch.as_tensor(value).detach().cpu()
    if pixels.ndim == 5 and pixels.shape[1] == 1:
        pixels = pixels[:, 0]
    if pixels.ndim != 4:
        raise ValueError(f"expected batched pixels [N,C,H,W], got {tuple(pixels.shape)}")
    if pixels.shape[1] not in (1, 3) and pixels.shape[-1] in (1, 3):
        pixels = pixels.permute(0, 3, 1, 2).contiguous()
    return pixels


def _batch_actions(value: Any) -> torch.Tensor:
    actions = torch.as_tensor(value).detach().cpu().float()
    if actions.ndim != 2:
        raise ValueError(f"expected batched actions [N,A], got {tuple(actions.shape)}")
    return actions


class ExecutedTransitionRecorder:
    """Policy proxy that records only actions actually returned to the env.

    A WorldModelPolicy emits one environment action at a time while LeWM is
    trained on action blocks. The proxy therefore closes a block only when the
    next block-boundary observation arrives. Candidate actions considered by
    CEM but not returned by the policy never enter the replay.
    """

    def __init__(
        self,
        policy: Any,
        *,
        episode_ids: list[Any] | tuple[Any, ...],
        start_steps: list[int] | tuple[int, ...],
        history_size: int = 3,
        action_block: int = 5,
        model_version: str,
        normalize_action: Callable[[torch.Tensor], torch.Tensor] | None = None,
        observation_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        data_kind: str = "continuous",
        group_ids: list[Any] | tuple[Any, ...] | None = None,
        snapshot_ids: list[Any] | tuple[Any, ...] | None = None,
        collector_version: str | None = None,
    ) -> None:
        if history_size < 1 or action_block < 1:
            raise ValueError("history_size and action_block must be positive")
        if len(episode_ids) != len(start_steps) or not episode_ids:
            raise ValueError("episode_ids and start_steps must match non-zero env count")
        self.policy = policy
        self.episode_ids = tuple(episode_ids)
        self.start_steps = tuple(int(value) for value in start_steps)
        self.history_size = int(history_size)
        self.action_block = int(action_block)
        self.model_version = str(model_version)
        self.data_kind = str(data_kind)
        if self.data_kind not in {"continuous", "grounded"}:
            raise ValueError("data_kind must be 'continuous' or 'grounded'")
        if group_ids is not None and len(group_ids) != len(episode_ids):
            raise ValueError("group_ids must match the environment count")
        if snapshot_ids is not None and len(snapshot_ids) != len(episode_ids):
            raise ValueError("snapshot_ids must match the environment count")
        self.group_ids = None if group_ids is None else tuple(group_ids)
        self.snapshot_ids = None if snapshot_ids is None else tuple(snapshot_ids)
        self.collector_version = str(collector_version or model_version)
        self.normalize_action = normalize_action or (lambda value: value.float())
        self.observation_transform = observation_transform
        count = len(self.episode_ids)
        self._steps = [0] * count
        self._block_observations: list[list[torch.Tensor]] = [[] for _ in range(count)]
        self._block_raw_observations: list[list[torch.Tensor]] = [[] for _ in range(count)]
        self._completed_actions: list[list[torch.Tensor]] = [[] for _ in range(count)]
        self._completed_raw_actions: list[list[torch.Tensor]] = [[] for _ in range(count)]
        self._current_actions: list[list[torch.Tensor]] = [[] for _ in range(count)]
        self._current_raw_actions: list[list[torch.Tensor]] = [[] for _ in range(count)]
        self._goals: list[torch.Tensor | None] = [None] * count
        self._initial_physical_info: dict[str, torch.Tensor] = {}
        self._records: list[dict[str, Any]] = []

    def set_env(self, env: Any) -> None:
        self.policy.set_env(env)

    def _processed_pixels(self, raw: torch.Tensor) -> torch.Tensor:
        if self.observation_transform is None:
            return raw.float()
        return torch.stack([self.observation_transform(frame) for frame in raw], dim=0)

    def get_action(self, info: dict[str, Any], **kwargs: Any) -> np.ndarray:
        raw_pixels = _batch_pixels(info["pixels"])
        pixels = self._processed_pixels(raw_pixels)
        if len(raw_pixels) != len(self.episode_ids):
            raise ValueError("policy info env count differs from recorder metadata")
        if "goal" in info:
            raw_goals = torch.as_tensor(info["goal"]).detach().cpu().clone()
            if raw_goals.ndim == 5 and raw_goals.shape[1] == 1:
                raw_goals = raw_goals[:, 0]
            if raw_goals.ndim == 4 and raw_goals.shape[1] not in (1, 3):
                if raw_goals.shape[-1] in (1, 3):
                    raw_goals = raw_goals.permute(0, 3, 1, 2).contiguous()
            if raw_goals.ndim >= 1 and len(raw_goals) == len(self.episode_ids):
                for index in range(len(self._goals)):
                    if self._goals[index] is None:
                        self._goals[index] = raw_goals[index]
        if not self._initial_physical_info:
            for key in ("state", "goal_state", "qpos", "goal_qpos"):
                if key in info:
                    self._initial_physical_info[key] = (
                        torch.as_tensor(info[key]).detach().cpu().clone()
                    )

        for index in range(len(self.episode_ids)):
            step = self._steps[index]
            if step % self.action_block == 0:
                if step > 0:
                    if len(self._current_actions[index]) != self.action_block:
                        raise RuntimeError("action block closed before collecting all environment actions")
                    self._completed_actions[index].append(
                        torch.cat(self._current_actions[index], dim=-1)
                    )
                    self._completed_raw_actions[index].append(
                        torch.cat(self._current_raw_actions[index], dim=-1)
                    )
                    self._current_actions[index].clear()
                    self._current_raw_actions[index].clear()
                self._block_observations[index].append(pixels[index].clone())
                self._block_raw_observations[index].append(raw_pixels[index].clone())
                if (
                    len(self._block_observations[index]) >= self.history_size + 1
                    and len(self._completed_actions[index]) >= self.history_size
                ):
                    observation_window = torch.stack(
                        self._block_observations[index][-self.history_size - 1 :]
                    )
                    raw_observation_window = torch.stack(
                        self._block_raw_observations[index][-self.history_size - 1 :]
                    )
                    action_window = torch.stack(
                        self._completed_actions[index][-self.history_size :]
                    )
                    raw_action_window = torch.stack(
                        self._completed_raw_actions[index][-self.history_size :]
                    )
                    self._records.append({
                        "observations": observation_window,
                        "raw_observations": raw_observation_window,
                        "actions": action_window,
                        "raw_actions": raw_action_window,
                        "episode_ids": (self.episode_ids[index],) * (self.history_size + 1),
                        "start_step": self.start_steps[index] + step - self.history_size * self.action_block,
                        "goal": self._goals[index],
                        "env_index": index,
                        "group_id": (
                            None if self.group_ids is None else self.group_ids[index]
                        ),
                        "snapshot_id": (
                            None
                            if self.snapshot_ids is None
                            else self.snapshot_ids[index]
                        ),
                    })

        raw_action = _batch_actions(self.policy.get_action(info, **kwargs))
        normalized_action = torch.as_tensor(self.normalize_action(raw_action)).detach().cpu().float()
        if tuple(normalized_action.shape) != tuple(raw_action.shape):
            raise ValueError("normalize_action changed the action batch shape")
        for index in range(len(self.episode_ids)):
            self._current_actions[index].append(normalized_action[index].clone())
            self._current_raw_actions[index].append(raw_action[index].clone())
            self._steps[index] += 1
        return raw_action.numpy()

    def __call__(self, info: dict[str, Any], **kwargs: Any) -> np.ndarray:
        return self.get_action(info, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)

    @property
    def environment_steps(self) -> int:
        return int(sum(self._steps))

    @property
    def initial_physical_info(self) -> dict[str, torch.Tensor]:
        """Return the first reset's physical state fields for cost accounting."""
        return {
            key: value.detach().cpu().clone()
            for key, value in self._initial_physical_info.items()
        }

    def replay(
        self,
        *,
        source: str = "online",
        outcomes: dict[Any, dict[str, Any]] | None = None,
    ) -> TransitionReplay:
        if not self._records:
            raise RuntimeError("no complete action-block transitions were recorded")
        goals = [record["goal"] for record in self._records]
        goals_tensor = None
        if all(goal is not None for goal in goals):
            goals_tensor = torch.stack([goal.float() for goal in goals])
        outcomes = outcomes or {}

        def outcome_value(record: dict[str, Any], name: str):
            episode_id = record["episode_ids"][0]
            env_index = int(record["env_index"])
            payload = outcomes.get(
                (episode_id, env_index),
                outcomes.get(
                    f"{episode_id}:{env_index}",
                    outcomes.get(episode_id, outcomes.get(str(episode_id), {})),
                ),
            )
            return payload.get(name) if isinstance(payload, dict) else None

        successes = [outcome_value(record, "success") for record in self._records]
        start_costs = [outcome_value(record, "physical_start_cost") for record in self._records]
        terminal_costs = [outcome_value(record, "physical_terminal_cost") for record in self._records]

        def optional_float_tensor(values):
            if not values or not all(value is not None for value in values):
                return None
            return torch.as_tensor(values, dtype=torch.float32)

        success_tensor = (
            None
            if not successes or not all(value is not None for value in successes)
            else torch.as_tensor(successes, dtype=torch.bool)
        )
        return TransitionReplay(
            observations=torch.stack([record["observations"] for record in self._records]),
            actions=torch.stack([record["actions"] for record in self._records]),
            episode_ids=tuple(record["episode_ids"] for record in self._records),
            start_steps=torch.tensor([record["start_step"] for record in self._records]),
            sources=(source,) * len(self._records),
            model_versions=(self.model_version,) * len(self._records),
            goals=goals_tensor,
            raw_observations=torch.stack(
                [record["raw_observations"] for record in self._records]
            ),
            raw_actions=torch.stack([record["raw_actions"] for record in self._records]),
            data_kinds=(self.data_kind,) * len(self._records),
            group_ids=tuple(
                record["group_id"]
                if record["group_id"] is not None
                else f"{record['episode_ids'][0]}:{record['start_step']}"
                for record in self._records
            ),
            snapshot_ids=(
                None
                if all(record["snapshot_id"] is None for record in self._records)
                else tuple(record["snapshot_id"] for record in self._records)
            ),
            collector_versions=(self.collector_version,) * len(self._records),
            successes=success_tensor,
            physical_start_costs=optional_float_tensor(start_costs),
            physical_terminal_costs=optional_float_tensor(terminal_costs),
            execution_lengths=torch.full(
                (len(self._records),),
                int(self.history_size * self.action_block),
                dtype=torch.long,
            ),
        )


__all__ = ["ExecutedTransitionRecorder"]
