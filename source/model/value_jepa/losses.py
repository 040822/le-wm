from __future__ import annotations

from typing import NamedTuple

import torch


class ValueLossResult(NamedTuple):
    loss: torch.Tensor
    td_error: torch.Tensor
    value: torch.Tensor
    target: torch.Tensor


def squared_latent_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).pow(2).mean(dim=-1)


def expectile_loss(td_error: torch.Tensor, tau: float) -> torch.Tensor:
    tau_tensor = torch.as_tensor(tau, dtype=td_error.dtype, device=td_error.device)
    weights = torch.abs(tau_tensor - (td_error < 0).to(td_error.dtype))
    return (weights * td_error.pow(2)).mean()


def value_td_loss(
    current: torch.Tensor,
    following: torch.Tensor,
    goal: torch.Tensor,
    reward: torch.Tensor,
    gamma: float,
    tau: float,
) -> ValueLossResult:
    value = -squared_latent_distance(current, goal)
    with torch.no_grad():
        following_value = -squared_latent_distance(following, goal)
        target = reward + gamma * following_value
    td_error = target - value
    return ValueLossResult(
        loss=expectile_loss(td_error, tau),
        td_error=td_error,
        value=value,
        target=target,
    )
