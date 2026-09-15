"""Action-space bounds for Fast-LeWAM's normalized action coordinates.

Fast-LeWAM predicts z-score normalized action blocks, while the environment
expects physical actions.  This module keeps the existing z-score transform
unchanged and provides an explicit, reversible description of the physical
action bounds in the normalized coordinates used by the planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


ProjectionMode = Literal["none", "clip", "global_scale"]


def _finite_vector(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


@dataclass(frozen=True)
class NormalizedActionBounds:
    """Physical and normalized bounds for one environment action vector.

    ``physical_*`` and ``normalized_*`` describe the base environment action
    dimension.  ``for_action_dim`` repeats the normalized bounds so the
    object can be applied directly to Fast-LeWAM action tokens.
    """

    physical_low: np.ndarray
    physical_high: np.ndarray
    normalized_low: np.ndarray
    normalized_high: np.ndarray
    action_block: int = 1

    def __post_init__(self) -> None:
        physical_low = _finite_vector(self.physical_low, name="physical_low")
        physical_high = _finite_vector(self.physical_high, name="physical_high")
        normalized_low = _finite_vector(
            self.normalized_low, name="normalized_low"
        )
        normalized_high = _finite_vector(
            self.normalized_high, name="normalized_high"
        )
        if physical_low.shape != physical_high.shape:
            raise ValueError("physical action bounds must have matching shapes")
        if normalized_low.shape != normalized_high.shape:
            raise ValueError("normalized action bounds must have matching shapes")
        if physical_low.shape != normalized_low.shape:
            raise ValueError(
                "physical and normalized base action bounds must have matching shapes"
            )
        if np.any(physical_low > physical_high):
            raise ValueError("physical_low must not exceed physical_high")
        if np.any(normalized_low > normalized_high):
            raise ValueError("normalized_low must not exceed normalized_high")
        action_block = int(self.action_block)
        if action_block < 1:
            raise ValueError("action_block must be positive")
        object.__setattr__(self, "physical_low", physical_low.copy())
        object.__setattr__(self, "physical_high", physical_high.copy())
        object.__setattr__(self, "normalized_low", normalized_low.copy())
        object.__setattr__(self, "normalized_high", normalized_high.copy())
        object.__setattr__(self, "action_block", action_block)

    @property
    def base_action_dim(self) -> int:
        """Number of physical environment action coordinates."""
        return int(self.physical_low.size)

    @property
    def action_block_dim(self) -> int:
        """Number of normalized coordinates in one action block."""
        return self.base_action_dim * self.action_block

    def for_action_dim(self, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
        """Return normalized bounds expanded to a model action dimension."""
        action_dim = int(action_dim)
        if action_dim < 1 or action_dim % self.base_action_dim != 0:
            raise ValueError(
                f"action_dim={action_dim} is not a multiple of base action "
                f"dimension {self.base_action_dim}"
            )
        repeats = action_dim // self.base_action_dim
        return (
            np.tile(self.normalized_low, repeats),
            np.tile(self.normalized_high, repeats),
        )

    def metadata(self, *, action_dim: int | None = None) -> dict[str, object]:
        """Serialize the bound specification for experiment traces."""
        normalized_low = self.normalized_low
        normalized_high = self.normalized_high
        if action_dim is not None:
            normalized_low, normalized_high = self.for_action_dim(action_dim)
        return {
            "physical_low": self.physical_low.tolist(),
            "physical_high": self.physical_high.tolist(),
            "normalized_low": normalized_low.tolist(),
            "normalized_high": normalized_high.tolist(),
            "base_action_dim": self.base_action_dim,
            "action_block": self.action_block,
            "action_dim": None if action_dim is None else int(action_dim),
        }


def compute_normalized_action_bounds(
    action_space,
    action_processor,
    *,
    action_block: int = 1,
) -> NormalizedActionBounds:
    """Map a physical action-space box through an existing StandardScaler.

    The evaluator fits ``action_processor`` on raw dataset actions and later
    calls ``inverse_transform`` before handing actions to the environment.
    Therefore a physical bound ``b`` corresponds to
    ``(b - processor.mean_) / processor.scale_`` in planner coordinates.
    """
    if not hasattr(action_space, "low") or not hasattr(action_space, "high"):
        raise TypeError("action_space must expose finite low/high bounds")
    if not hasattr(action_processor, "mean_") or not hasattr(
        action_processor, "scale_"
    ):
        raise TypeError("action_processor must expose mean_ and scale_")

    physical_low = _finite_vector(action_space.low, name="action_space.low")
    physical_high = _finite_vector(action_space.high, name="action_space.high")
    mean = _finite_vector(action_processor.mean_, name="action_processor.mean_")
    scale = _finite_vector(action_processor.scale_, name="action_processor.scale_")
    if physical_low.shape != mean.shape or physical_high.shape != mean.shape:
        raise ValueError(
            "action-space bounds and action processor statistics have different dimensions"
        )
    if np.any(scale <= 0.0):
        raise ValueError("action_processor.scale_ must be strictly positive")
    normalized_low = (physical_low - mean) / scale
    normalized_high = (physical_high - mean) / scale
    return NormalizedActionBounds(
        physical_low=physical_low,
        physical_high=physical_high,
        normalized_low=normalized_low,
        normalized_high=normalized_high,
        action_block=action_block,
    )


def _expanded_bounds(
    bounds: NormalizedActionBounds,
    action_dim: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    low, high = bounds.for_action_dim(action_dim)
    return (
        torch.as_tensor(low, device=device, dtype=dtype),
        torch.as_tensor(high, device=device, dtype=dtype),
    )


def _validate_projection_mode(mode: str) -> ProjectionMode:
    mode = str(mode).lower()
    if mode not in {"none", "clip", "global_scale"}:
        raise ValueError(
            "projection mode must be one of 'none', 'clip', or 'global_scale'"
        )
    return mode  # type: ignore[return-value]


def _scale_factors(
    actions: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Compute one around-zero scale factor per action path."""
    if torch.any(lower > 0) or torch.any(upper < 0):
        raise ValueError("global_scale requires bounds that contain zero")
    if actions.ndim == 1:
        flat = actions.reshape(1, -1)
        path_shape: tuple[int, ...] = ()
    elif actions.ndim == 2:
        flat = actions.reshape(actions.shape[0], actions.shape[1])
        path_shape = (actions.shape[0],)
    elif actions.ndim >= 3:
        flat = actions.reshape(-1, actions.shape[-2] * actions.shape[-1])
        path_shape = tuple(actions.shape[:-2])
    else:  # pragma: no cover - tensors cannot have fewer than one dimension
        raise ValueError("actions must have at least one dimension")

    repeated_lower = lower.repeat(flat.shape[-1] // lower.numel())
    repeated_upper = upper.repeat(flat.shape[-1] // upper.numel())
    positive_ratio = torch.where(
        flat > 0,
        repeated_upper.unsqueeze(0) / flat,
        torch.full_like(flat, float("inf")),
    )
    negative_ratio = torch.where(
        flat < 0,
        repeated_lower.unsqueeze(0) / flat,
        torch.full_like(flat, float("inf")),
    )
    ratios = torch.minimum(positive_ratio, negative_ratio)
    factors = torch.minimum(
        torch.ones(flat.shape[0], device=actions.device, dtype=actions.dtype),
        ratios.min(dim=-1).values,
    ).clamp_min(0.0)
    return factors.reshape(path_shape) if path_shape else factors.reshape(())


def project_normalized_actions(
    actions: torch.Tensor | np.ndarray,
    bounds: NormalizedActionBounds,
    *,
    mode: ProjectionMode = "none",
) -> torch.Tensor | np.ndarray:
    """Project normalized action paths without mutating the input.

    ``clip`` projects every coordinate independently.  ``global_scale`` uses
    one non-negative scale per path, centered at zero, so the path's relative
    action pattern is preserved.  The last two dimensions are interpreted as
    ``[path_length, action_dim]``; a two-dimensional ``[batch, action_dim]``
    input is treated as a path of length one.
    """
    mode = _validate_projection_mode(mode)
    if isinstance(actions, np.ndarray):
        if actions.ndim == 0 or not np.all(np.isfinite(actions)):
            raise ValueError("actions must be a finite non-scalar array")
        if actions.ndim == 1:
            action_dim = actions.shape[-1]
            low, high = bounds.for_action_dim(action_dim)
            projected = np.array(actions, copy=True)
            if mode == "clip":
                return np.clip(projected, low, high)
            if mode == "global_scale":
                tensor = torch.from_numpy(projected.astype(np.float64, copy=False))
                factor = _scale_factors(
                    tensor,
                    torch.from_numpy(low),
                    torch.from_numpy(high),
                ).item()
                return projected * factor
            return projected
        action_dim = actions.shape[-1]
        low, high = bounds.for_action_dim(action_dim)
        projected = np.array(actions, copy=True)
        if mode == "clip":
            return np.clip(projected, low, high)
        if mode == "global_scale":
            tensor = torch.from_numpy(projected.astype(np.float64, copy=False))
            factor = _scale_factors(
                tensor,
                torch.from_numpy(low),
                torch.from_numpy(high),
            )
            if projected.ndim == 2:
                return projected * factor.numpy()[..., None]
            return projected * factor.numpy()[..., None, None]
        return projected

    if not torch.is_tensor(actions):
        raise TypeError("actions must be a torch.Tensor or numpy.ndarray")
    if actions.ndim == 0 or not torch.isfinite(actions).all():
        raise ValueError("actions must be a finite non-scalar tensor")
    action_dim = int(actions.shape[-1])
    low, high = _expanded_bounds(
        bounds,
        action_dim,
        device=actions.device,
        dtype=actions.dtype,
    )
    projected = actions.clone()
    if mode == "clip":
        return torch.maximum(torch.minimum(projected, high), low)
    if mode == "global_scale":
        factors = _scale_factors(projected, low, high)
        if projected.ndim == 1:
            return projected * factors
        if projected.ndim == 2:
            return projected * factors[..., None]
        return projected * factors[..., None, None]
    return projected


def normalized_action_stats(
    actions: torch.Tensor | np.ndarray,
    bounds: NormalizedActionBounds,
    *,
    tolerance: float = 1e-6,
) -> dict[str, float | int | list[int]]:
    """Report legacy unit-threshold and true bound violations."""
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be a finite non-negative number")
    if torch.is_tensor(actions):
        array = actions.detach().float().cpu().numpy()
    else:
        array = np.asarray(actions)
    if array.ndim == 0 or not np.all(np.isfinite(array)):
        raise ValueError("actions must be a finite non-scalar array")
    low, high = bounds.for_action_dim(array.shape[-1])
    lower = low.reshape((1,) * (array.ndim - 1) + (len(low),))
    upper = high.reshape((1,) * (array.ndim - 1) + (len(high),))
    true_violation = (array < lower - tolerance) | (array > upper + tolerance)
    return {
        "shape": [int(item) for item in array.shape],
        "mean_abs": float(np.mean(np.abs(array))),
        "max_abs": float(np.max(np.abs(array))),
        "legacy_unit_threshold_fraction": float(np.mean(np.abs(array) > 1.0)),
        "true_normalized_bound_violation_fraction": float(
            np.mean(true_violation)
        ),
        "true_normalized_bound_violation_count": int(np.sum(true_violation)),
    }


__all__ = [
    "NormalizedActionBounds",
    "ProjectionMode",
    "compute_normalized_action_bounds",
    "normalized_action_stats",
    "project_normalized_actions",
]
