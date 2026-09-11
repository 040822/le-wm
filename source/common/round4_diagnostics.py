"""Small epoch-boundary diagnostics callback for Round 4 training."""

from __future__ import annotations

import json
from pathlib import Path

from lightning.pytorch.callbacks import Callback
import torch
import torch.nn.functional as F


class Round4DiagnosticsCallback(Callback):
    """Persist epoch 0, 5, and 10 branch diagnostics beside a run."""

    def __init__(self, output_name: str = "round4_diagnostics.jsonl"):
        super().__init__()
        self.output_name = output_name

    def _write(self, trainer, pl_module, epoch: int) -> None:
        metrics = dict(getattr(pl_module, "_round4_last_metrics", {}))
        payload = {"epoch": int(epoch), **metrics}
        root = getattr(trainer, "default_root_dir", None)
        if root is None:
            return
        path = Path(root) / self.output_name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def on_train_start(self, trainer, pl_module) -> None:
        self._write(trainer, pl_module, 0)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = int(trainer.current_epoch) + 1
        if epoch in {5, 10}:
            self._write(trainer, pl_module, epoch)


@torch.no_grad()
def compute_idm_diagnostics(
    model,
    real_paths: torch.Tensor,
    real_actions: torch.Tensor,
    *,
    z_start: torch.Tensor | None = None,
    z_goal: torch.Tensor | None = None,
    generated_paths: torch.Tensor | None = None,
) -> dict[str, float | None]:
    """Separate E error on real paths from B verification of decoded paths."""
    if real_paths.ndim != 3 or real_actions.ndim != 3:
        raise ValueError("real_paths must be [B,H+1,D] and real_actions must be [B,H,A]")
    predicted_real = model.predict_inverse_dynamics(real_paths)
    if predicted_real.shape != real_actions.shape:
        raise ValueError(
            f"IDM output shape {tuple(predicted_real.shape)} does not match "
            f"actions {tuple(real_actions.shape)}"
        )
    decoded_mse = F.mse_loss(predicted_real, real_actions)
    result: dict[str, float | None] = {
        "real_path_idm_action_mse": float(decoded_mse.cpu()),
        "decoded_action_b_verifier_mse": None,
        "generated_path_b_validation_mse": None,
        "generated_endpoint_mse": None,
    }
    decoded = predicted_real[:, None]
    if z_start is not None and z_goal is not None:
        if z_start.ndim != 2 or z_goal.shape != z_start.shape:
            raise ValueError("z_start and z_goal must have identical [B,D] shapes")
        verifier_cost = model.get_cost_from_latents(z_start, z_goal, decoded)
        result["decoded_action_b_verifier_mse"] = float(verifier_cost.mean().cpu())
    if generated_paths is not None:
        if generated_paths.ndim != 3:
            raise ValueError("generated_paths must have shape [B,H+1,D]")
        if z_start is None or z_goal is None:
            raise ValueError("generated path diagnostics require z_start and z_goal")
        generated_actions = model.predict_inverse_dynamics(generated_paths)
        predicted = model.rollout_latents(
            z_start,
            generated_actions,
            torch.ones(
                generated_paths.shape[0],
                device=generated_paths.device,
                dtype=generated_paths.dtype,
            ),
        )
        result["generated_path_b_validation_mse"] = float(
            F.mse_loss(predicted, generated_paths[:, 1:]).cpu()
        )
        result["generated_endpoint_mse"] = float(
            F.mse_loss(generated_paths[:, -1], z_goal).cpu()
        )
    return result


__all__ = ["Round4DiagnosticsCallback", "compute_idm_diagnostics"]
