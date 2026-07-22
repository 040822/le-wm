"""Fast-LeWAM Stage B 的 held-out batch 诊断指标。"""

from contextlib import nullcontext

import torch
from lightning.pytorch.callbacks import Callback


def _repeat_task_condition(task_condition, candidate_count):
    """把每个样本的 task condition 复制到它的全部候选动作。"""
    if task_condition is None:
        return None
    return (
        task_condition[:, None]
        .expand(-1, candidate_count, *task_condition.shape[1:])
        .reshape(-1, *task_condition.shape[1:])
    )


@torch.inference_mode()
def compute_stage_b_validation_metrics(
    *,
    model,
    batch,
    forward_output,
    noise_std=0.1,
    seed=42,
):
    """比较 predicted/clean endpoint，并测量 expert 对负候选的排序质量。"""
    if noise_std <= 0:
        raise ValueError("noise_std must be positive")
    if "emb" not in forward_output or "predicted_latents" not in forward_output:
        raise ValueError("forward_output must contain emb and predicted_latents")

    embeddings = forward_output["emb"].detach()
    predicted_latents = forward_output["predicted_latents"].detach()
    clean_actions = torch.nan_to_num(batch["action"], 0.0)[
        :, : model.action_horizon
    ]
    z0 = embeddings[:, 0]
    target_latents = embeddings[:, 1 : model.action_horizon + 1]
    expected_action_shape = (
        embeddings.shape[0],
        model.action_horizon,
        model.action_dim,
    )
    if tuple(clean_actions.shape) != expected_action_shape:
        raise ValueError(
            f"validation actions must have shape {expected_action_shape}, "
            f"got {tuple(clean_actions.shape)}"
        )
    if tuple(predicted_latents.shape) != tuple(target_latents.shape):
        raise ValueError(
            "predicted_latents and target_latents must have the same shape; "
            f"got {tuple(predicted_latents.shape)} and {tuple(target_latents.shape)}"
        )

    generator = torch.Generator(device=clean_actions.device).manual_seed(int(seed))
    perturbed = clean_actions + noise_std * torch.randn(
        clean_actions.shape,
        device=clean_actions.device,
        dtype=clean_actions.dtype,
        generator=generator,
    )
    negative_actions = [perturbed]
    if clean_actions.shape[0] > 1:
        negative_actions.append(clean_actions.roll(shifts=1, dims=0))
    if model.action_horizon > 1:
        negative_actions.append(clean_actions.roll(shifts=1, dims=1))

    candidates = torch.stack([clean_actions, *negative_actions], dim=1)
    batch_size, candidate_count, horizon, action_dim = candidates.shape
    candidate_z0 = (
        z0[:, None]
        .expand(-1, candidate_count, -1)
        .reshape(batch_size * candidate_count, -1)
    )
    candidate_actions = candidates.reshape(
        batch_size * candidate_count, horizon, action_dim
    )
    candidate_timestep = torch.ones(
        batch_size * candidate_count,
        device=z0.device,
        dtype=z0.dtype,
    )
    candidate_task_condition = _repeat_task_condition(
        batch.get("task_condition"), candidate_count
    )
    candidate_latents = model(
        candidate_z0,
        candidate_actions,
        candidate_timestep,
        mode="stage_b",
        task_condition=candidate_task_condition,
    )["predicted_latents"].reshape(
        batch_size,
        candidate_count,
        model.action_horizon,
        -1,
    )

    target_terminal = target_latents[:, -1]
    candidate_costs = (
        candidate_latents[:, :, -1] - target_terminal[:, None]
    ).square().mean(dim=-1)
    expert_cost = candidate_costs[:, 0]
    negative_cost = candidate_costs[:, 1:]
    predicted_terminal_mse = (
        predicted_latents[:, -1] - target_terminal
    ).square().mean()

    return {
        "validate/stage_b_clean_terminal_mse": expert_cost.mean(),
        "validate/stage_b_predicted_terminal_mse": predicted_terminal_mse,
        "validate/stage_b_expert_preference_accuracy": (
            expert_cost[:, None] < negative_cost
        ).float().mean(),
        "validate/stage_b_expert_top1_rate": (
            expert_cost < negative_cost.min(dim=1).values
        ).float().mean(),
        "validate/stage_b_expert_negative_margin": (
            negative_cost - expert_cost[:, None]
        ).mean(),
    }


class FastLeWAMValidationDiagnosticsCallback(Callback):
    """在少量 held-out batch 上记录 Stage B endpoint 与排序诊断。"""

    def __init__(self, max_batches=8, noise_std=0.1, seed=42):
        super().__init__()
        if max_batches < 1:
            raise ValueError("max_batches must be positive")
        if noise_std <= 0:
            raise ValueError("noise_std must be positive")
        self.max_batches = int(max_batches)
        self.noise_std = float(noise_std)
        self.seed = int(seed)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        """复用 validation forward 输出，并限制每个 epoch 的诊断 batch 数。"""
        if trainer.sanity_checking or batch_idx >= self.max_batches:
            return
        if getattr(pl_module, "train_mode", None) != "stage_ab":
            return
        precision_plugin = getattr(trainer, "precision_plugin", None)
        forward_context = (
            precision_plugin.forward_context()
            if precision_plugin is not None
            else nullcontext()
        )
        with forward_context:
            metrics = compute_stage_b_validation_metrics(
                model=pl_module.model,
                batch=batch,
                forward_output=outputs,
                noise_std=self.noise_std,
                seed=self.seed + int(batch_idx),
            )
        pl_module.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["action"].shape[0],
        )


__all__ = [
    "FastLeWAMValidationDiagnosticsCallback",
    "compute_stage_b_validation_metrics",
]
