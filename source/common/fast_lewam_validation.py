"""Fast-LeWAM Stage B 的 held-out batch 诊断指标。"""

from contextlib import nullcontext

import torch
from lightning.pytorch.callbacks import Callback


_STAGE_B_ATTENTION_MODES = {"strict_causal", "block_causal", "terminal_full"}


def _stage_b_attention_mode(model):
    """Return the configured Stage-B attention mode with a legacy default."""
    attention_mode = getattr(model, "stage_b_attention_mode", "strict_causal")
    if attention_mode not in _STAGE_B_ATTENTION_MODES:
        raise ValueError(
            "stage_b_attention_mode must be 'strict_causal', 'block_causal', "
            "or 'terminal_full'"
        )
    return attention_mode


def _stage_b_target_latents(embeddings, action_horizon, attention_mode):
    """Select the latent targets matching the model's Stage-B output length."""
    if attention_mode == "terminal_full":
        return embeddings[:, -1:]
    return embeddings[:, 1 : action_horizon + 1]


def _validate_stage_b_prediction_shape(
    predicted_latents,
    *,
    batch_size,
    latent_dim,
    action_horizon,
    attention_mode,
    name,
):
    """Validate Stage-B output length before endpoint diagnostics."""
    if predicted_latents.ndim != 3:
        raise ValueError(
            f"{name} must have shape [B,L,D], got {tuple(predicted_latents.shape)}"
        )
    expected_length = 1 if attention_mode == "terminal_full" else action_horizon
    expected_shape = (batch_size, expected_length, latent_dim)
    if tuple(predicted_latents.shape) != expected_shape:
        raise ValueError(
            f"{name} shape does not match the Stage-B attention mode; "
            f"expected {expected_shape}, got {tuple(predicted_latents.shape)}"
        )


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
    stage_b_attention_mode = _stage_b_attention_mode(model)
    clean_actions = torch.nan_to_num(batch["action"], 0.0)[
        :, : model.action_horizon
    ]
    z0 = embeddings[:, 0]
    target_latents = _stage_b_target_latents(
        embeddings, model.action_horizon, stage_b_attention_mode
    )
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
    _validate_stage_b_prediction_shape(
        predicted_latents,
        batch_size=embeddings.shape[0],
        latent_dim=target_latents.shape[-1],
        action_horizon=model.action_horizon,
        attention_mode=stage_b_attention_mode,
        name="predicted_latents",
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
    )["predicted_latents"]
    _validate_stage_b_prediction_shape(
        candidate_latents,
        batch_size=batch_size * candidate_count,
        latent_dim=target_latents.shape[-1],
        action_horizon=model.action_horizon,
        attention_mode=stage_b_attention_mode,
        name="candidate predicted_latents",
    )
    candidate_output_length = candidate_latents.shape[1]
    candidate_latents = candidate_latents.reshape(
        batch_size,
        candidate_count,
        candidate_output_length,
        target_latents.shape[-1],
    )

    target_terminal = target_latents[:, -1]
    candidate_costs = (
        candidate_latents[:, :, -1] - target_terminal[:, None]
    ).square().mean(dim=-1)
    expert_cost = candidate_costs[:, 0]
    negative_cost = candidate_costs[:, 1:]
    if getattr(model, "stage_b_dynamics", "parallel_prefix") == "serial_one_step":
        rollout_latents = model(
            z0,
            clean_actions,
            torch.ones(
                batch_size,
                device=z0.device,
                dtype=z0.dtype,
            ),
            mode="stage_b",
            task_condition=batch.get("task_condition"),
        )["predicted_latents"]
        _validate_stage_b_prediction_shape(
            rollout_latents,
            batch_size=batch_size,
            latent_dim=target_latents.shape[-1],
            action_horizon=model.action_horizon,
            attention_mode=stage_b_attention_mode,
            name="rollout predicted_latents",
        )
        predicted_terminal = rollout_latents[:, -1]
    else:
        predicted_terminal = predicted_latents[:, -1]
    predicted_terminal_mse = (
        predicted_terminal - target_terminal
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
        if getattr(pl_module, "train_mode", None) not in {"stage_ab", "stage_b"}:
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
