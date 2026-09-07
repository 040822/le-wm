"""Fast-LeWAM 的训练损失入口和 Lightning policy 封装。"""

from functools import partial

import stable_pretraining as spt
import torch
import torch.nn.functional as F

from source.policy.lewm import build_lewm_optim


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
        # Terminal supervision is deliberately [z_H] only.  In particular, do
        # not expand it over the prefix: doing so changes the objective and can
        # silently train every query against the terminal target.
        return embeddings[:, -1:]
    return embeddings[:, 1 : action_horizon + 1]


def _validate_stage_b_prediction_shape(
    predicted_latents, *, batch_size, target_latents, action_horizon, attention_mode
):
    """Validate the output length contract before computing a latent loss."""
    if predicted_latents.ndim != 3:
        raise ValueError(
            "predicted_latents must have shape [B,L,D], "
            f"got {tuple(predicted_latents.shape)}"
        )
    expected_length = 1 if attention_mode == "terminal_full" else action_horizon
    expected_shape = (batch_size, expected_length, target_latents.shape[-1])
    if tuple(predicted_latents.shape) != expected_shape:
        raise ValueError(
            "predicted_latents shape does not match the Stage-B attention mode; "
            f"expected {expected_shape}, got {tuple(predicted_latents.shape)}"
        )


def _predicted_action_probability(module, stage, mix_epochs):
    """
        计算 Stage B 使用 Stage A 预测动作的课程学习概率。
    """
    # 最大概率限制为 0.5，保证训练后期仍有一半样本使用真值动作进行 latent 监督，避免 Stage A 预测动作的偏差累积。
    max_probability = 0.5

    if stage != "fit" or mix_epochs <= 0:
        return 1.0
    return min(float(module.current_epoch) / float(mix_epochs), max_probability)


def fast_lewam_forward(
    self,
    batch,
    stage,
    action_horizon,
    train_mode,
    lambda_latent,
    lambda_sigreg,
    detach_clean_action,
    latent_loss_noise_threshold,
    latent_action_mix_epochs,
    stage_b_timestep_mode="legacy",
):
    """执行 Fast-LeWAM 前向，并组合 flow、causal-prefix latent 与 SIGReg 损失。"""
    if train_mode not in {"stage_ab", "stage_b", "stage_c"}:
        raise ValueError("train_mode must be 'stage_ab', 'stage_b', or 'stage_c'")
    if not 0 < latent_loss_noise_threshold <= 1:
        raise ValueError("latent_loss_noise_threshold must be in (0, 1]")
    if stage_b_timestep_mode not in {"legacy", "clean_action"}:
        raise ValueError(
            "stage_b_timestep_mode must be 'legacy' or 'clean_action'"
        )
    stage_b_attention_mode = _stage_b_attention_mode(self.model)
    stage_b_dynamics = getattr(self.model, "stage_b_dynamics", "parallel_prefix")
    if stage_b_attention_mode != "strict_causal" and not (
        train_mode == "stage_b" and stage_b_dynamics == "parallel_prefix"
    ):
        raise ValueError(
            "non-strict Stage-B attention requires train_mode='stage_b' and "
            "stage_b_dynamics='parallel_prefix'"
        )

    pixels = batch["pixels"]
    actions = torch.nan_to_num(batch["action"], 0.0)
    if pixels.ndim != 5 or pixels.shape[1] < action_horizon + 1:
        raise ValueError(
            "pixels must have shape [B,T,C,H,W] with T >= action_horizon + 1; "
            f"got {tuple(pixels.shape)}"
        )
    if actions.ndim != 3 or actions.shape[1] < action_horizon:
        raise ValueError(
            "action must have shape [B,T,A] with T >= action_horizon; "
            f"got {tuple(actions.shape)}"
        )

    # 一次共享编码得到当前 z0 和未来监督目标 z1:H，避免重复调用 Encoder。
    embeddings = self.model.encode_pixels(pixels[:, : action_horizon + 1])
    z0 = embeddings[:, 0]
    goal_latent = embeddings[:, action_horizon]
    target_latents = _stage_b_target_latents(
        embeddings, action_horizon, stage_b_attention_mode
    )
    clean_actions = actions[:, :action_horizon] # a0:H-1
    expected_action_shape = (
        pixels.shape[0],
        action_horizon,
        self.model.action_dim,
    )# (B, H, dim)
    
    if tuple(clean_actions.shape) != expected_action_shape:
        raise ValueError(
            f"training actions must have shape {expected_action_shape}, "
            f"got {tuple(clean_actions.shape)}"
        )

    task_condition = batch.get("task_condition")

    if train_mode == "stage_b":
        # Stage-B-only 只用数据中的 expert actions 学习并行 latent dynamics。
        # 这里前向推理获取 predicted_latents。
        latent_supervision_timestep = torch.ones(
            pixels.shape[0],
            device=clean_actions.device,
            dtype=clean_actions.dtype,
        )
        predicted_latents = self.model.predict_training_latents(
            embeddings[:, :action_horizon],
            clean_actions,
            task_condition=task_condition,
        )
    else:
        # 对于非stage_b模式，使用线性 flow matching 生成噪声动作和目标速度，为 Stage A/C 训练提供监督信号。
        # 线性 flow matching：x_t=(1-t)noise+t*action，目标速度为 action-noise。
        timestep = torch.rand(
            pixels.shape[0], device=clean_actions.device, dtype=clean_actions.dtype
        )
        noise = torch.randn_like(clean_actions)
        noisy_actions = (
            (1.0 - timestep[:, None, None]) * noise
            + timestep[:, None, None] * clean_actions
        )
        target_velocity = clean_actions - noise

    if train_mode == "stage_ab":
        # Stage A 预测动作速度，然后获取 clean_estimate。
        stage_a = self.model(
            z0,
            noisy_actions,
            timestep,
            mode="stage_a",
            task_condition=task_condition,
            goal_latent=goal_latent,
        )
        predicted_velocity = stage_a["action_velocity"]
        clean_estimate = noisy_actions + (
            1.0 - timestep[:, None, None]
        ) * predicted_velocity

        # 构建action for stage B
        # 训练初期混入真值动作，随后逐步过渡到完全使用 Stage A 预测动作。
        predicted_probability = _predicted_action_probability(
            self, stage, latent_action_mix_epochs
        )
        if predicted_probability >= 1.0:
            use_prediction = torch.ones(
                pixels.shape[0], dtype=torch.bool, device=pixels.device
            )
        elif predicted_probability <= 0.0:
            use_prediction = torch.zeros(
                pixels.shape[0], dtype=torch.bool, device=pixels.device
            )
        else:
            use_prediction = (
                torch.rand(pixels.shape[0], device=pixels.device)
                < predicted_probability
            )
        predicted_for_stage_b = (
            clean_estimate.detach() if detach_clean_action else clean_estimate
        )
        stage_b_actions = torch.where(
            use_prediction[:, None, None], predicted_for_stage_b, clean_actions
        )

        # source_t 记录 clean estimate 的 flow 来源；专家动作按 clean t=1 处理。
        source_timestep = torch.where(
            use_prediction,
            timestep,
            torch.ones_like(timestep),
        )
        stage_b_timestep = (
            torch.ones_like(timestep)
            if stage_b_timestep_mode == "clean_action"
            else source_timestep
        )
        latent_supervision_timestep = source_timestep
        predicted_latents = self.model(
            z0,
            stage_b_actions,
            stage_b_timestep,
            mode="stage_b",
            task_condition=task_condition,
        )["predicted_latents"]
    elif train_mode == "stage_c":
        # Stage C 直接同时预测动作速度和未来 latent，使用真值动作进行监督。
        joint = self.model(
            z0,
            noisy_actions,
            timestep,
            mode="stage_c",
            task_condition=task_condition,
        )
        predicted_velocity = joint["action_velocity"]
        predicted_latents = joint["predicted_latents"]
        clean_estimate = noisy_actions + (
            1.0 - timestep[:, None, None]
        ) * predicted_velocity
        latent_supervision_timestep = timestep
        predicted_probability = 1.0

    _validate_stage_b_prediction_shape(
        predicted_latents,
        batch_size=pixels.shape[0],
        target_latents=target_latents,
        action_horizon=action_horizon,
        attention_mode=stage_b_attention_mode,
    )

    # 计算 latent supervision loss，使用线性权重降低高噪声样本的监督强度。
    latent_per_sample = (predicted_latents - target_latents).square().mean(dim=(1, 2))
    
    # t 越小代表噪声越强；线性权重会降低高噪声样本的 latent 监督强度。
    noise_weight = (
        latent_supervision_timestep / latent_loss_noise_threshold
    ).clamp(max=1.0)
    weighted_latent_loss = (noise_weight * latent_per_sample).mean()
    latent_prefix_loss = latent_per_sample.mean() # 用于日志记录的 unweighted loss
    sigreg_loss = self.sigreg(embeddings.transpose(0, 1))
    
    # WM loss = lambda_latent * weighted_latent_loss + lambda_sigreg * sigreg_loss
    loss = lambda_latent * weighted_latent_loss + lambda_sigreg * sigreg_loss

    output = {
        "loss": loss,
        "latent_prefix_loss": latent_prefix_loss,
        "weighted_latent_prefix_loss": weighted_latent_loss,
        "sigreg_loss": sigreg_loss,
        "noise_weight": noise_weight.mean(),
        "predicted_latents": predicted_latents,
        "emb": embeddings,
    }
    if stage_b_attention_mode == "terminal_full":
        # Keep the long-standing prefix names as aliases for consumers that
        # aggregate the common latent metrics, while exposing unambiguous
        # terminal names for the new objective.
        output.update(
            latent_terminal_loss=latent_prefix_loss,
            weighted_latent_terminal_loss=weighted_latent_loss,
        )
    if train_mode != "stage_b":
        # action loss
        action_loss = F.mse_loss(predicted_velocity, target_velocity)
        loss = loss + action_loss
        output.update(
            loss=loss,
            action_loss=action_loss,
            predicted_action_probability=loss.new_tensor(predicted_probability),
            action_velocity=predicted_velocity,
            clean_action=clean_estimate,
        )
    metric_names = {
        "loss",
        "action_loss",
        "latent_prefix_loss",
        "weighted_latent_prefix_loss",
        "latent_terminal_loss",
        "weighted_latent_terminal_loss",
        "sigreg_loss",
        "noise_weight",
        "predicted_action_probability",
    }
    metrics = {
        f"{stage}/{name}": value.detach()
        for name, value in output.items()
        if name in metric_names
    }
    self.log_dict(metrics, on_step=True, sync_dist=True)
    return output


class FastLeWAMPolicy(spt.Module):
    """把 FastLeWAM 模型、SIGReg、损失入口与优化器组装成训练模块。"""

    def __init__(
        self,
        model,
        sigreg,
        action_horizon,
        optimizer,
        train_mode="stage_ab",
        lambda_latent=1.0,
        lambda_sigreg=0.09,
        detach_clean_action=False,
        latent_loss_noise_threshold=0.2,
        latent_action_mix_epochs=20,
        stage_b_timestep_mode="legacy",
        scheduler=None,
        optim_interval="epoch",
    ):
        """绑定训练超参数，并复用 LeWM 的 optimizer/scheduler checkpoint 逻辑。"""
        if train_mode not in {"stage_ab", "stage_b", "stage_c"}:
            raise ValueError(
                "train_mode must be 'stage_ab', 'stage_b', or 'stage_c'"
            )
        if stage_b_timestep_mode not in {"legacy", "clean_action"}:
            raise ValueError(
                "stage_b_timestep_mode must be 'legacy' or 'clean_action'"
            )
        stage_b_attention_mode = _stage_b_attention_mode(model)
        stage_b_dynamics = getattr(model, "stage_b_dynamics", "parallel_prefix")
        if stage_b_attention_mode != "strict_causal" and not (
            train_mode == "stage_b" and stage_b_dynamics == "parallel_prefix"
        ):
            raise ValueError(
                "non-strict Stage-B attention requires train_mode='stage_b' and "
                "stage_b_dynamics='parallel_prefix'"
            )
        if stage_b_dynamics == "serial_one_step" and train_mode != "stage_b":
            raise ValueError(
                "serial_one_step Stage-B dynamics require train_mode='stage_b'"
            )
        if train_mode == "stage_b":
            if model.action_positions is not None:
                model.action_positions.requires_grad_(False)
            model.action_head.requires_grad_(False)
            if model.goal_token_embedding is not None:
                model.goal_token_embedding.requires_grad_(False)
        forward = partial(
            fast_lewam_forward,
            action_horizon=action_horizon,
            train_mode=train_mode,
            lambda_latent=lambda_latent,
            lambda_sigreg=lambda_sigreg,
            detach_clean_action=detach_clean_action,
            latent_loss_noise_threshold=latent_loss_noise_threshold,
            latent_action_mix_epochs=latent_action_mix_epochs,
            stage_b_timestep_mode=stage_b_timestep_mode,
        )
        super().__init__(
            model=model,
            sigreg=sigreg,
            forward=forward,
            optim=build_lewm_optim(
                optimizer=optimizer,
                scheduler=scheduler,
                interval=optim_interval,
            ),
        )
        self.action_horizon = action_horizon
        self.train_mode = train_mode
        self.lambda_latent = lambda_latent
        self.lambda_sigreg = lambda_sigreg
        self.detach_clean_action = detach_clean_action
        self.latent_loss_noise_threshold = latent_loss_noise_threshold
        self.latent_action_mix_epochs = latent_action_mix_epochs
        self.stage_b_timestep_mode = stage_b_timestep_mode


__all__ = ["FastLeWAMPolicy", "fast_lewam_forward"]
