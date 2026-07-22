"""Fast-LeWAM 的训练损失入口和 Lightning policy 封装。"""

from functools import partial

import stable_pretraining as spt
import torch
import torch.nn.functional as F

from source.policy.lewm import build_lewm_optim


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
):
    """执行 Fast-LeWAM 前向，并组合 flow、causal-prefix latent 与 SIGReg 损失。"""
    if train_mode not in {"stage_ab", "stage_b", "stage_c"}:
        raise ValueError("train_mode must be 'stage_ab', 'stage_b', or 'stage_c'")
    if not 0 < latent_loss_noise_threshold <= 1:
        raise ValueError("latent_loss_noise_threshold must be in (0, 1]")

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
    target_latents = embeddings[:, 1 : action_horizon + 1] # z1:H
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
        latent_supervision_timestep = torch.ones(
            pixels.shape[0],
            device=clean_actions.device,
            dtype=clean_actions.dtype,
        )
        predicted_latents = self.model(
            z0,
            clean_actions,
            latent_supervision_timestep,
            mode="stage_b",
            task_condition=task_condition,
        )["predicted_latents"]
    else:
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
        # Stage A 预测动作速度，Stage B 使用 Stage A 预测动作或真值动作 进行 latent 监督。
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
        stage_b_timestep = torch.where(
            use_prediction,
            timestep,
            torch.ones_like(timestep),
        )
        latent_supervision_timestep = stage_b_timestep
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

    latent_per_sample = (predicted_latents - target_latents).square().mean(dim=(1, 2))
    # t 越小代表噪声越强；线性权重会降低高噪声样本的 latent 监督强度。
    noise_weight = (
        latent_supervision_timestep / latent_loss_noise_threshold
    ).clamp(max=1.0)
    weighted_latent_loss = (noise_weight * latent_per_sample).mean()
    latent_prefix_loss = latent_per_sample.mean()
    sigreg_loss = self.sigreg(embeddings.transpose(0, 1))
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
    if train_mode != "stage_b":
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
        scheduler=None,
        optim_interval="epoch",
    ):
        """绑定训练超参数，并复用 LeWM 的 optimizer/scheduler checkpoint 逻辑。"""
        if train_mode not in {"stage_ab", "stage_b", "stage_c"}:
            raise ValueError(
                "train_mode must be 'stage_ab', 'stage_b', or 'stage_c'"
            )
        if train_mode == "stage_b":
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


__all__ = ["FastLeWAMPolicy", "fast_lewam_forward"]
