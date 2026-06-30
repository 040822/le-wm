from functools import partial

import hydra
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

from source.model.value_jepa.losses import value_td_loss


def _to_container(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def build_lewm_optim(optimizer, scheduler=None, interval="epoch"):
    optimizer_cfg = _to_container(optimizer)
    scheduler_cfg = _to_container(scheduler) or {"type": "LinearWarmupCosineAnnealingLR"}
    return {
        "model_opt": {
            "modules": "model",
            "optimizer": optimizer_cfg,
            "scheduler": scheduler_cfg,
            "interval": interval,
        },
    }


def _window_value_loss(emb, gamma, tau):
    """Compute ValueJEPA expectile TD loss over goals sampled inside the batch window."""
    batch, length, dimension = emb.shape
    if length < 2:
        raise ValueError("value loss needs at least two encoded steps")

    transitions = length - 1
    current = emb[:, :-1]
    following = emb[:, 1:]

    terminal_goal = emb[:, -1:].expand(batch, transitions, dimension)
    terminal_reward = -torch.ones(
        (batch, transitions), dtype=emb.dtype, device=emb.device
    )

    permutation = torch.randperm(batch, device=emb.device)
    random_times = torch.randint(length, (batch,), device=emb.device)
    random_goal = emb[permutation, random_times]
    random_goal = random_goal[:, None, :].expand(batch, transitions, dimension)
    random_reward = -torch.ones_like(terminal_reward)

    same_trajectory = permutation == torch.arange(batch, device=emb.device)
    for batch_index in torch.nonzero(same_trajectory, as_tuple=False).flatten():
        time = int(random_times[batch_index])
        if time < transitions:
            random_reward[batch_index, time] = 0.0

    terminal_result = value_td_loss(
        current=current.reshape(-1, dimension),
        following=following.reshape(-1, dimension),
        goal=terminal_goal.reshape(-1, dimension),
        reward=terminal_reward.reshape(-1),
        gamma=gamma,
        tau=tau,
    )
    random_result = value_td_loss(
        current=current.reshape(-1, dimension),
        following=following.reshape(-1, dimension),
        goal=random_goal.reshape(-1, dimension),
        reward=random_reward.reshape(-1),
        gamma=gamma,
        tau=tau,
    )

    loss = 0.5 * (terminal_result.loss + random_result.loss)
    td_abs = 0.5 * (
        terminal_result.td_error.abs().mean() + random_result.td_error.abs().mean()
    )
    return loss, {
        "value_terminal": terminal_result.value.mean(),
        "value_random": random_result.value.mean(),
        "td_abs": td_abs,
    }


def value_jepa_forward(
    self,
    batch,
    stage,
    history_size,
    num_preds,
    sigreg_weight,
    value_weight,
    value_gamma,
    value_tau,
):
    """Encode observations, predict next states, and add ValueJEPA TD loss."""
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :history_size]
    ctx_act = act_emb[:, :history_size]

    tgt_emb = emb[:, num_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + sigreg_weight * output["sigreg_loss"]

    if value_weight:
        output["value_loss"], value_metrics = _window_value_loss(
            emb=emb,
            gamma=value_gamma,
            tau=value_tau,
        )
        output.update(value_metrics)
        output["loss"] = output["loss"] + value_weight * output["value_loss"]

    log_keys = {"loss", "pred_loss", "sigreg_loss", "value_loss", "td_abs"}
    log_keys.update({"value_terminal", "value_random"})
    metrics = {
        f"{stage}/{key}": value.detach()
        for key, value in output.items()
        if key in log_keys and torch.is_tensor(value)
    }
    self.log_dict(metrics, on_step=True, sync_dist=True)
    return output


def make_world_policy_from_model(
    model,
    solver_cfg,
    plan_config,
    process,
    transform,
    device="cuda",
):
    """Wrap a bare LeWM dynamics model as a stable-worldmodel environment policy."""
    if device is not None:
        model = model.to(device)
    model = model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    plan_kwargs = _to_container(plan_config)
    config = (
        plan_config
        if isinstance(plan_config, swm.PlanConfig)
        else swm.PlanConfig(**plan_kwargs)
    )
    solver = hydra.utils.instantiate(solver_cfg, model=model)
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
    )


def make_world_policy(policy_or_model, *args, **kwargs):
    """Build a stable-worldmodel policy from either a policy wrapper or bare JEPA model."""
    if hasattr(policy_or_model, "make_world_policy"):
        return policy_or_model.make_world_policy(*args, **kwargs)
    return make_world_policy_from_model(policy_or_model, *args, **kwargs)


class ValueJEPALeWMPolicy(spt.Module):
    """LeWM training policy with an additional ValueJEPA expectile TD loss."""

    def __init__(
        self,
        model,
        sigreg,
        history_size,
        num_preds,
        sigreg_weight,
        optimizer,
        value_weight=0.1,
        value_gamma=0.98,
        value_tau=0.8,
        scheduler=None,
        optim_interval="epoch",
    ):
        optim = build_lewm_optim(
            optimizer=optimizer,
            scheduler=scheduler,
            interval=optim_interval,
        )
        forward = partial(
            value_jepa_forward,
            history_size=history_size,
            num_preds=num_preds,
            sigreg_weight=sigreg_weight,
            value_weight=value_weight,
            value_gamma=value_gamma,
            value_tau=value_tau,
        )
        super().__init__(
            model=model,
            sigreg=sigreg,
            forward=forward,
            optim=optim,
        )
        self.history_size = history_size
        self.num_preds = num_preds
        self.sigreg_weight = sigreg_weight
        self.value_weight = value_weight
        self.value_gamma = value_gamma
        self.value_tau = value_tau

    def make_world_policy(
        self,
        solver_cfg,
        plan_config,
        process,
        transform,
        device="cuda",
    ):
        return make_world_policy_from_model(
            self.model,
            solver_cfg=solver_cfg,
            plan_config=plan_config,
            process=process,
            transform=transform,
            device=device,
        )


LeWMPolicy = ValueJEPALeWMPolicy
