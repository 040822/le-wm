from functools import partial

import hydra
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf


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


def lejepa_forward(self, batch, stage, history_size, num_preds, sigreg_weight):
    """Encode observations, predict next states, and compute LeWM losses."""
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

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
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
    config = plan_config if isinstance(plan_config, swm.PlanConfig) else swm.PlanConfig(**plan_kwargs)
    solver = hydra.utils.instantiate(solver_cfg, model=model)
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
    )


def make_world_policy(policy_or_model, *args, **kwargs):
    """Build a stable-worldmodel policy from either a LeWMPolicy or bare JEPA model."""
    if hasattr(policy_or_model, "make_world_policy"):
        return policy_or_model.make_world_policy(*args, **kwargs)
    return make_world_policy_from_model(policy_or_model, *args, **kwargs)


class LeWMPolicy(spt.Module):
    """Lightning training policy for LeWM, backed by stable-pretraining's Module."""

    def __init__(
        self,
        model,
        sigreg,
        history_size,
        num_preds,
        sigreg_weight,
        optimizer,
        scheduler=None,
        optim_interval="epoch",
    ):
        optim = build_lewm_optim(
            optimizer=optimizer,
            scheduler=scheduler,
            interval=optim_interval,
        )
        forward = partial(
            lejepa_forward,
            history_size=history_size,
            num_preds=num_preds,
            sigreg_weight=sigreg_weight,
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
