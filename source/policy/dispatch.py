"""Dispatch loaded checkpoints to their environment-facing policy adapter."""

from source.model.fast_lewam.jepa import FastLeWAM
from source.policy.fast_lewam_eval import make_fast_lewam_policy
from source.policy.lewm import make_world_policy as make_lewm_world_policy


def make_world_policy(policy_or_model, *args, mode=None, **kwargs):
    model = getattr(policy_or_model, "model", policy_or_model)
    if isinstance(model, FastLeWAM):
        return make_fast_lewam_policy(
            policy_or_model,
            *args,
            mode=mode or "stage_a",
            **kwargs,
        )
    return make_lewm_world_policy(policy_or_model, *args, **kwargs)


__all__ = ["make_world_policy"]
