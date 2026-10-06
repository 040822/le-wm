#!/usr/bin/env python3
"""CPU/small-batch checks for the Round 5 Phase 6.1 latent-flow path."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from source.common.checkpoint import load_policy_or_model
from source.model.fast_lewam.round4 import Round4FastLeWAM
from source.policy.fast_lewam import fast_lewam_forward


class TinyEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.proj = nn.Linear(3, latent_dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        pooled = pixels.mean(dim=(-2, -1))
        return SimpleNamespace(last_hidden_state=self.proj(pooled)[:, None])


def _model(*, flow_matching: bool, seed: int = 13) -> Round4FastLeWAM:
    torch.manual_seed(seed)
    return Round4FastLeWAM(
        encoder=TinyEncoder(8),
        projector=nn.Identity(),
        latent_dim=8,
        action_dim=4,
        action_horizon=5,
        latent_head_dim=16,
        latent_head_layers=2,
        heads=4,
        mlp_dim=32,
        stage_a_goal_injection="token",
        activation_checkpointing=False,
        latent_flow_matching=flow_matching,
        latent_flow_steps=2,
    )


class _TrainingProbe:
    def __init__(self, model):
        self.model = model
        self.sigreg = lambda values: values.square().mean() * 0.01
        self.current_epoch = 0
        self.rng_seed = 3072
        self._fast_lewam_rng_streams = {}

    def log_dict(self, *args, **kwargs):
        del args, kwargs


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(output: Path) -> dict:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(3072)
    batch_size, horizon, latent_dim, action_dim = 2, 5, 8, 4
    regression = _model(flow_matching=False)
    regression_clone = _model(flow_matching=False, seed=99)
    regression_clone.load_state_dict(regression.state_dict(), strict=True)
    assert not regression_clone.latent_flow_matching

    model = _model(flow_matching=True).eval()
    shared_initialization_equal = all(
        torch.equal(value, model.state_dict()[name])
        for name, value in regression.state_dict().items()
    )
    assert shared_initialization_equal
    z0 = torch.randn(batch_size, latent_dim)
    actions = torch.randn(batch_size, horizon, action_dim, requires_grad=True)
    source_timestep = torch.full((batch_size,), 0.4)
    epsilon = torch.randn(batch_size, horizon, latent_dim)
    tau = torch.full((batch_size,), 0.3)
    target = torch.randn_like(epsilon)
    noisy = (1.0 - tau[:, None, None]) * epsilon + tau[:, None, None] * target

    # A future noisy latent cannot affect any earlier query under the strict mask.
    with torch.no_grad():
        velocity = model.predict_future_latent_velocity(
            z0, actions, source_timestep, noisy, tau
        )
        perturbed = noisy.clone()
        perturbed[:, 3:] += 100.0
        velocity_perturbed = model.predict_future_latent_velocity(
            z0, actions, source_timestep, perturbed, tau
        )
        torch.testing.assert_close(velocity[:, :3], velocity_perturbed[:, :3])

    # Inspect the AdaLN condition directly. Source time and latent flow time are
    # independently encoded and added; the goal latent is absent from this API.
    conditions: list[torch.Tensor] = []
    hook = model.predictor.register_forward_pre_hook(
        lambda _module, inputs: conditions.append(inputs[1].detach().clone())
    )

    def condition(source, flow):
        conditions.clear()
        model.predict_future_latent_velocity(z0, actions, source, noisy, flow)
        return conditions[-1]

    flow_late = torch.full((batch_size,), 0.8)
    source_late = torch.full((batch_size,), 0.9)
    condition_00 = condition(source_timestep, tau)
    condition_01 = condition(source_timestep, flow_late)
    condition_10 = condition(source_late, tau)
    condition_11 = condition(source_late, flow_late)
    hook.remove()
    assert not torch.allclose(condition_00, condition_01)
    assert not torch.allclose(condition_00, condition_10)
    torch.testing.assert_close(
        condition_00 - condition_01,
        condition_10 - condition_11,
        rtol=2e-5,
        atol=2e-5,
    )
    assert "goal" not in inspect.signature(model.predict_future_latent_velocity).parameters

    # Euler integration executes K actual model forwards and remains differentiable.
    integrated = model.sample_future_latents(
        z0,
        actions,
        source_timestep,
        initial_noise=epsilon,
        num_steps=2,
    )
    assert integrated.shape == (batch_size, horizon, latent_dim)
    assert model.last_latent_flow_forward_count == 2
    action_gradient = torch.autograd.grad(integrated.square().mean(), actions)[0]
    assert torch.isfinite(action_gradient).all() and action_gradient.abs().sum() > 0

    # Run the actual joint-training loss on a synthetic B=2 minibatch. The FM
    # objective must carry gradients through B -> clean A actions and targets.
    pixels = torch.rand(batch_size, horizon + 1, 3, 8, 8)
    recorded_actions = torch.rand(batch_size, horizon, action_dim)
    fm_probe = _TrainingProbe(_model(flow_matching=True))
    fm_out = fast_lewam_forward(
        fm_probe,
        {"pixels": pixels, "action": recorded_actions},
        "validate",
        horizon,
        "stage_ab_fm",
        1.0,
        0.09,
        False,
        0.2,
        10,
        "legacy",
        "joint",
    )
    assert fm_out["predicted_latent_velocity"].shape == (
        batch_size,
        horizon,
        latent_dim,
    )
    assert "latent_flow_endpoint_mse" in fm_out
    fm_action_gradient = torch.autograd.grad(
        fm_out["weighted_latent_flow_velocity_loss"],
        fm_out["clean_action"],
        retain_graph=True,
    )[0]
    assert torch.isfinite(fm_action_gradient).all()
    assert fm_action_gradient.abs().sum() > 0
    fm_out["emb"].retain_grad()
    fm_out["loss"].backward()
    assert fm_probe.model.encoder.proj.weight.grad is not None
    assert fm_out["emb"].grad is not None
    assert fm_out["emb"].grad[:, 1:horizon].abs().sum() > 0
    assert fm_probe.model.action_head.weight.grad is not None

    # The old regression objective still leaves future encoder targets attached.
    regression_probe = _TrainingProbe(_model(flow_matching=False))
    regression_out = fast_lewam_forward(
        regression_probe,
        {"pixels": pixels, "action": recorded_actions},
        "validate",
        horizon,
        "stage_ab",
        1.0,
        0.09,
        False,
        0.2,
        10,
        "legacy",
        "joint",
    )
    regression_out["emb"].retain_grad()
    regression_out["loss"].backward()
    assert regression_probe.model.encoder.proj.weight.grad is not None
    assert regression_out["emb"].grad is not None
    assert regression_out["emb"].grad[:, 1:horizon].abs().sum() > 0

    checkpoints = {}
    for task in ("pusht", "reacher"):
        checkpoint = (
            ROOT
            / "outputs"
            / "round4"
            / f"ab_seed3072_{task}"
            / "checkpoints"
            / f"r4_ab_seed3072_weights_epoch_10.pt"
        )
        loaded_model, resolved = load_policy_or_model(str(checkpoint))
        assert resolved is not None and Path(resolved).resolve() == checkpoint.resolve()
        assert not loaded_model.latent_flow_matching
        checkpoints[task] = {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": _sha256_file(checkpoint),
            "strict_load": True,
            "latent_flow_matching": False,
        }
        del loaded_model
        gc.collect()

    report = {
        "schema_version": 1,
        "status": "passed",
        "device": "cpu",
        "batch_size": batch_size,
        "checks": {
            "legacy_model_schema_strict_load": True,
            "shared_parameters_same_seed_match_before_training": shared_initialization_equal,
            "future_noisy_latent_prefix_causality": True,
            "source_and_flow_time_separate_conditioning": True,
            "goal_latent_absent_from_B_velocity_interface": True,
            "euler_steps_actual_forward_count": 2,
            "gradient_through_latent_integrator_to_actions": True,
            "FM_training_backward_to_stage_A_actions": True,
            "FM_training_backward_to_encoder_and_future_targets": True,
            "regression_training_target_gradient_unchanged": True,
            "FM_validation_integrated_endpoint_mse": float(
                fm_out["latent_flow_endpoint_mse"].detach()
            ),
        },
        "historical_checkpoints": checkpoints,
        "smoke_model_new_flow_parameter_count": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith(
                ("latent_flow_input.", "latent_flow_time_mlp.", "latent_flow_velocity_head.")
            )
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/round5/phase6_1/smoke/cpu_smoke.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
