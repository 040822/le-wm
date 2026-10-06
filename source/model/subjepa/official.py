"""Load converted official Sub-JEPA weights without importing legacy pickles.

Sub-JEPA changes LeWM's training regularizer; its inference model and CEM
cost are the same. Conversion preserves that model and records the published
training configuration instead of creating an unused inference regularizer.
"""

from pathlib import Path
import re

import torch
from torch import nn

from source.model.lewm.jepa import JEPA
from source.model.lewm.modules import ARPredictor, Embedder, MLP


FORMAT = "subjepa_official_v1"


def infer_model_config(state, training_config, encoder_config):
    """Recover dimensions from the actual weights and check published metadata."""
    latent_dim = int(state["projector.net.3.weight"].shape[0])
    history_size = int(state["predictor.pos_embedding"].shape[1])
    action_weight = state["action_encoder.patch_embed.weight"]
    predictor = dict(training_config["predictor"])
    layer_ids = {
        int(match.group(1))
        for key in state
        if (match := re.match(r"predictor\.transformer\.layers\.(\d+)\.", key))
    }
    if layer_ids != set(range(int(predictor["depth"]))):
        raise ValueError("Predictor depth differs from the official training config")
    wm = training_config["wm"]
    frameskip = int(training_config["data"]["dataset"]["frameskip"])
    if (latent_dim, history_size, int(action_weight.shape[1])) != (
        int(wm["embed_dim"]), int(wm["history_size"]), frameskip * int(wm["action_dim"])
    ):
        raise ValueError("Official training dimensions do not match checkpoint weights")
    if "projector.net.1.running_mean" not in state or "pred_proj.net.1.running_mean" not in state:
        raise ValueError("Expected the official BatchNorm projection heads")
    action_hidden = int(state["action_encoder.embed.0.weight"].shape[0])
    if action_hidden % latent_dim:
        raise ValueError("Action embedding hidden dimension must be divisible by latent_dim")
    return {
        "encoder": encoder_config,
        "latent_dim": latent_dim,
        "history_size": history_size,
        "action_input_dim": int(action_weight.shape[1]),
        "action_smoothed_dim": int(action_weight.shape[0]),
        "action_mlp_scale": action_hidden // latent_dim,
        "projector_hidden_dim": int(state["projector.net.0.weight"].shape[0]),
        "pred_proj_hidden_dim": int(state["pred_proj.net.0.weight"].shape[0]),
        "predictor": predictor,
    }


class OfficialSubJEPA(JEPA):
    """Published LeWM-shaped Sub-JEPA model with explicit method provenance."""

    policy_kind = "subjepa_official"

    def __init__(self, model_config, provenance=None):
        from transformers import ViTConfig, ViTModel

        config = ViTConfig(**model_config["encoder"])
        # Official objects contain the eager ViT attention implementation.
        config._attn_implementation = "eager"
        encoder = ViTModel(config, add_pooling_layer=False, use_mask_token=False)
        dim = int(model_config["latent_dim"])
        super().__init__(
            encoder=encoder,
            predictor=ARPredictor(
                num_frames=int(model_config["history_size"]),
                input_dim=dim, hidden_dim=dim, output_dim=dim,
                **model_config["predictor"],
            ),
            action_encoder=Embedder(
                input_dim=int(model_config["action_input_dim"]),
                smoothed_dim=int(model_config["action_smoothed_dim"]),
                emb_dim=dim, mlp_scale=int(model_config["action_mlp_scale"]),
            ),
            projector=MLP(
                input_dim=config.hidden_size, output_dim=dim,
                hidden_dim=int(model_config["projector_hidden_dim"]),
                norm_fn=nn.BatchNorm1d,
            ),
            pred_proj=MLP(
                input_dim=dim, output_dim=dim,
                hidden_dim=int(model_config["pred_proj_hidden_dim"]),
                norm_fn=nn.BatchNorm1d,
            ),
        )
        self.history_size = int(model_config["history_size"])
        self.action_input_dim = int(model_config["action_input_dim"])
        self.provenance = dict(provenance or {})

    def rollout(self, info, action_sequence, history_size=None):
        return super().rollout(
            info, action_sequence,
            history_size=self.history_size if history_size is None else history_size,
        )

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, dict) or payload.get("format") != FORMAT:
            raise ValueError("Expected a converted official Sub-JEPA checkpoint")
        model = cls(payload["model_config"], payload["provenance"])
        model.load_state_dict(payload["state_dict"], strict=True)
        return model.eval().requires_grad_(False)


def load_official_checkpoint(policy_name, cache_dir=None):
    """Try direct and SWM-cache paths; leave other checkpoint formats alone."""
    raw = Path(policy_name).expanduser()
    if raw.suffix != ".pt":
        return None
    candidates = [raw]
    if not raw.is_absolute() and cache_dir is not None:
        candidates.extend([Path(cache_dir) / "checkpoints" / raw, Path(cache_dir) / raw])
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and payload.get("format") == FORMAT:
            return OfficialSubJEPA.from_payload(payload), path.resolve()
    return None


def conversion_checks(model, fixture, expected, *, atol=2e-5, rtol=2e-5):
    """Check encoder, action embedding, dynamics, rollout and candidate ranking."""
    with torch.no_grad():
        encoded = model.encode({"pixels": fixture["pixels"], "action": fixture["action"]})
        prediction = model.predict(encoded["emb"], encoded["act_emb"])
        info = {key: value.clone() for key, value in fixture["cost_info"].items()}
        cost = model.get_cost(info, fixture["candidates"])
    actual = {
        "emb": encoded["emb"], "act_emb": encoded["act_emb"],
        "prediction": prediction, "rollout": info["predicted_emb"], "cost": cost,
    }
    errors = {}
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], atol=atol, rtol=rtol)
        errors[name] = float((value - expected[name]).abs().max())
    if not torch.equal(cost.argmin(dim=1), expected["cost"].argmin(dim=1)):
        raise AssertionError("Conversion changed candidate selection")
    return {"atol": atol, "rtol": rtol, "max_absolute_error": errors, "ranking_equal": True}
