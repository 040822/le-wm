"""Load a LeWM checkpoint while remapping HuggingFace transformers 4.x ViT
state_dict keys to the 5.x naming used by the currently installed
`transformers` version.

Only keys under `encoder.encoder.layer.*` (old HF ViT layout) are remapped;
everything else (predictor, action_encoder, projector, pred_proj, and the
encoder embeddings/layernorm) is passed through unchanged.
"""

import re
from typing import Any

import torch
from hydra.utils import instantiate
from loguru import logger as logging

from stable_worldmodel.data import get_cache_dir
from stable_worldmodel.wm.utils import _resolve


_LAYER_RE = re.compile(r"^encoder\.encoder\.layer\.(\d+)\.(.+)$")

_LEGACY_TARGETS = {
    "stable_worldmodel.wm.lewm.LeWM": "source.model.lewm.jepa.JEPA",
    "stable_worldmodel.wm.lewm.module.Predictor": "source.model.lewm.modules.ARPredictor",
    "stable_worldmodel.wm.lewm.module.Embedder": "source.model.lewm.modules.Embedder",
    "stable_worldmodel.wm.lewm.module.MLP": "source.model.lewm.modules.MLP",
    "jepa.JEPA": "source.model.lewm.jepa.JEPA",
    "module.ARPredictor": "source.model.lewm.modules.ARPredictor",
    "module.Embedder": "source.model.lewm.modules.Embedder",
    "module.MLP": "source.model.lewm.modules.MLP",
}


def _remap_key(k: str) -> str:
    """Translate a single old-ViT state_dict key to the new-ViT naming."""
    m = _LAYER_RE.match(k)
    if not m:
        return k
    i, tail = m.group(1), m.group(2)
    # Order matters: replace the more-specific attention path before the
    # generic `output.dense` (which belongs to the MLP block).
    tail = (
        tail.replace("attention.attention.query", "attention.q_proj")
        .replace("attention.attention.key", "attention.k_proj")
        .replace("attention.attention.value", "attention.v_proj")
        .replace("attention.output.dense", "attention.o_proj")
        .replace("intermediate.dense", "mlp.fc1")
        .replace("output.dense", "mlp.fc2")
    )
    return f"encoder.layers.{i}.{tail}"


def remap_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a new state_dict with old-ViT keys renamed to new-ViT keys."""
    return {_remap_key(k): v for k, v in state_dict.items()}


def _remap_config_targets(config: Any) -> Any:
    if isinstance(config, dict):
        remapped = {}
        for key, value in config.items():
            if key == "_target_" and isinstance(value, str):
                remapped[key] = _LEGACY_TARGETS.get(value, value)
            else:
                remapped[key] = _remap_config_targets(value)
        return remapped
    if isinstance(config, list):
        return [_remap_config_targets(value) for value in config]
    return config


def _torch_load_state_dict(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location="cpu")


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected a checkpoint state_dict or a dict with a 'state_dict' entry, "
            f"got {type(checkpoint)!r}."
        )

    state_dict = {
        key: value
        for key, value in checkpoint.items()
        if isinstance(key, str) and torch.is_tensor(value)
    }
    if not state_dict:
        raise TypeError("Checkpoint did not contain tensor state_dict entries.")

    model_state_dict = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    return model_state_dict or state_dict


def load_pretrained_remapped(
    name: str,
    cache_dir: str = None,
    extra_args: dict[str, Any] | None = None,
) -> torch.nn.Module:
    """Load a LeWM checkpoint after remapping legacy ViT state_dict keys."""
    resolved_cache_dir = get_cache_dir(cache_dir, sub_folder="checkpoints")
    checkpoint_path, config = _resolve(name, resolved_cache_dir)
    state_dict = _extract_state_dict(_torch_load_state_dict(checkpoint_path))
    config = _remap_config_targets(config)

    if extra_args is not None:
        for key, value in extra_args.items():
            parts = key.split(".")
            d = config
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value

    model = instantiate(config)

    new_sd = remap_state_dict(state_dict)

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "State_dict remap left mismatches.\n"
            f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}\n"
            f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}"
            f"{'...' if len(unexpected) > 5 else ''}"
        )

    logging.info(f"Loaded checkpoint {checkpoint_path} (remapped keys).")
    return model


__all__ = ["load_pretrained_remapped", "remap_state_dict", "_remap_key"]
