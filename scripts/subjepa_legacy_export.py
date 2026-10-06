#!/usr/bin/env python3
"""Conversion worker run in an isolated legacy Transformers process."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # This path is a hash-verified, pinned upstream source snapshot, not DeWM.
    sys.path.insert(0, str(args.source_dir.resolve()))
    import torch
    import transformers

    if transformers.__version__ != "4.57.1":
        raise RuntimeError("Legacy export requires isolated transformers==4.57.1")
    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model.eval().requires_grad_(False)
    import jepa

    if type(model) is not jepa.JEPA:
        raise TypeError("Official checkpoint is not the pinned upstream JEPA class")
    generator = torch.Generator().manual_seed(3072)
    action_dim = int(model.action_encoder.patch_embed.in_channels)
    pixels = torch.randn(1, 3, 3, 224, 224, generator=generator)
    action = torch.randn(1, 3, action_dim, generator=generator)
    candidates = torch.randn(1, 3, 5, action_dim, generator=generator)
    current = pixels[:, :1].unsqueeze(1).expand(-1, 3, -1, -1, -1, -1).clone()
    goal = pixels[:, 1:2].unsqueeze(1).expand_as(current).clone()
    fixture = {
        "pixels": pixels, "action": action, "candidates": candidates,
        "cost_info": {"pixels": current, "goal": goal, "action": candidates[:, :, :1].clone()},
    }
    with torch.no_grad():
        encoded = model.encode({"pixels": pixels, "action": action})
        prediction = model.predict(encoded["emb"], encoded["act_emb"])
        info = {key: value.clone() for key, value in fixture["cost_info"].items()}
        cost = model.get_cost(info, candidates)
    expected = {
        "emb": encoded["emb"], "act_emb": encoded["act_emb"],
        "prediction": prediction, "rollout": info["predicted_emb"], "cost": cost,
    }
    encoder_config = json.loads(json.dumps(model.encoder.config.to_dict()))
    torch.save({
        "state_dict": model.state_dict(), "encoder_config": encoder_config,
        "fixture": fixture, "expected": expected,
        "reference_transformers": transformers.__version__,
    }, args.output)
    print(f"Exported upstream reference: {args.checkpoint.name}", flush=True)


if __name__ == "__main__":
    main()
