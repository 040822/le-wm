#!/usr/bin/env python3
"""Run the frozen Round 4 offline IDM/path diagnostics for one task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import default_collate

from source.common.checkpoint import torch_load_compat
from source.common.data import load_dataset
from source.common.round4_diagnostics import compute_idm_diagnostics
from source.common.round4_eval import validate_gpu_visibility


DATASETS = {
    "cube": "ogbench/cube_single.h5",
    "pusht": "pusht.h5",
    "reacher": "dmcontrol/reacher.h5",
    "tworoom": "tworoom.h5",
}


def _gpu_argument(value: str | None) -> str:
    if value is None or not value.isdigit() or int(value) not in range(4):
        raise argparse.ArgumentTypeError("--gpu must be one of physical GPU0-3")
    return value


def _action_normalizer(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.from_numpy(np.asarray(dataset.get_col_data("action"))).float()
    values = values[~torch.isnan(values).any(dim=1)]
    mean = values.mean(dim=0)
    std = values.std(dim=0).clamp_min(1e-6)
    return mean, std


def run(args: argparse.Namespace) -> dict:
    if args.device.startswith("cuda"):
        validate_gpu_visibility(args.device)
    device = torch.device(args.device)
    dataset = load_dataset(
        DATASETS[args.task],
        transform=None,
        num_steps=args.horizon + 1,
        frameskip=5,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    )
    count = min(int(args.batch_size), len(dataset))
    indices = random.Random(args.seed).sample(range(len(dataset)), count)
    batch = default_collate([dataset[index] for index in indices])
    action_mean, action_std = _action_normalizer(dataset)
    action_dim = int(batch["action"].shape[-1])
    if action_dim != action_mean.numel():
        if action_dim % action_mean.numel() != 0:
            raise ValueError(
                f"action normalization shape mismatch: sample={action_dim}, "
                f"statistics={action_mean.numel()}"
            )
        repeats = action_dim // action_mean.numel()
        action_mean = action_mean.repeat(repeats)
        action_std = action_std.repeat(repeats)
    pixels = batch["pixels"].to(device)
    actions = ((batch["action"].float() - action_mean) / action_std).to(device)
    policy = torch_load_compat(args.checkpoint, map_location="cpu")
    model = policy.model.to(device).eval()
    with torch.no_grad():
        real_paths = model.encode_pixels(pixels)
        real_actions = actions[:, : args.horizon]
        z_start = real_paths[:, 0]
        z_goal = real_paths[:, -1]
        generator = torch.Generator(device=device).manual_seed(args.seed)
        generated = model.sample_latent_paths(
            z_start,
            z_goal,
            num_samples=args.generated_samples,
            num_steps=args.flow_steps,
            generator=generator,
        )
        generated_paths = generated.reshape(
            count * args.generated_samples,
            args.horizon + 1,
            model.latent_dim,
        )
        repeated_start = z_start[:, None].expand(
            count, args.generated_samples, -1
        ).reshape(count * args.generated_samples, -1)
        repeated_goal = z_goal[:, None].expand(
            count, args.generated_samples, -1
        ).reshape(count * args.generated_samples, -1)
        diagnostics = compute_idm_diagnostics(
            model,
            real_paths.repeat_interleave(args.generated_samples, dim=0),
            real_actions.repeat_interleave(args.generated_samples, dim=0),
            z_start=repeated_start,
            z_goal=repeated_goal,
            generated_paths=generated_paths,
        )
    return {
        "task": args.task,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset": DATASETS[args.task],
        "seed": int(args.seed),
        "batch_size": count,
        "generated_samples_per_context": int(args.generated_samples),
        "flow_steps": int(args.flow_steps),
        "action_normalization": "full_dataset_mean_std",
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=tuple(DATASETS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gpu", type=_gpu_argument)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generated-samples", type=int, default=8)
    parser.add_argument("--flow-steps", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()
    if args.batch_size < 1 or args.generated_samples < 1 or args.flow_steps < 1:
        parser.error("batch/sample/flow counts must be positive")
    payload = run(args)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
