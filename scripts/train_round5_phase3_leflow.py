#!/usr/bin/env python3
"""Train the repository's LeFlow planner on one Phase 3 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.common.checkpoint import load_policy_or_model
from source.common.phase3_compat import PHASE3_DATASETS, PHASE3_TASKS
from source.model.leflow.latent_planner import (
    InverseDynamics,
    LatentPathFlow,
    checkpoint_payload,
    flow_matching_loss,
    inverse_dynamics_loss,
    lewm_consistency_loss,
    smoothness_loss,
)


SEED = 3072
FRAMESKIP = 5


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; launch this trainer with an explicit visible GPU")
    return device


def _normalise_pixels(pixels: torch.Tensor, device: torch.device) -> torch.Tensor:
    pixels = pixels.to(device, non_blocking=True)
    if pixels.dtype == torch.uint8:
        pixels = pixels.float().div_(255.0)
        mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
        std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
        pixels = (pixels - mean) / std
    else:
        pixels = pixels.float()
    return pixels


def train(args: argparse.Namespace) -> None:
    task = str(args.task)
    if task not in PHASE3_TASKS:
        raise ValueError(task)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / "latent_planner.pt"
    if final_path.is_file() and not args.force:
        print(json.dumps({"status": "reused", "checkpoint": str(final_path)}, sort_keys=True))
        return

    device = _device(args.device)
    from stable_worldmodel.data.formats.hdf5 import HDF5Dataset

    dataset = HDF5Dataset(
        path=PHASE3_DATASETS[task],
        frameskip=FRAMESKIP,
        num_steps=6,
        keys_to_load=["pixels", "action"],
    )
    split_generator = torch.Generator().manual_seed(int(args.seed))
    train_set, val_set = random_split(
        dataset,
        [float(args.train_split), 1.0 - float(args.train_split)],
        generator=split_generator,
    )
    loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(args.num_workers),
        persistent_workers=bool(args.num_workers),
        prefetch_factor=2 if int(args.num_workers) else None,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=True,
        num_workers=int(args.num_workers),
        persistent_workers=bool(args.num_workers),
        prefetch_factor=2 if int(args.num_workers) else None,
        pin_memory=True,
    )

    # LeWM's action encoder and the planner runtime consume dataset-standardized
    # action blocks.  HDF5 stores each base action row in physical coordinates;
    # compute the same full-dataset z-score used by train.py and repeat it over
    # the five packed frames in each action token.
    action_values = torch.from_numpy(
        np.asarray(dataset.get_col_data("action"), dtype=np.float32)
    )
    finite_rows = torch.isfinite(action_values).all(dim=1)
    if not bool(finite_rows.any()):
        raise ValueError(f"dataset {task} has no finite action rows")
    action_mean_base = action_values[finite_rows].mean(dim=0)
    action_std_base = action_values[finite_rows].std(dim=0).clamp_min(1e-6)
    action_mean = action_mean_base.repeat(FRAMESKIP)
    action_std = action_std_base.repeat(FRAMESKIP)

    lewm, resolved_lewm = load_policy_or_model(str(Path(args.lewm_checkpoint).resolve()))
    if hasattr(lewm, "model"):
        lewm = lewm.model
    lewm = lewm.to(device).eval()
    lewm.requires_grad_(False)
    latent_dim = int(args.embed_dim)
    action_dim = int(dataset.get_dim("action"))
    flow = LatentPathFlow(
        latent_dim=latent_dim,
        hidden_dim=int(args.flow_hidden_dim),
        depth=int(args.flow_depth),
        max_horizon=20,
        time_dim=64,
        dropout=0.0,
    ).to(device)
    inverse = InverseDynamics(
        latent_dim=latent_dim,
        action_dim=action_dim * FRAMESKIP,
        hidden_dim=int(args.inverse_hidden_dim),
        depth=int(args.inverse_depth),
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(flow.parameters()) + list(inverse.parameters()),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    config = {
        "task": task,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "train_split": float(args.train_split),
        "batch_size": int(args.batch_size),
        "action_block": FRAMESKIP,
        "latent_dim": latent_dim,
        "action_dim": action_dim * FRAMESKIP,
        "action_normalizer_mean": action_mean.tolist(),
        "action_normalizer_std": action_std.tolist(),
        "lewm_history_size": 3,
        "flow_hidden_dim": int(args.flow_hidden_dim),
        "flow_depth": int(args.flow_depth),
        "inverse_hidden_dim": int(args.inverse_hidden_dim),
        "inverse_depth": int(args.inverse_depth),
        "flow_weight": float(args.flow_weight),
        "inverse_weight": float(args.inverse_weight),
        "consistency_weight": float(args.consistency_weight),
        "smoothness_weight": float(args.smoothness_weight),
        "lewm_checkpoint": str(resolved_lewm or args.lewm_checkpoint),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    for epoch in range(1, int(args.epochs) + 1):
        flow.train()
        inverse.train()
        running = {"loss": 0.0, "flow": 0.0, "inverse": 0.0, "consistency": 0.0, "smoothness": 0.0, "steps": 0}
        for batch_index, batch in enumerate(loader):
            if args.max_train_batches is not None and batch_index >= int(args.max_train_batches):
                break
            pixels = _normalise_pixels(batch["pixels"], device)
            actions = batch["action"].to(device, non_blocking=True).float()
            actions = (actions - action_mean.to(device)) / action_std.to(device)
            with torch.no_grad():
                z_path = lewm.encode({"pixels": pixels})["emb"].detach()
            flow_loss = flow_matching_loss(flow, z_path, generator=generator)
            inverse_loss, predicted_actions = inverse_dynamics_loss(inverse, z_path, actions[:, :5])
            consistency = lewm_consistency_loss(lewm, z_path, predicted_actions, history_size=3)
            smoothness = smoothness_loss(z_path)
            loss = (
                float(args.flow_weight) * flow_loss
                + float(args.inverse_weight) * inverse_loss
                + float(args.consistency_weight) * consistency
                + float(args.smoothness_weight) * smoothness
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(flow.parameters()) + list(inverse.parameters()), float(args.grad_clip_norm))
            optimizer.step()
            running["loss"] += float(loss.detach())
            running["flow"] += float(flow_loss.detach())
            running["inverse"] += float(inverse_loss.detach())
            running["consistency"] += float(consistency.detach())
            running["smoothness"] += float(smoothness.detach())
            running["steps"] += 1
            if batch_index % 100 == 0:
                print(json.dumps({"epoch": epoch, "batch": batch_index, **{key: value / max(running["steps"], 1) for key, value in running.items() if key != "steps"}}, sort_keys=True), flush=True)

        flow.eval()
        inverse.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch_index, batch in enumerate(val_loader):
                if args.max_val_batches is not None and batch_index >= int(args.max_val_batches):
                    break
                pixels = _normalise_pixels(batch["pixels"], device)
                actions = batch["action"].to(device, non_blocking=True).float()
                actions = (actions - action_mean.to(device)) / action_std.to(device)
                z_path = lewm.encode({"pixels": pixels})["emb"].detach()
                flow_loss = flow_matching_loss(flow, z_path, generator=generator)
                inverse_loss, predicted_actions = inverse_dynamics_loss(inverse, z_path, actions[:, :5])
                consistency = lewm_consistency_loss(lewm, z_path, predicted_actions, history_size=3)
                smoothness = smoothness_loss(z_path)
                val_loss += float((float(args.flow_weight) * flow_loss + float(args.inverse_weight) * inverse_loss + float(args.consistency_weight) * consistency + float(args.smoothness_weight) * smoothness).detach())
                val_steps += 1
        payload = checkpoint_payload(
            lewm_checkpoint=str(resolved_lewm or args.lewm_checkpoint),
            action_block=5,
            flow=flow,
            inverse_dynamics=inverse,
            cfg={**config, "epoch": epoch, "train_loss": running["loss"] / max(running["steps"], 1), "val_loss": val_loss / max(val_steps, 1)},
        )
        epoch_path = output / f"latent_planner_epoch_{epoch}.pt"
        torch.save(payload, epoch_path)
        print(json.dumps({"status": "epoch_complete", "epoch": epoch, "train_loss": running["loss"] / max(running["steps"], 1), "val_loss": val_loss / max(val_steps, 1), "checkpoint": str(epoch_path)}, sort_keys=True), flush=True)

    torch.save(payload, final_path)
    (output / "training_complete.json").write_text(json.dumps({"status": "ok", "epochs": int(args.epochs), "checkpoint": str(final_path), "lewm_checkpoint": str(resolved_lewm or args.lewm_checkpoint)}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "checkpoint": str(final_path)}, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=PHASE3_TASKS)
    parser.add_argument("--lewm-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--flow-hidden-dim", type=int, default=512)
    parser.add_argument("--flow-depth", type=int, default=4)
    parser.add_argument("--inverse-hidden-dim", type=int, default=512)
    parser.add_argument("--inverse-depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--flow-weight", type=float, default=1.0)
    parser.add_argument("--inverse-weight", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
