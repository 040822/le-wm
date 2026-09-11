"""Train the LeFlow latent path flow and inverse-dynamics planner."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from source.model.leflow.latent_planner import (
    InverseDynamics,
    LatentPathFlow,
    checkpoint_payload,
    flow_matching_loss,
    inverse_dynamics_loss,
    lewm_consistency_loss,
    load_lewm,
    smoothness_loss,
)
from source.common.data import get_column_normalizer, get_img_preprocessor


def _cache_dir(sub_folder: str | None = None) -> Path:
    try:
        return Path(swm.data.utils.get_cache_dir(sub_folder=sub_folder))
    except TypeError:
        root = Path(swm.data.utils.get_cache_dir())
        return root if sub_folder is None else root / sub_folder


def freeze(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    module.requires_grad_(False)
    return module


def _split_path(cfg: DictConfig) -> Path:
    name = str(cfg.data.dataset.name).replace("/", "_")
    fraction = int(round(float(cfg.episode_split.train_fraction) * 100))
    return _cache_dir() / "splits" / f"{name}_seed{cfg.episode_split.seed}_train{fraction}.json"


def load_or_create_episode_split(
    cfg: DictConfig, num_episodes: int
) -> tuple[set[int], set[int], Path]:
    split_file = Path(cfg.episode_split.split_file) if cfg.episode_split.split_file else _split_path(cfg)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    if split_file.is_file():
        payload = json.loads(split_file.read_text(encoding="utf-8"))
    else:
        episodes = np.arange(num_episodes)
        np.random.default_rng(int(cfg.episode_split.seed)).shuffle(episodes)
        count = int(round(num_episodes * float(cfg.episode_split.train_fraction)))
        count = min(max(count, 1), max(num_episodes - 1, 1))
        payload = {
            "dataset": str(cfg.data.dataset.name),
            "seed": int(cfg.episode_split.seed),
            "train_fraction": float(cfg.episode_split.train_fraction),
            "num_episodes": int(num_episodes),
            "train_episodes": sorted(int(item) for item in episodes[:count]),
            "eval_episodes": sorted(int(item) for item in episodes[count:]),
        }
        split_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    train = {int(item) for item in payload["train_episodes"]}
    evaluation = {int(item) for item in payload["eval_episodes"]}
    if not train or not evaluation:
        raise ValueError("episode split must contain both train and eval episodes")
    return train, evaluation, split_file


@torch.no_grad()
def encode_latents(lewm: torch.nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    return lewm.encode({"pixels": batch["pixels"].to(device)})["emb"].detach()


def make_loaders(cfg: DictConfig):
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor("pixels", "pixels", img_size=cfg.img_size)]
    with open_dict(cfg):
        for column in cfg.data.dataset.keys_to_load:
            if column.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(dataset, column, column))
            setattr(cfg.data, f"{column}_dim", dataset.get_dim(column))
    dataset.transform = spt.data.transforms.Compose(*transforms)

    if cfg.episode_split.enabled:
        train_episodes, _, split_file = load_or_create_episode_split(
            cfg, num_episodes=len(dataset.lengths)
        )
        dataset.clip_indices = [
            (episode, start)
            for episode, start in dataset.clip_indices
            if int(episode) in train_episodes
        ]
        print(f"episode_split={split_file} train_episodes={len(train_episodes)}", flush=True)

    generator = torch.Generator().manual_seed(int(cfg.seed))
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=generator,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=generator
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )
    return train_loader, val_loader


def step_batch(*, batch, lewm, flow, inverse_dynamics, cfg, device):
    batch["action"] = torch.nan_to_num(batch["action"].to(device), 0.0)
    z_path = encode_latents(lewm, batch, device)
    actions = batch["action"][:, : cfg.planner.horizon]
    loss_flow = flow_matching_loss(flow, z_path)
    loss_inverse, predicted_actions = inverse_dynamics_loss(
        inverse_dynamics, z_path, actions
    )
    if cfg.loss.consistency.weight:
        if cfg.loss.consistency.detach_inverse:
            with torch.no_grad():
                loss_consistency = lewm_consistency_loss(
                    lewm,
                    z_path,
                    predicted_actions.detach(),
                    history_size=cfg.lewm_history_size,
                )
        else:
            loss_consistency = lewm_consistency_loss(
                lewm,
                z_path,
                predicted_actions,
                history_size=cfg.lewm_history_size,
            )
    else:
        loss_consistency = z_path.new_tensor(0.0)
    loss_smoothness = smoothness_loss(z_path)
    total = (
        cfg.loss.flow.weight * loss_flow
        + cfg.loss.inverse.weight * loss_inverse
        + cfg.loss.consistency.weight * loss_consistency
        + cfg.loss.smoothness.weight * loss_smoothness
    )
    return {
        "loss": total,
        "flow_loss": loss_flow.detach(),
        "inverse_loss": loss_inverse.detach(),
        "consistency_loss": loss_consistency.detach(),
        "smoothness_loss": loss_smoothness.detach(),
    }


@torch.no_grad()
def validate(*, loader, lewm, flow, inverse_dynamics, cfg, device):
    flow.eval()
    inverse_dynamics.eval()
    sums: dict[str, float] = {}
    count = 0
    for index, batch in enumerate(loader):
        output = step_batch(
            batch=batch,
            lewm=lewm,
            flow=flow,
            inverse_dynamics=inverse_dynamics,
            cfg=cfg,
            device=device,
        )
        batch_size = batch["pixels"].size(0)
        count += batch_size
        for key, value in output.items():
            sums[key] = sums.get(key, 0.0) + float(value) * batch_size
        if cfg.val_batches is not None and index + 1 >= cfg.val_batches:
            break
    flow.train()
    inverse_dynamics.train()
    return {key: value / max(count, 1) for key, value in sums.items()}


@hydra.main(version_base=None, config_path="./config/train", config_name="latent_planner")
def run(cfg: DictConfig):
    torch.manual_seed(int(cfg.seed))
    device = torch.device(str(cfg.device) if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_loaders(cfg)
    lewm = freeze(load_lewm(cfg.lewm_checkpoint).to(device))

    first_batch = next(iter(train_loader))
    latent_dim = encode_latents(lewm, first_batch, device).size(-1)
    action_dim = first_batch["action"].size(-1)
    flow = LatentPathFlow(
        latent_dim=latent_dim,
        max_horizon=cfg.planner.max_horizon,
        **cfg.flow,
    ).to(device)
    inverse_dynamics = InverseDynamics(
        latent_dim=latent_dim,
        action_dim=action_dim,
        **cfg.inverse_dynamics,
    ).to(device)
    parameters = list(flow.parameters()) + list(inverse_dynamics.parameters())
    optimizer = torch.optim.AdamW(parameters, **cfg.optimizer)

    run_dir = _cache_dir("checkpoints") / str(cfg.subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "latent_planner_config.yaml")
    for epoch in range(int(cfg.epochs)):
        flow.train()
        inverse_dynamics.train()
        for batch_index, batch in enumerate(train_loader):
            output = step_batch(
                batch=batch,
                lewm=lewm,
                flow=flow,
                inverse_dynamics=inverse_dynamics,
                cfg=cfg,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            output["loss"].backward()
            if cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters, float(cfg.grad_clip_norm))
            optimizer.step()
            if cfg.max_train_batches is not None and batch_index + 1 >= cfg.max_train_batches:
                break
            if (batch_index + 1) % int(cfg.log_interval) == 0:
                values = " ".join(
                    f"{key}={float(value):.5f}" for key, value in output.items()
                )
                print(f"epoch={epoch + 1} batch={batch_index + 1} {values}", flush=True)

        metrics = validate(
            loader=val_loader,
            lewm=lewm,
            flow=flow,
            inverse_dynamics=inverse_dynamics,
            cfg=cfg,
            device=device,
        )
        print(
            f"epoch={epoch + 1} validation="
            + " ".join(f"{key}={value:.5f}" for key, value in metrics.items()),
            flush=True,
        )
        payload = checkpoint_payload(
            lewm_checkpoint=str(cfg.lewm_checkpoint),
            action_block=int(cfg.planner.action_block),
            flow=flow,
            inverse_dynamics=inverse_dynamics,
            cfg=OmegaConf.to_container(cfg, resolve=True),
        )
        torch.save(payload, run_dir / "latent_planner.pt")

    print(f"saved={run_dir / 'latent_planner.pt'}", flush=True)


if __name__ == "__main__":
    run()
