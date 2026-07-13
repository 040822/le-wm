import os
from datetime import timedelta

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from source.common.checkpoint import (
    CHECKPOINTS_DIRNAME,
    SaveCkptCallback,
    resolve_training_resume_checkpoint,
)
from source.common.data import get_column_normalizer, get_img_preprocessor, load_dataset
from source.common.logging import get_run_dir, tee_output_to_file
from source.common.sampling import DistributedChunkLocalSampler


def _data_pipeline_options(cfg):
    pipeline = cfg.get("data_pipeline", {})
    return (
        bool(pipeline.get("gpu_image_preprocessing", False)),
        pipeline.get("hdf5_chunk_size", None),
    )


def _make_dataloaders(cfg, train_set, val_set, rnd_gen, chunk_size):
    train_sampler = None
    val_sampler = None
    if chunk_size is not None:
        train_sampler = DistributedChunkLocalSampler(
            train_set,
            chunk_size=int(chunk_size),
            shuffle=True,
            seed=cfg.seed,
        )
        val_sampler = DistributedChunkLocalSampler(
            val_set,
            chunk_size=int(chunk_size),
            shuffle=False,
            seed=cfg.seed,
        )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(
        val_set,
        **cfg.loader,
        sampler=val_sampler,
        shuffle=False,
        drop_last=False,
    )
    return train, val


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    run_dir, run_id = get_run_dir(cfg, dataset_name)
    log_path = tee_output_to_file(run_dir, dataset_name)

    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = load_dataset(
        dataset_name,
        transform=None,
        cache_dir=cache_dir,
        **dataset_cfg,
    )
    gpu_image_preprocessing, chunk_size = _data_pipeline_options(cfg)
    transforms = []
    if not gpu_image_preprocessing:
        transforms.append(
            get_img_preprocessor(
                source="pixels", target="pixels", img_size=cfg.img_size
            )
        )

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        action_block_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
        if "action_encoder" in cfg.policy.model:
            cfg.policy.model.action_encoder.input_dim = action_block_dim
        if "action_dim" in cfg.policy.model:
            cfg.policy.model.action_dim = action_block_dim

    dataset.transform = spt.data.transforms.Compose(*transforms)
    if gpu_image_preprocessing:
        print("Using channel-first uint8 pixels with device-side ImageNet normalization")
    if chunk_size is not None:
        print(f"Using distributed HDF5 chunk-local sampler (chunk_size={chunk_size})")

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    train, val = _make_dataloaders(
        cfg, train_set, val_set, rnd_gen, chunk_size
    )

    policy = hydra.utils.instantiate(cfg.policy)

    with open_dict(cfg):
        cfg.subdir = run_id
        cfg.trainer.default_root_dir = str(run_dir)

    logger = None
    if cfg.wandb.enabled:
        wandb_config = OmegaConf.to_container(cfg.wandb.config, resolve=True)
        wandb_config["id"] = run_id
        if not wandb_config.get("name") or wandb_config.get("name") == cfg.output_model_name:
            wandb_config["name"] = run_id
        wandb_config["save_dir"] = str(run_dir)
        os.environ["WANDB_DIR"] = str(run_dir)
        logger = WandbLogger(**wandb_config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    with open(run_dir / "run_paths.txt", "w") as f:
        f.write(f"run_dir={run_dir}\n")
        f.write(f"log_path={log_path}\n")

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.policy,
        epoch_interval=1,
        output_dir=run_dir / CHECKPOINTS_DIRNAME,
    )
    latest_checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir / CHECKPOINTS_DIRNAME,
        save_last=True,
        save_top_k=0,
        train_time_interval=timedelta(
            minutes=float(cfg.get("checkpoint_interval_minutes", 30))
        ),
        enable_version_counter=False,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, latest_checkpoint_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    resume_ckpt = resolve_training_resume_checkpoint(
        cfg.get("resume_ckpt", None),
        run_dir,
        cfg.output_model_name,
    )
    if resume_ckpt is not None:
        print(f"Resuming training from checkpoint: {resume_ckpt}")

    trainer.fit(
        policy,
        train_dataloaders=train,
        val_dataloaders=val,
        ckpt_path=resume_ckpt,
    )


if __name__ == "__main__":
    run()
