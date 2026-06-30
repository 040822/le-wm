import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from source.common.checkpoint import SaveCkptCallback
from source.common.data import get_column_normalizer, get_img_preprocessor, load_dataset
from source.common.logging import get_run_dir, tee_output_to_file


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

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
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.policy.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim("action")
        )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(
        val_set,
        **cfg.loader,
        shuffle=False,
        drop_last=False,
    )

    ##############################
    ##       policy / optim     ##
    ##############################

    policy = hydra.utils.instantiate(cfg.policy)

    ##########################
    ##       training       ##
    ##########################

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
        output_dir=run_dir,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    resume_ckpt = cfg.get("resume_ckpt", None)
    legacy_ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    if resume_ckpt is None and legacy_ckpt_path.exists():
        resume_ckpt = str(legacy_ckpt_path)

    trainer.fit(
        policy,
        train_dataloaders=train,
        val_dataloaders=val,
        ckpt_path=resume_ckpt,
    )


if __name__ == "__main__":
    run()
