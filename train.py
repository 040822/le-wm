import os
import atexit
import logging
import re
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback

#########################
##       日志模块       ##
#########################
class TeeStream:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        self.encoding = getattr(stream, "encoding", "utf-8")

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.stream.isatty()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "run"


def get_run_dir(cfg, dataset_name):
    try:
        hydra_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        hydra_dir = None

    subdir = cfg.get("subdir")
    subdir = str(subdir).strip() if subdir is not None else ""
    if subdir and subdir.lower() not in {"none", "null"} and not subdir.startswith("${"):
        run_dir = Path("outputs", safe_name(subdir))
    elif hydra_dir is not None:
        run_dir = hydra_dir
    else:
        stem = Path(str(dataset_name)).with_suffix("").as_posix()
        run_id = safe_name(
            f"{datetime.now():%Y%m%d_%H%M%S_%f}_{cfg.output_model_name}_{stem}_{os.getpid()}"
        )
        run_dir = Path("outputs", run_id)

    run_dir = run_dir.resolve()
    return run_dir, safe_name(run_dir.name)


def tee_output_to_file(run_dir, dataset_name):
    run_dir.mkdir(parents=True, exist_ok=True)
    log_stem = safe_name(Path(str(dataset_name)).with_suffix("").name)
    log_path = run_dir / f"{log_stem}.out"
    log_file = open(log_path, "a", buffering=1)
    stdout, stderr = sys.stdout, sys.stderr
    print(f"Logging stdout/stderr to {log_path}")
    sys.stdout = TeeStream(stdout, log_file)
    sys.stderr = TeeStream(stderr, log_file)

    py_handler = logging.FileHandler(log_path)
    py_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(py_handler)

    loguru_logger = None
    loguru_sink_id = None
    try:
        from loguru import logger as loguru_logger
        loguru_sink_id = loguru_logger.add(log_file, level="DEBUG", colorize=False)
    except Exception:
        pass

    def close_log():
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            if loguru_logger is not None and loguru_sink_id is not None:
                loguru_logger.remove(loguru_sink_id)
            logging.getLogger().removeHandler(py_handler)
            py_handler.close()
            if isinstance(sys.stdout, TeeStream) and sys.stdout.log_file is log_file:
                sys.stdout = stdout
            if isinstance(sys.stderr, TeeStream) and sys.stderr.log_file is log_file:
                sys.stderr = stderr
            log_file.close()

    atexit.register(close_log)
    return log_path

# API修复，用于兼容不同版本的stable_worldmodel和数据集接口
def load_dataset(dataset_name, cache_dir=None, **kwargs):
    if hasattr(swm.data, "load_dataset"):
        return swm.data.load_dataset(dataset_name, cache_dir=cache_dir, **kwargs)

    path = Path(dataset_name)
    if path.suffix in {".h5", ".lance"}:
        dataset_name = str(path.with_suffix(""))

    return swm.data.HDF5Dataset(dataset_name, cache_dir=cache_dir, **kwargs)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

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
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

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
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1, output_dir=run_dir,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
