import inspect
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms


def img_transform(cfg):
    """Build eval-time image transforms for current observations and goals."""
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


def fit_eval_processors(dataset, keys_to_cache):
    process = {}
    for col in keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    return process


def call_supported_kwargs(fn, **kwargs):
    params = inspect.signature(fn).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return fn(**kwargs)
    supported = {key: value for key, value in kwargs.items() if key in params}
    return fn(**supported)


def evaluate_from_dataset_compat(world, dataset, eval_start_idx, eval_episodes, cfg, video_path):
    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
    common_kwargs = {
        "dataset": dataset,
        "start_steps": eval_start_idx.tolist(),
        "episodes_idx": eval_episodes.tolist(),
        "eval_budget": cfg.eval.eval_budget,
        "callables": callables,
    }

    evaluate_params = inspect.signature(world.evaluate).parameters
    evaluate_accepts_dataset = "dataset" in evaluate_params or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in evaluate_params.values()
    )
    if evaluate_accepts_dataset:
        return call_supported_kwargs(
            world.evaluate,
            **common_kwargs,
            goal_offset=cfg.eval.goal_offset_steps,
            goal_offset_steps=cfg.eval.goal_offset_steps,
            video=video_path,
            video_path=video_path,
            save_video=True,
        )

    if hasattr(world, "evaluate_from_dataset"):
        return call_supported_kwargs(
            world.evaluate_from_dataset,
            **common_kwargs,
            goal_offset=cfg.eval.goal_offset_steps,
            goal_offset_steps=cfg.eval.goal_offset_steps,
            video=video_path,
            video_path=video_path,
            save_video=True,
        )

    raise TypeError(
        "当前 stable_worldmodel 版本既不支持 world.evaluate(dataset=...)，"
        "也没有 world.evaluate_from_dataset(...)；请升级 stable-worldmodel，"
        "或使用支持数据集驱动评测的版本。"
    )
