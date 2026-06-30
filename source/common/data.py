from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
from stable_pretraining import data as dt


def load_dataset(dataset_name, cache_dir=None, **kwargs):
    """Load a stable-worldmodel dataset across supported API versions."""
    if hasattr(swm.data, "load_dataset"):
        return swm.data.load_dataset(dataset_name, cache_dir=cache_dir, **kwargs)

    path = Path(dataset_name)
    if path.suffix in {".h5", ".lance"}:
        dataset_name = str(path.with_suffix(""))

    return swm.data.HDF5Dataset(dataset_name, cache_dir=cache_dir, **kwargs)


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer for spawned DataLoader workers."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get a z-score normalizer for a dataset column."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(
        ZScoreNormalizer(mean, std),
        source=source,
        target=target,
    )
