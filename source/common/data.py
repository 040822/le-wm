from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
from stable_pretraining import data as dt


def load_dataset(dataset_name, cache_dir=None, **kwargs):
    """Load a stable-worldmodel dataset across supported API versions."""
    # ``stable_worldmodel.data.load_dataset`` treats every string as a
    # registry/HuggingFace name in newer releases.  Phase 3 stores the HDF5
    # files directly under ``data/datasets``; resolve an existing local file
    # before delegating so training and evaluation never attempt a network
    # lookup for an already-downloaded dataset.
    local_path = Path(dataset_name).expanduser()
    if local_path.is_file() and local_path.suffix in {".h5", ".lance"}:
        return swm.data.HDF5Dataset(
            path=local_path,
            cache_dir=cache_dir,
            **kwargs,
        )
    if hasattr(swm.data, "load_dataset"):
        return swm.data.load_dataset(dataset_name, cache_dir=cache_dir, **kwargs)

    path = local_path
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


def get_column_normalizer(
    dataset,
    source: str,
    target: str,
    *,
    row_indices=None,
    return_stats: bool = False,
):
    """Get a z-score normalizer for a dataset column."""
    col_data = dataset.get_col_data(source)
    if row_indices is not None:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if row_indices.ndim != 1 or len(row_indices) == 0:
            raise ValueError("row_indices must be a non-empty one-dimensional array")
        col_data = col_data[row_indices]
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    normalizer = dt.transforms.WrapTorchTransform(
        ZScoreNormalizer(mean, std),
        source=source,
        target=target,
    )
    if not return_stats:
        return normalizer
    stats = {
        "mean": mean.reshape(-1).tolist(),
        "std": std.reshape(-1).tolist(),
        "sample_count": int(data.shape[0]),
        "std_correction": 1,
    }
    return normalizer, stats
