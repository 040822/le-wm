import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Subset

from source.common.data import load_dataset
from source.common.sampling import DistributedChunkLocalSampler
from tests.test_fast_lewam_model import make_model


ROOT = Path(__file__).resolve().parents[1]
CUBE_PATH = ROOT / "data" / "datasets" / "ogbench" / "cube_single.h5"


class SyntheticEpisodeDataset:
    def __init__(self, length):
        self.offsets = np.array([0])
        self.clip_indices = [(0, index) for index in range(length)]

    def __len__(self):
        return len(self.clip_indices)


class FastLeWAMDataPipelineTests(unittest.TestCase):
    def test_uint8_pixels_match_explicit_imagenet_normalization(self):
        torch.manual_seed(31)
        model = make_model().eval()
        pixels = torch.randint(0, 256, (2, 3, 3, 8, 8), dtype=torch.uint8)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        normalized = (pixels.float().div(255.0) - mean) / std

        expected = model.encode_pixels(normalized)
        actual = model.encode_pixels(pixels)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    @unittest.skipUnless(CUBE_PATH.exists(), "local Cube dataset is unavailable")
    def test_real_cube_loader_returns_channel_first_uint8_pixels(self):
        dataset = load_dataset(
            "ogbench/cube_single.h5",
            cache_dir=str(ROOT / "data"),
            num_steps=6,
            frameskip=5,
            keys_to_load=["pixels"],
        )

        pixels = dataset[0]["pixels"]

        self.assertEqual(pixels.dtype, torch.uint8)
        self.assertEqual(tuple(pixels.shape), (6, 3, 224, 224))

    def test_sampler_preserves_samples_and_groups_hdf5_chunks(self):
        dataset = SyntheticEpisodeDataset(length=24)
        subset = Subset(dataset, [11, 2, 20, 5, 17, 8, 23, 0, 14, 3, 19, 6])
        sampler = DistributedChunkLocalSampler(
            subset,
            chunk_size=4,
            shuffle=False,
            num_replicas=1,
            rank=0,
        )

        order = list(sampler)
        physical_rows = [subset.indices[index] for index in order]
        chunk_ids = [row // 4 for row in physical_rows]

        self.assertCountEqual(order, range(len(subset)))
        self.assertEqual(chunk_ids, sorted(chunk_ids))

    def test_fast_config_selects_the_high_throughput_data_path(self):
        cfg = OmegaConf.load(ROOT / "config" / "train" / "fast_lewam.yaml")

        self.assertTrue(cfg.data_pipeline.gpu_image_preprocessing)
        self.assertEqual(cfg.data_pipeline.hdf5_chunk_size, 100)
        self.assertFalse(cfg.trainer.use_distributed_sampler)


if __name__ == "__main__":
    unittest.main()
