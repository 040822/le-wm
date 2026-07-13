"""Sampling strategies for windowed episode datasets."""

from __future__ import annotations

import math
import os
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import Sampler, Subset


def _base_dataset_index(dataset, index):
    """Resolve a possibly nested Subset position to its episode dataset index."""
    while isinstance(dataset, Subset):
        index = int(dataset.indices[index])
        dataset = dataset.dataset
    return dataset, index


class DistributedChunkLocalSampler(Sampler[int]):
    """Shuffle HDF5 chunks while keeping windows from each chunk adjacent.

    The sampler yields indices for the dataset passed to it (including a
    ``Subset``), but derives locality from the underlying episode window's
    physical HDF5 row. Each rank receives one contiguous section of the
    chunk-local order, so only rank boundaries can split a chunk.
    """

    def __init__(
        self,
        dataset,
        *,
        chunk_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        num_replicas: int | None = None,
        rank: int | None = None,
    ):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if (num_replicas is None) != (rank is None):
            raise ValueError("num_replicas and rank must be provided together")
        if num_replicas is not None and not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")

        self.dataset = dataset
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self._num_replicas = num_replicas
        self._rank = rank
        self.epoch = 0
        self._chunk_groups = self._build_chunk_groups()

    def _build_chunk_groups(self):
        groups = defaultdict(list)
        for sample_index in range(len(self.dataset)):
            base, base_index = _base_dataset_index(self.dataset, sample_index)
            if not hasattr(base, "clip_indices") or not hasattr(base, "offsets"):
                raise TypeError(
                    "chunk-local sampling requires clip_indices and offsets"
                )
            episode, start = base.clip_indices[base_index]
            physical_row = int(base.offsets[episode]) + int(start)
            groups[physical_row // self.chunk_size].append(sample_index)
        return dict(groups)

    def _distributed_context(self):
        if self._num_replicas is not None:
            return self._num_replicas, self._rank
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(), dist.get_rank()
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        if not 0 <= rank < world_size:
            raise RuntimeError(
                f"distributed rank {rank} is invalid for world size {world_size}"
            )
        return world_size, rank

    def _ordered_indices(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        chunk_ids = sorted(self._chunk_groups)
        if self.shuffle:
            permutation = torch.randperm(len(chunk_ids), generator=generator).tolist()
            chunk_ids = [chunk_ids[index] for index in permutation]

        ordered = []
        for chunk_id in chunk_ids:
            indices = self._chunk_groups[chunk_id]
            if self.shuffle and len(indices) > 1:
                permutation = torch.randperm(
                    len(indices), generator=generator
                ).tolist()
                indices = [indices[index] for index in permutation]
            ordered.extend(indices)
        return ordered

    def __iter__(self):
        ordered = self._ordered_indices()
        num_replicas, rank = self._distributed_context()
        num_samples = self._num_samples(num_replicas)
        total_size = num_samples * num_replicas

        if self.drop_last:
            ordered = ordered[:total_size]
        elif total_size > len(ordered):
            padding = total_size - len(ordered)
            repeats = math.ceil(padding / len(ordered)) if ordered else 0
            ordered += (ordered * repeats)[:padding]

        start = rank * num_samples
        return iter(ordered[start : start + num_samples])

    def _num_samples(self, num_replicas):
        if self.drop_last:
            return len(self.dataset) // num_replicas
        return math.ceil(len(self.dataset) / num_replicas)

    def __len__(self):
        num_replicas, _ = self._distributed_context()
        return self._num_samples(num_replicas)

    def set_epoch(self, epoch):
        """Select a deterministic but different chunk order for an epoch."""
        self.epoch = epoch


__all__ = ["DistributedChunkLocalSampler"]
