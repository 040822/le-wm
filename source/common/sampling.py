"""面向窗口式 episode 数据集的采样策略，包含 Fast-LeWAM 的 HDF5 局部性优化。"""

from __future__ import annotations

import math
import os
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import Sampler, Subset


def _base_dataset_index(dataset, index):
    """递归展开嵌套 Subset，把局部位置解析为底层 episode 数据集索引。"""
    while isinstance(dataset, Subset):
        index = int(dataset.indices[index])
        dataset = dataset.dataset
    return dataset, index


class DistributedChunkLocalSampler(Sampler[int]):
    """按 HDF5 物理 chunk 聚合窗口，并为每个 DDP rank 分配连续样本区间。

    sampler 对外仍产生传入 dataset（包括 ``Subset``）的局部索引，但会通过
    底层数据集的 ``clip_indices`` 和 ``offsets`` 推导窗口起点所在的物理行。
    先打乱 chunk、再在 chunk 内打乱样本可兼顾随机性与解压局部性；连续的
    rank 分区则保证只有 rank 边界可能拆开一个 chunk。
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
        """记录采样配置，并预计算每个物理 HDF5 chunk 对应的 dataset 索引。"""
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
        """根据每个窗口的 episode offset 和起始行建立 chunk 到样本的映射。"""
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
        """按显式参数、torch.distributed、环境变量的优先级解析 world size 与 rank。"""
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
        """按当前 epoch 生成确定性的 chunk 顺序和 chunk 内样本顺序。"""
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
        """补齐或截断全局顺序，再返回当前 rank 对应的连续等长索引区间。"""
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
        """计算每个 rank 应产生的样本数，确保 DDP 各 rank step 数一致。"""
        if self.drop_last:
            return len(self.dataset) // num_replicas
        return math.ceil(len(self.dataset) / num_replicas)

    def __len__(self):
        """返回当前分布式上下文中单个 rank 的 sampler 长度。"""
        num_replicas, _ = self._distributed_context()
        return self._num_samples(num_replicas)

    def set_epoch(self, epoch):
        """设置 epoch，使各 rank 以相同种子切换到新的确定性 chunk 顺序。"""
        self.epoch = epoch


__all__ = ["DistributedChunkLocalSampler"]
