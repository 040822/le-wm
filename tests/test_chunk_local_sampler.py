import unittest

import numpy as np
from torch.utils.data import Subset

from source.common.sampling import DistributedChunkLocalSampler


class EpisodeWindows:
    def __init__(self, length):
        self.offsets = np.array([1000])
        self.clip_indices = [(0, index) for index in range(length)]

    def __len__(self):
        return len(self.clip_indices)


class DistributedChunkLocalSamplerTests(unittest.TestCase):
    def test_ranks_receive_equal_disjoint_partitions(self):
        base = EpisodeWindows(24)
        subset = Subset(base, [11, 2, 20, 5, 17, 8, 23, 0, 14, 3, 19, 6])
        samplers = [
            DistributedChunkLocalSampler(
                subset,
                chunk_size=4,
                shuffle=False,
                num_replicas=2,
                rank=rank,
            )
            for rank in range(2)
        ]

        partitions = [list(sampler) for sampler in samplers]

        self.assertEqual(len(partitions[0]), len(partitions[1]))
        self.assertTrue(set(partitions[0]).isdisjoint(partitions[1]))
        self.assertCountEqual(partitions[0] + partitions[1], range(len(subset)))

    def test_aligned_partitions_assign_whole_chunks_to_one_rank(self):
        dataset = EpisodeWindows(32)
        samplers = [
            DistributedChunkLocalSampler(
                dataset,
                chunk_size=4,
                shuffle=False,
                num_replicas=2,
                rank=rank,
            )
            for rank in range(2)
        ]

        chunk_sets = [
            {(dataset.offsets[0] + index) // 4 for index in sampler}
            for sampler in samplers
        ]

        self.assertTrue(chunk_sets[0].isdisjoint(chunk_sets[1]))

    def test_epoch_shuffle_is_deterministic_and_changes_order(self):
        dataset = EpisodeWindows(32)
        first = DistributedChunkLocalSampler(
            dataset,
            chunk_size=4,
            shuffle=True,
            seed=17,
            num_replicas=1,
            rank=0,
        )
        second = DistributedChunkLocalSampler(
            dataset,
            chunk_size=4,
            shuffle=True,
            seed=17,
            num_replicas=1,
            rank=0,
        )

        epoch_zero = list(first)
        self.assertEqual(epoch_zero, list(second))
        first.set_epoch(1)
        self.assertNotEqual(epoch_zero, list(first))


if __name__ == "__main__":
    unittest.main()
