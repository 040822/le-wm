import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from source.experiments.round3_phase2 import (
    MixedReplaySampler,
    ReplayShardPool,
    StageBOnlineAdapter,
    TransitionReplay,
    build_offline_window_manifest,
    load_offline_replay,
    freeze_lewm_encoder_projector,
    load_transition_replay,
    phase2_update_steps,
    save_transition_replay,
    concatenate_replays,
)


class ToyLeWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(4, 6))
        self.projector = nn.Sequential(nn.Linear(6, 4), nn.BatchNorm1d(4))
        self.action_encoder = nn.Linear(2, 4)
        self.predictor = nn.Linear(4, 4)
        self.pred_proj = nn.Linear(4, 4)

    def encode(self, info):
        pixels = info["pixels"].float()
        batch, steps = pixels.shape[:2]
        emb = self.projector(self.encoder(pixels.reshape(batch * steps, -1)))
        return {"emb": emb.reshape(batch, steps, -1)}

    def predict(self, emb, act_emb):
        return self.pred_proj(self.predictor(emb + act_emb))


class FakeOfflineDataset:
    column_names = ["ep_idx"]

    def __init__(self):
        self.lengths = np.asarray([21, 21])
        self.offsets = np.asarray([0, 21])
        self.frameskip = 1
        self.num_steps = 1

    def get_col_data(self, name):
        self.assert_name = name
        return np.repeat(np.asarray([10, 20]), 21)

    def load_chunk(self, episodes, starts, ends):
        chunks = []
        for episode, start, end in zip(episodes, starts, ends):
            pixels = torch.full((4, 3, 2, 2), int(episode), dtype=torch.uint8)
            actions = torch.arange(40, dtype=torch.float32).reshape(4, 10)
            chunks.append({"pixels": pixels, "action": actions})
        return chunks


class IdentityActionProcessor:
    def transform(self, value):
        return value


def make_replay(sources=("offline", "offline", "online", "online")):
    observations = torch.randn(4, 4, 1, 2, 2)
    actions = torch.randn(4, 3, 2)
    return TransitionReplay(
        observations=observations,
        actions=actions,
        goals=torch.randn(4, 3),
        episode_ids=((10, 10, 10, 10),) * 4,
        start_steps=torch.arange(4),
        sources=tuple(sources),
        model_versions=("e0:epoch10",) * 4,
    )


class Round3Phase2Tests(unittest.TestCase):
    def test_phase2_schedule_has_two_hundred_one_hundred_step_boundaries(self):
        steps = phase2_update_steps()
        self.assertEqual(len(steps), 200)
        self.assertEqual(steps[:2], (100, 200))
        self.assertEqual(steps[-1], 20000)
        with self.assertRaises(ValueError):
            phase2_update_steps(20000, 750)

    def test_replay_hash_and_episode_boundary_validation(self):
        replay = make_replay()
        self.assertEqual(len(replay.content_sha256()), 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.pt"
            save_transition_replay(path, replay)
            self.assertEqual(load_transition_replay(path).content_sha256(), replay.content_sha256())
        with self.assertRaises(ValueError):
            TransitionReplay(
                observations=replay.observations,
                actions=replay.actions,
                goals=replay.goals,
                episode_ids=((10, 10, 11, 11),) * 4,
                start_steps=replay.start_steps,
                sources=replay.sources,
                model_versions=replay.model_versions,
            )

    def test_grounded_metadata_round_trips_and_preserves_group_buckets(self):
        replay = TransitionReplay(
            observations=torch.randn(4, 4, 1, 2, 2),
            actions=torch.randn(4, 3, 2),
            goals=torch.randn(4, 3),
            goal_latents=torch.randn(4, 8),
            episode_ids=((10, 10, 10, 10),) * 4,
            start_steps=torch.arange(4),
            sources=("offline", "offline", "online", "online"),
            data_kinds=("offline", "offline", "grounded", "grounded"),
            group_ids=(0, 1, 7, 7),
            snapshot_ids=("s0", "s1", "snap", "snap"),
            collector_versions=("c0",) * 4,
            model_versions=("e5:epoch10",) * 4,
            successes=torch.tensor([False, True, True, False]),
            physical_start_costs=torch.tensor([2.0, 2.0, 3.0, 3.0]),
            physical_terminal_costs=torch.tensor([1.5, 1.0, 0.5, 2.0]),
            execution_lengths=torch.tensor([15, 15, 25, 25]),
        )
        self.assertEqual(replay.resolved_data_kinds, ("offline", "offline", "grounded", "grounded"))
        self.assertEqual(replay.resolved_group_ids, (0, 1, 7, 7))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grounded.pt"
            save_transition_replay(path, replay)
            restored = load_transition_replay(path)
        self.assertEqual(restored.content_sha256(), replay.content_sha256())
        self.assertEqual(restored.group_ids, replay.group_ids)
        torch.testing.assert_close(restored.goal_latents, replay.goal_latents)
        torch.testing.assert_close(restored.physical_terminal_costs, replay.physical_terminal_costs)
        subset = replay.subset([2, 3])
        self.assertEqual(subset.data_kinds, ("grounded", "grounded"))
        self.assertEqual(subset.group_ids, (7, 7))
        torch.testing.assert_close(subset.successes, torch.tensor([True, False]))

    def test_mixed_sampler_does_not_fabricate_missing_online_rows(self):
        replay = make_replay(("offline", "offline", "offline", "offline"))
        sampler = MixedReplaySampler(replay, seed=7)
        indices, counts = sampler.sample(4, mixed=True)
        self.assertEqual(indices.numel(), 0)
        self.assertEqual(counts, {"offline": 0, "online": 0})

        replay = make_replay()
        indices, counts = MixedReplaySampler(replay, seed=7).sample(4, mixed=True)
        self.assertEqual(indices.numel(), 4)
        self.assertEqual(counts, {"offline": 2, "online": 2})

        online_indices, online_counts = MixedReplaySampler(replay, seed=7).sample(
            4,
            mixed=False,
            source="online",
        )
        self.assertEqual(online_indices.numel(), 2)
        self.assertEqual(online_counts, {"offline": 0, "online": 2})

    def test_freeze_and_cpu_update_only_change_dynamics(self):
        torch.manual_seed(3)
        model = ToyLeWM()
        encoder_before = copy.deepcopy(model.encoder.state_dict())
        projector_before = copy.deepcopy(model.projector.state_dict())
        names = freeze_lewm_encoder_projector(model)
        self.assertTrue(names)
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.projector.parameters()))

        adapter = StageBOnlineAdapter(model, seed=3, device="cpu")
        before = {
            name: value.detach().clone()
            for name, value in model.named_parameters()
            if value.requires_grad
        }
        report = adapter.update(make_replay(), batch_size=4, updates=1, mixed=True)
        self.assertEqual(report.sample_count, 4)
        self.assertEqual(report.optimizer_steps, 1)
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.named_parameters() if value.requires_grad))
        self.assertEqual(model.encoder.state_dict().keys(), encoder_before.keys())
        self.assertEqual(model.projector.state_dict().keys(), projector_before.keys())
        for name, value in encoder_before.items():
            self.assertTrue(torch.equal(value, model.encoder.state_dict()[name]))
        for name, value in projector_before.items():
            self.assertTrue(torch.equal(value, model.projector.state_dict()[name]))
        self.assertFalse(model.projector[1].training)

    def test_online_only_update_has_an_optimizer_step(self):
        torch.manual_seed(9)
        model = ToyLeWM()
        adapter = StageBOnlineAdapter(model, seed=9, device="cpu")
        report = adapter.update(
            make_replay(("online",) * 4),
            batch_size=4,
            updates=1,
            mixed=False,
            source="online",
        )
        self.assertEqual(report.sample_count, 4)
        self.assertEqual(report.offline_count, 0)
        self.assertEqual(report.online_count, 4)
        self.assertEqual(report.optimizer_steps, 1)
        self.assertIsNotNone(report.loss)

    def test_offline_manifest_and_loader_keep_native_action_blocks(self):
        dataset = FakeOfflineDataset()
        manifest = build_offline_window_manifest(
            dataset,
            episode_ids=(10, 20),
            count=2,
            seed=3,
            frameskip=5,
            num_steps=4,
        )
        replay = load_offline_replay(
            dataset,
            manifest,
            process={"action": IdentityActionProcessor()},
            observation_transform=lambda value: value.float(),
            model_version="e0:test",
            history_size=3,
            frameskip=5,
            num_steps=4,
            goal_offset_steps=2,
        )
        self.assertEqual(replay.observations.shape, (2, 4, 3, 2, 2))
        self.assertEqual(replay.actions.shape, (2, 3, 10))
        self.assertEqual(replay.raw_observations.dtype, torch.uint8)
        self.assertEqual(replay.sources, ("offline", "offline"))
        self.assertEqual(tuple(replay.goals.shape), (2, 3, 2, 2))

    def test_concatenation_preserves_online_diagnostics_when_offline_has_none(self):
        offline = make_replay(("offline",) * 4)
        online = TransitionReplay(
            observations=torch.randn(2, 4, 1, 2, 2),
            actions=torch.randn(2, 3, 2),
            episode_ids=((20, 20, 20, 20),) * 2,
            start_steps=torch.arange(2),
            sources=("online",) * 2,
            data_kinds=("continuous",) * 2,
            model_versions=("e5:test",) * 2,
            successes=torch.tensor([True, False]),
            physical_start_costs=torch.tensor([2.0, 2.0]),
            physical_terminal_costs=torch.tensor([1.0, 3.0]),
            execution_lengths=torch.tensor([15, 15]),
        )
        mixed = concatenate_replays((offline, online))
        self.assertTrue(torch.equal(mixed.successes[-2:], torch.tensor([True, False])))
        self.assertTrue(torch.equal(mixed.physical_terminal_costs[-2:], torch.tensor([1.0, 3.0])))

    def test_append_only_shard_pool_samples_without_merging_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.pt"
            second = Path(directory) / "second.pt"
            first_replay = make_replay(("online",) * 4)
            second_replay = make_replay(("online",) * 4)
            save_transition_replay(first, first_replay)
            save_transition_replay(second, second_replay)
            pool = ReplayShardPool(seed=4)
            pool.add(first, count=first_replay.count, content_sha256=first_replay.content_sha256())
            pool.add(second, count=second_replay.count, content_sha256=second_replay.content_sha256())
            sample = pool.sample(5)
            self.assertEqual(pool.count, 8)
            self.assertEqual(sample.count, 5)
            self.assertEqual(set(sample.sources), {"online"})


if __name__ == "__main__":
    unittest.main()
