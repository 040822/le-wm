import unittest

import numpy as np
import torch

from source.experiments.round3_phase2_collection import ExecutedTransitionRecorder


class FakePolicy:
    def __init__(self):
        self.step = 0

    def set_env(self, env):
        self.env = env

    def get_action(self, info, **kwargs):
        count = len(info["pixels"])
        action = np.full((count, 2), self.step, dtype=np.float32)
        self.step += 1
        return action


class Round3Phase2CollectionTests(unittest.TestCase):
    def test_records_only_complete_action_block_windows(self):
        policy = FakePolicy()
        recorder = ExecutedTransitionRecorder(
            policy,
            episode_ids=[101, 102],
            start_steps=[4, 7],
            history_size=3,
            action_block=2,
            model_version="e0:epoch10",
        )
        for step in range(8):
            info = {
                "pixels": np.full((2, 1, 2, 2), step, dtype=np.uint8),
                "goal": np.zeros((2, 3), dtype=np.float32),
            }
            recorder.get_action(info)
        replay = recorder.replay()
        self.assertEqual(replay.count, 2)
        self.assertEqual(tuple(replay.observations.shape), (2, 4, 1, 2, 2))
        self.assertEqual(tuple(replay.actions.shape), (2, 3, 4))
        self.assertEqual(tuple(replay.raw_actions.shape), (2, 3, 4))
        self.assertEqual(replay.episode_ids[0], (101, 101, 101, 101))
        self.assertEqual(replay.start_steps.tolist(), [4, 7])
        self.assertTrue(torch.equal(replay.observations[:, 0], replay.raw_observations[:, 0].float()))

    def test_outcomes_are_bound_to_env_index_when_candidates_share_episode(self):
        policy = FakePolicy()
        recorder = ExecutedTransitionRecorder(
            policy,
            episode_ids=[7, 7, 7, 7],
            start_steps=[3, 3, 3, 3],
            history_size=3,
            action_block=2,
            model_version="e5:epoch10",
            data_kind="grounded",
            group_ids=["g"] * 4,
            snapshot_ids=["s"] * 4,
        )
        for step in range(8):
            recorder.get_action(
                {
                    "pixels": np.full((4, 1, 2, 2), step, dtype=np.uint8),
                    "goal": np.zeros((4, 1), dtype=np.float32),
                }
            )
        replay = recorder.replay(
            outcomes={
                (7, 0): {
                    "success": True,
                    "physical_start_cost": 2.0,
                    "physical_terminal_cost": 1.0,
                },
                (7, 1): {
                    "success": False,
                    "physical_start_cost": 2.0,
                    "physical_terminal_cost": 3.0,
                },
                (7, 2): {
                    "success": True,
                    "physical_start_cost": 2.0,
                    "physical_terminal_cost": 1.5,
                },
                (7, 3): {
                    "success": False,
                    "physical_start_cost": 2.0,
                    "physical_terminal_cost": 2.5,
                },
            }
        )
        self.assertEqual(replay.count, 4)
        self.assertTrue(torch.equal(replay.successes, torch.tensor([True, False, True, False])))
        self.assertEqual(replay.group_ids, ("g",) * 4)

    def test_finalize_closes_last_complete_block_without_extra_environment_step(self):
        policy = FakePolicy()
        recorder = ExecutedTransitionRecorder(
            policy,
            episode_ids=[7],
            start_steps=[3],
            history_size=3,
            action_block=2,
            model_version="e5:epoch10",
        )
        for step in range(8):
            recorder.get_action(
                {
                    "pixels": np.full((1, 1, 2, 2), step, dtype=np.uint8),
                    "goal": np.zeros((1, 1), dtype=np.float32),
                }
            )

        before = recorder.environment_steps
        recorder.finalize(
            {
                "pixels": np.full((1, 1, 2, 2), 8, dtype=np.uint8),
                "goal": np.zeros((1, 1), dtype=np.float32),
            }
        )
        replay = recorder.replay()

        self.assertEqual(recorder.environment_steps, before)
        self.assertEqual(replay.count, 2)
        self.assertEqual(tuple(replay.observations.shape), (2, 4, 1, 2, 2))
        self.assertEqual(tuple(replay.actions.shape), (2, 3, 4))
        self.assertEqual(replay.observations[-1, -1].mean().item(), 8.0)


if __name__ == "__main__":
    unittest.main()
