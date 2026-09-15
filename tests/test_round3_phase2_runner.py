import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import torch
from torch import nn

import scripts.round3_phase2 as phase2_script
from source.experiments.round3_phase2 import (
    TransitionReplay,
    UpdateReport,
    save_transition_replay,
)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


class FakeOptimizer:
    def __init__(self):
        self.steps = 0

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, value):
        self.steps = int(value["steps"])


class FakeAdapter:
    instances = []

    def __init__(self, model, **kwargs):
        self.model = model
        self.optimizer = FakeOptimizer()
        self.device = torch.device(kwargs["device"])
        self.optimizer_steps = 0
        self.sampler_seed = int(kwargs["seed"])
        self.updates = 0
        self.__class__.instances.append(self)

    def prepare_for_update(self):
        return None

    def update(self, replay, **kwargs):
        self.updates += 1
        self.optimizer_steps += 1
        self.optimizer.steps = self.optimizer_steps
        return UpdateReport(
            loss=0.5,
            sample_count=replay.count,
            offline_count=sum(value == "offline" for value in replay.sources),
            online_count=sum(value == "online" for value in replay.sources),
            optimizer_steps=self.optimizer_steps,
        )


class FakeDataset:
    pass


def _fake_replay(source):
    return TransitionReplay(
        observations=torch.zeros(4, 4, 1, 2, 2),
        actions=torch.zeros(4, 3, 2),
        episode_ids=((1, 1, 1, 1),) * 4,
        start_steps=torch.arange(4),
        sources=(source,) * 4,
        model_versions=("fake:e0",) * 4,
    )


class Round3Phase2RunnerTests(unittest.TestCase):
    def test_mock_runner_executes_each_1000_step_boundary_and_persists_state(self):
        FakeAdapter.instances = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "e0.ckpt"
            checkpoint.write_bytes(b"fake-e0")
            pool_path = root / "pool.json"
            pool_path.write_text(
                json.dumps({"online_episode_ids": list(range(100))}),
                encoding="utf-8",
            )

            def fake_load(_name):
                return FakeModel(), checkpoint

            def fake_collect_online(*, output, **kwargs):
                replay = _fake_replay("online")
                save_transition_replay(output, replay)
                return {
                    "environment_steps": 1000,
                    "target_environment_steps": 1000,
                    "next_pool_offset": int(kwargs["pool_offset"]) + 1,
                    "replay_count": replay.count,
                    "replay_sha256": replay.content_sha256(),
                    "episode_successes": [],
                    "status": "ok",
                }

            cfg = types.SimpleNamespace(
                eval=types.SimpleNamespace(eval_budget=50, dataset_name="fake"),
                plan_config=types.SimpleNamespace(action_block=5),
                dataset=types.SimpleNamespace(keys_to_cache=()),
            )

            with mock.patch.object(phase2_script, "compose_eval_config", return_value=cfg), \
                 mock.patch.object(phase2_script, "get_dataset", return_value=FakeDataset()), \
                 mock.patch("source.common.eval.fit_eval_processors", return_value={}), \
                 mock.patch("source.common.eval.img_transform", return_value=lambda value: value), \
                 mock.patch("source.common.round3_eval.validate_gpu_visibility"), \
                 mock.patch("source.common.checkpoint.load_policy_or_model", side_effect=fake_load), \
                 mock.patch("source.experiments.round3_phase2.StageBOnlineAdapter", FakeAdapter), \
                 mock.patch.object(phase2_script, "_load_offline_batch", return_value=(_fake_replay("offline"), ({"start": 0},))), \
                 mock.patch.object(phase2_script, "collect_online", side_effect=fake_collect_online), \
                 mock.patch.object(phase2_script, "_evaluate_phase2_dev"):
                result = phase2_script.run_phase2(
                    checkpoint=checkpoint,
                    pool_path=pool_path,
                    output_root=root / "run",
                    device="cuda:0",
                    max_environment_steps=2000,
                    update_interval=1000,
                    batch_size=4,
                    chunk_size=1,
                    skip_eval=True,
                )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(result["replay_shards"]), 2)
            self.assertEqual(len(FakeAdapter.instances), 2)
            self.assertEqual([adapter.updates for adapter in FakeAdapter.instances], [2, 2])
            state = json.loads((root / "run" / "run_state.json").read_text())
            self.assertEqual(state["current_environment_steps"], 2000)
            self.assertEqual(len([item for item in state["records"] if "update" in item]), 2)
            self.assertEqual(
                sorted(path.name for path in (root / "run" / "checkpoints").glob("*.ckpt")),
                [
                    "cube_offline_continue_envsteps_001000.ckpt",
                    "cube_offline_continue_envsteps_002000.ckpt",
                    "cube_online_adapt_envsteps_001000.ckpt",
                    "cube_online_adapt_envsteps_002000.ckpt",
                ],
            )

            state["status"] = "running"
            (root / "run" / "run_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            with mock.patch("source.common.round3_eval.validate_gpu_visibility"), \
                 mock.patch("source.common.checkpoint.load_policy_or_model", side_effect=fake_load), \
                 mock.patch("source.experiments.round3_phase2.StageBOnlineAdapter", FakeAdapter):
                resumed = phase2_script.run_phase2(
                    checkpoint=checkpoint,
                    pool_path=pool_path,
                    output_root=root / "run",
                    device="cuda:0",
                    max_environment_steps=2000,
                    update_interval=1000,
                    batch_size=4,
                    chunk_size=1,
                    skip_eval=True,
                    resume=True,
                )
            self.assertEqual(resumed["status"], "ok")
            self.assertEqual(resumed["current_environment_steps"], 2000)


if __name__ == "__main__":
    unittest.main()
