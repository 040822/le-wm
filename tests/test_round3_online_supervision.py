import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from source.experiments.round3_online_supervision import (
    FastOnlineSupervisionAdapter,
    SupervisionBatchSampler,
)
from source.experiments.round3_update_curve import (
    make_curve_row,
    optimizer_update_steps,
    validate_curve_rows,
    write_update_curve,
)
from source.experiments.round3_online_runner import run_fixed_replay_training
from source.experiments.round3_grounded_collection import (
    candidate_rms_offsets,
    make_grounded_candidates,
)
from source.experiments.round3_closed_loop import (
    closed_loop_collection_kind,
    run_closed_loop_training,
    validate_closed_loop_schedule,
)
from source.experiments.round3_phase2 import TransitionReplay
from tests.test_fast_lewam_model import make_model


def make_supervision_replay():
    count = 8
    return TransitionReplay(
        observations=torch.randn(count, 4, 3, 8, 8),
        actions=torch.randn(count, 3, 4),
        episode_ids=tuple((index, index, index, index) for index in range(count)),
        start_steps=torch.arange(count),
        sources=("offline",) * 4 + ("online",) * 4,
        data_kinds=("offline",) * 4 + ("continuous",) * 2 + ("grounded",) * 2,
        group_ids=(0, 1, 2, 3, 10, 11, 20, 20),
        model_versions=("e5:epoch10",) * count,
        successes=torch.tensor([False, False, False, False, False, True, True, False]),
        physical_start_costs=torch.full((count,), 2.0),
        physical_terminal_costs=torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 3.0]),
        execution_lengths=torch.full((count,), 15),
        goal_latents=torch.cat((torch.randn(6, 8), torch.ones(2, 8)), dim=0),
    )


class Round3OnlineSupervisionTests(unittest.TestCase):
    def test_closed_loop_schedule_is_four_continuous_updates_then_one_grounded(self):
        schedule = validate_closed_loop_schedule(10)
        self.assertEqual(
            [row["collection_kind"] for row in schedule[:5]],
            ["continuous", "continuous", "continuous", "continuous", "grounded"],
        )
        self.assertEqual(
            sum(row["continuous_environment_steps"] for row in schedule),
            800,
        )
        self.assertEqual(
            sum(row["grounded_environment_steps"] for row in schedule),
            200,
        )
        self.assertEqual(closed_loop_collection_kind(10), "grounded")

    def test_grounded_panel_has_one_anchor_and_three_rms_point03_perturbations(self):
        actor = torch.zeros(5, 10)
        panel, sources = make_grounded_candidates(actor, seed=9)
        self.assertEqual(tuple(panel.shape), (4, 5, 10))
        self.assertEqual(sources[0], "actor_anchor")
        self.assertTrue(torch.equal(panel[0], actor))
        torch.testing.assert_close(
            candidate_rms_offsets(panel)[1:],
            torch.full((3,), 0.03),
            rtol=0.0,
            atol=1e-6,
        )

    def test_sampler_preserves_the_t2_bucket_contract_and_ranking_labels(self):
        replay = make_supervision_replay()
        sampler = SupervisionBatchSampler(replay, seed=11)

        batch = sampler.sample("t2_online_b_mse")
        self.assertEqual(len(batch.b_indices), 64)
        kinds = replay.resolved_data_kinds
        self.assertEqual(sum(kinds[index] == "offline" for index in batch.b_indices), 32)
        self.assertEqual(sum(kinds[index] == "continuous" for index in batch.b_indices), 24)
        self.assertEqual(sum(kinds[index] == "grounded" for index in batch.b_indices), 8)

        ranking = sampler.sample("t3_online_ranking")
        self.assertTrue(ranking.ranking_pairs)
        for preferred, rejected in ranking.ranking_pairs:
            self.assertEqual(replay.resolved_group_ids[preferred], replay.resolved_group_ids[rejected])
            self.assertTrue(
                replay.successes[preferred] > replay.successes[rejected]
                or replay.physical_terminal_costs[preferred]
                < replay.physical_terminal_costs[rejected]
            )

        t1 = sampler.sample("t1_offline_b_mse")
        self.assertEqual(len(t1.offline_anchor_indices), 32)
        self.assertEqual(len(t1.offline_replacement_indices), 32)
        self.assertEqual(t1.counts["b_offline"], 64)

    def test_ranking_rejects_same_group_rows_with_different_snapshot_or_goal(self):
        replay = make_supervision_replay()
        changed_snapshot = TransitionReplay(
            observations=replay.observations,
            actions=replay.actions,
            episode_ids=replay.episode_ids,
            start_steps=replay.start_steps,
            sources=replay.sources,
            model_versions=replay.model_versions,
            data_kinds=replay.data_kinds,
            group_ids=replay.group_ids,
            snapshot_ids=("s0", "s1", "s2", "s3", "s4", "s5", "s6", "other"),
            successes=replay.successes,
            physical_start_costs=replay.physical_start_costs,
            physical_terminal_costs=replay.physical_terminal_costs,
            execution_lengths=replay.execution_lengths,
            goal_latents=replay.goal_latents,
        )
        self.assertEqual(
            SupervisionBatchSampler(changed_snapshot, seed=3)._grounded_pairs(32),
            (),
        )

    def test_t4_update_combines_b_and_stage_a_losses(self):
        torch.manual_seed(4)
        model = make_model()
        before = copy.deepcopy(model.state_dict())
        adapter = FastOnlineSupervisionAdapter(
            model,
            arm="t4_online_offline_a",
            seed=4,
            device="cpu",
        )
        report = adapter.update(make_supervision_replay())

        self.assertEqual(report.optimizer_step, 1)
        self.assertGreaterEqual(report.offline_a or 0.0, 0.0)
        self.assertEqual(report.weights.hindsight, 0.0)
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.state_dict().items()))
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad))

    def test_t5_hindsight_and_t6_distill_keep_separate_sample_contracts(self):
        replay = make_supervision_replay()
        t5 = SupervisionBatchSampler(replay, seed=3).sample("t5_online_hindsight_a")
        t6 = SupervisionBatchSampler(replay, seed=3).sample("t6_online_distill_a")
        self.assertEqual(len(t5.hindsight_indices), 32)
        self.assertEqual(len(t6.distill_indices), 32)
        eligible = set(SupervisionBatchSampler(replay, seed=3)._eligible_distill())
        self.assertTrue(set(t6.distill_indices).issubset(eligible))

    def test_t5_and_t6_updates_use_their_online_supervision_paths(self):
        replay = make_supervision_replay()
        for arm, field in (
            ("t5_online_hindsight_a", "hindsight"),
            ("t6_online_distill_a", "distill"),
        ):
            torch.manual_seed(12)
            adapter = FastOnlineSupervisionAdapter(model=make_model(), arm=arm, seed=12)
            report = adapter.update(replay)
            self.assertEqual(report.optimizer_step, 1)
            self.assertIsNotNone(getattr(report, field))
            self.assertTrue(torch.isfinite(torch.tensor(report.loss)))

    def test_update_curve_uses_optimizer_steps_and_final_cohort_identity(self):
        self.assertEqual(optimizer_update_steps(20, 10), (0, 10, 20))
        rows = tuple(
            make_curve_row(
                optimizer_update_step=step,
                environment_steps=step * 100,
                success_rate=0.5 + step / 100.0,
                episodes=200,
                checkpoint=f"checkpoint-{step}",
                checkpoint_sha256=f"sha-{step}",
                cohort_sha256="final-sha",
            )
            for step in optimizer_update_steps(20, 10)
        )
        validate_curve_rows(
            rows,
            expected_steps=(0, 10, 20),
            cohort_sha256="final-sha",
        )
        with tempfile.TemporaryDirectory() as directory:
            metadata = write_update_curve(
                Path(directory),
                task="reacher",
                experiment="e5",
                arm="t3_online_ranking",
                rows=rows,
                cohort={"kind": "final", "episodes": 200, "canonical_sha256": "final-sha"},
                visible_physical_gpus=(0,),
                max_updates=20,
                interval=10,
                require_complete=True,
            )
            self.assertEqual(metadata["step_unit"], "optimizer_update_step")
            self.assertIn("optimizer_update_step", (Path(directory) / "curve.csv").read_text())

    def test_fixed_replay_runner_publishes_resumable_optimizer_curve(self):
        replay = make_supervision_replay().subset((0, 1, 2, 3))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay_path = root / "replay.pt"
            checkpoint = root / "initial.pt"
            from source.experiments.round3_phase2 import save_transition_replay

            save_transition_replay(replay_path, replay)
            checkpoint.write_bytes(b"initial-checkpoint")

            def loader(path, device):
                return make_model(), path

            def evaluator(task, arm, checkpoint_path, update_step):
                return {
                    "success_rate": 0.5,
                    "episodes": 200,
                    "cohort_sha256": "final-cohort",
                }

            result = run_fixed_replay_training(
                checkpoint=checkpoint,
                replay_path=replay_path,
                output_root=root / "run",
                task="reacher",
                device="cpu",
                max_updates=2,
                curve_interval=1,
                arms=("t1_offline_b_mse",),
                model_loader=loader,
                evaluator=evaluator,
            )

            self.assertEqual(result["status"], "ok")
            state = json.loads(
                (root / "run" / "t1_offline_b_mse" / "run_state.json").read_text()
            )
            self.assertEqual(state["optimizer_update_step"], 2)
            self.assertEqual(
                [row["optimizer_update_step"] for row in state["curve_rows"]],
                [0, 1, 2],
            )
            self.assertEqual(
                json.loads(
                    (root / "run" / "t1_offline_b_mse" / "curve" / "curve.json").read_text()
                )["status"],
                "ok",
            )

            pending = run_fixed_replay_training(
                checkpoint=checkpoint,
                replay_path=replay_path,
                output_root=root / "pending-run",
                task="reacher",
                device="cpu",
                max_updates=1,
                curve_interval=1,
                arms=("t1_offline_b_mse",),
                model_loader=loader,
                evaluator=None,
            )
            self.assertEqual(pending["status"], "training_ok_evaluation_pending")
            self.assertTrue(
                (root / "pending-run" / "t1_offline_b_mse" / "curve" / "curve.csv").is_file()
            )

            frozen = run_fixed_replay_training(
                checkpoint=checkpoint,
                replay_path=replay_path,
                output_root=root / "frozen-run",
                task="reacher",
                device="cpu",
                max_updates=2,
                curve_interval=1,
                arms=("t0_frozen",),
                model_loader=loader,
                evaluator=evaluator,
            )
            self.assertEqual(frozen["status"], "ok")
            frozen_state = json.loads(
                (root / "frozen-run" / "t0_frozen" / "run_state.json").read_text()
            )
            self.assertEqual(frozen_state["updates"], [])
            self.assertEqual(
                [row["environment_steps"] for row in frozen_state["curve_rows"]],
                [0, 0, 0],
            )

    def test_closed_loop_runner_accepts_injected_collection_and_evaluation(self):
        full_replay = make_supervision_replay()
        replay = full_replay.subset((0, 1, 2, 3))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "initial.pt"
            replay_path = root / "offline.pt"
            checkpoint.write_bytes(b"initial-checkpoint")
            from source.experiments.round3_phase2 import save_transition_replay

            save_transition_replay(replay_path, replay)

            def loader(path, device):
                return make_model(), path

            def collector(arm, step, kind, model, shard_path):
                selected = full_replay.subset((4, 5))
                return selected, {"status": "ok", "kind": kind, "environment_steps": 100}

            def evaluator(task, arm, checkpoint_path, update_step):
                return {
                    "success_rate": 0.25,
                    "episodes": 200,
                    "cohort_sha256": "final-cohort",
                }

            result = run_closed_loop_training(
                checkpoint=checkpoint,
                offline_replay_path=replay_path,
                output_root=root / "closed-loop",
                task="reacher",
                max_updates=2,
                curve_interval=1,
                arms=("t1_offline_b_mse", "t2_online_b_mse"),
                model_loader=loader,
                collector=collector,
                evaluator=evaluator,
            )
            self.assertEqual(result["status"], "ok")
            state = json.loads(
                (root / "closed-loop" / "t2_online_b_mse" / "run_state.json").read_text()
            )
            self.assertEqual(state["optimizer_update_step"], 2)
            self.assertEqual(state["updates"][0]["collection_kind"], "continuous")
            self.assertEqual(state["curve_rows"][-1]["episodes"], 200)

    def test_fixed_replay_runner_executes_all_supervision_arms(self):
        replay = make_supervision_replay()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "initial.pt"
            replay_path = root / "replay.pt"
            checkpoint.write_bytes(b"initial-checkpoint")
            from source.experiments.round3_phase2 import save_transition_replay

            save_transition_replay(replay_path, replay)

            def loader(path, device):
                return make_model(), path

            def evaluator(task, arm, checkpoint_path, update_step):
                return {
                    "success_rate": 0.5,
                    "episodes": 200,
                    "cohort_sha256": "final-cohort",
                }

            result = run_fixed_replay_training(
                checkpoint=checkpoint,
                replay_path=replay_path,
                output_root=root / "all-arms",
                task="pusht",
                max_updates=1,
                curve_interval=1,
                model_loader=loader,
                evaluator=evaluator,
            )
            self.assertEqual(set(result["arm_results"]), {
                "t0_frozen",
                "t1_offline_b_mse",
                "t2_online_b_mse",
                "t3_online_ranking",
                "t4_online_offline_a",
                "t5_online_hindsight_a",
                "t6_online_distill_a",
            })
            for arm in result["arm_results"]:
                self.assertEqual(
                    json.loads(
                        (root / "all-arms" / arm / "run_state.json").read_text()
                    )["status"],
                    "ok",
                )


if __name__ == "__main__":
    unittest.main()
