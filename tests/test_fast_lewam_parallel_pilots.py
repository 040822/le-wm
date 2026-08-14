import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from source.experiments.fast_lewam_parallel_pilots import (
    build_launch_specs,
    decide_level_one,
    load_pilot_manifest,
    paired_success_delta_pp,
    reshape_action_blocks,
    valid_summary_payload,
    validate_action_training_budget,
)
from source.experiments.planner_transition_replay import (
    PlannerTransitionReplay,
    StageBReplayTrainer,
    fine_tune_stage_b,
    replay_sha256,
    split_train_holdout,
)
from source.experiments.planner_transition_collection import (
    BlockTrajectoryPolicy,
    _slot_candidate_panel,
)
from tests.test_fast_lewam_model import make_model


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "experiments" / "fast_lewam_parallel_pilots.yaml"


class FastLeWAMParallelPilotTests(unittest.TestCase):
    def test_manifest_exposes_the_approved_four_way_schedule(self):
        manifest = load_pilot_manifest(MANIFEST)

        self.assertEqual(
            {
                name: (pilot.gpu, pilot.task, pilot.direction)
                for name, pilot in manifest.pilots.items()
            },
            {
                "reacher_offline_alignment": (0, "reacher", "offline_alignment"),
                "cube_action_tokens": (1, "cube", "action_tokens_25"),
                "pusht_streaming_online": (2, "pusht", "streaming_online"),
                "pusht_offline_alignment": (3, "pusht", "offline_alignment"),
            },
        )
        self.assertEqual(manifest.levels[1].max_epochs, 3)
        self.assertEqual(manifest.levels[1].num_eval, 20)
        self.assertEqual(manifest.levels[2].max_epochs, 5)
        self.assertEqual(manifest.levels[2].num_eval, 50)
        for pilot in manifest.pilots.values():
            self.assertGreaterEqual(
                len(pilot.slots), manifest.levels[2].grounded_slots
            )

    def test_manifest_rejects_a_prohibited_gpu_before_launch(self):
        raw = OmegaConf.load(MANIFEST)
        raw.pilots.reacher_offline_alignment.gpu = 4
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            OmegaConf.save(raw, path)
            with self.assertRaisesRegex(ValueError, "GPU0-GPU3"):
                load_pilot_manifest(path)

    def test_launch_specs_bind_each_worker_to_one_permitted_physical_gpu(self):
        manifest = load_pilot_manifest(MANIFEST)

        specs = build_launch_specs(manifest, level=1)

        self.assertEqual([spec.gpu for spec in specs], [0, 1, 2, 3])
        for spec in specs:
            self.assertEqual(spec.environment["CUDA_VISIBLE_DEVICES"], str(spec.gpu))
            self.assertEqual(spec.environment["MUJOCO_EGL_DEVICE_ID"], str(spec.gpu))
            self.assertIn("--level", spec.command)
            self.assertIn("1", spec.command)
        cube = next(spec for spec in specs if spec.pilot == "cube_action_tokens")
        self.assertIn("--direction", cube.command)
        self.assertIn("action_tokens_25", cube.command)

    def test_action_block_reshape_preserves_physical_action_order(self):
        actions = torch.arange(2 * 5 * 10).reshape(2, 5, 10)

        unstacked = reshape_action_blocks(
            actions, source_block=5, target_block=1
        )
        restored = reshape_action_blocks(
            unstacked, source_block=1, target_block=5
        )

        self.assertEqual(tuple(unstacked.shape), (2, 25, 2))
        self.assertTrue(torch.equal(restored, actions))


    def test_paired_success_delta_uses_percentage_points_and_exact_cohort(self):
        baseline = {
            "success_rate": 82.0,
            "episodes": [{"dataset_episode": 5, "start_step": 9}],
        }
        pilot = {
            "success_rate": 86.0,
            "episodes": [{"dataset_episode": 5, "start_step": 9}],
        }
        self.assertEqual(paired_success_delta_pp(baseline, pilot), (82.0, 86.0, 4.0))
        pilot["episodes"][0]["start_step"] = 10
        with self.assertRaisesRegex(ValueError, "cohorts differ"):
            paired_success_delta_pp(baseline, pilot)

    def test_level_one_gate_requires_non_regression_and_a_mechanism_signal(self):
        thresholds = load_pilot_manifest(MANIFEST).promotion

        self.assertEqual(
            decide_level_one(
                {
                    "success_delta_pp": 0.0,
                    "spearman_delta": 0.12,
                    "regret_relative_delta": -0.02,
                },
                thresholds,
            ),
            "promote",
        )
        self.assertEqual(
            decide_level_one(
                {
                    "success_delta_pp": -10.0,
                    "spearman_delta": 0.20,
                    "regret_relative_delta": -0.20,
                },
                thresholds,
            ),
            "stop",
        )


    def test_replay_fine_tuning_updates_stage_b_but_freezes_visual_encoder(self):
        torch.manual_seed(19)
        model = make_model()
        replay = PlannerTransitionReplay(
            states=torch.randn(8, 3, 8),
            actions=torch.randn(8, 3, 4),
            targets=torch.randn(8, 3, 8),
            sources=("expert",) * 4 + ("planner",) * 4,
            groups=(0, 0, 1, 1, 2, 2, 3, 3),
            identity={"fixture": "true-transitions"},
        )
        frozen_before = model.encoder.proj.weight.detach().clone()
        stage_b_before = model.latent_head.weight.detach().clone()

        report = fine_tune_stage_b(
            model,
            replay,
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            weight_decay=0.0,
            expert_fraction=0.5,
            seed=7,
            device="cpu",
        )

        self.assertTrue(torch.equal(model.encoder.proj.weight, frozen_before))
        self.assertFalse(torch.equal(model.latent_head.weight, stage_b_before))
        self.assertEqual(report["source_updates"], {"expert": 8, "planner": 8})
        self.assertGreater(report["optimizer_steps"], 0)

    def test_trajectory_policy_records_true_action_block_boundaries(self):
        policy = BlockTrajectoryPolicy(
            torch.zeros(2, 25, 3).numpy(), action_block=5
        )
        for step in range(25):
            policy.get_action(
                {"pixels": torch.full((2, 3, 2, 2), step, dtype=torch.uint8)}
            )
        trajectory = policy.finish(
            {"pixels": torch.full((2, 3, 2, 2), 25, dtype=torch.uint8)}
        )

        self.assertEqual(tuple(trajectory.shape), (2, 6, 3, 2, 2))
        self.assertEqual(trajectory[:, :, 0, 0, 0].tolist(), [
            [0, 5, 10, 15, 20, 25],
            [0, 5, 10, 15, 20, 25],
        ])

        history_policy = BlockTrajectoryPolicy(
            torch.zeros(1, 5, 3).numpy(), action_block=5
        )
        history = torch.zeros(1, 1, 2, 2, 3, dtype=torch.uint8)
        history_policy.get_action({"pixels": history})
        history_trajectory = history_policy.finish({"pixels": history})
        self.assertEqual(tuple(history_trajectory.shape), (1, 2, 2, 2, 3))


    def test_replay_hash_covers_every_grounded_field(self):
        base = dict(
            states=torch.zeros(2, 3, 8),
            actions=torch.zeros(2, 3, 4),
            targets=torch.zeros(2, 3, 8),
            sources=("expert", "planner"),
            groups=(0, 0),
            identity={"fixture": "grounded"},
            goal_latents=torch.zeros(2, 8),
            physical_costs=torch.zeros(2),
            successes=torch.zeros(2, dtype=torch.bool),
        )
        original = PlannerTransitionReplay(**base)
        for field, changed in (
            ("goal_latents", torch.ones(2, 8)),
            ("physical_costs", torch.ones(2)),
            ("successes", torch.ones(2, dtype=torch.bool)),
        ):
            variant = PlannerTransitionReplay(**{**base, field: changed})
            self.assertNotEqual(replay_sha256(original), replay_sha256(variant))


    def test_grounded_holdout_is_disjoint_and_excludes_expert_anchors(self):
        replay = PlannerTransitionReplay(
            states=torch.zeros(6, 3, 8),
            actions=torch.zeros(6, 3, 4),
            targets=torch.zeros(6, 3, 8),
            sources=("expert", "planner", "planner") * 2,
            groups=(0, 0, 0, 1, 1, 1),
            identity={"fixture": "split"},
            goal_latents=torch.zeros(6, 8),
            physical_costs=torch.zeros(6),
            successes=torch.zeros(6, dtype=torch.bool),
        )
        train, holdout = split_train_holdout(replay)

        self.assertIn("expert", train.sources)
        self.assertNotIn("expert", holdout.sources)
        self.assertTrue(
            set(train.identity["indices"]).isdisjoint(holdout.identity["indices"])
        )

    def test_replay_trainer_state_resumes_model_optimizer_and_rng(self):
        torch.manual_seed(23)
        replay = PlannerTransitionReplay(
            states=torch.randn(8, 3, 8),
            actions=torch.randn(8, 3, 4),
            targets=torch.randn(8, 3, 8),
            sources=("expert",) * 4 + ("planner",) * 4,
            groups=(0,) * 8,
            identity={"fixture": "resume"},
        )
        first_model = make_model()
        second_model = make_model()
        first = StageBReplayTrainer(
            first_model, learning_rate=1e-3, weight_decay=0.0,
            expert_fraction=0.5, seed=31, device="cpu",
        )
        second = StageBReplayTrainer(
            second_model, learning_rate=1e-3, weight_decay=0.0,
            expert_fraction=0.5, seed=31, device="cpu",
        )
        first.train(replay, epochs=1, batch_size=4)
        second.load_state_dict(first.state_dict())
        first.train(replay, epochs=1, batch_size=4)
        second.train(replay, epochs=1, batch_size=4)

        for left, right in zip(first.model.parameters(), second.model.parameters()):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(first.report(), second.report())

    def test_streaming_ranker_changes_the_next_grounded_candidate_panel(self):
        class Ranker:
            def __init__(self, sign):
                self.sign = sign

            def get_cost(self, _info, actions):
                return self.sign * actions[..., 0, 0]

        candidates = torch.arange(50 * 3 * 4, dtype=torch.float32).reshape(50, 3, 4)
        kwargs = dict(
            detail={"candidates": candidates},
            expert=torch.zeros(3, 4),
            actor_action=torch.ones(3, 4),
            replay_cfg={
                "fixed_non_elites": 10,
                "random_candidates": 10,
                "actor_neighbors": 8,
                "actor_noise_std": 0.1,
            },
            seed=7,
            planning_info={"pixels": torch.zeros(1, 3, 2, 2)},
            device="cpu",
        )
        low_first = _slot_candidate_panel(ranking_model=Ranker(1), **kwargs)
        high_first = _slot_candidate_panel(ranking_model=Ranker(-1), **kwargs)

        self.assertEqual(len(low_first.actions), len(high_first.actions))
        self.assertNotEqual(low_first.sha256, high_first.sha256)


    def test_action_training_budget_requires_equal_examples_and_physical_span(self):
        self.assertEqual(
            validate_action_training_budget(320, 320, 25, 25),
            {
                "control_examples": 320,
                "variant_examples": 320,
                "control_physical_span": 25,
                "variant_physical_span": 25,
            },
        )
        with self.assertRaisesRegex(RuntimeError, "budgets differ"):
            validate_action_training_budget(320, 288, 25, 25)
        with self.assertRaisesRegex(RuntimeError, "budgets differ"):
            validate_action_training_budget(320, 320, 25, 20)

    def test_summary_validity_covers_continuation_artifacts(self):
        manifest = load_pilot_manifest(MANIFEST)
        pilot = manifest.pilots["reacher_offline_alignment"]
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "adaptation_state.pt"
            artifact.write_bytes(b"state-v1")
            import hashlib

            value = {
                "schema_version": 2,
                "status": "ok",
                "manifest_sha256": manifest.sha256,
                "pilot": pilot.name,
                "level": 1,
                "decision": "promote",
                "artifact_identity": {
                    str(artifact): hashlib.sha256(b"state-v1").hexdigest()
                },
            }
            self.assertTrue(
                valid_summary_payload(
                    value, manifest, pilot, 1, require_promote=True
                )
            )
            artifact.write_bytes(b"state-v2")
            self.assertFalse(
                valid_summary_payload(
                    value, manifest, pilot, 1, require_promote=True
                )
            )


if __name__ == "__main__":
    unittest.main()
