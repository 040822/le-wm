import unittest

import gymnasium as gym
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from source.policy.round4 import (
    Round4BestOfNPolicy,
    make_round4_policy,
    score_candidates_in_chunks,
)
from source.policy.fast_lewam_eval import ProjectedCEMSolver
from source.common.cvpr_table3_bridge import make_state_indexed_schedules
from omegaconf import OmegaConf
from tests.test_round4_model import make_round4_model


class Round4PlanningTests(unittest.TestCase):
    def test_action_flow_steps_override_reaches_p0_p2_and_p3(self):
        model = make_round4_model().eval()
        plan_config = {
            "horizon": 5,
            "receding_horizon": 5,
            "action_block": 1,
        }

        p0 = make_round4_policy(
            model,
            mode="P0",
            plan_config=plan_config,
            device="cpu",
            action_flow_steps=3,
        )
        self.assertEqual(p0.inference_steps, 3)

        solver_cfg = OmegaConf.create(
            {
                "_target_": "stable_worldmodel.solver.CEMSolver",
                "model": "???",
                "batch_size": 1,
                "num_samples": 2,
                "n_steps": 1,
                "topk": 1,
                "var_scale": 1.0,
                "device": "cpu",
            }
        )
        p2 = make_round4_policy(
            model,
            mode="P2",
            solver_cfg=solver_cfg,
            plan_config=plan_config,
            device="cpu",
            action_flow_steps=3,
        )
        self.assertEqual(p2.solver.model.inference_steps, 3)

        p3 = make_round4_policy(
            model,
            mode="P3",
            plan_config=plan_config,
            device="cpu",
            flow_steps=16,
            action_flow_steps=3,
        )
        self.assertEqual(p3.flow_steps, 3)
        self.assertEqual(p3.action_flow_steps, 3)

        heun = make_round4_policy(
            model,
            mode="P3",
            plan_config=plan_config,
            device="cpu",
            action_flow_steps=3,
            action_flow_integrator="heun",
        )
        self.assertEqual(heun.action_flow_integrator, "heun")
        self.assertEqual(heun.metadata()["action_flow_integrator"], "heun")

    def test_action_flow_steps_defaults_preserve_canonical_round4_values(self):
        model = make_round4_model().eval()
        plan_config = {
            "horizon": 5,
            "receding_horizon": 5,
            "action_block": 1,
        }
        p0 = make_round4_policy(
            model, mode="P0", plan_config=plan_config, device="cpu"
        )
        p3 = make_round4_policy(
            model, mode="P3", plan_config=plan_config, device="cpu"
        )
        self.assertEqual(p0.inference_steps, 16)
        self.assertEqual(p3.flow_steps, 16)
        self.assertEqual(p3.action_flow_steps, 16)

    def test_actor_warm_start_scale_is_optional_and_recorded_on_p2_view(self):
        model = make_round4_model().eval()
        plan_config = {
            "horizon": 5,
            "receding_horizon": 5,
            "action_block": 1,
        }
        solver_cfg = OmegaConf.create(
            {
                "_target_": "stable_worldmodel.solver.CEMSolver",
                "model": "???",
                "batch_size": 1,
                "num_samples": 2,
                "n_steps": 1,
                "topk": 1,
                "var_scale": 1.0,
                "device": "cpu",
            }
        )
        policy = make_round4_policy(
            model,
            mode="P2",
            solver_cfg=solver_cfg,
            plan_config=plan_config,
            device="cpu",
            action_flow_steps=3,
            actor_warm_start_scale=0.5,
        )
        self.assertEqual(policy.actor_warm_start_scale, 0.5)
        self.assertEqual(policy.solver.model.action_scale, 0.5)

    def test_action_bound_variants_are_available_for_p0_p1_p2_and_p3(self):
        model = make_round4_model().eval()
        plan_config = {
            "horizon": 5,
            "receding_horizon": 5,
            "action_block": 1,
        }
        process = {
            "action": StandardScaler().fit(
                np.array(
                    [
                        [-0.5, -0.4, -0.3, -0.2],
                        [0.0, 0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6, 0.7],
                    ]
                )
            )
        }
        p0 = make_round4_policy(
            model,
            mode="P0",
            plan_config=plan_config,
            process=process,
            device="cpu",
            action_bound_mode="clip",
        )
        self.assertEqual(p0.action_bound_mode, "clip")

        solver_cfg = OmegaConf.create(
            {
                "_target_": "stable_worldmodel.solver.CEMSolver",
                "model": "???",
                "batch_size": 1,
                "num_samples": 2,
                "n_steps": 1,
                "topk": 1,
                "var_scale": 1.0,
                "device": "cpu",
            }
        )
        p2 = make_round4_policy(
            model,
            mode="P2",
            solver_cfg=solver_cfg,
            plan_config=plan_config,
            process=process,
            device="cpu",
            action_flow_steps=1,
            action_bound_mode="candidate_clip",
        )
        self.assertEqual(p2.action_bound_mode, "candidate_clip")
        self.assertIsInstance(p2.solver, ProjectedCEMSolver)
        self.assertEqual(p2.solver.model.action_projection, "clip")

        p1 = make_round4_policy(
            model,
            mode="P1",
            solver_cfg=solver_cfg,
            plan_config=plan_config,
            process=process,
            device="cpu",
            action_bound_mode="candidate_clip",
        )
        self.assertEqual(p1.action_bound_mode, "candidate_clip")
        self.assertIsInstance(p1.solver, ProjectedCEMSolver)
        self.assertEqual(p1.solver.model.__class__.__name__, "StageBModelView")

        p3 = make_round4_policy(
            model,
            mode="P3",
            plan_config=plan_config,
            process=process,
            device="cpu",
            action_flow_steps=1,
            action_bound_mode="clip",
        )
        self.assertEqual(p3.action_bound_mode, "clip")

        p3_scale = make_round4_policy(
            model,
            mode="P3",
            plan_config=plan_config,
            process=process,
            device="cpu",
            action_flow_steps=1,
            action_bound_mode="global_scale",
        )
        self.assertEqual(p3_scale.action_bound_mode, "global_scale")

    def test_p3_projects_candidates_before_verifier_with_real_normalized_bounds(self):
        class FakeEnv:
            num_envs = 1
            single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,))
            action_space = gym.spaces.Box(-1.0, 1.0, shape=(1, 4))

        process = {
            "action": StandardScaler().fit(
                np.array(
                    [
                        [-0.5, -0.5, -0.5, -0.5],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.5, 0.5, 0.5, 0.5],
                    ]
                )
            )
        }
        policy = make_round4_policy(
            make_round4_model().eval(),
            mode="P3",
            plan_config={
                "horizon": 5,
                "receding_horizon": 5,
                "action_block": 1,
            },
            process=process,
            device="cpu",
            action_flow_steps=1,
            action_bound_mode="clip",
        )
        policy.set_env(FakeEnv())

        raw = torch.full((1, 4, 5, 4), 4.0)
        projected, summary = policy._project_candidates(raw)

        self.assertIsNotNone(policy.action_bounds)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["mode"], "clip")
        self.assertGreater(summary["raw_candidate_violation_fraction"], 0.0)
        self.assertEqual(summary["projected_candidate_violation_fraction"], 0.0)
        self.assertTrue(
            torch.all(
                projected
                <= torch.as_tensor(
                    policy.action_bounds.for_action_dim(4)[1]
                ).reshape(1, 1, 1, 4)
                + 1e-6
            )
        )

    def test_p0_clip_configures_bounds_and_keeps_environment_action_legal(self):
        class FakeEnv:
            num_envs = 1
            single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,))
            action_space = gym.spaces.Box(-1.0, 1.0, shape=(1, 4))

        model = make_round4_model().eval()
        process = {
            "action": StandardScaler().fit(
                np.array(
                    [
                        [-0.2, -0.2, -0.2, -0.2],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.2, 0.2, 0.2, 0.2],
                    ]
                )
            )
        }
        policy = make_round4_policy(
            model,
            mode="P0",
            plan_config={
                "horizon": 5,
                "receding_horizon": 5,
                "action_block": 1,
            },
            process=process,
            device="cpu",
            action_flow_steps=1,
            action_bound_mode="clip",
        )
        policy.set_env(FakeEnv())

        action = policy.get_action(
            {
                "pixels": torch.randn(1, 1, 3, 8, 8),
                "goal": torch.randn(1, 1, 3, 8, 8),
            }
        )

        self.assertIsNotNone(policy.action_bounds)
        self.assertTrue(np.isfinite(action).all())
        self.assertTrue((action <= 1.0 + 1e-6).all())
        self.assertTrue((action >= -1.0 - 1e-6).all())
        self.assertEqual(policy.planning_events[-1]["action_bound_mode"], "clip")

    def test_p2_candidate_clip_propagates_bounds_to_solver_and_actor(self):
        class FakeEnv:
            num_envs = 1
            single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,))
            action_space = gym.spaces.Box(-1.0, 1.0, shape=(1, 4))

        model = make_round4_model().eval()
        process = {
            "action": StandardScaler().fit(
                np.array(
                    [
                        [-0.2, -0.2, -0.2, -0.2],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.2, 0.2, 0.2, 0.2],
                    ]
                )
            )
        }
        solver_cfg = OmegaConf.create(
            {
                "_target_": "stable_worldmodel.solver.CEMSolver",
                "model": "???",
                "batch_size": 1,
                "num_samples": 2,
                "n_steps": 1,
                "topk": 1,
                "var_scale": 1.0,
                "device": "cpu",
            }
        )
        policy = make_round4_policy(
            model,
            mode="P2",
            solver_cfg=solver_cfg,
            plan_config={
                "horizon": 5,
                "receding_horizon": 5,
                "action_block": 1,
            },
            process=process,
            device="cpu",
            action_flow_steps=1,
            action_bound_mode="candidate_clip",
        )
        policy.set_env(FakeEnv())

        self.assertIsNotNone(policy.action_bounds)
        self.assertIs(policy.solver.action_bounds, policy.action_bounds)
        self.assertIs(policy.solver.model.action_bounds, policy.action_bounds)

    def test_p2_candidate_clip_executes_a_bounded_cem_plan(self):
        class FakeEnv:
            num_envs = 1
            single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,))
            action_space = gym.spaces.Box(-1.0, 1.0, shape=(1, 4))

        model = make_round4_model().eval()
        process = {
            "action": StandardScaler().fit(
                np.array(
                    [
                        [-0.2, -0.2, -0.2, -0.2],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.2, 0.2, 0.2, 0.2],
                    ]
                )
            )
        }
        solver_cfg = OmegaConf.create(
            {
                "_target_": "stable_worldmodel.solver.CEMSolver",
                "model": "???",
                "batch_size": 1,
                "num_samples": 2,
                "n_steps": 1,
                "topk": 1,
                "var_scale": 1.0,
                "device": "cpu",
            }
        )
        policy = make_round4_policy(
            model,
            mode="P2",
            solver_cfg=solver_cfg,
            plan_config={
                "horizon": 5,
                "receding_horizon": 5,
                "action_block": 1,
            },
            process=process,
            device="cpu",
            action_flow_steps=1,
            action_bound_mode="candidate_clip",
        )
        policy.set_env(FakeEnv())

        action = policy.get_action(
            {
                "pixels": torch.randn(1, 1, 3, 8, 8),
                "goal": torch.randn(1, 1, 3, 8, 8),
            }
        )

        self.assertEqual(action.shape, (1, 4))
        self.assertTrue(np.isfinite(action).all())
        self.assertTrue((action <= 1.0 + 1e-6).all())
        self.assertTrue((action >= -1.0 - 1e-6).all())
        self.assertIsNotNone(policy.solver.last_action_bound_projection)

    def test_candidate_split_does_not_change_verifier_argmin(self):
        model = make_round4_model().eval()
        policy = Round4BestOfNPolicy(
            model,
            proposal_source="latent",
            num_candidates=6,
            flow_steps=2,
            action_block=1,
            receding_horizon_blocks=5,
            solver_batch_size=1,
        )
        z_start = torch.randn(2, 8)
        z_goal = torch.randn(2, 8)
        candidates = torch.randn(2, 6, 5, 4)
        whole = policy.score_candidates(z_start, z_goal, candidates)
        split = score_candidates_in_chunks(
            lambda start, goal, actions: model.get_cost_from_latents(start, goal, actions),
            z_start,
            z_goal,
            candidates,
            solver_batch_size=2,
            candidate_batch_size=2,
        )
        torch.testing.assert_close(whole, split)
        self.assertTrue(torch.equal(whole.argmin(dim=1), split.argmin(dim=1)))

    def test_latent_proposal_is_decoded_then_verified_only_with_actions(self):
        model = make_round4_model().eval()
        policy = Round4BestOfNPolicy(
            model,
            proposal_source="latent",
            num_candidates=4,
            flow_steps=2,
            action_block=1,
            receding_horizon_blocks=5,
        )
        z_start = torch.randn(1, 8)
        z_goal = torch.randn(1, 8)
        paths = model.sample_latent_paths(
            z_start,
            z_goal,
            num_samples=4,
            num_steps=2,
            generator=torch.Generator().manual_seed(4),
        )
        actions = model.decode_latent_paths(paths)
        costs = policy.score_candidates(z_start, z_goal, actions)
        self.assertEqual(actions.shape, (1, 4, 5, 4))
        self.assertEqual(costs.shape, (1, 4))
        self.assertTrue(torch.isfinite(costs).all())

    def test_first_candidate_mode_never_needs_a_verifier(self):
        model = make_round4_model().eval()
        policy = Round4BestOfNPolicy(
            model,
            proposal_source="latent",
            num_candidates=4,
            flow_steps=2,
            action_block=1,
            receding_horizon_blocks=5,
            verifier="none",
            selection_rule="first",
        )
        self.assertEqual(policy.metadata()["selection_rule"], "first")
        self.assertEqual(policy.metadata()["verifier"], "none")

    def test_random_candidate_mode_uses_independent_seeded_selection_rng(self):
        model = make_round4_model().eval()
        first = Round4BestOfNPolicy(
            model,
            proposal_source="action",
            num_candidates=64,
            flow_steps=2,
            action_block=1,
            receding_horizon_blocks=5,
            verifier="none",
            selection_rule="random",
            seed=42,
        )
        second = Round4BestOfNPolicy(
            make_round4_model().eval(),
            proposal_source="action",
            num_candidates=64,
            flow_steps=2,
            action_block=1,
            receding_horizon_blocks=5,
            verifier="none",
            selection_rule="random",
            seed=42,
        )
        device = next(model.parameters()).device
        proposal_rng_first = first._generator(device)
        proposal_rng_second = second._generator(device)
        selected_first = torch.randint(
            64, (10,), generator=first._selection_generator(device), device=device
        )
        selected_second = torch.randint(
            64, (10,), generator=second._selection_generator(device), device=device
        )
        self.assertTrue(torch.equal(selected_first, selected_second))
        proposal_draw_first = torch.randn(10, generator=proposal_rng_first, device=device)
        proposal_draw_second = torch.randn(10, generator=proposal_rng_second, device=device)
        self.assertTrue(torch.equal(proposal_draw_first, proposal_draw_second))

    def test_state_indexed_schedule_replays_same_pool_and_random_choice(self):
        noise_schedule, selection_schedule, metadata = make_state_indexed_schedules(
            task="cube", evaluation_seed=42, policy_seed=20042
        )
        keys = ((3, 0), (7, 1))
        noise = noise_schedule(
            keys,
            num_candidates=4,
            action_horizon=5,
            action_dim=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        repeated_noise = noise_schedule(
            keys,
            num_candidates=4,
            action_horizon=5,
            action_dim=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(noise, repeated_noise))
        self.assertFalse(torch.equal(noise[0], noise[1]))
        self.assertEqual(
            selection_schedule(keys, num_candidates=64, device="cpu").tolist(),
            selection_schedule(keys, num_candidates=64, device="cpu").tolist(),
        )
        self.assertEqual(metadata["version"], "cvpr_table3_state_indexed_schedule_v1")

        policy = Round4BestOfNPolicy(
            make_round4_model().eval(),
            proposal_source="action",
            num_candidates=4,
            flow_steps=2,
            action_block=1,
            receding_horizon_blocks=5,
            candidate_noise_schedule=noise_schedule,
        )
        policy._active_schedule_keys = ((3, 0),)
        z_start, z_goal = torch.randn(1, 8), torch.randn(1, 8)
        first, _ = policy._propose_actions(
            z_start, z_goal, torch.Generator().manual_seed(1)
        )
        second, _ = policy._propose_actions(
            z_start, z_goal, torch.Generator().manual_seed(999)
        )
        self.assertTrue(torch.equal(first, second))

    def test_state_indexed_selection_tracks_each_slots_replan_count(self):
        class FakeEnv:
            num_envs = 2
            single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,))
            action_space = gym.spaces.Box(-1.0, 1.0, shape=(2, 4))

        noise_schedule, selection_schedule, _metadata = make_state_indexed_schedules(
            task="reacher", evaluation_seed=100, policy_seed=20100
        )
        policy = Round4BestOfNPolicy(
            make_round4_model().eval(),
            proposal_source="action",
            num_candidates=8,
            flow_steps=2,
            action_block=1,
            receding_horizon_blocks=1,
            verifier="none",
            selection_rule="random",
            candidate_noise_schedule=noise_schedule,
            selection_index_schedule=selection_schedule,
        )
        policy.set_env(FakeEnv())
        info = {
            "pixels": torch.randn(2, 1, 3, 8, 8),
            "goal": torch.randn(2, 1, 3, 8, 8),
            "terminated": np.asarray([False, False]),
        }
        policy.get_action(info)
        first = policy.planning_events[-1]
        self.assertEqual(first["state_indexed_schedule_keys"], [[0, 0], [1, 0]])
        expected_first = selection_schedule(
            ((0, 0), (1, 0)), num_candidates=8, device="cpu"
        ).tolist()
        self.assertEqual(first["selected_indices"], expected_first)

        policy.get_action(info)
        second = policy.planning_events[-1]
        self.assertEqual(second["state_indexed_schedule_keys"], [[0, 1], [1, 1]])
        expected_second = selection_schedule(
            ((0, 1), (1, 1)), num_candidates=8, device="cpu"
        ).tolist()
        self.assertEqual(second["selected_indices"], expected_second)

    def test_policy_executes_one_receding_action_block_and_records_plan_metadata(self):
        class FakeEnv:
            num_envs = 1
            single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,))
            action_space = gym.spaces.Box(-1.0, 1.0, shape=(1, 4))

        policy = Round4BestOfNPolicy(
            make_round4_model().eval(),
            proposal_source="latent",
            num_candidates=4,
            flow_steps=2,
            action_block=1,
            receding_horizon_blocks=1,
        )
        policy.set_env(FakeEnv())
        action = policy.get_action(
            {
                "pixels": torch.randn(1, 1, 3, 8, 8),
                "goal": torch.randn(1, 1, 3, 8, 8),
            }
        )
        self.assertEqual(action.shape, (1, 4))
        self.assertTrue(np.isfinite(action).all())
        self.assertEqual(policy.planning_events[-1]["proposal_source"], "latent")
        self.assertEqual(policy.planning_events[-1]["candidate_count"], 4)
        self.assertGreater(policy.planning_events[-1]["forward_count"], 0)

    def test_guidance_mode_is_threaded_and_validated(self):
        model = make_round4_model().eval()
        plan_config = {"horizon": 5, "receding_horizon": 5, "action_block": 1}
        p0 = make_round4_policy(
            model,
            mode="P0",
            plan_config=plan_config,
            device="cpu",
            action_flow_steps=2,
            guidance_mode="guided_flow",
            guidance_last_steps=2,
            guidance_inner_steps=2,
        )
        self.assertEqual(p0.guidance_mode, "guided_flow")
        self.assertEqual(p0.guidance_last_steps, 2)

        p3 = make_round4_policy(
            model,
            mode="P3",
            plan_config=plan_config,
            device="cpu",
            action_flow_steps=2,
            candidate_count=4,
            guidance_mode="post_opt_refine",
            proposal_chunk_size=3,
        )
        self.assertEqual(p3.guidance_mode, "post_opt_refine")
        self.assertEqual(p3.proposal_chunk_size, 3)
        self.assertEqual(p3.metadata()["guidance_mode"], "post_opt_refine")

        with self.assertRaises(ValueError):
            make_round4_policy(
                model,
                mode="P1",
                plan_config=plan_config,
                device="cpu",
                guidance_mode="guided_flow",
            )
        with self.assertRaises(ValueError):
            make_round4_policy(
                model,
                mode="P0",
                plan_config=plan_config,
                device="cpu",
                guidance_mode="post_opt_refine",
            )
        with self.assertRaises(ValueError):
            make_round4_policy(
                model,
                mode="P3",
                plan_config=plan_config,
                device="cpu",
                candidate_count=4,
                guidance_mode="unknown",
            )

    def test_p3_guidance_per_candidate_runs_and_counts_backwards(self):
        class FakeEnv:
            num_envs = 1
            single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,))
            action_space = gym.spaces.Box(-1.0, 1.0, shape=(1, 4))

        def run(guidance):
            policy = make_round4_policy(
                make_round4_model().eval(),
                mode="P3",
                plan_config={"horizon": 5, "receding_horizon": 5, "action_block": 1},
                device="cpu",
                action_flow_steps=2,
                candidate_count=4,
                candidate_batch_size=4,
                guidance_mode=guidance,
                guidance_last_steps=2,
                guidance_inner_steps=2,
                proposal_chunk_size=3,
            )
            policy.set_env(FakeEnv())
            action = policy.get_action(
                {
                    "pixels": torch.randn(1, 1, 3, 8, 8),
                    "goal": torch.randn(1, 1, 3, 8, 8),
                }
            )
            self.assertTrue(np.isfinite(action).all())
            return policy.planning_events[-1]

        guided = run("guided_flow")
        self.assertEqual(guided["guidance_mode"], "guided_flow")
        self.assertEqual(guided["guidance_backward_count"], 2 * 2 * 4)
        post = run("post_opt")
        self.assertEqual(post["guidance_backward_count"], 2 * 4)
        refine = run("post_opt_refine")
        self.assertEqual(refine["guidance_backward_count"], 2)
        self.assertEqual(refine["guidance_mode"], "post_opt_refine")

    def test_p3_post_opt_refine_calls_model_refinement(self):
        class FakeEnv:
            num_envs = 1
            single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,))
            action_space = gym.spaces.Box(-1.0, 1.0, shape=(1, 4))

        model = make_round4_model().eval()
        calls = {}
        original = model.post_optimize_actions

        def spy(z0, goal_latent, actions, **kwargs):
            calls["batch"] = int(actions.shape[0])
            calls["kwargs"] = kwargs
            return original(z0, goal_latent, actions, **kwargs)

        model.post_optimize_actions = spy
        policy = make_round4_policy(
            model,
            mode="P3",
            plan_config={"horizon": 5, "receding_horizon": 5, "action_block": 1},
            device="cpu",
            action_flow_steps=2,
            candidate_count=4,
            guidance_mode="post_opt_refine",
            guidance_inner_steps=2,
        )
        policy.set_env(FakeEnv())
        policy.get_action(
            {
                "pixels": torch.randn(1, 1, 3, 8, 8),
                "goal": torch.randn(1, 1, 3, 8, 8),
            }
        )
        self.assertEqual(calls["batch"], 1)
        self.assertEqual(calls["kwargs"]["inner_steps"], 2)

    def test_p3_diagnostic_callback_exposes_candidate_pool_context(self):
        class FakeEnv:
            num_envs = 1
            single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,))
            action_space = gym.spaces.Box(-1.0, 1.0, shape=(1, 4))

        events = []
        policy = make_round4_policy(
            make_round4_model().eval(),
            mode="P3",
            plan_config={"horizon": 5, "receding_horizon": 5, "action_block": 1},
            device="cpu",
            action_flow_steps=2,
            candidate_count=4,
            diagnostic_callback=events.append,
        )
        policy.set_env(FakeEnv())
        policy.get_action(
            {
                "pixels": torch.randn(1, 1, 3, 8, 8),
                "goal": torch.randn(1, 1, 3, 8, 8),
            }
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(tuple(events[0]["candidates"].shape), (1, 4, 5, 4))
        self.assertEqual(tuple(events[0]["costs"].shape), (1, 4))
        self.assertEqual(tuple(events[0]["z_start"].shape), (1, 8))
        self.assertEqual(tuple(events[0]["predicted_latents"].shape[:3]), (1, 4, 5))

    def test_p3_uses_separate_verifier_for_costs_and_predictions(self):
        class FakeEnv:
            num_envs = 1
            single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,))
            action_space = gym.spaces.Box(-1.0, 1.0, shape=(1, 4))

        proposal = make_round4_model().eval()
        verifier = make_round4_model().eval()
        with torch.no_grad():
            verifier.latent_head.bias.add_(1.0)
        events = []
        policy = make_round4_policy(
            proposal,
            verifier_policy_or_model=verifier,
            mode="P3",
            plan_config={"horizon": 5, "receding_horizon": 5, "action_block": 1},
            device="cpu",
            action_flow_steps=2,
            candidate_count=4,
            diagnostic_callback=events.append,
        )
        policy.set_env(FakeEnv())
        policy.get_action(
            {
                "pixels": torch.randn(1, 1, 3, 8, 8),
                "goal": torch.randn(1, 1, 3, 8, 8),
            }
        )

        event = events[0]
        self.assertTrue(policy.metadata()["separate_verifier_model"])
        expected_costs = verifier.get_cost_from_latents(
            event["verifier_z_start"],
            event["verifier_z_goal"],
            event["candidates"],
        )
        torch.testing.assert_close(event["costs"], expected_costs)
        expected_latents = verifier(
            event["verifier_z_start"][:, None]
            .expand(-1, event["candidates"].shape[1], -1)
            .reshape(-1, verifier.latent_dim),
            event["candidates"].reshape(-1, verifier.action_horizon, verifier.action_dim),
            torch.ones(event["candidates"].shape[1]),
            mode="stage_b",
        )["predicted_latents"].reshape(1, 4, 5, -1)
        torch.testing.assert_close(event["predicted_latents"], expected_latents)

    def test_p0_guidance_callback_pairs_same_initial_noise(self):
        class FakeEnv:
            num_envs = 1
            single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,))
            action_space = gym.spaces.Box(-1.0, 1.0, shape=(1, 4))

        events = []
        policy = make_round4_policy(
            make_round4_model().eval(),
            mode="P0",
            plan_config={"horizon": 5, "receding_horizon": 5, "action_block": 1},
            device="cpu",
            action_flow_steps=2,
            guidance_mode="post_opt",
            guidance_inner_steps=1,
            diagnostic_callback=events.append,
        )
        policy.set_env(FakeEnv())
        policy.get_action(
            {
                "pixels": torch.randn(1, 1, 3, 8, 8),
                "goal": torch.randn(1, 1, 3, 8, 8),
            }
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(tuple(events[0]["unguided_actions"].shape), (1, 5, 4))
        self.assertEqual(tuple(events[0]["guided_actions"].shape), (1, 5, 4))
        self.assertEqual(tuple(events[0]["predicted_cost_before"].shape), (1,))
        self.assertEqual(tuple(events[0]["predicted_cost_after"].shape), (1,))


if __name__ == "__main__":
    unittest.main()
