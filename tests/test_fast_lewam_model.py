import unittest
from types import SimpleNamespace

import torch
from torch import nn

from source.model.fast_lewam.jepa import FastLeWAM
from source.model.fast_lewam.modules import (
    block_causal_attention_mask,
    causal_attention_mask,
    stage_a_attention_mask,
    terminal_full_attention_mask,
)


class TinyEncoder(nn.Module):
    """Small encoder adapter used to exercise the public FastLeWAM interface."""

    def __init__(self, latent_dim):
        super().__init__()
        self.proj = nn.Linear(3, latent_dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        pooled = pixels.mean(dim=(-2, -1))
        cls = self.proj(pooled)
        return SimpleNamespace(last_hidden_state=cls[:, None])


class RecordingPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, tokens, condition, attention_mask=None):
        self.inputs.append(tokens.detach().clone())
        return tokens


def make_model(
    action_horizon=3,
    task_condition_dim=None,
    stage_a_goal_injection="none",
    token_encoding="legacy",
    stage_b_dynamics="parallel_prefix",
    stage_b_attention_mode="strict_causal",
):
    torch.manual_seed(7)
    return FastLeWAM(
        encoder=TinyEncoder(latent_dim=8),
        projector=nn.Identity(),
        latent_dim=8,
        action_dim=4,
        action_horizon=action_horizon,
        latent_head_dim=12,
        latent_head_layers=2,
        heads=3,
        mlp_dim=24,
        dropout=0.0,
        task_condition_dim=task_condition_dim,
        stage_a_goal_injection=stage_a_goal_injection,
        token_encoding=token_encoding,
        stage_b_dynamics=stage_b_dynamics,
        stage_b_attention_mode=stage_b_attention_mode,
    )


class FastLeWAMModelTests(unittest.TestCase):
    def test_serial_and_parallel_stage_b_have_identical_parameter_schema(self):
        parallel = make_model(stage_b_dynamics="parallel_prefix")
        serial = make_model(stage_b_dynamics="serial_one_step")

        self.assertEqual(set(parallel.state_dict()), set(serial.state_dict()))
        self.assertEqual(
            sum(parameter.numel() for parameter in parallel.parameters()),
            sum(parameter.numel() for parameter in serial.parameters()),
        )
        incompatible = serial.load_state_dict(parallel.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_serial_stage_b_rollout_matches_repeated_one_step_predictions(self):
        model = make_model(stage_b_dynamics="serial_one_step").eval()
        z0 = torch.randn(2, 8)
        actions = torch.randn(2, 3, 4)

        rollout = model(
            z0,
            actions,
            torch.ones(2),
            mode="stage_b",
        )["predicted_latents"]
        manual = []
        state = z0
        for index in range(actions.shape[1]):
            state = model.predict_one_step(state, actions[:, index])
            manual.append(state)

        self.assertTrue(torch.equal(rollout, torch.stack(manual, dim=1)))

    def test_serial_candidate_cost_uses_fifth_recursive_latent(self):
        model = make_model(
            action_horizon=5,
            stage_b_dynamics="serial_one_step",
        ).eval()
        z0 = torch.randn(1, 8)
        goal = torch.randn(1, 8)
        candidates = torch.randn(1, 4, 5, 4)

        cost = model.get_cost_from_latents(z0, goal, candidates)
        flat_z0 = z0.expand(4, -1)
        rollout = model(
            flat_z0,
            candidates.reshape(4, 5, 4),
            torch.ones(4),
            mode="stage_b",
        )["predicted_latents"]
        expected = (rollout[:, -1] - goal.expand(4, -1)).square().mean(dim=-1)

        self.assertTrue(torch.equal(cost, expected.reshape(1, 4)))

    def test_physical_time_type_encoding_aligns_stage_a_and_b_tokens(self):
        model = make_model(
            stage_a_goal_injection="token",
            token_encoding="physical_time_type",
        ).eval()
        recorder = RecordingPredictor()
        model.predictor = recorder
        z0 = torch.randn(2, 8)
        goal = torch.randn(2, 8)
        actions = torch.randn(2, 3, 4)
        timestep = torch.tensor([0.2, 0.8])

        model(
            z0,
            actions,
            timestep,
            mode="stage_a",
            goal_latent=goal,
        )
        model(z0, actions, torch.ones(2), mode="stage_b")
        stage_a_tokens, stage_b_tokens = recorder.inputs

        state_type = model.type_embeddings.weight[model.STATE_TYPE]
        goal_type = model.type_embeddings.weight[model.GOAL_TYPE]
        action_type = model.type_embeddings.weight[model.ACTION_TYPE]
        query_type = model.type_embeddings.weight[model.QUERY_TYPE]
        expected_actions = (
            model.action_input(actions)
            + model.time_positions[:, :3]
            + action_type
        )
        expected_queries = (
            model.query_content
            + model.time_positions[:, 1:4]
            + query_type
        )

        self.assertTrue(torch.allclose(stage_a_tokens[:, 0], (
            model.latent_input(z0) + model.time_positions[:, 0] + state_type
        )))
        self.assertTrue(torch.allclose(stage_a_tokens[:, 1], (
            model.latent_input(goal) + model.time_positions[:, 3] + goal_type
        )))
        self.assertTrue(torch.allclose(stage_a_tokens[:, 2:], expected_actions))
        self.assertTrue(torch.allclose(stage_b_tokens[:, 0], stage_a_tokens[:, 0]))
        self.assertTrue(torch.allclose(stage_b_tokens[:, 1::2], expected_actions))
        self.assertTrue(torch.allclose(stage_b_tokens[:, 2::2], expected_queries))

    def test_legacy_encoding_keeps_checkpoint_parameter_schema(self):
        original = make_model(stage_a_goal_injection="token")
        restored = make_model(
            stage_a_goal_injection="token",
            token_encoding="legacy",
        )

        incompatible = restored.load_state_dict(original.state_dict(), strict=True)
        keys = set(restored.state_dict())

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertFalse(any(key.startswith("time_positions") for key in keys))
        self.assertFalse(any(key.startswith("type_embeddings") for key in keys))
        self.assertFalse(any(key.startswith("query_content") for key in keys))

    def test_physical_encoding_preserves_causal_prefix_predictions(self):
        model = make_model(token_encoding="physical_time_type").eval()
        z0 = torch.randn(1, 8)
        clean_actions = torch.randn(1, 3, 4)
        changed = clean_actions.clone()
        changed[:, 2] += 100.0

        before = model(z0, clean_actions, torch.ones(1), mode="stage_b")
        after = model(z0, changed, torch.ones(1), mode="stage_b")

        self.assertTrue(
            torch.equal(
                before["predicted_latents"][:, :2],
                after["predicted_latents"][:, :2],
            )
        )

    def test_stage_a_mask_protects_state_anchor(self):
        mask = stage_a_attention_mask(action_horizon=3)
        expected = torch.ones(4, 4, dtype=torch.bool)
        expected[0, 1:] = False

        self.assertTrue(torch.equal(mask, expected))

    def test_disabled_task_condition_registers_no_parameters(self):
        model = make_model(task_condition_dim=None)

        self.assertIsNone(model.task_condition)
        self.assertFalse(
            any(name.startswith("task_condition.") for name, _ in model.named_parameters())
        )

    def test_enabled_task_condition_is_projected(self):
        model = make_model(task_condition_dim=5)
        z0 = torch.randn(2, 8)
        actions = torch.randn(2, 3, 4)
        timestep = torch.tensor([0.2, 0.8])

        output = model(
            z0,
            actions,
            timestep,
            mode="stage_a",
            task_condition=torch.randn(2, 5),
        )
        output["action_velocity"].sum().backward()

        self.assertEqual(output["action_velocity"].shape, (2, 3, 4))
        self.assertIsNotNone(model.task_condition.weight.grad)

    def test_modes_return_action_and_latent_chunks_with_expected_shapes(self):
        model = make_model()
        z0 = torch.randn(2, 8)
        actions = torch.randn(2, 3, 4)
        timestep = torch.tensor([0.2, 0.8])

        stage_a = model(z0, actions, timestep, mode="stage_a")
        stage_b = model(z0, actions, torch.ones(2), mode="stage_b")
        stage_ab = model(z0, actions, timestep, mode="stage_ab")
        stage_c = model(z0, actions, timestep, mode="stage_c")

        self.assertEqual(stage_a["action_velocity"].shape, (2, 3, 4))
        self.assertEqual(stage_b["predicted_latents"].shape, (2, 3, 8))
        self.assertEqual(stage_ab["clean_action"].shape, (2, 3, 4))
        self.assertEqual(stage_ab["predicted_latents"].shape, (2, 3, 8))
        self.assertEqual(stage_c["action_velocity"].shape, (2, 3, 4))
        self.assertEqual(stage_c["predicted_latents"].shape, (2, 3, 8))

    def test_causal_mask_and_predictions_hide_future_actions(self):
        mask = causal_attention_mask(action_horizon=3)
        self.assertEqual(mask.shape, (7, 7))
        self.assertTrue(torch.equal(mask, torch.ones(7, 7, dtype=torch.bool).tril()))

        model = make_model().eval()
        z0 = torch.randn(1, 8)
        clean_actions = torch.randn(1, 3, 4)
        changed = clean_actions.clone()
        changed[:, 2] += 100.0

        before = model(z0, clean_actions, torch.ones(1), mode="stage_b")
        after = model(z0, changed, torch.ones(1), mode="stage_b")

        self.assertTrue(
            torch.equal(
                before["predicted_latents"][:, :2],
                after["predicted_latents"][:, :2],
            )
        )

    def test_latent_loss_reaches_stage_a_action_head_without_detach(self):
        model = make_model()
        z0 = torch.randn(2, 8)
        noisy_actions = torch.randn(2, 3, 4)
        timestep = torch.tensor([0.3, 0.7])

        output = model(
            z0,
            noisy_actions,
            timestep,
            mode="stage_ab",
            detach_clean_action=False,
        )
        output["predicted_latents"].square().mean().backward()

        grad = model.action_head.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0.0)

    def test_stage_b_call_does_not_change_stage_a_output(self):
        model = make_model().eval()
        z0 = torch.randn(2, 8)
        noisy_actions = torch.randn(2, 3, 4)
        timestep = torch.tensor([0.25, 0.75])

        action_only = model(z0, noisy_actions, timestep, mode="stage_a")
        with_latents = model(z0, noisy_actions, timestep, mode="stage_ab")

        self.assertTrue(
            torch.equal(
                action_only["action_velocity"],
                with_latents["action_velocity"],
            )
        )


    def test_goal_token_mode_uses_two_anchors_and_requires_goal(self):
        mask = stage_a_attention_mask(action_horizon=3, num_anchor_tokens=2)
        expected = torch.ones(5, 5, dtype=torch.bool)
        expected[:2, 2:] = False
        self.assertTrue(torch.equal(mask, expected))
        model = make_model(stage_a_goal_injection="token")
        z0 = torch.randn(2, 8)
        actions = torch.randn(2, 3, 4)
        timestep = torch.tensor([0.2, 0.8])
        with self.assertRaisesRegex(ValueError, "goal_latent"):
            model(z0, actions, timestep, mode="stage_a")
        goal = torch.randn(2, 8, requires_grad=True)
        output = model(z0, actions, timestep, mode="stage_a", goal_latent=goal)
        output["action_velocity"].square().mean().backward()
        self.assertIsNotNone(goal.grad)
        self.assertGreater(goal.grad.abs().sum().item(), 0.0)

    def test_block_causal_mask_is_exact_transition_block_mask(self):
        mask = block_causal_attention_mask(action_horizon=3)
        expected = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        self.assertTrue(torch.equal(mask, expected))
        self.assertTrue(
            torch.equal(
                terminal_full_attention_mask(3),
                torch.ones(5, 5, dtype=torch.bool),
            )
        )

    def test_attention_modes_share_parameter_schema_and_strict_load(self):
        strict = make_model(stage_b_attention_mode="strict_causal")
        for mode in ("block_causal", "terminal_full"):
            candidate = make_model(stage_b_attention_mode=mode)
            self.assertEqual(set(strict.state_dict()), set(candidate.state_dict()))
            self.assertEqual(
                sum(parameter.numel() for parameter in strict.parameters()),
                sum(parameter.numel() for parameter in candidate.parameters()),
            )
            incompatible = candidate.load_state_dict(strict.state_dict(), strict=True)
            self.assertEqual(incompatible.missing_keys, [])
            self.assertEqual(incompatible.unexpected_keys, [])

    def test_block_causal_predictions_are_suffix_invariant(self):
        model = make_model(stage_b_attention_mode="block_causal").eval()
        z0 = torch.randn(1, 8)
        actions = torch.randn(1, 3, 4)
        changed = actions.clone()
        changed[:, 2] += 100.0
        before = model(z0, actions, torch.ones(1), mode="stage_b")
        after = model(z0, changed, torch.ones(1), mode="stage_b")
        self.assertEqual(before["predicted_latents"].shape, (1, 3, 8))
        self.assertTrue(
            torch.equal(
                before["predicted_latents"][:, :2],
                after["predicted_latents"][:, :2],
            )
        )

    def test_terminal_full_returns_one_action_dependent_terminal_latent(self):
        model = make_model(stage_b_attention_mode="terminal_full").eval()
        z0 = torch.randn(1, 8)
        actions = torch.randn(1, 3, 4)
        changed = actions.clone()
        changed[:, 0] += 10.0
        before = model(z0, actions, torch.ones(1), mode="stage_b")
        after = model(z0, changed, torch.ones(1), mode="stage_b")
        self.assertEqual(before["predicted_latents"].shape, (1, 1, 8))
        self.assertFalse(
            torch.equal(before["predicted_latents"], after["predicted_latents"])
        )

    def test_terminal_full_legacy_reuses_existing_positions(self):
        model = make_model(stage_b_attention_mode="terminal_full").eval()
        recorder = RecordingPredictor()
        model.predictor = recorder
        z0 = torch.randn(2, 8)
        actions = torch.randn(2, 3, 4)
        model(z0, actions, torch.ones(2), mode="stage_b")
        observed = recorder.inputs[-1]
        expected = torch.cat(
            (
                model.latent_input(z0).unsqueeze(1) + model.joint_positions[:, :1],
                model.action_input(actions) + model.joint_positions[:, 1::2],
                model.query_tokens[:, -1:].expand(2, -1, -1)
                + model.joint_positions[:, -1:],
            ),
            dim=1,
        )
        self.assertTrue(torch.allclose(observed, expected))

    def test_terminal_full_rejects_nonempty_prefix_actions(self):
        model = make_model(stage_b_attention_mode="terminal_full").eval()
        with self.assertRaisesRegex(ValueError, "prefix_actions.*terminal_full"):
            model.get_action_from_latents(
                torch.randn(1, 8),
                None,
                prefix_actions=torch.randn(1, 1, 4),
            )

    def test_stage_a_is_unchanged_by_stage_b_attention_mode(self):
        strict = make_model(stage_b_attention_mode="strict_causal").eval()
        block = make_model(stage_b_attention_mode="block_causal").eval()
        z0 = torch.randn(2, 8)
        actions = torch.randn(2, 3, 4)
        timestep = torch.tensor([0.2, 0.8])
        strict_output = strict(z0, actions, timestep, mode="stage_a")
        block_output = block(z0, actions, timestep, mode="stage_a")
        self.assertTrue(
            torch.equal(strict_output["action_velocity"], block_output["action_velocity"])
        )

    def test_attention_mode_rejects_invalid_combinations(self):
        with self.assertRaisesRegex(ValueError, "stage_b_attention_mode"):
            make_model(stage_b_attention_mode="not_a_mode")
        with self.assertRaisesRegex(ValueError, "parallel_prefix"):
            make_model(
                stage_b_attention_mode="block_causal",
                stage_b_dynamics="serial_one_step",
            )
        model = make_model(stage_b_attention_mode="block_causal")
        z0 = torch.randn(1, 8)
        actions = torch.randn(1, 3, 4)
        with self.assertRaisesRegex(ValueError, "stage_ab and stage_c"):
            model(z0, actions, torch.ones(1), mode="stage_ab")
        terminal = make_model(stage_b_attention_mode="terminal_full")
        with self.assertRaisesRegex(ValueError, "stage_ab and stage_c"):
            terminal(z0, actions, torch.ones(1), mode="stage_c")

if __name__ == "__main__":
    unittest.main()
