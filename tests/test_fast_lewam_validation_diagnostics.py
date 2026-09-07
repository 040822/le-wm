import unittest

import torch
from torch import nn

from source.common.fast_lewam_validation import (
    FastLeWAMValidationDiagnosticsCallback,
    compute_stage_b_validation_metrics,
)


class ActionAsLatentModel(nn.Module):
    action_dim = 2
    action_horizon = 2

    def forward(
        self,
        z0,
        actions,
        timestep,
        *,
        mode,
        task_condition=None,
    ):
        if mode != "stage_b":
            raise ValueError("diagnostics model only supports stage_b")
        return {"predicted_latents": actions}


class TerminalActionAsLatentModel(ActionAsLatentModel):
    stage_b_attention_mode = "terminal_full"

    def forward(
        self,
        z0,
        actions,
        timestep,
        *,
        mode,
        task_condition=None,
    ):
        output = super().forward(
            z0,
            actions,
            timestep,
            mode=mode,
            task_condition=task_condition,
        )
        return {"predicted_latents": output["predicted_latents"][:, -1:]}


class FakeTrainer:
    sanity_checking = False


class FakePolicy:
    train_mode = "stage_ab"

    def __init__(self):
        self.model = ActionAsLatentModel()
        self.logged = []

    def log_dict(self, metrics, **kwargs):
        self.logged.append((metrics, kwargs))


class FastLeWAMValidationMetricsTests(unittest.TestCase):
    def test_terminal_predictions_are_reshaped_by_one_and_compared_at_endpoint(self):
        clean_actions = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 0.5]],
                [[-1.0, 0.5], [-2.0, -0.5]],
            ]
        )
        target_latents = clean_actions.clone()
        embeddings = torch.cat(
            [torch.zeros(2, 1, 2), target_latents], dim=1
        )
        forward_output = {
            "emb": embeddings,
            "predicted_latents": target_latents[:, -1:] + 1.0,
        }

        metrics = compute_stage_b_validation_metrics(
            model=TerminalActionAsLatentModel(),
            batch={"action": clean_actions},
            forward_output=forward_output,
            noise_std=0.25,
            seed=7,
        )

        self.assertEqual(
            metrics["validate/stage_b_clean_terminal_mse"].item(), 0.0
        )
        self.assertEqual(
            metrics["validate/stage_b_predicted_terminal_mse"].item(), 1.0
        )
        self.assertEqual(
            metrics["validate/stage_b_expert_preference_accuracy"].item(), 1.0
        )

    def test_clean_and_predicted_terminal_quality_and_expert_ranking_are_separate(self):
        clean_actions = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 0.5]],
                [[-1.0, 0.5], [-2.0, -0.5]],
            ]
        )
        target_latents = clean_actions.clone()
        embeddings = torch.cat(
            [torch.zeros(2, 1, 2), target_latents], dim=1
        )
        forward_output = {
            "emb": embeddings,
            "predicted_latents": target_latents + 1.0,
        }
        batch = {"action": clean_actions}

        metrics = compute_stage_b_validation_metrics(
            model=ActionAsLatentModel(),
            batch=batch,
            forward_output=forward_output,
            noise_std=0.25,
            seed=7,
        )

        self.assertEqual(metrics["validate/stage_b_clean_terminal_mse"].item(), 0.0)
        self.assertEqual(
            metrics["validate/stage_b_predicted_terminal_mse"].item(), 1.0
        )
        self.assertEqual(
            metrics["validate/stage_b_expert_preference_accuracy"].item(), 1.0
        )
        self.assertEqual(
            metrics["validate/stage_b_expert_top1_rate"].item(), 1.0
        )
        self.assertGreater(
            metrics["validate/stage_b_expert_negative_margin"].item(), 0.0
        )

    def test_callback_logs_only_the_configured_number_of_validation_batches(self):
        clean_actions = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 0.5]],
                [[-1.0, 0.5], [-2.0, -0.5]],
            ]
        )
        target_latents = clean_actions.clone()
        batch = {"action": clean_actions}
        output = {
            "emb": torch.cat(
                [torch.zeros(2, 1, 2), target_latents], dim=1
            ),
            "predicted_latents": target_latents + 1.0,
        }
        policy = FakePolicy()
        callback = FastLeWAMValidationDiagnosticsCallback(
            max_batches=1,
            noise_std=0.25,
            seed=7,
        )

        callback.on_validation_batch_end(
            FakeTrainer(), policy, output, batch, batch_idx=0
        )
        callback.on_validation_batch_end(
            FakeTrainer(), policy, output, batch, batch_idx=1
        )

        self.assertEqual(len(policy.logged), 1)
        metrics, log_kwargs = policy.logged[0]
        self.assertIn("validate/stage_b_clean_terminal_mse", metrics)
        self.assertEqual(log_kwargs["batch_size"], 2)
        self.assertFalse(log_kwargs["on_step"])
        self.assertTrue(log_kwargs["on_epoch"])
        self.assertTrue(log_kwargs["sync_dist"])

    def test_callback_logs_stage_b_only_validation(self):
        clean_actions = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 0.5]],
                [[-1.0, 0.5], [-2.0, -0.5]],
            ]
        )
        target_latents = clean_actions.clone()
        policy = FakePolicy()
        policy.train_mode = "stage_b"
        callback = FastLeWAMValidationDiagnosticsCallback(
            max_batches=1,
            noise_std=0.25,
            seed=7,
        )

        callback.on_validation_batch_end(
            FakeTrainer(),
            policy,
            {
                "emb": torch.cat(
                    [torch.zeros(2, 1, 2), target_latents], dim=1
                ),
                "predicted_latents": target_latents,
            },
            {"action": clean_actions},
            batch_idx=0,
        )

        self.assertEqual(len(policy.logged), 1)
        self.assertIn(
            "validate/stage_b_expert_preference_accuracy",
            policy.logged[0][0],
        )

if __name__ == "__main__":
    unittest.main()
