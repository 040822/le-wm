from types import SimpleNamespace

import torch
from torch import nn

from source.model.lewm.jepa import JEPA
from source.policy.round5_phase1_7 import LeWMStageBVerifier


class TinyEncoder(nn.Module):
    def forward(self, pixels, interpolate_pos_encoding=True):
        del interpolate_pos_encoding
        values = pixels.mean(dim=(1, 2, 3)).unsqueeze(-1)
        tokens = values[:, None, :].expand(-1, 2, 192)
        return SimpleNamespace(last_hidden_state=tokens)


class TinyActionEncoder(nn.Module):
    def forward(self, actions):
        values = actions.mean(dim=-1, keepdim=True)
        return values.expand(-1, -1, 192)


class TinyPredictor(nn.Module):
    def forward(self, embeddings, action_embeddings):
        return embeddings + action_embeddings


def test_lewm_scorer_matches_standard_autoregressive_rollout():
    torch.manual_seed(7)
    model = JEPA(
        encoder=TinyEncoder(),
        predictor=TinyPredictor(),
        action_encoder=TinyActionEncoder(),
        projector=nn.Identity(),
        pred_proj=nn.Identity(),
    ).eval()
    adapter = LeWMStageBVerifier(
        model,
        history_size=3,
        action_dim=10,
        action_horizon=5,
        latent_dim=192,
    )
    history_pixels = torch.randn(2, 3, 3, 8, 8)
    goal_pixels = torch.randn(2, 1, 3, 8, 8)
    history_actions = torch.randn(2, 2, 10)
    candidates = torch.randn(2, 4, 5, 10)
    history_latents = model.encode({"pixels": history_pixels})["emb"]
    goal_latent = model.encode({"pixels": goal_pixels})["emb"][:, 0]

    actual = adapter.score_action_candidates(
        history_latents,
        goal_latent,
        candidates,
        context={"_phase17_history_actions": history_actions},
        candidate_batch_size=2,
        solver_batch_size=2,
    )

    expected_rows = []
    for batch_index in range(candidates.shape[0]):
        expected_candidates = []
        for candidate_index in range(candidates.shape[1]):
            actions = torch.cat(
                (
                    history_actions[batch_index : batch_index + 1],
                    candidates[batch_index : batch_index + 1, candidate_index],
                ),
                dim=1,
            )
            rollout = model.rollout(
                {"pixels": history_pixels[batch_index : batch_index + 1, None]},
                actions[:, None],
                history_size=3,
            )
            terminal = rollout["predicted_emb"][:, :, -1]
            cost = (terminal - goal_latent[batch_index : batch_index + 1, None]).square().sum(-1)
            expected_candidates.append(cost[:, 0])
        expected_rows.append(torch.cat(expected_candidates))
    expected = torch.stack(expected_rows)

    torch.testing.assert_close(actual, expected)
    assert adapter.last_forward_count == 14
