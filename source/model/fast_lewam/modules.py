"""Neural building blocks for the shared Fast-LeWAM DiT predictor."""

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from source.model.lewm.modules import MLP, SIGReg


def causal_attention_mask(action_horizon: int, device=None) -> torch.Tensor:
    """Return the inclusive causal mask for ``[z0, a1, q1, ..., aH, qH]``."""
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")
    length = 1 + 2 * action_horizon
    return torch.ones(length, length, dtype=torch.bool, device=device).tril()


def timestep_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal embeddings for flow timesteps in ``[0, 1]``."""
    if timestep.ndim != 1:
        raise ValueError(f"timestep must have shape [B], got {tuple(timestep.shape)}")
    half = dim // 2
    if half == 0:
        return timestep[:, None]
    frequencies = torch.exp(
        -math.log(10_000)
        * torch.arange(half, device=timestep.device)
        / max(half - 1, 1)
    )
    angles = timestep.float()[:, None] * frequencies[None] * 1000.0
    embedding = torch.cat([angles.cos(), angles.sin()], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.dropout = dropout
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, attention_mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            rearrange(value, "b t (h d) -> b h t d", h=self.heads)
            for value in qkv
        )
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.to_out(rearrange(output, "b h t d -> b t (h d)"))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DiTBlock(nn.Module):
    """AdaLN-conditioned Transformer block shared by Stage A, B, and C."""

    def __init__(self, dim, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(dim, heads=heads, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, condition, attention_mask=None):
        modulation = self.modulation(condition).unsqueeze(1).chunk(6, dim=-1)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = modulation
        x = x + gate_attn * self.attention(
            modulate(self.norm1(x), shift_attn, scale_attn),
            attention_mask=attention_mask,
        )
        x = x + gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class SharedDiT(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [DiTBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, condition, attention_mask=None):
        for layer in self.layers:
            tokens = layer(tokens, condition, attention_mask=attention_mask)
        return self.norm(tokens)


__all__ = [
    "Attention",
    "DiTBlock",
    "FeedForward",
    "MLP",
    "SIGReg",
    "SharedDiT",
    "causal_attention_mask",
    "timestep_embedding",
]
