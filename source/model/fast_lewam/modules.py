"""Fast-LeWAM 共享 DiT 预测器使用的神经网络基础模块。"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from source.model.lewm.modules import MLP, SIGReg


def causal_attention_mask(action_horizon: int, device=None) -> torch.Tensor:
    """为 ``[z0,a1,q1,...,aH,qH]`` 构造含对角线的下三角因果掩码。"""
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")
    length = 1 + 2 * action_horizon
    return torch.ones(length, length, dtype=torch.bool, device=device).tril()


def stage_a_attention_mask(
    action_horizon: int, device=None, num_anchor_tokens: int = 1
) -> torch.Tensor:
    """
    允许动作读取 latent 锚点，同时阻止锚点 query 读取噪声动作。
    1100
    1100
    1111
    1111
    """
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")
    if num_anchor_tokens not in {1, 2}:
        raise ValueError("num_anchor_tokens must be 1 or 2")
    length = action_horizon + num_anchor_tokens
    mask = torch.ones(
        length,
        length,
        dtype=torch.bool,
        device=device,
    )
    mask[:num_anchor_tokens, num_anchor_tokens:] = False
    return mask


def timestep_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
    """将形状为 ``[B]``、范围为 ``[0, 1]`` 的流时间步编码为正弦向量。"""
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
    """用 AdaLN 预测的平移量和缩放量调制已归一化的 token。"""
    return x * (1 + scale) + shift


class Attention(nn.Module):
    """支持可选注意力掩码的多头缩放点积自注意力层。"""

    def __init__(self, dim, heads=8, dropout=0.0):
        """初始化 QKV 投影、输出投影以及训练期注意力 dropout。"""
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.dropout = dropout
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, attention_mask=None):
        """对 ``[B,T,D]`` token 计算自注意力，并保持输入输出形状一致。"""
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
    """Transformer block 中使用的两层 GELU 前馈网络。"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        """按输入维度、隐藏维度和 dropout 比例构造前馈网络。"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """逐 token 应用前馈网络并返回与输入同形状的结果。"""
        return self.net(x)


class DiTBlock(nn.Module):
    """由 Stage A、B、C 共享、通过 AdaLN 条件化的 Transformer block。"""

    def __init__(self, dim, heads, mlp_dim, dropout=0.0):
        """构造注意力、MLP 和生成六组 AdaLN 参数的条件调制层。"""
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(dim, heads=heads, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        final = self.modulation[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias[2 * dim : 3 * dim].fill_(1e-3)
            final.bias[5 * dim : 6 * dim].fill_(1e-3)

    def forward(self, x, condition, attention_mask=None):
        """用样本级条件调制注意力和 MLP，并通过可学习门控更新 token。"""
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
    """堆叠多个 DiTBlock 的共享主干，可在不同阶段切换注意力掩码。"""

    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.0):
        """按照给定深度构造共享 DiT block 列表和最终 LayerNorm。"""
        super().__init__()
        self.layers = nn.ModuleList(
            [DiTBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, condition, attention_mask=None):
        """依次执行所有共享 block，并对最终 token 做归一化。"""
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
    "block_causal_attention_mask",
    "causal_attention_mask",
    "stage_b_attention_mask",
    "stage_a_attention_mask",
    "terminal_full_attention_mask",
    "timestep_embedding",
]


def block_causal_attention_mask(action_horizon: int, device=None) -> torch.Tensor:
    """为交错的动作/查询 token 构造按 transition block 的因果掩码。

    Token 顺序为 ``[z0, (a0,q1), ..., (a{H-1},qH)]``。状态锚点只能
    读取自身；每个 transition block 可以双向读取自身，并读取状态锚点
    和所有更早的 block。返回值使用 ``scaled_dot_product_attention`` 的
    bool mask 约定，即 ``True`` 表示允许注意力连接。
    """
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")
    length = 1 + 2 * action_horizon
    mask = torch.zeros(length, length, dtype=torch.bool, device=device)
    mask[0, 0] = True
    for block in range(action_horizon):
        end = 1 + 2 * (block + 1)
        mask[1 + 2 * block : end, :end] = True
    return mask


def terminal_full_attention_mask(action_horizon: int, device=None) -> torch.Tensor:
    """为终点 token 序列 ``[z0,a0,...,aH-1,qH]`` 构造全注意力掩码。"""
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")
    length = action_horizon + 2
    return torch.ones(length, length, dtype=torch.bool, device=device)


def stage_b_attention_mask(
    action_horizon: int, mode: str = "strict_causal", device=None
) -> torch.Tensor:
    """构造 Stage-B attention mode 对应的 bool mask。"""
    if mode == "strict_causal":
        return causal_attention_mask(action_horizon, device=device)
    if mode == "block_causal":
        return block_causal_attention_mask(action_horizon, device=device)
    if mode == "terminal_full":
        return terminal_full_attention_mask(action_horizon, device=device)
    raise ValueError(
        "mode must be 'strict_causal', 'block_causal', or 'terminal_full'"
    )
