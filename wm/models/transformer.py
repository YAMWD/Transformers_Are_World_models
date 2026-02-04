from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int
    depth: int
    heads: int
    mlp_ratio: int
    dropout: float


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by heads ({heads})")
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, *, is_causal: bool) -> torch.Tensor:
        # x: (B, S, D)
        batch, seq, dim = x.shape
        if dim != self.d_model:
            raise ValueError(f"Expected dim={self.d_model}, got {dim}")

        qkv = self.qkv(x)  # (B,S,3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, heads, S, head_dim)
        q = q.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)

        # PyTorch 2.x fused attention.
        attn_dropout = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=attn_dropout, is_causal=is_causal
        )  # (B, heads, S, head_dim)

        out = out.transpose(1, 2).contiguous().view(batch, seq, dim)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = SelfAttention(cfg.d_model, cfg.heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.mlp_ratio * cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_ratio * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, *, is_causal: bool) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), is_causal=is_causal)
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerCore(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.depth)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, *, is_causal: bool) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, is_causal=is_causal)
        return self.ln_f(x)

