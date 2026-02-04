from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from wm.models.decoder import ConvDecoder64
from wm.models.transformer import TransformerConfig, TransformerCore


@dataclass(frozen=True)
class ViTWMConfig:
    image_size: int
    patch_size: int
    z_dim: int
    action_dim: int
    d_model: int
    depth: int
    heads: int
    mlp_ratio: int
    dropout: float
    l_ctx: int
    mdn_k: int
    predict_done: bool


class ViTWorldModel(nn.Module):
    """
    Single-backbone transformer world model with two modes:
      - visual mode: pixels -> z (VAE-style latent head + optional recon decoder)
      - dynamics mode: (z, a) sequence -> MDN for next-z (+ done head)

    The transformer *core* weights are shared across both modes.
    """

    def __init__(self, cfg: ViTWMConfig, *, with_decoder: bool = True) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.image_size % cfg.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        grid = cfg.image_size // cfg.patch_size
        self.num_patches = grid * grid

        tcfg = TransformerConfig(
            d_model=cfg.d_model,
            depth=cfg.depth,
            heads=cfg.heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        )
        self.core = TransformerCore(tcfg)

        # Visual adapter: patchify via conv.
        self.patch_embed = nn.Conv2d(3, cfg.d_model, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos_visual = nn.Parameter(torch.zeros(1, 1 + self.num_patches, cfg.d_model))
        self.to_mu = nn.Linear(cfg.d_model, cfg.z_dim)
        self.to_logsigma = nn.Linear(cfg.d_model, cfg.z_dim)

        self.decoder = ConvDecoder64(cfg.z_dim) if with_decoder else None

        # Dynamics adapter: one token per step.
        self.dyn_embed = nn.Linear(cfg.z_dim + cfg.action_dim, cfg.d_model)
        self.pos_time = nn.Parameter(torch.zeros(1, cfg.l_ctx, cfg.d_model))

        # MDN heads.
        self.mdn_pi = nn.Linear(cfg.d_model, cfg.mdn_k)
        self.mdn_mu = nn.Linear(cfg.d_model, cfg.mdn_k * cfg.z_dim)
        self.mdn_logsigma = nn.Linear(cfg.d_model, cfg.mdn_k * cfg.z_dim)

        self.done_head = nn.Linear(cfg.d_model, 1) if cfg.predict_done else None

        self._init_params()

    def _init_params(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_visual, std=0.02)
        nn.init.trunc_normal_(self.pos_time, std=0.02)
        # Patch embed / heads use default init; good enough for baseline.

    @staticmethod
    def reparameterize(mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
        std = torch.exp(logsigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor, *, sample: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B,3,H,W) float in [0,1]
        returns: (z, mu, logsigma) with z shape (B, z_dim)
        """
        batch = x.shape[0]
        tokens = self.patch_embed(x)  # (B, D, Gh, Gw)
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, N, D)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+N, D)
        tokens = tokens + self.pos_visual

        y = self.core(tokens, is_causal=False)  # full attention
        cls_y = y[:, 0, :]
        mu = self.to_mu(cls_y)
        logsigma = self.to_logsigma(cls_y)
        z = self.reparameterize(mu, logsigma) if sample else mu
        return z, mu, logsigma

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("Decoder disabled (with_decoder=False).")
        return self.decoder(z)

    def forward_visual(
        self, x: torch.Tensor, *, sample: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        z, mu, logsigma = self.encode(x, sample=sample)
        recon = self.decode(z) if self.decoder is not None else None
        return z, mu, logsigma, recon

    def forward_dynamics(
        self, z_seq: torch.Tensor, a_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        z_seq: (B,S,z_dim)
        a_seq: (B,S,action_dim)

        Returns:
          pi_logits: (B,S,K)
          mu: (B,S,K,z_dim)
          logsigma: (B,S,K,z_dim)
          done_logit: (B,S) or None
          y: (B,S,d_model) final hidden states (for extracting h_t)
        """
        if z_seq.ndim != 3 or a_seq.ndim != 3:
            raise ValueError("z_seq and a_seq must be 3D tensors (B,S,*)")
        if z_seq.shape[:2] != a_seq.shape[:2]:
            raise ValueError("z_seq and a_seq must have same (B,S)")

        batch, seq, _ = z_seq.shape
        if seq > self.cfg.l_ctx:
            raise ValueError(f"seq length {seq} exceeds l_ctx {self.cfg.l_ctx}")

        u = torch.cat([z_seq, a_seq], dim=-1)
        tokens = self.dyn_embed(u)  # (B,S,D)
        tokens = tokens + self.pos_time[:, :seq, :]

        y = self.core(tokens, is_causal=True)

        pi_logits = self.mdn_pi(y)  # (B,S,K)
        mu = self.mdn_mu(y).view(batch, seq, self.cfg.mdn_k, self.cfg.z_dim)
        logsigma = self.mdn_logsigma(y).view(batch, seq, self.cfg.mdn_k, self.cfg.z_dim)

        done_logit = None
        if self.done_head is not None:
            done_logit = self.done_head(y).squeeze(-1)  # (B,S)

        return pi_logits, mu, logsigma, done_logit, y

