from __future__ import annotations

import math

import torch


def kl_standard_normal(mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
    """
    KL(q(z|x)=N(mu, sigma^2) || N(0, I)) for diagonal Gaussians.

    mu/logsigma: (..., z_dim) where logsigma is log stddev.
    returns: scalar tensor (mean over batch)
    """
    # logvar = 2*logsigma
    logvar = 2.0 * logsigma
    var = torch.exp(logvar)
    kl = 0.5 * torch.sum(var + mu**2 - 1.0 - logvar, dim=-1)
    return kl.mean()


def beta_anneal(step: int, total_steps: int, frac: float) -> float:
    if total_steps <= 0:
        return 1.0
    if frac <= 0.0:
        return 1.0
    warm = max(1, int(math.ceil(total_steps * frac)))
    return float(min(1.0, step / warm))

