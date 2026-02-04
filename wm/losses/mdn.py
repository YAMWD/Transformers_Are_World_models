from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MDNParams:
    pi_logits: torch.Tensor  # (B,S,K)
    mu: torch.Tensor  # (B,S,K,D)
    logsigma: torch.Tensor  # (B,S,K,D)


def mdn_nll(params: MDNParams, target: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood for a mixture of diagonal Gaussians.

    params: pi_logits (B,S,K), mu/logsigma (B,S,K,D)
    target: (B,S,D)
    returns: scalar (mean over batch+time)
    """
    pi_logits, mu, logsigma = params.pi_logits, params.mu, params.logsigma
    if target.ndim != 3:
        raise ValueError("target must have shape (B,S,D)")
    if pi_logits.shape[:2] != target.shape[:2]:
        raise ValueError("pi_logits and target must share (B,S)")
    if mu.shape[:2] != target.shape[:2] or logsigma.shape[:2] != target.shape[:2]:
        raise ValueError("mu/logsigma and target must share (B,S)")

    # Expand target to (B,S,1,D) for broadcast over K.
    x = target.unsqueeze(-2)
    # sigma = exp(logsigma)
    sigma = torch.exp(logsigma)
    z = (x - mu) / sigma
    log_norm = -0.5 * (z * z + 2.0 * logsigma + math.log(2.0 * math.pi))
    log_prob = torch.sum(log_norm, dim=-1)  # (B,S,K)

    log_pi = F.log_softmax(pi_logits, dim=-1)
    log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)  # (B,S)
    return (-log_mix).mean()


def mdn_sample(params: MDNParams, *, tau: float) -> torch.Tensor:
    """
    Sample from the MDN at temperature tau:
      - mixture weights soften with /tau
      - stddev scales by *tau

    params: tensors shaped (B,K,D) or (B,S,K,D) with batch-leading dims.
    returns: sample z with shape matching batch-leading dims and D.
    """
    if tau <= 0.0:
        raise ValueError("tau must be > 0")

    pi_logits = params.pi_logits
    mu = params.mu
    logsigma = params.logsigma

    if pi_logits.ndim == 3:
        # (B,S,K) -> (B*S,K) for sampling indices; mu/logsigma (B,S,K,D)
        b, s, k = pi_logits.shape
        d = mu.shape[-1]
        flat_logits = (pi_logits / tau).reshape(b * s, k)
        mix_idx = torch.distributions.Categorical(logits=flat_logits).sample()  # (B*S,)

        flat_mu = mu.reshape(b * s, k, d)
        flat_logsigma = logsigma.reshape(b * s, k, d)
        sel_mu = flat_mu[torch.arange(b * s, device=mu.device), mix_idx]  # (B*S,D)
        sel_sigma = torch.exp(flat_logsigma[torch.arange(b * s, device=mu.device), mix_idx]) * tau
        eps = torch.randn_like(sel_sigma)
        z = sel_mu + sel_sigma * eps
        return z.view(b, s, d)

    if pi_logits.ndim == 2:
        # (B,K)
        b, k = pi_logits.shape
        d = mu.shape[-1]
        mix_idx = torch.distributions.Categorical(logits=pi_logits / tau).sample()  # (B,)
        sel_mu = mu[torch.arange(b, device=mu.device), mix_idx]  # (B,D)
        sel_sigma = torch.exp(logsigma[torch.arange(b, device=mu.device), mix_idx]) * tau
        eps = torch.randn_like(sel_sigma)
        return sel_mu + sel_sigma * eps

    raise ValueError("pi_logits must have shape (B,K) or (B,S,K)")

