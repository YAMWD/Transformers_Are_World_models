from __future__ import annotations

import math

import torch

from wm.losses.mdn import MDNParams, mdn_nll, mdn_sample
from wm.losses.vae import kl_standard_normal


def test_kl_standard_normal_zero_is_zero() -> None:
    mu = torch.zeros(4, 16)
    logsigma = torch.zeros(4, 16)
    kl = kl_standard_normal(mu, logsigma)
    assert torch.isfinite(kl)
    assert abs(float(kl)) < 1e-6


def test_mdn_nll_finite_and_lower_when_on_target() -> None:
    # K=1 diagonal Gaussian should give lower NLL when mu==target and sigma small.
    b, s, k, d = 2, 5, 1, 8
    target = torch.randn(b, s, d)

    pi_logits = torch.zeros(b, s, k)
    mu_good = target.unsqueeze(-2).contiguous()
    logsigma_small = torch.full((b, s, k, d), -3.0)  # sigma ~ 0.05

    mu_bad = (target + 10.0).unsqueeze(-2).contiguous()
    logsigma_big = torch.zeros(b, s, k, d)  # sigma=1

    nll_good = mdn_nll(MDNParams(pi_logits, mu_good, logsigma_small), target)
    nll_bad = mdn_nll(MDNParams(pi_logits, mu_bad, logsigma_big), target)
    assert torch.isfinite(nll_good)
    assert torch.isfinite(nll_bad)
    assert float(nll_good) < float(nll_bad)


def test_mdn_sample_shape_and_temperature_effect() -> None:
    # Vectorized variance check: tau scales sigma by *tau inside mdn_sample.
    # Use K=1 to avoid categorical overhead.
    b, k, d = 4, 1, 6
    base_pi = torch.zeros(b, k)
    base_mu = torch.zeros(b, k, d)
    base_logsigma = torch.zeros(b, k, d)  # sigma=1

    # Draw many samples in one call by repeating batch dimension.
    n = 4096
    pi = base_pi.repeat(n, 1)  # (n*b, k)
    mu = base_mu.repeat(n, 1, 1)  # (n*b, k, d)
    logsigma = base_logsigma.repeat(n, 1, 1)  # (n*b, k, d)
    params = MDNParams(pi_logits=pi, mu=mu, logsigma=logsigma)

    s1 = mdn_sample(params, tau=1.0).view(n, b, d)
    s2 = mdn_sample(params, tau=2.0).view(n, b, d)
    assert s1.shape == (n, b, d)
    assert s2.shape == (n, b, d)

    v1 = s1.var(dim=0).mean()
    v2 = s2.var(dim=0).mean()
    assert float(v2) > float(v1) * 3.0  # tau=2 => variance ~4x
