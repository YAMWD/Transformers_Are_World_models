from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CMAESConfig:
    n_dim: int
    population: int = 64
    sigma_init: float = 0.5
    seed: int = 0
    maximize: bool = True
    mean_init: np.ndarray | None = None
    eig_update_every: int = 10


class CMAES:
    """
    Minimal CMA-ES implementation suitable for ~1k-dimensional controllers.

    Notes:
      - Uses full covariance.
      - Maximization supported via `maximize=True` (internally negates fitness).
    """

    def __init__(self, cfg: CMAESConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.n = int(cfg.n_dim)
        self.lam = int(cfg.population)
        if self.lam < 4:
            raise ValueError("population must be >= 4")

        self.mu = self.lam // 2
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)
        self.mu_eff = 1.0 / np.sum(self.weights**2)

        n = self.n
        mu_eff = self.mu_eff

        self.c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + self.c_sigma
        self.c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        self.c_mu = min(
            1.0 - self.c1,
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff),
        )
        self.chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        self.sigma = float(cfg.sigma_init)
        if cfg.mean_init is None:
            self.mean = np.zeros(n, dtype=np.float64)
        else:
            if cfg.mean_init.shape != (n,):
                raise ValueError("mean_init must have shape (n_dim,)")
            self.mean = cfg.mean_init.astype(np.float64, copy=True)

        self.C = np.eye(n, dtype=np.float64)
        self.p_sigma = np.zeros(n, dtype=np.float64)
        self.p_c = np.zeros(n, dtype=np.float64)

        self.B = np.eye(n, dtype=np.float64)
        self.D = np.ones(n, dtype=np.float64)
        self.inv_sqrt_C = np.eye(n, dtype=np.float64)

        self.generation = 0
        self._eig_updates = 0
        self._last_candidates: list[np.ndarray] | None = None

        self.best_x: np.ndarray | None = None
        self.best_f: float | None = None

    def ask(self) -> list[np.ndarray]:
        z = self.rng.standard_normal(size=(self.lam, self.n))
        # y_k = B @ (D * z_k)
        y = (z * self.D) @ self.B.T  # (lam,n)
        x = self.mean[None, :] + self.sigma * y
        candidates = [xk.astype(np.float32, copy=False) for xk in x]
        self._last_candidates = candidates
        return candidates

    def tell(self, candidates: list[np.ndarray], fitness: list[float]) -> None:
        if len(candidates) != self.lam or len(fitness) != self.lam:
            raise ValueError("tell() expects fitness/candidates of length population")

        # CMA-ES is usually formulated as minimization.
        fit = np.asarray(fitness, dtype=np.float64)
        if self.cfg.maximize:
            fit = -fit

        order = np.argsort(fit)  # ascending (best first)
        x_sorted = np.stack([candidates[i].astype(np.float64, copy=False) for i in order], axis=0)  # (lam,n)

        # Track best (in original maximize sense).
        best_idx = int(order[0])
        best_f = float(fitness[best_idx])
        if self.best_f is None or best_f > self.best_f:
            self.best_f = best_f
            self.best_x = candidates[best_idx].copy()

        # Recompute y in whitened coordinates relative to old mean.
        y_sorted = (x_sorted - self.mean[None, :]) / self.sigma  # (lam,n)

        y_w = np.sum(self.weights[:, None] * y_sorted[: self.mu], axis=0)  # (n,)
        mean_new = self.mean + self.sigma * y_w

        # Update evolution path p_sigma.
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + np.sqrt(
            self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff
        ) * (self.inv_sqrt_C @ y_w)

        # Step-size control.
        norm_p_sigma = np.linalg.norm(self.p_sigma)
        self.sigma *= float(np.exp((self.c_sigma / self.d_sigma) * (norm_p_sigma / self.chi_n - 1.0)))

        # Heaviside indicator.
        t = self.generation + 1
        denom = np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2.0 * t))
        h_sigma = 1.0 if (norm_p_sigma / denom) < (1.4 + 2.0 / (self.n + 1.0)) * self.chi_n else 0.0

        # Update p_c.
        self.p_c = (1.0 - self.c_c) * self.p_c + h_sigma * np.sqrt(
            self.c_c * (2.0 - self.c_c) * self.mu_eff
        ) * y_w

        # Rank-one and rank-mu updates.
        rank_one = np.outer(self.p_c, self.p_c)
        rank_mu = np.zeros_like(self.C)
        for i in range(self.mu):
            yi = y_sorted[i]
            rank_mu += self.weights[i] * np.outer(yi, yi)

        old_C = self.C
        self.C = (1.0 - self.c1 - self.c_mu) * old_C + self.c1 * rank_one + self.c_mu * rank_mu
        if h_sigma < 1.0:
            self.C += self.c1 * (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c) * old_C

        # Numerical symmetry.
        self.C = 0.5 * (self.C + self.C.T)

        self.mean = mean_new
        self.generation += 1

        # Update eigendecomposition periodically.
        if self.generation % max(1, int(self.cfg.eig_update_every)) == 0:
            self._update_eigensystem()

    def _update_eigensystem(self) -> None:
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-20)
        self.D = np.sqrt(eigvals)
        self.B = eigvecs
        self.inv_sqrt_C = (self.B * (1.0 / self.D)[None, :]) @ self.B.T
        self._eig_updates += 1
