from __future__ import annotations

import numpy as np

from wm.es.cmaes import CMAES, CMAESConfig


def test_cmaes_improves_on_sphere_maximize() -> None:
    # Maximize f(x) = -||x||^2, optimum at 0 with f=0.
    n = 6
    es = CMAES(CMAESConfig(n_dim=n, population=16, sigma_init=1.0, seed=0, maximize=True))

    def f(x: np.ndarray) -> float:
        return -float(np.sum(np.square(x)))

    best0 = None
    for gen in range(25):
        xs = es.ask()
        fs = [f(x) for x in xs]
        es.tell(xs, fs)
        if gen == 0:
            best0 = float(es.best_f) if es.best_f is not None else None

    assert best0 is not None
    assert es.best_f is not None
    # Should move closer to 0 (less negative) over generations.
    assert float(es.best_f) > best0
    # Final should be reasonably close to optimum.
    assert float(es.best_f) > -0.5

