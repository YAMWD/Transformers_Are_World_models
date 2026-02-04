from __future__ import annotations

from typing import Any


def make_carracing(env_id: str, *, render_mode: str | None = None, seed: int | None = None) -> Any:
    """
    Create CarRacing environment.

    Notes:
      - The original paper used Gym's `CarRacing-v0` (deprecated in many installs).
      - Newer Gymnasium exposes `CarRacing-v3` (recommended).
    """
    try:
        import gymnasium as gym  # type: ignore
    except Exception:  # pragma: no cover
        try:
            import gym  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Neither gymnasium nor gym is installed. Install gymnasium[box2d] to use CarRacing."
            ) from e

    kwargs: dict[str, Any] = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    env = gym.make(env_id, **kwargs)
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            env.seed(seed)
    return env
