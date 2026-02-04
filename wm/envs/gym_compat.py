from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class GymLike(Protocol):
    def reset(self, **kwargs: Any) -> Any: ...

    def step(self, action: Any) -> Any: ...

    def close(self) -> None: ...


@dataclass
class StepResult:
    obs: Any
    reward: float
    done: bool
    info: dict[str, Any]


class GymCompatWrapper:
    """
    Wrap gymnasium (terminated/truncated) and gym (done) into a single API:
      reset() -> obs
      step(a) -> StepResult(obs, reward, done, info)
    """

    def __init__(self, env: GymLike):
        self.env = env

    def reset(self, **kwargs: Any) -> Any:
        out = self.env.reset(**kwargs)
        # gymnasium: (obs, info); gym: obs
        if isinstance(out, tuple) and len(out) == 2:
            return out[0]
        return out

    def step_result(self, action: Any) -> StepResult:
        out = self.env.step(action)
        # gymnasium: obs, reward, terminated, truncated, info
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return StepResult(obs=obs, reward=float(reward), done=bool(terminated or truncated), info=dict(info))
        # gym: obs, reward, done, info
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            return StepResult(obs=obs, reward=float(reward), done=bool(done), info=dict(info))
        raise RuntimeError(f"Unexpected env.step() output: {type(out)} {out}")

    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        r = self.step_result(action)
        return r.obs, r.reward, r.done, r.info

    def close(self) -> None:
        self.env.close()
