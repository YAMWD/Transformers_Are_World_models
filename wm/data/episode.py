from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Episode:
    frames: np.ndarray  # uint8, (T,H,W,3)
    actions: np.ndarray  # float32, (T,A)
    dones: np.ndarray  # bool, (T,)  done after taking action at t (i.e. done_{t+1})
    rewards: np.ndarray | None = None  # float32, (T,)
    infos: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.frames.dtype != np.uint8:
            raise ValueError("frames must be uint8")
        if self.frames.ndim != 4 or self.frames.shape[-1] != 3:
            raise ValueError("frames must have shape (T,H,W,3)")
        if self.actions.ndim != 2:
            raise ValueError("actions must have shape (T,A)")
        if self.dones.ndim != 1:
            raise ValueError("dones must have shape (T,)")
        if len(self.frames) != len(self.actions) or len(self.frames) != len(self.dones):
            raise ValueError("frames/actions/dones must have same T")
        if self.rewards is not None and len(self.rewards) != len(self.frames):
            raise ValueError("rewards must have length T when present")

    @property
    def length(self) -> int:
        return int(self.frames.shape[0])


def save_episode(path: str | Path, ep: Episode) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "frames": ep.frames,
        "actions": ep.actions,
        "dones": ep.dones,
    }
    if ep.rewards is not None:
        payload["rewards"] = ep.rewards
    if ep.infos is not None:
        payload["infos_json"] = np.array([ep.infos], dtype=object)
    np.savez_compressed(path, **payload)


def load_episode(path: str | Path) -> Episode:
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        frames = data["frames"]
        actions = data["actions"]
        dones = data["dones"]
        rewards = data["rewards"] if "rewards" in data.files else None
        infos = None
        if "infos_json" in data.files:
            infos_arr = data["infos_json"]
            if len(infos_arr) >= 1 and isinstance(infos_arr[0], dict):
                infos = infos_arr[0]
        return Episode(frames=frames, actions=actions, dones=dones, rewards=rewards, infos=infos)

