from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from wm.data.episode import Episode


def _resize_u8(frame_hwc: np.ndarray, size: int) -> np.ndarray:
    if frame_hwc.shape[0] == size and frame_hwc.shape[1] == size:
        return frame_hwc.astype(np.uint8, copy=False)

    try:
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(frame_hwc).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        y = (x.clamp(0, 1) * 255.0).byte().squeeze(0).permute(1, 2, 0).contiguous().numpy()
        return y
    except Exception:  # pragma: no cover
        h, w, _ = frame_hwc.shape
        ys = (np.linspace(0, h - 1, size)).astype(np.int64)
        xs = (np.linspace(0, w - 1, size)).astype(np.int64)
        return frame_hwc[ys][:, xs].astype(np.uint8, copy=False)


def obs_to_frame_u8(obs: Any, *, size: int) -> np.ndarray:
    """
    Convert env observation to uint8 HWC RGB at desired size.
    Works for common gym/vizdoom image observations.
    """
    if isinstance(obs, np.ndarray):
        arr = obs
    else:
        arr = np.asarray(obs)

    # Common cases:
    # - HWC uint8
    # - CHW uint8
    if arr.ndim == 3 and arr.shape[-1] == 3:
        frame = arr
    elif arr.ndim == 3 and arr.shape[0] == 3:
        frame = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported observation shape for image: {arr.shape}")

    frame = frame.astype(np.uint8, copy=False)
    return _resize_u8(frame, size)


@dataclass(frozen=True)
class RolloutSpec:
    frame_size: int = 64
    max_steps: int = 1000


def collect_episode(
    env: Any,
    *,
    policy: Callable[[Any], np.ndarray],
    spec: RolloutSpec,
) -> Episode:
    """
    Collect one episode using the provided policy(env_obs)->action.
    Expects env.reset() and env.step(action) -> obs, reward, done, info.
    """
    obs = env.reset()

    frames: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    dones: list[bool] = []
    rewards: list[float] = []

    for _t in range(spec.max_steps):
        frame = obs_to_frame_u8(obs, size=spec.frame_size)
        a = policy(obs)
        obs, reward, done, _info = env.step(a)

        frames.append(frame)
        actions.append(np.asarray(a, dtype=np.float32))
        rewards.append(float(reward))
        dones.append(bool(done))

        if done:
            break

    frames_arr = np.stack(frames, axis=0).astype(np.uint8, copy=False)
    actions_arr = np.stack(actions, axis=0).astype(np.float32, copy=False)
    dones_arr = np.asarray(dones, dtype=bool)
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    return Episode(frames=frames_arr, actions=actions_arr, dones=dones_arr, rewards=rewards_arr)

