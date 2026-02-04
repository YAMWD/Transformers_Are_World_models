from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _find_vizdoom_scenario_cfg(name: str) -> Path:
    try:
        import vizdoom as vzd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("vizdoom is not installed. Install vizdoom to use TakeCover.") from e

    base = Path(vzd.__file__).resolve().parent
    candidates = [
        base / "scenarios" / f"{name}.cfg",
        base / "scenarios" / name / f"{name}.cfg",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not find VizDoom scenario cfg for '{name}' under {base / 'scenarios'}")


def _resize_u8(frame_hwc: np.ndarray, size: int) -> np.ndarray:
    """
    Resize uint8 HWC RGB to (size,size,3) using torch bilinear if available,
    otherwise fall back to a nearest-neighbor numpy method.
    """
    if frame_hwc.shape[0] == size and frame_hwc.shape[1] == size:
        return frame_hwc

    try:
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(frame_hwc).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        y = (x.clamp(0, 1) * 255.0).byte().squeeze(0).permute(1, 2, 0).contiguous().numpy()
        return y
    except Exception:  # pragma: no cover
        # Very simple NN fallback
        h, w, _ = frame_hwc.shape
        ys = (np.linspace(0, h - 1, size)).astype(np.int64)
        xs = (np.linspace(0, w - 1, size)).astype(np.int64)
        return frame_hwc[ys][:, xs]


@dataclass
class TakeCoverConfig:
    frame_size: int = 64
    scenario_name: str = "take_cover"
    frame_skip: int = 1
    max_steps: int = 2100
    seed: int | None = None


class TakeCoverEnv:
    """
    Minimal VizDoom TakeCover wrapper with a gym-like API:
      reset() -> obs (uint8 HWC RGB, resized to frame_size)
      step(action_scalar) -> obs, reward, done, info

    Action convention (paper-aligned controller output):
      action_scalar in {-1, 0, +1} for left / stay / right
    """

    def __init__(self, cfg: TakeCoverConfig) -> None:
        try:
            import vizdoom as vzd  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("vizdoom is not installed. Install vizdoom to use TakeCover.") from e

        self._vzd = vzd
        self.cfg = cfg
        self.steps = 0

        cfg_path = _find_vizdoom_scenario_cfg(cfg.scenario_name)
        self.game = vzd.DoomGame()
        self.game.load_config(str(cfg_path))
        self.game.set_window_visible(False)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)

        if cfg.seed is not None:
            try:
                self.game.set_seed(int(cfg.seed))
            except Exception:
                pass

        self.game.init()

        # Assume first two buttons correspond to left/right movement for this scenario.
        self.n_buttons = int(self.game.get_available_buttons_size())
        if self.n_buttons < 2:
            raise RuntimeError(f"Expected at least 2 available buttons, got {self.n_buttons}")

    def close(self) -> None:
        self.game.close()

    def reset(self) -> np.ndarray:
        self.game.new_episode()
        self.steps = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        state = self.game.get_state()
        if state is None or state.screen_buffer is None:
            # Episode finished.
            return np.zeros((self.cfg.frame_size, self.cfg.frame_size, 3), dtype=np.uint8)
        buf = state.screen_buffer
        # vizdoom commonly returns CHW for RGB24.
        if buf.ndim == 3 and buf.shape[0] == 3:
            frame = np.transpose(buf, (1, 2, 0))
        else:
            frame = buf
        frame = frame.astype(np.uint8, copy=False)
        return _resize_u8(frame, self.cfg.frame_size)

    def step(self, action_scalar: float | int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        self.steps += 1

        a = int(action_scalar)
        if a not in (-1, 0, 1):
            raise ValueError("action_scalar must be in {-1,0,1}")

        # action vector: first two buttons are assumed left/right.
        action_vec = [False] * self.n_buttons
        if a == -1:
            action_vec[0] = True
        elif a == 1:
            action_vec[1] = True

        _game_reward = self.game.make_action(action_vec, self.cfg.frame_skip)
        done = bool(self.game.is_episode_finished())
        obs = self._get_obs()

        # Paper's implicit reward for TakeCover: time steps alive.
        reward = 0.0 if done else 1.0

        if self.steps >= self.cfg.max_steps:
            done = True

        info: dict[str, Any] = {"game_reward": float(_game_reward)}
        return obs, float(reward), done, info

