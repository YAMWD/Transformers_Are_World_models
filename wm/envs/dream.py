from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wm.data.episode import load_episode
from wm.losses.mdn import MDNParams, mdn_sample
from wm.models.vit_wm import ViTWorldModel


@dataclass
class DreamState:
    z: torch.Tensor  # (z_dim,)
    h: torch.Tensor  # (d_model,)
    z_hist: list[torch.Tensor]  # list[(z_dim,)]
    a_hist: list[torch.Tensor]  # list[(a_dim,)]


class DreamEnv:
    """
    Latent-space environment driven by the model's dynamics mode.

    This is intentionally gym-like (reset/step) but does not require gym.
    """

    def __init__(
        self,
        *,
        model: ViTWorldModel,
        episode_paths: list[Path],
        tau: float,
        device: torch.device,
        reward_alive: float = 1.0,
        max_steps: int | None = None,
        seed: int = 0,
    ) -> None:
        if tau <= 0.0:
            raise ValueError("tau must be > 0")
        if len(episode_paths) == 0:
            raise ValueError("episode_paths is empty")

        self.model = model
        self.episode_paths = episode_paths
        self.tau = float(tau)
        self.device = device
        self.reward_alive = float(reward_alive)
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)

        self.state: DreamState | None = None
        self.steps = 0

    @property
    def z_dim(self) -> int:
        return self.model.cfg.z_dim

    @property
    def h_dim(self) -> int:
        return self.model.cfg.d_model

    @property
    def action_dim(self) -> int:
        return self.model.cfg.action_dim

    def _sample_initial_frame(self) -> np.ndarray:
        ep_path = self.episode_paths[int(self._rng.integers(0, len(self.episode_paths)))]
        ep = load_episode(ep_path)
        # Paper setup: start from episode start.
        return ep.frames[0]

    @torch.no_grad()
    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        self.model.eval()
        self.steps = 0

        frame_u8 = self._sample_initial_frame()  # (H,W,3) uint8
        x = torch.from_numpy(frame_u8).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(self.device)
        z0, _mu, _logsigma = self.model.encode(x, sample=False)
        z0 = z0[0]
        h0 = torch.zeros(self.h_dim, device=self.device)
        self.state = DreamState(z=z0, h=h0, z_hist=[], a_hist=[])
        obs = torch.cat([z0, h0], dim=0)
        return obs, {"tau": self.tau}

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        self.model.eval()
        self.steps += 1

        a = action.to(self.device).float().view(self.action_dim)

        # Append current token (z_t, a_t) to context.
        self.state.z_hist.append(self.state.z.detach())
        self.state.a_hist.append(a.detach())

        # Truncate context window.
        if len(self.state.z_hist) > self.model.cfg.l_ctx:
            self.state.z_hist = self.state.z_hist[-self.model.cfg.l_ctx :]
            self.state.a_hist = self.state.a_hist[-self.model.cfg.l_ctx :]

        z_seq = torch.stack(self.state.z_hist, dim=0).unsqueeze(0)  # (1,S,z)
        a_seq = torch.stack(self.state.a_hist, dim=0).unsqueeze(0)  # (1,S,a)

        pi_logits, mu, logsigma, done_logit, y = self.model.forward_dynamics(z_seq, a_seq)
        # Take last position output: predicts z_{t+1}.
        params = MDNParams(pi_logits=pi_logits[:, -1], mu=mu[:, -1], logsigma=logsigma[:, -1])
        z_next = mdn_sample(params, tau=self.tau)[0]  # (z_dim,)

        h_next = y[:, -1, :][0]  # (d_model,)

        done_prob = None
        done = False
        if done_logit is not None:
            done_prob = torch.sigmoid(done_logit[:, -1])[0].item()
            done = bool(done_prob > 0.5)

        if self.max_steps is not None and self.steps >= self.max_steps:
            done = True

        reward = self.reward_alive if not done else 0.0

        self.state.z = z_next
        self.state.h = h_next

        obs = torch.cat([z_next, h_next], dim=0)
        info: dict[str, Any] = {"tau": self.tau}
        if done_prob is not None:
            info["done_prob"] = float(done_prob)
        return obs, float(reward), done, info

