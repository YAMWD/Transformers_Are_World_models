from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from wm.controller.linear import ControllerSpec, controller_act
from wm.envs.rollout import obs_to_frame_u8
from wm.models.vit_wm import ViTWorldModel


@dataclass
class WMHistory:
    """Tracks (z,a) token history to compute transformer memory h via dynamics mode."""

    z_hist: list[torch.Tensor]
    a_hist: list[torch.Tensor]

    @staticmethod
    def empty() -> "WMHistory":
        return WMHistory(z_hist=[], a_hist=[])

    def append(self, *, z: torch.Tensor, a: torch.Tensor, l_ctx: int) -> None:
        self.z_hist.append(z.detach())
        self.a_hist.append(a.detach())
        if len(self.z_hist) > l_ctx:
            self.z_hist = self.z_hist[-l_ctx:]
            self.a_hist = self.a_hist[-l_ctx:]

    @torch.no_grad()
    def compute_h_next(self, model: ViTWorldModel) -> torch.Tensor:
        if len(self.z_hist) == 0:
            return torch.zeros(model.cfg.d_model, device=next(model.parameters()).device)
        z_seq = torch.stack(self.z_hist, dim=0).unsqueeze(0)
        a_seq = torch.stack(self.a_hist, dim=0).unsqueeze(0)
        _pi, _mu, _ls, _done, y = model.forward_dynamics(z_seq, a_seq)
        return y[:, -1, :][0]


def rollout_real_env(
    *,
    env,
    model: ViTWorldModel,
    controller_params: np.ndarray,
    controller_spec: ControllerSpec,
    map_action: Callable[[np.ndarray], np.ndarray],
    max_steps: int,
    device: torch.device,
) -> float:
    """
    Rollout in a *real* env using features (z,h) from the model.
    Env API expected: reset()->obs, step(action)->(obs,reward,done,info)
    """
    model.eval()
    with torch.no_grad():
        obs = env.reset()

        # Init z0
        frame0 = obs_to_frame_u8(obs, size=model.cfg.image_size)
        x0 = torch.from_numpy(frame0).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)
        z0, _mu0, _ls0 = model.encode(x0, sample=False)
        z_t = z0[0]
    h_t = torch.zeros(model.cfg.d_model, device=device)
    hist = WMHistory.empty()

    total = 0.0
    for _t in range(max_steps):
        with torch.no_grad():
            z_np = z_t.detach().cpu().numpy().astype(np.float32, copy=False)
            h_np = h_t.detach().cpu().numpy().astype(np.float32, copy=False)
            raw = controller_act(controller_params, controller_spec, z=z_np, h=h_np)
            a = map_action(raw)

            obs_next, reward, done, _info = env.step(a)
            total += float(reward)

            # Teacher-force next z from real obs
            frame_next = obs_to_frame_u8(obs_next, size=model.cfg.image_size)
            x_next = torch.from_numpy(frame_next).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)
            z_next, _mu, _ls = model.encode(x_next, sample=False)

            # Update model memory using (z_t, a_t)
            a_t = torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device).view(model.cfg.action_dim)
            hist.append(z=z_t, a=a_t, l_ctx=model.cfg.l_ctx)
            h_next = hist.compute_h_next(model)

            z_t = z_next[0]
            h_t = h_next

            if done:
                break
    return float(total)


def rollout_dream_env(
    *,
    env,
    controller_params: np.ndarray,
    controller_spec: ControllerSpec,
    map_action: Callable[[np.ndarray], np.ndarray],
    max_steps: int,
) -> float:
    """
    Rollout in a DreamEnv that returns obs=torch.cat([z,h]).
    Env API expected: reset()->(obs,info), step(action_tensor)->(obs,reward,done,info)
    """
    obs, _info = env.reset()
    total = 0.0

    z_dim = controller_spec.z_dim
    for _t in range(max_steps):
        obs_np = obs.detach().cpu().numpy().astype(np.float32, copy=False)
        z = obs_np[:z_dim]
        h = obs_np[z_dim:]
        raw = controller_act(controller_params, controller_spec, z=z, h=h if controller_spec.variant != "z_only" else None)
        a_np = map_action(raw)
        a_t = torch.from_numpy(a_np).to(obs.device)
        obs, reward, done, _info = env.step(a_t)
        total += float(reward)
        if done:
            break
    return float(total)
