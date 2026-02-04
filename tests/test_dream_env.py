from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from wm.data.episode import Episode, save_episode
from wm.envs.dream import DreamEnv
from wm.models.vit_wm import ViTWMConfig, ViTWorldModel


def test_dream_env_reset_and_step(tmp_path: Path) -> None:
    # Create a tiny fake episode on disk.
    frames = (np.random.rand(20, 64, 64, 3) * 255).astype(np.uint8)
    actions = np.random.randn(20, 3).astype(np.float32)
    dones = np.zeros(20, dtype=bool)
    ep = Episode(frames=frames, actions=actions, dones=dones)
    ep_path = tmp_path / "episodes" / "episode_000000.npz"
    save_episode(ep_path, ep)

    cfg = ViTWMConfig(
        image_size=64,
        patch_size=8,
        z_dim=32,
        action_dim=3,
        d_model=128,
        depth=2,
        heads=4,
        mlp_ratio=4,
        dropout=0.0,
        l_ctx=32,
        mdn_k=5,
        predict_done=True,
    )
    model = ViTWorldModel(cfg, with_decoder=False)
    device = torch.device("cpu")
    model.to(device)

    env = DreamEnv(model=model, episode_paths=[ep_path], tau=1.15, device=device, max_steps=5, seed=0)
    obs, info = env.reset()
    assert obs.shape == (cfg.z_dim + cfg.d_model,)
    assert "tau" in info

    a = torch.zeros(cfg.action_dim)
    obs2, r, done, info2 = env.step(a)
    assert obs2.shape == obs.shape
    assert isinstance(r, float)
    assert "tau" in info2
    assert "done_prob" in info2
    assert isinstance(done, bool)

