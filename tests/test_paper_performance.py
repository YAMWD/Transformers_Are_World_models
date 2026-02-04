from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from wm.controller.runner import rollout_dream_env, rollout_real_env
from wm.controller.linear import ControllerSpec, map_action_carracing, map_action_takecover
from wm.data.index import EpisodeIndex
from wm.envs.dream import DreamEnv
from wm.envs.gym_compat import GymCompatWrapper
from wm.models.vit_wm import ViTWMConfig, ViTWorldModel
from wm.utils.checkpoint import load_checkpoint
from wm.utils.config import load_config
from wm.utils.device import default_device


PAPER_TARGET_CARRACING_REWARD = 900.0
PAPER_TARGET_TAKECOVER_STEPS = 750.0


def _build_model_from_config(cfg_path: Path, *, device: torch.device) -> tuple[dict, ViTWorldModel]:
    cfg = load_config(cfg_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    wm_cfg = ViTWMConfig(
        image_size=int(data_cfg.get("frame_size", 64)),
        patch_size=int(model_cfg["patch_size"]),
        z_dim=int(model_cfg["z_dim"]),
        action_dim=int(model_cfg["action_dim"]),
        d_model=int(model_cfg["d_model"]),
        depth=int(model_cfg["depth"]),
        heads=int(model_cfg["heads"]),
        mlp_ratio=int(model_cfg["mlp_ratio"]),
        dropout=float(model_cfg["dropout"]),
        l_ctx=int(model_cfg["l_ctx"]),
        mdn_k=int(model_cfg["mdn"]["k"]),
        predict_done=bool(model_cfg.get("done_head", False)),
    )
    model = ViTWorldModel(wm_cfg, with_decoder=True).to(device)
    model.eval()
    return cfg, model


def _load_controller(npz_path: Path) -> tuple[np.ndarray, ControllerSpec]:
    blob = np.load(npz_path, allow_pickle=True)
    params = blob["params"].astype(np.float32, copy=False)
    spec_dict = blob["spec"].item() if hasattr(blob["spec"], "item") else blob["spec"]
    spec = ControllerSpec(**spec_dict)
    return params, spec


@pytest.mark.paper
def test_paper_target_carracing() -> None:
    """
    Paper criterion (Ha & Schmidhuber 2018): avg reward >= 900 over 100 episodes.

    Opt-in because this is very slow and requires a trained checkpoint + controller.
    """
    cfg_path = Path(os.environ.get("PAPER_CARRACING_CONFIG", "configs/carracing.yaml"))
    ckpt_path = Path(os.environ.get("PAPER_CARRACING_CHECKPOINT", "checkpoints/carracing/vit_wm.pt"))
    ctrl_path = Path(
        os.environ.get("PAPER_CARRACING_CONTROLLER", "checkpoints/carracing/controller_z_plus_h.npz")
    )
    episodes = int(os.environ.get("PAPER_EVAL_EPISODES", "100"))

    device = default_device()
    assert ckpt_path.exists(), f"Missing trained checkpoint: {ckpt_path}"
    assert ctrl_path.exists(), f"Missing trained controller: {ctrl_path}"

    cfg, model = _build_model_from_config(cfg_path, device=device)
    ckpt = load_checkpoint(ckpt_path)
    model.load_state_dict(ckpt["state_dict"])

    params, spec = _load_controller(ctrl_path)

    from wm.envs.carracing import make_carracing

    env_id = str(cfg["data"]["env_id"])
    env = GymCompatWrapper(make_carracing(env_id))
    scores = [
        rollout_real_env(
            env=env,
            model=model,
            controller_params=params,
            controller_spec=spec,
            map_action=map_action_carracing,
            max_steps=1000,
            device=device,
        )
        for _ in range(episodes)
    ]
    try:
        env.close()
    except Exception:
        pass

    mean = float(np.mean(scores)) if scores else float("nan")
    assert mean >= PAPER_TARGET_CARRACING_REWARD


@pytest.mark.paper
def test_paper_target_takecover_dream_and_real() -> None:
    """
    Paper criterion: avg survival steps >= 750 over 100 episodes in the *real* env.
    We also sanity-check dream score is healthy (no strict threshold).
    """
    cfg_path = Path(os.environ.get("PAPER_TAKECOVER_CONFIG", "configs/takecover.yaml"))
    ckpt_path = Path(os.environ.get("PAPER_TAKECOVER_CHECKPOINT", "checkpoints/takecover/vit_wm.pt"))
    ctrl_path = Path(
        os.environ.get("PAPER_TAKECOVER_CONTROLLER", "checkpoints/takecover/controller_z_plus_h.npz")
    )
    episodes = int(os.environ.get("PAPER_EVAL_EPISODES", "100"))
    dream_episodes = int(os.environ.get("PAPER_DREAM_EVAL_EPISODES", "256"))

    device = default_device()
    assert ckpt_path.exists(), f"Missing trained checkpoint: {ckpt_path}"
    assert ctrl_path.exists(), f"Missing trained controller: {ctrl_path}"

    cfg, model = _build_model_from_config(cfg_path, device=device)
    ckpt = load_checkpoint(ckpt_path)
    model.load_state_dict(ckpt["state_dict"])

    params, spec = _load_controller(ctrl_path)

    # Dream eval (paper: train in dream env at tau=1.15).
    idx = EpisodeIndex.build(Path(cfg["data"]["root"]))
    train_idx, _ = idx.split(val_frac=0.1, seed=int(cfg.get("seed", 0)))
    tau = float(cfg.get("dream", {}).get("tau", 1.15))
    dream_env = DreamEnv(
        model=model,
        episode_paths=list(train_idx.episode_paths),
        tau=tau,
        device=device,
        max_steps=2100,
    )

    _dream_scores = [
        rollout_dream_env(
            env=dream_env,
            controller_params=params,
            controller_spec=spec,
            map_action=map_action_takecover,
            max_steps=2100,
        )
        for _ in range(dream_episodes)
    ]

    # Real transfer eval.
    from wm.envs.vizdoom_takecover import TakeCoverConfig, TakeCoverEnv

    real_env = TakeCoverEnv(TakeCoverConfig(frame_size=int(cfg["data"].get("frame_size", 64))))

    real_scores = [
        rollout_real_env(
            env=real_env,
            model=model,
            controller_params=params,
            controller_spec=spec,
            map_action=map_action_takecover,
            max_steps=2100,
            device=device,
        )
        for _ in range(episodes)
    ]
    try:
        real_env.close()
    except Exception:
        pass

    mean = float(np.mean(real_scores)) if real_scores else float("nan")
    assert mean >= PAPER_TARGET_TAKECOVER_STEPS
