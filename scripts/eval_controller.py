from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from wm.controller.linear import ControllerSpec, map_action_carracing, map_action_takecover
from wm.controller.runner import rollout_dream_env, rollout_real_env
from wm.data.index import EpisodeIndex
from wm.envs.dream import DreamEnv
from wm.envs.gym_compat import GymCompatWrapper
from wm.models.vit_wm import ViTWMConfig, ViTWorldModel
from wm.utils.checkpoint import load_checkpoint
from wm.utils.config import load_config
from wm.utils.device import default_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--controller", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dream", action="store_true", help="Evaluate in dream env (TakeCover only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device) if args.device else default_device()

    data_cfg = cfg["data"]
    task = "carracing" if "env_id" in data_cfg else "takecover"

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
    ckpt = load_checkpoint(Path(args.checkpoint))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ctrl_npz = np.load(args.controller, allow_pickle=True)
    params = ctrl_npz["params"].astype(np.float32)
    spec_dict = ctrl_npz["spec"].item() if hasattr(ctrl_npz["spec"], "item") else ctrl_npz["spec"]
    ctrl_spec = ControllerSpec(**spec_dict)

    scores = []
    if task == "carracing":
        from wm.envs.carracing import make_carracing

        env = GymCompatWrapper(make_carracing(str(data_cfg["env_id"])))
        for _ in range(args.episodes):
            scores.append(
                rollout_real_env(
                    env=env,
                    model=model,
                    controller_params=params,
                    controller_spec=ctrl_spec,
                    map_action=map_action_carracing,
                    max_steps=1000,
                    device=device,
                )
            )
    else:
        if args.dream:
            idx = EpisodeIndex.build(Path(data_cfg["root"]))
            train_idx, _ = idx.split(val_frac=0.1, seed=int(cfg.get("seed", 0)))
            episode_paths = list(train_idx.episode_paths)
            tau = float(cfg.get("dream", {}).get("tau", 1.15))
            dream_env = DreamEnv(model=model, episode_paths=episode_paths, tau=tau, device=device, max_steps=2100)
            for _ in range(args.episodes):
                scores.append(
                    rollout_dream_env(
                        env=dream_env,
                        controller_params=params,
                        controller_spec=ctrl_spec,
                        map_action=map_action_takecover,
                        max_steps=2100,
                    )
                )
        else:
            from wm.envs.vizdoom_takecover import TakeCoverConfig, TakeCoverEnv

            env = TakeCoverEnv(TakeCoverConfig(frame_size=int(data_cfg.get("frame_size", 64))))
            # Evaluate in real env via teacher-forced model memory.
            for _ in range(args.episodes):
                scores.append(
                    rollout_real_env(
                        env=env,
                        model=model,
                        controller_params=params,
                        controller_spec=ctrl_spec,
                        map_action=map_action_takecover,
                        max_steps=2100,
                        device=device,
                    )
                )

    mean = float(np.mean(scores)) if scores else float("nan")
    std = float(np.std(scores)) if scores else float("nan")
    print(f"[eval] task={task} episodes={len(scores)} mean={mean:.2f} std={std:.2f}")


if __name__ == "__main__":
    main()

