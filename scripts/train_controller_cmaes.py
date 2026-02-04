from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from wm.controller.linear import (
    ControllerSpec,
    map_action_carracing,
    map_action_takecover,
)
from wm.controller.runner import rollout_dream_env, rollout_real_env
from wm.data.index import EpisodeIndex
from wm.envs.dream import DreamEnv
from wm.envs.gym_compat import GymCompatWrapper
from wm.es.cmaes import CMAES, CMAESConfig
from wm.models.vit_wm import ViTWMConfig, ViTWorldModel
from wm.utils.checkpoint import load_checkpoint
from wm.utils.config import load_config
from wm.utils.device import default_device
from wm.utils.logging import write_json
from wm.utils.seeding import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, default=None, help="Output controller params path")
    ap.add_argument("--gens", type=int, default=None, help="Override generations")
    ap.add_argument("--debug", action="store_true", help="Use tiny eval counts for quick debugging")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 0)))
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

    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else Path("checkpoints") / task / "vit_wm.pt"
    )
    ckpt = load_checkpoint(ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ctrl_cfg = cfg["controller"]
    variant = str(ctrl_cfg["variant"])

    ctrl_spec = ControllerSpec(
        z_dim=model.cfg.z_dim,
        h_dim=model.cfg.d_model,
        action_dim=3 if task == "carracing" else 1,
        variant=variant,
    )
    n_dim = ctrl_spec.num_params()

    pop = int(ctrl_cfg["population"])
    eval_rollouts = int(ctrl_cfg["eval_rollouts"])
    gens = int(args.gens if args.gens is not None else ctrl_cfg["gens"])
    eval_every = int(ctrl_cfg["eval_every"])

    if args.debug:
        eval_rollouts = 2
        gens = min(gens, 5)
        eval_every = 1

    es = CMAES(CMAESConfig(n_dim=n_dim, population=pop, sigma_init=0.5, seed=int(cfg.get("seed", 0))))

    if task == "carracing":
        from wm.envs.carracing import make_carracing

        env = GymCompatWrapper(make_carracing(str(data_cfg["env_id"]), seed=int(cfg.get("seed", 0))))

        def fitness_fn(params: np.ndarray) -> float:
            return float(
                np.mean(
                    [
                        rollout_real_env(
                            env=env,
                            model=model,
                            controller_params=params,
                            controller_spec=ctrl_spec,
                            map_action=map_action_carracing,
                            max_steps=1000,
                            device=device,
                        )
                        for _ in range(eval_rollouts)
                    ]
                )
            )

        def eval_best(params: np.ndarray) -> float:
            n_eval = int(ctrl_cfg.get("eval_episodes", 100))
            if args.debug:
                n_eval = 5
            return float(
                np.mean(
                    [
                        rollout_real_env(
                            env=env,
                            model=model,
                            controller_params=params,
                            controller_spec=ctrl_spec,
                            map_action=map_action_carracing,
                            max_steps=1000,
                            device=device,
                        )
                        for _ in range(n_eval)
                    ]
                )
            )

    else:
        idx = EpisodeIndex.build(Path(data_cfg["root"]))
        train_idx, _val_idx = idx.split(val_frac=0.1, seed=int(cfg.get("seed", 0)))
        episode_paths = list(train_idx.episode_paths)

        tau = float(cfg.get("dream", {}).get("tau", 1.15))
        dream_env = DreamEnv(model=model, episode_paths=episode_paths, tau=tau, device=device, max_steps=2100)

        def fitness_fn(params: np.ndarray) -> float:
            return float(
                np.mean(
                    [
                        rollout_dream_env(
                            env=dream_env,
                            controller_params=params,
                            controller_spec=ctrl_spec,
                            map_action=map_action_takecover,
                            max_steps=2100,
                        )
                        for _ in range(eval_rollouts)
                    ]
                )
            )

        def eval_best(params: np.ndarray) -> float:
            n_eval = int(ctrl_cfg.get("dream_eval_episodes", 256))
            if args.debug:
                n_eval = 10
            return float(
                np.mean(
                    [
                        rollout_dream_env(
                            env=dream_env,
                            controller_params=params,
                            controller_spec=ctrl_spec,
                            map_action=map_action_takecover,
                            max_steps=2100,
                        )
                        for _ in range(n_eval)
                    ]
                )
            )

    logs = []
    for gen in range(gens):
        candidates = es.ask()
        fitness = [fitness_fn(c) for c in candidates]
        es.tell(candidates, fitness)

        best_f = float(es.best_f) if es.best_f is not None else float("nan")
        print(f"[es] gen={gen+1}/{gens} best_f={best_f:.2f} sigma={es.sigma:.4f}")
        logs.append({"gen": gen + 1, "best_f": best_f, "sigma": float(es.sigma)})

        if (gen + 1) % eval_every == 0 and es.best_x is not None:
            score = eval_best(es.best_x)
            print(f"[eval] gen={gen+1} score={score:.2f}")
            logs[-1]["eval_score"] = float(score)

    out_path = Path(args.out) if args.out else Path("checkpoints") / task / f"controller_{variant}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, params=es.best_x.astype(np.float32), spec=vars(ctrl_spec), logs=logs)
    write_json(out_path.with_suffix(".meta.json"), {"task": task, "checkpoint": str(ckpt_path), "controller": str(out_path)})
    print(f"[es] saved controller to {out_path}")

    # Paper-faithful transfer evaluation for TakeCover: dream -> real env.
    if task == "takecover" and es.best_x is not None and not args.debug:
        n_real = int(ctrl_cfg.get("real_eval_episodes", 100))
        try:
            from wm.envs.vizdoom_takecover import TakeCoverConfig, TakeCoverEnv

            real_env = TakeCoverEnv(TakeCoverConfig(frame_size=int(data_cfg.get("frame_size", 64))))
            real_scores = [
                rollout_real_env(
                    env=real_env,
                    model=model,
                    controller_params=es.best_x,
                    controller_spec=ctrl_spec,
                    map_action=map_action_takecover,
                    max_steps=2100,
                    device=device,
                )
                for _ in range(n_real)
            ]
            print(f"[transfer] real_env mean={float(np.mean(real_scores)):.2f} std={float(np.std(real_scores)):.2f}")
        except Exception as e:
            print(f"[transfer] skipped real TakeCover eval (missing vizdoom or scenario): {e}")


if __name__ == "__main__":
    main()
