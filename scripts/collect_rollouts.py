from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from wm.data.episode import save_episode
from wm.envs.gym_compat import GymCompatWrapper
from wm.envs.rollout import RolloutSpec, collect_episode
from wm.utils.config import load_config
from wm.utils.logging import write_json
from wm.utils.seeding import seed_everything


def _random_policy_carracing(_obs: Any) -> np.ndarray:
    steer = np.random.uniform(-1.0, 1.0)
    gas = np.random.uniform(0.0, 1.0)
    brake = np.random.uniform(0.0, 1.0)
    return np.array([steer, gas, brake], dtype=np.float32)


def _random_policy_takecover(_obs: Any) -> np.ndarray:
    a = np.random.choice([-1.0, 0.0, 1.0])
    return np.array([a], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config (configs/*.yaml)")
    ap.add_argument("--start", type=int, default=0, help="Episode index to start writing at")
    ap.add_argument("--episodes", type=int, default=None, help="Override number of episodes to collect")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 0)))

    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    episodes_dir = root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    n_episodes = int(args.episodes if args.episodes is not None else data_cfg["episodes"])
    frame_size = int(data_cfg.get("frame_size", 64))

    meta = {
        "config": cfg,
        "frame_size": frame_size,
    }

    if "env_id" in data_cfg:
        from wm.envs.carracing import make_carracing

        env = GymCompatWrapper(make_carracing(str(data_cfg["env_id"]), seed=int(cfg.get("seed", 0))))
        spec = RolloutSpec(frame_size=frame_size, max_steps=1000)
        policy = _random_policy_carracing
        meta["task"] = "carracing"
        meta["env_id"] = str(data_cfg["env_id"])
        meta["action_dim"] = 3
    else:
        from wm.envs.vizdoom_takecover import TakeCoverConfig, TakeCoverEnv

        env = TakeCoverEnv(
            TakeCoverConfig(
                frame_size=frame_size,
                scenario_name=str(data_cfg.get("scenario", "take_cover")),
                seed=int(cfg.get("seed", 0)),
            )
        )
        spec = RolloutSpec(frame_size=frame_size, max_steps=2100)
        policy = _random_policy_takecover
        meta["task"] = "takecover"
        meta["scenario"] = str(data_cfg.get("scenario", "take_cover"))
        meta["action_dim"] = 1

    write_json(root / "meta.json", meta)

    for i in range(args.start, args.start + n_episodes):
        ep = collect_episode(env, policy=policy, spec=spec)
        out = episodes_dir / f"episode_{i:06d}.npz"
        save_episode(out, ep)
        if (i + 1) % 10 == 0:
            print(f"[collect] wrote {out}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()

