from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from wm.data.episode import Episode, load_episode


@dataclass(frozen=True)
class SampleBatch:
    frames: torch.Tensor  # (B,T,3,H,W) float32 in [0,1]
    actions: torch.Tensor  # (B,T,A) float32
    dones: torch.Tensor  # (B,T) float32 {0,1}


def _to_chw_float(frames_u8: np.ndarray) -> torch.Tensor:
    # (T,H,W,3) uint8 -> (T,3,H,W) float32 [0,1]
    x = torch.from_numpy(frames_u8).permute(0, 3, 1, 2).contiguous()
    return x.float().div(255.0)


def sample_random_frames(
    episode_paths: list[Path], *, batch_size: int, rng: np.random.Generator
) -> torch.Tensor:
    """Returns (B,3,H,W) float32 frames sampled uniformly from episodes."""
    frames = []
    for _ in range(batch_size):
        ep_path = episode_paths[int(rng.integers(0, len(episode_paths)))]
        ep = load_episode(ep_path)
        t = int(rng.integers(0, ep.length))
        frames.append(_to_chw_float(ep.frames[t : t + 1])[0])
    return torch.stack(frames, dim=0)


def sample_sequence_batch(
    episode_paths: list[Path],
    *,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
) -> SampleBatch:
    """
    Sample sequences of length seq_len from random episodes.

    Returned frames include the full seq_len (for encoding to z targets).
    actions/dones are aligned to frames: action at t transitions to frame t+1.
    """
    frames_b: list[torch.Tensor] = []
    actions_b: list[torch.Tensor] = []
    dones_b: list[torch.Tensor] = []

    attempts = 0
    max_attempts = batch_size * 50
    while len(frames_b) < batch_size and attempts < max_attempts:
        attempts += 1
        ep_path = episode_paths[int(rng.integers(0, len(episode_paths)))]
        ep: Episode = load_episode(ep_path)
        if ep.length < seq_len:
            continue
        start = int(rng.integers(0, ep.length - seq_len + 1))
        end = start + seq_len
        frames_b.append(_to_chw_float(ep.frames[start:end]))
        actions_b.append(torch.from_numpy(ep.actions[start:end]).float())
        dones_b.append(torch.from_numpy(ep.dones[start:end]).float())

    if len(frames_b) != batch_size:
        raise RuntimeError(
            f"Could not sample enough sequences of len={seq_len}; got {len(frames_b)} "
            f"of {batch_size} after {attempts} attempts. Consider collecting longer episodes."
        )
    return SampleBatch(
        frames=torch.stack(frames_b, dim=0),
        actions=torch.stack(actions_b, dim=0),
        dones=torch.stack(dones_b, dim=0),
    )
