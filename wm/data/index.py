from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EpisodeIndex:
    root: Path
    episode_paths: tuple[Path, ...]

    def __len__(self) -> int:
        return len(self.episode_paths)

    @staticmethod
    def build(root: str | Path) -> "EpisodeIndex":
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(root)
        eps = sorted(root.glob("episodes/*.npz"))
        return EpisodeIndex(root=root, episode_paths=tuple(eps))

    def split(self, *, val_frac: float, seed: int) -> tuple["EpisodeIndex", "EpisodeIndex"]:
        if not (0.0 < val_frac < 1.0):
            raise ValueError("val_frac must be in (0,1)")
        rng = np.random.default_rng(seed)
        idx = np.arange(len(self.episode_paths))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_idx = set(idx[:n_val].tolist())
        train = [p for i, p in enumerate(self.episode_paths) if i not in val_idx]
        val = [p for i, p in enumerate(self.episode_paths) if i in val_idx]
        return EpisodeIndex(self.root, tuple(train)), EpisodeIndex(self.root, tuple(val))

