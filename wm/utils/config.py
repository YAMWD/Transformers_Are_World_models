from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = OmegaConf.load(str(path))
    out = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(out, dict):
        raise ValueError("Config must be a mapping at top level.")
    return out  # type: ignore[return-value]

