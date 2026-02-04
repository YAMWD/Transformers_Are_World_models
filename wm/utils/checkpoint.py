from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, *, model, meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "state_dict": model.state_dict()}
    torch.save(payload, path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return torch.load(path, map_location="cpu")

