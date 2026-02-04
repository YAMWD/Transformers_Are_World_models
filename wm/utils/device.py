from __future__ import annotations

import os

import torch


def default_device() -> torch.device:
    forced = os.environ.get("WM_DEVICE")
    if forced:
        return torch.device(forced)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

