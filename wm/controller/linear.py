from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ControllerSpec:
    z_dim: int
    h_dim: int
    action_dim: int
    variant: str  # "z_only" | "z_plus_h" | "hidden40"

    def feature_dim(self) -> int:
        if self.variant == "z_only":
            return self.z_dim
        if self.variant == "z_plus_h":
            return self.z_dim + self.h_dim
        if self.variant == "hidden40":
            return self.z_dim
        raise ValueError(f"Unknown controller variant: {self.variant}")

    def num_params(self) -> int:
        feat = self.feature_dim()
        if self.variant in ("z_only", "z_plus_h"):
            return self.action_dim * feat + self.action_dim
        if self.variant == "hidden40":
            hidden = 40
            # W1: hidden x feat, b1: hidden, W2: action x hidden, b2: action
            return hidden * feat + hidden + self.action_dim * hidden + self.action_dim
        raise ValueError(f"Unknown controller variant: {self.variant}")


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def controller_act(params: np.ndarray, spec: ControllerSpec, *, z: np.ndarray, h: np.ndarray | None) -> np.ndarray:
    """
    Compute action logits/values from controller params and features.
    Returns raw outputs (pre-bounding); mapping to env action space is done separately.
    """
    if spec.variant == "z_only":
        feat = z
    elif spec.variant == "z_plus_h":
        if h is None:
            raise ValueError("h is required for z_plus_h")
        feat = np.concatenate([z, h], axis=0)
    elif spec.variant == "hidden40":
        feat = z
    else:
        raise ValueError(f"Unknown controller variant: {spec.variant}")

    feat = feat.astype(np.float32, copy=False)
    action_dim = spec.action_dim

    if spec.variant in ("z_only", "z_plus_h"):
        feat_dim = feat.shape[0]
        w_size = action_dim * feat_dim
        w = params[:w_size].reshape(action_dim, feat_dim)
        b = params[w_size : w_size + action_dim]
        return (w @ feat + b).astype(np.float32, copy=False)

    # hidden40
    hidden = 40
    feat_dim = feat.shape[0]
    idx = 0
    w1 = params[idx : idx + hidden * feat_dim].reshape(hidden, feat_dim)
    idx += hidden * feat_dim
    b1 = params[idx : idx + hidden]
    idx += hidden
    w2 = params[idx : idx + action_dim * hidden].reshape(action_dim, hidden)
    idx += action_dim * hidden
    b2 = params[idx : idx + action_dim]

    h1 = _tanh(w1 @ feat + b1)
    return (w2 @ h1 + b2).astype(np.float32, copy=False)


def map_action_carracing(raw: np.ndarray) -> np.ndarray:
    if raw.shape != (3,):
        raise ValueError("CarRacing expects raw action shape (3,)")
    o = np.tanh(raw)
    steer = float(o[0])
    gas = float((o[1] + 1.0) / 2.0)
    brake = float((o[2] + 1.0) / 2.0)
    return np.array([steer, gas, brake], dtype=np.float32)


def map_action_takecover(raw: np.ndarray) -> np.ndarray:
    # raw scalar -> {-1,0,+1} as float32 with shape (1,)
    if raw.size != 1:
        raise ValueError("TakeCover expects raw action scalar")
    o = float(np.tanh(raw.reshape(())))
    if o < -1.0 / 3.0:
        a = -1.0
    elif o > 1.0 / 3.0:
        a = 1.0
    else:
        a = 0.0
    return np.array([a], dtype=np.float32)

