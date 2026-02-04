from __future__ import annotations

import torch

from wm.models.vit_wm import ViTWMConfig, ViTWorldModel


def _small_cfg(*, z_dim: int = 32, action_dim: int = 3, predict_done: bool = False) -> ViTWMConfig:
    return ViTWMConfig(
        image_size=64,
        patch_size=8,
        z_dim=z_dim,
        action_dim=action_dim,
        d_model=128,
        depth=2,
        heads=4,
        mlp_ratio=4,
        dropout=0.0,
        l_ctx=256,
        mdn_k=5,
        predict_done=predict_done,
    )


def test_visual_forward_shapes_and_range() -> None:
    model = ViTWorldModel(_small_cfg(), with_decoder=True)
    x = torch.rand(2, 3, 64, 64)
    z, mu, logsigma, recon = model.forward_visual(x, sample=True)
    assert z.shape == (2, 32)
    assert mu.shape == (2, 32)
    assert logsigma.shape == (2, 32)
    assert recon is not None and recon.shape == (2, 3, 64, 64)
    assert float(recon.min()) >= 0.0
    assert float(recon.max()) <= 1.0


def test_dynamics_forward_shapes() -> None:
    model = ViTWorldModel(_small_cfg(), with_decoder=False)
    z_seq = torch.randn(3, 11, 32)
    a_seq = torch.randn(3, 11, 3)
    pi, mu, logsigma, done, y = model.forward_dynamics(z_seq, a_seq)
    assert pi.shape == (3, 11, 5)
    assert mu.shape == (3, 11, 5, 32)
    assert logsigma.shape == (3, 11, 5, 32)
    assert done is None
    assert y.shape == (3, 11, 128)


def test_dynamics_done_head_optional() -> None:
    model = ViTWorldModel(_small_cfg(z_dim=64, action_dim=1, predict_done=True), with_decoder=False)
    z_seq = torch.randn(2, 7, 64)
    a_seq = torch.randn(2, 7, 1)
    _pi, _mu, _ls, done, _y = model.forward_dynamics(z_seq, a_seq)
    assert done is not None
    assert done.shape == (2, 7)

