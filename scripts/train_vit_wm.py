from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wm.data.index import EpisodeIndex
from wm.data.sampling import sample_random_frames, sample_sequence_batch
from wm.losses.mdn import MDNParams, mdn_nll
from wm.losses.vae import beta_anneal, kl_standard_normal
from wm.models.vit_wm import ViTWMConfig, ViTWorldModel
from wm.utils.checkpoint import save_checkpoint
from wm.utils.config import load_config
from wm.utils.device import default_device
from wm.utils.logging import write_json
from wm.utils.seeding import seed_everything


def _estimate_mean_len(paths: list[Path], *, max_files: int = 200) -> float:
    from wm.data.episode import load_episode

    if len(paths) == 0:
        return 0.0
    lens = []
    for p in paths[: max_files]:
        try:
            lens.append(load_episode(p).length)
        except Exception:
            continue
    if not lens:
        return 0.0
    return float(np.mean(lens))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, default=None, help="Checkpoint output path (default: checkpoints/<task>/vit_wm.pt)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 0)))

    device = torch.device(args.device) if args.device else default_device()

    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    idx = EpisodeIndex.build(root)
    train_idx, val_idx = idx.split(val_frac=0.1, seed=int(cfg.get("seed", 0)))
    train_paths = list(train_idx.episode_paths)
    val_paths = list(val_idx.episode_paths)

    frame_size = int(data_cfg.get("frame_size", 64))
    model_cfg = cfg["model"]

    predict_done = bool(model_cfg.get("done_head", False))
    wm_cfg = ViTWMConfig(
        image_size=frame_size,
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
        predict_done=predict_done,
    )

    model = ViTWorldModel(wm_cfg, with_decoder=True).to(device)

    train_cfg = cfg["train"]
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    bs_frames = int(train_cfg["batch_size_frames"])
    bs_seq = int(train_cfg["batch_size_seq"])
    seq_len = int(train_cfg["seq_len"])
    epochs_v = int(train_cfg["epochs_v_warmup"])
    epochs_joint = int(train_cfg["epochs_joint"])
    alpha_v = float(train_cfg["alpha_v_in_joint"])
    kl_frac = float(train_cfg["kl_anneal_frac"])

    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    mean_len = _estimate_mean_len(train_paths)
    total_frames = max(1.0, mean_len * max(1, len(train_paths)))
    steps_per_epoch_v = int(math.ceil(total_frames / bs_frames))
    steps_per_epoch_seq = int(math.ceil((total_frames / max(1, seq_len)) / bs_seq))

    print(f"[train] device={device} train_eps={len(train_paths)} val_eps={len(val_paths)}")
    print(f"[train] mean_len≈{mean_len:.1f} total_frames≈{total_frames:.0f}")
    print(f"[train] steps/epoch: V={steps_per_epoch_v}  M={steps_per_epoch_seq}")

    global_step = 0

    # Phase 1: V warmup
    total_steps_v = epochs_v * steps_per_epoch_v
    for epoch in range(epochs_v):
        model.train()
        for _ in range(steps_per_epoch_v):
            x = sample_random_frames(train_paths, batch_size=bs_frames, rng=rng).to(device)
            _z, mu, logsigma, recon = model.forward_visual(x, sample=True)
            if recon is None:
                raise RuntimeError("Decoder required for V warmup.")
            recon_loss = F.mse_loss(recon, x)
            kl = kl_standard_normal(mu, logsigma)
            beta = beta_anneal(global_step, total_steps_v, kl_frac)
            loss = recon_loss + beta * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if global_step % 50 == 0:
                print(
                    f"[V] step={global_step} loss={loss.item():.4f} recon={recon_loss.item():.4f} kl={kl.item():.4f} beta={beta:.3f}"
                )
            global_step += 1

    # Phase 2: joint V + M
    total_steps_joint = epochs_joint * steps_per_epoch_seq
    for epoch in range(epochs_joint):
        model.train()
        for _ in range(steps_per_epoch_seq):
            batch = sample_sequence_batch(train_paths, batch_size=bs_seq, seq_len=seq_len, rng=rng)
            frames = batch.frames.to(device)  # (B,T,3,H,W)
            actions = batch.actions.to(device)  # (B,T,A)
            dones = batch.dones.to(device)  # (B,T)

            b, t, c, h, w = frames.shape
            x_flat = frames.view(b * t, c, h, w)
            z_flat, mu_flat, logsigma_flat = model.encode(x_flat, sample=False)
            z_all = z_flat.view(b, t, -1)

            z_in = z_all[:, :-1, :]
            z_tgt = z_all[:, 1:, :]
            a_in = actions[:, :-1, :]
            done_tgt = dones[:, :-1]

            pi_logits, mdn_mu, mdn_logsigma, done_logit, _y = model.forward_dynamics(z_in, a_in)
            mdn_params = MDNParams(pi_logits=pi_logits, mu=mdn_mu, logsigma=mdn_logsigma)
            loss_m = mdn_nll(mdn_params, z_tgt)
            if done_logit is not None:
                loss_done = F.binary_cross_entropy_with_logits(done_logit, done_tgt)
                loss_m = loss_m + loss_done

            # Anchor representation with a small V loss on the first frame in the sequence.
            x_v = frames[:, 0, :, :, :]
            _zv, muv, lsv, reconv = model.forward_visual(x_v, sample=True)
            if reconv is None:
                raise RuntimeError("Decoder required for joint training.")
            recon_loss_v = F.mse_loss(reconv, x_v)
            kl_v = kl_standard_normal(muv, lsv)
            beta = beta_anneal(global_step, total_steps_joint, kl_frac)
            loss_v = recon_loss_v + beta * kl_v

            loss = loss_m + alpha_v * loss_v

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if global_step % 50 == 0:
                print(
                    f"[J] step={global_step} loss={loss.item():.4f} "
                    f"M={loss_m.item():.4f} V={loss_v.item():.4f} beta={beta:.3f}"
                )
            global_step += 1

        # Simple validation probe
        model.eval()
        with torch.no_grad():
            vb = sample_sequence_batch(val_paths, batch_size=min(bs_seq, len(val_paths)), seq_len=seq_len, rng=rng)
            vframes = vb.frames.to(device)
            vactions = vb.actions.to(device)
            b2, t2, c2, h2, w2 = vframes.shape
            vz_flat, _vmu, _vls = model.encode(vframes.view(b2 * t2, c2, h2, w2), sample=False)
            vz_all = vz_flat.view(b2, t2, -1)
            vpi, vmu_m, vls_m, vdone, _vy = model.forward_dynamics(vz_all[:, :-1], vactions[:, :-1])
            vloss = mdn_nll(MDNParams(vpi, vmu_m, vls_m), vz_all[:, 1:])
            if vdone is not None:
                vloss = vloss + F.binary_cross_entropy_with_logits(vdone, vb.dones.to(device)[:, :-1])
            print(f"[val] epoch={epoch+1}/{epochs_joint} mdn_nll={vloss.item():.4f}")

    task = "carracing" if "env_id" in data_cfg else "takecover"
    out_path = Path(args.out) if args.out else Path("checkpoints") / task / "vit_wm.pt"
    save_checkpoint(out_path, model=model, meta={"config": cfg, "task": task})
    write_json(out_path.with_suffix(".meta.json"), {"config": cfg, "task": task, "checkpoint": str(out_path)})
    print(f"[train] saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()

