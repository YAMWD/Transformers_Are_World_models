# Transformers_Are_World_models

This repo implements a **paper-faithful World Models setup** (Ha & Schmidhuber, 2018) with one key change:

- Replace **V (ConvVAE) + M (MDN-RNN)** with a **single shared transformer backbone** (“ViT-WM”) that supports:
  - **Visual mode:** pixels → latent `z`
  - **Dynamics mode:** `(z, action)` sequence → MDN for `z_{t+1}` (+ done for Doom)
- Keep **C as a small linear controller trained with CMA-ES** (no PPO / gradient RL).

Code lives under `wm/` and runnable entrypoints are in `scripts/`. Configuration defaults are in `configs/`.

## Conda environment

Create + install editable package:

```bash
conda env create -f environment.yml
conda activate vit_wm
```

Run unit tests:

```bash
pytest -q
```

## Quickstart (high level)

1) Collect random rollouts (paper setting: 10,000 episodes):

```bash
python scripts/collect_rollouts.py --config configs/carracing.yaml
python scripts/collect_rollouts.py --config configs/takecover.yaml
```

2) Train the single-backbone world model:

```bash
python scripts/train_vit_wm.py --config configs/carracing.yaml
python scripts/train_vit_wm.py --config configs/takecover.yaml
```

3) Train the linear controller with CMA-ES (no PPO):

```bash
python scripts/train_controller_cmaes.py --config configs/carracing.yaml
python scripts/train_controller_cmaes.py --config configs/takecover.yaml
```

4) Evaluate a controller:

```bash
python scripts/eval_controller.py --config configs/carracing.yaml --checkpoint checkpoints/carracing/vit_wm.pt --controller checkpoints/carracing/controller_z_plus_h.npz
python scripts/eval_controller.py --config configs/takecover.yaml --checkpoint checkpoints/takecover/vit_wm.pt --controller checkpoints/takecover/controller_z_plus_h.npz --dream
```

## Notes
- This repo intentionally avoids PPO / gradient RL. **C is trained only via CMA-ES**, like the paper.
- If you cannot create `CarRacing-v0` in your local install, try switching `data.env_id` to `CarRacing-v2` in `configs/carracing.yaml`.
- Newer Gymnasium versions may error on `CarRacing-v0` / `CarRacing-v2`. The default in `configs/carracing.yaml` uses `CarRacing-v3`.
