# imputer/config.py
import torch

CFG = {
    "csv_path": "datasets/air_quality.csv",
    "seed": 42,
    "batch_size": 128,
    "vae_hidden": 512,
    "vae_latent": 64,
    "vae_epochs": 40,
    "vae_lr": 1e-3,
    "diff_hidden": 512,
    "diff_time_emb": 128,
    "diff_res_blocks": 3,
    "diff_T": 200,
    "diff_epochs": 120,
    "diff_lr": 1e-4,
    "missing_rate": 0.20,
    "train_mask_rate": 0.30,
    "clip_grad_norm": 1.0,
    "clamp_sampling": 2.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "print_every": 5,
    "consistency_weight": 0.06,
    "cond_drop_prob": 0.12,
    "num_ensemble": 4
}
