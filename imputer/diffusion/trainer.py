# imputer/diffusion/trainer.py
import torch
import numpy as np

def train_masked_latent_diffusion(denoiser, mask2latent, vae, X_input, mask_obs,
                                  obs_vals, alpha_cum, betas, alphas, cfg, device):
    denoiser.to(device); mask2latent.to(device)
    opt = torch.optim.Adam(list(denoiser.parameters()) + list(mask2latent.parameters()), lr=cfg["diff_lr"])
    n = X_input.shape[0]
    batch_size = cfg["batch_size"]
    indices = np.arange(n)
    n_batches = int(np.ceil(n / batch_size))
    CONSISTENCY_WEIGHT = cfg.get("consistency_weight", 0.06)
    COND_DROP_PROB = cfg.get("cond_drop_prob", 0.12)

    X_input_t = torch.tensor(X_input, dtype=torch.float32, device=device)
    mask_obs_t = torch.tensor(mask_obs.astype(float), dtype=torch.float32, device=device)
    obs_vals_t = torch.tensor(obs_vals, dtype=torch.float32, device=device)

    for epoch in range(cfg["diff_epochs"]):
        denoiser.train(); mask2latent.train()
        perm = np.random.permutation(indices)
        epoch_loss = 0.0
        for i in range(n_batches):
            batch_idx = perm[i*batch_size:(i+1)*batch_size]
            xb = X_input_t[batch_idx]
            maskb = mask_obs_t[batch_idx]
            obs_b = obs_vals_t[batch_idx]

            with torch.no_grad():
                mu_b, _ = vae.encode(xb, maskb)

            latent_gate = mask2latent(maskb)   # [B, L]
            latent_mask = latent_gate

            b = mu_b.shape[0]
            t = torch.randint(0, cfg["diff_T"], (b,), device=device, dtype=torch.long)
            noise = torch.randn_like(mu_b)

            a_hat = alpha_cum[t].unsqueeze(1)
            sqrt_a_hat = torch.sqrt(a_hat)
            sqrt_1m_a_hat = torch.sqrt(1.0 - a_hat)

            z_t = mu_b * (1.0 - latent_mask) + (sqrt_a_hat * mu_b + sqrt_1m_a_hat * noise) * latent_mask

            mu_cond = mu_b.clone()
            if COND_DROP_PROB > 0:
                drop = (torch.rand(b, device=device) < COND_DROP_PROB).float().unsqueeze(1)
                mu_cond = mu_cond * (1.0 - drop)

            keep_mask = ((torch.rand_like(maskb) > cfg["train_mask_rate"]).float() * maskb)
            cond_obs = obs_b * keep_mask

            pred_z0 = denoiser(z_t, mu_cond, cond_obs, t)

            sq = (pred_z0 - mu_b) ** 2
            weighted_sq = sq * latent_mask
            loss_z = weighted_sq.sum() / (latent_mask.sum() + 1e-8)

            cons_sq = ((pred_z0 - mu_b) ** 2) * latent_mask
            loss_cons = cons_sq.sum() / (latent_mask.sum() + 1e-8)

            loss = loss_z + CONSISTENCY_WEIGHT * loss_cons

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(denoiser.parameters()) + list(mask2latent.parameters()), cfg["clip_grad_norm"])
            opt.step()
            epoch_loss += loss.item()

        if (epoch + 1) % cfg["print_every"] == 0 or epoch == 0:
            print(f"[Diff] epoch {epoch+1}/{cfg['diff_epochs']} avg_loss={epoch_loss / n_batches:.6f}")

    return denoiser, mask2latent

