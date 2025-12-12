# imputer/inference.py
import torch
import numpy as np

def run_masked_latent_sampling(denoiser, mask2latent, vae, X_input, mask_obs, obs_vals,
                               alpha_cum, betas, alphas, cfg, device):
    denoiser.to(device); mask2latent.to(device); vae.to(device)
    n = X_input.shape[0]
    batch_size = cfg["batch_size"]
    batch_count = int(np.ceil(n / batch_size))
    final_imputed_scaled = np.zeros_like(X_input)
    NUM_ENSEMBLE = cfg.get("num_ensemble", 4)
    T_refine = cfg["diff_T"]

    X_input_t = torch.tensor(X_input, dtype=torch.float32, device=device)
    mask_obs_t = torch.tensor(mask_obs.astype(float), dtype=torch.float32, device=device)
    obs_vals_t = torch.tensor(obs_vals, dtype=torch.float32, device=device)

    with torch.no_grad():
        for bi in range(batch_count):
            idx = np.arange(bi*batch_size, min((bi+1)*batch_size, n))
            xb = X_input_t[idx]
            maskb = mask_obs_t[idx]
            obs_b = obs_vals_t[idx]

            mu_b, _ = vae.encode(xb, maskb)
            latent_gate = mask2latent(maskb)
            latent_mask = latent_gate
            cond_obs = obs_b

            ensemble_outputs = []
            for ens in range(NUM_ENSEMBLE):
                tT = (T_refine - 1) * torch.ones((mu_b.shape[0],), device=device, dtype=torch.long)
                noiseT = torch.randn_like(mu_b)
                a_hat_T = alpha_cum[tT].unsqueeze(1)
                z_cur = mu_b * (1.0 - latent_mask) + (torch.sqrt(a_hat_T) * mu_b + torch.sqrt(1.0 - a_hat_T) * noiseT) * latent_mask

                for tstep in reversed(range(T_refine)):
                    t_batch = torch.full((z_cur.shape[0],), tstep, device=device, dtype=torch.long)
                    pred_z0 = denoiser(z_cur, mu_b, cond_obs, t_batch)
                    a_hat = float(alpha_cum[tstep].item())
                    sqrt_a_hat = np.sqrt(a_hat)
                    sqrt_1m_a_hat = np.sqrt(max(1.0 - a_hat, 1e-12))
                    pred_noise = (z_cur - torch.tensor(sqrt_a_hat, device=device) * pred_z0) / (torch.tensor(sqrt_1m_a_hat, device=device) + 1e-12)

                    beta_t = float(betas[tstep].item())
                    alpha_t = float(alphas[tstep].item())
                    alpha_hat_t = float(alpha_cum[tstep].item())

                    coef1 = 1.0 / np.sqrt(alpha_t)
                    coef2 = beta_t / np.sqrt(max(1e-12, 1.0 - alpha_hat_t))
                    mean = coef1 * (z_cur - coef2 * pred_noise)

                    if tstep > 0:
                        z_noise = torch.randn_like(z_cur)
                        sigma = np.sqrt(beta_t)
                        z_next = mean + sigma * z_noise
                    else:
                        z_next = mean

                    z_next = z_next * latent_mask + mu_b * (1.0 - latent_mask)
                    z_cur = z_next

                dec = vae.decode(z_cur, maskb)
                combined = obs_b + dec * (1.0 - maskb)
                ensemble_outputs.append(combined.cpu().numpy())

            avg_combined = np.mean(np.stack(ensemble_outputs, axis=0), axis=0)
            final_imputed_scaled[idx, :] = avg_combined

    return final_imputed_scaled
