# imputer/main.py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from config import CFG
from data import load_numeric_dataset, make_missingness, scale_data
from vae import MaskedVAE, train_vae
from diffusion.schedule import cosine_alpha_cumprod
from diffusion.gating import MaskToLatentGate
from diffusion.denoiser import LatentDenoiserGuided
from diffusion.trainer import train_masked_latent_diffusion
from inference import run_masked_latent_sampling
from evaluate import evaluate_imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

def main():
    cfg = CFG
    device = torch.device(cfg["device"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # Load dataset
    df = load_numeric_dataset(cfg["csv_path"])
    feature_names = df.columns.tolist()
    X = df.values.astype(float)
    X_scaled, scaler = scale_data(X)

    # Create missingness mask and masked input
    X_input, mask_obs = make_missingness(X_scaled, cfg["missing_rate"], cfg["seed"])
    obs_vals = X_input * mask_obs  # observed values in place

    # Prepare DataLoader for VAE training
    X_input_t = torch.tensor(X_input, dtype=torch.float32)
    mask_obs_t = torch.tensor(mask_obs.astype(float), dtype=torch.float32)
    X_full_t = torch.tensor(X_scaled, dtype=torch.float32)
    dataset = TensorDataset(X_input_t, mask_obs_t, X_full_t)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=False)

    # Train VAE
    vae = MaskedVAE(X_scaled.shape[1], hidden=cfg["vae_hidden"], latent=cfg["vae_latent"])
    vae = train_vae(vae, loader, cfg, device)

    # Run VAE to get reconstructions and deterministic mu
    vae.to(device)
    with torch.no_grad():
        X_input_t_d = torch.tensor(X_input, dtype=torch.float32, device=device)
        mask_obs_t_d = torch.tensor(mask_obs.astype(float), dtype=torch.float32, device=device)
        recon_all, mu_all, _ = vae(X_input_t_d, mask_obs_t_d)
    vae_imputed_scaled = X_input_t_d * mask_obs_t_d + recon_all * (1.0 - mask_obs_t_d)
    vae_imputed_np = vae_imputed_scaled.cpu().numpy()

    # Set up diffusion schedule
    T = cfg["diff_T"]
    alpha_cum = cosine_alpha_cumprod(T, device=device).to(device)
    alphas = torch.zeros(T, device=device)
    betas = torch.zeros(T, device=device)
    for ti in range(T):
        prev = alpha_cum[ti-1] if ti > 0 else 1.0
        alphas[ti] = alpha_cum[ti] / prev
        betas[ti] = 1.0 - float(alphas[ti])
    betas = torch.clamp(betas, 1e-8, 0.999)

    # instantiate mask2latent and denoiser
    mask2latent = MaskToLatentGate(X_scaled.shape[1], cfg["vae_latent"]).to(device)
    denoiser = LatentDenoiserGuided(cfg["vae_latent"], X_scaled.shape[1],
                                    hidden=cfg["diff_hidden"], time_emb=cfg["diff_time_emb"],
                                    res_blocks=cfg["diff_res_blocks"], device=device).to(device)

    # Train diffusion
    denoiser, mask2latent = train_masked_latent_diffusion(
        denoiser, mask2latent, vae, X_input, mask_obs, obs_vals,
        alpha_cum, betas, alphas, cfg, device
    )

    # Inference / sampling
    final_imputed_scaled = run_masked_latent_sampling(
        denoiser, mask2latent, vae, X_input, mask_obs, obs_vals,
        alpha_cum, betas, alphas, cfg, device
    )

    # Inverse scaling
    final_imputed = scaler.inverse_transform(final_imputed_scaled)
    vae_only_imputed = scaler.inverse_transform(vae_imputed_np)
    original_full = scaler.inverse_transform(X_scaled)

    # Baselines
    X_missing_nan = X_scaled.copy()
    X_missing_nan[~mask_obs] = np.nan
    knn_imp = KNNImputer(n_neighbors=5).fit_transform(X_missing_nan)
    iter_imp = IterativeImputer(random_state=cfg["seed"], max_iter=10).fit_transform(X_missing_nan)
    knn_imp = scaler.inverse_transform(knn_imp)
    iter_imp = scaler.inverse_transform(iter_imp)

    # Evaluation
    methods = {
        "VAE + MaskedLatentDiffusion": final_imputed,
        "VAE-only": vae_only_imputed,
        "KNNImputer": knn_imp,
        "IterativeImputer": iter_imp
    }

    print("\nEvaluation (only at originally-missing entries):")
    results = {}
    for name, imp in methods.items():
        rmse, mae, r2, per_feat = evaluate_imputation(imp, original_full, mask_obs, feature_names)
        results[name] = (rmse, mae, r2, per_feat)
        print(f"\n{name}:")
        print(f"  RMSE = {rmse:.6f}")
        print(f"  MAE  = {mae:.6f}")
        print(f"  R2   = {r2:.6f}")
        print("  Per-feature RMSE:")
        for f, v in per_feat.items():
            print(f"    {f}: {v}")

    print("\nSummary (RMSE / MAE / R2):")
    for name, (rmse, mae, r2, _) in results.items():
        print(f"  {name}: RMSE={rmse:.6f}  MAE={mae:.6f}  R2={r2:.6f}")

    print("\nDone.")

if __name__ == "__main__":
    main()
