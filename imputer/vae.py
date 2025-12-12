# imputer/vae.py
import torch
import torch.nn as nn

class MaskedVAE(nn.Module):
    def __init__(self, input_dim, hidden=512, latent=64):
        super().__init__()
        self.input_dim = input_dim
        self.enc = nn.Sequential(
            nn.Linear(input_dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU()
        )
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent + input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, input_dim)
        )

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x_in, mask):
        enc_in = torch.cat([x_in, mask], dim=1)
        h = self.enc(enc_in)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def decode(self, z, mask):
        dec_in = torch.cat([z, mask], dim=1)
        recon = self.dec(dec_in)
        return recon

    def forward(self, x_in, mask):
        mu, logvar = self.encode(x_in, mask)
        z = self.reparam(mu, logvar)
        recon = self.decode(z, mask)
        return recon, mu, logvar

def vae_loss(recon, x_true, mu, logvar, observed_mask, obs_weight=0.1):
    missing_mask = 1.0 - observed_mask
    mse_elem = (recon - x_true) ** 2
    loss_missing = (mse_elem * missing_mask).sum() / (missing_mask.sum() + 1e-8)
    loss_obs = (mse_elem * observed_mask).sum() / (observed_mask.sum() + 1e-8)
    recon_loss = loss_missing + obs_weight * loss_obs
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 1e-3 * kl, recon_loss.detach().item(), kl.detach().item()

def train_vae(model, loader, cfg, device):
    opt = torch.optim.Adam(model.parameters(), lr=cfg["vae_lr"])
    model.to(device)
    for epoch in range(cfg["vae_epochs"]):
        model.train()
        loss_epoch = 0.0
        for xb, maskb, xt in loader:
            xb = xb.to(device); maskb = maskb.to(device); xt = xt.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(xb, maskb)
            loss, _, _ = vae_loss(recon, xt, mu, logvar, maskb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad_norm"])
            opt.step()
            loss_epoch += loss.item()
        if (epoch + 1) % cfg["print_every"] == 0 or epoch == 0:
            print(f"[VAE] epoch {epoch+1}/{cfg['vae_epochs']} avg_loss={loss_epoch/len(loader):.6f}")
    return model
