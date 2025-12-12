# imputer/diffusion/denoiser.py
import torch
import torch.nn as nn
from .schedule import sinusoidal_time_embedding

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
    def forward(self, x):
        return x + 0.5 * self.net(self.norm(x))

class LatentDenoiserGuided(nn.Module):
    def __init__(self, latent_dim, obs_dim, hidden=512, time_emb=128, res_blocks=3, device="cpu"):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_dim = time_emb
        self.device = device
        self.obs_proj = nn.Linear(obs_dim, latent_dim)
        self.in_dim = latent_dim * 3 + time_emb
        self.fc_in = nn.Linear(self.in_dim, hidden)
        self.resblocks = nn.ModuleList([ResBlock(hidden) for _ in range(res_blocks)])
        self.fc_out = nn.Sequential(nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, latent_dim))

    def forward(self, z_t, mu_cond, cond_obs, t):
        te = sinusoidal_time_embedding(t, self.time_dim, z_t.device)
        obs_lat = self.obs_proj(cond_obs)
        inp = torch.cat([z_t, mu_cond, obs_lat, te], dim=1)
        h = self.fc_in(inp)
        for block in self.resblocks:
            h = block(h)
        return self.fc_out(h)
