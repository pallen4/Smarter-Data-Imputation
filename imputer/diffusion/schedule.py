# imputer/diffusion/schedule.py
import math
import torch

def cosine_alpha_cumprod(T, s=0.008, device="cpu"):
    steps = T
    ts = torch.linspace(0, steps, steps + 1, dtype=torch.float64, device=device)
    alphas_cum = torch.cos(((ts / steps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cum = alphas_cum / alphas_cum[0]
    return alphas_cum[1:].to(dtype=torch.float32)

def sinusoidal_time_embedding(t, dim, device):
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / half)
    args = t.unsqueeze(1).float() * freq.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=device)], dim=1)
    return emb
