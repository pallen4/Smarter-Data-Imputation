# imputer/diffusion/gating.py
import torch.nn as nn

class MaskToLatentGate(nn.Module):
    def __init__(self, feat_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
            nn.Sigmoid()
        )
    def forward(self, feat_mask):
        missing = 1.0 - feat_mask
        return self.net(missing)
