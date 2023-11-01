import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # (B, C, H, W) -> (B, C, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (B, C, H//patch_size, W//patch_size) -> (B, C, H//patch_size * W//patch_size)
        x = x.transpose(1, 2)  # (B, C, H//patch_size * W//patch_size) -> (B, H//patch_size * W//patch_size, C)
        return x