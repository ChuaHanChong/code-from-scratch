# https://www.youtube.com/watch?v=ovB0ddFtzzA

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


class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
        
    def forward(self, x):
        n_samples, n_tokens, dim = x.shape
        
        if dim != self.dim:
            raise ValueError
        
        qkv = self.qkv(x)  # (B, n_patches + 1, dim) -> (B, n_patches + 1, dim * 3)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)  # (B, n_patches + 1, dim * 3) -> (B, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (B, n_patches + 1, 3, n_heads, head_dim) -> (3, B, n_heads, n_patches + 1, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]  # (3, B, n_heads, n_patches + 1, head_dim) -> (B, n_heads, n_patches + 1, head_dim)
        k_t = k.transpose(-2, -1)  # (B, n_heads, n_patches + 1, head_dim) -> (B, n_heads, head_dim, n_patches + 1)
        dp = (q @ k_t) * self.scale  # (B, n_heads, n_patches + 1, head_dim) @ (B, n_heads, head_dim, n_patches + 1) -> (B, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)  # (B, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)
        
        weighted_avg = attn @ v  # (B, n_heads, n_patches + 1, n_patches + 1) @ (B, n_heads, n_patches + 1, head_dim) -> (B, n_heads, n_patches + 1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)  # (B, n_heads, n_patches + 1, head_dim) -> (B, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (B, n_patches + 1, n_heads, head_dim) -> (B, n_patches + 1, n_heads * head_dim)
        
        x = self.proj(weighted_avg)  # (B, n_patches + 1, n_heads * head_dim) -> (B, n_patches + 1, dim)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)
        
    def forward(self, x):
        x = self.fc1(x)  # (B, n_patches + 1, dim) -> (B, n_patches + 1, hidden_features)
        x = self.act(x)  # (B, n_patches + 1, hidden_features)
        x = self.drop(x)  # (B, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (B, n_patches + 1, hidden_features) -> (B, n_patches + 1, out_features)
        x = self.drop(x)
        
        return x


class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, 
            n_heads=n_heads, 
            qkv_bias=qkv_bias, 
            attn_p=attn_p, 
            proj_p=p,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, 
            hidden_features=hidden_features, 
            out_features=dim, 
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # (B, n_patches + 1, dim) -> (B, n_patches + 1, dim)
        x = x + self.mlp(self.norm2(x))  # (B, n_patches + 1, dim) -> (B, n_patches + 1, dim) 
        
        return x


class VisionTransformer(nn.Module):
    def __ini__(
        self, 
        img_size=384,
        patch_size=16,
        in_chans=3,
        n_classes=1000,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        p=0.,
        attn_p=0.,
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # (1, 1, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))  # (1, 1 + n_patches, embed_dim)
        self.pos_drop = nn.Dropout(p=p)
        
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim, 
                    n_heads=n_heads, 
                    mlp_ratio=mlp_ratio, 
                    qkv_bias=qkv_bias, 
                    p=p, 
                    attn_p=attn_p,
                ) for _ in range(depth)
            ]
        )
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)
        
    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        
        cls_token = self.cls_token.expand(n_samples, -1, -1)  # (1, 1, embed_dim) -> (B, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (B, 1, embed_dim) + (B, n_patches, embed_dim) -> (B, 1 + n_patches, embed_dim)
        x = x + self.pos_embed  # (B, 1 + n_patches, embed_dim) + (1, 1 + n_patches, embed_dim) -> (B, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)  # (B, 1 + n_patches, embed_dim) -> (B, 1 + n_patches, embed_dim)
        
        x = self.norm(x)  # (B, 1 + n_patches, embed_dim) -> (B, 1 + n_patches, embed_dim)
        
        cls_token_final = x[:, 0]  # (B, 1 + n_patches, embed_dim) -> (B, embed_dim)
        x = self.head(cls_token_final)
        
        return x


def get_n_params(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_params


if __name__ == '__main__':
    model_name = "vit_base_patch16_384"
    model_official = timm.create_model(model_name, pretrained=True)
    model_official.eval()
    print(type(model_official))
    
    custom_config = {
        "img_size": 384,
        "in_chans": 3,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "n_heads": 12,
        "qkv_bias": True,
        "mlp_ratio": 4,
    }
    
    model_custom = VisionTransformer(**custom_config)
    model_custom.eval()
    
    def assert_tensors_equal(t1, t2):
        a1, a2 = t1.detach().numpy(), t2.detach().numpy()
        np.testing.assert_allclose(a1, a2)
    
    for (n_o, p_o), (n_c, p_c) in zip(model_official.named_parameters(), model_custom.named_parameters()):
        assert p_o.numel() == p_c.numel()
        print(f"{n_o} | {n_c}")
        
        p_c.data[:] = p_o.data
        
        assert_tensors_equal(p_c.data, p_o.data)
    
    inp = torch.rand(1, 3, 384, 384)
    res_c = model_custom(inp)
    res_o = model_official(inp)
    
    assert get_n_params(model_custom) == get_n_params(model_official)
    assert_tensors_equal(res_c, res_o)
        
        
        
        
    
    