import torch
from torch import nn
from .network_template import NetworkTemplate
from .convnext import LayerNorm

from timm.models.registry import register_model



class Block(nn.Module):
    def __init__(self, dim, embed_dim, num_heads, mlp_ratio=4, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (N, H, W, C)
        x = x.view(N, H * W, C)
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]
        x = x + self.mlp(self.norm2(x))
        x = x.view(N, H, W, C)
        x = x.permute(0, 3, 1, 2) # (N, C, H, W)
        return x

class ViT_swin(NetworkTemplate):
    def __init__(
        self,
        in_channels,
        dims,
        n_classes,
    ):
        super(ViT, self).__init__(
            block_modules=[
                Block(
                    dim=dims[i],
                    embed_dim=dims[i],
                    num_heads=8,
                    mlp_ratio=2,
                ) for i in range(len(dims))
            ],
            downsamplers=[
                nn.Sequential(
                    # nn.Conv2d(in_channels, dims[0], kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4, padding=0),
                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
                ),
                *[nn.Sequential(
                    LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2, padding=0)
                ) for i in range(1, len(dims))],
            ],
            head=nn.Sequential(
                nn.Linear(dims[-1], n_classes),
                nn.Softmax(dim=-1)
            )
        )

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, dim, 1, 1))

    def forward(self, x):
        return x + self.pos_embedding

class ViT(NetworkTemplate):
    def __init__(
        self,
        in_channels,
        dims,
        n_classes,
    ):
        super(ViT, self).__init__(
            block_modules=[
                Block(
                    dim=dims[i],
                    embed_dim=dims[i],
                    num_heads=12,
                    mlp_ratio=2,
                ) for i in range(len(dims))
            ],
            downsamplers=[
                nn.Sequential(
                    nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4, padding=0),
                    PositionalEncoding(dims[0]),
                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                ),
                *[nn.Identity() for i in range(1, len(dims))],
            ],
            head=nn.Sequential(
                nn.Linear(dims[-1], n_classes),
                nn.Softmax(dim=-1)
            )
        )

@register_model
def vit(**kwargs):
    return ViT(
        in_channels=3,
        dims=kwargs.get("dims", [96, 96, 96]),
        n_classes=10,
    )

@register_model
def vit_l(**kwargs):
    return ViT(
        in_channels=3,
        dims=kwargs.get("dims", [192, 192, 192]),
        n_classes=10,
    )