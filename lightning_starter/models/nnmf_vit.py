import torch
from torch import nn
import torch.nn.functional as F
from .network_template import NetworkTemplate
from .convnext import LayerNorm

from timm.models.registry import register_model
from .nnmf.modules import NNMFConv2d
from .convnext import LayerNorm


class Block(nn.Module):
    def __init__(
        self,
        dim,
        embed_dim,
        num_heads,
        mlp_ratio=4,
        drop=0.0,
        n_iterations=25,
        kernel_size=3,
        padding=1,
        normalize_channels=True,
        backward_method="all_grads",
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim, data_format="channels_first")
        self.attn = NNMFConv2d(
                dim,
                dim,
                n_iterations=n_iterations,
                kernel_size=kernel_size,
                padding=padding,
                normalize_channels=normalize_channels,
                backward_method=backward_method,
                groups=num_heads,
            )
    
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
        x_norm = self.norm1(x)
        x = x + self.attn(F.relu(x_norm))
        x = x.permute(0, 2, 3, 1)  # (N, H, W, C)
        x = x + self.mlp(self.norm2(x))
        x = x.permute(0, 3, 1, 2)  # (N, C, H, W)
        return x



class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, dim, 1, 1))

    def forward(self, x):
        return x + self.pos_embedding


class NNMFViT(NetworkTemplate):
    def __init__(
        self,
        in_channels,
        dims,
        n_classes,
        heads=12,
    ):
        super(NNMFViT, self).__init__(
            block_modules=[
                Block(
                    dim=dims[i],
                    embed_dim=dims[i],
                    num_heads=heads,
                    mlp_ratio=2,
                )
                for i in range(len(dims))
            ],
            downsamplers=[
                nn.Sequential(
                    nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4, padding=0),
                    PositionalEncoding(dims[0]),
                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                ),
                *[nn.Identity() for i in range(1, len(dims))],
            ],
            head=nn.Sequential(nn.Linear(dims[-1], n_classes), nn.Softmax(dim=-1)),
        )


@register_model
def nnmf_vit(**kwargs):
    return NNMFViT(
        in_channels=3,
        dims=kwargs.get("dims", [96, 96, 96]),
        n_classes=10,
    )

@register_model
def nnmf_vit_l(**kwargs):
    return NNMFViT(
        in_channels=3,
        dims=kwargs.get("dims", [256, 256, 256]),
        n_classes=10,
        heads= 16,
    )