import torch
from torch import nn
from .network_template import NetworkTemplate
from .convnext import LayerNorm

from einops import rearrange

from timm.models.registry import register_model


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = None, dropout = 0.):
        super().__init__()
        if dim_head is None:
            dim_head = dim // heads
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Block(nn.Module):
    def __init__(self, dim, embed_dim, num_heads, mlp_ratio=4, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads = num_heads, dim_head=embed_dim // num_heads, dropout = drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (N, H, W, C)
        x = x.view(N, H * W, C)
        x = x + self.attn(self.norm1(x))
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
    def __init__(self, dim, patch):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, dim, patch, patch))
        self.init_parameters()

    def forward(self, x):
        return x + self.pos_embedding

    def init_parameters(self):
        nn.init.trunc_normal_(self.pos_embedding, std=0.2)

class ViT(NetworkTemplate):
    def __init__(
        self,
        in_channels,
        dims,
        embed_dims,
        n_classes,
    ):
        super(ViT, self).__init__(
            block_modules=[
                Block(
                    dim=dims[i],
                    embed_dim=embed_dims[i],
                    num_heads=12,
                    mlp_ratio=2,
                ) for i in range(len(dims))
            ],
            downsamplers=[
                nn.Sequential(
                    nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4, padding=0),
                    PositionalEncoding(dims[0], patch= 7),
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
def vit_s(**kwargs):
    return ViT(
        in_channels=3,
        dims=kwargs.get("dims", [72, 72, 72]),
        embed_dims=kwargs.get("embed_dims", [96, 96, 96]),
        n_classes=10,
    )

@register_model
def vit(**kwargs):
    return ViT(
        in_channels=3,
        dims=kwargs.get("dims", [96, 96, 96]),
        embed_dims=kwargs.get("embed_dims", [96, 96, 96]),
        n_classes=10,
    )

@register_model
def vit_l(**kwargs):
    return ViT(
        in_channels=3,
        dims=kwargs.get("dims", [192, 192, 192]),
        embed_dims=kwargs.get("embed_dims", [192, 192, 192]),
        n_classes=10,
    )