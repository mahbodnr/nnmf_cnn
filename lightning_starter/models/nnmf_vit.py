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
                embed_dim,
                n_iterations=n_iterations,
                kernel_size=kernel_size,
                padding=padding,
                normalize_channels=normalize_channels,
                backward_method=backward_method,
                groups=num_heads,
                trainable_h=False,
                # normalize_input=True,
                # normalize_input_dim=1,
                # normalize_reconstruction=False,
                # normalize_reconstruction_dim=None,
            )
    
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
        x_norm = self.norm1(x)
        x = x + self.attn(torch.clamp(x_norm, min=1e-6))
        x = x.permute(0, 2, 3, 1)  # (N, H, W, C)
        x = x + self.mlp(self.norm2(x))
        x = x.permute(0, 3, 1, 2)  # (N, C, H, W)
        return x

class Block2(Block):
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
        super().__init__(
            dim,
            embed_dim,
            num_heads,
            mlp_ratio,
            drop,
            n_iterations,
            kernel_size,
            padding,
            normalize_channels,
            backward_method,
        )
        self.attn = nn.Sequential(
            self.attn,
            LayerNorm(embed_dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(embed_dim, dim, kernel_size=1),
        )

class Block3(Block):
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
        super().__init__(
            dim,
            embed_dim,
            num_heads,
            mlp_ratio,
            drop,
            n_iterations,
            kernel_size,
            padding,
            normalize_channels,
            backward_method,
        )
        self.attn = nn.Sequential(
            self.attn,
            LayerNorm(embed_dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, groups=num_heads),
            nn.GELU(),
            nn.Conv2d(embed_dim, dim, kernel_size=1),
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


class NNMFViT(NetworkTemplate):
    def __init__(
        self,
        in_channels,
        dims,
        embed_dims,
        n_classes,
        block_module=Block,
        heads=12,
    ):
        super(NNMFViT, self).__init__(
            block_modules=[
                block_module(
                    dim=dims[i],
                    embed_dim=embed_dims[i],
                    num_heads=heads,
                    mlp_ratio=2,
                )
                for i in range(len(dims))
            ],
            downsamplers=[
                nn.Sequential(
                    nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4, padding=0),
                    PositionalEncoding(dims[0], patch=7),
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
        embed_dims=kwargs.get("embed_dims", [96, 96, 96]),
        n_classes=10,
    )

@register_model
def cnnmf_vit(**kwargs):
    return NNMFViT(
        in_channels=3,
        dims=kwargs.get("dims", [96, 96, 96]),
        embed_dims=kwargs.get("embed_dims", [384, 384, 384]),
        # embed_dims=kwargs.get("embed_dims", [96, 96, 96]),
        n_classes=10,
        block_module=Block2,
    )

@register_model
def xnnmf_vit(**kwargs):
    return NNMFViT(
        in_channels=3,
        dims=kwargs.get("dims", [96, 96, 96]),
        embed_dims=kwargs.get("embed_dims", [384, 384, 384]),
        n_classes=10,
        block_module=Block3,
    )

@register_model
def nnmf_vit_l(**kwargs):
    return NNMFViT(
        in_channels=3,
        dims=kwargs.get("dims", [192, 192, 192]),
        embed_dims=kwargs.get("embed_dims", [192, 192, 192]),
        n_classes=10,
        heads= 12,
    )

@register_model
def cnnmf_vit_l(**kwargs):
    return NNMFViT(
        in_channels=3,
        dims=kwargs.get("dims", [192, 192, 192]),
        embed_dims=kwargs.get("embed_dims", [384, 384, 384]),
        n_classes=10,
        block_module=Block2,
        heads= 12,
    )

@register_model
def cnnmf_vit_s(**kwargs):
    return NNMFViT(
        in_channels=3,
        dims=kwargs.get("dims", [72, 72, 72]),
        embed_dims=kwargs.get("embed_dims", [192, 192, 192]),
        n_classes=10,
        block_module=Block2,
        heads= 12,
    )


@register_model
def cnnmf_vit_sh(**kwargs):
    return NNMFViT(
        in_channels=3,
        dims=kwargs.get("dims", [72, 72, 72]),
        embed_dims=kwargs.get("embed_dims", [192, 192, 192]),
        n_classes=10,
        block_module=Block2,
        heads= 1,
    )