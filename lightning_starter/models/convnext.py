import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

from .nnmf.modules import NNMFConv2d, NNMFLayer, SECURE_TENSOR_MIN, ForwardNNMF, NNMFDense
from .nnmf.parameters import NonNegativeParameter

from .network_template import NetworkTemplate

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, kernel_size= 7, padding=3, expansion_ratio= 2, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expansion_ratio * dim) # pointwise/1x1 convs, implemented with linear layers
        # self.dwconv = nn.Conv2d(dim, expansion_ratio * dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False) # depthwise conv
        # self.norm = LayerNorm(expansion_ratio * dim, eps=1e-6)
        # self.pwconv1 = nn.Linear(expansion_ratio * dim, dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion_ratio * dim, dim)
        # self.pwconv2 = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class NNMFBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, nnmf_iterations=20, kernel_size= 7, padding=3, expansion_ratio= 2, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = NNMFConv2d(dim, dim, normalize_channels=False, n_iterations=nnmf_iterations, backward_method="all_grads", kernel_size=kernel_size, padding=padding, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expansion_ratio * dim) # pointwise/1x1 convs, implemented with linear layers
        # self.dwconv = NNMFConv2d(dim, expansion_ratio * dim, normalize_channels=False, n_iterations=nnmf_iterations, backward_method="all_grads", kernel_size=kernel_size, padding=padding, groups=dim) # depthwise conv
        # self.norm = LayerNorm(expansion_ratio * dim, eps=1e-6)
        # self.pwconv1 = nn.Linear(expansion_ratio * dim, dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion_ratio * dim, dim)
        # self.pwconv2 = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = torch.clamp(x, min=1e-6)
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class NNMFConvs(NNMFLayer):
    def __init__(
        self,
        dim,
        expansion_ratio,
        kernel_size=7,
        padding=3,
        n_iterations=20,
        backward_method="all_grads",
            
    ):
        super(NNMFConvs, self).__init__(        
        n_iterations=n_iterations,
        backward_method=backward_method,
        h_update_rate=1,
        keep_h=False,
        activate_secure_tensors=True,
        return_reconstruction=False,
        convergence_threshold=0,
        phantom_damping_factor=0.5,
        unrolling_steps=5,
        normalize_input=True,
        normalize_input_dim=(1,2,3),
        normalize_reconstruction=True,
        normalize_reconstruction_dim=(1,2,3)
        )
        self.dim = dim
        self.expansion_ratio = expansion_ratio
        self.kernel_size = kernel_size
        self.padding = padding
        self.dwconv_weight = NonNegativeParameter(torch.rand(dim, 1, kernel_size, kernel_size))
        self.pwconv_weight = NonNegativeParameter(torch.rand(expansion_ratio * dim, dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.dwconv_weight, a=0, b=1)
        self.dwconv_weight.data = F.normalize(self.dwconv_weight.data, p=1, dim=(1, 2, 3))
        nn.init.uniform_(self.pwconv_weight, a=0, b=1)
        self.pwconv_weight.data = F.normalize(self.pwconv_weight.data, p=1, dim=1)

    def normalize_weights(self):
        normalized_weight = F.normalize(self.dwconv_weight.data, p=1, dim=(1, 2, 3))
        self.dwconv_weight.data = F.normalize(
            normalized_weight.clamp(min=SECURE_TENSOR_MIN), p=1, dim=(1, 2, 3)
        )
        normalized_weight = F.normalize(self.pwconv_weight.data, p=1, dim=1)
        self.pwconv_weight.data = F.normalize(
            normalized_weight.clamp(min=SECURE_TENSOR_MIN), p=1, dim=1
        )

    def _process_h(self, h):
        h = self._secure_tensor(h)
        h = F.normalize(h, p=1, dim=-1)
        return h

    def _reset_h(self, x):
        h = torch.ones(x.size(0), x.size(2), x.size(3), self.pwconv_weight.size(0),  device=x.device)
        self.h = self._process_h(h)

    def _forward(self, x):
        x = F.conv2d(x, weight=self.dwconv_weight, padding=self.padding, groups=self.dim)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = F.normalize(x, p=1, dim=-1)       
        x = F.linear(x, self.pwconv_weight)
        return x
    
    def _reconstruct(self, h, weight=None):
        x = F.linear(h, self.pwconv_weight.t())
        x = F.normalize(x, p=1, dim=-1)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = F.conv_transpose2d(x, weight=self.dwconv_weight, padding=self.padding, groups=self.dim)
        return x

    def _check_forward(self, input):
        assert self.dwconv_weight.sum((1, 2, 3), keepdim=True).allclose(
            torch.ones_like(self.dwconv_weight), atol=1e-6
        ), self.dwconv_weight.sum((1, 2, 3))
        assert self.pwconv_weight.sum(1, keepdim=True).allclose(
            torch.ones_like(self.pwconv_weight), atol=1e-6
        ), self.pwconv_weight.sum(1)
        assert (self.dwconv_weight >= 0).all(), self.dwconv_weight.min()
        assert (self.pwconv_weight >= 0).all(), self.pwconv_weight.min()
        assert (input >= 0).all(), input.min()

class NNMFBlock3(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, nnmf_iterations=20, kernel_size= 7, padding=3, expansion_ratio= 2, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.nnmf_dense = NNMFDense(dim, expansion_ratio*dim, n_iterations= nnmf_iterations, backward_method="all_grads", return_reconstruction=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = torch.clamp(x, min=1e-6)
        _, x = self.nnmf_dense(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    
class NNMFBlock3p(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, nnmf_iterations=20, kernel_size= 7, padding=3, expansion_ratio= 2, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.nnmf_dense = NNMFDense(dim, expansion_ratio*dim, n_iterations= nnmf_iterations, backward_method="all_grads", return_reconstruction=False)
        self.pwconv2 = nn.Linear(expansion_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = torch.clamp(x, min=1e-6)
        x = self.nnmf_dense(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class NNMFBlock2(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, nnmf_iterations=20, kernel_size= 7, padding=3, expansion_ratio= 2, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.nnmf_conv = NNMFConvs(dim, expansion_ratio, kernel_size, padding, nnmf_iterations, "all_grads")
        self.pwconv2 = nn.Linear(expansion_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = torch.clamp(x, min=1e-6)
        input = x
        x = self.nnmf_conv(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

    def __repr__(self):
        return f"LayerNorm(normalized_shape={self.normalized_shape}, data_format={self.data_format})"

class ConvNeXt(NetworkTemplate):
    def __init__(
        self,
        in_channels,
        dims,
        n_classes,
    ):
        super(ConvNeXt, self).__init__(
                block_modules=[
                    Block(
                        dim=dims[0],
                        kernel_size=7,
                        padding=3,
                        expansion_ratio=2,
                        drop_path=0.,
                        layer_scale_init_value=1e-6
                    ),
                    Block(
                        dim=dims[1],
                        kernel_size=7,
                        padding=3,
                        expansion_ratio=2,
                        drop_path=0.,
                        layer_scale_init_value=1e-6
                    ),
                    Block(
                        dim=dims[2],
                        kernel_size=7,
                        padding=3,
                        expansion_ratio=2,
                        drop_path=0.,
                        layer_scale_init_value=1e-6
                    ),
                ],
                downsamplers=[
                    nn.Sequential(
                                nn.Conv2d(in_channels, dims[0], kernel_size=3, stride=1, padding=1),
                                # nn.Conv2d(in_channels, dims[0], kernel_size=2, stride=2, padding=0),
                                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
                            ),
                    nn.Sequential(
                                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                                nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2, padding=0)
                            ),
                    nn.Sequential(
                                LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
                                nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2, padding=0)
                            ),
                ],
                head=nn.Sequential(
                    nn.Linear(dims[-1], n_classes),
                    nn.Softmax(dim=-1)
                )
        )

# Register the model
@register_model
def convnext(**kwargs):
    return ConvNeXt(
        in_channels=3,
        dims=kwargs.get("dims", [32, 64, 96]),
        n_classes=10,
    )

class NNMFConvNeXt(NetworkTemplate):
    def __init__(
        self,
        in_channels,
        dims,
        n_classes,
        kernel_size=7,
        padding=3,
        block = NNMFBlock,
        nnmf_iterations=20,
    ):
        super(NNMFConvNeXt, self).__init__(
                block_modules=[
                    block(
                        dim=dims[0],
                        nnmf_iterations=nnmf_iterations,
                        kernel_size=kernel_size,
                        padding=padding,
                        expansion_ratio=2,
                        drop_path=0.,
                        layer_scale_init_value=1e-6
                    ),
                    block(
                        dim=dims[1],
                        nnmf_iterations=nnmf_iterations,
                        kernel_size=kernel_size,
                        padding=padding,
                        expansion_ratio=2,
                        drop_path=0.,
                        layer_scale_init_value=1e-6
                    ),
                    block(
                        dim=dims[2],
                        nnmf_iterations=nnmf_iterations,
                        kernel_size=kernel_size,
                        padding=padding,
                        expansion_ratio=2,
                        drop_path=0.,
                        layer_scale_init_value=1e-6
                    ),
                ],
                downsamplers=[
                    nn.Sequential(
                                nn.Conv2d(in_channels, dims[0], kernel_size=3, stride=1, padding=1),
                                # nn.Conv2d(in_channels, dims[0], kernel_size=2, stride=2, padding=0),
                                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
                            ),
                    nn.Sequential(
                                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                                nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2, padding=0)
                            ),
                    nn.Sequential(
                                LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
                                nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2, padding=0)
                            ),
                ],
                head=nn.Sequential(
                    nn.Linear(dims[-1], n_classes),
                    nn.Softmax(dim=-1)
                )
        )

# Register the model
@register_model
def nnmf_convnext(**kwargs):
    return NNMFConvNeXt(
        in_channels=3,
        dims=kwargs.get("dims", [32, 64, 96]),
        n_classes=10,
        nnmf_iterations=kwargs.get("nnmf_iterations", 20),
    )

@register_model
def nnmf_convnext2(**kwargs):
    return NNMFConvNeXt(
        in_channels=3,
        dims=kwargs.get("dims", [32, 64, 96]),
        kernel_size=3,
        padding=1,
        n_classes=10,
        nnmf_iterations=kwargs.get("nnmf_iterations", 20),
        block=NNMFBlock2
    )

@register_model
def nnmf_convnext3(**kwargs):
    return NNMFConvNeXt(
        in_channels=3,
        dims=kwargs.get("dims", [32, 64, 96]),
        n_classes=10,
        nnmf_iterations=kwargs.get("nnmf_iterations", 20),
        block=NNMFBlock3p
    )
