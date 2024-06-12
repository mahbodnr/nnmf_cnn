from .nnmf.modules import NNMFConv2d

import torch
import torch.nn as nn
from timm.models.registry import register_model

from .ann import ANN
from .cnn import calculate_last_layer_size


class NnmfCnn(nn.Module):
    def __init__(
        self,
        features,
        n_iterations=5,
        kernel_size=3,
        batchnorm=True,
        activation=nn.ReLU(),
        pooling=True,
    ):
        super(NnmfCnn, self).__init__()
        self.features = features
        self.blocks = nn.ModuleList()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * (len(features) - 1)
        if isinstance(pooling, bool):
            pooling = [pooling] * (len(features) - 1)
        if isinstance(activation, nn.Module):
            activation = [activation] * (len(features) - 1)

        assert len(kernel_size) == len(features) - 1
        for feature_idx in range(len(features) - 1):
            block = nn.Sequential(
                nn.ReLU(),
                NNMFConv2d(
                    features[feature_idx],
                    features[feature_idx + 1],
                    n_iterations=n_iterations,
                    kernel_size=kernel_size[feature_idx],
                    # normalize_channels=True,
                    backward_method="all_grads",
                ),
                nn.Conv2d(
                    features[feature_idx + 1],
                    features[feature_idx + 1],
                    kernel_size=1,
                ),
            )
            if activation[feature_idx] is not None:
                block.append(activation[feature_idx])
            if batchnorm:
                block.append(nn.BatchNorm2d(features[feature_idx + 1]))
            if pooling[feature_idx]:
                block.append(nn.AvgPool2d(2, 2))

            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class NnmfCnnNetwork(nn.Module):
    def __init__(
        self,
        input_shape,
        nnmf_iterations=5,
        cnn_features=[64, 128, 256],
        cnn_kernel_sizes=3,
        ann_layers=[1024, 256, 64, 10],
        pooling=True,
        activation=nn.ReLU(),
    ):
        super(NnmfCnnNetwork, self).__init__()
        self.conv = NnmfCnn(
            [input_shape[0]] + cnn_features,
            kernel_size=cnn_kernel_sizes,
            n_iterations=nnmf_iterations,
            batchnorm=False,
            pooling=pooling,
            activation=activation,
        )
        if ann_layers:
            self.ann = ANN(
                [
                    calculate_last_layer_size(
                        input_shape,
                        cnn_features,
                    )
                ]
                + ann_layers,
                batchnorm=False,
                # output_activation=nn.Softmax(dim=-1),
            )
        else:
            self.ann = None

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        if self.ann:
            x = self.ann(x)
        return x


@register_model
def nnmf_cnn_top(**kwargs):
    return NnmfCnnNetwork(
        input_shape=kwargs.get("input_shape", (3, 32, 32)),
        cnn_features=kwargs.get("cnn_features", [32, 64, 96, 10]),
        cnn_kernel_sizes=[5, 5, 4, 1],
        pooling=[True, True, False, False],
        activation=[nn.ReLU()] * 3 + [nn.Softmax(dim=1)],
        ann_layers=[],
        nnmf_iterations=kwargs.get("nnmf_iterations", 5),
    )

@register_model
def nnmf_cnn_top_d(**kwargs):
    return NnmfCnnNetwork(
        input_shape=kwargs.get("input_shape", (3, 32, 32)),
        cnn_features=kwargs.get("cnn_features", [64, 128, 192, 10]),
        cnn_kernel_sizes=[5, 5, 4, 1],
        pooling=[True, True, False, False],
        activation=[nn.ReLU()] * 3 + [nn.Softmax(dim=1)],
        ann_layers=[],
        nnmf_iterations=kwargs.get("nnmf_iterations", 5),
    )

@register_model
def nnmf_cnn_top_m(**kwargs):
    return NnmfCnnNetwork(
        input_shape=kwargs.get("input_shape", (3, 32, 32)),
        cnn_features=kwargs.get("cnn_features", [64, 92, 128, 10]),
        cnn_kernel_sizes=[5, 5, 4, 1],
        pooling=[True, True, False, False],
        activation=[nn.ReLU()] * 3 + [nn.Softmax(dim=1)],
        ann_layers=[],
        nnmf_iterations=kwargs.get("nnmf_iterations", 5),
    )
