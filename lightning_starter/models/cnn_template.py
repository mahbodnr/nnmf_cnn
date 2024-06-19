from .nnmf.modules import NNMFConv2d

import torch
import torch.nn as nn
from timm.models.registry import register_model

from .ann import ANN
from .cnn import calculate_last_layer_size

from david.Y import Y


class CNNTemplate(nn.Module):
    def __init__(
        self,
        blockModule,
        features,
        n_iterations=5,
        kernel_size=3,
        batchnorm=True,
        activation=nn.ReLU(),
        pooling=True,
    ):
        super(CNNTemplate, self).__init__()
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
            block = blockModule(features[feature_idx], features[feature_idx + 1], kernel_size=kernel_size[feature_idx])
            assert isinstance(block, nn.Module) or isinstance(block, nn.Sequential)
            if not isinstance(block, nn.Sequential):
                block = nn.Sequential(block)
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


class CNNNetworkTemplate(nn.Module):
    def __init__(
        self,
        blockModule,
        input_shape,
        cnn_features=[64, 128, 256],
        cnn_kernel_sizes=3,
        ann_layers=[1024, 256, 64, 10],
        batchnorm=False,
        pooling=True,
        activation=nn.ReLU(),
    ):
        super(CNNNetworkTemplate, self).__init__()
        self.conv = CNNTemplate(
            blockModule,
            [input_shape[0]] + cnn_features,
            kernel_size=cnn_kernel_sizes,
            batchnorm=batchnorm,
            pooling=pooling,
            activation=activation,
        )
        if ann_layers:
            self.ann = ANN(
                [
                    calculate_last_layer_size(
                        self.conv,
                        input_shape,
                    )
                ]
                + ann_layers,
                batchnorm=batchnorm,
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
