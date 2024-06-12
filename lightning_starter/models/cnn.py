import torch
import torch.nn as nn
from timm.models.registry import register_model

from .ann import ANN


def calculate_last_layer_size(conv, input_shape):
    x = torch.randn(1, *input_shape)
    x = conv(x)
    return x.reshape(x.shape[0], -1).shape[1]

class CNN(nn.Module):
    def __init__(
        self,
        features,
        kernel_size=3,
        batchnorm=True,
        activation=nn.ReLU(),
        pooling=True,
    ):
        super(CNN, self).__init__()
        self.features = features
        self.blocks = nn.ModuleList()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * (len(features) - 1)
        assert len(kernel_size) == len(features) - 1
        for feature_idx in range(len(features) - 1):
            self.blocks.append(
                nn.Conv2d(
                    features[feature_idx],
                    features[feature_idx + 1],
                    kernel_size=kernel_size[feature_idx],
                )
            )
            self.blocks.append(activation)
            if batchnorm:
                self.blocks.append(nn.BatchNorm2d(features[feature_idx + 1]))
            if pooling:
                self.blocks.append(nn.MaxPool2d(2, 2))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class BaselineCNN(nn.Module):
    def __init__(
        self,
        input_shape,
        cnn_features=[64, 128, 256],
        cnn_kernel_sizes=3,
        ann_layers=[1024, 256, 64, 10],
    ):
        super(BaselineCNN, self).__init__()
        self.conv = CNN([input_shape[0]] + cnn_features, cnn_kernel_sizes)
        if ann_layers:
            self.ann = ANN(
                [
                    calculate_last_layer_size(
                        self.conv,
                        input_shape,
                    )
                ]
                + ann_layers
            )
        else:
            self.ann = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.ann(x)
        return x


@register_model
def baseline_newtork(**kwargs):
    return BaselineCNN(
        input_shape=kwargs.get("input_shape", (3, 32, 32)),
        cnn_features=kwargs.get("cnn_features", [32, 64, 96]),
        ann_layers=kwargs.get("ann_layers", [1024, 64, 10]),
    )


@register_model
def baseline_cnn(**kwargs):
    return BaselineCNN(
        input_shape=kwargs.get("input_shape", (3, 32, 32)),
        cnn_features=kwargs.get("cnn_features", [32, 64, 96]),
        ann_layers=kwargs.get("ann_layers", [10]),
    )
