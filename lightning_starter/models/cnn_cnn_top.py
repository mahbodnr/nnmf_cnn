import torch
import torch.nn as nn
from timm.models.registry import register_model

from .cnn_template import CNNNetworkTemplate


def block(features_in, features_out, kernel_size, batchnorm=True):
    output= nn.Sequential(
        nn.Conv2d(
            features_in,
            features_out,
            kernel_size=kernel_size,
        ),
        nn.BatchNorm2d(
            features_out,
            track_running_stats=False,
        ) if batchnorm else nn.Identity(),
        nn.ReLU(),
        nn.Conv2d(
            features_out,
            features_out,
            kernel_size=1,
        ),
        nn.BatchNorm2d(
            features_out,
            track_running_stats=False,
        ) if batchnorm else nn.Identity(),
    )

    # Init the cnn top layers 1x1 conv2d layers
    for netp in output[-2].parameters():
        with torch.no_grad():
            if netp.ndim == 1:
                netp.data *= 0
            if netp.ndim == 4:
                assert netp.shape[-2] == 1
                assert netp.shape[-1] == 1
                netp[: netp.shape[0], : netp.shape[0], 0, 0] = torch.eye(
                    netp.shape[0], dtype=netp.dtype, device=netp.device
                )
                netp[netp.shape[0] :, :, 0, 0] = 0
                netp[:, netp.shape[0] :, 0, 0] = 0

    return output

class CnnCnnNetwork(CNNNetworkTemplate):
    def __init__(
        self,
        input_shape,
        cnn_features=[64, 128, 256],
        cnn_kernel_sizes=3,
        ann_layers=[1024, 256, 64, 10],
        pooling=True,
        batchnorm=False,
        activation=nn.ReLU(),
    ):
        super(CnnCnnNetwork, self).__init__(
            blockModule=block,
            input_shape=input_shape,
            cnn_features=cnn_features,
            cnn_kernel_sizes=cnn_kernel_sizes,
            ann_layers=ann_layers,
            pooling=pooling,
            activation=activation,
            batchnorm=batchnorm,
        )


@register_model
def cnn_cnn_top(**kwargs):
    return CnnCnnNetwork(
        input_shape=kwargs.get("input_shape", (3, 28, 28)),
        cnn_features=kwargs.get("cnn_features", [32, 64, 96, 10]),
        cnn_kernel_sizes=[5, 5, 4, 1],
        pooling=[True, True, False, False],
        activation=[nn.ReLU()] * 3 + [nn.Softmax(dim=1)],
        batchnorm=False,
        ann_layers=[],
    )
