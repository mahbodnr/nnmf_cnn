import torch.nn as nn


class NetworkTemplate(nn.Module):
    def __init__(
        self,
        block_modules,
        downsamplers,
        head,
    ):
        super(NetworkTemplate, self).__init__()
        assert len(block_modules) == len(downsamplers)

        self.stages = nn.ModuleList()
        for downsampler, block_module in zip(downsamplers, block_modules):
            self.stages.append(downsampler)
            self.stages.append(block_module)

        self.head = head

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)

        x = x.mean([-1, -2])  # Global average pooling
        # TODO norm
        x = self.head(x)
        return x
