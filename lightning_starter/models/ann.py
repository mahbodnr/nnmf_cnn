import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, layers, dropout=0.0, batchnorm=False, activation=nn.ReLU(), output_activation=None):
        super(ANN, self).__init__()
        self.output_activation = output_activation
        self.blocks = nn.ModuleList()
        for feature_idx in range(len(layers) - 1):
            self.blocks.append(nn.Linear(layers[feature_idx], layers[feature_idx + 1]))
            self.blocks.append(activation)
            if batchnorm:
                self.blocks.append(nn.BatchNorm1d(layers[feature_idx + 1]))
            if dropout:
                self.blocks.append(nn.Dropout(dropout))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x