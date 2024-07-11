from lightning_starter.models.nnmf.modules import NNMFDense, NNMFConv2d
import torch

layer = NNMFDense(
    in_features=10, 
    out_features=5, 
    n_iterations=5,
    trainable_h= True,
)

optimizer = torch.optim.Adam(layer.parameters(), lr=1)
print(layer)
print(layer.h)
out = layer(torch.rand(3, 10).abs())
loss = out.pow(2).sum()
loss.backward()
print(layer.h)
print(layer.h.grad)
optimizer.step()
print(layer.h)
out = layer(torch.rand(3, 10).abs())
print(layer.h)

layer = NNMFConv2d(
    in_channels=3, 
    out_channels=5, 
    kernel_size=3, 
    n_iterations=5,
    trainable_h= True,
)

optimizer = torch.optim.Adam(layer.parameters(), lr=1)
print(layer)
print(layer.h)
out = layer(torch.rand(1, 3, 10, 10).abs())
loss = out.pow(2).sum()
loss.backward()
print(layer.h)
print(layer.h.grad)
optimizer.step()
print(layer.h)
out = layer(torch.rand(1, 3, 10, 10).abs())
