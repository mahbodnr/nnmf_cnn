import torch

from load_model_data import load_model_data
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

steps = 10000
device = "cuda"
model_path = r"model_checkpoints/nnmf_vit_cifar10_gnjdd_20240620110827.ckpt"

net, train_dl, test_dl, args = load_model_data(model_path)
net.eval()
net = net.to(device)
data, target = next(iter(test_dl))
data, target = data.to(device), target.to(device)
init_data = data.clone().detach()
data.requires_grad = True
# optimizer = torch.optim.SGD([data], lr=100)
optimizer = torch.optim.Adam([data], lr=1)


loss_history = []
for step in tqdm(range(steps)):
    output = net(data)
    loss = 0
    for module in net.modules():
        if hasattr(module, "reconstruction_mse"):
            loss += module.reconstruction_mse[-1]
    assert loss != 0, "No reconstruction loss found in the model."
    loss_history.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    data.data = torch.clamp(data, 0, 1)

    if step % 100 == 99:
        print(f"Step {step}, Loss: {loss.item()}")
        optimizer.param_groups[0]["lr"] *= 0.9


plt.plot(loss_history)
plt.xlabel("Step")
plt.ylabel("Reconstruction loss")
plt.savefig("reconstruction_loss.png")

# plot first 5 images
plt.figure()
for i in range(5):
    plt.subplot(2, 5, i+1)
    plt.imshow(init_data[i].cpu().permute(1, 2, 0).numpy())
    plt.subplot(2, 5, i+6)
    plt.imshow(data[i].cpu().detach().permute(1, 2, 0).numpy())

plt.tight_layout()
plt.suptitle("Reconstructed images")
plt.savefig("reconstructed_images.png")

