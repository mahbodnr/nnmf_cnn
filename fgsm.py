import torch
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import RandomSampler


from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from load_model_data import load_model_data

def get_mean_std(dataloader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std


models = [
    {"path": path, "name":name, "test_accuracies": []} 
    for path, name in [
        (r"model_checkpoints/cnnmf_vit_cifar10_ijwfs_20240621091141.ckpt", "CNNMF"),
        # (r"model_checkpoints/cnnmf_vit_l_cifar10_fqzus_20240620141357.ckpt", "CNNMF L"),
        # (r"model_checkpoints/nnmf_vit_cifar10_gnjdd_20240620110827.ckpt", "NNMF"),
        (r"model_checkpoints/vit_l_cifar10_glahb_20240620110641.ckpt", "ViT L"),
        (r"model_checkpoints/vit_cifar10_jwuvn_20240620105728.ckpt", "ViT"), 
        (r"model_checkpoints/cnn_cnn_top_cifar10_knyft_20240620104045.ckpt", "CNN"),
        ]
]

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

# restores the tensors to their original scale
def denorm(batch, mean, std, device="cuda"):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(batch.device)
    if isinstance(std, list):
        std = torch.tensor(std).to(batch.device)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def get_data_grads(model, device, test_loader):
    model.eval()
    data_grads = []
    # Get all data grads
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        data_grads.append(data_grad)

    return data_grads


def test( model, device, test_loader, epsilons, normalize=False, mean=None, std=None):
    assert not isinstance(test_loader, RandomSampler), "Set shuffle=False in DataLoader"
    data_grads = []
    # Get all data grads
    print("Getting data grads")
    data_grads = get_data_grads(model, device, test_loader)

    accuracies = []
    for epsilon in epsilons:
        # Accuracy counters
        correct = 0
        samples = 0
        for data_grad, (data, target) in zip(data_grads, test_loader):
            data, target = data.to(device), target.to(device)
            
            if normalize:
                assert mean is not None and std is not None
                data_denorm = denorm(data, mean, std, device=device)
                perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)
                perturbed_data_normalized = transforms.Normalize(mean, std)(perturbed_data)
                output = model(perturbed_data_normalized)
            else:
                perturbed_data = fgsm_attack(data, epsilon, data_grad)
                output = model(perturbed_data)

            # Check for success
            final_pred = output.max(1, keepdim=True)[1]
            correct += final_pred.eq(target.view_as(final_pred)).sum().item()
            samples += data.size(0)

        if epsilon == epsilons[-1]:
            for i in range(5):
                plt.subplot(2, 5, i+1)
                plt.imshow(data[i].cpu().permute(1, 2, 0).numpy())
                plt.subplot(2, 5, i+6)
                plt.imshow(perturbed_data[i].cpu().permute(1, 2, 0).numpy())
            plt.tight_layout()
            plt.suptitle(f"Epsilon: {epsilon}")
            plt.savefig(f"FGSM attack epsilon {epsilon}.png")
        # Calculate final accuracy for this epsilon
        final_acc = correct/samples
        accuracies.append(final_acc)
        print(f"Epsilon: {epsilon}\tTest Accuracy = {final_acc}")

    # Return the accuracy and an adversarial example
    return accuracies


epsilons = [0, .0001, 0.0005, 0.001, 0.005, 0.01]
epsilons = torch.cat([torch.tensor([0]), torch.logspace(-4, -2, steps= 10)])
# Run test for each epsilon
for model in models:
    print(f"\n\nTesting {model['name']}")
    net, train_dl, test_dl, args = load_model_data(
        model["path"], 
        args={
            "eval_batch_size": 512,
            }
    )

    net.eval()
    net = net.to("cuda")
    mean, std = get_mean_std(train_dl)
    mean, std = mean.to("cuda"), std.to("cuda")

    accuracies = test(net, "cuda", test_dl, epsilons, normalize=False, mean=mean, std=std)
    model["test_accuracies"] = accuracies

# plot accuracies
plt.figure(figsize=(10,5))
sns.set_theme()
for model in models:
    plt.plot(epsilons, model["test_accuracies"], label=model["name"], marker="o")
plt.legend()
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epsilon")
plt.xscale("log")

plt.savefig("FGSM attack.png")
