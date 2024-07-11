import torch
import pytorch_lightning as pl

from load_model_data import load_model_data
import matplotlib.pyplot as plt
import seaborn as sns


models = [
    {"path": path, "name":name, "test_accuracies": []} 
    for path, name in [
        (r"model_checkpoints/cnnmf_vit_cifar10_cccrb_20240620134827.ckpt", "CNNMF"),
        (r"model_checkpoints/cnnmf_vit_l_cifar10_fqzus_20240620141357.ckpt", "CNNMF L"),
        (r"model_checkpoints/nnmf_vit_cifar10_gnjdd_20240620110827.ckpt", "NNMF"),
        (r"model_checkpoints/vit_l_cifar10_glahb_20240620110641.ckpt", "ViT L"),
        (r"model_checkpoints/vit_cifar10_jwuvn_20240620105728.ckpt", "ViT"), 
        (r"model_checkpoints/cnn_cnn_top_cifar10_knyft_20240620104045.ckpt", "CNN"),
        ]
]

# noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
noise_levels = [0] + [i/25 for i in range(16)]

for model in models:
    net, train_dl, test_dl, args = load_model_data(model["path"])
    print(f"Testing {model['name']}")
    trainer = pl.Trainer()

    for noise_level in noise_levels:
        print(f"Testing with noise level {noise_level}")
        net.test_noise_level = noise_level
        res = trainer.test(net, test_dl, verbose=False)
        model["test_accuracies"].append(res[0]["test_acc"])


# plot the results
sns.set_theme()
plt.figure()
for model in models:
    plt.plot(noise_levels, model["test_accuracies"], label=model["name"], marker="o")
plt.xlabel("Noise level")
plt.ylabel("Test accuracy")
plt.legend()
plt.savefig("noise_robustness.png")