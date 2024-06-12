import torch
import torchvision  # type: ignore

from torch.utils.data import DataLoader


def get_the_data(
    dataset: str,
    batch_size_train: int,
    batch_size_test: int,
    torch_device: torch.device,
    input_dim_x: int,
    input_dim_y: int,
    flip_p: float = 0.5,
    jitter_brightness: float = 0.5,
    jitter_contrast: float = 0.1,
    jitter_saturation: float = 0.1,
    jitter_hue: float = 0.15,
):
    dataset = dataset.upper()
    if dataset == "MNIST":
        raise (NotImplementedError("This dataset is not implemented."))

    elif dataset == "FashionMNIST":
        raise (NotImplementedError("This dataset is not implemented."))

    elif dataset == "CIFAR10":
        # Data augmentation filter
        test_processing_chain = torchvision.transforms.Compose(
            transforms=[
                torchvision.transforms.CenterCrop((input_dim_x, input_dim_y)),
                torchvision.transforms.ToTensor(),
            ],
        )

        train_processing_chain = torchvision.transforms.Compose(
            transforms=[
                torchvision.transforms.RandomCrop((input_dim_x, input_dim_y)),
                torchvision.transforms.RandomHorizontalFlip(p=flip_p),
                torchvision.transforms.ColorJitter(
                    brightness=jitter_brightness,
                    contrast=jitter_contrast,
                    saturation=jitter_saturation,
                    hue=jitter_hue,
                ),
                torchvision.transforms.ToTensor(),
            ],
        )

        tv_dataset_train = torchvision.datasets.CIFAR10(
            root="~/data", train=True, download=True, transform=train_processing_chain
        )
        tv_dataset_test = torchvision.datasets.CIFAR10(
            root="~/data", train=False, download=True, transform=test_processing_chain
        )
    else:
        raise NotImplementedError("This dataset is not implemented.")

    if dataset == "MNIST" or dataset == "FashionMNIST":
        raise (NotImplementedError("This dataset is not implemented."))
    else:
        train_dataloader = DataLoader(
            tv_dataset_train,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=2,
        )

        test_dataloader = DataLoader(
            tv_dataset_test,
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=2,
        )

    return train_dataloader, test_dataloader
