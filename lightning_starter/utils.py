import random
import string
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import DEBUG
if DEBUG:
    import debug.functional as F
    
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from timm.data import create_transform

from lightning_starter.autoaugment import CIFAR10Policy, SVHNPolicy
from lightning_starter.criterions import LabelSmoothingCrossEntropyLoss, MarginLoss, MSECrossEntropyLoss
from lightning_starter.lr_schedulers import GradualWarmupScheduler, StopScheduler


def get_layer_outputs(model, input):
    layer_outputs = {}

    def hook(module, input, output):
        layer_name = f"{module.__class__.__name__}_{module.parent_name}"
        layer_outputs[layer_name] = output.detach()

    # Add parent name attribute to each module
    for name, module in model.named_modules():
        module.parent_name = name
    # Register the hook to each layer in the model
    for module in model.modules():
        module.register_forward_hook(hook)
    # Pass the input through the model
    _ = model(input)
    # Remove the hooks and parent name attribute
    for module in model.modules():
        module._forward_hooks.clear()
        delattr(module, "parent_name")

    return layer_outputs


def get_criterion(args):
    if args.criterion == "ce":
        if args.label_smoothing:
            criterion = LabelSmoothingCrossEntropyLoss(
                args.num_classes, smoothing=args.smoothing
            )
        else:
            criterion = nn.CrossEntropyLoss()
    elif args.criterion == "margin":
        criterion = MarginLoss(m_pos=0.9, m_neg=0.1, lambda_=0.5)
    elif args.criterion == "msece":
        criterion = MSECrossEntropyLoss()
    else:
        raise ValueError(f"Criterion {args.criterion} not implemented.")

    return criterion


def get_transform(args):
    train_transform = create_transform(
        input_size = args.input_shape,
        is_training = True,
        no_aug = args.no_aug,
        train_crop_mode = None, # Applied later manually
        scale = None,
        ratio = None,
        hflip = args.hflip,
        vflip = args.vflip,
        color_jitter = 0.4,
        color_jitter_prob = args.color_jitter_prob,
        grayscale_prob = args.grayscale_prob,
        gaussian_blur_prob = args.gaussian_blur_prob,
        auto_augment = args.autoaugment,
        mean=args.mean,
        std=args.std,
    )
    test_transform = create_transform(
        input_size = args.input_shape,
        is_training = False,
        no_aug = True,
        train_crop_mode = None, # Applied later manually
        scale = None,
        ratio = None,
        hflip = 0.,
        vflip = 0.,
        color_jitter = 0.,
        color_jitter_prob = None,
        grayscale_prob = 0.,
        gaussian_blur_prob = 0.,
        auto_augment = None, # Applied later manually
        mean=args.mean,
        std=args.std,
    )

    if args.random_crop:
        assert args.random_crop_size is not None, "random_crop_size is required."
        train_transform.transforms.append(
            transforms.RandomCrop(size=args.random_crop_size, padding=args.random_crop_padding)
        )
        test_transform.transforms.append(
            transforms.CenterCrop(size=args.random_crop_size)
        )
        args.input_shape = (args.in_c, args.random_crop_size, args.random_crop_size)

    return train_transform, test_transform


def get_dataloader(args):
    root = "~/data"

    if args.dataset == "cifar10":
        args.in_c = 3
        args.num_classes = 10
        args.input_shape = (3, 32, 32)
        args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR10(
            root,
            train=True,
            transform=train_transform,
            download=args.download_data,
        )
        test_ds = torchvision.datasets.CIFAR10(
            root,
            train=False,
            transform=test_transform,
            download=args.download_data,
        )

    elif args.dataset == "cifar100":
        args.in_c = 3
        args.num_classes = 100
        args.input_shape = (3, 32, 32)
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR100(
            root,
            train=True,
            transform=train_transform,
            download=args.download_data,
        )
        test_ds = torchvision.datasets.CIFAR100(
            root,
            train=False,
            transform=test_transform,
            download=args.download_data,
        )

    elif args.dataset == "svhn":
        args.in_c = 3
        args.num_classes = 10
        args.input_shape = (3, 32, 32)
        args.mean, args.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.SVHN(
            root,
            split="train",
            transform=train_transform,
            download=args.download_data,
        )
        test_ds = torchvision.datasets.SVHN(
            root,
            split="test",
            transform=test_transform,
            download=args.download_data,
        )

    elif args.dataset == "mnist":
        args.in_c = 1
        args.num_classes = 10
        args.size = (1, 28, 28)
        args.mean, args.std = [0.1307], [0.3081]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.MNIST(
            root,
            train=True,
            transform=train_transform,
            download=args.download_data,
        )
        test_ds = torchvision.datasets.MNIST(
            root,
            train=False,
            transform=test_transform,
            download=args.download_data,
        )
    elif args.dataset == "fashionmnist":
        args.in_c = 1
        args.num_classes = 10
        args.input_shape = (1, 28, 28)
        args.mean, args.std = [0.2860], [0.3530]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.FashionMNIST(
            root,
            train=True,
            transform=train_transform,
            download=args.download_data,
        )
        test_ds = torchvision.datasets.FashionMNIST(
            root,
            train=False,
            transform=test_transform,
            download=args.download_data,
        )

    else:
        raise NotImplementedError(f"{args.dataset} is not implemented yet.")

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    return train_dl, test_dl

def get_scheduler(optimizer, args):
    if args.lr_scheduler == "reduce_on_plateau":
        # TODO: Add ReduceLROnPlateau parameters
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
            verbose=True,
            # threshold=args.lr_scheduler_threshold,
            # threshold_mode="rel",
            # cooldown=args.lr_scheduler_cooldown,
            min_lr=args.min_lr,
            eps=1e-10,
        )
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=(
                args.max_epochs
                if args.get("lr_scheduler_T_max", None) is None
                else args.lr_scheduler_T_max
            ),
            eta_min=args.min_lr,
        )
    elif args.lr_scheduler == "cosine_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.lr_scheduler_T_0,
            T_mult=args.lr_scheduler_T_mult,
            eta_min=args.min_lr,
        )
    elif args.lr_scheduler is None or args.lr_scheduler.lower() == "none":
        return None
    else:
        raise NotImplementedError(
            f"Unknown lr_scheduler: {args.lr_scheduler}"
        )
    if args.lr_warmup_epochs > 0:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1.0,
            total_epoch=args.lr_warmup_epochs,
            after_scheduler=scheduler,
        )
    if args.lr_scheduler_stop_epoch is not None:
        scheduler = StopScheduler(
            optimizer,
            base_scheduler=scheduler,
            stop_epoch=args.lr_scheduler_stop_epoch,
        )

    return scheduler

def get_experiment_name(args):
    experiment_name = f"{args.model_name}_{args.dataset}"
    experiment_name += f"_{random_string(5)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return experiment_name


random_string = lambda n: "".join(
    [random.choice(string.ascii_lowercase) for i in range(n)]
)
