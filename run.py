import argparse
from lightning_starter.training import train


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-name",
    type=str,
)
# Logger
parser.add_argument("--comet", action="store_true", dest="use_comet")
parser.add_argument(
    "--comet-api-key", help="API Key for Comet.ml", dest="_comet_api_key", default=None
)
parser.add_argument("--wandb", action="store_true", dest="use_wandb")
parser.add_argument(
    "--wandb-api-key", help="API Key for WandB", dest="_wandb_api_key", default=None
)
parser.add_argument("--dry-run", action="store_true")
# data
parser.add_argument(
    "--dataset",
    default="cifar10",
    type=str,
    choices=["cifar10", "cifar100", "svhn", "mnist", "fashionmnist"],
)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", default=256, type=int)
parser.add_argument("--david-loader", action="store_true", help="Use David's data loader with specific augmentations.")
# Augmentation
parser.add_argument(
    "--autoaugment",
    default=None,
    type=str,
    choices=["original", "originalr", "v0", "v0r", "3a"],
    help="AutoAugment policy. One of 'original', 'originalr', 'v0', 'v0r', '3a'.",
)
parser.add_argument("--no-aug", action="store_true", help="Disable all augmentation.")
parser.add_argument("--hflip", default=0.5, type=float)
parser.add_argument("--vflip", default=0.0, type=float)
parser.add_argument("--color-jitter-prob", default=0.5, type=float)
parser.add_argument("--grayscale-prob", default=0.0, type=float)
parser.add_argument("--gaussian-blur-prob", default=0.0, type=float)
parser.add_argument("--random-crop", action="store_true", default=False)
parser.add_argument("--random-crop-size", default=None, type=int)
parser.add_argument("--random-crop-padding", default=0, type=int)
parser.add_argument("--rcpaste", action="store_true")
parser.add_argument("--cutmix", action="store_true")
parser.add_argument("--cutmix-beta", default=1.0, type=float)
parser.add_argument("--mixup", action="store_true")
parser.add_argument("--mixup-alpha", default=1.0, type=float)
# Optimizer
parser.add_argument("--optimizer", default="adam", type=str, dest="opt")
parser.add_argument("--weight-decay", default=0.0, type=float)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
# Learning rate scheduler
parser.add_argument(
    "--lr-scheduler",
    default=None,
    type=str,
)
parser.add_argument(
    "--lr-warmup-epochs",
    default=0,
    type=int,
    help="Number of warmup epochs for the learning rate. Set to 0 to disable warmup.",
)
parser.add_argument("--min-lr", default=1e-5, type=float)
parser.add_argument("--lr-scheduler-stop-epoch", default=None, type=int)
parser.add_argument("--lr-scheduler-monitor", default="val_loss", type=str)
parser.add_argument("--lr-scheduler-patience", default=10, type=int)
parser.add_argument("--lr-scheduler-factor", default=0.1, type=float)
# 
parser.add_argument(
    "--off-benchmark",
    action="store_false",
    dest="trainer_benchmark",
    help="The value (True or False) to set torch.backends.cudnn.benchmark to. The value for torch.backends.cudnn.benchmark set in the current session will be used (False if not manually set). If deterministic is set to True, this will default to False. You can read more about the interaction of torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic. Setting this flag to True can increase the speed of your system if your input sizes don’t change. However, if they do, then it might make your system slower. The CUDNN auto-tuner will try to find the best algorithm for the hardware when a new input size is encountered. This might also increase the memory usage.",
)
parser.add_argument("--max-epochs", default=100, type=int)
parser.add_argument(
    "--precision",
    default="32-true",
    type=str,
    choices=[
        "16",
        "32",
        "64",
        "bf16",
        "16-true",
        "16-mixed",
        "bf16-true",
        "bf16-mixed",
        "32-true",
        "64-true",
    ],
)
parser.add_argument(
    "--criterion", default="ce", type=str, choices=["ce", "margin", "msece"]
)
parser.add_argument("--label-smoothing", action="store_true")
parser.add_argument("--smoothing", default=0.1, type=float)


parser.add_argument(
    "--default-dtype",
    default="float32",
    type=str,
    choices=["float32", "float64"],
    help="Default dtype for the model.",
)
parser.add_argument(
    "--matmul-precision",
    default="medium",
    type=str,
    choices=["medium", "high", "highest"],
)
# Logging
parser.add_argument("--log-layer-outputs", action="store_true")
parser.add_argument(
    "--log-gradients", action="store_true", help="Save gradients during training."
)
parser.add_argument(
    "--no-log-weights",
    action="store_false",
    dest="log_weights",
    help="Disable logging weights during training.",
)
parser.add_argument("--log-all", action="store_true", help="Log all available metrics.")
parser.add_argument("--model-summary-depth", default=-1, type=int)
parser.add_argument(
    "--tags",
    default=None,
    type=str,
    help="Comma separated tags.",
)
# Misc
parser.add_argument("--profile", action="store_true")
parser.add_argument("--seed", default=9248, type=int)  # 92:48
parser.add_argument("--project-name", default="NNMF-CNN", type=str)
parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
parser.add_argument("--no-shuffle", action="store_false", dest="shuffle")
parser.add_argument("--allow-download", action="store_true", dest="download_data")
parser.add_argument("--debug", action="store_true", help="Enable debug mode.")

#### Models args:
# NNMF
parser.add_argument("--nnmf-iterations", default=20, type=int)
parser.add_argument("--nnmf-lr", default=None, type=float)
args = parser.parse_args()



if __name__ == "__main__":
    train(args)
