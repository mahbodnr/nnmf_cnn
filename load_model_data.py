
import torch
import argparse
from david.get_the_data import get_the_data
from lightning_starter.network import Net
from lightning_starter.utils import get_dataloader


def load_model_data(model_path, args:dict = None):
    state = torch.load(model_path)
    hparams = state["hyper_parameters"]
    if args is not None:
        hparams.update(args)
    args = argparse.Namespace(**hparams)

    # torch set default dtype
    if args.default_dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif args.default_dtype == "float32":
        torch.set_default_dtype(torch.float32)
        torch.set_float32_matmul_precision(args.matmul_precision)

    # Load the model
    net = Net(args)
    net.load_state_dict(state["state_dict"])

    # Load the data
    if args.david_loader:
        train_dl, test_dl = get_the_data(
                dataset= args.dataset,
                batch_size_train = args.batch_size,
                batch_size_test = args.eval_batch_size,
                torch_device = "cuda" if args.gpus else "cpu",
                input_dim_x = 28,
                input_dim_y = 28,
                flip_p=0.5,
                jitter_brightness=0.5,
                jitter_contrast=0.1,
                jitter_saturation=0.1,
                jitter_hue=0.15,
            )
    else:
        train_dl, test_dl = get_dataloader(args)

    return net, train_dl, test_dl, args