import os
from pprint import pprint

try :
    import comet_ml
except ImportError:
    pass

import torch
import pytorch_lightning as pl
import numpy as np

from lightning_starter.network import Net
from lightning_starter.utils import get_experiment_name, get_dataloader
from david.get_the_data import get_the_data


def train(args):
    # torch set default dtype
    if args.default_dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif args.default_dtype == "float32":
        torch.set_default_dtype(torch.float32)
        torch.set_float32_matmul_precision(args.matmul_precision)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.gpus = torch.cuda.device_count()
    args.num_workers = 4 * args.gpus if args.gpus else 8
    if not args.gpus:
        args.precision = 32
    if args.log_all:
        args.log_gradients = True
        args.log_weights = True
        args.log_layer_outputs = True


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

    args._sample_input_data = next(iter(train_dl))[0][0:10].to(
        "cuda" if args.gpus else "cpu"
    )
    print("Batch shape:", next(iter(train_dl))[0].shape)

    print("Arguments:")
    pprint({k: v for k, v in vars(args).items() if not k.startswith("_")})
    experiment_name = get_experiment_name(args)
    args.experiment_name = experiment_name
    # Set up logger
    print(f"Experiment: {experiment_name}")
    if args.use_comet:
        if args.use_wandb:
            print("[WARNING] Both Comet.ml and WandB are enabled. Using Comet.ml.")
        print("[INFO] Log with Comet.ml!")
        logger = pl.loggers.CometLogger(
            api_key=args._comet_api_key,
            save_dir="logs",
            project_name=args.project_name,
            experiment_name=experiment_name,
        )
        del args._comet_api_key  # remove the API key from args
    elif args.use_wandb:
        print("[INFO] Log with WandB!")
        import wandb

        wandb.login(key=args._wandb_api_key)
        logger = pl.loggers.WandbLogger(
            log_model="all",
            save_dir="logs",
            project=args.project_name,
            name=experiment_name,
            tags=args.tags.split(",") if args.tags else None,
        )
        del args._wandb_api_key  # remove the API key from args
    else:
        print("[INFO] Log with CSV")
        logger = pl.loggers.CSVLogger(save_dir="logs", name=experiment_name)

    # Set up model
    assert args.model_name is not None, "Model name is required."
    net = Net(args)

    # Set up trainer
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename=experiment_name + "-{epoch:03d}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    if args.profile:
        print("[INFO] Profiling!")
        profiler = pl.profilers.PyTorchProfiler(
            output_filename=f"logs/profile/{experiment_name}.txt",
            # use_cuda=True,
            profile_memory=True,
            export_to_chrome=True,
            use_cpu=False,
        )
    else:
        profiler = None
    trainer = pl.Trainer(
        precision=args.precision,
        fast_dev_run=args.dry_run,
        accelerator="auto",
        devices=args.gpus if args.gpus else "auto",
        benchmark=args.trainer_benchmark,
        logger=logger,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        enable_model_summary=False,  # Implemented seperately inside the Trainer
        profiler=profiler,
    )

    # Train model
    trainer.fit(model=net, train_dataloaders=train_dl, val_dataloaders=test_dl)

    # Save model
    save_models_dir = "model_checkpoints"
    save_model_path = os.path.join(save_models_dir, experiment_name + ".ckpt")
    trainer.save_checkpoint(save_model_path)
    print(f"Model saved to {save_model_path}")

    # Save model to logger
    # Comet.ml
    if args.use_comet:
        urls = logger.experiment.log_model(
            experiment_name, save_model_path, overwrite=True
        )
        print(f"Model saved to comet: {urls}")
    # WandB
    elif args.use_wandb:
        wandb.save(save_model_path)
        print(f"Model saved to WandB: {save_model_path}")
    else:
        print(f"Model saved to {save_model_path}")
