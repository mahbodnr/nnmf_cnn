import time
import pytorch_lightning as pl
import torch
import wandb

from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2
from timm.models import create_model

from lightning_starter.models import *
from lightning_starter.utils import get_criterion, get_scheduler, get_layer_outputs
from lightning_starter.augmentation import CutMix, MixUp

from .models.nnmf.parameters import NonNegativeParameter

class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        self.hparams.update(vars(hparams))
        self.save_hyperparameters(
            ignore=[key for key in self.hparams.keys() if key[0] == "_"]
        )
        self.model = create_model(
            # model_name= hparams.model_name,
            # pretrained= False,
            # pretrained_cfg = None,
            # pretrained_cfg_overlay = None,
            # checkpoint_path = '',
            # scriptable = None,
            # exportable = None,
            # no_jit = None,
            **hparams.__dict__,
        )
        self.criterion = get_criterion(hparams)
        # CutMix and MixUp
        if hparams.cutmix:
            self.cutmix = CutMix(hparams.size, beta=hparams.cutmix_beta)
        if hparams.mixup:
            self.mixup = MixUp(alpha=hparams.mixup_alpha)

        self.test_noise_level = None

    def log_time(func):
        """
        A decorator to measure the time of a function and log it.
        """

        def wrapper(self, *args, **kwargs):
            start = time.time()
            result = func(self, *args, **kwargs)
            end = time.time()
            self.log(f"{func.__name__}_time", end - start)
            return result

        return wrapper

    @log_time
    def forward(self, x):
        return self.model(x)

    def calculate_loss(self, out, label):
        return self.criterion(out, label)

    def configure_optimizers(self):
        """
        Example:
            param_group_fn = lambda model: [
                {"params": group_1, "lr":group_1_lr},
                {"params": group_2},
            ]
        """
        param_group_fn = lambda model: [
            {
                "params": [
                    p
                    for p in model.parameters()
                    if not isinstance(p, NonNegativeParameter)
                ],
                "lr": self.hparams.lr,
            },
            {
                "params": [
                    p for p in model.parameters() if isinstance(p, NonNegativeParameter)
                ],
                "lr": (
                    self.hparams.lr
                    if self.hparams.nnmf_lr is None
                    else self.hparams.nnmf_lr
                ),
            },
        ]
        self.optimizer = create_optimizer_v2(
            self, param_group_fn=param_group_fn, **optimizer_kwargs(cfg=self.hparams)
        )

        self.scheduler = get_scheduler(self.optimizer, self.hparams)

        if self.scheduler is None:
            return self.optimizer
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "monitor": self.hparams.lr_scheduler_monitor,
        }

    def on_fit_start(self):
        self.print_model_summary()

    def on_train_start(self):
        # wandb watch model
        if isinstance(self.logger, pl.loggers.WandbLogger):
            log = {
                (True, False): "gradients",
                (True, True): "all",
                (False, True): "parameters",
                (False, False): None,
            }[(self.hparams.log_gradients, self.hparams.log_weights)]
            print(f"[INFO] WandB watch log: {log}")
            self.logger.watch(
                self.model,
                log=log,
            )

        # Number of parameters:
        self.log(
            "trainable_params",
            float(sum(p.numel() for p in self.model.parameters() if p.requires_grad)),
        )
        self.log("total_params", float(sum(p.numel() for p in self.model.parameters())))

        # Log tags
        if hasattr(self.logger.experiment, "add_tags"):
            if self.hparams.tags:
                tags = self.hparams.tags.split(",")
                self.logger.experiment.add_tags(
                    [tag.strip() for tag in tags if tag.strip()]
                )

    def _step(self, img, label):
        if self.hparams.cutmix or self.hparams.mixup:
            img, label, rand_label, lambda_ = self.cutmix_mixup(img, label)
            out = self(img)
            loss = (self.calculate_loss(out, label) * lambda_) + (
                self.calculate_loss(out, rand_label) * (1.0 - lambda_)
            )
        else:
            out = self(img)
            loss = self.calculate_loss(out, label)

        return out, loss

    def training_step(self, batch, batch_idx):
        img, label = batch
        out, loss = self._step(img, label)

        # TODO: log input images of the first batch (only once)

        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def on_train_epoch_end(self):
        # log learning rate
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.log(f"lr_{i}", param_group["lr"], on_epoch=True)
        # check if there is any nan value in model parameters
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                raise ValueError(f"[ERROR] {name} has nan value. Training stopped.")
        # log output histogram of each layer
        if self.hparams.log_layer_outputs:
            self.log_layer_outputs()

    def optimizer_step(self, *args, **kwargs):
        """
        Add weight normalization, etc here.
        """
        super().optimizer_step(*args, **kwargs)

    def on_train_batch_end(self, out, batch, batch_idx):
        if batch_idx == self.trainer.num_training_batches - 1:  # only on last batch
            pass

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.calculate_loss(out, label)
        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        img, label = batch
        if self.test_noise_level is not None and self.test_noise_level > 0:
            img = img + self.test_noise_level * torch.randn_like(img) + img.mean()
        out = self(img)
        loss = self.calculate_loss(out, label)
        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def log_free_mem(self):
        free_memory, total_memory = torch.cuda.mem_get_info()
        self.log("free_memory", free_memory)

    def on_train_end(self):
        if hasattr(self.logger.experiment, "end"):
            self.logger.experiment.end()

    def print_model_summary(self):
        print(self.model)
        summary = pl.utilities.model_summary.ModelSummary(
            self, max_depth=self.hparams.model_summary_depth
        )
        if hasattr(self.logger.experiment, "log_asset_data"):
            self.logger.experiment.log_asset_data(
                str(summary), file_name="model_summary.txt"
            )
        print(summary)
        #### use torchsummary instead:
        # from torchsummary import summary
        # summary(self.model, self.hparams._sample_input_data.shape[1:], device="cuda")

    def cutmix_mixup(self, img, label):
        if self.hparams.cutmix:
            img, label, rand_label, lambda_ = self.cutmix((img, label))
        elif self.hparams.mixup:
            if torch.rand(1).item() <= 0.8:
                img, label, rand_label, lambda_ = self.mixup((img, label))
            else:
                img, label, rand_label, lambda_ = (
                    img,
                    label,
                    torch.zeros_like(label),
                    1.0,
                )
        return img, label, rand_label, lambda_

    def log_layer_outputs(self):
        # log the output of each layer
        layer_outputs = get_layer_outputs(self.model, self.hparams._sample_input_data)
        for name, output in layer_outputs.items():
            self.log_histogram(output, name + ".output", self.global_step)

    def log_histogram(self, tensor, name, step):
        """
        Log a histogram of a tensor.
        """
        # comet logger
        if isinstance(self.logger, pl.loggers.comet.CometLogger):
            try:
                self.logger.experiment.log_histogram_3d(
                    tensor.detach().cpu(),
                    name=name,
                    step=step,
                )
            except IndexError:
                # Values closer than 1e-20 to zerro will lead to index error
                positive_output = tensor[tensor > 0]
                pos_min = (
                    positive_output.min().item()
                    if positive_output.numel() > 0
                    else float("inf")
                )
                negative_output = tensor[tensor < 0]
                neg_min = (
                    abs(negative_output.max().item())
                    if negative_output.numel() > 0
                    else float("inf")
                )
                self.logger.experiment.log_histogram_3d(
                    tensor.detach().cpu(),
                    name=name,
                    step=step,
                    start=min(pos_min, neg_min),
                )
            except Exception as e:
                raise e
        # wandb logger
        elif isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log(
                {name: wandb.Histogram(tensor.detach().cpu().numpy())},
                step=step,
            )
        else:
            print(
                f"[INFO] histogram '{name}' cannot be logged. Logger not supported. Logger: {self.logger}"
            )
