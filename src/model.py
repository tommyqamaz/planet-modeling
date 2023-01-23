import pytorch_lightning as pl
from torch.nn import Module
from torch.optim import Optimizer
import typing as tp
from torchmetrics import MetricCollection, Metric


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model: Module,
        lr: float,
        loss_fn: Module,
        optimizer: Optimizer,
        scheduler: tp.Any,
        scheduler_kwargs: tp.Mapping,
        metrics: tp.Union[MetricCollection, Metric],
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = loss_fn()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs

        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        output = self.train_metrics(y_hat, y)
        self.log_dict(output)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.valid_metrics.update(y_hat, y)
        self.log("valid_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.test_metrics.update(y_hat, y)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, on_step=True)

    def on_test_start(self) -> None:
        self.test_metrics.reset()

    def on_validation_epoch_start(self) -> None:
        self.valid_metrics.reset()

    def validation_epoch_end(self, outputs):
        output = self.valid_metrics.compute()
        self.log_dict(output)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        scheduler = self.scheduler(optimizer, **self.scheduler_kwargs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
