import timm
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from src.config import config
from src.data import get_dataloaders
from src.model import LitModel


def main(config):
    dl_train, dl_val = get_dataloaders(config)

    model = timm.create_model(num_classes=config.num_classes, **config.model_kwargs)

    lit_model = LitModel(
        model,
        lr=config.lr,
        loss_fn=config.loss,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
        scheduler_kwargs=config.scheduler_kwargs,
        metrics=config.metrics,
    )

    if config.logger_kwargs:
        logger = WandbLogger(**config.logger_kwargs)
        logger.log_hyperparams(config.to_dict())
    else:
        logger = None

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        logger=logger,
        callbacks=config.callbacks,
        **config.trainer_kwargs,
    )
    trainer.fit(lit_model, dl_train, dl_val)


if __name__ == "__main__":
    pl.seed_everything(config.seed)
    torch.cuda.empty_cache()
    main(config)
