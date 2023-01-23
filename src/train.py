import os

import timm
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from config.config import config, data_config
from config.base_config import Config, DataConfig
from src.data import AmazonDM
from src.model import LitModel
from src.convert import convert_model
from src.utils import get_last_saved_model, save_ths


def _train(config: Config, data_config: DataConfig):

    model = timm.create_model(num_classes=config.num_classes, **config.model_kwargs)

    dm = AmazonDM(data_config)

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
    trainer.fit(lit_model, datamodule=dm)
    trainer.test(lit_model, datamodule=dm)


if __name__ == "__main__":
    pl.seed_everything(config.seed)
    torch.cuda.empty_cache()
    _train(config, data_config)
    print(os.listdir())
    models_path = get_last_saved_model("weights")
    onnx_path = os.path.join(
        "inference", config.model_kwargs["model_name"] + "test" + ".onnx"
    )

    convert_model(models_path, onnx_path)
    save_ths(models_path, data_config)
