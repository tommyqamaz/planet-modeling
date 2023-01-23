import os

from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchmetrics import MetricCollection
from pytorch_lightning import callbacks
import torchmetrics

from config.base_config import Config, DataConfig
from config.consts import NUM_CLASSES, SEED, IMG_SIZE, N_EPOCHS, BATCH_SIZE


config = Config(
    num_classes=NUM_CLASSES,
    seed=SEED,
    batch_size=BATCH_SIZE,
    lr=1e-3,
    loss=nn.BCEWithLogitsLoss,
    optimizer=optim.AdamW,
    optimizer_kwargs={},
    scheduler=lr_scheduler.CosineAnnealingLR,
    scheduler_kwargs={"T_max": 10, "eta_min": 1e-5},
    model_kwargs={"model_name": "pvt_v2_b1", "pretrained": True},
    img_size=IMG_SIZE,
    n_epochs=N_EPOCHS,
    experiment_name="soldatov_modeling",
    trainer_kwargs={
        "accelerator": "gpu",
        "gpus": 2,
        "devices": [0, 1],
        "strategy": "ddp",
        "fast_dev_run": False,
        "log_every_n_steps": 32,
        "auto_scale_batch_size": "binsearch",
        "enable_checkpointing": True,
    },
    logger_kwargs={
        "project": "soldatov_modelingv2",
        "log_model": "all",
        "name": "pvt-dvc",
    },
    metrics=MetricCollection(
        metrics=[
            torchmetrics.AUROC(task="binary", num_classes=NUM_CLASSES),
            torchmetrics.FBetaScore(task="binary", beta=2.0, num_classes=NUM_CLASSES),
        ]
    ),
    callbacks=[
        callbacks.ModelCheckpoint(
            dirpath="weights",
            # monitor="val_BinaryFBetaScore",
            filename="{epoch}-{val_BinaryFBetaScore:.3f}-{valid_loss_epoch:.4f}",
            auto_insert_metric_name=True,
            save_weights_only=True,
            every_n_epochs=1,
        ),
        callbacks.TQDMProgressBar(),
        callbacks.LearningRateMonitor(logging_interval="step"),
    ],
)


data_config = DataConfig(
    seed=SEED,
    img_size=IMG_SIZE,
    test_size=0.2,
    batch_size=BATCH_SIZE,
    data_path="./data/planet/",
    num_workers=os.cpu_count(),
    num_classes=NUM_CLASSES,
)
