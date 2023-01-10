from src.base_config import Config
from src.config import IMG_SIZE, preprocessing, custom_augment
from torch import nn
from torch.optim import lr_scheduler
from torch import optim
import torchmetrics
import os

DATA_PATH = "tests/fixtures/data"
NUM_CLASSES = 17

config = Config(
    num_classes=NUM_CLASSES,
    data_path=DATA_PATH,
    num_workers=os.cpu_count(),
    seed=12,
    lr=1e-3,
    loss=nn.BCEWithLogitsLoss,
    optimizer=optim.AdamW,
    optimizer_kwargs={},
    scheduler=lr_scheduler.CosineAnnealingLR,
    scheduler_kwargs={"T_max": 1000, "eta_min": 0.0005},
    model_kwargs={"model_name": "pvt_v2_b1", "pretrained": True},
    preprocessing=preprocessing,
    augmentations=custom_augment,
    batch_size=4,
    img_size=IMG_SIZE,
    n_epochs=1,
    experiment_name="test",
    trainer_kwargs={"accelerator": "cpu", "fast_dev_run": True, "deterministic": True},
    metrics=torchmetrics.MetricCollection(
        metrics=[
            torchmetrics.AUROC(task="binary", num_classes=NUM_CLASSES),
            torchmetrics.FBetaScore(task="binary", beta=2.0, num_classes=NUM_CLASSES),
        ]
    ),
    logger_kwargs={},
    callbacks=[],
)
