import os

from torch import nn
from torch.optim import lr_scheduler
from torch import optim
import torchmetrics

from config.base_config import Config, DataConfig
from config.config import IMG_SIZE

DATA_PATH = "tests/fixtures/data"
NUM_CLASSES = 17
SEED = 12
BATCH_SIZE = 4

config = Config(
    num_classes=NUM_CLASSES,
    seed=SEED,
    lr=1e-3,
    loss=nn.BCEWithLogitsLoss,
    optimizer=optim.AdamW,
    optimizer_kwargs={},
    scheduler=lr_scheduler.CosineAnnealingLR,
    scheduler_kwargs={"T_max": 1000, "eta_min": 0.0005},
    model_kwargs={"model_name": "pvt_v2_b1", "pretrained": True},
    batch_size=BATCH_SIZE,
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

data_config = DataConfig(
    seed=SEED,
    img_size=IMG_SIZE,
    test_size=0.4,
    batch_size=BATCH_SIZE,
    data_path="tests/fixtures/data",
    num_workers=os.cpu_count(),
    num_classes=NUM_CLASSES,
)
