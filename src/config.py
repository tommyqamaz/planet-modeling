import cv2
import numpy as np
import os

from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms as T
from torchmetrics import MetricCollection
from pytorch_lightning import callbacks
import torchmetrics

from src.base_config import Config

from dotenv import load_dotenv

load_dotenv()

N_GPUS = 2
NUM_CLASSES = 17
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 220
N_EPOCHS = 25
# ROOT_PATH = os.path.join(os.environ.get("ROOT_PATH"))
DATA_PATH = "./data/planet/"

preprocessing = T.Compose(
    [
        T.ToPILImage(),
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def custom_augment(img):  # noqa C901
    """
    Discrete rotation and horizontal flip.
    Random during training and non random during testing for TTA.
    Not implemented in torchvision.transforms, hence this function.
    """
    choice = np.random.randint(0, 6)  # if self.is_train else self.idx_tta
    if choice == 0:
        # Rotate 90
        img = cv2.rotate(img, rotateCode=cv2.ROTATE_90_CLOCKWISE)
    if choice == 1:
        # Rotate 90 and flip horizontally
        img = cv2.rotate(img, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        img = cv2.flip(img, flipCode=1)
    if choice == 2:
        # Rotate 180
        img = cv2.rotate(img, rotateCode=cv2.ROTATE_180)
    if choice == 3:
        # Rotate 180 and flip horizontally
        img = cv2.rotate(img, rotateCode=cv2.ROTATE_180)
        img = cv2.flip(img, flipCode=1)
    if choice == 4:
        # Rotate 90 counter-clockwise
        img = cv2.rotate(img, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
    if choice == 5:
        # Rotate 90 counter-clockwise and flip horizontally
        img = cv2.rotate(img, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.flip(img, flipCode=1)
    return img


config = Config(
    data_path=DATA_PATH,
    num_classes=NUM_CLASSES,
    num_workers=os.cpu_count(),
    seed=SEED,
    lr=1e-3,
    loss=nn.BCEWithLogitsLoss,
    optimizer=optim.AdamW,
    optimizer_kwargs={},
    scheduler=lr_scheduler.CosineAnnealingLR,
    scheduler_kwargs={"T_max": 10, "eta_min": 1e-5},
    model_kwargs={"model_name": "pvt_v2_b1", "pretrained": True},
    preprocessing=preprocessing,
    augmentations=custom_augment,
    batch_size=BATCH_SIZE,
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
        "project": "soldatov_modeling",
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
