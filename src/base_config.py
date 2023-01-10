from dataclasses import asdict

import typing as tp
from dataclasses import dataclass

import albumentations as albu
from torchvision import transforms
import torch
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection, Metric
from pytorch_lightning import callbacks


@dataclass
class Config:
    data_path: str
    num_classes: int
    num_workers: int
    seed: int
    loss: torch.nn.Module
    lr: int
    optimizer: Optimizer
    optimizer_kwargs: tp.Mapping
    scheduler: tp.Any
    scheduler_kwargs: tp.Mapping
    preprocessing: transforms.Compose
    img_size: int
    augmentations: tp.Union[albu.Compose, tp.Callable]
    batch_size: tp.Optional[int]
    n_epochs: int
    experiment_name: str
    model_kwargs: tp.Mapping
    trainer_kwargs: dict
    metrics: tp.Union[MetricCollection, Metric]
    logger_kwargs: dict
    callbacks: tp.List[callbacks.Callback]

    def to_dict(self) -> dict:
        # res = {}
        # for k, v in asdict(self).items():
        #     try:
        #         if isinstance(v, torch.nn.Module):
        #             res[k] = v.__class__.__name__
        #         # elif isinstance(v, dict):
        #         #     res[k] = json.dumps(v, indent=0, sort_keys=True)
        #         else:
        #             res[k] = str(v)
        #     except Exception:
        #         res[k] = str(v)
        return asdict(self)
