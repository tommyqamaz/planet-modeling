import typing as tp
from dataclasses import dataclass, asdict

import torch
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection, Metric
from pytorch_lightning import callbacks


@dataclass
class Config:
    experiment_name: str
    num_classes: int
    n_epochs: int
    batch_size: int
    seed: int
    loss: torch.nn.Module
    lr: int
    optimizer: tp.Type[Optimizer]
    optimizer_kwargs: tp.Mapping
    scheduler: tp.Any
    scheduler_kwargs: tp.Mapping
    img_size: int
    model_kwargs: tp.Mapping
    trainer_kwargs: dict
    metrics: tp.Union[MetricCollection, Metric]
    logger_kwargs: dict
    callbacks: tp.List[callbacks.Callback]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DataConfig:
    seed: int
    img_size: int
    test_size: float
    data_path: str
    num_workers: int
    batch_size: int
    num_classes: int

    def to_dict(self) -> dict:
        return asdict(self)
