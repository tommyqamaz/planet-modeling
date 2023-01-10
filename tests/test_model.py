import pytest
import torch
from src.model import LitModel
from src.config import config, IMG_SIZE

from tests.fixtures.model import NeuralNetwork


def test_forward():
    input_img = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)

    model = NeuralNetwork()

    lit_model = LitModel(
        model,
        lr=config.lr,
        loss_fn=config.loss,
        optimizer=config.optimizer,
        scheduler=config.optimizer,
        scheduler_kwargs=config.scheduler_kwargs,
        metrics=config.metrics,
    )

    assert lit_model(input_img).shape == (1, config.num_classes)
