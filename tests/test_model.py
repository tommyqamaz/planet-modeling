import pytest
import torch
from src.model import LitModel
from config.config import config, IMG_SIZE

from tests.fixtures.model import NeuralNetwork


@pytest.fixture()
def model():
    return NeuralNetwork()


def test_forward(model):
    input_img = torch.ones(1, 1, IMG_SIZE, IMG_SIZE)

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
