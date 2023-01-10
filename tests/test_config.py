import pytest
import os


def test_config():
    from src.config import config, SEED, IMG_SIZE, BATCH_SIZE, N_EPOCHS

    assert SEED
    assert IMG_SIZE > 0
    assert BATCH_SIZE > 0
    assert N_EPOCHS > 0
    assert config


def test_files_exist():
    from src.config import DATA_PATH

    assert os.path.exists(DATA_PATH)


def test_optimizers():
    from src.config import config
    from tests.fixtures.model import NeuralNetwork

    model = NeuralNetwork()

    optimizer = config.optimizer
    optimizer = optimizer(model.parameters(), **config.optimizer_kwargs)

    scheduler = config.scheduler
    scheduler = scheduler(optimizer, **config.scheduler_kwargs)
