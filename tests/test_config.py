import pytest
import os


def test_config():
    from config.config import config, SEED, IMG_SIZE, BATCH_SIZE, N_EPOCHS

    assert SEED
    assert IMG_SIZE > 0
    assert BATCH_SIZE > 0
    assert N_EPOCHS > 0
    assert config


def test_files_exist():
    from config.config import data_config

    assert os.path.exists(data_config.data_path)


def test_optimizers():
    from config.config import config
    from tests.fixtures.model import NeuralNetwork

    model = NeuralNetwork()

    optimizer = config.optimizer
    optimizer = optimizer(model.parameters(), **config.optimizer_kwargs)

    scheduler = config.scheduler
    scheduler = scheduler(optimizer, **config.scheduler_kwargs)
