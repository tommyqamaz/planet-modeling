import pytest

from src.train import _train
from tests.fixtures.config import config, data_config


def test_main():
    _train(config, data_config)
