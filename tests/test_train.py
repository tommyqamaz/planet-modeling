import pytest
from src.train import main
from tests.fixtures.config import config


def test_main():
    main(config)
