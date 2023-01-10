import pytest

from src.data import get_dataloaders
from tests.fixtures.config import config, IMG_SIZE


def test_dataloaders():
    dl_train, dl_val = get_dataloaders(config)

    batch_size = (config.batch_size, 3, IMG_SIZE, IMG_SIZE)

    batch_t = next(iter(dl_train))
    assert batch_t[0].shape == batch_size
    assert batch_t[1].shape == (config.batch_size, config.num_classes)

    batch_v = next(iter(dl_val))
    assert batch_v[0].shape == batch_size
    assert batch_v[1].shape == (config.batch_size, config.num_classes)
