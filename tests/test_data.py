import pytest

import numpy as np

from src.data import AmazonDataset
from tests.fixtures.config import data_config
from src.augmentations import get_train_augmentation, get_val_augmentation
from src.data import get_ohe, get_df
from config.consts import IMG_SIZE, NUM_CLASSES


@pytest.fixture
def dataset(request):
    df = get_df(data_config.data_path)
    ohe = get_ohe(df)
    if request.param == "train":
        augmentations = get_train_augmentation(data_config.img_size)
    else:
        augmentations = get_val_augmentation(data_config.img_size)
    ds = AmazonDataset(df, augmentations, ohe, data_config.data_path)
    return ds


@pytest.mark.parametrize("dataset", ["train", "val"], indirect=True)
def test_dataset(dataset):

    x, y = dataset[1]

    assert x.shape == (3, IMG_SIZE, IMG_SIZE)
    assert y.shape == (NUM_CLASSES,)

    assert x.dtype == np.float32
    assert y.dtype == np.float32
