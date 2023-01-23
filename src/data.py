import os
import cv2
import typing as tp

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from pytorch_lightning import LightningDataModule
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


from config.base_config import DataConfig
from src.augmentations import get_train_augmentation, get_val_augmentation


class AmazonDataset(Dataset):
    def __init__(self, df, augmentations, ohe_tags, path):
        super().__init__()
        self._df = df
        self._path = path
        self._ohe_tags = ohe_tags
        self._augmentations = augmentations

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int) -> tp.Tuple[np.ndarray, np.ndarray]:
        filename = self._df.iloc[idx].image_name + ".jpg"
        file_path = os.path.join(self._path, "train-jpg", filename)  # fix this later

        img = cv2.imread(file_path)

        img = self._augmentations(image=img)["image"]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))

        label = self._ohe_tags[idx]
        return img, label


def _multilabel_stratified_split(
    df: pd.DataFrame, targets: np.ndarray, train_size: float
) -> tp.Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:

    n_folds = _get_num_folds_from_train_size(train_size)
    mskf = MultilabelStratifiedKFold(n_splits=n_folds, random_state=777, shuffle=True)
    split = mskf.split(df, targets)
    train_index, other_index = list(split)[0]
    train_df = df.iloc[train_index]
    other_df = df.iloc[other_index]
    train_targets = targets[train_index]
    other_targets = targets[other_index]

    return train_df, train_targets, other_df, other_targets


def _get_num_folds_from_train_size(train_size: float) -> int:
    val_size = 1 - train_size
    return int(train_size / val_size) + 1


def get_ohe(df: pd.DataFrame) -> np.ndarray:
    encoder = MultiLabelBinarizer()
    targets = encoder.fit_transform(df.list_tags.values)
    return targets.astype(np.float32)


def get_split(
    df: pd.DataFrame, train_size: float
) -> tp.Tuple[tp.Tuple[pd.DataFrame, np.ndarray]]:
    """Split arrays or matrices into random train, val and test subsets.

    Args:
        df (pd.DataFrame):
        Init df.

        train_size (float):
        If float, should be between 0.0 and 1.0 and represent
        the proportion of the dataset to include in the test split.

    Returns:
        tp.Tuple[tp.Tuple[pd.DataFrame, np.ndarray]]:
        3 pairs of [pd.DataFrame, np.ndarray] where pd.Datagrames are features (images)
        and np.ndarrays are labels (ohe here).
    """
    targets = get_ohe(df)

    train_df, train_targets, other_df, other_targets = _multilabel_stratified_split(
        df, targets, train_size
    )
    val_df, val_targets, test_df, test_targets = _multilabel_stratified_split(
        other_df, other_targets, 0.5
    )
    return train_df, train_targets, val_df, val_targets, test_df, test_targets


def get_df(data_path: str) -> pd.DataFrame:
    path_class = os.path.join(data_path, "train_classes.csv")
    df_class = pd.read_csv(path_class)
    df_class["list_tags"] = df_class.tags.str.split(" ")

    return df_class


class AmazonDM(LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self._data_path = config.data_path
        self._batch_size = config.batch_size
        self._n_workers = config.num_workers
        self._train_size = 1 - config.test_size
        self._train_augs = get_train_augmentation(config.img_size)
        self._test_augs = get_val_augmentation(config.img_size)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: tp.Optional[str] = None):
        df = get_df(self._data_path)
        (
            train_df,
            ohe_train,
            val_df,
            ohe_val,
            test_df,
            ohe_test,
        ) = get_split(df, self._train_size)

        self.train_dataset = AmazonDataset(
            train_df, self._train_augs, ohe_tags=ohe_train, path=self._data_path
        )
        self.val_dataset = AmazonDataset(
            val_df, self._test_augs, ohe_tags=ohe_val, path=self._data_path
        )
        self.test_dataset = AmazonDataset(
            test_df, self._test_augs, ohe_tags=ohe_test, path=self._data_path
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
