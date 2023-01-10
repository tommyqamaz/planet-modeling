from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import os
import torch
import cv2
import pandas as pd
import numpy as np

# from matplotlib import pyplot as plt

from src.config import config


class AmazonDatasetError(Exception):
    pass


class AmazonDataset(Dataset):
    def __init__(
        self,
        df,
        ohe_tags,
        path,
        is_train=True,
        idx_tta=None,
        config=config,
    ):
        super().__init__()
        self.df = df
        self.ohe_tags = ohe_tags
        self.preprocessing = config.preprocessing
        self.augmentations = config.augmentations

        if isinstance(path, str):
            self.paths = [path]
        elif isinstance(path, (list, tuple)):
            self.paths = path
        else:
            raise AmazonDatasetError(
                f"Path type must be str, list or tuple, got: {type(path)}"
            )

        self.is_train = is_train
        if not is_train:
            if idx_tta not in list(range(6)):
                raise AmazonDatasetError(
                    f"In test mode, 'idx_tta' must be an int belonging to [0, 5], got: {repr(idx_tta)}"  # noqa E501
                )
            self.idx_tta = idx_tta

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.iloc[idx].image_name + ".jpg"
        for path in self.paths:
            if filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                break
        else:
            raise AmazonDatasetError(f"Can't fetch {filename} among {self.paths}")
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.ohe_tags[idx]
        return img, label

    def collate_fn(self, batch):
        imgs, labels = [], []
        for (img, label) in batch:
            img = self.augmentations(img)
            img = torch.tensor(img)
            img = img.permute(2, 0, 1)
            img = self.preprocessing(img)
            imgs.append(img[None])
            labels.append(label)

        # device = "cuda" if torch.cuda.is_available() else "cpu"

        imgs = torch.cat(imgs).float()  # .to(device)
        labels = torch.tensor(np.array(labels)).float()  # .to(device)
        return imgs, labels

    # def load_img(self, idx, ax=None):
    #     img, ohe_label = self[idx]
    #     label = self.df.iloc[idx].tags
    #     title = f"{label} - {ohe_label}"
    #     if ax is None:
    #         plt.imshow(img)
    #         plt.title(title)
    #     else:
    #         ax.imshow(img)
    #         ax.set_title(title)


def add_list_tags(data_path) -> pd.DataFrame:
    path_class = os.path.join(data_path, "train_classes.csv")
    df_class = pd.read_csv(path_class)
    df_class["list_tags"] = df_class.tags.str.split(" ")

    return df_class


def config_data(df_train, df_val, data_path, config):

    encoder = MultiLabelBinarizer()
    ohe_tags_train = encoder.fit_transform(df_train.list_tags.values)
    ohe_tags_val = encoder.transform(df_val.list_tags.values)

    path_train = os.path.join(data_path, "train-jpg")

    ds_train = AmazonDataset(df_train, ohe_tags_train, path=path_train)
    ds_val = AmazonDataset(df_val, ohe_tags_val, path=path_train)

    dl_train = DataLoader(
        ds_train,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=ds_train.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=ds_val.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
    )

    return ds_train, ds_val, dl_train, dl_val, encoder


def get_dataloaders(config):

    df_class = add_list_tags(config.data_path)

    df_train, df_val = train_test_split(
        df_class,
        test_size=0.2,
        random_state=config.seed,
    )  # TODO: config.test_size

    ds_train, ds_val, dl_train, dl_val, encoder = config_data(
        df_train, df_val, config.data_path, config
    )

    return dl_train, dl_val
