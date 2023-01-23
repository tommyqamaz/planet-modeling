import os
import typing as tp

import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from src.model import LitModel
from src.data import AmazonDM
from config.config import config
from config.base_config import DataConfig


def get_last_saved_model(weights_dir: str) -> str:
    """find the last saved model in ckpt format

    Args:
        weights_dir (str): path to dir where models are saved

    Returns:
        str: last saved model's path
    """
    files = [
        os.path.join(weights_dir, f)
        for f in os.listdir(weights_dir)
        if f.endswith("ckpt")
    ]
    files.sort(key=lambda x: os.path.getmtime(x))
    return files[-1]


def get_ys(
    dl: DataLoader, model: nn.Module, num_iter: int = 2
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """calculate y_pred and y_true

    Args:
        dl (DataLoader): any dataloader

        model (nn.Module): use it to predict, model must returns logits

        num_iter (int, optional): maximum batches to concat. Defaults to 2.

    Returns:
        tp.Tuple[torch.Tensor, torch.Tensor]:
        tensor with predictions and tensor with ground true values (y true)
    """
    y_pred, y_true = torch.tensor([]), torch.tensor([])
    i = 0
    for (x, y) in tqdm(dl):
        y_pred_now = model(x)
        y_true = torch.concat((y_true, y), dim=0)  # slow, append method is better
        y_pred = torch.concat((y_pred, y_pred_now), dim=0)

        if i == num_iter:
            return y_pred, y_true
        i += 1
    return y_pred, y_true


def find_th(
    y_pred: torch.Tensor, y_true: torch.Tensor, metric: tp.Callable, num_iter=1000
):
    """Greed search for better thresholds for multilabel classification
       y_pred.shape == y_true.shape

    Args:
        y_pred (torch.Tensor): prediction of model
        y_true (torch.Tensor): ground truth
        metric (function): any metric function (where first arg for preds,
        second one for gt)
        num_iter (int, optional): number of iterations for each class. Defaults to 1000.

    Returns:
        torch.Tensor: thresholds
    """
    best_score = 0.0
    best_th = 0.0
    ths = torch.zeros_like(y_pred[0])
    for i in range(ths.size()[0]):
        for new_th in torch.linspace(0, 0.35, num_iter):
            ths[i] = new_th
            new_score = metric((y_pred.sigmoid() > ths).to(int), y_true)
            if new_score > best_score:
                best_score = new_score
                best_th = new_th
        ths[i] = best_th

    return ths


def get_model_from_checkpoint(model_path: str) -> torch.nn.Module:
    """load pytorch lightning model from checkpoint

    Args:
        model_path (str): path to model

    Returns:
        torch.nn.Module: loaded lit model
    """
    lit_model = LitModel.load_from_checkpoint(model_path)
    lit_model.eval()
    return lit_model


def save_ths(model_path: str, data_config: DataConfig) -> None:
    """find and save thresholds

    Args:
        model_path (str): model to use to find thresholds
        data_config (DataConfig):
    """

    model = get_model_from_checkpoint(model_path)

    dm = AmazonDM(data_config)
    dm.setup()
    val_dl = dm.val_dataloader()
    y_pred, y_true = get_ys(val_dl, model)
    ths = find_th(y_pred, y_true, config.metrics["BinaryFBetaScore"], num_iter=1000)

    thresholds_path = os.path.join("inference", "thresholds.txt")

    with open(thresholds_path, "w") as f:
        f.write(" ".join(str(i) for i in ths.tolist()))
