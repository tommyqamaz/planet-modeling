from src.model import LitModel
import torch
from torch import nn
import argparse

model_path = "weights/epoch=24-val_BinaryFBetaScore=0.873-valid_loss_epoch=0.0859.ckpt"


class PredictModel(nn.Module):
    def __init__(self, lit_model, ths):
        super().__init__()
        self.model = lit_model
        self.ths = ths

    def forward(self, x):
        res = self.model(x)
        res = res.sigmoid()
        res = (res > self.ths).to(int)
        return res


def get_parser() -> argparse.ArgumentParser:
    """get parser
    Returns:
        argparse.ArgumentParser: parser
    """
    parser = argparse.ArgumentParser(description="get predict.")
    parser.add_argument("model_path", type=str, help="path to model to convert")
    parser.add_argument("onnx_model", type=str, help="where to save converted model")

    return parser


ths = torch.tensor(
    [
        0.16,
        0.06,
        0.17,
        0.13,
        0.11,
        0.12,
        0.05,
        0.1,
        0.16,
        0.23,
        0.14,
        0.1,
        0.16,
        0.22,
        0.28,
        0.08,
        0.13,
    ]
)


def convert_model(
    model_path: str, onnx_model_path: str = "pvtv2.onnx", verbose: bool = True
) -> None:
    lit_model = LitModel.load_from_checkpoint(model_path)
    lit_model.eval()
    model_to_onnx = PredictModel(lit_model, ths)
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model_to_onnx,
        dummy_input,
        onnx_model_path,
        verbose=False,
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
        export_params=True,
    )

    if verbose:
        print("Model saved to {onnx_model_path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    convert_model(args.model_path, args.onnx_model)
