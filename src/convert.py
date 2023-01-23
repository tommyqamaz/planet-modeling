import argparse

import torch

from config.config import config
from src.utils import get_model_from_checkpoint


def get_parser() -> argparse.ArgumentParser:
    """get parser to create onnx model
    Returns:
        argparse.ArgumentParser: parser
    """
    parser = argparse.ArgumentParser(description="get predict.")
    parser.add_argument("model_path", type=str, help="path to model to convert")
    parser.add_argument("onnx_model", type=str, help="where to save converted model")

    return parser


def convert_model(
    model_path: str, onnx_model_path: str = "pvtv2.onnx", verbose: bool = True
) -> None:
    """Convert model to ONNX

    Args:
        model_path (str): which model to convert
        onnx_model_path (str, optional): where save the converted model.
        Defaults to "pvtv2.onnx".
        verbose (bool, optional): verbosity. Defaults to True.
    """

    model_to_onnx = get_model_from_checkpoint(model_path)

    dummy_input = torch.randn(1, 3, config.img_size, config.img_size)

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
        print(f"Model saved to {onnx_model_path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    convert_model(args.model_path, args.onnx_model)
