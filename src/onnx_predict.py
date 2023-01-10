import onnx
import onnxruntime as ort

import numpy as np
import cv2
from typing import Tuple

import argparse

classes = np.array(
    [
        "agriculture",
        "artisinal_mine",
        "bare_ground",
        "blooming",
        "blow_down",
        "clear",
        "cloudy",
        "conventional_mine",
        "cultivation",
        "habitation",
        "haze",
        "partly_cloudy",
        "primary",
        "road",
        "selective_logging",
        "slash_burn",
        "water",
    ]
)


def get_parser() -> argparse.ArgumentParser:
    """get parser
    Returns:
        argparse.ArgumentParser: parser
    """
    parser = argparse.ArgumentParser(description="get predict.")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument(
        "-o",
        "--onnx",
        help="path to onnx model",
        default="pvtv2.onnx",
    )
    return parser


def onnx_preprocessing(
    image: np.ndarray,
    image_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Convert numpy-image to array for inference ONNX Runtime model.
    """

    # resize
    image = cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_LINEAR)

    # normalize
    mean = np.array((0.485, 0.456, 0.406), dtype=np.float32) * 255.0
    std = np.array((0.229, 0.224, 0.225), dtype=np.float32) * 255.0
    denominator = np.reciprocal(std, dtype=np.float32)
    image = image.astype(np.float32)
    image -= mean
    image *= denominator

    # transpose
    image = image.transpose((2, 0, 1))[None]
    return image


def predict(img: np.array, onnx_path: str = "pvtv2.onnx") -> np.array:
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    providers = [
        # 'CUDAExecutionProvider',
        "CPUExecutionProvider",
    ]

    ort_session = ort.InferenceSession(onnx_path, providers=providers)

    # img = cv2.imread(IMG_PATH)
    # готовим входной тензор
    onnx_input_tensor = onnx_preprocessing(img)

    ort_inputs = {ort_session.get_inputs()[0].name: onnx_input_tensor}

    ort_outputs = ort_session.run(None, ort_inputs)[0]

    return ort_outputs.flatten()


def convert_to_classnames(predictions: np.array, classnames: np.array) -> list:
    """returns list with predicted classes"""
    assert predictions.size == classnames.size

    return classnames[predictions.astype(bool).flatten()].tolist()


def main(img_path: str, onnx_path: str = "pvtv2.onnx", classnames=classes):
    img = cv2.imread(img_path)
    predictions = predict(img, onnx_path)
    result = convert_to_classnames(predictions, classnames)
    return result


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    result = main(args.image_path, args.onnx)
    print(result)
