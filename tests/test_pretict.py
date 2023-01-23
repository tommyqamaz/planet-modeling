import pytest

from src.onnx_predict import main


def test_predict():
    IMG_PATH = "tests/fixtures/data/train-jpg/train_0.jpg"
    MODEL_PATH = "inference/pvt_v2_b1.onnx"
    answer = "primary"

    result = main(
        IMG_PATH,
        MODEL_PATH,
    )

    assert answer in result
