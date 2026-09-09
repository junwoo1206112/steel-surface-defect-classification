from __future__ import annotations

import pytest
import torch

from defect_cls.inference import (
    DEFAULT_THRESHOLD,
    MAX_UPLOAD_BYTES,
    load_image_tensor,
    predict,
    validate_upload,
)
from defect_cls.model import build_model


def test_validate_upload_accepts_valid_file() -> None:
    ok, message = validate_upload("crazing_1.jpg", 1024)
    assert ok is True
    assert message == "ok"


@pytest.mark.parametrize(
    "filename,size_bytes",
    [
        ("payload.exe", 100),
        ("noext", 100),
        ("image.jpg", 0),
        ("image.png", MAX_UPLOAD_BYTES + 1),
    ],
)
def test_validate_upload_rejects_invalid_files(filename: str, size_bytes: int) -> None:
    ok, message = validate_upload(filename, size_bytes)
    assert ok is False
    assert message != "ok"


def test_load_image_tensor_from_bytes() -> None:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("L", (32, 32), color=128).save(buffer, format="JPEG")
    tensor = load_image_tensor(buffer.getvalue())
    assert tensor.shape == (1, 3, 200, 200)


def test_load_image_tensor_respects_grayscale_mode() -> None:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("L", (32, 32), color=128).save(buffer, format="PNG")
    tensor = load_image_tensor(buffer.getvalue(), image_mode="L")
    assert tensor.shape == (1, 1, 200, 200)


def test_load_image_tensor_rejects_corrupt_bytes() -> None:
    with pytest.raises(ValueError):
        load_image_tensor(b"this is not an image")


def test_predict_output_contract() -> None:
    model = build_model(pretrained=False)
    tensor = torch.randn(1, 3, 200, 200)
    result = predict(model, tensor, threshold=DEFAULT_THRESHOLD)
    assert set(result) == {
        "class_label",
        "class_index",
        "confidence",
        "probabilities",
        "needs_review",
        "threshold",
    }
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["needs_review"] is (result["confidence"] < DEFAULT_THRESHOLD)


def test_predict_low_confidence_review_flag() -> None:
    model = build_model(pretrained=False)
    tensor = torch.randn(1, 3, 200, 200)
    result = predict(model, tensor, threshold=0.999999)
    assert result["needs_review"] is True


def test_predict_temperature_preserves_argmax_and_scales_confidence() -> None:
    model = build_model(pretrained=False)
    tensor = torch.randn(1, 3, 200, 200)
    hot = predict(model, tensor, temperature=5.0)
    cold = predict(model, tensor, temperature=0.5)
    assert hot["class_index"] == cold["class_index"]
    assert hot["confidence"] < cold["confidence"]
    assert abs(sum(hot["probabilities"].values()) - 1.0) < 1e-5


def test_predict_rejects_nonpositive_temperature() -> None:
    model = build_model(pretrained=False)
    with pytest.raises(ValueError):
        predict(model, torch.randn(1, 3, 200, 200), temperature=0.0)
