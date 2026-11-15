from __future__ import annotations

import cv2
import numpy as np
import pytest

import defect_cls.opencv_pipeline as pipeline
from defect_cls.opencv_pipeline import (
    OpenCVValidationError,
    assess_opencv_quality,
    decode_and_validate_image,
    preprocess_defect_candidates,
)


def _png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_decode_validates_image_and_metadata() -> None:
    image = np.full((12, 20, 3), 120, dtype=np.uint8)
    bgr, metadata = decode_and_validate_image(_png_bytes(image), "sample.png")
    assert bgr.shape == (12, 20, 3)
    assert metadata == {"width": 20, "height": 12, "channels": 3, "extension": ".png"}


@pytest.mark.parametrize("filename,payload", [("bad.gif", b"GIF89a"), ("bad.png", b"not an image")])
def test_decode_rejects_invalid_uploads(filename: str, payload: bytes) -> None:
    with pytest.raises(OpenCVValidationError):
        decode_and_validate_image(payload, filename)


def test_decode_rejects_pixel_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "MAX_UPLOAD_PIXELS", 100)
    payload = _png_bytes(np.zeros((11, 10, 3), dtype=np.uint8))
    with pytest.raises(OpenCVValidationError, match="픽셀"):
        decode_and_validate_image(payload, "large.png")


def test_quality_orders_blur_and_contrast() -> None:
    rng = np.random.default_rng(7)
    sharp = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 2)
    flat = np.full((64, 64, 3), 128, dtype=np.uint8)
    assert assess_opencv_quality(blurred)["sharpness"] < assess_opencv_quality(sharp)["sharpness"]
    assert assess_opencv_quality(flat)["contrast"] < assess_opencv_quality(sharp)["contrast"]


def test_quality_flags_extreme_brightness_and_noise() -> None:
    dark = np.zeros((64, 64, 3), dtype=np.uint8)
    rng = np.random.default_rng(11)
    noisy = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert assess_opencv_quality(dark)["flags"]["extreme_brightness"] is True
    assert assess_opencv_quality(noisy)["noise_sigma"] > assess_opencv_quality(dark)["noise_sigma"]


def test_preprocessing_preserves_one_pixel_thin_structure() -> None:
    image = np.full((80, 80, 3), 100, dtype=np.uint8)
    cv2.line(image, (10, 20), (70, 20), (220, 220, 220), thickness=1)
    result = preprocess_defect_candidates(image)
    assert np.count_nonzero(result["candidate_mask"]) > 0


def test_preprocessing_produces_mask_and_does_not_mutate_input() -> None:
    image = np.full((80, 80, 3), 100, dtype=np.uint8)
    cv2.circle(image, (40, 40), 12, (220, 220, 220), -1)
    before = image.copy()
    result = preprocess_defect_candidates(image)
    assert np.array_equal(image, before)
    assert result["enhanced_gray"].shape == image.shape[:2]
    assert result["candidate_mask"].dtype == np.uint8
    assert result["candidate_overlay_bgr"].shape == image.shape
    assert np.count_nonzero(result["candidate_mask"]) > 0
