from __future__ import annotations

import cv2
import numpy as np
import pytest

from defect_cls.upload_review import inspect_upload, review_decision


def _encoded_image(extension: str) -> bytes:
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    ok, encoded = cv2.imencode(extension, image)
    assert ok
    return encoded.tobytes()


@pytest.mark.parametrize(
    ("extension", "filename"), [(".png", "image.png"), (".jpg", "image.jpg"), (".bmp", "image.bmp")]
)
def test_inspect_upload_accepts_supported_decodable_images(extension: str, filename: str) -> None:
    result = inspect_upload(_encoded_image(extension), filename, image_mode="RGB")
    assert result["tensor"].shape == (1, 3, 200, 200)
    assert result["metadata"]["extension"] == extension


def test_inspect_upload_rejects_corrupt_image() -> None:
    with pytest.raises(ValueError, match="손상|이미지"):
        inspect_upload(b"not an image", "broken.png", image_mode="RGB")


def test_review_decision_prioritizes_quality_over_confidence() -> None:
    cv_quality = {"flags": {"too_noisy": True}}
    legacy_quality = {"flags": {"too_noisy": False}}
    assert review_decision(cv_quality, legacy_quality, needs_review=False) == {
        "status": "quality",
        "flagged": ["OpenCV:too_noisy"],
    }


@pytest.mark.parametrize(
    ("cv_flags", "legacy_flags", "needs_review", "expected"),
    [
        ({"too_noisy": False}, {"too_noisy": True}, False, "quality"),
        ({"too_noisy": False}, {"too_noisy": False}, True, "confidence"),
        ({"too_noisy": False}, {"too_noisy": False}, False, "pass"),
    ],
)
def test_review_decision_handles_legacy_confidence_and_success(
    cv_flags: dict, legacy_flags: dict, needs_review: bool, expected: str
) -> None:
    result = review_decision({"flags": cv_flags}, {"flags": legacy_flags}, needs_review)
    assert result["status"] == expected
