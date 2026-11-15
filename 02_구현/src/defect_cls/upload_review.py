"""Pure upload inspection and review-priority helpers for the Streamlit demo."""

from __future__ import annotations

import io

from PIL import Image

from defect_cls.inference import load_image_tensor, validate_upload
from defect_cls.opencv_pipeline import assess_opencv_quality, decode_and_validate_image
from defect_cls.quality import assess_quality


def inspect_upload(image_bytes: bytes, filename: str, image_mode: str) -> dict:
    """Validate once and return independent model and OpenCV diagnostic inputs.

    The tensor still comes from the established PIL model-input route.  OpenCV
    data is returned only for diagnostics and visualization.
    """
    ok, message = validate_upload(filename, len(image_bytes))
    if not ok:
        raise ValueError(message)
    bgr, metadata = decode_and_validate_image(image_bytes, filename)
    tensor = load_image_tensor(image_bytes, image_mode=image_mode)
    with Image.open(io.BytesIO(image_bytes)) as image:
        legacy_quality = assess_quality(image)
    cv_quality = assess_opencv_quality(bgr)
    return {
        "tensor": tensor,
        "bgr": bgr,
        "metadata": metadata,
        "legacy_quality": legacy_quality,
        "cv_quality": cv_quality,
    }


def review_decision(cv_quality: dict, legacy_quality: dict, needs_review: bool) -> dict:
    """Apply the demo's explicit quality-before-confidence review precedence."""
    flagged = [f"OpenCV:{name}" for name, value in cv_quality["flags"].items() if value]
    flagged.extend(
        f"기존 게이트:{name}" for name, value in legacy_quality["flags"].items() if value
    )
    if flagged:
        return {"status": "quality", "flagged": flagged}
    if needs_review:
        return {"status": "confidence", "flagged": []}
    return {"status": "pass", "flagged": []}
