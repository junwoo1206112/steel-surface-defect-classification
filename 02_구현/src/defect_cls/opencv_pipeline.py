"""OpenCV diagnostics for uploaded images.

This module is deliberately separate from the PyTorch input transform.  Its
outputs are diagnostic and visualization aids only; they never alter the model
tensor or produce a defect location prediction.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from defect_cls.inference import ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES, MAX_UPLOAD_PIXELS


class OpenCVValidationError(ValueError):
    """An uploaded image does not meet the demo input contract."""


# These are display-only heuristics, not calibrated operating limits.
DEFAULT_OPENCV_QUALITY_THRESHOLDS = {
    "min_brightness": 20.0,
    "max_brightness": 235.0,
    "min_contrast": 12.0,
    "min_sharpness": 10.0,
    "max_noise_sigma": 20.0,
}


def decode_and_validate_image(image_bytes: bytes, filename: str) -> tuple[np.ndarray, dict]:
    """Validate image headers before OpenCV allocation, then decode for diagnostics."""
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise OpenCVValidationError(
            f"지원하지 않는 형식입니다: {extension or '(확장자 없음)'} — 허용: jpg, jpeg, png, bmp"
        )
    if not image_bytes:
        raise OpenCVValidationError("비어 있는 파일입니다.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise OpenCVValidationError(
            f"파일이 너무 큽니다: {len(image_bytes)} bytes (최대 {MAX_UPLOAD_BYTES} bytes)"
        )

    # PIL reads the container header without materializing decoded pixels.  Keep
    # this gate ahead of cv2.imdecode so a highly-compressed image cannot force
    # a large OpenCV allocation before the pixel ceiling is enforced.
    try:
        with Image.open(io.BytesIO(image_bytes)) as header:
            header_width, header_height = header.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise OpenCVValidationError("손상되었거나 이미지가 아닌 파일입니다.") from exc
    if header_width <= 0 or header_height <= 0:
        raise OpenCVValidationError("이미지 크기가 올바르지 않습니다.")
    if header_width * header_height > MAX_UPLOAD_PIXELS:
        raise OpenCVValidationError(
            f"이미지 픽셀이 너무 많습니다: {header_width}x{header_height} "
            f"(최대 {MAX_UPLOAD_PIXELS})"
        )

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    try:
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise OpenCVValidationError(f"OpenCV 디코딩 실패: {exc}") from exc
    if bgr is None or bgr.size == 0:
        raise OpenCVValidationError("손상되었거나 이미지가 아닌 파일입니다.")

    height, width = bgr.shape[:2]
    if width <= 0 or height <= 0:
        raise OpenCVValidationError("이미지 크기가 올바르지 않습니다.")
    if width * height > MAX_UPLOAD_PIXELS:
        raise OpenCVValidationError(
            f"이미지 픽셀이 너무 많습니다: {width}x{height} (최대 {MAX_UPLOAD_PIXELS})"
        )
    if (width, height) != (header_width, header_height):
        raise OpenCVValidationError("이미지 헤더와 디코딩 크기가 일치하지 않습니다.")
    return bgr, {"width": width, "height": height, "channels": int(bgr.shape[2]), "extension": extension}


def _gray(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("expected a three-channel BGR image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def assess_opencv_quality(bgr: np.ndarray, thresholds: dict | None = None) -> dict:
    """Measure brightness, contrast, blur proxy, and robust high-frequency noise."""
    thresholds = thresholds or DEFAULT_OPENCV_QUALITY_THRESHOLDS
    gray = _gray(bgr)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    residual = gray.astype(np.float64) - cv2.medianBlur(gray, 3).astype(np.float64)
    noise_sigma = float(np.median(np.abs(residual)) / 0.6745)
    flags = {
        "extreme_brightness": brightness < thresholds["min_brightness"] or brightness > thresholds["max_brightness"],
        "low_contrast": contrast < thresholds["min_contrast"],
        "too_blurry": sharpness < thresholds["min_sharpness"],
        "too_noisy": noise_sigma > thresholds["max_noise_sigma"],
    }
    return {
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "sharpness": round(sharpness, 4),
        "noise_sigma": round(noise_sigma, 4),
        "flags": flags,
        "passed": not any(flags.values()),
        "thresholds": dict(thresholds),
    }


def preprocess_defect_candidates(bgr: np.ndarray) -> dict:
    """Create a CLAHE preview and texture-anomaly candidate overlay.

    The binary mask is a deterministic visualization cue, not ground truth,
    object detection, segmentation, or a model-produced location.
    """
    gray = _gray(bgr)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    background = cv2.GaussianBlur(enhanced, (5, 5), 0)
    residual = cv2.absdiff(enhanced, background)
    _, mask = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), dtype=np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # A 3x3 opening erases one-pixel-wide scratches and thin edges completely.
    # Keep the pre-opening mask when cleanup would remove almost all candidates,
    # so the visualization does not silently hide thin defects.
    if np.count_nonzero(opened) >= max(1, np.count_nonzero(mask) // 10):
        mask = opened
    mask = cv2.dilate(mask, kernel, iterations=1)

    overlay = bgr.copy()
    overlay[mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(bgr, 0.65, overlay, 0.35, 0)
    return {"enhanced_gray": enhanced, "candidate_mask": mask, "candidate_overlay_bgr": overlay}
