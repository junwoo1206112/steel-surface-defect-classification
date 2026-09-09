from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageFilter

from defect_cls.perturbation import apply_perturbation
from defect_cls.quality import (
    DEFAULT_QUALITY_THRESHOLDS,
    assess_quality,
    estimate_noise_sigma,
    mean_brightness,
    sharpness_score,
)


@pytest.fixture
def sharp_image() -> Image.Image:
    rng = np.random.default_rng(42)
    return Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8))


def test_sharpness_blur_reduces_score(sharp_image: Image.Image) -> None:
    blurred = sharp_image.filter(ImageFilter.GaussianBlur(radius=2))
    assert sharpness_score(blurred) < sharpness_score(sharp_image)


def test_noise_sigma_detects_noise() -> None:
    clean_sigma = estimate_noise_sigma(Image.new("L", (64, 64), color=128))
    noisy = apply_perturbation(Image.new("L", (64, 64), color=128), "noise", 25.0)
    assert clean_sigma < 1.0
    assert estimate_noise_sigma(noisy) > clean_sigma


def test_mean_brightness_extremes() -> None:
    assert mean_brightness(Image.new("L", (8, 8), color=0)) == pytest.approx(0.0)
    assert mean_brightness(Image.new("L", (8, 8), color=255)) == pytest.approx(255.0)


def test_assess_quality_flags_blurry(sharp_image: Image.Image) -> None:
    result = assess_quality(sharp_image, thresholds={**DEFAULT_QUALITY_THRESHOLDS, "min_sharpness": 1e9})
    assert result["flags"]["too_blurry"] is True
    assert result["passed"] is False


def test_assess_quality_flags_noisy(sharp_image: Image.Image) -> None:
    result = assess_quality(sharp_image, thresholds={**DEFAULT_QUALITY_THRESHOLDS, "max_noise_sigma": 0.0})
    assert result["flags"]["too_noisy"] is True


def test_assess_quality_passes_clean() -> None:
    image = Image.new("L", (64, 64), color=128)
    result = assess_quality(image)
    assert result["passed"] is True
    assert result["thresholds"] == DEFAULT_QUALITY_THRESHOLDS


def test_assess_quality_flags_extreme_brightness() -> None:
    dark = assess_quality(Image.new("L", (8, 8), color=5))
    assert dark["flags"]["extreme_brightness"] is True
