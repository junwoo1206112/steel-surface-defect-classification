from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from defect_cls.perturbation import apply_perturbation


@pytest.fixture
def sample_image() -> Image.Image:
    return Image.new("L", (64, 64), color=128)


def test_clean_returns_rgb(sample_image: Image.Image) -> None:
    result = apply_perturbation(sample_image, "clean", 0.0)
    assert result.mode == "RGB"
    assert result.size == (64, 64)


def test_noise_changes_pixels_deterministically(sample_image: Image.Image) -> None:
    first = np.asarray(apply_perturbation(sample_image, "noise", 25.0))
    second = np.asarray(apply_perturbation(sample_image, "noise", 25.0))
    assert np.array_equal(first, second)
    assert first.shape == (64, 64, 3)
    assert not np.array_equal(first, np.full((64, 64, 3), 128, dtype=np.uint8))


def test_blur_and_brightness_preserve_mode(sample_image: Image.Image) -> None:
    for kind, level in (("blur", 1.0), ("brightness", 0.6)):
        result = apply_perturbation(sample_image, kind, level)
        assert result.mode == "RGB"
        assert result.size == (64, 64)


def test_unknown_kind_raises(sample_image: Image.Image) -> None:
    with pytest.raises(ValueError):
        apply_perturbation(sample_image, "unknown", 1.0)
