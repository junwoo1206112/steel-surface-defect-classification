from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

PERTURBATION_KINDS = ("noise", "blur", "brightness")


def apply_perturbation(image: Image.Image, kind: str, level: float, seed: int = 0) -> Image.Image:
    if kind == "clean":
        return image.convert("RGB")
    if kind not in PERTURBATION_KINDS:
        raise ValueError(f"unknown perturbation kind: {kind}")
    base = image.convert("RGB")
    if kind == "noise":
        rng = np.random.default_rng(seed)
        array = np.asarray(base, dtype=np.float32)
        array = array + rng.normal(0.0, level, array.shape)
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    if kind == "blur":
        return base.filter(ImageFilter.GaussianBlur(radius=level))
    return ImageEnhance.Brightness(base).enhance(level)
