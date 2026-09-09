from __future__ import annotations

from pathlib import Path

import torch

from defect_cls.data import IMAGE_SIZE
from defect_cls.model import load_checkpoint
from defect_cls.paths import resolve_project_path


def run_cpu_smoke_test(checkpoint_path: str | Path) -> dict:
    """Load a checkpoint safely and execute one CPU forward pass without data access."""
    path = resolve_project_path(checkpoint_path)
    model, checkpoint = load_checkpoint(path, map_location="cpu")
    channels = checkpoint.get("in_channels", 3)
    with torch.no_grad():
        logits = model(torch.zeros(1, channels, IMAGE_SIZE, IMAGE_SIZE))
    if tuple(logits.shape) != (1, len(checkpoint["classes"])):
        raise RuntimeError(f"unexpected CPU smoke output shape: {tuple(logits.shape)}")
    return {
        "checkpoint": path.as_posix(),
        "device": "cpu",
        "input_shape": [1, channels, IMAGE_SIZE, IMAGE_SIZE],
        "output_shape": list(logits.shape),
    }
