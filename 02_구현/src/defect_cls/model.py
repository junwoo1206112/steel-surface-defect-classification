from __future__ import annotations

from pathlib import Path

import torch
from torch.torch_version import TorchVersion
from torchvision import models

from defect_cls.data import CLASSES


def build_model(num_classes: int = len(CLASSES), pretrained: bool = True, in_channels: int = 3):
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.resnet18(weights=weights)
    if in_channels == 1:
        conv1 = torch.nn.Conv2d(
            1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            with torch.no_grad():
                conv1.weight.copy_(model.conv1.weight.mean(dim=1, keepdim=True))
        model.conv1 = conv1
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(path: str | Path, map_location: str = "cpu"):
    try:
        # Legacy project checkpoints stored ``torch.__version__`` as a TorchVersion
        # instance. Keep weights-only deserialization enabled and scope the sole
        # compatibility allowlist entry to this load operation.
        with torch.serialization.safe_globals([TorchVersion]):
            checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise ValueError(f"checkpoint could not be safely loaded: {path}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    if checkpoint.get("classes") != list(CLASSES):
        raise ValueError("checkpoint classes do not match the supported class contract")
    if checkpoint.get("in_channels", 3) not in {1, 3}:
        raise ValueError("checkpoint in_channels must be 1 or 3")
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("checkpoint is missing a state_dict")
    model = build_model(
        num_classes=len(checkpoint["classes"]),
        pretrained=False,
        in_channels=checkpoint.get("in_channels", 3),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
