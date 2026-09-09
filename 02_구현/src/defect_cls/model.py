from __future__ import annotations

from pathlib import Path

import torch
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
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(
        num_classes=len(checkpoint["classes"]),
        pretrained=False,
        in_channels=checkpoint.get("in_channels", 3),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
