from __future__ import annotations

import torch

from defect_cls.model import build_model


def test_build_model_default_3ch() -> None:
    model = build_model(pretrained=False)
    assert model.conv1.weight.shape == (64, 3, 7, 7)
    assert model.fc.out_features == 6


def test_build_model_1ch_shape() -> None:
    model = build_model(pretrained=False, in_channels=1)
    assert model.conv1.weight.shape == (64, 1, 7, 7)


def test_build_model_1ch_pretrained_weight_averaged() -> None:
    reference = build_model(pretrained=True, in_channels=3)
    grayscale_model = build_model(pretrained=True, in_channels=1)
    expected = reference.conv1.weight.mean(dim=1)
    assert torch.allclose(grayscale_model.conv1.weight.data[:, 0], expected)


def test_build_model_1ch_forward() -> None:
    model = build_model(pretrained=False, in_channels=1)
    output = model(torch.randn(2, 1, 200, 200))
    assert output.shape == (2, 6)
