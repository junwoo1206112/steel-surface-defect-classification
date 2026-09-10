from __future__ import annotations

import torch
import pytest
from torch.torch_version import TorchVersion

from defect_cls.data import CLASSES
from defect_cls.model import build_model, load_checkpoint
from defect_cls.smoke import run_cpu_smoke_test


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


def test_safe_checkpoint_load_and_cpu_smoke_test(tmp_path) -> None:
    model = build_model(pretrained=False)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": list(CLASSES),
            "experiment": "baseline",
            "in_channels": 3,
        },
        checkpoint_path,
    )
    loaded, _ = load_checkpoint(checkpoint_path)
    assert loaded.fc.out_features == len(CLASSES)
    result = run_cpu_smoke_test(checkpoint_path)
    assert result["device"] == "cpu"
    assert result["output_shape"] == [1, len(CLASSES)]


def test_safe_checkpoint_load_supports_legacy_torch_version_metadata(tmp_path) -> None:
    model = build_model(pretrained=False)
    checkpoint_path = tmp_path / "legacy-checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": list(CLASSES),
            "experiment": "baseline",
            "in_channels": 3,
            "environment": {"torch": TorchVersion(torch.__version__)},
        },
        checkpoint_path,
    )

    _, checkpoint = load_checkpoint(checkpoint_path)

    assert checkpoint["environment"]["torch"] == torch.__version__

def test_safe_checkpoint_load_rejects_wrong_class_contract(tmp_path) -> None:
    checkpoint_path = tmp_path / "bad.pt"
    torch.save({"state_dict": {}, "classes": ["wrong"], "in_channels": 3}, checkpoint_path)
    with pytest.raises(ValueError, match="classes"):
        load_checkpoint(checkpoint_path)
