from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from defect_cls.evaluate import compute_predictions


class EvalOnlyModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert not self.training
        return torch.stack((images[:, 0], -images[:, 0]), dim=1)


def test_compute_predictions_switches_model_to_eval_mode() -> None:
    model = EvalOnlyModel()
    model.train()
    loader = DataLoader(
        TensorDataset(
            torch.tensor([[1.0], [-1.0]]),
            torch.tensor([0, 1]),
        ),
        batch_size=2,
    )

    targets, predictions = compute_predictions(model, loader, torch.device("cpu"))

    assert model.training is False
    assert targets == [0, 1]
    assert predictions == [0, 1]
