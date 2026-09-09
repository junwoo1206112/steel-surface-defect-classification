from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from defect_cls.data import DefectDataset, eval_transform
from defect_cls.model import build_model, load_checkpoint
from defect_cls.train import read_manifest, resolve_device
from defect_cls.paths import PROCESSED_DATA_ROOT, resolve_project_path

T_GRID = [round(0.1 * t, 1) for t in range(5, 51)]


def collect_logits(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_all = []
    labels_all = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits_all.append(model(images).cpu())
            labels_all.append(labels)
    return torch.cat(logits_all), torch.cat(labels_all)


def nll_at_temperature(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> float:
    scaled = logits / temperature
    log_probs = torch.log_softmax(scaled, dim=1)
    return float(torch.nn.functional.nll_loss(log_probs, labels).item())


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    losses = [(t, nll_at_temperature(logits, labels, t)) for t in T_GRID]
    best_temperature, best_loss = min(losses, key=lambda pair: pair[1])
    baseline_loss = nll_at_temperature(logits, labels, 1.0)
    return {
        "grid": T_GRID,
        "best_temperature": best_temperature,
        "best_val_nll": round(best_loss, 6),
        "nll_at_T1": round(baseline_loss, 6),
        "nll_improvement": round(baseline_loss - best_loss, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit temperature scaling on validation logits")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROCESSED_DATA_ROOT / "manifest.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint_path = resolve_project_path(args.checkpoint)
    manifest_path = resolve_project_path(args.manifest)
    model, checkpoint = load_checkpoint(checkpoint_path, map_location=str(device))
    model.to(device)
    image_mode = checkpoint.get("config", {}).get("image_mode", "RGB")
    rows = read_manifest(manifest_path)
    dataset = DefectDataset(rows, "val", eval_transform(image_mode=image_mode), image_mode=image_mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logits, labels = collect_logits(model, loader, device)
    result = fit_temperature(logits, labels)
    payload = {
        "checkpoint": str(checkpoint_path),
        "experiment": checkpoint["experiment"],
        "split": "val",
        "num_samples": int(labels.numel()),
        "torch": torch.__version__,
        "device": str(device),
        **result,
    }
    out_path = checkpoint_path.parent / "temperature.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"best T={result['best_temperature']} "
        f"NLL {result['nll_at_T1']} -> {result['best_val_nll']} "
        f"(improvement {result['nll_improvement']})"
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
