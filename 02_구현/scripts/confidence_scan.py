from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from defect_cls.data import DefectDataset, eval_transform
from defect_cls.model import load_checkpoint
from defect_cls.train import read_manifest, resolve_device
from defect_cls.paths import PROCESSED_DATA_ROOT, resolve_project_path


def confidence_stats(values: list[float]) -> dict:
    ordered = sorted(values)

    def percentile(q: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round(q / 100 * (len(ordered) - 1)))))
        return ordered[index]

    return {
        "min": round(ordered[0], 6),
        "p1": round(percentile(1), 6),
        "p5": round(percentile(5), 6),
        "p50": round(percentile(50), 6),
        "mean": round(statistics.mean(ordered), 6),
        "below_0.60": sum(1 for value in ordered if value < 0.60),
    }


def scan_confidence(model, loader, device) -> tuple[list[float], int]:
    model.eval()
    values = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidence, preds = probs.max(dim=1)
            values.extend(confidence.cpu().tolist())
            correct += (preds.cpu() == labels).sum().item()
            total += labels.size(0)
    return values, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-set confidence scan for threshold evidence")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROCESSED_DATA_ROOT / "manifest.csv")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint_path = resolve_project_path(args.checkpoint)
    manifest_path = resolve_project_path(args.manifest)
    model, checkpoint = load_checkpoint(checkpoint_path, map_location=str(device))
    model.to(device)
    image_mode = checkpoint.get("config", {}).get(
        "image_mode", "L" if checkpoint.get("in_channels") == 1 else "RGB"
    )
    rows = read_manifest(manifest_path)
    dataset = DefectDataset(
        rows,
        args.split,
        eval_transform(image_mode=image_mode),
        image_mode=image_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    values, accuracy = scan_confidence(model, loader, device)
    result = {
        "checkpoint": str(checkpoint_path),
        "experiment": checkpoint["experiment"],
        "split": args.split,
        "num_samples": len(values),
        "accuracy": round(accuracy, 4),
        "confidence": confidence_stats(values),
        "current_threshold": 0.60,
        "torch": torch.__version__,
        "device": str(device),
    }
    out_path = checkpoint_path.parent / f"confidence-scan-{args.split}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["confidence"], ensure_ascii=False))
    print(f"accuracy={accuracy:.4f} n={len(values)} saved: {out_path}")


if __name__ == "__main__":
    main()
