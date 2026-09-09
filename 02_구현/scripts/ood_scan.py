from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from defect_cls.data import CLASSES, CLASS_TO_INDEX, eval_transform
from defect_cls.model import load_checkpoint
from defect_cls.paths import PROCESSED_DATA_ROOT, resolve_project_path, resolve_raw_path
from defect_cls.perturbation import apply_perturbation
from defect_cls.train import read_manifest, resolve_device

SCAN_CONFIGS = (
    ("clean", "clean", 0.0),
    ("noise_sigma15", "noise", 15.0),
    ("noise_sigma30", "noise", 30.0),
    ("blur_r1", "blur", 1.0),
    ("blur_r3", "blur", 3.0),
    ("brightness_0.6", "brightness", 0.6),
    ("brightness_1.4", "brightness", 1.4),
)


class PerturbedValSet(Dataset):
    def __init__(self, rows: list[dict], kind: str, level: float, transform, image_mode: str = "RGB") -> None:
        self.rows = [row for row in rows if row["split"] == "val"]
        if not self.rows:
            raise ValueError("no val rows")
        self.kind = kind
        self.level = level
        self.transform = transform
        self.image_mode = image_mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = apply_perturbation(Image.open(resolve_raw_path(row["filepath"])), self.kind, self.level)
        image = image.convert(self.image_mode)
        return self.transform(image), CLASS_TO_INDEX[row["class_label"]]


def scan(model, loader, device, temperature: float = 1.0):
    model.eval()
    confidences = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs / temperature, dim=1)
            confidence, preds = probs.max(dim=1)
            confidences.extend(confidence.cpu().tolist())
            correct += (preds.cpu() == labels).sum().item()
            total += labels.size(0)
    return {
        "accuracy": round(correct / total, 4),
        "num_samples": total,
        "mean_confidence": round(statistics.mean(confidences), 6),
        "min_confidence": round(min(confidences), 6),
        "p5_confidence": round(sorted(confidences)[max(0, int(round(0.05 * (len(confidences) - 1))))], 6),
        "below_threshold_0.60": sum(1 for value in confidences if value < 0.60),
        "below_threshold_ratio": round(sum(1 for value in confidences if value < 0.60) / total, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OOD-style robustness scan on validation set")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROCESSED_DATA_ROOT / "manifest.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0, help="softmax temperature for confidence")
    parser.add_argument("--out", type=Path, default=None, help="output json path (default: ood-robustness.json next to checkpoint)")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint_path = resolve_project_path(args.checkpoint)
    manifest_path = resolve_project_path(args.manifest)
    model, checkpoint = load_checkpoint(checkpoint_path, map_location=str(device))
    model.to(device)
    image_mode = checkpoint.get("config", {}).get("image_mode", "RGB")
    transform = eval_transform(image_mode=image_mode)
    rows = read_manifest(manifest_path)

    results = []
    for name, kind, level in SCAN_CONFIGS:
        dataset = PerturbedValSet(rows, kind, level, transform, image_mode=image_mode)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        stats = scan(model, loader, device, temperature=args.temperature)
        stats["config"] = name
        stats["kind"] = kind
        stats["level"] = level
        results.append(stats)
        print(
            f"{name:<16} acc={stats['accuracy']:.4f} mean_conf={stats['mean_confidence']:.6f} "
            f"min_conf={stats['min_confidence']:.6f} below_0.60={stats['below_threshold_0.60']}/{stats['num_samples']}"
        )

    payload = {
        "checkpoint": str(checkpoint_path),
        "experiment": checkpoint["experiment"],
        "split": "val",
        "threshold": 0.60,
        "temperature": args.temperature,
        "note": "NEU-CLS val 이미지에 결정론적 변형만 적용 (외부 이미지 미사용, 노이즈 시드 고정)",
        "torch": torch.__version__,
        "device": str(device),
        "results": results,
    }
    out_path = resolve_project_path(args.out) if args.out else checkpoint_path.parent / "ood-robustness.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
