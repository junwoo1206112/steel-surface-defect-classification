from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn.metrics as metrics
import torch
from torch.utils.data import DataLoader

from defect_cls.data import (
    AUGMENT_PIPELINES,
    CLASS_TO_INDEX,
    DefectDataset,
    eval_transform,
    seed_everything,
    train_transform,
)
from defect_cls.model import build_model
from defect_cls.paths import ARTIFACTS_ROOT, PROCESSED_DATA_ROOT, resolve_project_path, seed_artifact_dir
from defect_cls.seed_metrics import sha256_file


def read_manifest(manifest_path: Path) -> list[dict]:
    import csv

    with open(manifest_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["class_index"] = int(row["class_index"])
    return rows


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_preds = []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_targets.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "macro_f1": metrics.f1_score(all_targets, all_preds, average="macro", zero_division=0),
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def check_training_output(out_dir: Path, overwrite: bool) -> None:
    """Protect completed seed artifacts from accidental replacement."""
    existing = [path.name for path in (out_dir / "checkpoint.pt", out_dir / "history.json") if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"training artifacts already exist in {out_dir}: {', '.join(existing)}; "
            "use --overwrite only after preserving or reviewing the prior run"
        )

    parser = argparse.ArgumentParser(description="Train defect classification model")
def main() -> None:
    parser = argparse.ArgumentParser(description="Train defect classification model")
    parser.add_argument("--manifest", type=Path, default=PROCESSED_DATA_ROOT / "manifest.csv")
    parser.add_argument("--experiment", choices=["baseline", "augmented", "grayscale1ch"], required=True)
    parser.add_argument("--input-channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="replace existing checkpoint/history for this experiment and seed")
    args = parser.parse_args()

    augmented = args.experiment == "augmented"
    if args.experiment == "grayscale1ch" and args.input_channels != 1:
        raise SystemExit("grayscale1ch experiment requires --input-channels 1")
    image_mode = "L" if args.input_channels == 1 else "RGB"
    seed_everything(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = resolve_device(args.device)
    manifest_path = resolve_project_path(args.manifest)
    out_dir = seed_artifact_dir(args.experiment, args.seed, args.out_dir)
    check_training_output(out_dir, args.overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(manifest_path)
    train_set = DefectDataset(rows, "train", train_transform(augmented=augmented, image_mode=image_mode), image_mode=image_mode)
    val_set = DefectDataset(rows, "val", eval_transform(image_mode=image_mode), image_mode=image_mode)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(in_channels=args.input_channels).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_stats = run_epoch(model, train_loader, device, criterion, optimizer)
        val_stats = run_epoch(model, val_loader, device, criterion)
        scheduler.step()
        elapsed = time.time() - start
        record = {
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
            "elapsed_seconds": round(elapsed, 3),
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(record)
        print(
            f"[{args.experiment}] epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['accuracy']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['accuracy']:.4f} "
            f"val_macro_f1={val_stats['macro_f1']:.4f} ({elapsed:.1f}s)"
        )
        if val_stats["macro_f1"] > best_val_f1:
            best_val_f1 = val_stats["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": list(CLASS_TO_INDEX.keys()),
                    "experiment": args.experiment,
                    "in_channels": args.input_channels,
                    "epoch": epoch,
                    "val_macro_f1": best_val_f1,
                    "seed": args.seed,
                    "config": {
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "optimizer": "SGD(momentum=0.9)",
                        "scheduler": "CosineAnnealingLR",
                        "augment_pipeline": AUGMENT_PIPELINES[args.experiment],
                        "input_size": 200,
                        "input_channels": args.input_channels,
                        "image_mode": image_mode,
                        "manifest": str(manifest_path),
                        "manifest_sha256": sha256_file(manifest_path),
                    },
                    "environment": {
                        "torch": torch.__version__,
                        "cuda_available": torch.cuda.is_available(),
                        "cuda_version": torch.version.cuda,
                        "device": str(device),
                        "device_name": torch.cuda.get_device_name(0)
                        if device.type == "cuda"
                        else platform.processor(),
                        "python": platform.python_version(),
                        "os": platform.platform(),
                    },
                },
                out_dir / "checkpoint.pt",
            )

    (out_dir / "history.json").write_text(
        json.dumps(
            {
                "experiment": args.experiment,
                "seed": args.seed,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val_f1,
                "history": history,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[{args.experiment}] best epoch {best_epoch} val_macro_f1={best_val_f1:.4f}")
    print(f"[{args.experiment}] artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
