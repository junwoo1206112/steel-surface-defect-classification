from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import sklearn.metrics as metrics
import torch
from torch.utils.data import DataLoader

from defect_cls.data import DefectDataset, eval_transform
from defect_cls.model import load_checkpoint
from defect_cls.paths import PROCESSED_DATA_ROOT, resolve_project_path
from defect_cls.seed_metrics import sha256_file
from defect_cls.train import read_manifest, resolve_device


def validate_checkpoint_manifest(checkpoint: dict, manifest_path: Path) -> str:
    """Require evaluation data to be the exact manifest used for training."""
    expected = checkpoint.get("config", {}).get("manifest_sha256")
    if not isinstance(expected, str) or not expected:
        raise ValueError("checkpoint is missing manifest_sha256 provenance")
    actual = sha256_file(manifest_path)
    if actual != expected:
        raise ValueError(
            "checkpoint manifest SHA-256 does not match the evaluation manifest; "
            "refusing to produce non-comparable test metrics"
        )
    return actual


def compute_predictions(model, loader, device):
    model.eval()
    targets = []
    preds = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds.extend(outputs.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    return targets, preds


def build_metrics(targets: list[int], preds: list[int], class_names: list[str]) -> dict:
    report = metrics.precision_recall_fscore_support(
        targets, preds, labels=list(range(len(class_names))), zero_division=0
    )
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(report[0][i]),
            "recall": float(report[1][i]),
            "f1": float(report[2][i]),
            "support": int(report[3][i]),
        }
    confusion = metrics.confusion_matrix(
        targets, preds, labels=list(range(len(class_names)))
    )
    return {
        "accuracy": float(metrics.accuracy_score(targets, preds)),
        "macro_f1": float(metrics.f1_score(targets, preds, average="macro", zero_division=0)),
        "macro_precision": float(metrics.precision_score(targets, preds, average="macro", zero_division=0)),
        "macro_recall": float(metrics.recall_score(targets, preds, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "class_order": class_names,
        "num_samples": len(targets),
    }


def save_confusion_png(confusion: list[list[int]], class_names: list[str], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.array(confusion)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion Matrix (test)")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", fontsize=8)
    figure.colorbar(image)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on the test split")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROCESSED_DATA_ROOT / "manifest.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint_path = resolve_project_path(args.checkpoint)
    manifest_path = resolve_project_path(args.manifest)
    model, checkpoint = load_checkpoint(checkpoint_path, map_location=str(device))
    model.to(device)
    image_mode = checkpoint.get("config", {}).get("image_mode", "RGB")
    manifest_sha256 = validate_checkpoint_manifest(checkpoint, manifest_path)
    rows = read_manifest(manifest_path)
    test_set = DefectDataset(rows, "test", eval_transform(image_mode=image_mode), image_mode=image_mode)
    loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    targets, preds = compute_predictions(model, loader, device)
    result = build_metrics(targets, preds, checkpoint["classes"])
    result["checkpoint"] = str(checkpoint_path)
    result["manifest"] = str(manifest_path)
    result["manifest_sha256"] = manifest_sha256
    result["experiment"] = checkpoint["experiment"]
    result["selected_epoch"] = checkpoint["epoch"]
    result["seed"] = checkpoint["seed"]
    result["config"] = checkpoint["config"]
    result["environment"] = checkpoint["environment"]
    result["eval_device"] = str(device)
    result["eval_time_torch"] = torch.__version__
    result["split"] = "test"

    out_path = checkpoint_path.parent / "test-metrics.json"
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    csv_path = checkpoint_path.parent / "confusion-matrix.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred"] + result["class_order"])
        for name, row_values in zip(result["class_order"], result["confusion_matrix"]):
            writer.writerow([name] + row_values)
    save_confusion_png(
        result["confusion_matrix"],
        result["class_order"],
        checkpoint_path.parent / "confusion-matrix.png",
    )
    print(
        f"[{result['experiment']}] test accuracy={result['accuracy']:.4f} "
        f"macro_f1={result['macro_f1']:.4f} n={result['num_samples']}"
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
