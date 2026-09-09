from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from defect_cls.data import CLASS_TO_INDEX, eval_transform
from defect_cls.model import load_checkpoint
from defect_cls.perturbation import apply_perturbation
from defect_cls.quality import sharpness_score, estimate_noise_sigma, mean_brightness
from defect_cls.train import read_manifest
from defect_cls.paths import PROCESSED_DATA_ROOT, resolve_project_path, resolve_raw_path

SCAN_CONFIGS = (
    ("clean", "clean", 0.0),
    ("noise_sigma15", "noise", 15.0),
    ("noise_sigma30", "noise", 30.0),
    ("blur_r1", "blur", 1.0),
    ("blur_r3", "blur", 3.0),
    ("brightness_0.6", "brightness", 0.6),
    ("brightness_1.4", "brightness", 1.4),
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q / 100 * (len(ordered) - 1)))))
    return ordered[index]


def summarize(values: list[float]) -> dict:
    return {
        "min": round(min(values), 4),
        "p5": round(percentile(values, 5), 4),
        "p50": round(percentile(values, 50), 4),
        "p95": round(percentile(values, 95), 4),
        "max": round(max(values), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure image-quality metrics on val set per perturbation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROCESSED_DATA_ROOT / "manifest.csv")
    args = parser.parse_args()

    manifest_path = resolve_project_path(args.manifest)
    checkpoint_path = resolve_project_path(args.checkpoint)
    rows = [row for row in read_manifest(manifest_path) if row["split"] == "val"]
    image_paths = [row["filepath"] for row in rows]

    per_config = {}
    for name, kind, level in SCAN_CONFIGS:
        sharpness_list = []
        noise_list = []
        brightness_list = []
        for path in image_paths:
            image = apply_perturbation(Image.open(resolve_raw_path(path)), kind, level)
            sharpness_list.append(sharpness_score(image))
            noise_list.append(estimate_noise_sigma(image))
            brightness_list.append(mean_brightness(image))
        per_config[name] = {
            "kind": kind,
            "level": level,
            "num_samples": len(image_paths),
            "sharpness": summarize(sharpness_list),
            "noise_sigma": summarize(noise_list),
            "brightness": summarize(brightness_list),
        }
        print(
            f"{name:<16} sharpness p5/p50/p95 = {per_config[name]['sharpness']['p5']}/{per_config[name]['sharpness']['p50']}/{per_config[name]['sharpness']['p95']}"
            f" | noise p50/p95 = {per_config[name]['noise_sigma']['p50']}/{per_config[name]['noise_sigma']['p95']}"
            f" | brightness p5/p95 = {per_config[name]['brightness']['p5']}/{per_config[name]['brightness']['p95']}"
        )

    payload = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "note": "clean val 분포로 품질 게이트 임계값의 근거를 만든다. 임계값 확정 전에는 기본 상수를 신뢰하지 않는다.",
        "per_config": per_config,
    }
    out_path = checkpoint_path.parent / "quality-gate-calibration.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
