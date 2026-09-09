from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from defect_cls.data import IMAGE_SIZE
from defect_cls.inference import load_image_tensor
from defect_cls.model import load_checkpoint


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q / 100 * (len(ordered) - 1)))))
    return ordered[index]


def benchmark_device(model, device: torch.device, tensor: torch.Tensor, warmup: int, iters: int) -> dict:
    model = model.to(device)
    tensor = tensor.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            model(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "warmup": warmup,
        "iters": iters,
        "p50_ms": round(percentile(times, 50), 3),
        "p95_ms": round(percentile(times, 95), 3),
        "mean_ms": round(statistics.mean(times), 3),
        "stdev_ms": round(statistics.stdev(times), 3) if len(times) > 1 else 0.0,
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
    }


def check_overwrite(out_path: Path, force: bool) -> str:
    if not out_path.exists():
        return "write"
    if force:
        return "overwrite"
    return "blocked"


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-image inference benchmark")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=None, help="optional real image input")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--devices", default="cpu,cuda")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="allow overwriting an existing benchmark file")
    args = parser.parse_args()

    model, checkpoint = load_checkpoint(args.checkpoint)
    image_mode = checkpoint.get("config", {}).get(
        "image_mode", "L" if checkpoint.get("in_channels") == 1 else "RGB"
    )
    input_channels = checkpoint.get("in_channels", 1 if image_mode == "L" else 3)
    if args.image is not None:
        tensor = load_image_tensor(str(args.image), image_mode=image_mode)
    else:
        tensor = torch.randn(1, input_channels, IMAGE_SIZE, IMAGE_SIZE)

    requested = [token.strip().lower() for token in args.devices.split(",") if token.strip()]
    available = {"cpu": torch.device("cpu")}
    if torch.cuda.is_available():
        available["cuda"] = torch.device("cuda")
    targets = [available[name] for name in requested if name in available]

    results = {
        "checkpoint": str(args.checkpoint),
        "experiment": checkpoint["experiment"],
        "input": str(args.image) if args.image is not None else f"random tensor 1x{input_channels}x{IMAGE_SIZE}x{IMAGE_SIZE}",
        "batch_size": 1,
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "python": platform.python_version(),
        "os": platform.platform(),
        "threads": torch.get_num_threads(),
        "measurements": [benchmark_device(model, device, tensor, args.warmup, args.iters) for device in targets],
    }
    out_path = args.out or args.checkpoint.parent / "benchmark.json"
    decision = check_overwrite(out_path, args.force)
    if decision == "blocked":
        raise SystemExit(
            f"덮어쓰기 방지: {out_path} 이 이미 존재한다. --force로 갱신하거나 "
            "--out으로 다른 경로를 지정하라. (기존 산출물 보호)"
        )
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    for measurement in results["measurements"]:
        print(
            f"[{measurement['device']}] p50={measurement['p50_ms']}ms p95={measurement['p95_ms']}ms "
            f"mean={measurement['mean_ms']}ms (n={measurement['iters']})"
        )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
