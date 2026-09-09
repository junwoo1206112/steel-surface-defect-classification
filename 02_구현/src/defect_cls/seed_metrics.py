from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

from defect_cls.paths import resolve_project_path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(resolve_project_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_seed_metrics(metric_paths: list[str | Path], manifest_path: str | Path) -> dict:
    """Aggregate completed seed runs only when they share the exact manifest bytes."""
    manifest = resolve_project_path(manifest_path)
    manifest_sha256 = sha256_file(manifest)
    runs = []
    seen_seeds = set()
    for metric_path in metric_paths:
        path = resolve_project_path(metric_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("manifest_sha256") != manifest_sha256:
            raise ValueError(f"manifest SHA-256 mismatch: {path}")
        seed = payload.get("seed")
        if not isinstance(seed, int):
            raise ValueError(f"missing integer seed: {path}")
        if seed in seen_seeds:
            raise ValueError(f"duplicate seed {seed}: {path}")
        seen_seeds.add(seed)
        metrics = {key: payload.get(key) for key in ("accuracy", "macro_f1")}
        if not all(isinstance(value, (int, float)) for value in metrics.values()):
            raise ValueError(f"missing numeric metrics: {path}")
        runs.append({"seed": seed, "metrics_path": path.as_posix(), **metrics})
    if not runs:
        raise ValueError("at least one test-metrics.json file is required")
    runs.sort(key=lambda run: run["seed"])
    summary = {}
    for metric_name in ("accuracy", "macro_f1"):
        values = [run[metric_name] for run in runs]
        summary[metric_name] = {
            "mean": round(statistics.mean(values), 6),
            "stdev": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return {
        "manifest": manifest.as_posix(),
        "manifest_sha256": manifest_sha256,
        "run_count": len(runs),
        "runs": runs,
        "summary": summary,
        "note": "Aggregates existing measured test-metrics only; it does not train or infer metrics.",
    }
