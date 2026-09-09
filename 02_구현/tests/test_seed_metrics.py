from __future__ import annotations

import json
from pathlib import Path

import pytest

from defect_cls.seed_metrics import sha256_file, summarize_seed_metrics


def _write_metrics(path: Path, seed: int, manifest_hash: str, accuracy: float, macro_f1: float) -> None:
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "manifest_sha256": manifest_hash,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        ),
        encoding="utf-8",
    )


def test_summarize_seed_metrics_requires_the_same_manifest_hash(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("filepath,split\ndata/raw/example.bmp,test\n", encoding="utf-8")
    digest = sha256_file(manifest)
    first = tmp_path / "seed-1.json"
    second = tmp_path / "seed-2.json"
    _write_metrics(first, 1, digest, 0.9, 0.8)
    _write_metrics(second, 2, digest, 0.7, 0.6)

    result = summarize_seed_metrics([first, second], manifest)

    assert result["run_count"] == 2
    assert result["summary"]["accuracy"] == {
        "n": 2,
        "mean": 0.8,
        "sample_stddev": pytest.approx(0.14142135623730953),
        "min": 0.7,
        "max": 0.9,
    }
    assert result["summary"]["macro_f1"]["n"] == 2
    assert [run["seed"] for run in result["runs"]] == [1, 2]


def test_summarize_seed_metrics_rejects_manifest_or_seed_conflicts(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("filepath,split\ndata/raw/example.bmp,test\n", encoding="utf-8")
    first = tmp_path / "seed-1.json"
    second = tmp_path / "seed-1-repeat.json"
    _write_metrics(first, 1, "incorrect", 0.9, 0.8)
    with pytest.raises(ValueError, match="SHA-256"):
        summarize_seed_metrics([first], manifest)

    digest = sha256_file(manifest)
    _write_metrics(first, 1, digest, 0.9, 0.8)
    _write_metrics(second, 1, digest, 0.8, 0.7)
    with pytest.raises(ValueError, match="duplicate seed"):
        summarize_seed_metrics([first, second], manifest)


def test_summarize_seed_metrics_requires_two_unique_seeds(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("filepath,split\ndata/raw/example.bmp,test\n", encoding="utf-8")
    metrics_path = tmp_path / "seed-1.json"
    _write_metrics(metrics_path, 1, sha256_file(manifest), 0.9, 0.8)
    with pytest.raises(ValueError, match="two unique seeds"):
        summarize_seed_metrics([metrics_path], manifest)
