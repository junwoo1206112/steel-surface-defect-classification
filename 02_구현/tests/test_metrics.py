from __future__ import annotations

import pytest

from defect_cls.evaluate import build_metrics, validate_checkpoint_manifest
from defect_cls.preparation import (
    dedupe_entries,
    find_duplicates,
    find_near_duplicate_candidates,
    inspect_entries,
)


def test_build_metrics_known_values() -> None:
    targets = [0, 0, 1, 1]
    preds = [0, 1, 1, 1]
    result = build_metrics(targets, preds, ["a", "b"])
    assert result["accuracy"] == 0.75
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert set(result["per_class"]) == {"a", "b"}
    assert result["num_samples"] == 4


def test_build_metrics_preserves_unrounded_raw_floats() -> None:
    result = build_metrics([0, 0, 0, 1, 1], [0, 0, 1, 1, 1], ["a", "b"])
    assert result["accuracy"] == 0.8
    assert result["macro_f1"] == 0.8
    assert result["per_class"]["a"]["recall"] == pytest.approx(2 / 3)


def test_build_metrics_perfect_predictions() -> None:
    targets = [0, 1, 2, 2, 1]
    preds = [0, 1, 2, 2, 1]
    result = build_metrics(targets, preds, ["x", "y", "z"])
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0


def test_find_duplicates_detects_same_content(tmp_path) -> None:
    from PIL import Image

    first = tmp_path / "crazing_1.jpg"
    second = tmp_path / "crazing_2.jpg"
    third = tmp_path / "inclusion_1.jpg"
    Image.new("L", (8, 8), color=128).save(first)
    Image.new("L", (8, 8), color=128).save(second)
    Image.new("L", (8, 8), color=64).save(third)
    entries, _ = inspect_entries([first, second, third])
    duplicates = find_duplicates(entries)
    assert len(duplicates) == 1
    assert set(duplicates[0]["files"]) == {"crazing_1.jpg", "crazing_2.jpg"}

    kept, excluded = dedupe_entries(entries)
    assert len(kept) == 2
    assert len(excluded) == 1
    assert excluded[0]["kept"] == "crazing_1.jpg"
    assert excluded[0]["excluded"] == "crazing_2.jpg"


def test_find_near_duplicate_candidates_reports_similar_images(synthetic_raw_dir) -> None:
    from PIL import Image

    first = synthetic_raw_dir / "crazing_101.bmp"
    second = synthetic_raw_dir / "crazing_102.bmp"
    image = Image.new("L", (16, 16), color=0)
    for x in range(8, 16):
        for y in range(16):
            image.putpixel((x, y), 255)
    image.save(first)
    image.save(second)
    entries, _ = inspect_entries([first, second])
    candidates = find_near_duplicate_candidates(entries)
    assert candidates == [
        {
            "left": "crazing_101.bmp",
            "right": "crazing_102.bmp",
            "left_class": "crazing",
            "right_class": "crazing",
            "hamming_distance": 0,
        }
    ]


def test_evaluation_requires_the_training_manifest_hash(tmp_path) -> None:
    from defect_cls.seed_metrics import sha256_file

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("filepath,split\ndata/raw/example.bmp,test\n", encoding="utf-8")
    checkpoint = {"config": {"manifest_sha256": sha256_file(manifest)}}
    assert validate_checkpoint_manifest(checkpoint, manifest) == sha256_file(manifest)

    other_manifest = tmp_path / "other-manifest.csv"
    other_manifest.write_text("filepath,split\ndata/raw/example.bmp,train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        validate_checkpoint_manifest(checkpoint, other_manifest)
