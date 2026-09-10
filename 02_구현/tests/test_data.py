from __future__ import annotations

from pathlib import Path

import pytest
import torch

from defect_cls.data import (
    AUGMENT_PIPELINES,
    CLASSES,
    DefectDataset,
    build_stratified_split,
    eval_transform,
    parse_class_from_filename,
    train_transform,
)
from defect_cls.preparation import run_preparation
from defect_cls.paths import seed_artifact_dir


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("crazing_1.jpg", "crazing"),
        ("pitted_surface_300.jpg", "pitted_surface"),
        ("rolled-in_scale_12.jpg", "rolled-in_scale"),
        ("SCRATCHES_7.PNG", "scratches"),
        ("inclusion_5.bmp", "inclusion"),
        ("Cr_1.bmp", "crazing"),
        ("In_150.bmp", "inclusion"),
        ("Pa_2.bmp", "patches"),
        ("PS_88.bmp", "pitted_surface"),
        ("RS_64.bmp", "rolled-in_scale"),
        ("Sc_97.bmp", "scratches"),
        ("XX_5.jpg", None),
        ("unknown_5.jpg", None),
        ("noindex.jpg", None),
        ("patches.jpg", None),
    ],
)
def test_parse_class_from_filename(filename: str, expected: str | None) -> None:
    assert parse_class_from_filename(filename) == expected


def test_build_stratified_split_proportions_and_disjointness(synthetic_raw_dir: Path) -> None:
    from defect_cls.preparation import collect_images, inspect_entries

    entries, _ = inspect_entries(collect_images(synthetic_raw_dir))
    rows = build_stratified_split(entries, seed=42)
    assert len(rows) == 60
    filenames = [row["filename"] for row in rows]
    assert len(filenames) == len(set(filenames))
    for cls in CLASSES:
        class_rows = [row for row in rows if row["class_label"] == cls]
        counts = {split: 0 for split in ("train", "val", "test")}
        for row in class_rows:
            counts[row["split"]] += 1
        assert counts == {"train": 7, "val": 2, "test": 1}


def test_build_stratified_split_is_deterministic(synthetic_raw_dir: Path) -> None:
    from defect_cls.preparation import collect_images, inspect_entries

    entries, _ = inspect_entries(collect_images(synthetic_raw_dir))
    first = build_stratified_split(entries, seed=42)
    second = build_stratified_split(entries, seed=42)
    assert [(row["filename"], row["split"]) for row in first] == [
        (row["filename"], row["split"]) for row in second
    ]


def test_transforms_augmented_difference() -> None:
    baseline_types = {type(t) for t in train_transform(augmented=False).transforms}
    augmented_types = {type(t) for t in train_transform(augmented=True).transforms}
    assert len(augmented_types) > len(baseline_types)
    assert type(eval_transform()) is not None
    assert set(AUGMENT_PIPELINES) == {"baseline", "augmented", "grayscale1ch"}


def test_defect_dataset_shapes(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    manifest_path, _, _ = run_preparation(synthetic_raw_dir, tmp_path / "processed", 42, (0.7, 0.15, 0.15))
    import csv

    with open(manifest_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["class_index"] = int(row["class_index"])
    dataset = DefectDataset(rows, "train", eval_transform())
    image, label = dataset[0]
    assert image.shape == (3, 200, 200)
    assert 0 <= label < len(CLASSES)


def test_defect_dataset_grayscale_1ch(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    manifest_path, _, _ = run_preparation(synthetic_raw_dir, tmp_path / "processed", 42, (0.7, 0.15, 0.15))
    import csv

    with open(manifest_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["class_index"] = int(row["class_index"])
    dataset = DefectDataset(rows, "train", eval_transform(image_mode="L"), image_mode="L")
    image, _ = dataset[0]
    assert image.shape == (1, 200, 200)


def test_manifest_filepaths_are_posix(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    manifest_path, _, _ = run_preparation(synthetic_raw_dir, tmp_path / "processed", 42, (0.7, 0.15, 0.15))
    import csv

    with open(manifest_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        assert "\\" not in row["filepath"]
        assert row["filepath"].replace("\\", "/") == row["filepath"]
        assert row["filepath"].startswith("data/raw/")


def test_preparation_rejects_input_outside_data_raw(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="data/raw"):
        run_preparation(outside, tmp_path / "processed", 42, (0.7, 0.15, 0.15))


def test_dataset_paths_are_independent_of_current_working_directory(
    synthetic_raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _, _ = run_preparation(synthetic_raw_dir, tmp_path / "processed", 42, (0.7, 0.15, 0.15))
    import csv

    with open(manifest_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    monkeypatch.chdir(tmp_path)
    dataset = DefectDataset(rows, "train", eval_transform())
    image, _ = dataset[0]
    assert image.shape == (3, 200, 200)


def test_preparation_is_independent_of_current_working_directory(
    synthetic_raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest_path, _, quality = run_preparation(
        synthetic_raw_dir, tmp_path / "processed", 42, (0.7, 0.15, 0.15)
    )
    assert manifest_path.exists()
    assert quality["near_duplicate_review"]["candidate_count"] >= 0


def test_seed_artifact_directories_do_not_collide() -> None:
    first = seed_artifact_dir("baseline", 1)
    second = seed_artifact_dir("baseline", 2)
    assert first != second
    assert first.as_posix().endswith("baseline/seed-1")


def test_zip_path_traversal_is_rejected(tmp_path: Path) -> None:
    import zipfile

    from defect_cls.preparation import extract_archive

    archive_path = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "unsafe")
    with pytest.raises(ValueError, match="unsafe ZIP member path"):
        extract_archive(archive_path, tmp_path / "raw")
    assert not (tmp_path / "outside.txt").exists()


def test_neu_contract_rejects_non_neu_fixture(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="NEU-CLS intake contract failed"):
        run_preparation(
            synthetic_raw_dir,
            tmp_path / "processed",
            42,
            (0.7, 0.15, 0.15),
            enforce_neu_contract=True,
        )


def test_training_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    from defect_cls.train import check_training_output

    out_dir = tmp_path / "baseline" / "seed-42"
    out_dir.mkdir(parents=True)
    (out_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    with pytest.raises(FileExistsError, match="--overwrite"):
        check_training_output(out_dir, overwrite=False)
    check_training_output(out_dir, overwrite=True)


def test_preparation_rejects_rar_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import defect_cls.paths as paths

    monkeypatch.setattr(paths, "RAW_DATA_ROOT", tmp_path)
    archive_path = tmp_path / "unsafe-input.rar"
    archive_path.write_bytes(b"not-a-rar")
    with pytest.raises(ValueError, match="RAR 입력"):
        run_preparation(archive_path, tmp_path / "processed", 42, (0.7, 0.15, 0.15))

