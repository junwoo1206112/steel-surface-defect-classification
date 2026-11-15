from __future__ import annotations

import json
from pathlib import Path

from defect_cls.handoff import (
    EXPECTED_FOLDER_NAME,
    SCREENSHOT_EVIDENCE_RELATIVE,
    SCREENSHOT_RELATIVE,
    build_handoff_status,
    count_images,
    discover_checkpoint_files,
    docs_test_count_status,
    format_handoff_status,
    screenshot_status,
    summarize_manifest,
    reliability_metrics_for_checkpoints,
    sha256_file,
)


def _make_project(tmp_path: Path) -> Path:
    root = tmp_path / EXPECTED_FOLDER_NAME
    (root / "02_구현" / "data" / "raw").mkdir(parents=True)
    (root / "02_구현" / "data" / "processed").mkdir(parents=True)
    (root / "02_구현" / "data" / "artifacts").mkdir(parents=True)
    return root


def _write_manifest(root: Path, rows: list[dict]) -> Path:
    manifest = root / "02_구현" / "data" / "processed" / "manifest.csv"
    fieldnames = list(rows[0])
    import csv

    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_count_images_ignores_non_images(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    raw = root / "02_구현" / "data" / "raw"
    (raw / "a.bmp").write_bytes(b"x")
    (raw / "b.PNG").write_bytes(b"x")
    (raw / "notes.txt").write_text("x", encoding="utf-8")
    assert count_images(raw) == 2
    assert count_images(tmp_path / "missing") == 0


def test_summarize_manifest_counts_splits_classes_and_missing(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    raw = root / "02_구현" / "data" / "raw"
    (raw / "Cr_1.bmp").write_bytes(b"x")
    manifest = _write_manifest(
        root,
        [
            {
                "filepath": "data/raw/Cr_1.bmp",
                "split": "train",
                "class_label": "crazing",
            },
            {
                "filepath": "data/raw/Sc_1.bmp",
                "split": "test",
                "class_label": "scratches",
            },
        ],
    )
    summary = summarize_manifest(manifest, root / "02_구현")
    assert summary["exists"] is True
    assert summary["rows"] == 2
    assert summary["splits"] == {"train": 1, "test": 1}
    assert summary["classes"] == {"crazing": 1, "scratches": 1}
    assert summary["missing_file_count"] == 1
    assert summary["missing_examples"] == ["data/raw/Sc_1.bmp"]


def test_discover_checkpoint_files_supports_legacy_and_seed_layouts(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    artifacts = root / "02_구현" / "data" / "artifacts"
    (artifacts / "baseline").mkdir()
    (artifacts / "baseline" / "checkpoint.pt").write_bytes(b"x")
    (artifacts / "baseline" / "seed-42").mkdir()
    (artifacts / "baseline" / "seed-42" / "checkpoint.pt").write_bytes(b"x")
    (artifacts / "baseline" / "seed-42" / "notes.txt").write_text("x", encoding="utf-8")
    found = discover_checkpoint_files(artifacts)
    assert [path.name for path in found] == ["checkpoint.pt", "checkpoint.pt"]


def test_docs_test_count_status_requires_128(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    (root / "README.md").write_text("128개 통과", encoding="utf-8")
    (root / "02_구현" / "README.md").write_text("128개 통과", encoding="utf-8")
    (root / "02_구현" / "docs").mkdir()
    (root / "02_구현" / "docs" / "experiment-results.md").write_text("128", encoding="utf-8")
    assert docs_test_count_status(root)["consistent"] is True

    (root / "README.md").write_text("118개 통과", encoding="utf-8")
    result = docs_test_count_status(root)
    assert result["consistent"] is False
    assert result["missing_or_stale"] == ["README.md"]


def test_screenshot_status_reports_hash(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    screenshot = root / SCREENSHOT_RELATIVE
    screenshot.parent.mkdir(parents=True)
    screenshot.write_bytes(b"synthetic screenshot bytes")
    evidence = root / SCREENSHOT_EVIDENCE_RELATIVE
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(
        f"docs/screenshots/{screenshot.name}\n{sha256_file(screenshot)}",
        encoding="utf-8",
    )
    result = screenshot_status(root)
    assert result["exists"] is True
    assert result["evidence_exists"] is True
    assert result["evidence_valid"] is True
    assert result["sha256"] is not None
    assert len(result["sha256"]) == 64

    evidence.write_text("docs/screenshots/other.png\nnot-the-real-hash", encoding="utf-8")
    assert screenshot_status(root)["evidence_valid"] is False


def test_build_handoff_status_blocks_without_data_and_checkpoint(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    status = build_handoff_status(
        root,
        root / "02_구현" / "data" / "processed",
        root / "02_구현" / "data" / "artifacts",
        root / "02_구현" / "data" / "raw",
        include_environment=False,
    )
    tasks = {task["id"]: task for task in status["tasks"]}
    assert status["folder_name"]["actual"] == EXPECTED_FOLDER_NAME
    assert tasks["folder_name"]["status"] == "done"
    assert tasks["training"]["status"] == "blocked"
    assert tasks["reliability_measurement"]["status"] == "blocked"
    assert tasks["demo_screenshot"]["status"] == "pending"
    assert tasks["docs_test_count"]["status"] == "pending"
    assert status["next_action"]["id"] == "training"
    assert json.loads(json.dumps(status))["checkpoints"]["count"] == 0


def test_build_handoff_status_marks_training_ready_without_checkpoint(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    raw = root / "02_구현" / "data" / "raw"
    (raw / "Cr_1.bmp").write_bytes(b"x")
    _write_manifest(
        root,
        [
            {
                "filepath": "data/raw/Cr_1.bmp",
                "split": "train",
                "class_label": "crazing",
            }
        ],
    )
    status = build_handoff_status(
        root,
        root / "02_구현" / "data" / "processed",
        root / "02_구현" / "data" / "artifacts",
        raw,
        include_environment=False,
    )
    tasks = {task["id"]: task for task in status["tasks"]}
    assert tasks["training"]["status"] == "ready"
    assert tasks["reliability_measurement"]["status"] == "blocked"
    assert status["next_action"]["id"] == "reliability_measurement"


def test_build_handoff_status_marks_ready_when_artifacts_exist(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    raw = root / "02_구현" / "data" / "raw"
    (raw / "Cr_1.bmp").write_bytes(b"x")
    _write_manifest(
        root,
        [
            {
                "filepath": "data/raw/Cr_1.bmp",
                "split": "test",
                "class_label": "crazing",
            }
        ],
    )
    artifacts = root / "02_구현" / "data" / "artifacts" / "baseline" / "seed-42"
    artifacts.mkdir(parents=True)
    (artifacts / "checkpoint.pt").write_bytes(b"x")
    screenshot = root / SCREENSHOT_RELATIVE
    screenshot.parent.mkdir(parents=True)
    screenshot.write_bytes(b"synthetic screenshot bytes")
    (root / SCREENSHOT_EVIDENCE_RELATIVE).write_text(
        f"docs/screenshots/{screenshot.name}\n{sha256_file(screenshot)}",
        encoding="utf-8",
    )
    (root / "README.md").write_text("128", encoding="utf-8")
    (root / "02_구현" / "README.md").write_text("128", encoding="utf-8")
    (root / "02_구현" / "docs" / "experiment-results.md").write_text("128", encoding="utf-8")

    status = build_handoff_status(
        root,
        root / "02_구현" / "data" / "processed",
        root / "02_구현" / "data" / "artifacts",
        raw,
        include_environment=False,
    )
    tasks = {task["id"]: task for task in status["tasks"]}
    assert tasks["training"]["status"] == "done"
    assert tasks["reliability_measurement"]["status"] == "ready"
    assert tasks["demo_screenshot"]["status"] == "done"
    assert tasks["docs_test_count"]["status"] == "done"
    assert tasks["demo_runtime"]["status"] == "blocked"
    assert status["next_action"]["id"] == "demo_runtime"
    formatted = format_handoff_status(status)
    assert "다음 작업:" in formatted
    assert "reliability_measurement" in formatted


def test_reliability_metrics_require_a_reliability_block(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"x")
    metrics = checkpoint.with_name("test-metrics.json")
    metrics.write_text('{"accuracy": 1.0}', encoding="utf-8")
    assert reliability_metrics_for_checkpoints([checkpoint]) == []
    metrics.write_text('{"reliability": {"ece": 0.1}}', encoding="utf-8")
    assert reliability_metrics_for_checkpoints([checkpoint]) == [metrics]


def test_build_handoff_status_marks_reliability_done_with_metrics(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    raw = root / "02_구현" / "data" / "raw"
    (raw / "Cr_1.bmp").write_bytes(b"x")
    _write_manifest(root, [{"filepath": "data/raw/Cr_1.bmp", "split": "test", "class_label": "crazing"}])
    checkpoint = root / "02_구현" / "data" / "artifacts" / "baseline" / "seed-42" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")
    checkpoint.with_name("test-metrics.json").write_text('{"reliability": {"ece": 0.1}}', encoding="utf-8")
    status = build_handoff_status(root, root / "02_구현" / "data" / "processed", root / "02_구현" / "data" / "artifacts", raw, include_environment=False)
    tasks = {task["id"]: task for task in status["tasks"]}
    assert tasks["reliability_measurement"]["status"] == "done"
