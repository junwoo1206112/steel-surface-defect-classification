from __future__ import annotations

from pathlib import Path

import pytest

from defect_cls.artifacts import discover_checkpoints


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint")


def test_discover_checkpoints_supports_legacy_and_seed_layouts_deterministically(tmp_path: Path) -> None:
    _touch(tmp_path / "baseline" / "checkpoint.pt")
    _touch(tmp_path / "baseline" / "seed-42" / "checkpoint.pt")
    _touch(tmp_path / "augmented" / "seed-7" / "checkpoint.pt")

    options = discover_checkpoints(tmp_path)

    assert [option.label for option in options] == [
        "augmented · seed 7",
        "baseline · seed 42",
        "baseline · legacy (unversioned)",
    ]


def test_discover_checkpoints_ignores_unrelated_or_invalid_layouts(tmp_path: Path) -> None:
    _touch(tmp_path / "baseline" / "seed-latest" / "checkpoint.pt")
    _touch(tmp_path / "baseline" / "seed-8" / "other.pt")
    _touch(tmp_path / "notes" / "checkpoint.txt")

    assert discover_checkpoints(tmp_path) == []


def test_discover_checkpoints_rejects_ambiguous_seed_identity(tmp_path: Path) -> None:
    _touch(tmp_path / "baseline" / "seed-1" / "checkpoint.pt")
    _touch(tmp_path / "baseline" / "seed-01" / "checkpoint.pt")

    with pytest.raises(ValueError, match="ambiguous"):
        discover_checkpoints(tmp_path)
