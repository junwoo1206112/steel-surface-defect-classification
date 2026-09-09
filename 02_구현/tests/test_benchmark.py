from __future__ import annotations

from pathlib import Path

import pytest

from defect_cls.benchmark import check_overwrite


def test_check_overwrite_write(tmp_path: Path) -> None:
    assert check_overwrite(tmp_path / "missing.json", force=False) == "write"


def test_check_overwrite_blocked(tmp_path: Path) -> None:
    target = tmp_path / "existing.json"
    target.write_text("{}")
    assert check_overwrite(target, force=False) == "blocked"


def test_check_overwrite_forced(tmp_path: Path) -> None:
    target = tmp_path / "existing.json"
    target.write_text("{}")
    assert check_overwrite(target, force=True) == "overwrite"
