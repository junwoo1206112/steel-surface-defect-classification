from __future__ import annotations

from pathlib import Path

import pytest

from defect_cls.benchmark import check_overwrite, resolve_benchmark_devices


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


def test_resolve_benchmark_devices_keeps_requested_cpu() -> None:
    assert [str(device) for device in resolve_benchmark_devices("cpu", cuda_available=False)] == ["cpu"]


@pytest.mark.parametrize("requested", ["", "cuda", "tpu", "cpu,tpu"])
def test_resolve_benchmark_devices_rejects_unavailable_or_invalid_targets(requested: str) -> None:
    with pytest.raises(ValueError):
        resolve_benchmark_devices(requested, cuda_available=False)
