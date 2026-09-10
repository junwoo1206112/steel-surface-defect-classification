from __future__ import annotations

import subprocess
import sys


def test_train_help_is_available() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "defect_cls.train", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Train defect classification model" in result.stdout
