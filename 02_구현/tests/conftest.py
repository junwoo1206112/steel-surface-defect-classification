from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from defect_cls.data import CLASSES


@pytest.fixture
def synthetic_raw_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw" / "NEU-CLS"
    raw_dir.mkdir(parents=True)
    for class_index, class_name in enumerate(CLASSES):
        for i in range(1, 11):
            image = Image.new("L", (32, 32), color=(class_index * 40 + i) % 256)
            image.save(raw_dir / f"{class_name}_{i}.jpg")
    return raw_dir
