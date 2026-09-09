from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import pytest
from PIL import Image

from defect_cls.data import CLASSES
from defect_cls.paths import RAW_DATA_ROOT


@pytest.fixture
def synthetic_raw_dir(tmp_path: Path) -> Path:
    raw_dir = RAW_DATA_ROOT / f".pytest-{uuid.uuid4().hex}" / "NEU-CLS"
    raw_dir.mkdir(parents=True)
    for class_index, class_name in enumerate(CLASSES):
        for i in range(1, 11):
            image = Image.new("L", (32, 32), color=(class_index * 40 + i) % 256)
            image.save(raw_dir / f"{class_name}_{i}.jpg")
    try:
        yield raw_dir
    finally:
        shutil.rmtree(raw_dir.parent, ignore_errors=True)
