from __future__ import annotations

import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from defect_cls.data import CLASSES, CLASS_TO_INDEX, eval_transform, train_transform

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_PIXELS = 20_000_000
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_THRESHOLD = 0.60


def validate_upload(filename: str, size_bytes: int) -> tuple[bool, str]:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return False, f"지원하지 않는 형식입니다: {extension or '(확장자 없음)'} — 허용: jpg, jpeg, png, bmp"
    if size_bytes <= 0:
        return False, "비어 있는 파일입니다."
    if size_bytes > MAX_UPLOAD_BYTES:
        return False, f"파일이 너무 큽니다: {size_bytes} bytes (최대 {MAX_UPLOAD_BYTES} bytes)"
    return True, "ok"


def load_image_tensor(
    source: bytes | str | Path,
    augment: bool = False,
    image_mode: str = "RGB",
) -> torch.Tensor:
    if image_mode not in {"RGB", "L"}:
        raise ValueError(f"unsupported image mode: {image_mode}")
    transform = (
        train_transform(augmented=augment, image_mode=image_mode)
        if augment
        else eval_transform(image_mode=image_mode)
    )
    if isinstance(source, bytes):
        try:
            image = Image.open(io.BytesIO(source))
        except Exception as exc:
            raise ValueError(f"손상되었거나 이미지가 아닌 파일입니다: {exc}") from exc
    else:
        try:
            image = Image.open(source)
        except Exception as exc:
            raise ValueError(f"손상되었거나 이미지가 아닌 파일입니다: {exc}") from exc
    try:
        if image.width * image.height > MAX_UPLOAD_PIXELS:
            raise ValueError(
                f"image has too many pixels: {image.width}x{image.height} "
                f"(maximum {MAX_UPLOAD_PIXELS})"
            )
        image.load()
        converted = image.convert(image_mode)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"손상되었거나 이미지가 아닌 파일입니다: {exc}") from exc
    tensor = transform(converted)
    return tensor.unsqueeze(0)


def predict(
    model,
    tensor: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
    temperature: float = 1.0,
) -> dict:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits / temperature, dim=1)[0]
    confidence, index = probs.max(dim=0)
    class_name = CLASSES[index.item()]
    return {
        "class_label": class_name,
        "class_index": CLASS_TO_INDEX[class_name],
        "confidence": confidence.item(),
        "probabilities": {name: probs[CLASS_TO_INDEX[name]].item() for name in CLASSES},
        "needs_review": confidence.item() < threshold,
        "threshold": threshold,
    }
