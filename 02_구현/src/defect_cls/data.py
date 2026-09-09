from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASSES)}

FILENAME_ABBREVIATIONS = {
    "cr": "crazing",
    "in": "inclusion",
    "pa": "patches",
    "ps": "pitted_surface",
    "rs": "rolled-in_scale",
    "sc": "scratches",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRAYSCALE_MEAN = [0.449]
GRAYSCALE_STD = [0.226]
IMAGE_SIZE = 200

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

SPLIT_RATIOS = (0.70, 0.15, 0.15)
SPLIT_SEED = 42


def normalize_class_token(token: str) -> str:
    return token.strip().lower().replace(" ", "_").replace("-", "_")


def parse_class_from_filename(filename: str) -> str | None:
    stem = Path(filename).stem
    if "_" not in stem:
        return None
    class_token = stem.rsplit("_", 1)[0]
    normalized = normalize_class_token(class_token)
    if normalized in FILENAME_ABBREVIATIONS:
        return FILENAME_ABBREVIATIONS[normalized]
    for cls in CLASSES:
        if normalize_class_token(cls) == normalized:
            return cls
    return None


def build_stratified_split(
    entries: list[dict],
    seed: int = SPLIT_SEED,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
) -> list[dict]:
    if not abs(sum(ratios) - 1.0) < 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios}")
    by_class: dict[str, list[dict]] = {cls: [] for cls in CLASSES}
    for entry in entries:
        by_class[entry["class_label"]].append(entry)
    rng = random.Random(seed)
    result: list[dict] = []
    for cls in CLASSES:
        rows = sorted(by_class[cls], key=lambda r: r["filename"])
        rng.shuffle(rows)
        n = len(rows)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        for i, row in enumerate(rows):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            result.append({**row, "split": split})
    return result


def eval_transform(image_mode: str = "RGB") -> transforms.Compose:
    mean, std = (GRAYSCALE_MEAN, GRAYSCALE_STD) if image_mode == "L" else (IMAGENET_MEAN, IMAGENET_STD)
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def train_transform(augmented: bool, image_mode: str = "RGB") -> transforms.Compose:
    mean, std = (GRAYSCALE_MEAN, GRAYSCALE_STD) if image_mode == "L" else (IMAGENET_MEAN, IMAGENET_STD)
    base = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if augmented:
        base += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10, fill=0),
        ]
    base += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(base)


AUGMENT_PIPELINES = {
    "baseline": "Convert(RGB) -> Resize(200x200) -> ToTensor -> Normalize(ImageNet RGB)",
    "augmented": "Convert(RGB) -> Resize(200x200) -> HFlip(0.5) -> VFlip(0.5) -> Rotation(10deg, fill=0) -> ToTensor -> Normalize(ImageNet RGB)",
    "grayscale1ch": "Convert(L) -> Resize(200x200) -> ToTensor -> Normalize(gray 0.449/0.226)",
}


class DefectDataset(Dataset):
    def __init__(self, manifest_rows: list[dict], split: str, transform, image_mode: str = "RGB") -> None:
        self.rows = [row for row in manifest_rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"no rows for split '{split}'")
        self.split = split
        self.transform = transform
        self.image_mode = image_mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row["filepath"]).convert(self.image_mode)
        if self.transform is not None:
            image = self.transform(image)
        label = CLASS_TO_INDEX[row["class_label"]]
        return image, label


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
