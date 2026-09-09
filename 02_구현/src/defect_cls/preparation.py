from __future__ import annotations

import csv
import hashlib
import json
import shutil
import stat
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from defect_cls.data import CLASSES, build_stratified_split, parse_class_from_filename

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ARCHIVE_EXTENSIONS = {".zip", ".rar"}
NEU_EXPECTED_TOTAL = 1800
NEU_EXPECTED_PER_CLASS = 300
NEU_EXPECTED_SIZE = (200, 200)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def to_manifest_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def inspect_entries(files: list[Path]) -> tuple[list[dict], list[dict]]:
    entries = []
    anomalies = []
    for path in files:
        class_label = parse_class_from_filename(path.name)
        if class_label is None:
            anomalies.append({"file": str(path), "issue": "class not parseable from filename"})
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
        except Exception as exc:
            anomalies.append({"file": str(path), "issue": f"decode failed: {exc}"})
            continue
        entries.append(
            {
                "filepath": to_manifest_path(path),
                "filename": path.name,
                "class_label": class_label,
                "size_bytes": path.stat().st_size,
                "width": width,
                "height": height,
                "sha256": sha256_of(path),
            }
        )
    return entries, anomalies


def find_duplicates(entries: list[dict]) -> list[dict]:
    by_hash: dict[str, list[str]] = {}
    for entry in entries:
        by_hash.setdefault(entry["sha256"], []).append(entry["filename"])
    return [
        {"sha256": digest, "files": names}
        for digest, names in by_hash.items()
        if len(names) > 1
    ]


def validate_neu_contract(entries: list[dict], anomalies: list[dict]) -> list[str]:
    """Return violations of the documented NEU-CLS intake contract."""
    violations = []
    if anomalies:
        violations.append(f"{len(anomalies)} invalid or unparseable image(s)")
    if len(entries) != NEU_EXPECTED_TOTAL:
        violations.append(f"expected {NEU_EXPECTED_TOTAL} decoded images, found {len(entries)}")
    class_counts = Counter(entry["class_label"] for entry in entries)
    for class_name in CLASSES:
        count = class_counts[class_name]
        if count != NEU_EXPECTED_PER_CLASS:
            violations.append(
                f"expected {NEU_EXPECTED_PER_CLASS} images for {class_name}, found {count}"
            )
    invalid_sizes = [
        entry["filename"]
        for entry in entries
        if (entry["width"], entry["height"]) != NEU_EXPECTED_SIZE
    ]
    if invalid_sizes:
        violations.append(
            f"expected all images to be {NEU_EXPECTED_SIZE[0]}x{NEU_EXPECTED_SIZE[1]}, "
            f"found {len(invalid_sizes)} mismatch(es)"
        )
    return violations


def phash_of(path: Path) -> int:
    """Return a compact perceptual hash for duplicate-review candidates.

    This is intentionally a review signal, not an automatic exclusion rule:
    industrial texture images can be visually similar while still being valid
    independent samples.
    """
    with Image.open(path) as image:
        pixels = np.asarray(
            image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64
        )
    indices = np.arange(32)
    basis = np.cos(np.pi * (indices[:, None] * (2 * indices + 1)) / 64)
    coefficients = basis @ pixels @ basis.T
    low_frequency = coefficients[:8, :8]
    median = np.median(low_frequency.flatten()[1:])
    digest = 0
    for value in low_frequency.flatten():
        digest = (digest << 1) | int(value > median)
    return digest


def find_near_duplicate_candidates(
    entries: list[dict], max_hamming_distance: int = 2
) -> list[dict]:
    """Report visually near image pairs without changing the dataset split."""
    if max_hamming_distance < 0:
        raise ValueError("max_hamming_distance must be non-negative")
    hashes = [(entry, phash_of(Path(entry["filepath"]))) for entry in entries]
    candidates = []
    for left_index, (left, left_hash) in enumerate(hashes):
        for right, right_hash in hashes[left_index + 1 :]:
            distance = (left_hash ^ right_hash).bit_count()
            if distance <= max_hamming_distance:
                candidates.append(
                    {
                        "left": left["filename"],
                        "right": right["filename"],
                        "left_class": left["class_label"],
                        "right_class": right["class_label"],
                        "hamming_distance": distance,
                    }
                )
    return candidates


def dedupe_entries(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    seen: dict[str, str] = {}
    excluded: list[dict] = []
    for entry in entries:
        digest = entry["sha256"]
        if digest in seen:
            excluded.append(
                {
                    "kept": seen[digest],
                    "excluded": entry["filename"],
                    "sha256": digest,
                }
            )
        else:
            seen[digest] = entry["filename"]
            kept.append(entry)
    return kept, excluded


def write_manifest(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filepath",
        "filename",
        "class_label",
        "class_index",
        "split",
        "width",
        "height",
        "size_bytes",
        "sha256",
    ]
    class_to_index = {name: index for index, name in enumerate(CLASSES)}
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "class_index": class_to_index[row["class_label"]]})


def eda(rows: list[dict]) -> dict:
    distribution = {
        cls: {split: 0 for split in ("train", "val", "test")} for cls in CLASSES
    }
    for row in rows:
        distribution[row["class_label"]][row["split"]] += 1
    totals = {
        split: sum(distribution[cls][split] for cls in CLASSES)
        for split in ("train", "val", "test")
    }
    sizes = Counter((row["width"], row["height"]) for row in rows)
    return {
        "distribution": distribution,
        "totals": totals,
        "image_sizes": {f"{w}x{h}": count for (w, h), count in sizes.items()},
    }


SEVENZIP_CANDIDATES = (
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
)


def validate_zip_members(archive: zipfile.ZipFile, extract_dir: Path) -> None:
    """Reject path traversal and symlink entries before extracting a ZIP file."""
    root = extract_dir.resolve()
    for member in archive.infolist():
        target = (extract_dir / member.filename).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"unsafe ZIP member path: {member.filename}")
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(f"ZIP symlink entries are not supported: {member.filename}")


def extract_archive(input_path: Path, data_root: Path) -> Path:
    extract_dir = data_root / input_path.stem
    if extract_dir.exists():
        raise FileExistsError(f"extract dir already exists, remove it first: {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    suffix = input_path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            validate_zip_members(archive, extract_dir)
            archive.extractall(extract_dir)
    elif suffix == ".rar":
        sevenzip = shutil.which("7z") or next(
            (path for path in SEVENZIP_CANDIDATES if Path(path).exists()), None
        )
        if sevenzip is None:
            raise FileNotFoundError(
                "RAR 해제를 위한 7-Zip(7z.exe)을 찾지 못했다. 7-Zip 설치 후 재실행하거나 "
                "미리 압축을 푼 디렉터리를 --input으로 지정한다."
            )
        result = subprocess.run(
            [sevenzip, "x", str(input_path), f"-o{extract_dir}", "-y"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"7z extraction failed (exit {result.returncode}): {result.stderr}")
    else:
        raise ValueError(f"unsupported archive type: {suffix}")
    return extract_dir


def run_preparation(
    input_path: Path,
    out_dir: Path,
    seed: int,
    ratios: tuple[float, float, float],
    enforce_neu_contract: bool = False,
) -> tuple[Path, Path, dict]:
    data_root = Path("data/raw")
    if input_path.suffix.lower() in ARCHIVE_EXTENSIONS:
        scan_root = extract_archive(input_path, data_root)
    else:
        scan_root = input_path

    files = collect_images(scan_root)
    entries, anomalies = inspect_entries(files)
    contract_violations = validate_neu_contract(entries, anomalies)
    if enforce_neu_contract and contract_violations:
        raise ValueError("NEU-CLS intake contract failed: " + "; ".join(contract_violations))
    duplicates = find_duplicates(entries)
    unique_entries, excluded_duplicates = dedupe_entries(entries)
    near_duplicates = find_near_duplicate_candidates(unique_entries)
    split_rows = build_stratified_split(unique_entries, seed=seed, ratios=ratios)
    manifest_path = out_dir / "manifest.csv"
    write_manifest(split_rows, manifest_path)

    quality_path = out_dir / "data-quality.json"
    quality = {
        "input": str(input_path),
        "scanned_root": str(scan_root),
        "files_found": len(files),
        "entries_kept": len(entries),
        "unique_entries": len(unique_entries),
        "exact_duplicate_handling": {
            "policy": "SHA-256 동일 파일은 정렬 순 첫 파일만 유지하고 나머지는 분할 전 제외(누수 방지)",
            "excluded_count": len(excluded_duplicates),
            "details": excluded_duplicates,
        },
        "duplicate_hash_groups": len(duplicates),
        "near_duplicate_review": {
            "method": "perceptual hash (32px DCT, 8x8 low-frequency), Hamming distance <= 2; review-only, no automatic exclusion",
            "candidate_count": len(near_duplicates),
            "candidates": near_duplicates,
        },
        "anomalies": anomalies,
        "class_counts_raw": dict(Counter(entry["class_label"] for entry in entries)),
        "neu_contract": {
            "enforced": enforce_neu_contract,
            "passed": not contract_violations,
            "expected": {
                "classes": list(CLASSES),
                "total": NEU_EXPECTED_TOTAL,
                "per_class": NEU_EXPECTED_PER_CLASS,
                "image_size": list(NEU_EXPECTED_SIZE),
            },
            "violations": contract_violations,
        },
        "seed": seed,
        "ratios": list(ratios),
        "eda": eda(split_rows),
    }
    quality_path.write_text(
        json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest_path, quality_path, quality
