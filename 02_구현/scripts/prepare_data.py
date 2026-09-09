from __future__ import annotations

import argparse
from pathlib import Path

from defect_cls.data import CLASSES, SPLIT_RATIOS, SPLIT_SEED
from defect_cls.preparation import run_preparation


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate, split and manifest the defect dataset")
    parser.add_argument("--input", type=Path, required=True, help="NEU-CLS zip file or extracted directory")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--ratios", nargs=3, type=float, default=list(SPLIT_RATIOS))
    parser.add_argument(
        "--allow-non-neu",
        action="store_true",
        help="do not enforce the 1,800 / 6x300 / 200x200 NEU-CLS intake contract",
    )
    args = parser.parse_args()

    manifest_path, quality_path, quality = run_preparation(
        args.input.resolve(),
        args.out_dir,
        args.seed,
        tuple(args.ratios),
        enforce_neu_contract=not args.allow_non_neu,
    )

    print(f"files found: {quality['files_found']}, entries kept: {quality['entries_kept']}, unique after dedupe: {quality['unique_entries']}, anomalies: {len(quality['anomalies'])}")
    print(f"duplicate hash groups: {quality['duplicate_hash_groups']}, excluded duplicates: {quality['exact_duplicate_handling']['excluded_count']}")
    print("class x split distribution:")
    print(f"{'class':<16}{'train':>8}{'val':>8}{'test':>8}{'total':>8}")
    for cls in CLASSES:
        counts = quality["eda"]["distribution"][cls]
        total = sum(counts.values())
        print(f"{cls:<16}{counts['train']:>8}{counts['val']:>8}{counts['test']:>8}{total:>8}")
    totals = quality["eda"]["totals"]
    print(f"{'TOTAL':<16}{totals['train']:>8}{totals['val']:>8}{totals['test']:>8}{sum(totals.values()):>8}")
    print(f"manifest: {manifest_path}")
    print(f"quality report: {quality_path}")

    filenames = [row["filename"] for row in _read_rows(manifest_path)]
    if len(filenames) != len(set(filenames)):
        raise SystemExit("leakage check failed: duplicate filenames across splits")
    print("leakage check: no filename appears in more than one split")


def _read_rows(manifest_path: Path) -> list[dict]:
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        import csv

        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
