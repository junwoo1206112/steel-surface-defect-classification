from __future__ import annotations

import argparse
import json
from pathlib import Path

from defect_cls.paths import ARTIFACTS_ROOT, PROCESSED_DATA_ROOT, resolve_project_path
from defect_cls.seed_metrics import summarize_seed_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate already-measured seed metrics sharing one manifest SHA-256"
    )
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, default=PROCESSED_DATA_ROOT / "manifest.csv")
    parser.add_argument("--out", type=Path, default=ARTIFACTS_ROOT / "seed-metrics-summary.json")
    args = parser.parse_args()

    summary = summarize_seed_metrics(args.metrics, args.manifest)
    out_path = resolve_project_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"aggregated {summary['run_count']} existing run(s): {out_path}")


if __name__ == "__main__":
    main()
