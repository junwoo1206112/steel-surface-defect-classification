from __future__ import annotations

import argparse
import json
from pathlib import Path

from defect_cls.handoff import build_handoff_status, format_handoff_status
from defect_cls.paths import (
    ARTIFACTS_ROOT,
    PROCESSED_DATA_ROOT,
    RAW_DATA_ROOT,
    REPOSITORY_ROOT,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report handoff readiness without training, evaluating, or modifying files"
    )
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_DATA_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--raw-root", type=Path, default=RAW_DATA_ROOT)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    parser.add_argument(
        "--no-environment",
        action="store_true",
        help="skip optional package/torch probing",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when any required task is blocked or pending (ready is non-blocking)",
    )
    args = parser.parse_args()

    status = build_handoff_status(
        args.repository_root,
        args.processed_root,
        args.artifacts_root,
        args.raw_root,
        include_environment=not args.no_environment,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.json:
        print(json.dumps(status, indent=2, ensure_ascii=False))
    else:
        print(format_handoff_status(status))
    if args.strict and any(task["status"] in {"blocked", "pending"} for task in status["tasks"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
