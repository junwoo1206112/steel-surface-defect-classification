from __future__ import annotations

import argparse
import json
from pathlib import Path

from defect_cls.smoke import run_cpu_smoke_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely load a checkpoint and run one CPU forward pass")
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_cpu_smoke_test(args.checkpoint), ensure_ascii=False))


if __name__ == "__main__":
    main()
