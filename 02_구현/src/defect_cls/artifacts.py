from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_SEED_DIRECTORY = re.compile(r"seed-(\d+)$")


@dataclass(frozen=True)
class CheckpointOption:
    path: Path
    experiment: str
    seed: int | None

    @property
    def label(self) -> str:
        if self.seed is None:
            return f"{self.experiment} · legacy (unversioned)"
        return f"{self.experiment} · seed {self.seed}"


def discover_checkpoints(artifacts_dir: str | Path) -> list[CheckpointOption]:
    """Find only supported legacy and seed-scoped checkpoint layouts deterministically."""
    root = Path(artifacts_dir)
    if not root.exists():
        return []

    options: list[CheckpointOption] = []
    for path in root.glob("*/checkpoint.pt"):
        if path.is_file():
            options.append(CheckpointOption(path=path, experiment=path.parent.name, seed=None))
    for path in root.glob("*/seed-*/checkpoint.pt"):
        match = _SEED_DIRECTORY.fullmatch(path.parent.name)
        if path.is_file() and match:
            options.append(
                CheckpointOption(
                    path=path,
                    experiment=path.parent.parent.name,
                    seed=int(match.group(1)),
                )
            )

    options.sort(key=lambda option: (option.experiment, option.seed is None, option.seed or -1))
    identities = [(option.experiment, option.seed) for option in options]
    if len(identities) != len(set(identities)):
        raise ValueError("ambiguous checkpoint layout: duplicate experiment and seed")
    return options
