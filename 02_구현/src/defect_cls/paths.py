from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_ROOT = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_ROOT = PROJECT_ROOT / "data" / "artifacts"


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a user-facing relative path from the project root, not CWD."""
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def project_relative_path(path: str | Path) -> str:
    """Return a portable project-relative POSIX path."""
    return resolve_project_path(path).relative_to(PROJECT_ROOT).as_posix()


def resolve_raw_path(path: str | Path) -> Path:
    """Resolve a manifest image path and require that it stays under data/raw."""
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else resolve_project_path(candidate)
    try:
        resolved.relative_to(RAW_DATA_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"data path must stay under {RAW_DATA_ROOT}: {path}") from exc
    return resolved


def require_raw_input(path: str | Path) -> Path:
    """Require preparation input to be an existing local file/directory inside data/raw."""
    resolved = resolve_project_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"input does not exist: {resolved}")
    try:
        resolved.relative_to(RAW_DATA_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"input must be placed under data/raw before preparation: {resolved}"
        ) from exc
    return resolved


def seed_artifact_dir(experiment: str, seed: int, out_root: str | Path = ARTIFACTS_ROOT) -> Path:
    if not experiment:
        raise ValueError("experiment must not be empty")
    return resolve_project_path(out_root) / experiment / f"seed-{seed}"
