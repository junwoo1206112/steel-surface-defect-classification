"""Read-only handoff status for humans and LLM agents.

The status is intentionally observational: it does not train, evaluate, download,
or modify artifacts.  It answers "what is present, what is blocked, and what is
the next safe command" so a new owner does not have to infer state from prose.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_FOLDER_NAME = "steel-surface-defect-classification"
SCREENSHOT_RELATIVE = Path(
    "02_구현/docs/screenshots/demo-result-baseline-seed42-cr1-opencv.png"
)
SCREENSHOT_EVIDENCE_RELATIVE = Path("02_구현/docs/demo-capture-evidence.md")
DOC_TEST_COUNT_FILES = (
    Path("README.md"),
    Path("02_구현/README.md"),
    Path("02_구현/docs/experiment-results.md"),
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_images(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def summarize_manifest(path: Path, project_root: Path | None = None) -> dict:
    if not path.is_file():
        return {
            "exists": False,
            "rows": 0,
            "splits": {},
            "classes": {},
            "missing_file_count": 0,
            "missing_examples": [],
        }
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    splits = Counter(row.get("split", "") for row in rows)
    classes = Counter(row.get("class_label", "") for row in rows)
    missing = []
    if project_root is not None:
        for row in rows:
            filepath = row.get("filepath", "")
            if not filepath:
                missing.append("(empty filepath)")
                continue
            candidate = Path(filepath)
            resolved = candidate if candidate.is_absolute() else project_root / candidate
            if not resolved.is_file():
                missing.append(filepath)
    return {
        "exists": True,
        "rows": len(rows),
        "splits": dict(splits),
        "classes": dict(classes),
        "missing_file_count": len(missing),
        "missing_examples": missing[:3],
    }


def discover_checkpoint_files(artifacts_root: Path) -> list[Path]:
    if not artifacts_root.is_dir():
        return []
    paths = list(artifacts_root.glob("*/checkpoint.pt"))
    paths.extend(artifacts_root.glob("*/seed-*/checkpoint.pt"))
    return sorted(path for path in paths if path.is_file())


def reliability_metrics_for_checkpoints(checkpoints: list[Path]) -> list[Path]:
    """Return only metrics files that contain a completed reliability block."""
    completed = []
    for checkpoint in checkpoints:
        metrics_path = checkpoint.with_name("test-metrics.json")
        if not metrics_path.is_file():
            continue
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload.get("reliability"), dict):
            completed.append(metrics_path)
    return completed


def screenshot_status(project_root: Path) -> dict:
    screenshot = project_root / SCREENSHOT_RELATIVE
    evidence = project_root / SCREENSHOT_EVIDENCE_RELATIVE
    expected_evidence_path = screenshot.relative_to(project_root / "02_구현").as_posix()
    digest = sha256_file(screenshot) if screenshot.is_file() else None
    evidence_valid = False
    if digest is not None and evidence.is_file():
        try:
            evidence_text = evidence.read_text(encoding="utf-8")
            evidence_valid = (
                expected_evidence_path in evidence_text
                and digest.lower() in evidence_text.lower()
            )
        except OSError:
            evidence_valid = False
    return {
        "exists": screenshot.is_file(),
        "evidence_exists": evidence.is_file(),
        "evidence_valid": evidence_valid,
        "path": SCREENSHOT_RELATIVE.as_posix(),
        "evidence_path": expected_evidence_path,
        "sha256": digest,
    }


def docs_test_count_status(project_root: Path) -> dict:
    missing = []
    for relative in DOC_TEST_COUNT_FILES:
        path = project_root / relative
        if not path.is_file() or "128" not in path.read_text(encoding="utf-8"):
            missing.append(relative.as_posix())
    return {"current_count": 128, "consistent": not missing, "missing_or_stale": missing}


def environment_status(include_environment: bool = True) -> dict:
    if not include_environment:
        return {"checked": False}
    packages = {}
    for name in ("torch", "torchvision", "streamlit", "cv2", "matplotlib", "sklearn"):
        packages[name] = importlib.util.find_spec(name) is not None
    torch_info = {}
    if packages["torch"]:
        try:
            import torch

            torch_info = {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
            }
        except Exception as exc:  # pragma: no cover - environment dependent
            torch_info = {"error": str(exc)}
    return {
        "checked": True,
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": packages,
        "torch": torch_info,
    }


def _task(task_id: str, status: str, detail: str, command: str | None = None) -> dict:
    return {"id": task_id, "status": status, "detail": detail, "command": command}


def build_handoff_status(
    project_root: Path,
    processed_root: Path,
    artifacts_root: Path,
    raw_root: Path,
    include_environment: bool = True,
) -> dict:
    project_root = project_root.resolve()
    implementation_root = project_root / "02_구현"
    if not implementation_root.is_dir():
        implementation_root = project_root
    manifest = summarize_manifest(processed_root / "manifest.csv", implementation_root)
    raw_images = count_images(raw_root)
    checkpoints = discover_checkpoint_files(artifacts_root)
    reliability_metrics = reliability_metrics_for_checkpoints(checkpoints)
    screenshot = screenshot_status(project_root)
    docs = docs_test_count_status(project_root)
    environment = environment_status(include_environment)

    training_status = (
        "done" if checkpoints else ("ready" if raw_images and manifest["exists"] else "blocked")
    )
    tasks = [
        _task(
            "training",
            training_status,
            "checkpoint가 이미 있다."
            if checkpoints
            else f"data/raw 이미지 {raw_images}장, manifest {'있음' if manifest['exists'] else '없음'}"
            if raw_images
            else "data/raw에 학습 이미지가 없다. NEU-CLS를 먼저 배치해야 한다.",
            "python scripts/prepare_data.py --input data/raw/<NEU-CLS.zip 또는 해제 폴더>",
        ),
        _task(
            "reliability_measurement",
            "done"
            if reliability_metrics and manifest["exists"] and manifest["missing_file_count"] == 0
            else "ready"
            if checkpoints and manifest["exists"] and manifest["missing_file_count"] == 0
            else "blocked",
            f"reliability metrics {len(reliability_metrics)}개, checkpoint {len(checkpoints)}개, manifest {manifest['rows']}행, "
            f"누락 파일 {manifest['missing_file_count']}개"
            if checkpoints
            else "checkpoint가 없다. 학습 또는 기존 checkpoint 복원이 필요하다.",
            "python -m defect_cls.evaluate --checkpoint <checkpoint.pt> --device cuda",
        ),
        _task(
            "demo_screenshot",
            "done"
            if screenshot["exists"] and screenshot["evidence_exists"] and screenshot["evidence_valid"]
            else "pending",
            "현재 OpenCV 화면의 결과 캡처와 증빙 문서의 경로·SHA-256이 일치한다."
            if screenshot["exists"] and screenshot["evidence_exists"] and screenshot["evidence_valid"]
            else "현재 OpenCV 화면 캡처 또는 demo-capture-evidence.md의 경로·SHA-256 갱신이 없다.",
            "streamlit run app/demo.py  # 브라우저에서 결과 화면 캡처",
        ),
        _task(
            "demo_runtime",
            "ready" if environment.get("packages", {}).get("streamlit") else "blocked",
            "streamlit 설치됨" if environment.get("packages", {}).get("streamlit") else "streamlit 미설치",
            "streamlit run app/demo.py",
        ),
        _task(
            "docs_test_count",
            "done" if docs["consistent"] else "pending",
            "README·실험 리포트의 테스트 수가 128로 동기화됐다."
            if docs["consistent"]
            else f"128 표기가 없는 문서: {', '.join(docs['missing_or_stale'])}",
            "python -m pytest -q -p no:cacheprovider --basetemp=...  # 실제 수 확인",
        ),
        _task(
            "folder_name",
            "done" if project_root.name == EXPECTED_FOLDER_NAME else "pending",
            f"프로젝트 폴더명: {project_root.name}",
            None,
        ),
    ]
    next_task = next(
        (task for task in tasks if task["status"] in {"blocked", "pending"}), None
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": project_root.as_posix(),
        "folder_name": {"expected": EXPECTED_FOLDER_NAME, "actual": project_root.name},
        "data_raw": {"path": raw_root.as_posix(), "image_count": raw_images},
        "manifest": manifest,
        "checkpoints": {
            "count": len(checkpoints),
            "paths": [path.as_posix() for path in checkpoints],
        },
        "reliability_metrics": {
            "count": len(reliability_metrics),
            "paths": [path.as_posix() for path in reliability_metrics],
        },
        "screenshot": screenshot,
        "docs": docs,
        "environment": environment,
        "tasks": tasks,
        "next_action": next_task,
    }


def format_handoff_status(status: dict) -> str:
    lines = [
        "인수인계 상태",
        f"- 프로젝트: {status['project_root']}",
        f"- data/raw 이미지: {status['data_raw']['image_count']}장",
        f"- manifest: {status['manifest']['rows']}행, 누락 파일 {status['manifest']['missing_file_count']}개",
        f"- checkpoint: {status['checkpoints']['count']}개",
        f"- 스크린샷 증빙: {'있음' if status['screenshot']['exists'] else '없음'}",
        "",
        "작업 상태:",
    ]
    for task in status["tasks"]:
        lines.append(f"- [{task['status']}] {task['id']}: {task['detail']}")
    if status["next_action"]:
        lines.append("")
        lines.append(f"다음 작업: {status['next_action']['id']}")
        if status["next_action"].get("command"):
            lines.append(f"  {status['next_action']['command']}")
    return "\n".join(lines)
