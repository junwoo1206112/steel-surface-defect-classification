# AGENTS.md — 인수자·LLM 진입점

이 파일은 사람 인수자와 다른 LLM 에이전트가 **가장 먼저 읽는 요약**이다. 요구사항의 원본은 `01_협업_설계_페이즈/`에 있고, 이 파일은 그 원본을 대체하지 않는다.

## 1. 먼저 읽을 순서

1. `01_협업_설계_페이즈/README.md` — 인수인계 현황, 문서 순서, 현재 목표
2. `01_협업_설계_페이즈/04_decision_ledger.md` — 확정된 결정과 제약
3. `01_협업_설계_페이즈/05_verification_checklist.md` — 완료/미완료 점검 항목
4. `01_협업_설계_페이즈/08_ABC_교차검증_및_구현.md` — 2026-11-14 1차 교차 검증·구현 근거
5. `01_협업_설계_페이즈/09_잔여작업_ABC_교차검증.md` — 잔여 작업 재검증·인수인계 상태 도구
6. `02_구현/README.md` — 실행 명령과 결과 요약

## 2. 현재 상태 (2026-11-14)

- 전체 pytest: **128 passed** (수정 전 90개 수집·89 통과·1 실패)
- `python -m compileall -q src scripts tests app`: 성공
- `python -m pip check`: 정상
- `pip install --dry-run --only-binary=:all: -r requirements.txt`: 해결 성공
- 검증 환경: Python 3.12.10, torch 2.13.0 CPU wheel, numpy 2.2.6, opencv 4.12.0.88, pytest 8.4.2
- **검증 범위:** 합성 fixture·CPU 테스트. 실제 데이터 학습, GPU 측정, Streamlit 부팅은 이번에 재실행하지 않았다.

## 3. 최근 변경 요약

- OpenCV 후보 마스크가 1픽셀 얇은 스크래치·경계를 지우는 결함 수정 (`opencv_pipeline.py`)
- 신뢰도 평가 추가: ECE/MCE, Brier, NLL, top-3, risk-coverage, bootstrap CI (`reliability.py`, `evaluate.py`)
- 준비 파이프라인 원자성: RAR/미지원 압축 조기 거부, ZIP 실패 시 추출 디렉터리 정리, 빈 입력 거부 (`preparation.py`)
- 의존성 충돌 수정: `numpy==2.2.6` (opencv 4.12의 `numpy<2.3` 요구와 호환)
- train CLI 조기 검증·죽은 parser 제거 (`train.py`)
- 테스트 수·문서 동기화: 83 → 110 → **118**
- 읽기 전용 인수인계 점검 추가: `scripts/handoff_status.py` (data/raw·manifest·checkpoint·screenshot·폴더명·문서 수를 검사하고 다음 안전한 명령을 출력)
- 완료 확인: 데모 스크린샷 증빙·SHA-256, 현재 폴더명 `steel-surface-defect-classification`

## 4. 절대 규칙

- **측정하지 않은 Accuracy, F1, 지연시간, GPU/NPU 최적화 수치를 쓰지 않는다.** 기존 수치는 `08`의 `[Owner-states]` 표기를 확인한다.
- 프로젝트 표기는 **`산업 표면 결함 이미지 분류`**로 고정한다. 데이터가 강재이므로 `반도체`라고 쓰지 않고, 분류 과제이므로 `탐지`/`검출`을 쓰지 않는다.
- `data/raw` 원본 이미지와 checkpoint·가중치를 Git에 넣지 않는다.
- test set을 모델 선택·증강 선택에 사용하지 않는다.
- 테스트 수를 바꾸면 README·실험 리포트·포트폴리오 요약·`05`를 함께 갱신한다.
- OpenCV 후보 강조를 객체 탐지·세그멘테이션·결과 위치 예측으로 표현하지 않는다.
- 신뢰도 지표(ECE/Brier/risk-coverage/bootstrap)는 **고정 test set** 진단이다. 독립 캘리브레이션·도메인 일반화·운영 안전성으로 표현하지 않는다.

## 5. 실행

```bash
cd 02_구현
pip install -r requirements.txt
pip install -e .
python -m pytest -q -p no:cacheprovider --basetemp="$env:TEMP\opencode\pytest-tmp"
python scripts/handoff_status.py

# checkpoint가 있는 환경에서만
python -m defect_cls.evaluate --checkpoint data/artifacts/baseline/seed-42/checkpoint.pt --device cuda
streamlit run app/demo.py
```

Windows에서 pytest 기본 임시폴더 권한 오류가 있으므로 `--basetemp`를 지정한다.

## 6. 완료 상태와 후속 연구

- 실제 checkpoint CUDA 평가 및 reliability 문서화: **완료** — baseline seed=42, test 270장, ECE 0.003993 / Brier 0.002065 / NLL 0.004671. checkpoint와 manifest SHA-256은 `docs/experiment-results.md` 12절에 기록했다.
- 데모 스크린샷: **완료** — `baseline · seed 42`와 실제 업로드한 `Cr_1.bmp`의 OpenCV 화면을 `02_구현/docs/screenshots/demo-result-baseline-seed42-cr1-opencv.png`에 기록했다. 경로·SHA-256은 `docs/demo-capture-evidence.md`와 `handoff_status.py --strict`가 함께 검증한다.
- 상위 폴더명: **완료** (현재 `steel-surface-defect-classification`)
- 후속 연구(현재 범위 밖): 독립 split·다중 seed·외부 도메인에서의 calibration/reliability 검증
- 현재 상태는 `python scripts/handoff_status.py --strict`로 읽기 전용 확인 가능

상세한 발견·반박·합의 과정과 한계는 `01_협업_설계_페이즈/08_ABC_교차검증_및_구현.md`와 `09_잔여작업_ABC_교차검증.md`를 따른다.
