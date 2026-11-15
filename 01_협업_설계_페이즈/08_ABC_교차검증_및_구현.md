# 08. 전문가 A·B·C 교차 검증과 구현 기록 (2026-11-14)

기준 문서: `00`~`07`. 이 문서는 **Ouroboros 인터뷰 스킬(`skills/interview/SKILL.md`)의 Path B(Socratic fallback)** 절차를 적용한 후속 검증이다. MCP 인터뷰 서버가 없는 환경이므로 Path B의 원칙만 사용했다: 코드로 답할 수 있는 것은 먼저 확인하고, 사람(여기서는 인수자·기존 문서)의 판단이 필요한 것은 `[Owner-states]`로 분리하며, 모호성 원장과 Refine gate를 거쳐 구현 전에 합의를 고정한다.

- 검증 대상: `01_협업_설계_페이즈/`의 설계 원본 + `02_구현/`의 실제 코드·테스트·문서.
- 검증 환경: `%TEMP%\opencode\py-embed`의 임시 Python 3.12.10. 저장소 밖이며 검증 후 삭제할 수 있다.
- 환경 버전: torch 2.13.0(PyPI CPU wheel), torchvision 0.28.0, numpy 2.2.6, opencv-python-headless 4.12.0.88, pillow 12.2.0, scikit-learn 1.9.0, pytest 8.4.2.
- **한계:** RTX 3070/CUDA 환경이 아니고, `data/raw`·체크포인트·Streamlit이 없다. 따라서 기존의 GPU 학습·평가·벤치마크·데모 수치는 재측정하지 않았고, 이번 변경으로 새 성능 수치를 만들지 않았다.

## 출처 표기

| 표기 | 의미 | 이번 문서의 예 |
| --- | --- | --- |
| `[Code-verified]` | 저장소 코드를 읽거나 실행해 확인 | 테스트 90→110, OpenCV 마스크 결함, requirements 충돌, RAR 지연 실패, manifest 1,799장 |
| `[Owner-states]` | 기존 문서·인수자 진술이며 독립 재측정하지 않음 | test 1.0000, CPU/GPU 지연시간, OOD 표, 확률 보정 표, 1채널 0.9963 |

`data/processed/manifest.csv`와 `data-quality.json`은 저장소에 있으므로 직접 확인했다: **1,799행**, split **1,259/270/270**, 클래스 **300/300/299/300/300/300**, SHA-256 중복 배제 **1건**, 근접 중복 후보 **0건**. 이는 `[Code-verified]`다. 반면 `test-metrics.json`·`benchmark.json`·`ood-robustness.json`·체크포인트는 이 체크아웃에 없어 기존 수치는 `[Owner-states]`로 취급했다.

> **후속 기록:** 이 문서는 1차 수정 시점의 110개 테스트를 보존한다. 이후 `09_잔여작업_ABC_교차검증.md`에서 인수인계 점검 도구를 추가해 전체 118개가 됐고, 데모 스크린샷·폴더명은 완료, 실제 `reliability` 실측은 데이터/checkpoint 부재로 blocked로 판정했다.

---

## Q1 — What are we working on?

### 시스템 요약

공개 강재 표면 결함 데이터(NEU-CLS)를 입력으로, SHA-256 중복 제거 → stratified 70/15/15 분할 → ResNet18 전이학습 → 동일 test set 평가 → 단일 이미지 벤치마크 → OpenCV 입력 진단이 포함된 Streamlit 판단 보조 데모까지 이어지는 재현 실험이다. 산출물은 checkpoint·metric JSON·문서·테스트이며, 원본 데이터와 가중치는 Git에 넣지 않는다. 성능 주장의 범위는 "공개 데이터 기반 단일 split·seed 관측값"으로 제한된다.

### 자산(asset)

| 자산 | 왜 중요한가 | 보호 수단 |
| --- | --- | --- |
| `data/raw` 원본 이미지 | 재배포 금지·라이선스 제약 | `.gitignore`, `data/raw` 경계 검사 |
| `manifest.csv`·`data-quality.json` | 분할·중복·계약의 단일 출처 | SHA-256, split manifest, seed 고정 |
| checkpoint·metric·benchmark JSON | 성능·재현성 주장의 증거 | seed별 경로, 덮어쓰기 방지, manifest SHA-256 일치 |
| 문서의 수치·표현 | 채용 심사에서 신뢰와 직결 | 측정 없는 수치 금지, `분류` 표기 고정 |
| 데모 출력 | 사용자 판단 보조 | 저신뢰도·품질 게이트, 오해 방지 문구 |

### 진입점(entry point)과 신뢰 경계

| 진입점 | 입력 | 신뢰 경계 |
| --- | --- | --- |
| `scripts/prepare_data.py` | ZIP/디렉터리/RAR 시도 | 외부 압축·파일 → `data/raw` 내부만 허용 |
| `defect_cls.train` | manifest·CLI 인자 | 하이퍼파라미터 → checkpoint |
| `defect_cls.evaluate` | checkpoint·manifest | 학습 분할 → test 지표 |
| `defect_cls.benchmark` | checkpoint·이미지·장치 | 측정 환경 → 지연시간 주장 |
| `app/demo.py` | 업로드 이미지 | 외부 파일 → 모델 추론·품질 경고 |
| 문서 | 코드·JSON | 내부 측정 → 외부 주장 |

---

## Q2 — What can go wrong?

세 전문가가 독립적으로 결함 후보를 제시한 뒤 서로 반박했다.

### 전문가 A — 기술 아키텍트 (CV/ML 시스템)

| ID | 발견 | 근거 | 심각도 |
| --- | --- | --- | --- |
| A1 | OpenCV 후보 강조가 **1픽셀 얇은 스크래치·경계를 지운다.** `MORPH_OPEN(3×3)`이 얇은 구조를 전부 제거해 마스크가 0이 된다. | `preprocess_defect_candidates` 실행: 원 마스크 132px → opening 후 0px. 기존 테스트 `test_preprocessing_produces_mask_and_does_not_mutate_input` 실패. | **높음** |
| A2 | 평가가 Accuracy·macro F1·혼동행렬에만 머문다. 문서는 과신을 발견했지만, **ECE·Brier·NLL·risk-coverage·bootstrap 구간**을 산출하는 코드가 없다. | `evaluate.py`는 argmax만 반환. `reliability` 개념은 문서에만 존재. | **높음** |
| A3 | `evaluate.py`가 확률을 저장하지 않아, 기존 결과로 보정·선택적 예측을 재분석할 수 없다. | `compute_predictions`가 `argmax`만 반환. | 중간 |
| A4 | `train.py`에 실행되지 않는 죽은 `ArgumentParser` 줄이 있고, epochs/batch/lr 음수·0을 조기 거부하지 않는다. | `check_training_output` 내부의 stray parser, `main` 인자 검증 부재. | 낮음 |

**A의 판정:** A1은 제품 결함이자 포트폴리오 주장 결함이다. "후보 강조"가 얇은 결함을 숨기면 시각화의 목적 자체가 훼손된다. A2/A3는 포화된 정확도를 보완할 가장 값싼 평가 확장이다.

### 전문가 B — 포트폴리오 심사자

| ID | 발견 | 근거 | 심각도 |
| --- | --- | --- | --- |
| B1 | 문서의 **테스트 83개** 표기가 현재 코드와 불일치한다. 실제 수집 결과 **90개**였고, 그중 1개가 실패했다. | `pytest` 수정 전 실행: 89 passed, 1 failed. README·요약·리포트·체크리스트가 모두 83을 인용. | **높음** |
| B2 | 체크리스트의 OpenCV 테스트 항목이 미완료(`[ ]`)인데, 문서는 Phase 7 완료·83개 통과를 주장한다. | `05_verification_checklist.md` 37행, 실제 테스트 실패. | **높음** |
| B3 | "1.0000"만으로는 평가 역량이 드러나지 않는다. **불확실성·보정·보류**를 같은 test set에서 보여 주는 것이 심사자에게 더 강한 증거다. | 기존 문서의 과신 분석은 좋지만 재실행 도구가 없다. | 중간 |
| B4 | 데모 스크린샷이 없고, 상위 폴더명이 여전히 `반도체…탐지…`다. | `05`의 남은 항목, 07의 사용자 결정 항목. | 낮음(사용자 작업) |

**B의 판정:** B1/B2는 "측정과 문서가 일치해야 한다"는 프로젝트의 핵심 계약 위반이다. B3은 새 성능 자랑이 아니라 **정직한 불확실성 보고**로 해결해야 한다. B4는 코드로 해결할 수 없으므로 사용자 작업으로 남긴다.

### 전문가 C — 재현성·QA·안전성

| ID | 발견 | 근거 | 심각도 |
| --- | --- | --- | --- |
| C1 | `requirements.txt`의 **numpy 2.5.2와 opencv-python-headless 4.12.0.88이 충돌**한다. opencv 4.12는 `numpy<2.3.0`을 요구하므로 한 줄 설치가 해결 불가능하거나 numpy를 몰래 내린다. | 실제 설치에서 numpy 2.2.6으로 강제 다운그레이드. `pip install -r requirements.txt` 재현성 결함. | **높음** |
| C2 | RAR 거부가 **manifest 작성 이후**에 발생한다. 실패했는데 빈 `manifest.csv`가 남을 수 있다. | `run_preparation`의 late `if suffix == ".rar"` 블록. | 중간 |
| C3 | ZIP 검증 실패 시 **빈 추출 디렉터리**가 남아 다음 실행이 `FileExistsError`로 오해를 부른다. | `extract_archive`가 `mkdir` 후 검증. 기존 테스트는 경로 탈출만 확인. | 중간 |
| C4 | 이미지가 하나도 없는 디렉터리를 조용히 통과시켜 **빈 manifest**를 만들 수 있다. | `collect_images` 결과 0건 검사 없음. | 중간 |
| C5 | 테스트 수가 문서와 다르면 "테스트 통과" 주장 전체의 신뢰가 떨어진다. | B1과 동일. | **높음** |

**C의 판정:** C1은 설치 재현성의 최상위 결함이다. C2~C4는 "실패는 실패로, 산출물은 원자적으로" 원칙의 위반이다. 새 기능보다 먼저 고쳐야 한다.

### 교차 반박과 합의

| 쟁점 | A | B | C | 합의 |
| --- | --- | --- | --- | --- |
| OpenCV 얇은 구조 | 제품 결함, 즉시 수정 | 문서 주장과 불일치 | 회귀 테스트 필요 | **수정 + 테스트** |
| 테스트 83 vs 90 | 코드 사실 | 신뢰 문제 | 재현성 문제 | **실제 수로 동기화** |
| 신뢰도 평가 추가 | 찬성 | 찬성(차별점) | **실측 없는 수치 금지** 조건부 찬성 | **도구·테스트·문서만 추가, 실데이터 실행은 보류** |
| 다중 seed 재학습 | 가치 있음 | 수치 보강 | 데이터·체크포인트 없음, 무측정 수치 위험 | **보류** |
| Grad-CAM/위치 시각화 | 위치 주장 위험 | 시각적 임팩트 | 검증 비용·오해 위험 | **기각(범위 밖)** |
| 스크린샷 | 자동화 불가 | 있으면 좋음 | 사용자 작업 | **사용자 작업 유지** |
| requirements | 설치 문제 | 문서 신뢰 | 최우선 | **numpy 핀 수정** |

**모순 해소:** B가 원한 "새로운 시각적 증거"는 A/C의 반대에 부딪혔다. 대신 **기존 test set에서 언제든 재현 가능한 신뢰도 리포트**를 코드로 제공하고, 문서에는 수치를 쓰지 않는 것으로 합의했다. 이는 "측정하지 않은 수치 금지" 원칙과 "심사자가 3분 안에 검증" 원칙을 동시에 만족한다.

---

## Q3 — What are we going to do about it?

### 채택한 변경

| # | 변경 | 파일 | 완료 기준 |
| --- | --- | --- | --- |
| 1 | OpenCV 후보 마스크가 얇은 구조를 보존하도록 opening 결과가 거의 전부를 지우면 원 마스크를 유지 | `src/defect_cls/opencv_pipeline.py` | 기존 실패 테스트 + 1픽셀 선 테스트 통과 |
| 2 | 신뢰도·선택적 예측 모듈 추가: ECE/MCE, Brier, NLL, top-k, risk-coverage, bootstrap CI, macro F1 | `src/defect_cls/reliability.py` | 알려진 값 테스트 통과, PyTorch 비의존 |
| 3 | `evaluate.py`가 softmax 확률을 계산하고 `test-metrics.json`에 `reliability` 블록 저장 | `src/defect_cls/evaluate.py` | 기존 키 유지, `--calibration-bins`/`--bootstrap-samples` 검증 |
| 4 | 준비 파이프라인 원자성: RAR/미지원 압축 조기 거부, ZIP 실패 시 추출 디렉터리 정리, 빈 입력 거부 | `src/defect_cls/preparation.py` | 실패 후 manifest·추출 디렉터리 없음 |
| 5 | 의존성 충돌 수정: `numpy==2.2.6` (opencv 4.12의 `<2.3` 요구와 호환) | `requirements.txt` | `pip check` 정상, numpy/opencv 동시 설치 |
| 6 | 학습 CLI 조기 검증 + 죽은 parser 제거 | `src/defect_cls/train.py` | 음수/0 인자 거부, help 정상 |
| 7 | 테스트 수·문서 동기화, 신뢰도 문서 추가 | README·`docs/`·`01` 원장·체크리스트 | 110개 통과와 문서 수치 일치 |

### 명시적으로 하지 않은 것

- **재학습·재평가·다중 seed 실행:** `data/raw`와 checkpoint가 없고, 이 환경은 CUDA가 아니다. 새 수치를 만들면 `[Owner-states]`와 섞여 문서 무결성이 깨진다.
- **Grad-CAM·객체 탐지·세그멘테이션:** 위치 예측으로 오해될 위험이 크고, Phase 7의 "후보 강조는 모델 위치 예측 아님" 계약과 충돌한다.
- **블러 검출기 추가:** 기존 라플라시안 선명도가 이 텍스처에서 분리되지 않는다는 실측 결론을 뒤집을 근거가 없다. 새 지표는 독립 holdout 캘리브레이션이 필요하다.
- **데모 스크린샷 자동 캡처:** Streamlit·브라우저 자동화가 이 환경에 없고, 과거 headless 캡처는 스켈레톤만 얻었다. 허위 이미지를 넣지 않는다.
- **상위 폴더명 변경:** 외부 링크·경로에 영향을 주므로 사용자 결정으로 남긴다.

### 수용 기준(acceptance criteria)

1. `python -m compileall -q src scripts tests app` 성공.
2. 전체 pytest가 실패 0으로 통과하고, 문서의 테스트 수가 실제 수와 일치한다.
3. `test-metrics.json`의 기존 키가 유지되고 `reliability` 키가 추가된다.
4. 실패 경로(RAR, ZIP 경로 탈출, 빈 디렉터리)가 산출물을 남기지 않는다.
5. `pip check`가 정상이고 requirements의 numpy/opencv 제약이 호환된다.
6. 새 학습·평가 수치, 새 checkpoint, 원본 데이터가 Git diff에 추가되지 않는다.

---

## Q4 — Did we do a good job?

### 실행한 검증 (2026-11-14)

| 명령 | 결과 |
| --- | --- |
| `python -m compileall -q src scripts tests app` | 성공 |
| `python -m pytest -q -p no:cacheprovider --basetemp=...` | **110 passed** (14.69s 최초, 8.95s 재실행) |
| `python -m pip check` | `No broken requirements found.` |
| `python -m pip install --dry-run --only-binary=:all: -r requirements.txt` | 해결 성공 — numpy 2.2.6 + opencv 4.12 + torch 2.13.0+cu126 + streamlit 조합. 실제 설치·CUDA 실행은 하지 않음 |
| `python -c "import cv2, numpy; ..."` | numpy 2.2.6 / opencv 4.12.0 |
| `python -m defect_cls.evaluate --help` | 신뢰도 옵션 2개 표시 |
| 합성 12장·무작위 가중치 E2E `evaluate` | exit 0, `reliability` 키 11개 생성, 임시 산출물 정리 확인. 성능 수치 아님 |
| 수정 전 pytest | **89 passed, 1 failed** — 문서의 83개 표기와 불일치 확인 |

### 변경 전후

- 수정 전: 90개 수집, 89 통과, 1 실패(OpenCV 얇은 구조).
- 수정 후: 110개 수집, **110 통과** (최초 14.69s, 재실행 8.95s). 증가분 20개는 신뢰도 16개, 평가 확률 1개, 준비 실패 경로 2개, OpenCV 얇은 구조 1개다.
- 이 수치는 **합성 fixture·CPU 테스트** 결과다. 실제 데이터 정확도·GPU 학습·Streamlit 부팅을 새로 측정한 것이 아니다.

### 진입점별 커버리지

| 진입점 | 이번 검증 | 남은 확인 |
| --- | --- | --- |
| 데이터 준비 | RAR/7z 조기 거부, ZIP 실패 정리, 빈 디렉터리 거부, NEU 계약 | 실제 1,800장 재수신 |
| 학습 | help, 기존 artifact 보호, 인자 검증(코드) | 실제 GPU 학습 재실행 |
| 평가 | 확률 shape·정규화, 신뢰도 계약, manifest SHA-256 | 실제 checkpoint로 ECE/Brier 산출 |
| 벤치마크 | 기존 테스트 유지 | 실제 CPU/GPU 재측정 |
| 데모 | 기존 로직·테스트 유지 | Streamlit 부팅, 스크린샷 |
| 문서 | 110개 수 동기화, 신뢰도 문서 | 사용자 폴더명 결정 |

### 남은 위험과 열린 질문

1. **캘리브레이션 독립성:** ECE를 test set에서 계산하면 calibration과 evaluation이 같은 표본이다. 독립 holdout이 없으므로 "보정됐다"가 아니라 "관측됐다"로만 읽어야 한다.
2. **bootstrap의 범위:** 270장 test set의 표본 변동만 나타낸다. split·seed·도메인 변동이 아니다.
3. **블러 방어 부재:** 품질 게이트는 심한 노이즈만 잡고 블러는 잡지 못한다. 기존 한계가 그대로 남는다.
4. **환경 차이:** 이번 검증은 CPU wheel·임시 Python이다. CUDA 환경에서 동일 110개가 통과하는지는 별도 확인이 필요하다.
5. **문서 수치의 원출처:** 1.0000·지연시간·OOD 표는 이 체크아웃의 산출물이 없어 재검증하지 못했다. 기존 기록을 historical evidence로 유지한다.
6. **사용자 작업:** 데모 결과 스크린샷, 상위 폴더명 변경 여부.

### 인터뷰 피드백

- 코드로 답할 수 있는 질문(테스트 수, 의존성, 마스크 동작)을 먼저 확인한 것이 효과적이었다. 문서만 읽었다면 B1/C1을 놓쳤을 것이다.
- `[Owner-states]`를 분리하지 않았다면 "110 passed"와 "1.0000"이 같은 신뢰도로 섞였을 위험이 있다.
- 다음 프로젝트에서는 테스트 수를 README에 수동 기입하지 말고, CI 또는 스크립트가 문서 수치를 검사하는 편이 낫다.

---

## 변경 파일 요약

| 구분 | 파일 |
| --- | --- |
| 신규 코드 | `02_구현/src/defect_cls/reliability.py` |
| 수정 코드 | `evaluate.py`, `opencv_pipeline.py`, `preparation.py`, `train.py` |
| 신규 테스트 | `tests/test_reliability.py` |
| 수정 테스트 | `tests/test_evaluate.py`, `tests/test_data.py`, `tests/test_opencv_pipeline.py` |
| 의존성 | `requirements.txt` |
| 신규 문서 | `02_구현/docs/reliability-evaluation.md`, 이 문서 |
| 동기화 문서 | 루트 README, `02_구현/README.md`, `docs/experiment-results.md`, `docs/portfolio-summary.md`, `04`, `05`, `07` |

## 인수자용 다음 명령

```bash
cd 02_구현
pip install -r requirements.txt
pip install -e .
python -m pytest -q -p no:cacheprovider --basetemp="$env:TEMP\opencode\pytest-tmp"

# 체크포인트가 있는 환경에서만: 기존 지표 + 신뢰도 블록 생성
python -m defect_cls.evaluate --checkpoint data/artifacts/baseline/seed-42/checkpoint.pt --device cuda
```

실행 후 `test-metrics.json`의 `reliability` 블록을 `docs/reliability-evaluation.md`의 해석 규칙에 따라 읽는다. 이 문서는 그 전까지 **수치를 인용하지 않는다.**
