# 09. 잔여 작업 A·B·C 교차 검증과 인수인계 상태 도구 (2026-11-14)

기준 문서: `05_verification_checklist.md`, `07_현재_교차검증_정리.md`, `08_ABC_교차검증_및_구현.md`. 이 문서는 **Ouroboros 인터뷰 스킬 Path B**를 다시 적용해 “남은 작업”을 재검증한 기록이다.

> 2026-11-14 최신 갱신: legacy 스크린샷은 보관본이다. 현재 OpenCV UI는 `baseline · seed 42`와 실제 `Cr_1.bmp` 업로드로 새 캡처를 완료했으며, `scripts/handoff_status.py --strict`가 증빙의 파일 경로·SHA-256까지 검증한다. 최신 테스트 기준선은 **128 passed**다.

- 검증 환경: 저장소 밖 임시 Python 3.12.10, torch 2.13.0 CPU wheel, numpy 2.2.6, opencv-python-headless 4.12.0.88, pytest 8.4.2.
- 실제 GPU: RTX 3070 8GB 확인(`nvidia-smi`).
- **중요:** `data/raw` 이미지 **0장**, checkpoint **0개**. 따라서 실제 학습·reliability 실측은 이 환경에서 수행할 수 없다.
- 이 문서는 새 성능 수치를 만들지 않는다.

## Q1 — 남은 작업은 무엇인가

| # | 문서상 남은 작업 | 실제 상태 | 판정 |
| --- | --- | --- | --- |
| 1 | 데모 결과 화면 스크린샷 | `02_구현/docs/screenshots/demo-result-baseline-legacy-cr1-cropped.png` 존재. `docs/demo-capture-evidence.md`에 캡처 조건·SHA-256 기록. 해시 `2BD9A10D…996C2` 일치 확인 | **완료** (체크리스트가 stale) |
| 2 | 상위 작업 폴더명 변경 | 현재 저장소 경로가 `C:\AI\steel-surface-defect-classification`으로 이미 강재·분류 표기와 일치 | **완료** (변경 불필요) |
| 3 | 실제 checkpoint로 `reliability` 실측 | checkpoint 0개, `data/raw` 0장. `manifest.csv` 1,799행의 이미지 파일도 모두 누락 | **blocked** |
| 4 | 문서 테스트 수 동기화 | 1차 수정 후 110개, 인수인계 점검 테스트 8개 추가 후 118개 | **완료** (118로 갱신) |

## Q2 — 무엇이 잘못될 수 있는가

### 전문가 A — 기술 아키텍트

| ID | 발견 | 근거 | 심각도 |
| --- | --- | --- | --- |
| A1 | “남은 작업”이 산문에만 있어, 다른 인수자/LLM이 실제 파일 상태와 다른 결정을 내릴 수 있다. 예: 스크린샷이 이미 있는데도 다시 캡처하거나, 데이터가 없는데 평가를 시도한다. | 체크리스트·AGENTS·README가 서로 다른 완료 상태를 말함 | **높음** |
| A2 | 상태 판정이 자동화되지 않아 테스트 수(110/118), checkpoint 유무, `data/raw` 유무가 문서 갱신 때마다 어긋난다. | 83→90→110→118 변동 이력 | 중간 |
| A3 | `streamlit` 미설치 환경에서 데모 실행을 “가능”으로 오판할 수 있다. | 임시 환경에서 `import streamlit` 실패 | 중간 |

**A의 판정:** 사람이 읽는 문서만으로는 부족하다. **읽기 전용 상태 점검기**가 필요하다. 학습·평가·다운로드를 실행하지 않고, 현재 파일과 환경만 보고 다음 안전한 명령을 제시해야 한다.

### 전문가 B — 포트폴리오 심사자

| ID | 발견 | 근거 | 심각도 |
| --- | --- | --- | --- |
| B1 | 스크린샷 증빙이 이미 존재하는데 체크리스트는 미완료로 표시한다. “할 수 있는데 안 한 것”처럼 보여 신뢰를 깎는다. | `05`의 `[ ]` vs `docs/demo-capture-evidence.md` | **높음** |
| B2 | 폴더명도 현재 경로가 이미 기준과 일치하는데 “사용자 결정 대기”로 남아 있다. | 01 README·AGENTS의 남은 작업 목록 | 중간 |
| B3 | 실제 reliability 수치가 없는 상태에서 “신뢰도 평가를 구현했다”는 표현이 실측처럼 읽힐 수 있다. | 08·README의 신뢰도 행 | 중간 |

**B의 판정:** 완료는 완료로, blocked는 blocked로 **한곳에서 일관되게** 보여야 한다. 스크린샷은 재캡처가 아니라 증빙·해시 검증으로 닫고, reliability는 “도구는 있음 / 실측은 없음”을 분리해 쓴다.

### 전문가 C — 재현성·QA·안전성

| ID | 발견 | 근거 | 심각도 |
| --- | --- | --- | --- |
| C1 | 데이터·checkpoint 없이 `evaluate`를 실행하면 실패하거나, 합성/무작위 결과를 실제 성능처럼 오인할 위험이 있다. | `data/raw` 0장, checkpoint 0개 | **높음** |
| C2 | 상태 점검이 파일을 수정하거나 네트워크를 사용하면 인수인계 도구 자체가 위험이 된다. | 요구사항: 읽기 전용·결정론 | **높음** |
| C3 | `--strict`를 기본으로 하면 정상적인 “학습 전” 상태에서도 CI가 실패해 사용성이 나빠진다. | blocked는 정상 상태일 수 있음 | 중간 |
| C4 | 스크린샷 해시를 검증하지 않으면 다른 이미지로 교체돼도 알 수 없다. | evidence에 SHA-256 기록됨 | 중간 |

**C의 판정:** 상태 도구는 **읽기 전용**, **네트워크 없음**, **산출물 없음**이어야 한다. blocked는 실패가 아니라 정상 상태다. `--strict`는 선택 옵션으로 두고, 스크린샷은 해시를 함께 보고한다.

### 교차 반박과 합의

| 쟁점 | A | B | C | 합의 |
| --- | --- | --- | --- | --- |
| 스크린샷 | 재캡처 불필요 | 완료로 표기 | 해시 검증 | **기존 증빙·해시 확인 후 완료** |
| 폴더명 | 현재 경로 확인 | 완료로 표기 | 경로 변경 부작용 없음 | **완료** |
| reliability 실측 | 실행하고 싶음 | 수치가 있으면 좋음 | 데이터 없으면 금지 | **blocked 유지, 데이터 도착 후 실행** |
| 상태 도구 | 자동 판정 | 한눈에 보여야 | 읽기 전용·strict 선택 | **`handoff_status.py` 구현** |
| 합성 데이터 | 데모용으로 가능 | 오해 위험 | 실제 성능 오인 | **기각** |

## Q3 — 무엇을 하기로 했는가

### 채택한 구현

| # | 변경 | 파일 | 완료 기준 |
| --- | --- | --- | --- |
| 1 | 읽기 전용 인수인계 상태 라이브러리: 이미지 수, manifest 행/누락, checkpoint, 스크린샷 해시, 폴더명, 문서 테스트 수, 선택적 환경 진단 | `src/defect_cls/handoff.py` | 파일 수정·네트워크 없음, 테스트 8개 |
| 2 | CLI: 사람용 요약/JSON, `--strict` 선택, `--out` 선택 | `scripts/handoff_status.py` | 현재 환경에서 실행, blocked/done 판정 |
| 3 | 스크린샷 완료 처리: 증빙·SHA-256 확인 | `05`, `04`, `01/README`, `AGENTS`, `07` | 체크리스트·README 일치 |
| 4 | 폴더명 완료 처리: 현재 경로가 기준과 일치 | 동일 | “변경 대기” 문구 제거 |
| 5 | 문서 테스트 수 118 동기화 | README·실험 리포트·요약·원장·체크리스트 | `handoff_status.py`의 `docs_test_count=done` |

### 하지 않은 것

- **데이터 다운로드:** NEU-CLS 라이선스·획득 경로는 사용자 결정 사항이다. 공식 페이지 우회·Kaggle 토큰 대체를 하지 않는다.
- **학습·평가 실행:** `data/raw` 0장, checkpoint 0개. 합성/무작위 결과를 실제 성능처럼 기록하지 않는다.
- **스크린샷 재캡처:** 기존 증빙이 실제 화면과 SHA-256으로 확인됐다. 재캡처는 증거를 늘리지 않는다.
- **폴더명 변경:** 현재 이름이 이미 기준과 일치한다.

### 수용 기준

1. `handoff_status.py`가 파일을 수정하지 않는다.
2. 현재 환경에서 `training`·`reliability_measurement`가 **blocked**로, 스크린샷·폴더명·문서 수가 **done**으로 나온다.
3. 전체 pytest가 통과한다.
4. 문서의 테스트 수가 실제 수와 일치한다.
5. 실제 reliability 수치는 데이터·checkpoint가 생기기 전까지 어디에도 인용되지 않는다.

## Q4 — 잘 했는가

### 실행한 검증

| 명령 | 결과 |
| --- | --- |
| `python -m pytest -q ...` | **118 passed** (17.16s 최초, 17.23s 재실행) |
| `python -m compileall -q src scripts tests app` | 성공 |
| `python scripts/handoff_status.py` | `training`·`reliability_measurement` blocked, `demo_screenshot`·`folder_name`·`docs_test_count` done |
| 스크린샷 SHA-256 | `2BD9A10D09790681E1BAD0B74AD9FEE87C009AC3828CE7A13906A173860996C2` 일치 |
| `nvidia-smi` | RTX 3070 8GB 확인 (GPU 자체는 존재) |

현재 `handoff_status.py`의 핵심 판정:

```
- [blocked] training: data/raw에 학습 이미지가 없다. NEU-CLS를 먼저 배치해야 한다.
- [blocked] reliability_measurement: checkpoint가 없다. 학습 또는 기존 checkpoint 복원이 필요하다.
- [done] demo_screenshot: 결과 화면과 증빙 문서가 모두 있다.
- [blocked] demo_runtime: streamlit 미설치   # 임시 검증 환경 기준
- [done] docs_test_count: README·실험 리포트의 테스트 수가 118로 동기화됐다.
- [done] folder_name: 프로젝트 폴더명: steel-surface-defect-classification
```

### 남은 위험

1. **실제 reliability 수치 부재:** 도구는 준비됐지만 실제 test set에서 ECE/Brier/NLL을 계산하려면 checkpoint와 `data/raw`가 필요하다.
2. **임시 환경의 streamlit 미설치:** 사용자 환경에서는 설치돼 있을 수 있다. `handoff_status.py`는 환경을 실제로 확인해 blocked/ready를 나눈다.
3. **bootstrap/ECE 해석:** 여전히 고정 test set 진단이며, 독립 캘리브레이션·도메인 일반화가 아니다.
4. **문서 수 자동 검증:** `handoff_status.py`는 “110” 문자열 존재만 확인한다. 테스트 수가 바뀌면 상수도 함께 갱신해야 한다.

### 다음 명령 (데이터가 생긴 뒤)

```bash
cd 02_구현
# 1) NEU-CLS를 data/raw 아래에 배치
python scripts/prepare_data.py --input data/raw/<NEU-CLS.zip 또는 해제 폴더>
python -m defect_cls.train --experiment baseline --seed 42 --device cuda
python -m defect_cls.evaluate --checkpoint data/artifacts/baseline/seed-42/checkpoint.pt --device cuda
python scripts/handoff_status.py --strict
```

마지막 명령이 0으로 끝나면 잔여 작업이 모두 닫힌다. `reliability` 실측값은 그때 `docs/experiment-results.md`에 별도 절로 추가한다.

## 최신 완료 갱신 (2026-11-14)

이 문서의 앞부분은 data/raw와 checkpoint가 없던 당시의 교차 검증 기록으로 보존한다. 이후 사용자 승인으로 Northeastern University 공식 NEU-CLS를 확보·검증하고, RTX 3070에서 baseline(seed=42, 20 epoch)을 학습한 뒤 test 270장을 평가했다.

- manifest: 1,800장 입력, SHA-256 중복 1장 제외, 1,799장·train/val/test 1,259/270/270, 누수 없음
- checkpoint: selected epoch 5, SHA-256 `E8EA764B55366940A2C82D655CFFBF7F412C188FA1D67D0DCE5AE19C6F3E468A`
- reliability: ECE 0.003993, Brier 0.002065, NLL 0.004671, top-3 1.0000; 2,000회 bootstrap accuracy/macro F1 95% CI [1.0000, 1.0000]
- 판정: training·reliability_measurement **done**. 값은 고정 test split의 포화 관측이며 독립 split·seed·도메인 일반화 또는 운영 안전성 증거가 아니다.
