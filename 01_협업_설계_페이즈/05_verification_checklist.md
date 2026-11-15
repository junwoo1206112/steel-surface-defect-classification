# 05. 검증·공개 전 점검표

> 2026-09-09 기준으로 실제 확인한 결과만 표시한다. 아래의 과거 실행 기록은 당시 상태를 보존하며, 현재 상태는 문서 끝의 최신 상태 갱신을 따른다.

## 데이터와 권리

- [x] 공식 출처와 라이선스를 기록했다. (`02_구현/docs/data-governance.md` — 동북대학, 학술 연구 전용, 인용 조건)
- [x] 데이터 실제 도메인과 프로젝트 제목이 일치한다. (강재 → `산업 표면 결함 이미지 분류`, README 첫 문장에 반도체 아님 명시)
- [x] 원본 이미지와 가중치를 Git에 넣지 않았다. (`data/raw/`, `*.pt` → `.gitignore`)
- [x] train/validation/test 분할과 seed를 기록했다. (seed=42, 1259/270/270, `manifest.csv`)

## 모델과 실험

- [x] 설치 명령이 동작한다. (`pip install -e .` 통과 — 완전 새 환경 검증은 배포 시 남음)
- [x] 데이터 로더와 클래스 매핑 테스트가 통과한다. (약어 Cr/In/Pa/PS/RS/Sc 포함 128개 테스트, 2026-11-14)
- [x] 기준 모델과 증강 모델을 같은 test set으로 평가했다. (n=270, 둘 다 Accuracy 1.0000)
- [x] Accuracy, macro F1, 클래스별 지표, confusion matrix를 실제 생성했다. (`test-metrics.json`, `confusion-matrix.csv/png`)
- [x] 결과가 나쁘거나 증강이 악화한 경우도 숨기지 않았다. (증강 우위 없음·수렴 지연 사실을 리포트에 명시)
- [x] seed 고정 재실행으로 재현성을 확인했다. (epoch별·test 지표 완전 일치)

## 추론과 UI

- [x] 추론 시간에 환경·이미지 크기·warm-up·반복 수가 적혀 있다. (benchmark.json + experiment-results.md)
- [x] 정상 이미지, 비이미지, 손상 이미지, 낮은 신뢰도 흐름을 확인했다. (단위 테스트 + headless 부팅 HTTP 200)
- [x] UI가 실제 현장 검사나 불량 확정으로 오해되지 않는다. (화면 상단 고지 문구 + `판단 보조` 표기)
- [x] 임계값이 근거를 가진다. (val 최저 신뢰도 0.4995 측정, `confidence-scan-val.json`)

## 포트폴리오 문구

- [x] `공개 산업 이미지 데이터 기반`이라고 명시했다.
- [x] 실제 반도체 데이터가 아니면 `반도체`라고 쓰지 않았다. (문서 전체에서 `분류`만 사용)
- [x] TensorRT, NPU, 장비 연동, 현장 성능을 측정 없이 쓰지 않았다.
- [x] README, 실험 리포트, 포트폴리오 요약의 수치가 모두 일치한다. (동일 출처 JSON에서 인용)

## OpenCV 입력 진단 (Phase 7)

- [x] `opencv-python-headless` 설치 후 OpenCV 파이프라인 테스트를 실행한다. (2026-11-14: 4.12.0.88에서 실행. 1픽셀 얇은 구조를 지우는 opening 결함을 발견·수정하고 회귀 테스트 추가)
- [x] 모델 입력 경로(PIL → tensor)는 OpenCV 전처리 결과와 분리했다.
- [x] 문서·UI에서 후보 강조를 `전처리 기반 후보 강조(모델의 위치 예측 아님)`으로 표기하고 객체 탐지·세그멘테이션 주장을 제외했다.
- [x] 인수 파일·완료 게이트·비목표를 Phase 7에 기록했다.

## 신뢰도·재현성 (2026-11-14)

- [x] `evaluate`가 ECE/MCE·Brier·NLL·top-3·risk-coverage·bootstrap CI를 `test-metrics.json`의 `reliability`에 기록한다. (알려진 값 테스트 16개)
- [x] requirements의 numpy/opencv 제약이 호환된다. (`numpy==2.2.6`, `pip check` 정상)
- [x] 준비 파이프라인 실패가 빈 manifest나 빈 추출 디렉터리를 남기지 않는다. (RAR/7z 조기 거부, ZIP 실패 정리, 빈 입력 거부 테스트)
- [x] `compileall`·전체 pytest·`evaluate --help`를 실행했다. (128 passed)
- [x] `scripts/handoff_status.py`가 data/raw·manifest·checkpoint·screenshot·folder·docs 상태를 읽기 전용으로 보고한다. (2026-11-14, 테스트 8개 추가)

## 남은 항목 (선택)

- [x] 현재 OpenCV 데모 스크린샷 캡처 — `baseline · seed 42`와 `Cr_1.bmp`를 실제 업로드해 원본·CLAHE·후보 강조·예측·4개 품질 지표가 함께 보이는 화면을 `docs/screenshots/demo-result-baseline-seed42-cr1-opencv.png`에 저장했다. SHA-256 `023692E71F0F0DEAF827F0853E49F88E1B175626BB3A3D061D23A941A0F8C989`; legacy 캡처는 보관본으로 유지한다.
- [x] 상위 작업 폴더명 — 현재 저장소 경로가 `steel-surface-defect-classification`으로 이미 강재·분류 표기와 일치한다. 외부 링크 영향 없이 확인만 했다.
- [x] 완전 새 환경(가상환경)에서 설치 재검증 (2026-09-09: 4개 성공 기준 모두 통과 — `06_전문가_인터뷰_기록.md` 신규 가상환경 절 참조. requirements.txt에 `--extra-index-url` 추가로 설치 재현성 결함 수정)
- [x] 실제 checkpoint로 `reliability` 수치 산출 — 공식 NEU-CLS 재검증·RTX 3070 baseline seed=42 평가 완료: ECE 0.003993, Brier 0.002065, NLL 0.004671 (`data/artifacts/baseline/seed-42/test-metrics.json`).

## 최신 상태 갱신 (2026-11-14)

- 2026-11-14 후속 ABC 교차 검증에서 수정 전 pytest가 **90개 수집·89 통과·1 실패**임을 확인했다. 기존 문서의 83개 표기는 stale이었고, 실패는 OpenCV 후보 마스크가 1픽셀 얇은 구조를 지우는 결함이었다.
- 최신 후속 검증에서 전체 pytest는 **128개 통과**다. 실제 `baseline · seed 42` CUDA reliability와 현재 OpenCV 화면 캡처도 완료했으며, `scripts/handoff_status.py --strict`가 경로·SHA-256 일치까지 확인한다. CPU·합성 fixture 테스트와 GPU/실데이터 평가 결과는 구분해 해석한다.
- 상세 기록: `08_ABC_교차검증_및_구현.md`, 잔여 작업 검증: `09_잔여작업_ABC_교차검증.md`.
- **완료:** 데모 스크린샷(증빙·해시 확인), 폴더명(현재 경로가 기준과 일치), 문서 테스트 수 동기화.
- **남은 작업:** 현재 계획의 필수 구현·실측 작업은 모두 완료됐다. reliability는 고정 test split 진단이므로 독립 split·다중 seed·외부 도메인 평가는 후속 연구로만 취급한다.
