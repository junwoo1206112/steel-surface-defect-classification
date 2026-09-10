# 산업 표면 결함 이미지 분류 (Industrial Surface Defect Classification)

> **본 프로젝트는 반도체 데이터가 아닙니다.** 공개된 열간 압연 강재(hot-rolled steel strip) 표면 결함 데이터(NEU-CLS, Northeastern University)를 사용한 비영리 학습·포트폴리오 실험입니다. 실제 생산라인, 검사 장비, 실시간 품질 보증 성과를 나타내지 않습니다.

공개 산업 표면 결함 이미지로 재현 가능한 PyTorch 분류 파이프라인을 구현하고, 데이터 검증 → 누수 방지 분할 → 기준 모델 → 증강 비교 → 추론 벤치마크 → 업로드 데모까지 동일 조건으로 측정했습니다.

> **3줄 요약**: 강재 공개 벤치마크에서 데이터 누수 점검·재현 가능한 분류 실험·합성 훼손 상황의 과신 실패 분석을 구현했습니다. 1.0000은 단일 split/seed의 포화 벤치마크 관측값이며 현장 성능이 아닙니다. 데모의 신뢰도·품질 게이트는 실험적 판단 보조 신호입니다.

## 핵심 결과 (2026-09-09 실측, 전부 재현 가능)

| 항목 | 결과 |
| --- | --- |
| 데이터 | NEU-CLS 1,800장 검증(200×200, 6클래스×300) 중 완전 중복 1장 제외 → 1,799장 사용 |
| 분할 | stratified 70/15/15, seed=42 (train 1,259 / val 270 / test 270), test 비사용 원칙 |
| 모델 | ResNet18 ImageNet 전이학습, SGD+Cosine, 20 epochs, batch 32 |
| test 성능 | baseline: Accuracy 1.0000, macro F1 1.0000 · augmented: 동일 (혼동행렬 비대각 0) |
| 증강 비교 해석 | **증강의 우위를 주장하지 않음.** 벤치마크 포화로 test 지표 동일, 관찰 차이는 수렴 속도뿐(best epoch 5 vs 12) |
| 입력 채널 비교 | 1채널(conv1 평균 초기화) test 0.9963 vs 3채널 복제 1.0000 — 차이 1장, 단일 실행이므로 우열 주장 금지. 기본은 3채널 복제 유지 |
| 합성 훼손 스트레스 테스트 | 같은 NEU-CLS validation 이미지에 노이즈·블러·밝기 변형을 적용했을 때 정확도 46.7%·38.2%로 붕괴해도 **평균 신뢰도 0.95·0.78 유지(과신)** — 외부 OOD·도메인 일반화 평가는 아님 |
| 확률 보정 | val 피팅 T=0.5는 NLL을 낮추지만 합성 훼손 상황의 과신을 악화(T별 플래그 비율 표 측정) → 기본 T=1.0 유지 |
| 입력 품질 게이트 | 같은 validation set에서 탐색적으로 보정·측정: clean 플래그 0.4%, 심한 노이즈 σ30 플래그 99.3%. 독립 운영 성능이 아니며 블러는 분리 불가 |
| 재현성 | seed 고정 재실행에서 epoch별·test 지표 완전 일치 확인 |
| 추론 (batch=1) | 시스템 상태에 크게 좌우: 저부하 세션 CPU p50 10.4~11.7ms / GPU 1.65~1.87ms, 부하 경합 세션 CPU ~50.7ms / GPU ~6.2ms. 실행 조건을 함께 기록해야 함 |
| 임계값 근거 | baseline val 최저 신뢰도 0.4995 → 임계값 0.60은 정답 1건을 재검토 플래그하는 보수 설정 |
| 자동 테스트 | 83개 통과 (현재 코드 기준, 합성 fixture·CPU smoke test 포함) |

자세한 수치·측정 조건·한계: `docs/experiment-results.md` · 데이터 권리·중복 처리: `docs/data-governance.md` · 채용 담당자용 요약: `docs/portfolio-summary.md`

## 저장소 구조

```
02_구현/
  src/defect_cls/     data(데이터셋·증강) model train evaluate inference benchmark preparation
  scripts/            prepare_data.py(검증·분할·EDA) confidence_scan.py(임계값 근거)
  app/demo.py         Streamlit 판단 보조 데모
  tests/              83개 자동 테스트 (합성 fixture, 실데이터 불필요)
  docs/               data-governance, experiment-results, portfolio-summary
  data/               raw(Git 제외), processed(manifest·품질보고서), artifacts(체크포인트·지표)
```

## 실행 방법

요구사항: Python 3.12, CUDA GPU(선택 — CPU만으로도 학습 가능하지만 느림), NEU-CLS ZIP 또는 미리 해제한 디렉터리.

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest -q -p no:cacheprovider --basetemp="$env:TEMP\opencode\pytest-tmp"
```

> 참고: Windows에서 pytest 기본 임시폴더에 권한 오류(WinError 5)가 있는 환경이므로 `--basetemp`를 지정한다.

```bash
# 1) 데이터 준비: 다운로드한 ZIP 또는 미리 해제한 폴더를 data/raw/ 아래에 둔다.
#    입력은 data/raw/ 밖을 허용하지 않으며 NEU-CLS 계약(1,800/6x300/200x200)을 강제한다.
python scripts/prepare_data.py --input data/raw/<NEU-CLS.zip 또는 해제 폴더>

# 2) 학습 (baseline = 증강 없음, augmented = 기본 증강)
python -m defect_cls.train --experiment baseline --seed 42 --device cuda
python -m defect_cls.train --experiment augmented --seed 42 --device cuda

# 3) test 평가 (혼동행렬·클래스별 지표 산출)
python -m defect_cls.evaluate --checkpoint data/artifacts/baseline/seed-42/checkpoint.pt --device cuda
python -m defect_cls.evaluate --checkpoint data/artifacts/augmented/seed-42/checkpoint.pt --device cuda

# 4) 단일 이미지 추론 벤치마크 (CPU/GPU 병기)
#    기존 benchmark.json이 있으면 --force 또는 --out이 필요하다(덮어쓰기 방지).
#    --devices에는 사용 가능한 cpu/cuda만 지정하며, 요청 장치가 없으면 실패한다.
python -m defect_cls.benchmark --checkpoint data/artifacts/baseline/seed-42/checkpoint.pt --image data/raw/<이미지> --devices cpu,cuda

# 5) 임계값 근거 측정 + 보정/강건성/품질게이트 분석
python scripts/confidence_scan.py --checkpoint data/artifacts/baseline/seed-42/checkpoint.pt
python scripts/calibrate_temperature.py --checkpoint data/artifacts/baseline/seed-42/checkpoint.pt
python scripts/ood_scan.py --checkpoint data/artifacts/baseline/seed-42/checkpoint.pt
python scripts/quality_gate_calibration.py --checkpoint data/artifacts/baseline/seed-42/checkpoint.pt

# 6) 안전 로드 CPU smoke test / 이미 완료된 다중-seed 결과만 집계
python scripts/cpu_smoke_test.py --checkpoint data/artifacts/baseline/seed-42/checkpoint.pt
python scripts/aggregate_seed_metrics.py --metrics data/artifacts/baseline/seed-<seed>/test-metrics.json [...]

# 7) 데모
streamlit run app/demo.py
```

## 데모 동작 (판단 보조 원칙)

- 기본 `manifest`·`data/raw`·`data/artifacts` 경로는 프로젝트 위치를 기준으로 해석하므로, 설치 후에는 현재 작업 디렉터리에 의존하지 않는다. CLI에 상대 경로를 줄 때도 프로젝트 루트를 기준으로 해석한다.
- 업로드 형식(jpg/jpeg/png/bmp)·파일 크기(≤10MB)·픽셀 수(≤20,000,000)·이미지 유효성을 검사하고, 손상 파일은 오류로 안내한다.
- 예측 클래스와 신뢰도를 표시하며, **신뢰도가 임계값(기본 0.60) 미만이면 예측을 확정하지 않고 `판단 보조 결과 — 재검토 필요`를 표시한다.**
- **입력 품질 게이트**: 업로드 이미지의 노이즈·밝기 지표가 실험용 임계값을 벗어나면 신뢰도와 무관하게 재검토로 표시한다. 임계값은 200×200 validation 이미지에서 탐색적으로 정한 보조 신호이므로, 임의 해상도·압축 형식 업로드의 운영 품질을 보증하지 않는다.
- UI는 실제 검사 장비나 불량 확정 도구가 아님을 화면에 명시한다.
- 데모는 `data/artifacts/<experiment>/checkpoint.pt`의 legacy 체크포인트와 `data/artifacts/<experiment>/seed-<seed>/checkpoint.pt`의 seed별 체크포인트를 모두 표시한다. 사용자가 라벨(실험·seed)을 명시적으로 선택하기 전에는 모델을 로드하지 않는다.

## 데이터 라이선스·인용

- NEU surface defect database (NEU-CLS): Northeastern University, Kechen Song · Yunhui Yan. 제공 페이지에 기록된 학술 연구 목적·상업적 이용 제한·논문 인용 조건을 따른다. 실제 이용·공개·상업적 활용 전에는 제공 기관의 최신 조건을 별도로 확인해야 한다.
- 인용: K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," *Applied Surface Science*, vol. 285, pp. 858-864, 2013.
- 원본 데이터·압축파일·학습 가중치는 Git에 포함하지 않는다(`.gitignore`).

## 한계 (정직한 공개)

- 6클래스 벤치마크가 전이학습에 쉬워 단일 split·seed(test n=270) 지표가 포화(1.0000)됐다. 이 수치는 코드·절차 검증용이며 실무 검출 성능·일반화 성능이 아니다.
- 증강은 이 데이터에서 수렴을 늦추는 방향으로 관찰됐다. "증강이 좋다"는 주장은 하지 않는다.
- 같은 validation 이미지에 적용한 합성 노이즈·블러에서 모델이 과신한다. 이는 외부 OOD 평가가 아니며, 입력 품질 게이트는 심한 노이즈만 탐색적으로 플래그하고 블러는 분리하지 못한다. 실질 방어는 추가 연구 과제다.
- val 기준 확률 보정(temperature scaling)은 합성 훼손 상황의 과신을 완화하지 않았다(val 피팅 T=0.5는 오히려 확신을 강화). 보정·게이트 모두 독립 holdout 평가 전에는 운영 안전성 증거가 아니다.
- 추론 시간은 시스템 상태에 크게 좌우된다: 같은 조건에서 세션 간 CPU p50이 10.4ms(저부하) vs 50.7ms(부하 경합, 게임 실행 중)로 4~5배 달랐다. 수치 인용 시 측정 조건을 함께 명시해야 한다.
- 그레이스케일 이미지를 3채널로 반복해 RGB 전이학습에 투입했다. 1채널 교체 대안을 같은 조건으로 학습해 비교했고(0.9963 vs 1.0000, 차이 1장), 근거가 없어 기본 경로를 유지한다.

## 설계·결정 기록

협업 설계·전문가 교차 검증·결정 원장은 상위 폴더 `../01_협업_설계_페이즈/`에 있다. 모든 표기 원칙(`분류` 고정, `반도체`·`탐지` 금지, 측정 없는 주장 금지)는 해당 원장을 따른다.
