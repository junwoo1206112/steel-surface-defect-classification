# 04. 결정 원장

| 축 | 현재 결정 | 근거 | 제약 / 제외 범위 | 검증 방법 | 상태 |
| --- | --- | --- | --- | --- | --- |
| 문제 | 공개 데이터 기반 산업 표면 결함 이미지 분류 | 비전 AI 직무용 재현 가능한 프로젝트 | 실제 현장·반도체 장비 성능 주장 금지 | README·데모·실험 문서 일치 | 확정 |
| 데이터 | NEU-CLS 사용 (1,800장 수신, 중복 1장 배제 → 1,799장) | 2026-09-09 정합성 검사 통과(200×200, 6클래스, 손상 0) | 상업 이용·원본 재배포 금지, 중복 1쌍(Pa_101=Pa_105) 배제 기록 | `data-quality.json`, RAR SHA-256 기록 | 확정 |
| 환경 | CUDA 빌드 torch 2.13.0+cu126 사용, RTX 3070 확인 | 재설치 후 `cuda.is_available()=True` 실측 | CUDA 실패 시 CPU 폴백 | 재실행 명령 | 확정 |
| 표기 | 프로젝트명 `산업 표면 결함 이미지 분류` 고정 | 데이터가 강재임을 확인 | `반도체`·`탐지(검출)` 표기 금지, `분류`로만 표기 | README·문서 전수 검토 | 확정 |
| 모델 | ResNet18 전이학습 (IMAGENET1K_V1) | 2주 내 기준선·비교 실험 가능 | YOLO/세그멘테이션 제외 | config·학습·평가 재실행 | 확정 |
| 비교 | 증강 없음 vs 기본 증강 | 직무 관련 실험 설계 증거 | **우위 주장 금지 — test 지표 포화(동일 1.0000), 수렴 속도 차이만 관찰** | 동일 test set 지표 비교 | 확정 |
| 품질 | Accuracy + macro F1 + 클래스별 지표 + seed 고정 재실행 | 불균형 은폐 방지, 재현성 증명(재실행 완전 일치 확인) | 측정 결과만 공개 | metric JSON·confusion matrix | 확정 |
| 성능 | 단일 이미지 추론 측정: CPU p50 ~10.4ms / GPU p50 ~1.65ms | 실측(warm-up 10, 200회, 실이미지) | GPU/NPU 최적화·실시간 표현 금지, 실행별 변동 명시 | benchmark.json | 확정 |
| 임계값 | 신뢰도 임계값 0.60 유지 | val 최저 신뢰도 0.4995(baseline) 측정 — 보수 동작 확인 | 임의 상수로 보이지 않게 근거 기록 | confidence-scan-val.json | 확정 |
| UI | Streamlit 판단 보조 화면 | headless 부팅 HTTP 200 확인, 로직 테스트 커버 | 품질 보증·불량 확정 표현 금지 | 정상/오류/저신뢰도 테스트·부팅 확인 | 확정 |
| 보조 실험(입력 채널) | RGB 3채널 복제 유지 | 1채널 test 0.9963 vs 3채널 1.0000 — 1장 차이, 단일 실행이라 우열 주장 불가 | 1채널 교체 이득 근거 없음 | `grayscale1ch/test-metrics.json` | 확정 |
| 강건성 | 신뢰도 게이트는 '정상 품질 입력의 보수 장치'로만 유효 | OOD 스캔에서 과신 확인(붕괴 정확도에도 신뢰도 유지) | 훼손 입력 걸러내기 장치로 표현 금지, 개선 과제 명시 | `ood-robustness.json` | 확정 |
| 벤치마크 방법론 | 수치 인용 시 측정 조건(시스템 상태) 기록 의무 | 세션 간 CPU/GPU p50 4~5배 변동 실측(게임 부하 중) | 조건 없는 절대 속도 인용 금지 | `benchmark-variance.json` + 두 조건 병기 | 확정 |
| 확률 보정 | 기본 T=1.0 유지 | val 피팅 T=0.5는 NLL은 낮추나 OOD 과신 악화(T별 플래그 표) | "보정했으니 안전" 주장 금지 | `temperature.json`, `ood-robustness-T*.json` | 확정 |
| 입력 품질 게이트 | 노이즈>12.0, 밝기 [40,245], 선명도 플래그 비활성 | clean 오경보 0.4%, σ30 99.3% 차단 실측. 블러는 지표 분리 불가(텍스처) | 게이트를 완전한 안전장치로 표현 금지 | `quality-gate-calibration.json` | 확정 |
| 벤치마크 산출물 보호 | 기존 benchmark.json 존재 시 --force/--out 없이는 덮어쓰기 금지 | 산출물 덮어쓰기 사건 재발 방지 | 첫 실행 명령은 불변 | `check_overwrite` 테스트 | 확정 |
| OpenCV 입력 진단 | OpenCV를 모델과 분리한 입력 검증·진단·시각화 계층으로 추가 | 제조 비전 지원 시 OpenCV 구현 증거를 남기되 기존 모델 측정 계약을 보존 | 객체 탐지·위치 추정·세그멘테이션·현장 품질 보증은 제외 | 합성 fixture 테스트, Streamlit 화면, `docs/opencv-pipeline.md` | 확정 |
| OpenCV 후보 마스크 | 3×3 opening이 후보의 10% 미만만 남기면 opening 전 마스크를 유지 | 1픽셀 얇은 스크래치·경계가 opening에서 전부 사라지는 결함 실측(132px→0px) | 위치 예측·탐지 성능 주장 아님 | 1픽셀 선 보존 테스트, 기존 합성 마스크 테스트 | 확정 |
| 테스트 기준선 | 2026-11-14 전체 pytest **128개 통과**로 문서 동기화 | 1차 수정 후 110개, 인수 상태 완료 판정 테스트 2개를 추가해 120개, 업로드 검토 우선순위 테스트 8개를 추가해 128개. 수정 전 90개 수집·89 통과·1 실패, 기존 83개 표기는 stale | CPU·합성 fixture 결과이며 GPU/실데이터 reliability는 별도 결과 파일로 기록 | `pytest`, `compileall`, `pip check`, `handoff_status.py` | 확정 |
| 인수인계 상태 | `scripts/handoff_status.py`가 data/raw·manifest·checkpoint·screenshot·폴더명·문서 수를 읽기 전용으로 판정 | 다른 인수자/LLM이 산문 대신 실제 파일 상태와 다음 안전한 명령을 확인 | 상태를 변경하지 않음, 학습·평가·다운로드 없음 | `test_handoff.py` 8개, 실제 실행 출력 | 확정 |
| 잔여 작업 완료 판정 | 데모 스크린샷·폴더명·실제 reliability 실측 모두 완료 | 스크린샷 PNG SHA-256과 evidence 일치; 현재 폴더명이 기준과 일치; RTX 3070 baseline 평가의 `test-metrics.json` 존재 | reliability는 고정 test split 진단으로만 해석 | `handoff_status.py`, checkpoint/metrics SHA-256 | 확정 |
| 의존성 | `numpy==2.2.6` (opencv-python-headless 4.12의 `numpy<2.3` 요구와 호환) | numpy 2.5.2 핀과 opencv 4.12의 메타데이터 충돌로 한 줄 설치 재현 불가 | CUDA torch 휠 설치 자체는 이 환경에서 미검증 | `pip check`, numpy/opencv 동시 import | 확정 |
| 준비 파이프라인 원자성 | RAR/미지원 압축은 manifest 작성 전 거부; ZIP 실패 시 추출 디렉터리 정리; 빈 입력 거부 | 지연 RAR 거부가 빈 manifest를 남기고, ZIP 실패가 빈 디렉터리를 남기는 경로 확인 | RAR 자동 해제는 계속 미지원(안전상 수동 해제 요구) | `test_preparation_rejects_*`, `test_zip_path_traversal_is_rejected` | 확정 |
| 신뢰도 평가 | `evaluate`가 ECE/MCE·Brier·NLL·top-3·risk-coverage·bootstrap CI를 기록하고 RTX 3070 baseline test에서 ECE 0.003993/Brier 0.002065/NLL 0.004671을 산출 | 포화 정확도만으로 확신·보류·표본 변동을 설명할 수 없음 | 독립 캘리브레이션·일반화 증거 아님, bootstrap은 고정 test set 한정 | `test-metrics.json`, `test_reliability.py`, `docs/reliability-evaluation.md` | 확정 |

## 갱신 규칙

- `보류` 항목은 사실 확인 후에만 `확정`으로 바꾼다.
- 결과가 예상과 다르면 원장을 과거 결과에 맞추지 않고, 결과와 한계를 함께 남긴다.
