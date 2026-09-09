# 02. Phase별 설계·구현 계획

각 Phase는 바로 다음 Phase로 넘어가지 않는다. 완료 게이트의 증거 파일 또는 실행 결과를 남긴다.

| Phase | 목표 | 주요 산출물 | 완료 게이트 |
| --- | --- | --- | --- |
| 0. 범위·권리 | 데이터와 주장 범위 고정 | `docs/data-governance.md`, 결정 원장 | 공식 출처·라이선스·도메인·제외 범위 확인 |
| 1. 데이터 검증 | 데이터 품질과 분할 확인 | EDA 표, 클래스 분포, split manifest | 손상 파일·중복·누수 검사, test 미사용 원칙 기록 |
| 2. 기준 모델 | 재현 가능한 기준선 생성 | 학습 설정, checkpoint, 결과 JSON | seed 고정 재실행, validation 기반 모델 선택 |
| 3. 증강 비교 | 증강 전후의 공정한 비교 | 실험표, confusion matrix, metric JSON | 같은 test set·동일 모델·동일 조건 비교 |
| 4. 추론 벤치마크 | 속도·메모리 한계 기록 | benchmark 결과 | 환경, 이미지 크기, warm-up, 반복 수 기록 |
| 5. 데모 UI | 업로드 기반 추론 화면 | `app/` Streamlit 코드, 스크린샷 | 정상·낮은 신뢰도·비이미지 오류 흐름 확인 |
| 6. 포트폴리오 공개 | 처음 보는 사람이 실행·검증 가능 | README, case study, tests | 실행 명령·결과·한계·라이선스가 일치 |

## Phase 0 — 데이터 선택

권장 후보는 NEU-CLS다. 이는 강재 표면 결함 **분류** 데이터이므로 실제 사용 시 반도체 데이터라고 부르지 않는다. MVTec AD는 정상 이미지 중심 학습의 이상 탐지 벤치마크라서, 본 분류 프로젝트와는 과제 정의가 다르다.

## Phase 1 — 데이터 누수 방지

- 클래스 비율을 유지하는 train/validation/test 분할을 seed=42로 생성한다.
- test set은 모델과 증강 선택에 사용하지 않는다.
- 이미지 파일 경로·클래스·split을 manifest(CSV/JSON)로 고정한다.

## Phase 2~3 — 실험 계약

- Baseline: ResNet18 전이학습, 증강 없음 또는 최소 resize/normalize.
- Compare: 동일 ResNet18, 기본 증강(horizontal flip, 작은 rotation 등; 실제 적용값 문서화).
- 바꿀 수 있는 것은 한 실험에서 하나만 둔다. epoch, input size, seed, optimizer, learning rate는 표에 기록한다.
- 증강이 악화하면 결과를 숨기지 않고, 기본 모델을 최종 후보로 둘 수 있다.

## Phase 4 — 벤치마크 계약

- 단일 이미지에 대해 warm-up 후 여러 번 반복한다.
- 평균뿐 아니라 p50/p95 또는 평균·표준편차 중 실제 계산한 값을 기록한다.
- CPU/GPU 모델명, PyTorch 버전, 이미지 크기, batch size=1을 기록한다.
- 결과를 `실시간` 또는 `현장 성능`이라 부르지 않는다.

## Phase 5 — UI 계약

- 허용 형식과 최대 파일 크기를 검사한다.
- 결과에는 클래스, 신뢰도, 모델/실험 버전을 표시한다.
- 신뢰도 임계값 미만 또는 예외 발생 시 예측을 확정하지 않고 재검토·오류 안내를 표시한다.

## Phase 6 — 제출물

- `README.md`: 문제, 데이터, 실행법, 핵심 비교 결과, 데모, 한계.
- `docs/experiment-results.md`: 실험 조건과 표.
- `docs/data-governance.md`: 출처·라이선스·분할.
- `docs/portfolio-summary.md`: 채용 담당자용 1페이지 요약.
- `tests/`: 데이터 로더, 클래스 매핑, 단일 추론 형식, UI 입력 검증.
