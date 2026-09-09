# 포트폴리오 1페이지 요약 (portfolio-summary)

## 산업 표면 결함 이미지 분류 — 공개 데이터 기반 PyTorch 재현 실험

**한 줄 소개**: 강재 공개 벤치마크(NEU-CLS, 1,800장·6클래스)에서 데이터 누수 점검 → 재현 가능한 분류 실험 → 합성 훼손 상황의 과신 실패 분석 → 업로드 데모까지 구현했습니다.

**주의**: 본 프로젝트는 반도체 데이터가 아니며, 실제 생산라인·검사 장비 성과가 아닙니다. 공개 데이터 기반 실험입니다.

**해석 경계**: 1.0000은 단일 split·seed(test 270장)의 포화 관측값이고, 합성 훼손 테스트는 외부 OOD·도메인 일반화 평가가 아닙니다. 품질 게이트는 실험적 보조 신호이며 운영 안전성을 보증하지 않습니다.

## 3분 검증 포인트

| 질문 | 답 (수치는 전부 실측) |
| --- | --- |
| 데이터를 어떻게 신뢰했나 | 1,800장 전부 디코딩·크기(200×200)·클래스당 300장 검증. **배포본 자체의 완전 중복 1쌍(Pa_101=Pa_105)을 SHA-256으로 발견** — 초기 분할에서 train/val 누수가 실제로 발생함을 확인하고 분할 전 중복 제외 정책으로 해결 |
| 성능은 | 단일 split·seed의 동일 test set(270장)에서 baseline·증강 모두 Accuracy 1.0000 / macro F1 1.0000. **"증강이 좋다"고 주장하지 않음** — 쉬운 공개 벤치마크의 포화 관측값이며, 관찰된 차이는 수렴 속도(best epoch 5 vs 12)뿐 |
| 재현되나 | seed=42 고정 재실행에서 epoch별 지표·test 지표 완전 일치(변동 필드 제외). 데이터 분할·학습·평가·벤치마크 전부 스크립트로 재실행 가능 |
| 속도는 | 단일 이미지(batch=1, 200회 측정)는 시스템 상태에 크게 좌우: 저부하 세션 CPU p50 10.4~11.7ms / GPU 1.65~1.87ms, 부하 경합 세션(게임 실행 중) CPU ~50.7ms / GPU ~6.2ms. 측정 조건을 함께 기록하는 것이 옳은 벤치마크 관행임을 실측으로 입증 |
| 합성 훼손은 | 같은 validation 이미지의 노이즈·블러·밝기 변형에서 정확도 붕괴와 과신을 측정. 이는 외부 OOD 평가가 아니며, 소프트맥스 신뢰도만으로 훼손 입력을 못 걸러내는 한계를 문서화 |
| 데모는 | Streamlit 업로드 → 형식·크기 검증 → 예측+신뢰도 → 0.60 미만이면 '재검토 필요'. 0.60과 품질 게이트는 같은 validation set에서 탐색한 보조 신호로, 운영 안전성 기준이 아님 |
| 품질 관리는 | 자동 테스트 72개 통과. 클래스 파싱(공식 약어 Cr/In/Pa/PS/RS/Sc 포함), NEU-CLS 입력 계약·`data/raw` 경계, CWD 독립 manifest 경로, 업로드 크기·픽셀 제한, 안전 checkpoint 로드·CPU smoke test, 최소 2개 고유 seed·동일 manifest SHA-256 기반 결과 집계, 산출물 seed 분리를 코드로 고정. 이 테스트들은 새 학습 수치를 만들지 않는다. |
| 안전장치는 | 신뢰도 임계값(0.60, 근거 측정) + 입력 품질 게이트(캘리브레이션 근거: clean 오경보 0.4%, 심한 노이즈 99.3% 차단) 이중화. 단, 블러·미세 변형은 두 장치 모두를 통과하는 한계를 측정으로 공개 |

## 구현 범위와 의도적 제외

- 구현: 데이터 검증·분할 파이프라인, ResNet18 전이학습, 증강 비교 실험, 혼동행렬·클래스별 지표 산출, CPU/GPU 추론 벤치마크, 신뢰도 기반 판단 보조 UI
- 제외(의사결정 기록 있음): YOLO/세그멘테이션, 장비·PLC 연동, TensorRT/NPU 최적화, MLOps, 실시간 성능 주장

## 이 프로젝트가 보여주는 역량

1. **데이터 무결성 검증 습관**: 중복·누수를 "발견 → 위험 확인 → 정책 결정 → 기록"으로 처리
2. **공정한 실험 설계**: test set 비사용 원칙, 단일 변인(증강) 비교, seed 재현 검증
3. **정직한 보고**: 포화 성능을 과장하지 않고, 증강의 우위를 주장하지 않음, 한계를 문서화
4. **제품적 마무리**: 저신뢰도 보수 동작이 있는 사용자 데모와 근거 있는 임계값

## 실행

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest -q -p no:cacheprovider --basetemp="$env:TEMP\opencode\pytest-tmp"
python scripts/prepare_data.py --input <NEU-CLS 압축파일>
python -m defect_cls.train --experiment baseline --seed 42 --device cuda
python -m defect_cls.train --experiment augmented --seed 42 --device cuda
python -m defect_cls.evaluate --checkpoint data/artifacts/baseline/seed-42/checkpoint.pt
python -m defect_cls.evaluate --checkpoint data/artifacts/augmented/seed-42/checkpoint.pt
python -m defect_cls.benchmark --checkpoint data/artifacts/baseline/seed-42/checkpoint.pt --devices cpu,cuda
streamlit run app/demo.py
```
