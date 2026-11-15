# 신뢰도·선택적 예측 평가 (reliability evaluation)

> 이 문서는 **평가 도구의 정의와 사용 계약**이다. 이 변경에서는 새로운 학습·평가 수치를 만들지 않았다. 실제 수치는 체크포인트와 `manifest.csv`가 있는 환경에서 `defect_cls.evaluate`를 실행해 `test-metrics.json`의 `reliability` 블록으로만 기록한다.

## 1. 왜 필요한가

NEU-CLS 6클래스는 ImageNet 전이학습 ResNet18에 쉬워 단일 split·seed의 Accuracy와 macro F1이 포화(1.0000)됐다. 포화된 정확도만으로는 다음 질문에 답할 수 없다.

- 모델의 확신이 실제 정답률과 일치하는가? (calibration)
- 확률 자체가 얼마나 좋은가? (proper scoring rule)
- 확신이 낮은 입력을 보류하면 오류를 얼마나 줄일 수 있는가? (selective prediction)
- 표본 270장에서 본 지표가 우연 변동으로 얼마나 흔들릴 수 있는가? (bootstrap interval)

이 도구는 위 질문을 **같은 test set**에서 계산해, "1.0000"이라는 단일 숫자 대신 평가의 불확실성과 보류 가능성을 함께 보여 준다.

## 2. 입력과 출력

- 입력: 체크포인트, 학습 때와 동일한 `manifest.csv`(SHA-256 일치 필수), test split.
- 계산: `src/defect_cls/reliability.py` — PyTorch에 의존하지 않는 순수 Python 함수.
- 출력: `data/artifacts/<experiment>/seed-<seed>/test-metrics.json`의 `reliability` 객체.

```bash
python -m defect_cls.evaluate \
  --checkpoint data/artifacts/baseline/seed-42/checkpoint.pt \
  --device cuda \
  --calibration-bins 10 \
  --bootstrap-samples 2000
```

기존 `test-metrics.json` 키(accuracy, macro_f1, per_class, confusion_matrix 등)는 그대로 유지된다. `reliability` 키만 추가된다.

## 3. 지표 정의

| 키 | 정의 | 해석 주의 |
| --- | --- | --- |
| `calibration.ece` | confidence를 10개 동일 폭 구간으로 나눈 뒤, 구간별 `|accuracy − mean confidence|`를 표본 수로 가중 평균 | calibration set과 평가 set이 같으면 낙관적이다. 독립 holdout 보정이 아니다 |
| `calibration.mce` | 위 구간별 gap의 최댓값 | 표본이 적은 구간 하나에 좌우될 수 있다 |
| `calibration.bins` | 구간별 `count`, `mean_confidence`, `accuracy`, `gap` | 빈 구간은 `null`로 기록 |
| `brier_score` | `mean(sum((p − onehot)^2))` (다중분류, 클래스 축 합) | 확률의 정확성과 확신을 함께 벌점화 |
| `negative_log_likelihood` | 정답 클래스 확률의 `−log` 평균 | 틀린 확신에 큰 벌점 |
| `top_3_accuracy` | 정답이 상위 3개 안에 든 비율 | 6클래스 기준 보조 지표 |
| `risk_coverage` | confidence 내림차순으로 상위 `ceil(c·n)`개만 남겼을 때의 `selective_accuracy`와 `risk = 1 − accuracy` | 실제 운영 triage 성능이 아니다. 고정 test set에서의 선택적 예측 곡선이다 |
| `bootstrap.accuracy`, `bootstrap.macro_f1` | 고정 표본을 복원추출해 구한 percentile 95% 구간 | **split·seed·도메인 간 변동이 아니다.** 단일 test set의 표본 변동만 나타낸다 |

- `risk_coverage`의 coverage는 `(0, 1]`이며, 동률 confidence는 원래 행 순서로 결정한다.
- bootstrap은 `random.Random(seed)`로 결정론적이다. 기본 `n_boot=2000`, `seed=42`, `alpha=0.05`.
- macro F1은 고정 클래스 범위(6)에서 계산하며, 한 번도 등장하지 않은 클래스의 F1은 0으로 둔다.

## 4. 결과를 읽는 규칙

1. **Accuracy와 ECE를 함께 읽는다.** 정확도가 같아도 ECE가 크면 확신이 과하다.
2. **Brier/NLL은 확률의 품질**이다. argmax 정확도가 변하지 않아도 값이 달라질 수 있다.
3. **risk-coverage는 "보류했을 때"의 조건부 정확도**다. coverage 0.5의 selective accuracy가 전체 accuracy보다 높다는 사실만으로 운영 안전성이 증명되지 않는다.
4. **bootstrap 구간은 좁아도 일반화 구간이 아니다.** 270장 test set을 재표집할 뿐이며, 새로운 촬영 조건·장비·재질을 대표하지 않는다.
5. **임계값을 이 결과로 자동 변경하지 않는다.** 임계값 0.60은 별도 `confidence-scan-val.json` 근거를 가진 보수 설정이다.

## 5. 비목표

- 새로운 성능 우위 주장, 증강 우위 주장, 현장 검출 성능 주장.
- 외부 OOD·도메인 일반화 평가. 합성 훼손 스트레스 테스트(`ood-robustness.json`)와도 다르다.
- 운영 품질 보증 임계값 자동 선택.
- 객체 탐지·세그멘테이션·위치 추정.

## 6. 테스트

`tests/test_reliability.py`는 작은 결정론적 배열로 ECE·Brier·NLL·top-k·macro F1·bootstrap·risk-coverage의 알려진 값을 고정한다. 실데이터·체크포인트 없이 실행되며, 학습 수치를 만들지 않는다.
