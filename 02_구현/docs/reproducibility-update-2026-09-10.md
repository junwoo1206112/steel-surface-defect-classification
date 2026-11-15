# 재현성 재검증 기록 (2026-09-10)

## 수정 및 검증

- `python -m defect_cls.train --help`가 실행되도록 학습 CLI parser 초기화 위치를 수정했다.
- 평가는 `model.eval()` 모드에서 실행되도록 수정했다.
- 전체 pytest: 83 passed.

## 현재 코드로 재생성한 seed=42 비교

| 실험 | test Accuracy | test macro F1 | 선택 epoch |
| --- | ---: | ---: | ---: |
| baseline | 1.0000 | 1.0000 | 5 |
| augmented | 1.0000 | 1.0000 | 12 |

두 checkpoint와 test report는 모두 `data/processed/manifest.csv`의 SHA-256 `a9db30b8e3cc7590fb6cecdd66a4299019f57c179c40fde35fd190dddad097d3`를 기록한다.

## baseline 다중-seed test 결과

| seed | Accuracy | macro F1 |
| ---: | ---: | ---: |
| 7 | 0.9963 | 0.9963 |
| 42 | 1.0000 | 1.0000 |
| 123 | 0.9889 | 0.9889 |

## 2026-11-14 갱신

- 이 문서의 `83 passed`는 당시 기록이다. 같은 체크아웃의 2026-11-14 검증에서는 수정 전 90개 수집·89 통과·1 실패를 확인했고, OpenCV 마스크 결함 등을 수정한 뒤 110 passed, 인수인계 점검 테스트 8개를 더해 **118 passed**가 됐다.
- 새 학습·평가 수치는 만들지 않았으며, 위 표의 2026-09-10 실측값은 historical evidence로 보존한다. 상세: `01_협업_설계_페이즈/08_ABC_교차검증_및_구현.md`, `09_잔여작업_ABC_교차검증.md`.

