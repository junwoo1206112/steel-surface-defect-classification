# 산업 표면 결함 이미지 분류

> 공개 강재 표면 결함 데이터셋 NEU-CLS를 이용한 **PyTorch 기반 재현 실험**입니다.  
> 실제 생산 라인·검사 장비 성능이나 반도체 데이터가 아닙니다.

## 프로젝트 한눈에 보기

공개 산업 이미지로 데이터 검증부터 사용자 데모까지의 분류 파이프라인을 구현했습니다.

```
데이터 검증 → 중복 제거 → 누수 방지 분할 → ResNet18 전이학습
→ baseline/증강 비교 → 평가·추론 벤치마크 → Streamlit 판단 보조 데모
```

| 항목 | 내용 |
|---|---|
| 데이터 | NEU-CLS 1,800장 검증 후 중복 1장 제외, **1,799장** 사용 |
| 분할 | Stratified 70/15/15, seed=42 |
| 모델 | ImageNet 사전학습 ResNet18 |
| 평가 | baseline·증강 모두 test Accuracy / macro F1 **1.0000** |
| 재현성 | seed=42 재실행에서 epoch별·test 지표 일치 |
| 테스트 | 합성 fixture와 CPU smoke test를 포함한 **83개 통과** |

> 1.0000은 단일 split·seed의 포화 벤치마크 관측값입니다. 실제 현장 성능이나 도메인 일반화 성능을 의미하지 않습니다.

## 핵심 검증

- **데이터 무결성**: `Pa_101.bmp`와 `Pa_105.bmp`의 SHA-256 완전 중복을 발견해 분할 전 제외했습니다.
- **과신 분석**: 같은 validation 이미지에 노이즈·블러를 적용했을 때 정확도는 크게 낮아졌지만 신뢰도가 높게 유지되는 과신 문제를 측정했습니다.
- **판단 보조 UI**: Streamlit 데모에서 입력 형식·크기·픽셀을 검증하고, 저신뢰도 또는 품질 게이트 미달 입력에는 `재검토 필요`를 표시합니다.

## 기술 스택

`Python` · `PyTorch` · `Torchvision` · `Streamlit` · `scikit-learn` · `pytest`

## 실행

구현 코드와 상세 문서는 [02_구현](02_구현) 폴더에 있습니다.

```powershell
cd 02_구현
pip install -r requirements.txt
pip install -e .
python -m pytest -q -p no:cacheprovider --basetemp="$env:TEMP\opencode\pytest-tmp"

python scripts/prepare_data.py --input data/raw/<NEU-CLS.zip 또는 해제 폴더>
python -m defect_cls.train --experiment baseline --seed 42 --device cuda
streamlit run app/demo.py
```

## 문서

- [상세 README](02_구현/README.md)
- [실험 결과](02_구현/docs/experiment-results.md)
- [포트폴리오 1페이지 요약](02_구현/docs/portfolio-summary.md)
- [데이터 거버넌스](02_구현/docs/data-governance.md)

## 한계

- 공개 6클래스 벤치마크는 전이학습 모델에 상대적으로 쉬워 성능이 포화됐습니다.
- 실제 현장 데이터, 장비 연동, 실시간 처리 성능, 일반화 성능은 검증하지 않았습니다.
- 합성 훼손 실험은 외부 OOD 평가를 대체하지 않습니다.

## 데이터 출처

NEU Surface Defect Database (NEU-CLS), Northeastern University.  
K. Song and Y. Yan, *A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects*, Applied Surface Science, 2013.
