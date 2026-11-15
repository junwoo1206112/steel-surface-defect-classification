# OpenCV 기반 입력 품질 검증 및 전처리 파이프라인

이 문서는 Streamlit 판단 보조 데모에 추가한 OpenCV 경로의 계약과 인수 기준을 설명한다. OpenCV 결과는 PyTorch 분류 모델의 입력이나 확률을 바꾸지 않는 진단·시각화 계층이다.

## 입력 계약

- 허용 확장자: JPG/JPEG, PNG, BMP
- 빈 파일, 10MB 초과 파일, OpenCV 디코딩 실패(손상·비이미지), 20,000,000 픽셀 초과 이미지는 예측 전에 거부한다.
- OpenCV는 `cv2.imdecode`로 바이트를 다시 디코딩한다. 기존 PIL 기반 모델 입력 검증도 유지하므로 두 경로가 독립적으로 실패를 드러낸다.

## 진단과 시각화

| 항목 | OpenCV 방식 | UI 의미 |
| --- | --- | --- |
| 밝기 | grayscale 평균 | 극단적으로 어둡거나 밝은 입력 경고 |
| 대비 | grayscale 표준편차 | 저대비 입력 경고 |
| 선명도/블러 | Laplacian 분산 | 낮은 고주파 정보 경고 |
| 노이즈 | median blur 잔차의 MAD 기반 sigma 추정 | 높은 노이즈 경고 |
| 후보 강조 | CLAHE → Gaussian blur 대비 잔차 → Otsu threshold → morphology → 원본 오버레이 | 텍스처 변화의 전처리 기반 시각화 |

- 3×3 opening이 후보 픽셀의 10% 미만만 남기면 opening 전 마스크를 유지한다. 이 규칙은 1픽셀 얇은 스크래치·경계가 morphology에서 전부 사라지는 결함(2026-11-14 회귀 테스트에서 132px→0px 확인)을 막기 위한 것이다. 후보 강조의 목적이 "얇은 결함도 놓치지 않고 보여 주는 것"이므로, 노이즈 억제보다 구조 보존을 우선한다.

품질 임계값은 데모 휴리스틱이다. 공개 validation 데이터로 별도 보정하거나 현장 운영 성능을 측정한 값이 아니므로 품질 보증, 안전 기준, 결함 판정 기준으로 사용하지 않는다.

후보 강조는 **전처리 기반 후보 강조(모델의 위치 예측 아님)** 이다. 객체 탐지, 위치 추정, 결함 경계 분할, 면적 정확도 평가를 구현하거나 주장하지 않는다. 강재 표면 텍스처와 압축 노이즈도 후보로 보일 수 있다.

## 코드 경계

- `src/defect_cls/opencv_pipeline.py`: OpenCV 바이트 검증, 품질 지표, 시각화 순수 함수
- `app/demo.py`: 원본·CLAHE·후보 강조·분류 결과·품질 지표를 한 화면에 조합
- `src/defect_cls/inference.py`: 기존 PIL → 모델 tensor 경로를 유지
- `tests/test_opencv_pipeline.py`: 합성 이미지로 디코딩 거부, 지표 방향성, 시각화 출력 계약을 검증

## 검증 명령

```powershell
pip install -r requirements.txt
pip install -e .
python -m pytest -q -p no:cacheprovider --basetemp="$env:TEMP\opencode\pytest-tmp"
streamlit run app/demo.py
```

테스트는 실제 데이터, 학습, GPU 측정, 기존 성능 artifact를 바꾸지 않는다.
