from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from defect_cls import __version__
from defect_cls.data import CLASSES
from defect_cls.inference import (
    DEFAULT_THRESHOLD,
    load_image_tensor,
    predict,
    validate_upload,
)
from defect_cls.model import load_checkpoint
from defect_cls.quality import assess_quality

ARTIFACTS_DIR = Path("data/artifacts")
SPLITS_DIR = Path("data/processed")

st.set_page_config(page_title="산업 표면 결함 이미지 분류", page_icon="🔬", layout="centered")

st.title("산업 표면 결함 이미지 분류 — 판단 보조 데모")
st.caption(
    "공개 강재 표면 결함 데이터(NEU-CLS)로 학습된 실험 모델입니다. "
    "실제 검사 장비나 품질 확정 도구가 아니며, 결과는 판단 보조용 참고 자료입니다."
)


@st.cache_resource(show_spinner="모델을 불러오는 중...")
def load_model(checkpoint_path: str):
    model, checkpoint = load_checkpoint(checkpoint_path)
    return model, checkpoint


def available_experiments() -> list[Path]:
    if not ARTIFACTS_DIR.exists():
        return []
    return sorted(path for path in ARTIFACTS_DIR.glob("*/checkpoint.pt"))


checkpoints = available_experiments()
if not checkpoints:
    st.error(
        "학습된 모델이 없습니다. 먼저 데이터 준비와 학습을 완료하세요:\n\n"
        "1. `python scripts/prepare_data.py --input <NEU-CLS 압축파일>\n"
        "2. `python -m defect_cls.train --experiment baseline`\n"
        "3. `python -m defect_cls.evaluate --checkpoint data/artifacts/baseline/checkpoint.pt`"
    )
    st.stop()

experiment_names = [path.parent.name for path in checkpoints]
selected = st.sidebar.selectbox("실험 선택", experiment_names)
threshold = st.sidebar.slider(
    "신뢰도 임계값", 0.30, 0.95, DEFAULT_THRESHOLD, 0.05,
    help="이 값보다 신뢰도가 낮으면 재검토 안내를 표시합니다.",
)
st.sidebar.info(
    "참고: 심하게 훼손된 입력은 신뢰도가 높게 유지되는 과신 특성이 측정으로 확인돼 "
    "임계값만으로 걸러지지 않습니다. 이 데모는 판단 보조 참고용입니다."
)
checkpoint_path = next(path for path in checkpoints if path.parent.name == selected)

with st.sidebar:
    st.divider()
    st.subheader("실험 정보")
    st.markdown(
        f"- 앱 버전: `{__version__}`\n"
        f"- 실험: `{selected}`\n"
        f"- 클래스 수: {len(CLASSES)}\n"
        f"- 임계값: {threshold:.2f}"
    )

model, checkpoint = load_model(str(checkpoint_path))
image_mode = checkpoint.get("config", {}).get(
    "image_mode", "L" if checkpoint.get("in_channels") == 1 else "RGB"
)

uploaded = st.file_uploader(
    "이미지를 업로드하세요 (jpg, jpeg, png, bmp / 최대 10MB)",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded is not None:
    ok, message = validate_upload(uploaded.name, uploaded.size)
    if not ok:
        st.error(f"업로드 거부: {message}")
        st.stop()
    image_bytes = uploaded.getvalue()
    try:
        tensor = load_image_tensor(image_bytes, image_mode=image_mode)
    except ValueError as exc:
        st.error(f"처리 실패: {exc}")
        st.stop()

    quality = assess_quality(Image.open(io.BytesIO(image_bytes)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_gpu = model.to(device)
    result = predict(model_gpu, tensor.to(device), threshold=threshold)

    col_image, col_result = st.columns([1, 1.2])
    with col_image:
        st.image(image_bytes, caption=uploaded.name, use_container_width=True)
    with col_result:
        st.metric("예측 클래스", result["class_label"])
        st.metric("신뢰도", f"{result['confidence'] * 100:.1f}%")
        if not quality["passed"]:
            flagged = [name for name, flagged_value in quality["flags"].items() if flagged_value]
            st.warning(
                "판단 보조 결과 — 재검토 필요\n\n입력 이미지 품질 게이트 미달 ("
                + ", ".join(flagged)
                + "). 측정 예측은 확정하지 않습니다."
            )
        elif result["needs_review"]:
            st.warning("판단 보조 결과 — 재검토 필요\n\n신뢰도가 기준값보다 낮아 자동 확정하지 않습니다.")
        else:
            st.success("판단 보조 결과 — 기준 통과")

    st.subheader("클래스별 확률")
    probability_data = {
        name: result["probabilities"][name] for name in CLASSES
    }
    st.bar_chart(probability_data)

    st.divider()
    st.caption(
        f"모델: {selected} (epoch {checkpoint['epoch']}, seed {checkpoint['seed']}) · "
        f"torch {checkpoint['environment']['torch']} · "
        f"device {checkpoint['environment']['device']}"
    )
