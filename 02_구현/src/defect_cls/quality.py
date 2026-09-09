from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

DEFAULT_QUALITY_THRESHOLDS = {
    "min_sharpness": 0.0,
    "max_noise_sigma": 12.0,
    "min_brightness": 40.0,
    "max_brightness": 245.0,
}
THRESHOLD_BASIS = (
    "2026-09-09 clean val 270장 내부 탐색(data/artifacts/baseline/quality-gate-calibration.json) 근거: "
    "noise p95=8.9, brightness p5/p95=66.9/218.7. max_noise_sigma=12.0은 clean 플래그 0.4%(1/270)에서 "
    "심한 노이즈 σ30을 99.3% 걸러낸다. min_sharpness=0.0은 비활성화다 — 라플라시안 분산은 이 텍스처 데이터에서 "
    "clean과 블러(r1/r3) 분포가 거의 겹쳐(플래그 비율 동일) 분리 지표가 아니었으므로 제외했다. "
    "임계값 선택과 측정이 같은 validation set 기반이므로 독립 운영 성능이 아니다. "
    "블러·미세 변형은 이 게이트로 걸러지지 않는 한계가 실험 리포트 11절에 기록돼 있다."
)


def sharpness_score(image: Image.Image) -> float:
    gray = image.convert("L")
    laplacian = gray.filter(
        ImageFilter.Kernel((3, 3), (0, 1, 0, 1, -4, 1, 0, 1, 0), scale=1, offset=0)
    )
    array = np.asarray(laplacian, dtype=np.float64)
    return float(array.var())


def estimate_noise_sigma(image: Image.Image) -> float:
    gray = image.convert("L")
    array = np.asarray(gray, dtype=np.float64)
    median = np.asarray(gray.filter(ImageFilter.MedianFilter(size=3)), dtype=np.float64)
    deviation = np.abs(array - median)
    return float(np.median(deviation) / 0.6745)


def mean_brightness(image: Image.Image) -> float:
    return float(np.asarray(image.convert("L"), dtype=np.float64).mean())


def assess_quality(image: Image.Image, thresholds: dict | None = None) -> dict:
    thresholds = thresholds or DEFAULT_QUALITY_THRESHOLDS
    sharpness = sharpness_score(image)
    noise_sigma = estimate_noise_sigma(image)
    brightness = mean_brightness(image)
    flags = {
        "too_blurry": sharpness < thresholds["min_sharpness"],
        "too_noisy": noise_sigma > thresholds["max_noise_sigma"],
        "extreme_brightness": brightness < thresholds["min_brightness"]
        or brightness > thresholds["max_brightness"],
    }
    return {
        "sharpness": round(sharpness, 4),
        "noise_sigma": round(noise_sigma, 4),
        "brightness": round(brightness, 4),
        "flags": flags,
        "passed": not any(flags.values()),
        "thresholds": dict(thresholds),
    }
