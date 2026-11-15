"""Reliability and selective-prediction metrics.

This module is deliberately independent of PyTorch.  It consumes already
computed class probabilities so the same definitions can be tested with tiny
deterministic fixtures and reused by evaluation scripts.

The metrics here describe a *fixed* evaluation set.  Bootstrap intervals
resample that set only; they do not represent variation across training seeds,
data splits, cameras, or production domains.
"""

from __future__ import annotations

import math
import operator
import random
from collections.abc import Callable, Sequence

DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_COVERAGES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)


def _as_int(value: object, name: str) -> int:
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _as_probability_rows(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[int],
    num_classes: int | None = None,
) -> tuple[list[list[float]], list[int]]:
    rows = [[float(value) for value in row] for row in probabilities]
    if not rows:
        raise ValueError("probabilities must not be empty")
    if len(rows) != len(targets):
        raise ValueError("probabilities and targets must have the same length")
    width = len(rows[0])
    if num_classes is not None and num_classes != width:
        raise ValueError("num_classes does not match the probability width")
    if width < 2:
        raise ValueError("at least two classes are required")
    labels = [_as_int(target, "target") for target in targets]
    for row, label in zip(rows, labels):
        if len(row) != width:
            raise ValueError("all probability rows must have the same width")
        if not 0 <= label < width:
            raise ValueError("target index is outside the probability width")
        if any(not math.isfinite(value) or value < 0.0 for value in row):
            raise ValueError("probabilities must be finite and non-negative")
        if abs(sum(row) - 1.0) > 1e-6:
            raise ValueError("each probability row must sum to 1.0")
    return rows, labels


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool | int],
    n_bins: int = 10,
) -> dict:
    """Return equal-width ECE, MCE, and the per-bin reliability table."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if len(confidences) != len(correct):
        raise ValueError("confidences and correctness flags must have the same length")
    if not confidences:
        raise ValueError("confidences must not be empty")
    pairs = []
    for confidence, hit in zip(confidences, correct):
        value = float(confidence)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be a finite value in [0, 1]")
        pairs.append((value, 1.0 if bool(hit) else 0.0))

    total = len(pairs)
    bins = []
    ece = 0.0
    mce = 0.0
    for index in range(n_bins):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        members = [
            (confidence, hit)
            for confidence, hit in pairs
            if lower <= confidence < upper
            or (index == n_bins - 1 and confidence == 1.0)
        ]
        count = len(members)
        if count:
            mean_confidence = _mean([confidence for confidence, _ in members])
            accuracy = _mean([hit for _, hit in members])
            gap = abs(accuracy - mean_confidence)
            ece += (count / total) * gap
            mce = max(mce, gap)
        else:
            mean_confidence = None
            accuracy = None
            gap = None
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "gap": gap,
            }
        )
    return {"ece": ece, "mce": mce, "n_bins": n_bins, "n_samples": total, "bins": bins}


def multiclass_brier_score(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[int],
    num_classes: int | None = None,
) -> float:
    """Mean squared error against one-hot targets (sum over classes, then mean)."""
    rows, labels = _as_probability_rows(probabilities, targets, num_classes)
    total = 0.0
    for row, label in zip(rows, labels):
        total += sum(
            (probability - (1.0 if index == label else 0.0)) ** 2
            for index, probability in enumerate(row)
        )
    return total / len(rows)


def negative_log_likelihood(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[int],
    num_classes: int | None = None,
    epsilon: float = 1e-12,
) -> float:
    """Mean negative log probability assigned to the true class."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    rows, labels = _as_probability_rows(probabilities, targets, num_classes)
    total = 0.0
    for row, label in zip(rows, labels):
        total -= math.log(max(epsilon, min(1.0, row[label])))
    return total / len(rows)


def top_k_accuracy(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[int],
    k: int,
    num_classes: int | None = None,
) -> float:
    """Fraction of samples whose true class is in the top-k ranked classes."""
    rows, labels = _as_probability_rows(probabilities, targets, num_classes)
    if k < 1 or k > len(rows[0]):
        raise ValueError("k must be between 1 and the number of classes")
    hits = 0
    for row, label in zip(rows, labels):
        ranked = sorted(range(len(row)), key=row.__getitem__, reverse=True)[:k]
        if label in ranked:
            hits += 1
    return hits / len(rows)


def macro_f1_score(
    targets: Sequence[int],
    predictions: Sequence[int],
    num_classes: int,
) -> float:
    """Macro F1 using a fixed class range and zero division set to zero."""
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1")
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length")
    if not targets:
        raise ValueError("targets must not be empty")
    labels = [_as_int(target, "target") for target in targets]
    predicted = [_as_int(prediction, "prediction") for prediction in predictions]
    for label in labels + predicted:
        if not 0 <= label < num_classes:
            raise ValueError("target or prediction is outside the class range")

    f1_scores = []
    for class_index in range(num_classes):
        true_positive = sum(
            1
            for label, prediction in zip(labels, predicted)
            if label == class_index and prediction == class_index
        )
        false_positive = sum(
            1
            for label, prediction in zip(labels, predicted)
            if label != class_index and prediction == class_index
        )
        false_negative = sum(
            1
            for label, prediction in zip(labels, predicted)
            if label == class_index and prediction != class_index
        )
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1_scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return sum(f1_scores) / num_classes


def bootstrap_confidence_interval(
    sample: Sequence,
    statistic: Callable[[Sequence], float],
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap interval for a statistic over one fixed sample."""
    values = list(sample)
    if not values:
        raise ValueError("sample must not be empty")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    estimate = float(statistic(values))
    rng = random.Random(seed)
    size = len(values)
    estimates = []
    for _ in range(n_boot):
        resample = [values[rng.randrange(size)] for _ in range(size)]
        estimates.append(float(statistic(resample)))
    estimates.sort()
    last = len(estimates) - 1
    lower_index = max(0, min(last, int(math.floor((alpha / 2.0) * last))))
    upper_index = max(0, min(last, int(math.ceil((1.0 - alpha / 2.0) * last))))
    return {
        "estimate": estimate,
        "lower": estimates[lower_index],
        "upper": estimates[upper_index],
        "alpha": alpha,
        "n_boot": n_boot,
        "seed": seed,
        "method": "percentile bootstrap over the fixed evaluation set",
    }


def bootstrap_macro_f1_interval(
    targets: Sequence[int],
    predictions: Sequence[int],
    num_classes: int,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap macro F1 by resampling paired target/prediction rows."""
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length")
    pairs = list(zip(targets, predictions))
    if not pairs:
        raise ValueError("targets must not be empty")

    def statistic(resampled: Sequence) -> float:
        resampled_targets = [pair[0] for pair in resampled]
        resampled_predictions = [pair[1] for pair in resampled]
        return macro_f1_score(resampled_targets, resampled_predictions, num_classes)

    return bootstrap_confidence_interval(pairs, statistic, n_boot=n_boot, seed=seed, alpha=alpha)


def risk_coverage_curve(
    confidences: Sequence[float],
    correct: Sequence[bool | int],
    coverages: Sequence[float] = DEFAULT_COVERAGES,
) -> list[dict]:
    """Selective accuracy when keeping exactly the most confident fraction.

    Ties in confidence are broken by original sample order.  At a requested
    coverage ``c`` the number kept is ``ceil(c * n)`` with a minimum of one, so
    the curve is always defined for ``0 < c <= 1``.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correctness flags must have the same length")
    if not confidences:
        raise ValueError("confidences must not be empty")
    values = [float(value) for value in confidences]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("confidence must be a finite value in [0, 1]")
    hits = [1.0 if bool(hit) else 0.0 for hit in correct]
    requested = sorted(set(float(value) for value in coverages))
    if any(not 0.0 < value <= 1.0 for value in requested):
        raise ValueError("coverages must be in (0, 1]")

    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    curve = []
    for coverage in requested:
        kept = min(len(order), max(1, int(math.ceil(coverage * len(order) - 1e-12))))
        selected = order[:kept]
        selective_accuracy = _mean([hits[index] for index in selected])
        curve.append(
            {
                "coverage": coverage,
                "n_kept": kept,
                "threshold": values[selected[-1]],
                "selective_accuracy": selective_accuracy,
                "risk": 1.0 - selective_accuracy,
            }
        )
    return curve


def summarize_reliability(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[int],
    num_classes: int | None = None,
    n_bins: int = 10,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Return calibration, proper scoring, selective-prediction, and CI blocks."""
    rows, labels = _as_probability_rows(probabilities, targets, num_classes)
    class_count = len(rows[0])
    confidences = [max(row) for row in rows]
    predictions = [max(range(len(row)), key=row.__getitem__) for row in rows]
    correct = [1 if prediction == label else 0 for prediction, label in zip(predictions, labels)]
    calibration = expected_calibration_error(confidences, correct, n_bins=n_bins)
    return {
        "num_samples": len(rows),
        "num_classes": class_count,
        "accuracy": _mean(correct),
        "macro_f1": macro_f1_score(labels, predictions, class_count),
        "negative_log_likelihood": negative_log_likelihood(rows, labels, num_classes=class_count),
        "brier_score": multiclass_brier_score(rows, labels, num_classes=class_count),
        "top_3_accuracy": top_k_accuracy(rows, labels, min(3, class_count), num_classes=class_count),
        "calibration": calibration,
        "risk_coverage": risk_coverage_curve(confidences, correct, coverages=coverages),
        "bootstrap": {
            "accuracy": bootstrap_confidence_interval(
                correct, _mean, n_boot=n_boot, seed=seed, alpha=alpha
            ),
            "macro_f1": bootstrap_macro_f1_interval(
                labels,
                predictions,
                class_count,
                n_boot=n_boot,
                seed=seed,
                alpha=alpha,
            ),
        },
        "note": (
            "Bootstrap intervals resample this fixed evaluation set only; "
            "they do not estimate cross-split, cross-seed, or cross-domain variance."
        ),
    }
