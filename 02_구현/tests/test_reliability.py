from __future__ import annotations

import json
import math

import pytest

from defect_cls.reliability import (
    bootstrap_confidence_interval,
    bootstrap_macro_f1_interval,
    expected_calibration_error,
    macro_f1_score,
    multiclass_brier_score,
    negative_log_likelihood,
    risk_coverage_curve,
    summarize_reliability,
    top_k_accuracy,
)


def test_expected_calibration_error_known_bin() -> None:
    result = expected_calibration_error([0.9, 0.9], [True, False], n_bins=10)
    assert result["ece"] == pytest.approx(0.4)
    assert result["mce"] == pytest.approx(0.4)
    assert result["n_samples"] == 2
    occupied = [row for row in result["bins"] if row["count"]]
    assert len(occupied) == 1
    assert occupied[0]["count"] == 2
    assert occupied[0]["mean_confidence"] == pytest.approx(0.9)
    assert occupied[0]["accuracy"] == pytest.approx(0.5)


def test_expected_calibration_error_places_confidence_one_in_last_bin() -> None:
    result = expected_calibration_error([1.0], [True], n_bins=10)
    assert result["bins"][-1]["count"] == 1
    assert result["ece"] == pytest.approx(0.0)


def test_brier_score_known_values() -> None:
    assert multiclass_brier_score([[1.0, 0.0], [0.0, 1.0]], [0, 1]) == pytest.approx(0.0)
    assert multiclass_brier_score([[0.5, 0.5]], [0]) == pytest.approx(0.5)


def test_negative_log_likelihood_known_value() -> None:
    assert negative_log_likelihood([[0.5, 0.5]], [0]) == pytest.approx(math.log(2.0))


def test_top_k_accuracy_uses_ranked_classes() -> None:
    probabilities = [[0.7, 0.2, 0.1]]
    assert top_k_accuracy(probabilities, [1], 1) == 0.0
    assert top_k_accuracy(probabilities, [1], 2) == 1.0
    assert top_k_accuracy(probabilities, [2], 3) == 1.0


def test_macro_f1_matches_fixed_class_range_definition() -> None:
    targets = [0, 0, 1, 1]
    predictions = [0, 1, 1, 1]
    assert macro_f1_score(targets, predictions, 2) == pytest.approx((2 / 3 + 0.8) / 2)


def test_bootstrap_is_deterministic_and_brackets_the_estimate() -> None:
    sample = [0, 1, 0, 1, 1, 0]
    first = bootstrap_confidence_interval(sample, lambda values: sum(values) / len(values), n_boot=200, seed=7)
    second = bootstrap_confidence_interval(sample, lambda values: sum(values) / len(values), n_boot=200, seed=7)
    assert first == second
    assert first["estimate"] == pytest.approx(0.5)
    assert first["lower"] <= first["estimate"] <= first["upper"]
    assert first["n_boot"] == 200
    assert first["seed"] == 7


def test_bootstrap_macro_f1_resamples_paired_rows() -> None:
    targets = [0, 0, 1, 1, 2, 2]
    predictions = [0, 1, 1, 1, 2, 2]
    result = bootstrap_macro_f1_interval(targets, predictions, 3, n_boot=100, seed=11)
    assert result["estimate"] == pytest.approx(macro_f1_score(targets, predictions, 3))
    assert result["lower"] <= result["estimate"] <= result["upper"]


def test_risk_coverage_keeps_most_confident_rows() -> None:
    curve = risk_coverage_curve([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0], coverages=[0.5, 1.0])
    half = curve[0]
    full = curve[1]
    assert half["n_kept"] == 2
    assert half["threshold"] == pytest.approx(0.8)
    assert half["selective_accuracy"] == pytest.approx(0.5)
    assert full["n_kept"] == 4
    assert full["selective_accuracy"] == pytest.approx(0.5)
    assert full["risk"] == pytest.approx(0.5)


def test_summarize_reliability_contract() -> None:
    probabilities = [
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.6, 0.2, 0.2],
    ]
    targets = [0, 1, 2, 2]
    result = summarize_reliability(probabilities, targets, n_bins=5, n_boot=50, seed=3)
    assert result["num_samples"] == 4
    assert result["num_classes"] == 3
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["top_3_accuracy"] == pytest.approx(1.0)
    assert result["calibration"]["n_bins"] == 5
    assert result["risk_coverage"][-1]["coverage"] == 1.0
    assert result["bootstrap"]["accuracy"]["n_boot"] == 50
    assert result["bootstrap"]["macro_f1"]["n_boot"] == 50
    assert "fixed evaluation set" in result["note"]
    assert json.loads(json.dumps(result))["num_samples"] == 4


@pytest.mark.parametrize(
    "probabilities,targets",
    [
        ([], []),
        ([[0.5, 0.5]], []),
        ([[0.5, 0.5]], [2]),
        ([[0.2, 0.2]], [0]),
        ([[1.0, -0.1]], [0]),
    ],
)
def test_probability_contract_rejects_invalid_rows(probabilities, targets) -> None:
    with pytest.raises(ValueError):
        multiclass_brier_score(probabilities, targets)


def test_risk_coverage_rejects_invalid_coverage() -> None:
    with pytest.raises(ValueError):
        risk_coverage_curve([0.5], [1], coverages=[0.0])
    with pytest.raises(ValueError):
        risk_coverage_curve([0.5], [1], coverages=[1.5])
