import numpy as np
import pytest

from logitly import Metrics
from logitly.calibration import (
    accuracy,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)

CERTAIN_AND_RIGHT = np.array([[1.0, 0.0], [0.0, 1.0]])


def test_perfect_predictions_score_perfectly():
    labels = [0, 1]
    assert accuracy(CERTAIN_AND_RIGHT, labels) == 1.0
    assert brier_score(CERTAIN_AND_RIGHT, labels) == 0.0
    assert expected_calibration_error(CERTAIN_AND_RIGHT, labels) == 0.0
    assert negative_log_likelihood(CERTAIN_AND_RIGHT, labels) == pytest.approx(0.0)


def test_confident_and_wrong_is_maximally_miscalibrated():
    assert expected_calibration_error(CERTAIN_AND_RIGHT, [1, 0]) == pytest.approx(1.0)


def test_nll_matches_the_definition():
    probs = np.array([[0.7, 0.3], [0.2, 0.8]])
    expected = -(np.log(0.7) + np.log(0.8)) / 2
    assert negative_log_likelihood(probs, [0, 1]) == pytest.approx(expected)


def test_brier_matches_the_definition():
    probs = np.array([[0.6, 0.4]])
    assert brier_score(probs, [0]) == pytest.approx(0.4**2 + 0.4**2)


def test_a_coin_flip_on_a_coin_flip_is_calibrated():
    probs = np.tile([0.5, 0.5], (100, 1))
    labels = [i % 2 for i in range(100)]
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)


def test_metrics_bundle_reports_the_split_size():
    metrics = Metrics.compute(np.array([[0.9, 0.1], [0.4, 0.6]]), [0, 1])
    assert metrics.n == 2
    assert metrics.accuracy == 1.0
    assert metrics.mean_confidence == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("probs", "labels"),
    [
        (np.zeros(3), [0]),
        (np.zeros((2, 3)), [0]),
        (np.zeros((2, 3)), [0, 5]),
    ],
)
def test_malformed_input_is_rejected(probs, labels):
    with pytest.raises(ValueError):
        Metrics.compute(probs, labels)
