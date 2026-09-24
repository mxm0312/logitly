"""Turning raw option logits into probabilities you can act on."""

from logitly.calibration.dataset import Example, normalize_dataset, stratified_split
from logitly.calibration.fit import FitResult, fit_calibration
from logitly.calibration.metrics import (
    Metrics,
    accuracy,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)
from logitly.calibration.report import CalibrationReport
from logitly.calibration.transform import Calibration, softmax

__all__ = [
    "Calibration",
    "CalibrationReport",
    "Example",
    "FitResult",
    "Metrics",
    "accuracy",
    "brier_score",
    "expected_calibration_error",
    "fit_calibration",
    "negative_log_likelihood",
    "normalize_dataset",
    "softmax",
    "stratified_split",
]
