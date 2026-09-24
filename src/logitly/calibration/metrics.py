"""Scoring rules for judging whether calibration helped."""

from collections.abc import Sequence

import numpy as np

from logitly.config import Frozen

_EPS = 1e-12


def _check(probs: np.ndarray, labels: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if p.ndim != 2:
        raise ValueError(f"probs must be 2-D (n_examples, n_options), got shape {p.shape}")
    if y.shape != (p.shape[0],):
        raise ValueError(f"labels must have shape ({p.shape[0]},), got {y.shape}")
    if y.size and (y.min() < 0 or y.max() >= p.shape[1]):
        raise ValueError("labels contain an index outside the option range")
    return p, y


def negative_log_likelihood(probs: np.ndarray, labels: Sequence[int]) -> float:
    """Mean -log p(correct): the objective calibration minimises."""
    p, y = _check(probs, labels)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], _EPS, None)).mean())


def accuracy(probs: np.ndarray, labels: Sequence[int]) -> float:
    p, y = _check(probs, labels)
    return float((p.argmax(axis=1) == y).mean())


def brier_score(probs: np.ndarray, labels: Sequence[int]) -> float:
    """Mean squared error against the one-hot truth."""
    p, y = _check(probs, labels)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def expected_calibration_error(probs: np.ndarray, labels: Sequence[int], n_bins: int = 10) -> float:
    """Top-label ECE: how far confidence drifts from accuracy inside each bin."""
    p, y = _check(probs, labels)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    error = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(error)


class Metrics(Frozen):
    """Accuracy plus three proper scoring rules, on one split."""

    n: int
    accuracy: float
    nll: float
    ece: float
    brier: float
    mean_confidence: float

    @classmethod
    def compute(cls, probs: np.ndarray, labels: Sequence[int], n_bins: int = 10) -> "Metrics":
        p, y = _check(probs, labels)
        if not len(y):
            raise ValueError("cannot compute metrics on an empty split")
        return cls(
            n=len(y),
            accuracy=accuracy(p, y),
            nll=negative_log_likelihood(p, y),
            ece=expected_calibration_error(p, y, n_bins),
            brier=brier_score(p, y),
            mean_confidence=float(p.max(axis=1).mean()),
        )
