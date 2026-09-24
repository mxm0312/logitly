"""What every run reports, measured on the shared test set."""

from collections.abc import Sequence

import numpy as np

from logitly.calibration import Metrics
from logitly.config import Frozen

METRICS = ("accuracy", "nll", "ece", "brier", "mean_confidence", "changed")
LOWER_IS_BETTER = frozenset({"nll", "ece", "brier"})
LABELS = {
    "accuracy": "accuracy",
    "nll": "NLL",
    "ece": "ECE",
    "brier": "Brier",
    "mean_confidence": "mean confidence",
    "changed": "predictions changed",
}


class Bins(Frozen):
    """A reliability histogram, kept as sums so several seeds simply add up."""

    edges: list[float]
    count: list[int]
    confidence: list[float]
    correct: list[float]

    def __add__(self, other: "Bins") -> "Bins":
        if self.edges != other.edges:
            raise ValueError("cannot add histograms with different bin edges")
        return Bins(
            edges=self.edges,
            count=[a + b for a, b in zip(self.count, other.count, strict=True)],
            confidence=[a + b for a, b in zip(self.confidence, other.confidence, strict=True)],
            correct=[a + b for a, b in zip(self.correct, other.correct, strict=True)],
        )

    def means(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-bin (mean confidence, accuracy, share of examples); empty bins are dropped."""
        count = np.asarray(self.count, dtype=np.float64)
        keep = count > 0
        return (
            np.asarray(self.confidence)[keep] / count[keep],
            np.asarray(self.correct)[keep] / count[keep],
            count[keep] / count.sum(),
        )


def evaluate(
    probs: np.ndarray, labels: Sequence[int], *, baseline: np.ndarray, n_bins: int
) -> dict[str, float]:
    """The metric row for one set of predictions.

    `baseline` is the uncalibrated argmax, so `changed` counts the decisions
    calibration actually flipped.
    """
    scored = Metrics.compute(probs, labels, n_bins).model_dump(exclude={"n"})
    return scored | {"changed": float((probs.argmax(axis=1) != baseline).mean())}


def reliability(probs: np.ndarray, labels: Sequence[int], n_bins: int) -> Bins:
    """Bin by top-label confidence, exactly as the ECE in logitly does."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(labels)).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    return Bins(
        edges=edges.tolist(),
        count=np.bincount(index, minlength=n_bins).tolist(),
        confidence=np.bincount(index, conf, minlength=n_bins).tolist(),
        correct=np.bincount(index, correct, minlength=n_bins).tolist(),
    )
