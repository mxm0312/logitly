"""The before/after table printed at the end of calibration."""

from logitly.calibration.fit import FitResult
from logitly.calibration.metrics import Metrics
from logitly.config import Frozen

_ROWS = (
    ("accuracy", "accuracy", False),
    ("nll", "nll", True),
    ("ece", "ece", True),
    ("brier", "brier", True),
    ("mean confidence", "mean_confidence", None),
)
_WIDTH = 58


class CalibrationReport(Frozen):
    """What calibration changed, measured on data the fit never saw.

    Falls back to the train split when there is no test data, in which case
    held_out is False and the numbers are optimistic.
    """

    model: str
    question: str
    options: list[str]
    n_train: int
    n_test: int
    fit: FitResult
    before: Metrics
    after: Metrics
    held_out: bool

    @property
    def improved(self) -> bool:
        """Whether calibration lowered the negative log-likelihood."""
        return self.after.nll < self.before.nll

    def __str__(self) -> str:
        calibration = self.fit.calibration
        split = "held-out test split" if self.held_out else "TRAIN split (no test data)"
        bias = calibration.bias or []
        header = [
            "Calibration report",
            "=" * _WIDTH,
            f"model        : {self.model}",
            f"question     : {self.question}",
            f"options      : {', '.join(self.options)}",
            f"examples     : {self.n_train + self.n_test}"
            f" ({self.n_train} train / {self.n_test} test)",
            f"temperature  : {calibration.temperature:.4f}",
            "bias         : "
            + "  ".join(f"{o}={v:+.3f}" for o, v in zip(self.options, bias, strict=True)),
            f"optimiser    : {self.fit.iterations} steps"
            f"{'' if self.fit.converged else ' (hit max_iters)'}"
            f", nll {self.fit.loss_history[0]:.4f} -> {self.fit.loss_history[-1]:.4f} on train",
            "",
            f"measured on the {split}:",
            f"  {'metric':<16}{'before':>10}{'after':>10}{'change':>11}",
            "  " + "-" * (_WIDTH - 11),
        ]
        rows = []
        for title, attr, lower_is_better in _ROWS:
            before, after = getattr(self.before, attr), getattr(self.after, attr)
            delta = after - before
            mark = " "
            if lower_is_better is not None and delta:
                mark = "+" if (delta < 0) == lower_is_better else "-"
            rows.append(f"  {title:<16}{before:>10.4f}{after:>10.4f}{delta:>+10.4f} {mark}")
        return "\n".join([*header, *rows, "=" * _WIDTH])
